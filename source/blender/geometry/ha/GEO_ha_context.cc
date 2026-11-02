/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GEO_ha_context.hh"

#include <algorithm>
#include <array>
#include <mutex>
#include <string>

#include "BKE_global.hh"

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_base.hh"
#include "BLI_threads.h"

#include "GPU_capabilities.hh"
#include "GPU_compute.hh"
#include "GPU_context.hh"
#include "GPU_init_exit.hh"
#include "GPU_platform_backend_enum.h"
#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_storage_buffer.hh"

#include "DRW_engine.hh"

#include "GHOST_IContext.hh"

#include "WM_api.hh"

#include "gpu_shader_create_info.hh"

namespace blender::geometry::ha {

namespace {

constexpr int LOCAL_GROUP_SIZE = 64;

Map<std::string, gpu::Shader *> &shader_cache()
{
  static Map<std::string, gpu::Shader *> cache;
  return cache;
}

bool ha_debug_enabled()
{
  static const bool enabled = getenv("BLENDER_HA_DEBUG") != nullptr;
  return enabled;
}

void ha_debug(const char *message)
{
  if (ha_debug_enabled()) {
    /* stderr is unbuffered, so the trail survives a crash. */
    fprintf(stderr, "HA device: %s\n", message);
    fflush(stderr);
  }
}

/* -------------------------------------------------------------------- */
/* Dedicated compute context.
 *
 * A GL context can only be current on one thread at a time, so the whole GPU sequence of a zone is
 * serialized behind one mutex and uses one dedicated context (the draw manager's context belongs to
 * whichever thread is drawing). This mirrors what the render engine does in `RenderDisplay`: the
 * GHOST system context is created on the main thread and released right away so that it can be
 * activated on demand from any thread. */

std::mutex &ha_context_mutex()
{
  static std::mutex mutex;
  return mutex;
}

/**
 * The secondary context of this module. Created on the main thread by
 * #gpu::GPU_create_secondary_context, which also restores whatever context was active before (that
 * is what `WM_system_gpu_context_create` did *not* do: on Vulkan it left the main thread pointing at
 * our context and the next window redraw crashed in `VKContext::sync_backbuffer`).
 */
gpu::GPUSecondaryContextData &ha_secondary_context()
{
  static gpu::GPUSecondaryContextData data;
  return data;
}

GPUContext *ha_context()
{
  return ha_secondary_context().gpu_context;
}

/**
 * The context our shaders and buffers were created in. Every guard has to run in it: the `gpu::`
 * module stores a context's objects in a per-backend map keyed by the context, so using them from
 * another context is undefined behaviour.
 */
GPUContext *&ha_objects_context()
{
  static GPUContext *context = nullptr;
  return context;
}

/**
 * Drops the secondary context. Runs from a #GPU_exit hook, i.e. while the backend is still alive:
 * a context that outlives #GPU_exit leaves its allocations behind (Vulkan reports them as unfreed
 * and crashes afterwards).
 */
void ha_context_dispose(void * /*user_data*/)
{
  gpu::GPUSecondaryContextData &data = ha_secondary_context();
  if (data.gpu_context == nullptr) {
    return;
  }
  ha_context_mutex().lock();
  gpu::GPU_destroy_secondary_context(data);
  ha_bindings_invalidate();
  if (ha_objects_context() == data.gpu_context) {
    ha_objects_context() = nullptr;
  }
  ha_context_mutex().unlock();
  ha_debug("HA context destroyed");
}

bool ha_create_context()
{
  /* Creating a GHOST offscreen context is a main thread operation. */
  if (!BLI_thread_is_main() && getenv("BLENDER_HA_THREAD_CONTEXT") == nullptr) {
    ha_debug("cannot create the HA context off the main thread");
    return false;
  }
  if (GPU_backend_ghost_system_get() == nullptr) {
    ha_debug("no GHOST system, cannot create the HA context");
    return false;
  }
  gpu::GPUSecondaryContextData data = gpu::GPU_create_secondary_context();
  if (data.ghost_context == nullptr || data.gpu_context == nullptr) {
    ha_debug("GPU_create_secondary_context failed");
    return false;
  }
  ha_secondary_context() = data;
  GPU_exit_hook_register(ha_context_dispose, nullptr);
  ha_debug("HA context created");
  return true;
}

}  // namespace

bool ha_gpu_available()
{
  if (getenv("BLENDER_HA_NO_GPU") != nullptr) {
    return false;
  }
  return GPU_is_init() && GPU_backend_supported();
}

const char *ha_gpu_backend_name()
{
  if (!GPU_is_init()) {
    return "CPU (GPU not initialized)";
  }
  switch (GPU_backend_type_selection_get()) {
    case GPU_BACKEND_OPENGL:
      return "OpenGL";
    case GPU_BACKEND_VULKAN:
      return "Vulkan";
    case GPU_BACKEND_METAL:
      return "Metal";
    default:
      return "GPU";
  }
}

/* -------------------------------------------------------------------- */
/** \name Context
 * \{ */

namespace {

/**
 * State of the innermost #HaContextGuard of this thread. Guards nest (a solver step that owns the
 * context for its whole run does not pay for every dispatch), and the state is per thread because
 * a GL context is current on one thread at a time.
 */
struct HaGuardState {
  int depth = 0;
  bool valid = false;
  bool owns_mutex = false;
  bool enabled_by_us = false;
  /* What was current before this guard activated the dedicated context. It has to be put back:
   * the draw manager assumes its own context is active when it swaps the window's backbuffer, and
   * `VKContext::sync_backbuffer` (called from the swap-chain callback) reads the *active* context.
   * Leaving `nullptr` there crashed in `wm_draw_update` (measured). */
  GHOST_IContext *previous_ghost_context = nullptr;
  GPUContext *previous_gpu_context = nullptr;
};

thread_local HaGuardState g_guard_state;


}  // namespace

/**
 * The backend never initializes the GPU itself (that would pull in the window manager / GHOST).
 * In background mode the host has to call `gpu.init()` first.
 *
 * `gpu.init()` does *not* leave the context bound, so the work is wrapped in the draw manager's
 * context guard when there is no active context. Everything goes through the `gpu::` module, so
 * this works on OpenGL, Metal and Vulkan.
 */
HaContextGuard::HaContextGuard()
{
  if (!ha_gpu_available()) {
    return;
  }
  if (g_guard_state.depth > 0) {
    /* Nested: the outer guard already made the context current on this thread. */
    nested_ = true;
    g_guard_state.depth++;
    valid_ = g_guard_state.valid;
    return;
  }

  /* `BLENDER_HA_DRAW_CONTEXT=1` is kept as an A/B switch for the old behaviour. */
  const bool force_draw_context = getenv("BLENDER_HA_DRAW_CONTEXT") != nullptr;

  /* 1. Some context is already current on this thread: use it (this is the common case on the main
   *    thread, where the draw manager's context is active while the modifier is evaluated). */
  if (GPU_context_active_get() != nullptr) {
    g_guard_state = {1, true, false, false};
    valid_ = true;
    if (!this->accept_current_context()) {
      valid_ = false;
      g_guard_state = {};
    }
    return;
  }
  (void)force_draw_context;

  /* 2. Borrow the draw manager's context. `DRW_gpu_context_try_enable` also takes the draw ticket,
   *    so the main thread cannot draw (and swap the window's backbuffer) while we dispatch. The GPU
   *    module keeps one *globally* active context, so two contexts in use at the same time is what
   *    crashed in the GUI (`VKContext::sync_backbuffer` on a windowless context). */
  if (!G.background && DRW_gpu_context_try_enable()) {
    borrowed_draw_context_ = true;
    g_guard_state = {1, true, false, false};
    valid_ = GPU_context_active_get() != nullptr && this->accept_current_context();
    if (!valid_) {
      DRW_gpu_context_disable();
      borrowed_draw_context_ = false;
      g_guard_state = {};
    }
    return;
  }

  /* 3. Background mode (no window, nothing to draw): our own context, serialized behind the mutex.
   *    Only created here, so the GUI can never end up with objects in two different contexts. */
  ha_context_mutex().lock();
  owning_mutex_ = true;

  if (ha_context() == nullptr && !ha_create_context()) {
    ha_context_mutex().unlock();
    owning_mutex_ = false;
    return;
  }

  ha_debug("context activated (dedicated)");
  g_guard_state.previous_ghost_context = GHOST_IContext::getActiveDrawingContext();
  g_guard_state.previous_gpu_context = GPU_context_active_get();
  gpu::GPU_activate_secondary_context(ha_secondary_context());
  GPU_context_begin_frame(ha_context());
  enabled_by_us_ = true;
  valid_ = this->accept_current_context();
  if (!valid_) {
    GPU_context_end_frame(ha_context());
    gpu::GPU_deactivate_secondary_context(ha_secondary_context());
    if (g_guard_state.previous_ghost_context != nullptr) {
      g_guard_state.previous_ghost_context->activateDrawingContext();
    }
    if (g_guard_state.previous_gpu_context != nullptr) {
      GPU_context_active_set(g_guard_state.previous_gpu_context);
    }
    ha_context_mutex().unlock();
    owning_mutex_ = false;
    enabled_by_us_ = false;
    g_guard_state = {};
    return;
  }
  /* Bindings belong to the context, and another thread may have used the slots meanwhile. */
  ha_bindings_invalidate();
  g_guard_state = {1, true, owning_mutex_, enabled_by_us_, {}, {}};
}

bool HaContextGuard::accept_current_context()
{
  GPUContext *context = GPU_context_active_get();
  if (context == nullptr) {
    return false;
  }
  GPUContext *expected = ha_objects_context();
  if (expected == nullptr) {
    ha_objects_context() = context;
    return true;
  }
  if (expected != context) {
    /* Shaders and storage buffers belong to the context they were created in. Never mix them. */
    ha_debug("context mismatch, falling back to the CPU");
    return false;
  }
  return true;
}

HaContextGuard::~HaContextGuard()
{
  if (nested_) {
    g_guard_state.depth--;
    return;
  }
  if (g_guard_state.depth == 0) {
    return;
  }
  if (g_guard_state.enabled_by_us) {
    GPU_context_end_frame(ha_context());
    gpu::GPU_deactivate_secondary_context(ha_secondary_context());
    /* Give the thread back the contexts it had (the draw manager's one on the main thread). */
    if (g_guard_state.previous_ghost_context != nullptr) {
      g_guard_state.previous_ghost_context->activateDrawingContext();
    }
    if (g_guard_state.previous_gpu_context != nullptr) {
      GPU_context_active_set(g_guard_state.previous_gpu_context);
    }
  }
  if (borrowed_draw_context_) {
    DRW_gpu_context_disable();
    borrowed_draw_context_ = false;
  }
  if (g_guard_state.owns_mutex) {
    ha_context_mutex().unlock();
  }
  g_guard_state = {};
  enabled_by_us_ = false;
  owning_mutex_ = false;
  valid_ = false;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Backend scope
 * \{ */

namespace {

/* One slot per nested zone. Thread local, because the executor may evaluate different zones on
 * different threads. */
thread_local int ha_scope_depth = 0;
thread_local std::array<HaScopeBackend, 32> ha_scope_stack = {};

}  // namespace

HaScope::HaScope(const HaScopeBackend backend)
{
  if (ha_scope_depth < int(ha_scope_stack.size())) {
    ha_scope_stack[ha_scope_depth] = backend;
    ha_scope_depth++;
    pushed_ = true;
  }
}

HaScope::~HaScope()
{
  if (pushed_) {
    ha_scope_depth--;
  }
}

HaScopeBackend ha_scope_backend()
{
  if (ha_scope_depth == 0) {
    return HaScopeBackend::None;
  }
  return ha_scope_stack[ha_scope_depth - 1];
}

bool ha_scope_allows_gpu()
{
  const HaScopeBackend backend = ha_scope_backend();
  return backend == HaScopeBackend::Auto || backend == HaScopeBackend::Strict;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Buffers
 * \{ */

HaBuffer::~HaBuffer()
{
  this->free();
}bool HaBuffer::ensure(const int64_t elements)
{
  if (buf_ != nullptr && elements_ == elements) {
    return true;
  }
  this->free();
  if (elements <= 0) {
    return false;
  }
  buf_ = GPU_storagebuf_create_ex(size_t(elements) * sizeof(float),
                                  nullptr,
                                  GPU_USAGE_DYNAMIC,
                                  __func__);
  if (buf_ == nullptr) {
    return false;
  }
  elements_ = elements;
  return true;
}

void HaBuffer::upload(const Span<float> data)
{
  if (buf_ == nullptr) {
    return;
  }
  if (data.size() == elements_) {
    GPU_storagebuf_update(buf_, data.data());
    return;
  }
  /* The GPU storage buffer upload always transfers the full buffer, so pad (the extra elements are
   * processed by the padded invocations and the result is discarded). */
  Array<float> padded(int(elements_), 0.0f);
  const int64_t copy_size = std::min<int64_t>(data.size(), elements_);
  padded.as_mutable_span().take_front(int(copy_size)).copy_from(data.take_front(int(copy_size)));
  GPU_storagebuf_update(buf_, padded.data());
}

void HaBuffer::download(const MutableSpan<float> r_data) const
{
  if (buf_ == nullptr) {
    return;
  }
  /* `GPU_storagebuf_read` always reads the whole buffer, so read straight into the destination when
   * it is large enough instead of going through a temporary (that copy used to be 12 MB of pure
   * overhead per evaluation at a million elements). */
  if (r_data.size() >= elements_) {
    GPU_storagebuf_read(buf_, r_data.data());
    return;
  }
  Array<float> tmp(elements_);
  GPU_storagebuf_read(buf_, tmp.data());
  const int64_t copy_size = std::min<int64_t>(r_data.size(), elements_);
  r_data.take_front(int(copy_size))
      .copy_from(tmp.as_span().take_front(int(copy_size)));
}

void HaBuffer::free()
{
  if (buf_ != nullptr) {
    GPU_storagebuf_free(buf_);
    buf_ = nullptr;
    /* A new buffer may reuse the address of the freed one, which would make the binding cache skip
     * a bind that is actually needed. */
    ha_bindings_invalidate();
  }
  elements_ = 0;
}

HaBufferInt::~HaBufferInt()
{
  this->free();
}

bool HaBufferInt::ensure(const int64_t elements)
{
  if (buf_ != nullptr && elements_ == elements) {
    return true;
  }
  this->free();
  if (elements <= 0) {
    return false;
  }
  buf_ = GPU_storagebuf_create_ex(size_t(elements) * sizeof(int),
                                  nullptr,
                                  GPU_USAGE_DYNAMIC,
                                  __func__);
  if (buf_ == nullptr) {
    return false;
  }
  elements_ = elements;
  return true;
}

void HaBufferInt::upload(const Span<int> data)
{
  if (buf_ == nullptr) {
    return;
  }
  if (data.size() == elements_) {
    GPU_storagebuf_update(buf_, data.data());
    return;
  }
  const int padded_size = int(elements_);
  Array<int> padded(padded_size, 0);
  const int64_t copy_size = std::min<int64_t>(data.size(), elements_);
  padded.as_mutable_span().take_front(int(copy_size)).copy_from(data.take_front(int(copy_size)));
  GPU_storagebuf_update(buf_, padded.data());
}

void HaBufferInt::download(const MutableSpan<int> r_data) const
{
  if (buf_ == nullptr) {
    return;
  }
  const int64_t total = elements_;
  Array<int> tmp(total);
  GPU_storagebuf_read(buf_, tmp.data());
  const int64_t copy_size = std::min<int64_t>(r_data.size(), elements_);
  r_data.take_front(int(copy_size)).copy_from(tmp.as_span().take_front(int(copy_size)));
}

bool HaBufferInt::read_one(int &r_value) const
{
  if (buf_ == nullptr || elements_ < 1) {
    return false;
  }
  /* `GPU_storagebuf_read` always reads the whole buffer, so this is only used for tiny buffers. */
  GPU_storagebuf_read(buf_, &r_value);
  return true;
}

void HaBufferInt::clear_to_zero()
{
  if (buf_ != nullptr) {
    GPU_storagebuf_clear_to_zero(buf_);
  }
}

void HaBufferInt::free()
{
  if (buf_ != nullptr) {
    GPU_storagebuf_free(buf_);
    buf_ = nullptr;
    /* Same as #HaBuffer::free: the address may be reused by a new buffer. */
    ha_bindings_invalidate();
  }
  elements_ = 0;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shaders and dispatch
 * \{ */

namespace {

gpu::shader::Qualifier to_gpu_qualifier(const HaShaderQualifier qualifier)
{
  switch (qualifier) {
    case HaShaderQualifier::Read:
      return gpu::shader::Qualifier::read;
    case HaShaderQualifier::Write:
      return gpu::shader::Qualifier::write;
    case HaShaderQualifier::ReadWrite:
      return gpu::shader::Qualifier::read_write;
  }
  return gpu::shader::Qualifier::read;
}

}  // namespace

/**
 * Compiles \a source (a complete `main` plus the resource declaration block) with the given storage
 * buffers and caches it by \a cache_key.
 */
gpu::Shader *ha_shader_get_source(const StringRef cache_key,
                                 const StringRef source,
                                 const Span<const HaShaderResource> resources,
                                 const int local_group_size)
{
  /* Mixing the source into the key means an edited kernel can never hit a stale entry in the
   * in-memory cache or in the on-disk shader cache (the info name below is derived from it). */
  std::string key(cache_key);
  key += '#';
  key += std::to_string(std::hash<std::string>{}(std::string(source)));
  if (gpu::Shader **found = shader_cache().lookup_ptr(key)) {
    return *found;
  }

  using namespace gpu::shader;

  /* Version of the generated shader pipeline; bump whenever the resource layout or the kernel
   * templates change, so that the on-disk shader cache cannot return a stale binary. */
  constexpr int HA_SHADER_VERSION = 5;

  /* Unique per program: the resource declaration macros are derived from this name, and the shader
   * cache uses it as part of its key. */
  const std::string info_name = std::string("gpu_shader_ha_zone_") +
                                std::to_string(std::hash<std::string>{}(key)) + "_v" +
                                std::to_string(HA_SHADER_VERSION) + "_r" +
                                std::to_string(resources.size());

  /* Expands to the storage buffer declarations generated from the resources below. */
  const std::string full_source = std::string("#ifdef CREATE_INFO_RES_PASS_") + info_name + "\n" +
                                  "CREATE_INFO_RES_PASS_" + info_name + "\n" + "#endif\n" +
                                  std::string(source);

  ShaderCreateInfo info(info_name.c_str());
  info.local_group_size(local_group_size);
  info.builtins(BuiltinBits::GLOBAL_INVOCATION_ID);
  for (const HaShaderResource &resource : resources) {
    info.storage_buf(
        resource.slot, to_gpu_qualifier(resource.qualifier), resource.type, resource.name);
  }
  /* Has to point at the registered placeholder; the generated source below replaces it. */
  info.compute_source("gpu_shader_ha_zone.glsl");
  info.generated_sources.append({"gpu_shader_ha_zone.glsl", {}, full_source});

  if (ha_debug_enabled()) {
    /* Dump the generated shader to the temp directory so it can be inspected after a compile error
     * (the GPU module only reports a line/column). */
    const char *temp_dir = getenv("TMP");
    if (temp_dir == nullptr) {
      temp_dir = getenv("TEMP");
    }
    if (temp_dir == nullptr) {
      temp_dir = ".";
    }
    const std::string path = std::string(temp_dir) + "/ha_zone_shader_" +
                             std::to_string(std::hash<std::string>{}(key)) + ".glsl";
    if (FILE *file = fopen(path.c_str(), "w")) {
      fwrite(full_source.data(), 1, full_source.size(), file);
      fclose(file);
      fprintf(stderr, "HA device: shader written to %s\n", path.c_str());
      fflush(stderr);
    }
  }

  gpu::Shader *shader = GPU_shader_create_from_info(
      reinterpret_cast<GPUShaderCreateInfo *>(&info));
  if (shader == nullptr) {
    ha_debug("shader: creation failed");
    return nullptr;
  }
  shader_cache().add(key, shader);
  return shader;
}

namespace {

const std::array<HaShaderResource, 3> POSITION_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Write, "float", "position_out[]"},
    {2, HaShaderQualifier::Read, "float", "aggregate[]"},
}};

const std::array<HaShaderResource, 3> REDUCE_STAGE1_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "float", "params[]"},
    {2, HaShaderQualifier::Write, "float", "partial[]"},
}};

const std::array<HaShaderResource, 3> REDUCE_STAGE2_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "partial[]"},
    {1, HaShaderQualifier::Read, "float", "params[]"},
    {2, HaShaderQualifier::Write, "float", "aggregate[]"},
}};

Span<const HaShaderResource> as_span(const std::array<HaShaderResource, 3> &resources)
{
  return Span<const HaShaderResource>(resources.data(), int64_t(resources.size()));
}

/**
 * Helpers the generated expressions call. They mirror the BLI functions that the CPU node
 * implementations use (`safe_divide`, `safe_modf`, `safe_floored_modf`, `safe_powf`, `wrapf`), and
 * they exist because the guards differ per component for vectors: writing them out inline in every
 * expression would make the generated source explode and is easy to get wrong.
 *
 * Prepended to every kernel that evaluates generated expressions (the position kernel and the first
 * reduction pass). Keep it in sync with `GEO_ha_expr.cc`.
 */
const char *HA_GLSL_HELPERS =
    "float ha_safe_divide(float a, float b) { return b != 0.0 ? a / b : 0.0; }\n"
    "vec3 ha_safe_divide(vec3 a, vec3 b) {\n"
    "  return vec3(ha_safe_divide(a.x, b.x), ha_safe_divide(a.y, b.y), ha_safe_divide(a.z, b.z));\n"
    "}\n"
    "float ha_safe_mod(float a, float b) { return b != 0.0 ? a - b * trunc(a / b) : 0.0; }\n"
    "vec3 ha_safe_mod(vec3 a, vec3 b) {\n"
    "  return vec3(ha_safe_mod(a.x, b.x), ha_safe_mod(a.y, b.y), ha_safe_mod(a.z, b.z));\n"
    "}\n"
    "float ha_safe_floored_mod(float a, float b) { return b != 0.0 ? a - floor(a / b) * b : 0.0; }\n"
    "vec3 ha_safe_floored_mod(vec3 a, vec3 b) {\n"
    "  return vec3(ha_safe_floored_mod(a.x, b.x),\n"
    "              ha_safe_floored_mod(a.y, b.y),\n"
    "              ha_safe_floored_mod(a.z, b.z));\n"
    "}\n"
    "float ha_safe_pow(float b, float e) {\n"
    "  if (b >= 0.0) { return pow(b, e); }\n"
    "  if (e != trunc(e)) { return 0.0; }\n"
    "  return mod(e, 2.0) == 0.0 ? pow(-b, e) : -pow(-b, e);\n"
    "}\n"
    "vec3 ha_safe_pow(vec3 b, vec3 e) {\n"
    "  return vec3(ha_safe_pow(b.x, e.x), ha_safe_pow(b.y, e.y), ha_safe_pow(b.z, e.z));\n"
    "}\n"
    "float ha_wrap(float v, float mx, float mn) {\n"
    "  float range = mx - mn;\n"
    "  return range != 0.0 ? v - range * floor((v - mn) / range) : mn;\n"
    "}\n"
    "vec3 ha_wrap(vec3 v, vec3 mx, vec3 mn) {\n"
    "  return vec3(ha_wrap(v.x, mx.x, mn.x), ha_wrap(v.y, mx.y, mn.y), ha_wrap(v.z, mx.z, mn.z));\n"
    "}\n";

}  // namespace

gpu::Shader *ha_shader_get(const StringRef key,
                           const StringRef glsl_position,
                           const StringRef glsl_selection,
                           const Span<const HaShaderResource> extra_resources)
{
  const std::string source =
      std::string(HA_GLSL_HELPERS) +
      "void main()\n"
                  "{\n"
                  "  int index = int(gl_GlobalInvocationID.x);\n"
                  "  int base = index * 3;\n"
                  "  vec3 position = vec3(position_in[base], position_in[base + 1], "
                  "position_in[base + 2]);\n"
                  "  vec3 target = " +
      std::string(glsl_position) +
      ";\n"
      /* The implicit float -> bool conversion of the field system is `value > 0` (see
       * `float_to_bool` in `BKE_type_conversions`), so the selection test has to match it exactly:
       * `!= 0` would select the negative elements as well. */
      "  if (" +
      std::string(glsl_selection) +
      " > 0.0) {\n"
      "    position_out[base] = target.x;\n"
      "    position_out[base + 1] = target.y;\n"
      "    position_out[base + 2] = target.z;\n"
      "  }\n"
      "  else {\n"
      "    position_out[base] = position.x;\n"
      "    position_out[base + 1] = position.y;\n"
      "    position_out[base + 2] = position.z;\n"
      "  }\n"
      "}\n";

  Vector<HaShaderResource> resources;
  for (const HaShaderResource &resource : POSITION_RESOURCES) {
    resources.append(resource);
  }
  for (const HaShaderResource &resource : extra_resources) {
    resources.append(resource);
  }
  return ha_shader_get_source(
      key,
      source,
      Span<const HaShaderResource>(resources.data(), int64_t(resources.size())),
      LOCAL_GROUP_SIZE);
}

gpu::Shader *ha_shader_get_store(const StringRef key,
                                 const StringRef glsl_value,
                                 const int components,
                                 const Span<const HaShaderResource> extra_resources)
{
  const bool vector = components > 1;
  std::string body = std::string(HA_GLSL_HELPERS) +
                     "void main()\n"
                     "{\n"
                     "  int index = int(gl_GlobalInvocationID.x);\n"
                     "  int base = index * 3;\n"
                     "  vec3 position = vec3(position_in[base], position_in[base + 1], "
                     "position_in[base + 2]);\n"
                     "  ";
  if (vector) {
    body += "vec3 value = " + std::string(glsl_value) +
            ";\n"
            "  store_out[index * 3] = value.x;\n"
            "  store_out[index * 3 + 1] = value.y;\n"
            "  store_out[index * 3 + 2] = value.z;\n";
  }
  else {
    body += "store_out[index] = " + std::string(glsl_value) + ";\n";
  }
  body += "}\n";

  Vector<HaShaderResource> resources;
  resources.append({0, HaShaderQualifier::Read, "float", "position_in[]"});
  resources.append({1, HaShaderQualifier::Write, "float", "store_out[]"});
  resources.append({2, HaShaderQualifier::Read, "float", "aggregate[]"});
  for (const HaShaderResource &resource : extra_resources) {
    resources.append(resource);
  }
  return ha_shader_get_source(
      key,
      body,
      Span<const HaShaderResource>(resources.data(), int64_t(resources.size())),
      LOCAL_GROUP_SIZE);
}

gpu::Shader *ha_shader_get_reduce_stage1(const StringRef key,
                                         const StringRef glsl_value_expr,
                                         const StringRef combine,
                                         const StringRef identity)
{
  /* One invocation per block; the loop is serial inside the block so that the summation order is
   * fixed (deterministic and close to the CPU result). */
  const std::string source = std::string(HA_GLSL_HELPERS) + "void main()\n"
                                         "{\n"
                                         "  int block = int(gl_GlobalInvocationID.x);\n"
                                         "  int element_count = int(params[0]);\n"
                                         "  int block_count = int(params[1]);\n"
                                         "  if (block >= block_count) {\n"
                                         "    return;\n"
                                         "  }\n"
                                         "  int start = block * " +
                             std::to_string(HA_REDUCE_BLOCK_SIZE) +
                             ";\n"
                             "  int end = min(start + " +
                             std::to_string(HA_REDUCE_BLOCK_SIZE) +
                             ", element_count);\n"
                             "  float acc = " +
                             std::string(identity) +
                             ";\n"
                             "  for (int i = start; i < end; i++) {\n"
                             "    int index = i;\n"
                             "    int base = index * 3;\n"
                             "    vec3 position = vec3(position_in[base], position_in[base + 1], "
                             "position_in[base + 2]);\n"
                             "    float value = " +
                             std::string(glsl_value_expr) +
                             ";\n"
                             "    acc = " +
                             std::string(combine) +
                             ";\n"
                             "  }\n"
                             "  partial[block * 3] = acc;\n"
                             "}\n";

  return ha_shader_get_source(
      key, source, as_span(REDUCE_STAGE1_RESOURCES), LOCAL_GROUP_SIZE);
}

gpu::Shader *ha_shader_get_reduce_stage2(const StringRef key,
                                         const StringRef combine,
                                         const StringRef identity,
                                         const bool divide_by_count)
{
  std::string source = std::string("void main()\n"
                                   "{\n"
                                   "  if (int(gl_GlobalInvocationID.x) != 0) {\n"
                                   "    return;\n"
                                   "  }\n"
                                   "  int element_count = int(params[0]);\n"
                                   "  int block_count = int(params[1]);\n"
                                   "  int slot = int(params[2]);\n"
                                   "  float acc = ") +
                       std::string(identity) +
                       ";\n"
                       "  for (int b = 0; b < block_count; b++) {\n"
                       "    float value = partial[b * 3];\n"
                       "    acc = " +
                       std::string(combine) +
                       ";\n"
                       "  }\n";
  if (divide_by_count) {
    source += "  acc = acc / float(max(1, element_count));\n";
  }
  source += "  aggregate[slot * 3] = acc;\n}\n";

  return ha_shader_get_source(
      key, source, as_span(REDUCE_STAGE2_RESOURCES), LOCAL_GROUP_SIZE);
}

void ha_dispatch(gpu::Shader *shader, const int element_count)
{
  if (shader == nullptr || element_count <= 0) {
    return;
  }
  const uint groups = divide_ceil_u(uint(element_count), uint(LOCAL_GROUP_SIZE));
  if (int(groups) > GPU_max_work_group_count(0)) {
    /* Would need a 2D dispatch; not needed for the intended scale (4M elements). */
    return;
  }
  GPU_compute_dispatch(shader, groups, 1, 1);
}

namespace {

/** What #ha_bind_storage_buffers left bound, per slot. */
std::array<gpu::StorageBuf *, HA_MAX_BIND_SLOTS> g_bound_slots{};

}  // namespace

void ha_bind_storage_buffers(const Span<gpu::StorageBuf *> buffers, const int first_slot)
{
  for (const int i : buffers.index_range()) {
    const int slot = first_slot + i;
    if (slot >= HA_MAX_BIND_SLOTS) {
      break;
    }
    if (g_bound_slots[slot] == buffers[i]) {
      continue;
    }
    GPU_storagebuf_bind(buffers[i], slot);
    g_bound_slots[slot] = buffers[i];
  }
}

void ha_bindings_invalidate()
{
  g_bound_slots.fill(nullptr);
}

/** \} */

}  // namespace blender::geometry::ha