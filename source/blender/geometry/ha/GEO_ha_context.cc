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
  constexpr int HA_SHADER_VERSION = 7;

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
    "}\n"
    /* White Noise Texture: a port of `BLI_noise.cc` (Jenkins lookup3 over the float bit patterns).
     * Pure integer math, so it is bit identical to the CPU as long as the float <-> uint conversions
     * agree (they do: both round to nearest). */
    "uint ha_hash_rot(uint x, uint k) { return (x << k) | (x >> (32u - k)); }\n"
    "void ha_hash_mix(inout uint a, inout uint b, inout uint c) {\n"
    "  a -= c; a ^= ha_hash_rot(c, 4u); c += b;\n"
    "  b -= a; b ^= ha_hash_rot(a, 6u); a += c;\n"
    "  c -= b; c ^= ha_hash_rot(b, 8u); b += a;\n"
    "  a -= c; a ^= ha_hash_rot(c, 16u); c += b;\n"
    "  b -= a; b ^= ha_hash_rot(a, 19u); a += c;\n"
    "  c -= b; c ^= ha_hash_rot(b, 4u); b += a;\n"
    "}\n"
    "void ha_hash_final(inout uint a, inout uint b, inout uint c) {\n"
    "  c ^= b; c -= ha_hash_rot(b, 14u);\n"
    "  a ^= c; a -= ha_hash_rot(c, 11u);\n"
    "  b ^= a; b -= ha_hash_rot(a, 25u);\n"
    "  c ^= b; c -= ha_hash_rot(b, 16u);\n"
    "  a ^= c; a -= ha_hash_rot(c, 4u);\n"
    "  b ^= a; b -= ha_hash_rot(a, 14u);\n"
    "  c ^= b; c -= ha_hash_rot(b, 24u);\n"
    "}\n"
    "uint ha_hash1(uint kx) {\n"
    "  uint a = 0xdeadbeefu + (1u << 2u) + 13u;\n"
    "  uint b = a;\n"
    "  uint c = a;\n"
    "  a += kx;\n"
    "  ha_hash_final(a, b, c);\n"
    "  return c;\n"
    "}\n"
    "uint ha_hash2(uint kx, uint ky) {\n"
    "  uint a = 0xdeadbeefu + (2u << 2u) + 13u;\n"
    "  uint b = a;\n"
    "  uint c = a;\n"
    "  b += ky;\n"
    "  a += kx;\n"
    "  ha_hash_final(a, b, c);\n"
    "  return c;\n"
    "}\n"
    "uint ha_hash3(uint kx, uint ky, uint kz) {\n"
    "  uint a = 0xdeadbeefu + (3u << 2u) + 13u;\n"
    "  uint b = a;\n"
    "  uint c = a;\n"
    "  c += kz;\n"
    "  b += ky;\n"
    "  a += kx;\n"
    "  ha_hash_final(a, b, c);\n"
    "  return c;\n"
    "}\n"
    "uint ha_hash4(uint kx, uint ky, uint kz, uint kw) {\n"
    "  uint a = 0xdeadbeefu + (4u << 2u) + 13u;\n"
    "  uint b = a;\n"
    "  uint c = a;\n"
    "  a += kx;\n"
    "  b += ky;\n"
    "  c += kz;\n"
    "  ha_hash_mix(a, b, c);\n"
    "  a += kw;\n"
    "  ha_hash_final(a, b, c);\n"
    "  return c;\n"
    "}\n"
    /* `uint_to_float_01`: `float(k) / float(0xFFFFFFFFu)`, where the divisor rounds to 2^32. */
    "float ha_hash_to_float(uint k) { return float(k) / 4294967296.0; }\n"
    "float ha_white_noise_1d(float w) { return ha_hash_to_float(ha_hash1(floatBitsToUint(w))); }\n"
    "float ha_white_noise_2d(vec3 v) {\n"
    "  return ha_hash_to_float(ha_hash2(floatBitsToUint(v.x), floatBitsToUint(v.y)));\n"
    "}\n"
    "float ha_white_noise_3d(vec3 v) {\n"
    "  return ha_hash_to_float(\n"
    "      ha_hash3(floatBitsToUint(v.x), floatBitsToUint(v.y), floatBitsToUint(v.z)));\n"
    "}\n"
    "float ha_white_noise_4d(vec3 v, float w) {\n"
    "  return ha_hash_to_float(ha_hash4(floatBitsToUint(v.x),\n"
    "                                   floatBitsToUint(v.y),\n"
    "                                   floatBitsToUint(v.z),\n"
    "                                   floatBitsToUint(w)));\n"
    "}\n"
    /* Noise Texture: a port of `perlin_noise(float3)` / `perlin_signed` / the fractal types in
     * `BLI_noise.cc`. Same gradient hash (lookup3), same interpolation order, so the values match
     * the CPU. */
    "float ha_perlin_fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }\n"
    "float ha_negate_if(float v, uint c) { return c != 0u ? -v : v; }\n"
    "float ha_perlin_grad3(uint h, float x, float y, float z) {\n"
    "  h = h & 15u;\n"
    "  float u = h < 8u ? x : y;\n"
    "  float vt = (h == 12u || h == 14u) ? x : z;\n"
    "  float v = h < 4u ? y : vt;\n"
    "  return ha_negate_if(u, h & 1u) + ha_negate_if(v, h & 2u);\n"
    "}\n"
    "float ha_perlin_mix8(float v0, float v1, float v2, float v3, float v4, float v5, float v6,\n"
    "                     float v7, float x, float y, float z) {\n"
    "  float x1 = 1.0 - x;\n"
    "  float y1 = 1.0 - y;\n"
    "  float z1 = 1.0 - z;\n"
    "  return z1 * (y1 * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x)) +\n"
    "         z * (y1 * (v4 * x1 + v5 * x) + y * (v6 * x1 + v7 * x));\n"
    "}\n"
    "float ha_perlin_noise3(vec3 p) {\n"
    "  float fx = floor(p.x);\n"
    "  float fy = floor(p.y);\n"
    "  float fz = floor(p.z);\n"
    "  int X = int(fx);\n"
    "  int Y = int(fy);\n"
    "  int Z = int(fz);\n"
    "  float rx = p.x - fx;\n"
    "  float ry = p.y - fy;\n"
    "  float rz = p.z - fz;\n"
    "  float u = ha_perlin_fade(rx);\n"
    "  float v = ha_perlin_fade(ry);\n"
    "  float w = ha_perlin_fade(rz);\n"
    "  return ha_perlin_mix8(\n"
    "      ha_perlin_grad3(ha_hash3(uint(X), uint(Y), uint(Z)), rx, ry, rz),\n"
    "      ha_perlin_grad3(ha_hash3(uint(X + 1), uint(Y), uint(Z)), rx - 1.0, ry, rz),\n"
    "      ha_perlin_grad3(ha_hash3(uint(X), uint(Y + 1), uint(Z)), rx, ry - 1.0, rz),\n"
    "      ha_perlin_grad3(ha_hash3(uint(X + 1), uint(Y + 1), uint(Z)), rx - 1.0, ry - 1.0, rz),\n"
    "      ha_perlin_grad3(ha_hash3(uint(X), uint(Y), uint(Z + 1)), rx, ry, rz - 1.0),\n"
    "      ha_perlin_grad3(ha_hash3(uint(X + 1), uint(Y), uint(Z + 1)), rx - 1.0, ry, rz - 1.0),\n"
    "      ha_perlin_grad3(ha_hash3(uint(X), uint(Y + 1), uint(Z + 1)), rx, ry - 1.0, rz - 1.0),\n"
    "      ha_perlin_grad3(ha_hash3(uint(X + 1), uint(Y + 1), uint(Z + 1)),\n"
    "                      rx - 1.0, ry - 1.0, rz - 1.0),\n"
    "      u, v, w);\n"
    "}\n"
    "float ha_perlin_signed3(vec3 p) {\n"
    "  vec3 correction = 0.5 * vec3(abs(p.x) >= 1000000.0 ? 1.0 : 0.0,\n"
    "                              abs(p.y) >= 1000000.0 ? 1.0 : 0.0,\n"
    "                              abs(p.z) >= 1000000.0 ? 1.0 : 0.0);\n"
    "  p = ha_safe_mod(p, vec3(100000.0)) + correction;\n"
    "  return ha_perlin_noise3(p) * 0.9820;\n"
    "}\n"
    "vec3 ha_perlin_offset3(float seed) {\n"
    "  return vec3(100.0 + ha_hash_to_float(ha_hash2(floatBitsToUint(seed),\n"
    "                                                floatBitsToUint(0.0))) * 100.0,\n"
    "              100.0 + ha_hash_to_float(ha_hash2(floatBitsToUint(seed),\n"
    "                                                floatBitsToUint(1.0))) * 100.0,\n"
    "              100.0 + ha_hash_to_float(ha_hash2(floatBitsToUint(seed),\n"
    "                                                floatBitsToUint(2.0))) * 100.0);\n"
    "}\n"
    "vec3 ha_perlin_distortion3(vec3 p, float strength) {\n"
    "  return vec3(ha_perlin_signed3(p + ha_perlin_offset3(0.0)) * strength,\n"
    "              ha_perlin_signed3(p + ha_perlin_offset3(1.0)) * strength,\n"
    "              ha_perlin_signed3(p + ha_perlin_offset3(2.0)) * strength);\n"
    "}\n"
    "float ha_perlin_fbm3(vec3 p, float detail, float roughness, float lacunarity, bool normalize) {\n"
    "  float fscale = 1.0;\n"
    "  float amp = 1.0;\n"
    "  float maxamp = 0.0;\n"
    "  float sum = 0.0;\n"
    "  for (int i = 0; i <= int(detail); i++) {\n"
    "    float t = ha_perlin_signed3(fscale * p);\n"
    "    sum += t * amp;\n"
    "    maxamp += amp;\n"
    "    amp *= roughness;\n"
    "    fscale *= lacunarity;\n"
    "  }\n"
    "  float rmd = detail - floor(detail);\n"
    "  if (rmd != 0.0) {\n"
    "    float t = ha_perlin_signed3(fscale * p);\n"
    "    float sum2 = sum + t * amp;\n"
    "    if (normalize) {\n"
    "      return mix(0.5 * sum / maxamp + 0.5, 0.5 * sum2 / (maxamp + amp) + 0.5, rmd);\n"
    "    }\n"
    "    return mix(sum, sum2, rmd);\n"
    "  }\n"
    "  return normalize ? 0.5 * sum / maxamp + 0.5 : sum;\n"
    "}\n"
    "float ha_perlin_multi_fractal3(vec3 p, float detail, float roughness, float lacunarity) {\n"
    "  float value = 1.0;\n"
    "  float pwr = 1.0;\n"
    "  for (int i = 0; i <= int(detail); i++) {\n"
    "    value *= (pwr * ha_perlin_signed3(p) + 1.0);\n"
    "    pwr *= roughness;\n"
    "    p *= lacunarity;\n"
    "  }\n"
    "  float rmd = detail - floor(detail);\n"
    "  if (rmd != 0.0) {\n"
    "    value *= (rmd * pwr * ha_perlin_signed3(p) + 1.0);\n"
    "  }\n"
    "  return value;\n"
    "}\n"
    "float ha_perlin_hetero_terrain3(\n"
    "    vec3 p, float detail, float roughness, float lacunarity, float offset) {\n"
    "  float pwr = roughness;\n"
    "  float value = offset + ha_perlin_signed3(p);\n"
    "  p *= lacunarity;\n"
    "  for (int i = 1; i <= int(detail); i++) {\n"
    "    float increment = (ha_perlin_signed3(p) + offset) * pwr * value;\n"
    "    value += increment;\n"
    "    pwr *= roughness;\n"
    "    p *= lacunarity;\n"
    "  }\n"
    "  float rmd = detail - floor(detail);\n"
    "  if (rmd != 0.0) {\n"
    "    float increment = (ha_perlin_signed3(p) + offset) * pwr * value;\n"
    "    value += rmd * increment;\n"
    "  }\n"
    "  return value;\n"
    "}\n"
    "float ha_perlin_hybrid_multi_fractal3(vec3 p,\n"
    "                                      float detail,\n"
    "                                      float roughness,\n"
    "                                      float lacunarity,\n"
    "                                      float offset,\n"
    "                                      float gain) {\n"
    "  float pwr = 1.0;\n"
    "  float value = 0.0;\n"
    "  float weight = 1.0;\n"
    "  for (int i = 0; (weight > 0.001) && (i <= int(detail)); i++) {\n"
    "    weight = min(weight, 1.0);\n"
    "    float signal = (ha_perlin_signed3(p) + offset) * pwr;\n"
    "    pwr *= roughness;\n"
    "    value += weight * signal;\n"
    "    weight *= gain * signal;\n"
    "    p *= lacunarity;\n"
    "  }\n"
    "  float rmd = detail - floor(detail);\n"
    "  if ((rmd != 0.0) && (weight > 0.001)) {\n"
    "    weight = min(weight, 1.0);\n"
    "    float signal = (ha_perlin_signed3(p) + offset) * pwr;\n"
    "    value += rmd * weight * signal;\n"
    "  }\n"
    "  return value;\n"
    "}\n"
    "float ha_perlin_ridged_multi_fractal3(vec3 p,\n"
    "                                      float detail,\n"
    "                                      float roughness,\n"
    "                                      float lacunarity,\n"
    "                                      float offset,\n"
    "                                      float gain) {\n"
    "  float pwr = roughness;\n"
    "  float signal = offset - abs(ha_perlin_signed3(p));\n"
    "  signal *= signal;\n"
    "  float value = signal;\n"
    "  float weight = 1.0;\n"
    "  for (int i = 1; i <= int(detail); i++) {\n"
    "    p *= lacunarity;\n"
    "    weight = clamp(signal * gain, 0.0, 1.0);\n"
    "    signal = offset - abs(ha_perlin_signed3(p));\n"
    "    signal *= signal;\n"
    "    signal *= weight;\n"
    "    value += signal * pwr;\n"
    "    pwr *= roughness;\n"
    "  }\n"
    "  return value;\n"
    "}\n"
    "float ha_noise_select_3d(vec3 p,\n"
    "                         float detail,\n"
    "                         float roughness,\n"
    "                         float lacunarity,\n"
    "                         float offset,\n"
    "                         float gain,\n"
    "                         int type,\n"
    "                         bool normalize) {\n"
    "  if (type == 0) { return ha_perlin_multi_fractal3(p, detail, roughness, lacunarity); }\n"
    "  if (type == 1) { return ha_perlin_fbm3(p, detail, roughness, lacunarity, normalize); }\n"
    "  if (type == 2) {\n"
    "    return ha_perlin_hybrid_multi_fractal3(p, detail, roughness, lacunarity, offset, gain);\n"
    "  }\n"
    "  if (type == 3) {\n"
    "    return ha_perlin_ridged_multi_fractal3(p, detail, roughness, lacunarity, offset, gain);\n"
    "  }\n"
    "  if (type == 4) { return ha_perlin_hetero_terrain3(p, detail, roughness, lacunarity, offset); }\n"
    "  return 0.0;\n"
    "}\n"
    /* `Noise Texture > Factor`: the distortion is applied to the position once, then the fractal. */
    "float ha_noise_texture_3d(vec3 p,\n"
    "                          float detail,\n"
    "                          float roughness,\n"
    "                          float lacunarity,\n"
    "                          float offset,\n"
    "                          float gain,\n"
    "                          float distortion,\n"
    "                          int type,\n"
    "                          bool normalize) {\n"
    "  p += ha_perlin_distortion3(p, distortion);\n"
    "  return ha_noise_select_3d(p, detail, roughness, lacunarity, offset, gain, type, normalize);\n"
    "}\n"
    /* `Noise Texture > Color`: three fractals at the distorted position, the second and third offset
     * by `random_float3_offset(3)` / `(4)`. `Factor` is the red channel. */
    "vec3 ha_noise_texture_color_3d(vec3 p,\n"
    "                               float detail,\n"
    "                               float roughness,\n"
    "                               float lacunarity,\n"
    "                               float offset,\n"
    "                               float gain,\n"
    "                               float distortion,\n"
    "                               int type,\n"
    "                               bool normalize) {\n"
    "  p += ha_perlin_distortion3(p, distortion);\n"
    "  return vec3(ha_noise_select_3d(p, detail, roughness, lacunarity, offset, gain, type, normalize),\n"
    "              ha_noise_select_3d(p + ha_perlin_offset3(3.0),\n"
    "                                 detail,\n"
    "                                 roughness,\n"
    "                                 lacunarity,\n"
    "                                 offset,\n"
    "                                 gain,\n"
    "                                 type,\n"
    "                                 normalize),\n"
    "              ha_noise_select_3d(p + ha_perlin_offset3(4.0),\n"
    "                                 detail,\n"
    "                                 roughness,\n"
    "                                 lacunarity,\n"
    "                                 offset,\n"
    "                                 gain,\n"
    "                                 type,\n"
    "                                 normalize));\n"
    "}\n"
    /* Wave Texture: `node_shader_tex_wave.cc` (bands/rings + perlin_fbm distortion + three
     * profiles). The only noise it uses is `perlin_fbm` with lacunarity 2 and normalize. */
    "float ha_wave_texture(vec3 vector,\n"
    "                      float scale,\n"
    "                      float distortion,\n"
    "                      float detail,\n"
    "                      float dscale,\n"
    "                      float droughness,\n"
    "                      float phase,\n"
    "                      int wave_type,\n"
    "                      int bands_dir,\n"
    "                      int rings_dir,\n"
    "                      int profile) {\n"
    "  vec3 p = vector * scale;\n"
    "  /* Prevent precision issues on unit coordinates. */\n"
    "  p = (p + 0.000001) * 0.999999;\n"
    "  float n = 0.0;\n"
    "  if (wave_type == 0) {\n"
    "    if (bands_dir == 0) { n = p.x * 20.0; }\n"
    "    else if (bands_dir == 1) { n = p.y * 20.0; }\n"
    "    else if (bands_dir == 2) { n = p.z * 20.0; }\n"
    "    else { n = (p.x + p.y + p.z) * 10.0; }\n"
    "  }\n"
    "  else {\n"
    "    vec3 rp = p;\n"
    "    if (rings_dir == 0) { rp *= vec3(0.0, 1.0, 1.0); }\n"
    "    else if (rings_dir == 1) { rp *= vec3(1.0, 0.0, 1.0); }\n"
    "    else if (rings_dir == 2) { rp *= vec3(1.0, 1.0, 0.0); }\n"
    "    n = length(rp) * 20.0;\n"
    "  }\n"
    "  n += phase;\n"
    "  if (distortion != 0.0) {\n"
    "    n += distortion *\n"
    "         (ha_perlin_fbm3(p * dscale, detail, droughness, 2.0, true) * 2.0 - 1.0);\n"
    "  }\n"
    "  if (profile == 0) { return 0.5 + 0.5 * sin(n - 1.5707963267948966); }\n"
    "  n /= 6.283185307179586;\n"
    "  if (profile == 1) { return n - floor(n); }\n"
    "  return abs(n - floor(n + 0.5)) * 2.0;\n"
    "}\n"
    /* Voronoi Texture: the 3D path of `BLI_noise.cc`. Same PCG hash, same scan order (k, j, i from
     * -1 to 1), and `voronoi_distance_bound` keeps the squared distance for the Euclidean metric,
     * which is what makes the winning cell identical to the CPU. */
    "struct HaVoronoiOctave {\n"
    "  float distance;\n"
    "  vec3 position;\n"
    "};\n"
    "ivec3 ha_voronoi_pcg3d(ivec3 v) {\n"
    "  v = v * 1664525 + 1013904223;\n"
    "  v.x += v.y * v.z;\n"
    "  v.y += v.z * v.x;\n"
    "  v.z += v.x * v.y;\n"
    "  v = v ^ (v >> 16);\n"
    "  v.x += v.y * v.z;\n"
    "  v.y += v.z * v.x;\n"
    "  v.z += v.x * v.y;\n"
    "  return v;\n"
    "}\n"
    "float ha_voronoi_int_to_float_01(int k) {\n"
    "  return float(k & 0x7fffffff) * (1.0 / 2147483647.0);\n"
    "}\n"
    "vec3 ha_voronoi_hash3(ivec3 k) {\n"
    "  ivec3 h = ha_voronoi_pcg3d(k);\n"
    "  return vec3(ha_voronoi_int_to_float_01(h.x),\n"
    "              ha_voronoi_int_to_float_01(h.y),\n"
    "              ha_voronoi_int_to_float_01(h.z));\n"
    "}\n"
    "float ha_voronoi_distance(vec3 a, vec3 b, int metric, float exponent) {\n"
    "  if (metric == 0) { return length(a - b); }\n"
    "  if (metric == 1) {\n"
    "    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z);\n"
    "  }\n"
    "  if (metric == 2) {\n"
    "    return max(max(abs(a.x - b.x), abs(a.y - b.y)), abs(a.z - b.z));\n"
    "  }\n"
    "  return pow(pow(abs(a.x - b.x), exponent) + pow(abs(a.y - b.y), exponent) +\n"
    "                 pow(abs(a.z - b.z), exponent),\n"
    "             1.0 / exponent);\n"
    "}\n"
    "float ha_voronoi_distance_bound(vec3 a, vec3 b, int metric, float exponent) {\n"
    "  if (metric == 0) {\n"
    "    vec3 d = a - b;\n"
    "    return dot(d, d);\n"
    "  }\n"
    "  return ha_voronoi_distance(a, b, metric, exponent);\n"
    "}\n"
    "HaVoronoiOctave ha_voronoi_f1_3d(vec3 coord, float randomness, int metric, float exponent) {\n"
    "  vec3 cell_position_f = floor(coord);\n"
    "  vec3 local_position = coord - cell_position_f;\n"
    "  ivec3 cell_position = ivec3(cell_position_f);\n"
    "  float min_distance = 3.4028234663852886e+38;\n"
    "  vec3 target_position = vec3(0.0);\n"
    "  for (int k = -1; k <= 1; k++) {\n"
    "    for (int j = -1; j <= 1; j++) {\n"
    "      for (int i = -1; i <= 1; i++) {\n"
    "        ivec3 cell_offset = ivec3(i, j, k);\n"
    "        vec3 point_position =\n"
    "            vec3(cell_offset) + ha_voronoi_hash3(cell_position + cell_offset) * randomness;\n"
    "        float distance_to_point =\n"
    "            ha_voronoi_distance_bound(point_position, local_position, metric, exponent);\n"
    "        if (distance_to_point < min_distance) {\n"
    "          min_distance = distance_to_point;\n"
    "          target_position = point_position;\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave octave;\n"
    "  octave.distance = ha_voronoi_distance(target_position, local_position, metric, exponent);\n"
    "  octave.position = target_position + cell_position_f;\n"
    "  return octave;\n"
    "}\n"
    "HaVoronoiOctave ha_voronoi_smooth_f1_3d(\n"
    "    vec3 coord, float randomness, int metric, float exponent, float smoothness) {\n"
    "  vec3 cell_position_f = floor(coord);\n"
    "  vec3 local_position = coord - cell_position_f;\n"
    "  ivec3 cell_position = ivec3(cell_position_f);\n"
    "  float smooth_distance = 0.0;\n"
    "  vec3 smooth_position = vec3(0.0);\n"
    "  float h = -1.0;\n"
    "  for (int k = -2; k <= 2; k++) {\n"
    "    for (int j = -2; j <= 2; j++) {\n"
    "      for (int i = -2; i <= 2; i++) {\n"
    "        ivec3 cell_offset = ivec3(i, j, k);\n"
    "        vec3 point_position =\n"
    "            vec3(cell_offset) + ha_voronoi_hash3(cell_position + cell_offset) * randomness;\n"
    "        float distance_to_point =\n"
    "            ha_voronoi_distance(point_position, local_position, metric, exponent);\n"
    "        h = h == -1.0 ?\n"
    "                1.0 :\n"
    "                smoothstep(\n"
    "                    0.0, 1.0, 0.5 + 0.5 * (smooth_distance - distance_to_point) / smoothness);\n"
    "        float correction_factor = smoothness * h * (1.0 - h);\n"
    "        smooth_distance = mix(smooth_distance, distance_to_point, h) - correction_factor;\n"
    "        correction_factor /= 1.0 + 3.0 * smoothness;\n"
    "        smooth_position = mix(smooth_position, point_position, h) - correction_factor;\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave octave;\n"
    "  octave.distance = smooth_distance;\n"
    "  octave.position = cell_position_f + smooth_position;\n"
    "  return octave;\n"
    "}\n"
    "HaVoronoiOctave ha_voronoi_f2_3d(vec3 coord, float randomness, int metric, float exponent) {\n"
    "  vec3 cell_position_f = floor(coord);\n"
    "  vec3 local_position = coord - cell_position_f;\n"
    "  ivec3 cell_position = ivec3(cell_position_f);\n"
    "  float distance_f1 = 3.4028234663852886e+38;\n"
    "  float distance_f2 = 3.4028234663852886e+38;\n"
    "  vec3 position_f1 = vec3(0.0);\n"
    "  vec3 position_f2 = vec3(0.0);\n"
    "  for (int k = -1; k <= 1; k++) {\n"
    "    for (int j = -1; j <= 1; j++) {\n"
    "      for (int i = -1; i <= 1; i++) {\n"
    "        ivec3 cell_offset = ivec3(i, j, k);\n"
    "        vec3 point_position =\n"
    "            vec3(cell_offset) + ha_voronoi_hash3(cell_position + cell_offset) * randomness;\n"
    "        float distance_to_point =\n"
    "            ha_voronoi_distance(point_position, local_position, metric, exponent);\n"
    "        if (distance_to_point < distance_f1) {\n"
    "          distance_f2 = distance_f1;\n"
    "          distance_f1 = distance_to_point;\n"
    "          position_f2 = position_f1;\n"
    "          position_f1 = point_position;\n"
    "        }\n"
    "        else if (distance_to_point < distance_f2) {\n"
    "          distance_f2 = distance_to_point;\n"
    "          position_f2 = point_position;\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave octave;\n"
    "  octave.distance = distance_f2;\n"
    "  octave.position = position_f2 + cell_position_f;\n"
    "  return octave;\n"
    "}\n"
    "float ha_voronoi_distance_to_edge_3d(vec3 coord, float randomness) {\n"
    "  vec3 cell_position_f = floor(coord);\n"
    "  vec3 local_position = coord - cell_position_f;\n"
    "  ivec3 cell_position = ivec3(cell_position_f);\n"
    "  vec3 vector_to_closest = vec3(0.0);\n"
    "  float min_distance = 3.4028234663852886e+38;\n"
    "  for (int k = -1; k <= 1; k++) {\n"
    "    for (int j = -1; j <= 1; j++) {\n"
    "      for (int i = -1; i <= 1; i++) {\n"
    "        ivec3 cell_offset = ivec3(i, j, k);\n"
    "        vec3 vector_to_point = vec3(cell_offset) +\n"
    "                               ha_voronoi_hash3(cell_position + cell_offset) * randomness -\n"
    "                               local_position;\n"
    "        float distance_to_point = dot(vector_to_point, vector_to_point);\n"
    "        if (distance_to_point < min_distance) {\n"
    "          min_distance = distance_to_point;\n"
    "          vector_to_closest = vector_to_point;\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  min_distance = 3.4028234663852886e+38;\n"
    "  for (int k = -1; k <= 1; k++) {\n"
    "    for (int j = -1; j <= 1; j++) {\n"
    "      for (int i = -1; i <= 1; i++) {\n"
    "        ivec3 cell_offset = ivec3(i, j, k);\n"
    "        vec3 vector_to_point = vec3(cell_offset) +\n"
    "                               ha_voronoi_hash3(cell_position + cell_offset) * randomness -\n"
    "                               local_position;\n"
    "        vec3 perpendicular_to_edge = vector_to_point - vector_to_closest;\n"
    "        if (dot(perpendicular_to_edge, perpendicular_to_edge) > 0.0001) {\n"
    "          float distance_to_edge = dot((vector_to_closest + vector_to_point) / 2.0,\n"
    "                                       normalize(perpendicular_to_edge));\n"
    "          min_distance = min(min_distance, distance_to_edge);\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  return min_distance;\n"
    "}\n"
    "float ha_voronoi_n_sphere_radius_3d(vec3 coord, float randomness) {\n"
    "  vec3 cell_position_f = floor(coord);\n"
    "  vec3 local_position = coord - cell_position_f;\n"
    "  ivec3 cell_position = ivec3(cell_position_f);\n"
    "  vec3 closest_point = vec3(0.0);\n"
    "  ivec3 closest_point_offset = ivec3(0);\n"
    "  float min_distance_sq = 3.4028234663852886e+38;\n"
    "  for (int k = -1; k <= 1; k++) {\n"
    "    for (int j = -1; j <= 1; j++) {\n"
    "      for (int i = -1; i <= 1; i++) {\n"
    "        ivec3 cell_offset = ivec3(i, j, k);\n"
    "        vec3 point_position =\n"
    "            vec3(cell_offset) + ha_voronoi_hash3(cell_position + cell_offset) * randomness;\n"
    "        vec3 delta = point_position - local_position;\n"
    "        float distance_to_point_sq = dot(delta, delta);\n"
    "        if (distance_to_point_sq < min_distance_sq) {\n"
    "          min_distance_sq = distance_to_point_sq;\n"
    "          closest_point = point_position;\n"
    "          closest_point_offset = cell_offset;\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  min_distance_sq = 3.4028234663852886e+38;\n"
    "  vec3 closest_point_to_closest_point = vec3(0.0);\n"
    "  for (int k = -1; k <= 1; k++) {\n"
    "    for (int j = -1; j <= 1; j++) {\n"
    "      for (int i = -1; i <= 1; i++) {\n"
    "        if (i == 0 && j == 0 && k == 0) { continue; }\n"
    "        ivec3 cell_offset = ivec3(i, j, k) + closest_point_offset;\n"
    "        vec3 point_position =\n"
    "            vec3(cell_offset) + ha_voronoi_hash3(cell_position + cell_offset) * randomness;\n"
    "        vec3 delta = closest_point - point_position;\n"
    "        float distance_to_point_sq = dot(delta, delta);\n"
    "        if (distance_to_point_sq < min_distance_sq) {\n"
    "          min_distance_sq = distance_to_point_sq;\n"
    "          closest_point_to_closest_point = point_position;\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  return length(closest_point_to_closest_point - closest_point) / 2.0;\n"
    "}\n"
    "HaVoronoiOctave ha_voronoi_fractal_3d(vec3 coord,\n"
    "                                      float scale,\n"
    "                                      float detail,\n"
    "                                      float roughness,\n"
    "                                      float lacunarity,\n"
    "                                      float smoothness,\n"
    "                                      float exponent,\n"
    "                                      float randomness,\n"
    "                                      int feature,\n"
    "                                      int metric,\n"
    "                                      bool normalize) {\n"
    "  float max_distance = ha_voronoi_distance(\n"
    "                           vec3(0.0), vec3(0.5 + 0.5 * randomness), metric, exponent) *\n"
    "                       (feature == 1 ? 2.0 : 1.0);\n"
    "  float amplitude = 1.0;\n"
    "  float max_amplitude = 0.0;\n"
    "  float octave_scale = 1.0;\n"
    "  bool zero_input = detail == 0.0 || roughness == 0.0;\n"
    "  HaVoronoiOctave result;\n"
    "  result.distance = 0.0;\n"
    "  result.position = vec3(0.0);\n"
    "  int octave_count = int(ceil(detail));\n"
    "  for (int i = 0; i <= octave_count; i++) {\n"
    "    vec3 coord_scaled = coord * octave_scale;\n"
    "    HaVoronoiOctave octave;\n"
    "    if (feature == 1) {\n"
    "      octave = ha_voronoi_f2_3d(coord_scaled, randomness, metric, exponent);\n"
    "    }\n"
    "    else if (feature == 2 && smoothness != 0.0) {\n"
    "      octave = ha_voronoi_smooth_f1_3d(\n"
    "          coord_scaled, randomness, metric, exponent, smoothness);\n"
    "    }\n"
    "    else {\n"
    "      octave = ha_voronoi_f1_3d(coord_scaled, randomness, metric, exponent);\n"
    "    }\n"
    "    if (zero_input) {\n"
    "      max_amplitude = 1.0;\n"
    "      result = octave;\n"
    "      break;\n"
    "    }\n"
    "    if (float(i) <= detail) {\n"
    "      max_amplitude += amplitude;\n"
    "      result.distance += octave.distance * amplitude;\n"
    "      result.position = mix(result.position, octave.position / octave_scale, amplitude);\n"
    "      octave_scale *= lacunarity;\n"
    "      amplitude *= roughness;\n"
    "    }\n"
    "    else {\n"
    "      float remainder = detail - floor(detail);\n"
    "      if (remainder != 0.0) {\n"
    "        max_amplitude = mix(max_amplitude, max_amplitude + amplitude, remainder);\n"
    "        result.distance = mix(\n"
    "            result.distance, result.distance + octave.distance * amplitude, remainder);\n"
    "        result.position = mix(\n"
    "            result.position,\n"
    "            mix(result.position, octave.position / octave_scale, amplitude),\n"
    "            remainder);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  if (normalize) {\n"
    "    result.distance /= max_amplitude * max_distance;\n"
    "  }\n"
    "  result.position = (scale != 0.0) ? result.position / scale : vec3(0.0);\n"
    "  return result;\n"
    "}\n"
    "float ha_voronoi_fractal_distance_to_edge_3d(vec3 coord,\n"
    "                                             float detail,\n"
    "                                             float roughness,\n"
    "                                             float lacunarity,\n"
    "                                             float randomness,\n"
    "                                             bool normalize) {\n"
    "  float amplitude = 1.0;\n"
    "  float max_amplitude = 0.5 + 0.5 * randomness;\n"
    "  float octave_scale = 1.0;\n"
    "  float distance_value = 8.0;\n"
    "  bool zero_input = detail == 0.0 || roughness == 0.0;\n"
    "  int octave_count = int(ceil(detail));\n"
    "  for (int i = 0; i <= octave_count; i++) {\n"
    "    float octave_distance = ha_voronoi_distance_to_edge_3d(\n"
    "        coord * octave_scale, randomness);\n"
    "    if (zero_input) {\n"
    "      distance_value = octave_distance;\n"
    "      break;\n"
    "    }\n"
    "    if (float(i) <= detail) {\n"
    "      max_amplitude = mix(max_amplitude, (0.5 + 0.5 * randomness) / octave_scale, amplitude);\n"
    "      distance_value = mix(\n"
    "          distance_value, min(distance_value, octave_distance / octave_scale), amplitude);\n"
    "      octave_scale *= lacunarity;\n"
    "      amplitude *= roughness;\n"
    "    }\n"
    "    else {\n"
    "      float remainder = detail - floor(detail);\n"
    "      if (remainder != 0.0) {\n"
    "        float lerp_amplitude = mix(\n"
    "            max_amplitude, (0.5 + 0.5 * randomness) / octave_scale, amplitude);\n"
    "        max_amplitude = mix(max_amplitude, lerp_amplitude, remainder);\n"
    "        float lerp_distance = mix(\n"
    "            distance_value, min(distance_value, octave_distance / octave_scale), amplitude);\n"
    "        distance_value = mix(distance_value, min(distance_value, lerp_distance), remainder);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  if (normalize) {\n"
    "    distance_value /= max_amplitude;\n"
    "  }\n"
    "  return distance_value;\n"
    "}\n"
    "float ha_voronoi_texture_3d(vec3 vector,\n"
    "                            float scale,\n"
    "                            float detail,\n"
    "                            float roughness,\n"
    "                            float lacunarity,\n"
    "                            float smoothness,\n"
    "                            float exponent,\n"
    "                            float randomness,\n"
    "                            int feature,\n"
    "                            int metric,\n"
    "                            bool normalize) {\n"
    "  if (feature == 3) {\n"
    "    return ha_voronoi_fractal_distance_to_edge_3d(\n"
    "        vector * scale, detail, roughness, lacunarity, randomness, normalize);\n"
    "  }\n"
    "  HaVoronoiOctave result = ha_voronoi_fractal_3d(vector * scale,\n"
    "                                                 scale,\n"
    "                                                 detail,\n"
    "                                                 roughness,\n"
    "                                                 lacunarity,\n"
    "                                                 smoothness,\n"
    "                                                 exponent,\n"
    "                                                 randomness,\n"
    "                                                 feature,\n"
    "                                                 metric,\n"
    "                                                 normalize);\n"
    "  return result.distance;\n"
    "}\n"
    "vec3 ha_voronoi_position_3d(vec3 vector,\n"
    "                            float scale,\n"
    "                            float detail,\n"
    "                            float roughness,\n"
    "                            float lacunarity,\n"
    "                            float smoothness,\n"
    "                            float exponent,\n"
    "                            float randomness,\n"
    "                            int feature,\n"
    "                            int metric,\n"
    "                            bool do_normalize) {\n"
    "  HaVoronoiOctave result = ha_voronoi_fractal_3d(vector * scale,\n"
    "                                                 scale,\n"
    "                                                 detail,\n"
    "                                                 roughness,\n"
    "                                                 lacunarity,\n"
    "                                                 smoothness,\n"
    "                                                 exponent,\n"
    "                                                 randomness,\n"
    "                                                 feature,\n"
    "                                                 metric,\n"
    "                                                 do_normalize);\n"
    "  return result.position;\n"
    "}\n"
    "float ha_voronoi_radius_3d(vec3 vector, float scale, float randomness) {\n"
    "  return ha_voronoi_n_sphere_radius_3d(vector * scale, randomness);\n"
    "}\n"
    /* ---- Voronoi, 1D ---------------------------------------------------------------------- */
    "struct HaVoronoiOctave1 {\n"
    "  float distance;\n"
    "  float w;\n"
    "};\n"
    "HaVoronoiOctave1 ha_voronoi_f1_1d(float coord, float randomness) {\n"
    "  float cell_position = floor(coord);\n"
    "  float local_position = coord - cell_position;\n"
    "  float min_distance = 3.4028234663852886e+38;\n"
    "  float target_position = 0.0;\n"
    "  for (int i = -1; i <= 1; i++) {\n"
    "    float cell_offset = float(i);\n"
    "    float point_position =\n"
    "        cell_offset + ha_white_noise_1d(cell_position + cell_offset) * randomness;\n"
    "    float distance_to_point = abs(point_position - local_position);\n"
    "    if (distance_to_point < min_distance) {\n"
    "      min_distance = distance_to_point;\n"
    "      target_position = point_position;\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave1 octave;\n"
    "  octave.distance = min_distance;\n"
    "  octave.w = target_position + cell_position;\n"
    "  return octave;\n"
    "}\n"
    "HaVoronoiOctave1 ha_voronoi_smooth_f1_1d(float coord, float randomness, float smoothness) {\n"
    "  float cell_position = floor(coord);\n"
    "  float local_position = coord - cell_position;\n"
    "  float smooth_distance = 0.0;\n"
    "  float smooth_position = 0.0;\n"
    "  float h = -1.0;\n"
    "  for (int i = -2; i <= 2; i++) {\n"
    "    float cell_offset = float(i);\n"
    "    float point_position =\n"
    "        cell_offset + ha_white_noise_1d(cell_position + cell_offset) * randomness;\n"
    "    float distance_to_point = abs(point_position - local_position);\n"
    "    h = h == -1.0 ?\n"
    "            1.0 :\n"
    "            smoothstep(0.0, 1.0, 0.5 + 0.5 * (smooth_distance - distance_to_point) / smoothness);\n"
    "    float correction_factor = smoothness * h * (1.0 - h);\n"
    "    smooth_distance = mix(smooth_distance, distance_to_point, h) - correction_factor;\n"
    "    correction_factor /= 1.0 + 3.0 * smoothness;\n"
    "    smooth_position = mix(smooth_position, point_position, h) - correction_factor;\n"
    "  }\n"
    "  HaVoronoiOctave1 octave;\n"
    "  octave.distance = smooth_distance;\n"
    "  octave.w = cell_position + smooth_position;\n"
    "  return octave;\n"
    "}\n"
    "HaVoronoiOctave1 ha_voronoi_f2_1d(float coord, float randomness) {\n"
    "  float cell_position = floor(coord);\n"
    "  float local_position = coord - cell_position;\n"
    "  float distance_f1 = 3.4028234663852886e+38;\n"
    "  float distance_f2 = 3.4028234663852886e+38;\n"
    "  float position_f1 = 0.0;\n"
    "  float position_f2 = 0.0;\n"
    "  for (int i = -1; i <= 1; i++) {\n"
    "    float cell_offset = float(i);\n"
    "    float point_position =\n"
    "        cell_offset + ha_white_noise_1d(cell_position + cell_offset) * randomness;\n"
    "    float distance_to_point = abs(point_position - local_position);\n"
    "    if (distance_to_point < distance_f1) {\n"
    "      distance_f2 = distance_f1;\n"
    "      distance_f1 = distance_to_point;\n"
    "      position_f2 = position_f1;\n"
    "      position_f1 = point_position;\n"
    "    }\n"
    "    else if (distance_to_point < distance_f2) {\n"
    "      distance_f2 = distance_to_point;\n"
    "      position_f2 = point_position;\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave1 octave;\n"
    "  octave.distance = distance_f2;\n"
    "  octave.w = position_f2 + cell_position;\n"
    "  return octave;\n"
    "}\n"
    "float ha_voronoi_distance_to_edge_1d(float coord, float randomness) {\n"
    "  float cell_position = floor(coord);\n"
    "  float local_position = coord - cell_position;\n"
    "  float mid_point_position = ha_white_noise_1d(cell_position) * randomness;\n"
    "  float left_point_position =\n"
    "      -1.0 + ha_white_noise_1d(cell_position - 1.0) * randomness;\n"
    "  float right_point_position =\n"
    "      1.0 + ha_white_noise_1d(cell_position + 1.0) * randomness;\n"
    "  float distance_to_mid_left = abs((mid_point_position + left_point_position) / 2.0 -\n"
    "                                   local_position);\n"
    "  float distance_to_mid_right = abs((mid_point_position + right_point_position) / 2.0 -\n"
    "                                    local_position);\n"
    "  return min(distance_to_mid_left, distance_to_mid_right);\n"
    "}\n"
    "float ha_voronoi_n_sphere_radius_1d(float coord, float randomness) {\n"
    "  float cell_position = floor(coord);\n"
    "  float local_position = coord - cell_position;\n"
    "  float closest_point = 0.0;\n"
    "  float closest_point_offset = 0.0;\n"
    "  float min_distance = 3.4028234663852886e+38;\n"
    "  for (int i = -1; i <= 1; i++) {\n"
    "    float cell_offset = float(i);\n"
    "    float point_position =\n"
    "        cell_offset + ha_white_noise_1d(cell_position + cell_offset) * randomness;\n"
    "    float distance_to_point = abs(point_position - local_position);\n"
    "    if (distance_to_point < min_distance) {\n"
    "      min_distance = distance_to_point;\n"
    "      closest_point = point_position;\n"
    "      closest_point_offset = cell_offset;\n"
    "    }\n"
    "  }\n"
    "  min_distance = 3.4028234663852886e+38;\n"
    "  float closest_point_to_closest_point = 0.0;\n"
    "  for (int i = -1; i <= 1; i++) {\n"
    "    if (i == 0) { continue; }\n"
    "    float cell_offset = float(i) + closest_point_offset;\n"
    "    float point_position =\n"
    "        cell_offset + ha_white_noise_1d(cell_position + cell_offset) * randomness;\n"
    "    float distance_to_point = abs(closest_point - point_position);\n"
    "    if (distance_to_point < min_distance) {\n"
    "      min_distance = distance_to_point;\n"
    "      closest_point_to_closest_point = point_position;\n"
    "    }\n"
    "  }\n"
    "  return abs(closest_point_to_closest_point - closest_point) / 2.0;\n"
    "}\n"
    "HaVoronoiOctave1 ha_voronoi_fractal_1d(float coord,\n"
    "                                      float scale,\n"
    "                                      float detail,\n"
    "                                      float roughness,\n"
    "                                      float lacunarity,\n"
    "                                      float smoothness,\n"
    "                                      float exponent,\n"
    "                                      float randomness,\n"
    "                                      int feature,\n"
    "                                      int metric,\n"
    "                                      bool normalize) {\n"
    "  float max_distance = (0.5 + 0.5 * randomness) * (feature == 1 ? 2.0 : 1.0);\n"
    "  float amplitude = 1.0;\n"
    "  float max_amplitude = 0.0;\n"
    "  float octave_scale = 1.0;\n"
    "  bool zero_input = detail == 0.0 || roughness == 0.0;\n"
    "  HaVoronoiOctave1 result;\n"
    "  result.distance = 0.0;\n"
    "  result.w = 0.0;\n"
    "  int octave_count = int(ceil(detail));\n"
    "  for (int i = 0; i <= octave_count; i++) {\n"
    "    float coord_scaled = coord * octave_scale;\n"
    "    HaVoronoiOctave1 octave;\n"
    "    if (feature == 1) {\n"
    "      octave = ha_voronoi_f2_1d(coord_scaled, randomness);\n"
    "    }\n"
    "    else if (feature == 2 && smoothness != 0.0) {\n"
    "      octave = ha_voronoi_smooth_f1_1d(coord_scaled, randomness, smoothness);\n"
    "    }\n"
    "    else {\n"
    "      octave = ha_voronoi_f1_1d(coord_scaled, randomness);\n"
    "    }\n"
    "    if (zero_input) {\n"
    "      max_amplitude = 1.0;\n"
    "      result = octave;\n"
    "      break;\n"
    "    }\n"
    "    if (float(i) <= detail) {\n"
    "      max_amplitude += amplitude;\n"
    "      result.distance += octave.distance * amplitude;\n"
    "      result.w = mix(result.w, octave.w / octave_scale, amplitude);\n"
    "      octave_scale *= lacunarity;\n"
    "      amplitude *= roughness;\n"
    "    }\n"
    "    else {\n"
    "      float remainder = detail - floor(detail);\n"
    "      if (remainder != 0.0) {\n"
    "        max_amplitude = mix(max_amplitude, max_amplitude + amplitude, remainder);\n"
    "        result.distance = mix(\n"
    "            result.distance, result.distance + octave.distance * amplitude, remainder);\n"
    "        result.w = mix(result.w, mix(result.w, octave.w / octave_scale, amplitude), remainder);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  if (normalize) {\n"
    "    result.distance /= max_amplitude * max_distance;\n"
    "  }\n"
    "  result.w = (scale != 0.0) ? result.w / scale : 0.0;\n"
    "  return result;\n"
    "}\n"
    "float ha_voronoi_fractal_distance_to_edge_1d(float coord,\n"
    "                                             float detail,\n"
    "                                             float roughness,\n"
    "                                             float lacunarity,\n"
    "                                             float randomness,\n"
    "                                             bool normalize) {\n"
    "  float amplitude = 1.0;\n"
    "  float max_amplitude = 0.5 + 0.5 * randomness;\n"
    "  float octave_scale = 1.0;\n"
    "  float distance_value = 8.0;\n"
    "  bool zero_input = detail == 0.0 || roughness == 0.0;\n"
    "  int octave_count = int(ceil(detail));\n"
    "  for (int i = 0; i <= octave_count; i++) {\n"
    "    float octave_distance = ha_voronoi_distance_to_edge_1d(coord * octave_scale, randomness);\n"
    "    if (zero_input) {\n"
    "      distance_value = octave_distance;\n"
    "      break;\n"
    "    }\n"
    "    if (float(i) <= detail) {\n"
    "      max_amplitude = mix(max_amplitude, (0.5 + 0.5 * randomness) / octave_scale, amplitude);\n"
    "      distance_value = mix(\n"
    "          distance_value, min(distance_value, octave_distance / octave_scale), amplitude);\n"
    "      octave_scale *= lacunarity;\n"
    "      amplitude *= roughness;\n"
    "    }\n"
    "    else {\n"
    "      float remainder = detail - floor(detail);\n"
    "      if (remainder != 0.0) {\n"
    "        float lerp_amplitude = mix(\n"
    "            max_amplitude, (0.5 + 0.5 * randomness) / octave_scale, amplitude);\n"
    "        max_amplitude = mix(max_amplitude, lerp_amplitude, remainder);\n"
    "        float lerp_distance = mix(\n"
    "            distance_value, min(distance_value, octave_distance / octave_scale), amplitude);\n"
    "        distance_value = mix(distance_value, min(distance_value, lerp_distance), remainder);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  if (normalize) {\n"
    "    distance_value /= max_amplitude;\n"
    "  }\n"
    "  return distance_value;\n"
    "}\n"
    "float ha_voronoi_texture_1d(float coord,\n"
    "                            float scale,\n"
    "                            float detail,\n"
    "                            float roughness,\n"
    "                            float lacunarity,\n"
    "                            float smoothness,\n"
    "                            float exponent,\n"
    "                            float randomness,\n"
    "                            int feature,\n"
    "                            int metric,\n"
    "                            bool normalize) {\n"
    "  if (feature == 3) {\n"
    "    return ha_voronoi_fractal_distance_to_edge_1d(\n"
    "        coord * scale, detail, roughness, lacunarity, randomness, normalize);\n"
    "  }\n"
    "  HaVoronoiOctave1 result = ha_voronoi_fractal_1d(coord * scale,\n"
    "                                                 scale,\n"
    "                                                 detail,\n"
    "                                                 roughness,\n"
    "                                                 lacunarity,\n"
    "                                                 smoothness,\n"
    "                                                 exponent,\n"
    "                                                 randomness,\n"
    "                                                 feature,\n"
    "                                                 metric,\n"
    "                                                 normalize);\n"
    "  return result.distance;\n"
    "}\n"
    "float ha_voronoi_w_1d(float coord,\n"
    "                      float scale,\n"
    "                      float detail,\n"
    "                      float roughness,\n"
    "                      float lacunarity,\n"
    "                      float smoothness,\n"
    "                      float exponent,\n"
    "                      float randomness,\n"
    "                      int feature,\n"
    "                      int metric,\n"
    "                      bool normalize) {\n"
    "  HaVoronoiOctave1 result = ha_voronoi_fractal_1d(coord * scale,\n"
    "                                                 scale,\n"
    "                                                 detail,\n"
    "                                                 roughness,\n"
    "                                                 lacunarity,\n"
    "                                                 smoothness,\n"
    "                                                 exponent,\n"
    "                                                 randomness,\n"
    "                                                 feature,\n"
    "                                                 metric,\n"
    "                                                 normalize);\n"
    "  return result.w;\n"
    "}\n"
    "float ha_voronoi_radius_1d(float coord, float scale, float randomness) {\n"
    "  return ha_voronoi_n_sphere_radius_1d(coord * scale, randomness);\n"
    "}\n"
    /* ---- Voronoi, 2D ---------------------------------------------------------------------- */
    "struct HaVoronoiOctave2 {\n"
    "  float distance;\n"
    "  vec2 position;\n"
    "};\n"
    "ivec2 ha_voronoi_pcg2d(ivec2 v) {\n"
    "  v = v * 1664525 + 1013904223;\n"
    "  v.x += v.y * 1664525;\n"
    "  v.y += v.x * 1664525;\n"
    "  v = v ^ (v >> 16);\n"
    "  v.x += v.y * 1664525;\n"
    "  v.y += v.x * 1664525;\n"
    "  return v;\n"
    "}\n"
    "vec2 ha_voronoi_hash2(ivec2 k) {\n"
    "  ivec2 h = ha_voronoi_pcg2d(k);\n"
    "  return vec2(ha_voronoi_int_to_float_01(h.x), ha_voronoi_int_to_float_01(h.y));\n"
    "}\n"
    "float ha_voronoi_distance2(vec2 a, vec2 b, int metric, float exponent) {\n"
    "  if (metric == 0) { return length(a - b); }\n"
    "  if (metric == 1) { return abs(a.x - b.x) + abs(a.y - b.y); }\n"
    "  if (metric == 2) { return max(abs(a.x - b.x), abs(a.y - b.y)); }\n"
    "  return pow(pow(abs(a.x - b.x), exponent) + pow(abs(a.y - b.y), exponent),\n"
    "             1.0 / exponent);\n"
    "}\n"
    "float ha_voronoi_distance_bound2(vec2 a, vec2 b, int metric, float exponent) {\n"
    "  if (metric == 0) {\n"
    "    vec2 d = a - b;\n"
    "    return dot(d, d);\n"
    "  }\n"
    "  return ha_voronoi_distance2(a, b, metric, exponent);\n"
    "}\n"
    "HaVoronoiOctave2 ha_voronoi_f1_2d(vec2 coord, float randomness, int metric, float exponent) {\n"
    "  vec2 cell_position_f = floor(coord);\n"
    "  vec2 local_position = coord - cell_position_f;\n"
    "  ivec2 cell_position = ivec2(cell_position_f);\n"
    "  float min_distance = 3.4028234663852886e+38;\n"
    "  vec2 target_position = vec2(0.0);\n"
    "  for (int j = -1; j <= 1; j++) {\n"
    "    for (int i = -1; i <= 1; i++) {\n"
    "      ivec2 cell_offset = ivec2(i, j);\n"
    "      vec2 point_position =\n"
    "          vec2(cell_offset) + ha_voronoi_hash2(cell_position + cell_offset) * randomness;\n"
    "      float distance_to_point =\n"
    "          ha_voronoi_distance_bound2(point_position, local_position, metric, exponent);\n"
    "      if (distance_to_point < min_distance) {\n"
    "        min_distance = distance_to_point;\n"
    "        target_position = point_position;\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave2 octave;\n"
    "  octave.distance = ha_voronoi_distance2(target_position, local_position, metric, exponent);\n"
    "  octave.position = target_position + cell_position_f;\n"
    "  return octave;\n"
    "}\n"
    "HaVoronoiOctave2 ha_voronoi_smooth_f1_2d(\n"
    "    vec2 coord, float randomness, int metric, float exponent, float smoothness) {\n"
    "  vec2 cell_position_f = floor(coord);\n"
    "  vec2 local_position = coord - cell_position_f;\n"
    "  ivec2 cell_position = ivec2(cell_position_f);\n"
    "  float smooth_distance = 0.0;\n"
    "  vec2 smooth_position = vec2(0.0);\n"
    "  float h = -1.0;\n"
    "  for (int j = -2; j <= 2; j++) {\n"
    "    for (int i = -2; i <= 2; i++) {\n"
    "      ivec2 cell_offset = ivec2(i, j);\n"
    "      vec2 point_position =\n"
    "          vec2(cell_offset) + ha_voronoi_hash2(cell_position + cell_offset) * randomness;\n"
    "      float distance_to_point =\n"
    "          ha_voronoi_distance2(point_position, local_position, metric, exponent);\n"
    "      h = h == -1.0 ?\n"
    "              1.0 :\n"
    "              smoothstep(0.0, 1.0, 0.5 + 0.5 * (smooth_distance - distance_to_point) / smoothness);\n"
    "      float correction_factor = smoothness * h * (1.0 - h);\n"
    "      smooth_distance = mix(smooth_distance, distance_to_point, h) - correction_factor;\n"
    "      correction_factor /= 1.0 + 3.0 * smoothness;\n"
    "      smooth_position = mix(smooth_position, point_position, h) - correction_factor;\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave2 octave;\n"
    "  octave.distance = smooth_distance;\n"
    "  octave.position = cell_position_f + smooth_position;\n"
    "  return octave;\n"
    "}\n"
    "HaVoronoiOctave2 ha_voronoi_f2_2d(vec2 coord, float randomness, int metric, float exponent) {\n"
    "  vec2 cell_position_f = floor(coord);\n"
    "  vec2 local_position = coord - cell_position_f;\n"
    "  ivec2 cell_position = ivec2(cell_position_f);\n"
    "  float distance_f1 = 3.4028234663852886e+38;\n"
    "  float distance_f2 = 3.4028234663852886e+38;\n"
    "  vec2 position_f1 = vec2(0.0);\n"
    "  vec2 position_f2 = vec2(0.0);\n"
    "  for (int j = -1; j <= 1; j++) {\n"
    "    for (int i = -1; i <= 1; i++) {\n"
    "      ivec2 cell_offset = ivec2(i, j);\n"
    "      vec2 point_position =\n"
    "          vec2(cell_offset) + ha_voronoi_hash2(cell_position + cell_offset) * randomness;\n"
    "      float distance_to_point =\n"
    "          ha_voronoi_distance2(point_position, local_position, metric, exponent);\n"
    "      if (distance_to_point < distance_f1) {\n"
    "        distance_f2 = distance_f1;\n"
    "        distance_f1 = distance_to_point;\n"
    "        position_f2 = position_f1;\n"
    "        position_f1 = point_position;\n"
    "      }\n"
    "      else if (distance_to_point < distance_f2) {\n"
    "        distance_f2 = distance_to_point;\n"
    "        position_f2 = point_position;\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave2 octave;\n"
    "  octave.distance = distance_f2;\n"
    "  octave.position = position_f2 + cell_position_f;\n"
    "  return octave;\n"
    "}\n"
    "float ha_voronoi_distance_to_edge_2d(vec2 coord, float randomness) {\n"
    "  vec2 cell_position_f = floor(coord);\n"
    "  vec2 local_position = coord - cell_position_f;\n"
    "  ivec2 cell_position = ivec2(cell_position_f);\n"
    "  vec2 vector_to_closest = vec2(0.0);\n"
    "  float min_distance = 3.4028234663852886e+38;\n"
    "  for (int j = -1; j <= 1; j++) {\n"
    "    for (int i = -1; i <= 1; i++) {\n"
    "      ivec2 cell_offset = ivec2(i, j);\n"
    "      vec2 vector_to_point = vec2(cell_offset) +\n"
    "                             ha_voronoi_hash2(cell_position + cell_offset) * randomness -\n"
    "                             local_position;\n"
    "      float distance_to_point = dot(vector_to_point, vector_to_point);\n"
    "      if (distance_to_point < min_distance) {\n"
    "        min_distance = distance_to_point;\n"
    "        vector_to_closest = vector_to_point;\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  min_distance = 3.4028234663852886e+38;\n"
    "  for (int j = -1; j <= 1; j++) {\n"
    "    for (int i = -1; i <= 1; i++) {\n"
    "      ivec2 cell_offset = ivec2(i, j);\n"
    "      vec2 vector_to_point = vec2(cell_offset) +\n"
    "                             ha_voronoi_hash2(cell_position + cell_offset) * randomness -\n"
    "                             local_position;\n"
    "      vec2 perpendicular_to_edge = vector_to_point - vector_to_closest;\n"
    "      if (dot(perpendicular_to_edge, perpendicular_to_edge) > 0.0001) {\n"
    "        float distance_to_edge = dot((vector_to_closest + vector_to_point) / 2.0,\n"
    "                                     normalize(perpendicular_to_edge));\n"
    "        min_distance = min(min_distance, distance_to_edge);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  return min_distance;\n"
    "}\n"
    "float ha_voronoi_n_sphere_radius_2d(vec2 coord, float randomness) {\n"
    "  vec2 cell_position_f = floor(coord);\n"
    "  vec2 local_position = coord - cell_position_f;\n"
    "  ivec2 cell_position = ivec2(cell_position_f);\n"
    "  vec2 closest_point = vec2(0.0);\n"
    "  ivec2 closest_point_offset = ivec2(0);\n"
    "  float min_distance_sq = 3.4028234663852886e+38;\n"
    "  for (int j = -1; j <= 1; j++) {\n"
    "    for (int i = -1; i <= 1; i++) {\n"
    "      ivec2 cell_offset = ivec2(i, j);\n"
    "      vec2 point_position =\n"
    "          vec2(cell_offset) + ha_voronoi_hash2(cell_position + cell_offset) * randomness;\n"
    "      vec2 delta = point_position - local_position;\n"
    "      float distance_to_point_sq = dot(delta, delta);\n"
    "      if (distance_to_point_sq < min_distance_sq) {\n"
    "        min_distance_sq = distance_to_point_sq;\n"
    "        closest_point = point_position;\n"
    "        closest_point_offset = cell_offset;\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  min_distance_sq = 3.4028234663852886e+38;\n"
    "  vec2 closest_point_to_closest_point = vec2(0.0);\n"
    "  for (int j = -1; j <= 1; j++) {\n"
    "    for (int i = -1; i <= 1; i++) {\n"
    "      if (i == 0 && j == 0) { continue; }\n"
    "      ivec2 cell_offset = ivec2(i, j) + closest_point_offset;\n"
    "      vec2 point_position =\n"
    "          vec2(cell_offset) + ha_voronoi_hash2(cell_position + cell_offset) * randomness;\n"
    "      vec2 delta = closest_point - point_position;\n"
    "      float distance_to_point_sq = dot(delta, delta);\n"
    "      if (distance_to_point_sq < min_distance_sq) {\n"
    "        min_distance_sq = distance_to_point_sq;\n"
    "        closest_point_to_closest_point = point_position;\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  return length(closest_point_to_closest_point - closest_point) / 2.0;\n"
    "}\n"
    "HaVoronoiOctave2 ha_voronoi_fractal_2d(vec2 coord,\n"
    "                                      float scale,\n"
    "                                      float detail,\n"
    "                                      float roughness,\n"
    "                                      float lacunarity,\n"
    "                                      float smoothness,\n"
    "                                      float exponent,\n"
    "                                      float randomness,\n"
    "                                      int feature,\n"
    "                                      int metric,\n"
    "                                      bool normalize) {\n"
    "  float max_distance = ha_voronoi_distance2(\n"
    "                           vec2(0.0), vec2(0.5 + 0.5 * randomness), metric, exponent) *\n"
    "                       (feature == 1 ? 2.0 : 1.0);\n"
    "  float amplitude = 1.0;\n"
    "  float max_amplitude = 0.0;\n"
    "  float octave_scale = 1.0;\n"
    "  bool zero_input = detail == 0.0 || roughness == 0.0;\n"
    "  HaVoronoiOctave2 result;\n"
    "  result.distance = 0.0;\n"
    "  result.position = vec2(0.0);\n"
    "  int octave_count = int(ceil(detail));\n"
    "  for (int i = 0; i <= octave_count; i++) {\n"
    "    vec2 coord_scaled = coord * octave_scale;\n"
    "    HaVoronoiOctave2 octave;\n"
    "    if (feature == 1) {\n"
    "      octave = ha_voronoi_f2_2d(coord_scaled, randomness, metric, exponent);\n"
    "    }\n"
    "    else if (feature == 2 && smoothness != 0.0) {\n"
    "      octave = ha_voronoi_smooth_f1_2d(coord_scaled, randomness, metric, exponent, smoothness);\n"
    "    }\n"
    "    else {\n"
    "      octave = ha_voronoi_f1_2d(coord_scaled, randomness, metric, exponent);\n"
    "    }\n"
    "    if (zero_input) {\n"
    "      max_amplitude = 1.0;\n"
    "      result = octave;\n"
    "      break;\n"
    "    }\n"
    "    if (float(i) <= detail) {\n"
    "      max_amplitude += amplitude;\n"
    "      result.distance += octave.distance * amplitude;\n"
    "      result.position = mix(result.position, octave.position / octave_scale, amplitude);\n"
    "      octave_scale *= lacunarity;\n"
    "      amplitude *= roughness;\n"
    "    }\n"
    "    else {\n"
    "      float remainder = detail - floor(detail);\n"
    "      if (remainder != 0.0) {\n"
    "        max_amplitude = mix(max_amplitude, max_amplitude + amplitude, remainder);\n"
    "        result.distance = mix(\n"
    "            result.distance, result.distance + octave.distance * amplitude, remainder);\n"
    "        result.position = mix(\n"
    "            result.position,\n"
    "            mix(result.position, octave.position / octave_scale, amplitude),\n"
    "            remainder);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  if (normalize) {\n"
    "    result.distance /= max_amplitude * max_distance;\n"
    "  }\n"
    "  result.position = (scale != 0.0) ? result.position / scale : vec2(0.0);\n"
    "  return result;\n"
    "}\n"
    "float ha_voronoi_fractal_distance_to_edge_2d(vec2 coord,\n"
    "                                             float detail,\n"
    "                                             float roughness,\n"
    "                                             float lacunarity,\n"
    "                                             float randomness,\n"
    "                                             bool normalize) {\n"
    "  float amplitude = 1.0;\n"
    "  float max_amplitude = 0.5 + 0.5 * randomness;\n"
    "  float octave_scale = 1.0;\n"
    "  float distance_value = 8.0;\n"
    "  bool zero_input = detail == 0.0 || roughness == 0.0;\n"
    "  int octave_count = int(ceil(detail));\n"
    "  for (int i = 0; i <= octave_count; i++) {\n"
    "    float octave_distance = ha_voronoi_distance_to_edge_2d(coord * octave_scale, randomness);\n"
    "    if (zero_input) {\n"
    "      distance_value = octave_distance;\n"
    "      break;\n"
    "    }\n"
    "    if (float(i) <= detail) {\n"
    "      max_amplitude = mix(max_amplitude, (0.5 + 0.5 * randomness) / octave_scale, amplitude);\n"
    "      distance_value = mix(\n"
    "          distance_value, min(distance_value, octave_distance / octave_scale), amplitude);\n"
    "      octave_scale *= lacunarity;\n"
    "      amplitude *= roughness;\n"
    "    }\n"
    "    else {\n"
    "      float remainder = detail - floor(detail);\n"
    "      if (remainder != 0.0) {\n"
    "        float lerp_amplitude = mix(\n"
    "            max_amplitude, (0.5 + 0.5 * randomness) / octave_scale, amplitude);\n"
    "        max_amplitude = mix(max_amplitude, lerp_amplitude, remainder);\n"
    "        float lerp_distance = mix(\n"
    "            distance_value, min(distance_value, octave_distance / octave_scale), amplitude);\n"
    "        distance_value = mix(distance_value, min(distance_value, lerp_distance), remainder);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  if (normalize) {\n"
    "    distance_value /= max_amplitude;\n"
    "  }\n"
    "  return distance_value;\n"
    "}\n"
    "float ha_voronoi_texture_2d(vec3 vector,\n"
    "                            float scale,\n"
    "                            float detail,\n"
    "                            float roughness,\n"
    "                            float lacunarity,\n"
    "                            float smoothness,\n"
    "                            float exponent,\n"
    "                            float randomness,\n"
    "                            int feature,\n"
    "                            int metric,\n"
    "                            bool normalize) {\n"
    "  if (feature == 3) {\n"
    "    return ha_voronoi_fractal_distance_to_edge_2d(\n"
    "        vector.xy * scale, detail, roughness, lacunarity, randomness, normalize);\n"
    "  }\n"
    "  HaVoronoiOctave2 result = ha_voronoi_fractal_2d(vector.xy * scale,\n"
    "                                                 scale,\n"
    "                                                 detail,\n"
    "                                                 roughness,\n"
    "                                                 lacunarity,\n"
    "                                                 smoothness,\n"
    "                                                 exponent,\n"
    "                                                 randomness,\n"
    "                                                 feature,\n"
    "                                                 metric,\n"
    "                                                 normalize);\n"
    "  return result.distance;\n"
    "}\n"
    "vec3 ha_voronoi_position_2d(vec3 vector,\n"
    "                            float scale,\n"
    "                            float detail,\n"
    "                            float roughness,\n"
    "                            float lacunarity,\n"
    "                            float smoothness,\n"
    "                            float exponent,\n"
    "                            float randomness,\n"
    "                            int feature,\n"
    "                            int metric,\n"
    "                            bool normalize) {\n"
    "  HaVoronoiOctave2 result = ha_voronoi_fractal_2d(vector.xy * scale,\n"
    "                                                 scale,\n"
    "                                                 detail,\n"
    "                                                 roughness,\n"
    "                                                 lacunarity,\n"
    "                                                 smoothness,\n"
    "                                                 exponent,\n"
    "                                                 randomness,\n"
    "                                                 feature,\n"
    "                                                 metric,\n"
    "                                                 normalize);\n"
    "  return vec3(result.position, 0.0);\n"
    "}\n"
    "float ha_voronoi_radius_2d(vec3 vector, float scale, float randomness) {\n"
    "  return ha_voronoi_n_sphere_radius_2d(vector.xy * scale, randomness);\n"
    "}\n"
    /* ---- Voronoi, 4D ---------------------------------------------------------------------- */
    "struct HaVoronoiOctave4 {\n"
    "  float distance;\n"
    "  vec4 position;\n"
    "};\n"
    "ivec4 ha_voronoi_pcg4d(ivec4 v) {\n"
    "  v = v * 1664525 + 1013904223;\n"
    "  v.x += v.y * v.w;\n"
    "  v.y += v.z * v.x;\n"
    "  v.z += v.x * v.y;\n"
    "  v.w += v.y * v.z;\n"
    "  v = v ^ (v >> 16);\n"
    "  v.x += v.y * v.w;\n"
    "  v.y += v.z * v.x;\n"
    "  v.z += v.x * v.y;\n"
    "  v.w += v.y * v.z;\n"
    "  return v;\n"
    "}\n"
    "vec4 ha_voronoi_hash4(ivec4 k) {\n"
    "  ivec4 h = ha_voronoi_pcg4d(k);\n"
    "  return vec4(ha_voronoi_int_to_float_01(h.x),\n"
    "              ha_voronoi_int_to_float_01(h.y),\n"
    "              ha_voronoi_int_to_float_01(h.z),\n"
    "              ha_voronoi_int_to_float_01(h.w));\n"
    "}\n"
    "float ha_voronoi_distance4(vec4 a, vec4 b, int metric, float exponent) {\n"
    "  if (metric == 0) { return length(a - b); }\n"
    "  if (metric == 1) {\n"
    "    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z) + abs(a.w - b.w);\n"
    "  }\n"
    "  if (metric == 2) {\n"
    "    return max(max(abs(a.x - b.x), abs(a.y - b.y)),\n"
    "               max(abs(a.z - b.z), abs(a.w - b.w)));\n"
    "  }\n"
    "  return pow(pow(abs(a.x - b.x), exponent) + pow(abs(a.y - b.y), exponent) +\n"
    "                 pow(abs(a.z - b.z), exponent) + pow(abs(a.w - b.w), exponent),\n"
    "             1.0 / exponent);\n"
    "}\n"
    "float ha_voronoi_distance_bound4(vec4 a, vec4 b, int metric, float exponent) {\n"
    "  if (metric == 0) {\n"
    "    vec4 d = a - b;\n"
    "    return dot(d, d);\n"
    "  }\n"
    "  return ha_voronoi_distance4(a, b, metric, exponent);\n"
    "}\n"
    "HaVoronoiOctave4 ha_voronoi_f1_4d(vec4 coord, float randomness, int metric, float exponent) {\n"
    "  vec4 cell_position_f = floor(coord);\n"
    "  vec4 local_position = coord - cell_position_f;\n"
    "  ivec4 cell_position = ivec4(cell_position_f);\n"
    "  float min_distance = 3.4028234663852886e+38;\n"
    "  vec4 target_position = vec4(0.0);\n"
    "  for (int u = -1; u <= 1; u++) {\n"
    "    for (int k = -1; k <= 1; k++) {\n"
    "      for (int j = -1; j <= 1; j++) {\n"
    "        for (int i = -1; i <= 1; i++) {\n"
    "          ivec4 cell_offset = ivec4(i, j, k, u);\n"
    "          vec4 point_position =\n"
    "              vec4(cell_offset) + ha_voronoi_hash4(cell_position + cell_offset) * randomness;\n"
    "          float distance_to_point =\n"
    "              ha_voronoi_distance_bound4(point_position, local_position, metric, exponent);\n"
    "          if (distance_to_point < min_distance) {\n"
    "            min_distance = distance_to_point;\n"
    "            target_position = point_position;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave4 octave;\n"
    "  octave.distance = ha_voronoi_distance4(target_position, local_position, metric, exponent);\n"
    "  octave.position = target_position + cell_position_f;\n"
    "  return octave;\n"
    "}\n"
    "HaVoronoiOctave4 ha_voronoi_smooth_f1_4d(\n"
    "    vec4 coord, float randomness, int metric, float exponent, float smoothness) {\n"
    "  vec4 cell_position_f = floor(coord);\n"
    "  vec4 local_position = coord - cell_position_f;\n"
    "  ivec4 cell_position = ivec4(cell_position_f);\n"
    "  float smooth_distance = 0.0;\n"
    "  vec4 smooth_position = vec4(0.0);\n"
    "  float h = -1.0;\n"
    "  for (int u = -2; u <= 2; u++) {\n"
    "    for (int k = -2; k <= 2; k++) {\n"
    "      for (int j = -2; j <= 2; j++) {\n"
    "        for (int i = -2; i <= 2; i++) {\n"
    "          ivec4 cell_offset = ivec4(i, j, k, u);\n"
    "          vec4 point_position =\n"
    "              vec4(cell_offset) + ha_voronoi_hash4(cell_position + cell_offset) * randomness;\n"
    "          float distance_to_point =\n"
    "              ha_voronoi_distance4(point_position, local_position, metric, exponent);\n"
    "          h = h == -1.0 ?\n"
    "                  1.0 :\n"
    "                  smoothstep(0.0,\n"
    "                             1.0,\n"
    "                             0.5 + 0.5 * (smooth_distance - distance_to_point) / smoothness);\n"
    "          float correction_factor = smoothness * h * (1.0 - h);\n"
    "          smooth_distance = mix(smooth_distance, distance_to_point, h) - correction_factor;\n"
    "          correction_factor /= 1.0 + 3.0 * smoothness;\n"
    "          smooth_position = mix(smooth_position, point_position, h) - correction_factor;\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave4 octave;\n"
    "  octave.distance = smooth_distance;\n"
    "  octave.position = cell_position_f + smooth_position;\n"
    "  return octave;\n"
    "}\n"
    "HaVoronoiOctave4 ha_voronoi_f2_4d(vec4 coord, float randomness, int metric, float exponent) {\n"
    "  vec4 cell_position_f = floor(coord);\n"
    "  vec4 local_position = coord - cell_position_f;\n"
    "  ivec4 cell_position = ivec4(cell_position_f);\n"
    "  float distance_f1 = 3.4028234663852886e+38;\n"
    "  float distance_f2 = 3.4028234663852886e+38;\n"
    "  vec4 position_f1 = vec4(0.0);\n"
    "  vec4 position_f2 = vec4(0.0);\n"
    "  for (int u = -1; u <= 1; u++) {\n"
    "    for (int k = -1; k <= 1; k++) {\n"
    "      for (int j = -1; j <= 1; j++) {\n"
    "        for (int i = -1; i <= 1; i++) {\n"
    "          ivec4 cell_offset = ivec4(i, j, k, u);\n"
    "          vec4 point_position =\n"
    "              vec4(cell_offset) + ha_voronoi_hash4(cell_position + cell_offset) * randomness;\n"
    "          float distance_to_point =\n"
    "              ha_voronoi_distance4(point_position, local_position, metric, exponent);\n"
    "          if (distance_to_point < distance_f1) {\n"
    "            distance_f2 = distance_f1;\n"
    "            distance_f1 = distance_to_point;\n"
    "            position_f2 = position_f1;\n"
    "            position_f1 = point_position;\n"
    "          }\n"
    "          else if (distance_to_point < distance_f2) {\n"
    "            distance_f2 = distance_to_point;\n"
    "            position_f2 = point_position;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  HaVoronoiOctave4 octave;\n"
    "  octave.distance = distance_f2;\n"
    "  octave.position = position_f2 + cell_position_f;\n"
    "  return octave;\n"
    "}\n"
    "float ha_voronoi_distance_to_edge_4d(vec4 coord, float randomness) {\n"
    "  vec4 cell_position_f = floor(coord);\n"
    "  vec4 local_position = coord - cell_position_f;\n"
    "  ivec4 cell_position = ivec4(cell_position_f);\n"
    "  vec4 vector_to_closest = vec4(0.0);\n"
    "  float min_distance = 3.4028234663852886e+38;\n"
    "  for (int u = -1; u <= 1; u++) {\n"
    "    for (int k = -1; k <= 1; k++) {\n"
    "      for (int j = -1; j <= 1; j++) {\n"
    "        for (int i = -1; i <= 1; i++) {\n"
    "          ivec4 cell_offset = ivec4(i, j, k, u);\n"
    "          vec4 vector_to_point = vec4(cell_offset) +\n"
    "                                 ha_voronoi_hash4(cell_position + cell_offset) * randomness -\n"
    "                                 local_position;\n"
    "          float distance_to_point = dot(vector_to_point, vector_to_point);\n"
    "          if (distance_to_point < min_distance) {\n"
    "            min_distance = distance_to_point;\n"
    "            vector_to_closest = vector_to_point;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  min_distance = 3.4028234663852886e+38;\n"
    "  for (int u = -1; u <= 1; u++) {\n"
    "    for (int k = -1; k <= 1; k++) {\n"
    "      for (int j = -1; j <= 1; j++) {\n"
    "        for (int i = -1; i <= 1; i++) {\n"
    "          ivec4 cell_offset = ivec4(i, j, k, u);\n"
    "          vec4 vector_to_point = vec4(cell_offset) +\n"
    "                                 ha_voronoi_hash4(cell_position + cell_offset) * randomness -\n"
    "                                 local_position;\n"
    "          vec4 perpendicular_to_edge = vector_to_point - vector_to_closest;\n"
    "          if (dot(perpendicular_to_edge, perpendicular_to_edge) > 0.0001) {\n"
    "            float distance_to_edge = dot((vector_to_closest + vector_to_point) / 2.0,\n"
    "                                         normalize(perpendicular_to_edge));\n"
    "            min_distance = min(min_distance, distance_to_edge);\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  return min_distance;\n"
    "}\n"
    "float ha_voronoi_n_sphere_radius_4d(vec4 coord, float randomness) {\n"
    "  vec4 cell_position_f = floor(coord);\n"
    "  vec4 local_position = coord - cell_position_f;\n"
    "  ivec4 cell_position = ivec4(cell_position_f);\n"
    "  vec4 closest_point = vec4(0.0);\n"
    "  ivec4 closest_point_offset = ivec4(0);\n"
    "  float min_distance_sq = 3.4028234663852886e+38;\n"
    "  for (int u = -1; u <= 1; u++) {\n"
    "    for (int k = -1; k <= 1; k++) {\n"
    "      for (int j = -1; j <= 1; j++) {\n"
    "        for (int i = -1; i <= 1; i++) {\n"
    "          ivec4 cell_offset = ivec4(i, j, k, u);\n"
    "          vec4 point_position =\n"
    "              vec4(cell_offset) + ha_voronoi_hash4(cell_position + cell_offset) * randomness;\n"
    "          vec4 delta = point_position - local_position;\n"
    "          float distance_to_point_sq = dot(delta, delta);\n"
    "          if (distance_to_point_sq < min_distance_sq) {\n"
    "            min_distance_sq = distance_to_point_sq;\n"
    "            closest_point = point_position;\n"
    "            closest_point_offset = cell_offset;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  min_distance_sq = 3.4028234663852886e+38;\n"
    "  vec4 closest_point_to_closest_point = vec4(0.0);\n"
    "  for (int u = -1; u <= 1; u++) {\n"
    "    for (int k = -1; k <= 1; k++) {\n"
    "      for (int j = -1; j <= 1; j++) {\n"
    "        for (int i = -1; i <= 1; i++) {\n"
    "          if (i == 0 && j == 0 && k == 0 && u == 0) { continue; }\n"
    "          ivec4 cell_offset = ivec4(i, j, k, u) + closest_point_offset;\n"
    "          vec4 point_position =\n"
    "              vec4(cell_offset) + ha_voronoi_hash4(cell_position + cell_offset) * randomness;\n"
    "          vec4 delta = closest_point - point_position;\n"
    "          float distance_to_point_sq = dot(delta, delta);\n"
    "          if (distance_to_point_sq < min_distance_sq) {\n"
    "            min_distance_sq = distance_to_point_sq;\n"
    "            closest_point_to_closest_point = point_position;\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  return length(closest_point_to_closest_point - closest_point) / 2.0;\n"
    "}\n"
    "HaVoronoiOctave4 ha_voronoi_fractal_4d(vec4 coord,\n"
    "                                      float scale,\n"
    "                                      float detail,\n"
    "                                      float roughness,\n"
    "                                      float lacunarity,\n"
    "                                      float smoothness,\n"
    "                                      float exponent,\n"
    "                                      float randomness,\n"
    "                                      int feature,\n"
    "                                      int metric,\n"
    "                                      bool normalize) {\n"
    "  float max_distance = ha_voronoi_distance4(\n"
    "                           vec4(0.0), vec4(0.5 + 0.5 * randomness), metric, exponent) *\n"
    "                       (feature == 1 ? 2.0 : 1.0);\n"
    "  float amplitude = 1.0;\n"
    "  float max_amplitude = 0.0;\n"
    "  float octave_scale = 1.0;\n"
    "  bool zero_input = detail == 0.0 || roughness == 0.0;\n"
    "  HaVoronoiOctave4 result;\n"
    "  result.distance = 0.0;\n"
    "  result.position = vec4(0.0);\n"
    "  int octave_count = int(ceil(detail));\n"
    "  for (int i = 0; i <= octave_count; i++) {\n"
    "    vec4 coord_scaled = coord * octave_scale;\n"
    "    HaVoronoiOctave4 octave;\n"
    "    if (feature == 1) {\n"
    "      octave = ha_voronoi_f2_4d(coord_scaled, randomness, metric, exponent);\n"
    "    }\n"
    "    else if (feature == 2 && smoothness != 0.0) {\n"
    "      octave = ha_voronoi_smooth_f1_4d(coord_scaled, randomness, metric, exponent, smoothness);\n"
    "    }\n"
    "    else {\n"
    "      octave = ha_voronoi_f1_4d(coord_scaled, randomness, metric, exponent);\n"
    "    }\n"
    "    if (zero_input) {\n"
    "      max_amplitude = 1.0;\n"
    "      result = octave;\n"
    "      break;\n"
    "    }\n"
    "    if (float(i) <= detail) {\n"
    "      max_amplitude += amplitude;\n"
    "      result.distance += octave.distance * amplitude;\n"
    "      result.position = mix(result.position, octave.position / octave_scale, amplitude);\n"
    "      octave_scale *= lacunarity;\n"
    "      amplitude *= roughness;\n"
    "    }\n"
    "    else {\n"
    "      float remainder = detail - floor(detail);\n"
    "      if (remainder != 0.0) {\n"
    "        max_amplitude = mix(max_amplitude, max_amplitude + amplitude, remainder);\n"
    "        result.distance = mix(\n"
    "            result.distance, result.distance + octave.distance * amplitude, remainder);\n"
    "        result.position = mix(\n"
    "            result.position,\n"
    "            mix(result.position, octave.position / octave_scale, amplitude),\n"
    "            remainder);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  if (normalize) {\n"
    "    result.distance /= max_amplitude * max_distance;\n"
    "  }\n"
    "  result.position = (scale != 0.0) ? result.position / scale : vec4(0.0);\n"
    "  return result;\n"
    "}\n"
    "float ha_voronoi_fractal_distance_to_edge_4d(vec4 coord,\n"
    "                                             float detail,\n"
    "                                             float roughness,\n"
    "                                             float lacunarity,\n"
    "                                             float randomness,\n"
    "                                             bool normalize) {\n"
    "  float amplitude = 1.0;\n"
    "  float max_amplitude = 0.5 + 0.5 * randomness;\n"
    "  float octave_scale = 1.0;\n"
    "  float distance_value = 8.0;\n"
    "  bool zero_input = detail == 0.0 || roughness == 0.0;\n"
    "  int octave_count = int(ceil(detail));\n"
    "  for (int i = 0; i <= octave_count; i++) {\n"
    "    float octave_distance = ha_voronoi_distance_to_edge_4d(coord * octave_scale, randomness);\n"
    "    if (zero_input) {\n"
    "      distance_value = octave_distance;\n"
    "      break;\n"
    "    }\n"
    "    if (float(i) <= detail) {\n"
    "      max_amplitude = mix(max_amplitude, (0.5 + 0.5 * randomness) / octave_scale, amplitude);\n"
    "      distance_value = mix(\n"
    "          distance_value, min(distance_value, octave_distance / octave_scale), amplitude);\n"
    "      octave_scale *= lacunarity;\n"
    "      amplitude *= roughness;\n"
    "    }\n"
    "    else {\n"
    "      float remainder = detail - floor(detail);\n"
    "      if (remainder != 0.0) {\n"
    "        float lerp_amplitude = mix(\n"
    "            max_amplitude, (0.5 + 0.5 * randomness) / octave_scale, amplitude);\n"
    "        max_amplitude = mix(max_amplitude, lerp_amplitude, remainder);\n"
    "        float lerp_distance = mix(\n"
    "            distance_value, min(distance_value, octave_distance / octave_scale), amplitude);\n"
    "        distance_value = mix(distance_value, min(distance_value, lerp_distance), remainder);\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  if (normalize) {\n"
    "    distance_value /= max_amplitude;\n"
    "  }\n"
    "  return distance_value;\n"
    "}\n"
    "float ha_voronoi_texture_4d(vec3 vector,\n"
    "                            float w,\n"
    "                            float scale,\n"
    "                            float detail,\n"
    "                            float roughness,\n"
    "                            float lacunarity,\n"
    "                            float smoothness,\n"
    "                            float exponent,\n"
    "                            float randomness,\n"
    "                            int feature,\n"
    "                            int metric,\n"
    "                            bool normalize) {\n"
    "  vec4 coord = vec4(vector, w) * scale;\n"
    "  if (feature == 3) {\n"
    "    return ha_voronoi_fractal_distance_to_edge_4d(\n"
    "        coord, detail, roughness, lacunarity, randomness, normalize);\n"
    "  }\n"
    "  HaVoronoiOctave4 result = ha_voronoi_fractal_4d(coord,\n"
    "                                                 scale,\n"
    "                                                 detail,\n"
    "                                                 roughness,\n"
    "                                                 lacunarity,\n"
    "                                                 smoothness,\n"
    "                                                 exponent,\n"
    "                                                 randomness,\n"
    "                                                 feature,\n"
    "                                                 metric,\n"
    "                                                 normalize);\n"
    "  return result.distance;\n"
    "}\n"
    "vec3 ha_voronoi_position_4d(vec3 vector,\n"
    "                            float w,\n"
    "                            float scale,\n"
    "                            float detail,\n"
    "                            float roughness,\n"
    "                            float lacunarity,\n"
    "                            float smoothness,\n"
    "                            float exponent,\n"
    "                            float randomness,\n"
    "                            int feature,\n"
    "                            int metric,\n"
    "                            bool normalize) {\n"
    "  HaVoronoiOctave4 result = ha_voronoi_fractal_4d(vec4(vector, w) * scale,\n"
    "                                                 scale,\n"
    "                                                 detail,\n"
    "                                                 roughness,\n"
    "                                                 lacunarity,\n"
    "                                                 smoothness,\n"
    "                                                 exponent,\n"
    "                                                 randomness,\n"
    "                                                 feature,\n"
    "                                                 metric,\n"
    "                                                 normalize);\n"
    "  return result.position.xyz;\n"
    "}\n"
    "float ha_voronoi_w_4d(vec3 vector,\n"
    "                      float w,\n"
    "                      float scale,\n"
    "                      float detail,\n"
    "                      float roughness,\n"
    "                      float lacunarity,\n"
    "                      float smoothness,\n"
    "                      float exponent,\n"
    "                      float randomness,\n"
    "                      int feature,\n"
    "                      int metric,\n"
    "                      bool normalize) {\n"
    "  HaVoronoiOctave4 result = ha_voronoi_fractal_4d(vec4(vector, w) * scale,\n"
    "                                                 scale,\n"
    "                                                 detail,\n"
    "                                                 roughness,\n"
    "                                                 lacunarity,\n"
    "                                                 smoothness,\n"
    "                                                 exponent,\n"
    "                                                 randomness,\n"
    "                                                 feature,\n"
    "                                                 metric,\n"
    "                                                 normalize);\n"
    "  return result.position.w;\n"
    "}\n"
    "float ha_voronoi_radius_4d(vec3 vector, float w, float scale, float randomness) {\n"
    "  return ha_voronoi_n_sphere_radius_4d(vec4(vector, w) * scale, randomness);\n"
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

gpu::Shader *ha_shader_get_element(const StringRef key,
                                   const StringRef glsl_value,
                                   const StringRef glsl_selection,
                                   const int components,
                                   const Span<const HaShaderResource> extra_resources)
{
  const bool vector = components > 1;
  const char *type = vector ? "vec3" : "float";
  std::string body = std::string(HA_GLSL_HELPERS);
  body += "void main()\n";
  body += "{\n";
  body += "  int index = int(gl_GlobalInvocationID.x);\n";
  body += "  if (index >= int(params[0])) { return; }\n";
  body += "  ";
  body += type;
  body += " value = ";
  body += glsl_value;
  body += ";\n";
  body += "  if ((";
  body += glsl_selection;
  body += ") > 0.0) {\n";
  body += "    element_out[index] = value;\n";
  body += "  }\n";
  body += "  else {\n";
  body += "    element_out[index] = element_in[index];\n";
  body += "  }\n";
  body += "}\n";

  const std::array<HaShaderResource, 3> resources = {{
      {0, HaShaderQualifier::Read, type, "element_in[]"},
      {1, HaShaderQualifier::Write, type, "element_out[]"},
      {2, HaShaderQualifier::Read, "float", "params[]"},
  }};
  Vector<HaShaderResource> all_resources;
  all_resources.append(resources[0]);
  all_resources.append(resources[1]);
  all_resources.append(resources[2]);
  for (const HaShaderResource &resource : extra_resources) {
    all_resources.append(resource);
  }
  return ha_shader_get_source(
      key,
      body,
      Span<const HaShaderResource>(all_resources.data(), int64_t(all_resources.size())),
      LOCAL_GROUP_SIZE);
}

gpu::Shader *ha_shader_get_flip_faces(const StringRef key,
                                      const StringRef glsl_selection,
                                      const Span<const HaShaderResource> extra_resources)
{
  std::string body = std::string(HA_GLSL_HELPERS);
  body += "void main()\n";
  body += "{\n";
  body += "  int index = int(gl_GlobalInvocationID.x);\n";
  body += "  if (index >= int(params[0])) { return; }\n";
  body += "  int start = face_offsets[index];\n";
  body += "  int size = face_offsets[index + 1] - start;\n";
  body += "  if (size <= 0) { return; }\n";
  body += "  if ((";
  body += glsl_selection;
  body += ") > 0.0) {\n";
  /* `bke::mesh_flip_faces` swaps `corner_verts[face[j + 1]]` with `corner_verts[face.last(j)]`, so
   * the first corner stays and the rest is reversed; the edges are shifted by one, so it is the last
   * edge that stays. */
  body += "    for (int k = 0; k < size; k++) {\n";
  body += "      int src = (k == 0) ? 0 : (size - k);\n";
  body += "      corner_verts_out[start + k] = corner_verts_in[start + src];\n";
  body += "    }\n";
  body += "    for (int k = 0; k < size - 1; k++) {\n";
  body += "      corner_edges_out[start + k] = corner_edges_in[start + (size - 2 - k)];\n";
  body += "    }\n";
  body += "    corner_edges_out[start + size - 1] = corner_edges_in[start + size - 1];\n";
  body += "  }\n";
  body += "  else {\n";
  body += "    for (int k = 0; k < size; k++) {\n";
  body += "      corner_verts_out[start + k] = corner_verts_in[start + k];\n";
  body += "      corner_edges_out[start + k] = corner_edges_in[start + k];\n";
  body += "    }\n";
  body += "  }\n";
  body += "}\n";

  Vector<HaShaderResource> resources;
  resources.append({0, HaShaderQualifier::Read, "int", "corner_verts_in[]"});
  resources.append({1, HaShaderQualifier::Write, "int", "corner_verts_out[]"});
  resources.append({2, HaShaderQualifier::Read, "int", "corner_edges_in[]"});
  resources.append({3, HaShaderQualifier::Write, "int", "corner_edges_out[]"});
  resources.append({4, HaShaderQualifier::Read, "int", "face_offsets[]"});
  resources.append({5, HaShaderQualifier::Read, "float", "params[]"});
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