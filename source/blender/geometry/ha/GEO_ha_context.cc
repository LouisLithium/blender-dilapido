/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GEO_ha_context.hh"

#include <algorithm>
#include <array>
#include <mutex>
#include <string>

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

GHOST_IContext *&ha_system_context()
{
  static GHOST_IContext *context = nullptr;
  return context;
}

GPUContext *&ha_context()
{
  static GPUContext *context = nullptr;
  return context;
}

bool ha_create_context()
{
  /* Creating a GHOST offscreen context is documented as a main thread operation (see
   * `RenderDisplay::ensure_system_gpu_context`). Allow an opt-in for experiments: if the backend
   * tolerates it, background mode can use the GPU as well. */
  if (!BLI_thread_is_main() && getenv("BLENDER_HA_THREAD_CONTEXT") == nullptr) {
    ha_debug("cannot create the HA context off the main thread");
    return false;
  }
  GHOST_IContext *system_context = WM_system_gpu_context_create();
  if (system_context == nullptr) {
    ha_debug("WM_system_gpu_context_create failed");
    return false;
  }
  /* Creation leaves the context current; release it so the guard can bind it per dispatch. */
  WM_system_gpu_context_release(system_context);
  GPUContext *context = GPU_context_create(nullptr, system_context);
  if (context == nullptr) {
    WM_system_gpu_context_dispose(system_context);
    ha_debug("GPU_context_create failed");
    return false;
  }
  ha_system_context() = system_context;
  ha_context() = context;
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
  /* Fast path: the main thread usually has the draw manager's context bound already. */
  if (BLI_thread_is_main() && GPU_context_active_get() != nullptr) {
    valid_ = true;
    return;
  }

  /* Slow path: borrow the dedicated HA context. A GL context can only be current on one thread at
   * a time, so the whole dispatch sequence is serialized. */
  ha_context_mutex().lock();
  owning_mutex_ = true;

  if (ha_context() == nullptr && !ha_create_context()) {
    ha_context_mutex().unlock();
    owning_mutex_ = false;
    return;
  }

  WM_system_gpu_context_activate(ha_system_context());
  GPU_context_active_set(ha_context());
  GPU_context_begin_frame(ha_context());
  enabled_by_us_ = true;
  valid_ = true;
}

HaContextGuard::~HaContextGuard()
{
  if (enabled_by_us_) {
    GPU_context_end_frame(ha_context());
    GPU_context_active_set(nullptr);
    WM_system_gpu_context_release(ha_system_context());
  }
  if (owning_mutex_) {
    ha_context_mutex().unlock();
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Buffers
 * \{ */

HaBuffer::~HaBuffer()
{
  this->free();
}

bool HaBuffer::ensure(const int64_t elements)
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

void HaBuffer::download(const MutableSpan<float> r_data)
{
  if (buf_ == nullptr) {
    return;
  }
  const int64_t total = elements_;
  Array<float> tmp(total);
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
  }
  elements_ = 0;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shaders and dispatch
 * \{ */

namespace {

struct HaResource {
  int slot;
  gpu::shader::Qualifier qualifier;
  const char *type;
  const char *name;
};

/**
 * Compiles \a source (a complete `main` plus the resource declaration block) with the given storage
 * buffers and caches it by \a cache_key.
 */
gpu::Shader *ha_shader_from_source(const StringRef cache_key,
                                   const StringRef source,
                                   const Span<const HaResource> resources,
                                   const int local_group_size)
{
  std::string key(cache_key);
  if (gpu::Shader **found = shader_cache().lookup_ptr(key)) {
    return *found;
  }

  using namespace gpu::shader;

  /* Version of the generated shader pipeline; bump whenever the resource layout or the kernel
   * templates change, so that the on-disk shader cache cannot return a stale binary. */
  constexpr int HA_SHADER_VERSION = 2;

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
  for (const HaResource &resource : resources) {
    info.storage_buf(resource.slot, resource.qualifier, resource.type, resource.name);
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

const std::array<HaResource, 3> POSITION_RESOURCES = {{
    {0, gpu::shader::Qualifier::read, "float", "position_in[]"},
    {1, gpu::shader::Qualifier::write, "float", "position_out[]"},
    {2, gpu::shader::Qualifier::read, "float", "aggregate[]"},
}};

const std::array<HaResource, 3> REDUCE_STAGE1_RESOURCES = {{
    {0, gpu::shader::Qualifier::read, "float", "position_in[]"},
    {1, gpu::shader::Qualifier::read, "float", "params[]"},
    {2, gpu::shader::Qualifier::write, "float", "partial[]"},
}};

const std::array<HaResource, 3> REDUCE_STAGE2_RESOURCES = {{
    {0, gpu::shader::Qualifier::read, "float", "partial[]"},
    {1, gpu::shader::Qualifier::read, "float", "params[]"},
    {2, gpu::shader::Qualifier::write, "float", "aggregate[]"},
}};

Span<const HaResource> as_span(const std::array<HaResource, 3> &resources)
{
  return Span<const HaResource>(resources.data(), int64_t(resources.size()));
}

}  // namespace

gpu::Shader *ha_shader_get(const StringRef key, const StringRef glsl_body)
{
  const std::string source =
      std::string("void main()\n"
                  "{\n"
                  "  int index = int(gl_GlobalInvocationID.x);\n"
                  "  int base = index * 3;\n"
                  "  vec3 position = vec3(position_in[base], position_in[base + 1], "
                  "position_in[base + 2]);\n"
                  "  vec3 offset = ") +
      std::string(glsl_body) +
      ";\n"
      "  position_out[base] = position.x + offset.x;\n"
      "  position_out[base + 1] = position.y + offset.y;\n"
      "  position_out[base + 2] = position.z + offset.z;\n"
      "}\n";

  return ha_shader_from_source(
      key, source, as_span(POSITION_RESOURCES), LOCAL_GROUP_SIZE);
}

gpu::Shader *ha_shader_get_reduce_stage1(const StringRef key,
                                         const StringRef glsl_value_expr,
                                         const StringRef combine,
                                         const StringRef identity)
{
  /* One invocation per block; the loop is serial inside the block so that the summation order is
   * fixed (deterministic and close to the CPU result). */
  const std::string source = std::string("void main()\n"
                                         "{\n"
                                         "  int block = int(gl_GlobalInvocationID.x);\n"
                                         "  int element_count = int(params[0]);\n"
                                         "  int block_count = int(params[1]);\n"
                                         "  if (block >= block_count) {\n"
                                         "    return;\n"
                                         "  }\n"
                                         "  int start = block * ") +
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

  return ha_shader_from_source(
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

  return ha_shader_from_source(
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

/** \} */

}  // namespace blender::geometry::ha
