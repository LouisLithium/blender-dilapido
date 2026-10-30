/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Device layer of the hardware accelerated backend: compute context bootstrap, storage buffers,
 * shader creation from inline GLSL and dispatch.
 *
 * This intentionally uses the `gpu::` module (OpenGL / Metal / Vulkan) instead of talking to a
 * specific API. In background mode the GPU has to be initialized by the host (the Python side uses
 * `gpu.init()`), because this library must not depend on the window manager or GHOST.
 */

#pragma once

#include "BLI_span.hh"
#include "BLI_string_ref.hh"

namespace blender {
namespace gpu {
class Shader;
class StorageBuf;
}  // namespace gpu
namespace geometry::ha {

/** Elements processed by one invocation of the reduction pass. Also used by `GEO_ha_program`. */
inline constexpr int HA_REDUCE_BLOCK_SIZE = 1024;

/** The GPU backend is initialized and usable. Never initializes it on its own. */
bool ha_gpu_available();

/** Human readable backend name for debug output, e.g. "OPENGL / Intel(R) UHD Graphics". */
const char *ha_gpu_backend_name();

/**
 * Makes sure a GPU context is active for the current scope. When there is no active context the
 * work is skipped (dispatch would assert), so always check #is_valid().
 */
class HaContextGuard {
 public:
  HaContextGuard();
  ~HaContextGuard();
  HaContextGuard(const HaContextGuard &) = delete;
  HaContextGuard &operator=(const HaContextGuard &) = delete;

  bool is_valid() const
  {
    return valid_;
  }

 private:
  bool valid_ = false;
  bool enabled_by_us_ = false;
  bool owning_mutex_ = false;
};

/** A storage buffer holding `float` elements (used for interleaved positions: 3 per element). */
class HaBuffer {
 public:
  HaBuffer() = default;
  ~HaBuffer();
  HaBuffer(const HaBuffer &) = delete;
  HaBuffer &operator=(const HaBuffer &) = delete;

  /** Allocates (or reuses) a buffer for exactly \a elements floats. */
  bool ensure(int64_t elements);
  void upload(Span<float> data);
  void download(MutableSpan<float> r_data);
  void free();

  bool is_valid() const
  {
    return buf_ != nullptr;
  }

  gpu::StorageBuf *buf() const
  {
    return buf_;
  }

 private:
  gpu::StorageBuf *buf_ = nullptr;
  int64_t elements_ = 0;
};

/**
 * Returns a compute shader whose body is \a glsl_body (the statement list of `main`), cached by
 * \a key. The shader reads `position_in[]` and `aggregate[]`, writes `position_out[]` and provides
 * `vec3 position` and `int index` to the body.
 */
gpu::Shader *ha_shader_get(StringRef key, StringRef glsl_body);

/**
 * First reduction pass: one invocation per block of `BLOCK_SIZE` elements, writing one partial value
 * per block (at `partial[block * 3]`). `glsl_value_expr` is evaluated for every element with
 * `vec3 position` / `int index` in scope; `combine` is e.g. `"(acc + value)"` and `identity` is the
 * neutral element (`"0.0"` for a sum, `"-1e30"` / `"1e30"` for min/max).
 */
gpu::Shader *ha_shader_get_reduce_stage1(StringRef key,
                                         StringRef glsl_value_expr,
                                         StringRef combine,
                                         StringRef identity);

/** Second reduction pass: single invocation folding the partial values into the aggregate. */
gpu::Shader *ha_shader_get_reduce_stage2(
    StringRef key, StringRef combine, StringRef identity, bool divide_by_count);

/** Dispatches `ceil(element_count / 64)` work groups. */
void ha_dispatch(gpu::Shader *shader, int element_count);

}  // namespace geometry::ha
}  // namespace blender
