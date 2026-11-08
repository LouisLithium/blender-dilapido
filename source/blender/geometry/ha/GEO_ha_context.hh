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

#include <atomic>
#include <cstdint>

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
 * Makes sure a GPU context is active for the current scope. When there is no context available the
 * work is skipped (dispatch would assert), so always check #is_valid().
 *
 * All device work of this module runs in one **dedicated context** (see #ha_context), never in the
 * draw manager's context: shaders and storage buffers are per context, and an evaluation that ran
 * in the main thread one frame and on a depsgraph worker the next used to mix both, which crashes
 * (measured: `EXCEPTION_ACCESS_VIOLATION` inside the lazy-function executor and inside the GL
 * readback in the GUI). The context is created lazily on the main thread and released right after,
 * so any thread can activate it on demand; the whole dispatch sequence is serialized.
 *
 * Guards nest for free on the same thread: the outer one owns the context, the inner ones only
 * count. `BLENDER_HA_DRAW_CONTEXT=1` restores the old "use the thread's active context" behaviour
 * for A/B testing.
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
  /** Fails when the current context is not the one our GPU objects live in. */
  bool accept_current_context();

  bool valid_ = false;
  bool enabled_by_us_ = false;
  bool owning_mutex_ = false;
  bool nested_ = false;
  /** The draw manager's context was borrowed for this scope (see #DRW_gpu_context_try_enable). */
  bool borrowed_draw_context_ = false;
};

/* -------------------------------------------------------------------- */
/** \name Backend scope
 *
 * An `HW Accelerate` zone declares that everything inside it should run on the compute backend.
 * The zone body is still evaluated by the lazy-function executor on the CPU, so the request is
 * handed to the nodes through a thread-local scope: while the zone body runs, any node that
 * supports a GPU path (for example the XPBD solver) can ask #ha_scope_allows_gpu() and dispatch
 * itself on the device. Nested zones push/pop, so the innermost zone wins.
 * \{ */

enum class HaScopeBackend {
  None = 0,
  Auto,
  Strict,
  ForceCpu,
};

class HaScope {
 public:
  explicit HaScope(HaScopeBackend backend);
  ~HaScope();
  HaScope(const HaScope &) = delete;
  HaScope &operator=(const HaScope &) = delete;

 private:
  bool pushed_ = false;
};

/** Backend requested by the innermost zone on this thread, #HaScopeBackend::None outside of zones. */
HaScopeBackend ha_scope_backend();

/** True when the current evaluation may use the GPU (i.e. inside a zone that is not `Force CPU`). */
bool ha_scope_allows_gpu();

/**
 * Why the last #HaContextGuard on this thread could not be created, or an empty string when it was
 * valid. Used by the nodes to report an actionable reason instead of the generic "no GPU context".
 */
const char *ha_context_error();

/** \} */

/* -------------------------------------------------------------------- */
/** \name Device memory budget
 *
 * A `HW Accelerate` zone can carry a VRAM budget. Every `HaBuffer` allocation is charged to the
 * innermost budget on the current evaluation. Once an allocation would cross the limit the budget
 * is latched as exceeded and all further allocations fail, so the accelerated nodes stop dispatching
 * and fall back to the CPU instead of letting the driver run out of memory.
 *
 * The GPU work of this module is already serialized by the context guard, so one active budget at a
 * time is enough. Nested zones save and restore the previous budget.
 * \{ */

class HaMemoryBudget {
 public:
  void set_limit(int64_t bytes);
  int64_t limit() const;
  int64_t used() const;
  int64_t peak() const;
  bool exceeded() const;

  bool reserve(int64_t bytes);
  void release(int64_t bytes);

 private:
  std::atomic<int64_t> limit_{0};
  std::atomic<int64_t> used_{0};
  std::atomic<int64_t> peak_{0};
  std::atomic<bool> exceeded_{false};
};

class HaMemoryScope {
 public:
  explicit HaMemoryScope(HaMemoryBudget *budget);
  ~HaMemoryScope();
  HaMemoryScope(const HaMemoryScope &) = delete;
  HaMemoryScope &operator=(const HaMemoryScope &) = delete;

 private:
  HaMemoryBudget *previous_ = nullptr;
};

/** Budget of the innermost zone being evaluated on this thread, null outside of zones. */
HaMemoryBudget *ha_memory_budget_current();

/** \} */

/** A storage buffer holding `float` elements (used for interleaved positions: 3 per element). */class HaBuffer {
 public:
  HaBuffer() = default;
  ~HaBuffer();
  HaBuffer(const HaBuffer &) = delete;
  HaBuffer &operator=(const HaBuffer &) = delete;

  /** Allocates (or reuses) a buffer for exactly \a elements floats. */
  bool ensure(int64_t elements);
  void upload(Span<float> data);
  void download(MutableSpan<float> r_data) const;
  void free();

  bool is_valid() const
  {
    return buf_ != nullptr;
  }

  /** Number of floats the buffer can hold. */
  int64_t size() const
  {
    return elements_;
  }

  gpu::StorageBuf *buf() const
  {
    return buf_;
  }

 private:
  gpu::StorageBuf *buf_ = nullptr;
  int64_t elements_ = 0;
  HaMemoryBudget *budget_ = nullptr;
  int64_t reserved_bytes_ = 0;
};

/**
 * The same for `int` elements. Indices, counts and offsets are integers; keeping them in their own
 * buffer type means the shader can declare `int` arrays without bit casting through floats.
 */
class HaBufferInt {
 public:
  HaBufferInt() = default;
  ~HaBufferInt();
  HaBufferInt(const HaBufferInt &) = delete;
  HaBufferInt &operator=(const HaBufferInt &) = delete;

  bool ensure(int64_t elements);
  void upload(Span<int> data);
  void download(MutableSpan<int> r_data) const;
  /** Reads the whole buffer, used for small buffers only (`GPU_storagebuf_read` is a full read). */
  bool read_one(int &r_value) const;
  void clear_to_zero();
  void free();

  bool is_valid() const
  {
    return buf_ != nullptr;
  }

  int64_t size() const
  {
    return elements_;
  }

  gpu::StorageBuf *buf() const
  {
    return buf_;
  }

 private:
  gpu::StorageBuf *buf_ = nullptr;
  int64_t elements_ = 0;
  HaMemoryBudget *budget_ = nullptr;
  int64_t reserved_bytes_ = 0;
};

/** Access qualifier of one storage buffer binding, mirrors `gpu::shader::Qualifier`. */
enum class HaShaderQualifier {
  Read,
  Write,
  ReadWrite,
};

/** One storage buffer binding of a hand written compute kernel. */
struct HaShaderResource {
  int slot;
  HaShaderQualifier qualifier;
  /** GLSL element type, e.g. `"float"` or `"int"`. */
  const char *type;
  /** GLSL declaration, e.g. `"position_in[]"`. */
  const char *name;
};

/**
 * Returns a compute shader that writes `position_out[]` from \a glsl_position (a `vec3`
 * expression) for the elements where \a glsl_selection (a scalar expression) is not zero, and
 * copies the input position through otherwise. Cached by \a key. In scope of both expressions:
 * `vec3 position` (the current position) and `int index`.
 *
 * \param extra_resources additional bindings starting at slot 3, used for the POINT attributes a
 * program reads (declared as e.g. `attribute_0[]`, one float per element for a scalar attribute and
 * three for a vector one).
 */
gpu::Shader *ha_shader_get(StringRef key,
                           StringRef glsl_position,
                           StringRef glsl_selection,
                           Span<const HaShaderResource> extra_resources = {},
                           bool with_lut = false);

/**
 * Compiles a `Store Named Attribute` kernel: writes \a glsl_value (evaluated with `vec3 position`
 * and `int index` in scope) into `store_out[]`, which has \a components elements of \a glsl_type
 * per element. Used with the *final* position buffer bound as `position_in`, so the value field
 * sees the geometry as it leaves the zone.
 */
gpu::Shader *ha_shader_get_store(StringRef key,
                                 StringRef glsl_value,
                                 StringRef glsl_selection,
                                 bool with_old,
                                 const char *glsl_type,
                                 int components,
                                 Span<const HaShaderResource> extra_resources = {});

/**
 * Compiles a `Store Named Attribute` kernel for a non-POINT domain. One invocation per element;
 * `element_in[]` / `element_out[]` are flat scalar arrays (`int` for integer attributes, `float`
 * otherwise). The value is written for the selected elements and the old value is copied through
 * otherwise. `params[0]` is the element count.
 */
gpu::Shader *ha_shader_get_element_store(StringRef key,
                                         StringRef glsl_value,
                                         StringRef glsl_selection,
                                         const char *glsl_type,
                                         int components,
                                         Span<const HaShaderResource> extra_resources = {});

/**
 * Compiles a kernel for an element domain that is not POINT (FACE / EDGE): for every element, if
 * \a glsl_selection (a scalar expression, evaluated with `int index` in scope) is not zero, writes
 * \a glsl_value into `element_out[]`, otherwise copies `element_in[]` through. `params[0]` is the
 * element count (the dispatch is padded to the work group size). `components` is 1 or 3.
 */
gpu::Shader *ha_shader_get_element(StringRef key,
                                   StringRef glsl_value,
                                   StringRef glsl_selection,
                                   int components,
                                   Span<const HaShaderResource> extra_resources = {});

/**
 * Compiles a `Flip Faces` kernel: one invocation per face. For the selected faces the corner ring
 * is reversed the way `bke::mesh_flip_faces` does it (`corner_verts` keeps its first corner and
 * reverses the rest, `corner_edges` reverses everything but the last one); unselected faces are
 * copied through. `face_offsets[]` is the mesh's face offset array, `params[0]` the face count.
 */
gpu::Shader *ha_shader_get_flip_faces(StringRef key,
                                      StringRef glsl_selection,
                                      Span<const HaShaderResource> extra_resources = {});

/**
 * Compiles \a source (a complete `void main()`, without the storage buffer declarations) with the
 * given bindings and caches it by \a cache_key. The cache key is mixed with a hash of the source,
 * so editing a kernel can never hit a stale entry in the on-disk shader cache.
 *
 * The device layer prepends the `CREATE_INFO_RES_PASS_*` block so the generated shader gets the
 * resource declarations of its create-info.
 */
gpu::Shader *ha_shader_get_source(StringRef cache_key,
                                 StringRef source,
                                 Span<const HaShaderResource> resources,
                                 int local_group_size);

/**
 * First reduction pass: one invocation per block of `BLOCK_SIZE` elements, writing one partial value
 * per block (at `partial[block * 3]`). `glsl_value_expr` is evaluated for every element with
 * `vec3 position` / `int index` in scope; `combine` is e.g. `"(acc + value)"` and `identity` is the
 * neutral element (`"0.0"` for a sum, `"-1e30"` / `"1e30"` for min/max).
 */
gpu::Shader *ha_shader_get_reduce_stage1(StringRef key,
                                         StringRef glsl_value_expr,
                                         StringRef combine,
                                         StringRef identity,
                                         int components,
                                         Span<const HaShaderResource> extra_resources = {});

/** Second reduction pass: single invocation folding the partial values into the aggregate. */
gpu::Shader *ha_shader_get_reduce_stage2(
    StringRef key, StringRef combine, StringRef identity, int components, bool divide_by_count);

/** Dispatches `ceil(element_count / 64)` work groups. */
void ha_dispatch(gpu::Shader *shader, int element_count);

/** Highest storage buffer slot the device layer uses. */
constexpr int HA_MAX_BIND_SLOTS = 16;

/**
 * Binds \a buffers to the slots \a first_slot, \a first_slot + 1, ... and remembers what is bound,
 * so a pass that reuses a binding does not make another driver call. Nothing unbinds: leaving a
 * buffer bound is harmless, and the unbinds used to double the number of calls per pass (measured:
 * 8 particles still cost 12.7 ms per frame with 116 passes, i.e. the calls are the floor).
 *
 * Invalidate with #ha_bindings_invalidate whenever something other than this module may have bound
 * storage buffers in between (Blender's own draw/compositor code, or a direct bind in the geometry
 * layer).
 */
void ha_bind_storage_buffers(Span<gpu::StorageBuf *> buffers, int first_slot = 0);

/** Forgets what #ha_bind_storage_buffers bound. */
void ha_bindings_invalidate();

}  // namespace geometry::ha
}  // namespace blender
