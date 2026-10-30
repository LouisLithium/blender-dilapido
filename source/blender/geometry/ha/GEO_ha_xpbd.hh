/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * XPBD fluid on the compute backend.
 *
 * The CPU implementation lives in `GEO_xpbd_fluid.hh` and this is its device mirror: the same
 * uniform hash grid, the same neighbour lists and the same pass structure. Both are kept
 * side by side on purpose, because the CPU one is the reference implementation that the GPU one is
 * verified against (see `BLENDER_XPBD_GPU_NEIGHBORS` in the XPBD solver).
 */

#pragma once

#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

#include <functional>

#include "GEO_ha_context.hh"

namespace blender::geometry::ha {

/** Floats per particle in the packed property buffer of #HaXpbdFluid. */
inline constexpr int HA_PROP_STRIDE = 10;
/** Floats per particle in the packed solver state buffer of #HaXpbdFluid. */
inline constexpr int HA_STATE_STRIDE = 8;
/** Floats per boundary particle (`xyz` + mass). */
inline constexpr int HA_BOUNDARY_STRIDE = 4;
/** Indices inside one particle block of the packed solver state buffer. */
enum HaStateIndex {
  HA_STATE_LAMBDA = 0,
  HA_STATE_ERROR = 1,
  HA_STATE_DENSITY_ERROR = 2,
  HA_STATE_SOLID_FRACTION = 3,
  HA_STATE_CORRECTION = 4,
  HA_STATE_OMEGA_MAGNITUDE = 7,
};
/** Indices inside one particle block of the packed property buffer. */
enum HaPropIndex {
  HA_PROP_MASS = 0,
  HA_PROP_INV_MASS = 1,
  HA_PROP_REST_DENSITY = 2,
  HA_PROP_NEGATIVE_PRESSURE_LIMIT = 3,
  HA_PROP_VISCOSITY = 4,
  HA_PROP_DAMPING = 5,
  HA_PROP_VORTICITY = 6,
  HA_PROP_SURFACE_TENSION = 7,
  HA_PROP_FRICTION = 8,
  HA_PROP_USE_BOUNDARY = 9,
};

/**
 * Timing of the last build, host wall clock around the whole dispatch sequence. The read back of
 * the results is not part of it.
 */
struct HaXpbdGridStats {
  double grid_ms = 0.0;
  double neighbors_ms = 0.0;
  int grid_passes = 0;
  int neighbor_passes = 0;
};

/**
 * Device mirror of `xpbd::FluidParticleGrid` plus the CSR neighbour list of
 * `xpbd::FluidNeighborList`.
 *
 * The layout, the hash function, the table size and the iteration order match the CPU version
 * exactly, so the neighbour lists are identical element by element and the density sums have the
 * same summation order. That is what makes "same result on both backends" a checkable property
 * instead of a hope.
 *
 * Differences to the CPU version, all of them about determinism:
 * - the bucket scatter uses `atomicAdd` on a cursor (order is undefined), but every bucket is then
 *   sorted by particle index, which reproduces the counting sort of the CPU version;
 * - the prefix sums are two level scans (block of #HA_REDUCE_BLOCK_SIZE, then the block sums), all
 *   serial inside a block, so they do not depend on the scheduling either.
 */
struct HaXpbdGrid {
  int particle_count = 0;
  int table_size = 0;
  float cell_size = 0.0f;
  int block_count = 0;
  /** Number of neighbours of all particles together. Only valid after #ha_xpbd_neighbors_total. */
  int neighbor_total = 0;
  int neighbor_capacity = 0;

  bool built = false;
  bool neighbors_built = false;

  HaXpbdGridStats stats;

  /** Device copy of the particle positions (3 floats per particle), shared by all passes. */
  HaBuffer positions;
  /* params[0] = particle count, [1] = table size, [2] = cell size, [3] = r_sq_max,
   * [4] = scan element count, [5] = scan block count, [6..7] unused. */
  HaBuffer params;
  /** Same layout for the two level scan passes (element count, block count). */
  HaBuffer scan_params;
  /** Cell coordinates, 3 ints per particle. */
  HaBufferInt cells;
  /** Particles per hash bucket. */
  HaBufferInt bucket_counts;
  /** Exclusive prefix sum of #bucket_counts. */
  HaBufferInt bucket_offsets;
  /** Copy of #bucket_offsets that the scatter consumes. */
  HaBufferInt bucket_cursor;
  /** Particle indices grouped by bucket, ascending inside a bucket. */
  HaBufferInt sorted;
  /** Scratch of the two level scan. */
  HaBufferInt scan_block_sums;
  HaBufferInt scan_block_offsets;
  /** One element: the total of the last scan. */
  HaBufferInt scan_total;
  /** Neighbours per particle, and their exclusive prefix sum (CSR offsets). */
  HaBufferInt neighbor_counts;
  HaBufferInt neighbor_offsets;
  HaBufferInt neighbor_indices;

  bool is_valid() const
  {
    return built;
  }

  void free();
};

/**
 * Builds the cell/bucket/sorted arrays on the GPU, and uploads \a positions (the grid keeps them,
 * every later pass reads them from the device). Returns false when the GPU is not usable (no
 * context, allocation failure, shader error), in which case the caller has to use the CPU path.
 */
bool ha_xpbd_grid_build(const Span<float3> positions, float support_radius, HaXpbdGrid &r_grid);

/**
 * The same, for a grid whose #HaXpbdGrid::positions already holds the current particle positions
 * (that is the case for #HaXpbdFluid, where the positions never leave the device).
 */
bool ha_xpbd_grid_build_device(HaXpbdGrid &r_grid);

/**
 * Builds the CSR neighbour list on the GPU from the positions uploaded by #ha_xpbd_grid_build.
 *
 * The number of neighbour indices is not read back (that would stall the pipeline every substep):
 * the index buffer grows to a heuristic capacity and the fill pass clamps to it. Use
 * #ha_xpbd_neighbors_total once per frame to get the exact number, or read it from the last entry
 * of the offsets after #ha_xpbd_neighbors_download.
 */
bool ha_xpbd_neighbors_build(HaXpbdGrid &r_grid);

/** Reads the number of neighbour indices of the last build (stalls the pipeline). */
bool ha_xpbd_neighbors_total(const HaXpbdGrid &grid, int &r_total);

/**
 * Copies the neighbour list back to the host, for comparison with `xpbd::FluidNeighborList`.
 * `r_offsets` has to be `particle_count + 1` elements (the last one is the total).
 */
void ha_xpbd_neighbors_download(const HaXpbdGrid &grid,
                                const MutableSpan<int> r_offsets,
                                const MutableSpan<int> r_indices);

/* -------------------------------------------------------------------- */
/** \name Fluid solve
 * \{ */

/**
 * Everything the fluid passes need per particle and per frame.
 *
 * The per particle scalars are packed into one interleaved buffer (#HA_PROP_STRIDE floats per
 * particle) instead of ten separate buffers, because a GLSL program can only bind a limited number
 * of storage buffers and the density pass would otherwise need fourteen.
 *
 * The arrays are only read, so they can outlive the call when they belong to the solver state.
 */
struct HaXpbdFluidInputs {
  Span<float3> positions;
  Span<float3> velocities;
  Span<float3> external_force;
  Span<float> masses;
  Span<float> inv_masses;
  Span<float> rest_density;
  Span<float> negative_pressure_limit;
  Span<float> viscosity;
  Span<float> damping;
  Span<float> vorticity;
  Span<float> surface_tension;
  Span<float> boundary_friction;
  /** `true` where the particles may use the collider boundary particles, `false` otherwise. */
  Span<bool> use_boundary;
  /**
   * Particles sampled on the colliders (Akinci et al.), refreshed once per frame by the solver.
   * They take part in the density estimate and push the fluid away, but never move.
   */
  Span<float3> boundary_positions;
  Span<float> boundary_masses;
};

  /** Coarse timings of one substep, filled by #HaXpbdFluid::substep. */
struct HaXpbdFluidStats {
  double grid_ms = 0.0;
  double neighbors_ms = 0.0;
  double solve_ms = 0.0;
  double velocity_ms = 0.0;
  int passes = 0;
};

/** The same, summed over all substeps of one frame. */
struct HaXpbdFluidFrameStats {
  double grid_ms = 0.0;
  double neighbors_ms = 0.0;
  double solve_ms = 0.0;
  double velocity_ms = 0.0;
  int substeps = 0;
  int passes = 0;
};

/**
 * Point vs collider triangles, gathered on the host once per substep (`ExternalFaceContacts` in the
 * solver) and solved on the device. Only the fields the device needs are passed.
 */
struct HaXpbdContacts {
  Span<int> points;
  Span<float> radii;
  Span<float3> positions_on_face;
  Span<float3> collider_motion;
  Span<float3> collider_velocities;
  Span<float3> face_normals;
  Span<float> margins;
  Span<float> static_frictions;
  Span<float> dynamic_frictions;
  Span<float> compliance_terms;
  Span<float> error_scales;
  /** Seeds of the normal multiplier, carried over from the previous substep like on the host. */
  Span<float> lambdas_normal;
  Span<float> lambdas_friction;
};

/**
 * Device side of one fluid: positions, velocities and the lagrange multipliers of the current
 * substep. `HaXpbdFluid::bind` uploads the frame state, every `substep` advances the particles on
 * its own (grid, predict, constraint iterations, velocity update, velocity layer) and `download`
 * brings the result back.
 *
 * The math is a direct port of `GEO_xpbd_fluid.hh`; the comments there explain the individual
 * steps, here only the differences (all about parallelism) are noted.
 */
class HaXpbdFluid {
 public:
  HaXpbdFluid() = default;
  ~HaXpbdFluid();
  HaXpbdFluid(const HaXpbdFluid &) = delete;
  HaXpbdFluid &operator=(const HaXpbdFluid &) = delete;

  /**
   * Uploads the frame state. \a support_radius is the support radius of the kernel (`2 * radius`),
   * \a delta_time the length of one substep.
   */
  bool bind(const HaXpbdFluidInputs &inputs,
            float support_radius,
            float delta_time,
            float compliance_term,
            float artificial_pressure,
            float max_correction);

  /**
   * Called by #substep right after the predict pass, when the positions of this substep are on the
   * device. The host downloads them, gathers the collider contacts and uploads them with
   * #set_contacts. Returning false means "no contacts this substep".
   */
  using ContactHook = std::function<bool()>;

  /** Advances the bound fluid by one substep, running \a iterations density iterations. */
  bool substep(int iterations, const ContactHook &contact_hook = {});

  /**
   * The same substep, split in its three phases, for callers that have to interleave their own
   * passes (the solver's whole world loop, which applies the reaction of the pressure to the
   * vertices of the coupled solids between the iterations):
   *
   *   begin_substep(hook) -> iterate() * iterations -> finish_substep()
   *
   * #begin_substep builds the grids, predicts the positions and runs \a contact_hook;
   * #finish_substep runs the velocity update, the velocity layer and the contact friction.
   * #substep is exactly these three in order.
   */
  bool begin_substep(const ContactHook &contact_hook = {});
  bool iterate();
  bool finish_substep();

  /** Uploads the contacts gathered on the host for the current substep. */
  bool set_contacts(const HaXpbdContacts &contacts);

  /** Downloads the normal/friction multipliers so the host can seed the next substep with them. */
  bool download_contact_lambdas(const MutableSpan<float> r_lambdas_normal,
                                const MutableSpan<float> r_lambdas_friction) const;

  /** Copies the current positions to the host without touching the velocities. */
  bool download_positions(const MutableSpan<float3> r_positions);

  /** Copies positions and velocities back to the host. */
  bool download(const MutableSpan<float3> r_positions, const MutableSpan<float3> r_velocities);

  bool is_ready() const
  {
    return ready_;
  }

  int particle_count() const
  {
    return particle_count_;
  }

  const HaXpbdGrid &grid() const
  {
    return grid_;
  }

  /** Why the last #bind or #substep failed, for the debug log. */
  const char *last_error() const
  {
    return last_error_;
  }

  const HaXpbdFluidStats &stats() const
  {
    return stats_;
  }

  /** Timing of everything the last frame did (reset by #bind). */
  const HaXpbdFluidFrameStats &frame_stats() const
  {
    return frame_stats_;
  }

  void free();

 private:
  int particle_count_ = 0;
  bool ready_ = false;
  float delta_time_ = 0.0f;
  float support_radius_ = 0.0f;
  HaXpbdFluidStats stats_;
  HaXpbdFluidFrameStats frame_stats_;
  const char *last_error_ = "";

  HaXpbdGrid grid_;

  /** Packed per particle properties, see #HA_PROP_STRIDE. */
  HaBuffer props_;
  /**
   * Packed per particle solver state, #HA_STATE_STRIDE floats per particle: the lagrange
   * multiplier of the density constraint, the squared and the signed relative density error, the
   * solid fraction (share of the neighborhood that is boundary), the position correction and the
   * vorticity magnitude. One buffer instead of two keeps the density pass at nine storage buffers.
   */
  HaBuffer particle_state_;
  HaBuffer velocities_;
  HaBuffer velocities_tmp_;
  HaBuffer prev_positions_;
  HaBuffer external_force_;
  /** xyz + mass, #HA_BOUNDARY_STRIDE floats per boundary particle. */
  HaBuffer boundary_particles_;
  /** Grid of the boundary particles; its positions are only used for the cell of each particle. */
  HaXpbdGrid boundary_grid_;
  /** CSR list of the boundary particles around each *fluid* particle. */
  HaBufferInt boundary_offsets_;
  HaBufferInt boundary_indices_;
  HaBufferInt boundary_counts_;
  int boundary_count_ = 0;
  int boundary_capacity_ = 0;
  /** Per frame constants, see the `FluidParamIndex` enum in the implementation. */
  HaBuffer params_;

  /* Collider contacts of the current substep, packed as one vector buffer (face position, normal,
   * collider motion: 9 floats) and one scalar buffer (margin, compliance, static friction, dynamic
   * friction, error scale, radius: 6 floats). */
  int contact_count_ = 0;
  HaBuffer contact_vec_;
  HaBuffer contact_scalar_;
  HaBufferInt contact_points_;
  /** Two floats per contact: the normal multiplier and the friction multiplier. */
  HaBuffer contact_lambda_;

  /** Builds the CSR list of the boundary particles around every fluid particle. */
  bool build_boundary_neighbors(int pass_offset);

  /** Alternates between #velocities_ and #velocities_tmp_ (XSPH needs a ping pong buffer). */
  bool velocities_in_tmp_ = false;

  HaBuffer &velocity_current()
  {
    return velocities_in_tmp_ ? velocities_tmp_ : velocities_;
  }
  HaBuffer &velocity_other()
  {
    return velocities_in_tmp_ ? velocities_ : velocities_tmp_;
  }
};

/** \} */

}  // namespace blender::geometry::ha
