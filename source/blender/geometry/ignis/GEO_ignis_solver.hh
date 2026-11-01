/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Sparse smoke and fire solver (`Ignis Solver`).
 *
 * The solver is Eulerian: the state is a set of sparse OpenVDB grids (*density*, *temperature*,
 * *fuel*, *flame* and a *velocity* vector grid) that live in a `Volume` datablock. That volume is
 * the only state, so a simulation zone can keep it in its cache exactly like the XPBD solver keeps
 * its world bundle, and nothing is hidden inside the node.
 *
 * "Sparse" comes from using OpenVDB trees for the storage: only the voxels that carry smoke are
 * allocated, iterating a grid only visits those, and the bounding box is rebuilt from the active
 * voxels every frame. There is no dense fallback.
 *
 * The numerics are the usual staggered-free grid solver:
 *  - collocated velocity, semi-Lagrangian advection,
 *  - divergence with forward differences and the pressure gradient with backward differences, which
 *    is the same 2h wide 7 point Laplacian as a MAC grid but without coupling the velocity
 *    components into each other through an odd/even checkerboard mode,
 *  - a sparse preconditioned conjugate gradient on the active voxels for the pressure projection,
 *  - buoyancy from the temperature, vorticity confinement, dissipation, cooling and a simple
 *    fuel/temperature combustion model.
 */

#pragma once

#include "BLI_array.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

namespace blender {

struct Mesh;
struct Volume;

namespace geometry::ignis {

/** Names of the grids inside the state volume. Rendering a volume in Cycles picks `density`. */
namespace grids {
constexpr StringRefNull density = "density";
constexpr StringRefNull temperature = "temperature";
constexpr StringRefNull fuel = "fuel";
constexpr StringRefNull flame = "flame";
constexpr StringRefNull velocity = "velocity";
constexpr StringRefNull pressure = "pressure";
constexpr StringRefNull divergence = "divergence";
}  // namespace grids

/** Matches `openvdb::tools::Scheme`: the integrators the official `Grid Advect` node exposes. */
enum class AdvectionScheme : int {
  SemiLagrangian = 0,
  Midpoint = 1,
  MacCormack = 2,
  Bfecc = 3,
};

enum class AdvectionLimiter : int {
  None = 0,
  Clamp = 1,
  Revert = 2,
};

struct Settings {
  /* Domain. */
  float voxel_size = 0.05f;
  bool use_bounds = false;
  float3 bounds_min = float3(0.0f);
  float3 bounds_max = float3(1.0f);


  /* Stepping. */
  int substeps = 1;
  int max_substeps = 16;
  float cfl = 2.0f;

  /* Forces and transport. */
  float3 gravity = float3(0.0f, 0.0f, -9.81f);
  float3 wind = float3(0.0f);
  /** Acceleration per unit of temperature anomaly, in units of gravity. */
  float buoyancy = 1.0f;
  /** The temperature field is measured against this; 0 means the field is the anomaly itself. */
  float ambient_temperature = 0.0f;
  float vorticity = 0.0f;
  float drag = 0.0f;
  float cooling = 0.0f;
  float dissipation = 0.0f;
  float velocity_damping = 0.0f;
  int extrapolate = 4;
  /** Voxels of air around the smoke that take part in the pressure solve. A thin band cannot hold the
   * return flow of a rising plume and drags the smoke back down (see M66). */
  int domain_padding = 6;
  AdvectionScheme advection_scheme = AdvectionScheme::MacCormack;
  AdvectionLimiter advection_limiter = AdvectionLimiter::Clamp;

  /* Combustion. */
  float burn_rate = 0.0f;
  float heat_release = 0.0f;
  float smoke_per_burn = 1.0f;
  float flame_threshold = 0.05f;

  /* Projection. */
  int pressure_iterations = 40;
  float tolerance = 1e-3f;
  int threads = 1;
};

struct Stats {
  int active_voxels = 0;
  int substeps_used = 0;
  int pressure_iterations = 0;
  float residual = 0.0f;
  /** L2 norm of the velocity divergence before and after the projection, in 1/s. */
  float divergence_before = 0.0f;
  float divergence_after = 0.0f;
  /** Voxels of the frame that ended up inside a collider and were removed. */
  int solid_voxels = 0;
  /** Emitters that contributed to the frame (every volume of the source geometry). */
  int sources = 0;
  bool cfl_limited = false;
  bool resized = false;
  double milliseconds = 0.0;
};

/**
 * A collider of the frame: a mesh in world space.
 *
 * The solver rasterizes the surface into a thin solid shell and treats it as a wall: the smoke is
 * removed where it would be inside, the velocity may not point into it and the pressure is zero on
 * it. The mesh is expected to be closed for a volumetric obstacle; an open mesh (a ground plane)
 * works as a thin wall of one voxel.
 *
 * \note `Dilapido Collider` bakes its `Transform` into the geometry, so the mesh arrives here already
 * posed. A collider that moves more than one voxel per frame can be jumped over, because the surface
 * is sampled once per frame (the limitation the PBD solver had before it learned to sweep).
 */
struct Collider {
  const Mesh *mesh = nullptr;
};

/**
 * Advance the solver by one frame.
 *
 * \param previous: the state of the previous frame, or null on the first frame.
 * \param sources: the rasterized emitters of this frame (the volumes of `Init Ignis Data`), sampled
 *                 at the voxel centers of the state, so a source and the solver may use different
 *                 resolutions. They are added one after the other: the scalar fields keep the
 *                 larger magnitude per voxel, so several emitters of different strength and
 *                 temperature mix instead of overwriting each other. Optional.
 * \param colliders: the meshes the smoke cannot go through, already posed in world space.
 * \param r_stats: timing and diagnostics, also used to report a clamped CFL.
 * \return the new state (a new #Volume, the caller owns it) or null when there is nothing to do.
 */
Volume *solve_frame(const Volume *previous,
                    Span<const Volume *> sources,
                    Span<const Collider> colliders,
                    const Settings &settings,
                    float delta_time,
                    Stats &r_stats);

}  // namespace geometry::ignis
}  // namespace blender
