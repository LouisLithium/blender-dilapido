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

/** Attribute names written by `Ignis Source` and read by the solver. */
namespace attributes {
constexpr StringRefNull density = "ignis_density";
constexpr StringRefNull temperature = "ignis_temperature";
constexpr StringRefNull fuel = "ignis_fuel";
constexpr StringRefNull velocity = "ignis_velocity";
constexpr StringRefNull radius = "ignis_radius";
constexpr StringRefNull surface = "ignis_surface";
}  // namespace attributes

/** The emitters of one frame, gathered from the attributes of `Ignis Source`. */
struct SourceData {
  Span<float3> positions;
  Span<float3> velocity;
  Span<float> density;
  Span<float> temperature;
  Span<float> fuel;
  Span<float> radius;
  /** Sample the triangles below instead of the points (a burning surface rather than sparks). */
  bool sample_surface = false;
  Span<int3> corner_tris;
  Span<int> corner_verts;
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
  float buoyancy = 2.0f;
  float temperature_scale = 1.0f;
  float vorticity = 0.0f;
  float drag = 0.0f;
  float cooling = 0.0f;
  float dissipation = 0.0f;
  float velocity_damping = 0.0f;
  int extrapolate = 4;

  /* Combustion. */
  float burn_rate = 0.0f;
  float heat_release = 0.0f;
  float smoke_per_burn = 1.0f;
  float flame_threshold = 0.05f;

  /* Projection. */
  int pressure_iterations = 40;
  float tolerance = 1e-4f;
  int threads = 1;

  /* Output. */
  bool make_surface = true;
  float iso_level = 0.5f;
  float adaptivity = 0.0f;
};

struct Stats {
  int active_voxels = 0;
  int substeps_used = 0;
  int pressure_iterations = 0;
  float residual = 0.0f;
  bool cfl_limited = false;
  bool resized = false;
  double milliseconds = 0.0;
};

/**
 * Advance the solver by one frame.
 *
 * \param previous: the state of the previous frame, or null on the first frame.
 * \param sources: the emitters of this frame.
 * \param r_stats: timing and diagnostics, also used to report a clamped CFL.
 * \return the new state (a new #Volume, the caller owns it) or null when there is nothing to do.
 * The density isosurface is written to \a r_surface when `Settings::make_surface` is set.
 */
Volume *solve_frame(const Volume *previous,
                    Span<const SourceData> sources,
                    const Settings &settings,
                    float delta_time,
                    Mesh **r_surface,
                    Stats &r_stats);

}  // namespace geometry::ignis
}  // namespace blender
