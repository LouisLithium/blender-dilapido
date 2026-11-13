/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * `Textura Solver`: a Material Point Method (MPM) solver for low rigidity materials, i.e. fluids
 * and granular media such as sand.
 *
 * The solver is deliberately CPU only for now, but every buffer is a flat, index addressed
 * Structure-of-Arrays allocation. There are no pointers between particles and no per component
 * bookkeeping inside the solver, so a future device backend can upload the arrays as storage
 * buffers and run the same five passes (particle to grid, grid update, grid to particle, material
 * update, advection) as compute kernels.
 *
 * Coordinate conventions
 * ----------------------
 * - The grid is a regular lattice. Node `(i, j, k)` is at `origin + float3(i, j, k) * dx`, i.e.
 *   `origin` is the corner of node `(0, 0, 0)` (not the corner of a cell).
 * - `ParticleData::deformation` stores the deformation gradient `F` row-major, nine floats per
 *   particle. The geometry attributes store it as three columns (`textura_F_x/_y/_z`), because a
 *   `float3` attribute is the only matrix-like attribute type in geometry nodes.
 */

#pragma once

#include "BLI_math_matrix_types.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

namespace blender {
struct Mesh;
}

namespace blender::geometry::textura {

/** Low rigidity materials the solver is built for. */
enum class MaterialType : int {
  Fluid = 0,
  Sand = 1,
  Snow = 2,
};

namespace attribute_names {
constexpr StringRefNull type = "textura_type";
constexpr StringRefNull density = "textura_density";
constexpr StringRefNull volume = "textura_volume";
constexpr StringRefNull bulk_modulus = "textura_bulk_modulus";
constexpr StringRefNull shear_modulus = "textura_shear_modulus";
constexpr StringRefNull lame_lambda = "textura_lame_lambda";
constexpr StringRefNull viscosity = "textura_viscosity";
constexpr StringRefNull tension = "textura_tension";
/** Surface tension coefficient in N/m (water is 0.072). 0 disables the force. */
constexpr StringRefNull surface_tension = "textura_surface_tension";
constexpr StringRefNull friction = "textura_friction";
constexpr StringRefNull cohesion = "textura_cohesion";
/** Maximum tensile stress a snow particle transmits, in pascal. */
constexpr StringRefNull tensile_strength = "textura_tensile_strength";
/** Extra tangential friction against colliders (0..1), applied per particle. */
constexpr StringRefNull wall_friction = "textura_wall_friction";
/** Adhesion acceleration toward a nearby collider, in m/s^2. 0 disables it. */
constexpr StringRefNull adhesion = "textura_adhesion";
/** Maximum plastic compression, as a logarithmic volume strain (0.05 = 5%). */
constexpr StringRefNull compression_limit = "textura_compression_limit";
/** Maximum plastic stretch, as a logarithmic volume strain. */
constexpr StringRefNull stretch_limit = "textura_stretch_limit";
/** Rest density of a snow particle; used to cap how far it can compact. */
constexpr StringRefNull rest_density = "textura_rest_density";
/** How much the stiffness grows per unit of accumulated plastic strain. */
constexpr StringRefNull hardening_rate = "textura_hardening_rate";
/** Accumulated plastic strain of a snow particle (read only state). */
constexpr StringRefNull hardening = "textura_hardening";
/** Accumulated damage, 0 = intact, 1 = no shear strength left. Grows with plastic strain. */
constexpr StringRefNull damage = "textura_damage";
/** How much damage one unit of plastic strain adds. 0 disables damage. */
constexpr StringRefNull damage_rate = "textura_damage_rate";
constexpr StringRefNull damping = "textura_damping";
/** Emission mode written by `Init Textura Data`: 0 none, 1 per frame, 2 per substep. The asset
 * re-injects the emitter's particles while this is not 0. */
constexpr StringRefNull emission = "textura_emission";
/** Minimum distance between a newly emitted particle and the existing pool, in metres. 0 emits
 * everything, which lets a static emitter stack copies on top of each other. */
constexpr StringRefNull emission_spacing = "textura_emission_spacing";
/** Relative random variation of the emitted velocity (0..1). */
constexpr StringRefNull emission_velocity_jitter = "textura_emission_velocity_jitter";
constexpr StringRefNull deformation_x = "textura_F_x";
constexpr StringRefNull deformation_y = "textura_F_y";
constexpr StringRefNull deformation_z = "textura_F_z";
/** Plastic part of the deformation gradient, snow only. `F = F_elastic * F_plastic`. */
constexpr StringRefNull plastic_x = "textura_Fp_x";
constexpr StringRefNull plastic_y = "textura_Fp_y";
constexpr StringRefNull plastic_z = "textura_Fp_z";
constexpr StringRefNull rest_position = "textura_rest_position";
}  // namespace attribute_names

/** Parameters that apply to the whole solve, read from `Blender.TexturaSettings`. */
struct Settings {
  /** Grid spacing in metres. `0` derives it from the particle radius. */
  float voxel_size = 0.0f;
  /** Minimum number of substeps per frame. */
  int min_substeps = 8;
  /** Upper bound for the substeps the CFL condition asks for. */
  int max_substeps = 64;
  /** How many cells a particle may travel in one substep. */
  float cfl = 0.5f;
  bool use_bounds = false;
  float3 bounds_min = float3(-1.0f, -1.0f, 0.0f);
  float3 bounds_max = float3(1.0f, 1.0f, 2.0f);
  float3 gravity = float3(0.0f, 0.0f, -9.81f);
  /** Velocity damping of the grid, in 1/s. */
  float velocity_damping = 0.1f;
  /** Tangential velocity kept when a particle or grid node touches a collider. The coefficient is
   * dimensionless: 0 slips freely, 1 removes the tangential velocity within a few frames. It is
   * applied as a rate (per second), not per substep, so the result does not depend on `Substeps`. */
  float boundary_friction = 0.5f;
  /**
   * Width of the particle level contact projection, in grid cells. The MPM grid holds the lowest
   * layer about one to one and a half cells above a collider; the solver projects particles within
   * `radius + contact_skin * dx` down to their radius so the material visually touches the surface.
   * A cohesive sheet (snow) can hover a bit higher than that band, so raise this to 1.5 or 2 to
   * close the remaining gap (at the cost of pulling a second layer down as well).
   */
  float contact_skin = 1.0f;
  /**
   * Give fluids and sand their own velocity grid and exchange momentum between them. Without this
   * both materials share one grid velocity, so they cannot slide past each other and a fluid in
   * contact with sand looks viscous.
   */
  bool two_phase = false;
  /** Momentum exchange rate between the two phase grids, in 1/s. 0 lets them move independently. */
  float inter_phase_drag = 20.0f;
  /** Safety cap. When the box needs more cells the spacing is increased instead. */
  int max_grid_cells = 4 << 20;
};

/**
 * One collider.
 *
 * `Dilapido Collider` bakes its transform into the geometry, so the mesh is already in world
 * space and no matrix is needed here. Box colliders are solid, mesh colliders are shells whose
 * signed distance is built with `geometry::ha::ha_sdf_build()`.
 */
struct Collider {
  const Mesh *mesh = nullptr;
  bool box = false;
  /** True for `Dilapido Collider > Type = Convex Hull`; the mesh is used as the hull surface. */
  bool convex = false;
  /**
   * Treat a closed collider as a hollow container: the field becomes `|distance|`, so the surface
   * is a wall on both sides and the interior is free space. Ignored for open sheets, which are
   * already membranes.
   */
  bool invert = false;
  /** Coulomb friction of the surface (0..1), from the `Dilapido Collider` node. */
  float friction = 0.5f;
  /**
   * Oriented box frame. `Dilapido Collider` bakes its `Transform` into the geometry, so a box is
   * rebuilt in the object's local space: the world vertices are mapped back with
   * `box_transform_inverse`, the local extent becomes `box_local_min/max`, and the analytic box is
   * sampled there. Without a transform the matrix is the identity and this is the world AABB.
   */
  float4x4 box_transform = float4x4::identity();
  float4x4 box_transform_inverse = float4x4::identity();
  /** Average world scale of #box_transform, so a local box distance becomes world units. */
  float box_scale = 1.0f;
  float3 box_local_min = float3(0.0f);
  float3 box_local_max = float3(-1.0f);
};

/**
 * Flat particle storage.
 *
 * Every array has one entry per particle, and matrices are stored as nine consecutive floats.
 * This is the layout a storage buffer wants: no indirection, fixed stride, no per geometry
 * component state. `affine` is the APIC velocity gradient and is only valid inside one substep.
 */
struct ParticleData {
  Vector<float3> positions;
  Vector<float3> velocities;
  /** APIC velocity gradient `C`, row-major, nine floats per particle. */
  Vector<float> affine;
  /** Deformation gradient `F`, row-major, nine floats per particle. */
  Vector<float> deformation;
  /** Snow: plastic part `F_p` of `F = F_e F_p`, row-major. Identity for other materials. */
  Vector<float> plastic;

  Vector<int> material;
  Vector<float> density;
  Vector<float> volume;
  Vector<float> mass;
  Vector<float> bulk_modulus;
  Vector<float> shear_modulus;
  Vector<float> lame_lambda;
  Vector<float> viscosity;
  /** How much of a tensile (negative) pressure a fluid transmits. 0 = water, 1 = fully elastic. */
  Vector<float> tension;
  /** Surface tension coefficient, in N/m. */
  Vector<float> surface_tension;
  Vector<float> friction;
  Vector<float> cohesion;
  /** Snow: extra collider friction (0..1). */
  Vector<float> wall_friction;
  /** Snow: adhesion acceleration toward a collider, in m/s^2. */
  Vector<float> adhesion;
  /** Snow: tensile strength in pascal. */
  Vector<float> tensile_strength;
  /** Snow: plastic limits on the logarithmic volume strain. */
  Vector<float> compression_limit;
  Vector<float> stretch_limit;
  /** Snow: rest density, used to cap the plastic compaction. */
  Vector<float> rest_density;
  /** Snow: stiffness growth per unit of accumulated plastic strain. */
  Vector<float> hardening_rate;
  /** Snow: accumulated plastic strain (state, written back to the geometry). */
  Vector<float> hardening;
  /**
   * Damage in 0..1. Plastic strain accumulates it and it removes the shear modulus, the cohesion
   * and the internal friction, so a loaded sand block softens along its shear bands and breaks
   * into loose sand while the intact regions keep their shape.
   */
  Vector<float> damage;
  Vector<float> damage_rate;
  Vector<float> damping;
  Vector<float> radius;

  int size() const
  {
    return int(positions.size());
  }

  void resize(const int count);
  void fill_defaults();
};

/**
 * Dense grid.
 *
 * A sparse brick pool can replace the dense arrays later without touching the particle side: the
 * passes only use `index(x, y, z)` and `dims`, so the addressing is the single place to change
 * (the same abstraction the Ignis device solver uses).
 */
struct GridData {
  int3 dims = int3(0);
  /** World position of node `(0, 0, 0)`. */
  float3 origin = float3(0.0f);
  float dx = 0.0f;

  Vector<float> mass;
  Vector<float3> momentum;
  Vector<float3> velocity;
  /**
   * Smoothed colour field (1 inside the fluid, 0 outside), used by the surface tension force.
   * Only filled when at least one particle has a non zero surface tension coefficient.
   */
  Vector<float> color;
  /**
   * Indices of the cells that P2G touched in the current substep, in particle order. `clear` and
   * `update_grid` only visit these instead of the whole box, which matters because the domain is
   * an AABB and can be mostly empty.
   */
  Vector<int> active;

  int cell_count() const
  {
    return dims.x * dims.y * dims.z;
  }

  int index(const int x, const int y, const int z) const
  {
    return (z * dims.y + y) * dims.x + x;
  }

  bool contains(const int x, const int y, const int z) const
  {
    return x >= 0 && y >= 0 && z >= 0 && x < dims.x && y < dims.y && z < dims.z;
  }

  /** Allocates and zeroes the whole grid. Used once per frame when the domain is built. */
  void clear();

  /** Zeroes the cells of the previous substep and forgets the list. */
  void clear_active();
};

struct Stats {
  int particles = 0;
  int grid_cells = 0;
  int active_cells = 0;
  int substeps = 0;
  float max_speed = 0.0f;
  /** Largest `det(F)` at the end of the frame; a fluid should stay close to 1. */
  float max_volume_ratio = 1.0f;
  /** Smallest `det(F)` at the end of the frame. */
  float min_volume_ratio = 1.0f;
  int3 grid_dims = int3(0);
  bool cfl_limited = false;
  bool grid_limited = false;
  bool collider_sdf_failed = false;
  bool has_particles = false;
  /** Fluids and sand used separate velocity grids for this frame. */
  bool two_phase = false;
  double milliseconds = 0.0;
  /** Per pass breakdown, accumulated over all substeps of the frame. */
  double milliseconds_clear = 0.0;
  double milliseconds_particle_to_grid = 0.0;
  double milliseconds_update_grid = 0.0;
  double milliseconds_grid_to_particle = 0.0;
  double milliseconds_advect = 0.0;
};

/**
 * Advances the particles by `delta_time` seconds.
 *
 * The particle arrays are updated in place (positions, velocities and the deformation gradient).
 * Colliders are only used during the solve; nothing is written to them.
 */
void solve_frame(ParticleData &particles,
                 Span<const Collider> colliders,
                 const Settings &settings,
                 float delta_time,
                 Stats &r_stats,
                 bool device_requested = false,
                 bool *r_device_used = nullptr,
                 std::string *r_device_reason = nullptr);

}  // namespace blender::geometry::textura
