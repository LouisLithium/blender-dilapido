/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * White water on top of a solved fluid: foam that sticks to the surface, ballistic spray and
 * buoyant bubbles, with a cheap neighbour model so the particles can clump (cohesion), push each
 * other apart (separation) and split when the flow tears a clump apart.
 *
 * This is a one way post process: the fluid drives the white water, never the other way around.
 * Everything is CPU and deterministic (fixed neighbour order, Jacobi accumulation, no atomics).
 */

#pragma once

#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_vector.hh"

namespace blender {
struct Mesh;
}  // namespace blender

namespace blender::geometry::whitewater {

enum class ParticleType : int {
  /** Sticks to the fluid surface and drifts with it. */
  Foam = 0,
  /** Ballistic droplet: gravity and air drag, dies on impact or after its lifetime. */
  Spray = 1,
  /** Underwater and buoyant, turns into foam when it reaches the surface. */
  Bubble = 2,
};

enum class ClusterMode : int {
  /**
   * Clusters emerge from the neighbour links: a link carries cohesion while its relative normal
   * speed stays below `break_strain`, so a clump that is torn apart simply stops being connected.
   * A Houdini style explicit cluster mode (with per cluster lifetime and stickiness) is planned as
   * a second entry of the same menu.
   */
  Strain = 0,
};

struct Settings {
  bool foam = true;
  bool spray = true;
  bool bubbles = false;

  /** Emission in points per square metre per second, scaled by the frame time. */
  float foam_emission = 40.0f;
  float spray_emission = 20.0f;
  float bubble_emission = 8.0f;
  float emission_scale = 1.0f;

  /** Speed window that drives the emission density of each type (m/s). */
  float foam_threshold = 1.5f;
  float foam_full_speed = 5.0f;
  float spray_threshold = 2.5f;
  float spray_full_speed = 6.0f;
  float bubble_threshold = 0.5f;
  float bubble_full_speed = 2.0f;

  float foam_lifetime = 4.0f;
  float spray_lifetime = 1.2f;
  float bubble_lifetime = 3.0f;
  float foam_size = 0.02f;
  float spray_size = 0.015f;
  float bubble_size = 0.01f;

  /** Blend of a foam particle's velocity towards the fluid velocity (0 = static, 1 = rides). */
  float foam_stick = 0.7f;
  /** Extra speed along the surface normal when a spray particle is born (m/s). */
  float spray_speed = 1.5f;
  float bubble_buoyancy = 2.0f;
  float bubble_depth = 0.15f;
  bool bubble_to_foam = true;

  float3 gravity = float3(0.0f, 0.0f, -9.81f);
  /** Air drag of spray and bubbles (1/s). */
  float drag = 1.0f;

  ClusterMode cluster_mode = ClusterMode::Strain;
  /** Attraction at contact (m/s^2), linear falloff to `cohesion_radius`. */
  float cohesion = 2.0f;
  float cohesion_radius = 0.05f;
  /** Repulsion at overlap (m/s^2), linear falloff to `separation_radius`. */
  float separation = 40.0f;
  float separation_radius = 0.02f;
  bool same_type_only = true;
  /** Links break above this relative speed divided by the cohesion radius (1/s). */
  float break_strain = 8.0f;
  int max_cluster_size = 64;
  /** Relative velocity damping between linked neighbours (1/s). */
  float cluster_damping = 0.5f;

  int max_particles = 50000;

  bool use_bounds = false;
  float3 bounds_min = float3(-100.0f);
  float3 bounds_max = float3(100.0f);

  int seed = 0;
  /** Scene frame, only used to vary the emission sampling between frames. */
  int frame = 0;
};

struct Stats {
  int particles = 0;
  int foam = 0;
  int spray = 0;
  int bubbles = 0;
  int emitted = 0;
  int clusters = 0;
  int links = 0;
  float max_speed = 0.0f;
  double milliseconds = 0.0;
};

/** Flat particle state; the layout is what the node writes into the point cloud. */
struct ParticleData {
  Vector<float3> positions;
  Vector<float3> velocities;
  Vector<float> ages;
  Vector<float> lifetimes;
  Vector<float> radii;
  Vector<int> types;
  Vector<int> clusters;

  int size() const
  {
    return int(positions.size());
  }

  void clear();
  void reserve(int count);
  void append(const float3 &position,
              const float3 &velocity,
              float age,
              float lifetime,
              float radius,
              int type);

  /** Drop the `count` oldest particles (the front of the arrays). */
  void remove_front(int count);
};

/**
 * Advance the white water by one frame.
 *
 * \param particles: the state of the previous frame, replaced by this frame's state.
 * \param surface: fluid surface mesh, used for emission and foam projection (may be null).
 * \param fluid_positions: fluid particles, used to sample the flow velocity (may be empty).
 * \param fluid_velocities: same length as `fluid_positions`.
 * \param collider_triangles: flat triangle list (3 positions per triangle), spray dies inside it.
 * \param settings: all parameters of the node.
 * \param r_stats: counts and timing for the `Stats` output.
 */
void solve_frame(ParticleData &particles,
                 const Mesh *surface,
                 Span<const float3> fluid_positions,
                 Span<const float3> fluid_velocities,
                 Span<const float3> collider_triangles,
                 const Settings &settings,
                 float delta_time,
                 Stats &r_stats);

}  // namespace blender::geometry::whitewater
