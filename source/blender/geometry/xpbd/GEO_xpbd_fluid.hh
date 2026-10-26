/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>

#include "BLI_array.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_task.hh"

namespace blender::xpbd {

/**
 * Position based fluid, following Macklin & Müller, "Position Based Fluids", 2013.
 *
 * A fluid is completely described by a particle cloud with a radius and a mass per particle. The
 * density constraint is not a pairwise constraint: it touches all neighbors of a particle at once
 * and the positional correction of a particle is the accumulation of the contributions of all its
 * neighbors. That is why the fluid is a separate solve pass instead of a #ConstraintSet.
 *
 * The pass is split into three independent stages so that every stage can run in parallel
 * (Jacobi style), which also makes the result independent of the thread scheduling:
 *   1. Compute the density constraint and its lagrange multiplier for every particle.
 *   2. Accumulate the positional correction for every particle from the multipliers.
 *   3. Apply the corrections.
 */

/** Scalar kernels for a support radius #h. */
class FluidKernel {
 private:
  float h_ = 0.0f;
  float h_sq_ = 0.0f;
  float poly6_scale_ = 0.0f;
  float spiky_scale_ = 0.0f;

 public:
  FluidKernel() = default;

  explicit FluidKernel(const float h)
  {
    /* Local constant instead of M_PI to keep this header self contained. */
    constexpr float pi = 3.14159265358979323846f;
    h_ = h;
    h_sq_ = h * h;
    poly6_scale_ = (315.0f / (64.0f * pi)) / (h_sq_ * h_sq_ * h_sq_ * h_sq_ * h);
    spiky_scale_ = (45.0f / pi) / (h_sq_ * h_sq_ * h_sq_);
  }

  float h() const
  {
    return h_;
  }

  float h_sq() const
  {
    return h_sq_;
  }

  /** Poly6 kernel, used to estimate the density. */
  float poly6(const float r_sq) const
  {
    if (r_sq >= h_sq_) {
      return 0.0f;
    }
    const float diff = h_sq_ - r_sq;
    return poly6_scale_ * diff * diff * diff;
  }

  /** Length of the spiky kernel gradient at distance #r. */
  float spiky_gradient_length(const float r) const
  {
    if (r >= h_) {
      return 0.0f;
    }
    const float diff = h_ - r;
    return spiky_scale_ * diff * diff;
  }

  /**
   * Gradient of the kernel for \a r_vec = position_i - position_j. The gradient points towards the
   * neighbor, i.e. it is the gradient w.r.t. the position of the particle the density is evaluated
   * for.
   */
  float3 spiky_gradient(const float3 &r_vec, const float r) const
  {
    if (r <= 1e-12f) {
      return float3(0.0f);
    }
    return (-spiky_gradient_length(r) / r) * r_vec;
  }
};

/**
 * Uniform hash grid over the particles. Used to find the neighbors of each particle within the
 * support radius of the fluid kernel.
 *
 * The grid has to be rebuilt whenever the particle positions changed significantly, i.e. once per
 * substep. It is not rebuilt between constraint iterations because particles only move a fraction
 * of the support radius per iteration.
 */
class FluidParticleGrid {
 private:
  float cell_size_ = 0.0f;
  /** 0 while the grid is empty. */
  int table_size_ = 0;
  Array<int3> cells_;
  /** First particle index of each hash bucket, plus the total count at the end. */
  Array<int> bucket_begin_;
  /** Particle indices grouped by hash bucket. */
  Array<int> sorted_;

  static uint32_t hash_cell(const int3 &cell)
  {
    return (uint32_t(cell.x) * 73856093u) ^ (uint32_t(cell.y) * 19349663u) ^
           (uint32_t(cell.z) * 83492791u);
  }

 public:
  bool is_empty() const
  {
    return table_size_ == 0;
  }

  float cell_size() const
  {
    return cell_size_;
  }

  void build(const Span<float3> positions, const float support_radius)
  {
    const int points_num = int(positions.size());
    cell_size_ = support_radius;
    if (points_num == 0 || support_radius <= 0.0f) {
      table_size_ = 0;
      cells_.reinitialize(0);
      sorted_.reinitialize(0);
      bucket_begin_.reinitialize(0);
      return;
    }

    /* A load factor of ~0.5 keeps the bucket lists short without wasting much memory. */
    int table_size = 1024;
    while (table_size < points_num * 2 && table_size < (1 << 22)) {
      table_size <<= 1;
    }
    table_size_ = table_size;

    cells_.reinitialize(points_num);
    sorted_.reinitialize(points_num);
    bucket_begin_.reinitialize(table_size + 1);
    bucket_begin_.fill(0);

    const float inv_cell_size = 1.0f / support_radius;
    for (const int i : IndexRange(points_num)) {
      const float3 &pos = positions[i];
      const int3 cell(int(std::floor(pos.x * inv_cell_size)),
                      int(std::floor(pos.y * inv_cell_size)),
                      int(std::floor(pos.z * inv_cell_size)));
      cells_[i] = cell;
      bucket_begin_[int(hash_cell(cell)) & (table_size - 1)] += 1;
    }

    /* Convert the bucket sizes into the start offsets of each bucket. */
    int offset = 0;
    for (const int bucket : IndexRange(table_size)) {
      const int count = bucket_begin_[bucket];
      bucket_begin_[bucket] = offset;
      offset += count;
    }
    bucket_begin_[table_size] = offset;

    /* Scatter into the sorted array. This consumes the start offsets, they are restored below. */
    for (const int i : IndexRange(points_num)) {
      const int bucket = int(hash_cell(cells_[i])) & (table_size - 1);
      sorted_[bucket_begin_[bucket]++] = i;
    }
    for (int bucket = table_size; bucket > 0; --bucket) {
      bucket_begin_[bucket] = bucket_begin_[bucket - 1];
    }
    bucket_begin_[0] = 0;
  }

  /**
   * Calls \a fn for every particle within the support radius of particle \a i (including \a i
   * itself) with the arguments (particle index, position_i - position_j, squared distance).
   *
   * The iteration order is deterministic. Hash collisions are filtered by comparing the actual
   * cell coordinates.
   */
  template<typename Fn>
  void foreach_neighbor(const int i, const Span<float3> positions, Fn &&fn) const
  {
    if (table_size_ == 0) {
      return;
    }
    const int mask = table_size_ - 1;
    const int3 cell = cells_[i];
    const float3 &pos_i = positions[i];
    const float r_sq_max = cell_size_ * cell_size_;
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int3 neighbor_cell(cell.x + dx, cell.y + dy, cell.z + dz);
          const int bucket = int(hash_cell(neighbor_cell)) & mask;
          const IndexRange bucket_range = IndexRange::from_begin_end(bucket_begin_[bucket],
                                                                     bucket_begin_[bucket + 1]);
          for (const int sorted_i : bucket_range) {
            const int j = sorted_[sorted_i];
            if (cells_[j] != neighbor_cell) {
              continue;
            }
            const float3 r_vec = pos_i - positions[j];
            const float r_sq = math::length_squared(r_vec);
            if (r_sq >= r_sq_max) {
              continue;
            }
            fn(j, r_vec, r_sq);
          }
        }
      }
    }
  }
};

struct FluidDensitySettings {
  /** Target density of the fluid in kg/m³. */
  float rest_density = 1000.0f;
  /** XPBD compliance of the density constraint, already divided by the time step squared. */
  float compliance_term = 0.0f;
  /**
   * How much negative density constraint is allowed, i.e. how strongly under-dense regions (like
   * the surface of the fluid) pull particles together. PBF would use 1.0 here, but the SPH density
   * estimate is only half of the rest density at the surface, which makes small particle counts
   * implode. 0 disables cohesion entirely, which is the stable default.
   */
  float negative_pressure_limit = 0.0f;
  /** Akinci et al. artificial pressure, counteracts particle clumping at the surface. */
  float artificial_pressure = 0.0f;
  /** Radius of the artificial pressure kernel relative to the support radius. */
  float artificial_pressure_radius_factor = 0.2f;
  /** Maximum length of the position correction per iteration, 0 means unlimited. */
  float max_correction = 0.0f;
};

/**
 * Static particles sampled on the solid boundary, following Akinci et al. 2012 ("Versatile
 * Rigid-Fluid Coupling for Incompressible SPH"). Without them the SPH density estimate of a fluid
 * particle at a wall is truncated and only half of the rest density is seen, which is what makes
 * the fluid stick to walls and gives it a wrong pressure at the boundary.
 *
 * The boundary particles take part in the density estimate and push the fluid away, but they are
 * never moved themselves (their multiplier is zero, so only `lambda` of the fluid particle
 * contributes to the correction).
 */
struct FluidBoundary {
  const FluidParticleGrid *grid = nullptr;
  Span<float3> positions;
  Span<float> masses;

  bool is_valid() const
  {
    return grid != nullptr && !grid->is_empty() && !positions.is_empty();
  }
};

/** Diagnostics of a single density iteration. */
struct FluidDensityStats {
  float average_error_squared = 0.0f;
  float max_abs_error = 0.0f;
  float max_correction = 0.0f;
  int max_neighbors = 0;
};

/**
 * Damping of the velocity near boundary particles, i.e. the no-slip condition at a wall. The
 * factor is the fraction of the kernel weight that comes from the boundary, so a particle in the
 * middle of the fluid is not affected at all.
 *
 * \param solid_fractions: Fraction of the density estimate of each particle that comes from the
 *   boundary (0 inside the fluid, up to 1 at a wall).
 */
inline void fluid_apply_boundary_friction(const MutableSpan<float3> velocities,
                                          const Span<float> solid_fractions,
                                          const float friction)
{
  if (friction <= 0.0f) {
    return;
  }
  const int points_num = int(velocities.size());
  threading::parallel_for(IndexRange(points_num), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      const float factor = 1.0f - friction * solid_fractions[i];
      if (factor < 1.0f) {
        velocities[i] *= std::max(factor, 0.0f);
      }
    }
  });
}

/**
 * Runs one density constraint iteration.
 *
 * \param positions: Read and written positions of the particles. The correction is applied at the
 *   very end so all particles are corrected based on the same state.
 * \param masses: Mass of each particle in kg.
 * \param inv_masses: Inverse mass, 0 for pinned particles which participate in the density
 *   estimate but are not moved.
 * \param r_lambdas: Accumulated lagrange multipliers, reset before the first iteration.
 * \param r_errors: Scratch buffer for the squared relative density error of each particle.
 * \param r_neighbor_counts: Scratch buffer for the number of neighbors of each particle.
 * \param r_corrections: Scratch buffer for the position corrections.
 * \param boundary: Optional boundary particles. They are part of the density estimate and push the
 *   fluid away, but they are never moved.
 */
inline FluidDensityStats fluid_solve_density_iteration(const FluidParticleGrid &grid,
                                                       const FluidKernel &kernel,
                                                       const MutableSpan<float3> positions,
                                                       const Span<float> masses,
                                                       const Span<float> inv_masses,
                                                       const FluidDensitySettings &settings,
                                                       const MutableSpan<float> r_lambdas,
                                                       const MutableSpan<float> r_errors,
                                                       const MutableSpan<int> r_neighbor_counts,
                                                       const MutableSpan<float3> r_corrections,
                                                       const MutableSpan<float> r_solid_fractions,
                                                       const FluidBoundary &boundary)
{
  const int points_num = int(positions.size());
  if (points_num == 0 || grid.is_empty() || settings.rest_density <= 0.0f) {
    return {};
  }

  const float inv_rest_density = math::safe_rcp(settings.rest_density);
  const float inv_rest_density_sq = inv_rest_density * inv_rest_density;
  const float artificial_pressure = settings.artificial_pressure;
  const float inv_artificial_pressure_reference = [&]() {
    if (artificial_pressure <= 0.0f) {
      return 0.0f;
    }
    const float radius = settings.artificial_pressure_radius_factor * kernel.h();
    return math::safe_rcp(kernel.poly6(radius * radius));
  }();

  /* Compute the density constraint and its lagrange multiplier for every particle. */
  threading::parallel_for(IndexRange(points_num), 256, [&](const IndexRange range) {
    for (const int i : range) {
      float density = 0.0f;
      int neighbors = 0;
      float3 sum_gradient = float3(0.0f);
      float sum_gradient_sq = 0.0f;
      grid.foreach_neighbor(i, positions, [&](const int j, const float3 &r_vec, const float r_sq) {
        density += masses[j] * kernel.poly6(r_sq);
        neighbors += 1;
        if (j == i) {
          return;
        }
        if (UNLIKELY(r_sq < 1e-12f)) {
          /* Coincident particles have no defined gradient direction. Dividing by the distance
           * would produce a NaN that spreads over the whole fluid within one iteration. */
          return;
        }
        const float r = math::sqrt(r_sq);
        const float gradient_length = kernel.spiky_gradient_length(r);
        sum_gradient += masses[j] * (-(gradient_length / r) * r_vec);
        sum_gradient_sq += math::square(masses[j] * gradient_length);
      });
      if (boundary.is_valid()) {
        /* Boundary particles are part of the density estimate and of the gradient w.r.t. the fluid
         * particle, but since they cannot move they are not part of the per neighbor gradient
         * terms of the denominator. */
        float boundary_density = 0.0f;
        boundary.grid->foreach_neighbor(
            i, boundary.positions, [&](const int j, const float3 &r_vec, const float r_sq) {
              boundary_density += boundary.masses[j] * kernel.poly6(r_sq);
              if (UNLIKELY(r_sq < 1e-12f)) {
                return;
              }
              const float r = math::sqrt(r_sq);
              sum_gradient += boundary.masses[j] *
                              (-(kernel.spiky_gradient_length(r) / r) * r_vec);
            });
        density += boundary_density;
        r_solid_fractions[i] = boundary_density * math::safe_rcp(density);
      }
      else {
        r_solid_fractions[i] = 0.0f;
      }

      const float constraint = density * inv_rest_density - 1.0f;
      /* Only compression is solved by default, see #FluidDensitySettings::negative_pressure_limit.
       */
      const float solved_constraint = std::max(constraint, -settings.negative_pressure_limit);
      r_errors[i] = math::square(solved_constraint);
      r_neighbor_counts[i] = neighbors;
      if (inv_masses[i] <= 0.0f) {
        /* Pinned particles are part of the density estimate but are not solved for. */
        r_lambdas[i] = 0.0f;
        continue;
      }
      const float denominator = (math::length_squared(sum_gradient) + sum_gradient_sq) *
                                    inv_rest_density_sq +
                                settings.compliance_term + 1e-12f;
      /* The multiplier is not accumulated over the iterations: the correction below uses the
       * multipliers of all neighbors, so reusing the accumulated value would apply the same
       * correction again in every iteration (the fluid would keep expanding / churning).
       * The density constraint is also unilateral: clamping the multiplier keeps the solver from
       * pulling particles together, which would make the surface of the fluid implode. */
      const float max_lambda = settings.negative_pressure_limit / denominator;
      r_lambdas[i] = math::min(-solved_constraint / denominator, max_lambda);
    }
  });

  /* Accumulate the positional correction of each particle from the multipliers of its neighbors.
   */
  threading::parallel_for(IndexRange(points_num), 256, [&](const IndexRange range) {
    for (const int i : range) {
      if (inv_masses[i] <= 0.0f) {
        r_corrections[i] = float3(0.0f);
        continue;
      }
      const float lambda_i = r_lambdas[i];
      float3 correction = float3(0.0f);
      grid.foreach_neighbor(i, positions, [&](const int j, const float3 &r_vec, const float r_sq) {
        if (j == i) {
          return;
        }
        if (UNLIKELY(r_sq < 1e-12f)) {
          return;
        }
        const float r = math::sqrt(r_sq);
        float weight = lambda_i + r_lambdas[j];
        if (artificial_pressure > 0.0f) {
          weight -= artificial_pressure *
                    powf(kernel.poly6(r_sq) * inv_artificial_pressure_reference, 4.0f);
        }
        correction += weight * masses[j] * (-(kernel.spiky_gradient_length(r) / r) * r_vec);
      });
      if (boundary.is_valid() && lambda_i != 0.0f) {
        /* A boundary particle never moves, so only the multiplier of the fluid particle pushes. */
        boundary.grid->foreach_neighbor(
            i, boundary.positions, [&](const int j, const float3 &r_vec, const float r_sq) {
              if (UNLIKELY(r_sq < 1e-12f)) {
                return;
              }
              const float r = math::sqrt(r_sq);
              correction += lambda_i * boundary.masses[j] *
                            (-(kernel.spiky_gradient_length(r) / r) * r_vec);
            });
      }
      correction *= inv_rest_density;
      if (settings.max_correction > 0.0f) {
        const float length = math::length(correction);
        if (length > settings.max_correction) {
          correction *= settings.max_correction / length;
        }
      }
      if (UNLIKELY(!(std::isfinite(correction.x) && std::isfinite(correction.y) &&
                     std::isfinite(correction.z))))
      {
        /* Never let a single degenerate particle poison the whole fluid. */
        correction = float3(0.0f);
      }
      r_corrections[i] = correction;
    }
  });

  FluidDensityStats stats;
  float error_sum = 0.0f;
  for (const int i : IndexRange(points_num)) {
    error_sum += r_errors[i];
    stats.max_abs_error = std::max(stats.max_abs_error, math::sqrt(r_errors[i]));
    stats.max_correction = std::max(stats.max_correction, math::length(r_corrections[i]));
    stats.max_neighbors = std::max(stats.max_neighbors, r_neighbor_counts[i]);
  }

  threading::parallel_for(IndexRange(points_num), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      if (inv_masses[i] <= 0.0f) {
        continue;
      }
      positions[i] += r_corrections[i];
    }
  });

  stats.average_error_squared = error_sum / points_num;
  return stats;
}

/**
 * XSPH viscosity (Monaghan 1989): smooths the velocities with a weighted average of the neighbor
 * velocities. This is what makes the fluid look like a liquid instead of a pile of independent
 * particles.
 *
 * \param r_scratch: Scratch buffer for the new velocities.
 */
inline void fluid_apply_xsph_viscosity(const FluidParticleGrid &grid,
                                       const FluidKernel &kernel,
                                       const Span<float3> positions,
                                       const Span<float> masses,
                                       const float rest_density,
                                       const float viscosity,
                                       const MutableSpan<float3> velocities,
                                       const MutableSpan<float3> r_scratch)
{
  const int points_num = int(velocities.size());
  if (points_num == 0 || grid.is_empty() || viscosity <= 0.0f || rest_density <= 0.0f) {
    return;
  }
  const float inv_rest_density = math::safe_rcp(rest_density);

  threading::parallel_for(IndexRange(points_num), 256, [&](const IndexRange range) {
    for (const int i : range) {
      const float3 velocity_i = velocities[i];
      float3 sum = float3(0.0f);
      grid.foreach_neighbor(
          i, positions, [&](const int j, const float3 & /*r_vec*/, const float r_sq) {
            if (j == i) {
              return;
            }
            sum += masses[j] * (velocities[j] - velocity_i) * kernel.poly6(r_sq);
          });
      r_scratch[i] = velocity_i + (viscosity * inv_rest_density) * sum;
    }
  });

  threading::parallel_for(IndexRange(points_num), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      velocities[i] = r_scratch[i];
    }
  });
}

/**
 * Velocity damping: `v /= 1 + damping * dt`. The implicit form is unconditionally stable, so the
 * damping factor does not have to be clamped when the time step or the damping gets large.
 *
 * This damps bulk motion (e.g. a whole column of fluid oscillating on the floor), which the XSPH
 * viscosity cannot do because neighbors move in phase and have no relative velocity.
 *
 * \param damping: Damping in 1/s, 0 disables it.
 */
inline void fluid_apply_velocity_damping(const float damping,
                                         const float delta_time,
                                         const MutableSpan<float3> velocities)
{
  if (damping <= 0.0f || delta_time <= 0.0f) {
    return;
  }
  const float factor = math::safe_rcp(1.0f + damping * delta_time);
  threading::parallel_for(IndexRange(int(velocities.size())), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      velocities[i] *= factor;
    }
  });
}

/**
 * Vorticity confinement (Fedkiw et al. 2001): the curl of the velocity is computed on the
 * particles and a force along `N x omega` re-injects the small scale rotation that numerical
 * dissipation removes. Without it, a swirling fluid calms down into a smooth blob.
 *
 * \param strength: Confinement strength, 0 disables it. Values around 0.5 to 2 keep the small
 * scale rotation alive; above ~4 the force injects too much energy.
 * \param r_omegas: Scratch buffer for the vorticity of each particle.
 * \param r_omega_magnitudes: Scratch buffer for `|omega|` of each particle.
 * \param r_scratch: Scratch buffer for the velocity change of each particle.
 */
inline void fluid_apply_vorticity_confinement(const FluidParticleGrid &grid,
                                              const FluidKernel &kernel,
                                              const Span<float3> positions,
                                              const Span<float> masses,
                                              const float rest_density,
                                              const float strength,
                                              const float delta_time,
                                              const MutableSpan<float3> velocities,
                                              const MutableSpan<float3> r_omegas,
                                              const MutableSpan<float> r_omega_magnitudes,
                                              const MutableSpan<float3> r_scratch)
{
  const int points_num = int(velocities.size());
  if (points_num == 0 || grid.is_empty() || strength <= 0.0f || delta_time <= 0.0f ||
      rest_density <= 0.0f)
  {
    return;
  }
  const float inv_rest_density = math::safe_rcp(rest_density);

  /* omega_i = curl(v) at the particle. */
  threading::parallel_for(IndexRange(points_num), 256, [&](const IndexRange range) {
    for (const int i : range) {
      const float3 velocity_i = velocities[i];
      float3 omega = float3(0.0f);
      grid.foreach_neighbor(i, positions, [&](const int j, const float3 &r_vec, const float r_sq) {
        if (j == i || r_sq < 1e-12f) {
          return;
        }
        const float r = math::sqrt(r_sq);
        omega += masses[j] *
                 math::cross(velocities[j] - velocity_i, kernel.spiky_gradient(r_vec, r));
      });
      omega *= inv_rest_density;
      r_omegas[i] = omega;
      r_omega_magnitudes[i] = math::length(omega);
    }
  });

  /* f = epsilon * h * (N x omega), with N being the normalized gradient of |omega|. */
  const float max_velocity_change = 0.1f * kernel.h() * math::safe_rcp(delta_time);
  threading::parallel_for(IndexRange(points_num), 256, [&](const IndexRange range) {
    for (const int i : range) {
      r_scratch[i] = float3(0.0f);
      const float magnitude_i = r_omega_magnitudes[i];
      float3 gradient = float3(0.0f);
      grid.foreach_neighbor(i, positions, [&](const int j, const float3 &r_vec, const float r_sq) {
        if (j == i || r_sq < 1e-12f) {
          return;
        }
        const float r = math::sqrt(r_sq);
        gradient += masses[j] * (r_omega_magnitudes[j] - magnitude_i) *
                    kernel.spiky_gradient(r_vec, r);
      });
      const float gradient_length = math::length(gradient);
      if (!(gradient_length > 1e-6f)) {
        continue;
      }
      const float3 direction = gradient / gradient_length;
      float3 change = (strength * kernel.h() * delta_time) * math::cross(direction, r_omegas[i]);
      const float change_length = math::length(change);
      if (change_length > max_velocity_change) {
        change *= max_velocity_change / change_length;
      }
      if (UNLIKELY(
              !(std::isfinite(change.x) && std::isfinite(change.y) && std::isfinite(change.z))))
      {
        change = float3(0.0f);
      }
      r_scratch[i] = change;
    }
  });

  threading::parallel_for(IndexRange(points_num), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      velocities[i] += r_scratch[i];
    }
  });
}

/**
 * Cohesion kernel for surface tension (Akinci et al. 2013, "Versatile Surface Tension and Adhesion
 * for SPH Fluids"). It is zero at distance 0 and at the support radius, so overlapping particles
 * are not glued together; only the surface layer is pulled inwards.
 */
inline float fluid_cohesion_kernel(const float r, const float h)
{
  if (r <= 0.0f || r >= h) {
    return 0.0f;
  }
  constexpr float pi = 3.14159265358979323846f;
  const float h_cubed = h * h * h;
  const float diff = h - r;
  return (32.0f / (pi * h_cubed * h_cubed * h_cubed)) * diff * diff * diff * r * r * r;
}

/**
 * Surface tension as a pairwise cohesion force between the particles. It pulls the surface layer
 * together, which makes small blobs round and keeps a puddle from breaking apart into droplets.
 *
 * \param strength: Cohesion strength, 0 disables it. The value depends on the particle mass and
 * the support radius; for the setups in the tests (9 cm particles, 1000 kg/m3) 0.2 to 2 is usable
 * and values above ~4 inject too much energy.
 * \param r_scratch: Scratch buffer for the velocity change of each particle.
 */
inline void fluid_apply_surface_tension(const FluidParticleGrid &grid,
                                        const Span<float3> positions,
                                        const Span<float> masses,
                                        const float h,
                                        const float strength,
                                        const float delta_time,
                                        const MutableSpan<float3> velocities,
                                        const MutableSpan<float3> r_scratch)
{
  const int points_num = int(velocities.size());
  if (points_num == 0 || grid.is_empty() || strength <= 0.0f || delta_time <= 0.0f) {
    return;
  }

  threading::parallel_for(IndexRange(points_num), 256, [&](const IndexRange range) {
    for (const int i : range) {
      const float3 &position_i = positions[i];
      float3 change = float3(0.0f);
      grid.foreach_neighbor(
          i, positions, [&](const int j, const float3 & /*r_vec*/, const float r_sq) {
            if (j == i || r_sq < 1e-12f) {
              return;
            }
            const float r = math::sqrt(r_sq);
            const float kernel = fluid_cohesion_kernel(r, h);
            if (kernel <= 0.0f) {
              return;
            }
            /* The attraction points towards the neighbor. */
            change += masses[j] * kernel * (positions[j] - position_i) * math::safe_rcp(r);
          });
      if (UNLIKELY(
              !(std::isfinite(change.x) && std::isfinite(change.y) && std::isfinite(change.z))))
      {
        change = float3(0.0f);
      }
      r_scratch[i] = (strength * delta_time) * change;
    }
  });

  threading::parallel_for(IndexRange(points_num), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      velocities[i] += r_scratch[i];
    }
  });
}

}  // namespace blender::xpbd
