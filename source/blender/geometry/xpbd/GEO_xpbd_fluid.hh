/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>

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
    this->foreach_neighbor_in_cell(cells_[i], positions[i], positions, fn);
  }

  /**
   * Same as #foreach_neighbor, but for a query point that is not part of the grid. Used to find
   * the particles around a boundary particle, which lives in a separate point set.
   *
   * \param positions: Positions of the particles the grid was built from.
   */
  template<typename Fn>
  void foreach_neighbor_at(const float3 &query, const Span<float3> positions, Fn &&fn) const
  {
    if (table_size_ == 0) {
      return;
    }
    const float inv_cell_size = 1.0f / cell_size_;
    const int3 cell(int(std::floor(query.x * inv_cell_size)),
                    int(std::floor(query.y * inv_cell_size)),
                    int(std::floor(query.z * inv_cell_size)));
    this->foreach_neighbor_in_cell(cell, query, positions, fn);
  }

 private:
  template<typename Fn>
  void foreach_neighbor_in_cell(const int3 &cell,
                                const float3 &query,
                                const Span<float3> positions,
                                Fn &&fn) const
  {
    const int mask = table_size_ - 1;
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
            const float3 r_vec = query - positions[j];
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

/**
 * Points per parallel task in the fluid passes.
 *
 * A fixed grain cannot serve both ends of the range: 64 points per task is what makes a 512 particle
 * fluid use every core, but with a million particles it would create fifteen thousand tasks and the
 * task overhead (and the per task scratch) would dominate. The grain follows the particle count and
 * the thread count: target about 64 tasks per thread, never below 64 points (below that the task
 * overhead wins) and never above 1024 (about 0.1 ms of work per task), so a million particle fluid
 * gets ~1000 tasks instead of 15000.
 *
 * `XPBD_FLUID_GRAIN` overrides it, for the benchmark.
 */
inline int fluid_task_grain(const int points_num)
{
  static const int fixed = []() {
    const char *env = std::getenv("XPBD_FLUID_GRAIN");
    return env ? std::atoi(env) : 0;
  }();
  if (fixed > 0) {
    return fixed;
  }
#ifdef WITH_TBB
  static const int threads = std::max(1, int(tbb::this_task_arena::max_concurrency()));
#else
  static const int threads = 1;
#endif
  /* Aim for ~64 tasks per thread: enough for the work stealing scheduler to balance uneven particle
   * distributions, while a task is still ~0.1 ms of work at the ceiling. Measured on a 32k point
   * fluid, a coarser grain (8 tasks per thread) was 30% slower. */
  const int target = points_num / std::max(1, threads * 64);
  return std::max(64, std::min(target, 1024));
}

/**
 * Neighbour lists of one point set, in CSR layout.
 *
 * The spatial grid is rebuilt once per substep and the particle set does not change between the
 * constraint iterations, but every pass used to walk the hash again (27 cells, a hash lookup per cell)
 * for every particle, sixteen times per substep. Building the index list once turns those passes into
 * flat array loops. Measured on a 512 particle fluid in a closed box the density iterations dropped
 * from 44 ms to 13 ms, because the boundary particles had doubled the number of hash walks.
 *
 * The *distances* are not cached: particles move during the iterations, so they are recomputed from
 * the current positions, exactly like #FluidParticleGrid::foreach_neighbor does.
 */
struct FluidNeighborList {
  /** Neighbours of particle `i` are `indices[offsets[i] .. offsets[i + 1])`. */
  Array<int> offsets;
  Array<int> indices;
  /** Squared support radius the list was built with; the passes still filter with it. */
  float r_sq_max = 0.0f;

  bool is_empty() const
  {
    return offsets.is_empty();
  }

  int size() const
  {
    return offsets.is_empty() ? 0 : int(offsets.size()) - 1;
  }

  template<typename Fn>
  void foreach_neighbor(const int i, const Span<float3> positions, Fn &&fn) const
  {
    const float3 &position_i = positions[i];
    for (int k = offsets[i]; k < offsets[i + 1]; k++) {
      const int j = indices[k];
      const float3 r_vec = position_i - positions[j];
      const float r_sq = math::length_squared(r_vec);
      if (r_sq >= r_sq_max) {
        continue;
      }
      fn(j, r_vec, r_sq);
    }
  }

  /** Same, for a query point that is not part of the list (fluid particle against boundary samples). */
  template<typename Fn>
  void foreach_neighbor_at(const int query_i,
                           const Span<float3> query_positions,
                           const Span<float3> positions,
                           Fn &&fn) const
  {
    const float3 &query = query_positions[query_i];
    for (int k = offsets[query_i]; k < offsets[query_i + 1]; k++) {
      const int j = indices[k];
      const float3 r_vec = query - positions[j];
      const float r_sq = math::length_squared(r_vec);
      if (r_sq >= r_sq_max) {
        continue;
      }
      fn(j, r_vec, r_sq);
    }
  }
};

/** Builds the neighbour list of a grid that was built from \a positions. */
inline void fluid_build_neighbors(const FluidParticleGrid &grid,
                                  const Span<float3> positions,
                                  FluidNeighborList &r_list)
{
  const int points_num = int(positions.size());
  r_list.r_sq_max = grid.cell_size() * grid.cell_size();
  r_list.offsets.reinitialize(points_num + 1);
  if (points_num == 0 || grid.is_empty()) {
    r_list.offsets.fill(0);
    r_list.indices.reinitialize(0);
    return;
  }

  Array<int> counts(points_num, 0);
  threading::parallel_for(IndexRange(points_num), fluid_task_grain(points_num), [&](const IndexRange range) {
    for (const int i : range) {
      int count = 0;
      grid.foreach_neighbor(i, positions, [&](const int, const float3 &, const float) { count++; });
      counts[i] = count;
    }
  });

  int offset = 0;
  for (const int i : IndexRange(points_num)) {
    r_list.offsets[i] = offset;
    offset += counts[i];
  }
  r_list.offsets[points_num] = offset;
  r_list.indices.reinitialize(offset);

  /* Fill in parallel with a per particle cursor: every particle writes into its own range. */
  Array<int> cursors(points_num, 0);
  for (const int i : IndexRange(points_num)) {
    cursors[i] = r_list.offsets[i];
  }
  threading::parallel_for(IndexRange(points_num), fluid_task_grain(points_num), [&](const IndexRange range) {
    for (const int i : range) {
      int cursor = cursors[i];
      grid.foreach_neighbor(i, positions, [&](const int j, const float3 &, const float) {
        r_list.indices[cursor++] = j;
      });
    }
  });
}

/** Builds the neighbour list of \a query_positions in the grid of another point set (the boundary). */
inline void fluid_build_neighbors_at(const FluidParticleGrid &grid,
                                     const Span<float3> query_positions,
                                     const Span<float3> positions,
                                     FluidNeighborList &r_list)
{
  const int points_num = int(query_positions.size());
  r_list.r_sq_max = grid.cell_size() * grid.cell_size();
  r_list.offsets.reinitialize(points_num + 1);
  if (points_num == 0 || grid.is_empty()) {
    r_list.offsets.fill(0);
    r_list.indices.reinitialize(0);
    return;
  }

  Array<int> counts(points_num, 0);
  threading::parallel_for(IndexRange(points_num), fluid_task_grain(points_num), [&](const IndexRange range) {
    for (const int i : range) {
      int count = 0;
      grid.foreach_neighbor_at(
          query_positions[i], positions, [&](const int, const float3 &, const float) { count++; });
      counts[i] = count;
    }
  });

  int offset = 0;
  for (const int i : IndexRange(points_num)) {
    r_list.offsets[i] = offset;
    offset += counts[i];
  }
  r_list.offsets[points_num] = offset;
  r_list.indices.reinitialize(offset);

  Array<int> cursors(points_num, 0);
  for (const int i : IndexRange(points_num)) {
    cursors[i] = r_list.offsets[i];
  }
  threading::parallel_for(IndexRange(points_num), fluid_task_grain(points_num), [&](const IndexRange range) {
    for (const int i : range) {
      int cursor = cursors[i];
      grid.foreach_neighbor_at(
          query_positions[i], positions, [&](const int j, const float3 &, const float) {
            r_list.indices[cursor++] = j;
          });
    }
  });
}

struct FluidDensitySettings {
  /** XPBD compliance of the density constraint, already divided by the time step squared. */
  float compliance_term = 0.0f;
  /** Akinci et al. artificial pressure, counteracts particle clumping at the surface. */
  float artificial_pressure = 0.0f;
  /** Radius of the artificial pressure kernel relative to the support radius. */
  float artificial_pressure_radius_factor = 0.2f;
  /** Maximum length of the position correction per iteration, 0 means unlimited. */
  float max_correction = 0.0f;
};

/**
 * Particles on the solid boundary, following Akinci et al. 2012 ("Versatile Rigid-Fluid Coupling
 * for Incompressible SPH"). Without them the SPH density estimate of a fluid particle at a wall is
 * truncated and only half of the rest density is seen, which is what makes the fluid stick to
 * walls and gives it a wrong pressure at the boundary.
 *
 * They take part in the density estimate and push the fluid away. A boundary particle with a
 * non-zero inverse mass belongs to another *simulated* geometry (cloth, soft body) and is moved by
 * the pressure as well, see #fluid_apply_boundary_reaction. Collider samples have an inverse mass
 * of zero and never move.
 */
struct FluidBoundary {
  const FluidParticleGrid *grid = nullptr;
  /** Neighbour list of the fluid particles in this point set, built once per substep. */
  const FluidNeighborList *neighbors = nullptr;
  Span<float3> positions;
  Span<float> masses;
  /** Inverse mass per boundary particle, empty or 0 for particles that cannot move. */
  Span<float> inv_masses;

  bool is_valid() const
  {
    return neighbors != nullptr && !neighbors->is_empty() && !positions.is_empty();
  }
};

/** Diagnostics of a single density iteration. */
struct FluidDensityStats {
  float average_error_squared = 0.0f;
  float max_abs_error = 0.0f;
  float max_correction = 0.0f;
  int max_neighbors = 0;
  /**
   * Signed relative density error `density / rest_density - 1`, averaged over the particles. It is
   * negative when the particles are too far apart for their radius (the density constraint is
   * unilateral, so an under-dense fluid is not solved at all and behaves like loose sand). The
   * clamped error above is always zero in that case, which makes this the number to look at when a
   * fluid "does not behave like a fluid".
   */
  float average_density_error = 0.0f;
  float min_density_error = 0.0f;
  int under_dense_num = 0;
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
                                          const Span<float> friction)
{
  const int points_num = int(velocities.size());
  threading::parallel_for(IndexRange(points_num), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      const float factor = 1.0f - friction[i] * solid_fractions[i];
      if (factor < 1.0f) {
        velocities[i] *= std::max(factor, 0.0f);
      }
    }
  });
}

/**
 * Reaction of the fluid pressure on the boundary particles that can move, i.e. the two way
 * coupling with the cloth and soft body geometries in the same world.
 *
 * A fluid particle \a i and a boundary particle \a j share the density constraint of \a i. The
 * pressure applies
 *     dm_i = lambda_i * m_j * grad_i W_ij / rho0_i
 * to the fluid particle (see #fluid_solve_density_iteration), so the conjugate position change of
 * the boundary particle is
 *     dm_j = -lambda_i * m_i * grad_i W_ij / rho0_i
 * which makes `m_i * dm_i + m_j * dm_j` zero for the pair, i.e. the momentum is conserved no
 * matter how much the masses of the fluid and the solid differ. That is what keeps the coupling
 * stable: the cloth pushes back exactly as hard as the water presses on it.
 *
 * \param fluid_positions, fluid_masses, fluid_rest_density, fluid_lambdas: Per particle data of
 * the fluid, indexed like the grid.
 * \param coupling: Per fluid particle strength of the reaction, 0 disables the coupling.
 * \param boundary: The boundary particles, only the ones with a positive inverse mass are moved.
 * \param max_correction: Maximum length of the position change of one boundary particle per
 *   iteration, 0 means unlimited.
 * \param r_corrections: Position change of each boundary particle, 0 for static ones.
 */
inline void fluid_apply_boundary_reaction(const FluidParticleGrid &fluid_grid,
                                          const FluidKernel &kernel,
                                          const Span<float3> fluid_positions,
                                          const Span<float> fluid_masses,
                                          const Span<float> fluid_rest_density,
                                          const Span<float> fluid_lambdas,
                                          const Span<float> coupling,
                                          const FluidBoundary &boundary,
                                          const float max_correction,
                                          const MutableSpan<float3> r_corrections)
{
  const int boundary_num = int(boundary.positions.size());
  if (boundary_num == 0 || fluid_grid.is_empty() || boundary.inv_masses.is_empty()) {
    return;
  }
  if (r_corrections.size() != boundary_num) {
    return;
  }
  threading::parallel_for(IndexRange(boundary_num), 256, [&](const IndexRange range) {
    for (const int j : range) {
      r_corrections[j] = float3(0.0f);
      if (boundary.inv_masses[j] <= 0.0f) {
        continue;
      }
      float3 correction = float3(0.0f);
      fluid_grid.foreach_neighbor_at(boundary.positions[j],
                                     fluid_positions,
                                     [&](const int i, const float3 &r_vec, const float r_sq) {
                                       if (UNLIKELY(r_sq < 1e-12f)) {
                                         return;
                                       }
                                       const float lambda_i = fluid_lambdas[i] * coupling[i];
                                       if (lambda_i == 0.0f) {
                                         return;
                                       }
                                       const float r = math::sqrt(r_sq);
                                       /* r_vec is the boundary position minus the fluid position,
                                        * so this is the gradient of the kernel with respect to the
                                        * boundary particle. */
                                       correction += lambda_i * fluid_masses[i] *
                                                     (-(kernel.spiky_gradient_length(r) / r) *
                                                      r_vec) *
                                                     math::safe_rcp(fluid_rest_density[i]);
                                     });
      if (max_correction > 0.0f) {
        const float length = math::length(correction);
        if (length > max_correction) {
          correction *= max_correction / length;
        }
      }
      if (UNLIKELY(!(std::isfinite(correction.x) && std::isfinite(correction.y) &&
                     std::isfinite(correction.z))))
      {
        correction = float3(0.0f);
      }
      r_corrections[j] = correction;
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
 * \param r_solid_fractions: Fraction of the density estimate of each particle that comes from the
 *   boundary and the coupled particles, used for the no-slip friction.
 * \param boundary: Optional boundary particles sampled on the colliders. They are part of the
 *   density estimate and push the fluid away, but they are never moved. They are only used for the
 *   particles where `use_boundary` is set.
 * \param coupled: Optional particles of the other simulated geometries of the world. They are
 *   always part of the density estimate (they are what makes the coupling work at all) and they
 * are moved by #fluid_apply_boundary_reaction.
 */
inline FluidDensityStats fluid_solve_density_iteration(const FluidNeighborList &neighbors,
                                                       const FluidKernel &kernel,
                                                       const MutableSpan<float3> positions,
                                                       const Span<float> masses,
                                                       const Span<float> inv_masses,
                                                       const Span<float> rest_density,
                                                       const Span<float> negative_pressure_limit,
                                                       const Span<bool> use_boundary,
                                                       const FluidDensitySettings &settings,
                                                       const MutableSpan<float> r_lambdas,
                                                       const MutableSpan<float> r_errors,
                                                       const MutableSpan<float> r_density_errors,
                                                       const MutableSpan<int> r_neighbor_counts,
                                                       const MutableSpan<float3> r_corrections,
                                                       const MutableSpan<float> r_solid_fractions,
                                                       const FluidBoundary &boundary,
                                                       const FluidBoundary &coupled,
                                                       const MutableSpan<float3> r_coupled_corrections = {})
{
  const int points_num = int(positions.size());
  if (points_num == 0 || neighbors.is_empty()) {
    return {};
  }

  const float artificial_pressure = settings.artificial_pressure;
  const float inv_artificial_pressure_reference = [&]() {
    if (artificial_pressure <= 0.0f) {
      return 0.0f;
    }
    const float radius = settings.artificial_pressure_radius_factor * kernel.h();
    return math::safe_rcp(kernel.poly6(radius * radius));
  }();

  /* Compute the density constraint and its lagrange multiplier for every particle. */
  threading::parallel_for(IndexRange(points_num), fluid_task_grain(points_num), [&](const IndexRange range) {
    for (const int i : range) {
      const float inv_rest_density = math::safe_rcp(rest_density[i]);
      float density = 0.0f;
      int neighbor_count = 0;
      float3 sum_gradient = float3(0.0f);
      float sum_gradient_sq = 0.0f;
      /* Mass free kernel weight of the fluid neighbors, used to measure which fraction of the
       * neighborhood is solid for the no-slip friction. Mass free, because the friction should
       * depend on how much of the neighborhood is wall and not on how the boundary mass happens to
       * be calibrated for the density estimate. */
      float fluid_weight = 0.0f;
      neighbors.foreach_neighbor(i, positions, [&](const int j, const float3 &r_vec, const float r_sq) {
        density += masses[j] * kernel.poly6(r_sq);
        fluid_weight += kernel.poly6(r_sq);
        neighbor_count += 1;
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
      /* Boundary particles are part of the density estimate and of the gradient w.r.t. the fluid
       * particle, but since they cannot move they are not part of the per neighbor gradient terms
       * of the denominator. */
      float boundary_density = 0.0f;
      float boundary_weight = 0.0f;
      const auto accumulate_boundary = [&](const FluidBoundary &boundary_particles) {
        if (!boundary_particles.is_valid()) {
          return;
        }
        /* The boundary particles live in their own point set, so the neighbour list is indexed by
         * the fluid particle and resolved against the boundary positions. */
        boundary_particles.neighbors->foreach_neighbor_at(
            i,
            positions,
            boundary_particles.positions,
            [&](const int j, const float3 &r_vec, const float r_sq) {
              const float weight = kernel.poly6(r_sq);
              boundary_density += boundary_particles.masses[j] * weight;
              boundary_weight += weight;
              if (UNLIKELY(r_sq < 1e-12f)) {
                return;
              }
              const float r = math::sqrt(r_sq);
              sum_gradient += boundary_particles.masses[j] *
                              (-(kernel.spiky_gradient_length(r) / r) * r_vec);
            });
      };
      if (use_boundary[i]) {
        accumulate_boundary(boundary);
      }
      /* The particles of the coupled solids are always part of the density estimate, otherwise the
       * fluid would see a free surface inside the cloth and push through it. */
      accumulate_boundary(coupled);
      density += boundary_density;
      r_solid_fractions[i] = boundary_weight * math::safe_rcp(boundary_weight + fluid_weight);

      const float constraint = density * inv_rest_density - 1.0f;
      /* Only compression is solved by default, see #FluidDensitySettings. */
      const float solved_constraint = std::max(constraint, -negative_pressure_limit[i]);
      r_errors[i] = math::square(solved_constraint);
      r_density_errors[i] = constraint;
      r_neighbor_counts[i] = neighbor_count;
      if (inv_masses[i] <= 0.0f) {
        /* Pinned particles are part of the density estimate but are not solved for. */
        r_lambdas[i] = 0.0f;
        continue;
      }
      const float denominator = (math::length_squared(sum_gradient) + sum_gradient_sq) *
                                    inv_rest_density * inv_rest_density +
                                settings.compliance_term + 1e-12f;
      /* The multiplier is not accumulated over the iterations: the correction below uses the
       * multipliers of all neighbors, so reusing the accumulated value would apply the same
       * correction again in every iteration (the fluid would keep expanding / churning).
       * The density constraint is also unilateral: clamping the multiplier keeps the solver from
       * pulling particles together, which would make the surface of the fluid implode. */
      const float max_lambda = negative_pressure_limit[i] / denominator;
      r_lambdas[i] = math::min(-solved_constraint / denominator, max_lambda);
    }
  });

  /* Accumulate the positional correction of each particle from the multipliers of its neighbors.
   */
  threading::parallel_for(IndexRange(points_num), fluid_task_grain(points_num), [&](const IndexRange range) {
    for (const int i : range) {
      if (inv_masses[i] <= 0.0f) {
        r_corrections[i] = float3(0.0f);
        continue;
      }
      const float lambda_i = r_lambdas[i];
      float3 correction = float3(0.0f);
      /* The part of the correction that comes from the coupled solids. It is tracked separately so
       * the velocity layer can remove the normal velocity it would otherwise create (see
       * `update_linear_velocities` in the solver). */
      float3 coupled_correction = float3(0.0f);
      neighbors.foreach_neighbor(i, positions, [&](const int j, const float3 &r_vec, const float r_sq) {
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
      if (lambda_i != 0.0f) {
        /* The boundary and the coupled particles never have a multiplier of their own, so only the
         * multiplier of the fluid particle pushes here. */
        const auto accumulate_boundary = [&](const FluidBoundary &boundary_particles,
                                              float3 *r_coupled) {
          if (!boundary_particles.is_valid()) {
            return;
          }
          boundary_particles.neighbors->foreach_neighbor_at(
              i,
              positions,
              boundary_particles.positions,
              [&](const int j, const float3 &r_vec, const float r_sq) {
                if (UNLIKELY(r_sq < 1e-12f)) {
                  return;
                }
                const float r = math::sqrt(r_sq);
                const float3 contribution = lambda_i * boundary_particles.masses[j] *
                                            (-(kernel.spiky_gradient_length(r) / r) * r_vec);
                correction += contribution;
                if (r_coupled != nullptr) {
                  *r_coupled += contribution;
                }
              });
        };
        accumulate_boundary(coupled, &coupled_correction);
        if (use_boundary[i]) {
          accumulate_boundary(boundary, nullptr);
        }
      }
      const float inv_rest_density = math::safe_rcp(rest_density[i]);
      correction *= inv_rest_density;
      float clamp_scale = 1.0f;
      if (settings.max_correction > 0.0f) {
        const float length = math::length(correction);
        if (length > settings.max_correction) {
          clamp_scale = settings.max_correction / length;
          correction *= clamp_scale;
        }
      }
      if (UNLIKELY(!(std::isfinite(correction.x) && std::isfinite(correction.y) &&
                     std::isfinite(correction.z))))
      {
        /* Never let a single degenerate particle poison the whole fluid. */
        correction = float3(0.0f);
        coupled_correction = float3(0.0f);
      }
      r_corrections[i] = correction;
      if (!r_coupled_corrections.is_empty()) {
        /* Near a coupled solid (or a boundary particle) the *whole* density correction is a
         * depenetration. Track it, so the velocity layer can remove its normal component instead
         * of turning the pressure wave into a splash. */
        const float3 push = r_solid_fractions[i] > 0.0f ?
                                correction :
                                coupled_correction * inv_rest_density * clamp_scale;
        r_coupled_corrections[i] += push;
      }
    }
  });

  FluidDensityStats stats;
  float error_sum = 0.0f;
  float density_error_sum = 0.0f;
  for (const int i : IndexRange(points_num)) {
    error_sum += r_errors[i];
    stats.max_abs_error = std::max(stats.max_abs_error, math::sqrt(r_errors[i]));
    stats.max_correction = std::max(stats.max_correction, math::length(r_corrections[i]));
    stats.max_neighbors = std::max(stats.max_neighbors, r_neighbor_counts[i]);
    density_error_sum += r_density_errors[i];
    stats.min_density_error = std::min(stats.min_density_error, r_density_errors[i]);
    if (r_density_errors[i] < -0.05f) {
      stats.under_dense_num += 1;
    }
  }
  stats.average_density_error = density_error_sum / points_num;

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
inline void fluid_apply_xsph_viscosity(const FluidNeighborList &neighbors,
                                       const FluidKernel &kernel,
                                       const Span<float3> positions,
                                       const Span<float> masses,
                                       const Span<float> rest_density,
                                       const Span<float> viscosity,
                                       const MutableSpan<float3> velocities,
                                       const MutableSpan<float3> r_scratch)
{
  const int points_num = int(velocities.size());
  if (points_num == 0 || neighbors.is_empty()) {
    return;
  }

  threading::parallel_for(IndexRange(points_num), fluid_task_grain(points_num), [&](const IndexRange range) {
    for (const int i : range) {
      const float3 velocity_i = velocities[i];
      float3 sum = float3(0.0f);
      neighbors.foreach_neighbor(
          i, positions, [&](const int j, const float3 & /*r_vec*/, const float r_sq) {
            if (j == i) {
              return;
            }
            sum += masses[j] * (velocities[j] - velocity_i) * kernel.poly6(r_sq);
          });
      r_scratch[i] = velocity_i + (viscosity[i] * math::safe_rcp(rest_density[i])) * sum;
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
inline void fluid_apply_velocity_damping(const Span<float> damping,
                                         const float delta_time,
                                         const MutableSpan<float3> velocities)
{
  if (delta_time <= 0.0f) {
    return;
  }
  threading::parallel_for(IndexRange(int(velocities.size())), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      velocities[i] *= math::safe_rcp(1.0f + damping[i] * delta_time);
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
inline void fluid_apply_vorticity_confinement(const FluidNeighborList &neighbors,
                                              const FluidKernel &kernel,
                                              const Span<float3> positions,
                                              const Span<float> masses,
                                              const Span<float> rest_density,
                                              const Span<float> strength,
                                              const float delta_time,
                                              const MutableSpan<float3> velocities,
                                              const MutableSpan<float3> r_omegas,
                                              const MutableSpan<float> r_omega_magnitudes,
                                              const MutableSpan<float3> r_scratch)
{
  const int points_num = int(velocities.size());
  if (points_num == 0 || neighbors.is_empty() || delta_time <= 0.0f) {
    return;
  }

  /* omega_i = curl(v) at the particle. */
  threading::parallel_for(IndexRange(points_num), fluid_task_grain(points_num), [&](const IndexRange range) {
    for (const int i : range) {
      const float3 velocity_i = velocities[i];
      float3 omega = float3(0.0f);
      neighbors.foreach_neighbor(i, positions, [&](const int j, const float3 &r_vec, const float r_sq) {
        if (j == i || r_sq < 1e-12f) {
          return;
        }
        const float r = math::sqrt(r_sq);
        omega += masses[j] *
                 math::cross(velocities[j] - velocity_i, kernel.spiky_gradient(r_vec, r));
      });
      omega *= math::safe_rcp(rest_density[i]);
      r_omegas[i] = omega;
      r_omega_magnitudes[i] = math::length(omega);
    }
  });

  /* f = epsilon * h * (N x omega), with N being the normalized gradient of |omega|. */
  const float max_velocity_change = 0.1f * kernel.h() * math::safe_rcp(delta_time);
  threading::parallel_for(IndexRange(points_num), fluid_task_grain(points_num), [&](const IndexRange range) {
    for (const int i : range) {
      r_scratch[i] = float3(0.0f);
      const float magnitude_i = r_omega_magnitudes[i];
      float3 gradient = float3(0.0f);
      neighbors.foreach_neighbor(i, positions, [&](const int j, const float3 &r_vec, const float r_sq) {
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
      float3 change = (strength[i] * kernel.h() * delta_time) *
                      math::cross(direction, r_omegas[i]);
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
inline void fluid_apply_surface_tension(const FluidNeighborList &neighbors,
                                        const Span<float3> positions,
                                        const Span<float> masses,
                                        const float h,
                                        const Span<float> strength,
                                        const float delta_time,
                                        const MutableSpan<float3> velocities,
                                        const MutableSpan<float3> r_scratch)
{
  const int points_num = int(velocities.size());
  if (points_num == 0 || neighbors.is_empty() || delta_time <= 0.0f) {
    return;
  }

  threading::parallel_for(IndexRange(points_num), fluid_task_grain(points_num), [&](const IndexRange range) {
    for (const int i : range) {
      const float3 &position_i = positions[i];
      float3 change = float3(0.0f);
      neighbors.foreach_neighbor(
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
      r_scratch[i] = (strength[i] * delta_time) * change;
    }
  });

  threading::parallel_for(IndexRange(points_num), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      velocities[i] += r_scratch[i];
    }
  });
}

}  // namespace blender::xpbd
