/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>

#include "GEO_xpbd_constraint_coloring_utils.hh"
#include "GEO_xpbd_constraint_set_templated.hh"
#include "GEO_xpbd_constraint_shape_matching.hh"

namespace blender::xpbd {

/**
 * Contact between two granular particles.
 *
 * The normal part is the same unilateral distance constraint the self collision uses. On top of it
 * the relative tangential motion of the substep is cancelled, limited by the Coulomb friction
 * coefficient times the normal multiplier. That static friction is what lets a pile of particles
 * stand at an angle instead of flowing into a flat puddle.
 */
class GrainContactConstraintSet : public TemplatedConstraintSet<GrainContactConstraintSet> {
 private:
  Span<int2> point_pairs_;
  Span<float> min_distances_;
  Span<float> compliances_;
  Span<float> frictions_;
  Span<int> point_geo_;
  Span<int> point_local_;
  Span<float> inv_masses_;
  float error_scale_;
  MutableSpan<float> lambdas_;

 public:
  static constexpr StringRefNull debug_name = "Grain Contact";

  GrainContactConstraintSet(const Vector<int> &affected_geo_indices,
                            const Span<int2> point_pairs,
                            const Span<float> min_distances,
                            const Span<float> compliances,
                            const Span<float> frictions,
                            const Span<int> point_geo,
                            const Span<int> point_local,
                            const Span<float> inv_masses,
                            const float error_scale,
                            MutableSpan<float> lambdas)
      : TemplatedConstraintSet<GrainContactConstraintSet>(point_pairs.size(), affected_geo_indices),
        point_pairs_(point_pairs),
        min_distances_(min_distances),
        compliances_(compliances),
        frictions_(frictions),
        point_geo_(point_geo),
        point_local_(point_local),
        inv_masses_(inv_masses),
        error_scale_(error_scale),
        lambdas_(lambdas)
  {
  }

  void reset_force(const int constraint_i) const
  {
    lambdas_[constraint_i] = 0.0f;
  }

  template<typename UpdaterT>
  void solve_single(const ConstraintSetParams &params,
                    UpdaterT &updater,
                    const int constraint_i) const
  {
    const int2 &pair = point_pairs_[constraint_i];
    const int global_i0 = pair[0];
    const int global_i1 = pair[1];
    const int geo_i0 = point_geo_[global_i0];
    const int geo_i1 = point_geo_[global_i1];
    const int point_i0 = point_local_[global_i0];
    const int point_i1 = point_local_[global_i1];
    const float inv_m0 = inv_masses_[global_i0];
    const float inv_m1 = inv_masses_[global_i1];
    const float compliance = compliances_[constraint_i] * params.compliance_term_factor;
    if (inv_m0 + inv_m1 + compliance <= 0.0f) {
      return;
    }

    const float3 p0 = params.position(geo_i0, point_i0);
    const float3 p1 = params.position(geo_i1, point_i1);
    float3 normal;
    float length;
    normal = math::normalize_and_get_length(p1 - p0, length);
    if (length <= 1e-8f) {
      normal = float3(1.0f, 0.0f, 0.0f);
    }
    const float min_distance = min_distances_[constraint_i];
    const float error = length - min_distance;
    const float error_with_lambda = error + compliance * lambdas_[constraint_i];

    float3 offset0(0.0f);
    float3 offset1(0.0f);
    float normal_lambda = lambdas_[constraint_i];
    if (error < 0.0f) {
      const float delta_lambda = (-error_with_lambda) / (inv_m0 + inv_m1 + compliance);
      normal_lambda += delta_lambda;
      offset0 = -delta_lambda * inv_m0 * normal;
      offset1 = delta_lambda * inv_m1 * normal;
      lambdas_[constraint_i] = normal_lambda;
      updater.update_position(geo_i0, point_i0, offset0);
      updater.update_position(geo_i1, point_i1, offset1);
      updater.add_residual_error(geo_i0,
                                 math::square(error_with_lambda) * error_scale_);
    }

    /* Tangential friction. Previous positions are the state at the beginning of the substep, so
     * the relative tangential displacement of this substep is exactly what should be cancelled. */
    const float friction = frictions_[constraint_i];
    if (!(friction > 0.0f)) {
      return;
    }
    const float3 p0_new = p0 + offset0;
    const float3 p1_new = p1 + offset1;
    const float3 prev_delta = params.prev_position(geo_i1, point_i1) -
                              params.prev_position(geo_i0, point_i0);
    const float3 current_delta = p1_new - p0_new;
    const float3 relative = current_delta - prev_delta;
    const float3 tangential = relative - normal * math::dot(relative, normal);
    float tangent_length;
    const float3 tangent = math::normalize_and_get_length(tangential, tangent_length);
    if (tangent_length <= 1e-8f) {
      return;
    }

    /* The Coulomb limit uses the normal multiplier of this contact, so a light touch slides more
     * easily than a compressed contact. */
    const float limit = friction * std::abs(normal_lambda);
    if (limit <= 0.0f) {
      return;
    }
    float delta_lambda_t = -tangent_length / (inv_m0 + inv_m1 + compliance);
    delta_lambda_t = math::clamp(delta_lambda_t, -limit, limit);
    const float3 tangent_offset0 = -delta_lambda_t * inv_m0 * tangent;
    const float3 tangent_offset1 = delta_lambda_t * inv_m1 * tangent;
    updater.update_position(geo_i0, point_i0, tangent_offset0);
    updater.update_position(geo_i1, point_i1, tangent_offset1);
  }

  ConstraintColoring color_constraints(LinearAllocator<> &memory) const override
  {
    return color_constraints__binary(point_pairs_, memory);
  }
};

/**
 * Keeps the points of one temporary grain clump in a best fit shape.
 *
 * Same least squares fit as #ShapeMatchingConstraintSet, but a clump can break: when the maximum
 * distance of a point from the fitted shape exceeds `break_strain` (relative to the clump radius)
 * the points with the largest error are marked in `break_flags`. The solver turns those into a
 * cooldown, so the next frames are free grains that can later join a new clump.
 */
class GrainClumpConstraintSet : public TemplatedConstraintSet<GrainClumpConstraintSet> {
 private:
  OffsetIndices<int> group_offsets_;
  Span<int> points_;
  Span<int> point_geo_;
  Span<int> point_local_;
  Span<float3> rest_offsets_;
  Span<float> masses_;
  Span<float> stiffnesses_;
  Span<float> rest_radii_;
  MutableSpan<uint8_t> break_flags_;
  Span<float> break_strains_;
  Span<float> break_fractions_;
  bool detect_break_ = true;

 public:
  static constexpr StringRefNull debug_name = "Grain Clump";

  GrainClumpConstraintSet(const OffsetIndices<int> group_offsets,
                          const Span<int> points,
                          const Span<int> point_geo,
                          const Span<int> point_local,
                          const Span<float3> rest_offsets,
                          const Span<float> masses,
                          const Span<float> stiffnesses,
                          const Span<float> rest_radii,
                          MutableSpan<uint8_t> break_flags,
                          const Span<float> break_strains,
                          const Span<float> break_fractions)
      : TemplatedConstraintSet<GrainClumpConstraintSet>(group_offsets.size(), {}),
        group_offsets_(group_offsets),
        points_(points),
        point_geo_(point_geo),
        point_local_(point_local),
        rest_offsets_(rest_offsets),
        masses_(masses),
        stiffnesses_(stiffnesses),
        rest_radii_(rest_radii),
        break_flags_(break_flags),
        break_strains_(break_strains),
        break_fractions_(break_fractions)
  {
  }

  void set_detect_break(const bool value)
  {
    detect_break_ = value;
  }

  void reset_force(const int /*constraint_i*/) const {}

  template<typename UpdaterT>
  void solve_single(const ConstraintSetParams &params,
                    UpdaterT &updater,
                    const int group_i) const
  {
    const IndexRange group = group_offsets_[group_i];
    if (group.is_empty()) {
      return;
    }
    const float rest_radius = rest_radii_[group_i];
    if (!(rest_radius > 1e-6f)) {
      return;
    }

    /* A stale clump slot (for example after a topology change) is skipped instead of reading out
     * of bounds. */
    for (const int k : group) {
      const int point_i = points_[k];
      const int geo_i = point_geo_[point_i];
      const int local_i = point_local_[point_i];
      if (geo_i < 0 || geo_i >= params.geometry_refs().size() ||
          local_i < 0 || local_i >= params.positions(geo_i).size())
      {
        return;
      }
    }

    float total_mass = 0.0f;
    float3 center(0.0f);
    for (const int k : group) {
      const int point_i = points_[k];
      const int geo_i = point_geo_[point_i];
      const int local_i = point_local_[point_i];
      if (geo_i < 0 || geo_i >= params.geometry_refs().size() ||
          local_i < 0 || local_i >= params.positions(geo_i).size())
      {
        continue;
      }
      const float mass = masses_[k];
      total_mass += mass;
      center += mass * params.position(geo_i, local_i);
    }
    if (!(total_mass > 0.0f)) {
      return;
    }
    center /= total_mass;

    ShapeMatchingMatrix covariance{};
    for (const int k : group) {
      const int point_i = points_[k];
      const float3 current = params.position(point_geo_[point_i], point_local_[point_i]) - center;
      const float3 rest = rest_offsets_[k];
      const float mass = masses_[k];
      covariance.m[0][0] += mass * current.x * rest.x;
      covariance.m[0][1] += mass * current.x * rest.y;
      covariance.m[0][2] += mass * current.x * rest.z;
      covariance.m[1][0] += mass * current.y * rest.x;
      covariance.m[1][1] += mass * current.y * rest.y;
      covariance.m[1][2] += mass * current.y * rest.z;
      covariance.m[2][0] += mass * current.z * rest.x;
      covariance.m[2][1] += mass * current.z * rest.y;
      covariance.m[2][2] += mass * current.z * rest.z;
    }

    ShapeMatchingMatrix rotation;
    shape_matching_rotation(covariance, rotation);

    Array<float> errors(group.size());
    float max_error = 0.0f;
    for (const int i : group.index_range()) {
      const int k = group[i];
      const int point_i = points_[k];
      const float3 current = params.position(point_geo_[point_i], point_local_[point_i]);
      const float3 goal = shape_matching_mul(rotation, rest_offsets_[k]) + center;
      errors[i] = math::length(goal - current);
      max_error = std::max(max_error, errors[i]);
    }

    const float break_strain = break_strains_[group_i];
    const float break_fraction = break_fractions_[group_i];
    if (detect_break_ && break_strain > 0.0f &&
        max_error / rest_radius > break_strain)
    {
      Array<int> order(group.size());
      for (const int i : group.index_range()) {
        order[i] = i;
      }
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        if (errors[a] == errors[b]) {
          return a < b;
        }
        return errors[a] > errors[b];
      });
      const int break_count = math::clamp(
          int(std::round(float(group.size()) * break_fraction)), 1, int(group.size()));
      for (const int i : IndexRange(break_count)) {
        break_flags_[group[order[i]]] = 1;
      }
    }

    const float max_correction = 0.5f * rest_radius;
    float error_squared = 0.0f;
    for (const int k : group) {
      if (break_flags_[k]) {
        continue;
      }
      const int point_i = points_[k];
      const float3 current = params.position(point_geo_[point_i], point_local_[point_i]);
      const float3 goal = shape_matching_mul(rotation, rest_offsets_[k]) + center;
      const float3 remaining = goal - current;
      error_squared += math::length_squared(remaining);
      float3 offset = remaining * stiffnesses_[k];
      const float offset_length = math::length(offset);
      if (offset_length > max_correction) {
        offset *= max_correction / offset_length;
      }
      updater.update_position(point_geo_[point_i], point_local_[point_i], offset);
    }

    const float mean_error = error_squared / (float(group.size()) * rest_radius * rest_radius);
    updater.add_residual_error(point_geo_[points_[group[0]]], mean_error);
  }

  ConstraintColoring color_constraints(LinearAllocator<> & /*memory*/) const override
  {
    return color_constraints__all_independent(constraints_num_);
  }
};

}  // namespace blender::xpbd
