/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "GEO_xpbd_constraint_coloring_utils.hh"
#include "GEO_xpbd_constraint_set_templated.hh"

namespace blender::xpbd {

struct PointContactResult {
  /** Whether the two points are closer than their minimum distance. */
  bool is_active = false;
  float delta_lambda = 0.0f;
  float3 offset0 = float3(0.0f);
  float3 offset1 = float3(0.0f);
  /** Squared penetration, zero when the constraint does not do anything. */
  float residual_error_squared = 0.0f;
};

/**
 * Unilateral minimum distance constraint between two points of the simulation.
 *
 * Unlike #DistanceConstraintSet the constraint only acts when the points are *closer* than their
 * rest distance (which is the sum of the two collision radii here), so it can be used to keep the
 * points of a cloth, or of a cloth and a soft body, apart without gluing them together. This is the
 * cheapest form of self collision there is: it knows nothing about triangles or winding, it just
 * keeps two spheres from overlapping, which is enough to keep a folded sheet from passing through
 * itself as long as the points are dense enough.
 */
inline PointContactResult evaluate_point_contact(const float3 &p0,
                                                 const float3 &p1,
                                                 const float inv_m0,
                                                 const float inv_m1,
                                                 const float min_distance,
                                                 const float compliance_term,
                                                 const float lambda_prev)
{
  if (inv_m0 + inv_m1 == 0.0f) {
    return {};
  }

  const float3 p_diff = p1 - p0;
  float length;
  float3 normalized_dir = math::normalize_and_get_length(p_diff, length);
  if (length <= 1e-8f) {
    /* Two coincident points have no direction. Any axis separates them, and the next iteration has
     * a well defined direction again. */
    normalized_dir = float3(1.0f, 0.0f, 0.0f);
  }
  if (length >= min_distance) {
    return {};
  }

  /* Negative while the two spheres overlap, like the error of a distance constraint with the
   * contact distance as its rest length. */
  const float error = length - min_distance;
  const float error_squared = math::square(error + compliance_term * lambda_prev);
  if (!(inv_m0 + inv_m1 + compliance_term > 0.0f)) {
    return {};
  }
  const float delta_lambda = (-error - compliance_term * lambda_prev) /
                             (inv_m0 + inv_m1 + compliance_term);

  const float3 offset0 = -delta_lambda * inv_m0 * normalized_dir;
  const float3 offset1 = delta_lambda * inv_m1 * normalized_dir;

  return {true, delta_lambda, offset0, offset1, error_squared};
}

/**
 * The point/point contacts of the simulated solids.
 *
 * All points of all participating geometries live in one index space (#point_geo and #point_local
 * map a global point index to the geometry and to the point within it), so one set covers a cloth
 * against itself, two cloths against each other and a cloth against a soft body at the same time.
 *
 * The pairs are rebuilt every substep by the solver (they depend on the positions), the multipliers
 * are reset with the other constraints.
 */
class PointContactConstraintSet : public TemplatedConstraintSet<PointContactConstraintSet> {
 private:
  /** Global point indices of the two points of each contact. */
  Span<int2> point_pairs_;
  /** Distance to keep, i.e. the sum of the two radii plus a thickness. */
  Span<float> min_distances_;
  Span<float> compliances_;
  /** Global point index -> geometry index and index within that geometry. */
  Span<int> point_geo_;
  Span<int> point_local_;
  /**
   * Inverse mass of every point, zero for a hard pinned point: a pin is an infinitely stiff
   * constraint, so a contact against it must move the *other* point only. Using the mass of the
   * pinned point instead leaves half of the correction to be undone by the pin in the next
   * iteration, which makes a pinned sheet soft under a load (measured: a soft body with water on top
   * was pressed through a fully pinned cloth).
   */
  Span<float> inv_masses_;
  /* Scale factor for residual error. */
  float error_scale_;
  MutableSpan<float> lambdas_;

 public:
  static constexpr StringRefNull debug_name = "Point Contact";

  PointContactConstraintSet(const Vector<int> &affected_geo_indices,
                            const Span<int2> point_pairs,
                            const Span<float> min_distances,
                            const Span<float> compliances,
                            const Span<int> point_geo,
                            const Span<int> point_local,
                            const Span<float> inv_masses,
                            const float error_scale,
                            MutableSpan<float> lambdas)
      : TemplatedConstraintSet<PointContactConstraintSet>(point_pairs.size(), affected_geo_indices),
        point_pairs_(point_pairs),
        min_distances_(min_distances),
        compliances_(compliances),
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
    const int2 &point_pair = point_pairs_[constraint_i];
    const int global_i0 = point_pair[0];
    const int global_i1 = point_pair[1];
    const int geo_i0 = point_geo_[global_i0];
    const int geo_i1 = point_geo_[global_i1];
    const int point_i0 = point_local_[global_i0];
    const int point_i1 = point_local_[global_i1];

    const PointContactResult result = evaluate_point_contact(
        params.position(geo_i0, point_i0),
        params.position(geo_i1, point_i1),
        inv_masses_[global_i0],
        inv_masses_[global_i1],
        min_distances_[constraint_i],
        compliances_[constraint_i] * params.compliance_term_factor,
        lambdas_[constraint_i]);
    if (!result.is_active) {
      return;
    }

    lambdas_[constraint_i] += result.delta_lambda;
    updater.update_position(geo_i0, point_i0, result.offset0);
    updater.update_position(geo_i1, point_i1, result.offset1);
    updater.add_residual_error(geo_i0, result.residual_error_squared * error_scale_);
  }

  ConstraintColoring color_constraints(LinearAllocator<> &memory) const override
  {
    return color_constraints__binary(point_pairs_, memory);
  }
};

}  // namespace blender::xpbd
