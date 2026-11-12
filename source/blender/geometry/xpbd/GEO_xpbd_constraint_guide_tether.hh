/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "GEO_xpbd_constraint_coloring_utils.hh"
#include "GEO_xpbd_constraint_set_templated.hh"

namespace blender::xpbd {

struct GuideTetherResult {
  /** Whether the attachment does anything (always true unless both points are infinite mass). */
  bool is_active = false;
  float3 delta_lambda = float3(0.0f);
  float3 offset_child = float3(0.0f);
  float3 offset_guide = float3(0.0f);
  /** Squared distance from the rest offset. */
  float residual_error_squared = 0.0f;
};

/**
 * Compliant position attachment between a child hair point and the point of its guide.
 *
 * This is a three component XPBD constraint: `C = p_child - p_guide - rest_offset`. The correction
 * is distributed by inverse mass, so it has two jobs at the same time:
 *
 *  - the guide drags the child along (the usual guide hair workflow),
 *  - a child that is displaced by a collision pulls the guide with it, which is what lets the
 *    collisions of the children affect the guide.
 *
 * The compliance is the usual XPBD compliance in m/N (the solver already divides it by dt²).
 */
inline GuideTetherResult evaluate_guide_tether(const float3 &p_child,
                                               const float3 &p_guide,
                                               const float3 &rest_offset,
                                               const float inv_m_child,
                                               const float inv_m_guide,
                                               const float compliance_term,
                                               const float3 &lambda_prev)
{
  const float effective_inv_mass = inv_m_child + inv_m_guide;
  if (!(effective_inv_mass + compliance_term > 0.0f)) {
    return {};
  }
  const float3 error = p_child - p_guide - rest_offset;
  const float3 delta_lambda = (-error - compliance_term * lambda_prev) /
                              (effective_inv_mass + compliance_term);

  GuideTetherResult result;
  result.is_active = true;
  result.delta_lambda = delta_lambda;
  result.offset_child = delta_lambda * inv_m_child;
  result.offset_guide = -delta_lambda * inv_m_guide;
  result.residual_error_squared = math::length_squared(error);
  return result;
}

/**
 * The guide tether attaches child hair points to the points of their guides.
 *
 * `guide_curve_index` (the standard hair attribute) selects the guide curve of every child curve;
 * curves with the `Guide Tether` attribute off are the guides. Each child point is attached to the
 * guide point at the same normalized curve parameter. The constraint set spans all points (guide
 * and child), so the correction is two way and the colors are built from the two global point
 * indices of every tether.
 */
class GuideTetherConstraintSet : public TemplatedConstraintSet<GuideTetherConstraintSet> {
 private:
  /** Global point indices: child point, guide point. */
  Span<int2> point_pairs_;
  /** Rest offset of the child from its guide point. */
  Span<float3> rest_offsets_;
  Span<float> compliances_;
  Span<int> point_geo_;
  Span<int> point_local_;
  Span<float> inv_masses_;
  float error_scale_;
  MutableSpan<float3> lambdas_;

 public:
  static constexpr StringRefNull debug_name = "Guide Tether";

  GuideTetherConstraintSet(const Vector<int> &affected_geo_indices,
                           const Span<int2> point_pairs,
                           const Span<float3> rest_offsets,
                           const Span<float> compliances,
                           const Span<int> point_geo,
                           const Span<int> point_local,
                           const Span<float> inv_masses,
                           const float error_scale,
                           MutableSpan<float3> lambdas)
      : TemplatedConstraintSet<GuideTetherConstraintSet>(point_pairs.size(), affected_geo_indices),
        point_pairs_(point_pairs),
        rest_offsets_(rest_offsets),
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
    lambdas_[constraint_i] = float3(0.0f);
  }

  template<typename UpdaterT>
  void solve_single(const ConstraintSetParams &params,
                    UpdaterT &updater,
                    const int constraint_i) const
  {
    const int2 &point_pair = point_pairs_[constraint_i];
    const int geo_i_child = point_geo_[point_pair[0]];
    const int geo_i_guide = point_geo_[point_pair[1]];
    const int point_i_child = point_local_[point_pair[0]];
    const int point_i_guide = point_local_[point_pair[1]];
    const int geometries_num = params.geometry_refs().size();
    if (geo_i_child < 0 || geo_i_child >= geometries_num || geo_i_guide < 0 ||
        geo_i_guide >= geometries_num)
    {
      return;
    }
    if (point_i_child < 0 || point_i_child >= params.positions(geo_i_child).size() ||
        point_i_guide < 0 || point_i_guide >= params.positions(geo_i_guide).size())
    {
      return;
    }

    const GuideTetherResult result = evaluate_guide_tether(
        params.position(geo_i_child, point_i_child),
        params.position(geo_i_guide, point_i_guide),
        rest_offsets_[constraint_i],
        inv_masses_[point_pair[0]],
        inv_masses_[point_pair[1]],
        compliances_[constraint_i] * params.compliance_term_factor,
        lambdas_[constraint_i]);
    if (!result.is_active) {
      return;
    }

    lambdas_[constraint_i] += result.delta_lambda;
    updater.update_position(geo_i_child, point_i_child, result.offset_child);
    updater.update_position(geo_i_guide, point_i_guide, result.offset_guide);
    updater.add_residual_error(geo_i_child, result.residual_error_squared * error_scale_);
  }

  ConstraintColoring color_constraints(LinearAllocator<> &memory) const override
  {
    return color_constraints__binary(point_pairs_, memory);
  }
};

}  // namespace blender::xpbd
