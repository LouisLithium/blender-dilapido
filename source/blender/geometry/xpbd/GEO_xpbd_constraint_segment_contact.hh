/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "BLI_offset_indices.hh"

#include "GEO_xpbd_constraint_coloring_utils.hh"
#include "GEO_xpbd_constraint_set_templated.hh"

namespace blender::xpbd {

struct SegmentContactResult {
  /** Whether the two segments are closer than their minimum distance. */
  bool is_active = false;
  float delta_lambda = 0.0f;
  float3 offset0 = float3(0.0f);
  float3 offset1 = float3(0.0f);
  float3 offset2 = float3(0.0f);
  float3 offset3 = float3(0.0f);
  /** Squared penetration, zero when the constraint does not do anything. */
  float residual_error_squared = 0.0f;
};

/**
 * Closest points between the segments \a p0-\a p1 and \a q0-\a q1.
 *
 * The parameters are clamped to [0, 1], so the result is also valid for degenerate (zero length)
 * segments. This is the standard clamped segment/segment test from Ericson, "Real-Time Collision
 * Detection".
 */
inline void closest_points_on_segments(const float3 &p0,
                                       const float3 &p1,
                                       const float3 &q0,
                                       const float3 &q1,
                                       float &r_s,
                                       float &r_t)
{
  const float3 d1 = p1 - p0;
  const float3 d2 = q1 - q0;
  const float3 r = p0 - q0;
  const float a = math::dot(d1, d1);
  const float e = math::dot(d2, d2);
  const float f = math::dot(d2, r);
  constexpr float eps = 1e-12f;

  if (a <= eps && e <= eps) {
    r_s = 0.0f;
    r_t = 0.0f;
  }
  else if (a <= eps) {
    r_s = 0.0f;
    r_t = math::clamp(f / e, 0.0f, 1.0f);
  }
  else {
    const float c = math::dot(d1, r);
    if (e <= eps) {
      r_t = 0.0f;
      r_s = math::clamp(-c / a, 0.0f, 1.0f);
    }
    else {
      const float b = math::dot(d1, d2);
      const float denominator = a * e - b * b;
      r_s = denominator != 0.0f ? math::clamp((b * f - c * e) / denominator, 0.0f, 1.0f) : 0.0f;
      r_t = (b * r_s + f) / e;
      if (r_t < 0.0f) {
        r_t = 0.0f;
        r_s = math::clamp(-c / a, 0.0f, 1.0f);
      }
      else if (r_t > 1.0f) {
        r_t = 1.0f;
        r_s = math::clamp((b - c) / a, 0.0f, 1.0f);
      }
    }
  }
}

/**
 * Unilateral minimum distance constraint between two segments.
 *
 * This is the hair version of #PointContactConstraintSet: a hair strand is a chain of segments, and
 * a point/point test would only catch two strands when a point of one happens to land almost exactly
 * on a point of the other. The closest points of the two segments are used instead, and the
 * correction is distributed to the four endpoints with the same barycentric weights, which keeps the
 * constraint momentum conserving.
 *
 * The two virtual points have the effective inverse masses
 * `(1-s)^2 * w0 + s^2 * w1` and `(1-t)^2 * w2 + t^2 * w3`, which is exactly the denominator an XPBD
 * distance constraint would have for those two points.
 */
inline SegmentContactResult evaluate_segment_contact(const float3 &p0,
                                                     const float3 &p1,
                                                     const float3 &p2,
                                                     const float3 &p3,
                                                     const float inv_m0,
                                                     const float inv_m1,
                                                     const float inv_m2,
                                                     const float inv_m3,
                                                     const float min_distance,
                                                     const float compliance_term,
                                                     const float lambda_prev)
{
  if (inv_m0 + inv_m1 + inv_m2 + inv_m3 == 0.0f) {
    return {};
  }

  float s;
  float t;
  closest_points_on_segments(p0, p1, p2, p3, s, t);
  const float3 q0 = math::interpolate(p0, p1, s);
  const float3 q1 = math::interpolate(p2, p3, t);
  const float3 diff = q1 - q0;
  float length;
  float3 direction = math::normalize_and_get_length(diff, length);
  if (length <= 1e-8f) {
    /* Coincident segments have no direction. Any axis separates them, and the next iteration has a
     * well defined direction again. */
    direction = float3(1.0f, 0.0f, 0.0f);
  }
  if (length >= min_distance) {
    return {};
  }

  const float error = length - min_distance;
  const float w0 = 1.0f - s;
  const float w1 = s;
  const float w2 = 1.0f - t;
  const float w3 = t;
  const float effective_inv_mass = w0 * w0 * inv_m0 + w1 * w1 * inv_m1 + w2 * w2 * inv_m2 +
                                   w3 * w3 * inv_m3;
  if (!(effective_inv_mass + compliance_term > 0.0f)) {
    return {};
  }
  const float delta_lambda = (-error - compliance_term * lambda_prev) /
                             (effective_inv_mass + compliance_term);

  SegmentContactResult result;
  result.is_active = true;
  result.delta_lambda = delta_lambda;
  result.offset0 = -delta_lambda * inv_m0 * w0 * direction;
  result.offset1 = -delta_lambda * inv_m1 * w1 * direction;
  result.offset2 = delta_lambda * inv_m2 * w2 * direction;
  result.offset3 = delta_lambda * inv_m3 * w3 * direction;
  result.residual_error_squared = math::square(error);
  return result;
}

/**
 * Segment/segment contacts of the hair with itself.
 *
 * Every pair is four point indices (two segments), so the constraints are colored by the two segment
 * indices: constraints that share a segment get different colors, and two constraints that share a
 * point but not a segment are filtered out by the pair builder (adjacent segments of one strand).
 * This is much cheaper than coloring by all four points.
 */
class SegmentContactConstraintSet : public TemplatedConstraintSet<SegmentContactConstraintSet> {
 private:
  /** Four point indices per contact, as two segments. */
  GroupedSpan<int> points_;
  Span<float> min_distances_;
  Span<float> compliances_;
  /** Segment index pair of each contact, used for coloring. */
  Span<int2> segment_pairs_;
  /** Global point index -> geometry index and index within that geometry. */
  Span<int> point_geo_;
  Span<int> point_local_;
  Span<float> inv_masses_;
  float error_scale_;
  MutableSpan<float> lambdas_;

 public:
  static constexpr StringRefNull debug_name = "Segment Contact";

  SegmentContactConstraintSet(const Vector<int> &affected_geo_indices,
                              const GroupedSpan<int> points,
                              const Span<float> min_distances,
                              const Span<float> compliances,
                              const Span<int2> segment_pairs,
                              const Span<int> point_geo,
                              const Span<int> point_local,
                              const Span<float> inv_masses,
                              const float error_scale,
                              MutableSpan<float> lambdas)
      : TemplatedConstraintSet<SegmentContactConstraintSet>(points.size(), affected_geo_indices),
        points_(points),
        min_distances_(min_distances),
        compliances_(compliances),
        segment_pairs_(segment_pairs),
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
    const Span<int> points = points_[constraint_i];
    if (points.size() != 4) {
      return;
    }
    const int global_i0 = points[0];
    const int global_i1 = points[1];
    const int global_i2 = points[2];
    const int global_i3 = points[3];
    if (global_i0 < 0 || global_i0 >= point_geo_.size() ||
        global_i1 < 0 || global_i1 >= point_geo_.size() ||
        global_i2 < 0 || global_i2 >= point_geo_.size() ||
        global_i3 < 0 || global_i3 >= point_geo_.size())
    {
      return;
    }
    const int geo_i0 = point_geo_[global_i0];
    const int geo_i1 = point_geo_[global_i1];
    const int geo_i2 = point_geo_[global_i2];
    const int geo_i3 = point_geo_[global_i3];
    const int point_i0 = point_local_[global_i0];
    const int point_i1 = point_local_[global_i1];
    const int point_i2 = point_local_[global_i2];
    const int point_i3 = point_local_[global_i3];
    const int geometries_num = params.geometry_refs().size();
    if (geo_i0 < 0 || geo_i0 >= geometries_num || geo_i1 < 0 || geo_i1 >= geometries_num ||
        geo_i2 < 0 || geo_i2 >= geometries_num || geo_i3 < 0 || geo_i3 >= geometries_num)
    {
      return;
    }
    if (point_i0 < 0 || point_i0 >= params.positions(geo_i0).size() ||
        point_i1 < 0 || point_i1 >= params.positions(geo_i1).size() ||
        point_i2 < 0 || point_i2 >= params.positions(geo_i2).size() ||
        point_i3 < 0 || point_i3 >= params.positions(geo_i3).size())
    {
      return;
    }

    const SegmentContactResult result = evaluate_segment_contact(
        params.position(geo_i0, point_i0),
        params.position(geo_i1, point_i1),
        params.position(geo_i2, point_i2),
        params.position(geo_i3, point_i3),
        inv_masses_[global_i0],
        inv_masses_[global_i1],
        inv_masses_[global_i2],
        inv_masses_[global_i3],
        min_distances_[constraint_i],
        compliances_[constraint_i] * params.compliance_term_factor,
        lambdas_[constraint_i]);
    if (!result.is_active) {
      return;
    }

    lambdas_[constraint_i] += result.delta_lambda;
    updater.update_position(geo_i0, point_i0, result.offset0);
    updater.update_position(geo_i1, point_i1, result.offset1);
    updater.update_position(geo_i2, point_i2, result.offset2);
    updater.update_position(geo_i3, point_i3, result.offset3);
    updater.add_residual_error(geo_i0, result.residual_error_squared * error_scale_);
  }

  ConstraintColoring color_constraints(LinearAllocator<> &memory) const override
  {
    return color_constraints__binary(segment_pairs_, memory);
  }
};

}  // namespace blender::xpbd
