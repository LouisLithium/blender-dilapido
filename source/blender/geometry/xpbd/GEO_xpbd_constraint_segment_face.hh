/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "BLI_offset_indices.hh"

#include "GEO_xpbd_constraint_coloring_utils.hh"
#include "GEO_xpbd_constraint_set_templated.hh"

namespace blender::xpbd {

struct SegmentFaceContactResult {
  /** Whether the segment is closer to the triangle than their minimum distance. */
  bool is_active = false;
  float delta_lambda = 0.0f;
  float3 offset_a = float3(0.0f);
  float3 offset_b = float3(0.0f);
  float3 offset0 = float3(0.0f);
  float3 offset1 = float3(0.0f);
  float3 offset2 = float3(0.0f);
  /** Squared penetration, zero when the constraint does not do anything. */
  float residual_error_squared = 0.0f;
};

/**
 * Unilateral distance constraint between a hair segment `p0-p1` and a triangle `v0-v1-v2`.
 *
 * The closest point on the segment (`s`) and on the triangle (`w`) are stored by the contact
 * gatherer, together with the direction that pushes the segment away from the surface. During the
 * iterations only the signed distance along that direction is evaluated, which is the same fixed
 * normal formulation the fluid uses for its point/triangle contacts.
 *
 * The correction is distributed to the two segment endpoints and the three triangle vertices with
 * their barycentric weights, so the constraint is momentum conserving and the hair pushes the
 * cloth/soft body like it would in a real contact.
 */
inline SegmentFaceContactResult evaluate_segment_face_contact(const float3 &p0,
                                                              const float3 &p1,
                                                              const float3 &v0,
                                                              const float3 &v1,
                                                              const float3 &v2,
                                                              const float3 &normal,
                                                              const float segment_s,
                                                              const float3 &face_weights,
                                                              const float inv_m_a,
                                                              const float inv_m_b,
                                                              const float inv_m0,
                                                              const float inv_m1,
                                                              const float inv_m2,
                                                              const float min_distance,
                                                              const float compliance_term,
                                                              const float lambda_prev)
{
  const float3 q_segment = math::interpolate(p0, p1, segment_s);
  const float3 q_face = v0 * face_weights.x + v1 * face_weights.y + v2 * face_weights.z;
  const float distance = math::dot(q_segment - q_face, normal);
  if (distance >= min_distance) {
    return {};
  }

  const float s = segment_s;
  const float w0 = face_weights.x;
  const float w1 = face_weights.y;
  const float w2 = face_weights.z;
  const float effective_inv_mass = (1.0f - s) * (1.0f - s) * inv_m_a + s * s * inv_m_b +
                                   w0 * w0 * inv_m0 + w1 * w1 * inv_m1 + w2 * w2 * inv_m2;
  if (!(effective_inv_mass + compliance_term > 0.0f)) {
    return {};
  }

  const float error = distance - min_distance;
  const float delta_lambda = (-error - compliance_term * lambda_prev) /
                             (effective_inv_mass + compliance_term);

  SegmentFaceContactResult result;
  result.is_active = true;
  result.delta_lambda = delta_lambda;
  result.offset_a = normal * ((1.0f - s) * inv_m_a * delta_lambda);
  result.offset_b = normal * (s * inv_m_b * delta_lambda);
  result.offset0 = normal * (-w0 * inv_m0 * delta_lambda);
  result.offset1 = normal * (-w1 * inv_m1 * delta_lambda);
  result.offset2 = normal * (-w2 * inv_m2 * delta_lambda);
  result.residual_error_squared = math::square(error);
  return result;
}

/**
 * Contacts between the hair segments and the triangles of the cloth and the soft bodies.
 *
 * Every contact is five points: the two segment endpoints followed by the three triangle vertices.
 * The constraints are colored by all five points (`color_constraints__n_ary`), because two contacts
 * that share a triangle vertex must not be solved in parallel.
 */
class SegmentFaceContactConstraintSet
    : public TemplatedConstraintSet<SegmentFaceContactConstraintSet> {
 private:
  GroupedSpan<int> points_;
  Span<float3> normals_;
  Span<float2> segment_weights_;
  Span<float3> face_weights_;
  Span<float> min_distances_;
  Span<float> compliances_;
  Span<int> point_geo_;
  Span<int> point_local_;
  Span<float> inv_masses_;
  float error_scale_;
  MutableSpan<float> lambdas_;

 public:
  static constexpr StringRefNull debug_name = "Segment Face Contact";

  SegmentFaceContactConstraintSet(const Vector<int> &affected_geo_indices,
                                  const GroupedSpan<int> points,
                                  const Span<float3> normals,
                                  const Span<float2> segment_weights,
                                  const Span<float3> face_weights,
                                  const Span<float> min_distances,
                                  const Span<float> compliances,
                                  const Span<int> point_geo,
                                  const Span<int> point_local,
                                  const Span<float> inv_masses,
                                  const float error_scale,
                                  MutableSpan<float> lambdas)
      : TemplatedConstraintSet<SegmentFaceContactConstraintSet>(points.size(),
                                                                affected_geo_indices),
        points_(points),
        normals_(normals),
        segment_weights_(segment_weights),
        face_weights_(face_weights),
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
    const Span<int> points = points_[constraint_i];
    if (points.size() != 5) {
      return;
    }
    for (const int point : points) {
      if (point < 0 || point >= point_geo_.size()) {
        return;
      }
    }
    const int ga = point_geo_[points[0]];
    const int gb = point_geo_[points[1]];
    const int g0 = point_geo_[points[2]];
    const int g1 = point_geo_[points[3]];
    const int g2 = point_geo_[points[4]];
    const int ia = point_local_[points[0]];
    const int ib = point_local_[points[1]];
    const int i0 = point_local_[points[2]];
    const int i1 = point_local_[points[3]];
    const int i2 = point_local_[points[4]];
    const int geometries_num = params.geometry_refs().size();
    for (const int geo_i : {ga, gb, g0, g1, g2}) {
      if (geo_i < 0 || geo_i >= geometries_num) {
        return;
      }
    }
    if (ia < 0 || ia >= params.positions(ga).size() || ib < 0 || ib >= params.positions(gb).size() ||
        i0 < 0 || i0 >= params.positions(g0).size() || i1 < 0 || i1 >= params.positions(g1).size() ||
        i2 < 0 || i2 >= params.positions(g2).size())
    {
      return;
    }

    const SegmentFaceContactResult result = evaluate_segment_face_contact(
        params.position(ga, ia),
        params.position(gb, ib),
        params.position(g0, i0),
        params.position(g1, i1),
        params.position(g2, i2),
        normals_[constraint_i],
        segment_weights_[constraint_i].x,
        face_weights_[constraint_i],
        inv_masses_[points[0]],
        inv_masses_[points[1]],
        inv_masses_[points[2]],
        inv_masses_[points[3]],
        inv_masses_[points[4]],
        min_distances_[constraint_i],
        compliances_[constraint_i] * params.compliance_term_factor,
        lambdas_[constraint_i]);
    if (!result.is_active) {
      return;
    }

    lambdas_[constraint_i] += result.delta_lambda;
    updater.update_position(ga, ia, result.offset_a);
    updater.update_position(gb, ib, result.offset_b);
    updater.update_position(g0, i0, result.offset0);
    updater.update_position(g1, i1, result.offset1);
    updater.update_position(g2, i2, result.offset2);
    updater.add_residual_error(ga, result.residual_error_squared * error_scale_);
  }

  ConstraintColoring color_constraints(LinearAllocator<> &memory) const override
  {
    return color_constraints__n_ary(points_, memory);
  }
};

}  // namespace blender::xpbd
