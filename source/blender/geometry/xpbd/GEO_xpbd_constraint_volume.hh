/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "BLI_offset_indices.hh"

#include "GEO_xpbd_constraint_coloring_utils.hh"
#include "GEO_xpbd_constraint_set_templated.hh"

namespace blender::xpbd {

/**
 * Volume of the tetrahedron \a p0 \a p1 \a p2 \a p3, positive when the four points are wound so that
 * the volume is positive (like the faces that `Tetrahedralize` writes).
 */
inline float tetrahedron_volume(const float3 &p0,
                                const float3 &p1,
                                const float3 &p2,
                                const float3 &p3)
{
  return math::dot(p1 - p0, math::cross(p2 - p0, p3 - p0)) / 6.0f;
}

struct VolumeConstraintResult {
  float delta_lambda = 0.0f;
  float3 offset0 = float3(0.0f);
  float3 offset1 = float3(0.0f);
  float3 offset2 = float3(0.0f);
  float3 offset3 = float3(0.0f);
  /** Squared relative error, i.e. it does not depend on the size of the element. */
  float residual_error_squared = 0.0f;
};

/**
 * One XPBD volume constraint of a tetrahedron (Macklin et al. 2016).
 *
 * The volume of a tetrahedron is linear in each of its points, so the gradient with respect to a
 * point is the cross product of the two edges that do not touch it (divided by six), which makes
 * this one of the cheapest volume constraints there is. The constraint is normalized by the rest
 * volume, so that the compliance (which is in m^5/N when the constraint is left unnormalized) and
 * the residual error mean the same thing for a small and for a large element.
 */
inline VolumeConstraintResult evaluate_volume_constraint(const float3 &p0,
                                                         const float3 &p1,
                                                         const float3 &p2,
                                                         const float3 &p3,
                                                         const float inv_m0,
                                                         const float inv_m1,
                                                         const float inv_m2,
                                                         const float inv_m3,
                                                         const float rest_volume,
                                                         const float compliance_term,
                                                         const float lambda_prev)
{
  if (inv_m0 + inv_m1 + inv_m2 + inv_m3 == 0.0f || rest_volume <= 0.0f) {
    return {};
  }

  const float3 a = p1 - p0;
  const float3 b = p2 - p0;
  const float3 c = p3 - p0;
  const float inv_rest_volume = math::safe_rcp(rest_volume);

  /* Gradient of the volume with respect to each point. */
  const float3 gradient1 = math::cross(b, c) * (inv_rest_volume / 6.0f);
  const float3 gradient2 = math::cross(c, a) * (inv_rest_volume / 6.0f);
  const float3 gradient3 = math::cross(a, b) * (inv_rest_volume / 6.0f);
  const float3 gradient0 = -(gradient1 + gradient2 + gradient3);

  /* A soft constraint would otherwise be able to ask for a negative volume, which turns the element
   * inside out; ten percent of the rest volume is the floor. */
  const float volume_error = math::max(
      (math::dot(a, math::cross(b, c)) / 6.0f - rest_volume) * inv_rest_volume, -0.9f);

  const float denominator = inv_m0 * math::length_squared(gradient0) +
                            inv_m1 * math::length_squared(gradient1) +
                            inv_m2 * math::length_squared(gradient2) +
                            inv_m3 * math::length_squared(gradient3) + compliance_term;
  if (!(denominator > 0.0f)) {
    /* Fully degenerate element (collinear points have no gradient at all). */
    return {};
  }
  const float delta_lambda = (-volume_error - compliance_term * lambda_prev) / denominator;

  float3 offset0 = delta_lambda * inv_m0 * gradient0;
  float3 offset1 = delta_lambda * inv_m1 * gradient1;
  float3 offset2 = delta_lambda * inv_m2 * gradient2;
  float3 offset3 = delta_lambda * inv_m3 * gradient3;

  /* A nearly flat element (the surface snap of `Tetrahedralize` can leave slivers) has a tiny
   * gradient and would get a huge correction out of its tiny denominator. Bounding the movement by
   * the size of the element keeps that from turning into a NaN; a healthy element never comes close
   * to the limit. */
  const float size = std::max(
      std::max(std::max(math::distance(p0, p1), math::distance(p0, p2)),
               std::max(math::distance(p0, p3), math::distance(p1, p2))),
      std::max(math::distance(p1, p3), math::distance(p2, p3)));
  const float limit = 0.5f * size;
  const float movement = std::max(std::max(math::length(offset0), math::length(offset1)),
                                  std::max(math::length(offset2), math::length(offset3)));
  float scale = 1.0f;
  if (movement > limit) {
    scale = limit / movement;
    offset0 *= scale;
    offset1 *= scale;
    offset2 *= scale;
    offset3 *= scale;
  }

  return {delta_lambda * scale,
          offset0,
          offset1,
          offset2,
          offset3,
          math::square(volume_error)};
}

/**
 * The volume constraints of a tetrahedral mesh, one per element.
 *
 * \param elements: Four point indices per element, e.g. the corners of the four cornered faces that
 * `Tetrahedralize` writes.
 */
class VolumeConstraintSet : public TemplatedConstraintSet<VolumeConstraintSet> {
 private:
  GroupedSpan<int> elements_;
  Span<float> rest_volumes_;
  Span<float> compliances_;
  /* Scale factor for residual error. */
  float error_scale_;
  MutableSpan<float> lambdas_;

 public:
  static constexpr StringRefNull debug_name = "Volume Constraint";

  VolumeConstraintSet(const int geo_i,
                      const GroupedSpan<int> elements,
                      const Span<float> rest_volumes,
                      const Span<float> compliances,
                      const float error_scale,
                      const MutableSpan<float> lambdas)
      : TemplatedConstraintSet<VolumeConstraintSet>(elements.size(), {geo_i}),
        elements_(elements),
        rest_volumes_(rest_volumes),
        compliances_(compliances),
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
    const int geo_i = affected_geo_indices_[0];
    const Span<int> element = elements_[constraint_i];
    if (element.size() != 4) {
      return;
    }
    const int point_i0 = element[0];
    const int point_i1 = element[1];
    const int point_i2 = element[2];
    const int point_i3 = element[3];

    const VolumeConstraintResult result = evaluate_volume_constraint(
        params.position(geo_i, point_i0),
        params.position(geo_i, point_i1),
        params.position(geo_i, point_i2),
        params.position(geo_i, point_i3),
        params.inv_mass(geo_i, point_i0),
        params.inv_mass(geo_i, point_i1),
        params.inv_mass(geo_i, point_i2),
        params.inv_mass(geo_i, point_i3),
        rest_volumes_[constraint_i],
        compliances_[constraint_i] * params.compliance_term_factor,
        lambdas_[constraint_i]);

    lambdas_[constraint_i] += result.delta_lambda;
    updater.update_position(geo_i, point_i0, result.offset0);
    updater.update_position(geo_i, point_i1, result.offset1);
    updater.update_position(geo_i, point_i2, result.offset2);
    updater.update_position(geo_i, point_i3, result.offset3);
    updater.add_residual_error(geo_i, result.residual_error_squared * error_scale_);
  }

  ConstraintColoring color_constraints(LinearAllocator<> &memory) const override
  {
    return color_constraints__n_ary(elements_, memory);
  }
};

}  // namespace blender::xpbd
