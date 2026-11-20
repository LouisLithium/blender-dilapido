/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <cmath>

#include "BLI_offset_indices.hh"

#include "GEO_xpbd_constraint_coloring_utils.hh"
#include "GEO_xpbd_constraint_set_templated.hh"

namespace blender::xpbd {

/**
 * Row major 3x3 matrix. The storage layout of `float3x3` is column major, which is easy to mix up
 * in the middle of a least squares fit, so the fit below works on plain arrays and only converts
 * the final rotation.
 */
struct ShapeMatchingMatrix {
  float m[3][3];
};

/** Multiplies a row major 3x3 matrix with a vector. */
inline float3 shape_matching_mul(const ShapeMatchingMatrix &matrix, const float3 &vector)
{
  return float3(matrix.m[0][0] * vector.x + matrix.m[0][1] * vector.y +
                    matrix.m[0][2] * vector.z,
                matrix.m[1][0] * vector.x + matrix.m[1][1] * vector.y +
                    matrix.m[1][2] * vector.z,
                matrix.m[2][0] * vector.x + matrix.m[2][1] * vector.y +
                    matrix.m[2][2] * vector.z);
}

/**
 * The rotation that maps the rest offsets onto the current offsets, for the covariance
 * `sum(m * (p - center) * (q - rest_center)^T)` (Horn 1987).
 *
 * This is the same closed form the `Point Deform` node uses: the largest eigenvector of the
 * symmetric 4x4 matrix built from the covariance is the rotation as a quaternion.
 */
inline void shape_matching_rotation(const ShapeMatchingMatrix &covariance,
                                    ShapeMatchingMatrix &r_rotation)
{
  const auto at = [&](const int row, const int column) { return covariance.m[row][column]; };

  float n[4][4] = {
      {at(0, 0) + at(1, 1) + at(2, 2),
       at(1, 2) - at(2, 1),
       at(2, 0) - at(0, 2),
       at(0, 1) - at(1, 0)},
      {at(1, 2) - at(2, 1),
       at(0, 0) - at(1, 1) - at(2, 2),
       at(0, 1) + at(1, 0),
       at(2, 0) + at(0, 2)},
      {at(2, 0) - at(0, 2),
       at(0, 1) + at(1, 0),
       -at(0, 0) + at(1, 1) - at(2, 2),
       at(1, 2) + at(2, 1)},
      {at(0, 1) - at(1, 0),
       at(2, 0) + at(0, 2),
       at(1, 2) + at(2, 1),
       -at(0, 0) - at(1, 1) + at(2, 2)},
  };
  float vectors[4][4] = {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f},
  };
  /* Cyclic Jacobi: a handful of sweeps diagonalizes a 4x4 matrix, and a fixed number of them keeps
   * the result independent of the machine. */
  for (int sweep = 0; sweep < 16; sweep++) {
    float off_diagonal = 0.0f;
    for (int p = 0; p < 4; p++) {
      for (int q = p + 1; q < 4; q++) {
        off_diagonal += n[p][q] * n[p][q];
      }
    }
    if (off_diagonal < 1e-14f) {
      break;
    }
    for (int p = 0; p < 4; p++) {
      for (int q = p + 1; q < 4; q++) {
        if (std::abs(n[p][q]) < 1e-14f) {
          continue;
        }
        const float theta = (n[q][q] - n[p][p]) / (2.0f * n[p][q]);
        const float sign = theta >= 0.0f ? 1.0f : -1.0f;
        const float t = sign / (std::abs(theta) + math::sqrt(theta * theta + 1.0f));
        const float c = 1.0f / math::sqrt(t * t + 1.0f);
        const float s = t * c;
        for (int k = 0; k < 4; k++) {
          const float nkp = n[k][p];
          const float nkq = n[k][q];
          n[k][p] = c * nkp - s * nkq;
          n[k][q] = s * nkp + c * nkq;
        }
        for (int k = 0; k < 4; k++) {
          const float npk = n[p][k];
          const float nqk = n[q][k];
          n[p][k] = c * npk - s * nqk;
          n[q][k] = s * npk + c * nqk;
        }
        for (int k = 0; k < 4; k++) {
          const float vkp = vectors[k][p];
          const float vkq = vectors[k][q];
          vectors[k][p] = c * vkp - s * vkq;
          vectors[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  int best = 0;
  for (int i = 1; i < 4; i++) {
    if (n[i][i] > n[best][best]) {
      best = i;
    }
  }
  const float w = vectors[0][best];
  const float x = vectors[1][best];
  const float y = vectors[2][best];
  const float z = vectors[3][best];
  const float length = math::sqrt(w * w + x * x + y * y + z * z);
  if (!(length > 0.0f) || !std::isfinite(length)) {
    for (int row = 0; row < 3; row++) {
      for (int column = 0; column < 3; column++) {
        r_rotation.m[row][column] = row == column ? 1.0f : 0.0f;
      }
    }
    return;
  }
  const float iw = w / length;
  const float ix = x / length;
  const float iy = y / length;
  const float iz = z / length;

  r_rotation.m[0][0] = 1.0f - 2.0f * (iy * iy + iz * iz);
  r_rotation.m[0][1] = 2.0f * (ix * iy + iw * iz);
  r_rotation.m[0][2] = 2.0f * (ix * iz - iw * iy);
  r_rotation.m[1][0] = 2.0f * (ix * iy - iw * iz);
  r_rotation.m[1][1] = 1.0f - 2.0f * (ix * ix + iz * iz);
  r_rotation.m[1][2] = 2.0f * (iy * iz + iw * ix);
  r_rotation.m[2][0] = 2.0f * (ix * iz + iw * iy);
  r_rotation.m[2][1] = 2.0f * (iy * iz - iw * ix);
  r_rotation.m[2][2] = 1.0f - 2.0f * (ix * ix + iy * iy);
}

/**
 * Keeps a group of points in a least squares best fit shape (Mueller et al. 2005).
 *
 * The constraint is one group of points, not a pair: every solve computes the optimal rotation
 * between the rest offsets and the current offsets, and moves each point towards the rotated rest
 * pose. The weight of a point is its mass, so the correction does not move the center of mass.
 *
 * `stiffness` is a per iteration projection strength in [0, 1]: 1 makes the group rigid in one
 * solve, smaller values let it deform. This is a position based projection, not an XPBD compliance,
 * so the effective stiffness also depends on the number of constraint iterations.
 */
class ShapeMatchingConstraintSet : public TemplatedConstraintSet<ShapeMatchingConstraintSet> {
 private:
  OffsetIndices<int> group_offsets_;
  Span<int> points_;
  /** Rest positions relative to the group's rest center of mass. */
  Span<float3> rest_offsets_;
  Span<float> masses_;
  Span<float> stiffnesses_;
  /** Root mean square distance of the rest points to the rest center of mass. */
  Span<float> rest_radii_;

 public:
  static constexpr StringRefNull debug_name = "Shape Matching Constraint";

  ShapeMatchingConstraintSet(const int geo_i,
                             const OffsetIndices<int> group_offsets,
                             const Span<int> points,
                             const Span<float3> rest_offsets,
                             const Span<float> masses,
                             const Span<float> stiffnesses,
                             const Span<float> rest_radii)
      : TemplatedConstraintSet<ShapeMatchingConstraintSet>(group_offsets.size(),
                                                           {geo_i}),
        group_offsets_(group_offsets),
        points_(points),
        rest_offsets_(rest_offsets),
        masses_(masses),
        stiffnesses_(stiffnesses),
        rest_radii_(rest_radii)
  {
  }

  void reset_force(const int /*constraint_i*/) const {}

  template<typename UpdaterT>
  void solve_single(const ConstraintSetParams &params,
                    UpdaterT &updater,
                    const int group_i) const
  {
    const int geo_i = affected_geo_indices_[0];
    const IndexRange group = group_offsets_[group_i];
    if (group.is_empty()) {
      return;
    }
    const float rest_radius = rest_radii_[group_i];
    if (!(rest_radius > 1e-6f)) {
      return;
    }

    float total_mass = 0.0f;
    float3 center(0.0f);
    for (const int k : group) {
      const float mass = masses_[k];
      total_mass += mass;
      center += mass * params.position(geo_i, points_[k]);
    }
    if (!(total_mass > 0.0f)) {
      return;
    }
    center /= total_mass;

    ShapeMatchingMatrix covariance{};
    for (const int k : group) {
      const float3 current = params.position(geo_i, points_[k]) - center;
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

    /* A single solve can not move a point further than half of the group radius. An ill conditioned
     * covariance (a nearly collinear group) can otherwise produce a large artificial rotation. */
    const float max_correction = 0.5f * rest_radius;
    float error_squared = 0.0f;
    for (const int k : group) {
      const int point_i = points_[k];
      const float3 current = params.position(geo_i, point_i);
      const float3 goal = shape_matching_mul(rotation, rest_offsets_[k]) + center;
      const float3 remaining = goal - current;
      error_squared += math::length_squared(remaining);
      float3 offset = remaining * stiffnesses_[k];
      const float offset_length = math::length(offset);
      if (offset_length > max_correction) {
        offset *= max_correction / offset_length;
      }
      updater.update_position(geo_i, point_i, offset);
    }

    /* The residual is relative to the size of the group, so a large and a small piece contribute the
     * same amount to the solver's average error. */
    const float mean_error = error_squared / (float(group.size()) * rest_radius * rest_radius);
    updater.add_residual_error(geo_i, mean_error);
  }

  ConstraintColoring color_constraints(LinearAllocator<> & /*memory*/) const override
  {
    /* Every point belongs to exactly one group, so the groups are independent and can all be solved
     * at the same time. */
    return color_constraints__all_independent(constraints_num_);
  }
};

}  // namespace blender::xpbd
