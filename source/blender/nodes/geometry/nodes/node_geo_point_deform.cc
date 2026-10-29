/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo_nodes
 *
 * `Point Deform` transfers a deformation from one point set (the "lattice") to another geometry,
 * which is what makes it possible to simulate a cheap proxy and render an expensive mesh: the
 * animated proxy is the deformed lattice, a copy of it in its rest pose is the rest lattice, and the
 * high resolution mesh is the geometry to deform (the same idea as the Point Deform SOP of Houdini).
 *
 * For every point of the target the closest lattice points capture it, and the weighted best fit
 * between their rest and their deformed positions gives a local transform that is applied to the
 * point. Three fits are available:
 *
 * - `Rigid`: rotation and translation only (Horn's quaternion method). A rotation is transferred
 *   correctly, squash and stretch are not.
 * - `Rigid + Scale`: the same plus one uniform scale. This is the default, because a body that both
 *   turns and swells is the common case and the fit stays a similarity (no shear noise).
 * - `Affine`: the full linear map (least squares). It also carries shear and one sided squash, at
 *   the price of amplifying noise when few points are captured.
 *
 * A transform fitted from the k nearest lattice points alone is wrong wherever two parts of the
 * lattice are close in space but move differently (a bent elbow, two arms): points of both parts end
 * up in the same fit and the joint collapses. So when the rest lattice is a mesh, only the points of
 * the connected island of the nearest lattice point capture (Houdini does the same, and it is what
 * makes rotations correct there). `Use Connectivity` turns that off.
 */

#include <array>

#include "BLI_array.hh"
#include "BLI_kdtree.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_curves.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "BLT_translation.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_point_deform_cc {

/** How the local deformation is fitted from the captured lattice points. */
enum class DeformMode : int {
  /** Rotation and translation. */
  Rigid = 0,
  /** Rotation, translation and one uniform scale. */
  RigidScale = 1,
  /** Full linear map. */
  Affine = 2,
};

static const EnumPropertyItem deform_mode_items[] = {
    {int(DeformMode::Rigid),
     "RIGID",
     0,
     N_("Rigid"),
     N_("Rotate and translate only. A rotation is transferred correctly, squash and stretch are "
        "not")},
    {int(DeformMode::RigidScale),
     "RIGID_SCALE",
     0,
     N_("Rigid + Scale"),
     N_("Rotate, translate and scale uniformly. Handles a body that turns and swells without "
        "introducing shear")},
    {int(DeformMode::Affine),
     "AFFINE",
     0,
     N_("Affine"),
     N_("Full linear transform, which also carries shear and one sided squash, but amplifies noise "
        "when few points are captured")},
    {0, nullptr, 0, nullptr, nullptr},
};

/** Upper bound for the candidates of one target point, i.e. the scratch of one thread. */
constexpr int max_candidates = 512;

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr)
      .supported_type({GeometryComponent::Type::Mesh,
                       GeometryComponent::Type::PointCloud,
                       GeometryComponent::Type::Curve})
      .is_default_link_socket()
      .description("Geometry to deform, usually the high resolution mesh");
  b.add_input<decl::Geometry>("Rest Points"_ustr)
      .supported_type({GeometryComponent::Type::Mesh,
                       GeometryComponent::Type::PointCloud,
                       GeometryComponent::Type::Curve})
      .description("The lattice in its rest pose. It has to correspond to `Deformed Points`");
  b.add_input<decl::Geometry>("Deformed Points"_ustr)
      .supported_type({GeometryComponent::Type::Mesh,
                       GeometryComponent::Type::PointCloud,
                       GeometryComponent::Type::Curve})
      .description("The same lattice in its deformed pose. The difference to `Rest Points` is the "
                   "deformation that is transferred");
  {
    auto &panel = b.add_panel("Capture"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Radius"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description(
            "Maximum distance of a lattice point that captures a target point. 0 uses the distance "
            "of the farthest of `Maximum Points`");
    panel.add_input<decl::Int>("Minimum Points"_ustr)
        .default_value(1)
        .min(1)
        .max(64)
        .description(
            "Take at least this many lattice points, even when they are farther away than the "
            "radius, so that nothing is left behind (a jump, but usually better than a hole)");
    panel.add_input<decl::Int>("Maximum Points"_ustr)
        .default_value(8)
        .min(1)
        .max(64)
        .description("Upper bound for the captured points of one target point");
    panel.add_input<decl::Float>("Falloff"_ustr)
        .default_value(2.0f)
        .min(0.0f)
        .max(8.0f)
        .description(
            "Exponent of the smooth falloff of the capture weight, higher values make the closest "
            "lattice point dominate");
    panel.add_input<decl::Bool>("Use Connectivity"_ustr)
        .default_value(true)
        .description(
            "Only let the lattice points of the connected island of the nearest point capture "
            "(when the rest lattice is a mesh). Without it a transform can mix two parts that are "
            "close in space but move differently, which collapses a bent joint");
  }
  {
    auto &panel = b.add_panel("Deform"_ustr).default_closed(false);
    panel.add_input<decl::Menu>("Mode"_ustr)
        .static_items(deform_mode_items)
        .default_value(MenuValue(int(DeformMode::RigidScale)))
        .description("Kind of transform that is fitted to the captured lattice points");
    panel.add_input<decl::Float>("Blend"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .max(1.0f)
        .description("Blend between the rest deformation (0) and the full deformation (1)");
  }
  b.add_output<decl::Geometry>("Geometry"_ustr).propagate_all();
}

/** The points of a lattice input, with the optional `id` attribute that gives the correspondence. */
struct LatticePoints {
  Span<float3> positions;
  VArray<int> ids;
};

static LatticePoints get_lattice_points(const GeometrySet &geometry)
{
  if (const Mesh *mesh = geometry.get_mesh()) {
    const bke::AttributeAccessor attributes = mesh->attributes();
    return {mesh->vert_positions(), *attributes.lookup<int>("id", AttrDomain::Point)};
  }
  if (const PointCloud *pointcloud = geometry.get_pointcloud()) {
    const bke::AttributeAccessor attributes = pointcloud->attributes();
    return {pointcloud->positions(), *attributes.lookup<int>("id", AttrDomain::Point)};
  }
  if (const Curves *curves_id = geometry.get_curves()) {
    const bke::CurvesGeometry &curves = curves_id->geometry.wrap();
    const bke::AttributeAccessor attributes = curves.attributes();
    return {curves.positions(), *attributes.lookup<int>("id", AttrDomain::Point)};
  }
  return {};
}

/** The local transform of one target point: move the rest centroid onto the deformed one. */
struct LocalTransform {
  float3x3 linear;
  /** The same transform for normals: the rotation part, or the inverse transpose of #linear. */
  float3x3 normal;
  float3 rest_center;
  float3 deformed_center;

  float3 apply(const float3 &point) const
  {
    return linear * (point - rest_center) + deformed_center;
  }
};

static bool is_finite(const float3x3 &matrix)
{
  for (int row = 0; row < 3; row++) {
    for (int column = 0; column < 3; column++) {
      if (!std::isfinite(matrix[row][column])) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Matrices are handled as plain `float[3][3]` with the mathematical `[row][column]` order here: the
 * storage layout of `float3x3` is column major (`matrix[column][row]`), which is too easy to mix up
 * in the middle of a fit. The result is converted once, at the end.
 */
using Matrix3 = float[3][3];

/** The rotation that maps \a covariance (sum of w * deformed * rest^T) onto the identity (Horn 1987). */
static void best_fit_rotation(const Matrix3 &covariance, Matrix3 &r_rotation)
{
  const auto at = [&](const int row, const int column) { return covariance[row][column]; };

  /* The symmetric 4x4 matrix whose largest eigenvector is the rotation as a quaternion. */
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
        r_rotation[row][column] = row == column ? 1.0f : 0.0f;
      }
    }
    return;
  }
  const float iw = w / length;
  const float ix = x / length;
  const float iy = y / length;
  const float iz = z / length;

  r_rotation[0][0] = 1.0f - 2.0f * (iy * iy + iz * iz);
  r_rotation[0][1] = 2.0f * (ix * iy + iw * iz);
  r_rotation[0][2] = 2.0f * (ix * iz - iw * iy);
  r_rotation[1][0] = 2.0f * (ix * iy - iw * iz);
  r_rotation[1][1] = 1.0f - 2.0f * (ix * ix + iz * iz);
  r_rotation[1][2] = 2.0f * (iy * iz + iw * ix);
  r_rotation[2][0] = 2.0f * (ix * iz + iw * iy);
  r_rotation[2][1] = 2.0f * (iy * iz - iw * ix);
  r_rotation[2][2] = 1.0f - 2.0f * (ix * ix + iy * iy);
}

/** Multiplies a plain 3x3 matrix with a vector. */
static float3 mul_matrix3(const Matrix3 &matrix, const float3 &vector)
{
  return float3(matrix[0][0] * vector.x + matrix[0][1] * vector.y + matrix[0][2] * vector.z,
                matrix[1][0] * vector.x + matrix[1][1] * vector.y + matrix[1][2] * vector.z,
                matrix[2][0] * vector.x + matrix[2][1] * vector.y + matrix[2][2] * vector.z);
}

/** Inverts a 3x3 matrix, returns false when it is singular. */
static bool invert_matrix3(const Matrix3 &matrix, Matrix3 &r_inverse)
{
  const float determinant =
      matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
      matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
      matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
  if (!(std::abs(determinant) > 1e-20f) || !std::isfinite(determinant)) {
    return false;
  }
  const float inv_det = 1.0f / determinant;
  for (int row = 0; row < 3; row++) {
    for (int column = 0; column < 3; column++) {
      const int r0 = (row + 1) % 3;
      const int r1 = (row + 2) % 3;
      const int c0 = (column + 1) % 3;
      const int c1 = (column + 2) % 3;
      /* Transposed cofactor matrix. */
      const float cofactor = matrix[r0][c0] * matrix[r1][c1] - matrix[r0][c1] * matrix[r1][c0];
      r_inverse[row][column] = cofactor * inv_det;
    }
  }
  return true;
}

/** Converts the mathematical `[row][column]` matrix to the column major `float3x3`. */
static float3x3 to_matrix3x3(const Matrix3 &matrix)
{
  float3x3 result;
  for (int row = 0; row < 3; row++) {
    for (int column = 0; column < 3; column++) {
      result[column][row] = matrix[row][column];
    }
  }
  return result;
}

static bool is_finite(const Matrix3 &matrix)
{
  for (int row = 0; row < 3; row++) {
    for (int column = 0; column < 3; column++) {
      if (!std::isfinite(matrix[row][column])) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Fits the local transform of one target point.
 *
 * \param rest_offsets: rest positions of the captured points, relative to \a rest_center.
 * \param deformed_offsets: the same points after the deformation, relative to \a deformed_center.
 */
static LocalTransform fit_transform(const Span<float3> rest_offsets,
                                    const Span<float3> deformed_offsets,
                                    const Span<float> weights,
                                    const DeformMode mode,
                                    const float3 &rest_center,
                                    const float3 &deformed_center)
{
  /* Deformed as the rows and rest as the columns: the correlation matrix whose rotation maps rest
   * onto deformed, and whose product with the inverse of the rest covariance is the affine map. */
  Matrix3 covariance{};
  Matrix3 rest_covariance{};
  float rest_norm = 0.0f;
  for (const int i : rest_offsets.index_range()) {
    const float3 &r = rest_offsets[i];
    const float3 &d = deformed_offsets[i];
    const float w = weights[i];
    for (int row = 0; row < 3; row++) {
      for (int column = 0; column < 3; column++) {
        covariance[row][column] += w * d[row] * r[column];
        rest_covariance[row][column] += w * r[row] * r[column];
      }
    }
    rest_norm += w * math::length_squared(r);
  }

  LocalTransform transform{
      float3x3::identity(), float3x3::identity(), rest_center, deformed_center};

  /* A single point, or all of them on top of each other: only a translation can be fit. */
  if (!(rest_norm > 1e-12f)) {
    return transform;
  }

  if (mode == DeformMode::Affine) {
    /* Regularize, so that a flat capture region does not make the matrix singular. */
    const float epsilon = 1e-5f * rest_norm;
    for (int axis = 0; axis < 3; axis++) {
      rest_covariance[axis][axis] += epsilon;
    }
    Matrix3 inverse;
    if (!invert_matrix3(rest_covariance, inverse)) {
      return transform;
    }
    Matrix3 linear{};
    for (int row = 0; row < 3; row++) {
      for (int column = 0; column < 3; column++) {
        for (int k = 0; k < 3; k++) {
          linear[row][column] += covariance[row][k] * inverse[k][column];
        }
      }
    }
    if (!is_finite(linear)) {
      return transform;
    }
    transform.linear = to_matrix3x3(linear);
    /* Normals follow the inverse transpose: a shear or a one sided scale would otherwise turn them
     * the wrong way. */
    Matrix3 inverse_linear;
    if (invert_matrix3(linear, inverse_linear)) {
      Matrix3 transposed{};
      for (int row = 0; row < 3; row++) {
        for (int column = 0; column < 3; column++) {
          transposed[row][column] = inverse_linear[column][row];
        }
      }
      transform.normal = to_matrix3x3(transposed);
    }
    else {
      transform.normal = transform.linear;
    }
    return transform;
  }

  Matrix3 rotation;
  best_fit_rotation(covariance, rotation);
  if (mode == DeformMode::Rigid) {
    transform.linear = to_matrix3x3(rotation);
    transform.normal = transform.linear;
    return transform;
  }

  /* The uniform scale along the rotation: the least squares solution of d = s * R * r. */
  float dot = 0.0f;
  for (const int i : rest_offsets.index_range()) {
    dot += weights[i] * math::dot(deformed_offsets[i], mul_matrix3(rotation, rest_offsets[i]));
  }
  const float scale = math::clamp(dot / rest_norm, 1e-3f, 1e3f);
  Matrix3 scaled;
  for (int row = 0; row < 3; row++) {
    for (int column = 0; column < 3; column++) {
      scaled[row][column] = rotation[row][column] * scale;
    }
  }
  transform.linear = to_matrix3x3(scaled);
  /* A uniform scale does not change the direction of a normal. */
  transform.normal = to_matrix3x3(rotation);
  return transform;
}
/** The island (connected component) of every vertex of the rest lattice, over its edges. */
static Array<int> vertex_islands(const Mesh &mesh)
{
  const int verts_num = mesh.verts_num;
  Array<int> parent(verts_num);
  for (const int i : IndexRange(verts_num)) {
    parent[i] = i;
  }
  const auto find_root = [&](int index) {
    while (parent[index] != index) {
      parent[index] = parent[parent[index]];
      index = parent[index];
    }
    return index;
  };
  for (const int2 &edge : mesh.edges()) {
    const int root_a = find_root(edge[0]);
    const int root_b = find_root(edge[1]);
    if (root_a != root_b) {
      parent[root_b] = root_a;
    }
  }
  Array<int> islands(verts_num);
  for (const int i : IndexRange(verts_num)) {
    islands[i] = find_root(i);
  }
  return islands;
}

/** How many vertices the island of every vertex has. */
static Array<int> island_sizes(const Span<int> islands)
{
  Array<int> sizes(islands.size(), 0);
  for (const int island : islands) {
    sizes[island]++;
  }
  Array<int> vertex_sizes(islands.size());
  for (const int i : islands.index_range()) {
    vertex_sizes[i] = sizes[islands[i]];
  }
  return vertex_sizes;
}

/** Captures lattice points and fits the local transform of a target point. */
class PointDeformer {
 private:
  const Span<float3> rest_positions_;
  const Span<float3> deformed_positions_;
  /** Index into #deformed_positions_ for each rest point. */
  const Span<int> deformed_index_;
  const KDTree<float3> *tree_;
  /** Island of every rest point, empty when the connectivity is not used. */
  const Span<int> islands_;
  /** Number of points of the island of every rest point. */
  const Span<int> island_sizes_;
  const int min_points_;
  const int max_points_;
  const float radius_;
  const float falloff_;
  const DeformMode mode_;


 public:
  PointDeformer(const Span<float3> rest_positions,
                const Span<float3> deformed_positions,
                const Span<int> deformed_index,
                const KDTree<float3> *tree,
                const Span<int> islands,
                const Span<int> island_sizes,
                const int min_points,
                const int max_points,
                const float radius,
                const float falloff,
                const DeformMode mode)
      : rest_positions_(rest_positions),
        deformed_positions_(deformed_positions),
        deformed_index_(deformed_index),
        tree_(tree),
        islands_(islands),
        island_sizes_(island_sizes),
        min_points_(min_points),
        max_points_(max_points),
        radius_(radius),
        falloff_(falloff),
        mode_(mode)
  {
  }

  /**
   * \param candidates: Scratch buffer with room for at least #max_candidates entries.
   * \return False when no lattice point captures the point (the caller leaves it alone).
   */
  bool transform_for(const float3 &point,
                     const MutableSpan<KDTreeNearest<float3>> candidates,
                     LocalTransform &r_transform) const
  {

    const int found = this->capture(point, candidates);
    if (found == 0) {
      return false;
    }

    /* The distance that is used to normalize the weights, i.e. the radius of the capture region. */
    float weight_radius = radius_;
    if (weight_radius <= 0.0f) {
      weight_radius = candidates[found - 1].dist;
    }

    /* Stack buffers: `Maximum Points` is capped at 64, so this is at most two kilobytes and no
     * target point allocates anything (allocating the three arrays per point alone cost fifteen
     * times the fit itself). */
    std::array<float3, 64> rest_storage;
    std::array<float3, 64> deformed_storage;
    std::array<float, 64> weight_storage;
    const MutableSpan<float3> rest_offsets(rest_storage.data(), found);
    const MutableSpan<float3> deformed_offsets(deformed_storage.data(), found);
    const MutableSpan<float> weights(weight_storage.data(), found);
    float3 rest_center(0.0f);
    float3 deformed_center(0.0f);
    float weight_sum = 0.0f;
    for (const int i : IndexRange(found)) {
      const int rest_index = candidates[i].index;
      const float3 rest_position = rest_positions_[rest_index];
      const float3 deformed_position = deformed_positions_[deformed_index_[rest_index]];
      /* Inverse distance, normalized by the capture radius: the closest point dominates but every
       * captured point still contributes. A compact support kernel would give the points at the
       * edge of the capture a weight of zero, which quietly turns a capture of eight points into a
       * fit of four and leaves the transform underdetermined. */
      const float normalized = weight_radius > 0.0f ? candidates[i].dist / weight_radius : 0.0f;
      const float weight = falloff_ == 0.0f ?
                               1.0f :
                               1.0f / (math::pow(normalized, falloff_) + 1e-3f);
      rest_offsets[i] = rest_position;
      deformed_offsets[i] = deformed_position;
      weights[i] = weight;
      rest_center += rest_position * weight;
      deformed_center += deformed_position * weight;
      weight_sum += weight;
    }

    if (!(weight_sum > 0.0f)) {
      /* Everything sits on the weight radius: use equal weights. */
      for (const int i : IndexRange(found)) {
        weights[i] = 1.0f;
        rest_center += rest_offsets[i];
        deformed_center += deformed_offsets[i];
      }
      weight_sum = float(found);
    }
    const float inv_weight_sum = 1.0f / weight_sum;
    rest_center *= inv_weight_sum;
    deformed_center *= inv_weight_sum;
    for (const int i : IndexRange(found)) {
      rest_offsets[i] -= rest_center;
      deformed_offsets[i] -= deformed_center;
    }

    r_transform = fit_transform(
        rest_offsets, deformed_offsets, weights, mode_, rest_center, deformed_center);

    return true;
  }

 private:
  /** Fills \a candidates with the captured lattice points and returns how many there are. */
  int capture(const float3 &point, const MutableSpan<KDTreeNearest<float3>> candidates) const
  {
    const int lattice_num = int(rest_positions_.size());
    if (lattice_num == 0) {
      return 0;
    }
    /* The query keeps a sorted list of this size, so asking for many more than needed makes every
     * query several times more expensive. */
    int capacity = math::min(lattice_num, max_points_);
    int found = kdtree_find_nearest_n(tree_, point, candidates.data(), uint(capacity));
    if (found == 0) {
      return 0;
    }

    if (!islands_.is_empty()) {
      const int nearest_index = candidates[0].index;
      /* A mesh of loose vertices has one point per island; a lone point cannot be the only source
       * of a transform (it would turn the whole deformation into a translation), so those fall back
       * to the plain neighbours. */
      if (island_sizes_[nearest_index] >= math::max(2, min_points_)) {
        const int island = islands_[nearest_index];
        while (true) {
          int same_island = 0;
          for (int i = 0; i < found && same_island < max_points_; i++) {
            if (islands_[candidates[i].index] != island) {
              continue;
            }
            candidates[same_island] = candidates[i];
            same_island++;
          }
          if (same_island >= min_points_ || capacity >= math::min(lattice_num, max_candidates)) {
            found = math::max(same_island, 1);
            break;
          }
          /* The island is spread out over space: ask for more candidates. */
          capacity = math::min(math::min(lattice_num, max_candidates), capacity * 4);
          found = kdtree_find_nearest_n(tree_, point, candidates.data(), uint(capacity));
          if (found == 0) {
            return 0;
          }
        }
      }
    }

    if (radius_ > 0.0f) {
      const int in_radius = this->cut_to_radius(candidates, found);
      /* `Minimum Points` grows the radius instead of leaving a point behind. */
      found = math::clamp(in_radius, math::min(min_points_, found), found);
    }
    return found;
  }
  int cut_to_radius(const MutableSpan<KDTreeNearest<float3>> candidates, const int found) const
  {
    int cut = 0;
    while (cut < found && candidates[cut].dist <= radius_) {
      cut++;
    }
    return cut;
  }
};

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  const GeometrySet rest_geometry = params.extract_input<GeometrySet>("Rest Points"_ustr);
  const GeometrySet deformed_geometry = params.extract_input<GeometrySet>("Deformed Points"_ustr);

  const int min_points = math::clamp(params.extract_input<int>("Minimum Points"_ustr), 1, 64);
  const int max_points = math::clamp(
      params.extract_input<int>("Maximum Points"_ustr), min_points, 64);
  const float radius = math::max(params.extract_input<float>("Radius"_ustr), 0.0f);
  const float falloff = math::max(params.extract_input<float>("Falloff"_ustr), 0.0f);
  const auto mode = DeformMode(params.extract_input<int>("Mode"_ustr));
  const float blend = math::clamp(params.extract_input<float>("Blend"_ustr), 0.0f, 1.0f);
  const bool use_connectivity = params.extract_input<bool>("Use Connectivity"_ustr);


  const LatticePoints rest = get_lattice_points(rest_geometry);
  const LatticePoints deformed = get_lattice_points(deformed_geometry);
  const int lattice_num = int(rest.positions.size());
  if (lattice_num == 0 || deformed.positions.is_empty()) {
    params.error_message_add(NodeWarningType::Warning,
                             TIP_("The lattice is empty, the geometry is not deformed"));
    params.set_output("Geometry"_ustr, std::move(geometry));
    return;
  }

  /* Correspondence: the `id` attribute when both lattices have one, the point order otherwise. */
  Array<int> deformed_index(lattice_num);
  if (!rest.ids.is_empty() && !deformed.ids.is_empty()) {
    Map<int, int> index_by_id;
    for (const int i : deformed.positions.index_range()) {
      index_by_id.add_overwrite(deformed.ids[i], i);
    }
    bool complete = true;
    for (const int i : IndexRange(lattice_num)) {
      const int *found = index_by_id.lookup_ptr(rest.ids[i]);
      if (found == nullptr) {
        complete = false;
        break;
      }
      deformed_index[i] = *found;
    }
    if (!complete) {
      params.error_message_add(
          NodeWarningType::Warning,
          TIP_("The `id` attributes of the two lattices do not match, the geometry is not "
               "deformed"));
      params.set_output("Geometry"_ustr, std::move(geometry));
      return;
    }
  }
  else {
    if (int(deformed.positions.size()) != lattice_num) {
      params.error_message_add(NodeWarningType::Warning,
                               TIP_("The two lattices have a different number of points and no "
                                    "`id` attribute to match them, the geometry is not deformed"));
      params.set_output("Geometry"_ustr, std::move(geometry));
      return;
    }
    for (const int i : IndexRange(lattice_num)) {
      deformed_index[i] = i;
    }
  }

  /* The connectivity of the rest lattice decides who captures; only a mesh has one. */
  Array<int> islands;
  if (use_connectivity) {
    if (const Mesh *rest_mesh = rest_geometry.get_mesh()) {
      islands = vertex_islands(*rest_mesh);
    }
  }

  KDTree<float3> *tree = kdtree_new<float3>(uint(lattice_num));
  for (const int i : IndexRange(lattice_num)) {
    kdtree_insert(tree, i, rest.positions[i]);
  }
  kdtree_balance(tree);

  const Array<int> island_vertex_sizes = islands.is_empty() ? Array<int>() : island_sizes(islands);

  const PointDeformer deformer(rest.positions,
                               deformed.positions,
                               deformed_index,
                               tree,
                               islands,
                               island_vertex_sizes,
                               min_points,
                               max_points,
                               radius,
                               falloff,
                               mode);

  /**
   * Deforms one array of points. \a r_normal_matrices is filled with the transform of every point
   * when it is not empty (the mesh path needs it for the custom normals).
   */
  const auto deform = [&](const MutableSpan<float3> positions,
                          const MutableSpan<float3> handle_left,
                          const MutableSpan<float3> handle_right,
                          const MutableSpan<float3x3> r_normal_matrices) {
    if (positions.is_empty()) {
      return;
    }
    const int candidates_num = math::min(lattice_num, math::max(max_points, max_candidates));
    threading::parallel_for(positions.index_range(), 1024, [&](const IndexRange range) {
      Array<KDTreeNearest<float3>> candidates(candidates_num);
      for (const int i : range) {
        LocalTransform transform;
        if (!deformer.transform_for(positions[i], candidates, transform)) {
          continue;
        }
        const auto apply = [&](const float3 &point) {
          const float3 deformed = transform.apply(point);
          return blend >= 1.0f ? deformed : math::interpolate(point, deformed, blend);
        };
        positions[i] = apply(positions[i]);
        if (!handle_left.is_empty()) {
          handle_left[i] = apply(handle_left[i]);
          handle_right[i] = apply(handle_right[i]);
        }
        if (!r_normal_matrices.is_empty()) {
          r_normal_matrices[i] = transform.normal;
        }
      }
    });
  };

  if (Mesh *mesh = geometry.get_mesh_for_write()) {
    const bool has_custom_normals = mesh->attributes().contains("custom_normal");
    Array<float3x3> normal_matrices(has_custom_normals ? mesh->verts_num : 0);
    deform(mesh->vert_positions_for_write(), {}, {}, normal_matrices);
    /* Writing the positions through the raw span does not tell the mesh that its derived data (the
     * normals above all) is out of date, so without this the deformed mesh is shaded with the
     * normals of its rest pose. */
    mesh->tag_positions_changed();

    if (has_custom_normals) {
      /* The generated normals are recalculated now, but a custom split normal set keeps whatever the
       * artist authored. It is stored as the offset of the normal relative to the face, so the
       * deformation carries it along by itself when the face turns; only the absolute form (what an
       * importer or an older file writes) has to be rotated with its vertex. */
      const GroupedSpan<int> vert_to_corner = mesh->vert_to_corner_map();
      bke::MutableAttributeAccessor attributes = mesh->attributes_for_write();
      bke::SpanAttributeWriter<float3> custom_normals = attributes.lookup_or_add_for_write_span<float3>(
          "custom_normal", AttrDomain::Corner);
      if (!custom_normals.span.is_empty()) {
        threading::parallel_for(IndexRange(mesh->verts_num), 1024, [&](const IndexRange range) {
          for (const int i : range) {
            const float3x3 &matrix = normal_matrices[i];
            for (const int corner : vert_to_corner[i]) {
              const float3 rest_normal = custom_normals.span[corner];
              const float3 rotated = matrix * rest_normal;
              const float3 mixed = blend >= 1.0f ? rotated :
                                                     math::interpolate(rest_normal, rotated, blend);
              const float length = math::length(mixed);
              if (length > 0.0f) {
                custom_normals.span[corner] = mixed / length;
              }
            }
          }
        });
      }
      custom_normals.finish();
    }
  }  if (PointCloud *pointcloud = geometry.get_pointcloud_for_write()) {
    deform(pointcloud->positions_for_write(), {}, {}, {});
    pointcloud->tag_positions_changed();
  }
  if (Curves *curves_id = geometry.get_curves_for_write()) {
    bke::CurvesGeometry &curves = curves_id->geometry.wrap();
    bke::MutableAttributeAccessor attributes = curves.attributes_for_write();
    /* A curve handle is a position in the same frame, so it follows its control point. */
    if (attributes.contains("handle_left") && attributes.contains("handle_right")) {
      bke::SpanAttributeWriter<float3> handle_left = attributes.lookup_or_add_for_write_span<float3>(
          "handle_left", AttrDomain::Point);
      bke::SpanAttributeWriter<float3> handle_right =
          attributes.lookup_or_add_for_write_span<float3>("handle_right", AttrDomain::Point);
      deform(curves.positions_for_write(), handle_left.span, handle_right.span, {});
      handle_left.finish();
      handle_right.finish();
    }
    else {
      deform(curves.positions_for_write(), {}, {}, {});
    }
    curves.tag_positions_changed();
  }

  kdtree_free(tree);
  params.set_output("Geometry"_ustr, std::move(geometry));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodePointDeform"_ustr);
  ntype.ui_name = "Point Deform";
  ntype.ui_description =
      "Transfer a deformation from a rest point set to a geometry, with a local transform per "
      "target point";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_180;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_point_deform_cc
