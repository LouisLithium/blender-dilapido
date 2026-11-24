/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Remesh`: reconstruct a surface as a high quality triangle mesh.
 *
 * The mesh is triangulated, then refined and coarsened towards a target edge length:
 *
 * - edges longer than the target are split (one or more new vertices per edge),
 * - edges shorter than the target are collapsed, with the usual link/valence and face flip
 *   checks so the result stays a valid manifold surface,
 * - edges are flipped when that improves the triangle quality,
 * - vertices are relaxed tangentially and projected back onto the input surface.
 *
 * Boundary, non-manifold, sharp and (optionally) UV seam edges are hardened: they may be split,
 * but never flipped or collapsed, and vertices on a crease are only allowed to slide along it.
 *
 * `Adaptive` derives the local target length from the discrete curvature, then limits how fast it
 * may change across the surface (`Gradation`), so flat regions get larger triangles while curved
 * ones keep the detail.
 *
 * The output mesh is always fully triangulated. It carries three diagnostic attributes:
 * `remesh_size` (FLOAT, POINT), `remesh_quality` (FLOAT, FACE, 1 = equilateral) and
 * `remesh_hard_edge` (BOOL, EDGE).
 */

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "BLI_array.hh"
#include "BLI_kdopbvh.hh"
#include "BLI_math_vector.h"
#include "BLI_math_vector.hh"
#include "BLI_set.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_bvhutils.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"

#include "DNA_modifier_types.h"

#include "bmesh.hh"
#include "bmesh_tools.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_remesh_cc {

enum class EdgeLengthMode : int {
  Uniform = 0,
  Adaptive = 1,
};

static const EnumPropertyItem edge_length_mode_items[] = {
    {int(EdgeLengthMode::Uniform),
     "UNIFORM",
     0,
     N_("Uniform"),
     N_("Use one target edge length over the whole surface")},
    {int(EdgeLengthMode::Adaptive),
     "ADAPTIVE",
     0,
     N_("Adaptive"),
     N_("Use smaller triangles in curved areas and larger ones in flat areas")},
    {0, nullptr, 0, nullptr, nullptr},
};

struct RemeshSettings {
  EdgeLengthMode mode = EdgeLengthMode::Uniform;
  float target_size = 0.05f;
  float relative_density = 1.0f;
  float min_size = 0.0f;
  float max_size = 0.0f;
  float gradation = 0.5f;
  int iterations = 4;
  float smoothing = 0.5f;
  bool preserve_sharp = true;
  float sharp_angle = M_PI / 3.0f;
  bool preserve_uv_seams = true;
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Helpers
 * \{ */

static void remesh_ensure_indices(BMesh &bm)
{
  BM_mesh_elem_table_ensure(&bm, BM_VERT | BM_EDGE | BM_FACE);
  BM_mesh_elem_index_ensure(&bm, BM_VERT | BM_EDGE | BM_FACE);
}

static int remesh_target_layer(BMesh &bm)
{
  BM_data_layer_ensure_named(&bm, &bm.vdata, CD_PROP_FLOAT, "remesh_size");
  return CustomData_get_offset_named(&bm.vdata, CD_PROP_FLOAT, "remesh_size");
}

static void remesh_write_target_lengths(BMesh &bm, const Span<float> lengths)
{
  remesh_ensure_indices(bm);
  const int offset = remesh_target_layer(bm);
  BMIter iter;
  BMVert *vert;
  BM_ITER_MESH (vert, &iter, &bm, BM_VERTS_OF_MESH) {
    BM_ELEM_CD_SET_FLOAT(vert, offset, lengths[BM_elem_index_get(vert)]);
  }
}

static void remesh_read_target_lengths(BMesh &bm, MutableSpan<float> r_lengths)
{
  remesh_ensure_indices(bm);
  const int offset = remesh_target_layer(bm);
  BMIter iter;
  BMVert *vert;
  BM_ITER_MESH (vert, &iter, &bm, BM_VERTS_OF_MESH) {
    r_lengths[BM_elem_index_get(vert)] = BM_ELEM_CD_GET_FLOAT(vert, offset);
  }
}

/** Triangulate every face. Splitting an edge inserts vertices into its faces, so this is needed
 * after every split pass to keep the surface a triangle mesh. */
static void remesh_triangulate(BMesh &bm)
{
  BMIter iter;
  BMFace *face;
  BM_ITER_MESH (face, &iter, &bm, BM_FACES_OF_MESH) {
    BM_elem_flag_enable(face, BM_ELEM_TAG);
  }
  BMO_op_callf(&bm,
               BMO_FLAG_DEFAULTS,
               "triangulate faces=%hf quad_method=%i ngon_method=%i",
               BM_ELEM_TAG,
               int(MOD_TRIANGULATE_QUAD_SHORTEDGE),
               int(MOD_TRIANGULATE_NGON_BEAUTY));
  remesh_ensure_indices(bm);
}

/** Mark boundary, non-manifold, sharp and UV seam edges as hard (BM_ELEM_TAG). */
static void remesh_mark_hard_edges(BMesh &bm, const RemeshSettings &settings)
{
  BM_mesh_normals_update(&bm);
  BMIter iter;
  BMEdge *edge;
  BM_ITER_MESH (edge, &iter, &bm, BM_EDGES_OF_MESH) {
    BM_elem_flag_disable(edge, BM_ELEM_TAG);
  }

  BM_ITER_MESH (edge, &iter, &bm, BM_EDGES_OF_MESH) {
    bool hard = false;
    if (BM_edge_is_boundary(edge) || !BM_edge_is_manifold(edge)) {
      hard = true;
    }
    if (!hard && settings.preserve_uv_seams && BM_elem_flag_test(edge, BM_ELEM_SEAM)) {
      hard = true;
    }
    if (!hard && settings.preserve_sharp && !BM_elem_flag_test(edge, BM_ELEM_SMOOTH)) {
      hard = true;
    }
    if (!hard && settings.preserve_sharp && settings.sharp_angle < M_PI) {
      if (BM_edge_is_manifold(edge)) {
        const BMFace *face_a = edge->l->f;
        const BMFace *face_b = edge->l->radial_next->f;
        if (angle_v3v3(face_a->no, face_b->no) > settings.sharp_angle) {
          hard = true;
        }
      }
    }
    BM_elem_flag_set(edge, BM_ELEM_TAG, hard);
  }
}

/** Local target edge length per vertex. */
static void remesh_compute_target_lengths(BMesh &bm,
                                          const RemeshSettings &settings,
                                          MutableSpan<float> r_lengths)
{
  remesh_ensure_indices(bm);
  BLI_assert(r_lengths.size() == bm.totvert);

  if (settings.mode == EdgeLengthMode::Uniform) {
    r_lengths.fill(settings.target_size);
    return;
  }

  BM_mesh_normals_update(&bm);

  Array<float> curvature(bm.totvert, 0.0f);
  BMIter iter;
  BMEdge *edge;
  BM_ITER_MESH (edge, &iter, &bm, BM_EDGES_OF_MESH) {
    if (!BM_edge_is_manifold(edge)) {
      continue;
    }
    const float length = BM_edge_calc_length(edge);
    if (length < 1.0e-7f) {
      continue;
    }
    const float angle = angle_v3v3(edge->l->f->no, edge->l->radial_next->f->no);
    const float value = angle / length;
    const int index_a = BM_elem_index_get(edge->v1);
    const int index_b = BM_elem_index_get(edge->v2);
    curvature[index_a] = std::max(curvature[index_a], value);
    curvature[index_b] = std::max(curvature[index_b], value);
  }

  /* Relative density 1 roughly gives 16 segments around a curvature radius (the same meaning as
   * the counterpart in comparable remeshers). */
  const float density_scale = (2.0f * float(M_PI) / 16.0f) /
                              std::max(settings.relative_density, 0.01f);

  BMVert *vert;
  BM_ITER_MESH (vert, &iter, &bm, BM_VERTS_OF_MESH) {
    const int index = BM_elem_index_get(vert);
    const float radius = 1.0f / std::max(curvature[index], 1.0e-6f);
    float target = density_scale * radius;
    if (settings.min_size > 0.0f) {
      target = std::max(target, settings.min_size);
    }
    if (settings.max_size > 0.0f) {
      target = std::min(target, settings.max_size);
    }
    r_lengths[index] = std::max(target, 1.0e-5f);
  }

  /* Limit how fast the target size changes from one triangle to the next. Smaller values dominate,
   * so a few passes are enough to propagate them outwards. */
  const float gradation = std::max(settings.gradation, 0.0f);
  if (gradation > 0.0f) {
    for (int pass = 0; pass < 4; pass++) {
      BM_ITER_MESH (edge, &iter, &bm, BM_EDGES_OF_MESH) {
        const int index_a = BM_elem_index_get(edge->v1);
        const int index_b = BM_elem_index_get(edge->v2);
        float &length_a = r_lengths[index_a];
        float &length_b = r_lengths[index_b];
        const float max_a = length_b * (1.0f + gradation);
        const float max_b = length_a * (1.0f + gradation);
        if (length_a > max_a) {
          length_a = max_a;
        }
        if (length_b > max_b) {
          length_b = max_b;
        }
      }
    }
  }
}

static float triangle_quality(const float3 &a, const float3 &b, const float3 &c)
{
  const float3 ab = b - a;
  const float3 ac = c - a;
  const float3 bc = c - b;
  const float area_twice = math::length(math::cross(ab, ac));
  const float sum_squared = math::length_squared(ab) + math::length_squared(ac) +
                            math::length_squared(bc);
  if (sum_squared <= 1.0e-20f) {
    return 0.0f;
  }
  /* 4 * sqrt(3) * area / (a^2 + b^2 + c^2). Equilateral triangles give exactly 1. */
  return (2.0f * std::sqrt(3.0f) * area_twice) / sum_squared;
}

static float face_quality(const BMFace &face)
{
  if (face.len != 3) {
    return 0.0f;
  }
  const BMLoop *loop = BM_FACE_FIRST_LOOP(&face);
  const float3 a(loop->v->co);
  const float3 b(loop->next->v->co);
  const float3 c(loop->next->next->v->co);
  return triangle_quality(a, b, c);
}

static BMVert *face_opposite_vert(const BMFace &face, const BMEdge &edge)
{
  const BMLoop *first = BM_FACE_FIRST_LOOP(&face);
  const BMLoop *loop = first;
  do {
    if (!BM_vert_in_edge(&edge, loop->v)) {
      return loop->v;
    }
    loop = loop->next;
  } while (loop != first);
  return nullptr;
}

static void split_long_edges(BMesh &bm, const Span<float> target_lengths)
{
  remesh_ensure_indices(bm);

  Vector<BMEdge *> candidates;
  BMIter iter;
  BMEdge *edge;
  BM_ITER_MESH (edge, &iter, &bm, BM_EDGES_OF_MESH) {
    if (BM_edge_is_wire(edge)) {
      continue;
    }
    const float length = BM_edge_calc_length(edge);
    const float target = 0.5f * (target_lengths[BM_elem_index_get(edge->v1)] +
                                 target_lengths[BM_elem_index_get(edge->v2)]);
    if (length > target * (4.0f / 3.0f)) {
      candidates.append(edge);
    }
  }

  for (BMEdge *candidate : candidates) {
    const float length = BM_edge_calc_length(candidate);
    const float target = 0.5f * (target_lengths[BM_elem_index_get(candidate->v1)] +
                                 target_lengths[BM_elem_index_get(candidate->v2)]);
    int segments = int(std::round(length / std::max(target, 1.0e-5f)));
    if (segments < 2) {
      /* Splitting at the midpoint would create segments far below the target and, on a regular
       * grid, cannot be undone (the new valence four vertex lies on a straight edge). Leave the
       * edge alone instead of inflating the triangle count for no quality gain. */
      continue;
    }
    segments = std::min(segments, 32);
    BM_edge_split_n(&bm, candidate, segments - 1, nullptr);
  }
  remesh_triangulate(bm);
}

static bool collapse_keeps_surface_shape(BMEdge &edge, const float3 &new_position)
{
  BMVert *vert_a = edge.v1;
  BMVert *vert_b = edge.v2;

  const int degree_a = BM_vert_edge_count(vert_a);
  const int degree_b = BM_vert_edge_count(vert_b);
  /* The merged vertex is connected to (deg(a) - 1) + (deg(b) - 1) - 2 common neighbours. */
  if (degree_a + degree_b - 4 < 4) {
    return false;
  }

  /* Link condition: an interior edge may only share exactly the two opposite vertices with its
   * adjacent faces. More shared neighbours means the collapse would create a non-manifold edge. */
  int common = 0;
  BMIter iter;
  BMEdge *other_edge;
  BM_ITER_ELEM (other_edge, &iter, vert_a, BM_EDGES_OF_VERT) {
    BMVert *other = BM_edge_other_vert(other_edge, vert_a);
    if (BM_edge_exists(other, vert_b)) {
      common++;
    }
  }
  if (common != 2) {
    return false;
  }

  /* Simulate the local faces after the collapse and reject normal flips and degenerates. */
  BMFace *face;
  for (int side = 0; side < 2; side++) {
    BMVert *vert = side ? vert_b : vert_a;
    BM_ITER_ELEM (face, &iter, vert, BM_FACES_OF_VERT) {
      if (BM_edge_in_face(&edge, face)) {
        continue;
      }
      if (face->len != 3) {
        return false;
      }
      float3 positions[3];
      int count = 0;
      const BMLoop *first = BM_FACE_FIRST_LOOP(face);
      const BMLoop *loop = first;
      do {
        const BMVert *loop_vert = loop->v;
        positions[count++] = (loop_vert == vert_a || loop_vert == vert_b) ? new_position :
                                                                            float3(loop_vert->co);
        loop = loop->next;
      } while (loop != first);
      if (count != 3) {
        return false;
      }
      const float3 old_normal(face->no);
      const float3 new_normal = math::cross(positions[1] - positions[0],
                                            positions[2] - positions[0]);
      if (math::length_squared(new_normal) <= 1.0e-20f) {
        return false;
      }
      if (math::length_squared(old_normal) <= 1.0e-20f) {
        continue;
      }
      if (math::dot(math::normalize(new_normal), math::normalize(old_normal)) < 0.2f) {
        return false;
      }
    }
  }
  return true;
}

static void collapse_short_edges(BMesh &bm, const Span<float> target_lengths)
{
  remesh_ensure_indices(bm);
  BM_mesh_normals_update(&bm);
  const int size_offset = remesh_target_layer(bm);

  for (int pass = 0; pass < 8; pass++) {
    Vector<BMEdge *> candidates;
    BMIter iter;
    BMEdge *edge;
    BM_ITER_MESH (edge, &iter, &bm, BM_EDGES_OF_MESH) {
      if (BM_elem_flag_test(edge, BM_ELEM_TAG) || !BM_edge_is_manifold(edge)) {
        continue;
      }
      if (BM_edge_is_boundary(edge)) {
        continue;
      }
      if (BM_vert_is_boundary(edge->v1) || BM_vert_is_boundary(edge->v2)) {
        continue;
      }
      const float length = BM_edge_calc_length(edge);
      const float target = 0.5f * (target_lengths[BM_elem_index_get(edge->v1)] +
                                   target_lengths[BM_elem_index_get(edge->v2)]);
      if (length < target * (4.0f / 5.0f)) {
        candidates.append(edge);
      }
    }
    if (candidates.is_empty()) {
      return;
    }

    std::sort(candidates.begin(), candidates.end(), [](const BMEdge *a, const BMEdge *b) {
      return BM_edge_calc_length_squared(a) < BM_edge_calc_length_squared(b);
    });

    Set<BMVert *> touched;
    int collapsed = 0;
    for (BMEdge *candidate : candidates) {
      if (touched.contains(candidate->v1) || touched.contains(candidate->v2)) {
        continue;
      }
      const float3 midpoint = 0.5f * (float3(candidate->v1->co) + float3(candidate->v2->co));
      if (!collapse_keeps_surface_shape(*candidate, midpoint)) {
        continue;
      }
      touched.add(candidate->v1);
      touched.add(candidate->v2);
      const float size = 0.5f * (BM_ELEM_CD_GET_FLOAT(candidate->v1, size_offset) +
                                 BM_ELEM_CD_GET_FLOAT(candidate->v2, size_offset));
      /* Remove the lower valence endpoint so the surviving fan stays as regular as possible. */
      BMVert *vert_kill = BM_vert_edge_count(candidate->v1) < BM_vert_edge_count(candidate->v2) ?
                              candidate->v1 :
                              candidate->v2;
      BMVert *survivor = BM_edge_collapse(&bm, candidate, vert_kill, true, true);
      if (survivor != nullptr) {
        copy_v3_v3(survivor->co, midpoint);
        BM_ELEM_CD_SET_FLOAT(survivor, size_offset, size);
      }
      collapsed++;
    }
    if (collapsed == 0) {
      return;
    }
    remesh_ensure_indices(bm);
    BM_mesh_normals_update(&bm);
  }
}

static void flip_edges_for_quality(BMesh &bm, const Span<float> target_lengths)
{
  remesh_ensure_indices(bm);
  BM_mesh_normals_update(&bm);

  Vector<BMEdge *> candidates;
  BMIter iter;
  BMEdge *edge;
  BM_ITER_MESH (edge, &iter, &bm, BM_EDGES_OF_MESH) {
    if (BM_elem_flag_test(edge, BM_ELEM_TAG) || !BM_edge_is_manifold(edge)) {
      continue;
    }
    if (BM_edge_face_count(edge) != 2) {
      continue;
    }
    if (edge->l->f->len != 3 || edge->l->radial_next->f->len != 3) {
      continue;
    }
    candidates.append(edge);
  }

  for (BMEdge *candidate : candidates) {
    if (!BM_edge_is_manifold(candidate)) {
      continue;
    }
    BMFace *face_a = candidate->l->f;
    BMFace *face_b = candidate->l->radial_next->f;
    if (face_a->len != 3 || face_b->len != 3) {
      continue;
    }
    BMVert *opposite_a = face_opposite_vert(*face_a, *candidate);
    BMVert *opposite_b = face_opposite_vert(*face_b, *candidate);
    if (opposite_a == nullptr || opposite_b == nullptr || opposite_a == opposite_b) {
      continue;
    }
    if (BM_edge_exists(opposite_a, opposite_b)) {
      continue;
    }

    const int index_a = BM_elem_index_get(opposite_a);
    const int index_b = BM_elem_index_get(opposite_b);
    const float target = 0.5f * (target_lengths[index_a] + target_lengths[index_b]);
    const float diagonal_length = len_v3v3(opposite_a->co, opposite_b->co);
    if (diagonal_length < target * 0.25f || diagonal_length > target * 2.0f) {
      continue;
    }

    const float before = face_quality(*face_a) + face_quality(*face_b);
    const float3 a(opposite_a->co);
    const float3 b(opposite_b->co);
    const float3 c(candidate->v1->co);
    const float3 d(candidate->v2->co);
    const float after = triangle_quality(a, b, c) + triangle_quality(b, a, d);
    if (after <= before * 1.0005f) {
      continue;
    }

    if (BM_edge_rotate(
            &bm, candidate, true, BM_EDGEROT_CHECK_EXISTS | BM_EDGEROT_CHECK_DEGENERATE))
    {
      continue;
    }
    BM_edge_rotate(&bm, candidate, false, BM_EDGEROT_CHECK_EXISTS | BM_EDGEROT_CHECK_DEGENERATE);
  }
  remesh_ensure_indices(bm);
}

static void project_on_input_surface(const bke::BVHTreeFromMesh &bvh, float3 &position)
{
  BVHTreeNearest nearest = {};
  nearest.index = -1;
  nearest.dist_sq = FLT_MAX;
  BLI_bvhtree_find_nearest(bvh.tree,
                           position,
                           &nearest,
                           bvh.nearest_callback,
                           const_cast<bke::BVHTreeFromMesh *>(&bvh));
  if (nearest.index >= 0) {
    position = float3(nearest.co);
  }
}

/** Tangential relaxation, followed by a projection back onto the input surface. */
static void smooth_vertices(BMesh &bm,
                            const RemeshSettings &settings,
                            const Span<float> target_lengths,
                            const bke::BVHTreeFromMesh &bvh)
{
  if (settings.smoothing <= 0.0f) {
    return;
  }
  const int passes = std::max(1, int(settings.smoothing * 3.0f + 0.5f));
  const float factor = 0.5f * settings.smoothing;

  for (int pass = 0; pass < passes; pass++) {
    remesh_ensure_indices(bm);
    BM_mesh_normals_update(&bm);

    Array<float3> next_positions(bm.totvert);
    BMIter iter;
    BMIter iter_edge;
    BMVert *vert;
    BM_ITER_MESH (vert, &iter, &bm, BM_VERTS_OF_MESH) {
      const int index = BM_elem_index_get(vert);
      const float3 position(vert->co);
      next_positions[index] = position;

      if (!BM_vert_face_check(vert)) {
        continue;
      }

      Vector<float3> hard_directions;
      BMEdge *edge;
      BM_ITER_ELEM (edge, &iter_edge, vert, BM_EDGES_OF_VERT) {
        if (BM_elem_flag_test(edge, BM_ELEM_TAG)) {
          const float3 direction = float3(BM_edge_other_vert(edge, vert)->co) - position;
          if (math::length_squared(direction) > 1.0e-20f) {
            hard_directions.append(math::normalize(direction));
          }
        }
      }

      bool use_tangent = false;
      float3 tangent(0.0f);
      if (!hard_directions.is_empty()) {
        if (hard_directions.size() >= 3) {
          /* Corner: keep it pinned. */
          continue;
        }
        if (hard_directions.size() == 2) {
          const float alignment = math::dot(hard_directions[0], hard_directions[1]);
          if (alignment > -0.5f) {
            /* Two non-collinear creases meet here, this is a corner. */
            continue;
          }
          tangent = math::normalize(hard_directions[0] + hard_directions[1]);
        }
        else {
          tangent = hard_directions[0];
        }
        use_tangent = true;
      }

      float3 average(0.0f);
      int neighbor_count = 0;
      BM_ITER_ELEM (edge, &iter_edge, vert, BM_EDGES_OF_VERT) {
        if (use_tangent && !BM_elem_flag_test(edge, BM_ELEM_TAG)) {
          continue;
        }
        average += float3(BM_edge_other_vert(edge, vert)->co);
        neighbor_count++;
      }
      if (neighbor_count == 0) {
        continue;
      }
      average /= float(neighbor_count);

      float3 displacement = (average - position) * factor;
      if (use_tangent) {
        displacement = tangent * math::dot(displacement, tangent);
      }
      else {
        float3 normal(vert->no);
        if (math::length_squared(normal) <= 1.0e-20f) {
          continue;
        }
        normal = math::normalize(normal);
        displacement -= normal * math::dot(displacement, normal);
      }

      const float max_move = std::max(0.25f, 0.5f * target_lengths[index]);
      const float distance = math::length(displacement);
      if (distance > max_move) {
        displacement *= max_move / distance;
      }
      next_positions[index] = position + displacement;
    }

    BM_ITER_MESH (vert, &iter, &bm, BM_VERTS_OF_MESH) {
      const int index = BM_elem_index_get(vert);
      float3 position = next_positions[index];
      if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
        continue;
      }
      project_on_input_surface(bvh, position);
      copy_v3_v3(vert->co, position);
    }
  }
  remesh_ensure_indices(bm);
  BM_mesh_normals_update(&bm);
}

static Mesh *remesh_mesh(const Mesh &input_mesh,
                         RemeshSettings settings,
                         bool &r_too_many_triangles)
{
  r_too_many_triangles = false;

  if (settings.mode == EdgeLengthMode::Adaptive) {
    /* Flat regions have no curvature, so their target length would grow without bound. Keep it
     * within a sane fraction of the model unless the user set an explicit maximum. */
    float3 bbox_min(FLT_MAX);
    float3 bbox_max(-FLT_MAX);
    for (const float3 &position : input_mesh.vert_positions()) {
      bbox_min = math::min(bbox_min, position);
      bbox_max = math::max(bbox_max, position);
    }
    const float diagonal = math::distance(bbox_min, bbox_max);
    if (settings.max_size <= 0.0f) {
      settings.max_size = std::max(0.25f * diagonal, 1.0e-4f);
    }
    if (settings.min_size <= 0.0f) {
      settings.min_size = std::max(1.0e-4f * diagonal, 1.0e-5f);
    }
  }

  BMeshCreateParams create_params{};
  create_params.use_toolflags = true;
  BMesh *bm = BM_mesh_create(&bm_mesh_allocsize_default, &create_params);
  BMeshFromMeshParams from_params{};
  from_params.calc_face_normal = true;
  from_params.calc_vert_normal = true;
  BM_mesh_bm_from_me(bm, &input_mesh, &from_params);

  /* The remesher works on triangles only. */
  remesh_triangulate(*bm);

  /* The adaptive target size is computed once from the input surface and then carried through the
   * topology operations as a vertex attribute (split vertices interpolate it). Recomputing it from
   * the evolving mesh would feed the curvature estimate back into itself and collapse the surface
   * into ever smaller triangles. */
  {
    Array<float> lengths(bm->totvert);
    remesh_compute_target_lengths(*bm, settings, lengths);
    remesh_write_target_lengths(*bm, lengths);
  }

  const bke::BVHTreeFromMesh bvh = input_mesh.bvh_corner_tris();

  const int iterations = std::clamp(settings.iterations, 0, 30);
  for (int iteration = 0; iteration < iterations; iteration++) {
    remesh_mark_hard_edges(*bm, settings);

    {
      Array<float> lengths(bm->totvert);
      remesh_read_target_lengths(*bm, lengths);
      split_long_edges(*bm, lengths);
    }
    {
      remesh_mark_hard_edges(*bm, settings);
      Array<float> lengths(bm->totvert);
      remesh_read_target_lengths(*bm, lengths);
      collapse_short_edges(*bm, lengths);
    }

    remesh_mark_hard_edges(*bm, settings);

    {
      remesh_ensure_indices(*bm);
      Array<float> lengths(bm->totvert);
      remesh_read_target_lengths(*bm, lengths);
      flip_edges_for_quality(*bm, lengths);
      smooth_vertices(*bm, settings, lengths, bvh);
    }
  }

  remesh_mark_hard_edges(*bm, settings);

  /* Write the diagnostic attributes. `remesh_size` is maintained by the operations above. */
  {
    BMIter iter;
    BM_data_layer_ensure_named(bm, &bm->pdata, CD_PROP_FLOAT, "remesh_quality");
    const int quality_offset = CustomData_get_offset_named(
        &bm->pdata, CD_PROP_FLOAT, "remesh_quality");
    BMFace *face;
    BM_ITER_MESH (face, &iter, bm, BM_FACES_OF_MESH) {
      BM_ELEM_CD_SET_FLOAT(face, quality_offset, face_quality(*face));
    }

    BM_data_layer_ensure_named(bm, &bm->edata, CD_PROP_BOOL, "remesh_hard_edge");
    const int hard_offset = CustomData_get_offset_named(
        &bm->edata, CD_PROP_BOOL, "remesh_hard_edge");
    BMEdge *edge;
    BM_ITER_MESH (edge, &iter, bm, BM_EDGES_OF_MESH) {
      BM_ELEM_CD_SET_BOOL(edge, hard_offset, BM_elem_flag_test(edge, BM_ELEM_TAG));
    }
  }

  /* A runaway target size would otherwise allocate an enormous mesh. */
  if (bm->totface > 8000000) {
    r_too_many_triangles = true;
  }

  Mesh *result = BKE_mesh_new_nomain(0, 0, 0, 0);
  BMeshToMeshParams to_params{};
  to_params.calc_object_remap = false;
  BM_mesh_bm_to_me(nullptr, bm, result, &to_params);
  BM_mesh_free(bm);
  return result;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Node
 * \{ */

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();

  b.add_input<decl::Geometry>("Mesh"_ustr)
      .supported_type(GeometryComponent::Type::Mesh)
      .is_default_link_socket()
      .description("Surface to remesh. The output is always made of triangles");
  b.add_output<decl::Geometry>("Mesh"_ustr).propagate_all_geometry().align_with_previous();

  {
    auto &panel = b.add_panel("Element Sizing"_ustr).default_closed(false);
    panel.add_input<decl::Menu>("Edge Lengths"_ustr)
        .static_items(edge_length_mode_items)
        .default_value(MenuValue(int(EdgeLengthMode::Uniform)))
        .description("Use one target length everywhere, or adapt it to the local curvature");
    panel.add_input<decl::Float>("Target Size"_ustr)
        .default_value(0.05f)
        .min(1.0e-5f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Edge Lengths"_ustr, int(EdgeLengthMode::Uniform))
        .description("Target edge length of the reconstructed triangles");
    panel.add_input<decl::Float>("Relative Density"_ustr)
        .default_value(1.0f)
        .min(0.05f)
        .max(64.0f)
        .usage_by_menu("Edge Lengths"_ustr, int(EdgeLengthMode::Adaptive))
        .description(
            "Adaptive sampling density. 1 gives roughly 16 segments around a curvature "
            "radius, 2 twice as many");
    panel.add_input<decl::Float>("Min Size"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Edge Lengths"_ustr, int(EdgeLengthMode::Adaptive))
        .description("Lower bound for the adaptive edge length (0 = no limit)");
    panel.add_input<decl::Float>("Max Size"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Edge Lengths"_ustr, int(EdgeLengthMode::Adaptive))
        .description("Upper bound for the adaptive edge length (0 = no limit)");
    panel.add_input<decl::Float>("Gradation"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .max(10.0f)
        .usage_by_menu("Edge Lengths"_ustr, int(EdgeLengthMode::Adaptive))
        .description(
            "How fast the adaptive edge length may change between neighbouring "
            "triangles. 0 keeps it uniform");
  }

  {
    auto &panel = b.add_panel("Meshing"_ustr).default_closed(false);
    panel.add_input<decl::Int>("Iterations"_ustr)
        .default_value(4)
        .min(0)
        .max(30)
        .description(
            "Split/collapse/flip/relax passes. 0 only triangulates the input; 3 to 4 "
            "passes are usually enough");
    panel.add_input<decl::Float>("Smoothing"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .max(1.0f)
        .description(
            "Tangential relaxation per pass. Smooths the shape and improves the "
            "triangle quality; hard edges and boundaries only slide along themselves");
    panel.add_input<decl::Bool>("Preserve Sharp Edges"_ustr)
        .default_value(true)
        .description(
            "Keep sharp edges and corners: they can be split, but not flipped or "
            "collapsed");
    panel.add_input<decl::Float>("Sharp Angle"_ustr)
        .default_value(M_PI / 3.0f)
        .min(0.0f)
        .max(M_PI)
        .subtype(PROP_ANGLE)
        .usage_by_bool("Preserve Sharp Edges"_ustr, true)
        .description("Treat edges whose face angle is larger than this as sharp");
    panel.add_input<decl::Bool>("Preserve UV Seams"_ustr)
        .default_value(true)
        .description("Treat UV seams as sharp, so the UV layout is not pulled across islands");
  }
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh"_ustr);
  const Mesh *mesh = geometry_set.get_mesh();
  if (mesh == nullptr) {
    params.error_message_add(NodeWarningType::Info,
                             TIP_("Remesh: the input geometry is not a mesh"));
    params.set_output("Mesh"_ustr, std::move(geometry_set));
    return;
  }
  if (mesh->faces_num == 0) {
    params.set_output("Mesh"_ustr, std::move(geometry_set));
    return;
  }

  RemeshSettings settings;
  settings.mode = EdgeLengthMode(params.extract_input<int>("Edge Lengths"_ustr));
  settings.target_size = std::max(params.extract_input<float>("Target Size"_ustr), 1.0e-5f);
  settings.relative_density = std::max(params.extract_input<float>("Relative Density"_ustr),
                                       0.05f);
  settings.min_size = std::max(params.extract_input<float>("Min Size"_ustr), 0.0f);
  settings.max_size = std::max(params.extract_input<float>("Max Size"_ustr), 0.0f);
  settings.gradation = std::max(params.extract_input<float>("Gradation"_ustr), 0.0f);
  settings.iterations = params.extract_input<int>("Iterations"_ustr);
  settings.smoothing = std::clamp(params.extract_input<float>("Smoothing"_ustr), 0.0f, 1.0f);
  settings.preserve_sharp = params.extract_input<bool>("Preserve Sharp Edges"_ustr);
  settings.sharp_angle = std::clamp(
      params.extract_input<float>("Sharp Angle"_ustr), 0.0f, float(M_PI));
  settings.preserve_uv_seams = params.extract_input<bool>("Preserve UV Seams"_ustr);

  bool too_many = false;
  Mesh *result = remesh_mesh(*mesh, settings, too_many);
  if (result == nullptr) {
    params.error_message_add(NodeWarningType::Error, TIP_("Remesh: the remeshing failed"));
    params.set_output("Mesh"_ustr, std::move(geometry_set));
    return;
  }
  if (too_many) {
    params.error_message_add(
        NodeWarningType::Warning,
        TIP_("Remesh: the output has over 8 million triangles; increase the target size or "
             "reduce the input surface"));
  }
  geometry_set.replace_mesh(result);
  params.set_output("Mesh"_ustr, std::move(geometry_set));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeRemesh"_ustr);
  ntype.ui_name = "Remesh";
  ntype.ui_description =
      "Reconstruct a surface with high quality triangles. Uniform mode uses one target edge "
      "length, adaptive mode uses smaller triangles in curved areas";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_remesh_cc
