/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Voronoi fracture node.
 *
 * The input mesh is cut into Voronoi cells: a set of sites is scattered over the bounding box
 * (jittered grid, deterministic from the seed), and for every site the mesh is clipped by the
 * perpendicular bisector planes towards the neighboring sites. After every cut the open border
 * loop is filled, which gives closed fragments that can be used as rigid bodies.
 *
 * Outputs:
 * - `Geometry`: all fragments in one mesh (viewing).
 * - `Fragments`: one instance per fragment, centered at the origin, with the instance transform
 *   placing it in the world. Carries the stable `piece_id` attribute. This is the form the rigid
 *   body solver consumes directly.
 * This node only describes the *geometry* of the fracture. The constraints between the pieces are
 * made by `RBD Constraint from Pieces`, and mass/glue settings by `Init Rigid Body`, so that the
 * fracture stays a pure geometry operation.
 *
 * `piece_id` is the index of the site, which only depends on the cell size, the seed and the
 * bounding box, so it is stable across evaluations.
 */

#include <algorithm>
#include <cfloat>

#include "BKE_attribute.hh"
#include "BKE_bvhutils.hh"
#include "BKE_attribute_filter.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "BLI_hash.h"
#include "BLI_kdtree.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_vector.h"
#include "BLI_math_vector.hh"
#include "BLI_set.hh"
#include "BLI_string_utf8.h"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "GEO_join_geometries.hh"

#include "bmesh.hh"

#include "bmesh_tools.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_voronoi_fracture_cc {

enum class FractureSiteMode : int {
  Grid = 0,
  Custom = 1,
};

static const EnumPropertyItem fracture_site_mode_items[] = {
    {int(FractureSiteMode::Grid),
     "GRID",
     0,
     N_("Default"),
     N_("Jittered grid of cell centers, sized by Cell Size")},
    {int(FractureSiteMode::Custom),
     "CUSTOM",
     0,
     N_("Custom"),
     N_("Use the points of the Points input that are inside or on the surface of the geometry. "
        "Where the points are dense the fragments are small")},
    {0, nullptr, 0, nullptr, nullptr},
};

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr);
  b.add_input<decl::Menu>("Mode"_ustr)
      .static_items(fracture_site_mode_items)
      .default_value(MenuValue(int(FractureSiteMode::Grid)))
      .description(
          "Where the cell centers come from. Custom takes them from the Points input, so the "
          "fragment size can be painted with a point cloud instead of being uniform");
  b.add_input<decl::Bool>("Shell Mode"_ustr)
      .default_value(false)
      .description(
          "Only cut the existing surface and do not fill the cut open: every piece stays an open "
          "shell (no caps). Useful for splitting a surface, e.g. a terrain or a wall, into pieces "
          "without closing them");
  b.add_input<decl::Float>("Shell Thickness"_ustr)
      .default_value(0.01f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description(
          "Thickness used to give the shell pieces a mass (surface area x thickness x density). It "
          "does not create any geometry, the pieces stay open shells")
      .usage_by_bool("Shell Mode"_ustr, true);
  b.add_input<decl::Geometry>("Points"_ustr)
      .description(
          "Cell centers for the Custom mode (point cloud or mesh vertices). Only the points that "
          "are inside or on the surface of the geometry become cells, the ones outside are "
          "ignored");
  b.add_input<decl::Vector>("Cell Size"_ustr)
      .default_value(float3(0.5f))
      .min(0.001f)
      .subtype(PROP_DISTANCE)
      .description("Approximate size of the fracture cells, per axis. Different values per axis "
                   "give elongated cells, which is useful for e.g. wood like fracture")
      /* Only the grid mode uses a cell size; the custom mode takes its cell centers from Points. */
      .usage_by_single_menu(int(FractureSiteMode::Grid));
  b.add_input<decl::Int>("Seed"_ustr)
      .default_value(0)
      .description("Random seed for the cell layout")
      .usage_by_single_menu(int(FractureSiteMode::Grid));
  b.add_input<decl::Vector>("Jitter"_ustr)
      .default_value(float3(0.5f))
      .min(0.0f)
      .max(1.0f)
      .description("Random offset of the cell centers inside their grid cell, per axis. For "
                   "elongated (e.g. wood like) fragments keep the jitter at 0 along the grain: "
                   "jittering along the long axis turns the splinters into irregular blocks")
      .usage_by_single_menu(int(FractureSiteMode::Grid));
  b.add_input<decl::Bool>("Delete Degenerate Pieces"_ustr)
      .default_value(true)
      .description(
          "Remove fragments that collapsed to a surface (no volume). Such pieces can not be used "
          "as rigid bodies: their convex hull is degenerate and they squeeze through colliders")
      /* Shell pieces have no volume by definition, so this option does not apply to them. */
      .usage_by_bool("Shell Mode"_ustr, false);

  b.add_output<decl::Geometry>("Geometry"_ustr)
      .description("All fragments as a single mesh");
  b.add_output<decl::Geometry>("Pieces"_ustr)
      .description("One instance per fragment with a piece_id, ready for the rigid body solver");
}

/** Deterministic random value in [0, 1) for a grid coordinate and seed. */
static float site_random(const int x, const int y, const int z, const int seed, const int salt)
{
  const uint32_t h = BLI_hash_int_3d(uint32_t(x + 4096),
                                     uint32_t(y + 4096),
                                     uint32_t(z + 4096) ^ (uint32_t(seed) * 2654435761u)) ^
                     (uint32_t(salt) * 2246822519u);
  return float(h) / float(0xFFFFFFFFu);
}

/** Jittered grid of Voronoi sites over the bounding box of the mesh. */
static Vector<float3> generate_sites(const float3 &bounds_min,
                                     const float3 &bounds_max,
                                     const float3 cell_size,
                                     const int seed,
                                     const float3 jitter,
                                     int3 &r_res)
{
  Vector<float3> sites;
  const float3 size = bounds_max - bounds_min;
  const int3 res = int3(std::max(1, int(std::ceil(size.x / cell_size.x))),
                        std::max(1, int(std::ceil(size.y / cell_size.y))),
                        std::max(1, int(std::ceil(size.z / cell_size.z))));
  r_res = res;
  for (const int z : IndexRange(res.z)) {
    for (const int y : IndexRange(res.y)) {
      for (const int x : IndexRange(res.x)) {
        const float3 base = bounds_min + (float3(x, y, z) + float3(0.5f)) * cell_size;
        const float3 offset(site_random(x, y, z, seed, 0) - 0.5f,
                            site_random(x, y, z, seed, 1) - 0.5f,
                            site_random(x, y, z, seed, 2) - 0.5f);
        sites.append(base + offset * (jitter * cell_size));
      }
    }
  }
  return sites;
}

/** A site that can bound the cell of another site, with its distance. */
struct SiteCandidate {
  int index;
  float distance;
};

/**
 * Neighbour lookup for the Custom mode. The sites come from a point cloud there, so they do not
 * follow a grid and the grid neighbourhood of the default mode can not be used. They are hashed
 * into a regular grid that only serves the neighbour search.
 */
struct SiteGrid {
  Span<float3> sites;
  /** Sites farther away than this from a site can not bound its cell. */
  float cut_distance = 0.0f;
  int3 res = int3(1);
  float bucket_size = 1.0f;
  float3 bucket_min = float3(0.0f);
  /** CSR buckets: `bucket_start[i] .. bucket_start[i + 1]` indexes into `bucket_sites`. */
  Array<int> bucket_start;
  Array<int> bucket_sites;

  void gather_candidates(int site_index, Vector<SiteCandidate> &r_candidates) const;
};

void SiteGrid::gather_candidates(const int site_index,
                                 Vector<SiteCandidate> &r_candidates) const
{
  const float3 site = sites[site_index];
  const float3 local = (site - bucket_min) / bucket_size;
  const int3 center(int(std::floor(local.x)), int(std::floor(local.y)), int(std::floor(local.z)));
  /* The bucket size is the cut distance, so the 27 surrounding buckets hold every site that can
   * reach this cell. */
  for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        const int3 bucket = center + int3(dx, dy, dz);
        if (bucket.x < 0 || bucket.y < 0 || bucket.z < 0 || bucket.x >= res.x ||
            bucket.y >= res.y || bucket.z >= res.z)
        {
          continue;
        }
        const int index = (bucket.z * res.y + bucket.y) * res.x + bucket.x;
        for (const int i : IndexRange(bucket_start[index],
                                      bucket_start[index + 1] - bucket_start[index]))
        {
          const int other = bucket_sites[i];
          if (other == site_index) {
            continue;
          }
          const float distance = math::distance(site, sites[other]);
          if (distance <= cut_distance) {
            r_candidates.append({other, distance});
          }
        }
      }
    }
  }
}

/** The sites of the Custom mode: the input points that are inside or on the surface of the mesh. */
static Vector<float3> custom_sites_from_points(const Mesh &mesh, const GeometrySet &points)
{
  Vector<float3> candidates;
  if (const PointCloud *pointcloud = points.get_pointcloud()) {
    candidates.extend(pointcloud->positions());
  }
  if (const Mesh *points_mesh = points.get_mesh()) {
    candidates.extend(points_mesh->vert_positions());
  }
  if (candidates.is_empty() || mesh.faces_num == 0) {
    return {};
  }

  /* Closest surface point + the normal of that triangle: a point is inside when it is behind the
   * surface. That is exact for a closed mesh, and it keeps the points that sit exactly on the
   * surface (a painted shell of points is the most natural way to use this mode). */
  const bke::BVHTreeFromMesh bvh = mesh.bvh_corner_tris();
  Vector<float3> sites;
  for (const float3 &position : candidates) {
    BVHTreeNearest nearest;
    nearest.index = -1;
    nearest.dist_sq = FLT_MAX;
    const float co[3] = {position.x, position.y, position.z};
    if (BLI_bvhtree_find_nearest(bvh.tree, co, &nearest, nullptr, nullptr) < 0) {
      continue;
    }
    const float3 delta = position - float3(nearest.co);
    if (math::length_squared(delta) <= 1.0e-8f) {
      sites.append(position);
      continue;
    }
    const int3 tri = bvh.corner_tris[nearest.index];
    const float3 &a = bvh.vert_positions[bvh.corner_verts[tri[0]]];
    const float3 &b = bvh.vert_positions[bvh.corner_verts[tri[1]]];
    const float3 &c = bvh.vert_positions[bvh.corner_verts[tri[2]]];
    if (math::dot(delta, math::cross(b - a, c - a)) < 0.0f) {
      sites.append(position);
    }
  }
  return sites;
}

/** Bucket grid over the given sites, used for the neighbour search of the Custom mode. */
static SiteGrid build_site_grid(const Span<float3> sites)
{
  SiteGrid grid;
  grid.sites = sites;
  if (sites.is_empty()) {
    return grid;
  }

  /* Mean distance to the nearest other site drives both the search radius and the bucket size: a
   * Voronoi cell is at most a few nearest neighbour distances across, five of them leave a wide
   * margin while keeping the bucket scan small. */
  float spacing = 0.0f;
  {
    KDTree<float3> *tree = kdtree_new<float3>(uint(sites.size()));
    for (const int i : sites.index_range()) {
      kdtree_insert(tree, i, sites[i]);
    }
    kdtree_balance(tree);
    double spacing_sum = 0.0;
    int spacing_num = 0;
    KDTreeNearest<float3> nearest[2];
    for (const float3 &site : sites) {
      if (kdtree_find_nearest_n(tree, site, nearest, 2) >= 2) {
        spacing_sum += double(nearest[1].dist);
        spacing_num++;
      }
    }
    kdtree_free(tree);
    spacing = spacing_num > 0 ? float(spacing_sum / spacing_num) : 0.0f;
  }
  if (spacing <= 1.0e-5f) {
    spacing = 1.0e-3f;
  }
  grid.cut_distance = spacing * 5.0f;
  grid.bucket_size = grid.cut_distance;

  float3 bounds_min = sites.first();
  float3 bounds_max = sites.first();
  for (const float3 &site : sites) {
    bounds_min = math::min(bounds_min, site);
    bounds_max = math::max(bounds_max, site);
  }
  /* One bucket of margin on each side: the 27 bucket scan can then never point outside the grid for
   * a site on the border. */
  grid.bucket_min = bounds_min - float3(grid.bucket_size);
  const float3 size = (bounds_max - bounds_min) + float3(2.0f * grid.bucket_size);
  grid.res = int3(std::max(1, int(std::ceil(size.x / grid.bucket_size))),
                  std::max(1, int(std::ceil(size.y / grid.bucket_size))),
                  std::max(1, int(std::ceil(size.z / grid.bucket_size))));

  const int bucket_num = grid.res.x * grid.res.y * grid.res.z;
  Array<int> bucket_of_site(sites.size(), -1);
  Array<int> counts(bucket_num, 0);
  for (const int i : sites.index_range()) {
    const float3 local = (sites[i] - grid.bucket_min) / grid.bucket_size;
    const int3 bucket(int(std::floor(local.x)),
                      int(std::floor(local.y)),
                      int(std::floor(local.z)));
    if (bucket.x < 0 || bucket.y < 0 || bucket.z < 0 || bucket.x >= grid.res.x ||
        bucket.y >= grid.res.y || bucket.z >= grid.res.z)
    {
      continue;
    }
    const int index = (bucket.z * grid.res.y + bucket.y) * grid.res.x + bucket.x;
    bucket_of_site[i] = index;
    counts[index]++;
  }
  grid.bucket_start = Array<int>(bucket_num + 1, 0);
  for (const int i : IndexRange(bucket_num)) {
    grid.bucket_start[i + 1] = grid.bucket_start[i] + counts[i];
  }
  grid.bucket_sites = Array<int>(grid.bucket_start[bucket_num], -1);
  Array<int> written(bucket_num, 0);
  for (const int i : sites.index_range()) {
    const int bucket = bucket_of_site[i];
    if (bucket < 0) {
      continue;
    }
    grid.bucket_sites[grid.bucket_start[bucket] + written[bucket]] = i;
    written[bucket]++;
  }
  return grid;
}

static void bisect_plane(BMesh *bm, const float3 &co, const float3 &no)
{
  BMOperator op;
  BMO_op_init(bm, &op, BMO_FLAG_DEFAULTS, "bisect_plane");
  BMO_slot_buffer_from_all(bm, &op, op.slots_in, "geom", BM_ALL_NOLOOP);
  float plane_co[3] = {co.x, co.y, co.z};
  float plane_no[3] = {no.x, no.y, no.z};
  BMO_slot_vec_set(op.slots_in, "plane_co", plane_co);
  BMO_slot_vec_set(op.slots_in, "plane_no", plane_no);
  BMO_slot_bool_set(op.slots_in, "use_snap_center", true);
  BMO_slot_bool_set(op.slots_in, "clear_inner", false);
  BMO_slot_bool_set(op.slots_in, "clear_outer", true);
  BMO_slot_float_set(op.slots_in, "dist", 1.0e-5f);
  BMO_op_exec(bm, &op);
  BMO_op_finish(bm, &op);
}

/**
 * Mark the faces that the hole filling creates: a face that did not exist before the fill is a cut
 * cap.
 *
 * This uses a custom data layer rather than an element flag, because the mesh operators use the
 * element flags as scratch space (`bmo_holes_fill_exec` clears the edge *and face* tags, and
 * `bisect_plane` does the same), so a flag would be silently dropped. Face pointers are stable in
 * BMesh, which is what makes the before/after comparison work.
 */
static void mark_new_faces(BMesh *bm, const int original_offset, Set<const BMFace *> &before)
{
  BMIter iter;
  BMFace *face;
  BM_ITER_MESH (face, &iter, bm, BM_FACES_OF_MESH) {
    if (before.contains(face)) {
      continue;
    }
    BM_ELEM_CD_SET_BOOL(face, original_offset, false);
  }
}

static void fill_holes(BMesh *bm, const int original_offset)
{
  /* Only boundary edges may be passed to holes_fill, otherwise the edge-net fill has nothing
   * to work with. */
  BM_mesh_elem_hflag_disable_all(bm, BM_EDGE, BM_ELEM_TAG, false);
  int boundary_edges = 0;
  BMIter iter;
  BMEdge *edge;
  BM_ITER_MESH (edge, &iter, bm, BM_EDGES_OF_MESH) {
    if (BM_edge_is_boundary(edge)) {
      BM_elem_flag_enable(edge, BM_ELEM_TAG);
      boundary_edges++;
    }
  }
  if (boundary_edges == 0) {
    return;
  }

  Set<const BMFace *> before;
  BMFace *face;
  BM_ITER_MESH (face, &iter, bm, BM_FACES_OF_MESH) {
    before.add(face);
  }

  BMOperator op;
  BMO_op_init(bm, &op, BMO_FLAG_DEFAULTS, "holes_fill");
  BMO_slot_buffer_from_enabled_hflag(bm, &op, op.slots_in, "edges", BM_EDGE, BM_ELEM_TAG);
  BMO_slot_int_set(op.slots_in, "sides", 0);
  BMO_op_exec(bm, &op);
  BMO_op_finish(bm, &op);

  mark_new_faces(bm, original_offset, before);
}

static void recalc_face_normals(BMesh *bm)
{
  BMOperator op;
  BMO_op_init(bm, &op, BMO_FLAG_DEFAULTS, "recalc_face_normals");
  BMO_slot_buffer_from_all(bm, &op, op.slots_in, "faces", BM_FACE);
  BMO_op_exec(bm, &op);
  BMO_op_finish(bm, &op);
}

/** Result of cutting one Voronoi cell out of the source mesh. */
struct Fragment {
  /** Fragment in world space (used for the joined geometry output). */
  Mesh *mesh_world = nullptr;
  /** Fragment centered at the origin (used for the instance output). */
  Mesh *mesh_local = nullptr;
  float3 center = float3(0.0f);
  float3 bounds_min = float3(0.0f);
  float3 bounds_max = float3(0.0f);
  /**
   * True when the piece has no usable volume: all of its points are (nearly) coplanar, so it can not
   * be used as a rigid body (its convex hull is degenerate and it squeezes through colliders).
   */
  bool degenerate = false;
};

/** Volume of a closed BMesh (divergence theorem, fan triangulation of every face). */
static float bm_volume(BMesh *bm)
{
  float volume = 0.0f;
  BMIter iter;
  BMFace *face;
  BM_ITER_MESH (face, &iter, bm, BM_FACES_OF_MESH) {
    BMLoop *loop = BM_FACE_FIRST_LOOP(face);
    const float3 a(loop->v->co);
    BMLoop *l = loop->next;
    while (l->next != loop) {
      const float3 b(l->v->co);
      const float3 c(l->next->v->co);
      volume += math::dot(a, math::cross(b, c)) / 6.0f;
      l = l->next;
    }
  }
  return std::abs(volume);
}

static Mesh *bm_to_mesh(BMesh *bm)
{
  Mesh *mesh = BKE_mesh_new_nomain(0, 0, 0, 0);
  BMeshToMeshParams params{};
  params.calc_object_remap = false;
  BM_mesh_bm_to_me(nullptr, bm, mesh, &params);
  return mesh;
}


/**
 * Write the face attributes that tell the original surface from the cut caps: `original_face` is
 * true for the faces that came from the source mesh (also for the halves the cuts split them into)
 * and `cut_face` is true for the faces the hole filling created.
 *
 * The face order of the mesh matches the order of the BMesh faces after `bm_to_mesh`.
 */
static void write_face_kind(Mesh &mesh, BMesh *bm, const int original_offset)
{
  const int face_num = mesh.faces_num;
  if (face_num == 0) {
    return;
  }
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  bke::SpanAttributeWriter<bool> original = attributes.lookup_or_add_for_write_span<bool>(
      "original_face", AttrDomain::Face);
  bke::SpanAttributeWriter<bool> cut = attributes.lookup_or_add_for_write_span<bool>("cut_face",
                                                                                     AttrDomain::Face);
  BMIter iter;
  BMFace *face;
  int i = 0;
  BM_ITER_MESH (face, &iter, bm, BM_FACES_OF_MESH) {
    if (i >= face_num) {
      break;
    }
    const bool is_original = BM_ELEM_CD_GET_BOOL(face, original_offset);
    original.span[i] = is_original;
    cut.span[i] = !is_original;
    i++;
  }
  for (; i < face_num; i++) {
    original.span[i] = true;
    cut.span[i] = false;
  }
  original.finish();
  cut.finish();
}

/* -------------------------------------------------------------------- */
/** \name Interior cells
 *
 * A cell whose ball contains no triangle of the surface lies completely inside the mesh (or
 * completely outside it). Its piece is just the convex cell, so it can be built without a BMesh:
 * the cell boundary is the intersection of the half spaces, which is a convex polygon per plane.
 *
 * This is the common case for a deep fracture (a 11x11x11 grid has 512 interior cells out of 1000),
 * and it skips the whole mesh copy that the BMesh path needs for every cell.
 * \{ */

/** Diagnostics, enabled with `BLENDER_FRACTURE_DEBUG=1`. */
#define FRACTURE_DEBUG(...) \
  do { \
    if (std::getenv("BLENDER_FRACTURE_DEBUG") != nullptr) { \
      printf(__VA_ARGS__); \
      fflush(stdout); \
    } \
  } while (0)

/** A cell boundary: the kept side is `dot(p - co, no) <= 0`, `no` is normalized. */
struct CellPlane {
  float3 co;
  float3 no;
};

/** Is there any surface triangle within `radius` of `position`? */
static bool surface_near(const Mesh &source, const float3 &position, const float radius)
{
  const bke::BVHTreeFromMesh bvh = source.bvh_corner_tris();
  if (bvh.tree == nullptr) {
    return true;
  }
  bool found = false;
  BLI_bvhtree_range_query(
      bvh.tree,
      position,
      radius,
      [](void *user_data, int /*index*/, const float /*co*/[3], float /*dist_sq*/) {
        *static_cast<bool *>(user_data) = true;
      },
      &found);
  return found;
}

/** Is the point inside the (closed) mesh? Nearest surface point + the normal there. */
static bool point_inside_mesh(const bke::BVHTreeFromMesh &bvh, const float3 &position)
{
  if (bvh.tree == nullptr) {
    return false;
  }
  BVHTreeNearest nearest;
  nearest.index = -1;
  nearest.dist_sq = FLT_MAX;
  const float co[3] = {position.x, position.y, position.z};
  if (BLI_bvhtree_find_nearest(bvh.tree, co, &nearest, nullptr, nullptr) < 0) {
    return false;
  }
  const float3 delta = position - float3(nearest.co);
  if (math::length_squared(delta) <= 1.0e-8f) {
    return true;
  }
  const int3 tri = bvh.corner_tris[nearest.index];
  const float3 &a = bvh.vert_positions[bvh.corner_verts[tri[0]]];
  const float3 &b = bvh.vert_positions[bvh.corner_verts[tri[1]]];
  const float3 &c = bvh.vert_positions[bvh.corner_verts[tri[2]]];
  return math::dot(delta, math::cross(b - a, c - a)) < 0.0f;
}

/** The instance copy of a fragment: the mesh moved so that the fragment center is at the origin. */
static Mesh *fragment_local_copy(const Mesh &world, const float3 &center)
{
  Mesh *local = BKE_mesh_copy_for_eval(world);
  MutableSpan<float3> positions = local->vert_positions_for_write();
  for (float3 &position : positions) {
    position -= center;
  }
  return local;
}


/** 2D basis of a cap plane. */
struct CapFrame {
  float3 tangent;
  float3 bitangent;
};

static CapFrame cap_frame(const CellPlane &plane)
{
  CapFrame frame;
  frame.tangent = math::normalize(math::cross(
      plane.no, std::abs(plane.no.z) < 0.9f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f)));
  frame.bitangent = math::cross(plane.no, frame.tangent);
  return frame;
}

/**
 * The part of `planes[plane_index]` that lies inside the cell, as a convex polygon in the 2D frame of
 * that plane (coordinates are relative to the plane origin).
 *
 * \param r_unbounded: set when the polygon still touches the starting square, i.e. no plane bounds
 *   the cell in that direction.
 * \returns false when the cell does not reach the plane at all.
 */
static bool cell_cap_polygon(const Span<CellPlane> planes,
                             const int plane_index,
                             const float radius,
                             Vector<float2> &r_polygon,
                             bool &r_unbounded)
{
  const CellPlane &plane = planes[plane_index];
  const CapFrame frame = cap_frame(plane);
  /* The starting square is deliberately much larger than the cell: a bounded cell can reach out to
   * `radius` (that is the bound the caller uses), so the square has to be clearly beyond it, and a
   * vertex that is still *on* the square means no plane bounds the cell in that direction. */
  const float half = std::max(radius * 4.0f, 1.0e-3f);
  r_unbounded = false;

  r_polygon.clear();
  r_polygon.append(float2(-half, -half));
  r_polygon.append(float2(half, -half));
  r_polygon.append(float2(half, half));
  r_polygon.append(float2(-half, half));
  for (const int other_index : planes.index_range()) {
    if (other_index == plane_index) {
      continue;
    }
    const CellPlane &other = planes[other_index];
    const float2 normal(math::dot(other.no, frame.tangent), math::dot(other.no, frame.bitangent));
    /* The half space is `dot(p - other.co, other.no) <= 0` and a point of this plane is
     * `plane.co + t x + b y`, which gives `x n_t + y n_b <= dot(other.co - plane.co, other.no)`. */
    const float limit = math::dot(other.co - plane.co, other.no);
    Vector<float2> clipped;
    clipped.reserve(r_polygon.size() + 2);
    for (const int i : r_polygon.index_range()) {
      const float2 &a = r_polygon[i];
      const float2 &b = r_polygon[(i + 1) % r_polygon.size()];
      const float da = math::dot(a, normal) - limit;
      const float db = math::dot(b, normal) - limit;
      if (da <= 0.0f) {
        clipped.append(a);
      }
      if ((da > 0.0f) != (db > 0.0f)) {
        clipped.append(a + (b - a) * (da / (da - db)));
      }
    }
    r_polygon = std::move(clipped);
    if (r_polygon.size() < 3) {
      return false;
    }
  }
  for (const float2 &point : r_polygon) {
    if (math::length_squared(point) > half * half * 0.98f) {
      r_unbounded = true;
      break;
    }
  }
  return true;
}

/**
 * True when the planes close the cell on all sides. Only then a local cut is valid: whatever is
 * outside the cell (including the lid that closes a local surface patch) is cut away by the planes.
 */
static bool cell_is_bounded(const Span<CellPlane> planes, const float radius)
{
  if (planes.is_empty()) {
    return false;
  }
  Vector<float2> polygon;
  for (const int plane_index : planes.index_range()) {
    bool unbounded = false;
    cell_cap_polygon(planes, plane_index, radius, polygon, unbounded);
    if (unbounded) {
      return false;
    }
  }
  return true;
}

static void fill_holes_all(BMesh *bm, const int original_offset)
{
  /* Only boundary edges may be passed to holes_fill, otherwise the edge-net fill has nothing
   * to work with. */
  BM_mesh_elem_hflag_disable_all(bm, BM_EDGE, BM_ELEM_TAG, false);
  int boundary_edges = 0;
  BMIter iter;
  BMEdge *edge;
  BM_ITER_MESH (edge, &iter, bm, BM_EDGES_OF_MESH) {
    if (BM_edge_is_boundary(edge)) {
      BM_elem_flag_enable(edge, BM_ELEM_TAG);
      boundary_edges++;
    }
  }
  if (boundary_edges == 0) {
    return;
  }

  Set<const BMFace *> before;
  BMFace *face;
  BM_ITER_MESH (face, &iter, bm, BM_FACES_OF_MESH) {
    before.add(face);
  }

  BMOperator op;
  BMO_op_init(bm, &op, BMO_FLAG_DEFAULTS, "holes_fill");
  BMO_slot_buffer_from_enabled_hflag(bm, &op, op.slots_in, "edges", BM_EDGE, BM_ELEM_TAG);
  BMO_slot_int_set(op.slots_in, "sides", 0);
  BMO_op_exec(bm, &op);
  BMO_op_finish(bm, &op);

  mark_new_faces(bm, original_offset, before);
}

/**
 * Build the piece of a cell that does not touch the surface: the convex cell polyhedron.
 * Returns null when the planes do not bound the cell (then the BMesh path handles it).
 */
static std::optional<Fragment> interior_cell(const Span<CellPlane> planes,
                                             const float radius)
{
  if (planes.is_empty()) {
    return std::nullopt;
  }

  Vector<float3> verts;
  Map<uint64_t, int> vert_of_key;
  Vector<int> corner_verts;
  Vector<int> face_offsets;
  face_offsets.append(0);

  /* The corners of a cap lie on the intersection of two planes, so they are shared with the other
   * caps: they are welded by (quantized) position. */
  auto vertex_of_position = [&](const float3 &position) {
    const int32_t qx = int32_t(std::lround(double(position.x) * 1.0e5));
    const int32_t qy = int32_t(std::lround(double(position.y) * 1.0e5));
    const int32_t qz = int32_t(std::lround(double(position.z) * 1.0e5));
    uint64_t key = 0xc000000000000000ull;
    key ^= uint64_t(uint32_t(qx)) * 0x9e3779b97f4a7c15ull;
    key ^= uint64_t(uint32_t(qy)) * 0xc2b2ae3d27d4eb4full;
    key ^= uint64_t(uint32_t(qz)) * 0x165667b19e3779f9ull;
    key |= 0xc000000000000000ull;
    if (const int *existing = vert_of_key.lookup_ptr(key)) {
      return *existing;
    }
    const int index = int(verts.size());
    verts.append(position);
    vert_of_key.add(key, index);
    return index;
  };

  Vector<float2> polygon;
  Vector<int> cap_verts;

  for (const int plane_index : planes.index_range()) {
    const CellPlane &plane = planes[plane_index];
    const CapFrame frame = cap_frame(plane);
    bool unbounded = false;
    if (!cell_cap_polygon(planes, plane_index, radius, polygon, unbounded)) {
      /* The cell does not reach this plane, so it does not get a cap here. */
      FRACTURE_DEBUG("interior: plane %d empty\n", plane_index);
      continue;
    }
    if (unbounded) {
      /* No plane bounds the cell in that direction, so it is not a convex cell (the caller handles
       * those cells differently). */
      FRACTURE_DEBUG("interior: plane %d unbounded\n", plane_index);
      return std::nullopt;
    }
    /* Counter clockwise in the plane frame gives a normal along `plane.no`, which points away from
     * the cell. */
    float area2 = 0.0f;
    for (const int i : polygon.index_range()) {
      const float2 &a = polygon[i];
      const float2 &b = polygon[(i + 1) % polygon.size()];
      area2 += a.x * b.y - a.y * b.x;
    }
    if (std::abs(area2) < 1.0e-14f) {
      FRACTURE_DEBUG("interior: plane %d degenerate (area2 %g)\n", plane_index, double(area2));
      continue;
    }
    if (area2 < 0.0f) {
      std::reverse(polygon.begin(), polygon.end());
    }
    cap_verts.clear();
    for (const float2 &point : polygon) {
      cap_verts.append(vertex_of_position(plane.co + frame.tangent * point.x +
                                          frame.bitangent * point.y));
    }
    /* One face per cap: it is convex, so it does not have to be triangulated (the BMesh path keeps
     * n-gons as well). Neighbouring corners can weld to the same vertex when two cell corners are
     * closer than the weld resolution; a face with a repeated vertex would be invalid (and leave a
     * hole), so those are dropped. */
    int unique_verts[64];
    int unique_num = 0;
    for (const int vertex : cap_verts) {
      if (unique_num > 0 && unique_verts[unique_num - 1] == vertex) {
        continue;
      }
      if (unique_num == 64) {
        unique_num = 0;
        break;
      }
      unique_verts[unique_num++] = vertex;
    }
    while (unique_num >= 2 && unique_verts[0] == unique_verts[unique_num - 1]) {
      unique_num--;
    }
    bool repeated = false;
    for (int i = 0; i < unique_num && !repeated; i++) {
      for (int j = i + 1; j < unique_num; j++) {
        if (unique_verts[i] == unique_verts[j]) {
          repeated = true;
          break;
        }
      }
    }
    if (unique_num < 3 || repeated) {
      continue;
    }
    for (int i = 0; i < unique_num; i++) {
      corner_verts.append(unique_verts[i]);
    }
    face_offsets.append(int(corner_verts.size()));
  }

  if (face_offsets.size() < 2) {
    FRACTURE_DEBUG("interior: no caps\n");
    return std::nullopt;
  }

  /* The edges are stored explicitly in this version (`edges_for_write`), they are not derived from
   * the corners, so they have to be built here as well. */
  const int verts_num = int(verts.size());
  const int faces_num = int(face_offsets.size()) - 1;
  const int corners_num = int(corner_verts.size());
  Array<int> corner_edges(corners_num);
  Vector<int2> edge_verts;
  Map<uint64_t, int> edge_of_key;
  for (const int face : IndexRange(faces_num)) {
    const int start = face_offsets[face];
    const int size = face_offsets[face + 1] - start;
    for (const int i : IndexRange(size)) {
      const int va = corner_verts[start + i];
      const int vb = corner_verts[start + (i + 1) % size];
      const uint64_t key = (uint64_t(uint32_t(std::min(va, vb))) << 32) | uint64_t(uint32_t(std::max(va, vb)));
      const int edge = edge_of_key.lookup_or_add(key, int(edge_verts.size()));
      if (edge == edge_verts.size()) {
        edge_verts.append(int2(va, vb));
      }
      corner_edges[start + i] = edge;
    }
  }

  Fragment fragment;
  fragment.bounds_min = float3(FLT_MAX);
  fragment.bounds_max = float3(-FLT_MAX);
  for (const float3 &position : verts) {
    fragment.bounds_min = math::min(fragment.bounds_min, position);
    fragment.bounds_max = math::max(fragment.bounds_max, position);
  }
  fragment.center = (fragment.bounds_min + fragment.bounds_max) * 0.5f;
  /* A bounded cell always has volume. */
  fragment.degenerate = (fragment.bounds_max - fragment.bounds_min).x < 1.0e-6f;

  Mesh *mesh = BKE_mesh_new_nomain(verts_num, int(edge_verts.size()), faces_num, corners_num);
  MutableSpan<float3> positions = mesh->vert_positions_for_write();
  positions.copy_from(verts);
  mesh->edges_for_write().copy_from(edge_verts);
  mesh->corner_verts_for_write().copy_from(corner_verts);
  mesh->corner_edges_for_write().copy_from(corner_edges);
  mesh->face_offsets_for_write().copy_from(face_offsets);
  {
    /* No face of an interior piece comes from the surface: all of them are cut caps. */
    bke::MutableAttributeAccessor attributes = mesh->attributes_for_write();
    bke::SpanAttributeWriter<bool> original = attributes.lookup_or_add_for_write_span<bool>(
        "original_face", AttrDomain::Face);
    bke::SpanAttributeWriter<bool> cut = attributes.lookup_or_add_for_write_span<bool>(
        "cut_face", AttrDomain::Face);
    for (const int i : original.span.index_range()) {
      original.span[i] = false;
      cut.span[i] = true;
    }
    original.finish();
    cut.finish();
  }

  FRACTURE_DEBUG("interior cell: planes %d verts %d faces %d\n",
                 int(planes.size()), verts_num, faces_num);
  fragment.mesh_world = mesh;
  fragment.mesh_local = fragment_local_copy(*mesh, fragment.center);
  return fragment;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Direct shell clipping
 *
 * Shell mode only keeps the surface: the cuts are not capped, so a piece is simply the part of the
 * surface that lies inside its cell. That does not need a BMesh at all. The triangles near the cell
 * are clipped against the cell planes on the raw arrays, and the vertices are welded through
 * explicit keys (source edge + parameter), so the cost is proportional to the triangles of the piece
 * instead of the size of the whole mesh, and the pieces stay watertight among the same cell.
 * \{ */

/** Where an output vertex takes its attributes from: a source triangle and barycentric weights. */
struct ShellVertexRef {
  int tri = -1;
  float3 bary = float3(1.0f, 0.0f, 0.0f);
  /** Source point when the vertex is an untouched corner of the source mesh, -1 for a cut. */
  int point = -1;
};

/** The mesh of one cell (shell mode). */
struct ShellCell {
  Vector<float3> verts;
  Vector<ShellVertexRef> vert_refs;
  Map<uint64_t, int> vert_of_key;
  int interior_keys = 0;

  Vector<int> corner_verts;
  Vector<ShellVertexRef> corner_refs;
  Vector<int> face_offsets;
  Vector<int> face_src;
};

/** Key of a vertex on the source edge `(a, b)` at `t` (quantized to 16 bits). */
static uint64_t shell_edge_key(const int a, const int b, const int t_quantized)
{
  const int lo = std::min(a, b);
  const int hi = std::max(a, b);
  return (uint64_t(uint32_t(lo)) << 40) | (uint64_t(uint32_t(hi)) << 16) |
         uint64_t(uint32_t(t_quantized) & 0xffffu);
}

/** Unordered pair of vertices, for the edge deduplication and the source edge lookup. */
static uint64_t shell_pair_key(const int a, const int b)
{
  const uint64_t lo = uint64_t(uint32_t(std::min(a, b)));
  const uint64_t hi = uint64_t(uint32_t(std::max(a, b)));
  return (lo << 32) | hi;
}

static int shell_vertex(ShellCell &cell,
                        const uint64_t key,
                        const float3 &position,
                        const ShellVertexRef &ref)
{
  if (const int *existing = cell.vert_of_key.lookup_ptr(key)) {
    return *existing;
  }
  const int index = int(cell.verts.size());
  cell.verts.append(position);
  cell.vert_refs.append(ref);
  cell.vert_of_key.add(key, index);
  return index;
}

/** A vertex of the polygon that is being clipped. */
struct ShellPolyVert {
  int vert;
  float3 bary;
};

template<typename T> static T shell_blend3(const float3 &weights, const T &v0, const T &v1, const T &v2)
{
  if constexpr (std::is_same_v<T, float>) {
    return v0 * weights.x + v1 * weights.y + v2 * weights.z;
  }
  else if constexpr (std::is_same_v<T, float2>) {
    return v0 * weights.x + v1 * weights.y + v2 * weights.z;
  }
  else if constexpr (std::is_same_v<T, float3>) {
    return v0 * weights.x + v1 * weights.y + v2 * weights.z;
  }
  else if constexpr (std::is_same_v<T, float4>) {
    return v0 * weights.x + v1 * weights.y + v2 * weights.z;
  }
  else if constexpr (std::is_same_v<T, ColorGeometry4f>) {
    ColorGeometry4f result;
    result.r = v0.r * weights.x + v1.r * weights.y + v2.r * weights.z;
    result.g = v0.g * weights.x + v1.g * weights.y + v2.g * weights.z;
    result.b = v0.b * weights.x + v1.b * weights.y + v2.b * weights.z;
    result.a = v0.a * weights.x + v1.a * weights.y + v2.a * weights.z;
    return result;
  }
  else {
    /* Integers, booleans and byte colors can not be mixed meaningfully: take the nearest. */
    if (weights.x >= weights.y && weights.x >= weights.z) {
      return v0;
    }
    return weights.y >= weights.z ? v1 : v2;
  }
}

template<typename T>
static void shell_transfer_point(const bke::GAttributeReader &reader,
                                 bke::MutableAttributeAccessor &dst_attributes,
                                 const StringRef name,
                                 const ShellCell &cell,
                                 const Mesh &source)
{
  const VArray<T> values = reader.varray.typed<T>();
  bke::SpanAttributeWriter<T> writer = dst_attributes.lookup_or_add_for_write_span<T>(
      name, AttrDomain::Point);
  const Span<int3> corner_tris = source.corner_tris();
  const Span<int> corner_verts = source.corner_verts();
  for (const int i : writer.span.index_range()) {
    const ShellVertexRef &ref = cell.vert_refs[i];
    if (ref.tri < 0) {
      writer.span[i] = T();
      continue;
    }
    const int3 corners = corner_tris[ref.tri];
    writer.span[i] = shell_blend3(ref.bary,
                                  values[corner_verts[corners[0]]],
                                  values[corner_verts[corners[1]]],
                                  values[corner_verts[corners[2]]]);
  }
  writer.finish();
}

template<typename T>
static void shell_transfer_corner(const bke::GAttributeReader &reader,
                                  bke::MutableAttributeAccessor &dst_attributes,
                                  const StringRef name,
                                  const ShellCell &cell,
                                  const Mesh &source)
{
  const VArray<T> values = reader.varray.typed<T>();
  bke::SpanAttributeWriter<T> writer = dst_attributes.lookup_or_add_for_write_span<T>(
      name, AttrDomain::Corner);
  const Span<int3> corner_tris = source.corner_tris();
  for (const int i : writer.span.index_range()) {
    const ShellVertexRef &ref = cell.corner_refs[i];
    if (ref.tri < 0) {
      writer.span[i] = T();
      continue;
    }
    const int3 corners = corner_tris[ref.tri];
    writer.span[i] = shell_blend3(ref.bary, values[corners[0]], values[corners[1]], values[corners[2]]);
  }
  writer.finish();
}

template<typename T>
static void shell_transfer_face(const bke::GAttributeReader &reader,
                                bke::MutableAttributeAccessor &dst_attributes,
                                const StringRef name,
                                const ShellCell &cell)
{
  const VArray<T> values = reader.varray.typed<T>();
  bke::SpanAttributeWriter<T> writer = dst_attributes.lookup_or_add_for_write_span<T>(
      name, AttrDomain::Face);
  for (const int i : writer.span.index_range()) {
    const int src_face = cell.face_src[i];
    writer.span[i] = src_face >= 0 ? values[src_face] : T();
  }
  writer.finish();
}

template<typename T>
static void shell_transfer_edge(const bke::GAttributeReader &reader,
                               bke::MutableAttributeAccessor &dst_attributes,
                               const StringRef name,
                               const ShellCell &cell,
                               const Span<int2> edge_verts,
                               const Map<uint64_t, int> &source_edge_of_verts)
{
  const VArray<T> values = reader.varray.typed<T>();
  bke::SpanAttributeWriter<T> writer = dst_attributes.lookup_or_add_for_write_span<T>(
      name, AttrDomain::Edge);
  for (const int i : writer.span.index_range()) {
    /* A cut edge has no source edge, so it keeps the default value. */
    const int2 edge = edge_verts[i];
    const int point_a = cell.vert_refs[edge.x].point;
    const int point_b = cell.vert_refs[edge.y].point;
    if (point_a < 0 || point_b < 0) {
      writer.span[i] = T();
      continue;
    }
    const int *source_edge = source_edge_of_verts.lookup_ptr(shell_pair_key(point_a, point_b));
    writer.span[i] = source_edge != nullptr ? values[*source_edge] : T();
  }
  writer.finish();
}

/** Copy the attributes of the source mesh onto a shell piece. */
static void shell_transfer_attributes(const Mesh &source,
                                      Mesh &dst,
                                      const ShellCell &cell,
                                      const Span<int2> dst_edges,
                                      const Map<uint64_t, int> &source_edge_of_verts)
{
  const bke::AttributeAccessor src_attributes = source.attributes();
  bke::MutableAttributeAccessor dst_attributes = dst.attributes_for_write();
  const Span<int2> source_edges = source.edges();

  src_attributes.foreach_attribute([&](const bke::AttributeIter &iter) {
    const StringRef name = iter.name;
    /* Skip `position` (it is written from the clipped points) and the internal attributes of the
     * mesh (`.corner_vert`, `.corner_edge`, `.edge_verts`, ...): copying those would overwrite the
     * topology that was just built. */
    if (name == "position" || name.startswith(".")) {
      return;
    }
    const bke::GAttributeReader reader = iter.get();
    if (!reader) {
      return;
    }
    switch (iter.domain) {
      case bke::AttrDomain::Point: {
        switch (iter.data_type) {
          case bke::AttrType::Float: shell_transfer_point<float>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Float2: shell_transfer_point<float2>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Float3: shell_transfer_point<float3>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Float4: shell_transfer_point<float4>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::ColorFloat: shell_transfer_point<ColorGeometry4f>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::ColorByte: shell_transfer_point<ColorGeometry4b>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Int32: shell_transfer_point<int>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Int32_2D: shell_transfer_point<int2>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Int8: shell_transfer_point<int8_t>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Bool: shell_transfer_point<bool>(reader, dst_attributes, name, cell, source); break;
          default: break;
        }
        break;
      }
      case bke::AttrDomain::Corner: {
        switch (iter.data_type) {
          case bke::AttrType::Float: shell_transfer_corner<float>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Float2: shell_transfer_corner<float2>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Float3: shell_transfer_corner<float3>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Float4: shell_transfer_corner<float4>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::ColorFloat: shell_transfer_corner<ColorGeometry4f>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::ColorByte: shell_transfer_corner<ColorGeometry4b>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Int32: shell_transfer_corner<int>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Int32_2D: shell_transfer_corner<int2>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Int8: shell_transfer_corner<int8_t>(reader, dst_attributes, name, cell, source); break;
          case bke::AttrType::Bool: shell_transfer_corner<bool>(reader, dst_attributes, name, cell, source); break;
          default: break;
        }
        break;
      }
      case bke::AttrDomain::Face: {
        switch (iter.data_type) {
          case bke::AttrType::Float: shell_transfer_face<float>(reader, dst_attributes, name, cell); break;
          case bke::AttrType::Float2: shell_transfer_face<float2>(reader, dst_attributes, name, cell); break;
          case bke::AttrType::Float3: shell_transfer_face<float3>(reader, dst_attributes, name, cell); break;
          case bke::AttrType::ColorFloat: shell_transfer_face<ColorGeometry4f>(reader, dst_attributes, name, cell); break;
          case bke::AttrType::ColorByte: shell_transfer_face<ColorGeometry4b>(reader, dst_attributes, name, cell); break;
          case bke::AttrType::Int32: shell_transfer_face<int>(reader, dst_attributes, name, cell); break;
          case bke::AttrType::Int8: shell_transfer_face<int8_t>(reader, dst_attributes, name, cell); break;
          case bke::AttrType::Bool: shell_transfer_face<bool>(reader, dst_attributes, name, cell); break;
          default: break;
        }
        break;
      }
      case bke::AttrDomain::Edge: {
        /* Seams and sharp edges: copied for the edges that are still source edges. */
        switch (iter.data_type) {
          case bke::AttrType::Bool: shell_transfer_edge<bool>(reader, dst_attributes, name, cell, dst_edges, source_edge_of_verts); break;
          case bke::AttrType::Float: shell_transfer_edge<float>(reader, dst_attributes, name, cell, dst_edges, source_edge_of_verts); break;
          case bke::AttrType::Int32: shell_transfer_edge<int>(reader, dst_attributes, name, cell, dst_edges, source_edge_of_verts); break;
          default: break;
        }
        break;
      }
      default:
        break;
    }
  });
  UNUSED_VARS(source_edges);
}

/** Turn the accumulated shell cell into a mesh (positions offset by `offset`). */
static Mesh *shell_cell_to_mesh(const Mesh &source,
                                const ShellCell &cell,
                                const float3 offset,
                                const Map<uint64_t, int> &source_edge_of_verts)
{
  const int verts_num = int(cell.verts.size());
  const int faces_num = int(cell.face_src.size());
  const int corners_num = int(cell.corner_verts.size());

  Array<int> corner_edges(corners_num);
  Vector<int2> edge_verts;
  Map<uint64_t, int> edge_of_key;
  for (const int face : IndexRange(faces_num)) {
    const int start = cell.face_offsets[face];
    const int size = cell.face_offsets[face + 1] - start;
    for (const int i : IndexRange(size)) {
      const int va = cell.corner_verts[start + i];
      const int vb = cell.corner_verts[start + (i + 1) % size];
      const int edge = edge_of_key.lookup_or_add(shell_pair_key(va, vb), int(edge_verts.size()));
      if (edge == edge_verts.size()) {
        edge_verts.append(int2(va, vb));
      }
      corner_edges[start + i] = edge;
    }
  }

  Mesh *mesh = BKE_mesh_new_nomain(verts_num, int(edge_verts.size()), faces_num, corners_num);
  MutableSpan<float3> positions = mesh->vert_positions_for_write();
  for (const int i : IndexRange(verts_num)) {
    positions[i] = cell.verts[i] + offset;
  }
  mesh->edges_for_write().copy_from(edge_verts);
  mesh->corner_verts_for_write().copy_from(cell.corner_verts);
  mesh->corner_edges_for_write().copy_from(corner_edges);
  mesh->face_offsets_for_write().copy_from(cell.face_offsets);

  shell_transfer_attributes(source, *mesh, cell, edge_verts, source_edge_of_verts);

  /* Every face of a shell piece comes from the surface, none is a cut cap. */
  bke::MutableAttributeAccessor attributes = mesh->attributes_for_write();
  bke::SpanAttributeWriter<bool> original = attributes.lookup_or_add_for_write_span<bool>(
      "original_face", AttrDomain::Face);
  bke::SpanAttributeWriter<bool> cut = attributes.lookup_or_add_for_write_span<bool>(
      "cut_face", AttrDomain::Face);
  for (const int i : original.span.index_range()) {
    original.span[i] = true;
    cut.span[i] = false;
  }
  original.finish();
  cut.finish();
  return mesh;
}

/**
 * Cut one cell out of the surface, for shell mode. Only the triangles near the cell are visited and
 * nothing is capped, so this needs neither a BMesh nor a closed input.
 */
static std::optional<Fragment> shell_cell(const Mesh &source,
                                          const Span<CellPlane> planes,
                                          const float3 site,
                                          const float radius,
                                          const Map<uint64_t, int> &source_edge_of_verts)
{
  const bke::BVHTreeFromMesh bvh = source.bvh_corner_tris();
  if (bvh.tree == nullptr) {
    return std::nullopt;
  }
  const Span<float3> positions = source.vert_positions();
  const Span<int3> corner_tris = source.corner_tris();
  const Span<int> corner_tri_faces = source.corner_tri_faces();
  const Span<int> corner_verts = source.corner_verts();

  const float epsilon = std::max(radius, 1.0f) * 1.0e-6f;

  /* The triangles near the cell. Everything of the cell is within `radius` of the site. */
  Vector<int> tris;
  BLI_bvhtree_range_query(
      bvh.tree,
      site,
      radius * 1.001f + epsilon,
      [](void *user_data, int index, const float /*co*/[3], float /*dist_sq*/) {
        static_cast<Vector<int> *>(user_data)->append(index);
      },
      &tris);
  if (tris.is_empty()) {
    return std::nullopt;
  }

  ShellCell cell;
  cell.face_offsets.append(0);

  Vector<ShellPolyVert, 8> poly;
  Vector<ShellPolyVert, 8> next;

  for (const int tri : tris) {
    const int3 corners = corner_tris[tri];
    int points[3];
    float3 tri_pos[3];
    for (const int k : IndexRange(3)) {
      points[k] = corner_verts[corners[k]];
      tri_pos[k] = positions[points[k]];
    }

    poly.clear();
    for (const int k : IndexRange(3)) {
      float3 bary(0.0f);
      bary[k] = 1.0f;
      const int vert = shell_vertex(cell,
                                    shell_edge_key(points[k], points[k], 0),
                                    tri_pos[k],
                                    ShellVertexRef{tri, bary, points[k]});
      poly.append({vert, bary});
    }

    bool discarded = false;
    for (const CellPlane &plane : planes) {
      bool any_inside = false;
      bool any_outside = false;
      for (const ShellPolyVert &vertex : poly) {
        const float distance = math::dot(cell.verts[vertex.vert] - plane.co, plane.no);
        any_inside |= distance <= epsilon;
        any_outside |= distance > epsilon;
      }
      if (!any_outside) {
        continue;
      }
      if (!any_inside) {
        discarded = true;
        break;
      }
      next.clear();
      for (const int i : poly.index_range()) {
        const ShellPolyVert &a = poly[i];
        const ShellPolyVert &b = poly[(i + 1) % poly.size()];
        const float da = math::dot(cell.verts[a.vert] - plane.co, plane.no);
        const float db = math::dot(cell.verts[b.vert] - plane.co, plane.no);
        if (da <= epsilon) {
          next.append(a);
        }
        if ((da > epsilon) != (db > epsilon)) {
          const float s = da / (da - db);
          float3 bary = math::interpolate(a.bary, b.bary, s);
          /* Snap a barycentric coordinate that is on an edge of the triangle, so both triangles that
           * share that edge compute the same key (and therefore the same vertex). */
          int zeros = 0;
          int zero_axis = -1;
          for (const int k : IndexRange(3)) {
            if (bary[k] == 0.0f) {
              zeros++;
              zero_axis = k;
            }
          }
          uint64_t key;
          int point = -1;
          if (zeros == 2) {
            point = points[(zero_axis + 1) % 3];
            key = shell_edge_key(point, point, 0);
          }
          else if (zeros == 1) {
            const int a_axis = (zero_axis + 1) % 3;
            const int b_axis = (zero_axis + 2) % 3;
            const float t = (points[a_axis] <= points[b_axis]) ? bary[b_axis] : bary[a_axis];
            const int t_quantized = int(std::lround(t * 65536.0f));
            const float snapped_t = float(t_quantized) / 65536.0f;
            bary = float3(0.0f);
            if (points[a_axis] <= points[b_axis]) {
              bary[a_axis] = 1.0f - snapped_t;
              bary[b_axis] = snapped_t;
            }
            else {
              bary[a_axis] = snapped_t;
              bary[b_axis] = 1.0f - snapped_t;
            }
            key = shell_edge_key(points[a_axis], points[b_axis], t_quantized);
          }
          else {
            key = 0x8000000000000000ull | uint64_t(uint32_t(cell.interior_keys++));
          }
          const float3 position = tri_pos[0] * bary.x + tri_pos[1] * bary.y + tri_pos[2] * bary.z;
          const int vert = shell_vertex(cell, key, position, ShellVertexRef{tri, bary, point});
          next.append({vert, bary});
        }
      }
      poly = next;
      if (poly.size() < 3) {
        discarded = true;
        break;
      }
    }
    if (discarded) {
      continue;
    }

    /* Drop the vertices that repeat their neighbour: clipping can weld a cut point onto a corner. */
    int unique_verts[16];
    ShellVertexRef unique_refs[16];
    int unique_num = 0;
    for (const int i : poly.index_range()) {
      if (unique_num > 0 && unique_verts[unique_num - 1] == poly[i].vert) {
        continue;
      }
      if (unique_num == 16) {
        unique_num = 0;
        break;
      }
      unique_verts[unique_num] = poly[i].vert;
      unique_refs[unique_num] = ShellVertexRef{tri, poly[i].bary, cell.vert_refs[poly[i].vert].point};
      unique_num++;
    }
    while (unique_num >= 2 && unique_verts[0] == unique_verts[unique_num - 1]) {
      unique_num--;
    }
    if (unique_num < 3) {
      continue;
    }
    bool repeated = false;
    for (int i = 0; i < unique_num && !repeated; i++) {
      for (int j = i + 1; j < unique_num; j++) {
        if (unique_verts[i] == unique_verts[j]) {
          repeated = true;
          break;
        }
      }
    }
    if (repeated) {
      continue;
    }
    /* A polygon that collapsed to a line has no area. */
    float area = 0.0f;
    {
      float3 sum(0.0f);
      const float3 origin = cell.verts[unique_verts[0]];
      for (int i = 1; i < unique_num - 1; i++) {
        sum += math::cross(cell.verts[unique_verts[i]] - origin,
                           cell.verts[unique_verts[i + 1]] - origin);
      }
      area = math::length(sum) * 0.5f;
    }
    if (area < 1.0e-12f) {
      continue;
    }

    for (int i = 0; i < unique_num; i++) {
      cell.corner_verts.append(unique_verts[i]);
      cell.corner_refs.append(unique_refs[i]);
    }
    cell.face_offsets.append(int(cell.corner_verts.size()));
    cell.face_src.append(corner_tri_faces[tri]);
  }

  if (cell.face_offsets.size() < 2) {
    return std::nullopt;
  }

  Fragment fragment;
  fragment.bounds_min = float3(FLT_MAX);
  fragment.bounds_max = float3(-FLT_MAX);
  for (const float3 &position : cell.verts) {
    fragment.bounds_min = math::min(fragment.bounds_min, position);
    fragment.bounds_max = math::max(fragment.bounds_max, position);
  }
  fragment.center = (fragment.bounds_min + fragment.bounds_max) * 0.5f;
  /* A shell piece has no volume, so it is never degenerate. */
  fragment.degenerate = false;

  FRACTURE_DEBUG("shell cell: tris %d verts %d faces %d\n",
                 int(tris.size()),
                 int(cell.verts.size()),
                 int(cell.face_src.size()));

  fragment.mesh_world = shell_cell_to_mesh(source, cell, float3(0.0f), source_edge_of_verts);
  fragment.mesh_local = fragment_local_copy(*fragment.mesh_world, fragment.center);
  return fragment;
}

/** \} */

/**
 * Cut `mesh` with the planes of one cell (BMesh path).
 *
 * \param close_patch: close the mesh before cutting. A local surface patch is an open surface, and
 *   `fill_holes` can only cap the cuts of a closed mesh; the lid it adds sits outside the cell (the
 *   patch is gathered within the cell radius), so the planes cut it away again.
 */
static std::optional<Fragment> fracture_mesh_with_planes(const Mesh &mesh,
                                                         const Span<CellPlane> planes,
                                                         const float radius)
{
  /* Shell mode does not come through here any more (it is clipped directly, see `shell_cell`), so the
   * pieces of this path are always closed. */
  constexpr bool shell_mode = false;
  BMeshCreateParams create_params{};
  create_params.use_toolflags = true;
  BMesh *bm = BM_mesh_create(&bm_mesh_allocsize_default, &create_params);
  BMeshFromMeshParams from_params{};
  from_params.calc_face_normal = true;
  BM_mesh_bm_from_me(bm, &mesh, &from_params);

  /* A face custom data layer marks the faces of the source mesh. The cuts split them and the halves
   * inherit the value, while the hole filling creates new faces, which `mark_new_faces` clears
   * again. That is what tells the original surface from the cut caps afterwards. */
  BM_data_layer_add_named(bm, &bm->pdata, CD_PROP_BOOL, "original_face");
  const int original_offset = CustomData_get_offset_named(
      &bm->pdata, CD_PROP_BOOL, "original_face");
  {
    BMIter iter;
    BMFace *face;
    BM_ITER_MESH (face, &iter, bm, BM_FACES_OF_MESH) {
      BM_ELEM_CD_SET_BOOL(face, original_offset, true);
    }
  }

  for (const CellPlane &plane : planes) {
    const float3 plane_co = plane.co;
    const float3 plane_no = plane.no;
    /* Every bisect + hole fill is an expensive BMesh operator call (~130 us each), and after the
     * first cuts the fragment is much smaller than the cell: most of the remaining planes do not
     * touch it at all. Testing the current vertices against the plane costs almost nothing and
     * skips those cuts completely (and the whole cell when it is cut away). */
    bool all_kept = true;
    bool all_cut_away = true;
    {
      BMIter iter;
      BMVert *vert;
      BM_ITER_MESH (vert, &iter, bm, BM_VERTS_OF_MESH) {
        const float3 position(vert->co);
        const float distance = math::dot(position - plane_co, plane_no);
        if (distance > 1.0e-6f) {
          all_kept = false;
        }
        if (distance < -1.0e-6f) {
          all_cut_away = false;
        }
        if (!all_kept && !all_cut_away) {
          break;
        }
      }
    }
    if (all_kept) {
      /* The whole fragment is already on the kept side. */
      continue;
    }
    if (all_cut_away) {
      BM_mesh_free(bm);
      return std::nullopt;
    }
    bisect_plane(bm, plane_co, plane_no);
    if (bm->totface == 0) {
      break;
    }
    /* Cap the cut right away: a single planar loop is much more reliable to fill than the combined
     * border after all cuts. */
    if (!shell_mode) {
      fill_holes(bm, original_offset);
    }
  }

  if (bm->totface == 0) {
    BM_mesh_free(bm);
    return std::nullopt;
  }

  if (!shell_mode) {
    fill_holes(bm, original_offset);
  }
  recalc_face_normals(bm);

  /* Bounds and center of the fragment. */
  float3 bounds_min(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 bounds_max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  BMIter iter;
  BMVert *vert;
  BM_ITER_MESH (vert, &iter, bm, BM_VERTS_OF_MESH) {
    bounds_min.x = std::min(bounds_min.x, vert->co[0]);
    bounds_min.y = std::min(bounds_min.y, vert->co[1]);
    bounds_min.z = std::min(bounds_min.z, vert->co[2]);
    bounds_max.x = std::max(bounds_max.x, vert->co[0]);
    bounds_max.y = std::max(bounds_max.y, vert->co[1]);
    bounds_max.z = std::max(bounds_max.z, vert->co[2]);
  }

  Fragment fragment;
  fragment.center = (bounds_min + bounds_max) * 0.5f;
  fragment.bounds_min = bounds_min;
  fragment.bounds_max = bounds_max;
  /* A piece is degenerate when it collapsed to a surface: one of its extents is zero, or it has no
   * volume at all. Such a piece can not be a rigid body (its convex hull is degenerate and it
   * squeezes through colliders), so it is reported and can be dropped. Computed on the BMesh, before
   * the vertices are moved. */
  const float3 extents = bounds_max - bounds_min;
  const float min_extent = std::min({extents.x, extents.y, extents.z});
  /* A shell piece has no volume at all, so the volume test would throw every piece away. */
  fragment.degenerate = !shell_mode && (min_extent < 1.0e-4f || bm_volume(bm) < 1.0e-6f);
  fragment.mesh_world = bm_to_mesh(bm);
  write_face_kind(*fragment.mesh_world, bm, original_offset);

  /* The instance mesh is the same fragment centered at the origin, so it is a copy of the world
   * mesh with the positions moved instead of a second BMesh conversion (which is a third of the cost
   * of a cell). The old path can still be selected for benchmarks. */
  static const bool old_local_copy = std::getenv("BLENDER_FRACTURE_OLD_LOCAL_COPY") != nullptr;
  if (old_local_copy) {
    BM_ITER_MESH (vert, &iter, bm, BM_VERTS_OF_MESH) {
      sub_v3_v3(vert->co, fragment.center);
    }
    fragment.mesh_local = bm_to_mesh(bm);
    write_face_kind(*fragment.mesh_local, bm, original_offset);
  }
  else {
    fragment.mesh_local = fragment_local_copy(*fragment.mesh_world, fragment.center);
  }
  BM_mesh_free(bm);
  return fragment;
}

/** Cut the source mesh with the Voronoi cell of `site_index`. Returns null for empty cells. */
static std::optional<Fragment> fracture_cell(const Mesh &source,
                                             const Span<float3> sites,
                                             const int3 res,
                                             const float3 cell_size,
                                             const float3 jitter,
                                             const SiteGrid *custom_grid,
                                             const bool shell_mode,
                                             const Map<uint64_t, int> &source_edge_of_verts,
                                             const int site_index)
{
  const float3 site = sites[site_index];

  /* Cut with every site that is close enough to reach this cell.
   *
   * The Voronoi cell of a site is bounded by the bisectors towards *its* neighbours, and a bisector
   * can only touch the cell when the other site is within `2 * radius` of it, where `radius` is an
   * upper bound of the cell's extent around its site. Using the grid neighbourhood for this is
   * tempting but wrong for elongated cells (the cell reaches much further along the long axis than
   * a fixed number of grid steps), and a bound based on the nearest neighbour is wrong too (the
   * cell is not inside a ball of `nearest / 2`). Iterating all sites is O(n^2) but the distance
   * test is trivial and the number of actual cuts stays small, so it is both exact and fast enough
   * for realistic site counts.
   *
   * Only for very large site counts the neighbourhood is used as a fallback, which is an
   * approximation (jitter should stay <= 0.5 there). */
  const float3 half_cell = cell_size * 0.5f;
  const float radius = math::length(half_cell) * (1.0f + math::reduce_max(jitter)) + 1.0e-4f;
  const float cut_distance = 2.0f * radius;
  Vector<SiteCandidate> candidates;
  if (custom_grid != nullptr) {
    custom_grid->gather_candidates(site_index, candidates);
  }
  else if (sites.size() <= 4096) {
    for (const int other_i : sites.index_range()) {
      if (other_i == site_index) {
        continue;
      }
      const float distance = math::distance(site, sites[other_i]);
      if (distance <= cut_distance) {
        candidates.append({other_i, distance});
      }
    }
  }
  else {
    const int site_x = site_index % res.x;
    const int site_y = (site_index / res.x) % res.y;
    const int site_z = site_index / (res.x * res.y);
    const int3 steps = int3(std::max(1, int(std::ceil(cut_distance / cell_size.x))),
                            std::max(1, int(std::ceil(cut_distance / cell_size.y))),
                            std::max(1, int(std::ceil(cut_distance / cell_size.z))));
    for (int dz = -steps.z; dz <= steps.z; ++dz) {
      for (int dy = -steps.y; dy <= steps.y; ++dy) {
        for (int dx = -steps.x; dx <= steps.x; ++dx) {
          const int nx = site_x + dx;
          const int ny = site_y + dy;
          const int nz = site_z + dz;
          if (nx < 0 || ny < 0 || nz < 0 || nx >= res.x || ny >= res.y || nz >= res.z) {
            continue;
          }
          const int other_i = (nz * res.y + ny) * res.x + nx;
          if (other_i == site_index) {
            continue;
          }
          const float distance = math::distance(site, sites[other_i]);
          if (distance <= cut_distance) {
            candidates.append({other_i, distance});
          }
        }
      }
    }
  }
  if (candidates.is_empty() && sites.size() > 1) {
    /* Every other site is too far away to touch this cell: it contains no geometry. With a single
     * site there is nothing to cut against and the cell *is* the whole mesh, so it is kept. */
    return std::nullopt;
  }
  /* Cut from the closest site outwards: the mesh shrinks quickly, so the later cuts are cheap. */
  std::sort(candidates.begin(),
            candidates.end(),
            [](const SiteCandidate &a, const SiteCandidate &b) {
              return a.distance < b.distance;
            });

  Vector<CellPlane> planes;
  planes.reserve(candidates.size());
  for (const SiteCandidate &candidate : candidates) {
    const float3 other = sites[candidate.index];
    /* Keep the half space that contains this site. */
    planes.append({(site + other) * 0.5f, math::normalize(other - site)});
  }

  /* A cell whose ball contains no triangle does not touch the surface, so it is either completely
   * inside the mesh (its piece is the convex cell, see `interior_cell`) or completely outside it.
   * This is the common case for a deep fracture and skips the whole mesh copy of the BMesh path. */
  /* Shell mode only keeps the surface, so a piece is the part of the surface inside the cell:
   * that is clipped directly on the mesh arrays (see `shell_cell`), with no BMesh and no capping.
   * A single site is the whole mesh, which the BMesh path handles. */
  static const bool no_shell_clip = std::getenv("BLENDER_FRACTURE_NO_SHELL_CLIP") != nullptr;
  if (shell_mode && !planes.is_empty() && !no_shell_clip) {
    return shell_cell(source, planes, site, radius, source_edge_of_verts);
  }

  static const bool no_interior_path = std::getenv("BLENDER_FRACTURE_NO_INTERIOR") != nullptr;
  if (!no_interior_path && !surface_near(source, site, radius)) {
    if (shell_mode || planes.is_empty()) {
      /* A shell has no interior, and a single site is the whole mesh (which touches the surface). */
      return std::nullopt;
    }
    const bke::BVHTreeFromMesh surface_bvh = source.bvh_corner_tris();
    if (!point_inside_mesh(surface_bvh, site)) {
      return std::nullopt;
    }
    return interior_cell(planes, radius);
  }

  /* Solid surface cell. Cutting a local patch of the surface instead of the whole mesh makes the
   * cost depend on the piece instead of the object. It is valid whenever the planes close the cell:
   * everything outside the cell, including the lid that closes the patch, is cut away again. */
  static const bool use_patch = std::getenv("BLENDER_FRACTURE_PATCH") != nullptr;
  if (use_patch && cell_is_bounded(planes, radius)) {
    if (std::optional<Fragment> patch = shell_cell(source, {}, site, radius, source_edge_of_verts)) {
      std::optional<Fragment> fragment = fracture_mesh_with_planes(
          *patch->mesh_world, planes, radius);
      BKE_id_free(nullptr, patch->mesh_world);
      BKE_id_free(nullptr, patch->mesh_local);
      return fragment;
    }
  }
  return fracture_mesh_with_planes(source, planes, radius);
}

/** Attach the given piece id to the face and point domain of a mesh. */
static void set_piece_id(Mesh &mesh, const int piece_id)
{
  bke::MutableAttributeAccessor attributes = mesh.attributes_for_write();
  bke::SpanAttributeWriter<int> face_ids = attributes.lookup_or_add_for_write_span<int>(
      "piece_id", AttrDomain::Face);
  std::fill(face_ids.span.begin(), face_ids.span.end(), piece_id);
  face_ids.finish();
  bke::SpanAttributeWriter<int> point_ids = attributes.lookup_or_add_for_write_span<int>(
      "piece_id", AttrDomain::Point);
  std::fill(point_ids.span.begin(), point_ids.span.end(), piece_id);
  point_ids.finish();
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry"_ustr);
  const Mesh *source_mesh = geometry_set.get_mesh();
  if (!source_mesh || source_mesh->verts_num == 0) {
    params.set_default_remaining_outputs();
    return;
  }

  const float3 cell_size = math::max(params.extract_input<float3>("Cell Size"_ustr),
                                     float3(1.0e-3f));
  const int seed = params.extract_input<int>("Seed"_ustr);
  const float3 jitter = math::clamp(params.extract_input<float3>("Jitter"_ustr),
                                    float3(0.0f),
                                    float3(1.0f));


  const Span<float3> positions = source_mesh->vert_positions();
  float3 bounds_min = positions.first();
  float3 bounds_max = positions.first();
  for (const float3 &p : positions) {
    bounds_min.x = std::min(bounds_min.x, p.x);
    bounds_min.y = std::min(bounds_min.y, p.y);
    bounds_min.z = std::min(bounds_min.z, p.z);
    bounds_max.x = std::max(bounds_max.x, p.x);
    bounds_max.y = std::max(bounds_max.y, p.y);
    bounds_max.z = std::max(bounds_max.z, p.z);
  }
  /* Avoid a degenerate extent for flat meshes. */
  const float3 diagonal = bounds_max - bounds_min;
  const float3 extent(std::max(diagonal.x, 1.0e-4f),
                      std::max(diagonal.y, 1.0e-4f),
                      std::max(diagonal.z, 1.0e-4f));
  bounds_min -= extent * 0.001f;
  bounds_max += extent * 0.001f;

  /* Cell centers: either the jittered grid, or the points the user supplied. In the Custom mode the
   * points are filtered to the ones inside (or on) the mesh, so painting more points somewhere is
   * what makes the fragments smaller there. */
  const int mode = params.extract_input<int>("Mode"_ustr);
  const bool shell_mode = params.extract_input<bool>("Shell Mode"_ustr);
  const float shell_thickness = std::max(params.extract_input<float>("Shell Thickness"_ustr), 0.0f);
  int3 res = int3(1);
  Vector<float3> sites;
  SiteGrid custom_grid;
  if (mode == int(FractureSiteMode::Custom)) {
    const GeometrySet points = params.extract_input<GeometrySet>("Points"_ustr);
    sites = custom_sites_from_points(*source_mesh, points);
    if (sites.is_empty()) {
      /* Custom mode means "the cell centers come from Points". Without a usable point there is
       * nothing to fracture, and silently falling back to the grid would break that contract
       * (and look like the Points input is ignored). */
      params.error_message_add(
          NodeWarningType::Warning,
          "Custom mode needs points inside (or on) the geometry to place the cell centers, "
          "nothing was fractured. Connect a point cloud to the Points input, or switch the Mode "
          "back to Default");
      params.set_default_remaining_outputs();
      return;
    }
    custom_grid = build_site_grid(sites);
  }
  else {
    sites = generate_sites(bounds_min, bounds_max, cell_size, seed, jitter, res);
  }
  const SiteGrid *site_grid = custom_grid.sites.is_empty() ? nullptr : &custom_grid;

  /* Build the surface BVH once, before the cells are computed in parallel: it is used to find the
   * cells that do not touch the surface and to tell inside from outside (a lazily built cache would
   * race between the threads). */
  source_mesh->bvh_corner_tris();

  /* Vertex pair -> edge index, so a piece can keep the edge attributes (UV seams, sharp edges) of the
   * source edges it still has. Read only in the cells below. */
  Map<uint64_t, int> source_edge_of_verts;
  {
    const Span<int2> source_edges = source_mesh->edges();
    source_edge_of_verts.reserve(source_edges.size());
    for (const int i : source_edges.index_range()) {
      source_edge_of_verts.add(shell_pair_key(source_edges[i][0], source_edges[i][1]), i);
    }
  }

  /* Cells are independent, so they are computed in parallel. Each cell has its own BMesh; the
   * source mesh is only read. The piece ids are assigned afterwards, in site order, so the result
   * does not depend on the thread scheduling. */
  Array<std::optional<Fragment>> cell_fragments(sites.size());
  threading::parallel_for(sites.index_range(), 16, [&](const IndexRange range) {
    for (const int site_index : range) {
      cell_fragments[site_index] = fracture_cell(*source_mesh,
                                                 sites,
                                                 res,
                                                 cell_size,
                                                 jitter,
                                                 site_grid,
                                                 shell_mode,
                                                 source_edge_of_verts,
                                                 site_index);
    }
  });

  Vector<bke::GeometrySet> joined_parts;
  joined_parts.reserve(sites.size());
  Vector<Fragment> fragments;
  fragments.reserve(sites.size());

  int piece_id = 0;
  int degenerate_num = 0;
  const bool delete_degenerate = params.extract_input<bool>("Delete Degenerate Pieces"_ustr);
  for (const int site_index : sites.index_range()) {
    std::optional<Fragment> &fragment = cell_fragments[site_index];
    if (!fragment) {
      continue;
    }
    if (fragment->degenerate) {
      degenerate_num++;
      if (delete_degenerate) {
        continue;
      }
    }
    set_piece_id(*fragment->mesh_world, piece_id);
    set_piece_id(*fragment->mesh_local, piece_id);

    bke::GeometrySet world_set;
    world_set.replace_mesh(fragment->mesh_world);
    joined_parts.append(std::move(world_set));

    fragments.append(std::move(*fragment));
    piece_id++;
  }

  if (degenerate_num > 0) {
    std::string message = "Voronoi Fracture: " + std::to_string(degenerate_num) +
                          " degenerate fragment(s) generated (no volume). ";
    if (delete_degenerate) {
      message += "They were deleted; increase Cell Size or reduce Jitter to avoid them.";
    }
    else {
      message += "Enable 'Delete Degenerate Pieces' to remove them.";
    }
    params.error_message_add(NodeWarningType::Warning, message);
  }


  if (fragments.is_empty()) {
    params.set_default_remaining_outputs();
    return;
  }

  /* Fragments as instances, so the rigid body solver can use them directly. */
  const int fragment_num = int(fragments.size());
  bke::Instances *instances = new bke::Instances(fragment_num);
  MutableSpan<float4x4> transforms = instances->transforms_for_write();
  MutableSpan<int> handles = instances->reference_handles_for_write();
  for (const int i : IndexRange(fragment_num)) {
    bke::GeometrySet local_set;
    local_set.replace_mesh(fragments[i].mesh_local);
    handles[i] = instances->add_reference(bke::InstanceReference(std::move(local_set)));
    transforms[i] = math::from_location<float4x4>(fragments[i].center);
  }
  {
    bke::MutableAttributeAccessor attributes = instances->attributes_for_write();
    bke::SpanAttributeWriter<int> piece_ids = attributes.lookup_or_add_for_write_span<int>(
        "piece_id", AttrDomain::Instance);
    for (const int i : IndexRange(fragment_num)) {
      piece_ids.span[i] = i;
    }
    piece_ids.finish();
    if (shell_mode) {
      /* Init Rigid Body reads this to compute the mass from the surface area instead of the
       * (nonexistent) volume. */
      bke::SpanAttributeWriter<float> thicknesses =
          attributes.lookup_or_add_for_write_span<float>("shell_thickness", AttrDomain::Instance);
      for (const int i : IndexRange(fragment_num)) {
        thicknesses.span[i] = shell_thickness;
      }
      thicknesses.finish();
    }
  }
  bke::GeometrySet fragments_set;
  fragments_set.replace_instances(instances);

  bke::GeometrySet joined = geometry::join_geometries(
      joined_parts, bke::AttributeFilter::default_filter());

  params.set_output("Geometry"_ustr, std::move(joined));
  params.set_output("Pieces"_ustr, std::move(fragments_set));
}

static void node_label(const bNodeTree * /*ntree*/,
                       const bNode * /*node*/,
                       char *label,
                       const int label_maxncpy)
{
  BLI_strncpy_utf8(label, CTX_IFACE_(BLT_I18NCONTEXT_ID_NODETREE, "Voronoi Fracture"), label_maxncpy);
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeVoronoiFracture"_ustr);
  ntype.ui_name = "Voronoi Fracture";
  ntype.ui_description = "Split a mesh into Voronoi cells with stable piece ids";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  ntype.labelfunc = node_label;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_voronoi_fracture_cc

