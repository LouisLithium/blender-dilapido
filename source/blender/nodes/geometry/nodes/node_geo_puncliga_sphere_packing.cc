/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo_nodes
 *
 * Fills a closed mesh with spheres of varying size. The points are meant for the `Rigid` mode of
 * `Init Puncliga Data`: the spheres are only an approximation of the volume, so they are allowed to
 * overlap, which lets a small number of them cover thin parts of the shape as well.
 *
 * A regular grid of candidate centers is placed inside the mesh. The signed distance field decides
 * which candidates are inside; the distance to the closest surface gives the radius of the largest
 * sphere that fits at that point. The candidates are then sorted by radius and accepted greedily:
 * a candidate is kept when its sphere is not buried inside an already placed one. Large spheres
 * therefore cover the thick parts first and the small ones fill the gaps near the surface.
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include "BKE_attribute.hh"
#include "BKE_bvhutils.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"
#include "BKE_volume_grid.hh"

#include "BLI_array.hh"
#include "BLI_bounds.hh"
#include "BLI_kdopbvh.hh"
#include "BLI_map.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_string_utf8.h"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "BLT_translation.hh"

#include "GEO_mesh_to_volume.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_puncliga_sphere_packing_cc {

struct SpherePackingSettings {
  float spacing = 0.05f;
  float min_radius = 0.01f;
  float max_radius = 0.2f;
  float radius_scale = 1.0f;
  float surface_offset = 0.0f;
  float overlap = 0.3f;
  float jitter = 0.25f;
  int max_spheres = 5000;
  int seed = 0;
};

struct SpherePackingResult {
  Vector<float3> positions;
  Vector<float> radii;
  int candidates = 0;
  int accepted = 0;
  int capped = 0;
};

/** Trilinear sample of the distance field at a point in the space of the input mesh. */
static float sample_distance(const openvdb::FloatGrid &grid,
                             const openvdb::FloatGrid::ConstAccessor &accessor,
                             const float3 &point)
{
  const openvdb::Vec3d index_position = grid.transform().worldToIndex(
      openvdb::Vec3d(point.x, point.y, point.z));
  const int i = int(std::floor(index_position.x()));
  const int j = int(std::floor(index_position.y()));
  const int k = int(std::floor(index_position.z()));
  const float tx = float(index_position.x() - double(i));
  const float ty = float(index_position.y() - double(j));
  const float tz = float(index_position.z() - double(k));

  const float v000 = accessor.getValue(openvdb::Coord(i, j, k));
  const float v100 = accessor.getValue(openvdb::Coord(i + 1, j, k));
  const float v010 = accessor.getValue(openvdb::Coord(i, j + 1, k));
  const float v110 = accessor.getValue(openvdb::Coord(i + 1, j + 1, k));
  const float v001 = accessor.getValue(openvdb::Coord(i, j, k + 1));
  const float v101 = accessor.getValue(openvdb::Coord(i + 1, j, k + 1));
  const float v011 = accessor.getValue(openvdb::Coord(i, j + 1, k + 1));
  const float v111 = accessor.getValue(openvdb::Coord(i + 1, j + 1, k + 1));

  const float v00 = v000 + (v100 - v000) * tx;
  const float v10 = v010 + (v110 - v010) * tx;
  const float v01 = v001 + (v101 - v001) * tx;
  const float v11 = v011 + (v111 - v011) * tx;
  const float v0 = v00 + (v10 - v00) * ty;
  const float v1 = v01 + (v11 - v01) * ty;
  return v0 + (v1 - v0) * tz;
}

/** A cheap deterministic hash, so the jitter does not depend on the thread or the platform. */
static uint32_t hash_u32(uint32_t x)
{
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

static float hash_to_float(const uint32_t hash)
{
  return float(hash & 0xffffffu) / float(0x1000000u);
}

/** A conservative cell key; collisions only add a few distance checks, they never accept a wrong
 * sphere. */
static int64_t packing_cell_key(const int3 &cell)
{
  return (int64_t(cell.x) * 73856093) ^ (int64_t(cell.y) * 19349663) ^
         (int64_t(cell.z) * 83492791);
}

struct PackingCandidate {
  float3 position;
  float radius;
  int index;
};

static bool sphere_pack_mesh(const Mesh &mesh,
                             const SpherePackingSettings &settings,
                             SpherePackingResult &r_result,
                             std::string &r_error,
                             std::string &r_warning)
{
  const Span<float3> mesh_positions = mesh.vert_positions();
  const Span<int3> mesh_corner_tris = mesh.corner_tris();
  if (mesh_positions.is_empty() || mesh_corner_tris.is_empty()) {
    r_error = "The mesh has no faces to pack";
    return false;
  }

  std::optional<Bounds<float3>> bounds = mesh.bounds_min_max();
  if (!bounds) {
    r_error = "The mesh has no bounds";
    return false;
  }
  const float3 bounds_min = bounds->min;
  const float3 bounds_max = bounds->max;
  const float3 extent = bounds_max - bounds_min;

  const float min_radius = std::max(settings.min_radius, 1e-4f);
  const float max_radius = std::max(settings.max_radius, min_radius * 1.001f);
  const float radius_scale = std::max(settings.radius_scale, 0.0f);
  const float surface_offset = settings.surface_offset;
  const float overlap = math::clamp(settings.overlap, 0.0f, 0.95f);
  const float jitter = math::clamp(settings.jitter, 0.0f, 1.0f);
  const int max_spheres = std::max(settings.max_spheres, 1);
  const uint32_t seed = uint32_t(settings.seed);

  float spacing = std::max(settings.spacing, 1e-4f);
  int3 cells;
  int64_t candidate_count;
  constexpr int64_t max_candidates = 2000000;
  const auto count_cells = [&]() {
    cells = int3(std::max(1, int(std::ceil(extent.x / spacing))),
                 std::max(1, int(std::ceil(extent.y / spacing))),
                 std::max(1, int(std::ceil(extent.z / spacing))));
    candidate_count = int64_t(cells.x) * int64_t(cells.y) * int64_t(cells.z);
  };
  count_cells();
  if (candidate_count > max_candidates) {
    /* Keep a huge input from allocating a candidate array that it can never use. */
    spacing *= std::cbrt(float(candidate_count) / float(max_candidates));
    count_cells();
    r_warning = "The packing spacing was increased to keep the candidate grid below " +
                std::to_string(max_candidates) + " points";
  }

  /* The distance field is only used for the inside/outside test, so a coarse voxel is enough. */
  const float sdf_voxel = std::max(spacing * 0.5f, 1e-3f);
  const float sdf_band = 3.0f;
  const bke::VolumeGrid<float> sdf = [&]() {
    Vector<float3> positions(mesh_positions.begin(), mesh_positions.end());
    const Span<int> corner_verts = mesh.corner_verts();
    Vector<int> verts(corner_verts.begin(), corner_verts.end());
    Vector<int3> tris(mesh_corner_tris.begin(), mesh_corner_tris.end());
    return geometry::mesh_to_sdf_grid(positions, verts, tris, sdf_voxel, sdf_band);
  }();
  if (!sdf) {
    r_error = "The signed distance field of the input mesh could not be built";
    return false;
  }

  bke::BVHTreeFromMesh bvh = mesh.bvh_corner_tris();
  if (bvh.tree == nullptr) {
    r_error = "The input mesh has no triangles";
    return false;
  }

  Array<float3> candidate_positions(candidate_count);
  Array<float> candidate_radii(candidate_count);
  Array<uint8_t> candidate_valid(candidate_count, 0);

  bke::VolumeTreeAccessToken sdf_token;
  const openvdb::FloatGrid &sdf_grid = sdf.grid(sdf_token);
  threading::parallel_for(IndexRange(cells.z), 1, [&](const IndexRange k_range) {
    const openvdb::FloatGrid::ConstAccessor accessor = sdf_grid.getConstAccessor();
    for (const int k : k_range) {
      for (const int j : IndexRange(cells.y)) {
        for (const int i : IndexRange(cells.x)) {
          const int64_t index = (int64_t(k) * cells.y + j) * cells.x + i;
          const uint32_t h0 = hash_u32(uint32_t(index) ^ seed);
          const uint32_t h1 = hash_u32(uint32_t(index) * 3u + seed + 1u);
          const uint32_t h2 = hash_u32(uint32_t(index) * 5u + seed + 2u);
          float3 position = bounds_min + float3(float(i) + 0.5f, float(j) + 0.5f,
                                                float(k) + 0.5f) *
                                              spacing;
          if (jitter > 0.0f) {
            position += float3(hash_to_float(h0) - 0.5f,
                               hash_to_float(h1) - 0.5f,
                               hash_to_float(h2) - 0.5f) *
                        (spacing * jitter);
          }

          /* A candidate on or outside the surface can not carry a sphere. The level set has a
           * negative value everywhere inside, including far away from the narrow band. */
          if (sample_distance(sdf_grid, accessor, position) >= -1e-5f) {
            continue;
          }

          BVHTreeNearest nearest;
          nearest.index = -1;
          nearest.dist_sq = FLT_MAX;
          const float co[3] = {position.x, position.y, position.z};
          if (BLI_bvhtree_find_nearest(
                  bvh.tree, co, &nearest, bvh.nearest_callback, &bvh) < 0)
          {
            continue;
          }
          const float surface_distance = math::sqrt(nearest.dist_sq);
          const float radius = math::clamp((surface_distance + surface_offset) * radius_scale,
                                           min_radius,
                                           max_radius);
          if (!(radius >= min_radius)) {
            continue;
          }
          candidate_positions[index] = position;
          candidate_radii[index] = radius;
          candidate_valid[index] = 1;
        }
      }
    }
  });

  Vector<PackingCandidate> candidates;
  for (const int64_t index : IndexRange(candidate_count)) {
    if (!candidate_valid[index]) {
      continue;
    }
    candidates.append({candidate_positions[index], candidate_radii[index], int(index)});
  }
  r_result.candidates = int(candidates.size());
  if (candidates.is_empty()) {
    r_warning = "No candidate center fit inside the mesh; decrease the minimum radius or the "
                "surface offset";
    return true;
  }

  /* Large spheres first: they cover the thick parts, the small ones fill the rest. The index breaks
   * ties so the result does not depend on the sort implementation. */
  std::sort(candidates.begin(), candidates.end(), [](const PackingCandidate &a,
                                                      const PackingCandidate &b) {
    if (a.radius != b.radius) {
      return a.radius > b.radius;
    }
    return a.index < b.index;
  });

  const float cell_size = std::max(2.0f * max_radius, spacing);
  Map<int64_t, Vector<int>> kept_by_cell;
  const auto cell_of = [&](const float3 &position) {
    return int3(int(std::floor(position.x / cell_size)),
                int(std::floor(position.y / cell_size)),
                int(std::floor(position.z / cell_size)));
  };

  for (const PackingCandidate &candidate : candidates) {
    if (int(r_result.positions.size()) >= max_spheres) {
      r_result.capped = 1;
      break;
    }
    const int3 cell = cell_of(candidate.position);
    bool covered = false;
    for (int dz = -1; dz <= 1 && !covered; dz++) {
      for (int dy = -1; dy <= 1 && !covered; dy++) {
        for (int dx = -1; dx <= 1 && !covered; dx++) {
          const int64_t key = packing_cell_key(cell + int3(dx, dy, dz));
          const Vector<int> *bucket = kept_by_cell.lookup_ptr(key);
          if (bucket == nullptr) {
            continue;
          }
          for (const int kept_i : *bucket) {
            const float contact = (r_result.radii[kept_i] + candidate.radius) * (1.0f - overlap);
            if (math::distance_squared(r_result.positions[kept_i], candidate.position) <
                contact * contact)
            {
              covered = true;
              break;
            }
          }
        }
      }
    }
    if (covered) {
      continue;
    }
    const int kept_i = int(r_result.positions.size());
    r_result.positions.append(candidate.position);
    r_result.radii.append(candidate.radius);
    kept_by_cell.lookup_or_add_default(packing_cell_key(cell)).append(kept_i);
  }
  r_result.accepted = int(r_result.positions.size());
  return true;
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Mesh"_ustr)
      .description("A closed mesh whose volume is filled with spheres");
  {
    auto &panel = b.add_panel("Packing"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Spacing"_ustr)
        .default_value(0.05f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description(
            "Distance between candidate sphere centers. Smaller values fill the shape more "
            "accurately and produce more spheres");
    panel.add_input<decl::Float>("Min Radius"_ustr)
        .default_value(0.01f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Candidates closer to the surface than this are not used");
    panel.add_input<decl::Float>("Max Radius"_ustr)
        .default_value(0.2f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Largest sphere radius; thick parts are covered by several spheres instead");
    panel.add_input<decl::Float>("Radius Scale"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .max(2.0f)
        .description(
            "Multiplier for the distance to the surface. Values above 1 make the spheres protrude "
            "a little, which closes the surface better");
    panel.add_input<decl::Float>("Surface Offset"_ustr)
        .default_value(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Added to the surface distance before it becomes a radius");
    panel.add_input<decl::Float>("Overlap"_ustr)
        .default_value(0.3f)
        .min(0.0f)
        .max(1.0f)
        .description(
            "How much two spheres may overlap. 0 places them only up to contact, larger values let "
            "a sphere sink into an already placed one and cover the gaps");
    panel.add_input<decl::Float>("Jitter"_ustr)
        .default_value(0.25f)
        .min(0.0f)
        .max(1.0f)
        .description("Random offset of the candidate grid inside one cell");
    panel.add_input<decl::Int>("Seed"_ustr).default_value(0).description("Jitter seed");
  }
  {
    auto &panel = b.add_panel("Limits"_ustr).default_closed(false);
    panel.add_input<decl::Int>("Max Spheres"_ustr)
        .default_value(5000)
        .min(1)
        .max(1000000)
        .description("Stop after this many spheres have been placed");
  }
  b.add_output<decl::Geometry>("Points"_ustr).propagate_all();
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Mesh"_ustr);
  const Mesh *mesh = geometry.get_mesh();
  if (mesh == nullptr) {
    params.error_message_add(NodeWarningType::Error,
                             TIP_("Sphere packing needs a mesh input"));
    return;
  }
  if (mesh->faces_num == 0) {
    params.error_message_add(NodeWarningType::Error,
                             TIP_("Sphere packing needs a mesh with faces"));
    return;
  }

  SpherePackingSettings settings;
  settings.spacing = params.get_input<float>("Spacing"_ustr);
  settings.min_radius = params.get_input<float>("Min Radius"_ustr);
  settings.max_radius = params.get_input<float>("Max Radius"_ustr);
  settings.radius_scale = params.get_input<float>("Radius Scale"_ustr);
  settings.surface_offset = params.get_input<float>("Surface Offset"_ustr);
  settings.overlap = params.get_input<float>("Overlap"_ustr);
  settings.jitter = params.get_input<float>("Jitter"_ustr);
  settings.max_spheres = params.get_input<int>("Max Spheres"_ustr);
  settings.seed = params.get_input<int>("Seed"_ustr);

  SpherePackingResult result;
  std::string error;
  std::string warning;
  if (!sphere_pack_mesh(*mesh, settings, result, error, warning)) {
    params.error_message_add(NodeWarningType::Error, StringRef(error));
    params.set_output("Points"_ustr, GeometrySet());
    return;
  }
  if (!warning.empty()) {
    params.error_message_add(NodeWarningType::Warning, StringRef(warning));
  }
  if (result.capped) {
    params.error_message_add(
        NodeWarningType::Warning,
        TIP_("Sphere packing reached `Max Spheres`; the shape is only partially covered"));
  }

  const int spheres_num = int(result.positions.size());
  PointCloud *pointcloud = BKE_pointcloud_new_nomain(spheres_num);
  pointcloud->positions_for_write().copy_from(result.positions.as_span());
  bke::MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
  bke::SpanAttributeWriter<float> radius_writer =
      attributes.lookup_or_add_for_write_only_span<float>("radius", bke::AttrDomain::Point);
  bke::SpanAttributeWriter<int> id_writer =
      attributes.lookup_or_add_for_write_only_span<int>("id", bke::AttrDomain::Point);
  if (radius_writer && id_writer) {
    radius_writer.span.copy_from(result.radii.as_span());
    for (const int i : IndexRange(spheres_num)) {
      id_writer.span[i] = i;
    }
    radius_writer.finish();
    id_writer.finish();
  }

  if (std::getenv("BLENDER_PUNCLIGA_PACKING_DEBUG") != nullptr) {
    printf("PUNCLIGA_PACKING candidates %d accepted %d (%s)\n",
           result.candidates,
           result.accepted,
           result.capped ? "capped" : "ok");
  }

  GeometrySet output;
  output.replace_pointcloud(pointcloud);
  params.set_output("Points"_ustr, std::move(output));
}

static void node_label(const bNodeTree * /*ntree*/,
                       const bNode * /*node*/,
                       char *label,
                       const int label_maxncpy)
{
  BLI_strncpy_utf8(
      label, CTX_IFACE_(BLT_I18NCONTEXT_ID_NODETREE, "Puncliga Sphere Packing"), label_maxncpy);
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodePuncligaSpherePacking"_ustr);
  ntype.ui_name = "Puncliga Sphere Packing";
  ntype.ui_description =
      "Fill a closed mesh with spheres of varying size, for the Rigid mode of `Init Puncliga Data`";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_220;
  ntype.labelfunc = node_label;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_puncliga_sphere_packing_cc
