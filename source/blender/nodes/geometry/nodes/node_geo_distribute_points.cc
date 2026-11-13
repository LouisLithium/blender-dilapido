/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Distribute Points`: a point set generator for the MPM solvers (and general use).
 *
 * Two modes:
 *
 * - `Surface`: random points on the faces with a density derived from `Spacing`, optionally
 *   relaxed (repulsion + reprojection) so the points are evenly spread instead of clumped.
 * - `Volume`: a lattice inside a closed mesh. `Grid` is an axis aligned lattice, `Hexagonal` is a
 *   close packed (HCP) lattice, i.e. triangular layers stacked with the centroid offset and
 *   `spacing * sqrt(2/3)` between layers.
 *
 * The optional jitter breaks the regularity without letting points overlap or leave the mesh:
 * each axis moves at most `0.15 * jitter * spacing`, so two lattice neighbours can approach by at
 * most `0.52 * spacing`, and a jittered point that ends up outside the mesh is discarded (the
 * lattice point it came from is kept). Everything is deterministic: no global RNG, per point
 * hashes seeded by `Seed`.
 */

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

#include "BLI_array.hh"
#include "BLI_kdtree.hh"
#include "BLI_math_vector.hh"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_bvhutils.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_distribute_points_cc {

enum class Mode : int { Surface = 0, Volume = 1 };
enum class LatticeType : int { Grid = 0, Hexagonal = 1 };
enum class JitterLayer : int { All = 0, Surface = 1 };

static const EnumPropertyItem mode_items[] = {
    {int(Mode::Surface),
     "SURFACE",
     0,
     N_("Surface"),
     N_("Scatter points over the faces of the mesh, optionally relaxed so they do not clump")},
    {int(Mode::Volume),
     "VOLUME",
     0,
     N_("Volume"),
     N_("Fill the closed mesh with a lattice of points")},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem lattice_items[] = {
    {int(LatticeType::Grid),
     "GRID",
     0,
     N_("Grid"),
     N_("Axis aligned lattice: nearest neighbours are the six axis neighbours at `Spacing`")},
    {int(LatticeType::Hexagonal),
     "HEXAGONAL",
     0,
     N_("Hexagonal"),
     N_("Close packed lattice (HCP): twelve nearest neighbours at `Spacing`, denser than the grid "
        "for the same spacing")},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem jitter_layer_items[] = {
    {int(JitterLayer::All),
     "ALL",
     0,
     N_("All"),
     N_("Jitter every point of the volume lattice")},
    {int(JitterLayer::Surface),
     "SURFACE",
     0,
     N_("Surface Only"),
     N_("Only jitter the points within 1.5 spacings of the mesh surface, so the visible shell loses "
        "its regularity while the interior stays on the lattice")},
    {0, nullptr, 0, nullptr, nullptr},
};

/** Upper bounds so a typo cannot allocate gigabytes. */
constexpr int64_t MAX_POINTS = 8 << 20;

/**
 * Small deterministic PCG. The node must not use a global RNG: every point derives its random
 * numbers from its own index and `Seed`, so the result is identical with any thread count.
 */
struct Rng {
  uint32_t state;

  explicit Rng(const uint32_t seed)
  {
    state = seed * 747796405u + 2891336453u;
    next();
  }

  uint32_t next()
  {
    state = state * 747796405u + 2891336453u;
    const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
  }

  float next_float()
  {
    return float(next() >> 8) * (1.0f / 16777216.0f);
  }
};

static uint32_t point_seed(const uint32_t seed, const int64_t index, const uint32_t salt)
{
  uint32_t hash = uint32_t(index) * 0x9E3779B1u;
  hash ^= uint32_t(index >> 32) * 0x85EBCA77u;
  hash ^= seed * 0xC2B2AE3Du;
  hash ^= salt * 0x27D4EB2Fu;
  hash ^= hash >> 15;
  hash *= 0x2545F491u;
  hash ^= hash >> 13;
  return hash;
}

/* -------------------------------------------------------------------- */
/** \name Mesh queries
 * \{ */

/** Nearest point on the mesh; `nullptr` callbacks would leave `nearest.co` untouched. */
static bool mesh_nearest(const bke::BVHTreeFromMesh &bvh,
                         const float3 &position,
                         BVHTreeNearest &r_nearest)
{
  if (bvh.tree == nullptr) {
    return false;
  }
  r_nearest.index = -1;
  r_nearest.dist_sq = FLT_MAX;
  const float co[3] = {position.x, position.y, position.z};
  bke::BVHTreeFromMesh &bvh_mutable = const_cast<bke::BVHTreeFromMesh &>(bvh);
  return BLI_bvhtree_find_nearest(
             bvh.tree, co, &r_nearest, bvh.nearest_callback, &bvh_mutable) >= 0;
}

/**
 * Inside test from the nearest triangle: the mesh is treated as a shell and the sign of the
 * distance along the triangle normal decides. `r_distance` receives the unsigned distance.
 */
static bool mesh_contains(const bke::BVHTreeFromMesh &bvh,
                          const float3 &position,
                          const float orientation,
                          float *r_distance)
{
  BVHTreeNearest nearest;
  if (!mesh_nearest(bvh, position, nearest)) {
    return false;
  }
  if (r_distance != nullptr) {
    *r_distance = std::sqrt(nearest.dist_sq);
  }
  const float3 delta = position - float3(nearest.co);
  if (math::length_squared(delta) <= 1.0e-8f) {
    return true;
  }
  const int3 tri = bvh.corner_tris[nearest.index];
  const float3 a = bvh.vert_positions[bvh.corner_verts[tri[0]]];
  const float3 b = bvh.vert_positions[bvh.corner_verts[tri[1]]];
  const float3 c = bvh.vert_positions[bvh.corner_verts[tri[2]]];
  return orientation * math::dot(delta, math::cross(b - a, c - a)) < 0.0f;
}

/** True when every edge is used by exactly two face corners. */
static bool mesh_is_closed(const Mesh &mesh)
{
  const int edges_num = mesh.edges_num;
  if (edges_num == 0) {
    return false;
  }
  Array<int> uses(edges_num, 0);
  for (const int edge : mesh.corner_edges()) {
    uses[edge]++;
  }
  for (const int use : uses) {
    if (use != 2) {
      return false;
    }
  }
  return true;
}

/**
 * Signed volume of a closed triangle mesh. It is positive when the normals point outwards; a
 * negative value means the winding is reversed, so the inside test has to be flipped. This is what
 * makes Volume mode work on meshes that were mirrored or generated with inward faces.
 */
static float mesh_signed_volume(const Mesh &mesh)
{
  const Span<float3> verts = mesh.vert_positions();
  const Span<int> corner_verts = mesh.corner_verts();
  float volume = 0.0f;
  for (const int3 &tri : mesh.corner_tris()) {
    const float3 &a = verts[corner_verts[tri[0]]];
    const float3 &b = verts[corner_verts[tri[1]]];
    const float3 &c = verts[corner_verts[tri[2]]];
    volume += math::dot(a, math::cross(b, c));
  }
  return volume / 6.0f;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Surface distribution
 * \{ */

static void distribute_on_surface(const Mesh &mesh,
                                  const float spacing,
                                  const uint32_t seed,
                                  Vector<float3> &r_positions)
{
  const Span<float3> verts = mesh.vert_positions();
  const Span<int> corner_verts = mesh.corner_verts();
  const Span<int3> corner_tris = mesh.corner_tris();
  const float spacing_sq = std::max(spacing * spacing, 1.0e-10f);

  for (const int tri_index : corner_tris.index_range()) {
    const int3 tri = corner_tris[tri_index];
    const float3 &a = verts[corner_verts[tri[0]]];
    const float3 &b = verts[corner_verts[tri[1]]];
    const float3 &c = verts[corner_verts[tri[2]]];
    const float area = 0.5f * math::length(math::cross(b - a, c - a));
    const float expected = area / spacing_sq;
    Rng rng(point_seed(seed, tri_index, 1u));
    int count = int(std::floor(expected));
    if (rng.next_float() < expected - float(count)) {
      count++;
    }
    for (int i = 0; i < count; i++) {
      /* Uniform barycentric coordinates (the square root keeps the density uniform). */
      const float r1 = rng.next_float();
      const float r2 = rng.next_float();
      const float su = std::sqrt(r1);
      r_positions.append(a * (1.0f - su) + b * (su * (1.0f - r2)) + c * (su * r2));
    }
    if (r_positions.size() > MAX_POINTS) {
      return;
    }
  }
}

static void relax_surface_points(const bke::BVHTreeFromMesh &bvh,
                                 const MutableSpan<float3> positions,
                                 const float spacing,
                                 const int iterations,
                                 const float strength)
{
  if (iterations <= 0 || positions.size() < 2) {
    return;
  }
  const float radius = std::max(spacing, 1.0e-6f);
  Vector<float3> next(positions.size());
  for (int iteration = 0; iteration < iterations; iteration++) {
    KDTree<float3> *tree = kdtree_new<float3>(uint(positions.size()));
    for (const int i : positions.index_range()) {
      kdtree_insert(tree, i, positions[i]);
    }
    kdtree_balance(tree);

    threading::parallel_for(positions.index_range(), 1024, [&](const IndexRange range) {
      for (const int i : range) {
        const float3 p = positions[i];
        float3 push(0.0f);
        kdtree_range_search_cb(
            tree, p, radius, [&](const int other, const float3 &co, const float dist_sq) {
              if (other != i) {
                const float distance = std::sqrt(dist_sq);
                if (distance > 1.0e-6f) {
                  push += (p - co) * ((1.0f - distance / radius) / distance);
                }
              }
              return true;
            });
        float3 target = p + push * (spacing * strength);
        /* Move at most half a spacing per iteration so a point cannot jump to another part of the
         * surface through the projection. */
        const float move = math::length(target - p);
        if (move > 0.5f * spacing) {
          target = p + (target - p) * (0.5f * spacing / move);
        }
        BVHTreeNearest nearest;
        if (mesh_nearest(bvh, target, nearest)) {
          target = float3(nearest.co);
        }
        next[i] = target;
      }
    });

    positions.copy_from(next.as_span());
    kdtree_free(tree);
  }
}

static void jitter_surface_points(const bke::BVHTreeFromMesh &bvh,
                                  const MutableSpan<float3> positions,
                                  const float spacing,
                                  const float jitter,
                                  const uint32_t seed)
{
  const float axis = 0.15f * math::clamp(jitter, 0.0f, 1.0f) * spacing;
  if (axis <= 0.0f) {
    return;
  }
  threading::parallel_for(positions.index_range(), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      Rng rng(point_seed(seed, i, 2u));
      const float3 offset((rng.next_float() - 0.5f) * 2.0f * axis,
                          (rng.next_float() - 0.5f) * 2.0f * axis,
                          (rng.next_float() - 0.5f) * 2.0f * axis);
      BVHTreeNearest nearest;
      if (mesh_nearest(bvh, positions[i] + offset, nearest)) {
        /* Projecting keeps the point on the surface, so it can never leave the mesh. */
        positions[i] = float3(nearest.co);
      }
    }
  });
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Volume distribution
 * \{ */

static bool distribute_in_volume(const Mesh &mesh,
                                 const LatticeType lattice,
                                 const float spacing,
                                 const float orientation,
                                 Vector<float3> &r_positions,
                                 Vector<float> &r_surface_distance)
{
  const bke::BVHTreeFromMesh bvh = mesh.bvh_corner_tris();
  float3 min(std::numeric_limits<float>::max());
  float3 max(-std::numeric_limits<float>::max());
  for (const float3 &position : mesh.vert_positions()) {
    min = math::min(min, position);
    max = math::max(max, position);
  }
  if (!(max.x > min.x) || !(max.y > min.y) || !(max.z > min.z)) {
    return true;
  }
  const float3 extent = max - min;
  const float half = 0.5f * spacing;

  /* Lattice steps. `Hexagonal` stacks triangular layers: `dy = spacing * sqrt(3)/2` inside a layer
   * and `dz = spacing * sqrt(2/3)` between layers. */
  const bool hexagonal = lattice == LatticeType::Hexagonal;
  const float dy = hexagonal ? spacing * std::sqrt(3.0f) * 0.5f : spacing;
  const float dz = hexagonal ? spacing * std::sqrt(2.0f / 3.0f) : spacing;
  const int steps_x = int(std::floor(extent.x / spacing)) + 1;
  const int steps_y = int(std::floor(extent.y / dy)) + 1;
  const int steps_z = int(std::floor(extent.z / dz)) + 1;
  const int64_t total = int64_t(steps_x) * steps_y * steps_z;
  if (total > MAX_POINTS) {
    return false;
  }
  r_positions.reserve(int(total));
  r_surface_distance.reserve(int(total));

  for (int k = 0; k < steps_z; k++) {
    const bool odd_layer = hexagonal && (k & 1);
    const float ox = odd_layer ? 0.5f * spacing : 0.0f;
    const float oy = odd_layer ? dy / 3.0f : 0.0f;
    const float z = min.z + half + float(k) * dz;
    for (int j = 0; j < steps_y; j++) {
      /* Triangular rows: every other row is offset by half a spacing. Only the hexagonal lattice
       * is close packed; the grid must stay axis aligned (`row_shift` would put points on the
       * boundary and double the number of distinct columns). */
      const float row_shift = (hexagonal && (j & 1)) ? 0.5f * spacing : 0.0f;
      const float y = min.y + half + float(j) * dy + oy;
      for (int i = 0; i < steps_x; i++) {
        const float x = min.x + half + float(i) * spacing + row_shift + ox;
        const float3 point(x, y, z);
        float distance = 0.0f;
        if (!mesh_contains(bvh, point, orientation, &distance)) {
          continue;
        }
        r_positions.append(point);
        r_surface_distance.append(distance);
        if (r_positions.size() > MAX_POINTS) {
          return true;
        }
      }
    }
  }
  return true;
}

static void jitter_volume_points(const bke::BVHTreeFromMesh &bvh,
                                 const MutableSpan<float3> positions,
                                 const Span<float> surface_distance,
                                 const float spacing,
                                 const float jitter,
                                 const JitterLayer layer,
                                 const float orientation,
                                 const uint32_t seed)
{
  const float axis = 0.15f * math::clamp(jitter, 0.0f, 1.0f) * spacing;
  if (axis <= 0.0f) {
    return;
  }
  const float surface_band = 1.5f * spacing;
  threading::parallel_for(positions.index_range(), 2048, [&](const IndexRange range) {
    for (const int i : range) {
      if (layer == JitterLayer::Surface && surface_distance[i] > surface_band) {
        continue;
      }
      Rng rng(point_seed(seed, i, 3u));
      const float3 offset((rng.next_float() - 0.5f) * 2.0f * axis,
                          (rng.next_float() - 0.5f) * 2.0f * axis,
                          (rng.next_float() - 0.5f) * 2.0f * axis);
      const float3 candidate = positions[i] + offset;
      /* The lattice point is known to be inside, so falling back to it can never leave the mesh. */
      if (mesh_contains(bvh, candidate, orientation, nullptr)) {
        positions[i] = candidate;
      }
    }
  });
}

/** \} */

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();

  /* Match the native `Distribute Points on Faces`: the output sits above the inputs, even though
   * this node uses a custom socket order for its panels. */
  b.add_output<decl::Geometry>("Points"_ustr)
      .description("A point cloud with the distributed points and their `id`");

  b.add_input<decl::Geometry>("Geometry"_ustr)
      .supported_type(GeometryComponent::Type::Mesh)
      .is_default_link_socket()
      .description("Mesh to distribute points on or in. Volume mode needs a closed mesh");
  b.add_input<decl::Menu>("Mode"_ustr)
      .static_items(mode_items)
      .default_value(MenuValue(int(Mode::Surface)))
      .description("Scatter over the surface or fill the volume");
  b.add_input<decl::Float>("Spacing"_ustr)
      .default_value(0.05f)
      .min(1.0e-4f)
      .subtype(PROP_DISTANCE)
      .description("Target distance between neighbouring points in metres. On a surface one point "
                   "is placed per `Spacing^2` of area; in a volume it is the lattice step");

  {
    auto &panel = b.add_panel("Surface"_ustr).default_closed(false);
    panel.add_input<decl::Bool>("Relax"_ustr)
        .default_value(true)
        .usage_by_menu("Mode"_ustr, int(Mode::Surface))
        .description("Push points that are closer than `Spacing` apart, then project them back "
                     "onto the surface. Removes the clumps random placement leaves behind");
    panel.add_input<decl::Int>("Relax Iterations"_ustr)
        .default_value(8)
        .min(0)
        .max(100)
        .usage_by_menu("Mode"_ustr, int(Mode::Surface))
        .description("Relaxation passes; each pass is one KD-tree build plus one reprojection");
    panel.add_input<decl::Float>("Relax Strength"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Mode"_ustr, int(Mode::Surface))
        .description("Fraction of the repulsion applied per pass");
  }

  {
    auto &panel = b.add_panel("Volume"_ustr).default_closed(false);
    panel.add_input<decl::Menu>("Lattice"_ustr)
        .static_items(lattice_items)
        .default_value(MenuValue(int(LatticeType::Grid)))
        .usage_by_menu("Mode"_ustr, int(Mode::Volume))
        .description("Lattice arrangement of the interior points");
  }

  {
    auto &panel = b.add_panel("Jitter"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Jitter"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .description("Random offset as a fraction of the spacing. Capped at 15% of the spacing per "
                     "axis, so jittered points cannot cross each other");
    panel.add_input<decl::Menu>("Jitter Layer"_ustr)
        .static_items(jitter_layer_items)
        .default_value(MenuValue(int(JitterLayer::All)))
        .usage_by_menu("Mode"_ustr, int(Mode::Volume))
        .description("Which points of a volume lattice are jittered");
    panel.add_input<decl::Int>("Seed"_ustr)
        .default_value(0)
        .description("Random seed of the surface scattering and of the jitter");
  }
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  const Mesh *mesh = geometry.get_mesh();
  if (mesh == nullptr || mesh->verts_num == 0) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Distribute Points: the input geometry is not a mesh (or has no vertices)"));
    params.set_output("Points"_ustr, GeometrySet());
    return;
  }

  const Mode mode = Mode(params.extract_input<int>("Mode"_ustr));
  const float spacing = std::max(params.extract_input<float>("Spacing"_ustr), 1.0e-4f);
  const bool relax = params.extract_input<bool>("Relax"_ustr);
  const int relax_iterations = math::clamp(params.extract_input<int>("Relax Iterations"_ustr),
                                           0,
                                           100);
  const float relax_strength = math::clamp(params.extract_input<float>("Relax Strength"_ustr),
                                           0.0f,
                                           1.0f);
  const LatticeType lattice = LatticeType(params.extract_input<int>("Lattice"_ustr));
  const float jitter = math::clamp(params.extract_input<float>("Jitter"_ustr), 0.0f, 1.0f);
  const JitterLayer jitter_layer = JitterLayer(params.extract_input<int>("Jitter Layer"_ustr));
  const uint32_t seed = uint32_t(params.extract_input<int>("Seed"_ustr));

  const bke::BVHTreeFromMesh bvh = mesh->bvh_corner_tris();
  Vector<float3> positions;
  Vector<float> surface_distance;

  if (mode == Mode::Surface) {
    /* Refuse absurd counts before doing any work. */
    float area = 0.0f;
    for (const int3 &tri : mesh->corner_tris()) {
      const float3 &a = mesh->vert_positions()[mesh->corner_verts()[tri[0]]];
      const float3 &b = mesh->vert_positions()[mesh->corner_verts()[tri[1]]];
      const float3 &c = mesh->vert_positions()[mesh->corner_verts()[tri[2]]];
      area += 0.5f * math::length(math::cross(b - a, c - a));
    }
    if (float(MAX_POINTS) * spacing * spacing < area) {
      params.error_message_add(
          NodeWarningType::Warning,
          TIP_("Distribute Points: the requested spacing would create more than 8 million points; "
               "raise `Spacing`"));
      params.set_output("Points"_ustr, GeometrySet());
      return;
    }
    distribute_on_surface(*mesh, spacing, seed, positions);
    if (relax) {
      relax_surface_points(bvh, positions.as_mutable_span(), spacing, relax_iterations,
                           relax_strength);
    }
    jitter_surface_points(bvh, positions.as_mutable_span(), spacing, jitter, seed);
  }
  else {
    if (!mesh_is_closed(*mesh)) {
      params.error_message_add(
          NodeWarningType::Warning,
          TIP_("Distribute Points: the mesh is not closed, so the inside test (closest triangle "
               "normal) is not reliable. Seal the mesh for volume distribution"));
    }
    const float orientation = mesh_signed_volume(*mesh) < 0.0f ? -1.0f : 1.0f;
    const bool lattice_fits = distribute_in_volume(
        *mesh, lattice, spacing, orientation, positions, surface_distance);
    if (!lattice_fits) {
      params.error_message_add(
          NodeWarningType::Warning,
          TIP_("Distribute Points: the requested spacing would create more than 8 million lattice "
               "points; raise `Spacing`"));
      params.set_output("Points"_ustr, GeometrySet());
      return;
    }
    if (positions.is_empty()) {
      params.error_message_add(
          NodeWarningType::Warning,
          TIP_("Distribute Points: the lattice produced no points inside the mesh. Check the mesh "
               "winding (normals must point outwards) and `Spacing` (must be smaller than the "
               "thickness)"));
    }
    if (jitter > 0.0f && !positions.is_empty()) {
      jitter_volume_points(bvh, positions.as_mutable_span(), surface_distance.as_span(), spacing,
                           jitter, jitter_layer, orientation, seed);
    }
  }

  if (positions.size() > MAX_POINTS) {
    positions.resize(int(MAX_POINTS));
  }
  const int count = int(positions.size());
  PointCloud *pointcloud = BKE_pointcloud_new_nomain(count);
  if (count > 0) {
    pointcloud->positions_for_write().copy_from(positions.as_span());
    pointcloud->tag_positions_changed();
    bke::MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
    bke::SpanAttributeWriter<int> ids = attributes.lookup_or_add_for_write_span<int>(
        "id", AttrDomain::Point);
    for (const int i : IndexRange(count)) {
      ids.span[i] = i;
    }
    ids.finish();
  }
  params.set_output("Points"_ustr, GeometrySet::from_pointcloud(pointcloud));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeDistributePoints"_ustr);
  ntype.ui_name = "Distribute Points";
  ntype.ui_description =
      "Scatter points on a mesh surface (with relaxation) or fill a closed mesh with a grid or "
      "hexagonal lattice, with optional jitter";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_distribute_points_cc
