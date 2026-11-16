/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `White Water Solver`: foam, spray and bubbles on top of a solved fluid.
 *
 * The node is stateless like the other solvers: `White Water` is the point set of the previous
 * frame (fed back by a Simulation Zone), `Fluid` provides the flow velocity, `Surface` the mesh the
 * foam sticks to and the emission samples, `Colliders` kill the spray on impact. Everything else is
 * a parameter panel; the node returns the new point set plus one output per type.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "BLI_array.hh"
#include "BLI_math_base.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_curves.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"
#include "BKE_scene.hh"

#include "DEG_depsgraph_query.hh"

#include "NOD_bundle_type.hh"
#include "NOD_geometry_nodes_bundle.hh"

#include "node_geometry_util.hh"

#include "GEO_white_water_solver.hh"

namespace blender::nodes::node_geo_white_water_solver_cc {

using namespace blender::geometry;
using namespace blender::nodes::physics_bundles;

namespace {

constexpr StringRef COLLIDER_MESH_TYPE = "Blender.Collider.Mesh";
constexpr StringRef COLLIDER_BOX_TYPE = "Blender.Collider.Box";

/** Menu of the clustering model. A Houdini style explicit cluster mode will be a second item. */
enum class WhiteWaterClusterMode : int {
  Strain = 0,
};

static const EnumPropertyItem cluster_mode_items[] = {
    {int(WhiteWaterClusterMode::Strain),
     "STRAIN",
     0,
     N_("Strain"),
     N_("Clusters emerge from the neighbour links: a clump is held together while its relative "
        "speed stays below `Break Strain`, so a fast shear tears it apart. Links form again once "
        "the flow calms down")},
    {0, nullptr, 0, nullptr, nullptr},
};

template<typename T>
VArray<T> lookup_or_default(const bke::AttributeAccessor &attributes,
                            const StringRef name,
                            const AttrDomain domain,
                            const T &fallback,
                            const int size)
{
  const VArray<T> values = *attributes.lookup<T>(name, domain);
  if (values) {
    return values;
  }
  return VArray<T>::from_single(fallback, size);
}

/** Append the point set of one component to `r_particles` (the previous frame's state). */
void read_state(const GeometryComponent &component, whitewater::ParticleData &r_particles)
{
  const AttrDomain domain = AttrDomain::Point;
  const int count = component.attribute_domain_size(domain);
  if (count == 0) {
    return;
  }
  const std::optional<bke::AttributeAccessor> attributes = component.attributes();
  if (!attributes) {
    return;
  }
  const VArray<float3> positions = *attributes->lookup<float3>("position", domain);
  if (!positions) {
    return;
  }
  const VArray<float3> velocities = lookup_or_default<float3>(
      *attributes, "velocity", domain, float3(0.0f), count);
  const VArray<float> ages = lookup_or_default<float>(*attributes, "age", domain, 0.0f, count);
  const VArray<float> lifetimes = lookup_or_default<float>(
      *attributes, "lifetime", domain, 1.0f, count);
  const VArray<float> radii = lookup_or_default<float>(*attributes, "radius", domain, 0.01f, count);
  const VArray<int> types = lookup_or_default<int>(*attributes, "type", domain, 0, count);
  const VArray<int> clusters = lookup_or_default<int>(*attributes, "cluster", domain, -1, count);

  const int offset = r_particles.size();
  r_particles.reserve(offset + count);
  for (const int i : IndexRange(count)) {
    r_particles.positions.append(positions[i]);
    r_particles.velocities.append(velocities[i]);
    r_particles.ages.append(ages[i]);
    r_particles.lifetimes.append(lifetimes[i]);
    r_particles.radii.append(radii[i]);
    r_particles.types.append(types[i]);
    r_particles.clusters.append(clusters[i]);
    (void)offset;
  }
}

void read_fluid(const GeometryComponent &component,
                Vector<float3> &r_positions,
                Vector<float3> &r_velocities)
{
  const AttrDomain domain = AttrDomain::Point;
  const int count = component.attribute_domain_size(domain);
  if (count == 0) {
    return;
  }
  const std::optional<bke::AttributeAccessor> attributes = component.attributes();
  if (!attributes) {
    return;
  }
  const VArray<float3> positions = *attributes->lookup<float3>("position", domain);
  if (!positions) {
    return;
  }
  const VArray<float3> velocities = lookup_or_default<float3>(
      *attributes, "velocity", domain, float3(0.0f), count);
  for (const int i : IndexRange(count)) {
    r_positions.append(positions[i]);
    r_velocities.append(velocities[i]);
  }
}

void append_mesh_triangles(const Mesh &mesh, Vector<float3> &r_triangles)
{
  const Span<float3> verts = mesh.vert_positions();
  const Span<int3> tris = mesh.corner_tris();
  const Span<int> corner_verts = mesh.corner_verts();
  for (const int3 &tri : tris) {
    r_triangles.append(verts[corner_verts[tri[0]]]);
    r_triangles.append(verts[corner_verts[tri[1]]]);
    r_triangles.append(verts[corner_verts[tri[2]]]);
  }
}

void append_box_triangles(const Mesh &mesh, Vector<float3> &r_triangles)
{
  const Span<float3> verts = mesh.vert_positions();
  if (verts.is_empty()) {
    return;
  }
  float3 min = verts[0];
  float3 max = verts[0];
  for (const float3 &v : verts) {
    min = math::min(min, v);
    max = math::max(max, v);
  }
  const float3 c[8] = {
      float3(min.x, min.y, min.z),
      float3(max.x, min.y, min.z),
      float3(max.x, max.y, min.z),
      float3(min.x, max.y, min.z),
      float3(min.x, min.y, max.z),
      float3(max.x, min.y, max.z),
      float3(max.x, max.y, max.z),
      float3(min.x, max.y, max.z),
  };
  static const int quads[6][4] = {{0, 1, 2, 3}, {4, 7, 6, 5}, {0, 4, 5, 1},
                                  {1, 5, 6, 2}, {2, 6, 7, 3}, {3, 7, 4, 0}};
  for (const auto &quad : quads) {
    r_triangles.append(c[quad[0]]);
    r_triangles.append(c[quad[1]]);
    r_triangles.append(c[quad[2]]);
    r_triangles.append(c[quad[0]]);
    r_triangles.append(c[quad[2]]);
    r_triangles.append(c[quad[3]]);
  }
}

void append_collider(const Bundle &bundle, Vector<float3> &r_triangles)
{
  const GeometrySet *geometry = bundle.lookup_ptr<GeometrySet>(*BundleKey::from_str("geometry"));
  if (geometry == nullptr) {
    return;
  }
  const Mesh *mesh = geometry->get_mesh();
  if (mesh == nullptr) {
    return;
  }
  const std::optional<StringRef> type = bundle.type();
  if (type.has_value() && *type == COLLIDER_BOX_TYPE) {
    append_box_triangles(*mesh, r_triangles);
  }
  else {
    append_mesh_triangles(*mesh, r_triangles);
  }
}

void gather_colliders(const Bundle &bundle, Vector<float3> &r_triangles)
{
  auto is_collider = [](const StringRef type) {
    return type == COLLIDER_MESH_TYPE || type == COLLIDER_BOX_TYPE;
  };
  auto consume = [&](const Bundle &current) {
    const std::optional<StringRef> type = current.type();
    if (!type || !is_collider(*type)) {
      return;
    }
    append_collider(current, r_triangles);
  };
  consume(bundle);
  for (const StringRef path : gather_bundle_paths_by_bundle_type(bundle, is_collider)) {
    /* `Set Previous World Items` nests a copy of every collider; it is the same one a frame
     * earlier, not a second collider. */
    if (path.find("previous") != std::string::npos) {
      continue;
    }
    const BundlePtr *bundle_ptr = bundle.lookup_path_ptr<BundlePtr>(path);
    if (bundle_ptr == nullptr || !*bundle_ptr) {
      continue;
    }
    consume(**bundle_ptr);
  }
}

/** Write one type (or every type when `type_filter < 0`) into a new point cloud. */
void write_points(const whitewater::ParticleData &particles,
                  const int type_filter,
                  GeometrySet &r_geometry)
{
  int count = 0;
  for (const int i : IndexRange(particles.size())) {
    if (type_filter < 0 || particles.types[i] == type_filter) {
      count++;
    }
  }
  PointCloud *pointcloud = BKE_pointcloud_new_nomain(count);
  if (count == 0) {
    r_geometry.replace_pointcloud(pointcloud);
    return;
  }
  bke::MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
  const AttrDomain domain = AttrDomain::Point;
  bke::SpanAttributeWriter<float3> positions = attributes.lookup_or_add_for_write_span<float3>(
      "position", domain);
  bke::SpanAttributeWriter<float3> velocities = attributes.lookup_or_add_for_write_span<float3>(
      "velocity", domain);
  bke::SpanAttributeWriter<float> ages = attributes.lookup_or_add_for_write_span<float>("age",
                                                                                       domain);
  bke::SpanAttributeWriter<float> lifetimes = attributes.lookup_or_add_for_write_span<float>(
      "lifetime", domain);
  bke::SpanAttributeWriter<float> radii = attributes.lookup_or_add_for_write_span<float>("radius",
                                                                                        domain);
  bke::SpanAttributeWriter<int> types = attributes.lookup_or_add_for_write_span<int>("type",
                                                                                     domain);
  bke::SpanAttributeWriter<int> clusters = attributes.lookup_or_add_for_write_span<int>(
      "cluster", domain);

  int write = 0;
  for (const int i : IndexRange(particles.size())) {
    if (type_filter >= 0 && particles.types[i] != type_filter) {
      continue;
    }
    positions.span[write] = particles.positions[i];
    velocities.span[write] = particles.velocities[i];
    ages.span[write] = particles.ages[i];
    lifetimes.span[write] = particles.lifetimes[i];
    radii.span[write] = particles.radii[i];
    types.span[write] = particles.types[i];
    clusters.span[write] = particles.clusters[i];
    write++;
  }
  positions.finish();
  velocities.finish();
  ages.finish();
  lifetimes.finish();
  radii.finish();
  types.finish();
  clusters.finish();
  r_geometry.replace_pointcloud(pointcloud);
}

BundlePtr stats_to_bundle(const whitewater::Stats &stats)
{
  BundlePtr bundle_ptr = Bundle::create();
  Bundle &bundle = bundle_ptr.ensure_mutable_inplace();
  bundle.add(*BundleKey::from_str("particles"), stats.particles);
  bundle.add(*BundleKey::from_str("foam"), stats.foam);
  bundle.add(*BundleKey::from_str("spray"), stats.spray);
  bundle.add(*BundleKey::from_str("bubbles"), stats.bubbles);
  bundle.add(*BundleKey::from_str("emitted"), stats.emitted);
  bundle.add(*BundleKey::from_str("clusters"), stats.clusters);
  bundle.add(*BundleKey::from_str("links"), stats.links);
  bundle.add(*BundleKey::from_str("max_speed"), stats.max_speed);
  bundle.add(*BundleKey::from_str("milliseconds"), float(stats.milliseconds));
  return bundle_ptr;
}

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();

  b.add_input<decl::Menu>("Mode"_ustr)
      .static_items(cluster_mode_items)
      .default_value(MenuValue(int(WhiteWaterClusterMode::Strain)))
      .description("How the clumps of white water are built");

  b.add_input<decl::Geometry>("White Water"_ustr)
      .supported_type({GeometryComponent::Type::Mesh, GeometryComponent::Type::PointCloud})
      .description(
          "The white water of the previous frame, usually `Simulation Input`. Leave it empty to "
          "start a new simulation");
  b.add_input<decl::Geometry>("Fluid"_ustr)
      .supported_type({GeometryComponent::Type::Mesh, GeometryComponent::Type::PointCloud})
      .description(
          "Fluid particles with a `velocity` attribute: the flow speed drives the emission and the "
          "foam drifts with it. The point cloud of `Textura Dynamics` or `Puncliga Dynamics`");
  b.add_input<decl::Geometry>("Surface"_ustr)
      .supported_type({GeometryComponent::Type::Mesh})
      .description(
          "Fluid surface mesh (`Fluid Surface`): white water is emitted from its triangles and the "
          "foam sticks to it");
  b.add_input<decl::Bundle>("Colliders"_ustr)
      .structure_type(StructureType::Single)
      .description("`Dilapido Collider` bundles; the spray dies when it touches one");
  b.add_input<decl::Float>("Delta Time"_ustr)
      .default_value(1.0f / 24.0f)
      .min(0.0f)
      .subtype(PROP_TIME_ABSOLUTE)
      .description("Length of the frame in seconds; drive it with `Simulation Input > Delta Time`");

  {
    auto &panel = b.add_panel("Emission"_ustr).default_closed(false);
    panel.add_input<decl::Bool>("Foam"_ustr)
        .default_value(true)
        .description("Emit foam where the surface moves faster than `Foam Threshold`");
    panel.add_input<decl::Bool>("Spray"_ustr)
        .default_value(true)
        .description("Emit droplets that fly off the surface and fall back");
    panel.add_input<decl::Bool>("Bubbles"_ustr)
        .default_value(false)
        .description("Emit bubbles below the surface; they rise and pop into foam");
    panel.add_input<decl::Float>("Emission Scale"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .max(20.0f)
        .description("Multiplier of all emission rates");
    panel.add_input<decl::Int>("Seed"_ustr)
        .default_value(0)
        .description("Random seed of the emission and the sizes");
    panel.add_input<decl::Float>("Foam Emission"_ustr)
        .default_value(40.0f)
        .min(0.0f)
        .subtype(PROP_NONE)
        .description("Foam particles per square metre of surface per second at full speed");
    panel.add_input<decl::Float>("Foam Threshold"_ustr)
        .default_value(1.5f)
        .min(0.0f)
        .subtype(PROP_VELOCITY)
        .description("Surface speed below which no foam is emitted");
    panel.add_input<decl::Float>("Foam Full Speed"_ustr)
        .default_value(5.0f)
        .min(0.01f)
        .subtype(PROP_VELOCITY)
        .description("Surface speed at which the foam emission reaches `Foam Emission`");
    panel.add_input<decl::Float>("Spray Emission"_ustr)
        .default_value(20.0f)
        .min(0.0f)
        .subtype(PROP_NONE)
        .description("Spray particles per square metre of surface per second at full speed");
    panel.add_input<decl::Float>("Spray Threshold"_ustr)
        .default_value(2.5f)
        .min(0.0f)
        .subtype(PROP_VELOCITY)
        .description("Surface speed below which no spray is emitted");
    panel.add_input<decl::Float>("Spray Full Speed"_ustr)
        .default_value(6.0f)
        .min(0.01f)
        .subtype(PROP_VELOCITY)
        .description("Surface speed at which the spray emission reaches `Spray Emission`");
    panel.add_input<decl::Float>("Bubble Emission"_ustr)
        .default_value(8.0f)
        .min(0.0f)
        .subtype(PROP_NONE)
        .description("Bubbles per square metre of surface per second at full speed");
    panel.add_input<decl::Float>("Bubble Threshold"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .subtype(PROP_VELOCITY)
        .description("Surface speed below which no bubbles are emitted");
    panel.add_input<decl::Float>("Bubble Full Speed"_ustr)
        .default_value(2.0f)
        .min(0.01f)
        .subtype(PROP_VELOCITY)
        .description("Surface speed at which the bubble emission reaches `Bubble Emission`");
  }
  {
    auto &panel = b.add_panel("Foam"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Foam Lifetime"_ustr)
        .default_value(4.0f)
        .min(0.01f)
        .subtype(PROP_TIME_ABSOLUTE)
        .usage_by_bool("Foam"_ustr, true)
        .description("Seconds a foam particle lives");
    panel.add_input<decl::Float>("Foam Size"_ustr)
        .default_value(0.02f)
        .min(1e-4f)
        .subtype(PROP_DISTANCE)
        .usage_by_bool("Foam"_ustr, true)
        .description("Radius of a foam particle");
    panel.add_input<decl::Float>("Foam Stick"_ustr)
        .default_value(0.7f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_bool("Foam"_ustr, true)
        .description(
            "How strongly the foam rides the flow and stays on the surface (0 = static stickers)");
  }
  {
    auto &panel = b.add_panel("Spray"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Spray Lifetime"_ustr)
        .default_value(1.2f)
        .min(0.01f)
        .subtype(PROP_TIME_ABSOLUTE)
        .usage_by_bool("Spray"_ustr, true)
        .description("Seconds a droplet lives if it does not hit anything");
    panel.add_input<decl::Float>("Spray Size"_ustr)
        .default_value(0.015f)
        .min(1e-4f)
        .subtype(PROP_DISTANCE)
        .usage_by_bool("Spray"_ustr, true)
        .description("Radius of a spray droplet");
    panel.add_input<decl::Float>("Spray Speed"_ustr)
        .default_value(1.5f)
        .min(0.0f)
        .subtype(PROP_VELOCITY)
        .usage_by_bool("Spray"_ustr, true)
        .description("Extra speed along the surface normal when a droplet is born");
    panel.add_input<decl::Float>("Drag"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .usage_by_bool("Spray"_ustr, true)
        .description("Air drag of spray and bubbles (1/s)");
    panel.add_input<decl::Vector>("Gravity"_ustr)
        .default_value(float3(0.0f, 0.0f, -9.81f))
        .subtype(PROP_ACCELERATION)
        .usage_by_bool("Spray"_ustr, true)
        .description("Gravity of the spray droplets");
  }
  {
    auto &panel = b.add_panel("Bubbles"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Bubble Lifetime"_ustr)
        .default_value(3.0f)
        .min(0.01f)
        .subtype(PROP_TIME_ABSOLUTE)
        .usage_by_bool("Bubbles"_ustr, true)
        .description("Seconds a bubble lives");
    panel.add_input<decl::Float>("Bubble Size"_ustr)
        .default_value(0.01f)
        .min(1e-4f)
        .subtype(PROP_DISTANCE)
        .usage_by_bool("Bubbles"_ustr, true)
        .description("Radius of a bubble");
    panel.add_input<decl::Float>("Buoyancy"_ustr)
        .default_value(2.0f)
        .min(0.0f)
        .usage_by_bool("Bubbles"_ustr, true)
        .description("Upward acceleration of the bubbles (m/s^2)");
    panel.add_input<decl::Float>("Bubble Depth"_ustr)
        .default_value(0.15f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_bool("Bubbles"_ustr, true)
        .description("How far below the surface a bubble is born");
    panel.add_input<decl::Bool>("Bubble to Foam"_ustr)
        .default_value(true)
        .usage_by_bool("Bubbles"_ustr, true)
        .description("Turn a bubble into foam when it reaches the surface");
  }
  {
    auto &panel = b.add_panel("Clusters"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Cohesion"_ustr)
        .default_value(2.0f)
        .min(0.0f)
        .description(
            "Attraction between neighbours, as an acceleration at contact. Makes the white water "
            "clump instead of moving as independent points");
    panel.add_input<decl::Float>("Cohesion Radius"_ustr)
        .default_value(0.05f)
        .min(1e-4f)
        .subtype(PROP_DISTANCE)
        .description("Neighbours within this distance attract each other");
    panel.add_input<decl::Float>("Separation"_ustr)
        .default_value(40.0f)
        .min(0.0f)
        .description("Repulsion between overlapping particles, as an acceleration at contact");
    panel.add_input<decl::Float>("Separation Radius"_ustr)
        .default_value(0.02f)
        .min(1e-4f)
        .subtype(PROP_DISTANCE)
        .description("Particles closer than this push each other apart");
    panel.add_input<decl::Bool>("Same Type Only"_ustr)
        .default_value(true)
        .description("Foam, spray and bubbles only interact with their own type");
    panel.add_input<decl::Float>("Break Strain"_ustr)
        .default_value(8.0f)
        .min(0.0f)
        .description(
            "A cohesion link breaks when the relative speed divided by the cohesion radius is "
            "above this (1/s). Low values tear the clumps apart easily, high values keep them "
            "together");
    panel.add_input<decl::Int>("Max Cluster Size"_ustr)
        .default_value(64)
        .min(1)
        .description("Larger clumps are split into deterministic sub clusters");
    panel.add_input<decl::Float>("Cluster Damping"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .description("Relative velocity damping between linked neighbours (1/s)");
  }
  {
    auto &panel = b.add_panel("Limits"_ustr).default_closed(true);
    panel.add_input<decl::Int>("Max Particles"_ustr)
        .default_value(50000)
        .min(0)
        .description("Hard budget; the oldest particles are dropped first");
    panel.add_input<decl::Bool>("Use Bounds"_ustr)
        .default_value(false)
        .description("Delete particles that leave a box (useful to keep splash clean)");
    panel.add_input<decl::Vector>("Bounds Min"_ustr)
        .default_value(float3(-100.0f))
        .subtype(PROP_DISTANCE)
        .usage_by_bool("Use Bounds"_ustr, true)
        .description("Lower corner of the culling box");
    panel.add_input<decl::Vector>("Bounds Max"_ustr)
        .default_value(float3(100.0f))
        .subtype(PROP_DISTANCE)
        .usage_by_bool("Use Bounds"_ustr, true)
        .description("Upper corner of the culling box");
  }

  b.add_output<decl::Geometry>("White Water"_ustr).propagate_all();
  b.add_output<decl::Geometry>("Foam"_ustr);
  b.add_output<decl::Geometry>("Spray"_ustr);
  b.add_output<decl::Geometry>("Bubbles"_ustr);
  b.add_output<decl::Bundle>("Stats"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  whitewater::Settings settings;
  settings.foam = params.extract_input<bool>("Foam"_ustr);
  settings.spray = params.extract_input<bool>("Spray"_ustr);
  settings.bubbles = params.extract_input<bool>("Bubbles"_ustr);
  settings.emission_scale = std::max(0.0f, params.extract_input<float>("Emission Scale"_ustr));
  settings.seed = params.extract_input<int>("Seed"_ustr);
  settings.foam_emission = std::max(0.0f, params.extract_input<float>("Foam Emission"_ustr));
  settings.foam_threshold = std::max(0.0f, params.extract_input<float>("Foam Threshold"_ustr));
  settings.foam_full_speed = std::max(settings.foam_threshold + 1e-3f,
                                      params.extract_input<float>("Foam Full Speed"_ustr));
  settings.spray_emission = std::max(0.0f, params.extract_input<float>("Spray Emission"_ustr));
  settings.spray_threshold = std::max(0.0f, params.extract_input<float>("Spray Threshold"_ustr));
  settings.spray_full_speed = std::max(settings.spray_threshold + 1e-3f,
                                       params.extract_input<float>("Spray Full Speed"_ustr));
  settings.bubble_emission = std::max(0.0f, params.extract_input<float>("Bubble Emission"_ustr));
  settings.bubble_threshold = std::max(0.0f, params.extract_input<float>("Bubble Threshold"_ustr));
  settings.bubble_full_speed = std::max(settings.bubble_threshold + 1e-3f,
                                        params.extract_input<float>("Bubble Full Speed"_ustr));
  settings.foam_lifetime = std::max(0.01f, params.extract_input<float>("Foam Lifetime"_ustr));
  settings.foam_size = std::max(1e-4f, params.extract_input<float>("Foam Size"_ustr));
  settings.foam_stick = math::clamp(params.extract_input<float>("Foam Stick"_ustr), 0.0f, 1.0f);
  settings.spray_lifetime = std::max(0.01f, params.extract_input<float>("Spray Lifetime"_ustr));
  settings.spray_size = std::max(1e-4f, params.extract_input<float>("Spray Size"_ustr));
  settings.spray_speed = std::max(0.0f, params.extract_input<float>("Spray Speed"_ustr));
  settings.drag = std::max(0.0f, params.extract_input<float>("Drag"_ustr));
  settings.gravity = params.extract_input<float3>("Gravity"_ustr);
  settings.bubble_lifetime = std::max(0.01f, params.extract_input<float>("Bubble Lifetime"_ustr));
  settings.bubble_size = std::max(1e-4f, params.extract_input<float>("Bubble Size"_ustr));
  settings.bubble_buoyancy = std::max(0.0f, params.extract_input<float>("Buoyancy"_ustr));
  settings.bubble_depth = std::max(0.0f, params.extract_input<float>("Bubble Depth"_ustr));
  settings.bubble_to_foam = params.extract_input<bool>("Bubble to Foam"_ustr);
  settings.cohesion = std::max(0.0f, params.extract_input<float>("Cohesion"_ustr));
  settings.cohesion_radius = std::max(1e-4f, params.extract_input<float>("Cohesion Radius"_ustr));
  settings.separation = std::max(0.0f, params.extract_input<float>("Separation"_ustr));
  settings.separation_radius = std::max(1e-4f, params.extract_input<float>("Separation Radius"_ustr));
  settings.same_type_only = params.extract_input<bool>("Same Type Only"_ustr);
  settings.break_strain = std::max(0.0f, params.extract_input<float>("Break Strain"_ustr));
  settings.max_cluster_size = std::max(1, params.extract_input<int>("Max Cluster Size"_ustr));
  settings.cluster_damping = std::max(0.0f, params.extract_input<float>("Cluster Damping"_ustr));
  settings.max_particles = std::max(0, params.extract_input<int>("Max Particles"_ustr));
  settings.use_bounds = params.extract_input<bool>("Use Bounds"_ustr);
  settings.bounds_min = params.extract_input<float3>("Bounds Min"_ustr);
  settings.bounds_max = params.extract_input<float3>("Bounds Max"_ustr);

  const int mode = params.extract_input<MenuValue>("Mode"_ustr).value;
  settings.cluster_mode = mode == int(WhiteWaterClusterMode::Strain) ? whitewater::ClusterMode::Strain :
                                                                       whitewater::ClusterMode::Strain;

  if (const Scene *scene = DEG_get_input_scene(params.depsgraph())) {
    settings.frame = scene->r.cfra;
  }

  whitewater::ParticleData particles;
  const GeometrySet state_set = params.extract_input<GeometrySet>("White Water"_ustr);
  for (const GeometryComponent::Type type : {GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Mesh})
  {
    const GeometryComponent *component = state_set.get_component(type);
    if (component != nullptr) {
      read_state(*component, particles);
    }
  }

  Vector<float3> fluid_positions;
  Vector<float3> fluid_velocities;
  const GeometrySet fluid_set = params.extract_input<GeometrySet>("Fluid"_ustr);
  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud})
  {
    const GeometryComponent *component = fluid_set.get_component(type);
    if (component != nullptr) {
      read_fluid(*component, fluid_positions, fluid_velocities);
    }
  }

  const GeometrySet surface_set = params.extract_input<GeometrySet>("Surface"_ustr);
  const Mesh *surface = surface_set.get_mesh();

  Vector<float3> collider_triangles;
  if (BundlePtr colliders = params.get_input<BundlePtr>("Colliders"_ustr)) {
    gather_colliders(*colliders, collider_triangles);
  }

  const float delta_time = std::max(0.0f, params.extract_input<float>("Delta Time"_ustr));

  if (particles.size() == 0 && (surface == nullptr || fluid_positions.is_empty())) {
    /* Nothing to simulate and nothing to emit from. */
    params.set_default_remaining_outputs();
    return;
  }

  whitewater::Stats stats;
  const Span<const float3> fluid_position_span(fluid_positions.data(), fluid_positions.size());
  const Span<const float3> fluid_velocity_span(fluid_velocities.data(), fluid_velocities.size());
  const Span<const float3> collider_span(collider_triangles.data(), collider_triangles.size());
  whitewater::solve_frame(particles,
                          surface,
                          fluid_position_span,
                          fluid_velocity_span,
                          collider_span,
                          settings,
                          delta_time,
                          stats);

  GeometrySet all;
  write_points(particles, -1, all);
  params.set_output("White Water"_ustr, std::move(all));
  GeometrySet foam;
  write_points(particles, int(whitewater::ParticleType::Foam), foam);
  params.set_output("Foam"_ustr, std::move(foam));
  GeometrySet spray;
  write_points(particles, int(whitewater::ParticleType::Spray), spray);
  params.set_output("Spray"_ustr, std::move(spray));
  GeometrySet bubbles;
  write_points(particles, int(whitewater::ParticleType::Bubble), bubbles);
  params.set_output("Bubbles"_ustr, std::move(bubbles));
  params.set_output("Stats"_ustr, stats_to_bundle(stats));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeWhiteWaterSolver"_ustr);
  ntype.ui_name = "White Water Solver";
  ntype.ui_description =
      "Add foam, spray and bubbles to a solved fluid: emission from the surface, ballistic spray, "
      "buoyant bubbles and neighbour cohesion that lets the clumps merge and tear apart";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_white_water_solver_cc
