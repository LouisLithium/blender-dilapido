/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `White Water Solver`: foam, spray and bubbles on top of a solved fluid.
 *
 * Like the other Dilapido solvers the node only has `World`, `Delta Time` and `Stats`; everything it
 * works on lives in the world bundle:
 *
 *  - `geometry`: the white water of the previous frame (the state), written back,
 *  - `fluid`: the fluid particles with their `velocity` attribute (emission density, foam drift),
 *  - `surface`: the fluid surface mesh (emission samples, foam sticking),
 *  - `effectors`: the colliders (`Dilapido Collider`), the spray dies on them,
 *  - `settings`: `Blender.WhiteWaterSettings` with every parameter.
 *
 * The `White Water Dynamics` asset builds that world inside a simulation zone; the node is stateless
 * like the others, so replaying a frame always produces the same particles.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "BLI_array.hh"
#include "BLI_math_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"
#include "BKE_scene.hh"

#include "DEG_depsgraph_query.hh"

#include "NOD_bundle_type.hh"
#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_physics_bundles.hh"

#include "node_geometry_util.hh"

#include "GEO_white_water_solver.hh"

namespace blender::nodes::node_geo_white_water_solver_cc {

using namespace blender::geometry;
using namespace blender::nodes::physics_bundles;

namespace {

constexpr StringRef COLLIDER_MESH_TYPE = "Blender.Collider.Mesh";
constexpr StringRef COLLIDER_BOX_TYPE = "Blender.Collider.Box";

/** All solver parameters, stored in the world bundle so the asset can expose them in panels. */
class WhiteWaterSettingsBundle {
 public:
  static constexpr StringRefNull name = "Blender.WhiteWaterSettings";
  static const FlatBundleTypePtr &get_bundle_type();
};

const FlatBundleTypePtr &WhiteWaterSettingsBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(WhiteWaterSettingsBundle::name);
    b.add<decl::Bool>("foam"_ustr);
    b.add<decl::Bool>("spray"_ustr);
    b.add<decl::Bool>("bubbles"_ustr);
    b.add<decl::Float>("foam_emission"_ustr).min(0.0f);
    b.add<decl::Float>("spray_emission"_ustr).min(0.0f);
    b.add<decl::Float>("bubble_emission"_ustr).min(0.0f);
    b.add<decl::Float>("emission_scale"_ustr).min(0.0f);
    b.add<decl::Float>("vorticity_weight"_ustr).min(0.0f);
    b.add<decl::Float>("vorticity_scale"_ustr).min(1e-3f);
    b.add<decl::Float>("trapped_air_weight"_ustr).min(0.0f);
    b.add<decl::Float>("divergence_scale"_ustr).min(1e-3f);
    b.add<decl::Bool>("air_to_bubbles"_ustr);
    b.add<decl::Int>("seed"_ustr);
    b.add<decl::Float>("foam_threshold"_ustr).min(0.0f);
    b.add<decl::Float>("foam_full_speed"_ustr).min(0.01f);
    b.add<decl::Float>("spray_threshold"_ustr).min(0.0f);
    b.add<decl::Float>("spray_full_speed"_ustr).min(0.01f);
    b.add<decl::Float>("bubble_threshold"_ustr).min(0.0f);
    b.add<decl::Float>("bubble_full_speed"_ustr).min(0.01f);
    b.add<decl::Float>("foam_lifetime"_ustr).min(0.01f);
    b.add<decl::Float>("spray_lifetime"_ustr).min(0.01f);
    b.add<decl::Float>("bubble_lifetime"_ustr).min(0.01f);
    b.add<decl::Float>("foam_size"_ustr).min(1e-4f);
    b.add<decl::Float>("spray_size"_ustr).min(1e-4f);
    b.add<decl::Float>("bubble_size"_ustr).min(1e-4f);
    b.add<decl::Float>("foam_stick"_ustr).min(0.0f).max(1.0f);
    b.add<decl::Vector>("up"_ustr);
    b.add<decl::Float>("foam_upward"_ustr).min(-1.0f).max(1.0f);
    b.add<decl::Bool>("foam_underside_kill"_ustr);
    b.add<decl::Float>("spray_speed"_ustr).min(0.0f);
    b.add<decl::Float>("bubble_buoyancy"_ustr).min(0.0f);
    b.add<decl::Float>("bubble_depth"_ustr).min(0.0f);
    b.add<decl::Bool>("bubble_to_foam"_ustr);
    b.add<decl::Vector>("gravity"_ustr);
    b.add<decl::Float>("drag"_ustr).min(0.0f);
    b.add<decl::Int>("cluster_mode"_ustr).min(0);
    b.add<decl::Float>("cohesion"_ustr).min(0.0f);
    b.add<decl::Float>("cohesion_radius"_ustr).min(1e-4f);
    b.add<decl::Float>("separation"_ustr).min(0.0f);
    b.add<decl::Float>("separation_radius"_ustr).min(1e-4f);
    b.add<decl::Bool>("same_type_only"_ustr);
    b.add<decl::Float>("break_strain"_ustr).min(0.0f);
    b.add<decl::Int>("max_cluster_size"_ustr).min(1);
    b.add<decl::Float>("cluster_damping"_ustr).min(0.0f);
    b.add<decl::Int>("max_particles"_ustr).min(0);
    b.add<decl::Bool>("use_bounds"_ustr);
    b.add<decl::Vector>("bounds_min"_ustr);
    b.add<decl::Vector>("bounds_max"_ustr);
    const FlatBundleTypePtr flat_type = b.build();
    BundleTypeRegistry::register_type(flat_type);
    return flat_type;
  }();
  return bundle_type;
}

NestedBundleTypePtr make_world_type()
{
  Vector<std::shared_ptr<const FlatBundleType>> types;
  types.append(WhiteWaterSettingsBundle::get_bundle_type());
  types.append(MeshColliderBundle::get_bundle_type());
  types.append(BoxColliderBundle::get_bundle_type());

  NestedBundleTypePtr world_type = std::make_shared<const NestedBundleType>(
      "Blender.WhiteWaterSolverWorld", std::move(types));
  BundleTypeRegistry::register_type(world_type);
  return world_type;
}

/* -------------------------------------------------------------------- */
/** \name Reading the world
 * \{ */

float read_float(const Bundle &bundle, const StringRef key, const float fallback)
{
  if (const float *value = bundle.lookup_ptr<float>(*BundleKey::from_str(key))) {
    return *value;
  }
  return fallback;
}

int read_int(const Bundle &bundle, const StringRef key, const int fallback)
{
  if (const int *value = bundle.lookup_ptr<int>(*BundleKey::from_str(key))) {
    return *value;
  }
  return fallback;
}

bool read_bool(const Bundle &bundle, const StringRef key, const bool fallback)
{
  if (const bool *value = bundle.lookup_ptr<bool>(*BundleKey::from_str(key))) {
    return *value;
  }
  return fallback;
}

float3 read_vector(const Bundle &bundle, const StringRef key, const float3 &fallback)
{
  if (const float3 *value = bundle.lookup_ptr<float3>(*BundleKey::from_str(key))) {
    return *value;
  }
  return fallback;
}

whitewater::Settings read_settings(const Bundle &world)
{
  whitewater::Settings settings;
  const BundlePtr *bundle_ptr = world.lookup_ptr<BundlePtr>(*BundleKey::from_str("settings"));
  if (bundle_ptr == nullptr || !*bundle_ptr) {
    return settings;
  }
  const Bundle &bundle = **bundle_ptr;
  settings.foam = read_bool(bundle, "foam", settings.foam);
  settings.spray = read_bool(bundle, "spray", settings.spray);
  settings.bubbles = read_bool(bundle, "bubbles", settings.bubbles);
  settings.foam_emission = std::max(0.0f, read_float(bundle, "foam_emission", settings.foam_emission));
  settings.spray_emission = std::max(
      0.0f, read_float(bundle, "spray_emission", settings.spray_emission));
  settings.bubble_emission = std::max(
      0.0f, read_float(bundle, "bubble_emission", settings.bubble_emission));
  settings.emission_scale = std::max(0.0f,
                                     read_float(bundle, "emission_scale", settings.emission_scale));
  settings.vorticity_weight = std::max(0.0f,
                                       read_float(bundle, "vorticity_weight",
                                                  settings.vorticity_weight));
  settings.vorticity_scale = std::max(1e-3f,
                                      read_float(bundle, "vorticity_scale",
                                                 settings.vorticity_scale));
  settings.trapped_air_weight = std::max(0.0f,
                                         read_float(bundle, "trapped_air_weight",
                                                    settings.trapped_air_weight));
  settings.divergence_scale = std::max(1e-3f,
                                       read_float(bundle, "divergence_scale",
                                                  settings.divergence_scale));
  settings.air_to_bubbles = read_bool(bundle, "air_to_bubbles", settings.air_to_bubbles);
  settings.seed = read_int(bundle, "seed", settings.seed);
  settings.foam_threshold = std::max(0.0f,
                                     read_float(bundle, "foam_threshold", settings.foam_threshold));
  settings.foam_full_speed = std::max(settings.foam_threshold + 1e-3f,
                                      read_float(bundle, "foam_full_speed",
                                                 settings.foam_full_speed));
  settings.spray_threshold = std::max(
      0.0f, read_float(bundle, "spray_threshold", settings.spray_threshold));
  settings.spray_full_speed = std::max(settings.spray_threshold + 1e-3f,
                                       read_float(bundle, "spray_full_speed",
                                                  settings.spray_full_speed));
  settings.bubble_threshold = std::max(
      0.0f, read_float(bundle, "bubble_threshold", settings.bubble_threshold));
  settings.bubble_full_speed = std::max(settings.bubble_threshold + 1e-3f,
                                        read_float(bundle, "bubble_full_speed",
                                                   settings.bubble_full_speed));
  settings.foam_lifetime = std::max(0.01f,
                                    read_float(bundle, "foam_lifetime", settings.foam_lifetime));
  settings.spray_lifetime = std::max(0.01f,
                                     read_float(bundle, "spray_lifetime", settings.spray_lifetime));
  settings.bubble_lifetime = std::max(
      0.01f, read_float(bundle, "bubble_lifetime", settings.bubble_lifetime));
  settings.foam_size = std::max(1e-4f, read_float(bundle, "foam_size", settings.foam_size));
  settings.spray_size = std::max(1e-4f, read_float(bundle, "spray_size", settings.spray_size));
  settings.bubble_size = std::max(1e-4f, read_float(bundle, "bubble_size", settings.bubble_size));
  settings.foam_stick = math::clamp(read_float(bundle, "foam_stick", settings.foam_stick),
                                    0.0f,
                                    1.0f);
  settings.up = read_vector(bundle, "up", settings.up);
  settings.foam_upward = math::clamp(read_float(bundle, "foam_upward", settings.foam_upward),
                                      -1.0f,
                                      1.0f);
  settings.foam_underside_kill = read_bool(bundle, "foam_underside_kill",
                                           settings.foam_underside_kill);
  settings.spray_speed = std::max(0.0f, read_float(bundle, "spray_speed", settings.spray_speed));
  settings.bubble_buoyancy = std::max(0.0f,
                                      read_float(bundle, "bubble_buoyancy",
                                                 settings.bubble_buoyancy));
  settings.bubble_depth = std::max(0.0f,
                                   read_float(bundle, "bubble_depth", settings.bubble_depth));
  settings.bubble_to_foam = read_bool(bundle, "bubble_to_foam", settings.bubble_to_foam);
  settings.gravity = read_vector(bundle, "gravity", settings.gravity);
  settings.drag = std::max(0.0f, read_float(bundle, "drag", settings.drag));
  settings.cluster_mode = read_int(bundle, "cluster_mode", int(settings.cluster_mode)) == 0 ?
                              whitewater::ClusterMode::Strain :
                              whitewater::ClusterMode::Strain;
  settings.cohesion = std::max(0.0f, read_float(bundle, "cohesion", settings.cohesion));
  settings.cohesion_radius = std::max(1e-4f,
                                      read_float(bundle, "cohesion_radius",
                                                 settings.cohesion_radius));
  settings.separation = std::max(0.0f, read_float(bundle, "separation", settings.separation));
  settings.separation_radius = std::max(1e-4f,
                                        read_float(bundle, "separation_radius",
                                                   settings.separation_radius));
  settings.same_type_only = read_bool(bundle, "same_type_only", settings.same_type_only);
  settings.break_strain = std::max(0.0f,
                                   read_float(bundle, "break_strain", settings.break_strain));
  settings.max_cluster_size = std::max(1, read_int(bundle, "max_cluster_size",
                                                   settings.max_cluster_size));
  settings.cluster_damping = std::max(0.0f,
                                      read_float(bundle, "cluster_damping",
                                                 settings.cluster_damping));
  settings.max_particles = std::max(0, read_int(bundle, "max_particles", settings.max_particles));
  settings.use_bounds = read_bool(bundle, "use_bounds", settings.use_bounds);
  settings.bounds_min = read_vector(bundle, "bounds_min", settings.bounds_min);
  settings.bounds_max = read_vector(bundle, "bounds_max", settings.bounds_max);
  return settings;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Particles
 * \{ */

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

  r_particles.reserve(r_particles.size() + count);
  for (const int i : IndexRange(count)) {
    r_particles.positions.append(positions[i]);
    r_particles.velocities.append(velocities[i]);
    r_particles.ages.append(ages[i]);
    r_particles.lifetimes.append(lifetimes[i]);
    r_particles.radii.append(radii[i]);
    r_particles.types.append(types[i]);
    r_particles.clusters.append(clusters[i]);
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

/** Write the white water into a new point cloud. */
void write_particles(const whitewater::ParticleData &particles, GeometrySet &r_geometry)
{
  const int count = particles.size();
  PointCloud *pointcloud = BKE_pointcloud_new_nomain(count);
  if (count > 0) {
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
    bke::SpanAttributeWriter<float> radii = attributes.lookup_or_add_for_write_span<float>(
        "radius", domain);
    bke::SpanAttributeWriter<int> types = attributes.lookup_or_add_for_write_span<int>("type",
                                                                                       domain);
    bke::SpanAttributeWriter<int> clusters = attributes.lookup_or_add_for_write_span<int>(
        "cluster", domain);
    for (const int i : IndexRange(count)) {
      positions.span[i] = particles.positions[i];
      velocities.span[i] = particles.velocities[i];
      ages.span[i] = particles.ages[i];
      lifetimes.span[i] = particles.lifetimes[i];
      radii.span[i] = particles.radii[i];
      types.span[i] = particles.types[i];
      clusters.span[i] = particles.clusters[i];
    }
    positions.finish();
    velocities.finish();
    ages.finish();
    lifetimes.finish();
    radii.finish();
    types.finish();
    clusters.finish();
  }
  r_geometry.replace_pointcloud(pointcloud);
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
  const float3 corners[8] = {
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
    r_triangles.append(corners[quad[0]]);
    r_triangles.append(corners[quad[1]]);
    r_triangles.append(corners[quad[2]]);
    r_triangles.append(corners[quad[0]]);
    r_triangles.append(corners[quad[2]]);
    r_triangles.append(corners[quad[3]]);
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

/** \} */

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  static NestedBundleTypePtr world_type = make_world_type();

  b.add_input<decl::Bundle>("World"_ustr)
      .bundle_type(world_type)
      .structure_type(StructureType::Single)
      .description(
          "The simulation: `geometry` holds the white water of the previous frame, `fluid` the "
          "fluid particles (with `velocity`), `surface` the fluid surface mesh, `effectors` the "
          "colliders and `settings` the parameters (Blender.WhiteWaterSettings). Leave it empty to "
          "start a new simulation");
  b.add_output<decl::Bundle>("World"_ustr)
      .description("The world with the white water of this frame: feed it back inside the zone");
  b.add_input<decl::Float>("Delta Time"_ustr)
      .default_value(1.0f / 24.0f)
      .min(0.0f)
      .subtype(PROP_TIME_ABSOLUTE)
      .description("Length of the frame in seconds; drive it with `Simulation Input > Delta Time`");
  b.add_output<decl::Bundle>("Stats"_ustr)
      .description("Particle and cluster counts, emission and timing, for debugging");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  BundlePtr world_ptr = params.get_input<BundlePtr>("World"_ustr);
  if (!world_ptr) {
    params.set_default_remaining_outputs();
    return;
  }
  const float delta_time = std::max(0.0f, params.extract_input<float>("Delta Time"_ustr));
  whitewater::Settings settings = read_settings(*world_ptr);
  if (const Scene *scene = DEG_get_input_scene(params.depsgraph())) {
    settings.frame = scene->r.cfra;
  }

  whitewater::ParticleData particles;
  const GeometrySet *state_geometry = world_ptr->lookup_ptr<GeometrySet>(
      *BundleKey::from_str("geometry"));
  if (state_geometry != nullptr) {
    for (const GeometryComponent::Type type : {GeometryComponent::Type::PointCloud,
                                               GeometryComponent::Type::Mesh})
    {
      const GeometryComponent *component = state_geometry->get_component(type);
      if (component != nullptr) {
        read_state(*component, particles);
      }
    }
  }

  Vector<float3> fluid_positions;
  Vector<float3> fluid_velocities;
  if (const GeometrySet *fluid = world_ptr->lookup_ptr<GeometrySet>(
          *BundleKey::from_str("fluid")))
  {
    for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                               GeometryComponent::Type::PointCloud})
    {
      const GeometryComponent *component = fluid->get_component(type);
      if (component != nullptr) {
        read_fluid(*component, fluid_positions, fluid_velocities);
      }
    }
  }

  const Mesh *surface = nullptr;
  if (const GeometrySet *surface_geometry = world_ptr->lookup_ptr<GeometrySet>(
          *BundleKey::from_str("surface")))
  {
    surface = surface_geometry->get_mesh();
  }

  Vector<float3> collider_triangles;
  if (const BundlePtr *effectors = world_ptr->lookup_ptr<BundlePtr>(
          *BundleKey::from_str("effectors")))
  {
    if (*effectors) {
      gather_colliders(**effectors, collider_triangles);
    }
  }

  if (particles.size() == 0 && (surface == nullptr || fluid_positions.is_empty())) {
    /* Nothing to simulate and nothing to emit from: keep the world as it is. */
    params.set_output("World"_ustr, world_ptr);
    params.set_output("Stats"_ustr, stats_to_bundle(whitewater::Stats{}));
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

  Bundle &world = world_ptr.ensure_mutable_inplace();
  GeometrySet result;
  write_particles(particles, result);
  world.add_override(*BundleKey::from_str("geometry"), result);

  params.set_output("World"_ustr, std::move(world_ptr));
  params.set_output("Stats"_ustr, stats_to_bundle(stats));
}

static void node_register()
{
  static bke::bNodeType ntype;

  /* Register the bundle types even when the node is only referenced by an asset: the store nodes of
   * `White Water Dynamics` write them by name. */
  WhiteWaterSettingsBundle::get_bundle_type();
  make_world_type();

  geo_node_type_base(&ntype, "GeometryNodeWhiteWaterSolver"_ustr);
  ntype.ui_name = "White Water Solver";
  ntype.ui_description =
      "Add foam, spray and bubbles to a solved fluid: emission from the surface, ballistic spray, "
      "buoyant bubbles and neighbour cohesion that lets the clumps merge and tear apart";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_180;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_white_water_solver_cc
