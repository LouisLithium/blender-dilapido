/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Textura Solver`: the MPM solver for fluids and sand.
 *
 * Like the XPBD, rigid body and Ignis solvers, the node only has `World`, `Delta Time` and
 * `Stats`; everything it works on is read from the world bundle:
 *
 *  - `geometry`: the particle set (mesh vertices, point cloud points or curve points) with the
 *    attributes written by `Init Textura Data`, i.e. the state of the previous frame,
 *  - `effectors`: the colliders (`Dilapido Collider`) and the uniform forces (`Dilapido Force`),
 *  - `settings`: the simulation parameters (`Blender.TexturaSettings`).
 *
 * Only the particle positions, velocities and deformation gradients are written back into
 * `geometry`. The rest of the world passes through untouched.
 */

#include <algorithm>

#include "BLI_array.hh"
#include "BLI_math_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_curves.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "NOD_bundle_type.hh"
#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_physics_bundles.hh"

#include "node_geometry_util.hh"

#include "GEO_textura_solver.hh"

namespace blender::nodes::node_geo_textura_solver_cc {

using namespace blender::nodes::physics_bundles;

namespace {

constexpr StringRef COLLIDER_MESH_TYPE = "Blender.Collider.Mesh";
constexpr StringRef COLLIDER_BOX_TYPE = "Blender.Collider.Box";
constexpr StringRef FORCE_TYPE = "Blender.DilapidoForce";

/** Simulation parameters stored in the world, so the asset can expose a panel without sockets. */
class TexturaSettingsBundle {
 public:
  static constexpr StringRefNull name = "Blender.TexturaSettings";
  static const FlatBundleTypePtr &get_bundle_type();
};

const FlatBundleTypePtr &TexturaSettingsBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(TexturaSettingsBundle::name);
    b.add<decl::Float>("voxel_size"_ustr).min(0.0f);
    b.add<decl::Int>("substeps"_ustr).min(1);
    b.add<decl::Int>("max_substeps"_ustr).min(1);
    b.add<decl::Float>("cfl"_ustr).min(0.01f);
    b.add<decl::Bool>("use_bounds"_ustr);
    b.add<decl::Vector>("bounds_min"_ustr);
    b.add<decl::Vector>("bounds_max"_ustr);
    b.add<decl::Vector>("gravity"_ustr);
    b.add<decl::Float>("velocity_damping"_ustr).min(0.0f);
    b.add<decl::Float>("boundary_friction"_ustr).min(0.0f);
    b.add<decl::Bool>("two_phase"_ustr);
    b.add<decl::Float>("inter_phase_drag"_ustr).min(0.0f);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

static NestedBundleTypePtr make_world_type()
{
  Vector<std::shared_ptr<const FlatBundleType>> types;
  types.append(TexturaSettingsBundle::get_bundle_type());
  types.append(MeshColliderBundle::get_bundle_type());
  types.append(BoxColliderBundle::get_bundle_type());
  /* Registered here so an asset can create the force bundles in the same node tree. */
  ForceBundle::get_bundle_type();

  NestedBundleTypePtr world_type = std::make_shared<const NestedBundleType>(
      "Blender.TexturaSolverWorld", std::move(types));
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

const GeometrySet *lookup_geometry(const Bundle &bundle, const StringRef key)
{
  return bundle.lookup_ptr<GeometrySet>(*BundleKey::from_str(key));
}

geometry::textura::Settings read_settings(const Bundle &world)
{
  geometry::textura::Settings settings;
  const BundlePtr *bundle_ptr = world.lookup_ptr<BundlePtr>(*BundleKey::from_str("settings"));
  if (bundle_ptr == nullptr || !*bundle_ptr) {
    return settings;
  }
  const Bundle &s = **bundle_ptr;
  settings.voxel_size = std::max(0.0f, read_float(s, "voxel_size", settings.voxel_size));
  settings.min_substeps = std::max(1, read_int(s, "substeps", settings.min_substeps));
  settings.max_substeps = std::max(settings.min_substeps,
                                   read_int(s, "max_substeps", settings.max_substeps));
  settings.cfl = std::max(0.01f, read_float(s, "cfl", settings.cfl));
  settings.use_bounds = read_bool(s, "use_bounds", settings.use_bounds);
  settings.bounds_min = read_vector(s, "bounds_min", settings.bounds_min);
  settings.bounds_max = read_vector(s, "bounds_max", settings.bounds_max);
  settings.gravity = read_vector(s, "gravity", settings.gravity);
  settings.velocity_damping = std::max(0.0f,
                                       read_float(s, "velocity_damping", settings.velocity_damping));
  settings.boundary_friction = std::max(
      0.0f, read_float(s, "boundary_friction", settings.boundary_friction));
  settings.two_phase = read_bool(s, "two_phase", settings.two_phase);
  settings.inter_phase_drag = std::max(
      0.0f, read_float(s, "inter_phase_drag", settings.inter_phase_drag));
  return settings;
}

void append_collider(const Bundle &bundle, Vector<geometry::textura::Collider> &r_colliders)
{
  const GeometrySet *geometry = lookup_geometry(bundle, "geometry");
  if (geometry == nullptr) {
    return;
  }
  const Mesh *mesh = geometry->get_mesh();
  if (mesh == nullptr) {
    return;
  }
  geometry::textura::Collider collider;
  collider.mesh = mesh;
  collider.box = bundle.type().has_value() && *bundle.type() == COLLIDER_BOX_TYPE;
  collider.friction = std::clamp(read_float(bundle, "friction", 0.5f), 0.0f, 1.0f);
  r_colliders.append(collider);
}

/**
 * Collects the colliders and the uniform acceleration of the effector bundle.
 *
 * Procedural noise forces are Ignis specific and are ignored here; the uniform `acceleration` of
 * `Dilapido Force` is all an MPM solver needs to add wind or a custom gravity.
 */
void gather_effectors(const Bundle &effectors,
                      Vector<geometry::textura::Collider> &r_colliders,
                      float3 &r_acceleration)
{
  auto is_effector = [](const StringRef type) {
    return type == COLLIDER_MESH_TYPE || type == COLLIDER_BOX_TYPE || type == FORCE_TYPE;
  };
  auto consume = [&](const Bundle &bundle) {
    const std::optional<StringRef> type = bundle.type();
    if (!type) {
      return;
    }
    if (*type == COLLIDER_MESH_TYPE || *type == COLLIDER_BOX_TYPE) {
      append_collider(bundle, r_colliders);
    }
    else if (*type == FORCE_TYPE) {
      r_acceleration += read_vector(bundle, "acceleration", float3(0.0f));
    }
  };

  consume(effectors);
  const Vector<std::string> paths = gather_bundle_paths_by_bundle_type(effectors, is_effector);
  for (const StringRef path : paths) {
    /* `Set Previous World Items` nests a copy of every effector under `previous`; it is the same
     * collider one frame earlier, not another one. */
    if (path.find("previous") != std::string::npos) {
      continue;
    }
    const BundlePtr *bundle_ptr = effectors.lookup_path_ptr<BundlePtr>(path);
    if (!bundle_ptr || !*bundle_ptr) {
      continue;
    }
    consume(**bundle_ptr);
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Reading geometry
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

struct ComponentRange {
  int geometry_index = 0;
  bke::GeometryComponent::Type type;
  int offset = 0;
  int count = 0;
};

struct GeometryEntry {
  std::string path;
  GeometrySet geometry;
  Vector<ComponentRange> ranges;
};

void read_component(GeometryComponent &component,
                    const int offset,
                    geometry::textura::ParticleData &r_particles)
{
  const AttrDomain domain = AttrDomain::Point;
  const int count = component.attribute_domain_size(domain);
  if (count == 0) {
    return;
  }
  const bke::AttributeAccessor attributes = *component.attributes();
  const VArray<float3> positions = lookup_or_default<float3>(
      attributes, "position", domain, float3(0.0f), count);
  const VArray<float3> velocities = lookup_or_default<float3>(
      attributes, "velocity", domain, float3(0.0f), count);
  const VArray<int> material = lookup_or_default<int>(
      attributes, geometry::textura::attribute_names::type, domain, 0, count);
  const VArray<float> density = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::density, domain, 1000.0f, count);
  const VArray<float> volume = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::volume, domain, 0.0f, count);
  const VArray<float> mass = lookup_or_default<float>(
      attributes, "mass", domain, 0.0f, count);
  const VArray<float> bulk = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::bulk_modulus, domain, 5.0e4f, count);
  const VArray<float> shear = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::shear_modulus, domain, 0.0f, count);
  const VArray<float> lambda = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::lame_lambda, domain, 0.0f, count);
  const VArray<float> viscosity = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::viscosity, domain, 0.2f, count);
  const VArray<float> tension = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::tension, domain, 0.0f, count);
  const VArray<float> surface_tension = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::surface_tension, domain, 0.0f, count);
  const VArray<float> friction = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::friction, domain, 0.0f, count);
  const VArray<float> cohesion = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::cohesion, domain, 0.0f, count);
  const VArray<float> tensile_strength = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::tensile_strength, domain, 0.0f, count);
  const VArray<float> compression_limit = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::compression_limit, domain, 0.05f, count);
  const VArray<float> stretch_limit = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::stretch_limit, domain, 0.01f, count);
  const VArray<float> rest_density = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::rest_density, domain, 0.0f, count);
  const VArray<float> hardening_rate = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::hardening_rate, domain, 0.0f, count);
  const VArray<float> hardening = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::hardening, domain, 0.0f, count);
  const VArray<float> damage = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::damage, domain, 0.0f, count);
  const VArray<float> damage_rate = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::damage_rate, domain, 0.0f, count);
  const VArray<float> damping = lookup_or_default<float>(
      attributes, geometry::textura::attribute_names::damping, domain, 0.1f, count);
  const VArray<float> radius = lookup_or_default<float>(attributes, "radius", domain, 0.05f, count);
  const VArray<float3> f_x = lookup_or_default<float3>(
      attributes, geometry::textura::attribute_names::deformation_x, domain, float3(1, 0, 0), count);
  const VArray<float3> f_y = lookup_or_default<float3>(
      attributes, geometry::textura::attribute_names::deformation_y, domain, float3(0, 1, 0), count);
  const VArray<float3> f_z = lookup_or_default<float3>(
      attributes, geometry::textura::attribute_names::deformation_z, domain, float3(0, 0, 1), count);
  const VArray<float3> fp_x = lookup_or_default<float3>(
      attributes, geometry::textura::attribute_names::plastic_x, domain, float3(1, 0, 0), count);
  const VArray<float3> fp_y = lookup_or_default<float3>(
      attributes, geometry::textura::attribute_names::plastic_y, domain, float3(0, 1, 0), count);
  const VArray<float3> fp_z = lookup_or_default<float3>(
      attributes, geometry::textura::attribute_names::plastic_z, domain, float3(0, 0, 1), count);

  for (const int i : IndexRange(count)) {
    const int p = offset + i;
    r_particles.positions[p] = positions[i];
    r_particles.velocities[p] = velocities[i];
    r_particles.material[p] = material[i];
    r_particles.density[p] = density[i];
    r_particles.volume[p] = volume[i];
    r_particles.mass[p] = mass[i];
    r_particles.bulk_modulus[p] = bulk[i];
    r_particles.shear_modulus[p] = shear[i];
    r_particles.lame_lambda[p] = lambda[i];
    r_particles.viscosity[p] = viscosity[i];
    r_particles.tension[p] = tension[i];
    r_particles.surface_tension[p] = surface_tension[i];
    r_particles.friction[p] = friction[i];
    r_particles.cohesion[p] = cohesion[i];
    r_particles.tensile_strength[p] = tensile_strength[i];
    r_particles.compression_limit[p] = compression_limit[i];
    r_particles.stretch_limit[p] = stretch_limit[i];
    r_particles.rest_density[p] = rest_density[i];
    r_particles.hardening_rate[p] = hardening_rate[i];
    r_particles.hardening[p] = hardening[i];
    r_particles.damage[p] = damage[i];
    r_particles.damage_rate[p] = damage_rate[i];
    r_particles.damping[p] = damping[i];
    r_particles.radius[p] = radius[i];
    r_particles.deformation[p * 9 + 0] = f_x[i].x;
    r_particles.deformation[p * 9 + 1] = f_y[i].x;
    r_particles.deformation[p * 9 + 2] = f_z[i].x;
    r_particles.deformation[p * 9 + 3] = f_x[i].y;
    r_particles.deformation[p * 9 + 4] = f_y[i].y;
    r_particles.deformation[p * 9 + 5] = f_z[i].y;
    r_particles.deformation[p * 9 + 6] = f_x[i].z;
    r_particles.deformation[p * 9 + 7] = f_y[i].z;
    r_particles.deformation[p * 9 + 8] = f_z[i].z;
    r_particles.plastic[p * 9 + 0] = fp_x[i].x;
    r_particles.plastic[p * 9 + 1] = fp_y[i].x;
    r_particles.plastic[p * 9 + 2] = fp_z[i].x;
    r_particles.plastic[p * 9 + 3] = fp_x[i].y;
    r_particles.plastic[p * 9 + 4] = fp_y[i].y;
    r_particles.plastic[p * 9 + 5] = fp_z[i].y;
    r_particles.plastic[p * 9 + 6] = fp_x[i].z;
    r_particles.plastic[p * 9 + 7] = fp_y[i].z;
    r_particles.plastic[p * 9 + 8] = fp_z[i].z;
  }
}

/** Counts the points of a geometry set and records one range per component. */
bool collect_ranges(GeometrySet &geometry,
                    const int geometry_index,
                    Vector<ComponentRange> &r_ranges,
                    int &r_total_points)
{
  bool has_component = false;
  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Curve})
  {
    if (!geometry.has(type)) {
      continue;
    }
    GeometryComponent &component = geometry.get_component_for_write(type);
    const int count = component.attribute_domain_size(AttrDomain::Point);
    if (count == 0) {
      continue;
    }
    ComponentRange range;
    range.geometry_index = geometry_index;
    range.type = type;
    range.offset = r_total_points;
    range.count = count;
    r_ranges.append(range);
    r_total_points += count;
    has_component = true;
  }
  return has_component;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Writing geometry
 * \{ */

void write_component(GeometryComponent &component, const ComponentRange &range,
                     const geometry::textura::ParticleData &particles)
{
  const AttrDomain domain = AttrDomain::Point;
  const int count = range.count;
  if (count != component.attribute_domain_size(domain)) {
    return;
  }
  bke::MutableAttributeAccessor attributes = *component.attributes_for_write();

  bke::SpanAttributeWriter<float3> positions = attributes.lookup_or_add_for_write_span<float3>(
      "position", domain);
  bke::SpanAttributeWriter<float3> velocities = attributes.lookup_or_add_for_write_span<float3>(
      "velocity", domain);
  bke::SpanAttributeWriter<float3> f_x = attributes.lookup_or_add_for_write_span<float3>(
      geometry::textura::attribute_names::deformation_x, domain);
  bke::SpanAttributeWriter<float3> f_y = attributes.lookup_or_add_for_write_span<float3>(
      geometry::textura::attribute_names::deformation_y, domain);
  bke::SpanAttributeWriter<float3> f_z = attributes.lookup_or_add_for_write_span<float3>(
      geometry::textura::attribute_names::deformation_z, domain);
  bke::SpanAttributeWriter<float> damage = attributes.lookup_or_add_for_write_span<float>(
      geometry::textura::attribute_names::damage, domain);
  bool has_snow = false;
  for (const int i : IndexRange(count)) {
    if (particles.material[range.offset + i] == int(geometry::textura::MaterialType::Snow)) {
      has_snow = true;
      break;
    }
  }
  bke::SpanAttributeWriter<float> snow_volume;
  bke::SpanAttributeWriter<float> snow_density;
  bke::SpanAttributeWriter<float> snow_hardening;
  bke::SpanAttributeWriter<float> snow_rest_density;
  bke::SpanAttributeWriter<float3> snow_fp_x;
  bke::SpanAttributeWriter<float3> snow_fp_y;
  bke::SpanAttributeWriter<float3> snow_fp_z;
  if (has_snow) {
    snow_volume = attributes.lookup_or_add_for_write_span<float>(
        geometry::textura::attribute_names::volume, domain);
    snow_density = attributes.lookup_or_add_for_write_span<float>(
        geometry::textura::attribute_names::density, domain);
    snow_hardening = attributes.lookup_or_add_for_write_span<float>(
        geometry::textura::attribute_names::hardening, domain);
    snow_rest_density = attributes.lookup_or_add_for_write_span<float>(
        geometry::textura::attribute_names::rest_density, domain);
    snow_fp_x = attributes.lookup_or_add_for_write_span<float3>(
        geometry::textura::attribute_names::plastic_x, domain);
    snow_fp_y = attributes.lookup_or_add_for_write_span<float3>(
        geometry::textura::attribute_names::plastic_y, domain);
    snow_fp_z = attributes.lookup_or_add_for_write_span<float3>(
        geometry::textura::attribute_names::plastic_z, domain);
  }
  if (!positions || !velocities || !f_x || !f_y || !f_z) {
    if (positions) {
      positions.finish();
    }
    if (velocities) {
      velocities.finish();
    }
    if (f_x) {
      f_x.finish();
    }
    if (f_y) {
      f_y.finish();
    }
    if (f_z) {
      f_z.finish();
    }
    if (damage) {
      damage.finish();
    }
    return;
  }
  for (const int i : IndexRange(count)) {
    const int p = range.offset + i;
    positions.span[i] = particles.positions[p];
    velocities.span[i] = particles.velocities[p];
    f_x.span[i] = float3(particles.deformation[p * 9 + 0],
                         particles.deformation[p * 9 + 3],
                         particles.deformation[p * 9 + 6]);
    f_y.span[i] = float3(particles.deformation[p * 9 + 1],
                         particles.deformation[p * 9 + 4],
                         particles.deformation[p * 9 + 7]);
    f_z.span[i] = float3(particles.deformation[p * 9 + 2],
                         particles.deformation[p * 9 + 5],
                         particles.deformation[p * 9 + 8]);
    if (damage) {
      damage.span[i] = particles.damage[p];
    }
    if (snow_volume) {
      snow_volume.span[i] = particles.volume[p];
    }
    if (snow_density) {
      snow_density.span[i] = particles.density[p];
    }
    if (snow_hardening) {
      snow_hardening.span[i] = particles.hardening[p];
    }
    if (snow_rest_density) {
      snow_rest_density.span[i] = particles.rest_density[p];
    }
    if (snow_fp_x && snow_fp_y && snow_fp_z) {
      snow_fp_x.span[i] = float3(particles.plastic[p * 9 + 0],
                                 particles.plastic[p * 9 + 3],
                                 particles.plastic[p * 9 + 6]);
      snow_fp_y.span[i] = float3(particles.plastic[p * 9 + 1],
                                 particles.plastic[p * 9 + 4],
                                 particles.plastic[p * 9 + 7]);
      snow_fp_z.span[i] = float3(particles.plastic[p * 9 + 2],
                                 particles.plastic[p * 9 + 5],
                                 particles.plastic[p * 9 + 8]);
    }
  }
  positions.finish();
  velocities.finish();
  f_x.finish();
  f_y.finish();
  f_z.finish();
  if (damage) {
    damage.finish();
  }
  if (snow_volume) {
    snow_volume.finish();
  }
  if (snow_density) {
    snow_density.finish();
  }
  if (snow_hardening) {
    snow_hardening.finish();
  }
  if (snow_rest_density) {
    snow_rest_density.finish();
  }
  if (snow_fp_x) {
    snow_fp_x.finish();
  }
  if (snow_fp_y) {
    snow_fp_y.finish();
  }
  if (snow_fp_z) {
    snow_fp_z.finish();
  }
}

void write_geometry(GeometrySet &geometry,
                    const Vector<ComponentRange> &ranges,
                    const geometry::textura::ParticleData &particles)
{
  for (const ComponentRange &range : ranges) {
    if (!geometry.has(range.type)) {
      continue;
    }
    GeometryComponent &component = geometry.get_component_for_write(range.type);
    write_component(component, range, particles);
    if (range.type == GeometryComponent::Type::Mesh) {
      geometry.get_mesh_for_write()->tag_positions_changed();
    }
    else if (range.type == GeometryComponent::Type::PointCloud) {
      geometry.get_pointcloud_for_write()->tag_positions_changed();
    }
    else if (range.type == GeometryComponent::Type::Curve) {
      geometry.get_curves_for_write()->geometry.wrap().tag_positions_changed();
    }
  }
}

/** \} */

BundlePtr stats_to_bundle(const geometry::textura::Stats &stats)
{
  BundlePtr bundle_ptr = Bundle::create();
  Bundle &bundle = bundle_ptr.ensure_mutable_inplace();
  bundle.add(*BundleKey::from_str("particles"), stats.particles);
  bundle.add(*BundleKey::from_str("grid_cells"), stats.grid_cells);
  bundle.add(*BundleKey::from_str("active_cells"), stats.active_cells);
  bundle.add(*BundleKey::from_str("substeps"), stats.substeps);
  bundle.add(*BundleKey::from_str("max_speed"), stats.max_speed);
  bundle.add(*BundleKey::from_str("max_volume_ratio"), stats.max_volume_ratio);
  bundle.add(*BundleKey::from_str("min_volume_ratio"), stats.min_volume_ratio);
  bundle.add(*BundleKey::from_str("grid_x"), stats.grid_dims.x);
  bundle.add(*BundleKey::from_str("grid_y"), stats.grid_dims.y);
  bundle.add(*BundleKey::from_str("grid_z"), stats.grid_dims.z);
  bundle.add(*BundleKey::from_str("cfl_limited"), stats.cfl_limited);
  bundle.add(*BundleKey::from_str("grid_limited"), stats.grid_limited);
  bundle.add(*BundleKey::from_str("collider_sdf_failed"), stats.collider_sdf_failed);
  bundle.add(*BundleKey::from_str("two_phase"), stats.two_phase);
  bundle.add(*BundleKey::from_str("milliseconds"), float(stats.milliseconds));
  bundle.add(*BundleKey::from_str("milliseconds_clear"), float(stats.milliseconds_clear));
  bundle.add(*BundleKey::from_str("milliseconds_particle_to_grid"),
             float(stats.milliseconds_particle_to_grid));
  bundle.add(*BundleKey::from_str("milliseconds_update_grid"),
             float(stats.milliseconds_update_grid));
  bundle.add(*BundleKey::from_str("milliseconds_grid_to_particle"),
             float(stats.milliseconds_grid_to_particle));
  bundle.add(*BundleKey::from_str("milliseconds_advect"), float(stats.milliseconds_advect));
  return bundle_ptr;
}

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  static NestedBundleTypePtr world_type = make_world_type();

  b.add_input<decl::Bundle>("World"_ustr)
      .bundle_type(world_type)
      .structure_type(StructureType::Single)
      .description(
          "The simulation: `geometry` holds the particles of the previous frame (with the "
          "attributes of `Init Textura Data`), `effectors` the colliders and forces, and `settings` "
          "the parameters (Blender.TexturaSettings). Leave it empty to start a new simulation");
  b.add_output<decl::Bundle>("World"_ustr)
      .description("The world with the particle state of this frame: feed it back inside the zone");
  b.add_input<decl::Float>("Delta Time"_ustr)
      .default_value(1.0f / 24.0f)
      .min(0.0f)
      .subtype(PROP_TIME_ABSOLUTE)
      .description("Length of the frame in seconds; drive it with `Simulation Input > Delta Time`");
  b.add_output<decl::Bundle>("Stats"_ustr)
      .description("Particle and grid counts, substeps and timing, for debugging");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  BundlePtr world_ptr = params.get_input<BundlePtr>("World"_ustr);
  if (!world_ptr) {
    params.set_default_remaining_outputs();
    return;
  }
  const float delta_time = std::max(0.0f, params.extract_input<float>("Delta Time"_ustr));
  Bundle &world = world_ptr.ensure_mutable_inplace();

  geometry::textura::Settings settings = read_settings(world);

  Vector<geometry::textura::Collider> colliders;
  float3 force_acceleration = float3(0.0f);
  if (const BundlePtr *effectors = world.lookup_ptr<BundlePtr>(*BundleKey::from_str("effectors")))
  {
    if (*effectors) {
      gather_effectors(**effectors, colliders, force_acceleration);
    }
  }
  settings.gravity += force_acceleration;

  Vector<std::string> paths = gather_bundle_paths_by_data_type(world, SOCK_GEOMETRY);
  std::ranges::sort(paths);

  Vector<GeometryEntry> entries;
  geometry::textura::ParticleData particles;
  int total_points = 0;
  for (const StringRef path : paths) {
    GeometryEntry entry;
    entry.path = path;
    entry.geometry = std::move(*world.lookup_path_for_write_ptr<GeometrySet>(path));
    collect_ranges(entry.geometry, entries.size(), entry.ranges, total_points);
    entries.append(std::move(entry));
  }

  geometry::textura::Stats stats;
  if (total_points > 0) {
    particles.resize(total_points);
    for (GeometryEntry &entry : entries) {
      for (const ComponentRange &range : entry.ranges) {
        GeometryComponent &component = entry.geometry.get_component_for_write(range.type);
        read_component(component, range.offset, particles);
      }
    }
    const Span<const geometry::textura::Collider> collider_span(colliders.data(),
                                                                colliders.size());
    geometry::textura::solve_frame(particles, collider_span, settings, delta_time, stats);
    for (GeometryEntry &entry : entries) {
      write_geometry(entry.geometry, entry.ranges, particles);
    }
  }
  else {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Textura: the world holds no particle geometry. Connect the output of `Init Textura "
             "Data` to the `Geometry` input of `Textura Dynamics`"));
  }

  for (GeometryEntry &entry : entries) {
    world.add_path_override(entry.path, std::move(entry.geometry));
  }

  params.set_output("World"_ustr, std::move(world_ptr));
  params.set_output("Stats"_ustr, stats_to_bundle(stats));

  if (getenv("BLENDER_TEXTURA_PROFILE") != nullptr) {
    fprintf(stderr,
            "TEXTURA_PROFILE particles=%d cells=%d active=%d substeps=%d two_phase=%d "
            "total=%.2f clear=%.2f p2g=%.2f grid=%.2f g2p=%.2f advect=%.2f ms\n",
            stats.particles,
            stats.grid_cells,
            stats.active_cells,
            stats.substeps,
            stats.two_phase ? 1 : 0,
            stats.milliseconds,
            stats.milliseconds_clear,
            stats.milliseconds_particle_to_grid,
            stats.milliseconds_update_grid,
            stats.milliseconds_grid_to_particle,
            stats.milliseconds_advect);
  }

  if (stats.cfl_limited) {
    params.error_message_add(
        NodeWarningType::Warning,
        TIP_("Textura: the CFL condition asks for more substeps than `Max Substeps`; the result may "
             "lag behind. Lower `Voxel Size`, raise `Max Substeps` or raise the frame rate"));
  }
  if (stats.grid_limited) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Textura: the particle domain needs more grid cells than the safety limit; the grid was "
             "coarsened. Raise `Voxel Size` or shrink the simulation with `Use Bounds`"));
  }
  if (stats.collider_sdf_failed) {
    params.error_message_add(
        NodeWarningType::Warning,
        TIP_("Textura: the mesh collider distance field could not be built (OpenVDB support is "
             "required); the mesh colliders are ignored, box colliders still work"));
  }
}

static void node_register()
{
  static bke::bNodeType ntype;

  /* Register the bundle types even when the node is only referenced by an asset: the
   * `Store Bundle Item` nodes of `Textura Dynamics` write them by name. */
  TexturaSettingsBundle::get_bundle_type();
  make_world_type();

  geo_node_type_base(&ntype, "GeometryNodeTexturaSolver"_ustr);
  ntype.ui_name = "Textura Solver";
  ntype.ui_description =
      "Simulate fluids and sand with an MPM solver: particle to grid transfer, grid update, APIC "
      "grid to particle and Drucker-Prager plasticity";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_180;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_textura_solver_cc
