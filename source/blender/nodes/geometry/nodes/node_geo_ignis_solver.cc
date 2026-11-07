/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Ignis Solver`: the sparse smoke and fire solver.
 *
 * Like the XPBD and rigid body solvers, everything the solver works on comes in through the `World`
 * bundle; the node itself only has `Delta Time` and the output. The world holds
 *
 *  - `geometry`: the state of the previous frame (a volume with one grid per field),
 *  - `source`: the rasterized emitters of this frame (the volumes of `Init Ignis Data`; several of
 *    them are mixed when they are instanced, see #gather_source_volumes),
 *  - `effectors`: the colliders (`Dilapido Collider`) and the force fields (`Dilapido Force`),
 *  - `settings`: the simulation parameters (`Blender.IgnisSettings`), so the asset can expose the
 *    whole panel without every knob being a socket of this node.
 *
 * Only `geometry` is written back, everything else passes through.
 */

#include <algorithm>
#include <optional>

#include "BLI_array.hh"
#include "BLI_math_vector.hh"

#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_volume.hh"

#include "NOD_bundle_type.hh"
#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_lazy_function.hh"
#include "NOD_geometry_nodes_physics_bundles.hh"

#include "node_geometry_util.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "GEO_ha_context.hh"
#include "GEO_ha_status.hh"
#include "GEO_ignis_solver.hh"

namespace blender::nodes::node_geo_ignis_solver_cc {

using geometry::ignis::Stats;
using namespace blender::nodes::physics_bundles;

namespace {

/** Bundle types of `Dilapido Collider`, whose `geometry` is a mesh in world space. */
constexpr StringRef COLLIDER_MESH_TYPE = "Blender.Collider.Mesh";
constexpr StringRef COLLIDER_BOX_TYPE = "Blender.Collider.Box";
/** Bundle type of `Dilapido Force`, one uniform acceleration. */
constexpr StringRef FORCE_TYPE = "Blender.DilapidoForce";

/**
 * The simulation parameters, stored in the world.
 *
 * Keeping them in a bundle is what makes the node itself small: the asset exposes the panel and
 * writes the bundle, and nothing but `World` and `Delta Time` has to be wired into the solver.
 */
class IgnisSettingsBundle {
 public:
  static constexpr StringRefNull name = "Blender.IgnisSettings";
  static const FlatBundleTypePtr &get_bundle_type();
};

const FlatBundleTypePtr &IgnisSettingsBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(IgnisSettingsBundle::name);
    b.add<decl::Float>("voxel_size"_ustr).min(1e-4f);
    b.add<decl::Int>("substeps"_ustr).min(1);
    b.add<decl::Int>("max_substeps"_ustr).min(1);
    b.add<decl::Float>("cfl"_ustr).min(0.0f);
    b.add<decl::Int>("advection"_ustr).min(0).description("Semi-Lagrangian, Midpoint, MacCormack or BFECC");
    b.add<decl::Int>("limiter"_ustr).min(0).description("None, Clamp or Revert");
    b.add<decl::Bool>("use_bounds"_ustr);
    b.add<decl::Vector>("bounds_min"_ustr);
    b.add<decl::Vector>("bounds_max"_ustr);
    b.add<decl::Bool>("use_domain_mask"_ustr);
    b.add<decl::Int>("domain_field"_ustr).min(0).description(
        "0 = Density, 1 = Temperature, 2 = Fuel, 3 = Flame");
    b.add<decl::Float>("domain_threshold"_ustr).min(0.0f);
    b.add<decl::Vector>("gravity"_ustr);
    b.add<decl::Float>("buoyancy"_ustr);
    b.add<decl::Float>("ambient_temperature"_ustr);
    b.add<decl::Float>("vorticity"_ustr).min(0.0f);
    b.add<decl::Float>("drag"_ustr).min(0.0f);
    b.add<decl::Float>("velocity_damping"_ustr).min(0.0f);
    b.add<decl::Float>("cooling"_ustr).min(0.0f);
    b.add<decl::Float>("dissipation"_ustr).min(0.0f);
    b.add<decl::Int>("domain_padding"_ustr).min(0);
    b.add<decl::Int>("extrapolate"_ustr).min(0);
    b.add<decl::Float>("burn_rate"_ustr).min(0.0f);
    b.add<decl::Float>("heat_release"_ustr).min(0.0f);
    b.add<decl::Float>("smoke_per_burn"_ustr).min(0.0f);
    b.add<decl::Float>("flame_threshold"_ustr).min(0.0f);
    b.add<decl::Int>("pressure_iterations"_ustr).min(1);
    b.add<decl::Float>("tolerance"_ustr).min(0.0f);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

/** Bundle type of the `World` socket: what the solver may find in the world. */
static NestedBundleTypePtr make_world_type()
{
  Vector<std::shared_ptr<const FlatBundleType>> types;
  types.append(IgnisSettingsBundle::get_bundle_type());
  types.append(MeshColliderBundle::get_bundle_type());
  types.append(BoxColliderBundle::get_bundle_type());
  /* Registered here so that assets can create them in the same node tree. */
  ForceBundle::get_bundle_type();

  NestedBundleTypePtr world_type = std::make_shared<const NestedBundleType>(
      "Blender.IgnisSolverWorld", std::move(types));
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

float3 read_vector(const Bundle &bundle, const StringRef key, const float3 fallback)
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

/**
 * Collects the volumes of a geometry set, including the ones that are nested in instances.
 *
 * A `GeometrySet` holds at most one `Volume` component, and joining two of them is not implemented
 * in this branch (`join_volumes()` drops them: resampling grids with the same name and picking the
 * cell size of the result is an open question), so `Join Geometry` of two `Init Ignis Data` outputs
 * produces nothing at all. A geometry set does hold any number of instances though, and every
 * instance may contain a geometry set with a volume in it, so several emitters are mixed with
 * `Geometry to Instance` (or any other node that creates instances) and found by this walk.
 *
 * \note The transform of an instance is ignored: `Init Ignis Data` rasterizes its input in the space
 * it is given and a volume carries its own grid transform, so moving an emitter has to happen before
 * it is rasterized (`Transform Geometry` on the mesh upstream, or `Object Info > Transform`).
 */
static void gather_source_volumes(const GeometrySet &geometry,
                                  Vector<const Volume *> &r_volumes,
                                  const int depth)
{
  if (const Volume *volume = geometry.get_volume()) {
    r_volumes.append(volume);
  }
  /* Instances may contain instances; a malformed or deeply nested graph is simply cut off. */
  if (depth >= 8) {
    return;
  }
  const bke::Instances *instances = geometry.get_instances();
  if (instances == nullptr) {
    return;
  }
  /* `references()` is the deduplicated table, so a volume that is instanced several times (or with
   * different transforms) contributes once, which is what the solver wants. */
  for (const bke::InstanceReference &reference : instances->references()) {
    if (reference.type() != bke::InstanceReference::Type::GeometrySet) {
      continue;
    }
    gather_source_volumes(reference.geometry_set(), r_volumes, depth + 1);
  }
}

geometry::ignis::Settings read_settings(const Bundle &world)
{
  geometry::ignis::Settings settings;
  const BundlePtr *bundle_ptr = world.lookup_ptr<BundlePtr>(*BundleKey::from_str("settings"));
  if (bundle_ptr == nullptr || !*bundle_ptr) {
    return settings;
  }
  const Bundle &s = **bundle_ptr;
  settings.voxel_size = std::max(1e-4f, read_float(s, "voxel_size", settings.voxel_size));
  settings.substeps = read_int(s, "substeps", settings.substeps);
  settings.max_substeps = read_int(s, "max_substeps", settings.max_substeps);
  settings.cfl = read_float(s, "cfl", settings.cfl);
  settings.advection_scheme = geometry::ignis::AdvectionScheme(
      std::clamp(read_int(s, "advection", int(settings.advection_scheme)), 0, 3));
  settings.advection_limiter = geometry::ignis::AdvectionLimiter(
      std::clamp(read_int(s, "limiter", int(settings.advection_limiter)), 0, 2));
  settings.use_bounds = read_bool(s, "use_bounds", settings.use_bounds);
  settings.bounds_min = read_vector(s, "bounds_min", settings.bounds_min);
  settings.bounds_max = read_vector(s, "bounds_max", settings.bounds_max);
  settings.use_domain_mask = read_bool(s, "use_domain_mask", settings.use_domain_mask);
  settings.domain_field = geometry::ignis::DomainField(
      std::clamp(read_int(s, "domain_field", int(settings.domain_field)), 0, 3));
  settings.domain_threshold = std::max(read_float(s, "domain_threshold", settings.domain_threshold),
                                       0.0f);
  settings.gravity = read_vector(s, "gravity", settings.gravity);
  settings.buoyancy = read_float(s, "buoyancy", settings.buoyancy);
  settings.ambient_temperature = read_float(s, "ambient_temperature", settings.ambient_temperature);
  settings.vorticity = read_float(s, "vorticity", settings.vorticity);
  settings.drag = read_float(s, "drag", settings.drag);
  settings.velocity_damping = read_float(s, "velocity_damping", settings.velocity_damping);
  settings.cooling = read_float(s, "cooling", settings.cooling);
  settings.dissipation = read_float(s, "dissipation", settings.dissipation);
  settings.domain_padding = read_int(s, "domain_padding", settings.domain_padding);
  settings.extrapolate = read_int(s, "extrapolate", settings.extrapolate);
  settings.burn_rate = read_float(s, "burn_rate", settings.burn_rate);
  settings.heat_release = read_float(s, "heat_release", settings.heat_release);
  settings.smoke_per_burn = read_float(s, "smoke_per_burn", settings.smoke_per_burn);
  settings.flame_threshold = read_float(s, "flame_threshold", settings.flame_threshold);
  settings.pressure_iterations = read_int(s, "pressure_iterations", settings.pressure_iterations);
  /* A loose default on purpose: 0.04 % of the velocity scale is four orders below the smoke's own
   * numerical diffusion, and a tighter solve costs a third of the frame. */
  settings.tolerance = std::max(read_float(s, "tolerance", settings.tolerance), 0.0f);
  return settings;
}

/**
 * Appends the mesh of one collider bundle.
 *
 * `Dilapido Collider` bakes the transform into the geometry, so `mesh` is already the pose of this
 * frame. The `previous` bundle written by `Set Previous World Items` holds the mesh and the
 * transform of the previous frame; with `Sweep` on the solver rasterizes several poses between the
 * two so a fast collider cannot jump over the smoke, and with `Deforming` on it interpolates the
 * vertices instead of the object transform.
 */
void append_collider(const Bundle &bundle, Vector<geometry::ignis::Collider> &r_colliders)
{
  const GeometrySet *geometry = lookup_geometry(bundle, "geometry");
  if (geometry == nullptr) {
    return;
  }
  const Mesh *mesh = geometry->get_mesh();
  if (mesh == nullptr) {
    return;
  }
  geometry::ignis::Collider collider;
  collider.mesh = mesh;
  collider.sweep = read_bool(bundle, "sweep", true);
  collider.deforming = read_bool(bundle, "deforming", false);
  if (const float4x4 *transform = bundle.lookup_ptr<float4x4>(*BundleKey::from_str("transform"))) {
    collider.transform = *transform;
  }
  if (const BundlePtr *previous_ptr = bundle.lookup_ptr<BundlePtr>(
          *BundleKey::from_str("previous")))
  {
    if (*previous_ptr) {
      const Bundle &previous = **previous_ptr;
      if (const GeometrySet *previous_geometry = lookup_geometry(previous, "geometry")) {
        collider.previous_mesh = previous_geometry->get_mesh();
        collider.has_previous = collider.previous_mesh != nullptr;
      }
      if (const float4x4 *previous_transform = previous.lookup_ptr<float4x4>(
              *BundleKey::from_str("transform")))
      {
        collider.previous_transform = *previous_transform;
        collider.has_previous_transform = true;
      }
    }
  }
  r_colliders.append(collider);
}

/** One procedural force component of a `Dilapido Force` bundle. */
struct NoiseForce {
  bool configured = false;
  float strength = 0.0f;
  float scale = 1.0f;
  float speed = 1.0f;
  float lacunarity = 2.0f;
  float roughness = 0.5f;
  int seed = 0;
  int octaves = 1;
  int remap_field = 0;
  int remap_mode = 0;
  float remap_min = 0.0f;
  float remap_max = 1.0f;
  float remap_softness = 0.2f;
  float remap_scale = 1.0f;
};

/** All forces collected from the effector bundle. */
struct ForceFields {
  float3 acceleration = float3(0.0f);
  NoiseForce disturbance;
  NoiseForce turbulence;
  float time = 0.0f;
};

/** Adds the disturbance component of one `Dilapido Force` bundle. */
void append_disturbance(const Bundle &bundle, NoiseForce &r_force)
{
  const float strength = read_float(bundle, "disturbance", 0.0f);
  if (strength == 0.0f) {
    return;
  }
  if (!r_force.configured) {
    r_force.scale = read_float(bundle, "disturbance_scale", 0.5f);
    r_force.speed = read_float(bundle, "disturbance_speed", 1.0f);
    r_force.seed = read_int(bundle, "disturbance_seed", 0);
    r_force.octaves = read_int(bundle, "disturbance_octaves", 2);
    r_force.lacunarity = read_float(bundle, "disturbance_lacunarity", 2.0f);
    r_force.roughness = read_float(bundle, "disturbance_roughness", 0.5f);
    r_force.remap_field = read_int(bundle, "disturbance_remap_field", 0);
    r_force.remap_mode = read_int(bundle, "disturbance_remap_mode", 0);
    r_force.remap_min = read_float(bundle, "disturbance_remap_min", 0.0f);
    r_force.remap_max = read_float(bundle, "disturbance_remap_max", 1.0f);
    r_force.remap_softness = read_float(bundle, "disturbance_remap_softness", 0.2f);
    r_force.remap_scale = read_float(bundle, "disturbance_remap_scale", 1.0f);
    r_force.configured = true;
  }
  r_force.strength += strength;
}

/** Adds the turbulence component of one `Dilapido Force` bundle. */
void append_turbulence(const Bundle &bundle, NoiseForce &r_force)
{
  const float strength = read_float(bundle, "turbulence", 0.0f);
  if (strength == 0.0f) {
    return;
  }
  if (!r_force.configured) {
    r_force.scale = read_float(bundle, "turbulence_scale", 0.25f);
    r_force.speed = read_float(bundle, "turbulence_speed", 1.0f);
    r_force.seed = read_int(bundle, "turbulence_seed", 0);
    r_force.octaves = read_int(bundle, "turbulence_octaves", 3);
    r_force.lacunarity = read_float(bundle, "turbulence_lacunarity", 2.0f);
    r_force.roughness = read_float(bundle, "turbulence_roughness", 0.5f);
    r_force.remap_field = read_int(bundle, "turbulence_remap_field", 0);
    r_force.remap_mode = read_int(bundle, "turbulence_remap_mode", 0);
    r_force.remap_min = read_float(bundle, "turbulence_remap_min", 0.0f);
    r_force.remap_max = read_float(bundle, "turbulence_remap_max", 1.0f);
    r_force.remap_softness = read_float(bundle, "turbulence_remap_softness", 0.2f);
    r_force.remap_scale = read_float(bundle, "turbulence_remap_scale", 1.0f);
    r_force.configured = true;
  }
  r_force.strength += strength;
}

/** Adds the wind, disturbance, turbulence and time of one `Dilapido Force` bundle. */
void append_force(const Bundle &bundle, ForceFields &r_forces)
{
  r_forces.acceleration += read_vector(bundle, "acceleration", float3(0.0f));
  append_disturbance(bundle, r_forces.disturbance);
  append_turbulence(bundle, r_forces.turbulence);
  r_forces.time = read_float(bundle, "time", r_forces.time);
}

/**
 * Collects the colliders and the forces of the effector bundle.
 *
 * The bundle may be one effector or contain several nested with `Combine Bundle`, so both are
 * handled.
 */
void gather_effectors(const Bundle &effectors,
                      Vector<geometry::ignis::Collider> &r_colliders,
                      ForceFields &r_forces)
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
      append_force(bundle, r_forces);
    }
  };

  consume(effectors);
  const Vector<std::string> paths = gather_bundle_paths_by_bundle_type(effectors, is_effector);
  for (const StringRef path : paths) {
    /* `Set Previous World Items` nests a copy of every effector under `previous`; that copy is the
     * pose of the previous frame of a collider already in the list, not another collider. */
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

BundlePtr stats_to_bundle(const Stats &stats)
{
  BundlePtr bundle_ptr = Bundle::create();
  Bundle &bundle = bundle_ptr.ensure_mutable_inplace();
  bundle.add(*BundleKey::from_str("active_voxels"), stats.active_voxels);
  bundle.add(*BundleKey::from_str("sources"), stats.sources);
  bundle.add(*BundleKey::from_str("substeps"), stats.substeps_used);
  bundle.add(*BundleKey::from_str("pressure_iterations"), stats.pressure_iterations);
  bundle.add(*BundleKey::from_str("residual"), stats.residual);
  bundle.add(*BundleKey::from_str("divergence_before"), float(stats.divergence_before));
  bundle.add(*BundleKey::from_str("divergence_after"), float(stats.divergence_after));
  bundle.add(*BundleKey::from_str("solid_voxels"), stats.solid_voxels);
  bundle.add(*BundleKey::from_str("domain_removed"), stats.domain_removed);
  bundle.add(*BundleKey::from_str("cfl_limited"), stats.cfl_limited);
  bundle.add(*BundleKey::from_str("resized"), stats.resized);
  bundle.add(*BundleKey::from_str("milliseconds"), float(stats.milliseconds));
  bundle.add(*BundleKey::from_str("device"), stats.device);
  bundle.add(*BundleKey::from_str("device_passes"), stats.device_passes);
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
          "The simulation: `geometry` is the state of the previous frame (the volume), `source` the "
          "emitter of this frame, `effectors` the colliders and forces, and `settings` the "
          "parameters (Blender.IgnisSettings). Leave it empty to start a new simulation");
  b.add_output<decl::Bundle>("World"_ustr)
      .description("The world with the state of this frame: feed it back inside the zone");
  b.add_input<decl::Float>("Delta Time"_ustr)
      .default_value(1.0f / 24.0f)
      .min(0.0f)
      .subtype(PROP_TIME_ABSOLUTE)
      .description("Length of the frame in seconds; drive it with `Simulation Input > Delta Time`");
  b.add_output<decl::Bundle>("Stats"_ustr)
      .description("Active voxels, substeps, pressure residual and timing, for debugging");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  BundlePtr world_ptr = params.extract_input<BundlePtr>("World"_ustr);
  if (!world_ptr) {
    params.set_default_remaining_outputs();
    return;
  }
  const float delta_time = params.extract_input<float>("Delta Time"_ustr);

  /* `HW Accelerate` zone the solver is evaluated in, if any (0 = outside of a zone). The zone
   * wrapper puts the request into the evaluation data, see `GeoNodesUserData::ha_backend_request`. */
  const GeoNodesUserData &user_data = *static_cast<const GeoNodesUserData *>(params.user_data());
  const int backend_request = user_data.ha_backend_request ? *user_data.ha_backend_request : 0;
  if (std::getenv("BLENDER_IGNIS_HA_DEBUG") != nullptr) {
    fprintf(stderr, "IGNIS_HA node %s: backend_request=%d\n", params.node().name, backend_request);
    fflush(stderr);
  }

  geometry::ignis::Settings settings = read_settings(*world_ptr);

  const Volume *previous = nullptr;
  if (const GeometrySet *geometry = lookup_geometry(*world_ptr, "geometry")) {
    previous = geometry->get_volume();
  }
  Vector<const Volume *> sources;
  if (const GeometrySet *geometry = lookup_geometry(*world_ptr, "source")) {
    gather_source_volumes(*geometry, sources, 0);
    if (sources.is_empty()) {
      /* A world with a `source` item but no volume in it. That is normal on the frames an emitter is
       * switched off (an empty emitter), but it is also what `Join Geometry` of two emitters looks
       * like (that node drops volumes instead of joining them), so the message names the fix. */
      params.error_message_add(
          NodeWarningType::Info,
          TIP_("The source holds no volume. Several emitters are mixed with `Geometry to Instance` "
               "(`Join Geometry` drops volumes: a geometry set holds one)"));
    }
  }

  Vector<geometry::ignis::Collider> colliders;
  ForceFields forces;
  if (const BundlePtr *effectors = world_ptr->lookup_ptr<BundlePtr>(*BundleKey::from_str("effectors")))
  {
    if (*effectors) {
      gather_effectors(**effectors, colliders, forces);
    }
  }
  /* `Dilapido Force` is a uniform acceleration plus optional procedural noise forces. */
  settings.wind += forces.acceleration;
  if (forces.disturbance.configured) {
    settings.disturbance = forces.disturbance.strength;
    settings.disturbance_scale = forces.disturbance.scale;
    settings.disturbance_speed = forces.disturbance.speed;
    settings.disturbance_seed = forces.disturbance.seed;
    settings.disturbance_octaves = forces.disturbance.octaves;
    settings.disturbance_lacunarity = forces.disturbance.lacunarity;
    settings.disturbance_roughness = forces.disturbance.roughness;
    settings.disturbance_remap_field = forces.disturbance.remap_field;
    settings.disturbance_remap_mode = forces.disturbance.remap_mode;
    settings.disturbance_remap_min = forces.disturbance.remap_min;
    settings.disturbance_remap_max = forces.disturbance.remap_max;
    settings.disturbance_remap_softness = forces.disturbance.remap_softness;
    settings.disturbance_remap_scale = forces.disturbance.remap_scale;
  }
  if (forces.turbulence.configured) {
    settings.turbulence = forces.turbulence.strength;
    settings.turbulence_scale = forces.turbulence.scale;
    settings.turbulence_speed = forces.turbulence.speed;
    settings.turbulence_seed = forces.turbulence.seed;
    settings.turbulence_octaves = forces.turbulence.octaves;
    settings.turbulence_lacunarity = forces.turbulence.lacunarity;
    settings.turbulence_roughness = forces.turbulence.roughness;
    settings.turbulence_remap_field = forces.turbulence.remap_field;
    settings.turbulence_remap_mode = forces.turbulence.remap_mode;
    settings.turbulence_remap_min = forces.turbulence.remap_min;
    settings.turbulence_remap_max = forces.turbulence.remap_max;
    settings.turbulence_remap_softness = forces.turbulence.remap_softness;
    settings.turbulence_remap_scale = forces.turbulence.remap_scale;
  }
  settings.force_time = forces.time;

  if (std::getenv("BLENDER_IGNIS_DEBUG") != nullptr) {
    printf("IGNIS_DEBUG exec: previous %s, sources %d, colliders %d, acceleration %.2f %.2f %.2f, "
           "disturbance %.2f, turbulence %.2f, dt %.4f\n",
           previous != nullptr ? "yes" : "no",
           int(sources.size()),
           int(colliders.size()),
           double(forces.acceleration.x),
           double(forces.acceleration.y),
           double(forces.acceleration.z),
           double(settings.disturbance),
           double(settings.turbulence),
           delta_time);
    for (const geometry::ignis::Collider &collider : colliders) {
      printf("IGNIS_DEBUG collider: sweep %d, deforming %d, previous mesh %s, previous "
             "transform %s, velocity %.3f %.3f %.3f\n",
             int(collider.sweep),
             int(collider.deforming),
             collider.has_previous ? "yes" : "no",
             collider.has_previous_transform ? "yes" : "no",
             double(collider.velocity.x),
             double(collider.velocity.y),
             double(collider.velocity.z));
    }
  }

  Stats stats;
  Volume *volume = geometry::ignis::solve_frame(
      previous,
      Span<const Volume *>(sources.data(), sources.size()),
      Span<const geometry::ignis::Collider>(colliders.data(), colliders.size()),
      settings,
      delta_time,
      stats,
      backend_request);

  /* Report what the `HW Accelerate` zone actually did. A silent fallback is what makes a backend
   * hard to trust, so the reason is shown on the node (badge + warning) whenever the user asked for
   * the device. `Force CPU` is an explicit choice and gets no badge. */
  if (backend_request != 0) {
    const bNodeTree &tree = params.node().owner_tree();
    const bNode &node = params.node();
    if (stats.device) {
      geometry::ha::HaNodeStatus status;
      status.state = geometry::ha::HaNodeState::Device;
      status.backend = geometry::ha::ha_gpu_backend_name();
      status.elements = stats.active_voxels;
      geometry::ha::ha_status_set(tree, node, status);
    }
    else if (backend_request == 3) {
      geometry::ha::ha_status_clear(tree, node);
    }
    else {
      geometry::ha::HaNodeStatus status;
      status.state = geometry::ha::HaNodeState::Fallback;
      status.backend = geometry::ha::ha_gpu_backend_name();
      status.reason = stats.device_fallback_reason.empty() ?
                          std::string("the smoke is solved on the CPU") :
                          stats.device_fallback_reason;
      geometry::ha::ha_status_set(tree, node, status);
      if (backend_request == 2) {
        params.error_message_add(
            NodeWarningType::Warning,
            TIP_("HW Accelerate zone (Strict): the smoke is solved on the CPU (") + status.reason +
                ")");
      }
      else {
        params.error_message_add(
            NodeWarningType::Info,
            TIP_("HW Accelerate zone: the smoke is solved on the CPU (") + status.reason + ")");
      }
    }
  }
  else {
    geometry::ha::ha_status_clear(params.node().owner_tree(), params.node());
  }

  /* Only the state changes, everything else in the world is passed through. `add` does nothing when
   * the item already exists, so the state has to overwrite it. */
  BundlePtr world_output = world_ptr;
  Bundle &world = world_output.ensure_mutable_inplace();
  if (volume != nullptr) {
    GeometrySet geometry;
    geometry.replace_volume(volume);
    world.add_override(*BundleKey::from_str("geometry"), geometry);
  }
  params.set_output("World"_ustr, std::move(world_output));
  params.set_output("Stats"_ustr, stats_to_bundle(stats));

  if (stats.cfl_limited) {
    params.error_message_add(
        NodeWarningType::Warning,
        TIP_("Ignis: the CFL condition needs more substeps than `Max Substeps`, the smoke may lag "
             "behind. Lower `Voxel Size`, raise `Max Substeps` or raise the frame rate"));
  }
}

static void node_layout_ex(ui::Layout &layout, bContext * /*C*/, PointerRNA *current_node_ptr)
{
  bNode &node = *static_cast<bNode *>(current_node_ptr->data);
  geometry::ha::HaNodeStatus status;
  if (!geometry::ha::ha_status_get(node.owner_tree(), node, status)) {
    return;
  }
  layout.separator();
  if (status.state == geometry::ha::HaNodeState::Device) {
    layout.label("Device: " + status.backend + " · " + std::to_string(status.elements) + " voxels",
                 ICON_MEMORY);
  }
  else {
    layout.label("CPU: " + status.reason, ICON_INFO);
  }
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeIgnisSolver"_ustr);
  ntype.ui_name = "Ignis Solver";
  ntype.ui_description =
      "Simulate sparse smoke and fire with a grid solver: advection, buoyancy, vorticity and a "
      "pressure projection on an OpenVDB sparse grid";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.draw_buttons_ex = node_layout_ex;
  ntype.default_width = bke::NodeWidth::_180;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_ignis_solver_cc
