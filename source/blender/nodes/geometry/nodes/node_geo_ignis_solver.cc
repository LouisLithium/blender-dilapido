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
 *  - `source`: the rasterized emitter of this frame (the volume of `Init Ignis Data`),
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
#include "BKE_volume.hh"

#include "NOD_bundle_type.hh"
#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_physics_bundles.hh"

#include "node_geometry_util.hh"

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
  settings.tolerance = read_float(s, "tolerance", settings.tolerance);
  return settings;
}

/** Appends the mesh of one collider bundle. `Dilapido Collider` bakes the transform in already. */
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
  r_colliders.append(collider);
}

/** Adds the acceleration of one `Dilapido Force` bundle. */
void append_force(const Bundle &bundle, float3 &r_acceleration)
{
  r_acceleration += read_vector(bundle, "acceleration", float3(0.0f));
}

/**
 * Collects the colliders and the forces of the effector bundle.
 *
 * The bundle may be one effector or contain several nested with `Combine Bundle`, so both are
 * handled.
 */
void gather_effectors(const Bundle &effectors,
                      Vector<geometry::ignis::Collider> &r_colliders,
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
      append_force(bundle, r_acceleration);
    }
  };

  consume(effectors);
  const Vector<std::string> paths = gather_bundle_paths_by_bundle_type(effectors, is_effector);
  for (const StringRef path : paths) {
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
  bundle.add(*BundleKey::from_str("substeps"), stats.substeps_used);
  bundle.add(*BundleKey::from_str("pressure_iterations"), stats.pressure_iterations);
  bundle.add(*BundleKey::from_str("residual"), stats.residual);
  bundle.add(*BundleKey::from_str("divergence_before"), float(stats.divergence_before));
  bundle.add(*BundleKey::from_str("divergence_after"), float(stats.divergence_after));
  bundle.add(*BundleKey::from_str("solid_voxels"), stats.solid_voxels);
  bundle.add(*BundleKey::from_str("cfl_limited"), stats.cfl_limited);
  bundle.add(*BundleKey::from_str("resized"), stats.resized);
  bundle.add(*BundleKey::from_str("milliseconds"), float(stats.milliseconds));
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

  geometry::ignis::Settings settings = read_settings(*world_ptr);

  const Volume *previous = nullptr;
  if (const GeometrySet *geometry = lookup_geometry(*world_ptr, "geometry")) {
    previous = geometry->get_volume();
  }
  const Volume *source = nullptr;
  if (const GeometrySet *geometry = lookup_geometry(*world_ptr, "source")) {
    source = geometry->get_volume();
  }

  Vector<geometry::ignis::Collider> colliders;
  float3 acceleration = float3(0.0f);
  if (const BundlePtr *effectors = world_ptr->lookup_ptr<BundlePtr>(*BundleKey::from_str("effectors")))
  {
    if (*effectors) {
      gather_effectors(**effectors, colliders, acceleration);
    }
  }
  /* `Dilapido Force` is a uniform acceleration; the solver adds it to the wind term. */
  settings.wind += acceleration;

  if (std::getenv("BLENDER_IGNIS_DEBUG") != nullptr) {
    printf("IGNIS_DEBUG exec: previous %s, source %s, colliders %d, acceleration %.2f %.2f %.2f, "
           "dt %.4f\n",
           previous != nullptr ? "yes" : "no",
           source != nullptr ? "yes" : "no",
           int(colliders.size()),
           double(acceleration.x),
           double(acceleration.y),
           double(acceleration.z),
           delta_time);
  }

  Stats stats;
  Volume *volume = geometry::ignis::solve_frame(
      previous,
      source,
      Span<const geometry::ignis::Collider>(colliders.data(), colliders.size()),
      settings,
      delta_time,
      stats);

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
  ntype.default_width = bke::NodeWidth::_180;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_ignis_solver_cc
