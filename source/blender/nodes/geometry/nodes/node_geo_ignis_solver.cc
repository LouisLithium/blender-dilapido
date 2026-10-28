/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Ignis Solver`: the sparse smoke and fire solver.
 *
 * Like the XPBD and rigid body solvers, the whole state of the simulation travels in a `World`
 * bundle (the volume lives in its `geometry` item), and everything that acts on the simulation from
 * the outside is a separate bundle (`Effectors`: the colliders of `Dilapido Collider`). The node is
 * stateless apart from that bundle, so a simulation zone can keep it in its cache and the timeline
 * can be scrubbed.
 */

#include <algorithm>
#include <optional>

#include "BLI_array.hh"
#include "BLI_math_vector.hh"

#include "BKE_attribute.hh"
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

/** Same integrators as the official `Grid Advect` node, which uses `openvdb::tools::VolumeAdvection`. */
enum class AdvectionScheme : int {
  SemiLagrangian = 0,
  Midpoint = 1,
  MacCormack = 2,
  Bfecc = 3,
};

enum class AdvectionLimiter : int {
  None = 0,
  Clamp = 1,
  Revert = 2,
};

static const EnumPropertyItem advection_scheme_items[] = {
    {int(AdvectionScheme::SemiLagrangian),
     "SEMI",
     0,
     N_("Semi-Lagrangian"),
     N_("First order, cheapest and most diffusive")},
    {int(AdvectionScheme::Midpoint),
     "MID",
     0,
     N_("Midpoint"),
     N_("Second order midpoint integration: a good compromise for most cases")},
    {int(AdvectionScheme::MacCormack),
     "MAC",
     0,
     N_("MacCormack"),
     N_("Second order MacCormack with a limiter: much less dissipation than Semi-Lagrangian, so "
        "the smoke keeps its shape and its detail")},
    {int(AdvectionScheme::Bfecc),
     "BFECC",
     0,
     N_("BFECC"),
     N_("Back and forth error compensation and correction: the least dissipative scheme, and the "
        "most expensive")},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem advection_limiter_items[] = {
    {int(AdvectionLimiter::None),
     "NONE",
     0,
     N_("None"),
     N_("No limiting: fastest, but the higher order schemes can overshoot")},
    {int(AdvectionLimiter::Clamp),
     "CLAMP",
     0,
     N_("Clamp"),
     N_("Clamp the result to the range of the neighbourhood, which removes the overshoot")},
    {int(AdvectionLimiter::Revert),
     "REVERT",
     0,
     N_("Revert"),
     N_("Fall back to first order where clamping would be needed")},
    {0, nullptr, 0, nullptr, nullptr},
};

/** Bundle types of `Dilapido Collider`, whose `geometry` is a mesh in world space. */
constexpr StringRef COLLIDER_MESH_TYPE = "Blender.Collider.Mesh";
constexpr StringRef COLLIDER_BOX_TYPE = "Blender.Collider.Box";

/**
 * The bundle type of the `World` socket.
 *
 * The state itself is a plain volume in the `geometry` item; the types listed here are the nested
 * bundles the solver understands, i.e. the effectors it can share with the other solvers.
 */
static NestedBundleTypePtr make_world_type()
{
  Vector<std::shared_ptr<const FlatBundleType>> types;
  types.append(MeshColliderBundle::get_bundle_type());
  types.append(BoxColliderBundle::get_bundle_type());
  /* Registered here so that assets can create them in the same node tree. */
  ForceBundle::get_bundle_type();

  NestedBundleTypePtr world_type = std::make_shared<const NestedBundleType>(
      "Blender.IgnisSolverWorld", std::move(types));
  BundleTypeRegistry::register_type(world_type);
  return world_type;
}

/** Appends the mesh of one collider bundle. `Dilapido Collider` bakes the transform in already. */
void append_collider(const Bundle &bundle, Vector<geometry::ignis::Collider> &r_colliders)
{
  const GeometrySet *geometry = bundle.lookup_ptr<GeometrySet>(*BundleKey::from_str("geometry"));
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

/**
 * Collects the colliders of the effector bundle.
 *
 * The bundle may be one collider or contain several nested with `Combine Bundle`, so both are
 * handled.
 */
void gather_colliders(const Bundle &effectors, Vector<geometry::ignis::Collider> &r_colliders)
{
  auto is_collider = [](const StringRef type) {
    return type == COLLIDER_MESH_TYPE || type == COLLIDER_BOX_TYPE;
  };
  if (const std::optional<StringRef> type = effectors.type()) {
    if (is_collider(*type)) {
      append_collider(effectors, r_colliders);
      return;
    }
  }
  const Vector<std::string> paths = gather_bundle_paths_by_bundle_type(effectors, is_collider);
  for (const StringRef path : paths) {
    const BundlePtr *bundle_ptr = effectors.lookup_path_ptr<BundlePtr>(path);
    if (!bundle_ptr || !*bundle_ptr) {
      continue;
    }
    append_collider(**bundle_ptr, r_colliders);
  }
}

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
          "The state of the simulation, carried in the `geometry` item as a volume with the "
          "density, temperature, fuel, flame and velocity grids. Leave it empty to start a new "
          "simulation");
  b.add_output<decl::Bundle>("World"_ustr)
      .description("The state of this frame: feed it back into `World` inside the simulation zone");

  b.add_input<decl::Geometry>("Sources"_ustr)
      .supported_type(GeometryComponent::Type::Volume)
      .description(
          "The emitter of this frame: the `Volume` of `Init Ignis Data`. It is sampled at the "
          "voxel centers of the state, so the source may use its own resolution");
  b.add_input<decl::Bundle>("Effectors"_ustr)
      .description(
          "The colliders of the frame: the `Collider` bundles of `Dilapido Collider`, nested with "
          "`Combine Bundle`. Their surfaces become walls about one voxel thick, so a ground plane "
          "and a closed obstacle both work; a collider that moves more than a voxel per frame can "
          "be jumped over");
  b.add_input<decl::Float>("Delta Time"_ustr)
      .default_value(1.0f / 24.0f)
      .min(0.0f)
      .subtype(PROP_TIME_ABSOLUTE)
      .description("Length of the frame in seconds; drive it with `Simulation Input > Delta Time`");

  {
    auto &panel = b.add_panel("Domain"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Voxel Size"_ustr)
        .default_value(0.05f)
        .min(1e-4f)
        .subtype(PROP_DISTANCE)
        .description("Size of one voxel in metres. Changing it restarts the simulation");
    panel.add_input<decl::Bool>("Use Bounds"_ustr)
        .default_value(false)
        .description("Clip the smoke to a box, otherwise it only stops where the sources stop");
    panel.add_input<decl::Vector>("Bounds Min"_ustr)
        .default_value(float3(-1.0f, -1.0f, 0.0f))
        .subtype(PROP_DISTANCE)
        .usage_by_bool("Use Bounds"_ustr, true);
    panel.add_input<decl::Vector>("Bounds Max"_ustr)
        .default_value(float3(1.0f, 1.0f, 3.0f))
        .subtype(PROP_DISTANCE)
        .usage_by_bool("Use Bounds"_ustr, true);
    panel.add_input<decl::Int>("Substeps"_ustr)
        .default_value(1)
        .min(1)
        .max(64)
        .description("Minimum number of substeps per frame");
    panel.add_input<decl::Int>("Max Substeps"_ustr)
        .default_value(16)
        .min(1)
        .max(256)
        .description(
            "Upper limit for the substeps that the CFL condition asks for. The node warns when it "
            "is reached");
    panel.add_input<decl::Float>("CFL"_ustr)
        .default_value(2.0f)
        .min(0.1f)
        .max(20.0f)
        .description("How many voxels a particle may travel per substep");
    panel.add_input<decl::Menu>("Advection"_ustr)
        .static_items(advection_scheme_items)
        .default_value(MenuValue(int(AdvectionScheme::MacCormack)))
        .description("Integrator of the advection step (the same set as `Grid Advect`)");
    panel.add_input<decl::Menu>("Limiter"_ustr)
        .static_items(advection_limiter_items)
        .default_value(MenuValue(int(AdvectionLimiter::Clamp)))
        .description("What the higher order schemes do where they would overshoot");
  }
  {
    auto &panel = b.add_panel("Physics"_ustr).default_closed(false);
    panel.add_input<decl::Vector>("Gravity"_ustr)
        .default_value(float3(0.0f, 0.0f, -9.81f))
        .subtype(PROP_ACCELERATION)
        .description(
            "Direction and strength of gravity. It is what the buoyancy is measured against: the "
            "ambient air is in hydrostatic equilibrium, so gravity is not applied to the smoke "
            "directly, it only scales the buoyancy");
    panel.add_input<decl::Vector>("Wind"_ustr)
        .default_value(float3(0.0f))
        .subtype(PROP_ACCELERATION)
        .description("Uniform acceleration that pushes the smoke sideways");
    panel.add_input<decl::Float>("Buoyancy"_ustr)
        .default_value(1.0f)
        .description(
            "Acceleration per unit of temperature above the ambient, in units of gravity. 1 means "
            "one unit of temperature accelerates the smoke by `g`, so anything above the ambient "
            "temperature rises and anything below it sinks");
    panel.add_input<decl::Float>("Ambient Temperature"_ustr)
        .default_value(0.0f)
        .description(
            "Temperature of the surrounding air. The temperature field is measured against it, so "
            "at 0 the source temperatures are anomalies: 1 rises, -1 sinks. Set it to 293.15 to "
            "work in kelvin (and `Buoyancy` to 9.81 / 293.15 = 0.033 to get the physical "
            "acceleration of that temperature difference)");
    panel.add_input<decl::Float>("Vorticity"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description("Vorticity confinement: puts the small curls back that advection smooths away");
    panel.add_input<decl::Float>("Drag"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description("Velocity damping relative to the fluid, in 1/s");
    panel.add_input<decl::Float>("Velocity Damping"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description("Absolute velocity damping, in 1/s");
    panel.add_input<decl::Float>("Cooling"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description("Exponential decay of the temperature anomaly, in 1/s. It pulls the smoke back "
                     "to the ambient temperature");
    panel.add_input<decl::Float>("Dissipation"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description("Exponential decay of the density, in 1/s");
    panel.add_input<decl::Int>("Domain Padding"_ustr)
        .default_value(16)
        .min(2)
        .max(64)
        .description(
            "Voxels of air around the smoke that take part in the pressure solve. The rising plume "
            "needs room for the air that flows back down, otherwise the pressure drags it back down");
    panel.add_input<decl::Int>("Extrapolate"_ustr)
        .default_value(4)
        .min(0)
        .max(32)
        .description(
            "How many voxels of empty space around the smoke get a velocity, so the smoke keeps "
            "moving into new space instead of stalling at the edge of the active region");
  }
  {
    auto &panel = b.add_panel("Combustion"_ustr).default_closed(true);
    panel.add_input<decl::Float>("Burn Rate"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description("Fuel that burns per second; 0 turns combustion off");
    panel.add_input<decl::Float>("Heat Release"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description("Temperature anomaly added per unit of burned fuel");
    panel.add_input<decl::Float>("Smoke per Burn"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .description("Density added per unit of burned fuel");
    panel.add_input<decl::Float>("Flame Threshold"_ustr)
        .default_value(0.05f)
        .min(0.0f)
        .description(
            "The `flame` grid holds the rate at which the fuel burns, in K/s. Reactions weaker than "
            "this are dropped, so the smouldering tail of the fire does not glow");
  }
  {
    auto &panel = b.add_panel("Projection"_ustr).default_closed(true);
    panel.add_input<decl::Int>("Pressure Iterations"_ustr)
        .default_value(40)
        .min(1)
        .max(1000)
        .description("Maximum conjugate gradient iterations for the pressure solve");
    panel.add_input<decl::Float>("Tolerance"_ustr)
        .default_value(1e-4f)
        .min(1e-8f)
        .description("Relative residual at which the pressure solve stops");
  }

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
  const GeometrySet sources = params.extract_input<GeometrySet>("Sources"_ustr);
  const Volume *source = sources.get_volume();
  const float delta_time = params.extract_input<float>("Delta Time"_ustr);

  geometry::ignis::Settings settings;
  settings.voxel_size = std::max(1e-4f, params.extract_input<float>("Voxel Size"_ustr));
  settings.use_bounds = params.extract_input<bool>("Use Bounds"_ustr);
  settings.bounds_min = params.extract_input<float3>("Bounds Min"_ustr);
  settings.bounds_max = params.extract_input<float3>("Bounds Max"_ustr);
  settings.substeps = params.extract_input<int>("Substeps"_ustr);
  settings.max_substeps = params.extract_input<int>("Max Substeps"_ustr);
  settings.cfl = params.extract_input<float>("CFL"_ustr);
  settings.gravity = params.extract_input<float3>("Gravity"_ustr);
  settings.wind = params.extract_input<float3>("Wind"_ustr);
  settings.buoyancy = params.extract_input<float>("Buoyancy"_ustr);
  settings.ambient_temperature = params.extract_input<float>("Ambient Temperature"_ustr);
  settings.vorticity = params.extract_input<float>("Vorticity"_ustr);
  settings.drag = params.extract_input<float>("Drag"_ustr);
  settings.velocity_damping = params.extract_input<float>("Velocity Damping"_ustr);
  settings.cooling = params.extract_input<float>("Cooling"_ustr);
  settings.dissipation = params.extract_input<float>("Dissipation"_ustr);
  settings.extrapolate = params.extract_input<int>("Extrapolate"_ustr);
  settings.domain_padding = params.extract_input<int>("Domain Padding"_ustr);
  settings.advection_scheme = geometry::ignis::AdvectionScheme(
      params.extract_input<int>("Advection"_ustr));
  settings.advection_limiter = geometry::ignis::AdvectionLimiter(
      params.extract_input<int>("Limiter"_ustr));
  settings.burn_rate = params.extract_input<float>("Burn Rate"_ustr);
  settings.heat_release = params.extract_input<float>("Heat Release"_ustr);
  settings.smoke_per_burn = params.extract_input<float>("Smoke per Burn"_ustr);
  settings.flame_threshold = params.extract_input<float>("Flame Threshold"_ustr);
  settings.pressure_iterations = params.extract_input<int>("Pressure Iterations"_ustr);
  settings.tolerance = params.extract_input<float>("Tolerance"_ustr);

  Vector<geometry::ignis::Collider> colliders;
  if (const BundlePtr effectors = params.extract_input<BundlePtr>("Effectors"_ustr)) {
    gather_colliders(*effectors, colliders);
  }

  /* The state of the previous frame is the volume in the `geometry` item of the world. */
  const Volume *previous = nullptr;
  if (const GeometrySet *geometry = world_ptr->lookup_ptr<GeometrySet>(
          *BundleKey::from_str("geometry")))
  {
    previous = geometry->get_volume();
  }

  if (std::getenv("BLENDER_IGNIS_DEBUG") != nullptr) {
    printf("IGNIS_DEBUG exec: previous %s, source %s, colliders %d, dt %.4f\n",
           previous != nullptr ? "yes" : "no",
           source != nullptr ? "yes" : "no",
           int(colliders.size()),
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
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_ignis_solver_cc
