/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Ignis Solver`: the sparse smoke and fire solver. The state is a `Volume` (one grid per field), so
 * it can live in the cache of a simulation zone like any other geometry; everything else is passed
 * in and out every frame, which keeps the node stateless and makes it possible to scrub the
 * timeline.
 */

#include "BLI_array.hh"
#include "BLI_math_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_volume.hh"

#include "NOD_geometry_nodes_bundle.hh"

#include "node_geometry_util.hh"

#include "GEO_ignis_solver.hh"

namespace blender::nodes::node_geo_ignis_solver_cc {

using geometry::ignis::SourceData;
using geometry::ignis::Stats;

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

/** The emitters of one component, in the layout the solver wants. */
struct GatheredSource {
  Vector<float3> positions;
  Vector<float3> velocity;
  Vector<float> density;
  Vector<float> temperature;
  Vector<float> fuel;
  Vector<float> radius;
  bool sample_surface = false;
  Vector<int3> corner_tris;
  Vector<int> corner_verts;
};

void gather_component(const GeometryComponent &component, Vector<GatheredSource> &r_sources)
{
  const int points_num = component.attribute_domain_size(AttrDomain::Point);
  if (points_num == 0) {
    return;
  }
  const std::optional<bke::AttributeAccessor> attributes = component.attributes();
  if (!attributes) {
    return;
  }
  const VArray<float3> positions = *attributes->lookup<float3>("position", AttrDomain::Point);
  if (!positions) {
    return;
  }
  const VArray<float3> velocity = attributes
                                      ->lookup_or_default<float3>(geometry::ignis::attributes::velocity,
                                                                  AttrDomain::Point,
                                                                  float3(0.0f))
                                      .varray;
  const VArray<float> density = attributes
                                    ->lookup_or_default<float>(geometry::ignis::attributes::density,
                                                               AttrDomain::Point,
                                                               0.0f)
                                    .varray;
  const VArray<float> temperature =
      attributes
          ->lookup_or_default<float>(geometry::ignis::attributes::temperature, AttrDomain::Point, 0.0f)
          .varray;
  const VArray<float> fuel = attributes
                                 ->lookup_or_default<float>(geometry::ignis::attributes::fuel,
                                                            AttrDomain::Point,
                                                            0.0f)
                                 .varray;
  const VArray<float> radius = attributes
                                   ->lookup_or_default<float>(geometry::ignis::attributes::radius,
                                                              AttrDomain::Point,
                                                              0.0f)
                                   .varray;
  const VArray<bool> surface = attributes
                                   ->lookup_or_default<bool>(geometry::ignis::attributes::surface,
                                                             AttrDomain::Point,
                                                             false)
                                   .varray;
  r_sources.append(GatheredSource());
  GatheredSource &source = r_sources.last();
  source.positions.reinitialize(points_num);
  positions.materialize(source.positions.as_mutable_span());
  source.velocity.reinitialize(points_num);
  velocity.materialize(source.velocity.as_mutable_span());
  source.density.reinitialize(points_num);
  density.materialize(source.density.as_mutable_span());
  source.temperature.reinitialize(points_num);
  temperature.materialize(source.temperature.as_mutable_span());
  source.fuel.reinitialize(points_num);
  fuel.materialize(source.fuel.as_mutable_span());
  source.radius.reinitialize(points_num);
  radius.materialize(source.radius.as_mutable_span());

  /* A mesh in `Surface` mode emits from its triangles instead of its vertices. */
  if (surface[0] && component.type() == GeometryComponent::Type::Mesh) {
    const Mesh &mesh = *static_cast<const MeshComponent &>(component).get();
    const Span<int3> corner_tris = mesh.corner_tris();
    if (!corner_tris.is_empty()) {
      source.sample_surface = true;
      source.corner_tris.reinitialize(corner_tris.size());
      source.corner_tris.as_mutable_span().copy_from(corner_tris);
      source.corner_verts.reinitialize(mesh.corner_verts().size());
      source.corner_verts.as_mutable_span().copy_from(mesh.corner_verts());
    }
  }
}

void gather_sources(const GeometrySet &geometry, Vector<GatheredSource> &r_sources)
{
  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Curve})
  {
    const GeometryComponent *component = geometry.get_component(type);
    if (component != nullptr) {
      gather_component(*component, r_sources);
    }
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
  bundle.add(*BundleKey::from_str("cfl_limited"), stats.cfl_limited);
  bundle.add(*BundleKey::from_str("resized"), stats.resized);
  bundle.add(*BundleKey::from_str("milliseconds"), float(stats.milliseconds));
  return bundle_ptr;
}

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("State"_ustr)
      .supported_type(GeometryComponent::Type::Volume)
      .description(
          "The state of the previous frame (one grid per field). Leave it unconnected to start a "
          "new simulation; the output has to be fed back into the next frame inside a simulation "
          "zone");
  b.add_input<decl::Geometry>("Sources"_ustr)
      .description("Geometry marked with `Ignis Source`: the emitters of this frame");
  b.add_input<decl::Float>("Delta Time"_ustr)
      .default_value(1.0f / 24.0f)
      .min(0.0f)
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
        .subtype(PROP_ACCELERATION);
    panel.add_input<decl::Vector>("Wind"_ustr)
        .default_value(float3(0.0f))
        .subtype(PROP_ACCELERATION)
        .description("Uniform acceleration that pushes the smoke sideways");
    panel.add_input<decl::Float>("Buoyancy"_ustr)
        .default_value(2.0f)
        .min(0.0f)
        .description("Upward acceleration per unit of temperature");
    panel.add_input<decl::Float>("Temperature Scale"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .description("Scales the temperature before it is used for the buoyancy");
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
        .description("Exponential decay of the temperature, in 1/s");
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
        .description("Temperature added per unit of burned fuel");
    panel.add_input<decl::Float>("Smoke per Burn"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .description("Density added per unit of burned fuel");
    panel.add_input<decl::Float>("Flame Threshold"_ustr)
        .default_value(0.05f)
        .min(0.0f)
        .description("Temperature above which the render only `flame` grid is filled");
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
  {
    auto &panel = b.add_panel("Output"_ustr).default_closed(false);
    panel.add_input<decl::Bool>("Surface"_ustr)
        .default_value(true)
        .description("Extract the density isosurface every frame");
    panel.add_input<decl::Float>("Iso Level"_ustr)
        .default_value(0.5f)
        .min(0.01f)
        .usage_by_bool("Surface"_ustr, true);
    panel.add_input<decl::Float>("Adaptivity"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_bool("Surface"_ustr, true);
  }

  b.add_output<decl::Geometry>("State"_ustr)
      .description("Feed it back into `State` inside the simulation zone");
  b.add_output<decl::Geometry>("Surface"_ustr)
      .description("The density isosurface, or an empty geometry when `Surface` is off");
  b.add_output<decl::Bundle>("Stats"_ustr)
      .description("Active voxels, substeps, pressure residual and timing, for debugging");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const GeometrySet state_geometry = params.extract_input<GeometrySet>("State"_ustr);
  const GeometrySet sources_geometry = params.extract_input<GeometrySet>("Sources"_ustr);
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
  settings.temperature_scale = params.extract_input<float>("Temperature Scale"_ustr);
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
  settings.make_surface = params.extract_input<bool>("Surface"_ustr);
  settings.iso_level = params.extract_input<float>("Iso Level"_ustr);
  settings.adaptivity = params.extract_input<float>("Adaptivity"_ustr);

  Vector<GatheredSource> gathered;
  gather_sources(sources_geometry, gathered);

  Array<SourceData> sources(gathered.size());
  for (const int i : gathered.index_range()) {
    const GatheredSource &source = gathered[i];
    SourceData &data = sources[i];
    data.positions = source.positions;
    data.velocity = source.velocity;
    data.density = source.density;
    data.temperature = source.temperature;
    data.fuel = source.fuel;
    data.radius = source.radius;
    data.sample_surface = source.sample_surface;
    data.corner_tris = source.corner_tris;
    data.corner_verts = source.corner_verts;
  }

  if (std::getenv("BLENDER_IGNIS_DEBUG") != nullptr) {
    printf("IGNIS_DEBUG exec: sources %d, dt %.4f, state volume %s\n",
           int(sources.size()),
           delta_time,
           state_geometry.get_volume() ? "yes" : "no");
  }
  Stats stats;
  Mesh *surface = nullptr;
  Volume *volume = geometry::ignis::solve_frame(
      state_geometry.get_volume(),
      Span<const SourceData>(sources.data(), sources.size()),
      settings,
      delta_time,
      &surface,
      stats);

  GeometrySet state_output;
  if (volume != nullptr) {
    state_output.replace_volume(volume);
  }
  params.set_output("State"_ustr, std::move(state_output));

  GeometrySet surface_output;
  if (surface != nullptr) {
    surface_output.replace_mesh(surface);
  }
  params.set_output("Surface"_ustr, std::move(surface_output));

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
