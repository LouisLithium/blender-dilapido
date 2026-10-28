/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Init Ignis Data`: turns an emitter (mesh, point cloud) into the grids the smoke solver eats, i.e.
 * the "pyro source" and the "rasterize volume" of a Houdini style Pyro setup in one node.
 *
 * The rasterization happens at the resolution of *this* node, not of the solver, so a dense emitter
 * does not force a dense simulation and a coarse one cannot blur the solver. The result is a
 * `Volume` geometry, which the solver samples at its own voxel centers.
 */

#include "BLI_math_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"

#include "NOD_geometry_nodes_bundle.hh"

#include "node_geometry_util.hh"

#include "GEO_ignis_source.hh"

namespace blender::nodes::node_geo_init_ignis_data_cc {

using geometry::ignis::SourceMode;
using geometry::ignis::SourceSample;
using geometry::ignis::SourceSettings;
using geometry::ignis::SourceValues;

namespace {

static const EnumPropertyItem source_mode_items[] = {
    {int(SourceMode::Points),
     "POINTS",
     0,
     N_("Points"),
     N_("Every point emits inside the radius. Use `Distribute Points in Volume` upstream to fill a "
        "volume with emitters")},
    {int(SourceMode::Surface),
     "SURFACE",
     0,
     N_("Surface"),
     N_("The triangles emit along their whole surface, which is what a burning object looks like: "
        "the skin burns, not the vertices")},
    {int(SourceMode::Solid),
     "SOLID",
     0,
     N_("Solid"),
     N_("The whole inside of a closed mesh emits, so a box becomes a block of smoke")},
    {0, nullptr, 0, nullptr, nullptr},
};

/** Evaluates a field for every point of a component into `r_values`. */
void gather_component(GeometrySet &geometry,
                      const FunctionRef<void(GeometryComponent &)> &fn)
{
  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Curve})
  {
    if (!geometry.has(type)) {
      continue;
    }
    GeometryComponent &component = geometry.get_component_for_write(type);
    if (component.attribute_domain_size(AttrDomain::Point) == 0) {
      continue;
    }
    fn(component);
  }
}

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr)
      .is_default_link_socket()
      .description("The emitter: a mesh, a point cloud or a curve");
  b.add_input<decl::Menu>("Mode"_ustr)
      .static_items(source_mode_items)
      .default_value(MenuValue(int(SourceMode::Points)))
      .description("How the geometry becomes a field");
  b.add_input<decl::Float>("Voxel Size"_ustr)
      .default_value(0.05f)
      .min(1e-4f)
      .subtype(PROP_DISTANCE)
      .description(
          "Resolution of the rasterized source. Independent of the solver: the solver samples the "
          "result at its own voxel centers");
  b.add_input<decl::Float>("Radius"_ustr)
      .default_value(0.05f)
      .min(1e-5f)
      .subtype(PROP_DISTANCE)
      .description("Radius of the emission kernel around a point, and the thickness of a surface");
  b.add_input<decl::Float>("Band Width"_ustr)
      .default_value(0.05f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .usage_by_menu("Mode"_ustr, int(SourceMode::Solid))
      .description("How far the solid fill reaches outwards from the surface");
  b.add_input<decl::Float>("Strength"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .description("Multiplier for all fields, i.e. how strong the emitter is");

  {
    auto &panel = b.add_panel("Fields"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Density"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .structure_type(StructureType::Field)
        .description("Density the emitter holds its region at");
    panel.add_input<decl::Float>("Temperature"_ustr)
        .default_value(0.0f)
        .structure_type(StructureType::Field)
        .description("Temperature of the emitted smoke, which is what the buoyancy pushes up");
    panel.add_input<decl::Float>("Fuel"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .structure_type(StructureType::Field)
        .description("Fuel the emitter injects; burning it needs `Burn Rate` on the solver");
    panel.add_input<decl::Vector>("Velocity"_ustr)
        .default_value(float3(0.0f))
        .subtype(PROP_VELOCITY)
        .structure_type(StructureType::Field)
        .description("Velocity of the emitted smoke, for a jet or a muzzle flash");
  }

  b.add_output<decl::Geometry>("Volume"_ustr)
      .supported_type(GeometryComponent::Type::Volume)
      .description(
          "The rasterized emitter: one volume with the density, temperature, fuel and velocity "
          "grids. Connect it to the `Sources` input of `Ignis Solver`");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  const int mode = params.extract_input<int>("Mode"_ustr);
  const float voxel_size = std::max(1e-4f, params.extract_input<float>("Voxel Size"_ustr));
  const float radius = std::max(1e-5f, params.extract_input<float>("Radius"_ustr));
  const float band_width = std::max(0.0f, params.extract_input<float>("Band Width"_ustr));
  const float strength = std::max(0.0f, params.extract_input<float>("Strength"_ustr));
  const fn::Field<float> density_field = params.extract_input<fn::Field<float>>("Density"_ustr);
  const fn::Field<float> temperature_field =
      params.extract_input<fn::Field<float>>("Temperature"_ustr);
  const fn::Field<float> fuel_field = params.extract_input<fn::Field<float>>("Fuel"_ustr);
  const fn::Field<float3> velocity_field = params.extract_input<fn::Field<float3>>("Velocity"_ustr);

  Vector<SourceSample> samples;
  Vector<int3> corner_tris;
  Vector<int> corner_verts;

  gather_component(geometry, [&](GeometryComponent &component) {
    const int points_num = component.attribute_domain_size(AttrDomain::Point);
    const bke::GeometryFieldContext context(component, AttrDomain::Point);
    fn::FieldEvaluator evaluator(context, points_num);
    evaluator.add(density_field);
    evaluator.add(temperature_field);
    evaluator.add(fuel_field);
    evaluator.add(velocity_field);
    evaluator.evaluate();
    const VArray<float3> positions = *component.attributes()->lookup<float3>("position",
                                                                             AttrDomain::Point);
    const VArray<float> densities = evaluator.get_evaluated<float>(0);
    const VArray<float> temperatures = evaluator.get_evaluated<float>(1);
    const VArray<float> fuels = evaluator.get_evaluated<float>(2);
    const VArray<float3> velocities = evaluator.get_evaluated<float3>(3);

    /* The triangle indices have to be offset by the samples of the previous components. */
    const int offset = samples.size();
    samples.resize(offset + points_num);
    for (const int i : IndexRange(points_num)) {
      SourceSample &sample = samples[offset + i];
      sample.position = positions[i];
      sample.density = densities[i];
      sample.temperature = temperatures[i];
      sample.fuel = fuels[i];
      sample.velocity = velocities[i];
    }

    if (component.type() == GeometryComponent::Type::Mesh) {
      const Mesh &mesh = *static_cast<const MeshComponent &>(component).get();
      const Span<int3> tris = mesh.corner_tris();
      if (!tris.is_empty()) {
        const int tri_offset = corner_tris.size();
        corner_tris.resize(tri_offset + int(tris.size()));
        corner_verts.resize(corner_verts.size() + mesh.corner_verts().size());
        for (const int i : tris.index_range()) {
          corner_tris[tri_offset + i] = tris[i] + int3(offset);
        }
        MutableSpan<int> verts = corner_verts.as_mutable_span().take_back(
            mesh.corner_verts().size());
        verts.copy_from(mesh.corner_verts());
      }
    }
  });

  /* The inside of a solid source is filled with the average of its surface. */
  SourceValues mean_values;
  if (!samples.is_empty()) {
    float3 velocity_sum(0.0f);
    for (const SourceSample &sample : samples) {
      mean_values.density += sample.density;
      mean_values.temperature += sample.temperature;
      mean_values.fuel += sample.fuel;
      velocity_sum += sample.velocity;
    }
    const float inverse = 1.0f / float(samples.size());
    mean_values.density *= inverse;
    mean_values.temperature *= inverse;
    mean_values.fuel *= inverse;
    mean_values.velocity = velocity_sum * inverse;
  }

  SourceSettings settings;
  settings.voxel_size = voxel_size;
  settings.radius = radius;
  settings.mode = SourceMode(mode);
  settings.band_width = band_width;
  settings.scale = strength;

  Volume *volume = geometry::ignis::rasterize_source(
      Span<const SourceSample>(samples.data(), samples.size()),
      Span<const int3>(corner_tris.data(), corner_tris.size()),
      Span<const int>(corner_verts.data(), corner_verts.size()),
      mean_values,
      settings);
  if (volume == nullptr) {
    params.set_default_remaining_outputs();
    return;
  }

  GeometrySet volume_geometry;
  volume_geometry.replace_volume(volume);
  params.set_output("Volume"_ustr, std::move(volume_geometry));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeInitIgnisData"_ustr);
  ntype.ui_name = "Init Ignis Data";
  ntype.ui_description =
      "Rasterize a mesh or point cloud into the density, temperature, fuel and velocity grids that "
      "`Ignis Solver` simulates";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_init_ignis_data_cc
