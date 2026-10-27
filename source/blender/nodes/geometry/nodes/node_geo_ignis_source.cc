/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Ignis Source`: marks geometry as an emitter of the smoke and fire solver and stores its
 * parameters as attributes, exactly like `Init Puncliga Data` does for the XPBD solver. Keeping the
 * parameters on the geometry (instead of on the solver) is what allows several emitters with
 * different settings in one simulation, and it means `Join Geometry` is all that is needed to
 * combine them.
 */

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"

#include "node_geometry_util.hh"

#include "GEO_ignis_solver.hh"

namespace blender::nodes::node_geo_ignis_source_cc {

enum class SourceMode : int {
  Points = 0,
  Surface = 1,
};

static const EnumPropertyItem source_mode_items[] = {
    {int(SourceMode::Points),
     "POINTS",
     0,
     N_("Points"),
     N_("Every point of the geometry emits inside its radius. Use `Distribute Points in Volume` "
        "upstream to fill a volume with emitters")},
    {int(SourceMode::Surface),
     "SURFACE",
     0,
     N_("Surface"),
     N_("The triangles of a mesh emit along their whole surface, so a burning object emits from "
        "its skin instead of from its vertices")},
    {0, nullptr, 0, nullptr, nullptr},
};

namespace {

void write_source_attributes(GeometrySet &geometry,
                             const fn::Field<float> &density,
                             const fn::Field<float> &temperature,
                             const fn::Field<float> &fuel,
                             const fn::Field<float3> &velocity,
                             const float radius,
                             const bool surface)
{
  using namespace geometry::ignis::attributes;
  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Curve})
  {
    if (!geometry.has(type)) {
      continue;
    }
    GeometryComponent &component = geometry.get_component_for_write(type);
    const int points_num = component.attribute_domain_size(AttrDomain::Point);
    if (points_num == 0) {
      continue;
    }
    const bke::GeometryFieldContext context(component, AttrDomain::Point);
    fn::FieldEvaluator evaluator(context, points_num);
    evaluator.add(density);
    evaluator.add(temperature);
    evaluator.add(fuel);
    evaluator.add(velocity);
    evaluator.evaluate();
    const VArray<float> density_values = evaluator.get_evaluated<float>(0);
    const VArray<float> temperature_values = evaluator.get_evaluated<float>(1);
    const VArray<float> fuel_values = evaluator.get_evaluated<float>(2);
    const VArray<float3> velocity_values = evaluator.get_evaluated<float3>(3);

    std::optional<bke::MutableAttributeAccessor> attributes = component.attributes_for_write();
    if (!attributes) {
      continue;
    }
    auto write_float = [&](const StringRef name, const VArray<float> &values) {
      bke::SpanAttributeWriter<float> writer = attributes->lookup_or_add_for_write_only_span<float>(
          name, AttrDomain::Point);
      if (!writer) {
        return;
      }
      values.materialize(writer.span);
      writer.finish();
    };
    write_float(geometry::ignis::attributes::density, density_values);
    write_float(geometry::ignis::attributes::temperature, temperature_values);
    write_float(geometry::ignis::attributes::fuel, fuel_values);
    {
      bke::SpanAttributeWriter<float3> writer = attributes->lookup_or_add_for_write_only_span<float3>(
          geometry::ignis::attributes::velocity, AttrDomain::Point);
      if (writer) {
        velocity_values.materialize(writer.span);
        writer.finish();
      }
    }
    {
      bke::SpanAttributeWriter<float> writer = attributes->lookup_or_add_for_write_only_span<float>(
          geometry::ignis::attributes::radius, AttrDomain::Point);
      if (writer) {
        writer.span.fill(radius);
        writer.finish();
      }
    }
    {
      bke::SpanAttributeWriter<bool> writer = attributes->lookup_or_add_for_write_only_span<bool>(
          geometry::ignis::attributes::surface, AttrDomain::Point);
      if (writer) {
        writer.span.fill(surface);
        writer.finish();
      }
    }
  }
}

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr)
      .is_default_link_socket()
      .description("Geometry that should emit smoke or fire");
  b.add_input<decl::Menu>("Mode"_ustr)
      .static_items(source_mode_items)
      .default_value(MenuValue(int(SourceMode::Points)))
      .description("Whether the points or the surface of a mesh emit");
  b.add_input<decl::Float>("Radius"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description(
          "Radius of the emission kernel in metres. 0 uses one voxel of the solver, which is the "
          "tightest a source can be without holes");
  b.add_input<decl::Float>("Density"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .structure_type(StructureType::Field)
      .description("Density the source holds its region at");
  b.add_input<decl::Float>("Temperature"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .structure_type(StructureType::Field)
      .description("Temperature of the emitted smoke, which is what the buoyancy pushes up");
  b.add_input<decl::Float>("Fuel"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .structure_type(StructureType::Field)
      .description("Fuel the source injects; burning it needs `Burn Rate` on the solver");
  b.add_input<decl::Vector>("Velocity"_ustr)
      .default_value(float3(0.0f))
      .subtype(PROP_VELOCITY)
      .structure_type(StructureType::Field)
      .description("Initial velocity of the emitted smoke, for a jet or a muzzle flash");
  b.add_output<decl::Geometry>("Sources"_ustr)
      .description("Connect it to the `Sources` input of `Ignis Solver`, or join several of them");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  const int mode = params.extract_input<int>("Mode"_ustr);
  const float radius = std::max(0.0f, params.extract_input<float>("Radius"_ustr));
  const fn::Field<float> density = params.extract_input<fn::Field<float>>("Density"_ustr);
  const fn::Field<float> temperature = params.extract_input<fn::Field<float>>("Temperature"_ustr);
  const fn::Field<float> fuel = params.extract_input<fn::Field<float>>("Fuel"_ustr);
  const fn::Field<float3> velocity = params.extract_input<fn::Field<float3>>("Velocity"_ustr);

  write_source_attributes(geometry,
                          density,
                          temperature,
                          fuel,
                          velocity,
                          radius,
                          mode == int(SourceMode::Surface));

  params.set_output("Sources"_ustr, std::move(geometry));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeIgnisSource"_ustr);
  ntype.ui_name = "Ignis Source";
  ntype.ui_description =
      "Mark geometry as an emitter for `Ignis Solver` and store its density, temperature, fuel and "
      "velocity";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_180;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_ignis_source_cc
