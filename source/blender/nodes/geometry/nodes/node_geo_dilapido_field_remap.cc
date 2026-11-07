/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Dilapido Field Remap`: a declarative selector that scales a force by an Ignis grid value.
 *
 * The node does not have access to the simulation grids, so it only writes a typed bundle. The
 * solver samples the field and applies the window. `Dilapido Force` accepts this bundle on its
 * `Disturbance Remap` / `Turbulence Remap` inputs.
 */

#include <algorithm>

#include "NOD_bundle_type.hh"
#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_socket_declarations.hh"

#include "BLI_string_ref.hh"

#include "GEO_ignis_remap.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_dilapido_field_remap_cc {

using geometry::ignis::ForceRemapField;
using geometry::ignis::ForceRemapMode;

namespace {

constexpr StringRef REMAP_TYPE = "Blender.DilapidoFieldRemap";

class DilapidoFieldRemapBundle {
 public:
  static constexpr StringRefNull name = "Blender.DilapidoFieldRemap";
  static const FlatBundleTypePtr &get_bundle_type();
};

const FlatBundleTypePtr &DilapidoFieldRemapBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(DilapidoFieldRemapBundle::name);
    b.add<decl::Int>("field"_ustr)
        .min(0)
        .description("Ignis grid sampled by the solver: pressure, density, temperature, fuel, "
                     "flame or velocity magnitude");
    b.add<decl::Int>("mode"_ustr)
        .min(0)
        .description("Range, Outside, Above or Below");
    b.add<decl::Float>("min"_ustr).description("Lower bound of the field window");
    b.add<decl::Float>("max"_ustr).description("Upper bound of the field window");
    b.add<decl::Float>("softness"_ustr)
        .min(0.0f)
        .max(1.0f)
        .description("Fraction of the window used as a smooth edge");
    b.add<decl::Float>("scale"_ustr).description("Multiplier applied to the remapped force");
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const EnumPropertyItem field_items[] = {
    {int(ForceRemapField::Pressure),
     "PRESSURE",
     0,
     N_("Pressure"),
     N_("Use the pressure grid (previous frame's projection)")},
    {int(ForceRemapField::Density),
     "DENSITY",
     0,
     N_("Density"),
     N_("Use the smoke density grid")},
    {int(ForceRemapField::Temperature),
     "TEMPERATURE",
     0,
     N_("Temperature"),
     N_("Use the temperature anomaly grid")},
    {int(ForceRemapField::Fuel),
     "FUEL",
     0,
     N_("Fuel"),
     N_("Use the unburned fuel grid")},
    {int(ForceRemapField::Flame),
     "FLAME",
     0,
     N_("Flame"),
     N_("Use the reaction rate grid (K/s)")},
    {int(ForceRemapField::VelocityMagnitude),
     "VELOCITY",
     0,
     N_("Velocity Magnitude"),
     N_("Use the current velocity length")},
    {0, nullptr, 0, nullptr, nullptr},
};

const EnumPropertyItem mode_items[] = {
    {int(ForceRemapMode::Range), "RANGE", 0, N_("Range"), N_("Keep the force inside Min..Max")},
    {int(ForceRemapMode::Outside),
     "OUTSIDE",
     0,
     N_("Outside"),
     N_("Keep the force outside Min..Max")},
    {int(ForceRemapMode::Above), "ABOVE", 0, N_("Above"), N_("Keep the force above Min")},
    {int(ForceRemapMode::Below), "BELOW", 0, N_("Below"), N_("Keep the force below Max")},
    {0, nullptr, 0, nullptr, nullptr},
};

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Menu>("Field"_ustr)
      .static_items(field_items)
      .default_value(MenuValue(int(ForceRemapField::Pressure)))
      .description("Ignis grid whose value selects where the force applies");
  b.add_input<decl::Menu>("Mode"_ustr)
      .static_items(mode_items)
      .default_value(MenuValue(int(ForceRemapMode::Range)))
      .description("How the field window selects the force");
  b.add_input<decl::Float>("Min"_ustr)
      .default_value(0.0f)
      .description("Lower field value. For Above this is the only threshold");
  b.add_input<decl::Float>("Max"_ustr)
      .default_value(1.0f)
      .description("Upper field value. For Below this is the only threshold");
  b.add_input<decl::Float>("Softness"_ustr)
      .default_value(0.2f)
      .min(0.0f)
      .max(1.0f)
      .description("Fraction of the window used as a smooth edge; 0 is a hard step");
  b.add_input<decl::Float>("Scale"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .description("Multiplier applied to the remapped force; 0 disables it");
  b.add_output<decl::Bundle>("Remap"_ustr)
      .description("Connect it to `Dilapido Force > Disturbance Remap` or `Turbulence Remap`");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const int field = params.extract_input<int>("Field"_ustr);
  const int mode = params.extract_input<int>("Mode"_ustr);
  const float min = params.extract_input<float>("Min"_ustr);
  const float max = params.extract_input<float>("Max"_ustr);
  const float softness = std::clamp(params.extract_input<float>("Softness"_ustr), 0.0f, 1.0f);
  const float scale = std::max(0.0f, params.extract_input<float>("Scale"_ustr));

  BundlePtr bundle_ptr = Bundle::create();
  Bundle &bundle = bundle_ptr.ensure_mutable_inplace();
  bundle.add(Bundle::type_item_name, std::string(REMAP_TYPE));
  bundle.add(*BundleKey::from_str("field"), field);
  bundle.add(*BundleKey::from_str("mode"), mode);
  bundle.add(*BundleKey::from_str("min"), min);
  bundle.add(*BundleKey::from_str("max"), max);
  bundle.add(*BundleKey::from_str("softness"), softness);
  bundle.add(*BundleKey::from_str("scale"), scale);
  params.set_output("Remap"_ustr, std::move(bundle_ptr));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeDilapidoFieldRemap"_ustr);
  ntype.ui_name = "Dilapido Field Remap";
  ntype.ui_description =
      "Remap a force by an Ignis grid value: keep Disturbance or Turbulence only where pressure, "
      "density, temperature, fuel, flame or velocity is inside a range";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  DilapidoFieldRemapBundle::get_bundle_type();
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_dilapido_field_remap_cc
