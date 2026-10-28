/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Dilapido Force`: one force field as a bundle, the force counterpart of `Dilapido Collider`.
 *
 * The bundle is typed (`Blender.DilapidoForce`) and carries a single uniform acceleration, which is
 * what a wind is. It is nested into a solver with `Combine Bundle` (or by storing it under its own
 * path), exactly like a collider, so several forces and colliders can act at the same time.
 *
 * Key written:
 *
 * | key            | meaning                                   |
 * |----------------|-------------------------------------------|
 * | `Type`         | `Blender.DilapidoForce`                   |
 * | `acceleration` | the force per unit mass, in m/s² (vector) |
 *
 * A uniform acceleration is the one force a grid solver, a particle solver and a rigid body solver
 * can all apply without a per element closure, so this is the shared vocabulary the Dilapido nodes
 * grow from: a field that needs to vary in space can be added as another bundle type later without
 * touching the solvers.
 */

#include "NOD_bundle_type.hh"
#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_socket_declarations.hh"

#include "BLI_math_vector.hh"
#include "BLI_string_ref.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_dilapido_force_cc {

namespace {

constexpr StringRef FORCE_TYPE = "Blender.DilapidoForce";

/** The bundle type, registered so that assets can create and edit these bundles in a node tree. */
class DilapidoForceBundle {
 public:
  static constexpr StringRefNull name = "Blender.DilapidoForce";
  static const FlatBundleTypePtr &get_bundle_type();
};

const FlatBundleTypePtr &DilapidoForceBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(DilapidoForceBundle::name);
    b.add<decl::Vector>("acceleration"_ustr)
        .subtype(PROP_ACCELERATION)
        .description("Acceleration the field adds to whatever it acts on, in m/s²");
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Vector>("Direction"_ustr)
      .default_value(float3(1.0f, 0.0f, 0.0f))
      .subtype(PROP_DIRECTION)
      .description("Direction of the wind. It is normalized, so only the strength sets the size");
  b.add_input<decl::Float>("Strength"_ustr)
      .default_value(0.0f)
      .subtype(PROP_ACCELERATION)
      .description(
          "Acceleration the wind adds along the direction, in m/s². It is compared with gravity: "
          "`9.81` pushes as hard as gravity pulls; give it a negative value to blow the other way");
  b.add_output<decl::Bundle>("Force"_ustr)
      .description(
          "Connect it to the `Effectors` input of a solver, or nest it in a bundle with `Combine "
          "Bundle`");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const float3 direction = params.extract_input<float3>("Direction"_ustr);
  const float strength = params.extract_input<float>("Strength"_ustr);
  const float length = math::length(direction);
  /* A zero direction is allowed and simply does nothing, so the node never has to warn about it. */
  const float3 acceleration = length > 0.0f ? direction * (strength / length) : float3(0.0f);

  BundlePtr bundle_ptr = Bundle::create();
  Bundle &bundle = bundle_ptr.ensure_mutable_inplace();
  bundle.add(Bundle::type_item_name, std::string(FORCE_TYPE));
  bundle.add(*BundleKey::from_str("acceleration"), acceleration);
  params.set_output("Force"_ustr, std::move(bundle_ptr));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeDilapidoForce"_ustr);
  ntype.ui_name = "Dilapido Force";
  ntype.ui_description =
      "A force field (a uniform acceleration, i.e. wind) that the Dilapido solvers can consume";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  DilapidoForceBundle::get_bundle_type();
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_180;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_dilapido_force_cc
