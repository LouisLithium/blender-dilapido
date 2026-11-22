/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Crowd State`: one entry of a crowd state table (name + animation + blend times).
 * Combine any number of these with `Join States`; there is no fixed slot count.
 */

#include <algorithm>

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_crowd_state_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::String>("Name"_ustr).default_value("Idle");
  b.add_input<decl::Bundle>("Animation"_ustr)
      .description("Animation Info bundle for this state");
  b.add_input<decl::Float>("Blend In"_ustr).default_value(0.2f).min(0.0f);
  b.add_input<decl::Float>("Blend Out"_ustr).default_value(0.2f).min(0.0f);
  b.add_input<decl::Bool>("Loop"_ustr).default_value(true);
  b.add_input<decl::Float>("Speed"_ustr).default_value(1.0f).min(0.0f);

  b.add_output<decl::Bundle>("State"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const std::string name = params.extract_input<std::string>("Name"_ustr);
  BundlePtr animation = params.extract_input<BundlePtr>("Animation"_ustr);
  const float blend_in = std::max(params.extract_input<float>("Blend In"_ustr), 0.0f);
  const float blend_out = std::max(params.extract_input<float>("Blend Out"_ustr), 0.0f);
  const bool loop = params.extract_input<bool>("Loop"_ustr);
  const float speed = std::max(params.extract_input<float>("Speed"_ustr), 0.0f);

  BundlePtr state_ptr = Bundle::create();
  Bundle &state = state_ptr.ensure_mutable_inplace();
  state.add(*BundleKey::from_str("Type"), std::string(crowd::STATE_TYPE));
  state.add(*BundleKey::from_str("name"), name);
  state.add(*BundleKey::from_str("blend_in"), blend_in);
  state.add(*BundleKey::from_str("blend_out"), blend_out);
  state.add(*BundleKey::from_str("loop"), loop);
  state.add(*BundleKey::from_str("speed"), speed);
  if (animation) {
    state.add(*BundleKey::from_str("animation"), animation);
  }
  params.set_output("State"_ustr, std::move(state_ptr));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeCrowdState"_ustr);
  ntype.ui_name = "Crowd State";
  ntype.ui_description =
      "One crowd animation state (name, animation info and blend in/out); join any number with "
      "Join States";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_crowd_state_cc
