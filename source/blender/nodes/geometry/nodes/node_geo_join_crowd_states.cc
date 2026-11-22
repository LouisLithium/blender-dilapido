/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Join States`: merge any number of `Crowd State` bundles into one `Blender.CrowdStates` table.
 */

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include "BLI_string_ref.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_join_crowd_states_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();
  b.add_input<decl::Bundle>("State"_ustr)
      .multi_input()
      .description("States from Crowd State; connect as many as needed");
  b.add_output<decl::Bundle>("States"_ustr).align_with_previous().propagate_all();
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeoNodesMultiInput<BundlePtr> states = params.extract_input<GeoNodesMultiInput<BundlePtr>>(
      "State"_ustr);

  BundlePtr result_ptr = Bundle::create();
  Bundle &result = result_ptr.ensure_mutable_inplace();
  result.add(*BundleKey::from_str("Type"), std::string(crowd::STATES_TYPE));

  int count = 0;
  for (BundlePtr &state : states.values) {
    if (!state) {
      continue;
    }
    Bundle &nested = result.ensure_nested_bundle("state_" + std::to_string(count));
    for (const auto &item : state->items()) {
      nested.add(item.key, item.value);
    }
    count++;
  }
  result.add(*BundleKey::from_str("count"), count);

  if (count == 0) {
    params.error_message_add(NodeWarningType::Info,
                             TIP_("Join States: connect at least one Crowd State"));
  }
  params.set_output("States"_ustr, std::move(result_ptr));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeJoinCrowdStates"_ustr);
  ntype.ui_name = "Join States";
  ntype.ui_description = "Merge any number of crowd states into one state table";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_join_crowd_states_cc
