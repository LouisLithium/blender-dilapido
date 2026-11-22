/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Join Agents`: merge any number of `Blender.CrowdAgent` templates into one
 * `Blender.CrowdAgents` set. The input is a multi-input bundle socket, so the node itself has no
 * fixed slot count.
 */

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include <algorithm>

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"
#include "BLI_vector_set.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_join_agents_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();
  b.add_input<decl::Bundle>("Agent"_ustr)
      .multi_input()
      .description("Agent templates from Create Crowd Agent");
  b.add_input<decl::Bool>("Reassign IDs"_ustr)
      .default_value(true)
      .description(
          "Give each template a stable id based on its input order. Turn off to keep the Template "
          "ID from Create Crowd Agent (ids must still be unique)");
  b.add_output<decl::Bundle>("Agents"_ustr).align_with_previous().propagate_all();
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeoNodesMultiInput<BundlePtr> agents = params.extract_input<GeoNodesMultiInput<BundlePtr>>(
      "Agent"_ustr);
  const bool reassign = params.extract_input<bool>("Reassign IDs"_ustr);

  BundlePtr result_ptr = Bundle::create();
  Bundle &result = result_ptr.ensure_mutable_inplace();
  result.add(*BundleKey::from_str("Type"), std::string(crowd::AGENTS_TYPE));

  /* The order of a multi-input socket is not guaranteed to be the visual/link order. Sort by the
   * author-provided Template ID first, so Reassign IDs is deterministic even when the links were
   * created in another order. Equal ids keep the multi-input order. */
  Vector<BundlePtr> sorted;
  for (BundlePtr &agent : agents.values) {
    if (agent) {
      sorted.append(agent);
    }
  }
  std::stable_sort(sorted.begin(), sorted.end(), [](const BundlePtr &a, const BundlePtr &b) {
    return a->lookup_path<int>("template_id").value_or(0) <
           b->lookup_path<int>("template_id").value_or(0);
  });

  int count = 0;
  for (BundlePtr &agent : sorted) {
    Bundle &nested = result.ensure_nested_bundle("agent_" + std::to_string(count));
    for (const auto &item : agent->items()) {
      nested.add(item.key, item.value);
    }
    if (reassign) {
      nested.add_override(*BundleKey::from_str("template_id"), count);
    }
    else if (!nested.lookup_path<int>("template_id").has_value()) {
      nested.add(*BundleKey::from_str("template_id"), count);
    }
    count++;
  }

  result.add(*BundleKey::from_str("count"), count);
  if (count == 0) {
    params.error_message_add(NodeWarningType::Info,
                             TIP_("Join Agents: connect at least one Create Crowd Agent"));
  }
  params.set_output("Agents"_ustr, std::move(result_ptr));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeJoinCrowdAgents"_ustr);
  ntype.ui_name = "Join Agents";
  ntype.ui_description = "Merge any number of crowd agent templates into one agent set";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_join_agents_cc
