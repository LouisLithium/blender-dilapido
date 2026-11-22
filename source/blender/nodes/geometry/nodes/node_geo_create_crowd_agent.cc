/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Create Crowd Agent`: pack one character template (mesh + rest skeleton + state table) into an
 * `Blender.CrowdAgent` bundle. `Join Agents` merges several of these into the `Agent` input of
 * `Crowd Simulation`; `Init Crowd Data` uses them to spawn instances on points.
 */

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include <algorithm>

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_create_crowd_agent_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Mesh"_ustr)
      .description("Template mesh. Its vertex groups must be named after the bones");
  b.add_input<decl::Geometry>("Bones"_ustr)
      .description(
          "Rest/bind skeleton of the template (Armature Bones with Space = Rest), not one frame "
          "of the animation");
  b.add_input<decl::Bundle>("States"_ustr)
      .description("State table from Crowd Agent States that this template supports");
  b.add_input<decl::Int>("Spawn Group"_ustr)
      .default_value(0)
      .min(0)
      .description("Only points whose `spawn_group` matches this value can spawn this template");
  b.add_input<decl::Float>("Weight"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .description("Relative chance of this template when several templates share a spawn group");
  b.add_input<decl::Int>("Template ID"_ustr)
      .default_value(0)
      .min(0)
      .description("Stable id of the template; Join Agents can reassign ids by input order");

  b.add_output<decl::Bundle>("Agent"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet mesh = params.extract_input<GeometrySet>("Mesh"_ustr);
  GeometrySet bones = params.extract_input<GeometrySet>("Bones"_ustr);
  BundlePtr states = params.extract_input<BundlePtr>("States"_ustr);
  const int spawn_group = std::max(params.extract_input<int>("Spawn Group"_ustr), 0);
  const float weight = std::max(params.extract_input<float>("Weight"_ustr), 0.0f);
  const int template_id = std::max(params.extract_input<int>("Template ID"_ustr), 0);

  BundlePtr agent_ptr = Bundle::create();
  Bundle &agent = agent_ptr.ensure_mutable_inplace();
  agent.add(*BundleKey::from_str("Type"), std::string(crowd::AGENT_TYPE));
  agent.add(*BundleKey::from_str("mesh"), mesh);
  agent.add(*BundleKey::from_str("bones"), bones);
  if (states) {
    agent.add(*BundleKey::from_str("states"), states);
  }
  agent.add(*BundleKey::from_str("spawn_group"), spawn_group);
  agent.add(*BundleKey::from_str("weight"), weight);
  agent.add(*BundleKey::from_str("template_id"), template_id);

  params.set_output("Agent"_ustr, std::move(agent_ptr));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeCreateCrowdAgent"_ustr);
  ntype.ui_name = "Create Crowd Agent";
  ntype.ui_description =
      "Pack a character template (mesh, rest skeleton and state table) for a crowd";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_create_crowd_agent_cc
