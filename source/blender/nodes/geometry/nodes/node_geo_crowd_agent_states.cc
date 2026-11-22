/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Crowd Agent States`: a fixed-capacity state table (name + animation info + blend in/out).
 *
 * A state table is a `Blender.CrowdStates` bundle with `count` and `state_0..N` nested bundles.
 * The C2a version uses eight fixed slots; adding a dynamic `+` button would require implementing
 * dynamic socket declarations on top of this node, which is deliberately left for later.
 */

#include <algorithm>

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include "BLI_string_ref.hh"
#include "BLI_ustring.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_crowd_agent_states_cc {

constexpr int SLOT_COUNT = 8;

static std::string slot_name(const StringRef prefix, const int index)
{
  return std::string(prefix) + " " + std::to_string(index);
}

static UString slot_ustr(const StringRef prefix, const int index)
{
  return UString(StringRef(slot_name(prefix, index)));
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();

  for (const int i : IndexRange(1, SLOT_COUNT)) {
    const std::string panel = slot_name("State", i);
    PanelDeclarationBuilder &state_panel = b.add_panel(UString(StringRef(panel)));
    state_panel.default_closed(i > 3);
    state_panel.add_input<decl::Bool>(slot_ustr("Enabled", i), slot_ustr("enabled", i))
        .default_value(i <= 3);
    state_panel.add_input<decl::String>(slot_ustr("Name", i), slot_ustr("name", i))
        .default_value(i == 1 ? "Idle" : i == 2 ? "Run" : i == 3 ? "Death" : "");
    state_panel.add_input<decl::Float>(slot_ustr("Blend In", i), slot_ustr("blend_in", i))
        .default_value(0.2f)
        .min(0.0f);
    state_panel.add_input<decl::Float>(slot_ustr("Blend Out", i), slot_ustr("blend_out", i))
        .default_value(0.2f)
        .min(0.0f);
    state_panel.add_input<decl::Bool>(slot_ustr("Loop", i), slot_ustr("loop", i))
        .default_value(i != 3);
    state_panel.add_input<decl::Float>(slot_ustr("Speed", i), slot_ustr("speed", i))
        .default_value(1.0f)
        .min(0.0f);
    state_panel.add_input<decl::Bundle>(slot_ustr("Animation", i), slot_ustr("animation", i))
        .description("Animation Info bundle for this state");
  }

  b.add_output<decl::Bundle>("States"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  BundlePtr states_ptr = Bundle::create();
  Bundle &states = states_ptr.ensure_mutable_inplace();
  states.add(*BundleKey::from_str("Type"), std::string(crowd::STATES_TYPE));

  int count = 0;
  for (const int i : IndexRange(1, SLOT_COUNT)) {
    const std::string name = params.extract_input<std::string>(slot_ustr("name", i));
    const bool enabled = params.extract_input<bool>(slot_ustr("enabled", i));
    if (!enabled || name.empty()) {
      continue;
    }
    Bundle &state = states.ensure_nested_bundle("state_" + std::to_string(count));
    state.add(*BundleKey::from_str("name"), name);
    state.add(*BundleKey::from_str("blend_in"),
              std::max(params.extract_input<float>(slot_ustr("blend_in", i)), 0.0f));
    state.add(*BundleKey::from_str("blend_out"),
              std::max(params.extract_input<float>(slot_ustr("blend_out", i)), 0.0f));
    state.add(*BundleKey::from_str("loop"),
              params.extract_input<bool>(slot_ustr("loop", i)));
    state.add(*BundleKey::from_str("speed"),
              std::max(params.extract_input<float>(slot_ustr("speed", i)), 0.0f));
    BundlePtr animation = params.extract_input<BundlePtr>(slot_ustr("animation", i));
    if (animation) {
      state.add(*BundleKey::from_str("animation"), animation);
    }
    count++;
  }

  states.add(*BundleKey::from_str("count"), count);
  if (count == 0) {
    params.error_message_add(NodeWarningType::Info,
                             TIP_("Crowd Agent States: enable at least one named state"));
  }
  params.set_output("States"_ustr, std::move(states_ptr));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeCrowdAgentStates"_ustr);
  ntype.ui_name = "Crowd Agent States";
  ntype.ui_description =
      "State table for crowd agents: name, animation info and blend in/out per state";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_220;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_crowd_agent_states_cc
