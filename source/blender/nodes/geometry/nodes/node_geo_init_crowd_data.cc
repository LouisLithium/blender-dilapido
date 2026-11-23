/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Init Crowd Data`: turn a raw point cloud into crowd instances.
 *
 * Each point keeps its position/rotation/unknown attributes and gets:
 * `agent_id` (stable, unique), `template_id`, `spawn_group`, `state` / `state_id`, `phase`
 * (0..1), `speed`, `scale`, `active` and `physics_mode`.
 *
 * Templates are chosen deterministically from the `spawn_group` matching the point's
 * `spawn_group` (or `group_id`), weighted by `Create Crowd Agent > Weight`. The point's `state`
 * string is resolved against the state table; unknown/missing names fall back to the default/first
 * state. `phase` is randomized from `Seed` + `agent_id` when the point does not provide one.
 */

#include <algorithm>
#include <cmath>
#include <string>

#include <fmt/format.h>

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include "BLI_array.hh"
#include "BLI_math_base.hh"
#include "BLI_set.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_pointcloud.hh"

#include "DNA_meshdata_types.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_init_crowd_data_cc {

struct TemplateInfo {
  int template_id = 0;
  int spawn_group = 0;
  float weight = 1.0f;
};

struct StateInfo {
  std::string name;
};

static void read_templates(const Bundle &agents, Vector<TemplateInfo> &r_templates)
{
  const int count = agents.lookup_path<int>("count").value_or(0);
  r_templates.reserve(count);
  for (const int i : IndexRange(count)) {
    const std::optional<BundlePtr> agent = agents.lookup_path<BundlePtr>("agent_" +
                                                                         std::to_string(i));
    if (!agent || !*agent) {
      continue;
    }
    TemplateInfo info;
    info.template_id = (*agent)->lookup_path<int>("template_id").value_or(i);
    info.spawn_group = std::max((*agent)->lookup_path<int>("spawn_group").value_or(0), 0);
    info.weight = std::max((*agent)->lookup_path<float>("weight").value_or(1.0f), 0.0f);
    r_templates.append(info);
  }
}

static void read_states(const Bundle *states, Vector<StateInfo> &r_states)
{
  if (states == nullptr) {
    return;
  }
  const int count = states->lookup_path<int>("count").value_or(0);
  r_states.reserve(count);
  for (const int i : IndexRange(count)) {
    const std::optional<BundlePtr> state = states->lookup_path<BundlePtr>("state_" +
                                                                          std::to_string(i));
    if (!state || !*state) {
      continue;
    }
    StateInfo info;
    info.name = (*state)->lookup_path<std::string>("name").value_or("");
    r_states.append(info);
  }
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Points"_ustr)
      .description(
          "Spawn anchors. Each point becomes one agent instance; position/rotation drive the "
          "kinematic character, `spawn_group` selects templates and `state` requests a state");
  b.add_input<decl::Bundle>("Agents"_ustr).description("Agent set from Join Agents");
  b.add_input<decl::Bundle>("States"_ustr).description("State table from Crowd Agent States");
  b.add_input<decl::Int>("Seed"_ustr)
      .default_value(0)
      .description("Seed for template choice and random phase");
  b.add_input<decl::Float>("Phase Spread"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .max(1.0f)
      .description(
          "Range of the per-agent animation phase offset (0 = every agent starts in sync, "
          "1 = full cycle)");
  b.add_input<decl::Bool>("Fallback To Any Group"_ustr)
      .default_value(true)
      .description("Use any template when no template matches the point's spawn group");
  b.add_input<decl::String>("Default State"_ustr)
      .default_value("")
      .description("State used when a point has no state; empty uses the first enabled state");

  b.add_output<decl::Geometry>("Points"_ustr);
  b.add_output<decl::Bundle>("Agents"_ustr);
  b.add_output<decl::Bundle>("States"_ustr);
  b.add_output<decl::Int>("Active Count"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet points_geometry = params.extract_input<GeometrySet>("Points"_ustr);
  BundlePtr agents = params.extract_input<BundlePtr>("Agents"_ustr);
  BundlePtr states = params.extract_input<BundlePtr>("States"_ustr);
  const int seed = params.extract_input<int>("Seed"_ustr);
  const float phase_spread = math::clamp(
      params.extract_input<float>("Phase Spread"_ustr), 0.0f, 1.0f);
  const bool fallback_any = params.extract_input<bool>("Fallback To Any Group"_ustr);
  const std::string default_state = params.extract_input<std::string>("Default State"_ustr);

  const PointCloud *points = points_geometry.get_pointcloud();
  if (points == nullptr) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Init Crowd Data: the Points input is not a point cloud (Mesh to Points?)"));
    params.set_output("Points"_ustr, std::move(points_geometry));
    params.set_output("Agents"_ustr, std::move(agents));
    params.set_output("States"_ustr, std::move(states));
    params.set_output("Active Count"_ustr, 0);
    return;
  }
  if (!agents) {
    params.error_message_add(NodeWarningType::Info,
                             TIP_("Init Crowd Data: connect Join Agents to the Agents input"));
  }

  Vector<TemplateInfo> templates;
  if (agents) {
    read_templates(*agents, templates);
  }
  Vector<StateInfo> state_list;
  read_states(states.get(), state_list);

  const bke::AttributeAccessor src_attributes = points->attributes();
  const VArray<int> spawn_groups =
      src_attributes.contains("spawn_group") ?
          src_attributes.lookup_or_default<int>("spawn_group", AttrDomain::Point, 0).varray :
          src_attributes.lookup_or_default<int>("group_id", AttrDomain::Point, 0).varray;
  const VArray<MStringProperty> requested_states =
      src_attributes.lookup<MStringProperty>("state").varray;
  const VArray<int> ids =
      src_attributes.lookup_or_default<int>("id", AttrDomain::Point, -1).varray;
  const VArray<float> phases =
      src_attributes.lookup_or_default<float>("phase", AttrDomain::Point, -1.0f).varray;
  const VArray<float> speeds =
      src_attributes.lookup_or_default<float>("speed", AttrDomain::Point, 1.0f).varray;
  const VArray<float> scales =
      src_attributes.lookup_or_default<float>("scale", AttrDomain::Point, 1.0f).varray;
  const VArray<bool> actives =
      src_attributes.lookup_or_default<bool>("active", AttrDomain::Point, true).varray;
  const VArray<int> physics_modes =
      src_attributes.lookup_or_default<int>("physics_mode", AttrDomain::Point, 0).varray;

  PointCloud *result = BKE_pointcloud_copy_for_eval(points);
  bke::MutableAttributeAccessor out_attributes = result->attributes_for_write();
  const AttrDomain domain = AttrDomain::Point;
  bke::SpanAttributeWriter<int> agent_id_writer = out_attributes.lookup_or_add_for_write_span<int>(
      "agent_id", domain);
  bke::SpanAttributeWriter<int> template_id_writer =
      out_attributes.lookup_or_add_for_write_span<int>("template_id", domain);
  bke::SpanAttributeWriter<int> spawn_group_writer =
      out_attributes.lookup_or_add_for_write_span<int>("spawn_group", domain);
  bke::SpanAttributeWriter<int> state_id_writer = out_attributes.lookup_or_add_for_write_span<int>(
      "state_id", domain);
  bke::SpanAttributeWriter<MStringProperty> state_writer =
      out_attributes.lookup_or_add_for_write_span<MStringProperty>("state", domain);
  bke::SpanAttributeWriter<float> phase_writer =
      out_attributes.lookup_or_add_for_write_span<float>("phase", domain);
  bke::SpanAttributeWriter<float> phase_offset_writer =
      out_attributes.lookup_or_add_for_write_span<float>("phase_offset", domain);
  bke::SpanAttributeWriter<float> speed_writer =
      out_attributes.lookup_or_add_for_write_span<float>("speed", domain);
  bke::SpanAttributeWriter<float> scale_writer =
      out_attributes.lookup_or_add_for_write_span<float>("scale", domain);
  bke::SpanAttributeWriter<bool> active_writer = out_attributes.lookup_or_add_for_write_span<bool>(
      "active", domain);
  bke::SpanAttributeWriter<int> physics_mode_writer =
      out_attributes.lookup_or_add_for_write_span<int>("physics_mode", domain);

  Set<int> used_ids;
  int active_count = 0;
  int unmatched = 0;
  int duplicate_ids = 0;
  Vector<int> candidates;
  candidates.reserve(templates.size());

  for (const int point : IndexRange(points->totpoint)) {
    int stable_id = ids[point] >= 0 ? ids[point] : point;
    if (stable_id >= crowd::MAX_AGENT_ID) {
      stable_id %= crowd::MAX_AGENT_ID;
    }
    if (used_ids.contains(stable_id)) {
      duplicate_ids++;
      while (used_ids.contains(stable_id)) {
        stable_id = (stable_id + 1) % crowd::MAX_AGENT_ID;
      }
    }
    used_ids.add(stable_id);

    const int group = spawn_groups[point];
    candidates.clear();
    float total_weight = 0.0f;
    for (const int i : templates.index_range()) {
      if (templates[i].spawn_group == group) {
        candidates.append(i);
        total_weight += templates[i].weight;
      }
    }
    if (candidates.is_empty() && fallback_any) {
      for (const int i : templates.index_range()) {
        candidates.append(i);
        total_weight += templates[i].weight;
      }
    }

    const bool has_template = !candidates.is_empty();
    int chosen = -1;
    if (has_template) {
      const uint64_t hash = crowd::hash_ids(seed, stable_id);
      float pick = crowd::hash_to_unit(hash) * total_weight;
      chosen = candidates.last();
      for (const int candidate : candidates) {
        if (pick < templates[candidate].weight) {
          chosen = candidate;
          break;
        }
        pick -= templates[candidate].weight;
      }
    }
    else {
      unmatched++;
    }

    std::string state_name;
    int state_id = -1;
    if (has_template) {
      std::string requested = requested_states ? crowd::read_string(requested_states[point]) : "";
      if (requested.empty()) {
        requested = default_state;
      }
      const auto find_state = [&](const std::string &name) -> int {
        for (const int i : state_list.index_range()) {
          if (state_list[i].name == name) {
            return i;
          }
        }
        return -1;
      };
      state_id = find_state(requested);
      if (state_id < 0 && !default_state.empty()) {
        state_id = find_state(default_state);
      }
      if (state_id < 0 && !state_list.is_empty()) {
        state_id = 0;
      }
      if (state_id >= 0) {
        state_name = state_list[state_id].name;
      }
    }

    float phase = phases[point];
    if (!(phase >= 0.0f)) {
      phase = crowd::hash_to_unit(crowd::hash_ids(seed ^ 0x5bd1, stable_id)) * phase_spread;
    }
    phase -= std::floor(phase);

    agent_id_writer.span[point] = has_template ? stable_id : -1;
    template_id_writer.span[point] = has_template ? templates[chosen].template_id : -1;
    spawn_group_writer.span[point] = has_template ? templates[chosen].spawn_group : group;
    state_id_writer.span[point] = state_id;
    crowd::write_string(state_writer.span[point], state_name);
    phase_writer.span[point] = phase;
    phase_offset_writer.span[point] = phase;
    speed_writer.span[point] = std::max(speeds[point], 0.0f);
    scale_writer.span[point] = std::max(scales[point], 1.0e-4f);
    active_writer.span[point] = actives[point] && has_template;
    physics_mode_writer.span[point] = math::clamp(physics_modes[point], 0, 2);
    active_count += (actives[point] && has_template) ? 1 : 0;
  }

  if (templates.is_empty()) {
    params.error_message_add(NodeWarningType::Warning,
                             TIP_("Init Crowd Data: the Agents input has no templates"));
  }
  if (unmatched > 0) {
    params.error_message_add(
        NodeWarningType::Warning,
        fmt::format("Init Crowd Data: {} point(s) have no template for their spawn group "
                    "and were deactivated (enable Fallback To Any Group to avoid this)",
                    unmatched));
  }
  if (duplicate_ids > 0) {
    params.error_message_add(
        NodeWarningType::Info,
        fmt::format("Init Crowd Data: {} point(s) had duplicate `id` values; they were shifted to "
                    "keep agent ids unique",
                    duplicate_ids));
  }

  agent_id_writer.finish();
  template_id_writer.finish();
  spawn_group_writer.finish();
  state_id_writer.finish();
  state_writer.finish();
  phase_writer.finish();
  phase_offset_writer.finish();
  speed_writer.finish();
  scale_writer.finish();
  active_writer.finish();
  physics_mode_writer.finish();
  result->tag_positions_changed();

  params.set_output("Points"_ustr, GeometrySet::from_pointcloud(result));
  params.set_output("Agents"_ustr, std::move(agents));
  params.set_output("States"_ustr, std::move(states));
  params.set_output("Active Count"_ustr, active_count);
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeInitCrowdData"_ustr);
  ntype.ui_name = "Init Crowd Data";
  ntype.ui_description =
      "Assign templates, states, phases and stable agent ids to a spawn point cloud";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_init_crowd_data_cc
