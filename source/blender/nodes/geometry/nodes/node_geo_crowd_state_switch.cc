/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Crowd State Switch`: one rule that changes an agent's animation state when a condition is true.
 *
 * Chain several nodes for priority rules (`distance < 2 -> Idle`, `blocked -> Avoid`, ...). The
 * node records `prev_state`, `prev_phase`, resets `state_time` and (optionally) `phase`, so
 * `Crowd Bones` can cross-fade from the previous state and `Crowd Motion` starts its timer from
 * zero. With `Wait For Blend` the node ignores the condition until the running transition has
 * finished, so a noisy condition cannot restart the blend every frame.
 */

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include <algorithm>
#include <string>

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_string_ref.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_pointcloud.hh"

#include "FN_field_evaluation.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_crowd_state_switch_cc {

struct BlendTimes {
  float in = 0.0f;
  float out = 0.0f;
};

static Map<std::string, BlendTimes> read_blend_times(const Bundle *states)
{
  Map<std::string, BlendTimes> result;
  if (states == nullptr) {
    return result;
  }
  const int count = states->lookup_path<int>("count").value_or(0);
  for (const int i : IndexRange(count)) {
    const std::optional<BundlePtr> state = states->lookup_path<BundlePtr>("state_" +
                                                                          std::to_string(i));
    if (!state || !*state) {
      continue;
    }
    const std::string name = (*state)->lookup_path<std::string>("name").value_or("");
    if (name.empty()) {
      continue;
    }
    BlendTimes times;
    times.in = std::max((*state)->lookup_path<float>("blend_in").value_or(0.0f), 0.0f);
    times.out = std::max((*state)->lookup_path<float>("blend_out").value_or(0.0f), 0.0f);
    result.add(name, times);
  }
  return result;
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Points"_ustr)
      .description("Crowd points from Init Crowd Data / Crowd Motion");
  b.add_input<decl::Bool>("Condition"_ustr)
      .default_value(false)
      .structure_type(StructureType::Field)
      .description("When true on a point, switch that agent to the target state");
  b.add_input<decl::Bundle>("States"_ustr)
      .description(
          "Optional state table. Used by `Wait For Blend` to look up the target Blend In and the "
          "previous state's Blend Out");
  b.add_input<decl::String>("State"_ustr)
      .default_value("Idle")
      .description("Target state name (resolved by name in the state table)");
  b.add_input<decl::Bool>("Reset Phase"_ustr)
      .default_value(true)
      .description("Restart the target clip from the beginning (phase = 0) when switching");
  b.add_input<decl::Bool>("Reset State Time"_ustr)
      .default_value(true)
      .description("Reset state_time so Crowd Bones blends in from the previous state");
  b.add_input<decl::Bool>("Wait For Blend"_ustr)
      .default_value(true)
      .description(
          "Ignore the condition while the current transition is still blending. Requires the "
          "States input and a state_time that advances (Crowd Motion)");

  b.add_output<decl::Geometry>("Points"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Points"_ustr);
  const PointCloud *points = geometry.get_pointcloud();
  if (points == nullptr) {
    params.error_message_add(NodeWarningType::Info,
                             TIP_("Crowd State Switch: the Points input is not a point cloud"));
    params.set_output("Points"_ustr, std::move(geometry));
    return;
  }

  const std::string target_state = params.extract_input<std::string>("State"_ustr);
  const bool reset_phase = params.extract_input<bool>("Reset Phase"_ustr);
  const bool reset_time = params.extract_input<bool>("Reset State Time"_ustr);
  const bool wait_for_blend = params.extract_input<bool>("Wait For Blend"_ustr);
  BundlePtr states_bundle = params.extract_input<BundlePtr>("States"_ustr);
  const Map<std::string, BlendTimes> blend_times = read_blend_times(states_bundle.get());

  const bke::GeometryFieldContext field_context(*points, AttrDomain::Point);
  fn::FieldEvaluator evaluator(field_context, points->totpoint);
  evaluator.set_selection(params.extract_input<Field<bool>>("Condition"_ustr));
  evaluator.evaluate();
  const IndexMask switched = evaluator.get_evaluated_selection_as_mask();
  if (switched.is_empty() || target_state.empty()) {
    params.set_output("Points"_ustr, std::move(geometry));
    return;
  }

  PointCloud *result = BKE_pointcloud_copy_for_eval(points);
  bke::MutableAttributeAccessor attributes = result->attributes_for_write();
  const AttrDomain domain = AttrDomain::Point;

  bke::SpanAttributeWriter<MStringProperty> state_writer =
      attributes.lookup_or_add_for_write_span<MStringProperty>("state", domain);
  bke::SpanAttributeWriter<MStringProperty> prev_state_writer =
      attributes.lookup_or_add_for_write_span<MStringProperty>("prev_state", domain);
  bke::SpanAttributeWriter<int> state_id_writer = attributes.lookup_or_add_for_write_span<int>(
      "state_id", domain);
  bke::SpanAttributeWriter<float> state_time_writer =
      attributes.lookup_or_add_for_write_span<float>("state_time", domain);
  bke::SpanAttributeWriter<float> phase_writer = attributes.lookup_or_add_for_write_span<float>(
      "phase", domain);
  bke::SpanAttributeWriter<float> phase_offset_writer =
      attributes.lookup_or_add_for_write_span<float>("phase_offset", domain);
  bke::SpanAttributeWriter<float> prev_phase_writer =
      attributes.lookup_or_add_for_write_span<float>("prev_phase", domain);

  const VArray<MStringProperty> previous_states =
      points->attributes().lookup<MStringProperty>("state").varray;
  const VArray<MStringProperty> old_prev_states =
      points->attributes().lookup<MStringProperty>("prev_state").varray;
  const VArray<float> state_times =
      points->attributes().lookup_or_default<float>("state_time", domain, 0.0f).varray;
  const VArray<float> blend_durations =
      points->attributes().lookup<float>("blend_duration").varray;
  const VArray<float> src_phases =
      points->attributes().lookup_or_default<float>("phase", domain, 0.0f).varray;
  const VArray<float> src_phase_offsets =
      points->attributes().lookup<float>("phase_offset").varray;
  if (!src_phase_offsets) {
    /* Migrate old point clouds: the current phase becomes the stable per-agent offset, so a
     * state switch no longer syncs every agent to phase 0. */
    for (const int i : IndexRange(points->totpoint)) {
      phase_offset_writer.span[i] = src_phases[i];
    }
  }

  switched.foreach_index([&](const int i) {
    const std::string previous = previous_states ? crowd::read_string(previous_states[i]) : "";
    if (previous == target_state) {
      return;
    }
    if (wait_for_blend && !previous.empty()) {
      /* Prefer this node's own States input; otherwise use the duration written by `Crowd
       * Motion` (which lets existing trees wait without extra wiring). */
      float duration = -1.0f;
      if (!blend_times.is_empty()) {
        const std::string old_previous = old_prev_states ? crowd::read_string(old_prev_states[i]) :
                                                           "";
        if (!old_previous.empty() && old_previous != previous) {
          const BlendTimes current_times = blend_times.lookup_default(previous, BlendTimes());
          const BlendTimes old_times = blend_times.lookup_default(old_previous, BlendTimes());
          duration = std::max(current_times.in, old_times.out);
        }
        else {
          duration = 0.0f;
        }
      }
      if (duration < 0.0f && blend_durations) {
        duration = blend_durations[i];
      }
      if (duration > 0.0f && state_times[i] < duration) {
        return;
      }
    }
    /* Keep the phase the previous clip was playing at, so Crowd Bones can continue it during the
     * blend out even when `Reset Phase` restarts the target state. */
    const float phase_offset = src_phase_offsets ? src_phase_offsets[i] : src_phases[i];
    prev_phase_writer.span[i] = src_phases[i];
    phase_offset_writer.span[i] = phase_offset;
    if (!previous.empty()) {
      prev_state_writer.span[i] = previous_states[i];
    }
    MStringProperty target;
    crowd::write_string(target, target_state);
    state_writer.span[i] = target;
    state_id_writer.span[i] = -1;
    if (reset_time) {
      state_time_writer.span[i] = 0.0f;
    }
    if (reset_phase) {
      phase_writer.span[i] = phase_offset;
    }
  });

  state_writer.finish();
  prev_state_writer.finish();
  state_id_writer.finish();
  state_time_writer.finish();
  phase_writer.finish();
  phase_offset_writer.finish();
  prev_phase_writer.finish();

  params.set_output("Points"_ustr, GeometrySet::from_pointcloud(result));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeCrowdStateSwitch"_ustr);
  ntype.ui_name = "Crowd State Switch";
  ntype.ui_description =
      "Switch an agent to another animation state when a per-point condition becomes true";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_220;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_crowd_state_switch_cc
