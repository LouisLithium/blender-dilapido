/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Crowd Ragdoll Switch`: per-agent switch between animation and ragdoll simulation.
 *
 * It only works on the crowd point cloud. A point with `physics_mode == 1` is a dynamic ragdoll,
 * `physics_mode == 3` is recovering (kinematic blend back to animation) and `0` is animated.
 * The node also snapshots the ragdoll pose for the recovery blend and keeps a deterministic
 * `Max Dynamic` budget with a pending queue.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_vector.hh"
#include "BLI_set.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_pointcloud.hh"

#include "FN_field_evaluation.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_crowd_ragdoll_switch_cc {

struct BoneSnapshot {
  int agent_id = 0;
  int parent = -1;
  std::string name;
  float4x4 pose = float4x4::identity();
  float3 position = float3(0.0f);
};

static bool read_bone_snapshots(const GeometrySet &geometry, Vector<BoneSnapshot> &r_bones)
{
  if (const bke::Instances *instances = geometry.get_instances()) {
    const bke::AttributeAccessor attributes = instances->attributes();
    const VArray<int> agents =
        attributes.lookup_or_default<int>("agent_id", AttrDomain::Instance, 0).varray;
    const VArray<int> parents =
        attributes.lookup_or_default<int>("bone_parent", AttrDomain::Instance, -1).varray;
    const VArray<MStringProperty> names = attributes.lookup<MStringProperty>("bone_name").varray;
    const Span<float4x4> transforms = instances->transforms();
    const int count = instances->instances_num();
    for (const int i : IndexRange(count)) {
      BoneSnapshot bone;
      bone.agent_id = agents[i];
      bone.parent = parents[i];
      bone.name = names ? crowd::read_string(names[i]) : "";
      bone.pose = transforms[i];
      bone.position = transforms[i].location();
      r_bones.append(std::move(bone));
    }
    return true;
  }
  if (const PointCloud *points = geometry.get_pointcloud()) {
    const bke::AttributeAccessor attributes = points->attributes();
    const VArray<int> agents =
        attributes.lookup_or_default<int>("agent_id", AttrDomain::Point, 0).varray;
    const VArray<int> parents =
        attributes.lookup_or_default<int>("bone_parent", AttrDomain::Point, -1).varray;
    const VArray<MStringProperty> names = attributes.lookup<MStringProperty>("bone_name").varray;
    const VArray<float4x4> poses = attributes
                                       .lookup_or_default<float4x4>("bone_pose_matrix",
                                                                    AttrDomain::Point,
                                                                    float4x4::identity())
                                       .varray;
    const VArray<float3> positions =
        attributes.lookup_or_default<float3>("position", AttrDomain::Point, float3(0.0f)).varray;
    for (const int i : IndexRange(points->totpoint)) {
      BoneSnapshot bone;
      bone.agent_id = agents[i];
      bone.parent = parents[i];
      bone.name = names ? crowd::read_string(names[i]) : "";
      bone.pose = poses[i];
      bone.position = positions[i];
      r_bones.append(std::move(bone));
    }
    return true;
  }
  return false;
}

static PointCloud *build_recover_geometry(const Vector<BoneSnapshot> &previous_bones,
                                          const Vector<BoneSnapshot> &previous_recover,
                                          const Set<int> &recovering,
                                          const Set<int> &new_recover)
{
  Vector<int> agents;
  for (const int agent : recovering) {
    agents.append(agent);
  }
  std::sort(agents.begin(), agents.end());

  Vector<const BoneSnapshot *> selected;
  for (const int agent : agents) {
    const Vector<BoneSnapshot> &source = new_recover.contains(agent) ? previous_bones :
                                                                       previous_recover;
    for (const BoneSnapshot &bone : source) {
      if (bone.agent_id == agent) {
        selected.append(&bone);
      }
    }
  }

  PointCloud *result = BKE_pointcloud_new_nomain(selected.size());
  if (selected.is_empty()) {
    return result;
  }
  bke::MutableAttributeAccessor attributes = result->attributes_for_write();
  bke::SpanAttributeWriter<int> agent_writer = attributes.lookup_or_add_for_write_span<int>(
      "agent_id", AttrDomain::Point);
  bke::SpanAttributeWriter<int> parent_writer = attributes.lookup_or_add_for_write_span<int>(
      "bone_parent", AttrDomain::Point);
  bke::SpanAttributeWriter<MStringProperty> name_writer =
      attributes.lookup_or_add_for_write_span<MStringProperty>("bone_name", AttrDomain::Point);
  bke::SpanAttributeWriter<float4x4> pose_writer =
      attributes.lookup_or_add_for_write_span<float4x4>("bone_pose_matrix", AttrDomain::Point);
  bke::SpanAttributeWriter<float3> position_writer =
      attributes.lookup_or_add_for_write_span<float3>("position", AttrDomain::Point);

  for (const int i : selected.index_range()) {
    const BoneSnapshot &bone = *selected[i];
    agent_writer.span[i] = bone.agent_id;
    parent_writer.span[i] = bone.parent;
    crowd::write_string(name_writer.span[i], bone.name);
    pose_writer.span[i] = bone.pose;
    position_writer.span[i] = bone.position;
  }
  agent_writer.finish();
  parent_writer.finish();
  name_writer.finish();
  pose_writer.finish();
  position_writer.finish();
  return result;
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Points"_ustr)
      .description("Crowd points from Init Crowd Data / Crowd Motion");
  b.add_input<decl::Geometry>("Previous Bones"_ustr)
      .description(
          "Solver bone output from the previous frame (used to snapshot the ragdoll pose)");
  b.add_input<decl::Geometry>("Previous Recover"_ustr)
      .description("Recovery snapshot stored by the simulation zone last frame");
  b.add_input<decl::Bool>("Condition"_ustr)
      .default_value(false)
      .structure_type(StructureType::Field)
      .description("When true, switch the agent to ragdoll (if the dynamic budget allows)");
  b.add_input<decl::Bool>("Recover Condition"_ustr)
      .default_value(false)
      .structure_type(StructureType::Field)
      .description("When true on a ragdoll, start the recovery blend back to animation");
  b.add_input<decl::Float>("Time"_ustr)
      .default_value(0.0f)
      .description("Animation time in seconds (e.g. Scene Time > Seconds)");
  b.add_input<decl::Float>("Recover Duration"_ustr)
      .default_value(0.8f)
      .min(0.0f)
      .description("Length of the ragdoll -> animation pose blend");
  b.add_input<decl::Int>("Max Dynamic"_ustr)
      .default_value(32)
      .min(0)
      .description("Maximum number of simultaneously dynamic ragdolls (0 = unlimited)");
  b.add_input<decl::Float>("Pending Timeout"_ustr)
      .default_value(5.0f)
      .min(0.0f)
      .description("Discard a pending trigger after this many seconds when the budget is full");
  b.add_input<decl::Bool>("Once"_ustr)
      .default_value(true)
      .description("Only trigger a ragdoll once per agent");
  b.add_input<decl::Float>("Cooldown"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .description("Minimum time between two ragdoll triggers");

  b.add_output<decl::Geometry>("Points"_ustr);
  b.add_output<decl::Geometry>("Recover"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Points"_ustr);
  GeometrySet previous_bones_geometry = params.extract_input<GeometrySet>("Previous Bones"_ustr);
  GeometrySet previous_recover_geometry = params.extract_input<GeometrySet>(
      "Previous Recover"_ustr);
  const PointCloud *points = geometry.get_pointcloud();
  if (points == nullptr) {
    params.error_message_add(NodeWarningType::Info,
                             TIP_("Crowd Ragdoll Switch: the Points input is not a point cloud"));
    params.set_output("Points"_ustr, std::move(geometry));
    params.set_output("Recover"_ustr, GeometrySet::from_pointcloud(BKE_pointcloud_new_nomain(0)));
    return;
  }

  const float time = params.extract_input<float>("Time"_ustr);
  const float recover_duration = std::max(params.extract_input<float>("Recover Duration"_ustr),
                                          1.0e-4f);
  const int max_dynamic = std::max(params.extract_input<int>("Max Dynamic"_ustr), 0);
  const float pending_timeout = std::max(params.extract_input<float>("Pending Timeout"_ustr),
                                         0.0f);
  const bool once = params.extract_input<bool>("Once"_ustr);
  const float cooldown = std::max(params.extract_input<float>("Cooldown"_ustr), 0.0f);

  const bke::GeometryFieldContext field_context(*points, AttrDomain::Point);
  fn::FieldEvaluator evaluator(field_context, points->totpoint);
  evaluator.add(params.extract_input<Field<bool>>("Condition"_ustr));
  evaluator.add(params.extract_input<Field<bool>>("Recover Condition"_ustr));
  evaluator.evaluate();
  const VArray<bool> conditions = evaluator.get_evaluated<bool>(0);
  const VArray<bool> recover_conditions = evaluator.get_evaluated<bool>(1);

  PointCloud *result = BKE_pointcloud_copy_for_eval(points);
  bke::MutableAttributeAccessor attributes = result->attributes_for_write();
  const AttrDomain domain = AttrDomain::Point;

  bke::SpanAttributeWriter<int> mode_writer = attributes.lookup_or_add_for_write_span<int>(
      "physics_mode", domain);
  bke::SpanAttributeWriter<float> recover_start_writer =
      attributes.lookup_or_add_for_write_span<float>("recover_start", domain);
  bke::SpanAttributeWriter<float> recover_duration_writer =
      attributes.lookup_or_add_for_write_span<float>("recover_duration", domain);
  bke::SpanAttributeWriter<float> recover_weight_writer =
      attributes.lookup_or_add_for_write_span<float>("recover_weight", domain);
  bke::SpanAttributeWriter<float> switch_time_writer =
      attributes.lookup_or_add_for_write_span<float>("ragdoll_switch_time", domain);
  bke::SpanAttributeWriter<bool> pending_writer = attributes.lookup_or_add_for_write_span<bool>(
      "ragdoll_pending", domain);
  bke::SpanAttributeWriter<float> pending_time_writer =
      attributes.lookup_or_add_for_write_span<float>("ragdoll_pending_time", domain);
  bke::SpanAttributeWriter<bool> once_done_writer = attributes.lookup_or_add_for_write_span<bool>(
      "ragdoll_once_done", domain);
  bke::SpanAttributeWriter<float> state_time_writer =
      attributes.lookup_or_add_for_write_span<float>("state_time", domain);
  bke::SpanAttributeWriter<MStringProperty> prev_state_writer =
      attributes.lookup_or_add_for_write_span<MStringProperty>("prev_state", domain);
  bke::SpanAttributeWriter<float3> position_writer =
      attributes.lookup_or_add_for_write_span<float3>("position", domain);
  bke::SpanAttributeWriter<math::Quaternion> rotation_writer =
      attributes.lookup_or_add_for_write_span<math::Quaternion>("rotation", domain);

  const bke::AttributeAccessor src_attributes = points->attributes();
  const VArray<int> agents = src_attributes.lookup_or_default<int>("agent_id", domain, -1).varray;
  const VArray<int> modes =
      src_attributes.lookup_or_default<int>("physics_mode", domain, 0).varray;
  const VArray<float> recover_starts =
      src_attributes.lookup_or_default<float>("recover_start", domain, 0.0f).varray;
  const VArray<float> recover_weights =
      src_attributes.lookup_or_default<float>("recover_weight", domain, 0.0f).varray;
  const VArray<float> switch_times =
      src_attributes.lookup_or_default<float>("ragdoll_switch_time", domain, -1.0e9f).varray;
  const VArray<bool> pendings =
      src_attributes.lookup_or_default<bool>("ragdoll_pending", domain, false).varray;
  const VArray<float> pending_times =
      src_attributes.lookup_or_default<float>("ragdoll_pending_time", domain, 0.0f).varray;
  const VArray<bool> once_dones =
      src_attributes.lookup_or_default<bool>("ragdoll_once_done", domain, false).varray;
  const VArray<float> state_times =
      src_attributes.lookup_or_default<float>("state_time", domain, 0.0f).varray;
  const Span<float3> src_positions = points->positions();
  const VArray<math::Quaternion> rotations =
      src_attributes
          .lookup_or_default<math::Quaternion>("rotation", domain, math::Quaternion::identity())
          .varray;

  const int count = points->totpoint;
  Array<int> state_modes(count);
  Array<float> weights(count, 0.0f);
  for (const int i : IndexRange(count)) {
    state_modes[i] = modes[i];
    weights[i] = recover_weights[i];
  }

  Vector<int> order(count);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
      order.begin(), order.end(), [&](const int a, const int b) { return agents[a] < agents[b]; });

  Vector<BoneSnapshot> previous_bones;
  read_bone_snapshots(previous_bones_geometry, previous_bones);
  Vector<BoneSnapshot> previous_recover;
  read_bone_snapshots(previous_recover_geometry, previous_recover);

  Map<int, const BoneSnapshot *> first_bone;
  for (const BoneSnapshot &bone : previous_bones) {
    if (bone.parent < 0 && first_bone.lookup_ptr(bone.agent_id) == nullptr) {
      first_bone.add(bone.agent_id, &bone);
    }
  }

  Set<int> recovering;
  Set<int> new_recover;
  Vector<int> promote_pending;
  Vector<int> new_triggers;
  int dynamic_count = 0;

  /* First pass: update running modes and collect candidates for the budget. */
  for (const int i : order) {
    const int mode = state_modes[i];
    if (mode == 2) {
      continue;
    }
    if (mode == 1) {
      dynamic_count++;
      if (recover_conditions[i]) {
        state_modes[i] = 3;
        recover_start_writer.span[i] = time;
        recover_duration_writer.span[i] = recover_duration;
        recover_weight_writer.span[i] = 0.0f;
        weights[i] = 0.0f;
        recovering.add(agents[i]);
        new_recover.add(agents[i]);
        pending_writer.span[i] = false;
        dynamic_count--;
      }
      continue;
    }
    if (mode == 3) {
      const float start = recover_starts[i];
      const float duration = std::max(recover_duration_writer.span[i], 1.0e-4f);
      float weight = math::clamp((time - start) / duration, 0.0f, 1.0f);
      weight = weight * weight * (3.0f - 2.0f * weight);
      if (weight >= 1.0f && recover_weights[i] >= 1.0f - 1.0e-5f) {
        state_modes[i] = 0;
        recover_weight_writer.span[i] = 0.0f;
        state_time_writer.span[i] = 0.0f;
        MStringProperty empty;
        prev_state_writer.span[i] = empty;
        weights[i] = 0.0f;
        continue;
      }
      state_modes[i] = 3;
      recover_weight_writer.span[i] = weight;
      weights[i] = weight;
      recovering.add(agents[i]);
      continue;
    }

    /* mode == 0 */
    if (pendings[i]) {
      if (time - pending_times[i] > pending_timeout) {
        pending_writer.span[i] = false;
      }
      else {
        promote_pending.append(i);
      }
    }
    if (conditions[i] && !pendings[i] && (!once || !once_dones[i]) &&
        (time - switch_times[i] >= cooldown))
    {
      new_triggers.append(i);
    }
  }

  /* Second pass: promote pending triggers first, then new ones, in agent order. */
  for (const int i : promote_pending) {
    if (max_dynamic != 0 && dynamic_count >= max_dynamic) {
      pending_writer.span[i] = true;
      continue;
    }
    state_modes[i] = 1;
    switch_time_writer.span[i] = time;
    pending_writer.span[i] = false;
    once_done_writer.span[i] = true;
    dynamic_count++;
  }
  for (const int i : new_triggers) {
    if (max_dynamic != 0 && dynamic_count >= max_dynamic) {
      pending_writer.span[i] = true;
      pending_time_writer.span[i] = time;
      continue;
    }
    state_modes[i] = 1;
    switch_time_writer.span[i] = time;
    pending_writer.span[i] = false;
    once_done_writer.span[i] = true;
    dynamic_count++;
  }

  /* Write the final modes and move a recovering point to the ragdoll root. */
  for (const int i : IndexRange(count)) {
    const int mode = state_modes[i];
    mode_writer.span[i] = mode;
    if (mode != 3) {
      recover_weight_writer.span[i] = mode == 0 ? 0.0f : weights[i];
    }
    if (mode != 1 && mode != 3) {
      continue;
    }
    if (mode == 3 && new_recover.contains(agents[i])) {
      const BoneSnapshot *const *bone = first_bone.lookup_ptr(agents[i]);
      if (bone != nullptr) {
        const BoneSnapshot &root = **bone;
        const float3 root_position = root.position;
        position_writer.span[i] = float3(root_position.x, root_position.y, src_positions[i].z);
        const math::Quaternion pose_rotation = math::normalize(math::to_quaternion(root.pose));
        const float3 forward = math::transform_point(pose_rotation, float3(0.0f, -1.0f, 0.0f));
        if (math::length_squared(float2(forward.x, forward.y)) > 1.0e-8f) {
          const float yaw = std::atan2(forward.x, -forward.y);
          rotation_writer.span[i] = math::to_quaternion(math::EulerXYZ(float3(0.0f, 0.0f, yaw)));
        }
      }
    }
  }

  PointCloud *recover = build_recover_geometry(
      previous_bones, previous_recover, recovering, new_recover);

  mode_writer.finish();
  recover_start_writer.finish();
  recover_duration_writer.finish();
  recover_weight_writer.finish();
  switch_time_writer.finish();
  pending_writer.finish();
  pending_time_writer.finish();
  once_done_writer.finish();
  state_time_writer.finish();
  prev_state_writer.finish();
  position_writer.finish();
  rotation_writer.finish();

  result->tag_positions_changed();
  params.set_output("Points"_ustr, GeometrySet::from_pointcloud(result));
  params.set_output("Recover"_ustr, GeometrySet::from_pointcloud(recover));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeCrowdRagdollSwitch"_ustr);
  ntype.ui_name = "Crowd Ragdoll Switch";
  ntype.ui_description =
      "Switch crowd agents between animation and ragdoll, with a recovery blend and a dynamic "
      "budget";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_crowd_ragdoll_switch_cc
