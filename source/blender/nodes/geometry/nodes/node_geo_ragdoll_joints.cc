/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Ragdoll Joints`: build the constraint point cloud of a bone hierarchy.
 *
 * Every bone with a parent gets one joint at its head. The anchors and the joint frames are
 * expressed in the two bodies' local space and captured from the current (bind) pose, so the
 * angular limits of a swing-twist or hinge joint are measured from that relative pose. That is what
 * makes a ragdoll out of a welded chain: the existing fixed/soft constraints lock all three
 * rotation axes.
 */

#include <cmath>

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_vector.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_pointcloud.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_ragdoll_joints_cc {

enum class JointType : int {
  Fixed = 0,
  SwingTwist = 2,
  Hinge = 3,
  Cone = 4,
  Distance = 5,
};

static const EnumPropertyItem joint_type_items[] = {
    {int(JointType::SwingTwist),
     "SWING_TWIST",
     0,
     N_("Swing Twist"),
     N_("Ball joint with a cone and a twist limit (the usual ragdoll joint)")},
    {int(JointType::Hinge),
     "HINGE",
     0,
     N_("Hinge"),
     N_("One rotation axis with angular limits (knees, elbows)")},
    {int(JointType::Cone),
     "CONE",
     0,
     N_("Cone"),
     N_("Point joint with a single cone limit")},
    {int(JointType::Distance),
     "DISTANCE",
     0,
     N_("Distance"),
     N_("Keep the two joint points at a distance")},
    {int(JointType::Fixed), "FIXED", 0, N_("Fixed"), N_("Weld the two bones (no articulation)")},
    {0, nullptr, 0, nullptr, nullptr},
};

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Bones"_ustr).description("Point cloud from Armature Bones");
  b.add_input<decl::Geometry>("Bodies"_ustr)
      .description("Optional Init Ragdoll Bodies output, used to map bones to piece ids");
  b.add_input<decl::Menu>("Joint Type"_ustr)
      .static_items(joint_type_items)
      .default_value(MenuValue(int(JointType::SwingTwist)))
      .description("Default joint type; per-bone overrides can be stored as "
                   "`ragdoll_joint_type` on the bones");
  b.add_input<decl::Float>("Swing Limit"_ustr)
      .default_value(float(M_PI) * 0.25f)
      .min(0.0f)
      .max(float(M_PI))
      .subtype(PROP_ANGLE)
      .description("Half cone angle of a swing-twist or cone joint");
  b.add_input<decl::Float>("Twist Limit"_ustr)
      .default_value(float(M_PI) / 6.0f)
      .min(0.0f)
      .max(float(M_PI))
      .subtype(PROP_ANGLE)
      .description("Symmetric twist limit of a swing-twist joint");
  b.add_input<decl::Float>("Hinge Min"_ustr)
      .default_value(-0.5f * float(M_PI))
      .min(-float(M_PI))
      .max(float(M_PI))
      .subtype(PROP_ANGLE);
  b.add_input<decl::Float>("Hinge Max"_ustr)
      .default_value(0.5f * float(M_PI))
      .min(-float(M_PI))
      .max(float(M_PI))
      .subtype(PROP_ANGLE);
  b.add_input<decl::Vector>("Hinge Axis"_ustr)
      .default_value(float3(1.0f, 0.0f, 0.0f))
      .description("Hinge axis in the joint frame (the child bone's local space)");
  b.add_input<decl::Float>("Distance Min"_ustr).default_value(0.0f).min(0.0f);
  b.add_input<decl::Float>("Distance Max"_ustr).default_value(0.0f).min(0.0f);
  b.add_input<decl::Bool>("Disable Adjacent Collision"_ustr)
      .default_value(true)
      .description("Let the two bones pass through each other (they share the joint)");
  b.add_input<decl::Float>("Break Impulse"_ustr).default_value(0.0f).min(0.0f);
  b.add_input<decl::Float>("Break Distance"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .subtype(PROP_DISTANCE);
  b.add_input<decl::Float>("Break Angle"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .subtype(PROP_ANGLE);

  b.add_output<decl::Geometry>("Constraints"_ustr);
}

/** Return the piece id of every bone index in a ragdoll body set. */
static Map<int, int> read_body_ids(const GeometrySet &bodies)
{
  Map<int, int> ids;
  if (bodies.has_instances()) {
    const bke::Instances &instances = *bodies.get_instances();
    const bke::AttributeAccessor attributes = instances.attributes();
    const VArray<int> bone_indices =
        attributes.lookup_or_default<int>("bone_index", AttrDomain::Instance, -1).varray;
    const VArray<int> piece_ids =
        attributes.lookup_or_default<int>("piece_id", AttrDomain::Instance, -1).varray;
    for (const int i : IndexRange(instances.instances_num())) {
      if (bone_indices[i] >= 0 && piece_ids[i] >= 0) {
        ids.add(bone_indices[i], piece_ids[i]);
      }
    }
  }
  else if (bodies.has_pointcloud()) {
    const bke::AttributeAccessor attributes = bodies.get_pointcloud()->attributes();
    const VArray<int> bone_indices =
        attributes.lookup_or_default<int>("bone_index", AttrDomain::Point, -1).varray;
    const VArray<int> piece_ids =
        attributes.lookup_or_default<int>("piece_id", AttrDomain::Point, -1).varray;
    for (const int i : IndexRange(bodies.get_pointcloud()->totpoint)) {
      if (bone_indices[i] >= 0 && piece_ids[i] >= 0) {
        ids.add(bone_indices[i], piece_ids[i]);
      }
    }
  }
  return ids;
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet bones_geometry = params.extract_input<GeometrySet>("Bones"_ustr);
  const PointCloud *bones = bones_geometry.get_pointcloud();
  if (!bones || bones->totpoint == 0) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Ragdoll Joints: connect the Bones output of an Armature Bones node"));
    params.set_default_remaining_outputs();
    return;
  }

  GeometrySet bodies_geometry = params.extract_input<GeometrySet>("Bodies"_ustr);
  const Map<int, int> body_ids = read_body_ids(bodies_geometry);

  const int default_type = params.extract_input<MenuValue>("Joint Type"_ustr).value;
  const float default_swing = std::max(params.extract_input<float>("Swing Limit"_ustr), 0.0f);
  const float default_twist = std::max(params.extract_input<float>("Twist Limit"_ustr), 0.0f);
  const float default_hinge_min = params.extract_input<float>("Hinge Min"_ustr);
  const float default_hinge_max = params.extract_input<float>("Hinge Max"_ustr);
  const float3 default_hinge_axis = params.extract_input<float3>("Hinge Axis"_ustr);
  const float default_distance_min = std::max(params.extract_input<float>("Distance Min"_ustr),
                                               0.0f);
  const float default_distance_max = std::max(params.extract_input<float>("Distance Max"_ustr),
                                               default_distance_min);
  const bool disable_collision = params.extract_input<bool>("Disable Adjacent Collision"_ustr);
  const float break_impulse = std::max(params.extract_input<float>("Break Impulse"_ustr), 0.0f);
  const float break_distance = std::max(params.extract_input<float>("Break Distance"_ustr), 0.0f);
  const float break_angle = std::max(params.extract_input<float>("Break Angle"_ustr), 0.0f);

  const int count = bones->totpoint;
  const bke::AttributeAccessor attributes = bones->attributes();
  const VArray<float4x4> pose_matrices =
      attributes
          .lookup_or_default<float4x4>(
              "bone_pose_matrix", AttrDomain::Point, float4x4::identity())
          .varray;
  const VArray<int> bone_indices =
      attributes.lookup_or_default<int>("bone_index", AttrDomain::Point, -1).varray;
  const VArray<int> bone_parents =
      attributes.lookup_or_default<int>("bone_parent", AttrDomain::Point, -1).varray;

  const VArray<int> joint_types =
      attributes.lookup_or_default<int>("ragdoll_joint_type", AttrDomain::Point, default_type)
          .varray;
  const VArray<float> swing_limits =
      attributes.lookup_or_default<float>("ragdoll_swing_limit", AttrDomain::Point, default_swing)
          .varray;
  const VArray<float> twist_limits =
      attributes.lookup_or_default<float>("ragdoll_twist_limit", AttrDomain::Point, default_twist)
          .varray;
  const VArray<float> hinge_mins =
      attributes.lookup_or_default<float>("ragdoll_hinge_min", AttrDomain::Point, default_hinge_min)
          .varray;
  const VArray<float> hinge_maxs =
      attributes.lookup_or_default<float>("ragdoll_hinge_max", AttrDomain::Point, default_hinge_max)
          .varray;
  const VArray<float3> hinge_axes =
      attributes
          .lookup_or_default<float3>("ragdoll_hinge_axis", AttrDomain::Point, default_hinge_axis)
          .varray;
  const VArray<float> distance_mins =
      attributes
          .lookup_or_default<float>("ragdoll_distance_min", AttrDomain::Point, default_distance_min)
          .varray;
  const VArray<float> distance_maxs =
      attributes
          .lookup_or_default<float>("ragdoll_distance_max", AttrDomain::Point, default_distance_max)
          .varray;

  Map<int, int> index_of_bone;
  for (const int i : IndexRange(count)) {
    const int bone_index = bone_indices[i] >= 0 ? bone_indices[i] : i;
    index_of_bone.add(bone_index, i);
  }

  Vector<int> joint_bones;
  Vector<int> joint_parents;
  joint_bones.reserve(count);
  joint_parents.reserve(count);
  for (const int i : IndexRange(count)) {
    if (bone_parents[i] < 0) {
      continue;
    }
    if (!index_of_bone.contains(bone_parents[i])) {
      continue;
    }
    joint_bones.append(i);
    joint_parents.append(index_of_bone.lookup(bone_parents[i]));
  }

  const int joint_count = int(joint_bones.size());
  PointCloud *pointcloud = BKE_pointcloud_new_nomain(joint_count);
  if (joint_count == 0) {
    params.set_output("Constraints"_ustr, GeometrySet::from_pointcloud(pointcloud));
    return;
  }

  bke::MutableAttributeAccessor out_attributes = pointcloud->attributes_for_write();
  const AttrDomain domain = AttrDomain::Point;
  bke::SpanAttributeWriter<float3> positions = out_attributes.lookup_or_add_for_write_span<float3>(
      "position", domain);
  bke::SpanAttributeWriter<int> body_a_writer =
      out_attributes.lookup_or_add_for_write_span<int>("body_a", domain);
  bke::SpanAttributeWriter<int> body_b_writer =
      out_attributes.lookup_or_add_for_write_span<int>("body_b", domain);
  bke::SpanAttributeWriter<bool> enabled_writer =
      out_attributes.lookup_or_add_for_write_span<bool>("enabled", domain);
  bke::SpanAttributeWriter<int> type_writer =
      out_attributes.lookup_or_add_for_write_span<int>("type", domain);
  bke::SpanAttributeWriter<float3> anchor_a_writer =
      out_attributes.lookup_or_add_for_write_span<float3>("anchor_a", domain);
  bke::SpanAttributeWriter<float3> anchor_b_writer =
      out_attributes.lookup_or_add_for_write_span<float3>("anchor_b", domain);
  bke::SpanAttributeWriter<math::Quaternion> frame_a_writer =
      out_attributes.lookup_or_add_for_write_span<math::Quaternion>("frame_a", domain);
  bke::SpanAttributeWriter<math::Quaternion> frame_b_writer =
      out_attributes.lookup_or_add_for_write_span<math::Quaternion>("frame_b", domain);
  bke::SpanAttributeWriter<float> swing_writer =
      out_attributes.lookup_or_add_for_write_span<float>("swing_limit", domain);
  bke::SpanAttributeWriter<float> twist_writer =
      out_attributes.lookup_or_add_for_write_span<float>("twist_limit", domain);
  bke::SpanAttributeWriter<float> hinge_min_writer =
      out_attributes.lookup_or_add_for_write_span<float>("hinge_min", domain);
  bke::SpanAttributeWriter<float> hinge_max_writer =
      out_attributes.lookup_or_add_for_write_span<float>("hinge_max", domain);
  bke::SpanAttributeWriter<float3> hinge_axis_writer =
      out_attributes.lookup_or_add_for_write_span<float3>("hinge_axis", domain);
  bke::SpanAttributeWriter<float> distance_min_writer =
      out_attributes.lookup_or_add_for_write_span<float>("distance_min", domain);
  bke::SpanAttributeWriter<float> distance_max_writer =
      out_attributes.lookup_or_add_for_write_span<float>("distance_max", domain);
  bke::SpanAttributeWriter<bool> disable_collision_writer =
      out_attributes.lookup_or_add_for_write_span<bool>("disable_collision", domain);
  bke::SpanAttributeWriter<float> break_impulse_writer =
      out_attributes.lookup_or_add_for_write_span<float>("break_impulse", domain);
  bke::SpanAttributeWriter<float> break_distance_writer =
      out_attributes.lookup_or_add_for_write_span<float>("break_distance", domain);
  bke::SpanAttributeWriter<float> break_angle_writer =
      out_attributes.lookup_or_add_for_write_span<float>("break_angle", domain);
  bke::SpanAttributeWriter<int> fallback_writer =
      out_attributes.lookup_or_add_for_write_span<int>("fallback", domain);
  bke::SpanAttributeWriter<float> frequency_writer =
      out_attributes.lookup_or_add_for_write_span<float>("frequency", domain);
  bke::SpanAttributeWriter<float> damping_writer =
      out_attributes.lookup_or_add_for_write_span<float>("damping", domain);
  bke::SpanAttributeWriter<float> harden_time_writer =
      out_attributes.lookup_or_add_for_write_span<float>("harden_time", domain);
  bke::SpanAttributeWriter<bool> is_joint_writer =
      out_attributes.lookup_or_add_for_write_span<bool>("is_ragdoll_joint", domain);

  for (const int joint : IndexRange(joint_count)) {
    const int child = joint_bones[joint];
    const int parent = joint_parents[joint];
    const float4x4 child_pose = pose_matrices[child];
    const float4x4 parent_pose = pose_matrices[parent];
    const math::Quaternion child_rotation = math::normalize(math::to_quaternion(child_pose));
    const math::Quaternion parent_rotation = math::normalize(math::to_quaternion(parent_pose));
    const float3 joint_position = child_pose.location();
    const float3 parent_head = parent_pose.location();
    const float3 child_head = child_pose.location();

    const float3x3 parent_inverse = math::from_rotation<float3x3>(
        math::conjugate(parent_rotation));
    const float3x3 child_inverse = math::from_rotation<float3x3>(
        math::conjugate(child_rotation));
    const math::Quaternion frame_a = math::conjugate(parent_rotation) * child_rotation;

    const int child_bone = bone_indices[child] >= 0 ? bone_indices[child] : child;
    const int parent_bone = bone_indices[parent] >= 0 ? bone_indices[parent] : parent;
    const int child_id = body_ids.contains(child_bone) ? body_ids.lookup(child_bone) : child_bone;
    const int parent_id = body_ids.contains(parent_bone) ? body_ids.lookup(parent_bone) :
                                                           parent_bone;

    positions.span[joint] = joint_position;
    body_a_writer.span[joint] = parent_id;
    body_b_writer.span[joint] = child_id;
    enabled_writer.span[joint] = true;
    type_writer.span[joint] = joint_types[child];
    anchor_a_writer.span[joint] = math::transform_direction(parent_inverse,
                                                             joint_position - parent_head);
    anchor_b_writer.span[joint] = math::transform_direction(child_inverse,
                                                             joint_position - child_head);
    frame_a_writer.span[joint] = frame_a;
    frame_b_writer.span[joint] = math::Quaternion::identity();
    swing_writer.span[joint] = std::max(swing_limits[child], 0.0f);
    twist_writer.span[joint] = std::max(twist_limits[child], 0.0f);
    hinge_min_writer.span[joint] = hinge_mins[child];
    hinge_max_writer.span[joint] = hinge_maxs[child];
    hinge_axis_writer.span[joint] = hinge_axes[child];
    distance_min_writer.span[joint] = std::max(distance_mins[child], 0.0f);
    distance_max_writer.span[joint] = std::max(distance_maxs[child],
                                               distance_min_writer.span[joint]);
    disable_collision_writer.span[joint] = disable_collision;
    break_impulse_writer.span[joint] = break_impulse;
    break_distance_writer.span[joint] = break_distance;
    break_angle_writer.span[joint] = break_angle;
    fallback_writer.span[joint] = 0;
    frequency_writer.span[joint] = 5.0f;
    damping_writer.span[joint] = 0.5f;
    harden_time_writer.span[joint] = 0.0f;
    is_joint_writer.span[joint] = true;
  }

  positions.finish();
  body_a_writer.finish();
  body_b_writer.finish();
  enabled_writer.finish();
  type_writer.finish();
  anchor_a_writer.finish();
  anchor_b_writer.finish();
  frame_a_writer.finish();
  frame_b_writer.finish();
  swing_writer.finish();
  twist_writer.finish();
  hinge_min_writer.finish();
  hinge_max_writer.finish();
  hinge_axis_writer.finish();
  distance_min_writer.finish();
  distance_max_writer.finish();
  disable_collision_writer.finish();
  break_impulse_writer.finish();
  break_distance_writer.finish();
  break_angle_writer.finish();
  fallback_writer.finish();
  frequency_writer.finish();
  damping_writer.finish();
  harden_time_writer.finish();
  is_joint_writer.finish();

  pointcloud->tag_positions_changed();
  params.set_output("Constraints"_ustr, GeometrySet::from_pointcloud(pointcloud));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeRagdollJoints"_ustr);
  ntype.ui_name = "Ragdoll Joints";
  ntype.ui_description =
      "Build swing-twist, hinge, cone or distance joints from a bone hierarchy";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_ragdoll_joints_cc
