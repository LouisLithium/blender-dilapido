/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Ragdoll Motors`: per-bone PD targets for an active ragdoll.
 *
 * The target orientation is the current armature pose. The solver applies
 * `torque = strength * angle * axis - damping * angular_velocity` every substep, so a body with a
 * positive strength tries to follow the animation while still colliding and being pushed. Strength
 * 0 leaves the body passive; the field inputs allow a per-bone `physics blend`, e.g. a mask that
 * only drives the spine and arms.
 */

#include <cstdint>

#include "BLI_array.hh"
#include "BLI_map.hh"#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_vector.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_fields.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_pointcloud.hh"

#include "FN_field_evaluation.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_ragdoll_motors_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Bones"_ustr).description("Point cloud from Armature Bones");
  b.add_input<decl::Geometry>("Bodies"_ustr)
      .description("Optional Init Ragdoll Bodies output, used to map bones to piece ids");
  b.add_input<decl::Float>("Strength"_ustr)
      .default_value(120.0f)
      .min(0.0f)
      .structure_type(StructureType::Field)
      .description("Angular stiffness in rad/s² per rad, scaled by the body's inertia; "
                   "0 leaves the bone passive");
  b.add_input<decl::Float>("Damping"_ustr)
      .default_value(10.0f)
      .min(0.0f)
      .structure_type(StructureType::Field)
      .description("Angular damping in 1/s, scaled by the body's inertia");
  b.add_input<decl::Float>("Max Torque"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .structure_type(StructureType::Field)
      .description("Torque ceiling in N m; 0 means no limit");
  b.add_input<decl::Bool>("Mask"_ustr)
      .default_value(true)
      .structure_type(StructureType::Field)
      .description("Disable the servo per bone (useful as a `physics blend` mask)");

  b.add_output<decl::Geometry>("Motors"_ustr);
}

/** Combine an agent id and a bone index into one key (both non-negative). */
static int64_t make_bone_key(const int agent, const int bone)
{
  return (int64_t(uint32_t(agent)) << 32) | uint32_t(bone);
}

static Map<int64_t, int> read_body_ids(const GeometrySet &bodies)
{
  Map<int64_t, int> ids;
  if (bodies.has_instances()) {
    const bke::Instances &instances = *bodies.get_instances();
    const bke::AttributeAccessor attributes = instances.attributes();
    const VArray<int> bone_indices =
        attributes.lookup_or_default<int>("bone_index", AttrDomain::Instance, -1).varray;
    const VArray<int> agent_ids =
        attributes.lookup_or_default<int>("agent_id", AttrDomain::Instance, 0).varray;
    const VArray<int> piece_ids =
        attributes.lookup_or_default<int>("piece_id", AttrDomain::Instance, -1).varray;
    for (const int i : IndexRange(instances.instances_num())) {
      if (bone_indices[i] >= 0 && piece_ids[i] >= 0) {
        ids.add(make_bone_key(agent_ids[i], bone_indices[i]), piece_ids[i]);
      }
    }
  }
  else if (bodies.has_pointcloud()) {
    const bke::AttributeAccessor attributes = bodies.get_pointcloud()->attributes();
    const VArray<int> bone_indices =
        attributes.lookup_or_default<int>("bone_index", AttrDomain::Point, -1).varray;
    const VArray<int> agent_ids =
        attributes.lookup_or_default<int>("agent_id", AttrDomain::Point, 0).varray;
    const VArray<int> piece_ids =
        attributes.lookup_or_default<int>("piece_id", AttrDomain::Point, -1).varray;
    for (const int i : IndexRange(bodies.get_pointcloud()->totpoint)) {
      if (bone_indices[i] >= 0 && piece_ids[i] >= 0) {
        ids.add(make_bone_key(agent_ids[i], bone_indices[i]), piece_ids[i]);
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
        TIP_("Ragdoll Motors: connect the Bones output of an Armature Bones node"));
    params.set_default_remaining_outputs();
    return;
  }

  GeometrySet bodies_geometry = params.extract_input<GeometrySet>("Bodies"_ustr);
  const Map<int64_t, int> body_ids = read_body_ids(bodies_geometry);

  const int count = bones->totpoint;
  const bke::AttributeAccessor attributes = bones->attributes();
  const VArray<float4x4> pose_matrices =
      attributes
          .lookup_or_default<float4x4>(
              "bone_pose_matrix", AttrDomain::Point, float4x4::identity())
          .varray;
  const VArray<int> bone_indices =
      attributes.lookup_or_default<int>("bone_index", AttrDomain::Point, -1).varray;
  const VArray<int> agent_ids =
      attributes.lookup_or_default<int>("agent_id", AttrDomain::Point, 0).varray;

  bke::GeometryFieldContext field_context(*bones, AttrDomain::Point);
  fn::FieldEvaluator evaluator(field_context, count);
  evaluator.add(params.extract_input<Field<float>>("Strength"_ustr));
  evaluator.add(params.extract_input<Field<float>>("Damping"_ustr));
  evaluator.add(params.extract_input<Field<float>>("Max Torque"_ustr));
  evaluator.add(params.extract_input<Field<bool>>("Mask"_ustr));
  evaluator.evaluate();
  const VArray<float> strengths = evaluator.get_evaluated<float>(0);
  const VArray<float> dampings = evaluator.get_evaluated<float>(1);
  const VArray<float> max_torques = evaluator.get_evaluated<float>(2);
  const VArray<bool> masks = evaluator.get_evaluated<bool>(3);

  PointCloud *pointcloud = BKE_pointcloud_new_nomain(count);
  bke::MutableAttributeAccessor out_attributes = pointcloud->attributes_for_write();
  const AttrDomain domain = AttrDomain::Point;
  bke::SpanAttributeWriter<float3> positions = out_attributes.lookup_or_add_for_write_span<float3>(
      "position", domain);
  bke::SpanAttributeWriter<int> piece_id_writer =
      out_attributes.lookup_or_add_for_write_span<int>("piece_id", domain);
  bke::SpanAttributeWriter<int> agent_id_writer =
      out_attributes.lookup_or_add_for_write_span<int>("agent_id", domain);
  bke::SpanAttributeWriter<math::Quaternion> target_writer =
      out_attributes.lookup_or_add_for_write_span<math::Quaternion>("target_rotation", domain);
  bke::SpanAttributeWriter<float> strength_writer =
      out_attributes.lookup_or_add_for_write_span<float>("strength", domain);
  bke::SpanAttributeWriter<float> damping_writer =
      out_attributes.lookup_or_add_for_write_span<float>("damping", domain);
  bke::SpanAttributeWriter<float> max_torque_writer =
      out_attributes.lookup_or_add_for_write_span<float>("max_torque", domain);
  bke::SpanAttributeWriter<bool> marker_writer =
      out_attributes.lookup_or_add_for_write_span<bool>("is_ragdoll_motor", domain);

  for (const int i : IndexRange(count)) {
    const float4x4 pose = pose_matrices[i];
    const int bone_index = bone_indices[i] >= 0 ? bone_indices[i] : i;
    const int agent = std::max(agent_ids[i], 0);
    const int64_t key = make_bone_key(agent, bone_index);
    const int piece_id = body_ids.contains(key) ? body_ids.lookup(key) : bone_index;
    positions.span[i] = pose.location();
    piece_id_writer.span[i] = masks[i] ? piece_id : -1;
    agent_id_writer.span[i] = agent;
    target_writer.span[i] = math::normalize(math::to_quaternion(pose));
    strength_writer.span[i] = masks[i] ? std::max(strengths[i], 0.0f) : 0.0f;
    damping_writer.span[i] = std::max(dampings[i], 0.0f);
    max_torque_writer.span[i] = std::max(max_torques[i], 0.0f);
    marker_writer.span[i] = true;
  }

  positions.finish();
  piece_id_writer.finish();
  agent_id_writer.finish();
  target_writer.finish();
  strength_writer.finish();
  damping_writer.finish();
  max_torque_writer.finish();
  marker_writer.finish();
  pointcloud->tag_positions_changed();
  params.set_output("Motors"_ustr, GeometrySet::from_pointcloud(pointcloud));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeRagdollMotors"_ustr);
  ntype.ui_name = "Ragdoll Motors";
  ntype.ui_description =
      "Per-bone PD targets that make a ragdoll follow the armature pose (active ragdoll)";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_ragdoll_motors_cc
