/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Blend Bones`: blend a sampled ragdoll pose (`From Bones`) into the current animation pose
 * (`To Bones`) for agents that are recovering. Every other agent passes the animation pose
 * through unchanged.
 */

#include <string>

#include "NOD_geometry_nodes_crowd.hh"

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_vector.hh"
#include "BLI_string_ref.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_pointcloud.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_blend_bones_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("From Bones"_ustr)
      .description("Ragdoll pose snapshot (usually the `Recover` output of Crowd Ragdoll Switch)");
  b.add_input<decl::Geometry>("To Bones"_ustr)
      .description("Current animation bones from Crowd Bones");
  b.add_input<decl::Geometry>("Points"_ustr)
      .description("Crowd points carrying `physics_mode` and `recover_weight`");

  b.add_output<decl::Geometry>("Bones"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet from_geometry = params.extract_input<GeometrySet>("From Bones"_ustr);
  GeometrySet to_geometry = params.extract_input<GeometrySet>("To Bones"_ustr);
  GeometrySet points_geometry = params.extract_input<GeometrySet>("Points"_ustr);

  const PointCloud *to_points = to_geometry.get_pointcloud();
  const PointCloud *from_points = from_geometry.get_pointcloud();
  const PointCloud *crowd_points = points_geometry.get_pointcloud();
  if (to_points == nullptr || crowd_points == nullptr) {
    params.set_output("Bones"_ustr, std::move(to_geometry));
    return;
  }

  const bke::AttributeAccessor to_attributes = to_points->attributes();
  const VArray<int> to_agents =
      to_attributes.lookup_or_default<int>("agent_id", AttrDomain::Point, -1).varray;
  const VArray<MStringProperty> to_names =
      to_attributes.lookup<MStringProperty>("bone_name").varray;
  const VArray<float4x4> to_poses = to_attributes
                                        .lookup_or_default<float4x4>("bone_pose_matrix",
                                                                     AttrDomain::Point,
                                                                     float4x4::identity())
                                        .varray;
  if (!to_names || to_poses.size() < to_points->totpoint) {
    params.set_output("Bones"_ustr, std::move(to_geometry));
    return;
  }

  /* The recovery snapshot is empty until the first agent starts recovering, so a missing (or
   * empty) `From Bones` is not an error: it only means there is nothing to blend from. The
   * `physics_mode` / `animated` flags still have to be written, otherwise the simulation would
   * keep every agent kinematic. */
  Map<int, Map<std::string, float4x4>> recover_poses;
  if (from_points != nullptr) {
    const bke::AttributeAccessor from_attributes = from_points->attributes();
    const VArray<int> from_agents =
        from_attributes.lookup_or_default<int>("agent_id", AttrDomain::Point, -1).varray;
    const VArray<MStringProperty> from_names =
        from_attributes.lookup<MStringProperty>("bone_name").varray;
    const VArray<float4x4> from_poses = from_attributes
                                            .lookup_or_default<float4x4>("bone_pose_matrix",
                                                                         AttrDomain::Point,
                                                                         float4x4::identity())
                                            .varray;
    if (from_names && from_poses.size() >= from_points->totpoint) {
      for (const int i : IndexRange(from_points->totpoint)) {
        Map<std::string, float4x4> &agent_poses = recover_poses.lookup_or_add(
            from_agents[i], Map<std::string, float4x4>());
        agent_poses.add_overwrite(crowd::read_string(from_names[i]), from_poses[i]);
      }
    }
  }

  const bke::AttributeAccessor crowd_attributes = crowd_points->attributes();
  const VArray<int> crowd_agents =
      crowd_attributes.lookup_or_default<int>("agent_id", AttrDomain::Point, -1).varray;
  const VArray<int> modes =
      crowd_attributes.lookup_or_default<int>("physics_mode", AttrDomain::Point, 0).varray;
  const VArray<float> weights =
      crowd_attributes.lookup_or_default<float>("recover_weight", AttrDomain::Point, 0.0f).varray;
  Map<int, int> agent_modes;
  Map<int, float> agent_weights;
  for (const int i : IndexRange(crowd_points->totpoint)) {
    agent_modes.add_overwrite(crowd_agents[i], modes[i]);
    agent_weights.add_overwrite(crowd_agents[i], weights[i]);
  }

  PointCloud *result = BKE_pointcloud_copy_for_eval(to_points);
  bke::MutableAttributeAccessor attributes = result->attributes_for_write();
  bke::SpanAttributeWriter<float4x4> pose_writer =
      attributes.lookup_or_add_for_write_span<float4x4>("bone_pose_matrix", AttrDomain::Point);
  bke::SpanAttributeWriter<float3> position_writer =
      attributes.lookup_or_add_for_write_span<float3>("position", AttrDomain::Point);
  bke::SpanAttributeWriter<int> mode_writer = attributes.lookup_or_add_for_write_span<int>(
      "physics_mode", AttrDomain::Point);
  bke::SpanAttributeWriter<float> recover_weight_writer =
      attributes.lookup_or_add_for_write_span<float>("recover_weight", AttrDomain::Point);
  bke::SpanAttributeWriter<bool> animated_writer = attributes.lookup_or_add_for_write_span<bool>(
      "animated", AttrDomain::Point);

  for (const int i : IndexRange(result->totpoint)) {
    const int agent = to_agents[i];
    const int *mode = agent_modes.lookup_ptr(agent);
    const float *weight = agent_weights.lookup_ptr(agent);
    if (mode != nullptr) {
      mode_writer.span[i] = *mode;
      animated_writer.span[i] = *mode != 1;
    }
    if (weight != nullptr) {
      recover_weight_writer.span[i] = *weight;
    }
    if (mode == nullptr || *mode != 3 || weight == nullptr) {
      continue;
    }
    const Map<std::string, float4x4> *agent_poses = recover_poses.lookup_ptr(agent);
    if (agent_poses == nullptr) {
      continue;
    }
    const float4x4 *from_pose = agent_poses->lookup_ptr(crowd::read_string(to_names[i]));
    if (from_pose == nullptr) {
      continue;
    }
    const float blend = math::clamp(*weight, 0.0f, 1.0f);
    float3 from_loc;
    float3 from_scale;
    math::Quaternion from_rot;
    float3 to_loc;
    float3 to_scale;
    math::Quaternion to_rot;
    math::to_loc_rot_scale<true>(*from_pose, from_loc, from_rot, from_scale);
    math::to_loc_rot_scale<true>(to_poses[i], to_loc, to_rot, to_scale);
    const float4x4 blended = math::from_loc_rot_scale<float4x4>(
        math::interpolate(from_loc, to_loc, blend),
        math::interpolate(from_rot, to_rot, blend),
        math::interpolate(from_scale, to_scale, blend));
    pose_writer.span[i] = blended;
    position_writer.span[i] = blended.location();
  }

  pose_writer.finish();
  position_writer.finish();
  mode_writer.finish();
  recover_weight_writer.finish();
  animated_writer.finish();
  result->tag_positions_changed();
  params.set_output("Bones"_ustr, GeometrySet::from_pointcloud(result));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeBlendBones"_ustr);
  ntype.ui_name = "Blend Bones";
  ntype.ui_description =
      "Blend a ragdoll bone pose into the animation pose for recovering crowd agents";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_blend_bones_cc
