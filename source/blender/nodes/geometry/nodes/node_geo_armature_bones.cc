/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Armature Bones`: read an armature's bone hierarchy into a point cloud.
 *
 * `Bone Info` can only read one named bone, which is enough for a lookup but not for building a
 * whole ragdoll. This node emits one point per bone (optionally only deform bones) with the rest
 * and pose matrices, the parent index and the bone length, so the ragdoll nodes can turn the
 * hierarchy into rigid bodies and joints without any Python-side setup.
 *
 * Matrices and positions are in world space (the armature object transform is applied). The
 * `Space` menu switches the pose data to the rest pose, which is useful when the ragdoll should
 * start at the bind pose instead of the current animation frame.
 */

#include "BLI_listbase.h"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_string.h"
#include "BLI_string_utf8.h"
#include "BLI_vector.hh"

#include "BKE_armature.hh"
#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_object.hh"
#include "BKE_pointcloud.hh"

#include "DNA_action_types.h"
#include "DNA_armature_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_armature_bones_cc {

enum class BoneSpace : int { Pose = 0, Rest = 1 };

static const EnumPropertyItem space_items[] = {
    {int(BoneSpace::Pose), "POSE", 0, N_("Pose"), N_("Use the current evaluated pose of the bones")},
    {int(BoneSpace::Rest),
     "REST",
     0,
     N_("Rest"),
     N_("Use the rest (edit mode) pose, so the ragdoll starts at the bind pose")},
    {0, nullptr, 0, nullptr, nullptr},
};

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Object>("Armature"_ustr).description("Armature object to read the bones from");
  b.add_input<decl::Bool>("Deform Only"_ustr)
      .default_value(true)
      .description("Only output bones that deform the mesh (control and helper bones are skipped)");
  b.add_input<decl::Menu>("Space"_ustr)
      .static_items(space_items)
      .default_value(MenuValue(int(BoneSpace::Pose)));

  b.add_output<decl::Geometry>("Bones"_ustr);
}

static void write_bone_name(MStringProperty &dst, const StringRef name)
{
  const size_t name_size = size_t(name.size());
  const size_t size = std::min(name_size, sizeof(dst.s));
  memcpy(dst.s, name.data(), size);
  dst.s_len = char(size);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  Object *object = params.extract_input<Object *>("Armature"_ustr);
  const bool deform_only = params.extract_input<bool>("Deform Only"_ustr);
  const int space = params.extract_input<MenuValue>("Space"_ustr).value;

  if (!object) {
    params.set_default_remaining_outputs();
    return;
  }
  if (object->type != OB_ARMATURE) {
    params.error_message_add(NodeWarningType::Error, TIP_("Object is not an armature"));
    params.set_default_remaining_outputs();
    return;
  }
  bPose *pose = object->pose;
  if (!pose) {
    params.error_message_add(NodeWarningType::Error, TIP_("Object has no pose"));
    params.set_default_remaining_outputs();
    return;
  }

  const float4x4 object_matrix = object->object_to_world();

  Vector<const bPoseChannel *> selected;
  selected.reserve(64);
  for (bPoseChannel &pchan : pose->chanbase) {
    const Bone *bone = pchan.bone_get(*object);
    if (!bone) {
      continue;
    }
    if (deform_only && (bone->flag & BONE_NO_DEFORM)) {
      continue;
    }
    selected.append(&pchan);
  }

  const int count = int(selected.size());
  PointCloud *pointcloud = BKE_pointcloud_new_nomain(count);
  if (count == 0) {
    params.set_output("Bones"_ustr, GeometrySet::from_pointcloud(pointcloud));
    return;
  }

  Map<const Bone *, int> index_of_bone;
  index_of_bone.reserve(count);
  for (const int i : selected.index_range()) {
    index_of_bone.add(selected[i]->bone_get(*object), i);
  }

  bke::MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
  const AttrDomain domain = AttrDomain::Point;
  bke::SpanAttributeWriter<float3> positions = attributes.lookup_or_add_for_write_span<float3>(
      "position", domain);
  bke::SpanAttributeWriter<int> bone_indices = attributes.lookup_or_add_for_write_span<int>(
      "bone_index", domain);
  bke::SpanAttributeWriter<int> bone_parents = attributes.lookup_or_add_for_write_span<int>(
      "bone_parent", domain);
  bke::SpanAttributeWriter<bool> bone_connected = attributes.lookup_or_add_for_write_span<bool>(
      "bone_connected", domain);
  bke::SpanAttributeWriter<bool> bone_deform = attributes.lookup_or_add_for_write_span<bool>(
      "bone_deform", domain);
  bke::SpanAttributeWriter<float> bone_lengths = attributes.lookup_or_add_for_write_span<float>(
      "bone_length", domain);
  bke::SpanAttributeWriter<float3> bone_heads = attributes.lookup_or_add_for_write_span<float3>(
      "bone_head", domain);
  bke::SpanAttributeWriter<float3> bone_tails = attributes.lookup_or_add_for_write_span<float3>(
      "bone_tail", domain);
  bke::SpanAttributeWriter<float4x4> bone_rest_matrices =
      attributes.lookup_or_add_for_write_span<float4x4>("bone_rest_matrix", domain);
  bke::SpanAttributeWriter<float4x4> bone_pose_matrices =
      attributes.lookup_or_add_for_write_span<float4x4>("bone_pose_matrix", domain);
  bke::SpanAttributeWriter<MStringProperty> bone_names =
      attributes.lookup_or_add_for_write_span<MStringProperty>("bone_name", domain);

  const bool use_rest = (space == int(BoneSpace::Rest));

  for (const int i : selected.index_range()) {
    const bPoseChannel *pchan = selected[i];
    const Bone *bone = pchan->bone_get(*object);
    const float4x4 rest_matrix = object_matrix * float4x4(bone->arm_mat);
    const float4x4 pose_matrix = use_rest ? rest_matrix :
                                            object_matrix * float4x4(pchan->pose_mat);
    const float3 head = pose_matrix.location();
    const float3 tail = math::transform_point(pose_matrix, float3(0.0f, bone->length, 0.0f));

    positions.span[i] = head;
    bone_indices.span[i] = i;
    bone_lengths.span[i] = bone->length;
    bone_heads.span[i] = head;
    bone_tails.span[i] = tail;
    bone_rest_matrices.span[i] = rest_matrix;
    bone_pose_matrices.span[i] = pose_matrix;
    bone_connected.span[i] = (bone->flag & BONE_CONNECTED) != 0;
    bone_deform.span[i] = (bone->flag & BONE_NO_DEFORM) == 0;
    write_bone_name(bone_names.span[i], bone->name);

    int parent_index = -1;
    for (const Bone *parent = bone->parent; parent != nullptr; parent = parent->parent) {
      const int *found = index_of_bone.lookup_ptr(parent);
      if (found) {
        parent_index = *found;
        break;
      }
    }
    bone_parents.span[i] = parent_index;
  }

  positions.finish();
  bone_indices.finish();
  bone_parents.finish();
  bone_connected.finish();
  bone_deform.finish();
  bone_lengths.finish();
  bone_heads.finish();
  bone_tails.finish();
  bone_rest_matrices.finish();
  bone_pose_matrices.finish();
  bone_names.finish();

  pointcloud->tag_positions_changed();
  params.set_output("Bones"_ustr, GeometrySet::from_pointcloud(pointcloud));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeArmatureBones"_ustr);
  ntype.ui_name = "Armature Bones";
  ntype.ui_description =
      "Read the bone hierarchy of an armature into a point cloud (rest and pose matrices, parent "
      "indices and lengths)";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_armature_bones_cc
