/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Skin From Bones`: deform a mesh with the transforms of simulated bone bodies.
 *
 * The mesh's vertex groups are matched to the bone names of the `Bones` input (an instance set from
 * `Init Ragdoll Bodies`, or the point cloud from `Armature Bones`). For every bone the deform
 * matrix is `pose * inverse(rest)`, i.e. the usual armature skinning relation, so a mesh that is
 * already rigged can be driven by the ragdoll without writing the pose back to the armature.
 *
 * Two modes: linear blend skinning (fast, can collapse on twists) and dual quaternion skinning
 * (keeps volume on twists, the default for ragdolls).
 */

#include <cmath>

#include "BLI_array.hh"
#include "BLI_listbase.h"
#include "BLI_listbase_iterator.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_vector.hh"
#include "BLI_string.h"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_deform.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_skin_from_bones_cc {

enum class SkinMode : int { Linear = 0, DualQuaternion = 1 };

static const EnumPropertyItem mode_items[] = {
    {int(SkinMode::DualQuaternion),
     "DUAL_QUATERNION",
     0,
     N_("Dual Quaternion"),
     N_("Dual quaternion skinning: keeps volume when bones twist")},
    {int(SkinMode::Linear),
     "LINEAR",
     0,
     N_("Linear"),
     N_("Linear blend skinning: cheap, but twists can collapse")},
    {0, nullptr, 0, nullptr, nullptr},
};

struct BoneTransform {
  std::string name;
  int agent = 0;
  float4x4 deform;
  math::Quaternion rotation;
  float3 translation;
  math::Quaternion dual;
};

static std::string string_from_property(const MStringProperty &property)
{
  return std::string(property.s, size_t(property.s_len));
}

/** Read the bone names, transforms and rest matrices from instances or a point cloud. */
static bool read_bones(const GeometrySet &bones,
                       Vector<BoneTransform> &r_bones,
                       std::string &r_error)
{
  if (bones.has_instances()) {
    const bke::Instances &instances = *bones.get_instances();
    const bke::AttributeAccessor attributes = instances.attributes();
    const VArray<MStringProperty> names = attributes.lookup<MStringProperty>("bone_name").varray;
    if (!names) {
      r_error = "Bones has no `bone_name` attribute (use Armature Bones or Init Ragdoll Bodies)";
      return false;
    }
    const VArray<int> agents =
        attributes.lookup_or_default<int>("agent_id", AttrDomain::Instance, 0).varray;
    const VArray<float4x4> rests =
        attributes
            .lookup_or_default<float4x4>(
                "bone_rest_matrix", AttrDomain::Instance, float4x4::identity())
            .varray;
    const Span<float4x4> transforms = instances.transforms();
    const int count = instances.instances_num();
    r_bones.resize(count);
    for (const int i : IndexRange(count)) {
      BoneTransform &bone = r_bones[i];
      bone.name = string_from_property(names[i]);
      bone.agent = agents[i];
      bone.deform = transforms[i] * math::invert(rests[i]);
      bone.rotation = math::normalize(math::to_quaternion(bone.deform));
      bone.translation = bone.deform.location();
      bone.dual = math::Quaternion(0.0f,
                                   bone.translation.x * 0.5f,
                                   bone.translation.y * 0.5f,
                                   bone.translation.z * 0.5f) *
                  bone.rotation;
    }
    return true;
  }
  if (bones.has_pointcloud()) {
    const bke::AttributeAccessor attributes = bones.get_pointcloud()->attributes();
    const VArray<MStringProperty> names = attributes.lookup<MStringProperty>("bone_name").varray;
    if (!names) {
      r_error = "Bones has no `bone_name` attribute (use Armature Bones)";
      return false;
    }
    const VArray<int> agents =
        attributes.lookup_or_default<int>("agent_id", AttrDomain::Point, 0).varray;
    const VArray<float4x4> poses =
        attributes
            .lookup_or_default<float4x4>(
                "bone_pose_matrix", AttrDomain::Point, float4x4::identity())
            .varray;
    const VArray<float4x4> rests =
        attributes
            .lookup_or_default<float4x4>(
                "bone_rest_matrix", AttrDomain::Point, float4x4::identity())
            .varray;
    const int count = bones.get_pointcloud()->totpoint;
    r_bones.resize(count);
    for (const int i : IndexRange(count)) {
      BoneTransform &bone = r_bones[i];
      bone.name = string_from_property(names[i]);
      bone.agent = agents[i];
      bone.deform = poses[i] * math::invert(rests[i]);
      bone.rotation = math::normalize(math::to_quaternion(bone.deform));
      bone.translation = bone.deform.location();
      bone.dual = math::Quaternion(0.0f,
                                   bone.translation.x * 0.5f,
                                   bone.translation.y * 0.5f,
                                   bone.translation.z * 0.5f) *
                  bone.rotation;
    }
    return true;
  }
  r_error = "Bones has to be the instance set from Init Ragdoll Bodies or the point cloud from "
            "Armature Bones";
  return false;
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Mesh"_ustr).description("Mesh to deform, with vertex groups named "
                                                       "after the bones");
  b.add_input<decl::Geometry>("Bones"_ustr).description("Bone transforms (instances or a bone "
                                                        "point cloud)");
  b.add_input<decl::Menu>("Mode"_ustr)
      .static_items(mode_items)
      .default_value(MenuValue(int(SkinMode::DualQuaternion)))
      .description("Skinning method");
  b.add_input<decl::Int>("Max Influences"_ustr)
      .default_value(4)
      .min(1)
      .max(8)
      .description("Maximum number of bone weights per vertex");
  b.add_input<decl::Bool>("Normalize Weights"_ustr)
      .default_value(true)
      .description("Renormalize the selected weights to sum to one");

  b.add_output<decl::Geometry>("Mesh"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet mesh_geometry = params.extract_input<GeometrySet>("Mesh"_ustr);
  const Mesh *mesh = mesh_geometry.get_mesh();
  if (!mesh) {
    params.error_message_add(NodeWarningType::Info,
                             TIP_("Skin From Bones: the Mesh input is not a mesh"));
    params.set_output("Mesh"_ustr, std::move(mesh_geometry));
    return;
  }
  GeometrySet bones_geometry = params.extract_input<GeometrySet>("Bones"_ustr);
  Vector<BoneTransform> bones;
  std::string error;
  if (!read_bones(bones_geometry, bones, error)) {
    params.error_message_add(NodeWarningType::Warning, error);
    params.set_output("Mesh"_ustr, std::move(mesh_geometry));
    return;
  }

  const int mode = params.extract_input<MenuValue>("Mode"_ustr).value;
  const int max_influences = math::clamp(params.extract_input<int>("Max Influences"_ustr), 1, 8);
  const bool normalize = params.extract_input<bool>("Normalize Weights"_ustr);

  Map<int, Map<std::string, int>> bones_by_agent;
  bones_by_agent.reserve(bones.size());
  for (const int i : bones.index_range()) {
    bones_by_agent.lookup_or_add_default(bones[i].agent).add(bones[i].name, i);
  }

  const bke::AttributeAccessor attributes = mesh->attributes();
  struct WeightChannel {
    int bone_index;
    VArray<float> weights;
  };
  const bool has_vertex_agents = attributes.contains("agent_id");
  const VArray<int> vertex_agents =
      attributes.lookup_or_default<int>("agent_id", AttrDomain::Point, 0).varray;

  /* Resolve every vertex to the bone *of its own agent*, so several characters can share bone
   * names (and one joined mesh) without their vertex groups colliding. */
  const int mesh_vertex_count = int(mesh->verts_num);
  Array<int> vertex_slot(mesh_vertex_count, 0);
  Map<int, int> agent_slots;
  for (const int vertex : IndexRange(mesh_vertex_count)) {
    const int agent = has_vertex_agents ? vertex_agents[vertex] : 0;
    const int *slot = agent_slots.lookup_ptr(agent);
    if (slot != nullptr) {
      vertex_slot[vertex] = *slot;
    }
    else {
      const int new_slot = int(agent_slots.size());
      agent_slots.add(agent, new_slot);
      vertex_slot[vertex] = new_slot;
    }
  }
  Vector<int> slot_agents(agent_slots.size());
  for (const auto item : agent_slots.items()) {
    slot_agents[item.value] = item.key;
  }

  Vector<Vector<WeightChannel>> channels_by_agent(agent_slots.size());
  int channel_count = 0;
  for (const int slot : slot_agents.index_range()) {
    const Map<std::string, int> *by_name = bones_by_agent.lookup_ptr(slot_agents[slot]);
    if (by_name == nullptr) {
      continue;
    }
    Vector<WeightChannel> &channels = channels_by_agent[slot];
    for (const bDeformGroup &group : mesh->vertex_group_names) {
      const int *bone_index = by_name->lookup_ptr(std::string(group.name));
      if (!bone_index) {
        continue;
      }
      const VArray<float> weights = attributes.lookup<float>(group.name, AttrDomain::Point).varray;
      if (!weights) {
        continue;
      }
      channels.append({*bone_index, weights});
    }
    channel_count += int(channels.size());
  }
  if (channel_count == 0) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Skin From Bones: no vertex group matches a bone name, returning the mesh unchanged"));
    params.set_output("Mesh"_ustr, std::move(mesh_geometry));
    return;
  }

  Mesh *result = BKE_mesh_copy_for_eval(*mesh);
  MutableSpan<float3> positions = result->vert_positions_for_write();
  const Span<float3> rest_positions = mesh->vert_positions();
  const int vertex_count = std::min(positions.size(), rest_positions.size());

  struct Influence {
    int bone;
    float weight;
  };
  Array<Influence> influences(max_influences);

  for (const int vertex : IndexRange(vertex_count)) {
    const Vector<WeightChannel> &channels = channels_by_agent[vertex_slot[vertex]];
    int count = 0;
    for (const WeightChannel &channel : channels) {
      const float weight = channel.weights[vertex];
      if (!(weight > 0.0f)) {
        continue;
      }
      if (count < max_influences) {
        influences[count++] = {channel.bone_index, weight};
      }
      else {
        /* Keep the largest influences: replace the smallest one if this weight is larger. */
        int smallest = 0;
        for (const int i : IndexRange::from_begin_end(1, count)) {
          if (influences[i].weight < influences[smallest].weight) {
            smallest = i;
          }
        }
        if (weight > influences[smallest].weight) {
          influences[smallest] = {channel.bone_index, weight};
        }
      }
    }
    if (count == 0) {
      continue;
    }
    float selected_total = 0.0f;
    for (const int i : IndexRange(count)) {
      selected_total += influences[i].weight;
    }
    if (!(selected_total > 0.0f)) {
      continue;
    }
    const float inv_total = normalize ? 1.0f / selected_total : 1.0f;
    const float3 rest = rest_positions[vertex];

    if (mode == int(SkinMode::Linear)) {
      float3 result_position(0.0f);
      for (const int i : IndexRange(count)) {
        const Influence &influence = influences[i];
        result_position += math::transform_point(bones[influence.bone].deform, rest) *
                           (influence.weight * inv_total);
      }
      positions[vertex] = result_position;
      continue;
    }

    /* Dual quaternion skinning. */
    const int reference = influences[0].bone;
    math::Quaternion real(0.0f, 0.0f, 0.0f, 0.0f);
    math::Quaternion dual(0.0f, 0.0f, 0.0f, 0.0f);
    for (const int i : IndexRange(count)) {
      const Influence &influence = influences[i];
      math::Quaternion bone_real = bones[influence.bone].rotation;
      math::Quaternion bone_dual = bones[influence.bone].dual;
      if (math::dot(bone_real, bones[reference].rotation) < 0.0f) {
        bone_real = bone_real * -1.0f;
        bone_dual = bone_dual * -1.0f;
      }
      const float weight = influence.weight * inv_total;
      real.w += bone_real.w * weight;
      real.x += bone_real.x * weight;
      real.y += bone_real.y * weight;
      real.z += bone_real.z * weight;
      dual.w += bone_dual.w * weight;
      dual.x += bone_dual.x * weight;
      dual.y += bone_dual.y * weight;
      dual.z += bone_dual.z * weight;
    }
    const float length = math::sqrt(math::dot(real, real));
    if (!(length > 1.0e-6f)) {
      continue;
    }
    real = real * (1.0f / length);
    dual = dual * (1.0f / length);
    const math::Quaternion translation_quat = dual * math::conjugate(real) * 2.0f;
    const float3 translation(translation_quat.x,
                             translation_quat.y,
                             translation_quat.z);
    const float3x3 rotation = math::from_rotation<float3x3>(real);
    positions[vertex] = math::transform_direction(rotation, rest) + translation;
  }

  result->tag_positions_changed();
  params.set_output("Mesh"_ustr, GeometrySet::from_mesh(result));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeSkinFromBones"_ustr);
  ntype.ui_name = "Skin From Bones";
  ntype.ui_description =
      "Deform a rigged mesh with simulated bone transforms (linear blend or dual quaternion)";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_skin_from_bones_cc

