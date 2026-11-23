/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Skin From Bones`: deform a mesh with the transforms of simulated bone bodies.
 *
 * The mesh's vertex groups are matched to the bone names of the `Bones` input (an instance set
 * from `Init Ragdoll Bodies`, or the point cloud from `Armature Bones`). For every bone the deform
 * matrix is `pose * inverse(rest)`, i.e. the usual armature skinning relation, so a mesh that is
 * already rigged can be driven by the ragdoll without writing the pose back to the armature.
 *
 * Two modes: linear blend skinning (fast, can collapse on twists) and dual quaternion skinning
 * (keeps volume on twists, the default for ragdolls).
 *
 * Performance notes:
 * - Vertex group weights are read from the mesh's `MDeformVert` data (the actual storage), so a
 *   vertex only visits its own few influences instead of every vertex group of the character.
 *   Going through `AttributeAccessor::lookup(name)` per group would scan each vertex's weights
 *   once **per bone**.
 * - The per-vertex loop runs in parallel; every vertex writes only its own position.
 * - The hot loop reads a compact `BoneSkin` array (no strings) to stay cache friendly.
 */

#include <array>
#include <cmath>

#include "BLI_array.hh"
#include "BLI_listbase.h"
#include "BLI_listbase_iterator.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_vector.hh"
#include "BLI_string.h"
#include "BLI_task.hh"
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

/** Compact per-bone data used in the per-vertex hot loop. */
struct BoneSkin {
  float4x4 deform;
  math::Quaternion rotation;
  math::Quaternion dual;
};

struct BoneTransform {
  std::string name;
  int agent = 0;
  BoneSkin skin;
};

struct Influence {
  int bone;
  float weight;
};

static constexpr int MAX_INFLUENCES = 8;

static std::string string_from_property(const MStringProperty &property)
{
  return std::string(property.s, size_t(property.s_len));
}

static void set_bone_skin_from_deform(BoneSkin &skin, const float4x4 &deform)
{
  skin.deform = deform;
  skin.rotation = math::normalize(math::to_quaternion(skin.deform));
  const float3 translation = skin.deform.location();
  skin.dual = math::Quaternion(
                  0.0f, translation.x * 0.5f, translation.y * 0.5f, translation.z * 0.5f) *
              skin.rotation;
}

static void fill_bone_skin(BoneTransform &bone, const float4x4 &pose, const float4x4 &rest)
{
  set_bone_skin_from_deform(bone.skin, pose * math::invert(rest));
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
    const VArray<float4x4> rests = attributes
                                       .lookup_or_default<float4x4>("bone_rest_matrix",
                                                                    AttrDomain::Instance,
                                                                    float4x4::identity())
                                       .varray;
    const Span<float4x4> transforms = instances.transforms();
    const int count = instances.instances_num();
    r_bones.resize(count);
    for (const int i : IndexRange(count)) {
      BoneTransform &bone = r_bones[i];
      bone.name = string_from_property(names[i]);
      bone.agent = agents[i];
      fill_bone_skin(bone, transforms[i], rests[i]);
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
    const VArray<float4x4> poses = attributes
                                       .lookup_or_default<float4x4>("bone_pose_matrix",
                                                                    AttrDomain::Point,
                                                                    float4x4::identity())
                                       .varray;
    const VArray<float4x4> rests = attributes
                                       .lookup_or_default<float4x4>("bone_rest_matrix",
                                                                    AttrDomain::Point,
                                                                    float4x4::identity())
                                       .varray;
    const int count = bones.get_pointcloud()->totpoint;
    r_bones.resize(count);
    for (const int i : IndexRange(count)) {
      BoneTransform &bone = r_bones[i];
      bone.name = string_from_property(names[i]);
      bone.agent = agents[i];
      fill_bone_skin(bone, poses[i], rests[i]);
    }
    return true;
  }
  r_error =
      "Bones has to be the instance set from Init Ragdoll Bodies or the point cloud from "
      "Armature Bones";
  return false;
}

/** Keep at most `max` influences, preferring the largest weights. */
static void add_influence(
    Influence *influences, int &r_count, const int max, const int bone, const float weight)
{
  if (!(weight > 0.0f)) {
    return;
  }
  if (r_count < max) {
    influences[r_count++] = {bone, weight};
    return;
  }
  int smallest = 0;
  for (const int i : IndexRange::from_begin_end(1, r_count)) {
    if (influences[i].weight < influences[smallest].weight) {
      smallest = i;
    }
  }
  if (weight > influences[smallest].weight) {
    influences[smallest] = {bone, weight};
  }
}

static void apply_skin(const Span<BoneSkin> &skins,
                       const int mode,
                       const Influence *influences,
                       const int count,
                       const float inv_total,
                       const float3 rest,
                       float3 &r_position)
{
  if (mode == int(SkinMode::Linear)) {
    float3 result_position(0.0f);
    for (const int i : IndexRange(count)) {
      const Influence &influence = influences[i];
      result_position += math::transform_point(skins[influence.bone].deform, rest) *
                         (influence.weight * inv_total);
    }
    r_position = result_position;
    return;
  }

  /* Dual quaternion skinning. */
  const math::Quaternion reference_rotation = skins[influences[0].bone].rotation;
  math::Quaternion real(0.0f, 0.0f, 0.0f, 0.0f);
  math::Quaternion dual(0.0f, 0.0f, 0.0f, 0.0f);
  for (const int i : IndexRange(count)) {
    const Influence &influence = influences[i];
    const BoneSkin &skin = skins[influence.bone];
    math::Quaternion bone_real = skin.rotation;
    math::Quaternion bone_dual = skin.dual;
    if (math::dot(bone_real, reference_rotation) < 0.0f) {
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
    return;
  }
  real = real * (1.0f / length);
  dual = dual * (1.0f / length);
  const math::Quaternion translation_quat = dual * math::conjugate(real) * 2.0f;
  const float3 translation(translation_quat.x, translation_quat.y, translation_quat.z);
  const float3x3 rotation = math::from_rotation<float3x3>(real);
  r_position = math::transform_direction(rotation, rest) + translation;
}

/** Weights resolved per agent: group index -> bone index, or -1 when the group is unused. */
struct AgentGroups {
  Array<int> group_to_bone;
  bool valid = false;
};

struct WeightChannel {
  int bone_index;
  VArray<float> weights;
  Span<const float> span;
};

/** All data needed to skin vertices of one template mesh. */
struct TemplateSkin {
  const Mesh *mesh = nullptr;
  Span<float3> rest_positions;
  Span<MDeformVert> deform_verts;
  Vector<AgentGroups> agent_groups;
  /** Generic-attribute fallback; only filled when the mesh has no MDeformVert data. */
  Vector<Vector<WeightChannel>> channels_by_slot;
  bool has_any_group = false;
};

static Vector<AgentGroups> build_agent_groups(
    const Mesh &mesh,
    const Map<int, Map<std::string, int>> &bones_by_agent,
    const Span<int> slot_agents,
    bool &r_has_any_group)
{
  const int group_count = BLI_listbase_count(&mesh.vertex_group_names);
  Vector<AgentGroups> agent_groups(slot_agents.size());
  for (const int slot : slot_agents.index_range()) {
    const Map<std::string, int> *by_name = bones_by_agent.lookup_ptr(slot_agents[slot]);
    if (by_name == nullptr) {
      continue;
    }
    AgentGroups &groups = agent_groups[slot];
    groups.group_to_bone = Array<int>(group_count, -1);
    int group_index = 0;
    for (const bDeformGroup &group : mesh.vertex_group_names) {
      const int *bone_index = by_name->lookup_ptr(std::string(group.name));
      if (bone_index != nullptr) {
        groups.group_to_bone[group_index] = *bone_index;
        groups.valid = true;
        r_has_any_group = true;
      }
      group_index++;
    }
  }
  return agent_groups;
}

static std::unique_ptr<TemplateSkin> build_template_skin(
    const Mesh &mesh,
    const Map<int, Map<std::string, int>> &bones_by_agent,
    const Span<int> slot_agents)
{
  auto tpl = std::make_unique<TemplateSkin>();
  tpl->mesh = &mesh;
  tpl->rest_positions = mesh.vert_positions();
  tpl->deform_verts = mesh.deform_verts();
  tpl->agent_groups = build_agent_groups(mesh, bones_by_agent, slot_agents, tpl->has_any_group);
  if (!tpl->has_any_group || tpl->deform_verts.size() >= tpl->rest_positions.size()) {
    return tpl;
  }

  /* Without MDeformVert data, resolve the generic attributes once per agent slot. */
  tpl->channels_by_slot.resize(tpl->agent_groups.size());
  const bke::AttributeAccessor attributes = mesh.attributes();
  Vector<StringRefNull> group_names;
  for (const bDeformGroup &group : mesh.vertex_group_names) {
    group_names.append(group.name);
  }
  for (const int slot : tpl->agent_groups.index_range()) {
    const AgentGroups &groups = tpl->agent_groups[slot];
    if (!groups.valid) {
      continue;
    }
    Vector<WeightChannel> &channels = tpl->channels_by_slot[slot];
    for (const int i : group_names.index_range()) {
      const int bone_index = groups.group_to_bone[i];
      if (bone_index < 0) {
        continue;
      }
      const VArray<float> weights =
          attributes.lookup<float>(group_names[i], AttrDomain::Point).varray;
      if (!weights) {
        continue;
      }
      WeightChannel channel;
      channel.bone_index = bone_index;
      channel.weights = weights;
      if (weights.is_span()) {
        const Span<float> span = weights.get_internal_span();
        if (span.size() >= tpl->rest_positions.size()) {
          channel.span = Span<const float>(span.data(), span.size());
        }
      }
      channels.append(channel);
    }
  }
  return tpl;
}

/** Deform one template mesh for a single agent slot (used by the instances path). */
static void deform_template_slot(const TemplateSkin &tpl,
                                 const Span<BoneSkin> &skins,
                                 const int slot,
                                 const int mode,
                                 const int max_influences,
                                 const bool normalize,
                                 const MutableSpan<float3> positions)
{
  const int vertex_count = std::min(int(tpl.rest_positions.size()), int(positions.size()));
  if (slot < 0 || slot >= tpl.agent_groups.size() || vertex_count == 0) {
    return;
  }
  const AgentGroups &groups = tpl.agent_groups[slot];
  if (!groups.valid) {
    return;
  }
  const bool use_deform_verts = tpl.deform_verts.size() >= tpl.rest_positions.size();
  const Vector<WeightChannel> *channels = (slot < tpl.channels_by_slot.size() &&
                                           !tpl.channels_by_slot[slot].is_empty()) ?
                                              &tpl.channels_by_slot[slot] :
                                              nullptr;
  if (!use_deform_verts && channels == nullptr) {
    return;
  }

  std::array<Influence, MAX_INFLUENCES> influences;
  for (const int vertex : IndexRange(vertex_count)) {
    int count = 0;
    if (use_deform_verts) {
      const MDeformVert &dvert = tpl.deform_verts[vertex];
      for (const MDeformWeight &weight : Span<MDeformWeight>(dvert.dw, dvert.totweight)) {
        const int group = weight.def_nr;
        if (group < 0 || group >= groups.group_to_bone.size()) {
          continue;
        }
        const int bone = groups.group_to_bone[group];
        if (bone < 0) {
          continue;
        }
        add_influence(influences.data(), count, max_influences, bone, weight.weight);
      }
    }
    else {
      for (const WeightChannel &channel : *channels) {
        const float weight = channel.span.is_empty() ? channel.weights[vertex] :
                                                       channel.span[vertex];
        add_influence(influences.data(), count, max_influences, channel.bone_index, weight);
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
    apply_skin(skins,
               mode,
               influences.data(),
               count,
               inv_total,
               tpl.rest_positions[vertex],
               positions[vertex]);
  }
}

static Array<BoneSkin> make_local_skins(const Span<BoneSkin> &world_skins,
                                        const float4x4 &transform)
{
  const float4x4 inverse = math::invert(transform);
  Array<BoneSkin> local_skins(world_skins.size());
  for (const int i : world_skins.index_range()) {
    set_bone_skin_from_deform(local_skins[i], inverse * world_skins[i].deform * transform);
  }
  return local_skins;
}

/**
 * Skin each instance of a template mesh and keep the result as instances.
 *
 * This avoids `Realize Instances`: every instance gets its own deformed mesh, but the vertex
 * weights, topology and other attributes are shared with the template instead of being copied
 * per agent. The node output therefore stays an `Instances` component.
 */
static void skin_instances(GeoNodeExecParams &params,
                           const GeometrySet &input,
                           const bke::Instances &instances,
                           const Span<BoneTransform> &bones,
                           const int mode,
                           const int max_influences,
                           const bool normalize)
{
  const int instance_count = instances.instances_num();
  if (instance_count == 0) {
    params.set_output("Mesh"_ustr, input);
    return;
  }

  Map<int, Map<std::string, int>> bones_by_agent;
  bones_by_agent.reserve(bones.size());
  for (const int i : bones.index_range()) {
    bones_by_agent.lookup_or_add_default(bones[i].agent).add(bones[i].name, i);
  }

  const VArray<int> instance_agents =
      instances.attributes().lookup_or_default<int>("agent_id", AttrDomain::Instance, 0).varray;

  Map<int, int> agent_slots;
  Vector<int> slot_agents;
  auto ensure_agent = [&](const int agent) {
    if (!agent_slots.contains(agent)) {
      agent_slots.add(agent, int(slot_agents.size()));
      slot_agents.append(agent);
    }
  };
  for (const BoneTransform &bone : bones) {
    ensure_agent(bone.agent);
  }
  for (const int i : IndexRange(instance_count)) {
    ensure_agent(instance_agents[i]);
  }

  Array<int> instance_slots(instance_count, -1);
  for (const int i : IndexRange(instance_count)) {
    if (const int *slot = agent_slots.lookup_ptr(instance_agents[i])) {
      instance_slots[i] = *slot;
    }
  }

  Array<BoneSkin> world_skins(bones.size());
  for (const int i : bones.index_range()) {
    world_skins[i] = bones[i].skin;
  }

  const Span<int> instance_handles = instances.reference_handles();
  const Span<bke::InstanceReference> references = instances.references();
  Map<int, std::unique_ptr<TemplateSkin>> templates;
  for (const int i : IndexRange(instance_count)) {
    const int handle = instance_handles[i];
    if (templates.contains(handle)) {
      continue;
    }
    if (handle < 0 || handle >= references.size()) {
      continue;
    }
    const bke::InstanceReference &reference = references[handle];
    if (reference.type() != bke::InstanceReference::Type::GeometrySet) {
      continue;
    }
    const Mesh *template_mesh = reference.geometry_set().get_mesh();
    if (template_mesh == nullptr) {
      continue;
    }
    templates.add(handle, build_template_skin(*template_mesh, bones_by_agent, slot_agents));
  }

  auto result = std::make_unique<bke::Instances>(instance_count);
  result->transforms_for_write().copy_from(instances.transforms());
  bke::copy_attributes(instances.attributes(),
                       AttrDomain::Instance,
                       AttrDomain::Instance,
                       bke::AttributeFilter{},
                       result->attributes_for_write());

  Array<Mesh *> deformed_meshes(instance_count, nullptr);
  threading::parallel_for(IndexRange(instance_count), 1, [&](const IndexRange range) {
    for (const int i : range) {
      const int handle = instance_handles[i];
      const std::unique_ptr<TemplateSkin> *template_ptr = templates.lookup_ptr(handle);
      if (template_ptr == nullptr || !template_ptr->get()->has_any_group) {
        continue;
      }
      const TemplateSkin &tpl = *template_ptr->get();
      const int slot = instance_slots[i];
      if (slot < 0 || slot >= tpl.agent_groups.size() || !tpl.agent_groups[slot].valid) {
        continue;
      }
      Mesh *deformed = BKE_mesh_copy_for_eval(*tpl.mesh);
      const Array<BoneSkin> local_skins = make_local_skins(world_skins, instances.transforms()[i]);
      deform_template_slot(tpl,
                           local_skins,
                           slot,
                           mode,
                           max_influences,
                           normalize,
                           deformed->vert_positions_for_write());
      deformed->tag_positions_changed();
      deformed_meshes[i] = deformed;
    }
  });

  bool skinned_any = false;
  MutableSpan<int> handles = result->reference_handles_for_write();
  for (const int i : IndexRange(instance_count)) {
    if (deformed_meshes[i] != nullptr) {
      const int new_handle = result->add_new_reference(
          bke::InstanceReference(GeometrySet::from_mesh(deformed_meshes[i])));
      handles[i] = new_handle;
      skinned_any = true;
    }
    else {
      const int original_handle = instance_handles[i];
      if (original_handle >= 0 && original_handle < references.size()) {
        handles[i] = result->add_reference(references[original_handle]);
      }
    }
  }

  if (!skinned_any) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Skin From Bones: no instance mesh has vertex groups matching a bone name, returning "
             "the instances unchanged"));
    params.set_output("Mesh"_ustr, input);
    return;
  }

  GeometrySet output = input;
  output.replace_instances(result.release(), bke::GeometryOwnershipType::Owned);
  params.set_output("Mesh"_ustr, std::move(output));
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Mesh"_ustr)
      .description(
          "Mesh or Instances to deform. With Instances, every instance is skinned into "
          "its own mesh, avoiding `Realize Instances`");
  b.add_input<decl::Geometry>("Bones"_ustr)
      .description(
          "Bone transforms (instances or a bone "
          "point cloud)");
  b.add_input<decl::Menu>("Mode"_ustr)
      .static_items(mode_items)
      .default_value(MenuValue(int(SkinMode::DualQuaternion)))
      .description("Skinning method");
  b.add_input<decl::Int>("Max Influences"_ustr)
      .default_value(4)
      .min(1)
      .max(MAX_INFLUENCES)
      .description("Maximum number of bone weights per vertex");
  b.add_input<decl::Bool>("Normalize Weights"_ustr)
      .default_value(true)
      .description("Renormalize the selected weights to sum to one");

  b.add_output<decl::Geometry>("Mesh"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet mesh_geometry = params.extract_input<GeometrySet>("Mesh"_ustr);
  GeometrySet bones_geometry = params.extract_input<GeometrySet>("Bones"_ustr);
  Vector<BoneTransform> bones;
  std::string error;
  if (!read_bones(bones_geometry, bones, error)) {
    params.error_message_add(NodeWarningType::Warning, error);
    params.set_output("Mesh"_ustr, std::move(mesh_geometry));
    return;
  }

  const int mode = params.extract_input<MenuValue>("Mode"_ustr).value;
  const int max_influences = math::clamp(
      params.extract_input<int>("Max Influences"_ustr), 1, MAX_INFLUENCES);
  const bool normalize = params.extract_input<bool>("Normalize Weights"_ustr);

  /* Input can be a mesh or an instance set. Keeping instances avoids `Realize Instances`; every
   * instance is skinned into its own mesh while topology and weights stay shared. */
  if (const bke::Instances *instances = mesh_geometry.get_instances()) {
    skin_instances(params, mesh_geometry, *instances, bones, mode, max_influences, normalize);
    return;
  }

  const Mesh *mesh = mesh_geometry.get_mesh();
  if (!mesh) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Skin From Bones: the Mesh input is neither a mesh nor an instance set"));
    params.set_output("Mesh"_ustr, std::move(mesh_geometry));
    return;
  }

  Map<int, Map<std::string, int>> bones_by_agent;
  bones_by_agent.reserve(bones.size());
  for (const int i : bones.index_range()) {
    bones_by_agent.lookup_or_add_default(bones[i].agent).add(bones[i].name, i);
  }

  const bke::AttributeAccessor attributes = mesh->attributes();
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

  /* One group-index -> bone-index table per agent. This is resolved once, not per vertex. */
  const int group_count = BLI_listbase_count(&mesh->vertex_group_names);
  Vector<AgentGroups> agent_groups(agent_slots.size());
  bool has_any_group = false;
  for (const int slot : slot_agents.index_range()) {
    const Map<std::string, int> *by_name = bones_by_agent.lookup_ptr(slot_agents[slot]);
    if (by_name == nullptr) {
      continue;
    }
    AgentGroups &groups = agent_groups[slot];
    groups.group_to_bone = Array<int>(group_count, -1);
    int group_index = 0;
    for (const bDeformGroup &group : mesh->vertex_group_names) {
      const int *bone_index = by_name->lookup_ptr(std::string(group.name));
      if (bone_index != nullptr) {
        groups.group_to_bone[group_index] = *bone_index;
        groups.valid = true;
        has_any_group = true;
      }
      group_index++;
    }
  }
  if (!has_any_group) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_(
            "Skin From Bones: no vertex group matches a bone name, returning the mesh unchanged"));
    params.set_output("Mesh"_ustr, std::move(mesh_geometry));
    return;
  }

  Array<BoneSkin> skins(bones.size());
  for (const int i : bones.index_range()) {
    skins[i] = bones[i].skin;
  }

  Mesh *result = BKE_mesh_copy_for_eval(*mesh);
  MutableSpan<float3> positions = result->vert_positions_for_write();
  const Span<float3> rest_positions = mesh->vert_positions();
  const int vertex_count = std::min(positions.size(), rest_positions.size());

  const float3 *rest_data = rest_positions.data();
  float3 *position_data = positions.data();
  const AgentGroups *agent_data = agent_groups.data();
  const int *slot_data = vertex_slot.data();
  const BoneSkin *skin_data = skins.data();
  const Span<BoneSkin> skin_span(skin_data, skins.size());
  const int mode_value = mode;

  auto deform_vertex = [&](const int vertex, const Influence *influences, const int count) {
    if (count == 0) {
      return;
    }
    float selected_total = 0.0f;
    for (const int i : IndexRange(count)) {
      selected_total += influences[i].weight;
    }
    if (!(selected_total > 0.0f)) {
      return;
    }
    const float inv_total = normalize ? 1.0f / selected_total : 1.0f;
    apply_skin(skin_span,
               mode_value,
               influences,
               count,
               inv_total,
               rest_data[vertex],
               position_data[vertex]);
  };

  /* Fast path: read the actual deform weights, so each vertex visits only its own influences. */
  const Span<MDeformVert> deform_verts = mesh->deform_verts();
  if (deform_verts.size() >= vertex_count && vertex_count > 0) {
    const MDeformVert *deform_data = deform_verts.data();
    threading::parallel_for(IndexRange(vertex_count), 1024, [&](const IndexRange range) {
      for (const int vertex : range) {
        const AgentGroups &groups = agent_data[slot_data[vertex]];
        if (!groups.valid) {
          continue;
        }
        const Array<int> &group_to_bone = groups.group_to_bone;
        if (group_to_bone.is_empty()) {
          continue;
        }
        const MDeformVert &dvert = deform_data[vertex];
        std::array<Influence, MAX_INFLUENCES> influences;
        int count = 0;
        for (const MDeformWeight &weight : Span<MDeformWeight>(dvert.dw, dvert.totweight)) {
          const int group = weight.def_nr;
          if (group < 0 || group >= group_to_bone.size()) {
            continue;
          }
          const int bone = group_to_bone[group];
          if (bone < 0) {
            continue;
          }
          add_influence(influences.data(), count, max_influences, bone, weight.weight);
        }
        deform_vertex(vertex, influences.data(), count);
      }
    });
    result->tag_positions_changed();
    params.set_output("Mesh"_ustr, GeometrySet::from_mesh(result));
    return;
  }

  /* Fallback: weights are generic float attributes. Build one channel list per agent, reusing
   * the group->bone table above. */
  struct WeightChannel {
    int bone_index;
    VArray<float> weights;
    Span<const float> span;
  };
  Vector<Vector<WeightChannel>> channels_by_agent(agent_slots.size());
  int channel_count = 0;
  for (const int slot : slot_agents.index_range()) {
    const AgentGroups &groups = agent_groups[slot];
    if (!groups.valid) {
      continue;
    }
    Vector<WeightChannel> &channels = channels_by_agent[slot];
    int group_index = 0;
    for (const bDeformGroup &group : mesh->vertex_group_names) {
      const int bone_index = groups.group_to_bone[group_index++];
      if (bone_index < 0) {
        continue;
      }
      const VArray<float> weights = attributes.lookup<float>(group.name, AttrDomain::Point).varray;
      if (!weights) {
        continue;
      }
      WeightChannel channel;
      channel.bone_index = bone_index;
      channel.weights = weights;
      if (weights.is_span()) {
        const Span<float> span = weights.get_internal_span();
        if (span.size() >= vertex_count) {
          channel.span = Span<const float>(span.data(), span.size());
        }
      }
      channels.append(channel);
    }
    channel_count += int(channels.size());
  }
  if (channel_count == 0) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_(
            "Skin From Bones: no vertex group matches a bone name, returning the mesh unchanged"));
    params.set_output("Mesh"_ustr, std::move(mesh_geometry));
    return;
  }

  const Vector<WeightChannel> *channels_data = channels_by_agent.data();
  threading::parallel_for(IndexRange(vertex_count), 1024, [&](const IndexRange range) {
    for (const int vertex : range) {
      const int slot = slot_data[vertex];
      const Vector<WeightChannel> &channels = channels_data[slot];
      if (channels.is_empty()) {
        continue;
      }
      std::array<Influence, MAX_INFLUENCES> influences;
      int count = 0;
      for (const WeightChannel &channel : channels) {
        const float weight = channel.span.is_empty() ? channel.weights[vertex] :
                                                       channel.span[vertex];
        add_influence(influences.data(), count, max_influences, channel.bone_index, weight);
      }
      deform_vertex(vertex, influences.data(), count);
    }
  });

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
