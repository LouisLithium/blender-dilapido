/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Crowd Bones`: evaluate the per-state action once per agent and emit the posed bone cloud for the
 * whole crowd.
 *
 * C2b-1 handles a single state per agent (no cross-fade yet). The action is evaluated through the
 * animation system (`animrig::evaluate_action`) against the template armature, so only the action's
 * F-curves are used; bone constraints/drivers are not evaluated. Each agent's `phase` offsets the
 * clip in seconds/frames, and the point transform replaces the template armature root.
 *
 * Output point attributes match `Skin From Bones` / `Init Ragdoll Bodies`:
 * `position`, `bone_pose_matrix`, `bone_rest_matrix`, `bone_index`, `bone_parent`, `bone_length`,
 * `bone_name`, `agent_id`, `template_id`.
 */

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include <fmt/format.h>

#include "ANIM_evaluation.hh"

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_euler_types.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_rotation.hh"
#include "BLI_math_vector.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "BKE_action.hh"
#include "BKE_animsys.h"
#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_lib_id.hh"
#include "BKE_pointcloud.hh"

#include "DEG_depsgraph_query.hh"

#include "DNA_action_types.h"
#include "DNA_anim_types.h"
#include "DNA_armature_types.h"
#include "DNA_meshdata_types.h"
#include "DNA_object_types.h"

#include "RNA_access.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_crowd_bones_cc {

struct TemplateInfo {
  int template_id = 0;
  const PointCloud *rest = nullptr;
  const Bundle *states = nullptr;
};

struct BoneSample {
  float3 position;
  float4x4 pose;
  float4x4 rest;
  int bone_index;
  int bone_parent;
  float length;
  MStringProperty name;
  int agent_id;
  int template_id;
  bool animated;
};

static void read_templates(const Bundle &agents, Vector<TemplateInfo> &r_templates)
{
  const int count = agents.lookup_path<int>("count").value_or(0);
  for (const int i : IndexRange(count)) {
    const std::optional<BundlePtr> agent = agents.lookup_path<BundlePtr>(
        "agent_" + std::to_string(i));
    if (!agent || !*agent) {
      continue;
    }
    const std::optional<GeometrySet> bones = (*agent)->lookup_path<GeometrySet>("bones");
    if (!bones || !bones->get_pointcloud()) {
      continue;
    }
    TemplateInfo info;
    info.template_id = (*agent)->lookup_path<int>("template_id").value_or(i);
    info.rest = bones->get_pointcloud();
    const std::optional<BundlePtr> states = (*agent)->lookup_path<BundlePtr>("states");
    info.states = states ? states->get() : nullptr;
    r_templates.append(info);
  }
}

static const Bundle *state_by_index(const Bundle *states, const int state_id)
{
  if (states == nullptr || state_id < 0) {
    return nullptr;
  }
  const std::optional<BundlePtr> state = states->lookup_path<BundlePtr>(
      "state_" + std::to_string(state_id));
  return state ? state->get() : nullptr;
}

/**
 * Resolve the state by name first (the documented semantic), then fall back to the state table
 * index. Templates may carry their own table; otherwise the node-level fallback table is used.
 */
static const Bundle *find_state(const TemplateInfo &templ,
                                const Bundle *fallback_states,
                                const std::string &name,
                                const int state_id)
{
  const Bundle *states = templ.states != nullptr ? templ.states : fallback_states;
  if (states == nullptr) {
    return nullptr;
  }
  if (!name.empty()) {
    const int count = states->lookup_path<int>("count").value_or(0);
    for (const int i : IndexRange(count)) {
      const std::optional<BundlePtr> state = states->lookup_path<BundlePtr>(
          "state_" + std::to_string(i));
      if (!state || !*state) {
        continue;
      }
      if ((*state)->lookup_path<std::string>("name").value_or("") == name) {
        return state->get();
      }
    }
  }
  return state_by_index(states, state_id);
}

static float channel_value(const animrig::EvaluationResult &result,
                           const std::string &path,
                           const int index,
                           const float fallback)
{
  const animrig::PropIdentifier key(StringRefNull(path.c_str()), index);
  const animrig::AnimatedProperty *property = result.lookup_ptr(key);
  return property != nullptr ? property->value : fallback;
}

static math::Quaternion channel_rotation(const animrig::EvaluationResult &result,
                                         const Object *armature,
                                         const std::string &bone_name)
{
  const bPoseChannel *pchan = armature != nullptr ?
                                  BKE_pose_channel_find_name(armature->pose, bone_name.c_str()) :
                                  nullptr;
  const std::string path = fmt::format("pose.bones[\"{}\"].", bone_name);
  const short rotmode = pchan != nullptr ? pchan->rotmode : ROT_MODE_QUAT;
  if (rotmode == ROT_MODE_QUAT) {
    /* DNA stores w, x, y, z; math::Quaternion takes the same (w, x, y, z) order. */
    const math::Quaternion quat(channel_value(result, path + "rotation_quaternion", 0, 1.0f),
                                channel_value(result, path + "rotation_quaternion", 1, 0.0f),
                                channel_value(result, path + "rotation_quaternion", 2, 0.0f),
                                channel_value(result, path + "rotation_quaternion", 3, 0.0f));
    return math::normalize(quat);
  }
  /* C2b-1 supports quaternion and XYZ Euler channels. Other orders use the rest orientation. */
  if (rotmode == ROT_MODE_XYZ) {
    const float3 euler(channel_value(result, path + "rotation_euler", 0, 0.0f),
                       channel_value(result, path + "rotation_euler", 1, 0.0f),
                       channel_value(result, path + "rotation_euler", 2, 0.0f));
    return math::normalize(math::to_quaternion(math::EulerXYZ(euler)));
  }
  return math::Quaternion::identity();
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Bundle>("Agents"_ustr).description("Agent set from Join Agents");
  b.add_input<decl::Bundle>("States"_ustr)
      .description("Fallback state table when the template does not carry one");
  b.add_input<decl::Geometry>("Instances"_ustr)
      .description("Points from Init Crowd Data");
  b.add_input<decl::Float>("Time"_ustr)
      .default_value(0.0f)
      .description("Animation time in seconds (e.g. Scene Time > Seconds)");

  b.add_output<decl::Geometry>("Bones"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  BundlePtr agents = params.extract_input<BundlePtr>("Agents"_ustr);
  BundlePtr fallback_states = params.extract_input<BundlePtr>("States"_ustr);
  GeometrySet instances_geometry = params.extract_input<GeometrySet>("Instances"_ustr);
  const float time = params.extract_input<float>("Time"_ustr);

  const PointCloud *instances = instances_geometry.get_pointcloud();
  if (!instances || !agents) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Crowd Bones: connect Agents (Join Agents) and Instances (Init Crowd Data)"));
    params.set_output("Bones"_ustr, GeometrySet::from_pointcloud(BKE_pointcloud_new_nomain(0)));
    return;
  }

  Vector<TemplateInfo> templates;
  read_templates(*agents, templates);
  if (templates.is_empty()) {
    params.error_message_add(NodeWarningType::Warning,
                             TIP_("Crowd Bones: the Agents input has no templates"));
    params.set_output("Bones"_ustr, GeometrySet::from_pointcloud(BKE_pointcloud_new_nomain(0)));
    return;
  }
  Map<int, const TemplateInfo *> template_by_id;
  for (const TemplateInfo &templ : templates) {
    template_by_id.add(templ.template_id, &templ);
  }

  const bke::AttributeAccessor attributes = instances->attributes();
  const VArray<int> template_ids =
      attributes.lookup_or_default<int>("template_id", AttrDomain::Point, -1).varray;
  const VArray<int> agent_ids =
      attributes.lookup_or_default<int>("agent_id", AttrDomain::Point, -1).varray;
  const VArray<int> state_ids =
      attributes.lookup_or_default<int>("state_id", AttrDomain::Point, -1).varray;
  const VArray<MStringProperty> state_names_attr =
      attributes.lookup<MStringProperty>("state").varray;
  const VArray<float> phases =
      attributes.lookup_or_default<float>("phase", AttrDomain::Point, 0.0f).varray;
  const VArray<float> speeds =
      attributes.lookup_or_default<float>("speed", AttrDomain::Point, 1.0f).varray;
  const VArray<float> scales =
      attributes.lookup_or_default<float>("scale", AttrDomain::Point, 1.0f).varray;
  const VArray<bool> actives =
      attributes.lookup_or_default<bool>("active", AttrDomain::Point, true).varray;
  const VArray<int> physics_modes =
      attributes.lookup_or_default<int>("physics_mode", AttrDomain::Point, 0).varray;
  const VArray<math::Quaternion> rotations = attributes
                                                  .lookup_or_default<math::Quaternion>(
                                                      "rotation",
                                                      AttrDomain::Point,
                                                      math::Quaternion::identity())
                                                  .varray;
  const Span<float3> positions = instances->positions();

  Main *bmain = DEG_get_bmain(params.depsgraph());
  Vector<BoneSample> samples;

  for (const int point : IndexRange(instances->totpoint)) {
    if (!actives[point]) {
      continue;
    }
    const TemplateInfo *const *templ_ptr = template_by_id.lookup_ptr(template_ids[point]);
    if (templ_ptr == nullptr) {
      continue;
    }
    const TemplateInfo &templ = **templ_ptr;
    const PointCloud &rest = *templ.rest;
    const int bone_count = rest.totpoint;
    if (bone_count == 0) {
      continue;
    }

    /* Resolve the state's animation metadata. */
    const std::string state_name = state_names_attr ? crowd::read_string(state_names_attr[point]) :
                                                      "";
    const Bundle *state = find_state(templ,
                                     fallback_states ? fallback_states.get() : nullptr,
                                     state_name,
                                     state_ids[point]);
    const Bundle *animation = nullptr;
    if (state != nullptr) {
      const std::optional<BundlePtr> animation_ptr = state->lookup_path<BundlePtr>("animation");
      animation = animation_ptr ? animation_ptr->get() : nullptr;
    }
    const std::string armature_name = animation != nullptr ?
                                          animation->lookup_path<std::string>("armature_name")
                                              .value_or("") :
                                          "";
    const std::string action_name = animation != nullptr ?
                                        animation->lookup_path<std::string>("action_name")
                                            .value_or("") :
                                        "";
    float frame_start = animation != nullptr ?
                            animation->lookup_path<float>("frame_start").value_or(1.0f) :
                            1.0f;
    float frame_end = animation != nullptr ?
                          animation->lookup_path<float>("frame_end").value_or(frame_start + 1.0f) :
                          frame_start + 1.0f;
    const float fps = animation != nullptr ?
                          animation->lookup_path<float>("fps").value_or(24.0f) :
                          24.0f;
    const float anim_speed = animation != nullptr ?
                                 animation->lookup_path<float>("speed").value_or(1.0f) :
                                 1.0f;
    const float anim_phase = animation != nullptr ?
                                 animation->lookup_path<float>("phase").value_or(0.0f) :
                                 0.0f;
    const bool anim_loop = animation != nullptr ?
                               animation->lookup_path<bool>("loop").value_or(true) :
                               true;
    const bool root_motion = animation != nullptr &&
                             animation->lookup_path<int>("root_motion").value_or(0) != 0;
    /* The state may override the clip's playback settings. */
    const float state_speed = state != nullptr ?
                                  state->lookup_path<float>("speed").value_or(1.0f) :
                                  1.0f;
    const bool state_loop = state != nullptr ?
                                state->lookup_path<bool>("loop").value_or(anim_loop) :
                                anim_loop;
    const float loop_speed_scale = std::max(anim_speed, 0.0f) * std::max(state_speed, 0.0f);

    Object *armature = armature_name.empty() ?
                           nullptr :
                           (Object *)BKE_libblock_find_name(
                               bmain, ID_OB, armature_name.c_str());
    bAction *action = action_name.empty() ?
                          nullptr :
                          (bAction *)BKE_libblock_find_name(bmain, ID_AC, action_name.c_str());
    if (action == nullptr && armature != nullptr && armature->adt != nullptr) {
      action = armature->adt->action;
    }

    animrig::EvaluationResult evaluated;
    float action_frame = frame_start;
    if (action != nullptr && armature != nullptr && armature->adt != nullptr) {
      const float clip_frames = std::max(frame_end - frame_start, 1.0f);
      const float phase = phases[point] + anim_phase;
      float time_frames = (time * fps * loop_speed_scale * std::max(speeds[point], 0.0f)) +
                          phase * clip_frames;
      if (state_loop) {
        time_frames = std::fmod(time_frames, clip_frames);
        if (time_frames < 0.0f) {
          time_frames += clip_frames;
        }
      }
      else {
        time_frames = math::clamp(time_frames, 0.0f, clip_frames);
      }
      action_frame = frame_start + time_frames;
      /* The action may not be the one currently assigned to the armature, so its slot handle can
       * differ. Prefer the armature's slot when it belongs to this action, else the first object
       * slot of the action. */
      animrig::slot_handle_t slot_handle = armature->adt->slot_handle;
      animrig::Action &action_ref = action->wrap();
      if (action_ref.slot_for_handle(slot_handle) == nullptr) {
        slot_handle = animrig::Slot::unassigned;
        for (const animrig::Slot *slot : action_ref.slots()) {
          if (slot->idtype == ID_OB) {
            slot_handle = slot->handle;
            break;
          }
        }
        if (slot_handle == animrig::Slot::unassigned && !action_ref.slots().is_empty()) {
          slot_handle = action_ref.slots()[0]->handle;
        }
      }
      if (slot_handle != animrig::Slot::unassigned) {
        PointerRNA armature_ptr = RNA_id_pointer_create(&armature->id);
        const AnimationEvalContext eval_context = BKE_animsys_eval_context_construct(
            const_cast<Depsgraph *>(params.depsgraph()), action_frame);
        evaluated = animrig::evaluate_action(armature_ptr, action_ref, slot_handle, eval_context);
      }
    }

    /* Rest skeleton data. */
    const bke::AttributeAccessor rest_attributes = rest.attributes();
    const VArray<MStringProperty> names =
        rest_attributes.lookup<MStringProperty>("bone_name").varray;
    const VArray<int> bone_indices =
        rest_attributes.lookup_or_default<int>("bone_index", AttrDomain::Point, -1).varray;
    const VArray<int> bone_parents =
        rest_attributes.lookup_or_default<int>("bone_parent", AttrDomain::Point, -1).varray;
    const VArray<float> lengths =
        rest_attributes.lookup_or_default<float>("bone_length", AttrDomain::Point, 1.0f).varray;
    const VArray<float4x4> rest_matrices = rest_attributes
                                               .lookup_or_default<float4x4>(
                                                   "bone_rest_matrix",
                                                   AttrDomain::Point,
                                                   float4x4::identity())
                                               .varray;
    if (!names) {
      continue;
    }

    Map<int, int> index_of_bone;
    for (const int i : IndexRange(bone_count)) {
      index_of_bone.add(bone_indices[i] >= 0 ? bone_indices[i] : i, i);
    }
    Vector<int> order;
    order.reserve(bone_count);
    Array<bool> done(bone_count, false);
    for (const int pass : IndexRange(bone_count)) {
      bool progress = false;
      for (const int i : IndexRange(bone_count)) {
        if (done[i]) {
          continue;
        }
        const int parent_bone = bone_parents[i];
        const int *parent_point = parent_bone >= 0 ? index_of_bone.lookup_ptr(parent_bone) :
                                                     nullptr;
        if (parent_bone < 0 || (parent_point != nullptr && done[*parent_point])) {
          order.append(i);
          done[i] = true;
          progress = true;
        }
      }
      if (!progress) {
        break;
      }
    }
    for (const int i : IndexRange(bone_count)) {
      if (!done[i]) {
        order.append(i);
      }
    }

    const float scale = std::max(scales[point], 1.0e-4f);
    const float4x4 point_transform = math::from_loc_rot_scale<float4x4>(
        positions[point], rotations ? math::normalize(rotations[point]) : math::Quaternion::identity(),
        float3(scale));
    const int root_bone = order.is_empty() ? 0 : order[0];
    const float4x4 root_rest = rest_matrices[root_bone];
    /* The point is the spawn anchor; the bone channels carry the animation relative to rest. */
    const float4x4 instance_root = point_transform * math::invert(root_rest);

    Vector<float4x4> pose_matrices(bone_count, float4x4::identity());
    Vector<int> sample_index(bone_count, -1);
    for (const int i : order) {
      const std::string bone_name = crowd::read_string(names[i]);
      const std::string path = fmt::format("pose.bones[\"{}\"].", bone_name);
      /* In Place: ignore the animated translation of root bones (the point anchors the crowd). */
      const bool strip_root_motion = !root_motion && bone_parents[i] < 0;
      const float3 location = strip_root_motion ?
                                  float3(0.0f) :
                                  float3(channel_value(evaluated, path + "location", 0, 0.0f),
                                         channel_value(evaluated, path + "location", 1, 0.0f),
                                         channel_value(evaluated, path + "location", 2, 0.0f));
      const float3 size(channel_value(evaluated, path + "scale", 0, 1.0f),
                        channel_value(evaluated, path + "scale", 1, 1.0f),
                        channel_value(evaluated, path + "scale", 2, 1.0f));
      const math::Quaternion rotation = channel_rotation(evaluated, armature, bone_name);
      const float4x4 chan = math::from_loc_rot_scale<float4x4>(location, rotation, size);

      const int parent_bone = bone_parents[i];
      const int *parent_point = parent_bone >= 0 ? index_of_bone.lookup_ptr(parent_bone) :
                                                   nullptr;
      const bool has_parent = parent_point != nullptr && sample_index[*parent_point] >= 0;
      const float4x4 rest_local = has_parent ?
                                      math::invert(rest_matrices[*parent_point]) * rest_matrices[i] :
                                      float4x4::identity();
      const float4x4 pose = has_parent ?
                                pose_matrices[*parent_point] * rest_local * chan :
                                instance_root * rest_matrices[i] * chan;

      pose_matrices[i] = pose;
      sample_index[i] = samples.size();
      samples.append({});
      BoneSample &sample = samples.last();
      sample.position = pose.location();
      sample.pose = pose;
      sample.rest = instance_root * rest_matrices[i];
      sample.bone_index = bone_indices[i] >= 0 ? bone_indices[i] : i;
      sample.bone_parent = parent_bone;
      sample.length = lengths[i];
      sample.name = names[i];
      sample.agent_id = agent_ids[point];
      sample.template_id = templ.template_id;
      sample.animated = physics_modes[point] == 0;
    }
  }

  PointCloud *result = BKE_pointcloud_new_nomain(samples.size());
  bke::MutableAttributeAccessor out_attributes = result->attributes_for_write();
  const AttrDomain domain = AttrDomain::Point;
  bke::SpanAttributeWriter<float3> positions_writer =
      out_attributes.lookup_or_add_for_write_span<float3>("position", domain);
  bke::SpanAttributeWriter<float4x4> pose_writer =
      out_attributes.lookup_or_add_for_write_span<float4x4>("bone_pose_matrix", domain);
  bke::SpanAttributeWriter<float4x4> rest_writer =
      out_attributes.lookup_or_add_for_write_span<float4x4>("bone_rest_matrix", domain);
  bke::SpanAttributeWriter<int> bone_index_writer =
      out_attributes.lookup_or_add_for_write_span<int>("bone_index", domain);
  bke::SpanAttributeWriter<int> bone_parent_writer =
      out_attributes.lookup_or_add_for_write_span<int>("bone_parent", domain);
  bke::SpanAttributeWriter<float> length_writer =
      out_attributes.lookup_or_add_for_write_span<float>("bone_length", domain);
  bke::SpanAttributeWriter<MStringProperty> name_writer =
      out_attributes.lookup_or_add_for_write_span<MStringProperty>("bone_name", domain);
  bke::SpanAttributeWriter<int> agent_writer =
      out_attributes.lookup_or_add_for_write_span<int>("agent_id", domain);
  bke::SpanAttributeWriter<int> template_writer =
      out_attributes.lookup_or_add_for_write_span<int>("template_id", domain);
  bke::SpanAttributeWriter<bool> animated_writer =
      out_attributes.lookup_or_add_for_write_span<bool>("animated", domain);

  for (const int i : samples.index_range()) {
    const BoneSample &sample = samples[i];
    positions_writer.span[i] = sample.position;
    pose_writer.span[i] = sample.pose;
    rest_writer.span[i] = sample.rest;
    bone_index_writer.span[i] = sample.bone_index;
    bone_parent_writer.span[i] = sample.bone_parent;
    length_writer.span[i] = sample.length;
    name_writer.span[i] = sample.name;
    agent_writer.span[i] = sample.agent_id;
    template_writer.span[i] = sample.template_id;
    animated_writer.span[i] = sample.animated;
  }
  positions_writer.finish();
  pose_writer.finish();
  rest_writer.finish();
  bone_index_writer.finish();
  bone_parent_writer.finish();
  length_writer.finish();
  name_writer.finish();
  agent_writer.finish();
  template_writer.finish();
  animated_writer.finish();
  result->tag_positions_changed();

  params.set_output("Bones"_ustr, GeometrySet::from_pointcloud(result));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeCrowdBones"_ustr);
  ntype.ui_name = "Crowd Bones";
  ntype.ui_description =
      "Evaluate the per-agent animation state and emit the posed bone cloud for the crowd";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_crowd_bones_cc
