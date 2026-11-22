/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Animation Info`: describe one animation clip for `Crowd Agent States`.
 *
 * The crowd animation evaluator (C2b) reads `action_name` / frame range / speed / phase and
 * evaluates the action's F-curves per agent. Only the action channels are evaluated; bone
 * constraints and drivers are not, so non-baked rigs should be baked first.
 */

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"
#include "NOD_socket_declarations.hh"

#include <algorithm>

#include "BLI_math_base.hh"
#include "BLI_string_ref.hh"

#include "BKE_action.hh"
#include "BKE_anim_data.hh"

#include "DEG_depsgraph_query.hh"

#include "DNA_action_types.h"
#include "DNA_anim_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_crowd_animation_info_cc {

enum class RootMotion : int {
  InPlace = 0,
  FromAnimation = 1,
};

static const EnumPropertyItem root_motion_items[] = {
    {int(RootMotion::InPlace),
     "IN_PLACE",
     0,
     N_("In Place"),
     N_("The point transform moves the character; the animation's root stays at the origin")},
    {int(RootMotion::FromAnimation),
     "FROM_ANIMATION",
     0,
     N_("From Animation"),
     N_("The animation's root bone moves the character; the point is only the spawn anchor")},
    {0, nullptr, 0, nullptr, nullptr},
};

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Object>("Armature"_ustr)
      .description("Optional: read the action name / frame range from this armature");
  b.add_input<decl::Float>("Frame Start"_ustr)
      .default_value(1.0f)
      .description("First frame of the clip (overridden by the action's frame range when available)");
  b.add_input<decl::Float>("Frame End"_ustr)
      .default_value(1.0f)
      .description("Last frame of the clip (overridden by the action's frame range when available)");
  b.add_input<decl::Float>("FPS"_ustr).default_value(24.0f).min(0.0f);
  b.add_input<decl::Float>("Speed"_ustr).default_value(1.0f).min(0.0f);
  b.add_input<decl::Float>("Phase"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .max(1.0f)
      .description("Extra phase offset in 0..1; Init Crowd Data randomizes it per agent");
  b.add_input<decl::Bool>("Loop"_ustr).default_value(true);
  b.add_input<decl::Menu>("Root Motion"_ustr)
      .static_items(root_motion_items)
      .default_value(MenuValue(int(RootMotion::InPlace)));

  b.add_output<decl::Bundle>("Animation"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  Object *armature = params.extract_input<Object *>("Armature"_ustr);
  float frame_start = params.extract_input<float>("Frame Start"_ustr);
  float frame_end = params.extract_input<float>("Frame End"_ustr);
  float fps = params.extract_input<float>("FPS"_ustr);
  const float speed = std::max(params.extract_input<float>("Speed"_ustr), 0.0f);
  const float phase = math::clamp(params.extract_input<float>("Phase"_ustr), 0.0f, 1.0f);
  const bool loop = params.extract_input<bool>("Loop"_ustr);
  const int root_motion = params.extract_input<MenuValue>("Root Motion"_ustr).value;

  std::string action_name;
  if (armature != nullptr && armature->adt != nullptr && armature->adt->action != nullptr) {
    const bAction *action = armature->adt->action;
    action_name = std::string(action->id.name + 2);
    if (action->frame_start < action->frame_end) {
      frame_start = action->frame_start;
      frame_end = action->frame_end;
    }
  }
  if (const Scene *scene = DEG_get_input_scene(params.depsgraph())) {
    if (fps <= 0.0f) {
      fps = float(scene->r.frs_sec) / std::max(1.0f, float(scene->r.frs_sec_base));
    }
  }
  frame_end = std::max(frame_end, frame_start);

  BundlePtr animation_ptr = Bundle::create();
  Bundle &animation = animation_ptr.ensure_mutable_inplace();
  animation.add(*BundleKey::from_str("Type"), std::string(crowd::ANIMATION_TYPE));
  animation.add(*BundleKey::from_str("frame_start"), frame_start);
  animation.add(*BundleKey::from_str("frame_end"), frame_end);
  animation.add(*BundleKey::from_str("fps"), fps);
  animation.add(*BundleKey::from_str("speed"), speed);
  animation.add(*BundleKey::from_str("phase"), phase);
  animation.add(*BundleKey::from_str("loop"), loop);
  animation.add(*BundleKey::from_str("root_motion"), root_motion);
  animation.add(*BundleKey::from_str("action_name"), action_name);
  params.set_output("Animation"_ustr, std::move(animation_ptr));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeCrowdAnimationInfo"_ustr);
  ntype.ui_name = "Animation Info";
  ntype.ui_description =
      "Describe an animation clip (frame range, speed, phase, loop, root motion) for a crowd state";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_220;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_crowd_animation_info_cc
