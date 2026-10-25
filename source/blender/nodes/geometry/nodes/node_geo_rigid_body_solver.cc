/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Rigid body solver node, based on Jolt Physics.
 *
 * The node follows the declarative system design used by the XPBD solver: the whole
 * simulation state is carried by a single `World` bundle. Geometry sets with their tags,
 * constraint definitions and effector bundles (e.g. `Blender.Force` produced by the official
 * *Custom Force* asset) all live in that bundle, so effectors can be shared between the
 * cloth/hair and the rigid body systems.
 *
 * Two kinds of geometry sets are understood:
 *
 * - **Bodies from instances**: every instance is one rigid body. Its transform provides the
 *   body position and rotation, and the instanced geometry is used as convex hull collision
 *   shape. This is the form the *Voronoi Fracture* node produces.
 * - **Bodies from points**: every point of a point cloud is a sphere body, with the attributes
 *   `position`, `rotation`, `velocity`, `angular_velocity`, `mass` and `radius`.
 *
 * Bodies are identified by their `piece_id` attribute (falling back to the element index), which
 * is what constraints reference:
 *
 * - **Constraints from points**: a point cloud with the integer attributes `body_a` and `body_b`
 *   (piece ids) and an optional `enabled` boolean. Every point is one fixed (weld) constraint.
 */

#include <algorithm>
#include <optional>

#include "BKE_attribute.hh"
#include "BKE_instances.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion_types.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_set.hh"
#include "BLI_string_utf8.h"

#include "GEO_rigid_body_solver.hh"

#include "NOD_bundle_type.hh"
#include "NOD_geo_tag_filter.hh"
#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_list.hh"
#include "NOD_geometry_nodes_physics_bundles.hh"

#include "node_geometry_util.hh"

/* Temporary tracing for crash hunting; enabled with BLENDER_RBD_DEBUG=1. */
#include <cstdio>
#include <cstdlib>
static bool rbd_trace_enabled()
{
  static const bool enabled = std::getenv("BLENDER_RBD_DEBUG") != nullptr;
  return enabled;
}
#define RBD_TRACE(...) \
  do { \
    if (rbd_trace_enabled()) { \
      printf(__VA_ARGS__); \
      fflush(stdout); \
    } \
  } while (0)

namespace blender::nodes::node_geo_rigid_body_solver_cc {

using namespace blender::nodes::physics_bundles;
using namespace blender::geometry::rigid_body;

/**
 * Data reported back by the solver, stored in the world bundle at the "Solver Path".
 */
class RigidBodySolverDataBundle {
 public:
  static constexpr StringRefNull name = "Blender.RigidBodySolverData";
  static const FlatBundleTypePtr &get_bundle_type();
};

const FlatBundleTypePtr &RigidBodySolverDataBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(RigidBodySolverDataBundle::name);
    b.add<decl::Float>("residual_error"_ustr)
        .min(0.0f)
        .description("Average remaining relative error of the last solver step");
    b.add<decl::Int>("active_bodies"_ustr)
        .min(0)
        .description("Number of rigid bodies that were not sleeping in the last step");
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

/**
 * The bundle type carried by the `World` socket. It collects everything the solver can
 * consume: the solver output data plus the bundles that are shared with the official
 * physics system (colliders, damping and force effectors).
 */
static NestedBundleTypePtr make_world_type()
{
  Vector<std::shared_ptr<const FlatBundleType>> types;
  types.append(RigidBodySolverDataBundle::get_bundle_type());
  types.append(MeshColliderBundle::get_bundle_type());
  types.append(BoxColliderBundle::get_bundle_type());
  types.append(DampingBundle::get_bundle_type());
  types.append(ForceBundle::get_bundle_type());

  /* Registered here so that assets can create them in the same node tree. */
  CustomGeometryEffector::get_bundle_type();
  CustomWorldEffector::get_bundle_type();

  NestedBundleTypePtr world_type = std::make_shared<const NestedBundleType>(
      "Blender.RigidBodySolverWorld", std::move(types));
  BundleTypeRegistry::register_type(world_type);
  return world_type;
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();

  static NestedBundleTypePtr world_type = make_world_type();
  b.add_input<decl::Geometry>("Upstream"_ustr)
      .description(
          "Optional: the pieces *before* the simulation zone. The bodies that are marked as "
          "animated in Init Rigid Body take their pose from here, and the `active` flag is read from "
          "here as well: the zone state only reads its inputs on the first frame, so changes to "
          "those flags would otherwise never reach the solver. This input has to come from inside "
          "the simulation zone");
  b.add_input<decl::Bundle>("World"_ustr)
      .bundle_type(world_type)
      .evaluated_geometry_field()
      .structure_type(StructureType::Single)
      .description("Rigid body world state that is updated by the solver");
  b.add_output<decl::Bundle>("World"_ustr).pass_through_input_index(0).align_with_previous();
  b.add_input<decl::Float>("Delta Time"_ustr)
      .min(0)
      .default_value(1 / 60.0f)
      .subtype(PROP_TIME_ABSOLUTE);

  b.add_input<decl::String>("Filter"_ustr)
      .optional_label()
      .description("Filters the geometry sets to process based on their tags");

  b.add_input<decl::Matrix>("Simulation to World"_ustr);

  {
    auto &solver_panel = b.add_panel("Solver"_ustr).default_closed(true);
    solver_panel.add_input<decl::Int>("Substeps"_ustr).default_value(1).min(1);
    solver_panel.add_input<decl::Int>("Constraint Iterations"_ustr).default_value(10).min(1);

    solver_panel.add_input<decl::Bool>("Speed Limit"_ustr)
        .default_value(false)
        .description(
            "Clamp how fast a body may move. Off: no limit, so a fast projectile keeps its speed "
            "(Jolt's internal 500 m/s safety ceiling stays). On: bodies are clamped to Max Speed, "
            "which keeps the penetration recovery of a dense pile from launching small fragments");
    solver_panel.add_input<decl::Float>("Max Speed"_ustr)
        .default_value(50.0f)
        .min(0.0f)
        .subtype(PROP_VELOCITY)
        .description("Speed ceiling in m/s, used when Speed Limit is on (the angular speed is "
                     "limited to 0.6 times this)");
    solver_panel.add_input<decl::Int>("Threads"_ustr)
        .default_value(1)
        .description(
            "Threads the solver may use (0 = all available cores). The contact graph is solved in "
            "parallel, which is the largest speedup for big piles (about 2x on 8 cores), but the "
            "result then depends on the thread scheduling and a marginal pile can behave "
            "differently, so 1 thread (fully reproducible) is the default");
    solver_panel.add_input<decl::Bool>("Allow Sleeping"_ustr)
        .default_value(true)
        .description(
            "Let resting bodies go to sleep. A sleeping pile costs almost nothing, which matters a "
            "lot for thousands of fragments; the bodies wake up when something touches them");
    solver_panel.add_input<decl::Bool>("Wake All"_ustr)
        .default_value(false)
        .description("Keep every body awake, e.g. while placing the scene");
    solver_panel.add_input<decl::Float>("Sleep Threshold"_ustr)
        .default_value(0.03f)
        .min(0.0f)
        .subtype(PROP_VELOCITY)
        .description(
            "Point velocity below which a body may fall asleep, in m/s. Raise it to let a slowly "
            "creeping pile settle (and cost nothing), lower it when a nearly stopped pile still has "
            "to keep sliding");
    solver_panel.add_input<decl::Float>("Time Before Sleep"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .subtype(PROP_TIME)
        .description(
            "How long a body has to stay below the sleep threshold before it sleeps, in seconds");
    solver_panel.add_input<decl::Bool>("Ignore Initial Overlap"_ustr)
        .default_value(true)
        .description(
            "Take body pairs that already interpenetrate when the world is built out of collision. "
            "Without this the contact solver pushes them apart with a huge impulse on the first "
            "frame, which looks like the pile exploding on its own");
    solver_panel.add_input<decl::Bool>("Sweep Colliders"_ustr)
        .default_value(true)
        .description(
            "Sweep a collider that moved between frames through the substeps. The collider geometry "
            "is sampled once per frame, so without this a fast moving collider jumps past the "
            "bodies and never collides with them");
    solver_panel.add_input<decl::Bool>("Wake On Break"_ustr)
        .default_value(true)
        .description(
            "Wake both bodies when a constraint breaks. Off means a sleeping body stays asleep even "
            "when its glue lets go, which is only useful for debugging");
    solver_panel.add_input<decl::Bool>("Warm Start"_ustr)
        .default_value(true)
        .description(
            "Keep the physics world alive between frames so Jolt can reuse contacts and constraint "
            "impulses. The world is rebuilt whenever the structure or the incoming state changes");
    solver_panel.add_input<decl::Bool>("Glued Pieces Collide"_ustr)
        .default_value(false)
        .description(
            "Let glued pieces collide with each other. Off by default: the cut faces of two glued "
            "fragments touch, so the contact solver would push them apart while the glue pulls "
            "them back");
    solver_panel.add_input<decl::Vector>("Gravity"_ustr)
        .default_value(float3(0.0f, 0.0f, -9.81f))
        .description("Gravity used for this solver step");
    solver_panel.add_input<decl::String>("Solver Path"_ustr)
        .default_value("")
        .description("Optional output path in the world bundle for solver data");
  }
}

/** Collect the tags attached to a geometry set through its geometry bundle. */
static Set<std::string> get_geometry_tags(const GeometrySet &geometry)
{
  Set<std::string> tags;
  if (!geometry.has_bundle()) {
    return tags;
  }
  const Bundle &bundle = *geometry.bundle();
  const std::optional<GListPtr> tags_list_ptr = bundle.lookup_path<GListPtr>("tags");
  if (!tags_list_ptr || !*tags_list_ptr) {
    return tags;
  }
  const GList &tags_list = **tags_list_ptr;
  if (!tags_list.cpp_type().is<std::string>()) {
    return tags;
  }
  tags_list.typed<std::string>().foreach([&](const std::string &tag) { tags.add(tag); });
  return tags;
}

/* -------------------------------------------------------------------- */
/** \name Bodies from point clouds
 * \{ */

static Array<BodyState> read_point_body_states(const PointCloud &pointcloud)
{
  const bke::AttributeAccessor attributes = pointcloud.attributes();
  const VArray<float3> positions =
      attributes.lookup_or_default<float3>("position", AttrDomain::Point, float3(0.0f)).varray;
  const VArray<math::Quaternion> rotations =
      attributes
          .lookup_or_default<math::Quaternion>(
              "rotation", AttrDomain::Point, math::Quaternion::identity())
          .varray;
  const VArray<float3> linear_velocities =
      attributes.lookup_or_default<float3>("velocity", AttrDomain::Point, float3(0.0f)).varray;
  const VArray<float3> angular_velocities = attributes
                                                .lookup_or_default<float3>(
                                                    "angular_velocity",
                                                    AttrDomain::Point,
                                                    float3(0.0f))
                                                .varray;
  const VArray<float3> external_forces = attributes
                                             .lookup_or_default<float3>(
                                                 "external_force", AttrDomain::Point, float3(0.0f))
                                             .varray;
  const VArray<float3> external_torques = attributes
                                              .lookup_or_default<float3>(
                                                  "external_torque",
                                                  AttrDomain::Point,
                                                  float3(0.0f))
                                              .varray;
  const VArray<float> radii =
      attributes.lookup_or_default<float>("radius", AttrDomain::Point, 0.1f).varray;
  const VArray<float> masses =
      attributes.lookup_or_default<float>("mass", AttrDomain::Point, 1.0f).varray;
  const VArray<float> linear_dampings =
      attributes.lookup_or_default<float>("linear_damping", AttrDomain::Point, 0.05f).varray;
  const VArray<float> angular_dampings =
      attributes.lookup_or_default<float>("angular_damping", AttrDomain::Point, 0.05f).varray;

  Array<BodyState> states(pointcloud.totpoint);
  for (const int i : states.index_range()) {
    BodyState &state = states[i];
    state.position = positions[i];
    state.rotation = rotations[i];
    state.linear_velocity = linear_velocities[i];
    state.angular_velocity = angular_velocities[i];
    state.external_force = external_forces[i];
    state.external_torque = external_torques[i];
    state.radius = radii[i];
    state.mass = masses[i];
    state.linear_damping = linear_dampings[i];
    state.angular_damping = angular_dampings[i];
  }
  return states;
}

static void write_point_body_states(PointCloud &pointcloud, Span<BodyState> states)
{
  bke::MutableAttributeAccessor attributes = pointcloud.attributes_for_write();
  bke::SpanAttributeWriter<float3> positions = attributes.lookup_or_add_for_write_span<float3>(
      "position", AttrDomain::Point);
  bke::SpanAttributeWriter<math::Quaternion> rotations =
      attributes.lookup_or_add_for_write_span<math::Quaternion>("rotation", AttrDomain::Point);
  bke::SpanAttributeWriter<float3> linear_velocities =
      attributes.lookup_or_add_for_write_span<float3>("velocity", AttrDomain::Point);
  bke::SpanAttributeWriter<float3> angular_velocities =
      attributes.lookup_or_add_for_write_span<float3>("angular_velocity", AttrDomain::Point);

  for (const int i : states.index_range()) {
    const BodyState &state = states[i];
    positions.span[i] = state.position;
    rotations.span[i] = state.rotation;
    linear_velocities.span[i] = state.linear_velocity;
    angular_velocities.span[i] = state.angular_velocity;
  }

  positions.finish();
  rotations.finish();
  linear_velocities.finish();
  angular_velocities.finish();
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Bodies from instances
 * \{ */

static Array<BodyState> read_instance_body_states(const bke::Instances &instances)
{
  const int count = instances.instances_num();
  const bke::AttributeAccessor attributes = instances.attributes();
  const VArray<float3> linear_velocities =
      attributes.lookup_or_default<float3>("velocity", AttrDomain::Instance, float3(0.0f)).varray;
  const VArray<float3> angular_velocities = attributes
                                                .lookup_or_default<float3>(
                                                    "angular_velocity",
                                                    AttrDomain::Instance,
                                                    float3(0.0f))
                                                .varray;
  const VArray<float3> external_forces = attributes
                                             .lookup_or_default<float3>(
                                                 "external_force",
                                                 AttrDomain::Instance,
                                                 float3(0.0f))
                                             .varray;
  const VArray<float3> external_torques = attributes
                                              .lookup_or_default<float3>(
                                                  "external_torque",
                                                  AttrDomain::Instance,
                                                  float3(0.0f))
                                              .varray;
  const VArray<float> radii =
      attributes.lookup_or_default<float>("radius", AttrDomain::Instance, 0.1f).varray;
  const VArray<float> masses =
      attributes.lookup_or_default<float>("mass", AttrDomain::Instance, 1.0f).varray;
  const VArray<float> linear_dampings =
      attributes.lookup_or_default<float>("linear_damping", AttrDomain::Instance, 0.05f).varray;
  const VArray<float> angular_dampings =
      attributes.lookup_or_default<float>("angular_damping", AttrDomain::Instance, 0.05f).varray;
  const VArray<bool> animateds =
      attributes.lookup_or_default<bool>("animated", AttrDomain::Instance, false).varray;
  const VArray<bool> actives =
      attributes.lookup_or_default<bool>("active", AttrDomain::Instance, true).varray;

  Array<BodyState> states(count);
  const Span<float4x4> transforms = instances.transforms();
  for (const int i : states.index_range()) {
    BodyState &state = states[i];
    state.position = transforms[i].location();
    state.animated = animateds[i];
    state.rotation = math::to_quaternion(transforms[i]);
    state.linear_velocity = linear_velocities[i];
    state.angular_velocity = angular_velocities[i];
    state.external_force = external_forces[i];
    state.external_torque = external_torques[i];
    state.radius = radii[i];
    state.mass = masses[i];
    state.linear_damping = linear_dampings[i];
    state.angular_damping = angular_dampings[i];
    state.animated = animateds[i];
    state.active = actives[i];
  }
  return states;
}

static void write_instance_body_states(bke::Instances &instances, Span<BodyState> states)
{
  MutableSpan<float4x4> transforms = instances.transforms_for_write();
  for (const int i : states.index_range()) {
    transforms[i] = math::from_loc_rot_scale<float4x4>(
        states[i].position, states[i].rotation, float3(1.0f));
  }

  bke::MutableAttributeAccessor attributes = instances.attributes_for_write();
  bke::SpanAttributeWriter<float3> linear_velocities =
      attributes.lookup_or_add_for_write_span<float3>("velocity", AttrDomain::Instance);
  bke::SpanAttributeWriter<float3> angular_velocities =
      attributes.lookup_or_add_for_write_span<float3>("angular_velocity", AttrDomain::Instance);
  bke::SpanAttributeWriter<bool> sleeping =
      attributes.lookup_or_add_for_write_span<bool>("sleeping", AttrDomain::Instance);
  for (const int i : states.index_range()) {
    linear_velocities.span[i] = states[i].linear_velocity;
    angular_velocities.span[i] = states[i].angular_velocity;
    sleeping.span[i] = states[i].sleeping;
  }
  linear_velocities.finish();
  angular_velocities.finish();
  sleeping.finish();
}

/**
 * Derive collision shapes from instances: one instance per body, the instance geometry is used
 * as convex hull (in instance local space, which matches the body transform space).
 */
static void collect_body_shapes(const GeometrySet &geometry,
                                const int body_num,
                                Vector<BodyShape> &r_shapes,
                                Vector<float3> &r_points)
{
  r_shapes.resize(body_num);
  if (!geometry.has_instances()) {
    return;
  }
  const bke::Instances *instances = geometry.get_instances();
  if (!instances || instances->instances_num() != body_num) {
    return;
  }
  const Span<int> reference_handles = instances->reference_handles();
  if (reference_handles.size() != body_num) {
    return;
  }
  for (const int i : IndexRange(body_num)) {
    const int reference_i = reference_handles[i];
    const Span<bke::InstanceReference> references = instances->references();
    if (reference_i < 0 || reference_i >= references.size()) {
      continue;
    }
    const bke::InstanceReference &reference = references[reference_i];
    if (reference.type() != bke::InstanceReference::Type::GeometrySet) {
      continue;
    }
    const Mesh *mesh = reference.geometry_set().get_mesh();
    if (!mesh) {
      continue;
    }
    const Span<float3> verts = mesh->vert_positions();
    if (verts.size() < 4) {
      continue;
    }
    BodyShape &shape = r_shapes[i];
    shape.type = ShapeType::ConvexHull;
    shape.points_start = int(r_points.size());
    for (const float3 &p : verts) {
      r_points.append(p);
    }
    shape.points_count = int(r_points.size()) - shape.points_start;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Constraints
 * \{ */

/** A constraint as stored in the attributes, referencing bodies by `piece_id`. */
struct RawConstraint {
  int body_a = -1;
  int body_b = -1;
  bool enabled = true;
  bool has_anchors = false;
  float3 anchor_a = float3(0.0f);
  float3 anchor_b = float3(0.0f);
  float break_impulse = 0.0f;
  float break_distance = 0.0f;
  float break_angle = 0.0f;
  /** 0 = fixed (weld), 1 = soft. */
  int type = 0;
  float frequency = 5.0f;
  float damping = 0.5f;
  /** 0 = disable on break, 1 = replace with a soft constraint. */
  int fallback = 0;
  /** Seconds of stable deformation before the rest pose is re-baselined (0 = off). */
  float harden_time = 0.0f;
  /** Time the constraint has already been deformed, persisted across world rebuilds. */
  float harden_accum = 0.0f;
  /** Where the constraint comes from, so a broken state can be written back to it. */
  std::string source_path;
  int source_point = -1;
};

/** Returns true if the point cloud describes constraints rather than bodies. */
static bool is_constraint_cloud(const PointCloud &pointcloud)
{
  const bke::AttributeAccessor attributes = pointcloud.attributes();
  return attributes.contains("body_a") && attributes.contains("body_b");
}

/**
 * A force on one body, read from a point cloud that has one point per body and references the body
 * with `piece_id`. This is what the official force assets (`Blender.Force` bundles evaluated by
 * `Evaluate Forces`) fill in: they store into the point domain, so the rigid bodies get their
 * forces through a point cloud instead of through the instance domain.
 */
struct RawBodyForce {
  int piece_id = -1;
  float3 force = float3(0.0f);
  float3 torque = float3(0.0f);
  std::string source_path;
  int source_point = -1;
};

/** Returns true if the point cloud carries per-body forces (as opposed to being a body set). */
static bool is_body_force_cloud(const PointCloud &pointcloud)
{
  const bke::AttributeAccessor attributes = pointcloud.attributes();
  if (!attributes.contains("piece_id")) {
    return false;
  }
  return attributes.contains("external_force") || attributes.contains("external_torque");
}

static void read_body_forces(const PointCloud &pointcloud,
                             const StringRef path,
                             Vector<RawBodyForce> &r_forces)
{
  const bke::AttributeAccessor attributes = pointcloud.attributes();
  const bool has_force = attributes.contains("external_force");
  const bool has_torque = attributes.contains("external_torque");
  if (!has_force && !has_torque) {
    return;
  }
  const VArray<int> piece_ids =
      attributes.lookup_or_default<int>("piece_id", AttrDomain::Point, -1).varray;
  const VArray<float3> forces =
      attributes.lookup_or_default<float3>("external_force", AttrDomain::Point, float3(0.0f))
          .varray;
  const VArray<float3> torques =
      attributes.lookup_or_default<float3>("external_torque", AttrDomain::Point, float3(0.0f))
          .varray;

  for (const int i : IndexRange(pointcloud.totpoint)) {
    if (piece_ids[i] < 0) {
      continue;
    }
    RawBodyForce body_force;
    body_force.piece_id = piece_ids[i];
    body_force.force = has_force ? forces[i] : float3(0.0f);
    body_force.torque = has_torque ? torques[i] : float3(0.0f);
    body_force.source_path = path;
    body_force.source_point = i;
    r_forces.append(body_force);
  }
}

static Vector<RawConstraint> read_constraints(const PointCloud &pointcloud, const StringRef path)
{
  Vector<RawConstraint> constraints;
  const bke::AttributeAccessor attributes = pointcloud.attributes();
  const VArray<int> body_a = attributes.lookup_or_default<int>("body_a", AttrDomain::Point, -1).varray;
  const VArray<int> body_b = attributes.lookup_or_default<int>("body_b", AttrDomain::Point, -1).varray;
  const VArray<bool> enabled =
      attributes.lookup_or_default<bool>("enabled", AttrDomain::Point, true).varray;
  const bool has_anchors = attributes.contains("anchor_a") && attributes.contains("anchor_b");
  const VArray<float3> anchor_a =
      attributes.lookup_or_default<float3>("anchor_a", AttrDomain::Point, float3(0.0f)).varray;
  const VArray<float3> anchor_b =
      attributes.lookup_or_default<float3>("anchor_b", AttrDomain::Point, float3(0.0f)).varray;
  const VArray<float> break_impulse =
      attributes.lookup_or_default<float>("break_impulse", AttrDomain::Point, 0.0f).varray;
  const VArray<float> break_distance =
      attributes.lookup_or_default<float>("break_distance", AttrDomain::Point, 0.0f).varray;
  const VArray<float> break_angle =
      attributes.lookup_or_default<float>("break_angle", AttrDomain::Point, 0.0f).varray;
  const VArray<int> types = attributes.lookup_or_default<int>("type", AttrDomain::Point, 0).varray;
  const VArray<float> frequencies =
      attributes.lookup_or_default<float>("frequency", AttrDomain::Point, 5.0f).varray;
  const VArray<float> dampings =
      attributes.lookup_or_default<float>("damping", AttrDomain::Point, 0.5f).varray;
  const VArray<int> fallbacks =
      attributes.lookup_or_default<int>("fallback", AttrDomain::Point, 0).varray;
  const VArray<float> harden_times =
      attributes.lookup_or_default<float>("harden_time", AttrDomain::Point, 0.0f).varray;
  const VArray<float> harden_accums =
      attributes.lookup_or_default<float>("harden_accum", AttrDomain::Point, 0.0f).varray;

  for (const int i : IndexRange(pointcloud.totpoint)) {
    if (body_a[i] < 0 || body_b[i] < 0) {
      continue;
    }
    RawConstraint constraint;
    constraint.body_a = body_a[i];
    constraint.body_b = body_b[i];
    constraint.enabled = enabled[i];
    constraint.has_anchors = has_anchors;
    constraint.anchor_a = anchor_a[i];
    constraint.anchor_b = anchor_b[i];
    constraint.break_impulse = break_impulse[i];
    constraint.break_distance = break_distance[i];
    constraint.break_angle = break_angle[i];
    constraint.type = types[i];
    constraint.frequency = frequencies[i];
    constraint.damping = dampings[i];
    constraint.fallback = fallbacks[i];
    constraint.harden_time = harden_times[i];
    constraint.harden_accum = harden_accums[i];
    constraint.source_path = path;
    constraint.source_point = i;
    constraints.append(constraint);
  }
  return constraints;
}

/**
 * Read the `piece_id` of every body of a geometry set, so constraints can be mapped from ids
 * to the local body indices.
 */
static Array<int> read_body_ids(const GeometrySet &geometry, const int body_num)
{
  Array<int> ids(body_num);
  if (geometry.has_instances()) {
    const bke::Instances *instances = geometry.get_instances();
    const VArray<int> piece_ids = instances->attributes()
                                      .lookup_or_default<int>(
                                          "piece_id", AttrDomain::Instance, -1)
                                      .varray;
    for (const int i : ids.index_range()) {
      ids[i] = piece_ids[i] < 0 ? i : piece_ids[i];
    }
  }
  else if (geometry.has_pointcloud()) {
    const PointCloud *pointcloud = geometry.get_pointcloud();
    const VArray<int> piece_ids =
        pointcloud->attributes().lookup_or_default<int>("piece_id", AttrDomain::Point, -1).varray;
    for (const int i : ids.index_range()) {
      ids[i] = piece_ids[i] < 0 ? i : piece_ids[i];
    }
  }
  else {
    for (const int i : ids.index_range()) {
      ids[i] = i;
    }
  }
  return ids;
}

/** \} */

/**
 * Collect static colliders from `Blender.Collider.Mesh` and `Blender.Collider.Box` bundles in the
 * world. The mesh ones become triangle meshes, the box ones a solid box around their bounds.
 */
static void collect_colliders(const Bundle &world,
                              Vector<StaticCollider> &r_colliders,
                              Vector<float3> &r_vertices,
                              Vector<int> &r_indices)
{
  struct ColliderPaths {
    Vector<std::string> paths;
    bool use_box;
  };
  ColliderPaths groups[2];
  groups[0].paths = gather_bundle_paths_by_bundle_type(
      world, [](StringRef type) { return type == MeshColliderBundle::name; });
  groups[0].use_box = false;
  groups[1].paths = gather_bundle_paths_by_bundle_type(
      world, [](StringRef type) { return type == BoxColliderBundle::name; });
  groups[1].use_box = true;
  for (ColliderPaths &group : groups) {
    std::ranges::sort(group.paths);
  }

  for (const ColliderPaths &group : groups) {
  const bool use_box = group.use_box;
  for (const StringRef path : group.paths) {
    const BundlePtr *bundle_ptr = world.lookup_path_ptr<BundlePtr>(path);
    if (!bundle_ptr || !*bundle_ptr) {
      continue;
    }
    const Bundle &bundle = **bundle_ptr;
    const std::optional<GeometrySet> geometry = bundle.lookup<GeometrySet>(
        *BundleKey::from_str("geometry"));
    if (!geometry) {
      continue;
    }
    const Mesh *mesh = geometry->get_mesh();
    if (!mesh) {
      continue;
    }

    StaticCollider collider;
    collider.use_box = use_box;
    collider.vertices_start = int(r_vertices.size());
    for (const float3 &p : mesh->vert_positions()) {
      r_vertices.append(p);
    }
    collider.vertices_count = int(r_vertices.size()) - collider.vertices_start;

    collider.indices_start = int(r_indices.size());
    /* `corner_tris()` contains corner indices; map them to vertex indices first. */
    Array<int3> vert_tris(mesh->corner_tris().size());
    bke::mesh::vert_tris_from_corner_tris(
        mesh->corner_verts(), mesh->corner_tris(), vert_tris);
    for (const int3 &tri : vert_tris) {
      r_indices.append(tri.x);
      r_indices.append(tri.y);
      r_indices.append(tri.z);
    }
    collider.indices_count = int(r_indices.size()) - collider.indices_start;

    collider.friction = bundle.lookup<float>(*BundleKey::from_str("friction")).value_or(0.5f);
    /* Convex hull colliders use the mesh bundle type with this flag set. */
    collider.convex = bundle.lookup<bool>(*BundleKey::from_str("convex")).value_or(false);
    const bool enough_geometry = use_box ? collider.vertices_count >= 3
                                         : (collider.vertices_count >= 3 &&
                                            collider.indices_count >= 3);
    if (enough_geometry) {
      r_colliders.append(collider);
    }
  }
  }
}

/**
 * Write the current body positions back into the per-body force clouds. Position dependent force
 * fields are evaluated on those clouds, so without this they would keep sampling the positions the
 * bodies had when the simulation started.
 */
static void write_body_force_positions(Bundle &world,
                                       const Vector<std::string> &force_paths,
                                       const Vector<RawBodyForce> &forces,
                                       const Map<int, int> &id_to_index,
                                       const Span<BodyState> states)
{
  for (const std::string &path : force_paths) {
    GeometrySet *geometry = world.lookup_path_for_write_ptr<GeometrySet>(path);
    if (!geometry || !geometry->has_pointcloud()) {
      continue;
    }
    PointCloud *pointcloud = geometry->get_pointcloud_for_write();
    if (!pointcloud) {
      continue;
    }
    MutableSpan<float3> positions = pointcloud->positions_for_write();
    for (const RawBodyForce &body_force : forces) {
      if (body_force.source_path != path) {
        continue;
      }
      const int *index = id_to_index.lookup_ptr(body_force.piece_id);
      if (!index) {
        continue;
      }
      if (body_force.source_point >= 0 && body_force.source_point < positions.size()) {
        positions[body_force.source_point] = states[*index].position;
      }
    }
  }
}

static void node_geo_exec(GeoNodeExecParams params){
  BundlePtr world_ptr = params.get_input<BundlePtr>("World"_ustr);
  if (!world_ptr) {
    params.set_default_remaining_outputs();
    return;
  }

  RBD_TRACE("rbd trace: exec entry\n");
  const int substeps = params.get_input<int>("Substeps"_ustr);
  RBD_TRACE("rbd trace: substeps=%d\n", substeps);
  if (substeps <= 0) {
    params.set_output("World"_ustr, std::move(world_ptr));
    return;
  }

  const float delta_time = std::max(0.0f, params.get_input<float>("Delta Time"_ustr));
  const GeometrySet upstream_geometry = params.extract_input<GeometrySet>("Upstream"_ustr);
  const float3 gravity = params.get_input<float3>("Gravity"_ustr);
  const std::string geometry_tag_filter = params.extract_input<std::string>("Filter"_ustr);

  Bundle &world = world_ptr.ensure_mutable_inplace();

  Vector<StaticCollider> colliders;
  Vector<float3> collider_vertices;
  Vector<int> collider_indices;
  collect_colliders(world, colliders, collider_vertices, collider_indices);
  if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
    printf("RBD_DEBUG colliders=%d z0=%.3f\n",
           int(colliders.size()),
           collider_vertices.is_empty() ? 0.0 : double(collider_vertices[0].z));
    fflush(stdout);
  }

  Vector<std::string> paths = gather_bundle_paths_by_data_type(world, SOCK_GEOMETRY);
  /* Ensure the order is deterministic. */
  std::ranges::sort(paths);

  /* A collider that already contains rigid bodies is almost always a collider built from an object
   * in *local* space (for example `Object Info > Geometry` without `Object Info > Transform`:
   * the geometry then sits at the object's original position instead of following it). The bodies
   * inside get pushed out every frame, which looks like a pile that twitches and whose glued pieces
   * break without anything touching them, so warn about it instead of leaving it silent. */
  if (!colliders.is_empty() && !paths.is_empty()) {
    int overlapping = 0;
    for (const int i : colliders.index_range()) {
      const StaticCollider &collider = colliders[i];
      if (collider.vertices_count <= 0) {
        continue;
      }
      float3 min_point(FLT_MAX, FLT_MAX, FLT_MAX);
      float3 max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
      for (const int j : IndexRange(collider.vertices_count)) {
        const float3 &vertex = collider_vertices[collider.vertices_start + j];
        min_point = math::min(min_point, vertex);
        max_point = math::max(max_point, vertex);
      }
      bool hit = false;
      for (const StringRef path : paths) {
        const GeometrySet *geometry = world.lookup_path_ptr<GeometrySet>(path);
        if (!geometry || !geometry->has_instances()) {
          continue;
        }
        for (const float4x4 &transform : geometry->get_instances()->transforms()) {
          const float3 position = transform.location();
          if (position.x < min_point.x || position.y < min_point.y || position.z < min_point.z ||
              position.x > max_point.x || position.y > max_point.y || position.z > max_point.z)
          {
            continue;
          }
          hit = true;
          break;
        }
        if (hit) {
          break;
        }
      }
      overlapping += hit ? 1 : 0;
    }
    if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
      printf("RBD_DEBUG collider_overlap check colliders=%d overlapping=%d bodies=%d\n",
             int(colliders.size()),
             overlapping,
             int(paths.size()));
      fflush(stdout);
    }
    if (overlapping > 0) {
      std::string message = std::to_string(overlapping) +
                            " collider(s) already contain rigid bodies at their current position. "
                            "If the collider is built from an object, connect Object Info > Transform "
                            "to the shape's Transform input: without it the collider stays in the "
                            "object's local space, sits in the wrong place and keeps pushing the "
                            "bodies (this shows up as a twitching pile that breaks on its own).";
      params.error_message_add(NodeWarningType::Warning, message);
    }
  }

  /* Constraints and per-body forces may live in different geometry sets than the bodies, so gather
   * them first. */
  Vector<RawConstraint> raw_constraints;
  Vector<std::string> constraint_paths;
  Vector<RawBodyForce> raw_body_forces;
  Vector<std::string> body_force_paths;
  for (const StringRef path : paths) {
    const GeometrySet *geometry = world.lookup_path_ptr<GeometrySet>(path);
    if (!geometry || !geometry->has_pointcloud()) {
      continue;
    }
    if (!tag_filter_matches(geometry_tag_filter, get_geometry_tags(*geometry))) {
      continue;
    }
    const PointCloud *pointcloud = geometry->get_pointcloud();
    if (!pointcloud) {
      continue;
    }
    if (is_constraint_cloud(*pointcloud)) {
      raw_constraints.extend(read_constraints(*pointcloud, path));
      constraint_paths.append(std::string(path));
    }
    else if (is_body_force_cloud(*pointcloud)) {
      read_body_forces(*pointcloud, path, raw_body_forces);
      body_force_paths.append(std::string(path));
    }
  }
  Array<ConstraintFeedback> constraint_feedback(raw_constraints.size());

  SimulateSettings settings;
  settings.gravity = gravity;
  settings.delta_time = delta_time;
  settings.substeps = substeps;
  settings.constraint_iterations = std::max(1, params.get_input<int>("Constraint Iterations"_ustr));
  settings.disable_glued_collisions = !params.get_input<bool>("Glued Pieces Collide"_ustr);
  settings.warm_start = params.get_input<bool>("Warm Start"_ustr);
  settings.threads = params.get_input<int>("Threads"_ustr);
  settings.allow_sleeping = params.get_input<bool>("Allow Sleeping"_ustr);
  settings.wake_all = params.get_input<bool>("Wake All"_ustr);
  settings.sleep_threshold = std::max(params.get_input<float>("Sleep Threshold"_ustr), 0.0f);
  settings.time_before_sleep = std::max(
      params.get_input<float>("Time Before Sleep"_ustr), 0.0f);
  settings.wake_on_break = params.get_input<bool>("Wake On Break"_ustr);
  settings.sweep_colliders = params.get_input<bool>("Sweep Colliders"_ustr);
  settings.ignore_initial_overlap = params.get_input<bool>("Ignore Initial Overlap"_ustr);
  settings.speed_limit = params.get_input<bool>("Speed Limit"_ustr);
  settings.max_speed = std::max(params.get_input<float>("Max Speed"_ustr), 0.0f);
  if (settings.speed_limit && settings.max_speed < 1.0e-3f) {
    /* A ceiling of 0 freezes every body (it can only be moved by penetration recovery), which is
     * never what the user wants, so the limit is ignored and the node says why. */
    params.error_message_add(
        NodeWarningType::Warning,
        "Max Speed is 0, ignoring the speed limit (nothing would move). Set a positive speed or "
        "uncheck Speed Limit");
    settings.speed_limit = false;
  }

  /* One persistent physics world per solver node. The key must not be the compute context: that
   * one changes while descending into nested node groups and between the evaluations of a frame,
   * which gave the node several worlds that all wrote the same state. Each of them then rebuilt the
   * world (losing warm starting and re-welding the glue to its original rest pose) every frame,
   * which showed up as a pile that twitched periodically. */
  SolverWorldKey solver_key;
  if (const GeoNodesUserData *user_data = params.user_data()) {
    if (user_data->call_data != nullptr) {
      solver_key.object = user_data->call_data->self_object();
      solver_key.tree = user_data->call_data->root_ntree;
      solver_key.node = &params.node();
    }
    else {
      solver_key.node = static_cast<const void *>(user_data->compute_context);
    }
  }
  SolverHandle *solver_handle = settings.warm_start ? acquire_solver_handle(solver_key) : nullptr;


  RBD_TRACE("rbd trace: paths=%d raw_constraints=%d\n", int(paths.size()), int(raw_constraints.size()));
  for (const StringRef path : paths) {
    GeometrySet *geometry = world.lookup_path_for_write_ptr<GeometrySet>(path);
    if (!geometry) {
      continue;
    }
    if (!tag_filter_matches(geometry_tag_filter, get_geometry_tags(*geometry))) {
      continue;
    }
    /* A cloud with `piece_id` holds per-body data (forces), it is not a set of bodies. The pieces
     * geometry carries that cloud next to its instances now, so only skip it when there are no
     * instances. */
    if (geometry->has_pointcloud() && !geometry->has_instances()) {
      const PointCloud *pointcloud_read = geometry->get_pointcloud();
      if (pointcloud_read && is_constraint_cloud(*pointcloud_read)) {
        continue;
      }
      if (pointcloud_read && pointcloud_read->attributes().contains("piece_id")) {
        continue;
      }
    }

    const bool has_instances = geometry->has_instances();
    const bool has_points = geometry->has_pointcloud();
    if (!has_instances && !has_points) {
      continue;
    }

    Array<BodyState> states;
    bke::Instances *instances = nullptr;
    PointCloud *pointcloud = nullptr;
    if (has_instances) {
      instances = geometry->get_instances_for_write();
      if (!instances || instances->instances_num() == 0) {
        continue;
      }
      states = read_instance_body_states(*instances);
      {
        int massless = 0;
        for (const BodyState &state : states) {
          if (!(state.mass > 1.0e-6f)) {
            massless++;
          }
        }
        if (massless > 0) {
          std::string message = std::to_string(massless) +
                                " rigid body/bodies have no mass (their pieces have no usable "
                                "volume or area). They are given 10 g so they still fall, but check "
                                "the geometry: empty or degenerate pieces can not be simulated "
                                "properly";
          params.error_message_add(NodeWarningType::Warning, message);
        }
      }
      /* Animated bodies follow the pose that comes in through the `Animated` input, which is the
       * *un-simulated* pieces: the simulation zone keeps its state, so the animation of the
       * upstream geometry would otherwise only be seen on the first frame. */
      if (upstream_geometry.has_instances()) {
        const bke::Instances &upstream_instances = *upstream_geometry.get_instances();
        const Span<float4x4> upstream_transforms = upstream_instances.transforms();
        /* Both flags are taken from here: they live on the *pre-zone* pieces, and the zone state
         * would otherwise keep the values it had on the first frame (so switching the animation, or
         * the active flag, would never reach the solver). */
        const VArray<bool> animated_flags = upstream_instances.attributes()
                                                .lookup_or_default<bool>(
                                                    "animated", AttrDomain::Instance, false)
                                                .varray;
        const VArray<bool> active_flags = upstream_instances.attributes()
                                              .lookup_or_default<bool>(
                                                  "active", AttrDomain::Instance, true)
                                              .varray;
        const int upstream_num = std::min(int(upstream_transforms.size()), int(states.size()));
        for (const int i : IndexRange(upstream_num)) {
          states[i].animated = animated_flags[i];
          states[i].active = active_flags[i];
          if (!states[i].animated) {
            continue;
          }
          states[i].position = upstream_transforms[i].location();
          states[i].rotation = math::to_quaternion(upstream_transforms[i]);
        }
      }
    }
    else {
      pointcloud = geometry->get_pointcloud_for_write();
      if (!pointcloud || pointcloud->totpoint == 0) {
        continue;
      }
      states = read_point_body_states(*pointcloud);
    }

    const int body_num = int(states.size());
    RBD_TRACE("rbd trace: path=%s body_num=%d inst=%d points=%d\n",
              path.data(), body_num, int(has_instances), int(has_points));
    const Array<int> body_ids = read_body_ids(*geometry, body_num);
    RBD_TRACE("rbd trace: body_ids done constraint_num=%d\n", int(raw_constraints.size()));
    Map<int, int> id_to_index;
    for (const int i : IndexRange(body_num)) {
      id_to_index.add(body_ids[i], i);
    }

    /* Per-body forces from point clouds (e.g. evaluated from `Blender.Force` bundles by the
     * official `Evaluate Forces` asset) are added on top of the instance/point attributes. */
    for (const RawBodyForce &body_force : raw_body_forces) {
      const int *index = id_to_index.lookup_ptr(body_force.piece_id);
      if (!index) {
        continue;
      }
      states[*index].external_force += body_force.force;
      states[*index].external_torque += body_force.torque;
    }

    Vector<ConstraintDef> constraints;
    Array<int> constraint_source(raw_constraints.size());
    for (const int raw_i : raw_constraints.index_range()) {
      const RawConstraint &raw = raw_constraints[raw_i];
      const int *index_a = id_to_index.lookup_ptr(raw.body_a);
      const int *index_b = id_to_index.lookup_ptr(raw.body_b);
      if (!index_a || !index_b || *index_a == *index_b) {
        continue;
      }
      ConstraintDef def;
      def.body_a = *index_a;
      def.body_b = *index_b;
      def.enabled = raw.enabled;
      def.type = raw.type;
      def.has_anchors = raw.has_anchors;
      def.anchor_a = raw.anchor_a;
      def.anchor_b = raw.anchor_b;
      def.break_impulse = raw.break_impulse;
      def.break_distance = raw.break_distance;
      def.break_angle = raw.break_angle;
      def.frequency = raw.frequency;
      def.damping = raw.damping;
      def.fallback = raw.fallback;
      def.harden_time = std::max(raw.harden_time, 0.0f);
      def.harden_accum = std::max(raw.harden_accum, 0.0f);
      constraint_source[constraints.size()] = raw_i;
      constraints.append(def);
    }
    Array<ConstraintFeedback> broken(constraints.size());

    Vector<BodyShape> shapes;
    Vector<float3> shape_points;
    collect_body_shapes(*geometry, body_num, shapes, shape_points);

    RBD_TRACE("rbd trace: before simulate bodies=%d shapes=%d points=%d\n",
              int(states.size()), int(shapes.size()), int(shape_points.size()));
    simulate_bodies(states,
                    Span<BodyShape>(shapes.data(), shapes.size()),
                    Span<const float3>(shape_points.data(), shape_points.size()),
                    Span<const StaticCollider>(colliders.data(), colliders.size()),
                    Span<const float3>(collider_vertices.data(), collider_vertices.size()),
                    Span<const int>(collider_indices.data(), collider_indices.size()),
                    Span<const ConstraintDef>(constraints.data(), constraints.size()),
                    broken,
                    solver_handle,
                    settings);

    /* Remember which glue constraints broke, they are written back into the attributes below so
     * they stay broken on the next frame. */
    for (const int i : broken.index_range()) {
      constraint_feedback[constraint_source[i]] = broken[i];
    }

    if (instances) {
      write_instance_body_states(*instances, states);
    }
    else if (pointcloud) {
      write_point_body_states(*pointcloud, states);
    }

    write_body_force_positions(world, body_force_paths, raw_body_forces, id_to_index, states);
  }

  /* Write the broken state back into the constraint clouds (the world bundle persists across
   * frames in a simulation zone, so a broken constraint stays broken). */
  if (!constraint_paths.is_empty()) {
    for (const std::string &path : constraint_paths) {
      GeometrySet *geometry = world.lookup_path_for_write_ptr<GeometrySet>(path);
      if (!geometry || !geometry->has_pointcloud()) {
        continue;
      }
      PointCloud *pointcloud = geometry->get_pointcloud_for_write();
      if (!pointcloud) {
        continue;
      }
      bke::MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
      /* A freshly created attribute is zero initialized, which would read as "every constraint is
       * broken". Constraint clouds built by hand may not have the attribute yet. */
      const bool had_enabled = attributes.contains("enabled");
      bke::SpanAttributeWriter<bool> enabled = attributes.lookup_or_add_for_write_span<bool>(
          "enabled", AttrDomain::Point);
      if (!had_enabled) {
        enabled.span.fill(true);
      }
      bke::SpanAttributeWriter<int> types = attributes.lookup_or_add_for_write_span<int>(
          "type", AttrDomain::Point);
      bke::SpanAttributeWriter<float3> anchor_a = attributes.lookup_or_add_for_write_span<float3>(
          "anchor_a", AttrDomain::Point);
      bke::SpanAttributeWriter<float3> anchor_b = attributes.lookup_or_add_for_write_span<float3>(
          "anchor_b", AttrDomain::Point);
      bke::SpanAttributeWriter<float> harden_accums =
          attributes.lookup_or_add_for_write_span<float>("harden_accum", AttrDomain::Point);
      /* World space anchors, so a preview node can draw the connections between the pieces. */
      bke::SpanAttributeWriter<float3> anchor_a_world =
          attributes.lookup_or_add_for_write_span<float3>("anchor_a_world", AttrDomain::Point);
      bke::SpanAttributeWriter<float3> anchor_b_world =
          attributes.lookup_or_add_for_write_span<float3>("anchor_b_world", AttrDomain::Point);
      bke::SpanAttributeWriter<float3> body_a_world =
          attributes.lookup_or_add_for_write_span<float3>("body_a_world", AttrDomain::Point);
      bke::SpanAttributeWriter<float3> body_b_world =
          attributes.lookup_or_add_for_write_span<float3>("body_b_world", AttrDomain::Point);
      for (const int raw_i : raw_constraints.index_range()) {
        const ConstraintFeedback &feedback = constraint_feedback[raw_i];
        const RawConstraint &raw = raw_constraints[raw_i];
        if (raw.source_path != path) {
          continue;
        }
        const int point = raw.source_point;
        if (point < 0 || point >= enabled.span.size()) {
          continue;
        }
        if (feedback.result == ConstraintResult::Disabled) {
          enabled.span[point] = false;
        }
        else if (feedback.result == ConstraintResult::SoftFallback) {
          /* A soft fallback is permanent: the constraint becomes a spring and does not break again. */
          types.span[point] = 1;
          enabled.span[point] = true;
        }
        if (feedback.rest_pose_changed) {
          anchor_a.span[point] = feedback.anchor_a;
          anchor_b.span[point] = feedback.anchor_b;
        }
        harden_accums.span[point] = feedback.harden_accum;
        if (feedback.has_world_anchors) {
          anchor_a_world.span[point] = feedback.anchor_a_world;
          anchor_b_world.span[point] = feedback.anchor_b_world;
          body_a_world.span[point] = feedback.body_a_world;
          body_b_world.span[point] = feedback.body_b_world;
        }
      }
      enabled.finish();
      types.finish();
      anchor_a.finish();
      anchor_b.finish();
      harden_accums.finish();
      anchor_a_world.finish();
      anchor_b_world.finish();
      body_a_world.finish();
      body_b_world.finish();
    }
  }

  params.set_output("World"_ustr, std::move(world_ptr));
}

static void node_label(const bNodeTree * /*ntree*/,
                       const bNode * /*node*/,
                       char *label,
                       const int label_maxncpy)
{
  BLI_strncpy_utf8(
      label, CTX_IFACE_(BLT_I18NCONTEXT_ID_NODETREE, "Rigid Body Solver (WIP)"), label_maxncpy);
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeRigidBodySolver"_ustr);
  ntype.ui_name = "Rigid Body Solver";
  ntype.ui_description = "Simulate rigid body dynamics using Jolt";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  ntype.labelfunc = node_label;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_rigid_body_solver_cc






