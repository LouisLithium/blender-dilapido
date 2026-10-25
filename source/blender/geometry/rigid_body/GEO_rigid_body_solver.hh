/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "BLI_math_quaternion_types.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

namespace blender::geometry::rigid_body {

/**
 * State of a single rigid body. The member names mirror the attribute names used by the
 * geometry nodes physics system, so values can be copied from and to geometry attributes
 * without any translation.
 *
 * This is intentionally a plain data struct: it keeps the Jolt API out of the headers, so
 * node code and other callers do not have to include any Jolt header.
 */
struct BodyState {
  float3 position = float3(0.0f);
  math::Quaternion rotation = math::Quaternion::identity();
  float3 linear_velocity = float3(0.0f);
  float3 angular_velocity = float3(0.0f);
  float3 external_force = float3(0.0f);
  float3 external_torque = float3(0.0f);
  float radius = 0.1f;
  float mass = 1.0f;
  /** Velocity damping per second, matching Jolt's `mLinearDamping` / `mAngularDamping`. */
  float linear_damping = 0.05f;
  float angular_damping = 0.05f;
  /** True when the body is asleep (not integrated until something wakes it). */
  bool sleeping = false;
  /**
   * True when the body follows its animated pose instead of the simulation (a kinematic body). It
   * still collides with the other bodies and pushes them, but nothing can move it. Unchecking the
   * animation in the `Init Rigid Body` node turns it into a normal body, which then keeps the
   * velocity it had while it was animated.
   */
  bool animated = false;
  /**
   * False when the user switched the body off: it is not simulated (nothing can move it), but the
   * other bodies still collide with it, like an animated body without a new pose. Unlike
   * #animated this can change while the simulation runs: the Jolt motion type of the body is
   * switched in place, so the rest of the world keeps its warm started contacts.
   */
  bool active = true;
};

enum class ShapeType : int {
  Sphere = 0,
  Box = 1,
  ConvexHull = 2,
};

/**
 * Collision shape of a body. Convex hulls reference a slice of the point span that is passed
 * to #simulate_bodies, to avoid allocating per body.
 */
struct BodyShape {
  ShapeType type = ShapeType::Sphere;
  /** Used by #ShapeType::Box. */
  float3 half_extents = float3(0.5f);
  /** Used by #ShapeType::ConvexHull, indexes into the `shape_points` span. */
  int points_start = 0;
  int points_count = 0;
};

/**
 * A static mesh collider. Triangles are stored as index triples.
 */
struct StaticCollider {
  int vertices_start = 0;
  int vertices_count = 0;
  int indices_start = 0;
  int indices_count = 0;
  float friction = 0.5f;
  /** Build a solid box around the bounds instead of a triangle mesh (meshes are shells). */
  bool use_box = false;
  /**
   * Build the convex hull of the vertices. Also solid (so bodies can not end up behind the
   * surface), but it follows the shape of the geometry instead of its bounds, and it is much
   * cheaper to collide with than a triangle mesh.
   */
  bool convex = false;
};

struct SimulateSettings {
  float3 gravity = float3(0.0f, 0.0f, -9.81f);
  float delta_time = 1.0f / 60.0f;
  int substeps = 1;
  /** Solver iterations used for constraints (Jolt's velocity steps). */
  int constraint_iterations = 10;
  /**
   * Skip collision detection between two bodies that are glued together. Fragments that were cut
   * apart share their cut faces, so without this the contact solver keeps pushing them apart while
   * the glue pulls them back, which shows up as a pile that slowly spreads out.
   */
  bool disable_glued_collisions = true;
  /**
   * Keep the physics world alive between frames so Jolt can warm start (reuse contacts and
   * constraint impulses). The world is rebuilt whenever the structure or the incoming body state
   * changes, so this only trades frame-to-frame reproducibility for stability, never correctness.
   */
  bool warm_start = true;
  /**
   * Number of threads the solver may use. 0 means "all available cores". One thread makes the
   * result fully reproducible; more threads are usually a large speedup, because Jolt solves the
   * islands of the contact graph in parallel.
   */
  int threads = 0;
  /** Let Jolt put resting bodies to sleep. Sleeping bodies cost almost nothing until they are
   * woken. */
  bool allow_sleeping = true;
  /** Wake every body every frame, which keeps a settled pile from going to sleep. */
  bool wake_all = false;
  /** Point velocity below which a body may fall asleep (m/s). Jolt's default is 0.03. */
  float sleep_threshold = 0.03f;
  /** How long a body has to stay below the threshold before it sleeps (seconds). */
  float time_before_sleep = 0.5f;
  /** Wake both bodies of a constraint when it breaks, so a breaking glue can not leave a body
   * asleep in mid air. */
  bool wake_on_break = true;
  /**
   * Clamp how fast a body may move. Off means "no clamp" (Jolt still has an internal ceiling far
   * above anything a scene does), which is what a fast projectile or a whipped object needs.
   */
  bool speed_limit = false;
  /** Linear speed ceiling in m/s when #speed_limit is on (the angular one is 0.6x this). */
  float max_speed = 50.0f;
  /** Sweep a collider that moved between frames through the substeps instead of teleporting it
   * once per frame. Without it a fast moving collider passes through the bodies. */
  bool sweep_colliders = true;
  /**
   * Take body pairs that already interpenetrate when the world is built out of collision. The
   * contact solver would push them apart with a huge impulse on the first frame, which looks like
   * the pile exploding on its own.
   */
  bool ignore_initial_overlap = true;
};

/** What happened to a constraint during the frame; the caller writes this back into the geometry.
 */
enum class ConstraintResult : uint8_t {
  /** Still holding (or not broken yet). */
  Active = 0,
  /** Broken and disabled. */
  Disabled = 1,
  /** Broken but replaced by a soft constraint: write `type = 1` back. */
  SoftFallback = 2,
};

struct ConstraintFeedback {
  ConstraintResult result = ConstraintResult::Active;
  /**
   * Set when the rest pose was re-baselined by the hardening (the anchors below are then the new
   * rest pose, in body local space relative to the body origin).
   */
  bool rest_pose_changed = false;
  /** Accumulated time the constraint has been deformed, so it survives a world rebuild. */
  float harden_accum = 0.0f;
  float3 anchor_a = float3(0.0f);
  float3 anchor_b = float3(0.0f);
  /**
   * The same anchors in world space (this frame). A preview needs them to draw the connection
   * between the two pieces: the local anchors alone are not enough, and the pieces keep moving.
   */
  float3 anchor_a_world = float3(0.0f);
  float3 anchor_b_world = float3(0.0f);
  /**
   * The centers of the two bodies in world space. The anchors above are the *same* point (the joint
   * of the two pieces), so a line between them would have zero length: a preview draws the line
   * between the centers instead.
   */
  float3 body_a_world = float3(0.0f);
  float3 body_b_world = float3(0.0f);
  bool has_world_anchors = false;
};

/**
 * Opaque handle of the persistent physics world of one solver node. Acquire it once per node with
 * a stable key (e.g. the node's compute context) and pass it to #simulate_bodies; pass nullptr to
 * simulate without warm starting.
 */
struct SolverHandle;

/**
 * Identity of one solver node's persistent physics world.
 *
 * It must not depend on the *compute context*: that changes while descending into nested node
 * groups and between the evaluations of a frame, so the node got several worlds that all wrote the
 * same state. Each of them then rebuilt the world (losing warm starting and re-welding the glue) on
 * every frame, which showed up as a pile that twitched periodically. Object + root tree + node is
 * stable across all evaluations of the same node.
 */
struct SolverWorldKey {
  const void *object = nullptr;
  const void *tree = nullptr;
  const void *node = nullptr;

  bool operator==(const SolverWorldKey &other) const
  {
    return object == other.object && tree == other.tree && node == other.node;
  }
};

SolverHandle *acquire_solver_handle(const SolverWorldKey &key);
void release_solver_handle(const SolverWorldKey &key);

/** Constraint between two bodies, referencing body indices of the `bodies` span. */
struct ConstraintDef {
  int body_a = -1;
  int body_b = -1;
  bool enabled = true;
  /** 0 = fixed (weld), 1 = soft (springy translation, rigid rotation). */
  int type = 0;
  /**
   * What happens when this constraint breaks: 0 disables it, 1 replaces it with a soft constraint
   * (a weld that tears but keeps pulling, instead of letting go completely).
   */
  int fallback = 0;
  /**
   * When > 0, a constraint that stays deformed (but does not break) for this many seconds has its
   * rest pose re-baselined to the current pose, so the deformation becomes the new rest state.
   */
  float harden_time = 0.0f;
  /** Time this constraint has already been deformed (persisted from the previous frame). */
  float harden_accum = 0.0f;
  /**
   * When set, the anchors below are used (in body local space, relative to the center of mass).
   * Otherwise Jolt derives the anchor from the current relative pose, which makes the constraint
   * follow any drift that happened before (bad when the physics world is rebuilt every frame).
   */
  bool has_anchors = false;
  float3 anchor_a = float3(0.0f);
  float3 anchor_b = float3(0.0f);
  /** Break conditions of a glue constraint. Zero or negative disables the check. */
  float break_impulse = 0.0f;
  float break_distance = 0.0f;
  float break_angle = 0.0f;
  /**
   * Soft constraint settings: the oscillation frequency in Hz (higher = stiffer, 0 = hard) and the
   * damping ratio (0 = undamped, 1 = critically damped). A frequency is used instead of a
   * stiffness because it does not depend on the mass of the fragments.
   */
  float frequency = 5.0f;
  float damping = 0.5f;
};

/**
 * Simulate one frame of rigid body dynamics using Jolt.
 *
 * The physics world is rebuilt from the inputs on every call. That keeps the simulation
 * stateless from the caller's side: all state lives in the geometry attributes, which avoids
 * having to store a non-copyable physics world somewhere in the geometry nodes data model.
 * The trade-off is that warm starting and sleeping do not carry over between frames yet.
 *
 * \param bodies: state of the dynamic bodies, updated in place.
 * \param shapes: collision shape per body. If empty, every body is a sphere with its radius.
 * \param shape_points: point pool referenced by convex hull shapes.
 * \param colliders: static mesh colliders (already in world space).
 * \param collider_vertices: vertex pool for the colliders.
 * \param collider_indices: index pool for the colliders (triples).
 * \param constraints: constraints between the bodies, referencing indices into \a bodies.
 * \param r_constraints_broken: output, one flag per entry of \a constraints, true when the
 *   constraint broke this step (glue constraints with a break threshold).
 */
void simulate_bodies(MutableSpan<BodyState> bodies,
                     Span<BodyShape> shapes,
                     Span<const float3> shape_points,
                     Span<const StaticCollider> colliders,
                     Span<const float3> collider_vertices,
                     Span<const int> collider_indices,
                     Span<const ConstraintDef> constraints,
                     MutableSpan<ConstraintFeedback> r_constraint_feedback,
                     SolverHandle *handle,
                     const SimulateSettings &settings);

}  // namespace blender::geometry::rigid_body
