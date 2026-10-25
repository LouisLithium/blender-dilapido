/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Minimal Jolt integration for the rigid body solver. See
 * GEO_rigid_body_solver.hh for the design notes.
 */

#include "GEO_rigid_body_solver.hh"

#include "BLI_math_quaternion.hh"
#include "BLI_math_vector.hh"

// clang-format off
/* The Jolt headers do not include Jolt.h, it must always come first. */
#include <Jolt/Jolt.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Geometry/IndexedTriangle.h>
#include <Jolt/Math/Mat44.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/GroupFilterTable.h>
#include <Jolt/Physics/Collision/NarrowPhaseQuery.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Constraints/Constraint.h>
#include <Jolt/Physics/Constraints/FixedConstraint.h>
#include <Jolt/Physics/Constraints/SixDOFConstraint.h>
#include <Jolt/Physics/Constraints/SpringSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/RegisterTypes.h>
// clang-format on

#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace blender::geometry::rigid_body {
using namespace JPH;
namespace {
/* Two object layers are enough for now: static colliders and dynamic bodies. */
constexpr ObjectLayer k_layer_non_moving = 0;
constexpr ObjectLayer k_layer_moving = 1;
constexpr uint k_num_object_layers = 2;
struct BroadphaseLayers {
  static constexpr BroadPhaseLayer non_moving{0};
  static constexpr BroadPhaseLayer moving{1};
  static constexpr uint num_layers = 2;
};
class BPLayerInterfaceImpl final : public BroadPhaseLayerInterface {
 public:
  BPLayerInterfaceImpl()
  {
    object_to_broadphase_[k_layer_non_moving] = BroadphaseLayers::non_moving;
    object_to_broadphase_[k_layer_moving] = BroadphaseLayers::moving;
  }
  uint GetNumBroadPhaseLayers() const override
  {
    return BroadphaseLayers::num_layers;
  }
  BroadPhaseLayer GetBroadPhaseLayer(ObjectLayer inLayer) const override
  {
    JPH_ASSERT(inLayer < k_num_object_layers);
    return object_to_broadphase_[inLayer];
  }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
  const char *GetBroadPhaseLayerName(BroadPhaseLayer inLayer) const override
  {
    switch ((BroadPhaseLayer::Type)inLayer) {
      case (BroadPhaseLayer::Type)BroadphaseLayers::non_moving:
        return "NON_MOVING";
      case (BroadPhaseLayer::Type)BroadphaseLayers::moving:
        return "MOVING";
      default:
        JPH_ASSERT(false);
        return "INVALID";
    }
  }
#endif
 private:
  BroadPhaseLayer object_to_broadphase_[BroadphaseLayers::num_layers];
};
class ObjectVsBroadPhaseLayerFilterImpl final : public ObjectVsBroadPhaseLayerFilter {
 public:
  bool ShouldCollide(ObjectLayer inLayer1, BroadPhaseLayer inLayer2) const override
  {
    switch (inLayer1) {
      case k_layer_non_moving:
        return inLayer2 == BroadphaseLayers::moving;
      case k_layer_moving:
        return true;
      default:
        JPH_ASSERT(false);
        return false;
    }
  }
};
class ObjectLayerPairFilterImpl final : public ObjectLayerPairFilter {
 public:
  bool ShouldCollide(ObjectLayer inObject1, ObjectLayer inObject2) const override
  {
    switch (inObject1) {
      case k_layer_non_moving:
        return inObject2 == k_layer_moving;
      case k_layer_moving:
        return true;
      default:
        JPH_ASSERT(false);
        return false;
    }
  }
};
inline Vec3 to_jolt(const float3 &v)
{
  return Vec3(v.x, v.y, v.z);
}
inline RVec3 to_jolt_r(const float3 &v)
{
  return RVec3(v.x, v.y, v.z);
}
inline Quat to_jolt(const math::Quaternion &q)
{
  return Quat(q.x, q.y, q.z, q.w);
}
inline float3 to_blender(Vec3Arg v)
{
  return float3(v.GetX(), v.GetY(), v.GetZ());
}
/** Bookkeeping for a constraint that was added to the world, for the break evaluation. */
struct ActiveConstraint {
  int def_index = -1;
  /** Indices into the body arrays, for the world space anchors of the preview. */
  int body_index_a = -1;
  int body_index_b = -1;
  Ref<Constraint> constraint;
  BodyID body_a;
  BodyID body_b;
  Vec3 com_a = Vec3::sZero();
  Vec3 com_b = Vec3::sZero();
  Vec3 anchor_a = Vec3::sZero();
  Vec3 anchor_b = Vec3::sZero();
  bool has_anchors = false;
  /** Impulse the constraint applied so far in this frame (N s), for the break check. */ float
      impulse_accumulator = 0.0f;
  /** Outcome of the constraint during this frame, written back by the caller. */ ConstraintResult
      result = ConstraintResult::Active;
  /** Seconds the constraint has been deformed within its limits, for the hardening. */ float
      harden_accumulator = 0.0f;
  /** True once the rest pose was re-baselined;
 #anchor_a / #anchor_b are then the new rest pose. */
  bool rest_pose_changed = false;
  /** The new rest pose to write back (body local, relative to the body origin). */ float3
      rest_anchor_a = float3(0.0f);
  float3 rest_anchor_b = float3(0.0f);
};
/** Called once before the first use of Jolt. */ void initialize_jolt()
{
  static std::once_flag once;
  std::call_once(once, []() {
    RegisterDefaultAllocator();
    Factory::sInstance = new Factory();
    RegisterTypes();
  });
}
RefConst<Shape> make_body_shape(const BodyShape &shape,
                                const BodyState &state,
                                const float3 *shape_points)
{
  switch (shape.type) {
    case ShapeType::Box: {
      const Vec3 half_extents(std::max(shape.half_extents.x, 1.0e-3f),
                              std::max(shape.half_extents.y, 1.0e-3f),
                              std::max(shape.half_extents.z, 1.0e-3f));
      return new BoxShape(half_extents);
    }
    case ShapeType::ConvexHull: {
      if (shape.points_count >= 4) {
        /* A flat island (a single quad, a wall without thickness, the remains of an open mesh)
         * would only be inflated by the convex radius, which is far too thin to be hit reliably:
         * such a body tunnels through everything. Give it a minimum thickness by using a solid box
         * instead (checked below). */
        float3 flat_min(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 flat_max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (const int i : IndexRange(shape.points_count)) {
          const float3 &p = shape_points[shape.points_start + i];
          flat_min.x = std::min(flat_min.x, p.x);
          flat_min.y = std::min(flat_min.y, p.y);
          flat_min.z = std::min(flat_min.z, p.z);
          flat_max.x = std::max(flat_max.x, p.x);
          flat_max.y = std::max(flat_max.y, p.y);
          flat_max.z = std::max(flat_max.z, p.z);
        }
        const float3 flat_extents = flat_max - flat_min;
        const float largest_extent = std::max({flat_extents.x, flat_extents.y, flat_extents.z});
        const float smallest_extent = std::min({flat_extents.x, flat_extents.y, flat_extents.z});
        const bool is_flat = smallest_extent < std::max(0.004f, largest_extent * 0.02f);
        if (!is_flat) {
          /* The convex radius inflates the shape, but neighbouring fragments were cut apart and
           * their faces coincide: without compensating, adjacent pieces overlap by twice the
           * radius and the contact solver pushes them apart as soon as the simulation starts (the
           * pieces visibly "pop" on the first frame). Scale the points inwards so the collision
           * surface stays where the fragment geometry is. */
          constexpr float convex_radius = 0.002f;
          JPH::Array<Vec3> points;
          points.reserve(shape.points_count);
          float max_extent = 0.0f;
          for (const int i : IndexRange(shape.points_count)) {
            const float3 &p = shape_points[shape.points_start + i];
            const Vec3 point(p.x, p.y, p.z);
            max_extent = std::max(max_extent, point.Length());
            points.push_back(point);
          }
          if (max_extent > convex_radius) {
            const float scale = (max_extent - convex_radius) / max_extent;
            for (Vec3 &point : points) {
              point *= scale;
            }
          }
          /* Jolt builds the hull with a fixed 1 mm accuracy, which is impossible for a long or a
           * very thin piece (a splinter of a large object, or a piece of a shell): the build failed
           * and the body fell back to its bounding box, so e.g. a 30 m long 4 cm thin shell piece
           * became a solid half meter thick box that blasted every small fragment around it. The
           * tolerance has to grow with the size of the piece.
           * Note that `ShapeSettings::Create()` caches its result, so every attempt needs a fresh
           * settings object (retrying on the same one returns the cached failure). */
          const float tolerances[] = {std::max(1.0e-3f, max_extent * 1.0e-3f),
                                      std::max(1.0e-2f, max_extent * 1.0e-2f),
                                      std::max(5.0e-2f, max_extent * 5.0e-2f)};
          for (const float hull_tolerance : tolerances) {
            ConvexHullShapeSettings settings(points, convex_radius);
            settings.mHullTolerance = hull_tolerance;
            const ShapeSettings::ShapeResult result = settings.Create();
            if (!result.HasError()) {
              return result.Get();
            }
            if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
              printf("RBD_DEBUG convex hull failed (%d points, tolerance %.4f): %s\n",
                     shape.points_count,
                     double(hull_tolerance),
                     result.GetError().c_str());
            }
          }
        }
        else if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
          printf("RBD_DEBUG flat body (%d points, extents %.3f/%.3f/%.3f), using a box\n",
                 shape.points_count,
                 double(flat_extents.x),
                 double(flat_extents.y),
                 double(flat_extents.z));
        }
      }
      /* A body without a usable shape collides with nothing at all, so it would fall straight
       * through the level. Fall back to a box around the points; the default sphere below uses the
       * radius attribute, which can be ~0 and is then invisible to the collision detection. */
      if (shape.points_count > 0) {
        float3 min_point(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (const int i : IndexRange(shape.points_count)) {
          const float3 &p = shape_points[shape.points_start + i];
          min_point.x = std::min(min_point.x, p.x);
          min_point.y = std::min(min_point.y, p.y);
          min_point.z = std::min(min_point.z, p.z);
          max_point.x = std::max(max_point.x, p.x);
          max_point.y = std::max(max_point.y, p.y);
          max_point.z = std::max(max_point.z, p.z);
        }
        const float3 half_extents = (max_point - min_point) * 0.5f;
        /* 5 mm minimum thickness: a flat island would otherwise be a paper thin box that fast
         * bodies tunnel through. */
        return new BoxShape(Vec3(std::max(half_extents.x, 5.0e-3f),
                                 std::max(half_extents.y, 5.0e-3f),
                                 std::max(half_extents.z, 5.0e-3f)));
      }
      break;
    }
    case ShapeType::Sphere:
      break;
  }
  /* 1 cm minimum: a body with a ~0 radius is invisible to the collision detection and would fall
   * * through every collider in the scene. */
  return new SphereShape(std::max(state.radius, 0.01f));
}
/**
 * \param kinematic: build the collider as a kinematic body with `velocity`, so it sweeps through the
 *   substeps instead of jumping once per frame (a fast moving collider would otherwise teleport past
 *   the bodies and never generate a contact).
 * \param offset: where the body starts, relative to the geometry (the sweep starts one frame back).
 */
BodyID add_static_collider(BodyInterface &body_interface,
                           const StaticCollider &collider,
                           const float3 *vertices,
                           const int *indices,
                           const bool kinematic,
                           const float3 offset,
                           const float3 velocity)
{
  if (collider.convex) {
    if (collider.vertices_count < 4) {
      return BodyID();
    }
  }
  else if (collider.vertices_count < 3 || collider.indices_count < 3) {
    return BodyID();
  }
  const EMotionType motion_type = kinematic ? EMotionType::Kinematic : EMotionType::Static;
  if (collider.use_box) {
    /* A solid box around the bounds. Unlike a triangle mesh (which is a shell with one-sided     *
     * triangles) a box has volume: a fragment that sinks into it is pushed back out instead of
     * * disappearing behind the surface. */
    float3 min_point(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const int i : IndexRange(collider.vertices_count)) {
      const float3 &p = vertices[collider.vertices_start + i];
      min_point.x = std::min(min_point.x, p.x);
      min_point.y = std::min(min_point.y, p.y);
      min_point.z = std::min(min_point.z, p.z);
      max_point.x = std::max(max_point.x, p.x);
      max_point.y = std::max(max_point.y, p.y);
      max_point.z = std::max(max_point.z, p.z);
    }
    const float3 center = (min_point + max_point) * 0.5f;
    const float3 half_extents = (max_point - min_point) * 0.5f;
    const Vec3 half(std::max(half_extents.x, 1.0e-3f),
                    std::max(half_extents.y, 1.0e-3f),
                    std::max(half_extents.z, 1.0e-3f));
    BodyCreationSettings body_settings(new BoxShape(half),
                                       to_jolt_r(center + offset),
                                       Quat::sIdentity(),
                                       motion_type,
                                       k_layer_non_moving);
    body_settings.mFriction = collider.friction;
    body_settings.mLinearVelocity = to_jolt(velocity);
    const BodyID box_id = body_interface.CreateAndAddBody(
        body_settings, kinematic ? EActivation::Activate : EActivation::DontActivate);
    if (kinematic && !box_id.IsInvalid()) {
      body_interface.SetLinearVelocity(box_id, to_jolt(velocity));
    }
    return box_id;
  }
  if (collider.convex) {
    /* A solid convex hull of the vertices: it has volume like the box, but follows the shape of the
     * geometry (a rock, a bowl, ...) instead of its bounds. Much cheaper to collide with than the
     * triangle mesh. */
    if (collider.vertices_count < 4) {
      return BodyID();
    }
    JPH::Array<Vec3> points;
    points.reserve(collider.vertices_count);
    for (const int i : IndexRange(collider.vertices_count)) {
      const float3 &p = vertices[collider.vertices_start + i];
      points.push_back(Vec3(p.x, p.y, p.z));
    }
    ConvexHullShapeSettings settings(points, 0.001f);
    const ShapeSettings::ShapeResult result = settings.Create();
    if (result.HasError()) {
      if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
        printf("RBD_DEBUG convex collider failed (%d points)\n", collider.vertices_count);
      }
      return BodyID();
    }
    BodyCreationSettings body_settings(result.Get(),
                                       to_jolt_r(offset),
                                       Quat::sIdentity(),
                                       motion_type,
                                       k_layer_non_moving);
    body_settings.mFriction = collider.friction;
    body_settings.mLinearVelocity = to_jolt(velocity);
    const BodyID convex_id = body_interface.CreateAndAddBody(
        body_settings, kinematic ? EActivation::Activate : EActivation::DontActivate);
    if (kinematic && !convex_id.IsInvalid()) {
      body_interface.SetLinearVelocity(convex_id, to_jolt(velocity));
    }
    return convex_id;
  }
  VertexList jolt_vertices;
  jolt_vertices.reserve(collider.vertices_count);
  for (const int i : IndexRange(collider.vertices_count)) {
    const float3 &p = vertices[collider.vertices_start + i];
    jolt_vertices.push_back(Float3(p.x, p.y, p.z));
  }
  IndexedTriangleList jolt_triangles;
  const int triangle_count = collider.indices_count / 3;
  jolt_triangles.reserve(triangle_count);
  for (const int i : IndexRange(triangle_count)) {
    const int base = collider.indices_start + i * 3;
    jolt_triangles.push_back(IndexedTriangle(
        uint32(indices[base]), uint32(indices[base + 1]), uint32(indices[base + 2]), 0));
  }
  MeshShapeSettings settings(std::move(jolt_vertices), std::move(jolt_triangles));
  const ShapeSettings::ShapeResult result = settings.Create();
  if (result.HasError()) {
    return BodyID();
  }
  BodyCreationSettings body_settings(result.Get(),
                                     to_jolt_r(offset),
                                     Quat::sIdentity(),
                                     motion_type,
                                     k_layer_non_moving);
  body_settings.mFriction = collider.friction;
  body_settings.mLinearVelocity = to_jolt(velocity);
  const BodyID id = body_interface.CreateAndAddBody(
      body_settings, kinematic ? EActivation::Activate : EActivation::DontActivate);
  if (kinematic && !id.IsInvalid()) {
    /* Jolt does not always pick the creation time velocity up for a kinematic body, so it is set
     * again explicitly. */
    body_interface.SetLinearVelocity(id, to_jolt(velocity));
  }
  return id;
}
}  // namespace
   // namespace

/* -------------------------------------------------------------------------------------------------
 * * Warm starting * * The physics world used to be rebuilt from the attributes on every frame,
 * which threw away Jolt's * contact and constraint caches (no warm starting). To keep them, the
 * world and everything it owns * lives in a `SolverContext` that is cached per solver node. The
 * world is only reused when the * structure (bodies, shapes, constraints, colliders) and the
 * incoming body state are unchanged; otherwise it is rebuilt from scratch, so a stale cache can
 * never produce wrong results (a scrub, * a bake, an edit or a moved body simply rebuilds). *
 * --------------------------------------------------------------------------------------------- */
/** Keeps a shape reference alive (containers of const elements are not allowed). */
struct ShapeRef {
  RefConst<Shape> shape;
};
/** Everything that has to survive between frames for warm starting. */ struct SolverContext {
  BPLayerInterfaceImpl broad_phase_layer_interface;
  ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
  ObjectLayerPairFilterImpl object_layer_pair_filter;
  /* The temp allocator is where Jolt puts everything it needs while stepping (pairs, contacts,   *
   * constraint rows). A fixed block is not enough for a big pile: with a few thousand fragments
   * that   * touch each other the 16 MB block ran out in the first step and Jolt crashed (the user
   * hit this   * with 1653 bodies / 19395 constraints). The fallback variant keeps the
   * preallocated block for the   * common case and continues with malloc when it is exhausted, so
   * it can never run out. */
  TempAllocatorImplWithMallocFallback temp_allocator{32 * 1024 * 1024};
  /**   * The job system is recreated when the requested thread count changes, so it is held by
 pointer.   * One thread keeps the result deterministic, which is the baseline; the solver node
 defaults to   * using all cores because the contact solver of a large pile is the dominant cost.
 */
  std::unique_ptr<JobSystemThreadPool> job_system;
  int threads = -2;
  PhysicsSystem physics_system;
  Ref<GroupFilterTable> glue_filter;
  /* Jolt does not own the shapes, they have to stay alive as long as the bodies use them. */ std::
      vector<ShapeRef>
          shapes;
  std::vector<BodyID> body_ids;
  /** Bodies of the colliders, so a deforming collider can be replaced without a world rebuild. */
  std::vector<BodyID> collider_body_ids;
  /** Hash of the geometry each collider body was built from, so it can be refreshed on change. */
  std::vector<uint64_t> collider_hashes;
  /**
   * The body states this world ended the last frame with, as they were written back into the
   * geometry. The next frame's input is compared against these: that is exact, while comparing
   * against the bodies inside the physics world goes through a quaternion -> matrix ->
   * quaternion round trip and drifts past the tolerance, which rebuilt the world (and re-welded
   * every glue constraint to its original rest pose) on almost every frame.
   */
  std::vector<BodyState> written_states;
  std::vector<Ref<Constraint>> constraint_refs;
  std::vector<ActiveConstraint> active_constraints;
  uint64_t signature = 0;
};
struct SolverHandle {
  std::unique_ptr<SolverContext> context;
  /**
   * Centroids of the colliders in the previous frame. Kept here (not in the context) because the
   * context is thrown away whenever the world is rebuilt, and the sweep needs the history across
   * rebuilds.
   */
  std::vector<float3> collider_centers;
};
static uint64_t hash_mix(uint64_t hash, uint64_t value)
{
  return hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
}
static uint64_t hash_point(uint64_t hash, const float3 &point)
{
  hash = hash_mix(hash, uint64_t(std::llround(double(point.x) * 1.0e4)));
  hash = hash_mix(hash, uint64_t(std::llround(double(point.y) * 1.0e4)));
  hash = hash_mix(hash, uint64_t(std::llround(double(point.z) * 1.0e4)));
  return hash;
}
/** Hash of everything that changes the layout of the physics world (never of the dynamic state).
 */
static uint64_t context_signature(Span<BodyState> bodies,
                                  Span<BodyShape> shapes,
                                  const float3 *shape_points,
                                  const int shape_points_num,
                                  const StaticCollider *colliders,
                                  const int colliders_num,
                                  const float3 *collider_vertices,
                                  const int collider_vertices_num,
                                  const int collider_indices_num,
                                  const ConstraintDef *constraints,
                                  const int constraints_num,
                                  const float3 gravity,
                                  const bool allow_sleeping,
                                  const bool ignore_initial_overlap,
                                  const bool speed_limit,
                                  const float max_speed,
                                  const float sleep_threshold,
                                  const float time_before_sleep)
{
  uint64_t hash = 0xcbf29ce484222325ull;
  hash = hash_mix(hash, uint64_t(bodies.size()));
  /* The animated flag decides the motion type of a body, which is part of the world layout. */
  for (const BodyState &body : bodies) {
    hash = hash_mix(hash, uint64_t(body.animated));
  }
  hash = hash_mix(hash, uint64_t(std::llround(double(gravity.x) * 1.0e3)));
  hash = hash_mix(hash, uint64_t(std::llround(double(gravity.y) * 1.0e3)));
  hash = hash_mix(hash, uint64_t(std::llround(double(gravity.z) * 1.0e3)));
  /* Sleeping is part of the physics settings, which are only applied when the world is built. */
  hash = hash_mix(hash, uint64_t(allow_sleeping));
  hash = hash_mix(hash, uint64_t(ignore_initial_overlap));
  hash = hash_mix(hash, uint64_t(speed_limit));
  hash = hash_mix(hash, uint64_t(std::llround(double(max_speed) * 1.0e3)));
  hash = hash_mix(hash, uint64_t(std::llround(double(sleep_threshold) * 1.0e4)));
  hash = hash_mix(hash, uint64_t(std::llround(double(time_before_sleep) * 1.0e4)));
  hash = hash_mix(hash, uint64_t(colliders_num));
  hash = hash_mix(hash, uint64_t(constraints_num));
  for (const BodyShape &shape : shapes) {
    hash = hash_mix(hash, uint64_t(shape.type));
    hash = hash_mix(hash, uint64_t(shape.points_start));
    hash = hash_mix(hash, uint64_t(shape.points_count));
  }
  /* A coarse hash of the shape points, so a different fracture with the same piece count   *
   * invalidates the world as well. */
  uint64_t points_hash = 0;
  for (const int i : IndexRange(shape_points_num)) {
    points_hash = hash_point(points_hash, shape_points[i]);
  }
  hash = hash_mix(hash, points_hash);
  /* Collider geometry is never part of the signature: a collider that moves or deforms gets its
   * body refreshed in place instead, which keeps the dynamic bodies' warm started contacts. */
  for (const int i : IndexRange(constraints_num)) {
    const ConstraintDef &constraint = constraints[i];
    hash = hash_mix(hash, uint64_t(constraint.body_a));
    hash = hash_mix(hash, uint64_t(constraint.body_b));
    hash = hash_mix(hash, uint64_t(constraint.type));
    /* The break thresholds and spring settings are part of the cached state (they decide when a
     * * constraint gets disabled), so they have to invalidate the world as well: two solver nodes
     * * that only differ in these must not share a cached world. */
    hash = hash_mix(hash, uint64_t(std::llround(double(constraint.break_impulse) * 1.0e3)));
    hash = hash_mix(hash, uint64_t(std::llround(double(constraint.break_distance) * 1.0e3)));
    hash = hash_mix(hash, uint64_t(std::llround(double(constraint.break_angle) * 1.0e3)));
    hash = hash_mix(hash, uint64_t(std::llround(double(constraint.frequency) * 1.0e3)));
    hash = hash_mix(hash, uint64_t(std::llround(double(constraint.damping) * 1.0e3)));
  }
  return hash;
}
/** * True when the body states the caller passes in are the ones the world ended the previous
 * frame * with. If they differ, the caller's state is authoritative (a scrub, a bake, an edit) and
 * the * world has to be rebuilt. */
static bool finite3(const float3 &value)
{
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}
/** True when every value of a state is a real number (no NaN / inf). */
static bool state_is_finite(const BodyState &state)
{
  return std::isfinite(state.position.x) && std::isfinite(state.position.y) &&
         std::isfinite(state.position.z) && std::isfinite(state.rotation.w) &&
         std::isfinite(state.rotation.x) && std::isfinite(state.rotation.y) &&
         std::isfinite(state.rotation.z) && std::isfinite(state.linear_velocity.x) &&
         std::isfinite(state.linear_velocity.y) && std::isfinite(state.linear_velocity.z) &&
         std::isfinite(state.angular_velocity.x) && std::isfinite(state.angular_velocity.y) &&
         std::isfinite(state.angular_velocity.z);
}
static bool context_state_matches(const SolverContext &ctx, Span<BodyState> bodies)
{
  if (ctx.body_ids.size() != bodies.size()) {
    return false;
  }
  /* Prefer the states that were written back last frame: they are what the caller is expected to
   * hand in again, and they are exact. Comparing against the bodies inside the physics world goes
   * through a quaternion -> matrix -> quaternion round trip, which drifts past the tolerance and
   * rebuilt the world (re-welding every glue constraint to its original rest pose) on most frames. */
  if (ctx.written_states.size() == bodies.size()) {
    float worst_delta_sq = 0.0f;
    float worst_dot = 1.0f;
    bool mismatch = false;
    for (const int i : bodies.index_range()) {
      if (!state_is_finite(bodies[i])) {
        return false;
      }
      if (bodies[i].animated) {
        /* An animated body is *supposed* to come in with a different pose every frame: the node
         * tree animates it and the solver only follows. Comparing it would rebuild the world on
         * every frame. */
        continue;
      }
      const BodyState &previous = ctx.written_states[i];
      const float3 delta = previous.position - bodies[i].position;
      const float delta_sq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
      const float dot = std::abs(previous.rotation.w * bodies[i].rotation.w +
                                 previous.rotation.x * bodies[i].rotation.x +
                                 previous.rotation.y * bodies[i].rotation.y +
                                 previous.rotation.z * bodies[i].rotation.z);
      worst_delta_sq = std::max(worst_delta_sq, delta_sq);
      worst_dot = std::min(worst_dot, dot);
      if (!(delta_sq <= 1.0e-6f) || !(dot >= 0.9999f)) {
        mismatch = true;
      }
    }
    if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
      int bad = 0;
      int worst_index = -1;
      float worst_sq = 0.0f;
      int sleeping_bad = 0;
      for (const int i : bodies.index_range()) {
        const float3 delta = ctx.written_states[i].position - bodies[i].position;
        const float delta_sq = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        if (!(delta_sq <= 1.0e-6f)) {
          bad++;
          sleeping_bad += bodies[i].sleeping ? 1 : 0;
          if (delta_sq > worst_sq) {
            worst_sq = delta_sq;
            worst_index = i;
          }
        }
      }
      printf("RBD_DEBUG state_check written bad=%d/%d worst=%d worst_delta=%.4f "
             "sleeping_bad=%d min_dot=%.6f -> %s\n",
             bad,
             int(bodies.size()),
             worst_index,
             double(std::sqrt(worst_sq)),
             sleeping_bad,
             double(worst_dot),
             mismatch ? "rebuild" : "reuse");
      fflush(stdout);
    }
    return !mismatch;
  }
  for (const int i : bodies.index_range()) {
    const BodyID body_id = ctx.body_ids[i];
    if (body_id.IsInvalid()) {
      return false;
    }
    /* A NaN state would compare as "matching" below (every comparison with NaN is false), and the
     * * world would then be reused with a body that poisons the whole solver. */
    if (!state_is_finite(bodies[i])) {
      return false;
    }
    const Body *body = ctx.physics_system.GetBodyLockInterfaceNoLock().TryGetBody(body_id);
    if (body == nullptr) {
      return false;
    }
    const float3 delta = to_blender(Vec3(body->GetPosition())) - bodies[i].position;
    if (!(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z <= 1.0e-6f)) {
      return false;
    }
    const Quat rotation = body->GetRotation();
    const float dot = std::abs(
        rotation.GetW() * bodies[i].rotation.w + rotation.GetX() * bodies[i].rotation.x +
        rotation.GetY() * bodies[i].rotation.y + rotation.GetZ() * bodies[i].rotation.z);
    if (!(dot >= 0.9999f)) {
      return false;
    }
  }
  return true;
}
static std::mutex &handle_mutex()
{
  static std::mutex mutex;
  return mutex;
}
struct SolverWorldKeyHasher {
  uint64_t operator()(const SolverWorldKey &key) const
  {
    uint64_t hash = 0xcbf29ce484222325ull;
    for (const void *pointer : {key.object, key.tree, key.node}) {
      hash ^= uint64_t(reinterpret_cast<uintptr_t>(pointer));
      hash *= 0x100000001b3ull;
    }
    return hash;
  }
};

static std::unordered_map<SolverWorldKey, std::unique_ptr<SolverHandle>, SolverWorldKeyHasher> &
handle_map()
{
  static std::unordered_map<SolverWorldKey, std::unique_ptr<SolverHandle>, SolverWorldKeyHasher>
      map;
  return map;
}
/** The persistent world of one solver node (null when warm starting is not available). */
SolverHandle *acquire_solver_handle(const SolverWorldKey &key)
{
  if (key == SolverWorldKey{}) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(handle_mutex());
  auto &map = handle_map();
  const auto it = map.find(key);
  if (it != map.end()) {
    return it->second.get();
  }
  /* Bound the cache: solver nodes that are gone would otherwise keep their world forever. */
  if (map.size() >= 64) {
    map.erase(map.begin());
  }
  auto handle = std::make_unique<SolverHandle>();
  SolverHandle *result = handle.get();
  map.emplace(key, std::move(handle));
  return result;
}
void release_solver_handle(const SolverWorldKey &key)
{
  if (key == SolverWorldKey{}) {
    return;
  }
  std::lock_guard<std::mutex> lock(handle_mutex());
  handle_map().erase(key);
}
void simulate_bodies(MutableSpan<BodyState> bodies,
                     Span<BodyShape> shapes,
                     Span<const float3> shape_points,
                     Span<const StaticCollider> colliders,
                     Span<const float3> collider_vertices,
                     Span<const int> collider_indices,
                     Span<const ConstraintDef> constraints,
                     MutableSpan<ConstraintFeedback> r_constraint_feedback,
                     SolverHandle *handle,
                     const SimulateSettings &settings)
{
  for (ConstraintFeedback &feedback : r_constraint_feedback) {
    feedback = ConstraintFeedback{};
  }
  if (bodies.is_empty() && colliders.is_empty()) {
    return;
  }
  initialize_jolt();
  const uint num_bodies = uint(bodies.size() + colliders.size());
  /* Jolt silently drops body pairs and contacts beyond these limits. In a dense pile that shows up
   * * as a fixed subset of pieces that stop colliding with the ground and fall straight through:
   * the   * collider is a triangle mesh with one-sided triangles, so once a fragment is behind the
   * surface   * it never gets a contact again. Jolt's own samples use 65536 pairs / 10240 contacts
   * for a scene   * of about a thousand bodies, and a tightly packed stack needs roughly ten pairs
   * and ten contacts   * per body, so the old `bodies * 4` (which collapsed to the 1024 minimum)
   * was far too small. */
  const uint max_body_pairs = std::max(65536u, num_bodies * 16u);
  const uint max_contact_constraints = std::max(10240u, num_bodies * 16u);
  /* Warm start: reuse the physics world of the previous frame when the structure (bodies, shapes,
   * * constraints, colliders) and the incoming body state are unchanged, so Jolt can reuse its   *
   * contact and constraint caches. Otherwise the world is rebuilt from scratch, which makes a   *
   * stale cache impossible: a scrub, a bake, an edit or a moved body simply rebuilds. */
  const uint64_t signature = context_signature(bodies,
                                               shapes,
                                               shape_points.data(),
                                               int(shape_points.size()),
                                               colliders.data(),
                                               int(colliders.size()),
                                               collider_vertices.data(),
                                               int(collider_vertices.size()),
                                               int(collider_indices.size()),
                                               constraints.data(),
                                               int(constraints.size()),
                                               settings.gravity,
                                               settings.allow_sleeping,
                                               settings.ignore_initial_overlap,
                                               settings.speed_limit,
                                               settings.max_speed,
                                               settings.sleep_threshold,
                                               settings.time_before_sleep);
  std::unique_ptr<SolverContext> owned_context;
  SolverContext *ctx = nullptr;
  bool reuse = false;
  if (handle != nullptr && settings.warm_start && handle->context != nullptr &&
      handle->context->signature == signature && context_state_matches(*handle->context, bodies))
  {
    ctx = handle->context.get();
    reuse = true;
  }
  else {
    owned_context = std::make_unique<SolverContext>();
    ctx = owned_context.get();
    ctx->signature = signature;
  }
  /* Set BLENDER_RBD_DEBUG=1 to see whether the persistent world is being reused (warm start) or
   * * rebuilt. */
  if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
    printf("RBD_DEBUG handle=%p ctx=%p wrote=%d\n",
           (const void *)handle,
           (const void *)ctx,
           int(ctx->written_states.size()));
    fflush(stdout);
    printf("RBD_DEBUG %s bodies=%d constraints=%d signature=%llu\n",
           reuse ? "reuse" : "rebuild",
           int(bodies.size()),
           int(constraints.size()),
           (unsigned long long)signature);
    fflush(stdout);
  }
  if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
    float3 min_p(1.0e30f, 1.0e30f, 1.0e30f), max_p(-1.0e30f, -1.0e30f, -1.0e30f);
    float max_speed = 0.0f, max_spin = 0.0f;
    int non_finite = 0;
    for (const BodyState &state : bodies) {
      if (!state_is_finite(state)) {
        non_finite++;
        continue;
      }
      min_p.x = std::min(min_p.x, state.position.x);
      max_p.x = std::max(max_p.x, state.position.x);
      min_p.y = std::min(min_p.y, state.position.y);
      max_p.y = std::max(max_p.y, state.position.y);
      min_p.z = std::min(min_p.z, state.position.z);
      max_p.z = std::max(max_p.z, state.position.z);
      max_speed = std::max(max_speed,
                           std::sqrt(state.linear_velocity.x * state.linear_velocity.x +
                                     state.linear_velocity.y * state.linear_velocity.y +
                                     state.linear_velocity.z * state.linear_velocity.z));
      max_spin = std::max(max_spin,
                          std::sqrt(state.angular_velocity.x * state.angular_velocity.x +
                                    state.angular_velocity.y * state.angular_velocity.y +
                                    state.angular_velocity.z * state.angular_velocity.z));
    }
    printf(
        "RBD_DEBUG state pos=[%.2f %.2f %.2f]..[%.2f %.2f %.2f] max_speed=%.2f max_spin=%.2f "
        "nonfinite=%d\n",
        min_p.x,
        min_p.y,
        min_p.z,
        max_p.x,
        max_p.y,
        max_p.z,
        max_speed,
        max_spin,
        non_finite);
    fflush(stdout);
  }
  /* Bind the persistent members to the names used below. */ BPLayerInterfaceImpl
      &broad_phase_layer_interface = ctx->broad_phase_layer_interface;
  ObjectVsBroadPhaseLayerFilterImpl &object_vs_broadphase_layer_filter =
      ctx->object_vs_broadphase_layer_filter;
  ObjectLayerPairFilterImpl &object_layer_pair_filter = ctx->object_layer_pair_filter;
  TempAllocator &temp_allocator = ctx->temp_allocator;
  /* Jolt's job system is what parallelises the contact solver, so it is worth using all cores   *
   * (which is what a thread count of -1 means). Recreated only when the setting changes. */
  const int threads = settings.threads > 0 ? settings.threads : -1;
  if (ctx->job_system == nullptr || ctx->threads != threads) {
    ctx->job_system = std::make_unique<JobSystemThreadPool>(
        cMaxPhysicsJobs, cMaxPhysicsBarriers, threads);
    ctx->threads = threads;
  }
  JobSystemThreadPool &job_system = *ctx->job_system;
  PhysicsSystem &physics_system = ctx->physics_system;
  Ref<GroupFilterTable> &glue_filter = ctx->glue_filter;
  std::vector<BodyID> &body_ids = ctx->body_ids;
  std::vector<Ref<Constraint>> &constraint_refs = ctx->constraint_refs;
  std::vector<ActiveConstraint> &active_constraints = ctx->active_constraints;
  BodyInterface &body_interface = physics_system.GetBodyInterface();
  /* Collider motion. The collider geometry is sampled once per frame, so a collider that moves fast
   * would jump from one frame's pose to the next without ever touching the bodies in between. When
   * it moved, its body is created one frame back and given the matching velocity: Jolt then moves it
   * *during* the substeps, which is what makes fast moving colliders collide. */
  const float collider_dt = std::max(settings.delta_time, 1.0e-5f);
  std::vector<float3> previous_centers_fallback;
  std::vector<float3> collider_centers(colliders.size(), float3(0.0f));
  std::vector<float3> collider_offsets(colliders.size(), float3(0.0f));
  std::vector<float3> collider_velocities(colliders.size(), float3(0.0f));
  std::vector<float3> collider_bounds_min(colliders.size(), float3(0.0f));
  std::vector<float3> collider_bounds_max(colliders.size(), float3(0.0f));
  bool colliders_moving = false;
  for (const int i : colliders.index_range()) {
    const StaticCollider &collider = colliders[i];
    if (collider.vertices_count <= 0) {
      continue;
    }
    float3 sum(0.0f);
    float3 min_point(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const int j : IndexRange(collider.vertices_count)) {
      const float3 &vertex = collider_vertices[collider.vertices_start + j];
      sum += vertex;
      min_point.x = std::min(min_point.x, vertex.x);
      min_point.y = std::min(min_point.y, vertex.y);
      min_point.z = std::min(min_point.z, vertex.z);
      max_point.x = std::max(max_point.x, vertex.x);
      max_point.y = std::max(max_point.y, vertex.y);
      max_point.z = std::max(max_point.z, vertex.z);
    }
    collider_centers[i] = sum / float(collider.vertices_count);
    collider_bounds_min[i] = min_point;
    collider_bounds_max[i] = max_point;
  }
  /* Identity of each collider's geometry: when it changes (animated or deforming collider) only
   * that collider's body is rebuilt, the world itself is kept. */
  std::vector<uint64_t> collider_hashes(colliders.size(), 0);
  for (const int i : colliders.index_range()) {
    const StaticCollider &collider = colliders[i];
    uint64_t hash = 0;
    hash = hash_mix(hash, uint64_t(collider.use_box));
    hash = hash_mix(hash, uint64_t(collider.convex));
    hash = hash_mix(hash, uint64_t(collider.vertices_count));
    hash = hash_mix(hash, uint64_t(collider.indices_count));
    hash = hash_mix(hash, uint64_t(std::llround(double(collider.friction) * 1.0e4)));
    for (const int j : IndexRange(collider.vertices_count)) {
      hash = hash_point(hash, collider_vertices[collider.vertices_start + j]);
    }
    collider_hashes[i] = hash;
  }

  std::vector<float3> &previous_centers = handle != nullptr ? handle->collider_centers
                                                           : previous_centers_fallback;
  if (settings.sweep_colliders && previous_centers.size() == collider_centers.size()) {
    for (const int i : IndexRange(int(collider_centers.size()))) {
      const float3 delta = collider_centers[i] - previous_centers[i];
      if (delta.x * delta.x + delta.y * delta.y + delta.z * delta.z < 1.0e-8f) {
        continue;
      }
      collider_offsets[i] = -delta;
      collider_velocities[i] = delta / collider_dt;
      colliders_moving = true;
    }
  }
  if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
    printf("RBD_DEBUG sweep=%d prev=%d moving=%d c0=%.2f v0=%.2f\n",
           int(settings.sweep_colliders),
           int(previous_centers.size()),
           int(colliders_moving),
           collider_centers.empty() ? 0.0 : double(collider_centers[0].z),
           collider_velocities.empty() ? 0.0 : double(collider_velocities[0].z));
    fflush(stdout);
  }
  previous_centers = collider_centers;
  /* A collider can change every frame (an animated prop, a deforming ground). Its body is replaced
   * in place instead of rebuilding the world, so the dynamic bodies keep their velocities and their
   * warm started contacts. */
  std::vector<bool> collider_changed(colliders.size(), false);
  if (reuse) {
    int refreshed = 0;
    for (const int i : colliders.index_range()) {
      const bool changed = i >= int(ctx->collider_hashes.size()) ||
                           ctx->collider_hashes[i] != collider_hashes[i];
      if (!changed) {
        continue;
      }
      collider_changed[i] = true;
      refreshed++;
      if (i < int(ctx->collider_body_ids.size())) {
        const BodyID id = ctx->collider_body_ids[i];
        if (!id.IsInvalid()) {
          body_interface.RemoveBody(id);
          body_interface.DestroyBody(id);
        }
        ctx->collider_body_ids[i] = add_static_collider(
            body_interface,
            colliders[i],
            collider_vertices.data(),
            collider_indices.data(),
            collider_velocities[i].x != 0.0f || collider_velocities[i].y != 0.0f ||
                collider_velocities[i].z != 0.0f,
            collider_offsets[i],
            collider_velocities[i]);
      }
    }
    ctx->collider_hashes = collider_hashes;
    if (refreshed > 0 && std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
      printf("RBD_DEBUG collider refresh (%d/%d changed)\n", refreshed, int(colliders.size()));
      fflush(stdout);
    }
  }
  /* Glued bodies must not collide with each other (see SimulateSettings). Every dynamic body gets
   * * its own sub group, the pairs that are glued are disabled in the table below. The table costs
   * * `bodies^2 / 16` bytes, so it is only built for scenes where that stays reasonable. */
  const bool want_collision_filter = (settings.disable_glued_collisions && !constraints.is_empty()) ||
                                     settings.ignore_initial_overlap;
  /* The filter table costs `bodies^2 / 16` bytes, so it is still capped, but high enough for a
   * scene model that was split into thousands of islands. */
  const bool use_glue_filter = want_collision_filter && bodies.size() <= 20000;
  if (!reuse) {
    if (use_glue_filter) {
      glue_filter = new GroupFilterTable(uint(bodies.size()));
    }
    physics_system.Init(std::max(num_bodies, 1u),
                        0,
                        max_body_pairs,
                        max_contact_constraints,
                        broad_phase_layer_interface,
                        object_vs_broadphase_layer_filter,
                        object_layer_pair_filter);
    physics_system.SetGravity(to_jolt(settings.gravity));
    {
      PhysicsSettings physics_settings = physics_system.GetPhysicsSettings();
      physics_settings.mNumVelocitySteps = uint(std::clamp(settings.constraint_iterations, 1, 64));
      /* Penetration is resolved by the position steps, so a couple of them are not enough for a
       * * tightly packed pile: fragments would get squeezed through the ground collider. */
      physics_settings.mNumPositionSteps = uint(
          std::clamp(settings.constraint_iterations / 2, 2, 16));
      /* Static colliders are triangle soups with one-sided triangles: as soon as a fragment ends
       * up       * *behind* the surface it gets no contacts at all and falls straight through, so
       * a deep       * penetration can never be recovered from. Everything here is about not
       * getting there:       *       * - a speculative distance above the movement per substep
       * creates the contact *before* the       *   shapes touch (the substep length in the step
       * loop below is capped so this holds),       * - continuous collision detection stays on
       * (LinearCast per body) for the fast cases. */
      physics_settings.mSpeculativeContactDistance = 0.05f;
      /* A big glued pile is a single constraint island of thousands of bodies. Jolt splits those
       * * across cores by default, which also splits the load path from the top of the pile to the
       * * ground: the bottom fragments then stop being held and slowly sink through the collider.
       * * Solving the island as a whole is slower but keeps the pile on the ground. */
      physics_settings.mUseLargeIslandSplitter = false;
      /* 1 mm of slop: enough to stop the contact solver from fighting itself on a resting pile,
       * far       * too little for a fragment to sink behind the triangles (Jolt defaults to 2
       * cm). */
      physics_settings.mPenetrationSlop = 0.001f;
      /* Sleeping: resting bodies stop being integrated (and cost almost nothing) until something
       * * wakes them. With warm start this also means the pile stays settled across frames. */
      physics_settings.mAllowSleeping = settings.allow_sleeping;
      /* How still a body has to be, and for how long, before Jolt lets it sleep. Exposed because
       * the defaults (3 cm/s for half a second) can make a slowly creeping pile look frozen. */
      physics_settings.mPointVelocitySleepThreshold = std::max(settings.sleep_threshold, 0.0f);
      physics_settings.mTimeBeforeSleep = std::max(settings.time_before_sleep, 0.0f);
      physics_system.SetPhysicsSettings(physics_settings);
    }
    ctx->collider_body_ids.clear();
    ctx->collider_hashes = collider_hashes;
    for (const int i : colliders.index_range()) {
      ctx->collider_body_ids.push_back(
          add_static_collider(body_interface,
                              colliders[i],
                              collider_vertices.data(),
                              collider_indices.data(),
                              collider_velocities[i].x != 0.0f || collider_velocities[i].y != 0.0f ||
                                  collider_velocities[i].z != 0.0f,
                              collider_offsets[i],
                              collider_velocities[i]));
    }
    body_ids.reserve(bodies.size());
    for (const int i : bodies.index_range()) {
      const BodyState &state = bodies[i];
      if (!state_is_finite(state)) {
        /* A NaN/inf state would poison the whole solver (Jolt does not survive it), so this body
         * is       * left out of the world instead of taking the process down with it. */
        if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
          printf("RBD_DEBUG body %d has a non-finite state, skipped\n", i);
          fflush(stdout);
        }
        ctx->shapes.push_back({nullptr});
        body_ids.push_back(BodyID());
        continue;
      }
      const BodyShape shape = shapes.is_empty() ? BodyShape{} : shapes[i];
      const RefConst<Shape> jolt_shape = make_body_shape(shape, state, shape_points.data());
      ctx->shapes.push_back({jolt_shape});
      /* An animated body has to follow its animation and an inactive one must not move at all:
       * both are kinematic, so the other bodies collide with them but can not push them. */
      const bool kinematic = state.animated || !state.active;
      BodyCreationSettings body_settings(jolt_shape,
                                         to_jolt_r(state.position),
                                         to_jolt(state.rotation),
                                         kinematic ? EMotionType::Kinematic : EMotionType::Dynamic,
                                         k_layer_moving);
      body_settings.mLinearVelocity = to_jolt(state.linear_velocity);
      body_settings.mAngularVelocity = to_jolt(state.angular_velocity);
      /* The mass and the inertia are provided explicitly instead of letting Jolt derive them from
       * the shape: a thin or degenerate piece (a sliver, a shell piece of an open mesh) has a
       * volume of almost zero, so `CalculateInertia` produced an inertia that overflowed to
       * infinity. The solver then returned NaN positions, which looked like a pile that does not
       * fall while a few pieces explode. A box inertia from the shape's extents is always finite. */
      const float body_mass = std::max(state.mass, 1.0e-2f);
      float3 extents(0.0f);
      if (shape.type == ShapeType::Box) {
        extents = shape.half_extents * 2.0f;
      }
      else if (shape.type == ShapeType::ConvexHull && shape.points_count > 0) {
        float3 min_point(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        for (const int j : IndexRange(shape.points_count)) {
          const float3 &p = shape_points[shape.points_start + j];
          min_point.x = std::min(min_point.x, p.x);
          min_point.y = std::min(min_point.y, p.y);
          min_point.z = std::min(min_point.z, p.z);
          max_point.x = std::max(max_point.x, p.x);
          max_point.y = std::max(max_point.y, p.y);
          max_point.z = std::max(max_point.z, p.z);
        }
        extents = max_point - min_point;
      }
      else {
        extents = float3(std::max(state.radius, 0.01f) * 2.0f);
      }
      /* 5 mm minimum per axis, so a flat piece still has a usable inertia. */
      extents.x = std::max(extents.x, 5.0e-3f);
      extents.y = std::max(extents.y, 5.0e-3f);
      extents.z = std::max(extents.z, 5.0e-3f);
      body_settings.mOverrideMassProperties = EOverrideMassProperties::MassAndInertiaProvided;
      body_settings.mMassPropertiesOverride.mMass = body_mass;
      body_settings.mMassPropertiesOverride.mInertia = Mat44(
          Vec4(body_mass / 12.0f * (extents.y * extents.y + extents.z * extents.z), 0, 0, 0),
          Vec4(0, body_mass / 12.0f * (extents.x * extents.x + extents.z * extents.z), 0, 0),
          Vec4(0, 0, body_mass / 12.0f * (extents.x * extents.x + extents.y * extents.y), 0),
          Vec4(0, 0, 0, 1));
      body_settings.mFriction = 0.6f;
      body_settings.mRestitution = 0.0f;
      body_settings.mLinearDamping = std::clamp(state.linear_damping, 0.0f, 1.0f);
      body_settings.mAngularDamping = std::clamp(state.angular_damping, 0.0f, 1.0f);
      /* Continuous collision detection: fragments that get squeezed out of a pile can move very
       * fast     * and would otherwise pass straight through the ground collider. */
      body_settings.mMotionQuality = EMotionQuality::LinearCast;
      /* Optional speed ceiling. Off by default, so a fast projectile keeps its speed (Jolt's own
       * 500 m/s ceiling stays as a safety net). When on, 50 m/s / 30 rad/s is well above anything a
       * falling pile does and still catches the artefacts of the penetration recovery of a dense
       * stack (small fragments have a tiny mass, so the same impulse gives them a huge velocity). */
      if (settings.speed_limit) {
        const float max_speed = std::max(settings.max_speed, 0.0f);
        body_settings.mMaxLinearVelocity = max_speed;
        body_settings.mMaxAngularVelocity = max_speed * 0.6f;
      }
      else {
        /* Jolt's own defaults: no clamp worth mentioning for a scene (500 m/s, 47 rad/s), but still
         * a safety net so a numerical artefact can not turn into an absurd velocity. */
        body_settings.mMaxLinearVelocity = 500.0f;
        body_settings.mMaxAngularVelocity = 0.25f * JPH_PI * 60.0f;
      }
      /* A kinematic body is driven from the outside, it must never fall asleep. */
      body_settings.mAllowSleeping = settings.allow_sleeping && !kinematic;
      if (glue_filter) {
        body_settings.mCollisionGroup = CollisionGroup(glue_filter, 0, uint32(i));
      }
      const BodyID body_id = body_interface.CreateAndAddBody(body_settings, EActivation::Activate);
      body_ids.push_back(body_id);
      if (body_id.IsInvalid()) {
        continue;
      }
    }
    /* Bodies that already interpenetrate are taken out of collision for the lifetime of this
     * world: otherwise the contact solver pushes them apart with a huge impulse on the first frame
     * (a pile or a scene model that "explodes" without anything touching it). The narrow phase
     * finds the pairs precisely, using the actual penetration depth, which matters for geometry
     * that was authored with overlapping pieces. */
    if (glue_filter && settings.ignore_initial_overlap) {
      std::unordered_map<uint32, int> index_of_body;
      for (const int i : bodies.index_range()) {
        if (!body_ids[i].IsInvalid()) {
          index_of_body.emplace(body_ids[i].GetIndexAndSequenceNumber(), i);
        }
      }
      CollideShapeSettings collide_settings;
      /* Only report actual overlap, not the shapes being merely close. */
      collide_settings.mMaxSeparationDistance = 0.0f;
      constexpr float min_penetration = 0.002f;
      int ignored_pairs = 0;
      for (const int i : bodies.index_range()) {
        if (body_ids[i].IsInvalid() || i >= int(ctx->shapes.size()) || !ctx->shapes[i].shape) {
          continue;
        }
        AllHitCollisionCollector<CollideShapeCollector> collector;
        physics_system.GetNarrowPhaseQuery().CollideShape(
            ctx->shapes[i].shape,
            Vec3(1.0f, 1.0f, 1.0f),
            body_interface.GetCenterOfMassTransform(body_ids[i]),
            collide_settings,
            RVec3::sZero(),
            collector);
        for (const CollideShapeResult &hit : collector.mHits) {
          if (hit.mPenetrationDepth < min_penetration) {
            continue;
          }
          const auto other_it = index_of_body.find(hit.mBodyID2.GetIndexAndSequenceNumber());
          if (other_it == index_of_body.end()) {
            continue;
          }
          const int other = other_it->second;
          if (other == i) {
            continue;
          }
          glue_filter->DisableCollision(uint32(std::min(i, other)), uint32(std::max(i, other)));
          ignored_pairs++;
        }
      }
      if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
        printf("RBD_DEBUG initial_overlap ignored_pairs=%d bodies=%d\n",
               ignored_pairs,
               int(bodies.size()));
        fflush(stdout);
      }
    }
  }
  /* if (!reuse): bodies */ const auto make_constraint = [](const ConstraintDef &def,
                                                            Body *body_a,
                                                            Body *body_b,
                                                            const Vec3 anchor_a,
                                                            const Vec3 anchor_b,
                                                            const bool has_anchors,
                                                            Vec3 &com_a,
                                                            Vec3 &com_b) -> Ref<Constraint> {
    /* Build the Jolt constraint described by `def`. Used both when the world is built and when a
     * * constraint is replaced at runtime (soft fallback, hardening). */
    if (def.type == 1) {
      /* Soft constraint: the anchors are held together by a spring instead of being welded, and
       * * the rotation stays rigid (Jolt only supports springs on the translation axes). */
      SixDOFConstraintSettings settings;
      settings.mSpace = EConstraintSpace::LocalToBodyCOM;
      settings.mPosition1 = anchor_a;
      settings.mPosition2 = anchor_b;
      if (has_anchors) {
        com_a = body_a->GetShape()->GetCenterOfMass();
        com_b = body_b->GetShape()->GetCenterOfMass();
        settings.mPosition1 -= com_a;
        settings.mPosition2 -= com_b;
      }
      const SpringSettings spring(ESpringMode::FrequencyAndDamping,
                                  std::max(def.frequency, 0.0f),
                                  std::clamp(def.damping, 0.0f, 1.0f));
      for (const SixDOFConstraintSettings::EAxis axis :
           {SixDOFConstraintSettings::EAxis::TranslationX,
            SixDOFConstraintSettings::EAxis::TranslationY,
            SixDOFConstraintSettings::EAxis::TranslationZ})
      {
        settings.SetLimitedAxis(axis, 0.0f, 0.0f);
        settings.mLimitsSpringSettings[axis] = spring;
      }
      settings.MakeFixedAxis(SixDOFConstraintSettings::EAxis::RotationX);
      settings.MakeFixedAxis(SixDOFConstraintSettings::EAxis::RotationY);
      settings.MakeFixedAxis(SixDOFConstraintSettings::EAxis::RotationZ);
      return settings.Create(*body_a, *body_b);
    }
    FixedConstraintSettings settings;
    if (has_anchors) {
      /* The anchors are stored relative to the fragment origin, Jolt wants them relative to the
       * * body center of mass. Fragments are irregular convex hulls, so the two differ and using
       * * the origin would make the weld target a slightly wrong relative pose (which shows up as
       * * a soft, slowly drifting weld). */
      com_a = body_a->GetShape()->GetCenterOfMass();
      com_b = body_b->GetShape()->GetCenterOfMass();
      settings.mSpace = EConstraintSpace::LocalToBodyCOM;
      settings.mAutoDetectPoint = false;
      settings.mPoint1 = anchor_a - com_a;
      settings.mPoint2 = anchor_b - com_b;
    }
    else {
      settings.mSpace = EConstraintSpace::WorldSpace;
      settings.mAutoDetectPoint = true;
    }
    return settings.Create(*body_a, *body_b);
  };
  /* Jolt clears the force and torque accumulators of a body after every step, so external forces
   * * have to be applied again for each substep (below). Walking the bodies is only worth it when
   * at   * least one of them actually has a force. */
  bool any_external_force = false;
  for (const BodyState &state : bodies) {
    if (state.external_force != float3(0.0f) || state.external_torque != float3(0.0f)) {
      any_external_force = true;
      break;
    }
  }
  const auto apply_external_forces = [&]() {
    if (!any_external_force) {
      return;
    }
    for (const int i : bodies.index_range()) {
      const BodyID body_id = body_ids[i];
      if (body_id.IsInvalid()) {
        continue;
      }
      if (bodies[i].external_force != float3(0.0f) && finite3(bodies[i].external_force)) {
        body_interface.AddForce(body_id, to_jolt(bodies[i].external_force));
      }
      if (bodies[i].external_torque != float3(0.0f) && finite3(bodies[i].external_torque)) {
        body_interface.AddTorque(body_id, to_jolt(bodies[i].external_torque));
      }
    }
  };
  /* Constraints are owned by the caller of AddConstraint and have to stay alive until the physics
   * * system is destroyed, so they are kept in the context as well. */
  if (!reuse) {
    /* Build the Jolt constraint described by `def`;
   also used when a constraint is replaced at   * runtime (soft fallback, hardening). */
    for (const int def_index : constraints.index_range()) {
      const ConstraintDef &def = constraints[def_index];
      if (!def.enabled || def.body_a < 0 || def.body_b < 0) {
        continue;
      }
      if (def.body_a >= int(body_ids.size()) || def.body_b >= int(body_ids.size())) {
        continue;
      }
      const BodyID id_a = body_ids[def.body_a];
      const BodyID id_b = body_ids[def.body_b];
      if (id_a.IsInvalid() || id_b.IsInvalid()) {
        continue;
      }
      Body *body_a = physics_system.GetBodyLockInterfaceNoLock().TryGetBody(id_a);
      Body *body_b = physics_system.GetBodyLockInterfaceNoLock().TryGetBody(id_b);
      if (!body_a || !body_b) {
        continue;
      }
      /* With explicit anchors the constraint keeps the *original* relative pose, which matters
       * * because the physics world is rebuilt every frame: auto detected anchors would otherwise
       * * follow any drift that already happened. */
      ActiveConstraint active;
      active.def_index = def_index;
      active.body_index_a = def.body_a;
      active.body_index_b = def.body_b;
      active.body_a = id_a;
      active.body_b = id_b;
      Ref<Constraint> constraint = make_constraint(def,
                                                   body_a,
                                                   body_b,
                                                   to_jolt(def.anchor_a),
                                                   to_jolt(def.anchor_b),
                                                   def.has_anchors,
                                                   active.com_a,
                                                   active.com_b);
      if (!constraint) {
        continue;
      }
      physics_system.AddConstraint(constraint.GetPtr());
      if (glue_filter) {
        /* Let the two bodies pass through each other from now on: the glue holds them together and
         * * their shared cut faces would otherwise generate contacts that fight the constraint. */
        glue_filter->DisableCollision(uint32(def.body_a), uint32(def.body_b));
      }
      active.constraint = constraint;
      active.anchor_a = to_jolt(def.anchor_a);
      active.anchor_b = to_jolt(def.anchor_b);
      active.has_anchors = def.has_anchors;
    active.harden_accumulator = def.harden_accum;
      constraint_refs.push_back(constraint);
      active_constraints.push_back(active);
    }
  }
  /* if (!reuse): constraints */ else
  {
    /* Nothing structural changed: the world (and with it Jolt's contact and constraint caches,
     * * which is the warm start) is kept. Only the enabled state of the constraints can differ
     * * between frames, because the caller writes a broken glue constraint back into the     *
     * attributes. */
    for (const int def_index : constraints.index_range()) {
      if (def_index >= int(active_constraints.size())) {
        break;
      }
      active_constraints[def_index].constraint->SetEnabled(constraints[def_index].enabled);
      active_constraints[def_index].impulse_accumulator = 0.0f;
    }
  }
  if (handle != nullptr && settings.warm_start && !reuse) {
    handle->context = std::move(owned_context);
  }
  /* Evaluate the break conditions of glue constraints and disable the ones that exceeded a   *
   * threshold. The caller writes that state back into the geometry attributes, so a broken   *
   * constraint stays broken across frames (the physics world is rebuilt or kept alive).   *   *
   * This walks every constraint twice per substep (a body lookup each), which is pure overhead
   * when   * no constraint has a break threshold at all, so that case returns immediately. */
  bool any_constraint_update = false;
  for (const ConstraintDef &def : constraints) {
    if (def.break_impulse > 0.0f || def.break_distance > 0.0f || def.break_angle > 0.0f ||
        def.harden_time > 0.0f)
    {
      any_constraint_update = true;
      break;
    }
  }
  const int substeps = std::max(
      {settings.substeps, int(std::ceil(settings.delta_time * 240.0f)), 1});
  const float substep_delta_time = settings.delta_time / float(substeps);
  const auto evaluate_breaks = [&]() {
    if (!any_constraint_update) {
      return;
    }
    const BodyLockInterfaceNoLock &lock_interface = physics_system.GetBodyLockInterfaceNoLock();
    for (ActiveConstraint &active : active_constraints) {
      /* A constraint that already broke is disabled or replaced by a soft one, there is nothing
       * * left to measure on it. */
      if (active.result != ConstraintResult::Active || !active.constraint->GetEnabled()) {
        continue;
      }
      const ConstraintDef &def = constraints[active.def_index];
      Body *body_a = lock_interface.TryGetBody(active.body_a);
      Body *body_b = lock_interface.TryGetBody(active.body_b);
      if (!body_a || !body_b) {
        continue;
      }
      /* World positions of the anchors (stored relative to the body origin, Jolt wants them
       * * relative to the center of mass). */
      const RVec3 p_a = body_a->GetCenterOfMassPosition() +
                        body_a->GetRotation() * (active.anchor_a - active.com_a);
      const RVec3 p_b = body_b->GetCenterOfMassPosition() +
                        body_b->GetRotation() * (active.anchor_b - active.com_b);
      const float separation = float((p_a - p_b).Length());
      bool broken = false;
      if (def.break_impulse > 0.0f) {
        /* The threshold is compared against the impulse of the whole frame so far, not against a
         * * single substep: that keeps it independent of the substep count and makes it line up
         * with         * the momentum the constraint has to absorb (`mass * velocity change`). */
        Vec3 lambda;
        if (def.type == 1) {
          lambda = static_cast<const SixDOFConstraint *>(active.constraint.GetPtr())
                       ->GetTotalLambdaPosition();
        }
        else {
          lambda = static_cast<const FixedConstraint *>(active.constraint.GetPtr())
                       ->GetTotalLambdaPosition();
        }
        active.impulse_accumulator += lambda.Length();
        if (active.impulse_accumulator > def.break_impulse) {
          broken = true;
        }
      }
      if (!broken && def.break_distance > 0.0f && active.has_anchors) {
        if (separation > def.break_distance) {
          broken = true;
        }
      }
      if (!broken && def.break_angle > 0.0f) {
        const Quat relative = body_a->GetRotation().Inversed() * body_b->GetRotation();
        const float angle = 2.0f * std::acos(std::min(1.0f, std::abs(relative.GetW())));
        if (angle > def.break_angle) {
          broken = true;
        }
      }
      if (broken) {
        if (def.fallback == 1 && def.type == 0) {
          /* The weld tears but keeps pulling: replace it with a soft constraint right away, so the
           * * pieces stay loosely connected instead of flying apart. Its break thresholds are
           * cleared           * because the soft fallback is meant to be permanent. */
          physics_system.RemoveConstraint(active.constraint.GetPtr());
          ConstraintDef soft_def = def;
          soft_def.type = 1;
          soft_def.break_impulse = 0.0f;
          soft_def.break_distance = 0.0f;
          soft_def.break_angle = 0.0f;
          Ref<Constraint> soft = make_constraint(soft_def,
                                                 body_a,
                                                 body_b,
                                                 active.anchor_a,
                                                 active.anchor_b,
                                                 active.has_anchors,
                                                 active.com_a,
                                                 active.com_b);
          if (soft) {
            physics_system.AddConstraint(soft.GetPtr());
            constraint_refs.push_back(soft);
            active.constraint = soft;
            active.result = ConstraintResult::SoftFallback;
            active.impulse_accumulator = 0.0f;
          }
          else {
            active.constraint->SetEnabled(false);
            active.result = ConstraintResult::Disabled;
          }
        }
        else {
          active.constraint->SetEnabled(false);
          if (glue_filter) {
            /* The filter table is persistent now, so a broken pair has to be allowed to collide
             * * again by hand: in the stateless version the table was rebuilt without that pair.
             */
            glue_filter->EnableCollision(uint32(constraints[active.def_index].body_a),
                                         uint32(constraints[active.def_index].body_b));
          }
          active.result = ConstraintResult::Disabled;
        }
        /* A released constraint has to wake its bodies: a sleeping pile would not react at all. */
        if (settings.wake_on_break) {
          body_interface.ActivateBody(active.body_a);
          body_interface.ActivateBody(active.body_b);
        }
        continue;
      }
      /* Hardening: a constraint that stays deformed (but does not break) keeps its deformation and
       * * makes it the new rest pose after `harden_time` seconds. */
      if (def.harden_time > 0.0f && active.has_anchors) {
        if (separation > 1.0e-4f) {
          active.harden_accumulator += substep_delta_time;
          if (active.harden_accumulator >= def.harden_time) {
            /* New rest pose: the anchors stay where they are in each body's local frame, so the
             * * constraint now pulls to the current relative pose instead of the original one. */
            const Vec3 local_a = body_a->GetRotation().Inversed() * (p_a - body_a->GetPosition());
            const Vec3 local_b = body_b->GetRotation().Inversed() * (p_b - body_b->GetPosition());
            active.anchor_a = local_a;
            active.anchor_b = local_b;
            active.rest_anchor_a = to_blender(local_a);
            active.rest_anchor_b = to_blender(local_b);
            active.rest_pose_changed = true;
            active.harden_accumulator = 0.0f;
            /* Rebuild the constraint so it pulls towards the new rest pose. */ physics_system
                .RemoveConstraint(active.constraint.GetPtr());
            Ref<Constraint> rebuilt = make_constraint(def,
                                                      body_a,
                                                      body_b,
                                                      active.anchor_a,
                                                      active.anchor_b,
                                                      active.has_anchors,
                                                      active.com_a,
                                                      active.com_b);
            if (rebuilt) {
              physics_system.AddConstraint(rebuilt.GetPtr());
              constraint_refs.push_back(rebuilt);
              active.constraint = rebuilt;
              active.impulse_accumulator = 0.0f;
            }
          }
        }
        else {
          /* Back at its rest pose: the deformation is gone, so the timer starts over. */ active
              .harden_accumulator = 0.0f;
        }
      }
    }
  };
  /* Step one substep at a time and check the break conditions in between: a glue constraint that
   * * breaks has to release its bodies *before* the rest of the frame is simulated. Breaking only
   * at   * the end of the frame would be too late, the weld has cancelled the relative velocity by
   * then   * and the pieces would simply stay put instead of flying apart.   *   * The substep
   * length is capped at 1/240 s: the contact detection has to see the approach *before*   * the
   * shapes touch, otherwise an impact is missed between two steps and a fragment ends up behind
   * * the one-sided triangles of a mesh collider, where no contact can ever be generated again (it
   * * then falls straight through the level). 1/240 s keeps the movement per step well below the
   * * speculative contact distance for the speeds a falling pile reaches. */
  /* Animated bodies: the node tree hands in a new pose every frame. `MoveKinematic` moves the
   * kinematic body there across the substeps (it derives the velocity from the delta), so it pushes
   * the other bodies and ends exactly at the animated pose. That velocity is written back with the
   * other state, which is what a body keeps when the animation is switched off. */
  for (const int i : bodies.index_range()) {
    if (!bodies[i].animated) {
      continue;
    }
    const BodyID body_id = body_ids[i];
    if (body_id.IsInvalid()) {
      continue;
    }
    const float animated_dt = std::max(settings.delta_time, 1.0e-5f);
    /* The velocity of the animation is what a body keeps when the animation is switched off, so it
     * is computed here (Jolt resets a kinematic body's velocity once it reached its target). */
    if (ctx->written_states.size() == bodies.size()) {
      bodies[i].linear_velocity = (bodies[i].position - ctx->written_states[i].position) /
                                  animated_dt;
    }
    body_interface.MoveKinematic(body_id,
                                 to_jolt_r(bodies[i].position),
                                 to_jolt(bodies[i].rotation),
                                 animated_dt);
    /* Wake what the animated body can reach: a sleeping pile ignores a body that only moves by its
     * velocity. */
    float radius = 0.0f;
    if (i < int(shapes.size()) && shapes[i].points_count > 0) {
      for (const int j : IndexRange(shapes[i].points_count)) {
        const float3 &point = shape_points[shapes[i].points_start + j];
        radius = std::max(radius, std::sqrt(math::dot(point, point)));
      }
    }
    else {
      radius = std::max(bodies[i].radius, 0.1f);
    }
    const float wake_radius = radius + 1.0f;
    for (const int other : bodies.index_range()) {
      if (other == i) {
        continue;
      }
      const float3 delta = bodies[other].position - bodies[i].position;
      if (math::dot(delta, delta) > wake_radius * wake_radius) {
        continue;
      }
      const BodyID other_id = body_ids[other];
      if (!other_id.IsInvalid()) {
        body_interface.ActivateBody(other_id);
      }
    }
  }
  /* A collider that moves or deforms has to wake what it touches. Jolt only wakes bodies on a
   * contact *impulse*, and a static collider that is replaced (or a kinematic one that moves) does
   * not produce one for a body that is already asleep, so a fast sphere would fly straight through
   * a settled pile. Only the bodies inside the collider's bounds (plus a margin) are woken, so a
   * constantly moving collider does not keep the whole scene awake. */
  for (const int i : colliders.index_range()) {
    const StaticCollider &collider = colliders[i];
    const bool moving = collider_velocities[i].x != 0.0f ||
                        collider_velocities[i].y != 0.0f ||
                        collider_velocities[i].z != 0.0f;
    if (!collider_changed[i] && !moving) {
      continue;
    }
    const float3 margin(0.75f);
    const float3 min_point = collider_bounds_min[i] - margin;
    const float3 max_point = collider_bounds_max[i] + margin;
    for (const int body_i : bodies.index_range()) {
      const float3 &position = bodies[body_i].position;
      if (position.x < min_point.x || position.y < min_point.y || position.z < min_point.z ||
          position.x > max_point.x || position.y > max_point.y || position.z > max_point.z)
      {
        continue;
      }
      const BodyID body_id = body_ids[body_i];
      if (!body_id.IsInvalid()) {
        body_interface.ActivateBody(body_id);
      }
    }
  }
  /* Keep everything awake while the user is still placing the scene. */ if (settings.wake_all)
  {
    for (const BodyID body_id : body_ids) {
      if (!body_id.IsInvalid()) {
        body_interface.ActivateBody(body_id);
      }
    }
  }
  /* Bodies whose `active` flag changed are switched between dynamic and kinematic in place: a
   * rebuild would throw away the warm started contacts of the whole world (and the caller's state
   * would look "different" for every other body as well). The pose of a body that is switched off is
   * kept and its velocity is cleared, so it does not drag anything along when it is switched on
   * again. */
  {
    static const bool debug = std::getenv("BLENDER_RBD_DEBUG") != nullptr;
    const BodyLockInterfaceNoLock &lock_interface = physics_system.GetBodyLockInterfaceNoLock();
    for (const int i : bodies.index_range()) {
      const BodyID body_id = body_ids[i];
      if (body_id.IsInvalid() || bodies[i].animated) {
        /* Animated bodies are handled by `MoveKinematic` above. */
        continue;
      }
      const Body *body = lock_interface.TryGetBody(body_id);
      if (body == nullptr) {
        continue;
      }
      const EMotionType wanted = bodies[i].active ? EMotionType::Dynamic
                                                 : EMotionType::Kinematic;
      if (body->GetMotionType() == wanted) {
        continue;
      }
      body_interface.SetMotionType(body_id, wanted, EActivation::Activate);
      if (wanted == EMotionType::Kinematic) {
        body_interface.SetLinearVelocity(body_id, Vec3::sZero());
        body_interface.SetAngularVelocity(body_id, Vec3::sZero());
      }
      if (debug) {
        printf("RBD_DEBUG body %d switched to %s\n",
               i,
               wanted == EMotionType::Dynamic ? "dynamic" : "kinematic (inactive)");
        fflush(stdout);
      }
    }
  }

  for (const int step : IndexRange(substeps)) {
    apply_external_forces();
    physics_system.Update(substep_delta_time, 1, &temp_allocator, &job_system);
    evaluate_breaks();
  }
  /* The bodies are read back through the lock-free interface: the four accessors of the locking
   * * interface each take and release the body lock, which is measurable overhead for thousands of
   * * fragments and unnecessary here, since nothing else touches the world at this point. */
  const BodyLockInterfaceNoLock &body_lock_interface = physics_system.GetBodyLockInterfaceNoLock();
  for (const int i : bodies.index_range()) {
    const BodyID body_id = body_ids[i];
    if (body_id.IsInvalid()) {
      continue;
    }
    const Body *body = body_lock_interface.TryGetBody(body_id);
    if (body == nullptr) {
      continue;
    }
    BodyState &state = bodies[i];
    /* The incoming state is the fallback when the solver produced something that is not a number.
     */
    const BodyState previous = state;
    /* The state holds the body origin (that is what BodyCreationSettings expects), not the center
     * * of mass. Reading the center of mass here would shift every fragment by its own center of
     * * mass offset on every frame, which shows up as a pile that slowly creeps apart. */
    state.position = to_blender(Vec3(body->GetPosition()));
    const Quat rotation = body->GetRotation();
    state.rotation = math::Quaternion(
        rotation.GetW(), rotation.GetX(), rotation.GetY(), rotation.GetZ());
    if (!state.animated) {
      state.linear_velocity = to_blender(body->GetLinearVelocity());
    }
    /* Animated bodies keep the velocity that was derived from their animation. */
    state.angular_velocity = to_blender(body->GetAngularVelocity());
    if (!state.active) {
      /* An inactive body was not integrated: it keeps its pose, and any velocity it had when it was
       * switched off is dropped (Jolt reports zero for a kinematic body anyway). */
      state.linear_velocity = float3(0.0f);
      state.angular_velocity = float3(0.0f);
    }
    state.sleeping = !body_interface.IsActive(body_id);
    if (!state_is_finite(state)) {
      /* A body that went bad in the solver must not put NaN into the geometry: it would be fed
       * back       * in on the next frame and take down the whole world (and the warm start check
       * treated a NaN       * state as "matching", so the bad world was even reused). */
      if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
        printf("RBD_DEBUG body %d produced a non-finite state, reset\n", i);
        fflush(stdout);
      }
      state = previous;
      state.linear_velocity = float3(0.0f);
      state.angular_velocity = float3(0.0f);
    }
  }
  /* Report what happened to the constraints, so the caller can write it back into the geometry. */
  for (const ActiveConstraint &active : active_constraints) {
    ConstraintFeedback &feedback = r_constraint_feedback[active.def_index];
    /* World space anchors for the preview lines: the pose of this frame plus the local anchor. */
    if (active.body_index_a >= 0 && active.body_index_a < int(bodies.size()) &&
        active.body_index_b >= 0 && active.body_index_b < int(bodies.size()))
    {
      const BodyState &state_a = bodies[active.body_index_a];
      const BodyState &state_b = bodies[active.body_index_b];
      feedback.anchor_a_world = state_a.position +
                                math::transform_point(state_a.rotation, to_blender(active.anchor_a));
      feedback.anchor_b_world = state_b.position +
                                math::transform_point(state_b.rotation, to_blender(active.anchor_b));
      feedback.body_a_world = state_a.position;
      feedback.body_b_world = state_b.position;
      feedback.has_world_anchors = true;
    }
    feedback.result = active.result;
    feedback.rest_pose_changed = active.rest_pose_changed;
    feedback.harden_accum = active.harden_accumulator;
    if (active.rest_pose_changed) {
      feedback.anchor_a = active.rest_anchor_a;
      feedback.anchor_b = active.rest_anchor_b;
    }
  }
  /* Remember what was written back, so the next frame can tell "the caller passed our own state
   * back in" (reuse the world) from "the state was edited / scrubbed / baked" (rebuild). */
  if (handle != nullptr) {
    ctx->written_states.assign(bodies.begin(), bodies.end());
  }
}
}  // namespace blender::geometry::rigid_body
   // namespace blender::geometry::rigid_body
