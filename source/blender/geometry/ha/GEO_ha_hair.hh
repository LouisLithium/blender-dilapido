/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * XPBD hair on the compute backend (stage 1).
 *
 * This is the device mirror of the per point part of the hair solver in
 * `node_geo_xpbd_solver.cc`:
 *
 * - the inertial predict (linear and angular),
 * - `RodStretchAndShearConstraintSet`,
 * - `RodBendAndTwistConstraintSet`,
 * - `PinPositionConstraintSet` and `PinRotationConstraintSet`,
 * - the velocity update,
 * - `LinearDampingConstraintSet` / `AngularDampingConstraintSet`.
 *
 * Everything that is not per point stays on the CPU and the solver keeps such hair on the CPU
 * (see `hair_gpu_rejection` in the solver): hair self collision, the guide tether, the contacts
 * between the hair and cloth/soft bodies, and the contacts with mesh colliders.
 *
 * ## Parity
 *
 * The kernels mirror the CPU functions line by line and keep their sequential order where it
 * matters. A rod constraint is *one invocation per curve* and walks the segments serially, exactly
 * like `RodStretchAndShearConstraintSet::solve_single` /
 * `RodBendAndTwistConstraintSet::solve_single`
 * (which are independent per curve and serial per curve on the CPU as well), so no coloring or
 * Jacobi approximation is involved for the rods. The pins use one invocation per pin, like
 * `color_constraints__unary`; the damping constraints are independent per point. The only
 * difference left is float rounding, plus the residual error the CPU accumulates in its updater
 * (the device does not feed the node's `residual_error` output yet).
 *
 * The Lagrange multipliers live on the device for the whole frame and are only read back at the
 * end of the last substep, where the solver copies them into the attributes it writes back.
 */

#pragma once

#include "BLI_math_quaternion_types.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

#include <memory>

#include "GEO_ha_context.hh"

namespace blender::geometry::ha {

/** Per point state of one hair chunk, uploaded once per frame. */
struct HaHairInputs {
  Span<float3> positions;
  Span<float3> velocities;
  Span<float3> external_force;
  /** Uniform acceleration (wind); empty or one per point. */
  Span<float3> external_acceleration;
  Span<float3> external_torque;
  Span<float> inv_masses;
  Span<float3> moments_of_inertia;
  Span<float3> inv_moments_of_inertia;
  Span<math::Quaternion> rotations;
  Span<float3> angular_velocities;
};

/**
 * One `Rod Stretch and Shear` constraint set. All point indices are relative to the chunk, the
 * curves are the curves of the chunk in ascending order.
 */
struct HaHairRodStretchSet {
  /** First point of every curve. */
  Span<int> curve_first;
  /** Number of points of every curve. */
  Span<int> curve_size;
  /** Rest length of the segment that ends at the point (`rest_length` attribute). */
  Span<float> rest_lengths;
  /** XPBD compliance of the segment that ends at the point. */
  Span<float> compliances;
  /** Multipliers of the last substep, written back by #HaXpbdHair::download. */
  MutableSpan<float3> lambdas_pos;
  MutableSpan<float3> lambdas_rot;
};

/** One `Rod Bend and Twist` constraint set. */
struct HaHairRodBendSet {
  Span<int> curve_first;
  Span<int> curve_size;
  /** Rest bend rotation of the first point of the segment (`rest_bend_rotation` attribute). */
  Span<math::Quaternion> rest_rotations;
  Span<float> compliances;
  MutableSpan<float4> lambdas;
};

/**
 * One velocity constraint set (`LinearDampingConstraintSet` or `AngularDampingConstraintSet`).
 * The sets are solved in the order the solver builds them, which is the order of the CPU's
 * `ChunkData::static_velocity_constraints`.
 */
struct HaHairVelocitySet {
  bool angular = false;
  /** Damping per point. */
  Span<float> dampings;
  MutableSpan<float> lambdas;
};

/** One `Pin Position` constraint set. `targets` is refreshed every substep (animated pins). */
struct HaHairPinPositionSet {
  Span<int> points;
  Span<float3> targets;
  Span<float> compliances;
  MutableSpan<float> lambdas;
};

/** One `Pin Rotation` constraint set. `targets` is refreshed every substep (animated pins). */
struct HaHairPinRotationSet {
  Span<int> points;
  Span<math::Quaternion> targets;
  Span<float> compliances;
  MutableSpan<float4> lambdas;
};

/** Coarse timing of the last frame, filled by #HaXpbdHair. */
struct HaXpbdHairStats {
  double frame_ms = 0.0;
  int passes = 0;
};

/**
 * Device side of one hair chunk. `bind` uploads the frame state, every substep runs
 *
 *   predict() -> iterate() * iterations -> update_velocities() -> solve_velocity()
 *
 * and `download` brings the result back. The multithreaded chunk path of the solver calls these
 * from one thread per chunk; the context guard serializes the dispatches.
 */
class HaXpbdHair {
 public:
  HaXpbdHair() = default;
  ~HaXpbdHair();
  HaXpbdHair(const HaXpbdHair &) = delete;
  HaXpbdHair &operator=(const HaXpbdHair &) = delete;

  /**
   * Uploads the frame state. The constraint sets are copied into device buffers; `lambdas` and
   * the pin `targets` stay host references, so the solver can animate the pins and read the
   * multipliers back at the end of the frame.
   */
  bool bind(const HaHairInputs &inputs,
            Span<const HaHairRodStretchSet> stretch_sets,
            Span<const HaHairRodBendSet> bend_sets,
            Span<const HaHairVelocitySet> velocity_sets,
            Span<const HaHairPinPositionSet> pin_position_sets,
            Span<const HaHairPinRotationSet> pin_rotation_sets,
            float delta_time);

  /**
   * Starts one substep: clears the multipliers, uploads the current pin targets (the solver
   * updated them for this substep) and runs the inertial predict.
   */
  bool predict();

  /** One constraint iteration (rod stretch/shear, rod bend/twist, every pin set, in CPU order). */
  bool iterate();

  /** Velocity update from the position difference of the substep. */
  bool update_velocities();

  /** Velocity constraint sets (linear and angular damping), in the order they were bound. */
  bool solve_velocity();

  /**
   * Copies positions, velocities, rotations and angular velocities back to the host and writes
   * the multipliers of the last substep into the spans passed to #bind.
   */
  bool download(MutableSpan<float3> r_positions,
                MutableSpan<float3> r_velocities,
                MutableSpan<math::Quaternion> r_rotations,
                MutableSpan<float3> r_angular_velocities);

  bool is_ready() const
  {
    return ready_;
  }

  int point_count() const
  {
    return point_count_;
  }

  const HaXpbdHairStats &stats() const
  {
    return stats_;
  }

  /** Why the last #bind or substep call failed, for the debug log. */
  const char *last_error() const
  {
    return last_error_;
  }

  void free();

 private:
  struct RodSetGpu {
    HaBufferInt curve_first;
    HaBufferInt curve_size;
    /** Two floats per point: rest length, compliance. */
    HaBuffer data;
    /** Six floats per point: position multiplier (3), rotation multiplier (3). */
    HaBuffer lambda;
    /** `[0]` is the curve count, so the padded invocations of the last work group can return. */
    HaBuffer set_params;
    int curve_count = 0;
    int point_count = 0;
  };

  struct BendSetGpu {
    HaBufferInt curve_first;
    HaBufferInt curve_size;
    /** Five floats per point: rest bend rotation (`w, x, y, z`), compliance. */
    HaBuffer data;
    /** Four floats per point. */
    HaBuffer lambda;
    /** `[0]` is the curve count, so the padded invocations of the last work group can return. */
    HaBuffer set_params;
    int curve_count = 0;
    int point_count = 0;
  };

  struct VelocitySetGpu {
    bool angular = false;
    /** Damping per point. */
    HaBuffer damping;
    /** One float per point. */
    HaBuffer lambda;
    int point_count = 0;
  };

  struct PinSetGpu {
    bool angular = false;
    HaBufferInt points;
    /** Three (`vec3`) or four (`vec4`) floats per pin. */
    HaBuffer target;
    HaBuffer compliance;
    /** One or four floats per pin. */
    HaBuffer lambda;
    /** `[0]` is the pin count, so the padded invocations of the last work group can return. */
    HaBuffer set_params;
    int count = 0;
  };

  /** Creates every shader the substep uses, so a compile failure is caught by #bind. */
  bool ensure_shaders();

  int point_count_ = 0;
  bool ready_ = false;
  bool has_rotation_ = false;
  bool has_external_acceleration_ = false;
  float delta_time_ = 0.0f;
  HaXpbdHairStats stats_;
  const char *last_error_ = "";

  /* Per point state, three / four floats per point. */
  HaBuffer positions_;
  HaBuffer prev_positions_;
  HaBuffer velocities_;
  HaBuffer rotations_;
  HaBuffer prev_rotations_;
  HaBuffer angular_velocities_;
  /** #HA_HAIR_PROP floats per point. */
  HaBuffer props_;
  /** #HA_HAIR_PARAM_COUNT floats. */
  HaBuffer params_;

  Vector<std::unique_ptr<RodSetGpu>> rod_sets_;
  Vector<std::unique_ptr<BendSetGpu>> bend_sets_;
  Vector<std::unique_ptr<VelocitySetGpu>> velocity_sets_;
  Vector<std::unique_ptr<PinSetGpu>> pin_position_sets_;
  Vector<std::unique_ptr<PinSetGpu>> pin_rotation_sets_;

  /* Host references of the bound sets: pin targets are read every substep, the multipliers are
   * written back by #download. */
  Vector<HaHairRodStretchSet> stretch_sets_host_;
  Vector<HaHairRodBendSet> bend_sets_host_;
  Vector<HaHairPinPositionSet> pin_position_sets_host_;
  Vector<HaHairPinRotationSet> pin_rotation_sets_host_;
};

}  // namespace blender::geometry::ha
