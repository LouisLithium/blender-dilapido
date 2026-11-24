/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Device implementation of the XPBD hair passes (stage 1). See `GEO_ha_hair.hh` for the scope and
 * the parity notes.
 */

#include "GEO_ha_hair.hh"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include "BLI_array.hh"
#include "BLI_math_base.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_vector.hh"
#include "BLI_vector.hh"

#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_storage_buffer.hh"

namespace blender::geometry::ha {

namespace {

constexpr int LOCAL_GROUP_SIZE = 64;

/** Floats per point in the packed property buffer. */
constexpr int HA_HAIR_PROP = 16;
/** Indices inside one point block of the packed property buffer. */
enum HairPropIndex {
  HA_PROP_INV_MASS = 0,
  /** Moment of inertia (3). */
  HA_PROP_MOMENT = 1,
  /** Inverse moment of inertia (3). */
  HA_PROP_INV_MOMENT = 4,
  /** External force (3). */
  HA_PROP_FORCE = 7,
  /** External acceleration (3), the wind of `Blender.DilapidoForce`. */
  HA_PROP_ACCEL = 10,
  /** External torque (3). */
  HA_PROP_TORQUE = 13,
};
/** Floats of the per frame constant buffer. */
constexpr int HA_HAIR_PARAM_COUNT = 8;
/** Indices inside the per frame constant buffer, mirrored by the GLSL prelude. */
enum HairParamIndex {
  HA_PARAM_COUNT = 0,
  HA_PARAM_DT = 1,
  HA_PARAM_INV_DT = 2,
  HA_PARAM_HAS_ACCEL = 3,
  /** `1 / dt^2`, the factor of every XPBD compliance term. */
  HA_PARAM_COMPLIANCE = 4,
  HA_PARAM_HAS_ROTATION = 5,
};

template<size_t N> Span<const HaShaderResource> as_span(const std::array<HaShaderResource, N> &r)
{
  return Span<const HaShaderResource>(r.data(), int64_t(r.size()));
}

template<size_t N> Span<gpu::StorageBuf *> buffer_span(std::array<gpu::StorageBuf *, N> &r)
{
  return Span<gpu::StorageBuf *>(r.data(), int64_t(r.size()));
}

/**
 * Binds the buffers to slots 0..n-1 and dispatches. Mirrors `dispatch_pass` of
 * `GEO_ha_xpbd.cc`; the binding cache skips the binds that are already in place.
 */
void dispatch_pass(gpu::Shader *shader, Span<gpu::StorageBuf *> buffers, const int element_count)
{
  if (shader == nullptr) {
    return;
  }
  GPU_shader_bind(shader);
  ha_bind_storage_buffers(buffers);
  ha_dispatch(shader, element_count);
  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE);
}

/**
 * Shared helpers. Quaternions are `vec4(w, x, y, z)`, the same memory order as
 * `math::Quaternion`, so the buffers can be uploaded and read back without a swizzle.
 *
 * Every function mirrors the BLI one the CPU solver uses:
 * - `ha_qnormalize` matches `math::normalize(math::Quaternion)`,
 * - `ha_transform_point` is the optimized `math::transform_point`,
 * - `ha_normalize_get_length` matches `math::normalize_and_get_length` (zero below 1e-35),
 * - `ha_safe_rcp` / `ha_safe_divide` match `math::safe_rcp` / `math::safe_divide`.
 */
const char *GLSL_HAIR_PRELUDE =
    "#define HA_HAIR_PROP 16\n"
    "#define HA_PROP_INV_MASS 0\n"
    "#define HA_PROP_MOMENT 1\n"
    "#define HA_PROP_INV_MOMENT 4\n"
    "#define HA_PROP_FORCE 7\n"
    "#define HA_PROP_ACCEL 10\n"
    "#define HA_PROP_TORQUE 13\n"
    "#define HA_PARAM_COUNT 0\n"
    "#define HA_PARAM_DT 1\n"
    "#define HA_PARAM_INV_DT 2\n"
    "#define HA_PARAM_HAS_ACCEL 3\n"
    "#define HA_PARAM_COMPLIANCE 4\n"
    "#define HA_PARAM_HAS_ROTATION 5\n"
    "const vec4 HA_QID = vec4(1.0, 0.0, 0.0, 0.0);\n"
    "vec4 ha_qmul(vec4 a, vec4 b)\n"
    "{\n"
    "  return vec4(a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,\n"
    "              a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,\n"
    "              a.x * b.z + a.z * b.x + a.w * b.y - a.y * b.w,\n"
    "              a.x * b.w + a.w * b.x + a.y * b.z - a.z * b.y);\n"
    "}\n"
    "vec4 ha_qconj(vec4 q) { return vec4(q.x, -q.y, -q.z, -q.w); }\n"
    "vec4 ha_qnormalize(vec4 q)\n"
    "{\n"
    "  float length_squared = dot(q, q);\n"
    "  if (length_squared == 0.0) { return HA_QID; }\n"
    "  return q * (1.0 / sqrt(length_squared));\n"
    "}\n"
    "vec3 ha_transform_point(vec4 q, vec3 v)\n"
    "{\n"
    "  float sw = -q.y * v.x - q.z * v.y - q.w * v.z;\n"
    "  float sx = q.x * v.x + q.z * v.z - q.w * v.y;\n"
    "  float sy = q.x * v.y + q.w * v.x - q.y * v.z;\n"
    "  float sz = q.x * v.z + q.y * v.y - q.z * v.x;\n"
    "  return vec3(sw * (-q.y) + sx * q.x - sy * q.w + sz * q.z,\n"
    "              sw * (-q.z) + sy * q.x - sz * q.y + sx * q.w,\n"
    "              sw * (-q.w) + sz * q.x - sx * q.z + sy * q.y);\n"
    "}\n"
    "vec3 ha_normalize_get_length(vec3 v, out float out_length)\n"
    "{\n"
    "  float length_squared = dot(v, v);\n"
    "  if (length_squared > 1e-35) {\n"
    "    out_length = sqrt(length_squared);\n"
    "    return v / out_length;\n"
    "  }\n"
    "  out_length = 0.0;\n"
    "  return vec3(0.0);\n"
    "}\n"
    "float ha_safe_rcp(float a) { return a != 0.0 ? 1.0 / a : 0.0; }\n"
    "vec3 ha_safe_divide(vec3 a, vec3 b)\n"
    "{\n"
    "  return vec3(b.x != 0.0 ? a.x / b.x : 0.0,\n"
    "              b.y != 0.0 ? a.y / b.y : 0.0,\n"
    "              b.z != 0.0 ? a.z / b.z : 0.0);\n"
    "}\n";

/** Copies the positions and rotations of the substep start into the "previous" buffers. */
const char *KERNEL_HAIR_COPY_PREV =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= int(params[HA_PARAM_COUNT])) { return; }\n"
    "  int b3 = index * 3;\n"
    "  prev_position_out[b3 + 0] = position_in[b3 + 0];\n"
    "  prev_position_out[b3 + 1] = position_in[b3 + 1];\n"
    "  prev_position_out[b3 + 2] = position_in[b3 + 2];\n"
    "  int b4 = index * 4;\n"
    "  prev_rotation_out[b4 + 0] = rotation_in[b4 + 0];\n"
    "  prev_rotation_out[b4 + 1] = rotation_in[b4 + 1];\n"
    "  prev_rotation_out[b4 + 2] = rotation_in[b4 + 2];\n"
    "  prev_rotation_out[b4 + 3] = rotation_in[b4 + 3];\n"
    "}\n";

/** `integrate_linear_velocities` + `integrate_angular_velocities` of the solver. */
const char *KERNEL_HAIR_PREDICT =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= int(params[HA_PARAM_COUNT])) { return; }\n"
    "  int b3 = index * 3;\n"
    "  vec3 old_position = vec3(prev_position_in[b3 + 0],\n"
    "                           prev_position_in[b3 + 1],\n"
    "                           prev_position_in[b3 + 2]);\n"
    "  int prop_base = index * HA_HAIR_PROP;\n"
    "  float inv_mass = props[prop_base + HA_PROP_INV_MASS];\n"
    "  vec3 force = vec3(props[prop_base + HA_PROP_FORCE + 0],\n"
    "                    props[prop_base + HA_PROP_FORCE + 1],\n"
    "                    props[prop_base + HA_PROP_FORCE + 2]);\n"
    "  vec3 acceleration = vec3(0.0);\n"
    "  if (params[HA_PARAM_HAS_ACCEL] > 0.5 && inv_mass > 0.0) {\n"
    "    acceleration = vec3(props[prop_base + HA_PROP_ACCEL + 0],\n"
    "                        props[prop_base + HA_PROP_ACCEL + 1],\n"
    "                        props[prop_base + HA_PROP_ACCEL + 2]);\n"
    "  }\n"
    "  vec3 velocity = vec3(velocity_io[b3 + 0], velocity_io[b3 + 1], velocity_io[b3 + 2]);\n"
    "  velocity += (force * inv_mass + acceleration) * params[HA_PARAM_DT];\n"
    "  velocity_io[b3 + 0] = velocity.x;\n"
    "  velocity_io[b3 + 1] = velocity.y;\n"
    "  velocity_io[b3 + 2] = velocity.z;\n"
    "  vec3 position = old_position + velocity * params[HA_PARAM_DT];\n"
    "  position_out[b3 + 0] = position.x;\n"
    "  position_out[b3 + 1] = position.y;\n"
    "  position_out[b3 + 2] = position.z;\n"
    "  if (params[HA_PARAM_HAS_ROTATION] < 0.5) { return; }\n"
    "  vec3 inv_inertia = vec3(props[prop_base + HA_PROP_INV_MOMENT + 0],\n"
    "                          props[prop_base + HA_PROP_INV_MOMENT + 1],\n"
    "                          props[prop_base + HA_PROP_INV_MOMENT + 2]);\n"
    "  int b4 = index * 4;\n"
    "  vec4 old_rotation = vec4(prev_rotation_in[b4 + 0],\n"
    "                           prev_rotation_in[b4 + 1],\n"
    "                           prev_rotation_in[b4 + 2],\n"
    "                           prev_rotation_in[b4 + 3]);\n"
    "  if (all(equal(inv_inertia, vec3(0.0)))) {\n"
    "    rotation_out[b4 + 0] = old_rotation.x;\n"
    "    rotation_out[b4 + 1] = old_rotation.y;\n"
    "    rotation_out[b4 + 2] = old_rotation.z;\n"
    "    rotation_out[b4 + 3] = old_rotation.w;\n"
    "    return;\n"
    "  }\n"
    "  vec3 angular_velocity = vec3(angular_io[b3 + 0],\n"
    "                               angular_io[b3 + 1],\n"
    "                               angular_io[b3 + 2]);\n"
    "  vec3 torque = vec3(props[prop_base + HA_PROP_TORQUE + 0],\n"
    "                     props[prop_base + HA_PROP_TORQUE + 1],\n"
    "                     props[prop_base + HA_PROP_TORQUE + 2]);\n"
    "  vec3 precession = cross(angular_velocity, ha_safe_divide(angular_velocity, inv_inertia));\n"
    "  angular_velocity += params[HA_PARAM_DT] * (torque - precession) * inv_inertia;\n"
    "  angular_io[b3 + 0] = angular_velocity.x;\n"
    "  angular_io[b3 + 1] = angular_velocity.y;\n"
    "  angular_io[b3 + 2] = angular_velocity.z;\n"
    "  vec4 direction = ha_qmul(old_rotation, vec4(0.0, angular_velocity.x,\n"
    "                                              angular_velocity.y, angular_velocity.z));\n"
    "  vec4 rotation = ha_qnormalize(old_rotation + params[HA_PARAM_DT] * 0.5 * direction);\n"
    "  rotation_out[b4 + 0] = rotation.x;\n"
    "  rotation_out[b4 + 1] = rotation.y;\n"
    "  rotation_out[b4 + 2] = rotation.z;\n"
    "  rotation_out[b4 + 3] = rotation.w;\n"
    "}\n";

/**
 * `evaluate_rod_stretch_and_shear_constraint`, one invocation per curve, segments in order. The
 * segment loop reproduces the serial solve of `RodStretchAndShearConstraintSet::solve_single`
 * exactly (the curves are independent on the CPU as well, see
 * `color_constraints__all_independent`).
 */
const char *KERNEL_HAIR_ROD_STRETCH =
    "void main()\n"
    "{\n"
    "  int curve_i = int(gl_GlobalInvocationID.x);\n"
    "  if (curve_i >= int(set_params[0])) { return; }\n"
    "  int first = curve_first_in[curve_i];\n"
    "  int size = curve_size_in[curve_i];\n"
    "  float compliance_factor = params[HA_PARAM_COMPLIANCE];\n"
    "  for (int i = 0; i < size - 1; i++) {\n"
    "    int p0 = first + i;\n"
    "    int p1 = p0 + 1;\n"
    "    int b0 = p0 * 3;\n"
    "    int b1 = p1 * 3;\n"
    "    vec3 position0 = vec3(position_io[b0 + 0], position_io[b0 + 1], position_io[b0 + 2]);\n"
    "    vec3 position1 = vec3(position_io[b1 + 0], position_io[b1 + 1], position_io[b1 + 2]);\n"
    "    int r0 = p0 * 4;\n"
    "    vec4 rotation = vec4(rotation_io[r0 + 0], rotation_io[r0 + 1],\n"
    "                         rotation_io[r0 + 2], rotation_io[r0 + 3]);\n"
    "    int prop0 = p0 * HA_HAIR_PROP;\n"
    "    int prop1 = p1 * HA_HAIR_PROP;\n"
    "    float inv_mass0 = props[prop0 + HA_PROP_INV_MASS];\n"
    "    float inv_mass1 = props[prop1 + HA_PROP_INV_MASS];\n"
    "    vec3 inertia = vec3(props[prop0 + HA_PROP_MOMENT + 0],\n"
    "                        props[prop0 + HA_PROP_MOMENT + 1],\n"
    "                        props[prop0 + HA_PROP_MOMENT + 2]);\n"
    "    float inv_lumped_inertia = ha_safe_rcp(0.5 * (inertia.x + inertia.y + inertia.z));\n"
    "    int data_i = p1 * 2;\n"
    "    float rest_length = rod_data[data_i + 0];\n"
    "    float compliance = rod_data[data_i + 1] * compliance_factor;\n"
    "    int lambda_i = p1 * 6;\n"
    "    vec3 lambda_pos = vec3(rod_lambda_io[lambda_i + 0],\n"
    "                           rod_lambda_io[lambda_i + 1],\n"
    "                           rod_lambda_io[lambda_i + 2]);\n"
    "    vec3 lambda_rot = vec3(rod_lambda_io[lambda_i + 3],\n"
    "                           rod_lambda_io[lambda_i + 4],\n"
    "                           rod_lambda_io[lambda_i + 5]);\n"
    "    if (inv_mass0 == 0.0 && inv_mass1 == 0.0 && inv_lumped_inertia == 0.0) { continue; }\n"
    "    vec3 p_diff = position1 - position0;\n"
    "    float p_len = sqrt(dot(p_diff, p_diff));\n"
    "    vec3 forward_rest = ha_transform_point(rotation, vec3(0.0, 0.0, rest_length));\n"
    "    vec3 forward = ha_transform_point(rotation, vec3(0.0, 0.0, p_len));\n"
    "    vec3 residual_pos = p_diff - forward_rest;\n"
    "    vec3 residual_rot = p_diff - forward;\n"
    "    float weight_sum = inv_mass0 + inv_mass1 +\n"
    "                       4.0 * inv_lumped_inertia * rest_length * rest_length;\n"
    "    float weight_sum_rot = inv_mass0 + inv_mass1 +\n"
    "                           4.0 * inv_lumped_inertia * p_len * p_len;\n"
    "    vec3 delta_lambda_pos = (-residual_pos - compliance * lambda_pos) /\n"
    "                            (weight_sum + compliance);\n"
    "    vec3 delta_lambda_rot = (-residual_rot - compliance * lambda_rot) /\n"
    "                            (weight_sum_rot + compliance);\n"
    "    rod_lambda_io[lambda_i + 0] = lambda_pos.x + delta_lambda_pos.x;\n"
    "    rod_lambda_io[lambda_i + 1] = lambda_pos.y + delta_lambda_pos.y;\n"
    "    rod_lambda_io[lambda_i + 2] = lambda_pos.z + delta_lambda_pos.z;\n"
    "    rod_lambda_io[lambda_i + 3] = lambda_rot.x + delta_lambda_rot.x;\n"
    "    rod_lambda_io[lambda_i + 4] = lambda_rot.y + delta_lambda_rot.y;\n"
    "    rod_lambda_io[lambda_i + 5] = lambda_rot.z + delta_lambda_rot.z;\n"
    "    vec3 offset0 = -delta_lambda_pos * inv_mass0;\n"
    "    vec3 offset1 = delta_lambda_pos * inv_mass1;\n"
    "    vec3 position0_new = position0 + offset0;\n"
    "    vec3 position1_new = position1 + offset1;\n"
    "    position_io[b0 + 0] = position0_new.x;\n"
    "    position_io[b0 + 1] = position0_new.y;\n"
    "    position_io[b0 + 2] = position0_new.z;\n"
    "    position_io[b1 + 0] = position1_new.x;\n"
    "    position_io[b1 + 1] = position1_new.y;\n"
    "    position_io[b1 + 2] = position1_new.z;\n"
    "    vec3 rotation_offset = -delta_lambda_rot * inv_lumped_inertia * p_len;\n"
    "    vec4 offset_rot = ha_qmul(ha_qmul(vec4(0.0, rotation_offset.x,\n"
    "                                           rotation_offset.y, rotation_offset.z), rotation),\n"
    "                              vec4(0.0, 0.0, 0.0, -1.0));\n"
    "    vec4 rotation_new = ha_qnormalize(rotation + offset_rot);\n"
    "    rotation_io[r0 + 0] = rotation_new.x;\n"
    "    rotation_io[r0 + 1] = rotation_new.y;\n"
    "    rotation_io[r0 + 2] = rotation_new.z;\n"
    "    rotation_io[r0 + 3] = rotation_new.w;\n"
    "  }\n"
    "}\n";

/** `evaluate_align_rotations_constraint`, one invocation per curve, segments in order. */
const char *KERNEL_HAIR_ROD_BEND =
    "void main()\n"
    "{\n"
    "  int curve_i = int(gl_GlobalInvocationID.x);\n"
    "  if (curve_i >= int(set_params[0])) { return; }\n"
    "  int first = curve_first_in[curve_i];\n"
    "  int size = curve_size_in[curve_i];\n"
    "  float compliance_factor = params[HA_PARAM_COMPLIANCE];\n"
    "  for (int i = 0; i < size - 2; i++) {\n"
    "    int p0 = first + i;\n"
    "    int p1 = p0 + 1;\n"
    "    int r0 = p0 * 4;\n"
    "    int r1 = p1 * 4;\n"
    "    vec4 rotation0 = vec4(rotation_io[r0 + 0], rotation_io[r0 + 1],\n"
    "                          rotation_io[r0 + 2], rotation_io[r0 + 3]);\n"
    "    vec4 rotation1 = vec4(rotation_io[r1 + 0], rotation_io[r1 + 1],\n"
    "                          rotation_io[r1 + 2], rotation_io[r1 + 3]);\n"
    "    int prop0 = p0 * HA_HAIR_PROP;\n"
    "    int prop1 = p1 * HA_HAIR_PROP;\n"
    "    vec3 inertia0 = vec3(props[prop0 + HA_PROP_MOMENT + 0],\n"
    "                         props[prop0 + HA_PROP_MOMENT + 1],\n"
    "                         props[prop0 + HA_PROP_MOMENT + 2]);\n"
    "    vec3 inertia1 = vec3(props[prop1 + HA_PROP_MOMENT + 0],\n"
    "                         props[prop1 + HA_PROP_MOMENT + 1],\n"
    "                         props[prop1 + HA_PROP_MOMENT + 2]);\n"
    "    float inv_lumped_inertia0 = ha_safe_rcp(0.5 * (inertia0.x + inertia0.y + inertia0.z));\n"
    "    float inv_lumped_inertia1 = ha_safe_rcp(0.5 * (inertia1.x + inertia1.y + inertia1.z));\n"
    "    if (inv_lumped_inertia0 == 0.0 && inv_lumped_inertia1 == 0.0) { continue; }\n"
    "    int data_i = p0 * 5;\n"
    "    vec4 rest_rotation = vec4(bend_data[data_i + 0], bend_data[data_i + 1],\n"
    "                              bend_data[data_i + 2], bend_data[data_i + 3]);\n"
    "    float compliance = bend_data[data_i + 4] * compliance_factor;\n"
    "    int lambda_i = p0 * 4;\n"
    "    vec4 lambda = vec4(bend_lambda_io[lambda_i + 0], bend_lambda_io[lambda_i + 1],\n"
    "                       bend_lambda_io[lambda_i + 2], bend_lambda_io[lambda_i + 3]);\n"
    "    vec4 rotation_diff = ha_qmul(ha_qconj(rotation0), rotation1);\n"
    "    vec4 residual_neg = rotation_diff - rest_rotation;\n"
    "    vec4 residual_pos = rotation_diff + rest_rotation;\n"
    "    vec4 residual = dot(residual_neg, residual_neg) < dot(residual_pos, residual_pos) ?\n"
    "                        residual_neg :\n"
    "                        residual_pos;\n"
    "    vec4 delta_lambda = (-residual - compliance * lambda) /\n"
    "                        (inv_lumped_inertia0 + inv_lumped_inertia1 + compliance);\n"
    "    bend_lambda_io[lambda_i + 0] = lambda.x + delta_lambda.x;\n"
    "    bend_lambda_io[lambda_i + 1] = lambda.y + delta_lambda.y;\n"
    "    bend_lambda_io[lambda_i + 2] = lambda.z + delta_lambda.z;\n"
    "    bend_lambda_io[lambda_i + 3] = lambda.w + delta_lambda.w;\n"
    "    vec4 offset0 = ha_qmul(rotation1, ha_qconj(delta_lambda * inv_lumped_inertia0));\n"
    "    vec4 offset1 = ha_qmul(rotation0, delta_lambda * inv_lumped_inertia1);\n"
    "    vec4 rotation0_new = ha_qnormalize(rotation0 + offset0);\n"
    "    vec4 rotation1_new = ha_qnormalize(rotation1 + offset1);\n"
    "    rotation_io[r0 + 0] = rotation0_new.x;\n"
    "    rotation_io[r0 + 1] = rotation0_new.y;\n"
    "    rotation_io[r0 + 2] = rotation0_new.z;\n"
    "    rotation_io[r0 + 3] = rotation0_new.w;\n"
    "    rotation_io[r1 + 0] = rotation1_new.x;\n"
    "    rotation_io[r1 + 1] = rotation1_new.y;\n"
    "    rotation_io[r1 + 2] = rotation1_new.z;\n"
    "    rotation_io[r1 + 3] = rotation1_new.w;\n"
    "  }\n"
    "}\n";

/** `PinPositionConstraintSet::solve_single` (a distance constraint with a fixed target). */
const char *KERNEL_HAIR_PIN_POSITION =
    "void main()\n"
    "{\n"
    "  int pin_i = int(gl_GlobalInvocationID.x);\n"
    "  if (pin_i >= int(set_params[0])) { return; }\n"
    "  int point_i = points_in[pin_i];\n"
    "  int b3 = point_i * 3;\n"
    "  vec3 position = vec3(position_io[b3 + 0], position_io[b3 + 1], position_io[b3 + 2]);\n"
    "  float inv_mass = props[point_i * HA_HAIR_PROP + HA_PROP_INV_MASS];\n"
    "  vec3 target = vec3(target_in[pin_i * 3 + 0],\n"
    "                     target_in[pin_i * 3 + 1],\n"
    "                     target_in[pin_i * 3 + 2]);\n"
    "  float compliance = compliance_in[pin_i] * params[HA_PARAM_COMPLIANCE];\n"
    "  float lambda = lambda_io[pin_i];\n"
    "  if (inv_mass == 0.0) { return; }\n"
    "  vec3 p_diff = target - position;\n"
    "  float target_distance;\n"
    "  vec3 direction = ha_normalize_get_length(p_diff, target_distance);\n"
    "  float delta_lambda = (-target_distance - compliance * lambda) /\n"
    "                       (inv_mass + compliance);\n"
    "  lambda_io[pin_i] = lambda + delta_lambda;\n"
    "  vec3 offset = -delta_lambda * inv_mass * direction;\n"
    "  position_io[b3 + 0] = position.x + offset.x;\n"
    "  position_io[b3 + 1] = position.y + offset.y;\n"
    "  position_io[b3 + 2] = position.z + offset.z;\n"
    "}\n";

/** `PinRotationConstraintSet::solve_single`. */
const char *KERNEL_HAIR_PIN_ROTATION =
    "void main()\n"
    "{\n"
    "  int pin_i = int(gl_GlobalInvocationID.x);\n"
    "  if (pin_i >= int(set_params[0])) { return; }\n"
    "  int point_i = points_in[pin_i];\n"
    "  int b4 = point_i * 4;\n"
    "  vec4 rotation = vec4(rotation_io[b4 + 0], rotation_io[b4 + 1],\n"
    "                       rotation_io[b4 + 2], rotation_io[b4 + 3]);\n"
    "  int prop_base = point_i * HA_HAIR_PROP;\n"
    "  vec3 inertia = vec3(props[prop_base + HA_PROP_MOMENT + 0],\n"
    "                      props[prop_base + HA_PROP_MOMENT + 1],\n"
    "                      props[prop_base + HA_PROP_MOMENT + 2]);\n"
    "  float inv_lumped_inertia = ha_safe_rcp(0.5 * (inertia.x + inertia.y + inertia.z));\n"
    "  vec4 target = vec4(target_in[pin_i * 4 + 0], target_in[pin_i * 4 + 1],\n"
    "                     target_in[pin_i * 4 + 2], target_in[pin_i * 4 + 3]);\n"
    "  float compliance = compliance_in[pin_i] * params[HA_PARAM_COMPLIANCE];\n"
    "  vec4 lambda = vec4(lambda_io[pin_i * 4 + 0], lambda_io[pin_i * 4 + 1],\n"
    "                     lambda_io[pin_i * 4 + 2], lambda_io[pin_i * 4 + 3]);\n"
    "  if (inv_lumped_inertia == 0.0) { return; }\n"
    "  vec4 rest_rotation = HA_QID;\n"
    "  vec4 rotation_diff = ha_qmul(ha_qconj(rotation), target);\n"
    "  vec4 residual_neg = rotation_diff - rest_rotation;\n"
    "  vec4 residual_pos = rotation_diff + rest_rotation;\n"
    "  vec4 residual = dot(residual_neg, residual_neg) < dot(residual_pos, residual_pos) ?\n"
    "                      residual_neg :\n"
    "                      residual_pos;\n"
    "  vec4 delta_lambda = (-residual - compliance * lambda) / (inv_lumped_inertia + compliance);\n"
    "  lambda_io[pin_i * 4 + 0] = lambda.x + delta_lambda.x;\n"
    "  lambda_io[pin_i * 4 + 1] = lambda.y + delta_lambda.y;\n"
    "  lambda_io[pin_i * 4 + 2] = lambda.z + delta_lambda.z;\n"
    "  lambda_io[pin_i * 4 + 3] = lambda.w + delta_lambda.w;\n"
    "  vec4 offset = ha_qmul(target, ha_qconj(delta_lambda * inv_lumped_inertia));\n"
    "  vec4 rotation_new = ha_qnormalize(rotation + offset);\n"
    "  rotation_io[b4 + 0] = rotation_new.x;\n"
    "  rotation_io[b4 + 1] = rotation_new.y;\n"
    "  rotation_io[b4 + 2] = rotation_new.z;\n"
    "  rotation_io[b4 + 3] = rotation_new.w;\n"
    "}\n";

/** `update_linear_velocities` + `update_angular_velocities` of the solver. */
const char *KERNEL_HAIR_UPDATE_VELOCITY =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= int(params[HA_PARAM_COUNT])) { return; }\n"
    "  int b3 = index * 3;\n"
    "  vec3 position = vec3(position_in[b3 + 0], position_in[b3 + 1], position_in[b3 + 2]);\n"
    "  vec3 prev_position = vec3(prev_position_in[b3 + 0],\n"
    "                            prev_position_in[b3 + 1],\n"
    "                            prev_position_in[b3 + 2]);\n"
    "  vec3 velocity = (position - prev_position) * params[HA_PARAM_INV_DT];\n"
    "  velocity_out[b3 + 0] = velocity.x;\n"
    "  velocity_out[b3 + 1] = velocity.y;\n"
    "  velocity_out[b3 + 2] = velocity.z;\n"
    "  if (params[HA_PARAM_HAS_ROTATION] < 0.5) { return; }\n"
    "  int b4 = index * 4;\n"
    "  vec4 prev_rotation = vec4(prev_rotation_in[b4 + 0], prev_rotation_in[b4 + 1],\n"
    "                            prev_rotation_in[b4 + 2], prev_rotation_in[b4 + 3]);\n"
    "  vec4 rotation = vec4(rotation_in[b4 + 0], rotation_in[b4 + 1],\n"
    "                       rotation_in[b4 + 2], rotation_in[b4 + 3]);\n"
    "  vec4 diff_q = ha_qmul(ha_qconj(prev_rotation), rotation);\n"
    "  vec3 diff = vec3(diff_q.y, diff_q.z, diff_q.w);\n"
    "  if (abs(diff.x) < 1e-5) { diff.x = 0.0; }\n"
    "  if (abs(diff.y) < 1e-5) { diff.y = 0.0; }\n"
    "  if (abs(diff.z) < 1e-5) { diff.z = 0.0; }\n"
    "  vec3 angular_velocity = 2.0 * diff * params[HA_PARAM_INV_DT];\n"
    "  angular_out[b3 + 0] = angular_velocity.x;\n"
    "  angular_out[b3 + 1] = angular_velocity.y;\n"
    "  angular_out[b3 + 2] = angular_velocity.z;\n"
    "}\n";

/** `LinearDampingConstraintSet::solve_single`. */
const char *KERNEL_HAIR_DAMPING_LINEAR =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= int(params[HA_PARAM_COUNT])) { return; }\n"
    "  int b3 = index * 3;\n"
    "  vec3 velocity = vec3(velocity_io[b3 + 0], velocity_io[b3 + 1], velocity_io[b3 + 2]);\n"
    "  float factor = clamp(params[HA_PARAM_DT] * damping_in[index], 0.0, 1.0);\n"
    "  float lambda = lambda_io[index];\n"
    "  float velocity_length;\n"
    "  vec3 direction = ha_normalize_get_length(velocity, velocity_length);\n"
    "  float delta_lambda = -velocity_length * factor - lambda;\n"
    "  lambda_io[index] = lambda + delta_lambda;\n"
    "  vec3 offset = direction * delta_lambda;\n"
    "  velocity_io[b3 + 0] = velocity.x + offset.x;\n"
    "  velocity_io[b3 + 1] = velocity.y + offset.y;\n"
    "  velocity_io[b3 + 2] = velocity.z + offset.z;\n"
    "}\n";

/** `AngularDampingConstraintSet::solve_single`. */
const char *KERNEL_HAIR_DAMPING_ANGULAR =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= int(params[HA_PARAM_COUNT])) { return; }\n"
    "  int b3 = index * 3;\n"
    "  vec3 velocity = vec3(angular_io[b3 + 0], angular_io[b3 + 1], angular_io[b3 + 2]);\n"
    "  float factor = params[HA_PARAM_DT] * damping_in[index];\n"
    "  float lambda = lambda_io[index];\n"
    "  float angular_length;\n"
    "  vec3 direction = ha_normalize_get_length(velocity, angular_length);\n"
    "  float delta_lambda = -angular_length * factor - lambda;\n"
    "  lambda_io[index] = lambda + delta_lambda;\n"
    "  vec3 offset = direction * delta_lambda;\n"
    "  angular_io[b3 + 0] = velocity.x + offset.x;\n"
    "  angular_io[b3 + 1] = velocity.y + offset.y;\n"
    "  angular_io[b3 + 2] = velocity.z + offset.z;\n"
    "}\n";

/* -------------------------------------------------------------------- */
/** \name Resource layouts
 * \{ */

const std::array<HaShaderResource, 5> COPY_PREV_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Write, "float", "prev_position_out[]"},
    {2, HaShaderQualifier::Read, "float", "rotation_in[]"},
    {3, HaShaderQualifier::Write, "float", "prev_rotation_out[]"},
    {4, HaShaderQualifier::Read, "float", "params[]"},
}};

const std::array<HaShaderResource, 8> PREDICT_RESOURCES = {{
    {0, HaShaderQualifier::Write, "float", "position_out[]"},
    {1, HaShaderQualifier::Read, "float", "prev_position_in[]"},
    {2, HaShaderQualifier::ReadWrite, "float", "velocity_io[]"},
    {3, HaShaderQualifier::Write, "float", "rotation_out[]"},
    {4, HaShaderQualifier::Read, "float", "prev_rotation_in[]"},
    {5, HaShaderQualifier::ReadWrite, "float", "angular_io[]"},
    {6, HaShaderQualifier::Read, "float", "props[]"},
    {7, HaShaderQualifier::Read, "float", "params[]"},
}};

const std::array<HaShaderResource, 9> ROD_STRETCH_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "position_io[]"},
    {1, HaShaderQualifier::ReadWrite, "float", "rotation_io[]"},
    {2, HaShaderQualifier::Read, "float", "props[]"},
    {3, HaShaderQualifier::Read, "int", "curve_first_in[]"},
    {4, HaShaderQualifier::Read, "int", "curve_size_in[]"},
    {5, HaShaderQualifier::Read, "float", "rod_data[]"},
    {6, HaShaderQualifier::ReadWrite, "float", "rod_lambda_io[]"},
    {7, HaShaderQualifier::Read, "float", "set_params[]"},
    {8, HaShaderQualifier::Read, "float", "params[]"},
}};

const std::array<HaShaderResource, 8> ROD_BEND_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "rotation_io[]"},
    {1, HaShaderQualifier::Read, "float", "props[]"},
    {2, HaShaderQualifier::Read, "int", "curve_first_in[]"},
    {3, HaShaderQualifier::Read, "int", "curve_size_in[]"},
    {4, HaShaderQualifier::Read, "float", "bend_data[]"},
    {5, HaShaderQualifier::ReadWrite, "float", "bend_lambda_io[]"},
    {6, HaShaderQualifier::Read, "float", "set_params[]"},
    {7, HaShaderQualifier::Read, "float", "params[]"},
}};

const std::array<HaShaderResource, 8> PIN_POSITION_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "position_io[]"},
    {1, HaShaderQualifier::Read, "float", "props[]"},
    {2, HaShaderQualifier::Read, "int", "points_in[]"},
    {3, HaShaderQualifier::Read, "float", "target_in[]"},
    {4, HaShaderQualifier::Read, "float", "compliance_in[]"},
    {5, HaShaderQualifier::ReadWrite, "float", "lambda_io[]"},
    {6, HaShaderQualifier::Read, "float", "set_params[]"},
    {7, HaShaderQualifier::Read, "float", "params[]"},
}};

const std::array<HaShaderResource, 8> PIN_ROTATION_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "rotation_io[]"},
    {1, HaShaderQualifier::Read, "float", "props[]"},
    {2, HaShaderQualifier::Read, "int", "points_in[]"},
    {3, HaShaderQualifier::Read, "float", "target_in[]"},
    {4, HaShaderQualifier::Read, "float", "compliance_in[]"},
    {5, HaShaderQualifier::ReadWrite, "float", "lambda_io[]"},
    {6, HaShaderQualifier::Read, "float", "set_params[]"},
    {7, HaShaderQualifier::Read, "float", "params[]"},
}};

const std::array<HaShaderResource, 7> UPDATE_VELOCITY_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "float", "prev_position_in[]"},
    {2, HaShaderQualifier::Write, "float", "velocity_out[]"},
    {3, HaShaderQualifier::Read, "float", "rotation_in[]"},
    {4, HaShaderQualifier::Read, "float", "prev_rotation_in[]"},
    {5, HaShaderQualifier::Write, "float", "angular_out[]"},
    {6, HaShaderQualifier::Read, "float", "params[]"},
}};

const std::array<HaShaderResource, 4> DAMPING_LINEAR_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "velocity_io[]"},
    {1, HaShaderQualifier::Read, "float", "damping_in[]"},
    {2, HaShaderQualifier::ReadWrite, "float", "lambda_io[]"},
    {3, HaShaderQualifier::Read, "float", "params[]"},
}};

const std::array<HaShaderResource, 4> DAMPING_ANGULAR_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "angular_io[]"},
    {1, HaShaderQualifier::Read, "float", "damping_in[]"},
    {2, HaShaderQualifier::ReadWrite, "float", "lambda_io[]"},
    {3, HaShaderQualifier::Read, "float", "params[]"},
}};

/** \} */

float4 quaternion_to_float4(const math::Quaternion &q)
{
  return float4(q.w, q.x, q.y, q.z);
}

float3 read_float3(const Span<float> data, const int index)
{
  return float3(data[index * 3 + 0], data[index * 3 + 1], data[index * 3 + 2]);
}

}  // namespace

/* -------------------------------------------------------------------- */
/** \name HaXpbdHair
 * \{ */

HaXpbdHair::~HaXpbdHair()
{
  this->free();
}

void HaXpbdHair::free()
{
  positions_.free();
  prev_positions_.free();
  velocities_.free();
  rotations_.free();
  prev_rotations_.free();
  angular_velocities_.free();
  props_.free();
  params_.free();
  rod_sets_.clear();
  bend_sets_.clear();
  velocity_sets_.clear();
  pin_position_sets_.clear();
  pin_rotation_sets_.clear();
  stretch_sets_host_.clear();
  bend_sets_host_.clear();
  pin_position_sets_host_.clear();
  pin_rotation_sets_host_.clear();
  point_count_ = 0;
  ready_ = false;
  has_rotation_ = false;
  has_external_acceleration_ = false;
  delta_time_ = 0.0f;
  stats_ = {};
}

bool HaXpbdHair::ensure_shaders()
{
  const std::string prelude(GLSL_HAIR_PRELUDE);
  const auto get = [&](const StringRef key,
                       const char *source,
                       const Span<const HaShaderResource> resources) {
    return ha_shader_get_source(key, prelude + source, resources, LOCAL_GROUP_SIZE);
  };
  if (get("hair|copy_prev", KERNEL_HAIR_COPY_PREV, as_span(COPY_PREV_RESOURCES)) == nullptr ||
      get("hair|predict", KERNEL_HAIR_PREDICT, as_span(PREDICT_RESOURCES)) == nullptr ||
      get("hair|rod_stretch", KERNEL_HAIR_ROD_STRETCH, as_span(ROD_STRETCH_RESOURCES)) ==
          nullptr ||
      get("hair|rod_bend", KERNEL_HAIR_ROD_BEND, as_span(ROD_BEND_RESOURCES)) == nullptr ||
      get("hair|pin_position", KERNEL_HAIR_PIN_POSITION, as_span(PIN_POSITION_RESOURCES)) ==
          nullptr ||
      get("hair|pin_rotation", KERNEL_HAIR_PIN_ROTATION, as_span(PIN_ROTATION_RESOURCES)) ==
          nullptr ||
      get("hair|update_velocity",
          KERNEL_HAIR_UPDATE_VELOCITY,
          as_span(UPDATE_VELOCITY_RESOURCES)) == nullptr ||
      get("hair|damping_linear", KERNEL_HAIR_DAMPING_LINEAR, as_span(DAMPING_LINEAR_RESOURCES)) ==
          nullptr ||
      get("hair|damping_angular",
          KERNEL_HAIR_DAMPING_ANGULAR,
          as_span(DAMPING_ANGULAR_RESOURCES)) == nullptr)
  {
    last_error_ = "shader creation failed";
    return false;
  }
  return true;
}

bool HaXpbdHair::bind(const HaHairInputs &inputs,
                      const Span<const HaHairRodStretchSet> stretch_sets,
                      const Span<const HaHairRodBendSet> bend_sets,
                      const Span<const HaHairVelocitySet> velocity_sets,
                      const Span<const HaHairPinPositionSet> pin_position_sets,
                      const Span<const HaHairPinRotationSet> pin_rotation_sets,
                      const float delta_time)
{
  this->free();
  last_error_ = "";
  const int count = int(inputs.positions.size());
  if (count == 0 || delta_time < 0.0f) {
    last_error_ = "empty state or negative time step";
    return false;
  }
  if (!ha_gpu_available()) {
    last_error_ = "no GPU backend";
    return false;
  }
  /* The host may have bound other buffers since the last frame (draw, compositor, another node),
   * so the binding cache starts empty, exactly like the fluid does in its own bind. */
  ha_bindings_invalidate();
  HaContextGuard guard;
  if (!guard.is_valid()) {
    last_error_ = "no GPU context";
    return false;
  }
  const bool has_rotation = !inputs.rotations.is_empty();
  if (has_rotation != !inputs.angular_velocities.is_empty()) {
    last_error_ = "input size mismatch";
    return false;
  }
  if (int(inputs.velocities.size()) != count || int(inputs.external_force.size()) != count ||
      int(inputs.external_torque.size()) != count || int(inputs.inv_masses.size()) != count ||
      int(inputs.moments_of_inertia.size()) != count ||
      int(inputs.inv_moments_of_inertia.size()) != count ||
      int(inputs.rotations.size()) != (has_rotation ? count : 0) ||
      int(inputs.angular_velocities.size()) != (has_rotation ? count : 0) ||
      (!inputs.external_acceleration.is_empty() &&
       int(inputs.external_acceleration.size()) != count))
  {
    last_error_ = "input size mismatch";
    return false;
  }

  point_count_ = count;
  has_rotation_ = has_rotation;
  has_external_acceleration_ = int(inputs.external_acceleration.size()) == count;
  delta_time_ = delta_time;
  stats_ = {};

  const int64_t vec3_floats = int64_t(count) * 3;
  const int64_t vec4_floats = int64_t(count) * 4;
  const int64_t prop_floats = int64_t(count) * HA_HAIR_PROP;
  if (!positions_.ensure(vec3_floats) || !prev_positions_.ensure(vec3_floats) ||
      !velocities_.ensure(vec3_floats) || !rotations_.ensure(vec4_floats) ||
      !prev_rotations_.ensure(vec4_floats) || !angular_velocities_.ensure(vec3_floats) ||
      !props_.ensure(prop_floats) || !params_.ensure(HA_HAIR_PARAM_COUNT))
  {
    last_error_ = "buffer allocation failed";
    return false;
  }

  Array<float> host_positions(vec3_floats);
  Array<float> host_velocities(vec3_floats);
  Array<float> host_angular(vec3_floats);
  Array<float> host_rotations(vec4_floats);
  Array<float> host_props(prop_floats);
  for (const int i : IndexRange(count)) {
    const int b3 = i * 3;
    const int b4 = i * 4;
    host_positions[b3 + 0] = inputs.positions[i].x;
    host_positions[b3 + 1] = inputs.positions[i].y;
    host_positions[b3 + 2] = inputs.positions[i].z;
    host_velocities[b3 + 0] = inputs.velocities[i].x;
    host_velocities[b3 + 1] = inputs.velocities[i].y;
    host_velocities[b3 + 2] = inputs.velocities[i].z;
    if (has_rotation) {
      const float4 rotation = quaternion_to_float4(inputs.rotations[i]);
      host_rotations[b4 + 0] = rotation.x;
      host_rotations[b4 + 1] = rotation.y;
      host_rotations[b4 + 2] = rotation.z;
      host_rotations[b4 + 3] = rotation.w;
      host_angular[b3 + 0] = inputs.angular_velocities[i].x;
      host_angular[b3 + 1] = inputs.angular_velocities[i].y;
      host_angular[b3 + 2] = inputs.angular_velocities[i].z;
    }
    else {
      host_rotations[b4 + 0] = 1.0f;
    }
    const int p = i * HA_HAIR_PROP;
    host_props[p + HA_PROP_INV_MASS] = inputs.inv_masses[i];
    host_props[p + HA_PROP_MOMENT + 0] = inputs.moments_of_inertia[i].x;
    host_props[p + HA_PROP_MOMENT + 1] = inputs.moments_of_inertia[i].y;
    host_props[p + HA_PROP_MOMENT + 2] = inputs.moments_of_inertia[i].z;
    host_props[p + HA_PROP_INV_MOMENT + 0] = inputs.inv_moments_of_inertia[i].x;
    host_props[p + HA_PROP_INV_MOMENT + 1] = inputs.inv_moments_of_inertia[i].y;
    host_props[p + HA_PROP_INV_MOMENT + 2] = inputs.inv_moments_of_inertia[i].z;
    host_props[p + HA_PROP_FORCE + 0] = inputs.external_force[i].x;
    host_props[p + HA_PROP_FORCE + 1] = inputs.external_force[i].y;
    host_props[p + HA_PROP_FORCE + 2] = inputs.external_force[i].z;
    if (has_external_acceleration_) {
      host_props[p + HA_PROP_ACCEL + 0] = inputs.external_acceleration[i].x;
      host_props[p + HA_PROP_ACCEL + 1] = inputs.external_acceleration[i].y;
      host_props[p + HA_PROP_ACCEL + 2] = inputs.external_acceleration[i].z;
    }
    host_props[p + HA_PROP_TORQUE + 0] = inputs.external_torque[i].x;
    host_props[p + HA_PROP_TORQUE + 1] = inputs.external_torque[i].y;
    host_props[p + HA_PROP_TORQUE + 2] = inputs.external_torque[i].z;
  }
  positions_.upload(host_positions);
  prev_positions_.upload(host_positions);
  velocities_.upload(host_velocities);
  rotations_.upload(host_rotations);
  prev_rotations_.upload(host_rotations);
  angular_velocities_.upload(host_angular);
  props_.upload(host_props);

  std::array<float, HA_HAIR_PARAM_COUNT> params{};
  params[HA_PARAM_COUNT] = float(count);
  params[HA_PARAM_DT] = delta_time;
  params[HA_PARAM_INV_DT] = math::safe_rcp(delta_time);
  params[HA_PARAM_HAS_ACCEL] = has_external_acceleration_ ? 1.0f : 0.0f;
  params[HA_PARAM_COMPLIANCE] = math::safe_rcp(delta_time * delta_time);
  params[HA_PARAM_HAS_ROTATION] = has_rotation_ ? 1.0f : 0.0f;
  params_.upload(params);

  /* Rod stretch/shear. Every point of the chunk has a rest length and a compliance entry; the
   * entry of a curve's first point is unused (like on the CPU). */
  for (const HaHairRodStretchSet &set_in : stretch_sets) {
    if (int(set_in.rest_lengths.size()) != count || int(set_in.compliances.size()) != count ||
        set_in.curve_first.size() != set_in.curve_size.size())
    {
      last_error_ = "rod set size mismatch";
      return false;
    }
    const int curve_count = int(set_in.curve_first.size());
    if (curve_count <= 0) {
      continue;
    }
    auto set = std::make_unique<RodSetGpu>();
    set->curve_count = curve_count;
    set->point_count = count;
    if (!set->curve_first.ensure(curve_count) || !set->curve_size.ensure(curve_count) ||
        !set->data.ensure(int64_t(count) * 2) || !set->lambda.ensure(int64_t(count) * 6) ||
        !set->set_params.ensure(2))
    {
      last_error_ = "rod buffer allocation failed";
      return false;
    }
    Array<int> host_first(curve_count);
    Array<int> host_size(curve_count);
    Array<float> host_data(int64_t(count) * 2);
    for (const int curve_i : IndexRange(curve_count)) {
      host_first[curve_i] = set_in.curve_first[curve_i];
      host_size[curve_i] = set_in.curve_size[curve_i];
    }
    for (const int i : IndexRange(count)) {
      host_data[i * 2 + 0] = set_in.rest_lengths[i];
      host_data[i * 2 + 1] = set_in.compliances[i];
    }
    set->curve_first.upload(host_first);
    set->curve_size.upload(host_size);
    set->data.upload(host_data);
    set->lambda.clear_to_zero();
    const std::array<float, 2> host_set_params{float(curve_count), 0.0f};
    set->set_params.upload(host_set_params);
    rod_sets_.append(std::move(set));
    stretch_sets_host_.append(set_in);
  }

  /* Rod bend/twist. */
  for (const HaHairRodBendSet &set_in : bend_sets) {
    if (int(set_in.rest_rotations.size()) != count || int(set_in.compliances.size()) != count ||
        set_in.curve_first.size() != set_in.curve_size.size())
    {
      last_error_ = "bend set size mismatch";
      return false;
    }
    const int curve_count = int(set_in.curve_first.size());
    if (curve_count <= 0) {
      continue;
    }
    auto set = std::make_unique<BendSetGpu>();
    set->curve_count = curve_count;
    set->point_count = count;
    if (!set->curve_first.ensure(curve_count) || !set->curve_size.ensure(curve_count) ||
        !set->data.ensure(int64_t(count) * 5) || !set->lambda.ensure(int64_t(count) * 4) ||
        !set->set_params.ensure(2))
    {
      last_error_ = "bend buffer allocation failed";
      return false;
    }
    Array<int> host_first(curve_count);
    Array<int> host_size(curve_count);
    Array<float> host_data(int64_t(count) * 5);
    for (const int curve_i : IndexRange(curve_count)) {
      host_first[curve_i] = set_in.curve_first[curve_i];
      host_size[curve_i] = set_in.curve_size[curve_i];
    }
    for (const int i : IndexRange(count)) {
      const float4 rest = quaternion_to_float4(set_in.rest_rotations[i]);
      host_data[i * 5 + 0] = rest.x;
      host_data[i * 5 + 1] = rest.y;
      host_data[i * 5 + 2] = rest.z;
      host_data[i * 5 + 3] = rest.w;
      host_data[i * 5 + 4] = set_in.compliances[i];
    }
    set->curve_first.upload(host_first);
    set->curve_size.upload(host_size);
    set->data.upload(host_data);
    set->lambda.clear_to_zero();
    const std::array<float, 2> host_set_params{float(curve_count), 0.0f};
    set->set_params.upload(host_set_params);
    bend_sets_.append(std::move(set));
    bend_sets_host_.append(set_in);
  }

  /* Damping (velocity constraint sets, in the order the solver built them). */
  for (const HaHairVelocitySet &set_in : velocity_sets) {
    if (int(set_in.dampings.size()) != count) {
      last_error_ = "damping set size mismatch";
      return false;
    }
    auto set = std::make_unique<VelocitySetGpu>();
    set->angular = set_in.angular;
    set->point_count = count;
    if (!set->damping.ensure(count) || !set->lambda.ensure(count)) {
      last_error_ = "damping buffer allocation failed";
      return false;
    }
    set->damping.upload(set_in.dampings);
    set->lambda.clear_to_zero();
    velocity_sets_.append(std::move(set));
  }

  /* Pin positions. */
  for (const HaHairPinPositionSet &set_in : pin_position_sets) {
    const int pin_count = int(set_in.points.size());
    if (pin_count == 0) {
      continue;
    }
    if (int(set_in.targets.size()) != pin_count ||
        int(set_in.compliances.size()) != pin_count)
    {
      last_error_ = "pin set size mismatch";
      return false;
    }
    auto set = std::make_unique<PinSetGpu>();
    set->count = pin_count;
    if (!set->points.ensure(pin_count) || !set->target.ensure(int64_t(pin_count) * 3) ||
        !set->compliance.ensure(pin_count) || !set->lambda.ensure(pin_count) ||
        !set->set_params.ensure(2))
    {
      last_error_ = "pin buffer allocation failed";
      return false;
    }
    Array<float> host_targets(int64_t(pin_count) * 3);
    for (const int i : IndexRange(pin_count)) {
      host_targets[i * 3 + 0] = set_in.targets[i].x;
      host_targets[i * 3 + 1] = set_in.targets[i].y;
      host_targets[i * 3 + 2] = set_in.targets[i].z;
    }
    set->points.upload(set_in.points);
    set->target.upload(host_targets);
    set->compliance.upload(set_in.compliances);
    set->lambda.clear_to_zero();
    const std::array<float, 2> host_set_params{float(pin_count), 0.0f};
    set->set_params.upload(host_set_params);
    pin_position_sets_.append(std::move(set));
    pin_position_sets_host_.append(set_in);
  }

  /* Pin rotations. */
  for (const HaHairPinRotationSet &set_in : pin_rotation_sets) {
    const int pin_count = int(set_in.points.size());
    if (pin_count == 0) {
      continue;
    }
    if (int(set_in.targets.size()) != pin_count ||
        int(set_in.compliances.size()) != pin_count)
    {
      last_error_ = "pin rotation set size mismatch";
      return false;
    }
    auto set = std::make_unique<PinSetGpu>();
    set->angular = true;
    set->count = pin_count;
    if (!set->points.ensure(pin_count) || !set->target.ensure(int64_t(pin_count) * 4) ||
        !set->compliance.ensure(pin_count) || !set->lambda.ensure(int64_t(pin_count) * 4) ||
        !set->set_params.ensure(2))
    {
      last_error_ = "pin rotation buffer allocation failed";
      return false;
    }
    Array<float> host_targets(int64_t(pin_count) * 4);
    for (const int i : IndexRange(pin_count)) {
      const float4 target = quaternion_to_float4(set_in.targets[i]);
      host_targets[i * 4 + 0] = target.x;
      host_targets[i * 4 + 1] = target.y;
      host_targets[i * 4 + 2] = target.z;
      host_targets[i * 4 + 3] = target.w;
    }
    set->points.upload(set_in.points);
    set->target.upload(host_targets);
    set->compliance.upload(set_in.compliances);
    set->lambda.clear_to_zero();
    const std::array<float, 2> host_set_params{float(pin_count), 0.0f};
    set->set_params.upload(host_set_params);
    pin_rotation_sets_.append(std::move(set));
    pin_rotation_sets_host_.append(set_in);
  }

  if (!this->ensure_shaders()) {
    return false;
  }

  ready_ = true;
  return true;
}

bool HaXpbdHair::predict()
{
  if (!ready_) {
    last_error_ = "not bound";
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    last_error_ = "no GPU context";
    return false;
  }
  const std::string prelude(GLSL_HAIR_PRELUDE);
  gpu::Shader *copy_prev = ha_shader_get_source("hair|copy_prev",
                                                prelude + KERNEL_HAIR_COPY_PREV,
                                                as_span(COPY_PREV_RESOURCES),
                                                LOCAL_GROUP_SIZE);
  gpu::Shader *predict = ha_shader_get_source("hair|predict",
                                              prelude + KERNEL_HAIR_PREDICT,
                                              as_span(PREDICT_RESOURCES),
                                              LOCAL_GROUP_SIZE);
  if (copy_prev == nullptr || predict == nullptr) {
    last_error_ = "shader creation failed";
    return false;
  }

  /* The multipliers are reset once per substep, like `reset_forces` on the CPU. */
  for (const std::unique_ptr<RodSetGpu> &set : rod_sets_) {
    set->lambda.clear_to_zero();
  }
  for (const std::unique_ptr<BendSetGpu> &set : bend_sets_) {
    set->lambda.clear_to_zero();
  }
  for (const std::unique_ptr<VelocitySetGpu> &set : velocity_sets_) {
    set->lambda.clear_to_zero();
  }
  for (const std::unique_ptr<PinSetGpu> &set : pin_position_sets_) {
    set->lambda.clear_to_zero();
  }
  for (const std::unique_ptr<PinSetGpu> &set : pin_rotation_sets_) {
    set->lambda.clear_to_zero();
  }

  /* Pins are animated: the solver interpolated their targets for this substep. */
  for (const int set_i : pin_position_sets_.index_range()) {
    PinSetGpu &set = *pin_position_sets_[set_i];
    const Span<float3> targets = pin_position_sets_host_[set_i].targets;
    Array<float> host_targets(int64_t(set.count) * 3);
    for (const int i : IndexRange(set.count)) {
      host_targets[i * 3 + 0] = targets[i].x;
      host_targets[i * 3 + 1] = targets[i].y;
      host_targets[i * 3 + 2] = targets[i].z;
    }
    set.target.upload(host_targets);
  }
  for (const int set_i : pin_rotation_sets_.index_range()) {
    PinSetGpu &set = *pin_rotation_sets_[set_i];
    const Span<math::Quaternion> targets = pin_rotation_sets_host_[set_i].targets;
    Array<float> host_targets(int64_t(set.count) * 4);
    for (const int i : IndexRange(set.count)) {
      const float4 target = quaternion_to_float4(targets[i]);
      host_targets[i * 4 + 0] = target.x;
      host_targets[i * 4 + 1] = target.y;
      host_targets[i * 4 + 2] = target.z;
      host_targets[i * 4 + 3] = target.w;
    }
    set.target.upload(host_targets);
  }

  {
    std::array<gpu::StorageBuf *, 5> buffers{positions_.buf(),
                                             prev_positions_.buf(),
                                             rotations_.buf(),
                                             prev_rotations_.buf(),
                                             params_.buf()};
    dispatch_pass(copy_prev, buffer_span(buffers), point_count_);
  }
  {
    std::array<gpu::StorageBuf *, 8> buffers{positions_.buf(),
                                             prev_positions_.buf(),
                                             velocities_.buf(),
                                             rotations_.buf(),
                                             prev_rotations_.buf(),
                                             angular_velocities_.buf(),
                                             props_.buf(),
                                             params_.buf()};
    dispatch_pass(predict, buffer_span(buffers), point_count_);
  }
  stats_.passes += 2;
  return true;
}

bool HaXpbdHair::iterate()
{
  if (!ready_) {
    last_error_ = "not bound";
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    last_error_ = "no GPU context";
    return false;
  }
  const std::string prelude(GLSL_HAIR_PRELUDE);

  if (!rod_sets_.is_empty()) {
    gpu::Shader *stretch = ha_shader_get_source("hair|rod_stretch",
                                                prelude + KERNEL_HAIR_ROD_STRETCH,
                                                as_span(ROD_STRETCH_RESOURCES),
                                                LOCAL_GROUP_SIZE);
    if (stretch == nullptr) {
      last_error_ = "shader creation failed";
      return false;
    }
    for (const std::unique_ptr<RodSetGpu> &set : rod_sets_) {
      std::array<gpu::StorageBuf *, 9> buffers{positions_.buf(),
                                               rotations_.buf(),
                                               props_.buf(),
                                               set->curve_first.buf(),
                                               set->curve_size.buf(),
                                               set->data.buf(),
                                               set->lambda.buf(),
                                               set->set_params.buf(),
                                               params_.buf()};
      dispatch_pass(stretch, buffer_span(buffers), set->curve_count);
      stats_.passes++;
    }
  }
  if (!bend_sets_.is_empty()) {
    gpu::Shader *bend = ha_shader_get_source("hair|rod_bend",
                                             prelude + KERNEL_HAIR_ROD_BEND,
                                             as_span(ROD_BEND_RESOURCES),
                                             LOCAL_GROUP_SIZE);
    if (bend == nullptr) {
      last_error_ = "shader creation failed";
      return false;
    }
    for (const std::unique_ptr<BendSetGpu> &set : bend_sets_) {
      std::array<gpu::StorageBuf *, 8> buffers{rotations_.buf(),
                                               props_.buf(),
                                               set->curve_first.buf(),
                                               set->curve_size.buf(),
                                               set->data.buf(),
                                               set->lambda.buf(),
                                               set->set_params.buf(),
                                               params_.buf()};
      dispatch_pass(bend, buffer_span(buffers), set->curve_count);
      stats_.passes++;
    }
  }
  if (!pin_position_sets_.is_empty()) {
    gpu::Shader *pin = ha_shader_get_source("hair|pin_position",
                                            prelude + KERNEL_HAIR_PIN_POSITION,
                                            as_span(PIN_POSITION_RESOURCES),
                                            LOCAL_GROUP_SIZE);
    if (pin == nullptr) {
      last_error_ = "shader creation failed";
      return false;
    }
    for (const std::unique_ptr<PinSetGpu> &set : pin_position_sets_) {
      std::array<gpu::StorageBuf *, 8> buffers{positions_.buf(),
                                               props_.buf(),
                                               set->points.buf(),
                                               set->target.buf(),
                                               set->compliance.buf(),
                                               set->lambda.buf(),
                                               set->set_params.buf(),
                                               params_.buf()};
      dispatch_pass(pin, buffer_span(buffers), set->count);
      stats_.passes++;
    }
  }
  if (!pin_rotation_sets_.is_empty()) {
    gpu::Shader *pin = ha_shader_get_source("hair|pin_rotation",
                                            prelude + KERNEL_HAIR_PIN_ROTATION,
                                            as_span(PIN_ROTATION_RESOURCES),
                                            LOCAL_GROUP_SIZE);
    if (pin == nullptr) {
      last_error_ = "shader creation failed";
      return false;
    }
    for (const std::unique_ptr<PinSetGpu> &set : pin_rotation_sets_) {
      std::array<gpu::StorageBuf *, 8> buffers{rotations_.buf(),
                                               props_.buf(),
                                               set->points.buf(),
                                               set->target.buf(),
                                               set->compliance.buf(),
                                               set->lambda.buf(),
                                               set->set_params.buf(),
                                               params_.buf()};
      dispatch_pass(pin, buffer_span(buffers), set->count);
      stats_.passes++;
    }
  }
  return true;
}

bool HaXpbdHair::update_velocities()
{
  if (!ready_) {
    last_error_ = "not bound";
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    last_error_ = "no GPU context";
    return false;
  }
  gpu::Shader *update = ha_shader_get_source("hair|update_velocity",
                                             std::string(GLSL_HAIR_PRELUDE) +
                                                 KERNEL_HAIR_UPDATE_VELOCITY,
                                             as_span(UPDATE_VELOCITY_RESOURCES),
                                             LOCAL_GROUP_SIZE);
  if (update == nullptr) {
    last_error_ = "shader creation failed";
    return false;
  }
  std::array<gpu::StorageBuf *, 7> buffers{positions_.buf(),
                                           prev_positions_.buf(),
                                           velocities_.buf(),
                                           rotations_.buf(),
                                           prev_rotations_.buf(),
                                           angular_velocities_.buf(),
                                           params_.buf()};
  dispatch_pass(update, buffer_span(buffers), point_count_);
  stats_.passes++;
  return true;
}

bool HaXpbdHair::solve_velocity()
{
  if (!ready_) {
    last_error_ = "not bound";
    return false;
  }
  if (velocity_sets_.is_empty()) {
    return true;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    last_error_ = "no GPU context";
    return false;
  }
  const std::string prelude(GLSL_HAIR_PRELUDE);
  gpu::Shader *linear = ha_shader_get_source("hair|damping_linear",
                                             prelude + KERNEL_HAIR_DAMPING_LINEAR,
                                             as_span(DAMPING_LINEAR_RESOURCES),
                                             LOCAL_GROUP_SIZE);
  gpu::Shader *angular = nullptr;
  if (has_rotation_) {
    angular = ha_shader_get_source("hair|damping_angular",
                                   prelude + KERNEL_HAIR_DAMPING_ANGULAR,
                                   as_span(DAMPING_ANGULAR_RESOURCES),
                                   LOCAL_GROUP_SIZE);
  }
  if (linear == nullptr || (has_rotation_ && angular == nullptr)) {
    last_error_ = "shader creation failed";
    return false;
  }
  for (const std::unique_ptr<VelocitySetGpu> &set : velocity_sets_) {
    if (set->angular) {
      if (angular == nullptr) {
        continue;
      }
      std::array<gpu::StorageBuf *, 4> buffers{
          angular_velocities_.buf(), set->damping.buf(), set->lambda.buf(), params_.buf()};
      dispatch_pass(angular, buffer_span(buffers), set->point_count);
    }
    else {
      std::array<gpu::StorageBuf *, 4> buffers{
          velocities_.buf(), set->damping.buf(), set->lambda.buf(), params_.buf()};
      dispatch_pass(linear, buffer_span(buffers), set->point_count);
    }
    stats_.passes++;
  }
  return true;
}

bool HaXpbdHair::download(const MutableSpan<float3> r_positions,
                          const MutableSpan<float3> r_velocities,
                          const MutableSpan<math::Quaternion> r_rotations,
                          const MutableSpan<float3> r_angular_velocities)
{
  if (!ready_) {
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    return false;
  }
  const int count = std::min<int64_t>(r_positions.size(), point_count_);
  if (count <= 0) {
    return false;
  }
  Array<float> host_positions(int64_t(point_count_) * 3);
  positions_.download(host_positions);
  for (const int i : IndexRange(count)) {
    r_positions[i] = read_float3(host_positions.as_span(), i);
  }
  Array<float> host_velocities(int64_t(point_count_) * 3);
  velocities_.download(host_velocities);
  const int velocity_count = std::min<int64_t>(r_velocities.size(), point_count_);
  for (const int i : IndexRange(velocity_count)) {
    r_velocities[i] = read_float3(host_velocities.as_span(), i);
  }
  if (has_rotation_) {
    Array<float> host_rotations(int64_t(point_count_) * 4);
    rotations_.download(host_rotations);
    const int rotation_count = std::min<int64_t>(r_rotations.size(), point_count_);
    for (const int i : IndexRange(rotation_count)) {
      r_rotations[i] = math::Quaternion(host_rotations[i * 4 + 0],
                                        host_rotations[i * 4 + 1],
                                        host_rotations[i * 4 + 2],
                                        host_rotations[i * 4 + 3]);
    }
    Array<float> host_angular(int64_t(point_count_) * 3);
    angular_velocities_.download(host_angular);
    const int angular_count = std::min<int64_t>(r_angular_velocities.size(), point_count_);
    for (const int i : IndexRange(angular_count)) {
      r_angular_velocities[i] = read_float3(host_angular.as_span(), i);
    }
  }

  /* The multipliers are only valid for the last substep, which is why they are read back here
   * instead of inside every iteration. */
  if (stretch_sets_host_.size() == rod_sets_.size()) {
    for (const int set_i : rod_sets_.index_range()) {
      HaHairRodStretchSet &set_host = stretch_sets_host_[set_i];
      Array<float> host_lambda(int64_t(point_count_) * 6);
      rod_sets_[set_i]->lambda.download(host_lambda);
      const int lambda_count = std::min<int64_t>(set_host.lambdas_pos.size(), point_count_);
      const int rotation_count = std::min<int64_t>(set_host.lambdas_rot.size(), point_count_);
      for (const int i : IndexRange(lambda_count)) {
        const int b = i * 6;
        set_host.lambdas_pos[i] = float3(
            host_lambda[b + 0], host_lambda[b + 1], host_lambda[b + 2]);
      }
      for (const int i : IndexRange(rotation_count)) {
        const int b = i * 6;
        set_host.lambdas_rot[i] = float3(
            host_lambda[b + 3], host_lambda[b + 4], host_lambda[b + 5]);
      }
    }
  }
  if (bend_sets_host_.size() == bend_sets_.size()) {
    for (const int set_i : bend_sets_.index_range()) {
      HaHairRodBendSet &set_host = bend_sets_host_[set_i];
      Array<float> host_lambda(int64_t(point_count_) * 4);
      bend_sets_[set_i]->lambda.download(host_lambda);
      const int lambda_count = std::min<int64_t>(set_host.lambdas.size(), point_count_);
      for (const int i : IndexRange(lambda_count)) {
        set_host.lambdas[i] = float4(host_lambda[i * 4 + 0],
                                     host_lambda[i * 4 + 1],
                                     host_lambda[i * 4 + 2],
                                     host_lambda[i * 4 + 3]);
      }
    }
  }
  if (pin_position_sets_host_.size() == pin_position_sets_.size()) {
    for (const int set_i : pin_position_sets_.index_range()) {
      HaHairPinPositionSet &set_host = pin_position_sets_host_[set_i];
      const PinSetGpu &set = *pin_position_sets_[set_i];
      Array<float> host_lambda(set.count);
      set.lambda.download(host_lambda);
      const int lambda_count = std::min<int64_t>(set_host.lambdas.size(), set.count);
      for (const int i : IndexRange(lambda_count)) {
        set_host.lambdas[i] = host_lambda[i];
      }
    }
  }
  if (pin_rotation_sets_host_.size() == pin_rotation_sets_.size()) {
    for (const int set_i : pin_rotation_sets_.index_range()) {
      HaHairPinRotationSet &set_host = pin_rotation_sets_host_[set_i];
      const PinSetGpu &set = *pin_rotation_sets_[set_i];
      Array<float> host_lambda(int64_t(set.count) * 4);
      set.lambda.download(host_lambda);
      const int lambda_count = std::min<int64_t>(set_host.lambdas.size(), set.count);
      for (const int i : IndexRange(lambda_count)) {
        set_host.lambdas[i] = float4(host_lambda[i * 4 + 0],
                                     host_lambda[i * 4 + 1],
                                     host_lambda[i * 4 + 2],
                                     host_lambda[i * 4 + 3]);
      }
    }
  }
  return true;
}

/** \} */

}  // namespace blender::geometry::ha
