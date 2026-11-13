/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Device (compute backend) MPM substep for Textura. See `GEO_textura_device.hh` for the scope.
 */

#include "GEO_textura_device.hh"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "BLI_math_matrix.hh"
#include "BLI_math_vector.hh"
#include "BLI_string_ref.hh"

#include "GPU_init_exit.hh"
#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_storage_buffer.hh"

#include "GEO_ha_context.hh"

namespace blender::geometry::textura {

using namespace blender::geometry::ha;

namespace {

/* -------------------------------------------------------------------- */
/** \name Parameter layout (must match the GLSL defines below)
 * \{ */

enum ParamIndex : int {
  P_COUNT = 0,
  P_CELLS,
  P_DIM_X,
  P_DIM_Y,
  P_DIM_Z,
  P_ORIGIN_X,
  P_ORIGIN_Y,
  P_ORIGIN_Z,
  P_DX,
  P_DT,
  P_GRAVITY_X,
  P_GRAVITY_Y,
  P_GRAVITY_Z,
  P_VELOCITY_DAMPING,
  P_BOUNDARY_FRICTION,
  P_CONTACT_SKIN,
  P_BOX_COUNT,
  P_SDF_ENABLED,
  P_SDF_INVERTED,
  P_SDF_FRICTION,
  P_SDF_ORIGIN_X,
  P_SDF_ORIGIN_Y,
  P_SDF_ORIGIN_Z,
  P_SDF_SPACING,
  P_SDF_DIM_X,
  P_SDF_DIM_Y,
  P_SDF_DIM_Z,
  P_PARAM_COUNT,
};

/** Interleaved per particle scalars. */
enum PropIndex : int {
  PROP_MASS = 0,
  PROP_VOLUME,
  PROP_RADIUS,
  PROP_BULK,
  PROP_TENSION,
  PROP_DAMPING,
  PROP_VISCOSITY,
  PROP_COUNT,
};

/**
 * Floats per particle in the `props[]` buffer. Kept separate from #PROP_COUNT (and padded to
 * a power of two) so the host packing and the GLSL `props[p * HA_PROP_STRIDE + ...]` addressing
 * can never drift apart again; they must be equal.
 */
constexpr int PROP_STRIDE = 8;
static_assert(PROP_STRIDE == 8, "HA_PROP_STRIDE in GLSL_PRELUDE must match PROP_STRIDE");
static_assert(PROP_STRIDE >= PROP_COUNT, "PROP_STRIDE has to hold every PropIndex");

/** Floats per box collider: matrix, inverse, local bounds, friction, invert, scale. */
constexpr int BOX_FLOATS = 48;
constexpr int MAX_BOXES = 8;
constexpr int LOCAL_GROUP_SIZE = 64;

/* -------------------------------------------------------------------- */
/** \name Kernels
 * \{ */

const char *GLSL_PRELUDE =
    "#define P_COUNT 0\n"
    "#define P_CELLS 1\n"
    "#define P_DIM_X 2\n"
    "#define P_DIM_Y 3\n"
    "#define P_DIM_Z 4\n"
    "#define P_ORIGIN_X 5\n"
    "#define P_ORIGIN_Y 6\n"
    "#define P_ORIGIN_Z 7\n"
    "#define P_DX 8\n"
    "#define P_DT 9\n"
    "#define P_GRAVITY_X 10\n"
    "#define P_GRAVITY_Y 11\n"
    "#define P_GRAVITY_Z 12\n"
    "#define P_VELOCITY_DAMPING 13\n"
    "#define P_BOUNDARY_FRICTION 14\n"
    "#define P_CONTACT_SKIN 15\n"
    "#define P_BOX_COUNT 16\n"
    "#define P_SDF_ENABLED 17\n"
    "#define P_SDF_INVERTED 18\n"
    "#define P_SDF_FRICTION 19\n"
    "#define P_SDF_ORIGIN_X 20\n"
    "#define P_SDF_ORIGIN_Y 21\n"
    "#define P_SDF_ORIGIN_Z 22\n"
    "#define P_SDF_SPACING 23\n"
    "#define P_SDF_DIM_X 24\n"
    "#define P_SDF_DIM_Y 25\n"
    "#define P_SDF_DIM_Z 26\n"
    "#define HA_BOX_STRIDE 48\n"
    "#define PROP_MASS 0\n"
    "#define PROP_VOLUME 1\n"
    "#define PROP_RADIUS 2\n"
    "#define PROP_BULK 3\n"
    "#define PROP_TENSION 4\n"
    "#define PROP_DAMPING 5\n"
    "#define PROP_VISCOSITY 6\n"
    "#define HA_PROP_STRIDE 8\n"
    "#define HA_FIXED 262144.0\n"
    "#define HA_INV_FIXED (1.0 / 262144.0)\n"
    "\n"
    "vec3 ha_gravity()\n"
    "{\n"
    "  return vec3(params[P_GRAVITY_X], params[P_GRAVITY_Y], params[P_GRAVITY_Z]);\n"
    "}\n"
    "\n"
    "struct HaWeights\n"
    "{\n"
    "  int base;\n"
    "  float f;\n"
    "  vec3 w;\n"
    "};\n"
    "\n"
    "HaWeights ha_weights(float p, float origin, float dx)\n"
    "{\n"
    "  HaWeights r;\n"
    "  float gp = (p - origin) / dx;\n"
    "  r.base = int(floor(gp - 0.5));\n"
    "  r.f = gp - float(r.base);\n"
    "  float om = 1.5 - r.f;\n"
    "  r.w.x = 0.5 * om * om;\n"
    "  r.w.y = 0.75 - (r.f - 1.0) * (r.f - 1.0);\n"
    "  r.w.z = 0.5 * (r.f - 0.5) * (r.f - 0.5);\n"
    "  return r;\n"
    "}\n"
    "\n"
    "vec3 ha_wall_contact(vec3 v, vec3 n, float friction, float dt)\n"
    "{\n"
    "  float vn = dot(v, n);\n"
    "  if (vn < 0.0) { v -= n * vn; }\n"
    "  vec3 tangent = v - n * dot(v, n);\n"
    "  float decay = clamp(friction * 25.0 * dt, 0.0, 1.0);\n"
    "  return n * dot(v, n) + tangent * (1.0 - decay);\n"
    "}\n"
    "\n"
    "float ha_sdf_at(ivec3 c)\n"
    "{\n"
    "  ivec3 dims = ivec3(int(params[P_SDF_DIM_X]), int(params[P_SDF_DIM_Y]),\n"
    "                     int(params[P_SDF_DIM_Z]));\n"
    "  c = clamp(c, ivec3(0), dims - 1);\n"
    "  return sdf_data[(c.z * dims.y + c.y) * dims.x + c.x];\n"
    "}\n"
    "\n"
    "float ha_sdf_sample(vec3 p)\n"
    "{\n"
    "  vec3 origin = vec3(params[P_SDF_ORIGIN_X], params[P_SDF_ORIGIN_Y],\n"
    "                     params[P_SDF_ORIGIN_Z]);\n"
    "  float spacing = params[P_SDF_SPACING];\n"
    "  vec3 local = (p - origin) / spacing - vec3(0.5);\n"
    "  ivec3 base = ivec3(floor(local));\n"
    "  vec3 f = local - vec3(base);\n"
    "  float c000 = ha_sdf_at(base + ivec3(0, 0, 0));\n"
    "  float c100 = ha_sdf_at(base + ivec3(1, 0, 0));\n"
    "  float c010 = ha_sdf_at(base + ivec3(0, 1, 0));\n"
    "  float c110 = ha_sdf_at(base + ivec3(1, 1, 0));\n"
    "  float c001 = ha_sdf_at(base + ivec3(0, 0, 1));\n"
    "  float c101 = ha_sdf_at(base + ivec3(1, 0, 1));\n"
    "  float c011 = ha_sdf_at(base + ivec3(0, 1, 1));\n"
    "  float c111 = ha_sdf_at(base + ivec3(1, 1, 1));\n"
    "  float c00 = mix(c000, c100, f.x);\n"
    "  float c10 = mix(c010, c110, f.x);\n"
    "  float c01 = mix(c001, c101, f.x);\n"
    "  float c11 = mix(c011, c111, f.x);\n"
    "  float c0 = mix(c00, c10, f.y);\n"
    "  float c1 = mix(c01, c11, f.y);\n"
    "  return mix(c0, c1, f.z);\n"
    "}\n"
    "\n"
    "void ha_sample_solids(vec3 p, out float distance, out vec3 normal, out float friction)\n"
    "{\n"
    "  distance = 1e30;\n"
    "  normal = vec3(0.0, 0.0, 1.0);\n"
    "  friction = 0.0;\n"
    "  int box_count = int(params[P_BOX_COUNT]);\n"
    "  for (int b = 0; b < box_count; b++) {\n"
    "    int base = b * HA_BOX_STRIDE;\n"
    "    mat4 m = mat4(vec4(box_data[base + 0], box_data[base + 1], box_data[base + 2],\n"
    "                       box_data[base + 3]),\n"
    "                  vec4(box_data[base + 4], box_data[base + 5], box_data[base + 6],\n"
    "                       box_data[base + 7]),\n"
    "                  vec4(box_data[base + 8], box_data[base + 9], box_data[base + 10],\n"
    "                       box_data[base + 11]),\n"
    "                  vec4(box_data[base + 12], box_data[base + 13], box_data[base + 14],\n"
    "                       box_data[base + 15]));\n"
    "    mat4 inv = mat4(vec4(box_data[base + 16], box_data[base + 17], box_data[base + 18],\n"
    "                         box_data[base + 19]),\n"
    "                    vec4(box_data[base + 20], box_data[base + 21], box_data[base + 22],\n"
    "                         box_data[base + 23]),\n"
    "                    vec4(box_data[base + 24], box_data[base + 25], box_data[base + 26],\n"
    "                         box_data[base + 27]),\n"
    "                    vec4(box_data[base + 28], box_data[base + 29], box_data[base + 30],\n"
    "                         box_data[base + 31]));\n"
    "    vec3 lp = (inv * vec4(p, 1.0)).xyz;\n"
    "    vec3 lmin = vec3(box_data[base + 32], box_data[base + 33], box_data[base + 34]);\n"
    "    vec3 lmax = vec3(box_data[base + 35], box_data[base + 36], box_data[base + 37]);\n"
    "    vec3 center = (lmin + lmax) * 0.5;\n"
    "    vec3 half_size = (lmax - lmin) * 0.5;\n"
    "    vec3 d = lp - center;\n"
    "    vec3 q = abs(d) - half_size;\n"
    "    float bd;\n"
    "    vec3 bn;\n"
    "    if (q.x <= 0.0 && q.y <= 0.0 && q.z <= 0.0) {\n"
    "      int axis = 0;\n"
    "      if (q.y > q.x && q.y >= q.z) { axis = 1; }\n"
    "      else if (q.z > q.x && q.z > q.y) { axis = 2; }\n"
    "      bd = q[axis];\n"
    "      bn = vec3(0.0);\n"
    "      bn[axis] = d[axis] >= 0.0 ? 1.0 : -1.0;\n"
    "    }\n"
    "    else {\n"
    "      vec3 pq = max(q, vec3(0.0));\n"
    "      float out_len = length(pq);\n"
    "      bn = out_len > 1e-8\n"
    "               ? normalize(vec3(d.x >= 0.0 ? pq.x : -pq.x, d.y >= 0.0 ? pq.y : -pq.y,\n"
    "                                d.z >= 0.0 ? pq.z : -pq.z))\n"
    "               : vec3(0.0, 0.0, 1.0);\n"
    "      bd = out_len;\n"
    "    }\n"
    "    bd *= box_data[base + 40];\n"
    "    bn = normalize((m * vec4(bn, 0.0)).xyz);\n"
    "    float bf = box_data[base + 38];\n"
    "    if (box_data[base + 39] > 0.5 && bd < 0.0) {\n"
    "      bd = -bd;\n"
    "      bn = -bn;\n"
    "    }\n"
    "    if (bd < distance) {\n"
    "      distance = bd;\n"
    "      normal = bn;\n"
    "      friction = bf;\n"
    "    }\n"
    "  }\n"
    "  if (params[P_SDF_ENABLED] > 0.5) {\n"
    "    vec3 so = vec3(params[P_SDF_ORIGIN_X], params[P_SDF_ORIGIN_Y], params[P_SDF_ORIGIN_Z]);\n"
    "    float sp = params[P_SDF_SPACING];\n"
    "    vec3 sd = vec3(params[P_SDF_DIM_X], params[P_SDF_DIM_Y], params[P_SDF_DIM_Z]);\n"
    "    vec3 sb_max = so + (sd - vec3(1.0)) * sp;\n"
    "    float margin = 8.0 * sp;\n"
    "    if (p.x >= so.x - margin && p.x <= sb_max.x + margin && p.y >= so.y - margin &&\n"
    "        p.y <= sb_max.y + margin && p.z >= so.z - margin && p.z <= sb_max.z + margin)\n"
    "    {\n"
    "      float signed_d = ha_sdf_sample(p);\n"
    "      float sd_abs = params[P_SDF_INVERTED] > 0.5 ? abs(signed_d) : signed_d;\n"
    "      if (sd_abs < distance) {\n"
    "        distance = sd_abs;\n"
    "        friction = params[P_SDF_FRICTION];\n"
    "        float eps = params[P_DX] * 0.5;\n"
    "        vec3 gradient = vec3(ha_sdf_sample(p + vec3(eps, 0.0, 0.0)) -\n"
    "                                 ha_sdf_sample(p - vec3(eps, 0.0, 0.0)),\n"
    "                             ha_sdf_sample(p + vec3(0.0, eps, 0.0)) -\n"
    "                                 ha_sdf_sample(p - vec3(0.0, eps, 0.0)),\n"
    "                             ha_sdf_sample(p + vec3(0.0, 0.0, eps)) -\n"
    "                                 ha_sdf_sample(p - vec3(0.0, 0.0, eps)));\n"
    "        float gradient_len = length(gradient);\n"
    "        if (gradient_len > 1e-8) {\n"
    "          normal = gradient / gradient_len;\n"
    "          if (params[P_SDF_INVERTED] > 0.5 && signed_d < 0.0) { normal = -normal; }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n"
    "\n";

const char *KERNEL_CLEAR =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(params[P_CELLS])) { return; }\n"
    "  gmass[i] = 0;\n"
    "  gmom[i * 3 + 0] = 0;\n"
    "  gmom[i * 3 + 1] = 0;\n"
    "  gmom[i * 3 + 2] = 0;\n"
    "  gvel[i * 3 + 0] = 0.0;\n"
    "  gvel[i * 3 + 1] = 0.0;\n"
    "  gvel[i * 3 + 2] = 0.0;\n"
    "}\n";

const char *KERNEL_P2G =
    "void main()\n"
    "{\n"
    "  int p = int(gl_GlobalInvocationID.x);\n"
    "  if (p >= int(params[P_COUNT])) { return; }\n"
    "  int dims_x = int(params[P_DIM_X]);\n"
    "  int dims_y = int(params[P_DIM_Y]);\n"
    "  int dims_z = int(params[P_DIM_Z]);\n"
    "  float dx = params[P_DX];\n"
    "  float dt = params[P_DT];\n"
    "  vec3 origin = vec3(params[P_ORIGIN_X], params[P_ORIGIN_Y], params[P_ORIGIN_Z]);\n"
    "  vec3 x = vec3(pos[p * 3 + 0], pos[p * 3 + 1], pos[p * 3 + 2]);\n"
    "  float mass = props[p * HA_PROP_STRIDE + PROP_MASS];\n"
    "  float volume = props[p * HA_PROP_STRIDE + PROP_VOLUME];\n"
    "  float bulk = props[p * HA_PROP_STRIDE + PROP_BULK];\n"
    "  float tension = props[p * HA_PROP_STRIDE + PROP_TENSION];\n"
    "  mat3 f = mat3(vec3(deform[p * 9 + 0], deform[p * 9 + 3], deform[p * 9 + 6]),\n"
    "                vec3(deform[p * 9 + 1], deform[p * 9 + 4], deform[p * 9 + 7]),\n"
    "                vec3(deform[p * 9 + 2], deform[p * 9 + 5], deform[p * 9 + 8]));\n"
    "  mat3 c = mat3(vec3(affine[p * 9 + 0], affine[p * 9 + 3], affine[p * 9 + 6]),\n"
    "                vec3(affine[p * 9 + 1], affine[p * 9 + 4], affine[p * 9 + 7]),\n"
    "                vec3(affine[p * 9 + 2], affine[p * 9 + 5], affine[p * 9 + 8]));\n"
    "  float j = determinant(f);\n"
    "  if (!(abs(j) > 1e-8)) { j = 1.0; }\n"
    "  float pressure = bulk * (j - 1.0);\n"
    "  if (pressure > 0.0) { pressure *= clamp(tension, 0.0, 1.0); }\n"
    "  float stress_scale = -dt * 4.0 / (dx * dx);\n"
    "  vec3 a0 = vec3(pressure, 0.0, 0.0) * (stress_scale * volume) + c[0] * mass;\n"
    "  vec3 a1 = vec3(0.0, pressure, 0.0) * (stress_scale * volume) + c[1] * mass;\n"
    "  vec3 a2 = vec3(0.0, 0.0, pressure) * (stress_scale * volume) + c[2] * mass;\n"
    "  vec3 velocity = vec3(vel[p * 3 + 0], vel[p * 3 + 1], vel[p * 3 + 2]);\n"
    "  int bx;\n"
    "  int by;\n"
    "  int bz;\n"
    "  float fx;\n"
    "  float fy;\n"
    "  float fz;\n"
    "  float wx[3];\n"
    "  float wy[3];\n"
    "  float wz[3];\n"
    "  HaWeights wxs = ha_weights(x.x, origin.x, dx);\n"
    "  HaWeights wys = ha_weights(x.y, origin.y, dx);\n"
    "  HaWeights wzs = ha_weights(x.z, origin.z, dx);\n"
    "  bx = wxs.base;\n"
    "  by = wys.base;\n"
    "  bz = wzs.base;\n"
    "  fx = wxs.f;\n"
    "  fy = wys.f;\n"
    "  fz = wzs.f;\n"
    "  wx[0] = wxs.w.x; wx[1] = wxs.w.y; wx[2] = wxs.w.z;\n"
    "  wy[0] = wys.w.x; wy[1] = wys.w.y; wy[2] = wys.w.z;\n"
    "  wz[0] = wzs.w.x; wz[1] = wzs.w.y; wz[2] = wzs.w.z;\n"
    "  vec3 momentum_base = velocity * mass - a0 * (fx * dx) - a1 * (fy * dx) - a2 * (fz * dx);\n"
    "  for (int k = 0; k < 3; k++) {\n"
    "    float wzk = wz[k];\n"
    "    if (wzk <= 0.0) { continue; }\n"
    "    int z = bz + k;\n"
    "    if (z < 0 || z >= dims_z) { continue; }\n"
    "    for (int jj = 0; jj < 3; jj++) {\n"
    "      float wzy = wzk * wy[jj];\n"
    "      if (wzy <= 0.0) { continue; }\n"
    "      int y = by + jj;\n"
    "      if (y < 0 || y >= dims_y) { continue; }\n"
    "      for (int i = 0; i < 3; i++) {\n"
    "        float w = wzy * wx[i];\n"
    "        if (w <= 0.0) { continue; }\n"
    "        int xx = bx + i;\n"
    "        if (xx < 0 || xx >= dims_x) { continue; }\n"
    "        int index = (z * dims_y + y) * dims_x + xx;\n"
    "        atomicAdd(gmass[index], int(round(w * mass * HA_FIXED)));\n"
    "        vec3 m = momentum_base + a0 * (float(i) * dx) + a1 * (float(jj) * dx) +\n"
    "                 a2 * (float(k) * dx);\n"
    "        atomicAdd(gmom[index * 3 + 0], int(round(m.x * w * HA_FIXED)));\n"
    "        atomicAdd(gmom[index * 3 + 1], int(round(m.y * w * HA_FIXED)));\n"
    "        atomicAdd(gmom[index * 3 + 2], int(round(m.z * w * HA_FIXED)));\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n";

const char *KERNEL_GRID =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= int(params[P_CELLS])) { return; }\n"
    "  int dims_x = int(params[P_DIM_X]);\n"
    "  int dims_y = int(params[P_DIM_Y]);\n"
    "  int dims_z = int(params[P_DIM_Z]);\n"
    "  float dx = params[P_DX];\n"
    "  float dt = params[P_DT];\n"
    "  vec3 origin = vec3(params[P_ORIGIN_X], params[P_ORIGIN_Y], params[P_ORIGIN_Z]);\n"
    "  float mass = float(gmass[index]) * HA_INV_FIXED;\n"
    "  if (mass <= 1e-12) {\n"
    "    gvel[index * 3 + 0] = 0.0;\n"
    "    gvel[index * 3 + 1] = 0.0;\n"
    "    gvel[index * 3 + 2] = 0.0;\n"
    "    return;\n"
    "  }\n"
    "  vec3 v = vec3(float(gmom[index * 3 + 0]), float(gmom[index * 3 + 1]),\n"
    "                float(gmom[index * 3 + 2])) *\n"
    "           HA_INV_FIXED / mass;\n"
    "  v += ha_gravity() * dt;\n"
    "  v *= max(0.0, 1.0 - params[P_VELOCITY_DAMPING] * dt);\n"
    "  int x = index % dims_x;\n"
    "  int y = (index / dims_x) % dims_y;\n"
    "  int z = index / (dims_x * dims_y);\n"
    "  if ((x == 0 && v.x < 0.0) || (x == dims_x - 1 && v.x > 0.0)) { v.x = 0.0; }\n"
    "  if ((y == 0 && v.y < 0.0) || (y == dims_y - 1 && v.y > 0.0)) { v.y = 0.0; }\n"
    "  if ((z == 0 && v.z < 0.0) || (z == dims_z - 1 && v.z > 0.0)) { v.z = 0.0; }\n"
    "  vec3 world = origin + vec3(float(x), float(y), float(z)) * dx;\n"
    "  float distance;\n"
    "  vec3 normal;\n"
    "  float friction;\n"
    "  ha_sample_solids(world, distance, normal, friction);\n"
    "  if (distance < 0.0) {\n"
    "    v = vec3(0.0);\n"
    "  }\n"
    "  else if (distance < 1.5 * dx) {\n"
    "    float f = clamp(params[P_BOUNDARY_FRICTION] * 2.0 * friction, 0.0, 1.0);\n"
    "    v = ha_wall_contact(v, normal, f, dt);\n"
    "  }\n"
    "  gvel[index * 3 + 0] = v.x;\n"
    "  gvel[index * 3 + 1] = v.y;\n"
    "  gvel[index * 3 + 2] = v.z;\n"
    "}\n";

const char *KERNEL_G2P =
    "void main()\n"
    "{\n"
    "  int p = int(gl_GlobalInvocationID.x);\n"
    "  if (p >= int(params[P_COUNT])) { return; }\n"
    "  int dims_x = int(params[P_DIM_X]);\n"
    "  int dims_y = int(params[P_DIM_Y]);\n"
    "  int dims_z = int(params[P_DIM_Z]);\n"
    "  float dx = params[P_DX];\n"
    "  float dt = params[P_DT];\n"
    "  vec3 origin = vec3(params[P_ORIGIN_X], params[P_ORIGIN_Y], params[P_ORIGIN_Z]);\n"
    "  vec3 x = vec3(pos[p * 3 + 0], pos[p * 3 + 1], pos[p * 3 + 2]);\n"
    "  int bx;\n"
    "  int by;\n"
    "  int bz;\n"
    "  float fx;\n"
    "  float fy;\n"
    "  float fz;\n"
    "  float wx[3];\n"
    "  float wy[3];\n"
    "  float wz[3];\n"
    "  HaWeights wxs = ha_weights(x.x, origin.x, dx);\n"
    "  HaWeights wys = ha_weights(x.y, origin.y, dx);\n"
    "  HaWeights wzs = ha_weights(x.z, origin.z, dx);\n"
    "  bx = wxs.base;\n"
    "  by = wys.base;\n"
    "  bz = wzs.base;\n"
    "  fx = wxs.f;\n"
    "  fy = wys.f;\n"
    "  fz = wzs.f;\n"
    "  wx[0] = wxs.w.x; wx[1] = wxs.w.y; wx[2] = wxs.w.z;\n"
    "  wy[0] = wys.w.x; wy[1] = wys.w.y; wy[2] = wys.w.z;\n"
    "  wz[0] = wzs.w.x; wz[1] = wzs.w.y; wz[2] = wzs.w.z;\n"
    "  vec3 velocity = vec3(0.0);\n"
    "  mat3 c = mat3(0.0);\n"
    "  float inv_dx = 1.0 / dx;\n"
    "  for (int k = 0; k < 3; k++) {\n"
    "    int z = bz + k;\n"
    "    if (z < 0 || z >= dims_z) { continue; }\n"
    "    for (int jj = 0; jj < 3; jj++) {\n"
    "      int y = by + jj;\n"
    "      if (y < 0 || y >= dims_y) { continue; }\n"
    "      for (int i = 0; i < 3; i++) {\n"
    "        int xx = bx + i;\n"
    "        if (xx < 0 || xx >= dims_x) { continue; }\n"
    "        float w = wx[i] * wy[jj] * wz[k];\n"
    "        if (w <= 0.0) { continue; }\n"
    "        int index = (z * dims_y + y) * dims_x + xx;\n"
    "        vec3 gv = vec3(gvel[index * 3 + 0], gvel[index * 3 + 1], gvel[index * 3 + 2]);\n"
    "        velocity += gv * w;\n"
    "        vec3 dpos = (vec3(float(i), float(jj), float(k)) - vec3(fx, fy, fz)) * dx;\n"
    "        float factor = 4.0 * inv_dx * inv_dx * w;\n"
    "        c[0] += factor * gv * dpos.x;\n"
    "        c[1] += factor * gv * dpos.y;\n"
    "        c[2] += factor * gv * dpos.z;\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  float damping = props[p * HA_PROP_STRIDE + PROP_DAMPING];\n"
    "  float viscosity = props[p * HA_PROP_STRIDE + PROP_VISCOSITY];\n"
    "  velocity *= max(0.0, 1.0 - damping * dt);\n"
    "  velocity *= max(0.0, 1.0 - viscosity * dt);\n"
    "  vel[p * 3 + 0] = velocity.x;\n"
    "  vel[p * 3 + 1] = velocity.y;\n"
    "  vel[p * 3 + 2] = velocity.z;\n"
    "  affine[p * 9 + 0] = c[0].x;\n"
    "  affine[p * 9 + 1] = c[1].x;\n"
    "  affine[p * 9 + 2] = c[2].x;\n"
    "  affine[p * 9 + 3] = c[0].y;\n"
    "  affine[p * 9 + 4] = c[1].y;\n"
    "  affine[p * 9 + 5] = c[2].y;\n"
    "  affine[p * 9 + 6] = c[0].z;\n"
    "  affine[p * 9 + 7] = c[1].z;\n"
    "  affine[p * 9 + 8] = c[2].z;\n"
    "  mat3 f = mat3(vec3(deform[p * 9 + 0], deform[p * 9 + 3], deform[p * 9 + 6]),\n"
    "                vec3(deform[p * 9 + 1], deform[p * 9 + 4], deform[p * 9 + 7]),\n"
    "                vec3(deform[p * 9 + 2], deform[p * 9 + 5], deform[p * 9 + 8]));\n"
    "  mat3 new_f = f + dt * (c * f);\n"
    "  float j = determinant(new_f);\n"
    "  if (!(abs(j) > 1e-8)) {\n"
    "    new_f = mat3(1.0);\n"
    "    j = 1.0;\n"
    "  }\n"
    "  float target = clamp(j, 0.25, 4.0);\n"
    "  new_f *= pow(target / j, 1.0 / 3.0);\n"
    "  deform[p * 9 + 0] = new_f[0].x;\n"
    "  deform[p * 9 + 1] = new_f[1].x;\n"
    "  deform[p * 9 + 2] = new_f[2].x;\n"
    "  deform[p * 9 + 3] = new_f[0].y;\n"
    "  deform[p * 9 + 4] = new_f[1].y;\n"
    "  deform[p * 9 + 5] = new_f[2].y;\n"
    "  deform[p * 9 + 6] = new_f[0].z;\n"
    "  deform[p * 9 + 7] = new_f[1].z;\n"
    "  deform[p * 9 + 8] = new_f[2].z;\n"
    "}\n";

const char *KERNEL_ADVECT =
    "void main()\n"
    "{\n"
    "  int p = int(gl_GlobalInvocationID.x);\n"
    "  if (p >= int(params[P_COUNT])) { return; }\n"
    "  int dims_x = int(params[P_DIM_X]);\n"
    "  int dims_y = int(params[P_DIM_Y]);\n"
    "  int dims_z = int(params[P_DIM_Z]);\n"
    "  float dx = params[P_DX];\n"
    "  float dt = params[P_DT];\n"
    "  vec3 origin = vec3(params[P_ORIGIN_X], params[P_ORIGIN_Y], params[P_ORIGIN_Z]);\n"
    "  vec3 domain_max = origin + vec3(float(dims_x - 1), float(dims_y - 1),\n"
    "                                  float(dims_z - 1)) * dx;\n"
    "  vec3 position = vec3(pos[p * 3 + 0], pos[p * 3 + 1], pos[p * 3 + 2]);\n"
    "  vec3 velocity = vec3(vel[p * 3 + 0], vel[p * 3 + 1], vel[p * 3 + 2]);\n"
    "  position += velocity * dt;\n"
    "  float radius = props[p * HA_PROP_STRIDE + PROP_RADIUS];\n"
    "  float distance;\n"
    "  vec3 normal;\n"
    "  float friction;\n"
    "  ha_sample_solids(position, distance, normal, friction);\n"
    "  float skin = max(params[P_CONTACT_SKIN], 0.0) * dx;\n"
    "  float contact_friction = clamp(params[P_BOUNDARY_FRICTION] * 2.0 * friction, 0.0, 1.0);\n"
    "  if (distance < radius + skin) {\n"
    "    bool touching = distance < radius;\n"
    "    position += normal * (radius - distance);\n"
    "    if (touching) { velocity = ha_wall_contact(velocity, normal, contact_friction, dt); }\n"
    "  }\n"
    "  position = clamp(position, origin, domain_max);\n"
    "  pos[p * 3 + 0] = position.x;\n"
    "  pos[p * 3 + 1] = position.y;\n"
    "  pos[p * 3 + 2] = position.z;\n"
    "  vel[p * 3 + 0] = velocity.x;\n"
    "  vel[p * 3 + 1] = velocity.y;\n"
    "  vel[p * 3 + 2] = velocity.z;\n"
    "}\n";

/** \} */

/* -------------------------------------------------------------------- */
/** \name Shader resources
 * \{ */

template<size_t N> Span<const HaShaderResource> as_span(const std::array<HaShaderResource, N> &r)
{
  return Span<const HaShaderResource>(r.data(), int64_t(r.size()));
}

template<size_t N> Span<gpu::StorageBuf *> buffer_span(std::array<gpu::StorageBuf *, N> &r)
{
  return Span<gpu::StorageBuf *>(r.data(), int64_t(r.size()));
}

const std::array<HaShaderResource, 4> CLEAR_RESOURCES{{
    {0, HaShaderQualifier::Read, "float", "params[]"},
    {1, HaShaderQualifier::ReadWrite, "int", "gmass[]"},
    {2, HaShaderQualifier::ReadWrite, "int", "gmom[]"},
    {3, HaShaderQualifier::ReadWrite, "float", "gvel[]"},
}};

const std::array<HaShaderResource, 8> P2G_RESOURCES{{
    {0, HaShaderQualifier::Read, "float", "params[]"},
    {1, HaShaderQualifier::Read, "float", "pos[]"},
    {2, HaShaderQualifier::Read, "float", "vel[]"},
    {3, HaShaderQualifier::Read, "float", "affine[]"},
    {4, HaShaderQualifier::Read, "float", "deform[]"},
    {5, HaShaderQualifier::Read, "float", "props[]"},
    {6, HaShaderQualifier::ReadWrite, "int", "gmass[]"},
    {7, HaShaderQualifier::ReadWrite, "int", "gmom[]"},
}};

const std::array<HaShaderResource, 6> GRID_RESOURCES{{
    {0, HaShaderQualifier::Read, "float", "params[]"},
    {1, HaShaderQualifier::Read, "int", "gmass[]"},
    {2, HaShaderQualifier::Read, "int", "gmom[]"},
    {3, HaShaderQualifier::Write, "float", "gvel[]"},
    {4, HaShaderQualifier::Read, "float", "box_data[]"},
    {5, HaShaderQualifier::Read, "float", "sdf_data[]"},
}};

const std::array<HaShaderResource, 7> G2P_RESOURCES{{
    {0, HaShaderQualifier::Read, "float", "params[]"},
    {1, HaShaderQualifier::Read, "float", "pos[]"},
    {2, HaShaderQualifier::ReadWrite, "float", "vel[]"},
    {3, HaShaderQualifier::ReadWrite, "float", "affine[]"},
    {4, HaShaderQualifier::ReadWrite, "float", "deform[]"},
    {5, HaShaderQualifier::Read, "float", "props[]"},
    {6, HaShaderQualifier::Read, "float", "gvel[]"},
}};

const std::array<HaShaderResource, 6> ADVECT_RESOURCES{{
    {0, HaShaderQualifier::Read, "float", "params[]"},
    {1, HaShaderQualifier::ReadWrite, "float", "pos[]"},
    {2, HaShaderQualifier::ReadWrite, "float", "vel[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::Read, "float", "box_data[]"},
    {5, HaShaderQualifier::Read, "float", "sdf_data[]"},
}};

/** \} */

/** One frame's worth of device buffers. Reused between frames. */
struct DeviceState {
  HaBuffer params;
  HaBuffer positions;
  HaBuffer velocities;
  HaBuffer affine;
  HaBuffer deform;
  HaBuffer props;
  HaBufferInt grid_mass;
  HaBufferInt grid_momentum;
  HaBuffer grid_velocity;
  HaBuffer box_data;
  HaBuffer sdf_data;

  gpu::Shader *clear = nullptr;
  gpu::Shader *p2g = nullptr;
  gpu::Shader *grid = nullptr;
  gpu::Shader *g2p = nullptr;
  gpu::Shader *advect = nullptr;

  int particle_count = 0;
  int cell_count = 0;

  bool ensure(const int particles, const int cells, std::string &r_reason)
  {
    if (particle_count != particles || cell_count != cells) {
      particle_count = particles;
      cell_count = cells;
      if (!params.ensure(P_PARAM_COUNT) || !positions.ensure(int64_t(particles) * 3) ||
          !velocities.ensure(int64_t(particles) * 3) || !affine.ensure(int64_t(particles) * 9) ||
          !deform.ensure(int64_t(particles) * 9) || !props.ensure(int64_t(particles) * PROP_STRIDE) ||
          !grid_mass.ensure(cells) || !grid_momentum.ensure(int64_t(cells) * 3) ||
          !grid_velocity.ensure(int64_t(cells) * 3))
      {
        r_reason = "storage buffer allocation failed";
        return false;
      }
    }
    return true;
  }

  void release()
  {
    params.free();
    positions.free();
    velocities.free();
    affine.free();
    deform.free();
    props.free();
    grid_mass.free();
    grid_momentum.free();
    grid_velocity.free();
    box_data.free();
    sdf_data.free();
    clear = p2g = grid = g2p = advect = nullptr;
    particle_count = 0;
    cell_count = 0;
  }
};

DeviceState &device_state()
{
  static DeviceState state;
  return state;
}

/**
 * The device buffers outlive a frame on purpose, but they must not outlive the GPU backend: a
 * function-local static would only be destroyed at process exit, i.e. after `GPU_exit()`, and
 * Vulkan then reports the allocations as unfreed and crashes while tearing the device down.
 */
void release_device_state(void * /*user_data*/)
{
  device_state().release();
}

void ensure_exit_hook()
{
  static bool registered = false;
  if (!registered) {
    GPU_exit_hook_register(release_device_state, nullptr);
    registered = true;
  }
}

void dispatch(gpu::Shader *shader, Span<gpu::StorageBuf *> buffers, const int elements)
{
  if (shader == nullptr) {
    return;
  }
  GPU_shader_bind(shader);
  ha_bind_storage_buffers(buffers);
  ha_dispatch(shader, elements);
  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE);
}

bool debug_log_enabled()
{
  static const bool enabled = std::getenv("BLENDER_TEXTURA_GPU_LOG") != nullptr;
  return enabled;
}

/** \} */

}  // namespace

bool device_supported(const ParticleData &particles,
                      const Settings &settings,
                      std::string &r_reason)
{
  if (particles.size() == 0) {
    r_reason = "no particles";
    return false;
  }
  for (const int material : particles.material) {
    if (material != int(MaterialType::Fluid)) {
      r_reason = "sand and snow material updates are still CPU only";
      return false;
    }
  }
  if (settings.two_phase) {
    r_reason = "two-phase grids are still CPU only";
    return false;
  }
  for (const float value : particles.surface_tension) {
    if (value > 0.0f) {
      r_reason = "surface tension is still CPU only";
      return false;
    }
  }
  for (const float value : particles.wall_friction) {
    if (value > 0.0f) {
      r_reason = "per material wall friction is still CPU only";
      return false;
    }
  }
  for (const float value : particles.adhesion) {
    if (value > 0.0f) {
      r_reason = "adhesion is still CPU only";
      return false;
    }
  }
  return true;
}

bool device_solve(ParticleData &particles,
                  Span<const Collider> colliders,
                  const Settings &settings,
                  const GridData &grid,
                  const ha::HaSdfGrid *sdf,
                  float dt,
                  int substeps,
                  Stats &r_stats,
                  std::string &r_reason)
{
  const int particle_count = particles.size();
  const int cell_count = int(grid.cell_count());
  if (particle_count <= 0 || cell_count <= 0) {
    r_reason = "empty grid or particle set";
    return false;
  }

  int box_count = 0;
  for (const Collider &collider : colliders) {
    if (collider.box) {
      box_count++;
    }
  }
  if (box_count > MAX_BOXES) {
    r_reason = "more than 8 box colliders";
    return false;
  }

  const bool sdf_enabled = sdf != nullptr && sdf->is_valid();
  if (!sdf_enabled) {
    for (const Collider &collider : colliders) {
      if (collider.mesh != nullptr && !collider.box) {
        r_reason = "the mesh collider field is not available";
        return false;
      }
    }
  }

  HaContextGuard guard;
  if (!guard.is_valid()) {
    r_reason = ha_context_error();
    return false;
  }

  DeviceState &state = device_state();
  ensure_exit_hook();
  if (!state.ensure(particle_count, cell_count, r_reason)) {
    return false;
  }

  /* ---- upload ---- */
  {
    std::vector<float> params(P_PARAM_COUNT, 0.0f);
    params[P_COUNT] = float(particle_count);
    params[P_CELLS] = float(cell_count);
    params[P_DIM_X] = float(grid.dims.x);
    params[P_DIM_Y] = float(grid.dims.y);
    params[P_DIM_Z] = float(grid.dims.z);
    params[P_ORIGIN_X] = grid.origin.x;
    params[P_ORIGIN_Y] = grid.origin.y;
    params[P_ORIGIN_Z] = grid.origin.z;
    params[P_DX] = grid.dx;
    params[P_DT] = dt;
    params[P_GRAVITY_X] = settings.gravity.x;
    params[P_GRAVITY_Y] = settings.gravity.y;
    params[P_GRAVITY_Z] = settings.gravity.z;
    params[P_VELOCITY_DAMPING] = settings.velocity_damping;
    params[P_BOUNDARY_FRICTION] = settings.boundary_friction;
    params[P_CONTACT_SKIN] = std::max(settings.contact_skin, 0.0f);
    params[P_BOX_COUNT] = float(box_count);
    params[P_SDF_ENABLED] = sdf_enabled ? 1.0f : 0.0f;
    params[P_SDF_INVERTED] = (sdf_enabled && sdf->inverted) ? 1.0f : 0.0f;
    float sdf_friction = 0.0f;
    for (const Collider &collider : colliders) {
      if (!collider.box) {
        sdf_friction = std::max(sdf_friction, collider.friction);
      }
    }
    params[P_SDF_FRICTION] = sdf_friction;
    if (sdf_enabled) {
      params[P_SDF_ORIGIN_X] = sdf->origin.x;
      params[P_SDF_ORIGIN_Y] = sdf->origin.y;
      params[P_SDF_ORIGIN_Z] = sdf->origin.z;
      params[P_SDF_SPACING] = sdf->spacing;
      params[P_SDF_DIM_X] = float(sdf->dims.x);
      params[P_SDF_DIM_Y] = float(sdf->dims.y);
      params[P_SDF_DIM_Z] = float(sdf->dims.z);
    }
    state.params.upload(params);
  }

  state.positions.upload(
      Span<float>(const_cast<float *>(reinterpret_cast<const float *>(particles.positions.data())),
                  int64_t(particle_count) * 3));
  state.velocities.upload(
      Span<float>(const_cast<float *>(reinterpret_cast<const float *>(particles.velocities.data())),
                  int64_t(particle_count) * 3));
  state.affine.upload(
      Span<float>(const_cast<float *>(reinterpret_cast<const float *>(particles.affine.data())),
                  int64_t(particle_count) * 9));
  state.deform.upload(
      Span<float>(const_cast<float *>(reinterpret_cast<const float *>(particles.deformation.data())),
                  int64_t(particle_count) * 9));
  {
    std::vector<float> props(size_t(particle_count) * PROP_STRIDE, 0.0f);
    for (int i = 0; i < particle_count; i++) {
      float *dst = &props[size_t(i) * PROP_STRIDE];
      dst[PROP_MASS] = particles.mass[i];
      dst[PROP_VOLUME] = particles.volume[i];
      dst[PROP_RADIUS] = particles.radius[i];
      dst[PROP_BULK] = particles.bulk_modulus[i];
      dst[PROP_TENSION] = particles.tension[i];
      dst[PROP_DAMPING] = particles.damping[i];
      dst[PROP_VISCOSITY] = particles.viscosity[i];
    }
    state.props.upload(props);
  }

  {
    std::vector<float> boxes(size_t(std::max(box_count, 1)) * BOX_FLOATS, 0.0f);
    int box_index = 0;
    for (const Collider &collider : colliders) {
      if (!collider.box) {
        continue;
      }
      float *dst = &boxes[size_t(box_index) * BOX_FLOATS];
      for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
          dst[col * 4 + row] = collider.box_transform[col][row];
          dst[16 + col * 4 + row] = collider.box_transform_inverse[col][row];
        }
      }
      dst[32] = collider.box_local_min.x;
      dst[33] = collider.box_local_min.y;
      dst[34] = collider.box_local_min.z;
      dst[35] = collider.box_local_max.x;
      dst[36] = collider.box_local_max.y;
      dst[37] = collider.box_local_max.z;
      dst[38] = collider.friction;
      dst[39] = collider.invert ? 1.0f : 0.0f;
      dst[40] = collider.box_scale;
      box_index++;
    }
    if (!state.box_data.ensure(int64_t(boxes.size()))) {
      r_reason = "box collider buffer allocation failed";
      return false;
    }
    state.box_data.upload(boxes);
  }

  if (sdf_enabled) {
    if (!state.sdf_data.ensure(sdf->data.size())) {
      r_reason = "collider field buffer allocation failed";
      return false;
    }
    state.sdf_data.upload(
        Span<float>(const_cast<float *>(sdf->data.data()), int64_t(sdf->data.size())));
  }
  else if (!state.sdf_data.ensure(1)) {
    r_reason = "collider field buffer allocation failed";
    return false;
  }

  /* ---- shaders ---- */
  state.clear = ha_shader_get_source("textura|clear",
                                     std::string(GLSL_PRELUDE) + KERNEL_CLEAR,
                                     as_span(CLEAR_RESOURCES),
                                     LOCAL_GROUP_SIZE);
  state.p2g = ha_shader_get_source("textura|p2g",
                                   std::string(GLSL_PRELUDE) + KERNEL_P2G,
                                   as_span(P2G_RESOURCES),
                                   LOCAL_GROUP_SIZE);
  state.grid = ha_shader_get_source("textura|grid",
                                    std::string(GLSL_PRELUDE) + KERNEL_GRID,
                                    as_span(GRID_RESOURCES),
                                    LOCAL_GROUP_SIZE);
  state.g2p = ha_shader_get_source("textura|g2p",
                                   std::string(GLSL_PRELUDE) + KERNEL_G2P,
                                   as_span(G2P_RESOURCES),
                                   LOCAL_GROUP_SIZE);
  state.advect = ha_shader_get_source("textura|advect",
                                      std::string(GLSL_PRELUDE) + KERNEL_ADVECT,
                                      as_span(ADVECT_RESOURCES),
                                      LOCAL_GROUP_SIZE);
  if (state.clear == nullptr || state.p2g == nullptr || state.grid == nullptr ||
      state.g2p == nullptr || state.advect == nullptr)
  {
    r_reason = "shader creation failed";
    return false;
  }

  std::array<gpu::StorageBuf *, 4> clear_buffers{
      state.params.buf(), state.grid_mass.buf(), state.grid_momentum.buf(),
      state.grid_velocity.buf()};
  std::array<gpu::StorageBuf *, 8> p2g_buffers{
      state.params.buf(), state.positions.buf(), state.velocities.buf(), state.affine.buf(),
      state.deform.buf(), state.props.buf(), state.grid_mass.buf(),
      state.grid_momentum.buf()};
  std::array<gpu::StorageBuf *, 6> grid_buffers{
      state.params.buf(), state.grid_mass.buf(), state.grid_momentum.buf(),
      state.grid_velocity.buf(), state.box_data.buf(), state.sdf_data.buf()};
  std::array<gpu::StorageBuf *, 7> g2p_buffers{
      state.params.buf(), state.positions.buf(), state.velocities.buf(), state.affine.buf(),
      state.deform.buf(), state.props.buf(), state.grid_velocity.buf()};
  std::array<gpu::StorageBuf *, 6> advect_buffers{
      state.params.buf(), state.positions.buf(), state.velocities.buf(), state.props.buf(),
      state.box_data.buf(), state.sdf_data.buf()};

  /* ---- substeps ---- */
  for (int step = 0; step < substeps; step++) {
    dispatch(state.clear, buffer_span(clear_buffers), cell_count);
    dispatch(state.p2g, buffer_span(p2g_buffers), particle_count);
    dispatch(state.grid, buffer_span(grid_buffers), cell_count);
    dispatch(state.g2p, buffer_span(g2p_buffers), particle_count);
    dispatch(state.advect, buffer_span(advect_buffers), particle_count);
  }
  GPU_finish();

  /* ---- download ---- */
  state.positions.download(
      MutableSpan<float>(reinterpret_cast<float *>(particles.positions.data()),
                         int64_t(particle_count) * 3));
  state.velocities.download(
      MutableSpan<float>(reinterpret_cast<float *>(particles.velocities.data()),
                         int64_t(particle_count) * 3));
  state.affine.download(
      MutableSpan<float>(reinterpret_cast<float *>(particles.affine.data()),
                         int64_t(particle_count) * 9));
  state.deform.download(
      MutableSpan<float>(reinterpret_cast<float *>(particles.deformation.data()),
                         int64_t(particle_count) * 9));

  r_stats.grid_cells = cell_count;
  r_stats.grid_dims = grid.dims;
  r_stats.substeps = substeps;

  if (debug_log_enabled()) {
    std::fprintf(stderr,
                 "TEXTURA_GPU frame backend=device particles=%d cells=%d substeps=%d\n",
                 particle_count,
                 cell_count,
                 substeps);
  }
  return true;
}

}  // namespace blender::geometry::textura
