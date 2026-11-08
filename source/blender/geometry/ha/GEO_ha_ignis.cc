/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Dense device implementation of the Ignis smoke solver. See `GEO_ha_ignis.hh` for the data model:
 * the caller converts the sparse OpenVDB state to a dense box, this file runs all substeps of the
 * frame on the compute backend, and the caller converts the result back.
 *
 * Every pass is a plain per-voxel kernel with Jacobi semantics (read the previous pass' buffers,
 * write a different buffer), so the result does not depend on scheduling. The only reductions are
 * CG dot products, which use one invocation per fixed block (a two stage reduction with a fixed
 * summation order).
 */

#include "GEO_ha_ignis.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "BLI_array.hh"
#include "BLI_math_vector.hh"
#include "BLI_span.hh"
#include "BLI_vector.hh"

#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_storage_buffer.hh"

#include "GEO_ha_context.hh"

namespace blender::geometry::ha {

namespace {

/* -------------------------------------------------------------------- */
/** \name Parameters
 * \{ */

/** Layout of the small float parameter buffer every kernel reads. */
enum IgnisParam : int {
  P_CELLS = 0,
  P_DX,
  P_DY,
  P_DZ,
  P_VOXEL,
  P_DT,
  P_GRAVITY_X,
  P_GRAVITY_Y,
  P_GRAVITY_Z,
  P_WIND_X,
  P_WIND_Y,
  P_WIND_Z,
  P_BUOYANCY,
  P_AMBIENT,
  P_DRAG,
  P_DAMPING,
  P_VORTICITY,
  P_BURN_RATE,
  P_HEAT_RELEASE,
  P_SMOKE_PER_BURN,
  P_COOLING,
  P_DISSIPATION,
  P_FLAME_THRESHOLD,
  P_DOMAIN_FIELD,
  P_DOMAIN_THRESHOLD,
  P_USE_BOUNDS,
  P_BOUNDS_MIN_X,
  P_BOUNDS_MIN_Y,
  P_BOUNDS_MIN_Z,
  P_BOUNDS_MAX_X,
  P_BOUNDS_MAX_Y,
  P_BOUNDS_MAX_Z,
  P_RADIUS,
  P_EXTRAPOLATE,
  P_TOLERANCE,
  P_MAX_ITERATIONS,
  P_ITERATION,
  P_RHS_NORM,
  P_RES_NORM,
  P_RZ,
  P_PQ,
  P_ALPHA,
  P_BETA,
  P_CONVERGED,
  P_PARTIALS,
  P_ORIGIN_X,
  P_ORIGIN_Y,
  P_ORIGIN_Z,
  P_DILATE_BIT,
  P_DIV_BEFORE_SUM,
  P_DIV_AFTER_SUM,
  P_UNKNOWN_COUNT,
  P_USE_DOMAIN_MASK,
  P_FORCE_TIME,
  P_DISTURBANCE,
  P_DISTURBANCE_SCALE,
  P_DISTURBANCE_SPEED,
  P_DISTURBANCE_SEED,
  P_DISTURBANCE_OCTAVES,
  P_DISTURBANCE_LACUNARITY,
  P_DISTURBANCE_ROUGHNESS,
  P_TURBULENCE,
  P_TURBULENCE_SCALE,
  P_TURBULENCE_SPEED,
  P_TURBULENCE_SEED,
  P_TURBULENCE_OCTAVES,
  P_TURBULENCE_LACUNARITY,
  P_TURBULENCE_ROUGHNESS,
  P_DIST_REMAP_FIELD,
  P_DIST_REMAP_MODE,
  P_DIST_REMAP_MIN,
  P_DIST_REMAP_MAX,
  P_DIST_REMAP_SOFTNESS,
  P_DIST_REMAP_SCALE,
  P_TURB_REMAP_FIELD,
  P_TURB_REMAP_MODE,
  P_TURB_REMAP_MIN,
  P_TURB_REMAP_MAX,
  P_TURB_REMAP_SOFTNESS,
  P_TURB_REMAP_SCALE,
  /* Coarse grid used for the pressure initial guess (see `coarse_pressure_guess`). */
  P_COARSE_DX,
  P_COARSE_DY,
  P_COARSE_DZ,
  P_COUNT,
};

/** `HA_BRICK_BASE` in `GLSL_PRELUDE` must match, the brick pool is appended after the params. */
static_assert(P_COUNT == 83, "update HA_BRICK_BASE in GLSL_PRELUDE when adding a parameter");

const char *GLSL_PRELUDE =
    "#define HA_SMOKE 1\n"
    "#define HA_DOMAIN 2\n"
    "#define HA_UNKNOWN 4\n"
    "#define HA_SOLID 8\n"
    "#define HA_FILLED 16\n"
    "#define HA_KEEP 32\n"
    "#define HA_BAND 64\n"
    "#define HA_REMOVED 128\n"
    "#define HA_CELL_COUNT int(params[0])\n"
    "#define HA_DX int(params[1])\n"
    "#define HA_DY int(params[2])\n"
    "#define HA_DZ int(params[3])\n"
    "#define HA_VOXEL params[4]\n"
    "#define HA_DT params[5]\n"
    "#define HA_GRAVITY vec3(params[6], params[7], params[8])\n"
    "#define HA_WIND vec3(params[9], params[10], params[11])\n"
    "#define HA_BUOYANCY params[12]\n"
    "#define HA_AMBIENT params[13]\n"
    "#define HA_DRAG params[14]\n"
    "#define HA_DAMPING params[15]\n"
    "#define HA_VORTICITY params[16]\n"
    "#define HA_BURN_RATE params[17]\n"
    "#define HA_HEAT_RELEASE params[18]\n"
    "#define HA_SMOKE_PER_BURN params[19]\n"
    "#define HA_COOLING params[20]\n"
    "#define HA_DISSIPATION params[21]\n"
    "#define HA_FLAME_THRESHOLD params[22]\n"
    "#define HA_DOMAIN_FIELD int(params[23])\n"
    "#define HA_DOMAIN_THRESHOLD params[24]\n"
    "#define HA_USE_BOUNDS int(params[25])\n"
    "#define HA_BOUNDS_MIN vec3(params[26], params[27], params[28])\n"
    "#define HA_BOUNDS_MAX vec3(params[29], params[30], params[31])\n"
    "#define HA_RADIUS int(params[32])\n"
    "#define HA_EXTRAPOLATE int(params[33])\n"
    "#define HA_TOLERANCE params[34]\n"
    "#define HA_MAX_ITERATIONS int(params[35])\n"
    "#define HA_ITERATION int(params[36])\n"
    "#define HA_RHS_NORM params[37]\n"
    "#define HA_RES_NORM params[38]\n"
    "#define HA_RZ params[39]\n"
    "#define HA_PQ params[40]\n"
    "#define HA_ALPHA params[41]\n"
    "#define HA_BETA params[42]\n"
    "#define HA_CONVERGED int(params[43])\n"
    "#define HA_PARTIAL_COUNT int(params[44])\n"
    "#define HA_ORIGIN vec3(params[45], params[46], params[47])\n"
    "#define HA_DILATE_BIT int(params[48])\n"
    "#define HA_DIV_BEFORE_SUM params[49]\n"
    "#define HA_DIV_AFTER_SUM params[50]\n"
    "#define HA_UNKNOWN_COUNT params[51]\n"
    "#define HA_USE_DOMAIN_MASK int(params[52])\n"
    "#define HA_FORCE_TIME params[53]\n"
    "#define HA_DISTURBANCE params[54]\n"
    "#define HA_DIST_SCALE params[55]\n"
    "#define HA_DIST_SPEED params[56]\n"
    "#define HA_DIST_SEED int(params[57])\n"
    "#define HA_DIST_OCTAVES int(params[58])\n"
    "#define HA_DIST_LACUNARITY params[59]\n"
    "#define HA_DIST_ROUGHNESS params[60]\n"
    "#define HA_TURBULENCE params[61]\n"
    "#define HA_TURB_SCALE params[62]\n"
    "#define HA_TURB_SPEED params[63]\n"
    "#define HA_TURB_SEED int(params[64])\n"
    "#define HA_TURB_OCTAVES int(params[65])\n"
    "#define HA_TURB_LACUNARITY params[66]\n"
    "#define HA_TURB_ROUGHNESS params[67]\n"
    "#define HA_DIST_REMAP_FIELD int(params[68])\n"
    "#define HA_DIST_REMAP_MODE int(params[69])\n"
    "#define HA_DIST_REMAP_MIN params[70]\n"
    "#define HA_DIST_REMAP_MAX params[71]\n"
    "#define HA_DIST_REMAP_SOFTNESS params[72]\n"
    "#define HA_DIST_REMAP_SCALE params[73]\n"
    "#define HA_TURB_REMAP_FIELD int(params[74])\n"
    "#define HA_TURB_REMAP_MODE int(params[75])\n"
    "#define HA_TURB_REMAP_MIN params[76]\n"
    "#define HA_TURB_REMAP_MAX params[77]\n"
    "#define HA_TURB_REMAP_SOFTNESS params[78]\n"
    "#define HA_TURB_REMAP_SCALE params[79]\n"
    "#define HA_COARSE_DX int(params[80])\n"
    "#define HA_COARSE_DY int(params[81])\n"
    "#define HA_COARSE_DZ int(params[82])\n"
    /* Brick pool (doc/ignis-brick-sparse.md). The layout is appended to the param buffer so every
     * kernel already has it bound; `HA_BRICK_BASE` must stay equal to `P_COUNT` on the C++ side. */
    "#define HA_BRICK_BASE 83\n"
    "#define HA_BRICK_SPARSE int(params[HA_BRICK_BASE + 0])\n"
    "#define HA_BRICK_TILES int(params[HA_BRICK_BASE + 1])\n"
    "#define HA_BRICK_DX int(params[HA_BRICK_BASE + 2])\n"
    "#define HA_BRICK_DY int(params[HA_BRICK_BASE + 3])\n"
    "#define HA_BRICK_DZ int(params[HA_BRICK_BASE + 4])\n"
    "#define HA_BRICK_SLOTS (HA_BRICK_BASE + 8)\n"
    "#define HA_BRICK_COORDS (HA_BRICK_BASE + 8 + HA_BRICK_TILES)\n"
    "#define HA_BRICK_COUNT int(params[HA_BRICK_BASE + 6])\n"
    "#define HA_LIST_MODE int(params[HA_BRICK_BASE + 5])\n"
    "#define HA_BRICK_NEIGHBORS (HA_BRICK_COORDS + 3 * HA_BRICK_COUNT)\n"
    "int ha_flat(ivec3 c) { return c.x + HA_DX * (c.y + HA_DY * c.z); }\n"
    "void ha_cell(int i, out ivec3 c)\n"
    "{\n"
    "  if (HA_BRICK_SPARSE == 0) {\n"
    "    int dx = HA_DX;\n"
    "    int dy = HA_DY;\n"
    "    c.x = i % dx;\n"
    "    i /= dx;\n"
    "    c.y = i % dy;\n"
    "    c.z = i / dy;\n"
    "    return;\n"
    "  }\n"
    "  int slot = i >> 6;\n"
    "  int local = i & 63;\n"
    "  int base = HA_BRICK_COORDS + slot * 3;\n"
    "  c = ivec3(int(params[base + 0]) + (local & 3),\n"
    "            int(params[base + 1]) + ((local >> 2) & 3),\n"
    "            int(params[base + 2]) + ((local >> 4) & 3));\n"
    "}\n"
    "bool ha_inside(ivec3 c)\n"
    "{\n"
    "  return c.x >= 0 && c.y >= 0 && c.z >= 0 && c.x < HA_DX && c.y < HA_DY && c.z < HA_DZ;\n"
    "}\n"
    /* Cell addressing. Dense: a cell is its flat index. Sparse: the cell lives in a 4^3 brick of the
     * pool, `slot * 64 + local`. Either way a neighbour is `ha_at(c + offset)`, and the kernels never
     * index a field with `ha_flat` directly, so both builds share every kernel.
     *
     * `ha_at` is on every stencil edge, so the 27 bricks around the current one are cached in shared
     * memory (`ha_init_neighbors`): the tile map lookup then costs a shared read instead of a
     * dependent global load. Coordinates outside that neighbourhood (advection backtraces) fall back
     * to the global map. Every kernel that reaches a `ha_at` has to call `ha_init_neighbors` first. */
    "shared int ha_nb_slots[27];\n"
    "shared ivec3 ha_nb_center;\n"
    "void ha_init_neighbors(int index)\n"
    "{\n"
    "  if (HA_BRICK_SPARSE == 0) { return; }\n"
    "  ivec3 cc;\n"
    "  ha_cell(index, cc);\n"
    "  ivec3 t = ivec3(cc.x >> 2, cc.y >> 2, cc.z >> 2);\n"
    "  int local = int(gl_LocalInvocationIndex);\n"
    "  if (local == 0) { ha_nb_center = t; }\n"
    "  if (local < 27) {\n"
    "    ivec3 d = ivec3(local % 3, (local / 3) % 3, local / 9) - ivec3(1);\n"
    "    ivec3 nt = t + d;\n"
    "    int slot = -1;\n"
    "    if (nt.x >= 0 && nt.y >= 0 && nt.z >= 0 && nt.x < HA_BRICK_DX && nt.y < HA_BRICK_DY &&\n"
    "        nt.z < HA_BRICK_DZ) {\n"
    "      slot = int(params[HA_BRICK_SLOTS + nt.x + HA_BRICK_DX * (nt.y + HA_BRICK_DY * nt.z)]);\n"
    "    }\n"
    "    ha_nb_slots[local] = slot;\n"
    "  }\n"
    "  barrier();\n"
    "}\n"
    "int ha_at(ivec3 c)\n"
    "{\n"
    "  if (!ha_inside(c)) { return -1; }\n"
    "  if (HA_BRICK_SPARSE == 0) { return ha_flat(c); }\n"
    "  ivec3 t = ivec3(c.x >> 2, c.y >> 2, c.z >> 2);\n"
    "  ivec3 d = t - ha_nb_center;\n"
    "  int slot;\n"
    "  if (d.x >= -1 && d.x <= 1 && d.y >= -1 && d.y <= 1 && d.z >= -1 && d.z <= 1) {\n"
    "    slot = ha_nb_slots[(d.x + 1) + 3 * ((d.y + 1) + 3 * (d.z + 1))];\n"
    "  }\n"
    "  else {\n"
    "    slot = int(params[HA_BRICK_SLOTS + t.x + HA_BRICK_DX * (t.y + HA_BRICK_DY * t.z)]);\n"
    "  }\n"
    "  if (slot < 0) { return -1; }\n"
    "  int local = (c.x & 3) + 4 * ((c.y & 3) + 4 * (c.z & 3));\n"
    "  return slot * 64 + local;\n"
    "}\n"
    "int ha_at_global(ivec3 c)\n"
    "{\n"
    "  if (!ha_inside(c)) { return -1; }\n"
    "  if (HA_BRICK_SPARSE == 0) { return ha_flat(c); }\n"
    "  int slot = int(params[HA_BRICK_SLOTS + (c.x >> 2) + HA_BRICK_DX * ((c.y >> 2) + HA_BRICK_DY * (c.z >> 2))]);\n"
    "  if (slot < 0) { return -1; }\n"
    "  return slot * 64 + ((c.x & 3) + 4 * ((c.y & 3) + 4 * (c.z & 3)));\n"
    "}\n"
    "vec3 ha_world(ivec3 c) { return (HA_ORIGIN + vec3(c)) * HA_VOXEL; }\n";

/** Per pass controls of the advection kernels (a separate buffer, so they never clobber CG state).
 */
const char *GLSL_ADVECT_CONTROLS =
    "#define HA_ADVECT_DIR adv[0]\n"
    "#define HA_ADVECT_ORDER int(adv[1])\n"
    "#define HA_LIMITER int(adv[2])\n"
    "#define HA_COMPONENT int(adv[3])\n"
    "#define HA_CORRECT_MODE int(adv[4])\n";

HaShaderResource res(const int slot,
                     const HaShaderQualifier qualifier,
                     const char *type,
                     const char *name)
{
  HaShaderResource r;
  r.slot = slot;
  r.qualifier = qualifier;
  r.type = type;
  r.name = name;
  return r;
}

#define HA_READ(slot, type, name) res(slot, HaShaderQualifier::Read, type, name)
#define HA_WRITE(slot, type, name) res(slot, HaShaderQualifier::Write, type, name)
#define HA_RW(slot, type, name) res(slot, HaShaderQualifier::ReadWrite, type, name)

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Shared GLSL helpers
 * \{ */

const char *GLSL_MASK_HELPERS =
    "int ha_mask_at(ivec3 c) { int i = ha_at(c); return i < 0 ? 0 : mask[i]; }\n";

const char *GLSL_MASK_IN_HELPERS =
    "int ha_mask_in_at(ivec3 c) { int i = ha_at(c); return i < 0 ? 0 : mask_in[i]; }\n";

const char *GLSL_SCALAR_HELPERS =
    "float ha_scalar_at(ivec3 c, int component)\n"
    "{\n"
    "  int i = ha_at(c);\n"
    "  return i < 0 ? 0.0 : scalars[i * 4 + component];\n"
    "}\n";

const char *GLSL_VELOCITY_HELPERS =
    "vec3 ha_velocity(ivec3 c)\n"
    "{\n"
    "  int i = ha_at(c);\n"
    "  if (i < 0) { return vec3(0.0); }\n"
    "  int base = i * 3;\n"
    "  return vec3(velocity[base + 0], velocity[base + 1], velocity[base + 2]);\n"
    "}\n"
    "void ha_velocity_set(ivec3 c, vec3 v)\n"
    "{\n"
    "  int i = ha_at(c);\n"
    "  if (i < 0) { return; }\n"
    "  int base = i * 3;\n"
    "  velocity[base + 0] = v.x;\n"
    "  velocity[base + 1] = v.y;\n"
    "  velocity[base + 2] = v.z;\n"
    "}\n"
    "vec3 ha_velocity_sample(vec3 p)\n"
    "{\n"
    "  vec3 pf = floor(p);\n"
    "  ivec3 i = ivec3(pf);\n"
    "  vec3 f = p - pf;\n"
    "  vec3 v000 = ha_velocity(i);\n"
    "  vec3 v100 = ha_velocity(i + ivec3(1, 0, 0));\n"
    "  vec3 v010 = ha_velocity(i + ivec3(0, 1, 0));\n"
    "  vec3 v110 = ha_velocity(i + ivec3(1, 1, 0));\n"
    "  vec3 v001 = ha_velocity(i + ivec3(0, 0, 1));\n"
    "  vec3 v101 = ha_velocity(i + ivec3(1, 0, 1));\n"
    "  vec3 v011 = ha_velocity(i + ivec3(0, 1, 1));\n"
    "  vec3 v111 = ha_velocity(i + ivec3(1, 1, 1));\n"
    "  vec3 v00 = mix(v000, v100, f.x);\n"
    "  vec3 v10 = mix(v010, v110, f.x);\n"
    "  vec3 v01 = mix(v001, v101, f.x);\n"
    "  vec3 v11 = mix(v011, v111, f.x);\n"
    "  return mix(mix(v00, v10, f.y), mix(v01, v11, f.y), f.z);\n"
    "}\n";

const char *GLSL_ADVECT_VEC3_FIELD =
    "vec3 ha_field_read(ivec3 c)\n"
    "{\n"
    "  int i = ha_at(c);\n"
    "  if (i < 0) { return vec3(0.0); }\n"
    "  int base = i * 3;\n"
    "  return vec3(field_in[base + 0], field_in[base + 1], field_in[base + 2]);\n"
    "}\n"
    "void ha_field_write(int i, vec3 v)\n"
    "{\n"
    "  field_out[i * 3 + 0] = v.x;\n"
    "  field_out[i * 3 + 1] = v.y;\n"
    "  field_out[i * 3 + 2] = v.z;\n"
    "}\n"
    "vec3 ha_field_forward(int i)\n"
    "{\n"
    "  return vec3(field_out[i * 3 + 0], field_out[i * 3 + 1], field_out[i * 3 + 2]);\n"
    "}\n";

const char *GLSL_ADVECT_VEC3_AUX =
    "vec3 ha_field_aux(ivec3 c)\n"
    "{\n"
    "  int i = ha_at(c);\n"
    "  if (i < 0) { return vec3(0.0); }\n"
    "  int base = i * 3;\n"
    "  return vec3(field_aux[base + 0], field_aux[base + 1], field_aux[base + 2]);\n"
    "}\n";

const char *GLSL_ADVECT_SCALAR_FIELD =
    "float ha_field_read(ivec3 c)\n"
    "{\n"
    "  int i = ha_at(c);\n"
    "  return i < 0 ? 0.0 : field_in[i * 4 + HA_COMPONENT];\n"
    "}\n"
    "void ha_field_write(int i, float v) { field_out[i] = v; }\n"
    "float ha_field_forward(int i) { return field_out[i]; }\n";

const char *GLSL_ADVECT_SCALAR_AUX =
    "float ha_field_aux(ivec3 c) { int i = ha_at(c); return i < 0 ? 0.0 : field_aux[i]; }\n";

/** The same for a buffer that already has one float per voxel (the intermediate of a scalar pass).
 */
const char *GLSL_ADVECT_PLAIN_FIELD =
    "float ha_field_read(ivec3 c) { int i = ha_at(c); return i < 0 ? 0.0 : field_in[i]; }\n"
    "void ha_field_write(int i, float v) { field_out[i] = v; }\n";

const char *GLSL_ADVECT_SAMPLE_VEC3 =
    "vec3 ha_sample_field(vec3 p)\n"
    "{\n"
    "  vec3 pf = floor(p);\n"
    "  ivec3 i = ivec3(pf);\n"
    "  vec3 f = p - pf;\n"
    "  vec3 v000 = ha_field_read(i);\n"
    "  vec3 v100 = ha_field_read(i + ivec3(1, 0, 0));\n"
    "  vec3 v010 = ha_field_read(i + ivec3(0, 1, 0));\n"
    "  vec3 v110 = ha_field_read(i + ivec3(1, 1, 0));\n"
    "  vec3 v001 = ha_field_read(i + ivec3(0, 0, 1));\n"
    "  vec3 v101 = ha_field_read(i + ivec3(1, 0, 1));\n"
    "  vec3 v011 = ha_field_read(i + ivec3(0, 1, 1));\n"
    "  vec3 v111 = ha_field_read(i + ivec3(1, 1, 1));\n"
    "  vec3 v00 = mix(v000, v100, f.x);\n"
    "  vec3 v10 = mix(v010, v110, f.x);\n"
    "  vec3 v01 = mix(v001, v101, f.x);\n"
    "  vec3 v11 = mix(v011, v111, f.x);\n"
    "  return mix(mix(v00, v10, f.y), mix(v01, v11, f.y), f.z);\n"
    "}\n";

const char *GLSL_ADVECT_SAMPLE_SCALAR =
    "float ha_sample_field(vec3 p)\n"
    "{\n"
    "  vec3 pf = floor(p);\n"
    "  ivec3 i = ivec3(pf);\n"
    "  vec3 f = p - pf;\n"
    "  float v000 = ha_field_read(i);\n"
    "  float v100 = ha_field_read(i + ivec3(1, 0, 0));\n"
    "  float v010 = ha_field_read(i + ivec3(0, 1, 0));\n"
    "  float v110 = ha_field_read(i + ivec3(1, 1, 0));\n"
    "  float v001 = ha_field_read(i + ivec3(0, 0, 1));\n"
    "  float v101 = ha_field_read(i + ivec3(1, 0, 1));\n"
    "  float v011 = ha_field_read(i + ivec3(0, 1, 1));\n"
    "  float v111 = ha_field_read(i + ivec3(1, 1, 1));\n"
    "  float v00 = mix(v000, v100, f.x);\n"
    "  float v10 = mix(v010, v110, f.x);\n"
    "  float v01 = mix(v001, v101, f.x);\n"
    "  float v11 = mix(v011, v111, f.x);\n"
    "  return mix(mix(v00, v10, f.y), mix(v01, v11, f.y), f.z);\n"
    "}\n";

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Procedural force helpers
 *
 * A bit-for-bit port of the CPU hash and value noise in `GEO_ignis_solver.cc`. `uint` arithmetic
 * matches between C++ and GLSL, so the disturbance and turbulence patterns agree on both backends.
 * \{ */

const char *GLSL_FORCE_HELPERS =
    "uint ha_force_hash(uint x)\n"
    "{\n"
    "  x ^= x >> 16;\n"
    "  x *= 0x7feb352du;\n"
    "  x ^= x >> 15;\n"
    "  x *= 0x846ca68bu;\n"
    "  x ^= x >> 16;\n"
    "  return x;\n"
    "}\n"
    "float ha_force_channel(uint h, int channel)\n"
    "{\n"
    "  uint shift = uint(channel) * 10u;\n"
    "  return float((h >> shift) & 0x3FFu) * (2.0 / 1023.0) - 1.0;\n"
    "}\n"
    "vec3 ha_force_random3(ivec3 c, int seed)\n"
    "{\n"
    "  uint h = ha_force_hash(uint(c.x) + ha_force_hash(uint(c.y) + ha_force_hash(uint(c.z) + "
    "ha_force_hash(uint(seed)))));\n"
    "  return vec3(ha_force_channel(h, 0), ha_force_channel(h, 1), ha_force_channel(h, 2));\n"
    "}\n"
    "float ha_force_fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }\n"
    "vec3 ha_force_value_noise(vec3 p, int seed)\n"
    "{\n"
    "  vec3 base = floor(p);\n"
    "  ivec3 c = ivec3(base);\n"
    "  vec3 f = p - base;\n"
    "  vec3 u = vec3(ha_force_fade(f.x), ha_force_fade(f.y), ha_force_fade(f.z));\n"
    "  vec3 n000 = ha_force_random3(c + ivec3(0, 0, 0), seed);\n"
    "  vec3 n100 = ha_force_random3(c + ivec3(1, 0, 0), seed);\n"
    "  vec3 n010 = ha_force_random3(c + ivec3(0, 1, 0), seed);\n"
    "  vec3 n110 = ha_force_random3(c + ivec3(1, 1, 0), seed);\n"
    "  vec3 n001 = ha_force_random3(c + ivec3(0, 0, 1), seed);\n"
    "  vec3 n101 = ha_force_random3(c + ivec3(1, 0, 1), seed);\n"
    "  vec3 n011 = ha_force_random3(c + ivec3(0, 1, 1), seed);\n"
    "  vec3 n111 = ha_force_random3(c + ivec3(1, 1, 1), seed);\n"
    "  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),\n"
    "             mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);\n"
    "}\n"
    "vec3 ha_force_curl(vec3 p, int seed)\n"
    "{\n"
    "  float e = 0.5;\n"
    "  vec3 v0 = ha_force_value_noise(p, seed);\n"
    "  vec3 vx = ha_force_value_noise(p + vec3(e, 0.0, 0.0), seed);\n"
    "  vec3 vy = ha_force_value_noise(p + vec3(0.0, e, 0.0), seed);\n"
    "  vec3 vz = ha_force_value_noise(p + vec3(0.0, 0.0, e), seed);\n"
    "  return vec3(((vy.z - v0.z) - (vz.y - v0.y)) / e,\n"
    "              ((vz.x - v0.x) - (vx.z - v0.z)) / e,\n"
    "              ((vx.y - v0.y) - (vy.x - v0.x)) / e);\n"
    "}\n"
    "float ha_remap_window(float value, int mode, float minv, float maxv, float softness)\n"
    "{\n"
    "  if (maxv < minv) { float t = minv; minv = maxv; maxv = t; }\n"
    "  float range = max(maxv - minv, 1e-6);\n"
    "  float soft = clamp(softness, 0.0, 1.0) * range;\n"
    "  float lower = smoothstep(minv, minv + max(soft, 1e-6), value);\n"
    "  float upper = 1.0 - smoothstep(maxv - max(soft, 1e-6), maxv, value);\n"
    "  float inside = lower * upper;\n"
    "  if (mode == 1) { return 1.0 - inside; }\n"
    "  if (mode == 2) { return lower; }\n"
    "  if (mode == 3) { return upper; }\n"
    "  return inside;\n"
    "}\n"
    "float ha_remap_field_value(int field, ivec3 c, vec3 velocity_now)\n"
    "{\n"
    "  int index = ha_at(c);\n"
    "  if (index < 0) { return 0.0; }\n"
    "  if (field == 1) { return pressure[index]; }\n"
    "  if (field == 2) { return scalars[index * 4 + 0]; }\n"
    "  if (field == 3) { return scalars[index * 4 + 1]; }\n"
    "  if (field == 4) { return scalars[index * 4 + 2]; }\n"
    "  if (field == 5) { return scalars[index * 4 + 3]; }\n"
    "  if (field == 6) { return length(velocity_now); }\n"
    "  return 1.0;\n"
    "}\n"
    "vec3 ha_disturbance(vec3 p)\n"
    "{\n"
    "  if (HA_DISTURBANCE == 0.0) { return vec3(0.0); }\n"
    "  float scale = max(HA_DIST_SCALE, 1e-4);\n"
    "  vec3 q = p / scale + vec3(HA_FORCE_TIME * HA_DIST_SPEED,\n"
    "                           HA_FORCE_TIME * HA_DIST_SPEED * 0.73,\n"
    "                           HA_FORCE_TIME * HA_DIST_SPEED * 0.41);\n"
    "  float amp = 1.0;\n"
    "  float total = 0.0;\n"
    "  vec3 result = vec3(0.0);\n"
    "  for (int o = 0; o < 8; ++o) {\n"
    "    if (o >= HA_DIST_OCTAVES) { break; }\n"
    "    result += ha_force_value_noise(q + vec3(float(o) * 17.3, float(o) * 9.1, "
    "float(o) * 3.7), HA_DIST_SEED + o * 131) * amp;\n"
    "    total += amp;\n"
    "    amp *= HA_DIST_ROUGHNESS;\n"
    "    q *= HA_DIST_LACUNARITY;\n"
    "  }\n"
    "  return result * (HA_DISTURBANCE / max(total, 1e-4));\n"
    "}\n"
    "vec3 ha_turbulence(vec3 p)\n"
    "{\n"
    "  if (HA_TURBULENCE == 0.0) { return vec3(0.0); }\n"
    "  float scale = max(HA_TURB_SCALE, 1e-4);\n"
    "  vec3 q = p / scale + vec3(HA_FORCE_TIME * HA_TURB_SPEED,\n"
    "                           HA_FORCE_TIME * HA_TURB_SPEED * 0.61,\n"
    "                           HA_FORCE_TIME * HA_TURB_SPEED * 0.37);\n"
    "  float amp = 1.0;\n"
    "  float total = 0.0;\n"
    "  vec3 result = vec3(0.0);\n"
    "  for (int o = 0; o < 8; ++o) {\n"
    "    if (o >= HA_TURB_OCTAVES) { break; }\n"
    "    result += ha_force_curl(q + vec3(float(o) * 7.1, float(o) * 23.7, "
    "float(o) * 11.3), HA_TURB_SEED + o * 197) * amp;\n"
    "    total += amp;\n"
    "    amp *= HA_TURB_ROUGHNESS;\n"
    "    q *= HA_TURB_LACUNARITY;\n"
    "  }\n"
    "  return result * (HA_TURBULENCE / max(total, 1e-4));\n"
    "}\n";

/** \} */

/* -------------------------------------------------------------------- */
/** \name Kernels
 * \{ */

const char *KERNEL_SOURCE_MERGE =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= HA_CELL_COUNT) { return; }\n"
    "  float source_density = source[i * 7 + 0];\n"
    "  if (source_density <= 0.0) { return; }\n"
    "  scalars[i * 4 + 0] = max(scalars[i * 4 + 0], source_density);\n"
    "  float source_temperature = source[i * 7 + 1];\n"
    "  if (source_temperature != 0.0) {\n"
    "    float temperature = scalars[i * 4 + 1];\n"
    "    scalars[i * 4 + 1] = source_temperature > 0.0 ? max(temperature, source_temperature) :\n"
    "                                                    min(temperature, source_temperature);\n"
    "  }\n"
    "  float source_fuel = source[i * 7 + 2];\n"
    "  if (source_fuel > 0.0) { scalars[i * 4 + 2] = max(scalars[i * 4 + 2], source_fuel); }\n"
    "  vec3 incoming = vec3(source[i * 7 + 3], source[i * 7 + 4], source[i * 7 + 5]);\n"
    "  if (dot(incoming, incoming) > 0.0) {\n"
    "    float weight = clamp(source_density, 0.0, 1.0);\n"
    "    int base = i * 3;\n"
    "    vec3 current = vec3(velocity[base + 0], velocity[base + 1], velocity[base + 2]);\n"
    "    vec3 blended = current + (incoming - current) * weight;\n"
    "    velocity[base + 0] = blended.x;\n"
    "    velocity[base + 1] = blended.y;\n"
    "    velocity[base + 2] = blended.z;\n"
    "  }\n"
    "}\n";

const char *KERNEL_FORCES =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(i);\n"
    "  if (scalars[i * 4 + 0] <= 1e-4) { return; }\n"
    "  float anomaly = scalars[i * 4 + 1] - HA_AMBIENT;\n"
    "  int base = i * 3;\n"
    "  ivec3 c;\n"
    "  ha_cell(i, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  vec3 v = vec3(velocity[base + 0], velocity[base + 1], velocity[base + 2]);\n"
    "  if (anomaly != 0.0) { v -= HA_GRAVITY * (HA_BUOYANCY * anomaly * HA_DT); }\n"
    "  v += HA_WIND * HA_DT;\n"
    "  vec3 remap_velocity = v;\n"
    "  float dist_factor = 1.0;\n"
    "  float turb_factor = 1.0;\n"
    "  if (HA_DIST_REMAP_FIELD != 0) {\n"
    "    float fv = ha_remap_field_value(HA_DIST_REMAP_FIELD, c, remap_velocity);\n"
    "    dist_factor = ha_remap_window(fv, HA_DIST_REMAP_MODE, HA_DIST_REMAP_MIN,\n"
    "                                  HA_DIST_REMAP_MAX, HA_DIST_REMAP_SOFTNESS) *\n"
    "                  max(HA_DIST_REMAP_SCALE, 0.0);\n"
    "  }\n"
    "  if (HA_TURB_REMAP_FIELD != 0) {\n"
    "    float fv = ha_remap_field_value(HA_TURB_REMAP_FIELD, c, remap_velocity);\n"
    "    turb_factor = ha_remap_window(fv, HA_TURB_REMAP_MODE, HA_TURB_REMAP_MIN,\n"
    "                                  HA_TURB_REMAP_MAX, HA_TURB_REMAP_SOFTNESS) *\n"
    "                  max(HA_TURB_REMAP_SCALE, 0.0);\n"
    "  }\n"
    "  if (HA_DISTURBANCE != 0.0 || HA_TURBULENCE != 0.0) {\n"
    "    vec3 p = ha_world(c);\n"
    "    if (HA_DISTURBANCE != 0.0 && dist_factor != 0.0) {\n"
    "      v += ha_disturbance(p) * (HA_DT * dist_factor);\n"
    "    }\n"
    "    if (HA_TURBULENCE != 0.0 && turb_factor != 0.0) {\n"
    "      v += ha_turbulence(p) * (HA_DT * turb_factor);\n"
    "    }\n"
    "  }\n"
    "  if (HA_DRAG > 0.0) { v *= 1.0 / (1.0 + HA_DRAG * HA_DT); }\n"
    "  v *= 1.0 / (1.0 + HA_DAMPING * HA_DT);\n"
    "  velocity[base + 0] = v.x;\n"
    "  velocity[base + 1] = v.y;\n"
    "  velocity[base + 2] = v.z;\n"
    "}\n";

const char *KERNEL_VORTICITY_CURL =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(i);\n"
    "  ivec3 c;\n"
    "  ha_cell(i, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  bool band = ha_scalar_at(c, 0) > 1e-4 || ha_scalar_at(c + ivec3(1, 0, 0), 0) > 1e-4 ||\n"
    "              ha_scalar_at(c + ivec3(-1, 0, 0), 0) > 1e-4 ||\n"
    "              ha_scalar_at(c + ivec3(0, 1, 0), 0) > 1e-4 ||\n"
    "              ha_scalar_at(c + ivec3(0, -1, 0), 0) > 1e-4 ||\n"
    "              ha_scalar_at(c + ivec3(0, 0, 1), 0) > 1e-4 ||\n"
    "              ha_scalar_at(c + ivec3(0, 0, -1), 0) > 1e-4;\n"
    "  if (!band) {\n"
    "    mask[i] = mask[i] & ~HA_BAND;\n"
    "    return;\n"
    "  }\n"
    "  vec3 dx_p = ha_velocity(c + ivec3(1, 0, 0));\n"
    "  vec3 dx_m = ha_velocity(c + ivec3(-1, 0, 0));\n"
    "  vec3 dy_p = ha_velocity(c + ivec3(0, 1, 0));\n"
    "  vec3 dy_m = ha_velocity(c + ivec3(0, -1, 0));\n"
    "  vec3 dz_p = ha_velocity(c + ivec3(0, 0, 1));\n"
    "  vec3 dz_m = ha_velocity(c + ivec3(0, 0, -1));\n"
    "  vec3 omega = vec3((dy_p.z - dy_m.z) - (dz_p.y - dz_m.y),\n"
    "                    (dz_p.x - dz_m.x) - (dx_p.z - dx_m.z),\n"
    "                    (dx_p.y - dx_m.y) - (dy_p.x - dy_m.x)) * (0.5 / HA_VOXEL);\n"
    "  curl[i * 4 + 0] = omega.x;\n"
    "  curl[i * 4 + 1] = omega.y;\n"
    "  curl[i * 4 + 2] = omega.z;\n"
    "  curl[i * 4 + 3] = length(omega);\n"
    "  mask[i] = mask[i] | HA_BAND;\n"
    "}\n";

const char *KERNEL_VORTICITY_APPLY =
    "float ha_curl_length(ivec3 c) { int i = ha_at(c); return i < 0 ? 0.0 : curl[i * 4 + 3]; }\n"
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(i);\n"
    "  if ((mask[i] & HA_BAND) == 0) { return; }\n"
    "  ivec3 c;\n"
    "  ha_cell(i, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  vec3 normal = vec3(ha_curl_length(c + ivec3(1, 0, 0)) - ha_curl_length(c + ivec3(-1, 0, "
    "0)),\n"
    "                     ha_curl_length(c + ivec3(0, 1, 0)) - ha_curl_length(c + ivec3(0, -1, "
    "0)),\n"
    "                     ha_curl_length(c + ivec3(0, 0, 1)) - ha_curl_length(c + ivec3(0, 0, "
    "-1)));\n"
    "  float normal_length = length(normal);\n"
    "  if (normal_length < 1e-8) { return; }\n"
    "  vec3 value = vec3(curl[i * 4 + 0], curl[i * 4 + 1], curl[i * 4 + 2]);\n"
    "  vec3 force = cross(normal / normal_length, value) * (HA_VORTICITY * HA_VOXEL * HA_DT);\n"
    "  int base = i * 3;\n"
    "  velocity[base + 0] += force.x;\n"
    "  velocity[base + 1] += force.y;\n"
    "  velocity[base + 2] += force.z;\n"
    "}\n";

const char *KERNEL_ADVECT_PASS =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  vec3 p = vec3(c);\n"
    "  float dir = HA_ADVECT_DIR;\n"
    "  vec3 v = ha_velocity_sample(p) / HA_VOXEL;\n"
    "  vec3 traced = p + dir * HA_DT * v;\n"
    "  if (HA_ADVECT_ORDER >= 2) {\n"
    "    vec3 v_mid = ha_velocity_sample(p + dir * HA_DT * 0.5 * v) / HA_VOXEL;\n"
    "    traced = p + dir * HA_DT * v_mid;\n"
    "  }\n"
    "  ha_field_write(index, ha_sample_field(traced));\n"
    "}\n";

const char *KERNEL_ADVECT_CORRECT_VEC3 =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  vec3 original = ha_field_read(c);\n"
    "  vec3 backward = ha_field_aux(c);\n"
    "  vec3 forward = ha_field_forward(index);\n"
    "  vec3 value = HA_CORRECT_MODE == 0 ? forward + 0.5 * (original - backward) :\n"
    "                                      0.5 * (3.0 * original - backward);\n"
    "  ha_field_write(index, value);\n"
    "}\n";

const char *KERNEL_ADVECT_CORRECT_SCALAR =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  float original = ha_field_read(c);\n"
    "  float backward = ha_field_aux(c);\n"
    "  float forward = ha_field_forward(index);\n"
    "  float value = HA_CORRECT_MODE == 0 ? forward + 0.5 * (original - backward) :\n"
    "                                      0.5 * (3.0 * original - backward);\n"
    "  ha_field_write(index, value);\n"
    "}\n";

const char *KERNEL_ADVECT_LIMITER_VEC3 =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  if (HA_LIMITER == 0) { return; }\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  vec3 traced = vec3(c) - HA_DT * (ha_velocity_sample(vec3(c)) / HA_VOXEL);\n"
    "  vec3 pf = floor(traced);\n"
    "  ivec3 i = ivec3(pf);\n"
    "  vec3 f = traced - pf;\n"
    "  vec3 v000 = ha_field_read(i);\n"
    "  vec3 v100 = ha_field_read(i + ivec3(1, 0, 0));\n"
    "  vec3 v010 = ha_field_read(i + ivec3(0, 1, 0));\n"
    "  vec3 v110 = ha_field_read(i + ivec3(1, 1, 0));\n"
    "  vec3 v001 = ha_field_read(i + ivec3(0, 0, 1));\n"
    "  vec3 v101 = ha_field_read(i + ivec3(1, 0, 1));\n"
    "  vec3 v011 = ha_field_read(i + ivec3(0, 1, 1));\n"
    "  vec3 v111 = ha_field_read(i + ivec3(1, 1, 1));\n"
    "  vec3 value = ha_field_forward(index);\n"
    "  vec3 vmin = min(min(min(v000, v100), min(v010, v110)),\n"
    "                  min(min(v001, v101), min(v011, v111)));\n"
    "  vec3 vmax = max(max(max(v000, v100), max(v010, v110)),\n"
    "                  max(max(v001, v101), max(v011, v111)));\n"
    "  if (HA_LIMITER == 1) {\n"
    "    value = clamp(value, vmin, vmax);\n"
    "  }\n"
    "  else if (any(lessThan(value, vmin)) || any(greaterThan(value, vmax))) {\n"
    "    vec3 a = mix(v000, v100, f.x);\n"
    "    vec3 b = mix(v010, v110, f.x);\n"
    "    vec3 cc = mix(v001, v101, f.x);\n"
    "    vec3 dd = mix(v011, v111, f.x);\n"
    "    value = mix(mix(a, b, f.y), mix(cc, dd, f.y), f.z);\n"
    "  }\n"
    "  ha_field_write(index, value);\n"
    "}\n";

const char *KERNEL_ADVECT_LIMITER_SCALAR =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  if (HA_LIMITER == 0) { return; }\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  vec3 traced = vec3(c) - HA_DT * (ha_velocity_sample(vec3(c)) / HA_VOXEL);\n"
    "  vec3 pf = floor(traced);\n"
    "  ivec3 i = ivec3(pf);\n"
    "  vec3 f = traced - pf;\n"
    "  float v000 = ha_field_read(i);\n"
    "  float v100 = ha_field_read(i + ivec3(1, 0, 0));\n"
    "  float v010 = ha_field_read(i + ivec3(0, 1, 0));\n"
    "  float v110 = ha_field_read(i + ivec3(1, 1, 0));\n"
    "  float v001 = ha_field_read(i + ivec3(0, 0, 1));\n"
    "  float v101 = ha_field_read(i + ivec3(1, 0, 1));\n"
    "  float v011 = ha_field_read(i + ivec3(0, 1, 1));\n"
    "  float v111 = ha_field_read(i + ivec3(1, 1, 1));\n"
    "  float value = ha_field_forward(index);\n"
    "  float vmin = min(min(min(v000, v100), min(v010, v110)),\n"
    "                   min(min(v001, v101), min(v011, v111)));\n"
    "  float vmax = max(max(max(v000, v100), max(v010, v110)),\n"
    "                   max(max(v001, v101), max(v011, v111)));\n"
    "  if (HA_LIMITER == 1) {\n"
    "    value = clamp(value, vmin, vmax);\n"
    "  }\n"
    "  else if (value < vmin || value > vmax) {\n"
    "    float a = mix(v000, v100, f.x);\n"
    "    float b = mix(v010, v110, f.x);\n"
    "    float cc = mix(v001, v101, f.x);\n"
    "    float dd = mix(v011, v111, f.x);\n"
    "    value = mix(mix(a, b, f.y), mix(cc, dd, f.y), f.z);\n"
    "  }\n"
    "  ha_field_write(index, value);\n"
    "}\n";

const char *KERNEL_COPY_VEC3 =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  int base = index * 3;\n"
    "  field_out[base + 0] = field_in[base + 0];\n"
    "  field_out[base + 1] = field_in[base + 1];\n"
    "  field_out[base + 2] = field_in[base + 2];\n"
    "}\n";

const char *KERNEL_TMP_TO_SCALAR =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  field_out[index * 4 + HA_COMPONENT] = field_in[index];\n"
    "}\n";

/* Density, temperature and fuel live in one interleaved buffer with 4 floats per voxel. The fused
 * scalar advection packs the first three into a 3 float buffer (the same layout the velocity uses),
 * so one vec3 advection can handle all three components at once. */
const char *KERNEL_PACK_SCALARS =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  int src = index * 4;\n"
    "  int dst = index * 3;\n"
    "  packed[dst + 0] = scalars[src + 0];\n"
    "  packed[dst + 1] = scalars[src + 1];\n"
    "  packed[dst + 2] = scalars[src + 2];\n"
    "}\n";

const char *KERNEL_UNPACK_SCALARS =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  int src = index * 3;\n"
    "  int dst = index * 4;\n"
    "  scalars[dst + 0] = packed[src + 0];\n"
    "  scalars[dst + 1] = packed[src + 1];\n"
    "  scalars[dst + 2] = packed[src + 2];\n"
    "}\n";

const char *KERNEL_MASK_INIT =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  int m = solid[index];\n"
    "  if (scalars[index * 4 + 0] > 1e-4) { m = m | HA_SMOKE | HA_DOMAIN; }\n"
    "  mask_out[index] = m;\n"
    "}\n";

const char *KERNEL_DILATE =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  int bit = 1 << HA_DILATE_BIT;\n"
    "  int m = mask_in[index];\n"
    "  if ((m & bit) == 0) {\n"
    "    if ((ha_mask_in_at(c + ivec3(1, 0, 0)) & bit) != 0 ||\n"
    "        (ha_mask_in_at(c + ivec3(-1, 0, 0)) & bit) != 0 ||\n"
    "        (ha_mask_in_at(c + ivec3(0, 1, 0)) & bit) != 0 ||\n"
    "        (ha_mask_in_at(c + ivec3(0, -1, 0)) & bit) != 0 ||\n"
    "        (ha_mask_in_at(c + ivec3(0, 0, 1)) & bit) != 0 ||\n"
    "        (ha_mask_in_at(c + ivec3(0, 0, -1)) & bit) != 0) {\n"
    "      m = m | bit;\n"
    "    }\n"
    "  }\n"
    "  mask_out[index] = m;\n"
    "}\n";

const char *KERNEL_UNKNOWN =
    "bool ha_domain_cell(int value) { return (value & HA_DOMAIN) != 0 && (value & HA_SOLID) == 0; "
    "}\n"
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  int m = mask[index];\n"
    "  bool here = ha_domain_cell(m);\n"
    "  if (c.x + 1 >= HA_DX || c.y + 1 >= HA_DY || c.z + 1 >= HA_DZ) { here = false; }\n"
    "  if (here) {\n"
    "    here = ha_domain_cell(ha_mask_at(c + ivec3(1, 0, 0))) &&\n"
    "           ha_domain_cell(ha_mask_at(c + ivec3(0, 1, 0))) &&\n"
    "           ha_domain_cell(ha_mask_at(c + ivec3(0, 0, 1)));\n"
    "  }\n"
    "  mask[index] = here ? (m | HA_UNKNOWN) : (m & ~HA_UNKNOWN);\n"
    "}\n";

/* Two level initial guess for the pressure solve. `KERNEL_RESTRICT_COARSE` aggregates the fine
 * unknown mask and right hand side onto a grid of 2x2x2 blocks (a coarse cell is unknown when any
 * of its children is), `KERNEL_PROLONG_COARSE` injects the coarse solution back as the starting
 * pressure of the fine CG. Summing the eight fine equations gives a coarse operator of
 * `2^(d-1) * L`, while the coarse Laplacian is `2^2` times the fine one, hence the 1/2 factor so
 * that the prolongated pressure has the same magnitude as the fine solution. The coarse solve only
 * changes the starting point; the fine CG still runs to the requested tolerance. */
const char *KERNEL_RESTRICT_COARSE =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_COARSE_DX * HA_COARSE_DY * HA_COARSE_DZ) { return; }\n"
    "  int cdx = HA_COARSE_DX;\n"
    "  int cdy = HA_COARSE_DY;\n"
    "  int cz = index / (cdx * cdy);\n"
    "  int rem = index - cz * cdx * cdy;\n"
    "  ivec3 c = ivec3(rem % cdx, rem / cdx, cz);\n"
    "  int m = 0;\n"
    "  float sum = 0.0;\n"
    "  for (int dz = 0; dz < 2; dz++) {\n"
    "    for (int dy = 0; dy < 2; dy++) {\n"
    "      for (int dx = 0; dx < 2; dx++) {\n"
    "        ivec3 f = 2 * c + ivec3(dx, dy, dz);\n"
    "        int i = ha_at_global(f);\n"
    "        if (i < 0) { continue; }\n"
    "        if ((mask_in[i] & HA_UNKNOWN) != 0) {\n"
    "          m = HA_UNKNOWN;\n"
    "          sum += rhs_in[i];\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  mask_out[index] = m;\n"
    "  rhs_out[index] = 0.5 * sum;\n"
    "}\n";

const char *KERNEL_ZERO_FIELD =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  field[index] = 0.0;\n"
    "}\n";

const char *KERNEL_PROLONG_COARSE =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  if ((mask[index] & HA_UNKNOWN) == 0) {\n"
    "    pressure[index] = 0.0;\n"
    "    return;\n"
    "  }\n"
    "  int ci = (c.x / 2) + HA_COARSE_DX * ((c.y / 2) + HA_COARSE_DY * (c.z / 2));\n"
    "  pressure[index] = coarse_pressure[ci];\n"
    "}\n";

/* Per projection brick lists: `KERNEL_BRICK_FLAG` marks the bricks that contain a cell of the given
 * mask bit (UNKNOWN here), `KERNEL_BRICK_COMPACT` builds the list in slot order (deterministic) and
 * zeroes the reduction slots of the bricks that are not in it, so the `reduce_*` kernels can keep
 * summing over all bricks. */
const char *KERNEL_BRICK_FLAG =
    "void main()\n"
    "{\n"
    "  int brick = int(gl_GlobalInvocationID.x);\n"
    "  if (brick >= HA_BRICK_COUNT) { return; }\n"
    "  int base = brick * 64;\n"
    "  int has = 0;\n"
    "  for (int i = 0; i < 64; i++) {\n"
    "    if ((mask[base + i] & HA_UNKNOWN) != 0) { has = 1; break; }\n"
    "  }\n"
    "  flag[brick] = has;\n"
    "}\n";

const char *KERNEL_BRICK_COMPACT =
    "void main()\n"
    "{\n"
    "  if (int(gl_GlobalInvocationID.x) != 0) { return; }\n"
    "  int count = 0;\n"
    "  for (int brick = 0; brick < HA_BRICK_COUNT; brick++) {\n"
    "    if (flag[brick] != 0) {\n"
    "      brick_list[count] = brick;\n"
    "      count++;\n"
    "    }\n"
    "    else {\n"
    "      partial[brick * 4 + 0] = 0.0;\n"
    "      partial[brick * 4 + 1] = 0.0;\n"
    "      partial[brick * 4 + 2] = 0.0;\n"
    "      partial[brick * 4 + 3] = 0.0;\n"
    "    }\n"
    "  }\n"
    "  count_out[0] = count;\n"
    "}\n";

/** Index of the cell this invocation handles: from the brick list when the CG runs list mode. */
const char *GLSL_CG_LIST =
    "int ha_cg_index()\n"
    "{\n"
    "  if (HA_LIST_MODE != 0) {\n"
    "    return int(brick_list[gl_WorkGroupID.x]) * 64 + int(gl_LocalInvocationIndex);\n"
    "  }\n"
    "  return int(gl_GlobalInvocationID.x);\n"
    "}\n";

const char *KERNEL_RHS =
    "vec3 ha_vel_at(int mask_value, ivec3 c)\n"
    "{\n"
    "  if (!ha_inside(c) || (mask_value & HA_DOMAIN) == 0 || (mask_value & HA_SOLID) != 0) {\n"
    "    return vec3(0.0);\n"
    "  }\n"
    "  int base = ha_at(c) * 3;\n"
    "  return vec3(velocity[base + 0], velocity[base + 1], velocity[base + 2]);\n"
    "}\n"
    "float ha_divergence(ivec3 c)\n"
    "{\n"
    "  vec3 v = ha_vel_at(ha_mask_at(c), c);\n"
    "  return ((ha_vel_at(ha_mask_at(c + ivec3(1, 0, 0)), c + ivec3(1, 0, 0)).x - v.x) +\n"
    "          (ha_vel_at(ha_mask_at(c + ivec3(0, 1, 0)), c + ivec3(0, 1, 0)).y - v.y) +\n"
    "          (ha_vel_at(ha_mask_at(c + ivec3(0, 0, 1)), c + ivec3(0, 0, 1)).z - v.z)) /\n"
    "         HA_VOXEL;\n"
    "}\n"
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  if ((mask[index] & HA_UNKNOWN) != 0) {\n"
    "    rhs[index] = ha_divergence(c) * HA_VOXEL * HA_VOXEL;\n"
    "  }\n"
    "  else {\n"
    "    rhs[index] = 0.0;\n"
    "  }\n"
    "  if (index % 64 == 0) {\n"
    "    float sum = 0.0;\n"
    "    float count = 0.0;\n"
    "    int end = min(index + 64, HA_CELL_COUNT);\n"
    "    for (int j = index; j < end; j++) {\n"
    "      if ((mask[j] & HA_UNKNOWN) == 0) { continue; }\n"
    "      ivec3 cj;\n"
    "      ha_cell(j, cj);\n"
    "      float divergence = ha_divergence(cj);\n"
    "      sum += divergence * divergence;\n"
    "      count += 1.0;\n"
    "    }\n"
    "    partial[(index / 64) * 4 + 0] = sum;\n"
    "    partial[(index / 64) * 4 + 2] = count;\n"
    "  }\n"
    "}\n";

const char *KERNEL_CG_PRODUCT =
    "void main()\n"
    "{\n"
    "  int index = ha_cg_index();\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  if ((mask[index] & HA_UNKNOWN) == 0) { product[index] = 0.0; return; }\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  float sum = -6.0 * pressure[index];\n"
    "  int n;\n"
    "  n = ha_at(c + ivec3(1, 0, 0));\n"
    "  if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += pressure[n]; }\n"
    "  n = ha_at(c + ivec3(-1, 0, 0));\n"
    "  if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += pressure[n]; }\n"
    "  n = ha_at(c + ivec3(0, 1, 0));\n"
    "  if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += pressure[n]; }\n"
    "  n = ha_at(c + ivec3(0, -1, 0));\n"
    "  if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += pressure[n]; }\n"
    "  n = ha_at(c + ivec3(0, 0, 1));\n"
    "  if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += pressure[n]; }\n"
    "  n = ha_at(c + ivec3(0, 0, -1));\n"
    "  if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += pressure[n]; }\n"
    "  product[index] = sum;\n"
    "}\n";

const char *KERNEL_CG_INIT =
    "void main()\n"
    "{\n"
    "  int index = ha_cg_index();\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  float value = (mask[index] & HA_UNKNOWN) != 0 ? rhs[index] - product[index] : 0.0;\n"
    "  residual[index] = value;\n"
    "  direction[index] = value / 6.0;\n"
    "  if (index % 64 == 0) {\n"
    "    float sum = 0.0;\n"
    "    int end = min(index + 64, HA_CELL_COUNT);\n"
    "    for (int j = index; j < end; j++) {\n"
    "      float r = (mask[j] & HA_UNKNOWN) != 0 ? rhs[j] - product[j] : 0.0;\n"
    "      sum += r * r;\n"
    "    }\n"
    "    partial[(index / 64) * 4 + 1] = sum;\n"
    "  }\n"
    "}\n";

const char *KERNEL_REDUCE_INIT =
    "void main()\n"
    "{\n"
    "  if (int(gl_GlobalInvocationID.x) != 0) { return; }\n"
    "  float div_sum = 0.0;\n"
    "  float res_sum = 0.0;\n"
    "  float count = 0.0;\n"
    "  for (int b = 0; b < HA_PARTIAL_COUNT; b++) {\n"
    "    div_sum += partial[b * 4 + 0];\n"
    "    res_sum += partial[b * 4 + 1];\n"
    "    count += partial[b * 4 + 2];\n"
    "  }\n"
    "  params[49] = div_sum;\n"
    "  params[51] = count;\n"
    "  params[37] = sqrt(div_sum) * HA_VOXEL * HA_VOXEL;\n"
    "  params[38] = sqrt(res_sum);\n"
    "  params[39] = res_sum / 6.0;\n"
    "  params[36] = 0.0;\n"
    "  params[43] = (params[37] <= 1e-12 || params[38] <= HA_TOLERANCE * params[37]) ? 1.0 : "
    "0.0;\n"
    "}\n";

const char *KERNEL_CG_DIRECTION =
    "void main()\n"
    "{\n"
    "  int index = ha_cg_index();\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  if ((mask[index] & HA_UNKNOWN) == 0) { return; }\n"
    "  direction[index] = residual[index] / 6.0 + HA_BETA * direction[index];\n"
    "}\n";

/* The Laplacian pass also reduces `direction * product` per 64 cell work group (a shared memory
 * tree, fixed order, so it stays deterministic). That replaces the separate pq pass with a few
 * shared memory adds and one global write per group. `gl_LocalInvocationIndex` and the shared array
 * are sized by the dispatch group size (#LOCAL_GROUP_SIZE = 64), and every invocation reaches the
 * barriers, including the ones past the last voxel. */
const char *KERNEL_CG_LAPLACIAN =
    "shared float ha_block_pq[64];\n"
    "void main()\n"
    "{\n"
    "  int index = ha_cg_index();\n"
    "  int local = int(gl_LocalInvocationIndex);\n"
    "  ha_init_neighbors(index);\n"
    "  float pq = 0.0;\n"
    "  if (index < HA_CELL_COUNT) {\n"
    "    if ((mask[index] & HA_UNKNOWN) == 0) {\n"
    "      product[index] = 0.0;\n"
    "    }\n"
    "    else {\n"
    "      ivec3 c;\n"
    "      ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "      float sum = -6.0 * direction[index];\n"
    "      int n;\n"
    "      n = ha_at(c + ivec3(1, 0, 0));\n"
    "      if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += direction[n]; }\n"
    "      n = ha_at(c + ivec3(-1, 0, 0));\n"
    "      if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += direction[n]; }\n"
    "      n = ha_at(c + ivec3(0, 1, 0));\n"
    "      if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += direction[n]; }\n"
    "      n = ha_at(c + ivec3(0, -1, 0));\n"
    "      if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += direction[n]; }\n"
    "      n = ha_at(c + ivec3(0, 0, 1));\n"
    "      if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += direction[n]; }\n"
    "      n = ha_at(c + ivec3(0, 0, -1));\n"
    "      if (n >= 0 && (mask[n] & HA_UNKNOWN) != 0) { sum += direction[n]; }\n"
    "      product[index] = sum;\n"
    "      pq = direction[index] * sum;\n"
    "    }\n"
    "  }\n"
    "  ha_block_pq[local] = pq;\n"
    "  barrier();\n"
    "  for (int stride = 32; stride > 0; stride >>= 1) {\n"
    "    if (local < stride) { ha_block_pq[local] += ha_block_pq[local + stride]; }\n"
    "    barrier();\n"
    "  }\n"
    "  if (local == 0) {\n"
    "    partial[(index >> 6) * 4 + 0] = ha_block_pq[0];\n"
    "  }\n"
    "}\n";

const char *KERNEL_REDUCE_PQ =
    "void main()\n"
    "{\n"
    "  if (int(gl_GlobalInvocationID.x) != 0) { return; }\n"
    "  float pq = 0.0;\n"
    "  for (int b = 0; b < HA_PARTIAL_COUNT; b++) { pq += partial[b * 4 + 0]; }\n"
    "  params[40] = pq;\n"
    "  params[41] = abs(pq) < 1e-30 ? 0.0 : params[39] / pq;\n"
    "  if (abs(pq) < 1e-30) { params[43] = 1.0; }\n"
    "}\n";

const char *KERNEL_CG_UPDATE =
    "shared float ha_block_rr[64];\n"
    "void main()\n"
    "{\n"
    "  int index = ha_cg_index();\n"
    "  int local = int(gl_LocalInvocationIndex);\n"
    "  float rr = 0.0;\n"
    "  if (index < HA_CELL_COUNT && (mask[index] & HA_UNKNOWN) != 0) {\n"
    "    pressure[index] += HA_ALPHA * direction[index];\n"
    "    float value = residual[index] - HA_ALPHA * product[index];\n"
    "    residual[index] = value;\n"
    "    rr = value * value;\n"
    "  }\n"
    "  ha_block_rr[local] = rr;\n"
    "  barrier();\n"
    "  for (int stride = 32; stride > 0; stride >>= 1) {\n"
    "    if (local < stride) { ha_block_rr[local] += ha_block_rr[local + stride]; }\n"
    "    barrier();\n"
    "  }\n"
    "  if (local == 0) {\n"
    "    partial[(index >> 6) * 4 + 1] = ha_block_rr[0];\n"
    "  }\n"
    "}\n";

const char *KERNEL_REDUCE_RES =
    "void main()\n"
    "{\n"
    "  if (int(gl_GlobalInvocationID.x) != 0) { return; }\n"
    "  float res_sum = 0.0;\n"
    "  for (int b = 0; b < HA_PARTIAL_COUNT; b++) { res_sum += partial[b * 4 + 1]; }\n"
    "  params[38] = sqrt(res_sum);\n"
    "  params[36] = params[36] + 1.0;\n"
    "  if (params[38] <= HA_TOLERANCE * params[37]) { params[43] = 1.0; }\n"
    "  if (params[43] == 0.0) {\n"
    "    params[42] = (res_sum / 6.0) / max(params[39], 1e-30);\n"
    "    params[39] = res_sum / 6.0;\n"
    "  }\n"
    "}\n";

const char *KERNEL_CG_GRADIENT =
    "float ha_pressure_at(ivec3 c)\n"
    "{\n"
    "  int i = ha_at(c);\n"
    "  if (i < 0) { return 0.0; }\n"
    "  return (mask[i] & HA_UNKNOWN) != 0 ? pressure[i] : 0.0;\n"
    "}\n"
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  int m = mask[index];\n"
    "  if ((m & HA_DOMAIN) == 0 || (m & HA_SOLID) != 0) { return; }\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  float p = ha_pressure_at(c);\n"
    "  vec3 gradient = vec3(p - ha_pressure_at(c + ivec3(-1, 0, 0)),\n"
    "                       p - ha_pressure_at(c + ivec3(0, -1, 0)),\n"
    "                       p - ha_pressure_at(c + ivec3(0, 0, -1))) / HA_VOXEL;\n"
    "  int base = index * 3;\n"
    "  velocity[base + 0] -= gradient.x;\n"
    "  velocity[base + 1] -= gradient.y;\n"
    "  velocity[base + 2] -= gradient.z;\n"
    "}\n";

const char *KERNEL_DIVERGENCE_AFTER =
    "vec3 ha_vel_after(ivec3 c)\n"
    "{\n"
    "  if (!ha_inside(c) || (ha_mask_at(c) & HA_DOMAIN) == 0 ||\n"
    "      (ha_mask_at(c) & HA_SOLID) != 0) {\n"
    "    return vec3(0.0);\n"
    "  }\n"
    "  int base = ha_at(c) * 3;\n"
    "  return vec3(velocity[base + 0], velocity[base + 1], velocity[base + 2]);\n"
    "}\n"
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  if (index % 64 != 0) { return; }\n"
    "  float sum = 0.0;\n"
    "  int end = min(index + 64, HA_CELL_COUNT);\n"
    "  for (int j = index; j < end; j++) {\n"
    "    if ((mask[j] & HA_UNKNOWN) == 0) { continue; }\n"
    "    ivec3 cj;\n"
    "    ha_cell(j, cj);\n"
    "    vec3 v = ha_vel_after(cj);\n"
    "    float divergence = ((ha_vel_after(cj + ivec3(1, 0, 0)).x - v.x) +\n"
    "                        (ha_vel_after(cj + ivec3(0, 1, 0)).y - v.y) +\n"
    "                        (ha_vel_after(cj + ivec3(0, 0, 1)).z - v.z)) / HA_VOXEL;\n"
    "    sum += divergence * divergence;\n"
    "  }\n"
    "  partial[(index / 64) * 4 + 0] = sum;\n"
    "}\n";

const char *KERNEL_REDUCE_DIVERGENCE =
    "void main()\n"
    "{\n"
    "  if (int(gl_GlobalInvocationID.x) != 0) { return; }\n"
    "  float sum = 0.0;\n"
    "  for (int b = 0; b < HA_PARTIAL_COUNT; b++) { sum += partial[b * 4 + 0]; }\n"
    "  params[50] = sum;\n"
    "}\n";

/* The velocity clamp uses the *current* pose of the colliders, not the swept one: the sweep is what
 * keeps the smoke out of the path of a fast collider, while the wall velocity of a moving collider
 * is only meaningful where the collider is right now. The clip is against the wall velocity (a
 * relative normal velocity) instead of zero, so a wall that moves towards the smoke pushes it while
 * a wall that moves away lets it slide: imposing the wall velocity unconditionally would drag the
 * smoke along with a falling collider. The collider id is stored in bits 8..15 of the solid mask
 * (id + 1), so the wall component can be clamped against that collider's velocity. */
const char *KERNEL_CLAMP_SOLIDS =
    "int ha_solid_current_at(ivec3 c)\n"
    "{\n"
    "  int i = ha_at(c);\n"
    "  return i < 0 ? 0 : solid_current[i];\n"
    "}\n"
    "bool ha_collider_velocity(ivec3 c, out float vx, out float vy, out float vz)\n"
    "{\n"
    "  vx = 0.0;\n"
    "  vy = 0.0;\n"
    "  vz = 0.0;\n"
    "  if (!ha_inside(c)) { return false; }\n"
    "  int id = (solid_current[ha_at(c)] >> 8) & 0xff;\n"
    "  if (id == 0) { return false; }\n"
    "  int offset = (id - 1) * 3;\n"
    "  vx = collider_velocity[offset + 0];\n"
    "  vy = collider_velocity[offset + 1];\n"
    "  vz = collider_velocity[offset + 2];\n"
    "  return (vx * vx + vy * vy + vz * vz) > 0.0;\n"
    "}\n"
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  int base = index * 3;\n"
    "  if ((solid_current[index] & HA_SOLID) != 0) {\n"
    "    velocity[base + 0] = 0.0;\n"
    "    velocity[base + 1] = 0.0;\n"
    "    velocity[base + 2] = 0.0;\n"
    "    return;\n"
    "  }\n"
    "  vec3 v = vec3(velocity[base + 0], velocity[base + 1], velocity[base + 2]);\n"
    "  float wx, wy, wz;\n"
    "  if (ha_collider_velocity(c + ivec3(1, 0, 0), wx, wy, wz)) { v.x = min(v.x, wx); }\n"
    "  else if ((ha_solid_current_at(c + ivec3(1, 0, 0)) & HA_SOLID) != 0) { v.x = min(v.x, 0.0); }\n"
    "  if (ha_collider_velocity(c + ivec3(-1, 0, 0), wx, wy, wz)) { v.x = max(v.x, wx); }\n"
    "  else if ((ha_solid_current_at(c + ivec3(-1, 0, 0)) & HA_SOLID) != 0) { v.x = max(v.x, 0.0); }\n"
    "  if (ha_collider_velocity(c + ivec3(0, 1, 0), wx, wy, wz)) { v.y = min(v.y, wy); }\n"
    "  else if ((ha_solid_current_at(c + ivec3(0, 1, 0)) & HA_SOLID) != 0) { v.y = min(v.y, 0.0); }\n"
    "  if (ha_collider_velocity(c + ivec3(0, -1, 0), wx, wy, wz)) { v.y = max(v.y, wy); }\n"
    "  else if ((ha_solid_current_at(c + ivec3(0, -1, 0)) & HA_SOLID) != 0) { v.y = max(v.y, 0.0); }\n"
    "  if (ha_collider_velocity(c + ivec3(0, 0, 1), wx, wy, wz)) { v.z = min(v.z, wz); }\n"
    "  else if ((ha_solid_current_at(c + ivec3(0, 0, 1)) & HA_SOLID) != 0) { v.z = min(v.z, 0.0); }\n"
    "  if (ha_collider_velocity(c + ivec3(0, 0, -1), wx, wy, wz)) { v.z = max(v.z, wz); }\n"
    "  else if ((ha_solid_current_at(c + ivec3(0, 0, -1)) & HA_SOLID) != 0) { v.z = max(v.z, 0.0); }\n"
    "  velocity[base + 0] = v.x;\n"
    "  velocity[base + 1] = v.y;\n"
    "  velocity[base + 2] = v.z;\n"
    "}\n";

const char *KERNEL_COMBUST =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  float fuel = scalars[index * 4 + 2];\n"
    "  if (HA_BURN_RATE > 0.0 && fuel > 0.0) {\n"
    "    float burned = min(fuel, HA_BURN_RATE * HA_DT);\n"
    "    scalars[index * 4 + 2] = fuel - burned;\n"
    "    float heat = burned * HA_HEAT_RELEASE;\n"
    "    scalars[index * 4 + 1] += heat;\n"
    "    scalars[index * 4 + 0] += burned * HA_SMOKE_PER_BURN;\n"
    "    if (heat > 0.0) { scalars[index * 4 + 3] = max(scalars[index * 4 + 3], heat / HA_DT); }\n"
    "  }\n"
    "  scalars[index * 4 + 0] *= exp(-HA_DISSIPATION * HA_DT);\n"
    "  scalars[index * 4 + 1] *= exp(-HA_COOLING * HA_DT);\n"
    "}\n";

const char *KERNEL_CLIP_BOUNDS =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  if (HA_USE_BOUNDS == 0) { return; }\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  vec3 position = ha_world(c);\n"
    "  if (position.x < HA_BOUNDS_MIN.x || position.x > HA_BOUNDS_MAX.x ||\n"
    "      position.y < HA_BOUNDS_MIN.y || position.y > HA_BOUNDS_MAX.y ||\n"
    "      position.z < HA_BOUNDS_MIN.z || position.z > HA_BOUNDS_MAX.z) {\n"
    "    scalars[index * 4 + 0] = 0.0;\n"
    "    scalars[index * 4 + 1] = 0.0;\n"
    "    scalars[index * 4 + 2] = 0.0;\n"
    "  }\n"
    "}\n";

const char *KERNEL_MASK_KEEP_INIT =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  int m = solid[index];\n"
    "  if (scalars[index * 4 + 0] > 1e-4) { m = m | HA_SMOKE | HA_KEEP; }\n"
    "  mask_out[index] = m;\n"
    "}\n";

const char *KERNEL_EXTRAPOLATE_SETUP =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  int m = mask[index];\n"
    "  int base = index * 3;\n"
    "  vec3 v = vec3(velocity[base + 0], velocity[base + 1], velocity[base + 2]);\n"
    "  if ((m & HA_KEEP) == 0) {\n"
    "    velocity[base + 0] = 0.0;\n"
    "    velocity[base + 1] = 0.0;\n"
    "    velocity[base + 2] = 0.0;\n"
    "    m = m & ~HA_FILLED;\n"
    "  }\n"
    "  else if (dot(v, v) > 1e-12) {\n"
    "    m = m | HA_FILLED;\n"
    "  }\n"
    "  else {\n"
    "    m = m & ~HA_FILLED;\n"
    "  }\n"
    "  mask[index] = m;\n"
    "}\n";

const char *KERNEL_EXTRAPOLATE_PASS =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  ivec3 c;\n"
    "  ha_cell(index, c);\n"
    "  if (!ha_inside(c)) { return; }\n"
    "  int m = mask_in[index];\n"
    "  int out_mask = m;\n"
    "  if ((m & HA_KEEP) != 0 && (m & HA_FILLED) == 0) {\n"
    "    vec3 sum = vec3(0.0);\n"
    "    int count = 0;\n"
    "    if ((ha_mask_in_at(c + ivec3(1, 0, 0)) & HA_FILLED) != 0) {\n"
    "      sum += ha_velocity(c + ivec3(1, 0, 0));\n"
    "      count++;\n"
    "    }\n"
    "    if ((ha_mask_in_at(c + ivec3(-1, 0, 0)) & HA_FILLED) != 0) {\n"
    "      sum += ha_velocity(c + ivec3(-1, 0, 0));\n"
    "      count++;\n"
    "    }\n"
    "    if ((ha_mask_in_at(c + ivec3(0, 1, 0)) & HA_FILLED) != 0) {\n"
    "      sum += ha_velocity(c + ivec3(0, 1, 0));\n"
    "      count++;\n"
    "    }\n"
    "    if ((ha_mask_in_at(c + ivec3(0, -1, 0)) & HA_FILLED) != 0) {\n"
    "      sum += ha_velocity(c + ivec3(0, -1, 0));\n"
    "      count++;\n"
    "    }\n"
    "    if ((ha_mask_in_at(c + ivec3(0, 0, 1)) & HA_FILLED) != 0) {\n"
    "      sum += ha_velocity(c + ivec3(0, 0, 1));\n"
    "      count++;\n"
    "    }\n"
    "    if ((ha_mask_in_at(c + ivec3(0, 0, -1)) & HA_FILLED) != 0) {\n"
    "      sum += ha_velocity(c + ivec3(0, 0, -1));\n"
    "      count++;\n"
    "    }\n"
    "    if (count > 0) {\n"
    "      ha_velocity_set(c, sum / float(count));\n"
    "      out_mask = m | HA_FILLED;\n"
    "    }\n"
    "  }\n"
    "  mask_out[index] = out_mask;\n"
    "}\n";

const char *KERNEL_CARVE_SOLIDS =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  if ((solid_carve[index] & HA_SOLID) == 0) { return; }\n"
    "  scalars[index * 4 + 0] = 0.0;\n"
    "  scalars[index * 4 + 1] = 0.0;\n"
    "  scalars[index * 4 + 2] = 0.0;\n"
    "  int base = index * 3;\n"
    "  velocity[base + 0] = 0.0;\n"
    "  velocity[base + 1] = 0.0;\n"
    "  velocity[base + 2] = 0.0;\n"
    "}\n";

const char *KERNEL_FLAME_FILTER =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  scalars[index * 4 + 3] = max(0.0, scalars[index * 4 + 3] - HA_FLAME_THRESHOLD);\n"
    "}\n";

const char *KERNEL_DOMAIN_MASK =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= HA_CELL_COUNT) { return; }\n"
    "  ha_init_neighbors(index);\n"
    "  if (HA_USE_DOMAIN_MASK == 0) { return; }\n"
    "  if (abs(scalars[index * 4 + 0]) <= 1e-6) { return; }\n"
    "  float value = 0.0;\n"
    "  if (HA_DOMAIN_FIELD == 0) { value = scalars[index * 4 + 0]; }\n"
    "  else if (HA_DOMAIN_FIELD == 1) { value = scalars[index * 4 + 1]; }\n"
    "  else if (HA_DOMAIN_FIELD == 2) { value = scalars[index * 4 + 2]; }\n"
    "  else { value = scalars[index * 4 + 3]; }\n"
    "  if (abs(value) < HA_DOMAIN_THRESHOLD) {\n"
    "    scalars[index * 4 + 0] = 0.0;\n"
    "    scalars[index * 4 + 1] = 0.0;\n"
    "    scalars[index * 4 + 2] = 0.0;\n"
    "    scalars[index * 4 + 3] = 0.0;\n"
    "    int base = index * 3;\n"
    "    velocity[base + 0] = 0.0;\n"
    "    velocity[base + 1] = 0.0;\n"
    "    velocity[base + 2] = 0.0;\n"
    "    pressure[index] = 0.0;\n"
    "    mask[index] = mask[index] | HA_REMOVED;\n"
    "  }\n"
    "}\n";

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Brick grid (host side)
 *
 * The 4^3 brick pool of `doc/ignis-brick-sparse.md`, built on the host from the dense frame. The
 * device sees it through the param buffer appended after `P_COUNT` (tile -> slot map and brick
 * origins), so the kernels need no extra storage buffer.
 * \{ */

constexpr int IGNIS_BRICK_EDGE = 4;
constexpr int IGNIS_BRICK_CELLS = IGNIS_BRICK_EDGE * IGNIS_BRICK_EDGE * IGNIS_BRICK_EDGE;

struct IgnisBrickGrid {
  int3 dims = int3(0);
  int3 brick_dims = int3(0);
  int tile_count = 0;
  int brick_count = 0;
  /** Brick origin in voxels, sorted by (z, y, x) so the slot order is deterministic. */
  Vector<int3> coords;
  /** Brick coordinate -> slot, or -1. Indexed by `x + bx * (y + by * z)`. */
  Vector<int> slot_map;
  /** `slot * 6 + axis * 2 + (negative ? 1 : 0)`: the face neighbour's slot, or -1. */
  Vector<int> neighbor;
  int64_t pool_cells = 0;
};

static IgnisBrickGrid ignis_bricks_build(const IgnisDeviceFrame &frame,
                                         const int margin_voxels,
                                         const bool force_dense);
static void bricks_scatter_float(const IgnisBrickGrid &bricks,
                                 Span<const float> src,
                                 int stride,
                                 MutableSpan<float> dst);
static void bricks_gather_float(const IgnisBrickGrid &bricks,
                                Span<const float> src,
                                int stride,
                                MutableSpan<float> dst);
static void bricks_scatter_int(const IgnisBrickGrid &bricks,
                               Span<const int> src,
                               MutableSpan<int> dst);
static void bricks_gather_int(const IgnisBrickGrid &bricks,
                              Span<const int> src,
                              MutableSpan<int> dst);
/** Host side brick grid self test, defined at the end of this file. */
static void ignis_bricks_selftest(const IgnisDeviceFrame &frame,
                                  const int margin_voxels,
                                  const bool force_dense);

/** \} */

/* -------------------------------------------------------------------- */
/** \name Device solver
 * \{ */

constexpr int GROUP_SIZE = 64;

double time_now_ms()
{
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool debug_enabled()
{
  const char *value = std::getenv("BLENDER_IGNIS_HA_DEBUG");
  return value != nullptr && value[0] != '0' && value[0] != '\0';
}

void debug_log(const char *format, ...)
{
  if (!debug_enabled()) {
    return;
  }
  /* stderr survives a crash, stdout is block buffered when piped. */
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fflush(stderr);
}

class HaIgnisSolver {
 public:
  bool solve(IgnisDeviceFrame &frame, IgnisDeviceResult &r_result);
  /** Frees the device buffers; call while a #HaContextGuard is still active. */
  void release();

 private:
  bool allocate(const int64_t cells, const int64_t blocks, std::string &r_reason);
  bool bind(const IgnisDeviceFrame &frame, std::string &r_reason);
  void fill_params(const IgnisDeviceFrame &frame, const float dt, MutableSpan<float> r_params);
  void write_adv(const float dir,
                 const float order,
                 const float limiter,
                 const float component,
                 const float correct_mode);
  bool pass(gpu::Shader *shader, Span<gpu::StorageBuf *> buffers, const int count);
  void swap_masks();

  bool projection(const int64_t cells, std::string &r_reason);
  /** Two level coarse grid initial guess for the fine pressure CG. */
  bool coarse_pressure_guess(const int iterations, std::string &r_reason);
  void upload_coarse_params(Span<const float> fine_params);
  bool advect_velocity(const int64_t cells, std::string &r_reason);
  bool advect_scalars(const int64_t cells, std::string &r_reason);
  bool extrapolate(const int64_t cells, std::string &r_reason);
  bool dilate_mask(const int radius, const int bit_index, std::string &r_reason);

  gpu::Shader *shader(const char *key,
                      const std::string &source,
                      Span<const HaShaderResource> resources);
  bool shaders_valid() const;
  void create_shaders();

  HaBuffer scalars_;
  HaBuffer velocity_;
  HaBuffer velocity_tmp_;
  HaBuffer velocity_aux_;
  HaBuffer scalar_tmp_;
  HaBuffer scalar_aux_;
  HaBuffer scalar_pack_;
  HaBuffer pressure_;
  HaBuffer rhs_;
  HaBuffer residual_;
  HaBuffer direction_;
  HaBuffer product_;
  HaBuffer curl_;
  HaBuffer source_;
  HaBuffer params_;
  HaBuffer adv_;
  HaBuffer partial_;
  HaBuffer coarse_params_;
  HaBuffer coarse_pressure_;
  HaBuffer coarse_rhs_;
  HaBuffer coarse_residual_;
  HaBuffer coarse_direction_;
  HaBuffer coarse_product_;
  HaBuffer coarse_partial_;
  HaBufferInt coarse_mask_;
  HaBufferInt brick_flags_;
  HaBufferInt brick_list_;
  HaBufferInt brick_list_count_;
  HaBufferInt mask_a_;
  HaBufferInt mask_b_;
  HaBufferInt solid_;
  HaBufferInt solid_current_;
  HaBufferInt solid_carve_;
  HaBuffer collider_velocity_;

  HaBufferInt *mask_ = nullptr;
  HaBufferInt *mask_other_ = nullptr;
  int64_t cells_ = 0;
  int64_t blocks_ = 0;
  /** Brick pool of the current frame (`doc/ignis-brick-sparse.md`). The parameter buffer gets the
   * brick map and coordinates appended after `P_COUNT`. */
  IgnisBrickGrid bricks_;
  int64_t pool_cells_ = 0;
  int64_t pool_blocks_ = 0;
  int brick_block_ = 0;
  /** Host copy of the param buffer of the current frame, including the brick block appended after
   * `P_COUNT`. Every `params_` upload has to come from this array: uploading a `P_COUNT` sized span
   * pads the rest with zeros and would clear the brick addressing (`dilate_mask` did exactly that). */
  Array<float> params_host_;
  /** Dispatch the fine CG over the bricks that contain an unknown cell instead of the whole pool. */
  bool cg_list_ = true;
  int64_t coarse_cells_ = 0;
  int64_t coarse_blocks_ = 0;
  int coarse_iterations_ = 12;
  /* How many fine CG iterations run between two convergence read backs (each read back waits for
   * the outstanding dispatches). */
  int cg_check_interval_ = 8;
  int3 coarse_dims_ = int3(0);
  int64_t radius_ = 0;
  ignis::Settings settings_;
  IgnisDeviceResult *result_ = nullptr;

  /* Shaders, created once per solve call (the backend caches the compiled programs by key). */
  gpu::Shader *source_merge_ = nullptr;
  gpu::Shader *forces_ = nullptr;
  gpu::Shader *vorticity_curl_ = nullptr;
  gpu::Shader *vorticity_apply_ = nullptr;
  gpu::Shader *advect_vec3_ = nullptr;
  gpu::Shader *advect_scalar_ = nullptr;
  gpu::Shader *advect_scalar_tmp_ = nullptr;
  gpu::Shader *correct_vec3_ = nullptr;
  gpu::Shader *correct_scalar_ = nullptr;
  gpu::Shader *limiter_vec3_ = nullptr;
  gpu::Shader *limiter_scalar_ = nullptr;
  gpu::Shader *copy_vec3_ = nullptr;
  gpu::Shader *tmp_to_scalar_ = nullptr;
  gpu::Shader *pack_scalars_ = nullptr;
  gpu::Shader *unpack_scalars_ = nullptr;
  gpu::Shader *mask_init_ = nullptr;
  gpu::Shader *mask_keep_init_ = nullptr;
  gpu::Shader *dilate_ = nullptr;
  gpu::Shader *unknown_ = nullptr;
  gpu::Shader *coarse_restrict_ = nullptr;
  gpu::Shader *coarse_prolong_ = nullptr;
  gpu::Shader *zero_field_ = nullptr;
  gpu::Shader *rhs_shader_ = nullptr;
  gpu::Shader *cg_product_ = nullptr;
  gpu::Shader *brick_flag_ = nullptr;
  gpu::Shader *brick_compact_ = nullptr;
  gpu::Shader *cg_init_ = nullptr;
  gpu::Shader *reduce_init_ = nullptr;
  gpu::Shader *cg_direction_ = nullptr;
  gpu::Shader *cg_laplacian_ = nullptr;
  gpu::Shader *reduce_pq_ = nullptr;
  gpu::Shader *cg_update_ = nullptr;
  gpu::Shader *reduce_res_ = nullptr;
  gpu::Shader *cg_gradient_ = nullptr;
  gpu::Shader *divergence_after_ = nullptr;
  gpu::Shader *reduce_divergence_ = nullptr;
  gpu::Shader *clamp_solids_ = nullptr;
  gpu::Shader *combust_ = nullptr;
  gpu::Shader *clip_bounds_ = nullptr;
  gpu::Shader *extrapolate_setup_ = nullptr;
  gpu::Shader *extrapolate_pass_ = nullptr;
  gpu::Shader *carve_solids_ = nullptr;
  gpu::Shader *flame_filter_ = nullptr;
  gpu::Shader *domain_mask_ = nullptr;
};

gpu::Shader *HaIgnisSolver::shader(const char *key,
                                   const std::string &source,
                                   const Span<const HaShaderResource> resources)
{
  return ha_shader_get_source(key, source, resources, GROUP_SIZE);
}

void HaIgnisSolver::create_shaders()
{
  if (advect_vec3_ != nullptr) {
    return;
  }
  const std::string common = std::string(GLSL_PRELUDE);

  static const HaShaderResource source_merge_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_RW(1, "float", "scalars[]"),
      HA_RW(2, "float", "velocity[]"),
      HA_READ(3, "float", "source[]"),
  };
  source_merge_ = shader("ignis|source_merge",
                         common + KERNEL_SOURCE_MERGE,
                         Span<const HaShaderResource>(source_merge_res, 4));

  static const HaShaderResource forces_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "float", "scalars[]"),
      HA_RW(2, "float", "velocity[]"),
      HA_READ(3, "float", "pressure[]"),
  };
  forces_ = shader("ignis|forces",
                   common + GLSL_FORCE_HELPERS + KERNEL_FORCES,
                   Span<const HaShaderResource>(forces_res, 4));

  static const HaShaderResource curl_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "float", "scalars[]"),
      HA_READ(2, "float", "velocity[]"),
      HA_RW(3, "int", "mask[]"),
      HA_WRITE(4, "float", "curl[]"),
  };
  const std::string curl_source = common + GLSL_SCALAR_HELPERS + GLSL_VELOCITY_HELPERS +
                                  KERNEL_VORTICITY_CURL;
  vorticity_curl_ = shader(
      "ignis|vorticity_curl", curl_source, Span<const HaShaderResource>(curl_res, 5));

  static const HaShaderResource apply_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_RW(1, "float", "velocity[]"),
      HA_READ(2, "int", "mask[]"),
      HA_READ(3, "float", "curl[]"),
  };
  const std::string apply_source = common + GLSL_MASK_HELPERS + GLSL_VELOCITY_HELPERS +
                                   KERNEL_VORTICITY_APPLY;
  vorticity_apply_ = shader(
      "ignis|vorticity_apply", apply_source, Span<const HaShaderResource>(apply_res, 4));

  /* Advection: the same kernel for the velocity field (3 floats) and the scalar fields (the
   * interleaved 4 float buffer); `HA_COMPONENT` selects the scalar component at runtime. */
  {
    static const HaShaderResource advect_vec3_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "adv[]"),
        HA_READ(2, "float", "field_in[]"),
        HA_WRITE(3, "float", "field_out[]"),
        HA_READ(4, "float", "velocity[]"),
    };
    const std::string source = std::string(GLSL_PRELUDE) + GLSL_ADVECT_CONTROLS +
                               GLSL_VELOCITY_HELPERS + GLSL_ADVECT_VEC3_FIELD +
                               GLSL_ADVECT_SAMPLE_VEC3 + KERNEL_ADVECT_PASS;
    advect_vec3_ = shader(
        "ignis|advect_vec3", source, Span<const HaShaderResource>(advect_vec3_res, 5));
  }
  {
    static const HaShaderResource advect_scalar_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "adv[]"),
        HA_READ(2, "float", "field_in[]"),
        HA_WRITE(3, "float", "field_out[]"),
        HA_READ(4, "float", "velocity[]"),
    };
    const std::string source = std::string(GLSL_PRELUDE) + GLSL_ADVECT_CONTROLS +
                               GLSL_VELOCITY_HELPERS + GLSL_ADVECT_SCALAR_FIELD +
                               GLSL_ADVECT_SAMPLE_SCALAR + KERNEL_ADVECT_PASS;
    advect_scalar_ = shader(
        "ignis|advect_scalar", source, Span<const HaShaderResource>(advect_scalar_res, 5));
  }
  {
    static const HaShaderResource advect_scalar_tmp_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "adv[]"),
        HA_READ(2, "float", "field_in[]"),
        HA_WRITE(3, "float", "field_out[]"),
        HA_READ(4, "float", "velocity[]"),
    };
    const std::string source = std::string(GLSL_PRELUDE) + GLSL_ADVECT_CONTROLS +
                               GLSL_VELOCITY_HELPERS + GLSL_ADVECT_PLAIN_FIELD +
                               GLSL_ADVECT_SAMPLE_SCALAR + KERNEL_ADVECT_PASS;
    advect_scalar_tmp_ = shader(
        "ignis|advect_scalar_tmp", source, Span<const HaShaderResource>(advect_scalar_tmp_res, 5));
  }
  {
    static const HaShaderResource correct_vec3_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "adv[]"),
        HA_READ(2, "float", "field_in[]"),
        HA_RW(3, "float", "field_out[]"),
        HA_READ(4, "float", "field_aux[]"),
    };
    const std::string source = std::string(GLSL_PRELUDE) + GLSL_ADVECT_CONTROLS +
                               GLSL_ADVECT_VEC3_FIELD + GLSL_ADVECT_VEC3_AUX +
                               KERNEL_ADVECT_CORRECT_VEC3;
    correct_vec3_ = shader(
        "ignis|correct_vec3", source, Span<const HaShaderResource>(correct_vec3_res, 5));
  }
  {
    static const HaShaderResource correct_scalar_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "adv[]"),
        HA_READ(2, "float", "field_in[]"),
        HA_RW(3, "float", "field_out[]"),
        HA_READ(4, "float", "field_aux[]"),
    };
    const std::string source = std::string(GLSL_PRELUDE) + GLSL_ADVECT_CONTROLS +
                               GLSL_ADVECT_SCALAR_FIELD + GLSL_ADVECT_SCALAR_AUX +
                               KERNEL_ADVECT_CORRECT_SCALAR;
    correct_scalar_ = shader(
        "ignis|correct_scalar", source, Span<const HaShaderResource>(correct_scalar_res, 5));
  }
  {
    static const HaShaderResource limiter_vec3_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "adv[]"),
        HA_READ(2, "float", "field_in[]"),
        HA_RW(3, "float", "field_out[]"),
        HA_READ(4, "float", "velocity[]"),
    };
    const std::string source = std::string(GLSL_PRELUDE) + GLSL_ADVECT_CONTROLS +
                               GLSL_VELOCITY_HELPERS + GLSL_ADVECT_VEC3_FIELD +
                               KERNEL_ADVECT_LIMITER_VEC3;
    limiter_vec3_ = shader(
        "ignis|limiter_vec3", source, Span<const HaShaderResource>(limiter_vec3_res, 5));
  }
  {
    static const HaShaderResource limiter_scalar_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "adv[]"),
        HA_READ(2, "float", "field_in[]"),
        HA_RW(3, "float", "field_out[]"),
        HA_READ(4, "float", "velocity[]"),
    };
    const std::string source = std::string(GLSL_PRELUDE) + GLSL_ADVECT_CONTROLS +
                               GLSL_VELOCITY_HELPERS + GLSL_ADVECT_SCALAR_FIELD +
                               KERNEL_ADVECT_LIMITER_SCALAR;
    limiter_scalar_ = shader(
        "ignis|limiter_scalar", source, Span<const HaShaderResource>(limiter_scalar_res, 5));
  }
  {
    static const HaShaderResource copy_vec3_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "field_in[]"),
        HA_WRITE(2, "float", "field_out[]"),
    };
    copy_vec3_ = shader("ignis|copy_vec3",
                        common + KERNEL_COPY_VEC3,
                        Span<const HaShaderResource>(copy_vec3_res, 3));
  }
  {
    static const HaShaderResource tmp_to_scalar_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "adv[]"),
        HA_READ(2, "float", "field_in[]"),
        HA_RW(3, "float", "field_out[]"),
    };
    tmp_to_scalar_ = shader("ignis|tmp_to_scalar",
                            std::string(GLSL_PRELUDE) + GLSL_ADVECT_CONTROLS +
                                KERNEL_TMP_TO_SCALAR,
                            Span<const HaShaderResource>(tmp_to_scalar_res, 4));
  }
  {
    static const HaShaderResource pack_scalars_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "scalars[]"),
        HA_WRITE(2, "float", "packed[]"),
    };
    pack_scalars_ = shader("ignis|pack_scalars",
                           std::string(GLSL_PRELUDE) + KERNEL_PACK_SCALARS,
                           Span<const HaShaderResource>(pack_scalars_res, 3));
  }
  {
    static const HaShaderResource unpack_scalars_res[] = {
        HA_READ(0, "float", "params[]"),
        HA_READ(1, "float", "packed[]"),
        HA_RW(2, "float", "scalars[]"),
    };
    unpack_scalars_ = shader("ignis|unpack_scalars",
                             std::string(GLSL_PRELUDE) + KERNEL_UNPACK_SCALARS,
                             Span<const HaShaderResource>(unpack_scalars_res, 3));
  }

  static const HaShaderResource mask_init_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "float", "scalars[]"),
      HA_READ(2, "int", "solid[]"),
      HA_WRITE(3, "int", "mask_out[]"),
  };
  mask_init_ = shader("ignis|mask_init",
                      common + KERNEL_MASK_INIT,
                      Span<const HaShaderResource>(mask_init_res, 4));
  mask_keep_init_ = shader("ignis|mask_keep_init",
                           common + KERNEL_MASK_KEEP_INIT,
                           Span<const HaShaderResource>(mask_init_res, 4));

  static const HaShaderResource dilate_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "mask_in[]"),
      HA_WRITE(2, "int", "mask_out[]"),
  };
  dilate_ = shader("ignis|dilate",
                   common + GLSL_MASK_IN_HELPERS + KERNEL_DILATE,
                   Span<const HaShaderResource>(dilate_res, 3));

  static const HaShaderResource unknown_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_RW(1, "int", "mask[]"),
  };
  unknown_ = shader("ignis|unknown",
                    common + GLSL_MASK_HELPERS + KERNEL_UNKNOWN,
                    Span<const HaShaderResource>(unknown_res, 2));

  static const HaShaderResource coarse_restrict_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "mask_in[]"),
      HA_READ(2, "float", "rhs_in[]"),
      HA_WRITE(3, "int", "mask_out[]"),
      HA_WRITE(4, "float", "rhs_out[]"),
  };
  coarse_restrict_ = shader("ignis|coarse_restrict",
                            common + KERNEL_RESTRICT_COARSE,
                            Span<const HaShaderResource>(coarse_restrict_res, 5));

  static const HaShaderResource coarse_prolong_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "mask[]"),
      HA_READ(2, "float", "coarse_pressure[]"),
      HA_RW(3, "float", "pressure[]"),
  };
  coarse_prolong_ = shader("ignis|coarse_prolong",
                           common + KERNEL_PROLONG_COARSE,
                           Span<const HaShaderResource>(coarse_prolong_res, 4));

  static const HaShaderResource zero_field_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_RW(1, "float", "field[]"),
  };
  zero_field_ = shader("ignis|zero_field",
                       common + KERNEL_ZERO_FIELD,
                       Span<const HaShaderResource>(zero_field_res, 2));

  static const HaShaderResource rhs_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "float", "velocity[]"),
      HA_READ(2, "int", "mask[]"),
      HA_WRITE(3, "float", "rhs[]"),
      HA_RW(4, "float", "partial[]"),
  };
  rhs_shader_ = shader("ignis|rhs",
                       common + GLSL_MASK_HELPERS + KERNEL_RHS,
                       Span<const HaShaderResource>(rhs_res, 5));

  static const HaShaderResource brick_flag_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "mask[]"),
      HA_WRITE(2, "int", "flag[]"),
  };
  brick_flag_ = shader("ignis|brick_flag",
                       common + KERNEL_BRICK_FLAG,
                       Span<const HaShaderResource>(brick_flag_res, 3));

  static const HaShaderResource brick_compact_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "flag[]"),
      HA_WRITE(2, "int", "brick_list[]"),
      HA_RW(3, "float", "partial[]"),
      HA_WRITE(4, "int", "count_out[]"),
  };
  brick_compact_ = shader("ignis|brick_compact",
                          common + KERNEL_BRICK_COMPACT,
                          Span<const HaShaderResource>(brick_compact_res, 5));

  static const HaShaderResource cg_product_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "float", "pressure[]"),
      HA_READ(2, "int", "mask[]"),
      HA_WRITE(3, "float", "product[]"),
      HA_READ(4, "int", "brick_list[]"),
  };
  cg_product_ = shader("ignis|cg_product",
                       common + GLSL_MASK_HELPERS + GLSL_CG_LIST + KERNEL_CG_PRODUCT,
                       Span<const HaShaderResource>(cg_product_res, 5));

  static const HaShaderResource cg_init_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "float", "rhs[]"),
      HA_READ(2, "float", "product[]"),
      HA_READ(3, "int", "mask[]"),
      HA_WRITE(4, "float", "residual[]"),
      HA_WRITE(5, "float", "direction[]"),
      HA_RW(6, "float", "partial[]"),
      HA_READ(7, "int", "brick_list[]"),
  };
  cg_init_ = shader(
      "ignis|cg_init",
      common + GLSL_CG_LIST + KERNEL_CG_INIT,
      Span<const HaShaderResource>(cg_init_res, 8));

  static const HaShaderResource reduce_init_res[] = {
      HA_RW(0, "float", "params[]"),
      HA_READ(1, "float", "partial[]"),
  };
  reduce_init_ = shader("ignis|reduce_init",
                        common + KERNEL_REDUCE_INIT,
                        Span<const HaShaderResource>(reduce_init_res, 2));

  static const HaShaderResource cg_direction_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "mask[]"),
      HA_READ(2, "float", "residual[]"),
      HA_RW(3, "float", "direction[]"),
      HA_READ(4, "int", "brick_list[]"),
  };
  cg_direction_ = shader("ignis|cg_direction",
                         common + GLSL_CG_LIST + KERNEL_CG_DIRECTION,
                         Span<const HaShaderResource>(cg_direction_res, 5));

  static const HaShaderResource cg_laplacian_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "mask[]"),
      HA_READ(2, "float", "direction[]"),
      HA_RW(3, "float", "product[]"),
      HA_RW(4, "float", "partial[]"),
      HA_READ(5, "int", "brick_list[]"),
  };
  cg_laplacian_ = shader("ignis|cg_laplacian",
                         common + GLSL_MASK_HELPERS + GLSL_CG_LIST + KERNEL_CG_LAPLACIAN,
                         Span<const HaShaderResource>(cg_laplacian_res, 6));

  static const HaShaderResource reduce_pq_res[] = {
      HA_RW(0, "float", "params[]"),
      HA_READ(1, "float", "partial[]"),
  };
  reduce_pq_ = shader("ignis|reduce_pq",
                      common + KERNEL_REDUCE_PQ,
                      Span<const HaShaderResource>(reduce_pq_res, 2));

  static const HaShaderResource cg_update_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "mask[]"),
      HA_READ(2, "float", "direction[]"),
      HA_READ(3, "float", "product[]"),
      HA_RW(4, "float", "pressure[]"),
      HA_RW(5, "float", "residual[]"),
      HA_RW(6, "float", "partial[]"),
      HA_READ(7, "int", "brick_list[]"),
  };
  cg_update_ = shader("ignis|cg_update",
                      common + GLSL_CG_LIST + KERNEL_CG_UPDATE,
                      Span<const HaShaderResource>(cg_update_res, 8));

  static const HaShaderResource reduce_res_res[] = {
      HA_RW(0, "float", "params[]"),
      HA_READ(1, "float", "partial[]"),
  };
  reduce_res_ = shader("ignis|reduce_res",
                       common + KERNEL_REDUCE_RES,
                       Span<const HaShaderResource>(reduce_res_res, 2));

  static const HaShaderResource cg_gradient_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "mask[]"),
      HA_READ(2, "float", "pressure[]"),
      HA_RW(3, "float", "velocity[]"),
  };
  cg_gradient_ = shader("ignis|cg_gradient",
                        common + GLSL_MASK_HELPERS + KERNEL_CG_GRADIENT,
                        Span<const HaShaderResource>(cg_gradient_res, 4));

  static const HaShaderResource divergence_after_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "mask[]"),
      HA_READ(2, "float", "velocity[]"),
      HA_RW(3, "float", "partial[]"),
  };
  divergence_after_ = shader("ignis|divergence_after",
                             common + GLSL_MASK_HELPERS + KERNEL_DIVERGENCE_AFTER,
                             Span<const HaShaderResource>(divergence_after_res, 4));

  static const HaShaderResource reduce_divergence_res[] = {
      HA_RW(0, "float", "params[]"),
      HA_READ(1, "float", "partial[]"),
  };
  reduce_divergence_ = shader("ignis|reduce_divergence",
                              common + KERNEL_REDUCE_DIVERGENCE,
                              Span<const HaShaderResource>(reduce_divergence_res, 2));

  static const HaShaderResource clamp_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "solid_current[]"),
      HA_READ(2, "float", "collider_velocity[]"),
      HA_RW(3, "float", "velocity[]"),
  };
  clamp_solids_ = shader("ignis|clamp_solids",
                         common + KERNEL_CLAMP_SOLIDS,
                         Span<const HaShaderResource>(clamp_res, 4));

  static const HaShaderResource combust_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_RW(1, "float", "scalars[]"),
  };
  combust_ = shader(
      "ignis|combust", common + KERNEL_COMBUST, Span<const HaShaderResource>(combust_res, 2));

  static const HaShaderResource clip_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_RW(1, "float", "scalars[]"),
  };
  clip_bounds_ = shader(
      "ignis|clip_bounds", common + KERNEL_CLIP_BOUNDS, Span<const HaShaderResource>(clip_res, 2));

  static const HaShaderResource extrapolate_setup_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_RW(1, "int", "mask[]"),
      HA_RW(2, "float", "velocity[]"),
  };
  extrapolate_setup_ = shader("ignis|extrapolate_setup",
                              common + KERNEL_EXTRAPOLATE_SETUP,
                              Span<const HaShaderResource>(extrapolate_setup_res, 3));

  static const HaShaderResource extrapolate_pass_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "mask_in[]"),
      HA_WRITE(2, "int", "mask_out[]"),
      HA_RW(3, "float", "velocity[]"),
  };
  extrapolate_pass_ = shader("ignis|extrapolate_pass",
                             common + GLSL_MASK_IN_HELPERS + GLSL_VELOCITY_HELPERS +
                                 KERNEL_EXTRAPOLATE_PASS,
                             Span<const HaShaderResource>(extrapolate_pass_res, 4));

  static const HaShaderResource carve_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_READ(1, "int", "solid_carve[]"),
      HA_RW(2, "float", "scalars[]"),
      HA_RW(3, "float", "velocity[]"),
  };
  carve_solids_ = shader("ignis|carve_solids",
                         common + KERNEL_CARVE_SOLIDS,
                         Span<const HaShaderResource>(carve_res, 4));

  static const HaShaderResource flame_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_RW(1, "float", "scalars[]"),
  };
  flame_filter_ = shader("ignis|flame_filter",
                         common + KERNEL_FLAME_FILTER,
                         Span<const HaShaderResource>(flame_res, 2));

  static const HaShaderResource domain_mask_res[] = {
      HA_READ(0, "float", "params[]"),
      HA_RW(1, "float", "scalars[]"),
      HA_RW(2, "float", "velocity[]"),
      HA_RW(3, "float", "pressure[]"),
      HA_RW(4, "int", "mask[]"),
  };
  domain_mask_ = shader("ignis|domain_mask",
                        common + KERNEL_DOMAIN_MASK,
                        Span<const HaShaderResource>(domain_mask_res, 5));
}

bool HaIgnisSolver::shaders_valid() const
{
  return source_merge_ != nullptr && forces_ != nullptr && vorticity_curl_ != nullptr &&
         vorticity_apply_ != nullptr && advect_vec3_ != nullptr && advect_scalar_ != nullptr &&
         advect_scalar_tmp_ != nullptr && correct_vec3_ != nullptr && correct_scalar_ != nullptr &&
         limiter_vec3_ != nullptr && limiter_scalar_ != nullptr && copy_vec3_ != nullptr &&
         tmp_to_scalar_ != nullptr && pack_scalars_ != nullptr && unpack_scalars_ != nullptr &&
         mask_init_ != nullptr && mask_keep_init_ != nullptr &&
         dilate_ != nullptr && unknown_ != nullptr && rhs_shader_ != nullptr &&
          coarse_restrict_ != nullptr && coarse_prolong_ != nullptr &&
          zero_field_ != nullptr &&
         cg_product_ != nullptr && cg_init_ != nullptr && reduce_init_ != nullptr &&
          brick_flag_ != nullptr && brick_compact_ != nullptr &&
         cg_direction_ != nullptr && cg_laplacian_ != nullptr &&
         reduce_pq_ != nullptr && cg_update_ != nullptr &&
         reduce_res_ != nullptr && cg_gradient_ != nullptr && divergence_after_ != nullptr &&
         reduce_divergence_ != nullptr && clamp_solids_ != nullptr && combust_ != nullptr &&
         clip_bounds_ != nullptr && extrapolate_setup_ != nullptr &&
         extrapolate_pass_ != nullptr && carve_solids_ != nullptr && flame_filter_ != nullptr &&
         domain_mask_ != nullptr;
}

void HaIgnisSolver::release()
{
  scalars_.free();
  velocity_.free();
  velocity_tmp_.free();
  velocity_aux_.free();
  scalar_tmp_.free();
  scalar_aux_.free();
  scalar_pack_.free();
  pressure_.free();
  rhs_.free();
  residual_.free();
  direction_.free();
  product_.free();
  curl_.free();
  source_.free();
  params_.free();
  adv_.free();
  partial_.free();
  coarse_params_.free();
  coarse_pressure_.free();
  coarse_rhs_.free();
  coarse_residual_.free();
  coarse_direction_.free();
  coarse_product_.free();
  coarse_partial_.free();
  coarse_mask_.free();
  brick_flags_.free();
  brick_list_.free();
  brick_list_count_.free();
  mask_a_.free();
  mask_b_.free();
  solid_.free();
  solid_current_.free();
  solid_carve_.free();
  collider_velocity_.free();
  mask_ = nullptr;
  mask_other_ = nullptr;
}

bool HaIgnisSolver::allocate(const int64_t cells, const int64_t blocks, std::string &r_reason)
{
  if (!scalars_.ensure(cells * 4) || !velocity_.ensure(cells * 3) ||
      !velocity_tmp_.ensure(cells * 3) || !velocity_aux_.ensure(cells * 3) ||
      !scalar_tmp_.ensure(cells) || !scalar_aux_.ensure(cells) ||
      !scalar_pack_.ensure(cells * 3) || !pressure_.ensure(cells) ||
      !rhs_.ensure(cells) || !residual_.ensure(cells) || !direction_.ensure(cells) ||
      !product_.ensure(cells) || !curl_.ensure(cells * 4) || !source_.ensure(cells * 7) ||
      !params_.ensure(P_COUNT + brick_block_) || !adv_.ensure(8) || !partial_.ensure(blocks * 4) ||
      !coarse_params_.ensure(P_COUNT + 8) || !coarse_pressure_.ensure(coarse_cells_) ||
      !coarse_rhs_.ensure(coarse_cells_) || !coarse_residual_.ensure(coarse_cells_) ||
      !coarse_direction_.ensure(coarse_cells_) || !coarse_product_.ensure(coarse_cells_) ||
      !coarse_partial_.ensure(std::max<int64_t>(1, coarse_blocks_) * 4) ||
      !coarse_mask_.ensure(coarse_cells_) ||
      !brick_flags_.ensure(std::max<int64_t>(1, pool_blocks_)) ||
      !brick_list_.ensure(std::max<int64_t>(1, pool_blocks_)) ||
      !brick_list_count_.ensure(1) ||
      !mask_a_.ensure(cells) || !mask_b_.ensure(cells) || !solid_.ensure(cells) ||
      !solid_current_.ensure(cells) || !solid_carve_.ensure(cells) ||
      !collider_velocity_.ensure(3 * 256))
  {
    r_reason = "device allocation failed (out of GPU memory)";
    return false;
  }
  return true;
}

bool HaIgnisSolver::bind(const IgnisDeviceFrame &frame, std::string &r_reason)
{
  /* Scatter the dense frame into the brick pool, then upload. */
  Array<float> pool_scalars;
  Array<float> pool_velocity;
  Array<float> pool_pressure;
  Array<float> pool_source;
  pool_scalars.reinitialize(int(pool_cells_ * 4));
  pool_velocity.reinitialize(int(pool_cells_ * 3));
  pool_pressure.reinitialize(int(pool_cells_));
  pool_source.reinitialize(int(pool_cells_ * 7));
  bricks_scatter_float(bricks_,
                       Span<const float>(frame.scalars.data(), frame.scalars.size()),
                       4,
                       pool_scalars.as_mutable_span());
  bricks_scatter_float(bricks_,
                       Span<const float>(frame.velocity.data(), frame.velocity.size()),
                       3,
                       pool_velocity.as_mutable_span());
  bricks_scatter_float(bricks_,
                       Span<const float>(frame.pressure.data(), frame.pressure.size()),
                       1,
                       pool_pressure.as_mutable_span());
  bricks_scatter_float(bricks_,
                       Span<const float>(frame.source.data(), frame.source.size()),
                       7,
                       pool_source.as_mutable_span());
  scalars_.upload(pool_scalars.as_span());
  velocity_.upload(pool_velocity.as_span());
  pressure_.upload(pool_pressure.as_span());
  source_.upload(pool_source.as_span());

  Array<int> solid;
  solid.reinitialize(int(cells_));
  Array<int> solid_current;
  solid_current.reinitialize(int(cells_));
  Array<int> solid_carve;
  solid_carve.reinitialize(int(cells_));
  solid.fill(0);
  solid_current.fill(0);
  solid_carve.fill(0);
  for (int64_t i = 0; i < cells_; i++) {
    solid[i] = frame.mask[i] & (IGNIS_MASK_SOLID | (0xFF << 8));
    solid_current[i] = frame.current_mask.is_empty() ?
                           solid[i] :
                           (frame.current_mask[i] & (IGNIS_MASK_SOLID | (0xFF << 8)));
    solid_carve[i] = frame.carve_mask.is_empty() ?
                         0 :
                         (frame.carve_mask[i] & (IGNIS_MASK_SOLID | (0xFF << 8)));
  }
  Array<int> pool_solid;
  Array<int> pool_solid_current;
  Array<int> pool_solid_carve;
  pool_solid.reinitialize(int(pool_cells_));
  pool_solid_current.reinitialize(int(pool_cells_));
  pool_solid_carve.reinitialize(int(pool_cells_));  bricks_scatter_int(
      bricks_, Span<const int>(solid.data(), solid.size()), pool_solid.as_mutable_span());
  bricks_scatter_int(bricks_,
                     Span<const int>(solid_current.data(), solid_current.size()),
                     pool_solid_current.as_mutable_span());
  bricks_scatter_int(bricks_,
                     Span<const int>(solid_carve.data(), solid_carve.size()),
                     pool_solid_carve.as_mutable_span());
  solid_.upload(pool_solid.as_span());
  solid_current_.upload(pool_solid_current.as_span());
  solid_carve_.upload(pool_solid_carve.as_span());
  if (frame.collider_velocities.is_empty()) {
    Array<float> zeros(3 * 256, 0.0f);
    collider_velocity_.upload(zeros.as_span());
  }
  else {
    collider_velocity_.upload(frame.collider_velocities);
  }
  mask_a_.upload(pool_solid.as_span());
  mask_ = &mask_a_;
  mask_other_ = &mask_b_;
  if (debug_enabled()) {
    int64_t smoke_dense = 0;
    int64_t smoke_pool = 0;
    int64_t src_dense = 0;
    int64_t src_pool = 0;
    int64_t vel_dense = 0;
    int64_t vel_pool = 0;
    int64_t solid_dense = 0;
    int64_t solid_pool = 0;
    for (int64_t i = 0; i < cells_; i++) {
      smoke_dense += frame.scalars[i * 4 + 0] > 1e-4f ? 1 : 0;
      src_dense += frame.source[i * 7 + 0] > 1e-4f ? 1 : 0;
      solid_dense += solid[i] != 0 ? 1 : 0;
      vel_dense += (frame.velocity[i * 3 + 0] != 0.0f || frame.velocity[i * 3 + 1] != 0.0f ||
                    frame.velocity[i * 3 + 2] != 0.0f) ?
                       1 :
                       0;
    }
    for (int64_t i = 0; i < pool_cells_; i++) {
      smoke_pool += pool_scalars[i * 4 + 0] > 1e-4f ? 1 : 0;
      src_pool += pool_source[i * 7 + 0] > 1e-4f ? 1 : 0;
      solid_pool += pool_solid[i] != 0 ? 1 : 0;
      vel_pool += (pool_velocity[i * 3 + 0] != 0.0f || pool_velocity[i * 3 + 1] != 0.0f ||
                   pool_velocity[i * 3 + 2] != 0.0f) ?
                      1 :
                      0;
    }
    debug_log("IGNIS_HA bind: smoke %lld/%lld source %lld/%lld velocity %lld/%lld solid %lld/%lld\n",
              (long long)smoke_dense,
              (long long)smoke_pool,
              (long long)src_dense,
              (long long)src_pool,
              (long long)vel_dense,
              (long long)vel_pool,
              (long long)solid_dense,
              (long long)solid_pool);
  }
  if (!scalars_.is_valid() || !velocity_.is_valid() || !pressure_.is_valid() ||
      !source_.is_valid() || !solid_.is_valid() || !solid_current_.is_valid() ||
      !solid_carve_.is_valid() || !collider_velocity_.is_valid() || !mask_->is_valid() ||
      !mask_other_->is_valid())
  {
    r_reason = "device upload failed";
    return false;
  }
  return true;
}

void HaIgnisSolver::fill_params(const IgnisDeviceFrame &frame,
                                const float dt,
                                MutableSpan<float> r_params)
{
  r_params.take_front(P_COUNT).fill(0.0f);
  const ignis::Settings &s = frame.settings;
  r_params[P_CELLS] = float(pool_cells_);
  r_params[P_DX] = float(frame.dims.x);
  r_params[P_DY] = float(frame.dims.y);
  r_params[P_DZ] = float(frame.dims.z);
  r_params[P_VOXEL] = frame.voxel_size;
  r_params[P_DT] = dt;
  r_params[P_GRAVITY_X] = s.gravity.x;
  r_params[P_GRAVITY_Y] = s.gravity.y;
  r_params[P_GRAVITY_Z] = s.gravity.z;
  r_params[P_WIND_X] = s.wind.x;
  r_params[P_WIND_Y] = s.wind.y;
  r_params[P_WIND_Z] = s.wind.z;
  r_params[P_BUOYANCY] = s.buoyancy;
  r_params[P_AMBIENT] = s.ambient_temperature;
  r_params[P_DRAG] = std::max(0.0f, s.drag);
  r_params[P_DAMPING] = std::max(0.0f, s.velocity_damping);
  r_params[P_VORTICITY] = s.vorticity;
  r_params[P_BURN_RATE] = std::max(0.0f, s.burn_rate);
  r_params[P_HEAT_RELEASE] = s.heat_release;
  r_params[P_SMOKE_PER_BURN] = s.smoke_per_burn;
  r_params[P_COOLING] = std::max(0.0f, s.cooling);
  r_params[P_DISSIPATION] = std::max(0.0f, s.dissipation);
  r_params[P_FLAME_THRESHOLD] = std::max(0.0f, s.flame_threshold);
  r_params[P_DOMAIN_FIELD] = float(int(s.domain_field));
  r_params[P_DOMAIN_THRESHOLD] = std::max(0.0f, s.domain_threshold);
  r_params[P_USE_BOUNDS] = s.use_bounds ? 1.0f : 0.0f;
  r_params[P_BOUNDS_MIN_X] = s.bounds_min.x;
  r_params[P_BOUNDS_MIN_Y] = s.bounds_min.y;
  r_params[P_BOUNDS_MIN_Z] = s.bounds_min.z;
  r_params[P_BOUNDS_MAX_X] = s.bounds_max.x;
  r_params[P_BOUNDS_MAX_Y] = s.bounds_max.y;
  r_params[P_BOUNDS_MAX_Z] = s.bounds_max.z;
  r_params[P_RADIUS] = float(frame.solver_radius);
  r_params[P_EXTRAPOLATE] = float(frame.extrapolate);
  r_params[P_TOLERANCE] = std::max(1e-8f, s.tolerance);
  r_params[P_MAX_ITERATIONS] = float(pressure_iteration_budget(s, frame.dims));
  r_params[P_PARTIALS] = float(blocks_);
  r_params[P_ORIGIN_X] = float(frame.index_min.x);
  r_params[P_ORIGIN_Y] = float(frame.index_min.y);
  r_params[P_ORIGIN_Z] = float(frame.index_min.z);
  r_params[P_USE_DOMAIN_MASK] = s.use_domain_mask ? 1.0f : 0.0f;
  r_params[P_FORCE_TIME] = s.force_time;
  r_params[P_DISTURBANCE] = s.disturbance;
  r_params[P_DISTURBANCE_SCALE] = std::max(1e-4f, s.disturbance_scale);
  r_params[P_DISTURBANCE_SPEED] = s.disturbance_speed;
  r_params[P_DISTURBANCE_SEED] = float(s.disturbance_seed);
  r_params[P_DISTURBANCE_OCTAVES] = float(s.disturbance_octaves);
  r_params[P_DISTURBANCE_LACUNARITY] = std::max(1.0f, s.disturbance_lacunarity);
  r_params[P_DISTURBANCE_ROUGHNESS] = std::clamp(s.disturbance_roughness, 0.0f, 1.0f);
  r_params[P_TURBULENCE] = s.turbulence;
  r_params[P_TURBULENCE_SCALE] = std::max(1e-4f, s.turbulence_scale);
  r_params[P_TURBULENCE_SPEED] = s.turbulence_speed;
  r_params[P_TURBULENCE_SEED] = float(s.turbulence_seed);
  r_params[P_TURBULENCE_OCTAVES] = float(s.turbulence_octaves);
  r_params[P_TURBULENCE_LACUNARITY] = std::max(1.0f, s.turbulence_lacunarity);
  r_params[P_TURBULENCE_ROUGHNESS] = std::clamp(s.turbulence_roughness, 0.0f, 1.0f);
  r_params[P_DIST_REMAP_FIELD] = float(s.disturbance_remap_field);
  r_params[P_DIST_REMAP_MODE] = float(s.disturbance_remap_mode);
  r_params[P_DIST_REMAP_MIN] = s.disturbance_remap_min;
  r_params[P_DIST_REMAP_MAX] = s.disturbance_remap_max;
  r_params[P_DIST_REMAP_SOFTNESS] = std::clamp(s.disturbance_remap_softness, 0.0f, 1.0f);
  r_params[P_DIST_REMAP_SCALE] = std::max(0.0f, s.disturbance_remap_scale);
  r_params[P_TURB_REMAP_FIELD] = float(s.turbulence_remap_field);
  r_params[P_TURB_REMAP_MODE] = float(s.turbulence_remap_mode);
  r_params[P_TURB_REMAP_MIN] = s.turbulence_remap_min;
  r_params[P_TURB_REMAP_MAX] = s.turbulence_remap_max;
  r_params[P_TURB_REMAP_SOFTNESS] = std::clamp(s.turbulence_remap_softness, 0.0f, 1.0f);
  r_params[P_TURB_REMAP_SCALE] = std::max(0.0f, s.turbulence_remap_scale);
  r_params[P_COARSE_DX] = float(coarse_dims_.x);
  r_params[P_COARSE_DY] = float(coarse_dims_.y);
  r_params[P_COARSE_DZ] = float(coarse_dims_.z);
}

void HaIgnisSolver::upload_coarse_params(const Span<const float> fine_params)
{
  Array<float> params;
  params.reinitialize(P_COUNT + 8);
  params.fill(0.0f);
  std::copy(fine_params.begin(), fine_params.begin() + P_COUNT, params.begin());
  /* The coarse grid is a dense box of its own, so it keeps the dense cell addressing. */
  params[P_COUNT + 0] = 0.0f;
  params[P_CELLS] = float(coarse_cells_);
  params[P_DX] = float(coarse_dims_.x);
  params[P_DY] = float(coarse_dims_.y);
  params[P_DZ] = float(coarse_dims_.z);
  params[P_VOXEL] = fine_params[P_VOXEL] * 2.0f;
  params[P_PARTIALS] = float(coarse_blocks_);
  params[P_MAX_ITERATIONS] = float(std::max(0, coarse_iterations_));
  params[P_ITERATION] = 0.0f;
  params[P_CONVERGED] = 0.0f;
  params[P_RHS_NORM] = 0.0f;
  params[P_RES_NORM] = 0.0f;
  params[P_RZ] = 0.0f;
  params[P_PQ] = 0.0f;
  params[P_ALPHA] = 0.0f;
  params[P_BETA] = 0.0f;
  coarse_params_.upload(params.as_span());
}

void HaIgnisSolver::write_adv(const float dir,
                              const float order,
                              const float limiter,
                              const float component,
                              const float correct_mode)
{
  Array<float> values;
  values.reinitialize(8);
  values.fill(0.0f);
  values[0] = dir;
  values[1] = order;
  values[2] = limiter;
  values[3] = component;
  values[4] = correct_mode;
  adv_.upload(values.as_span());
}

bool HaIgnisSolver::pass(gpu::Shader *shader, Span<gpu::StorageBuf *> buffers, const int count)
{
  if (shader == nullptr) {
    return false;
  }
  GPU_shader_bind(shader);
  ha_bind_storage_buffers(buffers);
  ha_dispatch(shader, count);
  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE);
  result_->passes++;
  return true;
}

void HaIgnisSolver::swap_masks()
{
  std::swap(mask_, mask_other_);
}

bool HaIgnisSolver::dilate_mask(const int radius, const int bit_index, std::string &r_reason)
{
  if (radius <= 0) {
    return true;
  }
  /* Keep the device side params (the CG writes its statistics there): read the whole buffer, change
   * only the dilation bit and upload it back. It has to be the full size, otherwise the brick
   * addressing block appended after `P_COUNT` is cleared. */
  Array<float> values;
  values.reinitialize(int(params_host_.size()));
  params_.download(values.as_mutable_span());
  values[P_DILATE_BIT] = float(bit_index);
  params_.upload(values.as_span());
  for (int i = 0; i < radius; i++) {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(mask_->buf());
    buffers.append(mask_other_->buf());
    if (!pass(dilate_, buffers, int(pool_cells_))) {
      r_reason = "mask dilation pass failed";
      return false;
    }
    swap_masks();
  }
  return true;
}

bool HaIgnisSolver::coarse_pressure_guess(const int iterations, std::string &r_reason)
{
  if (coarse_cells_ <= 0 || iterations <= 0) {
    return true;
  }
  const int coarse_count = int(coarse_cells_);
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(mask_->buf());
    buffers.append(rhs_.buf());
    buffers.append(coarse_mask_.buf());
    buffers.append(coarse_rhs_.buf());
    if (!pass(coarse_restrict_, buffers, coarse_count)) {
      r_reason = "coarse restriction failed";
      return false;
    }
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(coarse_params_.buf());
    buffers.append(coarse_pressure_.buf());
    if (!pass(zero_field_, buffers, coarse_count)) {
      r_reason = "coarse pressure reset failed";
      return false;
    }
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(coarse_params_.buf());
    buffers.append(coarse_product_.buf());
    if (!pass(zero_field_, buffers, coarse_count)) {
      r_reason = "coarse product reset failed";
      return false;
    }
  }
  /* Reuse the fine CG kernels on the coarse params/buffers. This is only a starting point, so the
   * iteration count is fixed and nothing is read back. The coarse pressure starts at zero, so
   * `cg_init` sees a zero product and builds the residual straight from the restricted rhs. */
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(coarse_params_.buf());
    buffers.append(coarse_rhs_.buf());
    buffers.append(coarse_product_.buf());
    buffers.append(coarse_mask_.buf());
    buffers.append(coarse_residual_.buf());
    buffers.append(coarse_direction_.buf());
    buffers.append(coarse_partial_.buf());
    if (!pass(cg_init_, buffers, coarse_count)) {
      r_reason = "coarse CG initialisation failed";
      return false;
    }
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(coarse_params_.buf());
    buffers.append(coarse_partial_.buf());
    if (!pass(reduce_init_, buffers, 1)) {
      r_reason = "coarse CG initial reduction failed";
      return false;
    }
  }
  for (int iteration = 0; iteration < iterations; iteration++) {
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(coarse_params_.buf());
      buffers.append(coarse_mask_.buf());
      buffers.append(coarse_residual_.buf());
      buffers.append(coarse_direction_.buf());
      buffers.append(brick_list_.buf());
      if (!pass(cg_direction_, buffers, coarse_count)) {
        r_reason = "coarse CG direction pass failed";
        return false;
      }
    }
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(coarse_params_.buf());
      buffers.append(coarse_mask_.buf());
      buffers.append(coarse_direction_.buf());
      buffers.append(coarse_product_.buf());
      buffers.append(coarse_partial_.buf());
      buffers.append(brick_list_.buf());
      if (!pass(cg_laplacian_, buffers, coarse_count)) {
        r_reason = "coarse CG Laplacian pass failed";
        return false;
      }
    }
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(coarse_params_.buf());
      buffers.append(coarse_partial_.buf());
      if (!pass(reduce_pq_, buffers, 1)) {
        r_reason = "coarse CG pq reduction failed";
        return false;
      }
    }
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(coarse_params_.buf());
      buffers.append(coarse_mask_.buf());
      buffers.append(coarse_direction_.buf());
      buffers.append(coarse_product_.buf());
      buffers.append(coarse_pressure_.buf());
      buffers.append(coarse_residual_.buf());
      buffers.append(coarse_partial_.buf());
      buffers.append(brick_list_.buf());
      if (!pass(cg_update_, buffers, coarse_count)) {
        r_reason = "coarse CG update pass failed";
        return false;
      }
    }
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(coarse_params_.buf());
      buffers.append(coarse_partial_.buf());
      if (!pass(reduce_res_, buffers, 1)) {
        r_reason = "coarse CG residual finalize failed";
        return false;
      }
    }
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(mask_->buf());
    buffers.append(coarse_pressure_.buf());
    buffers.append(pressure_.buf());
    if (!pass(coarse_prolong_, buffers, int(pool_cells_))) {
      r_reason = "coarse prolongation failed";
      return false;
    }
  }
  return true;
}

bool HaIgnisSolver::projection(const int64_t cells, std::string &r_reason)
{
  (void)cells;
  /* Pressure domain: the smoke grown by the solver radius, face connected (same as the CPU
   * `smoke_tree(state, solver_radius)`), then the cells whose three forward neighbours exist. */
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(scalars_.buf());
    buffers.append(solid_.buf());
    buffers.append(mask_other_->buf());
    if (!pass(mask_init_, buffers, int(pool_cells_))) {
      r_reason = "pressure domain pass failed";
      return false;
    }
    swap_masks();
  }
  if (!dilate_mask(int(radius_), 1 /* HA_DOMAIN */, r_reason)) {
    return false;
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(mask_->buf());
    if (!pass(unknown_, buffers, int(pool_cells_))) {
      r_reason = "pressure unknown pass failed";
      return false;
    }
  }

  const int count = int(pool_cells_);
  /* Brick list of the pressure unknowns: the CG only has to visit the bricks that contain an
   * unknown cell (about a fifth of the pool here), not the whole dense box. The list is built on
   * the device in slot order (deterministic) and its length is read back once per projection. */
  int cg_count = count;
  if (cg_list_) {
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(mask_->buf());
      buffers.append(brick_flags_.buf());
      if (!pass(brick_flag_, buffers, int(bricks_.brick_count))) {
        r_reason = "brick flag pass failed";
        return false;
      }
    }
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(brick_flags_.buf());
      buffers.append(brick_list_.buf());
      buffers.append(partial_.buf());
      buffers.append(brick_list_count_.buf());
      if (!pass(brick_compact_, buffers, 1)) {
        r_reason = "brick list pass failed";
        return false;
      }
    }
    int list_count = 0;
    brick_list_count_.download(MutableSpan<int>(&list_count, 1));
    list_count = std::clamp(list_count, 0, int(bricks_.brick_count));
    cg_count = list_count * IGNIS_BRICK_CELLS;
    params_host_[P_COUNT + 5] = 1.0f;
    params_.upload(params_host_.as_span());
  }
  /* Right hand side + the divergence norm used by the stats. */
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(velocity_.buf());
    buffers.append(mask_->buf());
    buffers.append(rhs_.buf());
    buffers.append(partial_.buf());
    if (!pass(rhs_shader_, buffers, count)) {
      r_reason = "divergence pass failed";
      return false;
    }
  }
  if (!coarse_pressure_guess(coarse_iterations_, r_reason)) {
    return false;
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(pressure_.buf());
    buffers.append(mask_->buf());
    buffers.append(product_.buf());
    buffers.append(brick_list_.buf());
    if (!pass(cg_product_, buffers, cg_count)) {
      r_reason = "CG initial product failed";
      return false;
    }
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(rhs_.buf());
    buffers.append(product_.buf());
    buffers.append(mask_->buf());
    buffers.append(residual_.buf());
    buffers.append(direction_.buf());
    buffers.append(partial_.buf());
    buffers.append(brick_list_.buf());
    if (!pass(cg_init_, buffers, cg_count)) {
      r_reason = "CG initialisation failed";
      return false;
    }
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(partial_.buf());
    if (!pass(reduce_init_, buffers, 1)) {
      r_reason = "CG initial reduction failed";
      return false;
    }
  }

  Array<float> values;
  values.reinitialize(P_COUNT);
  params_.download(values.as_mutable_span());
  if (debug_enabled()) {
    debug_log("IGNIS_HA proj: unknown=%.0f rhs=%.5f\n",
              double(values[P_UNKNOWN_COUNT]),
              double(values[P_RHS_NORM]));
  }
  bool converged = values[P_CONVERGED] != 0.0f;
  int iterations = 0;
  const int max_iterations = std::max(1, int(values[P_MAX_ITERATIONS]));
  while (!converged && iterations < max_iterations) {
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(mask_->buf());
      buffers.append(residual_.buf());
      buffers.append(direction_.buf());
      buffers.append(brick_list_.buf());
      if (!pass(cg_direction_, buffers, cg_count)) {
        r_reason = "CG direction pass failed";
        return false;
      }
    }
    {
      /* Laplacian + the pq block reduction (shared memory, one dispatch). */
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(mask_->buf());
      buffers.append(direction_.buf());
      buffers.append(product_.buf());
      buffers.append(partial_.buf());
      buffers.append(brick_list_.buf());
      if (!pass(cg_laplacian_, buffers, cg_count)) {
        r_reason = "CG Laplacian pass failed";
        return false;
      }
    }
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(partial_.buf());
      if (!pass(reduce_pq_, buffers, 1)) {
        r_reason = "CG pq reduction failed";
        return false;
      }
    }
    {
      /* Update + the residual block reduction (shared memory, one dispatch). */
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(mask_->buf());
      buffers.append(direction_.buf());
      buffers.append(product_.buf());
      buffers.append(pressure_.buf());
      buffers.append(residual_.buf());
      buffers.append(partial_.buf());
      buffers.append(brick_list_.buf());
      if (!pass(cg_update_, buffers, cg_count)) {
        r_reason = "CG update pass failed";
        return false;
      }
    }
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(partial_.buf());
      if (!pass(reduce_res_, buffers, 1)) {
        r_reason = "CG residual finalize failed";
        return false;
      }
    }
    iterations++;
    /* Checking the convergence every few iterations keeps the number of GPU reads low: each read
     * waits for the outstanding dispatches. */
    if ((iterations % cg_check_interval_) == 0 || iterations == max_iterations) {
      params_.download(values.as_mutable_span());
      converged = values[P_CONVERGED] != 0.0f;
    }
  }

  /* u -= grad p on the whole band, then measure the remaining divergence. */
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(mask_->buf());
    buffers.append(pressure_.buf());
    buffers.append(velocity_.buf());
    if (!pass(cg_gradient_, buffers, count)) {
      r_reason = "CG gradient pass failed";
      return false;
    }
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(mask_->buf());
    buffers.append(velocity_.buf());
    buffers.append(partial_.buf());
    if (!pass(divergence_after_, buffers, count)) {
      r_reason = "divergence after pass failed";
      return false;
    }
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(partial_.buf());
    if (!pass(reduce_divergence_, buffers, 1)) {
      r_reason = "divergence after reduction failed";
      return false;
    }
  }
  return true;
}

bool HaIgnisSolver::advect_velocity(const int64_t cells, std::string &r_reason)
{
  const int count = int(cells);
  const ignis::AdvectionScheme scheme = settings_.advection_scheme;
  const int limiter = int(settings_.advection_limiter);

  auto advect =
      [&](gpu::Shader *shader, HaBuffer &field_in, HaBuffer &field_out, HaBuffer &velocity) {
        Vector<gpu::StorageBuf *> buffers;
        buffers.append(params_.buf());
        buffers.append(adv_.buf());
        buffers.append(field_in.buf());
        buffers.append(field_out.buf());
        buffers.append(velocity.buf());
        return pass(shader, buffers, count);
      };

  if (scheme == ignis::AdvectionScheme::SemiLagrangian ||
      scheme == ignis::AdvectionScheme::Midpoint)
  {
    const int order = scheme == ignis::AdvectionScheme::Midpoint ? 2 : 1;
    write_adv(-1, float(order), 0, 0, 0);
    if (!advect(advect_vec3_, velocity_, velocity_tmp_, velocity_)) {
      r_reason = "velocity advection failed";
      return false;
    }
  }
  else if (scheme == ignis::AdvectionScheme::MacCormack) {
    write_adv(-1, 1, 0, 0, 0);
    if (!advect(advect_vec3_, velocity_, velocity_tmp_, velocity_)) {
      r_reason = "velocity advection (forward) failed";
      return false;
    }
    write_adv(1, 1, 0, 0, 0);
    if (!advect(advect_vec3_, velocity_tmp_, velocity_aux_, velocity_)) {
      r_reason = "velocity advection (backward) failed";
      return false;
    }
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(adv_.buf());
      buffers.append(velocity_.buf());
      buffers.append(velocity_tmp_.buf());
      buffers.append(velocity_aux_.buf());
      write_adv(0, 1, 0, 0, 0);
      if (!pass(correct_vec3_, buffers, count)) {
        r_reason = "velocity advection (correction) failed";
        return false;
      }
      write_adv(-1, 1, float(limiter), 0, 0);
      if (!pass(limiter_vec3_, buffers, count)) {
        r_reason = "velocity advection (limiter) failed";
        return false;
      }
    }
  }
  else {
    r_reason = "BFECC advection is not implemented on the device";
    return false;
  }

  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(velocity_tmp_.buf());
    buffers.append(velocity_.buf());
    if (!pass(copy_vec3_, buffers, count)) {
      r_reason = "velocity advection copy failed";
      return false;
    }
  }
  return true;
}

bool HaIgnisSolver::advect_scalars(const int64_t cells, std::string &r_reason)
{
  const int count = int(cells);
  const ignis::AdvectionScheme scheme = settings_.advection_scheme;
  const int limiter = int(settings_.advection_limiter);
  if (scheme == ignis::AdvectionScheme::Bfecc) {
    r_reason = "BFECC advection is not implemented on the device";
    return false;
  }

  /* Pack density/temperature/fuel into the 3 float layout the vec3 advection kernels use. One
   * advection then shares the backtrace and the velocity samples between the three components,
   * instead of running a scalar advection per component. */
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(scalars_.buf());
    buffers.append(scalar_pack_.buf());
    if (!pass(pack_scalars_, buffers, count)) {
      r_reason = "scalar pack failed";
      return false;
    }
  }

  auto advect = [&](HaBuffer &field_in, HaBuffer &field_out) {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(adv_.buf());
    buffers.append(field_in.buf());
    buffers.append(field_out.buf());
    buffers.append(velocity_.buf());
    return pass(advect_vec3_, buffers, count);
  };

  write_adv(-1, scheme == ignis::AdvectionScheme::Midpoint ? 2 : 1, 0, 0, 0);
  if (!advect(scalar_pack_, velocity_tmp_)) {
    r_reason = "scalar advection failed";
    return false;
  }

  if (scheme == ignis::AdvectionScheme::MacCormack) {
    write_adv(1, 1, 0, 0, 0);
    if (!advect(velocity_tmp_, velocity_aux_)) {
      r_reason = "scalar advection (backward) failed";
      return false;
    }
    write_adv(0, 1, 0, 0, 0);
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(adv_.buf());
      buffers.append(scalar_pack_.buf());
      buffers.append(velocity_tmp_.buf());
      buffers.append(velocity_aux_.buf());
      if (!pass(correct_vec3_, buffers, count)) {
        r_reason = "scalar advection (correction) failed";
        return false;
      }
    }
    write_adv(-1, 1, float(limiter), 0, 0);
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(adv_.buf());
      buffers.append(scalar_pack_.buf());
      buffers.append(velocity_tmp_.buf());
      buffers.append(velocity_.buf());
      if (!pass(limiter_vec3_, buffers, count)) {
        r_reason = "scalar advection (limiter) failed";
        return false;
      }
    }
  }

  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(velocity_tmp_.buf());
    buffers.append(scalars_.buf());
    if (!pass(unpack_scalars_, buffers, count)) {
      r_reason = "scalar unpack failed";
      return false;
    }
  }
  return true;
}

bool HaIgnisSolver::extrapolate(const int64_t cells, std::string &r_reason)
{
  const int count = int(cells);
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(scalars_.buf());
    buffers.append(solid_.buf());
    buffers.append(mask_other_->buf());
    if (!pass(mask_keep_init_, buffers, count)) {
      r_reason = "velocity band init failed";
      return false;
    }
    swap_masks();
  }
  if (!dilate_mask(int(radius_), 5 /* HA_KEEP */, r_reason)) {
    return false;
  }
  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(mask_->buf());
    buffers.append(velocity_.buf());
    if (!pass(extrapolate_setup_, buffers, count)) {
      r_reason = "velocity band setup failed";
      return false;
    }
  }
  for (int i = 0; i < int(settings_.extrapolate); i++) {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(mask_->buf());
    buffers.append(mask_other_->buf());
    buffers.append(velocity_.buf());
    if (!pass(extrapolate_pass_, buffers, count)) {
      r_reason = "velocity extrapolation failed";
      return false;
    }
    swap_masks();
  }
  return true;
}

bool HaIgnisSolver::solve(IgnisDeviceFrame &frame, IgnisDeviceResult &r_result)
{
  result_ = &r_result;
  r_result = IgnisDeviceResult{};
  result_ = &r_result;
  settings_ = frame.settings;

  const int64_t cells = int64_t(frame.dims.x) * int64_t(frame.dims.y) * int64_t(frame.dims.z);
  if (cells <= 0) {
    r_result.reason = "empty voxel box";
    return false;
  }
  cells_ = cells;
  radius_ = frame.solver_radius;
  /* Brick pool (doc/ignis-brick-sparse.md): the content plus the same voxel margin the dense box
   * gives the solver, grown in brick units. The content bricks are measured to be ~10-16% of the
   * box, so the pool is a few times smaller than the dense box. */
  const int margin = std::min(64,
                              int(radius_) + 2 +
                                  int(std::ceil(std::max(1.0f, frame.settings.cfl))) *
                                      std::max(1, frame.substeps) +
                                  2);
  bool force_dense = false;
  if (const char *env = std::getenv("BLENDER_IGNIS_DENSE")) {
    force_dense = env[0] != '0';
  }
  bricks_ = ignis_bricks_build(frame, margin, force_dense);
  if (bricks_.brick_count <= 0) {
    r_result.reason = "the brick grid is empty";
    return false;
  }
  pool_cells_ = bricks_.pool_cells;
  pool_blocks_ = bricks_.brick_count;
  blocks_ = pool_blocks_;
  brick_block_ = 8 + bricks_.tile_count + bricks_.brick_count * 9;
  if (debug_enabled()) {
    debug_log("IGNIS_HA bricks: cells=%lld bricks=%d/%d pool=%lld (%.2fx) margin=%d\n",
              (long long)cells_,
              bricks_.brick_count,
              bricks_.tile_count,
              (long long)pool_cells_,
              cells_ > 0 ? double(pool_cells_) / double(cells_) : 0.0,
              margin);
  }
  coarse_dims_ = int3((frame.dims.x + 1) / 2, (frame.dims.y + 1) / 2, (frame.dims.z + 1) / 2);
  coarse_cells_ = int64_t(coarse_dims_.x) * int64_t(coarse_dims_.y) * int64_t(coarse_dims_.z);
  coarse_blocks_ = (coarse_cells_ + GROUP_SIZE - 1) / GROUP_SIZE;
  coarse_iterations_ = int(std::clamp<int64_t>(cells / 16384, 6, 12));
  if (cells < 32768) {
    /* Below a few ten thousand voxels the CG converges in a handful of iterations, so the extra
     * dispatches of the coarse solve cost more than they save. */
    coarse_iterations_ = 0;
  }
  if (const char *env = std::getenv("BLENDER_IGNIS_COARSE")) {
    if (env[0] == '0') {
      coarse_iterations_ = 0;
    }
  }
  if (const char *env = std::getenv("BLENDER_IGNIS_COARSE_ITERS")) {
    coarse_iterations_ = std::clamp(std::atoi(env), 0, 64);
  }
  if (const char *env = std::getenv("BLENDER_IGNIS_CG_CHECK")) {
    cg_check_interval_ = std::clamp(std::atoi(env), 1, 64);
  }
  cg_list_ = true;
  if (const char *env = std::getenv("BLENDER_IGNIS_NO_CG_LIST")) {
    if (env[0] != '0') {
      cg_list_ = false;
    }
  }

  if (frame.settings.advection_scheme == ignis::AdvectionScheme::Bfecc) {
    r_result.reason = "BFECC advection is not implemented on the device";
    return false;
  }

  const double start = time_now_ms();
  if (!allocate(pool_cells_, pool_blocks_, r_result.reason)) {
    return false;
  }
  create_shaders();
  if (!shaders_valid()) {
    r_result.reason = "shader creation failed";
    return false;
  }
  if (debug_enabled()) {
    int64_t smoke_in = 0;
    for (int64_t i = 0; i < cells_; i++) {
      if (frame.scalars[i * 4 + 0] > 1e-4f) {
        smoke_in++;
      }
    }
    debug_log("IGNIS_HA input: smoke=%lld\n", (long long)smoke_in);
  }
  if (!bind(frame, r_result.reason)) {
    return false;
  }
  if (const char *env = std::getenv("BLENDER_IGNIS_BRICK_SELFTEST")) {
    if (env[0] != '0') {
      ignis_bricks_selftest(frame, margin, force_dense);
    }
  }

  params_host_.reinitialize(P_COUNT + brick_block_);
  params_host_.fill(0.0f);
  /* Brick addressing block appended to the param buffer (see `GLSL_PRELUDE`). */
  {
    params_host_[P_COUNT + 0] = 1.0f; /* sparse */
    params_host_[P_COUNT + 1] = float(bricks_.tile_count);
    params_host_[P_COUNT + 2] = float(bricks_.brick_dims.x);
    params_host_[P_COUNT + 3] = float(bricks_.brick_dims.y);
    params_host_[P_COUNT + 4] = float(bricks_.brick_dims.z);
    params_host_[P_COUNT + 5] = 0.0f; /* list mode: off until `projection` builds the list */
    params_host_[P_COUNT + 6] = float(bricks_.brick_count);
    for (int tile = 0; tile < bricks_.tile_count; tile++) {
      params_host_[P_COUNT + 8 + tile] = float(bricks_.slot_map[tile]);
    }
    for (int slot = 0; slot < bricks_.brick_count; slot++) {
      const int base = P_COUNT + 8 + bricks_.tile_count + slot * 3;
      params_host_[base + 0] = float(bricks_.coords[slot].x);
      params_host_[base + 1] = float(bricks_.coords[slot].y);
      params_host_[base + 2] = float(bricks_.coords[slot].z);
    }
    /* Six face neighbours per brick, cached in shared memory by `ha_init_neighbors`. */
    const int nb_base = P_COUNT + 8 + bricks_.tile_count + bricks_.brick_count * 3;
    for (int slot = 0; slot < bricks_.brick_count; slot++) {
      for (int i = 0; i < 6; i++) {
        params_host_[nb_base + slot * 6 + i] = float(bricks_.neighbor[slot * 6 + i]);
      }
    }
  }
  const int substeps = frame.delta_time > 0.0f ? std::max(1, frame.substeps) : 0;
  bool has_source = false;
  for (const float value : frame.source) {
    if (value != 0.0f) {
      has_source = true;
      break;
    }
  }

  if (substeps == 0) {
    /* First or paused frame: CPU behaviour is "only show the sources, do not integrate". */
    fill_params(frame, 0.0f, params_host_);
    params_.upload(params_host_.as_span());
    if (has_source) {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(scalars_.buf());
      buffers.append(velocity_.buf());
      buffers.append(source_.buf());
      if (!pass(source_merge_, buffers, int(pool_cells_))) {
        r_result.reason = "source merge failed";
        return false;
      }
    }
    {
      Vector<gpu::StorageBuf *> buffers;
      buffers.append(params_.buf());
      buffers.append(solid_carve_.buf());
      buffers.append(scalars_.buf());
      buffers.append(velocity_.buf());
      if (!pass(carve_solids_, buffers, int(pool_cells_))) {
        r_result.reason = "solid carve failed";
        return false;
      }
    }
  }
  else {
    const float step_dt = frame.delta_time / float(substeps);
    for (int substep = 0; substep < substeps; substep++) {
      fill_params(frame, step_dt, params_host_);
      params_.upload(params_host_.as_span());
      upload_coarse_params(Span<const float>(params_host_.data(), params_host_.size()));

      if (has_source) {
        Vector<gpu::StorageBuf *> buffers;
        buffers.append(params_.buf());
        buffers.append(scalars_.buf());
        buffers.append(velocity_.buf());
        buffers.append(source_.buf());
        if (!pass(source_merge_, buffers, int(pool_cells_))) {
          r_result.reason = "source merge failed";
          return false;
        }
      }
      {
        Vector<gpu::StorageBuf *> buffers;
        buffers.append(params_.buf());
        buffers.append(scalars_.buf());
        buffers.append(velocity_.buf());
        buffers.append(pressure_.buf());
        if (!pass(forces_, buffers, int(pool_cells_))) {
          r_result.reason = "force pass failed";
          return false;
        }
      }
      if (settings_.vorticity > 0.0f) {
        Vector<gpu::StorageBuf *> buffers;
        buffers.append(params_.buf());
        buffers.append(scalars_.buf());
        buffers.append(velocity_.buf());
        buffers.append(mask_->buf());
        buffers.append(curl_.buf());
        if (!pass(vorticity_curl_, buffers, int(pool_cells_))) {
          r_result.reason = "vorticity curl pass failed";
          return false;
        }
        Vector<gpu::StorageBuf *> buffers_apply;
        buffers_apply.append(params_.buf());
        buffers_apply.append(velocity_.buf());
        buffers_apply.append(mask_->buf());
        buffers_apply.append(curl_.buf());
        if (!pass(vorticity_apply_, buffers_apply, int(pool_cells_))) {
          r_result.reason = "vorticity apply pass failed";
          return false;
        }
      }
      if (!advect_velocity(pool_cells_, r_result.reason)) {
        return false;
      }
      {
        /* Impose the moving wall before the projection so that its divergence is what pushes the
         * smoke around the collider, then clip again after the projection. */
        Vector<gpu::StorageBuf *> buffers;
        buffers.append(params_.buf());
        buffers.append(solid_current_.buf());
        buffers.append(collider_velocity_.buf());
        buffers.append(velocity_.buf());
        if (!pass(clamp_solids_, buffers, int(pool_cells_))) {
          r_result.reason = "solid clamp pass failed";
          return false;
        }
      }
      if (!projection(pool_cells_, r_result.reason)) {
        return false;
      }
      {
        Vector<gpu::StorageBuf *> buffers;
        buffers.append(params_.buf());
        buffers.append(solid_current_.buf());
        buffers.append(collider_velocity_.buf());
        buffers.append(velocity_.buf());
        if (!pass(clamp_solids_, buffers, int(pool_cells_))) {
          r_result.reason = "solid clamp pass failed";
          return false;
        }
      }
      if (!advect_scalars(pool_cells_, r_result.reason)) {
        return false;
      }
      {
        Vector<gpu::StorageBuf *> buffers;
        buffers.append(params_.buf());
        buffers.append(scalars_.buf());
        if (!pass(combust_, buffers, int(pool_cells_))) {
          r_result.reason = "combustion pass failed";
          return false;
        }
        if (!pass(clip_bounds_, buffers, int(pool_cells_))) {
          r_result.reason = "bounds pass failed";
          return false;
        }
      }
      if (!extrapolate(pool_cells_, r_result.reason)) {
        return false;
      }
      {
        Vector<gpu::StorageBuf *> buffers;
        buffers.append(params_.buf());
        buffers.append(solid_carve_.buf());
        buffers.append(scalars_.buf());
        buffers.append(velocity_.buf());
        if (!pass(carve_solids_, buffers, int(pool_cells_))) {
          r_result.reason = "solid carve failed";
          return false;
        }
      }
    }
  }

  {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(scalars_.buf());
    if (!pass(flame_filter_, buffers, int(pool_cells_))) {
      r_result.reason = "flame filter failed";
      return false;
    }
  }
  if (substeps > 0 && settings_.use_domain_mask) {
    Vector<gpu::StorageBuf *> buffers;
    buffers.append(params_.buf());
    buffers.append(scalars_.buf());
    buffers.append(velocity_.buf());
    buffers.append(pressure_.buf());
    buffers.append(mask_->buf());
    if (!pass(domain_mask_, buffers, int(pool_cells_))) {
      r_result.reason = "domain mask failed";
      return false;
    }
  }

  {
    Array<float> pool;
    pool.reinitialize(int(pool_cells_ * 4));
    scalars_.download(pool.as_mutable_span());
    bricks_gather_float(
        bricks_, Span<const float>(pool.data(), pool.size()), 4, frame.scalars);
  }
  {
    Array<float> pool;
    pool.reinitialize(int(pool_cells_ * 3));
    velocity_.download(pool.as_mutable_span());
    bricks_gather_float(
        bricks_, Span<const float>(pool.data(), pool.size()), 3, frame.velocity);
  }
  {
    Array<float> pool;
    pool.reinitialize(int(pool_cells_));
    pressure_.download(pool.as_mutable_span());
    bricks_gather_float(
        bricks_, Span<const float>(pool.data(), pool.size()), 1, frame.pressure);
  }
  Array<int> mask;
  mask.reinitialize(int(cells_));
  {
    Array<int> pool_mask;
    pool_mask.reinitialize(int(pool_cells_));
    mask_->download(pool_mask.as_mutable_span());
    bricks_gather_int(
        bricks_, Span<const int>(pool_mask.data(), pool_mask.size()), mask.as_mutable_span());
  }
  for (int64_t i = 0; i < cells_; i++) {
    frame.mask[i] = mask[i];
  }
  params_.download(params_host_.as_mutable_span());

  int active = 0;
  int solid_count = 0;
  int removed = 0;
  for (int64_t i = 0; i < cells_; i++) {
    if (std::abs(frame.scalars[i * 4 + 0]) > 1e-5f) {
      active++;
    }
    if ((mask[i] & IGNIS_MASK_SOLID) != 0) {
      solid_count++;
    }
    if ((mask[i] & 128) != 0) {
      removed++;
    }
  }

  const float unknown_count = std::max(1.0f, params_host_[P_UNKNOWN_COUNT]);
  r_result.ran = true;
  r_result.active_voxels = active;
  r_result.solid_voxels = solid_count;
  r_result.domain_removed = removed;
  r_result.pressure_iterations = int(params_host_[P_ITERATION] + 0.5f);
  r_result.pressure_iteration_budget = int(params_host_[P_MAX_ITERATIONS] + 0.5f);
  r_result.residual = params_host_[P_RHS_NORM] > 1e-12f ?
                          params_host_[P_RES_NORM] / params_host_[P_RHS_NORM] :
                          0.0f;
  r_result.divergence_before = std::sqrt(params_host_[P_DIV_BEFORE_SUM] / unknown_count);
  r_result.divergence_after = std::sqrt(params_host_[P_DIV_AFTER_SUM] / unknown_count);
  r_result.milliseconds = time_now_ms() - start;

  if (debug_enabled()) {
    float max_speed = 0.0f;
    for (int64_t i = 0; i < cells_; i++) {
      const float vx = frame.velocity[i * 3 + 0];
      const float vy = frame.velocity[i * 3 + 1];
      const float vz = frame.velocity[i * 3 + 2];
      max_speed = std::max(max_speed, std::sqrt(vx * vx + vy * vy + vz * vz));
    }
    debug_log(
        "IGNIS_HA frame: cells=%lld passes=%d substeps=%d active=%d pressure=%d "
        "budget=%d residual=%.5f div %.4f -> %.4f raw(before=%.6g after=%.6g count=%.1f "
        "rhs_norm=%.6g res_norm=%.6g) max_speed=%.4f ms=%.1f\n",
        (long long)cells_,
        r_result.passes,
        substeps,
        active,
        r_result.pressure_iterations,
        int(params_host_[P_MAX_ITERATIONS] + 0.5f),
        double(r_result.residual),
        double(r_result.divergence_before),
        double(r_result.divergence_after),
        double(params_host_[P_DIV_BEFORE_SUM]),
        double(params_host_[P_DIV_AFTER_SUM]),
        double(params_host_[P_UNKNOWN_COUNT]),
        double(params_host_[P_RHS_NORM]),
        double(params_host_[P_RES_NORM]),
        double(max_speed),
        r_result.milliseconds);
  }
  return true;
}

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Brick grid (host side) implementations
 *
 * See the declarations above the device solver. The builder marks every 4^3 brick that carries
 * anything the solve may read, grows the set by the same voxel margin the dense box uses, and sorts
 * the bricks by (z, y, x) so the slot order (and therefore every reduction order) is deterministic.
 * `BLENDER_IGNIS_BRICK_SELFTEST=1` round trips the dense fields through the pool and compares them
 * bit for bit.
 * \{ */

/** True when the voxel carries anything the solve may read: a mask bit or a non zero field. */
static bool brick_voxel_is_wanted(const IgnisDeviceFrame &frame, const int64_t index)
{
  if (frame.mask[index] != 0) {
    return true;
  }
  if (!frame.current_mask.is_empty() && frame.current_mask[index] != 0) {
    return true;
  }
  if (!frame.carve_mask.is_empty() && frame.carve_mask[index] != 0) {
    return true;
  }
  for (int k = 0; k < 4; k++) {
    if (frame.scalars[index * 4 + k] != 0.0f) {
      return true;
    }
  }
  for (int k = 0; k < 3; k++) {
    if (frame.velocity[index * 3 + k] != 0.0f) {
      return true;
    }
  }
  if (frame.pressure[index] != 0.0f) {
    return true;
  }
  for (int k = 0; k < 7; k++) {
    if (frame.source[index * 7 + k] != 0.0f) {
      return true;
    }
  }
  return false;
}

static IgnisBrickGrid ignis_bricks_build(const IgnisDeviceFrame &frame,
                                         const int margin_voxels,
                                         const bool force_dense)
{
  IgnisBrickGrid bricks;
  bricks.dims = frame.dims;
  bricks.brick_dims = int3((frame.dims.x + IGNIS_BRICK_EDGE - 1) / IGNIS_BRICK_EDGE,
                           (frame.dims.y + IGNIS_BRICK_EDGE - 1) / IGNIS_BRICK_EDGE,
                           (frame.dims.z + IGNIS_BRICK_EDGE - 1) / IGNIS_BRICK_EDGE);
  const int bx = bricks.brick_dims.x;
  const int by = bricks.brick_dims.y;
  const int bz = bricks.brick_dims.z;
  if (bx <= 0 || by <= 0 || bz <= 0) {
    return bricks;
  }
  bricks.tile_count = bx * by * bz;
  bricks.slot_map.reinitialize(bricks.tile_count);
  bricks.slot_map.fill(-1);

  Array<int> active;
  active.reinitialize(bricks.tile_count);
  active.fill(0);
  const int64_t cells = int64_t(frame.dims.x) * int64_t(frame.dims.y) * int64_t(frame.dims.z);
  if (force_dense) {
    active.fill(1);
  }
  else {
    for (int64_t i = 0; i < cells; i++) {
      if (!brick_voxel_is_wanted(frame, i)) {
        continue;
      }
      const int x = int(i % frame.dims.x);
      const int y = int((i / frame.dims.x) % frame.dims.y);
      const int z = int(i / (int64_t(frame.dims.x) * frame.dims.y));
      active[(x / IGNIS_BRICK_EDGE) + bx * ((y / IGNIS_BRICK_EDGE) +
                                            by * (z / IGNIS_BRICK_EDGE))] = 1;
    }
    /* The dense box gives the solver `margin_voxels` of empty room around the content (a plume has
     * to move and the pressure domain grows). The pool needs the same room, so grow the brick set by
     * `ceil(margin / 4)` bricks in every direction. Without it the flow would be clipped at the
     * content boundary. */
    const int margin_tiles = (margin_voxels + IGNIS_BRICK_EDGE - 1) / IGNIS_BRICK_EDGE;
    Array<int> grown;
    for (int layer = 0; layer < margin_tiles; layer++) {
      grown.reinitialize(bricks.tile_count);
      grown.fill(0);
      for (int z = 0; z < bz; z++) {
        for (int y = 0; y < by; y++) {
          for (int x = 0; x < bx; x++) {
            const int tile = x + bx * (y + by * z);
            if (active[tile] != 0) {
              grown[tile] = 1;
              continue;
            }
            bool near = false;
            for (int dz = -1; dz <= 1 && !near; dz++) {
              for (int dy = -1; dy <= 1 && !near; dy++) {
                for (int dx = -1; dx <= 1 && !near; dx++) {
                  const int nx = x + dx, ny = y + dy, nz = z + dz;
                  if (nx < 0 || ny < 0 || nz < 0 || nx >= bx || ny >= by || nz >= bz) {
                    continue;
                  }
                  near = active[nx + bx * (ny + by * nz)] != 0;
                }
              }
            }
            grown[tile] = near ? 1 : 0;
          }
        }
      }
      active = grown;
    }
  }

  /* Deterministic slot order: z, then y, then x. */
  for (int z = 0; z < bz; z++) {
    for (int y = 0; y < by; y++) {
      for (int x = 0; x < bx; x++) {
        const int tile = x + bx * (y + by * z);
        if (!active[tile]) {
          continue;
        }
        bricks.slot_map[tile] = int(bricks.coords.size());
        bricks.coords.append(
            int3(x * IGNIS_BRICK_EDGE, y * IGNIS_BRICK_EDGE, z * IGNIS_BRICK_EDGE));
      }
    }
  }
  bricks.brick_count = int(bricks.coords.size());
  bricks.pool_cells = int64_t(bricks.brick_count) * IGNIS_BRICK_CELLS;

  bricks.neighbor.reinitialize(bricks.brick_count * 6);
  bricks.neighbor.fill(-1);
  for (int slot = 0; slot < bricks.brick_count; slot++) {
    const int3 origin = bricks.coords[slot];
    const int ox = origin.x / IGNIS_BRICK_EDGE;
    const int oy = origin.y / IGNIS_BRICK_EDGE;
    const int oz = origin.z / IGNIS_BRICK_EDGE;
    for (int axis = 0; axis < 3; axis++) {
      for (int sign = 0; sign < 2; sign++) {
        int nx = ox, ny = oy, nz = oz;
        if (axis == 0) {
          nx += sign == 0 ? 1 : -1;
        }
        else if (axis == 1) {
          ny += sign == 0 ? 1 : -1;
        }
        else {
          nz += sign == 0 ? 1 : -1;
        }
        int value = -1;
        if (nx >= 0 && ny >= 0 && nz >= 0 && nx < bx && ny < by && nz < bz) {
          value = bricks.slot_map[nx + bx * (ny + by * nz)];
        }
        bricks.neighbor[slot * 6 + axis * 2 + sign] = value;
      }
    }
  }
  return bricks;
}

/** Copies a dense field (one value per voxel, `stride` floats per voxel) into the brick pool. */
static void bricks_scatter_float(const IgnisBrickGrid &bricks,
                                 const Span<const float> src,
                                 const int stride,
                                 const MutableSpan<float> dst)
{
  dst.fill(0.0f);
  for (int slot = 0; slot < bricks.brick_count; slot++) {
    const int3 origin = bricks.coords[slot];
    for (int lz = 0; lz < IGNIS_BRICK_EDGE; lz++) {
      for (int ly = 0; ly < IGNIS_BRICK_EDGE; ly++) {
        for (int lx = 0; lx < IGNIS_BRICK_EDGE; lx++) {
          const int x = origin.x + lx;
          const int y = origin.y + ly;
          const int z = origin.z + lz;
          if (x >= bricks.dims.x || y >= bricks.dims.y || z >= bricks.dims.z) {
            continue;
          }
          const int64_t dense = int64_t(x) + int64_t(bricks.dims.x) *
                                                  (int64_t(y) + int64_t(bricks.dims.y) * z);
          const int64_t pool = int64_t(slot) * IGNIS_BRICK_CELLS +
                               lx + IGNIS_BRICK_EDGE * (ly + IGNIS_BRICK_EDGE * lz);
          for (int k = 0; k < stride; k++) {
            dst[pool * stride + k] = src[dense * stride + k];
          }
        }
      }
    }
  }
}

static void bricks_gather_float(const IgnisBrickGrid &bricks,
                                const Span<const float> src,
                                const int stride,
                                const MutableSpan<float> dst)
{
  dst.fill(0.0f);
  for (int slot = 0; slot < bricks.brick_count; slot++) {
    const int3 origin = bricks.coords[slot];
    for (int lz = 0; lz < IGNIS_BRICK_EDGE; lz++) {
      for (int ly = 0; ly < IGNIS_BRICK_EDGE; ly++) {
        for (int lx = 0; lx < IGNIS_BRICK_EDGE; lx++) {
          const int x = origin.x + lx;
          const int y = origin.y + ly;
          const int z = origin.z + lz;
          if (x >= bricks.dims.x || y >= bricks.dims.y || z >= bricks.dims.z) {
            continue;
          }
          const int64_t dense = int64_t(x) + int64_t(bricks.dims.x) *
                                                  (int64_t(y) + int64_t(bricks.dims.y) * z);
          const int64_t pool = int64_t(slot) * IGNIS_BRICK_CELLS +
                               lx + IGNIS_BRICK_EDGE * (ly + IGNIS_BRICK_EDGE * lz);
          for (int k = 0; k < stride; k++) {
            dst[dense * stride + k] = src[pool * stride + k];
          }
        }
      }
    }
  }
}

static void bricks_scatter_int(const IgnisBrickGrid &bricks,
                               const Span<const int> src,
                               const MutableSpan<int> dst)
{
  dst.fill(0);
  for (int slot = 0; slot < bricks.brick_count; slot++) {
    const int3 origin = bricks.coords[slot];
    for (int lz = 0; lz < IGNIS_BRICK_EDGE; lz++) {
      for (int ly = 0; ly < IGNIS_BRICK_EDGE; ly++) {
        for (int lx = 0; lx < IGNIS_BRICK_EDGE; lx++) {
          const int x = origin.x + lx;
          const int y = origin.y + ly;
          const int z = origin.z + lz;
          if (x >= bricks.dims.x || y >= bricks.dims.y || z >= bricks.dims.z) {
            continue;
          }
          const int64_t dense = int64_t(x) + int64_t(bricks.dims.x) *
                                                  (int64_t(y) + int64_t(bricks.dims.y) * z);
          const int64_t pool = int64_t(slot) * IGNIS_BRICK_CELLS +
                               lx + IGNIS_BRICK_EDGE * (ly + IGNIS_BRICK_EDGE * lz);
          dst[pool] = src[dense];
        }
      }
    }
  }
}

static void bricks_gather_int(const IgnisBrickGrid &bricks,
                              const Span<const int> src,
                              const MutableSpan<int> dst)
{
  dst.fill(0);
  for (int slot = 0; slot < bricks.brick_count; slot++) {
    const int3 origin = bricks.coords[slot];
    for (int lz = 0; lz < IGNIS_BRICK_EDGE; lz++) {
      for (int ly = 0; ly < IGNIS_BRICK_EDGE; ly++) {
        for (int lx = 0; lx < IGNIS_BRICK_EDGE; lx++) {
          const int x = origin.x + lx;
          const int y = origin.y + ly;
          const int z = origin.z + lz;
          if (x >= bricks.dims.x || y >= bricks.dims.y || z >= bricks.dims.z) {
            continue;
          }
          const int64_t dense = int64_t(x) + int64_t(bricks.dims.x) *
                                                  (int64_t(y) + int64_t(bricks.dims.y) * z);
          const int64_t pool = int64_t(slot) * IGNIS_BRICK_CELLS +
                               lx + IGNIS_BRICK_EDGE * (ly + IGNIS_BRICK_EDGE * lz);
          dst[dense] = src[pool];
        }
      }
    }
  }
}

/** Builds the brick grid, round trips every field through the pool and reports the counts. */
static void ignis_bricks_selftest(const IgnisDeviceFrame &frame,
                                  const int margin_voxels,
                                  const bool force_dense)
{
  const IgnisBrickGrid content = ignis_bricks_build(frame, 0, force_dense);
  const IgnisBrickGrid bricks = ignis_bricks_build(frame, margin_voxels, force_dense);
  const int64_t cells = int64_t(frame.dims.x) * int64_t(frame.dims.y) * int64_t(frame.dims.z);
  const int64_t pool = bricks.pool_cells;

  Array<float> pool_scalars;
  Array<float> pool_velocity;
  Array<float> pool_pressure;
  Array<int> pool_mask;
  pool_scalars.reinitialize(int(pool * 4));
  pool_velocity.reinitialize(int(pool * 3));
  pool_pressure.reinitialize(int(pool));
  pool_mask.reinitialize(int(pool));
  bricks_scatter_float(bricks,
                       Span<const float>(frame.scalars.data(), frame.scalars.size()),
                       4,
                       pool_scalars.as_mutable_span());
  bricks_scatter_float(bricks,
                       Span<const float>(frame.velocity.data(), frame.velocity.size()),
                       3,
                       pool_velocity.as_mutable_span());
  bricks_scatter_float(bricks,
                       Span<const float>(frame.pressure.data(), frame.pressure.size()),
                       1,
                       pool_pressure.as_mutable_span());
  bricks_scatter_int(bricks,
                     Span<const int>(frame.mask.data(), frame.mask.size()),
                     pool_mask.as_mutable_span());

  Array<float> back_scalars;
  Array<float> back_velocity;
  Array<float> back_pressure;
  Array<int> back_mask;
  back_scalars.reinitialize(int(cells * 4));
  back_velocity.reinitialize(int(cells * 3));
  back_pressure.reinitialize(int(cells));
  back_mask.reinitialize(int(cells));
  bricks_gather_float(bricks,
                      Span<const float>(pool_scalars.data(), pool_scalars.size()),
                      4,
                      back_scalars.as_mutable_span());
  bricks_gather_float(bricks,
                      Span<const float>(pool_velocity.data(), pool_velocity.size()),
                      3,
                      back_velocity.as_mutable_span());
  bricks_gather_float(bricks,
                      Span<const float>(pool_pressure.data(), pool_pressure.size()),
                      1,
                      back_pressure.as_mutable_span());
  bricks_gather_int(
      bricks, Span<const int>(pool_mask.data(), pool_mask.size()), back_mask.as_mutable_span());

  const bool ok =
      std::memcmp(back_scalars.data(), frame.scalars.data(), size_t(cells * 4) * sizeof(float)) == 0 &&
      std::memcmp(back_velocity.data(), frame.velocity.data(), size_t(cells * 3) * sizeof(float)) == 0 &&
      std::memcmp(back_pressure.data(), frame.pressure.data(), size_t(cells) * sizeof(float)) == 0 &&
      std::memcmp(back_mask.data(), frame.mask.data(), size_t(cells) * sizeof(int)) == 0;

  /* How many bricks carry smoke / the velocity band, which is what the passes will dispatch. */
  int smoke_bricks = 0;
  int band_bricks = 0;
  for (int slot = 0; slot < bricks.brick_count; slot++) {
    bool smoke = false;
    bool band = false;
    const int3 origin = bricks.coords[slot];
    for (int lz = 0; lz < IGNIS_BRICK_EDGE; lz++) {
      for (int ly = 0; ly < IGNIS_BRICK_EDGE; ly++) {
        for (int lx = 0; lx < IGNIS_BRICK_EDGE; lx++) {
          const int x = origin.x + lx;
          const int y = origin.y + ly;
          const int z = origin.z + lz;
          if (x >= bricks.dims.x || y >= bricks.dims.y || z >= bricks.dims.z) {
            continue;
          }
          const int64_t dense = int64_t(x) + int64_t(bricks.dims.x) *
                                                  (int64_t(y) + int64_t(bricks.dims.y) * z);
          if (std::abs(frame.scalars[dense * 4 + 0]) > 1e-4f) {
            smoke = true;
          }
          if (std::abs(frame.velocity[dense * 3 + 0]) > 1e-6f ||
              std::abs(frame.velocity[dense * 3 + 1]) > 1e-6f ||
              std::abs(frame.velocity[dense * 3 + 2]) > 1e-6f)
          {
            band = true;
          }
        }
      }
    }
    smoke_bricks += smoke ? 1 : 0;
    band_bricks += band ? 1 : 0;
  }

  printf("IGNIS_BRICKS: cells=%lld bricks=%d/%d (%.1f%%) content=%d margin=%d "
         "pool=%lld (%.2fx cells) smoke=%d band=%d fields=%s\n",
         (long long)cells,
         bricks.brick_count,
         bricks.tile_count,
         bricks.tile_count > 0 ?
             100.0 * double(bricks.brick_count) / double(bricks.tile_count) :
             0.0,
         content.brick_count,
         margin_voxels,
         (long long)pool,
         cells > 0 ? double(pool) / double(cells) : 0.0,
         smoke_bricks,
         band_bricks,
         ok ? "ok" : "MISMATCH");
}

/** \} */

}  // namespace

bool ignis_device_available()
{
  const char *value = std::getenv("BLENDER_IGNIS_GPU");
  if (value != nullptr && value[0] == '0') {
    return false;
  }
  return ha_gpu_available();
}

bool ignis_device_solve(IgnisDeviceFrame &frame, IgnisDeviceResult &r_result)
{
  HaContextGuard guard;
  if (!guard.is_valid()) {
    const char *error = ha_context_error();
    r_result.reason = (error != nullptr && error[0] != '\0') ? std::string(error) :
                                                              std::string("no GPU context");
    return false;
  }
  /* Buffers are freed and reallocated every evaluation, so the binding cache (keyed by buffer
   * pointer) must not survive into this one. */
  ha_bindings_invalidate();
  HaIgnisSolver solver;
  const bool ok = solver.solve(frame, r_result);
  solver.release();
  return ok;
}

}  // namespace blender::geometry::ha
