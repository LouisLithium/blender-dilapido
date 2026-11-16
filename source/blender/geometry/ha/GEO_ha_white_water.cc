/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "BLI_array.hh"
#include "BLI_math_base.hh"
#include "BLI_time.h"
#include "BLI_math_vector.hh"
#include "BLI_vector.hh"

#include "GPU_shader.hh"
#include "GPU_state.hh"

#include "GEO_ha_context.hh"
#include "GEO_ha_white_water.hh"

namespace blender::geometry::ha {

namespace {

/* One float based parameter block shared by the kernels. Keep the indices in sync. */
enum WwParam : int {
  P_COUNT = 0,
  P_COHESION = 1,
  P_COHESION_RADIUS = 2,
  P_SEPARATION = 3,
  P_SEPARATION_RADIUS = 4,
  P_DAMPING = 5,
  P_SAME_TYPE = 6,
  P_DT = 7,
  P_MAX_DELTA = 8,
  P_COUNT_END = 9,
};

const char *GLSL_PRELUDE =
    "#define WW_COUNT params[0]\n"
    "#define WW_COHESION params[1]\n"
    "#define WW_COHESION_RADIUS params[2]\n"
    "#define WW_SEPARATION params[3]\n"
    "#define WW_SEPARATION_RADIUS params[4]\n"
    "#define WW_DAMPING params[5]\n"
    "#define WW_SAME_TYPE params[6]\n"
    "#define WW_DT params[7]\n"
    "#define WW_MAX_DELTA params[8]\n";

/**
 * One invocation per particle: the same neighbour list, the same accumulation order and the same
 * Jacobi semantics as the CPU pass, so the only thing that changes is where the loop runs.
 */
const char *KERNEL_FORCES =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(WW_COUNT)) { return; }\n"
    "  vec3 pi = vec3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);\n"
    "  vec3 vi = vec3(velocities[i * 3], velocities[i * 3 + 1], velocities[i * 3 + 2]);\n"
    "  int type_i = types[i];\n"
    "  float separation_radius = WW_SEPARATION_RADIUS;\n"
    "  float cohesion_radius = WW_COHESION_RADIUS;\n"
    "  vec3 delta = vec3(0.0);\n"
    "  int count = 0;\n"
    "  int begin = offsets[i];\n"
    "  int end = offsets[i + 1];\n"
    "  for (int k = begin; k < end; k++) {\n"
    "    int j = sorted[k];\n"
    "    vec3 to_j = vec3(positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]) - pi;\n"
    "    float distance = sqrt(dot(to_j, to_j));\n"
    "    if (WW_SAME_TYPE > 0.5 && types[j] != type_i) { continue; }\n"
    "    count++;\n"
    "    vec3 direction = to_j / max(distance, 1e-6);\n"
    "    if (WW_SEPARATION > 0.0 && distance < separation_radius) {\n"
    "      float factor = 1.0 - distance / max(separation_radius, 1e-6);\n"
    "      delta -= direction * (WW_SEPARATION * factor * WW_DT);\n"
    "    }\n"
    "    if (WW_COHESION > 0.0 && distance < cohesion_radius) {\n"
    "      float factor = 1.0 - distance / max(cohesion_radius, 1e-6);\n"
    "      delta += direction * (WW_COHESION * factor * WW_DT);\n"
    "    }\n"
    "    if (WW_DAMPING > 0.0 && distance < cohesion_radius) {\n"
    "      float factor = 1.0 - distance / max(cohesion_radius, 1e-6);\n"
    "      vec3 vj = vec3(velocities[j * 3], velocities[j * 3 + 1], velocities[j * 3 + 2]);\n"
    "      delta += (vj - vi) * (WW_DAMPING * factor * WW_DT);\n"
    "    }\n"
    "  }\n"
    "  if (count > 0) { delta /= float(count); }\n"
    "  if (dot(delta, delta) > WW_MAX_DELTA * WW_MAX_DELTA) {\n"
    "    delta = normalize(delta) * WW_MAX_DELTA;\n"
    "  }\n"
    "  deltas[i * 3] = delta.x;\n"
    "  deltas[i * 3 + 1] = delta.y;\n"
    "  deltas[i * 3 + 2] = delta.z;\n"
    "}\n";

/** Measures the host/device round trip of a buffer of the requested size (MB). */
void transfer_probe()
{
  const char *env = std::getenv("BLENDER_WHITE_WATER_TRANSFER_PROBE");
  const int megabytes = env != nullptr ? int(std::atoi(env)) : 15;
  if (megabytes <= 0) {
    return;
  }
  const int count = int((int64_t(megabytes) * 1024 * 1024) / 4);
  HaBufferInt buffer;
  if (!buffer.ensure(count)) {
    fprintf(stderr, "TRANSFER probe: allocation of %d MB failed\n", megabytes);
    return;
  }
  Vector<int> data(count, 1);
  Vector<int> out(count);
  const double t0 = BLI_time_now_seconds();
  buffer.upload(data.as_span());
  GPU_finish();
  const double t1 = BLI_time_now_seconds();
  buffer.download(out.as_mutable_span());
  const double t2 = BLI_time_now_seconds();
  buffer.download(out.as_mutable_span());
  const double t3 = BLI_time_now_seconds();
  fprintf(stderr,
          "TRANSFER probe %d MB: upload %.2f ms, download %.2f ms, download again %.2f ms\n",
          megabytes,
          (t1 - t0) * 1000.0,
          (t2 - t1) * 1000.0,
          (t3 - t2) * 1000.0);
  fflush(stderr);
}

void dispatch_pass(gpu::Shader *shader,
                   const Span<gpu::StorageBuf *> buffers,
                   const int element_count,
                   const int group_size)
{
  GPU_shader_bind(shader);
  ha_bind_storage_buffers(buffers);
  ha_dispatch(shader, element_count, group_size);
  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE);
}

}  // namespace

bool ha_white_water_forces(const Span<const float3> positions,
                           const MutableSpan<float3> r_velocities,
                           const Span<const int> types,
                           const Span<const int> offsets,
                           const Span<const int> indices,
                           const WhiteWaterForceSettings &settings,
                           std::string &r_reason)
{
  r_reason.clear();
  const int count = int(positions.size());
  if (count < 2) {
    r_reason = "fewer than two particles";
    return false;
  }
  if (r_velocities.size() != count || types.size() != count || offsets.size() != count + 1 ||
      indices.is_empty())
  {
    r_reason = "the particle attributes have the wrong size";
    return false;
  }

  const bool profile = std::getenv("BLENDER_WHITE_WATER_GPU_DEBUG") != nullptr;
  double profile_time = profile ? BLI_time_now_seconds() : 0.0;
  auto lap = [&](const char *label) {
    if (!profile) {
      return;
    }
    const double now = BLI_time_now_seconds();
    fprintf(stderr, "WHITE_WATER_GPU %-8s %6.2f ms\n", label, (now - profile_time) * 1000.0);
    profile_time = now;
  };

  HaContextGuard guard;
  if (!guard.is_valid()) {
    r_reason = ha_context_error();
    if (r_reason.empty()) {
      r_reason = "no GPU context";
    }
    return false;
  }

  HaBuffer positions_buf;
  HaBuffer velocities_buf;
  HaBuffer deltas_buf;
  HaBuffer params_buf;
  HaBufferInt types_buf;
  HaBufferInt offsets_buf;
  HaBufferInt indices_buf;
  if (!positions_buf.ensure(int64_t(count) * 3) || !velocities_buf.ensure(int64_t(count) * 3) ||
      !deltas_buf.ensure(int64_t(count) * 3) || !params_buf.ensure(P_COUNT_END) ||
      !types_buf.ensure(count) || !offsets_buf.ensure(int64_t(count) + 1) ||
      !indices_buf.ensure(indices.size()))
  {
    r_reason = "the device buffers could not be allocated";
    return false;
  }

  {
    Array<float> flat(int64_t(count) * 3);
    for (const int i : IndexRange(count)) {
      flat[i * 3 + 0] = positions[i].x;
      flat[i * 3 + 1] = positions[i].y;
      flat[i * 3 + 2] = positions[i].z;
    }
    positions_buf.upload(flat.as_span());
    for (const int i : IndexRange(count)) {
      flat[i * 3 + 0] = r_velocities[i].x;
      flat[i * 3 + 1] = r_velocities[i].y;
      flat[i * 3 + 2] = r_velocities[i].z;
    }
    velocities_buf.upload(flat.as_span());
  }
  {
    Array<int> flat(count);
    for (const int i : IndexRange(count)) {
      flat[i] = types[i];
    }
    types_buf.upload(flat.as_span());
  }
  {
    Vector<int> flat(offsets);
    offsets_buf.upload(flat.as_span());
  }
  {
    Vector<int> flat(indices);
    indices_buf.upload(flat.as_span());
  }

  Array<float> param_values(P_COUNT_END, 0.0f);
  param_values[P_COUNT] = float(count);
  param_values[P_COHESION] = settings.cohesion;
  param_values[P_COHESION_RADIUS] = settings.cohesion_radius;
  param_values[P_SEPARATION] = settings.separation;
  param_values[P_SEPARATION_RADIUS] = settings.separation_radius;
  param_values[P_DAMPING] = settings.cluster_damping;
  param_values[P_SAME_TYPE] = settings.same_type_only ? 1.0f : 0.0f;
  param_values[P_DT] = settings.delta_time;
  param_values[P_MAX_DELTA] = 0.25f;
  params_buf.upload(param_values.as_span());
  lap("upload");

  const HaShaderResource resources[] = {
      {0, HaShaderQualifier::Read, "float", "positions[]"},
      {1, HaShaderQualifier::Read, "float", "velocities[]"},
      {2, HaShaderQualifier::Read, "int", "types[]"},
      {3, HaShaderQualifier::Read, "int", "offsets[]"},
      {4, HaShaderQualifier::Read, "int", "sorted[]"},
      {5, HaShaderQualifier::Write, "float", "deltas[]"},
      {6, HaShaderQualifier::Read, "float", "params[]"},
  };
  gpu::Shader *forces = ha_shader_get_source(
      "white_water_forces_v2",
      std::string(GLSL_PRELUDE) + KERNEL_FORCES,
      Span<const HaShaderResource>(resources, 7),
      64);
  if (forces == nullptr) {
    r_reason = "the compute shader could not be created";
    return false;
  }

  dispatch_pass(forces,
                {positions_buf.buf(),
                 velocities_buf.buf(),
                 types_buf.buf(),
                 offsets_buf.buf(),
                 indices_buf.buf(),
                 deltas_buf.buf(),
                 params_buf.buf()},
                count,
                64);

  lap("dispatch");
  if (profile) {
    GPU_finish();
    lap("gpu");
  }
  Array<float> deltas(int64_t(count) * 3);
  deltas_buf.download(deltas.as_mutable_span());
  lap("download");
  for (const int i : IndexRange(count)) {
    r_velocities[i] += float3(deltas[i * 3], deltas[i * 3 + 1], deltas[i * 3 + 2]);
  }
  if (std::getenv("BLENDER_WHITE_WATER_TRANSFER_PROBE") != nullptr) {
    static bool probed = false;
    if (!probed) {
      probed = true;
      transfer_probe();
    }
  }
  return true;
}

}  // namespace blender::geometry::ha
