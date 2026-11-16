/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <cmath>
#include <string>
#include <utility>

#include "BLI_array.hh"

#include "BLI_math_base.hh"
#include "BLI_math_vector.hh"
#include "BLI_string_ref.hh"

#include "GPU_state.hh"

#include "GEO_ha_context.hh"
#include "GEO_ha_fluid_surface.hh"

namespace blender::geometry::ha {

namespace {

/* One float based parameter block, shared by the kernels. Keep the indices in sync. */
enum FsParam : int {
  P_CELLS = 0,
  P_DIMS_X = 1,
  P_DIMS_Y = 2,
  P_DIMS_Z = 3,
  P_ORIGIN_X = 4,
  P_ORIGIN_Y = 5,
  P_ORIGIN_Z = 6,
  P_VOXEL = 7,
  P_PARTICLES = 8,
  P_RADIUS_SCALE = 9,
  P_NORMALIZER = 10,
  P_MPM = 11,
  P_FIXED = 12,
  P_EDGE = 13,
  P_VALUE_SCALE = 14,
  P_COUNT = 15,
};

const char *GLSL_PRELUDE =
    "#define FS_CELLS params[0]\n"
    "#define FS_DX params[1]\n"
    "#define FS_DY params[2]\n"
    "#define FS_DZ params[3]\n"
    "#define FS_ORIGIN vec3(params[4], params[5], params[6])\n"
    "#define FS_VOXEL params[7]\n"
    "#define FS_PARTICLES params[8]\n"
    "#define FS_RADIUS_SCALE params[9]\n"
    "#define FS_NORMALIZER params[10]\n"
    "#define FS_MPM params[11]\n"
    "#define FS_FIXED params[12]\n"
    "#define FS_EDGE params[13]\n"
    "#define FS_VALUE_SCALE params[14]\n"
    "\n"
    "float fs_bspline(float u)\n"
    "{\n"
    "  float t = abs(u);\n"
    "  if (t < 0.5) { return 0.75 - t * t; }\n"
    "  if (t < 1.5) { float r = 1.5 - t; return 0.5 * r * r; }\n"
    "  return 0.0;\n"
    "}\n";

/** One invocation per particle, scattering its kernel into the fixed point field. */
const char *KERNEL_SPLAT =
    "void main()\n"
    "{\n"
    "  int p = int(gl_GlobalInvocationID.x);\n"
    "  if (p >= int(FS_PARTICLES)) { return; }\n"
    "  vec3 position = vec3(positions[p * 3], positions[p * 3 + 1], positions[p * 3 + 2]);\n"
    "  int dx = int(FS_DX);\n"
    "  int dy = int(FS_DY);\n"
    "  int dz = int(FS_DZ);\n"
    "  if (dx <= 0 || dy <= 0 || dz <= 0) { return; }\n"
    "  vec3 local = (position - FS_ORIGIN) / FS_VOXEL;\n"
    "  int base_x = int(floor(local.x));\n"
    "  int base_y = int(floor(local.y));\n"
    "  int base_z = int(floor(local.z));\n"
    "  float h;\n"
    "  float amplitude;\n"
    "  int reach;\n"
    "  bool mpm = FS_MPM > 0.5;\n"
    "  if (mpm) {\n"
    "    float natural = pow(max(volumes[p], 1e-20), 1.0 / 3.0);\n"
    "    float scale = max(1.0, 0.5 * FS_RADIUS_SCALE);\n"
    "    h = scale * natural;\n"
    "    amplitude = volumes[p] / (h * h * h);\n"
    "    reach = int(ceil(1.5 * h / FS_VOXEL));\n"
    "  }\n"
    "  else {\n"
    "    h = FS_RADIUS_SCALE * max(radii[p], 1e-5);\n"
    "    float h2 = h * h;\n"
    "    float kernel = (315.0 / (64.0 * 3.141592653589793)) / (h2 * h2 * h2 * h2 * h);\n"
    "    amplitude = volumes[p] * kernel / max(FS_NORMALIZER, 1e-6);\n"
    "    reach = int(ceil(h / FS_VOXEL));\n"
    "  }\n"
    "  float h2 = h * h;\n"
    "  float inv_h = 1.0 / h;\n"
    "  for (int k = base_z - reach; k <= base_z + reach; k++) {\n"
    "    if (k < 0 || k >= dz) { continue; }\n"
    "    float tz = (float(k) - local.z);\n"
    "    for (int j = base_y - reach; j <= base_y + reach; j++) {\n"
    "      if (j < 0 || j >= dy) { continue; }\n"
    "      float ty = (float(j) - local.y);\n"
    "      for (int i = base_x - reach; i <= base_x + reach; i++) {\n"
    "        if (i < 0 || i >= dx) { continue; }\n"
    "        float tx = (float(i) - local.x);\n"
    "        float weight;\n"
    "        if (mpm) {\n"
    "          weight = fs_bspline(tx * FS_VOXEL * inv_h) * fs_bspline(ty * FS_VOXEL * inv_h) *\n"
    "                   fs_bspline(tz * FS_VOXEL * inv_h) * amplitude;\n"
    "        }\n"
    "        else {\n"
    "          float d2 = (tx * tx + ty * ty + tz * tz) * FS_VOXEL * FS_VOXEL;\n"
    "          if (d2 >= h2) { continue; }\n"
    "          float falloff = h2 - d2;\n"
    "          weight = falloff * falloff * falloff * amplitude;\n"
    "        }\n"
    "        int index = ((k * dy) + j) * dx + i;\n"
    "        atomicAdd(field_fixed[index], int(weight * FS_FIXED));\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n";

const char *KERNEL_FINALIZE =
    "void main()\n"
    "{\n"
    "  int c = int(gl_GlobalInvocationID.x);\n"
    "  if (c >= int(FS_CELLS)) { return; }\n"
    "  field[c] = float(field_fixed[c]) / FS_FIXED;\n"
    "}\n";

/**
 * One relaxation pass. Only cells that already hold a value are updated, exactly like the CPU pass
 * that iterates the active voxels, and an empty neighbour counts as the center value.
 */
const char *KERNEL_SMOOTH =
    "void main()\n"
    "{\n"
    "  int c = int(gl_GlobalInvocationID.x);\n"
    "  if (c >= int(FS_CELLS)) { return; }\n"
    "  float center = field_in[c];\n"
    "  if (center == 0.0) { field_out[c] = 0.0; return; }\n"
    "  int dx = int(FS_DX);\n"
    "  int dy = int(FS_DY);\n"
    "  int dz = int(FS_DZ);\n"
    "  int x = c % dx;\n"
    "  int y = (c / dx) % dy;\n"
    "  int z = c / (dx * dy);\n"
    "  float sigma = FS_EDGE * max(FS_VALUE_SCALE, 1e-3);\n"
    "  float inv_sigma_sq = sigma > 0.0 ? 1.0 / (sigma * sigma) : 0.0;\n"
    "  float sum = 0.0;\n"
    "  float weight_sum = 0.0;\n"
    "  for (int axis = 0; axis < 3; axis++) {\n"
    "    for (int sign_i = 0; sign_i < 2; sign_i++) {\n"
    "      int offset = 0;\n"
    "      bool inside = true;\n"
    "      if (axis == 0) { offset = 1; inside = sign_i == 0 ? x + 1 < dx : x > 0; }\n"
    "      else if (axis == 1) { offset = dx; inside = sign_i == 0 ? y + 1 < dy : y > 0; }\n"
    "      else { offset = dx * dy; inside = sign_i == 0 ? z + 1 < dz : z > 0; }\n"
    "      float neighbour = center;\n"
    "      if (inside) {\n"
    "        int index = sign_i == 0 ? c + offset : c - offset;\n"
    "        float value = field_in[index];\n"
    "        neighbour = value != 0.0 ? value : center;\n"
    "      }\n"
    "      float weight = 1.0;\n"
    "      if (inv_sigma_sq > 0.0) {\n"
    "        float difference = neighbour - center;\n"
    "        weight = exp(-difference * difference * inv_sigma_sq);\n"
    "      }\n"
    "      sum += neighbour * weight;\n"
    "      weight_sum += weight;\n"
    "    }\n"
    "  }\n"
    "  float average = weight_sum > 0.0 ? sum / weight_sum : center;\n"
    "  field_out[c] = center + (average - center) * 0.5;\n"
    "}\n";

gpu::Shader *kernel(StringRef name,
                    StringRef source,
                    const HaShaderResource *resources,
                    const int resource_count,
                    const int group_size)
{
  return ha_shader_get_source(
      name, source, Span<const HaShaderResource>(resources, resource_count), group_size);
}

}  // namespace

bool ha_fluid_surface_field(const Span<const float3> positions,
                            const Span<const float> radii,
                            const Span<const float> volumes,
                            const FluidSurfaceSettings &settings,
                            FluidSurfaceField &r_field,
                            std::string &r_reason)
{
  r_reason.clear();
  const int count = int(positions.size());
  if (count == 0) {
    r_reason = "no particles";
    return false;
  }
  if (radii.size() != count || volumes.size() != count) {
    r_reason = "the particle attributes have the wrong size";
    return false;
  }

  HaContextGuard guard;
  if (!guard.is_valid()) {
    r_reason = ha_context_error();
    if (r_reason.empty()) {
      r_reason = "no GPU context";
    }
    return false;
  }

  /* Kernel support and bounding box, straight from the CPU implementation. */
  float3 min = positions[0];
  float3 max = positions[0];
  float max_support = 0.0f;
  for (const int i : IndexRange(count)) {
    min = math::min(min, positions[i]);
    max = math::max(max, positions[i]);
    if (settings.mpm_mode) {
      const float natural = std::cbrt(std::max(volumes[i], 1e-20f));
      max_support = std::max(max_support, 1.5f * std::max(1.0f, 0.5f * settings.radius_scale) *
                                              natural);
    }
    else {
      max_support = std::max(max_support, settings.radius_scale * std::max(radii[i], 1e-5f));
    }
  }
  const float cell = settings.voxel_size > 0.0f ? settings.voxel_size : 0.5f * max_support;
  const float margin = max_support;
  const float3 extent = (max - min) + float3(2.0f * margin);
  int3 dims(std::max(1, int(std::ceil(extent.x / cell))),
            std::max(1, int(std::ceil(extent.y / cell))),
            std::max(1, int(std::ceil(extent.z / cell))));
  const int64_t cells = int64_t(dims.x) * dims.y * dims.z;
  if (cells > settings.max_cells) {
    r_reason = "the surface grid would need " + std::to_string(cells) + " cells";
    return false;
  }
  const float3 origin = min - float3(margin);

  HaBufferInt field_fixed;
  HaBuffer field;
  HaBuffer scratch;
  HaBuffer params;
  HaBuffer positions_buf;
  HaBuffer radii_buf;
  HaBuffer volumes_buf;
  if (!field_fixed.ensure(cells) || !field.ensure(cells) || !scratch.ensure(cells) ||
      !params.ensure(P_COUNT) || !positions_buf.ensure(int64_t(count) * 3) ||
      !radii_buf.ensure(count) || !volumes_buf.ensure(count))
  {
    r_reason = "the device buffers could not be allocated";
    return false;
  }
  field_fixed.clear_to_zero();
  {
    Vector<float> flat(int64_t(count) * 3);
    for (const int i : IndexRange(count)) {
      flat[i * 3 + 0] = positions[i].x;
      flat[i * 3 + 1] = positions[i].y;
      flat[i * 3 + 2] = positions[i].z;
    }
    positions_buf.upload(flat.as_span());
  }
  {
    Vector<float> flat(radii.begin(), radii.end());
    radii_buf.upload(flat.as_span());
  }
  {
    Vector<float> flat(volumes.begin(), volumes.end());
    volumes_buf.upload(flat.as_span());
  }

  constexpr float FIXED = 1048576.0f;
  Array<float> param_values(P_COUNT, 0.0f);
  param_values[P_CELLS] = float(cells);
  param_values[P_DIMS_X] = float(dims.x);
  param_values[P_DIMS_Y] = float(dims.y);
  param_values[P_DIMS_Z] = float(dims.z);
  param_values[P_ORIGIN_X] = origin.x;
  param_values[P_ORIGIN_Y] = origin.y;
  param_values[P_ORIGIN_Z] = origin.z;
  param_values[P_VOXEL] = cell;
  param_values[P_PARTICLES] = float(count);
  param_values[P_RADIUS_SCALE] = settings.radius_scale;
  param_values[P_NORMALIZER] = settings.normalizer;
  param_values[P_MPM] = settings.mpm_mode ? 1.0f : 0.0f;
  param_values[P_FIXED] = FIXED;
  param_values[P_EDGE] = settings.edge_preserving;
  param_values[P_VALUE_SCALE] = settings.value_scale;
  params.upload(param_values.as_span());

  const HaShaderResource splat_resources[] = {
      {0, HaShaderQualifier::Read, "float", "positions[]"},
      {1, HaShaderQualifier::Read, "float", "radii[]"},
      {2, HaShaderQualifier::Read, "float", "volumes[]"},
      {3, HaShaderQualifier::Read, "float", "params[]"},
      {4, HaShaderQualifier::ReadWrite, "int", "field_fixed[]"},
  };
  const HaShaderResource finalize_resources[] = {
      {0, HaShaderQualifier::Read, "int", "field_fixed[]"},
      {1, HaShaderQualifier::Write, "float", "field[]"},
      {2, HaShaderQualifier::Read, "float", "params[]"},
  };
  const HaShaderResource smooth_resources[] = {
      {0, HaShaderQualifier::Read, "float", "field_in[]"},
      {1, HaShaderQualifier::Write, "float", "field_out[]"},
      {2, HaShaderQualifier::Read, "float", "params[]"},
  };

  const std::string key = "fluid_surface_field_v1";
  gpu::Shader *splat = kernel(key + ".splat",
                              std::string(GLSL_PRELUDE) + KERNEL_SPLAT,
                              splat_resources,
                              5,
                              64);
  gpu::Shader *finalize = kernel(key + ".finalize",
                                 std::string(GLSL_PRELUDE) + KERNEL_FINALIZE,
                                 finalize_resources,
                                 3,
                                 256);
  gpu::Shader *smooth = kernel(key + ".smooth",
                               std::string(GLSL_PRELUDE) + KERNEL_SMOOTH,
                               smooth_resources,
                               3,
                               256);
  if (splat == nullptr || finalize == nullptr || smooth == nullptr) {
    r_reason = "the compute shaders could not be created";
    return false;
  }

  /* Splat. */
  ha_bind_storage_buffers({positions_buf.buf(),
                           radii_buf.buf(),
                           volumes_buf.buf(),
                           params.buf(),
                           field_fixed.buf()});
  ha_dispatch(splat, count, 64);
  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE);

  ha_bind_storage_buffers({field_fixed.buf(), field.buf(), params.buf()});
  ha_dispatch(finalize, int(cells), 256);
  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE);

  /* Relaxation, ping-pong between `field` and `scratch`. */
  gpu::StorageBuf *source = field.buf();
  gpu::StorageBuf *target = scratch.buf();
  for (const int pass : IndexRange(std::max(settings.smoothing, 0))) {
    UNUSED_VARS(pass);
    ha_bind_storage_buffers({source, target, params.buf()});
    ha_dispatch(smooth, int(cells), 256);
    GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE);
    std::swap(source, target);
  }

  r_field.dims = dims;
  r_field.origin = origin;
  r_field.voxel_size = cell;
  r_field.values.reinitialize(cells);
  HaBuffer *reader = source == field.buf() ? &field : &scratch;
  reader->download(r_field.values.as_mutable_span());
  return true;
}

}  // namespace blender::geometry::ha
