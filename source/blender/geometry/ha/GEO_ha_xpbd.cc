/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * GPU implementation of the XPBD fluid passes. This file currently contains the spatial grid and
 * the neighbour list (M87, first step); the solver passes are built on top of it.
 */

#include "GEO_ha_xpbd.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "BLI_array.hh"
#include "BLI_math_base.hh"
#include "BLI_vector.hh"

#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_storage_buffer.hh"

namespace blender::geometry::ha {

namespace {

constexpr int LOCAL_GROUP_SIZE = 64;
/** Elements of an array that one invocation of a scan pass walks serially. */
constexpr int SCAN_BLOCK_SIZE = HA_REDUCE_BLOCK_SIZE;
static_assert(SCAN_BLOCK_SIZE == 1024, "the kernels below hard code this block size");

bool ha_xpbd_debug_enabled()
{
  static const bool enabled = getenv("BLENDER_HA_DEBUG") != nullptr;
  return enabled;
}

void ha_xpbd_debug(const char *message)
{
  if (ha_xpbd_debug_enabled()) {
    /* stderr survives a crash, stdout is block buffered when piped. */
    fprintf(stderr, "HA xpbd: %s\n", message);
    fflush(stderr);
  }
}

double time_now_ms()
{
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/** Helpers shared by all kernels; the hash has to match `FluidParticleGrid::hash_cell`. */
const char *GLSL_PRELUDE =
    "#define HA_PROP 11\n"
    "#define HA_STATE 8\n"
    "uint ha_hash_cell(int cx, int cy, int cz)\n"
    "{\n"
    "  return (uint(cx) * 73856093u) ^ (uint(cy) * 19349663u) ^ (uint(cz) * 83492791u);\n"
    "}\n"
    "bool ha_is_finite(vec3 v)\n"
    "{\n"
    "  return v.x == v.x && v.y == v.y && v.z == v.z && abs(v.x) < 1e30 && abs(v.y) < 1e30 &&\n"
    "         abs(v.z) < 1e30;\n"
    "}\n";

/**
 * Binds the buffers to slots 0..n-1 and dispatches. The binds that are already in place (the
 * density and delta kernels have the same layout, for instance) are skipped by
 * #ha_bind_storage_buffers, and nothing is unbound: on this GPU a pass of a few hundred particles
 * costs nothing but its calls, so the binds used to be the whole frame.
 */
void dispatch_pass(gpu::Shader *shader, Span<gpu::StorageBuf *> buffers, const int element_count)
{
  if (shader == nullptr) {
    return;
  }
  GPU_shader_bind(shader);
  ha_bind_storage_buffers(buffers);
  ha_dispatch(shader, element_count);
  /* The next pass reads what this one wrote. */
  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE);
}

/* -------------------------------------------------------------------- */
/** \name Kernels
 * \{ */

const char *KERNEL_CELLS =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  int particle_count = int(params[0]);\n"
    "  int table_size = int(params[1]);\n"
    "  if (index >= particle_count) { return; }\n"
    "  float inv_cell_size = 1.0 / params[2];\n"
    "  int base = index * 3;\n"
    "  int cx = int(floor(position_in[base + 0] * inv_cell_size));\n"
    "  int cy = int(floor(position_in[base + 1] * inv_cell_size));\n"
    "  int cz = int(floor(position_in[base + 2] * inv_cell_size));\n"
    "  cells_out[base + 0] = cx;\n"
    "  cells_out[base + 1] = cy;\n"
    "  cells_out[base + 2] = cz;\n"
    "  int bucket = int(ha_hash_cell(cx, cy, cz) & uint(table_size - 1));\n"
    "  atomicAdd(bucket_counts[bucket], 1);\n"
    "}\n";

const char *KERNEL_SCAN_BLOCK_SUMS =
    "void main()\n"
    "{\n"
    "  int block = int(gl_GlobalInvocationID.x);\n"
    "  int element_count = int(scan_params[0]);\n"
    "  int block_count = int(scan_params[1]);\n"
    "  if (block >= block_count) { return; }\n"
    "  int start = block * 1024;\n"
    "  int end = min(start + 1024, element_count);\n"
    "  int acc = 0;\n"
    "  for (int i = start; i < end; i++) { acc += counts[i]; }\n"
    "  block_sums[block] = acc;\n"
    "}\n";

const char *KERNEL_SCAN_BLOCK_OFFSETS =
    "void main()\n"
    "{\n"
    "  if (int(gl_GlobalInvocationID.x) != 0) { return; }\n"
    "  int block_count = int(scan_params[1]);\n"
    "  int acc = 0;\n"
    "  for (int b = 0; b < block_count; b++) {\n"
    "    int value = block_sums[b];\n"
    "    block_offsets[b] = acc;\n"
    "    acc += value;\n"
    "  }\n"
    "  totals[0] = acc;\n"
    "}\n";

const char *KERNEL_SCAN_WITHIN_BLOCK =
    "void main()\n"
    "{\n"
    "  int block = int(gl_GlobalInvocationID.x);\n"
    "  int element_count = int(scan_params[0]);\n"
    "  int block_count = int(scan_params[1]);\n"
    "  if (block >= block_count) { return; }\n"
    "  int start = block * 1024;\n"
    "  int end = min(start + 1024, element_count);\n"
    "  int acc = block_offsets[block];\n"
    "  for (int i = start; i < end; i++) {\n"
    "    int count = counts[i];\n"
    "    offsets[i] = acc;\n"
    "    acc += count;\n"
    "  }\n"
    "  if (block == block_count - 1) { offsets[element_count] = acc; }\n"
    "}\n";

const char *KERNEL_BUCKET_CURSOR =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= int(params[1])) { return; }\n"
    "  cursor[index] = offsets[index];\n"
    "}\n";

const char *KERNEL_SCATTER =
    "void main()\n"
    "{\n"
    "  int index = int(gl_GlobalInvocationID.x);\n"
    "  if (index >= int(params[0])) { return; }\n"
    "  int table_size = int(params[1]);\n"
    "  int base = index * 3;\n"
    "  int bucket = int(ha_hash_cell(cells[base + 0], cells[base + 1], cells[base + 2]) &\n"
    "                  uint(table_size - 1));\n"
    "  int slot = atomicAdd(cursor[bucket], 1);\n"
    "  sorted_out[slot] = index;\n"
    "}\n";

/**
 * Sorts every bucket by particle index. The scatter above uses atomics, so the order inside a
 * bucket would depend on the scheduling; the CPU counting sort always appends in ascending order
 * and the density sums then add up in exactly that order. Buckets hold ~2 particles (load factor
 * 0.5), so an insertion sort is the right tool.
 */
const char *KERNEL_BUCKET_SORT =
    "void main()\n"
    "{\n"
    "  int bucket = int(gl_GlobalInvocationID.x);\n"
    "  if (bucket >= int(params[1])) { return; }\n"
    "  int begin = offsets[bucket];\n"
    "  int len = counts[bucket];\n"
    "  for (int i = 1; i < len; i++) {\n"
    "    int key = sorted_io[begin + i];\n"
    "    int j = i - 1;\n"
    "    while (j >= 0 && sorted_io[begin + j] > key) {\n"
    "      sorted_io[begin + j + 1] = sorted_io[begin + j];\n"
    "      j -= 1;\n"
    "    }\n"
    "    sorted_io[begin + j + 1] = key;\n"
    "  }\n"
    "}\n";

/**
 * The 27 cell neighbourhood walk of `FluidParticleGrid::foreach_neighbor_in_cell`, emitted twice:
 * once to count the neighbours of every particle, once to write the indices into the CSR list.
 * \a emit is the statement that consumes one neighbour.
 */
std::string neighbor_kernel_source(const char *emit, const bool counting){
  std::string source;
  source += "void main()\n{\n";
  source += "  int i = int(gl_GlobalInvocationID.x);\n";
  source += "  int particle_count = int(params[0]);\n";
  source += "  int table_size = int(params[1]);\n";
  source += "  float r_sq_max = params[3];\n";
  source += "  if (i >= particle_count) { return; }\n";
  source += "  int mask = table_size - 1;\n";
  source += "  int base = i * 3;\n";
  source += "  int cx = cells[base + 0];\n";
  source += "  int cy = cells[base + 1];\n";
  source += "  int cz = cells[base + 2];\n";
  source += "  vec3 p = vec3(position_in[base + 0], position_in[base + 1], position_in[base + 2]);\n";
  source += "  int count = 0;\n";
  if (!counting) {
    source += "  int write_i = offsets_out[i];\n";
  }
  source += "  for (int dz = -1; dz <= 1; dz++) {\n";
  source += "    for (int dy = -1; dy <= 1; dy++) {\n";
  source += "      for (int dx = -1; dx <= 1; dx++) {\n";
  source += "        int nx = cx + dx;\n";
  source += "        int ny = cy + dy;\n";
  source += "        int nz = cz + dz;\n";
  source += "        int bucket = int(ha_hash_cell(nx, ny, nz) & uint(mask));\n";
  source += "        int begin = bucket_offsets[bucket];\n";
  source += "        int end = begin + bucket_counts[bucket];\n";
  source += "        for (int k = begin; k < end; k++) {\n";
  source += "          int j = sorted_in[k];\n";
  source += "          int jb = j * 3;\n";
  source += "          if (cells[jb + 0] != nx || cells[jb + 1] != ny || cells[jb + 2] != nz) {\n";
  source += "            continue;\n";
  source += "          }\n";
  source += "          vec3 r = p - vec3(position_in[jb + 0], position_in[jb + 1], "
            "position_in[jb + 2]);\n";
  source += "          if (dot(r, r) >= r_sq_max) { continue; }\n";
  source += "          ";
  source += emit;
  source += "\n";
  source += "        }\n";
  source += "      }\n";
  source += "    }\n";
  source += "  }\n";
  if (counting) {
    source += "  neighbor_counts[i] = count;\n";
  }
  source += "}\n";
  return source;
}

/* -------------------------------------------------------------------- */
/** \name Resource layouts
 * \{ */

std::array<HaShaderResource, 4> CELLS_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Write, "int", "cells_out[]"},
    {2, HaShaderQualifier::ReadWrite, "int", "bucket_counts[]"},
    {3, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 3> SCAN_BLOCK_SUMS_RESOURCES = {{
    {0, HaShaderQualifier::Read, "int", "counts[]"},
    {1, HaShaderQualifier::Read, "float", "scan_params[]"},
    {2, HaShaderQualifier::Write, "int", "block_sums[]"},
}};

std::array<HaShaderResource, 4> SCAN_BLOCK_OFFSETS_RESOURCES = {{
    {0, HaShaderQualifier::Read, "int", "block_sums[]"},
    {1, HaShaderQualifier::Write, "int", "block_offsets[]"},
    {2, HaShaderQualifier::Write, "int", "totals[]"},
    {3, HaShaderQualifier::Read, "float", "scan_params[]"},
}};

std::array<HaShaderResource, 4> SCAN_WITHIN_BLOCK_RESOURCES = {{
    {0, HaShaderQualifier::Read, "int", "counts[]"},
    {1, HaShaderQualifier::Read, "int", "block_offsets[]"},
    {2, HaShaderQualifier::Write, "int", "offsets[]"},
    {3, HaShaderQualifier::Read, "float", "scan_params[]"},
}};

std::array<HaShaderResource, 3> BUCKET_CURSOR_RESOURCES = {{
    {0, HaShaderQualifier::Read, "int", "offsets[]"},
    {1, HaShaderQualifier::Write, "int", "cursor[]"},
    {2, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 4> SCATTER_RESOURCES = {{
    {0, HaShaderQualifier::Read, "int", "cells[]"},
    {1, HaShaderQualifier::ReadWrite, "int", "cursor[]"},
    {2, HaShaderQualifier::Write, "int", "sorted_out[]"},
    {3, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 4> BUCKET_SORT_RESOURCES = {{
    {0, HaShaderQualifier::Read, "int", "offsets[]"},
    {1, HaShaderQualifier::Read, "int", "counts[]"},
    {2, HaShaderQualifier::ReadWrite, "int", "sorted_io[]"},
    {3, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 7> NEIGHBOR_COUNT_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "int", "cells[]"},
    {2, HaShaderQualifier::Read, "int", "bucket_offsets[]"},
    {3, HaShaderQualifier::Read, "int", "bucket_counts[]"},
    {4, HaShaderQualifier::Read, "int", "sorted_in[]"},
    {5, HaShaderQualifier::Write, "int", "neighbor_counts[]"},
    {6, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 8> NEIGHBOR_FILL_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "int", "cells[]"},
    {2, HaShaderQualifier::Read, "int", "bucket_offsets[]"},
    {3, HaShaderQualifier::Read, "int", "bucket_counts[]"},
    {4, HaShaderQualifier::Read, "int", "sorted_in[]"},
    {5, HaShaderQualifier::Read, "int", "offsets_out[]"},
    {6, HaShaderQualifier::Write, "int", "neighbor_indices[]"},
    {7, HaShaderQualifier::Read, "float", "params[]"},
}};

template<size_t N> Span<const HaShaderResource> as_span(const std::array<HaShaderResource, N> &r)
{
  return Span<const HaShaderResource>(r.data(), int64_t(r.size()));
}

template<size_t N> Span<gpu::StorageBuf *> buffer_span(std::array<gpu::StorageBuf *, N> &r)
{
  return Span<gpu::StorageBuf *>(r.data(), int64_t(r.size()));
}

/**
 * Two level exclusive prefix sum: \a counts -> \a r_offsets. The total is only read back when
 * \a r_total is not null (`GPU_storagebuf_read` stalls the pipeline, so this is avoided where the
 * value is not needed).
 */
bool scan_counts(HaXpbdGrid &grid,
                 HaBufferInt &counts,
                 HaBufferInt &r_offsets,
                 int *r_total,
                 const int element_count,
                 int &r_pass_count)
{
  if (element_count <= 0) {
    if (r_total != nullptr) {
      *r_total = 0;
    }
    return true;
  }
  const int block_count = (element_count + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
  if (!grid.scan_block_sums.ensure(block_count) || !grid.scan_block_offsets.ensure(block_count) ||
      !grid.scan_total.ensure(1) || !r_offsets.ensure(element_count + 1))
  {
    return false;
  }
  const std::array<float, 4> scan_params{float(element_count), float(block_count), 0.0f, 0.0f};
  grid.scan_params.upload(scan_params);

  gpu::Shader *stage_a = ha_shader_get_source("xpbd|scan_a",
                                              std::string(GLSL_PRELUDE) + KERNEL_SCAN_BLOCK_SUMS,
                                              as_span(SCAN_BLOCK_SUMS_RESOURCES),
                                              LOCAL_GROUP_SIZE);
  gpu::Shader *stage_b = ha_shader_get_source("xpbd|scan_b",
                                              std::string(GLSL_PRELUDE) + KERNEL_SCAN_BLOCK_OFFSETS,
                                              as_span(SCAN_BLOCK_OFFSETS_RESOURCES),
                                              LOCAL_GROUP_SIZE);
  gpu::Shader *stage_c = ha_shader_get_source("xpbd|scan_c",
                                              std::string(GLSL_PRELUDE) + KERNEL_SCAN_WITHIN_BLOCK,
                                              as_span(SCAN_WITHIN_BLOCK_RESOURCES),
                                              LOCAL_GROUP_SIZE);
  if (stage_a == nullptr || stage_b == nullptr || stage_c == nullptr) {
    return false;
  }
  {
    std::array<gpu::StorageBuf *, 3> buffers{
        counts.buf(), grid.scan_params.buf(), grid.scan_block_sums.buf()};
    dispatch_pass(stage_a, buffer_span(buffers), block_count);
  }
  {
    std::array<gpu::StorageBuf *, 4> buffers{grid.scan_block_sums.buf(),
                                             grid.scan_block_offsets.buf(),
                                             grid.scan_total.buf(),
                                             grid.scan_params.buf()};
    dispatch_pass(stage_b, buffer_span(buffers), 1);
  }
  {
    std::array<gpu::StorageBuf *, 4> buffers{counts.buf(),
                                             grid.scan_block_offsets.buf(),
                                             r_offsets.buf(),
                                             grid.scan_params.buf()};
    dispatch_pass(stage_c, buffer_span(buffers), block_count);
  }
  r_pass_count += 3;
  if (r_total == nullptr) {
    return true;
  }
  /* One int, but `GPU_storagebuf_read` reads the whole buffer, so it has to be the small one. */
  return grid.scan_total.read_one(*r_total);
}

/**
 * Sizes the grid buffers and writes the parameters. Does not dispatch anything: the caller either
 * uploads positions and calls #ha_xpbd_grid_build_device, or has the positions on the device
 * already (`HaXpbdFluid`).
 */
bool ha_xpbd_grid_prepare(HaXpbdGrid &r_grid,
                          const int particle_count,
                          const float support_radius)
{
  if (particle_count <= 0 || support_radius <= 0.0f) {
    return false;
  }
  /* Exactly the table size of `FluidParticleGrid::build`, so the bucket layout is identical. */
  int table_size = 1024;
  while (table_size < particle_count * 2 && table_size < (1 << 22)) {
    table_size <<= 1;
  }

  r_grid.particle_count = particle_count;
  r_grid.table_size = table_size;
  r_grid.cell_size = support_radius;
  r_grid.block_count = (table_size + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
  r_grid.built = false;
  r_grid.neighbors_built = false;

  if (!r_grid.positions.ensure(int64_t(particle_count) * 3) || !r_grid.params.ensure(8) ||
      !r_grid.scan_params.ensure(4) || !r_grid.cells.ensure(int64_t(particle_count) * 3) ||
      !r_grid.bucket_counts.ensure(table_size) || !r_grid.bucket_offsets.ensure(table_size + 1) ||
      !r_grid.bucket_cursor.ensure(table_size) || !r_grid.sorted.ensure(particle_count))
  {
    ha_xpbd_debug("grid allocation failed");
    return false;
  }

  const std::array<float, 8> params{float(particle_count),
                                    float(table_size),
                                    support_radius,
                                    support_radius * support_radius,
                                    0.0f,
                                    0.0f,
                                    0.0f,
                                    0.0f};
  r_grid.params.upload(params);
  return true;
}

/** `xpbd::FluidKernel::poly6` at a single distance, for the artificial pressure reference. */
float fluid_poly6_at(const float scale, const float h, const float r)
{
  const float h_sq = h * h;
  const float r_sq = r * r;
  if (r_sq >= h_sq) {
    return 0.0f;
  }
  const float diff = h_sq - r_sq;
  return scale * diff * diff * diff;
}

}  // namespace

/* -------------------------------------------------------------------- */
/** \name Public API
 * \{ */

void HaXpbdGrid::free()
{
  positions.free();
  params.free();
  scan_params.free();
  cells.free();
  bucket_counts.free();
  bucket_offsets.free();
  bucket_cursor.free();
  sorted.free();
  scan_block_sums.free();
  scan_block_offsets.free();
  scan_total.free();
  neighbor_counts.free();
  neighbor_offsets.free();
  neighbor_indices.free();
  built = false;
  neighbors_built = false;
  neighbor_total = 0;
  neighbor_capacity = 0;
}

bool ha_xpbd_grid_build(const Span<float3> positions,
                        const float support_radius,
                        HaXpbdGrid &r_grid)
{
  const int particle_count = int(positions.size());
  if (particle_count == 0 || support_radius <= 0.0f) {
    return false;
  }
  if (!ha_gpu_available()) {
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    ha_xpbd_debug("no GPU context");
    return false;
  }
  if (!ha_xpbd_grid_prepare(r_grid, particle_count, support_radius)) {
    return false;
  }
  const int64_t position_floats = int64_t(particle_count) * 3;
  Array<float> host_positions(position_floats);
  for (const int i : positions.index_range()) {
    host_positions[i * 3 + 0] = positions[i].x;
    host_positions[i * 3 + 1] = positions[i].y;
    host_positions[i * 3 + 2] = positions[i].z;
  }
  r_grid.positions.upload(host_positions);
  return ha_xpbd_grid_build_device(r_grid);
}

bool ha_xpbd_grid_build_device(HaXpbdGrid &r_grid)
{
  const int particle_count = r_grid.particle_count;
  const int table_size = r_grid.table_size;
  if (particle_count <= 0 || table_size <= 0) {
    return false;
  }
  if (!ha_gpu_available()) {
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    return false;
  }

  const double start_ms = time_now_ms();

  gpu::Shader *cells_shader = ha_shader_get_source("xpbd|cells",
                                                   std::string(GLSL_PRELUDE) + KERNEL_CELLS,
                                                   as_span(CELLS_RESOURCES),
                                                   LOCAL_GROUP_SIZE);  gpu::Shader *cursor_shader = ha_shader_get_source("xpbd|bucket_cursor",
                                                    std::string(GLSL_PRELUDE) + KERNEL_BUCKET_CURSOR,
                                                    as_span(BUCKET_CURSOR_RESOURCES),
                                                    LOCAL_GROUP_SIZE);
  gpu::Shader *scatter_shader = ha_shader_get_source("xpbd|scatter",
                                                     std::string(GLSL_PRELUDE) + KERNEL_SCATTER,
                                                     as_span(SCATTER_RESOURCES),
                                                     LOCAL_GROUP_SIZE);
  gpu::Shader *sort_shader = ha_shader_get_source("xpbd|bucket_sort",
                                                  std::string(GLSL_PRELUDE) + KERNEL_BUCKET_SORT,
                                                  as_span(BUCKET_SORT_RESOURCES),
                                                  LOCAL_GROUP_SIZE);
  if (cells_shader == nullptr || cursor_shader == nullptr || scatter_shader == nullptr ||
      sort_shader == nullptr)
  {
    return false;
  }

  int pass_count = 0;
  r_grid.bucket_counts.clear_to_zero();
  {
    std::array<gpu::StorageBuf *, 4> buffers{
        r_grid.positions.buf(), r_grid.cells.buf(), r_grid.bucket_counts.buf(), r_grid.params.buf()};
    dispatch_pass(cells_shader, buffer_span(buffers), particle_count);
    pass_count++;
  }
  if (!scan_counts(r_grid, r_grid.bucket_counts, r_grid.bucket_offsets, nullptr, table_size, pass_count))
  {
    return false;
  }
  {
    std::array<gpu::StorageBuf *, 3> buffers{
        r_grid.bucket_offsets.buf(), r_grid.bucket_cursor.buf(), r_grid.params.buf()};
    dispatch_pass(cursor_shader, buffer_span(buffers), table_size);
    pass_count++;
  }
  {
    std::array<gpu::StorageBuf *, 4> buffers{
        r_grid.cells.buf(), r_grid.bucket_cursor.buf(), r_grid.sorted.buf(), r_grid.params.buf()};
    dispatch_pass(scatter_shader, buffer_span(buffers), particle_count);
    pass_count++;
  }
  {
    std::array<gpu::StorageBuf *, 4> buffers{r_grid.bucket_offsets.buf(),
                                             r_grid.bucket_counts.buf(),
                                             r_grid.sorted.buf(),
                                             r_grid.params.buf()};
    dispatch_pass(sort_shader, buffer_span(buffers), table_size);
    pass_count++;
  }

  r_grid.built = true;
  r_grid.stats.grid_ms = time_now_ms() - start_ms;
  r_grid.stats.grid_passes = pass_count;
  if (ha_xpbd_debug_enabled()) {
    char buffer[256];
    snprintf(buffer,
             sizeof(buffer),
             "grid: %d particles, table %d, %d passes, %.3f ms",
             particle_count,
             table_size,
             pass_count,
             r_grid.stats.grid_ms);
    ha_xpbd_debug(buffer);
  }
  return true;
}

bool ha_xpbd_neighbors_build(HaXpbdGrid &r_grid)
{
  const int particle_count = r_grid.particle_count;
  if (!r_grid.built || particle_count == 0) {
    return false;
  }
  if (!ha_gpu_available()) {
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    return false;
  }
  const double start_ms = time_now_ms();

  if (!r_grid.neighbor_counts.ensure(particle_count)) {
    return false;
  }

  gpu::Shader *count_shader =
      ha_shader_get_source("xpbd|neighbor_count",
                           std::string(GLSL_PRELUDE) + neighbor_kernel_source("count += 1;", true),
                           as_span(NEIGHBOR_COUNT_RESOURCES),
                           LOCAL_GROUP_SIZE);
  gpu::Shader *fill_shader = ha_shader_get_source(
      "xpbd|neighbor_fill",
      std::string(GLSL_PRELUDE) +
          neighbor_kernel_source(
              "if (write_i < int(params[6])) { neighbor_indices[write_i] = j; }\n"
              "          write_i += 1;",
              false),
      as_span(NEIGHBOR_FILL_RESOURCES),
      LOCAL_GROUP_SIZE);
  if (count_shader == nullptr || fill_shader == nullptr) {
    return false;
  }

  int pass_count = 0;
  {
    std::array<gpu::StorageBuf *, 7> buffers{r_grid.positions.buf(),
                                             r_grid.cells.buf(),
                                             r_grid.bucket_offsets.buf(),
                                             r_grid.bucket_counts.buf(),
                                             r_grid.sorted.buf(),
                                             r_grid.neighbor_counts.buf(),
                                             r_grid.params.buf()};
    dispatch_pass(count_shader, buffer_span(buffers), particle_count);
    pass_count++;
  }
  /* No read back of the total here: it would stall the pipeline once per substep. The capacity is
   * a heuristic, the fill clamps to it, and the exact total is read once per frame by
   * #ha_xpbd_neighbors_total (when the frame result is downloaded anyway). */
  int total = 0;
  if (!scan_counts(
          r_grid, r_grid.neighbor_counts, r_grid.neighbor_offsets, nullptr, particle_count, pass_count))
  {
    return false;
  }

  const int capacity = std::max(r_grid.neighbor_capacity, std::max(64, particle_count * 48));
  if (capacity != r_grid.neighbor_capacity) {
    if (!r_grid.neighbor_indices.ensure(capacity)) {
      return false;
    }
    r_grid.neighbor_capacity = capacity;
    const std::array<float, 8> params{float(particle_count),
                                      float(r_grid.table_size),
                                      r_grid.cell_size,
                                      r_grid.cell_size * r_grid.cell_size,
                                      0.0f,
                                      0.0f,
                                      float(capacity),
                                      0.0f};
    r_grid.params.upload(params);
  }
  {
    std::array<gpu::StorageBuf *, 8> buffers{r_grid.positions.buf(),
                                             r_grid.cells.buf(),
                                             r_grid.bucket_offsets.buf(),
                                             r_grid.bucket_counts.buf(),
                                             r_grid.sorted.buf(),
                                             r_grid.neighbor_offsets.buf(),
                                             r_grid.neighbor_indices.buf(),
                                             r_grid.params.buf()};
    dispatch_pass(fill_shader, buffer_span(buffers), particle_count);
    pass_count++;
  }

  r_grid.neighbor_total = total;
  r_grid.neighbors_built = true;
  r_grid.stats.neighbors_ms = time_now_ms() - start_ms;
  r_grid.stats.neighbor_passes = pass_count;
  if (ha_xpbd_debug_enabled()) {
    char buffer[256];
    snprintf(buffer,
             sizeof(buffer),
             "neighbors: %d particles, capacity %d, %d passes, %.3f ms",
             particle_count,
             r_grid.neighbor_capacity,
             pass_count,
             r_grid.stats.neighbors_ms);
    ha_xpbd_debug(buffer);
  }
  return true;
}

bool ha_xpbd_neighbors_total(const HaXpbdGrid &grid, int &r_total)
{
  if (!grid.neighbors_built || grid.scan_total.size() < 1) {
    r_total = 0;
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    r_total = 0;
    return false;
  }
  int total = 0;
  if (!grid.scan_total.read_one(total)) {
    r_total = 0;
    return false;
  }
  r_total = total;
  return true;
}

void ha_xpbd_neighbors_download(const HaXpbdGrid &grid,
                                const MutableSpan<int> r_offsets,
                                const MutableSpan<int> r_indices)
{
  if (grid.neighbor_offsets.size() > 0 && !r_offsets.is_empty()) {
    const int count = std::min<int64_t>(r_offsets.size(), grid.neighbor_offsets.size());
    grid.neighbor_offsets.download(r_offsets.take_front(count));
  }
  /* `offsets[particle_count]` is the total written by the scan. */
  const int total = r_offsets.is_empty() ? grid.neighbor_total : r_offsets.last();
  if (!r_indices.is_empty() && grid.neighbor_indices.size() > 0) {
    grid.neighbor_indices.download(
        r_indices.take_front(std::min<int64_t>(r_indices.size(), total)));
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Fluid kernels
 *
 * A direct port of `GEO_xpbd_fluid.hh`. Every pass is Jacobi style: it reads the state of the
 * previous pass and writes a fresh one, so the result cannot depend on the scheduling. The
 * summations per particle run over the CSR neighbour list in the same order as the CPU version,
 * which is what makes the two backends comparable.
 *
 * The boundary particles of `FluidBoundary` (Akinci et al.) are part of the density estimate and of
 * the positional correction, but they never move; they live in their own grid and their own
 * neighbour list (indexed by *fluid* particle).
 * \{ */

/**
 * Indices inside the per frame constant buffer of #HaXpbdFluid. The first four entries mirror the
 * grid parameters (`HaXpbdGrid::params`) so that the fluid kernels can filter neighbours by the
 * support radius; the rest belongs to the fluid.
 */
enum FluidParamIndex {
  FLUID_PARAM_COUNT = 0,
  FLUID_PARAM_TABLE_SIZE = 1,
  FLUID_PARAM_CELL_SIZE = 2,
  FLUID_PARAM_R_SQ_MAX = 3,
  FLUID_PARAM_H = 4,
  FLUID_PARAM_H_SQ = 5,
  FLUID_PARAM_POLY6 = 6,
  FLUID_PARAM_SPIKY = 7,
  FLUID_PARAM_COMPLIANCE = 8,
  FLUID_PARAM_ARTIFICIAL = 9,
  FLUID_PARAM_INV_ARTIFICIAL_REF = 10,
  FLUID_PARAM_MAX_CORRECTION = 11,
  FLUID_PARAM_DT = 12,
  FLUID_PARAM_INV_DT = 13,
  FLUID_PARAM_VORTICITY_MAX_CHANGE = 14,
  FLUID_PARAM_COHESION = 15,
  FLUID_PARAM_BOUNDARY_TABLE_SIZE = 16,
  FLUID_PARAM_BOUNDARY_CAPACITY = 17,
  FLUID_PARAM_CONTACT_COUNT = 18,
  /** Hash table size of the coupled samples, for the fluid -> coupled query. */
  FLUID_PARAM_COUPLED_TABLE_SIZE = 19,
  FLUID_PARAM_COUPLED_CAPACITY = 20,
  /** Number of coupled samples, i.e. the element count of the reaction pass. */
  FLUID_PARAM_COUPLED_COUNT = 21,
  /** Capacity of the coupled contact list (particle index + push vector). */
  FLUID_PARAM_COUPLED_CONTACT_CAPACITY = 22,
  FLUID_PARAM_COUNT_TOTAL = 23,
};

/** Collider contacts: `face position`, `normal`, `collider motion` and `collider velocity`. */
constexpr int HA_CONTACT_VEC_STRIDE = 12;
/** Collider contacts: margin, compliance, static friction, dynamic friction, error scale, radius. */
constexpr int HA_CONTACT_SCALAR_STRIDE = 6;

const char *GLSL_FLUID_HELPERS =
    "float ha_poly6(float r_sq)\n"
    "{\n"
    "  if (r_sq >= params[5]) { return 0.0; }\n"
    "  float diff = params[5] - r_sq;\n"
    "  return params[6] * diff * diff * diff;\n"
    "}\n"
    "float ha_spiky_length(float r)\n"
    "{\n"
    "  if (r >= params[4]) { return 0.0; }\n"
    "  float diff = params[4] - r;\n"
    "  return params[7] * diff * diff;\n"
    "}\n"
    "vec3 ha_spiky_gradient(vec3 r_vec, float r)\n"
    "{\n"
    "  if (r <= 1e-12) { return vec3(0.0); }\n"
    "  return (-ha_spiky_length(r) / r) * r_vec;\n"
    "}\n"
    "float ha_cohesion(float r)\n"
    "{\n"
    "  if (r <= 0.0 || r >= params[4]) { return 0.0; }\n"
    "  float diff = params[4] - r;\n"
    "  return params[15] * diff * diff * diff * r * r * r;\n"
    "}\n";

/** The prologue shared by every per particle kernel: the index guard and the particle position. */
const char *FLUID_PROLOGUE =
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(params[0])) { return; }\n"
    "  int base = i * 3;\n"
    "  vec3 p_i = vec3(position_in[base + 0], position_in[base + 1], position_in[base + 2]);\n";

/**
 * Emits the neighbour walks of `FluidNeighborList::foreach_neighbor` (fluid particles) and of
 * `FluidNeighborList::foreach_neighbor_at` (the boundary particles around the fluid particle \a i).
 * `r_vec` is `position_i - position_j` and the pairs beyond the current support radius are skipped
 * again, because the lists were built at the beginning of the substep.
 */
std::string fluid_neighbor_loops(const char *fluid_body,
                                 const char *boundary_body,
                                 const bool with_boundary,
                                 const char *coupled_body = nullptr)
{
  std::string source;
  source += "  for (int k = neighbor_offsets[i]; k < neighbor_offsets[i + 1]; k++) {\n";
  source += "    int j = neighbor_indices[k];\n";
  source += "    int jb = j * 3;\n";
  source += "    vec3 r_vec = p_i - vec3(position_in[jb + 0], position_in[jb + 1], position_in[jb + "
            "2]);\n";
  source += "    float r_sq = dot(r_vec, r_vec);\n";
  source += "    if (r_sq >= params[3]) { continue; }\n";
  source += fluid_body;
  source += "  }\n";
  if (with_boundary) {
    /* The CPU only uses the boundary particles where `use_boundary` is set; they never move, so
     * only the fluid side of the density constraint gets a gradient from them. */
    source += "  if (props[i * HA_PROP + 9] > 0.5) {\n";
    source += "    for (int kb = boundary_offsets[i]; kb < boundary_offsets[i + 1]; kb++) {\n";
    source += "      int boundary_i = boundary_indices[kb];\n";
    source += "      int bp = boundary_i * 4;\n";
    source += "      vec3 boundary_pos = vec3(boundary_in[bp + 0], boundary_in[bp + 1],\n";
    source += "                              boundary_in[bp + 2]);\n";
    source += "      vec3 r_vec_b = p_i - boundary_pos;\n";
    source += "      float r_sq_b = dot(r_vec_b, r_vec_b);\n";
    source += "      if (r_sq_b >= params[3]) { continue; }\n";
    source += "      float mass_b = boundary_in[bp + 3];\n";
    source += boundary_body;
    source += "    }\n";
    source += "  }\n";
  }
  if (coupled_body != nullptr) {
    /* The coupled samples of the other simulated geometries are *always* part of the density
     * estimate (they are what makes the coupling hold water), so unlike the collider boundary they
     * are not gated by `use_boundary`. They can move, so the reaction pass gives them their own
     * inverse mass. */
    source += "  for (int kc = coupled_offsets[i]; kc < coupled_offsets[i + 1]; kc++) {\n";
    source += "    int c = coupled_indices[kc];\n";
    source += "    int cb = c * 5;\n";
    source += "    vec3 coupled_pos = vec3(coupled_in[cb + 0], coupled_in[cb + 1],\n";
    source += "                            coupled_in[cb + 2]);\n";
    source += "    vec3 r_vec_c = p_i - coupled_pos;\n";
    source += "    float r_sq_c = dot(r_vec_c, r_vec_c);\n";
    source += "    if (r_sq_c >= params[3]) { continue; }\n";
    source += "    float mass_c = coupled_in[cb + 3];\n";
    source += coupled_body;
    source += "  }\n";
  }
  return source;
}

std::string fluid_kernel(const char *prologue,
                         const char *fluid_body,
                         const char *boundary_body,
                         const char *epilogue,
                         const char *coupled_body = nullptr,
                         const char *coupled_epilogue = nullptr)
{
  const bool with_boundary = boundary_body != nullptr;
  std::string source = std::string(GLSL_PRELUDE) + GLSL_FLUID_HELPERS + "void main()\n{\n";
  source += FLUID_PROLOGUE;
  source += prologue;
  source += fluid_neighbor_loops(
      fluid_body, with_boundary ? boundary_body : "", with_boundary, coupled_body);
  source += epilogue;
  if (coupled_epilogue != nullptr) {
    source += coupled_epilogue;
  }
  source += "}\n";
  return source;
}

const char *KERNEL_PREDICT =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(params[0])) { return; }\n"
    "  int base = i * 3;\n"
    "  vec3 position = vec3(position_io[base + 0], position_io[base + 1], position_io[base + 2]);\n"
    "  vec3 velocity = vec3(velocity_io[base + 0], velocity_io[base + 1], velocity_io[base + 2]);\n"
    "  prev_position[base + 0] = position.x;\n"
    "  prev_position[base + 1] = position.y;\n"
    "  prev_position[base + 2] = position.z;\n"
    "  vec3 force = vec3(force_in[base + 0], force_in[base + 1], force_in[base + 2]);\n"
    "  velocity += force * props[i * HA_PROP + 1] * params[12];\n"
    "  position += velocity * params[12];\n"
    "  velocity_io[base + 0] = velocity.x;\n"
    "  velocity_io[base + 1] = velocity.y;\n"
    "  velocity_io[base + 2] = velocity.z;\n"
    "  position_io[base + 0] = position.x;\n"
    "  position_io[base + 1] = position.y;\n"
    "  position_io[base + 2] = position.z;\n"
    "}\n";

const char *KERNEL_APPLY_CORRECTION =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(params[0])) { return; }\n"
    "  if (props[i * HA_PROP + 1] <= 0.0) { return; }\n"
    "  int base = i * 3;\n"
    "  int sb = i * HA_STATE + 4;\n"
    "  position_io[base + 0] += particle_state[sb + 0];\n"
    "  position_io[base + 1] += particle_state[sb + 1];\n"
    "  position_io[base + 2] += particle_state[sb + 2];\n"
    "}\n";

const char *KERNEL_UPDATE_VELOCITY =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(params[0])) { return; }\n"
    "  int base = i * 3;\n"
    "  float inv_dt = params[13];\n"
    "  vec3 velocity = (vec3(position_in[base + 0], position_in[base + 1],\n"
    "                        position_in[base + 2]) -\n"
    "                   vec3(prev_position[base + 0], prev_position[base + 1],\n"
    "                        prev_position[base + 2])) * inv_dt;\n"
    "  vec3 push = vec3(coupled_push[base + 0], coupled_push[base + 1], coupled_push[base + 2]);\n"
    "  float push_length = length(push);\n"
    "  if (push_length > 1e-12) {\n"
    "    vec3 normal = push / push_length;\n"
    "    velocity -= dot(velocity, normal) * normal;\n"
    "  }\n"
    "  velocity_out[base + 0] = velocity.x;\n"
    "  velocity_out[base + 1] = velocity.y;\n"
    "  velocity_out[base + 2] = velocity.z;\n"
    "}\n";

const char *KERNEL_DAMPING =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(params[0])) { return; }\n"
    "  int base = i * 3;\n"
    "  float factor = 1.0 / (1.0 + props[i * HA_PROP + 5] * params[12]);\n"
    "  float solid_fraction = particle_state[i * HA_STATE + 3];\n"
    "  if (solid_fraction > 0.0) {\n"
    "    float friction_factor = 1.0 - props[i * HA_PROP + 8] * solid_fraction;\n"
    "    factor *= max(friction_factor, 0.0);\n"
    "  }\n"
    "  velocity_io[base + 0] *= factor;\n"
    "  velocity_io[base + 1] *= factor;\n"
    "  velocity_io[base + 2] *= factor;\n"
    "}\n";

/* --- density constraint ------------------------------------------------- */

const char *DENSITY_PROLOGUE =
    "  float inv_rest_density = 1.0 / props[i * HA_PROP + 2];\n"
    "  float density = 0.0;\n"
    "  float fluid_weight = 0.0;\n"
    "  float boundary_weight = 0.0;\n"
    "  float boundary_density = 0.0;\n"
    "  vec3 sum_gradient = vec3(0.0);\n"
    "  float sum_gradient_sq = 0.0;\n";

const char *DENSITY_LOOP =
    "    float weight = ha_poly6(r_sq);\n"
    "    density += props[j * HA_PROP + 0] * weight;\n"
    "    fluid_weight += weight;\n"
    "    if (j == i) { continue; }\n"
    "    if (r_sq < 1e-12) { continue; }\n"
    "    float r = sqrt(r_sq);\n"
    "    float gradient_length = ha_spiky_length(r);\n"
    "    float mass_j = props[j * HA_PROP + 0];\n"
    "    sum_gradient += mass_j * (-(gradient_length / r) * r_vec);\n"
    "    sum_gradient_sq += (mass_j * gradient_length) * (mass_j * gradient_length);\n";

const char *DENSITY_BOUNDARY_LOOP =
    "      float boundary_kernel = ha_poly6(r_sq_b);\n"
    "      boundary_density += mass_b * boundary_kernel;\n"
    "      boundary_weight += boundary_kernel;\n"
    "      if (r_sq_b < 1e-12) { continue; }\n"
    "      float r_b = sqrt(r_sq_b);\n"
    "      sum_gradient += mass_b * (-(ha_spiky_length(r_b) / r_b) * r_vec_b);\n";

const char *DENSITY_EPILOGUE =
    "  density += boundary_density;\n"
    "  float weight_sum = boundary_weight + fluid_weight;\n"
    "  particle_state[i * HA_STATE + 3] = weight_sum > 0.0 ? boundary_weight / weight_sum : 0.0;\n"
    "  float constraint = density * inv_rest_density - 1.0;\n"
    "  float negative_limit = props[i * HA_PROP + 3];\n"
    "  float solved_constraint = max(constraint, -negative_limit);\n"
    "  particle_state[i * HA_STATE + 1] = solved_constraint * solved_constraint;\n"
    "  particle_state[i * HA_STATE + 2] = constraint;\n"
    "  if (props[i * HA_PROP + 1] <= 0.0) {\n"
    "    particle_state[i * HA_STATE + 0] = 0.0;\n"
    "    return;\n"
    "  }\n"
    "  float denominator = (dot(sum_gradient, sum_gradient) + sum_gradient_sq) * inv_rest_density "
    "*\n"
    "                          inv_rest_density + params[8] + 1e-12;\n"
    "  float max_lambda = negative_limit / denominator;\n"
    "  particle_state[i * HA_STATE + 0] = min(-solved_constraint / denominator, max_lambda);\n";

const char *DELTA_PROLOGUE =
    "  if (props[i * HA_PROP + 1] <= 0.0) {\n"
    "    particle_state[i * HA_STATE + 4] = 0.0;\n"
    "    particle_state[i * HA_STATE + 5] = 0.0;\n"
    "    particle_state[i * HA_STATE + 6] = 0.0;\n"
    "    return;\n"
    "  }\n"
    "  float lambda_i = particle_state[i * HA_STATE + 0];\n"
    "  vec3 correction = vec3(0.0);\n"
    "  vec3 coupled_correction = vec3(0.0);\n"
    "  float clamp_scale = 1.0;\n";

const char *DELTA_LOOP =
    "    if (j == i) { continue; }\n"
    "    if (r_sq < 1e-12) { continue; }\n"
    "    float r = sqrt(r_sq);\n"
    "    float pair_weight = lambda_i + particle_state[j * HA_STATE + 0];\n"
    "    if (params[9] > 0.0) {\n"
    "      float ratio = ha_poly6(r_sq) * params[10];\n"
    "      pair_weight -= params[9] * ratio * ratio * ratio * ratio;\n"
    "    }\n"
    "    correction += pair_weight * props[j * HA_PROP + 0] *\n"
    "                  (-(ha_spiky_length(r) / r) * r_vec);\n";

const char *DELTA_BOUNDARY_LOOP =
    "      if (r_sq_b < 1e-12) { continue; }\n"
    "      float r_b = sqrt(r_sq_b);\n"
    "      correction += lambda_i * mass_b * (-(ha_spiky_length(r_b) / r_b) * r_vec_b);\n";

/* --- coupled solids (two way coupling) ---------------------------------- */

/**
 * The coupled samples of the other simulated geometries are always part of the density estimate,
 * exactly like `xpbd::fluid_solve_density_iteration` accumulates `coupled`: they add to the density,
 * to the solid fraction (for the no-slip friction) and to the gradient, but not to the per
 * neighbour gradient sum of the denominator (they are not fluid particles).
 */
const char *DENSITY_COUPLED_LOOP =
    "    float weight = ha_poly6(r_sq_c);\n"
    "    boundary_density += mass_c * weight;\n"
    "    boundary_weight += weight;\n"
    "    if (r_sq_c >= 1e-12) {\n"
    "      float r_c = sqrt(r_sq_c);\n"
    "      sum_gradient += mass_c * (-(ha_spiky_length(r_c) / r_c) * r_vec_c);\n"
    "    }\n";

const char *DELTA_COUPLED_LOOP =
    "    if (r_sq_c >= 1e-12) {\n"
    "      float r_c = sqrt(r_sq_c);\n"
    "      vec3 coupled_term = lambda_i * mass_c * (-(ha_spiky_length(r_c) / r_c) * r_vec_c);\n"
    "      correction += coupled_term;\n"
    "      coupled_correction += coupled_term;\n"
    "    }\n";

/**
 * Reaction of the fluid pressure on the coupled samples, the device mirror of
 * `xpbd::fluid_apply_boundary_reaction`. One invocation per coupled sample: query the fluid grid
 * around it and accumulate
 *
 *     dm_j = -lambda_i * m_i * grad_j W_ij / rho0_i
 *
 * (the sign of `r_vec` is flipped because the gradient is taken with respect to the coupled
 * particle). The host downloads this buffer after every iteration and spreads it over the vertices
 * of the solid.
 */
const char *KERNEL_COUPLED_REACTION =
    "void main()\n"
    "{\n"
    "  int c = int(gl_GlobalInvocationID.x);\n"
    "  if (c >= int(params[21])) { return; }\n"
    "  int cb = c * 5;\n"
    "  vec3 coupled_pos = vec3(coupled_in[cb + 0], coupled_in[cb + 1], coupled_in[cb + 2]);\n"
    "  vec3 correction = vec3(0.0);\n"
    "  if (coupled_in[cb + 4] > 0.0) {\n"
    "    int table_size = int(params[19]);\n"
    "    int mask = table_size - 1;\n"
    "    float inv_cell_size = 1.0 / params[2];\n"
    "    int cx = int(floor(coupled_pos.x * inv_cell_size));\n"
    "    int cy = int(floor(coupled_pos.y * inv_cell_size));\n"
    "    int cz = int(floor(coupled_pos.z * inv_cell_size));\n"
    "    for (int dz = -1; dz <= 1; dz++) {\n"
    "      for (int dy = -1; dy <= 1; dy++) {\n"
    "        for (int dx = -1; dx <= 1; dx++) {\n"
    "          int nx = cx + dx;\n"
    "          int ny = cy + dy;\n"
    "          int nz = cz + dz;\n"
    "          int bucket = int(ha_hash_cell(nx, ny, nz) & uint(mask));\n"
    "          int begin = bucket_offsets[bucket];\n"
    "          int end = begin + bucket_counts[bucket];\n"
    "          for (int k = begin; k < end; k++) {\n"
    "            int i = sorted_in[k];\n"
    "            if (cells[i * 3 + 0] != nx || cells[i * 3 + 1] != ny ||\n"
    "                cells[i * 3 + 2] != nz) { continue; }\n"
    "            vec3 fluid_pos = vec3(position_in[i * 3 + 0], position_in[i * 3 + 1],\n"
    "                                  position_in[i * 3 + 2]);\n"
    "            vec3 r_vec = coupled_pos - fluid_pos;\n"
    "            float r_sq = dot(r_vec, r_vec);\n"
    "            if (r_sq >= params[3] || r_sq < 1e-12) { continue; }\n"
    "            float lambda_i = particle_state[i * HA_STATE + 0] *\n"
    "                             props[i * HA_PROP + 10];\n"
    "            if (lambda_i == 0.0) { continue; }\n"
    "            float r = sqrt(r_sq);\n"
    "            correction += lambda_i * props[i * HA_PROP + 0] *\n"
    "                          (-(ha_spiky_length(r) / r) * r_vec) /\n"
    "                          props[i * HA_PROP + 2];\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "    float max_correction = params[11];\n"
    "    float length = length(correction);\n"
    "    if (max_correction > 0.0 && length > max_correction) {\n"
    "      correction *= max_correction / length;\n"
    "    }\n"
    "    if (!ha_is_finite(correction)) { correction = vec3(0.0); }\n"
    "  }\n"
    "  coupled_reaction_out[c * 3 + 0] += correction.x;\n"
    "  coupled_reaction_out[c * 3 + 1] += correction.y;\n"
    "  coupled_reaction_out[c * 3 + 2] += correction.z;\n"
    "}\n";

const char *DELTA_EPILOGUE =
    "  float inv_rest_density = 1.0 / props[i * HA_PROP + 2];\n"
    "  correction *= inv_rest_density;\n"
    "  coupled_correction *= inv_rest_density;\n"
    "  float max_correction = params[11];\n"
    "  float correction_length = length(correction);\n"
    "  if (max_correction > 0.0 && correction_length > max_correction) {\n"
    "    clamp_scale = max_correction / correction_length;\n"
    "    correction *= clamp_scale;\n"
    "  }\n"
    "  if (!ha_is_finite(correction)) { correction = vec3(0.0); coupled_correction = vec3(0.0); }\n"
    "  particle_state[i * HA_STATE + 4] = correction.x;\n"
    "  particle_state[i * HA_STATE + 5] = correction.y;\n"
    "  particle_state[i * HA_STATE + 6] = correction.z;\n";

const char *DELTA_COUPLED_EPILOGUE =
    "  coupled_correction *= clamp_scale;\n"
    "  vec3 density_push = particle_state[i * HA_STATE + 3] > 0.0 ? correction : coupled_correction;\n"
    "  coupled_push[base + 0] += density_push.x;\n"
    "  coupled_push[base + 1] += density_push.y;\n"
    "  coupled_push[base + 2] += density_push.z;\n";

const char *VORTICITY_PROLOGUE =
    "  vec3 velocity_i = vec3(velocity_in[i * 3 + 0], velocity_in[i * 3 + 1],\n"
    "                         velocity_in[i * 3 + 2]);\n"
    "  vec3 omega = vec3(0.0);\n";

const char *VORTICITY_LOOP =
    "    if (j == i) { continue; }\n"
    "    if (r_sq < 1e-12) { continue; }\n"
    "    float r = sqrt(r_sq);\n"
    "    vec3 velocity_j = vec3(velocity_in[jb + 0], velocity_in[jb + 1], velocity_in[jb + 2]);\n"
    "    omega += props[j * HA_PROP + 0] *\n"
    "             cross(velocity_j - velocity_i, ha_spiky_gradient(r_vec, r));\n";

const char *VORTICITY_EPILOGUE =
    "  omega *= 1.0 / props[i * HA_PROP + 2];\n"
    "  particle_state[i * HA_STATE + 4] = omega.x;\n"
    "  particle_state[i * HA_STATE + 5] = omega.y;\n"
    "  particle_state[i * HA_STATE + 6] = omega.z;\n"
    "  particle_state[i * HA_STATE + 7] = length(omega);\n";

const char *VORTICITY_PROLOGUE_PASS2 =
    "  float magnitude_i = particle_state[i * HA_STATE + 7];\n"
    "  vec3 gradient = vec3(0.0);\n"
    "  vec3 surface_change = vec3(0.0);\n";

const char *VORTICITY_LOOP_PASS2 =
    "    if (j == i) { continue; }\n"
    "    if (r_sq < 1e-12) { continue; }\n"
    "    float r = sqrt(r_sq);\n"
    "    gradient += props[j * HA_PROP + 0] *\n"
    "                (particle_state[j * HA_STATE + 7] - magnitude_i) *\n"
    "                ha_spiky_gradient(r_vec, r);\n"
    "    float cohesion = ha_cohesion(r);\n"
    "    if (cohesion > 0.0) {\n"
    "      vec3 offset = vec3(position_in[jb + 0], position_in[jb + 1], position_in[jb + 2]) - "
    "p_i;\n"
    "      surface_change += props[j * HA_PROP + 0] * cohesion * offset * (1.0 / r);\n"
    "    }\n";

const char *VORTICITY_EPILOGUE_PASS2 =
    "  vec3 change = vec3(0.0);\n"
    "  float gradient_length = length(gradient);\n"
    "  if (gradient_length > 1e-6) {\n"
    "    vec3 direction = gradient / gradient_length;\n"
    "    vec3 omega = vec3(particle_state[i * HA_STATE + 4], particle_state[i * HA_STATE + 5],\n"
    "                      particle_state[i * HA_STATE + 6]);\n"
    "    change = (props[i * HA_PROP + 6] * params[4] * params[12]) * cross(direction, omega);\n"
    "    float change_length = length(change);\n"
    "    if (change_length > params[14]) {\n"
    "      change *= params[14] / change_length;\n"
    "    }\n"
    "    if (!ha_is_finite(change)) { change = vec3(0.0); }\n"
    "  }\n"
    "  velocity_io[i * 3 + 0] += change.x;\n"
    "  velocity_io[i * 3 + 1] += change.y;\n"
    "  velocity_io[i * 3 + 2] += change.z;\n"
    "  // Surface tension, applied after the vorticity like on the CPU.\n"
    "  vec3 surface_delta = surface_change * props[i * HA_PROP + 7] * params[12];\n"
    "  if (!ha_is_finite(surface_delta)) { surface_delta = vec3(0.0); }\n"
    "  velocity_io[i * 3 + 0] += surface_delta.x;\n"
    "  velocity_io[i * 3 + 1] += surface_delta.y;\n"
    "  velocity_io[i * 3 + 2] += surface_delta.z;\n";

const char *SURFACE_TENSION_PROLOGUE =
    "  vec3 change = vec3(0.0);\n";

const char *SURFACE_TENSION_LOOP =
    "    if (j == i) { continue; }\n"
    "    if (r_sq < 1e-12) { continue; }\n"
    "    float r = sqrt(r_sq);\n"
    "    float cohesion = ha_cohesion(r);\n"
    "    if (cohesion <= 0.0) { continue; }\n"
    "    vec3 offset = vec3(position_in[jb + 0], position_in[jb + 1], position_in[jb + 2]) - "
    "p_i;\n"
    "    change += props[j * HA_PROP + 0] * cohesion * offset * (1.0 / r);\n";

const char *SURFACE_TENSION_EPILOGUE =
    "  if (!ha_is_finite(change)) { change = vec3(0.0); }\n"
    "  change *= props[i * HA_PROP + 7] * params[12];\n"
    "  velocity_io[i * 3 + 0] += change.x;\n"
    "  velocity_io[i * 3 + 1] += change.y;\n"
    "  velocity_io[i * 3 + 2] += change.z;\n";

const char *XSPH_PROLOGUE =
    "  vec3 velocity_i = vec3(velocity_in[i * 3 + 0], velocity_in[i * 3 + 1],\n"
    "                         velocity_in[i * 3 + 2]);\n"
    "  vec3 sum = vec3(0.0);\n";

const char *XSPH_LOOP =
    "    if (j == i) { continue; }\n"
    "    vec3 velocity_j = vec3(velocity_in[jb + 0], velocity_in[jb + 1], velocity_in[jb + 2]);\n"
    "    sum += props[j * HA_PROP + 0] * (velocity_j - velocity_i) * ha_poly6(r_sq);\n";

const char *XSPH_EPILOGUE =
    "  vec3 result = velocity_i + (props[i * HA_PROP + 4] / props[i * HA_PROP + 2]) * sum;\n"
    "  // Damping and the no-slip factor are per particle, applied here instead of a separate pass.\n"
    "  float factor = 1.0 / (1.0 + props[i * HA_PROP + 5] * params[12]);\n"
    "  float solid_fraction = particle_state[i * HA_STATE + 3];\n"
    "  if (solid_fraction > 0.0) {\n"
    "    factor *= max(1.0 - props[i * HA_PROP + 8] * solid_fraction, 0.0);\n"
    "  }\n"
    "  result *= factor;\n"
    "  velocity_out[i * 3 + 0] = result.x;\n"
    "  velocity_out[i * 3 + 1] = result.y;\n"
    "  velocity_out[i * 3 + 2] = result.z;\n";

/* --- boundary neighbours ------------------------------------------------- */

/**
 * The `foreach_neighbor_at` variant of #HaXpbdGrid: the invocation is a *fluid* particle, so its
 * cell is computed from its own position instead of read from the target grid. Emitted twice
 * (counting and filling) like the fluid neighbour kernels.
 */
std::string neighbor_query_kernel(const bool counting,
                                  const char *positions_base,
                                  const int positions_stride,
                                  const char *counts_name,
                                  const char *offsets_name,
                                  const char *indices_name,
                                  const int table_size_param,
                                  const int capacity_param)
{
  std::string source = std::string(GLSL_PRELUDE) + "void main()\n{\n";
  source += "  int i = int(gl_GlobalInvocationID.x);\n";
  source += "  int particle_count = int(params[0]);\n";
  source += "  if (i >= particle_count) { return; }\n";
  source += "  int table_size = int(params[" + std::to_string(table_size_param) + "]);\n";
  source += "  int mask = table_size - 1;\n";
  source += "  float inv_cell_size = 1.0 / params[2];\n";
  source += "  float r_sq_max = params[3];\n";
  source += "  int base = i * 3;\n";
  source +=
      "  vec3 p_i = vec3(position_in[base + 0], position_in[base + 1], position_in[base + 2]);\n";
  source += "  int cx = int(floor(p_i.x * inv_cell_size));\n";
  source += "  int cy = int(floor(p_i.y * inv_cell_size));\n";
  source += "  int cz = int(floor(p_i.z * inv_cell_size));\n";
  source += "  int count = 0;\n";
  if (!counting) {
    source += "  int write_i = " + std::string(offsets_name) + "[i];\n";
  }
  source += "  for (int dz = -1; dz <= 1; dz++) {\n";
  source += "    for (int dy = -1; dy <= 1; dy++) {\n";
  source += "      for (int dx = -1; dx <= 1; dx++) {\n";
  source += "        int nx = cx + dx;\n";
  source += "        int ny = cy + dy;\n";
  source += "        int nz = cz + dz;\n";
  source += "        int bucket = int(ha_hash_cell(nx, ny, nz) & uint(mask));\n";
  source += "        int begin = bucket_offsets[bucket];\n";
  source += "        int end = begin + bucket_counts[bucket];\n";
  source += "        for (int k = begin; k < end; k++) {\n";
  source += "          int target_i = sorted_in[k];\n";
  source += "          int cb = target_i * 3;\n";
  source += "          if (cells[cb + 0] != nx || cells[cb + 1] != ny || cells[cb + 2] != nz) {\n";
  source += "            continue;\n";
  source += "          }\n";
  source += "          int bp = target_i * " + std::to_string(positions_stride) + ";\n";
  source += "          vec3 r_vec = p_i - vec3(" + std::string(positions_base) + "[bp + 0],\n";
  source += "                                  " + std::string(positions_base) + "[bp + 1],\n";
  source += "                                  " + std::string(positions_base) + "[bp + 2]);\n";
  source += "          if (dot(r_vec, r_vec) >= r_sq_max) { continue; }\n";
  if (counting) {
    source += "          count += 1;\n";
  }
  else {
    source += "          if (write_i < int(params[" + std::to_string(capacity_param) + "])) {\n";
    source += "            " + std::string(indices_name) + "[write_i] = target_i;\n";
    source += "          }\n";
    source += "          write_i += 1;\n";
  }
  source += "        }\n";
  source += "      }\n";
  source += "    }\n";
  source += "  }\n";
  if (counting) {
    source += "  " + std::string(counts_name) + "[i] = count;\n";
  }
  source += "}\n";
  return source;
}

std::string boundary_query_kernel(const bool counting)
{
  return neighbor_query_kernel(counting,
                               "boundary_in",
                               4,
                               "boundary_counts",
                               "boundary_offsets",
                               "boundary_indices",
                               FLUID_PARAM_BOUNDARY_TABLE_SIZE,
                               FLUID_PARAM_BOUNDARY_CAPACITY);
}

std::string coupled_query_kernel(const bool counting)
{
  return neighbor_query_kernel(counting,
                               "coupled_in",
                               5,
                               "coupled_counts",
                               "coupled_offsets",
                               "coupled_indices",
                               FLUID_PARAM_COUPLED_TABLE_SIZE,
                               FLUID_PARAM_COUPLED_CAPACITY);
}

/* --- collider contacts --------------------------------------------------- */

/** Position solve of one contact, a direct port of `xpbd::CollisionFaceConstraintSet::solve_single`. */
const char *KERNEL_CONTACT_POSITION =
    "void main()\n"
    "{\n"
    "  int ci = int(gl_GlobalInvocationID.x);\n"
    "  if (ci >= int(params[18])) { return; }\n"
    "  int point_i = contact_points[ci];\n"
    "  if (point_i < 0 || point_i >= int(params[0])) { return; }\n"
    "  int base = point_i * 3;\n"
    "  float inv_m = props[point_i * HA_PROP + 1];\n"
    "  if (inv_m <= 0.0) { return; }\n"
    "  vec3 face_position = vec3(contact_vec[ci * 12 + 0], contact_vec[ci * 12 + 1],\n"
    "                            contact_vec[ci * 12 + 2]);\n"
    "  vec3 face_normal = vec3(contact_vec[ci * 12 + 3], contact_vec[ci * 12 + 4],\n"
    "                          contact_vec[ci * 12 + 5]);\n"
    "  vec3 collider_motion = vec3(contact_vec[ci * 12 + 6], contact_vec[ci * 12 + 7],\n"
    "                              contact_vec[ci * 12 + 8]);\n"
    "  float margin = contact_scalar[ci * 6 + 0];\n"
    "  float compliance = contact_scalar[ci * 6 + 1];\n"
    "  float static_friction = contact_scalar[ci * 6 + 2];\n"
    "  float radius = contact_scalar[ci * 6 + 5];\n"
    "  vec3 position = vec3(position_io[base + 0], position_io[base + 1], position_io[base + 2]);\n"
    "  float normal_distance = dot(position - face_position, face_normal) - margin - radius;\n"
    "  if (normal_distance >= 0.0) { return; }\n"
    "  float lambda_normal = contact_lambda[ci * 2 + 0];\n"
    "  float delta_lambda = -normal_distance / (inv_m + compliance);\n"
    "  vec3 offset = delta_lambda * inv_m * face_normal;\n"
    "  lambda_normal += delta_lambda;\n"
    "  contact_lambda[ci * 2 + 0] = lambda_normal;\n"
    "  vec3 prev = vec3(prev_position[base + 0], prev_position[base + 1], prev_position[base + 2]);\n"
    "  vec3 velocity = (position - prev) - collider_motion;\n"
    "  vec3 velocity_tangent = velocity - dot(velocity, face_normal) * face_normal;\n"
    "  vec3 scaled_tangent = velocity_tangent / (inv_m + compliance);\n"
    "  float lambda_tangent_sq = dot(scaled_tangent, scaled_tangent);\n"
    "  float static_limit = static_friction * lambda_normal;\n"
    "  if (lambda_tangent_sq < static_limit * static_limit) {\n"
    "    offset -= velocity_tangent * inv_m / (inv_m + compliance);\n"
    "  }\n"
    "  position += offset;\n"
    "  position_io[base + 0] = position.x;\n"
    "  position_io[base + 1] = position.y;\n"
    "  position_io[base + 2] = position.z;\n"
    "}\n";

/** Velocity friction of one contact, a port of `xpbd::FrictionFaceConstraintSet::solve_single`. */
const char *KERNEL_CONTACT_FRICTION =
    "void main()\n"
    "{\n"
    "  int ci = int(gl_GlobalInvocationID.x);\n"
    "  if (ci >= int(params[18])) { return; }\n"
    "  int point_i = contact_points[ci];\n"
    "  if (point_i < 0 || point_i >= int(params[0])) { return; }\n"
    "  int base = point_i * 3;\n"
    "  float inv_m = props[point_i * HA_PROP + 1];\n"
    "  if (inv_m <= 0.0) { return; }\n"
    "  vec3 axis = vec3(contact_vec[ci * 12 + 3], contact_vec[ci * 12 + 4],\n"
    "                   contact_vec[ci * 12 + 5]);\n"
    "  vec3 contact_velocity = vec3(contact_vec[ci * 12 + 9], contact_vec[ci * 12 + 10],\n"
    "                               contact_vec[ci * 12 + 11]);\n"
    "  float dynamic_friction = contact_scalar[ci * 6 + 3] * params[13];\n"
    "  float lambda_normal = contact_lambda[ci * 2 + 0];\n"
    "  vec3 velocity = vec3(velocity_io[base + 0], velocity_io[base + 1], velocity_io[base + 2]) -\n"
    "                  contact_velocity;\n"
    "  vec3 velocity_tangent = velocity - dot(velocity, axis) * axis;\n"
    "  float residual = length(velocity_tangent);\n"
    "  if (residual <= 0.0) { return; }\n"
    "  vec3 direction = velocity_tangent / residual;\n"
    "  float delta_lambda = min(dynamic_friction * lambda_normal, residual / inv_m);\n"
    "  contact_lambda[ci * 2 + 1] += delta_lambda;\n"
    "  vec3 change = -direction * inv_m * delta_lambda;\n"
    "  velocity_io[base + 0] += change.x;\n"
    "  velocity_io[base + 1] += change.y;\n"
    "  velocity_io[base + 2] += change.z;\n"
    "}\n";

/* --- resource layouts --------------------------------------------------- */

std::array<HaShaderResource, 6> PREDICT_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "position_io[]"},
    {1, HaShaderQualifier::Write, "float", "prev_position[]"},
    {2, HaShaderQualifier::ReadWrite, "float", "velocity_io[]"},
    {3, HaShaderQualifier::Read, "float", "force_in[]"},
    {4, HaShaderQualifier::Read, "float", "props[]"},
    {5, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 4> APPLY_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "position_io[]"},
    {1, HaShaderQualifier::Read, "float", "particle_state[]"},
    {2, HaShaderQualifier::Read, "float", "props[]"},
    {3, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 5> UPDATE_VELOCITY_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "float", "prev_position[]"},
    {2, HaShaderQualifier::Write, "float", "velocity_out[]"},
    {3, HaShaderQualifier::Read, "float", "params[]"},
    {4, HaShaderQualifier::Read, "float", "coupled_push[]"},
}};

/** Density pass with boundary particles. */
std::array<HaShaderResource, 9> DENSITY_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "int", "neighbor_offsets[]"},
    {2, HaShaderQualifier::Read, "int", "neighbor_indices[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::ReadWrite, "float", "particle_state[]"},
    {5, HaShaderQualifier::Read, "float", "params[]"},
    {6, HaShaderQualifier::Read, "float", "boundary_in[]"},
    {7, HaShaderQualifier::Read, "int", "boundary_offsets[]"},
    {8, HaShaderQualifier::Read, "int", "boundary_indices[]"},
}};


/** The same without boundary particles, so that fluid only scenes bind the minimum. */
std::array<HaShaderResource, 6> DENSITY_RESOURCES_NO_BOUNDARY = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "int", "neighbor_offsets[]"},
    {2, HaShaderQualifier::Read, "int", "neighbor_indices[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::ReadWrite, "float", "particle_state[]"},
    {5, HaShaderQualifier::Read, "float", "params[]"},
}};

/** Delta + apply with boundary particles: the positions are read and written. */
std::array<HaShaderResource, 9> DELTA_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "int", "neighbor_offsets[]"},
    {2, HaShaderQualifier::Read, "int", "neighbor_indices[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::ReadWrite, "float", "particle_state[]"},
    {5, HaShaderQualifier::Read, "float", "params[]"},
    {6, HaShaderQualifier::Read, "float", "boundary_in[]"},
    {7, HaShaderQualifier::Read, "int", "boundary_offsets[]"},
    {8, HaShaderQualifier::Read, "int", "boundary_indices[]"},
}};

std::array<HaShaderResource, 6> DELTA_RESOURCES_NO_BOUNDARY = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "int", "neighbor_offsets[]"},
    {2, HaShaderQualifier::Read, "int", "neighbor_indices[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::ReadWrite, "float", "particle_state[]"},
    {5, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 7> VORTICITY_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "int", "neighbor_offsets[]"},
    {2, HaShaderQualifier::Read, "int", "neighbor_indices[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::Read, "float", "velocity_in[]"},
    {5, HaShaderQualifier::ReadWrite, "float", "particle_state[]"},
    {6, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 7> VORTICITY_APPLY_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "int", "neighbor_offsets[]"},
    {2, HaShaderQualifier::Read, "int", "neighbor_indices[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::ReadWrite, "float", "velocity_io[]"},
    {5, HaShaderQualifier::Read, "float", "particle_state[]"},
    {6, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 6> SURFACE_TENSION_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "int", "neighbor_offsets[]"},
    {2, HaShaderQualifier::Read, "int", "neighbor_indices[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::ReadWrite, "float", "velocity_io[]"},
    {5, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 8> XSPH_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "int", "neighbor_offsets[]"},
    {2, HaShaderQualifier::Read, "int", "neighbor_indices[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::Read, "float", "velocity_in[]"},
    {5, HaShaderQualifier::Write, "float", "velocity_out[]"},
    {6, HaShaderQualifier::Read, "float", "params[]"},
    {7, HaShaderQualifier::Read, "float", "particle_state[]"},
}};

std::array<HaShaderResource, 4> DAMPING_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "velocity_io[]"},
    {1, HaShaderQualifier::Read, "float", "props[]"},
    {2, HaShaderQualifier::Read, "float", "particle_state[]"},
    {3, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 8> BOUNDARY_COUNT_RESOURCES = {{    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "float", "boundary_in[]"},
    {2, HaShaderQualifier::Read, "int", "cells[]"},
    {3, HaShaderQualifier::Read, "int", "bucket_offsets[]"},
    {4, HaShaderQualifier::Read, "int", "bucket_counts[]"},
    {5, HaShaderQualifier::Read, "int", "sorted_in[]"},
    {6, HaShaderQualifier::Write, "int", "boundary_counts[]"},
    {7, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 9> BOUNDARY_FILL_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "float", "boundary_in[]"},
    {2, HaShaderQualifier::Read, "int", "cells[]"},
    {3, HaShaderQualifier::Read, "int", "bucket_offsets[]"},
    {4, HaShaderQualifier::Read, "int", "bucket_counts[]"},
    {5, HaShaderQualifier::Read, "int", "sorted_in[]"},
    {6, HaShaderQualifier::Read, "int", "boundary_offsets[]"},
    {7, HaShaderQualifier::Write, "int", "boundary_indices[]"},
    {8, HaShaderQualifier::Read, "float", "params[]"},
}};

/** Coupled samples: the same query as the boundary particles, against the coupled grid. */
std::array<HaShaderResource, 8> COUPLED_COUNT_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "float", "coupled_in[]"},
    {2, HaShaderQualifier::Read, "int", "cells[]"},
    {3, HaShaderQualifier::Read, "int", "bucket_offsets[]"},
    {4, HaShaderQualifier::Read, "int", "bucket_counts[]"},
    {5, HaShaderQualifier::Read, "int", "sorted_in[]"},
    {6, HaShaderQualifier::Write, "int", "coupled_counts[]"},
    {7, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 9> COUPLED_FILL_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_in[]"},
    {1, HaShaderQualifier::Read, "float", "coupled_in[]"},
    {2, HaShaderQualifier::Read, "int", "cells[]"},
    {3, HaShaderQualifier::Read, "int", "bucket_offsets[]"},
    {4, HaShaderQualifier::Read, "int", "bucket_counts[]"},
    {5, HaShaderQualifier::Read, "int", "sorted_in[]"},
    {6, HaShaderQualifier::Read, "int", "coupled_offsets[]"},
    {7, HaShaderQualifier::Write, "int", "coupled_indices[]"},
    {8, HaShaderQualifier::Read, "float", "params[]"},
}};

/**
 * The density and delta passes share their layout: the fluid particles, then the optional collider
 * boundary set and the optional coupled solid set, in that order.
 */
Vector<HaShaderResource> fluid_step_resources(const bool with_boundary, const bool with_coupled)
{
  Vector<HaShaderResource> resources;
  resources.append(HaShaderResource{0, HaShaderQualifier::Read, "float", "position_in[]"});
  resources.append(HaShaderResource{1, HaShaderQualifier::Read, "int", "neighbor_offsets[]"});
  resources.append(HaShaderResource{2, HaShaderQualifier::Read, "int", "neighbor_indices[]"});
  resources.append(HaShaderResource{3, HaShaderQualifier::Read, "float", "props[]"});
  resources.append(HaShaderResource{4, HaShaderQualifier::ReadWrite, "float", "particle_state[]"});
  resources.append(HaShaderResource{5, HaShaderQualifier::Read, "float", "params[]"});
  /* The density pass does not use it, but both passes share this layout. */
  resources.append(HaShaderResource{6, HaShaderQualifier::ReadWrite, "float", "coupled_push[]"});
  int slot = 7;
  if (with_boundary) {
    resources.append(HaShaderResource{slot++, HaShaderQualifier::Read, "float", "boundary_in[]"});
    resources.append(
        HaShaderResource{slot++, HaShaderQualifier::Read, "int", "boundary_offsets[]"});
    resources.append(
        HaShaderResource{slot++, HaShaderQualifier::Read, "int", "boundary_indices[]"});
  }
  if (with_coupled) {
    resources.append(HaShaderResource{slot++, HaShaderQualifier::Read, "float", "coupled_in[]"});
    resources.append(
        HaShaderResource{slot++, HaShaderQualifier::Read, "int", "coupled_offsets[]"});
    resources.append(
        HaShaderResource{slot++, HaShaderQualifier::Read, "int", "coupled_indices[]"});
  }
  return resources;
}

/** Reaction of the fluid on the coupled samples: one invocation per coupled sample. */
std::array<HaShaderResource, 10> COUPLED_REACTION_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "coupled_in[]"},
    {1, HaShaderQualifier::Write, "float", "coupled_reaction_out[]"},
    {2, HaShaderQualifier::Read, "float", "position_in[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::Read, "float", "particle_state[]"},
    {5, HaShaderQualifier::Read, "float", "params[]"},
    {6, HaShaderQualifier::Read, "int", "cells[]"},
    {7, HaShaderQualifier::Read, "int", "bucket_offsets[]"},
    {8, HaShaderQualifier::Read, "int", "bucket_counts[]"},
    {9, HaShaderQualifier::Read, "int", "sorted_in[]"},
}};

std::array<HaShaderResource, 8> CONTACT_POSITION_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "position_io[]"},
    {1, HaShaderQualifier::Read, "float", "prev_position[]"},
    {2, HaShaderQualifier::Read, "float", "props[]"},
    {3, HaShaderQualifier::Read, "float", "contact_vec[]"},
    {4, HaShaderQualifier::Read, "float", "contact_scalar[]"},
    {5, HaShaderQualifier::Read, "int", "contact_points[]"},
    {6, HaShaderQualifier::ReadWrite, "float", "contact_lambda[]"},
    {7, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 8> CONTACT_FRICTION_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "velocity_io[]"},
    {1, HaShaderQualifier::Read, "float", "prev_position[]"},
    {2, HaShaderQualifier::Read, "float", "props[]"},
    {3, HaShaderQualifier::Read, "float", "contact_vec[]"},
    {4, HaShaderQualifier::Read, "float", "contact_scalar[]"},
    {5, HaShaderQualifier::Read, "int", "contact_points[]"},
    {6, HaShaderQualifier::ReadWrite, "float", "contact_lambda[]"},
    {7, HaShaderQualifier::Read, "float", "params[]"},
}};

/** Compiles (or fetches) a kernel and dispatches it with the given buffers. */
gpu::Shader *run_kernel(const StringRef key,
                        const std::string &source,
                        const Span<const HaShaderResource> resources,
                        const std::initializer_list<gpu::StorageBuf *> &buffers,
                        const int element_count)
{
  gpu::Shader *shader = ha_shader_get_source(key, source, resources, LOCAL_GROUP_SIZE);
  if (shader == nullptr) {
    return nullptr;
  }
  std::array<gpu::StorageBuf *, 16> array{};
  int count = 0;
  for (gpu::StorageBuf *buffer : buffers) {
    if (count >= int(array.size())) {
      return shader;
    }
    array[count++] = buffer;
  }
  dispatch_pass(shader, Span<gpu::StorageBuf *>(array.data(), count), element_count);
  return shader;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name HaXpbdFluid
 * \{ */

/* -------------------------------------------------------------------- */
/** \name Collider distance field
 * \{ */

/**
 * Trilinear sampling of the dense collider distance field (`GEO_ha_sdf`). The lattice
 * matches the host side: voxel `i` is centered at `origin + (i + 0.5) * spacing`.
 */
const char *GLSL_SDF_HELPERS =
    "float ha_sdf_texel(ivec3 cell, ivec3 dims)\n"
    "{\n"
    "  ivec3 c = clamp(cell, ivec3(0), dims - ivec3(1));\n"
    "  return sdf_data[(c.z * dims.y + c.y) * dims.x + c.x];\n"
    "}\n"
    "// Trilinear value and its analytic gradient from the same eight texels: the central differences\n"
    "// used to need six more samples per particle, which dominated the contact cost.\n"
    "void ha_sdf_value_normal(vec3 p, out float value, out vec3 normal)\n"
    "{\n"
    "  vec3 origin = vec3(sdf_params[0], sdf_params[1], sdf_params[2]);\n"
    "  float spacing = sdf_params[3];\n"
    "  ivec3 dims = ivec3(int(sdf_params[4]), int(sdf_params[5]), int(sdf_params[6]));\n"
    "  vec3 local = (p - origin) / spacing - vec3(0.5);\n"
    "  ivec3 base = ivec3(floor(local));\n"
    "  vec3 f = local - vec3(base);\n"
    "  float c000 = ha_sdf_texel(base, dims);\n"
    "  float c100 = ha_sdf_texel(base + ivec3(1, 0, 0), dims);\n"
    "  float c010 = ha_sdf_texel(base + ivec3(0, 1, 0), dims);\n"
    "  float c110 = ha_sdf_texel(base + ivec3(1, 1, 0), dims);\n"
    "  float c001 = ha_sdf_texel(base + ivec3(0, 0, 1), dims);\n"
    "  float c101 = ha_sdf_texel(base + ivec3(1, 0, 1), dims);\n"
    "  float c011 = ha_sdf_texel(base + ivec3(0, 1, 1), dims);\n"
    "  float c111 = ha_sdf_texel(base + ivec3(1, 1, 1), dims);\n"
    "  float c00 = mix(c000, c100, f.x);\n"
    "  float c10 = mix(c010, c110, f.x);\n"
    "  float c01 = mix(c001, c101, f.x);\n"
    "  float c11 = mix(c011, c111, f.x);\n"
    "  value = mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);\n"
    "  vec3 gradient = vec3((c100 - c000) * (1.0 - f.y) * (1.0 - f.z) +\n"
    "                           (c110 - c010) * f.y * (1.0 - f.z) +\n"
    "                           (c101 - c001) * (1.0 - f.y) * f.z +\n"
    "                           (c111 - c011) * f.y * f.z,\n"
    "                       (c010 - c000) * (1.0 - f.x) * (1.0 - f.z) +\n"
    "                           (c110 - c100) * f.x * (1.0 - f.z) +\n"
    "                           (c011 - c001) * (1.0 - f.x) * f.z +\n"
    "                           (c111 - c101) * f.x * f.z,\n"
    "                       (c001 - c000) * (1.0 - f.x) * (1.0 - f.y) +\n"
    "                           (c101 - c100) * f.x * (1.0 - f.y) +\n"
    "                           (c011 - c010) * (1.0 - f.x) * f.y +\n"
    "                           (c111 - c110) * f.x * f.y);\n"
    "  float length_sq = dot(gradient, gradient);\n"
    "  normal = length_sq < 1e-12 ? vec3(0.0, 0.0, 1.0) : gradient * inversesqrt(length_sq);\n"
    "}\n";

/** Pushes the particles out of the collider, one trilinear sample per particle. */
const char *KERNEL_SDF_CONTACT =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(params[0])) { return; }\n"
    "  if (sdf_params[10] <= 0.0) { return; }\n"
    "  if (props[i * HA_PROP + 1] <= 0.0) { return; }\n"
    "  int base = i * 3;\n"
    "  vec3 p = vec3(position_io[base + 0], position_io[base + 1], position_io[base + 2]);\n"
    "  float distance_to_surface = 0.0;\n"
    "  vec3 normal = vec3(0.0, 0.0, 1.0);\n"
    "  ha_sdf_value_normal(p, distance_to_surface, normal);\n"
    "  float contact_radius = sdf_params[7];\n"
    "  particle_state[i * HA_STATE + 7] = 0.0;\n"
    "  if (distance_to_surface >= contact_radius) { return; }\n"
    "  float push = contact_radius - distance_to_surface;\n"
    "  float max_correction = sdf_params[9];\n"
    "  if (max_correction > 0.0) { push = min(push, max_correction); }\n"
    "  vec3 correction = normal * push;\n"
    "  if (!ha_is_finite(correction)) { return; }\n"
    "  position_io[base + 0] += correction.x;\n"
    "  position_io[base + 1] += correction.y;\n"
    "  position_io[base + 2] += correction.z;\n"
    "  particle_state[i * HA_STATE + 7] = 1.0;\n"
    "}\n";

/**
 * Contact with the distance field of the *coupled* solids. It uses the same `sdf_data` /
 * `sdf_params` helper as the collider field (the buffers bound to those slots are the coupled
 * ones), pushes the fluid particle out and appends the correction to the frame contact list. The
 * host spreads that list over the solid vertices, which is the reaction of the fluid on the cloth.
 */
const char *KERNEL_COUPLED_SDF_CONTACT =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(params[0])) { return; }\n"
    "  if (sdf_params[10] <= 0.0) { return; }\n"
    "  if (props[i * HA_PROP + 1] <= 0.0) { return; }\n"
    "  int base = i * 3;\n"
    "  vec3 p = vec3(position_io[base + 0], position_io[base + 1], position_io[base + 2]);\n"
    "  // Outside the field the sampler clamps to the border, which would report a fake contact\n"
    "  // (and a border gradient). Only particles inside the field can touch the coupled solid.\n"
    "  vec3 origin = vec3(sdf_params[0], sdf_params[1], sdf_params[2]);\n"
    "  float spacing = sdf_params[3];\n"
    "  ivec3 dims = ivec3(int(sdf_params[4]), int(sdf_params[5]), int(sdf_params[6]));\n"
    "  vec3 local = (p - origin) / spacing - vec3(0.5);\n"
    "  if (local.x < 0.0 || local.y < 0.0 || local.z < 0.0 ||\n"
    "      local.x > float(dims.x - 1) || local.y > float(dims.y - 1) ||\n"
    "      local.z > float(dims.z - 1)) { return; }\n"
    "  float distance_to_surface = 0.0;\n"
    "  vec3 normal = vec3(0.0, 0.0, 1.0);\n"
    "  ha_sdf_value_normal(p, distance_to_surface, normal);\n"
    "  float contact_radius = sdf_params[7];\n"
    "  float band = sdf_params[11];\n"
    "  if (abs(distance_to_surface) >= band) { return; }\n"
    "  // The fluid lives on the positive side of the sheet (the side the triangle normals point to).\n"
    "  // Always push it back there. A side dependent push treated a particle that slipped through as\n"
    "  // a legitimate lower layer and pushed it further down, which is why the whole column ended up\n"
    "  // below the cloth. The velocity layer cancels the outward velocity this creates, so the water\n"
    "  // is held instead of being launched (that was the M117 explosion).\n"
    "  float signed_distance = distance_to_surface;\n"
    "  if (signed_distance >= contact_radius) { return; }\n"
    "  float push = contact_radius - signed_distance;\n"
    "  float max_correction = sdf_params[9];\n"
    "  if (max_correction > 0.0) { push = min(push, max_correction); }\n"
    "  vec3 correction = normal * push;\n"
    "  if (signed_distance < 0.0) { atomicAdd(coupled_crossings[0], 1); }\n"
    "  if (!ha_is_finite(correction)) { return; }\n"
    "  position_io[base + 0] += correction.x;\n"
    "  position_io[base + 1] += correction.y;\n"
    "  position_io[base + 2] += correction.z;\n"
    "  coupled_push_flags[i] = 1;\n"
    "  int slot = atomicAdd(coupled_contact_count[0], 1);\n"
    "  if (slot < int(params[22])) {\n"
    "    coupled_contact_indices[slot] = i;\n"
    "    coupled_contact_push[slot * 3 + 0] = correction.x;\n"
    "    coupled_contact_push[slot * 3 + 1] = correction.y;\n"
    "    coupled_contact_push[slot * 3 + 2] = correction.z;\n"
    "  }\n"
    "}\n";

std::array<HaShaderResource, 11> COUPLED_SDF_CONTACT_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "position_io[]"},
    {1, HaShaderQualifier::Read, "float", "sdf_data[]"},
    {2, HaShaderQualifier::Read, "float", "sdf_params[]"},
    {3, HaShaderQualifier::Read, "float", "props[]"},
    {4, HaShaderQualifier::Read, "float", "params[]"},
    {5, HaShaderQualifier::ReadWrite, "int", "coupled_contact_count[]"},
    {6, HaShaderQualifier::Write, "int", "coupled_contact_indices[]"},
    {7, HaShaderQualifier::Write, "float", "coupled_contact_push[]"},
    {8, HaShaderQualifier::Read, "float", "prev_positions[]"},
    {9, HaShaderQualifier::Write, "int", "coupled_push_flags[]"},
    {10, HaShaderQualifier::ReadWrite, "int", "coupled_crossings[]"},
}};

/**
 * Velocity layer of the coupled contact. A position projection turns the depenetration into
 * outward velocity (`v = (p - p_prev) / dt`), and on a sheet that is exactly what launched the
 * water off the cloth. The particles that the contact moved this substep therefore get their
 * normal velocity removed: the contact is inelastic and they can only slide along the surface.
 */
const char *KERNEL_COUPLED_SDF_VELOCITY =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(params[0])) { return; }\n"
    "  if (sdf_params[10] <= 0.0) { return; }\n"
    "  if (coupled_push_flags[i] == 0) { return; }\n"
    "  coupled_push_flags[i] = 0;\n"
    "  int base = i * 3;\n"
    "  vec3 p = vec3(position_io[base + 0], position_io[base + 1], position_io[base + 2]);\n"
    "  float d = 0.0;\n"
    "  vec3 normal = vec3(0.0, 0.0, 1.0);\n"
    "  ha_sdf_value_normal(p, d, normal);\n"
    "  float side = d >= 0.0 ? 1.0 : -1.0;\n"
    "  vec3 outward = normal * side;\n"
    "  vec3 v = vec3(velocity_io[base + 0], velocity_io[base + 1], velocity_io[base + 2]);\n"
    "  v -= outward * dot(v, outward);\n"
    "  velocity_io[base + 0] = v.x;\n"
    "  velocity_io[base + 1] = v.y;\n"
    "  velocity_io[base + 2] = v.z;\n"
    "}\n";

std::array<HaShaderResource, 7> COUPLED_SDF_VELOCITY_RESOURCES = {{
    {0, HaShaderQualifier::Read, "float", "position_io[]"},
    {1, HaShaderQualifier::Read, "float", "sdf_data[]"},
    {2, HaShaderQualifier::Read, "float", "sdf_params[]"},
    {3, HaShaderQualifier::ReadWrite, "float", "velocity_io[]"},
    {4, HaShaderQualifier::ReadWrite, "int", "coupled_push_flags[]"},
    {5, HaShaderQualifier::Read, "float", "props[]"},
    {6, HaShaderQualifier::Read, "float", "params[]"},
}};

/** No-slip friction against the collider, after the fluid velocity layer. */
const char *KERNEL_SDF_FRICTION =
    "void main()\n"
    "{\n"
    "  int i = int(gl_GlobalInvocationID.x);\n"
    "  if (i >= int(params[0])) { return; }\n"
    "  if (sdf_params[10] <= 0.0) { return; }\n"
    "  if (props[i * HA_PROP + 1] <= 0.0) { return; }\n"
    "  int base = i * 3;\n"
    "  vec3 p = vec3(position_in[base + 0], position_in[base + 1], position_in[base + 2]);\n"
    "  float contact_radius = sdf_params[7];\n"
    "  float distance_to_surface = 0.0;\n"
    "  vec3 normal = vec3(0.0, 0.0, 1.0);\n"
    "  ha_sdf_value_normal(p, distance_to_surface, normal);\n"
    "  if (distance_to_surface >= contact_radius) { return; }\n"
    "  vec3 v = vec3(velocity_io[base + 0], velocity_io[base + 1], velocity_io[base + 2]);\n"
    "  v -= normal * dot(v, normal);\n"
    "  vec3 tangent = v - normal * dot(v, normal);\n"
    "  v -= tangent * clamp(sdf_params[8], 0.0, 1.0);\n"
    "  velocity_io[base + 0] = v.x;\n"
    "  velocity_io[base + 1] = v.y;\n"
    "  velocity_io[base + 2] = v.z;\n"
    "}\n";

std::array<HaShaderResource, 6> SDF_CONTACT_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "position_io[]"},
    {1, HaShaderQualifier::Read, "float", "sdf_data[]"},
    {2, HaShaderQualifier::Read, "float", "sdf_params[]"},
    {3, HaShaderQualifier::ReadWrite, "float", "particle_state[]"},
    {4, HaShaderQualifier::Read, "float", "props[]"},
    {5, HaShaderQualifier::Read, "float", "params[]"},
}};

std::array<HaShaderResource, 6> SDF_FRICTION_RESOURCES = {{
    {0, HaShaderQualifier::ReadWrite, "float", "velocity_io[]"},
    {1, HaShaderQualifier::Read, "float", "sdf_data[]"},
    {2, HaShaderQualifier::Read, "float", "sdf_params[]"},
    {3, HaShaderQualifier::Read, "float", "position_in[]"},
    {4, HaShaderQualifier::Read, "float", "params[]"},
    {5, HaShaderQualifier::Read, "float", "props[]"},
}};

/** \} */

bool HaXpbdFluid::bind_sdf(const HaSdfGrid &grid,
                            const float radius,
                            const float friction,
                            const float max_correction)
{
  if (!ready_) {
    last_error_ = "not bound";
    return false;
  }
  if (!grid.is_valid()) {
    this->clear_sdf();
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    last_error_ = "no GPU context";
    return false;
  }
  if (!sdf_data_.ensure(grid.data.size()) || !sdf_params_.ensure(12)) {
    last_error_ = "sdf allocation failed";
    this->clear_sdf();
    return false;
  }
  sdf_data_.upload(grid.data.as_span());
  const std::array<float, 12> params{grid.origin.x,
                                     grid.origin.y,
                                     grid.origin.z,
                                     grid.spacing,
                                     float(grid.dims.x),
                                     float(grid.dims.y),
                                     float(grid.dims.z),
                                     radius,
                                     friction,
                                     max_correction,
                                     1.0f,
                                     0.0f};
  sdf_params_.upload(params);
  sdf_active_ = true;
  return true;
}

void HaXpbdFluid::clear_sdf()
{
  sdf_active_ = false;
  sdf_data_.free();
  sdf_params_.free();
}

bool HaXpbdFluid::bind_coupled_sdf(const HaSdfGrid &grid,
                                   const float radius,
                                   const float friction,
                                   const float max_correction,
                                   const float band)
{
  if (!ready_) {
    last_error_ = "not bound";
    return false;
  }
  if (!grid.is_valid()) {
    this->clear_coupled_sdf();
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    last_error_ = "no GPU context";
    return false;
  }
  if (!coupled_sdf_data_.ensure(grid.data.size()) || !coupled_sdf_params_.ensure(12)) {
    last_error_ = "coupled sdf allocation failed";
    this->clear_coupled_sdf();
    return false;
  }
  coupled_sdf_data_.upload(grid.data.as_span());
  const std::array<float, 12> params{grid.origin.x,
                                     grid.origin.y,
                                     grid.origin.z,
                                     grid.spacing,
                                     float(grid.dims.x),
                                     float(grid.dims.y),
                                     float(grid.dims.z),
                                     radius,
                                     friction,
                                     max_correction,
                                     1.0f,
                                     band};
  coupled_sdf_params_.upload(params);

  const int capacity = std::max(64, particle_count_ * 4);
  if (!coupled_contact_count_.ensure(1) || !coupled_contact_indices_.ensure(capacity) ||
      !coupled_contact_push_.ensure(int64_t(capacity) * 3) ||
      !coupled_push_flags_.ensure(particle_count_) || !coupled_crossings_.ensure(1))
  {
    last_error_ = "coupled contact allocation failed";
    this->clear_coupled_sdf();
    return false;
  }
  coupled_contact_capacity_ = capacity;
  coupled_contact_count_.clear_to_zero();
  coupled_push_flags_.clear_to_zero();
  coupled_crossings_.clear_to_zero();
  std::array<float, FLUID_PARAM_COUNT_TOTAL> fluid_params{};
  params_.download(fluid_params);
  fluid_params[FLUID_PARAM_COUPLED_CONTACT_CAPACITY] = float(capacity);
  params_.upload(fluid_params);

  coupled_sdf_active_ = true;
  return true;
}

void HaXpbdFluid::clear_coupled_sdf()
{
  coupled_sdf_active_ = false;
  coupled_sdf_data_.free();
  coupled_sdf_params_.free();
  coupled_contact_count_.free();
  coupled_contact_indices_.free();
  coupled_contact_push_.free();
  coupled_push_flags_.free();
  coupled_crossings_.free();
  coupled_contact_capacity_ = 0;
}

bool HaXpbdFluid::download_coupled_contacts(MutableSpan<int> r_indices,
                                            MutableSpan<float3> r_push,
                                            int &r_count) const
{
  r_count = 0;
  if (!coupled_sdf_active_ || coupled_contact_count_.size() < 1) {
    return true;
  }
  int count = 0;
  if (!coupled_contact_count_.read_one(count)) {
    return false;
  }
  count = std::min(count, coupled_contact_capacity_);
  count = int(std::min<int64_t>(count, r_indices.size()));
  count = int(std::min<int64_t>(count, r_push.size()));
  if (count <= 0) {
    return true;
  }
  coupled_contact_indices_.download(r_indices.take_front(count));
  Array<float> push(int64_t(count) * 3);
  coupled_contact_push_.download(push.as_mutable_span().take_front(int64_t(count) * 3));
  for (const int i : IndexRange(count)) {
    r_push[i] = float3(push[i * 3 + 0], push[i * 3 + 1], push[i * 3 + 2]);
  }
  if (getenv("XPBD_GPU_DEBUG") != nullptr) {
    float max_len = 0.0f;
    float sum_z = 0.0f;
    int up = 0;
    int down = 0;
    for (const int i : IndexRange(count)) {
      const float3 &p = r_push[i];
      max_len = std::max(max_len, std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z));
      sum_z += p.z;
      up += p.z > 0.0f ? 1 : 0;
      down += p.z < 0.0f ? 1 : 0;
    }
    int crossings = 0;
    coupled_crossings_.read_one(crossings);
    printf("XPBD_GPU coupled contacts count=%d max_push=%.4f mean_z=%.4f up=%d down=%d "
           "crossings=%d\n",
           count,
           double(max_len),
           double(sum_z / float(std::max(1, count))),
           up,
           down,
           crossings);
  }
  r_count = count;
  return true;
}

HaXpbdFluid::~HaXpbdFluid()
{
  this->free();
}

void HaXpbdFluid::free()
{
  this->clear_coupled_sdf();
  grid_.free();
  boundary_grid_.free();
  props_.free();
  particle_state_.free();
  coupled_push_.free();
  velocities_.free();
  velocities_tmp_.free();
  prev_positions_.free();
  external_force_.free();
  boundary_particles_.free();
  boundary_offsets_.free();
  boundary_indices_.free();
  boundary_counts_.free();
  coupled_particles_.free();
  coupled_grid_.free();
  coupled_offsets_.free();
  coupled_indices_.free();
  coupled_counts_.free();
  coupled_reaction_.free();
  coupled_count_ = 0;
  coupled_capacity_ = 0;
  coupled_table_size_ = 0;
  contact_vec_.free();
  contact_scalar_.free();
  contact_points_.free();
  contact_lambda_.free();
  contact_count_ = 0;
  params_.free();
  particle_count_ = 0;
  boundary_count_ = 0;
  boundary_capacity_ = 0;
  ready_ = false;
}

bool HaXpbdFluid::set_contacts(const HaXpbdContacts &contacts)
{
  const int count = int(contacts.points.size());
  contact_count_ = 0;
  if (count <= 0) {
    return true;
  }
  if (!contact_vec_.ensure(int64_t(count) * HA_CONTACT_VEC_STRIDE) ||
      !contact_scalar_.ensure(int64_t(count) * HA_CONTACT_SCALAR_STRIDE) ||
      !contact_points_.ensure(count) || !contact_lambda_.ensure(int64_t(count) * 2))
  {
    return false;
  }
  Array<float> host_vec(int64_t(count) * HA_CONTACT_VEC_STRIDE);
  Array<float> host_scalar(int64_t(count) * HA_CONTACT_SCALAR_STRIDE);
  Array<float> host_lambda(int64_t(count) * 2);
  Array<int> host_points(count);
  for (const int i : IndexRange(count)) {
    const int vb = i * HA_CONTACT_VEC_STRIDE;
    const int sb = i * HA_CONTACT_SCALAR_STRIDE;
    host_vec[vb + 0] = contacts.positions_on_face[i].x;
    host_vec[vb + 1] = contacts.positions_on_face[i].y;
    host_vec[vb + 2] = contacts.positions_on_face[i].z;
    host_vec[vb + 3] = contacts.face_normals[i].x;
    host_vec[vb + 4] = contacts.face_normals[i].y;
    host_vec[vb + 5] = contacts.face_normals[i].z;
    host_vec[vb + 6] = contacts.collider_motion[i].x;
    host_vec[vb + 7] = contacts.collider_motion[i].y;
    host_vec[vb + 8] = contacts.collider_motion[i].z;
    host_vec[vb + 9] = contacts.collider_velocities[i].x;
    host_vec[vb + 10] = contacts.collider_velocities[i].y;
    host_vec[vb + 11] = contacts.collider_velocities[i].z;
    host_scalar[sb + 0] = contacts.margins[i];
    host_scalar[sb + 1] = contacts.compliance_terms[i];
    host_scalar[sb + 2] = contacts.static_frictions[i];
    host_scalar[sb + 3] = contacts.dynamic_frictions[i];
    host_scalar[sb + 4] = contacts.error_scales[i];
    host_scalar[sb + 5] = contacts.radii[i];
    host_lambda[i * 2 + 0] = contacts.lambdas_normal[i];
    host_lambda[i * 2 + 1] = contacts.lambdas_friction[i];
    host_points[i] = contacts.points[i];
  }
  contact_vec_.upload(host_vec);
  contact_scalar_.upload(host_scalar);
  contact_lambda_.upload(host_lambda);
  contact_points_.upload(host_points);
  contact_count_ = count;

  std::array<float, FLUID_PARAM_COUNT_TOTAL> params{};
  params_.download(params);
  params[FLUID_PARAM_CONTACT_COUNT] = float(count);
  params_.upload(params);
  return true;
}

bool HaXpbdFluid::download_contact_lambdas(const MutableSpan<float> r_lambdas_normal,
                                           const MutableSpan<float> r_lambdas_friction) const
{
  if (contact_count_ <= 0 || !contact_lambda_.is_valid()) {
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    return false;
  }
  Array<float> host_lambda(int64_t(contact_count_) * 2);
  contact_lambda_.download(host_lambda);
  const int count = std::min<int64_t>(r_lambdas_normal.size(), contact_count_);
  for (const int i : IndexRange(count)) {
    r_lambdas_normal[i] = host_lambda[i * 2 + 0];
  }
  const int friction_count = std::min<int64_t>(r_lambdas_friction.size(), contact_count_);
  for (const int i : IndexRange(friction_count)) {
    r_lambdas_friction[i] = host_lambda[i * 2 + 1];
  }
  return true;
}

bool HaXpbdFluid::download_positions(const MutableSpan<float3> r_positions)
{
  if (!ready_) {
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    return false;
  }
  const int count = std::min<int64_t>(r_positions.size(), particle_count_);
  if (count <= 0) {
    return false;
  }
  Array<float> host_positions(int64_t(particle_count_) * 3);
  grid_.positions.download(host_positions);
  for (const int i : IndexRange(count)) {
    r_positions[i] = float3(
        host_positions[i * 3 + 0], host_positions[i * 3 + 1], host_positions[i * 3 + 2]);
  }
  return true;
}

bool HaXpbdFluid::download_prev_positions(const MutableSpan<float3> r_positions)
{
  if (!ready_) {
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    return false;
  }
  const int count = std::min<int64_t>(r_positions.size(), particle_count_);
  if (count <= 0) {
    return false;
  }
  Array<float> host_positions(int64_t(particle_count_) * 3);
  prev_positions_.download(host_positions);
  for (const int i : IndexRange(count)) {
    r_positions[i] = float3(
        host_positions[i * 3 + 0], host_positions[i * 3 + 1], host_positions[i * 3 + 2]);
  }
  return true;
}

bool HaXpbdFluid::bind(const HaXpbdFluidInputs &inputs,
                       const float support_radius,
                       const float delta_time,
                       const float compliance_term,
                       const float artificial_pressure,
                       const float max_correction)
{
  last_error_ = "";
  const int particle_count = int(inputs.positions.size());
  const int boundary_count = int(inputs.boundary_positions.size());
  if (particle_count == 0 || support_radius <= 0.0f || delta_time <= 0.0f) {
    last_error_ = "empty state or no time step";
    return false;
  }
  if (!ha_gpu_available()) {
    last_error_ = "no GPU backend";
    return false;
  }
  /* The host may have bound other buffers to the slots since the last frame (draw, compositor,
   * another node), so the binding cache starts empty for every frame. */
  ha_bindings_invalidate();
  this->clear_sdf();
  HaContextGuard guard;
  if (!guard.is_valid()) {
    last_error_ = "no GPU context";
    return false;
  }
  if (int(inputs.velocities.size()) != particle_count ||
      int(inputs.external_force.size()) != particle_count ||
      int(inputs.masses.size()) != particle_count ||
      int(inputs.inv_masses.size()) != particle_count ||
      int(inputs.rest_density.size()) != particle_count ||
      int(inputs.negative_pressure_limit.size()) != particle_count ||
      int(inputs.viscosity.size()) != particle_count ||
      int(inputs.damping.size()) != particle_count ||
      int(inputs.vorticity.size()) != particle_count ||
      int(inputs.surface_tension.size()) != particle_count ||
      int(inputs.boundary_friction.size()) != particle_count ||
      int(inputs.use_boundary.size()) != particle_count ||
      int(inputs.coupling.size()) != particle_count ||
      int(inputs.boundary_masses.size()) != boundary_count)
  {
    last_error_ = "input size mismatch";
    return false;
  }

  particle_count_ = particle_count;
  boundary_count_ = boundary_count;
  delta_time_ = delta_time;
  support_radius_ = support_radius;
  velocities_in_tmp_ = false;
  frame_stats_ = {};
  ready_ = false;

  if (!ha_xpbd_grid_prepare(grid_, particle_count, support_radius)) {
    last_error_ = "grid allocation failed";
    return false;
  }
  const int64_t vec_floats = int64_t(particle_count) * 3;
  if (!props_.ensure(int64_t(particle_count) * HA_PROP_STRIDE) ||
      !particle_state_.ensure(int64_t(particle_count) * HA_STATE_STRIDE) ||
      !velocities_.ensure(vec_floats) || !velocities_tmp_.ensure(vec_floats) ||
      !prev_positions_.ensure(vec_floats) || !external_force_.ensure(vec_floats) ||
      !coupled_push_.ensure(vec_floats) || !params_.ensure(FLUID_PARAM_COUNT_TOTAL))
  {
    last_error_ = "buffer allocation failed";
    return false;
  }
  coupled_push_.clear_to_zero();

  const int64_t prop_floats = int64_t(particle_count) * HA_PROP_STRIDE;
  Array<float> host_positions(vec_floats);
  Array<float> host_velocities(vec_floats);
  Array<float> host_forces(vec_floats);
  Array<float> host_props(prop_floats);
  for (const int i : IndexRange(particle_count)) {
    host_positions[i * 3 + 0] = inputs.positions[i].x;
    host_positions[i * 3 + 1] = inputs.positions[i].y;
    host_positions[i * 3 + 2] = inputs.positions[i].z;
    host_velocities[i * 3 + 0] = inputs.velocities[i].x;
    host_velocities[i * 3 + 1] = inputs.velocities[i].y;
    host_velocities[i * 3 + 2] = inputs.velocities[i].z;
    host_forces[i * 3 + 0] = inputs.external_force[i].x;
    host_forces[i * 3 + 1] = inputs.external_force[i].y;
    host_forces[i * 3 + 2] = inputs.external_force[i].z;
    const int prop_base = i * HA_PROP_STRIDE;
    host_props[prop_base + HA_PROP_MASS] = inputs.masses[i];
    host_props[prop_base + HA_PROP_INV_MASS] = inputs.inv_masses[i];
    host_props[prop_base + HA_PROP_REST_DENSITY] = inputs.rest_density[i];
    host_props[prop_base + HA_PROP_NEGATIVE_PRESSURE_LIMIT] = inputs.negative_pressure_limit[i];
    host_props[prop_base + HA_PROP_VISCOSITY] = inputs.viscosity[i];
    host_props[prop_base + HA_PROP_DAMPING] = inputs.damping[i];
    host_props[prop_base + HA_PROP_VORTICITY] = inputs.vorticity[i];
    host_props[prop_base + HA_PROP_SURFACE_TENSION] = inputs.surface_tension[i];
    host_props[prop_base + HA_PROP_FRICTION] = inputs.boundary_friction[i];
    host_props[prop_base + HA_PROP_USE_BOUNDARY] = inputs.use_boundary[i] ? 1.0f : 0.0f;
    host_props[prop_base + HA_PROP_COUPLING] = inputs.coupling[i];
  }
  grid_.positions.upload(host_positions);
  velocities_.upload(host_velocities);
  external_force_.upload(host_forces);
  props_.upload(host_props);

  /* Boundary particles: packed as `xyz` + mass, plus a grid that uses the same cell size as the
   * fluid grid so that both neighbour queries share the radius. */
  if (boundary_count > 0) {
    if (!boundary_particles_.ensure(int64_t(boundary_count) * HA_BOUNDARY_STRIDE) ||
        !boundary_counts_.ensure(particle_count) ||
        !ha_xpbd_grid_prepare(boundary_grid_, boundary_count, support_radius))
    {
      last_error_ = "boundary allocation failed";
      return false;
    }
    Array<float> host_boundary(int64_t(boundary_count) * HA_BOUNDARY_STRIDE);
    Array<float> host_boundary_positions(int64_t(boundary_count) * 3);
    for (const int i : IndexRange(boundary_count)) {
      const float3 &position = inputs.boundary_positions[i];
      host_boundary[i * 4 + 0] = position.x;
      host_boundary[i * 4 + 1] = position.y;
      host_boundary[i * 4 + 2] = position.z;
      host_boundary[i * 4 + 3] = inputs.boundary_masses[i];
      host_boundary_positions[i * 3 + 0] = position.x;
      host_boundary_positions[i * 3 + 1] = position.y;
      host_boundary_positions[i * 3 + 2] = position.z;
    }
    boundary_particles_.upload(host_boundary);
    boundary_grid_.positions.upload(host_boundary_positions);
    if (!ha_xpbd_grid_build_device(boundary_grid_)) {
      last_error_ = "boundary grid failed";
      return false;
    }
  }

  /* Kernel scales, computed exactly like `xpbd::FluidKernel` (same float expressions, so the
   * two backends evaluate the same kernels at the same points). */
  constexpr float pi = 3.14159265358979323846f;
  const float h = support_radius;
  const float h_sq = h * h;
  const float poly6_scale = (315.0f / (64.0f * pi)) / (h_sq * h_sq * h_sq * h_sq * h);
  const float spiky_scale = (45.0f / pi) / (h_sq * h_sq * h_sq);
  const float h_cubed = h * h * h;
  const float cohesion_scale = 32.0f / (pi * h_cubed * h_cubed * h_cubed);
  const float inv_artificial_ref = artificial_pressure > 0.0f ?
                                       (1.0f / fluid_poly6_at(poly6_scale, h, 0.2f * h)) :
                                       0.0f;

  std::array<float, FLUID_PARAM_COUNT_TOTAL> params{};
  params[FLUID_PARAM_COUNT] = float(particle_count);
  params[FLUID_PARAM_TABLE_SIZE] = float(grid_.table_size);
  params[FLUID_PARAM_CELL_SIZE] = h;
  params[FLUID_PARAM_R_SQ_MAX] = h_sq;
  params[FLUID_PARAM_H] = h;
  params[FLUID_PARAM_H_SQ] = h_sq;
  params[FLUID_PARAM_POLY6] = poly6_scale;
  params[FLUID_PARAM_SPIKY] = spiky_scale;
  params[FLUID_PARAM_COMPLIANCE] = compliance_term;
  params[FLUID_PARAM_ARTIFICIAL] = artificial_pressure;
  params[FLUID_PARAM_INV_ARTIFICIAL_REF] = inv_artificial_ref;
  params[FLUID_PARAM_MAX_CORRECTION] = max_correction;
  params[FLUID_PARAM_DT] = delta_time;
  params[FLUID_PARAM_INV_DT] = 1.0f / delta_time;
  params[FLUID_PARAM_VORTICITY_MAX_CHANGE] = 0.1f * h / delta_time;
  params[FLUID_PARAM_COHESION] = cohesion_scale;
  params[FLUID_PARAM_BOUNDARY_TABLE_SIZE] = float(boundary_count > 0 ? boundary_grid_.table_size :
                                                                      0);
  params[FLUID_PARAM_BOUNDARY_CAPACITY] = 0.0f;
  params[FLUID_PARAM_CONTACT_COUNT] = 0.0f;
  /* Coupled solids: the hash table size only depends on the sample count, and the CSR capacity is
   * the same heuristic as the collider boundary list (the fill pass clamps to it). */
  int coupled_table_size = 1024;
  while (coupled_table_size < int(inputs.coupled_positions.size()) * 2 &&
         coupled_table_size < (1 << 22))
  {
    coupled_table_size <<= 1;
  }
  params[FLUID_PARAM_COUPLED_TABLE_SIZE] = float(coupled_table_size);
  params[FLUID_PARAM_COUPLED_CAPACITY] = float(std::max(64, particle_count * 48));
  params[FLUID_PARAM_COUPLED_COUNT] = float(inputs.coupled_positions.size());
  coupled_table_size_ = coupled_table_size;
  params_.upload(params);

  if (!inputs.coupled_positions.is_empty()) {
    if (!this->update_coupled(inputs.coupled_positions,
                              inputs.coupled_masses,
                              inputs.coupled_inv_masses))
    {
      last_error_ = "coupled allocation failed";
      return false;
    }
  }
  else {
    coupled_count_ = 0;
  }
  if (coupled_count_ > 0) {
    /* The reaction accumulates over the whole frame (all substeps and iterations); the solver reads
     * it back once per frame. A readback per substep costs ~100 ms of Vulkan stall overhead. */
    Array<float> zeros(int64_t(coupled_count_) * 3);
    zeros.fill(0.0f);
    coupled_reaction_.upload(zeros);
  }

  contact_count_ = 0;
  ready_ = true;
  return true;
}

bool HaXpbdFluid::update_coupled(Span<float3> positions,
                                 Span<float> masses,
                                 Span<float> inv_masses)
{
  const int count = int(positions.size());
  coupled_count_ = count;
  if (count == 0) {
    return true;
  }
  if (int(masses.size()) != count || int(inv_masses.size()) != count) {
    last_error_ = "coupled size mismatch";
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    last_error_ = "no GPU context";
    return false;
  }
  if (!coupled_particles_.ensure(int64_t(count) * HA_COUPLED_STRIDE) ||
      !coupled_reaction_.ensure(int64_t(count) * 3))
  {
    last_error_ = "coupled allocation failed";
    return false;
  }
  Array<float> host_particles(int64_t(count) * HA_COUPLED_STRIDE);
  Array<float> host_positions(int64_t(count) * 3);
  for (const int i : IndexRange(count)) {
    const float3 &position = positions[i];
    host_particles[i * HA_COUPLED_STRIDE + 0] = position.x;
    host_particles[i * HA_COUPLED_STRIDE + 1] = position.y;
    host_particles[i * HA_COUPLED_STRIDE + 2] = position.z;
    host_particles[i * HA_COUPLED_STRIDE + 3] = masses[i];
    host_particles[i * HA_COUPLED_STRIDE + 4] = inv_masses[i];
    host_positions[i * 3 + 0] = position.x;
    host_positions[i * 3 + 1] = position.y;
    host_positions[i * 3 + 2] = position.z;
  }
  coupled_particles_.upload(host_particles);
  if (!ha_xpbd_grid_prepare(coupled_grid_, count, support_radius_)) {
    last_error_ = "coupled grid failed";
    return false;
  }
  coupled_grid_.positions.upload(host_positions);
  if (!ha_xpbd_grid_build_device(coupled_grid_)) {
    last_error_ = "coupled grid failed";
    return false;
  }
  /* The fluid -> coupled query masks the hash with the coupled table size. The sample count (and
   * therefore the table size) changes when the coupling is first detected, so the parameter has to
   * follow the grid. A stale (smaller) mask made the query read only a fraction of the buckets and
   * the coupled samples never entered the density. */
  if (coupled_grid_.table_size != coupled_table_size_) {
    coupled_table_size_ = coupled_grid_.table_size;
    std::array<float, FLUID_PARAM_COUNT_TOTAL> params{};
    params_.download(params);
    params[FLUID_PARAM_COUPLED_TABLE_SIZE] = float(coupled_table_size_);
    params_.upload(params);
  }
  return true;
}

bool HaXpbdFluid::download_coupled_reaction(MutableSpan<float3> r_corrections) const
{
  if (coupled_count_ == 0) {
    return true;
  }
  if (int(r_corrections.size()) < coupled_count_) {
    return false;
  }
  Array<float> host(int64_t(coupled_count_) * 3);
  coupled_reaction_.download(host);
  for (const int i : IndexRange(coupled_count_)) {
    r_corrections[i] = float3(host[i * 3 + 0], host[i * 3 + 1], host[i * 3 + 2]);
  }
  return true;
}

bool HaXpbdFluid::build_coupled_neighbors(const int pass_offset)
{
  if (coupled_count_ == 0) {
    return true;
  }
  int pass_count = pass_offset;
  gpu::Shader *count_shader = ha_shader_get_source("xpbd|coupled_count",
                                                   coupled_query_kernel(true),
                                                   as_span(COUPLED_COUNT_RESOURCES),
                                                   LOCAL_GROUP_SIZE);
  gpu::Shader *fill_shader = ha_shader_get_source("xpbd|coupled_fill",
                                                  coupled_query_kernel(false),
                                                  as_span(COUPLED_FILL_RESOURCES),
                                                  LOCAL_GROUP_SIZE);
  if (count_shader == nullptr || fill_shader == nullptr) {
    return false;
  }
  if (!coupled_counts_.ensure(particle_count_)) {
    return false;
  }
  {
    std::array<gpu::StorageBuf *, 8> buffers{grid_.positions.buf(),
                                             coupled_particles_.buf(),
                                             coupled_grid_.cells.buf(),
                                             coupled_grid_.bucket_offsets.buf(),
                                             coupled_grid_.bucket_counts.buf(),
                                             coupled_grid_.sorted.buf(),
                                             coupled_counts_.buf(),
                                             params_.buf()};
    dispatch_pass(count_shader, buffer_span(buffers), particle_count_);
    pass_count++;
  }
  /* Use the coupled grid's scan scratch, not the fluid grid's: the fluid one holds the total that
   * #ha_xpbd_neighbors_total reads, and overwriting it made the diagnostics report 0 neighbours. */
  const bool debug = std::getenv("XPBD_GPU_DEBUG") != nullptr;
  int coupled_total = 0;
  if (!scan_counts(coupled_grid_,
                   coupled_counts_,
                   coupled_offsets_,
                   debug ? &coupled_total : nullptr,
                   particle_count_,
                   pass_count))
  {
    return false;
  }
  if (debug) {
    printf("XPBD_GPU coupled samples=%d pairs=%d table=%d\n",
           coupled_count_,
           coupled_total,
           coupled_grid_.table_size);
    fflush(stdout);
  }
  const int capacity = std::max(coupled_capacity_, std::max(64, particle_count_ * 48));
  if (capacity != coupled_capacity_) {
    if (!coupled_indices_.ensure(capacity)) {
      return false;
    }
    coupled_capacity_ = capacity;
    std::array<float, FLUID_PARAM_COUNT_TOTAL> params{};
    params_.download(params);
    params[FLUID_PARAM_COUPLED_CAPACITY] = float(capacity);
    params_.upload(params);
  }
  {
    std::array<gpu::StorageBuf *, 9> buffers{grid_.positions.buf(),
                                             coupled_particles_.buf(),
                                             coupled_grid_.cells.buf(),
                                             coupled_grid_.bucket_offsets.buf(),
                                             coupled_grid_.bucket_counts.buf(),
                                             coupled_grid_.sorted.buf(),
                                             coupled_offsets_.buf(),
                                             coupled_indices_.buf(),
                                             params_.buf()};
    dispatch_pass(fill_shader, buffer_span(buffers), particle_count_);
    pass_count++;
  }
  stats_.passes += pass_count - pass_offset;
  return true;
}

bool HaXpbdFluid::build_boundary_neighbors(const int pass_offset)
{
  if (boundary_count_ == 0) {
    return true;
  }
  int pass_count = pass_offset;
  gpu::Shader *count_shader = ha_shader_get_source("xpbd|boundary_count",
                                                   boundary_query_kernel(true),
                                                   as_span(BOUNDARY_COUNT_RESOURCES),
                                                   LOCAL_GROUP_SIZE);
  gpu::Shader *fill_shader = ha_shader_get_source("xpbd|boundary_fill",
                                                  boundary_query_kernel(false),
                                                  as_span(BOUNDARY_FILL_RESOURCES),
                                                  LOCAL_GROUP_SIZE);
  if (count_shader == nullptr || fill_shader == nullptr) {
    return false;
  }
  {
    std::array<gpu::StorageBuf *, 8> buffers{grid_.positions.buf(),
                                             boundary_particles_.buf(),
                                             boundary_grid_.cells.buf(),
                                             boundary_grid_.bucket_offsets.buf(),
                                             boundary_grid_.bucket_counts.buf(),
                                             boundary_grid_.sorted.buf(),
                                             boundary_counts_.buf(),
                                             params_.buf()};
    dispatch_pass(count_shader, buffer_span(buffers), particle_count_);
    pass_count++;
  }
  int total = 0;
  if (!scan_counts(grid_,
                   boundary_counts_,
                   boundary_offsets_,
                   nullptr,
                   particle_count_,
                   pass_count))
  {
    return false;
  }
  const int capacity = std::max(boundary_capacity_, std::max(64, particle_count_ * 48));
  if (capacity != boundary_capacity_) {
    if (!boundary_indices_.ensure(capacity)) {
      return false;
    }
    boundary_capacity_ = capacity;
    std::array<float, FLUID_PARAM_COUNT_TOTAL> params{};
    params_.download(params);
    params[FLUID_PARAM_BOUNDARY_CAPACITY] = float(capacity);
    params_.upload(params);
  }
  {
    std::array<gpu::StorageBuf *, 9> buffers{grid_.positions.buf(),
                                             boundary_particles_.buf(),
                                             boundary_grid_.cells.buf(),
                                             boundary_grid_.bucket_offsets.buf(),
                                             boundary_grid_.bucket_counts.buf(),
                                             boundary_grid_.sorted.buf(),
                                             boundary_offsets_.buf(),
                                             boundary_indices_.buf(),
                                             params_.buf()};
    dispatch_pass(fill_shader, buffer_span(buffers), particle_count_);
    pass_count++;
  }
  stats_.passes += pass_count - pass_offset;
  (void)total;
  return true;
}

/**
 * The phase split (#begin_substep / #iterate / #finish_substep) exists for the solver's whole world
 * loop: that one has to apply the reaction of the fluid pressure on the coupled solids between the
 * iterations, so it cannot hand the whole substep to the device at once. Every phase takes its own
 * device context guard, which is a cheap check when the calling thread already has one.
 */
bool HaXpbdFluid::substep(const int iterations, const ContactHook &contact_hook)
{
  if (!this->begin_substep(contact_hook)) {
    return false;
  }
  for ([[maybe_unused]] const int iteration_i : IndexRange(iterations)) {
    if (!this->iterate()) {
      return false;
    }
  }
  return this->finish_substep();
}

bool HaXpbdFluid::begin_substep(const ContactHook &contact_hook)
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
  const int count = particle_count_;

  /* The velocity update removes the normal component of this push, so it has to start from zero
   * for every substep. */
  coupled_push_.clear_to_zero();
  stats_ = {};
  if (!ha_xpbd_grid_build_device(grid_) || !ha_xpbd_neighbors_build(grid_)) {
    last_error_ = "grid or neighbour pass failed";
    return false;
  }
  stats_.grid_ms = grid_.stats.grid_ms;
  stats_.neighbors_ms = grid_.stats.neighbors_ms;
  stats_.passes = grid_.stats.grid_passes + grid_.stats.neighbor_passes;
  if (!this->build_boundary_neighbors(stats_.passes)) {
    last_error_ = "boundary neighbour pass failed";
    return false;
  }
  if (!this->build_coupled_neighbors(stats_.passes)) {
    last_error_ = "coupled neighbour pass failed";
    return false;
  }
  const double solve_start = time_now_ms();
  gpu::Shader *predict = ha_shader_get_source("xpbd|predict",
                                              std::string(GLSL_PRELUDE) + KERNEL_PREDICT,
                                              as_span(PREDICT_RESOURCES),
                                              LOCAL_GROUP_SIZE);
  if (predict == nullptr) {
    last_error_ = "shader creation failed";
    return false;
  }
  {
    std::array<gpu::StorageBuf *, 6> buffers{grid_.positions.buf(),
                                             prev_positions_.buf(),
                                             velocity_current().buf(),
                                             external_force_.buf(),
                                             props_.buf(),
                                             params_.buf()};
    dispatch_pass(predict, buffer_span(buffers), count);
    stats_.passes++;
  }
  stats_.solve_ms = time_now_ms() - solve_start;

  /* The collider contacts are gathered on the host from the predicted positions, exactly like the
   * CPU implementation does before its constraint iterations. */
  if (contact_hook && !contact_hook()) {
    contact_count_ = 0;
  }
  return true;
}

bool HaXpbdFluid::iterate()
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
  const int count = particle_count_;
  const bool with_boundary = boundary_count_ > 0;
  const bool with_coupled = coupled_count_ > 0;

  Vector<HaShaderResource> step_resources = fluid_step_resources(with_boundary, with_coupled);
  const Span<const HaShaderResource> step_span(step_resources.data(), step_resources.size());
  gpu::Shader *density = ha_shader_get_source(
      with_coupled ? (with_boundary ? "xpbd|density_bc" : "xpbd|density_c") :
                     (with_boundary ? "xpbd|density_b" : "xpbd|density"),
      fluid_kernel(DENSITY_PROLOGUE,
                   DENSITY_LOOP,
                   with_boundary ? DENSITY_BOUNDARY_LOOP : nullptr,
                   DENSITY_EPILOGUE,
                   with_coupled ? DENSITY_COUPLED_LOOP : nullptr),
      step_span,
      LOCAL_GROUP_SIZE);
  gpu::Shader *delta = ha_shader_get_source(
      with_coupled ? (with_boundary ? "xpbd|delta_bc" : "xpbd|delta_c") :
                     (with_boundary ? "xpbd|delta_b" : "xpbd|delta"),
      fluid_kernel(DELTA_PROLOGUE,
                   DELTA_LOOP,
                   with_boundary ? DELTA_BOUNDARY_LOOP : nullptr,
                   DELTA_EPILOGUE,
                   with_coupled ? DELTA_COUPLED_LOOP : nullptr,
                   with_coupled ? DELTA_COUPLED_EPILOGUE : nullptr),
      step_span,
      LOCAL_GROUP_SIZE);
  gpu::Shader *apply = ha_shader_get_source("xpbd|apply",
                                            std::string(GLSL_PRELUDE) + KERNEL_APPLY_CORRECTION,
                                            as_span(APPLY_RESOURCES),
                                            LOCAL_GROUP_SIZE);
  gpu::Shader *contact_position = ha_shader_get_source(
      "xpbd|contact_position",
      std::string(GLSL_PRELUDE) + GLSL_FLUID_HELPERS + KERNEL_CONTACT_POSITION,
      as_span(CONTACT_POSITION_RESOURCES),
      LOCAL_GROUP_SIZE);
  gpu::Shader *coupled_reaction = nullptr;
  if (with_coupled) {
    coupled_reaction = ha_shader_get_source(
        "xpbd|coupled_reaction",
        std::string(GLSL_PRELUDE) + GLSL_FLUID_HELPERS + KERNEL_COUPLED_REACTION,
        as_span(COUPLED_REACTION_RESOURCES),
        LOCAL_GROUP_SIZE);
  }
  if (density == nullptr || delta == nullptr || apply == nullptr ||
      contact_position == nullptr || (with_coupled && coupled_reaction == nullptr))
  {
    last_error_ = "shader creation failed";
    return false;
  }

  const double iteration_start = time_now_ms();
  /* The buffer list has to match the bound shader variant (with or without boundary particles and
   * coupled solids), binding a null buffer to an unused slot crashes the driver. */
  Vector<gpu::StorageBuf *> step_buffers;
  step_buffers.append(grid_.positions.buf());
  step_buffers.append(grid_.neighbor_offsets.buf());
  step_buffers.append(grid_.neighbor_indices.buf());
  step_buffers.append(props_.buf());
  step_buffers.append(particle_state_.buf());
  step_buffers.append(params_.buf());
  step_buffers.append(coupled_push_.buf());
  if (with_boundary) {
    step_buffers.append(boundary_particles_.buf());
    step_buffers.append(boundary_offsets_.buf());
    step_buffers.append(boundary_indices_.buf());
  }
  if (with_coupled) {
    step_buffers.append(coupled_particles_.buf());
    step_buffers.append(coupled_offsets_.buf());
    step_buffers.append(coupled_indices_.buf());
  }
  const Span<gpu::StorageBuf *> step_span_buffers(step_buffers.data(), step_buffers.size());
  dispatch_pass(density, step_span_buffers, count);
  dispatch_pass(delta, step_span_buffers, count);
  {
    std::array<gpu::StorageBuf *, 4> buffers{
        grid_.positions.buf(), particle_state_.buf(), props_.buf(), params_.buf()};
    dispatch_pass(apply, buffer_span(buffers), count);
  }
  if (with_coupled) {
    /* The reaction uses the *corrected* positions, like `fluid_apply_boundary_reaction` on the
     * CPU (which runs after the density iteration applied its correction). */
    std::array<gpu::StorageBuf *, 10> buffers{coupled_particles_.buf(),
                                              coupled_reaction_.buf(),
                                              grid_.positions.buf(),
                                              props_.buf(),
                                              particle_state_.buf(),
                                              params_.buf(),
                                              grid_.cells.buf(),
                                              grid_.bucket_offsets.buf(),
                                              grid_.bucket_counts.buf(),
                                              grid_.sorted.buf()};
    dispatch_pass(coupled_reaction, buffer_span(buffers), coupled_count_);
  }
  if (sdf_active_) {
    gpu::Shader *sdf_contact = ha_shader_get_source(
        "xpbd|sdf_contact",
        std::string(GLSL_PRELUDE) + GLSL_SDF_HELPERS + KERNEL_SDF_CONTACT,
        as_span(SDF_CONTACT_RESOURCES),
        LOCAL_GROUP_SIZE);
    if (sdf_contact == nullptr) {
      last_error_ = "shader creation failed";
      return false;
    }
    std::array<gpu::StorageBuf *, 6> buffers{grid_.positions.buf(),
                                            sdf_data_.buf(),
                                            sdf_params_.buf(),
                                            particle_state_.buf(),
                                            props_.buf(),
                                            params_.buf()};
    dispatch_pass(sdf_contact, buffer_span(buffers), count);
    stats_.passes++;
  }
  if (coupled_sdf_active_) {
    gpu::Shader *coupled_contact = ha_shader_get_source(
        "xpbd|coupled_sdf_contact",
        std::string(GLSL_PRELUDE) + GLSL_SDF_HELPERS + KERNEL_COUPLED_SDF_CONTACT,
        as_span(COUPLED_SDF_CONTACT_RESOURCES),
        LOCAL_GROUP_SIZE);
    if (coupled_contact == nullptr) {
      last_error_ = "shader creation failed";
      return false;
    }
    std::array<gpu::StorageBuf *, 11> buffers{grid_.positions.buf(),
                                              coupled_sdf_data_.buf(),
                                              coupled_sdf_params_.buf(),
                                              props_.buf(),
                                              params_.buf(),
                                              coupled_contact_count_.buf(),
                                              coupled_contact_indices_.buf(),
                                              coupled_contact_push_.buf(),
                                              prev_positions_.buf(),
                                              coupled_push_flags_.buf(),
                                              coupled_crossings_.buf()};
    dispatch_pass(coupled_contact, buffer_span(buffers), count);
    stats_.passes++;
  }
  if (contact_count_ > 0) {
    std::array<gpu::StorageBuf *, 8> buffers{grid_.positions.buf(),
                                             prev_positions_.buf(),
                                             props_.buf(),
                                             contact_vec_.buf(),
                                             contact_scalar_.buf(),
                                             contact_points_.buf(),
                                             contact_lambda_.buf(),
                                             params_.buf()};
    dispatch_pass(contact_position, buffer_span(buffers), contact_count_);
  }
  stats_.passes += with_coupled ? 4 : 3;
  stats_.solve_ms += time_now_ms() - iteration_start;
  return true;
}

bool HaXpbdFluid::finish_substep()
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
  const int count = particle_count_;

  /* --- velocity layer --- */
  const double velocity_start = time_now_ms();
  gpu::Shader *update_velocity =
      ha_shader_get_source("xpbd|update_velocity",
                           std::string(GLSL_PRELUDE) + KERNEL_UPDATE_VELOCITY,
                           as_span(UPDATE_VELOCITY_RESOURCES),
                           LOCAL_GROUP_SIZE);
  gpu::Shader *vorticity = ha_shader_get_source(
      "xpbd|vorticity",
      fluid_kernel(VORTICITY_PROLOGUE, VORTICITY_LOOP, nullptr, VORTICITY_EPILOGUE),
      as_span(VORTICITY_RESOURCES),
      LOCAL_GROUP_SIZE);
  gpu::Shader *vorticity_surface = ha_shader_get_source(
      "xpbd|vorticity_surface",
      fluid_kernel(
          VORTICITY_PROLOGUE_PASS2, VORTICITY_LOOP_PASS2, nullptr, VORTICITY_EPILOGUE_PASS2),
      as_span(VORTICITY_APPLY_RESOURCES),
      LOCAL_GROUP_SIZE);
  gpu::Shader *xsph = ha_shader_get_source(
      "xpbd|xsph_damping",
      fluid_kernel(XSPH_PROLOGUE, XSPH_LOOP, nullptr, XSPH_EPILOGUE),
      as_span(XSPH_RESOURCES),
      LOCAL_GROUP_SIZE);
  if (update_velocity == nullptr || vorticity == nullptr || vorticity_surface == nullptr ||
      xsph == nullptr)
  {
    last_error_ = "shader creation failed";
    return false;
  }

  {
    std::array<gpu::StorageBuf *, 5> buffers{grid_.positions.buf(),
                                             prev_positions_.buf(),
                                             velocity_current().buf(),
                                             params_.buf(),
                                             coupled_push_.buf()};
    dispatch_pass(update_velocity, buffer_span(buffers), count);
    stats_.passes++;
  }
  {
    std::array<gpu::StorageBuf *, 7> buffers{grid_.positions.buf(),
                                             grid_.neighbor_offsets.buf(),
                                             grid_.neighbor_indices.buf(),
                                             props_.buf(),
                                             velocity_current().buf(),
                                             particle_state_.buf(),
                                             params_.buf()};
    dispatch_pass(vorticity, buffer_span(buffers), count);
    stats_.passes++;
  }
  {
    std::array<gpu::StorageBuf *, 7> buffers{grid_.positions.buf(),
                                             grid_.neighbor_offsets.buf(),
                                             grid_.neighbor_indices.buf(),
                                             props_.buf(),
                                             velocity_current().buf(),
                                             particle_state_.buf(),
                                             params_.buf()};
    dispatch_pass(vorticity_surface, buffer_span(buffers), count);
    stats_.passes++;
  }
  {
    /* XSPH reads the velocities of the neighbours, so it cannot write in place: the result goes to
     * the other buffer (with the damping and the no-slip factor already applied) and the two
     * swap. */
    std::array<gpu::StorageBuf *, 8> buffers{grid_.positions.buf(),
                                             grid_.neighbor_offsets.buf(),
                                             grid_.neighbor_indices.buf(),
                                             props_.buf(),
                                             velocity_current().buf(),
                                             velocity_other().buf(),
                                             params_.buf(),
                                             particle_state_.buf()};
    dispatch_pass(xsph, buffer_span(buffers), count);
    velocities_in_tmp_ = !velocities_in_tmp_;
    stats_.passes++;
  }
  if (contact_count_ > 0) {
    /* Same order as the CPU: the collider friction runs after the fluid velocity layer. */
    gpu::Shader *contact_friction = ha_shader_get_source(
        "xpbd|contact_friction",
        std::string(GLSL_PRELUDE) + GLSL_FLUID_HELPERS + KERNEL_CONTACT_FRICTION,
        as_span(CONTACT_FRICTION_RESOURCES),
        LOCAL_GROUP_SIZE);
    if (contact_friction == nullptr) {
      last_error_ = "shader creation failed";
      return false;
    }
    std::array<gpu::StorageBuf *, 8> buffers{velocity_current().buf(),
                                             prev_positions_.buf(),
                                             props_.buf(),
                                             contact_vec_.buf(),
                                             contact_scalar_.buf(),
                                             contact_points_.buf(),
                                             contact_lambda_.buf(),
                                             params_.buf()};
    dispatch_pass(contact_friction, buffer_span(buffers), contact_count_);
    stats_.passes++;
  }
  if (sdf_active_) {
    gpu::Shader *sdf_friction = ha_shader_get_source(
        "xpbd|sdf_friction",
        std::string(GLSL_PRELUDE) + GLSL_SDF_HELPERS + KERNEL_SDF_FRICTION,
        as_span(SDF_FRICTION_RESOURCES),
        LOCAL_GROUP_SIZE);
    if (sdf_friction == nullptr) {
      last_error_ = "shader creation failed";
      return false;
    }
    std::array<gpu::StorageBuf *, 6> buffers{velocity_current().buf(),
                                            sdf_data_.buf(),
                                            sdf_params_.buf(),
                                            grid_.positions.buf(),
                                            params_.buf(),
                                            props_.buf()};
    dispatch_pass(sdf_friction, buffer_span(buffers), count);
    stats_.passes++;
  }
  if (coupled_sdf_active_) {
    /* Runs last so nothing adds the outward velocity back. */
    gpu::Shader *coupled_velocity = ha_shader_get_source(
        "xpbd|coupled_sdf_velocity",
        std::string(GLSL_PRELUDE) + GLSL_SDF_HELPERS + KERNEL_COUPLED_SDF_VELOCITY,
        as_span(COUPLED_SDF_VELOCITY_RESOURCES),
        LOCAL_GROUP_SIZE);
    if (coupled_velocity == nullptr) {
      last_error_ = "shader creation failed";
      return false;
    }
    std::array<gpu::StorageBuf *, 7> buffers{grid_.positions.buf(),
                                             coupled_sdf_data_.buf(),
                                             coupled_sdf_params_.buf(),
                                             velocity_current().buf(),
                                             coupled_push_flags_.buf(),
                                             props_.buf(),
                                             params_.buf()};
    dispatch_pass(coupled_velocity, buffer_span(buffers), count);
    stats_.passes++;
  }
  stats_.velocity_ms = time_now_ms() - velocity_start;

  frame_stats_.grid_ms += stats_.grid_ms;
  frame_stats_.neighbors_ms += stats_.neighbors_ms;
  frame_stats_.solve_ms += stats_.solve_ms;
  frame_stats_.velocity_ms += stats_.velocity_ms;
  frame_stats_.passes += stats_.passes;
  frame_stats_.substeps += 1;
  return true;
}


bool HaXpbdFluid::download(const MutableSpan<float3> r_positions,
                           const MutableSpan<float3> r_velocities)
{
  if (!ready_) {
    return false;
  }
  HaContextGuard guard;
  if (!guard.is_valid()) {
    return false;
  }
  const int64_t vec_floats = int64_t(particle_count_) * 3;
  const int count = std::min<int64_t>(r_positions.size(), particle_count_);
  if (count <= 0) {
    return false;
  }
  Array<float> host_positions(vec_floats);
  Array<float> host_velocities(vec_floats);
  grid_.positions.download(host_positions);
  velocity_current().download(host_velocities);
  const int velocity_count = std::min<int64_t>(r_velocities.size(), particle_count_);
  for (const int i : IndexRange(count)) {
    r_positions[i] = float3(
        host_positions[i * 3 + 0], host_positions[i * 3 + 1], host_positions[i * 3 + 2]);
  }
  for (const int i : IndexRange(velocity_count)) {
    r_velocities[i] = float3(
        host_velocities[i * 3 + 0], host_velocities[i * 3 + 1], host_velocities[i * 3 + 2]);
  }
  return true;
}

/** \} */

}  // namespace blender::geometry::ha
