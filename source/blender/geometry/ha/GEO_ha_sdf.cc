/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 */

#ifdef WITH_OPENVDB

#  include "GEO_ha_sdf.hh"

#  include "BLI_math_base.hh"
#  include "BLI_task.hh"

#  include "BKE_volume_grid.hh"

#  include "GEO_mesh_to_volume.hh"

#  include <openvdb/openvdb.h>
#  include <openvdb/tools/Interpolation.h>

namespace blender::geometry::ha {

namespace {

/** Largest voxel size that fits the budget for a field of this world space size. */
float clamp_voxel_size(const float3 &extent, const float band_width, float voxel_size)
{
  float result = voxel_size;
  for ([[maybe_unused]] const int attempt : IndexRange(8)) {
    const int64_t pad = int64_t(std::ceil(band_width / result)) + 1;
    const int64_t dims_x = int64_t(std::ceil(extent.x / result)) + 2 * pad + 1;
    const int64_t dims_y = int64_t(std::ceil(extent.y / result)) + 2 * pad + 1;
    const int64_t dims_z = int64_t(std::ceil(extent.z / result)) + 2 * pad + 1;
    if (dims_x * dims_y * dims_z <= HA_SDF_MAX_VOXELS) {
      return result;
    }
    result *= 1.3f;
  }
  return result;
}

}  // namespace

float ha_sdf_sample(const HaSdfGrid &grid, const float3 &position)
{
  if (!grid.is_valid()) {
    return grid.spacing;
  }
  const float3 local = (position - grid.origin) / grid.spacing - float3(0.5f + 0.0f);
  const int3 base = int3(int(std::floor(local.x)), int(std::floor(local.y)),
                         int(std::floor(local.z)));
  const float3 fraction = local - float3(base);
  auto value_at = [&](const int x, const int y, const int z) -> float {
    const int cx = std::clamp(x, 0, grid.dims.x - 1);
    const int cy = std::clamp(y, 0, grid.dims.y - 1);
    const int cz = std::clamp(z, 0, grid.dims.z - 1);
    return grid.data[int64_t(cz) * grid.dims.y * grid.dims.x + int64_t(cy) * grid.dims.x + cx];
  };
  const float c000 = value_at(base.x, base.y, base.z);
  const float c100 = value_at(base.x + 1, base.y, base.z);
  const float c010 = value_at(base.x, base.y + 1, base.z);
  const float c110 = value_at(base.x + 1, base.y + 1, base.z);
  const float c001 = value_at(base.x, base.y, base.z + 1);
  const float c101 = value_at(base.x + 1, base.y, base.z + 1);
  const float c011 = value_at(base.x, base.y + 1, base.z + 1);
  const float c111 = value_at(base.x + 1, base.y + 1, base.z + 1);
  const float c00 = math::interpolate(c000, c100, fraction.x);
  const float c10 = math::interpolate(c010, c110, fraction.x);
  const float c01 = math::interpolate(c001, c101, fraction.x);
  const float c11 = math::interpolate(c011, c111, fraction.x);
  const float c0 = math::interpolate(c00, c10, fraction.y);
  const float c1 = math::interpolate(c01, c11, fraction.y);
  return math::interpolate(c0, c1, fraction.z);
}

bool ha_sdf_build(const Span<float3> positions,
                  const Span<int> corner_verts,
                  const Span<int3> corner_tris,
                  float voxel_size,
                  const float band_width,
                  HaSdfGrid &r_grid)
{
  if (positions.is_empty() || corner_tris.is_empty() || voxel_size <= 0.0f || band_width <= 0.0f) {
    return false;
  }

  float3 bbox_min = positions[0];
  float3 bbox_max = positions[0];
  for (const float3 &position : positions) {
    bbox_min = float3(std::min(bbox_min.x, position.x),
                      std::min(bbox_min.y, position.y),
                      std::min(bbox_min.z, position.z));
    bbox_max = float3(std::max(bbox_max.x, position.x),
                      std::max(bbox_max.y, position.y),
                      std::max(bbox_max.z, position.z));
  }
  const float3 extent = bbox_max - bbox_min;
  voxel_size = clamp_voxel_size(extent, band_width, voxel_size);
  const int band_voxels = std::max(3, int(std::ceil(band_width / voxel_size)));

  bke::VolumeGrid<float> volume_grid = geometry::mesh_to_sdf_grid(
      positions, corner_verts, corner_tris, voxel_size, float(band_voxels));
  bke::VolumeTreeAccessToken token;
  const openvdb::FloatGrid &level_set = volume_grid.grid(token);
  openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(level_set);

  r_grid.spacing = voxel_size;
  r_grid.origin = bbox_min - float3(band_voxels * voxel_size);
  r_grid.dims = int3(std::max(1, int(std::ceil(extent.x / voxel_size)) + 2 * band_voxels + 1),
                     std::max(1, int(std::ceil(extent.y / voxel_size)) + 2 * band_voxels + 1),
                     std::max(1, int(std::ceil(extent.z / voxel_size)) + 2 * band_voxels + 1));
  if (r_grid.voxels_num() > HA_SDF_MAX_VOXELS) {
    r_grid.clear();
    return false;
  }
  r_grid.data.reinitialize(r_grid.voxels_num());

  const int3 dims = r_grid.dims;
  const float3 origin = r_grid.origin;
  const float spacing = r_grid.spacing;
  /* Outside of the narrow band the level set is a constant (+band), which is exactly what the
   * sampler returns, so the whole dense box is well defined. */
  threading::parallel_for(IndexRange(dims.z), 1, [&](const IndexRange z_range) {
    for (const int z : z_range) {
      for (const int y : IndexRange(dims.y)) {
        for (const int x : IndexRange(dims.x)) {
          const float3 center = origin + (float3(x, y, z) + 0.5f) * spacing;
          const float distance = sampler.wsSample(
              openvdb::Vec3f(center.x, center.y, center.z));
          r_grid.data[int64_t(z) * dims.y * dims.x + int64_t(y) * dims.x + x] = distance;
        }
      }
    }
  });
  return true;
}

}  // namespace blender::geometry::ha

#endif /* WITH_OPENVDB */
