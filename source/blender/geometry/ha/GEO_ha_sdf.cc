/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 */

#ifdef WITH_OPENVDB

#  include "GEO_ha_sdf.hh"

#  include "BLI_math_base.hh"
#  include "BLI_math_geom.hh"
#  include "BLI_math_vector.hh"
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

/**
 * Clips one triangle against the six planes of \a region and appends the visible polygon as a
 * triangle fan. The clipped mesh only has to cover the part of the collider the fluid can reach;
 * a 20 m ground plane then costs a few centimetres of field instead of the whole plane.
 */
void clip_triangle_to_region(const float3 &v0,
                             const float3 &v1,
                             const float3 &v2,
                             const float3 &region_min,
                             const float3 &region_max,
                             Vector<float3> &r_positions,
                             Vector<int> &r_corner_verts,
                             Vector<int3> &r_corner_tris)
{
  Vector<float3> poly{v0, v1, v2};
  auto clip = [&](const float3 &normal, const float plane_distance) {
    Vector<float3> out;
    out.reserve(poly.size() + 1);
    for (const int i : poly.index_range()) {
      const float3 &a = poly[i];
      const float3 &b = poly[(i + 1) % poly.size()];
      const float da = math::dot(normal, a) - plane_distance;
      const float db = math::dot(normal, b) - plane_distance;
      if (da >= 0.0f) {
        out.append(a);
      }
      if ((da > 0.0f && db < 0.0f) || (da < 0.0f && db > 0.0f)) {
        const float t = da / (da - db);
        out.append(a + (b - a) * t);
      }
    }
    poly = std::move(out);
  };
  clip(float3(1.0f, 0.0f, 0.0f), region_min.x);
  clip(float3(-1.0f, 0.0f, 0.0f), -region_max.x);
  clip(float3(0.0f, 1.0f, 0.0f), region_min.y);
  clip(float3(0.0f, -1.0f, 0.0f), -region_max.y);
  clip(float3(0.0f, 0.0f, 1.0f), region_min.z);
  clip(float3(0.0f, 0.0f, -1.0f), -region_max.z);
  if (poly.size() < 3) {
    return;
  }
  const int base = int(r_positions.size());
  for (const int i : poly.index_range()) {
    r_positions.append(poly[i]);
    r_corner_verts.append(base + i);
  }
  for (int i = 1; i + 1 < poly.size(); i++) {
    r_corner_tris.append(int3(base, base + i, base + i + 1));
  }
}

/**
 * Signed distance from the triangle soup, using the normal of the closest triangle for the sign.
 *
 * OpenVDB builds the level set from the winding of a *closed* mesh. For an open mesh (a single
 * plane, the common floor) it can produce a field that is positive everywhere, i.e. with no
 * "inside" at all, and the device contact never triggers (the fluid falls straight through). This
 * fallback gives a one-sided field with the correct sign, which is all the contact needs.
 */
void build_one_sided_field(const Span<float3> positions,
                           const Span<int3> corner_tris,
                           const float3 &origin,
                           const float spacing,
                           const int3 &dims,
                           const float band_width,
                           const float splat_band,
                           const MutableSpan<float> r_data)
{
  /* Splat each triangle into the voxels of its own bounding box (plus `splat_band`) instead of
   * testing every triangle for every voxel. The value is the signed distance clamped to `+/-band`,
   * so voxels outside the band still get the sign of the closest triangle. That matters: a particle
   * that crossed the sheet lands beyond the band, and the crossing check compares the sign before
   * and after. A constant `+band` fill there would make the far side look like the near side and
   * the particle would never be caught. */
  r_data.fill(band_width);
  const int64_t voxel_count = int64_t(dims.x) * dims.y * dims.z;
  Vector<float> best_dist_sq(voxel_count, 1e30f);
  const float splat_band_sq = splat_band * splat_band;

  for (const int3 &tri : corner_tris) {
    const float3 &v0 = positions[tri.x];
    const float3 &v1 = positions[tri.y];
    const float3 &v2 = positions[tri.z];
    const float3 normal = math::normalize(math::cross(v1 - v0, v2 - v0));
    if (!(std::isfinite(normal.x) && std::isfinite(normal.y) && std::isfinite(normal.z))) {
      continue;
    }
    const float3 tri_min = math::min(math::min(v0, v1), v2) - float3(splat_band);
    const float3 tri_max = math::max(math::max(v0, v1), v2) + float3(splat_band);
    const int3 lo(std::max(0, int(std::floor((tri_min.x - origin.x) / spacing - 0.5f))),
                  std::max(0, int(std::floor((tri_min.y - origin.y) / spacing - 0.5f))),
                  std::max(0, int(std::floor((tri_min.z - origin.z) / spacing - 0.5f))));
    const int3 hi(std::min(dims.x - 1, int(std::ceil((tri_max.x - origin.x) / spacing - 0.5f))),
                  std::min(dims.y - 1, int(std::ceil((tri_max.y - origin.y) / spacing - 0.5f))),
                  std::min(dims.z - 1, int(std::ceil((tri_max.z - origin.z) / spacing - 0.5f))));
    const float a[3] = {v0.x, v0.y, v0.z};
    const float b[3] = {v1.x, v1.y, v1.z};
    const float c[3] = {v2.x, v2.y, v2.z};
    for (int z = lo.z; z <= hi.z; z++) {
      for (int y = lo.y; y <= hi.y; y++) {
        for (int x = lo.x; x <= hi.x; x++) {
          const float3 center = origin + (float3(x, y, z) + 0.5f) * spacing;
          const float point[3] = {center.x, center.y, center.z};
          float closest[3];
          closest_on_tri_to_point_v3(closest, point, a, b, c);
          const float3 closest_v(closest[0], closest[1], closest[2]);
          const float dist_sq = math::length_squared(center - closest_v);
          if (dist_sq > splat_band_sq) {
            continue;
          }
          const int64_t index = (int64_t(z) * dims.y + int64_t(y)) * dims.x + x;
          if (dist_sq < best_dist_sq[index]) {
            best_dist_sq[index] = dist_sq;
            r_data[index] = std::clamp(
                math::dot(center - closest_v, normal), -band_width, band_width);
          }
        }
      }
    }
  }
}

bool ha_sdf_build(const Span<float3> positions,
                  const Span<int> corner_verts,
                  const Span<int3> corner_tris,
                  float voxel_size,
                  const float band_width,
                  const float3 &region_min,
                  const float3 &region_max,
                  const float3 &reference_point,
                  const bool one_sided,
                  const bool flip_to_reference,
                  HaSdfGrid &r_grid)
{
  if (positions.is_empty() || corner_tris.is_empty() || voxel_size <= 0.0f || band_width <= 0.0f) {
    return false;
  }
  const float3 rmin = math::min(region_min, region_max);
  const float3 rmax = math::max(region_min, region_max);
  const float3 extent = math::max(rmax - rmin, float3(1e-6f));

  Vector<float3> clipped_positions;
  Vector<int> clipped_corner_verts;
  Vector<int3> clipped_corner_tris;
  for (const int3 &tri : corner_tris) {
    clip_triangle_to_region(positions[corner_verts[tri.x]],
                            positions[corner_verts[tri.y]],
                            positions[corner_verts[tri.z]],
                            rmin,
                            rmax,
                            clipped_positions,
                            clipped_corner_verts,
                            clipped_corner_tris);
  }
  if (clipped_corner_tris.is_empty()) {
    r_grid.clear();
    return false;
  }

  voxel_size = clamp_voxel_size(extent, band_width, voxel_size);
  const int band_voxels = std::max(3, int(std::ceil(band_width / voxel_size)));

  r_grid.spacing = voxel_size;
  r_grid.origin = rmin - float3(band_voxels * voxel_size);
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

  /* The grid extends one band beyond the requested region, so splat that far to give every voxel a
   * sign (the value itself stays clamped to the band). */
  const float splat_band = band_width + float(band_voxels) * voxel_size;
  if (one_sided) {
    /* A sheet has fluid on both sides, so there is no meaningful "outside": keep the sign of the
     * triangle normals and never flip it. The contact kernel pushes every particle away from the
     * surface on the side it is already on. This is also much cheaper than a level set (no OpenVDB
     * bake), which matters because the field is rebuilt from the moving cloth every frame. */
    build_one_sided_field(clipped_positions.as_span(),
                          clipped_corner_tris.as_span(),
                          origin,
                          spacing,
                          dims,
                          band_width,
                          splat_band,
                          r_grid.data.as_mutable_span());
    return true;
  }

  bke::VolumeGrid<float> volume_grid = geometry::mesh_to_sdf_grid(clipped_positions.as_span(),
                                                                  clipped_corner_verts.as_span(),
                                                                  clipped_corner_tris.as_span(),
                                                                  voxel_size,
                                                                  float(band_voxels));
  bke::VolumeTreeAccessToken token;
  const openvdb::FloatGrid &level_set = volume_grid.grid(token);
  openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(level_set);
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
  /* An open mesh can come out of OpenVDB with no negative side at all. Rebuild it one-sided from
   * the triangle normals in that case (the fluid is then pushed away from the surface). */
  bool has_negative = false;
  for (const float value : r_grid.data) {
    if (value < 0.0f) {
      has_negative = true;
      break;
    }
  }
  if (!has_negative) {
    build_one_sided_field(clipped_positions.as_span(),
                          clipped_corner_tris.as_span(),
                          r_grid.origin,
                          r_grid.spacing,
                          r_grid.dims,
                          band_width,
                          splat_band,
                          r_grid.data.as_mutable_span());
  }
  /* The sign of a level set built from an *open* mesh (a single plane, the common floor) is
   * ambiguous: OpenVDB picks a side from the winding, which can put the "outside" below the floor.
   * The fluid is always outside the solid collider, so if the field says the reference point is
   * inside, flip the whole grid. That flips the gradient the contact uses as well. This is skipped
   * for a closed mesh with corrected winding: the reference point may legitimately be inside a
   * hollow collider, and flipping there would turn the solid inside out. */
  if (flip_to_reference && ha_sdf_sample(r_grid, reference_point) < 0.0f) {
    for (float &value : r_grid.data) {
      value = -value;
    }
  }
  return true;
}

}  // namespace blender::geometry::ha

#endif /* WITH_OPENVDB */
