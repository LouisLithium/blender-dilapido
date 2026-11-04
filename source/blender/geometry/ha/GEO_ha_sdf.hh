/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Dense signed distance fields of colliders for the hardware accelerated (device) XPBD solver.
 *
 * A collider is baked into a narrow band level set once (per frame, or cached while it does not
 * move) and downloaded to the device as one flat array. Contacts then cost one trilinear sample per
 * particle per iteration: a BVH traversal is not GPU friendly, and the CPU gather it needs forces a
 * full pipeline readback every substep (which was the largest cost of the device fluid).
 */

#pragma once

#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_vector.hh"

namespace blender::geometry::ha {

/** Budget for one dense grid (8 M voxels = 32 MB), so a huge collider cannot eat the memory. */
inline constexpr int64_t HA_SDF_MAX_VOXELS = 8 << 20;

/**
 * World space signed distance of a collider on a dense regular lattice.
 *
 * Voxel `(i, j, k)` is centered at `origin + (float3(i, j, k) + 0.5f) * spacing`, which is the
 * mapping the device kernel mirrors.
 */
struct HaSdfGrid {
  int3 dims = int3(0);
  /** World position of the *corner* of voxel `(0, 0, 0)`, i.e. its center is `origin + 0.5 * s`. */
  float3 origin = float3(0.0f);
  float spacing = 0.0f;
  /** `dims.x * dims.y * dims.z` distances in world units, negative inside the collider. */
  Vector<float> data;

  int64_t voxels_num() const
  {
    return int64_t(dims.x) * int64_t(dims.y) * int64_t(dims.z);
  }

  bool is_valid() const
  {
    return spacing > 0.0f && dims.x > 0 && dims.y > 0 && dims.z > 0 &&
           data.size() == voxels_num();
  }

  void clear()
  {
    dims = int3(0);
    origin = float3(0.0f);
    spacing = 0.0f;
    data.clear();
  }
};

/**
 * Bakes the narrow band level set of a world space triangle mesh into a dense grid.
 *
 * \param voxel_size: Wanted resolution. It is increased automatically when the result would not fit
 *   into #HA_SDF_MAX_VOXELS.
 * \param band_width: Half width of the narrow band in world units. It has to be at least as large as
 *   the movement of one particle during one substep (plus the particle radius), otherwise a fast
 *   particle can cross the whole band of a thin collider and never see a contact.
 * \param region_min/region_max: World space box the field is built for. Only the triangles that
 *   intersect the box are clipped and rasterized, and the dense lattice covers the box. A large
 *   collider (a 20 m ground plane) therefore costs only the part the fluid can reach.
 * \param reference_point: A point known to be outside the solid collider (the fluid center). The
 *   level set sign of an open mesh (a single plane) is ambiguous, so the field is flipped when the
 *   distance at this point is negative.
 * \param one_sided: Builds the field from the triangle normals instead of OpenVDB and keeps that
 *   sign. A sheet (cloth, soft body surface) has fluid on *both* sides, so a single global sign is
 *   meaningless and flipping it (as #reference_point does) turns the water into a projectile the
 *   moment the fluid center crosses the sheet. The contact kernel pushes every particle away from
 *   the surface on the side it is already on, which only needs the sign to be consistent with the
 *   triangle winding.
 * \param r_grid: Filled on success. Callers usually cache it and only rebuild when the geometry
 *   changes (see the geometry hash in the solver).
 */
bool ha_sdf_build(Span<float3> positions,
                  Span<int> corner_verts,
                  Span<int3> corner_tris,
                  float voxel_size,
                  float band_width,
                  const float3 &region_min,
                  const float3 &region_max,
                  const float3 &reference_point,
                  bool one_sided,
                  HaSdfGrid &r_grid);

/** Interpolates the field on the host, used by the debug output and the tests. */
float ha_sdf_sample(const HaSdfGrid &grid, const float3 &position);

}  // namespace blender::geometry::ha
