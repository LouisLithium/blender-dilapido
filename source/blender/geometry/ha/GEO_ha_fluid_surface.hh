/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Device side of `Fluid Surface`: the particle to field transfer and the field relaxation.
 *
 * The isosurface extraction stays on the CPU (`openvdb::tools::volumeToMesh`); the two phases that
 * dominate the node (the splat of every particle into its kernel support, and the relaxation passes
 * over the active voxels) run on the compute backend. The field comes back as a dense box, which the
 * node turns into an `openvdb::FloatGrid` for the mesher, so the result is the same surface within
 * the fixed point precision of the splat.
 */

#pragma once

#include <string>

#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_vector.hh"

namespace blender::geometry::ha {

/** Dense scalar field produced on the device. `values` is `dims.x * dims.y * dims.z`, x fastest. */
struct FluidSurfaceField {
  int3 dims = int3(0);
  float3 origin = float3(0.0f);
  float voxel_size = 0.0f;
  Vector<float> values;
};

struct FluidSurfaceSettings {
  /** Resolution of the field. 0 uses half of the largest kernel support. */
  float voxel_size = 0.0f;
  /** Kernel support in particle radii (density mode) or half particle sizes (MPM mode). */
  float radius_scale = 2.0f;
  /** MPM mode: splat the material fraction with the B-spline instead of the poly6 density. */
  bool mpm_mode = false;
  /** Density mode normaliser (the lattice kernel sum), unused by the MPM mode. */
  float normalizer = 1.0f;
  int smoothing = 2;
  float edge_preserving = 0.0f;
  float value_scale = 0.5f;
  /** Hard limit of the dense box; above it the node falls back to the CPU. */
  int64_t max_cells = 8388608;
};

/**
 * Builds the fluid surface field on the device.
 *
 * \param positions: particle positions (3 floats each).
 * \param radii: particle radii (density mode).
 * \param volumes: particle volumes (both modes).
 * \param r_field: the dense field, on success.
 * \param r_reason: why the device path was not used, on failure.
 * \return true when the field was built on the device.
 */
bool ha_fluid_surface_field(Span<const float3> positions,
                            Span<const float> radii,
                            Span<const float> volumes,
                            const FluidSurfaceSettings &settings,
                            FluidSurfaceField &r_field,
                            std::string &r_reason);

}  // namespace blender::geometry::ha
