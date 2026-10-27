/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Turning geometry into the grid data the smoke solver consumes: the "pyro source" and "rasterize
 * volume" of Houdini in one step.
 *
 * The emitter is rasterized at its own resolution (independent of the solver), so a dense emitter
 * does not force a dense simulation and a coarse one does not blur the solver. The result is a
 * `Volume` with the same grid names the solver uses, which the solver samples at its own voxel
 * centers.
 */

#pragma once

#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

namespace blender {

struct Volume;

namespace geometry::ignis {

/** How the geometry becomes a field. */
enum class SourceMode : int {
  Points = 0,
  Surface = 1,
  Solid = 2,
};

struct SourceSettings {
  float voxel_size = 0.05f;
  /** Kernel radius around a point, and the band thickness of a surface. */
  float radius = 0.05f;
  SourceMode mode = SourceMode::Points;
  /** Distance the band of a surface is grown into the air. */
  float band_width = 0.05f;
  /** Multiplier for all field values, i.e. the strength of the emitter. */
  float scale = 1.0f;
};

/** One evaluated point of the emitter. */
struct SourceSample {
  float3 position;
  float density;
  float temperature;
  float fuel;
  float3 velocity;
};

/** Averages of the samples, used to fill the inside of a `Solid` source. */
struct SourceValues {
  float density = 0.0f;
  float temperature = 0.0f;
  float fuel = 0.0f;
  float3 velocity = float3(0.0f);
};

/**
 * Rasterizes the samples into a new `Volume` (density, temperature, fuel and velocity grids).
 *
 * \param samples: one entry per point of the emitter, in world space.
 * \param corner_tris: the triangles of the mesh, for the `Surface` and `Solid` modes.
 * \param corner_verts: the vertices of those triangles, indexing into \a samples.
 * \param mean_values: what the inside of a solid source is filled with.
 */
Volume *rasterize_source(Span<const SourceSample> samples,
                         Span<const int3> corner_tris,
                         Span<const int> corner_verts,
                         const SourceValues &mean_values,
                         const SourceSettings &settings);

}  // namespace geometry::ignis
}  // namespace blender
