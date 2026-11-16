/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Device side of the white water solver: the neighbour coupled forces of the foam/spray/bubbles.
 *
 * The white water spends most of its time in the neighbour loop of the force pass (cohesion,
 * separation and cluster damping). The solver builds its neighbour list (CSR) on the CPU anyway --
 * the union-find cluster pass reads it -- so the device path uploads that list and runs only the
 * force loop, which is the part that scales with the number of neighbours. The candidate order is
 * the CPU order, so the result is reproducible: two device runs of the same input are bit identical.
 */

#pragma once

#include <string>

#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

namespace blender::geometry::ha {

struct WhiteWaterForceSettings {
  float cohesion = 0.0f;
  float cohesion_radius = 0.0f;
  float separation = 0.0f;
  float separation_radius = 0.0f;
  float cluster_damping = 0.0f;
  bool same_type_only = true;
  float delta_time = 0.0f;
};

/**
 * Applies the neighbour coupled forces on the device and adds them to \a r_velocities.
 *
 * \param positions: the particle positions of this frame (never modified).
 * \param r_velocities: read for the relative velocity damping, updated in place.
 * \param types: particle type per element (`same_type_only` filters on it).
 * \param offsets: CSR offsets of the neighbour list, `count + 1` elements.
 * \param indices: CSR neighbour indices.
 * \param settings: the force parameters, already sanitised by the node.
 * \param r_reason: why the device path was not used, on failure.
 * \return true when the forces were applied on the device.
 */
bool ha_white_water_forces(Span<const float3> positions,
                           MutableSpan<float3> r_velocities,
                           Span<const int> types,
                           Span<const int> offsets,
                           Span<const int> indices,
                           const WhiteWaterForceSettings &settings,
                           std::string &r_reason);

}  // namespace blender::geometry::ha
