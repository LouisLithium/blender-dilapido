/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Compute backend (device) version of the Textura MPM substep.
 *
 * The CPU solver in `GEO_textura_solver.cc` is the reference implementation. This file covers fluid,
 * sand and snow particles on one velocity grid (the Drucker-Prager sand return mapping, the snow
 * compaction `F = F_e F_p`, per material wall friction/adhesion) plus the two-phase variant, where
 * fluids and solids each get their own grid and a drag pass exchanges momentum between them. The
 * continuum surface force runs on the fluid grid from a colour field accumulated in particle to
 * grid, exactly like the CPU.
 *
 * All state lives in flat storage buffers while a frame is solved; the host uploads the particle
 * arrays once, runs every substep on the device and downloads the result at the end. The dense grids
 * are resident too, so a single-phase substep is five dispatches (clear, particle to grid, grid
 * update, grid to particle, advect). A two-phase frame runs `max(fluid_steps, solid_steps)` outer
 * steps, each phase clearing, scattering, updating and gathering only when its own CFL schedule says
 * so, with the drag pass whenever both advance and advection every outer step.
 *
 * `deform[]` holds eighteen floats per particle (`F` at 0..8 and `F_p` at 9..17) and the mass buffer
 * holds two ints per cell (mass and the colour field), which keeps every kernel within the eight
 * storage buffer slots the GL backend guarantees.
 *
 * \note Particle to grid accumulates with `atomicAdd` on **fixed point integers** rather than on
 * floats: `GL_EXT_shader_atomic_float` is not available on the supported backends, and integer
 * atomics make the accumulation order-independent, so the same frame gives the same grid numbers on
 * every run (verified by the `textura_gpu_smoke` parity runs: < 1e-3 m in free fall after 160
 * substeps).
 */

#pragma once

#include <string>

#include "GEO_ha_sdf.hh"
#include "GEO_textura_solver.hh"

namespace blender::geometry::textura {

/**
 * True when #device_solve can handle this frame. `r_reason` describes the first limitation, which
 * the node shows when the zone asked for the device (Strict).
 */
bool device_supported(const ParticleData &particles,
                      const Settings &settings,
                      std::string &r_reason);

/**
 * Runs the MPM substeps of one frame on the compute backend. The particle arrays are updated in
 * place. Returns false (and leaves the particles untouched) when the device path cannot run; the
 * caller then evaluates the frame on the CPU.
 *
 * In a two-phase frame the two phases step at their own cadence: the outer loop runs
 * `max(fluid_substeps, solid_substeps)` times and each phase advances when its integer schedule says
 * so, so the water is not forced through the sand's much larger substep count. The drag pass runs
 * when both phases step, at the slower phase's `dt`.
 *
 * The device path is not bit deterministic with respect to the CPU: the reference accumulates in
 * grid order, the device sums per cell with integer atomics (whose integer rounding differs). The
 * CPU result stays the reference (the regression test compares against it); repeated *device* runs
 * are bit identical, because the fixed point sums do not depend on the addition order.
 */
bool device_solve(ParticleData &particles,
                  Span<const Collider> colliders,
                  const Settings &settings,
                  const GridData &grid,
                  const ha::HaSdfGrid *sdf,
                  float frame_dt,
                  int fluid_substeps,
                  int solid_substeps,
                  Stats &r_stats,
                  std::string &r_reason);

}  // namespace blender::geometry::textura
