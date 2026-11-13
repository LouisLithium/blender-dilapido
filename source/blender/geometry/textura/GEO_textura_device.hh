/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Compute backend (device) version of the Textura MPM substep.
 *
 * The CPU solver in `GEO_textura_solver.cc` is the reference implementation. This file starts the
 * port with the case that is both common and easy to verify: a single velocity grid holding fluid
 * particles. Sand/snow plasticity (3x3 SVD per particle and substep), two-phase grids and the
 * continuum surface force are still CPU only and are reported as fallback reasons.
 *
 * All state lives in flat storage buffers while a frame is solved; the host uploads the particle
 * arrays once, runs every substep on the device and downloads the result at the end. The dense grid
 * is resident too, so a substep is five dispatches: clear, particle to grid, grid update, grid to
 * particle, advect.
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
 * Runs `substeps` MPM substeps on the compute backend. The particle arrays are updated in place.
 * Returns false (and leaves the particles untouched) when the device path cannot run; the caller
 * then evaluates the frame on the CPU.
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
                  float dt,
                  int substeps,
                  Stats &r_stats,
                  std::string &r_reason);

}  // namespace blender::geometry::textura
