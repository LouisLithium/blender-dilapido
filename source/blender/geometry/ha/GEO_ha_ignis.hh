/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Device implementation of the Ignis smoke solver (`GEO_ha_ignis.cc`).
 *
 * Ignis is Eulerian: the state is a set of voxel fields, so unlike the XPBD fluid (particles with
 * a spatial hash) the natural device representation is a **dense box of voxels**. The CPU solver
 * keeps the fields in sparse OpenVDB trees, which is what makes it cheap when only a small part of
 * the volume is active, but a tree walk is the opposite of what a GPU wants. The device path
 * therefore trades the sparse storage for a dense box during the frame:
 *
 *   CPU: volume -> dense box, emitters rasterized into `source`, collider shell into `solid`
 *   GPU: all substeps on the dense fields (advection, forces, projection, ...)
 *   CPU: dense box -> volume
 *
 * The box is the active smoke region grown by the room the smoke can reach in this frame plus the
 * velocity band radius, so it contains every voxel the CPU solver would visit. Everything that is
 * still derived from a mesh or from another volume (emitter sampling, the collider shell) happens
 * on the CPU once per frame; the work that scales with the number of voxels happens on the device.
 *
 * The kernels mirror `GEO_ignis_solver.cc` operation by operation (same Boussinesq buoyancy, same
 * vorticity confinement, same MacCormack/BFECC advection with the openvdb limiters, the same 7
 * point Jacobi-CG pressure solve with an optional coarse grid initial guess, the same extrapolation
 * of the velocity band). The values are compared against the CPU solver in `doc/tests/ignis_ha_smoke.py`.
 */

#pragma once

#include <cstdint>
#include <string>

#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"

#include "GEO_ignis_solver.hh"

namespace blender::geometry::ha {

/** Bits of the per voxel `mask` of an #IgnisDeviceFrame. */
enum IgnisMask : int32_t {
  IGNIS_MASK_SMOKE = 1 << 0,
  IGNIS_MASK_DOMAIN = 1 << 1,
  IGNIS_MASK_UNKNOWN = 1 << 2,
  IGNIS_MASK_SOLID = 1 << 3,
  IGNIS_MASK_FILLED = 1 << 4,
  IGNIS_MASK_KEEP = 1 << 5,
};

/**
 * One frame of the Ignis solver in a dense box, prepared by the caller.
 *
 * All spans have one entry per voxel of the box (`dims.x * dims.y * dims.z`), with the exception
 * of `velocity` (three) and `source` (seven: density, temperature, fuel, velocity xyz and the
 * velocity blend weight). `scalars` holds density, temperature, fuel and flame in that order.
 * Values of the previous frame are read from the volume before the call and written back
 * afterwards.
 */
struct IgnisDeviceFrame {
  /** Index space origin of the box: the world position of the first voxel is `index_min * voxel`.
   */
  int3 index_min = int3(0);
  int3 dims = int3(0);
  float voxel_size = 0.05f;

  float delta_time = 0.0f;
  int substeps = 1;
  /** Radius in voxels of the velocity band / pressure domain (`max(3, domain_padding)`). */
  int solver_radius = 6;
  /** Number of extrapolation layers for the velocity band. */
  int extrapolate = 4;

  ignis::Settings settings;

  MutableSpan<float> scalars;
  MutableSpan<float> velocity;
  MutableSpan<float> pressure;
  MutableSpan<int32_t> mask;
  /**
   * The solid cells of the last pose (same size as #mask, zero when there are no colliders). The
   * collider id is stored in bits 8..15, so the device velocity pass knows which collider pushes.
   * When empty, #mask is used.
   */
  MutableSpan<int32_t> current_mask;
  /** Voxels that may be removed (interiors, plus the shell of colliders that do not move). */
  MutableSpan<int32_t> carve_mask;
  /** Average velocity of each collider, three floats per entry (world units per second). */
  Span<float> collider_velocities;
  MutableSpan<float> source;
};

struct IgnisDeviceResult {
  bool ran = false;
  /** Why the device path was not used, empty when it ran. */
  std::string reason;
  int passes = 0;
  int active_voxels = 0;
  int pressure_iterations = 0;
  /** Iteration budget that was available to the pressure solve (adaptive or fixed). */
  int pressure_iteration_budget = 0;
  float residual = 0.0f;
  float divergence_before = 0.0f;
  float divergence_after = 0.0f;
  int solid_voxels = 0;
  int domain_removed = 0;
  bool cfl_limited = false;
  double milliseconds = 0.0;
};

/** True when the GPU backend can be used at all (a context may still be missing per evaluation).
 */
bool ignis_device_available();

/**
 * Runs the whole simulation of one frame (all substeps) on the compute backend.
 *
 * The arrays in \a frame are read and written in place. Returns false and sets `result.reason`
 * when the device path cannot run (no context, shader creation failed, box too large, ...); the
 * caller then uses the CPU solver, so a failure is never silent.
 */
bool ignis_device_solve(IgnisDeviceFrame &frame, IgnisDeviceResult &r_result);

}  // namespace blender::geometry::ha
