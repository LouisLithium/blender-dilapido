/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "eevee_defines.hh"
#include "gpu_shader_utildefines_lib.glsl"

namespace eevee {

/**
 * World space volume grid sampling helpers.
 *
 * The grid stores two view independent 3D textures covering the bounds of all volume objects:
 * - `scatter_tx`: the in-scattered radiance *per unit length* (already includes the light
 *   intensity, the phase function and the emission term).
 * - `extinct_tx`: the extinction coefficient (per unit length).
 *
 * Integrating along a ray gives the radiance scattered towards the ray origin as well as the
 * transmittance of the participating media. The same data is used for volumetric GI (along the
 * GI rays) and for volumetric shadows (along the shadow rays).
 *
 * NOTE: The BSL C++ stubs do not support `out` / `inout` parameters, so the results are returned
 * by value.
 */

struct VolumeGridRayRange {
  bool hit;
  float t_near;
  float t_far;
};

struct VolumeGridIntegration {
  /* Radiance of the media scattered towards the ray origin. */
  float3 radiance;
  /* Transmittance of the media along the ray. */
  float3 transmittance;
};

/* Ray / axis aligned box intersection, clamped to [0, t_max]. */
VolumeGridRayRange volume_grid_ray_bounds(float3 bounds_min,
                                          float3 bounds_size,
                                          float3 origin,
                                          float3 direction,
                                          float t_max)
{
  VolumeGridRayRange range;
  /* NOTE: Do not use `safe_rcp` here: a zero component would collapse the slab to [0, 0] and
   * reject every axis aligned ray. `1/0 = inf` is the correct behavior for an AABB test. */
  const float3 inv_d = 1.0f / direction;
  const float3 t_a = (bounds_min - origin) * inv_d;
  const float3 t_b = (bounds_min + bounds_size - origin) * inv_d;
  const float3 t_small = min(t_a, t_b);
  const float3 t_big = max(t_a, t_b);
  range.t_near = max(max(t_small.x, t_small.y), max(t_small.z, 0.0f));
  range.t_far = min(min(t_big.x, t_big.y), min(t_big.z, t_max));
  range.hit = range.t_far > range.t_near;
  return range;
}

/**
 * Integrate the participating media along a ray between `origin` and `t_max`.
 * The result contains the in-scattered radiance and the transmittance of the media, so that the
 * caller can compose it with the radiance it already gathered (e.g. the surface radiance at the
 * ray hit) using `radiance * result.transmittance + result.radiance`.
 */
VolumeGridIntegration volume_grid_integrate(sampler3D scatter_tx,
                                            sampler3D extinct_tx,
                                            float3 bounds_min,
                                            float3 bounds_size,
                                            float3 origin,
                                            float3 direction,
                                            float t_max,
                                            float jitter,
                                            int steps)
{
  VolumeGridIntegration result;
  result.radiance = float3(0.0f);
  result.transmittance = float3(1.0f);

  const VolumeGridRayRange range = volume_grid_ray_bounds(
      bounds_min, bounds_size, origin, direction, t_max);
  if (!range.hit) {
    return result;
  }

  const float dt = (range.t_far - range.t_near) / float(steps);
  const float3 inv_size = 1.0f / bounds_size;
  /* Jitter the start of the integration to hide the banding of the fixed step count. */
  float t = range.t_near + jitter * dt;

  for (int i = 0; i < steps; i++) {
    const float3 X = origin + direction * t;
    const float3 uvw = (X - bounds_min) * inv_size;
    const float3 sigma_s = texture(scatter_tx, uvw).rgb;
    const float3 sigma_t = texture(extinct_tx, uvw).rgb;

    /* Analytic integration of a constant radiance segment:
     * `∫ T(s) σ_s L ds = T * σ_s L / σ_t * (1 - exp(-σ_t dt))`.
     * Falls back to the emissive case `σ_s L * dt` when there is no extinction. */
    const float3 step_T = exp(-sigma_t * dt);
    float3 step_opacity = (1.0f - step_T) / max(sigma_t, float3(1e-8f));
    step_opacity = mix(step_opacity, float3(dt), equal(sigma_t, float3(0.0f)));

    result.radiance += result.transmittance * sigma_s * step_opacity;
    result.transmittance *= step_T;

    t += dt;
  }
  return result;
}

}  // namespace eevee
