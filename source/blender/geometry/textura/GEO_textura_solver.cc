/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * MLS-MPM (Hu et al. 2018) for fluids and sand.
 *
 * A substep is the usual five pass MPM step:
 *
 *  1. particle to grid: scatter mass, momentum and the affine stress term (quadratic B-spline),
 *  2. grid update: gravity, damping, domain walls and collider boundary conditions,
 *  3. grid to particle: gather velocity and the APIC velocity gradient,
 *  4. material update: update `F`, return map sand with Drucker-Prager plasticity,
 *  5. advection: move the particles, apply collider contact and clamp to the domain.
 *
 * All passes read and write flat arrays. The particle to grid pass is serial because a scattering
 * write would race; apart from that everything is independent per particle or per node. A device
 * backend can keep the exact same arrays and replace the loop with a dispatch (P2G would use
 * atomics or a sorted bucket list, the way the XPBD device grid works).
 */

#include "GEO_textura_solver.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

#include "BLI_math_matrix.hh"
#include "BLI_math_vector.hh"
#include "BLI_task.hh"

#include "BKE_mesh.hh"

#include "GEO_ha_sdf.hh"

namespace blender::geometry::textura {

namespace {

constexpr float EPSILON = 1e-12f;

/** Fixed number of particle slices for the parallel particle to grid pass. */
constexpr int P2G_SLICE_COUNT = 8;
/** Upper bound for the slice scratch grids; above this the scattering falls back to one pass. */
constexpr int64_t P2G_SLICE_BUDGET = 64ll << 20;

inline double elapsed_ms(const std::chrono::steady_clock::time_point &start)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

/* -------------------------------------------------------------------- */
/** \name Small 3x3 helpers
 * \{ */

inline float determinant3(const float m[3][3])
{
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
         m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

/** Row-major variant for the nine float storage the particles use. */
inline float determinant3_data(const float *m)
{
  return m[0] * (m[4] * m[8] - m[5] * m[7]) -
         m[1] * (m[3] * m[8] - m[5] * m[6]) +
         m[2] * (m[3] * m[7] - m[4] * m[6]);
}

inline float3 multiply3(const float m[3][3], const float3 &v)
{
  return float3(m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z);
}

inline void multiply3(const float a[3][3], const float b[3][3], float r[3][3])
{
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      r[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
    }
  }
}

inline void transpose3(const float m[3][3], float r[3][3])
{
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      r[row][col] = m[col][row];
    }
  }
}

inline void identity3(float r[3][3])
{
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      r[row][col] = row == col ? 1.0f : 0.0f;
    }
  }
}

inline bool invert3(const float m[3][3], float r[3][3])
{
  const float det = determinant3(m);
  if (!std::isfinite(det) || std::fabs(det) < 1e-12f) {
    return false;
  }
  const float inv_det = 1.0f / det;
  r[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
  r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
  r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
  r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
  r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
  r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
  r[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
  r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
  r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
  return true;
}

/**
 * Rotation part of `F` by Newton iteration `R <- (R + R^-T) / 2`.
 *
 * The deformation gradient of the solver is always close to a rotation (the plastic step removes
 * shear), so a handful of iterations is enough and much cheaper than an SVD. A degenerate `F`
 * falls back to identity instead of producing NaNs.
 */
inline void polar_rotation(const float f[3][3], float r[3][3])
{
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      r[row][col] = f[row][col];
    }
  }
  for (int iteration = 0; iteration < 8; iteration++) {
    float inverse[3][3];
    if (!invert3(r, inverse)) {
      identity3(r);
      return;
    }
    float inverse_transposed[3][3];
    transpose3(inverse, inverse_transposed);
    float next[3][3];
    bool converged = true;
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        next[row][col] = 0.5f * (r[row][col] + inverse_transposed[row][col]);
        /* A deformed `F` is already close to a rotation, so two or three iterations are usually
         * enough; stop as soon as the update is below float noise. */
        if (std::fabs(next[row][col] - r[row][col]) > 1e-6f) {
          converged = false;
        }
        if (!std::isfinite(next[row][col])) {
          identity3(r);
          return;
        }
      }
    }
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        r[row][col] = next[row][col];
      }
    }
    if (converged) {
      return;
    }
  }
}

/** Jacobi eigen decomposition of a symmetric 3x3 matrix. Eigenvectors are the columns of `vec`. */
void jacobi_eigen3(const float input[3][3], float eigen_values[3], float vec[3][3])
{
  float a[3][3];
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      a[row][col] = input[row][col];
    }
  }
  identity3(vec);
  /* `off` is a sum of squares, so it cannot reach 1e-20 in float32 when the diagonal is order 1;
   * the old absolute threshold made every particle run all 32 sweeps. Use a relative bound and a
   * smaller cap: the rotation only has to be accurate at the plastic-projection level. */
  const float diag = std::fabs(a[0][0]) + std::fabs(a[1][1]) + std::fabs(a[2][2]) + 1e-6f;
  const float tolerance = 1e-12f * diag * diag;
  for (int sweep = 0; sweep < 12; sweep++) {
    const float off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if (off <= tolerance) {
      break;
    }
    const int pairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (const auto &pair : pairs) {
      const int p = pair[0];
      const int q = pair[1];
      if (std::fabs(a[p][q]) < 1e-12f) {
        continue;
      }
      const float theta = (a[q][q] - a[p][p]) / (2.0f * a[p][q]);
      const float sign = theta >= 0.0f ? 1.0f : -1.0f;
      const float t = sign / (std::fabs(theta) + std::sqrt(theta * theta + 1.0f));
      const float c = 1.0f / std::sqrt(t * t + 1.0f);
      const float s = t * c;
      for (int k = 0; k < 3; k++) {
        const float a_pk = a[p][k];
        const float a_qk = a[q][k];
        a[p][k] = c * a_pk - s * a_qk;
        a[q][k] = s * a_pk + c * a_qk;
      }
      for (int k = 0; k < 3; k++) {
        const float a_kp = a[k][p];
        const float a_kq = a[k][q];
        a[k][p] = c * a_kp - s * a_kq;
        a[k][q] = s * a_kp + c * a_kq;
      }
      for (int k = 0; k < 3; k++) {
        const float v_kp = vec[k][p];
        const float v_kq = vec[k][q];
        vec[k][p] = c * v_kp - s * v_kq;
        vec[k][q] = s * v_kp + c * v_kq;
      }
    }
  }
  eigen_values[0] = a[0][0];
  eigen_values[1] = a[1][1];
  eigen_values[2] = a[2][2];
}

/**
 * SVD of a 3x3 matrix: `a = U diag(s) V^T`, reduced to `s` and `V`.
 *
 * Only used by the sand return mapping, which rebuilds `F` as `F V diag(scale) V^T`, so `U` is not
 * needed. Dropping it saves a matrix vector product per column per particle per substep.
 */
struct Svd3 {
  float s[3] = {0.0f, 0.0f, 0.0f};
  /** Eigenvectors of `a^T a`, columns. */
  float v[3][3];
};

bool svd3(const float a[3][3], Svd3 &r)
{
  float ata[3][3];
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      ata[row][col] = a[0][row] * a[0][col] + a[1][row] * a[1][col] + a[2][row] * a[2][col];
    }
  }
  float eigen_values[3];
  jacobi_eigen3(ata, eigen_values, r.v);
  for (int i = 0; i < 3; i++) {
    r.s[i] = std::sqrt(std::max(eigen_values[i], 0.0f));
  }
  /* Sort the singular values from large to small, keeping `V` in sync. */
  for (int i = 0; i < 2; i++) {
    int best = i;
    for (int j = i + 1; j < 3; j++) {
      if (r.s[j] > r.s[best]) {
        best = j;
      }
    }
    if (best != i) {
      std::swap(r.s[i], r.s[best]);
      for (int row = 0; row < 3; row++) {
        std::swap(r.v[row][i], r.v[row][best]);
      }
    }
  }
  return std::isfinite(r.s[0]) && std::isfinite(r.s[1]) && std::isfinite(r.s[2]);
}

/** Rebuilds `F_new = F V diag(scale) V^T` after a plastic return mapping. */
void apply_singular_scale(const float f[3][3],
                          const Svd3 &svd,
                          const float scale[3],
                          float *r)
{
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      float sum = 0.0f;
      for (int i = 0; i < 3; i++) {
        float row_term = 0.0f;
        for (int j = 0; j < 3; j++) {
          row_term += f[row][j] * svd.v[j][i];
        }
        sum += row_term * scale[i] * svd.v[col][i];
      }
      r[row * 3 + col] = sum;
    }
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Material models
 * \{ */

/**
 * Drucker-Prager return mapping in principal log strain space (Klar et al. 2016).
 *
 * The yield surface is `q + p tan(phi) <= c`; writing it in log strains gives the deviatoric strain
 * bound used here. `friction` is `tan(phi)` and `cohesion` is in pascal. A material with no
 * friction and no cohesion has no shear strength and flows like a liquid.
 *
 * Damage: the strain the return mapping removes (the excess above the yield surface) is the plastic
 * strain increment. It accumulates into `damage`, which scales the shear modulus, the friction and
 * the cohesion down. Intact regions keep their shape while the bands that actually flowed turn into
 * loose, strength-less sand, which is what makes a falling block break up into chunks. Volumetric
 * stiffness is intentionally left alone, so damaged sand still resists compression (it does not
 * collapse to a point).
 */
constexpr float DAMAGE_SHEAR_DEADBAND = 2.0e-3f;
/* Grain crushing / compaction: volumetric compression beyond this adds damage. Impact compresses a
 * block by many percent while its own weight only compresses it by a fraction of a percent. */
constexpr float DAMAGE_CRUSH_DEADBAND = 3.0e-2f;
/**
 * Strain rate (1/s) below which nothing is damaged. A block in free fall or at rest has almost no
 * strain rate, an impact has tens per second, so this is what separates "sitting there" from
 * "being hit". Damage is driven by the strain rate because a Drucker-Prager material does not fail
 * in tension: it either flows or stays elastic, and the elastic branch still has to break.
 */
constexpr float DAMAGE_STRAIN_RATE = 10.0f;

struct SandStrength {
  float mu;
  float friction;
  float cohesion;
};

inline SandStrength sand_strength(const ParticleData &particles, const int particle)
{
  const float intact = 1.0f - math::clamp(particles.damage[particle], 0.0f, 1.0f);
  SandStrength strength;
  strength.mu = particles.shear_modulus[particle] * intact;
  strength.friction = std::max(particles.friction[particle], 0.0f) * intact;
  strength.cohesion = std::max(particles.cohesion[particle], 0.0f) * intact;
  return strength;
}

void return_map_sand(const int particle,
                     ParticleData &particles,
                     const float *f_in,
                     float *f_out,
                     const float strain_increment)
{
  float f[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      f[i][j] = f_in[i * 3 + j];
    }
  }
  Svd3 svd;
  if (!svd3(f, svd)) {
    for (int i = 0; i < 9; i++) {
      f_out[i] = f_in[i];
    }
    return;
  }
  float epsilon[3];
  float trace = 0.0f;
  for (int i = 0; i < 3; i++) {
    epsilon[i] = std::log(std::max(svd.s[i], 1e-6f));
    trace += epsilon[i];
  }
  float deviatoric[3];
  float deviatoric_norm = 0.0f;
  for (int i = 0; i < 3; i++) {
    deviatoric[i] = epsilon[i] - trace / 3.0f;
    deviatoric_norm += deviatoric[i] * deviatoric[i];
  }
  deviatoric_norm = std::sqrt(deviatoric_norm);

  const float lambda = particles.lame_lambda[particle];
  const SandStrength strength = sand_strength(particles, particle);
  /* `q + p tan(phi)` expressed as a bound on the deviatoric log strain. */
  const float pressure_coefficient = strength.mu > EPSILON ?
                                         (3.0f * lambda + 2.0f * strength.mu) /
                                             (4.0f * strength.mu * std::sqrt(1.5f)) :
                                         0.0f;
  const float yield_strain = std::max(0.0f, -trace) * strength.friction * pressure_coefficient +
                             (strength.mu > EPSILON ? strength.cohesion /
                                                          (2.0f * strength.mu) :
                                                      0.0f);

  if (deviatoric_norm > std::max(yield_strain, 1e-7f)) {
    const float keep = yield_strain / deviatoric_norm;
    float scale[3];
    for (int i = 0; i < 3; i++) {
      const float new_epsilon = deviatoric[i] * keep + trace / 3.0f;
      scale[i] = std::exp(new_epsilon) / std::max(svd.s[i], 1e-6f);
    }
    apply_singular_scale(f, svd, scale, &f_out[0]);
  }
  else {
    for (int i = 0; i < 9; i++) {
      f_out[i] = f_in[i];
    }
  }

  /* Damage: the strain increment above the impact threshold plus volumetric crushing. Both are per
   * substep increments, so the result does not depend on the substep count. */
  const float rate = std::max(particles.damage_rate[particle], 0.0f);
  if (rate > 0.0f) {
    float increment = strain_increment;
    increment += std::max(0.0f, -trace - DAMAGE_CRUSH_DEADBAND);
    if (increment > 0.0f) {
      particles.damage[particle] = math::clamp(
          particles.damage[particle] + rate * increment, 0.0f, 1.0f);
    }
  }
}

/**
 * Cauchy stress of a fluid: only the volumetric part, no shear stiffness.
 *
 * `K (J - 1)` is negative while the fluid is compressed (J < 1) and positive while it is stretched
 * (J > 1). The positive half is tensile stress, and real water has none: letting a fluid pull
 * itself together is what makes an explicit MPM fluid look like gel, because the free surface
 * glues itself shut instead of breaking up and spreading. `tension` scales that half of the curve
 * (0 = water, 1 = a fully elastic bulk material).
 */
void fluid_stress(const float f[3][3],
                  const float bulk,
                  const float tension,
                  float stress[3][3])
{
  const float j = std::isfinite(determinant3(f)) ? determinant3(f) : 1.0f;
  float pressure = bulk * (j - 1.0f);
  if (pressure > 0.0f) {
    pressure *= math::clamp(tension, 0.0f, 1.0f);
  }
  identity3(stress);
  for (int i = 0; i < 3; i++) {
    stress[i][i] = pressure;
  }
}

/**
 * Fixed corotated stress, the elastic part of the sand model.
 *
 * `sigma = (1/J) * (2 mu (F - R) F^T) + lambda (J - 1) I`; the `1/J` converts the first Piola
 * stress to Cauchy stress and the volumetric part only resists compression (dry sand cannot pull).
 */
void sand_stress(const float f[3][3],
                 const float mu,
                 const float lambda,
                 float stress[3][3])
{
  float rotation[3][3];
  polar_rotation(f, rotation);
  float elastic[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      elastic[i][j] = f[i][j] - rotation[i][j];
    }
  }
  float f_transposed[3][3];
  transpose3(f, f_transposed);
  float shear[3][3];
  multiply3(elastic, f_transposed, shear);

  const float j = std::isfinite(determinant3(f)) ? determinant3(f) : 1.0f;
  const float inv_j = 1.0f / std::max(j, 0.1f);
  /* Dry sand does not sustain tension: the volumetric part only resists compression. */
  const float volume = lambda * std::min(j - 1.0f, 0.0f);
  for (int i = 0; i < 3; i++) {
    for (int j_col = 0; j_col < 3; j_col++) {
      stress[i][j_col] = 2.0f * mu * shear[i][j_col] * inv_j;
    }
    stress[i][i] += volume;
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Collider contact
 * \{ */

/**
 * Removes the normal velocity and damps the tangential one against a collider.
 *
 * The damping is expressed as a rate (`1/s`), so it is independent of the substep count. Applying
 * a plain `v *= 1 - friction` every substep was killing 99.6% of the tangential speed per frame at
 * `friction = 0.5`, which turned every puddle into syrup.
 */
inline void apply_wall_contact(float3 &velocity,
                               const float3 &normal,
                               const float friction,
                               const float dt)
{
  const float normal_speed = math::dot(velocity, normal);
  if (normal_speed < 0.0f) {
    velocity -= normal * normal_speed;
  }
  const float3 tangent = velocity - normal * math::dot(velocity, normal);
  const float decay = math::clamp(friction * 25.0f * dt, 0.0f, 1.0f);
  velocity = normal * math::dot(velocity, normal) + tangent * (1.0f - decay);
}

/* -------------------------------------------------------------------- */
/** \name Colliders
 * \{ */

struct SolidSample {
  /** Signed distance in world units, negative inside. */
  float distance = std::numeric_limits<float>::max();
  float3 normal = float3(0.0f, 0.0f, 1.0f);
};

SolidSample sample_box(const float3 &position, const float3 &min, const float3 &max)
{
  const float3 center = (min + max) * 0.5f;
  const float3 half = (max - min) * 0.5f;
  const float3 d = position - center;
  const float3 q = float3(std::fabs(d.x) - half.x, std::fabs(d.y) - half.y, std::fabs(d.z) - half.z);
  const float3 positive_q = float3(std::max(q.x, 0.0f), std::max(q.y, 0.0f), std::max(q.z, 0.0f));
  SolidSample sample;
  if (q.x <= 0.0f && q.y <= 0.0f && q.z <= 0.0f) {
    /* Inside: the closest face decides the normal. */
    int axis = 0;
    if (q.y > q.x && q.y >= q.z) {
      axis = 1;
    }
    else if (q.z > q.x && q.z > q.y) {
      axis = 2;
    }
    sample.distance = q[axis];
    sample.normal = float3(0.0f);
    sample.normal[axis] = d[axis] >= 0.0f ? 1.0f : -1.0f;
    return sample;
  }
  const float outside = math::length(positive_q);
  if (outside > 1e-8f) {
    sample.normal = float3((d.x >= 0.0f ? positive_q.x : -positive_q.x),
                           (d.y >= 0.0f ? positive_q.y : -positive_q.y),
                           (d.z >= 0.0f ? positive_q.z : -positive_q.z)) /
                    outside;
  }
  sample.distance = outside;
  return sample;
}

SolidSample sample_solids(const float3 &position,
                          const Span<const Collider> colliders,
                          const Span<float3> box_min,
                          const Span<float3> box_max,
                          const ha::HaSdfGrid *sdf,
                          const float gradient_epsilon)
{
  SolidSample best;
  for (const int i : colliders.index_range()) {
    if (!colliders[i].box) {
      continue;
    }
    if (box_min[i].x > box_max[i].x) {
      continue;
    }
    const SolidSample box = sample_box(position, box_min[i], box_max[i]);
    if (box.distance < best.distance) {
      best = box;
    }
  }
  if (sdf != nullptr && sdf->is_valid()) {
    /* Fast reject: the surface is inside the grid box, so a point well outside it cannot be in
     * contact. Without this every active cell pays a trilinear sample (and often the six extra
     * ones for the gradient) even in a scene where the collider is only at the floor. */
    const float3 box_min = sdf->origin;
    const float3 box_max = sdf->origin +
                           float3(float(sdf->dims.x - 1), float(sdf->dims.y - 1),
                                  float(sdf->dims.z - 1)) *
                               sdf->spacing;
    const float skip_margin = 8.0f * sdf->spacing;
    if (position.x >= box_min.x - skip_margin && position.x <= box_max.x + skip_margin &&
        position.y >= box_min.y - skip_margin && position.y <= box_max.y + skip_margin &&
        position.z >= box_min.z - skip_margin && position.z <= box_max.z + skip_margin)
    {
      const float distance = ha::ha_sdf_sample(*sdf, position);
      if (distance < best.distance) {
        best.distance = distance;
        /* Beyond a few cells the gradient cannot matter: no node or particle is close enough for a
         * contact, so the six extra trilinear samples are skipped. */
        if (distance <= gradient_epsilon * 4.0f) {
          const float3 dx = float3(gradient_epsilon, 0.0f, 0.0f);
          const float3 dy = float3(0.0f, gradient_epsilon, 0.0f);
          const float3 dz = float3(0.0f, 0.0f, gradient_epsilon);
          const float3 gradient = float3(
              ha::ha_sdf_sample(*sdf, position + dx) - ha::ha_sdf_sample(*sdf, position - dx),
              ha::ha_sdf_sample(*sdf, position + dy) - ha::ha_sdf_sample(*sdf, position - dy),
              ha::ha_sdf_sample(*sdf, position + dz) - ha::ha_sdf_sample(*sdf, position - dz));
          const float length = math::length(gradient);
          if (length > 1e-8f) {
            best.normal = gradient / length;
          }
        }
      }
    }
  }
  return best;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Grid and pressure
 * \{ */

void make_domain(const ParticleData &particles,
                 const Span<const Collider> colliders,
                 const Settings &settings,
                 const float delta_time,
                 const float max_speed,
                 GridData &grid,
                 Stats &r_stats)
{
  float3 min = float3(std::numeric_limits<float>::max());
  float3 max = float3(-std::numeric_limits<float>::max());
  for (const float3 &position : particles.positions) {
    min = math::min(min, position);
    max = math::max(max, position);
  }
  const float3 center = (min + max) * 0.5f;

  float spacing = settings.voxel_size;
  if (spacing <= 0.0f) {
    float radius_sum = 0.0f;
    for (const float radius : particles.radius) {
      radius_sum += radius;
    }
    const float mean_radius = particles.size() > 0 ? radius_sum / particles.size() : 0.0f;
    spacing = mean_radius > 1e-5f ? 2.0f * mean_radius : 0.05f;
  }
  spacing = std::max(spacing, 1e-4f);

  /**
   * The domain has hard walls (velocity is zeroed on the outer node layer and positions are
   * clamped), so it has to cover everything the particles move through during the frame. The box
   * is rebuilt from the positions at the start of the frame, so with a fixed three cell margin a
   * free falling cloud moves further than the margin in one frame (0.2 m at 24 fps after half a
   * second, vs 0.15 m at dx 0.05): the lowest layer then hits the bottom wall, the bounding box
   * never grows downwards, and the material looks like it lands on an invisible ledge. Add the
   * distance the fastest particle travels this frame.
   */
  const float travel = math::clamp(max_speed * delta_time, 0.0f, 8.0f);
  const float margin = 3.0f * spacing + travel;
  if (settings.use_bounds) {
    min = settings.bounds_min;
    max = settings.bounds_max;
  }
  else {
    min -= float3(margin);
    max += float3(margin);
    /* Pull a nearby collider surface into the field, but only by a few cells: a 20 m ground plane
     * used to blow the box up to its own size (a 6 m floor around a 1 m puddle is ~15x the cells,
     * almost all of them empty). The box follows the particles, so the floor enters it as soon as
     * a particle is within a few cells of it. */
    for (const Collider &collider : colliders) {
      if (collider.mesh == nullptr) {
        continue;
      }
      const float reach = margin + 2.0f * spacing;
      float3 collider_min = float3(std::numeric_limits<float>::max());
      float3 collider_max = float3(-std::numeric_limits<float>::max());
      for (const float3 &position : collider.mesh->vert_positions()) {
        collider_min = math::min(collider_min, position);
        collider_max = math::max(collider_max, position);
      }
      collider_min = math::max(collider_min, min - float3(reach));
      collider_max = math::min(collider_max, max + float3(reach));
      if (collider_min.x > collider_max.x || collider_min.y > collider_max.y ||
          collider_min.z > collider_max.z)
      {
        continue;
      }
      min = math::min(min, collider_min);
      max = math::max(max, collider_max);
    }
  }
  if (!(max.x > min.x) || !(max.y > min.y) || !(max.z > min.z)) {
    min = center - float3(0.5f);
    max = center + float3(0.5f);
  }

  int3 dims = int3(std::max(4, int(std::ceil((max.x - min.x) / spacing)) + 1),
                   std::max(4, int(std::ceil((max.y - min.y) / spacing)) + 1),
                   std::max(4, int(std::ceil((max.z - min.z) / spacing)) + 1));
  const int64_t cells = int64_t(dims.x) * int64_t(dims.y) * int64_t(dims.z);
  if (cells > settings.max_grid_cells) {
    /* Coarsen the grid instead of allocating an unbounded box. */
    const float scale = std::cbrt(float(cells) / float(settings.max_grid_cells)) * 1.02f;
    spacing *= scale;
    dims = int3(std::max(4, int(std::ceil((max.x - min.x) / spacing)) + 1),
                std::max(4, int(std::ceil((max.y - min.y) / spacing)) + 1),
                std::max(4, int(std::ceil((max.z - min.z) / spacing)) + 1));
    r_stats.grid_limited = true;
  }
  grid.dims = dims;
  grid.origin = min;
  grid.dx = spacing;
  grid.clear();
  r_stats.grid_cells = grid.cell_count();
  r_stats.grid_dims = dims;
}

void clear_grid(GridData &grid)
{
  grid.clear_active();
}

/** Quadratic B-spline weights `w[0..2]`, the base node index and the local coordinate `fx`. */
inline void quadratic_weights(const float position,
                              const float origin,
                              const float dx,
                              int &r_base,
                              float &r_fx,
                              float r_weights[3])
{
  const float grid_position = (position - origin) / dx;
  r_base = int(std::floor(grid_position - 0.5f));
  r_fx = grid_position - float(r_base);
  const float one_minus = 1.5f - r_fx;
  r_weights[0] = 0.5f * one_minus * one_minus;
  r_weights[1] = 0.75f - (r_fx - 1.0f) * (r_fx - 1.0f);
  r_weights[2] = 0.5f * (r_fx - 0.5f) * (r_fx - 0.5f);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Passes
 * \{ */

void particle_to_grid(const ParticleData &particles,
                      const float dt,
                      const GridData &grid,
                      const Span<int> particle_order,
                      float *mass_out,
                      float3 *momentum_out,
                      float *color_out,
                      Vector<int> &r_active)
{
  const float inv_dx = 1.0f / grid.dx;
  const float stress_scale = -dt * 4.0f * inv_dx * inv_dx;
  /* The colour field is the volume fraction: a lattice with spacing == voxel size gives 1. */
  const float color_scale = 1.0f / (grid.dx * grid.dx * grid.dx);
  for (const int particle : particle_order) {
    const float3 position = particles.positions[particle];
    float f[3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        f[i][j] = particles.deformation[particle * 9 + i * 3 + j];
      }
    }
    float stress[3][3];
    if (particles.material[particle] == int(MaterialType::Fluid)) {
      fluid_stress(f,
                   particles.bulk_modulus[particle],
                   particles.tension[particle],
                   stress);
    }
    else {
      const SandStrength strength = sand_strength(particles, particle);
      sand_stress(f, strength.mu, particles.lame_lambda[particle], stress);
    }

    const float mass = particles.mass[particle];
    const float volume = particles.volume[particle];
    float3 affine_column[3];
    for (int column = 0; column < 3; column++) {
      const float3 c = float3(particles.affine[particle * 9 + 0 * 3 + column],
                              particles.affine[particle * 9 + 1 * 3 + column],
                              particles.affine[particle * 9 + 2 * 3 + column]);
      affine_column[column] =
          float3(stress[0][column], stress[1][column], stress[2][column]) *
              (stress_scale * volume) +
          c * mass;
    }

    int base_x;
    int base_y;
    int base_z;
    float fx;
    float fy;
    float fz;
    float weight_x[3];
    float weight_y[3];
    float weight_z[3];
    quadratic_weights(position.x, grid.origin.x, grid.dx, base_x, fx, weight_x);
    quadratic_weights(position.y, grid.origin.y, grid.dx, base_y, fy, weight_y);
    quadratic_weights(position.z, grid.origin.z, grid.dx, base_z, fz, weight_z);

    /* The stencil is inside the box for every particle that the domain was built around (it starts
     * one margin away from the border). Bail out to the guarded path only if that is not the case,
     * e.g. after a particle was clamped onto the domain face. */
    const int dims_x = grid.dims.x;
    const int dims_y = grid.dims.y;
    const int dims_z = grid.dims.z;
    const int stride_y = dims_x;
    const int stride_z = dims_x * dims_y;
    const float3 velocity = particles.velocities[particle];
    const float3 momentum_base = velocity * mass - affine_column[0] * (fx * grid.dx) -
                                 affine_column[1] * (fy * grid.dx) -
                                 affine_column[2] * (fz * grid.dx);
    /* `momentum = momentum_base + mx[i] + my[j] + mz[k]`, so the inner node only adds vectors. */
    float3 mx[3];
    float3 my[3];
    float3 mz[3];
    for (int a = 0; a < 3; a++) {
      mx[a] = affine_column[0] * (float(a) * grid.dx);
      my[a] = affine_column[1] * (float(a) * grid.dx);
      mz[a] = affine_column[2] * (float(a) * grid.dx);
    }

    if (base_x < 0 || base_y < 0 || base_z < 0 || base_x + 2 >= dims_x || base_y + 2 >= dims_y ||
        base_z + 2 >= dims_z)
    {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            const float weight = weight_x[i] * weight_y[j] * weight_z[k];
            if (weight <= 0.0f) {
              continue;
            }
            if (!grid.contains(base_x + i, base_y + j, base_z + k)) {
              continue;
            }
            const int index = grid.index(base_x + i, base_y + j, base_z + k);
            if (mass_out[index] == 0.0f) {
              r_active.append(index);
            }
            mass_out[index] += weight * mass;
            momentum_out[index] += (momentum_base + mx[i] + my[j] + mz[k]) * weight;
            if (color_out != nullptr) {
              color_out[index] += weight * volume * color_scale;
            }
          }
        }
      }
      continue;
    }

    const int base_index = (base_z * dims_y + base_y) * dims_x + base_x;
    for (int k = 0; k < 3; k++) {
      const float wz = weight_z[k];
      if (wz <= 0.0f) {
        continue;
      }
      const int kz = base_index + k * stride_z;
      const float3 mz_k = mz[k];
      for (int j = 0; j < 3; j++) {
        const float wzy = wz * weight_y[j];
        if (wzy <= 0.0f) {
          continue;
        }
        const int jz = kz + j * stride_y;
        const float3 my_j = my[j];
        for (int i = 0; i < 3; i++) {
          const float weight = wzy * weight_x[i];
          if (weight <= 0.0f) {
            continue;
          }
          const int index = jz + i;
          if (mass_out[index] == 0.0f) {
            r_active.append(index);
          }
          mass_out[index] += weight * mass;
          momentum_out[index] += (momentum_base + mx[i] + my_j + mz_k) * weight;
          if (color_out != nullptr) {
            color_out[index] += weight * volume * color_scale;
          }
        }
      }
    }
  }
}

/**
 * Per slice scratch buffers for the parallel particle to grid pass.
 *
 * A scattering write cannot be parallelized directly without atomics, and atomics would make the
 * result depend on the thread schedule (this solver is expected to be bit deterministic). Instead
 * the particle list is cut into a *fixed* number of slices by index, every slice accumulates into
 * its own dense grid, and the slices are merged in slice order afterwards. The slice count does not
 * depend on the thread count, so `-t 1` and all cores produce the same bits.
 */
struct P2gSlices {
  int count = 0;
  int cells = 0;
  Vector<Vector<float>> mass;
  Vector<Vector<float3>> momentum;
  Vector<Vector<float>> color;
  Vector<Vector<int>> active;

  bool ensure(const int slice_count, const int cell_count)
  {
    if (slice_count <= 1 || cell_count <= 0) {
      return false;
    }
    count = slice_count;
    if (int(mass.size()) == slice_count && cells == cell_count) {
      return true;
    }
    cells = cell_count;
    mass.clear();
    momentum.clear();
    color.clear();
    active.clear();
    mass.reserve(slice_count);
    momentum.reserve(slice_count);
    color.reserve(slice_count);
    active.reserve(slice_count);
    for (int i = 0; i < slice_count; i++) {
      mass.append(Vector<float>(cell_count, 0.0f));
      momentum.append(Vector<float3>(cell_count, float3(0.0f)));
      color.append(Vector<float>(cell_count, 0.0f));
      active.append(Vector<int>());
    }
    return true;
  }

  /** Sums the slices into `grid` in slice order and zeroes them again. */
  void merge_into(GridData &grid)
  {
    for (const int slice : IndexRange(count)) {
      for (const int index : active[slice]) {
        if (grid.mass[index] == 0.0f) {
          grid.active.append(index);
        }
        grid.mass[index] += mass[slice][index];
        grid.momentum[index] += momentum[slice][index];
        grid.color[index] += color[slice][index];
      }
    }
    for (const int slice : IndexRange(count)) {
      for (const int index : active[slice]) {
        mass[slice][index] = 0.0f;
        momentum[slice][index] = float3(0.0f);
        color[slice][index] = 0.0f;
      }
      active[slice].clear();
    }
  }
};

/** Scatters all particles, in parallel slices when the scratch grids fit in memory. */
void particle_to_grid_all(const ParticleData &particles,
                          const float dt,
                          GridData &grid,
                          const Span<int> particle_order,
                          const bool need_color,
                          P2gSlices *slices)
{
  float *color_out = need_color ? grid.color.data() : nullptr;
  if (slices == nullptr || slices->count <= 1) {
    particle_to_grid(particles, dt, grid, particle_order, grid.mass.data(),
                     grid.momentum.data(), color_out, grid.active);
    return;
  }
  const int count = slices->count;
  const int particles_num = particle_order.size();
  threading::parallel_for(IndexRange(count), 1, [&](const IndexRange range) {
    for (const int slice : range) {
      const int begin = int(int64_t(slice) * particles_num / count);
      const int end = int(int64_t(slice + 1) * particles_num / count);
      particle_to_grid(particles,
                       dt,
                       grid,
                       particle_order.slice(begin, end - begin),
                       slices->mass[slice].data(),
                       slices->momentum[slice].data(),
                       need_color ? slices->color[slice].data() : nullptr,
                       slices->active[slice]);
    }
  });
  slices->merge_into(grid);
}

void update_grid(GridData &grid,
                 const Settings &settings,
                 const float dt,
                 const float surface_tension,
                 const Span<const Collider> colliders,
                 const Span<float3> box_min,
                 const Span<float3> box_max,
                 const ha::HaSdfGrid *sdf,
                 Stats &r_stats)
{
  const float damping = std::max(0.0f, 1.0f - settings.velocity_damping * dt);
  const float contact_range = 1.5f * grid.dx;
  const Span<const int> active(grid.active.data(), grid.active.size());
  /* Continuum surface force (Brackbill): `f = sigma * kappa * grad(c)`, with the curvature from the
   * divergence of the *normalised* gradient (`kappa = -div(grad(c) / |grad(c)|)`). The bare
   * `laplacian(c)` form is not equivalent on a diffuse interface: it has opposite signs on the two
   * halves of the transition and sharpens the interface instead of pulling the surface in, which
   * makes a blob grow instead of round off. The colour is a volume fraction (1 inside, 0 outside),
   * so `grad(c)` points inwards and a positive kappa pushes inwards. */
  const bool has_surface_tension = surface_tension > 0.0f && !grid.color.is_empty();
  const float inv_2dx = 0.5f / grid.dx;
  const float cell_volume = grid.dx * grid.dx * grid.dx;
  constexpr float MAX_TENSION_ACCELERATION = 300.0f;

  threading::parallel_for(active.index_range(), 256, [&](const IndexRange range) {
    for (const int range_i : range) {
      const int i = active[range_i];
      const float mass = grid.mass[i];
      if (mass <= 1e-12f) {
        grid.velocity[i] = float3(0.0f);
        continue;
      }
      float3 velocity = grid.momentum[i] / mass;
      velocity += settings.gravity * dt;
      velocity *= damping;

      const int x = i % grid.dims.x;
      const int y = (i / grid.dims.x) % grid.dims.y;
      const int z = i / (grid.dims.x * grid.dims.y);

      if (has_surface_tension) {
        auto color_at = [&](const int cx, const int cy, const int cz) -> float {
          return grid.contains(cx, cy, cz) ? grid.color[grid.index(cx, cy, cz)] : 0.0f;
        };
        const float gx = (color_at(x + 1, y, z) - color_at(x - 1, y, z)) * inv_2dx;
        const float gy = (color_at(x, y + 1, z) - color_at(x, y - 1, z)) * inv_2dx;
        const float gz = (color_at(x, y, z + 1) - color_at(x, y, z - 1)) * inv_2dx;
        const float3 grad = float3(gx, gy, gz);
        const float grad_len = math::length(grad);
        /* Only the interface has a gradient; inside the fluid and far outside it is ~0. */
        if (grad_len > 1e-3f) {
          auto normal_at = [&](const int nx, const int ny, const int nz) -> float3 {
            const float3 local = float3(
                color_at(nx + 1, ny, nz) - color_at(nx - 1, ny, nz),
                color_at(nx, ny + 1, nz) - color_at(nx, ny - 1, nz),
                color_at(nx, ny, nz + 1) - color_at(nx, ny, nz - 1));
            const float local_len = math::length(local);
            return local_len > 1e-6f ? local / local_len : float3(0.0f);
          };
          const float3 n_xp = normal_at(x + 1, y, z);
          const float3 n_xm = normal_at(x - 1, y, z);
          const float3 n_yp = normal_at(x, y + 1, z);
          const float3 n_ym = normal_at(x, y - 1, z);
          const float3 n_zp = normal_at(x, y, z + 1);
          const float3 n_zm = normal_at(x, y, z - 1);
          const float divergence = (n_xp.x - n_xm.x + n_yp.y - n_ym.y + n_zp.z - n_zm.z) *
                                   inv_2dx;
          const float kappa = -divergence;
          const float3 accel = grad * (surface_tension * kappa * cell_volume / mass);
          const float magnitude = math::length(accel);
          velocity += (magnitude > MAX_TENSION_ACCELERATION ?
                           accel * (MAX_TENSION_ACCELERATION / magnitude) :
                           accel) *
                      dt;
        }
      }
      if ((x == 0 && velocity.x < 0.0f) || (x == grid.dims.x - 1 && velocity.x > 0.0f)) {
        velocity.x = 0.0f;
      }
      if ((y == 0 && velocity.y < 0.0f) || (y == grid.dims.y - 1 && velocity.y > 0.0f)) {
        velocity.y = 0.0f;
      }
      if ((z == 0 && velocity.z < 0.0f) || (z == grid.dims.z - 1 && velocity.z > 0.0f)) {
        velocity.z = 0.0f;
      }

      const float3 world = grid.origin +
                           float3(float(x), float(y), float(z)) * grid.dx;
      const SolidSample sample = sample_solids(
          world, colliders, box_min, box_max, sdf, grid.dx * 0.5f);
      if (sample.distance < 0.0f) {
        velocity = float3(0.0f);
      }
      else if (sample.distance < contact_range) {
        apply_wall_contact(velocity, sample.normal, settings.boundary_friction, dt);
      }
      grid.velocity[i] = velocity;
    }
  });

  r_stats.active_cells = int(active.size());
}

void grid_to_particle(ParticleData &particles,
                      const GridData &grid,
                      const float dt)
{
  const float inv_dx = 1.0f / grid.dx;
  threading::parallel_for(IndexRange(particles.size()), 256, [&](const IndexRange range) {
    for (const int particle : range) {
      const float3 position = particles.positions[particle];
      int base_x;
      int base_y;
      int base_z;
      float fx;
      float fy;
      float fz;
      float weight_x[3];
      float weight_y[3];
      float weight_z[3];
      quadratic_weights(position.x, grid.origin.x, grid.dx, base_x, fx, weight_x);
      quadratic_weights(position.y, grid.origin.y, grid.dx, base_y, fy, weight_y);
      quadratic_weights(position.z, grid.origin.z, grid.dx, base_z, fz, weight_z);

      float3 velocity = float3(0.0f);
      float c[3][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            const int x = base_x + i;
            const int y = base_y + j;
            const int z = base_z + k;
            if (!grid.contains(x, y, z)) {
              continue;
            }
            const float weight = weight_x[i] * weight_y[j] * weight_z[k];
            if (weight <= 0.0f) {
              continue;
            }
            const float3 grid_velocity = grid.velocity[grid.index(x, y, z)];
            velocity += grid_velocity * weight;
            const float3 dpos = float3(float(i) - fx, float(j) - fy, float(k) - fz) * grid.dx;
            /* `C = 4 / dx^2 * sum w * v * (x_i - x_p)` (Hu et al. 2018). `dpos` is in world
             * units, so the factor needs `inv_dx` twice: with only one the matrix is `dx` times
             * too small, the affine transfer term vanishes and APIC silently degrades to PIC,
             * which is very dissipative (fluids look like syrup and grains keep their lattice). */
            const float factor = 4.0f * inv_dx * inv_dx * weight;
            c[0][0] += factor * grid_velocity.x * dpos.x;
            c[0][1] += factor * grid_velocity.x * dpos.y;
            c[0][2] += factor * grid_velocity.x * dpos.z;
            c[1][0] += factor * grid_velocity.y * dpos.x;
            c[1][1] += factor * grid_velocity.y * dpos.y;
            c[1][2] += factor * grid_velocity.y * dpos.z;
            c[2][0] += factor * grid_velocity.z * dpos.x;
            c[2][1] += factor * grid_velocity.z * dpos.y;
            c[2][2] += factor * grid_velocity.z * dpos.z;
          }
        }
      }

      const float damping = std::max(0.0f, 1.0f - particles.damping[particle] * dt);
      velocity *= damping;
      velocity *= std::max(0.0f, 1.0f - particles.viscosity[particle] * dt);

      particles.velocities[particle] = velocity;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          particles.affine[particle * 9 + i * 3 + j] = c[i][j];
        }
      }

      /* `F <- (I + dt C) F` */
      float f[3][3];
      float new_f[3][3];
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          f[i][j] = particles.deformation[particle * 9 + i * 3 + j];
          new_f[i][j] = f[i][j];
        }
      }
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          new_f[i][j] += dt * (c[i][0] * f[0][j] + c[i][1] * f[1][j] + c[i][2] * f[2][j]);
        }
      }
      if (particles.material[particle] == int(MaterialType::Sand)) {
        /* Impact damage is driven by the strain rate `|C|`; a block in free fall or at rest is
         * almost rigid, an impact has tens per second. */
        float strain_increment = 0.0f;
        if (particles.damage_rate[particle] > 0.0f) {
          float c_norm_sq = 0.0f;
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
              c_norm_sq += c[i][j] * c[i][j];
            }
          }
          strain_increment = std::max(0.0f, std::sqrt(c_norm_sq) - DAMAGE_STRAIN_RATE) * dt;
        }
        return_map_sand(particle,
                        particles,
                        &new_f[0][0],
                        &particles.deformation[particle * 9],
                        strain_increment);
      }
      else {
        /* Keep a fluid's `F` bounded. The stress only depends on `det(F)`, so the volume is the
         * only part worth preserving; a runaway shear would otherwise grow without bound. */
        float j = determinant3(new_f);
        if (!std::isfinite(j) || std::fabs(j) < 1e-6f) {
          identity3(new_f);
          j = 1.0f;
        }
        const float target = math::clamp(j, 0.25f, 4.0f);
        const float scale = std::cbrt(target / j);
        for (int i = 0; i < 3; i++) {
          for (int j_col = 0; j_col < 3; j_col++) {
            particles.deformation[particle * 9 + i * 3 + j_col] = new_f[i][j_col] * scale;
          }
        }
      }
    }
  });

  /* Stats are collected once per frame (see `collect_stats`): doing it per substep was a serial
   * pass with a 3x3 determinant per particle, which is pure overhead for a diagnostic. */
}

void advect_and_collide(ParticleData &particles,
                        const Settings &settings,
                        const float dt,
                        const GridData &grid,
                        const Span<const Collider> colliders,
                        const Span<float3> box_min,
                        const Span<float3> box_max,
                        const ha::HaSdfGrid *sdf)
{
  const float3 domain_min = grid.origin;
  const float3 domain_max = grid.origin +
                            float3(float(grid.dims.x - 1), float(grid.dims.y - 1),
                                   float(grid.dims.z - 1)) *
                                grid.dx;
  threading::parallel_for(IndexRange(particles.size()), 256, [&](const IndexRange range) {
    for (const int particle : range) {
      float3 position = particles.positions[particle] + particles.velocities[particle] * dt;
      float3 velocity = particles.velocities[particle];
      const SolidSample sample = sample_solids(
          position, colliders, box_min, box_max, sdf, grid.dx * 0.5f);
      if (sample.distance < particles.radius[particle]) {
        position += sample.normal * (particles.radius[particle] - sample.distance);
        apply_wall_contact(velocity, sample.normal, settings.boundary_friction, dt);
      }
      position.x = math::clamp(position.x, domain_min.x, domain_max.x);
      position.y = math::clamp(position.y, domain_min.y, domain_max.y);
      position.z = math::clamp(position.z, domain_min.z, domain_max.z);
      particles.positions[particle] = position;
      particles.velocities[particle] = velocity;
    }
  });
}

/** Particle diagnostics, once per frame (serial, so the numbers do not depend on the threads). */
void collect_stats(const ParticleData &particles, Stats &r_stats)
{
  float max_speed = 0.0f;
  float max_j = 1.0f;
  float min_j = 1.0f;
  for (const int particle : IndexRange(particles.size())) {
    max_speed = std::max(max_speed, math::length(particles.velocities[particle]));
    const float j = determinant3_data(&particles.deformation[particle * 9]);
    if (std::isfinite(j)) {
      max_j = std::max(max_j, j);
      min_j = std::min(min_j, j);
    }
  }
  r_stats.max_speed = max_speed;
  r_stats.max_volume_ratio = max_j;
  r_stats.min_volume_ratio = min_j;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Substep
 * \{ */

float maximum_surface_tension(const ParticleData &particles)
{
  float sigma = 0.0f;
  for (const int particle : IndexRange(particles.size())) {
    sigma = std::max(sigma, particles.surface_tension[particle]);
  }
  return sigma;
}

float maximum_speed(const ParticleData &particles)
{
  float max_speed = 0.0f;
  for (const int particle : IndexRange(particles.size())) {
    max_speed = std::max(max_speed, math::length(particles.velocities[particle]));
  }
  return max_speed;
}

int choose_substeps(const ParticleData &particles,
                    const Settings &settings,
                    const float spacing,
                    const float dt,
                    const float max_speed,
                    int &r_required)
{
  float max_wave_speed = 0.0f;
  for (const int particle : IndexRange(particles.size())) {
    const float density = std::max(particles.density[particle], 1e-3f);
    float wave_speed = 0.0f;
    if (particles.material[particle] == int(MaterialType::Fluid)) {
      wave_speed = std::sqrt(std::max(particles.bulk_modulus[particle], 0.0f) / density);
    }
    else {
      wave_speed = std::sqrt(std::max(particles.lame_lambda[particle] +
                                          2.0f * particles.shear_modulus[particle],
                                      0.0f) /
                             density);
    }
    max_wave_speed = std::max(max_wave_speed, wave_speed);
  }
  int substeps = std::max(settings.min_substeps, 1);
  if (dt > 0.0f) {
    const float movement = max_speed * dt / (std::max(settings.cfl, 1e-3f) * spacing);
    const float wave = max_wave_speed * dt / (std::max(settings.cfl, 1e-3f) * spacing);
    substeps = std::max(substeps, int(std::ceil(std::max(movement, wave))));
  }
  r_required = substeps;
  substeps = std::min(substeps, std::max(settings.max_substeps, 1));
  return substeps;
}

void substep(ParticleData &particles,
             const Settings &settings,
             const float dt,
             GridData &grid,
             const Span<int> particle_order,
             const Span<const Collider> colliders,
             const Span<float3> box_min,
             const Span<float3> box_max,
             const ha::HaSdfGrid *sdf,
             const float surface_tension,
             P2gSlices *slices,
             Stats &r_stats)
{
  auto mark = std::chrono::steady_clock::now();
  clear_grid(grid);
  r_stats.milliseconds_clear += elapsed_ms(mark);
  mark = std::chrono::steady_clock::now();
  particle_to_grid_all(particles, dt, grid, particle_order, surface_tension > 0.0f, slices);
  r_stats.milliseconds_particle_to_grid += elapsed_ms(mark);
  mark = std::chrono::steady_clock::now();
  update_grid(grid, settings, dt, surface_tension, colliders, box_min, box_max, sdf, r_stats);
  r_stats.milliseconds_update_grid += elapsed_ms(mark);
  mark = std::chrono::steady_clock::now();
  grid_to_particle(particles, grid, dt);
  r_stats.milliseconds_grid_to_particle += elapsed_ms(mark);
  mark = std::chrono::steady_clock::now();
  advect_and_collide(particles, settings, dt, grid, colliders, box_min, box_max, sdf);
  r_stats.milliseconds_advect += elapsed_ms(mark);
}

/** \} */

}  // namespace

void ParticleData::resize(const int count)
{
  positions.resize(count);
  velocities.resize(count);
  affine.resize(count * 9);
  deformation.resize(count * 9);
  material.resize(count);
  density.resize(count);
  volume.resize(count);
  mass.resize(count);
  bulk_modulus.resize(count);
  shear_modulus.resize(count);
  lame_lambda.resize(count);
  viscosity.resize(count);
  tension.resize(count);
  surface_tension.resize(count);
  friction.resize(count);
  cohesion.resize(count);
  damage.resize(count);
  damage_rate.resize(count);
  damping.resize(count);
  radius.resize(count);

  std::fill(velocities.begin(), velocities.end(), float3(0.0f));
  std::fill(affine.begin(), affine.end(), 0.0f);
  for (int i = 0; i < count; i++) {
    positions[i] = float3(0.0f);
    deformation[i * 9 + 0] = 1.0f;
    deformation[i * 9 + 1] = 0.0f;
    deformation[i * 9 + 2] = 0.0f;
    deformation[i * 9 + 3] = 0.0f;
    deformation[i * 9 + 4] = 1.0f;
    deformation[i * 9 + 5] = 0.0f;
    deformation[i * 9 + 6] = 0.0f;
    deformation[i * 9 + 7] = 0.0f;
    deformation[i * 9 + 8] = 1.0f;
    material[i] = int(MaterialType::Fluid);
    density[i] = 1000.0f;
    volume[i] = 0.0f;
    mass[i] = 0.0f;
    bulk_modulus[i] = 5.0e4f;
    shear_modulus[i] = 0.0f;
    lame_lambda[i] = 0.0f;
    viscosity[i] = 0.2f;
    tension[i] = 0.0f;
    surface_tension[i] = 0.0f;
    friction[i] = 0.0f;
    cohesion[i] = 0.0f;
    damage[i] = 0.0f;
    damage_rate[i] = 0.0f;
    damping[i] = 0.1f;
    radius[i] = 0.05f;
  }
}

void ParticleData::fill_defaults()
{
  for (const int i : IndexRange(size())) {
    if (!std::isfinite(density[i]) || density[i] <= 0.0f) {
      density[i] = 1000.0f;
    }
    if (!std::isfinite(radius[i]) || radius[i] <= 0.0f) {
      radius[i] = 0.05f;
    }
    if (!std::isfinite(volume[i]) || volume[i] <= 0.0f) {
      volume[i] = (2.0f * radius[i]) * (2.0f * radius[i]) * (2.0f * radius[i]);
    }
    if (!std::isfinite(mass[i]) || mass[i] <= 0.0f) {
      mass[i] = density[i] * volume[i];
    }
    if (!std::isfinite(mass[i]) || mass[i] <= 0.0f) {
      volume[i] = 1.0f / density[i];
      mass[i] = 1.0f;
    }
    if (!std::isfinite(viscosity[i])) {
      viscosity[i] = 0.0f;
    }
    if (!std::isfinite(tension[i])) {
      tension[i] = 0.0f;
    }
    if (!std::isfinite(surface_tension[i]) || surface_tension[i] < 0.0f) {
      surface_tension[i] = 0.0f;
    }
    if (!std::isfinite(damage[i])) {
      damage[i] = 0.0f;
    }
    if (!std::isfinite(damage_rate[i]) || damage_rate[i] < 0.0f) {
      damage_rate[i] = 0.0f;
    }
    if (!std::isfinite(damping[i])) {
      damping[i] = 0.0f;
    }
  }
}

void GridData::clear()
{
  const int cells = cell_count();
  mass.resize(cells);
  momentum.resize(cells);
  velocity.resize(cells);
  color.resize(cells);
  std::fill(mass.begin(), mass.end(), 0.0f);
  std::fill(momentum.begin(), momentum.end(), float3(0.0f));
  std::fill(velocity.begin(), velocity.end(), float3(0.0f));
  std::fill(color.begin(), color.end(), 0.0f);
  active.clear();
}

void GridData::clear_active()
{
  for (const int index : active) {
    mass[index] = 0.0f;
    momentum[index] = float3(0.0f);
    velocity[index] = float3(0.0f);
    color[index] = 0.0f;
  }
  active.clear();
}

void solve_frame(ParticleData &particles,
                 const Span<const Collider> colliders,
                 const Settings &settings,
                 const float delta_time,
                 Stats &r_stats)
{
  const auto start = std::chrono::steady_clock::now();
  r_stats = Stats();
  r_stats.particles = particles.size();
  r_stats.has_particles = particles.size() > 0;
  if (particles.size() == 0) {
    r_stats.milliseconds = 0.0;
    return;
  }
  particles.fill_defaults();

  GridData grid;
  const float max_speed = maximum_speed(particles);
  const float surface_tension = maximum_surface_tension(particles);
  make_domain(particles, colliders, settings, delta_time, max_speed, grid, r_stats);

  /* Box bounds are resolved once: a box collider is analytic, only meshes need a field. */
  Vector<float3> box_min(colliders.size(), float3(0.0f));
  Vector<float3> box_max(colliders.size(), float3(-1.0f));
  Vector<float3> sdf_positions;
  Vector<int> sdf_corner_verts;
  Vector<int3> sdf_corner_tris;
  float max_radius = 0.0f;
  for (const int i : colliders.index_range()) {
    const Collider &collider = colliders[i];
    if (collider.mesh == nullptr) {
      continue;
    }
    float3 min = float3(std::numeric_limits<float>::max());
    float3 max = float3(-std::numeric_limits<float>::max());
    for (const float3 &position : collider.mesh->vert_positions()) {
      min = math::min(min, position);
      max = math::max(max, position);
    }
    if (collider.box) {
      box_min[i] = min;
      box_max[i] = max;
      continue;
    }
    const int vertex_offset = sdf_positions.size();
    sdf_positions.extend(collider.mesh->vert_positions());
    const Span<int> corner_verts = collider.mesh->corner_verts();
    const Span<int3> corner_tris = collider.mesh->corner_tris();
    const int corner_offset = sdf_corner_verts.size();
    sdf_corner_verts.reserve(corner_offset + corner_verts.size());
    for (const int index : corner_verts) {
      sdf_corner_verts.append(index + vertex_offset);
    }
    for (const int3 &tri : corner_tris) {
      sdf_corner_tris.append(tri + int3(corner_offset));
    }
  }
  for (const float radius : particles.radius) {
    max_radius = std::max(max_radius, radius);
  }

  ha::HaSdfGrid sdf;
  bool sdf_valid = false;
  if (!sdf_positions.is_empty()) {
    const float3 region_min = grid.origin;
    const float3 region_max = grid.origin +
                              float3(float(grid.dims.x - 1), float(grid.dims.y - 1),
                                     float(grid.dims.z - 1)) *
                                  grid.dx;
    const float band = std::max(2.0f * grid.dx, 2.0f * max_radius + grid.dx);
    sdf_valid = ha::ha_sdf_build(sdf_positions.as_span(),
                                 sdf_corner_verts.as_span(),
                                 sdf_corner_tris.as_span(),
                                 grid.dx,
                                 band,
                                 region_min,
                                 region_max,
                                 (region_min + region_max) * 0.5f,
                                 true,
                                 sdf);
    if (!sdf_valid) {
      r_stats.collider_sdf_failed = true;
    }
  }

  /* A stable particle order makes the scattering (and the slice merge) deterministic. */
  Vector<int> order(particles.size());
  std::iota(order.begin(), order.end(), 0);

  /* Parallel particle to grid: fixed slices, each with its own scratch grid, merged in order.
   * The slice count only depends on the grid size, never on the thread count, so the result is the
   * same with `-t 1` and with all cores. Falls back to one serial pass when even two slices do not
   * fit in the budget. */
  P2gSlices slices;
  const int64_t slice_bytes = int64_t(grid.cell_count()) *
                              int64_t(sizeof(float) + sizeof(float3));
  int slice_count = 1;
  if (particles.size() >= 512 && slice_bytes > 0) {
    slice_count = int(std::min<int64_t>(P2G_SLICE_COUNT, P2G_SLICE_BUDGET / slice_bytes));
    if (slice_count < 2) {
      slice_count = 1;
    }
  }
  const bool use_slices = slice_count >= 2 && slices.ensure(slice_count, grid.cell_count());

  int required_substeps = 0;
  const int substeps = choose_substeps(
      particles, settings, grid.dx, delta_time, max_speed, required_substeps);
  r_stats.substeps = substeps;
  r_stats.cfl_limited = required_substeps > std::max(settings.max_substeps, 1);
  const float dt = delta_time / std::max(substeps, 1);

  for (int step = 0; step < substeps; step++) {
    substep(particles,
            settings,
            dt,
            grid,
            order,
            colliders,
            box_min.as_span(),
            box_max.as_span(),
            sdf_valid ? &sdf : nullptr,
            surface_tension,
            use_slices ? &slices : nullptr,
            r_stats);
  }

  collect_stats(particles, r_stats);

  /* The CFL request is only known before the frame; the chosen substeps may have been clamped. */
  const auto end = std::chrono::steady_clock::now();
  r_stats.milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace blender::geometry::textura
