/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo_nodes
 *
 * `Fluid Surface` rebuilds a surface from the particles of a fluid.
 *
 * The approach is the one of [SplashSurf](https://github.com/InteractiveComputerGraphics/
 * splashsurf): instead of unioning spheres around the particles (which makes a blobby, inflated
 * surface that is limited by the particle radius) the particles are splatted into a *density*
 * field with a compact support kernel, and the isosurface of that field is extracted. Because the
 * kernel of a particle falls off smoothly and several particles overlap, the surface sits where
 * half of the kernel mass is, i.e. roughly at the positions of the outermost particles, and thin
 * splashes stay thin instead of turning into a string of blobs.
 *
 * The density is normalized so that a full particle lattice at the recommended spacing (`one
 * radius`) gives exactly 1.0, which makes `Iso Level` mean "fraction of the rest density" and
 * keeps it independent of the particle size. 0.6 (the SplashSurf default) puts the surface
 * slightly outside the outermost layer, 0.5 puts it through their centers.
 *
 * Afterwards the extracted mesh is smoothed with Taubin's λ|μ filter. Taubin is used instead of a
 * plain Laplacian because it does not shrink the surface: the positive and the negative step
 * cancel for the low frequencies, so the volume is preserved while the voxel scale noise of the
 * marching cubes output is removed.
 */

#include <cmath>

#include "BLI_array.hh"
#include "BLI_math_base.hh"
#include "BLI_kdtree.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_task.hh"
#include "BLI_threads.h"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_lib_id.hh"
#include "BKE_mesh.hh"
#include "BKE_volume.hh"
#include "BKE_volume_to_mesh.hh"

#include "BLT_translation.hh"

#include "node_geometry_util.hh"

#ifdef WITH_OPENVDB
#  include "BKE_volume_grid.hh"
#  include "BKE_volume_openvdb.hh"

#  include <openvdb/openvdb.h>
#  include <openvdb/tools/LevelSetFilter.h>
#endif

namespace blender::nodes::node_geo_fluid_surface_cc {

#ifdef WITH_OPENVDB

/** How the surface is reconstructed. */
enum class SurfaceMode : int {
  /** Sum of the particle kernels, extracted at `Iso Level` of the rest density. */
  Density = 0,
  /** Kernel weighted average of the per particle sphere distances (a smoothed distance field). */
  Distance = 1,
};

/** One fluid particle as far as the surface reconstruction is concerned. */
struct FluidParticle {
  float3 position;
  /** Radius of the particle, the kernel support is twice this. */
  float radius;
  /** Volume the particle occupies, i.e. `mass / rest_density` (or `radius^3` without mass). */
  float volume;
};

/**
 * Collects the particles of one component.
 *
 * `override_radius` (the node's `Particle Radius`) wins when it is not zero: an input that already
 * carries a `radius` attribute used to ignore the socket entirely, which made the input look dead.
 * Zero means "use the attribute", and 0.05 m when the geometry has none.
 */
static void gather_particles_from_component(const GeometryComponent &component,
                                            const float override_radius,
                                            Vector<FluidParticle> &r_particles)
{
  if (component.is_empty()) {
    return;
  }
  const int points_num = component.attribute_domain_size(AttrDomain::Point);
  if (points_num == 0) {
    return;
  }
  const std::optional<bke::AttributeAccessor> attributes = component.attributes();
  if (!attributes) {
    return;
  }
  const VArray<float3> positions = *attributes->lookup<float3>("position", AttrDomain::Point);
  if (!positions) {
    return;
  }
  const VArray<float> radii =
      attributes->lookup_or_default<float>("radius", AttrDomain::Point, 0.05f).varray;
  /* The fluid stores the physical particle mass and the rest density, so the volume of a particle
   * is `mass / density`. Without them the particle is assumed to fill the lattice cell of the
   * recommended spacing, which is exactly one radius. */
  const VArray<float> masses =
      attributes->lookup_or_default<float>("mass", AttrDomain::Point, 0.0f).varray;
  const VArray<float> densities = attributes
                                      ->lookup_or_default<float>(
                                          "puncliga_rest_density", AttrDomain::Point, 0.0f)
                                      .varray;

  const int offset = r_particles.size();
  r_particles.resize(offset + points_num);
  MutableSpan<FluidParticle> dst = r_particles.as_mutable_span().slice(offset, points_num);
  for (const int i : IndexRange(points_num)) {
    const float radius = override_radius > 0.0f ? override_radius : radii[i];
    const float safe_radius = std::max(radius, 1e-5f);
    const float mass = masses[i];
    const float density = densities[i];
    const float volume = (mass > 0.0f && density > 0.0f) ?
                             mass / density :
                             safe_radius * safe_radius * safe_radius;
    dst[i] = {positions[i], safe_radius, volume};
  }
}

/**
 * One particle's kernel shape: the unit sphere sheared into the ellipsoid that matches how its
 * neighbours are distributed (Yu & Turk 2013, "Reconstructing surfaces of particle-based fluids
 * using anisotropic kernels").
 *
 * A spherical kernel that is unrelated to the local particle layout is exactly why a flat sheet of
 * particles produces a bumpy surface: every particle bulges the field on its own. Fitting an
 * ellipsoid to the neighbours (flat across a sheet, round inside a droplet) makes the sheet smooth
 * while a droplet keeps its sphere, which is the "smooth without losing detail" tradeoff that an
 * isotropic blur cannot make. `rows` holds `S` (symmetric) and the kernel support is
 * `|S d| < h`, with `det(S) = 1` so the density integral and the meaning of `Iso Level` are
 * unchanged.
 */
struct KernelShape {
  float3 row_x = float3(1.0f, 0.0f, 0.0f);
  float3 row_y = float3(0.0f, 1.0f, 0.0f);
  float3 row_z = float3(0.0f, 0.0f, 1.0f);
};

/** Jacobi eigen decomposition of a symmetric 3x3 matrix (row major), eigenvalues ascending. */
static void symmetric_eigen3(const float m[3][3], float r_values[3], float3 r_vectors[3])
{
  float a[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      a[i][j] = m[i][j];
    }
  }
  float v[3][3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}};
  for (int sweep = 0; sweep < 12; sweep++) {
    const float off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    const float diag = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
    if (off <= 1e-12f * std::max(diag, 1e-12f)) {
      break;
    }
    for (int p_i = 0; p_i < 2; p_i++) {
      for (int q_i = p_i + 1; q_i < 3; q_i++) {
        if (std::fabs(a[p_i][q_i]) < 1e-20f) {
          continue;
        }
        const float theta = (a[q_i][q_i] - a[p_i][p_i]) / (2.0f * a[p_i][q_i]);
        const float sign = theta >= 0.0f ? 1.0f : -1.0f;
        const float t = sign / (std::fabs(theta) + std::sqrt(theta * theta + 1.0f));
        const float c = 1.0f / std::sqrt(t * t + 1.0f);
        const float s = t * c;
        for (int k = 0; k < 3; k++) {
          const float akp = a[k][p_i];
          const float akq = a[k][q_i];
          a[k][p_i] = c * akp - s * akq;
          a[k][q_i] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; k++) {
          const float apk = a[p_i][k];
          const float aqk = a[q_i][k];
          a[p_i][k] = c * apk - s * aqk;
          a[q_i][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; k++) {
          const float vkp = v[k][p_i];
          const float vkq = v[k][q_i];
          v[k][p_i] = c * vkp - s * vkq;
          v[k][q_i] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (int i = 0; i < 3; i++) {
    r_values[i] = a[i][i];
    r_vectors[i] = float3(v[0][i], v[1][i], v[2][i]);
  }
}

/**
 * Builds the kernel of every particle from its neighbourhood. The covariance of the neighbours
 * describes the local shape: a flat sheet has one tiny eigenvalue (the normal direction), a
 * droplet has three similar ones. The kernel is the inverse square root of that covariance,
 * normalised to `det = 1`.
 */
static void build_kernel_shapes(const Span<FluidParticle> particles,
                                const float mean_radius,
                                Array<KernelShape> &r_shapes)
{
  const int count = int(particles.size());
  r_shapes.reinitialize(count);
  if (count < 8) {
    return;
  }
  const float search = 3.0f * mean_radius;
  KDTree<float3> *tree = kdtree_new<float3>(uint(count));
  for (const int i : IndexRange(count)) {
    kdtree_insert(tree, i, particles[i].position);
  }
  kdtree_balance(tree);
  static constexpr int MAX_NEIGHBOURS = 32;
  const float min_lambda = 1e-8f * search * search;
  threading::parallel_for(IndexRange(count), 256, [&](const IndexRange range) {
    for (const int i : range) {
      const float3 center = particles[i].position;
      KDTreeNearest<float3> nearest[MAX_NEIGHBOURS];
      const int found = kdtree_find_nearest_n(tree, center, nearest, MAX_NEIGHBOURS);
      if (found < 6) {
        continue;
      }
      float covariance[3][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
      float weight_sum = 0.0f;
      for (int n = 0; n < found; n++) {
        const float distance = nearest[n].dist;
        if (distance > search && n > 0) {
          break;
        }
        const float3 delta = nearest[n].co - center;
        /* Compact support with a smooth falloff; the nearest node is the particle itself. */
        const float t = std::max(0.0f, 1.0f - (distance / search) * (distance / search));
        const float w = t * t;
        if (w <= 0.0f) {
          continue;
        }
        covariance[0][0] += w * delta.x * delta.x;
        covariance[0][1] += w * delta.x * delta.y;
        covariance[0][2] += w * delta.x * delta.z;
        covariance[1][1] += w * delta.y * delta.y;
        covariance[1][2] += w * delta.y * delta.z;
        covariance[2][2] += w * delta.z * delta.z;
        weight_sum += w;
      }
      if (weight_sum <= 0.0f) {
        continue;
      }
      const float inv_weight = 1.0f / weight_sum;
      covariance[0][0] *= inv_weight;
      covariance[0][1] *= inv_weight;
      covariance[0][2] *= inv_weight;
      covariance[1][1] *= inv_weight;
      covariance[1][2] *= inv_weight;
      covariance[2][2] *= inv_weight;
      covariance[1][0] = covariance[0][1];
      covariance[2][0] = covariance[0][2];
      covariance[2][1] = covariance[1][2];

      float values[3];
      float3 vectors[3];
      symmetric_eigen3(covariance, values, vectors);
      /* Clamp the anisotropy: a 4:1 squash already turns a sphere into a convincing disc, and more
       * than that starts to alias on the voxel grid. */
      constexpr float MAX_ANISOTROPY = 4.0f;
      float clamped[3];
      const float largest = std::max(std::max(values[0], values[1]), values[2]);
      for (int axis = 0; axis < 3; axis++) {
        clamped[axis] = std::max(values[axis], largest / (MAX_ANISOTROPY * MAX_ANISOTROPY));
        clamped[axis] = std::max(clamped[axis], min_lambda);
      }
      /* Scales are `1 / sqrt(lambda)`, normalised so their product is 1 (volume preserving). */
      float scales[3];
      float product = 1.0f;
      for (int axis = 0; axis < 3; axis++) {
        scales[axis] = 1.0f / std::sqrt(clamped[axis]);
        product *= scales[axis];
      }
      const float normalize = 1.0f / std::cbrt(product);
      for (int axis = 0; axis < 3; axis++) {
        scales[axis] *= normalize;
      }
      /* S = sum_axis scale[axis] * (u u^T), which is symmetric. */
      KernelShape shape;
      float3 rows[3] = {float3(0.0f), float3(0.0f), float3(0.0f)};
      for (int axis = 0; axis < 3; axis++) {
        const float3 u = vectors[axis];
        const float s = scales[axis];
        rows[0] += float3(u.x * u.x, u.x * u.y, u.x * u.z) * s;
        rows[1] += float3(u.y * u.x, u.y * u.y, u.y * u.z) * s;
        rows[2] += float3(u.z * u.x, u.z * u.y, u.z * u.z) * s;
      }
      shape.row_x = rows[0];
      shape.row_y = rows[1];
      shape.row_z = rows[2];
      r_shapes[i] = shape;
    }
  });
  kdtree_free(tree);
}

/**
 * Sum of the kernel over a full lattice with spacing `spacing`, times the volume of one lattice
 * cell. Dividing the splatted density by it makes the density of a lattice exactly 1.0.
 */
static double lattice_kernel_sum(const float spacing, const float radius_scale)
{
  const double s = spacing;
  const double h = double(radius_scale) * s;
  const double h2 = h * h;
  const double kernel = (315.0 / (64.0 * M_PI)) / (h2 * h2 * h2 * h2 * h);
  const int reach = int(std::ceil(h / s));
  double sum = 0.0;
  for (int k = -reach; k <= reach; k++) {
    for (int j = -reach; j <= reach; j++) {
      for (int i = -reach; i <= reach; i++) {
        const double d2 = (double(i) * i + double(j) * j + double(k) * k) * s * s;
        if (d2 >= h2) {
          continue;
        }
        const double falloff = h2 - d2;
        sum += kernel * falloff * falloff * falloff * s * s * s;
      }
    }
  }
  return sum > 0.0 ? sum : 1.0;
}

/** Poly6 splat of the given particles into `grid` (which is used by a single thread). */
static void splat_particles(const Span<FluidParticle> particles,
                            const Span<KernelShape> kernel_shapes,
                            const float radius_scale,
                            const float voxel_size,
                            const double normalizer,
                            openvdb::FloatGrid &grid)
{
  openvdb::FloatGrid::Accessor accessor = grid.getAccessor();
  const double voxel_size_sq = double(voxel_size) * double(voxel_size);
  for (const int particle_i : particles.index_range()) {
    const FluidParticle &particle = particles[particle_i];
    const double h = double(radius_scale) * double(particle.radius);
    if (h <= 0.0 || particle.volume <= 0.0f) {
      continue;
    }
    const double h2 = h * h;
    /* `315 / (64 pi h^9)`. */
    const double kernel = (315.0 / (64.0 * M_PI)) / (h2 * h2 * h2 * h2 * h);
    const double scale = double(particle.volume) * kernel / normalizer;

    const openvdb::Vec3d center = grid.transform().worldToIndex(
        openvdb::Vec3d(particle.position.x, particle.position.y, particle.position.z));
    /* OpenVDB voxel `i` is centered at `i * voxel_size` (unlike image space, there is no half
     * voxel offset), which is also why `worldToIndexCellCentered` rounds instead of flooring. */
    const bool shaped = kernel_shapes.size() == particles.size();
    /* World space extent of the (possibly sheared) kernel: an ellipsoid inside `|S d| < h`. */
    float extent = float(h);
    if (shaped) {
      const KernelShape &shape = kernel_shapes[particle_i];
      const float3 row_x = shape.row_x * float(h);
      const float3 row_y = shape.row_y * float(h);
      const float3 row_z = shape.row_z * float(h);
      extent = std::max({std::fabs(row_x.x) + std::fabs(row_x.y) + std::fabs(row_x.z),
                         std::fabs(row_y.x) + std::fabs(row_y.y) + std::fabs(row_y.z),
                         std::fabs(row_z.x) + std::fabs(row_z.y) + std::fabs(row_z.z)});
    }
    const int reach = int(std::ceil(double(extent) / double(voxel_size)));
    const int base_i = int(std::floor(center.x()));
    const int base_j = int(std::floor(center.y()));
    const int base_k = int(std::floor(center.z()));
    const KernelShape *shape = shaped ? &kernel_shapes[particle_i] : nullptr;
    for (int k = base_k - reach; k <= base_k + reach; k++) {
      const double dz = double(k) - center.z();
      for (int j = base_j - reach; j <= base_j + reach; j++) {
        const double dy = double(j) - center.y();
        for (int i = base_i - reach; i <= base_i + reach; i++) {
          const double dx = double(i) - center.x();
          double d2;
          if (shape == nullptr) {
            d2 = (dx * dx + dy * dy + dz * dz) * voxel_size_sq;
          }
          else {
            /* `|S d|^2` with `S` symmetric: the distance the kernel actually sees. */
            const float3 d = float3(float(dx * double(voxel_size)),
                                    float(dy * double(voxel_size)),
                                    float(dz * double(voxel_size)));
            const float3 sd = float3(math::dot(shape->row_x, d),
                                     math::dot(shape->row_y, d),
                                     math::dot(shape->row_z, d));
            d2 = double(math::dot(sd, sd));
          }
          if (d2 >= h2) {
            continue;
          }
          const double falloff = h2 - d2;
          const float value = float(scale * falloff * falloff * falloff);
          const openvdb::Coord coord(i, j, k);
          accessor.setValue(coord, accessor.getValue(coord) + value);
        }
      }
    }
  }
}

/**
 * Splats one bucket of particles as a union of spheres: `min(|x - p| - r)`.
 *
 * The union is the honest signed distance field of the particles, so the grid can be handed to
 * OpenVDB's level set filter below, which smooths it and re-normalises the gradient. Doing the
 * smoothing *on a distance field* (instead of on a density) is what makes a sheet come out flat:
 * the filter moves the surface, it does not just average values that already contain the bumps.
 */
static void splat_union_sdf(const Span<FluidParticle> particles,
                            const float radius_scale,
                            const float voxel_size,
                            openvdb::FloatGrid &grid)
{
  openvdb::FloatGrid::Accessor accessor = grid.getAccessor();
  const double voxel_size_sq = double(voxel_size) * double(voxel_size);
  for (const FluidParticle &particle : particles) {
    const double h = double(radius_scale) * double(particle.radius);
    if (h <= 0.0) {
      continue;
    }
    const double h2 = h * h;
    const openvdb::Vec3d center = grid.transform().worldToIndex(
        openvdb::Vec3d(particle.position.x, particle.position.y, particle.position.z));
    const int reach = int(std::ceil(h / double(voxel_size)));
    const int base_i = int(std::floor(center.x()));
    const int base_j = int(std::floor(center.y()));
    const int base_k = int(std::floor(center.z()));
    for (int k = base_k - reach; k <= base_k + reach; k++) {
      const double dz = double(k) - center.z();
      for (int j = base_j - reach; j <= base_j + reach; j++) {
        const double dy = double(j) - center.y();
        const double dyz2 = (dy * dy + dz * dz) * voxel_size_sq;
        if (dyz2 >= h2) {
          continue;
        }
        for (int i = base_i - reach; i <= base_i + reach; i++) {
          const double dx = double(i) - center.x();
          const double d2 = dyz2 + dx * dx * voxel_size_sq;
          if (d2 >= h2) {
            continue;
          }
          const float distance = float(std::sqrt(d2) - double(particle.radius));
          const openvdb::Coord coord(i, j, k);
          accessor.setValue(coord, std::min(accessor.getValue(coord), distance));
        }
      }
    }
  }
}

/** Taubin λ|μ smoothing: removes the marching cubes noise without shrinking the surface. */
static void smooth_positions_taubin(Mesh &mesh,
                                    const int iterations,
                                    const float lambda,
                                    const float mu)
{
  const int verts_num = mesh.verts_num;
  if (verts_num == 0 || iterations <= 0) {
    return;
  }
  const Span<int2> edges = mesh.edges();
  if (edges.is_empty()) {
    return;
  }

  /* Vertex neighbours in CSR form. */
  Array<int> offsets(verts_num + 1);
  offsets.fill(0);
  for (const int2 &edge : edges) {
    offsets[edge[0] + 1]++;
    offsets[edge[1] + 1]++;
  }
  for (const int i : IndexRange(verts_num)) {
    offsets[i + 1] += offsets[i];
  }
  Array<int> neighbours(offsets[verts_num]);
  Array<int> cursors = offsets;
  for (const int2 &edge : edges) {
    neighbours[cursors[edge[0]]++] = edge[1];
    neighbours[cursors[edge[1]]++] = edge[0];
  }

  MutableSpan<float3> positions = mesh.vert_positions_for_write();
  Array<float3> tmp(verts_num);
  auto apply = [&](const float amount) {
    threading::parallel_for(IndexRange(verts_num), 1024, [&](const IndexRange range) {
      for (const int i : range) {
        const Span<int> vert_neighbours = neighbours.as_span().slice(offsets[i],
                                                                     offsets[i + 1] - offsets[i]);
        if (vert_neighbours.is_empty()) {
          tmp[i] = positions[i];
          continue;
        }
        float3 average = float3(0.0f);
        for (const int other : vert_neighbours) {
          average += positions[other];
        }
        average /= float(vert_neighbours.size());
        tmp[i] = positions[i] + (average - positions[i]) * amount;
      }
    });
    positions.copy_from(tmp);
  };

  for (const int iteration : IndexRange(iterations)) {
    UNUSED_VARS(iteration);
    apply(lambda);
    apply(mu);
  }
  mesh.tag_positions_changed();
}

/**
 * Relax the density field towards the average of its neighbours. Splatting one kernel per particle
 * leaves a ripple at the scale of the particle spacing (the surface looks quilted), and no amount
 * of mesh smoothing can remove it afterwards, because the mesh only samples what the field holds.
 * Voxels outside the field count as the value of the center voxel, so the isosurface does not move
 * outwards or inwards, only the ripples flatten.
 */
static void smooth_density_field(openvdb::FloatGrid &grid,
                                 const int iterations,
                                 const float strength,
                                 const float edge_preserving,
                                 const float value_scale)
{
  if (iterations <= 0 || strength <= 0.0f) {
    return;
  }
  /* Bilateral weight on the value difference: a plain average flattens the particle-scale ripple and
   * a droplet with it, while this only accepts neighbours of similar value, so a thin jet or a
   * separated droplet keeps its own level. `edge_preserving = 0` reproduces the plain average and
   * `value_scale` is the reference amplitude of the field (iso level for density, one radius for a
   * distance field). */
  const float range_sigma = std::max(edge_preserving, 0.0f) * std::max(value_scale, 1e-3f);
  const float inv_range_sigma_sq = range_sigma > 0.0f ? 1.0f / (range_sigma * range_sigma) : 0.0f;
  static const openvdb::Coord offsets[6] = {
      openvdb::Coord(1, 0, 0),
      openvdb::Coord(-1, 0, 0),
      openvdb::Coord(0, 1, 0),
      openvdb::Coord(0, -1, 0),
      openvdb::Coord(0, 0, 1),
      openvdb::Coord(0, 0, -1),
  };
  openvdb::FloatGrid::Accessor accessor = grid.getAccessor();
  Vector<openvdb::Coord> coords;
  coords.reserve(int(grid.activeVoxelCount()));
  for (auto iter = grid.cbeginValueOn(); iter; ++iter) {
    coords.append(iter.getCoord());
  }
  Array<float> values(coords.size());
  for (const int iteration : IndexRange(iterations)) {
    UNUSED_VARS(iteration);
    for (const int i : coords.index_range()) {
      const openvdb::Coord coord = coords[i];
      const float center = accessor.getValue(coord);
      float sum = 0.0f;
      float weight_sum = 0.0f;
      for (const openvdb::Coord &offset : offsets) {
        const openvdb::Coord other = coord + offset;
        const float value = accessor.isValueOn(other) ? accessor.getValue(other) : center;
        float weight = 1.0f;
        if (inv_range_sigma_sq > 0.0f) {
          const float difference = value - center;
          weight = std::exp(-difference * difference * inv_range_sigma_sq);
        }
        sum += value * weight;
        weight_sum += weight;
      }
      const float average = weight_sum > 0.0f ? sum / weight_sum : center;
      values[i] = center + (average - center) * strength;
    }
    for (const int i : coords.index_range()) {
      accessor.setValue(coords[i], values[i]);
    }
  }
}

#endif /* WITH_OPENVDB */

static const EnumPropertyItem surface_mode_items[] = {
    {int(SurfaceMode::Density),
     "DENSITY",
     0,
     N_("Density"),
     N_("Sum the particle kernels into a density field and extract an iso surface. Compact volumes "
        "get the smoothest result; thin sheets keep one bump per particle")},
    {int(SurfaceMode::Distance),
     "DISTANCE",
     0,
     N_("Distance"),
     N_("Union of the particle spheres, smoothed as a level set. Flattens sheets and keeps "
        "droplets round; Iso Level 0.5 puts the surface at the particle radius, larger values push "
        "it outwards. Anisotropic and Edge Preserving do not apply in this mode")},
    {0, nullptr, 0, nullptr, nullptr},
};

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Menu>("Mode"_ustr)
      .static_items(surface_mode_items)
      .default_value(MenuValue(int(SurfaceMode::Density)))
      .description("How the surface is reconstructed");
  b.add_input<decl::Geometry>("Points"_ustr)
      .supported_type({GeometryComponent::Type::Mesh,
                       GeometryComponent::Type::PointCloud,
                       GeometryComponent::Type::Curve})
      .translation_context(BLT_I18NCONTEXT_ID_POINTCLOUD)
      .is_default_link_socket()
      .description("Particles of the fluid, usually the `Geometry` output of `Puncliga Dynamics`");
  b.add_input<decl::Float>("Voxel Size"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description("Resolution of the density field. 0 uses half of the particle radius");
  b.add_input<decl::Float>("Particle Radius"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description(
          "Radius of the particles. 0 uses the `radius` attribute of the input (0.05 m when the "
          "geometry has none); any other value overrides the attribute. The kernel support is this "
          "times `Kernel Radius`");
  b.add_input<decl::Float>("Iso Level"_ustr)
      .default_value(0.6f)
      .min(0.01f)
      .max(2.0f)
      .description(
          "Density of the surface, as a fraction of the rest density of the particles. 0.5 puts "
          "it through the centers of the outermost particles, higher values make it fuller");
  b.add_input<decl::Int>("Smoothing"_ustr)
      .default_value(2)
      .min(0)
      .max(60)
      .description(
          "Iterations of the volume preserving Taubin smoothing of the extracted surface. Removes "
          "the staircase look of the marching cubes");
  b.add_input<decl::Int>("Density Smoothing"_ustr)
      .default_value(2)
      .min(0)
      .max(100)
      .description(
          "Relaxation passes on the density field. The surface of the particles is quilted at the "
          "scale of the particle spacing without this; the mesh smoothing above only removes the "
          "smaller marching cubes steps");
  b.add_input<decl::Float>("Kernel Radius"_ustr)
      .default_value(2.0f)
      .min(1.0f)
      .max(4.0f)
      .description(
          "Support of one particle's kernel, in particle radii. The kernel has to cover about two "
          "particle spacings; raise it when the surface still shows one bump per particle");
  b.add_input<decl::Bool>("Anisotropic"_ustr)
      .default_value(false)
      .description(
          "Fit each particle's kernel to its neighbourhood (Yu & Turk): flat across a sheet, round "
          "inside a droplet. Smooths sheets and waterfalls without rounding off droplets or jets, "
          "at the cost of a neighbourhood query per particle");
  b.add_input<decl::Float>("Edge Preserving"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .max(1.0f)
      .description(
          "Bilateral weighting of the density relaxation: 0 is a plain average, 1 only averages "
          "neighbours of similar density so thin sheets and droplets keep their own level");
  b.add_input<decl::Float>("Adaptivity"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .max(1.0f)
      .description("Simplification of the surface: 0 keeps every triangle, 1 simplifies the most");
  b.add_input<decl::Bool>("Smooth Shading"_ustr)
      .default_value(true)
      .description("Shade the surface smooth instead of flat");
  b.add_output<decl::Geometry>("Mesh"_ustr);
  b.add_output<decl::Geometry>("Density"_ustr)
      .description(
          "The density field as a volume, 1.0 is the rest density (for rendering or "
          "debugging)");
}

static void node_geo_exec(GeoNodeExecParams params)
{
#ifdef WITH_OPENVDB
  const GeometrySet geometry = params.extract_input<GeometrySet>("Points"_ustr);
  const int mode = params.extract_input<MenuValue>("Mode"_ustr).value;
  const bool distance_mode = mode == int(SurfaceMode::Distance);
  /* 0 means "use the `radius` attribute of the input"; a value overrides it, which is what makes the
   * socket visible in the node instead of being ignored by attribute carrying inputs. */
  const float radius_override = std::max(0.0f,
                                         params.extract_input<float>("Particle Radius"_ustr));
  const float voxel_size_input = std::max(0.0f, params.extract_input<float>("Voxel Size"_ustr));
  const float iso_level = std::max(0.0f, params.extract_input<float>("Iso Level"_ustr));
  const int smoothing = std::max(0, params.extract_input<int>("Smoothing"_ustr));
  const int density_smoothing = std::max(0, params.extract_input<int>("Density Smoothing"_ustr));
  const float radius_scale = math::clamp(params.extract_input<float>("Kernel Radius"_ustr),
                                         1.0f,
                                         4.0f);
  const bool anisotropic = params.extract_input<bool>("Anisotropic"_ustr);
  const float edge_preserving = math::clamp(params.extract_input<float>("Edge Preserving"_ustr),
                                            0.0f,
                                            1.0f);
  const float adaptivity = math::clamp(params.extract_input<float>("Adaptivity"_ustr), 0.0f, 1.0f);
  const bool smooth_shading = params.extract_input<bool>("Smooth Shading"_ustr);

  Vector<FluidParticle> particles;
  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Curve})
  {
    const GeometryComponent *component = geometry.get_component(type);
    if (component == nullptr) {
      continue;
    }
    gather_particles_from_component(*component, radius_override, particles);
  }
  if (particles.is_empty()) {
    params.set_default_remaining_outputs();
    return;
  }

  double radius_sum = 0.0;
  for (const FluidParticle &particle : particles) {
    radius_sum += particle.radius;
  }
  const float mean_radius = float(radius_sum / double(particles.size()));
  if (mean_radius <= 0.0f) {
    params.set_default_remaining_outputs();
    return;
  }
  const float voxel_size = voxel_size_input > 0.0f ? voxel_size_input : 0.5f * mean_radius;
  const double normalizer = lattice_kernel_sum(mean_radius, radius_scale);
  if (std::getenv("BLENDER_FLUID_SURFACE_DEBUG") != nullptr) {
    printf("FLUID_SURFACE_DEBUG %ld particles mean_radius=%f voxel_size=%f normalizer=%f\n",
           long(particles.size()),
           mean_radius,
           voxel_size,
           normalizer);
  }

  /* The splat is done into one grid per bucket of particles and the grids are merged afterwards,
   * because an OpenVDB grid is not thread safe. One bucket per parallel task, so no two tasks ever
   * touch the same grid. */
  Array<KernelShape> kernel_shapes;
  if (anisotropic) {
    build_kernel_shapes(particles.as_span(), mean_radius, kernel_shapes);
  }

  const int bucket_num = std::clamp(BLI_system_thread_count(), 1, 32);
  const int64_t particles_num = particles.size();
  const int64_t bucket_size = (particles_num + bucket_num - 1) / bucket_num;

  openvdb::FloatGrid::Ptr grid;
  float iso_value = iso_level;
  if (distance_mode) {
    /* Distance mode: union of the particle spheres, then OpenVDB's level set filter, which smooths
     * the field *and* re-normalises it, so the surface is smoothed without collapsing. `Iso Level`
     * becomes a metric offset: 0.5 puts the surface at the particle radius, larger values push it
     * outwards like a larger density iso level does. */
    const float band = radius_scale * mean_radius;
    Array<openvdb::FloatGrid::Ptr> bucket_grids(bucket_num);
    for (const int i : IndexRange(bucket_num)) {
      bucket_grids[i] = openvdb::FloatGrid::create(double(band));
      bucket_grids[i]->setTransform(
          openvdb::math::Transform::createLinearTransform(double(voxel_size)));
    }
    threading::parallel_for(IndexRange(bucket_num), 1, [&](const IndexRange range) {
      for (const int bucket_i : range) {
        const int64_t start = bucket_i * bucket_size;
        const int64_t end = std::min(start + bucket_size, particles_num);
        if (start >= end) {
          continue;
        }
        splat_union_sdf(particles.as_span().slice(start, end - start),
                        radius_scale,
                        voxel_size,
                        *bucket_grids[bucket_i]);
      }
    });
    grid = bucket_grids[0];
    {
      openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
      for (const int bucket_i : IndexRange(1, bucket_num - 1)) {
        for (auto iter = bucket_grids[bucket_i]->cbeginValueOn(); iter; ++iter) {
          const openvdb::Coord coord = iter.getCoord();
          accessor.setValue(coord, std::min(accessor.getValue(coord), iter.getValue()));
        }
        bucket_grids[bucket_i].reset();
      }
    }
    grid->setGridClass(openvdb::GRID_LEVEL_SET);
    grid->tree().prune();
    if (grid->activeVoxelCount() > 0) {
      openvdb::tools::LevelSetFilter<openvdb::FloatGrid> filter(*grid);
      for (const int iteration : IndexRange(density_smoothing)) {
        UNUSED_VARS(iteration);
        filter.gaussian(1);
      }
    }
    iso_value = (iso_level - 0.5f) * 2.0f * mean_radius;
  }
  else {
    Array<openvdb::FloatGrid::Ptr> bucket_grids(bucket_num);
    for (const int i : IndexRange(bucket_num)) {
      bucket_grids[i] = openvdb::FloatGrid::create(0.0f);
      bucket_grids[i]->setTransform(
          openvdb::math::Transform::createLinearTransform(double(voxel_size)));
    }
    threading::parallel_for(IndexRange(bucket_num), 1, [&](const IndexRange range) {
      for (const int bucket_i : range) {
        const int64_t start = bucket_i * bucket_size;
        const int64_t end = std::min(start + bucket_size, particles_num);
        if (start >= end) {
          continue;
        }
        Span<KernelShape> shapes;
        if (!kernel_shapes.is_empty()) {
          shapes = kernel_shapes.as_span().slice(start, end - start);
        }
        splat_particles(particles.as_span().slice(start, end - start),
                        shapes,
                        radius_scale,
                        voxel_size,
                        normalizer,
                        *bucket_grids[bucket_i]);
      }
    });
    grid = bucket_grids[0];
    {
      openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
      for (const int bucket_i : IndexRange(1, bucket_num - 1)) {
        for (auto iter = bucket_grids[bucket_i]->cbeginValueOn(); iter; ++iter) {
          const openvdb::Coord coord = iter.getCoord();
          accessor.setValue(coord, accessor.getValue(coord) + iter.getValue());
        }
        bucket_grids[bucket_i].reset();
      }
    }
    grid->setGridClass(openvdb::GRID_FOG_VOLUME);
  }
  if (!distance_mode) {
    grid->tree().prune();
    smooth_density_field(*grid, density_smoothing, 0.5f, edge_preserving, iso_level);
  }

  Mesh *mesh = nullptr;
  if (grid->activeVoxelCount() > 0) {
    mesh = bke::volume_grid_to_mesh(*grid, iso_value, adaptivity);
  }
  if (mesh == nullptr) {
    mesh = BKE_mesh_new_nomain(0, 0, 0, 0);
  }
  smooth_positions_taubin(*mesh, smoothing, 0.5f, -0.53f);
  bke::mesh_smooth_set(*mesh, smooth_shading);

  GeometrySet mesh_set;
  mesh_set.replace_mesh(mesh);
  params.set_output("Mesh"_ustr, std::move(mesh_set));

  GeometrySet volume_set;
  if (grid->activeVoxelCount() > 0) {
    Volume *volume = BKE_id_new_nomain<Volume>(nullptr);
    BKE_volume_grid_add_vdb(*volume, "density", grid);
    volume_set.replace_volume(volume);
  }
  params.set_output("Density"_ustr, std::move(volume_set));
#else
  params.error_message_add(NodeWarningType::Error,
                           TIP_("Fluid Surface requires a build with OpenVDB"));
  params.set_default_remaining_outputs();
#endif
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeFluidSurface"_ustr);
  ntype.ui_name = "Fluid Surface";
  ntype.ui_description =
      "Rebuild a surface from fluid particles by extracting the isosurface of their density field "
      "(SplashSurf style)";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_fluid_surface_cc
