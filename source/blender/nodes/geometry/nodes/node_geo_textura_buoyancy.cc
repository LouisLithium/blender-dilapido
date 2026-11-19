/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Textura Buoyancy`: the fluid to rigid body half of the `Textura` / `RBD` coupling.
 *
 * The MPM fluid is a particle cloud, so the force on a body is derived from a coarse field built
 * from the particles every frame:
 *
 * - each particle deposits its volume (and mass weighted momentum) into a dense grid,
 * - the grid holds a volume fraction (`occupancy`, 0 = air, 1 = full material) and a velocity,
 * - the body is sampled with a lattice of points inside its mesh; each submerged sample carries
 *   `volume * occupancy` of displaced material,
 * - the samples apply buoyancy (Archimedes) and drag (relative to the local fluid velocity), and
 *   the forces are summed into `external_force` / `external_torque` on the instance domain, which
 *   is exactly what `Rigid Body Solver` reads.
 *
 * The other direction (the body pushing the fluid) is not done here: the asset realizes the pieces
 * and hands them to `Textura Solver` as a mesh collider, so the hull displaces the water. Between
 * the two, one frame of lag remains, like the `Ignis RBD Dynamics` staggered scheme.
 *
 * Sampling the body volume (instead of a single point at its centre) is what makes the result a
 * *boat*: buoyancy acts at the centre of the submerged volume, so the waterline produces the
 * righting torque, and the velocity differences across the hull damp roll, pitch and yaw.
 */

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <limits>

#include "BLI_array.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_vector.hh"
#include "BLI_span.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_bvhutils.hh"
#include "BKE_curves.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_physics_bundles.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_textura_buoyancy_cc {

namespace {

constexpr int MAX_GRID_CELLS = 262144;
constexpr int MAX_TOTAL_BODY_SAMPLES = 65536;
constexpr int GRID_MARGIN_CELLS = 3;
/** How far the free surface height is extrapolated into columns without particles (the hull). */
constexpr int HEIGHT_FILL_LAYERS = 32;

/* -------------------------------------------------------------------- */
/** \name Fluid particles
 * \{ */

struct FluidParticles {
  Vector<float3> positions;
  Vector<float3> velocities;
  Vector<float> volumes;
  Vector<float> weights;
  bool any = false;
};

template<typename T>
VArray<T> lookup_or_default(const bke::AttributeAccessor &attributes,
                            const StringRef name,
                            const AttrDomain domain,
                            const T &fallback,
                            const int size)
{
  const VArray<T> values = *attributes.lookup<T>(name, domain);
  if (values) {
    return values;
  }
  return VArray<T>::from_single(fallback, size);
}

void read_fluid_component(const GeometryComponent &component, FluidParticles &r_particles)
{
  const AttrDomain domain = AttrDomain::Point;
  const int count = component.attribute_domain_size(domain);
  if (count == 0) {
    return;
  }
  const bke::AttributeAccessor attributes = *component.attributes();
  const VArray<float3> positions = lookup_or_default<float3>(
      attributes, "position", domain, float3(0.0f), count);
  const VArray<float3> velocities = lookup_or_default<float3>(
      attributes, "velocity", domain, float3(0.0f), count);
  const VArray<float> volumes = lookup_or_default<float>(
      attributes, "textura_volume", domain, 0.0f, count);
  const VArray<float> masses = lookup_or_default<float>(attributes, "mass", domain, 0.0f, count);
  const VArray<float> radii = lookup_or_default<float>(attributes, "radius", domain, 0.0f, count);
  const VArray<int> materials = lookup_or_default<int>(
      attributes, "textura_type", domain, 0, count);
  const bool has_material = attributes.contains("textura_type");

  for (const int i : IndexRange(count)) {
    if (has_material && materials[i] != 0) {
      /* Sand and snow do not float a boat. */
      continue;
    }
    float volume = volumes[i];
    if (!(volume > 0.0f)) {
      const float radius = radii[i] > 0.0f ? radii[i] : 0.05f;
      volume = (2.0f * radius) * (2.0f * radius) * (2.0f * radius);
    }
    if (!(volume > 0.0f) || !std::isfinite(volume)) {
      continue;
    }
    const float3 &position = positions[i];
    if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
      continue;
    }
    float weight = masses[i];
    if (!(weight > 0.0f)) {
      weight = volume;
    }
    r_particles.positions.append(position);
    r_particles.velocities.append(velocities[i]);
    r_particles.volumes.append(volume);
    r_particles.weights.append(weight);
  }
}

void collect_fluid(const GeometrySet &geometry, FluidParticles &r_particles)
{
  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Curve})
  {
    if (!geometry.has(type)) {
      continue;
    }
    const GeometryComponent &component = *geometry.get_component(type);
    read_fluid_component(component, r_particles);
  }
  r_particles.any = !r_particles.positions.is_empty();
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Fluid field
 * \{ */

struct FluidSample {
  float occupancy = 0.0f;
  float3 velocity = float3(0.0f);
  /** Extrapolated free surface height of the column (for debugging and for the caller). */
  float surface_height = 0.0f;
  bool found = false;
};

/**
 * Dense volume fraction / velocity field.
 *
 * One cell is about two particle spacings, so a full cell contains exactly one particle volume per
 * cell volume. The occupancy is clamped to 1 and smoothed, which gives a stable waterline instead of
 * the individual particle bumps.
 */
class FluidField {
  int3 dims_ = int3(0);
  float3 origin_ = float3(0.0f);
  float cell_size_ = 0.0f;
  /* Particle volume per cell volume, before the clamp. */
  Vector<float> occupancy_;
  /* Mass weighted momentum, for the cell velocity. */
  Vector<float3> momentum_;
  Vector<float> weight_;

  /**
   * Free surface height per column, in world units, and the velocity of the top cells.
   *
   * The rigid body is handed to the MPM solver as a collider, so the particles are pushed out of the
   * hull: sampling the raw particle field inside the hull would return air and a floating boat
   * would receive no buoyancy at all. The height field is filled across the hull from the water
   * around it, which is the "undisplaced" free surface the displaced volume has to be compared to.
   * This is an approximation (it assumes a single valued water surface, not an overhanging wave),
   * but it is exactly the information buoyancy needs.
   */
  Vector<float> height_;
  Vector<float3> surface_velocity_;
  Vector<bool> height_filled_;

 public:
  bool build(const FluidParticles &particles,
             const float requested_cell_size,
             const int max_cells = MAX_GRID_CELLS)
  {
    occupancy_.clear();
    momentum_.clear();
    weight_.clear();
    height_.clear();
    surface_velocity_.clear();
    height_filled_.clear();
    dims_ = int3(0);
    if (particles.positions.is_empty()) {
      return false;
    }

    float3 min = particles.positions[0];
    float3 max = particles.positions[0];
    for (const float3 &p : particles.positions) {
      min = math::min(min, p);
      max = math::max(max, p);
    }

    /* Median spacing from the particle volumes: `volume` is the spacing cubed for the MPM solver. */
    Vector<float> spacings;
    spacings.reserve(particles.volumes.size());
    for (const float volume : particles.volumes) {
      spacings.append(std::cbrt(volume));
    }
    std::nth_element(spacings.begin(), spacings.begin() + spacings.size() / 2, spacings.end());
    const float median_spacing = spacings[spacings.size() / 2];

    float cell = requested_cell_size;
    if (!(cell > 0.0f)) {
      cell = 2.0f * median_spacing;
    }
    if (!(cell > 0.0f)) {
      const float3 extent = math::max(max - min, float3(1e-5f));
      cell = std::cbrt(extent.x * extent.y * extent.z / float(particles.positions.size()));
    }
    cell = math::clamp(cell, 1e-3f, 0.5f);

    const float3 extent = math::max(max - min, float3(1e-5f));
    for (int attempt = 0; attempt < 8; attempt++) {
      dims_ = int3(std::max(1, int(std::ceil(extent.x / cell))) + 2 * GRID_MARGIN_CELLS,
                   std::max(1, int(std::ceil(extent.y / cell))) + 2 * GRID_MARGIN_CELLS,
                   std::max(1, int(std::ceil(extent.z / cell))) + 2 * GRID_MARGIN_CELLS);
      const int64_t cells = int64_t(dims_.x) * dims_.y * dims_.z;
      if (cells <= max_cells) {
        break;
      }
      cell *= std::cbrt(double(cells) / double(max_cells));
    }
    const int64_t cells = int64_t(dims_.x) * dims_.y * dims_.z;
    if (cells > max_cells) {
      return false;
    }

    origin_ = min - float3(float(GRID_MARGIN_CELLS) * cell);
    cell_size_ = cell;
    const int cell_count = int(cells);
    const float cell_volume = cell * cell * cell;
    const float inv_cell_volume = 1.0f / cell_volume;

    occupancy_.reinitialize(cell_count);
    occupancy_.fill(0.0f);
    momentum_.reinitialize(cell_count);
    momentum_.fill(float3(0.0f));
    weight_.reinitialize(cell_count);
    weight_.fill(0.0f);

    for (const int i : IndexRange(particles.positions.size())) {
      const float3 local = (particles.positions[i] - origin_) / cell;
      const int base_x = int(std::floor(local.x));
      const int base_y = int(std::floor(local.y));
      const int base_z = int(std::floor(local.z));
      const float fx = local.x - float(base_x);
      const float fy = local.y - float(base_y);
      const float fz = local.z - float(base_z);
      const float volume = particles.volumes[i] * inv_cell_volume;
      const float3 momentum = particles.velocities[i] * particles.weights[i];
      const float weight = particles.weights[i];
      for (int dz = 0; dz < 2; dz++) {
        const int z = base_z + dz;
        if (z < 0 || z >= dims_.z) {
          continue;
        }
        const float wz = dz ? fz : 1.0f - fz;
        for (int dy = 0; dy < 2; dy++) {
          const int y = base_y + dy;
          if (y < 0 || y >= dims_.y) {
            continue;
          }
          const float wy = dy ? fy : 1.0f - fy;
          for (int dx = 0; dx < 2; dx++) {
            const int x = base_x + dx;
            if (x < 0 || x >= dims_.x) {
              continue;
            }
            const float wx = dx ? fx : 1.0f - fx;
            const float w = wx * wy * wz;
            const int index = (z * dims_.y + y) * dims_.x + x;
            occupancy_[index] += volume * w;
            momentum_[index] += momentum * w;
            weight_[index] += weight * w;
          }
        }
      }
    }

    for (const int i : IndexRange(cell_count)) {
      occupancy_[i] = math::clamp(occupancy_[i], 0.0f, 1.0f);
      if (weight_[i] > 0.0f) {
        momentum_[i] /= weight_[i];
      }
    }

    smooth_occupancy();
    extrapolate_velocity(8);
    build_surface_height();
    return true;
  }

  bool valid() const
  {
    return !occupancy_.is_empty();
  }

  float cell_size() const
  {
    return cell_size_;
  }

  FluidSample sample(const float3 &position) const
  {
    FluidSample result;
    if (occupancy_.is_empty()) {
      return result;
    }
    const float3 local = (position - origin_) / cell_size_;
    if (local.x < 0.0f || local.y < 0.0f || local.z < 0.0f || local.x >= float(dims_.x - 1) ||
        local.y >= float(dims_.y - 1) || local.z >= float(dims_.z - 1))
    {
      return result;
    }
    const int base_x = int(std::floor(local.x));
    const int base_y = int(std::floor(local.y));
    const int base_z = int(std::floor(local.z));
    const int column = base_y * dims_.x + base_x;
    if (height_filled_[column]) {
      result.surface_height = height_[column];
    }
    const float fx = local.x - float(base_x);
    const float fy = local.y - float(base_y);
    const float fz = local.z - float(base_z);

    float occupancy = 0.0f;
    float3 velocity(0.0f);
    float velocity_weight = 0.0f;
    for (int dz = 0; dz < 2; dz++) {
      const float wz = dz ? fz : 1.0f - fz;
      for (int dy = 0; dy < 2; dy++) {
        const float wy = dy ? fy : 1.0f - fy;
        for (int dx = 0; dx < 2; dx++) {
          const float wx = dx ? fx : 1.0f - fx;
          const float w = wx * wy * wz;
          const int index = ((base_z + dz) * dims_.y + (base_y + dy)) * dims_.x + (base_x + dx);
          occupancy += occupancy_[index] * w;
          const float vw = weight_[index] * w;
          velocity += momentum_[index] * vw;
          velocity_weight += vw;
        }
      }
    }
    if (velocity_weight > 0.0f) {
      velocity /= velocity_weight;
    }

    /* The particles are pushed out of a rigid collider, so a direct sample inside a hull sees air.
     * Extrapolate the free surface from the water around it: below the local height the material is
     * treated as the undisplaced fluid the hull is floating in. */
    if (occupancy < 0.05f) {
      if (height_filled_[column]) {
        const float surface = height_[column];
        if (position.z < surface) {
          occupancy = math::clamp((surface - position.z) / cell_size_, 0.0f, 1.0f);
          if (velocity_weight <= 0.0f) {
            velocity = surface_velocity_[column];
          }
        }
      }
    }

    result.occupancy = math::clamp(occupancy, 0.0f, 1.0f);
    result.velocity = velocity;
    result.found = true;
    return result;
  }

 private:
  void build_surface_height()
  {
    const int columns = dims_.x * dims_.y;
    height_.reinitialize(columns);
    surface_velocity_.reinitialize(columns);
    height_filled_.reinitialize(columns);
    height_.fill(-FLT_MAX);
    surface_velocity_.fill(float3(0.0f));
    height_filled_.fill(false);

    constexpr float SURFACE_THRESHOLD = 0.15f;
    for (int y = 0; y < dims_.y; y++) {
      for (int x = 0; x < dims_.x; x++) {
        int top = -1;
        for (int z = dims_.z - 1; z >= 0; z--) {
          if (occupancy_[(z * dims_.y + y) * dims_.x + x] > SURFACE_THRESHOLD) {
            top = z;
            break;
          }
        }
        if (top < 0) {
          continue;
        }
        const int column = y * dims_.x + x;
        height_[column] = origin_.z + (float(top) + 0.5f) * cell_size_;
        height_filled_[column] = true;
        float3 velocity(0.0f);
        int samples = 0;
        for (int z = top; z >= std::max(0, top - 2); z--) {
          const int index = (z * dims_.y + y) * dims_.x + x;
          if (weight_[index] <= 0.0f) {
            continue;
          }
          velocity += momentum_[index];
          samples++;
        }
        if (samples > 0) {
          surface_velocity_[column] = velocity / float(samples);
        }
      }
    }

    /* Fill the columns that the hull covers from the water around it. A rigid body is a collider of
     * the MPM solver, so its interior has no particles at all; without this the buoyancy of a
     * floating boat would be zero. The fill is a dilation of the free surface (a single valued
     * height field), which is the right approximation for a pool or the open sea. */
    Vector<float> next_height;
    Vector<float3> next_velocity;
    Vector<bool> next_filled;
    for (int layer = 0; layer < HEIGHT_FILL_LAYERS; layer++) {
      next_height = height_;
      next_velocity = surface_velocity_;
      next_filled = height_filled_;
      bool changed = false;
      for (int y = 0; y < dims_.y; y++) {
        for (int x = 0; x < dims_.x; x++) {
          const int column = y * dims_.x + x;
          if (height_filled_[column]) {
            continue;
          }
          /* Take the highest filled neighbour: the hull sits in a depression of the free surface,
           * and its undisplaced water level is the surrounding surface, not the average of the
           * crater rim (averaging made a boat float a few centimetres too deep). */
          float height_max = -FLT_MAX;
          float3 velocity_sum(0.0f);
          int neighbours = 0;
          for (int dy = -1; dy <= 1; dy++) {
            const int ny = y + dy;
            if (ny < 0 || ny >= dims_.y) {
              continue;
            }
            for (int dx = -1; dx <= 1; dx++) {
              const int nx = x + dx;
              if (nx < 0 || nx >= dims_.x || (dx == 0 && dy == 0)) {
                continue;
              }
              const int neighbour = ny * dims_.x + nx;
              if (!height_filled_[neighbour]) {
                continue;
              }
              height_max = std::max(height_max, height_[neighbour]);
              velocity_sum += surface_velocity_[neighbour];
              neighbours++;
            }
          }
          if (neighbours == 0) {
            continue;
          }
          next_height[column] = height_max;
          next_velocity[column] = velocity_sum / float(neighbours);
          next_filled[column] = true;
          changed = true;
        }
      }
      std::swap(height_, next_height);
      std::swap(surface_velocity_, next_velocity);
      std::swap(height_filled_, next_filled);
      if (!changed) {
        break;
      }
    }
  }

  void smooth_occupancy(const int passes = 1)
  {
    if (passes <= 0 || occupancy_.is_empty()) {
      return;
    }
    Vector<float> next(occupancy_.size());
    const int3 dims = dims_;
    for (int pass = 0; pass < passes; pass++) {
      for (int z = 0; z < dims.z; z++) {
        for (int y = 0; y < dims.y; y++) {
          for (int x = 0; x < dims.x; x++) {
            const int index = (z * dims.y + y) * dims.x + x;
            float sum = 0.0f;
            int count = 0;
            const int offsets[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                       {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
            for (const auto &offset : offsets) {
              const int nx = x + offset[0];
              const int ny = y + offset[1];
              const int nz = z + offset[2];
              if (nx < 0 || ny < 0 || nz < 0 || nx >= dims.x || ny >= dims.y || nz >= dims.z) {
                continue;
              }
              sum += occupancy_[(nz * dims.y + ny) * dims.x + nx];
              count++;
            }
            next[index] = count > 0 ? 0.5f * occupancy_[index] + 0.5f * (sum / float(count)) :
                                      occupancy_[index];
          }
        }
      }
      std::swap(occupancy_, next);
    }
  }

  void extrapolate_velocity(const int layers)
  {
    if (layers <= 0) {
      return;
    }
    const int3 dims = dims_;
    Vector<float3> next_momentum;
    Vector<float> next_weight;
    for (int layer = 0; layer < layers; layer++) {
      next_momentum = momentum_;
      next_weight = weight_;
      bool changed = false;
      for (int z = 0; z < dims.z; z++) {
        for (int y = 0; y < dims.y; y++) {
          for (int x = 0; x < dims.x; x++) {
            const int index = (z * dims.y + y) * dims.x + x;
            if (weight_[index] > 0.0f) {
              continue;
            }
            float3 velocity(0.0f);
            int count = 0;
            const int offsets[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                       {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
            for (const auto &offset : offsets) {
              const int nx = x + offset[0];
              const int ny = y + offset[1];
              const int nz = z + offset[2];
              if (nx < 0 || ny < 0 || nz < 0 || nx >= dims.x || ny >= dims.y || nz >= dims.z) {
                continue;
              }
              const int neighbour = (nz * dims.y + ny) * dims.x + nx;
              if (weight_[neighbour] <= 0.0f) {
                continue;
              }
              velocity += momentum_[neighbour];
              count++;
            }
            if (count == 0) {
              continue;
            }
            next_momentum[index] = velocity / float(count);
            next_weight[index] = 1.0f;
            changed = true;
          }
        }
      }
      std::swap(momentum_, next_momentum);
      std::swap(weight_, next_weight);
      if (!changed) {
        break;
      }
    }
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Body templates
 * \{ */

struct BodySample {
  float3 local_position = float3(0.0f);
  float volume = 0.0f;
};

struct BodyTemplate {
  Vector<BodySample> samples;
  /** Largest squared distance of a sample from the body origin, for the angular drag. */
  float radius_squared = 0.0f;
  bool valid = false;
};

/** Nearest point on the mesh; `nullptr` callbacks would leave `nearest.co` untouched. */
bool mesh_nearest(const bke::BVHTreeFromMesh &bvh,
                  const float3 &position,
                  BVHTreeNearest &r_nearest)
{
  if (bvh.tree == nullptr) {
    return false;
  }
  r_nearest.index = -1;
  r_nearest.dist_sq = FLT_MAX;
  const float co[3] = {position.x, position.y, position.z};
  bke::BVHTreeFromMesh &bvh_mutable = const_cast<bke::BVHTreeFromMesh &>(bvh);
  return BLI_bvhtree_find_nearest(
             bvh.tree, co, &r_nearest, bvh.nearest_callback, &bvh_mutable) >= 0;
}

bool mesh_contains(const bke::BVHTreeFromMesh &bvh,
                   const float3 &position,
                   const float orientation)
{
  BVHTreeNearest nearest;
  if (!mesh_nearest(bvh, position, nearest)) {
    return false;
  }
  const float3 delta = position - float3(nearest.co);
  if (math::length_squared(delta) <= 1.0e-8f) {
    return true;
  }
  const int3 tri = bvh.corner_tris[nearest.index];
  const float3 a = bvh.vert_positions[bvh.corner_verts[tri[0]]];
  const float3 b = bvh.vert_positions[bvh.corner_verts[tri[1]]];
  const float3 c = bvh.vert_positions[bvh.corner_verts[tri[2]]];
  return orientation * math::dot(delta, math::cross(b - a, c - a)) < 0.0f;
}

bool mesh_is_closed(const Mesh &mesh)
{
  const int edges_num = mesh.edges_num;
  if (edges_num == 0) {
    return false;
  }
  Array<int> uses(edges_num, 0);
  for (const int edge : mesh.corner_edges()) {
    uses[edge]++;
  }
  for (const int use : uses) {
    if (use != 2) {
      return false;
    }
  }
  return true;
}

float mesh_signed_volume(const Mesh &mesh)
{
  const Span<float3> verts = mesh.vert_positions();
  const Span<int> corner_verts = mesh.corner_verts();
  float volume = 0.0f;
  for (const int3 &tri : mesh.corner_tris()) {
    const float3 &a = verts[corner_verts[tri[0]]];
    const float3 &b = verts[corner_verts[tri[1]]];
    const float3 &c = verts[corner_verts[tri[2]]];
    volume += math::dot(a, math::cross(b, c));
  }
  return volume / 6.0f;
}

BodyTemplate make_sphere_template(const float radius)
{
  BodyTemplate result;
  const float r = std::max(radius, 1e-4f);
  constexpr float pi = 3.14159265358979323846f;
  const float volume = (4.0f / 3.0f) * pi * r * r * r;
  const float sample_volume = volume / 7.0f;
  result.samples.append({float3(0.0f), sample_volume});
  for (const float3 &axis : {float3(1, 0, 0),
                             float3(-1, 0, 0),
                             float3(0, 1, 0),
                             float3(0, -1, 0),
                             float3(0, 0, 1),
                             float3(0, 0, -1)})
  {
    result.samples.append({axis * (0.5f * r), sample_volume});
  }
  result.radius_squared = 0.25f * r * r;
  result.valid = true;
  return result;
}

/**
 * Builds a lattice of points inside a mesh.
 *
 * A single point at the body centre cannot feel the waterline: a half submerged hull would see
 * either full buoyancy or none. A volume lattice gives a submerged volume (and its centroid) that
 * changes continuously as the body moves through the surface. The lattice is normalized to the
 * signed volume of the mesh, so a coarse sample spacing still displaces the right amount of water.
 */
BodyTemplate build_mesh_template(const Mesh &mesh,
                                 const float requested_spacing,
                                 const int max_samples)
{
  BodyTemplate result;
  const Span<float3> verts = mesh.vert_positions();
  if (verts.size() < 4) {
    return result;
  }
  const bke::BVHTreeFromMesh bvh = mesh.bvh_corner_tris();
  if (bvh.tree == nullptr) {
    return result;
  }
  const float signed_volume = mesh_signed_volume(mesh);
  const float orientation = signed_volume < 0.0f ? -1.0f : 1.0f;
  const float mesh_volume = std::abs(signed_volume);
  const bool closed = mesh_is_closed(mesh);

  float3 min = verts[0];
  float3 max = verts[0];
  for (const float3 &p : verts) {
    min = math::min(min, p);
    max = math::max(max, p);
  }
  const float3 extent = math::max(max - min, float3(1e-6f));
  const float max_extent = std::max({extent.x, extent.y, extent.z});

  float spacing = requested_spacing;
  if (!(spacing > 0.0f)) {
    spacing = std::max(max_extent / 40.0f, 1e-3f);
  }
  const int limit = std::max(max_samples, 8);
  for (int attempt = 0; attempt < 12; attempt++) {
    const int3 counts(std::max(1, int(std::ceil(extent.x / spacing))),
                      std::max(1, int(std::ceil(extent.y / spacing))),
                      std::max(1, int(std::ceil(extent.z / spacing))));
    const int64_t total = int64_t(counts.x) * counts.y * counts.z;
    if (total <= limit) {
      break;
    }
    spacing *= std::cbrt(double(total) / double(limit)) * 1.01;
  }

  const int3 counts(std::max(1, int(std::ceil(extent.x / spacing))),
                    std::max(1, int(std::ceil(extent.y / spacing))),
                    std::max(1, int(std::ceil(extent.z / spacing))));
  for (int z = 0; z < counts.z; z++) {
    for (int y = 0; y < counts.y; y++) {
      for (int x = 0; x < counts.x; x++) {
        const float3 p = min + float3((float(x) + 0.5f) * spacing,
                                      (float(y) + 0.5f) * spacing,
                                      (float(z) + 0.5f) * spacing);
        if (mesh_contains(bvh, p, orientation)) {
          result.samples.append({p, 0.0f});
        }
      }
    }
  }

  if (result.samples.is_empty()) {
    /* A degenerate or inverted mesh: keep the centre as a last resort instead of dropping the body
     * from the coupling entirely. */
    result.samples.append({float3(0.0f), 0.0f});
  }

  const float sample_volume = (closed && mesh_volume > 1e-9f) ?
                                  mesh_volume / float(result.samples.size()) :
                                  spacing * spacing * spacing;
  for (BodySample &sample : result.samples) {
    sample.volume = sample_volume;
    result.radius_squared = std::max(result.radius_squared,
                                     math::length_squared(sample.local_position));
  }
  result.valid = true;
  return result;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Coupling
 * \{ */

struct BodyCoupling {
  float3 force = float3(0.0f);
  float3 torque = float3(0.0f);
  float submerged_volume = 0.0f;
  float water_height = 0.0f;
};

BodyCoupling evaluate_body(const BodyTemplate &body_template,
                           const float4x4 &transform,
                           const float3 &linear_velocity,
                           const float3 &angular_velocity,
                           const FluidField &field,
                           const float3 &buoyancy_direction,
                           const float gravity,
                           const float fluid_density,
                           const float buoyancy_scale,
                           const float drag,
                           const float angular_drag,
                           const float max_relative_speed)
{
  BodyCoupling result;
  if (!body_template.valid || body_template.samples.is_empty()) {
    return result;
  }
  const float3 origin = float3(transform[3][0], transform[3][1], transform[3][2]);
  result.water_height = field.sample(origin).surface_height;
  float3 centroid(0.0f);
  float3 fluid_velocity(0.0f);
  for (const BodySample &sample : body_template.samples) {
    const float3 world = math::transform_point(transform, sample.local_position);
    const FluidSample fluid = field.sample(world);
    if (!fluid.found || fluid.occupancy <= 1e-4f) {
      continue;
    }
    const float displaced = sample.volume * fluid.occupancy;
    if (displaced <= 0.0f) {
      continue;
    }
    result.submerged_volume += displaced;
    centroid += world * displaced;
    fluid_velocity += fluid.velocity * displaced;

    const float displaced_mass = fluid_density * displaced;
    const float3 buoyancy_force = buoyancy_direction *
                                  (displaced_mass * gravity * buoyancy_scale);
    result.force += buoyancy_force;
    result.torque += math::cross(world - origin, buoyancy_force);
  }
  if (result.submerged_volume <= 1e-8f) {
    return result;
  }

  /* Drag is evaluated once for the whole displaced volume, at its centroid. Applying it per sample
   * gives a much stronger torque: a splash produces very different velocities across the hull, and
   * the resulting torque (r x F over hundreds of samples) can spin a light boat at tens of rad/s.
   * The mix of the sample velocities still lets a wave carry the body, while the explicit angular
   * drag below handles rotation. */
  const float3 center = centroid / result.submerged_volume;
  const float3 average_fluid_velocity = fluid_velocity / result.submerged_volume;
  const float3 r_center = center - origin;
  const float3 point_velocity = linear_velocity + math::cross(angular_velocity, r_center);
  float3 relative = average_fluid_velocity - point_velocity;
  const float speed = math::length(relative);
  const float max_speed = std::max(max_relative_speed, 1e-3f);
  if (speed > max_speed) {
    relative *= max_speed / speed;
  }
  const float displaced_mass = fluid_density * result.submerged_volume;
  /* `relative = v_fluid - v_body`, so the drag force on the body is +c * relative: a body moving
   * through still water is slowed down, and a moving fluid carries the body with it. The sign was
   * inverted in the first version, which made the water *accelerate* the hull and launched it. */
  const float3 drag_force = relative * (displaced_mass * drag);
  result.force += drag_force;
  result.torque += math::cross(r_center, drag_force);

  const float angular_inertia = displaced_mass * std::max(body_template.radius_squared,
                                                          1e-6f);
  result.torque -= angular_velocity * (angular_drag * angular_inertia);
  return result;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Node
 * \{ */

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Fluid"_ustr)
      .description("The particles of `Textura Solver` (`Init Textura Data` writes the volumes and "
                   "velocities this needs)");
  b.add_input<decl::Geometry>("Pieces"_ustr)
      .description("The rigid bodies: instances (one mesh per body) or a point cloud (spheres)");
  b.add_input<decl::Vector>("Gravity"_ustr)
      .default_value(float3(0.0f, 0.0f, -9.81f))
      .subtype(PROP_ACCELERATION)
      .description("Gravity of the rigid body solver; it sets both the buoyancy direction and its "
                   "magnitude");
  b.add_input<decl::Float>("Fluid Density"_ustr)
      .default_value(1000.0f)
      .min(0.0f)
      .description("Density of the displaced fluid in kg/m^3 (water is 1000)");
  b.add_input<decl::Float>("Buoyancy"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .description("Scale of the Archimedes force. 0 disables the fluid -> body coupling");
  b.add_input<decl::Float>("Drag"_ustr)
      .default_value(2.0f)
      .min(0.0f)
      .description("Linear drag of the submerged body, in 1/s. It is what makes the body follow the "
                   "waves; 0 disables it");
  b.add_input<decl::Float>("Angular Drag"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .description("Angular damping of the submerged body, in 1/s. It keeps splashes from spinning a "
                   "light boat; 0 disables it");
  b.add_input<decl::Float>("Max Relative Speed"_ustr)
      .default_value(10.0f)
      .min(0.0f)
      .subtype(PROP_VELOCITY)
      .description("Clamp of |v_fluid - v_body|, so a splash cannot produce a huge force");
  b.add_input<decl::Float>("Grid Cell Size"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description("Cell size of the fluid sampling grid. 0 derives it from the particle spacing");
  b.add_input<decl::Float>("Sample Spacing"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description("Spacing of the body samples. 0 derives it from the fluid grid and the body "
                   "size");
  b.add_input<decl::Int>("Max Samples"_ustr)
      .default_value(2048)
      .min(8)
      .max(65536)
      .description("Upper limit of interior samples per body");
  b.add_input<decl::Float>("Max Force"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .description("Clamp of the total force on one body, in N. 0 means no clamp");
  b.add_output<decl::Geometry>("Pieces"_ustr).propagate_all();
}

static void write_instance_force(bke::Instances &instances,
                                 const Span<BodyCoupling> couplings,
                                 const int body_offset)
{
  bke::MutableAttributeAccessor attributes = instances.attributes_for_write();
  bke::SpanAttributeWriter<float3> forces = attributes.lookup_or_add_for_write_span<float3>(
      "external_force", AttrDomain::Instance);
  bke::SpanAttributeWriter<float3> torques = attributes.lookup_or_add_for_write_span<float3>(
      "external_torque", AttrDomain::Instance);
  bke::SpanAttributeWriter<float> submerged = attributes.lookup_or_add_for_write_span<float>(
      "textura_submerged_volume", AttrDomain::Instance);
  bke::SpanAttributeWriter<float> water_height = attributes.lookup_or_add_for_write_span<float>(
      "textura_water_height", AttrDomain::Instance);
  const int count = instances.instances_num();
  for (const int i : IndexRange(count)) {
    const BodyCoupling &coupling = couplings[body_offset + i];
    forces.span[i] = coupling.force;
    torques.span[i] = coupling.torque;
    submerged.span[i] = coupling.submerged_volume;
    water_height.span[i] = coupling.water_height;
  }
  forces.finish();
  torques.finish();
  submerged.finish();
  water_height.finish();
}

static void write_point_force(PointCloud &pointcloud,
                              const Span<BodyCoupling> couplings,
                              const int body_offset)
{
  bke::MutableAttributeAccessor attributes = pointcloud.attributes_for_write();
  bke::SpanAttributeWriter<float3> forces = attributes.lookup_or_add_for_write_span<float3>(
      "external_force", AttrDomain::Point);
  bke::SpanAttributeWriter<float3> torques = attributes.lookup_or_add_for_write_span<float3>(
      "external_torque", AttrDomain::Point);
  bke::SpanAttributeWriter<float> submerged = attributes.lookup_or_add_for_write_span<float>(
      "textura_submerged_volume", AttrDomain::Point);
  bke::SpanAttributeWriter<float> water_height = attributes.lookup_or_add_for_write_span<float>(
      "textura_water_height", AttrDomain::Point);
  const int count = int(pointcloud.positions().size());
  for (const int i : IndexRange(count)) {
    const BodyCoupling &coupling = couplings[body_offset + i];
    forces.span[i] = coupling.force;
    torques.span[i] = coupling.torque;
    submerged.span[i] = coupling.submerged_volume;
    water_height.span[i] = coupling.water_height;
  }
  forces.finish();
  torques.finish();
  submerged.finish();
  water_height.finish();
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet pieces = params.extract_input<GeometrySet>("Pieces"_ustr);
  GeometrySet fluid = params.extract_input<GeometrySet>("Fluid"_ustr);
  const float3 gravity = params.extract_input<float3>("Gravity"_ustr);
  const float gravity_magnitude = math::length(gravity);
  const float3 buoyancy_direction = gravity_magnitude > 1e-6f ? -gravity / gravity_magnitude :
                                                                  float3(0.0f, 0.0f, 1.0f);
  const float fluid_density = std::max(params.extract_input<float>("Fluid Density"_ustr), 0.0f);
  const float buoyancy_scale = std::max(params.extract_input<float>("Buoyancy"_ustr), 0.0f);
  const float drag = std::max(params.extract_input<float>("Drag"_ustr), 0.0f);
  const float angular_drag = std::max(params.extract_input<float>("Angular Drag"_ustr), 0.0f);
  const float max_relative_speed = std::max(
      params.extract_input<float>("Max Relative Speed"_ustr), 0.0f);
  const float requested_cell_size = std::max(params.extract_input<float>("Grid Cell Size"_ustr),
                                             0.0f);
  const float requested_spacing = std::max(params.extract_input<float>("Sample Spacing"_ustr),
                                           0.0f);
  const int max_samples = std::max(params.extract_input<int>("Max Samples"_ustr), 8);
  const float max_force = std::max(params.extract_input<float>("Max Force"_ustr), 0.0f);

  FluidParticles particles;
  collect_fluid(fluid, particles);

  FluidField field;
  if (particles.any) {
    field.build(particles, requested_cell_size);
  }

  Vector<BodyCoupling> couplings;
  if (pieces.has_instances()) {
    bke::Instances &instances = *pieces.get_instances_for_write();
    const int body_count = instances.instances_num();
    if (body_count > 0) {
      couplings.reinitialize(body_count);
      const Span<float4x4> transforms = instances.transforms();
      const Span<int> reference_handles = instances.reference_handles();
      const Span<bke::InstanceReference> references = instances.references();
      const bke::AttributeAccessor instance_attributes = instances.attributes();
      const VArray<float3> velocities = instance_attributes
                                            .lookup_or_default<float3>(
                                                "velocity", AttrDomain::Instance, float3(0.0f))
                                            .varray;
      const VArray<float3> angular_velocities = instance_attributes
                                                    .lookup_or_default<float3>("angular_velocity",
                                                                               AttrDomain::Instance,
                                                                               float3(0.0f))
                                                    .varray;
      const VArray<float> radii = instance_attributes
                                      .lookup_or_default<float>(
                                          "radius", AttrDomain::Instance, 0.1f)
                                      .varray;

      Vector<BodyTemplate> templates(references.size());
      Array<bool> template_ready(references.size(), false);
      const float auto_spacing = requested_spacing > 0.0f ? requested_spacing :
                                  (field.valid() ? field.cell_size() * 0.5f : 0.0f);
      const int per_body_limit = std::max(
          8,
          std::min(max_samples,
                   MAX_TOTAL_BODY_SAMPLES / std::max(body_count, 1)));

      for (const int i : IndexRange(body_count)) {
        const int reference_i = reference_handles[i];
        const float3 linear_velocity = velocities ? velocities[i] : float3(0.0f);
        const float3 angular_velocity = angular_velocities ? angular_velocities[i] :
                                                             float3(0.0f);
        if (reference_i >= 0 && reference_i < references.size()) {
          const bke::InstanceReference &reference = references[reference_i];
          if (reference.type() == bke::InstanceReference::Type::GeometrySet) {
            const Mesh *mesh = reference.geometry_set().get_mesh();
            if (mesh != nullptr) {
              if (!template_ready[reference_i]) {
                templates[reference_i] = build_mesh_template(
                    *mesh, auto_spacing, per_body_limit);
                template_ready[reference_i] = true;
              }
              if (templates[reference_i].valid) {
                couplings[i] = evaluate_body(templates[reference_i],
                                             transforms[i],
                                             linear_velocity,
                                             angular_velocity,
                                             field,
                                             buoyancy_direction,
                                             gravity_magnitude,
                                             fluid_density,
                                             buoyancy_scale,
                                             drag,
                                             angular_drag,
                                             max_relative_speed);
                continue;
              }
            }
          }
        }
        /* No usable mesh (an object/collection reference, or a body without geometry): use the
         * instance radius as a sphere. */
        couplings[i] = evaluate_body(make_sphere_template(std::max(radii[i], 1e-3f)),
                                     transforms[i],
                                     linear_velocity,
                                     angular_velocity,
                                     field,
                                     buoyancy_direction,
                                     gravity_magnitude,
                                     fluid_density,
                                     buoyancy_scale,
                                     drag,
                                     angular_drag,
                                     max_relative_speed);
      }
    }
  }
  else if (pieces.has_pointcloud()) {
    PointCloud &pointcloud = *pieces.get_pointcloud_for_write();
    const int body_count = int(pointcloud.positions().size());
    if (body_count > 0) {
      couplings.reinitialize(body_count);
      const bke::AttributeAccessor attributes = pointcloud.attributes();
      const VArray<float> radii = lookup_or_default<float>(
          attributes, "radius", AttrDomain::Point, 0.1f, body_count);
      const VArray<float3> velocities = lookup_or_default<float3>(
          attributes, "velocity", AttrDomain::Point, float3(0.0f), body_count);
      const VArray<float3> angular_velocities = lookup_or_default<float3>(
          attributes, "angular_velocity", AttrDomain::Point, float3(0.0f), body_count);
      const Span<float3> positions = pointcloud.positions();
      for (const int i : IndexRange(body_count)) {
        const float4x4 transform = math::from_location<float4x4>(positions[i]);
        const BodyTemplate sphere = make_sphere_template(std::max(radii[i], 1e-3f));
        couplings[i] = evaluate_body(sphere,
                                     transform,
                                     velocities[i],
                                     angular_velocities[i],
                                     field,
                                     buoyancy_direction,
                                     gravity_magnitude,
                                     fluid_density,
                                     buoyancy_scale,
                                     drag,
                                     angular_drag,
                                     max_relative_speed);
      }
    }
  }

  if (max_force > 0.0f) {
    for (BodyCoupling &coupling : couplings) {
      const float force = math::length(coupling.force);
      if (force > max_force) {
        coupling.force *= max_force / force;
      }
    }
  }

  if (pieces.has_instances()) {
    bke::Instances *instances = pieces.get_instances_for_write();
    write_instance_force(*instances, couplings, 0);
  }
  else if (pieces.has_pointcloud()) {
    write_point_force(*pieces.get_pointcloud_for_write(), couplings, 0);
  }

  params.set_output("Pieces"_ustr, std::move(pieces));
}

}  // namespace

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeTexturaBuoyancy"_ustr);
  ntype.ui_name = "Textura Buoyancy";
  ntype.ui_description =
      "Buoyancy and drag of a Textura fluid on rigid bodies: sample the fluid volume fraction "
      "around each body and apply the displaced weight and the relative flow";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_textura_buoyancy_cc
