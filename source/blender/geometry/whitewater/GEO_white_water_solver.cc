/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <algorithm>
#include <cmath>
#include <limits>

#include "BLI_array.hh"
#include "BLI_kdopbvh.hh"
#include "BLI_kdtree.hh"
#include "BLI_math_base.hh"
#include "BLI_math_geom.h"
#include "BLI_math_vector.hh"
#include "BLI_task.hh"
#include "BLI_time.h"
#include "BLI_vector.hh"

#include "BKE_bvhutils.hh"
#include "BKE_mesh.hh"

#include "GEO_white_water_solver.hh"

namespace blender::geometry::whitewater {

/* -------------------------------------------------------------------- */
/** \name Random numbers
 * \{ */

/** PCG-ish integer hash, the same shape the other solvers use. Deterministic, no state. */
static uint32_t hash_u32(uint32_t v)
{
  v *= 747796405u;
  v += 2891336453u;
  v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
  return (v >> 22u) ^ v;
}

static uint32_t hash_combine(uint32_t seed, uint32_t value)
{
  return hash_u32(seed ^ (value + 0x9e3779b9u + (seed << 6u) + (seed >> 2u)));
}

static float hash_to_unit(uint32_t value)
{
  return float(value >> 8u) * (1.0f / 16777216.0f);
}

static float random_unit(uint32_t seed, uint32_t a, uint32_t b, uint32_t c)
{
  return hash_to_unit(hash_combine(hash_combine(hash_combine(seed, a), b), c));
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Spatial hash
 * \{ */

/**
 * Counting sort spatial hash, the same construction as the XPBD fluid grid: buckets are filled in
 * particle order, so the neighbour order is fixed and the result does not depend on the thread
 * count.
 */
class ParticleGrid {
  float cell_size_ = 0.0f;
  int table_size_ = 0;
  Array<int3> cells_;

  /** Flat bucket storage: `indices_[bucket_begin_[b] .. bucket_begin_[b + 1])`. */
  Array<int> bucket_begin_;
  Array<int> indices_;

 public:
  bool is_empty() const
  {
    return table_size_ == 0;
  }

  float cell_size() const
  {
    return cell_size_;
  }

  void build(const Span<const float3> positions, const float cell_size)
  {
    const int count = int(positions.size());
    cell_size_ = cell_size;
    table_size_ = 0;
    cells_.reinitialize(0);
    bucket_begin_.reinitialize(0);
    indices_.reinitialize(0);
    if (count == 0 || cell_size <= 0.0f) {
      return;
    }
    int size = 1;
    while (size < count * 2) {
      size <<= 1;
    }
    table_size_ = size;

    cells_.reinitialize(count);
    Array<int> counts(size, 0);
    const float inv_cell = 1.0f / cell_size;
    for (const int i : IndexRange(count)) {
      const float3 &p = positions[i];
      const int3 cell(int(std::floor(p.x * inv_cell)),
                      int(std::floor(p.y * inv_cell)),
                      int(std::floor(p.z * inv_cell)));
      cells_[i] = cell;
      counts[hash_cell(cell) & (size - 1)]++;
    }

    bucket_begin_.reinitialize(size + 1);
    int total = 0;
    for (const int i : IndexRange(size)) {
      bucket_begin_[i] = total;
      total += counts[i];
    }
    bucket_begin_[size] = total;

    Array<int> cursor = bucket_begin_;
    indices_.reinitialize(count);
    for (const int i : IndexRange(count)) {
      indices_[cursor[hash_cell(cells_[i]) & (size - 1)]++] = i;
    }
  }

  template<typename Fn> void foreach_neighbor(const int i,
                                              const Span<const float3> positions,
                                              Fn &&fn) const
  {
    if (table_size_ == 0) {
      return;
    }
    const int mask = table_size_ - 1;
    const float3 &query = positions[i];
    const float support_sq = cell_size_ * cell_size_;
    const int3 cell = cells_[i];
    for (int dz = -1; dz <= 1; dz++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          const int bucket = hash_cell(int3(cell.x + dx, cell.y + dy, cell.z + dz)) & mask;
          for (int k = bucket_begin_[bucket]; k < bucket_begin_[bucket + 1]; k++) {
            const int j = indices_[k];
            if (j == i) {
              continue;
            }
            const float3 delta = positions[j] - query;
            const float distance_sq = math::length_squared(delta);
            if (distance_sq <= support_sq) {
              fn(j, delta, std::sqrt(distance_sq));
            }
          }
        }
      }
    }
  }

 private:
  static uint32_t hash_cell(const int3 &cell)
  {
    return uint32_t(cell.x * 73856093) ^ uint32_t(cell.y * 19349663) ^ uint32_t(cell.z * 83492791);
  }
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Coarse flow grid
 * \{ */

/**
 * Coarse velocity field built from the fluid particles once per frame.
 *
 * White water wants more than the speed of the nearest particle: the *curl* of the velocity is where
 * the foam belongs (shear layers, breaking crests) and its negative *divergence* is the trapped air
 * of a plunging jet. Both come from the same field, so one pass over the particles gives all three
 * criteria (speed, vorticity, trapped air) and the emission reads them with a trilinear sample
 * instead of a nearest particle lookup.
 *
 * Determinism: the particles are counting sorted into cells (ascending index order within a cell),
 * every cell sums its own bucket, and the derivative passes only read the resulting field, so the
 * result does not depend on the thread count and no atomics are involved.
 */
class FlowGrid {
 public:
  struct Sample {
    float3 velocity = float3(0.0f);
    float speed = 0.0f;
    /** Magnitude of the curl of the velocity (1/s). */
    float vorticity = 0.0f;
    /** Positive where the flow converges: air is being entrained (1/s). */
    float trapped_air = 0.0f;
    bool found = false;
  };

  void build(const Span<const float3> positions,
             const Span<const float3> velocities,
             const float mean_spacing)
  {
    cell_count_ = 0;
    active_cells_ = 0;
    positions_ = positions;
    velocities_ = velocities;
    const int count = int(positions.size());
    if (count == 0) {
      return;
    }
    float3 min = positions[0];
    float3 max = positions[0];
    for (const float3 &p : positions) {
      min = math::min(min, p);
      max = math::max(max, p);
    }
    const float3 extent = math::max(max - min, float3(1e-5f));
    /* One cell is about two particle spacings, so the central differences see real neighbours: a
     * grid derived from the bounding volume collapses on a thin sheet (its volume is almost zero)
     * and the derivative stencil then only reads the cell itself. */
    float cell = 2.0f * mean_spacing;
    if (!(cell > 1e-4f)) {
      cell = std::cbrt(extent.x * extent.y * extent.z / float(count));
    }
    cell = math::clamp(cell, 1e-3f, 0.5f);
    constexpr int MAX_CELLS = 262144;
    /* A margin keeps the field sampleable just outside the particle cloud: the surface mesh sits
     * about half a particle radius outside the outermost particles, which is outside a tight
     * bounding box on a thin sheet. */
    constexpr int MARGIN_CELLS = 3;
    for (int attempt = 0; attempt < 8; attempt++) {
      dims_ = int3(std::max(1, int(std::ceil(extent.x / cell))) + 2 * MARGIN_CELLS,
                   std::max(1, int(std::ceil(extent.y / cell))) + 2 * MARGIN_CELLS,
                   std::max(1, int(std::ceil(extent.z / cell))) + 2 * MARGIN_CELLS);
      const int64_t cells = int64_t(dims_.x) * dims_.y * dims_.z;
      if (cells <= MAX_CELLS) {
        break;
      }
      cell *= std::cbrt(double(cells) / double(MAX_CELLS));
    }
    origin_ = min - float3(float(MARGIN_CELLS) * cell);
    cell_size_ = cell;
    const int cells = dims_.x * dims_.y * dims_.z;
    cell_count_ = cells;

    Array<int> cell_of(count, 0);
    Array<int> counts(cells, 0);
    for (const int i : IndexRange(count)) {
      const int3 c = cell_of_position(positions[i]);
      const int index = ((c.z * dims_.y) + c.y) * dims_.x + c.x;
      cell_of[i] = index;
      counts[index]++;
    }
    offsets_.reinitialize(cells + 1);
    int total = 0;
    for (const int i : IndexRange(cells)) {
      offsets_[i] = total;
      total += counts[i];
    }
    offsets_[cells] = total;
    Array<int> cursor = offsets_;
    indices_.reinitialize(count);
    for (const int i : IndexRange(count)) {
      indices_[cursor[cell_of[i]]++] = i;
    }

    counts_.reinitialize(cells);
    counts_.fill(0);
    velocity_.reinitialize(cells);
    velocity_.fill(float3(0.0f));
    vorticity_.reinitialize(cells);
    vorticity_.fill(0.0f);
    trapped_air_.reinitialize(cells);
    trapped_air_.fill(0.0f);
    threading::parallel_for(IndexRange(cells), 4096, [&](const IndexRange range) {
      for (const int cell_index : range) {
        float3 sum(0.0f);
        int n = 0;
        for (int k = offsets_[cell_index]; k < offsets_[cell_index + 1]; k++) {
          sum += velocities_[indices_[k]];
          n++;
        }
        counts_[cell_index] = n;
        if (n > 0) {
          velocity_[cell_index] = sum / float(n);
        }
      }
    });
    for (const int i : IndexRange(cells)) {
      active_cells_ += counts_[i] > 0 ? 1 : 0;
    }

    /* Central differences of the cell centred velocity. A missing neighbour falls back to the cell
     * itself, which keeps a sparse cloud from inventing a derivative. */
    threading::parallel_for(IndexRange(cells), 4096, [&](const IndexRange range) {
      for (const int cell_index : range) {
        if (counts_[cell_index] == 0) {
          continue;
        }
        const int x = cell_index % dims_.x;
        const int y = (cell_index / dims_.x) % dims_.y;
        const int z = cell_index / (dims_.x * dims_.y);
        const float3 center = velocity_[cell_index];
        const float3 xp = velocity_or(x + 1, y, z, center);
        const float3 xm = velocity_or(x - 1, y, z, center);
        const float3 yp = velocity_or(x, y + 1, z, center);
        const float3 ym = velocity_or(x, y - 1, z, center);
        const float3 zp = velocity_or(x, y, z + 1, center);
        const float3 zm = velocity_or(x, y, z - 1, center);
        const float inv = 0.5f / cell_size_;
        const float3 curl((zp.y - zm.y) * inv - (yp.z - ym.z) * inv,
                          (xp.z - xm.z) * inv - (zp.x - zm.x) * inv,
                          (yp.x - ym.x) * inv - (xp.y - xm.y) * inv);
        const float divergence = ((xp.x - xm.x) + (yp.y - ym.y) + (zp.z - zm.z)) * inv;
        vorticity_[cell_index] = math::length(curl);
        trapped_air_[cell_index] = std::max(0.0f, -divergence);
      }
    });
  }

  bool is_valid() const
  {
    return cell_count_ > 0;
  }

  int cell_count() const
  {
    return cell_count_;
  }

  int active_cell_count() const
  {
    return active_cells_;
  }

  Sample sample(const float3 &position) const
  {
    Sample result;
    if (cell_count_ == 0) {
      return result;
    }
    const float3 local = (position - origin_) / cell_size_;
    /* Clamp into the grid (with the margin the surface is normally inside anyway): a point just
     * outside samples the nearest cell instead of falling back to the nearest particle. */
    const int base_x = math::clamp(int(std::floor(local.x)), 0, dims_.x - 1);
    const int base_y = math::clamp(int(std::floor(local.y)), 0, dims_.y - 1);
    const int base_z = math::clamp(int(std::floor(local.z)), 0, dims_.z - 1);
    const float fx = base_x == dims_.x - 1 ? 0.0f : math::clamp(local.x - float(base_x), 0.0f, 1.0f);
    const float fy = base_y == dims_.y - 1 ? 0.0f : math::clamp(local.y - float(base_y), 0.0f, 1.0f);
    const float fz = base_z == dims_.z - 1 ? 0.0f : math::clamp(local.z - float(base_z), 0.0f, 1.0f);
    float weight_sum = 0.0f;
    float3 velocity(0.0f);
    float vorticity = 0.0f;
    float trapped_air = 0.0f;
    for (int dz = 0; dz <= 1; dz++) {
      const float wz = dz == 0 ? 1.0f - fz : fz;
      for (int dy = 0; dy <= 1; dy++) {
        const float wy = dy == 0 ? 1.0f - fy : fy;
        for (int dx = 0; dx <= 1; dx++) {
          const float wx = dx == 0 ? 1.0f - fx : fx;
          const int x = base_x + dx;
          const int y = base_y + dy;
          const int z = base_z + dz;
          if (x < 0 || y < 0 || z < 0 || x >= dims_.x || y >= dims_.y || z >= dims_.z) {
            continue;
          }
          const int index = ((z * dims_.y) + y) * dims_.x + x;
          if (counts_[index] == 0) {
            continue;
          }
          const float w = wx * wy * wz;
          weight_sum += w;
          velocity += velocity_[index] * w;
          vorticity += vorticity_[index] * w;
          trapped_air += trapped_air_[index] * w;
        }
      }
    }
    if (weight_sum < 1e-4f) {
      return result;
    }
    result.velocity = velocity / weight_sum;
    result.speed = math::length(result.velocity);
    result.vorticity = vorticity / weight_sum;
    result.trapped_air = trapped_air / weight_sum;
    result.found = true;
    return result;
  }

 private:
  int3 cell_of_position(const float3 &position) const
  {
    const float3 local = (position - origin_) / cell_size_;
    return int3(math::clamp(int(std::floor(local.x)), 0, dims_.x - 1),
                math::clamp(int(std::floor(local.y)), 0, dims_.y - 1),
                math::clamp(int(std::floor(local.z)), 0, dims_.z - 1));
  }

  float3 velocity_or(const int x, const int y, const int z, const float3 &fallback) const
  {
    if (x < 0 || y < 0 || z < 0 || x >= dims_.x || y >= dims_.y || z >= dims_.z) {
      return fallback;
    }
    const int index = ((z * dims_.y) + y) * dims_.x + x;
    return counts_[index] > 0 ? velocity_[index] : fallback;
  }

  float3 origin_ = float3(0.0f);
  float cell_size_ = 0.0f;
  int3 dims_ = int3(0);
  int cell_count_ = 0;
  int active_cells_ = 0;
  Span<const float3> positions_;
  Span<const float3> velocities_;
  Array<int> offsets_;
  Array<int> indices_;
  Array<int> counts_;
  Array<float3> velocity_;
  Array<float> vorticity_;
  Array<float> trapped_air_;
};

/** \} */

/* -------------------------------------------------------------------- */
/** \name Triangle queries
 * \{ */

static void triangle_nearest_cb(void *userdata,
                                int index,
                                const float co[3],
                                BVHTreeNearest *nearest)
{
  const Span<const float3> &triangles = *static_cast<const Span<const float3> *>(userdata);
  const float3 &a = triangles[index * 3 + 0];
  const float3 &b = triangles[index * 3 + 1];
  const float3 &c = triangles[index * 3 + 2];
  float closest[3];
  closest_on_tri_to_point_v3(closest, co, a, b, c);
  const float dx = closest[0] - co[0];
  const float dy = closest[1] - co[1];
  const float dz = closest[2] - co[2];
  const float distance_sq = dx * dx + dy * dy + dz * dz;
  if (distance_sq < nearest->dist_sq) {
    nearest->index = index;
    nearest->dist_sq = distance_sq;
    copy_v3_v3(nearest->co, closest);
  }
}

/** Distance field of a flat triangle list (the colliders), used to kill spray on impact. */
class TriangleBVH {
  BVHTree *tree_ = nullptr;
  Span<const float3> triangles_;
  bool valid_ = false;

 public:
  void build(const Span<const float3> triangles)
  {
    triangles_ = triangles;
    valid_ = false;
    if (tree_ != nullptr) {
      BLI_bvhtree_free(tree_);
      tree_ = nullptr;
    }
    if (triangles.size() < 3) {
      return;
    }
    tree_ = BLI_bvhtree_new(int(triangles.size() / 3), 0.0f, 4, 6);
    if (tree_ == nullptr) {
      return;
    }
    for (int i = 0; i < int(triangles.size() / 3); i++) {
      BLI_bvhtree_insert(tree_, i, &triangles[i * 3].x, 3);
    }
    BLI_bvhtree_balance(tree_);
    valid_ = true;
  }

  bool is_valid() const
  {
    return valid_;
  }

  bool nearest(const float3 &position, float &r_distance) const
  {
    if (!valid_) {
      return false;
    }
    BVHTreeNearest nearest;
    nearest.index = -1;
    nearest.dist_sq = std::numeric_limits<float>::max();
    const float co[3] = {position.x, position.y, position.z};
    Span<const float3> triangles = triangles_;
    if (BLI_bvhtree_find_nearest(tree_, co, &nearest, triangle_nearest_cb, &triangles) < 0) {
      return false;
    }
    r_distance = std::sqrt(nearest.dist_sq);
    return true;
  }

  ~TriangleBVH()
  {
    if (tree_ != nullptr) {
      BLI_bvhtree_free(tree_);
    }
  }
};

static bool surface_nearest(const bke::BVHTreeFromMesh &bvh,
                            const float3 &position,
                            float3 &r_closest,
                            float &r_distance)
{
  if (bvh.tree == nullptr) {
    return false;
  }
  BVHTreeNearest nearest;
  nearest.index = -1;
  nearest.dist_sq = std::numeric_limits<float>::max();
  const float co[3] = {position.x, position.y, position.z};
  bke::BVHTreeFromMesh &mutable_bvh = const_cast<bke::BVHTreeFromMesh &>(bvh);
  if (BLI_bvhtree_find_nearest(bvh.tree, co, &nearest, bvh.nearest_callback, &mutable_bvh) < 0) {
    return false;
  }
  r_closest = float3(nearest.co);
  r_distance = std::sqrt(nearest.dist_sq);
  return true;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Particle storage
 * \{ */

void ParticleData::clear()
{
  positions.clear();
  velocities.clear();
  ages.clear();
  lifetimes.clear();
  radii.clear();
  types.clear();
  clusters.clear();
}

void ParticleData::reserve(const int count)
{
  positions.reserve(count);
  velocities.reserve(count);
  ages.reserve(count);
  lifetimes.reserve(count);
  radii.reserve(count);
  types.reserve(count);
  clusters.reserve(count);
}

void ParticleData::append(const float3 &position,
                          const float3 &velocity,
                          const float age,
                          const float lifetime,
                          const float radius,
                          const int type)
{
  positions.append(position);
  velocities.append(velocity);
  ages.append(age);
  lifetimes.append(lifetime);
  radii.append(radius);
  types.append(type);
  clusters.append(-1);
}

void ParticleData::remove_front(const int count)
{
  const int remove = std::min(count, this->size());
  if (remove <= 0) {
    return;
  }
  const int remaining = this->size() - remove;
  for (const int i : IndexRange(remaining)) {
    positions[i] = positions[i + remove];
    velocities[i] = velocities[i + remove];
    ages[i] = ages[i + remove];
    lifetimes[i] = lifetimes[i + remove];
    radii[i] = radii[i + remove];
    types[i] = types[i + remove];
    clusters[i] = clusters[i + remove];
  }
  for (const int i : IndexRange(remove)) {
    positions.pop_last();
    velocities.pop_last();
    ages.pop_last();
    lifetimes.pop_last();
    radii.pop_last();
    types.pop_last();
    clusters.pop_last();
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Emission
 * \{ */

struct FlowSample {
  float3 velocity = float3(0.0f);
  float speed = 0.0f;
  float vorticity = 0.0f;
  float trapped_air = 0.0f;
  bool found = false;
};

static FlowSample sample_flow(const KDTree<float3> *tree,
                              const Span<const float3> fluid_velocities,
                              const float3 &position)
{
  FlowSample sample;
  if (tree == nullptr) {
    return sample;
  }
  KDTreeNearest<float3> nearest;
  if (kdtree_find_nearest(tree, position, &nearest) < 0) {
    return sample;
  }
  if (nearest.index < 0 || nearest.index >= int(fluid_velocities.size())) {
    return sample;
  }
  sample.velocity = fluid_velocities[nearest.index];
  sample.speed = math::length(sample.velocity);
  sample.found = true;
  return sample;
}

static float emission_density(const float speed,
                              const float threshold,
                              const float full_speed,
                              const float rate,
                              const float delta_time,
                              const float scale)
{
  if (rate <= 0.0f || full_speed <= threshold) {
    return 0.0f;
  }
  const float factor = math::clamp((speed - threshold) / (full_speed - threshold), 0.0f, 1.0f);
  return rate * factor * delta_time * scale;
}

/**
 * Sample the fluid surface and append the new particles.
 *
 * Triangles are visited in mesh order and each sample uses a hash of (seed, frame, triangle,
 * sample), so the emission is deterministic and still differs between frames.
 */
static void emit_particles(ParticleData &particles,
                           const Mesh &surface,
                           const KDTree<float3> *flow_tree,
                           const FlowGrid &flow_grid,
                           const Span<const float3> fluid_velocities,
                           const Settings &settings,
                           const float delta_time,
                           Stats &r_stats)
{
  const Span<float3> verts = surface.vert_positions();
  const Span<int3> corner_tris = surface.corner_tris();
  const Span<int> corner_verts = surface.corner_verts();
  if (corner_tris.is_empty()) {
    return;
  }

  const uint32_t seed = uint32_t(settings.seed);
  const uint32_t frame = uint32_t(std::max(settings.frame, 0));

  for (const int tri_index : IndexRange(corner_tris.size())) {
    const int3 tri = corner_tris[tri_index];
    const float3 &a = verts[corner_verts[tri[0]]];
    const float3 &b = verts[corner_verts[tri[1]]];
    const float3 &c = verts[corner_verts[tri[2]]];
    const float3 normal_vec = math::cross(b - a, c - a);
    const float double_area = math::length(normal_vec);
    if (!(double_area > 1e-12f)) {
      continue;
    }
    const float area = 0.5f * double_area;
    const float3 normal = normal_vec / double_area;
    const float3 centroid = (a + b + c) / 3.0f;
    /* The coarse grid gives the speed, the curl and the divergence at once; the nearest particle is
     * the fallback when the grid has no mass at this point (a very sparse splash). */
    FlowSample flow;
    const FlowGrid::Sample grid_sample = flow_grid.sample(centroid);
    if (grid_sample.found) {
      flow.velocity = grid_sample.velocity;
      flow.speed = grid_sample.speed;
      flow.vorticity = grid_sample.vorticity;
      flow.trapped_air = grid_sample.trapped_air;
      flow.found = true;
    }
    else {
      flow = sample_flow(flow_tree, fluid_velocities, centroid);
    }
    if (!flow.found) {
      continue;
    }
    const float vorticity_factor = math::clamp(
        flow.vorticity / std::max(settings.vorticity_scale, 1e-3f), 0.0f, 1.0f);
    const float trapped_air_factor = math::clamp(
        flow.trapped_air / std::max(settings.divergence_scale, 1e-3f), 0.0f, 1.0f);

    for (int type = 0; type < 3; type++) {
      if (type == int(ParticleType::Foam) && !settings.foam) {
        continue;
      }
      if (type == int(ParticleType::Spray) && !settings.spray) {
        continue;
      }
      if (type == int(ParticleType::Bubble) && !settings.bubbles) {
        continue;
      }
      const float rate = type == int(ParticleType::Foam) ? settings.foam_emission :
                        type == int(ParticleType::Spray) ? settings.spray_emission :
                                                           settings.bubble_emission;
      const float threshold = type == int(ParticleType::Foam) ? settings.foam_threshold :
                              type == int(ParticleType::Spray) ? settings.spray_threshold :
                                                                 settings.bubble_threshold;
      const float full_speed = type == int(ParticleType::Foam) ?
                                   settings.foam_full_speed :
                               type == int(ParticleType::Spray) ? settings.spray_full_speed :
                                                                  settings.bubble_full_speed;
      /* Vorticity and trapped air multiply the speed driven emission; both weights default to 0,
       * which reproduces the plain speed window. Bubbles are the air itself, so they get the trapped
       * air term twice when `Air to Bubbles` is on. */
      float boost = 1.0f + settings.vorticity_weight * vorticity_factor +
                    settings.trapped_air_weight * trapped_air_factor;
      if (type == int(ParticleType::Bubble) && settings.air_to_bubbles) {
        boost += settings.trapped_air_weight * trapped_air_factor;
      }
      const float density = emission_density(
                                flow.speed, threshold, full_speed, rate, delta_time,
                                settings.emission_scale) *
                            boost;
      if (density <= 0.0f) {
        continue;
      }
      const float expected = area * density;
      int count = int(std::floor(expected));
      if (random_unit(seed, frame, uint32_t(tri_index), uint32_t(type) * 7919u) <
          expected - float(count)) {
        count += 1;
      }
      for (const int sample : IndexRange(count)) {
        const uint32_t base = uint32_t(tri_index) * 31u + uint32_t(type) * 9781u +
                              uint32_t(sample) * 13u;
        const float r0 = random_unit(seed, frame, base, 1u);
        const float r1 = random_unit(seed, frame, base, 2u);
        const float r2 = random_unit(seed, frame, base, 3u);
        const float r3 = random_unit(seed, frame, base, 4u);
        const float sqrt_r0 = std::sqrt(r0);
        const float w0 = 1.0f - sqrt_r0;
        const float w1 = sqrt_r0 * (1.0f - r1);
        const float w2 = sqrt_r0 * r1;
        const float3 position = a * w0 + b * w1 + c * w2;
        const float size = type == int(ParticleType::Foam) ? settings.foam_size :
                           type == int(ParticleType::Spray) ? settings.spray_size :
                                                              settings.bubble_size;
        const float lifetime = type == int(ParticleType::Foam) ? settings.foam_lifetime :
                               type == int(ParticleType::Spray) ? settings.spray_lifetime :
                                                                  settings.bubble_lifetime;
        float3 velocity = flow.velocity;
        float3 spawn = position;
        if (type == int(ParticleType::Foam)) {
          spawn += normal * (0.25f * size);
        }
        else if (type == int(ParticleType::Spray)) {
          spawn += normal * (0.5f * size);
          velocity += normal * (settings.spray_speed * (0.5f + r2));
        }
        else {
          /* Bubbles start below the surface; the depth is a fraction of `bubble_depth`. */
          spawn -= normal * (settings.bubble_depth * r2);
        }
        particles.append(spawn,
                         velocity,
                         0.0f,
                         lifetime * (0.75f + 0.5f * r3),
                         size * (0.6f + 0.8f * random_unit(seed, frame, base, 5u)),
                         type);
        r_stats.emitted++;
      }
    }
  }
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Clusters
 * \{ */

/**
 * Union-find over the neighbour links that are still "intact": a link survives while the relative
 * normal speed divided by the distance is below `break_strain`. Clumps therefore emerge from the
 * particles themselves: a fast shear tears the links and the clump separates, and the links form
 * again once the flow calms down.
 */
static void build_clusters(ParticleData &particles,
                           const ParticleGrid &grid,
                           const Span<const float3> positions,
                           const Settings &settings,
                           Stats &r_stats)
{
  const int count = particles.size();
  Array<int> parent(count);
  for (const int i : IndexRange(count)) {
    parent[i] = i;
  }
  auto find_root = [&](int index) {
    int root = index;
    while (parent[root] != root) {
      parent[root] = parent[parent[root]];
      root = parent[root];
    }
    return root;
  };

  int links = 0;
  const float radius = std::max(settings.cohesion_radius, 1e-5f);
  for (const int i : IndexRange(count)) {
    const int type_i = particles.types[i];
    grid.foreach_neighbor(
        i, positions, [&](const int j, const float3 &delta, const float distance) {
          if (j <= i) {
            return;
          }
          if (settings.same_type_only && particles.types[j] != type_i) {
            return;
          }
          if (distance > radius) {
            return;
          }
          /* Strain rate of the link: the relative speed over the cohesion radius. Using the full
           * relative speed (not only its normal part) is what makes a shear tear the clump apart;
           * a clump that moves as a whole has almost no relative speed and keeps its links. The
           * radius (instead of the tiny neighbour distance) keeps the value in a useful range. */
          const float3 relative = particles.velocities[j] - particles.velocities[i];
          const float strain = math::length(relative) / radius;
          if (strain > settings.break_strain) {
            return;
          }
          (void)delta;
          links++;
          const int root_i = find_root(i);
          const int root_j = find_root(j);
          if (root_i != root_j) {
            parent[root_j] = root_i;
          }
        });
  }
  r_stats.links = links;

  /* Dense ids, then split the oversized clumps deterministically in index order. */
  Array<int> root_id(count, -1);
  Vector<int> sizes;
  for (const int i : IndexRange(count)) {
    const int root = find_root(i);
    if (root_id[root] < 0) {
      root_id[root] = sizes.size();
      sizes.append(0);
    }
    sizes[root_id[root]]++;
  }
  const int max_size = std::max(settings.max_cluster_size, 1);
  Array<int> base_id(sizes.size(), 0);
  int next_id = 0;
  for (const int root : IndexRange(sizes.size())) {
    base_id[root] = next_id;
    next_id += std::max(1, (sizes[root] + max_size - 1) / max_size);
  }
  Array<int> used(sizes.size(), 0);
  for (const int i : IndexRange(count)) {
    const int id = root_id[find_root(i)];
    const int sub = used[id] / max_size;
    used[id]++;
    particles.clusters[i] = base_id[id] + sub;
  }
  r_stats.clusters = next_id;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Frame
 * \{ */

/**
 * Median distance from a fluid particle to its nearest neighbour: the spacing of the particle cloud,
 * which decides the cell size of the flow grid.
 */
static float estimate_fluid_spacing(const KDTree<float3> *tree, const Span<const float3> positions)
{
  const int count = int(positions.size());
  if (tree == nullptr || count < 8) {
    return 0.0f;
  }
  const int samples = std::min(count, 1024);
  const int step = std::max(1, count / samples);
  Vector<float> distances;
  distances.reserve(samples);
  for (int i = 0; i < count; i += step) {
    KDTreeNearest<float3> nearest[2];
    const int found = kdtree_find_nearest_n(tree, positions[i], nearest, 2);
    if (found >= 2 && nearest[1].dist > 0.0f) {
      distances.append(nearest[1].dist);
    }
  }
  if (distances.is_empty()) {
    return 0.0f;
  }
  std::nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
  return distances[distances.size() / 2];
}

void solve_frame(ParticleData &particles,
                 const Mesh *surface,
                 const Span<const float3> fluid_positions,
                 const Span<const float3> fluid_velocities,
                 const Span<const float3> collider_triangles,
                 const Settings &settings,
                 const float delta_time,
                 Stats &r_stats)
{
  const double start_time = BLI_time_now_seconds();
  const float dt = math::clamp(delta_time, 0.0f, 0.25f);
  r_stats = Stats{};

  KDTree<float3> *flow_tree = nullptr;
  if (fluid_positions.size() > 0 && fluid_positions.size() == fluid_velocities.size()) {
    flow_tree = kdtree_new<float3>(uint(fluid_positions.size()));
    for (const int i : IndexRange(fluid_positions.size())) {
      kdtree_insert(flow_tree, i, fluid_positions[i]);
    }
    kdtree_balance(flow_tree);
  }
  FlowGrid flow_grid;
  if (fluid_positions.size() > 0 && fluid_positions.size() == fluid_velocities.size()) {
    const float spacing = estimate_fluid_spacing(flow_tree, fluid_positions);
    flow_grid.build(fluid_positions, fluid_velocities, spacing);
    r_stats.grid_cells = flow_grid.active_cell_count();
  }

  const bke::BVHTreeFromMesh surface_bvh = surface != nullptr ? surface->bvh_corner_tris() :
                                                               bke::BVHTreeFromMesh();
  TriangleBVH collider_bvh;
  if (!collider_triangles.is_empty()) {
    collider_bvh.build(collider_triangles);
  }

  const int previous_count = particles.size();
  if (previous_count > 0 && dt > 0.0f) {
    /* Advect, steer foam towards the flow and the surface, age everything. */
    const float stick = math::clamp(settings.foam_stick, 0.0f, 1.0f);
    const float stick_blend = stick > 0.0f ? 1.0f - std::exp(-10.0f * stick * dt) : 0.0f;
    threading::parallel_for(IndexRange(previous_count), 1024, [&](const IndexRange range) {
      for (const int i : range) {
        const int type = particles.types[i];
        float3 &velocity = particles.velocities[i];
        float3 &position = particles.positions[i];
        if (type == int(ParticleType::Spray)) {
          velocity += settings.gravity * dt;
          velocity /= (1.0f + settings.drag * dt);
        }
        else if (type == int(ParticleType::Bubble)) {
          velocity.z += settings.bubble_buoyancy * dt;
          velocity /= (1.0f + settings.drag * dt);
        }
        else if (type == int(ParticleType::Foam) && stick_blend > 0.0f) {
          const FlowSample flow = sample_flow(flow_tree, fluid_velocities, position);
          if (flow.found) {
            velocity = math::interpolate(velocity, flow.velocity, stick_blend);
          }
          if (surface_bvh.tree != nullptr) {
            float3 closest;
            float distance = 0.0f;
            if (surface_nearest(surface_bvh, position, closest, distance)) {
              position = math::interpolate(position, closest, stick_blend);
            }
          }
        }
        position += velocity * dt;
        particles.ages[i] += dt;
      }
    });
  }

  /* Neighbour coupled forces: cohesion (clumping), separation (no overlap) and relative velocity
   * damping. Accumulated per particle (Jacobi) so the result is independent of the thread count. */
  const bool has_forces = settings.cohesion > 0.0f || settings.separation > 0.0f ||
                          settings.cluster_damping > 0.0f;
  Array<float3> deltas(particles.size(), float3(0.0f));
  ParticleGrid grid;
  if (has_forces && particles.size() > 1) {
    Vector<float3> positions(particles.size());
    for (const int i : IndexRange(particles.size())) {
      positions[i] = particles.positions[i];
    }
    const float cell = std::max({settings.cohesion_radius, settings.separation_radius, 1e-4f});
    const Span<const float3> position_span(positions.data(), positions.size());
    grid.build(position_span, cell);
    threading::parallel_for(IndexRange(particles.size()), 1024, [&](const IndexRange range) {
      for (const int i : range) {
        const int type_i = particles.types[i];
        float3 delta(0.0f);
        int neighbors = 0;
        grid.foreach_neighbor(
            i, position_span, [&](const int j, const float3 &to_j, const float distance) {
              if (settings.same_type_only && particles.types[j] != type_i) {
                return;
              }
              neighbors++;
              const float safe_distance = std::max(distance, 1e-6f);
              const float3 direction = to_j / safe_distance;
              if (settings.separation > 0.0f && distance < settings.separation_radius) {
                const float factor = 1.0f - distance / std::max(settings.separation_radius, 1e-6f);
                delta -= direction * (settings.separation * factor * dt);
              }
              if (settings.cohesion > 0.0f && distance < settings.cohesion_radius) {
                const float factor = 1.0f -
                                     distance / std::max(settings.cohesion_radius, 1e-6f);
                delta += direction * (settings.cohesion * factor * dt);
              }
              if (settings.cluster_damping > 0.0f && distance < settings.cohesion_radius) {
                const float factor = 1.0f -
                                     distance / std::max(settings.cohesion_radius, 1e-6f);
                delta += (particles.velocities[j] - particles.velocities[i]) *
                         (settings.cluster_damping * factor * dt);
              }
            });
        /* Average over the neighbours, so `Cohesion` is an acceleration and not a density
         * dependent sum, and never let one frame change the velocity by more than a sane amount. */
        if (neighbors > 0) {
          delta /= float(neighbors);
        }
        const float max_delta = 0.25f;
        if (math::length_squared(delta) > max_delta * max_delta) {
          delta = math::normalize(delta) * max_delta;
        }
        deltas[i] = delta;
      }
    });
    threading::parallel_for(IndexRange(particles.size()), 1024, [&](const IndexRange range) {
      for (const int i : range) {
        particles.velocities[i] += deltas[i];
      }
    });
  }

  if (settings.cluster_mode == ClusterMode::Strain && grid.cell_size() > 0.0f) {
    Vector<float3> positions(particles.size());
    for (const int i : IndexRange(particles.size())) {
      positions[i] = particles.positions[i];
    }
    const Span<const float3> position_span(positions.data(), positions.size());
    build_clusters(particles, grid, position_span, settings, r_stats);
  }
  else {
    for (const int i : IndexRange(particles.size())) {
      particles.clusters[i] = i;
    }
    r_stats.clusters = particles.size();
  }

  /* Kill, convert bubbles that reached the surface, and drop everything out of bounds. */
  Array<bool> keep(particles.size(), true);
  const bool has_bounds = settings.use_bounds;
  threading::parallel_for(IndexRange(particles.size()), 1024, [&](const IndexRange range) {
    for (const int i : range) {
      const float3 &position = particles.positions[i];
      if (particles.ages[i] > particles.lifetimes[i]) {
        keep[i] = false;
        continue;
      }
      if (has_bounds &&
          (position.x < settings.bounds_min.x || position.y < settings.bounds_min.y ||
           position.z < settings.bounds_min.z || position.x > settings.bounds_max.x ||
           position.y > settings.bounds_max.y || position.z > settings.bounds_max.z)) {
        keep[i] = false;
        continue;
      }
      if (particles.types[i] == int(ParticleType::Spray) && collider_bvh.is_valid()) {
        float distance = 0.0f;
        if (collider_bvh.nearest(position, distance) && distance < particles.radii[i]) {
          keep[i] = false;
          continue;
        }
      }
      if (particles.types[i] == int(ParticleType::Bubble) && surface_bvh.tree != nullptr &&
          settings.bubble_to_foam) {
        float3 closest;
        float distance = 0.0f;
        if (surface_nearest(surface_bvh, position, closest, distance) &&
            distance < particles.radii[i]) {
          /* The bubble popped: it becomes foam at the water line. */
          particles.types[i] = int(ParticleType::Foam);
          particles.ages[i] = 0.0f;
          particles.lifetimes[i] = settings.foam_lifetime;
          particles.radii[i] = std::max(particles.radii[i], settings.foam_size);
        }
      }
    }
  });
  /* Compact in place, preserving order (deterministic and keeps the oldest particles first). */
  int write = 0;
  for (const int i : IndexRange(particles.size())) {
    if (!keep[i]) {
      continue;
    }
    if (write != i) {
      particles.positions[write] = particles.positions[i];
      particles.velocities[write] = particles.velocities[i];
      particles.ages[write] = particles.ages[i];
      particles.lifetimes[write] = particles.lifetimes[i];
      particles.radii[write] = particles.radii[i];
      particles.types[write] = particles.types[i];
      particles.clusters[write] = particles.clusters[i];
    }
    write++;
  }
  if (write < particles.size()) {
    const int remove = particles.size() - write;
    for (int i = 0; i < remove; i++) {
      particles.positions.pop_last();
      particles.velocities.pop_last();
      particles.ages.pop_last();
      particles.lifetimes.pop_last();
      particles.radii.pop_last();
      particles.types.pop_last();
      particles.clusters.pop_last();
    }
  }

  if (surface != nullptr && flow_tree != nullptr && dt > 0.0f) {
    emit_particles(
        particles, *surface, flow_tree, flow_grid, fluid_velocities, settings, dt, r_stats);
  }

  /* Hard budget: drop the oldest particles first. */
  const int max_particles = std::max(settings.max_particles, 0);
  if (max_particles > 0 && particles.size() > max_particles) {
    particles.remove_front(particles.size() - max_particles);
  }
  else if (max_particles == 0) {
    particles.clear();
  }

  /* Statistics. */
  float max_speed = 0.0f;
  for (const int i : IndexRange(particles.size())) {
    const int type = particles.types[i];
    if (type == int(ParticleType::Foam)) {
      r_stats.foam++;
    }
    else if (type == int(ParticleType::Spray)) {
      r_stats.spray++;
    }
    else {
      r_stats.bubbles++;
    }
    max_speed = std::max(max_speed, math::length(particles.velocities[i]));
  }
  r_stats.particles = particles.size();
  r_stats.max_speed = max_speed;
  r_stats.milliseconds = (BLI_time_now_seconds() - start_time) * 1000.0;

  if (flow_tree != nullptr) {
    kdtree_free(flow_tree);
  }
}

}  // namespace blender::geometry::whitewater
