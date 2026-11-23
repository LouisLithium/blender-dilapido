/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Crowd Navigation`: a 2D walkable grid plus a multi-source flow field.
 *
 * Obstacles are rasterized into a square grid (in XY) with the agent radius as clearance.
 * Triangles that lie completely below `Ground Z + Max Step Height` (ground, low steps) or
 * completely above the agent's head (`Ground Z + Agent Height`) are ignored, so a crowd can walk
 * over low geometry and pass under beams while walls and tall boxes block it.
 *
 * Goal points become the sources of a multi-source Dijkstra distance field; the flow at every
 * reachable cell points downhill (towards the goals). Goals with a `target_group` attribute get
 * their own layer, and an extra `-1` layer contains all goals so agents without a matching group
 * still have a route.
 *
 * The result is a `Blender.CrowdNavigation` bundle:
 * - `geometry`: one point per (layer, cell) with `flow`, `nav_distance`, `reachable`, `blocked`,
 *   `nav_group`, `nav_layer` and `nav_cell`.
 * - `min` (vector), `cell_size` (float), `dims_x` / `dims_y` (int), `group_count` (int) and
 *   `group_<i>` (int).
 *
 * The grid is cached by a content hash of the obstacles, goals, bounds and parameters, so a static
 * scene only pays the rasterization + Dijkstra cost once.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_base.hh"
#include "BLI_math_vector.hh"
#include "BLI_set.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_crowd_navigation_cc {

inline constexpr StringRefNull NAVIGATION_TYPE = "Blender.CrowdNavigation";

inline constexpr float UNREACHABLE_DISTANCE = 1.0e9f;
inline constexpr int MAX_EXPLICIT_GROUPS = 16;
inline constexpr int MAX_CELLS = 2'000'000;

struct Goal {
  float2 position;
  int group = -1;
};

struct NavResult {
  float2 min = float2(0.0f);
  float cell_size = 0.25f;
  int dims_x = 0;
  int dims_y = 0;
  /** Group value of every layer; the last one is always -1 (all goals). */
  Vector<int> groups;
  /** Size = layers * cells, ordered layer by layer. */
  Array<float3> flow;
  Array<float> distance;
  Array<uint8_t> reachable;
  /** Size = cells; shared by all layers. */
  Array<uint8_t> blocked;

  int cell_count() const
  {
    return dims_x * dims_y;
  }
};

/* -------------------------------------------------------------------- hashing */

static void hash_bytes(uint64_t &hash, const void *data, const size_t size)
{
  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  for (const int i : IndexRange(int(size))) {
    hash ^= uint64_t(bytes[i]);
    hash *= 1099511628211ull;
  }
}

template<typename T> static void hash_value(uint64_t &hash, const T &value)
{
  hash_bytes(hash, &value, sizeof(T));
}

struct NavigationCache {
  std::mutex mutex;
  Map<uint64_t, std::shared_ptr<const NavResult>> entries;
  Vector<uint64_t> order;
};

static NavigationCache &navigation_cache()
{
  static NavigationCache cache;
  return cache;
}

static std::shared_ptr<const NavResult> cache_lookup(const uint64_t key)
{
  NavigationCache &cache = navigation_cache();
  std::lock_guard lock(cache.mutex);
  const std::shared_ptr<const NavResult> *found = cache.entries.lookup_ptr(key);
  return found != nullptr ? *found : nullptr;
}

static void cache_insert(const uint64_t key, std::shared_ptr<const NavResult> result)
{
  NavigationCache &cache = navigation_cache();
  std::lock_guard lock(cache.mutex);
  if (cache.entries.contains(key)) {
    return;
  }
  cache.entries.add(key, std::move(result));
  cache.order.append(key);
  constexpr int MAX_ENTRIES = 4;
  while (cache.order.size() > MAX_ENTRIES) {
    const uint64_t oldest = cache.order[0];
    cache.order.remove(0);
    cache.entries.remove(oldest);
  }
}

/* -------------------------------------------------------------------- geometry */

static void read_goals(const GeometrySet &geometry, Vector<Goal> &r_goals)
{
  if (const PointCloud *points = geometry.get_pointcloud()) {
    const VArray<int> groups =
        points->attributes().lookup_or_default<int>("target_group", AttrDomain::Point, -1).varray;
    const Span<float3> positions = points->positions();
    for (const int i : IndexRange(points->totpoint)) {
      r_goals.append({float2(positions[i].x, positions[i].y), groups[i]});
    }
    return;
  }
  if (const Mesh *mesh = geometry.get_mesh()) {
    const VArray<int> groups =
        mesh->attributes().lookup_or_default<int>("target_group", AttrDomain::Point, -1).varray;
    const Span<float3> positions = mesh->vert_positions();
    for (const int i : IndexRange(mesh->verts_num)) {
      r_goals.append({float2(positions[i].x, positions[i].y), groups[i]});
    }
  }
}

/** Extra geometry (for example the spawn surface) that the navigation grid has to cover. */
static void read_bounds_points(const GeometrySet &geometry, Vector<float2> &r_points)
{
  if (const PointCloud *points = geometry.get_pointcloud()) {
    for (const float3 &position : points->positions()) {
      r_points.append(float2(position.x, position.y));
    }
    return;
  }
  if (const Mesh *mesh = geometry.get_mesh()) {
    for (const float3 &position : mesh->vert_positions()) {
      r_points.append(float2(position.x, position.y));
    }
  }
}

static bool mesh_has_data(const Mesh *mesh)
{
  return mesh != nullptr && !mesh->corner_tris().is_empty() && !mesh->vert_positions().is_empty();
}

/* -------------------------------------------------------------------- 2D math */

static float distance_to_segment_2d(const float2 point, const float2 a, const float2 b)
{
  const float2 ab = b - a;
  const float length_sq = math::length_squared(ab);
  if (length_sq <= 1.0e-12f) {
    return math::length(point - a);
  }
  const float t = math::clamp(math::dot(point - a, ab) / length_sq, 0.0f, 1.0f);
  return math::length(point - (a + ab * t));
}

/** Distance from a point to a triangle in 2D; zero when the point is inside. */
static float distance_to_triangle_2d(const float2 point,
                                     const float2 a,
                                     const float2 b,
                                     const float2 c)
{
  /* Vertical triangles (walls) project to a line in XY; their area is zero and the inside test
   * would report every point as inside. Fall back to the edges in that case. */
  const float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  if (std::abs(area) <= 1.0e-9f) {
    return std::min({distance_to_segment_2d(point, a, b),
                     distance_to_segment_2d(point, b, c),
                     distance_to_segment_2d(point, c, a)});
  }
  const float cross_ab = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
  const float cross_bc = (c.x - b.x) * (point.y - b.y) - (c.y - b.y) * (point.x - b.x);
  const float cross_ca = (a.x - c.x) * (point.y - c.y) - (a.y - c.y) * (point.x - c.x);
  const bool has_negative = cross_ab < 0.0f || cross_bc < 0.0f || cross_ca < 0.0f;
  const bool has_positive = cross_ab > 0.0f || cross_bc > 0.0f || cross_ca > 0.0f;
  if (!(has_negative && has_positive)) {
    return 0.0f;
  }
  return std::min({distance_to_segment_2d(point, a, b),
                   distance_to_segment_2d(point, b, c),
                   distance_to_segment_2d(point, c, a)});
}

/* -------------------------------------------------------------------- build */

struct BuildParams {
  float agent_radius = 0.3f;
  float agent_height = 1.8f;
  float max_step_height = 0.35f;
  float ground_z = 0.0f;
  float cell_size = 0.0f;
  float margin = 2.0f;
  bool use_bounds = false;
  float3 bounds_min = float3(-10.0f);
  float3 bounds_max = float3(10.0f);
};

static int2 clamp_cell(const int2 cell, const int dims_x, const int dims_y)
{
  return int2(math::clamp(cell.x, 0, dims_x - 1), math::clamp(cell.y, 0, dims_y - 1));
}

static int flatten(const int2 cell, const int dims_x)
{
  return cell.y * dims_x + cell.x;
}

static float2 cell_center(const NavResult &nav, const int2 cell)
{
  return nav.min +
         float2((float(cell.x) + 0.5f) * nav.cell_size, (float(cell.y) + 0.5f) * nav.cell_size);
}

static void rasterize_obstacles(const Mesh *mesh, NavResult &nav, const BuildParams &params)
{
  const Span<float3> positions = mesh->vert_positions();
  const Span<int3> tris = mesh->corner_tris();
  const int dims_x = nav.dims_x;
  const int dims_y = nav.dims_y;
  const float cell = nav.cell_size;
  const float clearance = params.agent_radius + cell * 0.5f;
  const float low = params.ground_z + params.max_step_height;
  const float high = params.ground_z + params.agent_height;

  for (const int3 &tri : tris) {
    const float3 &p0 = positions[tri.x];
    const float3 &p1 = positions[tri.y];
    const float3 &p2 = positions[tri.z];
    const float z_min = std::min({p0.z, p1.z, p2.z});
    const float z_max = std::max({p0.z, p1.z, p2.z});
    /* Entirely below the step height: the crowd can walk over it. */
    if (z_max <= low) {
      continue;
    }
    /* Entirely above the head: it is a ceiling, not an obstacle in the 2D floor plan. */
    if (z_min >= high) {
      continue;
    }
    const float2 a(p0.x, p0.y);
    const float2 b(p1.x, p1.y);
    const float2 c(p2.x, p2.y);
    const float2 tri_min = float2(std::min({a.x, b.x, c.x}), std::min({a.y, b.y, c.y}));
    const float2 tri_max = float2(std::max({a.x, b.x, c.x}), std::max({a.y, b.y, c.y}));
    const int2 start = clamp_cell(
        int2(int(std::floor((tri_min.x - clearance - nav.min.x) / cell)),
             int(std::floor((tri_min.y - clearance - nav.min.y) / cell))),
        dims_x,
        dims_y);
    const int2 end = clamp_cell(int2(int(std::ceil((tri_max.x + clearance - nav.min.x) / cell)),
                                     int(std::ceil((tri_max.y + clearance - nav.min.y) / cell))),
                                dims_x,
                                dims_y);
    for (int y = start.y; y <= end.y; y++) {
      for (int x = start.x; x <= end.x; x++) {
        const int index = y * dims_x + x;
        if (nav.blocked[index]) {
          continue;
        }
        const float2 center = nav.min + float2((float(x) + 0.5f) * cell, (float(y) + 0.5f) * cell);
        if (distance_to_triangle_2d(center, a, b, c) <= clearance) {
          nav.blocked[index] = 1;
        }
      }
    }
  }
}

/** Nearest walkable cell for a goal; goals are often inside the agent clearance. */
static int find_goal_cell(const NavResult &nav, const float2 goal)
{
  const int2 base = clamp_cell(int2(int(std::floor((goal.x - nav.min.x) / nav.cell_size)),
                                    int(std::floor((goal.y - nav.min.y) / nav.cell_size))),
                               nav.dims_x,
                               nav.dims_y);
  auto cell_distance = [&](const int2 cell) {
    return math::length_squared(cell_center(nav, cell) - goal);
  };
  if (!nav.blocked[flatten(base, nav.dims_x)]) {
    return flatten(base, nav.dims_x);
  }
  int best = -1;
  float best_distance = std::numeric_limits<float>::max();
  constexpr int MAX_RING = 8;
  for (int ring = 1; ring <= MAX_RING; ring++) {
    for (int dy = -ring; dy <= ring; dy++) {
      for (int dx = -ring; dx <= ring; dx++) {
        if (std::max(std::abs(dx), std::abs(dy)) != ring) {
          continue;
        }
        const int2 cell(base.x + dx, base.y + dy);
        if (cell.x < 0 || cell.y < 0 || cell.x >= nav.dims_x || cell.y >= nav.dims_y) {
          continue;
        }
        const int index = flatten(cell, nav.dims_x);
        if (nav.blocked[index]) {
          continue;
        }
        const float distance = cell_distance(cell);
        if (distance < best_distance) {
          best_distance = distance;
          best = index;
        }
      }
    }
    if (best >= 0) {
      return best;
    }
  }
  return -1;
}

struct NeighborOffset {
  int dx;
  int dy;
  float scale;
};

static constexpr NeighborOffset NEIGHBORS[8] = {{1, 0, 1.0f},
                                                {-1, 0, 1.0f},
                                                {0, 1, 1.0f},
                                                {0, -1, 1.0f},
                                                {1, 1, 1.41421356f},
                                                {1, -1, 1.41421356f},
                                                {-1, 1, 1.41421356f},
                                                {-1, -1, 1.41421356f}};

static void build_layer(NavResult &nav,
                        const int layer,
                        const Span<int> goal_cells,
                        const BuildParams &params)
{
  const int cells = nav.cell_count();
  const int offset = layer * cells;
  Array<float> &distances = nav.distance;
  Array<uint8_t> &reachable = nav.reachable;
  for (const int i : IndexRange(cells)) {
    distances[offset + i] = UNREACHABLE_DISTANCE;
    reachable[offset + i] = 0;
    nav.flow[offset + i] = float3(0.0f);
  }
  if (goal_cells.is_empty()) {
    return;
  }

  using QueueItem = std::pair<float, int>;
  std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue;
  for (const int cell : goal_cells) {
    if (cell < 0 || cell >= cells || nav.blocked[cell]) {
      continue;
    }
    if (distances[offset + cell] > 0.0f) {
      distances[offset + cell] = 0.0f;
      queue.emplace(0.0f, cell);
    }
  }
  const float cell_size = nav.cell_size;
  while (!queue.empty()) {
    const auto [distance, cell] = queue.top();
    queue.pop();
    if (distance > distances[offset + cell]) {
      continue;
    }
    const int cx = cell % nav.dims_x;
    const int cy = cell / nav.dims_x;
    for (const NeighborOffset &neighbor : NEIGHBORS) {
      const int nx = cx + neighbor.dx;
      const int ny = cy + neighbor.dy;
      if (nx < 0 || ny < 0 || nx >= nav.dims_x || ny >= nav.dims_y) {
        continue;
      }
      const int next = ny * nav.dims_x + nx;
      if (nav.blocked[next]) {
        continue;
      }
      /* Do not cut corners between two diagonal obstacles. */
      if (neighbor.dx != 0 && neighbor.dy != 0) {
        if (nav.blocked[cy * nav.dims_x + nx] || nav.blocked[ny * nav.dims_x + cx]) {
          continue;
        }
      }
      const float next_distance = distance + neighbor.scale * cell_size;
      if (next_distance < distances[offset + next]) {
        distances[offset + next] = next_distance;
        queue.emplace(next_distance, next);
      }
    }
  }

  /* Flow from the distance field. */
  for (const int y : IndexRange(nav.dims_y)) {
    for (const int x : IndexRange(nav.dims_x)) {
      const int cell = y * nav.dims_x + x;
      if (nav.blocked[cell] || distances[offset + cell] >= UNREACHABLE_DISTANCE) {
        continue;
      }
      reachable[offset + cell] = 1;
      const float d = distances[offset + cell];
      float gx = 0.0f;
      float gy = 0.0f;
      bool has_x = false;
      bool has_y = false;
      if (x > 0 && distances[offset + cell - 1] < UNREACHABLE_DISTANCE) {
        if (x + 1 < nav.dims_x && distances[offset + cell + 1] < UNREACHABLE_DISTANCE) {
          gx = (distances[offset + cell + 1] - distances[offset + cell - 1]) / (2.0f * cell_size);
          has_x = true;
        }
        else {
          gx = (d - distances[offset + cell - 1]) / cell_size;
          has_x = true;
        }
      }
      else if (x + 1 < nav.dims_x && distances[offset + cell + 1] < UNREACHABLE_DISTANCE) {
        gx = (distances[offset + cell + 1] - d) / cell_size;
        has_x = true;
      }
      if (y > 0 && distances[offset + cell - nav.dims_x] < UNREACHABLE_DISTANCE) {
        if (y + 1 < nav.dims_y && distances[offset + cell + nav.dims_x] < UNREACHABLE_DISTANCE) {
          gy = (distances[offset + cell + nav.dims_x] - distances[offset + cell - nav.dims_x]) /
               (2.0f * cell_size);
          has_y = true;
        }
        else {
          gy = (d - distances[offset + cell - nav.dims_x]) / cell_size;
          has_y = true;
        }
      }
      else if (y + 1 < nav.dims_y && distances[offset + cell + nav.dims_x] < UNREACHABLE_DISTANCE)
      {
        gy = (distances[offset + cell + nav.dims_x] - d) / cell_size;
        has_y = true;
      }
      float2 direction(0.0f);
      if (has_x || has_y) {
        direction = float2(-gx, -gy);
      }
      if (math::length_squared(direction) <= 1.0e-12f) {
        /* At the goal, or on a plateau: steer to the lowest neighbour. */
        float best_distance = d;
        for (const NeighborOffset &neighbor : NEIGHBORS) {
          const int nx = x + neighbor.dx;
          const int ny = y + neighbor.dy;
          if (nx < 0 || ny < 0 || nx >= nav.dims_x || ny >= nav.dims_y) {
            continue;
          }
          const int next = ny * nav.dims_x + nx;
          if (nav.blocked[next]) {
            continue;
          }
          const float next_distance = distances[offset + next];
          if (next_distance + 1.0e-4f < best_distance) {
            best_distance = next_distance;
            direction = float2(float(neighbor.dx), float(neighbor.dy));
          }
        }
      }
      if (math::length_squared(direction) > 1.0e-12f) {
        const float2 normalized = math::normalize(direction);
        nav.flow[offset + cell] = float3(normalized.x, normalized.y, 0.0f);
      }
    }
  }

  /* Two smoothing passes remove the staircase pattern of the grid directions. */
  Array<float3> scratch(cells, float3(0.0f));
  for (int pass = 0; pass < 2; pass++) {
    for (const int y : IndexRange(nav.dims_y)) {
      for (const int x : IndexRange(nav.dims_x)) {
        const int cell = y * nav.dims_x + x;
        if (!reachable[offset + cell]) {
          scratch[cell] = float3(0.0f);
          continue;
        }
        float3 sum = nav.flow[offset + cell] * 2.0f;
        float weight = 2.0f;
        for (const NeighborOffset &neighbor : NEIGHBORS) {
          const int nx = x + neighbor.dx;
          const int ny = y + neighbor.dy;
          if (nx < 0 || ny < 0 || nx >= nav.dims_x || ny >= nav.dims_y) {
            continue;
          }
          const int next = ny * nav.dims_x + nx;
          if (!reachable[offset + next]) {
            continue;
          }
          sum += nav.flow[offset + next];
          weight += 1.0f;
        }
        scratch[cell] = weight > 0.0f ? sum / weight : float3(0.0f);
      }
    }
    for (const int cell : IndexRange(cells)) {
      const float3 value = scratch[cell];
      const float length = math::length(float2(value.x, value.y));
      nav.flow[offset + cell] = length > 1.0e-5f ?
                                    float3(value.x / length, value.y / length, 0.0f) :
                                    float3(0.0f);
    }
  }
  (void)params;
}

static std::shared_ptr<NavResult> build_navigation(const Mesh *obstacles,
                                                   const Span<Goal> goals,
                                                   const Span<float2> extra_bounds,
                                                   const BuildParams &params,
                                                   std::string &r_error)
{
  auto nav = std::make_shared<NavResult>();

  float2 min = float2(0.0f);
  float2 max = float2(0.0f);
  bool has_bounds = false;
  auto include_point = [&](const float2 point) {
    if (!has_bounds) {
      min = point;
      max = point;
      has_bounds = true;
    }
    else {
      min.x = std::min(min.x, point.x);
      min.y = std::min(min.y, point.y);
      max.x = std::max(max.x, point.x);
      max.y = std::max(max.y, point.y);
    }
  };

  if (params.use_bounds) {
    min = float2(params.bounds_min.x, params.bounds_min.y);
    max = float2(params.bounds_max.x, params.bounds_max.y);
    has_bounds = true;
  }
  else {
    if (obstacles != nullptr) {
      for (const float3 &position : obstacles->vert_positions()) {
        include_point(float2(position.x, position.y));
      }
    }
    for (const Goal &goal : goals) {
      include_point(goal.position);
    }
    for (const float2 &point : extra_bounds) {
      include_point(point);
    }
  }
  if (!has_bounds) {
    r_error = "no obstacles or goals to derive the navigation bounds from";
    return nullptr;
  }
  const float margin = std::max(params.margin, 0.0f);
  min -= float2(margin);
  max += float2(margin);
  if (max.x - min.x < 1.0e-4f) {
    max.x = min.x + 1.0f;
  }
  if (max.y - min.y < 1.0e-4f) {
    max.y = min.y + 1.0f;
  }

  float cell_size = params.cell_size > 0.0f ? params.cell_size :
                                              std::max(params.agent_radius * 0.75f, 0.05f);
  for (int attempt = 0; attempt < 24; attempt++) {
    const int dims_x = int(std::ceil((max.x - min.x) / cell_size));
    const int dims_y = int(std::ceil((max.y - min.y) / cell_size));
    if (int64_t(dims_x) * int64_t(dims_y) <= MAX_CELLS) {
      nav->dims_x = std::max(dims_x, 1);
      nav->dims_y = std::max(dims_y, 1);
      break;
    }
    cell_size *= 1.25f;
  }
  if (nav->dims_x <= 0 || nav->dims_y <= 0) {
    r_error = "the navigation grid is larger than the maximum cell count";
    return nullptr;
  }
  nav->min = min;
  nav->cell_size = cell_size;
  const int cells = nav->cell_count();
  nav->blocked = Array<uint8_t>(cells, 0);
  if (obstacles != nullptr) {
    rasterize_obstacles(obstacles, *nav, params);
  }

  /* Layers: one per explicit goal group, then the fallback layer with every goal. */
  Vector<int> group_values;
  {
    Set<int> explicit_groups;
    for (const Goal &goal : goals) {
      if (goal.group >= 0) {
        explicit_groups.add(goal.group);
      }
    }
    Vector<int> sorted;
    for (const int group : explicit_groups) {
      sorted.append(group);
    }
    std::sort(sorted.begin(), sorted.end());
    if (sorted.size() > MAX_EXPLICIT_GROUPS) {
      sorted.resize(MAX_EXPLICIT_GROUPS);
    }
    group_values = sorted;
    group_values.append(-1);
  }
  nav->groups = group_values;
  const int layers = int(group_values.size());
  nav->flow = Array<float3>(layers * cells, float3(0.0f));
  nav->distance = Array<float>(layers * cells, UNREACHABLE_DISTANCE);
  nav->reachable = Array<uint8_t>(layers * cells, 0);

  for (const int layer : IndexRange(layers)) {
    const int group = group_values[layer];
    Vector<int> goal_cells;
    for (const Goal &goal : goals) {
      if (group >= 0 && goal.group != group) {
        continue;
      }
      const int cell = find_goal_cell(*nav, goal.position);
      if (cell >= 0) {
        goal_cells.append(cell);
      }
    }
    build_layer(*nav, layer, goal_cells, params);
  }
  return nav;
}

/* -------------------------------------------------------------------- cache key */

static uint64_t navigation_hash(const Mesh *obstacles,
                                const Span<Goal> goals,
                                const Span<float2> extra_bounds,
                                const BuildParams &params,
                                const std::array<float2, 2> &bounds)
{
  uint64_t hash = 1469598103934665603ull;
  if (obstacles != nullptr) {
    const Span<float3> positions = obstacles->vert_positions();
    const Span<int3> tris = obstacles->corner_tris();
    hash_value(hash, int(positions.size()));
    hash_bytes(hash, positions.data(), positions.size() * sizeof(float3));
    hash_value(hash, int(tris.size()));
    hash_bytes(hash, tris.data(), tris.size() * sizeof(int3));
  }
  else {
    hash_value(hash, -1);
  }
  hash_value(hash, int(goals.size()));
  for (const Goal &goal : goals) {
    hash_value(hash, goal.position);
    hash_value(hash, goal.group);
  }
  hash_value(hash, int(extra_bounds.size()));
  for (const float2 &point : extra_bounds) {
    hash_value(hash, point);
  }
  hash_value(hash, params.agent_radius);
  hash_value(hash, params.agent_height);
  hash_value(hash, params.max_step_height);
  hash_value(hash, params.ground_z);
  hash_value(hash, params.cell_size);
  hash_value(hash, params.margin);
  hash_value(hash, int(params.use_bounds));
  hash_value(hash, params.bounds_min);
  hash_value(hash, params.bounds_max);
  hash_value(hash, bounds[0]);
  hash_value(hash, bounds[1]);
  return hash;
}

/* -------------------------------------------------------------------- output */

static PointCloud *create_navigation_points(const NavResult &nav, const float ground_z)
{
  const int cells = nav.cell_count();
  const int layers = int(nav.groups.size());
  const int total = cells * layers;
  PointCloud *points = BKE_pointcloud_new_nomain(total);
  bke::MutableAttributeAccessor attributes = points->attributes_for_write();
  const AttrDomain domain = AttrDomain::Point;

  bke::SpanAttributeWriter<float3> positions = attributes.lookup_or_add_for_write_span<float3>(
      "position", domain);
  bke::SpanAttributeWriter<float3> flows = attributes.lookup_or_add_for_write_span<float3>("flow",
                                                                                           domain);
  bke::SpanAttributeWriter<float> distances = attributes.lookup_or_add_for_write_span<float>(
      "nav_distance", domain);
  bke::SpanAttributeWriter<bool> reachable = attributes.lookup_or_add_for_write_span<bool>(
      "reachable", domain);
  bke::SpanAttributeWriter<bool> blocked = attributes.lookup_or_add_for_write_span<bool>("blocked",
                                                                                         domain);
  bke::SpanAttributeWriter<int> groups = attributes.lookup_or_add_for_write_span<int>("nav_group",
                                                                                      domain);
  bke::SpanAttributeWriter<int> layer_indices = attributes.lookup_or_add_for_write_span<int>(
      "nav_layer", domain);
  bke::SpanAttributeWriter<int> cell_indices = attributes.lookup_or_add_for_write_span<int>(
      "nav_cell", domain);

  int index = 0;
  for (const int layer : IndexRange(layers)) {
    const int offset = layer * cells;
    for (const int y : IndexRange(nav.dims_y)) {
      for (const int x : IndexRange(nav.dims_x)) {
        const int cell = y * nav.dims_x + x;
        const float2 center = cell_center(nav, int2(x, y));
        positions.span[index] = float3(center.x, center.y, ground_z);
        flows.span[index] = nav.flow[offset + cell];
        distances.span[index] = nav.reachable[offset + cell] ? nav.distance[offset + cell] :
                                                               UNREACHABLE_DISTANCE;
        reachable.span[index] = nav.reachable[offset + cell] != 0;
        blocked.span[index] = nav.blocked[cell] != 0;
        groups.span[index] = nav.groups[layer];
        layer_indices.span[index] = layer;
        cell_indices.span[index] = cell;
        index++;
      }
    }
  }

  positions.finish();
  flows.finish();
  distances.finish();
  reachable.finish();
  blocked.finish();
  groups.finish();
  layer_indices.finish();
  cell_indices.finish();
  points->tag_positions_changed();
  return points;
}

/* -------------------------------------------------------------------- node */

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Obstacles"_ustr)
      .description(
          "Obstacle mesh (walls, boxes, ground). Triangles below Ground Z + Max Step Height are "
          "walked over; triangles above Ground Z + Agent Height are ignored");
  b.add_input<decl::Geometry>("Goals"_ustr)
      .description(
          "Goal point cloud or mesh. Points may carry a `target_group` attribute; agents with the "
          "same group follow that group's field");
  b.add_input<decl::Float>("Agent Radius"_ustr)
      .default_value(0.3f)
      .min(0.0f)
      .description("Clearance around obstacles, also the scale for the automatic cell size");
  b.add_input<decl::Float>("Agent Height"_ustr)
      .default_value(1.8f)
      .min(0.0f)
      .description("Obstacles entirely above this height do not block the floor plan");
  b.add_input<decl::Float>("Max Step Height"_ustr)
      .default_value(0.35f)
      .min(0.0f)
      .description("Obstacles no taller than this can be walked over");
  b.add_input<decl::Float>("Ground Z"_ustr)
      .default_value(0.0f)
      .description("Height the crowd walks at; step and head heights are measured from here");
  b.add_input<decl::Float>("Cell Size"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .description("Grid resolution; 0 derives one from the agent radius");
  b.add_input<decl::Float>("Bounds Margin"_ustr)
      .default_value(2.0f)
      .min(0.0f)
      .description("Extra space around the obstacles and goals");
  b.add_input<decl::Bool>("Use Custom Bounds"_ustr)
      .default_value(false)
      .description("Use fixed bounds instead of the obstacle/goal bounding box");
  b.add_input<decl::Vector>("Bounds Min"_ustr)
      .default_value(float3(-10.0f))
      .description("Lower corner of a fixed navigation bound (XY is used)");
  b.add_input<decl::Vector>("Bounds Max"_ustr)
      .default_value(float3(10.0f))
      .description("Upper corner of a fixed navigation bound (XY is used)");
  b.add_input<decl::Geometry>("Bounds Geometry"_ustr)
      .description(
          "Optional geometry the grid must cover, such as the crowd spawn surface. Ignored when "
          "Use Custom Bounds is on");

  b.add_output<decl::Bundle>("Navigation"_ustr).description("Flow field consumed by Crowd Motion");
  b.add_output<decl::Geometry>("Grid"_ustr)
      .description("Debug geometry: one point per cell with the `flow` attribute");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const GeometrySet obstacles_geometry = params.extract_input<GeometrySet>("Obstacles"_ustr);
  const GeometrySet goals_geometry = params.extract_input<GeometrySet>("Goals"_ustr);
  const GeometrySet bounds_geometry = params.extract_input<GeometrySet>("Bounds Geometry"_ustr);

  BuildParams build_params;
  build_params.agent_radius = std::max(params.extract_input<float>("Agent Radius"_ustr), 0.01f);
  build_params.agent_height = std::max(params.extract_input<float>("Agent Height"_ustr), 0.01f);
  build_params.max_step_height = std::max(params.extract_input<float>("Max Step Height"_ustr),
                                          0.0f);
  build_params.ground_z = params.extract_input<float>("Ground Z"_ustr);
  build_params.cell_size = std::max(params.extract_input<float>("Cell Size"_ustr), 0.0f);
  build_params.margin = std::max(params.extract_input<float>("Bounds Margin"_ustr), 0.0f);
  build_params.use_bounds = params.extract_input<bool>("Use Custom Bounds"_ustr);
  build_params.bounds_min = params.extract_input<float3>("Bounds Min"_ustr);
  build_params.bounds_max = params.extract_input<float3>("Bounds Max"_ustr);

  const Mesh *obstacles = obstacles_geometry.get_mesh();
  if (obstacles != nullptr && !mesh_has_data(obstacles)) {
    obstacles = nullptr;
  }

  Vector<Goal> goals;
  read_goals(goals_geometry, goals);
  Vector<float2> extra_bounds;
  read_bounds_points(bounds_geometry, extra_bounds);

  Set<int> unique_groups;
  for (const Goal &goal : goals) {
    if (goal.group >= 0) {
      unique_groups.add(goal.group);
    }
  }
  const bool groups_truncated = unique_groups.size() > MAX_EXPLICIT_GROUPS;

  if (goals.is_empty()) {
    params.error_message_add(NodeWarningType::Info,
                             TIP_("Crowd Navigation: connect a goal point cloud or mesh"));
    params.set_output("Navigation"_ustr, BundlePtr());
    params.set_output("Grid"_ustr, GeometrySet());
    return;
  }

  std::array<float2, 2> bounds = {float2(0.0f), float2(0.0f)};
  if (build_params.use_bounds) {
    bounds[0] = float2(build_params.bounds_min.x, build_params.bounds_min.y);
    bounds[1] = float2(build_params.bounds_max.x, build_params.bounds_max.y);
  }
  const uint64_t key = navigation_hash(obstacles, goals, extra_bounds, build_params, bounds);

  std::shared_ptr<const NavResult> nav = cache_lookup(key);
  if (!nav) {
    std::string error;
    std::shared_ptr<NavResult> built = build_navigation(
        obstacles, goals, extra_bounds, build_params, error);
    if (!built) {
      params.error_message_add(NodeWarningType::Warning,
                               std::string(TIP_("Crowd Navigation: ")) + error);
      params.set_output("Navigation"_ustr, BundlePtr());
      params.set_output("Grid"_ustr, GeometrySet());
      return;
    }
    if (build_params.cell_size <= 0.0f) {
      const int grid_cells = built->cell_count();
      if (grid_cells > 0) {
        /* Only inform when auto resolution had to be coarsened a lot. */
        const float wanted = std::max(build_params.agent_radius * 0.75f, 0.05f);
        if (built->cell_size > wanted * 1.05f) {
          params.error_message_add(
              NodeWarningType::Info,
              fmt::format("Crowd Navigation: the grid cell size was raised from {:.3f} m to "
                          "{:.3f} m to stay within the cell budget",
                          wanted,
                          built->cell_size));
        }
      }
    }
    cache_insert(key, built);
    nav = std::move(built);
  }
  if (groups_truncated) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Crowd Navigation: only the first 16 goal groups get their own field; other agents "
             "use the field with all goals"));
  }

  PointCloud *points = create_navigation_points(*nav, build_params.ground_z);
  GeometrySet grid_geometry = GeometrySet::from_pointcloud(points);

  BundlePtr bundle_ptr = Bundle::create();
  Bundle &bundle = bundle_ptr.ensure_mutable_inplace();
  bundle.add(*BundleKey::from_str("Type"), std::string(NAVIGATION_TYPE));
  bundle.add(*BundleKey::from_str("geometry"), grid_geometry);
  bundle.add(*BundleKey::from_str("min"), float3(nav->min.x, nav->min.y, build_params.ground_z));
  bundle.add(*BundleKey::from_str("cell_size"), nav->cell_size);
  bundle.add(*BundleKey::from_str("dims_x"), nav->dims_x);
  bundle.add(*BundleKey::from_str("dims_y"), nav->dims_y);
  bundle.add(*BundleKey::from_str("group_count"), int(nav->groups.size()));
  for (const int i : nav->groups.index_range()) {
    bundle.add(*BundleKey::from_str("group_" + std::to_string(i)), nav->groups[i]);
  }

  params.set_output("Navigation"_ustr, std::move(bundle_ptr));
  params.set_output("Grid"_ustr, std::move(grid_geometry));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeCrowdNavigation"_ustr);
  ntype.ui_name = "Crowd Navigation";
  ntype.ui_description =
      "Rasterize obstacles into a walkable grid and build a multi-source flow field";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_crowd_navigation_cc
