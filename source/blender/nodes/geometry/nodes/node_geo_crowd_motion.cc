/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Crowd Motion`: one simulation step for a crowd without rigid bodies.
 *
 * Each point accelerates towards its target, is kept apart from nearby agents with a spatial hash
 * (soft separation + optional hard non-overlap), turns to face its velocity and advances
 * `state_time`. It does not create or push any physics body, so a crowd can walk through itself
 * visually only if `Hard Separation` is turned off.
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_crowd.hh"

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_euler_types.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_rotation.hh"
#include "BLI_math_vector.hh"
#include "BLI_string_ref.hh"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "FN_field_evaluation.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_crowd_motion_cc {

struct Target {
  float3 position;
  int group = -1;
};

/** Read-only view of the `Blender.CrowdNavigation` bundle. */
struct NavSampler {
  bool valid = false;
  float2 min = float2(0.0f);
  float cell_size = 0.0f;
  int dims_x = 0;
  int dims_y = 0;
  int cells = 0;
  Vector<int> groups;
  Map<int, int> group_to_layer;
  VArray<float3> flow;
  VArray<float> distance;
  VArray<bool> reachable;
  VArray<bool> blocked;
};

static bool read_navigation(const Bundle *navigation, NavSampler &r_sampler)
{
  if (navigation == nullptr) {
    return false;
  }
  const std::optional<GeometrySet> geometry = navigation->lookup_path<GeometrySet>("geometry");
  const PointCloud *points = geometry ? geometry->get_pointcloud() : nullptr;
  if (points == nullptr) {
    return false;
  }
  const float3 min = navigation->lookup_path<float3>("min").value_or(float3(0.0f));
  const float cell_size = navigation->lookup_path<float>("cell_size").value_or(0.0f);
  const int dims_x = navigation->lookup_path<int>("dims_x").value_or(0);
  const int dims_y = navigation->lookup_path<int>("dims_y").value_or(0);
  const int group_count = navigation->lookup_path<int>("group_count").value_or(0);
  if (cell_size <= 0.0f || dims_x <= 0 || dims_y <= 0 || group_count <= 0) {
    return false;
  }
  NavSampler sampler;
  sampler.min = float2(min.x, min.y);
  sampler.cell_size = cell_size;
  sampler.dims_x = dims_x;
  sampler.dims_y = dims_y;
  sampler.cells = dims_x * dims_y;
  for (const int i : IndexRange(group_count)) {
    const int group = navigation->lookup_path<int>("group_" + std::to_string(i)).value_or(-1);
    sampler.group_to_layer.add(group, i);
    sampler.groups.append(group);
  }
  sampler.flow = points->attributes()
                     .lookup_or_default<float3>("flow", AttrDomain::Point, float3(0.0f))
                     .varray;
  sampler.distance = points->attributes()
                         .lookup_or_default<float>("nav_distance", AttrDomain::Point, 1.0e9f)
                         .varray;
  sampler.reachable =
      points->attributes().lookup_or_default<bool>("reachable", AttrDomain::Point, false).varray;
  sampler.blocked =
      points->attributes().lookup_or_default<bool>("blocked", AttrDomain::Point, false).varray;
  if (sampler.flow.size() < int64_t(sampler.cells) * group_count ||
      sampler.reachable.size() < int64_t(sampler.cells) * group_count ||
      sampler.blocked.size() < sampler.cells)
  {
    return false;
  }
  sampler.valid = true;
  r_sampler = std::move(sampler);
  return true;
}

static int navigation_layer(const NavSampler &nav, const int target_group)
{
  if (const int *layer = nav.group_to_layer.lookup_ptr(target_group)) {
    return *layer;
  }
  if (const int *layer = nav.group_to_layer.lookup_ptr(-1)) {
    return *layer;
  }
  return 0;
}

/**
 * Sample the flow field at a world position. Prefers a bilinear average of the surrounding cell
 * flows; falls back to the nearest reachable cell, which also pushes an agent that was shoved
 * inside an obstacle back out.
 */
static bool sample_navigation(const NavSampler &nav,
                              const float3 position,
                              const int target_group,
                              float3 &r_flow,
                              float3 &r_push,
                              float &r_distance)
{
  r_push = float3(0.0f);
  if (!nav.valid) {
    return false;
  }
  const int layer = navigation_layer(nav, target_group);
  const int offset = layer * nav.cells;
  const float gx = (position.x - nav.min.x) / nav.cell_size - 0.5f;
  const float gy = (position.y - nav.min.y) / nav.cell_size - 0.5f;
  const int x0 = int(std::floor(gx));
  const int y0 = int(std::floor(gy));
  const float fx = gx - float(x0);
  const float fy = gy - float(y0);

  /* Push away from nearby blocked cells. Flow fields alone can lead an agent into the clearance
   * band along a wall; this keeps it off the obstacle while the flow still decides the direction.
   */
  const int push_radius = 2;
  const float push_reach = nav.cell_size * 3.5f;
  const int base_x = math::clamp(int(std::round(gx)), 0, nav.dims_x - 1);
  const int base_y = math::clamp(int(std::round(gy)), 0, nav.dims_y - 1);
  for (int dy = -push_radius; dy <= push_radius; dy++) {
    for (int dx = -push_radius; dx <= push_radius; dx++) {
      const int x = base_x + dx;
      const int y = base_y + dy;
      if (x < 0 || y < 0 || x >= nav.dims_x || y >= nav.dims_y) {
        continue;
      }
      if (!nav.blocked[y * nav.dims_x + x]) {
        continue;
      }
      const float2 center = nav.min + float2((float(x) + 0.5f) * nav.cell_size,
                                             (float(y) + 0.5f) * nav.cell_size);
      const float2 delta = float2(position.x, position.y) - center;
      const float distance = math::length(delta);
      const float weight = math::clamp(1.0f - distance / push_reach, 0.0f, 1.0f);
      if (weight <= 0.0f) {
        continue;
      }
      const float2 direction = distance > 1.0e-5f ? delta / distance : float2(1.0f, 0.0f);
      r_push.x += direction.x * weight;
      r_push.y += direction.y * weight;
    }
  }
  if (math::length(float2(r_push.x, r_push.y)) > 1.0e-4f) {
    const float2 direction = math::normalize(float2(r_push.x, r_push.y));
    r_push.x = direction.x;
    r_push.y = direction.y;
  }
  else {
    r_push = float3(0.0f);
  }

  float3 sum = float3(0.0f);
  float weight_sum = 0.0f;
  float nearest_distance = 1.0e9f;
  for (int dy = 0; dy <= 1; dy++) {
    for (int dx = 0; dx <= 1; dx++) {
      const int x = x0 + dx;
      const int y = y0 + dy;
      if (x < 0 || y < 0 || x >= nav.dims_x || y >= nav.dims_y) {
        continue;
      }
      const int index = offset + y * nav.dims_x + x;
      if (!nav.reachable[index]) {
        continue;
      }
      const float weight = (dx == 0 ? 1.0f - fx : fx) * (dy == 0 ? 1.0f - fy : fy);
      if (weight <= 1.0e-4f) {
        continue;
      }
      const float3 flow = nav.flow[index];
      if (math::length_squared(float2(flow.x, flow.y)) <= 1.0e-8f) {
        nearest_distance = std::min(nearest_distance, nav.distance[index]);
        continue;
      }
      sum += flow * weight;
      weight_sum += weight;
      nearest_distance = std::min(nearest_distance, nav.distance[index]);
    }
  }
  if (weight_sum > 0.2f) {
    const float2 direction = float2(sum.x, sum.y);
    const float length = math::length(direction);
    if (length > 1.0e-5f) {
      r_flow = float3(direction.x / length, direction.y / length, 0.0f);
      r_distance = nearest_distance;
      return true;
    }
    /* At/near the goal the flow is zero, but the distance is still meaningful for arrival. */
    if (nearest_distance <= nav.cell_size * 2.0f) {
      r_flow = float3(0.0f);
      r_distance = nearest_distance;
      return true;
    }
  }

  /* Nearest reachable cell fallback (also handles blocked cells). Prefer a cell that still has a
   * direction; otherwise return the nearest reachable distance so an agent can stop at the goal.
   */
  constexpr int MAX_RING = 4;
  for (int ring = 0; ring <= MAX_RING; ring++) {
    int best_flow = -1;
    float best_flow_length_sq = 0.0f;
    int best_any = -1;
    float best_any_length_sq = 0.0f;
    for (int dy = -ring; dy <= ring; dy++) {
      for (int dx = -ring; dx <= ring; dx++) {
        if (ring > 0 && std::max(std::abs(dx), std::abs(dy)) != ring) {
          continue;
        }
        const int x = base_x + dx;
        const int y = base_y + dy;
        if (x < 0 || y < 0 || x >= nav.dims_x || y >= nav.dims_y) {
          continue;
        }
        const int index = offset + y * nav.dims_x + x;
        if (!nav.reachable[index]) {
          continue;
        }
        const float length_sq = float(dx * dx + dy * dy);
        if (best_any < 0 || length_sq < best_any_length_sq) {
          best_any = index;
          best_any_length_sq = length_sq;
        }
        const float3 flow = nav.flow[index];
        const float flow_length_sq = math::length_squared(float2(flow.x, flow.y));
        if (flow_length_sq <= 1.0e-8f) {
          continue;
        }
        if (best_flow < 0 || length_sq < best_flow_length_sq) {
          best_flow = index;
          best_flow_length_sq = length_sq;
        }
      }
    }
    if (best_flow >= 0) {
      r_flow = nav.flow[best_flow];
      r_distance = nav.distance[best_flow];
      return true;
    }
    if (best_any >= 0) {
      r_flow = float3(0.0f);
      r_distance = nav.distance[best_any];
      return true;
    }
  }
  return false;
}

/**
 * Position correction that keeps an agent out of the blocked navigation cells. The navigation
 * grid already rasterizes the obstacles with the navigation agent radius, so the center of a free
 * cell is a walkable spot. When the center is inside a blocked cell (typically because hard
 * separation pushed the agent sideways), move it to the nearest free cell center.
 */
static float3 navigation_escape(const NavSampler &nav, const float3 position)
{
  if (!nav.valid) {
    return float3(0.0f);
  }
  const float gx = (position.x - nav.min.x) / nav.cell_size;
  const float gy = (position.y - nav.min.y) / nav.cell_size;
  const int x = int(std::floor(gx));
  const int y = int(std::floor(gy));
  if (x < 0 || y < 0 || x >= nav.dims_x || y >= nav.dims_y) {
    return float3(0.0f);
  }
  if (!nav.blocked[y * nav.dims_x + x]) {
    return float3(0.0f);
  }

  /* Chebyshev rings grow by one cell; an outer ring can never be closer than a cell found in the
   * current one, so the first ring with a free cell is enough. */
  constexpr int MAX_RING = 16;
  int best_x = -1;
  int best_y = -1;
  float best_distance_sq = std::numeric_limits<float>::max();
  for (int ring = 1; ring <= MAX_RING; ring++) {
    for (int dy = -ring; dy <= ring; dy++) {
      for (int dx = -ring; dx <= ring; dx++) {
        if (std::max(std::abs(dx), std::abs(dy)) != ring) {
          continue;
        }
        const int cx = x + dx;
        const int cy = y + dy;
        if (cx < 0 || cy < 0 || cx >= nav.dims_x || cy >= nav.dims_y) {
          continue;
        }
        if (nav.blocked[cy * nav.dims_x + cx]) {
          continue;
        }
        const float2 center = nav.min + float2((float(cx) + 0.5f) * nav.cell_size,
                                               (float(cy) + 0.5f) * nav.cell_size);
        const float distance_sq = math::length_squared(center - float2(position.x, position.y));
        if (distance_sq < best_distance_sq) {
          best_distance_sq = distance_sq;
          best_x = cx;
          best_y = cy;
        }
      }
    }
    if (best_x >= 0) {
      break;
    }
  }
  if (best_x < 0) {
    return float3(0.0f);
  }

  const float2 target = nav.min + float2((float(best_x) + 0.5f) * nav.cell_size,
                                         (float(best_y) + 0.5f) * nav.cell_size);
  float2 push = target - float2(position.x, position.y);
  const float max_push = nav.cell_size * 2.0f;
  const float length = math::length(push);
  if (length > max_push) {
    push *= max_push / std::max(length, 1.0e-6f);
  }
  return float3(push.x, push.y, 0.0f);
}

struct AgentGrid {
  float2 min;
  float cell_size = 1.0f;
  int2 dims = int2(1);
  Array<int> offsets;
  Array<int> indices;

  int cell_count() const
  {
    return dims.x * dims.y;
  }

  int flatten(const int2 cell) const
  {
    return cell.y * dims.x + cell.x;
  }

  int2 cell_of(const float3 position) const
  {
    const int2 cell = int2(int(std::floor((position.x - min.x) / cell_size)),
                           int(std::floor((position.y - min.y) / cell_size)));
    return int2(math::clamp(cell.x, 0, dims.x - 1), math::clamp(cell.y, 0, dims.y - 1));
  }
};

static void build_agent_grid(const Span<float3> positions,
                             const float cell_size,
                             AgentGrid &r_grid)
{
  const int count = positions.size();
  if (count == 0) {
    r_grid.offsets = Array<int>(1, 0);
    return;
  }

  float2 min = float2(positions[0].x, positions[0].y);
  float2 max = min;
  for (const float3 &position : positions) {
    min.x = std::min(min.x, position.x);
    min.y = std::min(min.y, position.y);
    max.x = std::max(max.x, position.x);
    max.y = std::max(max.y, position.y);
  }

  float effective_cell = std::max(cell_size, 1.0e-3f);
  int2 dims;
  for (int attempt = 0; attempt < 8; attempt++) {
    dims.x = std::max(1, int(std::ceil((max.x - min.x) / effective_cell)) + 1);
    dims.y = std::max(1, int(std::ceil((max.y - min.y) / effective_cell)) + 1);
    if (int64_t(dims.x) * int64_t(dims.y) <= 4'000'000) {
      break;
    }
    effective_cell *= 1.5f;
  }

  r_grid.min = min;
  r_grid.cell_size = effective_cell;
  r_grid.dims = dims;

  const int cell_count = r_grid.cell_count();
  Array<int> counts(cell_count, 0);
  for (const float3 &position : positions) {
    counts[r_grid.flatten(r_grid.cell_of(position))]++;
  }

  r_grid.offsets = Array<int>(cell_count + 1);
  r_grid.offsets[0] = 0;
  for (const int i : IndexRange(cell_count)) {
    r_grid.offsets[i + 1] = r_grid.offsets[i] + counts[i];
  }

  Array<int> cursor = r_grid.offsets;
  r_grid.indices = Array<int>(count);
  for (const int i : IndexRange(count)) {
    const int cell = r_grid.flatten(r_grid.cell_of(positions[i]));
    r_grid.indices[cursor[cell]++] = i;
  }
}

static void read_targets(const GeometrySet &geometry, Vector<Target> &r_targets)
{
  if (const PointCloud *points = geometry.get_pointcloud()) {
    const bke::AttributeAccessor attributes = points->attributes();
    const VArray<int> groups =
        attributes.lookup_or_default<int>("target_group", AttrDomain::Point, -1).varray;
    const Span<float3> positions = points->positions();
    for (const int i : IndexRange(points->totpoint)) {
      r_targets.append({positions[i], groups[i]});
    }
    return;
  }
  if (const Mesh *mesh = geometry.get_mesh()) {
    for (const float3 &position : mesh->vert_positions()) {
      r_targets.append({position, -1});
    }
  }
}

static Map<std::string, float> read_move_speeds(const Bundle *states)
{
  Map<std::string, float> result;
  if (states == nullptr) {
    return result;
  }
  const int count = states->lookup_path<int>("count").value_or(0);
  for (const int i : IndexRange(count)) {
    const std::optional<BundlePtr> state = states->lookup_path<BundlePtr>("state_" +
                                                                          std::to_string(i));
    if (!state || !*state) {
      continue;
    }
    const std::string name = (*state)->lookup_path<std::string>("name").value_or("");
    if (name.empty()) {
      continue;
    }
    result.add(name, (*state)->lookup_path<float>("move_speed").value_or(1.0f));
  }
  return result;
}

static int find_nearest_target(const Span<Target> targets,
                               const float3 position,
                               const int group,
                               const bool use_group)
{
  int best = -1;
  float best_distance = FLT_MAX;
  for (const int i : targets.index_range()) {
    if (use_group && targets[i].group != group) {
      continue;
    }
    const float distance = math::distance_squared(position, targets[i].position);
    if (distance < best_distance) {
      best_distance = distance;
      best = i;
    }
  }
  return best;
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Points"_ustr)
      .description("Crowd points from Init Crowd Data (one point per agent)");
  b.add_input<decl::Geometry>("Targets"_ustr)
      .description(
          "Optional target point cloud when `Navigation` is connected. With navigation the goals "
          "come from the navigation field; targets are only used as a direct-steering fallback. "
          "Points may carry a `target_group` attribute");
  b.add_input<decl::Bundle>("States"_ustr)
      .description("State table used to look up each state's Move Speed");
  b.add_input<decl::Float>("Delta Time"_ustr).default_value(1.0f / 24.0f).min(0.0f);

  b.add_input<decl::Float>("Max Speed"_ustr)
      .default_value(1.5f)
      .min(0.0f)
      .description("Upper bound for the walking speed (m/s)");
  b.add_input<decl::Float>("Max Acceleration"_ustr).default_value(10.0f).min(0.0f);
  b.add_input<decl::Float>("Turn Rate"_ustr)
      .default_value(8.0f)
      .min(0.0f)
      .description("How quickly the character turns to face its velocity (rad/s)");
  b.add_input<decl::Float>("Arrive Radius"_ustr)
      .default_value(0.5f)
      .min(0.0f)
      .description(
          "Slow down and stop inside this distance. With Navigation this uses the path distance");

  b.add_input<decl::Float>("Agent Radius"_ustr)
      .default_value(0.25f)
      .min(0.0f)
      .description("Radius used for hard non-overlap (a per-point `agent_radius` overrides it)");
  b.add_input<decl::Float>("Avoidance Radius"_ustr)
      .default_value(1.2f)
      .min(0.0f)
      .description("Neighbour search radius for the soft separation force");
  b.add_input<decl::Float>("Separation Strength"_ustr).default_value(4.0f).min(0.0f);
  b.add_input<decl::Bool>("Hard Separation"_ustr)
      .default_value(true)
      .description("Push agents apart after integration so they do not overlap");
  b.add_input<decl::Int>("Max Neighbors"_ustr).default_value(16).min(1).max(64);

  b.add_input<decl::Bundle>("Navigation"_ustr)
      .description(
          "Optional `Crowd Navigation` flow field. When connected, agents follow the flow around "
          "obstacles instead of walking straight at the target");
  b.add_input<decl::Float>("Navigation Strength"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .max(1.0f)
      .description(
          "With Targets connected, blends direct steering (0) with the navigation flow (1). With "
          "only Navigation, acts as a throttle (0 = stop, 1 = full speed)");

  b.add_output<decl::Geometry>("Points"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Points"_ustr);
  const PointCloud *points = geometry.get_pointcloud();
  if (points == nullptr) {
    params.error_message_add(NodeWarningType::Info,
                             TIP_("Crowd Motion: the Points input is not a point cloud"));
    params.set_output("Points"_ustr, std::move(geometry));
    return;
  }

  BundlePtr states = params.extract_input<BundlePtr>("States"_ustr);
  BundlePtr navigation = params.extract_input<BundlePtr>("Navigation"_ustr);
  const float navigation_strength = math::clamp(
      params.extract_input<float>("Navigation Strength"_ustr), 0.0f, 1.0f);
  GeometrySet targets_geometry = params.extract_input<GeometrySet>("Targets"_ustr);
  const float delta_time = std::max(params.extract_input<float>("Delta Time"_ustr), 0.0f);
  const float max_speed = std::max(params.extract_input<float>("Max Speed"_ustr), 0.0f);
  const float max_acceleration = std::max(params.extract_input<float>("Max Acceleration"_ustr),
                                          0.0f);
  const float turn_rate = std::max(params.extract_input<float>("Turn Rate"_ustr), 0.0f);
  const float arrive_radius = std::max(params.extract_input<float>("Arrive Radius"_ustr), 0.0f);
  const float default_radius = std::max(params.extract_input<float>("Agent Radius"_ustr), 0.0f);
  const float avoidance_radius = std::max(params.extract_input<float>("Avoidance Radius"_ustr),
                                          0.0f);
  const float separation_strength = std::max(
      params.extract_input<float>("Separation Strength"_ustr), 0.0f);
  const bool hard_separation = params.extract_input<bool>("Hard Separation"_ustr);
  const int max_neighbors = math::clamp(params.extract_input<int>("Max Neighbors"_ustr), 1, 64);

  Vector<Target> targets;
  read_targets(targets_geometry, targets);
  const Map<std::string, float> move_speeds = read_move_speeds(states.get());
  NavSampler navigation_sampler;
  read_navigation(navigation.get(), navigation_sampler);

  PointCloud *result = BKE_pointcloud_copy_for_eval(points);
  bke::MutableAttributeAccessor attributes = result->attributes_for_write();
  const AttrDomain domain = AttrDomain::Point;

  bke::SpanAttributeWriter<float3> position_writer =
      attributes.lookup_or_add_for_write_span<float3>("position", domain);
  bke::SpanAttributeWriter<float3> velocity_writer =
      attributes.lookup_or_add_for_write_span<float3>("velocity", domain);
  bke::SpanAttributeWriter<math::Quaternion> rotation_writer =
      attributes.lookup_or_add_for_write_span<math::Quaternion>("rotation", domain);
  bke::SpanAttributeWriter<float> state_time_writer =
      attributes.lookup_or_add_for_write_span<float>("state_time", domain);
  bke::SpanAttributeWriter<float> target_distance_writer =
      attributes.lookup_or_add_for_write_span<float>("target_distance", domain);

  const bke::AttributeAccessor src_attributes = points->attributes();
  const VArray<int> target_groups =
      src_attributes.lookup_or_default<int>("target_group", AttrDomain::Point, -1).varray;
  const VArray<float> point_speeds =
      src_attributes.lookup_or_default<float>("speed", AttrDomain::Point, 1.0f).varray;
  const VArray<float> agent_radii =
      src_attributes.lookup_or_default<float>("agent_radius", AttrDomain::Point, 0.0f).varray;
  const VArray<bool> actives =
      src_attributes.lookup_or_default<bool>("active", AttrDomain::Point, true).varray;
  const VArray<MStringProperty> src_states =
      src_attributes.lookup<MStringProperty>("state").varray;
  const VArray<float> src_state_times =
      src_attributes.lookup_or_default<float>("state_time", AttrDomain::Point, 0.0f).varray;
  const VArray<math::Quaternion> src_rotations = src_attributes
                                                     .lookup_or_default<math::Quaternion>(
                                                         "rotation",
                                                         AttrDomain::Point,
                                                         math::Quaternion::identity())
                                                     .varray;
  const VArray<float3> src_velocities =
      src_attributes.lookup_or_default<float3>("velocity", AttrDomain::Point, float3(0.0f)).varray;

  const int count = points->totpoint;
  const Span<float3> src_positions = points->positions();
  Array<float3> positions(count);
  Array<float3> velocities(count);
  Array<math::Quaternion> rotations(count);
  Array<float> radii(count);
  Array<int> target_of(count, -1);
  for (const int i : IndexRange(count)) {
    positions[i] = src_positions[i];
    velocities[i] = src_velocities[i];
    rotations[i] = math::normalize(src_rotations[i]);
    radii[i] = agent_radii[i] > 0.0f ? agent_radii[i] : default_radius;
    if (actives[i] && !targets.is_empty()) {
      const int group = target_groups[i];
      const bool use_group = group >= 0;
      int target = find_nearest_target(targets, positions[i], group, use_group);
      if (target < 0 && use_group) {
        target = find_nearest_target(targets, positions[i], -1, false);
      }
      target_of[i] = target;
    }
  }

  /* Spatial hash for the soft force and the hard non-overlap. */
  const float max_radius = count > 0 ? *std::max_element(
                                           radii.begin(),
                                           radii.end(),
                                           [](const float a, const float b) { return a < b; }) :
                                       default_radius;
  const float cell_size = std::max({avoidance_radius, 2.0f * max_radius, 1.0e-3f});
  AgentGrid grid;
  build_agent_grid(positions, cell_size, grid);

  Array<float3> separation(count, float3(0.0f));
  if (separation_strength > 0.0f && avoidance_radius > 0.0f) {
    threading::parallel_for(IndexRange(count), 512, [&](const IndexRange range) {
      for (const int i : range) {
        if (!actives[i]) {
          continue;
        }
        float3 force = float3(0.0f);
        const float radius = avoidance_radius;
        const int2 center = grid.cell_of(positions[i]);
        for (int dy = -1; dy <= 1; dy++) {
          const int y = center.y + dy;
          if (y < 0 || y >= grid.dims.y) {
            continue;
          }
          for (int dx = -1; dx <= 1; dx++) {
            const int x = center.x + dx;
            if (x < 0 || x >= grid.dims.x) {
              continue;
            }
            const int cell = grid.flatten(int2(x, y));
            const int begin = grid.offsets[cell];
            const int end = grid.offsets[cell + 1];
            int found = 0;
            for (const int slot : IndexRange(begin, end - begin)) {
              const int other = grid.indices[slot];
              if (other == i) {
                continue;
              }
              const float3 delta = positions[i] - positions[other];
              const float distance_sq = math::length_squared(delta);
              if (distance_sq >= radius * radius) {
                continue;
              }
              const float distance = std::sqrt(std::max(distance_sq, 1.0e-12f));
              force += (delta / distance) * (1.0f - distance / radius);
              found++;
              if (found >= max_neighbors) {
                break;
              }
            }
          }
        }
        separation[i] = force * separation_strength;
      }
    });
  }

  threading::parallel_for(IndexRange(count), 512, [&](const IndexRange range) {
    for (const int i : range) {
      if (!actives[i]) {
        state_time_writer.span[i] = src_state_times[i];
        position_writer.span[i] = positions[i];
        velocity_writer.span[i] = velocities[i];
        rotation_writer.span[i] = rotations[i];
        target_distance_writer.span[i] = target_of[i] >= 0 ?
                                             math::length(targets[target_of[i]].position -
                                                          positions[i]) :
                                             1.0e9f;
        continue;
      }

      const std::string state_name = src_states ? crowd::read_string(src_states[i]) : "";
      const float state_move_speed = move_speeds.lookup_default(state_name, 1.0f);

      const bool has_target = target_of[i] >= 0;
      float direct_distance = 1.0e9f;
      float3 direct_direction(0.0f);
      if (has_target) {
        const float3 to_target = targets[target_of[i]].position - positions[i];
        direct_distance = math::length(float3(to_target.x, to_target.y, 0.0f));
        if (direct_distance > 1.0e-6f) {
          direct_direction = float3(to_target.x, to_target.y, 0.0f) / direct_distance;
        }
      }

      float target_distance = has_target ? direct_distance : 1.0e9f;
      float steer_distance = has_target ? direct_distance : 1.0e9f;
      float3 steer_direction = direct_direction;
      float throttle = 1.0f;

      if (navigation_sampler.valid) {
        float3 flow;
        float3 push;
        float path_distance = 1.0e9f;
        if (sample_navigation(
                navigation_sampler, positions[i], target_groups[i], flow, push, path_distance))
        {
          target_distance = path_distance;
          /* Remove the part of the flow that points into the obstacle and add a small outward
           * component, so agents follow the wall instead of grinding into it. */
          const float2 push_2d(push.x, push.y);
          const float push_length = math::length(push_2d);
          if (push_length > 1.0e-4f) {
            const float2 normal = push_2d / push_length;
            float2 tangent(flow.x, flow.y);
            const float inward = std::min(math::dot(tangent, normal), 0.0f);
            tangent -= normal * inward;
            if (math::length(tangent) > 1.0e-4f) {
              tangent = math::normalize(tangent);
            }
            flow = float3(tangent.x, tangent.y, 0.0f) + float3(normal.x, normal.y, 0.0f) * 0.35f;
            const float length = math::length(flow);
            if (length > 1.0e-4f) {
              flow /= length;
            }
          }
          if (has_target) {
            if (navigation_strength > 0.0f) {
              const float3 blended = direct_direction * (1.0f - navigation_strength) +
                                     flow * navigation_strength;
              const float blended_length = math::length(blended);
              if (blended_length > 1.0e-5f) {
                steer_direction = blended / blended_length;
                steer_distance = path_distance;
              }
            }
          }
          else {
            /* Navigation is the only movement input: it supplies both the direction and the
             * arrival distance, and `Navigation Strength` acts as a throttle. */
            steer_direction = flow;
            steer_distance = path_distance;
            throttle = navigation_strength;
          }
        }
      }

      float3 desired_velocity = float3(0.0f);
      if (steer_distance > arrive_radius &&
          math::length(float2(steer_direction.x, steer_direction.y)) > 1.0e-5f)
      {
        float speed = max_speed * std::max(point_speeds[i], 0.0f) *
                      std::max(state_move_speed, 0.0f) * throttle;
        if (arrive_radius > 0.0f) {
          const float slowdown = math::clamp(
              (steer_distance - arrive_radius) / arrive_radius, 0.0f, 1.0f);
          speed *= slowdown;
        }
        desired_velocity = steer_direction * speed;
      }
      target_distance_writer.span[i] = target_distance;

      float3 velocity = velocities[i];
      const float3 delta_v = desired_velocity - velocity;
      const float delta_length = math::length(delta_v);
      if (delta_length > 1.0e-6f) {
        const float max_delta = max_acceleration * delta_time;
        velocity += delta_v * (max_delta > 0.0f ? std::min(1.0f, max_delta / delta_length) : 1.0f);
      }
      velocity += separation[i] * delta_time;

      const float speed_now = math::length(velocity);
      const float speed_limit = max_speed * std::max(point_speeds[i], 0.0f) *
                                std::max(state_move_speed, 1.0e-3f);
      if (speed_now > speed_limit) {
        velocity *= speed_limit / speed_now;
      }

      float3 new_position = positions[i] + velocity * delta_time;

      float3 forward = float3(velocity.x, velocity.y, 0.0f);
      const float forward_length = math::length(forward);
      if (forward_length > 1.0e-4f) {
        /* The character's forward axis is -Y. A rotation of `yaw` about +Z maps local -Y to
         * (sin(yaw), -cos(yaw)), so this yaw makes -Y point along the velocity. */
        const float yaw = std::atan2(forward.x, -forward.y);
        const math::Quaternion target_rotation = math::to_quaternion(
            math::EulerXYZ(float3(0.0f, 0.0f, yaw)));
        rotations[i] = math::interpolate(
            rotations[i], target_rotation, std::min(1.0f, turn_rate * delta_time));
      }

      positions[i] = new_position;
      velocities[i] = velocity;
      position_writer.span[i] = new_position;
      velocity_writer.span[i] = velocity;
      rotation_writer.span[i] = rotations[i];
      state_time_writer.span[i] = src_state_times[i] + delta_time;
    }
  });

  /* Hard non-overlap, Jacobi style: every agent pushes itself away from all overlapping
   * neighbours, then all pushes are applied together (deterministic, independent of thread
   * scheduling). Writing only the agent's own push avoids cross-thread writes: a pair
   * contributes half the overlap to each side. */
  if (hard_separation) {
    for (int iteration = 0; iteration < 2; iteration++) {
      Array<float3> push(count, float3(0.0f));
      threading::parallel_for(IndexRange(count), 512, [&](const IndexRange range) {
        for (const int i : range) {
          if (!actives[i] || radii[i] <= 0.0f) {
            continue;
          }
          const int2 center = grid.cell_of(positions[i]);
          int found = 0;
          for (int dy = -1; dy <= 1; dy++) {
            const int y = center.y + dy;
            if (y < 0 || y >= grid.dims.y) {
              continue;
            }
            for (int dx = -1; dx <= 1; dx++) {
              const int x = center.x + dx;
              if (x < 0 || x >= grid.dims.x) {
                continue;
              }
              const int cell = grid.flatten(int2(x, y));
              const int begin = grid.offsets[cell];
              const int end = grid.offsets[cell + 1];
              for (const int slot : IndexRange(begin, end - begin)) {
                const int other = grid.indices[slot];
                if (other == i) {
                  continue;
                }
                const float3 delta = positions[i] - positions[other];
                const float distance = math::length(delta);
                const float contact = radii[i] + radii[other];
                const float overlap = contact - distance;
                if (overlap <= 0.0f) {
                  continue;
                }
                const float3 direction = distance > 1.0e-6f ? delta / distance :
                                                              float3(1.0f, 0.0f, 0.0f);
                push[i] += direction * (overlap * 0.5f);
                found++;
                if (found >= max_neighbors) {
                  break;
                }
              }
              if (found >= max_neighbors) {
                break;
              }
            }
            if (found >= max_neighbors) {
              break;
            }
          }
        }
      });
      for (const int i : IndexRange(count)) {
        positions[i] += push[i];
        position_writer.span[i] = positions[i];
      }
      build_agent_grid(positions, cell_size, grid);
    }
  }

  /* Walkable constraint, after the agent-agent pushes: separation can squeeze agents into the
   * clearance band around an obstacle. Moving the center to a free navigation cell is a hard
   * constraint, so it runs last and is not mixed with separation. */
  if (navigation_sampler.valid) {
    for (int iteration = 0; iteration < 3; iteration++) {
      Array<float3> push(count, float3(0.0f));
      threading::parallel_for(IndexRange(count), 512, [&](const IndexRange range) {
        for (const int i : range) {
          if (!actives[i]) {
            continue;
          }
          const float3 escape = navigation_escape(navigation_sampler, positions[i]);
          push[i] = escape;
          const float2 escape_2d(escape.x, escape.y);
          const float escape_length = math::length(escape_2d);
          if (escape_length > 1.0e-5f) {
            /* Let the agent slide along the obstacle instead of re-entering it next frame. */
            const float2 direction = escape_2d / escape_length;
            const float inward = std::min(
                math::dot(float2(velocities[i].x, velocities[i].y), direction), 0.0f);
            velocities[i] -= float3(direction.x, direction.y, 0.0f) * inward;
          }
        }
      });
      bool moved = false;
      for (const int i : IndexRange(count)) {
        if (math::length_squared(push[i]) <= 0.0f) {
          continue;
        }
        moved = true;
        positions[i] += push[i];
        position_writer.span[i] = positions[i];
      }
      if (!moved) {
        break;
      }
    }
    for (const int i : IndexRange(count)) {
      velocity_writer.span[i] = velocities[i];
    }
  }

  position_writer.finish();
  velocity_writer.finish();
  rotation_writer.finish();
  state_time_writer.finish();
  target_distance_writer.finish();
  result->tag_positions_changed();

  params.set_output("Points"_ustr, GeometrySet::from_pointcloud(result));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeCrowdMotion"_ustr);
  ntype.ui_name = "Crowd Motion";
  ntype.ui_description =
      "Move crowd agents towards targets with local separation and hard non-overlap";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_crowd_motion_cc
