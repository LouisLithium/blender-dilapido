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
    const std::optional<BundlePtr> state = states->lookup_path<BundlePtr>(
        "state_" + std::to_string(i));
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

static void write_state(MStringProperty &dst, const StringRef state)
{
  crowd::write_string(dst, state);
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Points"_ustr)
      .description("Crowd points from Init Crowd Data (one point per agent)");
  b.add_input<decl::Geometry>("Targets"_ustr)
      .description(
          "Target point cloud. Points may carry a `target_group` attribute; an agent targets the "
          "nearest point in its own group");
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
      .description("Slow down and stop inside this distance from the target");

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

  b.add_input<decl::Bool>("State By Speed"_ustr)
      .default_value(false)
      .description("Automatically switch between the idle and walk states from the speed");
  b.add_input<decl::Float>("Idle Speed"_ustr).default_value(0.1f).min(0.0f);
  b.add_input<decl::String>("Idle State"_ustr).default_value("Idle");
  b.add_input<decl::String>("Walk State"_ustr).default_value("Walk");

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
  const bool state_by_speed = params.extract_input<bool>("State By Speed"_ustr);
  const float idle_speed = std::max(params.extract_input<float>("Idle Speed"_ustr), 0.0f);
  const std::string idle_state = params.extract_input<std::string>("Idle State"_ustr);
  const std::string walk_state = params.extract_input<std::string>("Walk State"_ustr);

  Vector<Target> targets;
  read_targets(targets_geometry, targets);
  const Map<std::string, float> move_speeds = read_move_speeds(states.get());

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
  bke::SpanAttributeWriter<MStringProperty> state_writer =
      attributes.lookup_or_add_for_write_span<MStringProperty>("state", domain);
  bke::SpanAttributeWriter<MStringProperty> prev_state_writer =
      attributes.lookup_or_add_for_write_span<MStringProperty>("prev_state", domain);
  bke::SpanAttributeWriter<int> state_id_writer =
      attributes.lookup_or_add_for_write_span<int>("state_id", domain);

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
  const VArray<math::Quaternion> src_rotations =
      src_attributes.lookup_or_default<math::Quaternion>("rotation",
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
  const float max_radius = count > 0 ?
                               *std::max_element(radii.begin(),
                                                 radii.end(),
                                                 [](const float a, const float b) {
                                                   return a < b;
                                                 }) :
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
      const float target_distance = target_of[i] >= 0 ?
                                        math::length(targets[target_of[i]].position -
                                                     positions[i]) :
                                        1.0e9f;
      target_distance_writer.span[i] = target_distance;
      float3 desired_velocity = float3(0.0f);
      if (target_of[i] >= 0) {
        const float3 to_target = targets[target_of[i]].position - positions[i];
        const float distance = math::length(float3(to_target.x, to_target.y, 0.0f));
        if (distance > arrive_radius) {
          const float3 direction = float3(to_target.x, to_target.y, 0.0f) /
                                   std::max(distance, 1.0e-6f);
          float speed = max_speed * std::max(point_speeds[i], 0.0f) *
                        std::max(state_move_speed, 0.0f);
          if (arrive_radius > 0.0f) {
            const float slowdown = math::clamp((distance - arrive_radius) / arrive_radius,
                                               0.0f,
                                               1.0f);
            speed *= slowdown;
          }
          desired_velocity = direction * speed;
        }
      }

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
        const float yaw = std::atan2(forward.x, forward.y);
        const math::Quaternion target_rotation = math::to_quaternion(
            math::EulerXYZ(float3(0.0f, 0.0f, yaw)));
        rotations[i] = math::interpolate(rotations[i],
                                         target_rotation,
                                         std::min(1.0f, turn_rate * delta_time));
      }

      float state_time = src_state_times[i] + delta_time;
      std::string new_state = state_name;
      if (state_by_speed) {
        const float horizontal_speed = math::length(float2(velocity.x, velocity.y));
        if (horizontal_speed < idle_speed && state_name != idle_state) {
          new_state = idle_state;
          state_time = 0.0f;
        }
        else if (horizontal_speed >= idle_speed && (state_name.empty() || state_name == idle_state) &&
                 state_name != walk_state)
        {
          new_state = walk_state;
          state_time = 0.0f;
        }
      }

      positions[i] = new_position;
      velocities[i] = velocity;
      position_writer.span[i] = new_position;
      velocity_writer.span[i] = velocity;
      rotation_writer.span[i] = rotations[i];
      state_time_writer.span[i] = state_time;
      if (new_state != state_name) {
        if (!state_name.empty()) {
          write_state(prev_state_writer.span[i], state_name);
        }
        write_state(state_writer.span[i], new_state);
        state_id_writer.span[i] = -1;
      }
    }
  });

  /* Hard non-overlap, Jacobi style: every agent pushes itself away from all overlapping neighbours,
   * then all pushes are applied together (deterministic, independent of thread scheduling). */
  if (hard_separation) {
    for (int iteration = 0; iteration < 2; iteration++) {
      Array<float3> push(count, float3(0.0f));
      threading::parallel_for(IndexRange(count), 512, [&](const IndexRange range) {
        for (const int i : range) {
          if (!actives[i] || radii[i] <= 0.0f) {
            continue;
          }
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
              for (const int slot : IndexRange(begin, end - begin)) {
                const int other = grid.indices[slot];
                if (other <= i) {
                  continue;
                }
                const float3 delta = positions[i] - positions[other];
                const float distance = math::length(delta);
                const float contact = radii[i] + radii[other];
                const float overlap = contact - distance;
                if (overlap <= 0.0f) {
                  continue;
                }
                const float3 direction = distance > 1.0e-6f ?
                                             delta / distance :
                                             float3(1.0f, 0.0f, 0.0f);
                push[i] += direction * (overlap * 0.5f);
                push[other] -= direction * (overlap * 0.5f);
              }
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

  position_writer.finish();
  velocity_writer.finish();
  rotation_writer.finish();
  state_time_writer.finish();
  target_distance_writer.finish();
  state_writer.finish();
  prev_state_writer.finish();
  state_id_writer.finish();
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
