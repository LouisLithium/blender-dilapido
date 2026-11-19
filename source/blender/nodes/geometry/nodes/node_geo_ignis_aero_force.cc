/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Ignis Aero Force`: turn the fields of an Ignis volume into a force and a torque on the rigid
 * bodies of a simulation.
 *
 * The naive version of this (a single sample at the body centre) is almost useless: Ignis is an
 * Eulerian solver and it treats every piece as a solid, so the velocity and density at the centre
 * of a large piece are zero (the flow is blocked at the surface and the interior is not part of
 * the pressure domain). Only small, fast pieces that have moved into free air see a non-zero
 * sample.
 *
 * This node samples the *surface* of each piece instead: a handful of its vertices (evenly picked
 * from the instance reference mesh, or a 7 point cross for point cloud bodies), offset along the
 * normal so the sample lands in the first fluid cell instead of in the solid shell. The drag of
 * those samples is summed into `external_force` / `external_torque` on the instance (or point)
 * domain, which is exactly what `Rigid Body Solver` reads.
 *
 * The force model:
 *
 *   v_rel = v_smoke - (v_body + omega x r)
 *   a     = AeroDrag * DensityScale * rho * min(|v_rel|, max) * v_rel
 *   F     = mass * mean(a) over the samples that actually sit in moving smoke
 *   T     = mean(r x (mass * a))
 *
 * `Density Scale` calibrates Ignis' density (which is not kg/m3); `Aero Drag` is an acceleration
 * scale, so light debris and heavy slabs react the same way.
 *
 * There is deliberately no pressure term: Ignis imposes `p = 0` on solid boundaries, so the
 * pressure at the surface of a closed piece is zero by construction. The momentum that pushes a
 * bluff body is represented by the drag samples instead (the windward face sees the jet, the
 * leeward face sits in its own wake and is excluded by the averaging).
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>

#include "BLI_array.hh"
#include "BLI_index_range.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_vector.hh"
#include "BLI_span.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"
#include "BKE_volume.hh"
#include "BKE_volume_grid.hh"

#include "DNA_mesh_types.h"
#include "DNA_pointcloud_types.h"

#include "NOD_rna_define.hh"

#include "UI_interface_layout.hh"

#include "node_geometry_util.hh"

#ifdef WITH_OPENVDB
#  include <openvdb/openvdb.h>
#  include <openvdb/tools/Interpolation.h>
#endif

namespace blender::nodes::node_geo_ignis_aero_force_cc {

namespace {

/** One sample point on the surface of a piece, in the piece's local space. */
struct LocalSample {
  float3 position;
  float3 normal;
};

/** The sampling data of one instance reference (shared by every instance that uses it). */
struct ReferenceSamples {
  Vector<LocalSample> samples;
  float3 center = float3(0.0f);
  bool valid = false;
};

constexpr const char *GRID_DENSITY = "density";
constexpr const char *GRID_VELOCITY = "velocity";

/** Directions of the fallback for point cloud bodies (a small cross plus the centre). */
const float3 FALLBACK_DIRECTIONS[7] = {
    float3(0.0f, 0.0f, 0.0f),
    float3(1.0f, 0.0f, 0.0f),
    float3(-1.0f, 0.0f, 0.0f),
    float3(0.0f, 1.0f, 0.0f),
    float3(0.0f, -1.0f, 0.0f),
    float3(0.0f, 0.0f, 1.0f),
    float3(0.0f, 0.0f, -1.0f),
};

#ifdef WITH_OPENVDB

template<typename GridT, typename AccessorT, typename ValueT>
ValueT sample_grid(const GridT &grid, const AccessorT &accessor, const float3 &position)
{
  const openvdb::Vec3R world(position.x, position.y, position.z);
  const openvdb::Vec3R index = grid.transform().worldToIndex(world);
  ValueT value;
  openvdb::tools::BoxSampler::sample(accessor, index, value);
  return value;
}

/** Builds the per reference sample set (positions and normals in local space). */
ReferenceSamples build_reference_samples(const Mesh &mesh, const int max_samples)
{
  ReferenceSamples result;
  const Span<float3> positions = mesh.vert_positions();
  if (positions.is_empty()) {
    return result;
  }
  /* Sample triangle centres, not vertices: a cube has only 8 vertices (all at the corners) and a
   * low poly piece would otherwise get no sample in the middle of its windward face. Centre
   * samples also come with a face normal, so the outward offset does not depend on vertex normals
   * (which are easily inverted on hand-built meshes). */
  const Span<int3> corner_tris = mesh.corner_tris();
  const Span<int> corner_verts = mesh.corner_verts();
  if (!corner_tris.is_empty()) {
    const int count = std::clamp(max_samples, 1, int(corner_tris.size()));
    const int stride = std::max(1, int(corner_tris.size()) / count);
    result.samples.reserve(count);
    for (int i = 0; i < int(corner_tris.size()) && result.samples.size() < count; i += stride) {
      const int3 &tri = corner_tris[i];
      const float3 &a = positions[corner_verts[tri.x]];
      const float3 &b = positions[corner_verts[tri.y]];
      const float3 &c = positions[corner_verts[tri.z]];
      const float3 normal = math::cross(b - a, c - a);
      const float normal_length = math::length(normal);
      if (!(normal_length > 1e-12f)) {
        continue;
      }
      LocalSample sample;
      sample.position = (a + b + c) / 3.0f;
      sample.normal = normal / normal_length;
      if (!std::isfinite(sample.position.x) || !std::isfinite(sample.position.y) ||
          !std::isfinite(sample.position.z))
      {
        continue;
      }
      result.samples.append(sample);
    }
  }
  else {
    const Span<float3> normals = mesh.vert_normals();
    const int count = std::clamp(max_samples, 1, int(positions.size()));
    const int stride = std::max(1, int(positions.size()) / count);
    result.samples.reserve(count);
    for (int i = 0; i < int(positions.size()) && result.samples.size() < count; i += stride) {
      LocalSample sample;
      sample.position = positions[i];
      sample.normal = i < int(normals.size()) ? normals[i] : float3(0.0f);
      result.samples.append(sample);
    }
  }
  if (result.samples.is_empty()) {
    return result;
  }

  float3 center(0.0f);
  for (const float3 &position : positions) {
    center += position;
  }
  result.center = center / float(positions.size());

  result.valid = true;
  return result;
}

ReferenceSamples build_sphere_samples(const float radius)
{
  ReferenceSamples result;
  const float safe_radius = std::max(radius, 1e-4f);
  result.samples.reserve(7);
  for (const float3 &direction : FALLBACK_DIRECTIONS) {
    LocalSample sample;
    sample.normal = direction;
    sample.position = direction * safe_radius;
    result.samples.append(sample);
  }
  result.center = float3(0.0f);
  result.valid = true;
  return result;
}

/** Uniformly picks the sample set of one instance reference, cached by reference index. */
const ReferenceSamples &get_samples(Map<int, ReferenceSamples> &cache,
                                    const bke::InstanceReference &reference,
                                    const int reference_i,
                                    const int max_samples,
                                    const float fallback_radius)
{
  ReferenceSamples *cached = cache.lookup_ptr(reference_i);
  if (cached != nullptr) {
    return *cached;
  }
  ReferenceSamples samples;
  if (reference.type() == bke::InstanceReference::Type::GeometrySet) {
    if (const Mesh *mesh = reference.geometry_set().get_mesh()) {
      samples = build_reference_samples(*mesh, max_samples);
    }
  }
  if (!samples.valid) {
    samples = build_sphere_samples(fallback_radius);
  }
  cache.add_new(reference_i, std::move(samples));
  return *cache.lookup_ptr(reference_i);
}

#endif /* WITH_OPENVDB */

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Volume"_ustr)
      .description("The Ignis fields of the previous frame (density and velocity)");
  b.add_input<decl::Geometry>("Pieces"_ustr)
      .description(
          "The pieces of `Init Rigid Body`; force and torque are written on the instance "
          "domain (a point cloud body set gets point attributes)");
  b.add_input<decl::Float>("Aero Drag"_ustr)
      .default_value(1.0f)
      .min(0.0f)
      .description("Acceleration scale of the aerodynamic drag. 0 disables the drag term");
  b.add_input<decl::Float>("Density Scale"_ustr)
      .default_value(5.0f)
      .min(0.0f)
      .description(
          "Calibration from the Ignis `density` grid (about 0..2, not kg/m3) to the aero "
          "density");
  b.add_input<decl::Float>("Max Relative Speed"_ustr)
      .default_value(20.0f)
      .min(0.0f)
      .description(
          "Clamp of |v_smoke - v_body| in m/s. Keeps a blast front from producing huge "
          "forces");
  b.add_input<decl::Float>("Sample Offset"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .description(
          "Extra distance along the surface normal for every sample. 0 uses half a voxel, "
          "which lands in the first fluid cell instead of in the solid shell");
  b.add_input<decl::Int>("Max Samples"_ustr)
      .default_value(32)
      .min(1)
      .max(4096)
      .description("Upper limit of surface samples per piece (the vertices are picked evenly)");
  b.add_input<decl::Float>("Max Force"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .description("Clamp of the total force in N (0 = no clamp)");
  b.add_output<decl::Geometry>("Pieces"_ustr);
}

#ifdef WITH_OPENVDB

/** The grids of the volume and their accessors, kept alive for one evaluation. */
struct VolumeSampler {
  bke::VolumeTreeAccessToken token;
  const openvdb::FloatGrid *density = nullptr;
  const openvdb::Vec3fGrid *velocity = nullptr;
  std::optional<openvdb::FloatGrid::ConstUnsafeAccessor> density_accessor;
  std::optional<openvdb::Vec3fGrid::ConstUnsafeAccessor> velocity_accessor;
  float voxel = 0.1f;

  bool valid() const
  {
    return density != nullptr && velocity != nullptr;
  }

  float sample_density(const float3 &position) const
  {
    return sample_grid<openvdb::FloatGrid, openvdb::FloatGrid::ConstUnsafeAccessor, float>(
        *density, *density_accessor, position);
  }

  float3 sample_velocity(const float3 &position) const
  {
    const openvdb::Vec3f value =
        sample_grid<openvdb::Vec3fGrid, openvdb::Vec3fGrid::ConstUnsafeAccessor, openvdb::Vec3f>(
            *velocity, *velocity_accessor, position);
    return float3(value.x(), value.y(), value.z());
  }
};

static bool init_sampler(const Volume &volume, VolumeSampler &r_sampler)
{
  const bke::VolumeGridData *density_data = BKE_volume_grid_find(&volume, GRID_DENSITY);
  const bke::VolumeGridData *velocity_data = BKE_volume_grid_find(&volume, GRID_VELOCITY);
  if (density_data == nullptr || velocity_data == nullptr) {
    return false;
  }
  if (density_data->grid_type() != VOLUME_GRID_FLOAT ||
      velocity_data->grid_type() != VOLUME_GRID_VECTOR_FLOAT)
  {
    return false;
  }
  const openvdb::GridBase &density_base = density_data->grid(r_sampler.token);
  const openvdb::GridBase &velocity_base = velocity_data->grid(r_sampler.token);
  r_sampler.density = static_cast<const openvdb::FloatGrid *>(&density_base);
  r_sampler.velocity = static_cast<const openvdb::Vec3fGrid *>(&velocity_base);
  r_sampler.density_accessor.emplace(r_sampler.density->getConstUnsafeAccessor());
  r_sampler.velocity_accessor.emplace(r_sampler.velocity->getConstUnsafeAccessor());
  r_sampler.voxel = float(r_sampler.density->transform().voxelSize()[0]);
  return true;
}

/** Force and torque of one body, from its surface samples. */
struct BodyForce {
  float3 force = float3(0.0f);
  float3 torque = float3(0.0f);
  bool any_sample = false;
};

static BodyForce evaluate_body(const VolumeSampler &sampler,
                               const ReferenceSamples &samples,
                               const float4x4 &transform,
                               const float3 &com_world,
                               const float3 &velocity,
                               const float3 &angular_velocity,
                               const float mass,
                               const float sample_offset,
                               const float aero_drag,
                               const float density_scale,
                               const float max_relative_speed,
                               const float max_force)
{
  BodyForce result;
  if (samples.samples.is_empty() || aero_drag <= 0.0f) {
    return result;
  }

  /* A body in a jet has a very uneven surface field: the windward face is a stagnation region
   * (slow), the edges see the free stream, and the leeward face sits in the body's own
   * recirculation. Averaging all samples can cancel the force and even reverse it. So the force
   * direction and magnitude come from the *strongest* band of samples (the ones in the free
   * stream), and the surface distribution is only used for the torque. */
  struct SampleState {
    float3 r;
    float3 v_relative;
    float3 normal;
    float rho;
    float speed;
  };
  Vector<SampleState> states;
  states.reserve(samples.samples.size());
  float max_speed = 0.0f;

  for (const LocalSample &local : samples.samples) {
    const float3 position = math::transform_point(transform, local.position);
    float3 normal = math::transform_direction(transform, local.normal);
    const float normal_length = math::length(normal);
    normal = normal_length > 1e-8f ? normal / normal_length : float3(0.0f);
    float3 sample_position = position + normal * sample_offset;
    float3 v_smoke = sampler.sample_velocity(sample_position);
    float rho = sampler.sample_density(sample_position);

    /* Hand-built and imported meshes are not always wound outwards. Sampling along the inverted
     * normal would land inside the piece (0 everywhere), so if this side is dead try the other
     * side and use whichever actually holds moving smoke. */
    if (!(rho > 0.0f) || math::length_squared(v_smoke) <= 0.0f) {
      const float3 other_position = position - normal * sample_offset;
      const float3 other_velocity = sampler.sample_velocity(other_position);
      const float other_rho = sampler.sample_density(other_position);
      if (other_rho * math::length(other_velocity) > rho * math::length(v_smoke)) {
        sample_position = other_position;
        v_smoke = other_velocity;
        rho = other_rho;
        normal = -normal;
      }
    }
    rho *= density_scale;

    const float3 r = position - com_world;
    const float3 v_point = velocity + math::cross(angular_velocity, r);
    const float3 v_relative = v_smoke - v_point;
    const float speed = math::length(v_relative);
    if (!(rho > 0.0f) || !(speed > 0.0f) || !std::isfinite(speed)) {
      continue;
    }
    SampleState state;
    state.r = r;
    state.v_relative = v_relative;
    state.normal = normal;
    state.rho = rho;
    state.speed = speed;
    states.append(state);
    max_speed = std::max(max_speed, speed);
  }

  if (states.is_empty() || max_speed <= 0.0f) {
    return {};
  }

  /* Momentum weighted average of the strong band. 0.25 of the peak keeps the edges of a jet and
   * drops the stagnation/wake samples. */
  const float band = 0.25f * max_speed;
  float3 velocity_sum(0.0f);
  float rho_sum = 0.0f;
  float weight_sum = 0.0f;
  for (const SampleState &state : states) {
    if (state.speed < band) {
      continue;
    }
    const float weight = state.rho * state.speed;
    velocity_sum += state.v_relative * weight;
    rho_sum += state.rho * weight;
    weight_sum += weight;
  }
  if (!(weight_sum > 0.0f)) {
    return {};
  }
  const float3 effective_velocity = velocity_sum / weight_sum;
  const float effective_rho = rho_sum / weight_sum;
  const float effective_speed = math::length(effective_velocity);
  if (!(effective_speed > 0.0f) || !(effective_rho > 0.0f)) {
    return {};
  }
  const float clamped_speed = std::min(effective_speed, max_relative_speed);
  result.force = effective_velocity * (mass * aero_drag * effective_rho * clamped_speed);
  result.any_sample = true;
  if (std::getenv("BLENDER_IGNIS_AERO_DEBUG") != nullptr) {
    printf(
        "IGNIS_AERO body samples=%d states=%d max_speed=%.4f eff_speed=%.4f "
        "eff_rho=%.4f mass=%.4f F=(%.4f %.4f %.4f)\n",
        int(samples.samples.size()),
        int(states.size()),
        max_speed,
        effective_speed,
        effective_rho,
        mass,
        result.force.x,
        result.force.y,
        result.force.z);
    fflush(stdout);
  }

  /* Torque: distribute that force over the windward surface (normal facing the effective flow). */
  const float3 effective_dir = effective_velocity / effective_speed;
  float weight_total = 0.0f;
  for (const SampleState &state : states) {
    const float facing = -math::dot(state.normal, effective_dir);
    if (facing > 0.0f) {
      weight_total += facing * state.rho * state.speed;
    }
  }
  if (weight_total > 0.0f) {
    for (const SampleState &state : states) {
      const float facing = -math::dot(state.normal, effective_dir);
      if (facing <= 0.0f) {
        continue;
      }
      const float weight = facing * state.rho * state.speed / weight_total;
      result.torque += math::cross(state.r, result.force * weight);
    }
  }

  if (max_force > 0.0f) {
    const float length = math::length(result.force);
    if (length > max_force) {
      const float scale = max_force / length;
      result.force *= scale;
      result.torque *= scale;
    }
  }
  if (!std::isfinite(result.force.x) || !std::isfinite(result.force.y) ||
      !std::isfinite(result.force.z) || !std::isfinite(result.torque.x) ||
      !std::isfinite(result.torque.y) || !std::isfinite(result.torque.z))
  {
    return {};
  }
  return result;
}

#endif /* WITH_OPENVDB */

static void write_force_instances(bke::Instances &instances, const Span<BodyForce> forces)
{
  bke::MutableAttributeAccessor attributes = instances.attributes_for_write();
  bke::SpanAttributeWriter<float3> force_writer = attributes.lookup_or_add_for_write_span<float3>(
      "external_force", AttrDomain::Instance);
  bke::SpanAttributeWriter<float3> torque_writer = attributes.lookup_or_add_for_write_span<float3>(
      "external_torque", AttrDomain::Instance);
  for (const int i : forces.index_range()) {
    force_writer.span[i] = forces[i].force;
    torque_writer.span[i] = forces[i].torque;
  }
  force_writer.finish();
  torque_writer.finish();
}

static void write_force_points(PointCloud &pointcloud, const Span<BodyForce> forces)
{
  bke::MutableAttributeAccessor attributes = pointcloud.attributes_for_write();
  bke::SpanAttributeWriter<float3> force_writer = attributes.lookup_or_add_for_write_span<float3>(
      "external_force", AttrDomain::Point);
  bke::SpanAttributeWriter<float3> torque_writer = attributes.lookup_or_add_for_write_span<float3>(
      "external_torque", AttrDomain::Point);
  for (const int i : forces.index_range()) {
    force_writer.span[i] = forces[i].force;
    torque_writer.span[i] = forces[i].torque;
  }
  force_writer.finish();
  torque_writer.finish();
}

static void node_geo_exec(GeoNodeExecParams params)
{
#ifdef WITH_OPENVDB
  GeometrySet geometry = params.extract_input<GeometrySet>("Pieces"_ustr);
  const GeometrySet volume_geometry = params.extract_input<GeometrySet>("Volume"_ustr);
  const Volume *volume = volume_geometry.get_volume();
  if (volume == nullptr) {
    /* The first frame has no volume yet; there is nothing to push with. */
    params.set_output("Pieces"_ustr, std::move(geometry));
    return;
  }

  const float aero_drag = std::max(params.extract_input<float>("Aero Drag"_ustr), 0.0f);
  const float density_scale = std::max(params.extract_input<float>("Density Scale"_ustr), 0.0f);
  const float max_relative_speed = std::max(params.extract_input<float>("Max Relative Speed"_ustr),
                                            0.0f);
  const float sample_offset_input = std::max(params.extract_input<float>("Sample Offset"_ustr),
                                             0.0f);
  const int max_samples = std::clamp(params.extract_input<int>("Max Samples"_ustr), 1, 4096);
  const float max_force = std::max(params.extract_input<float>("Max Force"_ustr), 0.0f);

  VolumeSampler sampler;
  if (!init_sampler(*volume, sampler)) {
    params.error_message_add(NodeWarningType::Info,
                             "Ignis Aero Force: the volume has no density/velocity grid");
    params.set_output("Pieces"_ustr, std::move(geometry));
    return;
  }
  const float sample_offset = sample_offset_input > 0.0f ? sample_offset_input :
                                                           1.5f * sampler.voxel;

  if (geometry.has_instances()) {
    bke::Instances &instances = *geometry.get_instances_for_write();
    const Span<float4x4> transforms = instances.transforms();
    const Span<int> reference_handles = instances.reference_handles();
    const Span<bke::InstanceReference> references = instances.references();
    const int count = instances.instances_num();
    if (count == 0) {
      params.set_output("Pieces"_ustr, std::move(geometry));
      return;
    }

    const bke::AttributeAccessor attributes = instances.attributes();
    const VArray<float3> velocities = attributes
                                          .lookup_or_default<float3>(
                                              "velocity", AttrDomain::Instance, float3(0.0f))
                                          .varray;
    const VArray<float3> angular_velocities = attributes
                                                  .lookup_or_default<float3>("angular_velocity",
                                                                             AttrDomain::Instance,
                                                                             float3(0.0f))
                                                  .varray;
    const VArray<float> masses =
        attributes.lookup_or_default<float>("mass", AttrDomain::Instance, 1.0f).varray;
    const VArray<float> radii =
        attributes.lookup_or_default<float>("radius", AttrDomain::Instance, 0.1f).varray;

    Map<int, ReferenceSamples> sample_cache;
    Array<BodyForce> forces(count);
    for (const int i : IndexRange(count)) {
      const int reference_i = reference_handles[i];
      if (reference_i < 0 || reference_i >= references.size()) {
        continue;
      }
      const ReferenceSamples &samples = get_samples(
          sample_cache, references[reference_i], reference_i, max_samples, radii[i]);
      if (!samples.valid) {
        continue;
      }
      const float4x4 &transform = transforms[i];
      const float3 com = math::transform_point(transform, samples.center);
      forces[i] = evaluate_body(sampler,
                                samples,
                                transform,
                                com,
                                velocities[i],
                                angular_velocities[i],
                                masses[i],
                                sample_offset,
                                aero_drag,
                                density_scale,
                                max_relative_speed,
                                max_force);
    }
    write_force_instances(instances, forces);
  }
  else if (PointCloud *pointcloud = geometry.get_pointcloud_for_write()) {
    const int count = pointcloud->totpoint;
    if (count == 0) {
      params.set_output("Pieces"_ustr, std::move(geometry));
      return;
    }
    const bke::AttributeAccessor attributes = pointcloud->attributes();
    const Span<float3> positions = pointcloud->positions();
    const VArray<float3> velocities =
        attributes.lookup_or_default<float3>("velocity", AttrDomain::Point, float3(0.0f)).varray;
    const VArray<float3> angular_velocities =
        attributes.lookup_or_default<float3>("angular_velocity", AttrDomain::Point, float3(0.0f))
            .varray;
    const VArray<float> masses =
        attributes.lookup_or_default<float>("mass", AttrDomain::Point, 1.0f).varray;
    const VArray<float> radii =
        attributes.lookup_or_default<float>("radius", AttrDomain::Point, 0.1f).varray;

    Array<BodyForce> forces(count);
    for (const int i : IndexRange(count)) {
      const ReferenceSamples samples = build_sphere_samples(radii[i]);
      const float4x4 transform = float4x4::identity();
      const float3 center = positions[i];
      forces[i] = evaluate_body(sampler,
                                samples,
                                transform,
                                center,
                                velocities[i],
                                angular_velocities[i],
                                masses[i],
                                sample_offset,
                                aero_drag,
                                density_scale,
                                max_relative_speed,
                                max_force);
    }
    write_force_points(*pointcloud, forces);
  }

  params.set_output("Pieces"_ustr, std::move(geometry));
#else
  node_geo_exec_with_missing_openvdb(params);
#endif
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeIgnisAeroForce"_ustr);
  ntype.ui_name = "Ignis Aero Force";
  ntype.ui_description =
      "Sample the density and velocity of an Ignis volume on the surface of rigid body "
      "pieces and write external_force / external_torque";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_ignis_aero_force_cc
