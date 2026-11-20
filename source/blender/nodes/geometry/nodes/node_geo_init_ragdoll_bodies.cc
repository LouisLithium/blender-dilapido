/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Init Ragdoll Bodies`: turn the bone points from `Armature Bones` into one capsule body per bone.
 *
 * The body origin is the bone head and the capsule is shifted half a bone length along local Y, so
 * the simulated transform maps back to a pose bone with a pure translation and rotation. Mass,
 * radius and damping are written as instance attributes, which is what the rigid body solver reads.
 * The capsule shape itself is described by `shape_*` attributes: the solver builds a Jolt capsule
 * instead of a convex hull of the display mesh.
 */

#include <cmath>

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_vector.hh"
#include "BLI_vector.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_fields.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "DNA_meshdata_types.h"

#include "FN_field_evaluation.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_init_ragdoll_bodies_cc {

/** Smooth display capsule: a cylinder with rounded caps, along local Y. */
static Mesh *make_capsule_mesh(const float half_height, const float radius)
{
  constexpr int segments = 12;
  constexpr int cap_rings = 4;
  const float h = std::max(half_height, 0.0f);
  const float r = std::max(radius, 1.0e-4f);

  Vector<float3> positions;
  positions.append(float3(0.0f, -(h + r), 0.0f)); /* Bottom pole. */

  Vector<int> ring_offsets;
  const auto add_ring = [&](const float y, const float ring_radius) {
    ring_offsets.append(int(positions.size()));
    for (const int i : IndexRange(segments)) {
      const float angle = 2.0f * float(M_PI) * float(i) / float(segments);
      positions.append(float3(ring_radius * std::cos(angle), y, ring_radius * std::sin(angle)));
    }
  };

  /* Bottom hemisphere, ending at the bottom equator ring. */
  for (const int k : IndexRange::from_begin_end(1, cap_rings + 1)) {
    const float theta = 0.5f * float(M_PI) * float(k) / float(cap_rings);
    add_ring(-(h + r * std::cos(theta)), r * std::sin(theta));
  }
  /* Top equator ring. */
  add_ring(h, r);
  /* Top hemisphere, from just below the top pole down to the equator. */
  for (const int k : IndexRange::from_begin_end(1, cap_rings)) {
    const float theta = 0.5f * float(M_PI) * float(cap_rings - k) / float(cap_rings);
    add_ring(h + r * std::cos(theta), r * std::sin(theta));
  }
  const int top_pole = int(positions.size());
  positions.append(float3(0.0f, h + r, 0.0f));

  /* The body origin is the bone head, so the display capsule starts at the origin and ends at
   * `length` along local Y. The collision shape uses the same offset through `shape_offset`. */
  const float3 display_offset(0.0f, h + r, 0.0f);
  for (float3 &position : positions) {
    position += display_offset;
  }

  const int bottom_ring_count = cap_rings;
  const int top_equator_ring = cap_rings;
  const int top_ring_count = cap_rings - 1;
  const int top_first_ring = top_equator_ring + 1;

  Vector<int> corner_verts;
  Vector<int> face_sizes;
  const auto add_triangle = [&](const int a, const int b, const int c) {
    corner_verts.append(a);
    corner_verts.append(b);
    corner_verts.append(c);
    face_sizes.append(3);
  };
  const auto add_quad = [&](const int a, const int b, const int c, const int d) {
    corner_verts.append(a);
    corner_verts.append(b);
    corner_verts.append(c);
    corner_verts.append(d);
    face_sizes.append(4);
  };
  const auto ring_vertex = [&](const int ring, const int i) {
    return ring_offsets[ring] + (i % segments);
  };
  const auto side_quad = [&](const int lower_ring, const int upper_ring) {
    for (const int i : IndexRange(segments)) {
      add_quad(ring_vertex(lower_ring, i),
               ring_vertex(upper_ring, i),
               ring_vertex(upper_ring, i + 1),
               ring_vertex(lower_ring, i + 1));
    }
  };

  for (const int i : IndexRange(segments)) {
    add_triangle(0, ring_vertex(0, i), ring_vertex(0, i + 1));
  }
  for (const int ring : IndexRange(bottom_ring_count - 1)) {
    side_quad(ring, ring + 1);
  }
  side_quad(bottom_ring_count - 1, top_equator_ring);
  side_quad(top_equator_ring, top_first_ring);
  for (const int ring : IndexRange(top_ring_count - 1)) {
    side_quad(top_first_ring + ring, top_first_ring + ring + 1);
  }
  for (const int i : IndexRange(segments)) {
    add_triangle(top_pole, ring_vertex(top_first_ring + top_ring_count - 1, i + 1),
                 ring_vertex(top_first_ring + top_ring_count - 1, i));
  }

  Array<int> face_offsets(face_sizes.size() + 1);
  face_offsets[0] = 0;
  for (const int i : face_sizes.index_range()) {
    face_offsets[i + 1] = face_offsets[i] + face_sizes[i];
  }

  Vector<int2> edge_verts;
  Map<uint64_t, int> edge_of_key;
  Array<int> corner_edges(corner_verts.size());
  int corner = 0;
  for (const int face : face_sizes.index_range()) {
    const int size = face_sizes[face];
    for (const int i : IndexRange(size)) {
      const int va = corner_verts[corner + i];
      const int vb = corner_verts[corner + (i + 1) % size];
      const uint64_t key = (uint64_t(uint32_t(std::min(va, vb))) << 32) |
                           uint64_t(uint32_t(std::max(va, vb)));
      const int edge = edge_of_key.lookup_or_add(key, int(edge_verts.size()));
      if (edge == int(edge_verts.size())) {
        edge_verts.append(int2(va, vb));
      }
      corner_edges[corner + i] = edge;
    }
    corner += size;
  }

  Mesh *mesh = BKE_mesh_new_nomain(int(positions.size()),
                                   int(edge_verts.size()),
                                   int(face_sizes.size()),
                                   int(corner_verts.size()));
  mesh->vert_positions_for_write().copy_from(positions.as_span());
  mesh->edges_for_write().copy_from(edge_verts.as_span());
  mesh->corner_verts_for_write().copy_from(corner_verts.as_span());
  mesh->corner_edges_for_write().copy_from(corner_edges.as_span());
  mesh->face_offsets_for_write().copy_from(face_offsets.as_span());
  mesh->tag_positions_changed();
  return mesh;
}

static float capsule_volume(const float half_height, const float radius)
{
  const float h = std::max(half_height, 0.0f);
  return float(M_PI) * radius * radius * (2.0f * h) +
         (4.0f / 3.0f) * float(M_PI) * radius * radius * radius;
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Bones"_ustr).description("Point cloud from Armature Bones");
  b.add_input<decl::Float>("Density"_ustr)
      .default_value(1000.0f)
      .min(0.0f)
      .description("Material density in kg/m³, used when Mass is 0");
  b.add_input<decl::Float>("Collision Radius"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .structure_type(StructureType::Field)
      .description("Capsule radius; 0 derives it from the bone length (12%)");
  b.add_input<decl::Float>("Mass"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .structure_type(StructureType::Field)
      .description("Mass per bone; 0 computes it from the capsule volume and density");
  b.add_input<decl::Float>("Linear Damping"_ustr)
      .default_value(0.05f)
      .min(0.0f)
      .max(1.0f);
  b.add_input<decl::Float>("Angular Damping"_ustr)
      .default_value(0.05f)
      .min(0.0f)
      .max(1.0f);
  b.add_input<decl::Int>("Group ID"_ustr)
      .default_value(0)
      .min(0)
      .description("Keeps piece ids of several ragdolls apart, like Init Rigid Body");
  b.add_input<decl::Bool>("Animate"_ustr)
      .default_value(false)
      .structure_type(StructureType::Field)
      .description("Drive the bone from the armature pose (kinematic) instead of simulating it");
  b.add_input<decl::Bool>("Active"_ustr)
      .default_value(true)
      .structure_type(StructureType::Field)
      .description("Switch a bone off without removing it from the ragdoll");
  b.add_input<decl::Vector>("Initial Velocity"_ustr)
      .default_value(float3(0.0f))
      .subtype(PROP_VELOCITY)
      .structure_type(StructureType::Field);
  b.add_input<decl::Vector>("Initial Angular Velocity"_ustr)
      .default_value(float3(0.0f))
      .subtype(PROP_VELOCITY)
      .structure_type(StructureType::Field);

  b.add_output<decl::Geometry>("Bodies"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet bones_geometry = params.extract_input<GeometrySet>("Bones"_ustr);
  const PointCloud *bones = bones_geometry.get_pointcloud();
  if (!bones || bones->totpoint == 0) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Init Ragdoll Bodies: connect the Bones output of an Armature Bones node"));
    params.set_default_remaining_outputs();
    return;
  }

  const int count = bones->totpoint;
  const float density = std::max(params.extract_input<float>("Density"_ustr), 0.0f);
  const float linear_damping = math::clamp(params.extract_input<float>("Linear Damping"_ustr),
                                           0.0f,
                                           1.0f);
  const float angular_damping = math::clamp(params.extract_input<float>("Angular Damping"_ustr),
                                            0.0f,
                                            1.0f);
  const int group_id = std::max(params.extract_input<int>("Group ID"_ustr), 0);

  const bke::AttributeAccessor attributes = bones->attributes();
  const VArray<float4x4> pose_matrices =
      attributes.lookup_or_default<float4x4>("bone_pose_matrix", AttrDomain::Point, float4x4::identity())
          .varray;
  const VArray<float> lengths =
      attributes.lookup_or_default<float>("bone_length", AttrDomain::Point, 1.0f).varray;
  const VArray<int> bone_indices =
      attributes.lookup_or_default<int>("bone_index", AttrDomain::Point, -1).varray;
  const VArray<int> bone_parents =
      attributes.lookup_or_default<int>("bone_parent", AttrDomain::Point, -1).varray;
  const VArray<MStringProperty> bone_names =
      attributes.lookup<MStringProperty>("bone_name").varray;
  const VArray<float4x4> bone_rest_matrices =
      attributes
          .lookup_or_default<float4x4>(
              "bone_rest_matrix", AttrDomain::Point, float4x4::identity())
          .varray;

  bke::GeometryFieldContext field_context(*bones, AttrDomain::Point);
  fn::FieldEvaluator evaluator(field_context, count);
  evaluator.add(params.extract_input<Field<float>>("Collision Radius"_ustr));
  evaluator.add(params.extract_input<Field<float>>("Mass"_ustr));
  evaluator.add(params.extract_input<Field<bool>>("Animate"_ustr));
  evaluator.add(params.extract_input<Field<bool>>("Active"_ustr));
  evaluator.add(params.extract_input<Field<float3>>("Initial Velocity"_ustr));
  evaluator.add(params.extract_input<Field<float3>>("Initial Angular Velocity"_ustr));
  evaluator.evaluate();
  const VArray<float> radii = evaluator.get_evaluated<float>(0);
  const VArray<float> masses = evaluator.get_evaluated<float>(1);
  const VArray<bool> animateds = evaluator.get_evaluated<bool>(2);
  const VArray<bool> actives = evaluator.get_evaluated<bool>(3);
  const VArray<float3> velocities = evaluator.get_evaluated<float3>(4);
  const VArray<float3> angular_velocities = evaluator.get_evaluated<float3>(5);

  bke::Instances *instances = new bke::Instances(count);
  MutableSpan<float4x4> transforms = instances->transforms_for_write();
  MutableSpan<int> handles = instances->reference_handles_for_write();

  Array<float> body_masses(count);
  Array<float> body_radii(count);
  Array<float> half_heights(count);
  Array<int> piece_ids(count);
  Array<float3> body_positions(count);

  for (const int i : IndexRange(count)) {
    const float length = std::max(lengths[i], 1.0e-3f);
    float radius = radii[i];
    if (radius <= 0.0f) {
      radius = std::max(length * 0.12f, 0.002f);
    }
    radius = math::clamp(radius, 0.002f, length * 0.5f);
    const float half_height = std::max(0.5f * length - radius, 0.0f);
    const float volume = capsule_volume(half_height, radius);
    float mass = masses[i];
    if (mass <= 0.0f) {
      mass = std::max(volume * density, 1.0e-4f);
    }

    const float4x4 pose = pose_matrices[i];
    const math::Quaternion rotation = math::normalize(math::to_quaternion(pose));
    transforms[i] = math::from_loc_rot_scale<float4x4>(pose.location(), rotation, float3(1.0f));

    bke::GeometrySet display_set;
    display_set.replace_mesh(make_capsule_mesh(half_height, radius));
    handles[i] = instances->add_reference(bke::InstanceReference(std::move(display_set)));

    body_masses[i] = mass;
    body_radii[i] = radius;
    half_heights[i] = half_height;
    const int bone_index = bone_indices[i] >= 0 ? bone_indices[i] : i;
    piece_ids[i] = bone_index + group_id * 1000000;
    body_positions[i] = pose.location();
  }

  {
    bke::MutableAttributeAccessor instance_attributes = instances->attributes_for_write();
    bke::SpanAttributeWriter<int> piece_id_writer =
        instance_attributes.lookup_or_add_for_write_span<int>("piece_id", AttrDomain::Instance);
    bke::SpanAttributeWriter<int> bone_index_writer =
        instance_attributes.lookup_or_add_for_write_span<int>("bone_index", AttrDomain::Instance);
    bke::SpanAttributeWriter<int> bone_parent_writer =
        instance_attributes.lookup_or_add_for_write_span<int>("bone_parent", AttrDomain::Instance);
    bke::SpanAttributeWriter<float> mass_writer =
        instance_attributes.lookup_or_add_for_write_span<float>("mass", AttrDomain::Instance);
    bke::SpanAttributeWriter<float> radius_writer =
        instance_attributes.lookup_or_add_for_write_span<float>("radius", AttrDomain::Instance);
    bke::SpanAttributeWriter<float> linear_writer =
        instance_attributes.lookup_or_add_for_write_span<float>("linear_damping",
                                                                AttrDomain::Instance);
    bke::SpanAttributeWriter<float> angular_writer =
        instance_attributes.lookup_or_add_for_write_span<float>("angular_damping",
                                                                AttrDomain::Instance);
    bke::SpanAttributeWriter<bool> animated_writer =
        instance_attributes.lookup_or_add_for_write_span<bool>("animated", AttrDomain::Instance);
    bke::SpanAttributeWriter<bool> active_writer =
        instance_attributes.lookup_or_add_for_write_span<bool>("active", AttrDomain::Instance);
    bke::SpanAttributeWriter<float3> velocity_writer =
        instance_attributes.lookup_or_add_for_write_span<float3>("velocity", AttrDomain::Instance);
    bke::SpanAttributeWriter<float3> angular_velocity_writer =
        instance_attributes.lookup_or_add_for_write_span<float3>("angular_velocity",
                                                                 AttrDomain::Instance);
    bke::SpanAttributeWriter<int> shape_writer =
        instance_attributes.lookup_or_add_for_write_span<int>("shape", AttrDomain::Instance);
    bke::SpanAttributeWriter<float> half_height_writer =
        instance_attributes.lookup_or_add_for_write_span<float>("shape_half_height",
                                                               AttrDomain::Instance);
    bke::SpanAttributeWriter<float> shape_radius_writer =
        instance_attributes.lookup_or_add_for_write_span<float>("shape_radius",
                                                               AttrDomain::Instance);
    bke::SpanAttributeWriter<float3> shape_offset_writer =
        instance_attributes.lookup_or_add_for_write_span<float3>("shape_offset",
                                                                AttrDomain::Instance);
    bke::SpanAttributeWriter<MStringProperty> bone_name_writer =
        instance_attributes.lookup_or_add_for_write_span<MStringProperty>("bone_name",
                                                                          AttrDomain::Instance);
    bke::SpanAttributeWriter<float4x4> bone_rest_writer =
        instance_attributes.lookup_or_add_for_write_span<float4x4>("bone_rest_matrix",
                                                                   AttrDomain::Instance);

    for (const int i : IndexRange(count)) {
      const int bone_index = bone_indices[i] >= 0 ? bone_indices[i] : i;
      piece_id_writer.span[i] = piece_ids[i];
      bone_index_writer.span[i] = bone_index;
      bone_parent_writer.span[i] = bone_parents[i];
      mass_writer.span[i] = body_masses[i];
      radius_writer.span[i] = body_radii[i];
      linear_writer.span[i] = linear_damping;
      angular_writer.span[i] = angular_damping;
      animated_writer.span[i] = animateds[i];
      active_writer.span[i] = actives[i];
      velocity_writer.span[i] = velocities[i];
      angular_velocity_writer.span[i] = angular_velocities[i];
      shape_writer.span[i] = 1; /* Capsule. */
      half_height_writer.span[i] = half_heights[i];
      shape_radius_writer.span[i] = body_radii[i];
      shape_offset_writer.span[i] = float3(0.0f, half_heights[i] + body_radii[i], 0.0f);
      bone_name_writer.span[i] = bone_names ? bone_names[i] : MStringProperty{};
      bone_rest_writer.span[i] = bone_rest_matrices[i];
    }

    piece_id_writer.finish();
    bone_index_writer.finish();
    bone_parent_writer.finish();
    mass_writer.finish();
    radius_writer.finish();
    linear_writer.finish();
    angular_writer.finish();
    animated_writer.finish();
    active_writer.finish();
    velocity_writer.finish();
    angular_velocity_writer.finish();
    shape_writer.finish();
    half_height_writer.finish();
    shape_radius_writer.finish();
    shape_offset_writer.finish();
    bone_name_writer.finish();
    bone_rest_writer.finish();
  }

  /* A point per bone, so the official force assets can evaluate per-bone forces and the solver can
   * map them back through `piece_id`. */
  PointCloud *body_points = BKE_pointcloud_new_nomain(count);
  {
    bke::MutableAttributeAccessor point_attributes = body_points->attributes_for_write();
    bke::SpanAttributeWriter<float3> positions =
        point_attributes.lookup_or_add_for_write_span<float3>("position", AttrDomain::Point);
    bke::SpanAttributeWriter<int> piece_id_writer =
        point_attributes.lookup_or_add_for_write_span<int>("piece_id", AttrDomain::Point);
    bke::SpanAttributeWriter<int> bone_index_writer =
        point_attributes.lookup_or_add_for_write_span<int>("bone_index", AttrDomain::Point);
    bke::SpanAttributeWriter<float> mass_writer =
        point_attributes.lookup_or_add_for_write_span<float>("mass", AttrDomain::Point);
    bke::SpanAttributeWriter<float> radius_writer =
        point_attributes.lookup_or_add_for_write_span<float>("radius", AttrDomain::Point);
    for (const int i : IndexRange(count)) {
      positions.span[i] = body_positions[i];
      piece_id_writer.span[i] = piece_ids[i];
      bone_index_writer.span[i] = bone_indices[i] >= 0 ? bone_indices[i] : i;
      mass_writer.span[i] = body_masses[i];
      radius_writer.span[i] = body_radii[i];
    }
    positions.finish();
    piece_id_writer.finish();
    bone_index_writer.finish();
    mass_writer.finish();
    radius_writer.finish();
    body_points->tag_positions_changed();
  }

  GeometrySet result;
  result.replace_instances(instances);
  GeometrySet point_set = GeometrySet::from_pointcloud(body_points);
  result.add(*point_set.get_component(GeometryComponent::Type::PointCloud));
  params.set_output("Bodies"_ustr, std::move(result));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeInitRagdollBodies"_ustr);
  ntype.ui_name = "Init Ragdoll Bodies";
  ntype.ui_description =
      "Turn bones into capsule rigid bodies with mass, damping and kinematic flags";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_init_ragdoll_bodies_cc
