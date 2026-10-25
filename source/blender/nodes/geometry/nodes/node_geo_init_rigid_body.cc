/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Init Rigid Body node.
 *
 * Takes the raw fracture output (`Voronoi Fracture` or anything else that produces pieces as
 * instances plus a constraint point cloud) and fills in everything the simulation needs on top of
 * the geometry:
 *
 * - Pieces: `mass`, derived from the piece volume and the density input.
 * - Bodies: one point per piece, used to evaluate forces on.
 *
 * The *behaviour* of the constraints (type, break thresholds, fallback, hardening) lives on the
 * `RBD Constraint from Pieces` node, which owns the constraint cloud. This node only passes the
 * constraints through.
 */

#include <algorithm>

#include "BKE_attribute.hh"
#include "BKE_geometry_fields.hh"

#include "FN_field_evaluation.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"
#include "BLI_array.hh"
#include "BLI_math_vector.hh"
#include "BLI_span.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_init_rigid_body_cc {

/** Piece ids of different rigid body groups are spaced by this much, so they never collide. */
static constexpr int k_group_stride = 1000000;

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Pieces"_ustr)
      .description("Pieces as instances, e.g. from the Voronoi Fracture node");

  b.add_input<decl::Float>("Density"_ustr)
      .default_value(1000.0f)
      .min(0.0f)
      .description("Mass per unit volume of the pieces, used to derive their mass");
  b.add_input<decl::Int>("Group ID"_ustr)
      .default_value(0)
      .min(0)
      .description("Which rigid body group these pieces belong to. Give every set that is solved "
                   "together a different small number (0, 1, 2, ...): the piece ids are made unique "
                   "automatically, no need to know how many pieces the other groups have");
  b.add_input<decl::Bool>("Animate"_ustr)
      .default_value(false)
      .description(
          "Let the pieces follow their animated transforms (the node tree animates them) instead of "
          "the simulation: they still collide with and push the other bodies, but nothing can move "
          "them. Uncheck this while the animation runs to hand the motion over to the solver; the "
          "bodies keep the velocity they had while they were animated");
  b.add_input<decl::Bool>("Active"_ustr)
      .default_value(true)
      .structure_type(StructureType::Field)
      .description(
          "Whether the piece is simulated. An inactive piece is not moved by the solver, but the "
          "other bodies still collide with it (a kinematic body, like an animated one). Supports a "
          "field and is read every frame, so a piece can be switched on or off while the simulation "
          "is running");
  b.add_input<decl::Vector>("Initial Velocity"_ustr)
      .default_value(float3(0.0f))
      .subtype(PROP_VELOCITY)
      .structure_type(StructureType::Field)
      .description(
          "Velocity the pieces start with, in m/s. Supports a field: use e.g. a Random Value or "
          "the piece position to give every piece its own velocity (an explosion), or keep it "
          "constant to launch the whole set in one direction. Zero leaves a `velocity` attribute "
          "that was stored upstream untouched");
  b.add_input<decl::Vector>("Initial Angular Velocity"_ustr)
      .default_value(float3(0.0f))
      .subtype(PROP_VELOCITY)
      .structure_type(StructureType::Field)
      .description(
          "Angular velocity the pieces start with, in radians/s (axis * speed). Supports a field "
          "as well. Zero leaves an `angular_velocity` attribute stored upstream untouched");
  b.add_input<decl::Float>("Linear Damping"_ustr)
      .default_value(0.05f)
      .min(0.0f)
      .max(1.0f)
      .description("How much the linear velocity of every piece is damped per second");
  b.add_input<decl::Float>("Angular Damping"_ustr)
      .default_value(0.05f)
      .min(0.0f)
      .max(1.0f)
      .description("How much the angular velocity of every piece is damped per second");

  b.add_output<decl::Geometry>("Pieces"_ustr);

}

/** Volume of a closed mesh, via the divergence theorem (sum of the signed tetrahedron volumes). */
static float mesh_volume(const Mesh &mesh)
{
  const Span<float3> positions = mesh.vert_positions();
  if (positions.size() < 4) {
    return 0.0f;
  }
  /* `corner_tris()` contains corner indices; map them to vertex indices first. */
  Array<int3> vert_tris(mesh.corner_tris().size());
  bke::mesh::vert_tris_from_corner_tris(mesh.corner_verts(), mesh.corner_tris(), vert_tris);
  double volume = 0.0;
  for (const int3 &tri : vert_tris) {
    const float3 &a = positions[tri.x];
    const float3 &b = positions[tri.y];
    const float3 &c = positions[tri.z];
    volume += math::dot(float3(a), math::cross(float3(b), float3(c)));
  }
  return float(std::abs(volume) / 6.0);
}

/** Surface area of a mesh (sum of the triangle areas), used for shell pieces. */
static float mesh_surface_area(const Mesh &mesh)
{
  const Span<float3> positions = mesh.vert_positions();
  if (positions.size() < 3) {
    return 0.0f;
  }
  Array<int3> vert_tris(mesh.corner_tris().size());
  bke::mesh::vert_tris_from_corner_tris(mesh.corner_verts(), mesh.corner_tris(), vert_tris);
  double area = 0.0;
  for (const int3 &tri : vert_tris) {
    const float3 &a = positions[tri.x];
    const float3 &b = positions[tri.y];
    const float3 &c = positions[tri.z];
    const float3 cross = math::cross(b - a, c - a);
    area += 0.5 * double(std::sqrt(math::dot(cross, cross)));
  }
  return float(area);
}

static void init_pieces(GeometrySet &pieces,
                        const float density,
                        const float linear_damping,
                        const float angular_damping,
                        const int group_id)
{
  if (!pieces.has_instances()) {
    return;
  }
  bke::Instances *instances = pieces.get_instances_for_write();
  if (!instances) {
    return;
  }
  const int instances_num = instances->instances_num();
  if (instances_num == 0) {
    return;
  }
  const Span<int> reference_handles = instances->reference_handles();
  const Span<bke::InstanceReference> references = instances->references();
  bke::MutableAttributeAccessor instance_attributes = instances->attributes_for_write();
  /* `piece_id` is what the constraints (and the per-body force cloud) reference, so it has to be
   * unique across everything that is solved together. */
  const bool had_piece_ids = instance_attributes.contains("piece_id");
  bke::SpanAttributeWriter<int> piece_ids = instance_attributes.lookup_or_add_for_write_span<int>(
      "piece_id", AttrDomain::Instance);
  bke::SpanAttributeWriter<float> masses = instance_attributes.lookup_or_add_for_write_span<float>(
      "mass", AttrDomain::Instance);
  /* Written by the Voronoi Fracture node in shell mode: the pieces are open shells, so their mass
   * comes from the surface area and this thickness instead of the volume. */
  const VArray<float> shell_thicknesses = instance_attributes
                                              .lookup_or_default<float>(
                                                  "shell_thickness", AttrDomain::Instance, 0.0f)
                                              .varray;
  bke::SpanAttributeWriter<float> linear_dampings =
      instance_attributes.lookup_or_add_for_write_span<float>("linear_damping",
                                                              AttrDomain::Instance);
  bke::SpanAttributeWriter<float> angular_dampings =
      instance_attributes.lookup_or_add_for_write_span<float>("angular_damping",
                                                              AttrDomain::Instance);
  for (const int i : IndexRange(instances_num)) {
    const int reference_i = reference_handles[i];
    if (reference_i < 0 || reference_i >= references.size()) {
      continue;
    }
    const bke::InstanceReference &reference = references[reference_i];
    if (reference.type() != bke::InstanceReference::Type::GeometrySet) {
      continue;
    }
    const Mesh *mesh = reference.geometry_set().get_mesh();
    if (!mesh) {
      continue;
    }
    const float shell_thickness = shell_thicknesses[i];
    float volume_or_shell = 0.0f;
    if (shell_thickness > 0.0f) {
      volume_or_shell = mesh_surface_area(*mesh) * shell_thickness;
    }
    else {
      volume_or_shell = mesh_volume(*mesh);
    }
    float mass = volume_or_shell * density;
    if (!(mass > 0.0f)) {
      /* A piece can come out without usable faces (a sliver, or a shell whose faces all collapsed).
       * A zero mass makes Jolt treat the body as infinitely heavy, so it does not move at all and
       * only throws its neighbours around. Fall back to a small estimate from the bounding box so
       * the piece behaves like a (very light) piece of geometry instead. */
      float3 min_point(FLT_MAX, FLT_MAX, FLT_MAX);
      float3 max_point(-FLT_MAX, -FLT_MAX, -FLT_MAX);
      for (const float3 &p : mesh->vert_positions()) {
        min_point.x = std::min(min_point.x, p.x);
        min_point.y = std::min(min_point.y, p.y);
        min_point.z = std::min(min_point.z, p.z);
        max_point.x = std::max(max_point.x, p.x);
        max_point.y = std::max(max_point.y, p.y);
        max_point.z = std::max(max_point.z, p.z);
      }
      const float3 size = math::max(max_point - min_point, float3(0.0f));
      const float estimate = size.x * size.y * size.z * 0.1f;
      mass = std::max(estimate, 1.0e-4f) * density;
      if (std::getenv("BLENDER_RBD_DEBUG") != nullptr) {
        printf("RBD_DEBUG piece %d has no usable volume, mass estimate %.6f kg\n",
               i,
               double(mass));
      }
    }
    masses.span[i] = mass;
    linear_dampings.span[i] = linear_damping;
    angular_dampings.span[i] = angular_damping;
    /* The group id is packed into the high digits, so the ids of different groups can never collide
     * while the user only has to type a small number (no need to count the pieces of other sets). */
    piece_ids.span[i] = (had_piece_ids ? piece_ids.span[i] : i) +
                        group_id * k_group_stride;
  }
  /* A piece without usable faces (or without a mesh at all) would end up with a mass of exactly
   * zero, which Jolt turns into an infinitely heavy body: it never moves by itself and only pushes
   * its neighbours around. Give every piece at least 1 g. */
  for (const int i : IndexRange(instances_num)) {
    if (!(masses.span[i] > 1.0e-3f)) {
      masses.span[i] = 1.0e-3f;
    }
  }
  piece_ids.finish();
  masses.finish();
  linear_dampings.finish();
  angular_dampings.finish();
}

/**
 * One point per piece, carrying `piece_id` and `mass`. Forces are evaluated on this cloud (the
 * official force assets store into the point domain) and the solver maps them back to the pieces by
 * `piece_id`.
 */
static GeometrySet make_bodies_cloud(const bke::Instances &instances)
{
  const int count = instances.instances_num();
  if (count == 0) {
    return {};
  }
  PointCloud *pointcloud = bke::pointcloud_new_no_attributes(count);
  const Span<float4x4> transforms = instances.transforms();
  MutableSpan<float3> positions = pointcloud->positions_for_write();
  for (const int i : IndexRange(count)) {
    positions[i] = transforms[i].location();
  }
  const bke::AttributeAccessor instance_attributes = instances.attributes();
  const VArray<int> piece_ids = instance_attributes
                                    .lookup_or_default<int>("piece_id", AttrDomain::Instance, -1)
                                    .varray;
  const VArray<float> masses = instance_attributes
                                   .lookup_or_default<float>("mass", AttrDomain::Instance, 1.0f)
                                   .varray;
  bke::MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
  bke::SpanAttributeWriter<int> point_piece_ids = attributes.lookup_or_add_for_write_span<int>(
      "piece_id", AttrDomain::Point);
  bke::SpanAttributeWriter<float> point_masses = attributes.lookup_or_add_for_write_span<float>(
      "mass", AttrDomain::Point);
  for (const int i : IndexRange(count)) {
    point_piece_ids.span[i] = piece_ids[i];
    point_masses.span[i] = masses[i];
  }
  point_piece_ids.finish();
  point_masses.finish();
  return GeometrySet::from_pointcloud(pointcloud);
}

static bool is_all_zero(const VArray<float3> &values)
{
  for (const int i : values.index_range()) {
    const float3 value = values[i];
    if (value.x != 0.0f || value.y != 0.0f || value.z != 0.0f) {
      return false;
    }
  }
  return true;
}

/**
 * Store the initial velocities on the instance domain, where the solver reads them. A zero field
 * writes nothing, so a `velocity` attribute stored upstream keeps working.
 */
static void apply_initial_velocities(bke::Instances &instances,
                                     const Field<float3> &linear_field,
                                     const Field<float3> &angular_field)
{
  const int count = instances.instances_num();
  if (count == 0) {
    return;
  }
  bke::GeometryFieldContext field_context(instances, AttrDomain::Instance);
  fn::FieldEvaluator evaluator(field_context, count);
  evaluator.add(linear_field);
  evaluator.add(angular_field);
  evaluator.evaluate();
  const VArray<float3> linear = evaluator.get_evaluated<float3>(0);
  const VArray<float3> angular = evaluator.get_evaluated<float3>(1);

  bke::MutableAttributeAccessor attributes = instances.attributes_for_write();
  if (!is_all_zero(linear)) {
    bke::SpanAttributeWriter<float3> velocity =
        attributes.lookup_or_add_for_write_span<float3>("velocity", AttrDomain::Instance);
    for (const int i : IndexRange(count)) {
      velocity.span[i] = linear[i];
    }
    velocity.finish();
  }
  if (!is_all_zero(angular)) {
    bke::SpanAttributeWriter<float3> angular_velocity =
        attributes.lookup_or_add_for_write_span<float3>("angular_velocity", AttrDomain::Instance);
    for (const int i : IndexRange(count)) {
      angular_velocity.span[i] = angular[i];
    }
    angular_velocity.finish();
  }
}

/**
 * Write the `active` flag that the solver reads per body.
 *
 * The default (`true` everywhere) does not write anything, so an `active` attribute that was stored
 * upstream keeps working (the same idea as the initial velocities).
 */
static void apply_active_flags(bke::Instances &instances, const Field<bool> &active_field)
{
  const int count = instances.instances_num();
  if (count == 0) {
    return;
  }
  bke::GeometryFieldContext field_context(instances, AttrDomain::Instance);
  fn::FieldEvaluator evaluator(field_context, count);
  evaluator.add(active_field);
  evaluator.evaluate();
  const VArray<bool> actives = evaluator.get_evaluated<bool>(0);

  bool any_inactive = false;
  for (const int i : IndexRange(count)) {
    if (!actives[i]) {
      any_inactive = true;
      break;
    }
  }
  if (!any_inactive) {
    return;
  }

  bke::MutableAttributeAccessor attributes = instances.attributes_for_write();
  bke::SpanAttributeWriter<bool> active = attributes.lookup_or_add_for_write_span<bool>(
      "active", AttrDomain::Instance);
  for (const int i : IndexRange(count)) {
    active.span[i] = actives[i];
  }
  active.finish();
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet pieces = params.extract_input<GeometrySet>("Pieces"_ustr);

  const float density = std::max(params.extract_input<float>("Density"_ustr), 0.0f);
  const float linear_damping = math::clamp(params.extract_input<float>("Linear Damping"_ustr),
                                           0.0f,
                                           1.0f);
  const float angular_damping = math::clamp(params.extract_input<float>("Angular Damping"_ustr),
                                            0.0f,
                                            1.0f);
  const int group_id = std::max(params.extract_input<int>("Group ID"_ustr), 0);
  const Field<float3> initial_velocity = params.extract_input<Field<float3>>(
      "Initial Velocity"_ustr);
  const Field<float3> initial_angular_velocity = params.extract_input<Field<float3>>(
      "Initial Angular Velocity"_ustr);

  const bool animate = params.extract_input<bool>("Animate"_ustr);
  const Field<bool> active_field = params.extract_input<Field<bool>>("Active"_ustr);

  init_pieces(pieces, density, linear_damping, angular_damping, group_id);
  if (bke::Instances *instances = pieces.get_instances_for_write()) {
    apply_initial_velocities(*instances, initial_velocity, initial_angular_velocity);
    apply_active_flags(*instances, active_field);
    /* The solver reads this flag per body and animates (kinematic) the bodies that have it. */
    bke::MutableAttributeAccessor attributes = instances->attributes_for_write();
    bke::SpanAttributeWriter<bool> animated =
        attributes.lookup_or_add_for_write_span<bool>("animated", AttrDomain::Instance);
    for (const int i : IndexRange(instances->instances_num())) {
      animated.span[i] = animate;
    }
    animated.finish();
  }

  /* A point per piece, so forces can be evaluated per piece (the official force assets work on the
   * point domain) and the solver can map them back by `piece_id`. */
  GeometrySet bodies;
  if (const bke::Instances *instances = pieces.get_instances()) {
    bodies = make_bodies_cloud(*instances);
  }

  /* The per-body cloud rides along in the pieces geometry (a GeometrySet can hold the instances and
   * the point cloud at once), so the user does not need a separate wire for it. */
  if (const GeometryComponent *cloud = bodies.get_component(GeometryComponent::Type::PointCloud)) {
    pieces.add(*cloud);
  }
  params.set_output("Pieces"_ustr, std::move(pieces));
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeInitRigidBody"_ustr);
  ntype.ui_name = "Init Rigid Body";
  ntype.ui_description =
      "Derive mass and damping for the pieces and build the per-body force cloud";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_init_rigid_body_cc
