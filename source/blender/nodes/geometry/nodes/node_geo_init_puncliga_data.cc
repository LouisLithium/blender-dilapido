/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo_nodes
 *
 * Marks a geometry as one of the simulated types of the `Puncliga Dynamics` asset and stores the
 * parameters of that type as attributes on the geometry.
 *
 * The attributes are the interface to the solver: the fluid reads its parameters per particle from
 * them (so several fluids with different densities can be simulated in one world), and the cloth
 * constraint bundles of the asset read the stiffness from them. The type is a plain attribute as
 * well, so the asset can branch the geometry without having to rely on tags (tags are per geometry
 * set, which would not survive a `Join Geometry`).
 */

#include "BKE_attribute_math.hh"
#include "BKE_curves.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "BLI_math_vector.hh"
#include "BLI_string_utf8.h"

#include "NOD_socket_search_link.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_init_puncliga_data_cc {

enum class PuncligaType : int {
  Cloth = 0,
  Fluid = 1,
  SoftBody = 2,
  Hair = 3,
};

/**
 * How the geometry of a fluid is fed into the simulation.
 *
 * The asset keeps the particles of the previous frame in the simulation state, so a plain fluid is
 * seeded once and then simulated. With an emission mode the asset injects the *current* geometry
 * again in every frame instead, which turns the input into an emitter (a nozzle, a moving hose).
 */
enum class PuncligaEmission : int {
  None = 0,
  PerFrame = 1,
  PerSubstep = 2,
};

static const EnumPropertyItem puncliga_type_items[] = {
    {int(PuncligaType::Cloth),
     "CLOTH",
     0,
     N_("Cloth"),
     N_("Simulate the mesh as a cloth: the asset builds the edge length and bend constraints and "
        "keeps the piece hanging where `Pin` is set")},
    {int(PuncligaType::Fluid),
     "FLUID",
     0,
     N_("Fluid"),
     N_("Simulate the points as a position based fluid")},
    {int(PuncligaType::SoftBody),
     "SOFT_BODY",
     0,
     N_("Soft Body"),
     N_("Placeholder: the points only react to gravity and colliders, no internal constraints")},
    {int(PuncligaType::Hair),
     "HAIR",
     0,
     N_("Hair"),
     N_("Simulate the curves as hair: rod stretch/shear and bend/twist constraints, with the root "
        "of every curve pinned by default")},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem puncliga_emission_items[] = {
    {int(PuncligaEmission::None),
     "NONE",
     0,
     N_("None"),
     N_("The geometry is the initial state of the fluid: it is put into the simulation once and "
        "then only simulated")},
    {int(PuncligaEmission::PerFrame),
     "PER_FRAME",
     0,
     N_("Per Frame"),
     N_("The geometry is emitted again in every frame, so it becomes a source of new particles. "
        "The particles inherit the motion of the emitter")},
    {int(PuncligaEmission::PerSubstep),
     "PER_SUBSTEP",
     0,
     N_("Per Substep"),
     N_("Like Per Frame, but the particles are spread over the substeps of the frame, between the "
        "pose of the previous and of the current frame. This fills the gaps of a fast moving "
        "emitter instead of leaving one clump per frame")},
    {0, nullptr, 0, nullptr, nullptr},
};

namespace attribute_names {
constexpr StringRefNull type = "puncliga_type";
constexpr StringRefNull piece_id = "puncliga_piece_id";
constexpr StringRefNull emission = "puncliga_emission";
constexpr StringRefNull rest_density = "puncliga_rest_density";
constexpr StringRefNull viscosity = "puncliga_viscosity";
constexpr StringRefNull damping = "puncliga_damping";
constexpr StringRefNull vorticity = "puncliga_vorticity";
constexpr StringRefNull surface_tension = "puncliga_surface_tension";
constexpr StringRefNull boundary_friction = "puncliga_boundary_friction";
constexpr StringRefNull negative_pressure_limit = "puncliga_negative_pressure_limit";
constexpr StringRefNull coupling = "puncliga_coupling";
constexpr StringRefNull coupling_compliance = "puncliga_coupling_compliance";
constexpr StringRefNull boundary_particles = "puncliga_boundary_particles";
constexpr StringRefNull volume_compliance = "puncliga_volume_compliance";
constexpr StringRefNull edge_compliance = "puncliga_edge_compliance";
constexpr StringRefNull stretchiness = "puncliga_stretchiness";
constexpr StringRefNull bendiness = "puncliga_bendiness";
constexpr StringRefNull root_bendiness = "puncliga_root_bendiness";
constexpr StringRefNull linear_damping = "puncliga_linear_damping";
constexpr StringRefNull angular_damping = "puncliga_angular_damping";
constexpr StringRefNull pin = "puncliga_pin";
constexpr StringRefNull pin_rotation = "puncliga_pin_rotation";
constexpr StringRefNull rest_position = "puncliga_rest_position";
constexpr StringRefNull self_collision = "puncliga_self_collision";
constexpr StringRefNull contact_thickness = "puncliga_contact_thickness";
constexpr StringRefNull hair_self_collision = "puncliga_hair_self_collision";
constexpr StringRefNull hair_contact_thickness = "puncliga_hair_contact_thickness";
constexpr StringRefNull same_strand_collision = "puncliga_same_strand_collision";
}  // namespace attribute_names

/** The socket declares its options per type, see `.usage_by_menu()` below. */
static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr);
  b.add_input<decl::Menu>("Type"_ustr)
      .static_items(puncliga_type_items)
      .default_value(MenuValue(int(PuncligaType::Fluid)))
      .description("How this piece of geometry is simulated");

  /* Every type specific socket is only used by one type, so the others are hidden.
   * `.usage_by_menu` feeds the socket usage inference, which is what drives the visibility of the
   * sockets (and of the panels, an empty panel is not drawn). Note that this is not the same as
   * `SOCK_UNAVAIL`, so `NodeSocket.is_unavailable` in Python stays False;
   * `NodeSocket.is_icon_visible` reports it. */
  const int cloth = int(PuncligaType::Cloth);
  const int fluid = int(PuncligaType::Fluid);
  const int soft = int(PuncligaType::SoftBody);
  const int hair = int(PuncligaType::Hair);

  /* General properties first, then the panels of the individual types. */
  {
    auto &panel = b.add_panel("Shared"_ustr).default_closed(true);
    panel.add_input<decl::Float>("Mass"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description(
            "Mass of one point in kg. 0 computes it (fluid: density * radius^3, soft body: the "
            "volume of the elements around the point, otherwise 0.1 kg)");
    panel.add_input<decl::Float>("Friction"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .description("Contact friction of the points against the colliders");
    panel.add_input<decl::Float>("Collision Radius"_ustr)
        .default_value(0.05f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, {cloth, fluid, soft})
        .description("Radius of the points for the point/mesh collider contacts");
    panel.add_input<decl::Int>("Piece ID"_ustr)
        .default_value(0)
        .min(0)
        .description(
            "Number of this piece, written as the `puncliga_piece_id` attribute. Several pieces of "
            "the same type can be told apart after the simulation with it (the `Split Puncliga "
            "Data` group filters by it); every piece defaults to 0, so it can be ignored when there "
            "is only one");
  }
  {
    auto &panel = b.add_panel("Cloth"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Stretchiness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, cloth)
        .description("How much the cloth may stretch; 0 is stiff, 1 is very soft");
    panel.add_input<decl::Float>("Bendiness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, cloth)
        .description("How much the cloth resists bending; 0 is stiff, 1 is very soft");
    panel.add_input<decl::Float>("Linear Damping"_ustr)
        .default_value(0.1f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, cloth)
        .description("Velocity damping of the cloth in 1/s");
    panel.add_input<decl::Bool>("Self Collision"_ustr)
        .default_value(false)
        .usage_by_menu("Type"_ustr, cloth)
        .description(
            "Push the points of the cloth apart when they come closer than the sum of their "
            "collision radii, so the cloth collides with itself and with the other active solids");
    panel.add_input<decl::Float>("Contact Thickness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, cloth)
        .description("Extra distance kept between two colliding points, on top of their radii");
    panel.add_input<decl::Float>("Pin"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .structure_type(StructureType::Field)
        .usage_by_menu("Type"_ustr, cloth)
        .description("Points where this is above 0.5 are pinned (a vertex group works)");
    panel.add_input<decl::Bool>("Invert Pin"_ustr)
        .default_value(false)
        .usage_by_menu("Type"_ustr, cloth)
        .description("Pin the points where `Pin` is below 0.5 instead");
  }
  {
    auto &panel = b.add_panel("Fluid"_ustr).default_closed(false);
    panel.add_input<decl::Menu>("Emission"_ustr)
        .static_items(puncliga_emission_items)
        .default_value(MenuValue(int(PuncligaEmission::None)))
        .usage_by_menu("Type"_ustr, fluid)
        .description(
            "Whether the geometry of a fluid is only the initial state, or a source that emits new "
            "particles (see the items for the difference between the two modes)");
    panel.add_input<decl::Float>("Rest Density"_ustr)
        .default_value(1000.0f)
        .min(1e-3f)
        .usage_by_menu("Type"_ustr, fluid)
        .description("Target density of the fluid in kg/m^3");
    panel.add_input<decl::Float>("Particle Radius"_ustr)
        .default_value(0.05f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, fluid)
        .description(
            "Kernel support radius is twice this value, the particle spacing should be "
            "one radius");
    panel.add_input<decl::Float>("Viscosity"_ustr)
        .default_value(0.02f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, fluid)
        .description("XSPH velocity smoothing");
    panel.add_input<decl::Float>("Damping"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, fluid)
        .description("Velocity damping in 1/s, damps the bulk motion of the fluid");
    panel.add_input<decl::Float>("Vorticity"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, fluid)
        .description("Vorticity confinement strength");
    panel.add_input<decl::Float>("Surface Tension"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, fluid)
        .description("Cohesion between the particles (0.2 to 2 is usable)");
    panel.add_input<decl::Bool>("Boundary Particles"_ustr)
        .default_value(false)
        .usage_by_menu("Type"_ustr, fluid)
        .description("Sample the colliders as solid boundary particles for this fluid");
    panel.add_input<decl::Float>("Boundary Friction"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, fluid)
        .description("No-slip damping of the velocity next to walls");
    panel.add_input<decl::Float>("Negative Pressure Limit"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, fluid)
        .description("0 solves only compression (stable default)");
    panel.add_input<decl::Float>("Coupling"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, fluid)
        .description(
            "How strongly this fluid pushes the other simulated geometries of the same simulation "
            "(cloth, soft bodies); 0 disables the two way coupling");
    panel.add_input<decl::Float>("Coupling Compliance"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, fluid)
        .description(
            "XPBD compliance of the point/triangle contact with the coupled cloth and soft bodies. "
            "Zero is a hard contact (the previous behaviour); a positive value lets the fluid sink "
            "a little into the surface before it is stopped, which is the same soft-constraint "
            "formulation the CPU uses for colliders. The solver divides it by the substep time "
            "squared, like every other XPBD compliance");
  }
  {
    auto &panel = b.add_panel("Soft Body"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Density"_ustr)
        .default_value(1000.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, soft)
        .description(
            "Density of the material in kg/m^3, used to compute the mass of the vertices from the "
            "volume of the tetrahedra around them");
    panel.add_input<decl::Float>("Volume Compliance"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, soft)
        .description(
            "Softness of the volume constraint of the soft body. Zero keeps the volume of the "
            "tetrahedra, larger values let them compress");
    panel.add_input<decl::Float>("Edge Compliance"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, soft)
        .description(
            "Softness of the distance constraints on the edges of the elements. This is the shear "
            "stiffness: zero is stiff jelly, 1e-3 is very soft");
    panel.add_input<decl::Bool>("Soft Body Self Collision"_ustr)
        .default_value(false)
        .usage_by_menu("Type"_ustr, soft)
        .description(
            "Push the points of the soft body apart when they come closer than the sum of their "
            "collision radii, so the soft body collides with itself and with the other active "
            "solids");
    panel.add_input<decl::Float>("Soft Body Contact Thickness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, soft)
        .description("Extra distance kept between two colliding points, on top of their radii");
  }
  {
    auto &panel = b.add_panel("Hair"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Hair Stretchiness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, hair)
        .description("How much the hair may stretch; 0 is stiff, 1 is very soft");
    panel.add_input<decl::Float>("Hair Bendiness"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, hair)
        .description("How much the hair resists bending; 0 is stiff, 1 is very soft");
    panel.add_input<decl::Float>("Hair Root Bendiness"_ustr)
        .default_value(0.2f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, hair)
        .description("Softness of the rotation pin at the root of every curve");
    panel.add_input<decl::Float>("Hair Linear Damping"_ustr)
        .default_value(3.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, hair)
        .description("Velocity damping of the hair in 1/s");
    panel.add_input<decl::Float>("Hair Angular Damping"_ustr)
        .default_value(3.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, hair)
        .description("Angular velocity damping of the hair in 1/s");
    panel.add_input<decl::Bool>("Hair Self Collision"_ustr)
        .default_value(false)
        .usage_by_menu("Type"_ustr, hair)
        .description(
            "Keep the hair strands apart with segment/segment contacts. The roots of neighbouring "
            "strands are ignored when they already touch in the rest pose");
    panel.add_input<decl::Float>("Hair Contact Thickness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, hair)
        .description("Extra distance kept between two hair segments, on top of their radii");
    panel.add_input<decl::Bool>("Same Strand Collision"_ustr)
        .default_value(false)
        .usage_by_menu("Type"_ustr, hair)
        .description("Also collide segments of the same strand; curly hair may want this on");
    panel.add_input<decl::Float>("Hair Thickness"_ustr)
        .default_value(0.002f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, hair)
        .description(
            "Radius of the hair strands, written as the `radius` attribute (contacts and self "
            "collision use it)");
    panel.add_input<decl::Bool>("Root Pin"_ustr)
        .default_value(true)
        .usage_by_menu("Type"_ustr, hair)
        .description("Pin the first point of every curve, which is the root of a hair strand");
    panel.add_input<decl::Bool>("Root Rotation Pin"_ustr)
        .default_value(true)
        .usage_by_menu("Type"_ustr, hair)
        .description("Pin the rotation of the root points as well");
    panel.add_input<decl::Float>("Hair Pin"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .structure_type(StructureType::Field)
        .usage_by_menu("Type"_ustr, hair)
        .description(
            "Additional points to pin (a vertex group works). The roots are pinned by `Root Pin`");
    panel.add_input<decl::Bool>("Hair Invert Pin"_ustr)
        .default_value(false)
        .usage_by_menu("Type"_ustr, hair)
        .description("Pin the points where `Hair Pin` is below 0.5 instead");
  }

  b.add_output<decl::Geometry>("Geometry"_ustr).propagate_all();
}

/** Writes a uniform value on the given domain of every supported component. */
template<typename T>
static void write_uniform(GeometrySet &geometry,
                          const StringRef name,
                          const bke::AttrDomain domain,
                          const T &value)
{
  auto write = [&](bke::MutableAttributeAccessor attributes) {
    if (attributes.domain_size(domain) == 0) {
      return;
    }
    bke::SpanAttributeWriter<T> writer = attributes.lookup_or_add_for_write_only_span<T>(name,
                                                                                         domain);
    if (!writer) {
      return;
    }
    writer.span.fill(value);
    writer.finish();
  };
  if (Mesh *mesh = geometry.get_mesh_for_write()) {
    write(mesh->attributes_for_write());
  }
  if (PointCloud *pointcloud = geometry.get_pointcloud_for_write()) {
    write(pointcloud->attributes_for_write());
  }
  if (Curves *curves_id = geometry.get_curves_for_write()) {
    write(curves_id->geometry.wrap().attributes_for_write());
  }
}

/**
 * Evaluates the `Pin` field and stores the result, together with the rest positions of the pinned
 * points. Both are plain attributes, so the constraint bundles of the asset can read them the same
 * way they read the rest lengths.
 */
static void write_pin_data(bke::MutableAttributeAccessor attributes,
                           const int points_num,
                           const VArray<float> &pin_values,
                           const Span<float3> positions,
                           const Span<bool> is_root,
                           const bool invert,
                           const bool root_pin,
                           const bool root_rotation_pin,
                           const bool write_rotation_pin)
{
  bke::SpanAttributeWriter<bool> pinned = attributes.lookup_or_add_for_write_only_span<bool>(
      attribute_names::pin, AttrDomain::Point);
  bke::SpanAttributeWriter<float3> pin_positions =
      attributes.lookup_or_add_for_write_only_span<float3>(attribute_names::rest_position,
                                                           AttrDomain::Point);
  /* The plain `rest_position` name as well, so every piece of the simulation has its rest pose under
   * the same name (a soft body writes it too). Post processing needs it: `Point Deform` takes the
   * rest lattice as an input, and that lattice is the simulated geometry with `position` replaced by
   * `rest_position`. */
  bke::SpanAttributeWriter<float3> plain_rest =
      attributes.lookup_or_add_for_write_only_span<float3>("rest_position", AttrDomain::Point);
  bke::SpanAttributeWriter<bool> rotation_pinned;
  if (write_rotation_pin) {
    rotation_pinned = attributes.lookup_or_add_for_write_only_span<bool>(
        attribute_names::pin_rotation, AttrDomain::Point);
  }
  for (const int i : IndexRange(points_num)) {
    const bool above = pin_values[i] > 0.5f;
    const bool field_pinned = invert ? !above : above;
    const bool root = root_pin && is_root[i];
    pinned.span[i] = root || field_pinned;
    if (rotation_pinned) {
      rotation_pinned.span[i] = root_rotation_pin && is_root[i];
    }
    pin_positions.span[i] = positions[i];
    plain_rest.span[i] = positions[i];
  }
  pinned.finish();
  pin_positions.finish();
  plain_rest.finish();
  if (rotation_pinned) {
    rotation_pinned.finish();
  }
}

/** Pins the points of a mesh (cloth). */
static void write_pin_mesh(GeometrySet &geometry,
                           const fn::Field<float> &pin_field,
                           const bool invert)
{
  Mesh *mesh = geometry.get_mesh_for_write();
  if (mesh == nullptr) {
    return;
  }
  const int points_num = mesh->verts_num;
  if (points_num == 0) {
    return;
  }
  bke::MutableAttributeAccessor attributes = mesh->attributes_for_write();
  const bke::GeometryFieldContext field_context{*mesh, AttrDomain::Point};
  fn::FieldEvaluator evaluator{field_context, points_num};
  evaluator.add(pin_field);
  evaluator.evaluate();
  const VArray<float> pin_values = evaluator.get_evaluated<float>(0);
  const Array<bool> is_root(points_num, false);
  write_pin_data(attributes,
                 points_num,
                 pin_values,
                 mesh->vert_positions(),
                 is_root,
                 invert,
                 false,
                 false,
                 false);
}

/**
 * Pins the roots of the hair strands and the extra points selected by the `Hair Pin` field. The
 * roots are the first point of every curve, which is what keeps a strand attached to the scalp.
 */
static void write_pin_curves(GeometrySet &geometry,
                             const fn::Field<float> &pin_field,
                             const bool invert,
                             const bool root_pin,
                             const bool root_rotation_pin)
{
  Curves *curves_id = geometry.get_curves_for_write();
  if (curves_id == nullptr) {
    return;
  }
  bke::CurvesGeometry &curves = curves_id->geometry.wrap();
  const int points_num = curves.points_num();
  if (points_num == 0) {
    return;
  }
  bke::MutableAttributeAccessor attributes = curves.attributes_for_write();
  const bke::CurvesFieldContext field_context{*curves_id, AttrDomain::Point};
  fn::FieldEvaluator evaluator{field_context, points_num};
  evaluator.add(pin_field);
  evaluator.evaluate();
  const VArray<float> pin_values = evaluator.get_evaluated<float>(0);

  Array<bool> is_root(points_num, false);
  const OffsetIndices<int> points_by_curve = curves.points_by_curve();
  for (const int curve_i : points_by_curve.index_range()) {
    const IndexRange points = points_by_curve[curve_i];
    if (!points.is_empty()) {
      is_root[points[0]] = true;
    }
  }
  write_pin_data(attributes,
                 points_num,
                 pin_values,
                 curves.positions(),
                 is_root,
                 invert,
                 root_pin,
                 root_rotation_pin,
                 true);
}

/**
 * Mass of the vertices of a tetrahedral mesh: the density times the volume of the elements that
 * touch the vertex, i.e. a quarter of each of them. This is what makes a soft body deform like a
 * material instead of like a mesh with uniform point masses (a coarse element would otherwise have
 * the same inertia as a small one).
 *
 * \return False when the geometry is not a tetrahedral mesh (four cornered faces + `rest_volume`).
 */
static bool write_soft_body_mass(GeometrySet &geometry, const float density)
{
  Mesh *mesh = geometry.get_mesh_for_write();
  if (mesh == nullptr) {
    return false;
  }
  const VArray<float> rest_volumes = *mesh->attributes().lookup<float>("rest_volume",
                                                                     bke::AttrDomain::Face);
  if (!rest_volumes) {
    return false;
  }
  const blender::OffsetIndices<int> faces = mesh->faces();
  if (faces.is_empty() || mesh->verts_num == 0) {
    return false;
  }

  Array<float> vertex_volumes(mesh->verts_num, 0.0f);
  const Span<int> corner_verts = mesh->corner_verts();
  for (const int face_i : IndexRange(faces.size())) {
    const IndexRange face = faces[face_i];
    const float share = rest_volumes[face_i] / float(face.size());
    for (const int corner : face) {
      vertex_volumes[corner_verts[corner]] += share;
    }
  }

  bke::SpanAttributeWriter<float> mass = mesh->attributes_for_write().lookup_or_add_for_write_span<
      float>("mass", bke::AttrDomain::Point);
  for (const int i : IndexRange(mesh->verts_num)) {
    mass.span[i] = std::max(density * vertex_volumes[i], 1e-6f);
  }
  mass.finish();
  return true;
}

/**
 * Moment of inertia of the hair points. A strand is a thin rod, so every point gets the inertia of
 * the segment that touches it: `(m_i + m_neighbor) * d^2 / 12`. Without this the solver falls back
 * to `(1, 1, 1)` and the angular motion of the hair is not physical.
 */
static void write_hair_inertia(GeometrySet &geometry, const float mass)
{
  Curves *curves_id = geometry.get_curves_for_write();
  if (curves_id == nullptr) {
    return;
  }
  bke::CurvesGeometry &curves = curves_id->geometry.wrap();
  const int points_num = curves.points_num();
  if (points_num == 0) {
    return;
  }
  bke::MutableAttributeAccessor attributes = curves.attributes_for_write();
  bke::SpanAttributeWriter<float3> inertia =
      attributes.lookup_or_add_for_write_span<float3>("moment_of_inertia", bke::AttrDomain::Point);
  const Span<float3> positions = curves.positions();
  const OffsetIndices<int> points_by_curve = curves.points_by_curve();
  const VArray<bool> cyclic = curves.cyclic();
  for (const int curve_i : points_by_curve.index_range()) {
    const IndexRange points = points_by_curve[curve_i];
    const int size = points.size();
    for (const int local_i : IndexRange(size)) {
      const int point_i = points[local_i];
      int neighbor_i = -1;
      if (local_i + 1 < size) {
        neighbor_i = points[local_i + 1];
      }
      else if (cyclic[curve_i] && size > 1) {
        neighbor_i = points[0];
      }
      else if (local_i > 0) {
        neighbor_i = points[local_i - 1];
      }
      const float length = neighbor_i >= 0 ? math::distance(positions[point_i],
                                                            positions[neighbor_i]) :
                                             0.0f;
      /* Two point masses at the ends of a thin rod: (m + m) * d^2 / 12. */
      const float value = std::max((mass + mass) * length * length / 12.0f, 1e-9f);
      inertia.span[point_i] = float3(value);
    }
  }
  inertia.finish();
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  const int type = params.extract_input<int>("Type"_ustr);
  const int emission = params.extract_input<int>("Emission"_ustr);

  const float radius = (type == int(PuncligaType::Hair)) ?
                           params.get_input<float>("Hair Thickness"_ustr) :
                           params.get_input<float>("Collision Radius"_ustr);
  const float friction = params.get_input<float>("Friction"_ustr);
  float mass = params.get_input<float>("Mass"_ustr);
  if (mass <= 0.0f) {
    if (type == int(PuncligaType::Fluid)) {
      mass = params.get_input<float>("Rest Density"_ustr) * radius * radius * radius;
    }
    else if (type == int(PuncligaType::Hair)) {
      mass = 0.01f;
    }
    else {
      mass = 0.1f;
    }
  }

  write_uniform<int>(geometry, attribute_names::type, bke::AttrDomain::Point, type);
  write_uniform<int>(geometry, attribute_names::emission, bke::AttrDomain::Point, emission);
  write_uniform<int>(geometry,
                     attribute_names::piece_id,
                     bke::AttrDomain::Point,
                     params.get_input<int>("Piece ID"_ustr));
  write_uniform<float>(geometry, "radius", bke::AttrDomain::Point, radius);
  write_uniform<float>(geometry, "mass", bke::AttrDomain::Point, mass);
  write_uniform<float>(geometry, "static_friction", bke::AttrDomain::Point, friction);
  write_uniform<float>(geometry, "dynamic_friction", bke::AttrDomain::Point, friction);

  switch (PuncligaType(type)) {
    case PuncligaType::Cloth: {
      const float stretchiness = params.get_input<float>("Stretchiness"_ustr);
      const float bendiness = params.get_input<float>("Bendiness"_ustr);
      const float damping = params.get_input<float>("Linear Damping"_ustr);
      write_uniform<float>(
          geometry, attribute_names::stretchiness, bke::AttrDomain::Point, stretchiness);
      write_uniform<float>(
          geometry, attribute_names::stretchiness, bke::AttrDomain::Edge, stretchiness);
      write_uniform<float>(
          geometry, attribute_names::bendiness, bke::AttrDomain::Point, bendiness);
      write_uniform<float>(
          geometry, attribute_names::linear_damping, bke::AttrDomain::Point, damping);
      write_uniform<bool>(geometry,
                          attribute_names::self_collision,
                          bke::AttrDomain::Point,
                          params.get_input<bool>("Self Collision"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::contact_thickness,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Contact Thickness"_ustr));
      const bool invert = params.get_input<bool>("Invert Pin"_ustr);
      write_pin_mesh(geometry, params.extract_input<Field<float>>("Pin"_ustr), invert);
      break;
    }
    case PuncligaType::Hair: {
      if (geometry.get_curves() == nullptr) {
        params.error_message_add(
            NodeWarningType::Warning,
            TIP_("Hair needs a curve geometry (hair curves or a curve object); mesh hair cards "
                 "should be simulated as a cloth or a soft body"));
        break;
      }
      write_uniform<float>(geometry,
                           attribute_names::stretchiness,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Hair Stretchiness"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::bendiness,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Hair Bendiness"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::root_bendiness,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Hair Root Bendiness"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::linear_damping,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Hair Linear Damping"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::angular_damping,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Hair Angular Damping"_ustr));
      write_uniform<bool>(geometry,
                          attribute_names::hair_self_collision,
                          bke::AttrDomain::Point,
                          params.get_input<bool>("Hair Self Collision"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::hair_contact_thickness,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Hair Contact Thickness"_ustr));
      write_uniform<bool>(geometry,
                          attribute_names::same_strand_collision,
                          bke::AttrDomain::Point,
                          params.get_input<bool>("Same Strand Collision"_ustr));
      write_hair_inertia(geometry, mass);
      write_pin_curves(geometry,
                       params.extract_input<Field<float>>("Hair Pin"_ustr),
                       params.get_input<bool>("Hair Invert Pin"_ustr),
                       params.get_input<bool>("Root Pin"_ustr),
                       params.get_input<bool>("Root Rotation Pin"_ustr));
      break;
    }
    case PuncligaType::Fluid: {
      write_uniform<float>(geometry,
                           attribute_names::rest_density,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Rest Density"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::viscosity,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Viscosity"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::damping,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Damping"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::vorticity,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Vorticity"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::surface_tension,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Surface Tension"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::boundary_friction,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Boundary Friction"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::negative_pressure_limit,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Negative Pressure Limit"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::coupling,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Coupling"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::coupling_compliance,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Coupling Compliance"_ustr));
      write_uniform<bool>(geometry,
                          attribute_names::boundary_particles,
                          bke::AttrDomain::Point,
                          params.get_input<bool>("Boundary Particles"_ustr));
      break;
    }
    case PuncligaType::SoftBody: {
      /* The solver builds the distance constraints of the elements from their rest lengths, so the
       * rest pose has to be stored before the geometry starts to move. */
      if (Mesh *mesh = geometry.get_mesh_for_write()) {
        if (!mesh->attributes().contains("rest_position")) {
          bke::SpanAttributeWriter<float3> writer =
              mesh->attributes_for_write().lookup_or_add_for_write_only_span<float3>(
                  "rest_position", bke::AttrDomain::Point);
          writer.span.copy_from(mesh->vert_positions());
          writer.finish();
        }
      }
      write_uniform<float>(geometry,
                           attribute_names::volume_compliance,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Volume Compliance"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::edge_compliance,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Edge Compliance"_ustr));
      write_uniform<bool>(geometry,
                          attribute_names::self_collision,
                          bke::AttrDomain::Point,
                          params.get_input<bool>("Soft Body Self Collision"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::contact_thickness,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Soft Body Contact Thickness"_ustr));
      /* The mass comes from the volume of the tetrahedra, not from a uniform point mass. */
      const float mass_input = params.get_input<float>("Mass"_ustr);
      if (mass_input <= 0.0f) {
        if (!write_soft_body_mass(geometry, params.get_input<float>("Density"_ustr))) {
          params.error_message_add(
              NodeWarningType::Warning,
              TIP_("A soft body needs a tetrahedral mesh (four cornered faces with a "
                   "`rest_volume` attribute, e.g. from the `Tetrahedralize` node); the mass of "
                   "the points is left uniform"));
        }
      }
      break;
    }
  }

  params.set_output("Geometry"_ustr, std::move(geometry));
}

static void node_label(const bNodeTree * /*ntree*/,
                       const bNode * /*node*/,
                       char *label,
                       const int label_maxncpy)
{
  BLI_strncpy_utf8(
      label, CTX_IFACE_(BLT_I18NCONTEXT_ID_NODETREE, "Init Puncliga Data"), label_maxncpy);
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeInitPuncligaData"_ustr);
  ntype.ui_name = "Init Puncliga Data";
  ntype.ui_description =
      "Mark geometry as cloth, fluid or soft body and store the parameters that `Puncliga "
      "Dynamics` needs";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  ntype.labelfunc = node_label;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_init_puncliga_data_cc
