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
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "BLI_string_utf8.h"

#include "NOD_socket_search_link.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_init_puncliga_data_cc {

enum class PuncligaType : int {
  Cloth = 0,
  Fluid = 1,
  SoftBody = 2,
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
constexpr StringRefNull emission = "puncliga_emission";
constexpr StringRefNull rest_density = "puncliga_rest_density";
constexpr StringRefNull viscosity = "puncliga_viscosity";
constexpr StringRefNull damping = "puncliga_damping";
constexpr StringRefNull vorticity = "puncliga_vorticity";
constexpr StringRefNull surface_tension = "puncliga_surface_tension";
constexpr StringRefNull boundary_friction = "puncliga_boundary_friction";
constexpr StringRefNull negative_pressure_limit = "puncliga_negative_pressure_limit";
constexpr StringRefNull coupling = "puncliga_coupling";
constexpr StringRefNull boundary_particles = "puncliga_boundary_particles";
constexpr StringRefNull stretchiness = "puncliga_stretchiness";
constexpr StringRefNull bendiness = "puncliga_bendiness";
constexpr StringRefNull linear_damping = "puncliga_linear_damping";
constexpr StringRefNull pin = "puncliga_pin";
constexpr StringRefNull rest_position = "puncliga_rest_position";
}  // namespace attribute_names

/** The socket declares its options per type, see `.usage_by_menu()` below. */
static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr);
  b.add_input<decl::Menu>("Type"_ustr)
      .static_items(puncliga_type_items)
      .default_value(MenuValue(int(PuncligaType::Fluid)))
      .description("How this piece of geometry is simulated");
  b.add_input<decl::Menu>("Emission"_ustr)
      .static_items(puncliga_emission_items)
      .default_value(MenuValue(int(PuncligaEmission::None)))
      .description(
          "Whether the geometry of a fluid is only the initial state, or a source that emits new "
          "particles (see the items for the difference between the two modes)");

  /* Every type specific socket is only used by one type, so the others are hidden.
   * `.usage_by_menu` feeds the socket usage inference, which is what drives the visibility of the
   * sockets (and of the panels, an empty panel is not drawn). Note that this is not the same as
   * `SOCK_UNAVAIL`, so `NodeSocket.is_unavailable` in Python stays False;
   * `NodeSocket.is_icon_visible` reports it. */
  const int cloth = int(PuncligaType::Cloth);
  const int fluid = int(PuncligaType::Fluid);
  const int soft = int(PuncligaType::SoftBody);

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
  }
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
        .description("Radius of the points for the point/mesh collider contacts");
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
}

/**
 * Evaluates the `Pin` field and stores the result, together with the rest positions of the pinned
 * points. Both are plain attributes, so the cloth constraint bundle of the asset can read them the
 * same way it reads the rest lengths.
 */
static void write_pin(GeometrySet &geometry, const fn::Field<float> &pin_field, const bool invert)
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
  const Span<float3> positions = mesh->vert_positions();

  bke::SpanAttributeWriter<bool> pinned = attributes.lookup_or_add_for_write_only_span<bool>(
      attribute_names::pin, AttrDomain::Point);
  bke::SpanAttributeWriter<float3> pin_positions =
      attributes.lookup_or_add_for_write_only_span<float3>(attribute_names::rest_position,
                                                           AttrDomain::Point);
  for (const int i : IndexRange(points_num)) {
    const bool above = pin_values[i] > 0.5f;
    pinned.span[i] = invert ? !above : above;
    pin_positions.span[i] = positions[i];
  }
  pinned.finish();
  pin_positions.finish();
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

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  const int type = params.extract_input<int>("Type"_ustr);
  const int emission = params.extract_input<int>("Emission"_ustr);

  const float radius = params.get_input<float>("Collision Radius"_ustr);
  const float friction = params.get_input<float>("Friction"_ustr);
  float mass = params.get_input<float>("Mass"_ustr);
  if (mass <= 0.0f) {
    mass = (type == int(PuncligaType::Fluid)) ?
               params.get_input<float>("Rest Density"_ustr) * radius * radius * radius :
               0.1f;
  }

  write_uniform<int>(geometry, attribute_names::type, bke::AttrDomain::Point, type);
  write_uniform<int>(geometry, attribute_names::emission, bke::AttrDomain::Point, emission);
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
      const bool invert = params.get_input<bool>("Invert Pin"_ustr);
      write_pin(geometry, params.extract_input<Field<float>>("Pin"_ustr), invert);
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
