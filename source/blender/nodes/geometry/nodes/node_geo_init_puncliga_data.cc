/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo_nodes
 *
 * Marks a geometry as one of the simulated types of the `Puncliga Dynamics` asset (cloth, fluid,
 * soft body, hair or a rigid piece) and stores the parameters of that type as attributes on the
 * geometry.
 *
 * The attributes are the interface to the solver: the fluid reads its parameters per particle from
 * them (so several fluids with different densities can be simulated in one world), and the cloth
 * constraint bundles of the asset read the stiffness from them. The type is a plain attribute as
 * well, so the asset can branch the geometry without having to rely on tags (tags are per geometry
 * set, which would not survive a `Join Geometry`).
 */

#include "BKE_attribute_math.hh"
#include "BKE_curves.hh"
#include "BKE_geometry_fields.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include <limits>

#include "BLI_hash.h"
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
  Rigid = 4,
  Grain = 5,
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

/** Which edges of a cloth are allowed to tear when they are stretched too far. */
enum class PuncligaTearMode : int {
  All = 0,
  Custom = 1,
  Voronoi = 2,
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
    {int(PuncligaType::Rigid),
     "RIGID",
     0,
     N_("Rigid"),
     N_("Simulate the points as quasi rigid pieces with a shape matching constraint. The points "
        "are usually a sphere packing of a mesh (`Distribute Points` in volume mode), and every "
        "piece is kept in its rest shape while the points still collide with each other")},
    {int(PuncligaType::Grain),
     "GRAIN",
     0,
     N_("Grain"),
     N_("Simulate the points as granular material: frictional particle contacts, optionally with "
        "temporary clumps that break apart and form new blocks after a cooldown")},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem puncliga_tear_mode_items[] = {
    {int(PuncligaTearMode::All),
     "ALL",
     0,
     N_("All"),
     N_("Any edge of the cloth can tear")},
    {int(PuncligaTearMode::Custom),
     "CUSTOM",
     0,
     N_("Custom"),
     N_("Only the edges selected by `Tear Edge Group` can tear")},
    {int(PuncligaTearMode::Voronoi),
     "VORONOI",
     0,
     N_("Voronoi"),
     N_("Only tear along the borders of a Voronoi pattern, so the cloth splits into islands")},
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
constexpr StringRefNull emission_spacing = "puncliga_emission_spacing";
constexpr StringRefNull emission_velocity_jitter = "puncliga_emission_velocity_jitter";
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
constexpr StringRefNull guide_tether = "puncliga_guide_tether";
constexpr StringRefNull tether_compliance = "puncliga_tether_compliance";
constexpr StringRefNull shape_matching = "puncliga_shape_matching";
constexpr StringRefNull shape_group = "puncliga_shape_group";
constexpr StringRefNull shape_stiffness = "puncliga_shape_stiffness";
constexpr StringRefNull tear_threshold = "puncliga_tear_threshold";
constexpr StringRefNull grain_softness = "puncliga_grain_softness";
constexpr StringRefNull grain_restitution = "puncliga_grain_restitution";
constexpr StringRefNull grain_clump = "puncliga_grain_clump";
constexpr StringRefNull grain_clump_radius = "puncliga_grain_clump_radius";
constexpr StringRefNull grain_clump_strength = "puncliga_grain_clump_strength";
constexpr StringRefNull grain_break_strain = "puncliga_grain_break_strain";
constexpr StringRefNull grain_break_fraction = "puncliga_grain_break_fraction";
constexpr StringRefNull grain_reclump_delay = "puncliga_grain_reclump_delay";
constexpr StringRefNull grain_clump_min_size = "puncliga_grain_clump_min_size";
constexpr StringRefNull grain_clump_max_size = "puncliga_grain_clump_max_size";
constexpr StringRefNull grain_clump_speed = "puncliga_grain_clump_speed";
constexpr StringRefNull grain_clump_id = "puncliga_grain_clump_id";
constexpr StringRefNull grain_clump_rest = "puncliga_grain_clump_rest";
constexpr StringRefNull grain_clump_cooldown = "puncliga_grain_clump_cooldown";
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
  const int rigid = int(PuncligaType::Rigid);
  const int grain = int(PuncligaType::Grain);

  /* General properties first, then the panels of the individual types. */
  {
    auto &panel = b.add_panel("Shared"_ustr).default_closed(true);
    panel.add_input<decl::Float>("Mass"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description(
            "Mass of one point in kg. 0 computes it (fluid: density * radius^3, grain: grain "
            "density * sphere volume, soft body: the volume of the elements around the point, "
            "otherwise 0.1 kg)");
    panel.add_input<decl::Float>("Friction"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .description("Contact friction of the points against the colliders and against each other");
    panel.add_input<decl::Float>("Collision Radius"_ustr)
        .default_value(0.05f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, {cloth, fluid, soft, rigid, grain})
        .description("Radius of the points for the point/mesh collider contacts");
    panel.add_input<decl::Int>("Piece ID"_ustr)
        .default_value(0)
        .min(0)
        .description(
            "Number of this piece, written as the `puncliga_piece_id` attribute. Several pieces of "
            "the same type can be told apart after the simulation with it (the `Split Puncliga "
            "Data` group filters by it); every piece defaults to 0, so it can be ignored when there "
            "is only one");
    panel.add_input<decl::Menu>("Emission"_ustr)
        .static_items(puncliga_emission_items)
        .default_value(MenuValue(int(PuncligaEmission::None)))
        .usage_by_menu("Type"_ustr, {fluid, grain})
        .description(
            "Whether the geometry is only the initial state, or a source that emits new particles "
            "(see the items for the difference between the two modes)");
    panel.add_input<decl::Float>("Emission Spacing"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, {fluid, grain})
        .description(
            "Do not emit a particle when the nearest existing particle is closer than this "
            "distance, and use it to decide how many copies a fast emitter needs per frame "
            "(0 disables the check)");
    panel.add_input<decl::Float>("Emission Velocity Jitter"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, {fluid, grain})
        .description(
            "Random per particle speed variation (fraction of the emitted velocity). A small value "
            "smears the discrete batches of `Per Frame` into a continuous stream");
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
    panel.add_input<decl::Bool>("Tear"_ustr)
        .default_value(false)
        .usage_by_menu("Type"_ustr, cloth)
        .description(
            "Split the cloth along edges that are stretched beyond `Tear Threshold`. A torn edge "
            "is no longer a connection between the two faces next to it, so the cloth can rip");
    panel.add_input<decl::Menu>("Tear Mode"_ustr)
        .static_items(puncliga_tear_mode_items)
        .default_value(MenuValue(int(PuncligaTearMode::All)))
        .usage_by_menu("Type"_ustr, cloth)
        .description("Which edges are allowed to tear");
    panel.add_input<decl::Float>("Tear Threshold"_ustr)
        .default_value(0.5f)
        .min(1e-4f)
        .usage_by_menu("Type"_ustr, cloth)
        .description(
            "Relative strain (current length / rest length - 1) above which an edge tears; a "
            "stiffer cloth needs a smaller value than a very soft one");
    panel.add_input<decl::Float>("Tear Edge Group"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .structure_type(StructureType::Field)
        .usage_by_menu("Type"_ustr, cloth)
        .description(
            "Only tear the edges where this field is above 0.5 (`Tear Mode = Custom`); a boolean "
            "edge attribute or an edge group works");
    panel.add_input<decl::Float>("Voronoi Scale"_ustr)
        .default_value(0.5f)
        .min(1e-4f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, cloth)
        .description(
            "Size of the Voronoi cells that form the untearable islands (`Tear Mode = Voronoi`)");
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
    panel.add_input<decl::Bool>("Guide Tether"_ustr)
        .default_value(false)
        .usage_by_menu("Type"_ustr, hair)
        .description(
            "Attach every child curve to the guide named by its `guide_curve_index`. Enable this "
            "only on the child geometry: curves with the tether switched off are the guides (give "
            "the guides their own `Init Puncliga Data` node). The attachment is compliant and two "
            "way: the guide drives the child, and a collision that moves the child also moves the "
            "guide");
    panel.add_input<decl::Float>("Tether Compliance"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, hair)
        .description(
            "Softness of the guide tether (XPBD compliance in m/N). Zero keeps the rest offset "
            "rigidly; a small value (1e-5 to 1e-3) lets collisions and motion stretch it");
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
  {
    auto &panel = b.add_panel("Rigid"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Rigid Density"_ustr)
        .default_value(1000.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, rigid)
        .description(
            "Density of the material in kg/m^3, used to compute the mass of a point from the volume "
            "of its sphere (4/3 pi r^3)");
    panel.add_input<decl::Float>("Rigid Linear Damping"_ustr)
        .default_value(0.1f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, rigid)
        .description("Velocity damping of the rigid pieces in 1/s");
    panel.add_input<decl::Bool>("Rigid Self Collision"_ustr)
        .default_value(true)
        .usage_by_menu("Type"_ustr, rigid)
        .description(
            "Push the points of the rigid pieces apart when they come closer than the sum of their "
            "collision radii, so pieces collide with each other");
    panel.add_input<decl::Float>("Rigid Contact Thickness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, rigid)
        .description("Extra distance kept between two colliding points, on top of their radii");
  }
  {
    auto &panel = b.add_panel("Shape Matching"_ustr).default_closed(false);
    panel.add_input<decl::Bool>("Shape Matching"_ustr)
        .default_value(true)
        .usage_by_menu("Type"_ustr, rigid)
        .description(
            "Keep every piece in its rest shape with a least squares shape matching constraint. "
            "Every piece is defined by `Shape Group` (by default the `Piece ID`)");
    panel.add_input<decl::Int>("Shape Group"_ustr)
        .default_value(-1)
        .min(-1)
        .structure_type(StructureType::Field)
        .usage_by_menu("Type"_ustr, rigid)
        .description(
            "Group id of every point. Points with the same id are one rigid piece. -1 keeps the "
            "`puncliga_shape_group` attribute of the input (a sphere packing can write it), or "
            "uses the `Piece ID` of this node when there is none");
    panel.add_input<decl::Float>("Shape Stiffness"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .max(1.0f)
        .structure_type(StructureType::Field)
        .usage_by_menu("Type"_ustr, rigid)
        .description(
            "Projection strength per constraint iteration: 1 keeps a piece rigid in one solve, "
            "smaller values let it deform. This is a position based projection, so the effective "
            "stiffness also depends on the constraint iterations");
  }
  {
    auto &panel = b.add_panel("Grain"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Grain Density"_ustr)
        .default_value(1600.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, grain)
        .description(
            "Density of the granular material in kg/m^3, used to compute the mass of a point from "
            "its sphere volume when `Mass` is zero");
    panel.add_input<decl::Float>("Grain Linear Damping"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, grain)
        .description("Velocity damping of the grains in 1/s");
    panel.add_input<decl::Float>("Grain Contact Softness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, grain)
        .description("Softness of the grain contacts; 0 is a hard contact, 1 is very soft");
    panel.add_input<decl::Float>("Grain Contact Thickness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, grain)
        .description("Extra distance kept between two grains, on top of their radii");
    panel.add_input<decl::Float>("Grain Restitution"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, grain)
        .description("Bounciness of the grain contacts");
    panel.add_input<decl::Bool>("Clump"_ustr)
        .default_value(false)
        .usage_by_menu("Type"_ustr, grain)
        .description(
            "Let slow grains that are close form temporary blocks. A block is held in its rest "
            "shape until it is deformed too much, then it tears and the pieces can clump again "
            "after a cooldown");
    panel.add_input<decl::Float>("Clump Radius"_ustr)
        .default_value(0.05f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .usage_by_menu("Type"_ustr, grain)
        .description("Spatial cell size used to group grains into one block");
    panel.add_input<decl::Float>("Clump Strength"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, grain)
        .description("How strongly a block keeps its rest shape");
    panel.add_input<decl::Float>("Break Strain"_ustr)
        .default_value(0.2f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, grain)
        .description(
            "Relative deformation of a block above which it starts to break; 0 disables breaking");
    panel.add_input<decl::Float>("Break Fraction"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, grain)
        .description(
            "Fraction of a block's grains with the largest deformation that become free when the "
            "block breaks");
    panel.add_input<decl::Float>("Re-clump Delay"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .subtype(PROP_TIME_ABSOLUTE)
        .usage_by_menu("Type"_ustr, grain)
        .description("Seconds that a broken grain stays free before it can join a new block");
    panel.add_input<decl::Int>("Clump Min Size"_ustr)
        .default_value(4)
        .min(1)
        .usage_by_menu("Type"_ustr, grain)
        .description("Smallest number of grains that form a block");
    panel.add_input<decl::Int>("Clump Max Size"_ustr)
        .default_value(64)
        .min(1)
        .usage_by_menu("Type"_ustr, grain)
        .description("Largest number of grains in one block");
    panel.add_input<decl::Float>("Clump Speed"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .subtype(PROP_VELOCITY)
        .usage_by_menu("Type"_ustr, grain)
        .description("Grains faster than this do not join a new block");
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

/** Writes the initial clump state of a granular material: no clump, rest at the current pose. */
static void write_grain_clump_state(GeometrySet &geometry)
{
  const auto write = [](bke::MutableAttributeAccessor attributes,
                        const int points_num,
                        const Span<float3> positions) {
    if (points_num == 0) {
      return;
    }
    bke::SpanAttributeWriter<int> id_writer =
        attributes.lookup_or_add_for_write_only_span<int>(attribute_names::grain_clump_id,
                                                          bke::AttrDomain::Point);
    bke::SpanAttributeWriter<float3> rest_writer =
        attributes.lookup_or_add_for_write_only_span<float3>(attribute_names::grain_clump_rest,
                                                             bke::AttrDomain::Point);
    bke::SpanAttributeWriter<float> cooldown_writer =
        attributes.lookup_or_add_for_write_only_span<float>(attribute_names::grain_clump_cooldown,
                                                            bke::AttrDomain::Point);
    if (!id_writer || !rest_writer || !cooldown_writer) {
      return;
    }
    id_writer.span.fill(-1);
    cooldown_writer.span.fill(0.0f);
    if (positions.size() == points_num) {
      rest_writer.span.copy_from(positions);
    }
    else {
      rest_writer.span.fill(float3(0.0f));
    }
    id_writer.finish();
    rest_writer.finish();
    cooldown_writer.finish();
  };
  if (Mesh *mesh = geometry.get_mesh_for_write()) {
    write(mesh->attributes_for_write(), mesh->verts_num, mesh->vert_positions());
  }
  if (PointCloud *pointcloud = geometry.get_pointcloud_for_write()) {
    write(pointcloud->attributes_for_write(), pointcloud->totpoint, pointcloud->positions());
  }
  if (Curves *curves_id = geometry.get_curves_for_write()) {
    bke::CurvesGeometry &curves = curves_id->geometry.wrap();
    write(curves.attributes_for_write(), curves.points_num(), curves.positions());
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

/** Stores the rest pose of every supported component under both rest attribute names. */
static void write_rest_positions(GeometrySet &geometry)
{
  const auto write = [](bke::MutableAttributeAccessor attributes, const Span<float3> positions) {
    if (positions.is_empty()) {
      return;
    }
    for (const StringRef name : {StringRef("rest_position"),
                                 StringRef("puncliga_rest_position")})
    {
      bke::SpanAttributeWriter<float3> writer =
          attributes.lookup_or_add_for_write_only_span<float3>(name, bke::AttrDomain::Point);
      if (!writer) {
        continue;
      }
      writer.span.copy_from(positions);
      writer.finish();
    }
  };
  if (Mesh *mesh = geometry.get_mesh_for_write()) {
    write(mesh->attributes_for_write(), mesh->vert_positions());
  }
  if (PointCloud *pointcloud = geometry.get_pointcloud_for_write()) {
    write(pointcloud->attributes_for_write(), pointcloud->positions());
  }
  if (Curves *curves_id = geometry.get_curves_for_write()) {
    bke::CurvesGeometry &curves = curves_id->geometry.wrap();
    write(curves.attributes_for_write(), curves.positions());
  }
}

/** Whether the geometry already carries a per point `radius` attribute (a sphere packing does). */
static bool has_point_radius(const GeometrySet &geometry)
{
  const auto has = [](const bke::AttributeAccessor attributes) {
    const VArray<float> radius = *attributes.lookup<float>("radius", bke::AttrDomain::Point);
    return bool(radius);
  };
  if (const Mesh *mesh = geometry.get_mesh()) {
    return has(mesh->attributes());
  }
  if (const PointCloud *pointcloud = geometry.get_pointcloud()) {
    return has(pointcloud->attributes());
  }
  if (const Curves *curves_id = geometry.get_curves()) {
    return has(curves_id->geometry.wrap().attributes());
  }
  return false;
}

/** Per point mass of a rigid piece: one point represents one sphere of the packing. */
static void write_rigid_mass(GeometrySet &geometry,
                             const float density,
                             const float fallback_radius)
{
  const auto write = [&](bke::MutableAttributeAccessor attributes, const int points_num,
                         const VArray<float> radii, const float fallback_radius) {
    bke::SpanAttributeWriter<float> mass = attributes.lookup_or_add_for_write_span<float>(
        "mass", bke::AttrDomain::Point);
    if (!mass) {
      return;
    }
    for (const int i : IndexRange(points_num)) {
      const float radius = radii ? radii[i] : fallback_radius;
      mass.span[i] = density * (4.0f / 3.0f) * 3.14159265358979323846f * radius * radius * radius;
    }
    mass.finish();
  };
  if (Mesh *mesh = geometry.get_mesh_for_write()) {
    const VArray<float> radii = *mesh->attributes().lookup<float>("radius", bke::AttrDomain::Point);
    write(mesh->attributes_for_write(), mesh->verts_num, radii, fallback_radius);
  }
  if (PointCloud *pointcloud = geometry.get_pointcloud_for_write()) {
    const VArray<float> radii =
        *pointcloud->attributes().lookup<float>("radius", bke::AttrDomain::Point);
    write(pointcloud->attributes_for_write(), pointcloud->totpoint, radii, fallback_radius);
  }
  if (Curves *curves_id = geometry.get_curves_for_write()) {
    bke::CurvesGeometry &curves = curves_id->geometry.wrap();
    const VArray<float> radii = *curves.attributes().lookup<float>("radius",
                                                                  bke::AttrDomain::Point);
    write(curves.attributes_for_write(), curves.points_num(), radii, fallback_radius);
  }
}

static void write_shape_matching_component(bke::MutableAttributeAccessor attributes,
                                           const fn::FieldContext &field_context,
                                           const int points_num,
                                           const bool enabled,
                                           const fn::Field<int> &group_field,
                                           const fn::Field<float> &stiffness_field,
                                           const int piece_id)
{
  if (points_num == 0) {
    return;
  }
  fn::FieldEvaluator evaluator{field_context, points_num};
  evaluator.add(group_field);
  evaluator.add(stiffness_field);
  evaluator.evaluate();
  const VArray<int> groups = evaluator.get_evaluated<int>(0);
  const VArray<float> stiffnesses = evaluator.get_evaluated<float>(1);
  /* A sphere packing can write the group itself; -1 keeps it instead of falling back to the
   * `Piece ID`. */
  const VArray<int> existing_groups = *attributes.lookup<int>(attribute_names::shape_group,
                                                              bke::AttrDomain::Point);

  bke::SpanAttributeWriter<bool> selection_writer =
      attributes.lookup_or_add_for_write_only_span<bool>(attribute_names::shape_matching,
                                                         bke::AttrDomain::Point);
  bke::SpanAttributeWriter<int> group_writer =
      attributes.lookup_or_add_for_write_only_span<int>(attribute_names::shape_group,
                                                        bke::AttrDomain::Point);
  bke::SpanAttributeWriter<float> stiffness_writer =
      attributes.lookup_or_add_for_write_only_span<float>(attribute_names::shape_stiffness,
                                                          bke::AttrDomain::Point);
  if (!selection_writer || !group_writer || !stiffness_writer) {
    return;
  }
  for (const int i : IndexRange(points_num)) {
    selection_writer.span[i] = enabled;
    const int existing_group = existing_groups ? existing_groups[i] : -1;
    group_writer.span[i] = groups[i] >= 0 ? groups[i] :
                                          (existing_group >= 0 ? existing_group : piece_id);
    stiffness_writer.span[i] = math::clamp(stiffnesses[i], 0.0f, 1.0f);
  }
  selection_writer.finish();
  group_writer.finish();
  stiffness_writer.finish();
}

static void write_shape_matching(GeometrySet &geometry,
                                 const bool enabled,
                                 const fn::Field<int> &group_field,
                                 const fn::Field<float> &stiffness_field,
                                 const int piece_id)
{
  if (Mesh *mesh = geometry.get_mesh_for_write()) {
    const bke::GeometryFieldContext field_context{*mesh, bke::AttrDomain::Point};
    write_shape_matching_component(mesh->attributes_for_write(),
                                   field_context,
                                   mesh->verts_num,
                                   enabled,
                                   group_field,
                                   stiffness_field,
                                   piece_id);
  }
  if (PointCloud *pointcloud = geometry.get_pointcloud_for_write()) {
    const bke::GeometryFieldContext field_context{*pointcloud};
    write_shape_matching_component(pointcloud->attributes_for_write(),
                                   field_context,
                                   pointcloud->totpoint,
                                   enabled,
                                   group_field,
                                   stiffness_field,
                                   piece_id);
  }
  if (Curves *curves_id = geometry.get_curves_for_write()) {
    bke::CurvesGeometry &curves = curves_id->geometry.wrap();
    const bke::CurvesFieldContext field_context{curves, bke::AttrDomain::Point};
    write_shape_matching_component(curves.attributes_for_write(),
                                   field_context,
                                   curves.points_num(),
                                   enabled,
                                   group_field,
                                   stiffness_field,
                                   piece_id);
  }
}

/** Deterministic pseudo random value in 0..1 for one Voronoi cell and one component. */
static float voronoi_cell_random(const int3 &cell, const uint32_t salt)
{
  const uint32_t hash = BLI_hash_int_3d(
      uint32_t(cell.x) + salt * 73856093u,
      uint32_t(cell.y) + salt * 19349663u,
      uint32_t(cell.z) + salt * 83492791u);
  return float(hash) / float(0xFFFFFFFFu);
}

/** Cell id of the jittered grid Voronoi site closest to the given rest position. */
static int voronoi_cell_id(const float3 &position, const float scale)
{
  const float safe_scale = std::max(scale, 1e-6f);
  const float3 query = position / safe_scale;
  const int3 base = int3(int(std::floor(query.x)),
                         int(std::floor(query.y)),
                         int(std::floor(query.z)));
  float best_distance_sq = std::numeric_limits<float>::max();
  int3 best_cell = base;
  for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        const int3 cell = base + int3(dx, dy, dz);
        const float3 site = float3(cell.x, cell.y, cell.z) +
                            float3(voronoi_cell_random(cell, 1),
                                   voronoi_cell_random(cell, 2),
                                   voronoi_cell_random(cell, 3));
        const float distance_sq = math::length_squared(query - site);
        if (distance_sq < best_distance_sq) {
          best_distance_sq = distance_sq;
          best_cell = cell;
        }
      }
    }
  }
  return int(BLI_hash_int_3d(
      uint32_t(best_cell.x), uint32_t(best_cell.y), uint32_t(best_cell.z)));
}

/**
 * Writes the strain threshold of every edge of a cloth. The threshold is baked per edge so the
 * simulation only has to compare the current length with the rest length; 0 means "this edge can
 * not tear". The actual splitting happens in the asset, after the solve.
 */
static void write_cloth_tear(Mesh &mesh,
                             const bool enabled,
                             const PuncligaTearMode mode,
                             const float threshold,
                             const fn::Field<float> &edge_group_field,
                             const float voronoi_scale)
{
  const int edge_num = mesh.edges_num;
  if (edge_num == 0) {
    return;
  }
  bke::SpanAttributeWriter<float> threshold_writer =
      mesh.attributes_for_write().lookup_or_add_for_write_only_span<float>(
          attribute_names::tear_threshold, bke::AttrDomain::Edge);
  if (!threshold_writer) {
    return;
  }
  if (!enabled) {
    threshold_writer.span.fill(0.0f);
    threshold_writer.finish();
    return;
  }

  if (mode == PuncligaTearMode::All) {
    threshold_writer.span.fill(threshold);
  }
  else if (mode == PuncligaTearMode::Custom) {
    const bke::GeometryFieldContext field_context{mesh, bke::AttrDomain::Edge};
    fn::FieldEvaluator evaluator{field_context, edge_num};
    evaluator.add(edge_group_field);
    evaluator.evaluate();
    const VArray<float> edge_group = evaluator.get_evaluated<float>(0);
    for (const int i : IndexRange(edge_num)) {
      threshold_writer.span[i] = edge_group[i] > 0.5f ? threshold : 0.0f;
    }
  }
  else if (mode == PuncligaTearMode::Voronoi) {
    const Span<float3> positions = mesh.vert_positions();
    Array<int> cell_ids(mesh.verts_num);
    for (const int i : IndexRange(mesh.verts_num)) {
      cell_ids[i] = voronoi_cell_id(positions[i], voronoi_scale);
    }
    const Span<int2> edges = mesh.edges();
    for (const int i : IndexRange(edge_num)) {
      const int2 &edge = edges[i];
      threshold_writer.span[i] = cell_ids[edge[0]] != cell_ids[edge[1]] ? threshold : 0.0f;
    }
  }
  threshold_writer.finish();
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
    else if (type == int(PuncligaType::Rigid)) {
      /* A point represents one sphere of the packing. */
      const float density = params.get_input<float>("Rigid Density"_ustr);
      mass = density * (4.0f / 3.0f) * 3.14159265358979323846f * radius * radius * radius;
    }
    else if (type == int(PuncligaType::Grain)) {
      const float density = params.get_input<float>("Grain Density"_ustr);
      mass = density * (4.0f / 3.0f) * 3.14159265358979323846f * radius * radius * radius;
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
  /* A sphere packing writes its own per point radius; keep it instead of replacing it with the
   * fallback of the shared `Collision Radius` socket. */
  const bool keep_radius = type == int(PuncligaType::Rigid) && has_point_radius(geometry);
  if (!keep_radius) {
    write_uniform<float>(geometry, "radius", bke::AttrDomain::Point, radius);
  }
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
      if (Mesh *mesh = geometry.get_mesh_for_write()) {
        write_cloth_tear(*mesh,
                         params.get_input<bool>("Tear"_ustr),
                         PuncligaTearMode(params.get_input<int>("Tear Mode"_ustr)),
                         params.get_input<float>("Tear Threshold"_ustr),
                         params.extract_input<Field<float>>("Tear Edge Group"_ustr),
                         params.get_input<float>("Voronoi Scale"_ustr));
      }
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
      write_uniform<bool>(geometry,
                          attribute_names::guide_tether,
                          bke::AttrDomain::Point,
                          params.get_input<bool>("Guide Tether"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::tether_compliance,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Tether Compliance"_ustr));
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
      write_uniform<float>(geometry,
                           attribute_names::emission_spacing,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Emission Spacing"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::emission_velocity_jitter,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Emission Velocity Jitter"_ustr));
      write_uniform<bool>(geometry,
                          attribute_names::boundary_particles,
                          bke::AttrDomain::Point,
                          params.get_input<bool>("Boundary Particles"_ustr));
      break;
    }
    case PuncligaType::Rigid: {
      if (geometry.get_mesh() == nullptr && geometry.get_pointcloud() == nullptr) {
        params.error_message_add(
            NodeWarningType::Warning,
            TIP_("Rigid needs a mesh or a point cloud (the points are usually a sphere packing of "
                 "the mesh, made with `Distribute Points` in volume mode)"));
        break;
      }
      write_rest_positions(geometry);
      if (params.get_input<float>("Mass"_ustr) <= 0.0f) {
        /* Variable sized spheres need a per point mass. */
        write_rigid_mass(
            geometry, params.get_input<float>("Rigid Density"_ustr), radius);
      }
      write_shape_matching(geometry,
                           params.get_input<bool>("Shape Matching"_ustr),
                           params.extract_input<Field<int>>("Shape Group"_ustr),
                           params.extract_input<Field<float>>("Shape Stiffness"_ustr),
                           params.get_input<int>("Piece ID"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::linear_damping,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Rigid Linear Damping"_ustr));
      write_uniform<bool>(geometry,
                          attribute_names::self_collision,
                          bke::AttrDomain::Point,
                          params.get_input<bool>("Rigid Self Collision"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::contact_thickness,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Rigid Contact Thickness"_ustr));
      break;
    }
    case PuncligaType::Grain: {
      write_uniform<float>(geometry,
                           attribute_names::grain_softness,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Grain Contact Softness"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::grain_restitution,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Grain Restitution"_ustr));
      write_uniform<bool>(geometry,
                          attribute_names::grain_clump,
                          bke::AttrDomain::Point,
                          params.get_input<bool>("Clump"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::grain_clump_radius,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Clump Radius"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::grain_clump_strength,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Clump Strength"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::grain_break_strain,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Break Strain"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::grain_break_fraction,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Break Fraction"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::grain_reclump_delay,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Re-clump Delay"_ustr));
      write_uniform<int>(geometry,
                         attribute_names::grain_clump_min_size,
                         bke::AttrDomain::Point,
                         params.get_input<int>("Clump Min Size"_ustr));
      write_uniform<int>(geometry,
                         attribute_names::grain_clump_max_size,
                         bke::AttrDomain::Point,
                         params.get_input<int>("Clump Max Size"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::grain_clump_speed,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Clump Speed"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::contact_thickness,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Grain Contact Thickness"_ustr));
      write_uniform<float>(geometry,
                           attribute_names::linear_damping,
                           bke::AttrDomain::Point,
                           params.get_input<float>("Grain Linear Damping"_ustr));
      write_grain_clump_state(geometry);
      write_rest_positions(geometry);
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
      "Mark geometry as cloth, fluid, soft body, hair or a rigid piece and store the parameters "
      "that `Puncliga Dynamics` needs";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  ntype.labelfunc = node_label;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_init_puncliga_data_cc
