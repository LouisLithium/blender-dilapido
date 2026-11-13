/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Init Textura Data`: mark a geometry as an MPM material and store the physical parameters as
 * point attributes.
 *
 * The MPM solver reads everything it needs per particle from these attributes, so several
 * materials with different densities, stiffnesses or friction angles can be simulated in one
 * world. The node is the MPM counterpart of `Init Puncliga Data`: it turns the input geometry into
 * a particle set (one particle per point) and writes the deformation gradient as identity, which
 * is the rest state of the solver.
 */

#include <cmath>

#include "BKE_attribute.hh"
#include "BKE_curves.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "BLI_math_vector.hh"

#include "node_geometry_util.hh"

#include "GEO_textura_solver.hh"

namespace blender::nodes::node_geo_init_textura_data_cc {

using geometry::textura::MaterialType;

namespace {

constexpr float PI = 3.14159265358979323846f;

static const EnumPropertyItem textura_type_items[] = {
    {int(MaterialType::Fluid),
     "FLUID",
     0,
     N_("Fluid"),
     N_("A weakly compressible fluid: only the volumetric part of the deformation is resisted. "
        "Use it for water, mud and other low viscosity materials")},
    {int(MaterialType::Sand),
     "SAND",
     0,
     N_("Sand"),
     N_("A granular material: elastic until the Drucker-Prager yield surface is reached, then it "
        "flows. Friction angle and cohesion control how steep a pile it can hold")},
    {int(MaterialType::Snow),
     "SNOW",
     0,
     N_("Snow"),
     N_("A compressible cohesive granular material with a small tensile strength. It compacts "
        "permanently under pressure, hardens as it compacts and breaks under tension or impact")},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem textura_emission_items[] = {
    {0,
     "NONE",
     0,
     N_("None"),
     N_("The particles are only the initial state: they are simulated once and never injected "
        "again")},
    {1,
     "PER_FRAME",
     0,
     N_("Per Frame"),
     N_("Inject the emitter's particles once per frame, at its pose of this frame. Use it for a "
        "nozzle/object that keeps pouring new material into the pool")},
    {2,
     "PER_SUBSTEP",
     0,
     N_("Per Substep"),
     N_("Inject several copies per frame, spread along the distance the particles travel, so a "
        "fast stream is continuous instead of one lump per frame")},
    {0, nullptr, 0, nullptr, nullptr},
};

struct MaterialParameters {
  int type = int(MaterialType::Fluid);
  int emission = 0;
  float emission_spacing = 0.0f;
  float emission_velocity_jitter = 0.0f;
  float density = 1000.0f;
  float radius = 0.05f;
  float volume = 0.0f;
  float mass = 0.0f;
  float damping = 0.1f;
  float bulk_modulus = 5.0e4f;
  float viscosity = 0.2f;
  float tension = 0.0f;
  float surface_tension = 0.0f;
  float young_modulus = 1.0e5f;
  float poisson_ratio = 0.3f;
  float friction_angle = 30.0f;
  float cohesion = 0.0f;
  float damage_rate = 0.0f;
  float tensile_strength = 0.0f;
  float compression_limit = 0.05f;
  float stretch_limit = 0.01f;
  float hardening_rate = 0.0f;
  float wall_friction = 0.0f;
  float adhesion = 0.0f;
};

void compute_lame(const MaterialParameters &params, float &r_mu, float &r_lambda)
{
  const float poisson = math::clamp(params.poisson_ratio, 0.0f, 0.49f);
  const float young = std::max(params.young_modulus, 1.0f);
  r_mu = young / (2.0f * (1.0f + poisson));
  r_lambda = young * poisson / ((1.0f + poisson) * (1.0f - 2.0f * poisson));
}

/** Writes the per point attributes of one component and evaluates the initial velocity field. */
void write_component(GeometryComponent &component,
                     const MaterialParameters &params,
                     const fn::Field<float3> &velocity_field)
{
  const AttrDomain domain = AttrDomain::Point;
  const int count = component.attribute_domain_size(domain);
  if (count == 0) {
    return;
  }
  bke::MutableAttributeAccessor attributes = *component.attributes_for_write();
  const VArray<float3> positions = *attributes.lookup<float3>("position", domain);
  const bool has_positions = bool(positions);

  const float radius = std::max(params.radius, 1e-5f);
  const float volume = params.volume > 0.0f ? params.volume :
                                              (2.0f * radius) * (2.0f * radius) * (2.0f * radius);
  const float mass = params.mass > 0.0f ? params.mass : params.density * volume;

  float mu = 0.0f;
  float lambda = 0.0f;
  float bulk = params.bulk_modulus;
  float viscosity = params.viscosity;
  float tension = params.tension;
  float surface_tension = params.surface_tension;
  float friction = 0.0f;
  float cohesion = 0.0f;
  float damage_rate = 0.0f;
  float tensile_strength = 0.0f;
  float compression_limit = 0.05f;
  float stretch_limit = 0.01f;
  float hardening_rate = 0.0f;
  if (params.type == int(MaterialType::Sand)) {
    compute_lame(params, mu, lambda);
    bulk = 0.0f;
    viscosity = 0.0f;
    tension = 0.0f;
    surface_tension = 0.0f;
    friction = std::tan(std::max(params.friction_angle, 0.0f) * PI / 180.0f);
    cohesion = std::max(params.cohesion, 0.0f);
    damage_rate = std::max(params.damage_rate, 0.0f);
  }
  else if (params.type == int(MaterialType::Snow)) {
    compute_lame(params, mu, lambda);
    bulk = 0.0f;
    viscosity = 0.0f;
    tension = 0.0f;
    surface_tension = 0.0f;
    friction = std::tan(std::max(params.friction_angle, 0.0f) * PI / 180.0f);
    cohesion = std::max(params.cohesion, 0.0f);
    damage_rate = std::max(params.damage_rate, 0.0f);
    tensile_strength = std::max(params.tensile_strength, 0.0f);
    compression_limit = std::max(params.compression_limit, 0.0f);
    stretch_limit = std::max(params.stretch_limit, 0.0f);
    hardening_rate = std::max(params.hardening_rate, 0.0f);
  }

  auto write_scalar = [&](const StringRef name, const float value) {
    bke::SpanAttributeWriter<float> writer =
        attributes.lookup_or_add_for_write_only_span<float>(name, domain);
    if (writer) {
      writer.span.fill(value);
      writer.finish();
    }
  };
  auto write_int = [&](const StringRef name, const int value) {
    bke::SpanAttributeWriter<int> writer =
        attributes.lookup_or_add_for_write_only_span<int>(name, domain);
    if (writer) {
      writer.span.fill(value);
      writer.finish();
    }
  };

  write_int(geometry::textura::attribute_names::type, params.type);
  write_scalar(geometry::textura::attribute_names::density, params.density);
  write_scalar(geometry::textura::attribute_names::volume, volume);
  write_scalar(geometry::textura::attribute_names::bulk_modulus, bulk);
  write_scalar(geometry::textura::attribute_names::shear_modulus, mu);
  write_scalar(geometry::textura::attribute_names::lame_lambda, lambda);
  write_scalar(geometry::textura::attribute_names::viscosity, viscosity);
  write_scalar(geometry::textura::attribute_names::tension, tension);
  write_scalar(geometry::textura::attribute_names::surface_tension, surface_tension);
  write_scalar(geometry::textura::attribute_names::friction, friction);
  write_scalar(geometry::textura::attribute_names::cohesion, cohesion);
  if (params.type == int(MaterialType::Snow)) {
    write_scalar(geometry::textura::attribute_names::tensile_strength, tensile_strength);
    write_scalar(geometry::textura::attribute_names::compression_limit, compression_limit);
    write_scalar(geometry::textura::attribute_names::stretch_limit, stretch_limit);
    write_scalar(geometry::textura::attribute_names::rest_density, mass / volume);
    write_scalar(geometry::textura::attribute_names::hardening_rate, hardening_rate);
    write_scalar(geometry::textura::attribute_names::wall_friction, params.wall_friction);
    write_scalar(geometry::textura::attribute_names::adhesion, params.adhesion);
    /* Keep an authored hardening field (for restarted states); otherwise start at zero. */
    if (!attributes.contains(geometry::textura::attribute_names::hardening)) {
      write_scalar(geometry::textura::attribute_names::hardening, 0.0f);
    }
  }
  /* An existing damage attribute is authored (weak planes / pre-cracked regions), so keep it. */
  if (!attributes.contains(geometry::textura::attribute_names::damage)) {
    write_scalar(geometry::textura::attribute_names::damage, 0.0f);
  }
  write_scalar(geometry::textura::attribute_names::damage_rate, damage_rate);
  write_scalar(geometry::textura::attribute_names::damping, params.damping);
  write_int(geometry::textura::attribute_names::emission, params.emission);
  write_scalar(geometry::textura::attribute_names::emission_spacing, params.emission_spacing);
  write_scalar(geometry::textura::attribute_names::emission_velocity_jitter,
               params.emission_velocity_jitter);
  write_scalar("radius", radius);
  write_scalar("mass", mass);

  if (has_positions) {
    bke::SpanAttributeWriter<float3> rest = attributes.lookup_or_add_for_write_only_span<float3>(
        "rest_position", domain);
    bke::SpanAttributeWriter<float3> textura_rest =
        attributes.lookup_or_add_for_write_only_span<float3>(
            geometry::textura::attribute_names::rest_position, domain);
    if (rest) {
      for (const int i : IndexRange(count)) {
        rest.span[i] = positions[i];
      }
      rest.finish();
    }
    if (textura_rest) {
      for (const int i : IndexRange(count)) {
        textura_rest.span[i] = positions[i];
      }
      textura_rest.finish();
    }
  }

  /* The rest state of the MPM solver is `F = I`. */
  bke::SpanAttributeWriter<float3> f_x = attributes.lookup_or_add_for_write_only_span<float3>(
      geometry::textura::attribute_names::deformation_x, domain);
  bke::SpanAttributeWriter<float3> f_y = attributes.lookup_or_add_for_write_only_span<float3>(
      geometry::textura::attribute_names::deformation_y, domain);
  bke::SpanAttributeWriter<float3> f_z = attributes.lookup_or_add_for_write_only_span<float3>(
      geometry::textura::attribute_names::deformation_z, domain);
  if (f_x) {
    f_x.span.fill(float3(1.0f, 0.0f, 0.0f));
    f_x.finish();
  }
  if (f_y) {
    f_y.span.fill(float3(0.0f, 1.0f, 0.0f));
    f_y.finish();
  }
  if (f_z) {
    f_z.span.fill(float3(0.0f, 0.0f, 1.0f));
    f_z.finish();
  }

  if (params.type == int(MaterialType::Snow)) {
    bke::SpanAttributeWriter<float3> fp_x = attributes.lookup_or_add_for_write_only_span<float3>(
        geometry::textura::attribute_names::plastic_x, domain);
    bke::SpanAttributeWriter<float3> fp_y = attributes.lookup_or_add_for_write_only_span<float3>(
        geometry::textura::attribute_names::plastic_y, domain);
    bke::SpanAttributeWriter<float3> fp_z = attributes.lookup_or_add_for_write_only_span<float3>(
        geometry::textura::attribute_names::plastic_z, domain);
    if (fp_x && fp_y && fp_z) {
      fp_x.span.fill(float3(1.0f, 0.0f, 0.0f));
      fp_y.span.fill(float3(0.0f, 1.0f, 0.0f));
      fp_z.span.fill(float3(0.0f, 0.0f, 1.0f));
      fp_x.finish();
      fp_y.finish();
      fp_z.finish();
    }
  }

  /* Evaluate the initial velocity field. A field that is zero everywhere does not write the
   * attribute, so a velocity stored upstream survives. */
  const bke::GeometryFieldContext field_context{component, domain};
  fn::FieldEvaluator evaluator{field_context, count};
  evaluator.add(velocity_field);
  evaluator.evaluate();
  const VArray<float3> velocities = evaluator.get_evaluated<float3>(0);
  bool all_zero = true;
  for (const int i : IndexRange(count)) {
    if (math::length_squared(velocities[i]) > 1e-20f) {
      all_zero = false;
      break;
    }
  }
  if (!all_zero) {
    bke::SpanAttributeWriter<float3> writer = attributes.lookup_or_add_for_write_only_span<float3>(
        "velocity", domain);
    if (writer) {
      for (const int i : IndexRange(count)) {
        writer.span[i] = velocities[i];
      }
      writer.finish();
    }
  }
}

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr)
      .is_default_link_socket()
      .description("The particles of the material: mesh vertices, point cloud points or curve "
                   "points. One point becomes one MPM particle");
  b.add_input<decl::Menu>("Type"_ustr)
      .static_items(textura_type_items)
      .default_value(MenuValue(int(MaterialType::Fluid)))
      .description("Low rigidity material model");

  {
    auto &panel = b.add_panel("Shared"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Density"_ustr)
        .default_value(1000.0f)
        .min(1e-3f)
        .description("Rest density of the material in kg/m^3. Water is 1000, dry sand about 1600 "
                     "and fresh snow 100 to 400");
    panel.add_input<decl::Float>("Particle Radius"_ustr)
        .default_value(0.05f)
        .min(1e-5f)
        .subtype(PROP_DISTANCE)
        .description("Radius of one particle. The particle volume is `(2 * radius)^3` when it is "
                     "not set explicitly, so the spacing of the input points should be about "
                     "`2 * radius`");
    panel.add_input<decl::Float>("Particle Volume"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description("Volume one particle represents in m^3. 0 derives it from the particle radius");
    panel.add_input<decl::Float>("Mass"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_MASS)
        .description("Mass of one particle. 0 derives it from the density and the particle volume");
    panel.add_input<decl::Float>("Damping"_ustr)
        .default_value(0.1f)
        .min(0.0f)
        .description("Velocity damping of the particles in 1/s");
    panel.add_input<decl::Vector>("Initial Velocity"_ustr)
        .default_value(float3(0.0f))
        .subtype(PROP_VELOCITY)
        .structure_type(StructureType::Field)
        .description("Velocity the particles start with. Use a field to give different parts of "
                     "the geometry different velocities");
    panel.add_input<decl::Menu>("Emission"_ustr)
        .static_items(textura_emission_items)
        .default_value(MenuValue(0))
        .description("Inject this geometry into `Textura Dynamics` every frame instead of only "
                     "seeding the first frame. `Per Substep` spreads the copies along the distance "
                     "they travel, so a fast stream stays continuous");
    panel.add_input<decl::Float>("Emission Spacing"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("New particles closer than this to the existing pool are skipped, in metres. "
                     "0 injects every copy (a static emitter then piles them up); about one "
                     "particle spacing keeps a stream at a stable density");
    panel.add_input<decl::Float>("Emission Velocity Jitter"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .description("Random multiplier spread for the emitted velocity (0 = identical, 0.3 = "
                     "±30%). It turns the per frame batches into a more continuous jet");
  }

  {
    auto &panel = b.add_panel("Fluid"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Bulk Modulus"_ustr)
        .default_value(5.0e4f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Fluid))
        .description("Resistance against volume change in pascal. Higher is stiffer and needs more "
                     "substeps; 5e4 is a good balance for water at a 5 cm grid");
    panel.add_input<decl::Float>("Viscosity"_ustr)
        .default_value(0.2f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Fluid))
        .description("Velocity damping of the fluid in 1/s. It stands in for the real viscosity at "
                     "the grid scale");
    panel.add_input<decl::Float>("Tensile Stress"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Fluid))
        .description("Fraction of the negative pressure a fluid transmits. Water has no tensile "
                     "strength, so 0 is the correct value; raising it makes the material pull "
                     "itself together like gel");
    panel.add_input<decl::Float>("Surface Tension"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(2.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Fluid))
        .description("Surface tension coefficient in N/m. Water is 0.072 at room temperature; 0 "
                     "disables the force. A colour field is splatted to the grid and the force is "
                     "sigma * laplacian(c) * normal, so the fluid rounds off and pulls small "
                     "droplets together. Calibrated for a particle spacing close to the voxel "
                     "size");
  }

  {
    auto &panel = b.add_panel("Sand"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Young's Modulus"_ustr)
        .default_value(1.0e6f)
        .min(1.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Sand))
        .description("Elastic stiffness of the sand before it yields, in pascal. Dry sand is stiff "
                     "(order 1 MPa); a soft value slumps under its own weight");
    panel.add_input<decl::Float>("Poisson Ratio"_ustr)
        .default_value(0.3f)
        .min(0.0f)
        .max(0.49f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Sand))
        .description("Lateral expansion under compression");
    panel.add_input<decl::Float>("Friction Angle"_ustr)
        .default_value(30.0f)
        .min(0.0f)
        .max(60.0f)
        .subtype(PROP_ANGLE)
        .usage_by_menu("Type"_ustr, int(MaterialType::Sand))
        .description("Angle of repose of the sand. 0 has no shear strength and flows like a liquid; "
                     "30 to 35 is dry sand");
    panel.add_input<decl::Float>("Cohesion"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Sand))
        .description("Shear strength that does not depend on pressure, in pascal. Use a small value "
                     "for damp sand that keeps its shape (1e4 holds a block, 0 is dry sand)");
    panel.add_input<decl::Float>("Damage Rate"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(100.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Sand))
        .description("How much damage one unit of plastic strain adds. Damaged sand loses its shear "
                     "modulus, friction and cohesion, so the bands that flowed turn into loose sand "
                     "and the block breaks apart. 0 disables damage; 1 to 3 breaks a block on impact");
  }

  {
    auto &panel = b.add_panel("Snow"_ustr).default_closed(false);
    panel.add_input<decl::Float>("Snow Young's Modulus"_ustr)
        .default_value(5.0e4f)
        .min(1.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("Elastic stiffness of fresh snow before it yields, in pascal. Snow is softer "
                     "than sand (5e4 to 1e6) and hardens as it compacts");
    panel.add_input<decl::Float>("Snow Poisson Ratio"_ustr)
        .default_value(0.2f)
        .min(0.0f)
        .max(0.49f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("Lateral expansion of snow under compression. Snow is more compressible than "
                     "dry sand, so a lower Poisson ratio is typical");
    panel.add_input<decl::Float>("Snow Friction Angle"_ustr)
        .default_value(30.0f)
        .min(0.0f)
        .max(60.0f)
        .subtype(PROP_ANGLE)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("Internal friction angle of snow; 25 to 30 is fresh snow");
    panel.add_input<decl::Float>("Snow Cohesion"_ustr)
        .default_value(2.0e3f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("Pressure independent shear strength of snow, in pascal. This is what holds a "
                     "snowball or a snowman together; 1e3 to 1e4 is a useful range");
    panel.add_input<decl::Float>("Snow Wall Friction"_ustr)
        .default_value(0.8f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("Extra tangential friction between snow particles and colliders (0..1). Snow "
                     "usually sticks to a roof or a shovel much more than water does");
    panel.add_input<decl::Float>("Adhesion"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(50.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("Adhesion acceleration toward a nearby collider, in m/s^2. 0 disables it; "
                     "values above 9.81 can hold snow under a ceiling or ledge");
    panel.add_input<decl::Float>("Tensile Strength"_ustr)
        .default_value(2.0e3f)
        .min(0.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("Largest tensile stress a snow particle transmits, in pascal. Fresh snow is "
                     "weak in tension (order 1e3 Pa), but it is what lets a snowball or a cornice "
                     "hold together. 0 makes it behave like dry sand");
    panel.add_input<decl::Float>("Compression Limit"_ustr)
        .default_value(0.05f)
        .min(0.001f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("Elastic logarithmic volume strain before snow compacts permanently. 0.05 "
                     "means it can compress 5% elastically; beyond that the particle gets denser "
                     "and harder");
    panel.add_input<decl::Float>("Stretch Limit"_ustr)
        .default_value(0.01f)
        .min(0.0f)
        .max(1.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("Elastic logarithmic volume strain before snow fails in tension. Small values "
                     "make snow break instead of stretching like rubber");
    panel.add_input<decl::Float>("Hardening"_ustr)
        .default_value(10.0f)
        .min(0.0f)
        .max(100.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("How much the stiffness and cohesion grow per unit of accumulated plastic "
                     "strain. This is what makes compacted snow firmer; 0 disables hardening");
    panel.add_input<decl::Float>("Snow Damage Rate"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(100.0f)
        .usage_by_menu("Type"_ustr, int(MaterialType::Snow))
        .description("How much damage one unit of plastic strain adds. Damaged snow loses its "
                     "shear modulus, friction, cohesion and tensile strength, so impact bands turn "
                     "into loose snow and the block breaks apart. 0 keeps it intact");
  }

  b.add_output<decl::Geometry>("Geometry"_ustr)
      .description("The same geometry with the MPM material attributes, ready for `Textura "
                   "Dynamics`");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  MaterialParameters material;
  material.type = params.extract_input<int>("Type"_ustr);
  material.density = std::max(params.extract_input<float>("Density"_ustr), 1e-3f);
  material.radius = std::max(params.extract_input<float>("Particle Radius"_ustr), 1e-5f);
  material.volume = std::max(params.extract_input<float>("Particle Volume"_ustr), 0.0f);
  material.mass = std::max(params.extract_input<float>("Mass"_ustr), 0.0f);
  material.damping = std::max(params.extract_input<float>("Damping"_ustr), 0.0f);
  material.bulk_modulus = std::max(params.extract_input<float>("Bulk Modulus"_ustr), 0.0f);
  material.viscosity = std::max(params.extract_input<float>("Viscosity"_ustr), 0.0f);
  material.tension = math::clamp(params.extract_input<float>("Tensile Stress"_ustr), 0.0f, 1.0f);
  material.surface_tension = math::clamp(params.extract_input<float>("Surface Tension"_ustr),
                                         0.0f,
                                         2.0f);
  material.young_modulus = std::max(params.extract_input<float>("Young's Modulus"_ustr), 1.0f);
  material.poisson_ratio = math::clamp(params.extract_input<float>("Poisson Ratio"_ustr),
                                       0.0f,
                                       0.49f);
  material.friction_angle = math::clamp(params.extract_input<float>("Friction Angle"_ustr),
                                        0.0f,
                                        60.0f);
  material.cohesion = std::max(params.extract_input<float>("Cohesion"_ustr), 0.0f);
  material.damage_rate = math::clamp(params.extract_input<float>("Damage Rate"_ustr), 0.0f, 100.0f);
  material.tensile_strength = std::max(params.extract_input<float>("Tensile Strength"_ustr), 0.0f);
  material.compression_limit = math::clamp(params.extract_input<float>("Compression Limit"_ustr),
                                           0.001f,
                                           1.0f);
  material.stretch_limit = math::clamp(params.extract_input<float>("Stretch Limit"_ustr),
                                       0.0f,
                                       1.0f);
  material.hardening_rate = math::clamp(params.extract_input<float>("Hardening"_ustr), 0.0f, 100.0f);
  material.emission = params.extract_input<int>("Emission"_ustr);
  material.emission_spacing = std::max(params.extract_input<float>("Emission Spacing"_ustr), 0.0f);
  material.emission_velocity_jitter = math::clamp(
      params.extract_input<float>("Emission Velocity Jitter"_ustr), 0.0f, 1.0f);
  if (material.type == int(MaterialType::Snow)) {
    material.young_modulus = std::max(params.extract_input<float>("Snow Young's Modulus"_ustr),
                                      1.0f);
    material.poisson_ratio = math::clamp(
        params.extract_input<float>("Snow Poisson Ratio"_ustr), 0.0f, 0.49f);
    material.friction_angle = math::clamp(
        params.extract_input<float>("Snow Friction Angle"_ustr), 0.0f, 60.0f);
    material.cohesion = std::max(params.extract_input<float>("Snow Cohesion"_ustr), 0.0f);
    material.damage_rate = math::clamp(params.extract_input<float>("Snow Damage Rate"_ustr),
                                       0.0f,
                                       100.0f);
    material.wall_friction = math::clamp(
        params.extract_input<float>("Snow Wall Friction"_ustr), 0.0f, 1.0f);
    material.adhesion = math::clamp(params.extract_input<float>("Adhesion"_ustr), 0.0f, 50.0f);
  }
  const fn::Field<float3> velocity_field = params.extract_input<fn::Field<float3>>(
      "Initial Velocity"_ustr);

  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Curve})
  {
    if (!geometry.has(type)) {
      continue;
    }
    GeometryComponent &component = geometry.get_component_for_write(type);
    if (component.attribute_domain_size(AttrDomain::Point) == 0) {
      continue;
    }
    write_component(component, material, velocity_field);
  }

  /* An empty input is the most common reason a Textura setup looks frozen: the material data is
   * written to nothing and the solver has no particles to move. Say so on the node instead of
   * leaving the user with a timeline that advances over a static object. */
  int point_count = 0;
  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Curve})
  {
    if (geometry.has(type)) {
      point_count += geometry.get_component(type)->attribute_domain_size(AttrDomain::Point);
    }
  }
  if (point_count == 0) {
    params.error_message_add(
        NodeWarningType::Info,
        TIP_("Init Textura Data: the input geometry holds no points, so there is nothing to "
             "simulate. Connect a mesh or point source (a `Bounding Box` of an empty object gives "
             "an empty point cloud)"));
  }

  params.set_output("Geometry"_ustr, std::move(geometry));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeInitTexturaData"_ustr);
  ntype.ui_name = "Init Textura Data";
  ntype.ui_description =
      "Mark geometry as an MPM fluid, sand or snow and store the density, stiffness, friction, "
      "tensile strength and rest deformation that `Textura Dynamics` needs";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_init_textura_data_cc
