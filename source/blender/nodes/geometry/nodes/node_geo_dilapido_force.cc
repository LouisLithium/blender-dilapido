/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Dilapido Force`: one force field as a bundle, the force counterpart of `Dilapido Collider`.
 *
 * The bundle is typed (`Blender.DilapidoForce`) and carries a uniform acceleration (wind) plus two
 * procedural noise forces: a low frequency disturbance and a curl noise turbulence. It is nested
 * into a solver with `Combine Bundle` (or by storing it under its own path), exactly like a
 * collider, so several forces and colliders can act at the same time.
 *
 * Keys written:
 *
 * | key                       | meaning                                      |
 * |---------------------------|----------------------------------------------|
 * | `Type`                    | `Blender.DilapidoForce`                      |
 * | `acceleration`            | uniform force per unit mass, in m/s²         |
 * | `disturbance`             | disturbance force amplitude, in m/s²         |
 * | `disturbance_scale`       | base feature size of the disturbance, in m   |
 * | `disturbance_speed`       | how fast the disturbance pattern moves       |
 * | `disturbance_seed`        | noise seed                                   |
 * | `disturbance_octaves`     | number of noise layers                       |
 * | `disturbance_lacunarity`  | feature size ratio between layers            |
 * | `disturbance_roughness`   | amplitude ratio between layers               |
 * | `turbulence`              | turbulence force amplitude, in m/s²          |
 * | `turbulence_scale`        | base swirl size of the turbulence, in m      |
 * | `turbulence_speed`        | how fast the turbulence pattern moves        |
 * | `turbulence_seed`         | noise seed                                   |
 * | `turbulence_octaves`      | number of noise layers                       |
 * | `turbulence_lacunarity`   | feature size ratio between layers            |
 * | `turbulence_roughness`    | amplitude ratio between layers               |
 * | `time`                    | scene time in seconds, for the noise motion  |
 *
 * The uniform acceleration is the one force every solver can apply without a per element closure.
 * The noise forces are consumed by the grid solver; a particle or rigid body solver that finds the
 * keys can sample the same fields without changing the bundle vocabulary.
 */

#include <algorithm>

#include "NOD_bundle_type.hh"
#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_socket_declarations.hh"

#include "BKE_scene.hh"

#include "DEG_depsgraph_query.hh"

#include "BLI_math_vector.hh"
#include "BLI_string_ref.hh"

#include "GEO_ignis_remap.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_dilapido_force_cc {

namespace {

constexpr StringRef FORCE_TYPE = "Blender.DilapidoForce";

/** The bundle type, registered so that assets can create and edit these bundles in a node tree. */
class DilapidoForceBundle {
 public:
  static constexpr StringRefNull name = "Blender.DilapidoForce";
  static const FlatBundleTypePtr &get_bundle_type();
};

const FlatBundleTypePtr &DilapidoForceBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(DilapidoForceBundle::name);
    b.add<decl::Vector>("acceleration"_ustr)
        .subtype(PROP_ACCELERATION)
        .description("Acceleration the field adds to whatever it acts on, in m/s²");
    b.add<decl::Float>("disturbance"_ustr)
        .subtype(PROP_ACCELERATION)
        .description("Low frequency noise force, in m/s²");
    b.add<decl::Float>("disturbance_scale"_ustr)
        .subtype(PROP_DISTANCE)
        .description("Base feature size of the disturbance, in metres");
    b.add<decl::Float>("disturbance_speed"_ustr).description(
        "How fast the disturbance pattern moves through space");
    b.add<decl::Int>("disturbance_seed"_ustr).description("Disturbance noise seed");
    b.add<decl::Int>("disturbance_octaves"_ustr).min(1).description(
        "Number of disturbance noise layers");
    b.add<decl::Float>("disturbance_lacunarity"_ustr).min(1.0f).description(
        "Feature size ratio between disturbance layers");
    b.add<decl::Float>("disturbance_roughness"_ustr).min(0.0f).max(1.0f).description(
        "Amplitude ratio between disturbance layers");
    b.add<decl::Float>("turbulence"_ustr)
        .subtype(PROP_ACCELERATION)
        .description("Curl noise turbulence force, in m/s²");
    b.add<decl::Float>("turbulence_scale"_ustr)
        .subtype(PROP_DISTANCE)
        .description("Base swirl size of the turbulence, in metres");
    b.add<decl::Float>("turbulence_speed"_ustr).description(
        "How fast the turbulence pattern moves through space");
    b.add<decl::Int>("turbulence_seed"_ustr).description("Turbulence noise seed");
    b.add<decl::Int>("turbulence_octaves"_ustr).min(1).description(
        "Number of turbulence noise layers");
    b.add<decl::Float>("turbulence_lacunarity"_ustr).min(1.0f).description(
        "Feature size ratio between turbulence layers");
    b.add<decl::Float>("turbulence_roughness"_ustr).min(0.0f).max(1.0f).description(
        "Amplitude ratio between turbulence layers");
    b.add<decl::Int>("disturbance_remap_field"_ustr).min(0).description(
        "Grid that selects where the disturbance applies");
    b.add<decl::Int>("disturbance_remap_mode"_ustr).min(0).description(
        "Range, Outside, Above or Below");
    b.add<decl::Float>("disturbance_remap_min"_ustr).description("Lower field bound");
    b.add<decl::Float>("disturbance_remap_max"_ustr).description("Upper field bound");
    b.add<decl::Float>("disturbance_remap_softness"_ustr)
        .min(0.0f)
        .max(1.0f)
        .description("Fraction of the field window used as a smooth edge");
    b.add<decl::Float>("disturbance_remap_scale"_ustr).description(
        "Multiplier applied to the remapped disturbance");
    b.add<decl::Int>("turbulence_remap_field"_ustr).min(0).description(
        "Grid that selects where the turbulence applies");
    b.add<decl::Int>("turbulence_remap_mode"_ustr).min(0).description(
        "Range, Outside, Above or Below");
    b.add<decl::Float>("turbulence_remap_min"_ustr).description("Lower field bound");
    b.add<decl::Float>("turbulence_remap_max"_ustr).description("Upper field bound");
    b.add<decl::Float>("turbulence_remap_softness"_ustr)
        .min(0.0f)
        .max(1.0f)
        .description("Fraction of the field window used as a smooth edge");
    b.add<decl::Float>("turbulence_remap_scale"_ustr).description(
        "Multiplier applied to the remapped turbulence");
    b.add<decl::Float>("time"_ustr).description("Scene time in seconds, used to animate the noise");
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

/** The values of one `Dilapido Field Remap` bundle. */
struct RemapSpec {
  int field = int(geometry::ignis::ForceRemapField::None);
  int mode = int(geometry::ignis::ForceRemapMode::Range);
  float min = 0.0f;
  float max = 1.0f;
  float softness = 0.2f;
  float scale = 1.0f;
};

RemapSpec read_remap(const BundlePtr &bundle_ptr)
{
  RemapSpec remap;
  if (!bundle_ptr) {
    return remap;
  }
  const Bundle &bundle = *bundle_ptr;
  if (const int *value = bundle.lookup_ptr<int>(*BundleKey::from_str("field"))) {
    remap.field = *value;
  }
  if (const int *value = bundle.lookup_ptr<int>(*BundleKey::from_str("mode"))) {
    remap.mode = *value;
  }
  if (const float *value = bundle.lookup_ptr<float>(*BundleKey::from_str("min"))) {
    remap.min = *value;
  }
  if (const float *value = bundle.lookup_ptr<float>(*BundleKey::from_str("max"))) {
    remap.max = *value;
  }
  if (const float *value = bundle.lookup_ptr<float>(*BundleKey::from_str("softness"))) {
    remap.softness = *value;
  }
  if (const float *value = bundle.lookup_ptr<float>(*BundleKey::from_str("scale"))) {
    remap.scale = *value;
  }
  return remap;
}

/** Stores the remap values under `<prefix>_remap_*` in the force bundle. */
void write_remap(Bundle &bundle, const StringRef prefix, const RemapSpec &remap)
{
  const std::string field_key = std::string(prefix) + "_remap_field";
  const std::string mode_key = std::string(prefix) + "_remap_mode";
  const std::string min_key = std::string(prefix) + "_remap_min";
  const std::string max_key = std::string(prefix) + "_remap_max";
  const std::string softness_key = std::string(prefix) + "_remap_softness";
  const std::string scale_key = std::string(prefix) + "_remap_scale";
  bundle.add(*BundleKey::from_str(field_key), remap.field);
  bundle.add(*BundleKey::from_str(mode_key), remap.mode);
  bundle.add(*BundleKey::from_str(min_key), remap.min);
  bundle.add(*BundleKey::from_str(max_key), remap.max);
  bundle.add(*BundleKey::from_str(softness_key), remap.softness);
  bundle.add(*BundleKey::from_str(scale_key), remap.scale);
}

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  /* Panels are only drawn when the declaration owns the socket order. Without this, the two
   * noise panels exist in the runtime but the node falls back to a flat socket list. */
  b.use_custom_socket_order();

  b.add_input<decl::Vector>("Direction"_ustr)
      .default_value(float3(1.0f, 0.0f, 0.0f))
      .subtype(PROP_DIRECTION)
      .description("Direction of the wind. It is normalized, so only the strength sets the size");
  b.add_input<decl::Float>("Strength"_ustr)
      .default_value(0.0f)
      .subtype(PROP_ACCELERATION)
      .description(
          "Acceleration the wind adds along the direction, in m/s². It is compared with gravity: "
          "`9.81` pushes as hard as gravity pulls; give it a negative value to blow the other way");

  auto &disturbance_panel = b.add_panel("Disturbance"_ustr).default_closed(false);
  disturbance_panel.add_input<decl::Float>("Disturbance"_ustr)
      .default_value(0.0f)
      .subtype(PROP_ACCELERATION)
      .description(
          "Low frequency noise force, in m/s². It adds large, soft pushes and swirls and is the "
          "Houdini `Gas Disturb` counterpart");
  disturbance_panel.add_input<decl::Float>("Disturbance Scale"_ustr)
      .default_value(0.5f)
      .min(1e-4f)
      .subtype(PROP_DISTANCE)
      .description("Base feature size of the disturbance, in metres");
  disturbance_panel.add_input<decl::Float>("Disturbance Speed"_ustr)
      .default_value(1.0f)
      .description("How fast the disturbance pattern moves through space");
  disturbance_panel.add_input<decl::Int>("Disturbance Octaves"_ustr)
      .default_value(2)
      .min(1)
      .max(8)
      .description("Number of noise layers");
  disturbance_panel.add_input<decl::Float>("Disturbance Lacunarity"_ustr)
      .default_value(2.0f)
      .min(1.0f)
      .description("Feature size ratio between successive layers");
  disturbance_panel.add_input<decl::Float>("Disturbance Roughness"_ustr)
      .default_value(0.5f)
      .min(0.0f)
      .max(1.0f)
      .description("Amplitude ratio between successive layers");
  disturbance_panel.add_input<decl::Int>("Disturbance Seed"_ustr)
      .default_value(0)
      .description("Noise seed; change it for a different disturbance pattern");
  disturbance_panel.add_input<decl::Bundle>("Disturbance Remap"_ustr)
      .description("Optional `Dilapido Field Remap` bundle that scales this force by a grid value");

  auto &turbulence_panel = b.add_panel("Turbulence"_ustr).default_closed(false);
  turbulence_panel.add_input<decl::Float>("Turbulence"_ustr)
      .default_value(0.0f)
      .subtype(PROP_ACCELERATION)
      .description(
          "Curl noise turbulence force, in m/s². It adds small scale, divergence free eddies and "
          "is the Houdini `Gas Turbulence` counterpart");
  turbulence_panel.add_input<decl::Float>("Turbulence Scale"_ustr)
      .default_value(0.25f)
      .min(1e-4f)
      .subtype(PROP_DISTANCE)
      .description("Base swirl size of the turbulence, in metres");
  turbulence_panel.add_input<decl::Float>("Turbulence Speed"_ustr)
      .default_value(1.0f)
      .description("How fast the turbulence pattern moves through space");
  turbulence_panel.add_input<decl::Int>("Turbulence Octaves"_ustr)
      .default_value(3)
      .min(1)
      .max(8)
      .description("Number of noise layers");
  turbulence_panel.add_input<decl::Float>("Turbulence Lacunarity"_ustr)
      .default_value(2.0f)
      .min(1.0f)
      .description("Feature size ratio between successive layers");
  turbulence_panel.add_input<decl::Float>("Turbulence Roughness"_ustr)
      .default_value(0.5f)
      .min(0.0f)
      .max(1.0f)
      .description("Amplitude ratio between successive layers");
  turbulence_panel.add_input<decl::Int>("Turbulence Seed"_ustr)
      .default_value(0)
      .description("Noise seed; change it for a different turbulence pattern");
  turbulence_panel.add_input<decl::Bundle>("Turbulence Remap"_ustr)
      .description("Optional `Dilapido Field Remap` bundle that scales this force by a grid value");

  b.add_output<decl::Bundle>("Force"_ustr)
      .description(
          "Connect it to the `Effectors` input of a solver, or nest it in a bundle with `Combine "
          "Bundle`");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const float3 direction = params.extract_input<float3>("Direction"_ustr);
  const float strength = params.extract_input<float>("Strength"_ustr);
  const float length = math::length(direction);
  /* A zero direction is allowed and simply does nothing, so the node never has to warn about it. */
  const float3 acceleration = length > 0.0f ? direction * (strength / length) : float3(0.0f);

  const float disturbance = params.extract_input<float>("Disturbance"_ustr);
  const float disturbance_scale = std::max(1e-4f, params.extract_input<float>("Disturbance Scale"_ustr));
  const float disturbance_speed = params.extract_input<float>("Disturbance Speed"_ustr);
  const int disturbance_octaves = std::clamp(
      params.extract_input<int>("Disturbance Octaves"_ustr), 1, 8);
  const float disturbance_lacunarity = std::max(
      1.0f, params.extract_input<float>("Disturbance Lacunarity"_ustr));
  const float disturbance_roughness = std::clamp(
      params.extract_input<float>("Disturbance Roughness"_ustr), 0.0f, 1.0f);
  const int disturbance_seed = params.extract_input<int>("Disturbance Seed"_ustr);

  const float turbulence = params.extract_input<float>("Turbulence"_ustr);
  const float turbulence_scale = std::max(1e-4f, params.extract_input<float>("Turbulence Scale"_ustr));
  const float turbulence_speed = params.extract_input<float>("Turbulence Speed"_ustr);
  const int turbulence_octaves = std::clamp(
      params.extract_input<int>("Turbulence Octaves"_ustr), 1, 8);
  const float turbulence_lacunarity = std::max(
      1.0f, params.extract_input<float>("Turbulence Lacunarity"_ustr));
  const float turbulence_roughness = std::clamp(
      params.extract_input<float>("Turbulence Roughness"_ustr), 0.0f, 1.0f);
  const int turbulence_seed = params.extract_input<int>("Turbulence Seed"_ustr);

  const RemapSpec disturbance_remap = read_remap(
      params.extract_input<BundlePtr>("Disturbance Remap"_ustr));
  const RemapSpec turbulence_remap = read_remap(
      params.extract_input<BundlePtr>("Turbulence Remap"_ustr));

  float time = 0.0f;
  if (const Scene *scene = DEG_get_input_scene(params.depsgraph())) {
    const double frame_rate = double(scene->r.frs_sec) / std::max(1.0, double(scene->r.frs_sec_base));
    time = float(BKE_scene_ctime_get(scene) / frame_rate);
  }

  BundlePtr bundle_ptr = Bundle::create();
  Bundle &bundle = bundle_ptr.ensure_mutable_inplace();
  bundle.add(Bundle::type_item_name, std::string(FORCE_TYPE));
  bundle.add(*BundleKey::from_str("acceleration"), acceleration);
  bundle.add(*BundleKey::from_str("disturbance"), disturbance);
  bundle.add(*BundleKey::from_str("disturbance_scale"), disturbance_scale);
  bundle.add(*BundleKey::from_str("disturbance_speed"), disturbance_speed);
  bundle.add(*BundleKey::from_str("disturbance_seed"), disturbance_seed);
  bundle.add(*BundleKey::from_str("disturbance_octaves"), disturbance_octaves);
  bundle.add(*BundleKey::from_str("disturbance_lacunarity"), disturbance_lacunarity);
  bundle.add(*BundleKey::from_str("disturbance_roughness"), disturbance_roughness);
  bundle.add(*BundleKey::from_str("turbulence"), turbulence);
  bundle.add(*BundleKey::from_str("turbulence_scale"), turbulence_scale);
  bundle.add(*BundleKey::from_str("turbulence_speed"), turbulence_speed);
  bundle.add(*BundleKey::from_str("turbulence_seed"), turbulence_seed);
  bundle.add(*BundleKey::from_str("turbulence_octaves"), turbulence_octaves);
  bundle.add(*BundleKey::from_str("turbulence_lacunarity"), turbulence_lacunarity);
  bundle.add(*BundleKey::from_str("turbulence_roughness"), turbulence_roughness);
  write_remap(bundle, "disturbance", disturbance_remap);
  write_remap(bundle, "turbulence", turbulence_remap);
  bundle.add(*BundleKey::from_str("time"), time);
  params.set_output("Force"_ustr, std::move(bundle_ptr));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeDilapidoForce"_ustr);
  ntype.ui_name = "Dilapido Force";
  ntype.ui_description =
      "A force field (uniform wind plus procedural disturbance and turbulence) that the Dilapido "
      "solvers can consume";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  DilapidoForceBundle::get_bundle_type();
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_dilapido_force_cc
