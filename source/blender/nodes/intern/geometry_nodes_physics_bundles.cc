/* SPDX-FileCopyrightText: 2025 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "NOD_bundle_type.hh"
#include "NOD_geometry_nodes_physics_bundles.hh"
#include "NOD_socket_declarations.hh"
#include "NOD_socket_declarations_geometry.hh"

namespace blender::nodes::physics_bundles {

static void add_filter(FlatBundleTypeBuilder &b)
{
  b.add<decl::String>("filter"_ustr);
}

const FlatBundleTypePtr &MeshColliderBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(MeshColliderBundle::name);
    add_filter(b);
    b.add<decl::Geometry>("geometry"_ustr);
    b.add<decl::Float>("margin"_ustr).min(0.0f).subtype(PROP_DISTANCE);
    b.add<decl::Float>("friction"_ustr).min(0.0f);
    b.add<decl::Float>("compliance"_ustr).min(0.0f);
    b.add<decl::Bool>("deforming"_ustr).default_value(false);
    b.add<decl::Bool>("use_edge_contacts"_ustr).default_value(false);
    b.add<decl::Bool>("is_boundary"_ustr).default_value(false);
    b.add<decl::Float>("error_threshold"_ustr)
        .default_value(1e-3f)
        .min(1e-5f)
        .subtype(PROP_DISTANCE);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &BoxColliderBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(BoxColliderBundle::name);
    add_filter(b);
    b.add<decl::Geometry>("geometry"_ustr);
    b.add<decl::Float>("friction"_ustr).min(0.0f);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &DampingBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(DampingBundle::name);
    add_filter(b);
    b.add<decl::Float>("linear_damping"_ustr).min(0.0f);
    b.add<decl::Float>("angular_damping"_ustr).min(0.0f);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &PinPositionBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(PinPositionBundle::name);
    add_filter(b);
    b.add<decl::Bool>("selection"_ustr).default_value(true).structure_type(StructureType::Field);
    b.add<decl::Vector>("position"_ustr).structure_type(StructureType::Field);
    b.add<decl::Float>("compliance"_ustr).min(0.0f).structure_type(StructureType::Field);
    b.add<decl::Float>("error_threshold"_ustr)
        .default_value(1e-3f)
        .min(1e-5f)
        .subtype(PROP_DISTANCE);
    b.add<decl::String>("lambda_attribute"_ustr);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &PinRotationBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(PinRotationBundle::name);
    add_filter(b);
    b.add<decl::Bool>("selection"_ustr).default_value(true).structure_type(StructureType::Field);
    b.add<decl::Rotation>("rotation"_ustr).structure_type(StructureType::Field);
    b.add<decl::Float>("compliance"_ustr).min(0.0f).structure_type(StructureType::Field);
    b.add<decl::Float>("error_threshold"_ustr).default_value(1e-2f).min(1e-5f).subtype(PROP_ANGLE);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &CollisionContactsBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(CollisionContactsBundle::name);
    add_filter(b);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &RodStretchShearBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(RodStretchShearBundle::name);
    add_filter(b);
    b.add<decl::Float>("rest_length"_ustr).min(0.0f).structure_type(StructureType::Field);
    b.add<decl::Float>("compliance"_ustr).default_value(1e-4f).min(0.0f);
    b.add<decl::Float>("error_threshold"_ustr)
        .default_value(1e-3f)
        .min(1e-5f)
        .subtype(PROP_DISTANCE);
    b.add<decl::String>("lambda_position_attribute"_ustr);
    b.add<decl::String>("lambda_rotation_attribute"_ustr);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &RodBendTwistBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(RodBendTwistBundle::name);
    add_filter(b);
    b.add<decl::Rotation>("rest_bend_rotation"_ustr).structure_type(StructureType::Field);
    b.add<decl::Float>("compliance"_ustr).default_value(1e-4f).min(0.0f);
    b.add<decl::Float>("error_threshold"_ustr).default_value(1e-2f).min(1e-5f).subtype(PROP_ANGLE);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &EdgeLengthConstraintBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(EdgeLengthConstraintBundle::name);
    add_filter(b);
    b.add<decl::Float>("rest_length"_ustr).min(0.0f).structure_type(StructureType::Field);
    b.add<decl::Float>("compliance"_ustr).default_value(1e-4f).min(0.0f);
    b.add<decl::Float>("error_threshold"_ustr)
        .default_value(1e-3f)
        .min(1e-5f)
        .subtype(PROP_DISTANCE);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &VolumeConstraintBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(VolumeConstraintBundle::name);
    add_filter(b);
    b.add<decl::Float>("compliance"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description(
            "Softness of the volume constraint. Zero keeps the volume of the elements exactly, "
            "larger values let them compress");
    b.add<decl::Float>("edge_compliance"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description(
            "Softness of the distance constraints on the six edges of every element, which is "
            "what resists shear (a volume constraint alone lets an element flatten)");
    b.add<decl::Float>("error_threshold"_ustr)
        .default_value(1e-3f)
        .min(1e-5f)
        .description("Relative error below which the constraints count as solved");
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &CrossEdgeLengthConstraintBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(CrossEdgeLengthConstraintBundle::name);
    add_filter(b);
    b.add<decl::Vector>("rest_position"_ustr).structure_type(StructureType::Field);
    b.add<decl::Float>("compliance"_ustr).default_value(1e-4f).min(0.0f);
    b.add<decl::Float>("error_threshold"_ustr)
        .default_value(1e-3f)
        .min(1e-5f)
        .subtype(PROP_DISTANCE);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &FluidBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(FluidBundle::name);
    add_filter(b);
    b.add<decl::Float>("rest_density"_ustr).default_value(1000.0f).min(1e-4f);
    b.add<decl::Float>("particle_radius"_ustr)
        .default_value(0.05f)
        .min(0.0f)
        .subtype(PROP_DISTANCE);
    b.add<decl::Float>("compliance"_ustr).default_value(0.0f).min(0.0f);
    b.add<decl::Float>("viscosity"_ustr).default_value(0.02f).min(0.0f);
    b.add<decl::Float>("damping"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .description("Velocity damping in 1/s, damps bulk motion of the fluid");
    b.add<decl::Float>("vorticity"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description("Vorticity confinement strength, re-injects small scale rotation");
    b.add<decl::Float>("surface_tension"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description("Cohesion strength between the particles, pulls the surface layer together");
    b.add<decl::Bool>("boundary_particles"_ustr)
        .default_value(true)
        .description(
            "Sample the colliders as solid boundary particles so the density estimate is correct "
            "at walls (Akinci et al. 2012)");
    b.add<decl::Float>("boundary_friction"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .max(1.0f)
        .description("No-slip damping of the velocity next to boundary particles");
    b.add<decl::Float>("negative_pressure_limit"_ustr).default_value(0.0f).min(0.0f).max(1.0f);
    b.add<decl::Float>("artificial_pressure"_ustr).default_value(0.0f).min(0.0f);
    b.add<decl::Float>("coupling"_ustr)
        .default_value(1.0f)
        .min(0.0f)
        .max(1.0f)
        .description(
            "How strongly the fluid pushes the other simulated geometries of the same world "
            "(cloth, soft bodies); 0 disables the two way coupling");
    b.add<decl::Float>("coupling_compliance"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description(
            "XPBD compliance of the point/triangle contact with the coupled solids; 0 is a hard "
            "contact. The solver divides it by the substep time squared");
    b.add<decl::Float>("error_threshold"_ustr).default_value(1e-2f).min(1e-5f);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &ForceBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(ForceBundle::name);
    add_filter(b);
    b.add<decl::Closure>("closure"_ustr);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &CustomGeometryEffector::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(CustomGeometryEffector::name);
    add_filter(b);
    b.add<decl::String>("stage"_ustr);
    b.add<decl::Closure>("closure"_ustr);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

const FlatBundleTypePtr &CustomWorldEffector::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(CustomWorldEffector::name);
    b.add<decl::String>("stage"_ustr);
    b.add<decl::Closure>("closure"_ustr);
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

}  // namespace blender::nodes::physics_bundles
