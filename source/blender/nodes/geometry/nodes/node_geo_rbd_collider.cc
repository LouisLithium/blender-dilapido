/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Dilapido Collider node.
 *
 * Builds one collider bundle that every Dilapido solver can consume. The bundle is *typed*
 * (`Blender.Collider.Mesh` / `Blender.Collider.Box`), which is what the solvers look for, so any
 * number of colliders of any type can be nested into the world with `Combine Bundle` (or by
 * storing them under different paths) and all of them are used at the same time.
 *
 * The node writes the *union* of the keys the solvers read:
 *
 * | key                                | rigid body solver | XPBD solver |
 * |------------------------------------|-------------------|-------------|
 * | `Type` (bundle type)               | yes               | mesh only   |
 * | `geometry`, `friction`             | yes               | yes         |
 * | `convex`                           | yes               | -           |
 * | `deforming`                        | -                 | yes         |
 * | `is_boundary`, `use_edge_contacts` | -                 | yes         |
 * | `margin`, `compliance`             | -                 | yes         |
 * | `error_threshold`, `filter`        | -                 | yes         |
 * | `transform`, `sweep`               | -                 | yes         |
 *
 * A solver ignores the keys it does not know, so one node does the job for all of them.
 *
 * A native node rather than an asset, because a node group can not expose a menu with items (the
 * interface socket evaluates to an empty string), and because the collider type is a fixed enum.
 */

#include "GEO_transform.hh"

#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_physics_bundles.hh"
#include "NOD_socket_declarations.hh"

#include "BLI_string_ref.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_dilapido_collider_cc {

using namespace blender::nodes::physics_bundles;

enum class ColliderType : int {
  Mesh = 0,
  Box = 1,
  Convex = 2,
};

static const EnumPropertyItem collider_type_items[] = {
    {int(ColliderType::Mesh),
     "MESH",
     0,
     N_("Mesh"),
     N_("Collide with the triangles of the geometry. Works in every solver. A mesh is a shell, so "
        "a body that ends up behind the surface can not be pushed back out")},
    {int(ColliderType::Box),
     "BOX",
     0,
     N_("Box"),
     N_("Collide with a solid box around the bounds of the geometry. A box has volume, so a body "
        "that sinks into it is pushed back out. Rigid body solver only")},
    {int(ColliderType::Convex),
     "CONVEX",
     0,
     N_("Convex Hull"),
     N_("Collide with the solid convex hull of the geometry. Has volume like the box, but follows "
        "the shape instead of the bounds, and is much cheaper than the triangle mesh. Rigid body "
        "solver only")},
    {0, nullptr, 0, nullptr, nullptr},
};

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr)
      .description(
          "Collision geometry, in local space. Wire Object Info > Transform into Transform "
          "to place it and to make it follow an animated object");
  b.add_input<decl::Matrix>("Transform"_ustr)
      .description(
          "Transform of the collider. Wire Object Info > Transform into it: Object Info hands the "
          "geometry over in *local* space, so a collider built straight from it never moves. The "
          "solver also uses it to sweep the collider between frames (see Sweep)");
  b.add_input<decl::Menu>("Type"_ustr)
      .static_items(collider_type_items)
      .default_value(MenuValue(int(ColliderType::Mesh)))
      .description("Shape the solver uses for the collider");
  b.add_input<decl::Float>("Friction"_ustr)
      .default_value(0.5f)
      .min(0.0f)
      .max(1.0f)
      .description(
          "Coulomb friction of the surface. The ground usually wants a high value "
          "(around 0.8), a slide a low one");
  {
    auto &panel = b.add_panel("Physics"_ustr).default_closed(false);
    panel.add_input<decl::Bool>("Sweep"_ustr)
        .default_value(true)
        .description(
            "Move the collider through the substeps of a frame instead of leaving it at the pose "
            "of the frame. Without this a fast collider (more than a particle radius per frame) "
            "can "
            "pass through everything between two frames. Requires Transform to be wired");
    panel.add_input<decl::Bool>("Deforming"_ustr)
        .default_value(false)
        .description(
            "The geometry itself changes shape from frame to frame (animated vertices, a "
            "mantaflow "
            "cache, ...). The solver then interpolates the mesh between the previous and the "
            "current frame for every substep. Expensive for dense meshes: it rebuilds the tree "
            "per "
            "substep");
    panel.add_input<decl::Bool>("Boundary"_ustr)
        .default_value(false)
        .description(
            "Invert the collider: points are kept *inside* it instead of being pushed out");
    panel.add_input<decl::Bool>("Edge Contacts"_ustr)
        .default_value(false)
        .description(
            "Also collide with the edges of the mesh, which closes the gaps left by point "
            "vs face contacts on a coarse mesh");
    panel.add_input<decl::Float>("Margin"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Extra distance around the surface, useful for very fast bodies");
    panel.add_input<decl::Float>("Softness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .max(1.0f)
        .description(
            "How soft the contact is; 0 is a hard constraint (stored as a compliance of "
            "softness^5, like the Collider asset)");
    panel.add_input<decl::Float>("Error Threshold"_ustr)
        .default_value(1e-3f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Contact distance that counts as an error, used for the reported residual");
    panel.add_input<decl::String>("Filter"_ustr)
        .description(
            "Only apply this collider to geometry with one of these tags (comma separated, empty "
            "applies to everything)");
  }
  b.add_output<decl::Bundle>("Collider"_ustr)
      .description(
          "Connect it to the Effectors input of a solver, or nest it in a bundle with "
          "Combine Bundle");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const int type = params.extract_input<int>("Type"_ustr);
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  const float4x4 transform = params.extract_input<float4x4>("Transform"_ustr);
  /* Applying the matrix here is what makes an animated object usable as a collider with a single
   * wire (see the socket description). The matrix is also written into the bundle, because the
   * solver needs it to interpolate the pose between two frames. */
  geometry::transform_geometry(geometry, transform);
  const float friction = std::clamp(params.extract_input<float>("Friction"_ustr), 0.0f, 1.0f);
  const bool sweep = params.extract_input<bool>("Sweep"_ustr);
  const bool deforming = params.extract_input<bool>("Deforming"_ustr);
  const bool is_boundary = params.extract_input<bool>("Boundary"_ustr);
  const bool use_edge_contacts = params.extract_input<bool>("Edge Contacts"_ustr);
  const float margin = std::max(0.0f, params.extract_input<float>("Margin"_ustr));
  const float softness = std::clamp(params.extract_input<float>("Softness"_ustr), 0.0f, 1.0f);
  const float error_threshold = std::max(0.0f,
                                         params.extract_input<float>("Error Threshold"_ustr));
  const std::string filter = params.extract_input<std::string>("Filter"_ustr);

  BundlePtr bundle_ptr = Bundle::create();
  Bundle &bundle = bundle_ptr.ensure_mutable_inplace();

  /* The "Type" item is what makes the bundle a typed collider bundle; the solvers find colliders
   * by looking for these names anywhere in the world bundle. */
  const bool use_box = type == int(ColliderType::Box);
  const bool use_convex = type == int(ColliderType::Convex);
  bundle.add(*BundleKey::from_str("Type"),
             std::string(use_box ? BoxColliderBundle::name : MeshColliderBundle::name));
  bundle.add(*BundleKey::from_str("geometry"), geometry);
  bundle.add(*BundleKey::from_str("friction"), friction);
  if (use_convex) {
    /* Same bundle type as the mesh one, the solver builds a convex hull instead of triangles. */
    bundle.add(*BundleKey::from_str("convex"), true);
  }

  /* Keys of the XPBD solver. It is the same asset interface as the `Collider` asset, so a scene
   * can switch between the two without rewiring. */
  bundle.add(*BundleKey::from_str("deforming"), deforming);
  bundle.add(*BundleKey::from_str("is_boundary"), is_boundary);
  bundle.add(*BundleKey::from_str("use_edge_contacts"), use_edge_contacts);
  bundle.add(*BundleKey::from_str("margin"), margin);
  /* Same mapping as the `Softness to Compliance` asset group. */
  bundle.add(*BundleKey::from_str("compliance"),
             softness * softness * softness * softness * softness);
  bundle.add(*BundleKey::from_str("error_threshold"), error_threshold);
  bundle.add(*BundleKey::from_str("filter"), filter);
  bundle.add(*BundleKey::from_str("transform"), transform);
  bundle.add(*BundleKey::from_str("sweep"), sweep);

  params.set_output("Collider"_ustr, std::move(bundle_ptr));
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  /* The uidname stays `GeometryNodeRBDCollider` so that existing scenes keep working, only the
   * name in the UI changed when the node became solver agnostic. */
  geo_node_type_base(&ntype, "GeometryNodeRBDCollider"_ustr);
  ntype.ui_name = "Dilapido Collider";
  ntype.ui_description =
      "Build a collider bundle for the rigid body and XPBD solvers (mesh, box or convex hull, "
      "with "
      "optional per substep sweeping)";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_180;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_dilapido_collider_cc
