/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * RBD Collider node.
 *
 * Builds one collider bundle for the rigid body solver. The bundle is *typed*
 * (`Blender.Collider.Mesh` / `Blender.Collider.Box`), which is what the solver looks for, so any
 * number of colliders of any type can be nested into the world with `Combine Bundle` (or by
 * storing them under different paths) and all of them are used at the same time.
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

namespace blender::nodes::node_geo_rbd_collider_cc {

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
     N_("Collide with the triangles of the geometry. A mesh is a shell, so a body that ends up "
        "behind the surface can not be pushed back out")},
    {int(ColliderType::Box),
     "BOX",
     0,
     N_("Box"),
     N_("Collide with a solid box around the bounds of the geometry. A box has volume, so a body "
        "that sinks into it is pushed back out")},
    {int(ColliderType::Convex),
     "CONVEX",
     0,
     N_("Convex Hull"),
     N_("Collide with the solid convex hull of the geometry. Has volume like the box, but follows "
        "the shape instead of the bounds, and is much cheaper than the triangle mesh")},
    {0, nullptr, 0, nullptr, nullptr},
};

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr)
      .description("Collision geometry, in local space. Use Transform to place it");
  b.add_input<decl::Matrix>("Transform"_ustr)
      .description(
          "Transform applied to the geometry. Wire Object Info > Transform into it to make the "
          "collider follow a (animated) object: Object Info hands the geometry over in *local* "
          "space, so a collider built straight from it never moves");
  b.add_input<decl::Menu>("Type"_ustr)
      .static_items(collider_type_items)
      .default_value(MenuValue(int(ColliderType::Mesh)))
      .description("Shape the solver uses for the collider");
  b.add_input<decl::Float>("Friction"_ustr)
      .default_value(0.5f)
      .min(0.0f)
      .max(1.0f)
      .description("Coulomb friction of the surface. The ground usually wants a high value "
                   "(around 0.8), a slide a low one");
  b.add_output<decl::Bundle>("Collider"_ustr)
      .description("Connect it to the Effectors input of the solver, or nest it in a bundle with "
                   "Combine Bundle");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const int type = params.extract_input<int>("Type"_ustr);
  GeometrySet geometry = params.extract_input<GeometrySet>("Geometry"_ustr);
  const float4x4 transform = params.extract_input<float4x4>("Transform"_ustr);
  /* Applying the matrix here is what makes an animated object usable as a collider with a single
   * wire (see the socket description). */
  geometry::transform_geometry(geometry, transform);
  const float friction = std::clamp(params.extract_input<float>("Friction"_ustr), 0.0f, 1.0f);

  BundlePtr bundle_ptr = Bundle::create();
  Bundle &bundle = bundle_ptr.ensure_mutable_inplace();

  /* The "Type" item is what makes the bundle a typed collider bundle; the solver finds colliders by
   * looking for these names anywhere in the world bundle. */
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

  params.set_output("Collider"_ustr, std::move(bundle_ptr));
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeRBDCollider"_ustr);
  ntype.ui_name = "RBD Collider";
  ntype.ui_description = "Build a collider bundle for the rigid body solver";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_180;
  blender::bke::node_register_type(ntype);
}

NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_rbd_collider_cc
