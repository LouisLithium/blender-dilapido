/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * RBD Constraint Lines node.
 *
 * Turns the constraint point cloud into a thin tube per constraint, so the glue network can be seen
 * in the viewport (a point cloud with a couple of attributes is not very readable).
 *
 * The end points come from the `anchor_a_world` / `anchor_b_world` attributes, which the rigid body
 * solver writes every frame: the local `anchor_a` / `anchor_b` alone can not be turned into a line
 * without looking up the pose of both pieces, and those keep moving.
 *
 * The output is a *mesh* and not a curve on purpose: an object with a geometry nodes modifier takes
 * its mesh component, so a curve output of the modifier would simply not be drawn on a mesh object.
 *
 * Every face carries `enabled`, `type` and (when the constraints came from `Fracture Cluster`)
 * `cluster_a` / `cluster_b`, and the points carry an RGB `rbd_color` (green = holding, orange = soft,
 * red = broken), so a material can colour the lines.
 */

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"
#include "BKE_pointcloud.hh"

#include "BLI_array.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_rbd_constraint_lines_cc {

/** Number of sides of the tube around one constraint. */
static constexpr int k_sides = 4;

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Constraints"_ustr)
      .supported_type(GeometryComponent::Type::PointCloud)
      .description("The Constraints of RBD Constraint from Pieces or Fracture Cluster, e.g. the "
                   "Constraints output of the Rigid Body Dynamics asset");
  b.add_input<decl::Float>("Thickness"_ustr)
      .default_value(0.01f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description("Radius of the tube that is drawn for every constraint");
  b.add_input<decl::Bool>("Show Disabled"_ustr)
      .default_value(false)
      .description("Also draw the constraints that already broke (off by default: a broken "
                   "constraint is gone");
  b.add_input<decl::Bool>("Show Soft"_ustr)
      .default_value(true)
      .description("Draw soft (spring) constraints");

  b.add_output<decl::Geometry>("Lines"_ustr)
      .description("One thin tube per drawn constraint, in world space");
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const GeometrySet constraints = params.extract_input<GeometrySet>("Constraints"_ustr);
  const float thickness = std::max(params.extract_input<float>("Thickness"_ustr), 0.0f);
  const bool show_disabled = params.extract_input<bool>("Show Disabled"_ustr);
  const bool show_soft = params.extract_input<bool>("Show Soft"_ustr);

  const PointCloud *pointcloud = constraints.get_pointcloud();
  if (pointcloud == nullptr || pointcloud->totpoint == 0) {
    params.set_default_remaining_outputs();
    return;
  }

  const bke::AttributeAccessor attributes = pointcloud->attributes();
  if (!attributes.contains("anchor_a_world") || !attributes.contains("anchor_b_world")) {
    params.error_message_add(NodeWarningType::Info,
                             "The constraints have no `anchor_a_world` / `anchor_b_world` "
                             "attributes. Connect the Constraints that come out of Rigid Body "
                             "Dynamics (the solver writes them every frame)");
    params.set_default_remaining_outputs();
    return;
  }

  const VArray<float3> anchors_a = attributes
                                       .lookup_or_default<float3>(
                                           "anchor_a_world", AttrDomain::Point, float3(0.0f))
                                       .varray;
  const VArray<float3> anchors_b = attributes
                                       .lookup_or_default<float3>(
                                           "anchor_b_world", AttrDomain::Point, float3(0.0f))
                                       .varray;
  /* The two anchors of a constraint are the *same* point (the joint), so the line is drawn between
   * the centers of the two bodies instead. That is what `body_a_world` / `body_b_world` are for. */
  const bool has_bodies = attributes.contains("body_a_world") && attributes.contains("body_b_world");
  const VArray<float3> bodies_a = has_bodies ? attributes
                                                   .lookup_or_default<float3>(
                                                       "body_a_world", AttrDomain::Point, float3(0.0f))
                                                   .varray :
                                               anchors_a;
  const VArray<float3> bodies_b = has_bodies ? attributes
                                                   .lookup_or_default<float3>(
                                                       "body_b_world", AttrDomain::Point, float3(0.0f))
                                                   .varray :
                                               anchors_b;
  const VArray<bool> enableds = attributes
                                    .lookup_or_default<bool>("enabled", AttrDomain::Point, true)
                                    .varray;
  const VArray<int> types = attributes.lookup_or_default<int>("type", AttrDomain::Point, 0).varray;
  const VArray<int> clusters_a = attributes
                                     .lookup_or_default<int>("cluster_a", AttrDomain::Point, -1)
                                     .varray;
  const VArray<int> clusters_b = attributes
                                     .lookup_or_default<int>("cluster_b", AttrDomain::Point, -1)
                                     .varray;
  const bool has_clusters = attributes.contains("cluster_a") && attributes.contains("cluster_b");

  /* Which constraints are drawn, so the geometry can be sized exactly. A constraint whose two
   * anchors coincide has no direction to extrude along, so it is dropped here: leaving it in would
   * produce a degenerate face (all corners the same vertex), and those are not valid geometry (the
   * face normals of such a mesh crash the viewport). */
  const int constraints_num = pointcloud->totpoint;
  Array<int> line_of_constraint(constraints_num, -1);
  int lines_num = 0;
  for (const int i : IndexRange(constraints_num)) {
    if (math::length(bodies_b[i] - bodies_a[i]) <= 1.0e-6f) {
      continue;
    }
    if (!show_disabled && !enableds[i]) {
      continue;
    }
    if (!show_soft && types[i] == 1) {
      continue;
    }
    line_of_constraint[i] = lines_num++;
  }
  if (lines_num == 0) {
    params.set_default_remaining_outputs();
    return;
  }

  const int verts_num = lines_num * k_sides * 2;
  const int faces_num = lines_num * k_sides;
  const int corners_num = faces_num * 4;

  Array<float3> positions(verts_num);
  Array<int> corner_verts(corners_num);
  Array<int> face_offsets(faces_num + 1);
  /* Every face is a quad. */
  for (const int i : IndexRange(faces_num + 1)) {
    face_offsets[i] = i * 4;
  }

  /* The sides of the tube, built directly: cheap, and independent of the curve geometry. */
  for (const int i : IndexRange(constraints_num)) {
    const int line = line_of_constraint[i];
    if (line < 0) {
      continue;
    }
    const float3 a = bodies_a[i];
    const float3 b = bodies_b[i];
    const float3 direction = b - a;
    const float3 axis = math::normalize(direction);
    /* Two perpendicular axes to the segment. */
    const float3 helper = std::abs(axis.z) < 0.9f ? float3(0.0f, 0.0f, 1.0f) :
                                                    float3(1.0f, 0.0f, 0.0f);
    const float3 u = math::normalize(math::cross(axis, helper));
    const float3 v = math::cross(axis, u);
    const int vert_base = line * k_sides * 2;
    for (const int side : IndexRange(k_sides)) {
      const float angle = 2.0f * float(M_PI) * float(side) / float(k_sides);
      const float3 offset = (u * std::cos(angle) + v * std::sin(angle)) * thickness;
      positions[vert_base + side] = a + offset;
      positions[vert_base + k_sides + side] = b + offset;
    }
    const int face_base = line * k_sides;
    for (const int side : IndexRange(k_sides)) {
      const int next = (side + 1) % k_sides;
      const int corner = face_base * 4 + side * 4;
      corner_verts[corner + 0] = vert_base + side;
      corner_verts[corner + 1] = vert_base + next;
      corner_verts[corner + 2] = vert_base + k_sides + next;
      corner_verts[corner + 3] = vert_base + k_sides + side;
    }
  }

  /* Edges are stored explicitly in this version (`edges_for_write`), they are not derived from the
   * corners, so they have to be built here as well. */
  Vector<int2> edge_verts;
  Map<uint64_t, int> edge_of_key;
  Array<int> corner_edges(corners_num);
  for (const int face : IndexRange(faces_num)) {
    for (const int i : IndexRange(4)) {
      const int va = corner_verts[face * 4 + i];
      const int vb = corner_verts[face * 4 + (i + 1) % 4];
      const uint64_t key = (uint64_t(uint32_t(std::min(va, vb))) << 32) |
                           uint64_t(uint32_t(std::max(va, vb)));
      const int edge = edge_of_key.lookup_or_add(key, int(edge_verts.size()));
      if (edge == edge_verts.size()) {
        edge_verts.append(int2(va, vb));
      }
      corner_edges[face * 4 + i] = edge;
    }
  }

  Mesh *mesh = BKE_mesh_new_nomain(verts_num, int(edge_verts.size()), faces_num, corners_num);
  mesh->vert_positions_for_write().copy_from(positions);
  mesh->edges_for_write().copy_from(edge_verts);
  mesh->corner_verts_for_write().copy_from(corner_verts);
  mesh->corner_edges_for_write().copy_from(corner_edges);
  mesh->face_offsets_for_write().copy_from(face_offsets);

  /* Attributes: the state per line (face domain), the colour per point. */
  bke::MutableAttributeAccessor mesh_attributes = mesh->attributes_for_write();
  bke::SpanAttributeWriter<bool> out_enabled = mesh_attributes.lookup_or_add_for_write_span<bool>(
      "enabled", AttrDomain::Face);
  bke::SpanAttributeWriter<int> out_type = mesh_attributes.lookup_or_add_for_write_span<int>(
      "type", AttrDomain::Face);
  bke::SpanAttributeWriter<float3> out_color = mesh_attributes.lookup_or_add_for_write_span<float3>(
      "rbd_color", AttrDomain::Point);
  bke::SpanAttributeWriter<int> out_cluster_a =
      has_clusters ? mesh_attributes.lookup_or_add_for_write_span<int>("cluster_a", AttrDomain::Face) :
                     bke::SpanAttributeWriter<int>();
  bke::SpanAttributeWriter<int> out_cluster_b =
      has_clusters ? mesh_attributes.lookup_or_add_for_write_span<int>("cluster_b", AttrDomain::Face) :
                     bke::SpanAttributeWriter<int>();

  for (const int i : IndexRange(constraints_num)) {
    const int line = line_of_constraint[i];
    if (line < 0) {
      continue;
    }
    const bool is_enabled = enableds[i];
    const bool is_soft = types[i] == 1;
    float3 color;
    if (!is_enabled) {
      color = float3(0.9f, 0.1f, 0.1f);
    }
    else if (is_soft) {
      color = float3(0.95f, 0.7f, 0.1f);
    }
    else {
      color = float3(0.15f, 0.85f, 0.25f);
    }
    for (const int side : IndexRange(k_sides)) {
      out_enabled.span[line * k_sides + side] = is_enabled;
      out_type.span[line * k_sides + side] = types[i];
      if (has_clusters) {
        out_cluster_a.span[line * k_sides + side] = clusters_a[i];
        out_cluster_b.span[line * k_sides + side] = clusters_b[i];
      }
    }
    for (const int vertex : IndexRange(k_sides * 2)) {
      out_color.span[line * k_sides * 2 + vertex] = color;
    }
  }

  out_enabled.finish();
  out_type.finish();
  out_color.finish();
  if (has_clusters) {
    out_cluster_a.finish();
    out_cluster_b.finish();
  }

  GeometrySet result;
  result.replace_mesh(mesh);
  params.set_output("Lines"_ustr, std::move(result));
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeRBDConstraintLines"_ustr);
  ntype.ui_name = "RBD Constraint Lines";
  ntype.ui_description = "Draw the constraints as thin tubes, for previewing the glue network";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_180;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_rbd_constraint_lines_cc
