/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo_nodes
 *
 * `Tetrahedralize` fills a closed surface mesh with tetrahedra. The result is meant for a soft body
 * simulation, where the elements are what a volume constraint needs (the surface alone only gives
 * you cloth), but it is useful anywhere the volume has to be broken into cells: it is also what
 * turns a mesh into a shell that can be deformed, frozen or re-meshed.
 *
 * `Mesh` is the tetrahedral mesh and its surface in one geometry. Blender has no element primitive,
 * so every tetrahedron is a face with four corners (its four vertices, wound so that its volume is
 * positive) and the `rest_volume` face attribute holds the rest volume of that element; the
 * boundary of the element set follows as three cornered faces, so the same mesh renders and collides
 * as a closed surface. That is the whole structure in the mesh data - no element list hidden in a
 * separate geometry - and the corner count is what tells an element from a boundary triangle.
 */

#include <algorithm>
#include <string>

#include "BLI_array.hh"

#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_offset_indices.hh"

#include "BLI_span.hh"
#include "BLI_task.hh"


#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_mesh.hh"

#include "BLT_translation.hh"

#include "GEO_tetrahedralize.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_tetrahedralize_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Mesh"_ustr)
      .supported_type(GeometryComponent::Type::Mesh)
      .is_default_link_socket()
      .description("Closed surface mesh to fill with tetrahedra");
  b.add_input<decl::Float>("Voxel Size"_ustr)
      .default_value(0.05f)
      .min(1e-4f)
      .subtype(PROP_DISTANCE)
      .description(
          "Edge length of the lattice the tetrahedra are built on. Smaller values follow the "
          "surface more closely and create more elements (the cost grows with the cube of the "
          "resolution)");
  b.add_input<decl::Bool>("Snap to Surface"_ustr)
      .default_value(true)
      .description(
          "Move the lattice nodes that are close to the surface onto it, so the boundary follows "
          "the input instead of the staircase of the lattice. Turning this off gives an exact "
          "lattice, which is slightly smaller than the input and visibly blocky");
  b.add_output<decl::Geometry>("Mesh"_ustr).description(
      "The tetrahedral mesh: one four cornered face per tetrahedron (its corners are the four "
      "vertices of the element, its rest volume is in the `rest_volume` face attribute) plus the "
      "boundary triangles, so the same mesh is also the closed surface that can be rendered and "
      "collided with");
}

static Mesh *create_tet_mesh(const geometry::TetrahedralizedMesh &tet_mesh)
{
  const int elements_num = int(tet_mesh.tets.size());
  const int triangles_num = int(tet_mesh.boundary_tris.size());
  const Span<float3> positions = tet_mesh.vert_positions;

  /* The boundary triangles are all there, so the surface of the mesh is complete. An element face
   * covers two of the four faces of its tetrahedron, but with the winding that gives the element a
   * positive volume those two face *into* the mesh: they are the interior structure, not a second
   * layer of the surface. */

  const int faces_num = elements_num + triangles_num;
  Mesh *mesh = BKE_mesh_new_nomain(tet_mesh.vert_positions.size(),
                                   0,
                                   faces_num,
                                   elements_num * 4 + triangles_num * 3);
  mesh->vert_positions_for_write().copy_from(positions);
  if (faces_num == 0) {
    return mesh;
  }

  MutableSpan<int> face_offsets = mesh->face_offsets_for_write();
  offset_indices::fill_constant_group_size(4, 0, face_offsets.take_front(elements_num + 1));
  if (triangles_num > 0) {
    offset_indices::fill_constant_group_size(
        3, elements_num * 4, face_offsets.drop_front(elements_num));
  }

  MutableSpan<int> corner_verts = mesh->corner_verts_for_write();
  threading::parallel_for(IndexRange(elements_num), 4096, [&](const IndexRange range) {
    for (const int i : range) {
      const int4 &element = tet_mesh.tets[i];
      corner_verts[i * 4 + 0] = element.x;
      corner_verts[i * 4 + 1] = element.y;
      corner_verts[i * 4 + 2] = element.z;
      corner_verts[i * 4 + 3] = element.w;
    }
  });
  threading::parallel_for(IndexRange(triangles_num), 4096, [&](const IndexRange range) {
    for (const int i : range) {
      const int3 &tri = tet_mesh.boundary_tris[i];
      const int corner = elements_num * 4 + i * 3;
      corner_verts[corner + 0] = tri.x;
      corner_verts[corner + 1] = tri.y;
      corner_verts[corner + 2] = tri.z;
    }
  });
  bke::mesh_calc_edges(*mesh, false, false);

  bke::MutableAttributeAccessor attributes = mesh->attributes_for_write();
  bke::SpanAttributeWriter<float> rest_volume = attributes.lookup_or_add_for_write_span<float>(
      "rest_volume", AttrDomain::Face);
  rest_volume.span.take_front(elements_num).copy_from(tet_mesh.tet_volumes.as_span());
  rest_volume.span.drop_front(elements_num).fill(0.0f);
  rest_volume.finish();

  /* Flat shading: the normals of a volume mesh have no meaning to interpolate (an element face is
   * surrounded by the faces of the neighbouring elements, which are inside the mesh), and averaging
   * them gives dark patches wherever an element face is part of the surface. `Set Shade Smooth` is
   * one node away for anyone who wants it. */
  bke::SpanAttributeWriter<bool> sharp_face = attributes.lookup_or_add_for_write_span<bool>(
      "sharp_face", AttrDomain::Face);
  sharp_face.span.fill(true);
  sharp_face.finish();
  return mesh;
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const GeometrySet geometry_set = params.extract_input<GeometrySet>("Mesh"_ustr);
  const Mesh *mesh = geometry_set.get_mesh();
  if (mesh == nullptr) {
    params.error_message_add(NodeWarningType::Error, TIP_("The input mesh is empty"));
    params.set_default_remaining_outputs();
    return;
  }

  geometry::TetrahedralizeSettings settings;
  settings.voxel_size = std::max(params.extract_input<float>("Voxel Size"_ustr), 1e-4f);
  settings.snap_to_surface = params.extract_input<bool>("Snap to Surface"_ustr);

  geometry::TetrahedralizedMesh tet_mesh;
  std::string error;
  if (!geometry::tetrahedralize_mesh(*mesh, settings, tet_mesh, error)) {
    params.error_message_add(NodeWarningType::Error, StringRef(error));
    params.set_default_remaining_outputs();
    return;
  }

  if (tet_mesh.is_empty()) {
    params.error_message_add(
        NodeWarningType::Warning,
        TIP_("No tetrahedra were created. The mesh has to be closed (it needs a volume) and "
             "thicker than the voxel size"));
    params.set_default_remaining_outputs();
    return;
  }

  GeometrySet mesh_set;
  mesh_set.replace_mesh(create_tet_mesh(tet_mesh));
  params.set_output("Mesh"_ustr, std::move(mesh_set));
}

static void node_label(const bNodeTree * /*ntree*/,
                       const bNode * /*node*/,
                       char *label,
                       const int label_maxncpy)
{
  BLI_strncpy_utf8(
      label, CTX_IFACE_(BLT_I18NCONTEXT_ID_NODETREE, "Tetrahedralize"), label_maxncpy);
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeTetrahedralize"_ustr);
  ntype.ui_name = "Tetrahedralize";
  ntype.ui_description =
      "Fill a closed mesh with tetrahedra, one four cornered face and a rest volume per element";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_tetrahedralize_cc