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
 * The output is split in two:
 *
 * - `Mesh` is the tetrahedral mesh itself. Blender has no element primitive, so every tetrahedron is
 *   a face with four corners (its four vertices, wound so that its volume is positive) and the
 *   `rest_volume` face attribute holds the rest volume of that element. That is the whole interior
 *   structure in the mesh data: no element list hidden in a separate geometry, and a solver reads
 *   the elements straight out of the corners of the faces.
 * - `Surface` is the boundary of that mesh, watertight, for rendering and collision.
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
      "The tetrahedral mesh itself: one four cornered face per tetrahedron, its corners being its "
      "four vertices, and the rest volume of the element in the `rest_volume` face attribute");
  b.add_output<decl::Geometry>("Surface"_ustr).description(
      "Boundary of the tetrahedral mesh, a watertight triangle mesh for rendering and collision");
}

static Mesh *create_tet_mesh(const geometry::TetrahedralizedMesh &tet_mesh)
{
  const int tets_num = int(tet_mesh.tets.size());
  Mesh *mesh = BKE_mesh_new_nomain(tet_mesh.vert_positions.size(), 0, tets_num, tets_num * 4);
  mesh->vert_positions_for_write().copy_from(tet_mesh.vert_positions);
  if (tets_num == 0) {
    return mesh;
  }

  MutableSpan<int> face_offsets = mesh->face_offsets_for_write();
  offset_indices::fill_constant_group_size(4, 0, face_offsets);
  MutableSpan<int> corner_verts = mesh->corner_verts_for_write();
  threading::parallel_for(IndexRange(tets_num), 4096, [&](const IndexRange range) {
    for (const int i : range) {
      const int4 &tet = tet_mesh.tets[i];
      corner_verts[i * 4 + 0] = tet.x;
      corner_verts[i * 4 + 1] = tet.y;
      corner_verts[i * 4 + 2] = tet.z;
      corner_verts[i * 4 + 3] = tet.w;
    }
  });
  bke::mesh_calc_edges(*mesh, false, false);

  bke::MutableAttributeAccessor attributes = mesh->attributes_for_write();
  bke::SpanAttributeWriter<float> rest_volume = attributes.lookup_or_add_for_write_span<float>(
      "rest_volume", AttrDomain::Face);
  rest_volume.span.copy_from(tet_mesh.tet_volumes.as_span());
  rest_volume.finish();
  return mesh;
}

static Mesh *create_surface_mesh(const geometry::TetrahedralizedMesh &tet_mesh)
{
  Mesh *mesh = BKE_mesh_new_nomain(
      tet_mesh.vert_positions.size(), 0, tet_mesh.boundary_tris.size(), tet_mesh.boundary_tris.size() * 3);
  mesh->vert_positions_for_write().copy_from(tet_mesh.vert_positions);
  if (!tet_mesh.boundary_tris.is_empty()) {
    MutableSpan<int> face_offsets = mesh->face_offsets_for_write();
    offset_indices::fill_constant_group_size(3, 0, face_offsets);
    MutableSpan<int> corner_verts = mesh->corner_verts_for_write();
    threading::parallel_for(tet_mesh.boundary_tris.index_range(), 4096, [&](const IndexRange range) {
      for (const int i : range) {
        const int3 &tri = tet_mesh.boundary_tris[i];
        corner_verts[i * 3 + 0] = tri.x;
        corner_verts[i * 3 + 1] = tri.y;
        corner_verts[i * 3 + 2] = tri.z;
      }
    });
    bke::mesh_calc_edges(*mesh, false, false);
  }
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
  }

  GeometrySet mesh_set;
  mesh_set.replace_mesh(create_tet_mesh(tet_mesh));
  params.set_output("Mesh"_ustr, std::move(mesh_set));

  GeometrySet surface_set;
  surface_set.replace_mesh(create_surface_mesh(tet_mesh));
  params.set_output("Surface"_ustr, std::move(surface_set));
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
