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
 * - `Mesh` is the boundary of the tetrahedral mesh, a watertight triangle mesh that follows the
 *   input surface. Use it for rendering and for collision with other objects.
 * - `Tetrahedra` is one point per element, at its center, carrying the four vertex indices (`v0` to
 *   `v3`, into the `Mesh` output) and the rest volume (`rest_volume`). The positions of the vertices
 *   are deliberately not duplicated: the elements refer to the mesh vertices, so a deformation
 *   moves one set of positions.
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
#include "BKE_pointcloud.hh"

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
      "Boundary of the tetrahedral mesh: a watertight triangle mesh with the vertices the "
      "tetrahedra refer to");
  b.add_output<decl::Geometry>("Tetrahedra"_ustr).description(
      "One point per tetrahedron, at its center, with the four vertex indices `v0` to `v3` and the "
      "rest volume `rest_volume`");
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

static PointCloud *create_tet_pointcloud(const geometry::TetrahedralizedMesh &tet_mesh)
{
  PointCloud *pointcloud = bke::pointcloud_new_no_attributes(tet_mesh.tets.size());
  bke::MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
  bke::SpanAttributeWriter<float3> positions = attributes.lookup_or_add_for_write_span<float3>(
      "position", AttrDomain::Point);
  bke::SpanAttributeWriter<int> v0 = attributes.lookup_or_add_for_write_span<int>("v0",
                                                                                 AttrDomain::Point);
  bke::SpanAttributeWriter<int> v1 = attributes.lookup_or_add_for_write_span<int>("v1",
                                                                                 AttrDomain::Point);
  bke::SpanAttributeWriter<int> v2 = attributes.lookup_or_add_for_write_span<int>("v2",
                                                                                 AttrDomain::Point);
  bke::SpanAttributeWriter<int> v3 = attributes.lookup_or_add_for_write_span<int>("v3",
                                                                                 AttrDomain::Point);
  bke::SpanAttributeWriter<float> rest_volume = attributes.lookup_or_add_for_write_span<float>(
      "rest_volume", AttrDomain::Point);

  const Span<float3> vert_positions = tet_mesh.vert_positions;
  threading::parallel_for(tet_mesh.tets.index_range(), 4096, [&](const IndexRange range) {
    for (const int i : range) {
      const int4 &tet = tet_mesh.tets[i];
      const float3 a = vert_positions[tet.x];
      const float3 b = vert_positions[tet.y];
      const float3 c = vert_positions[tet.z];
      const float3 d = vert_positions[tet.w];
      positions.span[i] = (a + b + c + d) * 0.25f;
      v0.span[i] = tet.x;
      v1.span[i] = tet.y;
      v2.span[i] = tet.z;
      v3.span[i] = tet.w;
      rest_volume.span[i] = tet_mesh.tet_volumes[i];
    }
  });

  positions.finish();
  v0.finish();
  v1.finish();
  v2.finish();
  v3.finish();
  rest_volume.finish();
  return pointcloud;
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
  mesh_set.replace_mesh(create_surface_mesh(tet_mesh));
  params.set_output("Mesh"_ustr, std::move(mesh_set));

  GeometrySet tets_set;
  tets_set.replace_pointcloud(create_tet_pointcloud(tet_mesh));
  params.set_output("Tetrahedra"_ustr, std::move(tets_set));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeTetrahedralize"_ustr);
  ntype.ui_name = "Tetrahedralize";
  ntype.ui_description =
      "Fill a closed mesh with tetrahedra, with the four vertex indices and rest volume of every "
      "element";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_tetrahedralize_cc
