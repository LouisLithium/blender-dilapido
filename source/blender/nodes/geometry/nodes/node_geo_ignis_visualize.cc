/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Ignis Visualize`: inspect one grid of an Ignis volume as a voxel-aligned slice.
 *
 * The node outputs two things:
 * - `Mesh`: a quad per voxel of the selected grid on one of the XY/XZ/YZ planes, spanning the
 *   grid's active bounding box. The grid value is stored on the face domain.
 * - `Field`: the grid sampled at the current position. The socket is float for scalar grids and
 *   vector for the velocity grid, so a `Viewer` node can color the generated mesh directly.
 *
 * The field output is what makes this a native node: a node group can not change the type of an
 * output socket from a menu.
 */

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>

#include "BLI_array.hh"
#include "BLI_index_mask.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_vector.hh"
#include "BLI_span.hh"
#include "BLI_string_ref.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_fields.hh"
#include "BKE_material.hh"
#include "BKE_mesh.hh"
#include "BKE_volume.hh"
#include "BKE_volume_grid.hh"
#include "BKE_volume_openvdb.hh"

#include "DNA_mesh_types.h"

#include "FN_field.hh"
#include "FN_multi_function.hh"

#include "NOD_rna_define.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "RNA_enum_types.hh"

#include "node_geometry_util.hh"

#ifdef WITH_OPENVDB
#  include <openvdb/openvdb.h>
#  include <openvdb/tools/Interpolation.h>
#endif

namespace blender::nodes::node_geo_ignis_visualize_cc {

namespace {

/** Grids that an Ignis volume can contain. Velocity is the only vector grid here. */
enum class Grid : int {
  Density = 0,
  Temperature,
  Fuel,
  Flame,
  Pressure,
  Divergence,
  Velocity,
  VelocityMagnitude,
};

/** Slice plane, named by the two axes that span the plane. */
enum class Plane : int {
  XY = 0,
  XZ,
  YZ,
};

const EnumPropertyItem grid_items[] = {
    {int(Grid::Density), "DENSITY", 0, N_("Density"), N_("Smoke density grid")},
    {int(Grid::Temperature), "TEMPERATURE", 0, N_("Temperature"), N_("Temperature grid")},
    {int(Grid::Fuel), "FUEL", 0, N_("Fuel"), N_("Unburned fuel grid")},
    {int(Grid::Flame), "FLAME", 0, N_("Flame"), N_("Reaction rate grid (K/s)")},
    {int(Grid::Pressure), "PRESSURE", 0, N_("Pressure"), N_("Projection pressure grid")},
    {int(Grid::Divergence), "DIVERGENCE", 0, N_("Divergence"), N_("Velocity divergence grid")},
    {int(Grid::Velocity), "VELOCITY", 0, N_("Velocity"), N_("Velocity vector grid")},
    {int(Grid::VelocityMagnitude),
     "VELOCITY_MAGNITUDE",
     0,
     N_("Velocity Magnitude"),
     N_("Length of the velocity grid (a scalar)")},
    {0, nullptr, 0, nullptr, nullptr},
};

const EnumPropertyItem plane_items[] = {
    {int(Plane::XY), "XY", 0, N_("XY"), N_("Slice along Z, spanning X and Y")},
    {int(Plane::XZ), "XZ", 0, N_("XZ"), N_("Slice along Y, spanning X and Z")},
    {int(Plane::YZ), "YZ", 0, N_("YZ"), N_("Slice along X, spanning Y and Z")},
    {0, nullptr, 0, nullptr, nullptr},
};

StringRefNull grid_name(const Grid grid)
{
  switch (grid) {
    case Grid::Density:
      return "density";
    case Grid::Temperature:
      return "temperature";
    case Grid::Fuel:
      return "fuel";
    case Grid::Flame:
      return "flame";
    case Grid::Pressure:
      return "pressure";
    case Grid::Divergence:
      return "divergence";
    case Grid::Velocity:
    case Grid::VelocityMagnitude:
      return "velocity";
  }
  return "density";
}

StringRefNull attribute_name(const Grid grid)
{
  return grid == Grid::VelocityMagnitude ? "velocity_magnitude" : grid_name(grid);
}

eNodeSocketDatatype grid_socket_type(const Grid grid)
{
  return grid == Grid::Velocity ? SOCK_VECTOR : SOCK_FLOAT;
}

/** Number of cells above which the slice is refused instead of allocating an enormous mesh. */
constexpr int64_t MAX_SLICE_CELLS = 16 * 1024 * 1024;

#ifdef WITH_OPENVDB

template<typename T>
void sample_grid(const bke::OpenvdbGridType<T> &grid,
                 const Span<float3> positions,
                 const IndexMask &mask,
                 MutableSpan<T> dst)
{
  using GridType = bke::OpenvdbGridType<T>;
  using GridValue = typename GridType::ValueType;
  using Traits = bke::VolumeGridTraits<T>;

  typename GridType::ConstUnsafeAccessor accessor = grid.getConstUnsafeAccessor();
  mask.foreach_index([&](const int64_t i) {
    const float3 &pos = positions[i];
    const openvdb::Vec3R world_pos(pos.x, pos.y, pos.z);
    const openvdb::Vec3R index_pos = grid.transform().worldToIndex(world_pos);
    GridValue value;
    if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int>) {
      openvdb::tools::PointSampler::sample(accessor, index_pos, value);
    }
    else {
      openvdb::tools::BoxSampler::sample(accessor, index_pos, value);
    }
    dst[i] = Traits::to_blender(value);
  });
}

/** Field implementation that samples the selected grid at the position of the evaluated element. */
class GridSampleFunction : public mf::MultiFunction {
 private:
  bke::GVolumeGrid grid_;
  mf::Signature signature_;
  VolumeGridType grid_type_;
  bool magnitude_;
  bke::VolumeTreeAccessToken tree_token_;
  const openvdb::GridBase *grid_base_ = nullptr;

 public:
  GridSampleFunction(bke::GVolumeGrid grid, const bool magnitude)
      : grid_(std::move(grid)), magnitude_(magnitude)
  {
    BLI_assert(grid_);
    grid_type_ = grid_->grid_type();

    const CPPType *output_type = &CPPType::get<float>();
    if (!magnitude_) {
      const std::optional<eNodeSocketDatatype> socket_type = bke::grid_type_to_socket_type(
          grid_type_);
      if (socket_type) {
        output_type = bke::socket_type_to_geo_nodes_base_cpp_type(*socket_type);
      }
    }

    mf::SignatureBuilder builder{"Ignis Visualize", signature_};
    builder.single_input<float3>("Position");
    builder.single_output("Field", *output_type);
    this->set_signature(&signature_);

    grid_base_ = &grid_->grid(tree_token_);
  }

  void call(const IndexMask &mask, mf::Params params, mf::Context /*context*/) const override
  {
    const VArraySpan<float3> positions = params.readonly_single_input<float3>(0, "Position");
    GMutableSpan dst = params.uninitialized_single_output(1, "Field");

    BKE_volume_grid_type_to_blender_value_type(grid_type_, [&]<typename T>() {
      if constexpr (std::is_same_v<T, float3>) {
        Array<float3> sampled(positions.size());
        sample_grid<float3>(static_cast<const bke::OpenvdbGridType<float3> &>(*grid_base_),
                            positions,
                            mask,
                            sampled.as_mutable_span());
        if (magnitude_) {
          MutableSpan<float> dst_float = dst.typed<float>();
          mask.foreach_index([&](const int64_t i) { dst_float[i] = math::length(sampled[i]); });
        }
        else {
          MutableSpan<float3> dst_vector = dst.typed<float3>();
          mask.foreach_index([&](const int64_t i) { dst_vector[i] = sampled[i]; });
        }
      }
      else if constexpr (is_same_any_v<T, float, int, bool>) {
        Array<T> sampled(positions.size());
        sample_grid<T>(static_cast<const bke::OpenvdbGridType<T> &>(*grid_base_),
                       positions,
                       mask,
                       sampled.as_mutable_span());
        MutableSpan<float> dst_float = dst.typed<float>();
        mask.foreach_index([&](const int64_t i) { dst_float[i] = float(sampled[i]); });
      }
    });
  }
};

#endif /* WITH_OPENVDB */

/** Checks that the grid really has the type implied by the selected menu entry. */
bool grid_type_matches(const Grid grid, const VolumeGridType type)
{
  if (ELEM(grid, Grid::Velocity, Grid::VelocityMagnitude)) {
    return type == VOLUME_GRID_VECTOR_FLOAT;
  }
  return type == VOLUME_GRID_FLOAT;
}

#ifdef WITH_OPENVDB
/**
 * Builds a voxel-aligned slice mesh. `factor` selects the plane position along the normal axis,
 * 0 = first voxel center and 1 = last voxel center.
 */
Mesh *build_slice_mesh(const openvdb::CoordBBox &bbox,
                       const float4x4 &grid_transform,
                       const Plane plane,
                       const float factor,
                       const StringRefNull name,
                       const GField &field)
{
  const int min[3] = {bbox.min().x(), bbox.min().y(), bbox.min().z()};
  const int max[3] = {bbox.max().x(), bbox.max().y(), bbox.max().z()};

  int u_axis = 0;
  int v_axis = 1;
  int n_axis = 2;
  const bool flip = plane == Plane::XZ;
  switch (plane) {
    case Plane::XY:
      break;
    case Plane::XZ:
      u_axis = 0;
      v_axis = 2;
      n_axis = 1;
      break;
    case Plane::YZ:
      u_axis = 1;
      v_axis = 2;
      n_axis = 0;
      break;
  }

  const int64_t cells_u = int64_t(max[u_axis]) - min[u_axis] + 1;
  const int64_t cells_v = int64_t(max[v_axis]) - min[v_axis] + 1;
  if (cells_u <= 0 || cells_v <= 0) {
    return nullptr;
  }
  if (cells_u * cells_v > MAX_SLICE_CELLS) {
    return nullptr;
  }

  const int verts_u = int(cells_u) + 1;
  const int verts_v = int(cells_v) + 1;
  const int verts_num = verts_u * verts_v;
  const int faces_num = int(cells_u * cells_v);
  const int corners_num = faces_num * 4;

  Mesh *mesh = bke::mesh_new_no_attributes(verts_num, 0, faces_num, corners_num);
  MutableSpan<float3> positions = mesh->vert_positions_for_write();

  const float normal_position = float(min[n_axis]) +
                                factor * float(max[n_axis] - min[n_axis]);
  int vert_index = 0;
  for (int j = 0; j < verts_v; j++) {
    for (int i = 0; i < verts_u; i++) {
      float3 index_pos;
      index_pos[u_axis] = float(min[u_axis]) - 0.5f + i;
      index_pos[v_axis] = float(min[v_axis]) - 0.5f + j;
      index_pos[n_axis] = normal_position;
      positions[vert_index++] = math::transform_point(grid_transform, index_pos);
    }
  }

  MutableSpan<int> face_offsets = mesh->face_offsets_for_write();
  MutableSpan<int> corner_verts = mesh->corner_verts_for_write();
  int face_index = 0;
  int corner_index = 0;
  for (int j = 0; j < int(cells_v); j++) {
    for (int i = 0; i < int(cells_u); i++) {
      const int v0 = j * verts_u + i;
      const int v1 = v0 + 1;
      const int v2 = v1 + verts_u;
      const int v3 = v0 + verts_u;
      face_offsets[face_index++] = corner_index;
      if (flip) {
        corner_verts[corner_index++] = v0;
        corner_verts[corner_index++] = v3;
        corner_verts[corner_index++] = v2;
        corner_verts[corner_index++] = v1;
      }
      else {
        corner_verts[corner_index++] = v0;
        corner_verts[corner_index++] = v1;
        corner_verts[corner_index++] = v2;
        corner_verts[corner_index++] = v3;
      }
    }
  }
  face_offsets[faces_num] = corners_num;

  bke::mesh_calc_edges(*mesh, false, false);
  BKE_id_material_eval_ensure_default_slot(reinterpret_cast<ID *>(mesh));

  /* Store the grid value on the cells too, so the mesh is useful in the spreadsheet even without
   * a Viewer node. The Viewer colors the mesh through the field output below. */
  GeometrySet geometry;
  geometry.replace_mesh(mesh);
  GeometryComponent &component = geometry.get_component_for_write(GeometryComponent::Type::Mesh);
  bke::try_capture_field_on_geometry(
      component, name, AttrDomain::Face, fn::Field<bool>(true), field);

  /* Transfer ownership back to the caller; the temporary geometry set must not free the mesh. */
  return geometry.get_component_for_write<bke::MeshComponent>().release();
}
#endif /* WITH_OPENVDB */

}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  const bNode *node = b.node_or_null();
  const Grid grid = node ? Grid(node->custom1) : Grid::Density;

  b.add_input<decl::Geometry>("Volume"_ustr)
      .supported_type(GeometryComponent::Type::Volume)
      .description("Volume from `Ignis Dynamics` (or any volume with Ignis grid names)");
  b.add_input<decl::Menu>("Plane"_ustr)
      .static_items(plane_items)
      .default_value(MenuValue(int(Plane::XY)))
      .optional_label()
      .description("Plane of the slice; the mesh spans the grid bounding box on the other axes");
  b.add_input<decl::Float>("Factor"_ustr)
      .default_value(0.5f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .description("Position of the slice along its normal: 0 is the first voxel, 1 the last");

  b.add_output<decl::Geometry>("Mesh"_ustr).description(
      "One quad per voxel of the selected grid on the slice plane, in the volume's object space");
  b.add_output(grid_socket_type(grid), "Field"_ustr)
      .structure_type(StructureType::Field)
      .description(
          "Grid sampled at the current position; connect it to a Viewer together with the mesh");
}

static void node_layout(ui::Layout &layout, bContext * /*C*/, PointerRNA *ptr)
{
  layout.use_property_split_set(true);
  layout.use_property_decorate_set(false);
  layout.prop(ptr, "grid", UI_ITEM_NONE, "", ICON_NONE);
}

static void node_init(bNodeTree * /*tree*/, bNode *node)
{
  node->custom1 = int(Grid::Density);
}

static void node_geo_exec(GeoNodeExecParams params)
{
#ifdef WITH_OPENVDB
  const Grid grid = Grid(params.node().custom1);
  const StringRefNull name = grid_name(grid);
  const StringRefNull attr_name = attribute_name(grid);

  const GeometrySet volume_geometry = params.extract_input<GeometrySet>("Volume"_ustr);
  const Volume *volume = volume_geometry.get_volume();
  if (!volume) {
    params.error_message_add(NodeWarningType::Warning,
                             "Ignis Visualize: the input has no volume");
    params.set_default_remaining_outputs();
    return;
  }

  const bke::VolumeGridData *grid_data = BKE_volume_grid_find(volume, name);
  if (!grid_data) {
    params.error_message_add(NodeWarningType::Warning,
                             std::string("Ignis Visualize: no '") + name +
                                 "' grid in the volume");
    params.set_default_remaining_outputs();
    return;
  }
  if (!grid_type_matches(grid, grid_data->grid_type())) {
    params.error_message_add(NodeWarningType::Warning,
                             std::string("Ignis Visualize: the '") + name +
                                 "' grid has an unexpected type");
    params.set_default_remaining_outputs();
    return;
  }

  const bool magnitude = grid == Grid::VelocityMagnitude;
  grid_data->add_user();
  bke::GVolumeGrid grid_handle(grid_data);

  const GField position_field = bke::AttributeFieldInput::get_field<float3, "position">();
  const auto sample_function = std::make_shared<GridSampleFunction>(grid_handle, magnitude);
  GField field(fn::FieldOperation::from(sample_function, {position_field}), 0);

  GeometrySet mesh_geometry;
  {
    bke::VolumeTreeAccessToken tree_token;
    const openvdb::CoordBBox bbox = grid_handle->active_bounds();
    if (!bbox.empty()) {
      const float4x4 transform = BKE_volume_transform_to_blender(grid_handle->transform());
      const float factor = std::clamp(params.extract_input<float>("Factor"_ustr), 0.0f, 1.0f);
      const Plane plane = Plane(params.extract_input<int>("Plane"_ustr));

      Mesh *mesh = build_slice_mesh(bbox, transform, plane, factor, attr_name, field);
      if (mesh != nullptr) {
        mesh_geometry.replace_mesh(mesh);
      }
      else {
        params.error_message_add(
            NodeWarningType::Warning,
            "Ignis Visualize: the slice is too large to build (or the grid is empty)");
      }
    }
  }

  params.set_output("Mesh"_ustr, std::move(mesh_geometry));
  params.set_output("Field"_ustr, std::move(field));
#else
  node_geo_exec_with_missing_openvdb(params);
#endif
}

static void node_rna(StructRNA *srna)
{
  RNA_def_node_enum(srna,
                    "grid",
                    "Grid",
                    "Ignis grid to visualize",
                    grid_items,
                    NOD_inline_enum_accessors(custom1),
                    int(Grid::Density));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeIgnisVisualize"_ustr);
  ntype.ui_name = "Ignis Visualize";
  ntype.ui_description =
      "Slice one grid of an Ignis volume into a voxel-aligned plane mesh and output the sampled "
      "field for a Viewer";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.draw_buttons = node_layout;
  ntype.initfunc = node_init;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);

  node_rna(ntype.rna_ext.srna);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_ignis_visualize_cc
