/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Downsample Grid`: reduce a volume grid to a coarser resolution.
 *
 * Each output voxel is the average of `Factor^3` input voxels (the background counts too, so the
 * average of a region is preserved instead of point sampling it). That is what makes this useful
 * for rendering: the total amount of, say, density or heat in a region stays the same, only the
 * sampling gets cheaper.
 */

#include <algorithm>
#include <cmath>

#include "BLI_array.hh"
#include "BLI_math_base.hh"
#include "BLI_span.hh"
#include "BLI_string_ref.hh"
#include "BLI_task.hh"

#include "BKE_volume_enums.hh"
#include "BKE_volume_grid.hh"
#include "BKE_volume_grid_process.hh"

#include "NOD_rna_define.hh"
#include "NOD_socket_search_link.hh"

#include "RNA_enum_types.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "node_geometry_util.hh"

#ifdef WITH_OPENVDB
#  include <openvdb/openvdb.h>
#endif

namespace blender::nodes::node_geo_grid_downsample_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();
  b.add_default_layout();
  const bNode *node = b.node_or_null();
  if (!node) {
    return;
  }
  const eNodeSocketDatatype data_type = eNodeSocketDatatype(node->custom1);
  b.add_input(data_type, "Grid"_ustr)
      .hide_value()
      .structure_type(StructureType::Grid)
      .description("Grid to sample at a lower resolution");
  b.add_input<decl::Int>("Factor"_ustr)
      .default_value(2)
      .min(1)
      .max(16)
      .description(
          "How many voxels along each axis become one output voxel. Only integer factors are "
          "supported");
  b.add_output(data_type, "Grid"_ustr)
      .structure_type(StructureType::Grid)
      .align_with_previous()
      .description("The downsampled grid, or the input grid when the factor is 1");
}

static void node_layout(ui::Layout &layout, bContext * /*C*/, PointerRNA *ptr)
{
  layout.prop(ptr, "data_type", UI_ITEM_NONE, "", ICON_NONE);
}

static std::optional<eNodeSocketDatatype> node_type_for_socket_type(const bNodeSocket &socket)
{
  switch (socket.type) {
    case SOCK_FLOAT:
      return SOCK_FLOAT;
    case SOCK_BOOLEAN:
      return SOCK_BOOLEAN;
    case SOCK_INT:
      return SOCK_INT;
    case SOCK_VECTOR:
    case SOCK_RGBA:
      return SOCK_VECTOR;
    default:
      return std::nullopt;
  }
}

static void node_gather_link_search_ops(GatherLinkSearchOpParams &params)
{
  const std::optional<eNodeSocketDatatype> data_type = node_type_for_socket_type(
      params.other_socket());
  if (!data_type) {
    return;
  }
  params.add_item(IFACE_("Grid"), [data_type](LinkSearchOpParams &params) {
    bNode &node = params.add_node("GeometryNodeGridDownsample"_ustr);
    node.custom1 = *data_type;
    params.update_and_connect_available_socket(node, "Grid"_ustr);
  });
}

#ifdef WITH_OPENVDB

static openvdb::Coord cell_of(const int64_t index, const openvdb::Coord &size)
{
  const int64_t x = index % size.x();
  const int64_t y = (index / size.x()) % size.y();
  const int64_t z = index / (int64_t(size.x()) * int64_t(size.y()));
  return openvdb::Coord(int(x), int(y), int(z));
}

template<typename GridT> static bool value_is_zero(const typename GridT::ValueType &value);

template<> bool value_is_zero<openvdb::FloatGrid>(const float &value)
{
  return value == 0.0f;
}

template<> bool value_is_zero<openvdb::Vec3fGrid>(const openvdb::Vec3f &value)
{
  return value.lengthSqr() == 0.0f;
}

template<> bool value_is_zero<openvdb::Vec4fGrid>(const openvdb::Vec4f &value)
{
  return value.lengthSqr() == 0.0f;
}

template<> bool value_is_zero<openvdb::Int32Grid>(const int &value)
{
  return value == 0;
}

/** Box average over `factor^3` voxels on a grid that is `factor` times coarser. */
template<typename GridT>
static typename GridT::Ptr downsample_box(const GridT &src, const int factor)
{
  using ValueT = typename GridT::ValueType;
  const openvdb::CoordBBox bbox = src.evalActiveVoxelBoundingBox();
  if (bbox.empty()) {
    return nullptr;
  }
  /* The blocks start at the active bounding box minimum, so the coarse grid covers the same region
   * without drifting towards +XYZ when the minimum is not a multiple of the factor. */
  const openvdb::Coord min = bbox.min();
  const openvdb::Coord extent = bbox.dim();
  const int cells_x = (extent.x() + factor - 1) / factor;
  const int cells_y = (extent.y() + factor - 1) / factor;
  const int cells_z = (extent.z() + factor - 1) / factor;
  const openvdb::Coord size(cells_x, cells_y, cells_z);
  const int64_t cells = int64_t(cells_x) * int64_t(cells_y) * int64_t(cells_z);
  if (cells <= 0) {
    return nullptr;
  }

  /* One coarse voxel covers `factor` source voxels: scale the index space, then place voxel (0,0,0)
   * at the center of the first block so the volume does not shift by half a coarse voxel. */
  const double half = 0.5 * double(factor - 1);
  const openvdb::Vec3d center = src.transform().indexToWorld(
      openvdb::Vec3d(min.x() + half, min.y() + half, min.z() + half));
  openvdb::math::Transform::Ptr transform = src.transform().copy();
  transform->preScale(double(factor));
  transform->postTranslate(center - transform->indexToWorld(openvdb::Vec3d(0.0)));

  typename GridT::Ptr dst = GridT::create(ValueT(0));
  dst->insertMeta(src);
  dst->setTransform(transform);

  const int64_t block = int64_t(factor) * int64_t(factor) * int64_t(factor);
  const float inv = 1.0f / float(block);

  if constexpr (std::is_same_v<ValueT, bool>) {
    /* A boolean grid keeps the majority of the block. */
    Array<int> counts(cells, 0);
    threading::parallel_for(IndexRange(cells), 2048, [&](const IndexRange range) {
      const typename GridT::ConstAccessor accessor = src.getConstAccessor();
      for (const int64_t i : range) {
        const openvdb::Coord cell = cell_of(i, size);
        const openvdb::Coord base(
            min.x() + cell.x() * factor, min.y() + cell.y() * factor, min.z() + cell.z() * factor);
        int count = 0;
        for (int dz = 0; dz < factor; dz++) {
          for (int dy = 0; dy < factor; dy++) {
            for (int dx = 0; dx < factor; dx++) {
              count += accessor.getValue(
                           openvdb::Coord(base.x() + dx, base.y() + dy, base.z() + dz)) ?
                           1 :
                           0;
            }
          }
        }
        counts[i] = count;
      }
    });
    typename GridT::Accessor writer = dst->getAccessor();
    for (const int64_t i : IndexRange(cells)) {
      if (counts[i] * 2 < int(block)) {
        continue;
      }
      writer.setValue(cell_of(i, size), true);
    }
  }
  else {
    Array<ValueT> values(cells, ValueT(0));
    threading::parallel_for(IndexRange(cells), 2048, [&](const IndexRange range) {
      /* One accessor per range: an OpenVDB accessor caches leaf pointers and is not thread safe. */
      const typename GridT::ConstAccessor accessor = src.getConstAccessor();
      for (const int64_t i : range) {
        const openvdb::Coord cell = cell_of(i, size);
        const openvdb::Coord base(
            min.x() + cell.x() * factor, min.y() + cell.y() * factor, min.z() + cell.z() * factor);
        ValueT sum = ValueT(0);
        for (int dz = 0; dz < factor; dz++) {
          for (int dy = 0; dy < factor; dy++) {
            for (int dx = 0; dx < factor; dx++) {
              sum += accessor.getValue(
                  openvdb::Coord(base.x() + dx, base.y() + dy, base.z() + dz));
            }
          }
        }
        values[i] = sum * inv;
      }
    });
    /* Serial write back: setting values in parallel can race on the shared active mask of a leaf
     * (see the Ignis determinism fix). */
    typename GridT::Accessor writer = dst->getAccessor();
    for (const int64_t i : IndexRange(cells)) {
      if (value_is_zero<GridT>(values[i])) {
        continue;
      }
      writer.setValue(cell_of(i, size), values[i]);
    }
  }
  return dst;
}

/** Returns the downsampled grid, or null when the type can not be averaged. */
static std::shared_ptr<openvdb::GridBase> downsample(const openvdb::GridBase &src,
                                                     const int factor)
{
  std::shared_ptr<openvdb::GridBase> result;
  const VolumeGridType grid_type = bke::volume_grid::get_type(src);
  BKE_volume_grid_type_to_static_type(grid_type,
                                      [&]<std::derived_from<openvdb::GridBase> GridT>() {
    if constexpr (bke::volume_grid::is_supported_grid_type<GridT>) {
      const GridT &typed = static_cast<const GridT &>(src);
      result = downsample_box(typed, factor);
    }
  });
  return result;
}

#endif /* WITH_OPENVDB */

static void node_geo_exec(GeoNodeExecParams params)
{
#ifdef WITH_OPENVDB
  bke::GVolumeGrid grid = params.extract_input<bke::GVolumeGrid>("Grid"_ustr);
  if (!grid) {
    params.set_default_remaining_outputs();
    return;
  }
  const int factor = math::clamp(params.extract_input<int>("Factor"_ustr), 1, 16);
  if (factor == 1) {
    params.set_output("Grid"_ustr, std::move(grid));
    return;
  }

  bke::VolumeTreeAccessToken token;
  const openvdb::GridBase &src = grid->grid(token);
  std::shared_ptr<openvdb::GridBase> downsampled = downsample(src, factor);
  if (!downsampled) {
    params.error_message_add(NodeWarningType::Error,
                             TIP_("This grid type can not be downsampled"));
    params.set_output("Grid"_ustr, std::move(grid));
    return;
  }
  params.set_output("Grid"_ustr, bke::GVolumeGrid(std::move(downsampled)));
#else
  node_geo_exec_with_missing_openvdb(params);
#endif
}

static void node_init(bNodeTree * /*tree*/, bNode *node)
{
  node->custom1 = SOCK_FLOAT;
}

static void node_rna(StructRNA *srna)
{
  RNA_def_node_enum(srna,
                    "data_type",
                    "Data Type",
                    "Node socket data type",
                    rna_enum_node_socket_data_type_items,
                    NOD_inline_enum_accessors(custom1),
                    SOCK_FLOAT,
                    grid_socket_type_items_filter_fn);
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeGridDownsample"_ustr);
  ntype.ui_name = "Downsample Grid";
  ntype.ui_description =
      "Sample a volume grid at a coarser resolution\n"
      "Each output voxel is the average of a block of input voxels, so the totals are preserved";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.draw_buttons = node_layout;
  ntype.initfunc = node_init;
  ntype.gather_link_search_ops = node_gather_link_search_ops;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_180;
  bke::node_register_type(ntype);

  node_rna(ntype.rna_ext.srna);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_grid_downsample_cc
