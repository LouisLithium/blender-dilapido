/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * #HW Accelerate zone: a *residency + backend* scope.
 *
 * It intentionally does **not** introduce new control flow and does **not** own state:
 * - state belongs in the Simulation Zone,
 * - iteration belongs in the Repeat Zone,
 * - the zone only declares "evaluate this part of the tree on a compute backend, keep the data
 *   device-resident, sync at the border".
 *
 * M85: skeleton only. The body is evaluated transparently on the CPU (pass-through), so the zone
 * is a no-op that can be verified against "the same tree without a zone". The lazy-function
 * wrapper lives in `geometry_nodes_lazy_function.cc` (see
 * `build_hw_accelerate_zone_function`), which is where the GPU planner will be hooked in.
 */

#include "BLI_string.h"
#include "BLI_string_utf8.h"

#include "BLT_translation.hh"

#include "BKE_node.hh"
#include "BKE_node_runtime.hh"

#include "DNA_node_types.h"

#include "NOD_common.hh"
#include "NOD_node_declaration.hh"

#include "RNA_access.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "node_geometry_util.hh"

namespace blender {

namespace nodes::node_geo_hw_accelerate_cc {

/* -------------------------------------------------------------------- */
/** \name Shared
 * \{ */

static void node_layout_ex(ui::Layout &layout, bContext * /*C*/, PointerRNA *current_node_ptr)
{
  bNode &node = *static_cast<bNode *>(current_node_ptr->data);
  if (node.type_legacy != GEO_NODE_HW_ACCELERATE_INPUT) {
    return;
  }
  layout.label(TIP_("Backend not implemented yet: this zone currently runs on the CPU"),
               ICON_INFO);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name HW Accelerate Input
 * \{ */

namespace hw_accelerate_input_node {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();
  b.add_input<decl::Geometry>("Geometry"_ustr);
  b.add_output<decl::Geometry>("Geometry"_ustr).align_with_previous().propagate_all();
}

static void node_init(bNodeTree * /*tree*/, bNode *node)
{
  NodeGeometryHwAccelerateInput *data = MEM_new<NodeGeometryHwAccelerateInput>(__func__);
  /* Needs to be initialized for the node to work. */
  data->output_node_id = 0;
  data->backend = NODE_HW_ACCELERATE_BACKEND_AUTO;
  node->storage = data;
}

static void node_label(const bNodeTree * /*tree*/,
                       const bNode * /*node*/,
                       char *label,
                       const int label_maxncpy)
{
  BLI_strncpy_utf8(label, CTX_IFACE_(BLT_I18NCONTEXT_ID_NODETREE, "HW Accelerate"), label_maxncpy);
}

static void node_register()
{
  static bke::bNodeType ntype;
  geo_node_type_base(
      &ntype, "GeometryNodeHWAccelerateInput"_ustr, GEO_NODE_HW_ACCELERATE_INPUT);
  ntype.ui_name = "HW Accelerate Input";
  ntype.ui_description =
      "Start of a hardware accelerated zone. The node graph inside the zone is evaluated on a "
      "compute backend and its data stays on the device until the zone is left";
  ntype.enum_name_legacy = "HW_ACCELERATE_INPUT";
  ntype.nclass = NODE_CLASS_INTERFACE;
  ntype.initfunc = node_init;
  ntype.declare = node_declare;
  ntype.labelfunc = node_label;
  ntype.no_muting = true;
  ntype.draw_buttons_ex = node_layout_ex;
  bke::node_type_storage(ntype,
                         "NodeGeometryHwAccelerateInput",
                         node_free_standard_storage,
                         node_copy_standard_storage);
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace hw_accelerate_input_node

/** \} */

/* -------------------------------------------------------------------- */
/** \name HW Accelerate Output
 * \{ */

namespace hw_accelerate_output_node {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();
  b.add_input<decl::Geometry>("Geometry"_ustr);
  b.add_output<decl::Geometry>("Geometry"_ustr).align_with_previous().propagate_all();
}

static void node_init(bNodeTree * /*tree*/, bNode *node)
{
  NodeGeometryHwAccelerateOutput *data = MEM_new<NodeGeometryHwAccelerateOutput>(__func__);
  node->storage = data;
}

static void node_label(const bNodeTree * /*tree*/,
                       const bNode * /*node*/,
                       char *label,
                       const int label_maxncpy)
{
  BLI_strncpy_utf8(label, CTX_IFACE_(BLT_I18NCONTEXT_ID_NODETREE, "HW Accelerate"), label_maxncpy);
}

static void node_register()
{
  static bke::bNodeType ntype;
  geo_node_type_base(
      &ntype, "GeometryNodeHWAccelerateOutput"_ustr, GEO_NODE_HW_ACCELERATE_OUTPUT);
  ntype.ui_name = "HW Accelerate Output";
  ntype.ui_description =
      "End of a hardware accelerated zone. Data is synchronized back to the CPU here";
  ntype.enum_name_legacy = "HW_ACCELERATE_OUTPUT";
  ntype.nclass = NODE_CLASS_INTERFACE;
  ntype.initfunc = node_init;
  ntype.declare = node_declare;
  ntype.labelfunc = node_label;
  ntype.no_muting = true;
  bke::node_type_storage(ntype,
                         "NodeGeometryHwAccelerateOutput",
                         node_free_standard_storage,
                         node_copy_standard_storage);
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace hw_accelerate_output_node

/** \} */

}  // namespace nodes::node_geo_hw_accelerate_cc

}  // namespace blender
