/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GEO_ha_program.hh"

#include <algorithm>

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_vector_types.hh"

#include "BKE_geometry_set.hh"
#include "BKE_mesh.h"
#include "BKE_node.hh"
#include "BKE_node_runtime.hh"
#include "BKE_node_tree_zones.hh"

#include "DNA_mesh_types.h"
#include "DNA_node_types.h"

#include "GPU_context.hh"
#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_storage_buffer.hh"

#include "GEO_ha_context.hh"
#include "GEO_ha_expr.hh"

namespace blender::geometry::ha {

namespace {

bool ha_program_debug_enabled()
{
  static const bool enabled = getenv("BLENDER_HA_DEBUG") != nullptr;
  return enabled;
}

void ha_program_debug(const char *message)
{
  if (ha_program_debug_enabled()) {
    /* stderr is unbuffered, so the trail survives a crash. */
    fprintf(stderr, "HA program: %s\n", message);
    fflush(stderr);
  }
}

}  // namespace

namespace {

constexpr int MAX_EXPRESSION_DEPTH = 64;

struct CompileContext {
  HaExprPool &pool;
  Vector<HaAggregate> &aggregates;
  /** The zone input node, used to verify that reductions read the zone geometry. */
  const bNode *zone_input_node = nullptr;
  bool ok = true;
  int depth = 0;
  Map<const bNodeSocket *, const HaExpr *> compiled_outputs;
};

const HaExpr *compile_socket(CompileContext &ctx, const bNodeSocket &socket);
const HaExpr *compile_output(CompileContext &ctx, const bNode &node, int socket_index);

float read_float_default(const bNodeSocket &socket)
{
  const auto *value = static_cast<const bNodeSocketValueFloat *>(socket.default_value);
  return value != nullptr ? value->value : 0.0f;
}

float3 read_vector_default(const bNodeSocket &socket)
{
  const auto *value = static_cast<const bNodeSocketValueVector *>(socket.default_value);
  return value != nullptr ? float3(value->value) : float3(0.0f);
}

int read_int_default(const bNodeSocket &socket)
{
  const auto *value = static_cast<const bNodeSocketValueInt *>(socket.default_value);
  return value != nullptr ? value->value : 0;
}

const HaExpr *compile_input(CompileContext &ctx, const bNode &node, const int socket_index)
{
  if (socket_index < 0 || socket_index >= node.input_sockets().size()) {
    ctx.ok = false;
    return &ctx.pool.constant(0.0f);
  }
  return compile_socket(ctx, node.input_socket(socket_index));
}

const HaExpr *compile_math(CompileContext &ctx, const bNode &node)
{
  const HaExpr *a = compile_input(ctx, node, 0);
  const HaExpr *b = compile_input(ctx, node, 1);
  switch (node.custom1) {
    case NODE_MATH_ADD:
      return &ctx.pool.add(HaType::Float, HaOp::Add, {a, b});
    case NODE_MATH_SUBTRACT:
      return &ctx.pool.add(HaType::Float, HaOp::Sub, {a, b});
    case NODE_MATH_MULTIPLY:
      return &ctx.pool.add(HaType::Float, HaOp::Mul, {a, b});
    case NODE_MATH_DIVIDE:
      return &ctx.pool.add(HaType::Float, HaOp::Div, {a, b});
    case NODE_MATH_MAXIMUM:
      return &ctx.pool.add(HaType::Float, HaOp::Max, {a, b});
    case NODE_MATH_MINIMUM:
      return &ctx.pool.add(HaType::Float, HaOp::Min, {a, b});
    case NODE_MATH_SINE:
      return &ctx.pool.add(HaType::Float, HaOp::Sin, {a});
    case NODE_MATH_COSINE:
      return &ctx.pool.add(HaType::Float, HaOp::Cos, {a});
    case NODE_MATH_ABSOLUTE:
      return &ctx.pool.add(HaType::Float, HaOp::Abs, {a});
    case NODE_MATH_FLOOR:
      return &ctx.pool.add(HaType::Float, HaOp::Floor, {a});
    case NODE_MATH_CEIL:
      return &ctx.pool.add(HaType::Float, HaOp::Ceil, {a});
    case NODE_MATH_FRACTION:
      return &ctx.pool.add(HaType::Float, HaOp::Fract, {a});
    case NODE_MATH_SQRT:
      return &ctx.pool.add(HaType::Float, HaOp::Sqrt, {a});
    case NODE_MATH_POWER:
      return &ctx.pool.add(HaType::Float, HaOp::Pow, {a, b});
    default:
      break;
  }
  ctx.ok = false;
  return &ctx.pool.constant(0.0f);
}

const HaExpr *compile_vector_math(CompileContext &ctx, const bNode &node)
{
  const HaExpr *a = compile_input(ctx, node, 0);
  const HaExpr *b = compile_input(ctx, node, 1);
  switch (node.custom1) {
    case NODE_VECTOR_MATH_ADD:
      return &ctx.pool.add(HaType::Float3, HaOp::Add, {a, b});
    case NODE_VECTOR_MATH_SUBTRACT:
      return &ctx.pool.add(HaType::Float3, HaOp::Sub, {a, b});
    case NODE_VECTOR_MATH_MULTIPLY:
      return &ctx.pool.add(HaType::Float3, HaOp::Mul, {a, b});
    case NODE_VECTOR_MATH_DIVIDE:
      return &ctx.pool.add(HaType::Float3, HaOp::Div, {a, b});
    case NODE_VECTOR_MATH_SCALE:
      /* Inputs: Vector, Vector_001, Vector_002, Scale. */
      return &ctx.pool.add(HaType::Float3, HaOp::Mul, {a, compile_input(ctx, node, 3)});
    case NODE_VECTOR_MATH_LENGTH:
      return &ctx.pool.add(HaType::Float, HaOp::Length, {a});
    case NODE_VECTOR_MATH_ABSOLUTE:
      return &ctx.pool.add(HaType::Float3, HaOp::Abs, {a});
    case NODE_VECTOR_MATH_FLOOR:
      return &ctx.pool.add(HaType::Float3, HaOp::Floor, {a});
    case NODE_VECTOR_MATH_CEIL:
      return &ctx.pool.add(HaType::Float3, HaOp::Ceil, {a});
    case NODE_VECTOR_MATH_FRACTION:
      return &ctx.pool.add(HaType::Float3, HaOp::Fract, {a});
    case NODE_VECTOR_MATH_NORMALIZE:
      return &ctx.pool.add(HaType::Float3, HaOp::Normalize, {a});
    case NODE_VECTOR_MATH_DOT_PRODUCT:
      return &ctx.pool.add(HaType::Float, HaOp::Dot, {a, b});
    case NODE_VECTOR_MATH_DISTANCE:
      return &ctx.pool.add(HaType::Float, HaOp::Distance, {a, b});
    default:
      break;
  }
  ctx.ok = false;
  return &ctx.pool.constant(0.0f);
}

const HaExpr *compile_statistic(CompileContext &ctx, const bNode &node, const int socket_index)
{
  /* Outputs: 0 = Mean, 1 = Median, 2 = Sum, 3 = Min, 4 = Max, 5 = Range, 6 = Std Dev, 7 = Variance.
   * Only reductions that the backend can do in one pass are supported. */
  HaAggregateOp op;
  switch (socket_index) {
    case 0:
      op = HaAggregateOp::Mean;
      break;
    case 2:
      op = HaAggregateOp::Sum;
      break;
    case 3:
      op = HaAggregateOp::Min;
      break;
    case 4:
      op = HaAggregateOp::Max;
      break;
    default:
      ctx.ok = false;
      return &ctx.pool.constant(0.0f);
  }

  /* Only scalar statistics for now (vector statistics would need a 3-component reduction). */
  if (node.custom1 != CD_PROP_FLOAT) {
    ctx.ok = false;
    return &ctx.pool.constant(0.0f);
  }

  /* The reduction has to run over the zone geometry (the GPU program only has that one). */
  if (node.input_sockets().is_empty()) {
    ctx.ok = false;
    return &ctx.pool.constant(0.0f);
  }
  const bNodeSocket &geometry_input = node.input_socket(0);
  if (geometry_input.link == nullptr || geometry_input.link->fromnode != ctx.zone_input_node) {
    ctx.ok = false;
    return &ctx.pool.constant(0.0f);
  }

  /* `Attribute` is input socket 2. */
  const HaExpr *value = compile_input(ctx, node, 2);
  if (!ctx.ok || value == nullptr || value->type != HaType::Float) {
    ctx.ok = false;
    return &ctx.pool.constant(0.0f);
  }

  HaExpr &read = ctx.pool.add(HaType::Float, HaOp::AggregateRead);
  read.aggregate_slot = int(ctx.aggregates.size());
  ctx.aggregates.append({ha_emit_glsl(*value), op});
  return &read;
}

const HaExpr *compile_output(CompileContext &ctx, const bNode &node, const int socket_index)
{
  if (socket_index < 0 || socket_index >= node.output_sockets().size()) {
    ctx.ok = false;
    return &ctx.pool.constant(0.0f);
  }
  const bNodeSocket &socket = node.output_socket(socket_index);
  if (const HaExpr **compiled = ctx.compiled_outputs.lookup_ptr(&socket)) {
    return *compiled;
  }
  if (++ctx.depth > MAX_EXPRESSION_DEPTH) {
    ctx.ok = false;
    ctx.depth--;
    return &ctx.pool.constant(0.0f);
  }

  const HaExpr *result = nullptr;
  if (node.is_type("GeometryNodeInputPosition"_ustr)) {
    result = &ctx.pool.add(HaType::Float3, HaOp::Position);
  }
  else if (node.is_type("GeometryNodeInputIndex"_ustr)) {
    result = &ctx.pool.add(HaType::Float, HaOp::Index);
  }
  else if (node.is_type("ShaderNodeMath"_ustr)) {
    result = compile_math(ctx, node);
  }
  else if (node.is_type("ShaderNodeVectorMath"_ustr)) {
    result = compile_vector_math(ctx, node);
  }
  else if (node.is_type("ShaderNodeCombineXYZ"_ustr)) {
    result = &ctx.pool.add(HaType::Float3,
                          HaOp::Combine,
                          {compile_input(ctx, node, 0),
                           compile_input(ctx, node, 1),
                           compile_input(ctx, node, 2)});
  }
  else if (node.is_type("ShaderNodeSeparateXYZ"_ustr)) {
    const HaOp op = socket_index == 0 ? HaOp::SplitX :
                    socket_index == 1 ? HaOp::SplitY :
                                        HaOp::SplitZ;
    result = &ctx.pool.add(HaType::Float, op, {compile_input(ctx, node, 0)});
  }
  else if (node.is_type("ShaderNodeClamp"_ustr)) {
    /* `custom1` is the clamp type: NODE_CLAMP_MINMAX uses the Min/Max sockets, NODE_CLAMP_RANGE
     * clamps to 0..1. */
    if (node.custom1 == NODE_CLAMP_RANGE) {
      result = &ctx.pool.add(HaType::Float,
                            HaOp::Clamp,
                            {compile_input(ctx, node, 0),
                             &ctx.pool.constant(0.0f),
                             &ctx.pool.constant(1.0f)});
    }
    else if (node.custom1 == NODE_CLAMP_MINMAX) {
      result = &ctx.pool.add(HaType::Float,
                            HaOp::Clamp,
                            {compile_input(ctx, node, 0),
                             compile_input(ctx, node, 1),
                             compile_input(ctx, node, 2)});
    }
    else {
      ctx.ok = false;
      result = &ctx.pool.constant(0.0f);
    }
  }
  else if (node.is_type("GeometryNodeAttributeStatistic"_ustr)) {
    result = compile_statistic(ctx, node, socket_index);
  }
  else {
    ctx.ok = false;
    result = &ctx.pool.constant(0.0f);
  }

  ctx.depth--;
  ctx.compiled_outputs.add_new(&socket, result);
  return result;
}

const HaExpr *compile_socket(CompileContext &ctx, const bNodeSocket &socket)
{
  if (socket.link == nullptr) {
    switch (socket.type) {
      case SOCK_FLOAT:
        return &ctx.pool.constant(read_float_default(socket));
      case SOCK_VECTOR:
      case SOCK_RGBA:
        return &ctx.pool.constant(read_vector_default(socket));
      case SOCK_INT:
        return &ctx.pool.constant(float(read_int_default(socket)));
      default:
        break;
    }
    ctx.ok = false;
    return &ctx.pool.constant(0.0f);
  }
  return compile_output(ctx, *socket.link->fromnode, socket.link->fromsock->index());
}

void aggregate_op_strings(const HaAggregateOp op, std::string &r_combine, std::string &r_identity)
{
  switch (op) {
    case HaAggregateOp::Sum:
    case HaAggregateOp::Mean:
      r_combine = "(acc + value)";
      r_identity = "0.0";
      break;
    case HaAggregateOp::Min:
      r_combine = "min(acc, value)";
      r_identity = "1e30";
      break;
    case HaAggregateOp::Max:
      r_combine = "max(acc, value)";
      r_identity = "-1e30";
      break;
  }
}

}  // namespace

HaZoneProgram::~HaZoneProgram() = default;

std::unique_ptr<HaZoneProgram> HaZoneProgram::try_build(const bNodeTree & /*tree*/,
                                                        const bke::bNodeTreeZone &zone)
{
  const bNode *input_node = zone.input_node();
  const bNode *output_node = zone.output_node();
  if (input_node == nullptr || output_node == nullptr) {
    return nullptr;
  }

  /* Only a single "Set Position" node is supported for now. */
  const Vector<const bNode *> child_nodes = zone.child_nodes();
  if (child_nodes.size() != 1) {
    return nullptr;
  }
  const bNode &set_position = *child_nodes[0];
  if (!set_position.is_type("GeometryNodeSetPosition"_ustr)) {
    return nullptr;
  }
  if (set_position.input_sockets().size() < 4 || set_position.output_sockets().is_empty()) {
    return nullptr;
  }

  /* The geometry has to enter directly from the zone input node... */
  const bNodeSocket &geometry_input = set_position.input_socket(0);
  if (geometry_input.link == nullptr || geometry_input.link->fromnode != input_node) {
    return nullptr;
  }
  /* ...and leave directly into the zone output node. */
  if (output_node->input_sockets().is_empty()) {
    return nullptr;
  }
  const bNodeSocket &zone_output_input = output_node->input_socket(0);
  if (zone_output_input.link == nullptr || zone_output_input.link->fromnode != &set_position) {
    return nullptr;
  }

  HaExprPool pool;
  auto program = std::make_unique<HaZoneProgram>();
  CompileContext ctx{pool, program->aggregates_};
  ctx.zone_input_node = input_node;
  /* `Set Position`: Geometry, Selection, Position, Offset. A constant offset is supported too. */
  const HaExpr *offset = compile_input(ctx, set_position, 3);
  if (!ctx.ok) {
    return nullptr;
  }

  program->glsl_offset_ = ha_emit_glsl(*offset);
  return program;
}

bool HaZoneProgram::evaluate(const bke::GeometrySet &input, bke::GeometrySet &r_output) const
{
  const Mesh *src_mesh = input.get_mesh();
  if (src_mesh == nullptr) {
    return false;
  }
  const Span<float3> positions = src_mesh->vert_positions();
  const int element_count = int(positions.size());
  if (element_count <= 0) {
    return false;
  }
  /* This path binds its buffers itself, so the cache of #ha_bind_storage_buffers is stale from
   * here on (the local #HaBuffer of the dispatch invalidates it again when it is freed). */
  ha_bindings_invalidate();

  gpu::Shader *shader = ha_shader_get(glsl_offset_, glsl_offset_);
  if (shader == nullptr) {
    return false;
  }
  ha_program_debug("shader ready");

  /* Invocations are padded to the local group size; the extra ones only touch the padding. */
  constexpr int local_size = 64;
  const int padded_count = (element_count + local_size - 1) / local_size * local_size;

  HaBuffer buffer_in;
  HaBuffer buffer_out;
  if (!buffer_in.ensure(int64_t(padded_count) * 3) ||
      !buffer_out.ensure(int64_t(padded_count) * 3))
  {
    return false;
  }

  Array<float> host_input(padded_count * 3, 0.0f);
  for (const int i : positions.index_range()) {
    host_input[i * 3 + 0] = positions[i].x;
    host_input[i * 3 + 1] = positions[i].y;
    host_input[i * 3 + 2] = positions[i].z;
  }
  buffer_in.upload(host_input);
  ha_program_debug("uploaded");

  /* Aggregates first: the position kernel reads them. Two passes per aggregate: one invocation per
   * block writes a partial value, then a single invocation folds them. */
  const int aggregate_count = int(aggregates_.size());
  HaBuffer buffer_aggregate;
  if (!buffer_aggregate.ensure(std::max<int64_t>(1, int64_t(aggregate_count) * 3))) {
    return false;
  }
  if (aggregate_count > 0) {
    const int block_count = (element_count + HA_REDUCE_BLOCK_SIZE - 1) / HA_REDUCE_BLOCK_SIZE;
    HaBuffer buffer_partial;
    HaBuffer buffer_params;
    if (!buffer_partial.ensure(int64_t(block_count) * 3) || !buffer_params.ensure(4)) {
      return false;
    }
    for (const int slot : aggregates_.index_range()) {
      const HaAggregate &aggregate = aggregates_[slot];
      std::string combine;
      std::string identity;
      aggregate_op_strings(aggregate.op, combine, identity);

      const std::string key_prefix = aggregate.glsl_value + "|" + combine + "|";
      gpu::Shader *stage1 = ha_shader_get_reduce_stage1(
          "s1|" + key_prefix, aggregate.glsl_value, combine, identity);
      gpu::Shader *stage2 = ha_shader_get_reduce_stage2(
          "s2|" + key_prefix, combine, identity, aggregate.op == HaAggregateOp::Mean);
      if (stage1 == nullptr || stage2 == nullptr) {
        return false;
      }

      const std::array<float, 4> params{
          float(element_count), float(block_count), float(slot), 0.0f};
      buffer_params.upload(params);

      GPU_shader_bind(stage1);
      GPU_storagebuf_bind(buffer_in.buf(), 0);
      GPU_storagebuf_bind(buffer_params.buf(), 1);
      GPU_storagebuf_bind(buffer_partial.buf(), 2);
      ha_dispatch(stage1, block_count);
      GPU_storagebuf_unbind(buffer_partial.buf());
      GPU_storagebuf_unbind(buffer_params.buf());
      GPU_storagebuf_unbind(buffer_in.buf());
      GPU_shader_unbind();

      GPU_shader_bind(stage2);
      GPU_storagebuf_bind(buffer_partial.buf(), 0);
      GPU_storagebuf_bind(buffer_params.buf(), 1);
      GPU_storagebuf_bind(buffer_aggregate.buf(), 2);
      ha_dispatch(stage2, 1);
      GPU_storagebuf_unbind(buffer_aggregate.buf());
      GPU_storagebuf_unbind(buffer_params.buf());
      GPU_storagebuf_unbind(buffer_partial.buf());
      GPU_shader_unbind();
    }
    GPU_finish();
    ha_program_debug("aggregates ready");
  }

  GPU_shader_bind(shader);
  GPU_storagebuf_bind(buffer_in.buf(), 0);
  GPU_storagebuf_bind(buffer_out.buf(), 1);
  GPU_storagebuf_bind(buffer_aggregate.buf(), 2);
  ha_dispatch(shader, padded_count);
  GPU_storagebuf_unbind(buffer_aggregate.buf());
  GPU_storagebuf_unbind(buffer_in.buf());
  GPU_storagebuf_unbind(buffer_out.buf());
  GPU_shader_unbind();
  ha_program_debug("dispatched");
  GPU_finish();

  Array<float> host_output(padded_count * 3);
  buffer_out.download(host_output);
  ha_program_debug("downloaded");

  Mesh *mesh = BKE_mesh_copy_for_eval(*src_mesh);
  MutableSpan<float3> dst_positions = mesh->vert_positions_for_write();
  for (const int i : positions.index_range()) {
    dst_positions[i] = float3(
        host_output[i * 3 + 0], host_output[i * 3 + 1], host_output[i * 3 + 2]);
  }
  /* Positions changed, so the derived normals have to be recomputed. */
  mesh->tag_positions_changed();

  bke::GeometrySet output = input;
  output.replace_mesh(mesh);
  r_output = std::move(output);
  return true;
}

}  // namespace blender::geometry::ha
