/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GEO_ha_program.hh"

#include <algorithm>

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_vector_types.hh"

#include "BKE_geometry_set.hh"
#include "BKE_attribute.hh"
#include "BKE_lib_id.hh"
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

#include <chrono>

#include <fmt/format.h>

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

/** Wall clock of one stage, printed by #ha_program_debug_timings when `BLENDER_HA_DEBUG` is set. */
class StageTimer {
 public:
  StageTimer() : last_(std::chrono::steady_clock::now()) {}

  /** Milliseconds since the previous call (and restarts the clock). */
  double lap()
  {
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - last_).count();
    last_ = now;
    return ms;
  }

 private:
  std::chrono::steady_clock::time_point last_;
};

}  // namespace

namespace {

constexpr int MAX_EXPRESSION_DEPTH = 64;
/** Upper bound for the number of geometry nodes in one zone body. Each one is a dispatch. */
constexpr int MAX_CHAIN_STEPS = 16;
/** Upper bound for the `Store Named Attribute` nodes after the chain. Each one is a dispatch. */
constexpr int MAX_STORE_STEPS = 8;

struct CompileContext {
  HaExprPool &pool;
  Vector<HaAggregate> &aggregates;
  Vector<HaAttribute> &attributes;
  /** The zone input node, used to verify that reductions read the zone geometry. */
  const bNode *zone_input_node = nullptr;
  bool ok = true;
  int depth = 0;
  /** Why the compilation stopped, in a form the node UI can show. */
  std::string reason;
  Map<const bNodeSocket *, const HaExpr *> compiled_outputs;
};

const HaExpr *compile_socket(CompileContext &ctx, const bNodeSocket &socket);
const HaExpr *compile_output(CompileContext &ctx, const bNode &node, int socket_index);

/**
 * Marks the compilation as failed. Only the first reason is kept: it is the one closest to the zone
 * output, which is the one the user has to fix first.
 */
bool fail(CompileContext &ctx, std::string reason)
{
  if (ctx.ok) {
    ctx.reason = std::move(reason);
  }
  ctx.ok = false;
  return false;
}

/**
 * Fails when one of \a socket_indices is linked, for inputs that the node's own implementation
 * reads but this program does not. Ignoring such a connection would silently compute a different
 * result than the CPU, which is the one thing this backend must never do.
 *
 * Inputs that the node itself ignores (an unused third input of `Math`, for example) do not have to
 * be listed: both sides ignore them, so the results still match.
 */
bool reject_linked_inputs(CompileContext &ctx,
                          const bNode &node,
                          const std::initializer_list<int> socket_indices)
{
  for (const int index : socket_indices) {
    if (index < 0 || index >= node.input_sockets().size()) {
      continue;
    }
    const bNodeSocket &socket = node.input_socket(index);
    if (socket.link != nullptr) {
      return fail(ctx,
                  fmt::format("node '{}': input '{}' is not supported",
                              node.name,
                              socket.name));
    }
  }
  return true;
}

float read_float_default(const bNodeSocket &socket)
{
  const auto *value = static_cast<const bNodeSocketValueFloat *>(socket.default_value);
  return value != nullptr ? value->value : 0.0f;
}

bool read_bool_default(const bNodeSocket &socket)
{
  const auto *value = static_cast<const bNodeSocketValueBoolean *>(socket.default_value);
  return value != nullptr ? value->value : false;
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

/** One entry of the Math / Vector Math dispatch tables. */
struct HaOpInfo {
  HaOp op = HaOp::Const;
  /** Number of inputs the operation reads (from socket 0 upwards). */
  int arg_count = 2;
  /** Result type; `Float3` only for the vector math table. */
  HaType type = HaType::Float;
  /** Extra input index when it is not simply `arg_count - 1` (Vector Math `Scale`). */
  int extra_input = -1;
};

const HaExpr *compile_math(CompileContext &ctx, const bNode &node)
{
  HaOpInfo info;
  switch (node.custom1) {
    case NODE_MATH_ADD:
      info = {HaOp::Add, 2};
      break;
    case NODE_MATH_SUBTRACT:
      info = {HaOp::Sub, 2};
      break;
    case NODE_MATH_MULTIPLY:
      info = {HaOp::Mul, 2};
      break;
    case NODE_MATH_DIVIDE:
      info = {HaOp::Div, 2};
      break;
    case NODE_MATH_MAXIMUM:
      info = {HaOp::Max, 2};
      break;
    case NODE_MATH_MINIMUM:
      info = {HaOp::Min, 2};
      break;
    case NODE_MATH_POWER:
      info = {HaOp::Pow, 2};
      break;
    case NODE_MATH_LOGARITHM:
      info = {HaOp::Log, 2};
      break;
    case NODE_MATH_ARCTAN2:
      info = {HaOp::Atan2, 2};
      break;
    case NODE_MATH_MODULO:
      info = {HaOp::Modulo, 2};
      break;
    case NODE_MATH_FLOORED_MODULO:
      info = {HaOp::FlooredModulo, 2};
      break;
    case NODE_MATH_SNAP:
      info = {HaOp::Snap, 2};
      break;
    case NODE_MATH_PINGPONG:
      info = {HaOp::PingPong, 2};
      break;
    case NODE_MATH_LESS_THAN:
      info = {HaOp::LessThan, 2};
      break;
    case NODE_MATH_GREATER_THAN:
      info = {HaOp::GreaterThan, 2};
      break;
    case NODE_MATH_MULTIPLY_ADD:
      info = {HaOp::MulAdd, 3};
      break;
    case NODE_MATH_WRAP:
      info = {HaOp::Wrap, 3};
      break;
    case NODE_MATH_SMOOTH_MIN:
      info = {HaOp::SmoothMin, 3};
      break;
    case NODE_MATH_SMOOTH_MAX:
      info = {HaOp::SmoothMax, 3};
      break;
    case NODE_MATH_COMPARE:
      info = {HaOp::CompareEps, 3};
      break;
    case NODE_MATH_SINE:
      info = {HaOp::Sin, 1};
      break;
    case NODE_MATH_COSINE:
      info = {HaOp::Cos, 1};
      break;
    case NODE_MATH_TANGENT:
      info = {HaOp::Tan, 1};
      break;
    case NODE_MATH_ARCSINE:
      info = {HaOp::Asin, 1};
      break;
    case NODE_MATH_ARCCOSINE:
      info = {HaOp::Acos, 1};
      break;
    case NODE_MATH_ARCTANGENT:
      info = {HaOp::Atan, 1};
      break;
    case NODE_MATH_ABSOLUTE:
      info = {HaOp::Abs, 1};
      break;
    case NODE_MATH_FLOOR:
      info = {HaOp::Floor, 1};
      break;
    case NODE_MATH_CEIL:
      info = {HaOp::Ceil, 1};
      break;
    case NODE_MATH_FRACTION:
      info = {HaOp::Fract, 1};
      break;
    case NODE_MATH_SQRT:
      info = {HaOp::Sqrt, 1};
      break;
    case NODE_MATH_INV_SQRT:
      info = {HaOp::InvSqrt, 1};
      break;
    case NODE_MATH_SIGN:
      info = {HaOp::Sign, 1};
      break;
    case NODE_MATH_EXPONENT:
      info = {HaOp::Exp, 1};
      break;
    case NODE_MATH_ROUND:
      info = {HaOp::Round, 1};
      break;
    case NODE_MATH_TRUNC:
      info = {HaOp::Trunc, 1};
      break;
    case NODE_MATH_SINH:
      info = {HaOp::Sinh, 1};
      break;
    case NODE_MATH_COSH:
      info = {HaOp::Cosh, 1};
      break;
    case NODE_MATH_TANH:
      info = {HaOp::Tanh, 1};
      break;
    default:
      fail(ctx, fmt::format("node '{}': operation is not supported", node.name));
      return &ctx.pool.constant(0.0f);
  }

  Vector<const HaExpr *> args;
  for (const int i : IndexRange(info.arg_count)) {
    args.append(compile_input(ctx, node, i));
  }
  HaExpr &result = ctx.pool.add(info.type, info.op, args.as_span());
  /* The `Clamp` checkbox of the Math node clamps the result to 0..1, and the node's own
   * implementation does the same, so it has to be part of the expression. */
  if ((node.custom2 & SHD_MATH_CLAMP) != 0) {
    return &ctx.pool.add(
        HaType::Float, HaOp::Clamp, {&result, &ctx.pool.constant(0.0f), &ctx.pool.constant(1.0f)});
  }
  return &result;
}

const HaExpr *compile_vector_math(CompileContext &ctx, const bNode &node)
{
  HaOpInfo info{HaOp::Const, 2, HaType::Float3};
  switch (node.custom1) {
    case NODE_VECTOR_MATH_ADD:
      info.op = HaOp::Add;
      break;
    case NODE_VECTOR_MATH_SUBTRACT:
      info.op = HaOp::Sub;
      break;
    case NODE_VECTOR_MATH_MULTIPLY:
      info.op = HaOp::Mul;
      break;
    case NODE_VECTOR_MATH_DIVIDE:
      info.op = HaOp::Div;
      break;
    case NODE_VECTOR_MATH_MINIMUM:
      info.op = HaOp::Min;
      break;
    case NODE_VECTOR_MATH_MAXIMUM:
      info.op = HaOp::Max;
      break;
    case NODE_VECTOR_MATH_MODULO:
      info.op = HaOp::Modulo;
      break;
    case NODE_VECTOR_MATH_SNAP:
      info.op = HaOp::Snap;
      break;
    case NODE_VECTOR_MATH_CROSS_PRODUCT:
      info.op = HaOp::Cross;
      break;
    case NODE_VECTOR_MATH_PROJECT:
      info.op = HaOp::Project;
      break;
    case NODE_VECTOR_MATH_REFLECT:
      info.op = HaOp::Reflect;
      break;
    case NODE_VECTOR_MATH_WRAP:
      info = {HaOp::Wrap, 3, HaType::Float3};
      break;
    case NODE_VECTOR_MATH_MULTIPLY_ADD:
      info = {HaOp::MulAdd, 3, HaType::Float3};
      break;
    case NODE_VECTOR_MATH_SCALE:
      /* Inputs: Vector, Vector_001, Vector_002, Scale. */
      info = {HaOp::Mul, 1, HaType::Float3, 3};
      break;
    case NODE_VECTOR_MATH_LENGTH:
      info = {HaOp::Length, 1, HaType::Float};
      break;
    case NODE_VECTOR_MATH_DOT_PRODUCT:
      info = {HaOp::Dot, 2, HaType::Float};
      break;
    case NODE_VECTOR_MATH_DISTANCE:
      info = {HaOp::Distance, 2, HaType::Float};
      break;
    case NODE_VECTOR_MATH_NORMALIZE:
      info = {HaOp::Normalize, 1, HaType::Float3};
      break;
    case NODE_VECTOR_MATH_ABSOLUTE:
      info = {HaOp::Abs, 1, HaType::Float3};
      break;
    case NODE_VECTOR_MATH_FLOOR:
      info = {HaOp::Floor, 1, HaType::Float3};
      break;
    case NODE_VECTOR_MATH_CEIL:
      info = {HaOp::Ceil, 1, HaType::Float3};
      break;
    case NODE_VECTOR_MATH_FRACTION:
      info = {HaOp::Fract, 1, HaType::Float3};
      break;
    default:
      fail(ctx, fmt::format("node '{}': operation is not supported", node.name));
      return &ctx.pool.constant(0.0f);
  }

  Vector<const HaExpr *> args;
  for (const int i : IndexRange(info.arg_count)) {
    args.append(compile_input(ctx, node, i));
  }
  if (info.extra_input >= 0) {
    args.append(compile_input(ctx, node, info.extra_input));
  }
  return &ctx.pool.add(info.type, info.op, args.as_span());
}

/**
 * `FunctionNodeCompare`, float only. `EQ`/`NE` use the `Epsilon` input (a plain `<=` on the absolute
 * difference, exactly like the node's own multi-function).
 */
const HaExpr *compile_compare(CompileContext &ctx, const bNode &node)
{
  const auto &storage = *static_cast<const NodeFunctionCompare *>(node.storage);
  if (storage.data_type != SOCK_FLOAT) {
    fail(ctx, "Compare: only the float mode is supported");
    return &ctx.pool.constant(0.0f);
  }
  if (storage.mode != NODE_COMPARE_MODE_ELEMENT) {
    fail(ctx, "Compare: only the element-wise mode is supported");
    return &ctx.pool.constant(0.0f);
  }

  HaParam param;
  switch (storage.operation) {
    case NODE_COMPARE_LESS_THAN:
      param = HaParam::CompareLess;
      break;
    case NODE_COMPARE_LESS_EQUAL:
      param = HaParam::CompareLessEqual;
      break;
    case NODE_COMPARE_GREATER_THAN:
      param = HaParam::CompareGreater;
      break;
    case NODE_COMPARE_GREATER_EQUAL:
      param = HaParam::CompareGreaterEqual;
      break;
    case NODE_COMPARE_EQUAL:
      param = HaParam::CompareEqual;
      break;
    case NODE_COMPARE_NOT_EQUAL:
      param = HaParam::CompareNotEqual;
      break;
    default:
      fail(ctx, "Compare: this operation is not supported");
      return &ctx.pool.constant(0.0f);
  }

  const bool with_epsilon = ELEM(param, HaParam::CompareEqual, HaParam::CompareNotEqual);
  Vector<const HaExpr *> args;
  args.append(compile_input(ctx, node, 0));
  args.append(compile_input(ctx, node, 1));
  if (with_epsilon) {
    if (node.input_sockets().size() < 3) {
      fail(ctx, "Compare: the Epsilon input is missing");
      return &ctx.pool.constant(0.0f);
    }
    args.append(compile_input(ctx, node, 2));
  }
  HaExpr &result = ctx.pool.add(HaType::Float, HaOp::Compare, args.as_span());
  result.param = int(param);
  return &result;
}

const HaExpr *compile_bool_math(CompileContext &ctx, const bNode &node)
{
  HaParam param;
  switch (node.custom1) {
    case NODE_BOOLEAN_MATH_AND:
      param = HaParam::BoolAnd;
      break;
    case NODE_BOOLEAN_MATH_OR:
      param = HaParam::BoolOr;
      break;
    case NODE_BOOLEAN_MATH_NOT:
      param = HaParam::BoolNot;
      break;
    case NODE_BOOLEAN_MATH_NAND:
      param = HaParam::BoolNand;
      break;
    case NODE_BOOLEAN_MATH_NOR:
      param = HaParam::BoolNor;
      break;
    case NODE_BOOLEAN_MATH_XOR:
      param = HaParam::BoolXor;
      break;
    case NODE_BOOLEAN_MATH_XNOR:
      param = HaParam::BoolXnor;
      break;
    default:
      fail(ctx, "Boolean Math: this operation is not supported");
      return &ctx.pool.constant(0.0f);
  }
  /* `NOT` only reads one input, the others read two. */
  const int arg_count = param == HaParam::BoolNot ? 1 : 2;
  Vector<const HaExpr *> args;
  for (const int i : IndexRange(arg_count)) {
    args.append(compile_input(ctx, node, i));
  }
  HaExpr &result = ctx.pool.add(HaType::Float, HaOp::BoolMath, args.as_span());
  result.param = int(param);
  return &result;
}

/** `ShaderNodeMix` for the float and vector types (colours and rotations are not supported). */
const HaExpr *compile_mix(CompileContext &ctx, const bNode &node, const int socket_index)
{
  const auto &storage = *static_cast<const NodeShaderMix *>(node.storage);
  /* Inputs: Factor_Float(0), Factor_Vector(1), A_Float(2), B_Float(3), A_Vector(4), B_Vector(5). */
  int factor_index;
  int a_index;
  int b_index;
  HaType type;
  switch (storage.data_type) {
    case SOCK_FLOAT:
      factor_index = 0;
      a_index = 2;
      b_index = 3;
      type = HaType::Float;
      if (socket_index != 0) {
        fail(ctx, "Mix: unexpected output socket");
        return &ctx.pool.constant(0.0f);
      }
      break;
    case SOCK_VECTOR:
      factor_index = storage.factor_mode == NODE_MIX_MODE_UNIFORM ? 0 : 1;
      a_index = 4;
      b_index = 5;
      type = HaType::Float3;
      if (socket_index != 1) {
        fail(ctx, "Mix: unexpected output socket");
        return &ctx.pool.constant(0.0f);
      }
      break;
    default:
      fail(ctx, "Mix: only the float and vector modes are supported");
      return &ctx.pool.constant(0.0f);
  }
  if (storage.blend_type != MA_RAMP_BLEND) {
    /* Only used for colours, which are not supported anyway. */
    fail(ctx, "Mix: only the mix blend mode is supported");
    return &ctx.pool.constant(0.0f);
  }

  HaExpr &result = ctx.pool.add(HaType(type),
                                HaOp::Mix,
                                {compile_input(ctx, node, factor_index),
                                 compile_input(ctx, node, a_index),
                                 compile_input(ctx, node, b_index)});
  if (storage.clamp_factor != 0) {
    result.param |= HA_PARAM_CLAMP;
  }
  return &result;
}

/** `GeometryNodeInputNamedAttribute`: reads a POINT attribute uploaded with the positions. */
const HaExpr *compile_named_attribute(CompileContext &ctx,
                                      const bNode &node,
                                      const int socket_index)
{
  if (socket_index != 0) {
    /* The `Exists` output would need to know at compile time whether the attribute is there. */
    fail(ctx, "Named Attribute: the 'Exists' output is not supported");
    return &ctx.pool.constant(0.0f);
  }
  const auto &storage = *static_cast<const NodeGeometryInputNamedAttribute *>(node.storage);
  int components;
  switch (storage.data_type) {
    case CD_PROP_FLOAT:
      components = 1;
      break;
    case CD_PROP_FLOAT3:
      components = 3;
      break;
    default:
      fail(ctx, "Named Attribute: only the float and vector types are supported");
      return &ctx.pool.constant(0.0f);
  }
  if (node.input_sockets().is_empty()) {
    fail(ctx, "Named Attribute: missing the name input");
    return &ctx.pool.constant(0.0f);
  }
  const bNodeSocket &name_socket = node.input_socket(0);
  if (name_socket.link != nullptr) {
    fail(ctx, "Named Attribute: the name has to be a constant");
    return &ctx.pool.constant(0.0f);
  }
  const auto *value = static_cast<const bNodeSocketValueString *>(name_socket.default_value);
  const std::string name = value != nullptr ? std::string(value->value) : std::string();
  if (name.empty()) {
    fail(ctx, "Named Attribute: the name is empty");
    return &ctx.pool.constant(0.0f);
  }

  int slot = -1;
  for (const int i : ctx.attributes.index_range()) {
    if (ctx.attributes[i].name == name && ctx.attributes[i].components == components) {
      slot = i;
      break;
    }
  }
  if (slot < 0) {
    slot = int(ctx.attributes.size());
    ctx.attributes.append({name, components});
  }

  HaExpr &read = ctx.pool.add(components == 1 ? HaType::Float : HaType::Float3,
                              HaOp::AttributeRead);
  read.attribute_slot = slot;
  return &read;
}

/** `GeometryNodeStoreNamedAttribute`: the value field is written back after the position steps. */
bool compile_store(CompileContext &ctx, const bNode &node, HaStore &r_store)
{
  const auto &storage = *static_cast<const NodeGeometryStoreNamedAttribute *>(node.storage);
  int components;
  switch (storage.data_type) {
    case CD_PROP_FLOAT:
      components = 1;
      break;
    case CD_PROP_FLOAT3:
      components = 3;
      break;
    default:
      return fail(ctx, "Store Named Attribute: only the float and vector types are supported");
  }
  if (storage.domain != int8_t(bke::AttrDomain::Point)) {
    return fail(ctx, "Store Named Attribute: only the point domain is supported");
  }
  if (node.input_sockets().size() < 4) {
    return fail(ctx, "Store Named Attribute: unexpected sockets");
  }
  /* A partial write would have to blend with the previous values, which are not on the device. */
  if (node.input_socket(1).link != nullptr) {
    return fail(ctx, "Store Named Attribute: the 'Selection' input is not supported");
  }
  const bNodeSocket &name_socket = node.input_socket(2);
  if (name_socket.link != nullptr) {
    return fail(ctx, "Store Named Attribute: the name has to be a constant");
  }
  const auto *value = static_cast<const bNodeSocketValueString *>(name_socket.default_value);
  const std::string name = value != nullptr ? std::string(value->value) : std::string();
  if (name.empty()) {
    return fail(ctx, "Store Named Attribute: the name is empty");
  }

  const HaExpr *compiled = compile_input(ctx, node, 3);
  if (!ctx.ok) {
    return false;
  }
  if (components == 1 && compiled->type != HaType::Float) {
    return fail(ctx, "Store Named Attribute: expected a float value");
  }
  if (components == 3 && compiled->type != HaType::Float3) {
    return fail(ctx, "Store Named Attribute: expected a vector value");
  }

  r_store.name = name;
  r_store.components = components;
  r_store.glsl_value = components == 1 ? ha_emit_glsl_scalar(*compiled) : ha_emit_glsl(*compiled);
  return true;
}

/** `ShaderNodeMapRange` for the float and vector types. */
const HaExpr *compile_map_range(CompileContext &ctx, const bNode &node, const int socket_index)
{
  const auto &storage = *static_cast<const NodeMapRange *>(node.storage);
  /* Inputs: Value(0) From Min(1) From Max(2) To Min(3) To Max(4) Steps(5), then the vector copies
   * at 6..11. */
  const bool vector = storage.data_type == CD_PROP_FLOAT3;
  const int base = vector ? 6 : 0;
  const HaType type = vector ? HaType::Float3 : HaType::Float;
  if (socket_index != (vector ? 1 : 0)) {
    fail(ctx, "Map Range: unexpected output socket");
    return &ctx.pool.constant(0.0f);
  }

  HaParam interpolation;
  switch (storage.interpolation_type) {
    case NODE_MAP_RANGE_LINEAR:
      interpolation = HaParam::MapRangeLinear;
      break;
    case NODE_MAP_RANGE_STEPPED:
      interpolation = HaParam::MapRangeStepped;
      break;
    case NODE_MAP_RANGE_SMOOTHSTEP:
      interpolation = HaParam::MapRangeSmoothStep;
      break;
    case NODE_MAP_RANGE_SMOOTHERSTEP:
      interpolation = HaParam::MapRangeSmootherStep;
      break;
    default:
      fail(ctx, "Map Range: this interpolation is not supported");
      return &ctx.pool.constant(0.0f);
  }

  Vector<const HaExpr *> args;
  for (const int i : IndexRange(5)) {
    args.append(compile_input(ctx, node, base + i));
  }
  if (interpolation == HaParam::MapRangeStepped) {
    args.append(compile_input(ctx, node, base + 5));
  }
  HaExpr &result = ctx.pool.add(type, HaOp::MapRange, args.as_span());
  result.param = int(interpolation);
  if (storage.clamp != 0) {
    result.param |= HA_PARAM_CLAMP;
  }
  return &result;
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
      fail(ctx, "Attribute Statistic: this output is not supported");
      return &ctx.pool.constant(0.0f);
  }

  /* Only scalar statistics for now (vector statistics would need a 3-component reduction). */
  if (node.custom1 != CD_PROP_FLOAT) {
    fail(ctx, "Attribute Statistic: only float statistics are supported");
    return &ctx.pool.constant(0.0f);
  }

  /* `Selection` is applied by the node on the CPU; ignoring it here would reduce over a different
   * set of elements. */
  if (!reject_linked_inputs(ctx, node, {1})) {
    return &ctx.pool.constant(0.0f);
  }

  /* The reduction has to run over the zone geometry (the GPU program only has that one). */
  if (node.input_sockets().is_empty()) {
    fail(ctx, "Attribute Statistic: missing geometry input");
    return &ctx.pool.constant(0.0f);
  }
  const bNodeSocket &geometry_input = node.input_socket(0);
  if (geometry_input.link == nullptr || geometry_input.link->fromnode != ctx.zone_input_node) {
    fail(ctx, "Attribute Statistic: the geometry has to come from the zone input");
    return &ctx.pool.constant(0.0f);
  }

  /* `Attribute` is input socket 2. */
  const HaExpr *value = compile_input(ctx, node, 2);
  if (!ctx.ok || value == nullptr || value->type != HaType::Float) {
    if (ctx.ok) {
      fail(ctx, "Attribute Statistic: only scalar attributes are supported");
    }
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
    /* `custom1` is the clamp type. Both modes read the `Min`/`Max` sockets: `Range` only swaps them
     * when they are in the wrong order (see `node_shader_clamp.cc`), it does *not* clamp to 0..1. */
    const HaExpr *value = compile_input(ctx, node, 0);
    const HaExpr *min_value = compile_input(ctx, node, 1);
    const HaExpr *max_value = compile_input(ctx, node, 2);
    if (node.custom1 == NODE_CLAMP_RANGE) {
      result = &ctx.pool.add(HaType::Float,
                             HaOp::Clamp,
                             {value,
                              &ctx.pool.add(HaType::Float, HaOp::Min, {min_value, max_value}),
                              &ctx.pool.add(HaType::Float, HaOp::Max, {min_value, max_value})});
    }
    else if (node.custom1 == NODE_CLAMP_MINMAX) {
      result = &ctx.pool.add(HaType::Float, HaOp::Clamp, {value, min_value, max_value});
    }
    else {
      fail(ctx, fmt::format("node '{}': clamp mode is not supported", node.name));
      result = &ctx.pool.constant(0.0f);
    }
  }
  else if (node.is_type("GeometryNodeAttributeStatistic"_ustr)) {
    result = compile_statistic(ctx, node, socket_index);
  }
  else if (node.is_type("FunctionNodeCompare"_ustr)) {
    result = compile_compare(ctx, node);
  }
  else if (node.is_type("FunctionNodeBooleanMath"_ustr)) {
    result = compile_bool_math(ctx, node);
  }
  else if (node.is_type("ShaderNodeMix"_ustr)) {
    result = compile_mix(ctx, node, socket_index);
  }
  else if (node.is_type("ShaderNodeMapRange"_ustr)) {
    result = compile_map_range(ctx, node, socket_index);
  }
  else if (node.is_type("GeometryNodeInputNamedAttribute"_ustr)) {
    result = compile_named_attribute(ctx, node, socket_index);
  }
  else {
    fail(ctx,
         fmt::format("node '{}' ({}) is not supported", node.name, node.idname));
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
      case SOCK_BOOLEAN:
        return &ctx.pool.constant(read_bool_default(socket) ? 1.0f : 0.0f);
      default:
        break;
    }
    fail(ctx,
         fmt::format("unlinked input '{}' of an unsupported socket type", socket.name));
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
                                                        const bke::bNodeTreeZone &zone,
                                                        std::string *r_reason)
{
  auto fail = [&](std::string reason) -> std::unique_ptr<HaZoneProgram> {
    if (r_reason != nullptr) {
      *r_reason = std::move(reason);
    }
    return nullptr;
  };

  const bNode *input_node = zone.input_node();
  const bNode *output_node = zone.output_node();
  if (input_node == nullptr || output_node == nullptr) {
    return fail("the zone is not paired");
  }

  /* The zone output has to be fed directly by a `Set Position` node. Field nodes (`Position`,
   * `Math`, ...) do not appear in `zone.child_nodes()` because they do not depend on the zone
   * geometry, so this is the only structural requirement that is needed: everything that reaches
   * the output goes through that node, and its inputs are what gets compiled below. */
  if (output_node->input_sockets().is_empty()) {
    return fail("the zone output has no input socket");
  }
  const bNodeSocket &zone_output_input = output_node->input_socket(0);
  if (zone_output_input.link == nullptr) {
    return fail("the zone output is not connected");
  }

  /* `Store Named Attribute` nodes may sit between the position chain and the zone output. They are
   * evaluated on the geometry as it leaves the zone, so they all have to come *after* the chain (a
   * store in the middle would need the positions at that point). */
  Vector<const bNode *> store_nodes;
  const bNode *current = zone_output_input.link->fromnode;
  while (current->is_type("GeometryNodeStoreNamedAttribute"_ustr)) {
    if (store_nodes.size() >= MAX_STORE_STEPS) {
      return fail(fmt::format("the zone contains more than {} 'Store Named Attribute' nodes",
                              MAX_STORE_STEPS));
    }
    store_nodes.append(current);
    if (current->input_sockets().is_empty()) {
      return fail("'Store Named Attribute' has no geometry input");
    }
    const bNodeSocket &geometry_input = current->input_socket(0);
    if (geometry_input.link == nullptr) {
      return fail(fmt::format("node '{}': the geometry input is not connected", current->name));
    }
    current = geometry_input.link->fromnode;
  }

  /* Walk back through the chain of `Set Position` nodes until the zone input is reached. The whole
   * chain is compiled into one program, so it costs one upload and one download no matter how many
   * nodes it contains. */
  Vector<const bNode *> chain;
  while (true) {
    if (chain.size() >= MAX_CHAIN_STEPS) {
      return fail(fmt::format("the zone contains more than {} 'Set Position' nodes in a row",
                              MAX_CHAIN_STEPS));
    }
    if (!current->is_type("GeometryNodeSetPosition"_ustr)) {
      return fail(fmt::format("node '{}' ({}) is not supported", current->name, current->idname));
    }
    if (current->input_sockets().size() < 4 || current->output_sockets().is_empty()) {
      return fail("'Set Position' has unexpected sockets");
    }
    chain.append(current);

    const bNodeSocket &geometry_input = current->input_socket(0);
    if (geometry_input.link == nullptr) {
      return fail(fmt::format("node '{}': the geometry input is not connected", current->name));
    }
    const bNode *previous = geometry_input.link->fromnode;
    if (previous == input_node) {
      break;
    }
    current = previous;
  }

  HaExprPool pool;
  auto program = std::make_unique<HaZoneProgram>();
  CompileContext ctx{pool, program->aggregates_, program->attributes_};
  ctx.zone_input_node = input_node;

  /* Compile first step -> last step (`chain` is stored last -> first). */
  for (const int i : chain.index_range()) {
    const bNode &node = *chain[chain.size() - 1 - i];

    /* `Set Position`: Geometry, Selection, Position, Offset.
     *
     * - `Position` unlinked means "keep the current position" (the socket's default input type is
     *   the position field), so the base of the expression is the element position. In a chain the
     *   "current position" is the previous step's result, which is exactly what the kernel reads.
     * - `Selection` defaults to true; the node writes `Position + Offset` for selected elements
     *   only, so the kernel picks between the old and the new position instead of skipping the
     *   write. */
    const HaExpr *base = node.input_socket(2).link != nullptr ?
                             compile_input(ctx, node, 2) :
                             &pool.add(HaType::Float3, HaOp::Position);
    const HaExpr *offset = compile_input(ctx, node, 3);
    const HaExpr *selection = compile_input(ctx, node, 1);
    if (!ctx.ok) {
      return fail(ctx.reason.empty() ? std::string("the field is not supported") : ctx.reason);
    }
    if (selection->type != HaType::Float) {
      /* A non scalar selection would need the same implicit conversion the CPU field evaluation
       * does; refusing keeps the GPU and CPU results identical. */
      return fail("the 'Selection' of 'Set Position' has to be a scalar");
    }

    const HaExpr &final_position = pool.add(HaType::Float3, HaOp::Add, {base, offset});
    program->steps_.append(
        {ha_emit_glsl(final_position), ha_emit_glsl_scalar(*selection)});
  }

  /* Then the stores, first -> last (`store_nodes` is stored last -> first). */
  for (const int i : store_nodes.index_range()) {
    HaStore store;
    if (!compile_store(ctx, *store_nodes[store_nodes.size() - 1 - i], store)) {
      return fail(ctx.reason.empty() ? std::string("the store is not supported") : ctx.reason);
    }
    program->stores_.append(std::move(store));
  }
  return program;
}

std::string HaZoneProgram::debug_description() const
{
  std::string result;
  for (const int i : steps_.index_range()) {
    if (!result.empty()) {
      result += " -> ";
    }
    result += fmt::format("[{}] position = {}, selection = {}",
                          i,
                          steps_[i].glsl_position,
                          steps_[i].glsl_selection);
  }
  return result;
}

bool HaZoneProgram::evaluate(const bke::GeometrySet &input,
                             bke::GeometrySet &r_output,
                             int *r_elements) const
{
  const Mesh *src_mesh = input.get_mesh();
  if (src_mesh == nullptr) {
    return false;
  }
  const Span<float3> positions = src_mesh->vert_positions();
  const int element_count = int(positions.size());
  if (element_count <= 0 || steps_.is_empty()) {
    return false;
  }
  /* This path binds its buffers itself, so the cache of #ha_bind_storage_buffers is stale from
   * here on (the local #HaBuffer of the dispatch invalidates it again when it is freed). */
  ha_bindings_invalidate();

  /* Invocations are padded to the local group size; the extra ones only touch the padding. */
  constexpr int local_size = 64;
  const int padded_count = (element_count + local_size - 1) / local_size * local_size;

  StageTimer timer;
  /* Two buffers, ping-ponged between the steps: this is what makes a chain cost one upload and one
   * download instead of two transfers per node. */
  HaBuffer buffers[2];
  if (!buffers[0].ensure(int64_t(padded_count) * 3) ||
      !buffers[1].ensure(int64_t(padded_count) * 3))
  {
    return false;
  }

  /* `positions` already is three interleaved floats per element, so both directions are a plain
   * byte copy instead of a `float3` component loop (which cost ~6 ms per million elements each
   * way, more than the dispatch itself). */
  Array<float> host_input(padded_count * 3, 0.0f);
  host_input.as_mutable_span().take_front(element_count * 3).copy_from(
      Span<float>(reinterpret_cast<const float *>(positions.data()), positions.size() * 3));
  buffers[0].upload(host_input);
  const double upload_ms = timer.lap();
  ha_program_debug("uploaded");

  /* The POINT attributes the program reads. They are taken from the *zone input* geometry: the
   * steps of the chain only move points around, so the attribute values are the same for every
   * step (a node that changes attributes has to invalidate this, see the header). */
  /* One buffer per attribute. `Array` (not `Array<T, N>`, whose second parameter is only the inline
   * capacity) so the elements actually exist. */
  Array<HaBuffer> attribute_buffers(attributes_.size());
  Vector<std::string> attribute_names;
  Vector<HaShaderResource> extra_resources;
  /* Reserve up front: the resource list points into the strings, and a reallocation of the vector
   * would invalidate those pointers (small strings keep their characters inline). */
  attribute_names.reserve(attributes_.size());
  extra_resources.reserve(attributes_.size());
  {
    const bke::AttributeAccessor src_attributes = src_mesh->attributes();
    for (const int i : attributes_.index_range()) {
      const HaAttribute &attribute = attributes_[i];
      const int64_t total = int64_t(padded_count) * attribute.components;
      Array<float> host(total, 0.0f);

      const std::optional<bke::AttributeMetaData> meta = src_attributes.lookup_meta_data(
          attribute.name);
      if (meta.has_value()) {
        if (meta->domain != bke::AttrDomain::Point) {
          /* Interpolating from another domain is not implemented; the CPU path handles it. */
          return false;
        }
        if (attribute.components == 1) {
          if (!ELEM(meta->data_type, bke::AttrType::Float, bke::AttrType::Int32,
                    bke::AttrType::Bool))
          {
            return false;
          }
          const bke::AttributeReader<float> reader = src_attributes.lookup<float>(
              attribute.name, bke::AttrDomain::Point);
          if (reader) {
            const VArray<float> &values = reader.varray;
            for (const int element : IndexRange(element_count)) {
              host[element] = values[element];
            }
          }
        }
        else {
          if (meta->data_type != bke::AttrType::Float3) {
            return false;
          }
          const bke::AttributeReader<float3> reader = src_attributes.lookup<float3>(
              attribute.name, bke::AttrDomain::Point);
          if (reader) {
            const VArray<float3> &values = reader.varray;
            for (const int element : IndexRange(element_count)) {
              const float3 value = values[element];
              host[element * 3 + 0] = value.x;
              host[element * 3 + 1] = value.y;
              host[element * 3 + 2] = value.z;
            }
          }
        }
      }

      if (!attribute_buffers[i].ensure(total)) {
        return false;
      }
      attribute_buffers[i].upload(host);
      attribute_names.append("attribute_" + std::to_string(i) + "[]");
      extra_resources.append(
          {3 + i, HaShaderQualifier::Read, "float", attribute_names.last().c_str()});
    }
  }

  /* Aggregates first: every step reads them, and they are all reduced over the *zone input*
   * geometry (that is what `Attribute Statistic` is wired to). Two passes per aggregate: one
   * invocation per block writes a partial value, then a single invocation folds them. */
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
      GPU_storagebuf_bind(buffers[0].buf(), 0);
      GPU_storagebuf_bind(buffer_params.buf(), 1);
      GPU_storagebuf_bind(buffer_partial.buf(), 2);
      ha_dispatch(stage1, block_count);
      GPU_storagebuf_unbind(buffer_partial.buf());
      GPU_storagebuf_unbind(buffer_params.buf());
      GPU_storagebuf_unbind(buffers[0].buf());
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

  /* The steps: each one reads the previous result and writes the other buffer. Only the last one
   * has to be finished before the readback, the intermediate dispatches stay queued. */
  double dispatch_ms = 0.0;
  for (const int i : steps_.index_range()) {
    const HaProgramStep &step = steps_[i];
    gpu::Shader *shader = ha_shader_get(step.glsl_position + '#' + step.glsl_selection,
                                        step.glsl_position,
                                        step.glsl_selection,
                                        Span<const HaShaderResource>(extra_resources.data(),
                                                                     int64_t(
                                                                         extra_resources.size())));
    if (shader == nullptr) {
      return false;
    }
    HaBuffer &src = buffers[i % 2];
    HaBuffer &dst = buffers[(i + 1) % 2];
    GPU_shader_bind(shader);
    GPU_storagebuf_bind(src.buf(), 0);
    GPU_storagebuf_bind(dst.buf(), 1);
    GPU_storagebuf_bind(buffer_aggregate.buf(), 2);
    for (const int attribute_i : attributes_.index_range()) {
      GPU_storagebuf_bind(attribute_buffers[attribute_i].buf(), 3 + attribute_i);
    }
    ha_dispatch(shader, padded_count);
    for (const int attribute_i : attributes_.index_range()) {
      GPU_storagebuf_unbind(attribute_buffers[attribute_i].buf());
    }
    GPU_storagebuf_unbind(buffer_aggregate.buf());
    GPU_storagebuf_unbind(dst.buf());
    GPU_storagebuf_unbind(src.buf());
    GPU_shader_unbind();
    if (i + 1 == steps_.size()) {
      GPU_finish();
      dispatch_ms = timer.lap();
    }
  }
  ha_program_debug("dispatched");
  if (r_elements != nullptr) {
    *r_elements = element_count;
  }

  /* After `n` steps the result is in buffer `n % 2`. */
  const HaBuffer &result = buffers[steps_.size() % 2];

  /* The stores read the *final* positions, so they run after the chain. */
  Array<HaBuffer> store_buffers(stores_.size());
  for (const int i : stores_.index_range()) {
    const HaStore &store = stores_[i];
    if (!store_buffers[i].ensure(int64_t(padded_count) * store.components)) {
      return false;
    }
    gpu::Shader *shader = ha_shader_get_store("store|" + store.glsl_value,
                                             store.glsl_value,
                                             store.components,
                                             Span<const HaShaderResource>(extra_resources.data(),
                                                                          int64_t(
                                                                              extra_resources.size())));
    if (shader == nullptr) {
      return false;
    }
    GPU_shader_bind(shader);
    GPU_storagebuf_bind(result.buf(), 0);
    GPU_storagebuf_bind(store_buffers[i].buf(), 1);
    GPU_storagebuf_bind(buffer_aggregate.buf(), 2);
    for (const int attribute_i : attributes_.index_range()) {
      GPU_storagebuf_bind(attribute_buffers[attribute_i].buf(), 3 + attribute_i);
    }
    ha_dispatch(shader, padded_count);
    for (const int attribute_i : attributes_.index_range()) {
      GPU_storagebuf_unbind(attribute_buffers[attribute_i].buf());
    }
    GPU_storagebuf_unbind(buffer_aggregate.buf());
    GPU_storagebuf_unbind(store_buffers[i].buf());
    GPU_storagebuf_unbind(result.buf());
    GPU_shader_unbind();
  }
  if (!stores_.is_empty()) {
    GPU_finish();
  }

  Array<float> host_output(padded_count * 3);
  result.download(host_output);
  const double download_ms = timer.lap();
  ha_program_debug("downloaded");

  Mesh *mesh = BKE_mesh_copy_for_eval(*src_mesh);
  MutableSpan<float3> dst_positions = mesh->vert_positions_for_write();
  dst_positions.copy_from(Span<float3>(reinterpret_cast<const float3 *>(host_output.data()),
                                       element_count));
  /* Positions changed, so the derived normals have to be recomputed. */
  mesh->tag_positions_changed();

  /* Write the stored attributes back. */
  if (!stores_.is_empty()) {
    bke::MutableAttributeAccessor mesh_attributes = mesh->attributes_for_write();
    for (const int i : stores_.index_range()) {
      const HaStore &store = stores_[i];
      Array<float> host_store(padded_count * store.components);
      store_buffers[i].download(host_store);
      if (store.components == 1) {
        bke::SpanAttributeWriter<float> writer =
            mesh_attributes.lookup_or_add_for_write_span<float>(store.name,
                                                                bke::AttrDomain::Point);
        if (!writer) {
          BKE_id_free(nullptr, mesh);
          return false;
        }
        writer.span.copy_from(host_store.as_span().take_front(element_count));
        writer.finish();
      }
      else {
        bke::SpanAttributeWriter<float3> writer =
            mesh_attributes.lookup_or_add_for_write_span<float3>(store.name,
                                                                bke::AttrDomain::Point);
        if (!writer) {
          BKE_id_free(nullptr, mesh);
          return false;
        }
        writer.span.copy_from(Span<float3>(reinterpret_cast<const float3 *>(host_store.data()),
                                           element_count));
        writer.finish();
      }
    }
  }

  bke::GeometrySet output = input;
  output.replace_mesh(mesh);
  r_output = std::move(output);
  if (ha_program_debug_enabled()) {
    /* The transfer is what decides whether this backend is worth it for element wise work: at 1M
     * points the CPU reference is a few milliseconds, so a slow upload or a stalling readback is
     * the whole budget. `dispatches` is what a chain buys: it grows, the two transfers do not. */
    fprintf(stderr,
            "HA program: %d elements, %d step(s), %d attribute(s), %d store(s): upload %.2f ms, "
            "dispatch %.2f ms (%d), download %.2f ms, write %.2f ms\n",
            element_count,
            int(steps_.size()),
            int(attributes_.size()),
            int(stores_.size()),
            upload_ms,
            dispatch_ms,
            int(steps_.size()),
            download_ms,
            timer.lap());
    fflush(stderr);
  }
  return true;
}

}  // namespace blender::geometry::ha
