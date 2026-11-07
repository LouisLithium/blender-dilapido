/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GEO_ha_program.hh"

#include <algorithm>
#include <cstring>

#include "BLI_array.hh"
#include "BLI_map.hh"
#include "BLI_math_matrix.hh"
#include "BLI_math_rotation.hh"
#include "BLI_math_vector_types.hh"

#include "BKE_anim_data.hh"
#include "BKE_geometry_set.hh"
#include "BKE_attribute.hh"
#include "BKE_curves.hh"
#include "BKE_lib_id.hh"
#include "BKE_material.hh"
#include "BKE_mesh.h"
#include "BKE_pointcloud.hh"
#include "BKE_node.hh"
#include "BKE_node_runtime.hh"
#include "BKE_node_tree_zones.hh"

#include "DNA_action_types.h"
#include "DNA_anim_types.h"
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
/** Upper bound for the FACE/EDGE attribute passes. Each one is a dispatch. */
constexpr int MAX_ELEMENT_STEPS = 8;

struct CompileContext {
  HaExprPool &pool;
  Vector<HaAggregate> &aggregates;
  Vector<HaAttribute> &attributes;
  /** Animated socket defaults of the whole program, shared by the element contexts. */
  Vector<HaParamSource> &params;
  /** False in kernels that do not bind `user_params[]` (stores, element passes, reductions). */
  bool allow_params = true;
  /** The zone input node, used to verify that reductions read the zone geometry. */
  const bNode *zone_input_node = nullptr;
  /** The tree, used to reject animated socket values that would change the shader source. */
  const bNodeTree *tree = nullptr;
  bool ok = true;
  int depth = 0;
  /** Why the compilation stopped, in a form the node UI can show. */
  std::string reason;
  /** Set while compiling a field for a FACE/EDGE element pass: `Position` and reductions are not
   * available there, and `Named Attribute` reads that domain instead of POINT. */
  bool element_domain = false;
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

float4 read_color_default(const bNodeSocket &socket)
{
  const auto *value = static_cast<const bNodeSocketValueRGBA *>(socket.default_value);
  return value != nullptr ? float4(value->value) : float4(0.0f);
}

int ha_type_components(const HaType type)
{
  switch (type) {
    case HaType::Float:
    case HaType::Int:
      return 1;
    case HaType::Float2:
      return 2;
    case HaType::Float3:
      return 3;
    case HaType::Color:
      return 4;
  }
  return 1;
}

const char *ha_type_glsl(const HaType type)
{
  switch (type) {
    case HaType::Float:
      return "float";
    case HaType::Int:
      return "int";
    case HaType::Float2:
      return "vec2";
    case HaType::Float3:
      return "vec3";
    case HaType::Color:
      return "vec4";
  }
  return "float";
}

/**
 * GLSL type of an attribute *storage* buffer. The GPU module only lets `int*` / `uint*` / `float*`
 * type names through verbatim (`ShaderCreateInfo::buffer_typename`); everything else gets a
 * `_host_shared_` suffix. Attribute buffers are therefore flat scalar arrays, and the emitter
 * indexes them per component (`attribute_0[index * 3 + 2]`).
 */
const char *ha_attribute_buffer_glsl(const HaType type)
{
  return type == HaType::Int ? "int" : "float";
}

int read_int_default(const bNodeSocket &socket)
{
  const auto *value = static_cast<const bNodeSocketValueInt *>(socket.default_value);
  return value != nullptr ? value->value : 0;
}

/**
 * True when the default value of \a socket is driven by a driver or an action.
 *
 * Such a value changes without the node tree changing, so inlining it into the shader source
 * would produce a new program (and a new shader compilation) every frame. The field zone in
 * `HWA_test.blend` did exactly that with a `frame * 0.1` driver on a `Vector Math` input:
 * 1.3 s per frame on the device instead of 19 ms. Uniform parameters are the long term fix; until
 * then the zone falls back to the CPU, which is correct and much faster than recompiling.
 */
bool socket_value_is_animated(const bNodeTree &tree, const bNodeSocket &socket)
{
  const AnimData *adt = BKE_animdata_from_id(&tree.id);
  if (adt == nullptr) {
    return false;
  }
  const bNode *node = socket.runtime->owner_node;
  if (node == nullptr) {
    return false;
  }
  const std::string path = fmt::format(
      "nodes[\"{}\"].inputs[{}].default_value", node->name, socket.index());
  for (const FCurve *fcu = static_cast<const FCurve *>(adt->drivers.first); fcu != nullptr;
       fcu = fcu->next)
  {
    if (fcu->rna_path != nullptr && path == fcu->rna_path) {
      return true;
    }
  }
  if (adt->action != nullptr) {
    for (const FCurve *fcu = static_cast<const FCurve *>(adt->action->curves.first); fcu != nullptr;
         fcu = fcu->next)
    {
      if (fcu->rna_path != nullptr && path == fcu->rna_path) {
        return true;
      }
    }
  }
  return false;
}

const HaExpr *compile_input(CompileContext &ctx, const bNode &node, const int socket_index)
{
  if (socket_index < 0 || socket_index >= node.input_sockets().size()) {
    ctx.ok = false;
    return &ctx.pool.constant(0.0f);
  }
  return compile_socket(ctx, node.input_socket(socket_index));
}

/** Inserts the explicit conversion the field system would do implicitly. */
const HaExpr *coerce(CompileContext &ctx, const HaExpr *value, const HaType expected)
{
  if (!ctx.ok || value == nullptr || value->type == expected) {
    return value;
  }
  HaOp op;
  switch (expected) {
    case HaType::Float:
      op = HaOp::ConvertToFloat;
      break;
    case HaType::Int:
      op = HaOp::ConvertToInt;
      break;
    case HaType::Float2:
      op = HaOp::ConvertToFloat2;
      break;
    case HaType::Float3:
      op = HaOp::ConvertToFloat3;
      break;
    case HaType::Color:
      op = HaOp::ConvertToColor;
      break;
    default:
      return value;
  }
  return &ctx.pool.add(expected, op, {value});
}

const HaExpr *compile_input_as(CompileContext &ctx,
                               const bNode &node,
                               const int socket_index,
                               const HaType expected)
{
  return coerce(ctx, compile_input(ctx, node, socket_index), expected);
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
    args.append(compile_input_as(ctx, node, i, HaType::Float));
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

/**
 * `FunctionNodeIntegerMath`. Integer division/modulo are the exact CPU semantics
 * (`multi_function_common.cc`), including the "divisor is zero" guards. `Power`, `GCD` and `LCM`
 * are not implemented yet and fall back to the CPU.
 */
const HaExpr *compile_integer_math(CompileContext &ctx, const bNode &node)
{
  HaOp op;
  int arg_count = 2;
  switch (node.custom1) {
    case NODE_INTEGER_MATH_ADD:
      op = HaOp::Add;
      break;
    case NODE_INTEGER_MATH_SUBTRACT:
      op = HaOp::Sub;
      break;
    case NODE_INTEGER_MATH_MULTIPLY:
      op = HaOp::Mul;
      break;
    case NODE_INTEGER_MATH_DIVIDE:
      op = HaOp::Div;
      break;
    case NODE_INTEGER_MATH_MULTIPLY_ADD:
      op = HaOp::MulAdd;
      arg_count = 3;
      break;
    case NODE_INTEGER_MATH_FLOORED_MODULO:
      op = HaOp::FlooredModulo;
      break;
    case NODE_INTEGER_MATH_MODULO:
      op = HaOp::Modulo;
      break;
    case NODE_INTEGER_MATH_ABSOLUTE:
      op = HaOp::Abs;
      arg_count = 1;
      break;
    case NODE_INTEGER_MATH_SIGN:
      op = HaOp::Sign;
      arg_count = 1;
      break;
    case NODE_INTEGER_MATH_MINIMUM:
      op = HaOp::Min;
      break;
    case NODE_INTEGER_MATH_MAXIMUM:
      op = HaOp::Max;
      break;
    case NODE_INTEGER_MATH_NEGATE:
      return &ctx.pool.add(
          HaType::Int,
          HaOp::Sub,
          {&ctx.pool.constant(0), compile_input_as(ctx, node, 0, HaType::Int)});
    case NODE_INTEGER_MATH_DIVIDE_FLOOR:
      op = HaOp::DivideFloor;
      break;
    case NODE_INTEGER_MATH_DIVIDE_CEIL:
      op = HaOp::DivideCeil;
      break;
    case NODE_INTEGER_MATH_DIVIDE_ROUND:
      op = HaOp::DivideRound;
      break;
    default:
      fail(ctx, "Integer Math: this operation is not supported");
      return &ctx.pool.constant(0.0f);
  }

  Vector<const HaExpr *> args;
  for (const int i : IndexRange(arg_count)) {
    args.append(compile_input_as(ctx, node, i, HaType::Int));
  }
  return &ctx.pool.add(HaType::Int, op, args.as_span());
}

/**
 * `ShaderNodeVectorMath` (float3). Scalar inputs are broadcast, exactly like the CPU evaluator's
 * `eval_arg_as_vector` and the old GLSL `vec3(...)` promotion.
 */
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
    args.append(compile_input_as(ctx, node, i, HaType::Float3));
  }
  if (info.extra_input >= 0) {
    args.append(compile_input_as(ctx, node, info.extra_input, HaType::Float));
  }
  return &ctx.pool.add(info.type, info.op, args.as_span());
}

/**
 * `FunctionNodeCompare` for floats and integers. In the float mode `EQ`/`NE` use the `Epsilon`
 * input (a plain `<=` on the absolute difference, exactly like the node's own multi-function); the
 * integer mode compares exactly.
 */
const HaExpr *compile_compare(CompileContext &ctx, const bNode &node)
{
  const auto &storage = *static_cast<const NodeFunctionCompare *>(node.storage);
  const bool is_int = storage.data_type == SOCK_INT;
  if (!ELEM(storage.data_type, SOCK_FLOAT, SOCK_INT)) {
    fail(ctx, "Compare: only the float and integer modes are supported");
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

  const bool with_epsilon = !is_int && ELEM(param, HaParam::CompareEqual, HaParam::CompareNotEqual);
  const HaType arg_type = is_int ? HaType::Int : HaType::Float;
  Vector<const HaExpr *> args;
  args.append(compile_input_as(ctx, node, 0, arg_type));
  args.append(compile_input_as(ctx, node, 1, arg_type));
  if (with_epsilon) {
    if (node.input_sockets().size() < 3) {
      fail(ctx, "Compare: the Epsilon input is missing");
      return &ctx.pool.constant(0.0f);
    }
    args.append(compile_input_as(ctx, node, 2, HaType::Float));
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
    args.append(compile_input_as(ctx, node, i, HaType::Float));
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

  const HaType factor_type = type == HaType::Float ? HaType::Float :
                             (storage.factor_mode == NODE_MIX_MODE_UNIFORM ? HaType::Float :
                                                                            HaType::Float3);
  HaExpr &result = ctx.pool.add(type,
                                HaOp::Mix,
                                {compile_input_as(ctx, node, factor_index, factor_type),
                                 compile_input_as(ctx, node, a_index, type),
                                 compile_input_as(ctx, node, b_index, type)});
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
  HaType type;
  switch (storage.data_type) {
    case CD_PROP_FLOAT:
      type = HaType::Float;
      break;
    case CD_PROP_INT32:
    case CD_PROP_BOOL:
      /* The IR has no boolean type; booleans are read as 0/1 integers. */
      type = HaType::Int;
      break;
    case CD_PROP_FLOAT2:
      type = HaType::Float2;
      break;
    case CD_PROP_FLOAT3:
      type = HaType::Float3;
      break;
    case CD_PROP_COLOR:
      type = HaType::Color;
      break;
    default:
      fail(ctx, "Named Attribute: this attribute type is not supported");
      return &ctx.pool.constant(0.0f);
  }
  const int components = ha_type_components(type);
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
    if (ctx.attributes[i].name == name && ctx.attributes[i].type == type) {
      slot = i;
      break;
    }
  }
  if (slot < 0) {
    slot = int(ctx.attributes.size());
    ctx.attributes.append({name, type, components});
  }

  HaExpr &read = ctx.pool.add(type, HaOp::AttributeRead);
  read.attribute_slot = slot;
  return &read;
}

/**
 * `GeometryNodeTransform` in the components mode with constant inputs. The matrix is built here (the
 * IR has no matrix type) and emitted as three dot products plus a translation, which is exactly how
 * `math::transform_point` evaluates `mat * vec4(p, 1)`.
 *
 * The `Matrix` mode and linked component inputs are refused: they would need a uniform (single)
 * value in the expression language, which only knows per-element values.
 */
const HaExpr *compile_transform(CompileContext &ctx, const bNode &node, HaExprPool &pool)
{
  if (node.input_sockets().size() < 5) {
    fail(ctx, "Transform Geometry: unexpected sockets");
    return nullptr;
  }
  const auto *mode = static_cast<const bNodeSocketValueMenu *>(node.input_socket(1).default_value);
  if (mode != nullptr && mode->value == GEO_NODE_TRANSFORM_MODE_MATRIX) {
    fail(ctx, "Transform Geometry: the matrix mode is not supported");
    return nullptr;
  }
  for (const int i : {2, 3, 4}) {
    const bNodeSocket &socket = node.input_socket(i);
    if (socket.link != nullptr) {
      fail(ctx,
           "Transform Geometry: translation, rotation and scale have to be constant values");
      return nullptr;
    }
    if (ctx.tree != nullptr && socket_value_is_animated(*ctx.tree, socket)) {
      fail(ctx,
           "Transform Geometry: an animated translation/rotation/scale would rebuild the shader "
           "every frame");
      return nullptr;
    }
  }
  const auto *translation_value = static_cast<const bNodeSocketValueVector *>(
      node.input_socket(2).default_value);
  const auto *rotation_value = static_cast<const bNodeSocketValueRotation *>(
      node.input_socket(3).default_value);
  const auto *scale_value = static_cast<const bNodeSocketValueVector *>(
      node.input_socket(4).default_value);
  if (translation_value == nullptr || rotation_value == nullptr || scale_value == nullptr) {
    fail(ctx, "Transform Geometry: missing input values");
    return nullptr;
  }

  const float3 translation(translation_value->value);
  const float3 scale(scale_value->value);
  const math::Quaternion rotation = math::to_quaternion(
      math::EulerXYZ(float3(rotation_value->value_euler)));
  const float4x4 mat = math::from_loc_rot_scale<float4x4>(translation, rotation, scale);

  /* `float4x4` is column major, so row j is `(mat[0][j], mat[1][j], mat[2][j])`. */
  const HaExpr *base = &pool.add(HaType::Float3, HaOp::Position);
  const HaExpr *x = &pool.add(
      HaType::Float, HaOp::Dot, {base, &pool.constant(float3(mat[0][0], mat[1][0], mat[2][0]))});
  const HaExpr *y = &pool.add(
      HaType::Float, HaOp::Dot, {base, &pool.constant(float3(mat[0][1], mat[1][1], mat[2][1]))});
  const HaExpr *z = &pool.add(
      HaType::Float, HaOp::Dot, {base, &pool.constant(float3(mat[0][2], mat[1][2], mat[2][2]))});
  const HaExpr *combined = &pool.add(HaType::Float3, HaOp::Combine, {x, y, z});
  return &pool.add(HaType::Float3,
                   HaOp::Add,
                   {combined, &pool.constant(float3(mat[3][0], mat[3][1], mat[3][2]))});
}

/**
 * `ShaderNodeTexWhiteNoise > Value`. The vector input defaults to the position field when it is not
 * linked, exactly like `Set Position > Position`. Only the `Value` output is supported: `Color` is
 * an RGBA socket and the IR has no colour type, so returning a vector there could silently change
 * the meaning of a downstream conversion.
 */
const HaExpr *compile_white_noise(CompileContext &ctx, const bNode &node, const int socket_index)
{
  if (socket_index != 0) {
    fail(ctx, "White Noise Texture: only the 'Value' output is supported");
    return &ctx.pool.constant(0.0f);
  }
  const int dimensions = node.custom1;
  if (dimensions < 1 || dimensions > 4) {
    fail(ctx, "White Noise Texture: unsupported dimensions");
    return &ctx.pool.constant(0.0f);
  }

  HaExpr &noise = ctx.pool.add(HaType::Float, HaOp::WhiteNoise);
  noise.param = dimensions;

  if (dimensions != 1) {
    const bNodeSocket &vector_socket = node.input_socket(0);
    const HaExpr *vector = vector_socket.link != nullptr ?
                               compile_input_as(ctx, node, 0, HaType::Float3) :
                               &ctx.pool.add(HaType::Float3, HaOp::Position);
    if (vector->type != HaType::Float3) {
      fail(ctx, "White Noise Texture: the vector input has to be a vector");
      return &ctx.pool.constant(0.0f);
    }
    noise.args.append(vector);
  }
  if (dimensions == 1 || dimensions == 4) {
    const HaExpr *w = compile_input_as(ctx, node, 1, HaType::Float);
    if (w->type != HaType::Float) {
      fail(ctx, "White Noise Texture: 'W' has to be a scalar");
      return &ctx.pool.constant(0.0f);
    }
    noise.args.append(w);
  }
  return &noise;
}

/**
 * `ShaderNodeTexNoise > Factor`, 3D. The vector defaults to the position field, `Scale` multiplies
 * it, `Detail` is clamped to [0, 15] and `Roughness` to >= 0, exactly like the node. `Color` is
 * refused for the same reason as in `White Noise Texture`.
 */
const HaExpr *compile_noise_texture(CompileContext &ctx, const bNode &node, const int socket_index)
{
  /* Outputs: Factor(0), Color(1). The color is RGB only: the IR has no RGBA, and the node's alpha is
   * always 1. */
  if (socket_index != 0 && socket_index != 1) {
    fail(ctx, "Noise Texture: only the 'Factor' and 'Color' outputs are supported");
    return &ctx.pool.constant(0.0f);
  }
  const auto &storage = *static_cast<const NodeTexNoise *>(node.storage);
  if (storage.dimensions != 4 && storage.dimensions != 3 && storage.dimensions != 2 &&
      storage.dimensions != 1)
  {
    fail(ctx, "Noise Texture: only the 1D, 2D, 3D and 4D dimensions are supported");
    return &ctx.pool.constant(0.0f);
  }
  const bool is_1d = storage.dimensions == 1;
  if (node.input_sockets().size() < 9) {
    fail(ctx, "Noise Texture: unexpected sockets");
    return &ctx.pool.constant(0.0f);
  }

  /* Inputs: Vector(0), W(1), Scale(2), Detail(3), Roughness(4), Lacunarity(5), Offset(6), Gain(7),
   * Distortion(8). 1D uses `W * Scale`, 3D `Vector * Scale`. */
  HaExpr *position = nullptr;
  const HaExpr *w = &ctx.pool.constant(0.0f);
  const HaExpr *scale = compile_input_as(ctx, node, 2, HaType::Float);
  if (is_1d) {
    const HaExpr *w_input = compile_input_as(ctx, node, 1, HaType::Float);
    if (w_input->type != HaType::Float) {
      fail(ctx, "Noise Texture: 'W' has to be a scalar");
      return &ctx.pool.constant(0.0f);
    }
    w = &ctx.pool.add(HaType::Float, HaOp::Mul, {w_input, scale});
  }
  else {
    const HaExpr *vector = node.input_socket(0).link != nullptr ?
                               compile_input_as(ctx, node, 0, HaType::Float3) :
                               &ctx.pool.add(HaType::Float3, HaOp::Position);
    if (vector->type != HaType::Float3) {
      fail(ctx, "Noise Texture: the vector input has to be a vector");
      return &ctx.pool.constant(0.0f);
    }
    position = &ctx.pool.add(HaType::Float3, HaOp::Mul, {vector, scale});
    if (storage.dimensions == 4) {
      /* 4D samples `vec4(vector * scale, w * scale)`. */
      const HaExpr *w_input = compile_input_as(ctx, node, 1, HaType::Float);
      if (w_input->type != HaType::Float) {
        fail(ctx, "Noise Texture: 'W' has to be a scalar");
        return &ctx.pool.constant(0.0f);
      }
      w = &ctx.pool.add(HaType::Float, HaOp::Mul, {w_input, scale});
    }
  }
  const HaExpr *detail = &ctx.pool.add(HaType::Float,
                                       HaOp::Clamp,
                                       {compile_input_as(ctx, node, 3, HaType::Float),
                                        &ctx.pool.constant(0.0f),
                                        &ctx.pool.constant(15.0f)});
  const HaExpr *roughness = &ctx.pool.add(
      HaType::Float,
      HaOp::Max,
      {compile_input_as(ctx, node, 4, HaType::Float), &ctx.pool.constant(0.0f)});
  const HaExpr *lacunarity = compile_input_as(ctx, node, 5, HaType::Float);
  const HaExpr *offset = compile_input_as(ctx, node, 6, HaType::Float);
  const HaExpr *gain = compile_input_as(ctx, node, 7, HaType::Float);
  const HaExpr *distortion = compile_input_as(ctx, node, 8, HaType::Float);
  if (!ctx.ok) {
    fail(ctx, ctx.reason.empty() ? std::string("the field is not supported") : ctx.reason);
    return &ctx.pool.constant(0.0f);
  }

  const HaExpr *vector_arg = is_1d ? &ctx.pool.constant(float3(0.0f)) : position;
  HaExpr &noise = ctx.pool.add(socket_index == 1 ? HaType::Float3 : HaType::Float,
                               socket_index == 1 ? HaOp::NoiseTextureColor : HaOp::NoiseTexture,
                               {vector_arg,
                                w,
                                detail,
                                roughness,
                                lacunarity,
                                offset,
                                gain,
                                distortion});
  noise.param = int(storage.type) | (storage.normalize != 0 ? HA_PARAM_CLAMP : 0) |
                ((storage.dimensions & 0x7) << HA_PARAM_NOISE_DIM_SHIFT);
  return &noise;
}

/**
 * `ShaderNodeTexWave > Fac`. The node only uses `perlin_fbm` (lacunarity 2, normalized) for the
 * distortion, so the whole thing is one helper call. `Color` is refused like the other textures.
 */
const HaExpr *compile_wave_texture(CompileContext &ctx, const bNode &node, const int socket_index)
{
  /* `Wave Texture` declares `Color` first and `Fac` second (the opposite order of `Noise
   * Texture`), so the factor is output 1. */
  if (socket_index != 1) {
    fail(ctx, "Wave Texture: only the 'Fac' output is supported");
    return &ctx.pool.constant(0.0f);
  }
  const auto &storage = *static_cast<const NodeTexWave *>(node.storage);
  if (node.input_sockets().size() < 7) {
    fail(ctx, "Wave Texture: unexpected sockets");
    return &ctx.pool.constant(0.0f);
  }
  /* Inputs: Vector(0), Scale(1), Distortion(2), Detail(3), Detail Scale(4),
   * Detail Roughness(5), Phase Offset(6). */
  const HaExpr *vector = node.input_socket(0).link != nullptr ?
                             compile_input_as(ctx, node, 0, HaType::Float3) :
                             &ctx.pool.add(HaType::Float3, HaOp::Position);
  if (vector->type != HaType::Float3) {
    fail(ctx, "Wave Texture: the vector input has to be a vector");
    return &ctx.pool.constant(0.0f);
  }
  HaExpr &wave = ctx.pool.add(HaType::Float,
                              HaOp::WaveTexture,
                              {vector,
                               compile_input_as(ctx, node, 1, HaType::Float),
                               compile_input_as(ctx, node, 2, HaType::Float),
                               compile_input_as(ctx, node, 3, HaType::Float),
                               compile_input_as(ctx, node, 4, HaType::Float),
                               compile_input_as(ctx, node, 5, HaType::Float),
                               compile_input_as(ctx, node, 6, HaType::Float)});
  if (!ctx.ok) {
    fail(ctx, ctx.reason.empty() ? std::string("the field is not supported") : ctx.reason);
    return &ctx.pool.constant(0.0f);
  }
  wave.param = (storage.wave_type & 0x3) | ((storage.bands_direction & 0x3) << 2) |
               ((storage.rings_direction & 0x3) << 4) | ((storage.wave_profile & 0x3) << 6);
  return &wave;
}

/**
 * `ShaderNodeTexVoronoi`, 1D/2D/3D/4D. `Distance`, `Position`, `W` and the `N-Sphere Radius` output
 * are supported; `Color` is refused (the IR has no colour type).
 */
const HaExpr *compile_voronoi_texture(CompileContext &ctx,
                                      const bNode &node,
                                      const int socket_index)
{
  const auto &storage = *static_cast<const NodeTexVoronoi *>(node.storage);
  const int dimensions = storage.dimensions;
  if (dimensions < 1 || dimensions > 4) {
    fail(ctx, "Voronoi Texture: unsupported dimensions");
    return &ctx.pool.constant(0.0f);
  }
  if (node.input_sockets().size() < 9) {
    fail(ctx, "Voronoi Texture: unexpected sockets");
    return &ctx.pool.constant(0.0f);
  }

  /* Outputs: Distance(0), Color(1), Position(2), W(3), Radius(4). */
  const int feature = storage.feature;
  if (socket_index == 1) {
    fail(ctx, "Voronoi Texture: the 'Color' output is not supported");
    return &ctx.pool.constant(0.0f);
  }
  if ((socket_index == 4) != (feature == 4)) {
    fail(ctx, "Voronoi Texture: the 'Radius' output only exists for the 'N-Sphere Radius' feature");
    return &ctx.pool.constant(0.0f);
  }
  if (socket_index == 2 && (feature >= 3 || dimensions == 1)) {
    fail(ctx, "Voronoi Texture: this feature and dimension have no 'Position' output");
    return &ctx.pool.constant(0.0f);
  }
  if (socket_index == 3 && dimensions != 1 && dimensions != 4) {
    fail(ctx, "Voronoi Texture: the 'W' output only exists in the 1D and 4D dimensions");
    return &ctx.pool.constant(0.0f);
  }

  /* Args of every op: vector, w, scale, detail, roughness, lacunarity, smoothness, exponent,
   * randomness. `vector` is unused in 1D, `w` in 2D/3D. */
  const HaExpr *vector = &ctx.pool.constant(float3(0.0f));
  if (dimensions != 1) {
    vector = node.input_socket(0).link != nullptr ?
                 compile_input_as(ctx, node, 0, HaType::Float3) :
                 &ctx.pool.add(HaType::Float3, HaOp::Position);
    if (vector->type != HaType::Float3) {
      fail(ctx, "Voronoi Texture: the vector input has to be a vector");
      return &ctx.pool.constant(0.0f);
    }
  }
  const HaExpr *w = &ctx.pool.constant(0.0f);
  if (dimensions == 1 || dimensions == 4) {
    w = compile_input_as(ctx, node, 1, HaType::Float);
    if (w->type != HaType::Float) {
      fail(ctx, "Voronoi Texture: 'W' has to be a scalar");
      return &ctx.pool.constant(0.0f);
    }
  }
  const HaExpr *scale = compile_input_as(ctx, node, 2, HaType::Float);
  /* `randomness` is clamped to [0, 1] by the node. */
  const HaExpr *randomness = &ctx.pool.add(
      HaType::Float,
      HaOp::Clamp,
      {compile_input_as(ctx, node, 8, HaType::Float),
       &ctx.pool.constant(0.0f),
       &ctx.pool.constant(1.0f)});
  if (!ctx.ok) {
    fail(ctx, ctx.reason.empty() ? std::string("the field is not supported") : ctx.reason);
    return &ctx.pool.constant(0.0f);
  }

  const int param = (feature & 0xF) | ((storage.distance & 0xF) << 4) |
                    (storage.normalize != 0 ? HA_PARAM_CLAMP : 0) |
                    ((dimensions & 0x7) << HA_PARAM_VORONOI_DIM_SHIFT);

  if (socket_index == 4) {
    HaExpr &radius = ctx.pool.add(HaType::Float,
                                  HaOp::VoronoiRadius,
                                  {vector, w, scale, randomness});
    radius.param = param;
    return &radius;
  }

  const int metric = storage.distance;
  const HaExpr *detail = compile_input_as(ctx, node, 3, HaType::Float);
  const HaExpr *roughness = compile_input_as(ctx, node, 4, HaType::Float);
  const HaExpr *lacunarity = compile_input_as(ctx, node, 5, HaType::Float);
  /* `smoothness` is only used by `Smooth F1`, where the node halves it and clamps it to [0, 0.5]. */
  const HaExpr *smoothness = &ctx.pool.constant(0.0f);
  if (feature == 2) {
    const HaExpr *halved = &ctx.pool.add(
        HaType::Float,
        HaOp::Div,
        {compile_input_as(ctx, node, 6, HaType::Float), &ctx.pool.constant(2.0f)});
    smoothness = &ctx.pool.add(HaType::Float,
                               HaOp::Clamp,
                               {halved, &ctx.pool.constant(0.0f), &ctx.pool.constant(0.5f)});
  }
  /* The node only reads `Exponent` for `Minkowski` and not in 1D. */
  const HaExpr *exponent = metric == 3 && dimensions != 1 ?
                               compile_input_as(ctx, node, 7, HaType::Float) :
                               &ctx.pool.constant(0.0f);
  if (!ctx.ok) {
    fail(ctx, ctx.reason.empty() ? std::string("the field is not supported") : ctx.reason);
    return &ctx.pool.constant(0.0f);
  }

  const HaOp op = socket_index == 2 ? HaOp::VoronoiPosition :
                  socket_index == 3 ? HaOp::VoronoiW :
                                      HaOp::VoronoiTexture;
  const HaType type = socket_index == 2 ? HaType::Float3 : HaType::Float;
  HaExpr &voronoi = ctx.pool.add(type,
                                 op,
                                 {vector,
                                  w,
                                  scale,
                                  detail,
                                  roughness,
                                  lacunarity,
                                  smoothness,
                                  exponent,
                                  randomness});
  voronoi.param = param;
  return &voronoi;
}

/**
 * `GeometryNodeSetShadeSmooth`: writes `sharp_face` (FACE) or `sharp_edge` (EDGE) for the selected
 * elements. Its two fields are compiled for the element domain: they may read that domain's
 * attributes and the element index, but not `Position` (a face has no position of its own).
 */
bool compile_set_shade_smooth(CompileContext &point_ctx, const bNode &node, HaElementStep &r_step)
{
  const bke::AttrDomain domain = bke::AttrDomain(node.custom1);
  if (!ELEM(domain, bke::AttrDomain::Face, bke::AttrDomain::Edge)) {
    fail(point_ctx, "Set Shade Smooth: only the Face and Edge domains are supported");
    return false;
  }
  const Span<const bNodeSocket *> inputs = node.input_sockets();
  if (inputs.size() < 3) {
    fail(point_ctx, "Set Shade Smooth: unexpected sockets");
    return false;
  }
  r_step.domain = domain == bke::AttrDomain::Face ? HaElementDomain::Face : HaElementDomain::Edge;
  r_step.attribute = domain == bke::AttrDomain::Face ? "sharp_face" : "sharp_edge";
  r_step.components = 1;
  /* Creating `sharp_face` / `sharp_edge` starts from `true` (measured: with a partial selection and
   * no existing attribute, the unselected elements are sharp on the CPU). */
  r_step.default_value = 1.0f;

  /* Compile the fields for the element domain: `Named Attribute` reads that domain, and the kernel
   * binds those buffers instead of the POINT ones. */
  HaExprPool element_pool;
  Vector<HaAggregate> element_aggregates;
  CompileContext ctx{element_pool, element_aggregates, r_step.attributes, point_ctx.params};
  ctx.allow_params = false;
  ctx.zone_input_node = point_ctx.zone_input_node;
  ctx.tree = point_ctx.tree;
  ctx.element_domain = true;

  /* Inputs: Geometry(0), Selection(1), Shade Smooth(2). */
  const HaExpr *selection = compile_input_as(ctx, node, 1, HaType::Float);
  const HaExpr *smooth = compile_input_as(ctx, node, 2, HaType::Float);
  if (!ctx.ok) {
    fail(point_ctx, ctx.reason.empty() ? std::string("the field is not supported") : ctx.reason);
    return false;
  }
  if (selection->type != HaType::Float || smooth->type != HaType::Float) {
    fail(point_ctx, "Set Shade Smooth: 'Selection' and 'Shade Smooth' have to be scalars");
    return false;
  }
  r_step.glsl_selection = ha_emit_glsl_scalar(*selection);
  /* `sharp` is the inverse of the node's `Shade Smooth`; the field's `float -> bool` conversion is
   * `> 0`, exactly like the field system. */
  r_step.glsl_value = "(1.0 - ((" + ha_emit_glsl_scalar(*smooth) + ") > 0.0 ? 1.0 : 0.0))";
  return true;
}

/**
 * `GeometryNodeSetMaterial`: writes `material_index` (FACE) for the selected faces. The material has
 * to be a constant (a material *field* is not supported), and the slot index is resolved at
 * execution time, when the material is appended to the result mesh exactly like the CPU does.
 */
bool compile_set_material(CompileContext &point_ctx, const bNode &node, HaElementStep &r_step)
{
  const Span<const bNodeSocket *> inputs = node.input_sockets();
  if (inputs.size() < 3) {
    fail(point_ctx, "Set Material: unexpected sockets");
    return false;
  }
  r_step.domain = HaElementDomain::Face;
  r_step.attribute = "material_index";
  r_step.components = 1;
  r_step.type = HaElementType::Int;
  r_step.default_value = 0.0f;
  r_step.material_step = true;

  /* Inputs: Geometry(0), Selection(1), Material(2). */
  const bNodeSocket &material_socket = node.input_socket(2);
  if (material_socket.link != nullptr) {
    fail(point_ctx, "Set Material: a linked 'Material' is not supported");
    return false;
  }
  const auto *material_value = static_cast<const bNodeSocketValueMaterial *>(
      material_socket.default_value);
  r_step.material = material_value != nullptr ? material_value->value : nullptr;

  HaExprPool element_pool;
  Vector<HaAggregate> element_aggregates;
  CompileContext ctx{element_pool, element_aggregates, r_step.attributes, point_ctx.params};
  ctx.allow_params = false;
  ctx.zone_input_node = point_ctx.zone_input_node;
  ctx.tree = point_ctx.tree;
  ctx.element_domain = true;
  const HaExpr *selection = compile_input_as(ctx, node, 1, HaType::Float);
  if (!ctx.ok) {
    fail(point_ctx, ctx.reason.empty() ? std::string("the field is not supported") : ctx.reason);
    return false;
  }
  if (selection->type != HaType::Float) {
    fail(point_ctx, "Set Material: 'Selection' has to be a scalar");
    return false;
  }
  r_step.glsl_selection = ha_emit_glsl_scalar(*selection);
  /* The CPU takes a different path when every face is selected (it rebuilds the attribute instead of
   * filling a mask), and that path also decides whether a default slot is added. */
  r_step.all_selected = selection->op == HaOp::Const && selection->value[0] != 0.0f;
  return true;
}

/**
 * `GeometryNodeFlipFaces`: reverses the winding of the selected faces. The selection is a FACE field,
 * so it goes through the same element-domain context as the other element nodes.
 */
bool compile_flip_faces(CompileContext &point_ctx, const bNode &node, HaCornerStep &r_step)
{
  const Span<const bNodeSocket *> inputs = node.input_sockets();
  if (inputs.size() < 2) {
    fail(point_ctx, "Flip Faces: unexpected sockets");
    return false;
  }
  HaExprPool element_pool;
  Vector<HaAggregate> element_aggregates;
  CompileContext ctx{element_pool, element_aggregates, r_step.attributes, point_ctx.params};
  ctx.allow_params = false;
  ctx.zone_input_node = point_ctx.zone_input_node;
  ctx.tree = point_ctx.tree;
  ctx.element_domain = true;
  /* Inputs: Geometry(0), Selection(1). */
  const HaExpr *selection = compile_input_as(ctx, node, 1, HaType::Float);
  if (!ctx.ok) {
    fail(point_ctx, ctx.reason.empty() ? std::string("the field is not supported") : ctx.reason);
    return false;
  }
  if (selection->type != HaType::Float) {
    fail(point_ctx, "Flip Faces: 'Selection' has to be a scalar");
    return false;
  }
  r_step.glsl_selection = ha_emit_glsl_scalar(*selection);
  return true;
}

/**
 * `AttributeAccessor::lookup_meta_data(name)` returns the first attribute with that name, regardless
 * of its domain. A name can exist on more than one domain (for example a FACE and a CORNER
 * attribute), so look the domain up explicitly.
 */
std::optional<bke::AttributeMetaData> lookup_meta_data_in_domain(
    const bke::AttributeAccessor &attributes, const StringRef name, const bke::AttrDomain domain)
{
  std::optional<bke::AttributeMetaData> result;
  attributes.foreach_attribute([&](const bke::AttributeIter &iter) {
    if (iter.name == name && iter.domain == domain) {
      result.emplace(bke::AttributeMetaData{iter.domain, iter.data_type});
      iter.stop();
    }
  });
  return result;
}

/** Uploads one attribute buffer. Returns false when the attribute's actual type/domain is not
 * supported, in which case the caller falls back to the CPU. */
bool upload_one_attribute(const bke::AttributeAccessor &src_attributes,
                          const HaAttribute &attribute,
                          const bke::AttrDomain domain,
                          const int count,
                          const int padded,
                          HaBuffer &r_buffer)
{
  const int components = attribute.components;
  const int64_t total = int64_t(padded) * components;
  const std::optional<bke::AttributeMetaData> meta = lookup_meta_data_in_domain(
      src_attributes, attribute.name, domain);
  if (meta.has_value() && meta->domain != domain) {
    /* Interpolating from another domain is not implemented; the CPU path handles it. */
    return false;
  }

  if (attribute.type == HaType::Int) {
    Array<int> host(total, 0);
    if (meta.has_value()) {
      if (meta->data_type == bke::AttrType::Int32) {
        const bke::AttributeReader<int> reader = src_attributes.lookup<int>(attribute.name,
                                                                            domain);
        if (reader) {
          const VArray<int> &values = reader.varray;
          for (const int element : IndexRange(count)) {
            host[element] = values[element];
          }
        }
      }
      else if (meta->data_type == bke::AttrType::Bool) {
        const bke::AttributeReader<bool> reader = src_attributes.lookup<bool>(attribute.name,
                                                                              domain);
        if (reader) {
          const VArray<bool> &values = reader.varray;
          for (const int element : IndexRange(count)) {
            host[element] = values[element] ? 1 : 0;
          }
        }
      }
      else {
        return false;
      }
    }
    if (!r_buffer.ensure(total)) {
      return false;
    }
    /* The SSBO is raw bytes; the kernel declares this binding as `int[]`. */
    r_buffer.upload(Span<float>(reinterpret_cast<const float *>(host.data()), total));
    return true;
  }

  Array<float> host(total, 0.0f);
  if (meta.has_value()) {
    switch (attribute.type) {
      case HaType::Float: {
        if (meta->data_type == bke::AttrType::Float) {
          const bke::AttributeReader<float> reader = src_attributes.lookup<float>(attribute.name,
                                                                                  domain);
          if (reader) {
            const VArray<float> &values = reader.varray;
            for (const int element : IndexRange(count)) {
              host[element] = values[element];
            }
          }
        }
        else if (meta->data_type == bke::AttrType::Int32) {
          const bke::AttributeReader<int> reader = src_attributes.lookup<int>(attribute.name,
                                                                              domain);
          if (reader) {
            const VArray<int> &values = reader.varray;
            for (const int element : IndexRange(count)) {
              host[element] = float(values[element]);
            }
          }
        }
        else if (meta->data_type == bke::AttrType::Bool) {
          const bke::AttributeReader<bool> reader = src_attributes.lookup<bool>(attribute.name,
                                                                                domain);
          if (reader) {
            const VArray<bool> &values = reader.varray;
            for (const int element : IndexRange(count)) {
              host[element] = values[element] ? 1.0f : 0.0f;
            }
          }
        }
        else {
          return false;
        }
        break;
      }
      case HaType::Float2: {
        if (meta->data_type != bke::AttrType::Float2) {
          return false;
        }
        const bke::AttributeReader<float2> reader = src_attributes.lookup<float2>(attribute.name,
                                                                                  domain);
        if (reader) {
          const VArray<float2> &values = reader.varray;
          for (const int element : IndexRange(count)) {
            const float2 value = values[element];
            host[element * 2 + 0] = value.x;
            host[element * 2 + 1] = value.y;
          }
        }
        break;
      }
      case HaType::Float3: {
        if (meta->data_type != bke::AttrType::Float3) {
          return false;
        }
        const bke::AttributeReader<float3> reader = src_attributes.lookup<float3>(attribute.name,
                                                                                  domain);
        if (reader) {
          const VArray<float3> &values = reader.varray;
          for (const int element : IndexRange(count)) {
            const float3 value = values[element];
            host[element * 3 + 0] = value.x;
            host[element * 3 + 1] = value.y;
            host[element * 3 + 2] = value.z;
          }
        }
        break;
      }
      case HaType::Color: {
        if (meta->data_type != bke::AttrType::ColorFloat) {
          return false;
        }
        const bke::AttributeReader<ColorGeometry4f> reader =
            src_attributes.lookup<ColorGeometry4f>(attribute.name, domain);
        if (reader) {
          const VArray<ColorGeometry4f> &values = reader.varray;
          for (const int element : IndexRange(count)) {
            const ColorGeometry4f value = values[element];
            host[element * 4 + 0] = value.r;
            host[element * 4 + 1] = value.g;
            host[element * 4 + 2] = value.b;
            host[element * 4 + 3] = value.a;
          }
        }
        break;
      }
      case HaType::Int:
        break;
    }
  }
  if (!r_buffer.ensure(total)) {
    return false;
  }
  r_buffer.upload(host);
  return true;
}

/**
 * Uploads the attributes a FACE/EDGE/CORNER pass reads, from the matching domain of the source
 * mesh. Returns false when an attribute cannot be represented (wrong domain or type), which makes
 * the caller fall back to the CPU.
 */
bool upload_domain_attributes(const bke::AttributeAccessor &src_attributes,
                              const Span<const HaAttribute> attributes,
                              const bke::AttrDomain domain,
                              const int count,
                              const int padded,
                              Array<HaBuffer> &r_buffers,
                              Vector<std::string> &r_names,
                              Vector<HaShaderResource> &r_resources,
                              const int first_slot)
{
  r_names.reserve(attributes.size());
  r_resources.reserve(attributes.size());
  for (const int attribute_i : attributes.index_range()) {
    const HaAttribute &attribute = attributes[attribute_i];
    if (!upload_one_attribute(
            src_attributes, attribute, domain, count, padded, r_buffers[attribute_i]))
    {
      return false;
    }
    r_names.append("attribute_" + std::to_string(attribute_i) + "[]");
    r_resources.append({first_slot + attribute_i,
                        HaShaderQualifier::Read,
                        ha_attribute_buffer_glsl(attribute.type),
                        r_names.last().c_str()});
  }
  return true;
}

/**
 * `GeometryNodeStoreNamedAttribute`: the value field is written back after the position steps and
 * after the structural element nodes. POINT stores use the position kernel; FACE / EDGE / CORNER
 * stores become their own element pass.
 */
bool compile_store(CompileContext &ctx,
                   const bNode &node,
                   HaStore &r_store,
                   HaElementStore &r_element_store,
                   bool &r_is_element)
{
  const auto &storage = *static_cast<const NodeGeometryStoreNamedAttribute *>(node.storage);
  HaType type;
  switch (storage.data_type) {
    case CD_PROP_FLOAT:
      type = HaType::Float;
      break;
    case CD_PROP_INT32:
      type = HaType::Int;
      break;
    case CD_PROP_FLOAT2:
      type = HaType::Float2;
      break;
    case CD_PROP_FLOAT3:
      type = HaType::Float3;
      break;
    case CD_PROP_COLOR:
      type = HaType::Color;
      break;
    default:
      return fail(ctx, "Store Named Attribute: this attribute type is not supported");
  }
  const int components = ha_type_components(type);
  if (node.input_sockets().size() < 4) {
    return fail(ctx, "Store Named Attribute: unexpected sockets");
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

  const bke::AttrDomain domain = bke::AttrDomain(storage.domain);
  if (domain == bke::AttrDomain::Point) {
    /* The POINT store kernel does not bind `user_params[]` (it already uses slots 0..3 + the
     * attributes), so an animated constant falls back to the CPU. */
    const bool old_allow_params = ctx.allow_params;
    ctx.allow_params = false;
    const HaExpr *selection = compile_input_as(ctx, node, 1, HaType::Float);
    const HaExpr *compiled = compile_input_as(ctx, node, 3, type);
    ctx.allow_params = old_allow_params;
    if (!ctx.ok) {
      return false;
    }
    if (compiled->type != type) {
      return fail(ctx, "Store Named Attribute: the value has an unsupported type");
    }
    r_store.name = name;
    r_store.type = type;
    r_store.components = components;
    r_store.glsl_value = ha_emit_glsl_natural(*compiled);
    r_store.glsl_selection = ha_emit_glsl_scalar(*selection);
    r_is_element = false;
    return true;
  }

  HaElementDomain element_domain;
  if (domain == bke::AttrDomain::Face) {
    element_domain = HaElementDomain::Face;
  }
  else if (domain == bke::AttrDomain::Edge) {
    element_domain = HaElementDomain::Edge;
  }
  else if (domain == bke::AttrDomain::Corner) {
    element_domain = HaElementDomain::Corner;
  }
  else {
    return fail(ctx, "Store Named Attribute: only the POINT/FACE/EDGE/CORNER domains are supported");
  }

  /* Compile the value and the selection on the target domain: `Named Attribute` reads that
   * domain, and `Position`/`Attribute Statistic` are refused there. */
  HaExprPool element_pool;
  Vector<HaAggregate> element_aggregates;
  CompileContext element_ctx{
      element_pool, element_aggregates, r_element_store.attributes, ctx.params};
  element_ctx.allow_params = false;
  element_ctx.zone_input_node = ctx.zone_input_node;
  element_ctx.tree = ctx.tree;
  element_ctx.element_domain = true;
  const HaExpr *selection = compile_input_as(element_ctx, node, 1, HaType::Float);
  const HaExpr *compiled = compile_input_as(element_ctx, node, 3, type);
  if (!element_ctx.ok) {
    return fail(ctx,
                element_ctx.reason.empty() ? std::string("the field is not supported") :
                                             element_ctx.reason);
  }
  if (selection->type != HaType::Float || compiled->type != type) {
    return fail(ctx, "Store Named Attribute: the value or selection has an unsupported type");
  }

  r_element_store.name = name;
  r_element_store.domain = element_domain;
  r_element_store.type = type;
  r_element_store.components = components;
  r_element_store.glsl_value = ha_emit_glsl_natural(*compiled);
  r_element_store.glsl_selection = ha_emit_glsl_scalar(*selection);
  r_element_store.default_value = 0.0f;
  r_is_element = true;
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
    args.append(compile_input_as(ctx, node, base + i, type));
  }
  if (interpolation == HaParam::MapRangeStepped) {
    args.append(compile_input_as(ctx, node, base + 5, type));
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
  if (ctx.element_domain) {
    /* The reduction reads the POINT position buffer, which an element pass does not bind. */
    fail(ctx, "Attribute Statistic: not available on FACE/EDGE element domains");
    return &ctx.pool.constant(0.0f);
  }
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

  /* `Attribute` is input socket 2. The reduction kernel does not bind `user_params[]`. */
  const bool old_allow_params = ctx.allow_params;
  ctx.allow_params = false;
  const HaExpr *value = compile_input_as(ctx, node, 2, HaType::Float);
  ctx.allow_params = old_allow_params;
  if (!ctx.ok || value == nullptr || value->type != HaType::Float) {
    if (ctx.ok) {
      fail(ctx, "Attribute Statistic: only scalar attributes are supported");
    }
    return &ctx.pool.constant(0.0f);
  }

  HaExpr &read = ctx.pool.add(HaType::Float, HaOp::AggregateRead);
  read.aggregate_slot = int(ctx.aggregates.size());
  /* The reduction kernel stores the value in a `float`, so the expression must not be promoted to a
   * vector here (`ha_emit_glsl` would wrap a scalar in `vec3(...)`, which does not compile). */
  ctx.aggregates.append({ha_emit_glsl_scalar(*value), op});
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
    if (ctx.element_domain) {
      /* A face does not have a position of its own; Blender would use the face center, which needs
       * the corner topology. Refusing keeps the device result from silently differing. */
      fail(ctx, "the 'Position' field is not available on FACE/EDGE element domains");
      return &ctx.pool.constant(0.0f);
    }
    result = &ctx.pool.add(HaType::Float3, HaOp::Position);
  }
  else if (node.is_type("GeometryNodeInputIndex"_ustr)) {
    result = &ctx.pool.add(HaType::Int, HaOp::Index);
  }
  else if (node.is_type("ShaderNodeMath"_ustr)) {
    result = compile_math(ctx, node);
  }
  else if (node.is_type("FunctionNodeIntegerMath"_ustr)) {
    result = compile_integer_math(ctx, node);
  }
  else if (node.is_type("ShaderNodeVectorMath"_ustr)) {
    result = compile_vector_math(ctx, node);
  }
  else if (node.is_type("ShaderNodeCombineXYZ"_ustr)) {
    const HaExpr *x = compile_input_as(ctx, node, 0, HaType::Float);
    const HaExpr *y = compile_input_as(ctx, node, 1, HaType::Float);
    const HaExpr *z = compile_input_as(ctx, node, 2, HaType::Float);
    if (x->type != HaType::Float || y->type != HaType::Float || z->type != HaType::Float) {
      /* Blender converts a linked vector into a scalar here (averaging the components); the IR has
       * no such conversion, and emitting it would produce `vec3(vec3, 0, 0)` in GLSL. */
      fail(ctx, "Combine XYZ: the components have to be scalars");
      result = &ctx.pool.constant(float3(0.0f));
    }
    else {
      result = &ctx.pool.add(HaType::Float3, HaOp::Combine, {x, y, z});
    }
  }
  else if (node.is_type("ShaderNodeSeparateXYZ"_ustr)) {
    const HaOp op = socket_index == 0 ? HaOp::SplitX :
                    socket_index == 1 ? HaOp::SplitY :
                                        HaOp::SplitZ;
    result = &ctx.pool.add(
        HaType::Float, op, {compile_input_as(ctx, node, 0, HaType::Float3)});
  }
  else if (node.is_type("ShaderNodeClamp"_ustr)) {
    /* `custom1` is the clamp type. Both modes read the `Min`/`Max` sockets: `Range` only swaps them
     * when they are in the wrong order (see `node_shader_clamp.cc`), it does *not* clamp to 0..1. */
    const HaExpr *value = compile_input_as(ctx, node, 0, HaType::Float);
    const HaExpr *min_value = compile_input_as(ctx, node, 1, HaType::Float);
    const HaExpr *max_value = compile_input_as(ctx, node, 2, HaType::Float);
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
  else if (node.is_type("ShaderNodeTexWhiteNoise"_ustr)) {
    result = compile_white_noise(ctx, node, socket_index);
  }
  else if (node.is_type("ShaderNodeTexNoise"_ustr)) {
    result = compile_noise_texture(ctx, node, socket_index);
  }
  else if (node.is_type("ShaderNodeTexWave"_ustr)) {
    result = compile_wave_texture(ctx, node, socket_index);
  }
  else if (node.is_type("ShaderNodeTexVoronoi"_ustr)) {
    result = compile_voronoi_texture(ctx, node, socket_index);
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
    if (ctx.tree != nullptr && socket_value_is_animated(*ctx.tree, socket)) {
      const bNode *node = socket.runtime->owner_node;
      if (!ctx.allow_params || node == nullptr) {
        fail(ctx,
             fmt::format("node '{}': input '{}' is animated by a driver, which is not supported in "
                         "this kernel",
                         node != nullptr ? node->name : "",
                         socket.name));
        return &ctx.pool.constant(0.0f);
      }
      HaType type;
      switch (socket.type) {
        case SOCK_FLOAT:
        case SOCK_BOOLEAN:
          type = HaType::Float;
          break;
        case SOCK_INT:
          type = HaType::Int;
          break;
        case SOCK_VECTOR:
          type = HaType::Float3;
          break;
        case SOCK_RGBA:
          type = HaType::Color;
          break;
        default:
          fail(ctx, fmt::format("animated input '{}' has an unsupported socket type", socket.name));
          return &ctx.pool.constant(0.0f);
      }
      const int slot = int(ctx.params.size());
      ctx.params.append({node->name, socket.index(), type});
      HaExpr &read = ctx.pool.add(type, HaOp::ParamRead);
      read.param_slot = slot;
      return &read;
    }
    switch (socket.type) {
      case SOCK_FLOAT:
        return &ctx.pool.constant(read_float_default(socket));
      case SOCK_VECTOR:
        return &ctx.pool.constant(read_vector_default(socket));
      case SOCK_RGBA:
        return &ctx.pool.constant_color(read_color_default(socket));
      case SOCK_INT:
        return &ctx.pool.constant(read_int_default(socket));
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

std::unique_ptr<HaZoneProgram> HaZoneProgram::try_build(const bNodeTree &tree,
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
  Vector<const bNode *> element_nodes;
  Vector<const bNode *> corner_nodes;
  while (true) {
    if (current == input_node) {
      break;
    }
    if (chain.size() >= MAX_CHAIN_STEPS) {
      return fail(fmt::format("the zone contains more than {} 'Set Position' nodes in a row",
                              MAX_CHAIN_STEPS));
    }
    /* Nodes that write an attribute on a domain without positions are a pass-through for the
     * geometry; they become their own device pass. */
    if (current->is_type("GeometryNodeSetShadeSmooth"_ustr) ||
        current->is_type("GeometryNodeSetMaterial"_ustr) ||
        current->is_type("GeometryNodeFlipFaces"_ustr)) {
      if (element_nodes.size() + corner_nodes.size() >= MAX_ELEMENT_STEPS) {
        return fail(fmt::format("the zone contains more than {} element passes",
                                MAX_ELEMENT_STEPS));
      }
      if (current->is_type("GeometryNodeFlipFaces"_ustr)) {
        corner_nodes.append(current);
      }
      else {
        element_nodes.append(current);
      }
      const bNodeSocket &geometry_input = current->input_socket(0);
      if (geometry_input.link == nullptr) {
        return fail(fmt::format("node '{}': the geometry input is not connected", current->name));
      }
      current = geometry_input.link->fromnode;
      if (current == input_node) {
        break;
      }
      continue;
    }
    if (!current->is_type("GeometryNodeSetPosition"_ustr) &&
        !current->is_type("GeometryNodeTransform"_ustr))
    {
      return fail(fmt::format("node '{}' ({}) is not supported", current->name, current->idname));
    }
    if (current->input_sockets().size() < 4 || current->output_sockets().is_empty()) {
      return fail(fmt::format("'{}' has unexpected sockets", current->name));
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
  CompileContext ctx{pool, program->aggregates_, program->attributes_, program->params_};
  ctx.zone_input_node = input_node;
  ctx.tree = &tree;

  /* Compile first step -> last step (`chain` is stored last -> first). */
  for (const int i : chain.index_range()) {
    const bNode &node = *chain[chain.size() - 1 - i];

    if (node.is_type("GeometryNodeTransform"_ustr)) {
      const HaExpr *transformed = compile_transform(ctx, node, pool);
      if (transformed == nullptr) {
        return fail(ctx.reason.empty() ? std::string("the transform is not supported") : ctx.reason);
      }
      program->steps_.append(
          {ha_emit_glsl(*transformed), ha_emit_glsl_scalar(pool.constant(1.0f))});
      continue;
    }

    /* `Set Position`: Geometry, Selection, Position, Offset.
     *
     * - `Position` unlinked means "keep the current position" (the socket's default input type is
     *   the position field), so the base of the expression is the element position. In a chain the
     *   "current position" is the previous step's result, which is exactly what the kernel reads.
     * - `Selection` defaults to true; the node writes `Position + Offset` for selected elements
     *   only, so the kernel picks between the old and the new position instead of skipping the
     *   write. */
    const HaExpr *base = node.input_socket(2).link != nullptr ?
                             compile_input_as(ctx, node, 2, HaType::Float3) :
                             &pool.add(HaType::Float3, HaOp::Position);
    const HaExpr *offset = compile_input_as(ctx, node, 3, HaType::Float3);
    const HaExpr *selection = compile_input_as(ctx, node, 1, HaType::Float);
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

  /* Then the FACE/EDGE passes, first -> last (`element_nodes` is stored last -> first). */
  for (const int i : element_nodes.index_range()) {
    const bNode &element_node = *element_nodes[element_nodes.size() - 1 - i];
    HaElementStep step;
    const bool ok = element_node.is_type("GeometryNodeSetMaterial"_ustr) ?
                        compile_set_material(ctx, element_node, step) :
                        compile_set_shade_smooth(ctx, element_node, step);
    if (!ok) {
      return fail(ctx.reason.empty() ? std::string("the element node is not supported") :
                                      ctx.reason);
    }
    if (!step.attribute.empty()) {
      program->element_steps_.append(std::move(step));
    }
  }

  /* Then the corner passes, first -> last (`corner_nodes` is stored last -> first). */
  for (const int i : corner_nodes.index_range()) {
    HaCornerStep step;
    if (!compile_flip_faces(ctx, *corner_nodes[corner_nodes.size() - 1 - i], step)) {
      return fail(ctx.reason.empty() ? std::string("the corner node is not supported") :
                                      ctx.reason);
    }
    program->corner_steps_.append(std::move(step));
  }

  /* A zone that only contains element passes still needs one position step: the chain is what
   * carries the geometry through the device buffers (and the stores read its result), so insert an
   * identity step instead of falling back. */
  if (program->steps_.is_empty()) {
    program->steps_.append({"(position + vec3(0.0))", "1.0"});
  }

  /* Then the stores, first -> last (`store_nodes` is stored last -> first). */
  for (const int i : store_nodes.index_range()) {
    HaStore store;
    HaElementStore element_store;
    bool is_element = false;
    if (!compile_store(ctx,
                       *store_nodes[store_nodes.size() - 1 - i],
                       store,
                       element_store,
                       is_element))
    {
      return fail(ctx.reason.empty() ? std::string("the store is not supported") : ctx.reason);
    }
    if (is_element) {
      program->element_stores_.append(std::move(element_store));
    }
    else {
      program->stores_.append(std::move(store));
    }
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
  for (const HaElementStep &step : element_steps_) {
    if (!result.empty()) {
      result += " -> ";
    }
    result += fmt::format("[{}] = {} where {} > 0",
                          step.attribute,
                          step.glsl_value,
                          step.glsl_selection);
  }
  return result;
}

bke::AttrDomain ha_element_domain_to_attr(const HaElementDomain domain)
{
  switch (domain) {
    case HaElementDomain::Face:
      return bke::AttrDomain::Face;
    case HaElementDomain::Edge:
      return bke::AttrDomain::Edge;
    case HaElementDomain::Corner:
      return bke::AttrDomain::Corner;
  }
  return bke::AttrDomain::Point;
}

/** Downloads \a buffer and writes it into \a dst as the given attribute type and domain. */
bool write_attribute_from_buffer(bke::MutableAttributeAccessor &dst,
                                 const StringRef name,
                                 const HaType type,
                                 const bke::AttrDomain domain,
                                 const int count,
                                 const int padded,
                                 const HaBuffer &buffer)
{
  if (type == HaType::Int) {
    Array<int> host(padded);
    buffer.download(MutableSpan<float>(reinterpret_cast<float *>(host.data()), padded));
    bke::SpanAttributeWriter<int> writer = dst.lookup_or_add_for_write_span<int>(name, domain);
    if (!writer) {
      return false;
    }
    writer.span.copy_from(host.as_span().take_front(count));
    writer.finish();
    return true;
  }

  const int components = ha_type_components(type);
  Array<float> host(int64_t(padded) * components);
  buffer.download(host);
  switch (type) {
    case HaType::Float: {
      bke::SpanAttributeWriter<float> writer = dst.lookup_or_add_for_write_span<float>(name, domain);
      if (!writer) {
        return false;
      }
      writer.span.copy_from(host.as_span().take_front(count));
      writer.finish();
      break;
    }
    case HaType::Float2: {
      bke::SpanAttributeWriter<float2> writer = dst.lookup_or_add_for_write_span<float2>(name,
                                                                                         domain);
      if (!writer) {
        return false;
      }
      for (const int element : IndexRange(count)) {
        writer.span[element] = float2(host[element * 2], host[element * 2 + 1]);
      }
      writer.finish();
      break;
    }
    case HaType::Float3: {
      bke::SpanAttributeWriter<float3> writer = dst.lookup_or_add_for_write_span<float3>(name,
                                                                                         domain);
      if (!writer) {
        return false;
      }
      writer.span.copy_from(Span<float3>(reinterpret_cast<const float3 *>(host.data()), count));
      writer.finish();
      break;
    }
    case HaType::Color: {
      bke::SpanAttributeWriter<ColorGeometry4f> writer =
          dst.lookup_or_add_for_write_span<ColorGeometry4f>(name, domain);
      if (!writer) {
        return false;
      }
      for (const int element : IndexRange(count)) {
        writer.span[element] = ColorGeometry4f(host[element * 4 + 0],
                                               host[element * 4 + 1],
                                               host[element * 4 + 2],
                                               host[element * 4 + 3]);
      }
      writer.finish();
      break;
    }
    case HaType::Int:
      break;
  }
  return true;
}

bool HaZoneProgram::evaluate(const bke::GeometrySet &input,
                             bke::GeometrySet &r_output,
                             int *r_elements,
                             const bNodeTree *tree) const
{
  /* The zone works on every component that has a POINT domain: meshes, point clouds and curves. The
   * FACE / EDGE / CORNER passes only exist on meshes (they are refused below). */
  const Mesh *src_mesh = input.get_mesh();
  const PointCloud *src_pointcloud = src_mesh == nullptr ? input.get_pointcloud() : nullptr;
  const Curves *src_curves = (src_mesh == nullptr && src_pointcloud == nullptr) ?
                                 input.get_curves() :
                                 nullptr;
  Span<float3> positions;
  std::optional<bke::AttributeAccessor> input_attributes_storage;
  if (src_mesh != nullptr) {
    positions = src_mesh->vert_positions();
    input_attributes_storage.emplace(src_mesh->attributes());
  }
  else if (src_pointcloud != nullptr) {
    positions = src_pointcloud->positions();
    input_attributes_storage.emplace(src_pointcloud->attributes());
  }
  else if (src_curves != nullptr) {
    const bke::CurvesGeometry &geometry = src_curves->geometry.wrap();
    positions = geometry.positions();
    input_attributes_storage.emplace(geometry.attributes());
  }
  else {
    return false;
  }
  const bke::AttributeAccessor &input_attributes = *input_attributes_storage;
  const int element_count = int(positions.size());
  if (element_count <= 0 || steps_.is_empty()) {
    return false;
  }
  if (src_mesh == nullptr &&
      (!element_steps_.is_empty() || !element_stores_.is_empty() || !corner_steps_.is_empty())) {
    /* Those passes write FACE / EDGE / CORNER data, which only a mesh has. */
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
    for (const int i : attributes_.index_range()) {
      const HaAttribute &attribute = attributes_[i];
      if (!upload_one_attribute(input_attributes,
                                attribute,
                                bke::AttrDomain::Point,
                                element_count,
                                padded_count,
                                attribute_buffers[i]))
      {
        return false;
      }
      attribute_names.append("attribute_" + std::to_string(i) + "[]");
      extra_resources.append({3 + i,
                              HaShaderQualifier::Read,
                              ha_attribute_buffer_glsl(attribute.type),
                              attribute_names.last().c_str()});
    }
  }

  /* The generated source does not spell out the attribute buffer types (they are only in the
   * resource list), so the shader key has to include them. Otherwise an int and a float attribute
   * at the same slot would share a cached shader with the wrong declaration. */
  std::string attribute_key;
  for (const HaAttribute &attribute : attributes_) {
    attribute_key += "|";
    attribute_key += ha_attribute_buffer_glsl(attribute.type);
  }

  /* Animated socket defaults: one 4-float slot per parameter, uploaded every evaluation. The
   * shader source only contains `user_params[slot * 4 + component]`, so a changing value never
   * rebuilds the shader. Slot 15 is reserved for this buffer; keep the attribute count below 13. */
  HaBuffer params_buffer;
  Vector<HaShaderResource> position_resources;
  position_resources.reserve(extra_resources.size() + 1);
  for (const HaShaderResource &resource : extra_resources) {
    position_resources.append(resource);
  }
  std::string param_key;
  if (!params_.is_empty()) {
    if (attributes_.size() > 12) {
      return false;
    }
    Array<float> host_params(int64_t(params_.size()) * 4, 0.0f);
    for (const int i : params_.index_range()) {
      const HaParamSource &param = params_[i];
      const bNode *node = tree != nullptr ?
                              bke::node_find_node_by_name(const_cast<bNodeTree &>(*tree),
                                                          param.node_name) :
                              nullptr;
      const bNodeSocket *socket =
          node != nullptr && param.socket_index >= 0 &&
                  param.socket_index < node->input_sockets().size() ?
              &node->input_socket(param.socket_index) :
              nullptr;
      if (socket == nullptr) {
        return false;
      }
      switch (param.type) {
        case HaType::Float:
          host_params[i * 4] = read_float_default(*socket);
          break;
        case HaType::Int: {
          const int value = read_int_default(*socket);
          std::memcpy(&host_params[i * 4], &value, sizeof(int));
          break;
        }
        case HaType::Float2: {
          const float3 value = read_vector_default(*socket);
          host_params[i * 4] = value.x;
          host_params[i * 4 + 1] = value.y;
          break;
        }
        case HaType::Float3: {
          const float3 value = read_vector_default(*socket);
          host_params[i * 4] = value.x;
          host_params[i * 4 + 1] = value.y;
          host_params[i * 4 + 2] = value.z;
          break;
        }
        case HaType::Color: {
          const float4 value = read_color_default(*socket);
          host_params[i * 4] = value.x;
          host_params[i * 4 + 1] = value.y;
          host_params[i * 4 + 2] = value.z;
          host_params[i * 4 + 3] = value.w;
          break;
        }
      }
    }
    if (!params_buffer.ensure(int64_t(params_.size()) * 4)) {
      return false;
    }
    params_buffer.upload(host_params);
    position_resources.append({15, HaShaderQualifier::Read, "float", "user_params[]"});
    param_key = "|params";
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
    gpu::Shader *shader = ha_shader_get(
        step.glsl_position + '#' + step.glsl_selection + attribute_key + param_key,
        step.glsl_position,
        step.glsl_selection,
        Span<const HaShaderResource>(position_resources.data(), int64_t(position_resources.size())));
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
    if (!params_.is_empty()) {
      GPU_storagebuf_bind(params_buffer.buf(), 15);
    }
    ha_dispatch(shader, padded_count);
    if (!params_.is_empty()) {
      GPU_storagebuf_unbind(params_buffer.buf());
    }
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

  /* The stores run after the structural element passes, so a store that follows a `Set Shade
   * Smooth` / `Flip Faces` in the graph sees the updated attributes. See below. */

  Array<float> host_output(padded_count * 3);
  result.download(host_output);
  const double download_ms = timer.lap();
  ha_program_debug("downloaded");

  bke::GeometrySet output = input;
  Mesh *mesh = nullptr;
  ID *dst_id = nullptr;
  std::optional<bke::MutableAttributeAccessor> dst_attributes_storage;
  const Span<float3> new_positions(reinterpret_cast<const float3 *>(host_output.data()),
                                   element_count);
  if (src_mesh != nullptr) {
    mesh = BKE_mesh_copy_for_eval(*src_mesh);
    mesh->vert_positions_for_write().copy_from(new_positions);
    /* Positions changed, so the derived normals have to be recomputed. */
    mesh->tag_positions_changed();
    dst_id = &mesh->id;
    dst_attributes_storage.emplace(mesh->attributes_for_write());
    output.replace_mesh(mesh);
  }
  else if (src_pointcloud != nullptr) {
    PointCloud *pointcloud = BKE_pointcloud_copy_for_eval(src_pointcloud);
    pointcloud->positions_for_write().copy_from(new_positions);
    pointcloud->tag_positions_changed();
    dst_id = &pointcloud->id;
    dst_attributes_storage.emplace(pointcloud->attributes_for_write());
    output.replace_pointcloud(pointcloud);
  }
  else {
    Curves *curves = BKE_curves_copy_for_eval(src_curves);
    bke::CurvesGeometry &geometry = curves->geometry.wrap();
    geometry.positions_for_write().copy_from(new_positions);
    geometry.tag_positions_changed();
    dst_id = &curves->id;
    dst_attributes_storage.emplace(geometry.attributes_for_write());
    output.replace_curves(curves);
  }
  bke::MutableAttributeAccessor dst_attributes = *dst_attributes_storage;

  /* The stores are written back after the structural element passes (see below). */

  /* The FACE / EDGE passes: they do not touch positions, so they run after the chain and before the
   * stores, writing straight into the result mesh. */
  if (!element_steps_.is_empty()) {
    const bke::AttributeAccessor &src_attributes = input_attributes;
    bke::MutableAttributeAccessor mesh_attributes = mesh->attributes_for_write();
    for (const HaElementStep &step : element_steps_) {
      const bke::AttrDomain domain = step.domain == HaElementDomain::Face ? bke::AttrDomain::Face :
                                       step.domain == HaElementDomain::Edge ? bke::AttrDomain::Edge :
                                                                              bke::AttrDomain::Corner;
      const int count = int(mesh_attributes.domain_size(domain));
      if (count <= 0) {
        continue;
      }
      const int padded = (count + local_size - 1) / local_size * local_size;
      HaBuffer element_in;
      HaBuffer element_out;
      HaBuffer element_params;
      if (!element_in.ensure(int64_t(padded) * step.components) ||
          !element_out.ensure(int64_t(padded) * step.components) || !element_params.ensure(4))
      {
        return false;
      }
      Array<float> host_in(int64_t(padded) * step.components, 0.0f);
      bool attribute_exists = false;
      if (const std::optional<bke::AttributeMetaData> meta =
              src_attributes.lookup_meta_data(step.attribute))
      {
        const bool type_matches = step.type == HaElementType::Int ?
                                      meta->data_type == bke::AttrType::Int32 :
                                      meta->data_type == bke::AttrType::Bool;
        if (meta->domain == domain && type_matches) {
          attribute_exists = true;
          if (step.type == HaElementType::Int) {
            const bke::AttributeReader<int> reader = src_attributes.lookup<int>(step.attribute,
                                                                               domain);
            if (reader) {
              const VArray<int> &values = reader.varray;
              for (const int element : IndexRange(count)) {
                host_in[element] = float(values[element]);
              }
            }
          }
          else {
            const bke::AttributeReader<bool> reader = src_attributes.lookup<bool>(step.attribute,
                                                                                 domain);
            if (reader) {
              const VArray<bool> &values = reader.varray;
              for (const int element : IndexRange(count)) {
                host_in[element] = values[element] ? 1.0f : 0.0f;
              }
            }
          }
        }
      }
      if (!attribute_exists) {
        /* The CPU creates the attribute and leaves unselected elements at its default (which is
         * `true` for `sharp_face` / `sharp_edge`), so the pass has to start from there. */
        std::fill_n(host_in.begin(), count, step.default_value);
      }
      element_in.upload(host_in);

      /* `Set Material`: the value is the material's slot index on the result mesh, which is only
       * known now. This mirrors `assign_material_to_id_geometry`: a partial selection gets a default
       * slot first (so unselected faces keep a valid index), then the material is looked up and
       * appended if it is not there yet. */
      std::string value_expression = step.glsl_value;
      if (step.material_step) {
        if (!step.all_selected) {
          BKE_id_material_eval_ensure_default_slot(&mesh->id);
        }
        int new_index = -1;
        const int material_count = *BKE_id_material_len_p(&mesh->id);
        if (Material **materials = *BKE_id_material_array_p(&mesh->id)) {
          for (int material_i = 0; material_i < material_count; material_i++) {
            if (materials[material_i] == step.material) {
              new_index = material_i;
              break;
            }
          }
        }
        if (new_index == -1) {
          new_index = material_count;
          BKE_id_material_eval_assign(&mesh->id, new_index + 1, step.material);
        }
        value_expression = std::to_string(new_index) + ".0";
      }

      /* The attributes this step's fields read, taken from the same domain of the source mesh. */
      Array<HaBuffer> step_attribute_buffers(step.attributes.size());
      Vector<std::string> step_attribute_names;
      Vector<HaShaderResource> step_resources;
      if (!upload_domain_attributes(src_attributes,
                                    Span<const HaAttribute>(step.attributes.data(),
                                                            step.attributes.size()),
                                    domain,
                                    count,
                                    padded,
                                    step_attribute_buffers,
                                    step_attribute_names,
                                    step_resources,
                                    3)) {
        return false;
      }

      const std::array<float, 4> element_params_host{float(count), 0.0f, 0.0f, 0.0f};
      element_params.upload(element_params_host);

      std::string step_attribute_key;
      for (const HaAttribute &attribute : step.attributes) {
        step_attribute_key += "|";
        step_attribute_key += ha_attribute_buffer_glsl(attribute.type);
      }
      gpu::Shader *element_shader = ha_shader_get_element(
          "element|" + value_expression + "|" + step.glsl_selection + "|" +
              std::to_string(step.components) + step_attribute_key,
          value_expression,
          step.glsl_selection,
          step.components,
          Span<const HaShaderResource>(step_resources.data(), int64_t(step_resources.size())));
      if (element_shader == nullptr) {
        return false;
      }
      GPU_shader_bind(element_shader);
      GPU_storagebuf_bind(element_in.buf(), 0);
      GPU_storagebuf_bind(element_out.buf(), 1);
      GPU_storagebuf_bind(element_params.buf(), 2);
      for (const int attribute_i : step.attributes.index_range()) {
        GPU_storagebuf_bind(step_attribute_buffers[attribute_i].buf(), 3 + attribute_i);
      }
      ha_dispatch(element_shader, padded);
      for (const int attribute_i : step.attributes.index_range()) {
        GPU_storagebuf_unbind(step_attribute_buffers[attribute_i].buf());
      }
      GPU_storagebuf_unbind(element_params.buf());
      GPU_storagebuf_unbind(element_out.buf());
      GPU_storagebuf_unbind(element_in.buf());
      GPU_shader_unbind();
      GPU_finish();

      Array<float> host_out(int64_t(padded) * step.components);
      element_out.download(host_out);

      if (step.type == HaElementType::Int) {
        bke::SpanAttributeWriter<int> writer = mesh_attributes.lookup_or_add_for_write_span<int>(
            step.attribute, domain);
        if (!writer) {
          BKE_id_free(nullptr, mesh);
          return false;
        }
        for (const int element : IndexRange(count)) {
          writer.span[element] = int(std::lround(host_out[element]));
        }
        writer.finish();
      }
      else {
        bke::SpanAttributeWriter<bool> writer = mesh_attributes.lookup_or_add_for_write_span<bool>(
            step.attribute, domain);
        if (!writer) {
          BKE_id_free(nullptr, mesh);
          return false;
        }
        for (const int element : IndexRange(count)) {
          writer.span[element] = host_out[element] > 0.5f;
        }
        writer.finish();
      }
    }
  }

  /* The `Flip Faces` passes: they rewrite the corner topology of the result mesh. */
  if (!corner_steps_.is_empty()) {
    const bke::AttributeAccessor &src_attributes = input_attributes;
    /* `bke::mesh_flip_faces` also reverses every non-single CORNER attribute; this pass only handles
     * the corner indices, so a mesh with such an attribute has to stay on the CPU. */
    bool corner_attributes_unsupported = false;
    src_attributes.foreach_attribute([&](const bke::AttributeIter &iter) {
      if (iter.domain != bke::AttrDomain::Corner) {
        return;
      }
      if (ELEM(StringRef(iter.name), ".corner_vert", ".corner_edge")) {
        return;
      }
      if (iter.data_type == bke::AttrType::String) {
        return;
      }
      if (iter.storage_type == bke::AttrStorageType::Single) {
        return;
      }
      corner_attributes_unsupported = true;
    });
    if (corner_attributes_unsupported) {
      BKE_id_free(nullptr, mesh);
      return false;
    }
    const int face_count = int(src_mesh->faces_num);
    const int corner_count = int(src_mesh->corners_num);
    if (face_count > 0 && corner_count > 0) {
      const int padded_faces = (face_count + local_size - 1) / local_size * local_size;
      const OffsetIndices<int> faces = src_mesh->faces();
      const Span<int> face_offsets = faces.data();
      for (const HaCornerStep &step : corner_steps_) {
        HaBufferInt corner_verts_in;
        HaBufferInt corner_verts_out;
        HaBufferInt corner_edges_in;
        HaBufferInt corner_edges_out;
        HaBufferInt face_offsets_buffer;
        HaBuffer corner_params_buffer;
        if (!corner_verts_in.ensure(corner_count) || !corner_verts_out.ensure(corner_count) ||
            !corner_edges_in.ensure(corner_count) || !corner_edges_out.ensure(corner_count) ||
            !face_offsets_buffer.ensure(int64_t(face_offsets.size())) ||
            !corner_params_buffer.ensure(4))
        {
          BKE_id_free(nullptr, mesh);
          return false;
        }
        corner_verts_in.upload(src_mesh->corner_verts());
        corner_edges_in.upload(src_mesh->corner_edges());
        face_offsets_buffer.upload(face_offsets);
        const std::array<float, 4> corner_params{float(face_count), 0.0f, 0.0f, 0.0f};
        corner_params_buffer.upload(corner_params);

        Array<HaBuffer> step_attribute_buffers(step.attributes.size());
        Vector<std::string> step_attribute_names;
        Vector<HaShaderResource> step_resources;
        if (!upload_domain_attributes(src_attributes,
                                      Span<const HaAttribute>(step.attributes.data(),
                                                              step.attributes.size()),
                                      bke::AttrDomain::Face,
                                      face_count,
                                      padded_faces,
                                      step_attribute_buffers,
                                      step_attribute_names,
                                      step_resources,
                                      6))
        {
          BKE_id_free(nullptr, mesh);
          return false;
        }

        gpu::Shader *corner_shader = ha_shader_get_flip_faces(
            "flip|" + step.glsl_selection,
            step.glsl_selection,
            Span<const HaShaderResource>(step_resources.data(), int64_t(step_resources.size())));
        if (corner_shader == nullptr) {
          BKE_id_free(nullptr, mesh);
          return false;
        }
        GPU_shader_bind(corner_shader);
        GPU_storagebuf_bind(corner_verts_in.buf(), 0);
        GPU_storagebuf_bind(corner_verts_out.buf(), 1);
        GPU_storagebuf_bind(corner_edges_in.buf(), 2);
        GPU_storagebuf_bind(corner_edges_out.buf(), 3);
        GPU_storagebuf_bind(face_offsets_buffer.buf(), 4);
        GPU_storagebuf_bind(corner_params_buffer.buf(), 5);
        for (const int attribute_i : step.attributes.index_range()) {
          GPU_storagebuf_bind(step_attribute_buffers[attribute_i].buf(), 6 + attribute_i);
        }
        ha_dispatch(corner_shader, padded_faces);
        for (const int attribute_i : step.attributes.index_range()) {
          GPU_storagebuf_unbind(step_attribute_buffers[attribute_i].buf());
        }
        GPU_storagebuf_unbind(corner_params_buffer.buf());
        GPU_storagebuf_unbind(face_offsets_buffer.buf());
        GPU_storagebuf_unbind(corner_edges_out.buf());
        GPU_storagebuf_unbind(corner_edges_in.buf());
        GPU_storagebuf_unbind(corner_verts_out.buf());
        GPU_storagebuf_unbind(corner_verts_in.buf());
        GPU_shader_unbind();
        GPU_finish();

        Array<int> host_verts(corner_count);
        Array<int> host_edges(corner_count);
        corner_verts_out.download(host_verts);
        corner_edges_out.download(host_edges);
        mesh->corner_verts_for_write().copy_from(host_verts);
        mesh->corner_edges_for_write().copy_from(host_edges);
        mesh->tag_face_winding_changed();
      }
    }
  }

  /* The stores: POINT stores use the final position buffer, element stores run as their own pass.
   * Both read the result mesh, so they see the structural element passes above. */
  if (!stores_.is_empty()) {
    Array<HaBuffer> store_buffers(stores_.size());
    Array<HaBuffer> store_in_buffers(stores_.size());
    for (const int i : stores_.index_range()) {
      const HaStore &store = stores_[i];
      const bool with_old = !store.glsl_selection.empty();
      if (ha_program_debug_enabled() && with_old) {
        fprintf(stderr, "HA program: store '%s' with selection\n", store.name.c_str());
      }
      if (!store_buffers[i].ensure(int64_t(padded_count) * store.components)) {
        return false;
      }
      if (with_old) {
        const HaAttribute attribute{store.name, store.type, store.components};
        if (!upload_one_attribute(dst_attributes,
                                  attribute,
                                  bke::AttrDomain::Point,
                                  element_count,
                                  padded_count,
                                  store_in_buffers[i]))
        {
          return false;
        }
      }

      Vector<HaShaderResource> store_resources;
      store_resources.reserve(attributes_.size());
      for (const int attribute_i : attributes_.index_range()) {
        store_resources.append({(with_old ? 4 : 3) + attribute_i,
                                HaShaderQualifier::Read,
                                ha_attribute_buffer_glsl(attributes_[attribute_i].type),
                                attribute_names[attribute_i].c_str()});
      }

      gpu::Shader *shader = ha_shader_get_store(
          "store|" + store.glsl_value + "|" + ha_type_glsl(store.type) +
              (with_old ? "|sel" : "") + attribute_key,
          store.glsl_value,
          store.glsl_selection,
          with_old,
          ha_type_glsl(store.type),
          store.components,
          Span<const HaShaderResource>(store_resources.data(), int64_t(store_resources.size())));
      if (shader == nullptr) {
        return false;
      }
      GPU_shader_bind(shader);
      GPU_storagebuf_bind(result.buf(), 0);
      GPU_storagebuf_bind(store_buffers[i].buf(), 1);
      GPU_storagebuf_bind(buffer_aggregate.buf(), 2);
      if (with_old) {
        GPU_storagebuf_bind(store_in_buffers[i].buf(), 3);
      }
      for (const int attribute_i : attributes_.index_range()) {
        GPU_storagebuf_bind(attribute_buffers[attribute_i].buf(), (with_old ? 4 : 3) + attribute_i);
      }
      ha_dispatch(shader, padded_count);
      for (const int attribute_i : attributes_.index_range()) {
        GPU_storagebuf_unbind(attribute_buffers[attribute_i].buf());
      }
      if (with_old) {
        GPU_storagebuf_unbind(store_in_buffers[i].buf());
      }
      GPU_storagebuf_unbind(buffer_aggregate.buf());
      GPU_storagebuf_unbind(store_buffers[i].buf());
      GPU_storagebuf_unbind(result.buf());
      GPU_shader_unbind();
    }
    GPU_finish();
    for (const int i : stores_.index_range()) {
      const HaStore &store = stores_[i];
      if (!write_attribute_from_buffer(dst_attributes,
                                       store.name,
                                       store.type,
                                       bke::AttrDomain::Point,
                                       element_count,
                                       padded_count,
                                       store_buffers[i]))
      {
        return false;
      }
    }
  }

  if (!element_stores_.is_empty()) {
    bke::MutableAttributeAccessor mesh_attributes = mesh->attributes_for_write();
    for (const HaElementStore &store : element_stores_) {
      const bke::AttrDomain domain = ha_element_domain_to_attr(store.domain);
      if (ha_program_debug_enabled()) {
        fprintf(stderr, "HA program: element store '%s'\n", store.name.c_str());
      }
      const int count = int(mesh_attributes.domain_size(domain));
      if (count <= 0) {
        continue;
      }
      const int padded = (count + local_size - 1) / local_size * local_size;
      HaBuffer element_in;
      HaBuffer element_out;
      HaBuffer element_params;
      if (!element_in.ensure(int64_t(padded) * store.components) ||
          !element_out.ensure(int64_t(padded) * store.components) || !element_params.ensure(4))
      {
        return false;
      }

      const HaAttribute attribute{store.name, store.type, store.components};
      if (!upload_one_attribute(mesh_attributes, attribute, domain, count, padded, element_in)) {
        return false;
      }

      Array<HaBuffer> step_attribute_buffers(store.attributes.size());
      Vector<std::string> step_attribute_names;
      Vector<HaShaderResource> step_resources;
      if (!upload_domain_attributes(mesh_attributes,
                                    Span<const HaAttribute>(store.attributes.data(),
                                                            store.attributes.size()),
                                    domain,
                                    count,
                                    padded,
                                    step_attribute_buffers,
                                    step_attribute_names,
                                    step_resources,
                                    3))
      {
        return false;
      }

      const std::array<float, 4> params{float(count), 0.0f, 0.0f, 0.0f};
      element_params.upload(params);

      std::string step_attribute_key;
      for (const HaAttribute &field_attribute : store.attributes) {
        step_attribute_key += "|";
        step_attribute_key += ha_attribute_buffer_glsl(field_attribute.type);
      }
      gpu::Shader *shader = ha_shader_get_element_store(
          "element_store|" + store.glsl_value + "|" + store.glsl_selection + "|" +
              std::to_string(store.components) + "|" + ha_type_glsl(store.type) +
              step_attribute_key,
          store.glsl_value,
          store.glsl_selection,
          ha_type_glsl(store.type),
          store.components,
          Span<const HaShaderResource>(step_resources.data(), int64_t(step_resources.size())));
      if (shader == nullptr) {
        return false;
      }
      GPU_shader_bind(shader);
      GPU_storagebuf_bind(element_in.buf(), 0);
      GPU_storagebuf_bind(element_out.buf(), 1);
      GPU_storagebuf_bind(element_params.buf(), 2);
      for (const int attribute_i : store.attributes.index_range()) {
        GPU_storagebuf_bind(step_attribute_buffers[attribute_i].buf(), 3 + attribute_i);
      }
      ha_dispatch(shader, padded);
      for (const int attribute_i : store.attributes.index_range()) {
        GPU_storagebuf_unbind(step_attribute_buffers[attribute_i].buf());
      }
      GPU_storagebuf_unbind(element_params.buf());
      GPU_storagebuf_unbind(element_out.buf());
      GPU_storagebuf_unbind(element_in.buf());
      GPU_shader_unbind();
      GPU_finish();

      if (!write_attribute_from_buffer(mesh_attributes,
                                       store.name,
                                       store.type,
                                       domain,
                                       count,
                                       padded,
                                       element_out))
      {
        return false;
      }
    }
  }

  /* The output component was replaced when the copy was made (mesh, point cloud or curves). */
  r_output = std::move(output);
  if (ha_program_debug_enabled()) {
    int element_attribute_count = 0;
    for (const HaElementStep &step : element_steps_) {
      element_attribute_count += int(step.attributes.size());
    }
    /* The transfer is what decides whether this backend is worth it for element wise work: at 1M
     * points the CPU reference is a few milliseconds, so a slow upload or a stalling readback is
     * the whole budget. `dispatches` is what a chain buys: it grows, the two transfers do not. */
    fprintf(stderr,
            "HA program: %d elements, %d step(s), %d attribute(s), %d store(s), %d element "
            "store(s), %d param(s), %d element pass(es), %d element attribute(s), %d corner "
            "pass(es): upload %.2f ms, dispatch %.2f ms (%d), download %.2f ms, write %.2f ms\n",
            element_count,
            int(steps_.size()),
            int(attributes_.size()),
            int(stores_.size()),
            int(element_stores_.size()),
            int(params_.size()),
            int(element_steps_.size()),
            element_attribute_count,
            int(corner_steps_.size()),
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
