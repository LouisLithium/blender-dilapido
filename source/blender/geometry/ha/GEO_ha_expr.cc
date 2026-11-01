/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GEO_ha_expr.hh"

#include <cmath>
#include <cstdio>

#include "BLI_math_base.hh"
#include "BLI_math_vector.hh"

namespace blender::geometry::ha {

/* -------------------------------------------------------------------- */
/** \name Pool
 * \{ */

HaExpr &HaExprPool::add(HaType type, HaOp op, std::initializer_list<const HaExpr *> args)
{
  exprs_.append(std::make_unique<HaExpr>());
  HaExpr &expr = *exprs_.last();
  expr.type = type;
  expr.op = op;
  for (const HaExpr *arg : args) {
    expr.args.append(arg);
  }
  return expr;
}

HaExpr &HaExprPool::constant(const float value)
{
  HaExpr &expr = this->add(HaType::Float, HaOp::Const);
  expr.value[0] = value;
  return expr;
}

HaExpr &HaExprPool::constant(const float3 &value)
{
  HaExpr &expr = this->add(HaType::Float3, HaOp::Const);
  expr.value[0] = value.x;
  expr.value[1] = value.y;
  expr.value[2] = value.z;
  return expr;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name CPU evaluation
 * \{ */

namespace {

float eval_float(const HaExpr &expr, int index, Span<float> positions);
float3 eval_vector(const HaExpr &expr, int index, Span<float> positions);

/* Arguments of vector ops treat scalar expressions as a uniform vector, matching GLSL's implicit
 * `vec3(x)` promotion that #emit_as_vector() adds on the GPU side. */
float3 eval_arg_as_vector(const HaExpr &expr, int index, Span<float> positions)
{
  if (expr.type == HaType::Float3) {
    return eval_vector(expr, index, positions);
  }
  const float value = eval_float(expr, index, positions);
  return float3(value, value, value);
}

float eval_binary_float(const HaExpr &expr, int index, Span<float> positions)
{
  const float a = eval_float(*expr.args[0], index, positions);
  const float b = eval_float(*expr.args[1], index, positions);
  switch (expr.op) {
    case HaOp::Add:
      return a + b;
    case HaOp::Sub:
      return a - b;
    case HaOp::Mul:
      return a * b;
    case HaOp::Div:
      return b == 0.0f ? 0.0f : a / b;
    case HaOp::Max:
      return math::max(a, b);
    case HaOp::Min:
      return math::min(a, b);
    default:
      return 0.0f;
  }
}

float eval_float(const HaExpr &expr, int index, Span<float> positions)
{
  switch (expr.op) {
    case HaOp::Const:
      return expr.value[0];
    case HaOp::Index:
      return float(index);
    case HaOp::AggregateRead:
      /* Reduction results are only available on the GPU; the CPU reference for them is the regular
       * node evaluation (the `Force CPU` path). */
      return 0.0f;
    case HaOp::Sin:
      return math::sin(eval_float(*expr.args[0], index, positions));
    case HaOp::Cos:
      return math::cos(eval_float(*expr.args[0], index, positions));
    case HaOp::SplitX:
      return eval_vector(*expr.args[0], index, positions).x;
    case HaOp::SplitY:
      return eval_vector(*expr.args[0], index, positions).y;
    case HaOp::SplitZ:
      return eval_vector(*expr.args[0], index, positions).z;
    case HaOp::Length:
      return math::length(eval_vector(*expr.args[0], index, positions));
    case HaOp::Abs:
      return fabsf(eval_float(*expr.args[0], index, positions));
    case HaOp::Floor:
      return floorf(eval_float(*expr.args[0], index, positions));
    case HaOp::Ceil:
      return ceilf(eval_float(*expr.args[0], index, positions));
    case HaOp::Fract: {
      const float value = eval_float(*expr.args[0], index, positions);
      return value - floorf(value);
    }
    case HaOp::Sqrt:
      return sqrtf(math::max(0.0f, eval_float(*expr.args[0], index, positions)));
    case HaOp::Pow:
      return powf(math::max(0.0f, eval_float(*expr.args[0], index, positions)),
                  eval_float(*expr.args[1], index, positions));
    case HaOp::Dot: {
      const float3 a = eval_vector(*expr.args[0], index, positions);
      const float3 b = eval_vector(*expr.args[1], index, positions);
      return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    case HaOp::Distance: {
      const float3 a = eval_vector(*expr.args[0], index, positions);
      const float3 b = eval_vector(*expr.args[1], index, positions);
      return math::length(a - b);
    }
    case HaOp::Clamp: {
      const float value = eval_float(*expr.args[0], index, positions);
      const float min_value = eval_float(*expr.args[1], index, positions);
      const float max_value = eval_float(*expr.args[2], index, positions);
      return math::min(math::max(value, min_value), max_value);
    }
    default:
      return eval_binary_float(expr, index, positions);
  }
}

float3 eval_vector(const HaExpr &expr, int index, Span<float> positions)
{
  switch (expr.op) {
    case HaOp::Const:
      return float3(expr.value[0], expr.value[1], expr.value[2]);
    case HaOp::Position: {
      const int i = index * 3;
      if (i + 2 >= positions.size()) {
        return float3(0.0f);
      }
      return float3(positions[i], positions[i + 1], positions[i + 2]);
    }
    case HaOp::Combine:
      return float3(eval_float(*expr.args[0], index, positions),
                    eval_float(*expr.args[1], index, positions),
                    eval_float(*expr.args[2], index, positions));
    case HaOp::Abs: {
      const float3 v = eval_vector(*expr.args[0], index, positions);
      return float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
    }
    case HaOp::Floor: {
      const float3 v = eval_vector(*expr.args[0], index, positions);
      return float3(floorf(v.x), floorf(v.y), floorf(v.z));
    }
    case HaOp::Ceil: {
      const float3 v = eval_vector(*expr.args[0], index, positions);
      return float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
    }
    case HaOp::Fract: {
      const float3 v = eval_vector(*expr.args[0], index, positions);
      return float3(v.x - floorf(v.x), v.y - floorf(v.y), v.z - floorf(v.z));
    }
    case HaOp::Normalize: {
      const float3 v = eval_vector(*expr.args[0], index, positions);
      const float len = math::length(v);
      return len > 1e-20f ? v / len : float3(0.0f);
    }
    default:
      break;
  }

  const float3 a = eval_arg_as_vector(*expr.args[0], index, positions);
  const float3 b = eval_arg_as_vector(*expr.args[1], index, positions);
  switch (expr.op) {
    case HaOp::Add:
      return a + b;
    case HaOp::Sub:
      return a - b;
    case HaOp::Mul:
      return a * b;
    case HaOp::Div:
      return float3(b.x == 0.0f ? 0.0f : a.x / b.x,
                    b.y == 0.0f ? 0.0f : a.y / b.y,
                    b.z == 0.0f ? 0.0f : a.z / b.z);
    case HaOp::Max:
      return float3(math::max(a.x, b.x), math::max(a.y, b.y), math::max(a.z, b.z));
    case HaOp::Min:
      return float3(math::min(a.x, b.x), math::min(a.y, b.y), math::min(a.z, b.z));
    default:
      return float3(0.0f);
  }
}

}  // namespace

float3 ha_evaluate(const HaExpr &expr, const int index, Span<float> positions)
{
  if (expr.type == HaType::Float3) {
    return eval_vector(expr, index, positions);
  }
  const float value = eval_float(expr, index, positions);
  return float3(value, value, value);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name GLSL emission
 * \{ */

namespace {

std::string format_float(const float value)
{
  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%.9g", value);
  std::string text = buffer;
  if (text.find_first_of(".eE") == std::string::npos) {
    /* Keep it a float literal: `1` would be an int in GLSL. */
    text += ".0";
  }
  return text;
}

std::string emit(const HaExpr &expr);
std::string emit_as_vector(const HaExpr &expr);

std::string emit_binary(const char *op, const HaExpr &expr)
{
  /* Only promote the operands when the result is a vector. Promoting a float operation produces
   * `vec3(vec3(...))` which some drivers reject. */
  if (expr.type == HaType::Float3) {
    return std::string("(") + emit_as_vector(*expr.args[0]) + " " + op + " " +
           emit_as_vector(*expr.args[1]) + ")";
  }
  return std::string("(") + emit(*expr.args[0]) + " " + op + " " + emit(*expr.args[1]) + ")";
}

std::string emit(const HaExpr &expr)
{
  switch (expr.op) {
    case HaOp::Const:
      if (expr.type == HaType::Float3) {
        return "vec3(" + format_float(expr.value[0]) + ", " + format_float(expr.value[1]) + ", " +
               format_float(expr.value[2]) + ")";
      }
      return format_float(expr.value[0]);
    case HaOp::Position:
      return "position";
    case HaOp::Index:
      return "float(index)";
    case HaOp::Add:
      return emit_binary("+", expr);
    case HaOp::Sub:
      return emit_binary("-", expr);
    case HaOp::Mul:
      return emit_binary("*", expr);
    case HaOp::Div:
      return emit_binary("/", expr);
    case HaOp::Max:
    case HaOp::Min: {
      const char *fn = expr.op == HaOp::Max ? "max" : "min";
      if (expr.type == HaType::Float3) {
        return std::string(fn) + "(" + emit_as_vector(*expr.args[0]) + ", " +
               emit_as_vector(*expr.args[1]) + ")";
      }
      return std::string(fn) + "(" + emit(*expr.args[0]) + ", " + emit(*expr.args[1]) + ")";
    }
    case HaOp::Sin:
      return "sin(" + emit(*expr.args[0]) + ")";
    case HaOp::Cos:
      return "cos(" + emit(*expr.args[0]) + ")";
    case HaOp::Combine:
      return "vec3(" + emit(*expr.args[0]) + ", " + emit(*expr.args[1]) + ", " +
             emit(*expr.args[2]) + ")";
    case HaOp::SplitX:
      return "(" + emit_as_vector(*expr.args[0]) + ").x";
    case HaOp::SplitY:
      return "(" + emit_as_vector(*expr.args[0]) + ").y";
    case HaOp::SplitZ:
      return "(" + emit_as_vector(*expr.args[0]) + ").z";
    case HaOp::Length:
      return "length(" + emit_as_vector(*expr.args[0]) + ")";
    case HaOp::Abs:
      return "abs(" + emit(*expr.args[0]) + ")";
    case HaOp::Floor:
      return "floor(" + emit(*expr.args[0]) + ")";
    case HaOp::Ceil:
      return "ceil(" + emit(*expr.args[0]) + ")";
    case HaOp::Fract:
      return "fract(" + emit(*expr.args[0]) + ")";
    case HaOp::Sqrt:
      return "sqrt(max(0.0, " + emit(*expr.args[0]) + "))";
    case HaOp::Pow:
      return "pow(max(0.0, " + emit(*expr.args[0]) + "), " + emit(*expr.args[1]) + ")";
    case HaOp::Normalize: {
      /* `normalize(vec3(0))` is undefined in GLSL (NaN), while the CPU evaluator returns 0. Emit
       * the same guard so both sides stay bit-comparable. */
      const std::string value = emit_as_vector(*expr.args[0]);
      return "(dot(" + value + ", " + value + ") > 0.0 ? normalize(" + value +
             ") : vec3(0.0))";
    }
    case HaOp::Dot:
      return "dot(" + emit_as_vector(*expr.args[0]) + ", " + emit_as_vector(*expr.args[1]) + ")";
    case HaOp::Distance:
      return "distance(" + emit_as_vector(*expr.args[0]) + ", " + emit_as_vector(*expr.args[1]) +
             ")";
    case HaOp::Clamp:
      return "clamp(" + emit(*expr.args[0]) + ", " + emit(*expr.args[1]) + ", " +
             emit(*expr.args[2]) + ")";
    case HaOp::AggregateRead:
      return "aggregate[" + std::to_string(expr.aggregate_slot) + " * 3]";
  }
  return "0.0";
}

std::string emit_as_vector(const HaExpr &expr)
{
  if (expr.type == HaType::Float3) {
    return emit(expr);
  }
  return "vec3(" + emit(expr) + ")";
}

}  // namespace

std::string ha_emit_glsl(const HaExpr &expr)
{
  if (expr.type == HaType::Float3) {
    return emit(expr);
  }
  return emit_as_vector(expr);
}

std::string ha_emit_glsl_scalar(const HaExpr &expr)
{
  if (expr.type == HaType::Float3) {
    /* Callers check the type; a vector would need an explicit component. */
    return emit(expr) + ".x";
  }
  return emit(expr);
}

/** \} */

}  // namespace blender::geometry::ha
