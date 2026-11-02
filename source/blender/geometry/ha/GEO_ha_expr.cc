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

HaExpr &HaExprPool::add(HaType type, HaOp op, Span<const HaExpr *> args)
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
/** \name Shared helpers
 *
 * Copies of the Blender functions the node implementations use. They are duplicated here on
 * purpose: the CPU evaluator has to be bit identical to the real node evaluation, and the GLSL
 * emitter has to produce the same formula (see the note in the header).
 * \{ */

namespace {

float ha_safe_divide(const float a, const float b)
{
  return b != 0.0f ? a / b : 0.0f;
}

float ha_safe_modf(const float a, const float b)
{
  return b != 0.0f ? fmodf(a, b) : 0.0f;
}

float ha_safe_floored_modf(const float a, const float b)
{
  return b != 0.0f ? a - floorf(a / b) * b : 0.0f;
}

float ha_safe_logf(const float a, const float base)
{
  if (a <= 0.0f || base <= 0.0f) {
    return 0.0f;
  }
  return ha_safe_divide(logf(a), logf(base));
}

float ha_safe_powf(const float base, const float exponent)
{
  if (base < 0.0f && exponent != float(int(exponent))) {
    return 0.0f;
  }
  return powf(base, exponent);
}

float ha_safe_sqrtf(const float a)
{
  return sqrtf(math::max(a, 0.0f));
}

float ha_safe_inverse_sqrtf(const float a)
{
  return a > 0.0f ? 1.0f / sqrtf(a) : 0.0f;
}

float ha_compatible_signf(const float a)
{
  if (a > 0.0f) {
    return 1.0f;
  }
  if (a < 0.0f) {
    return -1.0f;
  }
  return 0.0f;
}

float ha_truncf(const float a)
{
  return a >= 0.0f ? floorf(a) : ceilf(a);
}

float ha_roundf(const float a)
{
  return floorf(a + 0.5f);
}

float ha_snapf(const float a, const float b)
{
  return floorf(ha_safe_divide(a, b)) * b;
}

float ha_wrapf(const float value, const float max, const float min)
{
  const float range = max - min;
  return range != 0.0f ? value - (range * floorf((value - min) / range)) : min;
}

float ha_pingpongf(const float value, const float scale)
{
  if (scale == 0.0f) {
    return 0.0f;
  }
  const float t = (value - scale) / (scale * 2.0f);
  return fabsf((t - floorf(t)) * scale * 2.0f - scale);
}

float ha_smoothminf(const float a, const float b, const float c)
{
  if (c != 0.0f) {
    const float h = math::max(c - fabsf(a - b), 0.0f) / c;
    return math::min(a, b) - h * h * h * c * (1.0f / 6.0f);
  }
  return math::min(a, b);
}

/** `math::interpolate`: `a * (1 - t) + b * t`. */
float ha_interpolate(const float a, const float b, const float t)
{
  return a * (1.0f - t) + b * t;
}

float3 ha_interpolate(const float3 &a, const float3 &b, const float3 &t)
{
  return a * (float3(1.0f) - t) + b * t;
}

float ha_clamp_range(const float value, const float min, const float max)
{
  return (min > max) ? math::min(math::max(value, max), min) :
                       math::min(math::max(value, min), max);
}

float ha_compare(const int op, const float a, const float b, const float epsilon)
{
  switch (HaParam(op)) {
    case HaParam::CompareLess:
      return a < b ? 1.0f : 0.0f;
    case HaParam::CompareLessEqual:
      return a <= b ? 1.0f : 0.0f;
    case HaParam::CompareGreater:
      return a > b ? 1.0f : 0.0f;
    case HaParam::CompareGreaterEqual:
      return a >= b ? 1.0f : 0.0f;
    case HaParam::CompareEqual:
      return fabsf(a - b) <= epsilon ? 1.0f : 0.0f;
    case HaParam::CompareNotEqual:
      return fabsf(a - b) > epsilon ? 1.0f : 0.0f;
    default:
      return 0.0f;
  }
}

float ha_bool_math(const int op, const float a, const float b)
{
  /* The field system's `float -> bool` conversion is `value > 0`. */
  const bool ba = a > 0.0f;
  const bool bb = b > 0.0f;
  switch (HaParam(op)) {
    case HaParam::BoolAnd:
      return (ba && bb) ? 1.0f : 0.0f;
    case HaParam::BoolOr:
      return (ba || bb) ? 1.0f : 0.0f;
    case HaParam::BoolNot:
      return ba ? 0.0f : 1.0f;
    case HaParam::BoolNand:
      return (ba && bb) ? 0.0f : 1.0f;
    case HaParam::BoolNor:
      return (ba || bb) ? 0.0f : 1.0f;
    case HaParam::BoolXor:
      return (ba != bb) ? 1.0f : 0.0f;
    case HaParam::BoolXnor:
      return (ba == bb) ? 1.0f : 0.0f;
    default:
      return 0.0f;
  }
}

/** One `Map Range` step, shared by the scalar and the per component evaluation. */
float ha_map_range(const int param,
                   const float value,
                   const float from_min,
                   const float from_max,
                   const float to_min,
                   const float to_max,
                   const float steps)
{
  float factor = ha_safe_divide(value - from_min, from_max - from_min);
  switch (HaParam(param & 0xFF)) {
    case HaParam::MapRangeStepped:
      factor = ha_safe_divide(floorf(factor * (steps + 1.0f)), steps);
      break;
    case HaParam::MapRangeSmoothStep:
      factor = math::min(math::max(factor, 0.0f), 1.0f);
      factor = (3.0f - 2.0f * factor) * (factor * factor);
      break;
    case HaParam::MapRangeSmootherStep:
      factor = math::min(math::max(factor, 0.0f), 1.0f);
      factor = factor * factor * factor * (factor * (factor * 6.0f - 15.0f) + 10.0f);
      break;
    default:
      break;
  }
  float result = to_min + factor * (to_max - to_min);
  if ((param & HA_PARAM_CLAMP) != 0) {
    result = ha_clamp_range(result, to_min, to_max);
  }
  return result;
}

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

float eval_unary_float(const HaExpr &expr, int index, Span<float> positions)
{
  const float a = eval_float(*expr.args[0], index, positions);
  switch (expr.op) {
    case HaOp::Round:
      return ha_roundf(a);
    case HaOp::Trunc:
      return ha_truncf(a);
    case HaOp::Sign:
      return ha_compatible_signf(a);
    case HaOp::Exp:
      return expf(a);
    case HaOp::InvSqrt:
      return ha_safe_inverse_sqrtf(a);
    case HaOp::Tan:
      return tanf(a);
    case HaOp::Asin:
      return asinf(math::min(math::max(a, -1.0f), 1.0f));
    case HaOp::Acos:
      return acosf(math::min(math::max(a, -1.0f), 1.0f));
    case HaOp::Atan:
      return atanf(a);
    case HaOp::Sinh:
      return sinhf(a);
    case HaOp::Cosh:
      return coshf(a);
    case HaOp::Tanh:
      return tanhf(a);
    default:
      return 0.0f;
  }
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
      return ha_safe_divide(a, b);
    case HaOp::Max:
      return math::max(a, b);
    case HaOp::Min:
      return math::min(a, b);
    case HaOp::LessThan:
      return a < b ? 1.0f : 0.0f;
    case HaOp::GreaterThan:
      return a > b ? 1.0f : 0.0f;
    case HaOp::Atan2:
      return atan2f(a, b);
    case HaOp::Log:
      return ha_safe_logf(a, b);
    case HaOp::Modulo:
      return ha_safe_modf(a, b);
    case HaOp::FlooredModulo:
      return ha_safe_floored_modf(a, b);
    case HaOp::Snap:
      return ha_snapf(a, b);
    case HaOp::PingPong:
      return ha_pingpongf(a, b);
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
    case HaOp::AttributeRead:
      /* Attribute buffers are only uploaded for the device path; the CPU reference is the node
       * evaluation. */
      return 0.0f;
    case HaOp::WhiteNoise:
      /* The hash lives in the GLSL helpers; the CPU reference is the node evaluation. */
      return 0.0f;
    case HaOp::NoiseTexture:
      /* The Perlin noise lives in the GLSL helpers; the CPU reference is the node evaluation. */
      return 0.0f;
    case HaOp::WaveTexture:
      /* Same: the wave is a helper; the CPU reference is the node evaluation. */
      return 0.0f;
    case HaOp::VoronoiTexture:
    case HaOp::VoronoiRadius:
    case HaOp::VoronoiW:
      /* Same: the voronoi kernel is a helper; the CPU reference is the node evaluation. */
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
      return ha_safe_sqrtf(eval_float(*expr.args[0], index, positions));
    case HaOp::Pow:
      return ha_safe_powf(eval_float(*expr.args[0], index, positions),
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
    case HaOp::MulAdd: {
      const float a = eval_float(*expr.args[0], index, positions);
      const float b = eval_float(*expr.args[1], index, positions);
      const float c = eval_float(*expr.args[2], index, positions);
      return a * b + c;
    }
    case HaOp::CompareEps: {
      const float a = eval_float(*expr.args[0], index, positions);
      const float b = eval_float(*expr.args[1], index, positions);
      const float epsilon = eval_float(*expr.args[2], index, positions);
      return ((a == b) || (fabsf(a - b) <= math::max(epsilon, FLT_EPSILON))) ? 1.0f : 0.0f;
    }
    case HaOp::SmoothMin: {
      const float a = eval_float(*expr.args[0], index, positions);
      const float b = eval_float(*expr.args[1], index, positions);
      const float k = eval_float(*expr.args[2], index, positions);
      return ha_smoothminf(a, b, k);
    }
    case HaOp::SmoothMax: {
      const float a = eval_float(*expr.args[0], index, positions);
      const float b = eval_float(*expr.args[1], index, positions);
      const float k = eval_float(*expr.args[2], index, positions);
      return -ha_smoothminf(-a, -b, k);
    }
    case HaOp::Wrap: {
      const float value = eval_float(*expr.args[0], index, positions);
      const float max_value = eval_float(*expr.args[1], index, positions);
      const float min_value = eval_float(*expr.args[2], index, positions);
      return ha_wrapf(value, max_value, min_value);
    }
    case HaOp::Mix: {
      float factor = eval_float(*expr.args[0], index, positions);
      const float a = eval_float(*expr.args[1], index, positions);
      const float b = eval_float(*expr.args[2], index, positions);
      if ((expr.param & HA_PARAM_CLAMP) != 0) {
        factor = math::min(math::max(factor, 0.0f), 1.0f);
      }
      return ha_interpolate(a, b, factor);
    }
    case HaOp::MapRange: {
      const float value = eval_float(*expr.args[0], index, positions);
      const float from_min = eval_float(*expr.args[1], index, positions);
      const float from_max = eval_float(*expr.args[2], index, positions);
      const float to_min = eval_float(*expr.args[3], index, positions);
      const float to_max = eval_float(*expr.args[4], index, positions);
      const float steps = expr.args.size() > 5 ? eval_float(*expr.args[5], index, positions) : 0.0f;
      return ha_map_range(expr.param, value, from_min, from_max, to_min, to_max, steps);
    }
    case HaOp::BoolMath:
      return ha_bool_math(expr.param,
                          eval_float(*expr.args[0], index, positions),
                          expr.args.size() > 1 ? eval_float(*expr.args[1], index, positions) : 0.0f);
    case HaOp::Compare:
      return ha_compare(expr.param,
                        eval_float(*expr.args[0], index, positions),
                        eval_float(*expr.args[1], index, positions),
                        expr.args.size() > 2 ? eval_float(*expr.args[2], index, positions) : 0.0f);
    case HaOp::Round:
    case HaOp::Trunc:
    case HaOp::Sign:
    case HaOp::Exp:
    case HaOp::InvSqrt:
    case HaOp::Tan:
    case HaOp::Asin:
    case HaOp::Acos:
    case HaOp::Atan:
    case HaOp::Sinh:
    case HaOp::Cosh:
    case HaOp::Tanh:
      return eval_unary_float(expr, index, positions);
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
    case HaOp::VoronoiPosition:
      /* The voronoi kernel lives in the GLSL helpers; the CPU reference is the node evaluation. */
      return float3(0.0f);
    case HaOp::NoiseTextureColor:
      /* Same, the three channels are separate fractals in the helper. */
      return float3(0.0f);
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
    case HaOp::Cross: {
      const float3 a = eval_vector(*expr.args[0], index, positions);
      const float3 b = eval_vector(*expr.args[1], index, positions);
      return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }
    case HaOp::Project: {
      const float3 a = eval_vector(*expr.args[0], index, positions);
      const float3 b = eval_vector(*expr.args[1], index, positions);
      const float len_sq = b.x * b.x + b.y * b.y + b.z * b.z;
      if (len_sq == 0.0f) {
        return float3(0.0f);
      }
      const float scale = (a.x * b.x + a.y * b.y + a.z * b.z) / len_sq;
      return b * scale;
    }
    case HaOp::Reflect: {
      const float3 a = eval_vector(*expr.args[0], index, positions);
      const float3 b = eval_vector(*expr.args[1], index, positions);
      const float dot = b.x * a.x + b.y * a.y + b.z * a.z;
      return a - b * (2.0f * dot);
    }
    case HaOp::Mix: {
      const float3 factor = eval_arg_as_vector(*expr.args[0], index, positions);
      const float3 a = eval_vector(*expr.args[1], index, positions);
      const float3 b = eval_vector(*expr.args[2], index, positions);
      float3 t = factor;
      if ((expr.param & HA_PARAM_CLAMP) != 0) {
        t = float3(math::min(math::max(t.x, 0.0f), 1.0f),
                   math::min(math::max(t.y, 0.0f), 1.0f),
                   math::min(math::max(t.z, 0.0f), 1.0f));
      }
      return ha_interpolate(a, b, t);
    }
    case HaOp::MapRange: {
      const float3 value = eval_vector(*expr.args[0], index, positions);
      const float3 from_min = eval_vector(*expr.args[1], index, positions);
      const float3 from_max = eval_vector(*expr.args[2], index, positions);
      const float3 to_min = eval_vector(*expr.args[3], index, positions);
      const float3 to_max = eval_vector(*expr.args[4], index, positions);
      const float3 steps = expr.args.size() > 5 ?
                               eval_vector(*expr.args[5], index, positions) :
                               float3(0.0f);
      return float3(ha_map_range(expr.param,
                                 value.x,
                                 from_min.x,
                                 from_max.x,
                                 to_min.x,
                                 to_max.x,
                                 steps.x),
                    ha_map_range(expr.param,
                                 value.y,
                                 from_min.y,
                                 from_max.y,
                                 to_min.y,
                                 to_max.y,
                                 steps.y),
                    ha_map_range(expr.param,
                                 value.z,
                                 from_min.z,
                                 from_max.z,
                                 to_min.z,
                                 to_max.z,
                                 steps.z));
    }
    case HaOp::MulAdd: {
      const float3 a = eval_arg_as_vector(*expr.args[0], index, positions);
      const float3 b = eval_arg_as_vector(*expr.args[1], index, positions);
      const float3 c = eval_arg_as_vector(*expr.args[2], index, positions);
      return a * b + c;
    }
    case HaOp::Snap: {
      const float3 a = eval_arg_as_vector(*expr.args[0], index, positions);
      const float3 b = eval_arg_as_vector(*expr.args[1], index, positions);
      return float3(ha_snapf(a.x, b.x), ha_snapf(a.y, b.y), ha_snapf(a.z, b.z));
    }
    case HaOp::Wrap: {
      const float3 value = eval_arg_as_vector(*expr.args[0], index, positions);
      const float3 max_value = eval_arg_as_vector(*expr.args[1], index, positions);
      const float3 min_value = eval_arg_as_vector(*expr.args[2], index, positions);
      return float3(ha_wrapf(value.x, max_value.x, min_value.x),
                    ha_wrapf(value.y, max_value.y, min_value.y),
                    ha_wrapf(value.z, max_value.z, min_value.z));
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
      return float3(ha_safe_divide(a.x, b.x), ha_safe_divide(a.y, b.y), ha_safe_divide(a.z, b.z));
    case HaOp::Max:
      return float3(math::max(a.x, b.x), math::max(a.y, b.y), math::max(a.z, b.z));
    case HaOp::Min:
      return float3(math::min(a.x, b.x), math::min(a.y, b.y), math::min(a.z, b.z));
    case HaOp::Modulo:
      return float3(ha_safe_modf(a.x, b.x), ha_safe_modf(a.y, b.y), ha_safe_modf(a.z, b.z));
    case HaOp::FlooredModulo:
      return float3(ha_safe_floored_modf(a.x, b.x),
                    ha_safe_floored_modf(a.y, b.y),
                    ha_safe_floored_modf(a.z, b.z));
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

/** Emits an argument as a vector when the result is a vector, otherwise as a scalar. */
std::string emit_typed(const HaExpr &expr, const bool as_vector)
{
  return as_vector ? emit_as_vector(expr) : emit(expr);
}

/**
 * The `safe_*` helpers are generated into the kernel (see `HA_GLSL_HELPERS` in `GEO_ha_context.cc`),
 * because their guards differ per component for vectors and writing them out here would make the
 * emitted expression explode. They mirror the BLI functions the CPU nodes use.
 */
std::string emit_call(const char *fn, const HaExpr &a, const HaExpr &b, const bool vec)
{
  return std::string(fn) + "(" + emit_typed(a, vec) + ", " + emit_typed(b, vec) + ")";
}

std::string emit_map_range(const HaExpr &expr)
{
  const bool vec = expr.type == HaType::Float3;
  const std::string zero = vec ? "vec3(0.0)" : "0.0";
  const std::string one = vec ? "vec3(1.0)" : "1.0";

  const std::string value = emit_typed(*expr.args[0], vec);
  const std::string from_min = emit_typed(*expr.args[1], vec);
  const std::string from_max = emit_typed(*expr.args[2], vec);
  const std::string to_min = emit_typed(*expr.args[3], vec);
  const std::string to_max = emit_typed(*expr.args[4], vec);

  std::string factor = "ha_safe_divide((" + value + ") - (" + from_min + "), (" + from_max + ") - (" +
                       from_min + "))";
  if (HaParam(expr.param & 0xFF) == HaParam::MapRangeStepped && expr.args.size() > 5) {
    const std::string steps = emit_typed(*expr.args[5], vec);
    factor = "ha_safe_divide(floor((" + factor + ") * ((" + steps + ") + " + one + ")), (" + steps +
             "))";
  }
  else if (HaParam(expr.param & 0xFF) == HaParam::MapRangeSmoothStep) {
    factor = "((3.0 - 2.0 * clamp(" + factor + ", " + zero + ", " + one + ")) * (clamp(" + factor +
             ", " + zero + ", " + one + ") * clamp(" + factor + ", " + zero + ", " + one + ")))";
  }
  else if (HaParam(expr.param & 0xFF) == HaParam::MapRangeSmootherStep) {
    const std::string f = "clamp(" + factor + ", " + zero + ", " + one + ")";
    factor = "((" + f + " * " + f + " * " + f + ") * ((" + f + " * (" + f + " * 6.0 - 15.0)) + " +
             one + "0.0))";
  }

  std::string result = "((" + to_min + ") + (" + factor + ") * ((" + to_max + ") - (" + to_min + ")))";
  if ((expr.param & HA_PARAM_CLAMP) != 0) {
    /* `clamp_range` swaps the bounds when `to_min > to_max`. */
    result = "clamp(" + result + ", min(" + to_min + ", " + to_max + "), max(" + to_min + ", " +
             to_max + "))";
  }
  return result;
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
      return emit_call("ha_safe_divide", *expr.args[0], *expr.args[1], expr.type == HaType::Float3);
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
    case HaOp::Pow: {
      /* `safe_powf`: a negative base with a non integer exponent is 0, and `pow()` is undefined for
       * negative bases in GLSL, so the integer case goes through `|base|` (see `HA_GLSL_HELPERS`). */
      return emit_call("ha_safe_pow", *expr.args[0], *expr.args[1], false);
    }
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
    case HaOp::AttributeRead: {
      const std::string buffer = "attribute_" + std::to_string(expr.attribute_slot);
      if (expr.type == HaType::Float3) {
        return "vec3(" + buffer + "[index * 3], " + buffer + "[index * 3 + 1], " + buffer +
               "[index * 3 + 2])";
      }
      return buffer + "[index]";
    }

    /* --- Math node ---------------------------------------------------------------------- */
    case HaOp::Round:
      return "floor(" + emit(*expr.args[0]) + " + 0.5)";
    case HaOp::Trunc: {
      const std::string a = emit(*expr.args[0]);
      return "(" + a + " >= 0.0 ? floor(" + a + ") : ceil(" + a + "))";
    }
    case HaOp::Sign:
      return "sign(" + emit(*expr.args[0]) + ")";
    case HaOp::Exp:
      return "exp(" + emit(*expr.args[0]) + ")";
    case HaOp::InvSqrt: {
      const std::string a = emit(*expr.args[0]);
      return "(" + a + " > 0.0 ? 1.0 / sqrt(" + a + ") : 0.0)";
    }
    case HaOp::Tan:
      return "tan(" + emit(*expr.args[0]) + ")";
    case HaOp::Asin:
      return "asin(clamp(" + emit(*expr.args[0]) + ", -1.0, 1.0))";
    case HaOp::Acos:
      return "acos(clamp(" + emit(*expr.args[0]) + ", -1.0, 1.0))";
    case HaOp::Atan:
      return "atan(" + emit(*expr.args[0]) + ")";
    case HaOp::Sinh:
      return "sinh(" + emit(*expr.args[0]) + ")";
    case HaOp::Cosh:
      return "cosh(" + emit(*expr.args[0]) + ")";
    case HaOp::Tanh:
      return "tanh(" + emit(*expr.args[0]) + ")";
    case HaOp::LessThan:
      return "((" + emit(*expr.args[0]) + " < " + emit(*expr.args[1]) + ") ? 1.0 : 0.0)";
    case HaOp::GreaterThan:
      return "((" + emit(*expr.args[0]) + " > " + emit(*expr.args[1]) + ") ? 1.0 : 0.0)";
    case HaOp::Atan2:
      return "atan(" + emit(*expr.args[0]) + ", " + emit(*expr.args[1]) + ")";
    case HaOp::Log: {
      const std::string a = emit(*expr.args[0]);
      const std::string b = emit(*expr.args[1]);
      return "(" + a + " <= 0.0 || " + b + " <= 0.0 ? 0.0 : (log(" + b + ") != 0.0 ? log(" + a +
             ") / log(" + b + ") : 0.0))";
    }
    case HaOp::Modulo:
    case HaOp::FlooredModulo:
      return emit_call(expr.op == HaOp::Modulo ? "ha_safe_mod" : "ha_safe_floored_mod",
                       *expr.args[0],
                       *expr.args[1],
                       expr.type == HaType::Float3);
    case HaOp::Snap: {
      const bool vec = expr.type == HaType::Float3;
      const std::string b = emit_typed(*expr.args[1], vec);
      return "floor(" + emit_call("ha_safe_divide", *expr.args[0], *expr.args[1], vec) + ") * (" + b +
             ")";
    }
    case HaOp::PingPong: {
      const std::string value = emit(*expr.args[0]);
      const std::string scale = emit(*expr.args[1]);
      const std::string t = "((" + value + " - " + scale + ") / (" + scale + " * 2.0))";
      return "(" + scale + " == 0.0 ? 0.0 : abs(fract(" + t + ") * " + scale + " * 2.0 - " + scale +
             "))";
    }
    case HaOp::MulAdd: {
      if (expr.type == HaType::Float3) {
        return "(" + emit_as_vector(*expr.args[0]) + " * " + emit_as_vector(*expr.args[1]) + " + " +
               emit_as_vector(*expr.args[2]) + ")";
      }
      return "(" + emit(*expr.args[0]) + " * " + emit(*expr.args[1]) + " + " + emit(*expr.args[2]) +
             ")";
    }
    case HaOp::CompareEps: {
      const std::string a = emit(*expr.args[0]);
      const std::string b = emit(*expr.args[1]);
      const std::string epsilon = emit(*expr.args[2]);
      return "(((" + a + ") == (" + b + ") || abs((" + a + ") - (" + b + ")) <= max(" + epsilon +
             ", 1.1920929e-7)) ? 1.0 : 0.0)";
    }
    case HaOp::SmoothMin: {
      const std::string a = emit(*expr.args[0]);
      const std::string b = emit(*expr.args[1]);
      const std::string k = emit(*expr.args[2]);
      const std::string h = "(max(" + k + " - abs((" + a + ") - (" + b + ")), 0.0) / (" + k + "))";
      return "((" + k + ") != 0.0 ? min(" + a + ", " + b + ") - ((" + h + " * " + h + " * " + h +
             ") * (" + k + ") * (1.0 / 6.0)) : min(" + a + ", " + b + "))";
    }
    case HaOp::SmoothMax: {
      const std::string a = "(-" + emit(*expr.args[0]) + ")";
      const std::string b = "(-" + emit(*expr.args[1]) + ")";
      const std::string k = emit(*expr.args[2]);
      const std::string h = "(max(" + k + " - abs((" + a + ") - (" + b + ")), 0.0) / (" + k + "))";
      return "-(" + std::string("((") + k + ") != 0.0 ? min(" + a + ", " + b + ") - ((" + h + " * " +
             h + " * " + h + ") * (" + k + ") * (1.0 / 6.0)) : min(" + a + ", " + b + "))";
    }
    case HaOp::Wrap: {
      const bool vec = expr.type == HaType::Float3;
      return std::string("ha_wrap(") + emit_typed(*expr.args[0], vec) + ", " +
             emit_typed(*expr.args[1], vec) + ", " + emit_typed(*expr.args[2], vec) + ")";
    }

    /* --- Vector math -------------------------------------------------------------------- */
    case HaOp::Cross:
      return "cross(" + emit_as_vector(*expr.args[0]) + ", " + emit_as_vector(*expr.args[1]) + ")";
    case HaOp::Project: {
      const std::string a = emit_as_vector(*expr.args[0]);
      const std::string b = emit_as_vector(*expr.args[1]);
      return "(dot(" + b + ", " + b + ") != 0.0 ? (dot(" + a + ", " + b + ") / dot(" + b + ", " + b +
             ")) * " + b + " : vec3(0.0))";
    }
    case HaOp::Reflect: {
      const std::string a = emit_as_vector(*expr.args[0]);
      const std::string b = emit_as_vector(*expr.args[1]);
      return "(" + a + " - " + b + " * (2.0 * dot(" + b + ", " + a + ")))";
    }

    /* --- Conversion nodes --------------------------------------------------------------- */
    case HaOp::Mix: {
      const bool vec = expr.type == HaType::Float3;
      std::string factor = vec ? emit_as_vector(*expr.args[0]) : emit(*expr.args[0]);
      const std::string a = vec ? emit_as_vector(*expr.args[1]) : emit(*expr.args[1]);
      const std::string b = vec ? emit_as_vector(*expr.args[2]) : emit(*expr.args[2]);
      if ((expr.param & HA_PARAM_CLAMP) != 0) {
        factor = "clamp(" + factor + ", " + (vec ? "vec3(0.0)" : "0.0") + ", " +
                 (vec ? "vec3(1.0)" : "1.0") + ")";
      }
      return "((" + a + ") * (" + (vec ? "vec3(1.0)" : "1.0") + " - (" + factor + ")) + (" + b +
             ") * (" + factor + "))";
    }
    case HaOp::MapRange:
      return emit_map_range(expr);
    case HaOp::BoolMath: {
      const std::string a = emit(*expr.args[0]);
      const std::string b = expr.args.size() > 1 ? emit(*expr.args[1]) : "0.0";
      const std::string ba = "(" + a + " > 0.0)";
      const std::string bb = "(" + b + " > 0.0)";
      switch (HaParam(expr.param)) {
        case HaParam::BoolAnd:
          return "((" + ba + " && " + bb + ") ? 1.0 : 0.0)";
        case HaParam::BoolOr:
          return "((" + ba + " || " + bb + ") ? 1.0 : 0.0)";
        case HaParam::BoolNot:
          return "(" + ba + " ? 0.0 : 1.0)";
        case HaParam::BoolNand:
          return "((" + ba + " && " + bb + ") ? 0.0 : 1.0)";
        case HaParam::BoolNor:
          return "((" + ba + " || " + bb + ") ? 0.0 : 1.0)";
        case HaParam::BoolXor:
          return "((" + ba + " != " + bb + ") ? 1.0 : 0.0)";
        case HaParam::BoolXnor:
          return "((" + ba + " == " + bb + ") ? 1.0 : 0.0)";
        default:
          return "0.0";
      }
    }
    case HaOp::Compare: {
      const std::string a = emit(*expr.args[0]);
      const std::string b = emit(*expr.args[1]);
      switch (HaParam(expr.param)) {
        case HaParam::CompareEqual:
        case HaParam::CompareNotEqual: {
          const std::string epsilon = expr.args.size() > 2 ? emit(*expr.args[2]) : "0.0";
          const char *op = HaParam(expr.param) == HaParam::CompareEqual ? "<=" : ">";
          return "((abs((" + a + ") - (" + b + ")) " + op + " " + epsilon + ") ? 1.0 : 0.0)";
        }
        default:
          break;
      }
      const char *op = "==";
      switch (HaParam(expr.param)) {
        case HaParam::CompareLess:
          op = "<";
          break;
        case HaParam::CompareLessEqual:
          op = "<=";
          break;
        case HaParam::CompareGreater:
          op = ">";
          break;
        case HaParam::CompareGreaterEqual:
          op = ">=";
          break;
        default:
          break;
      }
      return "((" + a + " " + op + " " + b + ") ? 1.0 : 0.0)";
    }
    case HaOp::WhiteNoise: {
      switch (expr.param) {
        case 1:
          return "ha_white_noise_1d(" + emit(*expr.args[0]) + ")";
        case 2:
          return "ha_white_noise_2d(" + emit_as_vector(*expr.args[0]) + ")";
        case 3:
          return "ha_white_noise_3d(" + emit_as_vector(*expr.args[0]) + ")";
        case 4:
          return "ha_white_noise_4d(" + emit_as_vector(*expr.args[0]) + ", " +
                 emit(*expr.args[1]) + ")";
        default:
          return "0.0";
      }
    }
    case HaOp::NoiseTexture: {
      const int type = expr.param & 0xFF;
      const bool normalize = (expr.param & HA_PARAM_CLAMP) != 0;
      return "ha_noise_texture_3d(" + emit_as_vector(*expr.args[0]) + ", " + emit(*expr.args[1]) +
             ", " + emit(*expr.args[2]) + ", " + emit(*expr.args[3]) + ", " + emit(*expr.args[4]) +
             ", " + emit(*expr.args[5]) + ", " + emit(*expr.args[6]) + ", " +
             std::to_string(type) + ", " + (normalize ? "true" : "false") + ")";
    }
    case HaOp::NoiseTextureColor: {
      const int type = expr.param & 0xFF;
      const bool normalize = (expr.param & HA_PARAM_CLAMP) != 0;
      return "ha_noise_texture_color_3d(" + emit_as_vector(*expr.args[0]) + ", " +
             emit(*expr.args[1]) + ", " + emit(*expr.args[2]) + ", " + emit(*expr.args[3]) + ", " +
             emit(*expr.args[4]) + ", " + emit(*expr.args[5]) + ", " + emit(*expr.args[6]) + ", " +
             std::to_string(type) + ", " + (normalize ? "true" : "false") + ")";
    }
    case HaOp::WaveTexture: {
      const int wave_type = expr.param & 0x3;
      const int bands_dir = (expr.param >> 2) & 0x3;
      const int rings_dir = (expr.param >> 4) & 0x3;
      const int profile = (expr.param >> 6) & 0x3;
      return "ha_wave_texture(" + emit_as_vector(*expr.args[0]) + ", " + emit(*expr.args[1]) + ", " +
             emit(*expr.args[2]) + ", " + emit(*expr.args[3]) + ", " + emit(*expr.args[4]) + ", " +
             emit(*expr.args[5]) + ", " + emit(*expr.args[6]) + ", " + std::to_string(wave_type) +
             ", " + std::to_string(bands_dir) + ", " + std::to_string(rings_dir) + ", " +
             std::to_string(profile) + ")";
    }
    case HaOp::VoronoiTexture:
    case HaOp::VoronoiPosition:
    case HaOp::VoronoiW: {
      const int feature = expr.param & 0xF;
      const int metric = (expr.param >> 4) & 0xF;
      const bool normalize = (expr.param & HA_PARAM_CLAMP) != 0;
      const int dimensions = (expr.param >> HA_PARAM_VORONOI_DIM_SHIFT) & 0x7;
      /* Args: vector (vec3), w, scale, detail, roughness, lacunarity, smoothness, exponent,
       * randomness. The 1D kernel only reads `w`, the 2D/3D kernels only the vector and the 4D
       * kernel both. */
      std::string args;
      if (dimensions == 1) {
        args = emit(*expr.args[1]);
      }
      else {
        args = emit_as_vector(*expr.args[0]);
        if (dimensions == 4) {
          args += ", " + emit(*expr.args[1]);
        }
      }
      for (int i = 2; i < 9; i++) {
        args += ", " + emit(*expr.args[i]);
      }
      args += ", " + std::to_string(feature) + ", " + std::to_string(metric) + ", " +
              (normalize ? "true" : "false");

      const char *function = "ha_voronoi_texture_3d";
      if (expr.op == HaOp::VoronoiTexture) {
        function = dimensions == 1 ? "ha_voronoi_texture_1d" :
                   dimensions == 2 ? "ha_voronoi_texture_2d" :
                   dimensions == 3 ? "ha_voronoi_texture_3d" :
                                     "ha_voronoi_texture_4d";
      }
      else if (expr.op == HaOp::VoronoiPosition) {
        function = dimensions == 2 ? "ha_voronoi_position_2d" :
                   dimensions == 3 ? "ha_voronoi_position_3d" :
                                     "ha_voronoi_position_4d";
      }
      else {
        function = dimensions == 1 ? "ha_voronoi_w_1d" : "ha_voronoi_w_4d";
      }
      return std::string(function) + "(" + args + ")";
    }
    case HaOp::VoronoiRadius: {
      const int dimensions = (expr.param >> HA_PARAM_VORONOI_DIM_SHIFT) & 0x7;
      const std::string vector = emit_as_vector(*expr.args[0]);
      const std::string w = emit(*expr.args[1]);
      const std::string scale = emit(*expr.args[2]);
      const std::string randomness = emit(*expr.args[3]);
      if (dimensions == 1) {
        return "ha_voronoi_radius_1d(" + w + ", " + scale + ", " + randomness + ")";
      }
      if (dimensions == 4) {
        return "ha_voronoi_radius_4d(" + vector + ", " + w + ", " + scale + ", " + randomness +
               ")";
      }
      return "ha_voronoi_radius_" + std::to_string(dimensions) + "d(" + vector + ", " + scale +
             ", " + randomness + ")";
    }
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
