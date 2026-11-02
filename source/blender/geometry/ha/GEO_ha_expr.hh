/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Expression IR for the hardware accelerated backend.
 *
 * The whole point of this file is that CPU and GPU evaluation share **one** description of the
 * computation: a field chain is compiled into a tree of #HaExpr, which is then either walked by
 * #ha_evaluate (reference implementation, used for the `Force CPU` backend and for the parity
 * test) or emitted as GLSL by #ha_emit_glsl (used by the compute backend).
 *
 * Keep the op set small and exact. Everything that cannot be expressed here must fall back to the
 * regular CPU node evaluation instead of being approximated.
 *
 * ⚠️ The CPU side of every op mirrors a specific Blender function (`safe_powf`, `compatible_signf`,
 * `smoothminf`, `wrapf`, `pingpongf`, `math::interpolate`, ...). When adding an op, copy the
 * implementation from there instead of inventing one: the parity test compares against the real
 * node evaluation, and a "close enough" formula shows up as a mismatch.
 */

#pragma once

#include <memory>
#include <string>

#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_vector.hh"

namespace blender::geometry::ha {

enum class HaType {
  Float,
  Float3,
};

/** Integer parameter of an op (#HaExpr::param): comparison, interpolation or boolean operation. */
enum class HaParam : int {
  None = 0,
  /** `Compare`: the `FunctionNodeCompare` / `Math > Compare` operation. */
  CompareLess = 1,
  CompareLessEqual = 2,
  CompareGreater = 3,
  CompareGreaterEqual = 4,
  CompareEqual = 5,
  CompareNotEqual = 6,
  /** `MapRange` interpolation. */
  MapRangeLinear = 1,
  MapRangeStepped = 2,
  MapRangeSmoothStep = 3,
  MapRangeSmootherStep = 4,
  /** `BoolMath`. */
  BoolAnd = 1,
  BoolOr = 2,
  BoolNot = 3,
  BoolNand = 4,
  BoolNor = 5,
  BoolXor = 6,
  BoolXnor = 7,
};

enum class HaOp {
  Const,
  /** vec3: the position of the current element. */
  Position,
  /** float: the index of the current element. */
  Index,
  Add,
  Sub,
  Mul,
  Div,
  Max,
  Min,
  Sin,
  Cos,
  /** float, float, float -> vec3. */
  Combine,
  /** vec3 -> float. */
  SplitX,
  SplitY,
  SplitZ,
  /** vec3 -> float. */
  Length,
  /** Same type as the argument. */
  Abs,
  Floor,
  Ceil,
  Fract,
  /** float -> float. */
  Sqrt,
  /** float, float -> float. */
  Pow,
  /** vec3 -> vec3 (zero vector when the length is 0). */
  Normalize,
  /** vec3, vec3 -> float. */
  Dot,
  Distance,
  /** value, min, max -> value (float). */
  Clamp,
  /**
   * A value produced by a reduction pass over the whole domain (e.g. `Attribute Statistic`).
   * #HaExpr::aggregate_slot indexes the program's aggregate buffer.
   */
  AggregateRead,
  /**
   * A POINT attribute uploaded for this evaluation (`Named Attribute`). #HaExpr::attribute_slot
   * indexes the program's attribute list; the buffer has one float per element for a scalar
   * attribute and three for a vector one.
   */
  AttributeRead,

  /* --- Math node: unary ---------------------------------------------------------------- */
  /** float -> float. Mirrors `math_functions.cc` / `multi_function_common.cc` exactly. */
  Round,
  Trunc,
  Sign,
  Exp,
  InvSqrt,
  Tan,
  Asin,
  Acos,
  Atan,
  Sinh,
  Cosh,
  Tanh,

  /* --- Math node: binary --------------------------------------------------------------- */
  /** float, float -> float. */
  LessThan,
  GreaterThan,
  Atan2,
  Log,
  Modulo,
  FlooredModulo,
  Snap,
  PingPong,

  /* --- Math node: three inputs --------------------------------------------------------- */
  /** float, float, float -> float. */
  MulAdd,
  /** value, epsilon, a, b -> float (the `Math > Compare` operation). */
  CompareEps,
  /** a, b, smoothing -> float. */
  SmoothMin,
  SmoothMax,
  /** value, max, min -> float. */
  Wrap,

  /* --- Vector math --------------------------------------------------------------------- */
  /** vec3, vec3 -> vec3. */
  Cross,
  Project,
  Reflect,

  /* --- Conversion nodes ---------------------------------------------------------------- */
  /** factor, a, b -> value (#HaExpr::param: 1 = clamp the factor). */
  Mix,
  /** value, from_min, from_max, to_min, to_max, [steps] -> value. */
  MapRange,
  /** a, b -> 0 or 1 (#HaExpr::param: #HaParam::Bool*). */
  BoolMath,
  /** a, b -> 0 or 1 (#HaExpr::param: #HaParam::Compare*). */
  Compare,
  /**
   * `White Noise Texture > Value` (#HaExpr::param: dimensions 1..4). 1D takes `W`, 2D/3D take the
   * vector, 4D takes the vector and `W`. The hash is a port of `BLI_noise.cc`, so it is bit
   * identical to the CPU.
   */
  WhiteNoise,
  /**
   * `Noise Texture > Factor`, 3D (`#HaExpr::param`: noise type in the low byte, bit 8 = normalize).
   * Args: position (vec3, already scaled), detail, roughness, lacunarity, offset, gain, distortion.
   * A port of `BLI_noise.cc`, so the values match the CPU.
   */
  NoiseTexture,
};

/** Bit of #HaExpr::param that enables clamping (`Mix` factor, `Map Range` result). */
inline constexpr int HA_PARAM_CLAMP = 1 << 8;

struct HaExpr {
  HaType type = HaType::Float;
  HaOp op = HaOp::Const;
  float value[3] = {0.0f, 0.0f, 0.0f};
  /** Used by #HaOp::AggregateRead. */
  int aggregate_slot = -1;
  /** Used by #HaOp::AttributeRead. */
  int attribute_slot = -1;
  /** Used by the ops that need an operation selector (see #HaParam). */
  int param = 0;
  Vector<const HaExpr *> args;
};

/** Owns the expression nodes of one compiled program. */
class HaExprPool {
 public:
  HaExpr &add(HaType type, HaOp op, std::initializer_list<const HaExpr *> args = {});
  HaExpr &add(HaType type, HaOp op, Span<const HaExpr *> args);
  HaExpr &constant(float value);
  HaExpr &constant(const float3 &value);

 private:
  Vector<std::unique_ptr<HaExpr>> exprs_;
};

/**
 * CPU reference evaluation of \a expr for element \a index.
 * \param positions Interleaved element positions (3 floats per element).
 * \return The value; for #HaType::Float only `.x` is meaningful.
 */
float3 ha_evaluate(const HaExpr &expr, int index, Span<float> positions);

/** GLSL expression. In scope: `vec3 position` and `int index`. */
std::string ha_emit_glsl(const HaExpr &expr);

/** The same for a scalar expression, without the implicit `vec3(...)` promotion. */
std::string ha_emit_glsl_scalar(const HaExpr &expr);

}  // namespace blender::geometry::ha
