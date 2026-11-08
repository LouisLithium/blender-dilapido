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
  Int,
  Float2,
  Float3,
  /** RGBA, the field system's `ColorGeometry4f`. Not the same as `Float4`: the implicit
   * conversions differ (`float3 -> Color` sets alpha to 1, `float3 -> float4` sets w to 0). */
  Color,
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
  /**
   * The same attribute read at an arbitrary element index (`Sample Index`). `args[0]` is the
   * index; `attribute_slot` is the attribute; `param` bit 0 = clamp the index. The element count
   * is uploaded in `gather_params[0]`.
   */
  AttributeGather,
  /**
   * A value from the program's uniform parameter buffer (an animated socket default). The slot is
   * `param_slot * 4`; the buffer is declared as `user_params[]`.
   */
  ParamRead,

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
  /** condition, false_value, true_value -> value (`GeometryNodeSwitch`). The condition is a
   * float; greater than zero selects `true_value`. */
  Select,
  /** index (int), item 0 .. item N-1 -> value (`GeometryNodeIndexSwitch`). Out-of-range indices
   * produce the type's default value, exactly like the CPU multi-function. */
  IndexSelect,

  /* --- Implicit type conversions (see `blenkernel/intern/type_conversions.cc`) ---------- */
  /** The result type is the expression's type; the source type is `args[0]->type`. */
  ConvertToFloat,
  ConvertToInt,
  ConvertToFloat2,
  ConvertToFloat3,
  ConvertToColor,

  /* --- Integer math (`FunctionNodeIntegerMath`) --------------------------------------- */
  /** int, int -> int; safe when the divisor is zero (returns 0). */
  DivideFloor,
  DivideCeil,
  DivideRound,

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
  /**
   * `Noise Texture > Color`, 3D: the same packing as #NoiseTexture. Returns a vec3 (RGB); the node's
   * alpha is always 1, and the IR has no RGBA type.
   */
  NoiseTextureColor,
  /**
   * `Wave Texture > Fac` (#HaExpr::param: wave type, bands/rings direction and profile packed in
   * 2 bit fields each). Args: vector, scale, distortion, detail, detail scale, detail roughness,
   * phase. A port of `node_shader_tex_wave.cc` on top of the Perlin fbm.
   */
  WaveTexture,
  /**
   * `Voronoi Texture > Distance`, 3D. Args: vector, scale, detail, roughness, lacunarity,
   * smoothness, exponent, randomness. #HaExpr::param packs the feature (bits 0-3), the distance
   * metric (bits 4-7) and `normalize` (bit 8). A port of the 3D path in `BLI_noise.cc`.
   */
  VoronoiTexture,
  /** `Voronoi Texture > Position` (2D/3D/4D): same args and packing as #VoronoiTexture. */
  VoronoiPosition,
  /** `Voronoi Texture > W` (1D/4D): same args and packing as #VoronoiTexture. */
  VoronoiW,
  /** `Voronoi Texture > Radius` (N-Sphere feature): args vector, w, scale, randomness. */
  VoronoiRadius,
  /**
   * `Float Curve`: `args[0]` is the value; `lut_offset` points into `lut_table[]`, where
   * `[0..CM_TABLE]` are the table Y values followed by mintable/range/first_x/last_x/ext_in/
   * ext_out/extrapolate. Evaluated by the `ha_float_curve()` helper.
   */
  LutSample,
  /** `Color Ramp`: `args[0]` is the factor; `lut_offset` points into the baked RGBA table. */
  ColorRamp,
  /** vec4 -> float: the alpha component of a `Color` expression (`Color Ramp > Alpha`). */
  SplitW,
};

/** Bit offset of the Voronoi dimension (1..4) in #HaExpr::param. */
inline constexpr int HA_PARAM_VORONOI_DIM_SHIFT = 9;

/** Bit offset of the Noise Texture dimension (1 or 3, 2D/4D pending) in #HaExpr::param. */
inline constexpr int HA_PARAM_NOISE_DIM_SHIFT = 9;

/** Bit of #HaExpr::param that enables clamping (`Mix` factor, `Map Range` result). */
inline constexpr int HA_PARAM_CLAMP = 1 << 8;

struct HaExpr {
  HaType type = HaType::Float;
  HaOp op = HaOp::Const;
  float value[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  /** Used by #HaType::Int constants. */
  int int_value = 0;
  /** Used by #HaOp::AggregateRead. */
  int aggregate_slot = -1;
  /** Used by #HaOp::AttributeRead. */
  int attribute_slot = -1;
  /** Used by #HaOp::ParamRead. */
  int param_slot = -1;
  /** Used by #HaOp::LutSample / #HaOp::ColorRamp: base offset into `lut_table[]`. */
  int lut_offset = -1;
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
  HaExpr &constant(int value);
  HaExpr &constant(const float2 &value);
  HaExpr &constant(const float3 &value);
  /** RGBA (the field system's `ColorGeometry4f`). */
  HaExpr &constant_color(const float4 &value);

 private:
  Vector<std::unique_ptr<HaExpr>> exprs_;
};

/**
 * CPU reference evaluation of \a expr for element \a index.
 * \param positions Interleaved element positions (3 floats per element).
 * \return The value; for #HaType::Float only `.x` is meaningful.
 */
float3 ha_evaluate(const HaExpr &expr, int index, Span<float> positions);

/** Typed result of #ha_evaluate_typed. `f` holds Float/Float2/Float3/Color, `i` holds Int. */
struct HaValue {
  HaType type = HaType::Float;
  float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int i = 0;
};

/** The same as #ha_evaluate but without promoting everything to a vector. */
HaValue ha_evaluate_typed(const HaExpr &expr, int index, Span<float> positions);

/** GLSL expression of the expression's natural type (float/int/vec2/vec3/vec4). */
std::string ha_emit_glsl_natural(const HaExpr &expr);

/** GLSL expression. In scope: `vec3 position` and `int index`. Always a `vec3`. */
std::string ha_emit_glsl(const HaExpr &expr);

/** The same for a scalar expression, without the implicit `vec3(...)` promotion. */
std::string ha_emit_glsl_scalar(const HaExpr &expr);

}  // namespace blender::geometry::ha
