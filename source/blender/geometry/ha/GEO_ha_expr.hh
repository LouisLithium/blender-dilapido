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
};

struct HaExpr {
  HaType type = HaType::Float;
  HaOp op = HaOp::Const;
  float value[3] = {0.0f, 0.0f, 0.0f};
  /** Used by #HaOp::AggregateRead. */
  int aggregate_slot = -1;
  Vector<const HaExpr *> args;
};

/** Owns the expression nodes of one compiled program. */
class HaExprPool {
 public:
  HaExpr &add(HaType type, HaOp op, std::initializer_list<const HaExpr *> args = {});
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

}  // namespace blender::geometry::ha
