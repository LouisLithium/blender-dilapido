/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Compiles the body of a `HW Accelerate` zone into a compute program.
 *
 * M86 supports exactly one shape, which is enough to validate the whole GPU data path:
 *
 *   zone input -> Set Position -> zone output
 *
 * where `Offset` is a field built from the nodes the expression IR understands (position, index,
 * vector math, math, combine/separate). Everything else returns `nullptr` and the zone falls back
 * to the regular CPU evaluation, so this file can only ever make things faster, not wrong.
 */

#pragma once

#include <memory>
#include <string>

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

namespace blender {
struct bNodeTree;
namespace bke {
struct GeometrySet;
class bNodeTreeZone;
}  // namespace bke
namespace geometry::ha {

/** How a reduction pass combines the per-element values. */
enum class HaAggregateOp {
  Sum,
  Min,
  Max,
  Mean,
};

struct HaAggregate {
  /** GLSL expression evaluated for every element (`vec3 position` / `int index` in scope). */
  std::string glsl_value;
  HaAggregateOp op = HaAggregateOp::Sum;
};

class HaZoneProgram {
 public:
  ~HaZoneProgram();

  /** Returns nullptr when the zone body cannot be compiled for the GPU. */
  static std::unique_ptr<HaZoneProgram> try_build(const bNodeTree &tree,
                                                  const bke::bNodeTreeZone &zone);

  /**
   * Evaluates the program for \a input. Returns false when the input cannot be handled (no mesh),
   * in which case the caller has to fall back to the CPU evaluation.
   */
  bool evaluate(const bke::GeometrySet &input, bke::GeometrySet &r_output) const;

  StringRefNull glsl_offset() const
  {
    return glsl_offset_;
  }

 private:
  std::string glsl_offset_;
  Vector<HaAggregate> aggregates_;
};

}  // namespace geometry::ha
}  // namespace blender
