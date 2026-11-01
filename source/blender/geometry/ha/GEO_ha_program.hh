/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Compiles the body of a `HW Accelerate` zone into a compute program.
 *
 * The supported shape is one `Set Position` node whose geometry comes straight from the zone input
 * and goes straight into the zone output:
 *
 *   zone input -> Set Position -> zone output
 *
 * All three field inputs of that node are supported, built from the nodes the expression IR
 * understands (position, index, vector math, math, combine/separate, clamp, reductions).
 *
 * Everything else returns `nullptr` **with a reason** and the zone falls back to the regular CPU
 * evaluation, so this file can only ever make things faster, not wrong. Two rules keep it that way:
 *
 * - A socket the node's own implementation reads but this program does not (for example
 *   `Attribute Statistic > Selection`) has to make the compilation fail, never be ignored.
 * - A socket that only changes the result through a node property (for example the `Clamp`
 *   checkbox of the `Math` node) has to be honoured or rejected. `MATH_CLAMP` and the two `Clamp`
 *   modes are handled; that class of bug is exactly what the parity tests look for.
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

  /**
   * Returns nullptr when the zone body cannot be compiled for the GPU. \a r_reason (optional) then
   * describes what stopped it, in a form that is meant for the user: the zone shows it in its node
   * hint and reports it in `Strict` mode instead of falling back silently.
   */
  static std::unique_ptr<HaZoneProgram> try_build(const bNodeTree &tree,
                                                  const bke::bNodeTreeZone &zone,
                                                  std::string *r_reason = nullptr);

  /**
   * Evaluates the program for \a input. Returns false when the input cannot be handled (no mesh),
   * in which case the caller has to fall back to the CPU evaluation.
   * \param r_elements Number of elements that were processed, for the node UI hint.
   */
  bool evaluate(const bke::GeometrySet &input,
                bke::GeometrySet &r_output,
                int *r_elements = nullptr) const;

  /** The GLSL expression of the final position, for debug output. */
  StringRefNull glsl_position() const
  {
    return glsl_position_;
  }

  /** The GLSL expression of the selection, for debug output. */
  StringRefNull glsl_selection() const
  {
    return glsl_selection_;
  }

 private:
  std::string glsl_position_;
  /** `1.0` when every element is affected, otherwise a scalar expression compared against zero. */
  std::string glsl_selection_;
  Vector<HaAggregate> aggregates_;
};

}  // namespace geometry::ha
}  // namespace blender
