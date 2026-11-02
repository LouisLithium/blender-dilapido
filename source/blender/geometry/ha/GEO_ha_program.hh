/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Compiles the body of a `HW Accelerate` zone into a compute program.
 *
 * A program is a **chain** of `Set Position` nodes that starts at the zone input and ends at the
 * zone output:
 *
 *   zone input -> Set Position -> Set Position -> ... -> zone output
 *
 * The whole chain is evaluated on the device with **one upload and one download**: the steps
 * ping-pong between two device buffers, so adding a node to the zone costs a dispatch and nothing
 * else (measured: the transfer is ~14 ms per million elements, a dispatch ~4 ms).
 *
 * Everything else returns `nullptr` **with a reason** and the zone falls back to the regular CPU
 * evaluation, so this file can only ever make things faster, not wrong. Three rules keep it that
 * way:
 *
 * - A socket the node's own implementation reads but this program does not (for example
 *   `Attribute Statistic > Selection`) has to make the compilation fail, never be ignored.
 * - A socket that only changes the result through a node property (for example the `Clamp`
 *   checkbox of the `Math` node) has to be honoured or rejected. `MATH_CLAMP` and the two `Clamp`
 *   modes are handled; that class of bug is exactly what the parity tests look for.
 * - The device buffers live for exactly one evaluation (the steps of the chain stay resident for
 *   the duration of that evaluation). Keeping them across frames buys nothing measurable: the
 *   depsgraph does not re-evaluate a zone whose inputs did not change, so the interesting frames
 *   always have new data to upload.
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

/** One geometry modifying node of the chain: where every element ends up, and which ones move. */
struct HaProgramStep {
  /** `vec3` expression, with `vec3 position` / `int index` in scope. */
  std::string glsl_position;
  /** Scalar expression, compared against zero (the field system's `float -> bool` conversion). */
  std::string glsl_selection;
};

/** One POINT attribute the program reads, uploaded once per evaluation. */
struct HaAttribute {
  std::string name;
  /** 1 for a float attribute, 3 for a vector one. */
  int components = 1;
};

/** One `Store Named Attribute` node at the end of the chain (written back after the steps). */
struct HaStore {
  std::string name;
  /** 1 for a float attribute, 3 for a vector one. */
  int components = 1;
  /** GLSL expression, evaluated with `vec3 position` (the *final* position) and `int index`. */
  std::string glsl_value;
};

/** Element domain of a #HaElementStep. Only the domains with supported nodes get an entry. */
enum class HaElementDomain {
  Face,
  Edge,
};

/**
 * One node that writes an attribute on a domain that has no positions of its own (FACE / EDGE).
 * Currently produced by `Set Shade Smooth`, which needs both of its fields to be constant: a field
 * would have to be evaluated on that domain, which needs its own attribute bindings.
 */
struct HaElementStep {
  std::string attribute;
  HaElementDomain domain = HaElementDomain::Face;
  /** 1 for a scalar attribute, 3 for a vector one. */
  int components = 1;
  /** Scalar expression, evaluated with `int index` (the element index) in scope. */
  std::string glsl_value;
  /** Scalar expression; greater than zero means "write this element". */
  std::string glsl_selection;
  /**
   * Value the attribute has when it does not exist yet. `Set Shade Smooth` creates `sharp_*` with
   * `true` (measured against the CPU), so unselected elements of a missing attribute read as sharp.
   */
  float default_value = 0.0f;
  /** Attributes read by this step's fields, bound at slot 3 + index (same layout as POINT). */
  Vector<HaAttribute> attributes;
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

  int step_count() const
  {
    return int(steps_.size());
  }

  /** Number of POINT attributes the program reads. */
  int attribute_count() const
  {
    return int(attributes_.size());
  }

  /** Number of `Store Named Attribute` nodes the program writes back. */
  int store_count() const
  {
    return int(stores_.size());
  }

  /** Number of FACE/EDGE attribute passes. */
  int element_step_count() const
  {
    return int(element_steps_.size());
  }

  /** Human readable form of the compiled chain, for the debug log. */
  std::string debug_description() const;

 private:
  Vector<HaProgramStep> steps_;
  Vector<HaAggregate> aggregates_;
  /** Attributes read by the program, in binding order (slot 3 + index). */
  Vector<HaAttribute> attributes_;
  /** `Store Named Attribute` nodes, in evaluation order (all after the position chain). */
  Vector<HaStore> stores_;
  /** FACE / EDGE attribute passes (all after the stores; they do not touch positions). */
  Vector<HaElementStep> element_steps_;
};

}  // namespace geometry::ha
}  // namespace blender
