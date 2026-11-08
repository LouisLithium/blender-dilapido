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

#include "GEO_ha_expr.hh"

namespace blender {
struct bNodeTree;
struct Material;
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
  /** 1 for a float reduction, 3 for a vector one. */
  int components = 1;
  /** True when the expression reads `aggregate[]` (variance: the mean from a previous pass). */
  bool read_aggregate = false;
  /** True for `Attribute Statistic > Median`: computed on the host with a sort. */
  bool median = false;
};

/** One running-total field (`Accumulate Field`, P1-10). */
struct HaAccumulation {
  /** GLSL expression of the accumulated value (component count from #type). */
  std::string glsl_value;
  HaType type = HaType::Float;
  /** 0 = leading (inclusive), 1 = trailing (exclusive), 2 = total. */
  int mode = 0;
  /** GLSL `int` expression of the group id; empty for the ungrouped case. */
  std::string glsl_group;
  bool grouped = false;
};

/** One geometry modifying node of the chain: where every element ends up, and which ones move. */
struct HaProgramStep {
  /** `vec3` expression, with `vec3 position` / `int index` in scope. */
  std::string glsl_position;
  /** Scalar expression, compared against zero (the field system's `float -> bool` conversion). */
  std::string glsl_selection;
};

/** Where a #HaAttribute's values come from. `Input` reads the zone input geometry; the others are
 * computed by the host from the mesh (P1-04 / P1-05). */
enum class HaAttributeKind {
  Input,
  FaceArea,
  Normal,
  TrueNormal,
  VertexCount,
  VertexFaceCount,
  /** `Edge Vertices`: the two endpoint vertex indices of every edge (P1-08). */
  EdgeVertex1Index,
  EdgeVertex2Index,
  /** `Edge Vertices`: the positions of the two endpoint vertices (valid before a position step). */
  EdgeVertex1Position,
  EdgeVertex2Position,
  /** `UV Tangent`: host-computed CORNER tangents from the referenced UV attribute. */
  UVTangent,
};

/** One POINT attribute the program reads, uploaded once per evaluation. */
struct HaAttribute {
  std::string name;
  HaType type = HaType::Float;
  /** 1 for a float/int attribute, 2/3/4 for a vector one. */
  int components = 1;
  HaAttributeKind kind = HaAttributeKind::Input;
  /** `bke::AttrDomain` as an int (the domain the values are uploaded in). */
  int domain = 0;
  /** Extra input for synthetic attributes: the source attribute name (`UV Tangent` UV map). */
  std::string ref_name;
  /** `UV Tangent` method (0 = Exact, 1 = Fast) and the source UV data type (CD_PROP_FLOAT2/3). */
  int ref_method = 0;
  int ref_data_type = 0;
};

/**
 * One animated socket default (`driver` / action). The value is read from the evaluated node tree
 * every frame and uploaded to the uniform parameter buffer; it is never inlined into the shader
 * source, so the shader does not change when only the value changes.
 */
struct HaParamSource {
  std::string node_name;
  int socket_index = 0;
  HaType type = HaType::Float;
  /** True for `Named Attribute > Exists`: the value is 0/1 depending on the input attribute. */
  bool attribute_exists = false;
  std::string attribute_name;
  /** `bke::AttrDomain` as an int, to keep this header free of the attribute header. */
  int attribute_domain = 0;
};

/** Element domain of a #HaElementStep / #HaElementStore. */
enum class HaElementDomain {
  Face,
  Edge,
  Corner,
};

/** One `Store Named Attribute` node at the end of the chain (written back after the steps). */
struct HaStore {
  std::string name;
  HaType type = HaType::Float;
  /** 1 for a float/int attribute, 2/3/4 for a vector one. */
  int components = 1;
  /** GLSL expression, evaluated with `vec3 position` (the *final* position) and `int index`. */
  std::string glsl_value;
  /** Scalar expression; empty means "all elements". Greater than zero means "write this one". */
  std::string glsl_selection;
  /** Slot in the program's attribute list; downstream `Named Attribute` reads bind this slot. */
  int attribute_slot = -1;
};

/** One entry of the linear chain: a position step, a POINT store, or a `Capture Attribute` node. */
struct HaChainOperation {
  enum class Kind {
    Step,
    Store,
    Capture,
  };
  Kind kind = Kind::Step;
  /** Index into #HaZoneProgram::steps_ / #HaZoneProgram::stores_ / #HaZoneProgram::captures_. */
  int index = 0;
};

/**
 * `Capture Attribute` (P1-11): a pass-through node that snapshots fields at its point in the chain.
 * The captured values are written to internal attribute buffers by ordinary store kernels; a
 * downstream node reads them through `HaOp::AttributeRead`, exactly like the zone input attributes.
 */
struct HaCapture {
  /** Range into #HaZoneProgram::capture_stores_. */
  int store_offset = 0;
  int store_count = 0;
};

/**
 * A `Store Named Attribute` on a domain that is not POINT (FACE / EDGE / CORNER). It is executed as
 * its own element pass, after the position chain and after the structural element nodes, so the
 * value field sees the geometry as it leaves the zone.
 */
struct HaElementStore {
  std::string name;
  HaElementDomain domain = HaElementDomain::Face;
  HaType type = HaType::Float;
  int components = 1;
  /** GLSL expression, evaluated with `int index` (the element index) in scope. */
  std::string glsl_value;
  /** Scalar expression; greater than zero means "write this element". */
  std::string glsl_selection;
  /** Value the attribute has when it does not exist yet (zero for `Store`). */
  float default_value = 0.0f;
  /** Attributes read by this step's fields, bound at slot 3 + index. */
  Vector<HaAttribute> attributes;
  /** Reductions over this element domain (`Attribute Statistic` inside the value field). */
  Vector<HaAggregate> aggregates;
  /** True when the value/selection reads `Edge Vertices > Position` (binds `position_gather[]`). */
  bool uses_position_gather = false;
};

/** Type of the attribute an element pass writes. */
enum class HaElementType {
  Bool,
  Int,
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
  /** Attribute type written back (bool for `sharp_*`, int for `material_index`). */
  HaElementType type = HaElementType::Bool;
  /** Scalar expression, evaluated with `int index` (the element index) in scope. */
  std::string glsl_value;
  /** Scalar expression; greater than zero means "write this element". */
  std::string glsl_selection;
  /**
   * `Set Material`: the value is the material's slot index on the mesh, which is only known at
   * execution time (the material is appended there), so it replaces #glsl_value.
   */
  bool material_step = false;
  struct Material *material = nullptr;
  /** True when the selection is the constant `true`; the CPU takes a different slot path then. */
  bool all_selected = true;
  /**
   * Value the attribute has when it does not exist yet. `Set Shade Smooth` creates `sharp_*` with
   * `true` (measured against the CPU), so unselected elements of a missing attribute read as sharp.
   */
  float default_value = 0.0f;
  /** Attributes read by this step's fields, bound at slot 3 + index (same layout as POINT). */
  Vector<HaAttribute> attributes;
  /** Reductions over this element domain (`Attribute Statistic` inside the fields). */
  Vector<HaAggregate> aggregates;
  /** True when the fields read `Edge Vertices > Position` (binds `position_gather[]`). */
  bool uses_position_gather = false;
};

/** One `Flip Faces` node: reverses the winding of the selected faces (CORNER domain). */
struct HaCornerStep {
  /** Per-face scalar expression, evaluated with `int index` (the face index) in scope. */
  std::string glsl_selection;
  /** Face attributes read by the selection. */
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
                int *r_elements = nullptr,
                const bNodeTree *tree = nullptr) const;

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

  /** Number of animated socket defaults uploaded per evaluation. */
  int param_count() const
  {
    return int(params_.size());
  }

  /** Number of FACE/EDGE attribute passes. */
  int element_step_count() const
  {
    return int(element_steps_.size());
  }

  /** Number of FACE/EDGE/CORNER `Store Named Attribute` passes. */
  int element_store_count() const
  {
    return int(element_stores_.size());
  }

  /** Number of `Flip Faces` passes. */
  int corner_step_count() const
  {
    return int(corner_steps_.size());
  }

  /** Number of `Capture Attribute` nodes. */
  int capture_count() const
  {
    return int(captures_.size());
  }

  /** Number of `Accumulate Field` outputs. */
  int accumulation_count() const
  {
    return int(accumulations_.size());
  }

  /** Human readable form of the compiled chain, for the debug log. */
  std::string debug_description() const;

 private:
  Vector<HaProgramStep> steps_;
  Vector<HaAggregate> aggregates_;
  /** Running totals for `Accumulate Field`, uploaded to `accumulation[]` (slot 12). */
  Vector<HaAccumulation> accumulations_;
  bool has_accumulation_ = false;
  /** Attributes read by the program, in binding order (slot 3 + index). */
  Vector<HaAttribute> attributes_;
  /** Animated socket defaults, uploaded to `user_params[]` (slot 15). */
  Vector<HaParamSource> params_;
  /** `Store Named Attribute` nodes (POINT), in evaluation order. */
  Vector<HaStore> stores_;
  /** Internal stores of the `Capture Attribute` nodes; downstream reads bind these. */
  Vector<HaStore> capture_stores_;
  /** `Capture Attribute` nodes, in chain order. */
  Vector<HaCapture> captures_;
  /** Linear order of the position steps and the POINT stores. */
  Vector<HaChainOperation> operations_;
  /** `Store Named Attribute` nodes on FACE / EDGE / CORNER, in evaluation order. */
  Vector<HaElementStore> element_stores_;
  /** FACE / EDGE attribute passes (before the stores; they do not touch positions). */
  Vector<HaElementStep> element_steps_;
  /** `Flip Faces` passes (after the element passes; they rewrite the corner topology). */
  Vector<HaCornerStep> corner_steps_;
  /** True when the program contains an `AttributeGather` (needs `gather_params[]`). */
  bool has_gather_ = false;
  /** Baked LUT tables for `Float Curve` / `Color Ramp` (needs `lut_table[]`). */
  Vector<float> lut_data_;
  bool has_lut_ = false;
};

}  // namespace geometry::ha
}  // namespace blender
