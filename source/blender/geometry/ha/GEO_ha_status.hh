/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * What the last evaluation of a node did on the compute backend.
 *
 * The node UI uses it for two things: a small badge in the node header (only nodes that reported a
 * state get one, so it only appears where it means something) and a read only info block in the
 * node itself. A node publishes its state at the end of its execution, so the information is the
 * *actual* result of the last frame instead of "this node supports the device".
 *
 * ⚠️ **The key is the tree and node *name*, not the pointer.** Geometry nodes are evaluated on a
 * copy of the node tree (the depsgraph evaluates `bNodeTree` as an ID), so the `bNode *` a node
 * publishes with is a different object than the one the node editor draws. Measured: publishing
 * reported `tree=0x...F9308` while the UI holds `0x...F708` for the same tree, so a pointer keyed
 * table never produced a single badge. Names are preserved by the copy, which is why they work.
 */

#pragma once

#include <cstdint>
#include <string>

struct bNode;
struct bNodeTree;

namespace blender::geometry::ha {

enum class HaNodeState : int {
  /** The node ran on the compute backend for the last evaluation. */
  Device,
  /** The node is inside an `HW Accelerate` zone but the CPU path was used. */
  Fallback,
};

struct HaNodeStatus {
  HaNodeState state = HaNodeState::Device;
  /** Human readable backend name, e.g. "Vulkan". */
  std::string backend;
  /** Why the CPU path was used, empty for #HaNodeState::Device. */
  std::string reason;
  /** Elements the node handled, e.g. particles for the fluid solver. Zero when unknown. */
  int elements = 0;
  /**
   * Geometry nodes the device evaluated as one program (the `HW Accelerate` chain). Nodes that are
   * in the same zone share one transfer, so this is what tells the user that adding a node *inside*
   * the zone is cheap while a second zone is not. Zero when the node does not report it.
   */
  int steps = 0;
  /** Device storage buffers charged to this zone during the last evaluation. */
  int64_t vram_used = 0;
  int64_t vram_peak = 0;
  /** Zone budget in bytes; zero means unlimited. */
  int64_t vram_limit = 0;
};

/** Publishes the state of the last evaluation of \a node. */
void ha_status_set(const bNodeTree &tree, const bNode &node, const HaNodeStatus &status);

/** Forgets the state of \a node (used when it is not hardware accelerated at all). */
void ha_status_clear(const bNodeTree &tree, const bNode &node);

/** Reads the state of the last evaluation. False when the node did not publish one. */
bool ha_status_get(const bNodeTree &tree, const bNode &node, HaNodeStatus &r_status);

}  // namespace blender::geometry::ha
