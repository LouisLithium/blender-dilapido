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
 */

#pragma once

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
};

/** Publishes the state of the last evaluation of \a node. */
void ha_status_set(const bNodeTree &tree, const bNode &node, const HaNodeStatus &status);

/** Forgets the state of \a node (used when it is not hardware accelerated at all). */
void ha_status_clear(const bNodeTree &tree, const bNode &node);

/** Reads the state of the last evaluation. False when the node did not publish one. */
bool ha_status_get(const bNodeTree &tree, const bNode &node, HaNodeStatus &r_status);

}  // namespace blender::geometry::ha
