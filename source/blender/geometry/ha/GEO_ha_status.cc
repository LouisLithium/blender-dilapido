/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 */

#include "GEO_ha_status.hh"

#include <mutex>

#include "BLI_map.hh"

#include "DNA_node_types.h"

namespace blender::geometry::ha {

namespace {

struct StatusEntry {
  /** The tree the node belonged to when it published, plus its identifier: a plain `bNode *` key
   * would be reused by a different node after an edit (see the caching pitfall in the rigid body
   * solver), and a stale badge is worse than no badge. */
  const bNodeTree *tree = nullptr;
  int identifier = 0;
  HaNodeStatus status;
};

/** Small on purpose: it only holds the nodes that reported a state in the last evaluation. */
constexpr int MAX_ENTRIES = 4096;

std::mutex &status_mutex()
{
  static std::mutex mutex;
  return mutex;
}

Map<const bNode *, StatusEntry> &status_map()
{
  static Map<const bNode *, StatusEntry> map;
  return map;
}

}  // namespace

void ha_status_set(const bNodeTree &tree, const bNode &node, const HaNodeStatus &status)
{
  std::lock_guard lock{status_mutex()};
  Map<const bNode *, StatusEntry> &map = status_map();
  if (map.size() > MAX_ENTRIES) {
    /* The UI hint is not worth unbounded memory. Losing it for a moment is fine, the next
     * evaluation fills it again. */
    map.clear();
  }
  map.add_overwrite(&node, StatusEntry{&tree, node.identifier, status});
}

void ha_status_clear(const bNodeTree &tree, const bNode &node)
{
  std::lock_guard lock{status_mutex()};
  Map<const bNode *, StatusEntry> &map = status_map();
  const StatusEntry *entry = map.lookup_ptr(&node);
  if (entry != nullptr && entry->tree == &tree && entry->identifier == node.identifier) {
    map.remove(&node);
  }
}

bool ha_status_get(const bNodeTree &tree, const bNode &node, HaNodeStatus &r_status)
{
  std::lock_guard lock{status_mutex()};
  const StatusEntry *entry = status_map().lookup_ptr(&node);
  if (entry == nullptr || entry->tree != &tree || entry->identifier != node.identifier) {
    return false;
  }
  r_status = entry->status;
  return true;
}

}  // namespace blender::geometry::ha
