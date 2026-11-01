/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 */

#include "GEO_ha_status.hh"

#include <mutex>

#include "BLI_map.hh"
#include "BLI_string_ref.hh"

#include "DNA_ID.h"
#include "DNA_node_types.h"

namespace blender::geometry::ha {

namespace {

/**
 * Geometry nodes run on a copy of the node tree, so neither the `bNodeTree *` nor the `bNode *` of
 * the evaluation is the one the node editor draws (see the header). The names survive the copy, so
 * they are the key.
 */
std::string status_key(const bNodeTree &tree, const bNode &node)
{
  /* `ID.name` carries a two character type prefix ("NT"). */
  return std::string(StringRef(tree.id.name + 2)) + '|' + StringRef(node.name);
}

/** Small on purpose: it only holds the nodes that reported a state in the last evaluation. */
constexpr int MAX_ENTRIES = 4096;

std::mutex &status_mutex()
{
  static std::mutex mutex;
  return mutex;
}

Map<std::string, HaNodeStatus> &status_map()
{
  static Map<std::string, HaNodeStatus> map;
  return map;
}

}  // namespace

void ha_status_set(const bNodeTree &tree, const bNode &node, const HaNodeStatus &status)
{
  std::lock_guard lock{status_mutex()};
  Map<std::string, HaNodeStatus> &map = status_map();
  if (map.size() > MAX_ENTRIES) {
    /* The UI hint is not worth unbounded memory. Losing it for a moment is fine, the next
     * evaluation fills it again. */
    map.clear();
  }
  map.add_overwrite(status_key(tree, node), status);
}

void ha_status_clear(const bNodeTree &tree, const bNode &node)
{
  std::lock_guard lock{status_mutex()};
  status_map().remove(status_key(tree, node));
}

bool ha_status_get(const bNodeTree &tree, const bNode &node, HaNodeStatus &r_status)
{
  std::lock_guard lock{status_mutex()};
  const HaNodeStatus *status = status_map().lookup_ptr(status_key(tree, node));
  if (status == nullptr) {
    return false;
  }
  r_status = *status;
  return true;
}

}  // namespace blender::geometry::ha
