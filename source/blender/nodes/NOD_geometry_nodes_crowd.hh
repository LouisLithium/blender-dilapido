/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Shared data format of the crowd nodes (`Create Crowd Agent`, `Join Agents`,
 * `Crowd Agent States`, `Animation Info`, `Init Crowd Data`).
 *
 * All crowd data is carried by nested bundles so it survives the simulation zone and can be
 * extended without changing socket types:
 *
 * - `Blender.CrowdAgent` (one template):
 *   `mesh` / `bones` (Geometry), `states` (Bundle), `spawn_group` / `template_id` (int),
 *   `weight` (float).
 * - `Blender.CrowdAgents`: `count` (int) + `agent_0..N` (nested `Blender.CrowdAgent`).
 * - `Blender.CrowdStates`: `count` (int) + `state_0..N` (`name`, `blend_in`, `blend_out`,
 *   `loop`, `speed`, `animation`).
 * - `Blender.CrowdAnimation`: `frame_start` / `frame_end` / `fps` / `speed` / `phase` /
 *   `root_motion` / `action_name`.
 *
 * Per-point instance attributes written by `Init Crowd Data`:
 *   `agent_id` (int, stable), `template_id` (int, -1 = unassigned), `spawn_group` (int),
 *   `state` (string), `state_id` (int), `phase` (float 0..1), `speed` (float), `scale` (float),
 *   `active` (bool), `physics_mode` (int, 0 = kinematic, 1 = ragdoll, 2 = disabled).
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "BLI_hash.h"
#include "BLI_string_ref.hh"

#include "DNA_meshdata_types.h"

namespace blender::nodes::crowd {

inline constexpr StringRefNull AGENT_TYPE = "Blender.CrowdAgent";
inline constexpr StringRefNull AGENTS_TYPE = "Blender.CrowdAgents";
inline constexpr StringRefNull STATES_TYPE = "Blender.CrowdStates";
inline constexpr StringRefNull ANIMATION_TYPE = "Blender.CrowdAnimation";

inline constexpr int MAX_AGENT_ID = 1'000'000;

inline void write_string(MStringProperty &dst, const StringRef src)
{
  const size_t size = std::min(size_t(src.size()), sizeof(dst.s));
  memcpy(dst.s, src.data(), size);
  dst.s_len = char(size);
}

inline std::string read_string(const MStringProperty &src)
{
  return std::string(src.s, size_t(src.s_len));
}

/** Deterministic 64-bit mix (same style as the solver's signature hash). */
inline uint64_t hash_mix(uint64_t hash, uint64_t value)
{
  return hash ^ (value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
}

inline uint64_t hash_ids(const int seed, const int id)
{
  return uint64_t(BLI_hash_int_2d(uint32_t(seed), uint32_t(id)));
}

/** Uniform float in [0, 1). */
inline float hash_to_unit(uint64_t hash)
{
  return float(uint32_t(hash)) * (1.0f / float(0xFFFFFFFFu));
}

}  // namespace blender::nodes::crowd
