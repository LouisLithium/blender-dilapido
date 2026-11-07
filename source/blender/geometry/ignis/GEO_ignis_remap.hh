/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Shared vocabulary for remapping a force by an Ignis grid value. `Dilapido Force` stores the
 * selector in its bundle; the solver samples the field because only it has the grids.
 */

#pragma once

#include "BLI_string_ref.hh"

namespace blender::geometry::ignis {

enum class ForceRemapField : int {
  None = 0,
  Pressure = 1,
  Density = 2,
  Temperature = 3,
  Fuel = 4,
  Flame = 5,
  VelocityMagnitude = 6,
};

enum class ForceRemapMode : int {
  /** Keep the force inside Min..Max. */
  Range = 0,
  /** Keep the force outside Min..Max. */
  Outside = 1,
  /** Keep the force above Min. */
  Above = 2,
  /** Keep the force below Max. */
  Below = 3,
};

namespace remap_keys {
constexpr const char *field = "field";
constexpr const char *mode = "mode";
constexpr const char *min = "min";
constexpr const char *max = "max";
constexpr const char *softness = "softness";
constexpr const char *scale = "scale";
}  // namespace remap_keys

}  // namespace blender::geometry::ignis
