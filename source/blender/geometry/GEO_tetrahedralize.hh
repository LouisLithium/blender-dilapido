/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Turning a closed surface mesh into a tetrahedral mesh.
 *
 * The elements are built on a regular lattice (see #tetrahedralize_mesh), which keeps the topology
 * trivial to build and makes the result deterministic: the tetrahedra of a cell never depend on a
 * global triangulation graph, only on which of its eight nodes are inside the input.
 */

#pragma once

#include <string>

#include "BLI_math_vector_types.hh"
#include "BLI_vector.hh"

namespace blender {

struct Mesh;

namespace geometry {

struct TetrahedralizeSettings {
  /** Spacing of the lattice the tetrahedra are built on, in world units. */
  float voxel_size = 0.05f;
  /**
   * Move the lattice nodes that are closer to the surface than one voxel onto the surface, so that
   * the boundary of the result follows the input instead of the staircase of the lattice.
   */
  bool snap_to_surface = true;
};

/**
 * The result of a tetrahedralization: a boundary surface plus the elements.
 *
 * The elements reference the vertices of the surface, so both are useful on their own (the surface
 * for rendering and collision, the elements for a volume constraint) without duplicating positions.
 */
struct TetrahedralizedMesh {
  /** Positions of the vertices of the mesh, in the space of the input mesh. */
  Vector<float3> vert_positions;
  /** Boundary triangles, wound so that the normal points out of the mesh. */
  Vector<int3> boundary_tris;
  /** The four vertex indices of each tetrahedron; the winding gives a positive volume. */
  Vector<int4> tets;
  /** Rest volume of each tetrahedron, in cubic world units. */
  Vector<float> tet_volumes;
  /** Sum of #tet_volumes. */
  float volume = 0.0f;
  /** Tetrahedra that were dropped because they had no volume (the snap can collapse a cell). */
  int dropped_tets = 0;
  /** Tetrahedra that the surface projection turned inside out and that had to be flipped. */
  int flipped_tets = 0;
  /** Number of lattice nodes that the flood fill had to classify (deep inside the mesh). */
  int interior_nodes = 0;

  bool is_empty() const
  {
    return tets.is_empty();
  }
};

/**
 * Fill the volume of \a mesh with tetrahedra.
 *
 * \param r_error: Set to a human readable message when false is returned.
 * \return False when the input has no faces, is too large for the requested voxel size, or the
 * signed distance field could not be built.
 */
bool tetrahedralize_mesh(const Mesh &mesh,
                         const TetrahedralizeSettings &settings,
                         TetrahedralizedMesh &r_result,
                         std::string &r_error);

}  // namespace geometry
}  // namespace blender
