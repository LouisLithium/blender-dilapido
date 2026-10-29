/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * The tetrahedra are built on a regular lattice, which is the cheapest topology that can be
 * generated from an implicit description of the volume:
 *
 * 1. The signed distance field of the input decides which lattice nodes are solid. Near the surface
 *    the field is an accurate distance; farther inside it is clamped to the interior value, so the
 *    sign is what counts. A flood fill resolves the nodes that read the background (they are far
 *    outside, or inside a shape that the level set left unfilled).
 * 2. Every node of a cell that is part of the result has to be solid or at most one voxel outside,
 *    and the nodes of that outer shell are pulled onto the surface. The shell is what keeps the
 *    outermost layer of cells: without it a cell whose corner sticks out by more than half a voxel
 *    would be dropped and the boundary would fall back onto the layer behind it, which is the first
 *    order shrink of a naive voxel fill.
 * 3. A cell becomes six tetrahedra when all eight of its nodes are solid or in the shell and at
 *    least one of them is strictly inside. Cells that only touch the surface from the outside are
 *    left out; the ones that the snap collapsed onto the surface have no volume and are dropped,
 *    which never opens the boundary: a dropped element only removes triangles, it cannot create a
 *    hole, because the boundary is made of the faces that appear exactly once.
 * 4. Finally the vertices of the boundary faces - and only those - are projected onto the input
 *    surface. After step 2 the outermost elements can still reach the surface through a node that
 *    is half a voxel deep (their outer cells were the ones that collapsed), and this is what makes
 *    the boundary follow the input exactly instead of the lattice.
 *
 * Everything that scales with the lattice is parallel: sampling the field, the surface projection
 * (one BVH query per boundary vertex), placing the vertices, the elements and their rest volumes,
 * and sorting the faces. The only serial parts are the prefix sums per slice and the flood fill (a
 * sequential graph traversal by nature).
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>

#include "BLI_array.hh"
#include "BLI_bounds.hh"
#include "BLI_bounds_types.hh"
#include "BLI_index_range.hh"
#include "BLI_math_base.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_sort.hh"
#include "BLI_span.hh"
#include "BLI_task.hh"
#include "BLI_vector.hh"

#include "BKE_bvhutils.hh"
#include "BKE_mesh.hh"

#include "GEO_mesh_to_volume.hh"
#include "GEO_tetrahedralize.hh"

#ifdef WITH_OPENVDB
#  include "BKE_volume_grid.hh"

#  include <openvdb/openvdb.h>
#endif

namespace blender::geometry {

#ifdef WITH_OPENVDB

namespace {

/**
 * Soft limit for the lattice. Every node costs ~40 bytes of scratch and contributes up to six
 * tetrahedra, so anything beyond this is a mistake in the voxel size rather than a use case.
 */
constexpr int64_t max_nodes_num = 4'000'000;

/** Half the width of the narrow band of the distance field, in voxels. */
constexpr float band_width_in_voxels = 3.0f;

/** What the distance field says about a lattice node. */
enum NodeState : int8_t {
  /** Farther from the surface than the band, in either direction: the flood fill decides. */
  NODE_UNKNOWN = 0,
  /** Inside the mesh. */
  NODE_INSIDE = 1,
  /** Outside the mesh and farther than a voxel from it: not part of the result. */
  NODE_OUTSIDE = 2,
  /** Outside the mesh but within a voxel of it: pulled onto the surface and used by the elements. */
  NODE_SHELL = 3,
};

/**
 * The eight nodes of a cell, as offsets. The order also defines the tetrahedra below, so it is not
 * free: `0` and `6` are the opposite corners of the cell diagonal the decomposition uses.
 */
const int corner_offsets[8][3] = {{0, 0, 0},
                                  {1, 0, 0},
                                  {1, 1, 0},
                                  {0, 1, 0},
                                  {0, 0, 1},
                                  {1, 0, 1},
                                  {1, 1, 1},
                                  {0, 1, 1}};

/**
 * The six tetrahedra of a cell, split along the diagonal from corner 0 to corner 6. All of them use
 * that diagonal, which is what makes the decomposition consistent between neighboring cells: the
 * triangles on the shared faces of two cells match, so the interior faces always cancel out in the
 * boundary extraction.
 *
 * Every tetrahedron is wound so that its volume is positive (see #tet_volume).
 */
const int tet_corners[6][4] = {
    {0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}};

inline int lattice_index(const int3 &nodes, const int i, const int j, const int k)
{
  return (k * nodes.y + j) * nodes.x + i;
}

/** Trilinear sample of the distance field at a point in the space of the input mesh. */
inline float sample_distance(const openvdb::FloatGrid &grid,
                             const openvdb::FloatGrid::ConstAccessor &accessor,
                             const float3 &point)
{
  const openvdb::Vec3d index_position = grid.transform().worldToIndex(
      openvdb::Vec3d(point.x, point.y, point.z));
  const int i = int(std::floor(index_position.x()));
  const int j = int(std::floor(index_position.y()));
  const int k = int(std::floor(index_position.z()));
  const float tx = float(index_position.x() - double(i));
  const float ty = float(index_position.y() - double(j));
  const float tz = float(index_position.z() - double(k));

  const float v000 = accessor.getValue(openvdb::Coord(i, j, k));
  const float v100 = accessor.getValue(openvdb::Coord(i + 1, j, k));
  const float v010 = accessor.getValue(openvdb::Coord(i, j + 1, k));
  const float v110 = accessor.getValue(openvdb::Coord(i + 1, j + 1, k));
  const float v001 = accessor.getValue(openvdb::Coord(i, j, k + 1));
  const float v101 = accessor.getValue(openvdb::Coord(i + 1, j, k + 1));
  const float v011 = accessor.getValue(openvdb::Coord(i, j + 1, k + 1));
  const float v111 = accessor.getValue(openvdb::Coord(i + 1, j + 1, k + 1));

  const float v00 = v000 + (v100 - v000) * tx;
  const float v10 = v010 + (v110 - v010) * tx;
  const float v01 = v001 + (v101 - v001) * tx;
  const float v11 = v011 + (v111 - v011) * tx;
  const float v0 = v00 + (v10 - v00) * ty;
  const float v1 = v01 + (v11 - v01) * ty;
  return v0 + (v1 - v0) * tz;
}

/**
 * Samples the field on the whole lattice and classifies the nodes from its sign. Parallel over the
 * z slices, since the arrays are ordered along z and every task then writes its own contiguous
 * block.
 */
void classify_nodes(const openvdb::FloatGrid &grid,
                    const int3 &nodes,
                    const float3 &origin,
                    const float voxel_size,
                    const float band,
                    const float shell_distance,
                    const MutableSpan<int8_t> r_states)
{
  threading::parallel_for(IndexRange(nodes.z), 1, [&](const IndexRange k_range) {
    const openvdb::FloatGrid::ConstAccessor accessor = grid.getConstAccessor();
    for (const int k : k_range) {
      const float pz = origin.z + float(k) * voxel_size;
      for (int j = 0; j < nodes.y; j++) {
        const float py = origin.y + float(j) * voxel_size;
        const int row = (k * nodes.y + j) * nodes.x;
        for (int i = 0; i < nodes.x; i++) {
          const float distance = sample_distance(
              grid, accessor, float3(origin.x + float(i) * voxel_size, py, pz));
          if (distance <= 0.0f) {
            r_states[row + i] = NODE_INSIDE;
          }
          else if (distance <= shell_distance) {
            r_states[row + i] = NODE_SHELL;
          }
          else if (distance < band * 0.9f) {
            r_states[row + i] = NODE_OUTSIDE;
          }
          else {
            /* The value is the background of the narrow band: the node is either far outside the
             * mesh or inside a shape that the level set left unfilled. #flood_fill_outside decides,
             * and treats what it cannot reach as solid. */
            r_states[row + i] = NODE_UNKNOWN;
          }
        }
      }
    }
  });
}

/**
 * Marks every node that is not inside the mesh as outside, walking from the nodes that the field
 * already knows to be outside. Nodes it cannot reach are left unknown (and become solid) so an
 * unfilled interior is not hollowed out.
 */
void flood_fill_outside(const int3 &nodes, const MutableSpan<int8_t> states)
{
  const int nx = nodes.x;
  const int ny = nodes.y;
  const int nz = nodes.z;
  Vector<int> queue;
  queue.reserve(1024);

  for (int k = 0; k < nz; k++) {
    const bool border_k = (k == 0 || k == nz - 1);
    for (int j = 0; j < ny; j++) {
      const bool border_j = (j == 0 || j == ny - 1);
      const int row = (k * ny + j) * nx;
      for (int i = 0; i < nx; i++) {
        const int index = row + i;
        const int8_t state = states[index];
        if (state == NODE_OUTSIDE || state == NODE_SHELL) {
          queue.append(index);
        }
        else if (state == NODE_UNKNOWN && (border_k || border_j || i == 0 || i == nx - 1)) {
          /* The background value and on the boundary of the lattice: the mesh is inside the
           * bounding box, so this node cannot be solid. */
          states[index] = NODE_OUTSIDE;
          queue.append(index);
        }
      }
    }
  }

  for (int head = 0; head < int(queue.size()); head++) {
    const int index = queue[head];
    const int i = index % nx;
    const int j = (index / nx) % ny;
    const int k = index / (nx * ny);
    const int neighbours[6] = {i > 0 ? index - 1 : -1,
                               i + 1 < nx ? index + 1 : -1,
                               j > 0 ? index - nx : -1,
                               j + 1 < ny ? index + nx : -1,
                               k > 0 ? index - nx * ny : -1,
                               k + 1 < nz ? index + nx * ny : -1};
    for (const int neighbour : neighbours) {
      if (neighbour >= 0 && states[neighbour] == NODE_UNKNOWN) {
        states[neighbour] = NODE_OUTSIDE;
        queue.append(neighbour);
      }
    }
  }
}

/** The face \a f of a tetrahedron, wound so that its normal points out of the tetrahedron. */
inline int3 tet_face(const int4 &tet, const int f)
{
  switch (f) {
    case 0:
      return int3(tet.y, tet.z, tet.w);
    case 1:
      return int3(tet.x, tet.w, tet.z);
    case 2:
      return int3(tet.x, tet.y, tet.w);
    default:
      return int3(tet.x, tet.z, tet.y);
  }
}

inline int3 sorted_triple(int3 tri)
{
  if (tri.x > tri.y) {
    std::swap(tri.x, tri.y);
  }
  if (tri.y > tri.z) {
    std::swap(tri.y, tri.z);
  }
  if (tri.x > tri.y) {
    std::swap(tri.x, tri.y);
  }
  return tri;
}

inline float tet_volume(const float3 &a, const float3 &b, const float3 &c, const float3 &d)
{
  return math::dot(b - a, math::cross(c - a, d - a)) / 6.0f;
}

/**
 * The boundary of an element set is the set of its faces that appear exactly once. Sorting the
 * faces by their vertex triple groups the pairs, which is a good deal faster than a hash set of
 * four triangles per element and leaves the output order deterministic.
 *
 * \param r_boundary_verts: Marked (1) for every vertex the boundary uses when not empty.
 */
void build_boundary(const Span<int4> tets,
                    Vector<int3> &r_faces,
                    const MutableSpan<int8_t> r_boundary_verts)
{
  r_faces.clear();
  if (tets.is_empty()) {
    return;
  }

  Array<int4> faces(int64_t(tets.size()) * 4);
  threading::parallel_for(tets.index_range(), 1024, [&](const IndexRange range) {
    for (const int t : range) {
      const int4 &tet = tets[t];
      for (int f = 0; f < 4; f++) {
        const int3 tri = sorted_triple(tet_face(tet, f));
        faces[t * 4 + f] = int4(tri.x, tri.y, tri.z, t * 4 + f);
      }
    }
  });

  blender::parallel_sort(faces.begin(), faces.end(), [](const int4 &a, const int4 &b) {
    if (a.x != b.x) {
      return a.x < b.x;
    }
    if (a.y != b.y) {
      return a.y < b.y;
    }
    return a.z < b.z;
  });

  for (int i = 0; i < int(faces.size());) {
    int run_end = i + 1;
    while (run_end < int(faces.size()) && faces[run_end].x == faces[i].x &&
           faces[run_end].y == faces[i].y && faces[run_end].z == faces[i].z)
    {
      run_end++;
    }
    if (run_end - i == 1) {
      const int face_id = faces[i].w;
      const int3 tri = tet_face(tets[face_id / 4], face_id % 4);
      r_faces.append(tri);
      if (!r_boundary_verts.is_empty()) {
        r_boundary_verts[tri.x] = 1;
        r_boundary_verts[tri.y] = 1;
        r_boundary_verts[tri.z] = 1;
      }
    }
    i = run_end;
  }
}

/** Volume of every element, in parallel. */
void compute_tet_volumes(const Span<int4> tets,
                         const Span<float3> positions,
                         const MutableSpan<float> r_volumes)
{
  threading::parallel_for(tets.index_range(), 1024, [&](const IndexRange range) {
    for (const int t : range) {
      const int4 &tet = tets[t];
      r_volumes[t] = tet_volume(
          positions[tet.x], positions[tet.y], positions[tet.z], positions[tet.w]);
    }
  });
}

}  // namespace

bool tetrahedralize_mesh(const Mesh &mesh,
                         const TetrahedralizeSettings &settings,
                         TetrahedralizedMesh &r_result,
                         std::string &r_error)
{
  r_result = {};
  const Span<float3> mesh_positions = mesh.vert_positions();
  const Span<int3> mesh_corner_tris = mesh.corner_tris();
  if (mesh_positions.is_empty() || mesh_corner_tris.is_empty()) {
    r_error = "The input mesh has no faces";
    return false;
  }
  const std::optional<Bounds<float3>> bounds = mesh.bounds_min_max();
  if (!bounds) {
    r_error = "The input mesh has no bounds";
    return false;
  }

  const bool debug = std::getenv("TET_DEBUG") != nullptr;
  auto stamp = [&](const char *name) {
    if (!debug) {
      return;
    }
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    printf("TET_DEBUG %-10s %7.1f ms\n",
           name,
           std::chrono::duration<double, std::milli>(now - last).count());
    last = now;
  };

  const float voxel_size = math::clamp(settings.voxel_size, 1e-5f, 1e5f);
  const float3 origin = bounds->min;
  const float3 extent = bounds->max - bounds->min;

  /* The lattice covers the bounding box and has at least one cell per axis, so a flat object still
   * produces a (empty) result rather than nothing at all. */
  auto cell_count = [&](const float length) {
    return std::max(1, int(std::ceil(length / voxel_size - 1e-4f)));
  };
  const int3 cells(cell_count(extent.x), cell_count(extent.y), cell_count(extent.z));
  const int3 nodes(cells.x + 1, cells.y + 1, cells.z + 1);
  const int64_t nodes_num = int64_t(nodes.x) * int64_t(nodes.y) * int64_t(nodes.z);
  if (nodes_num > max_nodes_num) {
    r_error = "The voxel size is too small for this mesh: " + std::to_string(nodes_num) +
              " lattice nodes, the maximum is " + std::to_string(max_nodes_num) +
              ". Use a larger voxel size.";
    return false;
  }

  /* How far the nodes of the outer shell and the vertices of the boundary may travel to land on the
   * surface. A whole voxel for the shell is what keeps the outermost layer of cells, and one and a
   * half for the boundary covers the case of an element that reaches the surface through a node
   * that is half a voxel deep. */
  const float shell_distance = settings.snap_to_surface ? 1.0f * voxel_size : 0.0f;
  const float boundary_distance = settings.snap_to_surface ? 1.5f * voxel_size : 0.0f;
  const float band = band_width_in_voxels * voxel_size;

  const bke::VolumeGrid<float> sdf = [&]() {
    /* `mesh_to_sdf_grid` takes mutable spans, so the shared arrays are copied here once. */
    Vector<float3> positions(mesh_positions.begin(), mesh_positions.end());
    const Span<int> corner_verts = mesh.corner_verts();
    Vector<int> verts(corner_verts.begin(), corner_verts.end());
    Vector<int3> tris(mesh_corner_tris.begin(), mesh_corner_tris.end());
    return mesh_to_sdf_grid(positions, verts, tris, voxel_size, band_width_in_voxels);
  }();
  if (!sdf) {
    r_error = "The signed distance field of the input mesh could not be built";
    return false;
  }
  stamp("sdf");

  Array<int8_t> states(nodes_num);
  {
    bke::VolumeTreeAccessToken token;
    const openvdb::FloatGrid &grid = sdf.grid(token);
    classify_nodes(grid, nodes, origin, voxel_size, band, shell_distance, states);
  }
  stamp("sample");
  flood_fill_outside(nodes, states);
  stamp("fill");

  int interior_nodes = 0;
  for (const int i : states.index_range()) {
    if (states[i] == NODE_UNKNOWN) {
      states[i] = NODE_INSIDE;
      interior_nodes++;
    }
  }
  r_result.interior_nodes = interior_nodes;

  if (debug) {
    int counts[4] = {0, 0, 0, 0};
    for (const int8_t state : states) {
      counts[state]++;
    }
    printf("TET_DEBUG lattice (%d %d %d) voxel %g: inside %d shell %d outside %d filled %d\n",
           nodes.x,
           nodes.y,
           nodes.z,
           voxel_size,
           counts[NODE_INSIDE],
           counts[NODE_SHELL],
           counts[NODE_OUTSIDE],
           interior_nodes);
  }

  /* The elements: every cell whose eight nodes are solid (or within the shell) and that has at
   * least one node strictly inside. Counted per slice first, so the fill below can run in parallel
   * without atomics. */
  auto cell_is_inside = [&](const int i, const int j, const int k) {
    bool all_usable = true;
    bool any_inside = false;
    for (const auto &offset : corner_offsets) {
      const int8_t state = states[lattice_index(nodes, i + offset[0], j + offset[1], k + offset[2])];
      if (state == NODE_OUTSIDE) {
        all_usable = false;
        break;
      }
      any_inside = any_inside || (state == NODE_INSIDE);
    }
    return all_usable && any_inside;
  };

  const int slices = nodes.z - 1;
  Array<int> cell_starts(slices + 1);
  {
    Array<int> slice_counts(slices, 0);
    threading::parallel_for(IndexRange(slices), 1, [&](const IndexRange k_range) {
      for (const int k : k_range) {
        int count = 0;
        for (int j = 0; j + 1 < nodes.y; j++) {
          for (int i = 0; i + 1 < nodes.x; i++) {
            count += cell_is_inside(i, j, k) ? 1 : 0;
          }
        }
        slice_counts[k] = count;
      }
    });
    int total = 0;
    for (const int k : IndexRange(slices)) {
      cell_starts[k] = total;
      total += slice_counts[k];
    }
    cell_starts[slices] = total;
  }
  stamp("cells");

  const int tets_num = cell_starts[slices];
  if (tets_num == 0) {
    return true;
  }

  Array<int4> tets_node(int64_t(tets_num) * 6);
  threading::parallel_for(IndexRange(slices), 1, [&](const IndexRange k_range) {
    for (const int k : k_range) {
      int cursor = cell_starts[k] * 6;
      for (int j = 0; j + 1 < nodes.y; j++) {
        for (int i = 0; i + 1 < nodes.x; i++) {
          if (!cell_is_inside(i, j, k)) {
            continue;
          }
          int corner[8];
          for (int c = 0; c < 8; c++) {
            corner[c] = lattice_index(
                nodes, i + corner_offsets[c][0], j + corner_offsets[c][1], k + corner_offsets[c][2]);
          }
          for (const auto &tet : tet_corners) {
            tets_node[cursor++] = int4(
                corner[tet[0]], corner[tet[1]], corner[tet[2]], corner[tet[3]]);
          }
        }
      }
    }
  });

  /* Only the nodes that ended up in an element become vertices, so the mesh carries no loose
   * vertices. Writing the same value from several threads is benign. */
  Array<int8_t> used_nodes(nodes_num, 0);
  threading::parallel_for(IndexRange(tets_num * 6), 2048, [&](const IndexRange range) {
    for (const int t : range) {
      const int4 &tet = tets_node[t];
      used_nodes[tet.x] = 1;
      used_nodes[tet.y] = 1;
      used_nodes[tet.z] = 1;
      used_nodes[tet.w] = 1;
    }
  });

  /* Place the vertices: compact the used nodes, and pull the nodes of the outer shell onto the
   * surface so that the cells there can be built at all. */
  Array<int> node_to_vert(nodes_num, -1);
  {
    bke::BVHTreeFromMesh bvh = mesh.bvh_corner_tris();
    Array<int> slice_starts(nodes.z + 1);
    int verts_num = 0;
    {
      Array<int> slice_counts(nodes.z, 0);
      threading::parallel_for(IndexRange(nodes.z), 1, [&](const IndexRange k_range) {
        for (const int k : k_range) {
          int count = 0;
          const int row = k * nodes.y * nodes.x;
          const int row_size = nodes.y * nodes.x;
          for (int n = row; n < row + row_size; n++) {
            count += used_nodes[n] ? 1 : 0;
          }
          slice_counts[k] = count;
        }
      });
      int total = 0;
      for (const int k : IndexRange(nodes.z)) {
        slice_starts[k] = total;
        total += slice_counts[k];
      }
      slice_starts[nodes.z] = total;
      verts_num = total;
    }

    r_result.vert_positions.reinitialize(verts_num);
    const MutableSpan<float3> positions = r_result.vert_positions;
    const float shell_sq = shell_distance * shell_distance;
    threading::parallel_for(IndexRange(nodes.z), 1, [&](const IndexRange k_range) {
      for (const int k : k_range) {
        int cursor = slice_starts[k];
        for (int j = 0; j < nodes.y; j++) {
          for (int i = 0; i < nodes.x; i++) {
            const int node = lattice_index(nodes, i, j, k);
            if (!used_nodes[node]) {
              continue;
            }
            node_to_vert[node] = cursor;
            float3 position = origin + float3(float(i), float(j), float(k)) * voxel_size;
            if (shell_sq > 0.0f && states[node] == NODE_SHELL && bvh.tree != nullptr) {
              BVHTreeNearest nearest;
              nearest.index = -1;
              nearest.dist_sq = FLT_MAX;
              const float co[3] = {position.x, position.y, position.z};
              if (BLI_bvhtree_find_nearest(
                      bvh.tree, co, &nearest, bvh.nearest_callback, &bvh) >= 0 &&
                  nearest.dist_sq <= shell_sq)
              {
                position = float3(nearest.co[0], nearest.co[1], nearest.co[2]);
              }
            }
            positions[cursor++] = position;
          }
        }
      }
    });
  }
  used_nodes.reinitialize(0);
  stamp("verts");

  /* Remap the elements to the vertices and keep the ones that have a volume. A cell whose nodes all
   * landed on the same spot of a flat surface collapses here, which is expected: the elements of
   * that cell would be empty. */
  const float min_volume = 1e-6f * voxel_size * voxel_size * voxel_size;
  {
    Array<int4> tets(tets_num * 6);
    Array<float> volumes(tets_num * 6);
    threading::parallel_for(IndexRange(tets_num * 6), 1024, [&](const IndexRange range) {
      for (const int t : range) {
        const int4 &tet = tets_node[t];
        tets[t] = int4(node_to_vert[tet.x], node_to_vert[tet.y], node_to_vert[tet.z],
                       node_to_vert[tet.w]);
      }
    });
    compute_tet_volumes(tets, r_result.vert_positions, volumes);
    r_result.tets.reserve(tets_num);
    r_result.tet_volumes.reserve(tets_num);
    for (const int t : IndexRange(int(volumes.size()))) {
      if (!(volumes[t] > min_volume)) {
        r_result.dropped_tets++;
        continue;
      }
      r_result.tets.append(tets[t]);
      r_result.tet_volumes.append(volumes[t]);
    }
  }
  tets_node.reinitialize(0);
  node_to_vert.reinitialize(0);
  stamp("elements");

  /* The boundary, then its vertices on the surface: the elements that touch the surface can reach it
   * through a node that is half a voxel deep (their outermost cells are the ones that collapsed), and
   * moving only the vertices of the boundary is what makes the surface follow the input exactly. */
  {
    Array<int8_t> boundary_verts(r_result.vert_positions.size(), 0);
    build_boundary(r_result.tets, r_result.boundary_tris, boundary_verts);
    stamp("boundary");

    if (boundary_distance > 0.0f) {
      const float limit_sq = boundary_distance * boundary_distance;
      const int verts_num = int(r_result.vert_positions.size());
      bke::BVHTreeFromMesh bvh = mesh.bvh_corner_tris();

      /* Where every boundary vertex would like to go: the closest point of the input surface. */
      Array<float3> base(verts_num);
      base.as_mutable_span().copy_from(r_result.vert_positions);
      Array<float3> target(verts_num);
      target.as_mutable_span().copy_from(r_result.vert_positions);
      if (bvh.tree != nullptr) {
        threading::parallel_for(IndexRange(verts_num), 1024, [&](const IndexRange range) {
          for (const int i : range) {
            if (!boundary_verts[i]) {
              continue;
            }
            const float3 position = base[i];
            BVHTreeNearest nearest;
            nearest.index = -1;
            nearest.dist_sq = FLT_MAX;
            const float co[3] = {position.x, position.y, position.z};
            if (BLI_bvhtree_find_nearest(
                    bvh.tree, co, &nearest, bvh.nearest_callback, &bvh) >= 0 &&
                nearest.dist_sq <= limit_sq)
            {
              target[i] = float3(nearest.co[0], nearest.co[1], nearest.co[2]);
            }
          }
        });
      }
      stamp("project");

      /* Moving a vertex can turn one of its elements inside out, and dropping that element would
       * leave the ones behind it touching along an edge only: a pinched, non manifold surface. So
       * the elements that end up inverted back off the vertices they are made of instead, which
       * converges in a couple of passes (a vertex that is pulled back to half its move can only
       * invert elements that were almost flat already). */
      Array<float> blend(verts_num, 1.0f);
      Array<float> volumes(r_result.tets.size());
      for (int iteration = 0; iteration < 6; iteration++) {
        MutableSpan<float3> positions = r_result.vert_positions;
        threading::parallel_for(IndexRange(verts_num), 1024, [&](const IndexRange range) {
          for (const int i : range) {
            positions[i] = math::interpolate(base[i], target[i], blend[i]);
          }
        });
        compute_tet_volumes(r_result.tets, r_result.vert_positions, volumes);

        int inverted = 0;
        for (const int t : IndexRange(int(volumes.size()))) {
          if (volumes[t] > min_volume) {
            continue;
          }
          inverted++;
          const int4 &tet = r_result.tets[t];
          blend[tet.x] *= 0.5f;
          blend[tet.y] *= 0.5f;
          blend[tet.z] *= 0.5f;
          blend[tet.w] *= 0.5f;
        }
        if (inverted == 0) {
          break;
        }
        if (iteration == 5) {
          /* Last resort: a vertex that is shared by many inverted elements can keep halving its
           * move forever. Flipping the winding keeps the volume and the closed surface. */
          for (const int t : IndexRange(int(volumes.size()))) {
            if (volumes[t] < 0.0f) {
              std::swap(r_result.tets[t].z, r_result.tets[t].w);
              volumes[t] = -volumes[t];
              r_result.flipped_tets++;
            }
          }
        }
      }

      r_result.tet_volumes.reinitialize(volumes.size());
      r_result.tet_volumes.as_mutable_span().copy_from(volumes);
      build_boundary(r_result.tets, r_result.boundary_tris, {});
      stamp("reboundary");
    }
  }

  r_result.volume = threading::parallel_reduce(
      IndexRange(r_result.tets.size()),
      1024,
      0.0f,
      [&](const IndexRange range, const float init) {
        float sum = init;
        for (const int i : range) {
          sum += r_result.tet_volumes[i];
        }
        return sum;
      },
      std::plus<float>());
  stamp("volume");

  if (debug) {
    printf("TET_DEBUG %d elements, %d dropped, %d flipped, %d boundary triangles, volume %.5f\n",
           int(r_result.tets.size()),
           r_result.dropped_tets,
           r_result.flipped_tets,
           int(r_result.boundary_tris.size()),
           r_result.volume);
  }

  r_error.clear();
  return true;
}

#else

bool tetrahedralize_mesh(const Mesh & /*mesh*/,
                         const TetrahedralizeSettings & /*settings*/,
                         TetrahedralizedMesh & /*r_result*/,
                         std::string &r_error)
{
  r_error = "Tetrahedralize requires a build with OpenVDB";
  return false;
}

#endif /* WITH_OPENVDB */

}  // namespace blender::geometry
