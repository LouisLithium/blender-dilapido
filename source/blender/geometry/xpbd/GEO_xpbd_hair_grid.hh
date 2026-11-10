/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "BLI_array.hh"
#include "BLI_math_vector.hh"
#include "BLI_span.hh"

namespace blender::xpbd {

/**
 * Dense AABB grid for hair segments.
 *
 * The hair self collision broad phase used to hash the segment *midpoints* into a grid whose cell
 * size covered a whole segment. For large hairstyles that means every segment looks at 27 cells and
 * the cells contain hundreds of segments, most of which are far away. This grid inserts every
 * segment into the cells its AABB overlaps instead, with a cell size close to the contact distance,
 * and a query walks only the cells of the segment's AABB expanded by the contact distance.
 *
 * The grid is rebuilt every substep. Building it is two linear passes (count, prefix sum, fill), and
 * the memory is `cells + insertions` ints. If the hair spans a huge volume the cell size is increased
 * until the grid fits into `MAX_CELLS`, which keeps the memory bounded and only makes the broad
 * phase more conservative.
 *
 * The query may return the same segment more than once (a segment can be in several cells of the
 * query range); the caller sorts and deduplicates the final pair list.
 */
class HairSegmentGrid {
 private:
  float3 origin_ = float3(0.0f);
  int3 dims_ = int3(0);
  float cell_size_ = 0.0f;
  /** First segment of each cell, plus the total count at the end. */
  Array<int> cell_offsets_;
  Array<int> cell_segments_;

  static constexpr int MAX_CELLS = 8 * 1024 * 1024;

  int3 cell_of(const float3 &point) const
  {
    return int3(int(std::floor((point.x - origin_.x) / cell_size_)),
                int(std::floor((point.y - origin_.y) / cell_size_)),
                int(std::floor((point.z - origin_.z) / cell_size_)));
  }

  int cell_index(const int3 &cell) const
  {
    return (cell.z * dims_.y + cell.y) * dims_.x + cell.x;
  }

  void segment_cell_range(const float3 &p0,
                          const float3 &p1,
                          const float expand,
                          int3 &r_min,
                          int3 &r_max) const
  {
    const float3 min_point(math::min(p0.x, p1.x) - expand,
                           math::min(p0.y, p1.y) - expand,
                           math::min(p0.z, p1.z) - expand);
    const float3 max_point(math::max(p0.x, p1.x) + expand,
                           math::max(p0.y, p1.y) + expand,
                           math::max(p0.z, p1.z) + expand);
    const int3 raw_min = cell_of(min_point);
    const int3 raw_max = cell_of(max_point);
    r_min = int3(std::clamp(raw_min.x, 0, dims_.x - 1),
                 std::clamp(raw_min.y, 0, dims_.y - 1),
                 std::clamp(raw_min.z, 0, dims_.z - 1));
    r_max = int3(std::clamp(raw_max.x, 0, dims_.x - 1),
                 std::clamp(raw_max.y, 0, dims_.y - 1),
                 std::clamp(raw_max.z, 0, dims_.z - 1));
  }

 public:
  bool is_empty() const
  {
    return cell_offsets_.is_empty();
  }

  /**
   * Builds the grid for the segments `point_a[i] - point_b[i]`.
   *
   * \param search_distance: cell size, usually a bit larger than the maximum contact distance.
   */
  void build(const Span<float3> positions,
             const Span<int> point_a,
             const Span<int> point_b,
             const float search_distance)
  {
    cell_offsets_.reinitialize(0);
    cell_segments_.reinitialize(0);
    const int segments_num = point_a.size();
    if (segments_num == 0 || point_b.size() != segments_num || !(search_distance > 0.0f)) {
      return;
    }

    float3 bounds_min = positions[point_a[0]];
    float3 bounds_max = bounds_min;
    float max_segment_length = 0.0f;
    for (const int segment_i : IndexRange(segments_num)) {
      const float3 &p0 = positions[point_a[segment_i]];
      const float3 &p1 = positions[point_b[segment_i]];
      if (!(std::isfinite(p0.x) && std::isfinite(p0.y) && std::isfinite(p0.z) &&
            std::isfinite(p1.x) && std::isfinite(p1.y) && std::isfinite(p1.z)))
      {
        return;
      }
      bounds_min = math::min(bounds_min, math::min(p0, p1));
      bounds_max = math::max(bounds_max, math::max(p0, p1));
      max_segment_length = std::max(max_segment_length, math::distance(p0, p1));
    }
    const float3 extent = bounds_max - bounds_min;
    /* A very long segment (a broken/exploded strand) would otherwise be inserted into a huge number
     * of cells. Eight cells along its length is enough for the AABB broad phase and keeps the
     * insertion count bounded even for pathological input. */
    cell_size_ = std::max(search_distance, max_segment_length / 8.0f);
    if (!(cell_size_ > 0.0f) || !std::isfinite(cell_size_)) {
      return;
    }
    /* One cell of padding on every side, so the min cell index is always valid. */
    origin_ = bounds_min - float3(cell_size_);
    const auto make_dims = [&]() -> int3 {
      const double x = std::ceil(double(extent.x) / double(cell_size_)) + 3.0;
      const double y = std::ceil(double(extent.y) / double(cell_size_)) + 3.0;
      const double z = std::ceil(double(extent.z) / double(cell_size_)) + 3.0;
      return int3(int(std::min(x, 1.0e6)), int(std::min(y, 1.0e6)), int(std::min(z, 1.0e6)));
    };
    int3 dims = make_dims();
    while (int64_t(dims.x) * int64_t(dims.y) * int64_t(dims.z) > MAX_CELLS) {
      cell_size_ *= 1.5f;
      if (!std::isfinite(cell_size_)) {
        cell_offsets_.reinitialize(0);
        cell_segments_.reinitialize(0);
        return;
      }
      origin_ = bounds_min - float3(cell_size_);
      dims = make_dims();
    }
    dims_ = dims;
    const int cells_num = dims_.x * dims_.y * dims_.z;

    /* Count the insertions per cell (offset i + 1), then prefix sum. */
    cell_offsets_.reinitialize(cells_num + 1);
    cell_offsets_.fill(0);
    for (const int segment_i : IndexRange(segments_num)) {
      int3 min_cell;
      int3 max_cell;
      this->segment_cell_range(positions[point_a[segment_i]],
                               positions[point_b[segment_i]],
                               0.0f,
                               min_cell,
                               max_cell);
      for (int z = min_cell.z; z <= max_cell.z; ++z) {
        for (int y = min_cell.y; y <= max_cell.y; ++y) {
          for (int x = min_cell.x; x <= max_cell.x; ++x) {
            cell_offsets_[cell_index(int3(x, y, z)) + 1]++;
          }
        }
      }
    }
    for (const int cell_i : IndexRange(cells_num)) {
      cell_offsets_[cell_i + 1] += cell_offsets_[cell_i];
    }

    cell_segments_.reinitialize(cell_offsets_[cells_num]);
    Array<int> cursors(cells_num);
    for (const int cell_i : IndexRange(cells_num)) {
      cursors[cell_i] = cell_offsets_[cell_i];
    }
    for (const int segment_i : IndexRange(segments_num)) {
      int3 min_cell;
      int3 max_cell;
      this->segment_cell_range(positions[point_a[segment_i]],
                               positions[point_b[segment_i]],
                               0.0f,
                               min_cell,
                               max_cell);
      for (int z = min_cell.z; z <= max_cell.z; ++z) {
        for (int y = min_cell.y; y <= max_cell.y; ++y) {
          for (int x = min_cell.x; x <= max_cell.x; ++x) {
            cell_segments_[cursors[cell_index(int3(x, y, z))]++] = segment_i;
          }
        }
      }
    }
  }

  /**
   * Calls \a fn for every segment in the cells of segment \a segment_i's AABB expanded by
   * \a query_expand. The same segment may be passed more than once.
   */
  template<typename Fn>
  void foreach_candidate(const int segment_i,
                         const Span<float3> positions,
                         const Span<int> point_a,
                         const Span<int> point_b,
                         const float query_expand,
                         Fn &&fn) const
  {
    if (cell_offsets_.is_empty()) {
      return;
    }
    int3 min_cell;
    int3 max_cell;
    this->segment_cell_range(positions[point_a[segment_i]],
                             positions[point_b[segment_i]],
                             query_expand,
                             min_cell,
                             max_cell);
    for (int z = min_cell.z; z <= max_cell.z; ++z) {
      for (int y = min_cell.y; y <= max_cell.y; ++y) {
        for (int x = min_cell.x; x <= max_cell.x; ++x) {
          const int cell_i = cell_index(int3(x, y, z));
          for (const int candidate_i : IndexRange(cell_offsets_[cell_i],
                                                  cell_offsets_[cell_i + 1] -
                                                      cell_offsets_[cell_i]))
          {
            fn(cell_segments_[candidate_i]);
          }
        }
      }
    }
  }
};

}  // namespace blender::xpbd
