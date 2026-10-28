/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * Rasterization of emitters into the grids of `Init Ignis Data`.
 */

#include <algorithm>
#include <cmath>

#include "BLI_array.hh"
#include "BLI_bounds.hh"
#include "BLI_math_base.hh"
#include "BLI_math_vector.hh"

#include "BKE_lib_id.hh"
#include "BKE_volume.hh"

#include "GEO_ignis_source.hh"
#include "GEO_ignis_solver.hh"

#ifdef WITH_OPENVDB
#  include "BKE_volume_grid.hh"
#  include "BKE_volume_openvdb.hh"

#  include "GEO_mesh_to_volume.hh"

#  include <openvdb/openvdb.h>
#  include <openvdb/tools/Morphology.h>
#endif

namespace blender::geometry::ignis {

#ifdef WITH_OPENVDB

namespace {

using openvdb::Coord;

openvdb::FloatGrid::Ptr create_float_grid(const openvdb::math::Transform::Ptr &transform)
{
  openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0f);
  grid->setTransform(transform);
  grid->setGridClass(openvdb::GRID_FOG_VOLUME);
  return grid;
}

/**
 * Calls `fn(coord, weight)` for every voxel the kernel of a sample covers. The weight is one at the
 * center and falls to zero at the radius, smoothly, so a source does not show the voxel grid.
 */
template<typename Function>
void foreach_kernel_voxel(const openvdb::math::Transform &transform,
                          const float3 position,
                          const float radius,
                          const Function &fn)
{
  if (radius <= 0.0f) {
    return;
  }
  const double voxel = transform.voxelSize()[0];
  const openvdb::Vec3d center = transform.worldToIndex(
      openvdb::Vec3d(double(position.x), double(position.y), double(position.z)));
  const int reach = int(std::ceil(double(radius) / voxel)) + 1;
  const int base_i = int(std::floor(center.x()));
  const int base_j = int(std::floor(center.y()));
  const int base_k = int(std::floor(center.z()));
  const double radius_sq = double(radius) * double(radius);
  for (int k = base_k - reach; k <= base_k + reach; k++) {
    const double dz = (double(k) - center.z()) * voxel;
    for (int j = base_j - reach; j <= base_j + reach; j++) {
      const double dy = (double(j) - center.y()) * voxel;
      const double dyz_sq = dy * dy + dz * dz;
      if (dyz_sq >= radius_sq) {
        continue;
      }
      for (int i = base_i - reach; i <= base_i + reach; i++) {
        const double dx = (double(i) - center.x()) * voxel;
        const double distance_sq = dyz_sq + dx * dx;
        if (distance_sq >= radius_sq) {
          continue;
        }
        const double falloff = 1.0 - distance_sq / radius_sq;
        fn(Coord(i, j, k), float(falloff * falloff));
      }
    }
  }
}

struct Accessors {
  openvdb::FloatGrid::Accessor density;
  openvdb::FloatGrid::Accessor temperature;
  openvdb::FloatGrid::Accessor fuel;
  openvdb::Vec3fGrid::Accessor velocity;
};

void splat_value(openvdb::FloatGrid::Accessor &accessor,
                 const Coord &coord,
                 const float value,
                 const float weight)
{
  const float incoming = value * weight;
  if (incoming <= 0.0f) {
    return;
  }
  accessor.setValue(coord, std::max(accessor.getValue(coord), incoming));
}

/**
 * Like #splat_value, but keeps the strongest value of either sign.
 *
 * The temperature is an anomaly relative to the ambient, so a source may be hotter *or* colder than
 * the air around it; `splat_value` would drop everything below zero and a cooling emitter would do
 * nothing.
 */
void splat_signed_value(openvdb::FloatGrid::Accessor &accessor,
                        const Coord &coord,
                        const float value,
                        const float weight)
{
  const float incoming = value * weight;
  if (incoming == 0.0f) {
    return;
  }
  const float current = accessor.getValue(coord);
  accessor.setValue(coord,
                    incoming > 0.0f ? std::max(current, incoming) : std::min(current, incoming));
}

void splat_sample(Accessors &accessors,
                  const openvdb::math::Transform &transform,
                  const SourceSample &sample,
                  const float radius,
                  const float scale)
{
  const bool has_velocity = math::length_squared(sample.velocity) > 0.0f;
  foreach_kernel_voxel(transform, sample.position, radius, [&](const Coord &coord,
                                                              const float weight) {
    splat_value(accessors.density, coord, sample.density * scale, weight);
    splat_signed_value(accessors.temperature, coord, sample.temperature * scale, weight);
    splat_value(accessors.fuel, coord, sample.fuel * scale, weight);
    if (has_velocity) {
      const openvdb::Vec3f current = accessors.velocity.getValue(coord);
      const openvdb::Vec3f incoming(sample.velocity.x, sample.velocity.y, sample.velocity.z);
      accessors.velocity.setValue(coord, current + (incoming - current) * weight);
    }
  });
}

/** Samples the triangles at roughly half a voxel so a surface emits continuously. */
void rasterize_surface(const Span<const SourceSample> samples,
                       const Span<const int3> corner_tris,
                       const Span<const int> corner_verts,
                       Accessors &accessors,
                       const openvdb::math::Transform &transform,
                       const SourceSettings &settings)
{
  const float spacing = std::max(settings.voxel_size * 0.5f, 1e-5f);
  for (const int3 &tri : corner_tris) {
    const SourceSample &a = samples[corner_verts[tri.x]];
    const SourceSample &b = samples[corner_verts[tri.y]];
    const SourceSample &c = samples[corner_verts[tri.z]];
    const float longest = std::max({math::distance(a.position, b.position),
                                    math::distance(b.position, c.position),
                                    math::distance(c.position, a.position)});
    const int steps = std::max(1, int(std::ceil(longest / spacing)));
    const float step = 1.0f / float(steps);
    for (int i = 0; i <= steps; i++) {
      for (int j = 0; j <= steps - i; j++) {
        const float u = float(i) * step;
        const float v = float(j) * step;
        const float w = 1.0f - u - v;
        SourceSample sample;
        sample.position = a.position * u + b.position * v + c.position * w;
        sample.density = a.density * u + b.density * v + c.density * w;
        sample.temperature = a.temperature * u + b.temperature * v + c.temperature * w;
        sample.fuel = a.fuel * u + b.fuel * v + c.fuel * w;
        sample.velocity = a.velocity * u + b.velocity * v + c.velocity * w;
        splat_sample(accessors, transform, sample, settings.radius, settings.scale);
      }
    }
  }
}

/** Fills everything inside the closed mesh with the average of the surface values. */
void fill_solid(const Span<const SourceSample> samples,
                const Span<const int3> corner_tris,
                const Span<const int> corner_verts,
                Accessors &accessors,
                const openvdb::math::Transform &transform,
                const SourceValues &mean_values,
                const SourceSettings &settings)
{
  Vector<float3> positions(samples.size());
  for (const int i : samples.index_range()) {
    positions[i] = samples[i].position;
  }
  /* `mesh_to_sdf_grid` takes mutable spans, so the shared arrays are copied here once. */
  Vector<int> verts(corner_verts.size());
  for (const int i : verts.index_range()) {
    verts[i] = corner_verts[i];
  }
  Vector<int3> tris(corner_tris.size());
  for (const int i : tris.index_range()) {
    tris[i] = corner_tris[i];
  }
  const bke::VolumeGrid<float> sdf = geometry::mesh_to_sdf_grid(
      positions,
      verts,
      tris,
      settings.voxel_size,
      std::max(settings.band_width, 2.0f * settings.voxel_size));
  if (!sdf) {
    return;
  }
  bke::VolumeTreeAccessToken token;
  const openvdb::GridBase &base = sdf.grid(token);
  if (base.type() != openvdb::FloatGrid::gridType()) {
    return;
  }
  const openvdb::FloatGrid &grid = static_cast<const openvdb::FloatGrid &>(base);
  openvdb::FloatGrid::ConstAccessor reader = grid.getConstAccessor();
  const openvdb::Vec3f velocity(mean_values.velocity.x,
                                mean_values.velocity.y,
                                mean_values.velocity.z);
  for (auto iter = grid.cbeginValueOn(); iter; ++iter) {
    if (iter.getValue() > 0.0f) {
      /* The SDF band reaches outwards, only the inside is a solid. */
      continue;
    }
    const Coord coord = iter.getCoord();
    const float density = mean_values.density * settings.scale;
    splat_value(accessors.density, coord, density, 1.0f);
    splat_signed_value(accessors.temperature, coord, mean_values.temperature * settings.scale, 1.0f);
    splat_value(accessors.fuel, coord, mean_values.fuel * settings.scale, 1.0f);
    accessors.velocity.setValue(coord, velocity);
  }
}

}  // namespace

Volume *rasterize_source(const Span<const SourceSample> samples,
                         const Span<const int3> corner_tris,
                         const Span<const int> corner_verts,
                         const SourceValues &mean_values,
                         const SourceSettings &settings)
{
  if (samples.is_empty()) {
    return nullptr;
  }
  const float voxel_size = std::max(1e-4f, settings.voxel_size);
  const float radius = std::max(settings.radius, 1e-5f);

  const openvdb::math::Transform::Ptr transform =
      openvdb::math::Transform::createLinearTransform(double(voxel_size));
  openvdb::FloatGrid::Ptr density = create_float_grid(transform);
  openvdb::FloatGrid::Ptr temperature = create_float_grid(transform);
  openvdb::FloatGrid::Ptr fuel = create_float_grid(transform);
  openvdb::Vec3fGrid::Ptr velocity = openvdb::Vec3fGrid::create(openvdb::Vec3f(0.0f));
  velocity->setTransform(transform);

  Accessors accessors{density->getAccessor(),
                      temperature->getAccessor(),
                      fuel->getAccessor(),
                      velocity->getAccessor()};

  const bool has_triangles = !corner_tris.is_empty() && !corner_verts.is_empty();
  switch (settings.mode) {
    case SourceMode::Points: {
      for (const SourceSample &sample : samples) {
        splat_sample(accessors, *transform, sample, radius, settings.scale);
      }
      break;
    }
    case SourceMode::Surface: {
      if (has_triangles) {
        rasterize_surface(samples, corner_tris, corner_verts, accessors, *transform, settings);
      }
      else {
        for (const SourceSample &sample : samples) {
          splat_sample(accessors, *transform, sample, radius, settings.scale);
        }
      }
      break;
    }
    case SourceMode::Solid: {
      if (has_triangles) {
        rasterize_surface(samples, corner_tris, corner_verts, accessors, *transform, settings);
        fill_solid(samples,
                   corner_tris,
                   corner_verts,
                   accessors,
                   *transform,
                   mean_values,
                   settings);
      }
      else {
        for (const SourceSample &sample : samples) {
          splat_sample(accessors, *transform, sample, radius, settings.scale);
        }
      }
      break;
    }
  }

  Volume *volume = BKE_id_new_nomain<Volume>(nullptr);
  density->tree().prune(1e-6f);
  temperature->tree().prune(1e-6f);
  fuel->tree().prune(1e-6f);
  velocity->tree().prune(openvdb::Vec3f(1e-6f));
  BKE_volume_grid_add_vdb(*volume, grids::density, density);
  BKE_volume_grid_add_vdb(*volume, grids::temperature, temperature);
  BKE_volume_grid_add_vdb(*volume, grids::fuel, fuel);
  BKE_volume_grid_add_vdb(*volume, grids::velocity, velocity);
  return volume;
}

#else /* WITH_OPENVDB */

Volume *rasterize_source(const Span<const SourceSample> /*samples*/,
                         const Span<const int3> /*corner_tris*/,
                         const Span<const int> /*corner_verts*/,
                         const SourceValues & /*mean_values*/,
                         const SourceSettings & /*settings*/)
{
  return nullptr;
}

#endif /* WITH_OPENVDB */

}  // namespace blender::geometry::ignis
