/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo
 *
 * The sparse smoke and fire solver behind the `Ignis Solver` node. See `GEO_ignis_solver.hh` for
 * the overview; this file is the numerical core.
 *
 * Everything is stored in OpenVDB trees, so iterating a field only visits the voxels that exist.
 * The state is rebuilt from the active region every frame: there is no dense domain, and the
 * bounding box is only used to clip the sources and the result.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <limits>

#include "BLI_array.hh"
#include "BLI_math_base.hh"
#include "BLI_math_vector.hh"
#include "BLI_task.hh"

#include "BKE_lib_id.hh"
#include "BKE_mesh.hh"
#include "BKE_volume.hh"

#include "GEO_ignis_solver.hh"

#ifdef WITH_OPENVDB
#  include "BKE_volume_grid.hh"
#  include "BKE_volume_openvdb.hh"
#  include "BKE_volume_to_mesh.hh"

#  include <openvdb/openvdb.h>
#  include <openvdb/tools/Interpolation.h>
#  include <openvdb/tools/Morphology.h>
#  include <openvdb/tools/VolumeAdvect.h>
#endif

namespace blender::geometry::ignis {

#ifdef WITH_OPENVDB

namespace {

using openvdb::Coord;

/** The whole state of the simulation. Nothing else is kept between frames. */
struct State {
  openvdb::FloatGrid::Ptr density;
  openvdb::FloatGrid::Ptr temperature;
  openvdb::FloatGrid::Ptr fuel;
  openvdb::FloatGrid::Ptr flame;
  openvdb::FloatGrid::Ptr pressure;
  openvdb::Vec3fGrid::Ptr velocity;

  bool valid() const
  {
    return density != nullptr;
  }

  float voxel_size() const
  {
    return float(this->density->transform().voxelSize()[0]);
  }

  float3 world_position(const Coord &coord) const
  {
    const openvdb::Vec3d position = density->transform().indexToWorld(coord);
    return float3(float(position.x()), float(position.y()), float(position.z()));
  }
};

openvdb::Vec3f to_vdb(const float3 value)
{
  return openvdb::Vec3f(value.x, value.y, value.z);
}

float3 from_vdb(const openvdb::Vec3f value)
{
  return float3(value.x(), value.y(), value.z());
}

openvdb::FloatGrid::Ptr create_float_grid(const openvdb::math::Transform::Ptr &transform)
{
  openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.0f);
  grid->setTransform(transform);
  return grid;
}

openvdb::tools::Scheme::SemiLagrangian to_scheme(const AdvectionScheme scheme)
{
  switch (scheme) {
    case AdvectionScheme::SemiLagrangian:
      return openvdb::tools::Scheme::SEMI;
    case AdvectionScheme::Midpoint:
      return openvdb::tools::Scheme::MID;
    case AdvectionScheme::MacCormack:
      return openvdb::tools::Scheme::MAC;
    case AdvectionScheme::Bfecc:
      return openvdb::tools::Scheme::BFECC;
  }
  return openvdb::tools::Scheme::MAC;
}

openvdb::tools::Scheme::Limiter to_limiter(const AdvectionLimiter limiter)
{
  switch (limiter) {
    case AdvectionLimiter::None:
      return openvdb::tools::Scheme::NO_LIMITER;
    case AdvectionLimiter::Clamp:
      return openvdb::tools::Scheme::CLAMP;
    case AdvectionLimiter::Revert:
      return openvdb::tools::Scheme::REVERT;
  }
  return openvdb::tools::Scheme::CLAMP;
}

/* -------------------------------------------------------------------- */
/** \name Reading and writing the state volume
 * \{ */

openvdb::FloatGrid::Ptr read_float_grid(const Volume &volume, const StringRef name)
{
  const bke::VolumeGridData *grid_data = BKE_volume_grid_find(&volume, name);
  if (grid_data == nullptr) {
    return nullptr;
  }
  bke::VolumeTreeAccessToken token;
  const openvdb::GridBase &base = grid_data->grid(token);
  if (base.type() != openvdb::FloatGrid::gridType()) {
    return nullptr;
  }
  return openvdb::gridPtrCast<openvdb::FloatGrid>(base.deepCopyGrid());
}

openvdb::Vec3fGrid::Ptr read_vec3_grid(const Volume &volume, const StringRef name)
{
  const bke::VolumeGridData *grid_data = BKE_volume_grid_find(&volume, name);
  if (grid_data == nullptr) {
    return nullptr;
  }
  bke::VolumeTreeAccessToken token;
  const openvdb::GridBase &base = grid_data->grid(token);
  if (base.type() != openvdb::Vec3fGrid::gridType()) {
    return nullptr;
  }
  return openvdb::gridPtrCast<openvdb::Vec3fGrid>(base.deepCopyGrid());
}

State read_state(const Volume &volume)
{
  State state;
  state.density = read_float_grid(volume, grids::density);
  state.temperature = read_float_grid(volume, grids::temperature);
  state.fuel = read_float_grid(volume, grids::fuel);
  state.flame = read_float_grid(volume, grids::flame);
  state.pressure = read_float_grid(volume, grids::pressure);
  state.velocity = read_vec3_grid(volume, grids::velocity);
  return state;
}

void add_float_grid(Volume &volume, const StringRef name, openvdb::FloatGrid::Ptr grid)
{
  if (!grid) {
    return;
  }
  grid->setGridClass(openvdb::GRID_FOG_VOLUME);
  grid->tree().prune(1e-5f);
  BKE_volume_grid_add_vdb(volume, name, grid);
}

void write_state(Volume &volume, const State &state)
{
  add_float_grid(volume, grids::density, state.density);
  add_float_grid(volume, grids::temperature, state.temperature);
  add_float_grid(volume, grids::fuel, state.fuel);
  add_float_grid(volume, grids::flame, state.flame);
  add_float_grid(volume, grids::pressure, state.pressure);
  if (state.velocity) {
    state.velocity->tree().prune(openvdb::Vec3f(1e-6f));
    BKE_volume_grid_add_vdb(volume, grids::velocity, state.velocity);
  }
}

/** Creates empty grids at the given resolution. */
void init_state(State &state, const float voxel_size)
{
  const openvdb::math::Transform::Ptr transform =
      openvdb::math::Transform::createLinearTransform(double(voxel_size));
  state.density = create_float_grid(transform);
  state.temperature = create_float_grid(transform);
  state.fuel = create_float_grid(transform);
  state.flame = create_float_grid(transform);
  state.pressure.reset();
  state.velocity = openvdb::Vec3fGrid::create(openvdb::Vec3f(0.0f));
  state.velocity->setTransform(transform);
}

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Sparse helpers
 * \{ */

/** Topology of `density > 0` grown by `layers`, as the region the velocity is kept in. */
openvdb::FloatTree smoke_tree(const State &state, const int layers)
{
  openvdb::FloatTree tree(0.0f);
  for (auto iter = state.density->cbeginValueOn(); iter; ++iter) {
    if (iter.getValue() > 1e-4f) {
      tree.setValue(iter.getCoord(), 1.0f);
    }
  }
  if (layers > 0 && tree.activeVoxelCount() > 0) {
    openvdb::tools::dilateActiveValues(
        tree, layers, openvdb::tools::NN_FACE, openvdb::tools::EXPAND_TILES);
  }
  return tree;
}

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Source rasterization
 * \{ */

struct SplatAccessors {
  openvdb::FloatGrid::Accessor density;
  openvdb::FloatGrid::Accessor temperature;
  openvdb::FloatGrid::Accessor fuel;
  openvdb::Vec3fGrid::Accessor velocity;
};

/**
 * Calls `fn(coord, weight)` for every voxel the kernel of a particle covers. The weight is one at
 * the center and falls to zero at the radius, smoothly, so a source does not show the voxel grid.
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
  const double radius_in_voxels = double(radius) / voxel;
  const int reach = int(std::ceil(radius_in_voxels)) + 1;
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

/** A source value is a *target*, not an addition: a steady emitter holds its region at its own
 * value instead of piling up until the simulation explodes. */
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

void splat_source_point(SplatAccessors &accessors,
                        const openvdb::math::Transform &transform,
                        const float3 position,
                        const float radius,
                        const float density,
                        const float temperature,
                        const float fuel,
                        const float3 velocity)
{
  const bool has_velocity = math::length_squared(velocity) > 0.0f;
  foreach_kernel_voxel(transform, position, radius, [&](const Coord &coord, const float weight) {
    splat_value(accessors.density, coord, density, weight);
    splat_value(accessors.temperature, coord, temperature, weight);
    splat_value(accessors.fuel, coord, fuel, weight);
    if (has_velocity) {
      const openvdb::Vec3f current = accessors.velocity.getValue(coord);
      accessors.velocity.setValue(coord, current + (to_vdb(velocity) - current) * weight);
    }
  });
}

/**
 * Samples the triangles of a mesh at roughly half a voxel, so a mesh source emits from its whole
 * surface instead of only from its vertices. The attributes are interpolated with the barycentric
 * weights of the sample.
 */
void rasterize_surface(const SourceData &source,
                       SplatAccessors &accessors,
                       const openvdb::math::Transform &transform,
                       const float voxel)
{
  const float spacing = std::max(voxel * 0.5f, 1e-5f);
  for (const int3 &tri : source.corner_tris) {
    const int index_a = source.corner_verts[tri.x];
    const int index_b = source.corner_verts[tri.y];
    const int index_c = source.corner_verts[tri.z];
    const float3 a = source.positions[index_a];
    const float3 b = source.positions[index_b];
    const float3 c = source.positions[index_c];
    const float longest = std::max(
        {math::distance(a, b), math::distance(b, c), math::distance(c, a)});
    const int steps = std::max(1, int(std::ceil(longest / spacing)));
    const float step = 1.0f / float(steps);
    for (int i = 0; i <= steps; i++) {
      for (int j = 0; j <= steps - i; j++) {
        const float u = float(i) * step;
        const float v = float(j) * step;
        const float w = 1.0f - u - v;
        splat_source_point(accessors,
                           transform,
                           a * u + b * v + c * w,
                           source.radius[index_a] * u + source.radius[index_b] * v +
                               source.radius[index_c] * w,
                           source.density[index_a] * u + source.density[index_b] * v +
                               source.density[index_c] * w,
                           source.temperature[index_a] * u + source.temperature[index_b] * v +
                               source.temperature[index_c] * w,
                           source.fuel[index_a] * u + source.fuel[index_b] * v +
                               source.fuel[index_c] * w,
                           source.velocity[index_a] * u + source.velocity[index_b] * v +
                               source.velocity[index_c] * w);
      }
    }
  }
}

void rasterize_sources(const Span<const SourceData> sources, State &state)
{
  SplatAccessors accessors{state.density->getAccessor(),
                           state.temperature->getAccessor(),
                           state.fuel->getAccessor(),
                           state.velocity->getAccessor()};
  const openvdb::math::Transform &transform = state.density->transform();
  const float voxel = state.voxel_size();
  for (const SourceData &source : sources) {
    const int points_num = source.positions.size();
    if (points_num == 0) {
      continue;
    }
    if (source.sample_surface && !source.corner_tris.is_empty()) {
      rasterize_surface(source, accessors, transform, voxel);
      continue;
    }
    for (const int i : IndexRange(points_num)) {
      const float radius = source.radius[i] > 0.0f ? source.radius[i] : voxel;
      splat_source_point(accessors,
                         transform,
                         source.positions[i],
                         radius,
                         source.density[i],
                         source.temperature[i],
                         source.fuel[i],
                         source.velocity[i]);
    }
  }
}

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Transport
 * \{ */

/**
 * Advects a grid with the same tool the official `Grid Advect` node uses
 * (`openvdb::tools::VolumeAdvection`), which brings its own topology growth and the higher order
 * schemes with limiters. Doing this by hand only ever gives first/second order.
 */
template<typename GridType>
typename GridType::Ptr advect_with_velocity(const GridType &grid,
                                           const openvdb::Vec3fGrid &velocity,
                                           const float delta_time,
                                           const Settings &settings)
{
  openvdb::tools::VolumeAdvection<openvdb::Vec3fGrid, false> advection(velocity);
  advection.setIntegrator(to_scheme(settings.advection_scheme));
  advection.setLimiter(to_limiter(settings.advection_limiter));
  return advection.template advect<GridType, openvdb::tools::Sampler<1>>(grid, delta_time);
}

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Forces
 * \{ */

void apply_forces(State &state, const Settings &settings, const float delta_time)
{
  const float dt = delta_time;
  const float buoyancy = settings.buoyancy * settings.temperature_scale;
  const float drag = std::max(0.0f, settings.drag);
  const float damping = std::max(0.0f, settings.velocity_damping);
  const float damping_factor = 1.0f / (1.0f + damping * dt);

  openvdb::Vec3fGrid::Accessor velocity = state.velocity->getAccessor();
  openvdb::FloatGrid::ConstAccessor temperature = state.temperature->getConstAccessor();
  openvdb::FloatGrid::ConstAccessor density = state.density->getConstAccessor();
  for (auto iter = state.velocity->beginValueOn(); iter; ++iter) {
    const Coord coord = iter.getCoord();
    /* The forces act on the smoke, not on the empty air: the weight of the ambient air is balanced by
     * the hydrostatic pressure that this solver does not model, so accelerating the whole velocity
     * band with gravity would drag the smoke down through the pressure solve. */
    if (density.getValue(coord) <= 1e-4f) {
      continue;
    }
    float3 value = from_vdb(iter.getValue());
    value += settings.gravity * dt;
    value += settings.wind * dt;
    value.z += temperature.getValue(coord) * buoyancy * dt;
    if (drag > 0.0f) {
      value *= 1.0f / (1.0f + drag * dt);
    }
    value *= damping_factor;
    velocity.setValue(coord, to_vdb(value));
  }
}

/** Curl and vorticity confinement: `f = epsilon * h * (N x omega)` with `N = grad |omega|`. */
void apply_vorticity(State &state, const Settings &settings, const float delta_time)
{
  if (settings.vorticity <= 0.0f || state.velocity->activeVoxelCount() == 0) {
    return;
  }
  const double voxel = double(state.voxel_size());
  const openvdb::Vec3fGrid::ConstAccessor velocity = state.velocity->getConstAccessor();
  openvdb::Vec3fGrid::Ptr curl = openvdb::Vec3fGrid::create(openvdb::Vec3f(0.0f));
  curl->setTransform(state.velocity->transformPtr());
  openvdb::Vec3fGrid::Accessor curl_accessor = curl->getAccessor();
  openvdb::FloatGrid::Ptr curl_length = create_float_grid(state.velocity->transformPtr());
  openvdb::FloatGrid::Accessor curl_length_accessor = curl_length->getAccessor();

  for (auto iter = state.velocity->cbeginValueOn(); iter; ++iter) {
    const Coord coord = iter.getCoord();
    const openvdb::Vec3f dx_p = velocity.getValue(coord.offsetBy(1, 0, 0));
    const openvdb::Vec3f dx_m = velocity.getValue(coord.offsetBy(-1, 0, 0));
    const openvdb::Vec3f dy_p = velocity.getValue(coord.offsetBy(0, 1, 0));
    const openvdb::Vec3f dy_m = velocity.getValue(coord.offsetBy(0, -1, 0));
    const openvdb::Vec3f dz_p = velocity.getValue(coord.offsetBy(0, 0, 1));
    const openvdb::Vec3f dz_m = velocity.getValue(coord.offsetBy(0, 0, -1));
    const openvdb::Vec3f omega((dy_p.z() - dy_m.z()) - (dz_p.y() - dz_m.y()),
                               (dz_p.x() - dz_m.x()) - (dx_p.z() - dx_m.z()),
                               (dx_p.y() - dx_m.y()) - (dy_p.x() - dy_m.x()));
    const openvdb::Vec3f scaled = omega * float(0.5 / voxel);
    curl_accessor.setValue(coord, scaled);
    curl_length_accessor.setValue(coord, scaled.length());
  }

  const float strength = settings.vorticity * float(voxel) * delta_time;
  openvdb::Vec3fGrid::Accessor velocity_writer = state.velocity->getAccessor();
  for (auto iter = curl->cbeginValueOn(); iter; ++iter) {
    const Coord coord = iter.getCoord();
    openvdb::Vec3f normal(curl_length_accessor.getValue(coord.offsetBy(1, 0, 0)) -
                              curl_length_accessor.getValue(coord.offsetBy(-1, 0, 0)),
                          curl_length_accessor.getValue(coord.offsetBy(0, 1, 0)) -
                              curl_length_accessor.getValue(coord.offsetBy(0, -1, 0)),
                          curl_length_accessor.getValue(coord.offsetBy(0, 0, 1)) -
                              curl_length_accessor.getValue(coord.offsetBy(0, 0, -1)));
    const float length = normal.length();
    if (length < 1e-8f) {
      continue;
    }
    normal /= length;
    const openvdb::Vec3f force = normal.cross(iter.getValue()) * strength;
    velocity_writer.setValue(coord, velocity_writer.getValue(coord) + force);
  }
}

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Pressure projection
 * \{ */

/**
 * Makes the velocity divergence free on the active region.
 *
 * The divergence uses forward differences and the gradient backward differences, so the composite
 * operator is the plain 7 point Laplacian (widened to 2h) with no odd/even decoupling, i.e. no
 * checkerboard pressure. The region boundary is a free surface (`p = 0`), which is what openvdb's
 * Poisson solver assumes for voxels that are not in the domain.
 */
void project_velocity(State &state, const Settings &settings, Stats &r_stats)
{
  if (state.velocity->activeVoxelCount() == 0 || state.density->activeVoxelCount() == 0) {
    return;
  }
  const double voxel = double(state.voxel_size());
  const openvdb::Vec3fGrid::ConstAccessor velocity = state.velocity->getConstAccessor();

  /* Solve the pressure on a *band* around the smoke, not on the smoke itself: a free surface
   * boundary at the edge of the smoke would force the velocity to zero there and stop a buoyant
   * puff from moving at all. A few layers of the extrapolated air give the return flow room. */
  const openvdb::FloatTree domain = smoke_tree(state, std::max(3, settings.extrapolate + 2));

  static const Coord offsets[6] = {
      Coord(1, 0, 0),
      Coord(-1, 0, 0),
      Coord(0, 1, 0),
      Coord(0, -1, 0),
      Coord(0, 0, 1),
      Coord(0, 0, -1),
  };

  /* A linear index for every voxel of the domain (-1 outside), so the neighbour lookups of the
   * matrix-vector product are one grid access instead of a hash lookup. */
  openvdb::Int32Grid::Ptr index_grid = openvdb::Int32Grid::create(-1);
  openvdb::Int32Grid::Accessor index = index_grid->getAccessor();
  Vector<Coord> coords;
  coords.reserve(int(domain.activeVoxelCount()));
  for (auto iter = domain.cbeginValueOn(); iter; ++iter) {
    index.setValue(iter.getCoord(), int(coords.size()));
    coords.append(iter.getCoord());
  }
  const int domain_size = int(coords.size());
  if (domain_size == 0) {
    return;
  }

  /* The divergence uses forward differences, so a cell can only be made divergence free when its
   * three forward neighbours exist. Using the boundary cells too is what made the system
   * inconsistent (their row picks up the opposite sign, see M66), so they are treated as the open
   * boundary instead: they keep their divergence and the smoke can leave through them. */
  openvdb::Int32Grid::Ptr unknown_grid = openvdb::Int32Grid::create(-1);
  openvdb::Int32Grid::Accessor unknown = unknown_grid->getAccessor();
  Vector<Coord> unknowns;
  unknowns.reserve(domain_size);
  for (const Coord &coord : coords) {
    if (index.getValue(coord.offsetBy(1, 0, 0)) < 0 ||
        index.getValue(coord.offsetBy(0, 1, 0)) < 0 ||
        index.getValue(coord.offsetBy(0, 0, 1)) < 0)
    {
      continue;
    }
    unknown.setValue(coord, int(unknowns.size()));
    unknowns.append(coord);
  }
  const int count = int(unknowns.size());
  if (count == 0) {
    return;
  }

  /* Outside the band the velocity counts as zero: that is the same free surface the pressure has. */
  auto velocity_at = [&](const Coord &coord) {
    return index.getValue(coord) >= 0 ? velocity.getValue(coord) : openvdb::Vec3f(0.0f);
  };
  auto divergence_at = [&](auto &&velocity_fn, const Coord &coord) {
    const openvdb::Vec3f value = velocity_fn(coord);
    return double((velocity_fn(coord.offsetBy(1, 0, 0)).x() - value.x()) +
                  (velocity_fn(coord.offsetBy(0, 1, 0)).y() - value.y()) +
                  (velocity_fn(coord.offsetBy(0, 0, 1)).z() - value.z())) /
           voxel;
  };

  /* Right hand side `L p = divergence * h^2`, with `L` the index space Laplacian. */
  Array<double> rhs(count);
  double divergence_sq = 0.0;
  for (const int i : IndexRange(count)) {
    const double divergence = divergence_at(velocity_at, unknowns[i]);
    rhs[i] = divergence * voxel * voxel;
    divergence_sq += divergence * divergence;
  }
  r_stats.divergence_before = float(std::sqrt(divergence_sq / double(count)));

  /* The 7 point Laplacian with the boundary voxels pinned to zero pressure: symmetric positive
   * definite, so the whole pressure field is determined. */
  auto apply_laplacian = [&](const Span<double> x, MutableSpan<double> r) {
    for (const int i : IndexRange(count)) {
      double sum = -6.0 * x[i];
      const Coord &coord = unknowns[i];
      for (const Coord &offset : offsets) {
        const int j = unknown.getValue(coord + offset);
        if (j >= 0) {
          sum += x[j];
        }
      }
      r[i] = sum;
    }
  };
  auto dot = [](const Span<double> a, const Span<double> b) {
    double sum = 0.0;
    for (const int i : a.index_range()) {
      sum += a[i] * b[i];
    }
    return sum;
  };

  /* Jacobi preconditioned conjugate gradient. The diagonal is the constant -6, so the preconditioner
   * is a single multiply. */
  Array<double> pressure(count);
  Array<double> residual(count);
  Array<double> preconditioned(count);
  Array<double> direction(count);
  Array<double> product(count);
  pressure.fill(0.0);
  residual.as_mutable_span().copy_from(rhs);
  const double rhs_norm = std::sqrt(dot(rhs, rhs));
  int iterations = 0;
  double residual_norm = rhs_norm;
  if (rhs_norm > 1e-12) {
    const int max_iterations = std::max(1, settings.pressure_iterations);
    const double tolerance = std::max(1e-8f, settings.tolerance);
    for (const int i : IndexRange(count)) {
      preconditioned[i] = residual[i] / 6.0;
      direction[i] = preconditioned[i];
    }
    double rz = dot(residual, preconditioned);
    for (; iterations < max_iterations; iterations++) {
      apply_laplacian(direction, product);
      const double pq = dot(direction, product);
      if (std::abs(pq) < 1e-30) {
        break;
      }
      const double alpha = rz / pq;
      for (const int i : IndexRange(count)) {
        pressure[i] += alpha * direction[i];
        residual[i] -= alpha * product[i];
      }
      residual_norm = std::sqrt(dot(residual, residual));
      if (residual_norm <= tolerance * rhs_norm) {
        iterations++;
        break;
      }
      for (const int i : IndexRange(count)) {
        preconditioned[i] = residual[i] / 6.0;
      }
      const double rz_next = dot(residual, preconditioned);
      const double beta = rz_next / rz;
      for (const int i : IndexRange(count)) {
        direction[i] = preconditioned[i] + beta * direction[i];
      }
      rz = rz_next;
    }
    r_stats.residual = float(residual_norm / rhs_norm);
  }
  else {
    r_stats.residual = 0.0f;
  }
  r_stats.pressure_iterations = iterations;

  /* `u -= grad p` with backward differences on every voxel of the band (the pressure of a voxel that
   * is not part of the system is zero, which keeps the field smooth towards the open boundary). */
  openvdb::Vec3fGrid::Accessor velocity_writer = state.velocity->getAccessor();
  auto pressure_at = [&](const Coord &coord) {
    const int j = unknown.getValue(coord);
    return j >= 0 ? pressure[j] : 0.0;
  };
  for (const Coord &coord : coords) {
    const double pressure_here = pressure_at(coord);
    const double px = (pressure_here - pressure_at(coord.offsetBy(-1, 0, 0))) / voxel;
    const double py = (pressure_here - pressure_at(coord.offsetBy(0, -1, 0))) / voxel;
    const double pz = (pressure_here - pressure_at(coord.offsetBy(0, 0, -1))) / voxel;
    velocity_writer.setValue(coord,
                             velocity.getValue(coord) - openvdb::Vec3f(px, py, pz));
  }

  /* Report the residual divergence on the cells that were actually constrained, so the projection can
   * be verified instead of trusted. */
  const openvdb::Vec3fGrid::ConstAccessor corrected = state.velocity->getConstAccessor();
  auto corrected_at = [&](const Coord &coord) {
    return index.getValue(coord) >= 0 ? corrected.getValue(coord) : openvdb::Vec3f(0.0f);
  };
  double after_sq = 0.0;
  for (const int i : IndexRange(count)) {
    const double divergence = divergence_at(corrected_at, unknowns[i]);
    after_sq += divergence * divergence;
  }
  r_stats.divergence_after = float(std::sqrt(after_sq / double(count)));

  openvdb::FloatGrid::Ptr pressure_grid = openvdb::FloatGrid::create(0.0f);
  pressure_grid->setTransform(state.velocity->transformPtr());
  openvdb::FloatGrid::Accessor pressure_writer = pressure_grid->getAccessor();
  for (const int i : IndexRange(count)) {
    pressure_writer.setValue(unknowns[i], float(pressure[i]));
  }
  state.pressure = pressure_grid;
}

/**
 * Keeps the velocity only in a band around the smoke and fills the band by averaging the known
 * neighbours outwards. Without this the extrapolated air velocity would be part of the active set
 * forever and the grid would grow every frame.
 */
void limit_and_extrapolate_velocity(State &state, const Settings &settings)
{
  if (!state.velocity) {
    return;
  }
  if (state.density->activeVoxelCount() == 0) {
    state.velocity->clear();
    return;
  }
  const openvdb::FloatTree keep = smoke_tree(
      state, std::max(settings.extrapolate + 2, settings.domain_padding));
  openvdb::Vec3fGrid::Ptr result = openvdb::Vec3fGrid::create(openvdb::Vec3f(0.0f));
  result->setTransform(state.velocity->transformPtr());
  openvdb::Vec3fGrid::Accessor writer = result->getAccessor();
  {
    const openvdb::Vec3fGrid::ConstAccessor reader = state.velocity->getConstAccessor();
    for (auto iter = keep.cbeginValueOn(); iter; ++iter) {
      const Coord coord = iter.getCoord();
      if (reader.isValueOn(coord)) {
        writer.setValue(coord, reader.getValue(coord));
      }
    }
  }
  for (int pass = 0; pass <= settings.extrapolate; pass++) {
    bool changed = false;
    for (auto iter = keep.cbeginValueOn(); iter; ++iter) {
      const Coord coord = iter.getCoord();
      openvdb::Vec3f unused;
      if (result->tree().probeValue(coord, unused)) {
        continue;
      }
      openvdb::Vec3f sum(0.0f);
      int count = 0;
      for (const Coord &offset : {Coord(1, 0, 0),
                                  Coord(-1, 0, 0),
                                  Coord(0, 1, 0),
                                  Coord(0, -1, 0),
                                  Coord(0, 0, 1),
                                  Coord(0, 0, -1)})
      {
        openvdb::Vec3f neighbour;
        if (result->tree().probeValue(coord + offset, neighbour)) {
          sum += neighbour;
          count++;
        }
      }
      if (count > 0) {
        writer.setValue(coord, sum / float(count));
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
  }
  state.velocity = result;
}

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Combustion and decay
 * \{ */

/** Gives the velocity field a voxel wherever there is smoke, so forces have something to act on. */
void ensure_velocity_topology(State &state)
{
  if (state.density->activeVoxelCount() == 0) {
    return;
  }
  const openvdb::FloatTree smoke = smoke_tree(state, 0);
  openvdb::Vec3fGrid::Accessor writer = state.velocity->getAccessor();
  for (auto iter = smoke.cbeginValueOn(); iter; ++iter) {
    const Coord coord = iter.getCoord();
    if (!writer.isValueOn(coord)) {
      writer.setValue(coord, openvdb::Vec3f(0.0f));
    }
  }
}

/** Largest speed in the field, used for the CFL condition. */
float maximum_speed(const State &state)
{
  float speed = 0.0f;
  for (auto iter = state.velocity->cbeginValueOn(); iter; ++iter) {
    speed = std::max(speed, iter.getValue().length());
  }
  return speed;
}

void combust_and_decay(State &state, const Settings &settings, const float delta_time)
{
  const float dt = delta_time;
  openvdb::FloatGrid::Accessor density = state.density->getAccessor();
  openvdb::FloatGrid::Accessor temperature = state.temperature->getAccessor();
  openvdb::FloatGrid::Accessor fuel = state.fuel->getAccessor();

  const float density_decay = std::exp(-std::max(0.0f, settings.dissipation) * dt);
  const float temperature_decay = std::exp(-std::max(0.0f, settings.cooling) * dt);
  const float burn_rate = std::max(0.0f, settings.burn_rate);

  if (burn_rate > 0.0f) {
    for (auto iter = state.fuel->beginValueOn(); iter; ++iter) {
      const Coord coord = iter.getCoord();
      const float available = iter.getValue();
      if (available <= 0.0f) {
        continue;
      }
      const float burned = std::min(available, burn_rate * dt);
      fuel.setValue(coord, available - burned);
      if (burned > 0.0f) {
        temperature.setValue(coord,
                             temperature.getValue(coord) + burned * settings.heat_release);
        density.setValue(coord, density.getValue(coord) + burned * settings.smoke_per_burn);
      }
    }
  }

  if (density_decay < 1.0f) {
    for (auto iter = state.density->beginValueOn(); iter; ++iter) {
      iter.setValue(iter.getValue() * density_decay);
    }
  }
  if (temperature_decay < 1.0f) {
    for (auto iter = state.temperature->beginValueOn(); iter; ++iter) {
      iter.setValue(iter.getValue() * temperature_decay);
    }
  }
}

/** `flame` is a render only field: the parts that are hotter than the threshold glow. */
void update_flame(State &state, const Settings &settings)
{
  state.flame = create_float_grid(state.density->transformPtr());
  openvdb::FloatGrid::Accessor flame = state.flame->getAccessor();
  const float threshold = settings.flame_threshold;
  for (auto iter = state.temperature->cbeginValueOn(); iter; ++iter) {
    const float value = iter.getValue() - threshold;
    if (value > 0.0f) {
      flame.setValue(iter.getCoord(), value);
    }
  }
}

void clip_to_bounds(State &state, const Settings &settings)
{
  if (!settings.use_bounds) {
    return;
  }
  const float3 min = math::min(settings.bounds_min, settings.bounds_max);
  const float3 max = math::max(settings.bounds_min, settings.bounds_max);
  auto outside = [&](const Coord &coord) {
    const float3 position = state.world_position(coord);
    return position.x < min.x || position.x > max.x || position.y < min.y || position.y > max.y ||
           position.z < min.z || position.z > max.z;
  };
  openvdb::FloatGrid::Accessor density = state.density->getAccessor();
  for (auto iter = state.density->cbeginValueOn(); iter; ++iter) {
    if (outside(iter.getCoord())) {
      density.setValue(iter.getCoord(), 0.0f);
    }
  }
  openvdb::FloatGrid::Accessor temperature = state.temperature->getAccessor();
  for (auto iter = state.temperature->cbeginValueOn(); iter; ++iter) {
    if (outside(iter.getCoord())) {
      temperature.setValue(iter.getCoord(), 0.0f);
    }
  }
  openvdb::FloatGrid::Accessor fuel = state.fuel->getAccessor();
  for (auto iter = state.fuel->cbeginValueOn(); iter; ++iter) {
    if (outside(iter.getCoord())) {
      fuel.setValue(iter.getCoord(), 0.0f);
    }
  }
}

/** \{ */

Mesh *build_surface(const State &state, const Settings &settings)
{
  if (!settings.make_surface || state.density->activeVoxelCount() == 0) {
    return nullptr;
  }
  Mesh *mesh = bke::volume_grid_to_mesh(
      *state.density, settings.iso_level, std::max(0.0f, settings.adaptivity));
  if (mesh != nullptr) {
    bke::mesh_smooth_set(*mesh, true);
  }
  return mesh;
}

}  // namespace

Volume *solve_frame(const Volume *previous,
                    const Span<const SourceData> sources,
                    const Settings &settings,
                    const float delta_time,
                    Mesh **r_surface,
                    Stats &r_stats)
{
  const auto start = std::chrono::steady_clock::now();
  const float voxel_size = std::max(1e-5f, settings.voxel_size);

  State state;
  if (previous != nullptr) {
    state = read_state(*previous);
  }
  if (!state.valid() || !state.velocity) {
    init_state(state, voxel_size);
    r_stats.resized = true;
  }
  else if (std::abs(state.voxel_size() - voxel_size) > 1e-6f * voxel_size) {
    /* The resolution is part of the state: changing it invalidates the fields. */
    init_state(state, voxel_size);
    r_stats.resized = true;
  }

  const float dt = std::max(0.0f, delta_time);
  if (dt <= 0.0f) {
    /* First frame or a paused frame: only show the sources, do not integrate. */
    rasterize_sources(sources, state);
    ensure_velocity_topology(state);
    update_flame(state, settings);
    state.density->tree().prune(1e-5f);
    r_stats.active_voxels = int(state.density->activeVoxelCount());
    Volume *volume = BKE_id_new_nomain<Volume>(nullptr);
    write_state(*volume, state);
    if (r_surface != nullptr) {
      *r_surface = build_surface(state, settings);
    }
    r_stats.substeps_used = 0;
    r_stats.milliseconds = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    return volume;
  }

  /* CFL: a particle may cross `cfl` voxels per substep. */
  int substeps = std::clamp(settings.substeps, 1, 64);
  float max_speed = 0.0f;
  for (auto iter = state.velocity->cbeginValueOn(); iter; ++iter) {
    max_speed = std::max(max_speed, iter.getValue().length());
  }
  if (max_speed > 0.0f) {
    const int required = int(std::ceil(dt * max_speed / (std::max(0.1f, settings.cfl) * voxel_size)));
    if (required > substeps) {
      const int limit = std::max(1, settings.max_substeps);
      substeps = std::min(required, limit);
      r_stats.cfl_limited = required > limit;
    }
  }
  const float step_dt = dt / float(substeps);
  r_stats.substeps_used = substeps;

  for ([[maybe_unused]] const int substep : IndexRange(substeps)) {
    rasterize_sources(sources, state);
    ensure_velocity_topology(state);
    apply_forces(state, settings, step_dt);
    apply_vorticity(state, settings, step_dt);
    if (state.velocity->activeVoxelCount() > 0) {
      state.velocity = advect_with_velocity<openvdb::Vec3fGrid>(
          *state.velocity, *state.velocity, step_dt, settings);
    }
    project_velocity(state, settings, r_stats);
    state.density = advect_with_velocity<openvdb::FloatGrid>(
        *state.density, *state.velocity, step_dt, settings);
    state.temperature = advect_with_velocity<openvdb::FloatGrid>(
        *state.temperature, *state.velocity, step_dt, settings);
    state.fuel = advect_with_velocity<openvdb::FloatGrid>(
        *state.fuel, *state.velocity, step_dt, settings);
    combust_and_decay(state, settings, step_dt);
    clip_to_bounds(state, settings);
    limit_and_extrapolate_velocity(state, settings);
  }

  update_flame(state, settings);
  state.density->tree().prune(1e-5f);
  state.temperature->tree().prune(1e-5f);
  state.fuel->tree().prune(1e-5f);
  state.velocity->tree().prune(openvdb::Vec3f(1e-6f));
  r_stats.active_voxels = int(state.density->activeVoxelCount());

  if (r_surface != nullptr) {
    *r_surface = build_surface(state, settings);
  }
  Volume *volume = BKE_id_new_nomain<Volume>(nullptr);
  write_state(*volume, state);

  if (std::getenv("BLENDER_IGNIS_DEBUG") != nullptr) {
    float max_velocity = 0.0f;
    int64_t velocity_num = 0;
    for (auto iter = state.velocity->cbeginValueOn(); iter; ++iter) {
      max_velocity = std::max(max_velocity, iter.getValue().length());
      velocity_num++;
    }
    const openvdb::CoordBBox bbox = state.density->evalActiveVoxelBoundingBox();
    const openvdb::Vec3d bbox_min = state.density->transform().indexToWorld(bbox.min());
    const openvdb::Vec3d bbox_max = state.density->transform().indexToWorld(bbox.max());
    double weighted_vz = 0.0;
    double weight_sum = 0.0;
    {
      const openvdb::FloatGrid::ConstAccessor density_acc = state.density->getConstAccessor();
      for (auto iter = state.velocity->cbeginValueOn(); iter; ++iter) {
        const float density_here = density_acc.getValue(iter.getCoord());
        weighted_vz += double(density_here) * double(iter.getValue().z());
        weight_sum += double(density_here);
      }
    }
    float max_temperature = 0.0f;
    for (auto iter = state.temperature->cbeginValueOn(); iter; ++iter) {
      max_temperature = std::max(max_temperature, iter.getValue());
    }
    printf("IGNIS_DEBUG temperature max %.3f | ", double(max_temperature));
    printf("IGNIS_DEBUG bbox z %.3f..%.3f mean_vz %.3f | ",
           bbox_min.z(),
           bbox_max.z(),
           weight_sum > 0.0 ? weighted_vz / weight_sum : 0.0);
    printf("IGNIS_DEBUG step: density=%lld velocity=%lld max_v=%.3f div %.4f -> %.4f "
           "(%.1f%%) substeps=%d pressure=%d residual=%.5f cfl=%d\n",
           (long long)state.density->activeVoxelCount(),
           (long long)velocity_num,
           double(max_velocity),
           double(r_stats.divergence_before),
           double(r_stats.divergence_after),
           r_stats.divergence_before > 0.0f ?
               100.0 * double(r_stats.divergence_after / r_stats.divergence_before) :
               0.0,
           r_stats.substeps_used,
           r_stats.pressure_iterations,
           double(r_stats.residual),
           int(r_stats.cfl_limited));
  }
  r_stats.milliseconds = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
  return volume;
}

#else /* WITH_OPENVDB */

Volume *solve_frame(const Volume * /*previous*/,
                    const Span<const SourceData> /*sources*/,
                    const Settings & /*settings*/,
                    const float /*delta_time*/,
                    Mesh **r_surface,
                    Stats & /*r_stats*/)
{
  if (r_surface != nullptr) {
    *r_surface = nullptr;
  }
  return nullptr;
}

#endif /* WITH_OPENVDB */

}  // namespace blender::geometry::ignis
