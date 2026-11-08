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
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "BLI_array.hh"
#include "BLI_math_base.hh"
#include "BLI_math_geom.h"
#include "BLI_math_matrix.hh"
#include "BLI_math_vector.hh"
#include "BLI_task.hh"

#include "BKE_lib_id.hh"
#include "BKE_mesh.hh"
#include "BKE_volume.hh"

#include "GEO_ignis_solver.hh"
#include "GEO_ha_ignis.hh"

#ifdef WITH_OPENVDB
#  include "BKE_volume_grid.hh"
#  include "BKE_volume_openvdb.hh"

#  include <openvdb/openvdb.h>
#  include <openvdb/tools/Interpolation.h>
#  include <openvdb/tools/Morphology.h>
#  include <openvdb/tools/VolumeAdvect.h>
#endif

namespace blender::geometry::ignis {

#ifdef WITH_OPENVDB

namespace {

using openvdb::Coord;

/** Voxels per parallel chunk. Fixed so that a result never depends on the thread count. */
constexpr int64_t PARALLEL_CHUNK = 4096;

/**
 * Runs `fn(begin, end)` over fixed chunks and adds the per chunk results **in chunk order**, so the
 * sum is the same no matter how many threads run it. Used for every reduction in the solver.
 */
template<typename Func> double parallel_sum(const int64_t count, Func &&fn)
{
  const int64_t chunks = std::max<int64_t>(1, (count + PARALLEL_CHUNK - 1) / PARALLEL_CHUNK);
  Array<double> partial(chunks, 0.0);
  threading::parallel_for(IndexRange(chunks), 1, [&](const IndexRange range) {
    for (const int64_t chunk : range) {
      const int64_t begin = chunk * PARALLEL_CHUNK;
      const int64_t end = std::min(begin + PARALLEL_CHUNK, count);
      partial[chunk] = fn(begin, end);
    }
  });
  double total = 0.0;
  for (const double value : partial) {
    total += value;
  }
  return total;
}

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

/**
 * Radius in voxels of the region around the smoke that the solver works on.
 *
 * `Domain Padding` used to grow only the *velocity* band, while the pressure domain used
 * `extrapolate + 2`; the band was therefore three times bigger than the region that actually consumed
 * it, and the advection (the most expensive phase) ran over all of it. Now one radius drives the
 * pressure domain, the velocity band and the collider mask, so `Domain Padding` is what it says.
 */
int solver_radius(const Settings &settings)
{
  return std::max(3, settings.domain_padding);
}

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Source volumes
 * \{ */

/**
 * Adds a rasterized emitter (the volume of `Init Ignis Data`) to the state.
 *
 * The source may use a different resolution than the solver, so it is sampled at the voxel centers of
 * the state instead of being copied. Density, temperature and fuel keep the larger magnitude of the
 * two (a steady emitter holds its region at its own level instead of piling up until the simulation
 * explodes), and the velocity is blended in by the strength of the source. Merging by magnitude is
 * what makes several emitters add up: a hot source and a cold one next to it keep both their
 * temperatures instead of the last one overwriting the first.
 */
void add_source_volume(State &state, const Volume &source_volume)
{
  const bke::VolumeGridData *density_data = BKE_volume_grid_find(&source_volume, grids::density);
  if (density_data == nullptr) {
    return;
  }
  bke::VolumeTreeAccessToken density_token;
  const openvdb::GridBase &density_base = density_data->grid(density_token);
  if (density_base.type() != openvdb::FloatGrid::gridType()) {
    return;
  }
  const openvdb::FloatGrid &source_density = static_cast<const openvdb::FloatGrid &>(density_base);
  if (source_density.activeVoxelCount() == 0) {
    return;
  }

  const openvdb::FloatGrid *source_temperature = nullptr;
  const openvdb::FloatGrid *source_fuel = nullptr;
  const openvdb::Vec3fGrid *source_velocity = nullptr;
  bke::VolumeTreeAccessToken temperature_token;
  bke::VolumeTreeAccessToken fuel_token;
  bke::VolumeTreeAccessToken velocity_token;
  if (const bke::VolumeGridData *data = BKE_volume_grid_find(&source_volume,
                                                             grids::temperature))
  {
    const openvdb::GridBase &base = data->grid(temperature_token);
    if (base.type() == openvdb::FloatGrid::gridType()) {
      source_temperature = static_cast<const openvdb::FloatGrid *>(&base);
    }
  }
  if (const bke::VolumeGridData *data = BKE_volume_grid_find(&source_volume, grids::fuel)) {
    const openvdb::GridBase &base = data->grid(fuel_token);
    if (base.type() == openvdb::FloatGrid::gridType()) {
      source_fuel = static_cast<const openvdb::FloatGrid *>(&base);
    }
  }
  if (const bke::VolumeGridData *data = BKE_volume_grid_find(&source_volume, grids::velocity)) {
    const openvdb::GridBase &base = data->grid(velocity_token);
    if (base.type() == openvdb::Vec3fGrid::gridType()) {
      source_velocity = static_cast<const openvdb::Vec3fGrid *>(&base);
    }
  }

  openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> density_sampler(
      source_density);
  std::optional<openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>>
      temperature_sampler;
  std::optional<openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>>
      fuel_sampler;
  std::optional<openvdb::tools::GridSampler<openvdb::Vec3fGrid, openvdb::tools::BoxSampler>>
      velocity_sampler;
  if (source_temperature != nullptr) {
    temperature_sampler.emplace(*source_temperature);
  }
  if (source_fuel != nullptr) {
    fuel_sampler.emplace(*source_fuel);
  }
  if (source_velocity != nullptr) {
    velocity_sampler.emplace(*source_velocity);
  }

  /* Iterate the voxels of the *state* that the source can reach, so the sampling is right in both
   * directions: a coarse source is interpolated and a fine one is subsampled. */
  const openvdb::CoordBBox source_bbox = source_density.evalActiveVoxelBoundingBox();
  const openvdb::Vec3d min_world = source_density.transform().indexToWorld(source_bbox.min());
  const openvdb::Vec3d max_world = source_density.transform().indexToWorld(source_bbox.max());
  const openvdb::CoordBBox state_bbox = state.density->transform().worldToIndexCellCentered(
      openvdb::BBoxd(min_world, max_world));

  openvdb::FloatGrid::Accessor density = state.density->getAccessor();
  openvdb::FloatGrid::Accessor temperature = state.temperature->getAccessor();
  openvdb::FloatGrid::Accessor fuel = state.fuel->getAccessor();
  openvdb::Vec3fGrid::Accessor velocity = state.velocity->getAccessor();

  for (int k = state_bbox.min().z(); k <= state_bbox.max().z(); k++) {
    for (int j = state_bbox.min().y(); j <= state_bbox.max().y(); j++) {
      for (int i = state_bbox.min().x(); i <= state_bbox.max().x(); i++) {
        const Coord coord(i, j, k);
        const openvdb::Vec3d world = state.density->transform().indexToWorld(coord);
        const float source_value = density_sampler.wsSample(world);
        if (source_value <= 0.0f) {
          continue;
        }
        density.setValue(coord, std::max(density.getValue(coord), source_value));
        if (temperature_sampler.has_value()) {
          const float value = temperature_sampler->wsSample(world);
          if (value != 0.0f) {
            /* The temperature is an anomaly relative to the ambient, so a source can heat *and* cool:
             * a positive value pushes the temperature up, a negative one down. */
            const float current = temperature.getValue(coord);
            temperature.setValue(coord,
                                 value > 0.0f ? std::max(current, value) : std::min(current, value));
          }
        }
        if (fuel_sampler.has_value()) {
          const float value = fuel_sampler->wsSample(world);
          if (value > 0.0f) {
            fuel.setValue(coord, std::max(fuel.getValue(coord), value));
          }
        }
        if (velocity_sampler.has_value()) {
          const openvdb::Vec3f incoming = velocity_sampler->wsSample(world);
          if (incoming.lengthSqr() > 0.0f) {
            const float weight = std::clamp(source_value, 0.0f, 1.0f);
            const openvdb::Vec3f current = velocity.getValue(coord);
            velocity.setValue(coord, current + (incoming - current) * weight);
          }
        }
      }
    }
  }
}

/* -------------------------------------------------------------------- */
/** \name Transport
 * \{ */

/**
 * Advects a grid with the same tool the official `Grid Advect` node uses
 * (`openvdb::tools::VolumeAdvection`), which brings its own topology growth and the higher order
 * schemes with limiters. Doing this by hand only ever gives first/second order.
 */
using Advection = openvdb::tools::VolumeAdvection<openvdb::Vec3fGrid, false>;

/** Applies the scheme and limiter of the settings to a freshly built advection tool. */
void configure_advection(Advection &advection, const Settings &settings)
{
  advection.setIntegrator(to_scheme(settings.advection_scheme));
  advection.setLimiter(to_limiter(settings.advection_limiter));
}

template<typename GridType>
typename GridType::Ptr advect_with_velocity(const GridType &grid,
                                           Advection &advection,
                                           const float delta_time)
{
  return advection.template advect<GridType, openvdb::tools::Sampler<1>>(grid, delta_time);
}

/**
 * True when density, temperature and fuel occupy exactly the same voxels.
 *
 * This is the common case (temperature and fuel are written by the source and the combustion at
 * the same voxels as density), and then the three grids can be advected as one Vec3 grid: the
 * backtrace and the four velocity samples are shared instead of being repeated per field. The
 * limiter still works per component, so the values are the same.
 */
bool scalar_grids_share_topology(const State &state)
{
  const int64_t density_count = state.density->activeVoxelCount();
  if (density_count == 0) {
    return false;
  }
  if (state.temperature->activeVoxelCount() != density_count ||
      state.fuel->activeVoxelCount() != density_count)
  {
    return false;
  }
  const openvdb::FloatGrid::ConstAccessor temperature = state.temperature->getConstAccessor();
  const openvdb::FloatGrid::ConstAccessor fuel = state.fuel->getConstAccessor();
  for (auto iter = state.density->cbeginValueOn(); iter; ++iter) {
    const openvdb::Coord &coord = iter.getCoord();
    if (!temperature.isValueOn(coord) || !fuel.isValueOn(coord)) {
      return false;
    }
  }
  return true;
}

/** Advects density, temperature and fuel in one Vec3 grid. Returns false when not applicable. */
bool advect_scalars_packed(State &state, const Settings &settings, const float delta_time)
{
  if (std::getenv("BLENDER_IGNIS_NO_PACK_SCALARS") != nullptr) {
    return false;
  }
  if (!scalar_grids_share_topology(state)) {
    return false;
  }
  const openvdb::FloatGrid::ConstAccessor density = state.density->getConstAccessor();
  const openvdb::FloatGrid::ConstAccessor temperature = state.temperature->getConstAccessor();
  const openvdb::FloatGrid::ConstAccessor fuel = state.fuel->getConstAccessor();

  openvdb::Vec3fGrid::Ptr packed = openvdb::Vec3fGrid::create(openvdb::Vec3f(0.0f));
  packed->setTransform(state.density->transformPtr());
  openvdb::Vec3fGrid::Accessor packed_writer = packed->getAccessor();
  for (auto iter = state.density->cbeginValueOn(); iter; ++iter) {
    const openvdb::Coord &coord = iter.getCoord();
    packed_writer.setValue(coord,
                           openvdb::Vec3f(density.getValue(coord),
                                          temperature.getValue(coord),
                                          fuel.getValue(coord)));
  }

  Advection advection(*state.velocity);
  configure_advection(advection, settings);
  openvdb::Vec3fGrid::Ptr result = advect_with_velocity<openvdb::Vec3fGrid>(
      *packed, advection, delta_time);

  openvdb::FloatGrid::Ptr new_density = create_float_grid(state.density->transformPtr());
  openvdb::FloatGrid::Ptr new_temperature = create_float_grid(state.temperature->transformPtr());
  openvdb::FloatGrid::Ptr new_fuel = create_float_grid(state.fuel->transformPtr());
  openvdb::FloatGrid::Accessor density_writer = new_density->getAccessor();
  openvdb::FloatGrid::Accessor temperature_writer = new_temperature->getAccessor();
  openvdb::FloatGrid::Accessor fuel_writer = new_fuel->getAccessor();
  const openvdb::Vec3fGrid::ConstAccessor result_reader = result->getConstAccessor();
  for (auto iter = result->cbeginValueOn(); iter; ++iter) {
    const openvdb::Coord &coord = iter.getCoord();
    const openvdb::Vec3f value = result_reader.getValue(coord);
    density_writer.setValue(coord, value.x());
    temperature_writer.setValue(coord, value.y());
    fuel_writer.setValue(coord, value.z());
  }
  state.density = new_density;
  state.temperature = new_temperature;
  state.fuel = new_fuel;

  if (std::getenv("BLENDER_IGNIS_DEBUG") != nullptr) {
    printf("IGNIS_DEBUG scalars packed: %lld voxels\n",
           (long long)state.density->activeVoxelCount());
  }
  return true;
}

/** \{ */

/* -------------------------------------------------------------------- */
/** \name Forces
 * \{ */

/* -------------------------------------------------------------------- */
/** \name Procedural forces
 *
 * The same 32 bit hash and value noise are used by the device kernels, so the CPU and the GPU see
 * the same disturbance / turbulence pattern. The hash is cheap, integer only and independent of
 * the thread that evaluates it.
 * \{ */

uint32_t force_hash_u32(uint32_t x)
{
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

/** One of three signed random values packed into the 32 bit hash. */
float force_hash_channel(const uint32_t hash, const int channel)
{
  const uint32_t shift = uint32_t(channel) * 10u;
  return float((hash >> shift) & 0x3FFu) * (2.0f / 1023.0f) - 1.0f;
}

float3 force_random3(const int3 cell, const int seed)
{
  const uint32_t hash = force_hash_u32(
      uint32_t(cell.x) +
      force_hash_u32(uint32_t(cell.y) + force_hash_u32(uint32_t(cell.z) +
                                                       force_hash_u32(uint32_t(seed)))));
  return float3(force_hash_channel(hash, 0),
                force_hash_channel(hash, 1),
                force_hash_channel(hash, 2));
}

float force_fade(const float t)
{
  return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

float3 force_lerp(const float3 a, const float3 b, const float t)
{
  /* Match GLSL `mix()` exactly. */
  return a * (1.0f - t) + b * t;
}

float3 force_value_noise(const float3 p, const int seed)
{
  const float3 base(std::floor(p.x), std::floor(p.y), std::floor(p.z));
  const int3 cell(int(base.x), int(base.y), int(base.z));
  const float3 f = p - base;
  const float3 u(force_fade(f.x), force_fade(f.y), force_fade(f.z));

  const float3 n000 = force_random3(int3(cell.x, cell.y, cell.z), seed);
  const float3 n100 = force_random3(int3(cell.x + 1, cell.y, cell.z), seed);
  const float3 n010 = force_random3(int3(cell.x, cell.y + 1, cell.z), seed);
  const float3 n110 = force_random3(int3(cell.x + 1, cell.y + 1, cell.z), seed);
  const float3 n001 = force_random3(int3(cell.x, cell.y, cell.z + 1), seed);
  const float3 n101 = force_random3(int3(cell.x + 1, cell.y, cell.z + 1), seed);
  const float3 n011 = force_random3(int3(cell.x, cell.y + 1, cell.z + 1), seed);
  const float3 n111 = force_random3(int3(cell.x + 1, cell.y + 1, cell.z + 1), seed);

  const float3 n00 = force_lerp(n000, n100, u.x);
  const float3 n10 = force_lerp(n010, n110, u.x);
  const float3 n01 = force_lerp(n001, n101, u.x);
  const float3 n11 = force_lerp(n011, n111, u.x);
  const float3 n0 = force_lerp(n00, n10, u.y);
  const float3 n1 = force_lerp(n01, n11, u.y);
  return force_lerp(n0, n1, u.z);
}

/** Forward difference curl of the vector value noise (divergence free up to the stencil error). */
float3 force_curl_noise(const float3 p, const int seed)
{
  constexpr float eps = 0.5f;
  const float3 v0 = force_value_noise(p, seed);
  const float3 vx = force_value_noise(p + float3(eps, 0.0f, 0.0f), seed);
  const float3 vy = force_value_noise(p + float3(0.0f, eps, 0.0f), seed);
  const float3 vz = force_value_noise(p + float3(0.0f, 0.0f, eps), seed);
  return float3(((vy.z - v0.z) - (vz.y - v0.y)) / eps,
                ((vz.x - v0.x) - (vx.z - v0.z)) / eps,
                ((vx.y - v0.y) - (vy.x - v0.x)) / eps);
}

float3 force_disturbance(const float3 p, const Settings &settings)
{
  if (settings.disturbance == 0.0f) {
    return float3(0.0f);
  }
  const float scale = std::max(1e-4f, settings.disturbance_scale);
  const float3 flow = float3(settings.force_time * settings.disturbance_speed,
                             settings.force_time * settings.disturbance_speed * 0.73f,
                             settings.force_time * settings.disturbance_speed * 0.41f);
  float3 q = p / scale + flow;
  const int octaves = std::clamp(settings.disturbance_octaves, 1, 8);
  const float lacunarity = std::max(1.0f, settings.disturbance_lacunarity);
  const float roughness = std::clamp(settings.disturbance_roughness, 0.0f, 1.0f);
  float amplitude = 1.0f;
  float total = 0.0f;
  float3 result(0.0f);
  for (int octave = 0; octave < octaves; octave++) {
    const float3 offset(float(octave) * 17.3f, float(octave) * 9.1f, float(octave) * 3.7f);
    result += force_value_noise(q + offset, settings.disturbance_seed + octave * 131) * amplitude;
    total += amplitude;
    amplitude *= roughness;
    q *= lacunarity;
  }
  return result * (settings.disturbance / std::max(total, 1e-4f));
}

float3 force_turbulence(const float3 p, const Settings &settings)
{
  if (settings.turbulence == 0.0f) {
    return float3(0.0f);
  }
  const float scale = std::max(1e-4f, settings.turbulence_scale);
  const float3 flow = float3(settings.force_time * settings.turbulence_speed,
                             settings.force_time * settings.turbulence_speed * 0.61f,
                             settings.force_time * settings.turbulence_speed * 0.37f);
  float3 q = p / scale + flow;
  const int octaves = std::clamp(settings.turbulence_octaves, 1, 8);
  const float lacunarity = std::max(1.0f, settings.turbulence_lacunarity);
  const float roughness = std::clamp(settings.turbulence_roughness, 0.0f, 1.0f);
  float amplitude = 1.0f;
  float total = 0.0f;
  float3 result(0.0f);
  for (int octave = 0; octave < octaves; octave++) {
    const float3 offset(float(octave) * 7.1f, float(octave) * 23.7f, float(octave) * 11.3f);
    result += force_curl_noise(q + offset, settings.turbulence_seed + octave * 197) * amplitude;
    total += amplitude;
    amplitude *= roughness;
    q *= lacunarity;
  }
  return result * (settings.turbulence / std::max(total, 1e-4f));
}

/** Smooth window used by the force remap. `softness` is a fraction of the field range. */
float force_remap_window(const float value,
                         const int mode,
                         float min,
                         float max,
                         const float softness)
{
  if (min > max) {
    std::swap(min, max);
  }
  const float range = std::max(max - min, 1e-6f);
  const float soft = std::clamp(softness, 0.0f, 1.0f) * range;
  auto smooth = [](const float edge0, const float edge1, const float x) {
    if (edge1 <= edge0) {
      return x < edge0 ? 0.0f : 1.0f;
    }
    const float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
  };
  const float lower = smooth(min, min + soft, value);
  const float upper = 1.0f - smooth(max - soft, max, value);
  const float inside = lower * upper;
  switch (mode) {
    case 1:
      return 1.0f - inside;
    case 2:
      return lower;
    case 3:
      return upper;
    case 0:
    default:
      return inside;
  }
}

/** \} */

void apply_forces(State &state, const Settings &settings, const float delta_time)
{
  const float dt = delta_time;
  const float buoyancy = settings.buoyancy;
  const float ambient = settings.ambient_temperature;
  const float drag = std::max(0.0f, settings.drag);
  const float damping = std::max(0.0f, settings.velocity_damping);
  const float damping_factor = 1.0f / (1.0f + damping * dt);

  /* The forces act on the smoke, not on the empty air: the weight of the ambient air is balanced by
   * the hydrostatic pressure that this solver does not model, so accelerating the whole velocity
   * band with gravity would drag the smoke down through the pressure solve. Iterating the smoke (a
   * tenth of the band) instead of the band and testing the density is most of the win here, and the
   * coordinates are collected once so the work can be split over the threads. Every write goes to a
   * voxel that already exists, so the chunks cannot race. */
  Vector<Coord> coords;
  coords.reserve(int(state.density->activeVoxelCount()));
  for (auto iter = state.density->cbeginValueOn(); iter; ++iter) {
    if (iter.getValue() > 1e-4f) {
      coords.append(iter.getCoord());
    }
  }
  auto grid_accessor = [](const openvdb::FloatGrid::Ptr &grid)
      -> std::optional<openvdb::FloatGrid::ConstAccessor> {
    if (!grid) {
      return std::nullopt;
    }
    return grid->getConstAccessor();
  };
  const std::optional<openvdb::FloatGrid::ConstAccessor> density_accessor =
      grid_accessor(state.density);
  const std::optional<openvdb::FloatGrid::ConstAccessor> temperature_accessor =
      grid_accessor(state.temperature);
  const std::optional<openvdb::FloatGrid::ConstAccessor> fuel_accessor =
      grid_accessor(state.fuel);
  const std::optional<openvdb::FloatGrid::ConstAccessor> flame_accessor =
      grid_accessor(state.flame);
  const std::optional<openvdb::FloatGrid::ConstAccessor> pressure_accessor =
      grid_accessor(state.pressure);

  auto read_field = [&](const int field, const float velocity_length, const Coord &coord) {
    switch (field) {
      case 1:
        return pressure_accessor ? pressure_accessor->getValue(coord) : 0.0f;
      case 2:
        return density_accessor ? density_accessor->getValue(coord) : 0.0f;
      case 3:
        return temperature_accessor ? temperature_accessor->getValue(coord) : 0.0f;
      case 4:
        return fuel_accessor ? fuel_accessor->getValue(coord) : 0.0f;
      case 5:
        return flame_accessor ? flame_accessor->getValue(coord) : 0.0f;
      case 6:
        return velocity_length;
      default:
        return 1.0f;
    }
  };

  threading::parallel_for(IndexRange(int64_t(coords.size())), PARALLEL_CHUNK,
                          [&](const IndexRange range) {
    openvdb::Vec3fGrid::Accessor velocity = state.velocity->getAccessor();
    for (const int64_t i : range) {
      const Coord &coord = coords[i];
      float3 value = from_vdb(velocity.getValue(coord));
      /* Boussinesq buoyancy: the ambient air is in hydrostatic equilibrium, so the only body force
       * on the smoke is its density anomaly. With a temperature relative to the ambient that anomaly
       * is just `T - ambient`, and `buoyancy = 1` means one unit above ambient accelerates the smoke
       * at `g`: positive temperatures rise, negative ones sink, no separate gravity is applied. */
      const float temperature = temperature_accessor ? temperature_accessor->getValue(coord) : 0.0f;
      const float anomaly = temperature - ambient;
      if (anomaly != 0.0f) {
        /* Opposite to gravity, so a positive anomaly rises and a negative one sinks. */
        value -= settings.gravity * (buoyancy * anomaly * dt);
      }
      value += settings.wind * dt;
      if (settings.disturbance != 0.0f || settings.turbulence != 0.0f) {
        const openvdb::Vec3d world = state.velocity->transform().indexToWorld(coord);
        const float3 position(float(world.x()), float(world.y()), float(world.z()));
        const float velocity_length = math::length(value);

        auto remap_factor = [&](const int field,
                                const int mode,
                                const float min,
                                const float max,
                                const float softness,
                                const float scale) {
          if (field == 0) {
            return 1.0f;
          }
          if (scale <= 0.0f) {
            return 0.0f;
          }
          const float field_value = read_field(field, velocity_length, coord);
          return force_remap_window(field_value, mode, min, max, softness) * scale;
        };

        const float disturbance_factor = remap_factor(settings.disturbance_remap_field,
                                                      settings.disturbance_remap_mode,
                                                      settings.disturbance_remap_min,
                                                      settings.disturbance_remap_max,
                                                      settings.disturbance_remap_softness,
                                                      settings.disturbance_remap_scale);
        const float turbulence_factor = remap_factor(settings.turbulence_remap_field,
                                                     settings.turbulence_remap_mode,
                                                     settings.turbulence_remap_min,
                                                     settings.turbulence_remap_max,
                                                     settings.turbulence_remap_softness,
                                                     settings.turbulence_remap_scale);
        if (disturbance_factor != 0.0f) {
          value += force_disturbance(position, settings) * (dt * disturbance_factor);
        }
        if (turbulence_factor != 0.0f) {
          value += force_turbulence(position, settings) * (dt * turbulence_factor);
        }
      }
      if (drag > 0.0f) {
        value *= 1.0f / (1.0f + drag * dt);
      }
      value *= damping_factor;
      velocity.setValue(coord, to_vdb(value));
    }
  });
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

  /* The curl is only needed where the smoke is: the confinement pushes the smoke, and the velocity
   * of the extrapolated air is rebuilt from it every substep anyway. One layer of overlap keeps the
   * stencil inside the band. */
  const openvdb::FloatTree band = smoke_tree(state, 1);
  for (auto iter = band.cbeginValueOn(); iter; ++iter) {
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
/** Accumulates milliseconds into a phase, for the `BLENDER_IGNIS_DEBUG` profile. */
struct ScopedTimer {
  double &target;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  explicit ScopedTimer(double &target) : target(target) {}
  ~ScopedTimer()
  {
    target += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                  .count();
  }
};

/** \name Colliders
 * \{ */

/**
 * A thin wall around the surface of the colliders, on a dense box around the smoke.
 *
 * Only the surface is marked, not the inside of a closed mesh: the smoke cannot cross a shell
 * either, and it lets an open mesh (a ground plane) and a closed one (a rock) take the same code
 * path with no inside/outside test, which is what makes it cheap enough to rebuild every frame.
 */
struct SolidMask {
  Coord min{0, 0, 0};
  Coord max{-1, -1, -1};
  Array<int8_t> cells;

  bool empty() const
  {
    return cells.is_empty();
  }

  bool contains(const Coord &coord) const
  {
    return coord.x() >= min.x() && coord.x() <= max.x() && coord.y() >= min.y() &&
           coord.y() <= max.y() && coord.z() >= min.z() && coord.z() <= max.z();
  }

  bool is_solid(const Coord &coord) const
  {
    if (cells.is_empty() || !this->contains(coord)) {
      return false;
    }
    const int64_t x = coord.x() - min.x();
    const int64_t y = coord.y() - min.y();
    const int64_t z = coord.z() - min.z();
    return cells[int64_t(x) + int64_t(max.x() - min.x() + 1) *
                                   (int64_t(y) + int64_t(max.y() - min.y() + 1) * int64_t(z))] != 0;
  }

  /** The collider index + 1 of a solid voxel, or 0. */
  int8_t id(const Coord &coord) const
  {
    if (cells.is_empty() || !this->contains(coord)) {
      return 0;
    }
    const int64_t x = coord.x() - min.x();
    const int64_t y = coord.y() - min.y();
    const int64_t z = coord.z() - min.z();
    return cells[int64_t(x) + int64_t(max.x() - min.x() + 1) *
                                   (int64_t(y) + int64_t(max.y() - min.y() + 1) * int64_t(z))];
  }

  int64_t count() const
  {
    int64_t total = 0;
    for (const int8_t value : cells) {
      total += value != 0 ? 1 : 0;
    }
    return total;
  }
};

/** Marks the voxels whose center is within \a radius of the triangle \a a \a b \a c. */
void mark_triangle(SolidMask &mask,
                   const openvdb::math::Transform &transform,
                   const float3 a,
                   const float3 b,
                   const float3 c,
                   const float radius,
                   const int8_t value)
{
  const float voxel = float(transform.voxelSize()[0]);
  const float3 grow(radius + voxel);
  const float3 lo = math::min(math::min(a, b), c) - grow;
  const float3 hi = math::max(math::max(a, b), c) + grow;
  const Coord bmin = transform.worldToIndexCellCentered(openvdb::Vec3d(lo.x, lo.y, lo.z));
  const Coord bmax = transform.worldToIndexCellCentered(openvdb::Vec3d(hi.x, hi.y, hi.z));
  /* Only the part of the triangle that is inside the mask is visited: a 20 m ground plane costs the
   * same as a small one. */
  const int x0 = std::max(bmin.x(), mask.min.x());
  const int y0 = std::max(bmin.y(), mask.min.y());
  const int z0 = std::max(bmin.z(), mask.min.z());
  const int x1 = std::min(bmax.x(), mask.max.x());
  const int y1 = std::min(bmax.y(), mask.max.y());
  const int z1 = std::min(bmax.z(), mask.max.z());
  if (x0 > x1 || y0 > y1 || z0 > z1) {
    return;
  }
  const int dim_x = mask.max.x() - mask.min.x() + 1;
  const int dim_y = mask.max.y() - mask.min.y() + 1;
  const float a3[3] = {a.x, a.y, a.z};
  const float b3[3] = {b.x, b.y, b.z};
  const float c3[3] = {c.x, c.y, c.z};
  const float radius_sq = radius * radius;
  for (int z = z0; z <= z1; z++) {
    for (int y = y0; y <= y1; y++) {
      for (int x = x0; x <= x1; x++) {
        const openvdb::Vec3d world = transform.indexToWorld(Coord(x, y, z));
        const float point[3] = {float(world.x()), float(world.y()), float(world.z())};
        float closest[3];
        closest_on_tri_to_point_v3(closest, point, a3, b3, c3);
        const float dx = point[0] - closest[0];
        const float dy = point[1] - closest[1];
        const float dz = point[2] - closest[2];
        if (dx * dx + dy * dy + dz * dz > radius_sq) {
          continue;
        }
        const int64_t ix = x - mask.min.x();
        const int64_t iy = y - mask.min.y();
        const int64_t iz = z - mask.min.z();
        mask.cells[ix + int64_t(dim_x) * (iy + int64_t(dim_y) * iz)] = value;
      }
    }
  }
}

/** The box the smoke is in, grown by the room it can reach in one frame. */
openvdb::CoordBBox solid_region(const State &state, const Settings &settings)
{
  openvdb::CoordBBox box = state.density->evalActiveVoxelBoundingBox();
  if (box.empty()) {
    return box;
  }
  const int pad = solver_radius(settings) + 2;
  box.expand(pad);
  return box;
}

/**
 * The shape of the colliders of one frame.
 *
 * The colliders are rasterized as a thin shell (that is all an open mesh can give), so the interior
 * of a closed mesh is filled separately: a solid volume is what the pressure solve needs, and it is
 * also what tells `carve` that a voxel is inside the object rather than in the one voxel thick shell
 * around its surface. Carving the shell of a descending collider would remove a layer of smoke above
 * it every frame (the "collider eats the smoke" artifact), so the shell is only carved when the
 * collider does not move.
 */
struct ColliderMasks {
  /** Union of the collider poses over the frame, used for the wall and the pressure domain. */
  SolidMask swept;
  /** The pose of the last substep, used to impose the collider velocity on the smoke. */
  SolidMask current;
  /** The current pose with the interior of the closed meshes filled in (the solid volume). */
  SolidMask filled;
  /** Voxels that may be removed: interiors, plus the shell of colliders that do not move. */
  SolidMask carve;
};

static void init_solid_mask(SolidMask &mask, const openvdb::CoordBBox &region)
{
  mask.min = region.min();
  mask.max = region.max();
  const int dim_x = mask.max.x() - mask.min.x() + 1;
  const int dim_y = mask.max.y() - mask.min.y() + 1;
  const int dim_z = mask.max.z() - mask.min.z() + 1;
  mask.cells = Array<int8_t>(int64_t(dim_x) * dim_y * dim_z, 0);
}

/**
 * Flood fills the outside of the current shell from the border of the mask, which marks every cell
 * that is not reachable from the outside as interior. For an open mesh (a ground plane) everything
 * is reachable, so #r_filled stays a shell and nothing is carved.
 */
static void fill_collider_interior(const SolidMask &shell,
                                   Span<Collider> colliders,
                                   SolidMask &r_filled,
                                   SolidMask &r_carve)
{
  r_filled = shell;
  const openvdb::CoordBBox region(shell.min, shell.max);
  init_solid_mask(r_carve, region);
  if (shell.empty()) {
    return;
  }
  const int dim_x = shell.max.x() - shell.min.x() + 1;
  const int dim_y = shell.max.y() - shell.min.y() + 1;
  const int dim_z = shell.max.z() - shell.min.z() + 1;
  const int64_t cells = int64_t(dim_x) * dim_y * dim_z;
  auto index_of = [&](const int x, const int y, const int z) -> int64_t {
    return int64_t(x) + int64_t(dim_x) * (int64_t(y) + int64_t(dim_y) * int64_t(z));
  };

  Array<uint8_t> outside(cells, 0);
  Vector<int64_t> stack;
  stack.reserve(int64_t(cells) / 4 + 1);
  auto push_if_free = [&](const int x, const int y, const int z) {
    if (x < 0 || y < 0 || z < 0 || x >= dim_x || y >= dim_y || z >= dim_z) {
      return;
    }
    const int64_t index = index_of(x, y, z);
    if (!outside[index] && shell.cells[index] == 0) {
      outside[index] = 1;
      stack.append(index);
    }
  };
  for (int z = 0; z < dim_z; z++) {
    for (int y = 0; y < dim_y; y++) {
      push_if_free(0, y, z);
      push_if_free(dim_x - 1, y, z);
    }
  }
  for (int z = 0; z < dim_z; z++) {
    for (int x = 0; x < dim_x; x++) {
      push_if_free(x, 0, z);
      push_if_free(x, dim_y - 1, z);
    }
  }
  for (int y = 0; y < dim_y; y++) {
    for (int x = 0; x < dim_x; x++) {
      push_if_free(x, y, 0);
      push_if_free(x, y, dim_z - 1);
    }
  }

  while (!stack.is_empty()) {
    const int64_t index = stack.pop_last();
    const int x = int(index % dim_x);
    const int y = int((index / dim_x) % dim_y);
    const int z = int(index / (int64_t(dim_x) * dim_y));
    push_if_free(x + 1, y, z);
    push_if_free(x - 1, y, z);
    push_if_free(x, y + 1, z);
    push_if_free(x, y - 1, z);
    push_if_free(x, y, z + 1);
    push_if_free(x, y, z - 1);
  }

  /* A moving collider must not carve at all: the smoke it pushes out of the way has to stay in the
   * domain. Deleting the cells it occupies (or the ones the swept shell touches) is what makes a
   * falling object look like it eats the smoke. The pressure solve and the velocity clamp keep the
   * smoke out of the solid volume instead; when the collider has passed, the smoke is still there. */
  bool any_moving = false;
  for (const Collider &collider : colliders) {
    if (math::length_squared(collider.velocity) > 0.0f) {
      any_moving = true;
      break;
    }
  }

  for (int64_t i = 0; i < cells; i++) {
    if (outside[i]) {
      continue;
    }
    r_filled.cells[i] = 1;
    if (!any_moving) {
      r_carve.cells[i] = 1;
    }
  }
}

/**
 * Rasterizes the colliders into a solid mask for the frame.
 *
 * Colliders are sampled at several poses between the previous and the current frame (when `Sweep`
 * is on), so a fast mover cannot jump over the smoke. Each cell stores the collider index + 1, which
 * lets the velocity pass know how the wall at that cell moves. `Deforming` interpolates the vertices
 * of the previous mesh instead of the object transform.
 */
ColliderMasks build_collider_masks(const openvdb::math::Transform &transform,
                                   MutableSpan<Collider> colliders,
                                   const openvdb::CoordBBox &region,
                                   const float delta_time)
{
  ColliderMasks masks;
  if (colliders.is_empty() || region.empty()) {
    return masks;
  }
  init_solid_mask(masks.swept, region);
  init_solid_mask(masks.current, region);

  /* A voxel is solid when its center is within half a voxel diagonal of the surface, so the shell
   * has no holes even where a triangle is smaller than a voxel. */
  const float voxel = float(transform.voxelSize()[0]);
  const float radius = voxel * 0.9f;
  for (const int collider_i : colliders.index_range()) {
    Collider &collider = colliders[collider_i];
    const Mesh *mesh = collider.mesh;
    if (mesh == nullptr) {
      continue;
    }
    const Span<float3> positions = mesh->vert_positions();
    const Span<int> corner_verts = mesh->corner_verts();
    const Span<int3> corner_tris = mesh->corner_tris();
    if (positions.is_empty() || corner_tris.is_empty()) {
      continue;
    }
    const int8_t value = int8_t(std::min(collider_i + 1, 127));

    const Mesh *previous_mesh = collider.previous_mesh;
    const bool deforming = collider.deforming && previous_mesh != nullptr;
    const Span<float3> previous_positions = deforming ? previous_mesh->vert_positions() : Span<float3>();
    const Span<int> previous_corner_verts = deforming ? previous_mesh->corner_verts() : Span<int>();
    const bool can_deform = deforming && previous_positions.size() == positions.size() &&
                            previous_corner_verts.size() == corner_verts.size();
    const bool can_sweep_rigid = collider.has_previous_transform && !can_deform;
    const float4x4 delta = can_sweep_rigid ? collider.previous_transform *
                                                math::invert(collider.transform) :
                                            float4x4::identity();

    /* How far the collider moves this frame: it decides how many poses the swept mask needs. */
    float max_motion = 0.0f;
    if (can_deform) {
      for (const int i : positions.index_range()) {
        max_motion = std::max(max_motion, math::distance(positions[i], previous_positions[i]));
      }
    }
    else if (can_sweep_rigid) {
      for (const int i : positions.index_range()) {
        max_motion = std::max(
            max_motion, math::distance(positions[i], math::transform_point(delta, positions[i])));
      }
    }

    /* Average motion per second, used to push the smoke out of the way. */
    if (delta_time > 0.0f) {
      if (can_sweep_rigid) {
        const float3 current_origin = math::transform_point(collider.transform, float3(0.0f));
        const float3 previous_origin = math::transform_point(collider.previous_transform, float3(0.0f));
        collider.velocity = (current_origin - previous_origin) / delta_time;
      }
      else if (can_deform) {
        float3 sum(0.0f);
        for (const int i : positions.index_range()) {
          sum += positions[i] - previous_positions[i];
        }
        if (!positions.is_empty()) {
          collider.velocity = sum / (float(positions.size()) * delta_time);
        }
      }
    }

    const int samples = (collider.sweep && (can_deform || can_sweep_rigid)) ?
                            std::clamp(int(std::ceil(max_motion / std::max(voxel, 1e-5f))), 1, 16) :
                            0;

    for (int sample = 0; sample <= samples; sample++) {
      const float t = samples == 0 ? 1.0f : float(sample) / float(samples);
      float4x4 rigid = float4x4::identity();
      if (can_sweep_rigid && sample < samples) {
        rigid = math::interpolate(delta, float4x4::identity(), t);
      }
      for (const int3 &tri : corner_tris) {
        const int i0 = corner_verts[tri.x];
        const int i1 = corner_verts[tri.y];
        const int i2 = corner_verts[tri.z];
        float3 a, b, c;
        if (can_deform) {
          a = math::interpolate(previous_positions[i0], positions[i0], t);
          b = math::interpolate(previous_positions[i1], positions[i1], t);
          c = math::interpolate(previous_positions[i2], positions[i2], t);
        }
        else {
          a = math::transform_point(rigid, positions[i0]);
          b = math::transform_point(rigid, positions[i1]);
          c = math::transform_point(rigid, positions[i2]);
        }
        mark_triangle(masks.swept, transform, a, b, c, radius, value);
        if (sample == samples) {
          mark_triangle(masks.current, transform, a, b, c, radius, value);
        }
      }
    }
  }
  fill_collider_interior(masks.current, colliders, masks.filled, masks.carve);
  return masks;
}

/** Removes the smoke and stops the velocity inside the colliders. */
void carve_solids(State &state, const SolidMask &mask)
{
  if (mask.empty()) {
    return;
  }
  /* Every write goes to a voxel that already exists, so the chunks cannot race: each one gets its own
   * accessor and the tree is never restructured. */
  Vector<Coord> coords;
  coords.reserve(int(state.density->activeVoxelCount()));
  for (auto iter = state.density->cbeginValueOn(); iter; ++iter) {
    if (mask.is_solid(iter.getCoord())) {
      coords.append(iter.getCoord());
    }
  }
  if (coords.is_empty()) {
    return;
  }
  threading::parallel_for(IndexRange(int64_t(coords.size())), PARALLEL_CHUNK,
                          [&](const IndexRange range) {
    openvdb::FloatGrid::Accessor density = state.density->getAccessor();
    openvdb::FloatGrid::Accessor temperature = state.temperature->getAccessor();
    openvdb::FloatGrid::Accessor fuel = state.fuel->getAccessor();
    openvdb::Vec3fGrid::Accessor velocity = state.velocity->getAccessor();
    for (const int64_t i : range) {
      const Coord &coord = coords[i];
      density.setValue(coord, 0.0f);
      temperature.setValue(coord, 0.0f);
      fuel.setValue(coord, 0.0f);
      velocity.setValue(coord, openvdb::Vec3f(0.0f));
    }
  });
}

/**
 * Stops the velocity from pointing into a collider and follows a moving wall.
 *
 * The pressure solve treats a solid as `p = 0`, i.e. as a wall, but the collocated velocity has no
 * face to project onto, so the component that pushes the smoke into the wall is simply clipped: the
 * smoke slides along the surface instead of being eaten by the carve. With a moving collider the
 * clip is against the wall velocity (relative normal velocity), not against zero, so a descending
 * lid pushes the smoke ahead of it while smoke beside or above the moving surface is free to slide
 * (free slip). Imposing the wall velocity unconditionally would drag the smoke along with the
 * collider, which is what makes falling objects look like they swallow the smoke.
 */
void clamp_velocity_at_solids(State &state,
                              const SolidMask &mask,
                              Span<Collider> colliders)
{
  if (mask.empty()) {
    return;
  }
  Vector<Coord> coords;
  coords.reserve(int(state.velocity->activeVoxelCount()));
  for (auto iter = state.velocity->cbeginValueOn(); iter; ++iter) {
    coords.append(iter.getCoord());
  }
  threading::parallel_for(IndexRange(int64_t(coords.size())), PARALLEL_CHUNK,
                          [&](const IndexRange range) {
    openvdb::Vec3fGrid::Accessor velocity = state.velocity->getAccessor();
    auto wall_velocity = [&](const Coord &coord) -> std::optional<float3> {
      const int8_t id = mask.id(coord);
      if (id == 0 || id > colliders.size()) {
        return std::nullopt;
      }
      const float3 wall = colliders[id - 1].velocity;
      if (math::length_squared(wall) <= 0.0f) {
        return std::nullopt;
      }
      return wall;
    };
    for (const int64_t i : range) {
      const Coord &coord = coords[i];
      if (mask.is_solid(coord)) {
        velocity.setValue(coord, openvdb::Vec3f(0.0f));
        continue;
      }
      openvdb::Vec3f value = velocity.getValue(coord);
      if (mask.is_solid(coord.offsetBy(1, 0, 0))) {
        const std::optional<float3> wall = wall_velocity(coord.offsetBy(1, 0, 0));
        value.x() = wall ? std::min(value.x(), wall->x) : std::min(value.x(), 0.0f);
      }
      if (mask.is_solid(coord.offsetBy(-1, 0, 0))) {
        const std::optional<float3> wall = wall_velocity(coord.offsetBy(-1, 0, 0));
        value.x() = wall ? std::max(value.x(), wall->x) : std::max(value.x(), 0.0f);
      }
      if (mask.is_solid(coord.offsetBy(0, 1, 0))) {
        const std::optional<float3> wall = wall_velocity(coord.offsetBy(0, 1, 0));
        value.y() = wall ? std::min(value.y(), wall->y) : std::min(value.y(), 0.0f);
      }
      if (mask.is_solid(coord.offsetBy(0, -1, 0))) {
        const std::optional<float3> wall = wall_velocity(coord.offsetBy(0, -1, 0));
        value.y() = wall ? std::max(value.y(), wall->y) : std::max(value.y(), 0.0f);
      }
      if (mask.is_solid(coord.offsetBy(0, 0, 1))) {
        const std::optional<float3> wall = wall_velocity(coord.offsetBy(0, 0, 1));
        value.z() = wall ? std::min(value.z(), wall->z) : std::min(value.z(), 0.0f);
      }
      if (mask.is_solid(coord.offsetBy(0, 0, -1))) {
        const std::optional<float3> wall = wall_velocity(coord.offsetBy(0, 0, -1));
        value.z() = wall ? std::max(value.z(), wall->z) : std::max(value.z(), 0.0f);
      }
      velocity.setValue(coord, value);
    }
  });
}

/** \} */

/** \name Pressure projection
 * \{ */

/**
 * Makes the velocity divergence free on the active region.
 *
 * The divergence uses forward differences and the gradient backward differences, so the composite
 * operator is the plain 7 point Laplacian (widened to 2h) with no odd/even decoupling, i.e. no
 * checkerboard pressure. The region boundary is a free surface (`p = 0`), which is what openvdb's
 * Poisson solver assumes for voxels that are not in the domain.
 *
 * The domain is a dense box, so the index of every cell lives in a flat array: the inner loop of the
 * CG then does six array reads instead of six tree lookups, which is where most of the solve went.
 * The previous frame's pressure is used as the initial guess (a plume evolves slowly, so this cuts
 * the iteration count by half or more).
 */
void project_velocity(State &state,
                      const Settings &settings,
                      Stats &r_stats,
                      const SolidMask &solids)
{
  if (state.velocity->activeVoxelCount() == 0 || state.density->activeVoxelCount() == 0) {
    return;
  }
  const double voxel = double(state.voxel_size());
  const openvdb::Vec3fGrid::ConstAccessor velocity = state.velocity->getConstAccessor();

  /* Solve the pressure on a *band* around the smoke, not on the smoke itself: a free surface
   * boundary at the edge of the smoke would force the velocity to zero there and stop a buoyant
   * puff from moving at all. A few layers of the extrapolated air give the return flow room. */
  const openvdb::FloatTree domain = smoke_tree(state, solver_radius(settings));
  openvdb::CoordBBox box;
  domain.evalActiveVoxelBoundingBox(box);
  if (box.empty()) {
    return;
  }

  static const Coord offsets[6] = {
      Coord(1, 0, 0),
      Coord(-1, 0, 0),
      Coord(0, 1, 0),
      Coord(0, -1, 0),
      Coord(0, 0, 1),
      Coord(0, 0, -1),
  };

  /* A flat index for every cell of the box (-1 outside the domain), so the matrix-vector product is
   * one array read per neighbour. */
  const Coord bmin = box.min();
  const Coord bmax = box.max();
  const Coord dims(bmax.x() - bmin.x() + 1, bmax.y() - bmin.y() + 1, bmax.z() - bmin.z() + 1);
  const int64_t cells = int64_t(dims.x()) * int64_t(dims.y()) * int64_t(dims.z());
  Array<int> cell_index(cells, -1);
  Array<int> unknown_index(cells, -1);

  auto flat = [&](const Coord &coord) {
    return int64_t(coord.x() - bmin.x()) +
           int64_t(dims.x()) * (int64_t(coord.y() - bmin.y()) +
                                int64_t(dims.y()) * int64_t(coord.z() - bmin.z()));
  };
  auto in_box = [&](const Coord &coord) {
    return coord.x() >= bmin.x() && coord.x() <= bmax.x() && coord.y() >= bmin.y() &&
           coord.y() <= bmax.y() && coord.z() >= bmin.z() && coord.z() <= bmax.z();
  };
  /* -1 for a cell that is outside the box, outside the domain or solid. */
  auto index_at = [&](const Coord &coord) { return in_box(coord) ? cell_index[flat(coord)] : -1; };

  Vector<Coord> coords;
  coords.reserve(int(domain.activeVoxelCount()));
  for (auto iter = domain.cbeginValueOn(); iter; ++iter) {
    if (solids.is_solid(iter.getCoord())) {
      /* A collider is a wall: it is not part of the fluid, so the pressure is zero there and the
       * cells next to it see the same free surface a domain boundary would give them. */
      continue;
    }
    cell_index[flat(iter.getCoord())] = int(coords.size());
    coords.append(iter.getCoord());
  }
  if (coords.is_empty()) {
    return;
  }

  /* The velocity is read three times per constrained voxel (the right hand side, the gradient and the
   * residual check). Over the box it is one dense array, so those become array reads as well. */
  Array<openvdb::Vec3f> velocity_dense(cells, openvdb::Vec3f(0.0f));
  for (const Coord &coord : coords) {
    velocity_dense[flat(coord)] = velocity.getValue(coord);
  }

  /* The divergence uses forward differences, so a cell can only be made divergence free when its
   * three forward neighbours exist. Using the boundary cells too is what made the system
   * inconsistent (their row picks up the opposite sign, see M66), so they are treated as the open
   * boundary instead: they keep their divergence and the smoke can leave through them. */
  Vector<Coord> unknowns;
  unknowns.reserve(coords.size());
  for (const Coord &coord : coords) {
    if (index_at(coord.offsetBy(1, 0, 0)) < 0 || index_at(coord.offsetBy(0, 1, 0)) < 0 ||
        index_at(coord.offsetBy(0, 0, 1)) < 0)
    {
      continue;
    }
    unknown_index[flat(coord)] = int(unknowns.size());
    unknowns.append(coord);
  }
  const int count = int(unknowns.size());
  if (count == 0) {
    return;
  }
  auto unknown_at = [&](const Coord &coord) {
    return in_box(coord) ? unknown_index[flat(coord)] : -1;
  };

  /* Outside the band the velocity counts as zero: that is the same free surface the pressure has. */
  auto velocity_at = [&](const Coord &coord) {
    const int i = index_at(coord);
    return i >= 0 ? velocity_dense[flat(coord)] : openvdb::Vec3f(0.0f);
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
  const double divergence_sq = parallel_sum(count, [&](const int64_t begin, const int64_t end) {
    double sum = 0.0;
    for (int64_t i = begin; i < end; i++) {
      const double divergence = divergence_at(velocity_at, unknowns[int(i)]);
      rhs[i] = divergence * voxel * voxel;
      sum += divergence * divergence;
    }
    return sum;
  });
  r_stats.divergence_before = float(std::sqrt(divergence_sq / double(count)));

  /* The 7 point Laplacian with the boundary voxels pinned to zero pressure: symmetric positive
   * definite, so the whole pressure field is determined. */
  auto apply_laplacian = [&](const Span<double> x, MutableSpan<double> r) {
    threading::parallel_for(IndexRange(count), PARALLEL_CHUNK, [&](const IndexRange range) {
      for (const int i : range) {
        double sum = -6.0 * x[i];
        const Coord &coord = unknowns[i];
        for (const Coord &offset : offsets) {
          const int j = unknown_at(coord + offset);
          if (j >= 0) {
            sum += x[j];
          }
        }
        r[i] = sum;
      }
    });
  };
  auto dot = [](const Span<double> a, const Span<double> b) {
    const int count = int(a.size());
    return parallel_sum(count, [&](const int64_t begin, const int64_t end) {
      double sum = 0.0;
      for (int64_t i = begin; i < end; i++) {
        sum += a[int(i)] * b[int(i)];
      }
      return sum;
    });
  };

  /* Jacobi preconditioned conjugate gradient. The diagonal is the constant -6, so the preconditioner
   * is a single multiply. */
  Array<double> pressure(count);
  Array<double> residual(count);
  Array<double> preconditioned(count);
  Array<double> direction(count);
  Array<double> product(count);

  /* Warm start: the pressure of the previous frame is a good guess for this one, because the flow
   * changes slowly compared to the solve. It is only a guess, the solve still converges to the same
   * field. */
  if (state.pressure && state.pressure->activeVoxelCount() > 0) {
    const openvdb::FloatGrid &previous_grid = *state.pressure;
    threading::parallel_for(IndexRange(count), PARALLEL_CHUNK, [&](const IndexRange range) {
      const openvdb::FloatGrid::ConstAccessor previous = previous_grid.getConstAccessor();
      for (const int i : range) {
        pressure[i] = double(previous.getValue(unknowns[i]));
      }
    });
  }
  else {
    pressure.fill(0.0);
  }

  apply_laplacian(pressure, product);
  const double rhs_norm = std::sqrt(dot(rhs, rhs));
  threading::parallel_for(IndexRange(count), PARALLEL_CHUNK, [&](const IndexRange range) {
    for (const int i : range) {
      residual[i] = rhs[i] - product[i];
    }
  });
  int iterations = 0;
  double residual_norm = std::sqrt(dot(residual, residual));
  if (rhs_norm > 1e-12 && residual_norm > 0.0) {
    const int max_iterations = std::max(1, settings.pressure_iterations);
    const double tolerance = std::max(1e-8f, settings.tolerance);
    threading::parallel_for(IndexRange(count), PARALLEL_CHUNK, [&](const IndexRange range) {
      for (const int i : range) {
        preconditioned[i] = residual[i] / 6.0;
        direction[i] = preconditioned[i];
      }
    });
    double rz = dot(residual, preconditioned);
    for (; iterations < max_iterations; iterations++) {
      apply_laplacian(direction, product);
      const double pq = dot(direction, product);
      if (std::abs(pq) < 1e-30) {
        break;
      }
      const double alpha = rz / pq;
      threading::parallel_for(IndexRange(count), PARALLEL_CHUNK, [&](const IndexRange range) {
        for (const int i : range) {
          pressure[i] += alpha * direction[i];
          residual[i] -= alpha * product[i];
        }
      });
      residual_norm = std::sqrt(dot(residual, residual));
      if (residual_norm <= tolerance * rhs_norm) {
        iterations++;
        break;
      }
      threading::parallel_for(IndexRange(count), PARALLEL_CHUNK, [&](const IndexRange range) {
        for (const int i : range) {
          preconditioned[i] = residual[i] / 6.0;
        }
      });
      const double rz_next = dot(residual, preconditioned);
      const double beta = rz_next / rz;
      threading::parallel_for(IndexRange(count), PARALLEL_CHUNK, [&](const IndexRange range) {
        for (const int i : range) {
          direction[i] = preconditioned[i] + beta * direction[i];
        }
      });
      rz = rz_next;
    }
    r_stats.residual = float(residual_norm / rhs_norm);
  }
  else {
    r_stats.residual = rhs_norm > 1e-12 ? float(residual_norm / rhs_norm) : 0.0f;
  }
  r_stats.pressure_iterations = iterations;

  /* `u -= grad p` with backward differences on every voxel of the band (the pressure of a voxel that
   * is not part of the system is zero, which keeps the field smooth towards the open boundary). */
  auto pressure_at = [&](const Coord &coord) {
    const int j = unknown_at(coord);
    return j >= 0 ? pressure[j] : 0.0;
  };
  {
    const int64_t count_coords = int64_t(coords.size());
    threading::parallel_for(IndexRange(count_coords), PARALLEL_CHUNK, [&](const IndexRange range) {
      openvdb::Vec3fGrid::Accessor writer = state.velocity->getAccessor();
      for (const int64_t i : range) {
        const Coord &coord = coords[i];
        const double pressure_here = pressure_at(coord);
        const double px = (pressure_here - pressure_at(coord.offsetBy(-1, 0, 0))) / voxel;
        const double py = (pressure_here - pressure_at(coord.offsetBy(0, -1, 0))) / voxel;
        const double pz = (pressure_here - pressure_at(coord.offsetBy(0, 0, -1))) / voxel;
        const openvdb::Vec3f corrected = velocity_dense[flat(coord)] - openvdb::Vec3f(px, py, pz);
        velocity_dense[flat(coord)] = corrected;
        writer.setValue(coord, corrected);
      }
    });
  }

  /* Report the residual divergence on the cells that were actually constrained, so the projection can
   * be verified instead of trusted. */
  auto corrected_at = [&](const Coord &coord) {
    const int i = index_at(coord);
    return i >= 0 ? velocity_dense[flat(coord)] : openvdb::Vec3f(0.0f);
  };
  const double after_sq = parallel_sum(count, [&](const int64_t begin, const int64_t end) {
    double sum = 0.0;
    for (int64_t i = begin; i < end; i++) {
      const double divergence = divergence_at(corrected_at, unknowns[int(i)]);
      sum += divergence * divergence;
    }
    return sum;
  });
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
  const int radius = solver_radius(settings);
  const openvdb::FloatTree keep = smoke_tree(state, radius);
  openvdb::CoordBBox box;
  keep.evalActiveVoxelBoundingBox(box);
  if (box.empty()) {
    state.velocity->clear();
    return;
  }

  /* A flat index over the box, so the frontier search is array reads instead of tree probes. */
  const Coord bmin = box.min();
  const Coord bmax = box.max();
  const Coord dims(bmax.x() - bmin.x() + 1, bmax.y() - bmin.y() + 1, bmax.z() - bmin.z() + 1);
  const int64_t cells = int64_t(dims.x()) * int64_t(dims.y()) * int64_t(dims.z());
  Array<int32_t> cell_index(cells, -1);
  auto flat = [&](const Coord &coord) {
    return int64_t(coord.x() - bmin.x()) +
           int64_t(dims.x()) * (int64_t(coord.y() - bmin.y()) +
                                int64_t(dims.y()) * int64_t(coord.z() - bmin.z()));
  };
  auto index_at = [&](const Coord &coord) {
    return (coord.x() >= bmin.x() && coord.x() <= bmax.x() && coord.y() >= bmin.y() &&
            coord.y() <= bmax.y() && coord.z() >= bmin.z() && coord.z() <= bmax.z()) ?
               cell_index[flat(coord)] :
               -1;
  };

  Vector<Coord> coords;
  coords.reserve(int(keep.activeVoxelCount()));
  for (auto iter = keep.cbeginValueOn(); iter; ++iter) {
    cell_index[flat(iter.getCoord())] = int(coords.size());
    coords.append(iter.getCoord());
  }

  openvdb::Vec3fGrid::Ptr result = openvdb::Vec3fGrid::create(openvdb::Vec3f(0.0f));
  result->setTransform(state.velocity->transformPtr());
  openvdb::Vec3fGrid::Accessor writer = result->getAccessor();
  Array<uint8_t> filled(cells, 0);
  {
    const openvdb::Vec3fGrid::ConstAccessor reader = state.velocity->getConstAccessor();
    for (const Coord &coord : coords) {
      if (!reader.isValueOn(coord)) {
        continue;
      }
      writer.setValue(coord, reader.getValue(coord));
      filled[flat(coord)] = 1;
    }
  }

  static const Coord offsets[6] = {
      Coord(1, 0, 0),
      Coord(-1, 0, 0),
      Coord(0, 1, 0),
      Coord(0, -1, 0),
      Coord(0, 0, 1),
      Coord(0, 0, -1),
  };

  /* Fill the empty cells from their known neighbours, one layer per pass. Only the *frontier* of each
   * layer is visited, instead of walking the whole band once per layer like before: that is the same
   * result for a fraction of the work. */
  Vector<Coord> frontier;
  Vector<Coord> next;
  Array<uint8_t> queued(cells, 0);
  for (const Coord &coord : coords) {
    if (filled[flat(coord)]) {
      continue;
    }
    for (const Coord &offset : offsets) {
      const int j = index_at(coord + offset);
      if (j >= 0 && filled[flat(coord + offset)]) {
        frontier.append(coord);
        queued[flat(coord)] = 1;
        break;
      }
    }
  }

  const int passes = std::max(0, settings.extrapolate);
  for (int pass = 0; pass < passes && !frontier.is_empty(); pass++) {
    for (const Coord &coord : frontier) {
      openvdb::Vec3f sum(0.0f);
      int count = 0;
      for (const Coord &offset : offsets) {
        const int j = index_at(coord + offset);
        if (j >= 0 && filled[flat(coord + offset)]) {
          sum += writer.getValue(coord + offset);
          count++;
        }
      }
      if (count > 0) {
        writer.setValue(coord, sum / float(count));
        filled[flat(coord)] = 1;
      }
    }
    next.clear();
    for (const Coord &coord : frontier) {
      for (const Coord &offset : offsets) {
        const Coord neighbour = coord + offset;
        if (index_at(neighbour) < 0) {
          continue;
        }
        const int64_t i = flat(neighbour);
        if (filled[i] || queued[i]) {
          continue;
        }
        queued[i] = 1;
        next.append(neighbour);
      }
    }
    frontier = next;
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
  openvdb::FloatGrid::Accessor flame = state.flame->getAccessor();

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
        const float heat = burned * settings.heat_release;
        temperature.setValue(coord, temperature.getValue(coord) + heat);
        density.setValue(coord, density.getValue(coord) + burned * settings.smoke_per_burn);
        if (heat > 0.0f) {
          /* `flame` is the *rate* of the reaction, not the temperature: it marks the voxels where the
           * fuel actually burns, so the material can glow there and leave the smoke above it dark. A
           * voxel that burns in several substeps keeps the strongest rate of the frame. */
          flame.setValue(coord, std::max(flame.getValue(coord), heat / dt));
        }
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

/**
 * Removes the weak reactions from the render only `flame` field.
 *
 * `flame` is filled by the combustion step with the *rate* at which the fuel burns (in K/s), so it
 * marks the voxels where the fire actually is, and the smoke that the fire pushed up stays dark.
 * `Flame Threshold` drops the smouldering tail of the reaction so thin wisps do not glow.
 */
void filter_flame(State &state, const Settings &settings)
{
  const float threshold = std::max(0.0f, settings.flame_threshold);
  if (threshold > 0.0f) {
    for (auto iter = state.flame->beginValueOn(); iter; ++iter) {
      iter.setValue(std::max(0.0f, iter.getValue() - threshold));
    }
  }

  if (std::getenv("BLENDER_IGNIS_DEBUG") != nullptr) {
    float max_flame = 0.0f;
    float z_min = 1e30f;
    float z_max = -1e30f;
    const openvdb::math::Transform &transform = state.flame->transform();
    for (auto iter = state.flame->cbeginValueOn(); iter; ++iter) {
      const float value = iter.getValue();
      if (value <= 0.0f) {
        continue;
      }
      max_flame = std::max(max_flame, value);
      const openvdb::Vec3d position = transform.indexToWorld(iter.getCoord().asVec3d());
      z_min = math::min(z_min, float(position.z()));
      z_max = math::max(z_max, float(position.z()));
    }
    printf("IGNIS_DEBUG flame: %lld voxels max %.3f z %.3f..%.3f (threshold %.3f)\n",
           (long long)state.flame->activeVoxelCount(),
           double(max_flame),
           max_flame > 0.0f ? double(z_min) : 0.0,
           max_flame > 0.0f ? double(z_max) : 0.0,
           double(threshold));
  }
}

/**
 * Removes the voxels of the adaptive domain whose selected field is below `Domain Threshold`.
 *
 * Every field of a failing voxel is deactivated, so the next frame's smoke tree, pressure domain,
 * velocity band and rendering domain all shrink. `abs()` is used so that a negative temperature
 * (cold smoke) is pruned by magnitude as well.
 */
void apply_domain_mask(State &state, const Settings &settings, Stats &r_stats)
{
  r_stats.domain_removed = 0;
  if (!settings.use_domain_mask) {
    return;
  }
  if (!state.density || !state.temperature || !state.fuel || !state.velocity) {
    return;
  }

  const openvdb::FloatGrid *field = nullptr;
  switch (settings.domain_field) {
    case DomainField::Density:
      field = state.density.get();
      break;
    case DomainField::Temperature:
      field = state.temperature.get();
      break;
    case DomainField::Fuel:
      field = state.fuel.get();
      break;
    case DomainField::Flame:
      field = state.flame.get();
      break;
  }
  if (field == nullptr) {
    /* An old state may not have the selected grid. Density always exists, so use it instead of
     * silently doing nothing. */
    field = state.density.get();
  }

  /* The adaptive domain is the active set of `density`; the other fields may be sparse or missing
   * on some of its voxels (e.g. `flame` is empty before the fuel burns), so sample the selected
   * field on every density voxel instead of iterating the selected field itself. A missing value is
   * the grid background (0), which is what the threshold should see. */
  const openvdb::FloatGrid::ConstAccessor values = field->getConstAccessor();
  const float threshold = std::max(0.0f, settings.domain_threshold);
  Vector<Coord> removed;
  for (auto iter = state.density->cbeginValueOn(); iter; ++iter) {
    if (std::abs(values.getValue(iter.getCoord())) < threshold) {
      removed.append(iter.getCoord());
    }
  }
  if (removed.is_empty()) {
    return;
  }

  auto &density = state.density->tree();
  auto &temperature = state.temperature->tree();
  auto &fuel = state.fuel->tree();
  auto *flame = state.flame ? &state.flame->tree() : nullptr;
  auto *pressure = state.pressure ? &state.pressure->tree() : nullptr;
  auto &velocity = state.velocity->tree();
  for (const Coord &coord : removed) {
    density.setValueOff(coord, 0.0f);
    temperature.setValueOff(coord, 0.0f);
    fuel.setValueOff(coord, 0.0f);
    if (flame != nullptr) {
      flame->setValueOff(coord, 0.0f);
    }
    if (pressure != nullptr) {
      pressure->setValueOff(coord, 0.0f);
    }
    velocity.setValueOff(coord, openvdb::Vec3f(0.0f));
  }
  r_stats.domain_removed = int(removed.size());
  if (std::getenv("BLENDER_IGNIS_DEBUG") != nullptr) {
    printf("IGNIS_DEBUG domain mask: field=%d threshold=%.4g removed=%d\n",
           int(settings.domain_field),
           double(settings.domain_threshold),
           r_stats.domain_removed);
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

}  // namespace

/**
 * Converts the sparse state to a dense box, runs the device solver and converts the result back.
 *
 * The conversion (reading the volume, sampling the emitters at the state's voxel centers, the
 * collider shell) is the part that is inherently CPU work; every pass that scales with the number
 * of voxels runs on the device. Returns null with \a r_reason when the device path cannot run.
 */
static Volume *solve_frame_device(const Volume *previous,
                                  const Span<const Volume *> sources,
                                  const Span<const Collider> colliders,
                                  const Settings &settings,
                                  const float delta_time,
                                  Stats &r_stats,
                                  std::string &r_reason)
{
  namespace ha = geometry::ha;

  if (!ha::ignis_device_available()) {
    r_reason = "no GPU backend";
    return nullptr;
  }
  const float voxel_size = std::max(1e-5f, settings.voxel_size);

  State state;
  if (previous != nullptr) {
    state = read_state(*previous);
  }
  const bool resolution_ok = state.valid() && state.velocity != nullptr &&
                             std::abs(state.voxel_size() - voxel_size) <= 1e-6f * voxel_size;
  if (!resolution_ok) {
    init_state(state, voxel_size);
    r_stats.resized = true;
  }
  const openvdb::math::Transform &transform = state.density->transform();
  const float dt = std::max(0.0f, delta_time);

  int substeps = std::clamp(settings.substeps, 1, 64);
  bool cfl_limited = false;
  if (dt > 0.0f) {
    float max_speed = 0.0f;
    for (auto iter = state.velocity->cbeginValueOn(); iter; ++iter) {
      max_speed = std::max(max_speed, iter.getValue().length());
    }
    if (max_speed > 0.0f) {
      const int required = int(std::ceil(dt * max_speed / (std::max(0.1f, settings.cfl) *
                                                          voxel_size)));
      if (required > substeps) {
        const int limit = std::max(1, settings.max_substeps);
        substeps = std::min(required, limit);
        cfl_limited = required > limit;
      }
    }
  }
  else {
    substeps = 0;
  }

  /* The box has to contain every voxel the CPU solver would touch: the smoke, the emitters, and the
   * air the smoke can move into this frame plus the velocity band radius. */
  openvdb::CoordBBox box;
  bool box_valid = false;
  if (state.density->activeVoxelCount() > 0) {
    box = state.density->evalActiveVoxelBoundingBox();
    box_valid = !box.empty();
  }
  for (const Volume *source : sources) {
    const bke::VolumeGridData *data = BKE_volume_grid_find(source, grids::density);
    if (data == nullptr) {
      continue;
    }
    bke::VolumeTreeAccessToken token;
    const openvdb::GridBase &base = data->grid(token);
    if (base.type() != openvdb::FloatGrid::gridType()) {
      continue;
    }
    const openvdb::FloatGrid &grid = static_cast<const openvdb::FloatGrid &>(base);
    const openvdb::CoordBBox source_box = grid.evalActiveVoxelBoundingBox();
    if (source_box.empty()) {
      continue;
    }
    const openvdb::Vec3d min_world = grid.transform().indexToWorld(source_box.min());
    const openvdb::Vec3d max_world = grid.transform().indexToWorld(source_box.max());
    const openvdb::CoordBBox in_state(
        transform.worldToIndexCellCentered(min_world),
        transform.worldToIndexCellCentered(max_world));
    if (!box_valid) {
      box = in_state;
      box_valid = true;
    }
    else {
      box.expand(in_state);
    }
  }
  if (!box_valid) {
    /* Nothing to simulate yet: hand out an empty volume, like the CPU path does. */
    State empty = state;
    empty.flame = create_float_grid(empty.density->transformPtr());
    Volume *volume = BKE_id_new_nomain<Volume>(nullptr);
    write_state(*volume, empty);
    r_stats.device = true;
    r_stats.substeps_used = 0;
    return volume;
  }

  const int radius = solver_radius(settings);
  int margin = radius + 2 + int(std::ceil(std::max(1.0f, settings.cfl))) * std::max(1, substeps) + 2;
  margin = std::min(margin, 64);
  box.expand(margin);

  if (settings.use_bounds) {
    const float3 lo = math::min(settings.bounds_min, settings.bounds_max);
    const float3 hi = math::max(settings.bounds_min, settings.bounds_max);
    const int pad = radius + 3;
    const openvdb::CoordBBox bounds(
        transform.worldToIndexCellCentered(openvdb::Vec3d(lo.x, lo.y, lo.z)),
        transform.worldToIndexCellCentered(openvdb::Vec3d(hi.x, hi.y, hi.z)));
    openvdb::CoordBBox padded = bounds;
    padded.expand(pad);
    box.intersect(padded);
    if (box.empty()) {
      r_reason = "the bounds do not overlap the simulation";
      return nullptr;
    }
  }

  const openvdb::Coord bmin = box.min();
  const openvdb::Coord bmax = box.max();
  const int64_t dim_x = int64_t(bmax.x()) - bmin.x() + 1;
  const int64_t dim_y = int64_t(bmax.y()) - bmin.y() + 1;
  const int64_t dim_z = int64_t(bmax.z()) - bmin.z() + 1;
  const int64_t cells = dim_x * dim_y * dim_z;
  /* No artificial voxel cap: the device allocation either succeeds or reports its own failure
   * (and the zone VRAM budget can reject it before the driver runs out of memory). */

  Array<float> scalars;
  Array<float> velocity;
  Array<float> pressure;
  Array<int32_t> mask;
  Array<int32_t> current_mask;
  Array<int32_t> carve_mask;
  Array<float> collider_velocities;
  Array<float> source_values;
  scalars.reinitialize(int(cells * 4));
  velocity.reinitialize(int(cells * 3));
  pressure.reinitialize(int(cells));
  mask.reinitialize(int(cells));
  current_mask.reinitialize(int(cells));
  carve_mask.reinitialize(int(cells));
  collider_velocities.reinitialize(int(colliders.size()) * 3);
  source_values.reinitialize(int(cells * 7));
  scalars.fill(0.0f);
  velocity.fill(0.0f);
  pressure.fill(0.0f);
  mask.fill(0);
  current_mask.fill(0);
  carve_mask.fill(0);
  collider_velocities.fill(0.0f);
  source_values.fill(0.0f);

  auto inside = [&](const openvdb::Coord &coord) {
    return coord.x() >= bmin.x() && coord.x() <= bmax.x() && coord.y() >= bmin.y() &&
           coord.y() <= bmax.y() && coord.z() >= bmin.z() && coord.z() <= bmax.z();
  };
  auto flat = [&](const openvdb::Coord &coord) {
    return int64_t(coord.x() - bmin.x()) +
           dim_x * (int64_t(coord.y() - bmin.y()) + dim_y * int64_t(coord.z() - bmin.z()));
  };

  /* Sparse to dense: only the active voxels are visited, everything else stays zero. */
  if (state.density) {
    for (auto iter = state.density->cbeginValueOn(); iter; ++iter) {
      if (inside(iter.getCoord())) {
        scalars[flat(iter.getCoord()) * 4 + 0] = iter.getValue();
      }
    }
  }
  if (state.temperature) {
    for (auto iter = state.temperature->cbeginValueOn(); iter; ++iter) {
      if (inside(iter.getCoord())) {
        scalars[flat(iter.getCoord()) * 4 + 1] = iter.getValue();
      }
    }
  }
  if (state.fuel) {
    for (auto iter = state.fuel->cbeginValueOn(); iter; ++iter) {
      if (inside(iter.getCoord())) {
        scalars[flat(iter.getCoord()) * 4 + 2] = iter.getValue();
      }
    }
  }
  if (state.velocity) {
    for (auto iter = state.velocity->cbeginValueOn(); iter; ++iter) {
      if (!inside(iter.getCoord())) {
        continue;
      }
      const int64_t i = flat(iter.getCoord()) * 3;
      velocity[i + 0] = iter.getValue().x();
      velocity[i + 1] = iter.getValue().y();
      velocity[i + 2] = iter.getValue().z();
    }
  }
  if (state.pressure) {
    for (auto iter = state.pressure->cbeginValueOn(); iter; ++iter) {
      if (inside(iter.getCoord())) {
        pressure[flat(iter.getCoord())] = iter.getValue();
      }
    }
  }

  /* Collider shell (the CPU path rebuilds it once per frame as well). */
  int solid_voxels = 0;
  if (!colliders.is_empty()) {
    Array<Collider> collider_data;
    collider_data.reinitialize(colliders.size());
    for (const int i : colliders.index_range()) {
      collider_data[i] = colliders[i];
    }
    const ColliderMasks masks = build_collider_masks(transform,
                                                     collider_data.as_mutable_span(),
                                                     box,
                                                     dt);
    solid_voxels = int(masks.filled.count());
    for (const int i : collider_data.index_range()) {
      const float3 &velocity = collider_data[i].velocity;
      collider_velocities[i * 3 + 0] = velocity.x;
      collider_velocities[i * 3 + 1] = velocity.y;
      collider_velocities[i * 3 + 2] = velocity.z;
    }
    for (int64_t z = 0; z < dim_z; z++) {
      for (int64_t y = 0; y < dim_y; y++) {
        for (int64_t x = 0; x < dim_x; x++) {
          const openvdb::Coord coord(int(bmin.x() + x), int(bmin.y() + y), int(bmin.z() + z));
          const int64_t index = x + dim_x * (y + dim_y * z);
          if (masks.filled.id(coord) != 0) {
            mask[index] = ha::IGNIS_MASK_SOLID;
          }
          const int8_t current_id = masks.current.id(coord);
          if (current_id != 0) {
            current_mask[index] = (int32_t(current_id) << 8) | ha::IGNIS_MASK_SOLID;
          }
          if (masks.carve.id(coord) != 0) {
            carve_mask[index] = ha::IGNIS_MASK_SOLID;
          }
        }
      }
    }
  }

  /* Emitters: sampled once at the voxel centers of the state, exactly like `add_source_volume`. */
  for (const Volume *source : sources) {
    const bke::VolumeGridData *density_data = BKE_volume_grid_find(source, grids::density);
    if (density_data == nullptr) {
      continue;
    }
    bke::VolumeTreeAccessToken density_token;
    const openvdb::GridBase &density_base = density_data->grid(density_token);
    if (density_base.type() != openvdb::FloatGrid::gridType()) {
      continue;
    }
    const openvdb::FloatGrid &source_density = static_cast<const openvdb::FloatGrid &>(density_base);
    if (source_density.activeVoxelCount() == 0) {
      continue;
    }
    const openvdb::FloatGrid *source_temperature = nullptr;
    const openvdb::FloatGrid *source_fuel = nullptr;
    const openvdb::Vec3fGrid *source_velocity = nullptr;
    bke::VolumeTreeAccessToken temperature_token;
    bke::VolumeTreeAccessToken fuel_token;
    bke::VolumeTreeAccessToken velocity_token;
    if (const bke::VolumeGridData *data = BKE_volume_grid_find(source, grids::temperature)) {
      const openvdb::GridBase &base = data->grid(temperature_token);
      if (base.type() == openvdb::FloatGrid::gridType()) {
        source_temperature = static_cast<const openvdb::FloatGrid *>(&base);
      }
    }
    if (const bke::VolumeGridData *data = BKE_volume_grid_find(source, grids::fuel)) {
      const openvdb::GridBase &base = data->grid(fuel_token);
      if (base.type() == openvdb::FloatGrid::gridType()) {
        source_fuel = static_cast<const openvdb::FloatGrid *>(&base);
      }
    }
    if (const bke::VolumeGridData *data = BKE_volume_grid_find(source, grids::velocity)) {
      const openvdb::GridBase &base = data->grid(velocity_token);
      if (base.type() == openvdb::Vec3fGrid::gridType()) {
        source_velocity = static_cast<const openvdb::Vec3fGrid *>(&base);
      }
    }

    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> density_sampler(
        source_density);
    std::optional<openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>>
        temperature_sampler;
    std::optional<openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler>>
        fuel_sampler;
    std::optional<openvdb::tools::GridSampler<openvdb::Vec3fGrid, openvdb::tools::BoxSampler>>
        velocity_sampler;
    if (source_temperature != nullptr) {
      temperature_sampler.emplace(*source_temperature);
    }
    if (source_fuel != nullptr) {
      fuel_sampler.emplace(*source_fuel);
    }
    if (source_velocity != nullptr) {
      velocity_sampler.emplace(*source_velocity);
    }

    const openvdb::CoordBBox source_box = source_density.evalActiveVoxelBoundingBox();
    const openvdb::Vec3d min_world = source_density.transform().indexToWorld(source_box.min());
    const openvdb::Vec3d max_world = source_density.transform().indexToWorld(source_box.max());
    const openvdb::CoordBBox state_box = transform.worldToIndexCellCentered(
        openvdb::BBoxd(min_world, max_world));
    const int x0 = std::max(bmin.x(), state_box.min().x());
    const int y0 = std::max(bmin.y(), state_box.min().y());
    const int z0 = std::max(bmin.z(), state_box.min().z());
    const int x1 = std::min(bmax.x(), state_box.max().x());
    const int y1 = std::min(bmax.y(), state_box.max().y());
    const int z1 = std::min(bmax.z(), state_box.max().z());
    for (int k = z0; k <= z1; k++) {
      for (int j = y0; j <= y1; j++) {
        for (int i = x0; i <= x1; i++) {
          const openvdb::Coord coord(i, j, k);
          const openvdb::Vec3d world = transform.indexToWorld(coord);
          const float source_value = density_sampler.wsSample(world);
          if (source_value <= 0.0f) {
            continue;
          }
          const int64_t base = flat(coord) * 7;
          source_values[base + 0] = std::max(source_values[base + 0], source_value);
          if (temperature_sampler.has_value()) {
            const float value = temperature_sampler->wsSample(world);
            if (value != 0.0f) {
              float current = source_values[base + 1];
              source_values[base + 1] = value > 0.0f ? std::max(current, value) :
                                                       std::min(current, value);
            }
          }
          if (fuel_sampler.has_value()) {
            const float value = fuel_sampler->wsSample(world);
            if (value > 0.0f) {
              source_values[base + 2] = std::max(source_values[base + 2], value);
            }
          }
          if (velocity_sampler.has_value()) {
            const openvdb::Vec3f incoming = velocity_sampler->wsSample(world);
            if (incoming.lengthSqr() > 0.0f) {
              /* Store the source velocity itself; the blend weight is the sampled density and is
               * applied by `KERNEL_SOURCE_MERGE` every substep, like `add_source_volume` does. */
              source_values[base + 3] = incoming.x();
              source_values[base + 4] = incoming.y();
              source_values[base + 5] = incoming.z();
            }
          }
        }
      }
    }
  }

  ha::IgnisDeviceFrame frame;
  frame.index_min = int3(int(bmin.x()), int(bmin.y()), int(bmin.z()));
  frame.dims = int3(int(dim_x), int(dim_y), int(dim_z));
  frame.voxel_size = voxel_size;
  frame.delta_time = dt;
  frame.substeps = substeps;
  frame.solver_radius = radius;
  frame.extrapolate = std::max(0, settings.extrapolate);
  frame.settings = settings;
  frame.scalars = scalars.as_mutable_span();
  frame.velocity = velocity.as_mutable_span();
  frame.pressure = pressure.as_mutable_span();
  frame.mask = mask.as_mutable_span();
  frame.current_mask = current_mask.as_mutable_span();
  frame.carve_mask = carve_mask.as_mutable_span();
  frame.collider_velocities = Span<float>(collider_velocities.data(), int64_t(collider_velocities.size()));
  frame.source = source_values.as_mutable_span();

  ha::IgnisDeviceResult device;
  const bool ran = ha::ignis_device_solve(frame, device);
  if (!ran) {
    r_reason = device.reason.empty() ? "device solve failed" : device.reason;
    return nullptr;
  }

  /* Dense to sparse. Values below the tree prune thresholds stay inactive, which is what the CPU
   * path ends up with as well. */
  State output;
  output.density = create_float_grid(state.density->transformPtr());
  output.temperature = create_float_grid(state.density->transformPtr());
  output.fuel = create_float_grid(state.density->transformPtr());
  output.flame = create_float_grid(state.density->transformPtr());
  output.pressure = create_float_grid(state.density->transformPtr());
  output.velocity = openvdb::Vec3fGrid::create(openvdb::Vec3f(0.0f));
  output.velocity->setTransform(state.density->transformPtr());

  openvdb::FloatGrid::Accessor density_writer = output.density->getAccessor();
  openvdb::FloatGrid::Accessor temperature_writer = output.temperature->getAccessor();
  openvdb::FloatGrid::Accessor fuel_writer = output.fuel->getAccessor();
  openvdb::FloatGrid::Accessor flame_writer = output.flame->getAccessor();
  openvdb::FloatGrid::Accessor pressure_writer = output.pressure->getAccessor();
  openvdb::Vec3fGrid::Accessor velocity_writer = output.velocity->getAccessor();
  for (int64_t z = 0; z < dim_z; z++) {
    for (int64_t y = 0; y < dim_y; y++) {
      for (int64_t x = 0; x < dim_x; x++) {
        const int64_t i = x + dim_x * (y + dim_y * z);
        const openvdb::Coord coord(int(bmin.x() + x), int(bmin.y() + y), int(bmin.z() + z));
        const float density = scalars[i * 4 + 0];
        const float temperature = scalars[i * 4 + 1];
        const float fuel = scalars[i * 4 + 2];
        const float flame = scalars[i * 4 + 3];
        if (std::abs(density) > 1e-6f) {
          density_writer.setValue(coord, density);
        }
        if (std::abs(temperature) > 1e-6f) {
          temperature_writer.setValue(coord, temperature);
        }
        if (std::abs(fuel) > 1e-6f) {
          fuel_writer.setValue(coord, fuel);
        }
        if (flame > 1e-6f) {
          flame_writer.setValue(coord, flame);
        }
        const float vx = velocity[i * 3 + 0];
        const float vy = velocity[i * 3 + 1];
        const float vz = velocity[i * 3 + 2];
        if (vx * vx + vy * vy + vz * vz > 1e-12f) {
          velocity_writer.setValue(coord, openvdb::Vec3f(vx, vy, vz));
        }
        if (std::abs(pressure[i]) > 1e-9f) {
          pressure_writer.setValue(coord, pressure[i]);
        }
      }
    }
  }

  Volume *volume = BKE_id_new_nomain<Volume>(nullptr);
  write_state(*volume, output);

  r_stats.device = true;
  r_stats.device_passes = device.passes;
  r_stats.active_voxels = device.active_voxels;
  r_stats.substeps_used = substeps;
  r_stats.pressure_iterations = device.pressure_iterations;
  r_stats.residual = device.residual;
  r_stats.divergence_before = device.divergence_before;
  r_stats.divergence_after = device.divergence_after;
  r_stats.solid_voxels = solid_voxels;
  r_stats.domain_removed = device.domain_removed;
  r_stats.cfl_limited = cfl_limited;
  r_stats.milliseconds = device.milliseconds;
  return volume;
}

Volume *solve_frame(const Volume *previous,
                    const Span<const Volume *> sources,
                    const Span<const Collider> colliders,
                    const Settings &settings,
                    const float delta_time,
                    Stats &r_stats,
                    const int backend_request)
{
  r_stats = Stats{};

  /* The device path is opt in through the `HW Accelerate` zone (`Auto` / `Strict`) or the
   * `BLENDER_IGNIS_GPU=1` test override. `Force CPU` (3) and the outside of a zone (0) stay on the
   * CPU, which also keeps every old scene bit for bit identical. */
  const char *gpu_env = std::getenv("BLENDER_IGNIS_GPU");
  const bool force_gpu = gpu_env != nullptr && std::string(gpu_env) == "1";
  const bool force_cpu = gpu_env != nullptr && std::string(gpu_env) == "0";
  const bool device_requested = !force_cpu && (force_gpu || backend_request == 1 ||
                                               backend_request == 2);
  if (device_requested) {
    Stats device_stats;
    std::string reason;
    if (Volume *volume = solve_frame_device(previous,
                                            sources,
                                            colliders,
                                            settings,
                                            delta_time,
                                            device_stats,
                                            reason)) {
      r_stats = device_stats;
      return volume;
    }
    r_stats.device_fallback_reason = reason;
    if (std::getenv("BLENDER_IGNIS_HA_DEBUG") != nullptr) {
      fprintf(stderr, "IGNIS_HA fallback: %s\n", reason.c_str());
      fflush(stderr);
    }
  }

  r_stats.sources = int(sources.size());
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
  Array<Collider> collider_data;
  collider_data.reinitialize(colliders.size());
  for (const int i : colliders.index_range()) {
    collider_data[i] = colliders[i];
  }
  const MutableSpan<Collider> dynamic_colliders = collider_data.as_mutable_span();
  if (dt <= 0.0f) {
    /* First frame or a paused frame: only show the sources, do not integrate. */
    for (const Volume *source : sources) {
      add_source_volume(state, *source);
    }
    ensure_velocity_topology(state);
    state.flame = create_float_grid(state.density->transformPtr());
    filter_flame(state, settings);
    if (!colliders.is_empty()) {
      const ColliderMasks masks = build_collider_masks(*state.density->transformPtr(),
                                                      dynamic_colliders,
                                                      solid_region(state, settings),
                                                      0.0f);
      r_stats.solid_voxels = int(masks.filled.count());
      carve_solids(state, masks.carve);
    }
    /* Do not mask the seeding / paused frame: it only shows the sources, and a field like `flame`
     * is still empty there. The next simulated frame applies the mask. */
    state.density->tree().prune(1e-5f);
    r_stats.active_voxels = int(state.density->activeVoxelCount());
    Volume *volume = BKE_id_new_nomain<Volume>(nullptr);
    write_state(*volume, state);
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
  /* The flame of a frame is the reaction of that frame only, so it starts empty. */
  state.flame = create_float_grid(state.density->transformPtr());

  /* The wall of the colliders is rebuilt once per frame. The region is the smoke of the previous
   * frame grown by the room the smoke can reach, so a collider outside it costs nothing. */
  ColliderMasks collider_masks;
  bool solids_built = colliders.is_empty();

  double ms_sources = 0.0;
  double ms_colliders = 0.0;
  double ms_forces = 0.0;
  double ms_velocity = 0.0;
  double ms_projection = 0.0;
  double ms_scalars = 0.0;
  double ms_combustion = 0.0;
  double ms_solids = 0.0;

  for ([[maybe_unused]] const int substep : IndexRange(substeps)) {
    if (!sources.is_empty()) {
      ScopedTimer timer(ms_sources);
      for (const Volume *source : sources) {
        add_source_volume(state, *source);
      }
    }
    ensure_velocity_topology(state);
    if (!solids_built) {
      ScopedTimer timer(ms_colliders);
      collider_masks = build_collider_masks(*state.density->transformPtr(),
                                            dynamic_colliders,
                                            solid_region(state, settings),
                                            dt);
      solids_built = true;
      r_stats.solid_voxels = int(collider_masks.filled.count());
    }
    {
      ScopedTimer timer(ms_forces);
      apply_forces(state, settings, step_dt);
      apply_vorticity(state, settings, step_dt);
    }
    {
      ScopedTimer timer(ms_velocity);
      if (state.velocity->activeVoxelCount() > 0) {
        Advection advection(*state.velocity);
        configure_advection(advection, settings);
        state.velocity = advect_with_velocity<openvdb::Vec3fGrid>(
            *state.velocity, advection, step_dt);
      }
    }
    {
      ScopedTimer timer(ms_projection);
      /* The wall velocity has to be in the field before the projection: the divergence it creates
       * is what pushes the smoke around a moving collider instead of letting the collider slide
       * through it. It is imposed again after the projection, because that is where the clip has to
       * survive (the pressure solve does not know about the wall velocity). */
      clamp_velocity_at_solids(state, collider_masks.current, dynamic_colliders);
      project_velocity(state, settings, r_stats, collider_masks.filled);
      clamp_velocity_at_solids(state, collider_masks.current, dynamic_colliders);
    }
    {
      ScopedTimer timer(ms_scalars);
      /* Density, temperature and fuel usually occupy exactly the same voxels, in which case one
       * Vec3 advection shares the backtrace and the velocity samples between the three fields.
       * When they don't, fall back to one advection per field. */
      if (!advect_scalars_packed(state, settings, step_dt)) {
        const openvdb::Vec3fGrid &velocity = *state.velocity;
        openvdb::FloatGrid::Ptr *targets[3] = {&state.density, &state.temperature, &state.fuel};
        threading::parallel_for(IndexRange(3), 1, [&](const IndexRange range) {
          for (const int i : range) {
            if ((*targets[i])->activeVoxelCount() == 0) {
              continue;
            }
            Advection advection(velocity);
            configure_advection(advection, settings);
            *targets[i] = advect_with_velocity<openvdb::FloatGrid>(**targets[i], advection, step_dt);
          }
        });
      }
    }
    {
      ScopedTimer timer(ms_combustion);
      combust_and_decay(state, settings, step_dt);
      clip_to_bounds(state, settings);
    }
    {
      ScopedTimer timer(ms_solids);
      limit_and_extrapolate_velocity(state, settings);
      carve_solids(state, collider_masks.carve);
    }
  }

  filter_flame(state, settings);
  apply_domain_mask(state, settings, r_stats);
  state.density->tree().prune(1e-5f);
  state.temperature->tree().prune(1e-5f);
  state.fuel->tree().prune(1e-5f);
  state.velocity->tree().prune(openvdb::Vec3f(1e-6f));
  r_stats.active_voxels = int(state.density->activeVoxelCount());

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
    printf("IGNIS_DEBUG profile s=%.1f coll=%.1f forces=%.1f vel=%.1f proj=%.1f scal=%.1f "
           "burn=%.1f tail=%.1f ms=%.1f\n",
           ms_sources,
           ms_colliders,
           ms_forces,
           ms_velocity,
           ms_projection,
           ms_scalars,
           ms_combustion,
           ms_solids,
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
               .count());
    printf("IGNIS_DEBUG step: density=%lld velocity=%lld max_v=%.3f div %.4f -> %.4f "
           "(%.1f%%) substeps=%d pressure=%d residual=%.5f solids=%d cfl=%d\n",
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
           r_stats.solid_voxels,
           int(r_stats.cfl_limited));
  }
  r_stats.milliseconds = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - start)
                             .count();
  return volume;
}

#else /* WITH_OPENVDB */

Volume *solve_frame(const Volume * /*previous*/,
                    const Span<const Volume *> /*sources*/,
                    const Span<const Collider> /*colliders*/,
                    const Settings & /*settings*/,
                    const float /*delta_time*/,
                    Stats & /*r_stats*/,
                    const int /*backend_request*/)
{
  return nullptr;
}

#endif /* WITH_OPENVDB */

}  // namespace blender::geometry::ignis
