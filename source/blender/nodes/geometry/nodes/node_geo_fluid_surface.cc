/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo_nodes
 *
 * `Fluid Surface` rebuilds a surface from the particles of a fluid.
 *
 * The approach is the one of [SplashSurf](https://github.com/InteractiveComputerGraphics/
 * splashsurf): instead of unioning spheres around the particles (which makes a blobby, inflated
 * surface that is limited by the particle radius) the particles are splatted into a *density*
 * field with a compact support kernel, and the isosurface of that field is extracted. Because the
 * kernel of a particle falls off smoothly and several particles overlap, the surface sits where
 * half of the kernel mass is, i.e. roughly at the positions of the outermost particles, and thin
 * splashes stay thin instead of turning into a string of blobs.
 *
 * The density is normalized so that a full particle lattice at the recommended spacing (`one
 * radius`) gives exactly 1.0, which makes `Iso Level` mean "fraction of the rest density" and
 * keeps it independent of the particle size. 0.6 (the SplashSurf default) puts the surface
 * slightly outside the outermost layer, 0.5 puts it through their centers.
 *
 * Afterwards the extracted mesh is smoothed with Taubin's λ|μ filter. Taubin is used instead of a
 * plain Laplacian because it does not shrink the surface: the positive and the negative step
 * cancel for the low frequencies, so the volume is preserved while the voxel scale noise of the
 * marching cubes output is removed.
 */

#include <cmath>

#include "BLI_array.hh"
#include "BLI_math_base.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_span.hh"
#include "BLI_task.hh"
#include "BLI_threads.h"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_lib_id.hh"
#include "BKE_mesh.hh"
#include "BKE_volume.hh"
#include "BKE_volume_to_mesh.hh"

#include "BLT_translation.hh"

#include "node_geometry_util.hh"

#ifdef WITH_OPENVDB
#  include "BKE_volume_grid.hh"
#  include "BKE_volume_openvdb.hh"

#  include <openvdb/openvdb.h>
#endif

namespace blender::nodes::node_geo_fluid_surface_cc {

#ifdef WITH_OPENVDB

/** One fluid particle as far as the surface reconstruction is concerned. */
struct FluidParticle {
  float3 position;
  /** Radius of the particle, the kernel support is twice this. */
  float radius;
  /** Volume the particle occupies, i.e. `mass / rest_density` (or `radius^3` without mass). */
  float volume;
};

static void gather_particles_from_component(const GeometryComponent &component,
                                            const float fallback_radius,
                                            Vector<FluidParticle> &r_particles)
{
  if (component.is_empty()) {
    return;
  }
  const int points_num = component.attribute_domain_size(AttrDomain::Point);
  if (points_num == 0) {
    return;
  }
  const std::optional<bke::AttributeAccessor> attributes = component.attributes();
  if (!attributes) {
    return;
  }
  const VArray<float3> positions = *attributes->lookup<float3>("position", AttrDomain::Point);
  if (!positions) {
    return;
  }
  const VArray<float> radii =
      attributes->lookup_or_default<float>("radius", AttrDomain::Point, fallback_radius).varray;
  /* The fluid stores the physical particle mass and the rest density, so the volume of a particle
   * is `mass / density`. Without them the particle is assumed to fill the lattice cell of the
   * recommended spacing, which is exactly one radius. */
  const VArray<float> masses =
      attributes->lookup_or_default<float>("mass", AttrDomain::Point, 0.0f).varray;
  const VArray<float> densities = attributes
                                      ->lookup_or_default<float>(
                                          "puncliga_rest_density", AttrDomain::Point, 0.0f)
                                      .varray;

  const int offset = r_particles.size();
  r_particles.resize(offset + points_num);
  MutableSpan<FluidParticle> dst = r_particles.as_mutable_span().slice(offset, points_num);
  for (const int i : IndexRange(points_num)) {
    const float radius = std::max(0.0f, radii[i]);
    const float mass = masses[i];
    const float density = densities[i];
    const float volume = (mass > 0.0f && density > 0.0f) ? mass / density :
                                                           radius * radius * radius;
    dst[i] = {positions[i], radius, volume};
  }
}

/**
 * Sum of the kernel over a full lattice with spacing `spacing`, times the volume of one lattice
 * cell. Dividing the splatted density by it makes the density of a lattice exactly 1.0.
 */
static double lattice_kernel_sum(const float spacing)
{
  const double s = spacing;
  const double h = 2.0 * s;
  const double h2 = h * h;
  const double kernel = (315.0 / (64.0 * M_PI)) / (h2 * h2 * h2 * h2 * h);
  const int reach = int(std::ceil(h / s));
  double sum = 0.0;
  for (int k = -reach; k <= reach; k++) {
    for (int j = -reach; j <= reach; j++) {
      for (int i = -reach; i <= reach; i++) {
        const double d2 = (double(i) * i + double(j) * j + double(k) * k) * s * s;
        if (d2 >= h2) {
          continue;
        }
        const double falloff = h2 - d2;
        sum += kernel * falloff * falloff * falloff * s * s * s;
      }
    }
  }
  return sum > 0.0 ? sum : 1.0;
}

/** Poly6 splat of the given particles into `grid` (which is used by a single thread). */
static void splat_particles(const Span<FluidParticle> particles,
                            const float voxel_size,
                            const double normalizer,
                            openvdb::FloatGrid &grid)
{
  openvdb::FloatGrid::Accessor accessor = grid.getAccessor();
  const double voxel_size_sq = double(voxel_size) * double(voxel_size);
  for (const FluidParticle &particle : particles) {
    const double h = 2.0 * double(particle.radius);
    if (h <= 0.0 || particle.volume <= 0.0f) {
      continue;
    }
    const double h2 = h * h;
    /* `315 / (64 pi h^9)`. */
    const double kernel = (315.0 / (64.0 * M_PI)) / (h2 * h2 * h2 * h2 * h);
    const double scale = double(particle.volume) * kernel / normalizer;

    const openvdb::Vec3d center = grid.transform().worldToIndex(
        openvdb::Vec3d(particle.position.x, particle.position.y, particle.position.z));
    /* OpenVDB voxel `i` is centered at `i * voxel_size` (unlike image space, there is no half
     * voxel offset), which is also why `worldToIndexCellCentered` rounds instead of flooring. */
    const int reach = int(std::ceil(h / double(voxel_size)));
    const int base_i = int(std::floor(center.x()));
    const int base_j = int(std::floor(center.y()));
    const int base_k = int(std::floor(center.z()));
    for (int k = base_k - reach; k <= base_k + reach; k++) {
      const double dz = double(k) - center.z();
      for (int j = base_j - reach; j <= base_j + reach; j++) {
        const double dy = double(j) - center.y();
        const double dyz2 = (dy * dy + dz * dz) * voxel_size_sq;
        if (dyz2 >= h2) {
          continue;
        }
        for (int i = base_i - reach; i <= base_i + reach; i++) {
          const double dx = double(i) - center.x();
          const double d2 = dyz2 + dx * dx * voxel_size_sq;
          if (d2 >= h2) {
            continue;
          }
          const double falloff = h2 - d2;
          const float value = float(scale * falloff * falloff * falloff);
          const openvdb::Coord coord(i, j, k);
          accessor.setValue(coord, accessor.getValue(coord) + value);
        }
      }
    }
  }
}

/** Taubin λ|μ smoothing: removes the marching cubes noise without shrinking the surface. */
static void smooth_positions_taubin(Mesh &mesh,
                                    const int iterations,
                                    const float lambda,
                                    const float mu)
{
  const int verts_num = mesh.verts_num;
  if (verts_num == 0 || iterations <= 0) {
    return;
  }
  const Span<int2> edges = mesh.edges();
  if (edges.is_empty()) {
    return;
  }

  /* Vertex neighbours in CSR form. */
  Array<int> offsets(verts_num + 1);
  offsets.fill(0);
  for (const int2 &edge : edges) {
    offsets[edge[0] + 1]++;
    offsets[edge[1] + 1]++;
  }
  for (const int i : IndexRange(verts_num)) {
    offsets[i + 1] += offsets[i];
  }
  Array<int> neighbours(offsets[verts_num]);
  Array<int> cursors = offsets;
  for (const int2 &edge : edges) {
    neighbours[cursors[edge[0]]++] = edge[1];
    neighbours[cursors[edge[1]]++] = edge[0];
  }

  MutableSpan<float3> positions = mesh.vert_positions_for_write();
  Array<float3> tmp(verts_num);
  auto apply = [&](const float amount) {
    threading::parallel_for(IndexRange(verts_num), 1024, [&](const IndexRange range) {
      for (const int i : range) {
        const Span<int> vert_neighbours = neighbours.as_span().slice(offsets[i],
                                                                     offsets[i + 1] - offsets[i]);
        if (vert_neighbours.is_empty()) {
          tmp[i] = positions[i];
          continue;
        }
        float3 average = float3(0.0f);
        for (const int other : vert_neighbours) {
          average += positions[other];
        }
        average /= float(vert_neighbours.size());
        tmp[i] = positions[i] + (average - positions[i]) * amount;
      }
    });
    positions.copy_from(tmp);
  };

  for (const int iteration : IndexRange(iterations)) {
    UNUSED_VARS(iteration);
    apply(lambda);
    apply(mu);
  }
  mesh.tag_positions_changed();
}

/**
 * Relax the density field towards the average of its neighbours. Splatting one kernel per particle
 * leaves a ripple at the scale of the particle spacing (the surface looks quilted), and no amount
 * of mesh smoothing can remove it afterwards, because the mesh only samples what the field holds.
 * Voxels outside the field count as the value of the center voxel, so the isosurface does not move
 * outwards or inwards, only the ripples flatten.
 */
static void smooth_density_field(openvdb::FloatGrid &grid,
                                 const int iterations,
                                 const float strength)
{
  if (iterations <= 0 || strength <= 0.0f) {
    return;
  }
  static const openvdb::Coord offsets[6] = {
      openvdb::Coord(1, 0, 0),
      openvdb::Coord(-1, 0, 0),
      openvdb::Coord(0, 1, 0),
      openvdb::Coord(0, -1, 0),
      openvdb::Coord(0, 0, 1),
      openvdb::Coord(0, 0, -1),
  };
  openvdb::FloatGrid::Accessor accessor = grid.getAccessor();
  Vector<openvdb::Coord> coords;
  coords.reserve(int(grid.activeVoxelCount()));
  for (auto iter = grid.cbeginValueOn(); iter; ++iter) {
    coords.append(iter.getCoord());
  }
  Array<float> values(coords.size());
  for (const int iteration : IndexRange(iterations)) {
    UNUSED_VARS(iteration);
    for (const int i : coords.index_range()) {
      const openvdb::Coord coord = coords[i];
      const float center = accessor.getValue(coord);
      float sum = 0.0f;
      for (const openvdb::Coord &offset : offsets) {
        const openvdb::Coord other = coord + offset;
        sum += accessor.isValueOn(other) ? accessor.getValue(other) : center;
      }
      values[i] = center + (sum / 6.0f - center) * strength;
    }
    for (const int i : coords.index_range()) {
      accessor.setValue(coords[i], values[i]);
    }
  }
}

#endif /* WITH_OPENVDB */

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Points"_ustr)
      .supported_type({GeometryComponent::Type::Mesh,
                       GeometryComponent::Type::PointCloud,
                       GeometryComponent::Type::Curve})
      .translation_context(BLT_I18NCONTEXT_ID_POINTCLOUD)
      .is_default_link_socket()
      .description("Particles of the fluid, usually the `Geometry` output of `Puncliga Dynamics`");
  b.add_input<decl::Float>("Voxel Size"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description("Resolution of the density field. 0 uses half of the particle radius");
  b.add_input<decl::Float>("Particle Radius"_ustr)
      .default_value(0.05f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description(
          "Radius of the particles that do not carry a `radius` attribute; the support of the "
          "kernel is twice this");
  b.add_input<decl::Float>("Iso Level"_ustr)
      .default_value(0.6f)
      .min(0.01f)
      .max(2.0f)
      .description(
          "Density of the surface, as a fraction of the rest density of the particles. 0.5 puts "
          "it through the centers of the outermost particles, higher values make it fuller");
  b.add_input<decl::Int>("Smoothing"_ustr)
      .default_value(2)
      .min(0)
      .max(20)
      .description(
          "Iterations of the volume preserving Taubin smoothing of the extracted surface. Removes "
          "the staircase look of the marching cubes");
  b.add_input<decl::Int>("Density Smoothing"_ustr)
      .default_value(2)
      .min(0)
      .max(10)
      .description(
          "Relaxation passes on the density field. The surface of the particles is quilted at the "
          "scale of the particle spacing without this; the mesh smoothing above only removes the "
          "smaller marching cubes steps");
  b.add_input<decl::Float>("Adaptivity"_ustr)
      .default_value(0.0f)
      .min(0.0f)
      .max(1.0f)
      .description("Simplification of the surface: 0 keeps every triangle, 1 simplifies the most");
  b.add_input<decl::Bool>("Smooth Shading"_ustr)
      .default_value(true)
      .description("Shade the surface smooth instead of flat");
  b.add_output<decl::Geometry>("Mesh"_ustr);
  b.add_output<decl::Geometry>("Density"_ustr)
      .description(
          "The density field as a volume, 1.0 is the rest density (for rendering or "
          "debugging)");
}

static void node_geo_exec(GeoNodeExecParams params)
{
#ifdef WITH_OPENVDB
  const GeometrySet geometry = params.extract_input<GeometrySet>("Points"_ustr);
  const float fallback_radius = std::max(0.0f,
                                         params.extract_input<float>("Particle Radius"_ustr));
  const float voxel_size_input = std::max(0.0f, params.extract_input<float>("Voxel Size"_ustr));
  const float iso_level = std::max(0.0f, params.extract_input<float>("Iso Level"_ustr));
  const int smoothing = std::max(0, params.extract_input<int>("Smoothing"_ustr));
  const int density_smoothing = std::max(0, params.extract_input<int>("Density Smoothing"_ustr));
  const float adaptivity = math::clamp(params.extract_input<float>("Adaptivity"_ustr), 0.0f, 1.0f);
  const bool smooth_shading = params.extract_input<bool>("Smooth Shading"_ustr);

  Vector<FluidParticle> particles;
  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Curve})
  {
    const GeometryComponent *component = geometry.get_component(type);
    if (component == nullptr) {
      continue;
    }
    gather_particles_from_component(*component, fallback_radius, particles);
  }
  if (particles.is_empty()) {
    params.set_default_remaining_outputs();
    return;
  }

  double radius_sum = 0.0;
  for (const FluidParticle &particle : particles) {
    radius_sum += particle.radius;
  }
  const float mean_radius = float(radius_sum / double(particles.size()));
  if (mean_radius <= 0.0f) {
    params.set_default_remaining_outputs();
    return;
  }
  const float voxel_size = voxel_size_input > 0.0f ? voxel_size_input : 0.5f * mean_radius;
  const double normalizer = lattice_kernel_sum(mean_radius);
  if (std::getenv("BLENDER_FLUID_SURFACE_DEBUG") != nullptr) {
    printf("FLUID_SURFACE_DEBUG %ld particles mean_radius=%f voxel_size=%f normalizer=%f\n",
           long(particles.size()),
           mean_radius,
           voxel_size,
           normalizer);
  }

  /* The splat is done into one grid per bucket of particles and the grids are merged afterwards,
   * because an OpenVDB grid is not thread safe. One bucket per parallel task, so no two tasks ever
   * touch the same grid. */
  const int bucket_num = std::clamp(BLI_system_thread_count(), 1, 32);
  Array<openvdb::FloatGrid::Ptr> bucket_grids(bucket_num);
  for (const int i : IndexRange(bucket_num)) {
    bucket_grids[i] = openvdb::FloatGrid::create(0.0f);
    bucket_grids[i]->setTransform(
        openvdb::math::Transform::createLinearTransform(double(voxel_size)));
  }
  const int64_t particles_num = particles.size();
  const int64_t bucket_size = (particles_num + bucket_num - 1) / bucket_num;
  threading::parallel_for(IndexRange(bucket_num), 1, [&](const IndexRange range) {
    for (const int bucket_i : range) {
      const int64_t start = bucket_i * bucket_size;
      const int64_t end = std::min(start + bucket_size, particles_num);
      if (start >= end) {
        continue;
      }
      splat_particles(particles.as_span().slice(start, end - start),
                      voxel_size,
                      normalizer,
                      *bucket_grids[bucket_i]);
    }
  });

  openvdb::FloatGrid::Ptr grid = bucket_grids[0];
  {
    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
    for (const int bucket_i : IndexRange(1, bucket_num - 1)) {
      for (auto iter = bucket_grids[bucket_i]->cbeginValueOn(); iter; ++iter) {
        const openvdb::Coord coord = iter.getCoord();
        accessor.setValue(coord, accessor.getValue(coord) + iter.getValue());
      }
      bucket_grids[bucket_i].reset();
    }
  }
  grid->setGridClass(openvdb::GRID_FOG_VOLUME);
  grid->tree().prune();
  smooth_density_field(*grid, density_smoothing, 0.5f);

  Mesh *mesh = nullptr;
  if (grid->activeVoxelCount() > 0) {
    mesh = bke::volume_grid_to_mesh(*grid, iso_level, adaptivity);
  }
  if (mesh == nullptr) {
    mesh = BKE_mesh_new_nomain(0, 0, 0, 0);
  }
  smooth_positions_taubin(*mesh, smoothing, 0.5f, -0.53f);
  bke::mesh_smooth_set(*mesh, smooth_shading);

  GeometrySet mesh_set;
  mesh_set.replace_mesh(mesh);
  params.set_output("Mesh"_ustr, std::move(mesh_set));

  GeometrySet volume_set;
  if (grid->activeVoxelCount() > 0) {
    Volume *volume = BKE_id_new_nomain<Volume>(nullptr);
    BKE_volume_grid_add_vdb(*volume, "density", grid);
    volume_set.replace_volume(volume);
  }
  params.set_output("Density"_ustr, std::move(volume_set));
#else
  params.error_message_add(NodeWarningType::Error,
                           TIP_("Fluid Surface requires a build with OpenVDB"));
  params.set_default_remaining_outputs();
#endif
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeFluidSurface"_ustr);
  ntype.ui_name = "Fluid Surface";
  ntype.ui_description =
      "Rebuild a surface from fluid particles by extracting the isosurface of their density field "
      "(SplashSurf style)";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_fluid_surface_cc
