/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

#include "BKE_bvhutils.hh"
#include "BKE_curves.hh"
#include "BKE_geometry_fields.hh"
#include "BKE_grease_pencil.hh"
#include "BKE_instances.hh"
#include "BKE_lib_id.hh"
#include "BKE_mesh.h"
#include "BKE_mesh_mapping.hh"
#include "BKE_mesh_sample.hh"
#include "BKE_pointcloud.hh"

#include "BLI_hash.h"
#include "BLI_kdopbvh.hh"
#include "BLI_math_bits.h"
#include "BLI_math_geom.h"
#include "BLI_set.hh"
#include "BLI_sort.hh"
#include "BLI_stack.hh"
#include "BLI_string_utf8.h"
#include "BLI_virtual_array_range_spans.hh"

#include "DNA_curves_types.h"
#include "DNA_mesh_types.h"

#include "GEO_xpbd_constraint_collision_edge.hh"
#include "GEO_xpbd_constraint_collision_face.hh"
#include "GEO_xpbd_constraint_damping_angular.hh"
#include "GEO_xpbd_constraint_damping_linear.hh"
#include "GEO_xpbd_constraint_distance.hh"
#include "GEO_xpbd_constraint_friction_edge.hh"
#include "GEO_xpbd_constraint_friction_face.hh"
#include "GEO_xpbd_constraint_guide_tether.hh"
#include "GEO_xpbd_constraint_pin_position.hh"
#include "GEO_xpbd_constraint_pin_rotation.hh"
#include "GEO_xpbd_constraint_point_contact.hh"
#include "GEO_xpbd_constraint_segment_contact.hh"
#include "GEO_xpbd_constraint_segment_face.hh"
#include "GEO_xpbd_hair_grid.hh"
#include "GEO_xpbd_constraint_rod_bend_twist.hh"
#include "GEO_xpbd_constraint_rod_stretch_shear.hh"
#include "GEO_xpbd_constraint_volume.hh"
#include "GEO_xpbd_fluid.hh"

#include "GEO_ha_context.hh"
#include "GEO_ha_status.hh"
#include "GEO_ha_xpbd.hh"

#include "GPU_init_exit.hh"

#include "NOD_geo_tag_filter.hh"
#include "NOD_geometry_nodes_bundle.hh"
#include "NOD_geometry_nodes_list.hh"
#include "NOD_geometry_nodes_physics_bundles.hh"

#include "RNA_access.hh"

#include "UI_interface_layout.hh"
#include "UI_resources.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_xpbd_solver_cc {

using namespace physics_bundles;

namespace attribute_names {

constexpr StringRefNull position = "position";
constexpr StringRefNull velocity = "velocity";
constexpr StringRefNull rotation = "rotation";
constexpr StringRefNull angular_velocity = "angular_velocity";
constexpr StringRefNull external_force = "external_force";
constexpr StringRefNull external_torque = "external_torque";
constexpr StringRefNull mass = "mass";
constexpr StringRefNull moment_of_inertia = "moment_of_inertia";
constexpr StringRefNull static_friction = "static_friction";
constexpr StringRefNull dynamic_friction = "dynamic_friction";
constexpr StringRefNull radius = "radius";
constexpr StringRefNull self_collision = "puncliga_self_collision";
constexpr StringRefNull contact_thickness = "puncliga_contact_thickness";
constexpr StringRefNull hair_self_collision = "puncliga_hair_self_collision";
constexpr StringRefNull hair_contact_thickness = "puncliga_hair_contact_thickness";
constexpr StringRefNull same_strand_collision = "puncliga_same_strand_collision";
constexpr StringRefNull guide_tether = "puncliga_guide_tether";
constexpr StringRefNull tether_compliance = "puncliga_tether_compliance";

}  // namespace attribute_names

class XPBDSolverDataBundle {
 public:
  static constexpr StringRefNull name = "Blender.XPBDSolverData";
  static const FlatBundleTypePtr &get_bundle_type();
};

const FlatBundleTypePtr &XPBDSolverDataBundle::get_bundle_type()
{
  static const FlatBundleTypePtr bundle_type = []() {
    FlatBundleTypeBuilder b(XPBDSolverDataBundle::name);
    b.add<decl::Float>("residual_error"_ustr)
        .min(0.0f)
        .description(
            "Average remaining relative error, values smaller than one are below the constraint "
            "threshold.");
    const FlatBundleTypePtr bundle_type = b.build();
    BundleTypeRegistry::register_type(bundle_type);
    return bundle_type;
  }();
  return bundle_type;
}

static NestedBundleTypePtr make_world_type()
{
  Vector<std::shared_ptr<const FlatBundleType>> types;
  types.append(XPBDSolverDataBundle::get_bundle_type());
  types.append(DampingBundle::get_bundle_type());
  types.append(MeshColliderBundle::get_bundle_type());
  types.append(CollisionContactsBundle::get_bundle_type());
  types.append(RodStretchShearBundle::get_bundle_type());
  types.append(RodBendTwistBundle::get_bundle_type());
  types.append(PinPositionBundle::get_bundle_type());
  types.append(PinRotationBundle::get_bundle_type());
  types.append(EdgeLengthConstraintBundle::get_bundle_type());
  types.append(VolumeConstraintBundle::get_bundle_type());
  types.append(CrossEdgeLengthConstraintBundle::get_bundle_type());
  types.append(FluidBundle::get_bundle_type());

  /* Not actually used by the node but only registered here. */
  ForceBundle::get_bundle_type();
  CustomGeometryEffector::get_bundle_type();
  CustomWorldEffector::get_bundle_type();

  NestedBundleTypePtr world_type = std::make_shared<const NestedBundleType>(
      "Blender.XPBDSolverWorld", std::move(types));
  BundleTypeRegistry::register_type(world_type);
  return world_type;
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.use_custom_socket_order();
  b.allow_any_socket_order();

  static NestedBundleTypePtr world_type = make_world_type();
  b.add_input<decl::Bundle>("World"_ustr)
      .bundle_type(world_type)
      .evaluated_geometry_field()
      .structure_type(StructureType::Single)
      .description("World state that is updated by the solver");
  b.add_output<decl::Bundle>("World"_ustr).pass_through_input_index(0).align_with_previous();
  b.add_input<decl::Float>("Delta Time"_ustr)
      .min(0)
      .default_value(1 / 25.0f)
      .subtype(PROP_TIME_ABSOLUTE);

  b.add_input<decl::String>("Filter"_ustr)
      .optional_label()
      .description("Filters the geometry sets to process based on their tags");

  b.add_input<decl::Matrix>("Simulation to World"_ustr);

  {
    auto &solver_panel = b.add_panel("Solver"_ustr).default_closed(true);
    solver_panel.add_input<decl::Int>("Substeps"_ustr).default_value(10).min(1);
    solver_panel.add_input<decl::Int>("Constraint Iterations"_ustr).default_value(1).min(1);
    solver_panel.add_input<decl::Bool>("Self Collision"_ustr)
        .default_value(false)
        .description(
            "Push the points of the simulated solids apart when they come closer than the sum of "
            "their collision radii, so a cloth collides with itself and with soft bodies (the "
            "points need a `radius` attribute, which the `Init Puncliga Data` node writes)");
    solver_panel.add_input<decl::Float>("Contact Thickness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Extra distance kept between two colliding points, on top of their radii");
    solver_panel.add_input<decl::Bool>("Hair Self Collision"_ustr)
        .default_value(false)
        .description(
            "Keep the hair strands apart with segment/segment contacts. The segments need a "
            "`radius` attribute, which `Init Puncliga Data > Hair` writes; the roots of neighbouring "
            "strands are ignored when they already touch in the rest pose");
    solver_panel.add_input<decl::Float>("Hair Contact Thickness"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Extra distance kept between two hair segments, on top of their radii");
    solver_panel.add_input<decl::Bool>("Same Strand Collision"_ustr)
        .default_value(false)
        .description(
            "Also collide segments of the same strand with each other. Off keeps a strand from "
            "pushing its own neighbouring segments apart; curly hair may want it on");
    solver_panel.add_input<decl::String>("Solver Path"_ustr)
        .default_value("")
        .description("Optional output path in the world bundle for solver data");
  }
  {
    auto &p = b.add_panel("Interpolation Range"_ustr).default_closed(true);
    p.add_input<decl::Float>("Begin"_ustr).default_value(0.0).min(0.0);
    p.add_input<decl::Float>("End"_ustr).default_value(1.0).min(0.0);
  }
}

struct DataKey {
  int geo_bundle_i;
  bke::GeometryComponent::Type type;
  std::optional<int> layer_i;

  uint64_t hash() const
  {
    return get_default_hash(this->geo_bundle_i, this->type, this->layer_i.value_or(0));
  }

  friend bool operator==(const DataKey &a, const DataKey &b) = default;
};

struct GeometryDataChunk {
  int data_key_i;
  IndexRange points_range;
  /** Only used when the geometry data has curves. */
  std::optional<IndexRange> curves_range;
};

struct DampingConstraint {
  std::string path;
};
struct DampingConstraintUsage {
  /** Index of corresponding #DampingConstraint. */
  int constraint_i;
  VArrayRangeSpans<float> linear_dampings;
  VArrayRangeSpans<float> angular_dampings;
  MutableSpan<float> linear_damping_lambdas;
  MutableSpan<float> angular_damping_lambdas;
};

struct RodStretchShearConstraint {
  std::string path;
  float error_threshold;
  std::string lambda_pos_attr;
  std::string lambda_rot_attr;
};
struct RodStretchShearConstraintUsage {
  /** Index of corresponding #RodStretchShearConstraint. */
  int constraint_i;
  bool is_valid = false;
  VArraySpan<float> rest_lengths;
  VArrayRangeSpans<float> compliances;
  MutableSpan<float3> lambdas_pos;
  MutableSpan<float3> lambdas_rot;
};

struct RodBendTwistConstraint {
  std::string path;
  float error_threshold;
};
struct RodBendTwistConstraintUsage {
  /** Index of corresponding #RodBendTwistConstraint. */
  int constraint_i;
  VArraySpan<math::Quaternion> rest_bend_rotations;
  VArrayRangeSpans<float> compliances;
  MutableSpan<float4> lambdas;
};

struct PinPositionConstraint {
  std::string path;
  float error_threshold;
  std::string lambda_attr;
};
struct PinPositionConstraintUsage {
  /** Index of corresponding #PinPositionConstraint. */
  int constraint_i;

  Span<int> points;
  Span<float3> begin_positions;
  Span<float3> end_positions;
  Span<float> compliances;

  MutableSpan<float3> current_positions;
  MutableSpan<float> lambdas;
};
struct PinPositionConstraintChunkUsage {
  /** Index of the corresponding #PinPositionConstraintUsage. */
  int constraint_usage_i;
  IndexRange pin_range;
};

struct PinRotationConstraint {
  std::string path;
  float error_threshold;
};
struct PinRotationConstraintUsage {
  /** Index of corresponding #PinRotationConstraint. */
  int constraint_i;

  Span<int> points;
  Span<math::Quaternion> begin_rotations;
  Span<math::Quaternion> end_rotations;
  Span<float> compliances;

  MutableSpan<math::Quaternion> current_rotations;
  MutableSpan<float4> lambdas;
};
struct PinRotationConstraintChunkUsage {
  /** Index of the corresponding #PinRotationConstraintUsage. */
  int constraint_usage_i;
  IndexRange pin_range;
};

struct EdgeLengthConstraint {
  std::string path;
  float error_threshold;
};
struct EdgeLengthConstraintUsage {
  /** Index of corresponding #EdgeLengthConstraint. */
  int constraint_i;
  bool is_valid = false;
  VArraySpan<float> rest_lengths;
  VArraySpan<float> compliances;
  MutableSpan<float> lambdas;

  xpbd::ConstraintColoring coloring;
};

struct CrossEdgeLengthConstraint {
  std::string path;
  float error_threshold;
};
struct CrossEdgeLengthConstraintUsage {
  /** Index of the corresponding #CrossEdgeLengthConstraint. */
  int constraint_i;

  Vector<int2> cross_edge_points;
  Vector<float> distances;
  Vector<float> compliances;
  MutableSpan<float> lambdas;

  xpbd::ConstraintColoring coloring;
};

/** One volume constraint per four cornered face of a tetrahedral mesh. */
struct VolumeConstraint {
  std::string path;
  float compliance = 0.0f;
  /** Compliance of the distance constraints on the six edges of every element. */
  float edge_compliance = 0.0f;
  float error_threshold = 1e-3f;
};
struct VolumeConstraintUsage {
  /** Index of the corresponding #VolumeConstraint. */
  int constraint_i;
  /** Four point indices per element (only the elements that are not slivers). */
  GroupedSpan<int> elements;
  Vector<int> element_points;
  Array<int> element_offsets;
  Vector<float> rest_volumes;
  /** Materialized compliance per element, the constraint set holds a span into this. */
  Vector<float> compliances;
  MutableSpan<float> lambdas;

  /**
   * The six edges of every element (deduplicated), which get distance constraints as well: a
   * volume constraint alone does not resist shear. The vectors have to outlive the constraint sets,
   * which only hold spans.
   */
  Vector<int2> edge_points;
  Vector<float> edge_rest_lengths;
  Vector<float> edge_compliances;
  MutableSpan<float> edge_lambdas;

  xpbd::ConstraintColoring coloring;
};

/** Fluid parameters, gathered from a `Blender.Fluid` bundle in the world. */
struct FluidConstraint {
  std::string path;
  float rest_density = 1000.0f;
  float compliance = 0.0f;
  float viscosity = 0.0f;
  float damping = 0.0f;
  float vorticity = 0.0f;
  float surface_tension = 0.0f;
  float negative_pressure_limit = 0.0f;
  float artificial_pressure = 0.0f;
  /** Used when the geometry has no `radius` attribute. */
  float particle_radius = 0.05f;
  bool boundary_particles = true;
  float boundary_friction = 0.5f;
  /** Strength of the two way coupling with the other simulated geometries in the world. */
  float coupling = 1.0f;
  /**
   * XPBD compliance of the point/triangle contacts with the coupled solids (cloth, soft bodies).
   * Zero keeps the hard contact; the gather multiplies it by `1 / dt^2` like every other XPBD
   * compliance.
   */
  float coupling_compliance = 0.0f;
  float error_threshold = 1e-2f;
};

/** Per chunk state of the fluid solve pass. */
struct FluidChunkData {
  bool active = false;
  int constraint_i = -1;
  /** Support radius of the kernel, twice the particle radius. */
  float support_radius = 0.0f;
  xpbd::FluidParticleGrid grid;
  /**
   * Neighbour indices, rebuilt with the grid every substep. The constraint iterations and the
   * velocity passes then walk a flat array instead of the hash, which is where most of the frame
   * went when boundary particles are in play (they double the number of hash walks per particle).
   */
  xpbd::FluidNeighborList neighbors;
  /** The same, resolved against the boundary and the coupled particles (query by fluid particle). */
  xpbd::FluidNeighborList boundary_neighbors;
  xpbd::FluidNeighborList coupled_neighbors;
  /** Accumulated lagrange multipliers, reset every substep. */
  Array<float> lambdas;
  /** Scratch buffers, sized like the chunk. */
  Array<float> errors;
  /** Signed relative density error per particle, negative when the particles are too far apart. */
  Array<float> density_errors;
  /** Mean of the signed density error of the last solved substep, used for the node warning. */
  float last_density_error_mean = 0.0f;
  Array<int> neighbor_counts;
  Array<float3> corrections;
  /** Only used when vorticity confinement is enabled. */
  Array<float3> omegas;
  Array<float> omega_magnitudes;

  /**
   * Device side of this fluid (M87). When `gpu_active` the whole substep runs on the compute
   * backend and every CPU fluid pass for this chunk is skipped; the state is uploaded once per
   * frame and downloaded at the end of the last substep.
   */
  geometry::ha::HaXpbdFluid gpu_fluid;
  bool gpu_active = false;
  /**
   * Dense distance field of the collider of this fluid, when it has exactly one static mesh
   * collider. Contacts are then solved on the device by sampling it (no host gather, no readback).
   */
  std::shared_ptr<geometry::ha::HaSdfGrid> sdf;
  bool sdf_ready = false;
  /**
   * Distance field of the coupled solids (cloth, soft bodies), rebuilt once per frame. The device
   * samples it for the hard contact and appends the contacts so the reaction can be given back.
   */
  std::shared_ptr<geometry::ha::HaSdfGrid> coupled_sdf;
  bool coupled_sdf_ready = false;
  /** Half width of the coupled field's band, needed by the crossing check of the contact. */
  float coupled_sdf_band = 0.0f;
  /** Hash of the collider and the field resolution that produced #sdf. */
  uint64_t sdf_hash = 0;
  /** Wall clock when the device work of the current frame started, for the debug log. */
  double gpu_start_ms = 0.0;

  /**
   * Particles sampled on the colliders that the fluid can reach. They are rebuilt once per frame
   * and act as the solid boundary of the fluid (only used where `puncliga_boundary_particles` is
   * set).
   */
  bool boundary_active = false;
  Vector<float3> boundary_positions;
  Vector<float> boundary_masses;
  xpbd::FluidParticleGrid boundary_grid;

  /**
   * Particles of the other simulated geometries of the world (cloth, soft bodies). They are part
   * of the density estimate like the collider samples and they are moved by the fluid pressure,
   * which is the two way coupling. They are refreshed from the solver positions every substep,
   * because the solids move during the solve.
   *
   * A particle is either a sample on a triangle of the solid (with barycentric weights, so the
   * reaction is spread over the vertices) or, for geometries without faces, one of its points.
   */
  bool has_coupled = false;
  Vector<float3> coupled_positions;
  Vector<float> coupled_masses;
  Vector<float> coupled_inv_masses;
  /** Source of each particle: the geometry, the vertices it belongs to and their weights. */
  Vector<int> coupled_geometry;
  Vector<int3> coupled_vertices;
  Vector<float3> coupled_weights;
  /** Position change of the coupled particles, computed by the reaction pass. */
  Array<float3> coupled_corrections;
  /**
   * Per coupled geometry scratch space for the reaction: the correction is accumulated per vertex
   * (weighted by the barycentric weights) and then divided by the accumulated weight, so the
   * displacement of a vertex does not depend on how many samples happen to touch it.
   */
  struct CoupledVertexScratch {
    int geometry_i = -1;
    Array<float3> correction_sums;
    Array<float> weight_sums;
  };
  Vector<CoupledVertexScratch> coupled_scratch;
  xpbd::FluidParticleGrid coupled_grid;

  /** Fraction of the density of each particle that comes from the boundary particles. */
  Array<float> solid_fractions;
  /**
   * Sum of the part of the density correction that comes from the coupled solids over the
   * iterations of the current substep. The velocity update removes its normal component so a
   * depenetration does not turn into a bounce (see #update_linear_velocities).
   */
  Array<float3> coupled_push;
};

/**
 * Last baked collider distance field, shared across frames.
 *
 * `chunks_data_` is reinitialized every evaluation, so a cache stored in `FluidChunkData` would be
 * thrown away every frame and the field would be baked again (measured: 2.6 s per frame for a 20 m
 * ground plane at 4.4 M voxels). The content hash includes the collider transform, a sample of its
 * vertices and the field resolution, so a moving/deforming collider still misses and rebakes.
 */
struct SdfCacheEntry {
  uint64_t hash = 0;
  float voxel_size = 0.0f;
  float band_width = 0.0f;
  /** World space box the field was built for; the fluid may move inside it without a rebake. */
  float3 region_min = float3(0.0f);
  float3 region_max = float3(0.0f);
  std::shared_ptr<geometry::ha::HaSdfGrid> grid;
};
static std::mutex sdf_cache_mutex;
static SdfCacheEntry sdf_cache;

/** Frees the cached field while Blender's allocator is still alive. */
static void sdf_cache_dispose(void * /*user_data*/)
{
  std::lock_guard<std::mutex> lock(sdf_cache_mutex);
  sdf_cache.grid.reset();
}

static void ensure_sdf_cache_hook()
{
  static std::once_flag once;
  std::call_once(once, []() { GPU_exit_hook_register(sdf_cache_dispose, nullptr); });
}

struct StaticMeshInfo {
  const Mesh *mesh;
  bke::BVHTreeFromMesh corner_tris_bvh;
  bke::BVHTreeFromMesh edges_bvh;
};

struct DeformingMeshInfo {
  /** This is one longer than the number of substeps because it also contains the initial mesh. */
  Vector<const Mesh *> substep_meshes;
  /**
   * This contains one bvh tree per substep. A future optimization could be to not build
   * independent BVH trees for each mesh because the are usually very similar.
   */
  Vector<bke::BVHTreeFromMesh> substep_corner_tris_bvh_trees;
  Vector<bke::BVHTreeFromMesh> substep_edges_bvh_trees;
};

struct MeshCollider {
  std::string path;
  Vector<int> instance_ids;
  std::variant<StaticMeshInfo, DeformingMeshInfo> mesh;
  float4x4 begin_transform;
  float4x4 end_transform;
  float margin;
  float friction;
  float compliance;
  bool use_edge_contacts;
  /** If true, the points are expected to stay inside of the mesh instead of being pushed out. */
  bool is_boundary;
  float error_threshold;
};
struct MeshColliderUsage {
  /** Index of corresponding #MeshCollider. */
  int constraint_i;
};

struct CollisionContacts {
  std::string path;
};

struct ConstraintWithColoring {
  xpbd::ConstraintSet *constraint;
  xpbd::ConstraintColoring coloring;
};

struct GeometryData {
  bke::MutableAttributeAccessor attributes;
  AttrDomain domain;
  int size;
  bool uses_rotation = false;
  /** Easy access to curve data for curves and grease pencil layers. */
  bke::CurvesGeometry *curves = nullptr;

  IndexRange chunks;
  /**
   * Allows fast lookup of which chunk a point belongs to. This allows e.g. grouping collisions by
   * chunk efficiently.
   */
  Array<int> point_to_chunk;

  bke::SpanAttributeWriter<float3> position_attr;
  bke::SpanAttributeWriter<float3> velocity_attr;
  bke::SpanAttributeWriter<math::Quaternion> rotation_attr;
  bke::SpanAttributeWriter<float3> angular_velocity_attr;

  /**
   * Temporary arrays for positions and rotations. This is necessary because xpbd requires the old
   * and new positions in the end to compute the new velocities.
   */
  Array<float3> temp_positions;
  Array<math::Quaternion> temp_rotations;

  VArray<float> masses;
  VArraySpan<float3> moments_of_inertia;

  VArraySpan<float3> external_force_attr;
  VArraySpan<float3> external_torque_attr;
  /**
   * Uniform acceleration from `Blender.DilapidoForce` bundles (wind). It is added to the force per
   * unit mass in the inertial update, so it applies to hair, cloth and soft bodies without needing a
   * closure field.
   */
  Vector<float3> external_acceleration;

  CacheMutex load_friction_mutex;
  VArray<float> static_frictions;
  VArray<float> dynamic_frictions;

  CacheMutex load_radius_mutex;
  VArray<float> radii;
  /** Actually the same as radii for now. */
  VArray<float> prev_radii;

  Array<float> inv_masses;
  Array<float3> inv_moments_of_inertia;

  Array<bool> is_hard_pinned;

  Vector<DampingConstraintUsage> damping_constraints;
  Vector<PinPositionConstraintUsage> pin_position_constraints;
  Vector<PinRotationConstraintUsage> pin_rotation_constraints;
  Vector<EdgeLengthConstraintUsage> edge_length_constraints;
  Vector<VolumeConstraintUsage> volume_constraints;
  Vector<CrossEdgeLengthConstraintUsage> cross_edge_length_constraints;
  Vector<MeshColliderUsage> mesh_colliders;
  Vector<RodStretchShearConstraintUsage> rod_stretch_shear_constraints;
  Vector<RodBendTwistConstraintUsage> rod_bend_twist_constraints;

  Vector<ConstraintWithColoring> static_constraints;

  /** Index into #ConstraintsInfo::fluid_constraints, -1 when this geometry is not a fluid. */
  int fluid_constraint_i = -1;
  /** Materialized mass per point, only initialized for fluids. */
  Array<float> fluid_masses;
  /** Materialized particle radius per point, only initialized for fluids. */
  Array<float> fluid_radii;
  /** Support radius of the fluid kernel. */
  float fluid_support_radius = 0.0f;
  /**
   * Materialized fluid parameters per point. They come from the `puncliga_*` attributes that
   * `Init Puncliga Data` writes, so several fluids with different settings can be simulated in one
   * world; the bundle values are only the fallback.
   */
  Array<float> fluid_rest_density;
  Array<float> fluid_viscosity;
  Array<float> fluid_damping;
  Array<float> fluid_vorticity;
  Array<float> fluid_surface_tension;
  Array<float> fluid_boundary_friction;
  Array<float> fluid_negative_pressure_limit;
  Array<bool> fluid_use_boundary;
  /** Per point strength of the two way coupling with the other simulated geometries. */
  Array<float> fluid_coupling;
  /** Per point compliance of the point/triangle contacts with the coupled solids. */
  Array<float> fluid_coupling_compliance;

  GeometryData(const bke::MutableAttributeAccessor attributes) : attributes(attributes) {}
};

struct GeometrySetData {
  std::string path;
  /**
   * The geometry is moved out of the world bundle for local processing and is moved back in the
   * end.
   */
  GeometrySet geometry;
  Set<std::string> tags;
};

struct Geometries {
  Vector<GeometrySetData> geometry_sets;

  VectorSet<DataKey> data_keys;
  Vector<GeometryData *> data;

  /**
   * By having static chunks, across the entire solve step allows for more efficient
   * multi-threading (and preparation for multi-threading).
   */
  Vector<GeometryDataChunk> chunks;
  int max_chunk_size = -1;
  int64_t total_points_num = -1;

  /**
   * Two arrays of geometry references are used because the arrays containing the previous and
   * current positions/rotations are swapped for different time steps to avoid unnecessary copying.
   *
   * Index 0: Previous data is stored in temporary arrays, current data in the geometry attributes.
   * Index 1: Previous data is stored in the geometry attributes, current data in temporary arrays.
   */
  std::array<Array<xpbd::GeometryRef>, 2> solver_refs;
};

/**
 * One of the other simulated geometries of the world (a cloth, a soft body), prepared as a
 * collision surface for the fluid. The mesh is a copy that only shares the topology with the world
 * geometry: its positions are written from the solver state every substep, and the BVH is rebuilt
 * with them, so the fluid can collide with the triangles however coarse or fine the mesh is.
 */
struct CoupledSurface {
  int geometry_i = -1;
  /** Owned copy of the mesh, only used for the topology and the positions of the BVH. */
  Mesh *mesh = nullptr;
  bke::BVHTreeFromMesh bvh;
  /** Positions of the previous substep, to know how fast the surface moves in a contact. */
  Array<float3> prev_positions;
};

/**
 * One contact between a hair segment and a triangle of a cloth or a soft body.
 *
 * The segment is pushed away from the triangle, and the triangle vertices are pushed in the
 * opposite direction with the barycentric weights, so the hair and the solid body interact in both
 * directions.
 */
struct HairSolidContact {
  int segment = -1;
  /** Global point indices of the triangle corners. */
  int3 triangle = int3(0);
  /** Unit direction that pushes the segment away from the surface. */
  float3 normal = float3(0.0f);
  /** Barycentric position of the closest point on the segment. */
  float segment_s = 0.0f;
  /** Barycentric weights of the closest point on the triangle. */
  float3 face_weights = float3(0.0f);
  float min_distance = 0.0f;
};

/**
 * Segment/triangle contacts between the hair and the cloth/soft bodies it shares a world with.
 *
 * The hair points and the solid vertices share one flat point space (like the hair self contacts)
 * so one constraint set can move both sides. The solid triangles are static topology; only their
 * positions and BVH trees are refreshed every substep.
 */
struct HairSolidContacts {
  /** Global point space: hair points first, then the vertices of every participating solid. */
  Vector<int> point_geo;
  Vector<int> point_local;
  Vector<float> point_inv_mass;

  /** Hair segments in the global point space. */
  Vector<int> segment_point_a;
  Vector<int> segment_point_b;
  Vector<float> segment_radius;
  Vector<float> segment_thickness;

  /** Triangles of the solids, in the global point space. */
  Vector<int3> triangle_points;

  /** For the debug output of the constraint set. */
  Vector<int> affected_geo_indices;

  Array<int> contact_offsets;
  Vector<int> contact_points;
  Vector<float3> contact_normals;
  Vector<float2> contact_segment_weights;
  Vector<float3> contact_face_weights;
  Vector<float> contact_min_distances;
  Vector<float> contact_compliances;
  Array<float> contact_lambdas;
  xpbd::ConstraintColoring coloring;
  std::optional<xpbd::SegmentFaceContactConstraintSet> constraint_set;

  bool is_empty() const
  {
    return segment_point_a.is_empty() || triangle_points.is_empty();
  }
};

/**
 * Compliant attachments between the simulated child hair points and the points of their guides.
 *
 * The guide of a child curve comes from the standard `guide_curve_index` attribute (curve domain);
 * curves without that attribute are guides. Every child point is attached to the guide point at the
 * same normalized parameter. The constraint is two way, so a child that a collision moves drags the
 * guide with it, and the guide drags the child.
 */
struct GuideTetherContacts {
  /** Global point space: all points of the participating hair geometries. */
  Vector<int> point_geo;
  Vector<int> point_local;
  Vector<float> point_inv_mass;

  /** Child point index and guide point index (global) per tether. */
  Vector<int2> point_pairs;
  Vector<float3> rest_offsets;
  Vector<float> compliances;
  Array<float3> lambdas;
  xpbd::ConstraintColoring coloring;
  std::optional<xpbd::GuideTetherConstraintSet> constraint_set;
  Vector<int> affected_geo_indices;

  bool is_empty() const
  {
    return point_pairs.is_empty();
  }
};

/** Barycentric weights of a point on a triangle, clamped like the solver's `triangle_weights`. */
static float3 hair_triangle_weights(const float3 &v0,
                                    const float3 &v1,
                                    const float3 &v2,
                                    const float3 &point)
{
  const float3 e0 = v1 - v0;
  const float3 e1 = v2 - v0;
  const float3 offset = point - v0;
  const float d00 = math::dot(e0, e0);
  const float d01 = math::dot(e0, e1);
  const float d11 = math::dot(e1, e1);
  const float d20 = math::dot(offset, e0);
  const float d21 = math::dot(offset, e1);
  const float denominator = d00 * d11 - d01 * d01;
  if (std::abs(denominator) < 1e-16f) {
    return float3(1.0f, 0.0f, 0.0f);
  }
  const float b = (d11 * d20 - d01 * d21) / denominator;
  const float c = (d00 * d21 - d01 * d20) / denominator;
  const float a = 1.0f - b - c;
  float3 weights(
      math::clamp(a, 0.0f, 1.0f), math::clamp(b, 0.0f, 1.0f), math::clamp(c, 0.0f, 1.0f));
  const float sum = weights.x + weights.y + weights.z;
  if (sum > 1e-8f) {
    weights /= sum;
  }
  return weights;
}

/**
 * Closest point pair between the segment `p0-p1` and the triangle `v0-v1-v2`.
 *
 * The minimum of the three segment/edge distances and the two endpoint/triangle distances is exact:
 * the closest features of a segment and a triangle are either on an edge pair, or at a segment
 * endpoint, or perpendicular above the triangle interior (which the endpoint tests cover when the
 * projection is inside, and an edge covers otherwise).
 */
static float closest_segment_triangle_distance_squared(const float3 &p0,
                                                       const float3 &p1,
                                                       const float3 &v0,
                                                       const float3 &v1,
                                                       const float3 &v2,
                                                       float &r_s,
                                                       float3 &r_face_weights)
{
  float best_sq = 1e30f;
  float best_s = 0.0f;
  float3 best_weights(1.0f, 0.0f, 0.0f);
  const auto consider = [&](const float s,
                            const float3 &weights,
                            const float3 &q_segment,
                            const float3 &q_face) {
    const float sq = math::length_squared(q_face - q_segment);
    if (sq < best_sq) {
      best_sq = sq;
      best_s = s;
      best_weights = weights;
    }
  };

  const float3 corners[3] = {v0, v1, v2};
  for (const int edge_i : IndexRange(3)) {
    const int e0 = edge_i;
    const int e1 = (edge_i + 1) % 3;
    float s;
    float t;
    xpbd::closest_points_on_segments(p0, p1, corners[e0], corners[e1], s, t);
    float3 weights(0.0f);
    weights[e0] = 1.0f - t;
    weights[e1] = t;
    consider(s,
             weights,
             math::interpolate(p0, p1, s),
             math::interpolate(corners[e0], corners[e1], t));
  }
  for (const int endpoint_i : IndexRange(2)) {
    const float3 &point = endpoint_i == 0 ? p0 : p1;
    float nearest[3];
    closest_on_tri_to_point_v3(
        nearest, reinterpret_cast<const float *>(&point), reinterpret_cast<const float *>(&v0),
        reinterpret_cast<const float *>(&v1), reinterpret_cast<const float *>(&v2));
    const float3 nearest_v(nearest[0], nearest[1], nearest[2]);
    consider(float(endpoint_i),
             hair_triangle_weights(v0, v1, v2, nearest_v),
             point,
             nearest_v);
  }

  r_s = best_s;
  r_face_weights = best_weights;
  return best_sq;
}

/** Query state of one hair segment against one solid BVH (see `BLI_bvhtree_range_query`). */
struct HairSolidRangeQuery {
  const CoupledSurface *surface = nullptr;
  int surface_point_start = 0;
  int segment = -1;
  float3 p0 = float3(0.0f);
  float3 p1 = float3(0.0f);
  float segment_radius = 0.0f;
  float segment_thickness = 0.0f;
  /** Largest collision radius of the solid (conservative for the contact distance). */
  float solid_radius = 0.0f;
  Vector<HairSolidContact> *out = nullptr;
  int output_start = 0;
  int max_contacts_per_segment = 32;
};

static void hair_solid_range_query_callback(void *user_data,
                                            const int index,
                                            const float * /*co*/,
                                            const float /*dist_sq*/)
{
  HairSolidRangeQuery &query = *static_cast<HairSolidRangeQuery *>(user_data);
  if (query.out->size() - query.output_start >= query.max_contacts_per_segment) {
    return;
  }
  const Mesh &mesh = *query.surface->mesh;
  const Span<int3> corner_tris = mesh.corner_tris();
  if (index < 0 || index >= corner_tris.size()) {
    return;
  }
  const Span<int> corner_verts = mesh.corner_verts();
  const Span<float3> positions = mesh.vert_positions();
  const int3 &tri = corner_tris[index];
  const int3 verts(corner_verts[tri[0]], corner_verts[tri[1]], corner_verts[tri[2]]);
  const float3 v0 = positions[verts[0]];
  const float3 v1 = positions[verts[1]];
  const float3 v2 = positions[verts[2]];

  float segment_s;
  float3 face_weights;
  const float distance_sq = closest_segment_triangle_distance_squared(
      query.p0, query.p1, v0, v1, v2, segment_s, face_weights);
  const float min_distance = query.segment_radius + query.solid_radius +
                             std::max(query.segment_thickness, 0.0f);
  if (!(distance_sq < min_distance * min_distance)) {
    return;
  }

  const float3 q_segment = math::interpolate(query.p0, query.p1, segment_s);
  const float3 q_face = v0 * face_weights.x + v1 * face_weights.y + v2 * face_weights.z;
  float3 normal = q_segment - q_face;
  const float length = math::length(normal);
  if (length > 1e-7f) {
    normal /= length;
  }
  else {
    /* The segment crosses the triangle: push it to the side its midpoint is on. */
    normal = math::normalize(math::cross(v1 - v0, v2 - v0));
    if (!(math::length_squared(normal) > 1e-12f)) {
      return;
    }
    if (math::dot(query.p0 + query.p1 - 2.0f * q_face, normal) < 0.0f) {
      normal = -normal;
    }
  }

  HairSolidContact contact;
  contact.segment = query.segment;
  contact.triangle = int3(query.surface_point_start + verts[0],
                          query.surface_point_start + verts[1],
                          query.surface_point_start + verts[2]);
  contact.normal = normal;
  contact.segment_s = segment_s;
  contact.face_weights = face_weights;
  contact.min_distance = min_distance;
  query.out->append(contact);
}

/**
 * The point to point contacts of the simulated solids with themselves (see
 * #xpbd::PointContactConstraintSet).
 *
 * Every point of every solid geometry is just a sphere with a radius here, which is what lets one
 * pass keep a cloth from passing through itself, two cloths from passing through each other and a
 * cloth from passing through a soft body: the two closest points are pushed apart before their
 * spheres overlap. The pairs depend on the positions, so they are rebuilt every substep, while the
 * topological neighbors (which must never push each other apart) are gathered once per frame.
 */
struct SelfContacts {
  /** The participating geometries, with the range of the global points that belongs to each. */
  Vector<int> geometry_data_key;
  Vector<int> geometry_start;
  Vector<int> geometry_size;
  /** For the debug output of the constraint set. */
  Vector<int> affected_geo_indices;

  /** All points of all participating geometries, in one index space. */
  Vector<int> point_geo;
  Vector<int> point_local;
  Vector<float> point_radius;
  /** Extra contact distance of the point's geometry (the `Contact Thickness` of `Init Puncliga Data`). */
  Vector<float> point_thickness;
  /** Inverse mass per point; zero for a hard pinned point (see the constraint set). */
  Vector<float> point_inv_mass;
  /** Concatenated positions and rest positions of all those points. */
  Array<float3> positions;
  Array<float3> rest_positions;

  /**
   * Sorted list of the points that are topologically connected to a point (the corners of the
   * faces it belongs to and the edges it is part of). A cloth would push its own triangles apart
   * and every element of a soft body would inflate without this.
   */
  Array<int> bond_offsets;
  Array<int> bond_indices;

  Array<int2> point_pairs;
  Array<float> min_distances;
  Array<float> compliances;
  Array<float> lambdas;
  /**
   * Connected island of every point (union-find over the bonds). The rest distance filter below is
   * only valid inside one island: two points of the same sheet that lay next to each other in the
   * rest pose must not push each other apart, but two separate sheets must collide even if they were
   * modeled a millimeter apart (a stack of cloth layers).
   */
  Array<int> island_ids;
  xpbd::ConstraintColoring coloring;
  std::optional<xpbd::PointContactConstraintSet> constraint_set;

  bool is_empty() const
  {
    return point_geo.is_empty();
  }
};

/** Packed key of an unordered segment pair, used by the rest pose filter. */
static uint64_t hair_pair_key(const int segment_i, const int segment_j)
{
  const uint32_t a = uint32_t(std::min(segment_i, segment_j));
  const uint32_t b = uint32_t(std::max(segment_i, segment_j));
  return (uint64_t(a) << 32) | uint64_t(b);
}

/**
 * Open addressing hash set of packed segment pairs (see #hair_pair_key).
 *
 * The rest pose filter is queried once for every candidate of the broad phase. The previous
 * implementation kept the keys sorted and ran a binary search (about 20 comparisons for a large
 * hairstyle); with open addressing and a 50% load factor a query is usually one or two probes.
 *
 * The table size is rounded up to a power of two, which lets the probe sequence wrap with a mask.
 * `EMPTY` is `UINT64_MAX`, which can never be a valid key because the low word is a segment index.
 */
class HairPairSet {
 public:
  static constexpr uint64_t EMPTY = ~uint64_t(0);

  void build(const Span<uint64_t> pairs)
  {
    keys_.reinitialize(0);
    mask_ = 0;
    if (pairs.is_empty()) {
      return;
    }
    int64_t capacity = 16;
    while (capacity < int64_t(pairs.size()) * 2) {
      capacity <<= 1;
    }
    keys_.reinitialize(capacity);
    keys_.fill(EMPTY);
    mask_ = uint32_t(capacity - 1);
    for (const uint64_t key : pairs) {
      uint32_t slot = uint32_t(this->hash(key)) & mask_;
      while (keys_[slot] != EMPTY) {
        slot = (slot + 1) & mask_;
      }
      keys_[slot] = key;
    }
  }

  bool contains(const uint64_t key) const
  {
    if (keys_.is_empty()) {
      return false;
    }
    uint32_t slot = uint32_t(this->hash(key)) & mask_;
    while (true) {
      const uint64_t stored = keys_[slot];
      if (stored == key) {
        return true;
      }
      if (stored == EMPTY) {
        return false;
      }
      slot = (slot + 1) & mask_;
    }
  }

 private:
  Array<uint64_t> keys_;
  uint32_t mask_ = 0;

  static uint64_t hash(uint64_t key)
  {
    /* Fibonacci hashing: the high 32 bits (the smaller segment index) are the coarse part, the
     * multiplicative mixing spreads both halves over the table. */
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    return key;
  }
};

/**
 * Segment/segment contacts of the hair strands with themselves (see
 * #xpbd::SegmentContactConstraintSet).
 *
 * A hair strand is a chain of segments. Every segment of every curve geometry takes part in one
 * index space. The pairs are built once per frame with a displacement margin (the strands can move
 * between the substeps without rebuilding the broad phase), and the rest pose is used to skip pairs
 * that already touch in the rest pose (the roots of neighbouring strands are always close), so the
 * pass does not explode a combed hairstyle.
 *
 * All arrays are flat and only indexed by integers, which is what a future device implementation
 * needs: `pair_points` / `pair_min_distances` / `pair_lambdas` can be uploaded as SSBOs and solved
 * with one invocation per pair, mirroring `KERNEL_CONTACT_POSITION` of the fluid.
 */
struct HairSelfContacts {
  /** The participating curve geometries, with the range of the global points of each. */
  Vector<int> geometry_data_key;
  Vector<int> geometry_start;
  Vector<int> geometry_size;
  Vector<int> affected_geo_indices;

  Vector<int> point_geo;
  Vector<int> point_local;
  Vector<float> point_inv_mass;
  Array<float3> positions;
  Array<float3> rest_positions;

  /** Segments, one per consecutive pair of points of a curve. */
  Vector<int> segment_geometry;
  Vector<int> segment_strand;
  Vector<int> segment_point_a;
  Vector<int> segment_point_b;
  Vector<float> segment_radius;
  /** Extra contact distance of the segment's geometry (`Hair Contact Thickness`). */
  Vector<float> segment_thickness;
  /** Whether segments of this strand may collide with each other (`Same Strand Collision`). */
  Vector<uint8_t> segment_same_strand;
  /**
   * Upper bound of how far the two endpoints of the segment can move during the frame. The broad
   * phase is built once per frame, so a pair that is still this much apart can become a contact in a
   * later substep and has to be kept. This makes the pair list conservative instead of re-collecting
   * it every substep.
   */
  Vector<float> segment_margin;

  /** Hash set of the segment pairs that already touch in the rest pose. */
  HairPairSet rest_pairs;
  int rest_pair_count = 0;

  /** Contacts, four global point indices per pair (two segments). */
  Array<int> pair_offsets;
  Vector<int> pair_points;
  Vector<int2> pair_segments;
  Vector<float> pair_min_distances;
  Vector<float> pair_compliances;
  Array<float> pair_lambdas;
  xpbd::ConstraintColoring coloring;
  std::optional<xpbd::SegmentContactConstraintSet> constraint_set;

  bool is_empty() const
  {
    return point_geo.is_empty();
  }
};

/**
 * Process wide cache of the rest pose pair sets.
 *
 * The rest pose of a groom does not change while the hair is simulated, but the pair set costs
 * hundreds of milliseconds for a large hairstyle (a 10k strand groom has more than a million pairs
 * that already touch in the rest pose). The cache is keyed by a content hash of the rest positions
 * and radii, so an edited or animated groom simply misses and rebuilds.
 *
 * Only one entry is kept: alternating between two large grooms rebuilds each frame, which is the
 * uncached behaviour and still correct.
 */
struct HairRestPairCache {
  std::mutex mutex;
  uint64_t key = 0;
  int pair_count = 0;
  HairPairSet pairs;
};

static HairRestPairCache &hair_rest_pair_cache()
{
  static HairRestPairCache cache;
  return cache;
}

static uint64_t hair_rest_pair_key_hash(const HairSelfContacts &contacts)
{
  uint64_t hash = 1469598103934665603ULL;
  const auto mix = [&](const void *data, const size_t size) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; i++) {
      hash ^= uint64_t(bytes[i]);
      hash *= 1099511628211ULL;
    }
  };
  const int segments_num = contacts.segment_geometry.size();
  const int points_num = contacts.rest_positions.size();
  mix(&segments_num, sizeof(segments_num));
  mix(&points_num, sizeof(points_num));
  mix(contacts.rest_positions.data(), contacts.rest_positions.size() * sizeof(float3));
  mix(contacts.segment_radius.data(), contacts.segment_radius.size() * sizeof(float));
  mix(contacts.segment_thickness.data(), contacts.segment_thickness.size() * sizeof(float));
  return hash;
}

/** Squared closest distance between two segments of \a contacts in the given point space. */
static float hair_segment_distance_squared(const HairSelfContacts &contacts,
                                           const Span<float3> &points,
                                           const int segment_i,
                                           const int segment_j,
                                           float &r_s,
                                           float &r_t)
{
  const float3 p0 = points[contacts.segment_point_a[segment_i]];
  const float3 p1 = points[contacts.segment_point_b[segment_i]];
  const float3 q0 = points[contacts.segment_point_a[segment_j]];
  const float3 q1 = points[contacts.segment_point_b[segment_j]];
  xpbd::closest_points_on_segments(p0, p1, q0, q1, r_s, r_t);
  const float3 qa = math::interpolate(p0, p1, r_s);
  const float3 qb = math::interpolate(q0, q1, r_t);
  return math::length_squared(qb - qa);
}

/**
 * Greedy coloring specialized for hair segment pairs.
 *
 * The generic `color_constraints__binary` builds a multi value map from every segment to all its
 * constraints and allocates a lot for large hairstyles (measured: 1.8 s for 710k pairs). The pair
 * builder already guarantees that two constraints only conflict when they share a *segment*, so a
 * per segment color bitmask is enough: for a new pair, the union of the two masks is the set of
 * colors that cannot be used, and the lowest free bit is the color.
 *
 * 8 words = 512 colors. That is far more than the degree of a hair segment in practice; if a scene
 * ever exceeds it, the generic coloring is used as a fallback.
 */
static xpbd::ConstraintColoring color_hair_segment_pairs(const Span<int2> pairs,
                                                         const int segments_num,
                                                         LinearAllocator<> &memory)
{
  constexpr int WORDS = 8;
  Array<uint64_t> masks(segments_num * WORDS, uint64_t(0));
  Array<int> colors(pairs.size());
  for (const int pair_i : pairs.index_range()) {
    const int segment_i = pairs[pair_i].x;
    const int segment_j = pairs[pair_i].y;
    int color = -1;
    for (const int word : IndexRange(WORDS)) {
      const uint64_t used = masks[segment_i * WORDS + word] | masks[segment_j * WORDS + word];
      const uint64_t available = ~used;
      if (available != 0) {
        const int bit = int(bitscan_forward_uint64(available));
        color = word * 64 + bit;
        const uint64_t bit_mask = uint64_t(1) << bit;
        masks[segment_i * WORDS + word] |= bit_mask;
        masks[segment_j * WORDS + word] |= bit_mask;
        break;
      }
    }
    if (color < 0) {
      return xpbd::color_constraints__binary(pairs, memory);
    }
    colors[pair_i] = color;
  }

  xpbd::ConstraintColoring coloring;
  if (colors.is_empty()) {
    return coloring;
  }
  const int colors_num = *std::max_element(colors.begin(), colors.end()) + 1;
  Array<Vector<int>> color_indices(colors_num);
  for (const int pair_i : colors.index_range()) {
    color_indices[colors[pair_i]].append(pair_i);
  }
  for (const int color_i : color_indices.index_range()) {
    coloring.colors.append(IndexMask::from_indices<int>(color_indices[color_i], memory));
  }
  return coloring;
}

struct SubstepInterval {
  int current_i;
  float interpolate_begin;
  float interpolate_end;

  SubstepInterval(const int substeps,
                  const int current_i,
                  const float interpolation_begin,
                  const float interpolation_end)
      : current_i(current_i),
        interpolate_begin(float(current_i) / substeps * (interpolation_end - interpolation_begin) +
                          interpolation_begin),
        interpolate_end(float(current_i + 1) / substeps *
                            (interpolation_end - interpolation_begin) +
                        interpolation_begin)
  {
  }
};

struct MeshContactId {
  int mesh_collider_i;
  int point_i;

  uint64_t hash() const
  {
    return get_default_hash(this->mesh_collider_i, this->point_i);
  }

  friend bool operator==(const MeshContactId &a, const MeshContactId &b) = default;
};

struct ExternalFaceContacts {
  Map<MeshContactId, int> mesh_contact_indices;

  Vector<int> points;
  Vector<float> point_radii;
  Vector<float3> positions_on_face;
  /* The movement of the collider in the current substep. */
  Vector<float3> collider_motion;
  Vector<float3> collider_velocities;
  Vector<float3> face_normals;
  Vector<float> face_margins;
  Vector<float> static_frictions;
  Vector<float> dynamic_frictions;
  Vector<float> compliance_terms;
  Vector<float> error_scales;

  Vector<bool> active_states;
  Vector<float> lambdas_normal;
  Vector<float> lambdas_friction;

  void init_or_preserve_state(const ExternalFaceContacts &prev_contacts,
                              const std::optional<int> &prev_i)
  {
    if (prev_i) {
      this->active_states.append(prev_contacts.active_states[*prev_i]);
      this->lambdas_normal.append(prev_contacts.lambdas_normal[*prev_i]);
      this->lambdas_friction.append(prev_contacts.lambdas_friction[*prev_i]);
    }
    else {
      this->active_states.append(false);
      this->lambdas_normal.append(0.0f);
      this->lambdas_friction.append(0.0f);
    }
  }
};

struct ExternalEdgeContacts {
  Map<MeshContactId, int> mesh_contact_indices;

  Vector<int2> point_pairs;
  Vector<float2> point_radii;
  Vector<float3> positions_on_edge;
  /* The movement of the collider in the current substep. */
  Vector<float3> collider_motion;
  Vector<float3> collider_velocities;
  Vector<float3> edge_directions;
  Vector<float3> edge_normals;
  Vector<float> edge_margins;
  Vector<float> static_frictions;
  Vector<float> dynamic_frictions;
  Vector<float> compliance_terms;
  Vector<float> error_scales;

  Vector<bool> active_states;
  Vector<float> point_mix_factors;
  Vector<float> lambdas_normal;
  Vector<float> lambdas_friction;

  void init_or_preserve_state(const ExternalEdgeContacts &prev_contacts,
                              const std::optional<int> &prev_i)
  {
    if (prev_i) {
      this->active_states.append(prev_contacts.active_states[*prev_i]);
      this->point_mix_factors.append(prev_contacts.point_mix_factors[*prev_i]);
      this->lambdas_normal.append(prev_contacts.lambdas_normal[*prev_i]);
      this->lambdas_friction.append(prev_contacts.lambdas_friction[*prev_i]);
    }
    else {
      this->active_states.append(false);
      this->point_mix_factors.append(0.0f);
      this->lambdas_normal.append(0.0f);
      this->lambdas_friction.append(0.0f);
    }
  }
};

/**
 * Contacts between the fluid points and the triangles of the coupled simulated geometries (a cloth
 * or a soft body that is part of the same simulation). The contacts themselves are plain
 * `ExternalFaceContacts`, solved by the same constraint set as the contacts with a collider mesh.
 *
 * The extra arrays say which triangle produced the contact and where on it the closest point is,
 * so the impulse can be given back to the vertices of the triangle after the solve.
 */
struct CoupledFaceContacts {
  ExternalFaceContacts contacts;
  /** Vertices of the triangle of each contact, -1 when the contact has no triangle. */
  Vector<int3> face_vertices;
  Vector<float3> face_barycentric;
  Vector<int> face_geometry;
  /**
   * Part of the normal multiplier that was already given back to the surface, so the reaction only
   * applies the difference (the multiplier accumulates over the iterations of a substep).
   */
  Vector<float> applied_lambdas;

  void clear()
  {
    contacts = {};
    face_vertices.clear();
    face_barycentric.clear();
    face_geometry.clear();
    applied_lambdas.clear();
  }
};

struct ChunkData {
  Vector<xpbd::ConstraintSet *> static_constraints;
  Vector<xpbd::VelocityConstraintSet *> static_velocity_constraints;

  Vector<PinPositionConstraintChunkUsage> pin_position_constraints;
  Vector<PinRotationConstraintChunkUsage> pin_rotation_constraints;

  ExternalFaceContacts external_face_contacts;
  ExternalEdgeContacts external_edge_contacts;

  /**
   * Contacts between the fluid points and the triangles of the coupled simulated geometries. They
   * are solved like the contacts with a collider mesh, but the triangles move with the simulation,
   * so the impulse has to be given back to their vertices (see `CoupledFaceContacts`).
   */
  CoupledFaceContacts coupled_face_contacts;

  FluidChunkData fluid;
};

struct ConstraintsInfo {
  Vector<MeshCollider> mesh_colliders;
  Vector<CollisionContacts> collision_contacts;
  Vector<DampingConstraint> damping_constraints;
  Vector<RodStretchShearConstraint> rod_stretch_shear_constraints;
  Vector<RodBendTwistConstraint> rod_bend_twist_constraints;
  Vector<PinPositionConstraint> pin_position_constraints;
  Vector<PinRotationConstraint> pin_rotation_constraints;
  Vector<EdgeLengthConstraint> edge_length_constraints;
  Vector<VolumeConstraint> volume_constraints;
  Vector<CrossEdgeLengthConstraint> cross_edge_length_constraints;
  Vector<FluidConstraint> fluid_constraints;
};

class XpbdSolverStep {
 public:
  struct Result {
    /* Average relative residual error of all constraints. The average absolute error of a specific
     * constraint type is scaled by a threshold value, anything below the threshold is considered
     * "solved". This allows combining constraint residuals of different types into a single value.
     */
    float total_residual_error;
  };

 private:
  struct TLS {
    ResourceScope scope;
    LinearAllocator<> &allocator;

    TLS() : scope(1024), allocator(scope.allocator()) {}
  };

  threading::EnumerableThreadSpecific<TLS> tls_;

  /** The simulation world that is being modified. */
  Bundle &world_;

  const int substeps_;
  const float sub_delta_time_;
  float substep_compliance_factor_;

  /** Whether the solid points of the world collide with each other (see #SelfContacts). */
  const bool self_collision_;
  /** Extra distance kept between two colliding points, on top of their two radii. */
  const float contact_thickness_;

  /** Whether the hair segments collide with each other (see #HairSelfContacts). */
  const bool hair_self_collision_;
  /** Extra distance kept between two hair segments, on top of their two radii. */
  const float hair_contact_thickness_;
  /** Whether segments of the same strand may collide with each other. */
  const bool hair_same_strand_collision_;

  /* Fine grained profile of the substep loop (only filled when XPBD_PROFILE is set). */
  double prof_coupled_ = 0.0;
  double prof_constraints_ = 0.0;
  double prof_iterations_ = 0.0;
  double prof_velocity_ = 0.0;
  double prof_final_ = 0.0;

  /* Sum of the length of every reaction offset applied to a solid vertex, per frame. Used by the
   * `XPBD_REACTION_DEBUG` measurement to compare the CPU and device coupling. */
  std::atomic<double> debug_reaction_density_{0.0};
  std::atomic<double> debug_reaction_contact_{0.0};

  const float interpolation_begin_;
  const float interpolation_end_;

  const float4x4 world_to_simulation_;

  std::string geometry_tag_filter_;

  int constraint_iterations_;

  /**
   * Backend request of the `HW Accelerate` zone that is evaluating this solver, 0 outside of a
   * zone. The values are `geometry::ha::HaScopeBackend` (1 = auto, 2 = strict, 3 = force CPU).
   */
  const int gpu_backend_request_;

  Geometries geometries_;
  ConstraintsInfo constraints_;
  Array<ChunkData> chunks_data_;

  /**
   * The other simulated geometries of the world (cloth, soft bodies), prepared as collision
   * surfaces for the fluid. The meshes are copies that only share the topology with the world
   * geometries; their positions are updated from the solver every substep.
   */
  Vector<CoupledSurface> coupled_surfaces_;

  /**
   * Collision surfaces of the cloth and soft bodies for the hair, plus the segment/triangle
   * contacts between them. CPU only for now.
   */
  Vector<CoupledSurface> hair_solid_surfaces_;
  Vector<float> hair_solid_surface_max_radius;
  /** First global point index of each surface (its vertices follow the hair points). */
  Vector<int> hair_solid_surface_point_start_;
  HairSolidContacts hair_solid_contacts_;
  bool hair_solid_active_ = false;
  std::unique_ptr<ResourceScope> hair_solid_memory_;

  /** Compliant guide/child attachments (two way). */
  GuideTetherContacts guide_tether_contacts_;
  bool guide_tether_active_ = false;
  std::unique_ptr<ResourceScope> guide_tether_memory_;

  /** Point to point contacts of the solids with themselves (empty when disabled). */
  SelfContacts self_contacts_;
  /** Whether at least one geometry enabled `Self Collision` for this frame. */
  bool self_contacts_active_ = false;
  /**
   * Memory of the contact colors of the current substep. The colors are rebuilt together with the
   * pairs, so the memory of the previous substep is released instead of accumulating over the
   * substeps of the frame.
   */
  std::unique_ptr<ResourceScope> self_contacts_memory_;

  /** Segment to segment contacts of the hair with itself (empty when disabled). */
  HairSelfContacts hair_contacts_;
  /** Whether at least one hair geometry enabled `Hair Self Collision` for this frame. */
  bool hair_contacts_active_ = false;
  /** Memory of the contact colors of the current substep (see #self_contacts_memory_). */
  std::unique_ptr<ResourceScope> hair_contacts_memory_;

  Mutex field_evaluators_mutex_;

  Result result_;
  Mutex warnings_mutex_;
  VectorSet<std::pair<NodeWarningType, std::string>> warnings_;

  MultiValueMap<std::string, std::string> nested_bundle_paths_;

 public:
  XpbdSolverStep(Bundle &world,
                 const float total_delta_time,
                 const int substeps,
                 const int constraint_iterations,
                 const bool self_collision,
                 const float contact_thickness,
                 const bool hair_self_collision,
                 const float hair_contact_thickness,
                 const bool hair_same_strand_collision,
                 const float interpolation_begin,
                 const float interpolation_end,
                 const StringRef geometry_tag_filter,
                 const float4x4 &simulation_to_world,
                 const int gpu_backend_request)
      : world_(world),
        substeps_(substeps),
        sub_delta_time_(total_delta_time / substeps_),
        self_collision_(self_collision),
        contact_thickness_(std::max(contact_thickness, 0.0f)),
        hair_self_collision_(hair_self_collision),
        hair_contact_thickness_(std::max(hair_contact_thickness, 0.0f)),
        hair_same_strand_collision_(hair_same_strand_collision),
        interpolation_begin_(interpolation_begin),
        interpolation_end_(interpolation_end),
        world_to_simulation_(math::invert(simulation_to_world)),
        geometry_tag_filter_(geometry_tag_filter),
        constraint_iterations_(constraint_iterations),
        gpu_backend_request_(gpu_backend_request)
  {
    substep_compliance_factor_ = math::safe_rcp(pow2f(sub_delta_time_));
  }

  void do_step()
  {
    /* Coarse phase profile, enabled with XPBD_PROFILE=1. */
    const bool profile = std::getenv("XPBD_PROFILE") != nullptr;
    double ms_gather = 0.0;
    double ms_prepare = 0.0;
    double ms_simulate = 0.0;
    double ms_writeback = 0.0;
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto ms_since = [](const std::chrono::steady_clock::time_point &start) {
      return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
          .count();
    };
    std::chrono::steady_clock::time_point phase_start = now();
    this->gather_nested_bundle_paths();
    this->gather_from_world__geometries();
    this->gather_from_world__fluids();
    this->prepare_geometry_chunks();
    this->prepare_inverse_masses();
    this->prepare_inverse_moments_of_inertia();
    this->prepare_chunk_fluids();

    this->gather_from_world__mesh_colliders();
    this->gather_from_world__stretch_shear_constraints();
    this->gather_from_world__bend_twist_constraints();
    this->gather_from_world__edge_length_constraints();
    this->gather_from_world__volume_constraints();
    this->gather_from_world__cross_edge_length_constraints();
    this->gather_from_world__damping();
    this->gather_from_world__pin_positions();
    this->gather_from_world__pin_rotations();
    this->gather_from_world__collision_contacts();
    this->gather_from_world__dilapido_forces();

    if (profile) {
      ms_gather = ms_since(phase_start);
      phase_start = now();
    }

    this->parallel_for_each_chunk(32, [&](const int chunk_i) {
      TLS &tls = tls_.local();
      this->create_chunk_constraints__rod_stretch_shear(tls, chunk_i);
      this->create_chunk_constraints__rod_bend_twist(tls, chunk_i);
      this->create_chunk_constraints__damping(tls, chunk_i);
      this->create_chunk_constraints__pin_positions(tls, chunk_i);
      this->create_chunk_constraints__pin_rotations(tls, chunk_i);
    });

    this->prepare_fluid_boundaries();
    this->prepare_coupled_surfaces();
    this->prepare_self_contacts();
    this->prepare_hair_contacts();
    this->prepare_hair_solid_contacts();
    this->prepare_guide_tethers();
    this->prepare_fluid_gpu();

    if (profile) {
      ms_prepare = ms_since(phase_start);
      phase_start = now();
    }

    if (std::getenv("XPBD_DEBUG_NAN") != nullptr) {
      int static_constraints = 0;
      for (const int data_key_i : geometries_.data_keys.index_range()) {
        static_constraints += int(geometries_.data[data_key_i]->static_constraints.size());
      }
      printf("XPBD_DEBUG_NAN do_step: geometries %d chunks %d substeps %d iterations %d dt %g "
             "chunk_path %d static_constraints %d\n",
             int(geometries_.geometry_sets.size()),
             int(geometries_.chunks.size()),
             substeps_,
             constraint_iterations_,
             double(sub_delta_time_),
             this->support_chunk_simulation() ? 1 : 0,
             static_constraints);
    }
    result_ = this->do_simulation();
    if (std::getenv("XPBD_DEBUG_NAN") != nullptr) {
      for (const int data_key_i : geometries_.data_keys.index_range()) {
        GeometryData &geo_data = *geometries_.data[data_key_i];
        bke::AttributeAccessor attributes = geo_data.attributes;
        const VArray<float3> positions = *attributes.lookup_or_default<float3>(
            "position", AttrDomain::Point, float3(0.0f));
        int bad = 0;
        for (const int i : positions.index_range()) {
          const float3 &p = positions[i];
          bad += (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) ? 0 : 1;
        }
        printf("XPBD_DEBUG_NAN after do_step: geometry %d has %d non finite positions of %d\n",
               data_key_i,
               bad,
               int(positions.size()));
      }
    }

    if (profile) {
      ms_simulate = ms_since(phase_start);
      phase_start = now();
    }
    this->report_fluid_density_warnings();
    this->write_back__pin_positions();
    this->write_back__rod_stretch_shear();

    this->free_coupled_surfaces();
    this->free_hair_solid_surfaces();

    this->finish_common_attribute_writers();
    this->write_back_geometries_to_world();

    this->write_back__contacts();

    if (std::getenv("XPBD_REACTION_DEBUG") != nullptr) {
      printf("XPBD_REACTION density=%.5f contact=%.5f\n",
             debug_reaction_density_.load(),
             debug_reaction_contact_.load());
      fflush(stdout);
      debug_reaction_density_.store(0.0);
      debug_reaction_contact_.store(0.0);
    }

    if (profile) {
      ms_writeback = ms_since(phase_start);
      printf("XPBD_PROFILE gather=%.1f prepare=%.1f simulate=%.1f write=%.1f total=%.1f ms "
             "(chunks=%d substeps=%d iterations=%d)\n",
             ms_gather,
             ms_prepare,
             ms_simulate,
             ms_writeback,
             ms_gather + ms_prepare + ms_simulate + ms_writeback,
             int(this->geometries_.chunks.size()),
             this->substeps_,
             this->constraint_iterations_);
      printf("XPBD_PROFILE   substeps: coupled=%.1f constraints=%.1f iterations=%.1f "
             "velocity=%.1f final=%.1f ms\n",
             prof_coupled_,
             prof_constraints_,
             prof_iterations_,
             prof_velocity_,
             prof_final_);
      prof_coupled_ = prof_constraints_ = prof_iterations_ = prof_velocity_ = prof_final_ = 0.0;
      fflush(stdout);
    }
  }

  const Result &result() const
  {
    return result_;
  }
  Span<std::pair<NodeWarningType, std::string>> warnings() const
  {
    return warnings_;
  }

 private:
  static float error_scale_from_threshold(const float error_threshold)
  {
    return 1.0f / std::max(math::square(error_threshold), 1e-12f);
  }

  void gather_nested_bundle_paths()
  {
    foreach_nested_bundle_item(world_,
                               [&](const Span<BundleKey> path, const BundleItemValue &value) {
                                 const BundlePtr *bundle_ptr = value.as_pointer<BundlePtr>();
                                 if (!bundle_ptr || !*bundle_ptr) {
                                   return;
                                 }
                                 const Bundle &bundle = **bundle_ptr;
                                 const std::optional<StringRef> type = bundle.type();
                                 if (!type.has_value()) {
                                   return;
                                 }
                                 nested_bundle_paths_.add_as(*type, Bundle::combine_path(path));
                               });
    /* Ensure the order is deterministic. */
    for (MutableSpan<std::string> paths : nested_bundle_paths_.values()) {
      std::ranges::sort(paths);
    }
  }

  void gather_from_world__geometries()
  {
    TLS &tls = tls_.local();
    /* Gather geometry sets to process from the world. */
    Vector<std::string> paths = gather_bundle_paths_by_data_type(world_, SOCK_GEOMETRY);
    /* Ensure the order is deterministic. */
    std::ranges::sort(paths);
    for (const StringRef path : paths) {
      GeometrySetData geo_set_data;
      geo_set_data.path = path;
      geo_set_data.geometry = std::move(*world_.lookup_path_for_write_ptr<GeometrySet>(path));
      if (geo_set_data.geometry.has_bundle()) {
        const Bundle &bundle_in_geo = *geo_set_data.geometry.bundle();
        if (const std::optional<GListPtr> tags_list_ptr = bundle_in_geo.lookup_path<GListPtr>(
                "tags"))
        {
          if (*tags_list_ptr) {
            const GList &tags_list = **tags_list_ptr;
            if (tags_list.cpp_type().is<std::string>()) {
              tags_list.typed<std::string>().foreach(
                  [&](const std::string &tag) { geo_set_data.tags.add(tag); });
            }
          }
        }
      }
      if (!tag_filter_matches(geometry_tag_filter_, geo_set_data.tags)) {
        continue;
      }
      geometries_.geometry_sets.append(std::move(geo_set_data));
    }

    /* Gather individual components that should be simulated. There may be more than geometry sets
     * because each geometry set could contain e.g. a mesh and curves. */
    for (const int geo_bundle_i : geometries_.geometry_sets.index_range()) {
      GeometrySetData &geo_set_data = geometries_.geometry_sets[geo_bundle_i];
      GeometrySet &geometry = geo_set_data.geometry;
      for (bke::GeometryComponent::Type type : {bke::GeometryComponent::Type::Mesh,
                                                bke::GeometryComponent::Type::PointCloud,
                                                bke::GeometryComponent::Type::Curve,
                                                bke::GeometryComponent::Type::Instance})
      {
        if (!geometry.has(type)) {
          continue;
        }
        bke::GeometryComponent &component = geometry.get_component_for_write(type);
        geometries_.data_keys.add_new({geo_bundle_i, type, std::nullopt});
        GeometryData &geo_data = tls.scope.construct<GeometryData>(
            *component.attributes_for_write());
        geo_data.curves = type == bke::GeometryComponent::Type::Curve ?
                              &geometry.get_curves_for_write()->geometry.wrap() :
                              nullptr;
        geo_data.domain = type == bke::GeometryComponent::Type::Instance ? AttrDomain::Instance :
                                                                           AttrDomain::Point;
        geometries_.data.append(&geo_data);
      }
      if (geometry.has_grease_pencil()) {
        using namespace blender::bke::greasepencil;
        GreasePencil &grease_pencil = *geometry.get_grease_pencil_for_write();
        for (const int layer_i : grease_pencil.layers().index_range()) {
          Layer &layer = grease_pencil.layer(layer_i);
          Drawing *drawing = grease_pencil.get_eval_drawing(layer);
          if (!drawing) {
            continue;
          }
          bke::CurvesGeometry &curves = drawing->strokes_for_write();
          geometries_.data_keys.add_new(
              {geo_bundle_i, bke::GeometryComponent::Type::Curve, layer_i});
          GeometryData &geo_data = tls.scope.construct<GeometryData>(
              curves.attributes_for_write());
          geo_data.domain = AttrDomain::Point;
          geo_data.curves = &curves;
          geometries_.data.append(&geo_data);
        }
      }
    }
    geometries_.total_points_num = 0;
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      const AttrDomain domain = geo_data.domain;
      geo_data.size = geo_data.attributes.domain_size(domain);
      geometries_.total_points_num += geo_data.size;
      geo_data.temp_positions.reinitialize(geo_data.size);
      geo_data.temp_rotations.reinitialize(geo_data.size);
      geo_data.position_attr = this->ensure_attribute<float3>(
          geo_data.attributes, attribute_names::position, domain);
      geo_data.velocity_attr = this->ensure_attribute<float3>(
          geo_data.attributes, attribute_names::velocity, domain);
      geo_data.external_force_attr = this->lookup_attribute_default<float3>(
          data_key_i, attribute_names::external_force, domain, float3(0, 0, 0));
      geo_data.masses = this->lookup_attribute_default<float>(
          data_key_i, attribute_names::mass, domain, 1.0f);

      geo_data.is_hard_pinned.reinitialize(geo_data.size);
      geo_data.is_hard_pinned.fill(false);
      if (geo_data.attributes.contains(attribute_names::rotation)) {
        geo_data.uses_rotation = true;
        geo_data.rotation_attr = this->ensure_attribute<math::Quaternion>(
            geo_data.attributes, attribute_names::rotation, domain);
        geo_data.angular_velocity_attr = this->ensure_attribute<float3>(
            geo_data.attributes, attribute_names::angular_velocity, domain);
        geo_data.external_torque_attr = this->lookup_attribute_default<float3>(
            data_key_i, attribute_names::external_torque, domain, float3(0, 0, 0));
        geo_data.moments_of_inertia = this->lookup_attribute_default<float3>(
            data_key_i, attribute_names::moment_of_inertia, domain, float3(1.0f));
      }
    }
  }

  void ensure_friction_loaded(const int data_key_i)
  {
    GeometryData &geo_data = *geometries_.data[data_key_i];
    geo_data.load_friction_mutex.ensure([&]() {
      geo_data.static_frictions = this->lookup_attribute_default<float>(
          data_key_i, attribute_names::static_friction, geo_data.domain, 0.0f);
      geo_data.dynamic_frictions = this->lookup_attribute_default<float>(
          data_key_i, attribute_names::dynamic_friction, geo_data.domain, 0.0f);
    });
  }

  void ensure_radius_loaded(const int data_key_i, const float default_radius = 0.0f)
  {
    GeometryData &geo_data = *geometries_.data[data_key_i];
    geo_data.load_radius_mutex.ensure([&]() {
      if (default_radius <= 0.0f) {
        geo_data.radii = this->lookup_attribute_default<float>(
            data_key_i, attribute_names::radius, geo_data.domain, 0.0f);
      }
      else {
        /* Fluids get their radius from the fluid bundle, so a missing attribute is not a
         * problem and should not produce a warning. */
        const VArray<float> radius_attr = *geo_data.attributes.lookup<float>(
            attribute_names::radius, geo_data.domain);
        geo_data.radii = radius_attr ? radius_attr :
                                       VArray<float>::from_single(default_radius, geo_data.size);
      }
      geo_data.prev_radii = geo_data.radii;
    });
  }

  /**
   * Gathers `Blender.Fluid` bundles from the world. Every geometry that the bundle applies to
   * (based on its tag filter) is turned into a position based fluid.
   */
  void gather_from_world__fluids()
  {
    const Span<std::string> paths = nested_bundle_paths_.lookup_as(FluidBundle::name);
    for (const StringRef path : paths) {
      const BundlePtr *bundle_ptr = world_.lookup_path_ptr<BundlePtr>(path);
      if (!bundle_ptr || !*bundle_ptr) {
        continue;
      }
      const Bundle &bundle = **bundle_ptr;

      if (std::getenv("BLENDER_FLUID_DEBUG") != nullptr) {
        const std::string filter =
            bundle.lookup<std::string>(*BundleKey::from_str("filter")).value_or("");
        printf(
            "FLUID_DEBUG bundle at '%s' filter '%s'\n", std::string(path).c_str(), filter.c_str());
      }

      FluidConstraint fluid;
      fluid.path = path;
      fluid.rest_density =
          bundle.lookup<float>(*BundleKey::from_str("rest_density")).value_or(1000.0f);
      fluid.compliance = std::max(
          0.0f, bundle.lookup<float>(*BundleKey::from_str("compliance")).value_or(0.0f));
      fluid.viscosity = std::max(
          0.0f, bundle.lookup<float>(*BundleKey::from_str("viscosity")).value_or(0.0f));
      fluid.damping = std::max(
          0.0f, bundle.lookup<float>(*BundleKey::from_str("damping")).value_or(1.0f));
      fluid.vorticity = std::max(
          0.0f, bundle.lookup<float>(*BundleKey::from_str("vorticity")).value_or(0.0f));
      fluid.surface_tension = std::max(
          0.0f, bundle.lookup<float>(*BundleKey::from_str("surface_tension")).value_or(0.0f));
      fluid.boundary_particles =
          bundle.lookup<bool>(*BundleKey::from_str("boundary_particles")).value_or(true);
      fluid.boundary_friction = math::clamp(
          bundle.lookup<float>(*BundleKey::from_str("boundary_friction")).value_or(0.5f),
          0.0f,
          1.0f);
      fluid.coupling = math::clamp(
          bundle.lookup<float>(*BundleKey::from_str("coupling")).value_or(1.0f), 0.0f, 1.0f);
      fluid.coupling_compliance = std::max(
          0.0f, bundle.lookup<float>(*BundleKey::from_str("coupling_compliance")).value_or(0.0f));
      fluid.negative_pressure_limit = math::clamp(
          bundle.lookup<float>(*BundleKey::from_str("negative_pressure_limit")).value_or(0.0f),
          0.0f,
          1.0f);
      fluid.artificial_pressure = std::max(
          0.0f, bundle.lookup<float>(*BundleKey::from_str("artificial_pressure")).value_or(0.0f));
      fluid.particle_radius =
          bundle.lookup<float>(*BundleKey::from_str("particle_radius")).value_or(0.05f);
      fluid.error_threshold =
          bundle.lookup<float>(*BundleKey::from_str("error_threshold")).value_or(1e-2f);
      if (!(fluid.rest_density > 0.0f)) {
        this->report(NodeWarningType::Error,
                     fmt::format("Fluid \"{}\" has a rest density of zero, ignoring it", path));
        continue;
      }

      const int constraint_i = constraints_.fluid_constraints.append_and_get_index(
          std::move(fluid));
      const FluidConstraint &fluid_params = constraints_.fluid_constraints[constraint_i];

      for (const int data_key_i : geometries_.data_keys.index_range()) {
        GeometryData &geo_data = *geometries_.data[data_key_i];
        if (!this->effector_applies_to_geometry(path, bundle, data_key_i)) {
          continue;
        }
        const std::string &geometry_path =
            geometries_.geometry_sets[geometries_.data_keys[data_key_i].geo_bundle_i].path;
        if (geo_data.fluid_constraint_i >= 0) {
          this->report(
              NodeWarningType::Warning,
              fmt::format("Geometry \"{}\" is affected by more than one fluid, ignoring \"{}\"",
                          geometry_path,
                          path));
          continue;
        }

        this->ensure_radius_loaded(data_key_i, fluid_params.particle_radius);
        /* Materialize the per point arrays once, they are accessed for every neighbor lookup. */
        geo_data.fluid_radii.reinitialize(geo_data.size);
        geo_data.radii.materialize(geo_data.fluid_radii);
        geo_data.fluid_masses.reinitialize(geo_data.size);
        geo_data.masses.materialize(geo_data.fluid_masses);

        /* Per point parameters, so several fluids with different settings can live in one world.
         * The bundle values are the fallback for geometries without the attributes. */
        this->materialize_float_param(data_key_i,
                                      "puncliga_rest_density",
                                      fluid_params.rest_density,
                                      1e-3f,
                                      FLT_MAX,
                                      geo_data.fluid_rest_density);
        this->materialize_float_param(data_key_i,
                                      "puncliga_viscosity",
                                      fluid_params.viscosity,
                                      0.0f,
                                      FLT_MAX,
                                      geo_data.fluid_viscosity);
        this->materialize_float_param(data_key_i,
                                      "puncliga_damping",
                                      fluid_params.damping,
                                      0.0f,
                                      FLT_MAX,
                                      geo_data.fluid_damping);
        this->materialize_float_param(data_key_i,
                                      "puncliga_vorticity",
                                      fluid_params.vorticity,
                                      0.0f,
                                      FLT_MAX,
                                      geo_data.fluid_vorticity);
        this->materialize_float_param(data_key_i,
                                      "puncliga_surface_tension",
                                      fluid_params.surface_tension,
                                      0.0f,
                                      FLT_MAX,
                                      geo_data.fluid_surface_tension);
        this->materialize_float_param(data_key_i,
                                      "puncliga_boundary_friction",
                                      fluid_params.boundary_friction,
                                      0.0f,
                                      1.0f,
                                      geo_data.fluid_boundary_friction);
        this->materialize_float_param(data_key_i,
                                      "puncliga_negative_pressure_limit",
                                      fluid_params.negative_pressure_limit,
                                      0.0f,
                                      1.0f,
                                      geo_data.fluid_negative_pressure_limit);
        this->materialize_bool_param(data_key_i,
                                     "puncliga_boundary_particles",
                                     fluid_params.boundary_particles,
                                     geo_data.fluid_use_boundary);
        this->materialize_float_param(data_key_i,
                                      "puncliga_coupling",
                                      fluid_params.coupling,
                                      0.0f,
                                      1.0f,
                                      geo_data.fluid_coupling);
        this->materialize_float_param(data_key_i,
                                      "puncliga_coupling_compliance",
                                      fluid_params.coupling_compliance,
                                      0.0f,
                                      FLT_MAX,
                                      geo_data.fluid_coupling_compliance);
        if (std::getenv("XPBD_REACTION_DEBUG") != nullptr) {
          float coupling_min = 1e30f;
          float coupling_max = -1e30f;
          for (const float c : geo_data.fluid_coupling) {
            coupling_min = std::min(coupling_min, c);
            coupling_max = std::max(coupling_max, c);
          }
          printf("XPBD_REACTION coupling min=%.3f max=%.3f n=%d\n",
                 double(coupling_min),
                 double(coupling_max),
                 int(geo_data.fluid_coupling.size()));
          fflush(stdout);
        }

        float max_radius = 0.0f;
        for (const float radius : geo_data.fluid_radii) {
          max_radius = std::max(max_radius, radius);
        }
        /* Support radius of the kernel is twice the particle radius, as usual in SPH. */
        geo_data.fluid_support_radius = 2.0f * max_radius;
        if (!(geo_data.fluid_support_radius > 0.0f)) {
          this->report(NodeWarningType::Error,
                       fmt::format("Fluid \"{}\" has a particle radius of zero on \"{}\", "
                                   "set the radius attribute or the particle radius of the fluid",
                                   path,
                                   geometry_path));
          continue;
        }
        geo_data.fluid_constraint_i = constraint_i;
      }
    }
  }

  /** Materializes a float attribute into a per point array, with a clamped uniform fallback. */
  void materialize_float_param(const int data_key_i,
                               const StringRef name,
                               const float fallback,
                               const float min_value,
                               const float max_value,
                               Array<float> &r_values)
  {
    GeometryData &geo_data = *geometries_.data[data_key_i];
    r_values.reinitialize(geo_data.size);
    const VArray<float> values = *geo_data.attributes.lookup<float>(name, geo_data.domain);
    if (!values) {
      r_values.fill(math::clamp(fallback, min_value, max_value));
      return;
    }
    values.materialize(r_values);
    for (float &value : r_values) {
      value = math::clamp(value, min_value, max_value);
    }
  }

  void materialize_bool_param(const int data_key_i,
                              const StringRef name,
                              const bool fallback,
                              Array<bool> &r_values)
  {
    GeometryData &geo_data = *geometries_.data[data_key_i];
    r_values.reinitialize(geo_data.size);
    const VArray<bool> values = *geo_data.attributes.lookup<bool>(name, geo_data.domain);
    if (!values) {
      r_values.fill(fallback);
      return;
    }
    values.materialize(r_values);
  }

  void prepare_chunk_fluids()
  {
    for (const int chunk_i : geometries_.chunks.index_range()) {
      const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
      const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
      if (std::getenv("BLENDER_FLUID_DEBUG") != nullptr) {
        const std::string &path =
            geometries_.geometry_sets[geometries_.data_keys[chunk.data_key_i].geo_bundle_i].path;
        printf("FLUID_DEBUG chunk %d: geometry '%s' size %d points %d fluid %d\n",
               chunk_i,
               path.c_str(),
               int(geo_data.size),
               int(chunk.points_range.size()),
               geo_data.fluid_constraint_i);
      }
      if (geo_data.fluid_constraint_i < 0) {
        continue;
      }
      const int points_num = chunk.points_range.size();
      if (points_num < 2) {
        continue;
      }
      FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
      fluid.active = true;
      fluid.constraint_i = geo_data.fluid_constraint_i;
      fluid.support_radius = geo_data.fluid_support_radius;
      fluid.boundary_active =
          constraints_.fluid_constraints[geo_data.fluid_constraint_i].boundary_particles;
      fluid.lambdas.reinitialize(points_num);
      fluid.lambdas.fill(0.0f);
      fluid.errors.reinitialize(points_num);
      fluid.density_errors.reinitialize(points_num);
      fluid.neighbor_counts.reinitialize(points_num);
      fluid.corrections.reinitialize(points_num);
      fluid.omegas.reinitialize(points_num);
      fluid.omega_magnitudes.reinitialize(points_num);
      fluid.solid_fractions.reinitialize(points_num);
      fluid.solid_fractions.fill(0.0f);
      fluid.coupled_push.reinitialize(points_num);
      fluid.coupled_push.fill(float3(0.0f));
    }
  }

  void gather_from_world__mesh_colliders()
  {
    const Span<std::string> paths = nested_bundle_paths_.lookup_as(MeshColliderBundle::name);
    for (const StringRef path : paths) {
      const BundlePtr *bundle_ptr = world_.lookup_path_ptr<BundlePtr>(path);
      if (!bundle_ptr || !*bundle_ptr) {
        continue;
      }
      const Bundle &bundle = **bundle_ptr;
      const Bundle *previous_bundle = this->get_previous_bundle(bundle);
      const bke::GeometrySet *geometry = bundle.lookup_ptr<bke::GeometrySet>(
          *BundleKey::from_str("geometry"));
      const float margin = bundle.lookup<float>(*BundleKey::from_str("margin")).value_or(0.0f);
      const float friction = bundle.lookup<float>(*BundleKey::from_str("friction")).value_or(0.0f);
      const float compliance =
          bundle.lookup<float>(*BundleKey::from_str("compliance")).value_or(0.0f);
      const bool deforming =
          bundle.lookup<bool>(*BundleKey::from_str("deforming")).value_or(false);
      const bool use_edge_contacts =
          bundle.lookup<bool>(*BundleKey::from_str("use_edge_contacts")).value_or(false);
      const bool is_boundary =
          bundle.lookup<bool>(*BundleKey::from_str("is_boundary")).value_or(false);
      const float error_threshold =
          bundle.lookup<float>(*BundleKey::from_str("error_threshold")).value_or(1e-3f);
      const bke::GeometrySet *prev_geometry = previous_bundle ?
                                                  previous_bundle->lookup_ptr<bke::GeometrySet>(
                                                      *BundleKey::from_str("geometry")) :
                                                  nullptr;
      if (!geometry) {
        continue;
      }

      /* Per substep sweeping: the geometry in the bundle is already baked with the transform of
       * the `transform` item, so the motion between the previous and the current frame is
       * `previous * current^-1` in the space of that geometry. Starting the substeps at the
       * previous pose and ending at the identity makes the contact solve follow the collider
       * through the frame instead of teleporting it once per frame. Without this a collider that
       * moves further than a particle radius per frame passes through the fluid in one step. */
      const bool sweep = bundle.lookup<bool>(*BundleKey::from_str("sweep")).value_or(true);
      float4x4 begin_transform = world_to_simulation_;
      const float4x4 *transform_ptr = bundle.lookup_ptr<float4x4>(
          *BundleKey::from_str("transform"));
      if (sweep && transform_ptr != nullptr) {
        const float4x4 *prev_transform_ptr = previous_bundle ?
                                                 previous_bundle->lookup_ptr<float4x4>(
                                                     *BundleKey::from_str("transform")) :
                                                 nullptr;
        if (prev_transform_ptr != nullptr) {
          const float4x4 swept = world_to_simulation_ * (*prev_transform_ptr) *
                                 math::invert(*transform_ptr);
          bool finite = true;
          for (const int column : IndexRange(4)) {
            for (const int row : IndexRange(4)) {
              finite &= std::isfinite(swept[column][row]);
            }
          }
          if (finite) {
            begin_transform = swept;
          }
        }
      }

      Vector<int> affected_data;
      for (const int data_key_i : geometries_.data_keys.index_range()) {
        if (this->effector_applies_to_geometry(path, bundle, data_key_i)) {
          affected_data.append(data_key_i);
          this->ensure_friction_loaded(data_key_i);
          this->ensure_radius_loaded(data_key_i);
        }
      }
      Vector<int> instance_id_stack;
      this->gather_colliders_in_geometry(path,
                                         world_to_simulation_,
                                         begin_transform,
                                         *geometry,
                                         prev_geometry,
                                         margin,
                                         friction,
                                         compliance,
                                         deforming,
                                         use_edge_contacts,
                                         is_boundary,
                                         error_threshold,
                                         affected_data,
                                         instance_id_stack);
    }
  }

  void gather_contacts__mesh_colliders(const int chunk_i,
                                       const float max_distance,
                                       const int solver_refs_i,
                                       const SubstepInterval &substep,
                                       const ExternalFaceContacts &prev_face_contacts,
                                       const ExternalEdgeContacts &prev_edge_contacts,
                                       ExternalFaceContacts &r_face_contacts,
                                       ExternalEdgeContacts &r_edge_contacts)
  {
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    for (const MeshColliderUsage &collider_usage : geo_data.mesh_colliders) {
      const MeshCollider &collider = constraints_.mesh_colliders[collider_usage.constraint_i];
      const float4x4 &mesh_to_local = math::interpolate(
          collider.begin_transform, collider.end_transform, substep.interpolate_end);
      const float4x4 &prev_mesh_to_local = math::interpolate(
          collider.begin_transform, collider.end_transform, substep.interpolate_begin);
      const float4x4 local_to_mesh = math::invert(mesh_to_local);

      if (std::holds_alternative<StaticMeshInfo>(collider.mesh)) {
        this->gather_contacts__mesh_collider(chunk_i,
                                             max_distance,
                                             solver_refs_i,
                                             false,
                                             substep,
                                             collider,
                                             collider_usage,
                                             mesh_to_local,
                                             prev_mesh_to_local,
                                             local_to_mesh,
                                             prev_face_contacts,
                                             prev_edge_contacts,
                                             r_face_contacts,
                                             r_edge_contacts);
      }
      else if (std::holds_alternative<DeformingMeshInfo>(collider.mesh)) {
        this->gather_contacts__mesh_collider(chunk_i,
                                             max_distance,
                                             solver_refs_i,
                                             true,
                                             substep,
                                             collider,
                                             collider_usage,
                                             mesh_to_local,
                                             prev_mesh_to_local,
                                             local_to_mesh,
                                             prev_face_contacts,
                                             prev_edge_contacts,
                                             r_face_contacts,
                                             r_edge_contacts);
      }
    }
  }

  void gather_contacts__mesh_collider(const int chunk_i,
                                      const float max_distance,
                                      const int solver_refs_i,
                                      const bool is_deforming,
                                      const SubstepInterval &substep,
                                      const MeshCollider &collider,
                                      const MeshColliderUsage &collider_usage,
                                      const float4x4 &mesh_to_local,
                                      const float4x4 &prev_mesh_to_local,
                                      const float4x4 &local_to_mesh,
                                      const ExternalFaceContacts &prev_face_contacts,
                                      const ExternalEdgeContacts &prev_edge_contacts,
                                      ExternalFaceContacts &r_face_contacts,
                                      ExternalEdgeContacts &r_edge_contacts)
  {
    const Mesh *mesh;
    const bke::BVHTreeFromMesh *corner_tris_bvh;
    const bke::BVHTreeFromMesh *edges_bvh;
    Span<float3> prev_vert_positions;
    if (is_deforming) {
      BLI_assert(std::holds_alternative<DeformingMeshInfo>(collider.mesh));
      const auto &deforming_mesh = std::get<DeformingMeshInfo>(collider.mesh);
      mesh = deforming_mesh.substep_meshes[substep.current_i + 1];
      corner_tris_bvh = &deforming_mesh.substep_corner_tris_bvh_trees[substep.current_i];
      edges_bvh = &deforming_mesh.substep_edges_bvh_trees[substep.current_i];
      prev_vert_positions = deforming_mesh.substep_meshes[substep.current_i]->vert_positions();
    }
    else {
      BLI_assert(std::holds_alternative<StaticMeshInfo>(collider.mesh));
      const auto &static_mesh = std::get<StaticMeshInfo>(collider.mesh);
      mesh = static_mesh.mesh;
      corner_tris_bvh = &static_mesh.corner_tris_bvh;
      edges_bvh = &static_mesh.edges_bvh;
    }
    const Span<float3> vert_positions = mesh->vert_positions();
    const Span<int2> edge_verts = mesh->edges();
    const Span<float3> vert_normals = mesh->vert_normals();
    const Span<int> corner_verts = mesh->corner_verts();
    const Span<int3> corner_tris = mesh->corner_tris();

    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    const Span<float3> positions =
        geometries_.solver_refs[solver_refs_i][chunk.data_key_i].positions;

    /* Only uniform scaling is correctly handled here. For non-uniform scaling, just the max
     * scale along one axis is used. */
    const float3 mesh_to_local_radius_scale = math::abs(math::to_scale(local_to_mesh));
    const float local_to_mesh_radius_factor = std::max({mesh_to_local_radius_scale.x,
                                                        mesh_to_local_radius_scale.y,
                                                        mesh_to_local_radius_scale.z});
    const float mesh_to_local_radius_factor = math::safe_rcp(local_to_mesh_radius_factor);
    const float margin_local = mesh_to_local_radius_factor * collider.margin;

    for (const int point_i : chunk.points_range) {
      if (geo_data.is_hard_pinned[point_i]) {
        continue;
      }
      const float3 &pos_local = positions[point_i];
      const float3 pos_mesh = math::transform_point(local_to_mesh, pos_local);
      const float radius_local = math::interpolate(
          geo_data.prev_radii[point_i], geo_data.radii[point_i], substep.interpolate_end);
      const float radius_mesh = local_to_mesh_radius_factor * radius_local;
      const std::optional<ClosestMeshFaceContact> contact = this->get_closest_mesh_face_contact(
          pos_mesh,
          *corner_tris_bvh,
          corner_tris,
          corner_verts,
          vert_positions,
          max_distance + radius_mesh);
      if (!contact) {
        continue;
      }

      const float3 contact_pos_local = math::transform_point(mesh_to_local, contact->nearest_pos);
      float3 prev_contact_pos_mesh;
      if (is_deforming) {
        const int3 &tri = corner_tris[contact->tri_i];
        prev_contact_pos_mesh = bke::attribute_math::mix3(
            contact->bary_coords,
            prev_vert_positions[corner_verts[tri[0]]],
            prev_vert_positions[corner_verts[tri[1]]],
            prev_vert_positions[corner_verts[tri[2]]]);
      }
      else {
        prev_contact_pos_mesh = contact->nearest_pos;
      }
      const float3 prev_contact_pos_local = math::transform_point(prev_mesh_to_local,
                                                                  prev_contact_pos_mesh);

      /* Separating axis to move self out of penetration. */
      const float3 collision_axis = (pos_local - contact_pos_local) *
                                    (contact->is_inside ? -1.0f : 1.0f) *
                                    (collider.is_boundary ? -1.0f : 1.0f);
      const float3 valid_axis = math::normalize(math::is_zero(collision_axis, 1e-6f) ?
                                                    math::transpose(float3x3(local_to_mesh)) *
                                                        contact->face_nor *
                                                        (collider.is_boundary ? -1.0f : 1.0f) :
                                                    collision_axis);
      if (!(std::isfinite(valid_axis.x) && std::isfinite(valid_axis.y) &&
            std::isfinite(valid_axis.z)) ||
          math::is_zero(valid_axis))
      {
        /* A degenerate contact (e.g. a point exactly on the surface with an invalid normal) would
         * write NaN into the point positions and poison the whole simulation. */
        if (std::getenv("BLENDER_XPBD_DEBUG") != nullptr) {
          printf("XPBD_DEBUG degenerate contact normal (%f %f %f), is_inside=%d\n",
                 double(valid_axis.x),
                 double(valid_axis.y),
                 double(valid_axis.z),
                 int(contact->is_inside));
        }
        continue;
      }

      const float static_friction = this->compute_contact_friction(
          geo_data.static_frictions[point_i], collider.friction);
      const float dynamic_friction = this->compute_contact_friction(
          geo_data.dynamic_frictions[point_i], collider.friction);

      const int contact_i = r_face_contacts.points.append_and_get_index(point_i);
      r_face_contacts.point_radii.append(radius_local);
      r_face_contacts.positions_on_face.append(contact_pos_local);
      r_face_contacts.collider_motion.append(contact_pos_local - prev_contact_pos_local);
      r_face_contacts.face_normals.append(valid_axis);
      r_face_contacts.face_margins.append(margin_local);
      r_face_contacts.static_frictions.append(static_friction);
      r_face_contacts.dynamic_frictions.append(dynamic_friction);
      r_face_contacts.compliance_terms.append(
          std::max(0.0f, substep_compliance_factor_ * collider.compliance));
      r_face_contacts.error_scales.append(error_scale_from_threshold(collider.error_threshold));

      const MeshContactId contact_id{collider_usage.constraint_i, point_i};
      r_face_contacts.mesh_contact_indices.add(contact_id, contact_i);
      r_face_contacts.init_or_preserve_state(
          prev_face_contacts, prev_face_contacts.mesh_contact_indices.lookup_try(contact_id));
    }

    if (collider.use_edge_contacts) {
      auto handle_edge = [&](const int geo_contact_id, const int point0, const int point1) {
        if (geo_data.is_hard_pinned[point0] && geo_data.is_hard_pinned[point1]) {
          return;
        }
        const float3 &pos_local0 = positions[point0];
        const float3 &pos_local1 = positions[point1];
        const float3 pos_mesh0 = math::transform_point(local_to_mesh, pos_local0);
        const float3 pos_mesh1 = math::transform_point(local_to_mesh, pos_local1);
        const float radius_local0 = math::interpolate(
            geo_data.prev_radii[point0], geo_data.radii[point0], substep.interpolate_end);
        const float radius_local1 = math::interpolate(
            geo_data.prev_radii[point1], geo_data.radii[point1], substep.interpolate_end);
        /* Use max radius for collision detection. */
        const float max_radius_mesh = local_to_mesh_radius_factor *
                                      std::max(radius_local0, radius_local1);
        const std::optional<ClosestMeshEdgeContact> contact = this->get_closest_mesh_edge_contact(
            pos_mesh0,
            pos_mesh1,
            *edges_bvh,
            edge_verts,
            vert_positions,
            max_distance + max_radius_mesh);
        if (!contact) {
          return;
        }

        /* Add a contact for each face adjacent to the closest edge. */
        const int2 &edge = edge_verts[contact->edge_i];

        const float3 contact_pos_local = math::transform_point(mesh_to_local,
                                                               contact->nearest_pos);
        float3 prev_contact_pos_mesh;
        if (is_deforming) {
          prev_contact_pos_mesh = bke::attribute_math::mix2(
              contact->edge_factor, prev_vert_positions[edge[0]], prev_vert_positions[edge[1]]);
        }
        else {
          prev_contact_pos_mesh = contact->nearest_pos;
        }
        const float3 prev_contact_pos_local = math::transform_point(prev_mesh_to_local,
                                                                    prev_contact_pos_mesh);

        const float3 edge_direction_mesh = math::normalize(vert_positions[edge[1]] -
                                                           vert_positions[edge[0]]);
        const float3 edge_direction_local = math::normalize(
            math::transform_direction(mesh_to_local, edge_direction_mesh));
        /* Contact with the closest edge is active if the segment intersects with the
         * half-plane defined by the edge normal. A segment intersecting with any of the
         * adjacent faces would need to be moved along a tangent of the respective face to
         * resolve the collision, but since in combination with point-face collisions it is
         * sufficient to define a single edge normal, as long as the half-plane it defines is
         * inside both of the half-planes of the adjacent faces, i.e. between the two normal
         * directions. */
        const float3 averaged_vert_normal_mesh = math::midpoint(vert_normals[edge[0]],
                                                                vert_normals[edge[1]]);
        /* Transforming normal with transpose of inverse. */
        const float3 averaged_vert_normal_local = math::transform_direction(
            math::transpose(local_to_mesh), averaged_vert_normal_mesh);
        const float3 edge_normal_local = math::normalize(math::cross(
            math::cross(edge_direction_local, averaged_vert_normal_local), edge_direction_local));
        BLI_assert(math::is_unit(math::cross(edge_direction_local, edge_normal_local)));

        /* Use geometric average as edge friction coefficient. */
        const float static_friction = math::sqrt(
            math::square(this->compute_contact_friction(geo_data.static_frictions[point0],
                                                        collider.friction)) +
            math::square(this->compute_contact_friction(geo_data.static_frictions[point1],
                                                        collider.friction)));
        const float dynamic_friction = math::sqrt(
            math::square(this->compute_contact_friction(geo_data.dynamic_frictions[point0],
                                                        collider.friction)) +
            math::square(this->compute_contact_friction(geo_data.dynamic_frictions[point1],
                                                        collider.friction)));

        const int contact_i = r_edge_contacts.point_pairs.append_and_get_index({point0, point1});
        r_edge_contacts.point_radii.append({radius_local0, radius_local1});
        r_edge_contacts.positions_on_edge.append(contact_pos_local);
        r_edge_contacts.collider_motion.append(contact_pos_local - prev_contact_pos_local);
        r_edge_contacts.edge_directions.append(edge_direction_local);
        r_edge_contacts.edge_normals.append(edge_normal_local *
                                            (collider.is_boundary ? -1.0f : 1.0f));
        r_edge_contacts.edge_margins.append(margin_local);
        r_edge_contacts.static_frictions.append(static_friction);
        r_edge_contacts.dynamic_frictions.append(dynamic_friction);
        r_edge_contacts.compliance_terms.append(
            std::max(0.0f, substep_compliance_factor_ * collider.compliance));
        r_edge_contacts.error_scales.append(error_scale_from_threshold(collider.error_threshold));

        const MeshContactId contact_id{collider_usage.constraint_i, geo_contact_id};
        r_edge_contacts.mesh_contact_indices.add(contact_id, contact_i);
        r_edge_contacts.init_or_preserve_state(
            prev_edge_contacts, prev_edge_contacts.mesh_contact_indices.lookup_try(contact_id));
      };

      if (chunk.curves_range) {
        const OffsetIndices points_by_curve = geo_data.curves->points_by_curve();
        for (const int curve : *chunk.curves_range) {
          for (const int point0 : points_by_curve[curve].drop_back(1)) {
            const int geo_contact_id = point0;
            handle_edge(geo_contact_id, point0, point0 + 1);
          }
        }
      }
    }
  }

  void gather_from_world__collision_contacts()
  {
    const Span<std::string> paths = nested_bundle_paths_.lookup(CollisionContactsBundle::name);
    for (const StringRef path : paths) {
      const BundlePtr *bundle_ptr = world_.lookup_path_ptr<BundlePtr>(path);
      if (!bundle_ptr || !*bundle_ptr) {
        continue;
      }

      constraints_.collision_contacts.append({path});
    }
  }

  /**
   * Applies the uniform `acceleration` of every `Blender.DilapidoForce` bundle in the world to all
   * simulated geometries. This is the wind force for hair, cloth and soft bodies (the Ignis solver
   * reads the same bundle directly). The bundle has no filter, so it applies to every geometry in
   * the world; `Combine Bundle` can still be used to group it with colliders.
   */
  void gather_from_world__dilapido_forces()
  {
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      geo_data.external_acceleration.reinitialize(geo_data.size);
      geo_data.external_acceleration.fill(float3(0.0f));
    }

    const Span<std::string> paths = nested_bundle_paths_.lookup(
        std::string("Blender.DilapidoForce"));
    for (const StringRef path : paths) {
      const BundlePtr *bundle_ptr = world_.lookup_path_ptr<BundlePtr>(path);
      if (!bundle_ptr || !*bundle_ptr) {
        continue;
      }
      const Bundle &bundle = **bundle_ptr;
      const float3 acceleration = bundle.lookup<float3>(*BundleKey::from_str("acceleration"))
                                      .value_or(float3(0.0f));
      if (math::is_zero(acceleration)) {
        continue;
      }
      if (std::getenv("BLENDER_XPBD_DEBUG") != nullptr) {
        printf("XPBD_DILAPIDO_FORCE path '%s' acceleration (%.3f %.3f %.3f)\n",
               std::string(path).c_str(),
               double(acceleration.x),
               double(acceleration.y),
               double(acceleration.z));
      }
      for (const int data_key_i : geometries_.data_keys.index_range()) {
        GeometryData &geo_data = *geometries_.data[data_key_i];
        if (geo_data.external_acceleration.size() != geo_data.size) {
          continue;
        }
        threading::parallel_for(IndexRange(geo_data.size), 4096, [&](const IndexRange range) {
          for (const int i : range) {
            geo_data.external_acceleration[i] += acceleration;
          }
        });
      }
    }
  }

  struct ClosestMeshFaceContact {
    /** The nearest position exactly on the mesh surface. */
    float3 nearest_pos;
    float3 face_nor;
    float3 bary_coords;
    /** This does not take the radius into account. */
    bool is_inside;
    int tri_i;
  };

  struct ClosestMeshEdgeContact {
    /** The nearest position exactly on the mesh edge. */
    float3 nearest_pos;
    float edge_factor;
    int edge_i;
  };

  std::optional<ClosestMeshFaceContact> get_closest_mesh_face_contact(
      const float3 &sample_pos,
      const bke::BVHTreeFromMesh &corner_tris_bvh,
      const Span<int3> corner_tris,
      const Span<int> corner_verts,
      const Span<float3> vert_positions,
      const float max_distance) const
  {
    BVHTreeNearest nearest{};
    nearest.index = -1;
    nearest.dist_sq = pow2f(max_distance);
    BLI_bvhtree_find_nearest(corner_tris_bvh.tree,
                             sample_pos,
                             &nearest,
                             corner_tris_bvh.nearest_callback,
                             (void *)&corner_tris_bvh);
    if (nearest.index == -1) {
      return std::nullopt;
    }
    const int tri_i = nearest.index;
    const int3 &tri = corner_tris[tri_i];
    const float3 &contact_pos = float3(nearest.co);
    const float3 &contact_nor = float3(nearest.no);
    const float3 bary_coords = bke::mesh_surface_sample::compute_bary_coord_in_triangle(
        vert_positions, corner_verts, tri, contact_pos);
    const float3 direction_to_mesh = contact_pos - sample_pos;
    bool is_inside;
    if (this->is_bary_coord_close_to_edge(bary_coords)) {
      /* The nearest point is on an edge, so its normal is unreliable, use a more robust test. */
      is_inside = this->is_inside_ray_using_rays(sample_pos, corner_tris_bvh, direction_to_mesh);
    }
    else {
      is_inside = math::dot(direction_to_mesh, contact_nor) > 0.0f;
    }
    return ClosestMeshFaceContact{contact_pos, contact_nor, bary_coords, is_inside, tri_i};
  }

  std::optional<ClosestMeshEdgeContact> get_closest_mesh_edge_contact(
      const float3 &sample_pos0,
      const float3 &sample_pos1,
      const bke::BVHTreeFromMesh &edges_bvh,
      const Span<int2> edges,
      const Span<float3> vert_positions,
      const float max_distance) const
  {
    struct UserData {
      Span<int2> edges;
      Span<float3> vert_positions;
      float3 sample_pos0;
      float3 sample_pos1;
    };

    const auto closest_edge_cb =
        [](void *user_data, int index, const BVHTreeRay * /*ray*/, BVHTreeRayHit *hit) {
          const auto &data = *static_cast<const UserData *>(user_data);

          const int2 &edge = data.edges[index];
          const float3 &v0 = data.vert_positions[edge[0]];
          const float3 &v1 = data.vert_positions[edge[1]];

          float3 closest_on_ray, closest_on_edge;
          isect_seg_seg_v3(
              data.sample_pos0, data.sample_pos1, v0, v1, closest_on_ray, closest_on_edge);
          const float dist_squared = math::distance_squared(closest_on_ray, closest_on_edge);
          if (dist_squared <= math::square(hit->dist)) {
            hit->dist = math::sqrt(dist_squared);
            hit->index = index;
            copy_v3_v3(hit->co, closest_on_edge);
          }
        };

    float ray_len;
    const float3 ray_dir = math::normalize_and_get_length(sample_pos1 - sample_pos0, ray_len);

    UserData user_data = {edges, vert_positions, sample_pos0, sample_pos1};
    /* Needs to be initialized. */
    BVHTreeRayHit hit;
    hit.index = -1;
    hit.dist = ray_len;
    BLI_bvhtree_ray_cast_ex(
        edges_bvh.tree, sample_pos0, ray_dir, max_distance, &hit, closest_edge_cb, &user_data, 0);
    if (hit.index == -1) {
      return std::nullopt;
    }

    const int edge_i = hit.index;
    const int2 &edge = edges[edge_i];
    const float3 &contact_pos = float3(hit.co);
    const float3 v0 = vert_positions[edge[0]];
    const float3 v1 = vert_positions[edge[1]];
    const float edge_factor = math::sqrt(math::safe_divide(math::distance_squared(contact_pos, v0),
                                                           math::distance_squared(v1, v0)));
    return ClosestMeshEdgeContact{contact_pos, edge_factor, edge_i};
  }

  bool is_bary_coord_close_to_edge(const float3 &bary_coords) const
  {
    constexpr float epsilon = 1e-3f;
    return math::abs(bary_coords[0]) < epsilon || math::abs(bary_coords[1]) < epsilon ||
           math::abs(bary_coords[2]) < epsilon;
  }

  bool is_inside_ray_using_rays(const float3 &pos,
                                const bke::BVHTreeFromMesh &bvh,
                                const float3 &approx_ray_direction) const
  {
    /* Shoot rays in the approximate direction of where the nearest point is. This is a heuristic
     * for better performance to make the rays shorter. */
    const float3 approx_ray_direction_normalized = math::normalize(approx_ray_direction);
    const std::array<float3, 3> dirs = {
        {math::normalize(approx_ray_direction_normalized + float3{0.125f, -0.164f, 0.172f}),
         math::normalize(approx_ray_direction_normalized + float3{0.176f, 0.197f, -0.176f}),
         math::normalize(approx_ray_direction_normalized + float3{-0.140f, 0.151f, -0.126f})}};
    int inside_count = 0;
    for (const float3 &dir : dirs) {
      BVHTreeRayHit hit{};
      hit.index = -1;
      hit.dist = FLT_MAX;
      BLI_bvhtree_ray_cast(bvh.tree, pos, dir, 0.0f, &hit, bvh.raycast_callback, (void *)&bvh);
      if (hit.index != -1) {
        const float3 dir = float3(hit.co) - pos;
        const bool is_inside = math::dot(dir, float3(hit.no)) > 0.0f;
        inside_count += is_inside ? 1 : -1;
        if (std::abs(inside_count) >= 2) {
          break;
        }
      }
      else {
        return false;
      }
    }
    return inside_count > 0;
  }

  void gather_colliders_in_geometry(const StringRef path,
                                    const float4x4 &transform,
                                    const float4x4 &prev_transform,
                                    const GeometrySet &collider_geo,
                                    const GeometrySet *prev_collider_geo,
                                    const float margin,
                                    const float friction,
                                    const float compliance,
                                    const bool deforming,
                                    const bool use_edge_contacts,
                                    const bool is_boundary,
                                    const float error_threshold,
                                    const Span<int> affected_data,
                                    Vector<int> &instance_id_stack)
  {
    if (const Mesh *mesh = collider_geo.get_mesh()) {
      if (mesh->faces_num > 0) {
        MeshCollider mesh_collider;
        mesh_collider.path = path;
        mesh_collider.instance_ids = instance_id_stack;
        mesh_collider.margin = margin;
        mesh_collider.friction = friction;
        mesh_collider.compliance = compliance;
        mesh_collider.use_edge_contacts = use_edge_contacts;
        mesh_collider.is_boundary = is_boundary;
        mesh_collider.error_threshold = error_threshold;
        mesh_collider.begin_transform = prev_transform;
        mesh_collider.end_transform = transform;
        const Mesh *prev_mesh = prev_collider_geo ? prev_collider_geo->get_mesh() : nullptr;
        mesh_collider.mesh = this->make_collider_mesh_info(*mesh, prev_mesh, deforming);

        const int collider_i = constraints_.mesh_colliders.append_and_get_index(
            std::move(mesh_collider));
        for (const int data_key_i : affected_data) {
          geometries_.data[data_key_i]->mesh_colliders.append({collider_i});
        }
      }
    }
    if (const bke::Instances *instances = collider_geo.get_instances()) {
      struct PrevInstanceItem {
        const float4x4 *transform;
        const bke::InstanceReference *reference;
      };
      Map<int, PrevInstanceItem> prev_instance_by_id;
      if (prev_collider_geo) {
        const bke::Instances *prev_instances = prev_collider_geo->get_instances();
        if (prev_instances) {
          const Span<float4x4> prev_instance_transforms = prev_instances->transforms();
          const Span<int> prev_instance_ids = prev_instances->unique_ids();
          const Span<int> prev_handles = prev_instances->reference_handles();
          const Span<bke::InstanceReference> prev_references = prev_instances->references();
          for (const int i : prev_instance_transforms.index_range()) {
            const int prev_instance_id = prev_instance_ids[i];
            const float4x4 &prev_instance_transform = prev_instance_transforms[i];
            const bke::InstanceReference &prev_reference = prev_references[prev_handles[i]];
            prev_instance_by_id.add(prev_instance_id, {&prev_instance_transform, &prev_reference});
          }
        }
      }

      const Span<float4x4> instance_transforms = instances->transforms();
      const Span<bke::InstanceReference> references = instances->references();
      const Span<int> handles = instances->reference_handles();
      const Span<int> instance_ids = instances->unique_ids();
      for (const int instance_i : instance_transforms.index_range()) {
        const int handle = handles[instance_i];
        if (!references.index_range().contains(handle)) {
          continue;
        }
        const int instance_id = instance_ids[instance_i];
        const float4x4 instance_transform = instance_transforms[instance_i];
        const PrevInstanceItem *prev_instance_item = prev_instance_by_id.lookup_ptr(instance_id);
        const bke::InstanceReference &reference = references[handle];
        GeometrySet reference_geo;
        reference.to_geometry_set(reference_geo);
        GeometrySet prev_reference_geo;
        float4x4 prev_instance_transform = instance_transform;
        if (prev_instance_item) {
          prev_instance_item->reference->to_geometry_set(prev_reference_geo);
          prev_instance_transform = *prev_instance_item->transform;
        }

        instance_id_stack.append(instance_id);
        BLI_SCOPED_DEFER([&]() { instance_id_stack.pop_last(); });
        this->gather_colliders_in_geometry(path,
                                           transform * instance_transform,
                                           prev_transform * prev_instance_transform,
                                           reference_geo,
                                           prev_instance_item ? &prev_reference_geo : nullptr,
                                           margin,
                                           friction,
                                           compliance,
                                           deforming,
                                           use_edge_contacts,
                                           is_boundary,
                                           error_threshold,
                                           affected_data,
                                           instance_id_stack);
      }
    }
  }

  std::variant<StaticMeshInfo, DeformingMeshInfo> make_collider_mesh_info(const Mesh &mesh,
                                                                          const Mesh *prev_mesh,
                                                                          const bool deforming)
  {
    if (!deforming || !prev_mesh) {
      return StaticMeshInfo{&mesh, mesh.bvh_corner_tris(), mesh.bvh_edges()};
    }
    if (mesh.verts_num != prev_mesh->verts_num) {
      return StaticMeshInfo{&mesh, mesh.bvh_corner_tris(), mesh.bvh_edges()};
    }
    const int verts_num = mesh.verts_num;
    DeformingMeshInfo result;
    result.substep_meshes.resize(substeps_ + 1);
    result.substep_corner_tris_bvh_trees.resize(substeps_);
    result.substep_edges_bvh_trees.resize(substeps_);

    const Span<float3> begin_positions = prev_mesh->vert_positions();
    const Span<float3> end_positions = mesh.vert_positions();

    threading::parallel_for(
        IndexRange(substeps_ + 1), std::max(1024 / verts_num, 1), [&](const IndexRange range) {
          TLS &tls = tls_.local();
          for (const int mesh_i : range) {
            const float mix_factor = (mesh_i - 1) / float(substeps_) *
                                         (interpolation_end_ - interpolation_begin_) +
                                     interpolation_begin_;
            const Mesh *substep_mesh = nullptr;
            if (mix_factor == 0.0f && mesh_i == 0) {
              substep_mesh = prev_mesh;
            }
            else if (mix_factor == 1.0f) {
              substep_mesh = &mesh;
            }
            else {
              Mesh *interpolated_mesh = BKE_mesh_copy_for_eval(mesh);
              tls.scope.add_destruct_call(
                  [interpolated_mesh]() { BKE_id_free(nullptr, interpolated_mesh); });
              MutableSpan<float3> substep_positions =
                  interpolated_mesh->vert_positions_for_write();
              for (const int i : IndexRange(verts_num)) {
                substep_positions[i] = math::interpolate(
                    begin_positions[i], end_positions[i], mix_factor);
              }
              interpolated_mesh->tag_positions_changed();
              substep_mesh = interpolated_mesh;
            }
            result.substep_meshes[mesh_i] = substep_mesh;
            if (mesh_i > 0) {
              /* The bvh tree is not needed for the first substep. */
              result.substep_corner_tris_bvh_trees[mesh_i - 1] = substep_mesh->bvh_corner_tris();
              result.substep_edges_bvh_trees[mesh_i - 1] = substep_mesh->bvh_edges();
            }
          }
        });
    return result;
  }

  void prepare_geometry_chunks()
  {
    constexpr int approx_points_per_chunk = 256;
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      if (geo_data.size == 0) {
        continue;
      }
      const int old_chunks_num = geometries_.chunks.size();

      if (geo_data.fluid_constraint_i >= 0) {
        /* Neighbors of a fluid particle can be anywhere in the geometry, so a fluid is always a
         * single chunk. Splitting it would silently drop all neighbors across chunk boundaries. */
        geometries_.chunks.append({data_key_i, IndexRange(geo_data.size)});
      }
      else if (geo_data.curves) {
        /* For curve data, align the ranges with curve boundaries. */
        const bke::CurvesGeometry &curves = *geo_data.curves;
        const OffsetIndices<int> points_by_curve = curves.points_by_curve();
        Stack<IndexRange> curve_ranges_to_check;
        curve_ranges_to_check.push(curves.curves_range());
        while (!curve_ranges_to_check.is_empty()) {
          const IndexRange curves_range = curve_ranges_to_check.pop();
          const IndexRange points_range = points_by_curve[curves_range];
          if (points_range.is_empty()) {
            continue;
          }
          if (curves_range.size() == 1 || points_range.size() <= approx_points_per_chunk) {
            geometries_.chunks.append({data_key_i, points_range, curves_range});
            continue;
          }
          const int split_pos = curves_range.size() / 2;
          const IndexRange curves_range_left = curves_range.take_front(split_pos);
          const IndexRange curves_range_right = curves_range.drop_front(split_pos);
          /* Pushing in this order ensures that the chunks are sorted in the end. */
          curve_ranges_to_check.push(curves_range_right);
          curve_ranges_to_check.push(curves_range_left);
        }
      }
      else {
        const IndexRange points_range = IndexRange(geo_data.size);
        Stack<IndexRange> ranges_to_check;
        ranges_to_check.push(points_range);
        while (!ranges_to_check.is_empty()) {
          const IndexRange points_range = ranges_to_check.pop();
          if (points_range.is_empty()) {
            continue;
          }
          if (points_range.size() <= approx_points_per_chunk) {
            geometries_.chunks.append({data_key_i, points_range});
            continue;
          }
          const int split_pos = points_range.size() / 2;
          const IndexRange points_range_left = points_range.take_front(split_pos);
          const IndexRange points_range_right = points_range.drop_front(split_pos);
          /* Pushing in this order ensures that the chunks are sorted in the end. */
          ranges_to_check.push(points_range_right);
          ranges_to_check.push(points_range_left);
        }
      }
      geo_data.chunks = IndexRange::from_begin_end(old_chunks_num, geometries_.chunks.size());

      /* Create mapping from points to chunks. */
      geo_data.point_to_chunk.reinitialize(geo_data.size);
      for (const int chunk_i : geo_data.chunks) {
        const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
        geo_data.point_to_chunk.as_mutable_span().slice(chunk.points_range).fill(chunk_i);
      }
    }
    chunks_data_.reinitialize(geometries_.chunks.size());
    for (const int chunk_i : geometries_.chunks.index_range()) {
      const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
      geometries_.max_chunk_size = std::max<int>(geometries_.max_chunk_size,
                                                 chunk.points_range.size());
    }
  }

  void prepare_inverse_masses()
  {
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      geo_data.inv_masses.reinitialize(geo_data.size);
      MutableSpan<float> inv_masses = geo_data.inv_masses;

      if (const std::optional<float> mass = geo_data.masses.get_if_single()) {
        inv_masses.fill(math::safe_rcp(*mass));
      }
      else {
        const VArraySpan<float> masses = geo_data.masses;
        threading::parallel_for(IndexRange(geo_data.size), 2048, [&](const IndexRange range) {
          for (const int i : range) {
            const float mass = masses[i];
            inv_masses[i] = math::safe_rcp(mass);
          }
        });
      }
    }
  }

  void prepare_inverse_moments_of_inertia()
  {
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      if (!geo_data.uses_rotation) {
        continue;
      }
      geo_data.inv_moments_of_inertia.reinitialize(geo_data.size);
      MutableSpan<float3> inv_moments_of_inertia = geo_data.inv_moments_of_inertia;

      threading::parallel_for(IndexRange(geo_data.size), 2048, [&](const IndexRange range) {
        for (const int i : range) {
          const float3 &moment_of_inertia = geo_data.moments_of_inertia[i];
          if (math::is_zero(moment_of_inertia)) {
            inv_moments_of_inertia[i] = float3(0.0f);
          }
          else {
            inv_moments_of_inertia[i] = math::safe_rcp(moment_of_inertia);
          }
        }
      });
    }
  }

  void gather_from_world__stretch_shear_constraints()
  {
    TLS &tls = tls_.local();
    const Span<std::string> paths = nested_bundle_paths_.lookup(RodStretchShearBundle::name);
    for (const StringRef path : paths) {
      const Bundle &bundle = **world_.lookup_path_ptr<BundlePtr>(path);
      RodStretchShearConstraint constraint;
      constraint.path = path;
      constraint.error_threshold =
          bundle.lookup<float>(*BundleKey::from_str("error_threshold")).value_or(1e-3f);
      constraint.lambda_pos_attr = bundle
                                       .lookup<std::string>(
                                           *BundleKey::from_str("lambda_position_attribute"))
                                       .value_or("");
      constraint.lambda_rot_attr = bundle
                                       .lookup<std::string>(
                                           *BundleKey::from_str("lambda_rotation_attribute"))
                                       .value_or("");

      const int constraint_i = constraints_.rod_stretch_shear_constraints.append_and_get_index(
          std::move(constraint));

      for (const int data_key_i : geometries_.data_keys.index_range()) {
        GeometryData &geo_data = *geometries_.data[data_key_i];
        if (!geo_data.curves) {
          continue;
        }
        if (!geo_data.uses_rotation) {
          this->report(NodeWarningType::Error,
                       TIP_("Rod stretch/shear constraint requires \"rotation\" attribute"));
          continue;
        }
        if (!this->effector_applies_to_geometry(path, bundle, data_key_i)) {
          continue;
        }
        geo_data.rod_stretch_shear_constraints.append({constraint_i});
      }
    }
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (RodStretchShearConstraintUsage &constraint_usage :
           geo_data.rod_stretch_shear_constraints)
      {
        const RodStretchShearConstraint &constraint =
            constraints_.rod_stretch_shear_constraints[constraint_usage.constraint_i];
        VArray<float> rest_lengths = this->lookup_attribute_required<float>(
            data_key_i, this->prop_attr_name(constraint.path, "rest_length"), geo_data.domain);
        if (!rest_lengths) {
          continue;
        }

        constraint_usage.compliances = this->make_range_spans(
            tls,
            this->lookup_attribute_default<float>(
                data_key_i,
                this->prop_attr_name(constraint.path, "compliance"),
                geo_data.domain,
                0.0f));
        constraint_usage.rest_lengths = std::move(rest_lengths);
        constraint_usage.is_valid = true;

        constraint_usage.lambdas_pos = tls.allocator.allocate_array<float3>(geo_data.size);
        constraint_usage.lambdas_rot = tls.allocator.allocate_array<float3>(geo_data.size);
      }
    }
  }

  void create_chunk_constraints__rod_stretch_shear(TLS &tls, const int chunk_i)
  {
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    ChunkData &chunk_data = chunks_data_[chunk_i];
    const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    if (geo_data.rod_stretch_shear_constraints.is_empty()) {
      return;
    }
    const bke::CurvesGeometry &curves = *geo_data.curves;
    const OffsetIndices<int> points_by_curve = curves.points_by_curve();
    for (const RodStretchShearConstraintUsage &constraint_usage :
         geo_data.rod_stretch_shear_constraints)
    {
      if (!constraint_usage.is_valid) {
        continue;
      }
      const RodStretchShearConstraint &constraint =
          constraints_.rod_stretch_shear_constraints[constraint_usage.constraint_i];
      chunk_data.static_constraints.append(
          &tls.scope.construct<xpbd::RodStretchAndShearConstraintSet>(
              chunk.data_key_i,
              *chunk.curves_range,
              points_by_curve,
              constraint_usage.rest_lengths,
              constraint_usage.compliances.get_span_for_range(chunk.points_range),
              error_scale_from_threshold(constraint.error_threshold),
              constraint_usage.lambdas_pos,
              constraint_usage.lambdas_rot));
    }
  }

  void write_back__rod_stretch_shear()
  {
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (const RodStretchShearConstraintUsage &constraint_usage :
           geo_data.rod_stretch_shear_constraints)
      {
        if (!constraint_usage.is_valid) {
          continue;
        }
        const RodStretchShearConstraint &constraint =
            constraints_.rod_stretch_shear_constraints[constraint_usage.constraint_i];
        geo_data.attributes.remove(constraint.lambda_pos_attr);
        geo_data.attributes.remove(constraint.lambda_rot_attr);
        if (bke::SpanAttributeWriter<float3> lambda_pos_attr =
                this->get_output_attribute_writer<float3>(
                    data_key_i, constraint.lambda_pos_attr, geo_data.domain))
        {
          lambda_pos_attr.span.copy_from(constraint_usage.lambdas_pos);
          lambda_pos_attr.finish();
        }
        if (bke::SpanAttributeWriter<float3> lambda_rot_attr =
                this->get_output_attribute_writer<float3>(
                    data_key_i, constraint.lambda_rot_attr, geo_data.domain))
        {
          lambda_rot_attr.span.copy_from(constraint_usage.lambdas_rot);
          lambda_rot_attr.finish();
        }
      }
    }
  }

  void gather_from_world__bend_twist_constraints()
  {
    TLS &tls = tls_.local();
    const Span<std::string> paths = nested_bundle_paths_.lookup(RodBendTwistBundle::name);
    for (const StringRef path : paths) {
      const Bundle &bundle = **world_.lookup_path_ptr<BundlePtr>(path);

      RodBendTwistConstraint constraint;
      constraint.path = path;
      constraint.error_threshold =
          bundle.lookup<float>(*BundleKey::from_str("error_threshold")).value_or(1e-2f);

      const int constraint_i = constraints_.rod_bend_twist_constraints.append_and_get_index(
          std::move(constraint));

      for (const int data_key_i : geometries_.data.index_range()) {
        GeometryData &geo_data = *geometries_.data[data_key_i];
        if (!geo_data.curves) {
          continue;
        }
        if (!geo_data.uses_rotation) {
          this->report(NodeWarningType::Error,
                       TIP_("Rod stretch/shear constraint requires \"rotation\" attribute"));
          continue;
        }
        if (!this->effector_applies_to_geometry(path, bundle, data_key_i)) {
          continue;
        }
        geo_data.rod_bend_twist_constraints.append({constraint_i});
      }
    }
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (RodBendTwistConstraintUsage &constraint_usage : geo_data.rod_bend_twist_constraints) {
        const RodBendTwistConstraint &constraint =
            constraints_.rod_bend_twist_constraints[constraint_usage.constraint_i];
        constraint_usage.lambdas = tls.allocator.allocate_array<float4>(geo_data.size);
        constraint_usage.compliances = this->make_range_spans(
            tls,
            this->lookup_attribute_default<float>(
                data_key_i,
                this->prop_attr_name(constraint.path, "compliance"),
                geo_data.domain,
                0.0f));
        constraint_usage.rest_bend_rotations = this->lookup_attribute_default<math::Quaternion>(
            data_key_i,
            this->prop_attr_name(constraint.path, "rest_bend_rotation"),
            geo_data.domain,
            math::Quaternion::identity());
      }
    }
  }

  void create_chunk_constraints__rod_bend_twist(TLS &tls, const int chunk_i)
  {
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    ChunkData &chunk_data = chunks_data_[chunk_i];
    const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    if (geo_data.rod_bend_twist_constraints.is_empty()) {
      return;
    }
    const bke::CurvesGeometry &curves = *geo_data.curves;
    const OffsetIndices<int> points_by_curve = curves.points_by_curve();
    for (const RodBendTwistConstraintUsage &constraint_usage : geo_data.rod_bend_twist_constraints)
    {
      const RodBendTwistConstraint &constraint =
          constraints_.rod_bend_twist_constraints[constraint_usage.constraint_i];
      chunk_data.static_constraints.append(
          &tls.scope.construct<xpbd::RodBendAndTwistConstraintSet>(
              chunk.data_key_i,
              *chunk.curves_range,
              points_by_curve,
              constraint_usage.rest_bend_rotations,
              constraint_usage.compliances.get_span_for_range(chunk.points_range),
              error_scale_from_threshold(constraint.error_threshold),
              constraint_usage.lambdas));
    }
  }

  void gather_from_world__edge_length_constraints()
  {
    TLS &tls = tls_.local();
    const Span<std::string> paths = nested_bundle_paths_.lookup(EdgeLengthConstraintBundle::name);
    for (const StringRef path : paths) {
      const Bundle &bundle = **world_.lookup_path_ptr<BundlePtr>(path);

      const float error_threshold =
          bundle.lookup<float>(*BundleKey::from_str("error_threshold")).value_or(1e-3f);
      const int constraint_i = constraints_.edge_length_constraints.append_and_get_index(
          {path, error_threshold});

      for (const int data_key_i : geometries_.data.index_range()) {
        const DataKey &data_key = geometries_.data_keys[data_key_i];
        GeometryData &geo_data = *geometries_.data[data_key_i];
        if (data_key.type != GeometryComponent::Type::Mesh) {
          continue;
        }
        if (this->effector_applies_to_geometry(path, bundle, data_key_i)) {
          EdgeLengthConstraintUsage constraint_usage;
          constraint_usage.constraint_i = constraint_i;
          geo_data.edge_length_constraints.append(std::move(constraint_usage));
        }
      }
    }

    for (const int data_key_i : geometries_.data_keys.index_range()) {
      const DataKey &data_key = geometries_.data_keys[data_key_i];
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (EdgeLengthConstraintUsage &constraint_usage : geo_data.edge_length_constraints) {
        const EdgeLengthConstraint &constraint =
            constraints_.edge_length_constraints[constraint_usage.constraint_i];
        const Mesh &mesh = *geometries_.geometry_sets[data_key.geo_bundle_i].geometry.get_mesh();
        const Span<int2> edges = mesh.edges();
        const int edge_num = edges.size();

        VArray<float> rest_lengths = this->lookup_attribute_required<float>(
            data_key_i, this->prop_attr_name(constraint.path, "rest_length"), AttrDomain::Edge);
        if (!rest_lengths) {
          continue;
        }

        constraint_usage.lambdas = tls.allocator.allocate_array<float>(edge_num);
        constraint_usage.rest_lengths = rest_lengths;
        constraint_usage.compliances = this->lookup_attribute_default<float>(
            data_key_i,
            this->prop_attr_name(constraint.path, "compliance"),
            AttrDomain::Edge,
            0.0f);
        constraint_usage.is_valid = true;

        auto &constraint_set = tls.scope.construct<xpbd::DistanceConstraintSet>(
            data_key_i,
            edges,
            constraint_usage.rest_lengths,
            constraint_usage.compliances,
            error_scale_from_threshold(constraint.error_threshold),
            constraint_usage.lambdas);
        xpbd::ConstraintColoring coloring = constraint_set.color_constraints(tls.allocator);
        geo_data.static_constraints.append({&constraint_set, std::move(coloring)});
      }
    }
  }

  /**
   * Volume constraints of tetrahedral meshes: every face of the mesh is one element, its four
   * corners are the four points and the `rest_volume` face attribute is the rest volume (that is
   * what the `Tetrahedralize` node writes).
   */
  void gather_from_world__volume_constraints()
  {
    TLS &tls = tls_.local();
    const Span<std::string> paths = nested_bundle_paths_.lookup_as(VolumeConstraintBundle::name);
    for (const StringRef path : paths) {
      const Bundle &bundle = **world_.lookup_path_ptr<BundlePtr>(path);
      VolumeConstraint constraint;
      constraint.path = path;
      constraint.compliance = bundle.lookup<float>(*BundleKey::from_str("compliance"))
                                  .value_or(0.0f);
      constraint.edge_compliance = bundle.lookup<float>(*BundleKey::from_str("edge_compliance"))
                                       .value_or(0.0f);
      constraint.error_threshold = bundle.lookup<float>(*BundleKey::from_str("error_threshold"))
                                       .value_or(1e-3f);
      const int constraint_i = constraints_.volume_constraints.append_and_get_index(
          std::move(constraint));

      for (const int data_key_i : geometries_.data.index_range()) {
        const DataKey &data_key = geometries_.data_keys[data_key_i];
        GeometryData &geo_data = *geometries_.data[data_key_i];
        if (data_key.type != GeometryComponent::Type::Mesh) {
          continue;
        }
        if (this->effector_applies_to_geometry(path, bundle, data_key_i)) {
          VolumeConstraintUsage constraint_usage;
          constraint_usage.constraint_i = constraint_i;
          geo_data.volume_constraints.append(std::move(constraint_usage));
        }
      }
    }

    for (const int data_key_i : geometries_.data.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (VolumeConstraintUsage &constraint_usage : geo_data.volume_constraints) {
        const VolumeConstraint &constraint =
            constraints_.volume_constraints[constraint_usage.constraint_i];
        const GeometrySetData &geo_set_data =
            geometries_.geometry_sets[geometries_.data_keys[data_key_i].geo_bundle_i];
        const Mesh &mesh = *geo_set_data.geometry.get_mesh();

        const OffsetIndices<int> faces = mesh.faces();
        const VArray<float> rest_volumes = *geo_data.attributes.lookup<float>("rest_volume",
                                                                             AttrDomain::Face);
        /* Per point compliance written by `Init Puncliga Data > Soft Body`. The bundle values are
         * only the fallback for hand-built worlds. */
        const VArray<float> volume_compliances = this->lookup_attribute_default<float>(
            data_key_i,
            "puncliga_volume_compliance",
            AttrDomain::Point,
            constraint.compliance);
        const VArray<float> edge_compliances = this->lookup_attribute_default<float>(
            data_key_i,
            "puncliga_edge_compliance",
            AttrDomain::Point,
            constraint.edge_compliance);
        /* The elements are the four cornered faces. A tetrahedral mesh also carries its boundary as
         * three cornered faces (`Tetrahedralize` writes them so the same mesh is a closed surface),
         * and those are not elements: they have no rest volume and no gradient of their own. */
        int elements_num = 0;
        for (const int face_i : IndexRange(faces.size())) {
          elements_num += faces[face_i].size() == 4 ? 1 : 0;
        }
        if (elements_num == 0 || !rest_volumes) {
          this->report(NodeWarningType::Warning,
                       fmt::format("Volume constraint \"{}\" needs a tetrahedral mesh (four "
                                   "cornered faces with a `rest_volume` attribute, e.g. from the "
                                   "`Tetrahedralize` node), but \"{}\" is not one",
                                   constraint.path,
                                   geo_set_data.path));
          continue;
        }

        /* A surface snap can leave slivers behind (a curved surface forces the lattice nodes of an
         * element onto a plane). Their volume gradient is tiny, so their volume constraint is
         * ill-conditioned and destabilizes the whole mesh; they are left out, their edges still
         * keep them attached. */
        const VArray<float3> rest_positions = this->lookup_attribute_optional<float3>(
            data_key_i, "rest_position", AttrDomain::Point);
        const VArraySpan<float3> rest_positions_span = rest_positions;
        const Span<int> corner_verts = mesh.corner_verts();
        Vector<int> &element_points = constraint_usage.element_points;
        element_points.reserve(faces.size() * 4);
        for (const int face_i : IndexRange(faces.size())) {
          const IndexRange face = faces[face_i];
          if (face.size() != 4) {
            continue;
          }
          if (rest_positions && !this->is_valid_element(face, corner_verts, rest_positions_span,
                                                       rest_volumes[face_i]))
          {
            continue;
          }
          for (const int corner : face) {
            element_points.append(corner_verts[corner]);
          }
          constraint_usage.rest_volumes.append(rest_volumes[face_i]);
        }
        const int element_num = int(constraint_usage.rest_volumes.size());
        if (element_num == 0) {
          continue;
        }
        constraint_usage.element_offsets.reinitialize(element_num + 1);
        offset_indices::fill_constant_group_size(
            4, 0, constraint_usage.element_offsets.as_mutable_span());
        constraint_usage.elements = GroupedSpan<int>(
            OffsetIndices<int>(constraint_usage.element_offsets.as_span()),
            element_points.as_span());
        constraint_usage.compliances.reinitialize(element_num);
        for (const int element_i : IndexRange(element_num)) {
          constraint_usage.compliances[element_i] =
              volume_compliances[element_points[element_i * 4]];
        }
        constraint_usage.lambdas = tls.allocator.allocate_array<float>(element_num);
        constraint_usage.lambdas.fill(0.0f);

        auto &constraint_set = tls.scope.construct<xpbd::VolumeConstraintSet>(
            data_key_i,
            constraint_usage.elements,
            constraint_usage.rest_volumes.as_span(),
            constraint_usage.compliances.as_span(),
            error_scale_from_threshold(constraint.error_threshold),
            constraint_usage.lambdas);
        xpbd::ConstraintColoring coloring = constraint_set.color_constraints(tls.allocator);
        geo_data.static_constraints.append({&constraint_set, std::move(coloring)});

        /* A volume constraint alone does not resist shear: a tetrahedron keeps its volume while it
         * flattens into a pancake (a cube of jelly would collapse into a sheet on the floor). The
         * six edges of every element are what makes the material hold its shape, which is the pair
         * of constraints that a position based soft body uses (Macklin et al. 2016). They need the
         * rest pose, which `Init Puncliga Data` stores for exactly this reason. */
        if (!rest_positions) {
          this->report(
              NodeWarningType::Warning,
              fmt::format("The soft body \"{}\" has no `rest_position` attribute, so its elements "
                          "only get their volume constraint (a `Init Puncliga Data` node with the "
                          "type set to Soft Body writes it)",
                          geo_set_data.path));
          continue;
        }

        /* The edges come from *all* elements, including the slivers whose volume constraint was
         * left out: a vertex that only belongs to slivers would otherwise have no constraint at all
         * and would stay on the floor while the rest of the body moves away (which is exactly what
         * a "vertex glued to the ground" looks like). A sliver's edges have a normal length, so
         * they are well conditioned even though its volume is not. */
        Vector<int2> &edges = constraint_usage.edge_points;
        Vector<float> &edge_rest_lengths = constraint_usage.edge_rest_lengths;
        Set<int64_t> edge_set;
        edges.reserve(faces.size() * 3);
        edge_rest_lengths.reserve(faces.size() * 3);
        for (const int face_i : IndexRange(faces.size())) {
          const IndexRange face = faces[face_i];
          if (face.size() != 4) {
            continue;
          }
          for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
              const int v0 = corner_verts[face[i]];
              const int v1 = corner_verts[face[j]];
              const int64_t key = (int64_t(std::min(v0, v1)) << 32) | int64_t(std::max(v0, v1));
              if (!edge_set.add(key)) {
                continue;
              }
              edges.append({v0, v1});
              edge_rest_lengths.append(math::distance(rest_positions[v0], rest_positions[v1]));
            }
          }
        }
        constraint_usage.edge_compliances.reinitialize(edges.size());
        for (const int edge_i : IndexRange(edges.size())) {
          constraint_usage.edge_compliances[edge_i] = edge_compliances[edges[edge_i].x];
        }
        constraint_usage.edge_lambdas = tls.allocator.allocate_array<float>(edges.size());
        constraint_usage.edge_lambdas.fill(0.0f);

        auto &edge_set_object = tls.scope.construct<xpbd::DistanceConstraintSet>(
            data_key_i,
            edges.as_span(),
            edge_rest_lengths.as_span(),
            constraint_usage.edge_compliances.as_span(),
            error_scale_from_threshold(constraint.error_threshold),
            constraint_usage.edge_lambdas);
        xpbd::ConstraintColoring edge_coloring = edge_set_object.color_constraints(tls.allocator);
        geo_data.static_constraints.append({&edge_set_object, std::move(edge_coloring)});
      }
    }
  }

  /** Whether a tetrahedron is not too flat to carry a volume constraint. */
  static bool is_valid_element(const IndexRange face,
                               const Span<int> corner_verts,
                               const Span<float3> rest_positions,
                               const float rest_volume) noexcept
  {
    if (face.size() != 4 || rest_volume <= 0.0f) {
      return false;
    }
    const float3 &p0 = rest_positions[corner_verts[face[0]]];
    const float3 &p1 = rest_positions[corner_verts[face[1]]];
    const float3 &p2 = rest_positions[corner_verts[face[2]]];
    const float3 &p3 = rest_positions[corner_verts[face[3]]];
    const float longest = std::max(
        std::max(std::max(math::distance(p0, p1), math::distance(p0, p2)),
                 std::max(math::distance(p0, p3), math::distance(p1, p2))),
        std::max(math::distance(p1, p3), math::distance(p2, p3)));
    /* A well shaped element of this lattice decomposition has V = L^3 / 31; anything below a
     * thousandth of L^3 is a sliver. */
    return rest_volume > 1e-3f * longest * longest * longest;
  }

  void gather_from_world__cross_edge_length_constraints()
  {
    TLS &tls = tls_.local();
    const Span<std::string> paths = nested_bundle_paths_.lookup_as(
        CrossEdgeLengthConstraintBundle::name);
    for (const StringRef path : paths) {
      const Bundle &bundle = **world_.lookup_path_ptr<BundlePtr>(path);
      const float error_threshold =
          bundle.lookup<float>(*BundleKey::from_str("error_threshold")).value_or(1e-3f);
      const int constraint_i = constraints_.cross_edge_length_constraints.append_and_get_index(
          {path, error_threshold});
      for (const int data_key_i : geometries_.data.index_range()) {
        const DataKey &data_key = geometries_.data_keys[data_key_i];
        GeometryData &geo_data = *geometries_.data[data_key_i];
        if (data_key.type != GeometryComponent::Type::Mesh) {
          continue;
        }
        if (this->effector_applies_to_geometry(path, bundle, data_key_i)) {
          CrossEdgeLengthConstraintUsage constraint_usage;
          constraint_usage.constraint_i = constraint_i;
          geo_data.cross_edge_length_constraints.append(std::move(constraint_usage));
        }
      }
    }

    for (const int data_key_i : geometries_.data_keys.index_range()) {
      const DataKey &data_key = geometries_.data_keys[data_key_i];
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (CrossEdgeLengthConstraintUsage &constraint_usage :
           geo_data.cross_edge_length_constraints)
      {
        const CrossEdgeLengthConstraint &constraint =
            constraints_.cross_edge_length_constraints[constraint_usage.constraint_i];
        const Mesh &mesh = *geometries_.geometry_sets[data_key.geo_bundle_i].geometry.get_mesh();
        const VArraySpan<float3> rest_positions = this->lookup_attribute_required<float3>(
            data_key_i, this->prop_attr_name(constraint.path, "rest_position"), AttrDomain::Point);
        if (rest_positions.is_empty()) {
          continue;
        }
        const VArray<float> edge_compliances = this->lookup_attribute_default<float>(
            data_key_i,
            this->prop_attr_name(constraint.path, "compliance"),
            AttrDomain::Edge,
            0.0f);

        Vector<int2> &cross_edges = constraint_usage.cross_edge_points;
        Vector<float> &cross_edge_rest_lengths = constraint_usage.distances;
        Vector<float> &cross_edge_compliances = constraint_usage.compliances;

        this->foreach_cross_edge(mesh, [&](const int edge_i, const int v0, const int v1) {
          const float3 &rest_position0 = rest_positions[v0];
          const float3 &rest_position1 = rest_positions[v1];
          const float rest_length = math::distance(rest_position0, rest_position1);
          const float compliance = edge_compliances[edge_i];
          cross_edges.append({v0, v1});
          cross_edge_rest_lengths.append(rest_length);
          cross_edge_compliances.append(compliance);
        });

        constraint_usage.lambdas = tls.allocator.allocate_array<float>(cross_edges.size());

        auto &constraint_set = tls.scope.construct<xpbd::DistanceConstraintSet>(
            data_key_i,
            cross_edges,
            cross_edge_rest_lengths,
            cross_edge_compliances,
            error_scale_from_threshold(constraint.error_threshold),
            constraint_usage.lambdas);
        xpbd::ConstraintColoring coloring = constraint_set.color_constraints(tls.allocator);
        geo_data.static_constraints.append({&constraint_set, std::move(coloring)});
      }
    }
  }

  void foreach_cross_edge(const Mesh &mesh,
                          std::invocable<int /*edge_i*/, int /*v0*/, int /*v1*/> auto &&fn)
  {
    /* Don't use triangulated mesh because the triangulation depends on vertex positions which
     * change during the simulation which introduces instability. */
    const Span<int2> edges = mesh.edges();
    const Span<int> corner_verts = mesh.corner_verts();
    const OffsetIndices<int> faces = mesh.faces();

    Array<int> offsets;
    Array<int> indices;
    const GroupedSpan<int> edge_to_faces = bke::mesh::build_edge_to_face_map(
        faces, mesh.corner_edges(), mesh.edges_num, offsets, indices);

    for (const int edge_i : edges.index_range()) {
      const int2 &edge = edges[edge_i];
      const int edge_v0 = edge[0];
      const int edge_v1 = edge[1];
      const Span<int> incident_faces = edge_to_faces[edge_i];
      if (incident_faces.size() != 2) {
        continue;
      }
      int face0 = incident_faces[0];
      int face1 = incident_faces[1];
      if (faces[face0].size() > faces[face1].size()) {
        std::swap(face0, face1);
      }
      const IndexRange face0_corners = faces[face0];
      const IndexRange face1_corners = faces[face1];
      const int face0_size = face0_corners.size();
      const int face1_size = face1_corners.size();
      if (face0_size == 3 && face1_size == 3) {
        const int face0_v0 = corner_verts[face0_corners[0]];
        const int face0_v1 = corner_verts[face0_corners[1]];
        const int face0_v2 = corner_verts[face0_corners[2]];

        const int face1_v0 = corner_verts[face1_corners[0]];
        const int face1_v1 = corner_verts[face1_corners[1]];
        const int face1_v2 = corner_verts[face1_corners[2]];

        /* Create a cross edge between opposite corners of the two faces. */
        const int cross_edge_v0 = edge_v0 ^ edge_v1 ^ face0_v0 ^ face0_v1 ^ face0_v2;
        const int cross_edge_v1 = edge_v0 ^ edge_v1 ^ face1_v0 ^ face1_v1 ^ face1_v2;
        fn(edge_i, cross_edge_v0, cross_edge_v1);
      }
      else if (face0_size == 3 && face1_size == 4) {
        const int face0_v0 = corner_verts[face0_corners[0]];
        const int face0_v1 = corner_verts[face0_corners[1]];
        const int face0_v2 = corner_verts[face0_corners[2]];

        /* Create two cross edges between the triangle and quad. */
        const int cross_edge_v0 = edge_v0 ^ edge_v1 ^ face0_v0 ^ face0_v1 ^ face0_v2;
        for (const int face1_corner : face1_corners) {
          const int face1_v = corner_verts[face1_corner];
          if (!ELEM(face1_v, edge_v0, edge_v1)) {
            fn(edge_i, cross_edge_v0, face1_v);
          }
        }
      }
      else {
        /* Create cross edges between "diagonal" corners. */
        const Span<int> face0_verts = corner_verts.slice(face0_corners);
        const Span<int> face1_verts = corner_verts.slice(face1_corners);
        const int face0_i0 = face0_verts.first_index(edge_v0);
        const int face0_dir = face0_verts[(face0_i0 + 1) % face0_size] == edge_v1 ? 1 : -1;
        const int face1_i1 = face1_verts.first_index(edge_v1);
        const int face1_dir = face1_verts[(face1_i1 + 1) % face1_size] == edge_v0 ? 1 : -1;
        /* Some corners may be ignored if they if both faces have a different number of
         * corners. */
        const int cross_edge_num = std::max(face0_size, face1_size) - 2;
        for (const int i : IndexRange(cross_edge_num)) {
          const int cross_edge_v0 = face0_verts[mod_i(face0_i0 - face0_dir * (i + 1), face0_size)];
          const int cross_edge_v1 = face1_verts[mod_i(face1_i1 - face1_dir * (i + 1), face1_size)];
          fn(edge_i, cross_edge_v0, cross_edge_v1);
        }
      }
    }
  }

  void gather_from_world__damping()
  {
    TLS &tls = tls_.local();
    const Span<std::string> paths = nested_bundle_paths_.lookup(DampingBundle::name);
    for (const StringRef path : paths) {
      const BundlePtr *bundle_ptr = world_.lookup_path_ptr<BundlePtr>(path);
      if (!bundle_ptr || !*bundle_ptr) {
        continue;
      }
      const Bundle &bundle = **bundle_ptr;

      const int constraint_i = constraints_.damping_constraints.append_and_get_index({path});

      for (const int data_key_i : geometries_.data_keys.index_range()) {
        GeometryData &geo_data = *geometries_.data[data_key_i];
        if (this->effector_applies_to_geometry(path, bundle, data_key_i)) {
          geo_data.damping_constraints.append({constraint_i});
        }
      }
    }

    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (DampingConstraintUsage &constraint_usage : geo_data.damping_constraints) {
        const DampingConstraint &constraint =
            constraints_.damping_constraints[constraint_usage.constraint_i];
        constraint_usage.linear_dampings = this->make_range_spans(
            tls,
            this->lookup_attribute_default<float>(data_key_i,
                                                  this->prop_attr_name(constraint.path, "linear"),
                                                  geo_data.domain,
                                                  0.0f));
        constraint_usage.angular_dampings = this->make_range_spans(
            tls,
            this->lookup_attribute_default<float>(data_key_i,
                                                  this->prop_attr_name(constraint.path, "angular"),
                                                  geo_data.domain,
                                                  0.0f));

        constraint_usage.linear_damping_lambdas = tls.allocator.allocate_array<float>(
            geo_data.size);
        constraint_usage.angular_damping_lambdas = tls.allocator.allocate_array<float>(
            geo_data.size);
      }
    }
  }

  void create_chunk_constraints__damping(TLS &tls, const int chunk_i)
  {
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    ChunkData &chunk_data = chunks_data_[chunk_i];
    const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];

    for (const DampingConstraintUsage &constraint_usage : geo_data.damping_constraints) {
      chunk_data.static_velocity_constraints.append(
          &tls.scope.construct<xpbd::LinearDampingConstraintSet>(
              chunk.data_key_i,
              chunk.points_range,
              constraint_usage.linear_dampings.get_span_for_range(chunk.points_range),
              constraint_usage.linear_damping_lambdas.slice(chunk.points_range)));
      if (geo_data.uses_rotation) {
        chunk_data.static_velocity_constraints.append(
            &tls.scope.construct<xpbd::AngularDampingConstraintSet>(
                chunk.data_key_i,
                chunk.points_range,
                constraint_usage.angular_dampings.get_span_for_range(chunk.points_range),
                constraint_usage.angular_damping_lambdas.slice(chunk.points_range)));
      }
    }
  }

  void gather_from_world__pin_positions()
  {
    TLS &tls = tls_.local();
    const Span<std::string> paths = nested_bundle_paths_.lookup(PinPositionBundle::name);
    for (const StringRef path : paths) {
      const Bundle &bundle = **world_.lookup_path_ptr<BundlePtr>(path);
      PinPositionConstraint constraint;
      constraint.path = path;
      constraint.error_threshold =
          bundle.lookup<float>(*BundleKey::from_str("error_threshold")).value_or(1e-3f);
      constraint.lambda_attr =
          bundle.lookup<std::string>(*BundleKey::from_str("lambda_attribute")).value_or("");
      const int constraint_i = constraints_.pin_position_constraints.append_and_get_index(
          std::move(constraint));

      for (const int data_key_i : geometries_.data_keys.index_range()) {
        GeometryData &geo_data = *geometries_.data[data_key_i];
        if (this->effector_applies_to_geometry(path, bundle, data_key_i)) {
          geo_data.pin_position_constraints.append({constraint_i});
        }
      }
    }
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (PinPositionConstraintUsage &constraint_usage : geo_data.pin_position_constraints) {
        const PinPositionConstraint &constraint =
            constraints_.pin_position_constraints[constraint_usage.constraint_i];

        const VArray<bool> selection_attr = this->lookup_attribute_required<bool>(
            data_key_i, this->prop_attr_name(constraint.path, "selection"), geo_data.domain);
        const VArray<float3> positions_attr = this->lookup_attribute_required<float3>(
            data_key_i, this->prop_attr_name(constraint.path, "position"), geo_data.domain);
        if (!positions_attr || !selection_attr) {
          continue;
        }
        const IndexMask pin_selection = IndexMask::from_bools(selection_attr, tls.allocator);
        if (pin_selection.is_empty()) {
          continue;
        }
        const VArray<float> compliances_attr = this->lookup_attribute_default<float>(
            data_key_i,
            this->prop_attr_name(constraint.path, "compliance"),
            geo_data.domain,
            0.0f);
        VArray<bool> prev_selection_attr;
        VArray<float3> prev_positions_attr;
        if (sub_delta_time_ > 0.0f) {
          prev_selection_attr = this->lookup_attribute_optional<bool>(
              data_key_i,
              this->prev_prop_attr_name(constraint.path, "selection"),
              geo_data.domain);
          prev_positions_attr = this->lookup_attribute_optional<float3>(
              data_key_i, this->prev_prop_attr_name(constraint.path, "position"), geo_data.domain);
        }

        const int pin_num = pin_selection.size();
        MutableSpan<int> points = tls.allocator.allocate_array<int>(pin_num);
        pin_selection.to_indices(points);

        MutableSpan<float3> end_positions = tls.allocator.allocate_array<float3>(pin_num);
        MutableSpan<float> compliances = tls.allocator.allocate_array<float>(pin_num);

        positions_attr.materialize_compressed_to_uninitialized(pin_selection, end_positions);
        compliances_attr.materialize_compressed_to_uninitialized(pin_selection, compliances);

        constraint_usage.points = points;
        constraint_usage.end_positions = end_positions;
        constraint_usage.compliances = compliances;
        /* These will be initialized later. */
        constraint_usage.lambdas = tls.allocator.allocate_array<float>(pin_num);
        constraint_usage.current_positions = tls.allocator.allocate_array<float3>(pin_num);

        /* Initialize the previous pin position. */
        if (prev_selection_attr && prev_positions_attr) {
          MutableSpan<float3> begin_positions = tls.allocator.allocate_array<float3>(pin_num);
          threading::parallel_for(pin_selection.index_range(), 1024, [&](const IndexRange range) {
            for (const int pin_i : range) {
              const int point_i = points[pin_i];
              const bool was_pinned = prev_selection_attr[point_i];
              if (was_pinned) {
                begin_positions[pin_i] = prev_positions_attr[point_i];
              }
              else {
                begin_positions[pin_i] = geo_data.position_attr.span[point_i];
              }
            }
          });
          constraint_usage.begin_positions = begin_positions;
        }
        else {
          constraint_usage.begin_positions = end_positions;
        }
      }
    }
  }

  void create_chunk_constraints__pin_positions(TLS &tls, const int chunk_i)
  {
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    ChunkData &chunk_data = chunks_data_[chunk_i];
    GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    for (const int constraint_usage_i : geo_data.pin_position_constraints.index_range()) {
      const PinPositionConstraintUsage &constraint_usage =
          geo_data.pin_position_constraints[constraint_usage_i];
      const PinPositionConstraint &constraint =
          constraints_.pin_position_constraints[constraint_usage.constraint_i];

      /* Detect hard pinned points. */
      for (const int pin_i : constraint_usage.points.index_range()) {
        const int point_i = constraint_usage.points[pin_i];
        const float compliance = constraint_usage.compliances[pin_i];
        if (compliance <= 0.0f) {
          geo_data.is_hard_pinned[point_i] = true;
        }
      }

      const IndexRange pin_range = unique_sorted_indices::find_content_range<int>(
          constraint_usage.points, chunk.points_range);
      if (pin_range.is_empty()) {
        continue;
      }
      chunk_data.pin_position_constraints.append({constraint_usage_i, pin_range});
      chunk_data.static_constraints.append(&tls.scope.construct<xpbd::PinPositionConstraintSet>(
          chunk.data_key_i,
          constraint_usage.points.slice(pin_range),
          constraint_usage.current_positions.slice(pin_range),
          constraint_usage.compliances.slice(pin_range),
          error_scale_from_threshold(constraint.error_threshold),
          constraint_usage.lambdas.slice(pin_range)));
    }
  }

  void write_back__pin_positions()
  {
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (const PinPositionConstraintUsage &constraint_usage : geo_data.pin_position_constraints)
      {
        const PinPositionConstraint &constraint =
            constraints_.pin_position_constraints[constraint_usage.constraint_i];
        geo_data.attributes.remove(constraint.lambda_attr);
      }
      for (const PinPositionConstraintUsage &constraint_usage : geo_data.pin_position_constraints)
      {
        if (constraint_usage.points.is_empty()) {
          /* If there is nothing pinned, these attributes don't need to exist. */
          continue;
        }
        const PinPositionConstraint &constraint =
            constraints_.pin_position_constraints[constraint_usage.constraint_i];
        if (bke::SpanAttributeWriter<float> lambda_attr = this->get_output_attribute_writer<float>(
                data_key_i, constraint.lambda_attr, geo_data.domain))
        {
          for (const int pin_i : constraint_usage.points.index_range()) {
            const int point_i = constraint_usage.points[pin_i];
            lambda_attr.span[point_i] = constraint_usage.lambdas[pin_i];
          }
          lambda_attr.finish();
        }
      }
    }
  }

  void gather_from_world__pin_rotations()
  {
    TLS &tls = tls_.local();
    const Span<std::string> paths = nested_bundle_paths_.lookup(PinRotationBundle::name);
    for (const StringRef path : paths) {
      const Bundle &bundle = **world_.lookup_path_ptr<BundlePtr>(path);

      PinRotationConstraint constraint;
      constraint.path = path;
      constraint.error_threshold =
          bundle.lookup<float>(*BundleKey::from_str("error_threshold")).value_or(1e-2f);
      const int constraint_i = constraints_.pin_rotation_constraints.append_and_get_index(
          std::move(constraint));

      for (const int data_key_i : geometries_.data_keys.index_range()) {
        GeometryData &geo_data = *geometries_.data[data_key_i];
        if (!geo_data.uses_rotation) {
          continue;
        }
        if (this->effector_applies_to_geometry(path, bundle, data_key_i)) {
          geo_data.pin_rotation_constraints.append({constraint_i});
        }
      }
    }
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (PinRotationConstraintUsage &constraint_usage : geo_data.pin_rotation_constraints) {
        const PinRotationConstraint &constraint =
            constraints_.pin_rotation_constraints[constraint_usage.constraint_i];

        const VArray<bool> selection_attr = this->lookup_attribute_required<bool>(
            data_key_i, this->prop_attr_name(constraint.path, "selection"), geo_data.domain);
        const VArray<math::Quaternion> rotation_attr =
            this->lookup_attribute_required<math::Quaternion>(
                data_key_i, this->prop_attr_name(constraint.path, "rotation"), geo_data.domain);
        if (!selection_attr || !rotation_attr) {
          continue;
        }
        const IndexMask pin_selection = IndexMask::from_bools(selection_attr, tls.allocator);
        if (pin_selection.is_empty()) {
          continue;
        }
        const VArray<float> compliances_attr = this->lookup_attribute_default<float>(
            data_key_i,
            this->prop_attr_name(constraint.path, "compliance"),
            geo_data.domain,
            0.0f);
        VArray<bool> prev_selection_attr;
        VArray<math::Quaternion> prev_rotations_attr;
        if (sub_delta_time_ > 0.0f) {
          prev_selection_attr = this->lookup_attribute_optional<bool>(
              data_key_i,
              this->prev_prop_attr_name(constraint.path, "selection"),
              geo_data.domain);
          prev_rotations_attr = this->lookup_attribute_optional<math::Quaternion>(
              data_key_i, this->prev_prop_attr_name(constraint.path, "rotation"), geo_data.domain);
        }

        const int pin_num = pin_selection.size();
        MutableSpan<int> points = tls.allocator.allocate_array<int>(pin_num);
        pin_selection.to_indices(points);

        MutableSpan<math::Quaternion> end_rotations =
            tls.allocator.allocate_array<math::Quaternion>(pin_num);
        MutableSpan<float> compliances = tls.allocator.allocate_array<float>(pin_num);

        rotation_attr.materialize_compressed_to_uninitialized(pin_selection, end_rotations);
        compliances_attr.materialize_compressed_to_uninitialized(pin_selection, compliances);

        constraint_usage.points = points;
        constraint_usage.end_rotations = end_rotations;
        constraint_usage.compliances = compliances;
        /* These will be initialized later. */
        constraint_usage.lambdas = tls.allocator.allocate_array<float4>(pin_num);
        constraint_usage.current_rotations = tls.allocator.allocate_array<math::Quaternion>(
            pin_num);

        /* Initialize the previous pin rotation. */
        if (prev_selection_attr && prev_rotations_attr) {
          MutableSpan<math::Quaternion> begin_rotations =
              tls.allocator.allocate_array<math::Quaternion>(pin_num);
          threading::parallel_for(pin_selection.index_range(), 1024, [&](const IndexRange range) {
            for (const int pin_i : range) {
              const int point_i = points[pin_i];
              const bool was_pinned = prev_selection_attr[point_i];
              if (was_pinned) {
                begin_rotations[pin_i] = prev_rotations_attr[point_i];
                continue;
              }
              begin_rotations[pin_i] = geo_data.rotation_attr.span[point_i];
            }
          });
          constraint_usage.begin_rotations = begin_rotations;
        }
        else {
          constraint_usage.begin_rotations = end_rotations;
        }
      }
    }
  }

  void create_chunk_constraints__pin_rotations(TLS &tls, const int chunk_i)
  {
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    ChunkData &chunk_data = chunks_data_[chunk_i];
    const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    for (const int constraint_usage_i : geo_data.pin_rotation_constraints.index_range()) {
      const PinRotationConstraintUsage &constraint_usage =
          geo_data.pin_rotation_constraints[constraint_usage_i];
      const PinRotationConstraint &constraint =
          constraints_.pin_rotation_constraints[constraint_usage.constraint_i];

      const IndexRange pin_range = unique_sorted_indices::find_content_range<int>(
          constraint_usage.points, chunk.points_range);
      if (pin_range.is_empty()) {
        continue;
      }
      chunk_data.pin_rotation_constraints.append({constraint_usage_i, pin_range});
      chunk_data.static_constraints.append(&tls.scope.construct<xpbd::PinRotationConstraintSet>(
          chunk.data_key_i,
          constraint_usage.points.slice(pin_range),
          constraint_usage.current_rotations.slice(pin_range),
          constraint_usage.compliances.slice(pin_range),
          error_scale_from_threshold(constraint.error_threshold),
          constraint_usage.lambdas.slice(pin_range)));
    }
  }

  struct ResidualErrorTLS {
    float total_error_squared = 0.0f;
    int total_error_count = 0;
  };

  /** Debug helper: reports the first non finite position of any geometry (XPBD_DEBUG_NAN). */
  static void debug_check_finite(const char *label, const Span<xpbd::GeometryRef> refs)
  {
    if (std::getenv("XPBD_DEBUG_NAN") == nullptr) {
      return;
    }
    for (const int geo_i : refs.index_range()) {
      const Span<float3> positions = refs[geo_i].positions;
      for (const int point_i : positions.index_range()) {
        const float3 &p = positions[point_i];
        if (!(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))) {
          printf("XPBD_DEBUG_NAN %s: geo %d point %d = (%g %g %g)\n",
                 label,
                 geo_i,
                 point_i,
                 double(p.x),
                 double(p.y),
                 double(p.z));
          return;
        }
      }
    }
  }

  /**
   * Publishes what the last step did on the compute backend, so that the node can show it (header
   * badge and the info block). Called only when the solver runs inside an `HW Accelerate` zone.
   */
 public:
  void publish_ha_status(const bNodeTree &tree, const bNode &node) const
  {
    geometry::ha::HaNodeStatus status;
    status.backend = geometry::ha::ha_gpu_backend_name();
    bool has_fluid = false;
    for (const ChunkData &chunk_data : chunks_data_) {
      const FluidChunkData &fluid = chunk_data.fluid;
      if (!fluid.active) {
        continue;
      }
      has_fluid = true;
      if (fluid.gpu_active) {
        status.state = geometry::ha::HaNodeState::Device;
        status.elements = fluid.gpu_fluid.particle_count();
        break;
      }
    }
    if (status.state != geometry::ha::HaNodeState::Device) {
      if (!has_fluid) {
        /* Nothing to accelerate: no badge at all. */
        geometry::ha::ha_status_clear(tree, node);
        return;
      }
      status.state = geometry::ha::HaNodeState::Fallback;
      status.reason = gpu_rejection_reason_;
    }
    geometry::ha::ha_status_set(tree, node, status);
  }

 private:
  Result do_simulation()
  {
    this->prepare_solver_geometry_refs();
    float average_error_squared = 0.0f;

    /* One device context for the whole step when any fluid runs on the device: the fluid work is
     * split over many small scopes (bind, every substep phase, every readback), and each of them
     * would activate the context again otherwise. Nested guards are free. */
    std::optional<geometry::ha::HaContextGuard> device_guard;
    for (const ChunkData &chunk_data : chunks_data_) {
      if (chunk_data.fluid.gpu_active) {
        device_guard.emplace();
        break;
      }
    }

    if (this->support_chunk_simulation()) {
      threading::EnumerableThreadSpecific<ResidualErrorTLS> error_tls;

      this->parallel_for_each_chunk(1, [&](const int chunk_i) {
        float chunk_error_squared;
        int chunk_error_count;
        int solver_refs_i = 0;
        const bool prof = std::getenv("XPBD_PROFILE") != nullptr;
        auto since = [](const std::chrono::steady_clock::time_point &start) {
          return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count();
        };
        for (const int substep_i : IndexRange(substeps_)) {
          const SubstepInterval substep(
              substeps_, substep_i, interpolation_begin_, interpolation_end_);
          std::chrono::steady_clock::time_point mark = std::chrono::steady_clock::now();
          this->simulate__update_pins__chunk(chunk_i, substep);
          /* The device path advances the whole fluid substep at once (grid, predict, constraint
           * iterations, velocity layer), so it replaces the inertial update below. */
          this->simulate__gpu_fluid_substep(
              chunk_i, solver_refs_i, substep, substep_i + 1 == substeps_);
          this->simulate__inertial_update__chunk(chunk_i, solver_refs_i);
          if (prof) {
            prof_coupled_ += since(mark);
          }
          mark = std::chrono::steady_clock::now();
          this->simulate__gather_dynamic_constraints__chunk(substep, chunk_i, solver_refs_i);
          this->simulate__reset_forces__chunk(chunk_i);
          /* The multiplier of every constraint is reset once per substep, including the sets that
           * are shared by all chunks (a volume or cross edge constraint is not tied to a chunk). */
          this->simulate__reset_forces();
          if (prof) {
            prof_constraints_ += since(mark);
            mark = std::chrono::steady_clock::now();
          }

          const Span<xpbd::GeometryRef> solver_refs = geometries_.solver_refs[solver_refs_i];
          xpbd::ConstraintSetParams solve_params{solver_refs, sub_delta_time_};
          debug_check_finite("chunk after gather", solver_refs);
          for ([[maybe_unused]] const int iter_i : IndexRange(constraint_iterations_)) {
            xpbd::GaussSeidelUpdater updater{solver_refs};
            this->simulate__position_solve__single_iteration__chunk(
                chunk_i, solve_params, updater);
            debug_check_finite("chunk after iteration", solver_refs);
            chunk_error_squared = updater.total_error_squared();
            chunk_error_count = updater.total_error_count();
          }
          if (prof) {
            prof_iterations_ += since(mark);
            mark = std::chrono::steady_clock::now();
          }

          this->simulate__update_velocities__chunk(chunk_i, solver_refs_i);
          this->simulate__velocity_solve__chunk(chunk_i, solver_refs_i);
          if (prof) {
            prof_velocity_ += since(mark);
          }
          solver_refs_i = 1 - solver_refs_i;
        }
        this->simulate__ensure_final_data_in_outputs__chunk(chunk_i);

        if (chunk_error_count) {
          error_tls.local().total_error_squared += chunk_error_squared;
          error_tls.local().total_error_count += chunk_error_count;
        }
      });

      float total_error_squared = 0.0f;
      int total_error_count = 0;
      for (const ResidualErrorTLS &error : error_tls) {
        total_error_squared += error.total_error_squared;
        total_error_count += error.total_error_count;
      }
      if (total_error_count) {
        average_error_squared = total_error_squared / total_error_count;
      }
    }
    else {
      int solver_refs_i = 0;
      for (const int substep_i : IndexRange(substeps_)) {
        const SubstepInterval substep(
            substeps_, substep_i, interpolation_begin_, interpolation_end_);
        const bool prof = std::getenv("XPBD_PROFILE") != nullptr;
        std::chrono::steady_clock::time_point mark = std::chrono::steady_clock::now();
        auto since = [](const std::chrono::steady_clock::time_point &start) {
          return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
              .count();
        };
        this->parallel_for_each_chunk(16, [&](const int chunk_i) {
          this->simulate__update_pins__chunk(chunk_i, substep);
          this->simulate__inertial_update__chunk(chunk_i, solver_refs_i);
        });
        /* The collision surfaces of the coupled solids move with the simulation, so their BVH has
         * to be rebuilt before the contacts of this substep are gathered. */
        if (prof) {
          mark = std::chrono::steady_clock::now();
        }
        this->simulate__update_coupled_surfaces(solver_refs_i);
        if (prof) {
          prof_coupled_ += since(mark);
          mark = std::chrono::steady_clock::now();
        }
        this->simulate__update_self_contacts(solver_refs_i);
        this->simulate__update_hair_contacts(solver_refs_i);
        this->simulate__update_hair_solid_contacts(solver_refs_i);
        this->simulate__gather_dynamic_constraints(substep, solver_refs_i);
        this->simulate__reset_forces();
        /* A fluid that runs on the device owns its whole substep (grid, predict, the density
         * iterations and the velocity layer), so it is advanced here: after the host gathered the
         * constraints of the other chunks (the gather skips device fluids) and reset their
         * multipliers. Nothing else in this path reads a device fluid between the substeps. */
        this->simulate__gpu_fluid_substep_unchunked(substep, solver_refs_i);
        if (prof) {
          prof_constraints_ += since(mark);
          mark = std::chrono::steady_clock::now();
        }
        for ([[maybe_unused]] const int iter_i : IndexRange(constraint_iterations_)) {
          this->simulate__position_solve__single_iteration(solver_refs_i, average_error_squared);
          /* The fluid pushes the coupled cloth and soft bodies with the multipliers of this
           * iteration. Applied after the parallel solve so no chunk writes into another one. */
          this->simulate__fluid_boundary_reaction(solver_refs_i);
        }
        if (prof) {
          prof_iterations_ += since(mark);
          mark = std::chrono::steady_clock::now();
        }
        this->parallel_for_each_chunk(16, [&](const int chunk_i) {
          this->simulate__update_velocities__chunk(chunk_i, solver_refs_i);
        });
        this->simulate__velocity_solve(solver_refs_i);
        if (prof) {
          prof_velocity_ += since(mark);
          mark = std::chrono::steady_clock::now();
        }
        this->parallel_for_each_chunk(16, [&](const int chunk_i) {
          this->simulate__ensure_final_data_in_outputs__chunk(chunk_i);
        });
        if (prof) {
          prof_final_ += since(mark);
        }
        solver_refs_i = 1 - solver_refs_i;
      }
    }

    return Result{math::sqrt(average_error_squared)};
  }

  bool support_chunk_simulation() const
  {
    /* The self contacts of the solids are a constraint set that spans all geometries, so they need
     * the substep loop that owns the whole world (the multi chunk path solves every chunk
     * independently). */
    if (self_contacts_active_ || self_collision_) {
      return false;
    }
    /* The hair contacts also span all curve geometries, so they need the whole world loop. */
    if (hair_contacts_active_ || hair_self_collision_) {
      return false;
    }
    /* The hair/cloth/soft body segment contacts span geometries too. */
    if (hair_solid_active_) {
      return false;
    }
    /* The guide tethers attach child curves to guide curves, also across data keys. */
    if (guide_tether_active_) {
      return false;
    }
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      const GeometryData &geo_data = *geometries_.data[data_key_i];
      if (!geo_data.static_constraints.is_empty()) {
        return false;
      }
    }
    /* The two way fluid coupling needs the extra reaction phase between the constraint iterations,
     * which only the multi chunk path has. */
    for (const ChunkData &chunk_data : chunks_data_) {
      if (chunk_data.fluid.active && chunk_data.fluid.has_coupled) {
        return false;
      }
    }
    return true;
  }

  void prepare_solver_geometry_refs()
  {
    for (const int direction : IndexRange(2)) {
      Array<xpbd::GeometryRef> &refs = geometries_.solver_refs[direction];
      refs.reinitialize(geometries_.data_keys.size());
      for (const int data_key_i : geometries_.data_keys.index_range()) {
        GeometryData &geo_data = *geometries_.data[data_key_i];
        xpbd::GeometryRef &ref = refs[data_key_i];

        if (direction == 0) {
          ref.positions = geo_data.temp_positions;
          ref.prev_positions = geo_data.position_attr.span;
          if (geo_data.uses_rotation) {
            ref.rotations = geo_data.temp_rotations;
            ref.prev_rotations = geo_data.rotation_attr.span;
          }
        }
        else {
          ref.positions = geo_data.position_attr.span;
          ref.prev_positions = geo_data.temp_positions;
          if (geo_data.uses_rotation) {
            ref.rotations = geo_data.rotation_attr.span;
            ref.prev_rotations = geo_data.temp_rotations;
          }
        }
        ref.velocities = geo_data.velocity_attr.span;
        ref.inv_masses = geo_data.inv_masses;
        if (geo_data.uses_rotation) {
          ref.angular_velocities = geo_data.angular_velocity_attr.span;
          ref.moments_of_inertia = geo_data.moments_of_inertia;
          ref.inv_moments_of_inertia = geo_data.inv_moments_of_inertia;
        }
      }
    }
  }

  void simulate__update_pins__chunk(const int chunk_i, const SubstepInterval &substep)
  {
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const int data_key_i = chunk.data_key_i;
    GeometryData &geo_data = *geometries_.data[data_key_i];
    ChunkData &chunk_data = chunks_data_[chunk_i];

    /* Update animated pin positions. */
    for (const PinPositionConstraintChunkUsage &constraint_chunk_usage :
         chunk_data.pin_position_constraints)
    {
      const PinPositionConstraintUsage &constraint_usage =
          geo_data.pin_position_constraints[constraint_chunk_usage.constraint_usage_i];
      for (const int pin_i : constraint_chunk_usage.pin_range) {
        const float3 &begin_pos = constraint_usage.begin_positions[pin_i];
        const float3 &end_pos = constraint_usage.end_positions[pin_i];
        const float3 pin_pos = math::interpolate(begin_pos, end_pos, substep.interpolate_end);
        constraint_usage.current_positions[pin_i] = pin_pos;
      }
    }
    if (geo_data.uses_rotation) {
      for (const PinRotationConstraintChunkUsage &constraint_chunk_usage :
           chunk_data.pin_rotation_constraints)
      {
        const PinRotationConstraintUsage &constraint_usage =
            geo_data.pin_rotation_constraints[constraint_chunk_usage.constraint_usage_i];
        for (const int pin_i : constraint_chunk_usage.pin_range) {
          const math::Quaternion &begin_rot = constraint_usage.begin_rotations[pin_i];
          const math::Quaternion &end_rot = constraint_usage.end_rotations[pin_i];
          const math::Quaternion pin_rot = math::interpolate(
              begin_rot, end_rot, substep.interpolate_end);
          constraint_usage.current_rotations[pin_i] = pin_rot;
        }
      }
    }
  }

  void simulate__inertial_update__chunk(const int chunk_i, const int solver_refs_i)
  {
    if (chunks_data_[chunk_i].fluid.gpu_active) {
      /* The device already did the predict step for this chunk. */
      return;
    }
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const IndexRange points_range = chunk.points_range;
    const int data_key_i = chunk.data_key_i;
    GeometryData &geo_data = *geometries_.data[data_key_i];
    xpbd::GeometryRef &ref = geometries_.solver_refs[solver_refs_i][data_key_i];

    this->integrate_linear_velocities(sub_delta_time_,
                                      ref.prev_positions.slice(points_range),
                                      ref.positions.slice(points_range),
                                      geo_data.velocity_attr.span.slice(points_range),
                                      geo_data.inv_masses.as_span().slice(points_range),
                                      geo_data.external_force_attr.slice(points_range),
                                      geo_data.external_acceleration.as_span().slice(points_range));
    if (geo_data.uses_rotation) {
      this->integrate_angular_velocities(
          sub_delta_time_,
          ref.prev_rotations.slice(points_range),
          ref.rotations.slice(points_range),
          geo_data.angular_velocity_attr.span.slice(points_range),
          geo_data.inv_moments_of_inertia.as_span().slice(points_range),
          geo_data.external_torque_attr.slice(points_range));
    }
  }

  void simulate__gather_dynamic_constraints(const SubstepInterval &substep,
                                            const int solver_refs_i)
  {
    /* Debug: `BLENDER_XPBD_NO_FACE_CONTACTS=1` gathers no collider contacts, which is how the
     * device path (boundary particles, no contacts yet) can be compared with the CPU solve. */
    if (std::getenv("BLENDER_XPBD_NO_FACE_CONTACTS") != nullptr) {
      for (ChunkData &chunk_data : chunks_data_) {
        chunk_data.external_face_contacts = {};
        chunk_data.external_edge_contacts = {};
      }
      this->parallel_for_each_chunk(1, [&](const int chunk_i) {
        this->simulate__update_fluid_neighborhood__chunk(chunk_i, solver_refs_i);
      });
      return;
    }
    this->parallel_for_each_chunk(1, [&](const int chunk_i) {
      this->simulate__gather_dynamic_constraints__chunk(substep, chunk_i, solver_refs_i);
    });
  }

  void simulate__gather_dynamic_constraints__chunk(const SubstepInterval &substep,
                                                   const int chunk_i,
                                                   const int solver_refs_i,
                                                   const bool force = false)
  {
    if (chunks_data_[chunk_i].fluid.gpu_active && !force) {
      /* The device substep gathers its contacts itself, from its own predicted positions. */
      return;
    }
    const float max_distance = this->get_max_search_distance(sub_delta_time_);
    ChunkData &chunk_data = chunks_data_[chunk_i];
    const ExternalFaceContacts &prev_face_contacts = chunk_data.external_face_contacts;
    const ExternalEdgeContacts &prev_edge_contacts = chunk_data.external_edge_contacts;
    ExternalFaceContacts new_face_contacts;
    ExternalEdgeContacts new_edge_contacts;
    this->gather_contacts__mesh_colliders(chunk_i,
                                          max_distance,
                                          solver_refs_i,
                                          substep,
                                          prev_face_contacts,
                                          prev_edge_contacts,
                                          new_face_contacts,
                                          new_edge_contacts);

    for (const int i : IndexRange(new_face_contacts.points.size())) {
      new_face_contacts.collider_velocities.append(
          math::safe_divide(new_face_contacts.collider_motion[i], sub_delta_time_));
    }
    for (const int i : IndexRange(new_edge_contacts.point_pairs.size())) {
      new_edge_contacts.collider_velocities.append(
          math::safe_divide(new_edge_contacts.collider_motion[i], sub_delta_time_));
    }
    chunk_data.external_face_contacts = std::move(new_face_contacts);
    chunk_data.external_edge_contacts = std::move(new_edge_contacts);

    this->simulate__update_fluid_neighborhood__chunk(chunk_i, solver_refs_i);

    /* Contacts with the triangles of the coupled geometries (cloth, soft bodies). */
    this->gather_contacts__coupled_surfaces(
        chunk_i, max_distance, solver_refs_i, substep, chunk_data.coupled_face_contacts);
  }

  /**
   * Debug only (`BLENDER_XPBD_GPU_NEIGHBORS=1`): builds the same grid and the same neighbour list
   * on the GPU and compares them with the CPU result. The simulation itself keeps using the CPU
   * list, so this never changes a result; it exists so the device implementation can be verified on
   * real data from the existing fluid scenes instead of a synthetic scene.
   */
  void debug_compare_gpu_neighbors(const Span<float3> positions,
                                   const float support_radius,
                                   const xpbd::FluidNeighborList &cpu_list)
  {
    static const bool enabled = std::getenv("BLENDER_XPBD_GPU_NEIGHBORS") != nullptr;
    if (!enabled) {
      return;
    }
    /* Bounded so that playing a scene does not pay for it forever. */
    static std::atomic<int> run_count{0};
    if (run_count.fetch_add(1) >= 8) {
      return;
    }
    const int particle_count = int(positions.size());
    if (particle_count == 0) {
      return;
    }

    geometry::ha::HaXpbdGrid gpu_grid;
    if (!geometry::ha::ha_xpbd_grid_build(positions, support_radius, gpu_grid) ||
        !geometry::ha::ha_xpbd_neighbors_build(gpu_grid))
    {
      this->debug_gpu_neighbors_report("XPBD_GPU_NEIGHBORS unavailable particles=%d\n",
                                       particle_count);
      return;
    }

    const double grid_ms = gpu_grid.stats.grid_ms;
    const double neighbors_ms = gpu_grid.stats.neighbors_ms;
    Array<int> gpu_offsets(particle_count + 1);
    Array<int> gpu_indices(gpu_grid.neighbor_capacity > 0 ? gpu_grid.neighbor_capacity : 1);
    geometry::ha::ha_xpbd_neighbors_download(gpu_grid, gpu_offsets, gpu_indices);
    const int total = gpu_offsets[particle_count];

    /* Same list twice: the device path has to be deterministic, which for the atomics + local sort
     * combination is not free. */
    auto same_ints = [](const Span<int> a, const Span<int> b) {
      if (a.size() != b.size()) {
        return false;
      }
      for (const int i : a.index_range()) {
        if (a[i] != b[i]) {
          return false;
        }
      }
      return true;
    };
    bool deterministic = true;
    {
      geometry::ha::HaXpbdGrid second_grid;
      if (geometry::ha::ha_xpbd_grid_build(positions, support_radius, second_grid) &&
          geometry::ha::ha_xpbd_neighbors_build(second_grid))
      {
        Array<int> second_offsets(particle_count + 1);
        Array<int> second_indices(second_grid.neighbor_capacity > 0 ? second_grid.neighbor_capacity
                                                                  : 1);
        geometry::ha::ha_xpbd_neighbors_download(second_grid, second_offsets, second_indices);
        const int second_total = second_offsets[particle_count];
        deterministic =
            second_total == total &&
            same_ints(second_offsets.as_span(), gpu_offsets.as_span()) &&
            same_ints(second_indices.as_span().take_front(total),
                      gpu_indices.as_span().take_front(total));
      }
    }

    int mismatched_particles = 0;
    int first_bad_particle = -1;
    for (const int i : IndexRange(particle_count)) {
      const int cpu_begin = cpu_list.offsets[i];
      const int cpu_end = cpu_list.offsets[i + 1];
      const int gpu_begin = gpu_offsets[i];
      const int gpu_end = gpu_offsets[i + 1];
      bool mismatched = cpu_end - cpu_begin != gpu_end - gpu_begin;
      if (!mismatched) {
        for (int k = 0; k < cpu_end - cpu_begin; k++) {
          if (cpu_list.indices[cpu_begin + k] != gpu_indices[gpu_begin + k]) {
            mismatched = true;
            break;
          }
        }
      }
      if (mismatched) {
        mismatched_particles++;
        if (first_bad_particle < 0) {
          first_bad_particle = i;
        }
      }
    }
    if (cpu_list.offsets.last() != total) {
      mismatched_particles = std::max(mismatched_particles, 1);
    }

    printf("XPBD_GPU_NEIGHBORS particles=%d neighbors=%d cpu_neighbors=%d "
           "mismatched_particles=%d first_bad=%d deterministic=%d grid_ms=%.3f neighbors_ms=%.3f "
           "passes=%d/%d\n",
           particle_count,
           total,
           cpu_list.offsets.last(),
           mismatched_particles,
           first_bad_particle,
           deterministic ? 1 : 0,
           grid_ms,
           neighbors_ms,
           gpu_grid.stats.grid_passes,
           gpu_grid.stats.neighbor_passes);
    fflush(stdout);
    this->debug_gpu_neighbors_report(
        "XPBD_GPU_NEIGHBORS particles=%d neighbors=%d cpu_neighbors=%d mismatched_particles=%d "
        "first_bad=%d deterministic=%d grid_ms=%.3f neighbors_ms=%.3f passes=%d/%d\n",
        particle_count,
        total,
        cpu_list.offsets.last(),
        mismatched_particles,
        first_bad_particle,
        deterministic ? 1 : 0,
        grid_ms,
        neighbors_ms,
        gpu_grid.stats.grid_passes,
        gpu_grid.stats.neighbor_passes);
  }

  /**
   * Mirrors the debug comparison into the file named by `BLENDER_XPBD_GPU_NEIGHBORS`, so the
   * regression test can assert on the numbers (the C++ output of the evaluator is not reachable
   * from Python).
   */
  void debug_gpu_neighbors_report(const char *format, ...)
  {
    va_list args;
    va_start(args, format);
    debug_gpu_write_report("BLENDER_XPBD_GPU_NEIGHBORS", format, args);
    va_end(args);
  }

  /** Milliseconds on the steady clock, for the device debug log. */
  static double time_now_ms()
  {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  /** Appends one line to the file named by \a env_name, unless the variable names no file. */
  static void debug_gpu_write_report(const char *env_name, const char *format, va_list args)  {
    const char *path = std::getenv(env_name);
    if (path == nullptr || path[0] == '\0' || (path[0] == '1' && path[1] == '\0')) {
      return;
    }
    FILE *file = fopen(path, "a");
    if (file == nullptr) {
      return;
    }
    vfprintf(file, format, args);
    fclose(file);
  }

  /** Logs one line to \a env_name and (when \a print_too) to stdout. */
  static void debug_gpu_log(const char *env_name, const bool print_too, const char *format, ...)
  {
    va_list args;
    va_start(args, format);
    if (print_too) {
      va_list console;
      va_copy(console, args);
      vprintf(format, console);
      va_end(console);
      fflush(stdout);
    }
    debug_gpu_write_report(env_name, format, args);
    va_end(args);
  }

  /**
   * Rebuilds the neighbor grid of the fluid once per substep. The grid is not rebuilt for every
   * constraint iteration because particles only move a fraction of the support radius per
   * iteration.
   */
  void simulate__update_fluid_neighborhood__chunk(const int chunk_i, const int solver_refs_i)
  {
    FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    if (!fluid.active || fluid.gpu_active) {
      return;
    }
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const Span<float3> positions = geometries_.solver_refs[solver_refs_i][chunk.data_key_i]
                                       .positions.slice(chunk.points_range);
    fluid.grid.build(positions, fluid.support_radius);
    xpbd::fluid_build_neighbors(fluid.grid, positions, fluid.neighbors);
    this->debug_compare_gpu_neighbors(positions, fluid.support_radius, fluid.neighbors);
    fluid.lambdas.fill(0.0f);
    fluid.coupled_push.fill(float3(0.0f));

    this->simulate__update_fluid_boundary__chunk(chunk_i, solver_refs_i);

    /* The neighbour sets are frozen for the substep (the grid is only rebuilt here), so the
     * boundary and coupled lists can be built once as well. */
    if (fluid.boundary_active) {
      xpbd::fluid_build_neighbors_at(
          fluid.boundary_grid, positions, fluid.boundary_positions, fluid.boundary_neighbors);
    }
    if (fluid.has_coupled) {
      xpbd::fluid_build_neighbors_at(
          fluid.coupled_grid, positions, fluid.coupled_positions, fluid.coupled_neighbors);
    }

    if (std::getenv("BLENDER_XPBD_DEBUG") != nullptr) {
      for (const int i : IndexRange(positions.size())) {
        const float3 &p = positions[i];
        if (!(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))) {
          printf(
              "XPBD_DEBUG non-finite fluid position before the density solve at %d: "
              "(%.3f %.3f %.3f)\n",
              int(i + chunk.points_range.start()),
              double(p.x),
              double(p.y),
              double(p.z));
          break;
        }
      }
    }
  }

  /**
   * Refreshes the boundary particles that belong to other simulated geometries and rebuilds the
   * boundary grid. The collider samples are static for the whole frame, but the cloth and the soft
   * bodies move, so their positions have to be copied from the solver state every substep.
   */
  void simulate__update_fluid_boundary__chunk(const int chunk_i, const int solver_refs_i)
  {
    FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    if (!fluid.has_coupled) {
      return;
    }
    const Span<xpbd::GeometryRef> refs = geometries_.solver_refs[solver_refs_i];
    for (const int i : fluid.coupled_geometry.index_range()) {
      const Span<float3> positions = refs[fluid.coupled_geometry[i]].positions;
      const int3 &verts = fluid.coupled_vertices[i];
      const float3 &weights = fluid.coupled_weights[i];
      /* A sample follows its triangle, so a deforming solid moves it without resampling. */
      fluid.coupled_positions[i] = positions[verts[0]] * weights.x +
                                   positions[verts[1]] * weights.y +
                                   positions[verts[2]] * weights.z;
    }
    fluid.coupled_grid.build(fluid.coupled_positions, fluid.support_radius);
    fluid.coupled_corrections.fill(float3(0.0f));
  }

  /**
   * Samples boundary particles on the collider meshes that the fluid can reach. The sampling is
   * clipped to the bounding box of the fluid (plus the support radius), so a large ground plane
   * only contributes the part the fluid can touch.
   *
   * This is a per frame operation: the colliders are sampled at their end pose of the frame. The
   * grid is reused for all substeps of the frame.
   */
  void prepare_fluid_boundaries()
  {
    this->parallel_for_each_chunk(
        8, [&](const int chunk_i) { this->prepare_fluid_boundary__chunk(chunk_i); });
  }

  /**
   * Warns about a fluid whose particles are much further apart than their radius allows. The
   * density constraint is unilateral (it only solves compression), so an under-dense fluid gets no
   * correction at all and behaves like loose sand: the particles drift, hover and never settle
   * like a liquid, which is easy to mistake for a problem in the solver.
   */
  void report_fluid_density_warnings()
  {
    for (const int chunk_i : geometries_.chunks.index_range()) {
      const FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
      if (!fluid.active || fluid.last_density_error_mean > -0.05f) {
        continue;
      }
      const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
      const std::string &path =
          geometries_.geometry_sets[geometries_.data_keys[chunk.data_key_i].geo_bundle_i].path;
      this->report(NodeWarningType::Warning,
                   fmt::format("Fluid \"{}\": the particles are about {:.0f}% below the rest "
                               "density, so they are further apart than their radius allows. Only "
                               "compression is solved, so nothing is solved and the fluid behaves "
                               "like loose sand instead of a liquid. Use a particle spacing of "
                               "about one particle radius.",
                               path,
                               double(-100.0f * fluid.last_density_error_mean)));
    }
  }

  static bool bounds_overlap(const float3 &v0,
                             const float3 &v1,
                             const float3 &v2,
                             const float3 &bounds_min,
                             const float3 &bounds_max)
  {
    for (const int axis : IndexRange(3)) {
      const float tri_min = std::min({v0[axis], v1[axis], v2[axis]});
      const float tri_max = std::max({v0[axis], v1[axis], v2[axis]});
      if (tri_max < bounds_min[axis] || tri_min > bounds_max[axis]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Collects the triangle indices whose bounding boxes overlap \a bounds_min / \a bounds_max.
   * The BVH range query is a sphere around the box; the caller still runs the exact triangle test,
   * so a slightly larger candidate set is fine. This is what keeps the boundary sampling of a large
   * collider proportional to the fluid's neighbourhood instead of the whole mesh.
   */
  static void collect_triangles_in_bounds(const BVHTree &tree,
                                          const float3 &bounds_min,
                                          const float3 &bounds_max,
                                          Vector<int> &r_indices)
  {
    r_indices.clear();
    const float3 center = (bounds_min + bounds_max) * 0.5f;
    const float radius = math::length(bounds_max - bounds_min) * 0.5f;
    BLI_bvhtree_range_query_cpp(
        tree, center, radius, [&](const int index, const float3 & /*co*/, const float /*dist_sq*/) {
          r_indices.append(index);
        });
    /* Process the triangles in mesh order, like the old loop over every triangle did. This keeps
     * the boundary particle order (and therefore the floating point summation order) stable. */
    std::sort(r_indices.begin(), r_indices.end());
  }

  /**
   * Appends one boundary particle per sub triangle, recursively splitting until the triangles are
   * smaller than the particle spacing. Triangles outside the bounds are skipped early, so the cost
   * is proportional to the area that the fluid can reach instead of the whole collider.
   */
  static void sample_boundary_triangle(const float3 &v0,
                                       const float3 &v1,
                                       const float3 &v2,
                                       const float spacing,
                                       const float3 &bounds_min,
                                       const float3 &bounds_max,
                                       Vector<float3> &r_positions,
                                       Vector<float> &r_areas,
                                       const int depth)
  {
    if (depth >= 16 || !bounds_overlap(v0, v1, v2, bounds_min, bounds_max)) {
      return;
    }
    const float max_edge = std::max(
        {math::length(v1 - v0), math::length(v2 - v1), math::length(v0 - v2)});
    if (max_edge > spacing) {
      const float3 v01 = (v0 + v1) * 0.5f;
      const float3 v12 = (v1 + v2) * 0.5f;
      const float3 v20 = (v2 + v0) * 0.5f;
      sample_boundary_triangle(
          v0, v01, v20, spacing, bounds_min, bounds_max, r_positions, r_areas, depth + 1);
      sample_boundary_triangle(
          v01, v1, v12, spacing, bounds_min, bounds_max, r_positions, r_areas, depth + 1);
      sample_boundary_triangle(
          v20, v12, v2, spacing, bounds_min, bounds_max, r_positions, r_areas, depth + 1);
      sample_boundary_triangle(
          v01, v12, v20, spacing, bounds_min, bounds_max, r_positions, r_areas, depth + 1);
      return;
    }
    const float3 center = (v0 + v1 + v2) * (1.0f / 3.0f);
    if (center.x < bounds_min.x || center.x > bounds_max.x || center.y < bounds_min.y ||
        center.y > bounds_max.y || center.z < bounds_min.z || center.z > bounds_max.z)
    {
      return;
    }
    r_positions.append(center);
    r_areas.append(0.5f * math::length(math::cross(v1 - v0, v2 - v0)));
  }

  void prepare_fluid_boundary__chunk(const int chunk_i)
  {
    FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    fluid.boundary_positions.clear();
    fluid.boundary_masses.clear();
    fluid.boundary_grid.build(Span<float3>(), 0.0f);
    fluid.coupled_positions.clear();
    fluid.coupled_masses.clear();
    fluid.coupled_inv_masses.clear();
    fluid.coupled_geometry.clear();
    fluid.coupled_vertices.clear();
    fluid.coupled_weights.clear();
    fluid.coupled_corrections.reinitialize(0);
    fluid.coupled_scratch.clear();
    fluid.coupled_grid.build(Span<float3>(), 0.0f);
    fluid.has_coupled = false;
    if (!fluid.active) {
      return;
    }
    const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];

    /* Bounds of the fluid, expanded by the support radius. */
    float3 bounds_min(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 bounds_max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const float3 &pos : geo_data.position_attr.span) {
      bounds_min = math::min(bounds_min, pos);
      bounds_max = math::max(bounds_max, pos);
    }
    if (bounds_min.x > bounds_max.x) {
      return;
    }
    const float margin = 2.0f * fluid.support_radius;
    bounds_min -= float3(margin);
    bounds_max += float3(margin);

    const float spacing = 0.5f * fluid.support_radius;
    if (fluid.boundary_active) {
      const float density = constraints_.fluid_constraints[fluid.constraint_i].rest_density;
      this->prepare_fluid_boundary__colliders(
          fluid, geo_data, spacing, density, bounds_min, bounds_max);
      /* The samples are spaced at half the support radius, but the grid is queried over the whole
       * kernel support, like the fluid grid itself. */
      fluid.boundary_grid.build(fluid.boundary_positions, fluid.support_radius);
    }

    const int coupled_num = this->prepare_fluid_boundary__solids(
        fluid, chunk, bounds_min, bounds_max);
    fluid.has_coupled = coupled_num > 0;
    if (fluid.has_coupled) {
      fluid.coupled_corrections.reinitialize(fluid.coupled_positions.size());
      fluid.coupled_corrections.fill(float3(0.0f));
      /* Full kernel support: the coupled particles have to contribute to the density exactly like

       * * the fluid particles do, otherwise a coarse cloth cannot hold any water at all. */
      fluid.coupled_grid.build(fluid.coupled_positions, fluid.support_radius);
    }

    if (std::getenv("BLENDER_FLUID_DEBUG") != nullptr) {
      printf(
          "FLUID_DEBUG boundary: %d collider samples, %d coupled particles (spacing %.4f) for %d "
          "colliders\n",
          int(fluid.boundary_positions.size()),
          coupled_num,
          double(spacing),
          int(geo_data.mesh_colliders.size()));
    }
  }

  /**
   * Samples the collider meshes the fluid can reach as (static) boundary particles.
   */
  void prepare_fluid_boundary__colliders(FluidChunkData &fluid,
                                         const GeometryData &geo_data,
                                         const float spacing,
                                         const float density,
                                         const float3 &bounds_min,
                                         const float3 &bounds_max)
  {
    /* Samples of one triangle, collected before they are added so they can share the same mass. */
    Vector<float3> positions;
    Vector<float> areas;

    for (const MeshColliderUsage &collider_usage : geo_data.mesh_colliders) {
      const MeshCollider &collider = constraints_.mesh_colliders[collider_usage.constraint_i];
      const Mesh *mesh = nullptr;
      if (std::holds_alternative<StaticMeshInfo>(collider.mesh)) {
        mesh = std::get<StaticMeshInfo>(collider.mesh).mesh;
      }
      else if (std::holds_alternative<DeformingMeshInfo>(collider.mesh)) {
        const auto &deforming = std::get<DeformingMeshInfo>(collider.mesh);
        mesh = deforming.substep_meshes.last();
      }
      if (mesh == nullptr) {
        continue;
      }
      const float4x4 &mesh_to_local = collider.end_transform;
      const Span<int3> corner_tris = mesh->corner_tris();
      const Span<int> corner_verts = mesh->corner_verts();
      const Span<float3> vert_positions = mesh->vert_positions();

      /* Only visit the triangles whose BVH bounds overlap the fluid. Looping over every triangle
       * of a large collider (a 20 m ground plane subdivided into thousands of faces) made the CPU
       * path stall before the fluid even touched it. */
      const BVHTree *collider_tree = nullptr;
      if (std::holds_alternative<StaticMeshInfo>(collider.mesh)) {
        collider_tree = std::get<StaticMeshInfo>(collider.mesh).corner_tris_bvh.tree;
      }
      else if (std::holds_alternative<DeformingMeshInfo>(collider.mesh)) {
        const auto &deforming = std::get<DeformingMeshInfo>(collider.mesh);
        if (!deforming.substep_corner_tris_bvh_trees.is_empty()) {
          collider_tree = deforming.substep_corner_tris_bvh_trees.last().tree;
        }
      }
      Vector<int> candidate_tris;
      if (collider_tree != nullptr) {
        /* The BVH stores the mesh-space triangles; the fluid bounds are in simulation space. */
        const float4x4 local_to_mesh = math::invert(mesh_to_local);
        float3 local_bounds_min(FLT_MAX);
        float3 local_bounds_max(-FLT_MAX);
        for (const int corner_i : IndexRange(8)) {
          const float3 corner((corner_i & 1) ? bounds_max.x : bounds_min.x,
                              (corner_i & 2) ? bounds_max.y : bounds_min.y,
                              (corner_i & 4) ? bounds_max.z : bounds_min.z);
          const float3 local_corner = math::transform_point(local_to_mesh, corner);
          local_bounds_min = math::min(local_bounds_min, local_corner);
          local_bounds_max = math::max(local_bounds_max, local_corner);
        }
        collect_triangles_in_bounds(
            *collider_tree, local_bounds_min, local_bounds_max, candidate_tris);
      }

      const auto add_triangle = [&](const int tri_i) {
        const int3 &tri = corner_tris[tri_i];
        const float3 v0 = math::transform_point(mesh_to_local,
                                                vert_positions[corner_verts[tri[0]]]);
        const float3 v1 = math::transform_point(mesh_to_local,
                                                vert_positions[corner_verts[tri[1]]]);
        const float3 v2 = math::transform_point(mesh_to_local,
                                                vert_positions[corner_verts[tri[2]]]);
        if (!bounds_overlap(v0, v1, v2, bounds_min, bounds_max)) {
          return;
        }
        positions.clear();
        areas.clear();
        sample_boundary_triangle(v0, v1, v2, spacing, bounds_min, bounds_max, positions, areas, 0);
        if (positions.is_empty()) {
          return;
        }
        float area_sum = 0.0f;
        for (const float area : areas) {
          area_sum += area;
        }
        /* Each boundary particle represents a patch of the surface with the fluid volume behind
         * it, so its mass is the rest density times the sampled volume. The shell has to be thin:
         * a settled fluid one spacing above the floor already has almost the rest density from its
         * own particles, so a full thickness of a particle spacing over-supplies the density and
         * inflates the fluid (measured, see M56). The coupled cloth uses a different rule
         * (`prepare_fluid_boundary__solids`) because water sits directly on it. */
        constexpr float thickness_factor = 0.25f;
        const float point_mass = density * (area_sum * thickness_factor * spacing) /
                                 positions.size();
        for (const int i : positions.index_range()) {
          fluid.boundary_positions.append(positions[i]);
          fluid.boundary_masses.append(point_mass);
        }
      };

      if (collider_tree != nullptr) {
        for (const int tri_i : candidate_tris) {
          add_triangle(tri_i);
        }
      }
      else {
        for (const int tri_i : IndexRange(corner_tris.size())) {
          add_triangle(tri_i);
        }
      }
    }
  }

  /**
   * Adds the surface of the other simulated geometries of the world (cloth, soft bodies) that the
   * fluid can reach to the boundary particles. They take part in the density estimate like the
   * collider samples, but they have an inverse mass and are moved by the reaction pass, which is
   * what couples the fluid and the solids in both directions.
   *
   * The particles are *not* the points of the solid: its triangles are sampled at half the support
   * radius of the fluid (the same density the collider sampler uses), so a coarse cloth behaves
   * like a continuous sheet instead of a net, and a fine one does not cost more than the fluid
   * needs. Each sample keeps the barycentric weights of its triangle, which is how the reaction is
   * spread over the vertices again (see #simulate__fluid_boundary_reaction).
   *
   * Geometries without faces (a point cloud, a curve) fall back to one particle per point.
   *
   * \return The number of particles that were added.
   */
  int prepare_fluid_boundary__solids(FluidChunkData &fluid,
                                     const GeometryDataChunk &chunk,
                                     const float3 &bounds_min,
                                     const float3 &bounds_max)
  {
    const FluidConstraint &constraint = constraints_.fluid_constraints[fluid.constraint_i];
    if (!(constraint.coupling > 0.0f)) {
      return 0;
    }
    /* Samples spaced half a support radius apart. The mass of a sample comes from the area it
     * stands for, so the surface density the fluid sees is the same no matter how dense the
     * sampling turns out to be (the subdivision of a triangle does not land on an exact grid). The
     * factor makes a flat patch supply the density of a solid half space: a plane at distance 0
     * integrates `1.2305 / h` of the poly6 kernel, so `mass = area * rest_density * h / 1.2305`
     * (measured to be the value that actually holds water, see M60). */
    const float spacing = 0.5f * fluid.support_radius;
    const float mass_per_area = constraint.rest_density * fluid.support_radius / 1.2305f;
    if (!(mass_per_area > 0.0f)) {
      return 0;
    }

    int coupled_num = 0;
    Vector<float3> samples;
    Vector<float> areas;
    const int self_data_key_i = chunk.data_key_i;
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      if (data_key_i == self_data_key_i) {
        continue;
      }
      const GeometryData &other = *geometries_.data[data_key_i];
      /* Only point clouds, meshes and curve points: those are the particle based simulations. */
      if (other.domain != bke::AttrDomain::Point) {
        continue;
      }
      /* Fluids couple to the fluid solve instead. */
      if (other.fluid_constraint_i >= 0) {
        continue;
      }
      if (other.size == 0 || other.inv_masses.size() != other.size) {
        continue;
      }
      const Span<float3> positions = other.position_attr.span;
      const Span<float> inv_masses = other.inv_masses;
      const Mesh *mesh = geometries_.geometry_sets[geometries_.data_keys[data_key_i].geo_bundle_i]
                             .geometry.get_mesh();
      const Span<int3> corner_tris = mesh != nullptr ? mesh->corner_tris() : Span<int3>();

      if (!corner_tris.is_empty()) {
        /* A surface: sample it, so the resolution of the solid does not matter. Only visit the
         * triangles whose BVH bounds overlap the fluid, like the collider sampling above. */
        const Span<int> corner_verts = mesh->corner_verts();
        const bke::BVHTreeFromMesh &mesh_bvh = mesh->bvh_corner_tris();
        Vector<int> candidate_tris;
        if (mesh_bvh.tree != nullptr) {
          collect_triangles_in_bounds(*mesh_bvh.tree, bounds_min, bounds_max, candidate_tris);
        }
        const auto add_triangle = [&](const int tri_i) {
          const int3 &tri = corner_tris[tri_i];
          const int3 verts(corner_verts[tri[0]], corner_verts[tri[1]], corner_verts[tri[2]]);
          const float3 v0 = positions[verts[0]];
          const float3 v1 = positions[verts[1]];
          const float3 v2 = positions[verts[2]];
          if (!bounds_overlap(v0, v1, v2, bounds_min, bounds_max)) {
            return;
          }
          samples.clear();
          areas.clear();
          sample_boundary_triangle(v0, v1, v2, spacing, bounds_min, bounds_max, samples, areas, 0);
          for (const int sample_i : samples.index_range()) {
            const float3 &sample = samples[sample_i];
            const float3 weights = triangle_weights(v0, v1, v2, sample);
            fluid.coupled_positions.append(sample);
            fluid.coupled_masses.append(areas[sample_i] * mass_per_area);
            fluid.coupled_inv_masses.append(inv_masses[verts[0]] * weights.x +
                                            inv_masses[verts[1]] * weights.y +
                                            inv_masses[verts[2]] * weights.z);
            fluid.coupled_geometry.append(data_key_i);
            fluid.coupled_vertices.append(verts);
            fluid.coupled_weights.append(weights);
            coupled_num += 1;
          }
        };
        if (mesh_bvh.tree != nullptr) {
          for (const int tri_i : candidate_tris) {
            add_triangle(tri_i);
          }
        }
        else {
          for (const int tri_i : IndexRange(corner_tris.size())) {
            add_triangle(tri_i);
          }
        }
        continue;
      }

      /* No faces: one particle per point. Its mass has to come from the point spacing, because the
       * points stand in for the solid volume the fluid would otherwise occupy. A locally flat
       * layer of points spaced `s` contributes `~0.31 / s^3 * m` to the density estimate, so `1.6
       * * rest_density * s^3` makes the solid provide the missing half of the density at its
       * surface. Too light and the fluid simply falls through (measured). */
      const float point_spacing = estimate_point_spacing(positions);
      const float point_mass = 1.6f * constraint.rest_density * point_spacing * point_spacing *
                               point_spacing;
      if (!(point_mass > 0.0f)) {
        continue;
      }
      for (const int i : IndexRange(other.size)) {
        const float3 &pos = positions[i];
        if (!bounds_contains(pos, bounds_min, bounds_max)) {
          continue;
        }
        /* Pinned points still take part in the density estimate (like the collider samples), they
         * just cannot be moved by the reaction. */
        fluid.coupled_positions.append(pos);
        fluid.coupled_masses.append(point_mass);
        fluid.coupled_inv_masses.append(inv_masses[i]);
        fluid.coupled_geometry.append(data_key_i);
        fluid.coupled_vertices.append(int3(i, i, i));
        fluid.coupled_weights.append(float3(1.0f, 0.0f, 0.0f));
        if (inv_masses[i] > 0.0f) {
          coupled_num += 1;
        }
      }
    }

    /* One scratch buffer per coupled geometry, for the per vertex accumulation of the reaction. */
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      if (fluid.coupled_geometry.first_index_of_try(data_key_i) < 0) {
        continue;
      }
      fluid.coupled_scratch.append({});
      FluidChunkData::CoupledVertexScratch &scratch = fluid.coupled_scratch.last();
      scratch.geometry_i = data_key_i;
      scratch.correction_sums.reinitialize(geometries_.data[data_key_i]->size);
      scratch.correction_sums.fill(float3(0.0f));
      scratch.weight_sums.reinitialize(geometries_.data[data_key_i]->size);
      scratch.weight_sums.fill(0.0f);
    }
    return coupled_num;
  }

  /** Barycentric weights of a point inside a triangle, clamped to the triangle. */
  static float3 triangle_weights(const float3 &v0,
                                 const float3 &v1,
                                 const float3 &v2,
                                 const float3 &point)
  {
    const float3 e0 = v1 - v0;
    const float3 e1 = v2 - v0;
    const float3 offset = point - v0;
    const float d00 = math::dot(e0, e0);
    const float d01 = math::dot(e0, e1);
    const float d11 = math::dot(e1, e1);
    const float d20 = math::dot(offset, e0);
    const float d21 = math::dot(offset, e1);
    const float denominator = d00 * d11 - d01 * d01;
    if (std::abs(denominator) < 1e-16f) {
      return float3(1.0f, 0.0f, 0.0f);
    }
    const float b = (d11 * d20 - d01 * d21) / denominator;
    const float c = (d00 * d21 - d01 * d20) / denominator;
    const float a = 1.0f - b - c;
    float3 weights(
        math::clamp(a, 0.0f, 1.0f), math::clamp(b, 0.0f, 1.0f), math::clamp(c, 0.0f, 1.0f));
    const float sum = weights.x + weights.y + weights.z;
    if (!(sum > 0.0f)) {
      return float3(1.0f, 0.0f, 0.0f);
    }
    weights /= sum;
    return weights;
  }

  /**
   * Rough point spacing of a point set, from its bounds and its size. Assumes that the points are
   * evenly distributed, which holds for the meshes and grids the solver deals with.
   */
  static float estimate_point_spacing(const Span<float3> positions)
  {
    if (positions.size() < 2) {
      return 0.0f;
    }
    float3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const float3 &pos : positions) {
      min = math::min(min, pos);
      max = math::max(max, pos);
    }
    float extents[3] = {max.x - min.x, max.y - min.y, max.z - min.z};
    std::sort(extents, extents + 3);
    const float largest = extents[2];
    if (!(largest > 0.0f)) {
      return 0.0f;
    }
    const float points_num = float(positions.size());
    if (extents[0] < 0.3f * largest) {
      /* A sheet (cloth, surface). */
      if (extents[1] > 0.0f) {
        return std::sqrt(extents[1] * largest / points_num);
      }
      /* A line (rod, hair). */
      return largest / points_num;
    }
    return std::cbrt(extents[0] * extents[1] * largest / points_num);
  }

  static bool bounds_contains(const float3 &pos,
                              const float3 &bounds_min,
                              const float3 &bounds_max)
  {
    for (const int axis : IndexRange(3)) {
      if (pos[axis] < bounds_min[axis] || pos[axis] > bounds_max[axis]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Creates a collision surface for every geometry that a fluid is coupled with, so the fluid can
   * also collide with the *triangles* of the cloth and the soft bodies (one sided, with the
   * friction of a real contact). The samples in `prepare_fluid_boundary__solids` handle the
   * density of a surface; these contacts handle fast or lone particles that a density estimate
   * cannot stop.
   */
  void prepare_coupled_surfaces()
  {
    VectorSet<int> geometry_ids;
    for (const ChunkData &chunk_data : chunks_data_) {
      if (!chunk_data.fluid.has_coupled) {
        continue;
      }
      for (const int geometry_i : chunk_data.fluid.coupled_geometry) {
        geometry_ids.add(geometry_i);
      }
    }
    for (const int geometry_i : geometry_ids) {
      const GeometryData &geo_data = *geometries_.data[geometry_i];
      const Mesh *mesh = geometries_.geometry_sets[geometries_.data_keys[geometry_i].geo_bundle_i]
                             .geometry.get_mesh();
      if (mesh == nullptr || mesh->corner_tris().is_empty() || mesh->verts_num != geo_data.size) {
        continue;
      }
      coupled_surfaces_.append({});
      CoupledSurface &surface = coupled_surfaces_.last();
      surface.geometry_i = geometry_i;
      surface.mesh = BKE_mesh_copy_for_eval(*mesh);
      surface.prev_positions.reinitialize(mesh->verts_num);
      surface.prev_positions.as_mutable_span().copy_from(mesh->vert_positions());
    }
  }

  void free_coupled_surfaces()
  {
    for (CoupledSurface &surface : coupled_surfaces_) {
      if (surface.mesh != nullptr) {
        BKE_id_free(nullptr, surface.mesh);
        surface.mesh = nullptr;
      }
    }
    coupled_surfaces_.clear();
  }

  void free_hair_solid_surfaces()
  {
    for (CoupledSurface &surface : hair_solid_surfaces_) {
      if (surface.mesh != nullptr) {
        BKE_id_free(nullptr, surface.mesh);
        surface.mesh = nullptr;
      }
    }
    hair_solid_surfaces_.clear();
    hair_solid_surface_max_radius.clear();
    hair_solid_surface_point_start_.clear();
  }

  /**
   * Prepares the CPU segment/triangle contacts between the hair and the cloth/soft bodies of the
   * world.
   *
   * Hair points and solid vertices are put into one flat point space so one constraint set can move
   * both sides. The solid triangles are topology, so only their positions and the BVH trees are
   * refreshed every substep. The whole pass is skipped when there is no hair or no solid mesh in the
   * world.
   */
  void prepare_hair_solid_contacts()
  {
    HairSolidContacts &contacts = hair_solid_contacts_;
    contacts = {};
    hair_solid_active_ = false;
    this->free_hair_solid_surfaces();

    /* Cheap gate first: without a solid mesh there is nothing to couple the hair with, and the
     * per frame gather of the hair points is not free for a large hairstyle. */
    Vector<int> solid_geometry_ids;
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      const GeometryData &geo_data = *geometries_.data[data_key_i];
      if (geo_data.curves != nullptr || geo_data.fluid_constraint_i >= 0) {
        continue;
      }
      if (geo_data.domain != AttrDomain::Point) {
        continue;
      }
      if (geo_data.size == 0 || geo_data.inv_masses.size() != geo_data.size) {
        continue;
      }
      const DataKey &data_key = geometries_.data_keys[data_key_i];
      const GeometrySet &geometry = geometries_.geometry_sets[data_key.geo_bundle_i].geometry;
      const Mesh *mesh = geometry.get_mesh();
      if (mesh == nullptr || mesh->corner_tris().is_empty() || mesh->verts_num != geo_data.size) {
        continue;
      }
      solid_geometry_ids.append(data_key_i);
    }
    if (solid_geometry_ids.is_empty()) {
      return;
    }

    Vector<int> affected_geo_indices;
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      if (geo_data.curves == nullptr || geo_data.domain != AttrDomain::Point) {
        continue;
      }
      if (geo_data.size == 0 || geo_data.inv_masses.size() != geo_data.size) {
        continue;
      }
      const DataKey &data_key = geometries_.data_keys[data_key_i];
      GeometrySetData &geo_set_data = geometries_.geometry_sets[data_key.geo_bundle_i];
      GeometrySet &geometry = geo_set_data.geometry;
      if (!geometry.has(data_key.type)) {
        continue;
      }

      this->ensure_radius_loaded(data_key_i, 0.002f);
      const VArraySpan<float> radii = geo_data.radii;
      const Span<float3> positions = geometry_component_positions(geometry, data_key.type);
      if (positions.size() != geo_data.size) {
        continue;
      }
      Array<float> thickness_values;
      this->materialize_float_param(data_key_i,
                                    attribute_names::hair_contact_thickness,
                                    hair_contact_thickness_,
                                    0.0f,
                                    1.0e9f,
                                    thickness_values);

      const int start = contacts.point_geo.size();
      const int size = geo_data.size;
      for (const int point_i : IndexRange(size)) {
        contacts.point_geo.append(data_key_i);
        contacts.point_local.append(point_i);
        contacts.point_inv_mass.append(geo_data.is_hard_pinned[point_i] ?
                                           0.0f :
                                           geo_data.inv_masses[point_i]);
      }

      const bke::CurvesGeometry &curves = *geo_data.curves;
      const OffsetIndices<int> points_by_curve = curves.points_by_curve();
      const VArray<bool> cyclic = curves.cyclic();
      for (const int curve_i : points_by_curve.index_range()) {
        const IndexRange points = points_by_curve[curve_i];
        const int points_num = points.size();
        if (points_num < 2) {
          continue;
        }
        const int segments_num = points_num - (cyclic[curve_i] ? 0 : 1);
        for (const int local_i : IndexRange(segments_num)) {
          const int point_a = points[local_i];
          const int point_b = points[(local_i + 1) % points_num];
          contacts.segment_point_a.append(start + point_a);
          contacts.segment_point_b.append(start + point_b);
          contacts.segment_radius.append(std::max(radii[point_a], radii[point_b]));
          contacts.segment_thickness.append(
              std::max(thickness_values[point_a], thickness_values[point_b]));
        }
      }
      affected_geo_indices.append(data_key_i);
    }
    if (contacts.segment_point_a.is_empty()) {
      contacts = {};
      return;
    }

    for (const int data_key_i : solid_geometry_ids) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      const DataKey &data_key = geometries_.data_keys[data_key_i];
      GeometrySetData &geo_set_data = geometries_.geometry_sets[data_key.geo_bundle_i];
      GeometrySet &geometry = geo_set_data.geometry;
      const Mesh *mesh = geometry.get_mesh();

      this->ensure_radius_loaded(data_key_i, 0.01f);
      const VArraySpan<float> radii = geo_data.radii;
      float max_radius = 0.0f;
      for (const float radius : radii) {
        max_radius = std::max(max_radius, radius);
      }

      hair_solid_surfaces_.append({});
      CoupledSurface &surface = hair_solid_surfaces_.last();
      surface.geometry_i = data_key_i;
      surface.mesh = BKE_mesh_copy_for_eval(*mesh);
      surface.prev_positions.reinitialize(mesh->verts_num);
      surface.prev_positions.as_mutable_span().copy_from(mesh->vert_positions());
      hair_solid_surface_max_radius.append(max_radius);
      hair_solid_surface_point_start_.append(contacts.point_geo.size());

      const int start = contacts.point_geo.size();
      for (const int vertex_i : IndexRange(mesh->verts_num)) {
        contacts.point_geo.append(data_key_i);
        contacts.point_local.append(vertex_i);
        contacts.point_inv_mass.append(geo_data.is_hard_pinned[vertex_i] ?
                                           0.0f :
                                           geo_data.inv_masses[vertex_i]);
      }
      const Span<int3> corner_tris = mesh->corner_tris();
      const Span<int> corner_verts = mesh->corner_verts();
      for (const int3 &tri : corner_tris) {
        contacts.triangle_points.append(
            int3(start + corner_verts[tri[0]],
                 start + corner_verts[tri[1]],
                 start + corner_verts[tri[2]]));
      }
      affected_geo_indices.append(data_key_i);
    }

    if (contacts.triangle_points.is_empty()) {
      contacts = {};
      this->free_hair_solid_surfaces();
      return;
    }

    VectorSet<int> unique_geo_indices;
    for (const int data_key_i : affected_geo_indices) {
      unique_geo_indices.add(data_key_i);
    }
    contacts.affected_geo_indices.reserve(unique_geo_indices.size());
    for (const int data_key_i : unique_geo_indices) {
      contacts.affected_geo_indices.append(data_key_i);
    }
    hair_solid_active_ = true;
  }

  /**
   * The positions of the component that makes up \a data_key, in its own point space.
   */
  static Span<float3> geometry_component_positions(const GeometrySet &geometry,
                                                   const GeometryComponent::Type type)
  {
    switch (type) {
      case GeometryComponent::Type::Mesh:
        if (const Mesh *mesh = geometry.get_mesh()) {
          return mesh->vert_positions();
        }
        break;
      case GeometryComponent::Type::PointCloud:
        if (const PointCloud *pointcloud = geometry.get_pointcloud()) {
          return pointcloud->positions();
        }
        break;
      case GeometryComponent::Type::Curve:
        if (const Curves *curves_id = geometry.get_curves()) {
          return curves_id->geometry.wrap().positions();
        }
        break;
      default:
        break;
    }
    return {};
  }

  /**
   * The rest pose of a geometry, which is what tells a pair of points that merely touches at rest
   * (`Setup Structural Rest Data` and the soft body keep it) apart from a pair that folds onto
   * itself during the simulation. Both names are checked because a cloth and a soft body write
   * their own.
   */
  VArray<float3> lookup_rest_positions(const int data_key_i)
  {
    VArray<float3> rest_positions = this->lookup_attribute_optional<float3>(
        data_key_i, "rest_position", AttrDomain::Point);
    if (!rest_positions) {
      rest_positions = this->lookup_attribute_optional<float3>(
          data_key_i, "puncliga_rest_position", AttrDomain::Point);
    }
    return rest_positions;
  }

  /**
   * Gathers all simulated points of the solid geometries (cloth, soft bodies and any other
   * simulated geometry that is not a fluid) into one index space, and records which of them are
   * topologically connected and therefore must never push each other apart.
   *
   * Runs once per frame: the topology does not change during the substeps, only which pairs are
   * actually in contact does (they are rebuilt in #simulate__update_self_contacts).
   */
  void prepare_self_contacts()
  {
    SelfContacts &contacts = self_contacts_;
    contacts = {};
    self_contacts_active_ = false;

    /* The pairs that are connected by an edge or by the corners of a face, in global indices. */
    Vector<int2> bonds;
    /** Concatenated rest positions, and current positions as the fallback. */
    Vector<float3> rest_positions_combined;

    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      if (geo_data.domain != AttrDomain::Point) {
        continue;
      }
      if (geo_data.size == 0 || geo_data.inv_masses.size() != geo_data.size) {
        continue;
      }
      /* A fluid collides with itself through the density constraint and its points overlap by
       * design, so it is not part of this pass. */
      if (geo_data.fluid_constraint_i >= 0) {
        continue;
      }
      /* Hair and other curve networks have their own segment/segment contacts. The point/point pass
       * is for meshes (cloth, soft bodies); mixing both contact sets on the same curve points is
       * redundant and makes `Self Collision` depend on the hair solver. */
      if (geo_data.curves != nullptr) {
        continue;
      }
      const DataKey &data_key = geometries_.data_keys[data_key_i];
      GeometrySetData &geo_set_data = geometries_.geometry_sets[data_key.geo_bundle_i];
      GeometrySet &geometry = geo_set_data.geometry;
      if (!geometry.has(data_key.type)) {
        continue;
      }

      /* The switches live on the `Init Puncliga Data` node of the geometry. Without the attribute
       * the solver input is the fallback, which keeps hand built worlds working. */
      bool participates = self_collision_;
      if (const VArray<bool> values = *geo_data.attributes.lookup<bool>(
              attribute_names::self_collision, AttrDomain::Point))
      {
        participates = values.size() > 0 && values[0];
      }
      if (!participates) {
        continue;
      }
      Array<float> thickness_values;
      this->materialize_float_param(data_key_i,
                                    attribute_names::contact_thickness,
                                    contact_thickness_,
                                    0.0f,
                                    1.0e9f,
                                    thickness_values);

      this->ensure_radius_loaded(data_key_i);
      const VArraySpan<float> radii = geo_data.radii;
      const Span<float3> positions = geometry_component_positions(geometry, data_key.type);
      if (positions.size() != geo_data.size) {
        continue;
      }
      const VArray<float3> rest_attr = this->lookup_rest_positions(data_key_i);
      const VArraySpan<float3> rest_positions = rest_attr;
      const bool has_rest_positions = rest_positions.size() == positions.size();

      float max_radius = 0.0f;
      for (const float radius : radii) {
        max_radius = std::max(max_radius, radius);
      }
      if (max_radius <= 0.0f) {
        this->report(NodeWarningType::Warning,
                     fmt::format("Self collision: \"{}\" has no `radius` attribute, so its points "
                                 "only collide within the contact thickness (the `Init Puncliga "
                                 "Data` node writes a `Collision Radius`)",
                                 geo_set_data.path));
      }

      const int start = contacts.point_geo.size();
      const int size = geo_data.size;
      contacts.geometry_data_key.append(data_key_i);
      contacts.geometry_start.append(start);
      contacts.geometry_size.append(size);
      contacts.affected_geo_indices.append(data_key_i);
      const Span<float> inv_masses = geo_data.inv_masses;
      /**
       * A pinned point is a boundary condition for the contacts: if it keeps its mass, the pin
       * undoes its half of every contact correction in the next iteration, and a heavy stack creeps
       * through a pinned sheet (measured: a soft body with water on top went through a fully pinned
       * cloth in five frames with the default four iterations, and held with ten). A pin that is
       * already effectively rigid has to be treated like an infinite mass here, which costs nothing
       * because the pin does not move anyway.
       */
      Array<bool> rigid_pin(size, false);
      for (const PinPositionConstraintUsage &usage : geo_data.pin_position_constraints) {
        for (const int pin_i : usage.points.index_range()) {
          const int point_i = usage.points[pin_i];
          if (!IndexRange(size).contains(point_i)) {
            continue;
          }
          const float compliance_term = usage.compliances[pin_i] * substep_compliance_factor_;
          if (compliance_term < 0.01f * inv_masses[point_i]) {
            rigid_pin[point_i] = true;
          }
        }
      }
      for (const int point_i : IndexRange(size)) {
        contacts.point_geo.append(data_key_i);
        contacts.point_local.append(point_i);
        contacts.point_radius.append(radii[point_i]);
        contacts.point_thickness.append(thickness_values[point_i]);
        const bool pinned = geo_data.is_hard_pinned[point_i] || rigid_pin[point_i];
        contacts.point_inv_mass.append(pinned ? 0.0f : inv_masses[point_i]);
      }

      /* Rest positions of the world geometry are already in the space of the solver (the world is
       * not transformed), so nothing has to be converted here. */
      rest_positions_combined.reserve(rest_positions_combined.size() + size);
      for (const int point_i : IndexRange(size)) {
        rest_positions_combined.append(has_rest_positions ? rest_positions[point_i] :
                                                           positions[point_i]);
      }

      if (data_key.type == GeometryComponent::Type::Mesh) {
        const Mesh &mesh = *geometry.get_mesh();
        const Span<int2> edges = mesh.edges();
        for (const int2 &edge : edges) {
          bonds.append({start + edge[0], start + edge[1]});
        }
        /* The corners of a face are connected to each other as well: a cloth quad has two
         * diagonals and a tetrahedron has six edges, of which the mesh only stores four. */
        const OffsetIndices<int> faces = mesh.faces();
        const Span<int> corner_verts = mesh.corner_verts();
        for (const int face_i : IndexRange(faces.size())) {
          const IndexRange face = faces[face_i];
          if (face.size() < 3 || face.size() > 4) {
            continue;
          }
          for (const int i : IndexRange(face.size())) {
            for (const int j : IndexRange(i + 1, face.size() - i - 1)) {
              bonds.append({start + corner_verts[face[i]], start + corner_verts[face[j]]});
            }
          }
        }
      }
      else if (data_key.type == GeometryComponent::Type::Curve && geo_data.curves != nullptr) {
        /* Consecutive points of a curve are connected, and so are the last and the first point of
         * a cyclic one. */
        const bke::CurvesGeometry &curves = *geo_data.curves;
        const VArray<bool> cyclic = curves.cyclic();
        const OffsetIndices<int> curves_points = curves.points_by_curve();
        for (const int curve_i : curves_points.index_range()) {
          const IndexRange points = curves_points[curve_i];
          for (const int i : IndexRange(points.size())) {
            const int point_i0 = int(points[i]);
            if (i + 1 < points.size()) {
              bonds.append({start + point_i0, start + int(points[i + 1])});
            }
            else if (cyclic[curve_i]) {
              bonds.append({start + point_i0, start + int(points[0])});
            }
          }
        }
      }
    }

    const int points_num = contacts.point_geo.size();
    if (points_num == 0) {
      return;
    }
    self_contacts_active_ = true;
    contacts.positions.reinitialize(points_num);
    contacts.rest_positions.reinitialize(points_num);
    contacts.rest_positions.as_mutable_span().copy_from(rest_positions_combined.as_span());

    /* Connected islands, so the rest distance filter only applies to points that the material
     * actually connects (see SelfContacts::island_ids). */
    contacts.island_ids.reinitialize(points_num);
    for (const int point_i : IndexRange(points_num)) {
      contacts.island_ids[point_i] = point_i;
    }
    const auto find_island = [&](int point_i) {
      while (contacts.island_ids[point_i] != point_i) {
        contacts.island_ids[point_i] = contacts.island_ids[contacts.island_ids[point_i]];
        point_i = contacts.island_ids[point_i];
      }
      return point_i;
    };
    for (const int2 &bond : bonds) {
      const int root0 = find_island(bond[0]);
      const int root1 = find_island(bond[1]);
      if (root0 != root1) {
        contacts.island_ids[root0] = root1;
      }
    }
    for (const int point_i : IndexRange(points_num)) {
      contacts.island_ids[point_i] = find_island(point_i);
    }

    /* Build the per point list of connected points (CSR, sorted for a binary search). */
    Array<int> counts(points_num, 0);
    for (const int2 &bond : bonds) {
      counts[bond[0]]++;
      counts[bond[1]]++;
    }
    contacts.bond_offsets.reinitialize(points_num + 1);
    int offset = 0;
    for (const int point_i : IndexRange(points_num)) {
      contacts.bond_offsets[point_i] = offset;
      offset += counts[point_i];
    }
    contacts.bond_offsets[points_num] = offset;
    contacts.bond_indices.reinitialize(offset);
    if (offset > 0) {
      Array<int> cursors(points_num);
      for (const int point_i : IndexRange(points_num)) {
        cursors[point_i] = contacts.bond_offsets[point_i];
      }
      for (const int2 &bond : bonds) {
        contacts.bond_indices[cursors[bond[0]]++] = bond[1];
        contacts.bond_indices[cursors[bond[1]]++] = bond[0];
      }
      threading::parallel_for(IndexRange(points_num), 1024, [&](const IndexRange range) {
        for (const int point_i : range) {
          MutableSpan<int> list = contacts.bond_indices.as_mutable_span().slice(
              contacts.bond_offsets[point_i],
              contacts.bond_offsets[point_i + 1] - contacts.bond_offsets[point_i]);
          std::sort(list.begin(), list.end());
        }
      });
    }
  }

  /** Whether two points are connected by an edge or by a face and must not collide. */
  static bool self_contact_is_bonded(const SelfContacts &contacts, const int point_i0,
                                     const int point_i1)
  {
    if (contacts.bond_offsets.is_empty()) {
      return false;
    }
    const Span<int> bonded = contacts.bond_indices.as_span().slice(
        contacts.bond_offsets[point_i0],
        contacts.bond_offsets[point_i0 + 1] - contacts.bond_offsets[point_i0]);
    return std::binary_search(bonded.begin(), bonded.end(), point_i1);
  }

  /** Rebuilds the pairs that are close enough to collide at the current positions. */
  void build_self_contact_pairs(SelfContacts &contacts)
  {
    contacts.point_pairs.reinitialize(0);
    contacts.min_distances.reinitialize(0);
    contacts.compliances.reinitialize(0);
    contacts.lambdas.reinitialize(0);
    contacts.coloring = {};
    contacts.constraint_set.reset();

    const int points_num = contacts.point_geo.size();
    if (points_num < 2) {
      return;
    }

    float max_radius = 0.0f;
    for (const float radius : contacts.point_radius) {
      max_radius = std::max(max_radius, radius);
    }
    float max_thickness = 0.0f;
    for (const float thickness : contacts.point_thickness) {
      max_thickness = std::max(max_thickness, thickness);
    }
    /* Points within 25% of the contact distance are kept as candidates as well, so that a pair
     * that another constraint pushes into contact during the iterations is already in the list. */
    constexpr float probe_factor = 1.25f;
    const float cell_size = (2.0f * max_radius + max_thickness) * probe_factor;
    if (!(cell_size > 0.0f)) {
      return;
    }

    const Span<float3> positions = contacts.positions;
    const Span<float3> rest_positions = contacts.rest_positions;
    const Span<int> island_ids = contacts.island_ids;
    const Span<float> radii = contacts.point_radius;
    const Span<float> thickness = contacts.point_thickness;
    const float cell_sq = cell_size * cell_size;

    const auto is_candidate = [&](const int point_i0,
                                  const int point_i1,
                                  const float r_sq) -> bool {
      if (point_i1 <= point_i0 || r_sq >= cell_sq) {
        return false;
      }
      const float contact_distance = radii[point_i0] + radii[point_i1] +
                                     std::max(thickness[point_i0], thickness[point_i1]);
      const float probe = contact_distance * probe_factor;
      if (r_sq >= probe * probe) {
        return false;
      }
      if (self_contact_is_bonded(contacts, point_i0, point_i1)) {
        return false;
      }
      /* Inside one island, points that are already that close in the rest pose are neighbors in a
       * way the topology does not express (the third ring of a finely tessellated sheet); they
       * would push each other apart forever. Two *different* islands are never filtered: a stack of
       * cloth layers has to collide even if the layers were modeled a millimeter apart. */
      if (island_ids[point_i0] == island_ids[point_i1] &&
          math::length_squared(rest_positions[point_i0] - rest_positions[point_i1]) <
              probe * probe)
      {
        return false;
      }
      return true;
    };

    xpbd::FluidParticleGrid grid;
    grid.build(positions, cell_size);

    Array<int> counts(points_num, 0);
    threading::parallel_for(IndexRange(points_num), 1024, [&](const IndexRange range) {
      for (const int point_i0 : range) {
        int count = 0;
        grid.foreach_neighbor(
            point_i0, positions, [&](const int point_i1, const float3 &, const float r_sq) {
              count += is_candidate(point_i0, point_i1, r_sq) ? 1 : 0;
            });
        counts[point_i0] = count;
      }
    });

    Array<int> offsets(points_num + 1);
    int offset = 0;
    for (const int point_i : IndexRange(points_num)) {
      offsets[point_i] = offset;
      offset += counts[point_i];
    }
    offsets[points_num] = offset;
    const int pairs_num = offset;

    contacts.point_pairs.reinitialize(pairs_num);
    contacts.min_distances.reinitialize(pairs_num);
    contacts.compliances.reinitialize(pairs_num);
    contacts.compliances.fill(0.0f);

    threading::parallel_for(IndexRange(points_num), 1024, [&](const IndexRange range) {
      for (const int point_i0 : range) {
        int cursor = offsets[point_i0];
        grid.foreach_neighbor(
            point_i0, positions, [&](const int point_i1, const float3 &, const float r_sq) {
              if (!is_candidate(point_i0, point_i1, r_sq)) {
                return;
              }
              contacts.point_pairs[cursor] = int2(point_i0, point_i1);
              contacts.min_distances[cursor] = radii[point_i0] + radii[point_i1] +
                                               std::max(thickness[point_i0], thickness[point_i1]);
              cursor++;
            });
      }
    });

    if (std::getenv("XPBD_SELF_DEBUG") != nullptr) {
      int pinned_num = 0;
      for (const float inv_mass : contacts.point_inv_mass) {
        pinned_num += inv_mass == 0.0f ? 1 : 0;
      }
      printf("XPBD_SELF points %d bonds %d pairs %d pinned %d\n",
             points_num,
             int(contacts.bond_indices.size()),
             pairs_num,
             pinned_num);
    }
    if (pairs_num == 0) {
      return;
    }

    contacts.lambdas.reinitialize(pairs_num);
    contacts.lambdas.fill(0.0f);

    /* The colors are only valid for the pairs of this substep, so the memory of the previous
     * substep is released instead of accumulating over the frame. */
    self_contacts_memory_ = std::make_unique<ResourceScope>(size_t(1) << 16);
    contacts.coloring = xpbd::color_constraints__binary(contacts.point_pairs.as_span(),
                                                        self_contacts_memory_->allocator());
    contacts.constraint_set.emplace(contacts.affected_geo_indices,
                                    contacts.point_pairs.as_span(),
                                    contacts.min_distances.as_span(),
                                    contacts.compliances.as_span(),
                                    contacts.point_geo.as_span(),
                                    contacts.point_local.as_span(),
                                    contacts.point_inv_mass.as_span(),
                                    error_scale_from_threshold(0.01f),
                                    contacts.lambdas.as_mutable_span());
  }

  /**
   * Copies the current solver positions into the contact point space and rebuilds the pairs. Runs
   * once per substep, before the constraint iterations.
   */
  void simulate__update_self_contacts(const int solver_refs_i)
  {
    SelfContacts &contacts = self_contacts_;
    if (contacts.is_empty()) {
      return;
    }
    const Span<xpbd::GeometryRef> refs = geometries_.solver_refs[solver_refs_i];
    for (const int i : contacts.geometry_data_key.index_range()) {
      const IndexRange slice(contacts.geometry_start[i], contacts.geometry_size[i]);
      const Span<float3> positions = refs[contacts.geometry_data_key[i]].positions;
      if (positions.size() != slice.size()) {
        continue;
      }
      contacts.positions.as_mutable_span().slice(slice).copy_from(positions);
    }
    this->build_self_contact_pairs(contacts);
  }

  /**
   * Gathers all hair segments of the curve geometries into one index space and prepares the rest
   * positions, the inverse masses and the per segment displacement margins. Runs once per frame and
   * also builds the segment pairs: the margin makes the pair list conservative enough to stay valid
   * for every substep of the frame.
   */
  void prepare_hair_contacts()
  {
    HairSelfContacts &contacts = hair_contacts_;
    contacts = {};
    hair_contacts_active_ = false;

    const float frame_dt = sub_delta_time_ * float(substeps_);
    Vector<float3> positions_combined;
    Vector<float3> rest_positions_combined;
    Vector<float3> velocities_combined;
    Vector<float> geometry_contact_max;
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      if (geo_data.curves == nullptr) {
        continue;
      }
      if (geo_data.domain != AttrDomain::Point) {
        continue;
      }
      if (geo_data.size == 0 || geo_data.inv_masses.size() != geo_data.size) {
        continue;
      }
      const DataKey &data_key = geometries_.data_keys[data_key_i];
      GeometrySetData &geo_set_data = geometries_.geometry_sets[data_key.geo_bundle_i];
      GeometrySet &geometry = geo_set_data.geometry;
      if (!geometry.has(data_key.type)) {
        continue;
      }

      /* The switches live on the `Init Puncliga Data` node of the geometry. Without the attribute
       * the solver inputs are the fallback, which keeps hand built worlds working. */
      bool participates = hair_self_collision_;
      if (const VArray<bool> values = *geo_data.attributes.lookup<bool>(
              attribute_names::hair_self_collision, AttrDomain::Point))
      {
        participates = values.size() > 0 && values[0];
      }
      if (!participates) {
        continue;
      }
      Array<float> thickness_values;
      this->materialize_float_param(data_key_i,
                                    attribute_names::hair_contact_thickness,
                                    hair_contact_thickness_,
                                    0.0f,
                                    1.0e9f,
                                    thickness_values);
      bool allow_same_strand = hair_same_strand_collision_;
      if (const VArray<bool> strand_values = *geo_data.attributes.lookup<bool>(
              attribute_names::same_strand_collision, AttrDomain::Point))
      {
        allow_same_strand = strand_values.size() > 0 && strand_values[0];
      }

      this->ensure_radius_loaded(data_key_i, 0.002f);
      const VArraySpan<float> radii = geo_data.radii;
      const Span<float3> positions = geometry_component_positions(geometry, data_key.type);
      if (positions.size() != geo_data.size) {
        continue;
      }
      const VArray<float3> rest_attr = this->lookup_rest_positions(data_key_i);
      const VArraySpan<float3> rest_positions = rest_attr;
      const bool has_rest_positions = rest_positions.size() == positions.size();

      const int start = contacts.point_geo.size();
      const int size = geo_data.size;
      const int geometry_i = contacts.geometry_data_key.size();
      contacts.geometry_data_key.append(data_key_i);
      contacts.geometry_start.append(start);
      contacts.geometry_size.append(size);
      contacts.affected_geo_indices.append(data_key_i);

      Array<bool> rigid_pin(size, false);
      for (const PinPositionConstraintUsage &usage : geo_data.pin_position_constraints) {
        for (const int pin_i : usage.points.index_range()) {
          const int point_i = usage.points[pin_i];
          if (!IndexRange(size).contains(point_i)) {
            continue;
          }
          const float compliance_term = usage.compliances[pin_i] * substep_compliance_factor_;
          if (compliance_term < 0.01f * geo_data.inv_masses[point_i]) {
            rigid_pin[point_i] = true;
          }
        }
      }
      const Span<float3> velocities = geo_data.velocity_attr.span;
      for (const int point_i : IndexRange(size)) {
        velocities_combined.append(velocities.size() == size ? velocities[point_i] : float3(0.0f));
      }
      float max_radius = 0.0f;
      for (const float radius : radii) {
        max_radius = std::max(max_radius, radius);
      }
      float max_thickness = 0.0f;
      for (const float value : thickness_values) {
        max_thickness = std::max(max_thickness, value);
      }
      geometry_contact_max.append(2.0f * max_radius + max_thickness);
      for (const int point_i : IndexRange(size)) {
        contacts.point_geo.append(data_key_i);
        contacts.point_local.append(point_i);
        const bool pinned = geo_data.is_hard_pinned[point_i] || rigid_pin[point_i];
        contacts.point_inv_mass.append(pinned ? 0.0f : geo_data.inv_masses[point_i]);
        positions_combined.append(positions[point_i]);
        rest_positions_combined.append(has_rest_positions ? rest_positions[point_i] :
                                                           positions[point_i]);
      }

      const bke::CurvesGeometry &curves = *geo_data.curves;
      const OffsetIndices<int> points_by_curve = curves.points_by_curve();
      const VArray<bool> cyclic = curves.cyclic();
      for (const int curve_i : points_by_curve.index_range()) {
        const IndexRange points = points_by_curve[curve_i];
        const int points_num = points.size();
        if (points_num < 2) {
          continue;
        }
        const int segments_num = points_num - (cyclic[curve_i] ? 0 : 1);
        for (const int local_i : IndexRange(segments_num)) {
          const int point_a = points[local_i];
          const int point_b = points[(local_i + 1) % points_num];
          contacts.segment_geometry.append(geometry_i);
          contacts.segment_strand.append(curve_i);
          contacts.segment_point_a.append(start + point_a);
          contacts.segment_point_b.append(start + point_b);
          contacts.segment_radius.append(std::max(radii[point_a], radii[point_b]));
          contacts.segment_thickness.append(
              std::max(thickness_values[point_a], thickness_values[point_b]));
          contacts.segment_same_strand.append(allow_same_strand ? 1 : 0);
        }
      }
    }

    const int points_num = contacts.point_geo.size();
    if (points_num == 0) {
      return;
    }
    hair_contacts_active_ = true;
    contacts.positions.reinitialize(points_num);
    contacts.positions.as_mutable_span().copy_from(positions_combined.as_span());
    contacts.rest_positions.reinitialize(points_num);
    contacts.rest_positions.as_mutable_span().copy_from(rest_positions_combined.as_span());

    this->build_rest_close_pairs(contacts);

    /* The pair list is built once per frame, so every segment is expanded by how far it can move
     * relative to the rest of the hair before the next rebuild. Uniform accelerations (gravity,
     * wind) move all points together and do not change the distance between two strands, so the
     * margin uses the deviation from the mean velocity; that covers a swinging strand without
     * inflating the query for gravity. The margin is capped relative to the contact distance:
     * sparse grooms can afford a generous margin, dense grooms cannot (every extra pair is solved in
     * every iteration). A very dense groom (many segments already touch in the rest pose) keeps a
     * tight list. */
    const int segments_num = contacts.segment_geometry.size();
    contacts.segment_margin.reinitialize(segments_num);
    const float rest_ratio = float(contacts.rest_pair_count) / float(std::max(1, segments_num));
    float margin_scale = std::clamp(1.5f / (1.0f + 8.0f * rest_ratio), 0.25f, 1.5f);
    if (rest_ratio > 4.0f) {
      margin_scale = 0.0f;
    }
    if (const char *env = std::getenv("XPBD_HAIR_MARGIN_SCALE")) {
      margin_scale = std::max(float(std::atof(env)), 0.0f);
    }
    Array<float> point_margins(points_num, 0.0f);
    for (const int geometry_i : IndexRange(contacts.geometry_start.size())) {
      const IndexRange slice(contacts.geometry_start[geometry_i], contacts.geometry_size[geometry_i]);
      const float margin_cap = std::max(geometry_contact_max[geometry_i] * margin_scale, 1e-4f);
      float3 mean_velocity(0.0f);
      for (const int point_i : slice) {
        mean_velocity += velocities_combined[point_i];
      }
      mean_velocity /= float(slice.size());
      for (const int point_i : slice) {
        const float speed = frame_dt > 0.0f ?
                                math::length(velocities_combined[point_i] - mean_velocity) :
                                0.0f;
        point_margins[point_i] = std::min(speed * frame_dt * 1.5f + 1e-4f, margin_cap);
      }
    }
    for (const int segment_i : IndexRange(segments_num)) {
      contacts.segment_margin[segment_i] = std::max(
          point_margins[contacts.segment_point_a[segment_i]],
          point_margins[contacts.segment_point_b[segment_i]]);
    }

    /* Build the pairs once per frame; `simulate__update_hair_contacts` only mirrors the positions
     * (they are used for debugging), the contact solver reads the live solver positions. */
    this->build_hair_contact_pairs(contacts);
  }

  /**
   * Builds the compliant guide/child attachments.
   *
   * `Init Puncliga Data > Hair > Guide Tether` enables the pass for a geometry and
   * `Tether Compliance` is its softness. The guide of every child curve comes from the standard
   * `guide_curve_index` attribute: curves without a valid index are the guides of that geometry, and
   * the index selects one of them. Every child point is attached to the guide point at the same
   * normalized parameter, with the rest offset stored once per frame.
   *
   * The constraint is solved after the contacts, so a collision that moves a child also moves its
   * guide (the correction is split by inverse mass).
   */
  void prepare_guide_tethers()
  {
    GuideTetherContacts &contacts = guide_tether_contacts_;
    contacts = {};
    guide_tether_active_ = false;

    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      if (geo_data.curves == nullptr || geo_data.domain != AttrDomain::Point) {
        continue;
      }
      if (geo_data.size == 0 || geo_data.inv_masses.size() != geo_data.size) {
        continue;
      }

      bool enabled = false;
      if (const VArray<bool> values = *geo_data.attributes.lookup<bool>(
              attribute_names::guide_tether, AttrDomain::Point))
      {
        for (const int point_i : IndexRange(values.size())) {
          if (values[point_i]) {
            enabled = true;
            break;
          }
        }
      }
      if (!enabled) {
        continue;
      }

      Array<float> compliances;
      this->materialize_float_param(data_key_i,
                                    attribute_names::tether_compliance,
                                    0.0f,
                                    0.0f,
                                    1.0e9f,
                                    compliances);
      Array<bool> tether_values;
      this->materialize_bool_param(
          data_key_i, attribute_names::guide_tether, false, tether_values);
      const DataKey &data_key = geometries_.data_keys[data_key_i];
      GeometrySetData &geo_set_data = geometries_.geometry_sets[data_key.geo_bundle_i];

      const bke::CurvesGeometry &curves = *geo_data.curves;
      const OffsetIndices<int> points_by_curve = curves.points_by_curve();
      const VArray<int> guide_indices =
          geo_data.attributes
              .lookup_or_default<int>("guide_curve_index", AttrDomain::Curve, -1)
              .varray;
      const VArray<float3> rest_attr = this->lookup_rest_positions(data_key_i);
      const VArraySpan<float3> rest_positions = rest_attr;
      const Span<float3> positions = geo_data.position_attr.span;
      const bool has_rest_positions = rest_positions.size() == geo_data.size;

      /**
       * The curves that have `Guide Tether` on are the children, the others are the guides. A
       * separate `Init Puncliga Data` node per role is the intended setup: one for the guides with
       * the tether off, one for the children with it on. Every child curve then names its guide
       * with `guide_curve_index`, indexing the guide curves of the same geometry in curve order.
       */
      const auto curve_is_child = [&](const int curve_i) {
        const IndexRange points = points_by_curve[curve_i];
        return !points.is_empty() && tether_values[points[0]];
      };
      Vector<int> guide_curves;
      for (const int curve_i : points_by_curve.index_range()) {
        if (!curve_is_child(curve_i)) {
          guide_curves.append(curve_i);
        }
      }
      if (guide_curves.is_empty()) {
        this->report(NodeWarningType::Warning,
                     fmt::format("Guide Tether is enabled on \"{}\", but the geometry has no guide "
                                 "curves (a guide is a curve with the tether switched off, usually "
                                 "from a separate `Init Puncliga Data` node)",
                                 geo_set_data.path));
        continue;
      }

      const int offset = contacts.point_geo.size();
      for (const int point_i : IndexRange(geo_data.size)) {
        contacts.point_geo.append(data_key_i);
        contacts.point_local.append(point_i);
        contacts.point_inv_mass.append(geo_data.is_hard_pinned[point_i] ?
                                           0.0f :
                                           geo_data.inv_masses[point_i]);
      }

      constexpr int MAX_TETHERS = 1 << 20;
      for (const int curve_i : points_by_curve.index_range()) {
        if (contacts.point_pairs.size() >= MAX_TETHERS) {
          break;
        }
        if (!curve_is_child(curve_i)) {
          continue;
        }
        const int guide_i = guide_indices[curve_i];
        if (guide_i < 0 || guide_i >= guide_curves.size()) {
          continue;
        }
        const IndexRange child_points = points_by_curve[curve_i];
        const IndexRange guide_points = points_by_curve[guide_curves[guide_i]];
        if (child_points.is_empty() || guide_points.is_empty()) {
          continue;
        }
        const int child_num = child_points.size();
        const int guide_num = guide_points.size();
        for (const int child_k : IndexRange(child_num)) {
          if (contacts.point_pairs.size() >= MAX_TETHERS) {
            break;
          }
          const float f = child_num > 1 ? float(child_k) / float(child_num - 1) : 0.0f;
          const int guide_k = int(std::floor(f * float(guide_num - 1) + 0.5f));
          const int child_point = child_points[child_k];
          const int guide_point = guide_points[std::min(guide_k, guide_num - 1)];
          contacts.point_pairs.append(int2(offset + child_point, offset + guide_point));
          const float3 rest_offset =
              has_rest_positions ?
                  rest_positions[child_point] - rest_positions[guide_point] :
                  positions[child_point] - positions[guide_point];
          contacts.rest_offsets.append(rest_offset);
          contacts.compliances.append(compliances[child_point]);
        }
      }
      contacts.affected_geo_indices.append(data_key_i);
    }

    if (contacts.point_pairs.is_empty()) {
      contacts = {};
      return;
    }

    const int tethers_num = contacts.point_pairs.size();
    contacts.lambdas.reinitialize(tethers_num);
    contacts.lambdas.fill(float3(0.0f));
    guide_tether_memory_ = std::make_unique<ResourceScope>(size_t(1) << 16);
    contacts.coloring = xpbd::color_constraints__binary(contacts.point_pairs.as_span(),
                                                        guide_tether_memory_->allocator());
    contacts.constraint_set.emplace(contacts.affected_geo_indices,
                                    contacts.point_pairs.as_span(),
                                    contacts.rest_offsets.as_span(),
                                    contacts.compliances.as_span(),
                                    contacts.point_geo.as_span(),
                                    contacts.point_local.as_span(),
                                    contacts.point_inv_mass.as_span(),
                                    error_scale_from_threshold(0.01f),
                                    contacts.lambdas.as_mutable_span());
    guide_tether_active_ = true;
  }

  /**
   * Rebuilds the set of segment pairs that already touch in the rest pose.
   *
   * This is static for the whole frame, so it is computed once. Without it, every candidate pair
   * would run the exact rest segment/segment distance again in every frame.
   */
  void build_rest_close_pairs(HairSelfContacts &contacts)
  {
    contacts.rest_pairs = {};
    contacts.rest_pair_count = 0;
    const int segments_num = contacts.segment_geometry.size();
    if (segments_num < 2) {
      return;
    }
    const bool debug = std::getenv("XPBD_HAIR_DEBUG") != nullptr;
    const auto start_time = std::chrono::steady_clock::now();

    const uint64_t cache_key = hair_rest_pair_key_hash(contacts);
    HairRestPairCache &cache = hair_rest_pair_cache();
    {
      std::lock_guard<std::mutex> lock(cache.mutex);
      if (cache.key == cache_key) {
        contacts.rest_pairs = cache.pairs;
        contacts.rest_pair_count = cache.pair_count;
        if (debug) {
          printf("XPBD_HAIR rest_pairs %d cache=hit\n", contacts.rest_pair_count);
        }
        return;
      }
    }

    const Span<float3> rest_positions = contacts.rest_positions;
    float max_radius = 0.0f;
    for (const float radius : contacts.segment_radius) {
      max_radius = std::max(max_radius, radius);
    }
    float max_thickness = 0.0f;
    for (const float thickness : contacts.segment_thickness) {
      max_thickness = std::max(max_thickness, thickness);
    }
    constexpr float probe_factor = 1.25f;
    const float contact_max = 2.0f * max_radius + max_thickness;
    const float cell_size = std::max(contact_max, 1e-4f) * probe_factor;
    if (!(cell_size > 0.0f)) {
      return;
    }

    xpbd::HairSegmentGrid grid;
    grid.build(rest_positions,
               contacts.segment_point_a.as_span(),
               contacts.segment_point_b.as_span(),
               cell_size);

    const int chunk_size = 1024;
    const int chunks_num = (segments_num + chunk_size - 1) / chunk_size;
    Array<Vector<uint64_t>> chunk_pairs(chunks_num);
    threading::parallel_for(IndexRange(chunks_num), 1, [&](const IndexRange chunk_range) {
      for (const int chunk_i : chunk_range) {
        const int begin = chunk_i * chunk_size;
        const int end = std::min(begin + chunk_size, segments_num);
        Vector<uint64_t> &pairs = chunk_pairs[chunk_i];
        for (const int segment_i : IndexRange(begin, end - begin)) {
          grid.foreach_candidate(segment_i,
                                 rest_positions,
                                 contacts.segment_point_a.as_span(),
                                 contacts.segment_point_b.as_span(),
                                 contact_max * probe_factor,
                                 [&](const int segment_j) {
                                   if (segment_j <= segment_i) {
                                     return;
                                   }
                                   /* Adjacent segments of one strand share an endpoint; the pair
                                    * builder never creates a contact for them, so they must not
                                    * inflate the rest set (that would also make the groom look
                                    * denser than it is). */
                                   if (contacts.segment_point_a[segment_i] ==
                                           contacts.segment_point_a[segment_j] ||
                                       contacts.segment_point_a[segment_i] ==
                                           contacts.segment_point_b[segment_j] ||
                                       contacts.segment_point_b[segment_i] ==
                                           contacts.segment_point_a[segment_j] ||
                                       contacts.segment_point_b[segment_i] ==
                                           contacts.segment_point_b[segment_j])
                                   {
                                     return;
                                   }
                                   const float contact_distance =
                                       contacts.segment_radius[segment_i] +
                                       contacts.segment_radius[segment_j] +
                                       std::max(contacts.segment_thickness[segment_i],
                                                contacts.segment_thickness[segment_j]);
                                   const float probe = contact_distance * probe_factor;
                                   float s;
                                   float t;
                                   if (hair_segment_distance_squared(contacts,
                                                                     rest_positions,
                                                                     segment_i,
                                                                     segment_j,
                                                                     s,
                                                                     t) < probe * probe)
                                   {
                                     pairs.append(hair_pair_key(segment_i, segment_j));
                                   }
                                 });
        }
      }
    });

    Vector<uint64_t> sorted_pairs;
    int total = 0;
    for (const Vector<uint64_t> &pairs : chunk_pairs) {
      total += pairs.size();
    }
    sorted_pairs.reserve(total);
    for (const Vector<uint64_t> &pairs : chunk_pairs) {
      sorted_pairs.extend(pairs);
    }
    parallel_sort(sorted_pairs.as_mutable_span());
    auto last = std::unique(sorted_pairs.begin(), sorted_pairs.end());
    sorted_pairs.resize(std::distance(sorted_pairs.begin(), last));
    contacts.rest_pair_count = sorted_pairs.size();
    contacts.rest_pairs.build(sorted_pairs.as_span());
    {
      std::lock_guard<std::mutex> lock(cache.mutex);
      cache.key = cache_key;
      cache.pairs = contacts.rest_pairs;
      cache.pair_count = contacts.rest_pair_count;
    }
    if (debug) {
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start_time)
                            .count();
      printf("XPBD_HAIR rest_pairs %d build_ms %.1f cache=miss\n",
             contacts.rest_pair_count,
             ms);
    }
  }

  /**
   * Builds the hair segment pairs that can collide during this frame.
   *
   * The broad phase is a uniform AABB grid over the segments, with a cell size close to the contact
   * distance. Every segment is expanded by its own displacement margin (how far its endpoints can
   * move until the end of the frame), so a pair that is not touching yet but can touch in a later
   * substep is still collected. That makes the pair list valid for all substeps instead of
   * re-collecting it in each of them.
   *
   * The pair filter uses the rest-pose hash set for the pairs that already touch, and finally the
   * exact segment/segment distance. Dense regions are capped: a segment keeps at most
   * #MAX_NEIGHBORS_PER_SEGMENT contacts and the whole list at most #max_pairs pairs. The caps are
   * applied after the deterministic sort, so the kept set does not depend on thread scheduling.
   */
  void build_hair_contact_pairs(HairSelfContacts &contacts)
  {
    constexpr int MAX_NEIGHBORS_PER_SEGMENT = 64;

    contacts.pair_offsets.reinitialize(0);
    contacts.pair_points.clear();
    contacts.pair_segments.clear();
    contacts.pair_min_distances.clear();
    contacts.pair_compliances.clear();
    contacts.pair_lambdas.reinitialize(0);
    contacts.coloring = {};
    contacts.constraint_set.reset();

    const int segments_num = contacts.segment_geometry.size();
    if (segments_num < 2) {
      return;
    }
    const bool debug = std::getenv("XPBD_HAIR_DEBUG") != nullptr;
    const auto start_time = std::chrono::steady_clock::now();
    auto collect_end = start_time;
    const Span<float3> positions = contacts.positions;
    float max_radius = 0.0f;
    for (const float radius : contacts.segment_radius) {
      max_radius = std::max(max_radius, radius);
    }
    float max_thickness = 0.0f;
    for (const float thickness : contacts.segment_thickness) {
      max_thickness = std::max(max_thickness, thickness);
    }
    constexpr float probe_factor = 1.25f;
    const float contact_max = 2.0f * max_radius + max_thickness;
    const float cell_size = std::max(contact_max, 1e-4f) * probe_factor;
    if (!(cell_size > 0.0f)) {
      return;
    }

    const Span<int> segment_geometry = contacts.segment_geometry;
    const Span<int> segment_strand = contacts.segment_strand;
    const Span<int> segment_point_a = contacts.segment_point_a;
    const Span<int> segment_point_b = contacts.segment_point_b;
    const Span<float> segment_radius = contacts.segment_radius;
    const Span<float> segment_thickness = contacts.segment_thickness;
    const Span<uint8_t> segment_same_strand = contacts.segment_same_strand;
    const Span<float> segment_margin = contacts.segment_margin;
    const HairPairSet &rest_pairs = contacts.rest_pairs;

    const auto is_candidate = [&](const int segment_i, const int segment_j) -> bool {
      if (segment_j <= segment_i) {
        return false;
      }
      const bool same_geometry = segment_geometry[segment_i] == segment_geometry[segment_j];
      const bool same_curve = same_geometry && segment_strand[segment_i] == segment_strand[segment_j];
      if (same_curve && segment_same_strand[segment_i] == 0) {
        return false;
      }
      /* Adjacent segments of one strand share an endpoint and must not push each other apart. */
      if (segment_point_a[segment_i] == segment_point_a[segment_j] ||
          segment_point_a[segment_i] == segment_point_b[segment_j] ||
          segment_point_b[segment_i] == segment_point_a[segment_j] ||
          segment_point_b[segment_i] == segment_point_b[segment_j])
      {
        return false;
      }
      const float contact_distance = segment_radius[segment_i] + segment_radius[segment_j] +
                                     std::max(segment_thickness[segment_i],
                                              segment_thickness[segment_j]);
      /* Pairs that already touch in the rest pose (the roots of neighbouring strands) would push
       * each other apart forever. */
      if (rest_pairs.contains(hair_pair_key(segment_i, segment_j))) {
        return false;
      }
      /* Both segments can move by their margin before the end of the frame. */
      const float probe = contact_distance * probe_factor + segment_margin[segment_i] +
                          segment_margin[segment_j];
      float s;
      float t;
      return hair_segment_distance_squared(contacts, positions, segment_i, segment_j, s, t) <
             probe * probe;
    };

    xpbd::HairSegmentGrid grid;
    grid.build(positions,
               contacts.segment_point_a.as_span(),
               contacts.segment_point_b.as_span(),
               cell_size);

    int64_t max_pairs = std::max<int64_t>(1 << 18, int64_t(segments_num) * 16);
    max_pairs = std::min<int64_t>(max_pairs, 1 << 22);
    /* A pair can be emitted from several cells of the query AABB, so the collection buffer is larger
     * than the final pair list; it is still bounded to keep pathological input from exhausting the
     * memory before the duplicate removal. */
    const int collect_cap = int(std::min<int64_t>(max_pairs * 8, int64_t(1) << 30));
    const int chunk_size = 1024;
    const int chunks_num = (segments_num + chunk_size - 1) / chunk_size;
    const int chunk_cap = std::max(256, collect_cap / std::max(1, chunks_num));
    Array<Vector<int2>> chunk_pairs(chunks_num);
    std::atomic<int> capped_chunks{0};
    threading::parallel_for(IndexRange(chunks_num), 1, [&](const IndexRange chunk_range) {
      for (const int chunk_i : chunk_range) {
        const int begin = chunk_i * chunk_size;
        const int end = std::min(begin + chunk_size, segments_num);
        Vector<int2> &pairs = chunk_pairs[chunk_i];
        for (const int segment_i : IndexRange(begin, end - begin)) {
          if (pairs.size() >= size_t(chunk_cap)) {
            capped_chunks.fetch_add(1, std::memory_order_relaxed);
            break;
          }
          const float query_expand = contact_max * probe_factor + segment_margin[segment_i];
          grid.foreach_candidate(segment_i,
                                 positions,
                                 contacts.segment_point_a.as_span(),
                                 contacts.segment_point_b.as_span(),
                                 query_expand,
                                 [&](const int segment_j) {
                                   if (pairs.size() >= size_t(chunk_cap)) {
                                     return;
                                   }
                                   if (is_candidate(segment_i, segment_j)) {
                                     pairs.append(int2(segment_i, segment_j));
                                   }
                                 });
        }
      }
    });

    int total = 0;
    for (const Vector<int2> &pairs : chunk_pairs) {
      total += pairs.size();
    }
    if (total == 0) {
      return;
    }
    contacts.pair_segments.reserve(total);
    for (const Vector<int2> &pairs : chunk_pairs) {
      contacts.pair_segments.extend(pairs);
    }
    collect_end = std::chrono::steady_clock::now();
    parallel_sort(contacts.pair_segments.as_mutable_span(),
                  [](const int2 &a, const int2 &b) {
                    return a.x < b.x || (a.x == b.x && a.y < b.y);
                  });
    auto last = std::unique(contacts.pair_segments.begin(),
                            contacts.pair_segments.end(),
                            [](const int2 &a, const int2 &b) {
                              return a.x == b.x && a.y == b.y;
                            });
    int pairs_num = int(std::distance(contacts.pair_segments.begin(), last));
    const auto sort_end = std::chrono::steady_clock::now();

    /* Cap the pair list and the degree of every segment. The list is sorted, so the kept set is
     * deterministic; the caps are a safety valve for pathological input (a whole hairstyle collapsed
     * into one point would otherwise produce a quadratic number of pairs). */
    const bool list_capped = pairs_num > max_pairs;
    if (list_capped) {
      pairs_num = int(max_pairs);
    }
    Array<int> neighbor_counts(segments_num, 0);
    Vector<int2> limited_pairs;
    limited_pairs.reserve(pairs_num);
    int neighbor_dropped = 0;
    for (const int pair_i : IndexRange(pairs_num)) {
      const int2 &pair = contacts.pair_segments[pair_i];
      if (neighbor_counts[pair.x] >= MAX_NEIGHBORS_PER_SEGMENT ||
          neighbor_counts[pair.y] >= MAX_NEIGHBORS_PER_SEGMENT)
      {
        neighbor_dropped++;
        continue;
      }
      neighbor_counts[pair.x]++;
      neighbor_counts[pair.y]++;
      limited_pairs.append(pair);
    }
    if (list_capped || neighbor_dropped > 0 || capped_chunks.load() > 0) {
      contacts.pair_segments = std::move(limited_pairs);
      pairs_num = contacts.pair_segments.size();
      this->report(NodeWarningType::Warning,
                   "Hair self collision reached its contact safety limit; some contacts were "
                   "skipped. Reduce the strand count or Hair Thickness, or increase the solver "
                   "iterations.");
    }
    else {
      contacts.pair_segments.resize(pairs_num);
    }

    contacts.pair_offsets.reinitialize(pairs_num + 1);
    contacts.pair_points.reinitialize(pairs_num * 4);
    contacts.pair_min_distances.reinitialize(pairs_num);
    contacts.pair_compliances.reinitialize(pairs_num);
    contacts.pair_compliances.fill(0.0f);
    for (const int pair_i : IndexRange(pairs_num)) {
      const int2 &pair = contacts.pair_segments[pair_i];
      contacts.pair_offsets[pair_i] = pair_i * 4;
      const int base = pair_i * 4;
      contacts.pair_points[base + 0] = segment_point_a[pair.x];
      contacts.pair_points[base + 1] = segment_point_b[pair.x];
      contacts.pair_points[base + 2] = segment_point_a[pair.y];
      contacts.pair_points[base + 3] = segment_point_b[pair.y];
      contacts.pair_min_distances[pair_i] = segment_radius[pair.x] + segment_radius[pair.y] +
                                            std::max(segment_thickness[pair.x],
                                                     segment_thickness[pair.y]);
    }
    contacts.pair_offsets[pairs_num] = pairs_num * 4;

    contacts.pair_lambdas.reinitialize(pairs_num);
    contacts.pair_lambdas.fill(0.0f);

    hair_contacts_memory_ = std::make_unique<ResourceScope>(size_t(1) << 16);
    contacts.coloring = color_hair_segment_pairs(contacts.pair_segments.as_span(),
                                                 segments_num,
                                                 hair_contacts_memory_->allocator());
    const auto color_end = std::chrono::steady_clock::now();

    const GroupedSpan<int> pair_points(OffsetIndices<int>(contacts.pair_offsets),
                                       contacts.pair_points.as_span());
    contacts.constraint_set.emplace(contacts.affected_geo_indices,
                                    pair_points,
                                    contacts.pair_min_distances.as_span(),
                                    contacts.pair_compliances.as_span(),
                                    contacts.pair_segments.as_span(),
                                    contacts.point_geo.as_span(),
                                    contacts.point_local.as_span(),
                                    contacts.point_inv_mass.as_span(),
                                    error_scale_from_threshold(0.01f),
                                    contacts.pair_lambdas.as_mutable_span());

    if (debug) {
      const auto ms = [](const auto &a, const auto &b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      printf("XPBD_HAIR points %d segments %d rest_pairs %d pairs %d capped %d neighbor_dropped %d "
             "collect_ms %.1f sort_ms %.1f color_ms %.1f total_ms %.1f margin_max %.4f\n",
             int(contacts.point_geo.size()),
             segments_num,
             contacts.rest_pair_count,
             pairs_num,
             int(capped_chunks.load()),
             neighbor_dropped,
             ms(start_time, collect_end),
             ms(collect_end, sort_end),
             ms(sort_end, color_end),
             ms(start_time, color_end),
             contacts.segment_margin.is_empty() ?
                 0.0f :
                 *std::max_element(contacts.segment_margin.begin(), contacts.segment_margin.end()));
    }
  }

  /**
   * Mirrors the current solver positions into the hair contact space.
   *
   * The pairs are built once per frame (with a displacement margin), so this no longer rebuilds the
   * broad phase. The mirrored positions are only used for debugging; the constraint solver reads the
   * live solver positions directly.
   */
  void simulate__update_hair_contacts(const int solver_refs_i)
  {
    HairSelfContacts &contacts = hair_contacts_;
    if (contacts.is_empty()) {
      return;
    }
    const Span<xpbd::GeometryRef> refs = geometries_.solver_refs[solver_refs_i];
    for (const int i : contacts.geometry_data_key.index_range()) {
      const IndexRange slice(contacts.geometry_start[i], contacts.geometry_size[i]);
      const Span<float3> positions = refs[contacts.geometry_data_key[i]].positions;
      if (positions.size() != slice.size()) {
        continue;
      }
      contacts.positions.as_mutable_span().slice(slice).copy_from(positions);
    }
  }

  /**
   * Writes the current solver positions into the collision surfaces and rebuilds their BVH trees.
   * Runs once per substep, before the contacts of the substep are gathered.
   */
  void simulate__update_coupled_surfaces(const int solver_refs_i)
  {
    if (coupled_surfaces_.is_empty()) {
      return;
    }
    const Span<xpbd::GeometryRef> refs = geometries_.solver_refs[solver_refs_i];
    for (CoupledSurface &surface : coupled_surfaces_) {
      const Span<float3> positions = refs[surface.geometry_i].positions;
      if (positions.size() != surface.mesh->verts_num) {
        continue;
      }
      surface.prev_positions.as_mutable_span().copy_from(surface.mesh->vert_positions());
      MutableSpan<float3> surface_positions = surface.mesh->vert_positions_for_write();
      surface_positions.copy_from(positions);
      surface.mesh->tag_positions_changed();
      surface.bvh = surface.mesh->bvh_corner_tris();
    }
  }

  /**
   * Refreshes the solid BVH trees and gathers the segment/triangle contacts of this substep.
   *
   * The contacts are rebuilt every substep because both sides move. They are collected in per
   * chunk vectors, sorted and deduplicated so the contact set (and the coloring) is deterministic,
   * and capped per segment and in total as a safety valve for pathological input.
   */
  void simulate__update_hair_solid_contacts(const int solver_refs_i)
  {
    HairSolidContacts &contacts = hair_solid_contacts_;
    if (!hair_solid_active_) {
      return;
    }
    contacts.contact_offsets.reinitialize(0);
    contacts.contact_points.clear();
    contacts.contact_normals.clear();
    contacts.contact_segment_weights.clear();
    contacts.contact_face_weights.clear();
    contacts.contact_min_distances.clear();
    contacts.contact_compliances.clear();
    contacts.contact_lambdas.reinitialize(0);
    contacts.coloring = {};
    contacts.constraint_set.reset();

    const Span<xpbd::GeometryRef> refs = geometries_.solver_refs[solver_refs_i];
    for (CoupledSurface &surface : hair_solid_surfaces_) {
      const Span<float3> positions = refs[surface.geometry_i].positions;
      if (positions.size() != surface.mesh->verts_num) {
        continue;
      }
      surface.prev_positions.as_mutable_span().copy_from(surface.mesh->vert_positions());
      MutableSpan<float3> surface_positions = surface.mesh->vert_positions_for_write();
      surface_positions.copy_from(positions);
      surface.mesh->tag_positions_changed();
      surface.bvh = surface.mesh->bvh_corner_tris();
    }

    const auto position = [&](const int global_point_i) -> const float3 & {
      return refs[contacts.point_geo[global_point_i]].positions[contacts.point_local[global_point_i]];
    };

    const int segments_num = contacts.segment_point_a.size();
    const int chunk_size = 1024;
    const int chunks_num = std::max(1, (segments_num + chunk_size - 1) / chunk_size);
    Array<Vector<HairSolidContact>> chunk_contacts(chunks_num);
    constexpr int MAX_CONTACTS_PER_SEGMENT = 32;
    threading::parallel_for(IndexRange(chunks_num), 1, [&](const IndexRange chunk_range) {
      for (const int chunk_i : chunk_range) {
        const int begin = chunk_i * chunk_size;
        const int end = std::min(begin + chunk_size, segments_num);
        Vector<HairSolidContact> &out = chunk_contacts[chunk_i];
        for (const int segment_i : IndexRange(begin, end - begin)) {
          const float3 p0 = position(contacts.segment_point_a[segment_i]);
          const float3 p1 = position(contacts.segment_point_b[segment_i]);
          const float3 mid = 0.5f * (p0 + p1);
          const float half_length = 0.5f * math::distance(p0, p1);
          const float radius = contacts.segment_radius[segment_i];
          const float thickness = std::max(contacts.segment_thickness[segment_i], 0.0f);
          const int output_start = out.size();
          for (const int surface_i : hair_solid_surfaces_.index_range()) {
            const CoupledSurface &surface = hair_solid_surfaces_[surface_i];
            if (surface.bvh.tree == nullptr) {
              continue;
            }
            HairSolidRangeQuery query;
            query.surface = &surface;
            query.surface_point_start = hair_solid_surface_point_start_[surface_i];
            query.segment = segment_i;
            query.p0 = p0;
            query.p1 = p1;
            query.segment_radius = radius;
            query.segment_thickness = thickness;
            query.solid_radius = hair_solid_surface_max_radius[surface_i];
            query.out = &out;
            query.output_start = output_start;
            query.max_contacts_per_segment = MAX_CONTACTS_PER_SEGMENT;
            const float search_radius = half_length + radius + query.solid_radius + thickness;
            BLI_bvhtree_range_query(surface.bvh.tree,
                                    reinterpret_cast<const float *>(&mid),
                                    search_radius,
                                    hair_solid_range_query_callback,
                                    &query);
          }
        }
      }
    });

    int total = 0;
    for (const Vector<HairSolidContact> &chunk : chunk_contacts) {
      total += chunk.size();
    }
    if (total == 0) {
      return;
    }
    Vector<HairSolidContact> all_contacts;
    all_contacts.reserve(total);
    for (const Vector<HairSolidContact> &chunk : chunk_contacts) {
      all_contacts.extend(chunk);
    }
    const auto key_less = [](const HairSolidContact &a, const HairSolidContact &b) {
      if (a.segment != b.segment) {
        return a.segment < b.segment;
      }
      if (a.triangle.x != b.triangle.x) {
        return a.triangle.x < b.triangle.x;
      }
      if (a.triangle.y != b.triangle.y) {
        return a.triangle.y < b.triangle.y;
      }
      return a.triangle.z < b.triangle.z;
    };
    parallel_sort(all_contacts.as_mutable_span(), key_less);
    const auto key_equal = [](const HairSolidContact &a, const HairSolidContact &b) {
      return a.segment == b.segment && a.triangle.x == b.triangle.x &&
             a.triangle.y == b.triangle.y && a.triangle.z == b.triangle.z;
    };
    auto last = std::unique(all_contacts.begin(), all_contacts.end(), key_equal);

    const int64_t max_contacts = std::min<int64_t>(1 << 20, std::max<int64_t>(1 << 14,
                                                                              int64_t(segments_num) * 8));
    Vector<HairSolidContact> limited;
    limited.reserve(std::min<int64_t>(std::distance(all_contacts.begin(), last), max_contacts));
    int last_segment = -1;
    int segment_contacts = 0;
    for (auto it = all_contacts.begin(); it != last; ++it) {
      if (it->segment != last_segment) {
        last_segment = it->segment;
        segment_contacts = 0;
      }
      if (segment_contacts >= MAX_CONTACTS_PER_SEGMENT) {
        continue;
      }
      segment_contacts++;
      limited.append(*it);
      if (limited.size() >= max_contacts) {
        break;
      }
    }

    const int contacts_num = limited.size();
    if (contacts_num == 0) {
      return;
    }
    contacts.contact_offsets.reinitialize(contacts_num + 1);
    contacts.contact_points.reinitialize(contacts_num * 5);
    contacts.contact_normals.reinitialize(contacts_num);
    contacts.contact_segment_weights.reinitialize(contacts_num);
    contacts.contact_face_weights.reinitialize(contacts_num);
    contacts.contact_min_distances.reinitialize(contacts_num);
    contacts.contact_compliances.reinitialize(contacts_num);
    contacts.contact_compliances.fill(0.0f);
    for (const int contact_i : IndexRange(contacts_num)) {
      const HairSolidContact &contact = limited[contact_i];
      const int base = contact_i * 5;
      contacts.contact_offsets[contact_i] = base;
      contacts.contact_points[base + 0] = contacts.segment_point_a[contact.segment];
      contacts.contact_points[base + 1] = contacts.segment_point_b[contact.segment];
      contacts.contact_points[base + 2] = contact.triangle.x;
      contacts.contact_points[base + 3] = contact.triangle.y;
      contacts.contact_points[base + 4] = contact.triangle.z;
      contacts.contact_normals[contact_i] = contact.normal;
      contacts.contact_segment_weights[contact_i] = float2(contact.segment_s, 0.0f);
      contacts.contact_face_weights[contact_i] = contact.face_weights;
      contacts.contact_min_distances[contact_i] = contact.min_distance;
    }
    contacts.contact_offsets[contacts_num] = contacts_num * 5;

    contacts.contact_lambdas.reinitialize(contacts_num);
    contacts.contact_lambdas.fill(0.0f);

    hair_solid_memory_ = std::make_unique<ResourceScope>(size_t(1) << 16);
    const GroupedSpan<int> contact_points(OffsetIndices<int>(contacts.contact_offsets),
                                          contacts.contact_points.as_span());
    contacts.coloring = xpbd::color_constraints__n_ary(contact_points,
                                                       hair_solid_memory_->allocator());
    contacts.constraint_set.emplace(contacts.affected_geo_indices,
                                    contact_points,
                                    contacts.contact_normals.as_span(),
                                    contacts.contact_segment_weights.as_span(),
                                    contacts.contact_face_weights.as_span(),
                                    contacts.contact_min_distances.as_span(),
                                    contacts.contact_compliances.as_span(),
                                    contacts.point_geo.as_span(),
                                    contacts.point_local.as_span(),
                                    contacts.point_inv_mass.as_span(),
                                    error_scale_from_threshold(0.01f),
                                    contacts.contact_lambdas.as_mutable_span());
  }

  /**
   * Gathers the contacts between the fluid points of a chunk and the triangles of the coupled
   * surfaces. This mirrors the contacts with a collider mesh, but the surface moves with the
   * simulation: the velocity of the closest point comes from the positions of the previous
   * substep, and the triangle (and where on it the contact is) is stored so the impulse can be
   * given back.
   */
  void gather_contacts__coupled_surfaces(const int chunk_i,
                                         const float max_distance,
                                         const int solver_refs_i,
                                         const SubstepInterval &substep,
                                         CoupledFaceContacts &r_contacts)
  {
    r_contacts.clear();
    if (coupled_surfaces_.is_empty()) {
      return;
    }
    if (std::getenv("BLENDER_NO_COUPLED_CONTACTS") != nullptr) {
      return;
    }
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    if (geo_data.fluid_constraint_i < 0 || !chunks_data_[chunk_i].fluid.has_coupled) {
      return;
    }
    /* The friction of the point is only loaded for geometries that collide with a collider. */
    this->ensure_friction_loaded(chunk.data_key_i);
    const Span<float3> positions =
        geometries_.solver_refs[solver_refs_i][chunk.data_key_i].positions;
    const Span<float3> prev_positions =
        geometries_.solver_refs[solver_refs_i][chunk.data_key_i].prev_positions;
    const Span<float> coupling_compliances = geo_data.fluid_coupling_compliance.as_span();

    for (const CoupledSurface &surface : coupled_surfaces_) {
      const Mesh &mesh = *surface.mesh;
      const Span<float3> vert_positions = mesh.vert_positions();
      const Span<int3> corner_tris = mesh.corner_tris();
      const Span<int> corner_verts = mesh.corner_verts();
      if (corner_tris.is_empty()) {
        continue;
      }
      const GeometryData &other = *geometries_.data[surface.geometry_i];
      const VArray<float> other_static_frictions =
          other.attributes
              .lookup_or_default<float>("static_friction", bke::AttrDomain::Point, 0.5f)
              .varray;
      const VArray<float> other_dynamic_frictions =
          other.attributes
              .lookup_or_default<float>("dynamic_friction", bke::AttrDomain::Point, 0.5f)
              .varray;

      for (const int point_i : chunk.points_range) {
        if (geo_data.is_hard_pinned[point_i]) {
          continue;
        }
        const float3 &pos = positions[point_i];
        const float radius = math::interpolate(
            geo_data.prev_radii[point_i], geo_data.radii[point_i], substep.interpolate_end);
        /* The contact only has to catch what the samples cannot hold: a particle that got deep
         * into the surface (a fast drop, a lone particle). With the full particle radius a resting
         * fluid pushes against both the samples and the contact, which makes the surface jump
         * around and counts the weight of the water twice. */
        const float contact_radius = 0.5f * radius;
        const std::optional<ClosestMeshFaceContact> contact = this->get_closest_mesh_face_contact(
            pos,
            surface.bvh,
            corner_tris,
            corner_verts,
            vert_positions,
            max_distance + contact_radius);
        if (!contact) {
          continue;
        }
        /* A sheet has no volume, so a particle that is *beside* the surface (the closest point is
         * on the border of the triangle) is not inside it: the normal distance there is nearly
         * zero only because the offset is along the sheet, not across it. The contact constraint
         * would push such a particle onto the plane of the sheet and hold it there, which makes a
         * "plate" of particles hovering next to every fold and border. Those particles are handled
         * by the density samples instead. */
        /* A sheet has no volume, so the normal distance to a triangle *border* is almost zero for
         * a particle that is beside the sheet rather than above or below it: the offset is along
         * the sheet, not across it. The contact constraint would push such a particle onto the
         * plane of the sheet and hold it there, which makes a "plate" of particles hovering next
         * to every border and fold (the contact search reaches 1.25 m, so the plate was not even
         * limited to the support radius). Only skip those: a particle that is above or below the
         * border (its offset is mostly along the normal) is a real contact. */
        if (this->is_bary_coord_close_to_edge(contact->bary_coords)) {
          const float3 normal = contact->face_nor;
          const float3 offset = pos - contact->nearest_pos;
          const float distance = math::length(offset);
          const float normal_length = math::length(normal);
          if (distance > 1e-6f && normal_length > 1e-6f &&
              std::abs(math::dot(offset, normal)) < 0.5f * distance * normal_length)
          {
            continue;
          }
        }
        const int3 &tri = corner_tris[contact->tri_i];
        const int3 verts(corner_verts[tri[0]], corner_verts[tri[1]], corner_verts[tri[2]]);

        /* Where the closest point was in the previous substep, to know how the surface moves. */
        const float3 prev_pos_on_face = bke::attribute_math::mix3(
            contact->bary_coords,
            surface.prev_positions[verts[0]],
            surface.prev_positions[verts[1]],
            surface.prev_positions[verts[2]]);
        const float3 motion = contact->nearest_pos - prev_pos_on_face;

        /* Direction to move the point out of the surface.
         *
         * This has to be the *surface normal*, not the direction from the closest point to the
         * particle: at the border of an open sheet the closest point is on the edge, so that
         * direction is almost parallel to the sheet and a particle touching the rim would be shot
         * sideways (measured: 65 m/s off the rim of a cloth).
         *
         * A sheet has no inside, so the side is decided by where the particle came from: if it
         * crossed the surface within this substep it is pushed back to the side it started on,
         * which is what stops a fast particle that jumped through the sheet in one step. Otherwise
         * it is pushed further away from the side it is on (a particle that is behind the sheet
         * for good is not teleported through it again). */
        float3 normal = contact->face_nor;
        const float normal_length = math::length(normal);
        float3 axis = float3(0.0f);
        if (normal_length > 1e-6f) {
          normal /= normal_length;
          const float side_now = math::dot(pos - contact->nearest_pos, normal);
          const float side_prev = math::dot(prev_positions[point_i] - contact->nearest_pos,
                                            normal);
          const bool crossed = (side_now >= 0.0f) != (side_prev >= 0.0f);
          const bool positive_side = crossed ? (side_prev >= 0.0f) : (side_now >= 0.0f);
          axis = positive_side ? normal : -normal;
        }
        else {
          /* Degenerate normal: fall back to the geometric direction. */
          const float3 direction = pos - contact->nearest_pos;
          if (!math::is_zero(direction, 1e-6f)) {
            axis = math::normalize(direction);
          }
        }
        if (!(std::isfinite(axis.x) && std::isfinite(axis.y) && std::isfinite(axis.z)) ||
            math::is_zero(axis))
        {
          continue;
        }
        const float other_static_friction = other_static_frictions[verts[0]];
        const float other_dynamic_friction = other_dynamic_frictions[verts[0]];
        const float static_friction = this->compute_contact_friction(
            geo_data.static_frictions[point_i], other_static_friction);
        const float dynamic_friction = this->compute_contact_friction(
            geo_data.dynamic_frictions[point_i], other_dynamic_friction);

        r_contacts.contacts.points.append(point_i);
        r_contacts.contacts.point_radii.append(contact_radius);
        r_contacts.contacts.positions_on_face.append(contact->nearest_pos);
        r_contacts.contacts.collider_motion.append(motion);
        /* The device contact table is a fixed layout; the velocity of the surface is not used by the
         * position or friction kernel, but it has to exist or the upload reads past the end. */
        r_contacts.contacts.collider_velocities.append(float3(0.0f));
        r_contacts.contacts.face_normals.append(axis);
        r_contacts.contacts.face_margins.append(0.0f);
        r_contacts.contacts.static_frictions.append(static_friction);
        r_contacts.contacts.dynamic_frictions.append(dynamic_friction);
        /* Same XPBD scaling as the collider contacts: the raw compliance is divided by the
         * substep time squared, so the value the user sets does not depend on the frame rate or
         * the substep count. */
        const float coupling_compliance = point_i < coupling_compliances.size() ?
                                              coupling_compliances[point_i] :
                                              0.0f;
        r_contacts.contacts.compliance_terms.append(
            std::max(0.0f, substep_compliance_factor_ * coupling_compliance));
        r_contacts.contacts.error_scales.append(error_scale_from_threshold(1e-2f));
        r_contacts.contacts.active_states.append(false);
        r_contacts.contacts.lambdas_normal.append(0.0f);
        r_contacts.contacts.lambdas_friction.append(0.0f);

        r_contacts.face_vertices.append(verts);
        r_contacts.face_barycentric.append(contact->bary_coords);
        r_contacts.face_geometry.append(surface.geometry_i);
      }
    }
  }

  void simulate__fluid_position_solve__chunk(const int chunk_i,
                                             const xpbd::ConstraintSetParams &solve_params,
                                             xpbd::GaussSeidelUpdater &updater)
  {
    FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    if (!fluid.active || fluid.gpu_active || fluid.grid.is_empty()) {
      return;
    }
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const int geo_i = chunk.data_key_i;
    const GeometryData &geo_data = *geometries_.data[geo_i];
    const FluidConstraint &constraint = constraints_.fluid_constraints[fluid.constraint_i];

    xpbd::FluidDensitySettings settings;
    settings.compliance_term = constraint.compliance * solve_params.compliance_term_factor;
    settings.artificial_pressure = constraint.artificial_pressure;
    /* Maximum positional correction of one particle per iteration. */
    settings.max_correction = 0.1f * fluid.support_radius;

    const IndexRange points = chunk.points_range;
    xpbd::FluidBoundary boundary;
    if (fluid.boundary_active) {
      boundary.grid = &fluid.boundary_grid;
      boundary.neighbors = &fluid.boundary_neighbors;
      boundary.positions = fluid.boundary_positions;
      boundary.masses = fluid.boundary_masses;
    }
    xpbd::FluidBoundary coupled;
    if (fluid.has_coupled) {
      coupled.grid = &fluid.coupled_grid;
      coupled.neighbors = &fluid.coupled_neighbors;
      coupled.positions = fluid.coupled_positions;
      coupled.masses = fluid.coupled_masses;
      coupled.inv_masses = fluid.coupled_inv_masses;
    }
    const xpbd::FluidDensityStats stats = xpbd::fluid_solve_density_iteration(
        fluid.neighbors,
        xpbd::FluidKernel(fluid.support_radius),
        solve_params.geometry_refs()[geo_i].positions.slice(points),
        geo_data.fluid_masses.as_span().slice(points),
        geo_data.inv_masses.as_span().slice(points),
        geo_data.fluid_rest_density.as_span().slice(points),
        geo_data.fluid_negative_pressure_limit.as_span().slice(points),
        geo_data.fluid_use_boundary.as_span().slice(points),
        settings,
        fluid.lambdas,
        fluid.errors,
        fluid.density_errors,
        fluid.neighbor_counts,
        fluid.corrections,
        fluid.solid_fractions,
        boundary,
        coupled,
        fluid.has_coupled ? fluid.coupled_push.as_mutable_span().slice(points) :
                            MutableSpan<float3>());
    updater.add_residual_error(geo_i,
                               stats.average_error_squared *
                                   error_scale_from_threshold(constraint.error_threshold));
    fluid.last_density_error_mean = stats.average_density_error;

    if (std::getenv("BLENDER_FLUID_DEBUG") != nullptr) {
      printf(
          "FLUID_DEBUG density: avg_err=%.4f max_err=%.4f max_correction=%.5f "
          "max_neighbors=%d h=%.4f | density error mean %+.3f min %+.3f, under dense %d/%d\n",
          double(stats.average_error_squared),
          double(stats.max_abs_error),
          double(stats.max_correction),
          stats.max_neighbors,
          double(fluid.support_radius),
          double(stats.average_density_error),
          double(stats.min_density_error),
          stats.under_dense_num,
          int(fluid.lambdas.size()));
    }

    this->simulate__fluid_boundary_reaction__chunk(chunk_i, solve_params);
  }

  /**
   * Computes the reaction of the fluid pressure on the boundary particles that belong to other
   * simulated geometries. The result is stored in the chunk, #simulate__fluid_boundary_reaction
   * applies it. Splitting the two steps keeps the write to the positions of another chunk out of
   * the parallel loop.
   */
  void simulate__fluid_boundary_reaction__chunk(const int chunk_i,
                                                const xpbd::ConstraintSetParams &solve_params)
  {
    FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    if (!fluid.active || !fluid.has_coupled || fluid.grid.is_empty()) {
      return;
    }
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const int geo_i = chunk.data_key_i;
    const GeometryData &geo_data = *geometries_.data[geo_i];
    const IndexRange points = chunk.points_range;

    xpbd::FluidBoundary coupled;
    coupled.grid = &fluid.coupled_grid;
    coupled.positions = fluid.coupled_positions;
    coupled.masses = fluid.coupled_masses;
    coupled.inv_masses = fluid.coupled_inv_masses;

    xpbd::fluid_apply_boundary_reaction(
        fluid.grid,
        xpbd::FluidKernel(fluid.support_radius),
        solve_params.geometry_refs()[geo_i].positions.slice(points),
        geo_data.fluid_masses.as_span().slice(points),
        geo_data.fluid_rest_density.as_span().slice(points),
        fluid.lambdas,
        geo_data.fluid_coupling.as_span().slice(points),
        coupled,
        0.1f * fluid.support_radius,
        fluid.coupled_corrections);
  }

  /**
   * Applies the reactions that the fluids computed during the last position solve iteration to the
   * geometries they are coupled with. This runs after the parallel solve, so two fluids that touch
   * the same solid cannot lose each other's correction.
   *
   * The correction of a sample is spread over the vertices of its triangle with the barycentric
   * weights and *averaged* there (divided by the accumulated weight), so moving a vertex by the
   * correction of one sample does not depend on the resolution of the solid. Pinned vertices are
   * skipped, so a sample whose corners are all pinned does not move anything.
   */
  void simulate__fluid_boundary_reaction(const int solver_refs_i)
  {
    const Span<xpbd::GeometryRef> refs = geometries_.solver_refs[solver_refs_i];
    for (const int chunk_i : geometries_.chunks.index_range()) {
      FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
      if (!fluid.active || !fluid.has_coupled) {
        continue;
      }
      /* A device fluid applies both reactions itself: the contact multipliers after every substep
       * in #simulate__gpu_fluid_substep and the density reaction once per frame at the last substep.
       * The CPU path below runs once per constraint iteration, so without this guard the same
       * multipliers would be applied a second (and third, ...) time and the cloth would be launched
       * out of the water (measured on `HWA_test`: a 0.6 m/frame cloth jump right when the water
       * reaches it). */
      if (fluid.gpu_active) {
        continue;
      }
      for (FluidChunkData::CoupledVertexScratch &scratch : fluid.coupled_scratch) {
        scratch.correction_sums.fill(float3(0.0f));
        scratch.weight_sums.fill(0.0f);
      }
      for (const int i : fluid.coupled_geometry.index_range()) {
        const float3 &correction = fluid.coupled_corrections[i];
        if (correction.x == 0.0f && correction.y == 0.0f && correction.z == 0.0f) {
          continue;
        }
        const int geometry_i = fluid.coupled_geometry[i];
        FluidChunkData::CoupledVertexScratch *scratch = nullptr;
        for (FluidChunkData::CoupledVertexScratch &candidate : fluid.coupled_scratch) {
          if (candidate.geometry_i == geometry_i) {
            scratch = &candidate;
            break;
          }
        }
        if (scratch == nullptr) {
          continue;
        }
        const int3 &verts = fluid.coupled_vertices[i];
        const float3 &weights = fluid.coupled_weights[i];
        const Span<float> inv_masses = geometries_.data[geometry_i]->inv_masses;
        for (const int k : IndexRange(3)) {
          const int vertex_i = verts[k];
          const float weight = weights[k];
          if (weight <= 0.0f || inv_masses[vertex_i] <= 0.0f) {
            continue;
          }
          scratch->correction_sums[vertex_i] += correction * weight;
          scratch->weight_sums[vertex_i] += weight;
        }
      }
      for (FluidChunkData::CoupledVertexScratch &scratch : fluid.coupled_scratch) {
        const MutableSpan<float3> positions = refs[scratch.geometry_i].positions;
        for (const int vertex_i : scratch.weight_sums.index_range()) {
          const float weight_sum = scratch.weight_sums[vertex_i];
          if (weight_sum <= 0.0f) {
            continue;
          }
          const float3 offset = scratch.correction_sums[vertex_i] / weight_sum;
          positions[vertex_i] += offset;
          debug_reaction_density_.fetch_add(double(math::length(offset)),
                                            std::memory_order_relaxed);
        }
      }

      this->simulate__coupled_contact_reaction__chunk(chunk_i, refs, /*clamp_sum=*/false);
    }
  }

  /**
   * Refreshes the coupled sample positions of a device fluid from the solid vertices and uploads
   * them. The CPU path does the same in #simulate__update_fluid_boundary__chunk; the device fluid
   * skips that function (it rebuilds its own grid), so it needs its own refresh once per substep.
   */
  void simulate__update_gpu_coupled(FluidChunkData &fluid, const int solver_refs_i)
  {
    if (!fluid.has_coupled || !fluid.gpu_active) {
      return;
    }
    const Span<xpbd::GeometryRef> refs = geometries_.solver_refs[solver_refs_i];
    for (const int i : fluid.coupled_geometry.index_range()) {
      const Span<float3> positions = refs[fluid.coupled_geometry[i]].positions;
      const int3 &verts = fluid.coupled_vertices[i];
      const float3 &weights = fluid.coupled_weights[i];
      fluid.coupled_positions[i] = positions[verts[0]] * weights.x +
                                   positions[verts[1]] * weights.y +
                                   positions[verts[2]] * weights.z;
    }
    const bool updated = fluid.gpu_fluid.update_coupled(fluid.coupled_positions,
                                                       fluid.coupled_masses,
                                                       fluid.coupled_inv_masses);
    if (std::getenv("XPBD_GPU_DEBUG") != nullptr) {
      printf("XPBD_GPU update_coupled host=%d updated=%d error=%s\n",
             int(fluid.coupled_positions.size()),
             updated ? 1 : 0,
             fluid.gpu_fluid.last_error());
      fflush(stdout);
    }
  }

  /**
   * Applies the reaction of a device fluid to the solid vertices. The spread is the same as
   * #simulate__fluid_boundary_reaction: barycentric weights, averaged per vertex so the result does
   * not depend on the resolution of the solid.
   */
  void simulate__gpu_coupled_reaction(FluidChunkData &fluid,
                                      const Span<float3> corrections,
                                      const int solver_refs_i)
  {
    if (std::getenv("XPBD_GPU_DEBUG") != nullptr) {
      float max_len = 0.0f;
      for (const float3 &c : corrections) {
        max_len = std::max(max_len, math::length(c));
      }
      printf("XPBD_GPU coupled reaction count=%d max=%g\n",
             int(corrections.size()),
             double(max_len));
      fflush(stdout);
    }
    const Span<xpbd::GeometryRef> refs = geometries_.solver_refs[solver_refs_i];
    for (FluidChunkData::CoupledVertexScratch &scratch : fluid.coupled_scratch) {
      scratch.correction_sums.fill(float3(0.0f));
      scratch.weight_sums.fill(0.0f);
    }
    for (const int i : fluid.coupled_geometry.index_range()) {
      const float3 &correction = corrections[i];
      if (correction.x == 0.0f && correction.y == 0.0f && correction.z == 0.0f) {
        continue;
      }
      const int geometry_i = fluid.coupled_geometry[i];
      FluidChunkData::CoupledVertexScratch *scratch = nullptr;
      for (FluidChunkData::CoupledVertexScratch &candidate : fluid.coupled_scratch) {
        if (candidate.geometry_i == geometry_i) {
          scratch = &candidate;
          break;
        }
      }
      if (scratch == nullptr) {
        continue;
      }
      const int3 &verts = fluid.coupled_vertices[i];
      const float3 &weights = fluid.coupled_weights[i];
      const Span<float> inv_masses = geometries_.data[geometry_i]->inv_masses;
      for (const int k : IndexRange(3)) {
        const int vertex_i = verts[k];
        const float weight = weights[k];
        if (weight <= 0.0f || inv_masses[vertex_i] <= 0.0f) {
          continue;
        }
        scratch->correction_sums[vertex_i] += correction * weight;
        scratch->weight_sums[vertex_i] += weight;
      }
    }
    for (FluidChunkData::CoupledVertexScratch &scratch : fluid.coupled_scratch) {
      const MutableSpan<float3> positions = refs[scratch.geometry_i].positions;
      for (const int vertex_i : scratch.weight_sums.index_range()) {
        const float weight_sum = scratch.weight_sums[vertex_i];
        if (weight_sum <= 0.0f) {
          continue;
        }
        const float3 offset = scratch.correction_sums[vertex_i] / weight_sum;
        positions[vertex_i] += offset;
        debug_reaction_density_.fetch_add(double(math::length(offset)),
                                          std::memory_order_relaxed);
      }
    }
  }

  /**
   * Applies the hard contacts of the coupled distance field to the solid vertices. Each contact is
   * a fluid particle index and the position correction that the device applied to it; the closest
   * triangle is looked up on the host and the conjugate correction (split by inverse mass, scaled
   * by the coupling) is spread over its vertices with the barycentric weights.
   */
  void simulate__gpu_coupled_sdf_reaction(FluidChunkData &fluid,
                                          const int chunk_i,
                                          const Span<int> indices,
                                          const Span<float3> pushes,
                                          const int solver_refs_i)
  {
    if (indices.is_empty() || coupled_surfaces_.is_empty()) {
      return;
    }
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    const IndexRange points = chunk.points_range;
    const Span<xpbd::GeometryRef> refs = geometries_.solver_refs[solver_refs_i];
    const Span<float3> fluid_positions = refs[chunk.data_key_i].positions.slice(points);
    const Span<float> fluid_inv_masses = geo_data.inv_masses.as_span().slice(points);
    const Span<float> coupling = geo_data.fluid_coupling.as_span().slice(points);
    CoupledSurface &surface = coupled_surfaces_[0];
    if (surface.mesh == nullptr) {
      return;
    }
    const Span<int3> corner_tris = surface.mesh->corner_tris();
    const Span<int> corner_verts = surface.mesh->corner_verts();
    const Span<float3> surface_positions = surface.mesh->vert_positions();
    const Span<float> surface_inv_masses = geometries_.data[surface.geometry_i]->inv_masses;

    for (FluidChunkData::CoupledVertexScratch &scratch : fluid.coupled_scratch) {
      scratch.correction_sums.fill(float3(0.0f));
      scratch.weight_sums.fill(0.0f);
    }
    FluidChunkData::CoupledVertexScratch *scratch = nullptr;
    for (FluidChunkData::CoupledVertexScratch &candidate : fluid.coupled_scratch) {
      if (candidate.geometry_i == surface.geometry_i) {
        scratch = &candidate;
        break;
      }
    }
    if (scratch == nullptr) {
      return;
    }

    for (const int k : indices.index_range()) {
      const int i = indices[k];
      const float3 &push = pushes[k];
      if (push.x == 0.0f && push.y == 0.0f && push.z == 0.0f) {
        continue;
      }
      const float c = coupling[i];
      if (!(c > 0.0f)) {
        continue;
      }
      BVHTreeNearest nearest;
      nearest.index = -1;
      BLI_bvhtree_find_nearest(surface.bvh.tree,
                               fluid_positions[i],
                               &nearest,
                               surface.bvh.nearest_callback,
                               &surface.bvh);
      if (nearest.index < 0 || nearest.index >= corner_tris.size()) {
        continue;
      }
      const int3 &tri = corner_tris[nearest.index];
      const int3 verts(corner_verts[tri.x], corner_verts[tri.y], corner_verts[tri.z]);
      const float3 nearest_co(nearest.co[0], nearest.co[1], nearest.co[2]);
      const float3 weights = triangle_weights(surface_positions[verts[0]],
                                              surface_positions[verts[1]],
                                              surface_positions[verts[2]],
                                              nearest_co);
      const float inv_m_i = fluid_inv_masses[i];
      for (const int v : IndexRange(3)) {
        const int vertex_i = verts[v];
        const float weight = weights[v];
        if (weight <= 0.0f || surface_inv_masses[vertex_i] <= 0.0f) {
          continue;
        }
        const float inv_m_v = surface_inv_masses[vertex_i];
        /* Conjugate of the fluid push, split by inverse mass and scaled by the coupling. */
        const float share = inv_m_v / (inv_m_i + inv_m_v);
        scratch->correction_sums[vertex_i] += push * (-c * share * weight);
        scratch->weight_sums[vertex_i] += weight;
      }
    }

    const MutableSpan<float3> positions = refs[surface.geometry_i].positions;
    /* The whole frame's contacts are applied at once (one readback per frame), so a dense contact
     * patch can sum to far more than one substep's impulse and blow the cloth up. Clamp the total
     * per vertex to one support radius, which is the order of a single substep's correction. */
    const float max_reaction = fluid.support_radius;
    for (const int vertex_i : scratch->weight_sums.index_range()) {
      if (scratch->weight_sums[vertex_i] <= 0.0f) {
        continue;
      }
      /* A contact is an independent impulse: a vertex touched by more water is pushed more, so the
       * contributions are summed (the density reaction averages instead, because it is a field). */
      float3 correction = scratch->correction_sums[vertex_i];
      const float length = math::length(correction);
      if (max_reaction > 0.0f && length > max_reaction) {
        correction *= max_reaction / length;
      }
      positions[vertex_i] += correction;
    }
  }

  /**
   * Gives the contact impulses of a chunk back to the triangles of the coupled surfaces. The
   * normal multiplier of a contact is the impulse that stopped the fluid point (in position units,
   * so the momentum of the surface is `-lambda * normal`), which is spread over the three vertices
   * with the barycentric weights, weighted by their inverse mass. Unlike the density reaction
   * there is no division by the accumulated weight: every contact is an independent impulse, so a
   * vertex that is touched by more water is pushed more, which is what a real surface feels.
   *
   * Only the part of the multiplier that was not applied yet is used, because it accumulates over
   * the constraint iterations of a substep.
   */
  void simulate__coupled_contact_reaction__chunk(const int chunk_i,
                                                 const Span<xpbd::GeometryRef> refs,
                                                 const bool clamp_sum)
  {
    CoupledFaceContacts &contacts = chunks_data_[chunk_i].coupled_face_contacts;
    if (contacts.contacts.points.is_empty()) {
      return;
    }
    const int contacts_num = int(contacts.contacts.points.size());
    if (std::getenv("XPBD_REACTION_DEBUG") != nullptr) {
      float max_lambda = 0.0f;
      for (const float l : contacts.contacts.lambdas_normal) {
        max_lambda = std::max(max_lambda, std::abs(l));
      }
      printf("XPBD_REACTION contact_call n=%d max_lambda=%.6g\n",
             contacts_num,
             double(max_lambda));
      fflush(stdout);
    }
    if (contacts.applied_lambdas.size() != contacts_num) {
      contacts.applied_lambdas.reinitialize(contacts_num);
      contacts.applied_lambdas.fill(0.0f);
    }
    FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    const float max_correction = 0.1f * fluid.support_radius;
    if (!clamp_sum) {
      /* CPU path: every contact is an independent impulse, applied directly. This is the reference
       * behaviour and it holds the water (measured: the CPU leaks if the sum is clamped). */
      for (const int contact_i : IndexRange(contacts_num)) {
        const float lambda = contacts.contacts.lambdas_normal[contact_i];
        const float delta = lambda - contacts.applied_lambdas[contact_i];
        if (delta == 0.0f) {
          continue;
        }
        contacts.applied_lambdas[contact_i] = lambda;
        const int geometry_i = contacts.face_geometry[contact_i];
        if (geometry_i < 0) {
          continue;
        }
        const int point_i = contacts.contacts.points[contact_i];
        const GeometryData &fluid_data = *geometries_.data[geometries_.chunks[chunk_i].data_key_i];
        const float coupling = point_i < fluid_data.fluid_coupling.size() ?
                                   fluid_data.fluid_coupling[point_i] :
                                   1.0f;
        if (!(coupling > 0.0f)) {
          continue;
        }
        const Span<float> inv_masses = geometries_.data[geometry_i]->inv_masses;
        const MutableSpan<float3> positions = refs[geometry_i].positions;
        const int3 &verts = contacts.face_vertices[contact_i];
        const float3 &weights = contacts.face_barycentric[contact_i];
        const float3 &normal = contacts.contacts.face_normals[contact_i];
        for (const int k : IndexRange(3)) {
          const int vertex_i = verts[k];
          const float weight = weights[k];
          if (weight <= 0.0f || inv_masses[vertex_i] <= 0.0f) {
            continue;
          }
          float3 offset = -delta * coupling * inv_masses[vertex_i] * weight * normal;
          const float length = math::length(offset);
          if (length > max_correction) {
            offset *= max_correction / length;
          }
          if (UNLIKELY(
                  !(std::isfinite(offset.x) && std::isfinite(offset.y) && std::isfinite(offset.z))))
          {
            continue;
          }
          positions[vertex_i] += offset;
          debug_reaction_contact_.fetch_add(double(math::length(offset)),
                                            std::memory_order_relaxed);
        }
      }
      return;
    }

    /* Device path: accumulate first, clamp the *summed* offset once per vertex, then apply. A dense
     * contact patch has dozens of contacts on the same vertex; clamping only each contact let the
     * sum grow with the number of contacts, which pushed the cloth through the water. */
    for (FluidChunkData::CoupledVertexScratch &scratch : fluid.coupled_scratch) {
      scratch.correction_sums.fill(float3(0.0f));
    }
    for (const int contact_i : IndexRange(contacts_num)) {
      const float lambda = contacts.contacts.lambdas_normal[contact_i];
      const float delta = lambda - contacts.applied_lambdas[contact_i];
      if (delta == 0.0f) {
        continue;
      }
      contacts.applied_lambdas[contact_i] = lambda;
      const int geometry_i = contacts.face_geometry[contact_i];
      if (geometry_i < 0) {
        continue;
      }
      FluidChunkData::CoupledVertexScratch *scratch = nullptr;
      for (FluidChunkData::CoupledVertexScratch &candidate : fluid.coupled_scratch) {
        if (candidate.geometry_i == geometry_i) {
          scratch = &candidate;
          break;
        }
      }
      if (scratch == nullptr) {
        continue;
      }
      const int point_i = contacts.contacts.points[contact_i];
      const GeometryData &fluid_data = *geometries_.data[geometries_.chunks[chunk_i].data_key_i];
      const float coupling = point_i < fluid_data.fluid_coupling.size() ?
                                 fluid_data.fluid_coupling[point_i] :
                                 1.0f;
      if (!(coupling > 0.0f)) {
        continue;
      }
      const Span<float> inv_masses = geometries_.data[geometry_i]->inv_masses;
      const int3 &verts = contacts.face_vertices[contact_i];
      const float3 &weights = contacts.face_barycentric[contact_i];
      const float3 &normal = contacts.contacts.face_normals[contact_i];
      for (const int k : IndexRange(3)) {
        const int vertex_i = verts[k];
        const float weight = weights[k];
        if (weight <= 0.0f || inv_masses[vertex_i] <= 0.0f) {
          continue;
        }
        scratch->correction_sums[vertex_i] +=
            -delta * coupling * inv_masses[vertex_i] * weight * normal;
      }
    }
    for (FluidChunkData::CoupledVertexScratch &scratch : fluid.coupled_scratch) {
      const MutableSpan<float3> positions = refs[scratch.geometry_i].positions;
      for (const int vertex_i : scratch.correction_sums.index_range()) {
        float3 offset = scratch.correction_sums[vertex_i];
        if (offset.x == 0.0f && offset.y == 0.0f && offset.z == 0.0f) {
          continue;
        }
        const float length = math::length(offset);
        if (max_correction > 0.0f && length > max_correction) {
          offset *= max_correction / length;
        }
        if (UNLIKELY(
                !(std::isfinite(offset.x) && std::isfinite(offset.y) && std::isfinite(offset.z))))
        {
          continue;
        }
        positions[vertex_i] += offset;
        debug_reaction_contact_.fetch_add(double(math::length(offset)),
                                          std::memory_order_relaxed);
      }
    }
  }

  void simulate__fluid_velocity_solve__chunk(const int chunk_i, const int solver_refs_i)
  {
    FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    if (!fluid.active || fluid.gpu_active || fluid.grid.is_empty()) {
      return;
    }
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const int geo_i = chunk.data_key_i;
    const GeometryData &geo_data = *geometries_.data[geo_i];
    const FluidConstraint &constraint = constraints_.fluid_constraints[fluid.constraint_i];
    const IndexRange points = chunk.points_range;
    const xpbd::FluidKernel kernel(fluid.support_radius);
    const Span<float3> positions = geometries_.solver_refs[solver_refs_i][geo_i].positions.slice(
        points);
    const Span<float> masses = geo_data.fluid_masses.as_span().slice(points);
    MutableSpan<float3> velocities = geo_data.velocity_attr.span.slice(points);
    const Span<float> rest_density = geo_data.fluid_rest_density.as_span().slice(points);
    const Span<float> viscosity = geo_data.fluid_viscosity.as_span().slice(points);
    const Span<float> damping = geo_data.fluid_damping.as_span().slice(points);
    const Span<float> vorticity = geo_data.fluid_vorticity.as_span().slice(points);
    const Span<float> surface_tension = geo_data.fluid_surface_tension.as_span().slice(points);
    const Span<float> boundary_friction = geo_data.fluid_boundary_friction.as_span().slice(points);

    /* Added velocity from vorticity confinement and surface tension, on top of the velocities that
     * the position solve produced. */
    xpbd::fluid_apply_vorticity_confinement(fluid.neighbors,
                                            kernel,
                                            positions,
                                            masses,
                                            rest_density,
                                            vorticity,
                                            sub_delta_time_,
                                            velocities,
                                            fluid.omegas,
                                            fluid.omega_magnitudes,
                                            fluid.corrections);
    xpbd::fluid_apply_surface_tension(fluid.neighbors,
                                      positions,
                                      masses,
                                      fluid.support_radius,
                                      surface_tension,
                                      sub_delta_time_,
                                      velocities,
                                      fluid.corrections);
    xpbd::fluid_apply_xsph_viscosity(fluid.neighbors,
                                     kernel,
                                     positions,
                                     masses,
                                     rest_density,
                                     viscosity,
                                     velocities,
                                     fluid.corrections);
    if (fluid.boundary_active) {
      xpbd::fluid_apply_boundary_friction(
          velocities, fluid.solid_fractions.as_span().slice(points), boundary_friction);
    }
    xpbd::fluid_apply_velocity_damping(damping, sub_delta_time_, velocities);
  }

  /**
   * Decides whether the fluid of a chunk can be advanced on the compute backend and uploads its
   * state. Requires:
   * - an `HW Accelerate` zone that allows the GPU (or `BLENDER_XPBD_GPU=1`),
   * - no two way coupling with another simulated geometry (the reaction of the pressure on the
   *   vertices of the solids is not implemented on the device yet),
   * - no constraints on the fluid geometry itself (the density constraint is the only one the
   *   device knows), and one chunk holding the whole fluid.
   */
  /**
   * Why a chunk cannot use the device, `nullptr` when it can. `r_soft` tells whether the limitation
   * is only "not implemented on the device yet" (coupling), which an explicit
   * `BLENDER_XPBD_GPU=1` may override for A/B testing, or a hard structural one.
   */
  /** Why the fluid could not use the device in the last evaluation, for the node UI. */
  std::string gpu_rejection_reason_;

  /** Particle count below which the fluid stays on the CPU (see #fluid_gpu_rejection). */
  int device_min_particles() const
  {
    static const int threshold = [] {
      const char *value = std::getenv("BLENDER_XPBD_GPU_MIN_PARTICLES");
      if (value != nullptr) {
        return int(std::atoi(value));
      }
      return 16384;
    }();
    return threshold;
  }

  const char *fluid_gpu_rejection(const int chunk_i, bool &r_soft) const
  {
    r_soft = false;
    const FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    if (!fluid.active) {
      return "not a fluid";
    }
    /* Coupled fluids run on the device with the *compliant* point/triangle contacts of the CPU:
     * the host gathers them once per substep from the predicted positions and the device solves the
     * same constraint set together with the density iterations. The per point compliance comes
     * from `Coupling Compliance` (0 is the old hard contact), so the device and the CPU solve the
     * exact same constraint. The old hard distance-field projection could not hold a dense falling
     * column (M118). */
    /* Small fluids are faster on the (multi threaded) CPU: the device pays ~0.1 ms per dispatch and
     * ~100 passes per frame, which is all fixed cost below a few thousand particles. Measured on
     * earlier hardware: 8.8k particles were 206 ms on the device and 208 ms on the CPU, i.e. the
     * zone made nothing faster and the fixed cost was visible. Re-measure on the target GPU. */
    if (int(geometries_.chunks[chunk_i].points_range.size()) < this->device_min_particles()) {
      r_soft = true;
      return "below the device particle threshold";
    }
    /* The multi chunk path (`!support_chunk_simulation()`, i.e. self contacts or constraint sets
     * that span the whole world) runs the device fluid too, see
     * #simulate__gpu_fluid_substep_unchunked. */
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    if (geo_data.uses_rotation || geo_data.curves != nullptr) {
      return "curves or rotations";
    }
    if (!geo_data.static_constraints.is_empty()) {
      return "static constraints";
    }
    const ChunkData &chunk_data = chunks_data_[chunk_i];
    if (!chunk_data.static_constraints.is_empty() ||
        !chunk_data.static_velocity_constraints.is_empty() ||
        !chunk_data.pin_position_constraints.is_empty() ||
        !chunk_data.pin_rotation_constraints.is_empty())
    {
      return "per chunk constraints";
    }
    if (chunk.points_range != IndexRange(geo_data.size)) {
      return "split into several chunks";
    }
    return nullptr;
  }

  bool fluid_can_use_gpu(const int chunk_i) const
  {
    bool soft = false;
    return this->fluid_gpu_rejection(chunk_i, soft) == nullptr;
  }

  /**
   * Bakes the collider distance field of a fluid (see #FluidChunkData::sdf). Only one static mesh
   * collider is supported for now, which covers the common cases (a floor, a basin, a plate);
   * deformable or animated colliders keep using the host side BVH contacts.
   */
  void prepare_fluid_sdf(FluidChunkData &fluid, const GeometryData &geo_data)
  {
    ensure_sdf_cache_hook();
    fluid.sdf_ready = false;
    fluid.sdf.reset();
    if (geo_data.mesh_colliders.size() != 1) {
      return;
    }
    const MeshCollider &collider = constraints_.mesh_colliders[geo_data.mesh_colliders[0]
                                                                  .constraint_i];
    if (collider.is_boundary) {
      /* Boundary particles already keep the fluid out of the collider, no contact needed. */
      return;
    }
    const StaticMeshInfo *static_info = std::get_if<StaticMeshInfo>(&collider.mesh);
    if (static_info == nullptr || static_info->mesh == nullptr) {
      return;
    }
    const Mesh &mesh = *static_info->mesh;
    if (mesh.corner_tris().is_empty()) {
      return;
    }

    /* Half a particle radius per voxel: fine enough for the contact normal, cheap to bake. */
    const float voxel_size = std::max(1e-4f, 0.5f * fluid.support_radius);
    /* The band has to be wider than one substep of movement plus the radius, otherwise a fast
     * particle can cross the whole band of a thin collider. */
    /* Wide enough that a particle moving a full substep cannot cross the band in one step: the
     * device solves the contact by sampling the field, and once a particle is deeper than the band
     * the sample returns the background value and the contact is lost (observed with a fluid
     * falling onto a floor: min z went to -7 m). The band costs voxels only in the slab around the
     * collider, so the extra safety is cheap. */
    const float band_width = std::max(4.0f * voxel_size, 4.0f * fluid.support_radius);

    /* Content hash, not the mesh pointer: the geometry nodes evaluation hands out a new mesh every
     * frame, so a pointer based key rebuilt the field on every single frame (measured: 5.2 s per
     * frame instead of 200 ms). A handful of vertex positions is enough to notice a moving or
     * deformed collider. The resolution is part of the key too: changing the particle radius
     * changes the voxel size and the field has to be rebaked. */
    uint64_t hash = uint64_t(mesh.verts_num) * 2654435761u;
    hash ^= uint64_t(mesh.corner_tris().size()) * 40503u;
    hash = hash * 33u + uint64_t(BLI_hash_int(int(voxel_size * 1e6f)));
    hash = hash * 33u + uint64_t(BLI_hash_int(int(band_width * 1e6f)));
    for (const int column : IndexRange(4)) {
      for (const int row : IndexRange(4)) {
        hash = hash * 33u + uint64_t(BLI_hash_int(int(collider.end_transform[column][row] * 4096.0f)));
      }
    }
    {
      const Span<float3> positions = mesh.vert_positions();
      const int step = std::max(1, int(positions.size()) / 16);
      for (int i = 0; i < int(positions.size()); i += step) {
        hash = hash * 33u + uint64_t(BLI_hash_int(int(positions[i].x * 4096.0f)));
        hash = hash * 33u + uint64_t(BLI_hash_int(int(positions[i].y * 4096.0f)));
        hash = hash * 33u + uint64_t(BLI_hash_int(int(positions[i].z * 4096.0f)));
      }
    }

    /* Build the field only where the fluid can touch the collider: a 20 m ground plane otherwise
     * costs 4.4 M voxels and ~0.5 s (GUI) / 2.6 s (headless -t 1) to bake. The region is the
     * intersection of the fluid bounding box (plus a margin for its movement) and the collider
     * bounding box (plus the band). For a floor this is a thin slab around z = 0, not the whole
     * 20 m plane. The cache entry keeps the region, so the fluid can move inside it without a
     * rebake. */
    const Span<float3> fluid_positions = geo_data.position_attr.span;
    if (fluid_positions.is_empty()) {
      return;
    }
    float3 fluid_min = fluid_positions[0];
    float3 fluid_max = fluid_positions[0];
    for (const float3 &position : fluid_positions) {
      fluid_min = math::min(fluid_min, position);
      fluid_max = math::max(fluid_max, position);
    }

    const Span<float3> local_positions = mesh.vert_positions();
    Array<float3> world_positions(local_positions.size());
    threading::parallel_for(local_positions.index_range(), 4096, [&](const IndexRange range) {
      for (const int i : range) {
        world_positions[i] = math::transform_point(collider.end_transform, local_positions[i]);
      }
    });
    float3 collider_min = world_positions[0];
    float3 collider_max = world_positions[0];
    for (const float3 &position : world_positions) {
      collider_min = math::min(collider_min, position);
      collider_max = math::max(collider_max, position);
    }

    /* A small margin in X/Y (the fluid barely moves sideways) and a generous one in Z (a falling
     * fluid may still be a couple of metres above the floor). The Z extent of the field itself is
     * clipped to the collider, so the large Z margin costs nothing. */
    const float margin_xy = std::max(0.5f, 8.0f * band_width);
    const float margin_z = std::max(2.0f, 8.0f * band_width);
    const float3 fluid_pad_min(fluid_min.x - margin_xy, fluid_min.y - margin_xy,
                               fluid_min.z - margin_z);
    const float3 fluid_pad_max(fluid_max.x + margin_xy, fluid_max.y + margin_xy,
                               fluid_max.z + margin_z);
    const float3 region_min = math::max(fluid_pad_min, collider_min - float3(band_width));
    const float3 region_max = math::min(fluid_pad_max, collider_max + float3(band_width));
    if (region_min.x > region_max.x || region_min.y > region_max.y ||
        region_min.z > region_max.z)
    {
      /* The fluid is nowhere near this collider; no contacts and no field needed. */
      fluid.sdf_hash = 0;
      return;
    }

    /* `chunks_data_` is reinitialized every frame, so the cache has to live outside the per-chunk
     * state. A single entry is enough for the common case of one static collider; a second
     * collider simply replaces it. */
    {
      std::lock_guard<std::mutex> lock(sdf_cache_mutex);
      const bool region_same = std::abs(region_min.x - sdf_cache.region_min.x) < 1e-4f &&
                               std::abs(region_min.y - sdf_cache.region_min.y) < 1e-4f &&
                               std::abs(region_min.z - sdf_cache.region_min.z) < 1e-4f &&
                               std::abs(region_max.x - sdf_cache.region_max.x) < 1e-4f &&
                               std::abs(region_max.y - sdf_cache.region_max.y) < 1e-4f &&
                               std::abs(region_max.z - sdf_cache.region_max.z) < 1e-4f;
      const bool fluid_inside = fluid_min.x >= sdf_cache.region_min.x &&
                                fluid_max.x <= sdf_cache.region_max.x &&
                                fluid_min.y >= sdf_cache.region_min.y &&
                                fluid_max.y <= sdf_cache.region_max.y;
      if (sdf_cache.grid && sdf_cache.hash == hash && sdf_cache.voxel_size == voxel_size &&
          sdf_cache.band_width == band_width && sdf_cache.grid->is_valid() &&
          (region_same || fluid_inside))
      {
        fluid.sdf = sdf_cache.grid;
        fluid.sdf_hash = hash;
        fluid.sdf_ready = true;
        return;
      }
    }
    auto grid = std::make_shared<geometry::ha::HaSdfGrid>();
    if (!geometry::ha::ha_sdf_build(world_positions,
                                    mesh.corner_verts(),
                                    mesh.corner_tris(),
                                    voxel_size,
                                    band_width,
                                    region_min,
                                    region_max,
                                    (fluid_min + fluid_max) * 0.5f,
                                    /*one_sided=*/false,
                                    *grid))
    {
      fluid.sdf_hash = 0;
      return;
    }
    {
      std::lock_guard<std::mutex> lock(sdf_cache_mutex);
      sdf_cache.hash = hash;
      sdf_cache.voxel_size = voxel_size;
      sdf_cache.band_width = band_width;
      sdf_cache.region_min = region_min;
      sdf_cache.region_max = region_max;
      sdf_cache.grid = grid;
    }
    fluid.sdf = grid;
    fluid.sdf_hash = hash;
    fluid.sdf_ready = grid->is_valid();
    debug_gpu_log("BLENDER_XPBD_GPU_LOG",
                  std::getenv("XPBD_GPU_DEBUG") != nullptr,
                  "XPBD_GPU sdf baked collider=%s voxels=%lld spacing=%.4f band=%.4f\n",
                  collider.path.c_str(),
                  (long long)grid->voxels_num(),
                  double(grid->spacing),
                  double(band_width));
  }

  /**
   * Builds a distance field of the coupled solids (cloth, soft bodies) once per frame. The device
   * samples it for the hard contact, which is what the density coupling alone cannot do for fast or
   * under-dense water. The field is rebuilt per frame (not per substep) because the solids move
   * slowly compared to the fluid substeps, and the build is the expensive part.
   */
  void prepare_coupled_sdf(FluidChunkData &fluid, const GeometryData &geo_data)
  {
    fluid.coupled_sdf_ready = false;
    fluid.coupled_sdf.reset();
    if (!fluid.has_coupled || coupled_surfaces_.is_empty()) {
      return;
    }
    const CoupledSurface &surface = coupled_surfaces_[0];
    if (surface.mesh == nullptr || surface.mesh->corner_tris().is_empty()) {
      return;
    }
    const Mesh &mesh = *surface.mesh;
    const Span<float3> positions = mesh.vert_positions();
    if (positions.is_empty()) {
      return;
    }
    float3 bbox_min = positions[0];
    float3 bbox_max = positions[0];
    for (const float3 &position : positions) {
      bbox_min = math::min(bbox_min, position);
      bbox_max = math::max(bbox_max, position);
    }
    /* The fluid is outside the solid, which fixes the sign of an open mesh (a cloth sheet). */
    const Span<float3> fluid_positions = geo_data.position_attr.span;
    if (fluid_positions.is_empty()) {
      return;
    }
    float3 fluid_min = fluid_positions[0];
    float3 fluid_max = fluid_positions[0];
    for (const float3 &position : fluid_positions) {
      fluid_min = math::min(fluid_min, position);
      fluid_max = math::max(fluid_max, position);
    }
    const float voxel = std::max(1e-4f, 0.5f * fluid.support_radius);
    const float band = std::max(4.0f * voxel, 2.0f * fluid.support_radius);
    const float3 region_min = bbox_min - float3(band);
    const float3 region_max = bbox_max + float3(band);
    auto grid = std::make_shared<geometry::ha::HaSdfGrid>();
    if (!geometry::ha::ha_sdf_build(positions,
                                    mesh.corner_verts(),
                                    mesh.corner_tris(),
                                    voxel,
                                    band,
                                    region_min,
                                    region_max,
                                    (fluid_min + fluid_max) * 0.5f,
                                    /*one_sided=*/true,
                                    *grid))
    {
      return;
    }
    fluid.coupled_sdf = grid;
    fluid.coupled_sdf_ready = grid->is_valid();
    fluid.coupled_sdf_band = band;
    debug_gpu_log("BLENDER_XPBD_GPU_LOG",
                  std::getenv("XPBD_GPU_DEBUG") != nullptr,
                  "XPBD_GPU coupled sdf baked voxels=%lld spacing=%.4f band=%.4f\n",
                  (long long)grid->voxels_num(),
                  double(grid->spacing),
                  double(band));
    if (std::getenv("XPBD_GPU_DEBUG_FIELD") != nullptr && grid->is_valid()) {
      const int cx = grid->dims.x / 2;
      const int cy = grid->dims.y / 2;
      printf("XPBD_GPU field z-slice:");
      for (int z = 0; z < grid->dims.z; z++) {
        const int64_t index = (int64_t(z) * grid->dims.y + cy) * grid->dims.x + cx;
        printf(" %.2f", double(grid->data[index]));
      }
      printf("\n");
    }
  }

  void prepare_fluid_gpu()
  {
    gpu_rejection_reason_.clear();
    const char *force = std::getenv("BLENDER_XPBD_GPU");
    const bool force_gpu = force != nullptr && StringRef(force) == "1";
    const bool force_cpu = force != nullptr && StringRef(force) == "0";
    /* An `HW Accelerate` zone is what asks for the device; the environment variable is only an
     * override for the tests. */
    const bool requested = force_cpu ? false :
                           force_gpu ? true :
                                       /* Inside an `HW Accelerate` zone: the request comes with the
                                        * evaluation data (see GeoNodesUserData). The thread local is a
                                        * fallback for calls that happen on the zone's own thread. */
                                       (gpu_backend_request_ == 1 || gpu_backend_request_ == 2 ||
                                        geometry::ha::ha_scope_allows_gpu());
    const bool print = std::getenv("XPBD_GPU_DEBUG") != nullptr;
    bool soft_rejection = false;
    for (const int chunk_i : chunks_data_.index_range()) {
      FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
      fluid.gpu_active = false;
      if (!fluid.active) {
        continue;
      }
      if (!requested) {
        debug_gpu_log("BLENDER_XPBD_GPU_LOG",
                      print,
                      "XPBD_GPU backend=cpu particles=%d reason=zone not requested\n",
                      int(geometries_.chunks[chunk_i].points_range.size()));
        continue;
      }
      if (const char *rejection = this->fluid_gpu_rejection(chunk_i, soft_rejection)) {
        if (gpu_rejection_reason_.empty()) {
          gpu_rejection_reason_ = rejection;
        }
        /* Only the "small fluid" performance rejection is soft: an explicit device request
         * (`Strict`, `BLENDER_XPBD_GPU=1`) may override it. Correctness rejections (coupling,
         * curves, constraints, split chunks) never get here with `soft_rejection` set. */
        const bool ignore_soft = force_gpu || gpu_backend_request_ == 2; /* Backend = Strict */
        if (!(ignore_soft && soft_rejection)) {
          debug_gpu_log("BLENDER_XPBD_GPU_LOG",
                        print,
                        "XPBD_GPU backend=cpu particles=%d reason=%s\n",
                        int(geometries_.chunks[chunk_i].points_range.size()),
                        rejection);
          if (gpu_backend_request_ == 2) {
            /* Strict: the zone promised the device, so say why that did not happen. */
            this->report(NodeWarningType::Warning,
                         fmt::format("HW Accelerate zone (Strict): the fluid is solved on the CPU "
                                     "({})",
                                     rejection));
          }
          else if (gpu_backend_request_ == 1) {
            /* Auto: a silent fallback is what made the whole feature hard to trust, so say it
             * once per evaluation on the node (info, not a warning: nothing went wrong). */
            this->report(NodeWarningType::Info,
                         fmt::format("HW Accelerate zone: the fluid is solved on the CPU ({})",
                                     rejection));
          }
          continue;
        }
        /* Explicit `BLENDER_XPBD_GPU=1`: run it anyway, for A/B comparisons of the parts that are
         * implemented on the device already. */
      }
      const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
      const GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
      const FluidConstraint &constraint = constraints_.fluid_constraints[fluid.constraint_i];
      const IndexRange points = chunk.points_range;

      geometry::ha::HaXpbdFluidInputs inputs;
      inputs.positions = geo_data.position_attr.span.slice(points);
      inputs.velocities = geo_data.velocity_attr.span.slice(points);
      const Span<float3> external_force = geo_data.external_force_attr.slice(points);
      const Span<float3> external_acceleration = geo_data.external_acceleration.as_span().slice(
          points);
      Vector<float3> external_force_with_wind;
      if (external_acceleration.size() == points.size()) {
        const Span<float> masses = geo_data.fluid_masses.as_span().slice(points);
        external_force_with_wind.reinitialize(points.size());
        for (const int i : points.index_range()) {
          external_force_with_wind[i] = external_force[i] + external_acceleration[i] * masses[i];
        }
        inputs.external_force = external_force_with_wind.as_span();
      }
      else {
        inputs.external_force = external_force;
      }
      inputs.masses = geo_data.fluid_masses.as_span().slice(points);
      inputs.inv_masses = geo_data.inv_masses.as_span().slice(points);
      inputs.rest_density = geo_data.fluid_rest_density.as_span().slice(points);
      inputs.negative_pressure_limit = geo_data.fluid_negative_pressure_limit.as_span().slice(
          points);
      inputs.viscosity = geo_data.fluid_viscosity.as_span().slice(points);
      inputs.damping = geo_data.fluid_damping.as_span().slice(points);
      inputs.vorticity = geo_data.fluid_vorticity.as_span().slice(points);
      inputs.surface_tension = geo_data.fluid_surface_tension.as_span().slice(points);
      inputs.boundary_friction = geo_data.fluid_boundary_friction.as_span().slice(points);
      inputs.use_boundary = geo_data.fluid_use_boundary.as_span().slice(points);
      inputs.coupling = geo_data.fluid_coupling.as_span().slice(points);
      inputs.boundary_positions = fluid.boundary_positions.as_span();
      inputs.boundary_masses = fluid.boundary_masses.as_span();
      inputs.coupled_positions = fluid.coupled_positions;
      inputs.coupled_masses = fluid.coupled_masses;
      inputs.coupled_inv_masses = fluid.coupled_inv_masses;

      fluid.gpu_start_ms = time_now_ms();
      fluid.gpu_active = fluid.gpu_fluid.bind(inputs,
                                             fluid.support_radius,
                                             sub_delta_time_,
                                             constraint.compliance * substep_compliance_factor_,
                                             constraint.artificial_pressure,
                                             0.1f * fluid.support_radius);
      /* Bake the collider field only when the device is actually going to use it. Doing it before
       * `bind` made a headless / context-less run pay for a 4.4 M voxel field that was then thrown
       * away (measured: 2.6 s per frame). */
      if (fluid.gpu_active) {
        this->prepare_fluid_sdf(fluid, geo_data);
      }
      if (fluid.gpu_active && fluid.sdf_ready) {
        /* Contact radius is the particle radius, the density support radius is twice that. */
        const bool bound = fluid.gpu_fluid.bind_sdf(
            *fluid.sdf,
            0.5f * fluid.support_radius,
            std::clamp(constraints_.mesh_colliders[geo_data.mesh_colliders[0].constraint_i]
                           .friction,
                       0.0f,
                       1.0f),
            /* The device pushes a particle out of the field by at most this much per iteration.
             * 0.1 * support radius (3 mm for a 3 cm support) was slower than a fast particle falls,
             * so the fluid sank through the band before the contact could catch up. One support
             * radius per iteration (30 mm) resolves a deeply penetrated particle in a couple of
             * iterations without a single large jump. */
            fluid.support_radius);
        debug_gpu_log("BLENDER_XPBD_GPU_LOG",
                      print,
                      "XPBD_GPU collision backend=%s voxels=%lld spacing=%g\n",
                      bound ? "sdf" : "cpu-gather",
                      (long long)fluid.sdf->voxels_num(),
                      double(fluid.sdf->spacing));
      }
      /* The coupled solids use the compliant point/triangle contacts (gathered on the host per
       * substep, solved on the device), not the distance field hard projection: the latter could
       * not hold a dense falling column (M118). The field code is kept for reference but off. */
      fluid.coupled_sdf_ready = false;
      fluid.coupled_sdf.reset();
      if (fluid.gpu_active) {
        fluid.gpu_fluid.clear_coupled_sdf();
      }
      debug_gpu_log("BLENDER_XPBD_GPU_LOG",
                    print,
                    "XPBD_GPU backend=%s particles=%d substeps=%d iterations=%d support_radius=%g%s%s\n",
                    fluid.gpu_active ? "device" : "cpu",
                    int(points.size()),
                    substeps_,
                    constraint_iterations_,
                    double(fluid.support_radius),
                    fluid.gpu_active ? "" : " reason=",
                    fluid.gpu_active ? "" : fluid.gpu_fluid.last_error());
    }
  }

  /**
   * Runs the whole fluid substep on the device and hands the result back to the solver buffers.
   * Only the last substep of the frame is downloaded: nothing between the substeps reads the CPU
   * copy of a device fluid.
   */
  void simulate__gpu_fluid_substep(const int chunk_i,
                                   const int solver_refs_i,
                                   const SubstepInterval &substep,
                                   const bool download_result)
  {
    FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    if (!fluid.gpu_active) {
      return;
    }
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    const IndexRange points = chunk.points_range;
    const MutableSpan<float3> positions =
        geometries_.solver_refs[solver_refs_i][chunk.data_key_i].positions.slice(points);

    /* Contacts are gathered on the host once per substep (that needs the predicted positions),
     * then solved on the device like on the CPU. */
    /* With a distance field the device solves the collider contacts itself, so the host does not
     * gather anything and there is no readback per substep. */
    const bool gathers_collider = !geo_data.mesh_colliders.is_empty() &&
                                  !fluid.gpu_fluid.has_sdf();
    /* The coupled solids (cloth, soft body) use the *compliant* point/triangle contacts of the CPU:
     * the gather runs on the host and the device solves the same constraint set. A distance field
     * hard projection could not hold a dense falling column (M118), a compliant contact can. */
    const bool gathers_coupled = fluid.has_coupled;
    /* `GeometryRef::prev_positions` is exposed as a read-only span, but the data it points at
     * (`temp_positions` or the position attribute) is the solver's own mutable buffer. The device
     * gather needs to write the device's substep-start positions into it. */
    const Span<float3> prev_span =
        geometries_.solver_refs[solver_refs_i][chunk.data_key_i].prev_positions;
    MutableSpan<float3> prev_positions(const_cast<float3 *>(prev_span.data()), prev_span.size());
    prev_positions = prev_positions.slice(points);
    geometry::ha::HaXpbdFluid::ContactHook hook;
    if (gathers_collider || gathers_coupled) {
      hook = [&]() -> bool {
        fluid.gpu_fluid.download_positions(positions);
        if (gathers_coupled) {
          fluid.gpu_fluid.download_prev_positions(prev_positions);
          this->gather_contacts__coupled_surfaces(
              chunk_i,
              this->get_max_search_distance(sub_delta_time_),
              solver_refs_i,
              substep,
              chunks_data_[chunk_i].coupled_face_contacts);
          const bool uploaded = this->upload_contacts_to_gpu(
              chunk_i, chunks_data_[chunk_i].coupled_face_contacts.contacts);
          if (std::getenv("XPBD_GPU_DEBUG") != nullptr) {
            printf("XPBD_GPU coupled contacts substep=%d count=%d uploaded=%d\n",
                   substep.current_i,
                   int(chunks_data_[chunk_i].coupled_face_contacts.contacts.points.size()),
                   uploaded ? 1 : 0);
            fflush(stdout);
          }
          return uploaded;
        }
        this->simulate__gather_dynamic_constraints__chunk(substep, chunk_i, solver_refs_i, true);
        const bool uploaded = this->upload_contacts_to_gpu(
            chunk_i, chunks_data_[chunk_i].external_face_contacts);
        if (std::getenv("XPBD_GPU_DEBUG") != nullptr) {
          printf("XPBD_GPU contacts substep=%d count=%d uploaded=%d\n",
                 substep.current_i,
                 int(chunks_data_[chunk_i].external_face_contacts.points.size()),
                 uploaded ? 1 : 0);
          fflush(stdout);
        }
        return uploaded;
      };
    }

    /* The coupled samples follow the solid vertices, so refresh them once per substep (the CPU
     * path does the same in `simulate__update_fluid_boundary__chunk`). */
    this->simulate__update_gpu_coupled(fluid, solver_refs_i);

    const bool ok = fluid.gpu_fluid.substep(constraint_iterations_, hook);
    /* Give the multipliers of the compliant coupled contacts back to the solid vertices, like the
     * CPU does after every iteration. The device accumulates them over the iterations of the
     * substep, so one readback per substep applies the whole impulse. */
    if (ok && gathers_coupled) {
      CoupledFaceContacts &coupled = chunks_data_[chunk_i].coupled_face_contacts;
      ExternalFaceContacts &contacts = coupled.contacts;
      if (!contacts.points.is_empty()) {
        fluid.gpu_fluid.download_contact_lambdas(contacts.lambdas_normal.as_mutable_span(),
                                                 contacts.lambdas_friction.as_mutable_span());
        coupled.applied_lambdas.reinitialize(int(contacts.points.size()));
        coupled.applied_lambdas.fill(0.0f);
        this->simulate__coupled_contact_reaction__chunk(
            chunk_i, geometries_.solver_refs[solver_refs_i], /*clamp_sum=*/true);
      }
    }
    /* The reaction of the whole frame (all substeps and iterations) is read back once, at the end
     * of the last substep: a readback per substep costs ~100 ms of Vulkan stall overhead. */
    if (ok && download_result && fluid.gpu_fluid.coupled_count() > 0) {
      Array<float3> corrections(fluid.gpu_fluid.coupled_count());
      if (fluid.gpu_fluid.download_coupled_reaction(corrections.as_mutable_span())) {
        this->simulate__gpu_coupled_reaction(fluid, corrections, solver_refs_i);
      }
    }
    /* Hard contacts with the coupled distance field, also once per frame. */
    if (ok && download_result) {
      const int max_contacts = int(points.size());
      Array<int> contact_indices(max_contacts);
      Array<float3> contact_push(max_contacts);
      int contact_count = 0;
      if (fluid.gpu_fluid.download_coupled_contacts(contact_indices.as_mutable_span(),
                                                   contact_push.as_mutable_span(),
                                                   contact_count))
      {
        this->simulate__gpu_coupled_sdf_reaction(
            fluid,
            chunk_i,
            contact_indices.as_span().take_front(contact_count),
            contact_push.as_span().take_front(contact_count),
            solver_refs_i);
      }
    }
    if (gathers_collider) {
      /* Seed the next gather with the multipliers the device ended up with. */
      ExternalFaceContacts &contacts = chunks_data_[chunk_i].external_face_contacts;
      fluid.gpu_fluid.download_contact_lambdas(contacts.lambdas_normal.as_mutable_span(),
                                               contacts.lambdas_friction.as_mutable_span());
    }
    if (!ok || download_result) {
      const MutableSpan<float3> velocities = geo_data.velocity_attr.span.slice(points);
      fluid.gpu_fluid.download(positions, velocities);
      if (download_result) {
        const geometry::ha::HaXpbdFluidFrameStats &stats = fluid.gpu_fluid.frame_stats();
        int neighbor_total = 0;
        geometry::ha::ha_xpbd_neighbors_total(fluid.gpu_fluid.grid(), neighbor_total);
        debug_gpu_log("BLENDER_XPBD_GPU_LOG",
                      std::getenv("XPBD_GPU_DEBUG") != nullptr,
                      "XPBD_GPU frame backend=%s particles=%d substeps=%d frame_ms=%.2f "
                      "grid_ms=%.2f neighbors_ms=%.2f solve_ms=%.2f velocity_ms=%.2f passes=%d "
                      "neighbors=%d%s%s\n",
                      ok ? "device" : "cpu",
                      fluid.gpu_fluid.particle_count(),
                      stats.substeps,
                      time_now_ms() - fluid.gpu_start_ms,
                      stats.grid_ms,
                      stats.neighbors_ms,
                      stats.solve_ms,
                      stats.velocity_ms,
                      stats.passes,
                      neighbor_total,
                      ok ? "" : " reason=",
                      ok ? "" : fluid.gpu_fluid.last_error());
      }
    }
    if (!ok) {
      /* Fall back to the CPU for the rest of the frame: the device state is unchanged, so the
       * download above is a valid starting point. */
      fluid.gpu_active = false;
      fluid.gpu_fluid.free();
    }
  }

  /**
   * The same for the path that solves the whole world in one substep loop (`Self Collision` or
   * constraint sets that span several geometries). The device fluid does not take part in those
   * constraints, so it can advance its whole substep at once; only the collider contacts of the
   * fluid are gathered here, inside the device substep.
   */
  void simulate__gpu_fluid_substep_unchunked(const SubstepInterval &substep,
                                             const int solver_refs_i)
  {
    const bool download_result = substep.current_i + 1 == substeps_;
    for (const int chunk_i : chunks_data_.index_range()) {
      if (!chunks_data_[chunk_i].fluid.gpu_active) {
        continue;
      }
      this->simulate__gpu_fluid_substep(chunk_i, solver_refs_i, substep, download_result);
    }
  }

  /** Hands the contacts of the current substep to the device. */
  bool upload_contacts_to_gpu(const int chunk_i, const ExternalFaceContacts &contacts)
  {
    FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    if (contacts.points.is_empty()) {
      return false;
    }
    geometry::ha::HaXpbdContacts device_contacts;
    device_contacts.points = contacts.points.as_span();
    device_contacts.radii = contacts.point_radii.as_span();
    device_contacts.positions_on_face = contacts.positions_on_face.as_span();
    device_contacts.collider_motion = contacts.collider_motion.as_span();
    device_contacts.collider_velocities = contacts.collider_velocities.as_span();
    device_contacts.face_normals = contacts.face_normals.as_span();
    device_contacts.margins = contacts.face_margins.as_span();
    device_contacts.static_frictions = contacts.static_frictions.as_span();
    device_contacts.dynamic_frictions = contacts.dynamic_frictions.as_span();
    device_contacts.compliance_terms = contacts.compliance_terms.as_span();
    device_contacts.error_scales = contacts.error_scales.as_span();
    device_contacts.lambdas_normal = contacts.lambdas_normal.as_span();
    device_contacts.lambdas_friction = contacts.lambdas_friction.as_span();
    return fluid.gpu_fluid.set_contacts(device_contacts);
  }

  void simulate__reset_forces()
  {
    this->parallel_for_each_chunk(
        16, [&](const int chunk_i) { this->simulate__reset_forces__chunk(chunk_i); });
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];
      for (ConstraintWithColoring &constraint : geo_data.static_constraints) {
        constraint.constraint->reset_forces();
      }
    }
    /* The hair contacts are built once per frame and keep their multipliers across the substeps, so
     * reset them like every other constraint set. */
    if (hair_contacts_.constraint_set.has_value()) {
      hair_contacts_.constraint_set->reset_forces();
    }
    if (guide_tether_contacts_.constraint_set.has_value()) {
      guide_tether_contacts_.constraint_set->reset_forces();
    }
  }

  void simulate__reset_forces__chunk(const int chunk_i)
  {
    ChunkData &chunk_data = chunks_data_[chunk_i];
    chunk_data.external_face_contacts.lambdas_normal.fill(0.0f);
    chunk_data.external_face_contacts.lambdas_friction.fill(0.0f);
    for (xpbd::ConstraintSet *constraint : chunk_data.static_constraints) {
      constraint->reset_forces();
    }
    for (xpbd::VelocityConstraintSet *constraint : chunk_data.static_velocity_constraints) {
      constraint->reset_forces();
    }
  }

  void simulate__position_solve__single_iteration(const int solver_refs_i,
                                                  float &r_average_error_squared)
  {
    const Span<xpbd::GeometryRef> solver_refs = geometries_.solver_refs[solver_refs_i];
    xpbd::ConstraintSetParams solve_params{solver_refs, sub_delta_time_};
    threading::EnumerableThreadSpecific<ResidualErrorTLS> error_tls;

    this->parallel_for_each_chunk(1, [&](const int chunk_i) {
      xpbd::GaussSeidelUpdater updater{solver_refs};
      this->simulate__position_solve__single_iteration__chunk(chunk_i, solve_params, updater);
      error_tls.local().total_error_squared += updater.total_error_squared();
      error_tls.local().total_error_count += updater.total_error_count();
    });

    for (const int data_key_i : geometries_.data_keys.index_range()) {
      const GeometryData &geo_data = *geometries_.data[data_key_i];
      for (const ConstraintWithColoring &constraint : geo_data.static_constraints) {
        if (std::getenv("XPBD_DEBUG_NAN") != nullptr) {
          for (const int geo_i : constraint.constraint->get_affected_geo_indices()) {
            const Span<float3> positions = solver_refs[geo_i].positions;
            for (const int point_i : positions.index_range()) {
              const float3 &p = positions[point_i];
              if (!(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))) {
                printf("XPBD_DEBUG_NAN before \"%s\": geo %d point %d = (%g %g %g)\n",
                       constraint.constraint->debug_name().c_str(),
                       geo_i,
                       point_i,
                       double(p.x),
                       double(p.y),
                       double(p.z));
                break;
              }
            }
          }
        }
        for (const int color_i : constraint.coloring.colors.index_range()) {
          const IndexMask &mask = constraint.coloring.colors[color_i];
          threading::parallel_for(mask.index_range(), 512, [&](const IndexRange range) {
            const IndexMask sliced_mask = mask.slice(range);
            xpbd::GaussSeidelUpdater updater{solver_refs};
            constraint.constraint->solve_sequential(solve_params, updater, sliced_mask);
            error_tls.local().total_error_squared += updater.total_error_squared();
            error_tls.local().total_error_count += updater.total_error_count();
          });
        }
      }
    }

    if (std::getenv("XPBD_SELF_DEBUG") != nullptr && self_contacts_.constraint_set) {
      const SelfContacts &contacts = self_contacts_;
      float worst = 0.0f;
      int active = 0;
      for (const int pair_i : contacts.point_pairs.index_range()) {
        const int g0 = contacts.point_pairs[pair_i][0];
        const int g1 = contacts.point_pairs[pair_i][1];
        const float3 &p0 = solver_refs[contacts.point_geo[g0]].positions[contacts.point_local[g0]];
        const float3 &p1 = solver_refs[contacts.point_geo[g1]].positions[contacts.point_local[g1]];
        const float distance = math::distance(p0, p1);
        const float penetration = contacts.min_distances[pair_i] - distance;
        if (penetration > 0.0f) {
          active++;
          worst = std::max(worst, penetration);
        }
      }
      printf("XPBD_SELF penetration worst %g active %d/%d\n",
             double(worst),
             active,
             int(contacts.point_pairs.size()));
    }

    if (self_contacts_.constraint_set.has_value()) {
      /* The contacts span all geometries (a cloth against a soft body is one contact), which is
       * why this is not part of the per chunk loop above. */
      xpbd::PointContactConstraintSet &constraint = *self_contacts_.constraint_set;
      for (const int color_i : self_contacts_.coloring.colors.index_range()) {
        const IndexMask &mask = self_contacts_.coloring.colors[color_i];
        threading::parallel_for(mask.index_range(), 512, [&](const IndexRange range) {
          const IndexMask sliced_mask = mask.slice(range);
          xpbd::GaussSeidelUpdater updater{solver_refs};
          constraint.solve_sequential(solve_params, updater, sliced_mask);
          error_tls.local().total_error_squared += updater.total_error_squared();
          error_tls.local().total_error_count += updater.total_error_count();
        });
      }
    }

    if (hair_contacts_.constraint_set.has_value()) {
      /* The hair contacts also span all curve geometries, so they are solved after the per chunk
       * constraints and the point contacts. */
      xpbd::SegmentContactConstraintSet &constraint = *hair_contacts_.constraint_set;
      for (const int color_i : hair_contacts_.coloring.colors.index_range()) {
        const IndexMask &mask = hair_contacts_.coloring.colors[color_i];
        threading::parallel_for(mask.index_range(), 512, [&](const IndexRange range) {
          const IndexMask sliced_mask = mask.slice(range);
          xpbd::GaussSeidelUpdater updater{solver_refs};
          constraint.solve_sequential(solve_params, updater, sliced_mask);
          error_tls.local().total_error_squared += updater.total_error_squared();
          error_tls.local().total_error_count += updater.total_error_count();
        });
      }
    }

    if (hair_solid_contacts_.constraint_set.has_value()) {
      /* Hair against the cloth and the soft bodies: the constraint moves the hair segment and the
       * triangle it touches, so the coupling is two way. */
      xpbd::SegmentFaceContactConstraintSet &constraint =
          *hair_solid_contacts_.constraint_set;
      for (const int color_i : hair_solid_contacts_.coloring.colors.index_range()) {
        const IndexMask &mask = hair_solid_contacts_.coloring.colors[color_i];
        threading::parallel_for(mask.index_range(), 512, [&](const IndexRange range) {
          const IndexMask sliced_mask = mask.slice(range);
          xpbd::GaussSeidelUpdater updater{solver_refs};
          constraint.solve_sequential(solve_params, updater, sliced_mask);
          error_tls.local().total_error_squared += updater.total_error_squared();
          error_tls.local().total_error_count += updater.total_error_count();
        });
      }
    }

    if (guide_tether_contacts_.constraint_set.has_value()) {
      /* The guide tethers are solved after the contacts: a child that a contact moved now pulls its
       * guide with it (the correction is split by inverse mass, so the effect is two way). */
      xpbd::GuideTetherConstraintSet &constraint = *guide_tether_contacts_.constraint_set;
      for (const int color_i : guide_tether_contacts_.coloring.colors.index_range()) {
        const IndexMask &mask = guide_tether_contacts_.coloring.colors[color_i];
        threading::parallel_for(mask.index_range(), 512, [&](const IndexRange range) {
          const IndexMask sliced_mask = mask.slice(range);
          xpbd::GaussSeidelUpdater updater{solver_refs};
          constraint.solve_sequential(solve_params, updater, sliced_mask);
          error_tls.local().total_error_squared += updater.total_error_squared();
          error_tls.local().total_error_count += updater.total_error_count();
        });
      }
    }

    float total_error_squared = 0.0f;
    int total_error_count = 0;
    for (const ResidualErrorTLS &error : error_tls) {
      total_error_squared += error.total_error_squared;
      total_error_count += error.total_error_count;
    }
    if (total_error_count) {
      r_average_error_squared = total_error_squared / total_error_count;
    }
  }

  void simulate__position_solve__single_iteration__chunk(const int chunk_i,
                                                         xpbd::ConstraintSetParams solve_params,
                                                         xpbd::GaussSeidelUpdater &updater)
  {
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    ChunkData &chunk_data = chunks_data_[chunk_i];

    for (xpbd::ConstraintSet *constraint : chunk_data.static_constraints) {
      constraint->solve_sequential_all(solve_params, updater);
    }

    this->simulate__fluid_position_solve__chunk(chunk_i, solve_params, updater);

    if (!chunk_data.fluid.gpu_active && !chunk_data.external_face_contacts.points.is_empty()) {
      ExternalFaceContacts &contacts = chunk_data.external_face_contacts;
      xpbd::CollisionFaceConstraintSet constraint(chunk.data_key_i,
                                                  contacts.points,
                                                  contacts.point_radii,
                                                  contacts.positions_on_face,
                                                  contacts.collider_motion,
                                                  contacts.face_normals,
                                                  contacts.face_margins,
                                                  contacts.compliance_terms,
                                                  contacts.static_frictions,
                                                  contacts.dynamic_frictions,
                                                  contacts.error_scales,
                                                  contacts.active_states,
                                                  contacts.lambdas_normal);
      constraint.solve_sequential_all(solve_params, updater);
    }
    if (!chunk_data.external_edge_contacts.point_pairs.is_empty()) {
      ExternalEdgeContacts &contacts = chunk_data.external_edge_contacts;
      xpbd::CollisionEdgeConstraintSet constraint(chunk.data_key_i,
                                                  contacts.point_pairs,
                                                  contacts.point_radii,
                                                  contacts.positions_on_edge,
                                                  contacts.collider_motion,
                                                  contacts.edge_directions,
                                                  contacts.edge_normals,
                                                  contacts.edge_margins,
                                                  contacts.compliance_terms,
                                                  contacts.static_frictions,
                                                  contacts.dynamic_frictions,
                                                  contacts.error_scales,
                                                  contacts.active_states,
                                                  contacts.point_mix_factors,
                                                  contacts.lambdas_normal);
      constraint.solve_sequential_all(solve_params, updater);
    }
    if (!chunk_data.coupled_face_contacts.contacts.points.is_empty()) {
      ExternalFaceContacts &contacts = chunk_data.coupled_face_contacts.contacts;
      xpbd::CollisionFaceConstraintSet constraint(chunk.data_key_i,
                                                  contacts.points,
                                                  contacts.point_radii,
                                                  contacts.positions_on_face,
                                                  contacts.collider_motion,
                                                  contacts.face_normals,
                                                  contacts.face_margins,
                                                  contacts.compliance_terms,
                                                  contacts.static_frictions,
                                                  contacts.dynamic_frictions,
                                                  contacts.error_scales,
                                                  contacts.active_states,
                                                  contacts.lambdas_normal);
      constraint.solve_sequential_all(solve_params, updater);
    }
  }

  void simulate__update_velocities__chunk(const int chunk_i, const int solver_refs_i)
  {
    if (chunks_data_[chunk_i].fluid.gpu_active) {
      /* The velocities already come from the device, see #simulate__gpu_fluid_substep. */
      return;
    }
    if (sub_delta_time_ <= 0.0f) {
      /* This may happen for example on the first frame when no time has passed yet. */
      return;
    }

    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    const IndexRange points_range = chunk.points_range;
    const int data_key_i = chunk.data_key_i;
    GeometryData &geo_data = *geometries_.data[data_key_i];
    xpbd::GeometryRef &ref = geometries_.solver_refs[solver_refs_i][data_key_i];
    FluidChunkData &fluid = chunks_data_[chunk_i].fluid;
    const Span<float3> coupled_push = fluid.active ?
                                          fluid.coupled_push.as_span().slice(points_range) :
                                          Span<float3>();
    this->update_linear_velocities(sub_delta_time_,
                                   ref.prev_positions.slice(points_range),
                                   ref.positions.slice(points_range),
                                   geo_data.velocity_attr.span.slice(points_range),
                                   coupled_push);
    if (geo_data.uses_rotation) {
      this->update_angular_velocities(sub_delta_time_,
                                      ref.prev_rotations.slice(points_range),
                                      ref.rotations.slice(points_range),
                                      geo_data.angular_velocity_attr.span.slice(points_range));
    }
  }

  void simulate__velocity_solve(const int solver_refs_i)
  {
    this->parallel_for_each_chunk(8, [&](const int chunk_i) {
      this->simulate__velocity_solve__chunk(chunk_i, solver_refs_i);
    });
  }

  void simulate__velocity_solve__chunk(const int chunk_i, const int solver_refs_i)
  {
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    ChunkData &chunk_data = chunks_data_[chunk_i];

    const Span<xpbd::GeometryRef> solver_refs = geometries_.solver_refs[solver_refs_i];
    xpbd::ConstraintSetParams params{solver_refs, sub_delta_time_};
    xpbd::VelocityUpdater velocity_updater{solver_refs};

    for (xpbd::VelocityConstraintSet *constraint : chunk_data.static_velocity_constraints) {
      constraint->solve_sequential(params, velocity_updater);
    }

    this->simulate__fluid_velocity_solve__chunk(chunk_i, solver_refs_i);

    if (chunk_data.fluid.gpu_active) {
      /* The device also solved the collider friction of this substep. */
      return;
    }
    if (!chunk_data.external_face_contacts.points.is_empty()) {
      ExternalFaceContacts &contacts = chunk_data.external_face_contacts;
      xpbd::FrictionFaceConstraintSet constraint(chunk.data_key_i,
                                                 contacts.points,
                                                 contacts.face_normals,
                                                 contacts.collider_velocities,
                                                 contacts.dynamic_frictions,
                                                 contacts.lambdas_normal,
                                                 contacts.lambdas_friction);
      constraint.solve_sequential(params, velocity_updater);
    }
    if (!chunk_data.external_edge_contacts.point_pairs.is_empty()) {
      ExternalEdgeContacts &contacts = chunk_data.external_edge_contacts;
      xpbd::FrictionEdgeConstraintSet constraint(chunk.data_key_i,
                                                 contacts.point_pairs,
                                                 contacts.edge_normals,
                                                 contacts.collider_velocities,
                                                 contacts.dynamic_frictions,
                                                 contacts.lambdas_normal,
                                                 contacts.point_mix_factors,
                                                 contacts.lambdas_friction);
      constraint.solve_sequential(params, velocity_updater);
    }
  }

  void simulate__ensure_final_data_in_outputs__chunk(const int chunk_i)
  {
    if (substeps_ % 2 == 0) {
      /* Nothing to do. */
      return;
    }
    const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
    GeometryData &geo_data = *geometries_.data[chunk.data_key_i];
    const IndexRange points_range = chunk.points_range;
    geo_data.position_attr.span.slice(points_range)
        .copy_from(geo_data.temp_positions.as_span().slice(points_range));
    if (geo_data.uses_rotation) {
      geo_data.rotation_attr.span.slice(points_range)
          .copy_from(geo_data.temp_rotations.as_span().slice(points_range));
    }
  }

  void finish_common_attribute_writers()
  {
    for (const int data_key_i : geometries_.data_keys.index_range()) {
      GeometryData &geo_data = *geometries_.data[data_key_i];

      geo_data.position_attr.finish();
      geo_data.velocity_attr.finish();
      geo_data.rotation_attr.finish();
      geo_data.angular_velocity_attr.finish();
    }
  }

  void integrate_linear_velocities(const float delta_time,
                                   const Span<float3> old_positions,
                                   MutableSpan<float3> new_positions,
                                   MutableSpan<float3> velocities,
                                   const Span<float> inv_masses,
                                   const Span<float3> forces,
                                   const Span<float3> external_accelerations)
  {
    const bool has_external_acceleration = external_accelerations.size() == old_positions.size();
    for (const int i : old_positions.index_range()) {
      const float3 external_force = forces[i];
      const float inv_mass = inv_masses[i];
      const float3 &old_pos = old_positions[i];
      /* A force field applies to free points only. Pinned points have zero inverse mass and must
       * stay where the pin constraint put them; adding the acceleration unconditionally would move
       * the roots (and cloth pins) every substep before the pin pulls them back. */
      const float3 acceleration = external_force * inv_mass +
                                  (has_external_acceleration && inv_mass > 0.0f ?
                                       external_accelerations[i] :
                                       float3(0.0f));
      const float3 new_velocity = velocities[i] + acceleration * delta_time;
      velocities[i] = new_velocity;
      new_positions[i] = old_pos + new_velocity * delta_time;
    }
  }

  void integrate_angular_velocities(const float delta_time,
                                    const Span<math::Quaternion> old_rotations,
                                    MutableSpan<math::Quaternion> new_rotations,
                                    MutableSpan<float3> angular_velocities,
                                    const Span<float3> inv_inertias,
                                    const Span<float3> torques)
  {
    for (const int i : old_rotations.index_range()) {
      const math::Quaternion &old_rotation = old_rotations[i];
      const float3 &external_torque = torques[i];
      const float3 &inv_inertia = inv_inertias[i];
      if (math::is_zero(inv_inertia)) {
        new_rotations[i] = old_rotation;
        continue;
      }
      float3 &angular_velocity = angular_velocities[i];
      const float3 precession = math::cross(angular_velocity,
                                            math::safe_divide(angular_velocity, inv_inertia));
      angular_velocity += delta_time * (external_torque - precession) * inv_inertia;
      const math::Quaternion direction = old_rotation * math::Quaternion(0, angular_velocity);
      new_rotations[i] = math::normalize(
          math::Quaternion(float4(old_rotation) + delta_time * 0.5f * float4(direction)));
    }
  }

  void update_linear_velocities(const float delta_time,
                                const Span<float3> prev_positions,
                                const Span<float3> new_positions,
                                MutableSpan<float3> r_velocities,
                                const Span<float3> position_offsets = {})
  {
    const float inv_delta_time = math::safe_rcp(delta_time);
    const bool has_offsets = position_offsets.size() == r_velocities.size();
    for (const int i : r_velocities.index_range()) {
      const float3 &old_pos = prev_positions[i];
      const float3 &new_pos = new_positions[i];
      float3 velocity = (new_pos - old_pos) * inv_delta_time;
      if (has_offsets) {
        /* A position correction from the coupled solids is a depenetration, not a physical
         * velocity. Remove the velocity component along the correction, so a particle that was
         * pushed out of the cloth does not bounce back up (the fluid still gets the tangential
         * part, which is what makes it slide along the surface). */
        const float3 &offset = position_offsets[i];
        const float length = math::length(offset);
        if (length > 1e-12f) {
          const float3 normal = offset / length;
          velocity -= math::dot(velocity, normal) * normal;
        }
      }
      r_velocities[i] = velocity;
    }
  }

  void update_angular_velocities(const float delta_time,
                                 const Span<math::Quaternion> prev_rotations,
                                 const Span<math::Quaternion> new_rotations,
                                 MutableSpan<float3> r_angular_velocities)
  {
    const float inv_delta_time = math::safe_rcp(delta_time);
    for (const int i : r_angular_velocities.index_range()) {
      float3 diff =
          (math::invert_normalized(prev_rotations[i]) * new_rotations[i]).imaginary_part();
      for (const int j : IndexRange(3)) {
        if (math::abs(diff[j]) < 1e-5f) {
          diff[j] = 0.0f;
        }
      }
      const float3 new_angular_velocity = 2.0f * diff * inv_delta_time;
      r_angular_velocities[i] = new_angular_velocity;
    }
  }

  float compute_contact_friction(const float point_friction, const float collider_friction) const
  {
    return math::sqrt(point_friction * collider_friction);
  }

  float get_max_search_distance(const float delta_time)
  {
    /* This is 60 m/s or 216 km/h. */
    constexpr float max_velocity = 60.0f;
    const float max_search_distance = 2.0f * max_velocity * delta_time;
    return max_search_distance;
  }

  template<typename T>
  bke::SpanAttributeWriter<T> ensure_attribute(bke::MutableAttributeAccessor attributes,
                                               const StringRef name,
                                               const bke::AttrDomain domain)
  {
    const std::optional<bke::AttributeMetaData> meta_data = attributes.lookup_meta_data(name);
    if (meta_data) {
      if (meta_data->domain != domain ||
          meta_data->data_type != bke::cpp_type_to_attribute_type(CPPType::get<T>()))
      {
        attributes.remove(name);
      }
    }
    return attributes.lookup_or_add_for_write_span<T>(name, domain);
  }

  template<typename T>
  bke::SpanAttributeWriter<T> get_output_attribute_writer(const int geo_data_i,
                                                          const StringRef name,
                                                          const AttrDomain domain)
  {
    if (name.is_empty()) {
      return {};
    }
    GeometryData &geo_data = *geometries_.data[geo_data_i];
    return geo_data.attributes.lookup_or_add_for_write_span<T>(name, domain);
  }

  const Bundle *get_previous_bundle(const Bundle &bundle) const
  {
    const BundlePtr *previous_bundle_ptr = bundle.lookup_ptr<BundlePtr>(
        *BundleKey::from_str("previous"));
    if (!previous_bundle_ptr || !*previous_bundle_ptr) {
      return nullptr;
    }
    return &**previous_bundle_ptr;
  }

  bool effector_applies_to_geometry([[maybe_unused]] const StringRef effector_path,
                                    const Bundle &effector,
                                    const int data_key_i) const
  {
    const DataKey &data_key = geometries_.data_keys[data_key_i];
    const GeometrySetData &geo_set_data = geometries_.geometry_sets[data_key.geo_bundle_i];

    const std::string filter =
        effector.lookup<std::string>(*BundleKey::from_str("filter")).value_or("");
    return tag_filter_matches(filter, geo_set_data.tags);
  }

  void parallel_for_each_chunk(const int grain_size, const FunctionRef<void(int chunk_i)> fn)
  {
    threading::parallel_for(
        geometries_.chunks.index_range(), grain_size, [&](const IndexRange chunk_range) {
          for (const int chunk_i : chunk_range) {
            fn(chunk_i);
          }
        });
  }

  void write_back_geometries_to_world()
  {
    for (const int geo_bundle_i : geometries_.geometry_sets.index_range()) {
      GeometrySetData &geo_set_data = geometries_.geometry_sets[geo_bundle_i];
      world_.add_path_override(geo_set_data.path, std::move(geo_set_data.geometry));
    }
  }

  PointCloud *write_back__plane_contacts(const IndexRange mesh_colliders_range,
                                         const OffsetIndices<int> points_by_chunk)
  {
    PointCloud *pointcloud = BKE_pointcloud_new_nomain(points_by_chunk.total_size());
    MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
    bke::SpanAttributeWriter<int> geometries_writer =
        attributes.lookup_or_add_for_write_only_span<int>("geometry", AttrDomain::Point);
    bke::SpanAttributeWriter<int> colliders_writer =
        attributes.lookup_or_add_for_write_only_span<int>("collider", AttrDomain::Point);
    bke::SpanAttributeWriter<int> geometry_points0_writer =
        attributes.lookup_or_add_for_write_only_span<int>("geometry_point0", AttrDomain::Point);
    MutableSpan<float3> positions = pointcloud->positions_for_write();
    bke::SpanAttributeWriter<float3> collider_velocities_writer =
        attributes.lookup_or_add_for_write_only_span<float3>("collider_velocity",
                                                             AttrDomain::Point);
    bke::SpanAttributeWriter<float3> normals_writer =
        attributes.lookup_or_add_for_write_only_span<float3>("normal", AttrDomain::Point);
    bke::SpanAttributeWriter<float> static_frictions_writer =
        attributes.lookup_or_add_for_write_only_span<float>("static_friction", AttrDomain::Point);
    bke::SpanAttributeWriter<float> dynamic_frictions_writer =
        attributes.lookup_or_add_for_write_only_span<float>("dynamic_friction", AttrDomain::Point);
    bke::SpanAttributeWriter<bool> active_states_writer =
        attributes.lookup_or_add_for_write_only_span<bool>("active", AttrDomain::Point);
    bke::SpanAttributeWriter<float> lambdas_normal_writer =
        attributes.lookup_or_add_for_write_only_span<float>("lambda_normal", AttrDomain::Point);
    bke::SpanAttributeWriter<float> lambdas_friction_writer =
        attributes.lookup_or_add_for_write_only_span<float>("lambda_friction", AttrDomain::Point);

    this->parallel_for_each_chunk(16, [&](const int chunk_i) {
      const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
      const ChunkData &chunk_data = chunks_data_[chunk_i];
      const ExternalFaceContacts &contacts = chunk_data.external_face_contacts;

      const IndexRange points = points_by_chunk[chunk_i];
      if (points.is_empty()) {
        /* Avoids checking the filter a second time. */
        return;
      }

      /* Store collider indices and affected geometry point indices that map a contact to the
       * simulated geometry. */
      MutableSpan<int> geometries = geometries_writer.span.slice(points);
      MutableSpan<int> colliders = colliders_writer.span.slice(points);
      MutableSpan<int> geometry_points0 = geometry_points0_writer.span.slice(points);

      geometries.fill(chunk.data_key_i);
      for (const auto &item : contacts.mesh_contact_indices.items()) {
        colliders[item.value] = mesh_colliders_range[item.key.mesh_collider_i];
        geometry_points0[item.value] = contacts.points[item.value];
      }

      array_utils::copy(contacts.positions_on_face.as_span(), positions.slice(points));
      array_utils::copy(contacts.collider_velocities.as_span(),
                        collider_velocities_writer.span.slice(points));
      array_utils::copy(contacts.face_normals.as_span(), normals_writer.span.slice(points));
      array_utils::copy(contacts.static_frictions.as_span(),
                        static_frictions_writer.span.slice(points));
      array_utils::copy(contacts.dynamic_frictions.as_span(),
                        dynamic_frictions_writer.span.slice(points));
      array_utils::copy(contacts.active_states.as_span(), active_states_writer.span.slice(points));
      array_utils::copy(contacts.lambdas_normal.as_span(),
                        lambdas_normal_writer.span.slice(points));
      array_utils::copy(contacts.lambdas_friction.as_span(),
                        lambdas_friction_writer.span.slice(points));
    });
    pointcloud->tag_positions_changed();
    geometries_writer.finish();
    colliders_writer.finish();
    geometry_points0_writer.finish();
    collider_velocities_writer.finish();
    normals_writer.finish();
    static_frictions_writer.finish();
    dynamic_frictions_writer.finish();
    active_states_writer.finish();
    lambdas_normal_writer.finish();
    lambdas_friction_writer.finish();

    return pointcloud;
  }

  PointCloud *write_back__edge_contacts(const IndexRange mesh_colliders_range,
                                        const OffsetIndices<int> points_by_chunk)
  {
    PointCloud *pointcloud = BKE_pointcloud_new_nomain(points_by_chunk.total_size());
    MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
    bke::SpanAttributeWriter<int> geometries_writer =
        attributes.lookup_or_add_for_write_only_span<int>("geometry", AttrDomain::Point);
    bke::SpanAttributeWriter<int> colliders_writer =
        attributes.lookup_or_add_for_write_only_span<int>("collider", AttrDomain::Point);
    bke::SpanAttributeWriter<int> geometry_points0_writer =
        attributes.lookup_or_add_for_write_only_span<int>("geometry_point0", AttrDomain::Point);
    bke::SpanAttributeWriter<int> geometry_points1_writer =
        attributes.lookup_or_add_for_write_only_span<int>("geometry_point1", AttrDomain::Point);
    MutableSpan<float3> positions = pointcloud->positions_for_write();
    bke::SpanAttributeWriter<float3> collider_velocities_writer =
        attributes.lookup_or_add_for_write_only_span<float3>("collider_velocity",
                                                             AttrDomain::Point);
    bke::SpanAttributeWriter<float3> normals_writer =
        attributes.lookup_or_add_for_write_only_span<float3>("normal", AttrDomain::Point);
    bke::SpanAttributeWriter<float> static_frictions_writer =
        attributes.lookup_or_add_for_write_only_span<float>("static_friction", AttrDomain::Point);
    bke::SpanAttributeWriter<float> dynamic_frictions_writer =
        attributes.lookup_or_add_for_write_only_span<float>("dynamic_friction", AttrDomain::Point);
    bke::SpanAttributeWriter<bool> active_states_writer =
        attributes.lookup_or_add_for_write_only_span<bool>("active", AttrDomain::Point);
    bke::SpanAttributeWriter<float> lambdas_normal_writer =
        attributes.lookup_or_add_for_write_only_span<float>("lambda_normal", AttrDomain::Point);
    bke::SpanAttributeWriter<float> lambdas_friction_writer =
        attributes.lookup_or_add_for_write_only_span<float>("lambda_friction", AttrDomain::Point);

    this->parallel_for_each_chunk(16, [&](const int chunk_i) {
      const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
      const ChunkData &chunk_data = chunks_data_[chunk_i];
      const ExternalEdgeContacts &contacts = chunk_data.external_edge_contacts;

      const IndexRange points = points_by_chunk[chunk_i];
      if (points.is_empty()) {
        /* Avoids checking the filter a second time. */
        return;
      }

      /* Store collider indices and affected geometry point indices that map a contact to the
       * simulated geometry. */
      MutableSpan<int> geometries = geometries_writer.span.slice(points);
      MutableSpan<int> colliders = colliders_writer.span.slice(points);
      MutableSpan<int> geometry_points0 = geometry_points0_writer.span.slice(points);
      MutableSpan<int> geometry_points1 = geometry_points1_writer.span.slice(points);
      geometries.fill(chunk.data_key_i);
      for (const auto &item : contacts.mesh_contact_indices.items()) {
        colliders[item.value] = mesh_colliders_range[item.key.mesh_collider_i];
        geometry_points0[item.value] = contacts.point_pairs[item.value][0];
        geometry_points1[item.value] = contacts.point_pairs[item.value][1];
      }

      array_utils::copy(contacts.positions_on_edge.as_span(), positions.slice(points));
      array_utils::copy(contacts.collider_velocities.as_span(),
                        collider_velocities_writer.span.slice(points));
      array_utils::copy(contacts.edge_normals.as_span(), normals_writer.span.slice(points));
      array_utils::copy(contacts.static_frictions.as_span(),
                        static_frictions_writer.span.slice(points));
      array_utils::copy(contacts.dynamic_frictions.as_span(),
                        dynamic_frictions_writer.span.slice(points));
      array_utils::copy(contacts.active_states.as_span(), active_states_writer.span.slice(points));
      array_utils::copy(contacts.lambdas_normal.as_span(),
                        lambdas_normal_writer.span.slice(points));
      array_utils::copy(contacts.lambdas_friction.as_span(),
                        lambdas_friction_writer.span.slice(points));
    });
    pointcloud->tag_positions_changed();
    geometries_writer.finish();
    colliders_writer.finish();
    geometry_points0_writer.finish();
    geometry_points1_writer.finish();
    collider_velocities_writer.finish();
    normals_writer.finish();
    static_frictions_writer.finish();
    dynamic_frictions_writer.finish();
    active_states_writer.finish();
    lambdas_normal_writer.finish();
    lambdas_friction_writer.finish();

    return pointcloud;
  }

  void write_back__contacts()
  {
    for (CollisionContacts &contacts : this->constraints_.collision_contacts) {
      const Bundle &bundle = **world_.lookup_path_ptr<BundlePtr>(contacts.path);
      const std::string plane_points_path = Bundle::combine_path(
          {contacts.path, "plane_contacts"});
      const std::string edge_points_path = Bundle::combine_path({contacts.path, "edge_contacts"});
      const std::string collider_map_path = Bundle::combine_path({contacts.path, "collider_map"});

      /* Collider paths are combined in a single array, these are offsets for collider indices. */
      const IndexRange geometries_range = geometries_.geometry_sets.index_range();
      const IndexRange mesh_colliders_range = geometries_range.after(
          constraints_.mesh_colliders.size());
      Vector<std::string> collider_paths;
      collider_paths.reserve(geometries_range.size() + mesh_colliders_range.size());
      for (const GeometrySetData &geometry_set_data : geometries_.geometry_sets) {
        collider_paths.append_unchecked(geometry_set_data.path);
      }
      for (const MeshCollider &collider : constraints_.mesh_colliders) {
        collider_paths.append_unchecked(collider.path);
      }

      Array<int> plane_contacts_offsets(geometries_.chunks.size() + 1);
      Array<int> edge_contacts_offsets(geometries_.chunks.size() + 1);
      this->parallel_for_each_chunk(16, [&](const int chunk_i) {
        const GeometryDataChunk &chunk = geometries_.chunks[chunk_i];
        const ChunkData &chunk_data = chunks_data_[chunk_i];

        if (!effector_applies_to_geometry(contacts.path, bundle, chunk.data_key_i)) {
          plane_contacts_offsets[chunk_i] = 0;
          edge_contacts_offsets[chunk_i] = 0;
          return;
        }

        plane_contacts_offsets[chunk_i] = chunk_data.external_face_contacts.points.size();
        edge_contacts_offsets[chunk_i] = chunk_data.external_edge_contacts.point_pairs.size();
      });
      const OffsetIndices plane_contacts_by_chunk = offset_indices::accumulate_counts_to_offsets(
          plane_contacts_offsets);
      const OffsetIndices edge_contacts_by_chunk = offset_indices::accumulate_counts_to_offsets(
          edge_contacts_offsets);

      GeometrySet plane_contacts_geometry;
      GeometrySet edge_contacts_geometry;
      if (plane_contacts_by_chunk.total_size() > 0) {
        plane_contacts_geometry.replace_pointcloud(
            write_back__plane_contacts(mesh_colliders_range, plane_contacts_by_chunk));
      }
      if (edge_contacts_by_chunk.total_size() > 0) {
        edge_contacts_geometry.replace_pointcloud(
            write_back__edge_contacts(mesh_colliders_range, edge_contacts_by_chunk));
      }

      world_.add_path_override(plane_points_path, std::move(plane_contacts_geometry));
      world_.add_path_override(edge_points_path, std::move(edge_contacts_geometry));
      /* TODO Currently have to create an explicit BundleItemSocketValue to store a list with
       * add_path_override. It relies on socket_type_info_by_static_type, which only supports
       * fields and single values currently, but not lists. */
      world_.add_path_override(
          collider_map_path,
          BundleItemSocketValue{
              bke::node_socket_type_find_static(SOCK_STRING),
              bke::SocketValueVariant::From(GList::from_container(std::move(collider_paths)))});
    }
  }

  template<typename T> VArrayRangeSpans<T> make_range_spans(TLS &tls, const VArray<T> &varray)
  {
    return VArrayRangeSpans<T>(tls.scope, varray, geometries_.max_chunk_size);
  }

  template<typename T>
  VArray<T> lookup_attribute_default(const int data_key_i,
                                     const StringRef attribute_name,
                                     const AttrDomain domain,
                                     const T &default_value)
  {
    GeometryData &geo_data = *geometries_.data[data_key_i];
    VArray<T> varray = *geo_data.attributes.lookup<T>(attribute_name, domain);
    if (!varray) {
      this->report(
          NodeWarningType::Info,
          fmt::format(
              "Attribute \"{}\" not found in \"{}\", using fallback value",
              attribute_name,
              geometries_.geometry_sets[geometries_.data_keys[data_key_i].geo_bundle_i].path));
      return VArray<T>::from_single(default_value, geo_data.attributes.domain_size(domain));
    }
    return varray;
  }

  template<typename T>
  VArray<T> lookup_attribute_required(const int data_key_i,
                                      const StringRef attribute_name,
                                      const AttrDomain domain)
  {
    GeometryData &geo_data = *geometries_.data[data_key_i];
    VArray<T> varray = *geo_data.attributes.lookup<T>(attribute_name, domain);
    if (!varray) {
      this->report(
          NodeWarningType::Error,
          fmt::format(
              "Attribute \"{}\" not found on \"{}\", some functionality is disabled",
              attribute_name,
              geometries_.geometry_sets[geometries_.data_keys[data_key_i].geo_bundle_i].path));
    }
    return varray;
  }

  template<typename T>
  VArray<T> lookup_attribute_optional(const int data_key_i,
                                      const StringRef attribute_name,
                                      const AttrDomain domain)
  {
    GeometryData &geo_data = *geometries_.data[data_key_i];
    VArray<T> varray = *geo_data.attributes.lookup<T>(attribute_name, domain);
    if (!varray) {
      this->report(
          NodeWarningType::Info,
          fmt::format(
              "Optional attribute \"{}\" not found on \"{}\"",
              attribute_name,
              geometries_.geometry_sets[geometries_.data_keys[data_key_i].geo_bundle_i].path));
    }
    return varray;
  }

  std::string prop_attr_name(const StringRef effector_path, const StringRef prop_name) const
  {
    return fmt::format("sim:prop:{}:{}", effector_path, prop_name);
  }

  std::string prev_prop_attr_name(const StringRef effector_path, const StringRef prop_name) const
  {
    return fmt::format("sim:prop_prev:{}:{}", effector_path, prop_name);
  }

  void report(NodeWarningType type, std::string warning)
  {
    std::lock_guard<Mutex> lock(warnings_mutex_);
    warnings_.add({type, std::move(warning)});
  }
};

/**
 * Read only info block: what the last evaluation did on the compute backend. Blender's rule for
 * node UI is to keep status out of the node body unless it says something, so this only draws when
 * the node reported a state (i.e. it runs inside an `HW Accelerate` zone).
 */
static void node_layout_ex(ui::Layout &layout, bContext * /*C*/, PointerRNA *current_node_ptr)
{
  bNode &node = *static_cast<bNode *>(current_node_ptr->data);
  const bNodeTree &tree = node.owner_tree();
  geometry::ha::HaNodeStatus status;
  if (!geometry::ha::ha_status_get(tree, node, status)) {
    return;
  }
  layout.separator();
  if (status.state == geometry::ha::HaNodeState::Device) {
    layout.label(fmt::format("Device: {} · {} elements", status.backend, status.elements),
                 ICON_MEMORY);
  }
  else {
    layout.label(fmt::format("CPU: {}", status.reason), ICON_INFO);
  }
}

static void node_geo_exec(GeoNodeExecParams params)
{
  BundlePtr world_ptr = params.get_input<BundlePtr>("World"_ustr);
  if (!world_ptr) {
    params.set_default_remaining_outputs();
    return;
  }
  const int substeps = params.get_input<int>("Substeps"_ustr);
  if (substeps <= 0) {
    params.set_output("World"_ustr, std::move(world_ptr));
    return;
  }
  const int constraint_iterations = params.get_input<int>("Constraint Iterations"_ustr);
  const bool self_collision = params.get_input<bool>("Self Collision"_ustr);
  const float contact_thickness = params.get_input<float>("Contact Thickness"_ustr);
  const bool hair_self_collision = params.get_input<bool>("Hair Self Collision"_ustr);
  const float hair_contact_thickness = params.get_input<float>("Hair Contact Thickness"_ustr);
  const bool hair_same_strand_collision = params.get_input<bool>("Same Strand Collision"_ustr);
  const float delta_time = std::max(0.0f, params.get_input<float>("Delta Time"_ustr));
  Bundle &world = world_ptr.ensure_mutable_inplace();
  const std::string geometry_tag_filter = params.extract_input<std::string>("Filter"_ustr);
  const float4x4 simulation_to_world = params.extract_input<float4x4>("Simulation to World"_ustr);
  const float interpolation_begin = params.extract_input<float>("Begin"_ustr);
  const float interpolation_end = params.extract_input<float>("End"_ustr);
  const std::string solver_path = params.extract_input<std::string>("Solver Path"_ustr);

  /* `HW Accelerate` zone the solver is evaluated in, if any (0 = outside of a zone). The zone
   * wrapper puts the request into the evaluation data, see `GeoNodesUserData::ha_backend_request`. */
  const GeoNodesUserData &user_data = *static_cast<const GeoNodesUserData *>(params.user_data());
  const int backend_request = user_data.ha_backend_request ? *user_data.ha_backend_request : 0;

  XpbdSolverStep step(world,
                      delta_time,
                      substeps,
                      constraint_iterations,
                      self_collision,
                      contact_thickness,
                      hair_self_collision,
                      hair_contact_thickness,
                      hair_same_strand_collision,
                      interpolation_begin,
                      interpolation_end,
                      geometry_tag_filter,
                      simulation_to_world,
                      backend_request);
  step.do_step();
  if (backend_request != 0) {
    step.publish_ha_status(params.node().owner_tree(), params.node());
  }
  else {
    /* Outside of a zone: drop the state, so that the badge does not linger after the node was
     * taken out of the zone. */
    geometry::ha::ha_status_clear(params.node().owner_tree(), params.node());
  }

  if (!solver_path.empty()) {
    BundlePtr solver_data_ptr = Bundle::create();
    Bundle &solver_data = solver_data_ptr.ensure_mutable_inplace();
    solver_data.add(Bundle::type_item_name, std::string(XPBDSolverDataBundle::name));
    solver_data.add_path("residual_error", step.result().total_residual_error);
    world.add_path_override(solver_path, std::move(solver_data_ptr));
  }
  for (const std::pair<NodeWarningType, std::string> &warning : step.warnings()) {
    params.error_message_add(warning.first, warning.second);
  }

  params.set_output("World"_ustr, std::move(world_ptr));
}

static void node_label(const bNodeTree * /*ntree*/,
                       const bNode * /*node*/,
                       char *label,
                       const int label_maxncpy)
{
  BLI_strncpy_utf8(
      label, CTX_IFACE_(BLT_I18NCONTEXT_ID_NODETREE, "XPBD Solver (Experimental)"), label_maxncpy);
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeXPBDSolver"_ustr);
  ntype.ui_name = "XPBD Solver";
  ntype.ui_description = "Simulate physics using the XPBD framework";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  ntype.labelfunc = node_label;
  ntype.draw_buttons_ex = node_layout_ex;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_xpbd_solver_cc
