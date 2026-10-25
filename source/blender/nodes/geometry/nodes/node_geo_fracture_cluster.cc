/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * Fracture Cluster node.
 *
 * Groups fragments into clusters and turns the contacts between them into cluster aware
 * constraints. This is what makes two level (secondary) fracture work: a coarse grid defines the
 * big chunks, the constraints *between* chunks are the weak "glue" that breaks first, while the
 * pieces inside a chunk stay together.
 *
 * Inputs:
 * - `Pieces`: fragments as instances (with `piece_id`), e.g. from `Voronoi Fracture`.
 * - `Constraints`: the contact point cloud, e.g. from `RBD Constraint from Pieces`.
 * - `Cluster Size`: spacing of the coarse grid, per axis (a different value per axis gives
 *   elongated chunks, e.g. for wood).
 *
 * Outputs:
 * - `Pieces`: the same fragments plus a `cluster_id` attribute (instance domain).
 * - `Constraints`: *all* the incoming contacts plus `cluster_a` / `cluster_b` (the cluster of each
 *   piece) and `is_inter_cluster` (true when the two pieces are in different clusters). The
 *   contacts that cross a cluster boundary, and one extra appended contact per pair of touching
 *   clusters, take the `Cluster Constraints` settings and are flagged with
 *   `is_cluster_constraint = true`. That is what makes two level fracture work: the chunk glue can
 *   be made weaker than the glue inside the chunks, so the chunks come apart first.
 *
 * The appended contacts have `cluster_a = cluster_b = -1` (they belong to no single cluster).
 *
 * The cluster of a fragment is the coarse grid site closest to its center, so it is deterministic
 * and stable across evaluations.
 */

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BKE_pointcloud.hh"

#include "BLI_hash.h"
#include "BLI_index_range.hh"
#include "BLI_math_color.h"
#include "BLI_math_vector.hh"
#include "BLI_string_utf8.h"
#include "BLI_vector.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_fracture_cluster_cc {

/** Deterministic random value in [0, 1) for a grid coordinate and seed. */
static float site_random(const int x, const int y, const int z, const int seed, const int salt)
{
  const uint32_t h = BLI_hash_int_3d(uint32_t(x + 4096),
                                     uint32_t(y + 4096),
                                     uint32_t(z + 4096) ^ (uint32_t(seed) * 2654435761u)) ^
                     (uint32_t(salt) * 2246822519u);
  return float(h) / float(0xFFFFFFFFu);
}

/** Jittered grid of cluster sites over the bounding box of the fragments. */
static Vector<float3> generate_cluster_sites(const float3 &bounds_min,
                                             const float3 &bounds_max,
                                             const float3 cluster_size,
                                             const int seed,
                                             const float jitter)
{
  Vector<float3> sites;
  const float3 size = bounds_max - bounds_min;
  const int3 res = int3(std::max(1, int(std::ceil(size.x / cluster_size.x))),
                        std::max(1, int(std::ceil(size.y / cluster_size.y))),
                        std::max(1, int(std::ceil(size.z / cluster_size.z))));
  for (const int z : IndexRange(res.z)) {
    for (const int y : IndexRange(res.y)) {
      for (const int x : IndexRange(res.x)) {
        const float3 base = bounds_min + (float3(x, y, z) + float3(0.5f)) * cluster_size;
        const float3 offset(site_random(x, y, z, seed, 0) - 0.5f,
                            site_random(x, y, z, seed, 1) - 0.5f,
                            site_random(x, y, z, seed, 2) - 0.5f);
        sites.append(base + offset * (jitter * cluster_size));
      }
    }
  }
  return sites;
}

/** The index of the site closest to the given point. */
static int closest_site(const float3 &point, const Span<float3> sites)
{
  int best = 0;
  float best_distance = FLT_MAX;
  for (const int i : sites.index_range()) {
    const float distance = math::distance_squared(point, sites[i]);
    if (distance < best_distance) {
      best_distance = distance;
      best = i;
    }
  }
  return best;
}

/** Copy one point attribute of `source` into `destination`, if it exists. */
template<typename T>
static void copy_point_attribute(const bke::AttributeAccessor &source,
                                 bke::MutableAttributeAccessor &destination,
                                 const StringRef name,
                                 const Span<int> selection)
{
  if (!source.contains(name)) {
    return;
  }
  const VArray<T> values = source.lookup_or_default<T>(name, AttrDomain::Point, T()).varray;
  bke::SpanAttributeWriter<T> writer = destination.lookup_or_add_for_write_span<T>(
      name, AttrDomain::Point);
  for (const int i : selection.index_range()) {
    writer.span[i] = values[selection[i]];
  }
  writer.finish();
}

enum class ClusterConstraintType : int {
  Fixed = 0,
  Soft = 1,
};

static const EnumPropertyItem cluster_constraint_type_items[] = {
    {int(ClusterConstraintType::Fixed), "FIXED", 0, N_("Fixed"), N_("Weld the clusters together")},
    {int(ClusterConstraintType::Soft),
     "SOFT",
     0,
     N_("Soft"),
     N_("Hold the clusters with a spring")},
    {0, nullptr, 0, nullptr, nullptr},
};

enum class ClusterConstraintFallback : int {
  Disable = 0,
  Soft = 1,
};

static const EnumPropertyItem cluster_constraint_fallback_items[] = {
    {int(ClusterConstraintFallback::Disable),
     "DISABLE",
     0,
     N_("None"),
     N_("Let the clusters go when their glue breaks")},
    {int(ClusterConstraintFallback::Soft),
     "SOFT",
     0,
     N_("Soft"),
     N_("Keep the clusters attached with a soft spring after their glue breaks")},
    {0, nullptr, 0, nullptr, nullptr},
};

/**
 * A stable, well spread colour for a cluster id, as a plain RGB vector (no alpha: it is only used to
 * tell the clusters apart, and a float3 is a quarter smaller than a colour).
 */
static float3 cluster_color(const int cluster)
{
  /* Golden ratio stepping keeps neighbouring ids far apart in hue. */
  const float hue = std::fmod(float(cluster) * 0.6180339887f + 0.13f, 1.0f);
  float red, green, blue;
  hsv_to_rgb(hue, 0.65f, 0.95f, &red, &green, &blue);
  return float3(red, green, blue);
}

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Pieces"_ustr)
      .description("Pieces as instances, e.g. from the Voronoi Fracture node");
  b.add_input<decl::Geometry>("Constraints"_ustr)
      .description(
          "The contacts between the pieces, e.g. from the RBD Constraint from Pieces node");
  b.add_input<decl::Vector>("Cluster Size"_ustr)
      .default_value(float3(1.5f))
      .min(0.001f)
      .subtype(PROP_DISTANCE)
      .description(
          "Approximate size of the clusters, per axis. Different values per axis give "
          "elongated chunks, e.g. for wood like fracture");
  b.add_input<decl::Int>("Seed"_ustr)
      .default_value(0)
      .description("Random seed for the cluster layout");
  b.add_input<decl::Bool>("Store Cluster Color"_ustr)
      .default_value(false)
      .description("Write a `cluster_color` attribute (one RGB colour per cluster, no alpha) on the "
                   "pieces, so the clusters are easy to tell apart in the viewport with a material "
                   "that reads it");
  b.add_input<decl::Float>("Jitter"_ustr)
      .default_value(0.5f)
      .min(0.0f)
      .max(1.0f)
      .description("Random offset of the cluster centers inside their grid cell");
  {
    auto &panel = b.add_panel("Cluster Constraints"_ustr).default_closed(true);
    panel.add_input<decl::Float>("Inter Cluster Tolerance"_ustr)
        .default_value(0.02f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Two clusters are connected when their bounds are at most this far apart");
    panel.add_input<decl::Menu>("Constraint Type"_ustr)
        .static_items(cluster_constraint_type_items)
        .default_value(MenuValue(int(ClusterConstraintType::Fixed)))
        .description("How neighbouring clusters are held together");
    panel.add_input<decl::Float>("Soft Frequency"_ustr)
        .default_value(5.0f)
        .min(0.0f)
        .description("Oscillation frequency of the cluster constraints in Hz");
    panel.add_input<decl::Float>("Soft Damping"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .max(1.0f)
        .description("Damping ratio of the cluster constraints");
    panel.add_input<decl::Menu>("Fallback"_ustr)
        .static_items(cluster_constraint_fallback_items)
        .default_value(MenuValue(int(ClusterConstraintFallback::Disable)))
        .description("What happens to a cluster constraint when it breaks");
    panel.add_input<decl::Float>("Harden Time"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_TIME)
        .description(
            "Re-baseline the rest pose of a cluster constraint after this many seconds of "
            "stable deformation (0 = off)");
    panel.add_input<decl::Float>("Break Impulse"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description(
            "Break a cluster constraint when the impulse it carried in a frame exceeds this "
            "in N s (0 = never). Set it lower than the piece glue so the clusters come apart "
            "first");
    panel.add_input<decl::Float>("Break Distance"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Break a cluster constraint when its anchors separate more than this");
    panel.add_input<decl::Float>("Break Angle"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_ANGLE)
        .description("Break a cluster constraint when the relative rotation exceeds this angle");
  }
  b.add_output<decl::Geometry>("Pieces"_ustr);
  b.add_output<decl::Geometry>("Constraints"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet fragments = params.extract_input<GeometrySet>("Pieces"_ustr);
  GeometrySet constraints = params.extract_input<GeometrySet>("Constraints"_ustr);

  const float3 cluster_size = math::max(params.extract_input<float3>("Cluster Size"_ustr),
                                        float3(1.0e-3f));
  const int seed = params.extract_input<int>("Seed"_ustr);
  const float jitter = math::clamp(params.extract_input<float>("Jitter"_ustr), 0.0f, 1.0f);
  const float inter_tolerance = std::max(
      params.extract_input<float>("Inter Cluster Tolerance"_ustr), 0.0f);
  const int cluster_type = params.extract_input<int>("Constraint Type"_ustr);
  const float cluster_frequency = std::max(params.extract_input<float>("Soft Frequency"_ustr),
                                           0.0f);
  const float cluster_damping = math::clamp(
      params.extract_input<float>("Soft Damping"_ustr), 0.0f, 1.0f);
  const int cluster_fallback = params.extract_input<int>("Fallback"_ustr);
  const float cluster_harden = std::max(params.extract_input<float>("Harden Time"_ustr), 0.0f);
  const float cluster_break_impulse = std::max(params.extract_input<float>("Break Impulse"_ustr),
                                               0.0f);
  const float cluster_break_distance = std::max(params.extract_input<float>("Break Distance"_ustr),
                                                0.0f);
  const float cluster_break_angle = std::max(params.extract_input<float>("Break Angle"_ustr),
                                             0.0f);

  bke::Instances *instances = fragments.has_instances() ? fragments.get_instances_for_write() :
                                                          nullptr;
  if (!instances || instances->instances_num() == 0) {
    params.set_output("Pieces"_ustr, std::move(fragments));
    params.set_output("Constraints"_ustr, std::move(constraints));
    return;
  }

  const int instances_num = instances->instances_num();
  const Span<float4x4> transforms = instances->transforms();
  const VArray<int> piece_ids =
      instances->attributes().lookup_or_default<int>("piece_id", AttrDomain::Instance, -1).varray;

  /* Bounding box of the fragment centers. */
  float3 bounds_min = transforms[0].location();
  float3 bounds_max = bounds_min;
  for (const int i : IndexRange(instances_num)) {
    const float3 location = transforms[i].location();
    bounds_min = math::min(bounds_min, location);
    bounds_max = math::max(bounds_max, location);
  }
  const float3 sites_size = math::max(bounds_max - bounds_min, float3(1.0e-4f));
  const Vector<float3> sites = generate_cluster_sites(
      bounds_min, bounds_min + sites_size, cluster_size, seed, jitter);

  /* Cluster per fragment, and a piece id -> cluster map for the constraints. */
  Array<int> cluster_of_instance(instances_num);
  Map<int, int> cluster_of_piece;
  bke::SpanAttributeWriter<int> cluster_ids =
      instances->attributes_for_write().lookup_or_add_for_write_span<int>("cluster_id",
                                                                          AttrDomain::Instance);
  const bool store_cluster_color = params.extract_input<bool>("Store Cluster Color"_ustr);
  bke::SpanAttributeWriter<float3> cluster_colors =
      store_cluster_color ? instances->attributes_for_write().lookup_or_add_for_write_span<float3>(
                                "cluster_color", AttrDomain::Instance) :
                            bke::SpanAttributeWriter<float3>();
  for (const int i : IndexRange(instances_num)) {
    const int cluster = closest_site(transforms[i].location(), sites);
    cluster_ids.span[i] = cluster;
    if (store_cluster_color) {
      cluster_colors.span[i] = cluster_color(cluster);
    }
    cluster_of_instance[i] = cluster;
    if (piece_ids[i] >= 0) {
      cluster_of_piece.add(piece_ids[i], cluster);
    }
  }
  cluster_ids.finish();
  if (store_cluster_color) {
    cluster_colors.finish();
  }

  if (!constraints.has_pointcloud()) {
    params.set_output("Pieces"_ustr, std::move(fragments));
    params.set_output("Constraints"_ustr, std::move(constraints));
    return;
  }
  const PointCloud *source = constraints.get_pointcloud();
  if (!source || source->totpoint == 0) {
    params.set_output("Pieces"_ustr, std::move(fragments));
    params.set_output("Constraints"_ustr, std::move(constraints));
    return;
  }

  const bke::AttributeAccessor source_attributes = source->attributes();
  const VArray<int> body_a =
      source_attributes.lookup_or_default<int>("body_a", AttrDomain::Point, -1).varray;
  const VArray<int> body_b =
      source_attributes.lookup_or_default<int>("body_b", AttrDomain::Point, -1).varray;

  Array<int> constraint_cluster_a(source->totpoint);
  Array<int> constraint_cluster_b(source->totpoint);
  Array<bool> inter_cluster(source->totpoint);
  Vector<int> selected;
  for (const int i : IndexRange(source->totpoint)) {
    const int *cluster_a = cluster_of_piece.lookup_ptr(body_a[i]);
    const int *cluster_b = cluster_of_piece.lookup_ptr(body_b[i]);
    constraint_cluster_a[i] = cluster_a ? *cluster_a : -1;
    constraint_cluster_b[i] = cluster_b ? *cluster_b : -1;
    inter_cluster[i] = (constraint_cluster_a[i] != constraint_cluster_b[i]);
    /* The piece level constraints are always kept: the cluster node only *adds* the constraints
     * between clusters on top of them. */
    selected.append(i);
  }

  /* Cluster level constraints: one per pair of clusters whose bounds touch. They use the cluster
   * settings, so the clusters can come apart before the glue inside them breaks. */
  struct ClusterLink {
    int piece_a;
    int piece_b;
    float3 center_a;
    float3 center_b;
  };
  Vector<ClusterLink> cluster_links;
  {
    Array<float3> cluster_min(sites.size(), float3(FLT_MAX, FLT_MAX, FLT_MAX));
    Array<float3> cluster_max(sites.size(), float3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
    for (const int i : IndexRange(instances_num)) {
      const int cluster = cluster_of_instance[i];
      const float3 location = transforms[i].location();
      cluster_min[cluster] = math::min(cluster_min[cluster], location);
      cluster_max[cluster] = math::max(cluster_max[cluster], location);
    }
    for (const int a : sites.index_range()) {
      if (cluster_min[a].x > cluster_max[a].x) {
        continue;
      }
      for (const int b : IndexRange(a + 1, sites.size() - a - 1)) {
        if (cluster_min[b].x > cluster_max[b].x) {
          continue;
        }
        /* Largest per axis gap between the two cluster bounds. */
        const float3 gap = math::max(cluster_min[a], cluster_min[b]) -
                           math::min(cluster_max[a], cluster_max[b]);
        if (gap.x > inter_tolerance || gap.y > inter_tolerance || gap.z > inter_tolerance) {
          continue;
        }
        /* Closest pair of pieces between the two clusters. */
        int best_a = -1;
        int best_b = -1;
        float best_distance = FLT_MAX;
        for (const int i : IndexRange(instances_num)) {
          if (cluster_of_instance[i] != a) {
            continue;
          }
          for (const int j : IndexRange(instances_num)) {
            if (cluster_of_instance[j] != b) {
              continue;
            }
            const float distance = math::distance_squared(transforms[i].location(),
                                                          transforms[j].location());
            if (distance < best_distance) {
              best_distance = distance;
              best_a = i;
              best_b = j;
            }
          }
        }
        if (best_a >= 0 && best_b >= 0 && piece_ids[best_a] >= 0 && piece_ids[best_b] >= 0) {
          cluster_links.append({piece_ids[best_a],
                                piece_ids[best_b],
                                transforms[best_a].location(),
                                transforms[best_b].location()});
        }
      }
    }
  }

  const int piece_constraint_num = int(selected.size());
  const int cluster_link_num = int(cluster_links.size());

  /* Rebuild the cloud: the piece level contacts are all kept, and the cluster links are appended.
   * Only the attributes of the rigid body pipeline are carried over; anything else the user added
   * would have to be re-added afterwards. */
  PointCloud *filtered = bke::pointcloud_new_no_attributes(piece_constraint_num +
                                                           cluster_link_num);
  MutableSpan<float3> positions = filtered->positions_for_write();
  for (const int i : selected.index_range()) {
    positions[i] = source->positions()[selected[i]];
  }
  for (const int k : cluster_links.index_range()) {
    const ClusterLink &link = cluster_links[k];
    positions[piece_constraint_num + k] = (link.center_a + link.center_b) * 0.5f;
  }
  bke::MutableAttributeAccessor destination_attributes = filtered->attributes_for_write();
  copy_point_attribute<int>(source_attributes, destination_attributes, "body_a", selected);
  copy_point_attribute<int>(source_attributes, destination_attributes, "body_b", selected);
  copy_point_attribute<float3>(source_attributes, destination_attributes, "anchor_a", selected);
  copy_point_attribute<float3>(source_attributes, destination_attributes, "anchor_b", selected);
  copy_point_attribute<bool>(source_attributes, destination_attributes, "enabled", selected);
  copy_point_attribute<int>(source_attributes, destination_attributes, "type", selected);
  copy_point_attribute<float>(source_attributes, destination_attributes, "frequency", selected);
  copy_point_attribute<float>(source_attributes, destination_attributes, "damping", selected);
  copy_point_attribute<int>(source_attributes, destination_attributes, "fallback", selected);
  copy_point_attribute<float>(source_attributes, destination_attributes, "harden_time", selected);
  copy_point_attribute<float>(
      source_attributes, destination_attributes, "break_impulse", selected);
  copy_point_attribute<float>(
      source_attributes, destination_attributes, "break_distance", selected);
  copy_point_attribute<float>(source_attributes, destination_attributes, "break_angle", selected);
  {
    bke::SpanAttributeWriter<int> cluster_a =
        destination_attributes.lookup_or_add_for_write_span<int>("cluster_a", AttrDomain::Point);
    bke::SpanAttributeWriter<int> cluster_b =
        destination_attributes.lookup_or_add_for_write_span<int>("cluster_b", AttrDomain::Point);
    bke::SpanAttributeWriter<bool> is_inter =
        destination_attributes.lookup_or_add_for_write_span<bool>("is_inter_cluster",
                                                                  AttrDomain::Point);
    bke::SpanAttributeWriter<bool> is_cluster_level =
        destination_attributes.lookup_or_add_for_write_span<bool>("is_cluster_constraint",
                                                                  AttrDomain::Point);
    bke::SpanAttributeWriter<int> body_a_out =
        destination_attributes.lookup_or_add_for_write_span<int>("body_a", AttrDomain::Point);
    bke::SpanAttributeWriter<int> body_b_out =
        destination_attributes.lookup_or_add_for_write_span<int>("body_b", AttrDomain::Point);
    bke::SpanAttributeWriter<float3> anchor_a_out =
        destination_attributes.lookup_or_add_for_write_span<float3>("anchor_a", AttrDomain::Point);
    bke::SpanAttributeWriter<float3> anchor_b_out =
        destination_attributes.lookup_or_add_for_write_span<float3>("anchor_b", AttrDomain::Point);
    bke::SpanAttributeWriter<bool> enabled_out =
        destination_attributes.lookup_or_add_for_write_span<bool>("enabled", AttrDomain::Point);
    bke::SpanAttributeWriter<int> type_out =
        destination_attributes.lookup_or_add_for_write_span<int>("type", AttrDomain::Point);
    bke::SpanAttributeWriter<float> frequency_out =
        destination_attributes.lookup_or_add_for_write_span<float>("frequency", AttrDomain::Point);
    bke::SpanAttributeWriter<float> damping_out =
        destination_attributes.lookup_or_add_for_write_span<float>("damping", AttrDomain::Point);
    bke::SpanAttributeWriter<int> fallback_out =
        destination_attributes.lookup_or_add_for_write_span<int>("fallback", AttrDomain::Point);
    bke::SpanAttributeWriter<float> harden_out =
        destination_attributes.lookup_or_add_for_write_span<float>("harden_time",
                                                                   AttrDomain::Point);
    bke::SpanAttributeWriter<float> break_impulse_out =
        destination_attributes.lookup_or_add_for_write_span<float>("break_impulse",
                                                                   AttrDomain::Point);
    bke::SpanAttributeWriter<float> break_distance_out =
        destination_attributes.lookup_or_add_for_write_span<float>("break_distance",
                                                                   AttrDomain::Point);
    bke::SpanAttributeWriter<float> break_angle_out =
        destination_attributes.lookup_or_add_for_write_span<float>("break_angle",
                                                                   AttrDomain::Point);
    for (const int i : selected.index_range()) {
      const int source_i = selected[i];
      cluster_a.span[i] = constraint_cluster_a[source_i];
      cluster_b.span[i] = constraint_cluster_b[source_i];
      is_inter.span[i] = inter_cluster[source_i];
      /* Contacts that cross a cluster boundary are the chunk glue: they take the cluster settings,
       * so the chunks come apart before the pieces inside them. */
      if (!inter_cluster[source_i]) {
        is_cluster_level.span[i] = false;
        continue;
      }
      type_out.span[i] = cluster_type;
      frequency_out.span[i] = cluster_frequency;
      damping_out.span[i] = cluster_damping;
      fallback_out.span[i] = cluster_fallback;
      harden_out.span[i] = cluster_harden;
      break_impulse_out.span[i] = cluster_break_impulse;
      break_distance_out.span[i] = cluster_break_distance;
      break_angle_out.span[i] = cluster_break_angle;
      is_cluster_level.span[i] = true;
    }
    for (const int k : cluster_links.index_range()) {
      const int i = piece_constraint_num + k;
      const ClusterLink &link = cluster_links[k];
      const float3 offset = (link.center_b - link.center_a) * 0.5f;
      body_a_out.span[i] = link.piece_a;
      body_b_out.span[i] = link.piece_b;
      anchor_a_out.span[i] = offset;
      anchor_b_out.span[i] = -offset;
      enabled_out.span[i] = true;
      type_out.span[i] = cluster_type;
      frequency_out.span[i] = cluster_frequency;
      damping_out.span[i] = cluster_damping;
      fallback_out.span[i] = cluster_fallback;
      harden_out.span[i] = cluster_harden;
      break_impulse_out.span[i] = cluster_break_impulse;
      break_distance_out.span[i] = cluster_break_distance;
      break_angle_out.span[i] = cluster_break_angle;
      cluster_a.span[i] = -1;
      cluster_b.span[i] = -1;
      is_inter.span[i] = true;
      is_cluster_level.span[i] = true;
    }
    cluster_a.finish();
    cluster_b.finish();
    is_inter.finish();
    is_cluster_level.finish();
    body_a_out.finish();
    body_b_out.finish();
    anchor_a_out.finish();
    anchor_b_out.finish();
    enabled_out.finish();
    type_out.finish();
    frequency_out.finish();
    damping_out.finish();
    fallback_out.finish();
    harden_out.finish();
    break_impulse_out.finish();
    break_distance_out.finish();
    break_angle_out.finish();
  }

  GeometrySet result;
  result.replace_pointcloud(filtered);
  params.set_output("Pieces"_ustr, std::move(fragments));
  params.set_output("Constraints"_ustr, std::move(result));
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeFractureCluster"_ustr);
  ntype.ui_name = "Fracture Cluster";
  ntype.ui_description = "Group fragments into clusters and constrain them per cluster";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_fracture_cluster_cc
