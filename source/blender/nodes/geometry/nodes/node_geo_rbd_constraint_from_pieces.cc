/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * RBD Constraint from Pieces node.
 *
 * Turns a set of pieces into constraints, so that the fracture nodes stay pure geometry operations.
 *
 * Two pieces are connected when their bounding boxes touch (within `Tolerance`), and the constraint
 * anchors are derived from the two piece *centers*: the anchor of a constraint is the midpoint
 * between the centers, expressed in the local space of each piece. Pieces are referenced by
 * `piece_id` (as everywhere else in the rigid body pipeline).
 *
 * Input either:
 * - `Pieces` as instances (e.g. the `Fragments` output of Voronoi Fracture), with `piece_id` on the
 *   instance domain, or
 * - `Pieces` as a mesh that carries a `piece_id` attribute (face and/or point domain), e.g. the
 *   `Geometry` output of Voronoi Fracture. Each piece's center and bounds are derived from its
 *   vertices.
 *
 * Output: a point cloud with one point per constraint, carrying `body_a` / `body_b` (the
 * `piece_id`s), `anchor_a` / `anchor_b` (in the local space of the two pieces) and `position`
 * (the midpoint, for viewing).
 */

#include <algorithm>
#include <cfloat>

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_instances.hh"
#include "BLI_math_matrix.hh"

#include "BKE_collection.hh"
#include "BKE_mesh.hh"
#include "BKE_object.hh"
#include "BKE_pointcloud.hh"

#include "BLI_array.hh"
#include "BLI_index_range.hh"
#include "BLI_listbase.h"
#include "BLI_map.hh"
#include "BLI_math_vector.hh"
#include "BLI_vector.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_rbd_constraint_from_pieces_cc {

/** Must match `InitRigidBody`: piece ids of different groups are spaced by this much. */
static constexpr int k_group_stride = 1000000;

/** Center and bounds of one piece, in world space. */
struct PieceInfo {
  int piece_id = -1;
  float3 center = float3(0.0f);
  float3 bounds_min = float3(0.0f);
  float3 bounds_max = float3(0.0f);
};

static void add_piece(PieceInfo &&info,
                      Map<int, int> &r_piece_of_id,
                      Vector<PieceInfo> &r_pieces)
{
  /* Keep the first occurrence of a piece id: an id that appears twice would otherwise be split
   * into two pieces. */
  if (r_piece_of_id.contains(info.piece_id)) {
    return;
  }
  r_piece_of_id.add(info.piece_id, int(r_pieces.size()));
  r_pieces.append(std::move(info));
}

/**
 * Bounds of one instance's content in world space.
 *
 * `InstanceReference::geometry_set()` may only be called for references that actually hold a
 * geometry set: for object or collection references (e.g. from a Collection Info node) it would
 * return a null pointer and crash. Object and collection references are handled through their
 * bound boxes instead, which is all this node needs (the constraint anchors are derived from the
 * piece centers).
 */
static bool instance_bounds(const bke::InstanceReference &reference,
                            const float4x4 &transform,
                            float3 &r_bounds_min,
                            float3 &r_bounds_max)
{
  switch (reference.type()) {
    case bke::InstanceReference::Type::GeometrySet: {
      const Mesh *mesh = reference.geometry_set().get_mesh();
      if (!mesh || mesh->verts_num == 0) {
        return false;
      }
      r_bounds_min = float3(FLT_MAX);
      r_bounds_max = float3(-FLT_MAX);
      for (const float3 &point : mesh->vert_positions()) {
        const float3 world = math::transform_point(transform, point);
        r_bounds_min = math::min(r_bounds_min, world);
        r_bounds_max = math::max(r_bounds_max, world);
      }
      return true;
    }
    case bke::InstanceReference::Type::Object: {
      const Object &object = reference.object();
      const std::optional<Bounds<float3>> bounds = BKE_object_boundbox_get(&object);
      if (!bounds) {
        return false;
      }
      r_bounds_min = math::transform_point(transform, bounds->min);
      r_bounds_max = math::transform_point(transform, bounds->max);
      return true;
    }
    case bke::InstanceReference::Type::Collection: {
      const Collection &collection = reference.collection();
      r_bounds_min = float3(FLT_MAX);
      r_bounds_max = float3(-FLT_MAX);
      bool found = false;
      for (const CollectionObject *collection_object =
               static_cast<const CollectionObject *>(collection.gobject.first);
           collection_object != nullptr;
           collection_object = collection_object->next)
      {
        const Object *object = collection_object->ob;
        if (!object) {
          continue;
        }
        const std::optional<Bounds<float3>> bounds = BKE_object_boundbox_get(object);
        if (!bounds) {
          continue;
        }
        /* The instance transform places the whole collection; the object's own transform is part
         * of its bound box (which is in object space). */
        const float4x4 world_matrix = transform * object->object_to_world();
        const float3 corner_a = math::transform_point(world_matrix, bounds->min);
        const float3 corner_b = math::transform_point(world_matrix, bounds->max);
        r_bounds_min = math::min(r_bounds_min, math::min(corner_a, corner_b));
        r_bounds_max = math::max(r_bounds_max, math::max(corner_a, corner_b));
        found = true;
      }
      return found;
    }
    case bke::InstanceReference::Type::None:
      break;
  }
  return false;
}

static void collect_pieces(const GeometrySet &geometry, Vector<PieceInfo> &r_pieces)
{
  Map<int, int> piece_of_id;
  if (geometry.has_instances()) {
    const bke::Instances *instances = geometry.get_instances();
    const Span<float4x4> transforms = instances->transforms();
    const Span<int> reference_handles = instances->reference_handles();
    const Span<bke::InstanceReference> references = instances->references();
    const VArray<int> piece_ids = instances->attributes()
                                      .lookup_or_default<int>("piece_id",
                                                              AttrDomain::Instance,
                                                              -1)
                                      .varray;
    for (const int i : IndexRange(instances->instances_num())) {
      const int reference_i = reference_handles[i];
      if (reference_i < 0 || reference_i >= references.size()) {
        continue;
      }
      PieceInfo info;
      info.piece_id = piece_ids[i] >= 0 ? piece_ids[i] : i;
      if (!instance_bounds(references[reference_i], transforms[i], info.bounds_min, info.bounds_max)) {
        continue;
      }
      info.center = (info.bounds_min + info.bounds_max) * 0.5f;
      add_piece(std::move(info), piece_of_id, r_pieces);
    }
    return;
  }

  const Mesh *mesh = geometry.get_mesh();
  if (!mesh) {
    return;
  }
  const bke::AttributeAccessor attributes = mesh->attributes();
  const VArray<int> point_piece_ids =
      attributes.lookup_or_default<int>("piece_id", AttrDomain::Point, -1).varray;
  if (attributes.contains("piece_id") && point_piece_ids.size() == mesh->verts_num) {
    /* A mesh with `piece_id`: group the vertices by piece. */
    for (const int i : IndexRange(mesh->verts_num)) {
      const int piece_id = point_piece_ids[i] >= 0 ? point_piece_ids[i] : 0;
      const float3 position = mesh->vert_positions()[i];
      if (piece_of_id.contains(piece_id)) {
        PieceInfo &info = r_pieces[piece_of_id.lookup(piece_id)];
        info.bounds_min = math::min(info.bounds_min, position);
        info.bounds_max = math::max(info.bounds_max, position);
        continue;
      }
      PieceInfo info;
      info.piece_id = piece_id;
      info.bounds_min = position;
      info.bounds_max = position;
      add_piece(std::move(info), piece_of_id, r_pieces);
    }
    for (PieceInfo &info : r_pieces) {
      info.center = (info.bounds_min + info.bounds_max) * 0.5f;
    }
    return;
  }

  /* No `piece_id`: treat the whole mesh as a single piece. */
  PieceInfo info;
  info.piece_id = 0;
  info.bounds_min = float3(FLT_MAX);
  info.bounds_max = float3(-FLT_MAX);
  for (const float3 &position : mesh->vert_positions()) {
    info.bounds_min = math::min(info.bounds_min, position);
    info.bounds_max = math::max(info.bounds_max, position);
  }
  info.center = (info.bounds_min + info.bounds_max) * 0.5f;
  add_piece(std::move(info), piece_of_id, r_pieces);
}

enum class ConstraintType : int {
  Fixed = 0,
  Soft = 1,
};

static const EnumPropertyItem constraint_type_items[] = {
    {int(ConstraintType::Fixed),
     "FIXED",
     0,
     N_("Fixed"),
     N_("Weld the pieces together, they keep their relative pose (a glue constraint)")},
    {int(ConstraintType::Soft),
     "SOFT",
     0,
     N_("Soft"),
     N_("Hold the pieces together with a spring, so they can stretch and swing")},
    {0, nullptr, 0, nullptr, nullptr},
};

/**
 * What a glue constraint becomes when it breaks. A soft fallback keeps the pieces attached with a
 * spring, so a tear looks like stretching instead of the pieces flying apart.
 */
enum class ConstraintFallback : int {
  Disable = 0,
  Soft = 1,
};

static const EnumPropertyItem constraint_fallback_items[] = {
    {int(ConstraintFallback::Disable),
     "DISABLE",
     0,
     N_("None"),
     N_("Let the pieces go when the glue breaks")},
    {int(ConstraintFallback::Soft),
     "SOFT",
     0,
     N_("Soft"),
     N_("Keep the pieces attached with a soft spring after the glue breaks")},
    {0, nullptr, 0, nullptr, nullptr},
};

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Pieces"_ustr)
      .description("Pieces as instances (with `piece_id`), or a mesh with a `piece_id` attribute");
  b.add_input<decl::Float>("Tolerance"_ustr)
      .default_value(0.02f)
      .min(0.0f)
      .subtype(PROP_DISTANCE)
      .description("Two pieces are connected when their bounding boxes are at most this far apart");
  b.add_input<decl::Bool>("Connect Diagonal"_ustr)
      .default_value(true)
      .description("Also connect pieces that only share an edge or a corner of their bounding box");
  {
    auto &panel = b.add_panel("Constraints"_ustr).default_closed(true);
    panel.add_input<decl::Menu>("Constraint Type"_ustr)
        .static_items(constraint_type_items)
        .default_value(MenuValue(int(ConstraintType::Fixed)))
        .description("How neighbouring pieces are held together");
    panel.add_input<decl::Float>("Soft Frequency"_ustr)
        .default_value(5.0f)
        .min(0.0f)
        .description("Oscillation frequency of soft constraints in Hz: higher is stiffer, "
                     "0 makes them hard. A frequency is used instead of a stiffness because it "
                     "does not depend on the mass of the pieces");
    panel.add_input<decl::Float>("Soft Damping"_ustr)
        .default_value(0.5f)
        .min(0.0f)
        .max(1.0f)
        .description("Damping ratio of soft constraints (0 = undamped, 1 = critically damped)");
    panel.add_input<decl::Menu>("Fallback"_ustr)
        .static_items(constraint_fallback_items)
        .default_value(MenuValue(int(ConstraintFallback::Disable)))
        .description("What happens to a glue constraint when it breaks");
    panel.add_input<decl::Float>("Harden Time"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_TIME)
        .description(
            "When > 0, a constraint that stays deformed (but does not break) for this many seconds "
            "keeps its deformation: the current pose becomes the new rest pose. 0 disables it");
    panel.add_input<decl::Float>("Break Impulse"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .description(
            "Break a glue constraint when the impulse it carried in a frame exceeds this value "
            "in N s (0 = never)");
    panel.add_input<decl::Float>("Break Distance"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_DISTANCE)
        .description("Break a glue constraint when the two anchor points separate more than this "
                     "(0 = never)");
    panel.add_input<decl::Float>("Break Angle"_ustr)
        .default_value(0.0f)
        .min(0.0f)
        .subtype(PROP_ANGLE)
        .description("Break a glue constraint when the relative rotation of the two pieces exceeds "
                     "this angle (0 = never)");
  }
  b.add_output<decl::Geometry>("Constraints"_ustr);
}

static void node_geo_exec(GeoNodeExecParams params)
{
  const GeometrySet geometry = params.extract_input<GeometrySet>("Pieces"_ustr);
  const float tolerance = std::max(params.extract_input<float>("Tolerance"_ustr), 0.0f);
  const bool connect_diagonal = params.extract_input<bool>("Connect Diagonal"_ustr);
  const int constraint_type = params.extract_input<int>("Constraint Type"_ustr);
  const float frequency = std::max(params.extract_input<float>("Soft Frequency"_ustr), 0.0f);
  const float damping = math::clamp(params.extract_input<float>("Soft Damping"_ustr), 0.0f, 1.0f);
  const int fallback = params.extract_input<int>("Fallback"_ustr);
  const float harden_time = std::max(params.extract_input<float>("Harden Time"_ustr), 0.0f);
  const float break_impulse = std::max(params.extract_input<float>("Break Impulse"_ustr), 0.0f);
  const float break_distance = std::max(params.extract_input<float>("Break Distance"_ustr), 0.0f);
  const float break_angle = std::max(params.extract_input<float>("Break Angle"_ustr), 0.0f);

  Vector<PieceInfo> pieces;
  collect_pieces(geometry, pieces);
  if (pieces.size() < 2) {
    params.set_default_remaining_outputs();
    return;
  }

  struct Contact {
    /** Indices into `pieces`, not `piece_id`s: ids can be sparse or carry a `Group ID` offset. */
    int index_a;
    int index_b;
    float3 center_a;
    float3 center_b;
  };
  Vector<Contact> contacts;
  for (const int i : IndexRange(pieces.size())) {
    for (const int j : IndexRange::from_begin_end(i + 1, pieces.size())) {
      const PieceInfo &a = pieces[i];
      const PieceInfo &b = pieces[j];
      const bool overlap_x = a.bounds_min.x <= b.bounds_max.x + tolerance &&
                             b.bounds_min.x <= a.bounds_max.x + tolerance;
      const bool overlap_y = a.bounds_min.y <= b.bounds_max.y + tolerance &&
                             b.bounds_min.y <= a.bounds_max.y + tolerance;
      const bool overlap_z = a.bounds_min.z <= b.bounds_max.z + tolerance &&
                             b.bounds_min.z <= a.bounds_max.z + tolerance;
      if (!overlap_x || !overlap_y || !overlap_z) {
        continue;
      }
      if (!connect_diagonal) {
        /* Require a real overlap on at least one axis (a shared face). */
        const bool touches_x = std::min(a.bounds_max.x, b.bounds_max.x) -
                                   std::max(a.bounds_min.x, b.bounds_min.x) >
                               -tolerance;
        const bool touches_y = std::min(a.bounds_max.y, b.bounds_max.y) -
                                   std::max(a.bounds_min.y, b.bounds_min.y) >
                               -tolerance;
        const bool touches_z = std::min(a.bounds_max.z, b.bounds_max.z) -
                                   std::max(a.bounds_min.z, b.bounds_min.z) >
                               -tolerance;
        if (int(touches_x) + int(touches_y) + int(touches_z) < 2) {
          continue;
        }
      }
      contacts.append({i, j, a.center, b.center});
    }
  }

  if (contacts.is_empty()) {
    params.set_default_remaining_outputs();
    return;
  }

  PointCloud *pointcloud = bke::pointcloud_new_no_attributes(int(contacts.size()));
  bke::MutableAttributeAccessor attributes = pointcloud->attributes_for_write();
  bke::SpanAttributeWriter<float3> positions =
      attributes.lookup_or_add_for_write_span<float3>("position", AttrDomain::Point);
  bke::SpanAttributeWriter<int> body_a =
      attributes.lookup_or_add_for_write_span<int>("body_a", AttrDomain::Point);
  bke::SpanAttributeWriter<int> body_b =
      attributes.lookup_or_add_for_write_span<int>("body_b", AttrDomain::Point);
  bke::SpanAttributeWriter<float3> anchor_a =
      attributes.lookup_or_add_for_write_span<float3>("anchor_a", AttrDomain::Point);
  bke::SpanAttributeWriter<float3> anchor_b =
      attributes.lookup_or_add_for_write_span<float3>("anchor_b", AttrDomain::Point);
  bke::SpanAttributeWriter<bool> enabled =
      attributes.lookup_or_add_for_write_span<bool>("enabled", AttrDomain::Point);
  bke::SpanAttributeWriter<int> types =
      attributes.lookup_or_add_for_write_span<int>("type", AttrDomain::Point);
  bke::SpanAttributeWriter<float> frequencies =
      attributes.lookup_or_add_for_write_span<float>("frequency", AttrDomain::Point);
  bke::SpanAttributeWriter<float> dampings =
      attributes.lookup_or_add_for_write_span<float>("damping", AttrDomain::Point);
  bke::SpanAttributeWriter<int> fallbacks =
      attributes.lookup_or_add_for_write_span<int>("fallback", AttrDomain::Point);
  bke::SpanAttributeWriter<float> harden_times =
      attributes.lookup_or_add_for_write_span<float>("harden_time", AttrDomain::Point);
  bke::SpanAttributeWriter<float> break_impulses =
      attributes.lookup_or_add_for_write_span<float>("break_impulse", AttrDomain::Point);
  bke::SpanAttributeWriter<float> break_distances =
      attributes.lookup_or_add_for_write_span<float>("break_distance", AttrDomain::Point);
  bke::SpanAttributeWriter<float> break_angles =
      attributes.lookup_or_add_for_write_span<float>("break_angle", AttrDomain::Point);

  for (const int i : contacts.index_range()) {
    const Contact &contact = contacts[i];
    /* The constraint holds the midpoint between the two centers, expressed in the local space of
     * each piece (whose origin is its center). */
    const float3 offset = (contact.center_b - contact.center_a) * 0.5f;
    positions.span[i] = (contact.center_a + contact.center_b) * 0.5f;
    /* `body_a` / `body_b` reference `piece_id`, so they have to move with the offset as well,
     * otherwise the constraints would point at the wrong pieces after joining several sets. */
    /* Use the actual `piece_id`s of the pieces: they already carry the group offset from
     * Init Rigid Body, so the user never has to repeat anything here. Note that the ids are not
     * indices: with a `Group ID` they start at a million, and they can be sparse in general, so
     * looking a piece up by its id would read far out of bounds. */
    body_a.span[i] = pieces[contact.index_a].piece_id;
    body_b.span[i] = pieces[contact.index_b].piece_id;
    anchor_a.span[i] = offset;
    anchor_b.span[i] = -offset;
    enabled.span[i] = true;
    types.span[i] = constraint_type;
    frequencies.span[i] = frequency;
    dampings.span[i] = damping;
    fallbacks.span[i] = fallback;
    harden_times.span[i] = harden_time;
    break_impulses.span[i] = break_impulse;
    break_distances.span[i] = break_distance;
    break_angles.span[i] = break_angle;
  }

  positions.finish();
  body_a.finish();
  body_b.finish();
  anchor_a.finish();
  anchor_b.finish();
  enabled.finish();
  types.finish();
  frequencies.finish();
  dampings.finish();
  fallbacks.finish();
  harden_times.finish();
  break_impulses.finish();
  break_distances.finish();
  break_angles.finish();

  GeometrySet result;
  result.replace_pointcloud(pointcloud);
  params.set_output("Constraints"_ustr, std::move(result));
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeRBDConstraintFromPieces"_ustr);
  ntype.ui_name = "RBD Constraint from Pieces";
  ntype.ui_description = "Connect pieces of a fracture with constraints, using their centers";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_200;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_rbd_constraint_from_pieces_cc