/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup nodes
 *
 * `Merge Ragdoll Bones`: concatenate bone point clouds while keeping *all* attributes.
 *
 * `Join Geometry` drops string attributes (its realize/join path does not support them yet), and
 * `bone_name` is a string. A crowd of characters is one shared bone cloud, so the merge has to keep
 * the names that `Skin From Bones` matches its vertex groups against. This node only handles point
 * clouds, which is all the ragdoll pipeline needs.
 */

#include <optional>
#include <string>

#include "BLI_array.hh"
#include "BLI_math_vector_types.hh"
#include "BLI_string_ref.hh"
#include "BLI_vector.hh"
#include "BLI_vector_set.hh"

#include "BKE_attribute.hh"
#include "BKE_geometry_set.hh"
#include "BKE_pointcloud.hh"

#include "DNA_meshdata_types.h"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_ragdoll_bones_merge_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("A"_ustr).description("First bone point cloud");
  b.add_input<decl::Geometry>("B"_ustr).description("Second bone point cloud");
  b.add_output<decl::Geometry>("Bones"_ustr);
}

static void collect_attribute_names(const PointCloud &src, VectorSet<std::string> &r_names)
{
  bke::AttributeAccessor src_attributes = src.attributes();
  src_attributes.foreach_attribute([&](const bke::AttributeIter &iter) {
    if (iter.domain != AttrDomain::Point || iter.name == "position") {
      return;
    }
    r_names.add(std::string(iter.name));
  });
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet a_geometry = params.extract_input<GeometrySet>("A"_ustr);
  GeometrySet b_geometry = params.extract_input<GeometrySet>("B"_ustr);
  const PointCloud *a = a_geometry.get_pointcloud();
  const PointCloud *b = b_geometry.get_pointcloud();

  if (a == nullptr && b == nullptr) {
    params.set_output("Bones"_ustr, GeometrySet::from_pointcloud(BKE_pointcloud_new_nomain(0)));
    return;
  }
  if (a == nullptr) {
    params.set_output("Bones"_ustr, b_geometry);
    return;
  }
  if (b == nullptr) {
    params.set_output("Bones"_ustr, a_geometry);
    return;
  }
  if (a->totpoint == 0) {
    params.set_output("Bones"_ustr, b_geometry);
    return;
  }
  if (b->totpoint == 0) {
    params.set_output("Bones"_ustr, a_geometry);
    return;
  }

  const int count_a = a->totpoint;
  const int count_b = b->totpoint;
  PointCloud *result = BKE_pointcloud_new_nomain(count_a + count_b);
  bke::MutableAttributeAccessor dst_attributes = result->attributes_for_write();

  bke::SpanAttributeWriter<float3> positions =
      dst_attributes.lookup_or_add_for_write_span<float3>("position", AttrDomain::Point);
  positions.span.slice(0, count_a).copy_from(a->positions());
  positions.span.slice(count_a, count_b).copy_from(b->positions());
  positions.finish();

  VectorSet<std::string> names;
  collect_attribute_names(*a, names);
  collect_attribute_names(*b, names);

  for (const std::string &name : names) {
    const std::optional<bke::AttributeMetaData> meta_a = a->attributes().lookup_meta_data(name);
    const std::optional<bke::AttributeMetaData> meta_b = b->attributes().lookup_meta_data(name);
    const bke::AttributeMetaData *meta = meta_a ? &*meta_a : &*meta_b;
    if (meta->domain != AttrDomain::Point) {
      continue;
    }
    if (meta_a && meta_a->domain != AttrDomain::Point) {
      continue;
    }
    if (meta_b && meta_b->domain != AttrDomain::Point) {
      continue;
    }
    if (meta_a && meta_b && meta_a->data_type != meta_b->data_type) {
      continue;
    }
    const bke::AttrType type = meta->data_type;

    if (type == bke::AttrType::String) {
      bke::SpanAttributeWriter<MStringProperty> writer =
          dst_attributes.lookup_or_add_for_write_span<MStringProperty>(name, AttrDomain::Point);
      if (!writer) {
        continue;
      }
      if (meta_a) {
        const VArray<MStringProperty> src = a->attributes()
                                                .lookup<MStringProperty>(name, AttrDomain::Point)
                                                .varray;
        if (src) {
          for (const int i : IndexRange(count_a)) {
            writer.span[i] = src[i];
          }
        }
      }
      if (meta_b) {
        const VArray<MStringProperty> src = b->attributes()
                                                .lookup<MStringProperty>(name, AttrDomain::Point)
                                                .varray;
        if (src) {
          for (const int i : IndexRange(count_b)) {
            writer.span[count_a + i] = src[i];
          }
        }
      }
      writer.finish();
      continue;
    }

    bke::GSpanAttributeWriter writer = dst_attributes.lookup_or_add_for_write_span(
        name, AttrDomain::Point, type);
    if (!writer) {
      continue;
    }
    const CPPType &cpp_type = bke::attribute_type_to_cpp_type(type);
    if (meta_a) {
      const GVArray src = *a->attributes().lookup(name, AttrDomain::Point, type);
      if (src) {
        GVArraySpan src_span{src};
        cpp_type.copy_construct_n(src_span.data(), writer.span[0], count_a);
      }
    }
    if (meta_b) {
      const GVArray src = *b->attributes().lookup(name, AttrDomain::Point, type);
      if (src) {
        GVArraySpan src_span{src};
        cpp_type.copy_construct_n(src_span.data(), writer.span[count_a], count_b);
      }
    }
    writer.finish();
  }

  result->tag_positions_changed();
  params.set_output("Bones"_ustr, GeometrySet::from_pointcloud(result));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeRagdollBonesMerge"_ustr);
  ntype.ui_name = "Merge Ragdoll Bones";
  ntype.ui_description =
      "Concatenate two bone point clouds, keeping every attribute (including `bone_name`)";
  ntype.nclass = NODE_CLASS_GEOMETRY;
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.default_width = bke::NodeWidth::_240;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_ragdoll_bones_merge_cc
