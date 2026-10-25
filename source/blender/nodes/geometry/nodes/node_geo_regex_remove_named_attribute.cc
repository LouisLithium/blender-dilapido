/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup geo_node_type
 *
 * Remove attributes whose name matches one of several regular expressions. The patterns are
 * separated by commas, so a single node can clean up a whole naming scheme.
 */

#include <regex>
#include <string>

#include "BKE_attribute.hh"
#include "BKE_instances.hh"

#include "BLI_string_ref.hh"
#include "BLI_vector.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_regex_remove_named_attribute_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Geometry"_ustr).description("Geometry to remove attributes from");
  b.add_input<decl::String>("Pattern"_ustr)
      .description(
          "Regular expressions (ECMAScript syntax) matched against the attribute names, separated "
          "by commas. An attribute is removed when any of them matches anywhere in its name");
  b.add_output<decl::Geometry>("Geometry"_ustr).propagate_all_geometry();
}

/** Split the comma separated pattern list and strip the surrounding whitespace of each entry. */
static void split_patterns(const std::string &text, Vector<std::string> &r_patterns)
{
  size_t start = 0;
  while (start <= text.size()) {
    size_t end = text.find(',', start);
    if (end == std::string::npos) {
      end = text.size();
    }
    const std::string piece = text.substr(start, end - start);
    const size_t first = piece.find_first_not_of(" \t");
    if (first != std::string::npos) {
      const size_t last = piece.find_last_not_of(" \t");
      r_patterns.append(piece.substr(first, last - first + 1));
    }
    start = end + 1;
  }
}

static bool matches_any(const std::string &name, const Vector<std::regex> &patterns)
{
  for (const std::regex &pattern : patterns) {
    if (std::regex_search(name, pattern)) {
      return true;
    }
  }
  return false;
}

struct RemovalState {
  Set<std::string> removed;
  Set<std::string> failed;
};

static void remove_matching_recursive(GeometrySet &geometry_set,
                                      const Vector<std::regex> &patterns,
                                      RemovalState &state)
{
  for (const GeometryComponent::Type type : {GeometryComponent::Type::Mesh,
                                             GeometryComponent::Type::PointCloud,
                                             GeometryComponent::Type::Curve,
                                             GeometryComponent::Type::Instance,
                                             GeometryComponent::Type::GreasePencil})
  {
    if (!geometry_set.has(type)) {
      continue;
    }
    /* Only look at the attribute list of the read-only component first: getting the writable
     * component may make a copy. */
    const GeometryComponent &read_only_component = *geometry_set.get_component(type);
    Vector<std::string> attributes_to_remove;
    read_only_component.attributes()->foreach_attribute([&](const bke::AttributeIter &iter) {
      if (bke::attribute_name_is_anonymous(iter.name)) {
        return;
      }
      const std::string name(iter.name);
      if (matches_any(name, patterns)) {
        attributes_to_remove.append(name);
      }
    });

    if (attributes_to_remove.is_empty()) {
      continue;
    }

    GeometryComponent &component = geometry_set.get_component_for_write(type);
    for (const StringRef attribute_name : attributes_to_remove) {
      if (!bke::allow_procedural_attribute_access(attribute_name)) {
        continue;
      }
      if (component.attributes_for_write()->remove(attribute_name)) {
        state.removed.add(attribute_name);
      }
      else {
        state.failed.add(attribute_name);
      }
    }
  }

  if (bke::Instances *instances = geometry_set.get_instances_for_write()) {
    instances->ensure_geometry_instances();
    for (bke::InstanceReference &reference : instances->references_for_write()) {
      if (reference.type() == bke::InstanceReference::Type::GeometrySet) {
        remove_matching_recursive(reference.geometry_set(), patterns, state);
      }
    }
  }
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.extract_input<GeometrySet>("Geometry"_ustr);
  const std::string pattern_text = params.extract_input<std::string>("Pattern"_ustr);

  Vector<std::string> raw_patterns;
  split_patterns(pattern_text, raw_patterns);

  Vector<std::regex> patterns;
  for (const std::string &raw : raw_patterns) {
    try {
      patterns.append(std::regex(raw, std::regex::ECMAScript));
    }
    catch (const std::regex_error &error) {
      params.error_message_add(
          NodeWarningType::Error,
          "Invalid regular expression \"" + raw + "\": " + std::string(error.what()));
    }
  }

  if (patterns.is_empty()) {
    if (!pattern_text.empty() && raw_patterns.is_empty()) {
      params.error_message_add(NodeWarningType::Warning, "No pattern given, nothing removed");
    }
    params.set_output("Geometry"_ustr, std::move(geometry_set));
    return;
  }

  RemovalState state;
  remove_matching_recursive(geometry_set, patterns, state);

  for (const StringRef attribute_name : state.removed) {
    params.used_named_attribute(attribute_name, NamedAttributeUsage::Remove);
  }

  if (!state.failed.is_empty()) {
    std::string names;
    for (const StringRef attribute_name : state.failed) {
      if (!names.empty()) {
        names += ", ";
      }
      names += "\"" + attribute_name + "\"";
    }
    params.error_message_add(NodeWarningType::Warning,
                             "Cannot remove built-in attributes: " + names);
  }

  if (state.removed.is_empty() && state.failed.is_empty()) {
    params.error_message_add(NodeWarningType::Warning,
                             "No attribute matched the given pattern");
  }

  params.set_output("Geometry"_ustr, std::move(geometry_set));
}

static void node_register()
{
  static bke::bNodeType ntype;

  geo_node_type_base(&ntype, "GeometryNodeRegexRemoveNamedAttribute"_ustr);
  ntype.ui_name = "Regex Remove Named Attribute";
  ntype.ui_description =
      "Delete the attributes whose name matches one of the given regular expressions. The patterns "
      "are separated by commas";
  ntype.nclass = NODE_CLASS_ATTRIBUTE;
  ntype.declare = node_declare;
  ntype.default_width = bke::NodeWidth::_180;
  ntype.geometry_node_execute = node_geo_exec;
  bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_regex_remove_named_attribute_cc
