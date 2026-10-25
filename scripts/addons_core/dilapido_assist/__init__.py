# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Dilapido Assist: helpers for the Jolt rigid body geometry nodes.

The panel lives in the sidebar of the Geometry Nodes editor and offers a single button, `Quick RBD`,
that builds a working rigid body setup in the current node tree:

    Group Input -> Voronoi Fracture -> Fracture Cluster -> Init Rigid Body -> Rigid Body Dynamics
                                    \-> RBD Constraint from Pieces -> (constraints of the cluster)

Everything is wired, so a modifier that has this node tree simulates right away; the sizes are
sensible defaults that can be tweaked afterwards.
"""

bl_info = {
    "name": "Dilapido Assist",
    "author": "Dilapido",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "Geometry Nodes editor > Sidebar > Dilapido",
    "description": "One click rigid body setups with the Jolt solver",
    "category": "Node",
}

import os

import bpy
from bpy.types import Operator, Panel

#: Name of the asset node group that wraps the simulation zone.
ASSET_GROUP_NAME = "Rigid Body Dynamics"

#: The asset library file that ships the groups (relative to `datafiles`).
ASSET_FILE = "assets/nodes/geometry_nodes_rigid_body_assets.blend"


def ensure_asset_group(name=ASSET_GROUP_NAME):
    """Return one of the shipped asset groups, appending it from the library when needed."""
    group = bpy.data.node_groups.get(name)
    if group is not None:
        return group

    datafiles = bpy.utils.system_resource("DATAFILES")
    if not datafiles:
        return None
    path = os.path.join(datafiles, *ASSET_FILE.split("/"))
    if not os.path.exists(path):
        return None
    try:
        with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
            if name in data_from.node_groups:
                data_to.node_groups = [name]
    except (OSError, RuntimeError):
        return None
    return bpy.data.node_groups.get(name)


def _socket(node, name, kind=None):
    """Find a socket by name (and optionally its socket type), for robustness."""
    for socket in list(node.inputs) + list(node.outputs):
        if socket.name != name:
            continue
        if kind is not None and socket.bl_idname != kind:
            continue
        return socket
    return None


def _input(node, name):
    for socket in node.inputs:
        if socket.name == name:
            return socket
    return None


def _output(node, name):
    for socket in node.outputs:
        if socket.name == name:
            return socket
    return None


def build_quick_rbd(node_tree, spacing=260.0):
    """Build a basic rigid body setup in `node_tree` and return the created nodes.

    The tree has to be a geometry node tree with a `Group Input` and a `Group Output` node (both are
    created when missing). Returns a dict with the created nodes, or None when the asset group is
    missing.
    """
    asset = ensure_asset_group()
    if asset is None:
        return None

    nodes = node_tree.nodes
    links = node_tree.links

    def find_or_add(bl_idname, label):
        for node in nodes:
            if node.bl_idname == bl_idname:
                return node
        node = nodes.new(bl_idname)
        node.label = label
        return node

    group_input = find_or_add("NodeGroupInput", "Input")
    group_output = find_or_add("NodeGroupOutput", "Output")
    if not any(item.item_type == "SOCKET" and item.in_out == "INPUT" for item in
               node_tree.interface.items_tree):
        node_tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    if not any(item.item_type == "SOCKET" and item.in_out == "OUTPUT" for item in
               node_tree.interface.items_tree):
        node_tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    fracture = nodes.new("GeometryNodeVoronoiFracture")
    fracture.label = "Fracture"
    fracture.inputs["Cell Size"].default_value = (0.5, 0.5, 0.5)
    fracture.inputs["Jitter"].default_value = (0.3, 0.3, 0.3)

    cluster = nodes.new("GeometryNodeFractureCluster")
    cluster.label = "Cluster"
    cluster.inputs["Cluster Size"].default_value = (1.0, 1.0, 1.0)
    cluster.inputs["Jitter"].default_value = 0.3

    constraints = nodes.new("GeometryNodeRBDConstraintFromPieces")
    constraints.label = "Constraints"
    constraints.inputs["Tolerance"].default_value = 0.02

    init = nodes.new("GeometryNodeInitRigidBody")
    init.label = "Rigid Body"

    simulation = nodes.new("GeometryNodeGroup")
    simulation.label = "Dynamics"
    simulation.node_tree = asset

    # Lay the nodes out left to right, two rows (the constraints below the pieces).
    group_input.location = (-400.0, 0.0)
    fracture.location = (-160.0, 120.0)
    cluster.location = (80.0, 120.0)
    constraints.location = (80.0, -180.0)
    init.location = (340.0, 120.0)
    simulation.location = (600.0, 120.0)
    group_output.location = (880.0, 120.0)
    for node in (fracture, cluster, constraints, init, simulation):
        node.width = 200.0

    def wire(from_node, from_socket, to_node, to_socket):
        source = _output(from_node, from_socket)
        target = _input(to_node, to_socket)
        if source is None or target is None:
            return False
        links.new(source, target)
        return True

    wire(group_input, "Geometry", fracture, "Geometry")
    wire(fracture, "Pieces", cluster, "Pieces")
    wire(fracture, "Pieces", constraints, "Pieces")
    wire(constraints, "Constraints", cluster, "Constraints")
    wire(cluster, "Pieces", init, "Pieces")
    wire(init, "Pieces", simulation, "Pieces")
    wire(cluster, "Constraints", simulation, "Constraints")
    wire(simulation, "Geometry", group_output, "Geometry")

    return {
        "fracture": fracture,
        "cluster": cluster,
        "constraints": constraints,
        "init": init,
        "simulation": simulation,
    }


class DILAPIDO_OT_quick_rbd(Operator):
    """Build a rigid body setup (fracture, clusters, constraints, simulation) in this node tree"""

    bl_idname = "dilapido.quick_rbd"
    bl_label = "Quick RBD"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return (space is not None and space.type == "NODE_EDITOR" and
                space.tree_type == "GeometryNodeTree" and space.edit_tree is not None)

    def execute(self, context):
        node_tree = context.space_data.edit_tree
        created = build_quick_rbd(node_tree)
        if created is None:
            self.report({"ERROR"},
                        "The asset node group 'Rigid Body Dynamics' could not be loaded")
            return {"CANCELLED"}
        self.report({"INFO"}, "Rigid body setup created")
        return {"FINISHED"}


class DILAPIDO_PT_assist(Panel):
    """Rigid body helpers"""

    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Dilapido"
    bl_label = "Dilapido Assist"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space is not None and space.type == "NODE_EDITOR" and \
            space.tree_type == "GeometryNodeTree"

    def draw(self, context):
        layout = self.layout
        column = layout.column(align=True)
        column.operator("dilapido.quick_rbd", icon="NODETREE")
        layout.label(text="Fracture, cluster, constraints and the solver.", icon="INFO")


classes = (
    DILAPIDO_OT_quick_rbd,
    DILAPIDO_PT_assist,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
