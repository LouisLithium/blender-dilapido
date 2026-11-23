# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Dilapido Assist: helpers for the Jolt rigid body and crowd geometry nodes.

The panel lives in the sidebar of the Geometry Nodes editor and offers two buttons:

- `Quick RBD` builds a working rigid body setup in the current node tree:
  `Group Input -> Voronoi Fracture -> Fracture Cluster -> Init Rigid Body -> Rigid Body Dynamics`.
- `Quick Crowd Walk` builds a walking crowd template: three states, spawn scattering,
  `Crowd Motion` with local separation, `Crowd Bones` and skinning. It uses the native nodes
  directly (no asset group).

Everything is wired, so a modifier that has this node tree runs right away; the sizes are sensible
defaults that can be tweaked afterwards.
"""

bl_info = {
    "name": "Dilapido Assist",
    "author": "Dilapido",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "Geometry Nodes editor > Sidebar > Dilapido",
    "description": "One click rigid body and crowd setups with the Jolt solver",
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


def _ensure_interface_socket(node_tree, name, in_out, socket_type, description=""):
    for item in node_tree.interface.items_tree:
        if item.item_type == "SOCKET" and item.in_out == in_out and item.name == name:
            return
    socket = node_tree.interface.new_socket(name, in_out=in_out, socket_type=socket_type)
    if description:
        socket.description = description


def build_quick_crowd_walk(node_tree):
    """Build the fixed `Crowd Walk` template tree (no asset group, no rigid bodies)."""
    nodes = node_tree.nodes
    links = node_tree.links

    def find_or_add(bl_idname, label):
        for node in nodes:
            if node.bl_idname == bl_idname:
                return node
        node = nodes.new(bl_idname)
        node.label = label
        return node

    group_input = find_or_add("NodeGroupInput", "Crowd Inputs")
    group_output = find_or_add("NodeGroupOutput", "Crowd Output")
    _ensure_interface_socket(
        node_tree, "Mesh", "INPUT", "NodeSocketGeometry",
        "Character mesh. Its vertex groups must be named after the bones")
    _ensure_interface_socket(node_tree, "Armature", "INPUT", "NodeSocketObject",
                             "Rest rig used for the skeleton and Action evaluation")
    _ensure_interface_socket(node_tree, "Spawn Surface", "INPUT", "NodeSocketGeometry",
                             "Surface to scatter the crowd spawn points on")
    _ensure_interface_socket(node_tree, "Target", "INPUT", "NodeSocketGeometry",
                             "Target point/geometry the crowd walks towards")
    _ensure_interface_socket(node_tree, "Obstacles", "INPUT", "NodeSocketGeometry",
                             "Optional obstacle mesh. Leave unconnected for open walking")
    _ensure_interface_socket(node_tree, "Geometry", "OUTPUT", "NodeSocketGeometry")

    # ---------------------------------------------------------------- states
    state_nodes = []
    for name, move_speed, label in (("Idle", 0.0, "Idle"),
                                    ("Walk", 1.4, "Walk"),
                                    ("Run", 4.0, "Run")):
        animation = nodes.new("GeometryNodeCrowdAnimationInfo")
        animation.label = f"{label} Animation"
        animation.inputs["Action"].default_value = label
        animation.inputs["FPS"].default_value = 24.0
        animation.inputs["Loop"].default_value = True

        state = nodes.new("GeometryNodeCrowdState")
        state.label = f"{label} State"
        state.inputs["Name"].default_value = name
        state.inputs["Move Speed"].default_value = move_speed
        state.inputs["Blend In"].default_value = 0.25
        state.inputs["Blend Out"].default_value = 0.25
        links.new(animation.outputs["Animation"], state.inputs["Animation"])
        state_nodes.append((state, animation))

    join_states = nodes.new("GeometryNodeJoinCrowdStates")
    join_states.label = "States"
    for state, _animation in state_nodes:
        links.new(state.outputs["State"], join_states.inputs["State"])

    # ---------------------------------------------------------------- template
    armature_bones = nodes.new("GeometryNodeArmatureBones")
    armature_bones.label = "Rest Bones"
    links.new(group_input.outputs["Armature"], armature_bones.inputs["Armature"])

    agent = nodes.new("GeometryNodeCreateCrowdAgent")
    agent.label = "Agent Template"
    links.new(group_input.outputs["Mesh"], agent.inputs["Mesh"])
    links.new(armature_bones.outputs["Bones"], agent.inputs["Bones"])
    links.new(group_input.outputs["Armature"], agent.inputs["Armature"])
    links.new(join_states.outputs["States"], agent.inputs["States"])

    join_agents = nodes.new("GeometryNodeJoinCrowdAgents")
    join_agents.label = "Agents"
    links.new(agent.outputs["Agent"], join_agents.inputs["Agent"])

    # ---------------------------------------------------------------- spawn
    scatter = nodes.new("GeometryNodeDistributePoints")
    scatter.label = "Spawn Points"
    scatter.inputs["Spacing"].default_value = 2.0
    links.new(group_input.outputs["Spawn Surface"], scatter.inputs["Geometry"])

    init = nodes.new("GeometryNodeInitCrowdData")
    init.label = "Init Crowd"
    init.inputs["Default State"].default_value = "Walk"
    init.inputs["Fallback To Any Group"].default_value = True
    links.new(scatter.outputs["Points"], init.inputs["Points"])
    links.new(join_agents.outputs["Agents"], init.inputs["Agents"])
    links.new(join_states.outputs["States"], init.inputs["States"])

    # ---------------------------------------------------------------- simulation zone
    navigation = nodes.new("GeometryNodeCrowdNavigation")
    navigation.label = "Navigation"
    navigation.inputs["Agent Radius"].default_value = 0.35
    navigation.inputs["Max Step Height"].default_value = 0.35
    links.new(group_input.outputs["Obstacles"], navigation.inputs["Obstacles"])
    links.new(group_input.outputs["Target"], navigation.inputs["Goals"])
    links.new(group_input.outputs["Spawn Surface"], navigation.inputs["Bounds Geometry"])

    sim_input = nodes.new("GeometryNodeSimulationInput")
    sim_output = nodes.new("GeometryNodeSimulationOutput")
    sim_input.pair_with_output(sim_output)
    sim_input.location = (320.0, 160.0)
    sim_output.location = (1500.0, 160.0)
    sim_geometry = next(s for s in sim_input.outputs if s.type == "GEOMETRY")
    sim_seed = next(s for s in sim_input.inputs if s.type == "GEOMETRY")
    sim_delta = next(s for s in sim_input.outputs if "Delta" in s.name)
    sim_output_input = next(s for s in sim_output.inputs if s.type == "GEOMETRY")
    sim_output_geometry = next(s for s in sim_output.outputs if s.type == "GEOMETRY")
    links.new(init.outputs["Points"], sim_seed)

    motion = nodes.new("GeometryNodeCrowdMotion")
    motion.label = "Walk"
    motion.inputs["Max Speed"].default_value = 2.0
    motion.inputs["Max Acceleration"].default_value = 12.0
    motion.inputs["Turn Rate"].default_value = 8.0
    motion.inputs["Arrive Radius"].default_value = 0.8
    motion.inputs["Agent Radius"].default_value = 0.35
    motion.inputs["Avoidance Radius"].default_value = 1.6
    motion.inputs["Separation Strength"].default_value = 4.0
    motion.inputs["Hard Separation"].default_value = True
    links.new(sim_geometry, motion.inputs["Points"])
    links.new(group_input.outputs["Target"], motion.inputs["Targets"])
    links.new(join_states.outputs["States"], motion.inputs["States"])
    links.new(navigation.outputs["Navigation"], motion.inputs["Navigation"])
    links.new(sim_delta, motion.inputs["Delta Time"])

    distance_attr = nodes.new("GeometryNodeInputNamedAttribute")
    distance_attr.label = "Distance To Target"
    distance_attr.data_type = "FLOAT"
    distance_attr.inputs["Name"].default_value = "target_distance"

    close_compare = nodes.new("FunctionNodeCompare")
    close_compare.label = "Reached Target"
    close_compare.data_type = "FLOAT"
    close_compare.operation = "LESS_THAN"
    close_compare.inputs["B"].default_value = 1.5
    links.new(distance_attr.outputs["Attribute"], close_compare.inputs["A"])

    far_compare = nodes.new("FunctionNodeCompare")
    far_compare.label = "Leaving Target"
    far_compare.data_type = "FLOAT"
    far_compare.operation = "GREATER_THAN"
    far_compare.inputs["B"].default_value = 2.5
    links.new(distance_attr.outputs["Attribute"], far_compare.inputs["A"])

    switch_idle = nodes.new("GeometryNodeCrowdStateSwitch")
    switch_idle.label = "Reached -> Idle"
    switch_idle.inputs["State"].default_value = "Idle"
    links.new(motion.outputs["Points"], switch_idle.inputs["Points"])
    links.new(close_compare.outputs["Result"], switch_idle.inputs["Condition"])

    switch_walk = nodes.new("GeometryNodeCrowdStateSwitch")
    switch_walk.label = "Leaving -> Walk"
    switch_walk.inputs["State"].default_value = "Walk"
    links.new(switch_idle.outputs["Points"], switch_walk.inputs["Points"])
    links.new(far_compare.outputs["Result"], switch_walk.inputs["Condition"])
    links.new(switch_walk.outputs["Points"], sim_output_input)

    # ---------------------------------------------------------------- animation and skinning
    scene_time = nodes.new("GeometryNodeInputSceneTime")
    scene_time.label = "Scene Time"

    bones = nodes.new("GeometryNodeCrowdBones")
    bones.label = "Crowd Bones"
    links.new(join_agents.outputs["Agents"], bones.inputs["Agents"])
    links.new(join_states.outputs["States"], bones.inputs["States"])
    links.new(sim_output_geometry, bones.inputs["Instances"])
    links.new(scene_time.outputs["Seconds"], bones.inputs["Time"])

    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.label = "Instance Agents"
    links.new(group_input.outputs["Mesh"], instance.inputs["Instance"])
    links.new(sim_output_geometry, instance.inputs["Points"])

    rotation_attr = nodes.new("GeometryNodeInputNamedAttribute")
    rotation_attr.label = "Agent Rotation"
    rotation_attr.data_type = "QUATERNION"
    rotation_attr.inputs["Name"].default_value = "rotation"
    links.new(rotation_attr.outputs["Attribute"], instance.inputs["Rotation"])

    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.label = "Realize Crowd"
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])

    skin = nodes.new("GeometryNodeSkinFromBones")
    skin.label = "Skin Crowd"
    skin.inputs["Mode"].default_value = "Dual Quaternion"
    links.new(realize.outputs["Geometry"], skin.inputs["Mesh"])
    links.new(bones.outputs["Bones"], skin.inputs["Bones"])
    links.new(skin.outputs["Mesh"], group_output.inputs["Geometry"])

    # ---------------------------------------------------------------- layout
    group_input.location = (-700.0, 100.0)
    for index, (state, animation) in enumerate(state_nodes):
        animation.location = (-520.0, 420.0 - index * 220.0)
        state.location = (-300.0, 420.0 - index * 220.0)
    join_states.location = (-60.0, 380.0)
    armature_bones.location = (-520.0, -220.0)
    agent.location = (-260.0, 80.0)
    join_agents.location = (-60.0, 80.0)
    scatter.location = (-300.0, -420.0)
    init.location = (160.0, -40.0)
    navigation.location = (160.0, -260.0)
    motion.location = (560.0, 120.0)
    distance_attr.location = (800.0, 420.0)
    close_compare.location = (980.0, 480.0)
    far_compare.location = (980.0, 300.0)
    switch_idle.location = (1200.0, 420.0)
    switch_walk.location = (1420.0, 360.0)
    sim_output.location = (1500.0, 160.0)
    scene_time.location = (1500.0, -260.0)
    bones.location = (1750.0, 260.0)
    instance.location = (1750.0, -60.0)
    rotation_attr.location = (1500.0, -500.0)
    realize.location = (1980.0, -60.0)
    skin.location = (2220.0, 80.0)
    group_output.location = (2480.0, 80.0)

    for node in (state_nodes[0][0], state_nodes[1][0], state_nodes[2][0], join_states,
                 agent, join_agents, init, motion, bones, instance, realize, skin):
        node.width = 200.0
    return {
        "init": init,
        "motion": motion,
        "bones": bones,
        "skin": skin,
        "sim_input": sim_input,
        "sim_output": sim_output,
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


class DILAPIDO_OT_quick_crowd_walk(Operator):
    """Build a walking crowd template (states, motion, local separation, skinning) in this tree"""

    bl_idname = "dilapido.quick_crowd_walk"
    bl_label = "Quick Crowd Walk"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return (space is not None and space.type == "NODE_EDITOR" and
                space.tree_type == "GeometryNodeTree" and space.edit_tree is not None)

    def execute(self, context):
        node_tree = context.space_data.edit_tree
        try:
            build_quick_crowd_walk(node_tree)
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not build Quick Crowd Walk: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Crowd walk template created")
        return {"FINISHED"}


class DILAPIDO_PT_assist(Panel):
    """Rigid body and crowd helpers"""

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
        column.operator("dilapido.quick_crowd_walk", icon="COMMUNITY")
        layout.label(text="Quick RBD: fracture, clusters, constraints and the solver.", icon="INFO")
        layout.label(text="Quick Crowd Walk: states, target motion, separation and skinning.",
                     icon="INFO")


classes = (
    DILAPIDO_OT_quick_rbd,
    DILAPIDO_OT_quick_crowd_walk,
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
