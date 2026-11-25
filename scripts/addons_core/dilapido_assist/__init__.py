# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Dilapido Assist: helpers for the Jolt rigid body and crowd geometry nodes.

The panel lives in the sidebar of the Geometry Nodes editor and offers:

- `Quick RBD` builds a working rigid body setup in the current node tree:
  `Group Input -> Voronoi Fracture -> Fracture Cluster -> Init Rigid Body -> Rigid Body Dynamics`.
- `Quick Crowd Walk` builds a walking crowd template: three states, spawn scattering,
  `Crowd Motion` with local separation, navigation, `Crowd Bones` and skinning. The crowd stays
  instanced (`Skin From Bones` deforms each instance) so no `Realize Instances` runs per frame. It
  uses the native nodes directly (no asset group) and does **not** change the group interface:
  external geometry/objects are picked on `Object Info` / `Object` nodes, and numeric defaults are
  `Value` nodes.
- `Quick Crowd Ragdoll` builds the same walk crowd plus a shared ragdoll world, switch and
  recovery blend.
- `Auto Arrange` arranges selected nodes (or all) with the Node Arrange extension
  (`bpy.ops.node.na_arrange_selected()`), falling back to a built-in layered layout.
- `Update Selected Node Group` refreshes the asset behind the selected group node: linked copies
  reload their library (in place), appended copies are replaced with a fresh copy from the shipped
  asset and every user is switched over with the links restored by socket name.

Everything is wired, so a modifier that has this node tree runs right away; the sizes are sensible
defaults that can be tweaked afterwards.
"""

bl_info = {
    "name": "Dilapido Assist",
    "author": "Dilapido",
    "version": (0, 2, 0),
    "blender": (5, 2, 0),
    "location": "Geometry Nodes editor > Sidebar > Dilapido",
    "description": "One click rigid body and crowd setups plus asset updates with the Jolt solver",
    "category": "Node",
}

import os
import re

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


#: Cache for `_asset_file_for_group()`; cleared every time the update operator runs.
_ASSET_FILE_CACHE = {}


def _asset_blend_files():
    """Every installed .blend that may contain an asset node group."""
    roots = []
    datafiles = bpy.utils.system_resource("DATAFILES")
    if datafiles:
        roots.append(os.path.join(datafiles, "assets"))
    for library in bpy.context.preferences.filepaths.asset_libraries:
        path = bpy.path.abspath(library.path)
        if path:
            roots.append(path)

    files = []
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for directory, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for filename in sorted(filenames):
                if not filename.endswith(".blend"):
                    continue
                path = os.path.join(directory, filename)
                key = os.path.normcase(os.path.normpath(path))
                if key in seen:
                    continue
                seen.add(key)
                files.append(path)
    return files


def _asset_file_for_group(name):
    """Return the .blend that contains a node group called `name`, or None."""
    if name in _ASSET_FILE_CACHE:
        return _ASSET_FILE_CACHE[name]
    path = None
    for candidate in _asset_blend_files():
        try:
            with bpy.data.libraries.load(candidate, link=False) as (data_from, _data_to):
                if name in data_from.node_groups:
                    path = candidate
                    break
        except (OSError, RuntimeError):
            continue
    _ASSET_FILE_CACHE[name] = path
    return path


def _asset_source_name(name):
    """Return the source name and file for a local copy, which may carry a `.001` suffix."""
    candidates = [name]
    stripped = re.sub(r"\.\d{3}$", "", name)
    if stripped != name:
        candidates.append(stripped)
    for candidate in candidates:
        path = _asset_file_for_group(candidate)
        if path is not None:
            return candidate, path
    return None, None


def _group_node_users(group):
    """Group nodes that instantiate `group`, with their links recorded by socket name."""
    users = []
    for node_tree in bpy.data.node_groups:
        for node in node_tree.nodes:
            if node.bl_idname != "GeometryNodeGroup" or node.node_tree is not group:
                continue
            inputs = [(socket.name, link.from_socket)
                      for socket in node.inputs for link in socket.links]
            outputs = [(socket.name, link.to_socket)
                       for socket in node.outputs for link in socket.links]
            users.append((node_tree, node, inputs, outputs))
    return users


def _append_asset_group(source_name, path):
    """Append a fresh copy of `source_name` from `path` (including its dependencies)."""
    before = set(bpy.data.node_groups)
    try:
        bpy.ops.wm.append(
            filepath=os.path.join(path, "NodeTree"),
            directory=os.path.join(path, "NodeTree") + os.sep,
            filename=source_name,
            do_reuse_local_id=False,
        )
    except RuntimeError:
        return None
    for group in bpy.data.node_groups:
        if group in before:
            continue
        if group.name == source_name or group.name.startswith(source_name + "."):
            return group
    return None


def _replace_appended_group(group):
    """Replace an appended asset group with a fresh copy, keeping every user connected."""
    source_name, path = _asset_source_name(group.name)
    if source_name is None:
        return False
    users = _group_node_users(group)
    fresh = _append_asset_group(source_name, path)
    if fresh is None:
        return False

    for node_tree, node, inputs, outputs in users:
        node.node_tree = fresh
        for socket_name, from_socket in inputs:
            socket = node.inputs.get(socket_name)
            if socket is None or from_socket is None:
                continue
            if not any(link.from_socket == from_socket for link in socket.links):
                node_tree.links.new(from_socket, socket)
        for socket_name, to_socket in outputs:
            socket = node.outputs.get(socket_name)
            if socket is None or to_socket is None:
                continue
            if not any(link.from_socket == socket for link in to_socket.links):
                node_tree.links.new(socket, to_socket)

    if group.users:
        group.user_remap(fresh)
    fresh.asset_clear()
    fresh.use_fake_user = group.use_fake_user
    local_name = group.name
    bpy.data.node_groups.remove(group)
    fresh.name = local_name
    return True


def update_node_groups(groups):
    """Refresh node groups from their asset source.

    Linked groups reload their library, which updates every group of that file in place.
    Appended groups get a fresh copy from the shipped asset, every node/modifier using them is
    switched over, and the links are restored by socket name.

    Returns `(linked, appended, skipped)`.
    """
    linked = 0
    appended = 0
    skipped = 0
    reloaded = set()
    seen = set()
    for group in groups:
        key = group.as_pointer()
        if key in seen:
            continue
        seen.add(key)
        if group.library is not None:
            library = group.library
            if library.name in reloaded:
                linked += 1
                continue
            try:
                library.reload()
            except (RuntimeError, OSError):
                skipped += 1
                continue
            reloaded.add(library.name)
            linked += 1
        elif _replace_appended_group(group):
            appended += 1
        else:
            skipped += 1
    return linked, appended, skipped


def _selected_group_nodes(context):
    """The selected group nodes of the active node editor."""
    space = context.space_data
    node_tree = getattr(space, "edit_tree", None)
    if node_tree is None:
        return []
    nodes = getattr(context, "selected_nodes", None)
    if not nodes:
        nodes = [node for node in node_tree.nodes if node.select]
    return [node for node in nodes
            if node.bl_idname == "GeometryNodeGroup" and node.node_tree is not None]


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


def _node_height_estimate(node):
    def counts(sockets):
        try:
            return len(sockets)
        except TypeError:
            return 0

    return max(90.0, 40.0 + 22.0 * max(counts(node.inputs), counts(node.outputs)))


def _fallback_arrange(node_tree, nodes, column_gap=90.0, row_gap=60.0):
    """Layered (Sugiyama-style) layout used when the Node Arrange extension is unavailable."""
    nodes = list(nodes)
    if not nodes:
        return
    node_set = set(nodes)
    links = [
        link for link in node_tree.links
        if link.from_node in node_set and link.to_node in node_set
    ]

    # Longest path from a source: every node ends up right of all of its inputs.
    depth = {node: 0 for node in nodes}
    for _ in range(len(nodes)):
        changed = False
        for link in links:
            source_depth = depth.get(link.from_node, 0) + 1
            if depth.get(link.to_node, 0) < source_depth:
                depth[link.to_node] = source_depth
                changed = True
        if not changed:
            break

    inputs_of = {node: [] for node in nodes}
    outputs_of = {node: [] for node in nodes}
    for link in links:
        inputs_of[link.to_node].append(link.from_node)
        outputs_of[link.from_node].append(link.to_node)

    columns = {}
    for node in nodes:
        columns.setdefault(depth[node], []).append(node)

    def row_of(node):
        column_nodes = columns.get(depth.get(node, 0))
        if column_nodes is None:
            return 0
        try:
            return column_nodes.index(node)
        except ValueError:
            return 0

    for _ in range(6):
        for column in sorted(columns)[1:]:
            items = columns[column]
            rows = {node: index for index, node in enumerate(items)}

            def forward_key(node, rows=rows):
                neighbours = inputs_of[node]
                if not neighbours:
                    return rows[node]
                return sum(row_of(n) for n in neighbours) / len(neighbours)

            items.sort(key=forward_key)
        for column in reversed(sorted(columns)[:-1]):
            items = columns[column]
            rows = {node: index for index, node in enumerate(items)}

            def backward_key(node, rows=rows):
                neighbours = outputs_of[node]
                if not neighbours:
                    return rows[node]
                return sum(row_of(n) for n in neighbours) / len(neighbours)

            items.sort(key=backward_key)

    x = 0.0
    for column in sorted(columns):
        items = columns[column]
        heights = [_node_height_estimate(node) for node in items]
        total = sum(heights) + row_gap * (len(items) - 1)
        y = total * 0.5
        for node, height in zip(items, heights):
            node.location = (x, y - height * 0.5)
            y -= height + row_gap
        x += max(180.0, max(getattr(node, "width", 0.0) for node in items)) + column_gap


def _arrange_nodes(node_tree, nodes=None):
    """Arrange nodes with the Node Arrange extension, falling back to a built-in layout.

    Returns True when the extension did the work. When `nodes` is None, selected nodes are
    arranged, or the whole tree if nothing is selected.
    """
    if nodes is None:
        selected = [node for node in node_tree.nodes if node.select]
        target = selected if selected else list(node_tree.nodes)
    else:
        target = list(nodes)
    if not target:
        return True

    old_selection = [(node, node.select) for node in node_tree.nodes]
    for node in node_tree.nodes:
        node.select = node in target

    used_extension = False
    space = getattr(bpy.context, "space_data", None)
    if space is not None and getattr(space, "edit_tree", None) == node_tree:
        try:
            result = bpy.ops.node.na_arrange_selected()
            used_extension = "FINISHED" in result
        except (AttributeError, RuntimeError, TypeError):
            used_extension = False
        finally:
            for node, selected in old_selection:
                node.select = selected
    else:
        for node, selected in old_selection:
            node.select = selected

    if not used_extension:
        _fallback_arrange(node_tree, target)
    return used_extension


def build_quick_crowd_walk(node_tree, realize_output=False):
    """Build the fixed `Crowd Walk` template tree (no asset group, no rigid bodies).

    With `realize_output=False` (default) the crowd stays instanced: `Skin From Bones` skins every
    instance directly, so no `Realize Instances` runs per frame. Set `realize_output=True` to keep
    the old single-mesh output.
    """
    nodes = node_tree.nodes
    links = node_tree.links

    def find_or_add(bl_idname, label):
        for node in nodes:
            if node.bl_idname == bl_idname:
                return node
        node = nodes.new(bl_idname)
        node.label = label
        return node

    group_output = find_or_add("NodeGroupOutput", "Crowd Output")

    # External inputs are node pickers, not group interface sockets. The tree is self-contained.
    mesh_info = nodes.new("GeometryNodeObjectInfo")
    mesh_info.label = "Crowd Mesh"
    spawn_info = nodes.new("GeometryNodeObjectInfo")
    spawn_info.label = "Spawn Surface"
    target_info = nodes.new("GeometryNodeObjectInfo")
    target_info.label = "Crowd Target"
    obstacles_info = nodes.new("GeometryNodeObjectInfo")
    obstacles_info.label = "Crowd Obstacles"
    armature_input = nodes.new("GeometryNodeInputObject")
    armature_input.label = "Crowd Armature"

    spacing_value = nodes.new("ShaderNodeValue")
    spacing_value.label = "Spawn Spacing"
    spacing_value.outputs[0].default_value = 2.0
    speed_value = nodes.new("ShaderNodeValue")
    speed_value.label = "Max Speed"
    speed_value.outputs[0].default_value = 2.0
    radius_value = nodes.new("ShaderNodeValue")
    radius_value.label = "Agent Radius"
    radius_value.outputs[0].default_value = 0.35

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
    links.new(armature_input.outputs["Object"], armature_bones.inputs["Armature"])

    agent = nodes.new("GeometryNodeCreateCrowdAgent")
    agent.label = "Agent Template"
    links.new(mesh_info.outputs["Geometry"], agent.inputs["Mesh"])
    links.new(armature_bones.outputs["Bones"], agent.inputs["Bones"])
    links.new(armature_input.outputs["Object"], agent.inputs["Armature"])
    links.new(join_states.outputs["States"], agent.inputs["States"])

    join_agents = nodes.new("GeometryNodeJoinCrowdAgents")
    join_agents.label = "Agents"
    links.new(agent.outputs["Agent"], join_agents.inputs["Agent"])

    # ---------------------------------------------------------------- spawn
    scatter = nodes.new("GeometryNodeDistributePoints")
    scatter.label = "Spawn Points"
    links.new(spacing_value.outputs[0], scatter.inputs["Spacing"])
    links.new(spawn_info.outputs["Geometry"], scatter.inputs["Geometry"])

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
    links.new(radius_value.outputs[0], navigation.inputs["Agent Radius"])
    navigation.inputs["Max Step Height"].default_value = 0.35
    links.new(obstacles_info.outputs["Geometry"], navigation.inputs["Obstacles"])
    links.new(target_info.outputs["Geometry"], navigation.inputs["Goals"])
    links.new(spawn_info.outputs["Geometry"], navigation.inputs["Bounds Geometry"])
    navmesh_info = nodes.new("GeometryNodeObjectInfo")
    navmesh_info.label = "Crowd Nav Mesh"
    links.new(navmesh_info.outputs["Geometry"], navigation.inputs["Walkable Surface"])

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
    links.new(speed_value.outputs[0], motion.inputs["Max Speed"])
    motion.inputs["Max Acceleration"].default_value = 12.0
    motion.inputs["Turn Rate"].default_value = 8.0
    motion.inputs["Arrive Radius"].default_value = 0.8
    links.new(radius_value.outputs[0], motion.inputs["Agent Radius"])
    motion.inputs["Avoidance Radius"].default_value = 1.6
    motion.inputs["Separation Strength"].default_value = 4.0
    motion.inputs["Hard Separation"].default_value = True
    links.new(sim_geometry, motion.inputs["Points"])
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
    switch_idle.inputs["Wait For Blend"].default_value = True
    links.new(motion.outputs["Points"], switch_idle.inputs["Points"])
    links.new(close_compare.outputs["Result"], switch_idle.inputs["Condition"])
    links.new(join_states.outputs["States"], switch_idle.inputs["States"])

    switch_walk = nodes.new("GeometryNodeCrowdStateSwitch")
    switch_walk.label = "Leaving -> Walk"
    switch_walk.inputs["State"].default_value = "Walk"
    switch_walk.inputs["Wait For Blend"].default_value = True
    links.new(switch_idle.outputs["Points"], switch_walk.inputs["Points"])
    links.new(far_compare.outputs["Result"], switch_walk.inputs["Condition"])
    links.new(join_states.outputs["States"], switch_walk.inputs["States"])
    links.new(switch_walk.outputs["Points"], sim_output_input)

    # ---------------------------------------------------------------- animation and skinning
    scene_time = nodes.new("GeometryNodeInputSceneTime")
    scene_time.label = "Scene Time"

    bones = nodes.new("GeometryNodeCrowdBones")
    bones.label = "Crowd Bones"
    bones.inputs["Speed Match"].default_value = True
    links.new(join_agents.outputs["Agents"], bones.inputs["Agents"])
    links.new(join_states.outputs["States"], bones.inputs["States"])
    links.new(sim_output_geometry, bones.inputs["Instances"])
    links.new(scene_time.outputs["Seconds"], bones.inputs["Time"])

    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.label = "Instance Agents"
    links.new(mesh_info.outputs["Geometry"], instance.inputs["Instance"])
    links.new(sim_output_geometry, instance.inputs["Points"])

    rotation_attr = nodes.new("GeometryNodeInputNamedAttribute")
    rotation_attr.label = "Agent Rotation"
    rotation_attr.data_type = "QUATERNION"
    rotation_attr.inputs["Name"].default_value = "rotation"
    links.new(rotation_attr.outputs["Attribute"], instance.inputs["Rotation"])

    skin = nodes.new("GeometryNodeSkinFromBones")
    skin.label = "Skin Crowd"
    skin.inputs["Mode"].default_value = "Dual Quaternion"
    links.new(instance.outputs["Instances"], skin.inputs["Mesh"])
    links.new(bones.outputs["Bones"], skin.inputs["Bones"])

    realize = None
    if realize_output:
        realize = nodes.new("GeometryNodeRealizeInstances")
        realize.label = "Realize Crowd"
        links.new(skin.outputs["Mesh"], realize.inputs["Geometry"])

    final_socket = realize.outputs["Geometry"] if realize is not None else skin.outputs["Mesh"]
    output_index = next(
        (index for index, socket in enumerate(group_output.inputs) if socket.type == "GEOMETRY"),
        None,
    )
    output_linked = output_index is not None
    if output_linked:
        links.new(final_socket, group_output.inputs[output_index])

    # ---------------------------------------------------------------- layout
    nodes_to_size = [state_nodes[0][0], state_nodes[1][0], state_nodes[2][0], join_states,
                     agent, join_agents, init, motion, navigation, bones, instance, skin]
    if realize is not None:
        nodes_to_size.append(realize)
    for node in nodes_to_size:
        node.width = 200.0
    for node in (mesh_info, spawn_info, target_info, obstacles_info, armature_input):
        node.width = 180.0

    # Use the Node Arrange extension (same algorithm as `bpy.ops.node.na_arrange_selected()`),
    # falling back to the built-in layered layout when it is unavailable (headless/tests).
    _arrange_nodes(node_tree)

    return {
        "init": init,
        "motion": motion,
        "bones": bones,
        "skin": skin,
        "navigation": navigation,
        "instance": instance,
        "mesh_info": mesh_info,
        "spawn_info": spawn_info,
        "target_info": target_info,
        "obstacles_info": obstacles_info,
        "armature_input": armature_input,
        "sim_input": sim_input,
        "sim_output": sim_output,
        "output_linked": output_linked,
    }


def build_quick_crowd_ragdoll(node_tree, realize_output=False):
    """Build a walking crowd whose agents can switch to one shared ragdoll world.

    The tree contains the same states / spawn / navigation / motion chain as `Quick Crowd Walk`,
    plus a seeded ragdoll world, a `Crowd Ragdoll Switch` + `Blend Bones` pair inside one
    simulation zone, and `Skin From Bones` on the solved bones. All external inputs are node
    pickers (`Crowd Mesh` / `Crowd Armature` / `Spawn Surface` / `Crowd Target` / `Crowd
    Obstacles`) and `Value` nodes, so the node group interface is left untouched.
    """
    nodes = node_tree.nodes
    links = node_tree.links

    def find_or_add(bl_idname, label):
        for node in nodes:
            if node.bl_idname == bl_idname:
                return node
        node = nodes.new(bl_idname)
        node.label = label
        return node

    def store_bundle(bundle_input, path, socket_type="GEOMETRY"):
        node = nodes.new("NodeStoreBundleItem")
        node.socket_type = socket_type
        node.structure_type = "SINGLE"
        node.inputs["Path"].default_value = path
        if bundle_input is not None:
            links.new(bundle_input, node.inputs["Bundle"])
        return node

    def get_bundle(bundle_socket, path):
        node = nodes.new("NodeGetBundleItem")
        node.socket_type = "GEOMETRY"
        node.structure_type = "SINGLE"
        node.inputs["Path"].default_value = path
        links.new(bundle_socket, node.inputs["Bundle"])
        return node

    def named_attribute(name, data_type):
        node = nodes.new("GeometryNodeInputNamedAttribute")
        node.data_type = data_type
        node.inputs["Name"].default_value = name
        return node

    def mark_state_cloud(geometry_socket):
        """Tag the crowd/bone clouds so the solver does not simulate them as point bodies."""
        node = nodes.new("GeometryNodeStoreNamedAttribute")
        node.label = "Mark Crowd State"
        node.data_type = "INT"
        node.domain = "POINT"
        node.inputs["Name"].default_value = "piece_id"
        node.inputs["Value"].default_value = -1
        links.new(geometry_socket, node.inputs["Geometry"])
        return node

    group_output = find_or_add("NodeGroupOutput", "Crowd Output")

    # External inputs are node pickers, not group interface sockets.
    mesh_info = nodes.new("GeometryNodeObjectInfo")
    mesh_info.label = "Crowd Mesh"
    spawn_info = nodes.new("GeometryNodeObjectInfo")
    spawn_info.label = "Spawn Surface"
    target_info = nodes.new("GeometryNodeObjectInfo")
    target_info.label = "Crowd Target"
    obstacles_info = nodes.new("GeometryNodeObjectInfo")
    obstacles_info.label = "Crowd Obstacles"
    armature_input = nodes.new("GeometryNodeInputObject")
    armature_input.label = "Crowd Armature"

    spacing_value = nodes.new("ShaderNodeValue")
    spacing_value.label = "Spawn Spacing"
    spacing_value.outputs[0].default_value = 2.0
    speed_value = nodes.new("ShaderNodeValue")
    speed_value.label = "Max Speed"
    speed_value.outputs[0].default_value = 2.0
    radius_value = nodes.new("ShaderNodeValue")
    radius_value.label = "Agent Radius"
    radius_value.outputs[0].default_value = 0.35
    trigger_value = nodes.new("ShaderNodeValue")
    trigger_value.label = "Ragdoll Trigger"
    trigger_value.outputs[0].default_value = 0.0
    recover_value = nodes.new("ShaderNodeValue")
    recover_value.label = "Recover After"
    recover_value.outputs[0].default_value = 2.5

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
    links.new(armature_input.outputs["Object"], armature_bones.inputs["Armature"])

    agent = nodes.new("GeometryNodeCreateCrowdAgent")
    agent.label = "Agent Template"
    links.new(mesh_info.outputs["Geometry"], agent.inputs["Mesh"])
    links.new(armature_bones.outputs["Bones"], agent.inputs["Bones"])
    links.new(armature_input.outputs["Object"], agent.inputs["Armature"])
    links.new(join_states.outputs["States"], agent.inputs["States"])

    join_agents = nodes.new("GeometryNodeJoinCrowdAgents")
    join_agents.label = "Agents"
    links.new(agent.outputs["Agent"], join_agents.inputs["Agent"])

    # ---------------------------------------------------------------- spawn
    scatter = nodes.new("GeometryNodeDistributePoints")
    scatter.label = "Spawn Points"
    links.new(spacing_value.outputs[0], scatter.inputs["Spacing"])
    links.new(spawn_info.outputs["Geometry"], scatter.inputs["Geometry"])

    init = nodes.new("GeometryNodeInitCrowdData")
    init.label = "Init Crowd"
    init.inputs["Default State"].default_value = "Walk"
    init.inputs["Fallback To Any Group"].default_value = True
    links.new(scatter.outputs["Points"], init.inputs["Points"])
    links.new(join_agents.outputs["Agents"], init.inputs["Agents"])
    links.new(join_states.outputs["States"], init.inputs["States"])

    scene_time = nodes.new("GeometryNodeInputSceneTime")
    scene_time.label = "Scene Time"

    # ---------------------------------------------------------------- seed world
    seed_bones = nodes.new("GeometryNodeCrowdBones")
    seed_bones.label = "Seed Bones"
    links.new(join_agents.outputs["Agents"], seed_bones.inputs["Agents"])
    links.new(join_states.outputs["States"], seed_bones.inputs["States"])
    links.new(init.outputs["Points"], seed_bones.inputs["Instances"])
    links.new(scene_time.outputs["Seconds"], seed_bones.inputs["Time"])

    seed_bodies = nodes.new("GeometryNodeInitRagdollBodies")
    seed_bodies.label = "Seed Ragdoll"
    seed_bodies.inputs["Animate"].default_value = True
    links.new(seed_bones.outputs["Bones"], seed_bodies.inputs["Bones"])
    links.new(named_attribute("agent_id", "INT").outputs["Attribute"],
              seed_bodies.inputs["Group ID"])

    joints = nodes.new("GeometryNodeRagdollJoints")
    joints.label = "Ragdoll Joints"
    links.new(seed_bones.outputs["Bones"], joints.inputs["Bones"])
    links.new(seed_bodies.outputs["Bodies"], joints.inputs["Bodies"])

    empty_snapshot = nodes.new("GeometryNodePoints")
    empty_snapshot.label = "Empty Snapshot"
    empty_snapshot.inputs["Count"].default_value = 0

    seed_points = mark_state_cloud(init.outputs["Points"])

    world = store_bundle(None, "geometry")
    links.new(seed_bodies.outputs["Bodies"], world.inputs["Item"])
    world = store_bundle(world.outputs["Bundle"], "constraints")
    links.new(joints.outputs["Constraints"], world.inputs["Item"])
    world = store_bundle(world.outputs["Bundle"], "points")
    links.new(seed_points.outputs["Geometry"], world.inputs["Item"])
    world = store_bundle(world.outputs["Bundle"], "recover")
    links.new(empty_snapshot.outputs["Points"], world.inputs["Item"])

    # ---------------------------------------------------------------- simulation zone
    sim_input = nodes.new("GeometryNodeSimulationInput")
    sim_output = nodes.new("GeometryNodeSimulationOutput")
    sim_input.pair_with_output(sim_output)
    sim_state = sim_output.state_items[0]
    sim_state.name = "World"
    sim_state.socket_type = "BUNDLE"
    sim_input.location = (320.0, 220.0)
    sim_output.location = (2200.0, 220.0)
    sim_delta = next(s for s in sim_input.outputs if "Delta" in s.name)
    links.new(world.outputs["Bundle"], sim_input.inputs["World"])

    world_state = sim_input.outputs["World"]
    previous_bones = get_bundle(world_state, "geometry")
    state_points = get_bundle(world_state, "points")
    state_recover = get_bundle(world_state, "recover")

    navigation = nodes.new("GeometryNodeCrowdNavigation")
    navigation.label = "Navigation"
    links.new(radius_value.outputs[0], navigation.inputs["Agent Radius"])
    navigation.inputs["Max Step Height"].default_value = 0.35
    links.new(obstacles_info.outputs["Geometry"], navigation.inputs["Obstacles"])
    links.new(target_info.outputs["Geometry"], navigation.inputs["Goals"])
    links.new(spawn_info.outputs["Geometry"], navigation.inputs["Bounds Geometry"])
    navmesh_info = nodes.new("GeometryNodeObjectInfo")
    navmesh_info.label = "Crowd Nav Mesh"
    links.new(navmesh_info.outputs["Geometry"], navigation.inputs["Walkable Surface"])

    motion = nodes.new("GeometryNodeCrowdMotion")
    motion.label = "Walk"
    links.new(speed_value.outputs[0], motion.inputs["Max Speed"])
    motion.inputs["Max Acceleration"].default_value = 12.0
    motion.inputs["Turn Rate"].default_value = 8.0
    motion.inputs["Arrive Radius"].default_value = 0.8
    links.new(radius_value.outputs[0], motion.inputs["Agent Radius"])
    motion.inputs["Avoidance Radius"].default_value = 1.6
    motion.inputs["Separation Strength"].default_value = 4.0
    motion.inputs["Hard Separation"].default_value = True
    links.new(state_points.outputs["Item"], motion.inputs["Points"])
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
    switch_idle.inputs["Wait For Blend"].default_value = True
    links.new(motion.outputs["Points"], switch_idle.inputs["Points"])
    links.new(close_compare.outputs["Result"], switch_idle.inputs["Condition"])
    links.new(join_states.outputs["States"], switch_idle.inputs["States"])

    switch_walk = nodes.new("GeometryNodeCrowdStateSwitch")
    switch_walk.label = "Leaving -> Walk"
    switch_walk.inputs["State"].default_value = "Walk"
    switch_walk.inputs["Wait For Blend"].default_value = True
    links.new(switch_idle.outputs["Points"], switch_walk.inputs["Points"])
    links.new(far_compare.outputs["Result"], switch_walk.inputs["Condition"])
    links.new(join_states.outputs["States"], switch_walk.inputs["States"])

    # ---------------------------------------------------------------- ragdoll switch + recovery
    trigger_compare = nodes.new("FunctionNodeCompare")
    trigger_compare.label = "Trigger Ragdoll"
    trigger_compare.data_type = "FLOAT"
    trigger_compare.operation = "GREATER_THAN"
    trigger_compare.inputs["B"].default_value = 0.5
    links.new(trigger_value.outputs[0], trigger_compare.inputs["A"])

    switch_time_attr = named_attribute("ragdoll_switch_time", "FLOAT")
    elapsed_math = nodes.new("ShaderNodeMath")
    elapsed_math.label = "Time Since Switch"
    elapsed_math.operation = "SUBTRACT"
    links.new(scene_time.outputs["Seconds"], elapsed_math.inputs[0])
    links.new(switch_time_attr.outputs["Attribute"], elapsed_math.inputs[1])

    recover_compare = nodes.new("FunctionNodeCompare")
    recover_compare.label = "Recover Ragdoll"
    recover_compare.data_type = "FLOAT"
    recover_compare.operation = "GREATER_THAN"
    links.new(elapsed_math.outputs[0], recover_compare.inputs["A"])
    links.new(recover_value.outputs[0], recover_compare.inputs["B"])

    ragdoll_switch = nodes.new("GeometryNodeCrowdRagdollSwitch")
    ragdoll_switch.label = "Ragdoll Switch"
    ragdoll_switch.inputs["Recover Duration"].default_value = 0.8
    ragdoll_switch.inputs["Max Dynamic"].default_value = 32
    ragdoll_switch.inputs["Pending Timeout"].default_value = 5.0
    ragdoll_switch.inputs["Once"].default_value = True
    ragdoll_switch.inputs["Cooldown"].default_value = 0.0
    links.new(switch_walk.outputs["Points"], ragdoll_switch.inputs["Points"])
    links.new(previous_bones.outputs["Item"], ragdoll_switch.inputs["Previous Bones"])
    links.new(state_recover.outputs["Item"], ragdoll_switch.inputs["Previous Recover"])
    links.new(trigger_compare.outputs["Result"], ragdoll_switch.inputs["Condition"])
    links.new(recover_compare.outputs["Result"], ragdoll_switch.inputs["Recover Condition"])
    links.new(scene_time.outputs["Seconds"], ragdoll_switch.inputs["Time"])

    blend = nodes.new("GeometryNodeBlendBones")
    blend.label = "Blend Recovery"
    links.new(ragdoll_switch.outputs["Recover"], blend.inputs["From Bones"])
    links.new(ragdoll_switch.outputs["Points"], blend.inputs["Points"])

    live_bones = nodes.new("GeometryNodeCrowdBones")
    live_bones.label = "Live Bones"
    live_bones.inputs["Speed Match"].default_value = True
    links.new(join_agents.outputs["Agents"], live_bones.inputs["Agents"])
    links.new(join_states.outputs["States"], live_bones.inputs["States"])
    links.new(ragdoll_switch.outputs["Points"], live_bones.inputs["Instances"])
    links.new(scene_time.outputs["Seconds"], live_bones.inputs["Time"])
    links.new(live_bones.outputs["Bones"], blend.inputs["To Bones"])

    live_bodies = nodes.new("GeometryNodeInitRagdollBodies")
    live_bodies.label = "Live Ragdoll"
    links.new(blend.outputs["Bones"], live_bodies.inputs["Bones"])
    links.new(named_attribute("agent_id", "INT").outputs["Attribute"],
              live_bodies.inputs["Group ID"])
    links.new(named_attribute("animated", "BOOLEAN").outputs["Attribute"],
              live_bodies.inputs["Animate"])

    solver = nodes.new("GeometryNodeRigidBodySolver")
    solver.label = "Ragdoll Solver"
    solver.inputs["Substeps"].default_value = 4
    solver.inputs["Constraint Iterations"].default_value = 12
    solver.inputs["Allow Sleeping"].default_value = False
    solver.inputs["Threads"].default_value = 1
    links.new(world_state, solver.inputs["World"])
    links.new(sim_delta, solver.inputs["Delta Time"])
    links.new(live_bodies.outputs["Bodies"], solver.inputs["Upstream"])

    solved = store_bundle(solver.outputs["World"], "points")
    links.new(ragdoll_switch.outputs["Points"], solved.inputs["Item"])
    solved = store_bundle(solved.outputs["Bundle"], "recover")
    marked_recover = mark_state_cloud(ragdoll_switch.outputs["Recover"])
    links.new(marked_recover.outputs["Geometry"], solved.inputs["Item"])
    links.new(solved.outputs["Bundle"], sim_output.inputs["World"])

    # ---------------------------------------------------------------- skinning
    solved_world = sim_output.outputs["World"]
    solved_bones = get_bundle(solved_world, "geometry")
    solved_points = get_bundle(solved_world, "points")

    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.label = "Instance Agents"
    links.new(mesh_info.outputs["Geometry"], instance.inputs["Instance"])
    links.new(solved_points.outputs["Item"], instance.inputs["Points"])

    rotation_attr = nodes.new("GeometryNodeInputNamedAttribute")
    rotation_attr.label = "Agent Rotation"
    rotation_attr.data_type = "QUATERNION"
    rotation_attr.inputs["Name"].default_value = "rotation"
    links.new(rotation_attr.outputs["Attribute"], instance.inputs["Rotation"])

    skin = nodes.new("GeometryNodeSkinFromBones")
    skin.label = "Skin Crowd"
    skin.inputs["Mode"].default_value = "Dual Quaternion"
    links.new(instance.outputs["Instances"], skin.inputs["Mesh"])
    links.new(solved_bones.outputs["Item"], skin.inputs["Bones"])

    realize = None
    if realize_output:
        realize = nodes.new("GeometryNodeRealizeInstances")
        realize.label = "Realize Crowd"
        links.new(skin.outputs["Mesh"], realize.inputs["Geometry"])

    final_socket = realize.outputs["Geometry"] if realize is not None else skin.outputs["Mesh"]
    output_index = next(
        (index for index, socket in enumerate(group_output.inputs) if socket.type == "GEOMETRY"),
        None,
    )
    output_linked = output_index is not None
    if output_linked:
        links.new(final_socket, group_output.inputs[output_index])

    # ---------------------------------------------------------------- layout
    nodes_to_size = [state_nodes[0][0], state_nodes[1][0], state_nodes[2][0], join_states,
                     agent, join_agents, init, motion, navigation, seed_bones, seed_bodies,
                     joints, live_bones, live_bodies, ragdoll_switch, blend, solver, instance,
                     skin]
    if realize is not None:
        nodes_to_size.append(realize)
    for node in nodes_to_size:
        node.width = 200.0
    for node in (mesh_info, spawn_info, target_info, obstacles_info,
                 armature_input, trigger_value, recover_value):
        node.width = 180.0

    _arrange_nodes(node_tree)

    return {
        "init": init,
        "motion": motion,
        "navigation": navigation,
        "seed_bones": seed_bones,
        "live_bones": live_bones,
        "ragdoll_switch": ragdoll_switch,
        "blend": blend,
        "solver": solver,
        "skin": skin,
        "instance": instance,
        "trigger": trigger_value,
        "recover_after": recover_value,
        "mesh_info": mesh_info,
        "spawn_info": spawn_info,
        "target_info": target_info,
        "obstacles_info": obstacles_info,
        "armature_input": armature_input,
        "sim_input": sim_input,
        "sim_output": sim_output,
        "output_linked": output_linked,
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

    realize_output: bpy.props.BoolProperty(
        name="Realize Output",
        description=("Merge the skinned crowd into one mesh. Slower with many agents; leave off "
                     "to keep the crowd instanced"),
        default=False,
    )

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return (space is not None and space.type == "NODE_EDITOR" and
                space.tree_type == "GeometryNodeTree" and space.edit_tree is not None)

    def execute(self, context):
        node_tree = context.space_data.edit_tree
        try:
            created = build_quick_crowd_walk(node_tree, realize_output=self.realize_output)
            if created is not None and not created.get("output_linked", True):
                self.report(
                    {"WARNING"},
                    "The crowd tree was created but no Geometry output socket exists. Add one to "
                    "the group and connect 'Skin Crowd'",
                )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not build Quick Crowd Walk: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Crowd walk template created")
        return {"FINISHED"}


class DILAPIDO_OT_quick_crowd_ragdoll(Operator):
    """Build a walking crowd with a shared ragdoll world, budget and recovery blend"""

    bl_idname = "dilapido.quick_crowd_ragdoll"
    bl_label = "Quick Crowd Ragdoll"
    bl_options = {"REGISTER", "UNDO"}

    realize_output: bpy.props.BoolProperty(
        name="Realize Output",
        description=("Merge the skinned crowd into one mesh. Slower with many agents; leave off "
                     "to keep the crowd instanced"),
        default=False,
    )

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return (space is not None and space.type == "NODE_EDITOR" and
                space.tree_type == "GeometryNodeTree" and space.edit_tree is not None)

    def execute(self, context):
        node_tree = context.space_data.edit_tree
        try:
            created = build_quick_crowd_ragdoll(node_tree, realize_output=self.realize_output)
            if created is not None and not created.get("output_linked", True):
                self.report(
                    {"WARNING"},
                    "The crowd tree was created but no Geometry output socket exists. Add one to "
                    "the group and connect 'Skin Crowd'",
                )
        except Exception as exc:  # noqa: BLE001
            self.report({"ERROR"}, f"Could not build Quick Crowd Ragdoll: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Crowd ragdoll template created")
        return {"FINISHED"}


class DILAPIDO_OT_auto_arrange(Operator):
    """Arrange the selected nodes (or every node) in the current geometry node tree"""

    bl_idname = "dilapido.auto_arrange"
    bl_label = "Auto Arrange"
    bl_description = (
        "Arrange selected nodes, or all nodes when nothing is selected. Uses the Node Arrange "
        "extension when available and a built-in layered layout otherwise"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return (space is not None and space.type == "NODE_EDITOR" and
                space.tree_type == "GeometryNodeTree" and space.edit_tree is not None)

    def execute(self, context):
        node_tree = context.space_data.edit_tree
        used_extension = _arrange_nodes(node_tree)
        if used_extension:
            self.report({"INFO"}, "Nodes arranged")
        else:
            self.report({"INFO"}, "Nodes arranged with the built-in layout")
        return {"FINISHED"}


class DILAPIDO_OT_update_selected_node_group(Operator):
    """Update the asset node group used by the selected group node(s)"""

    bl_idname = "dilapido.update_selected_node_group"
    bl_label = "Update Selected Node Group"
    bl_description = (
        "Refresh the asset behind the selected group node: linked copies reload their library, "
        "appended copies are replaced with a fresh copy from the installed asset. Connections "
        "are restored by socket name"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        if not (space is not None and space.type == "NODE_EDITOR" and
                space.tree_type == "GeometryNodeTree" and space.edit_tree is not None):
            return False
        return bool(_selected_group_nodes(context))

    def execute(self, context):
        groups = []
        seen = set()
        for node in _selected_group_nodes(context):
            group = node.node_tree
            key = group.as_pointer()
            if key in seen:
                continue
            seen.add(key)
            groups.append(group)
        if not groups:
            self.report({"WARNING"}, "Select a node that instantiates an asset node group")
            return {"CANCELLED"}

        _ASSET_FILE_CACHE.clear()
        linked, appended, skipped = update_node_groups(groups)
        if linked + appended == 0:
            self.report({"WARNING"},
                        "No installed asset was found for the selected node group(s)")
            return {"CANCELLED"}
        message = f"Updated {appended} appended and {linked} linked node group(s)"
        if skipped:
            message += f" ({skipped} without an installed asset)"
        self.report({"INFO"}, message)
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
        column.operator("dilapido.quick_crowd_ragdoll", icon="PHYSICS")
        column.separator()
        column.operator("dilapido.update_selected_node_group", icon="FILE_REFRESH")
        column.operator("dilapido.auto_arrange", icon="NODETREE")
        layout.label(text="Quick RBD: fracture, clusters, constraints and the solver.", icon="INFO")
        layout.label(text="Quick Crowd Walk: states, target motion, separation and skinning.",
                     icon="INFO")
        layout.label(text="Quick Crowd Ragdoll: the walk crowd plus a shared ragdoll switch.",
                     icon="INFO")
        layout.label(text="Ragdoll colliders/floor are not created: add them to the world bundle.",
                     icon="INFO")
        layout.label(text="Update Selected Node Group: refresh an asset node after it changed.",
                     icon="INFO")
        layout.label(text="Auto Arrange: selected nodes, or all nodes when nothing is selected.",
                     icon="INFO")


classes = (
    DILAPIDO_OT_quick_rbd,
    DILAPIDO_OT_quick_crowd_walk,
    DILAPIDO_OT_quick_crowd_ragdoll,
    DILAPIDO_OT_auto_arrange,
    DILAPIDO_OT_update_selected_node_group,
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





