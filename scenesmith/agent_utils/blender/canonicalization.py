"""Mesh canonicalization implementation for BlenderServer.

This module contains the core bpy logic for mesh canonicalization that runs
inside the BlenderServer subprocess. It is called via the /canonicalize HTTP
endpoint.
"""

import logging

from pathlib import Path

console_logger = logging.getLogger(__name__)


def choose_fallback_front_axis(up_axis: str) -> str:
    """Choose a sensible fallback front axis when VLM predicts parallel axes.

    Args:
        up_axis: The up axis in Blender coordinates (e.g., "+Z", "-Y").

    Returns:
        A perpendicular front axis in Blender coordinates.
    """
    # Extract the axis letter (last character after optional sign).
    axis_letter = up_axis[-1].upper()

    # Choose perpendicular axis based on up axis.
    if axis_letter == "Z":
        # If up is ±Z, use +Y as front (most common for furniture).
        return "+Y"
    elif axis_letter == "Y":
        # If up is ±Y, use +X as front.
        return "+X"
    else:  # axis_letter == "X"
        # If up is ±X, use +Y as front.
        return "+Y"


def canonicalize_mesh_impl(
    input_path: Path,
    output_path: Path,
    up_axis: str,
    front_axis: str,
    object_type: str = "furniture",
) -> Path:
    """Canonicalize GLTF mesh to standard orientation using Blender.

    This is the server-side implementation that runs inside BlenderServer.

    Pipeline:
    - Input: Y-up GLTF (from convert_glb_to_gltf with export_yup=True)
    - Blender imports and converts Y-up GLTF to Z-up internally
    - VLM axes (analyzed in Blender's Z-up space) can be used directly
    - Rotate mesh to canonical orientation in Blender's Z-up space
    - Export as Y-up GLTF (Drake will convert Y-up to Z-up on load)

    Target orientation in Blender (Z-up):
    - Up direction → +Z axis
    - Front direction → +Y axis
    - Right direction → +X axis

    Object-type-specific placement in Blender (Z-up):
    - furniture/manipuland: Bottom at z=0, centered in XY
    - ceiling_mounted: Top at z=0, centered in XY
    - wall_mounted: Min Y at y=0, centered in XZ

    Args:
        input_path: Path to input Y-up GLTF file.
        output_path: Path where canonicalized GLTF will be saved.
        up_axis: Up axis in Blender coordinates (e.g., "+Z", "-Y").
        front_axis: Front axis in Blender coordinates (e.g., "+Y", "+X").
        object_type: Type of object (determines placement strategy).
            One of: "furniture", "manipuland", "wall_mounted", "ceiling_mounted".

    Returns:
        Path to the canonicalized GLTF file.

    Raises:
        FileNotFoundError: If input GLTF file doesn't exist.
        RuntimeError: If Blender processing fails.
    """
    import bpy
    import mathutils

    if not input_path.exists():
        raise FileNotFoundError(f"GLTF file not found: {input_path}")

    console_logger.info(
        f"Canonicalizing GLTF from {input_path} "
        f"(up={up_axis}, front={front_axis}, type={object_type})"
    )

    # Clear scene.
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Import GLTF (Blender converts Y-up → Z-up automatically).
    bpy.ops.import_scene.gltf(filepath=str(input_path))

    # Find root object(s).
    top_level_objects = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    if not top_level_objects:
        raise RuntimeError("No top-level objects found after GLTF import")

    # Use first root object with children (or first object if none have children).
    root_obj = next(
        (obj for obj in top_level_objects if obj.children), top_level_objects[0]
    )

    # Make it active.
    bpy.context.view_layer.objects.active = root_obj

    # Apply transforms.
    bpy.ops.object.select_all(action="DESELECT")
    root_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Parse axes (VLM axes are in Blender coordinates).
    def axis_to_vector(axis_str: str) -> mathutils.Vector:
        """Convert axis string to Blender Vector."""
        sign = -1 if axis_str.startswith("-") else 1
        base = axis_str.lstrip("-+")
        if base.upper() == "X":
            return mathutils.Vector((sign, 0, 0))
        elif base.upper() == "Y":
            return mathutils.Vector((0, sign, 0))
        elif base.upper() == "Z":
            return mathutils.Vector((0, 0, sign))
        else:
            raise ValueError(f"Invalid axis string: {axis_str}")

    up = axis_to_vector(up_axis)
    front = axis_to_vector(front_axis)

    # Check if up and front are parallel.
    dot_product = abs(up.dot(front))
    if dot_product > 0.99:
        original_front = front_axis
        front_axis = choose_fallback_front_axis(up_axis)
        console_logger.warning(
            f"VLM predicted parallel up and front axes (up={up_axis}, "
            f"front={original_front}). Using fallback front axis: {front_axis}"
        )
        front = axis_to_vector(front_axis)

    # Compute right axis via cross product, then recompute front to ensure
    # right-handed coordinate system.
    right = front.cross(up)

    if right.length < 1e-6:
        # Handle parallel axes with fallback.
        if abs(up.x) < 0.99:
            right = up.cross(mathutils.Vector((1, 0, 0)))
        else:
            right = up.cross(mathutils.Vector((0, 1, 0)))

    right.normalize()
    front = up.cross(right)
    front.normalize()

    # Build rotation matrix to align object to canonical orientation.
    #
    # `up_axis` / `front_axis` come from VLM analysis in Blender's Z-up space and
    # describe the mesh's CURRENT orientation (i.e., which world direction is "up"
    # and which is "front" for the imported asset).
    #
    # Canonical target orientation in Blender (Z-up) is:
    # - right  → +X
    # - front  → +Y
    # - up     → +Z
    #
    # Therefore we need a rotation that maps the VLM-predicted basis to the
    # canonical basis (not the other way around).
    canonical_right = mathutils.Vector((1, 0, 0))
    canonical_front = mathutils.Vector((0, 1, 0))
    canonical_up = mathutils.Vector((0, 0, 1))

    current_matrix = mathutils.Matrix((right, front, up)).transposed()
    target_matrix = mathutils.Matrix(
        (canonical_right, canonical_front, canonical_up)
    ).transposed()

    # Rotation matrix: target @ current^-1
    rotation_matrix = target_matrix @ current_matrix.inverted()
    rotation_matrix = rotation_matrix.to_4x4()

    # Apply rotation.
    root_obj.matrix_world = rotation_matrix @ root_obj.matrix_world
    # CRITICAL: Blender's depsgraph is lazy — setting ``root_obj.matrix_world``
    # does not immediately propagate to descendants' ``matrix_world``. Without
    # this update, the snapshot loop below reads stale (pre-rotation) child
    # world matrices, the bake writes un-rotated vertices, and the rotation
    # gets discarded entirely (any ``front_axis`` produces identical output).
    bpy.context.view_layer.update()
    # CRITICAL: ``bpy.ops.object.transform_apply`` does NOT reliably bake a
    # parent Empty's rotation into descendant mesh vertices. When called on
    # an Empty parent in isolation, the rotation gets pushed down into each
    # child's local transform (and then exports as a 90°-off node rotation
    # in the resulting GLTF — visible as chairs sitting tangent/sideways to
    # the dining table). Even selecting all descendants doesn't help reliably
    # because the iteration order of transform_apply is not guaranteed.
    #
    # Manually bake the world rotation into the vertex coordinates of every
    # descendant mesh, then zero out the local transform of every node so the
    # baked vertex coordinates ARE the final world positions.
    #
    # Assigning ``matrix_world = Identity`` on a parented child does NOT zero
    # the local transform: Blender back-solves ``matrix_local`` against the
    # (still-rotated) parent, leaving an inverse rotation in the local node.
    # The GLTF exporter then writes that inverse rotation on the node and the
    # baked rotation gets undone on re-import (visible to Blender) while
    # ``trimesh.load`` ignores node transforms and sees the baked rotation —
    # producing a 90° offset between support-surface extraction and renders.
    #
    # Snapshot every world matrix BEFORE mutating anything: resetting the
    # parent first would cause children's auto-computed ``matrix_world`` to
    # lose the rotation before we read it.
    nodes = [root_obj, *root_obj.children_recursive]
    world_matrices = {obj: obj.matrix_world.copy() for obj in nodes}
    for obj, world_mat in world_matrices.items():
        if obj.type == "MESH":
            for vert in obj.data.vertices:
                vert.co = world_mat @ vert.co
            obj.data.update()
    for obj in nodes:
        obj.matrix_basis = mathutils.Matrix.Identity(4)
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)

    # Compute bounding box in world coordinates (from all mesh children).
    all_mesh_verts_world = []
    for obj in root_obj.children_recursive:
        if obj.type == "MESH":
            all_mesh_verts_world.extend(
                [obj.matrix_world @ v.co for v in obj.data.vertices]
            )

    # If root is also a mesh, include its vertices.
    if root_obj.type == "MESH":
        all_mesh_verts_world.extend(
            [root_obj.matrix_world @ v.co for v in root_obj.data.vertices]
        )

    if not all_mesh_verts_world:
        raise RuntimeError("No mesh vertices found for bounding box computation")

    xs = [v.x for v in all_mesh_verts_world]
    ys = [v.y for v in all_mesh_verts_world]
    zs = [v.z for v in all_mesh_verts_world]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    # Apply object-type-specific placement logic.
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    object_type_lower = object_type.lower()
    if object_type_lower in ("furniture", "manipuland"):
        # Center x/y, bottom at z=0 (object just above ground).
        loc_x = -center_x
        loc_y = -center_y
        loc_z = -min_z
    elif object_type_lower == "ceiling_mounted":
        # Center x/y, top at z=0 (object just below ceiling).
        loc_x = -center_x
        loc_y = -center_y
        loc_z = -max_z
    elif object_type_lower == "wall_mounted":
        # Center x/z, min_y at y=0 (object just touches wall).
        center_z = (min_z + max_z) / 2
        loc_x = -center_x
        loc_y = -min_y
        loc_z = -center_z
    else:
        raise NotImplementedError(f"Placement for {object_type} not implemented")

    root_obj.location = mathutils.Vector((loc_x, loc_y, loc_z))
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    # Merge duplicate vertices to reduce file size (~80% reduction).
    # Select all mesh objects and merge vertices by distance.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

    if bpy.context.selected_objects:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.object.mode_set(mode="OBJECT")
        console_logger.debug("Merged duplicate vertices in mesh")

    # Ensure output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export as Y-up GLTF for Drake (Drake will convert Y-up to Z-up on load).
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLTF_SEPARATE",
        use_selection=False,
    )

    console_logger.info(f"Canonicalized GLTF saved to {output_path}")

    return output_path
