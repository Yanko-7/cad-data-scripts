import bpy
import math
import sys
import argparse
import numpy as np
from pathlib import Path


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_scene(scene, resolution):
    engine = "BLENDER_EEVEE_NEXT" if bpy.app.version[0] >= 4 else "BLENDER_EEVEE"
    scene.render.engine = engine
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    scene.world = world


def import_obj(obj_path: str):
    try:
        bpy.ops.wm.obj_import(filepath=obj_path)        # Blender 3.3+
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=obj_path)     # Blender < 3.3
    meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    return meshes[0] if meshes else None


def normalize_object(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    max_dim = max(obj.dimensions)
    if max_dim > 0:
        s = 2.0 / max_dim
        obj.scale = (s, s, s)
        bpy.ops.object.transform_apply(scale=True)


def apply_flat_material(obj):
    mat = bpy.data.materials.new("FlatGray")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (0.75, 0.75, 0.75, 1.0)
    emit.inputs["Strength"].default_value = 1.0
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def add_camera(eye, scene):
    import mathutils
    for o in list(bpy.data.objects):
        if o.type == "CAMERA":
            bpy.data.objects.remove(o, do_unlink=True)
    cam = bpy.data.cameras.new("Camera")
    cam.lens_unit = "FOV"
    cam.angle = math.radians(45)
    cam_obj = bpy.data.objects.new("Camera", cam)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = mathutils.Vector(eye)
    direction = mathutils.Vector(eye).normalized()
    cam_obj.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam_obj


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    clear_scene()
    scene = bpy.context.scene
    setup_scene(scene, args.resolution)

    obj = import_obj(args.input)
    if obj is None:
        print(f"[Error] Could not import {args.input}")
        sys.exit(1)

    normalize_object(obj)
    apply_flat_material(obj)

    out_dir = Path(args.output)
    stem = Path(args.input).stem
    azimuths = rng.uniform(0, 2 * math.pi, args.n_views)
    elevations = rng.uniform(math.radians(15), math.radians(75), args.n_views)

    for i, (az, el) in enumerate(zip(azimuths, elevations)):
        d = args.distance
        eye = (
            d * math.cos(el) * math.cos(az),
            d * math.cos(el) * math.sin(az),
            d * math.sin(el),
        )
        add_camera(eye, scene)
        scene.render.filepath = str(out_dir / f"{stem}_v{i:02d}.png")
        bpy.ops.render.render(write_still=True)


main()
