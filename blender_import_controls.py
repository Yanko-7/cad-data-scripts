"""
Blender script: import output_controls.npz and visualize control nets.
- Each face / edge gets a distinct HSV color.
- Control points are rendered as icospheres.
- Control cage / polygon lines are rendered as beveled curves (tubes).

Usage (Blender Scripting editor):
    1. Set NPZ_PATH to the absolute path of output_controls.npz
    2. Run Script

Usage (headless):
    blender --python blender_import_controls.py
"""

import colorsys
import numpy as np
import bpy
import bmesh

NPZ_PATH    = "C:\\Users\\Yanko\\Desktop\\candidate\\output_controls.npz"
SPHERE_R    = 0.03   # control-point sphere radius  (normalised space ≈ [-1, 1])
SPHERE_SUB  = 2      # icosphere subdivisions: 1 → 20 tris, 2 → 80 tris, 3 → 320 tris
WIRE_BEVEL  = 0.005  # half-thickness of wireframe tubes

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def hsv_palette(n, s=0.75, v=0.90):
    return [colorsys.hsv_to_rgb(i / max(n, 1), s, v) for i in range(n)]


def make_material(name, rgb):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf:
        bsdf.inputs[0].default_value = (*rgb, 1.0)   # Base Color (index 0, language-independent)
        bsdf.inputs["Roughness"].default_value = 0.35
    mat.diffuse_color = (*rgb, 1.0)   # viewport solid / material-preview color
    return mat


def clear_collection(name):
    col = bpy.data.collections.get(name)
    if col:
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col)
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


_sphere_proto = None   # shared sphere mesh (one copy in memory, instanced everywhere)


def _get_sphere_proto():
    global _sphere_proto
    if _sphere_proto is None:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=SPHERE_SUB, radius=SPHERE_R)
        _sphere_proto = bpy.data.meshes.new("_CtrlPt_Sphere_Proto")
        bm.to_mesh(_sphere_proto)
        bm.free()
        _sphere_proto.materials.append(None)   # one slot → per-object colour override
        for poly in _sphere_proto.polygons:
            poly.use_smooth = True
        _sphere_proto.update()
    return _sphere_proto


def make_spheres_obj(name, pts, mat, col):
    # Parent: point-cloud mesh (one vertex per control point)
    cloud_me = bpy.data.meshes.new(name + "_cloud")
    cloud_me.from_pydata([p.tolist() for p in pts], [], [])
    cloud_me.update()
    cloud_obj = bpy.data.objects.new(name + "_pts", cloud_me)
    cloud_obj.instance_type = "VERTS"   # render child at every vertex
    col.objects.link(cloud_obj)

    # Child: one sphere object shared across all points, coloured per net
    sphere_obj = bpy.data.objects.new(name + "_sphere", _get_sphere_proto())
    sphere_obj.parent = cloud_obj
    col.objects.link(sphere_obj)
    sphere_obj.material_slots[0].link = "OBJECT"   # override mesh-level slot
    sphere_obj.material_slots[0].material = mat

    return cloud_obj


def make_lines_obj(name, polylines, mat, col):
    curve_data = bpy.data.curves.new(name + "_lines", "CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = WIRE_BEVEL
    curve_data.bevel_resolution = 3
    for pts in polylines:
        spline = curve_data.splines.new("POLY")
        spline.points.add(len(pts) - 1)
        for i, p in enumerate(pts):
            spline.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
    obj = bpy.data.objects.new(name + "_lines", curve_data)
    obj.data.materials.append(mat)
    col.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

data = np.load(NPZ_PATH)
face_controls = data["face_controls"]   # [F, 4, 4, 4]  (x, y, z, w)
edge_controls = data["edge_controls"]   # [E, 4, 4]

n_faces, n_edges = len(face_controls), len(edge_controls)
print(f"Loaded: {n_faces} face patches, {n_edges} edge curves")

face_colors = hsv_palette(n_faces, s=0.70, v=0.90)
edge_colors  = hsv_palette(n_edges, s=0.50, v=0.80)

# ---------------------------------------------------------------------------
# Face control nets  (4 × 4 grid)
# ---------------------------------------------------------------------------

face_col = clear_collection("ControlNets_Faces")
for fi, patch in enumerate(face_controls):
    pts = patch[:, :, :3].reshape(-1, 3)   # (16, 3)
    mat = make_material(f"Mat_Face_{fi:04d}", face_colors[fi])
    make_spheres_obj(f"Face_{fi:04d}", pts, mat, face_col)

    polylines = []
    for i in range(4):
        polylines.append([pts[i * 4 + j] for j in range(4)])   # row i
        polylines.append([pts[j * 4 + i] for j in range(4)])   # col i
    make_lines_obj(f"Face_{fi:04d}", polylines, mat, face_col)

# ---------------------------------------------------------------------------
# Edge control polygons  (4-point polyline)
# ---------------------------------------------------------------------------

edge_col = clear_collection("ControlNets_Edges")
for ei, curve in enumerate(edge_controls):
    pts = curve[:, :3]   # (4, 3)
    mat = make_material(f"Mat_Edge_{ei:04d}", edge_colors[ei])
    make_spheres_obj(f"Edge_{ei:04d}", pts, mat, edge_col)
    make_lines_obj(f"Edge_{ei:04d}", [pts], mat, edge_col)

print(f"Done. Collections: 'ControlNets_Faces' ({n_faces}), 'ControlNets_Edges' ({n_edges})")
