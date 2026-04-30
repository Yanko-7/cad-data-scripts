import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import trimesh
from PIL import Image
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location

def load_step(path: str):
    reader = STEPControl_Reader()
    reader.ReadFile(path)
    reader.TransferRoots()
    return reader.OneShape()

def tessellate(shape, deflection=0.1):
    BRepMesh_IncrementalMesh(shape, deflection).Perform()
    verts, tris = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri:
            offset = len(verts)
            verts += [[tri.Node(i).X(), tri.Node(i).Y(), tri.Node(i).Z()]
                      for i in range(1, tri.NbNodes() + 1)]
            tris += [[offset + t.Get()[j] - 1 for j in range(3)]
                     for t in [tri.Triangle(i) for i in range(1, tri.NbTriangles() + 1)]]
        explorer.Next()
    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.int32)

def render_step(step_path: str, output_path: str, width=800, height=600):
    shape = load_step(step_path)
    verts, faces = tessellate(shape)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    bounds = mesh.bounds
    center = mesh.centroid
    dist = np.linalg.norm(bounds[1] - bounds[0]) * 2
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = center + [0, 0, dist]
    scene.add(camera, pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0))

    renderer = pyrender.OffscreenRenderer(width, height)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    Image.fromarray(color).save(output_path)

if __name__ == "__main__":
    render_step("零件28_processed.stp", "output.png")