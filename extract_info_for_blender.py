import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from utils import load_and_filter_step, preprocess_shape, extract_bicubic_features_dir


def shape_to_obj(shape, out_path, center, scale, linear_deflection=0.01):
    BRepMesh_IncrementalMesh(shape, linear_deflection).Perform()
    vertices, triangles, offset = [], [], 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri:
            trsf = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                vertices.append([p.X(), p.Y(), p.Z()])
            for i in range(1, tri.NbTriangles() + 1):
                n1, n2, n3 = tri.Triangle(i).Get()
                triangles.append([n1 - 1 + offset, n2 - 1 + offset, n3 - 1 + offset])
            offset += tri.NbNodes()
        exp.Next()

    verts = (np.array(vertices, dtype=np.float32) - center) * scale
    with open(out_path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in triangles:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
    print(f"OBJ saved: {out_path}  ({len(verts)} verts, {len(triangles)} tris)")


step_path = "FreeCAD_Rookies_057-Body.step"
shape = load_and_filter_step(step_path)
shape = preprocess_shape(shape)
features = extract_bicubic_features_dir(shape)

center = features["center"]
scale = features["scale"]

shape_to_obj(shape, "output.obj", center, scale)

np.savez_compressed(
    "output_controls.npz",
    face_controls=features["face_controls"],
    edge_controls=features["edge_controls"],
    outer_edge_indices=features["outer_edge_indices"],
    face_outer_offsets=features["face_outer_offsets"],
    inner_edge_indices=features["inner_edge_indices"],
    inner_loop_offsets=features["inner_loop_offsets"],
    face_inner_offsets=features["face_inner_offsets"],
)
print(
    f"NPZ saved: output_controls.npz  "
    f"({len(features['face_controls'])} faces, {len(features['edge_controls'])} edges)"
)
