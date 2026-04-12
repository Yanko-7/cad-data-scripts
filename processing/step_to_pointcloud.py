import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import TimeoutError
from pathlib import Path

import numpy as np
import trimesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Extend.DataExchange import read_step_file
from pebble import ProcessExpired, ProcessPool
from tqdm import tqdm

NUM_POINTS = 4096
MAX_WORKERS = 80
TIMEOUT = 120


def shape_to_mesh(shape):
    BRepMesh_IncrementalMesh(shape, True).Perform()
    vertices, triangles = [], []
    offset = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri:
            tf = loc.Transformation()
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(tf)
                vertices.append([p.X(), p.Y(), p.Z()])
            for i in range(1, tri.NbTriangles() + 1):
                n1, n2, n3 = tri.Triangle(i).Get()
                triangles.append([n1 - 1 + offset, n2 - 1 + offset, n3 - 1 + offset])
            offset += tri.NbNodes()
        explorer.Next()
    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)


def process_one(step_path: str, out_dir: str, num_points: int = NUM_POINTS) -> bool:
    try:
        shape = read_step_file(step_path)
        verts, faces = shape_to_mesh(shape)
        if len(faces) == 0:
            return False
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        out_path = Path(out_dir) / f"{Path(step_path).stem}.ply"
        trimesh.PointCloud(pts).export(str(out_path), file_type="ply")
        return True
    except Exception as e:
        print(f"[Error] {step_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert STEP files to point cloud PLY files")
    parser.add_argument("-i", "--input", required=True, help="Input directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-n", "--num-points", type=int, default=NUM_POINTS, help="Points per cloud")
    parser.add_argument("-w", "--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--split", type=str, default=None, help="JSON split file with {split_key: [stem, ...]}")
    parser.add_argument("--split-key", type=str, default=None, help="Key in split JSON (e.g. train/test). If omitted, use all splits.")
    args = parser.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = args.num_points

    step_index = defaultdict(list)
    for p in in_dir.rglob("*"):
        if p.suffix.lower() in (".step", ".stp"):
            key = re.sub(r"_step_\d+$", "", p.stem)
            step_index[key].append(str(p))

    if args.split:
        with open(args.split) as f:
            split_data = json.load(f)
        if args.split_key:
            keys = [args.split_key]
        else:
            keys = list(split_data.keys())
        tasks = []
        for key in keys:
            sub_dir = out_dir if args.split_key else out_dir / key
            sub_dir.mkdir(parents=True, exist_ok=True)
            for s in split_data[key]:
                for fp in step_index.get(s, []):
                    tasks.append((fp, str(sub_dir)))
        tasks.sort()
        step_files = [t[0] for t in tasks]
        out_dirs = [t[1] for t in tasks]
        print(f"Split filter: {len(step_files)} files from {sum(len(split_data[k]) for k in keys)} stems in {in_dir}")
    else:
        step_files = sorted([f for fs in step_index.values() for f in fs])
        out_dirs = [str(out_dir)] * len(step_files)
        print(f"Found {len(step_files)} STEP files in {in_dir}")

    ok = 0
    with ProcessPool(max_workers=args.workers, max_tasks=1) as pool:
        future = pool.map(process_one, step_files, out_dirs, [n] * len(step_files), timeout=TIMEOUT)
        it = future.result()
        pbar = tqdm(total=len(step_files), desc="STEP → PLY")
        while True:
            try:
                if next(it):
                    ok += 1
            except StopIteration:
                break
            except (TimeoutError, ProcessExpired, Exception) as e:
                print(f"\n[Warning] {e}")
            pbar.update(1)
        pbar.close()

    print(f"Done: {ok}/{len(step_files)} converted → {out_dir}")


if __name__ == "__main__":
    main()
