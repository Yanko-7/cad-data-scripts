import json
import os
from pebble import ProcessExpired, ProcessPool
import trimesh
import argparse
from pathlib import Path
from tqdm import tqdm
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import (
    TopAbs_FACE,
)
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Extend.DataExchange import read_step_file


def process_single_file(step_path, out_dir):
    try:
        # 读取 STEP 文件并转换为网格
        origin_shape = read_step_file(str(step_path))
        verts, faces = shape2mesh(origin_shape)
        if len(faces) > 0:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            points, _ = trimesh.sample.sample_surface(mesh, 2000)

            point_cloud = trimesh.PointCloud(points)

            out_path = Path(out_dir) / f"{Path(step_path).stem}.ply"
            point_cloud.export(str(out_path), file_type="ply")
    except Exception as e:
        print(f"Error processing {step_path}: {e}")
        return False
    return True


def load_dataset_fast(json_path, root_dir, ext=".npz"):
    file_map = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(ext):
                file_map[f[:8]] = os.path.join(root, f)

    with open(json_path, "r") as f:
        data = json.load(f)

    return {
        split: [
            file_map[Path(item).stem[:8]]
            for item in items
            if Path(item).stem[:8] in file_map
        ]
        for split, items in data.items()
    }


def shape2mesh(shape):

    mesh = BRepMesh_IncrementalMesh(shape, True)
    mesh.Perform()

    vertices, triangles = [], []
    offset = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = topods.Face(explorer.Current())
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation:
            transf = loc.Transformation()
            for i in range(1, triangulation.NbNodes() + 1):
                pnt = triangulation.Node(i).Transformed(transf)
                vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
            for i in range(1, triangulation.NbTriangles() + 1):
                n1, n2, n3 = triangulation.Triangle(i).Get()
                triangles.append([n1 - 1 + offset, n2 - 1 + offset, n3 - 1 + offset])
            offset += triangulation.NbNodes()
        explorer.Next()
    return np.array(vertices, dtype=np.float32), np.array(triangles, dtype=np.int32)


def find_step_paths(json_path: str, step_root: str, split: str) -> list:
    with open(json_path) as f:
        ids = json.load(f)[split]
    valid_ids = set(ids)
    lengths = sorted({len(v) for v in valid_ids}, reverse=True)
    result = {}
    for r, _, files in os.walk(step_root):
        for f in files:
            if not f.endswith(".step"):
                continue
            stem = f[:-5]
            for length in lengths:
                if len(stem) >= length and stem[:length] in valid_ids:
                    fid = stem[:length]
                    if fid not in result:
                        result[fid] = Path(r) / f
                    break
    return list(result.values())


def main(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.json:
        step_files = find_step_paths(args.json, args.step_root, args.split)
        print(f"Found {len(step_files)} STEP files from JSON ({args.split} split)")
    else:
        step_files = list(Path(args.input).glob("*.step"))
        print(f"Found {len(step_files)} STEP files in {args.input}")

    success_count = 0
    with ProcessPool(max_workers=120, max_tasks=1) as pool:
        future = pool.map(
            process_single_file, step_files, [out_dir] * len(step_files), timeout=120
        )
        iterator = future.result()

        pbar = tqdm(total=len(step_files), desc="Converting to PLY")
        while True:
            try:
                # 逐个获取结果，隔离崩溃
                result = next(iterator)
                if result:
                    success_count += 1
                pbar.update(1)
            except StopIteration:
                break  # 队列处理完毕
            except TimeoutError:
                print("\n[Warning] OpenCASCADE Timeout (Infinite Loop). Task killed.")
                pbar.update(1)
            except ProcessExpired as e:
                print(f"\n[Warning] C++ Segfault! Worker crashed: {e}")
                pbar.update(1)
            except Exception as e:
                print(f"\n[Warning] Python Exception: {e}")
                pbar.update(1)
        pbar.close()

    print(
        f"Done! Successfully converted {success_count}/{len(step_files)} files to {out_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--input", help="Input directory containing STEP files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for PLY files")
    source.add_argument("--json", default=None, help="Filtered JSON path (use instead of --input)")
    parser.add_argument("--step_root", default="/cache/yanko/dataset/abc-origin/", help="STEP files root dir")
    parser.add_argument("--split", default="test", help="Split to process (train/val/test)")
    main(parser.parse_args())
