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
from OCC.Extend.DataExchange import read_step_file, write_ply_file


def process_single_file(step_path, out_dir):
    try:
        # 读取 STEP 文件并转换为网格
        origin_shape = read_step_file(str(step_path))
        verts, faces = shape2mesh(origin_shape)
        if len(faces) > 0:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            points, _ = trimesh.sample.sample_surface(mesh, 4096)

            point_cloud = trimesh.PointCloud(points)

            out_path = Path(out_dir) / f"{Path(step_path).stem}.ply"
            point_cloud.export(str(out_path), file_type="ply")
    except Exception as e:
        print(f"Error processing {step_path}: {e}")
        return False
    return True


def load_dataset_fast(
    json_path: str, root_dir: str, ext: str = ".npz"
) -> dict[str, list[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    p2s = {p: split for split, prefixes in data.items() for p in prefixes}
    lengths = sorted({len(p) for p in p2s}, reverse=True)

    result = {split: [] for split in data}

    for r, _, files in os.walk(root_dir):
        for f in filter(lambda x: x.endswith(ext), files):
            for length in lengths:
                if len(f) >= length and (prefix := f[:length]) in p2s:
                    result[p2s[prefix]].append(os.path.join(r, f))
                    break

    return result



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


def run_pool(step_files, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    with ProcessPool(max_workers=120, max_tasks=1) as pool:
        future = pool.map(
            process_single_file, step_files, [out_dir] * len(step_files), timeout=120
        )
        iterator = future.result()
        pbar = tqdm(total=len(step_files), desc=f"Converting to {out_dir.name}")
        while True:
            try:
                result = next(iterator)
                if result:
                    success_count += 1
                pbar.update(1)
            except StopIteration:
                break
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
    return success_count


def main(args):
    in_dir, out_dir = Path(args.input), Path(args.output)

    if args.split:
        splits = load_dataset_fast(args.split, str(in_dir), ext=".step")
        total_success = 0
        for split_name, step_files in splits.items():
            print(f"[{split_name}] {len(step_files)} files")
            n = run_pool(step_files, out_dir / split_name)
            print(f"[{split_name}] Done: {n}/{len(step_files)}")
            total_success += n
        total = sum(len(v) for v in splits.values())
        print(f"\nAll splits done: {total_success}/{total}")
    else:
        step_files = list(in_dir.glob("*.step"))
        print(f"Found {len(step_files)} STEP files in {in_dir}")
        n = run_pool(step_files, out_dir)
        print(f"Done! Successfully converted {n}/{len(step_files)} files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help="Input directory containing STEP files"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for PLY files"
    )
    parser.add_argument(
        "-s", "--split", default=None, help="JSON split file for train/test/val"
    )
    main(parser.parse_args())
