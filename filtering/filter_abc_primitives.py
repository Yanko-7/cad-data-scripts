import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import numpy as np
from tqdm import tqdm

MAX_FACE = 50
PER_FACE_EDGE_LIMIT = 30
TOTAL_EDGE_LIMIT = 1000
BBOX_THRES = 1 / (2 ** (10 - 1))
BREPGEN_THRES = 0.05 / 3


def point_bbox(pts: np.ndarray) -> np.ndarray:
    flat = pts.reshape(-1, 3)
    if not np.isfinite(flat).all():
        raise ValueError("invalid_nan_or_inf_points")
    vmin, vmax = flat.min(axis=0), flat.max(axis=0)
    # if (vmax - vmin).max() < BBOX_THRES:
    #     raise ValueError("degenerate_geometry_zero_span")
    return np.concatenate([vmin, vmax])


def check_topology(outer_edges, face_outer_offsets, inner_edges, inner_loop_offsets, face_inner_offsets):
    num_faces = len(face_outer_offsets) - 1
    if not (0 < num_faces <= MAX_FACE):
        return False, [], "invalid_face_count"
    num_edges = int(max(np.max(outer_edges, initial=-1), np.max(inner_edges, initial=-1))) + 1
    if num_edges > TOTAL_EDGE_LIMIT:
        return False, [], "exceeds_total_edge_limit"
    edge_to_faces = [[] for _ in range(num_edges)]
    face_edges_adj = []
    for f_id in range(num_faces):
        e_ids = outer_edges[face_outer_offsets[f_id]:face_outer_offsets[f_id + 1]].tolist()
        for l_idx in range(face_inner_offsets[f_id], face_inner_offsets[f_id + 1]):
            e_ids.extend(inner_edges[inner_loop_offsets[l_idx]:inner_loop_offsets[l_idx + 1]])
        if len(e_ids) > PER_FACE_EDGE_LIMIT:
            return False, [], "exceeds_per_face_edge_limit"
        face_edges_adj.append(e_ids)
        for e_id in set(e_ids):
            edge_to_faces[e_id].append(f_id)
    if not all(len(faces) in (0, 2) for faces in edge_to_faces):
        return False, [], "non_manifold_edges"
    return True, face_edges_adj, "ok"


def has_duplicate_bboxes(bboxes: np.ndarray) -> bool:
    if len(bboxes) < 2:
        return False
    diffs = np.max(np.abs(bboxes[:, None, :] - bboxes[None, :, :]), axis=-1)
    np.fill_diagonal(diffs, np.inf)
    return bool((diffs < BREPGEN_THRES).any())


def build_id_to_path(root_dir: str, valid_ids: set, ext: str = ".npz") -> dict:
    lengths = sorted({len(v) for v in valid_ids}, reverse=True)
    result = {}
    for r, _, files in os.walk(root_dir):
        for f in files:
            if not f.endswith(ext):
                continue
            stem = f[: -len(ext)]
            for length in lengths:
                if len(stem) >= length and stem[:length] in valid_ids:
                    fid = stem[:length]
                    if fid not in result:
                        result[fid] = os.path.join(r, f)
                    break
    return result


def check_npz(path: str) -> Tuple[str, bool, str]:
    try:
        with np.load(path) as data:
            if len(data["face_outer_offsets"]) - 1 != len(data["face_points"]):
                return path, False, "mismatch_face_points"
            is_valid, face_edges_adj, reason = check_topology(
                data["outer_edge_indices"], data["face_outer_offsets"],
                data["inner_edge_indices"], data["inner_loop_offsets"],
                data["face_inner_offsets"],
            )
            if not is_valid:
                return path, False, reason
            try:
                face_bboxes = np.array([point_bbox(fp) for fp in data["face_points"]])
                edge_bboxes = np.array([point_bbox(ep) for ep in data["edge_points"]])
            except ValueError as e:
                return path, False, str(e)
            if has_duplicate_bboxes(face_bboxes):
                return path, False, "duplicate_face_bboxes"
            for e_ids in face_edges_adj:
                if not e_ids:
                    return path, False, "empty_face_edges"
                if has_duplicate_bboxes(edge_bboxes[e_ids]):
                    return path, False, "duplicate_edge_bboxes"
            return path, True, "passed"
    except Exception as e:
        return path, False, f"load_error: {type(e).__name__}"


def run_stats(
    split_json: str,
    npz_root: str,
    output_json: str = None,
    max_workers: int = 100,
):
    with open(split_json) as f:
        split_ids = json.load(f)

    all_ids = {fid for ids in split_ids.values() for fid in ids}
    print(f"Building NPZ path map for {len(all_ids)} IDs...")
    id_to_path = build_id_to_path(npz_root, all_ids)
    print(f"Matched {len(id_to_path)} / {len(all_ids)} NPZ files.\n")

    # build reverse map: path -> fid
    path_to_fid = {v: k for k, v in id_to_path.items()}

    total_stats = Counter()
    filtered = {}
    for split, ids in split_ids.items():
        paths = [id_to_path[fid] for fid in ids if fid in id_to_path]
        stats = Counter()
        passed_ids = []
        futures_map = {}
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures_map = {ex.submit(check_npz, p): p for p in paths}
            for future in tqdm(as_completed(futures_map), total=len(futures_map), desc=split):
                path, ok, reason = future.result()
                stats[reason] += 1
                if ok:
                    passed_ids.append(path_to_fid[path])
        passed = stats.pop("passed", 0)
        print(f"[{split}] passed={passed} / {len(paths)}")
        for reason, count in stats.most_common():
            print(f"  {reason}: {count}")
        total_stats += stats
        total_stats["passed"] += passed
        filtered[split] = sorted(passed_ids)

    print("\n=== Total ===")
    print(f"passed: {total_stats.pop('passed', 0)}")
    for reason, count in total_stats.most_common():
        print(f"  {reason}: {count}")

    if output_json:
        with open(output_json, "w") as f:
            json.dump(filtered, f, indent=2)
        print(f"\nSaved to: {output_json}")


if __name__ == "__main__":
    run_stats(
        split_json="ABC-dataset/abc_data_split_6bit.json",
        npz_root="/cache/yanko/dataset/abc_primitives_npz",
        output_json="configs/filtered_abc_primitives.json",
        max_workers=100,
    )
