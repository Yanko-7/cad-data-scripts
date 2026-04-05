import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

MAX_FACE = 100
PER_FACE_EDGE_LIMIT = 20
TOTAL_EDGE_LIMIT = 1000
BBOX_THRES = 1 / (2 ** (10 - 1))


def load_split_paths(
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


def normalize_points_with_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(points).all():
        raise ValueError("invalid_nan_or_inf_points")

    xyz, w = points[..., :3], points[..., 3:4]
    w_norm = (w - 1) / (w + 1 + 1e-8)

    pts_flat = xyz.reshape(-1, 3)
    vmin, vmax = pts_flat.min(axis=0), pts_flat.max(axis=0)
    span = (vmax - vmin).max()

    if span < BBOX_THRES:
        raise ValueError("degenerate_geometry_zero_span")

    norm_xyz = (xyz - (vmin + vmax) / 2.0) * (2.0 / span)
    return np.concatenate([norm_xyz, w_norm], axis=-1), np.concatenate([vmin, vmax])


def check_topology(
    outer_edges: np.ndarray,
    face_outer_offsets: np.ndarray,
    inner_edges: np.ndarray,
    inner_loop_offsets: np.ndarray,
    face_inner_offsets: np.ndarray,
) -> Tuple[bool, List[List[int]], str]:
    num_faces = len(face_outer_offsets) - 1
    if not (0 < num_faces <= MAX_FACE):
        return False, [], "invalid_face_count"

    num_edges = (
        int(max(np.max(outer_edges, initial=-1), np.max(inner_edges, initial=-1))) + 1
    )
    if num_edges > TOTAL_EDGE_LIMIT:
        return False, [], "exceeds_total_edge_limit"

    edge_to_faces = [[] for _ in range(num_edges)]
    face_edges_adj = []

    for f_id in range(num_faces):
        e_ids = outer_edges[
            face_outer_offsets[f_id] : face_outer_offsets[f_id + 1]
        ].tolist()
        for l_idx in range(face_inner_offsets[f_id], face_inner_offsets[f_id + 1]):
            e_ids.extend(
                inner_edges[inner_loop_offsets[l_idx] : inner_loop_offsets[l_idx + 1]]
            )

        # if len(e_ids) > PER_FACE_EDGE_LIMIT:
        #     return False, [], "exceeds_per_face_edge_limit"

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
    return bool((diffs < BBOX_THRES).any())


def is_ok_file(file_path: str) -> Tuple[str, bool, str]:
    try:
        with np.load(file_path) as data:
            if len(data["face_outer_offsets"]) - 1 != len(data["face_controls"]):
                return file_path, False, "mismatch_face_controls"

            is_valid, face_edges_adj, reason = check_topology(
                data["outer_edge_indices"],
                data["face_outer_offsets"],
                data["inner_edge_indices"],
                data["inner_loop_offsets"],
                data["face_inner_offsets"],
            )
            if not is_valid:
                return file_path, False, reason

            try:
                face_bboxes = np.array(
                    [normalize_points_with_bbox(fc)[1] for fc in data["face_controls"]]
                )
                edge_bboxes = np.array(
                    [normalize_points_with_bbox(ec)[1] for ec in data["edge_controls"]]
                )
            except ValueError as e:
                return file_path, False, str(e)

            if has_duplicate_bboxes(face_bboxes):
                return file_path, False, "duplicate_face_bboxes"

            for e_ids in face_edges_adj:
                if not e_ids:
                    return file_path, False, "empty_face_edges"
                if has_duplicate_bboxes(edge_bboxes[e_ids]):
                    return file_path, False, "duplicate_edge_bboxes"

            return file_path, True, "passed"
    except Exception:
        return file_path, False, "corrupted_or_load_error"


def filter_dataset(
    dataset_paths: Dict[str, List[str]], output_json: str, max_workers: int = None
):
    filtered_paths = {}
    stats = Counter()

    for split, paths in dataset_paths.items():
        if not paths:
            filtered_paths[split] = []
            continue

        valid_ids = set()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(is_ok_file, path) for path in paths]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Filtering {split}"
            ):
                path, is_ok, reason = future.result()
                stats[reason] += 1
                if is_ok:
                    splits = os.path.basename(path).split("_")
                    if len(splits) >= 2:
                        valid_ids.add(f"{splits[0]}_{splits[1]}")

        filtered_paths[split] = list(valid_ids)
        print(f"[{split}] Kept: {len(valid_ids)} / Total: {len(paths)}\n")

    if "validation" in filtered_paths:
        filtered_paths["val"] = filtered_paths.pop("validation")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(filtered_paths, f, indent=4)

    print("\n" + "=" * 40)
    print("🎯 Filtering Statistics Report")
    print("=" * 40)
    for reason, count in stats.most_common():
        marker = "✅" if reason == "passed" else "❌"
        print(f"{marker} {reason:<30} : {count}")
    print("=" * 40)
    print(f"🎉 Results saved to: {output_json}")


if __name__ == "__main__":
    paths_dict = load_split_paths(
        json_path="configs/abc1m_split.json",
        root_dir="/cache/yanko/dataset/abc_solids_conveted_bezier_dynamic_tol/",
    )
    # paths_dict = load_split_paths(
    #     json_path="brepgen_deepcad_data_split_6bit.json",
    #     root_dir="/cache/yanko/dataset/deepcad",
    # )
    for k, v in paths_dict.items():
        print(f"{k}: {len(v)} files")

    filter_dataset(
        dataset_paths=paths_dict,
        output_json="configs/filtered_brep_abc1m_paths.json",
        max_workers=100,
    )
