import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

MAX_FACE = 50
PER_FACE_EDGE_LIMIT = 30
BBOX_THRES = 0.05 / 3  # 替换原有的 threshold_value / scaled_value 比例
TOTOAL_EDGE_LIMIT = 300


def load_split_paths(
    json_path: str, root_dir: str, ext: str = ".npz"
) -> Dict[str, List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_map = defaultdict(list)
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(ext):
                prefix = f[: -len(ext)].split("_")[0]
                file_map[prefix].append(os.path.join(r, f))

    return {
        split: [path for prefix in prefixes for path in file_map.get(prefix, [])]
        for split, prefixes in data.items()
    }


def normalize_points_with_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 1. 检测是否包含 null (NaN) 或无穷大 (Inf) 的值
    if not np.isfinite(points).all():
        raise ValueError("Input points contain null (NaN) or infinite values.")

    xyz, w = points[..., :3], points[..., 3:4]

    # 防止 w=-1 导致分母为 0 (除以 0 会产生新的 NaN/Inf)
    # 也可以在上面 isfinite 之后加个安全断言或者加一个小 epsilon
    w_norm = (w - 1) / (w + 1 + 1e-8)

    pts_flat = xyz.reshape(-1, 3)
    vmin, vmax = pts_flat.min(axis=0), pts_flat.max(axis=0)
    span = (vmax - vmin).max()

    if span < 0.004:
        raise ValueError(f"Degenerate geometry with zero span ({span})")

    center = (vmin + vmax) / 2.0
    norm_xyz = (xyz - center) * (2.0 / span)

    return np.concatenate([norm_xyz, w_norm], axis=-1), np.concatenate([vmin, vmax])


def check_topology_and_get_adj(
    outer_edges: np.ndarray,
    face_outer_offsets: np.ndarray,
    inner_edges: np.ndarray,
    inner_loop_offsets: np.ndarray,
    face_inner_offsets: np.ndarray,
) -> Tuple[bool, List[List[int]]]:
    num_faces = len(face_outer_offsets) - 1
    if not (0 < num_faces <= MAX_FACE):
        return False, []

    num_edges = (
        int(max(np.max(outer_edges, initial=-1), np.max(inner_edges, initial=-1))) + 1
    )
    edge_to_faces = [[] for _ in range(num_edges)]
    face_edges_adj = []

    for f_id in range(num_faces):
        e_ids = list(
            outer_edges[face_outer_offsets[f_id] : face_outer_offsets[f_id + 1]]
        )
        for l_idx in range(face_inner_offsets[f_id], face_inner_offsets[f_id + 1]):
            e_ids.extend(
                inner_edges[inner_loop_offsets[l_idx] : inner_loop_offsets[l_idx + 1]]
            )

        if len(e_ids) > PER_FACE_EDGE_LIMIT:
            return False, []

        face_edges_adj.append(e_ids)
        for e_id in set(e_ids):
            edge_to_faces[e_id].append(f_id)

    if len(edge_to_faces) > TOTOAL_EDGE_LIMIT or not all(
        len(faces) in (0, 2) for faces in edge_to_faces
    ):
        return False, []

    return True, face_edges_adj


def has_duplicate_bboxes(bboxes: np.ndarray, threshold: float = BBOX_THRES) -> bool:
    """使用 NumPy 广播机制快速计算两两包围盒差异，判断是否存在重叠/过近"""
    if len(bboxes) < 2:
        return False
    diffs = np.max(np.abs(bboxes[:, None, :] - bboxes[None, :, :]), axis=-1)
    np.fill_diagonal(diffs, np.inf)
    return bool((diffs < threshold).any())


def is_ok_file(file_path: str) -> Tuple[str, bool]:
    try:
        with np.load(file_path) as data:
            if len(data["face_outer_offsets"]) - 1 != len(data["face_controls"]):
                return file_path, False

            is_valid_topo, face_edges_adj = check_topology_and_get_adj(
                data["outer_edge_indices"],
                data["face_outer_offsets"],
                data["inner_edge_indices"],
                data["inner_loop_offsets"],
                data["face_inner_offsets"],
            )
            if not is_valid_topo:
                return file_path, False

            face_bboxes = np.array(
                [normalize_points_with_bbox(fc)[1] for fc in data["face_controls"]]
            )
            if has_duplicate_bboxes(face_bboxes):
                return file_path, False

            edge_bboxes = np.array(
                [normalize_points_with_bbox(ec)[1] for ec in data["edge_controls"]]
            )

            for e_ids in face_edges_adj:
                if not e_ids:
                    return file_path, False
                if has_duplicate_bboxes(edge_bboxes[e_ids]):
                    return file_path, False

            return file_path, True
    except Exception:
        return file_path, False


def filter_dataset_multiprocessing(
    dataset_paths: Dict[str, List[str]],
    output_json: str = "filtered_dataset_paths.json",
    max_workers: int = None,
):
    filtered_paths = {}

    for split, paths in dataset_paths.items():
        if not paths:
            filtered_paths[split] = []
            continue

        valid_short_ids = set()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(is_ok_file, path) for path in paths]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Filtering {split}"
            ):
                path, is_ok = future.result()
                if is_ok:
                    valid_short_ids.add(os.path.basename(path).split("_")[0])

        filtered_paths[split] = list(valid_short_ids)
        print(
            f"[{split}] Done! Kept: {len(valid_short_ids)} IDs / Total files: {len(paths)}\n"
        )

    if "validation" in filtered_paths:
        filtered_paths["val"] = filtered_paths.pop("validation")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(filtered_paths, f, indent=4)
    print(f"🎉 Results saved to: {output_json}")


if __name__ == "__main__":
    dataset_paths = load_split_paths(
        json_path="brep_abc_data_split_6bit.json",
        root_dir="/cache/yanko/sciprts/brepgen-abc",
        ext=".npz",
    )

    filter_dataset_multiprocessing(
        dataset_paths,
        output_json="filtered_brep_abc_data_split_6bit_paths.json",
        max_workers=100,
    )
