from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from json2files import load_dataset_fast
from src.data.brep_utils import arrays_to_sequence


def resolve_npz_path(file_item: str, dataset_json_path: Path) -> Path:
    path = Path(file_item)
    candidates = [path, Path.cwd() / path, dataset_json_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def count_single_npz(npz_file: str) -> tuple[int, int, int]:
    with np.load(npz_file) as npz_data:
        face_controls = npz_data["face_controls"]
        edge_controls = npz_data["edge_controls"]
        outer_edge_indices = npz_data["outer_edge_indices"]
        face_outer_offsets = npz_data["face_outer_offsets"]
        inner_edge_indices = npz_data["inner_edge_indices"]
        inner_loop_offsets = npz_data["inner_loop_offsets"]
        face_inner_offsets = npz_data["face_inner_offsets"]

        seq = arrays_to_sequence(
            face_controls,
            edge_controls,
            outer_edge_indices,
            face_outer_offsets,
            inner_edge_indices,
            inner_loop_offsets,
            face_inner_offsets,
            1024,
        )

    face_geom_count = len(seq.surface_geom_indexes)
    edge_geom_count = len(seq.curve_geom_indexes)
    text_count = len(seq.text_indexes)
    return face_geom_count, edge_geom_count, text_count


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description="多核统计 face/edge/text 占比")
    parser.add_argument(
        "--dataset-json",
        default="abc_filtered_final.json",
        help="数据集切分 JSON 路径",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="并行进程数，默认等于 CPU 核心数",
    )
    parser.add_argument(
        "--max-geom-entities",
        type=int,
        default=1024,
        help="传给 arrays_to_sequence 的 max_geom_entities",
    )
    args = parser.parse_args()

    dataset_json_path = Path(args.dataset_json)
    dataset = load_dataset_fast(
        str(dataset_json_path), root_dir="/cache/yanko/dataset/abc-splited-bezier-abc/"
    )
    files = dataset["train"] + dataset["val"] + dataset["test"]
    npz_files = [
        str(resolve_npz_path(file_item, dataset_json_path)) for file_item in files
    ]

    total_face = 0
    total_edge = 0
    total_text = 0
    success_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(count_single_npz, npz_file): npz_file
            for npz_file in npz_files
        }

        for future in as_completed(futures):
            npz_file = futures[future]
            try:
                face_geom_count, edge_geom_count, text_count = future.result()
            except Exception as exc:
                failed_count += 1
                print(f"[WARN] 处理失败: {npz_file} ({exc})")
                continue

            total_face += face_geom_count
            total_edge += edge_geom_count
            total_text += text_count
            success_count += 1

    total = total_face + total_edge + total_text
    face_ratio = safe_ratio(total_face, total)
    edge_ratio = safe_ratio(total_edge, total)
    text_ratio = safe_ratio(total_text, total)

    print("=== 统计结果 ===")
    print(f"成功文件数: {success_count}")
    print(f"失败文件数: {failed_count}")
    print(f"face 总数: {total_face}")
    print(f"edge 总数: {total_edge}")
    print(f"text 总数: {total_text}")
    print(f"face 占比: {face_ratio:.6f}")
    print(f"edge 占比: {edge_ratio:.6f}")
    print(f"text 占比: {text_ratio:.6f}")

    max_ratio = max(face_ratio, edge_ratio, text_ratio)
    if max_ratio > 0:
        face_copy = math.ceil(max_ratio / face_ratio) if face_ratio > 0 else 0
        edge_copy = math.ceil(max_ratio / edge_ratio) if edge_ratio > 0 else 0
        text_copy = math.ceil(max_ratio / text_ratio) if text_ratio > 0 else 0
        print("\n=== 建议复制倍率(以最大占比类为 1x 对齐) ===")
        print(f"face: {face_copy}x")
        print(f"edge: {edge_copy}x")
        print(f"text: {text_copy}x")


### 输出BBox Coordinate
# === 统计结果 ===
# face 总数: 5334899
# edge 总数: 14915758
# text 总数: 224685551
# face 占比: 0.021781
# edge 占比: 0.060897
# text 占比: 0.917323

# === 建议复制倍率(以最大占比类为 1x 对齐) ===
# face: 43x
# edge: 16x
# text: 1x

### 无BBox
# === 统计结果 ===
# 成功文件数: 225489
# 失败文件数: 0
# face 总数: 5334899
# edge 总数: 14915758
# text 总数: 103181609
# face 占比: 0.043221
# edge 占比: 0.120842
# text 占比: 0.835937

# === 建议复制倍率(以最大占比类为 1x 对齐) ===
# face: 20x
# edge: 7x
# text: 1x
if __name__ == "__main__":
    main()
