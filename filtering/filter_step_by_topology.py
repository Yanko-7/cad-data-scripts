import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_fast_stats, is_watertight, load_and_filter_step, split_all_closed_edges, split_all_closed_faces

MAX_FACE = 50
PER_FACE_EDGE_LIMIT = 30
TOTAL_EDGE_LIMIT = 1000

def load_split_paths(
    json_path: str, root_dir: str, ext: str = ".step"
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


def is_ok_file(file_path: str) -> Tuple[str, bool, str]:
    try:
        shape = load_and_filter_step(file_path)

        if not is_watertight(shape):
            return file_path, False, "non_manifold_edges"

        shape = split_all_closed_faces(shape)
        shape = split_all_closed_edges(shape)

        total_faces, total_edges, face_edge_counts = get_fast_stats(shape)

        if not (0 < total_faces <= MAX_FACE):
            return file_path, False, "invalid_face_count"
        if total_edges > TOTAL_EDGE_LIMIT:
            return file_path, False, "exceeds_total_edge_limit"
        if face_edge_counts and max(face_edge_counts) > PER_FACE_EDGE_LIMIT:
            return file_path, False, "exceeds_per_face_edge_limit"

        return file_path, True, "passed"
    except Exception as e:
        return file_path, False, f"load_or_process_error: {type(e).__name__}"


def filter_dataset(
    dataset_paths: Dict[str, List[str]], output_json: str, max_workers: int = None
):
    filtered_paths = {}
    stats = Counter()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for split, paths in dataset_paths.items():
            if not paths:
                filtered_paths[split] = []
                continue

            valid_ids = set()
            futures = [executor.submit(is_ok_file, path) for path in paths]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=f"Filtering {split}"
            ):
                path, is_ok, reason = future.result()
                stats[reason] += 1
                if is_ok:
                    parts = os.path.basename(path).split("_")
                    if len(parts) >= 2:
                        valid_ids.add(f"{parts[0]}_{parts[1]}")

            filtered_paths[split] = sorted(valid_ids)
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
    print("Scanning dataset directory...")
    paths_dict = load_split_paths(
        json_path="ABC-dataset/abc_data_split_6bit.json",
        root_dir="/cache/yanko/dataset/abc-origin/",
    )
    for k, v in paths_dict.items():
        print(f"{k}: {len(v)} files")

    filter_dataset(
        dataset_paths={k: v for k, v in paths_dict.items() if k in ("train", "test")},
        output_json="configs/filtered_abc_step_topology.json",
        max_workers=64,
    )
