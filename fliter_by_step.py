import json
from json2files import load_dataset_fast
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from pathlib import Path
from utils import (
    load_and_filter_step,
    check_euler_poincare,
    check_validity,
    get_fast_stats,
)

MAX_FACES = 90
MAX_EDGES_PER_FACE = 50
MAX_FACE_EDGE_TOTAL = 1000


def check_file(path):
    try:
        shape = load_and_filter_step(path)
        check_validity(shape)
        check_euler_poincare(shape)
        total_faces, total_edges, face_edge_counts = get_fast_stats(shape)

        if (
            total_faces > MAX_FACES
            or max(face_edge_counts) > MAX_EDGES_PER_FACE
            or (total_faces + total_edges > MAX_FACE_EDGE_TOTAL)
        ):
            return None
        return path
    except Exception as e:
        # print(f"Invalid file: {path}, due to {e}")
        return None


json_path = "abc_dedup_no_pca.json"
dataset = load_dataset_fast(
    json_path, root_dir="/cache/yanko/dataset/abc/", ext=".step"
)

all_paths = dataset["train"] + dataset["val"] + dataset["test"]
print(f"Filtering {len(all_paths)} files...")

with Pool(125) as p:
    # 结果为所有合法的绝对路径集合
    valid_paths = set(
        tqdm(p.imap_unordered(check_file, all_paths), total=len(all_paths))
    )

filtered_dataset = {
    k: [Path(x).stem[:8] for x in v if x in valid_paths] for k, v in dataset.items()
}

stats = {k: len(v) for k, v in filtered_dataset.items()}
print(f"Result: {stats}, Total Dropped: {len(all_paths) - sum(stats.values())}")

with open("abc_filtered_final.json", "w") as f:
    json.dump(filtered_dataset, f, indent=4)
