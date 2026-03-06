import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from json2files import load_dataset_fast

MAX_FACES = 100
MAX_EDGES_PER_FACE = 50
MAX_FACE_EDGE_TOTAL = 500


def check_file(path):
    try:
        with np.load(path) as data:
            # 1. Face Count Check
            fo = data["face_outer_offsets"]
            if len(fo) - 1 > MAX_FACES:
                return None
            # 2. Edge Count Check (Vectorized)
            # Outer edges count per face
            c_outer = fo[1:] - fo[:-1]

            # Inner edges count per face
            # Logic: face -> [loops] -> [edges]
            # inner_loop_offsets[ face_inner_offsets[i+1] ] - inner_loop_offsets[ face_inner_offsets[i] ]
            fi = data["face_inner_offsets"]
            lo = data["inner_loop_offsets"]
            c_inner = lo[fi[1:]] - lo[fi[:-1]]

            if np.any((c_outer + c_inner) > MAX_EDGES_PER_FACE):
                return None
            # 3. Total Face-Edge Count Check
            if np.sum(c_outer + c_inner) + len(fo) - 1 > MAX_FACE_EDGE_TOTAL:
                return None

            return path
    except Exception:
        return None


json_path = "deepcad.json"
dataset = load_dataset_fast(
    json_path, root_dir="/cache/yanko/dataset/abc_preprocessed/organized_by_face_count/"
)

all_paths = dataset["train"] + dataset["val"] + dataset["test"]
print(f"Filtering {len(all_paths)} files...")

with Pool(32) as p:
    # 结果为所有合法的绝对路径集合
    valid_paths = set(
        tqdm(p.imap_unordered(check_file, all_paths), total=len(all_paths))
    )

filtered_dataset = {
    k: [Path(x).stem[:8] for x in v if x in valid_paths] for k, v in dataset.items()
}

stats = {k: len(v) for k, v in filtered_dataset.items()}
print(f"Result: {stats}, Total Dropped: {len(all_paths) - sum(stats.values())}")

with open("deepcad_filtered_final.json", "w") as f:
    json.dump(filtered_dataset, f, indent=4)
