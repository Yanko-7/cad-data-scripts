import json
import hashlib
import shutil
import random
import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from itertools import product
from tqdm import tqdm

# --- Configuration ---
ROOT = "/cache/yanko/dataset/abc-reorder-p32/"
ORG_DIR = f"{ROOT}/organized"
BITS = 6
# WORKERS = 1
WORKERS = max(1, os.cpu_count() - 2)
SIGNS = np.array(list(product([1, -1], repeat=3)), dtype=np.float32)


def compute_hash(path):
    try:
        with np.load(path, mmap_mode="r") as d:
            flat_pts = d["face_points"].reshape(-1, 3)
            pts = (flat_pts - np.mean(flat_pts, axis=0)).astype(np.float32)
            pts /= np.max(np.abs(pts)) + 1e-8
            max_val = (2**BITS) - 1
            min_h = None
            for sign in SIGNS:
                q = np.clip((pts * sign + 1) * (max_val / 2), 0, max_val).astype(
                    np.int32
                )
                h = hashlib.sha256(q[np.lexsort(q.T)].tobytes()).hexdigest()
                if min_h is None or h < min_h:
                    min_h = h
            return path.stem[:8], str(path), min_h
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def organize_file(item):
    """物理移动文件的 Worker"""
    f_id, p_str = item
    p = Path(p_str)
    cat = f"cat_{min(int(p.stem.split('_')[-2]) // 20, 5)}"
    dest = Path(ORG_DIR) / cat / p.name
    dest.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy2(p, dest)
    return True


def run_pipeline():
    all_files = list(Path(ROOT).rglob("*.npz"))
    print(f"Found {len(all_files)} files. Using {WORKERS} cores.")

    # 1. Multi-core Deduplication
    with Pool(WORKERS) as p:
        results = [
            r
            for r in tqdm(
                p.imap(compute_hash, all_files, chunksize=100),
                total=len(all_files),
                desc="Hashing",
            )
        ]

    seen, unique_map = set(), {}
    for f_id, f_path, f_hash in results:
        if f_hash and f_hash not in seen:
            seen.add(f_hash)
            unique_map[f_id] = f_path

    # 2. Multi-core Organization
    print(f"Organizing {len(unique_map)} unique files...")
    with Pool(WORKERS) as p:
        list(
            tqdm(
                p.imap(organize_file, unique_map.items(), chunksize=50),
                total=len(unique_map),
                desc="Moving",
            )
        )

    # 3. Split
    ids = list(unique_map.keys())
    random.shuffle(ids)
    split = {
        "train": ids[: int(len(ids) * 0.9)],
        "val": ids[int(len(ids) * 0.9) : int(len(ids) * 0.95)],
        "test": ids[int(len(ids) * 0.95) :],
    }
    with open("dataset_split.json", "w") as f:
        json.dump(split, f, indent=4)
    print("Done.")


if __name__ == "__main__":
    run_pipeline()
