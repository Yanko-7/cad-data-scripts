import json
import argparse
import numpy as np
import hashlib
import os
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument("--bit", type=int, default=6)
parser.add_argument("--workers", type=int, default=os.cpu_count() - 2)
parser.add_argument("--split", type=str, default="../dataset_split.json")
parser.add_argument("--out", type=str, default="abc_dedup_no_pca.json")
parser.add_argument(
    "--root",
    type=str,
    default="/cache/yanko/dataset/abc_preprocessed/organized_by_face_count/",
)
args = parser.parse_args()

# 预计算 8 种翻转组合 (保留这个是为了解决坐标系定义差异, 如左手/右手系)
SIGNS = np.array(list(product([1, -1], repeat=3)), dtype=np.float32)


def real2bit(data, n_bits=8, min_range=-1, max_range=1):
    range_quantize = 2**n_bits - 1
    data_quantize = (data - min_range) * range_quantize / (max_range - min_range)
    data_quantize = np.clip(data_quantize, a_min=0, a_max=range_quantize)
    return data_quantize.astype(np.int32)


def compute_hash(path):
    try:
        full_path = Path(path)
        if not full_path.exists():
            return path, None
        # ID Generation
        stem = full_path.stem
        parts = stem.split("_")
        simple_id = f"{parts[0]}_{parts[-2]}_{parts[-1]}" if len(parts) >= 3 else stem
        with np.load(full_path, mmap_mode="r") as d:
            if "face_points" not in d:
                return path, None
            pts = d["face_points"].reshape(-1, 3).astype(np.float32)

        if pts.shape[0] < 3:
            return path, None

        # 1. Translation (Centering) - KEEP
        pts -= np.mean(pts, axis=0)

        # 2. Rotation (PCA) - REMOVED!
        # (已删除 PCA 代码，保持原始旋转姿态)

        # 3. Scale (Normalization) - KEEP
        max_val = np.max(np.abs(pts))
        if max_val > 1e-8:
            pts /= max_val

        # 4. Hash (Flip + Quantize)
        min_h = None
        dummy = np.zeros((1, 3), dtype=np.int32)
        v_dtype = np.dtype((np.void, dummy.dtype.itemsize * 3))

        for sign in SIGNS:
            q = real2bit(pts * sign, n_bits=args.bit, min_range=-1, max_range=1)
            # 全局排序消除点序影响
            q_contig = np.ascontiguousarray(q)
            q_sorted = q[np.argsort(q_contig.view(v_dtype).ravel())]

            curr_h = hashlib.sha256(q_sorted.tobytes()).hexdigest()
            if min_h is None or curr_h < min_h:
                min_h = curr_h

        return path, (simple_id, min_h)
    except:
        return path, None


from json2files import load_dataset_fast

if __name__ == "__main__":
    dataset = load_dataset_fast(
        args.split,
        root_dir="/cache/yanko/dataset/abc_preprocessed/organized_by_face_count/",
    )
    all_files = dataset["train"] + dataset["val"] + dataset["test"]

    print(f"Processing {len(all_files)} files (No PCA)...")
    with Pool(args.workers) as pool:
        results = dict(tqdm(pool.imap(compute_hash, all_files), total=len(all_files)))

    seen, new_data = set(), {k: [] for k in ["train", "val", "test"]}
    stats = {k: 0 for k in new_data}

    for split in ["train", "val", "test"]:
        for path in dataset[split]:
            res = results.get(path)
            if res and res[1] and (res[1] not in seen):
                seen.add(res[1])
                new_data[split].append(res[0])
            else:
                stats[split] += 1

    print(f"Removed: {stats}")
    print(f"Final: { {k: len(v) for k, v in new_data.items()} }")

    with open(args.out, "w") as f:
        json.dump(new_data, f, indent=4)
