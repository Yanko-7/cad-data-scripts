import hashlib
import json
import os
import random
import shutil
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm
ROOT = Path("/cache/yanko/dataset/abc-splited-bezier-all/")
ORG_DIR = ROOT / "organized"
BITS = 8
WORKERS = max(1, os.cpu_count() - 2)
SIGNS = np.array(list(product([1, -1], repeat=3)), dtype=np.float32)
SPLIT_RATIOS = (0.9, 0.05, 0.05)
MAX_FACE=100

def process_file(path):
    if path.name.endswith(".tmp.npz"):
        return None

    try:
        with np.load(path) as d:
            faces = d["face_controls"]
            edges = d["edge_controls"]
            n_faces, n_edges = len(faces), len(edges)

        if n_faces > MAX_FACE:
            return None

        ctrls = faces.reshape(-1, 16, 4)
        xyz = ctrls[..., :3].reshape(-1, 3).astype(np.float32)
        xyz -= xyz.mean(axis=0)

        evals, evecs = np.linalg.eigh(np.cov(xyz, rowvar=False))
        xyz = xyz @ evecs[:, np.argsort(evals)[::-1]]
        xyz /= np.abs(xyz).max() + 1e-8

        w = ctrls[..., 3].astype(np.float32)
        w /= w.mean(axis=1, keepdims=True) + 1e-8
        w = w.reshape(-1, 1) / (w.max() + 1e-8)

        max_val = (1 << BITS) - 1
        hashes = []
        for sign in SIGNS:
            q_xyz = (xyz * sign + 1) * (max_val / 2)
            q = np.clip(
                np.concatenate([q_xyz, w * max_val], axis=1), 0, max_val
            ).astype(np.int32)
            hashes.append(hashlib.sha256(q[np.lexsort(q.T)].tobytes()).hexdigest())

        parts = path.stem.split("_")
        f_id = f"{parts[0]}_{parts[1]}"
        new_name = f"{f_id}_{n_faces}_{n_edges}.npz"

        return f_id, path, min(hashes), n_faces, new_name

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def organize_file(args):
    _, path, n_faces, new_name = args
    dest = ORG_DIR / f"cat_{n_faces // 20}" / new_name
    dest.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy2(path, dest)


def run_pipeline():
    files = list(ROOT.rglob("*.npz"))
    print(f"Found {len(files)} files. Using {WORKERS} cores.")

    seen, unique_map = set(), {}

    with Pool(WORKERS) as p:
        results = tqdm(
            p.imap(process_file, files, chunksize=100), total=len(files), desc="Hashing"
        )
        for res in results:
            if res and res[2] not in seen:
                f_id, path, f_hash, n_faces, new_name = res
                seen.add(f_hash)
                unique_map[f_id] = (path, n_faces, new_name)

    print(f"Organizing {len(unique_map)} unique files...")
    org_tasks = [(k, *v) for k, v in unique_map.items()]
    with Pool(WORKERS) as p:
        list(
            tqdm(
                p.imap(organize_file, org_tasks, chunksize=50),
                total=len(org_tasks),
                desc="Moving",
            )
        )

    ids = list(unique_map.keys())
    random.shuffle(ids)

    n = len(ids)
    t_idx = int(n * SPLIT_RATIOS[0])
    v_idx = t_idx + int(n * SPLIT_RATIOS[1])

    with open("dataset_split.json", "w") as f:
        json.dump(
            {"train": ids[:t_idx], "val": ids[t_idx:v_idx], "test": ids[v_idx:]},
            f,
            indent=4,
        )

    print("Done.")


if __name__ == "__main__":
    run_pipeline()
