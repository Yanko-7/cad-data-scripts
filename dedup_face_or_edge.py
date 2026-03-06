import json
import pickle
import argparse
import hashlib
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--bit", type=int, default=6)
parser.add_argument("--edge", action="store_true")
args = parser.parse_args()

split_path = f"abc_deduplicated_all_{args.bit}bit.json"
base_root = Path("/cache/yanko/dataset/abc_preprocessed/organized_by_face_count/")
target_key = "edge_points" if args.edge else "face_points"
output_name = (
    f"abc_unique_{'edge' if args.edge else 'face'}_canonical_{args.bit}bit.pkl"
)


def get_canonical_hash(points):
    if points.shape[0] < 3:
        quantized = np.floor(points * (2**args.bit)).astype(np.int32)
        return hashlib.sha256(quantized.tobytes()).hexdigest()

    # 1. Translation
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # 2. Rotation (PCA)
    cov = np.dot(centered.T, centered)
    evals, evecs = np.linalg.eigh(cov)
    sort_idx = np.argsort(evals)[::-1]
    evecs = evecs[:, sort_idx]
    rotated = np.dot(centered, evecs)

    # 3. Scale (Normalize to unit cube)
    max_val = np.max(np.abs(rotated))
    if max_val > 1e-8:
        rotated /= max_val

    # 4. Quantization
    scale = 2**args.bit
    quantized_base = np.floor(rotated * scale).astype(np.int32)

    # 5. Flip Ambiguity Check
    min_h_val = None
    signs = np.array(
        [[x, y, z] for x in [1, -1] for y in [1, -1] for z in [1, -1]], dtype=np.int32
    )

    for sign in signs:
        q_flip = quantized_base * sign

        # Sort for permutation invariance
        q_contig = np.ascontiguousarray(q_flip)
        dtype_view = np.dtype((np.void, q_contig.dtype.itemsize * q_contig.shape[1]))
        sort_order = np.argsort(q_contig.view(dtype_view).ravel())
        q_sorted = q_flip[sort_order]

        curr_h = hashlib.sha256(q_sorted.tobytes()).hexdigest()
        if min_h_val is None or curr_h < min_h_val:
            min_h_val = curr_h

    return min_h_val


def compute_hashes(path):
    try:
        full_path = base_root / path
        if not full_path.exists():
            return None

        with np.load(full_path, mmap_mode="r") as f:
            if target_key not in f:
                return None

            data = f[target_key]
            if data.shape[0] == 0:
                return None

            data_reshaped = data.reshape(data.shape[0], -1, 3)
            # Read into memory for processing
            batch_points = np.array(data_reshaped)

            local_hashes = [get_canonical_hash(p) for p in batch_points]
            return (path, local_hashes)
    except:
        return None


def fetch_unique_data(task):
    path, indices = task
    try:
        full_path = base_root / path
        with np.load(full_path, mmap_mode="r") as f:
            return np.array(f[target_key][indices])
    except:
        return []


if __name__ == "__main__":
    with open(split_path, "r") as f:
        dataset = json.load(f)

    final_data = {}

    with Pool() as pool:
        for split in ["train", "val", "test"]:
            file_list = dataset[split]
            print(f"Processing {split}...")

            seen_hashes = set()
            fetch_tasks = []

            iterator = pool.imap_unordered(compute_hashes, file_list, chunksize=5)
            for result in tqdm(iterator, total=len(file_list)):
                if result is None:
                    continue

                path, hashes = result
                unique_indices = []
                for idx, h_val in enumerate(hashes):
                    if h_val not in seen_hashes:
                        seen_hashes.add(h_val)
                        unique_indices.append(idx)

                if unique_indices:
                    fetch_tasks.append((path, unique_indices))

            print(f"Unique canonical shapes: {len(seen_hashes)}")

            split_items = []
            data_iterator = pool.imap_unordered(
                fetch_unique_data, fetch_tasks, chunksize=10
            )
            for data_batch in tqdm(data_iterator, total=len(fetch_tasks)):
                if len(data_batch) > 0:
                    split_items.extend(data_batch)

            final_data[split] = split_items

    with open(output_name, "wb") as f:
        pickle.dump(final_data, f)
    print(f"Saved to {output_name}")
