import json
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

split_path = "../dataset_split.json"
base_root = Path("/cache/yanko/dataset/abc_preprocessed/organized_by_face_count/")


def get_nbytes(path):
    with np.load(base_root / path, mmap_mode="r") as data:
        return data["face_points"].nbytes


if __name__ == "__main__":
    with open(split_path, "r") as f:
        parsed = json.load(f)
        data = parsed["train"] + parsed["test"] + parsed["val"]

    file_paths = [p for p in data if (base_root / p).exists()]

    with Pool() as pool:
        sizes = list(tqdm(pool.imap(get_nbytes, file_paths), total=len(file_paths)))

    sizes = np.array(sizes) / 1024 / 1024  # Convert to MB

    print(f"Average: {np.mean(sizes):.4f} MB")
    print(f"Median:  {np.median(sizes):.4f} MB")
    print(f"Max:     {np.max(sizes):.4f} MB")
    print(f"Min:     {np.min(sizes):.4f} MB")
