"""
Count how many rows of the ABC-1M `train` split survive filtering.

Pipeline (same as the training DataModule, minus everything after pre_filter):
    pushdown filter (num_faces_after_splitting range + scaled_unique)
      -> pre_filter (geometry / topology checks + easy-sample downsampling)

Only 5 tiny columns are read, so this is cheap despite the ~900GB dataset.

Usage:
    python analysis/stats_abc1m_training_rows.py                              # stream straight from HuggingFace
    python analysis/stats_abc1m_training_rows.py /path/to/abc-1m              # local dir containing a train/ folder
"""

import io
import sys

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
from tqdm import tqdm

# ---- filter parameters (defaults from ARDataModule / BaseDataModule) ----
MIN_FACE = 0
MAX_FACE = 100
MAX_EDGE = 1000
BIT = 10                       # data quantized to 10 bits over [-1, 1]
SCALED_UNIQUE = True
TOL = 1 / (2 ** (BIT - 1))     # tiny-face / tiny-edge tolerance
BATCH_ROWS = 4096

# columns needed for pushdown filter + pre_filter
COLUMNS = ["num_faces_after_splitting", "scaled_unique",
           "face_edge_incidence", "face_bbox_world", "edge_bbox_world"]


def deserialize_array(serialized: bytes) -> np.ndarray:
    memfile = io.BytesIO()
    memfile.write(serialized)
    memfile.seek(0)
    return np.load(memfile)


def open_train_dataset(data_root: str) -> ds.Dataset:
    """Open the train split, from a local dir or straight from HuggingFace."""
    if data_root:
        return ds.dataset(f"{data_root}/train", format="parquet")
    # Stream from HuggingFace hub (needs: pip install huggingface_hub)
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    paths = fs.glob("datasets/ADSKAILab/ABC-1M/data/train-*.parquet") or \
            fs.glob("datasets/ADSKAILab/ABC-1M/train/*.parquet")
    if not paths:
        raise FileNotFoundError("Could not locate train parquet files on the hub.")
    return ds.dataset(paths, format="parquet", filesystem=fs)


def deterministic_pre_filter(row) -> bool:
    """pre_filter without the random easy-sample downsampling. Returns keep?"""
    face_edge_adj = deserialize_array(row["face_edge_incidence"])

    if face_edge_adj.ndim != 2 or face_edge_adj.size == 0:                                  # [1] empty
        return False
    if np.any(np.all(np.logical_not(face_edge_adj), axis=1)):    # [3] face with no edge
        return False
    if np.any(np.sum(face_edge_adj.sum(0) != 2)):               # [4] non-manifold
        return False
    if face_edge_adj.shape[1] > MAX_EDGE:                        # [5] too many edges
        return False

    face_pos = deserialize_array(row["face_bbox_world"])         # [6] tiny face
    if np.any(np.all(np.abs(face_pos[:, 0:3] - face_pos[:, 3:6]) < TOL, axis=-1)):
        return False

    edge_pos = deserialize_array(row["edge_bbox_world"])         # [7] tiny edge
    if np.any(np.all(np.abs(edge_pos[:, 0:3] - edge_pos[:, 3:6]) < TOL, axis=-1)):
        return False

    return True


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else ""
    dataset = open_train_dataset(data_root)

    expr = ((pc.field("num_faces_after_splitting") >= MIN_FACE) &
            (pc.field("num_faces_after_splitting") <= MAX_FACE))
    if SCALED_UNIQUE:
        expr = expr & pc.field("scaled_unique")

    total = dataset.count_rows()
    after_pushdown = dataset.count_rows(filter=expr)
    print(f"total train rows            : {total:,}")
    print(f"after pushdown filter       : {after_pushdown:,} "
          f"(-{total - after_pushdown:,})")

    scanner = dataset.scanner(columns=COLUMNS, filter=expr, batch_size=BATCH_ROWS)

    passed_det = 0     # survives the deterministic geometry/topology checks
    easy = 0           # of those, samples with < 25 faces (downsampled at train time)
    hard = 0           # of those, samples with >= 25 faces (always kept)

    for batch in tqdm(scanner.to_batches(), desc="scanning", unit="batch"):
        cols = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
        for idx in range(batch.num_rows):
            row = {k: cols[k][idx].as_py() for k in cols}
            if not deterministic_pre_filter(row):
                continue
            passed_det += 1
            num_faces = deserialize_array(row["face_edge_incidence"]).shape[0]
            if num_faces < 25:
                easy += 1
            else:
                hard += 1

    # easy samples are kept with 10% probability (num_faces<25 & random<0.9 -> drop)
    expected_after_random = hard + 0.1 * easy

    print()
    print(f"after deterministic pre_filter : {passed_det:,} "
          f"(-{after_pushdown - passed_det:,})")
    print(f"  easy (<25 faces)             : {easy:,}  (kept ~10% at train time)")
    print(f"  hard (>=25 faces)            : {hard:,}  (always kept)")
    print(f"expected per-epoch training rows: {expected_after_random:,.0f}")
    print()
    print(f"filtered out by pre_filter (deterministic): "
          f"{after_pushdown - passed_det:,} / {after_pushdown:,} "
          f"({100 * (after_pushdown - passed_det) / max(after_pushdown, 1):.1f}%)")


if __name__ == "__main__":
    main()
