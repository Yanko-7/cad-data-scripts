import json
import os
import gc
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import TimeoutError, as_completed
from pebble import ProcessPool, ProcessExpired
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepTools import breptools
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.TopoDS import TopoDS_Shape
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    extract_bicubic_features_dir,
    get_fast_stats,
    preprocess_shape,
)

MAX_FACES = 100
MAX_EDGES = 1000
# PERFACE_EDGE = 40


def worker_task(file_path_str, output_dir_str):
    gc.disable()
    file_path = Path(file_path_str)
    base_name = file_path.stem
    shard_dir = Path(output_dir_str) / base_name[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)

    if next(shard_dir.glob(f"{base_name}_f*_e*.npz"), None):
        return "SKIPPED", None

    try:
        shape = TopoDS_Shape()
        breptools.Read(shape, str(file_path), BRep_Builder())
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        shape = fixer.Shape()
        shape = preprocess_shape(shape)

        data = extract_bicubic_features_dir(shape)

        f_count, e_count, _ = get_fast_stats(shape)
        data["f_count"] = np.array([f_count], dtype=np.int32)
        data["e_count"] = np.array([e_count], dtype=np.int32)
        # 动态生成包含面和边数量的最终文件名
        final_filename = f"{base_name.split('_')[0]}_{base_name.split('_')[1]}_f{f_count}_e{e_count}.npz"
        final_path = shard_dir / final_filename
        temp_path = shard_dir / f"{base_name}.tmp.npz"

        np.savez_compressed(temp_path, **data)
        temp_path.replace(final_path)

        return "SUCCESS", None
    except Exception as e:
        return "ERROR", str(e)


def filter_and_dedup(file_paths):
    filtered = {}
    for f in file_paths:
        parts = f.stem.split("_")
        if len(parts) != 4:
            continue
        try:
            model_id, faces, edges = parts[0], int(parts[2]), int(parts[3])
            if faces <= MAX_FACES and edges <= MAX_EDGES:
                key = (model_id, faces, edges)
                if key not in filtered:
                    filtered[key] = f
        except ValueError:
            continue
    return list(filtered.values())


def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos : pos + size]


def load_split_paths(json_path: str, root_dir: str, ext: str = ".npz") -> dict[str, list[str]]:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count() - 4)
    parser.add_argument("-t", "--timeout", type=int, default=600)
    args = parser.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=out_dir / "failures.log",
        level=logging.ERROR,
        format="%(asctime)s|%(message)s",
    )
    # paths_dict = load_split_paths(
    #     json_path="brepgen_deepcad_data_split_6bit.json",
    #     root_dir="/cache/yanko/dataset/abc",
    #     ext=".step"
    # )
    # files = paths_dict["train"] + paths_dict["val"] + paths_dict["test"]

    # # print("[INFO] Scanning and pre-filtering files...")
    # # with open("ABC-dataset/abc_data_split_6bit.json", "r") as f:
    # #     split = json.load(f)
    # #     dataset = set()
    # #     for key, value in split.items():
    # #         dataset.update(value)
    # #     print(f"Total unique models in split: {len(dataset)}")

    search_path = Path(in_dir)
    files = list(search_path.rglob("*.brep"))

    total = len(files)
    batch_size = args.workers * 500000
    stats = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0, "CRASH": 0, "TIMEOUT": 0}

    print(f"Target Files: {total} | Workers: {args.workers} | Batch: {batch_size}")

    with tqdm(total=total, smoothing=0.1) as pbar:
        for chunk in chunker(files, batch_size):
            with ProcessPool(max_workers=args.workers, max_tasks=5000) as pool:
                futures = {
                    pool.schedule(
                        worker_task, args=(str(f), str(out_dir)), timeout=args.timeout
                    ): f
                    for f in chunk
                }

                for future in as_completed(futures):
                    f_name = futures[future]
                    try:
                        status, msg = future.result()
                        stats[status] += 1
                        if status == "ERROR":
                            logging.error(f"ERROR|{f_name}|{msg}")
                    except TimeoutError:
                        stats["TIMEOUT"] += 1
                        logging.error(f"TIMEOUT|{f_name}")
                    except ProcessExpired:
                        stats["CRASH"] += 1
                        logging.error(f"CRASH|{f_name}")
                    except Exception as e:
                        stats["ERROR"] += 1
                        logging.error(f"UNKNOWN|{f_name}|{e}")

                    pbar.set_postfix(
                        ok=stats["SUCCESS"],
                        skip=stats["SKIPPED"],
                        fail=stats["ERROR"] + stats["CRASH"] + stats["TIMEOUT"],
                    )
                    pbar.update(1)

    print("\n--- Done ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
