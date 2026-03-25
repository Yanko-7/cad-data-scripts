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
from utils import (
    extract_bicubic_features_dir,
    get_fast_stats,
    is_watertight,
    load_and_filter_step,
    preprocess_shape,
)

MAX_FACES = 300
MAX_EDGES = 2000
PERFACE_EDGE = 40


def worker_task(file_path_str, output_dir_str):
    gc.disable()
    file_path = Path(file_path_str)
    base_name = file_path.stem
    shard_dir = Path(output_dir_str) / base_name[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)

    # 最佳实践：使用 glob 匹配特征名，保留 Early Exit 性能优势，避免加载已处理的模型
    if next(shard_dir.glob(f"{base_name}_f*_e*.npz"), None):
        return "SKIPPED", None

    try:
        shape = load_and_filter_step(str(file_path))
        # shape = read_step_file(str(file_path))
        # shape = split_to_bicubic(shape)
        shape = preprocess_shape(shape)
        if not is_watertight(shape):
            return "SKIPPED", "Not watertight"

        data = extract_bicubic_features_dir(shape)

        f_count, e_count, face_edge_counts = get_fast_stats(shape)
        data["f_count"] = np.array([f_count], dtype=np.int32)
        data["e_count"] = np.array([e_count], dtype=np.int32)
        # 动态生成包含面和边数量的最终文件名
        final_filename = f"{base_name.split('_')[0]}_f{f_count}_e{e_count}.npz"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count() - 4)
    parser.add_argument("-t", "--timeout", type=int, default=120)
    args = parser.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=out_dir / "failures.log",
        level=logging.ERROR,
        format="%(asctime)s|%(message)s",
    )

    print("[INFO] Scanning and pre-filtering files...")
    with open("ABC-dataset/abc_data_split_6bit.json", "r") as f:
        split = json.load(f)
        dataset = set()
        for key, value in split.items():
            dataset.update(value)
        print(f"Total unique models in split: {len(dataset)}")

    SEARCH_ROOT = "/cache/yanko/dataset/abc/"
    search_path = Path(SEARCH_ROOT)
    files = []
    for file in search_path.rglob("*.step"):
        name = file.stem
        if name.split("_")[0] in dataset:
            files.append(file)

    total = len(files)
    batch_size = args.workers * 50000
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
                    f_name = futures[future].name
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
