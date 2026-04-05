import os
import gc
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import TimeoutError, as_completed
from pebble import ProcessPool, ProcessExpired

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    check_validity,
    extract_bicubic_features,
    get_fast_stats,
    is_watertight,
    load_and_filter_step,
    preprocess_shape,
    split_to_bicubic,
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

    final_path = shard_dir / f"{base_name}.npz"
    if final_path.exists():
        return "SKIPPED", None

    try:
        shape = load_and_filter_step(str(file_path))
        shape = preprocess_shape(split_to_bicubic(shape))

        if not is_watertight(shape):
            return "SKIPPED", "Not watertight"

        check_validity(shape)
        f_count, e_count, counts = get_fast_stats(shape)

        # 二次校验：确保处理后拓扑依然满足限制
        if (
            f_count > MAX_FACES
            or e_count > MAX_EDGES
            or any(c > PERFACE_EDGE for c in counts)
        ):
            return "SKIPPED", "Too complex post-process"

        data = extract_bicubic_features(shape)
        data["f_count"] = np.array([f_count], dtype=np.int32)
        data["e_count"] = np.array([e_count], dtype=np.int32)

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
    raw_files = in_dir.rglob("*.step")
    files = filter_and_dedup(raw_files)

    total = len(files)
    batch_size = args.workers * 500
    stats = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0, "CRASH": 0, "TIMEOUT": 0}

    print(f"Target Files: {total} | Workers: {args.workers} | Batch: {batch_size}")

    with tqdm(total=total, smoothing=0.1) as pbar:
        for chunk in chunker(files, batch_size):
            with ProcessPool(max_workers=args.workers, max_tasks=500) as pool:
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
