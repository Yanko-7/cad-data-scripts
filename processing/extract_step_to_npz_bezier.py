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
from OCC.Core.ShapeFix import ShapeFix_Shape
import sys
from OCC.Extend.DataExchange import read_step_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    extract_bicubic_features_dir,
    get_fast_stats,
    preprocess_shape,
)

def worker_task(file_path_str, output_dir_str, max_faces, max_edges, perface_edge):
    gc.disable()
    file_path = Path(file_path_str)
    base_name = file_path.stem
    shard_dir = Path(output_dir_str) / base_name[:2]
    shard_dir.mkdir(parents=True, exist_ok=True)

    # 最佳实践：使用 glob 匹配特征名，保留 Early Exit 性能优势，避免加载已处理的模型
    if next(shard_dir.glob(f"{base_name}_f*_e*.npz"), None):
        return "SKIPPED", None

    try:
        shape = read_step_file(str(file_path), True)
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        shape = fixer.Shape()
        shape = preprocess_shape(shape)

        f_count, e_count, face_edge_counts = get_fast_stats(shape)
        if f_count > max_faces or e_count > max_edges:
            return "SKIPPED", None
        if perface_edge and any(c > perface_edge for c in face_edge_counts):
            return "SKIPPED", None

        data = extract_bicubic_features_dir(shape)
        actual_e_count = len(data["edge_controls"])
        data["f_count"] = np.array([f_count], dtype=np.int32)
        data["e_count"] = np.array([actual_e_count], dtype=np.int32)

        final_filename = f"{base_name.split('_')[0]}_{base_name.split('_')[1]}_f{f_count}_e{actual_e_count}.npz"
        final_path = shard_dir / final_filename
        temp_path = shard_dir / f"{base_name}.tmp.npz"

        np.savez_compressed(temp_path, **data)
        temp_path.replace(final_path)

        return "SUCCESS", None
    except Exception as e:
        return "ERROR", str(e)


def filter_and_dedup(file_paths, max_faces, max_edges):
    filtered = {}
    for f in file_paths:
        parts = f.stem.split("_")
        if len(parts) != 4:
            continue
        try:
            model_id, faces, edges = parts[0], int(parts[2]), int(parts[3])
            if faces <= max_faces and edges <= max_edges:
                key = (model_id, faces, edges)
                if key not in filtered:
                    filtered[key] = f
        except ValueError:
            continue
    return list(filtered.values())


def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos : pos + size]


def load_split_paths(json_path: str, root_dir: str, splits: list[str] | None = None, ext: str = ".step") -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target = set(splits) if splits else set(data.keys())
    p2s = {p: split for split, prefixes in data.items() if split in target for p in prefixes}
    lengths = sorted({len(p) for p in p2s}, reverse=True)

    result = []
    for r, _, files in os.walk(root_dir):
        for f in filter(lambda x: x.endswith(ext), files):
            for length in lengths:
                if len(f) >= length and f[:length] in p2s:
                    result.append(os.path.join(r, f))
                    break

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count() - 4)
    parser.add_argument("-t", "--timeout", type=int, default=600)
    parser.add_argument("--max-faces", type=int, default=100)
    parser.add_argument("--max-edges", type=int, default=1000)
    parser.add_argument("--perface-edge", type=int, default=None)
    parser.add_argument("--split", type=str, default=None, help="Path to split JSON file")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to use, e.g. train val test")
    args = parser.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=out_dir / "failures.log",
        level=logging.ERROR,
        format="%(asctime)s|%(message)s",
    )
    if args.split:
        files = [Path(f) for f in load_split_paths(args.split, str(in_dir), splits=args.splits)]
    else:
        files = list(Path(in_dir).rglob("*.step"))

    total = len(files)
    batch_size = args.workers * 500000
    stats = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0, "CRASH": 0, "TIMEOUT": 0}

    print(f"Target Files: {total} | Workers: {args.workers} | Batch: {batch_size}")

    with tqdm(total=total, smoothing=0.1) as pbar:
        for chunk in chunker(files, batch_size):
            with ProcessPool(max_workers=args.workers, max_tasks=5000) as pool:
                futures = {
                    pool.schedule(
                        worker_task,
                        args=(str(f), str(out_dir), args.max_faces, args.max_edges, args.perface_edge),
                        timeout=args.timeout,
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

# python extract_step_to_npz_bezier.py -i /data/step -o /data/out \
#     --split DEEPCAD-dataset/brepgen_deepcad_data_split_6bit.json \
#     --splits train val \
#     --max-faces 50 --max-edges 500
if __name__ == "__main__":
    main()
