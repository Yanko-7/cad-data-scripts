import os
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    check_validity,
    get_fast_stats,
    is_watertight,
    load_and_filter_step,
    preprocess_shape,
    extract_primitive,
)

MAX_FACES = 100
PERFACE_EDGE = 50


def worker_task(file_path, output_dir):
    """
    运行在子进程中的任务
    返回: (状态, 信息)
    """
    file_path = Path(file_path)
    base_name = file_path.stem
    output_dir = Path(output_dir)

    # --- [断点续传] ---
    # 只要输出目录存在该文件名的任意版本 (.npz)，直接跳过
    # 相比于在主进程 glob，这里虽然多了一点点 IO，但代码解耦得更好
    existing = list(output_dir.glob(f"{base_name}_*.npz"))
    if existing:
        return "SKIPPED", None
    # 限制最大文件为 5MB (5 * 1024 * 1024 bytes)
    if file_path.stat().st_size > 5_242_880:
        return "SKIPPED", "File size exceeds 5MB limit"

    try:
        shape = load_and_filter_step(str(file_path))
        total_faces, total_edges, face_edge_counts = get_fast_stats(shape)
        if total_faces > MAX_FACES or any(c > PERFACE_EDGE for c in face_edge_counts):
            return (
                "SKIPPED",
                f"Too complex: {total_faces} faces, max {max(face_edge_counts)} edges/face",
            )
        if not is_watertight(shape):
            return "SKIPPED", "Shape is not watertight"
        shape = preprocess_shape(shape)
        check_validity(shape)
        data = extract_primitive(shape)
        f_count = total_faces
        e_count = total_edges

        # 5. 保存结果
        save_name = output_dir / f"{base_name}_{f_count}_{e_count}.npz"
        np.savez_compressed(save_name, **data)

        return "SUCCESS", None

    except Exception as e:
        # 捕获常规逻辑错误 (如几何无效、空文件等)
        return "ERROR", str(e)


# ==========================================
# 4. 主程序：并行调度与异常管理
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Robust STEP Processor")
    parser.add_argument("--input", "-i", required=True, help="Input folder")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument(
        "--workers", "-w", type=int, default=os.cpu_count() - 10, help="Num processes"
    )
    parser.add_argument(
        "--timeout", "-t", type=int, default=100, help="Timeout per file (seconds)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置错误日志
    logging.basicConfig(
        filename=output_dir / "processing_failures.log",
        level=logging.ERROR,
        format="%(asctime)s - %(message)s",
    )

    print(f"[INFO] Scanning files in {input_dir} ...")
    files = list(input_dir.glob("**/*.step"))
    total_files = len(files)
    print(
        f"[INFO] Found {total_files} files. Workers: {args.workers}, Timeout: {args.timeout}s"
    )

    # 统计计数器
    stats = {"success": 0, "skipped": 0, "error": 0, "crash": 0, "timeout": 0}

    # --- Pebble 并行池 ---
    with ProcessPool(max_workers=args.workers) as pool:
        # 1. 提交任务
        future_map = {}
        for f in files:
            # schedule 立即返回 future 对象
            future = pool.schedule(
                worker_task, args=(f, output_dir), timeout=args.timeout
            )
            future_map[future] = f

        # 2. 处理结果 (带进度条)
        pbar = tqdm(total=total_files, unit="file", smoothing=0.1)

        for future in future_map:
            file_path = future_map[future]
            try:
                # 获取结果 (如果超时或崩溃，这里会抛出异常)
                status, msg = future.result()

                if status == "SUCCESS":
                    stats["success"] += 1
                elif status == "SKIPPED":
                    stats["skipped"] += 1
                else:  # ERROR
                    stats["error"] += 1
                    logging.error(f"ERROR | {file_path.name} | {msg}")

            except TimeoutError:
                stats["timeout"] += 1
                logging.error(f"TIMEOUT | {file_path.name} | Exceeded {args.timeout}s")

            except ProcessExpired:
                stats["crash"] += 1
                logging.error(f"CRASH | {file_path.name} | Segfault/Hard Crash")

            except Exception as e:
                stats["error"] += 1
                logging.error(f"UNKNOWN | {file_path.name} | {e}")

            # 更新进度条描述
            pbar.set_postfix(
                ok=stats["success"],
                skip=stats["skipped"],
                fail=stats["error"] + stats["crash"] + stats["timeout"],
            )
            pbar.update(1)

        pbar.close()

    print("\n" + "=" * 40)
    print("Processing Complete.")
    print(f"Success : {stats['success']}")
    print(f"Skipped : {stats['skipped']}")
    print(f"Errors  : {stats['error']} (Check logs)")
    print(f"Crashes : {stats['crash']} (Segfaults)")
    print(f"Timeouts: {stats['timeout']}")
    print("=" * 40)


if __name__ == "__main__":
    main()
