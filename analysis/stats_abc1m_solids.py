import json
import os
import statistics
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from concurrent.futures import as_completed
from pebble import ProcessPool
from tqdm import tqdm

from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.TopoDS import topods
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Extend.DataExchange import read_step_file
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_fast_stats

JSON_IN = Path("configs/abc1m_stems.json")
STEP_DIR = Path("/cache/yanko/dataset/abc-origin")
STATS_OUT = Path("configs/abc1m_stats.json")

MAX_FACE = 50


def process_step(file_path: Path, target_counts: set[int]) -> list[tuple[int, int]]:
    """返回该文件中所有有效目标 solid 的 (face_count, edge_count) 统计。"""
    stats: list[tuple[int, int]] = []
    try:
        if (shape := read_step_file(str(file_path))) is None:
            return stats

        for count, solid in enumerate(TopologyExplorer(shape).solids()):
            if count not in target_counts:
                continue
            try:
                fixer = ShapeFix_Shape(solid)
                fixer.Perform()
                fixed_solid = topods.Solid(fixer.Shape())
                if BRepCheck_Analyzer(fixed_solid).IsValid():
                    f_count, e_count, _ = get_fast_stats(fixed_solid)
                    if 0 < f_count <= MAX_FACE:
                        stats.append((f_count, e_count))
            except Exception:
                # print(f"Error processing solid {count}: {e}")
                continue

        return stats
    except Exception:
        return stats


def summarize(name: str, values: list[int]) -> dict:
    """给定一维整数序列，返回常用统计量。"""
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
        "stdev": round(statistics.pstdev(values), 2),
        "p90": int(np.percentile(values, 90)),
        "p95": int(np.percentile(values, 95)),
        "p99": int(np.percentile(values, 99)),
    }


def main():
    with open(JSON_IN, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构建 file_id -> 目标 count 集合的映射，实现 O(1) 查询
    target_map = defaultdict(set)
    for stems in data.values():
        for s in stems:
            # s[:8] 是 file_id，s[-4:] 是 count，转成 int 去除前导 0 以匹配 enumerate
            target_map[s[:8]].add(int(s[-4:]))

    # 筛选任务，并将目标 count 集合一并绑定传给子进程
    tasks = []
    for p in STEP_DIR.rglob("*.step"):
        if (file_id := p.name[:8]) in target_map:
            tasks.append((p, target_map[file_id]))

    if not tasks:
        return

    face_counts: list[int] = []
    edge_counts: list[int] = []
    valid_files = 0        # 至少含一个有效 solid 的文件数
    empty_files = 0        # 无有效 solid 的文件数

    with ProcessPool(max_workers=os.cpu_count()) as pool:
        futures = {pool.schedule(process_step, args=(p, counts), timeout=60): p for p, counts in tasks}

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                try:
                    solid_stats = future.result()
                except Exception:
                    solid_stats = []

                if solid_stats:
                    valid_files += 1
                    for f_count, e_count in solid_stats:
                        face_counts.append(f_count)
                        edge_counts.append(e_count)
                else:
                    empty_files += 1

                pbar.set_postfix(files=valid_files, solids=len(face_counts))
                pbar.update(1)

    # 面数直方图（按 5 分箱）
    face_hist = Counter((fc - 1) // 5 * 5 + 1 for fc in face_counts)
    face_hist = {f"{lo}-{lo + 4}": face_hist[lo] for lo in sorted(face_hist)}

    report = {
        "total_files": len(tasks),
        "valid_files": valid_files,
        "empty_files": empty_files,
        "total_valid_solids": len(face_counts),
        "face_count": summarize("face", face_counts),
        "edge_count": summarize("edge", edge_counts),
        "face_histogram_bin5": face_hist,
    }

    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n统计结果已保存至 {STATS_OUT}")


if __name__ == "__main__":
    main()
