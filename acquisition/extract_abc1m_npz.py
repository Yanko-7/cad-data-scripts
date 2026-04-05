import json
import os
import numpy as np
from collections import defaultdict
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
from utils import extract_bicubic_features_dir, get_fast_stats

JSON_IN = Path("configs/abc1m_stems.json")
STEP_DIR = Path("/cache/yanko/dataset/abc")
OUT_DIR = Path("/cache/yanko/dataset/abc_solids_bezier2")

def process_step(file_path: Path, target_counts: set[int]) -> bool:
    file_id = file_path.name[:8]
    try:
        if (shape := read_step_file(str(file_path))) is None:
            return False
            
        has_valid_solid = False
        for count, solid in enumerate(TopologyExplorer(shape).solids()):
            # 核心优化：如果当前实体的 count 不在目标集合中，直接跳过耗时的 OCC 计算
            if count not in target_counts:
                continue
            try:
                fixer = ShapeFix_Shape(solid)
                fixer.Perform()
                fixed_solid = topods.Solid(fixer.Shape())

                if BRepCheck_Analyzer(fixed_solid).IsValid():
                    sub_dir = OUT_DIR / file_id[:6]
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    base_name = f"{file_id}_{count:04d}"
                    shard_dir = sub_dir

                    data = extract_bicubic_features_dir(fixed_solid)

                    f_count, e_count, _ = get_fast_stats(fixed_solid)
                    data["f_count"] = np.array([f_count], dtype=np.int32)
                    data["e_count"] = np.array([e_count], dtype=np.int32)
                    # 动态生成包含面和边数量的最终文件名
                    final_filename = f"{base_name.split('_')[0]}_{base_name.split('_')[1]}_f{f_count}_e{e_count}.npz"
                    final_path = shard_dir / final_filename
                    temp_path = shard_dir / f"{base_name}.tmp.npz"

                    np.savez_compressed(temp_path, **data)
                    temp_path.replace(final_path)
                    has_valid_solid = True
            except Exception as e:
                # print(f"Error processing solid {count}: {e}")
                continue
                
        return has_valid_solid
    except Exception:
        return False

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    if not tasks: return

    success_cnt, fail_cnt = 0, 0

    with ProcessPool(max_workers=os.cpu_count()) as pool:
        # 将 target_counts 传入 process_step
        futures = {pool.schedule(process_step, args=(p, counts), timeout=60): p for p, counts in tasks}
        
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                try:
                    if future.result():
                        success_cnt += 1
                    else:
                        fail_cnt += 1
                except Exception:
                    fail_cnt += 1
                
                pbar.set_postfix(succ=success_cnt, fail=fail_cnt)
                pbar.update(1)

if __name__ == "__main__":
    main()