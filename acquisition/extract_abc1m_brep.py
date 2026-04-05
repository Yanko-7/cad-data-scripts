import json
import os
from collections import defaultdict
from pathlib import Path
from concurrent.futures import as_completed
from pebble import ProcessPool
from tqdm import tqdm

from OCC.Core.ShapeFix import ShapeFix_Solid
from OCC.Core.TopoDS import topods
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepTools import breptools
from OCC.Extend.DataExchange import read_step_file

JSON_IN = Path("configs/abc1m_stems.json")
STEP_DIR = Path("/cache/yanko/dataset/abc")
OUT_DIR = Path("/cache/yanko/dataset/abc_solids_brep")

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
                fixer = ShapeFix_Solid(solid)
                fixer.Perform()
                fixed_solid = topods.Solid(fixer.Shape())

                analyzer = BRepCheck_Analyzer(fixed_solid)
                if analyzer.IsValid():
                    sub_dir = OUT_DIR / file_id[:6]
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    out_path = sub_dir / f"{file_id}_{count:04d}.brep"
                    breptools.Write(fixed_solid, str(out_path))
                    has_valid_solid = True
            except Exception as e:
                with open(OUT_DIR / "errors.log", "a") as log:
                    log.write(f"{file_id}_{count:04d}: {e}\n")
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