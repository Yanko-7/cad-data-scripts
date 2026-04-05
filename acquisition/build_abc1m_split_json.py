import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

JSON_IN = Path("configs/abc1m_stems.json")
# STEP_DIR = Path("/cache/yanko/dataset/abc-splited") # 替换为你的 step 文件夹路径
JSON_OUT = Path("configs/abc1m_split.json")

def parse_id(p: Path) -> str | None:
    parts = p.stem.split('_')
    # 提取并补零：parts[0] 为 8位数，parts[1] 补齐 4位
    return f"{parts[0]}_{parts[-1].zfill(4)}" if len(parts) >= 2 else None

def main():
    with open(JSON_IN, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data = {}
    for k, v in data.items():
        new_data[k] = [parse_id(Path(x)) for x in v]
    
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(new_data, f,indent=4)

if __name__ == "__main__":
    main()