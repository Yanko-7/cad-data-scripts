import json
import shutil
from pathlib import Path

SRC_DIR = Path(__file__).parent / "Furniture"
DATASET_DIR = Path(__file__).parent
SPLITS = ["train", "val", "test"]

for split in SPLITS:
    entries = json.loads((DATASET_DIR / f"{split}.json").read_text())
    out_dir = DATASET_DIR / split
    out_dir.mkdir(exist_ok=True)

    missing = []
    for entry in entries:
        data_id, category, type_ = entry["data_id"], entry["category"], entry["type"]
        base = SRC_DIR / category / type_
        src = next((p for p in [base / f"{type_}_{data_id}.step", base / f"{data_id}.step"] if p.exists()), None)
        if src:
            shutil.copy2(src, out_dir / src.name)
        else:
            missing.append(str(base / f"{data_id}.step"))

    print(f"[{split}] {len(entries)} entries, {len(missing)} missing")
    for m in missing:
        print(f"  MISSING: {m}")
