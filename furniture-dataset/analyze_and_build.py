import argparse
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

ROOT = Path(__file__).parent
SPLITS = ["train", "val", "test"]


def count_faces(step_path: str) -> tuple:
    from OCC.Extend.DataExchange import read_step_file
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    try:
        shape = read_step_file(step_path)
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        n = 0
        while exp.More():
            n += 1
            exp.Next()
        return Path(step_path).stem, n
    except Exception:
        return Path(step_path).stem, None


def print_stats(split: str, face_counts: dict, total: int):
    counts = list(face_counts.values())
    if not counts:
        print(f"[{split}] No valid files")
        return
    s = sorted(counts)
    n = len(s)
    print(f"\n[{split}] {n}/{total} files analyzed")
    print(f"  min={s[0]}  max={s[-1]}  mean={sum(s)/n:.1f}  median={s[n//2]}")
    freq = Counter(counts)
    print("  Face distribution:")
    for fc, num in sorted(freq.items()):
        bar = "#" * min(num, 50)
        print(f"    {fc:4d} faces: {num:4d}  {bar}")


def analyze_files(label: str, step_files: list) -> dict:
    face_counts = {}
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(count_faces, str(f)): f for f in step_files}
        for future in tqdm(as_completed(futures), total=len(step_files), desc=f"[{label}]"):
            stem, count = future.result()
            if count is not None:
                face_counts[stem] = count
    return face_counts


def with_duplicate_suffix(ids: list[str]) -> list[str]:
    seen = Counter()
    out = []
    for id_ in ids:
        seen[id_] += 1
        n = seen[id_]
        out.append(id_ if n == 1 else f"{id_}__{n}")
    return out


def scan_splits():
    config = {}
    raw_config = {}
    file_config = {}
    for split in SPLITS:
        entries = json.loads((ROOT / f"{split}.json").read_text())
        split_dir = ROOT / split
        step_files = sorted(split_dir.glob("*.step"))
        available = set()
        for p in step_files:
            stem = p.stem
            available.add(stem)
            if "_" in stem:
                available.add(stem.split("_")[-1])
        all_ids = [e["data_id"] for e in entries]
        kept = [id_ for id_ in all_ids if id_ in available]
        skipped = len(all_ids) - len(kept)
        raw_config[split] = kept
        config[split] = with_duplicate_suffix(kept)
        file_config[split] = [str(Path(split) / p.name) for p in step_files]
        print(f"  {split}: {len(kept)}/{len(all_ids)} entries kept  ({skipped} missing STEP files skipped)")

    config_out = ROOT.parent / "configs" / "furniture_split.json"
    config_out.write_text(json.dumps(config, indent=4))
    print(f"Config saved → {config_out}")

    raw_config_out = ROOT.parent / "configs" / "furniture_split_raw_ids.json"
    raw_config_out.write_text(json.dumps(raw_config, indent=4))
    print(f"Raw ids config saved → {raw_config_out}")

    file_config_out = ROOT.parent / "configs" / "furniture_split_paths.json"
    file_config_out.write_text(json.dumps(file_config, indent=4))
    print(f"Training paths config saved → {file_config_out}")

    # resolve actual stems from Furniture/ for exact prefix matching
    furniture_dir = ROOT / "Furniture"
    stems_config: dict[str, list[str]] = {}
    missing_total = 0
    for split in SPLITS:
        entries = json.loads((ROOT / f"{split}.json").read_text())
        stems: list[str] = []
        missing = 0
        for e in entries:
            base = furniture_dir / e["category"] / e["type"]
            candidates = [
                base / f"{e['type']}_{e['data_id']}.step",
                base / f"{e['data_id']}.step",
            ]
            found = next((p for p in candidates if p.exists()), None)
            if found:
                stems.append(found.stem)
            else:
                missing += 1
        stems_config[split] = stems
        missing_total += missing
        print(f"  {split}: {len(stems)}/{len(entries)} resolved  ({missing} not found in Furniture/)")
    stems_out = ROOT.parent / "configs" / "furniture_split_furniture_stems.json"
    stems_out.write_text(json.dumps(stems_config, indent=4))
    print(f"Furniture stems config saved → {stems_out}  (missing total: {missing_total})")

    print("\n=== Face Count Statistics (by split) ===")
    for split in SPLITS: 
        step_files = list((ROOT / split).glob("*.step"))
        if not step_files:
            print(f"[{split}] No STEP files found")
            continue
        face_counts = analyze_files(split, step_files)
        print_stats(split, face_counts, len(step_files))


def scan_furniture():
    furniture_dir = ROOT / "Furniture"
    step_files = list(furniture_dir.rglob("*.step"))
    print(f"Found {len(step_files)} STEP files in Furniture/")

    # group by category (Furniture/{category}/{type}/{id}.step)
    by_category: dict[str, list] = {}
    for f in step_files:
        category = f.parts[-3]
        by_category.setdefault(category, []).append(f)

    print("\n=== Face Count Statistics (Furniture/ by category) ===")
    all_counts: dict[str, int] = {}
    for category in sorted(by_category):
        files = by_category[category]
        face_counts = analyze_files(category, files)
        print_stats(category, face_counts, len(files))
        all_counts.update(face_counts)

    print_stats("ALL", all_counts, len(step_files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["splits", "furniture"],
        default="splits",
        help="splits: analyze train/val/test dirs (default); furniture: analyze Furniture/ by category",
    )
    args = parser.parse_args()

    if args.source == "furniture":
        scan_furniture()
    else:
        scan_splits()


if __name__ == "__main__":
    main()
