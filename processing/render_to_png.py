import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

WORKER_SCRIPT = Path(__file__).parent / "blender_render_worker.py"


def render_one(blender: str, obj_path: Path, out_dir: Path, n_views: int, resolution: int, distance: float, seed: int) -> bool:
    cmd = [
        blender, "--background",
        "--python", str(WORKER_SCRIPT),
        "--",
        "--input", str(obj_path),
        "--output", str(out_dir),
        "--n_views", str(n_views),
        "--resolution", str(resolution),
        "--distance", str(distance),
        "--seed", str(seed),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        print(f"\n[Error] {obj_path.name}\n{result.stderr.decode()[-500:]}")
        return False
    return True


def main(args):
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_files = sorted(in_dir.glob("*.obj"))
    print(f"Found {len(obj_files)} OBJ files → rendering {args.n_views} views each")

    success = 0
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, obj_path in enumerate(obj_files):
            f = executor.submit(render_one, args.blender, obj_path, out_dir, args.n_views, args.resolution, args.distance, i)
            futures[f] = obj_path

        pbar = tqdm(total=len(obj_files), desc="Rendering")
        for f in as_completed(futures):
            try:
                if f.result():
                    success += 1
            except subprocess.TimeoutExpired:
                print(f"\n[Warning] Timeout: {futures[f].name}")
            except Exception as e:
                print(f"\n[Warning] {futures[f].name}: {e}")
            pbar.update(1)
        pbar.close()

    print(f"Done: {success}/{len(obj_files)} rendered to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Directory with OBJ files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for PNGs")
    parser.add_argument("--blender", default="blender", help="Path to Blender executable")
    parser.add_argument("--n_views", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--workers", type=int, default=4, help="Parallel Blender instances")
    main(parser.parse_args())
