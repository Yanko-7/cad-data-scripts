from __future__ import annotations

import hashlib
import json
import os
import sys
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SIGNS = np.array(list(product([1, -1], repeat=3)), dtype=np.float32)
_RES_SURF, _RES_CURV = 32, 32
_OCC_EXTS = {".step", ".stp", ".brep"}
_ALL_EXTS = ("*.npz", "*.step", "*.stp", "*.brep")


def _bernstein(t: np.ndarray) -> np.ndarray:
    return np.stack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3]).astype(np.float32)


def _eval_surfaces(ctrl: np.ndarray, res: int = _RES_SURF) -> np.ndarray:
    b = _bernstein(np.linspace(0, 1, res, dtype=np.float32))
    hw = ctrl.astype(np.float32).copy()
    hw[..., :3] *= hw[..., 3:4]
    out = np.einsum("iu,jv,nijk->nuvk", b, b, hw)
    return out[..., :3] / np.where(np.abs(out[..., 3:4]) < 1e-8, 1e-8, out[..., 3:4])


def _eval_curves(ctrl: np.ndarray, res: int = _RES_CURV) -> np.ndarray:
    b = _bernstein(np.linspace(0, 1, res, dtype=np.float32))
    hw = ctrl.astype(np.float32).copy()
    hw[..., :3] *= hw[..., 3:4]
    out = np.einsum("pi,mpk->mik", b, hw)
    return out[..., :3] / np.where(np.abs(out[..., 3:4]) < 1e-8, 1e-8, out[..., 3:4])


def _ctrl_to_pts(face_ctrl: np.ndarray, edge_ctrl: np.ndarray) -> np.ndarray:
    parts = [_eval_surfaces(face_ctrl).reshape(-1, 3)]
    if edge_ctrl.size > 0:
        parts.append(_eval_curves(edge_ctrl).reshape(-1, 3))
    return np.concatenate(parts, axis=0)


def _load_occ_shape(path: Path):
    if path.suffix.lower() in {".step", ".stp"}:
        from OCC.Extend.DataExchange import read_step_file
        return read_step_file(str(path))
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopoDS import TopoDS_Shape
    shape = TopoDS_Shape()
    breptools.Read(shape, str(path), BRep_Builder())
    return shape


def _to_pts(item) -> np.ndarray:
    if isinstance(item, (str, Path)):
        item = Path(item)
        if item.suffix.lower() in _OCC_EXTS:
            from utils import extract_bicubic_features_dir, preprocess_shape
            d = extract_bicubic_features_dir(preprocess_shape(_load_occ_shape(item)))
            return _ctrl_to_pts(np.asarray(d["face_controls"], dtype=np.float32),
                                 np.asarray(d["edge_controls"], dtype=np.float32))
        with np.load(item, mmap_mode="r") as d:
            if "face_controls" in d:
                ec = np.asarray(d["edge_controls"], dtype=np.float32) if "edge_controls" in d else np.empty(0, dtype=np.float32)
                return _ctrl_to_pts(np.asarray(d["face_controls"], dtype=np.float32), ec)
            if "face_points" in d:
                return np.asarray(d["face_points"], dtype=np.float32).reshape(-1, 3)
        raise ValueError(f"Unsupported npz format: {item}")
    arr = np.asarray(item, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[1:3] == (4, 4) and arr.shape[-1] == 4:
        return _eval_surfaces(arr).reshape(-1, 3)
    if arr.ndim == 3 and arr.shape[1] == 4 and arr.shape[-1] == 4:
        return _eval_curves(arr).reshape(-1, 3)
    if arr.ndim < 2 or arr.shape[-1] < 3:
        raise ValueError("Expected xyz in last dimension.")
    return arr[..., :3].reshape(-1, 3)


def _quantize(pts: np.ndarray, n_bits: int) -> np.ndarray:
    q = (1 << n_bits) - 1
    return np.clip((pts + 1.0) * q / 2.0, 0, q).astype(np.int32)


def _canonical_hash(item, n_bits: int, use_sign: bool = True) -> str:
    pts = _to_pts(item)
    if pts.shape[0] < 3:
        raise ValueError("Need ≥3 points for a stable hash.")
    pts -= pts.mean(0)
    if (s := np.max(np.abs(pts))) > 1e-8:
        pts /= s
    vt = np.dtype((np.void, np.dtype(np.int32).itemsize * 3))
    signs = _SIGNS if use_sign else _SIGNS[:1]
    best = None
    for sign in signs:
        q = _quantize(pts * sign, n_bits)
        h = hashlib.sha256(q[np.argsort(np.ascontiguousarray(q).view(vt).ravel())].tobytes()).hexdigest()
        if best is None or h < best:
            best = h
    return best


def _hash_one(args: tuple) -> str | None:
    item, n_bits, use_sign = args
    try:
        return _canonical_hash(item, n_bits, use_sign)
    except Exception:
        return None


def _hash_set(items: Iterable, n_bits: int, desc: str = "", workers: int = 1, use_sign: bool = True) -> set[str]:
    items = list(items)
    result = set()
    if workers <= 1:
        for item in tqdm(items, desc=desc):
            try:
                result.add(_canonical_hash(item, n_bits, use_sign))
            except Exception:
                pass
        return result
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for h in tqdm(ex.map(_hash_one, [(x, n_bits, use_sign) for x in items]), total=len(items), desc=desc):
            if h is not None:
                result.add(h)
    return result


def compute_uniqueness(generated: Iterable, n_bits: int = 8, workers: int = 1, use_sign: bool = True) -> float:
    gen = list(generated)
    return len(_hash_set(gen, n_bits, desc="uniqueness", workers=workers, use_sign=use_sign)) / len(gen) if gen else 0.0


def compute_novelty(generated: Iterable, reference: Iterable, n_bits: int = 8, workers: int = 1, use_sign: bool = True) -> float:
    gen = list(generated)
    if not gen:
        return 0.0
    gen_h = _hash_set(gen, n_bits, desc="hashing generated", workers=workers, use_sign=use_sign)
    ref_h = _hash_set(reference, n_bits, desc="hashing reference", workers=workers, use_sign=use_sign)
    return len(gen_h - ref_h) / len(gen_h) if gen_h else 0.0



def _collect_files(source: str | Path, root_dir: str | Path | None = None, ext: str = ".npz") -> list[Path]:
    source = Path(source)
    if source.is_dir():
        return sorted(p for e in _ALL_EXTS for p in source.rglob(e))
    with open(source) as f:
        data = json.load(f)
    if root_dir is None:
        paths = data if isinstance(data, list) else [p for v in data.values() for p in v]
        return [Path(p) for p in paths]
    prefixes = data if isinstance(data, list) else [p for v in data.values() for p in v]
    lookup = set(prefixes)
    lengths = sorted({len(p) for p in lookup}, reverse=True)
    result = []
    for r, _, files in tqdm(os.walk(root_dir), desc="scanning", unit="dir"):
        for fname in filter(lambda x: x.endswith(ext), files):
            for n in lengths:
                if len(fname) >= n and fname[:n] in lookup:
                    result.append(Path(r) / fname)
                    break
    if not result:
        import warnings
        warnings.warn(f"No {ext} files matched any prefix in {root_dir}. "
                      "Check if the JSON contains full paths (omit --root) or IDs that match filenames.")
    return sorted(result)


def compute_novelty_from_files(
    source: str | Path,
    target_dir: str | Path,
    n_bits: int = 8,
    root_dir: str | Path | None = None,
    ext: str = ".npz",
    workers: int = 1,
    use_sign: bool = True,
) -> float:
    gen = _collect_files(target_dir)
    ref = _collect_files(source, root_dir=root_dir, ext=ext)
    pct = compute_novelty(gen, ref, n_bits, workers=workers, use_sign=use_sign) * 100
    print(f"valid: {len(gen)}, ref: {len(ref)}, novelty: {pct:.2f}%")
    return pct


def compute_uniqueness_from_files(source_dir: str | Path, n_bits: int = 8, workers: int = 1, use_sign: bool = True) -> float:
    files = _collect_files(source_dir)
    pct = compute_uniqueness(files, n_bits, workers=workers, use_sign=use_sign) * 100
    print(f"valid: {len(files)}, uniqueness: {pct:.2f}%")
    return pct

# python metrics/metrics.py novelty \
#     --source configs/filtered_brep_abc_data_split_6bit_paths.json --root /cache/yanko/dataset/brepgen-abc \
#     --target generated_breps \
#     --workers 16

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_nov = sub.add_parser("novelty", help="compute novelty against a reference set")
    p_nov.add_argument("--source", required=True, help="train dir or JSON split file")
    p_nov.add_argument("--target", required=True, help="generated files dir")
    p_nov.add_argument("--root", default=None, help="root dir to resolve JSON prefixes")
    p_nov.add_argument("--ext", default=".npz", help="extension when resolving JSON (default: .npz)")
    p_nov.add_argument("--bits", type=int, default=8)
    p_nov.add_argument("--workers", type=int, default=os.cpu_count())
    p_nov.add_argument("--no-sign", action="store_true")

    p_uniq = sub.add_parser("uniqueness", help="compute uniqueness within a set")
    p_uniq.add_argument("--target", required=True, help="generated files dir")
    p_uniq.add_argument("--bits", type=int, default=8)
    p_uniq.add_argument("--workers", type=int, default=os.cpu_count())
    p_uniq.add_argument("--no-sign", action="store_true")

    args = parser.parse_args()
    if args.cmd == "novelty":
        compute_novelty_from_files(args.source, args.target, args.bits, root_dir=args.root, ext=args.ext, workers=args.workers, use_sign=not args.no_sign)
    else:
        compute_uniqueness_from_files(args.target, args.bits, workers=args.workers, use_sign=not args.no_sign)