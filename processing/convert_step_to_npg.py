import os
import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# OpenCASCADE imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer

# PyVista for fast rendering
import pyvista as pv


def step_to_stl(step_path: str, stl_path: str, deflection: float = 0.5) -> bool:
    """使用 PythonOCC 将 STEP 转换为 STL"""
    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)

        if status != IFSelect_RetDone:
            return False

        reader.TransferRoots()
        shape = reader.OneShape()

        # 网格离散化 (deflection 控制精度，越小越精细但越慢)
        mesh = BRepMesh_IncrementalMesh(shape, deflection)
        mesh.Perform()

        # 导出二进制 STL (比 ASCII 快很多)
        writer = StlAPI_Writer()
        writer.SetASCIIMode(False)
        return bool(writer.Write(shape, stl_path))
    except Exception as e:
        print(f"[Error] Failed to process shape {step_path}: {e}")
        return False

def render_mesh(stl_path: str, png_path: str, resolution: int = 512):
    """使用 PyVista 离屏渲染"""
    plotter = None
    try:
        mesh = pv.read(stl_path)
        mesh = mesh.compute_normals(
            cell_normals=False,
            point_normals=True,
            auto_orient_normals=True, # 关键：强制所有法线朝外
            consistent_normals=True   # 关键：确保相邻面法线一致
        )
        # 开启离屏渲染
        plotter = pv.Plotter(off_screen=True, window_size=[resolution, resolution])

        # 设置材质 (模拟类似深度学习中常用的漫反射灰白材质)
        plotter.add_mesh(mesh, color='white', smooth_shading=True,
                         ambient=0.2, diffuse=0.8, specular=0.1)

        # 设置纯黑或纯白背景，方便后续作为 Mask 或直接送入网络
        plotter.set_background('black')

        # 视角归一化：等距视角并自动缩放相机包围盒
        plotter.view_isometric()
        plotter.reset_camera()

        # 渲染并保存
        plotter.screenshot(png_path)
        return True
    except Exception as e:
        print(f"[Error] Failed to render {stl_path}: {e}")
        return False
    finally:
        if plotter is not None:
            plotter.close()

def process_single_file(step_path: str, output_dir: str):
    """单文件处理管线"""
    filename = Path(step_path).stem
    final_png = os.path.join(output_dir, f"{filename}.png")
    with tempfile.TemporaryDirectory(prefix="step_render_") as temp_dir:
        temp_stl = os.path.join(temp_dir, "mesh.stl")
        if not step_to_stl(step_path, temp_stl, deflection=0.1):
            return False
        if not render_mesh(temp_stl, final_png, resolution=512):
            return False
    print(f"Processed: {filename}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Render STEP files to PNG images")
    parser.add_argument("-i", "--input", required=True, type=Path,
                        help="STEP file or directory (non-recursive)")
    parser.add_argument("-o", "--output", default="rendered_pngs")
    parser.add_argument("-w", "--workers", type=int,
                        default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--xvfb", action="store_true",
                        help="Start a virtual display on a headless Linux server")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.input.is_file() and args.input.suffix.lower() in {".step", ".stp"}:
        step_files = [args.input]
    elif args.input.is_dir():
        step_files = sorted(p for p in args.input.iterdir()
                            if p.is_file() and p.suffix.lower() in {".step", ".stp"})
    else:
        parser.error("--input must be a STEP file or directory")
    if not step_files:
        parser.error("No STEP files found")
    if len({p.stem for p in step_files}) != len(step_files):
        parser.error("Input files have duplicate stems and would overwrite PNG output")
    os.makedirs(args.output, exist_ok=True)
    if args.xvfb:
        pv.start_xvfb()
    max_workers = min(args.workers, len(step_files))
    print(f"Rendering {len(step_files)} files with {max_workers} processes...")
    failures = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, str(p), args.output): p
                   for p in step_files}
        for future in as_completed(futures):
            try:
                if not future.result():
                    failures += 1
            except Exception as exc:
                failures += 1
                print(f"[Error] Failed to process {futures[future]}: {exc}")
    print(f"Completed: {len(step_files) - failures}; failed: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
