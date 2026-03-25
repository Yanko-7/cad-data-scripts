from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BRepData:
    face_points: np.ndarray
    edge_points: np.ndarray
    outer_edge_indices: np.ndarray
    face_outer_offsets: np.ndarray
    inner_edge_indices: np.ndarray
    inner_loop_offsets: np.ndarray
    face_inner_offsets: np.ndarray


class PointGridVisualizer:
    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize

    def visualize(
        self,
        brep_data: BRepData,
        save_path: str = "brep_visualization.png",
        title="B-Rep Point Grid",
    ):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        all_points = []

        # 1. 渲染面 (Faces)
        num_faces = len(brep_data.face_points)
        colors = cm.tab20(np.linspace(0, 1, max(num_faces, 1)))

        for i, face_grid in enumerate(brep_data.face_points):
            pts = face_grid.reshape(-1, 3)
            if len(pts) == 0:
                continue

            all_points.append(pts)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=[colors[i % 20]],
                s=5,
                alpha=0.6,
                edgecolors="none",
            )

        # 2. 渲染边 (Edges)
        for edge_curve in brep_data.edge_points:
            pts = edge_curve.reshape(-1, 3)
            if len(pts) == 0:
                continue

            all_points.append(pts)
            ax.plot(
                pts[:, 0], pts[:, 1], pts[:, 2], color="black", linewidth=2.0, alpha=0.9
            )

        # 3. 统一坐标轴比例 (等比例显示)
        if all_points:
            concat_pts = np.vstack(all_points)
            min_pt, max_pt = concat_pts.min(axis=0), concat_pts.max(axis=0)
            center = (max_pt + min_pt) / 2
            max_range = (max_pt - min_pt).max() / 2.0

            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            ax.set_box_aspect([1, 1, 1])

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()

        # 核心修改：保存高分辨率图片并关闭画布
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def get_bernstein_poly(t: np.ndarray) -> np.ndarray:
    """生成三次伯恩斯坦多项式矩阵，形状为 (4, len(t))"""
    return np.array([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3])


def eval_rational_bezier_curves(ctrl_pts: np.ndarray, res: int = 50) -> np.ndarray:
    t = np.linspace(0, 1, res)
    B = get_bernstein_poly(t)

    # 1. 物理坐标转齐次坐标: xyz *= w
    hw_ctrls = ctrl_pts.copy()
    hw_ctrls[..., :3] *= hw_ctrls[..., 3:4]

    # 2. 矩阵乘法
    curves_hw = np.einsum("pi, mpk -> mik", B, hw_ctrls)

    # 3. 齐次转回物理: xyz /= w
    xyz, w = curves_hw[..., :3], curves_hw[..., 3:4]
    return xyz / np.where(w == 0, 1e-8, w)


def eval_rational_bezier_surfaces(ctrl_pts: np.ndarray, res: int = 20) -> np.ndarray:
    u = np.linspace(0, 1, res)
    v = np.linspace(0, 1, res)
    Bu = get_bernstein_poly(u)
    Bv = get_bernstein_poly(v)

    # 1. 物理坐标转齐次坐标: xyz *= w
    hw_ctrls = ctrl_pts.copy()
    hw_ctrls[..., :3] *= hw_ctrls[..., 3:4]

    # 2. 张量积计算曲面
    surfs_hw = np.einsum("iu, jv, nijk -> nuvk", Bu, Bv, hw_ctrls)

    # 3. 齐次转回物理: xyz /= w
    xyz, w = surfs_hw[..., :3], surfs_hw[..., 3:4]
    return xyz / np.where(w == 0, 1e-8, w)


folder_path = Path("myABC")

# rglob('*') 会递归查找所有文件
# path.is_file() 确保排除掉文件夹本身
all_files = [str(f) for f in folder_path.rglob("*.npz") if f.is_file()]
indices = np.random.choice(len(all_files), size=20, replace=False)

# Use a list comprehension to grab the items
draw = [all_files[i] for i in indices]
for file_path in draw:
    base_name = Path(file_path).stem
    data = np.load(file_path)
    surfaces = eval_rational_bezier_surfaces(data["face_controls"], res=30)
    curves = eval_rational_bezier_curves(data["edge_controls"], res=50)
    brep_data = BRepData(
        face_points=surfaces,
        edge_points=curves,
        outer_edge_indices=data["outer_edge_indices"],
        face_outer_offsets=data["face_outer_offsets"],
        inner_edge_indices=data["inner_edge_indices"],
        inner_loop_offsets=data["inner_loop_offsets"],
        face_inner_offsets=data["face_inner_offsets"],
    )
    visualizer = PointGridVisualizer()
    visualizer.visualize(
        brep_data,
        save_path=f"visualizations/{base_name}_visualization.png",
        title=f"B-Rep Visualization: {base_name}",
    )
