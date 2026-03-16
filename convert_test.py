import numpy as np

from inference import render_and_save, render_and_save_html
from src.data.brep_utils import arrays_to_sequence, normalize_points_with_bbox
from OCC.Extend.DataExchange import write_step_file
from utils import (
    check_euler_poincare,
    check_validity,
    extract_bicubic_features,
    load_and_filter_step,
    preprocess_shape,
    split_to_bicubic,
    get_fast_stats,
)

file_path = "/cache/yanko/DAR-Brep/scripts/cad-data-scripts/00302019_3_27_75.step"
shape = load_and_filter_step(str(file_path))
shape = split_to_bicubic(shape)

shape = preprocess_shape(shape)
check_validity(shape)
a, b, c = get_fast_stats(shape)
print(f"Vertices: {a}, Edges: {b}, ")
# check_euler_poincare(shape)
data = extract_bicubic_features(shape)
# "face_controls": face_controls,  # [F, 4, 4, 4]
# "edge_controls": edge_controls,  # [E, 4, 4]
# "outer_edge_indices": np.array(outer_edge_indices, dtype=np.int32),
# "face_outer_offsets": np.array(face_outer_offsets, dtype=np.int32),
# "inner_edge_indices": np.array(inner_edge_indices, dtype=np.int32),
# "inner_loop_offsets": np.array(inner_loop_offsets, dtype=np.int32),
# "face_inner_offsets": np.array(face_inner_offsets, dtype=np.int32),
# 使用 savez_compressed 进行压缩保存，文件后缀自动加上 .npz
np.savez_compressed("output_data.npz", **data)
face_controls = data["face_controls"]
edge_controls = data["edge_controls"]
num_faces = len(face_controls)
num_edges = len(edge_controls)
face_norms = [normalize_points_with_bbox(face_controls[i]) for i in range(num_faces)]
edge_norms = [normalize_points_with_bbox(edge_controls[i]) for i in range(num_edges)]
fp = np.stack([f for f, b in face_norms])
ep = np.stack([e for e, b in edge_norms])
fb = [b for f, b in face_norms]
eb = [b for e, b in edge_norms]
print(len(fp), len(ep), len(fb), len(eb))

render_and_save_html(fp, ep, fb, eb)

write_step_file(shape, "output_shape.step")
