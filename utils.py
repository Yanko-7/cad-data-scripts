import numpy as np
from OCC.Core import TopoDS
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.BRepCheck import BRepCheck_Analyzer, BRepCheck_ListIteratorOfListOfStatus
from OCC.Core.BRepTools import BRepTools_WireExplorer, breptools
from OCC.Core.Geom import (
    Geom_BezierCurve,
    Geom_BezierSurface,
    Geom_RectangularTrimmedSurface,
    Geom_TrimmedCurve,
)
from OCC.Core.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C2,GeomAbs_C1
from OCC.Core.GeomConvert import (
    GeomConvert_BSplineCurveToBezierCurve,
    GeomConvert_BSplineSurfaceToBezierSurface,
    geomconvert,
)
from OCC.Core.Precision import precision
from OCC.Core.ShapeCustom import ShapeCustom_RestrictionParameters, shapecustom
from OCC.Core.ShapeFix import ShapeFix_Shape
from OCC.Core.ShapeUpgrade import (
    ShapeUpgrade_ShapeConvertToBezier,
    ShapeUpgrade_ShapeDivideClosed,
    ShapeUpgrade_ShapeDivideClosedEdges,
)
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopoDS import TopoDS_Shape, topods
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
)
from OCC.Extend.DataExchange import read_step_file
import os
from OCC.Extend.DataExchange import write_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.TColgp import TColgp_Array1OfPnt


def estimate_token_count(n_faces: int, n_edges: int) -> int:
    # Sequence =
    #   [BOS] ....................................... +1

    #   Face 1:
    #     Face_Skeleton (Start+BBox+Geom+End) ....... +11
    #     Loop_Skeleton (Start+End) ................. +2
    #     Edge A (New) .............................. +11
    #     Edge B (New) .............................. +11

    #   Face 2:
    #     Face_Skeleton ............................. +11
    #     Loop_Skeleton ............................. +2
    #     Edge A (Ref) .............................. +2
    #     Edge C (New) .............................. +11
    #   ...
    #   [EOS] ....................................... +1
    return 13 * (n_faces + n_edges) + 2


# copy from occwl
def split_all_closed_faces(shape, max_tol=0.01, precision=0.01, num_splits=1):
    """
    Split all the closed faces in this shape

    Args:
        max_tol (float, optional): Maximum tolerance allowed. Defaults to 0.01.
        precision (float, optional): Precision of the tool when splitting. Defaults to 0.01.
        num_splits (int, optional): Number of splits to perform. Each split face will result in num_splits + 1 faces. Defaults to 1.

    Returns:
        occwl.*.*: Shape with closed faces split
    """
    divider = ShapeUpgrade_ShapeDivideClosed(shape)
    divider.SetPrecision(precision)
    divider.SetMinTolerance(0.1 * max_tol)
    divider.SetMaxTolerance(max_tol)
    divider.SetNbSplitPoints(num_splits)
    ok = divider.Perform()
    if not ok:
        # Splitting failed or there were no closed faces to split
        # Return the original shape
        return shape
    return divider.Result()


# copy from occwl
def split_all_closed_edges(shape, max_tol=0.01, precision=0.01, num_splits=1):
    """
    Split all the closed edges in this shape

    Args:
        max_tol (float, optional): Maximum tolerance allowed. Defaults to 0.01.
        precision (float, optional): Precision of the tool when splitting. Defaults to 0.01.
        num_splits (int, optional): Number of splits to perform. Each split edge will result in num_splits + 1 edges. Defaults to 1.

    Returns:
        occwl.*.*: Shape with closed edges split
    """
    divider = ShapeUpgrade_ShapeDivideClosedEdges(shape)
    divider.SetPrecision(precision)
    divider.SetMinTolerance(0.1 * max_tol)
    divider.SetMaxTolerance(max_tol)
    divider.SetNbSplitPoints(num_splits)
    ok = divider.Perform()
    if not ok:
        # Splitting failed or there were no closed edges to split
        # Return the original shape
        return shape
    return divider.Result()


def sample_face_points(
    u_num: int, v_num: int, face: TopoDS.TopoDS_Face
) -> np.ndarray:  # [u_num,v_num,3]
    adaptor = BRepAdaptor_Surface(face)
    umin, umax, vmin, vmax = breptools.UVBounds(face)
    if (
        precision.IsInfinite(umin)
        or precision.IsInfinite(umax)
        or precision.IsInfinite(vmin)
        or precision.IsInfinite(vmax)
    ):
        raise ValueError("Invalid UV bounds for face.")
    us = np.linspace(umin, umax, u_num)
    vs = np.linspace(vmin, vmax, v_num)
    points = [[adaptor.Value(float(u), float(v)).Coord() for v in vs] for u in us]
    return np.array(points)


def sample_edge_points(num: int, edge: TopoDS.TopoDS_Edge) -> np.ndarray:  # [num,3]
    adapter = BRepAdaptor_Curve(edge)
    st, ed = adapter.FirstParameter(), adapter.LastParameter()
    if precision.IsInfinite(st) or precision.IsInfinite(ed):
        raise ValueError("Invalid parameter bounds for edge.")
    t_s = np.linspace(st, ed, num)
    samples = [adapter.Value(t).Coord() for t in t_s]
    return np.array(samples)


def extract_primitive(shape: TopoDS_Shape):
    edge_map = TopTools_IndexedMapOfShape()

    all_face_points = []
    all_edge_points = []

    outer_edge_indices = []
    face_outer_offsets = [0]

    inner_edge_indices = []
    inner_loop_offsets = [0]
    face_inner_offsets = [0]

    edge_face_temp = {}

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0

    while face_exp.More():
        face = topods.Face(face_exp.Current())

        all_face_points.append(sample_face_points(32, 32, face))

        outer_wire = breptools.OuterWire(face)

        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            is_outer = wire.IsSame(outer_wire)
            current_loop_edge_ids = []

            edge_exp = BRepTools_WireExplorer(wire)
            while edge_exp.More():
                edge = topods.Edge(edge_exp.Current())

                idx = edge_map.FindIndex(edge)
                if idx == 0:
                    idx = edge_map.Add(edge)
                    all_edge_points.append(sample_edge_points(32, edge))
                    edge_face_temp[idx - 1] = []

                global_edge_id = idx - 1
                current_loop_edge_ids.append(global_edge_id)

                existing_faces = edge_face_temp[global_edge_id]
                if not existing_faces or existing_faces[-1] != face_idx:
                    existing_faces.append(face_idx)

                edge_exp.Next()

            if is_outer:
                outer_edge_indices.extend(current_loop_edge_ids)
                face_outer_offsets.append(len(outer_edge_indices))
            else:
                if current_loop_edge_ids:
                    inner_edge_indices.extend(current_loop_edge_ids)
                    inner_loop_offsets.append(len(inner_edge_indices))

            wire_exp.Next()

        face_inner_offsets.append(len(inner_loop_offsets) - 1)
        face_idx += 1
        face_exp.Next()

    try:
        with np.errstate(over="raise", invalid="raise"):
            all_face_points = np.array(all_face_points)
            all_edge_points = np.array(all_edge_points)
            all_pts = all_face_points.reshape(-1, 3)

            if all_pts.size > 0:
                vmin, vmax = all_pts.min(axis=0), all_pts.max(axis=0)
                center = (vmin + vmax) / 2
                max_span = np.max(vmax - vmin)
                if max_span < 1e-8:
                    raise ValueError("Normalization error: max_span is too small.")

                scale = 2.0 / max_span

                if all_edge_points.size > 0:
                    all_edge_points = (all_edge_points - center) * scale
                if all_face_points.size > 0:
                    all_face_points = (all_face_points - center) * scale

    except Exception as e:
        print(f"Normalization error: {e}")
        raise

    if all_face_points.size == 0 or all_edge_points.size == 0:
        raise ValueError("Extraction error: No face or edge points extracted.")
    if len(all_face_points) != len(face_outer_offsets) - 1:
        raise ValueError("Extraction error: Face count mismatch in outer offsets.")
    if len(all_face_points) != len(face_inner_offsets) - 1:
        raise ValueError("Extraction error: Inner loop offsets count mismatch.")
    if len(outer_edge_indices) != face_outer_offsets[-1]:
        raise ValueError("Extraction error: Outer edge indices count mismatch.")
    if len(inner_loop_offsets) != face_inner_offsets[-1] + 1:
        raise ValueError("Extraction error: Inner edge indices count mismatch.")
    return {
        # Geometry
        "face_points": all_face_points,
        "edge_points": all_edge_points,
        # Face -> Edge Topology (Outer)
        "outer_edge_indices": np.array(outer_edge_indices, dtype=np.int32),
        "face_outer_offsets": np.array(face_outer_offsets, dtype=np.int32),
        # Face -> Edge Topology (Inner)
        "inner_edge_indices": np.array(inner_edge_indices, dtype=np.int32),
        "inner_loop_offsets": np.array(inner_loop_offsets, dtype=np.int32),
        "face_inner_offsets": np.array(face_inner_offsets, dtype=np.int32),
    }


def convert_special_surface(surf, u1, u2, v1, v2):
    trimmed = Geom_RectangularTrimmedSurface(surf, u1, u2, v1, v2)
    bspline = geomconvert.SurfaceToBSplineSurface(trimmed)
    if bspline is None:
        raise ValueError("严重错误：截断曲面转 B-Spline 失败。")
    converter = GeomConvert_BSplineSurfaceToBezierSurface(bspline)
    nb_u = converter.NbUPatches()
    nb_v = converter.NbVPatches()
    if nb_u > 1 or nb_v > 1:
        raise ValueError(
            f"严格限制：该 B-Spline 曲面包含内部节点，会被切分为 {nb_u}x{nb_v} 个 Bezier 面，拒绝处理！"
        )
    bspline = geomconvert.SurfaceToBSplineSurface(converter.Patch(1, 1))
    return bspline


def extract_bicubic_surf_patches(face):
    face_ds = topods.Face(face)
    surf = BRep_Tool.Surface(face_ds)

    surf_type = surf.DynamicType().Name()

    u1, u2, v1, v2 = breptools.UVBounds(face_ds)
    if (
        precision.IsInfinite(u1)
        or precision.IsInfinite(u2)
        or precision.IsInfinite(v1)
        or precision.IsInfinite(v2)
    ):
        raise ValueError("Invalid UV bounds for face.")

    if surf_type == "Geom_BSplineSurface":
        bspline = Geom_BSplineSurface.DownCast(surf)
    elif surf_type == "Geom_BezierSurface":
        bezier = Geom_BezierSurface.DownCast(surf)
        bspline = geomconvert.SurfaceToBSplineSurface(bezier)
    else:
        bspline = convert_special_surface(surf, u1, u2, v1, v2)

    bspline.Segment(u1, u2, v1, v2)

    if bspline.UDegree() < 3 or bspline.VDegree() < 3:
        bspline.IncreaseDegree(max(3, bspline.UDegree()), max(3, bspline.VDegree()))

    if bspline.NbUPoles() != 4 or bspline.NbVPoles() != 4:
        raise ValueError(
            f"Not a single Bicubic patch: {bspline.NbUPoles()}x{bspline.NbVPoles()}"
        )

    bspline.Transform(face_ds.Location().Transformation())
    is_rational = bspline.IsURational() or bspline.IsVRational()

    poles = []
    for i in range(1, 5):
        row = []
        for j in range(1, 5):
            p = bspline.Pole(i, j)
            w = bspline.Weight(i, j) if is_rational else 1.0
            row.append([p.X(), p.Y(), p.Z(), w])
        poles.append(row)

    return np.array(poles, dtype=np.float32)


def convert_special_curve(crv, first, last):
    trimmed_crv = Geom_TrimmedCurve(crv, first, last)

    bspline_crv = geomconvert.CurveToBSplineCurve(trimmed_crv)
    if bspline_crv is None:
        raise ValueError("截断曲线转 B-Spline 失败。")

    converter = GeomConvert_BSplineCurveToBezierCurve(bspline_crv)

    nb_arcs = converter.NbArcs()

    if nb_arcs > 1:
        raise ValueError(
            f"严格限制：该 B-Spline 曲线包含内部节点，会被切分为 {nb_arcs} 段 Bezier 曲线，拒绝处理！"
        )
    bezier_crv = converter.Arc(1)
    if bezier_crv is None:
        raise ValueError("提取 Bezier 曲线段失败。")
    return geomconvert.CurveToBSplineCurve(bezier_crv)


def extract_bicubic_curv_patches(edge):
    edge_ds = topods.Edge(edge)
    curve, t1, t2 = BRep_Tool.Curve(edge_ds)
    if curve is None:
        raise ValueError(
            "严重错误：该 Edge 没有底层的 3D 曲线 (可能是退化边 Degenerated Edge)。"
        )

    if precision.IsInfinite(t1) or precision.IsInfinite(t2):
        raise ValueError("Invalid bounds for curve (Infinite).")

    curve_type = curve.DynamicType().Name()

    if curve_type == "Geom_BSplineCurve":
        bspline = Geom_BSplineCurve.DownCast(curve)
    elif curve_type == "Geom_BezierCurve":
        bezier = Geom_BezierCurve.DownCast(curve)
        bspline = geomconvert.CurveToBSplineCurve(bezier)
    else:
        bspline = convert_special_curve(curve, t1, t2)

    bspline.Segment(t1, t2)
    if bspline.Degree() < 3:
        bspline.IncreaseDegree(3)

    if bspline.NbPoles() != 4:
        raise ValueError(f"Not a single Cubic curve: {bspline.NbPoles()}")

    bspline.Transform(edge_ds.Location().Transformation())
    is_rational = bspline.IsRational()

    poles = []
    for i in range(1, 5):
        p = bspline.Pole(i)
        w = bspline.Weight(i) if is_rational else 1.0
        poles.append([p.X(), p.Y(), p.Z(), w])

    return np.array(poles, dtype=np.float32)


def extract_bicubic_features(shape: TopoDS_Shape):
    edge_map = TopTools_IndexedMapOfShape()

    # 核心数据：控制点 (Control Points / Poles)
    face_controls = []  # Shape: [N_faces, 4, 4, 4]
    edge_controls = []  # Shape: [N_edges, 4, 4]

    # 拓扑结构：索引与偏移
    outer_edge_indices = []
    face_outer_offsets = [0]

    inner_edge_indices = []
    inner_loop_offsets = [0]
    face_inner_offsets = [0]

    # 临时映射：Edge ID -> [Face ID, ...]
    edge_to_face_map = {}

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0

    while face_exp.More():
        face = topods.Face(face_exp.Current())

        face_controls.append(extract_bicubic_surf_patches(face))

        outer_wire = breptools.OuterWire(face)
        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)

        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            is_outer = wire.IsSame(outer_wire)

            loop_edge_ids = []

            edge_exp = BRepTools_WireExplorer(wire)
            while edge_exp.More():
                edge = topods.Edge(edge_exp.Current())

                e_idx = edge_map.FindIndex(edge)
                if e_idx == 0:
                    e_idx = edge_map.Add(edge)
                    edge_controls.append(extract_bicubic_curv_patches(edge))
                    edge_to_face_map[e_idx - 1] = []

                global_id = e_idx - 1
                loop_edge_ids.append(global_id)

                # 记录邻接
                adj_faces = edge_to_face_map[global_id]
                if not adj_faces or adj_faces[-1] != face_idx:
                    adj_faces.append(face_idx)

                edge_exp.Next()

            # 3. 更新 Wire 拓扑
            if is_outer:
                outer_edge_indices.extend(loop_edge_ids)
                face_outer_offsets.append(len(outer_edge_indices))
            else:
                if loop_edge_ids:
                    inner_edge_indices.extend(loop_edge_ids)
                    inner_loop_offsets.append(len(inner_edge_indices))

            wire_exp.Next()

        face_inner_offsets.append(len(inner_loop_offsets) - 1)
        face_idx += 1
        face_exp.Next()

    try:
        face_controls = np.array(face_controls, dtype=np.float32)
        edge_controls = np.array(edge_controls, dtype=np.float32)

        all_pts_4d = face_controls.reshape(-1, 4)
        if edge_controls.size > 0:
            all_pts_4d = np.concatenate([all_pts_4d, edge_controls.reshape(-1, 4)])

        if all_pts_4d.size > 0:
            xyz_physical = all_pts_4d[:, :3]
            vmin, vmax = xyz_physical.min(axis=0), xyz_physical.max(axis=0)
            center = (vmin + vmax) / 2.0
            scale = 2.0 / (np.max(vmax - vmin) + 1e-8)
            face_controls[..., :3] = (face_controls[..., :3] - center) * scale
            if edge_controls.size > 0:
                edge_controls[..., :3] = (edge_controls[..., :3] - center) * scale
    except Exception as e:
        raise ValueError(f"Normalization failed: {e}")
    return {
        "face_controls": face_controls,  # [F, 4, 4, 4]
        "edge_controls": edge_controls,  # [E, 4, 4]
        "outer_edge_indices": np.array(outer_edge_indices, dtype=np.int32),
        "face_outer_offsets": np.array(face_outer_offsets, dtype=np.int32),
        "inner_edge_indices": np.array(inner_edge_indices, dtype=np.int32),
        "inner_loop_offsets": np.array(inner_loop_offsets, dtype=np.int32),
        "face_inner_offsets": np.array(face_inner_offsets, dtype=np.int32),
    }


def check_euler_poincare(shape):
    """
    Check whether the shape contains only a single entity, and calculate the Euler characteristic (Chi = V - E + F).
    Return the value of Chi (2 for simple convex shapes, and it decreases by 2 for each additional hole).
    """
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    if not exp.More():
        raise ValueError("Error: No solids found.")
    exp.Next()
    if exp.More():
        raise ValueError("Error: Multiple solids found. Expected exactly one.")

    map_v = TopTools_IndexedMapOfShape()
    map_e = TopTools_IndexedMapOfShape()
    map_f = TopTools_IndexedMapOfShape()

    topexp.MapShapes(shape, TopAbs_VERTEX, map_v)
    topexp.MapShapes(shape, TopAbs_EDGE, map_e)
    topexp.MapShapes(shape, TopAbs_FACE, map_f)

    v, e, f = map_v.Size(), map_e.Size(), map_f.Size()

    chi = v - e + f

    if chi % 2 != 0:
        raise ValueError(
            f"Topology Error: Euler Characteristic is odd ({chi}). Shape likely not closed."
        )


def restrict_infinite_surfaces(shape):
    builder = BRep_Builder()
    ex = TopExp_Explorer(shape, TopAbs_FACE)

    while ex.More():
        face = topods.Face(ex.Current())
        surf = BRep_Tool.Surface(face)

        umin, umax, vmin, vmax = breptools.UVBounds(face)

        is_uv_finite = not (
            precision.IsInfinite(umin)
            or precision.IsInfinite(umax)
            or precision.IsInfinite(vmin)
            or precision.IsInfinite(vmax)
        )

        if is_uv_finite:
            try:
                trimmed_surf = Geom_RectangularTrimmedSurface(
                    surf, umin, umax, vmin, vmax
                )
                bspline_surf = geomconvert.SurfaceToBSplineSurface(trimmed_surf)

                builder.UpdateFace(face, bspline_surf, face.Location(), 1e-4)
            except Exception as e:
                raise ValueError(
                    f"Surface conversion error: Failed to convert face to B-spline surface. Error: {e}"
                )

        ex.Next()
    return shape


def print_shape_errors(shape):
    analyzer = BRepCheck_Analyzer(shape)
    if analyzer.IsValid():
        return

    types = [TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID]
    for t in types:
        exp = TopExp_Explorer(shape, t)
        while exp.More():
            sub = exp.Current()
            res = analyzer.Result(sub)
            status_list = res.Status()

            it = BRepCheck_ListIteratorOfListOfStatus(status_list)
            while it.More():
                status = it.Value()
                if status != 0:
                    print(f"ShapeType: {sub.ShapeType()} | Error Status: {status}")
                it.Next()
            exp.Next()


def split_to_bicubic(shape):

    shape = BRepBuilderAPI_NurbsConvert(shape).Shape()
    params = ShapeCustom_RestrictionParameters()
    shape = shapecustom.BSplineRestriction(
        shape, 0.05, 0.005, 3, 4, GeomAbs_C0, GeomAbs_C0, True, False, params
    )
    fixer = ShapeFix_Shape(shape)
    fixer.Perform()
    shape = fixer.Shape()
    converter = ShapeUpgrade_ShapeConvertToBezier(shape)
    converter.SetSurfaceConversion(True)
    converter.Set2dConversion(True)
    converter.Set3dConversion(True)
    converter.Perform()
    split_shape = converter.Result()
    fixer = ShapeFix_Shape(split_shape)
    fixer.Perform()
    return fixer.Shape()


def load_and_filter_step(filename):
    try:
        origin_shape = read_step_file(filename)

        fixer = ShapeFix_Shape(origin_shape)
        fixer.Perform()
        shape = fixer.Shape()
        # shape = get_most_complex_subshape(shape)
    except Exception as e:
        raise ValueError(f"Failed to load STEP file: {e}")
    analyzer = BRepCheck_Analyzer(shape)
    if not analyzer.IsValid():
        raise ValueError("Loaded shape is not valid.")
    return shape


def check_validity(shape):
    analyzer = BRepCheck_Analyzer(shape)
    if not analyzer.IsValid():
        raise ValueError("Shape is not valid.")


def get_most_complex_subshape(shape: TopoDS_Shape) -> TopoDS_Shape:
    if shape.ShapeType() != TopAbs_COMPOUND:
        return shape

    def count_faces(s: TopoDS_Shape) -> int:
        exp = TopExp_Explorer(s, TopAbs_FACE)
        count = 0
        while exp.More():
            count += 1
            exp.Next()
        return count

    for topo_type in (TopAbs_COMPSOLID, TopAbs_SOLID, TopAbs_SHELL):
        exp = TopExp_Explorer(shape, topo_type)
        candidates = []
        while exp.More():
            candidates.append(exp.Current())
            exp.Next()
        if candidates:
            return max(candidates, key=count_faces)

    return shape


def is_watertight(shape: TopoDS_Shape) -> bool:
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    for i in range(1, edge_face_map.Size() + 1):
        if edge_face_map.FindFromIndex(i).Size() < 2:
            return False
    return True


def preprocess_shape(shape: TopoDS_Shape) -> TopoDS_Shape:
    shape = split_all_closed_faces(shape, num_splits=1)
    shape = split_all_closed_edges(shape, num_splits=1)
    return shape


def get_fast_stats(shape):
    global_map_f = TopTools_IndexedMapOfShape()
    global_map_e = TopTools_IndexedMapOfShape()

    topexp.MapShapes(shape, TopAbs_FACE, global_map_f)
    topexp.MapShapes(shape, TopAbs_EDGE, global_map_e)

    total_faces = global_map_f.Size()
    total_edges = global_map_e.Size()

    face_edge_counts = []

    local_map_e = TopTools_IndexedMapOfShape()

    for i in range(1, total_faces + 1):
        face = topods.Face(global_map_f.FindKey(i))

        local_map_e.Clear()
        topexp.MapShapes(face, TopAbs_EDGE, local_map_e)

        face_edge_counts.append(local_map_e.Size())

    return total_faces, total_edges, face_edge_counts


def get_topo_count(shape, topo_type):
    topo_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, topo_type, topo_map)
    return topo_map.Size()


def split_and_classify_step(input_file, base_dir="output_solids", bin_size=30):
    file_id = os.path.basename(input_file)[:8]
    shape = read_step_file(input_file)

    for count, solid in enumerate(TopologyExplorer(shape).solids(), 1):
        fixer = ShapeFix_Shape(solid)
        fixer.Perform()
        fixed_solid = topods.Solid(fixer.Shape())

        if BRepCheck_Analyzer(fixed_solid).IsValid():
            faces = get_topo_count(fixed_solid, TopAbs_FACE)
            edges = get_topo_count(fixed_solid, TopAbs_EDGE)

            lower = (faces // bin_size) * bin_size
            upper = lower + bin_size - 1
            out_dir = os.path.join(base_dir, f"faces_{lower}_{upper}")
            os.makedirs(out_dir, exist_ok=True)

            out_name = f"{file_id}_{count}_{faces}_{edges}.step"
            out_path = os.path.join(out_dir, out_name)

            write_step_file(fixed_solid, out_path)


def get_info_pipeline(file_path):
    shape = load_and_filter_step(file_path)
    shape = split_to_bicubic(shape)
    shape = preprocess_shape(shape)
    print_shape_errors(shape)
    check_validity(shape)
    check_euler_poincare(shape)
    return extract_bicubic_features(shape)


# print(get_info_pipeline("test.step"))


def extract_or_fit_bicubic_patch(face) -> np.ndarray:
    face_ds = topods.Face(face)
    u1, u2, v1, v2 = breptools.UVBounds(face_ds)
    if any(map(precision.IsInfinite, (u1, u2, v1, v2))):
        raise ValueError("Invalid UV bounds for face.")

    surf = BRep_Tool.Surface(face_ds)
    bspline = None

    # 1. 尝试直接提取并转换为单一 Bicubic B-Spline 面片
    try:
        surf_type = surf.DynamicType().Name()
        if surf_type == "Geom_BSplineSurface":
            copied_surf = surf.Copy()
            bspline = Geom_BSplineSurface.DownCast(copied_surf)
        elif surf_type == "Geom_BezierSurface":
            bspline = geomconvert.SurfaceToBSplineSurface(
                Geom_BezierSurface.DownCast(surf)
            )
        else:
            trimmed = Geom_RectangularTrimmedSurface(surf, u1, u2, v1, v2)
            bspline = geomconvert.SurfaceToBSplineSurface(trimmed)
            conv = GeomConvert_BSplineSurfaceToBezierSurface(bspline)
            if conv.NbUPatches() == 1 and conv.NbVPatches() == 1:
                bspline = geomconvert.SurfaceToBSplineSurface(conv.Patch(1, 1))

        if bspline:
            bspline.Segment(u1, u2, v1, v2)
            bspline.IncreaseDegree(max(3, bspline.UDegree()), max(3, bspline.VDegree()))
            if bspline.NbUPoles() != 4 or bspline.NbVPoles() != 4:
                bspline = None
    except Exception:
        bspline = None

    # 2. 回退方案：采样 4x4 个点进行双三次拟合 (4点插值必定生成 Degree 3 且无内部节点)
    if not bspline:
        bspline = point2bspline(face_ds, u1, u2, v1, v2)
        if bspline is None:
            raise RuntimeError("Fallback B-Spline surface fitting failed.")

    # 3. 施加变换并提取权重与控制点
    bspline.Transform(face_ds.Location().Transformation())
    is_rational = bspline.IsURational() or bspline.IsVRational()
    assert bspline.NbUPoles() == 4 and bspline.NbVPoles() == 4, (
        "Final B-Spline surface must have exactly 4x4 poles."
    )
    poles = np.zeros((4, 4, 4), dtype=np.float32)
    for i in range(1, 5):
        for j in range(1, 5):
            p = bspline.Pole(i, j)
            w = bspline.Weight(i, j) if is_rational else 1.0
            poles[i - 1, j - 1] = [p.X(), p.Y(), p.Z(), w]

    return poles

def _compute_tol(pts, is_curve):
    pts = np.asarray(pts)
    diag = np.linalg.norm(np.ptp(pts, axis=0))
    if diag < 1e-10: 
        return 1e-7
    
    cov = np.cov(pts.T)
    extents = np.ptp(pts @ np.linalg.eigh(cov)[1], axis=0) if cov.ndim == 2 else np.ptp(pts, axis=0)
    feat_size = np.sort(extents)[-1 if is_curve else -2]
    
    return np.clip(diag * 1e-3, 1e-7, feat_size * 0.2)
    
def point2bspline_curve(edge_ds, t1, t2):
    adaptor = BRepAdaptor_Curve(edge_ds)
    pts = TColgp_Array1OfPnt(1, 16)
    raw_pts = []
    
    for i, t in enumerate(np.linspace(t1, t2, 16), 1):
        p = adaptor.Value(float(t))
        pts.SetValue(i, p)
        raw_pts.append([p.X(), p.Y(), p.Z()])
        
    tol = _compute_tol(raw_pts, is_curve=True)
    
    for _ in range(9):
        fitter = GeomAPI_PointsToBSpline(pts, 3, 3, GeomAbs_C1, tol)
        if fitter.IsDone() and fitter.Curve().NbPoles() == 4:
            return fitter.Curve()
        tol *= 2
        
    return None

def point2bspline(face_ds, u1, u2, v1, v2):
    adaptor = BRepAdaptor_Surface(face_ds)
    pts = TColgp_Array2OfPnt(1, 4, 1, 4)
    raw_pts = []
    
    for i, u in enumerate(np.linspace(u1, u2, 4), 1):
        for j, v in enumerate(np.linspace(v1, v2, 4), 1):
            p = adaptor.Value(float(u), float(v))
            pts.SetValue(i, j, p)
            raw_pts.append([p.X(), p.Y(), p.Z()])
            
    tol = _compute_tol(raw_pts, is_curve=False)
    
    for _ in range(9):
        fitter = GeomAPI_PointsToBSplineSurface(pts, 3, 3, GeomAbs_C1, tol)
        if fitter.IsDone() and fitter.Surface().NbUPoles() == 4 and fitter.Surface().NbVPoles() == 4:
            return fitter.Surface()
        tol *= 2
        
    return None


def extract_or_fit_cubic_curve(edge) -> np.ndarray:
    edge_ds = topods.Edge(edge)
    curve, t1, t2 = BRep_Tool.Curve(edge_ds)

    if curve is None:
        raise ValueError("Edge has no underlying 3D curve.")
    if any(map(precision.IsInfinite, (t1, t2))):
        raise ValueError("Invalid bounds for curve (Infinite).")

    bspline = None

    # 1. 尝试直接提取并转换为单一 Cubic B-Spline 曲线段
    try:
        curve_type = curve.DynamicType().Name()
        if curve_type == "Geom_BSplineCurve":
            copied_curve = curve.Copy()
            bspline = Geom_BSplineCurve.DownCast(copied_curve)
        elif curve_type == "Geom_BezierCurve":
            bspline = geomconvert.CurveToBSplineCurve(Geom_BezierCurve.DownCast(curve))
        else:
            trimmed = Geom_TrimmedCurve(curve, t1, t2)
            bspline = geomconvert.CurveToBSplineCurve(trimmed)
            conv = GeomConvert_BSplineCurveToBezierCurve(bspline)
            if conv.NbArcs() == 1:
                bspline = geomconvert.CurveToBSplineCurve(conv.Arc(1))

        if bspline:
            bspline.Segment(t1, t2)
            bspline.IncreaseDegree(max(3, bspline.Degree()))
            # 严格校验：单一三次曲线控制点必为 4
            if bspline.NbPoles() != 4:
                bspline = None
    except Exception:
        bspline = None

    # 2. 回退方案：采样 4 个点进行三次拟合 (4点插值必定生成 Degree 3 且无内部节点的线)
    if not bspline:
        bspline = point2bspline_curve(edge_ds, t1, t2)
        if bspline is None:
            raise RuntimeError("Fallback B-Spline fitting failed.")

    # 3. 施加变换并提取权重与控制点
    bspline.Transform(edge_ds.Location().Transformation())
    is_rational = bspline.IsRational()
    assert bspline.NbPoles() == 4, "Final B-Spline curve must have exactly 4 poles."
    poles = np.zeros((4, 4), dtype=np.float32)
    for i in range(1, 5):
        p = bspline.Pole(i)
        w = bspline.Weight(i) if is_rational else 1.0
        poles[i - 1] = [p.X(), p.Y(), p.Z(), w]

    return poles


def extract_bicubic_features_dir(shape: TopoDS_Shape):
    edge_map = TopTools_IndexedMapOfShape()

    # 核心数据：控制点 (Control Points / Poles)
    face_controls = []  # Shape: [N_faces, 4, 4, 4]
    edge_controls = []  # Shape: [N_edges, 4, 4]

    # 拓扑结构：索引与偏移
    outer_edge_indices = []
    face_outer_offsets = [0]

    inner_edge_indices = []
    inner_loop_offsets = [0]
    face_inner_offsets = [0]

    # 临时映射：Edge ID -> [Face ID, ...]
    edge_to_face_map = {}

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0

    while face_exp.More():
        face = topods.Face(face_exp.Current())

        face_controls.append(extract_or_fit_bicubic_patch(face))

        outer_wire = breptools.OuterWire(face)
        wire_exp = TopExp_Explorer(face, TopAbs_WIRE)

        while wire_exp.More():
            wire = topods.Wire(wire_exp.Current())
            is_outer = wire.IsSame(outer_wire)

            loop_edge_ids = []

            edge_exp = BRepTools_WireExplorer(wire)
            while edge_exp.More():
                edge = topods.Edge(edge_exp.Current())

                e_idx = edge_map.FindIndex(edge)
                if e_idx == 0:
                    e_idx = edge_map.Add(edge)
                    edge_controls.append(extract_or_fit_cubic_curve(edge))
                    edge_to_face_map[e_idx - 1] = []

                global_id = e_idx - 1
                loop_edge_ids.append(global_id)

                # 记录邻接
                adj_faces = edge_to_face_map[global_id]
                if not adj_faces or adj_faces[-1] != face_idx:
                    adj_faces.append(face_idx)

                edge_exp.Next()

            # 3. 更新 Wire 拓扑
            if is_outer:
                outer_edge_indices.extend(loop_edge_ids)
                face_outer_offsets.append(len(outer_edge_indices))
            else:
                if loop_edge_ids:
                    inner_edge_indices.extend(loop_edge_ids)
                    inner_loop_offsets.append(len(inner_edge_indices))

            wire_exp.Next()

        face_inner_offsets.append(len(inner_loop_offsets) - 1)
        face_idx += 1
        face_exp.Next()

    try:
        face_controls = np.array(face_controls, dtype=np.float32)
        edge_controls = np.array(edge_controls, dtype=np.float32)

        all_pts_4d = face_controls.reshape(-1, 4)
        if edge_controls.size > 0:
            all_pts_4d = np.concatenate([all_pts_4d, edge_controls.reshape(-1, 4)])

        if all_pts_4d.size > 0:
            xyz_physical = all_pts_4d[:, :3]
            vmin, vmax = xyz_physical.min(axis=0), xyz_physical.max(axis=0)
            center = (vmin + vmax) / 2.0
            scale = 2.0 / (np.max(vmax - vmin) + 1e-8)
            face_controls[..., :3] = (face_controls[..., :3] - center) * scale
            if edge_controls.size > 0:
                edge_controls[..., :3] = (edge_controls[..., :3] - center) * scale
    except Exception as e:
        raise ValueError(f"Normalization failed: {e}")
    return {
        "face_controls": face_controls,  # [F, 4, 4, 4]
        "edge_controls": edge_controls,  # [E, 4, 4]
        "outer_edge_indices": np.array(outer_edge_indices, dtype=np.int32),
        "face_outer_offsets": np.array(face_outer_offsets, dtype=np.int32),
        "inner_edge_indices": np.array(inner_edge_indices, dtype=np.int32),
        "inner_loop_offsets": np.array(inner_loop_offsets, dtype=np.int32),
        "face_inner_offsets": np.array(face_inner_offsets, dtype=np.int32),
    }
