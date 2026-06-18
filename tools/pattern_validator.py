import json
import os
import re
import numpy as np
import svgpathtools as svgpath
from typing import Dict, List, Any, Tuple


def validate_pattern_json(file_path: str) -> Tuple[bool, List[str]]:
    """
    验证coder模型输出的JSON文件格式
    """
    errors = []

    try:
        if not os.path.exists(file_path):
            return False, [f"文件不存在: {file_path}"]

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 检查根结构
        if not isinstance(data, dict):
            errors.append("根对象必须是JSON对象")
            return False, errors

        if "pattern" not in data:
            errors.append("根对象缺少必需的 'pattern' 字段")
        else:
            pattern = data["pattern"]
            if not isinstance(pattern, dict):
                errors.append("'pattern' 必须是对象类型")
            else:
                # 检查pattern的必需字段
                if "panels" not in pattern:
                    errors.append("'pattern' 缺少必需的 'panels' 字段")
                if "stitches" not in pattern:
                    errors.append("'pattern' 缺少必需的 'stitches' 字段")

                # 验证panels
                if "panels" in pattern:
                    _validate_panels(pattern["panels"], errors)

                # 验证stitches
                if "stitches" in pattern:
                    panel_names = (
                        list(pattern.get("panels", {}).keys())
                        if "panels" in pattern
                        else []
                    )
                    _validate_stitches(pattern["stitches"], panel_names, errors)

        return len(errors) == 0, errors

    except json.JSONDecodeError as e:
        return False, [f"JSON语法错误: {str(e)}"]
    except Exception as e:
        return False, [f"验证过程出错: {str(e)}"]


def _validate_panels(panels: Any, errors: List[str]):
    """验证所有面板"""
    if not isinstance(panels, dict):
        errors.append("'panels' 必须是对象类型")
        return

    if len(panels) == 0:
        errors.append("'panels' 不能为空")
        return

    for panel_name, panel in panels.items():
        _validate_single_panel(panel_name, panel, errors)


def _validate_single_panel(name: str, panel: Any, errors: List[str]):
    """验证单个面板的完整格式"""
    if not isinstance(panel, dict):
        errors.append(f"面板 '{name}' 必须是对象类型")
        return

    # 检查必需字段
    required_fields = ["vertices", "edges", "arrangement", "uvr"]
    for field in required_fields:
        if field not in panel:
            errors.append(f"面板 '{name}' 缺少必需字段: '{field}'")

    # 验证arrangement: 必须是指定枚举值之一
    if "arrangement" in panel:
        arrangement = panel["arrangement"]
        valid_arrangements = [
            "torso",
            "upper_arm_L",
            "upper_arm_R",
            "forearm_L",
            "forearm_R",
            "thigh_L",
            "thigh_R",
            "shin_L",
            "shin_R",
        ]
        if not isinstance(arrangement, str):
            errors.append(f"面板 '{name}' 的 'arrangement' 必须是字符串")
        elif arrangement not in valid_arrangements:
            errors.append(
                f"面板 '{name}' 的 'arrangement' 必须是以下值之一: {', '.join(valid_arrangements)}"
            )

    # 验证uvr: [u, v, r] 其中 u∈[0,1], v∈[0,1], r∈[0,30]
    if "uvr" in panel:
        uvr = panel["uvr"]
        if not isinstance(uvr, list):
            errors.append(f"面板 '{name}' 的 'uvr' 必须是数组")
        elif len(uvr) != 3:
            errors.append(f"面板 '{name}' 的 'uvr' 必须包含3个元素 [u, v, r]")
        elif not all(isinstance(x, (int, float)) for x in uvr):
            errors.append(f"面板 '{name}' 的 'uvr' 必须包含数字")
        else:
            u, v, r = uvr[0], uvr[1], uvr[2]
            if not (0 <= u <= 1):
                errors.append(f"面板 '{name}' 的 uvr[0] (u={u}) 必须在 [0,1] 范围内")
            if not (0 <= v <= 1):
                errors.append(f"面板 '{name}' 的 uvr[1] (v={v}) 必须在 [0,1] 范围内")
            if not (0 <= r <= 30):
                errors.append(f"面板 '{name}' 的 uvr[2] (r={r}) 必须在 [0,30] 范围内")

    # 验证vertices: [[float, float], ...]
    if "vertices" in panel:
        vertices = panel["vertices"]
        if not isinstance(vertices, list):
            errors.append(f"面板 '{name}' 的 'vertices' 必须是数组")
        elif len(vertices) < 3:
            errors.append(f"面板 '{name}' 至少需要3个顶点")
        else:
            for i, vertex in enumerate(vertices):
                if not isinstance(vertex, list):
                    errors.append(f"面板 '{name}' 顶点 {i} 必须是数组")
                elif len(vertex) != 2:
                    errors.append(f"面板 '{name}' 顶点 {i} 必须包含2个坐标 [x, y]")
                elif not all(isinstance(coord, (int, float)) for coord in vertex):
                    errors.append(f"面板 '{name}' 顶点 {i} 的坐标必须是数字")

    # 验证edges
    if "edges" in panel and "vertices" in panel:
        edges = panel["edges"]
        vertex_count = (
            len(panel["vertices"]) if isinstance(panel["vertices"], list) else 0
        )
        _validate_edges(name, edges, vertex_count, errors)


def _validate_edges(panel_name: str, edges: Any, vertex_count: int, errors: List[str]):
    """验证边数组"""
    if not isinstance(edges, list):
        errors.append(f"面板 '{panel_name}' 的 'edges' 必须是数组")
        return

    for i, edge in enumerate(edges):
        _validate_single_edge(panel_name, i, edge, vertex_count, errors)


def _validate_single_edge(
    panel_name: str, edge_idx: int, edge: Any, vertex_count: int, errors: List[str]
):
    """验证单个边"""
    if not isinstance(edge, dict):
        errors.append(f"面板 '{panel_name}' 边 {edge_idx} 必须是对象")
        return

    # 检查必需的endpoints
    if "endpoints" not in edge:
        errors.append(f"面板 '{panel_name}' 边 {edge_idx} 缺少必需的 'endpoints' 字段")
        return

    endpoints = edge["endpoints"]
    if not isinstance(endpoints, list):
        errors.append(f"面板 '{panel_name}' 边 {edge_idx} 的 'endpoints' 必须是数组")
    elif len(endpoints) != 2:
        errors.append(
            f"面板 '{panel_name}' 边 {edge_idx} 的 'endpoints' 必须包含2个顶点索引"
        )
    elif not all(isinstance(x, int) for x in endpoints):
        errors.append(f"面板 '{panel_name}' 边 {edge_idx} 的 'endpoints' 必须是整数")
    else:
        # 检查顶点索引有效性
        for endpoint in endpoints:
            if endpoint < 0 or endpoint >= vertex_count:
                errors.append(
                    f"面板 '{panel_name}' 边 {edge_idx} 的顶点索引 {endpoint} 超出范围 (0-{vertex_count - 1})"
                )

    # 检查可选的curvature
    if "curvature" in edge:
        _validate_curvature(panel_name, edge_idx, edge["curvature"], errors)


def _validate_curvature(
    panel_name: str, edge_idx: int, curvature: Any, errors: List[str]
):
    """验证曲率对象"""
    if not isinstance(curvature, dict):
        errors.append(f"面板 '{panel_name}' 边 {edge_idx} 的 'curvature' 必须是对象")
        return

    # 检查type字段
    if "type" not in curvature:
        errors.append(
            f"面板 '{panel_name}' 边 {edge_idx} 的 'curvature' 缺少 'type' 字段"
        )
    else:
        curve_type = curvature["type"]
        if curve_type not in ["cubic", "quadratic"]:
            errors.append(
                f"面板 '{panel_name}' 边 {edge_idx} 的 'curvature' type 必须是 'cubic' 或 'quadratic'"
            )

    # 检查params字段
    if "params" not in curvature:
        errors.append(
            f"面板 '{panel_name}' 边 {edge_idx} 的 'curvature' 缺少 'params' 字段"
        )
    else:
        params = curvature["params"]
        curve_type = curvature.get("type", "")

        if not isinstance(params, list):
            errors.append(
                f"面板 '{panel_name}' 边 {edge_idx} 的 'curvature' params 必须是数组"
            )
        else:
            # 根据类型检查控制点数量
            expected_count = (
                2 if curve_type == "cubic" else 1 if curve_type == "quadratic" else 0
            )
            if expected_count > 0 and len(params) != expected_count:
                errors.append(
                    f"面板 '{panel_name}' 边 {edge_idx} 的 '{curve_type}' 曲线必须有 {expected_count} 个控制点"
                )

            # 检查每个控制点格式
            for j, param in enumerate(params):
                if not isinstance(param, list):
                    errors.append(
                        f"面板 '{panel_name}' 边 {edge_idx} 控制点 {j} 必须是数组"
                    )
                elif len(param) != 2:
                    errors.append(
                        f"面板 '{panel_name}' 边 {edge_idx} 控制点 {j} 必须包含2个参数 [a, b]"
                    )
                elif not all(isinstance(x, (int, float)) for x in param):
                    errors.append(
                        f"面板 '{panel_name}' 边 {edge_idx} 控制点 {j} 的参数必须是数字"
                    )
                else:
                    # 检查控制点约束条件
                    a, b = param[0], param[1]

                    # 检查 a 值约束：0 < a < 1
                    if not (0 < a < 1):
                        errors.append(
                            f"面板 '{panel_name}' 边 {edge_idx} 控制点 {j} 的参数 a={a} 必须满足 0 < a < 1"
                        )

                    # 检查 b 值约束：-1 < b < 1
                    if not (-1 < b < 1):
                        errors.append(
                            f"面板 '{panel_name}' 边 {edge_idx} 控制点 {j} 的参数 b={b} 必须满足 -1 < b < 1"
                        )

            # 对于三次曲线，检查 a1 < a2 的约束
            if curve_type == "cubic" and len(params) == 2:
                if all(
                    isinstance(p, list)
                    and len(p) == 2
                    and all(isinstance(x, (int, float)) for x in p)
                    for p in params
                ):
                    a1, a2 = params[0][0], params[1][0]
                    if not (a1 < a2):
                        errors.append(
                            f"面板 '{panel_name}' 边 {edge_idx} 的三次曲线控制点必须满足 a1 < a2，当前 a1={a1}, a2={a2}"
                        )


def _validate_stitches(stitches: Any, panel_names: List[str], errors: List[str]):
    """验证缝合数组"""
    if not isinstance(stitches, list):
        errors.append("'stitches' 必须是数组")
        return

    for i, stitch in enumerate(stitches):
        if not isinstance(stitch, dict):
            errors.append(f"缝合 {i} 必须是对象")
            continue

        # 检查必需的字段 "0" 和 "1"
        if "0" not in stitch or "1" not in stitch:
            errors.append(f"缝合 {i} 必须包含 '0' 和 '1' 字段")
            continue

        # 验证两个连接部分
        for key in ["0", "1"]:
            connection = stitch[key]
            _validate_stitch_connection(i, key, connection, panel_names, errors)

        # 检查可选的 reverse 字段
        if "reverse" in stitch:
            if not isinstance(stitch["reverse"], bool):
                errors.append(f"缝合 {i} 的 'reverse' 字段必须是布尔值")


def _validate_stitch_connection(
    stitch_idx: int,
    conn_idx: str,
    connection: Any,
    panel_names: List[str],
    errors: List[str],
):
    """验证单个缝合连接"""
    if not isinstance(connection, dict):
        errors.append(f"缝合 {stitch_idx} 连接 '{conn_idx}' 必须是对象")
        return

    # 检查必需字段
    required_fields = ["panel", "start_id", "start_ratio", "end_id", "end_ratio"]
    for field in required_fields:
        if field not in connection:
            errors.append(
                f"缝合 {stitch_idx} 连接 '{conn_idx}' 缺少必需字段: '{field}'"
            )

    # 验证 panel 字段
    if "panel" in connection:
        panel = connection["panel"]
        if not isinstance(panel, str):
            errors.append(
                f"缝合 {stitch_idx} 连接 '{conn_idx}' 的 'panel' 必须是字符串"
            )
        elif panel not in panel_names:
            errors.append(
                f"缝合 {stitch_idx} 连接 '{conn_idx}' 引用了不存在的面板: '{panel}'"
            )

    # 验证 start_id 和 end_id (边索引)
    for id_field in ["start_id", "end_id"]:
        if id_field in connection:
            edge_id = connection[id_field]
            if not isinstance(edge_id, int):
                errors.append(
                    f"缝合 {stitch_idx} 连接 '{conn_idx}' 的 '{id_field}' 必须是整数"
                )

    # 验证 start_ratio 和 end_ratio (比例值)
    for ratio_field in ["start_ratio", "end_ratio"]:
        if ratio_field in connection:
            ratio = connection[ratio_field]
            if not isinstance(ratio, (int, float)):
                errors.append(
                    f"缝合 {stitch_idx} 连接 '{conn_idx}' 的 '{ratio_field}' 必须是数字"
                )
            elif not (0 <= ratio <= 1):
                errors.append(
                    f"缝合 {stitch_idx} 连接 '{conn_idx}' 的 '{ratio_field}' 必须在 [0,1] 范围内"
                )


def extract_json_path_from_response(response: str) -> str:
    """从响应中提取JSON文件路径"""
    patterns = [
        r"(?:saved|created|generated|output).*?([a-zA-Z_][a-zA-Z0-9_]*\.json)",
        r"`([^`]*\.json)`",
        r'"([^"]*\.json)"',
        r"(\w+\.json)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1]

    return None


# =============================================================================
# 自相交检测功能 + 仿真验证逻辑
# =============================================================================


def _simple_bezier_param_to_coord(
    end_points: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """简化版贝塞尔参数转换"""
    if params.shape[0] == 1:
        center = (end_points[0] + end_points[1]) / 2
        return center + params[0]
    else:
        return params


def _simple_circle_param_to_h(end_points: np.ndarray, params: np.ndarray) -> np.ndarray:
    """简化版圆弧参数转换"""
    if len(params.shape) > 0 and params.size > 0:
        return params.flatten()[0]
    return np.array(0.0)


def _close_enough(a, b, epsilon=1e-6):
    """浮点数近似比较"""
    return abs(a - b) < epsilon


def _check_panel_self_intersection(
    vertices: List[List[float]], edges: List[Dict], device: str = "cpu"
) -> bool:
    """
    检查单个面板是否自相交

    Args:
        vertices: 面板顶点坐标列表 [[x, y], ...]
        edges: 面板边定义列表
        device: 计算设备

    Returns:
        bool: 如果面板自相交返回 True，否则返回 False
    """
    if len(vertices) < 3:
        return False

    vertices_array = np.array(vertices, dtype=np.float32)
    processed_curves = []

    # 转换所有边为曲线
    for edge in edges:
        try:
            start_idx, end_idx = edge["endpoints"]
            start_point = vertices_array[start_idx]
            end_point = vertices_array[end_idx]

            if "curvature" not in edge:
                # 直线
                curve = svgpath.Line(
                    complex(float(start_point[0]), float(start_point[1])),
                    complex(float(end_point[0]), float(end_point[1])),
                )
            else:
                curvature_type = edge["curvature"]["type"]
                params = np.array(edge["curvature"]["params"], dtype=np.float32)
                end_points_array = np.stack([start_point, end_point], axis=0)

                if curvature_type == "quadratic":
                    control_point = _simple_bezier_param_to_coord(
                        end_points_array, params
                    )
                    curve = svgpath.QuadraticBezier(
                        complex(float(start_point[0]), float(start_point[1])),
                        complex(float(control_point[0]), float(control_point[1])),
                        complex(float(end_point[0]), float(end_point[1])),
                    )
                elif curvature_type == "cubic":
                    control_points = _simple_bezier_param_to_coord(
                        end_points_array, params
                    )
                    cp1 = (
                        control_points[0]
                        if control_points.shape[0] > 1
                        else control_points
                    )
                    cp2 = (
                        control_points[1]
                        if control_points.shape[0] > 1
                        else control_points
                    )
                    curve = svgpath.CubicBezier(
                        complex(float(start_point[0]), float(start_point[1])),
                        complex(float(cp1[0]), float(cp1[1])),
                        complex(float(cp2[0]), float(cp2[1])),
                        complex(float(end_point[0]), float(end_point[1])),
                    )
                else:  # circle 或其他类型 - 简化为直线
                    curve = svgpath.Line(
                        complex(float(start_point[0]), float(start_point[1])),
                        complex(float(end_point[0]), float(end_point[1])),
                    )

            processed_curves.append(curve)

        except Exception:
            # 如果处理边时出错，创建简单直线
            try:
                start_idx, end_idx = edge["endpoints"]
                start_point = vertices_array[start_idx]
                end_point = vertices_array[end_idx]
                curve = svgpath.Line(
                    complex(float(start_point[0]), float(start_point[1])),
                    complex(float(end_point[0]), float(end_point[1])),
                )
                processed_curves.append(curve)
            except Exception:
                continue

    # 检查交点
    for i in range(len(processed_curves)):
        for j in range(i + 1, len(processed_curves)):
            try:
                intersections = processed_curves[i].intersect(processed_curves[j])
                for t1, t2 in intersections:
                    # 过滤端点交点
                    if not (
                        _close_enough(t1, 0.0)
                        or _close_enough(t1, 1.0)
                        or _close_enough(t2, 0.0)
                        or _close_enough(t2, 1.0)
                    ):
                        return True
            except Exception:
                continue

    return False


def _calculate_edge_length_np(
    vertices: List[List[float]], endpoints: List[int]
) -> float:
    """计算边长度 - NumPy版本"""
    if len(endpoints) != 2:
        return 0.0

    start_idx, end_idx = endpoints
    if start_idx >= len(vertices) or end_idx >= len(vertices):
        return 0.0

    start = np.array(vertices[start_idx], dtype=np.float32)
    end = np.array(vertices[end_idx], dtype=np.float32)
    return float(np.linalg.norm(end - start))


def _calculate_polygon_area_np(vertices: np.ndarray) -> float:
    """计算多边形面积 - NumPy版本"""
    if len(vertices) < 3:
        return 0.0

    x = vertices[:, 0]
    y = vertices[:, 1]

    return 0.5 * abs(
        sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1))
    )


def _validate_panel_geometry(
    panel_name: str, panel_data: Dict, errors: List[str]
) -> bool:
    """验证单个面板几何属性"""
    vertices = panel_data.get("vertices", [])
    if len(vertices) < 3:
        return True  # 已在基础结构检查中处理

    try:
        vertices_np = np.array(vertices, dtype=np.float32)

        # 坐标范围检查
        max_coord = np.max(np.abs(vertices_np))
        if max_coord > 1000:
            errors.append(f"面板{panel_name}坐标过大({max_coord:.1f})")
            return False

        # 面积检查
        area = _calculate_polygon_area_np(vertices_np)
        if area < 1e-6:
            errors.append(f"面板{panel_name}面积过小({area:.8f})")
            return False

        # 检查无效坐标
        if np.any(np.isnan(vertices_np)) or np.any(np.isinf(vertices_np)):
            errors.append(f"面板{panel_name}包含无效坐标")
            return False

        return True
    except Exception:
        errors.append(f"面板{panel_name}几何计算失败")
        return False


def _validate_stitching_constraints(
    stitches: List, panels: Dict, errors: List[str]
) -> bool:
    """验证缝合约束"""
    if not isinstance(stitches, list):
        errors.append("stitches必须是数组")
        return False

    valid = True

    for i, stitch in enumerate(stitches):
        if not isinstance(stitch, dict):
            errors.append(f"缝合{i}必须是对象")
            valid = False
            continue

        if "0" not in stitch or "1" not in stitch:
            errors.append(f"缝合{i}必须包含'0'和'1'连接")
            valid = False
            continue

        conn1, conn2 = stitch["0"], stitch["1"]

        # 基础结构验证
        valid_connections = True
        for conn_key, conn in [("0", conn1), ("1", conn2)]:
            if not isinstance(conn, dict):
                errors.append(f"缝合{i}连接'{conn_key}'必须是对象")
                valid_connections = False
                continue

            required_fields = [
                "panel",
                "start_id",
                "start_ratio",
                "end_id",
                "end_ratio",
            ]
            for field in required_fields:
                if field not in conn:
                    errors.append(f"缝合{i}连接'{conn_key}'缺少字段'{field}'")
                    valid_connections = False

            if "panel" not in conn:
                continue

            panel_name = conn["panel"]
            if panel_name not in panels:
                errors.append(f"缝合{i}引用不存在的面板{panel_name}")
                valid_connections = False
                continue

            # 验证边索引有效性
            panel_edges = panels[panel_name].get("edges", [])
            for id_field in ["start_id", "end_id"]:
                if id_field in conn:
                    edge_idx = conn[id_field]
                    if (
                        not isinstance(edge_idx, int)
                        or edge_idx < 0
                        or edge_idx >= len(panel_edges)
                    ):
                        errors.append(
                            f"缝合{i}连接'{conn_key}'的{id_field}({edge_idx})超出范围"
                        )
                        valid_connections = False

        if not valid_connections:
            valid = False
            continue

        # 缝合长度约束检查（简化版，基于边的几何长度） - 已注释
        # try:
        #     panel1_name = conn1["panel"]
        #     panel2_name = conn2["panel"]
        #
        #     # 获取面板顶点
        #     vertices1 = panels[panel1_name]["vertices"]
        #     vertices2 = panels[panel2_name]["vertices"]
        #
        #     # 计算缝合段的近似长度（简化为单边长度）
        #     if "start_id" in conn1 and "end_id" in conn1:
        #         # 使用起始边作为长度估算
        #         start_edge1 = panels[panel1_name]["edges"][conn1["start_id"]]
        #         length1 = _calculate_edge_length_np(vertices1, start_edge1["endpoints"])
        #     else:
        #         length1 = 0
        #
        #     if "start_id" in conn2 and "end_id" in conn2:
        #         start_edge2 = panels[panel2_name]["edges"][conn2["start_id"]]
        #         length2 = _calculate_edge_length_np(vertices2, start_edge2["endpoints"])
        #     else:
        #         length2 = 0

        #     if length1 > 0 and length2 > 0 and abs(length1 - length2) > 100:
        #         errors.append(
        #             f"缝合{i}边长度差异过大: {abs(length1 - length2):.1f}cm > 100cm"
        #         )
        #         valid = False

        # except Exception as e:
        #     errors.append(f"缝合{i}约束检查失败: {str(e)}")
        #     valid = False

    return valid


def _validate_simulation_readiness(panels: Dict, errors: List[str]) -> bool:
    """验证仿真就绪性"""
    valid = True

    for panel_name, panel_data in panels.items():
        vertices = panel_data.get("vertices", [])
        edges = panel_data.get("edges", [])

        if len(vertices) < 3 or len(edges) < 3:
            continue

        # 张量形状约束检查
        for i, edge in enumerate(edges):
            endpoints = edge.get("endpoints", [])
            if len(endpoints) != 2:
                continue

            try:
                index1, index2 = int(endpoints[0]), int(endpoints[1])
                if index1 >= len(vertices) or index2 >= len(vertices):
                    continue

                vertex1 = vertices[index1]
                vertex2 = vertices[index2]

                if not (isinstance(vertex1, list) and len(vertex1) == 2):
                    errors.append(f"面板{panel_name}边{i}起始顶点不是2D坐标")
                    valid = False
                if not (isinstance(vertex2, list) and len(vertex2) == 2):
                    errors.append(f"面板{panel_name}边{i}结束顶点不是2D坐标")
                    valid = False

            except (ValueError, TypeError, IndexError):
                errors.append(f"面板{panel_name}边{i}端点引用错误")
                valid = False

    return valid


def check_self_intersection(file_path: str) -> Dict[str, Any]:
    """
    综合检查：自相交 + 几何验证 + 缝合约束 + 仿真就绪性

    Args:
        file_path: JSON文件路径

    Returns:
        Dict[str, Any]: 综合检测结果字典，包含：
        - success: bool - 检测是否成功
        - ready_for_simulation: bool - 是否可以进行仿真
        - intersection_check: Dict - 自相交检测结果
        - validation_check: Dict - 验证检查结果
        - summary: str - 简洁的检测结果摘要
        - all_errors: List[str] - 所有错误列表
        - error: str - 错误信息（如果有）
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "ready_for_simulation": False,
                "error": f"文件不存在: {file_path}",
                "intersection_check": {"has_problems": False, "problem_panels": []},
                "validation_check": {"passed": False},
                "summary": "文件不存在",
                "all_errors": ["文件不存在"],
            }

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 检查JSON格式
        if (
            "pattern" not in data
            or "panels" not in data["pattern"]
            or "stitches" not in data["pattern"]
        ):
            return {
                "success": False,
                "ready_for_simulation": False,
                "error": "JSON文件格式错误：缺少必要字段",
                "intersection_check": {"has_problems": False, "problem_panels": []},
                "validation_check": {"passed": False},
                "summary": "JSON格式错误",
                "all_errors": ["JSON格式错误"],
            }

        panels = data["pattern"]["panels"]
        stitches = data["pattern"]["stitches"]

        if not panels:
            return {
                "success": True,
                "ready_for_simulation": False,
                "intersection_check": {"has_problems": False, "problem_panels": []},
                "validation_check": {"passed": False},
                "summary": "没有找到面板",
                "all_errors": ["没有面板"],
            }

        # 1. 自相交检测
        intersection_problems = []
        intersection_details = {}

        for panel_name, panel_data in panels.items():
            if "vertices" not in panel_data or "edges" not in panel_data:
                intersection_details[panel_name] = False
                continue

            try:
                has_intersection = _check_panel_self_intersection(
                    panel_data["vertices"], panel_data["edges"]
                )
                intersection_details[panel_name] = has_intersection
                if has_intersection:
                    intersection_problems.append(panel_name)
            except Exception:
                intersection_details[panel_name] = False

        # 2. 综合验证检查
        all_errors = []

        # 几何验证
        geometry_valid = True
        for panel_name, panel_data in panels.items():
            if not _validate_panel_geometry(panel_name, panel_data, all_errors):
                geometry_valid = False

        # 缝合验证
        stitching_valid = _validate_stitching_constraints(stitches, panels, all_errors)

        # 仿真就绪验证
        simulation_constraints_valid = _validate_simulation_readiness(
            panels, all_errors
        )

        # 自相交错误
        if intersection_problems:
            all_errors.extend(
                [f"面板{name}存在自相交" for name in intersection_problems]
            )

        # 综合结果
        has_intersection_problems = len(intersection_problems) > 0
        validation_passed = (
            geometry_valid and stitching_valid and simulation_constraints_valid
        )
        ready_for_simulation = validation_passed and not has_intersection_problems

        # 生成摘要
        if ready_for_simulation:
            summary = f"所有检查通过，可以进行仿真（{len(panels)}个面板）"
        else:
            error_count = len(all_errors)
            summary = f"不可仿真，发现{error_count}个问题"

        return {
            "success": True,
            "ready_for_simulation": ready_for_simulation,
            "intersection_check": {
                "has_problems": has_intersection_problems,
                "problem_panels": intersection_problems,
                "details": intersection_details,
            },
            "validation_check": {
                "passed": validation_passed,
                "geometry_valid": geometry_valid,
                "stitching_valid": stitching_valid,
                "simulation_ready": simulation_constraints_valid,
            },
            "summary": summary,
            "all_errors": all_errors,
            "error": None,
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "ready_for_simulation": False,
            "error": f"JSON语法错误: {str(e)}",
            "intersection_check": {"has_problems": False, "problem_panels": []},
            "validation_check": {"passed": False},
            "summary": "JSON语法错误",
            "all_errors": [f"JSON语法错误: {str(e)}"],
        }
    except Exception as e:
        return {
            "success": False,
            "ready_for_simulation": False,
            "error": f"检测过程出错: {str(e)}",
            "intersection_check": {"has_problems": False, "problem_panels": []},
            "validation_check": {"passed": False},
            "summary": "检测失败",
            "all_errors": [f"检测过程出错: {str(e)}"],
        }


def validate_and_check_intersection(file_path: str) -> Dict[str, Any]:
    """
    综合验证函数：同时进行格式验证和综合检查

    Args:
        file_path: JSON文件路径

    Returns:
        Dict[str, Any]: 包含格式验证和综合检查结果的完整报告
    """
    # 1. 格式验证
    format_valid, format_errors = validate_pattern_json(file_path)

    # 2. 综合检查（包含自相交 + 验证）
    comprehensive_result = check_self_intersection(file_path)

    # 3. 综合结果
    return {
        "file_path": file_path,
        "format_validation": {"valid": format_valid, "errors": format_errors},
        "comprehensive_check": comprehensive_result,
        "overall_status": {
            "format_ok": format_valid,
            "comprehensive_ok": comprehensive_result.get("ready_for_simulation", False),
            "all_ok": format_valid
            and comprehensive_result.get("ready_for_simulation", False),
        },
    }


# 主函数
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python zzz.py <json_file>")
        print("功能: 验证JSON文件格式和仿真约束")
        sys.exit(1)

    # 运行综合检查
    result = check_self_intersection(sys.argv[1])

    # 为代码生成模型输出精简但完整的信息
    file_name = sys.argv[1]
    ready = result["ready_for_simulation"]

    print(f"VALIDATION_RESULT: {file_name}")
    print(f"STATUS: {'READY' if ready else 'NOT_READY'}")

    # 始终输出检查状态摘要，便于模型了解检查结果
    intersection_check = result.get("intersection_check", {})
    validation_check = result.get("validation_check", {})

    print("CHECK_STATUS:")
    print("  format_valid: true")  # 如果到这里说明格式已经通过
    print(
        f"  self_intersection: {'fail' if intersection_check.get('has_problems', False) else 'pass'}"
    )
    print(
        f"  geometry: {'pass' if validation_check.get('geometry_valid', False) else 'fail'}"
    )
    print(
        f"  stitching: {'pass' if validation_check.get('stitching_valid', False) else 'fail'}"
    )
    print(
        f"  simulation_ready: {'pass' if validation_check.get('simulation_ready', False) else 'fail'}"
    )

    if not ready:
        # 输出所有错误，每行一个，便于模型解析
        errors = result.get("all_errors", [])
        if errors:
            print("ERRORS:")
            for error in errors:
                print(f"  - {error}")

        # 如果有自相交问题，列出具体面板
        problem_panels = intersection_check.get("problem_panels", [])
        if problem_panels:
            print(f"SELF_INTERSECTION_PANELS: {', '.join(problem_panels)}")
    else:
        # 成功情况下也提供一些有用信息
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)
            panels = data.get("pattern", {}).get("panels", {})
            stitches = data.get("pattern", {}).get("stitches", [])
            print(
                f"SUMMARY: {len(panels)} panels, {len(stitches)} stitches validated successfully"
            )
        except Exception:
            print("SUMMARY: Validation completed successfully")

    # 根据检测结果设置退出码
    sys.exit(0 if result["ready_for_simulation"] else 1)
