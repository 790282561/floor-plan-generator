#!/usr/bin/env python3
"""在既有墙体 DXF 上识别并生成真实窗洞及窗框。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import ezdxf
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_WIDTH_MM = 10700.0
DEFAULT_HEIGHT_MM = 13990.0
DEFAULT_WALL_WIDTHS_MM = [240.0, 180.0, 120.0]
WALL_LAYERS = {"WALLS", "INNER_WALLS"}
WINDOW_LAYER = "WINDOWS"


@dataclass(frozen=True)
class AxisLine:
    """一条从图片中提取的水平或竖直直线。"""

    axis: float
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class PixelWindow:
    orientation: str
    start: float
    end: float
    face1: float
    center: float
    face2: float
    confidence: float


@dataclass(frozen=True)
class CadWindow:
    orientation: str
    start: float
    end: float
    face1: float
    face2: float
    confidence: float
    source: str = "DETECTED"

    @property
    def width(self) -> float:
        return self.end - self.start

    @property
    def wall_width(self) -> float:
        return self.face2 - self.face1


def read_gray(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def load_wall_widths(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("wall_width", data.get("wall_wdith"))
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}: wall_width 必须是非空数组")
    widths = sorted({float(value) for value in values}, reverse=True)
    if any(value <= 0 for value in widths):
        raise ValueError(f"{path}: wall_width 中的值必须大于 0")
    return widths


def merge_intervals(items: Sequence[tuple[float, float]], gap: float = 3.0) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(items):
        if not merged or start > merged[-1][1] + gap:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def collapse_axis_lines(lines: Iterable[AxisLine], axis_tolerance: float = 2.5) -> list[AxisLine]:
    """合并抗锯齿造成的同轴重复线，但不跨越真实洞口。"""

    groups: list[list[AxisLine]] = []
    for line in sorted(lines, key=lambda item: (item.axis, item.start, item.end)):
        target: list[AxisLine] | None = None
        for group in reversed(groups):
            group_axis = float(np.median([item.axis for item in group]))
            if line.axis - group_axis > axis_tolerance:
                break
            if abs(line.axis - group_axis) <= axis_tolerance:
                target = group
                break
        if target is None:
            target = []
            groups.append(target)
        target.append(line)

    result: list[AxisLine] = []
    for group in groups:
        axis = float(np.median([item.axis for item in group]))
        intervals = merge_intervals([(item.start, item.end) for item in group], gap=3.0)
        result.extend(AxisLine(axis, start, end) for start, end in intervals)
    return sorted(result, key=lambda item: (item.axis, item.start, item.end))


def extract_axis_lines(ink: np.ndarray, orientation: str, min_length: int = 9) -> list[AxisLine]:
    if orientation == "horizontal":
        opened = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN, np.ones((1, max(7, min_length)), np.uint8)
        )
    else:
        opened = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN, np.ones((max(7, min_length), 1), np.uint8)
        )

    count, _, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    lines: list[AxisLine] = []
    for index in range(1, count):
        x, y, width, height, _ = (int(value) for value in stats[index])
        if orientation == "horizontal":
            if width >= min_length and height <= 7:
                lines.append(AxisLine(y + (height - 1) / 2, x, x + width - 1))
        elif height >= min_length and width <= 7:
            lines.append(AxisLine(x + (width - 1) / 2, y, y + height - 1))
    return collapse_axis_lines(lines)


def cluster_axis_support(lines: Sequence[AxisLine], tolerance: float = 4.0) -> list[tuple[float, float, float, float]]:
    """返回轴坐标、覆盖长度和覆盖起止位置，用于估计建筑外框。"""

    groups: list[list[AxisLine]] = []
    for line in sorted(lines, key=lambda item: item.axis):
        if not groups:
            groups.append([line])
            continue
        axis = float(np.median([item.axis for item in groups[-1]]))
        if abs(line.axis - axis) <= tolerance:
            groups[-1].append(line)
        else:
            groups.append([line])

    result: list[tuple[float, float, float, float]] = []
    for group in groups:
        axis = float(np.median([item.axis for item in group]))
        intervals = merge_intervals([(item.start, item.end) for item in group], gap=8.0)
        support = sum(end - start for start, end in intervals)
        result.append((axis, support, min(start for start, _ in intervals), max(end for _, end in intervals)))
    return result


def detect_plan_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """利用建筑两侧长竖墙估计墙体外包框，排除左侧尺寸链。"""

    height, width = gray.shape
    ink = np.uint8(gray < 210) * 255
    vertical = extract_axis_lines(ink, "vertical", min_length=max(15, height // 80))
    supports = cluster_axis_support(vertical)
    candidates = [
        item
        for item in supports
        if width * 0.17 <= item[0] <= width * 0.98 and item[1] >= height * 0.22
    ]
    if len(candidates) < 2:
        raise RuntimeError("无法自动确定图片中的墙体外包框；请使用 --image-wall-bbox 指定")

    left = min(candidates, key=lambda item: item[0])
    right = max(candidates, key=lambda item: item[0])
    if right[0] - left[0] < width * 0.35:
        raise RuntimeError("自动识别的墙体外包框过窄；请使用 --image-wall-bbox 指定")
    x1 = int(round(left[0]))
    x2 = int(round(right[0]))
    y1 = int(round(min(left[2], right[2])))
    y2 = int(round(max(left[3], right[3])))
    return x1, y1, x2, y2


def interval_coverage(outer: AxisLine, inner: AxisLine, tolerance: float = 5.0) -> float:
    overlap = max(0.0, min(outer.end, inner.end) - max(outer.start, inner.start))
    if inner.length <= 0:
        return 0.0
    return overlap / inner.length


def endcap_score(
    ink: np.ndarray,
    orientation: str,
    start: float,
    end: float,
    face1: float,
    face2: float,
) -> float:
    scores: list[float] = []
    height, width = ink.shape
    for endpoint in (start, end):
        if orientation == "horizontal":
            x1 = max(0, int(round(endpoint)) - 3)
            x2 = min(width, int(round(endpoint)) + 4)
            y1 = max(0, int(round(face1)) - 2)
            y2 = min(height, int(round(face2)) + 3)
            region = ink[y1:y2, x1:x2]
        else:
            x1 = max(0, int(round(face1)) - 2)
            x2 = min(width, int(round(face2)) + 3)
            y1 = max(0, int(round(endpoint)) - 3)
            y2 = min(height, int(round(endpoint)) + 4)
            region = ink[y1:y2, x1:x2]
        scores.append(float(np.count_nonzero(region)) / max(region.size, 1))
    return min(scores)


def find_window_centrelines(
    ink: np.ndarray,
    orientation: str,
    plan_bbox: tuple[int, int, int, int],
    max_wall_px: float,
) -> list[PixelWindow]:
    """找两条等长窗扇内线，并验证外侧墙面线与两端封边。"""

    x1, y1, x2, y2 = plan_bbox
    plan_along = (x2 - x1) if orientation == "horizontal" else (y2 - y1)
    minimum_length = max(22.0, plan_along * 0.025)
    maximum_length = plan_along * 0.36
    lines = extract_axis_lines(ink, orientation, min_length=9)
    result: list[PixelWindow] = []

    inner_pairs: list[tuple[AxisLine, AxisLine]] = []
    for index, first_inner in enumerate(lines):
        for second_inner in lines[index + 1 :]:
            inner_gap = second_inner.axis - first_inner.axis
            if inner_gap > max_wall_px * 0.55:
                break
            if inner_gap < 3.0:
                continue
            if (
                abs(first_inner.start - second_inner.start) > 6.0
                or abs(first_inner.end - second_inner.end) > 6.0
            ):
                continue
            shorter = min(first_inner.length, second_inner.length)
            longer = max(first_inner.length, second_inner.length)
            if shorter <= 0 or shorter / longer < 0.92:
                continue
            inner_pairs.append((first_inner, second_inner))

    for first_inner, second_inner in inner_pairs:
        centre_start = (first_inner.start + second_inner.start) / 2.0
        centre_end = (first_inner.end + second_inner.end) / 2.0
        centre_axis = (first_inner.axis + second_inner.axis) / 2.0
        centre_length = centre_end - centre_start
        if centre_length < minimum_length or centre_length > maximum_length:
            continue
        if orientation == "horizontal":
            if not (x1 - 5 <= centre_start < centre_end <= x2 + 5 and y1 <= centre_axis <= y2):
                continue
        elif not (y1 - 5 <= centre_start < centre_end <= y2 + 5 and x1 <= centre_axis <= x2):
            continue

        side1 = [
            line
            for line in lines
            if 3.0 <= first_inner.axis - line.axis <= max_wall_px
            and interval_coverage(line, first_inner) >= 0.86
        ]
        side2 = [
            line
            for line in lines
            if 3.0 <= line.axis - second_inner.axis <= max_wall_px
            and interval_coverage(line, second_inner) >= 0.86
        ]
        best: tuple[float, AxisLine, AxisLine] | None = None
        for first in side1:
            for second in side2:
                distance1 = first_inner.axis - first.axis
                distance2 = second.axis - second_inner.axis
                ratio = max(distance1, distance2) / max(min(distance1, distance2), 1e-6)
                if ratio > 2.1:
                    continue
                wall_span = second.axis - first.axis
                inner_gap = second_inner.axis - first_inner.axis
                if wall_span < inner_gap * 2.2 or wall_span > max_wall_px * 1.35:
                    continue
                cap = endcap_score(
                    ink,
                    orientation,
                    centre_start,
                    centre_end,
                    first.axis,
                    second.axis,
                )
                if cap < 0.12:
                    continue
                score = abs(distance1 - distance2) + (1.0 - cap) * 4.0
                if best is None or score < best[0]:
                    best = (score, first, second)
        if best is None:
            continue
        _, first, second = best
        confidence = min(0.98, 0.78 + endcap_score(
            ink, orientation, centre_start, centre_end, first.axis, second.axis
        ) * 0.3)
        result.append(
            PixelWindow(
                orientation=orientation,
                start=centre_start,
                end=centre_end,
                face1=first.axis,
                center=centre_axis,
                face2=second.axis,
                confidence=round(confidence, 3),
            )
        )

    deduplicated: list[PixelWindow] = []
    for candidate in sorted(result, key=lambda item: (item.orientation, item.center, item.start)):
        duplicate = False
        for kept in deduplicated:
            overlap = max(0.0, min(candidate.end, kept.end) - max(candidate.start, kept.start))
            shorter = min(candidate.end - candidate.start, kept.end - kept.start)
            if (
                candidate.orientation == kept.orientation
                and abs(candidate.center - kept.center) <= max_wall_px * 0.35
                and overlap >= shorter * 0.8
            ):
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(candidate)
    return deduplicated


def detect_windows(
    gray: np.ndarray,
    plan_bbox: tuple[int, int, int, int],
    overall_width_mm: float,
    overall_height_mm: float,
    wall_widths_mm: Sequence[float],
) -> tuple[list[PixelWindow], np.ndarray]:
    ink = np.uint8(gray < 210) * 255
    x1, y1, x2, y2 = plan_bbox
    horizontal_max = max(wall_widths_mm) * (y2 - y1) / overall_height_mm + 7.0
    vertical_max = max(wall_widths_mm) * (x2 - x1) / overall_width_mm + 7.0
    windows = find_window_centrelines(ink, "horizontal", plan_bbox, horizontal_max)
    windows.extend(find_window_centrelines(ink, "vertical", plan_bbox, vertical_max))

    mask = np.zeros_like(gray)
    for item in windows:
        if item.orientation == "horizontal":
            cv2.rectangle(
                mask,
                (int(round(item.start)), int(round(item.face1))),
                (int(round(item.end)), int(round(item.face2))),
                255,
                -1,
            )
        else:
            cv2.rectangle(
                mask,
                (int(round(item.face1)), int(round(item.start))),
                (int(round(item.face2)), int(round(item.end))),
                255,
                -1,
            )
    return windows, mask


def to_cad_window(
    item: PixelWindow,
    plan_bbox: tuple[int, int, int, int],
    overall_width_mm: float,
    overall_height_mm: float,
) -> CadWindow:
    x1, y1, x2, y2 = plan_bbox

    def cad_x(value: float) -> float:
        return (value - x1) * overall_width_mm / (x2 - x1)

    def cad_y(value: float) -> float:
        return (y2 - value) * overall_height_mm / (y2 - y1)

    if item.orientation == "horizontal":
        face_values = sorted((cad_y(item.face1), cad_y(item.face2)))
        return CadWindow(
            item.orientation,
            cad_x(item.start),
            cad_x(item.end),
            face_values[0],
            face_values[1],
            item.confidence,
        )
    face_values = sorted((cad_x(item.face1), cad_x(item.face2)))
    return CadWindow(
        item.orientation,
        cad_y(item.end),
        cad_y(item.start),
        face_values[0],
        face_values[1],
        item.confidence,
    )


def entity_axis_line(entity) -> tuple[str, float, float, float] | None:
    if entity.dxftype() != "LINE" or entity.dxf.layer not in WALL_LAYERS:
        return None
    start = entity.dxf.start
    end = entity.dxf.end
    if abs(float(start.y) - float(end.y)) <= 0.5:
        return "horizontal", float(start.y + end.y) / 2.0, min(float(start.x), float(end.x)), max(float(start.x), float(end.x))
    if abs(float(start.x) - float(end.x)) <= 0.5:
        return "vertical", float(start.x + end.x) / 2.0, min(float(start.y), float(end.y)), max(float(start.y), float(end.y))
    return None


def explode_wall_polylines(msp) -> int:
    """把正交墙体多段线转成可切洞的线段，尺寸及其他图层保持不变。"""

    converted = 0
    for entity in list(msp):
        if entity.dxftype() != "LWPOLYLINE" or entity.dxf.layer not in WALL_LAYERS:
            continue
        points = [(float(x), float(y)) for x, y in entity.get_points("xy")]
        if entity.closed and points:
            points.append(points[0])
        for start, end in zip(points, points[1:]):
            if start != end:
                msp.add_line(start, end, dxfattribs={"layer": entity.dxf.layer})
        msp.delete_entity(entity)
        converted += 1
    return converted


def unique_wall_coordinates(msp, orientation: str) -> tuple[list[float], list[float]]:
    perpendicular: list[float] = []
    along: list[float] = []
    for entity in msp:
        item = entity_axis_line(entity)
        if item is None:
            continue
        item_orientation, axis, start, end = item
        if item_orientation == orientation:
            perpendicular.append(axis)
            along.extend((start, end))
        else:
            along.append(axis)
            perpendicular.extend((start, end))
    return sorted(set(perpendicular)), sorted(set(along))


def nearest(value: float, choices: Sequence[float], tolerance: float) -> float:
    if not choices:
        return value
    candidate = min(choices, key=lambda item: abs(item - value))
    return candidate if abs(candidate - value) <= tolerance else value


def normalize_window_to_wall(
    item: CadWindow,
    msp,
    wall_widths_mm: Sequence[float],
) -> tuple[CadWindow | None, str | None]:
    face_coords, along_coords = unique_wall_coordinates(msp, item.orientation)
    face_tolerance = max(wall_widths_mm) * 0.8
    face1 = nearest(item.face1, face_coords, face_tolerance)
    face2 = nearest(item.face2, face_coords, face_tolerance)
    if face1 == face2:
        return None, "无法匹配两条不同的墙面线"
    face1, face2 = sorted((face1, face2))
    actual_width = face2 - face1
    expected_width = min(wall_widths_mm, key=lambda value: abs(value - actual_width))
    if abs(actual_width - expected_width) > 90.0:
        return None, f"匹配墙厚 {actual_width:.1f} mm 不属于配置候选值"

    endpoint_tolerance = max(180.0, expected_width)
    start = nearest(item.start, along_coords, endpoint_tolerance)
    end = nearest(item.end, along_coords, endpoint_tolerance)
    if end - start < 300.0:
        return None, "窗户净宽小于 300 mm"
    return (
        CadWindow(
            orientation=item.orientation,
            start=round(start, 3),
            end=round(end, 3),
            face1=round(face1, 3),
            face2=round(face2, 3),
            confidence=item.confidence,
        ),
        None,
    )


def split_wall_faces(msp, window: CadWindow, tolerance: float = 2.0) -> tuple[int, int]:
    """切除窗洞内连续墙面线；返回改动实体数和实际重叠数。"""

    changed = 0
    overlaps = 0
    for entity in list(msp):
        item = entity_axis_line(entity)
        if item is None:
            continue
        orientation, axis, start, end = item
        if orientation != window.orientation:
            continue
        if min(abs(axis - window.face1), abs(axis - window.face2)) > tolerance:
            continue
        overlap_start = max(start, window.start)
        overlap_end = min(end, window.end)
        if overlap_end - overlap_start <= tolerance:
            continue
        overlaps += 1
        layer = entity.dxf.layer
        msp.delete_entity(entity)
        if window.start - start > tolerance:
            if orientation == "horizontal":
                msp.add_line((start, axis), (window.start, axis), dxfattribs={"layer": layer})
            else:
                msp.add_line((axis, start), (axis, window.start), dxfattribs={"layer": layer})
        if end - window.end > tolerance:
            if orientation == "horizontal":
                msp.add_line((window.end, axis), (end, axis), dxfattribs={"layer": layer})
            else:
                msp.add_line((axis, window.end), (axis, end), dxfattribs={"layer": layer})
        changed += 1
    return changed, overlaps


def remove_closed_collinear_vertices(
    points: list[tuple[float, float]], tolerance: float = 0.01
) -> list[tuple[float, float]]:
    """删除闭环中的重复点和共线中间点。"""

    cleaned: list[tuple[float, float]] = []
    for point in points:
        rounded = (round(point[0], 3), round(point[1], 3))
        if not cleaned or rounded != cleaned[-1]:
            cleaned.append(rounded)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    changed = True
    while changed and len(cleaned) >= 4:
        changed = False
        result: list[tuple[float, float]] = []
        count = len(cleaned)
        for index, point in enumerate(cleaned):
            previous = cleaned[(index - 1) % count]
            following = cleaned[(index + 1) % count]
            same_x = abs(previous[0] - point[0]) <= tolerance and abs(point[0] - following[0]) <= tolerance
            same_y = abs(previous[1] - point[1]) <= tolerance and abs(point[1] - following[1]) <= tolerance
            if same_x or same_y:
                changed = True
            else:
                result.append(point)
        cleaned = result
    return cleaned


def rebuild_closed_wall_polylines(
    msp, maximum_closure_mm: float
) -> dict:
    """补齐墙厚方向的小封边，并把所有墙体线追踪为闭合多段线。"""

    line_entities = [
        entity
        for entity in msp
        if entity.dxftype() == "LINE" and entity.dxf.layer in WALL_LAYERS
    ]
    if not line_entities:
        closed = [
            entity
            for entity in msp
            if entity.dxftype() == "LWPOLYLINE"
            and entity.dxf.layer in WALL_LAYERS
            and entity.closed
        ]
        if not closed:
            raise RuntimeError("没有可重建的墙体边界")
        return {
            "added_wall_caps": 0,
            "closed_wall_polylines": len(closed),
            "remaining_wall_lines": 0,
            "open_wall_endpoints": 0,
        }

    edges_by_layer: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = {}
    for entity in line_entities:
        start = (round(float(entity.dxf.start.x), 3), round(float(entity.dxf.start.y), 3))
        end = (round(float(entity.dxf.end.x), 3), round(float(entity.dxf.end.y), 3))
        if start != end:
            edges_by_layer.setdefault(entity.dxf.layer, []).append((start, end))

    added_caps = 0
    loops: list[tuple[str, list[tuple[float, float]]]] = []
    for layer, edges in edges_by_layer.items():
        degree: dict[tuple[float, float], int] = {}
        for start, end in edges:
            degree[start] = degree.get(start, 0) + 1
            degree[end] = degree.get(end, 0) + 1
        branch_nodes = [point for point, value in degree.items() if value > 2]
        if branch_nodes:
            raise RuntimeError(
                f"{layer} 存在 {len(branch_nodes)} 个墙线分叉端点，无法重建单一闭环"
            )

        open_nodes = [point for point, value in degree.items() if value == 1]
        while open_nodes:
            first = open_nodes[0]
            candidates: list[tuple[float, tuple[float, float]]] = []
            for second in open_nodes[1:]:
                dx = abs(first[0] - second[0])
                dy = abs(first[1] - second[1])
                if dx <= 0.01 and 0.01 < dy <= maximum_closure_mm:
                    candidates.append((dy, second))
                elif dy <= 0.01 and 0.01 < dx <= maximum_closure_mm:
                    candidates.append((dx, second))
            if not candidates:
                raise RuntimeError(
                    f"{layer} 墙体仍有无法封闭的端点 {first}；停止生成，禁止输出散线墙体"
                )
            _, second = min(candidates, key=lambda item: item[0])
            edges.append((first, second))
            degree[first] += 1
            degree[second] += 1
            added_caps += 1
            open_nodes = [point for point, value in degree.items() if value == 1]

        adjacency: dict[tuple[float, float], list[tuple[float, float]]] = {}
        unused: set[tuple[tuple[float, float], tuple[float, float]]] = set()
        for start, end in edges:
            adjacency.setdefault(start, []).append(end)
            adjacency.setdefault(end, []).append(start)
            unused.add(tuple(sorted((start, end))))
        invalid = [point for point, neighbours in adjacency.items() if len(neighbours) != 2]
        if invalid:
            raise RuntimeError(
                f"{layer} 墙体闭合重建失败：{len(invalid)} 个端点的连接度不等于 2"
            )

        while unused:
            first_edge = next(iter(unused))
            start, current = first_edge
            loop = [start]
            previous = start
            unused.remove(first_edge)
            safety = 0
            while current != start:
                loop.append(current)
                neighbours = adjacency[current]
                following = neighbours[0] if neighbours[0] != previous else neighbours[1]
                edge_key = tuple(sorted((current, following)))
                if edge_key not in unused:
                    raise RuntimeError(f"{layer} 墙体轮廓在 {current} 处提前重复")
                unused.remove(edge_key)
                previous, current = current, following
                safety += 1
                if safety > len(edges) + 1:
                    raise RuntimeError(f"{layer} 墙体轮廓追踪超过安全上限")
            loop = remove_closed_collinear_vertices(loop)
            if len(loop) < 4:
                raise RuntimeError(f"{layer} 生成了少于 4 个顶点的无效墙体闭环")
            loops.append((layer, loop))

    for entity in line_entities:
        msp.delete_entity(entity)
    for layer, points in loops:
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})

    remaining_lines = sum(
        1
        for entity in msp
        if entity.dxftype() == "LINE" and entity.dxf.layer in WALL_LAYERS
    )
    closed_polylines = sum(
        1
        for entity in msp
        if entity.dxftype() == "LWPOLYLINE"
        and entity.dxf.layer in WALL_LAYERS
        and entity.closed
    )
    if remaining_lines or closed_polylines != len(loops):
        raise RuntimeError("墙体闭合多段线验收失败")
    return {
        "added_wall_caps": added_caps,
        "closed_wall_polylines": closed_polylines,
        "remaining_wall_lines": remaining_lines,
        "open_wall_endpoints": 0,
    }


def add_window_geometry(msp, window: CadWindow) -> None:
    inner1 = window.face1 + window.wall_width / 3.0
    inner2 = window.face1 + window.wall_width * 2.0 / 3.0
    if window.orientation == "horizontal":
        corners = [
            (window.start, window.face1),
            (window.end, window.face1),
            (window.end, window.face2),
            (window.start, window.face2),
        ]
        msp.add_line(
            (window.start, inner1),
            (window.end, inner1),
            dxfattribs={"layer": WINDOW_LAYER},
        )
        msp.add_line(
            (window.start, inner2),
            (window.end, inner2),
            dxfattribs={"layer": WINDOW_LAYER},
        )
    else:
        corners = [
            (window.face1, window.start),
            (window.face2, window.start),
            (window.face2, window.end),
            (window.face1, window.end),
        ]
        msp.add_line(
            (inner1, window.start),
            (inner1, window.end),
            dxfattribs={"layer": WINDOW_LAYER},
        )
        msp.add_line(
            (inner2, window.start),
            (inner2, window.end),
            dxfattribs={"layer": WINDOW_LAYER},
        )
    msp.add_lwpolyline(corners, close=True, dxfattribs={"layer": WINDOW_LAYER})


def count_wall_overlaps(msp, window: CadWindow, tolerance: float = 2.0) -> int:
    count = 0
    for entity in msp:
        items: list[tuple[str, float, float, float]] = []
        line_item = entity_axis_line(entity)
        if line_item is not None:
            items.append(line_item)
        elif entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer in WALL_LAYERS:
            points = [(float(x), float(y)) for x, y in entity.get_points("xy")]
            if entity.closed and points:
                points.append(points[0])
            for first, second in zip(points, points[1:]):
                if abs(first[1] - second[1]) <= 0.5:
                    items.append(("horizontal", (first[1] + second[1]) / 2.0, min(first[0], second[0]), max(first[0], second[0])))
                elif abs(first[0] - second[0]) <= 0.5:
                    items.append(("vertical", (first[0] + second[0]) / 2.0, min(first[1], second[1]), max(first[1], second[1])))
        for orientation, axis, start, end in items:
            if orientation != window.orientation:
                continue
            if min(abs(axis - window.face1), abs(axis - window.face2)) > tolerance:
                continue
            if min(end, window.end) - max(start, window.start) > tolerance:
                count += 1
    return count


def write_detection_artifacts(
    source: Path,
    gray: np.ndarray,
    mask: np.ndarray,
    plan_bbox: tuple[int, int, int, int],
    windows: Sequence[PixelWindow],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", mask)[1].tofile(str(output.with_name(output.stem + "_window_mask.png")))
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = plan_bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 170, 0), 2)
    for item in windows:
        if item.orientation == "horizontal":
            first = (int(round(item.start)), int(round(item.face1)))
            second = (int(round(item.end)), int(round(item.face2)))
        else:
            first = (int(round(item.face1)), int(round(item.start)))
            second = (int(round(item.face2)), int(round(item.end)))
        cv2.rectangle(overlay, first, second, (255, 120, 0), 3)
    cv2.imencode(".png", overlay)[1].tofile(str(output.with_name(output.stem + "_detection.png")))


def draw_preview(doc, path: Path, overall_width_mm: float, overall_height_mm: float) -> None:
    width = 1200
    margin = 80
    scale = (width - margin * 2) / overall_width_mm
    height = int(overall_height_mm * scale + margin * 2)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def point(x: float, y: float) -> tuple[float, float]:
        return margin + x * scale, height - margin - y * scale

    for entity in doc.modelspace():
        if entity.dxf.layer not in WALL_LAYERS | {WINDOW_LAYER}:
            continue
        color = (0, 160, 185) if entity.dxf.layer == WINDOW_LAYER else (0, 0, 0)
        if entity.dxftype() == "LINE":
            draw.line(
                (point(float(entity.dxf.start.x), float(entity.dxf.start.y)), point(float(entity.dxf.end.x), float(entity.dxf.end.y))),
                fill=color,
                width=2,
            )
        elif entity.dxftype() == "LWPOLYLINE":
            points = [point(float(x), float(y)) for x, y in entity.get_points("xy")]
            if entity.closed and points:
                points.append(points[0])
            if len(points) >= 2:
                draw.line(points, fill=color, width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def load_preprocessing_evidence(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"预处理结果不存在：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "walls": len(data.get("walls", [])),
        "windows": len(data.get("windows", [])),
        "warnings": data.get("warnings", []),
    }


def generate(
    source: Path,
    wall_dxf: Path,
    output: Path,
    data_path: Path,
    overall_width_mm: float,
    overall_height_mm: float,
    image_wall_bbox: tuple[int, int, int, int] | None = None,
    preprocessing_result: Path | None = None,
) -> dict:
    if not wall_dxf.exists():
        raise FileNotFoundError(f"墙体 DXF 不存在：{wall_dxf}")
    gray = read_gray(source)
    wall_widths = load_wall_widths(data_path)
    plan_bbox = image_wall_bbox or detect_plan_bbox(gray)
    pixel_windows, mask = detect_windows(
        gray, plan_bbox, overall_width_mm, overall_height_mm, wall_widths
    )
    if not pixel_windows:
        raise RuntimeError("未找到满足‘两条窗扇内线 + 两侧墙面线 + 两端封边’规则的窗户")

    doc = ezdxf.readfile(wall_dxf)
    doc.units = ezdxf.units.MM
    if WINDOW_LAYER not in doc.layers:
        doc.layers.add(WINDOW_LAYER, color=4, linetype="CONTINUOUS")
    msp = doc.modelspace()
    converted_polylines = explode_wall_polylines(msp)

    accepted: list[CadWindow] = []
    records: list[dict] = []
    for index, pixel in enumerate(pixel_windows, start=1):
        raw = to_cad_window(pixel, plan_bbox, overall_width_mm, overall_height_mm)
        normalized, reason = normalize_window_to_wall(raw, msp, wall_widths)
        if normalized is None:
            records.append(
                {
                    "id": f"W{index:02d}",
                    "status": "REJECTED",
                    "reason": reason,
                    "pixel": asdict(pixel),
                    "mapped_cad": asdict(raw),
                }
            )
            continue
        if any(
            old.orientation == normalized.orientation
            and min(old.end, normalized.end) - max(old.start, normalized.start) > 1.0
            and abs((old.face1 + old.face2) - (normalized.face1 + normalized.face2)) <= 4.0
            for old in accepted
        ):
            records.append(
                {
                    "id": f"W{index:02d}",
                    "status": "REJECTED",
                    "reason": "与已接受窗户重叠",
                    "pixel": asdict(pixel),
                    "mapped_cad": asdict(normalized),
                }
            )
            continue
        changed, previous_overlaps = split_wall_faces(msp, normalized)
        remaining_overlaps = count_wall_overlaps(msp, normalized)
        status = "ACCEPTED" if remaining_overlaps == 0 else "REJECTED"
        if status == "ACCEPTED":
            add_window_geometry(msp, normalized)
        records.append(
            {
                "id": f"W{index:02d}",
                "status": status,
                "reason": None if status == "ACCEPTED" else "窗洞范围内仍有墙线重叠",
                "pixel": asdict(pixel),
                "cad": asdict(normalized),
                "opening_state": "CUT_FROM_CONTINUOUS_WALL" if changed else "EXISTING_OPENING",
                "cut_wall_entities": changed,
                "wall_overlaps_before": previous_overlaps,
                "wall_overlaps_after": remaining_overlaps,
            }
        )
        if status == "ACCEPTED":
            accepted.append(normalized)

    if not accepted:
        raise RuntimeError("窗户候选全部被墙体匹配规则拒绝")

    wall_topology = rebuild_closed_wall_polylines(
        msp, maximum_closure_mm=max(wall_widths) + 1.0
    )
    final_wall_overlaps = sum(count_wall_overlaps(msp, item) for item in accepted)
    if final_wall_overlaps:
        raise RuntimeError(
            f"闭合墙体重建后，窗洞范围内仍有 {final_wall_overlaps} 条墙体边界"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output)
    auditor = doc.audit()
    report = {
        "source": str(source.resolve()),
        "wall_dxf": str(wall_dxf.resolve()),
        "output": str(output.resolve()),
        "overall_width_mm": overall_width_mm,
        "overall_height_mm": overall_height_mm,
        "image_wall_bbox": {"x1": plan_bbox[0], "y1": plan_bbox[1], "x2": plan_bbox[2], "y2": plan_bbox[3]},
        "wall_width_candidates_mm": wall_widths,
        "detector": "two-window-inner-lines-with-two-wall-faces-and-endcaps",
        "preprocessing_evidence": load_preprocessing_evidence(preprocessing_result),
        "pixel_candidates": len(pixel_windows),
        "accepted_windows": len(accepted),
        "rejected_windows": len(records) - len(accepted),
        "converted_wall_polylines": converted_polylines,
        "windows": records,
        "topology": {
            "wall_overlaps_after": final_wall_overlaps,
            "wall_topology": "closed-lwpolylines",
            **wall_topology,
            "closed_window_frames": sum(
                1
                for entity in msp
                if entity.dxftype() == "LWPOLYLINE"
                and entity.dxf.layer == WINDOW_LAYER
                and entity.closed
            ),
        },
        "audit_errors": len(auditor.errors),
        "audit_fixes": len(auditor.fixes),
        "target_cad_visual_check": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_detection_artifacts(source, gray, mask, plan_bbox, pixel_windows, output)
    draw_preview(doc, output.with_suffix(".png"), overall_width_mm, overall_height_mm)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在已校准墙体 DXF 上识别窗户、切出真实窗洞并生成窗框"
    )
    parser.add_argument("source", type=Path, help="包含窗户的户型图片")
    parser.add_argument("wall_dxf", type=Path, help="前一阶段生成并校验通过的墙体 DXF")
    parser.add_argument("output", type=Path, help="带窗户的输出 DXF")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("data.json"),
        help="包含 wall_width 候选值的 JSON",
    )
    parser.add_argument("--overall-width-mm", type=float, default=DEFAULT_WIDTH_MM)
    parser.add_argument("--overall-height-mm", type=float, default=DEFAULT_HEIGHT_MM)
    parser.add_argument(
        "--image-wall-bbox",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="图片中的墙体外包框；省略时自动识别",
    )
    parser.add_argument(
        "--preprocessing-result",
        type=Path,
        help="processing.py 生成的 result.json，用于在报告中保留预处理证据",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = generate(
        source=args.source,
        wall_dxf=args.wall_dxf,
        output=args.output,
        data_path=args.data,
        overall_width_mm=args.overall_width_mm,
        overall_height_mm=args.overall_height_mm,
        image_wall_bbox=tuple(args.image_wall_bbox) if args.image_wall_bbox else None,
        preprocessing_result=args.preprocessing_result,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
