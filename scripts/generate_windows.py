#!/usr/bin/env python3
"""在既有墙体+门 DXF 上，通过建筑外轮廓剩余缺口生成窗洞及窗框。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
DOOR_LAYER = "DOORS"
ROOM_NAME_LAYER = "ROOM_NAMES"


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
    evidence_line_count: int = 4
    source: str = "FOUR_LINE_DETECTED"
    outline_gap_id: str | None = None


@dataclass(frozen=True)
class CadWindow:
    orientation: str
    start: float
    end: float
    face1: float
    face2: float
    confidence: float
    source: str = "DETECTED"
    evidence_line_count: int = 4

    @property
    def width(self) -> float:
        return self.end - self.start

    @property
    def wall_width(self) -> float:
        return self.face2 - self.face1


def find_corner_window_pairs(
    windows: Sequence[CadWindow], tolerance: float
) -> list[dict]:
    """Pair perpendicular windows one-to-one when both terminate at one wall corner."""
    candidates: list[tuple[float, int, int, float, float, float, float]] = []
    for horizontal_index, horizontal in enumerate(windows):
        if horizontal.orientation != "horizontal":
            continue
        horizontal_axis = (horizontal.face1 + horizontal.face2) / 2.0
        for vertical_index, vertical in enumerate(windows):
            if vertical.orientation != "vertical":
                continue
            vertical_axis = (vertical.face1 + vertical.face2) / 2.0
            horizontal_corner = min(
                (horizontal.start, horizontal.end), key=lambda value: abs(value - vertical_axis)
            )
            vertical_corner = min(
                (vertical.start, vertical.end), key=lambda value: abs(value - horizontal_axis)
            )
            x_error = abs(horizontal_corner - vertical_axis)
            y_error = abs(vertical_corner - horizontal_axis)
            if x_error <= tolerance and y_error <= tolerance:
                horizontal_far = (
                    horizontal.end if horizontal_corner == horizontal.start else horizontal.start
                )
                vertical_far = vertical.end if vertical_corner == vertical.start else vertical.start
                candidates.append((
                    x_error + y_error,
                    horizontal_index,
                    vertical_index,
                    horizontal_corner,
                    vertical_corner,
                    horizontal_far,
                    vertical_far,
                ))

    groups: list[dict] = []
    used: set[int] = set()
    for error, horizontal_index, vertical_index, horizontal_corner, vertical_corner, horizontal_far, vertical_far in sorted(candidates):
        if horizontal_index in used or vertical_index in used:
            continue
        used.update((horizontal_index, vertical_index))
        horizontal = windows[horizontal_index]
        vertical = windows[vertical_index]
        groups.append({
            "type": "corner_window",
            "horizontal_index": horizontal_index,
            "vertical_index": vertical_index,
            "horizontal": asdict(horizontal),
            "vertical": asdict(vertical),
            "corner": [
                round((vertical.face1 + vertical.face2) / 2.0, 3),
                round((horizontal.face1 + horizontal.face2) / 2.0, 3),
            ],
            "horizontal_corner": round(horizontal_corner, 3),
            "vertical_corner": round(vertical_corner, 3),
            "horizontal_far": round(horizontal_far, 3),
            "vertical_far": round(vertical_far, 3),
            "connection_error_mm": round(error, 3),
            "evidence_line_count": 4,
            "source_evidence_line_counts": [
                horizontal.evidence_line_count,
                vertical.evidence_line_count,
            ],
            "status": "MERGED_CONTINUOUS_L_SHAPED_FRAME",
        })
    return groups


def identify_corner_window_groups(
    windows: Sequence[CadWindow], tolerance: float
) -> list[dict]:
    return find_corner_window_pairs(windows, tolerance)


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


def resolve_multimodal_result(
    explicit_path: Path | None, preprocessing_result: Path | None
) -> Path | None:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"多模态结果不存在：{explicit_path}")
        return explicit_path
    if preprocessing_result is None:
        return None
    candidate = preprocessing_result.parent / "multimodal.json"
    return candidate if candidate.exists() else None


def load_outline_context(
    multimodal_result: Path | None, image_shape: tuple[int, int]
) -> tuple[dict | None, np.ndarray, np.ndarray]:
    outline_mask = np.zeros(image_shape, dtype=np.uint8)
    footprint_mask = np.zeros(image_shape, dtype=np.uint8)
    if multimodal_result is None:
        return None, outline_mask, footprint_mask
    data = json.loads(multimodal_result.read_text(encoding="utf-8"))
    outline = data.get("building_outline")
    if not isinstance(outline, dict) or not outline.get("closed"):
        return None, outline_mask, footprint_mask
    points = np.asarray(outline.get("polygon_px", []), dtype=np.int32)
    if len(points) < 3:
        return None, outline_mask, footprint_mask
    polygon = points.reshape((-1, 1, 2))
    cv2.polylines(outline_mask, [polygon], True, 255, 2)
    cv2.fillPoly(footprint_mask, [polygon], 255)
    return outline, outline_mask, footprint_mask


def load_frozen_wall_mask(
    preprocessing_result: Path | None, image_shape: tuple[int, int]
) -> tuple[np.ndarray | None, Path | None, str | None]:
    if preprocessing_result is None:
        return None, None, None
    path = preprocessing_result.parent / "masks" / "walls.png"
    if not path.exists():
        return None, path, None
    mask = read_gray(path)
    if mask.shape != image_shape:
        raise ValueError(
            f"墙体 mask 尺寸 {mask.shape[::-1]} 与原图尺寸 {image_shape[::-1]} 不一致"
        )
    return np.uint8(mask > 0) * 255, path, file_sha256(path)


def load_preprocessing_door_boxes(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes: list[dict] = []
    for item in data.get("doors", []):
        box = item.get("bbox") or {}
        width = float(box.get("width", 0))
        height = float(box.get("height", 0))
        if width > 0 and height > 0:
            boxes.append({
                "x": float(box.get("x", 0)),
                "y": float(box.get("y", 0)),
                "width": width,
                "height": height,
                "source": "preprocessing-door-bbox",
            })
    return boxes


def _entity_xy_points(entity) -> list[tuple[float, float]]:
    kind = entity.dxftype()
    if kind == "LINE":
        return [
            (float(entity.dxf.start.x), float(entity.dxf.start.y)),
            (float(entity.dxf.end.x), float(entity.dxf.end.y)),
        ]
    if kind == "LWPOLYLINE":
        return [(float(x), float(y)) for x, y in entity.get_points("xy")]
    if kind == "ARC":
        start = float(entity.dxf.start_angle)
        end = float(entity.dxf.end_angle)
        if end < start:
            end += 360.0
        steps = max(8, int(math.ceil((end - start) / 5.0)))
        center = entity.dxf.center
        radius = float(entity.dxf.radius)
        return [
            (
                float(center.x) + radius * math.cos(math.radians(start + (end - start) * i / steps)),
                float(center.y) + radius * math.sin(math.radians(start + (end - start) * i / steps)),
            )
            for i in range(steps + 1)
        ]
    if kind == "ELLIPSE":
        center = entity.dxf.center
        major = entity.dxf.major_axis
        ratio = float(entity.dxf.ratio)
        minor = (-float(major.y) * ratio, float(major.x) * ratio)
        start = float(entity.dxf.start_param)
        end = float(entity.dxf.end_param)
        if end < start:
            end += math.tau
        return [
            (
                float(center.x) + float(major.x) * math.cos(t) + minor[0] * math.sin(t),
                float(center.y) + float(major.y) * math.cos(t) + minor[1] * math.sin(t),
            )
            for t in np.linspace(start, end, 25)
        ]
    return []


def load_dxf_door_boxes(
    msp,
    plan_bbox: tuple[int, int, int, int],
    overall_width_mm: float,
    overall_height_mm: float,
) -> list[dict]:
    x1, y1, x2, y2 = plan_bbox

    def pixel_x(value: float) -> float:
        return x1 + value * (x2 - x1) / overall_width_mm

    def pixel_y(value: float) -> float:
        return y2 - value * (y2 - y1) / overall_height_mm

    boxes: list[dict] = []
    for entity in msp:
        if entity.dxf.layer != DOOR_LAYER:
            continue
        points = _entity_xy_points(entity)
        if not points:
            continue
        xs = [pixel_x(point[0]) for point in points]
        ys = [pixel_y(point[1]) for point in points]
        boxes.append({
            "x": min(xs),
            "y": min(ys),
            "width": max(max(xs) - min(xs), 1.0),
            "height": max(max(ys) - min(ys), 1.0),
            "source": f"dxf-{entity.dxftype().lower()}",
            "handle": entity.dxf.handle,
        })
    return boxes


def _point_in_door_box(x: float, y: float, boxes: Sequence[dict], padding: float = 3.0) -> bool:
    for box in boxes:
        x1 = float(box["x"]) - padding
        y1 = float(box["y"]) - padding
        x2 = float(box["x"]) + float(box["width"]) + padding
        y2 = float(box["y"]) + float(box["height"]) + padding
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def _continuous_false_runs(values: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist() + [True]):
        if not value and start is None:
            start = index
        elif value and start is not None:
            if index - start >= minimum:
                runs.append((start, index - 1))
            start = None
    return runs


def _inward_sign(
    footprint_mask: np.ndarray, orientation: str, axis: int, middle: int, probe: int
) -> int:
    height, width = footprint_mask.shape
    if orientation == "horizontal":
        negative = footprint_mask[max(0, axis - probe), min(width - 1, middle)] > 0
        positive = footprint_mask[min(height - 1, axis + probe), min(width - 1, middle)] > 0
    else:
        negative = footprint_mask[min(height - 1, middle), max(0, axis - probe)] > 0
        positive = footprint_mask[min(height - 1, middle), min(width - 1, axis + probe)] > 0
    if positive and not negative:
        return 1
    if negative and not positive:
        return -1
    return 1


def _gap_face_axes(
    gray: np.ndarray,
    footprint_mask: np.ndarray,
    orientation: str,
    start: int,
    end: int,
    outline_axis: int,
    wall_width_px: float,
) -> tuple[float, float, float, int]:
    height, width = gray.shape
    middle = (start + end) // 2
    sign = _inward_sign(
        footprint_mask, orientation, outline_axis, middle, max(4, int(round(wall_width_px * 0.65)))
    )
    distance = max(5, int(round(wall_width_px * 1.55)))
    low = outline_axis - 3 if sign > 0 else outline_axis - distance
    high = outline_axis + distance if sign > 0 else outline_axis + 3
    ink = gray < 210
    trim = max(1, int((end - start + 1) * 0.08))
    along_start = start + trim
    along_end = max(along_start + 1, end - trim + 1)
    if orientation == "horizontal":
        low, high = max(0, low), min(height - 1, high)
        scores = np.count_nonzero(ink[low : high + 1, along_start:along_end], axis=1)
    else:
        low, high = max(0, low), min(width - 1, high)
        scores = np.count_nonzero(ink[along_start:along_end, low : high + 1], axis=0)
    threshold = max(3, int((along_end - along_start) * 0.28))
    supported = np.flatnonzero(scores >= threshold) + low
    clusters: list[list[int]] = []
    for value in supported.tolist():
        if not clusters or value > clusters[-1][-1] + 2:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    axes = [float(np.median(group)) for group in clusters]
    if len(axes) >= 2:
        face1, face2 = min(axes), max(axes)
        evidence_count = min(4, len(axes))
    else:
        face1 = float(outline_axis)
        face2 = float(outline_axis + sign * wall_width_px)
        face1, face2 = sorted((face1, face2))
        evidence_count = len(axes)
    return face1, (face1 + face2) / 2.0, face2, evidence_count


def detect_outline_gap_windows(
    gray: np.ndarray,
    wall_mask: np.ndarray,
    footprint_mask: np.ndarray,
    outline: dict,
    door_boxes: Sequence[dict],
    plan_bbox: tuple[int, int, int, int],
    overall_width_mm: float,
    overall_height_mm: float,
    wall_widths_mm: Sequence[float],
) -> tuple[list[PixelWindow], np.ndarray, list[dict]]:
    points = [tuple(map(int, point)) for point in outline.get("polygon_px", [])]
    gap_mask = np.zeros_like(gray)
    windows: list[PixelWindow] = []
    diagnostics: list[dict] = []
    x1, y1, x2, y2 = plan_bbox
    horizontal_wall_px = max(wall_widths_mm) * (y2 - y1) / overall_height_mm
    vertical_wall_px = max(wall_widths_mm) * (x2 - x1) / overall_width_mm
    height, width = gray.shape
    for segment_index, (first, second) in enumerate(zip(points, points[1:] + points[:1]), start=1):
        dx, dy = second[0] - first[0], second[1] - first[1]
        if abs(dx) >= max(4, abs(dy) * 4):
            orientation = "horizontal"
            axis = int(round((first[1] + second[1]) / 2.0))
            seg_start, seg_end = sorted((first[0], second[0]))
            wall_width_px = horizontal_wall_px
            minimum = max(10, int(round(300.0 * (x2 - x1) / overall_width_mm)))
        elif abs(dy) >= max(4, abs(dx) * 4):
            orientation = "vertical"
            axis = int(round((first[0] + second[0]) / 2.0))
            seg_start, seg_end = sorted((first[1], second[1]))
            wall_width_px = vertical_wall_px
            minimum = max(10, int(round(300.0 * (y2 - y1) / overall_height_mm)))
        else:
            diagnostics.append({"segment": segment_index, "status": "SKIPPED_NON_ORTHOGONAL"})
            continue
        seg_start = max(0, seg_start)
        seg_end = min((width - 1) if orientation == "horizontal" else (height - 1), seg_end)
        if seg_end - seg_start + 1 < minimum:
            continue
        band = max(4, int(round(wall_width_px * 1.25)))
        covered: list[bool] = []
        for along in range(seg_start, seg_end + 1):
            if orientation == "horizontal":
                yy1, yy2 = max(0, axis - band), min(height, axis + band + 1)
                xx1, xx2 = max(0, along - 1), min(width, along + 2)
                wall_covered = bool(np.any(wall_mask[yy1:yy2, xx1:xx2]))
                door_covered = _point_in_door_box(along, axis, door_boxes)
            else:
                xx1, xx2 = max(0, axis - band), min(width, axis + band + 1)
                yy1, yy2 = max(0, along - 1), min(height, along + 2)
                wall_covered = bool(np.any(wall_mask[yy1:yy2, xx1:xx2]))
                door_covered = _point_in_door_box(axis, along, door_boxes)
            covered.append(wall_covered or door_covered)
        coverage = np.asarray(covered, dtype=np.uint8)
        coverage = cv2.morphologyEx(coverage.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8)).reshape(-1).astype(bool)
        runs = _continuous_false_runs(coverage, minimum)
        diagnostics.append({
            "segment": segment_index,
            "orientation": orientation,
            "axis": axis,
            "start": seg_start,
            "end": seg_end,
            "gap_runs": len(runs),
        })
        for run_index, (relative_start, relative_end) in enumerate(runs, start=1):
            start = seg_start + relative_start
            end = seg_start + relative_end
            face1, center, face2, evidence_count = _gap_face_axes(
                gray, footprint_mask, orientation, start, end, axis, wall_width_px
            )
            confidence = 0.94 if evidence_count >= 4 else 0.87 if evidence_count >= 2 else 0.8
            gap_id = f"OUTLINE_S{segment_index:02d}_G{run_index:02d}"
            item = PixelWindow(
                orientation=orientation,
                start=float(start),
                end=float(end),
                face1=face1,
                center=center,
                face2=face2,
                confidence=confidence,
                evidence_line_count=evidence_count,
                source="OUTLINE_GAP_CONFIRMED" if evidence_count >= 2 else "OUTLINE_GAP_INFERRED",
                outline_gap_id=gap_id,
            )
            windows.append(item)
            if orientation == "horizontal":
                cv2.rectangle(gap_mask, (start, int(round(face1))), (end, int(round(face2))), 255, -1)
            else:
                cv2.rectangle(gap_mask, (int(round(face1)), start), (int(round(face2)), end), 255, -1)
    return windows, gap_mask, diagnostics


def merge_outline_and_line_evidence(
    outline_windows: Sequence[PixelWindow], line_windows: Sequence[PixelWindow]
) -> tuple[list[PixelWindow], list[dict]]:
    merged: list[PixelWindow] = []
    unmatched = list(line_windows)
    for outline in outline_windows:
        match = None
        for line in unmatched:
            overlap = max(0.0, min(outline.end, line.end) - max(outline.start, line.start))
            shorter = min(outline.end - outline.start, line.end - line.start)
            if (
                outline.orientation == line.orientation
                and shorter > 0
                and overlap >= shorter * 0.45
                and abs(outline.center - line.center) <= max(10.0, abs(outline.face2 - outline.face1) * 1.5)
            ):
                match = line
                break
        if match is None:
            merged.append(outline)
            continue
        unmatched.remove(match)
        merged.append(PixelWindow(
            orientation=outline.orientation,
            start=outline.start,
            end=outline.end,
            face1=match.face1,
            center=match.center,
            face2=match.face2,
            confidence=max(0.96, outline.confidence, match.confidence),
            evidence_line_count=4,
            source="OUTLINE_GAP+FOUR_LINE_CONFIRMED",
            outline_gap_id=outline.outline_gap_id,
        ))
    rejected = [
        {"window": asdict(item), "reason": "四线候选不位于建筑外轮廓的墙体/门剩余缺口上"}
        for item in unmatched
    ]
    return merged, rejected


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
            source=item.source,
            evidence_line_count=item.evidence_line_count,
        )
    face_values = sorted((cad_x(item.face1), cad_x(item.face2)))
    return CadWindow(
        item.orientation,
        cad_y(item.end),
        cad_y(item.start),
        face_values[0],
        face_values[1],
        item.confidence,
        source=item.source,
        evidence_line_count=item.evidence_line_count,
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

    # Disabled in image-baseline mode. Converting every wall polyline changes
    # the complete wall representation and can silently replace the measured
    # image baseline with a reconstructed line model. Openings must be handled
    # locally; unmatched candidates are drawn as image-only window geometry.
    return 0

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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            source=item.source,
            evidence_line_count=item.evidence_line_count,
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

    # Compatibility entry point only. Image-faithful mode never modifies or
    # rejects walls based on closure, even if older callers still invoke it.
    return {
        "wall_topology": "source-image-preserved",
        "added_wall_caps": 0,
        "closed_wall_polylines": sum(
            1 for entity in msp
            if entity.dxftype() == "LWPOLYLINE"
            and entity.dxf.layer in WALL_LAYERS
            and entity.closed
        ),
        "remaining_wall_lines": sum(
            1 for entity in msp
            if entity.dxftype() == "LINE" and entity.dxf.layer in WALL_LAYERS
        ),
    }

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


def window_frame_axes(window: CadWindow) -> list[float]:
    return [
        window.face1,
        window.face1 + window.wall_width / 3.0,
        window.face1 + window.wall_width * 2.0 / 3.0,
        window.face2,
    ]


def add_corner_window_geometry(msp, group: dict) -> list[list[list[float]]]:
    """Draw four continuous L-shaped polylines following window_corner_legend.jpg."""
    horizontal = CadWindow(**group["horizontal"])
    vertical = CadWindow(**group["vertical"])
    horizontal_corner = float(group["horizontal_corner"])
    vertical_corner = float(group["vertical_corner"])
    horizontal_far = float(group["horizontal_far"])
    vertical_far = float(group["vertical_far"])
    horizontal_direction = 1 if horizontal_far > horizontal_corner else -1
    vertical_direction = 1 if vertical_far > vertical_corner else -1

    horizontal_axes = window_frame_axes(horizontal)
    vertical_axes = window_frame_axes(vertical)
    if horizontal_direction != vertical_direction:
        horizontal_axes = list(reversed(horizontal_axes))

    paths: list[list[list[float]]] = []
    for vertical_axis, horizontal_axis in zip(vertical_axes, horizontal_axes):
        points = [
            (horizontal_far, horizontal_axis),
            (vertical_axis, horizontal_axis),
            (vertical_axis, vertical_far),
        ]
        msp.add_lwpolyline(points, close=False, dxfattribs={"layer": WINDOW_LAYER})
        paths.append([[round(x, 3), round(y, 3)] for x, y in points])
    return paths


def add_merged_window_geometry(
    msp, windows: Sequence[CadWindow], tolerance: float
) -> tuple[list[dict], int]:
    groups = find_corner_window_pairs(windows, tolerance)
    grouped_indices: set[int] = set()
    for group_index, group in enumerate(groups, start=1):
        grouped_indices.update((group["horizontal_index"], group["vertical_index"]))
        group["id"] = f"CW{group_index:02d}"
        group["geometry"] = add_corner_window_geometry(msp, group)
        group["entity_type"] = "LWPOLYLINE"
        group["continuous_l_polylines"] = 4
    ordinary_count = 0
    for index, window in enumerate(windows):
        if index in grouped_indices:
            continue
        add_window_geometry(msp, window)
        ordinary_count += 1
    return groups, ordinary_count


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
    outline_mask: np.ndarray | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", mask)[1].tofile(str(output.with_name(output.stem + "_window_mask.png")))
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = plan_bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 170, 0), 2)
    if outline_mask is not None and np.any(outline_mask):
        overlay[outline_mask > 0] = (0, 0, 220)
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
        if entity.dxf.layer not in WALL_LAYERS | {WINDOW_LAYER, DOOR_LAYER}:
            continue
        if entity.dxf.layer == WINDOW_LAYER:
            color = (0, 160, 185)
        elif entity.dxf.layer == DOOR_LAYER:
            color = (185, 80, 40)
        else:
            color = (0, 0, 0)
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
        elif entity.dxftype() in {"ARC", "ELLIPSE"}:
            points = [point(x, y) for x, y in _entity_xy_points(entity)]
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
        "doors": len(data.get("doors", [])),
        "windows": len(data.get("windows", [])),
        "warnings": data.get("warnings", []),
    }


def load_preprocessing_windows(path: Path | None) -> list[PixelWindow]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    result: list[PixelWindow] = []
    for item in data.get("windows", []):
        box = item.get("bbox") or {}
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        width = float(box.get("width", 0))
        height = float(box.get("height", 0))
        orientation = item.get("orientation")
        evidence_line_count = int(item.get("evidence_line_count", 0))
        if (
            width <= 0
            or height <= 0
            or orientation not in {"horizontal", "vertical"}
            or evidence_line_count != 4
        ):
            continue
        if orientation == "horizontal":
            start, end = x, x + width
            face1, face2 = y, y + height
        else:
            start, end = y, y + height
            face1, face2 = x, x + width
        result.append(PixelWindow(
            orientation=orientation,
            start=start,
            end=end,
            face1=face1,
            center=(face1 + face2) / 2.0,
            face2=face2,
            confidence=float(item.get("confidence", 0.5)),
            evidence_line_count=4,
        ))
    return result


def suppress_windows_in_door_arc_regions(
    windows: Sequence[PixelWindow], door_boxes: Sequence[dict]
) -> tuple[list[PixelWindow], list[dict]]:
    if not door_boxes:
        return list(windows), []
    kept: list[PixelWindow] = []
    suppressed: list[dict] = []
    for window in windows:
        if window.orientation == "horizontal":
            wx1, wy1, wx2, wy2 = window.start, window.face1, window.end, window.face2
        else:
            wx1, wy1, wx2, wy2 = window.face1, window.start, window.face2, window.end
        window_area = max((wx2 - wx1) * (wy2 - wy1), 1.0)
        matched = None
        for box in door_boxes:
            dx1 = float(box.get("x", 0))
            dy1 = float(box.get("y", 0))
            dx2 = dx1 + float(box.get("width", 0))
            dy2 = dy1 + float(box.get("height", 0))
            overlap = max(0.0, min(wx2, dx2) - max(wx1, dx1)) * max(
                0.0, min(wy2, dy2) - max(wy1, dy1)
            )
            center_inside = dx1 <= (wx1 + wx2) / 2.0 <= dx2 and dy1 <= (wy1 + wy2) / 2.0 <= dy2
            if center_inside or overlap / window_area >= 0.15:
                matched = box
                break
        if matched is None:
            kept.append(window)
        else:
            suppressed.append({
                "window": asdict(window),
                "door_bbox": matched,
                "reason": "已完成门图元/门候选优先，不允许同位置生成窗",
            })
    return kept, suppressed


def preserved_layer_handles(msp, layers: set[str]) -> dict[str, list[str]]:
    return {
        layer: sorted(
            entity.dxf.handle
            for entity in msp
            if entity.dxf.layer == layer and entity.dxf.handle is not None
        )
        for layer in sorted(layers)
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
    multimodal_result: Path | None = None,
) -> dict:
    if not wall_dxf.exists():
        raise FileNotFoundError(f"墙体 DXF 不存在：{wall_dxf}")
    gray = read_gray(source)
    wall_widths = load_wall_widths(data_path)
    plan_bbox = image_wall_bbox or detect_plan_bbox(gray)
    line_windows, line_mask = detect_windows(
        gray, plan_bbox, overall_width_mm, overall_height_mm, wall_widths
    )

    doc = ezdxf.readfile(wall_dxf)
    doc.units = ezdxf.units.MM
    if WINDOW_LAYER not in doc.layers:
        doc.layers.add(WINDOW_LAYER, color=4, linetype="CONTINUOUS")
    msp = doc.modelspace()
    preserved_before = preserved_layer_handles(msp, {DOOR_LAYER, ROOM_NAME_LAYER})

    resolved_multimodal = resolve_multimodal_result(multimodal_result, preprocessing_result)
    outline, outline_mask, footprint_mask = load_outline_context(
        resolved_multimodal, gray.shape
    )
    frozen_wall_mask, frozen_wall_mask_path, frozen_wall_mask_sha256 = load_frozen_wall_mask(
        preprocessing_result, gray.shape
    )
    door_boxes = load_preprocessing_door_boxes(preprocessing_result)
    dxf_door_boxes = load_dxf_door_boxes(
        msp, plan_bbox, overall_width_mm, overall_height_mm
    )
    door_boxes.extend(dxf_door_boxes)

    outline_gap_diagnostics: list[dict] = []
    outline_rejected_line_candidates: list[dict] = []
    outline_windows: list[PixelWindow] = []
    if outline is not None and frozen_wall_mask is not None:
        outline_windows, mask, outline_gap_diagnostics = detect_outline_gap_windows(
            gray,
            frozen_wall_mask,
            footprint_mask,
            outline,
            door_boxes,
            plan_bbox,
            overall_width_mm,
            overall_height_mm,
            wall_widths,
        )
        pixel_windows, outline_rejected_line_candidates = merge_outline_and_line_evidence(
            outline_windows, line_windows
        )
    else:
        pixel_windows = list(line_windows)
        mask = line_mask
    preprocessing_fallback = False
    if not pixel_windows:
        pixel_windows = load_preprocessing_windows(preprocessing_result)
        preprocessing_fallback = bool(pixel_windows)
    pixel_windows, door_arc_suppressed = suppress_windows_in_door_arc_regions(
        pixel_windows, door_boxes
    )
    stage_issues: list[str] = []
    if outline is None:
        stage_issues.append("缺少闭合 building_outline；本次仅执行旧四线窗兼容检测")
    if frozen_wall_mask is None:
        stage_issues.append("缺少冻结的 masks/walls.png；无法执行外轮廓减墙体检测")
    if not pixel_windows:
        stage_issues.append("外轮廓扣除墙体和门后未形成连续窗缺口；保留输入 DXF")

    converted_polylines = explode_wall_polylines(msp)

    accepted: list[CadWindow] = []
    records: list[dict] = []
    for index, pixel in enumerate(pixel_windows, start=1):
        raw = to_cad_window(pixel, plan_bbox, overall_width_mm, overall_height_mm)
        normalized, reason = normalize_window_to_wall(raw, msp, wall_widths)
        if normalized is None:
            # The image candidate remains authoritative even when the current
            # wall geometry cannot provide a usable pair of faces. Draw the
            # window in image-mapped CAD coordinates and keep the mismatch in
            # the report instead of dropping the detected window.
            accepted.append(raw)
            records.append(
                {
                    "id": f"W{index:02d}",
                    "status": "ACCEPTED_IMAGE_ONLY",
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
        stage_issues.append("窗户候选全部被墙体匹配规则拒绝；保留原墙体并继续流程")

    corner_window_groups, ordinary_window_count = add_merged_window_geometry(
        msp, accepted, tolerance=max(wall_widths) * 2.0
    )

    # Preserve the source wall geometry. Wall closure is neither rebuilt nor
    # used as a gate for window entity generation.
    wall_topology = {"wall_topology": "source-image-preserved"}
    final_wall_overlaps = sum(count_wall_overlaps(msp, item) for item in accepted)
    if final_wall_overlaps:
        stage_issues.append(f"闭合墙体重建后，窗洞范围内仍有 {final_wall_overlaps} 条墙体边界")

    preserved_after = preserved_layer_handles(msp, {DOOR_LAYER, ROOM_NAME_LAYER})
    preserved_ok = preserved_before == preserved_after
    if not preserved_ok:
        raise RuntimeError("窗阶段改变了输入中的 DOORS 或 ROOM_NAMES 实体，已停止保存")
    frozen_wall_mask_unchanged = (
        frozen_wall_mask_path is None
        or frozen_wall_mask_sha256 == file_sha256(frozen_wall_mask_path)
    )
    if not frozen_wall_mask_unchanged:
        raise RuntimeError("窗阶段改变了冻结的 masks/walls.png，已停止保存")

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output)
    auditor = doc.audit()
    report = {
        "source": str(source.resolve()),
        "wall_dxf": str(wall_dxf.resolve()),
        "wall_baseline_sha256": file_sha256(wall_dxf),
        "wall_baseline_policy": "fixed-source-image-geometry",
        "output": str(output.resolve()),
        "overall_width_mm": overall_width_mm,
        "overall_height_mm": overall_height_mm,
        "image_wall_bbox": {"x1": plan_bbox[0], "y1": plan_bbox[1], "x2": plan_bbox[2], "y2": plan_bbox[3]},
        "wall_width_candidates_mm": wall_widths,
        "detector": "building-outline-minus-walls-minus-doors-with-line-confirmation",
        "processing_order": ["building_outline", "wall", "door", "window"],
        "multimodal_result": str(resolved_multimodal.resolve()) if resolved_multimodal else None,
        "building_outline": {
            "id": outline.get("id"),
            "closed": outline.get("closed"),
            "confidence": outline.get("confidence"),
            "polygon_point_count": len(outline.get("polygon_px", [])),
        } if outline else None,
        "outline_gap_rule": "building_outline - frozen_wall_mask - completed_doors",
        "outline_gap_candidates": len(outline_windows),
        "outline_gap_diagnostics": outline_gap_diagnostics,
        "four_line_candidates": len(line_windows),
        "four_line_candidates_outside_outline_gaps": outline_rejected_line_candidates,
        "frozen_wall_mask": str(frozen_wall_mask_path.resolve()) if frozen_wall_mask_path and frozen_wall_mask_path.exists() else None,
        "frozen_wall_mask_sha256": frozen_wall_mask_sha256,
        "frozen_wall_mask_unchanged": frozen_wall_mask_unchanged,
        "preprocessing_evidence": load_preprocessing_evidence(preprocessing_result),
        "pixel_candidates": len(pixel_windows),
        "preprocessing_candidate_fallback": preprocessing_fallback,
        "door_arc_suppressed_windows": door_arc_suppressed,
        "door_arc_suppressed_window_count": len(door_arc_suppressed),
        "door_exclusion_boxes": {
            "preprocessing": len(door_boxes) - len(dxf_door_boxes),
            "dxf_entities": len(dxf_door_boxes),
        },
        "preserved_layers": {
            "before": {layer: len(handles) for layer, handles in preserved_before.items()},
            "after": {layer: len(handles) for layer, handles in preserved_after.items()},
            "handles_unchanged": preserved_ok,
        },
        "accepted_windows": len(accepted),
        "accepted_image_only_windows": sum(
            1 for item in records if item.get("status") == "ACCEPTED_IMAGE_ONLY"
        ),
        "corner_window_groups": corner_window_groups,
        "corner_window_group_count": len(corner_window_groups),
        "ordinary_window_count": ordinary_window_count,
        "generated_window_objects": len(corner_window_groups) + ordinary_window_count,
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
            "open_corner_window_polylines": sum(
                1
                for entity in msp
                if entity.dxftype() == "LWPOLYLINE"
                and entity.dxf.layer == WINDOW_LAYER
                and not entity.closed
            ),
        },
        "audit_errors": len(auditor.errors),
        "audit_fixes": len(auditor.fixes),
        "target_cad_visual_check": False,
        "stage_status": "needs_repair" if stage_issues else "ready",
        "repair_queue": stage_issues,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_detection_artifacts(
        source, gray, mask, plan_bbox, pixel_windows, output, outline_mask=outline_mask
    )
    draw_preview(doc, output.with_suffix(".png"), overall_width_mm, overall_height_mm)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从建筑外轮廓扣除已完成墙体和门，识别剩余窗缺口并生成窗框"
    )
    parser.add_argument("source", type=Path, help="包含窗户的户型图片")
    parser.add_argument("wall_dxf", type=Path, help="前一阶段生成并校验通过的墙体+门 DXF")
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
    parser.add_argument(
        "--multimodal-result",
        type=Path,
        help="multimodal_fusion.py 生成的 multimodal.json；省略时从 result.json 同目录自动读取",
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
        multimodal_result=args.multimodal_result,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
