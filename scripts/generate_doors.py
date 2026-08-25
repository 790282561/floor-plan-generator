#!/usr/bin/env python3
"""在已生成墙体和窗户的 DXF 上识别并生成门。

门阶段采用“当前门阶段图片 - 上一阶段墙体窗户图片”的差分证据，避免把
墙线、窗线、尺寸线和文字误识别成门。平开门必须检测到四分之一圆门弧；
推拉门必须检测到细长的多线框。门洞最终会写回闭合墙体多段线。
"""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class PixelLeaf:
    center_x: float
    center_y: float
    radius: float
    quadrant_start_image_deg: int
    leaf_image_angle_deg: int
    confidence: float


@dataclass(frozen=True)
class PixelDoor:
    kind: str
    x: int
    y: int
    width: int
    height: int
    leaves: tuple[PixelLeaf, ...]
    confidence: float


@dataclass(frozen=True)
class CadDoor:
    kind: str
    orientation: str
    start: float
    end: float
    face1: float
    face2: float
    confidence: float
    leaves: tuple[PixelLeaf, ...]

    @property
    def opening_width(self) -> float:
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
    if not path.exists():
        return DEFAULT_WALL_WIDTHS_MM.copy()
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("wall_width", data.get("wall_wdith", DEFAULT_WALL_WIDTHS_MM))
    result = sorted({float(value) for value in values if float(value) > 0}, reverse=True)
    if not result:
        raise ValueError("data.json 中没有有效的 wall_width 候选值")
    return result


def merge_intervals(items: Sequence[tuple[int, int]], gap: int = 3) -> list[tuple[int, int]]:
    if not items:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(items):
        if not merged or start > merged[-1][1] + gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def detect_plan_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """识别墙体所在区域，排除外围尺寸标注。"""

    # 墙体和窗户阶段必须使用同一坐标基准。这里复用窗户脚本中已经过
    # 本项目验证的长竖墙定位器，避免把左侧尺寸链误算进建筑外包框。
    from generate_windows import detect_plan_bbox as detect_wall_bbox

    return detect_wall_bbox(gray)


def circle_quadrant_and_leaf(
    mask: np.ndarray, center_x: float, center_y: float, radius: float
) -> tuple[int, int, float]:
    """识别门弧所在象限，并判断哪条半径是门扇直线。"""

    scores: list[float] = []
    for quadrant in (0, 90, 180, 270):
        score = 0.0
        for angle in np.linspace(quadrant + 3, quadrant + 87, 43):
            radians = math.radians(angle)
            for offset in (-2.5, 0.0, 2.5):
                x = int(round(center_x + (radius + offset) * math.cos(radians)))
                y = int(round(center_y + (radius + offset) * math.sin(radians)))
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                    score += 1.0
                    break
        scores.append(score)
    quadrant = int(np.argmax(scores)) * 90

    def radial_support(angle: int) -> float:
        radians = math.radians(angle)
        hits = 0
        total = 0
        for distance in np.linspace(radius * 0.12, radius * 0.96, 45):
            x = int(round(center_x + distance * math.cos(radians)))
            y = int(round(center_y + distance * math.sin(radians)))
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                total += 1
                window = mask[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
                hits += int(np.count_nonzero(window) > 0)
        return hits / max(total, 1)

    first = quadrant
    second = (quadrant + 90) % 360
    first_score = radial_support(first)
    second_score = radial_support(second)
    leaf_angle = first if first_score >= second_score else second
    confidence = min(0.99, 0.72 + max(scores) / 180.0 + max(first_score, second_score) * 0.12)
    return quadrant, leaf_angle, round(confidence, 3)


def radial_line_support(
    mask: np.ndarray, center_x: float, center_y: float, radius: float, angle: int
) -> float:
    radians = math.radians(angle)
    hits = 0
    total = 0
    for distance in np.linspace(radius * 0.10, radius * 0.96, 52):
        x = int(round(center_x + distance * math.cos(radians)))
        y = int(round(center_y + distance * math.sin(radians)))
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            total += 1
            window = mask[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3]
            hits += int(np.count_nonzero(window) > 0)
    return hits / max(total, 1)


def fit_swing_leaf(
    difference: np.ndarray,
    support_mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> PixelLeaf | None:
    x, y, width, height = box
    padding = 15
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(difference.shape[1], x + width + padding)
    y2 = min(difference.shape[0], y + height + padding)
    roi = difference[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    minimum = max(20, int(min(width, height) * 0.48))
    maximum = max(minimum + 4, int(max(width, height) * 1.15))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(18, minimum // 2),
        param1=80,
        param2=10,
        minRadius=minimum,
        maxRadius=maximum,
    )
    if circles is None:
        return None
    candidates: list[PixelLeaf] = []
    for local_x, local_y, radius in circles[0]:
        global_x = float(local_x + x1)
        global_y = float(local_y + y1)
        if not (x - padding <= global_x <= x + width + padding):
            continue
        if not (y - padding <= global_y <= y + height + padding):
            continue
        _, _, confidence = circle_quadrant_and_leaf(
            support_mask, global_x, global_y, float(radius)
        )
        left = global_x <= x + width / 2.0
        top = global_y <= y + height / 2.0
        if left and top:
            quadrant = 0
        elif not left and top:
            quadrant = 90
        elif not left and not top:
            quadrant = 180
        else:
            quadrant = 270
        first_angle = quadrant
        second_angle = (quadrant + 90) % 360
        leaf_angle = (
            first_angle
            if radial_line_support(support_mask, global_x, global_y, float(radius), first_angle)
            >= radial_line_support(support_mask, global_x, global_y, float(radius), second_angle)
            else second_angle
        )
        candidates.append(
            PixelLeaf(
                center_x=round(global_x, 3),
                center_y=round(global_y, 3),
                radius=round(float(radius), 3),
                quadrant_start_image_deg=quadrant,
                leaf_image_angle_deg=leaf_angle,
                confidence=confidence,
            )
        )
    if not candidates:
        return None
    expected_radius = min(width, height) * 0.86
    return max(
        candidates,
        key=lambda item: item.confidence
        - abs(item.radius - expected_radius) / max(expected_radius, 1.0) * 0.24,
    )


def detect_doors(
    source_gray: np.ndarray,
    previous_gray: np.ndarray,
    plan_bbox: tuple[int, int, int, int],
) -> tuple[list[PixelDoor], np.ndarray, list[dict]]:
    if source_gray.shape != previous_gray.shape:
        raise ValueError("门阶段图片与上一阶段图片尺寸不一致，无法进行可靠差分")
    difference = cv2.absdiff(source_gray, previous_gray)
    mask = np.where(difference >= 45, 255, 0).astype(np.uint8)
    x1, y1, x2, y2 = plan_bbox
    plan_mask = np.zeros_like(mask)
    plan_mask[y1 : y2 + 1, x1 : x2 + 1] = 255
    mask = cv2.bitwise_and(mask, plan_mask)
    grouped = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(grouped)
    plan_minimum = float(min(x2 - x1 + 1, y2 - y1 + 1))

    swing_parts: list[tuple[tuple[int, int, int, int], PixelLeaf, int]] = []
    sliding: list[PixelDoor] = []
    rejected: list[dict] = []
    for index in range(1, count):
        x, y, width, height, dilated_area = [int(value) for value in stats[index]]
        raw_pixels = int(np.count_nonzero(mask[y : y + height, x : x + width]))
        if dilated_area < 50 or raw_pixels < 80:
            continue
        minor = min(width, height)
        major = max(width, height)
        record = {
            "bbox": {"x": x, "y": y, "width": width, "height": height},
            "raw_pixels": raw_pixels,
        }
        swing_shape = (
            minor >= plan_minimum * 0.048
            and major <= plan_minimum * 0.15
            and raw_pixels >= 220
        )
        if swing_shape:
            leaf = fit_swing_leaf(difference, mask, (x, y, width, height))
            if leaf is not None and leaf.confidence >= 0.70:
                swing_parts.append(((x, y, width, height), leaf, raw_pixels))
                continue
            record.update({"status": "REJECTED", "reason": "方形差异区域未检测到可靠四分之一圆门弧"})
            rejected.append(record)
            continue

        sliding_shape = (
            major >= plan_minimum * 0.09
            and minor <= plan_minimum * 0.038
            and raw_pixels >= max(420, int(major * 3.0))
        )
        if sliding_shape:
            sliding.append(
                PixelDoor(
                    kind="sliding",
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    leaves=(),
                    confidence=round(min(0.96, 0.78 + raw_pixels / max(major * 18.0, 1.0)), 3),
                )
            )
            continue
        record.update({"status": "REJECTED", "reason": "不满足平开门弧或推拉门多线框的几何规则"})
        rejected.append(record)

    # 同一高度左右相邻的两个门弧属于同一个双向平开门：它们共用同一
    # 个竖向门洞，但门扇分别位于墙体两侧，开启方向相反。
    swing_parts.sort(key=lambda item: (item[0][1], item[0][0]))
    used: set[int] = set()
    swing_doors: list[PixelDoor] = []
    for index, (box, leaf, _) in enumerate(swing_parts):
        if index in used:
            continue
        x, y, width, height = box
        partner_index = None
        for other_index in range(index + 1, len(swing_parts)):
            if other_index in used:
                continue
            other_box, _, _ = swing_parts[other_index]
            ox, oy, ow, oh = other_box
            gap = ox - (x + width)
            if (
                abs(oy - y) <= max(8, int(height * 0.22))
                and abs(oh - height) <= max(8, int(height * 0.22))
                and -8 <= gap <= plan_minimum * 0.04
            ):
                partner_index = other_index
                break
        if partner_index is None:
            swing_doors.append(
                PixelDoor(
                    kind="swing_single",
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    leaves=(leaf,),
                    confidence=leaf.confidence,
                )
            )
            used.add(index)
            continue
        other_box, other_leaf, _ = swing_parts[partner_index]
        ox, oy, ow, oh = other_box
        swing_doors.append(
            PixelDoor(
                kind="swing_double",
                x=min(x, ox),
                y=min(y, oy),
                width=max(x + width, ox + ow) - min(x, ox),
                height=max(y + height, oy + oh) - min(y, oy),
                leaves=(leaf, other_leaf),
                confidence=round(min(leaf.confidence, other_leaf.confidence), 3),
            )
        )
        used.update((index, partner_index))

    return sorted(swing_doors + sliding, key=lambda item: (item.y, item.x)), mask, rejected


def axis_clusters(values: np.ndarray, minimum_support: float) -> list[tuple[float, float]]:
    indexes = np.flatnonzero(values >= minimum_support)
    intervals = merge_intervals([(int(value), int(value)) for value in indexes], gap=2)
    return [((start + end) / 2.0, float(values[start : end + 1].max())) for start, end in intervals]


def detect_wall_context(
    previous_gray: np.ndarray,
    door: PixelDoor,
    plan_bbox: tuple[int, int, int, int],
    wall_widths_mm: Sequence[float],
    overall_width_mm: float,
    overall_height_mm: float,
    preferred_orientation: str | None = None,
) -> tuple[str, float, float]:
    """从上一阶段图片中识别门洞所属墙体方向和两条墙面像素坐标。"""

    ink = previous_gray < 205
    px1, py1, px2, py2 = plan_bbox
    probe = max(45, int(min(px2 - px1 + 1, py2 - py1 + 1) * 0.065))
    margin = 4

    left1 = max(px1, door.x - probe)
    left2 = max(left1, door.x - margin)
    right1 = min(px2 + 1, door.x + door.width + margin)
    right2 = min(px2 + 1, door.x + door.width + probe)
    y_start = max(py1, door.y - probe)
    y_end = min(py2 + 1, door.y + door.height + probe)
    horizontal_support = np.zeros(y_end - y_start, dtype=float)
    if left2 > left1:
        horizontal_support += ink[y_start:y_end, left1:left2].sum(axis=1)
    if right2 > right1:
        horizontal_support += ink[y_start:y_end, right1:right2].sum(axis=1)
    horizontal_clusters = axis_clusters(horizontal_support, max(10.0, probe * 0.28))

    top1 = max(py1, door.y - probe)
    top2 = max(top1, door.y - margin)
    bottom1 = min(py2 + 1, door.y + door.height + margin)
    bottom2 = min(py2 + 1, door.y + door.height + probe)
    x_start = max(px1, door.x - probe)
    x_end = min(px2 + 1, door.x + door.width + probe)
    vertical_support = np.zeros(x_end - x_start, dtype=float)
    if top2 > top1:
        vertical_support += ink[top1:top2, x_start:x_end].sum(axis=0)
    if bottom2 > bottom1:
        vertical_support += ink[bottom1:bottom2, x_start:x_end].sum(axis=0)
    vertical_clusters = axis_clusters(vertical_support, max(10.0, probe * 0.28))

    horizontal_expected = [value / overall_height_mm * (py2 - py1) for value in wall_widths_mm]
    vertical_expected = [value / overall_width_mm * (px2 - px1) for value in wall_widths_mm]

    def best_pair(
        clusters: Sequence[tuple[float, float]], expected: Sequence[float], offset: int
    ) -> tuple[float, float, float] | None:
        candidates: list[tuple[float, float, float]] = []
        for first_index, (first, first_score) in enumerate(clusters):
            for second, second_score in clusters[first_index + 1 :]:
                separation = second - first
                error = min(abs(separation - value) for value in expected)
                if error <= max(4.0, max(expected) * 0.38):
                    score = first_score + second_score - error * 4.0
                    candidates.append((score, first + offset, second + offset))
        return max(candidates, default=None, key=lambda item: item[0])

    horizontal_pair = best_pair(horizontal_clusters, horizontal_expected, y_start)
    vertical_pair = best_pair(vertical_clusters, vertical_expected, x_start)
    if preferred_orientation == "horizontal":
        if horizontal_pair is None:
            raise RuntimeError("门型方向为水平，但附近未找到水平墙体的两条墙面线")
        return "horizontal", horizontal_pair[1], horizontal_pair[2]
    if preferred_orientation == "vertical":
        if vertical_pair is None:
            raise RuntimeError("门型方向为竖直，但附近未找到竖直墙体的两条墙面线")
        return "vertical", vertical_pair[1], vertical_pair[2]
    if horizontal_pair is None and vertical_pair is None:
        raise RuntimeError("门候选附近无法识别两条有效墙面线")
    if vertical_pair is None or (
        horizontal_pair is not None and horizontal_pair[0] >= vertical_pair[0]
    ):
        assert horizontal_pair is not None
        return "horizontal", horizontal_pair[1], horizontal_pair[2]
    return "vertical", vertical_pair[1], vertical_pair[2]


def pixel_to_cad_x(value: float, bbox: tuple[int, int, int, int], overall_width_mm: float) -> float:
    return (value - bbox[0]) / max(bbox[2] - bbox[0], 1) * overall_width_mm


def pixel_to_cad_y(value: float, bbox: tuple[int, int, int, int], overall_height_mm: float) -> float:
    return (bbox[3] - value) / max(bbox[3] - bbox[1], 1) * overall_height_mm


def raw_cad_door(
    door: PixelDoor,
    orientation: str,
    face_pixel1: float,
    face_pixel2: float,
    bbox: tuple[int, int, int, int],
    overall_width_mm: float,
    overall_height_mm: float,
) -> CadDoor:
    if orientation == "horizontal":
        face_values = sorted(
            (
                pixel_to_cad_y(face_pixel1, bbox, overall_height_mm),
                pixel_to_cad_y(face_pixel2, bbox, overall_height_mm),
            )
        )
        if door.kind == "sliding":
            pixel_start, pixel_end = door.x, door.x + door.width
        else:
            values: list[float] = []
            for leaf in door.leaves:
                values.append(leaf.center_x)
                horizontal_angle = (
                    leaf.quadrant_start_image_deg
                    if leaf.quadrant_start_image_deg % 180 == 0
                    else (leaf.quadrant_start_image_deg + 90) % 360
                )
                values.append(leaf.center_x + leaf.radius * math.cos(math.radians(horizontal_angle)))
            pixel_start, pixel_end = min(values), max(values)
        start = pixel_to_cad_x(pixel_start, bbox, overall_width_mm)
        end = pixel_to_cad_x(pixel_end, bbox, overall_width_mm)
    else:
        face_values = sorted(
            (
                pixel_to_cad_x(face_pixel1, bbox, overall_width_mm),
                pixel_to_cad_x(face_pixel2, bbox, overall_width_mm),
            )
        )
        if door.kind == "sliding":
            pixel_start, pixel_end = door.y, door.y + door.height
        else:
            values = []
            for leaf in door.leaves:
                values.append(leaf.center_y)
                vertical_angle = (
                    leaf.quadrant_start_image_deg
                    if leaf.quadrant_start_image_deg % 180 == 90
                    else (leaf.quadrant_start_image_deg + 90) % 360
                )
                values.append(leaf.center_y + leaf.radius * math.sin(math.radians(vertical_angle)))
            cad_values = [pixel_to_cad_y(value, bbox, overall_height_mm) for value in values]
            start, end = min(cad_values), max(cad_values)
            return CadDoor(
                door.kind, orientation, start, end, face_values[0], face_values[1], door.confidence, door.leaves
            )
        cad_values = [pixel_to_cad_y(pixel_start, bbox, overall_height_mm), pixel_to_cad_y(pixel_end, bbox, overall_height_mm)]
        start, end = min(cad_values), max(cad_values)
        return CadDoor(
            door.kind, orientation, start, end, face_values[0], face_values[1], door.confidence, door.leaves
        )
    start, end = sorted((start, end))
    return CadDoor(
        door.kind, orientation, start, end, face_values[0], face_values[1], door.confidence, door.leaves
    )


def entity_axis_line(entity) -> tuple[str, float, float, float] | None:
    if entity.dxftype() != "LINE" or entity.dxf.layer not in WALL_LAYERS:
        return None
    start = entity.dxf.start
    end = entity.dxf.end
    if abs(float(start.y) - float(end.y)) <= 0.5:
        return "horizontal", (float(start.y) + float(end.y)) / 2.0, min(float(start.x), float(end.x)), max(float(start.x), float(end.x))
    if abs(float(start.x) - float(end.x)) <= 0.5:
        return "vertical", (float(start.x) + float(end.x)) / 2.0, min(float(start.y), float(end.y)), max(float(start.y), float(end.y))
    return None


def explode_wall_polylines(msp) -> int:
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


def unique_wall_coordinates(msp, orientation: str) -> list[float]:
    coordinates: list[float] = []
    for entity in msp:
        item = entity_axis_line(entity)
        if item is not None and item[0] == orientation:
            coordinates.append(item[1])
    return sorted(set(coordinates))


def nearest(value: float, choices: Sequence[float], tolerance: float) -> float:
    if not choices:
        return value
    candidate = min(choices, key=lambda item: abs(item - value))
    return candidate if abs(candidate - value) <= tolerance else value


def jamb_positions(
    msp, orientation: str, face1: float, face2: float, tolerance: float = 70.0
) -> list[float]:
    positions: list[float] = []
    for entity in msp:
        item = entity_axis_line(entity)
        if item is None or item[0] == orientation:
            continue
        _, axis, start, end = item
        if abs(start - face1) <= tolerance and abs(end - face2) <= tolerance:
            positions.append(axis)
    return sorted(set(positions))


def normalize_door_to_wall(
    item: CadDoor, msp, wall_widths_mm: Sequence[float]
) -> tuple[CadDoor | None, str | None]:
    faces = unique_wall_coordinates(msp, item.orientation)
    tolerance = max(wall_widths_mm) * 0.9
    face1 = nearest(item.face1, faces, tolerance)
    face2 = nearest(item.face2, faces, tolerance)
    face1, face2 = sorted((face1, face2))
    if abs(face2 - face1) < 1.0:
        return None, "无法匹配两条不同墙面线"
    actual = face2 - face1
    expected = min(wall_widths_mm, key=lambda value: abs(value - actual))
    if abs(actual - expected) > 90.0:
        return None, f"匹配墙厚 {actual:.1f} mm 不属于 wall_width 候选值"

    jambs = jamb_positions(msp, item.orientation, face1, face2)
    endpoint_tolerance = max(320.0, expected * 1.5)
    start = item.start
    end = item.end
    if item.kind == "swing_single":
        # 单扇门图片中的圆弧端点容易受压缩噪声影响；若墙体中已有一对
        # 门槛封边，则以封边为准恢复真实门洞，避免向错误方向切墙。
        pairs: list[tuple[float, float, float]] = []
        for first_index, first in enumerate(jambs):
            for second in jambs[first_index + 1 :]:
                width = second - first
                if 500.0 <= width <= 1800.0:
                    endpoint_error = min(abs(first - item.start), abs(first - item.end)) + min(abs(second - item.start), abs(second - item.end))
                    if min(abs(first - item.start), abs(first - item.end), abs(second - item.start), abs(second - item.end)) <= endpoint_tolerance:
                        pairs.append((endpoint_error, first, second))
        if pairs:
            _, start, end = min(pairs, key=lambda value: value[0])
        else:
            start = nearest(item.start, jambs, endpoint_tolerance)
            end = nearest(item.end, jambs, endpoint_tolerance)
    else:
        start = nearest(item.start, jambs, endpoint_tolerance)
        end = nearest(item.end, jambs, endpoint_tolerance)
    start, end = sorted((start, end))
    width = end - start
    if width < 500.0:
        return None, f"门洞净宽 {width:.1f} mm 小于 500 mm"
    if width > 2800.0:
        return None, f"门洞净宽 {width:.1f} mm 大于 2800 mm"
    return (
        CadDoor(
            kind=item.kind,
            orientation=item.orientation,
            start=round(start, 3),
            end=round(end, 3),
            face1=round(face1, 3),
            face2=round(face2, 3),
            confidence=item.confidence,
            leaves=item.leaves,
        ),
        None,
    )


def split_wall_faces(msp, door: CadDoor, tolerance: float = 2.0) -> tuple[int, int]:
    changed = 0
    overlaps = 0
    for entity in list(msp):
        item = entity_axis_line(entity)
        if item is None:
            continue
        orientation, axis, start, end = item
        if orientation != door.orientation:
            continue
        if min(abs(axis - door.face1), abs(axis - door.face2)) > tolerance:
            continue
        overlap_start = max(start, door.start)
        overlap_end = min(end, door.end)
        if overlap_end - overlap_start <= tolerance:
            continue
        overlaps += 1
        layer = entity.dxf.layer
        msp.delete_entity(entity)
        if door.start - start > tolerance:
            if orientation == "horizontal":
                msp.add_line((start, axis), (door.start, axis), dxfattribs={"layer": layer})
            else:
                msp.add_line((axis, start), (axis, door.start), dxfattribs={"layer": layer})
        if end - door.end > tolerance:
            if orientation == "horizontal":
                msp.add_line((door.end, axis), (end, axis), dxfattribs={"layer": layer})
            else:
                msp.add_line((axis, door.end), (axis, end), dxfattribs={"layer": layer})
        changed += 1
    return changed, overlaps


def remove_closed_collinear_vertices(
    points: list[tuple[float, float]], tolerance: float = 0.01
) -> list[tuple[float, float]]:
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


def rebuild_closed_wall_polylines(msp, maximum_closure_mm: float) -> dict:
    line_entities = [
        entity for entity in msp
        if entity.dxftype() == "LINE" and entity.dxf.layer in WALL_LAYERS
    ]
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
        if any(value > 2 for value in degree.values()):
            raise RuntimeError(f"{layer} 墙线存在分叉端点，无法重建闭合墙体")
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
                raise RuntimeError(f"{layer} 墙体端点 {first} 无法封闭；停止输出")
            _, second = min(candidates, key=lambda value: value[0])
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
            raise RuntimeError(f"{layer} 有 {len(invalid)} 个端点连接度不等于 2")
        while unused:
            start, current = next(iter(unused))
            first_edge = tuple(sorted((start, current)))
            unused.remove(first_edge)
            loop = [start]
            previous = start
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
                    raise RuntimeError(f"{layer} 墙体闭环追踪超过安全上限")
            loop = remove_closed_collinear_vertices(loop)
            if len(loop) < 4:
                raise RuntimeError(f"{layer} 产生无效墙体闭环")
            loops.append((layer, loop))
    for entity in line_entities:
        msp.delete_entity(entity)
    for layer, points in loops:
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})
    remaining = sum(
        1 for entity in msp
        if entity.dxftype() == "LINE" and entity.dxf.layer in WALL_LAYERS
    )
    closed = sum(
        1 for entity in msp
        if entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer in WALL_LAYERS and entity.closed
    )
    if remaining or closed != len(loops):
        raise RuntimeError("墙体闭合多段线验收失败")
    return {
        "added_wall_caps": added_caps,
        "closed_wall_polylines": closed,
        "remaining_wall_lines": remaining,
        "open_wall_endpoints": 0,
    }


def count_wall_overlaps(msp, door: CadDoor, tolerance: float = 2.0) -> int:
    count = 0
    for entity in msp:
        segments: list[tuple[str, float, float, float]] = []
        line = entity_axis_line(entity)
        if line is not None:
            segments.append(line)
        elif entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer in WALL_LAYERS:
            points = [(float(x), float(y)) for x, y in entity.get_points("xy")]
            if entity.closed and points:
                points.append(points[0])
            for first, second in zip(points, points[1:]):
                if abs(first[1] - second[1]) <= 0.5:
                    segments.append(("horizontal", (first[1] + second[1]) / 2.0, min(first[0], second[0]), max(first[0], second[0])))
                elif abs(first[0] - second[0]) <= 0.5:
                    segments.append(("vertical", (first[0] + second[0]) / 2.0, min(first[1], second[1]), max(first[1], second[1])))
        for orientation, axis, start, end in segments:
            if orientation != door.orientation:
                continue
            if min(abs(axis - door.face1), abs(axis - door.face2)) > tolerance:
                continue
            if min(end, door.end) - max(start, door.start) > tolerance:
                count += 1
    return count


def overlaps_window(msp, door: CadDoor, tolerance: float = 20.0) -> bool:
    for entity in msp:
        if entity.dxftype() != "LWPOLYLINE" or entity.dxf.layer != WINDOW_LAYER:
            continue
        points = [(float(x), float(y)) for x, y in entity.get_points("xy")]
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if door.orientation == "horizontal":
            if min(max(xs), door.end) - max(min(xs), door.start) > tolerance and min(max(ys), door.face2) - max(min(ys), door.face1) > tolerance:
                return True
        else:
            if min(max(ys), door.end) - max(min(ys), door.start) > tolerance and min(max(xs), door.face2) - max(min(xs), door.face1) > tolerance:
                return True
    return False


def cad_leaf_angles(leaf: PixelLeaf) -> tuple[float, float, float]:
    image_start = leaf.quadrant_start_image_deg
    image_end = (image_start + 90) % 360
    cad_start = (-image_end) % 360
    cad_end = (-image_start) % 360
    leaf_angle = (-leaf.leaf_image_angle_deg) % 360
    return float(cad_start), float(cad_end), float(leaf_angle)


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _segment_intersection(a, b, c, d, tolerance=1.0):
    """返回交点及是否为明显共线重叠。"""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    den = rx * sy - ry * sx
    if abs(den) <= 1e-9:
        if abs(_cross(a, b, c)) > tolerance:
            return [], False
        norm = max(math.hypot(rx, ry), 1.0)
        denom = rx * rx + ry * ry
        t0 = ((c[0] - a[0]) * rx + (c[1] - a[1]) * ry) / denom
        t1 = ((d[0] - a[0]) * rx + (d[1] - a[1]) * ry) / denom
        param_tol = tolerance / norm
        lo, hi = max(0.0, min(t0, t1)), min(1.0, max(t0, t1))
        if hi < lo - param_tol:
            return [], False
        point = (a[0] + ((lo + hi) / 2.0) * rx, a[1] + ((lo + hi) / 2.0) * ry)
        return [point], hi - lo > param_tol
    qpx, qpy = c[0] - a[0], c[1] - a[1]
    t = (qpx * sy - qpy * sx) / den
    u = (qpx * ry - qpy * rx) / den
    t_tol = tolerance / max(math.hypot(rx, ry), 1.0)
    u_tol = tolerance / max(math.hypot(sx, sy), 1.0)
    if -t_tol <= t <= 1.0 + t_tol and -u_tol <= u <= 1.0 + u_tol:
        return [(a[0] + t * rx, a[1] + t * ry)], False
    return [], False


def _entity_segments(entity):
    kind = entity.dxftype()
    if kind == "LINE":
        return [((float(entity.dxf.start.x), float(entity.dxf.start.y)), (float(entity.dxf.end.x), float(entity.dxf.end.y)))]
    if kind == "ARC":
        start, end = float(entity.dxf.start_angle), float(entity.dxf.end_angle)
        while end <= start:
            end += 360.0
        steps = max(12, int(abs(end - start) / 5.0))
        angles = np.linspace(start, end, steps + 1)
        center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
        radius = float(entity.dxf.radius)
        points = [(center[0] + radius * math.cos(math.radians(a)), center[1] + radius * math.sin(math.radians(a))) for a in angles]
        return list(zip(points, points[1:]))
    if kind == "LWPOLYLINE":
        points = [(float(x), float(y)) for x, y in entity.get_points("xy")]
        if len(points) < 2:
            return []
        if entity.closed:
            points.append(points[0])
        return list(zip(points, points[1:]))
    return []


def validate_door_geometry(msp, tolerance: float = 2.0) -> list[dict]:
    """检查 DOORS 的直线/弧线/多段线是否穿入墙体或窗线。"""
    targets = [e for e in msp if e.dxftype() in {"LINE", "LWPOLYLINE"} and e.dxf.layer in WALL_LAYERS | {WINDOW_LAYER}]
    collisions = []
    for door_entity in msp:
        if door_entity.dxf.layer != DOOR_LAYER or door_entity.dxftype() not in {"LINE", "ARC", "LWPOLYLINE"}:
            continue
        for da, db in _entity_segments(door_entity):
            for target in targets:
                target_layer = target.dxf.layer
                for ta, tb in _entity_segments(target):
                    points, collinear = _segment_intersection(da, db, ta, tb, tolerance)
                    for point in points:
                        endpoint_contact = _distance(point, da) <= tolerance * 1.5 or _distance(point, db) <= tolerance * 1.5
                        if target_layer in WALL_LAYERS and endpoint_contact and not collinear:
                            continue
                        collisions.append({"door_handle": door_entity.dxf.handle, "door_type": door_entity.dxftype(), "target_layer": target_layer, "target_type": target.dxftype(), "point": [round(point[0], 3), round(point[1], 3)], "collinear_overlap": bool(collinear)})
    return collisions


def add_swing_geometry(
    msp,
    door: CadDoor,
    pixel: PixelDoor,
    bbox: tuple[int, int, int, int],
    overall_width_mm: float,
    overall_height_mm: float,
) -> list[dict]:
    records: list[dict] = []
    for leaf in pixel.leaves:
        mapped_x = pixel_to_cad_x(leaf.center_x, bbox, overall_width_mm)
        mapped_y = pixel_to_cad_y(leaf.center_y, bbox, overall_height_mm)
        arc_start, arc_end, leaf_angle = cad_leaf_angles(leaf)
        if door.orientation == "horizontal":
            hinge_x = min((door.start, door.end), key=lambda value: abs(value - mapped_x))
            hinge_y = (
                min((door.face1, door.face2), key=lambda value: abs(value - mapped_y))
                if door.kind == "swing_double"
                else (door.face2 if math.sin(math.radians(leaf_angle)) > 0 else door.face1)
            )
        else:
            hinge_y = min((door.start, door.end), key=lambda value: abs(value - mapped_y))
            hinge_x = (
                min((door.face1, door.face2), key=lambda value: abs(value - mapped_x))
                if door.kind == "swing_double"
                else (door.face1 if math.cos(math.radians(leaf_angle)) < 0 else door.face2)
            )
        radius = door.opening_width
        leaf_end = (
            hinge_x + radius * math.cos(math.radians(leaf_angle)),
            hinge_y + radius * math.sin(math.radians(leaf_angle)),
        )
        msp.add_line((hinge_x, hinge_y), leaf_end, dxfattribs={"layer": DOOR_LAYER})
        msp.add_arc(
            center=(hinge_x, hinge_y),
            radius=radius,
            start_angle=arc_start,
            end_angle=arc_end,
            dxfattribs={"layer": DOOR_LAYER},
        )
        records.append(
            {
                "hinge": [round(hinge_x, 3), round(hinge_y, 3)],
                "radius_mm": round(radius, 3),
                "leaf_angle_deg": leaf_angle,
                "arc_start_deg": arc_start,
                "arc_end_deg": arc_end,
            }
        )
    return records


def add_sliding_geometry(msp, door: CadDoor) -> list[dict]:
    midpoint = (door.start + door.end) / 2.0
    overlap = min(120.0, door.opening_width * 0.10)
    panel_depth = max(24.0, min(45.0, door.wall_width * 0.18))
    center = (door.face1 + door.face2) / 2.0
    if door.orientation == "horizontal":
        first = [
            (door.start, center - panel_depth),
            (midpoint + overlap, center - panel_depth),
            (midpoint + overlap, center),
            (door.start, center),
        ]
        second = [
            (midpoint - overlap, center),
            (door.end, center),
            (door.end, center + panel_depth),
            (midpoint - overlap, center + panel_depth),
        ]
    else:
        first = [
            (center - panel_depth, door.start),
            (center, door.start),
            (center, midpoint + overlap),
            (center - panel_depth, midpoint + overlap),
        ]
        second = [
            (center, midpoint - overlap),
            (center + panel_depth, midpoint - overlap),
            (center + panel_depth, door.end),
            (center, door.end),
        ]
    msp.add_lwpolyline(first, close=True, dxfattribs={"layer": DOOR_LAYER})
    msp.add_lwpolyline(second, close=True, dxfattribs={"layer": DOOR_LAYER})
    return [{"panel_depth_mm": panel_depth, "overlap_mm": overlap}]


def write_detection_artifacts(
    gray: np.ndarray,
    mask: np.ndarray,
    plan_bbox: tuple[int, int, int, int],
    doors: Sequence[PixelDoor],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", mask)[1].tofile(str(output.with_name(output.stem + "_door_mask.png")))
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x1, y1, x2, y2 = plan_bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 170, 0), 2)
    for door in doors:
        color = (255, 120, 0) if door.kind == "sliding" else (0, 0, 255)
        cv2.rectangle(overlay, (door.x, door.y), (door.x + door.width, door.y + door.height), color, 3)
        for leaf in door.leaves:
            cv2.circle(overlay, (int(round(leaf.center_x)), int(round(leaf.center_y))), int(round(leaf.radius)), color, 1)
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
        color = (210, 40, 40) if entity.dxf.layer == DOOR_LAYER else ((0, 160, 185) if entity.dxf.layer == WINDOW_LAYER else (0, 0, 0))
        if entity.dxftype() == "LINE":
            draw.line((point(float(entity.dxf.start.x), float(entity.dxf.start.y)), point(float(entity.dxf.end.x), float(entity.dxf.end.y))), fill=color, width=2)
        elif entity.dxftype() == "LWPOLYLINE":
            points = [point(float(x), float(y)) for x, y in entity.get_points("xy")]
            if entity.closed and points:
                points.append(points[0])
            if len(points) >= 2:
                draw.line(points, fill=color, width=2)
        elif entity.dxftype() == "ARC":
            center = entity.dxf.center
            radius = float(entity.dxf.radius)
            box = [point(float(center.x) - radius, float(center.y) + radius), point(float(center.x) + radius, float(center.y) - radius)]
            draw.arc(box, start=-float(entity.dxf.end_angle), end=-float(entity.dxf.start_angle), fill=color, width=2)
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


def generate(
    source: Path,
    previous_image: Path,
    wall_window_dxf: Path,
    output: Path,
    data_path: Path,
    overall_width_mm: float,
    overall_height_mm: float,
    image_wall_bbox: tuple[int, int, int, int] | None = None,
    preprocessing_result: Path | None = None,
) -> dict:
    source_gray = read_gray(source)
    previous_gray = read_gray(previous_image)
    wall_widths = load_wall_widths(data_path)
    plan_bbox = image_wall_bbox or detect_plan_bbox(previous_gray)
    pixel_doors, door_mask, rejected_components = detect_doors(source_gray, previous_gray, plan_bbox)
    stage_issues: list[str] = []
    if not pixel_doors:
        stage_issues.append("差分图中未找到满足规则的平开门或推拉门；保留上一阶段图形并继续流程")

    doc = ezdxf.readfile(wall_window_dxf)
    doc.units = ezdxf.units.MM
    if DOOR_LAYER not in doc.layers:
        doc.layers.add(DOOR_LAYER, color=4, linetype="CONTINUOUS")
    msp = doc.modelspace()
    converted = explode_wall_polylines(msp)
    accepted: list[CadDoor] = []
    records: list[dict] = []
    for index, pixel in enumerate(pixel_doors, start=1):
        try:
            preferred_orientation = None
            if pixel.kind == "sliding":
                preferred_orientation = "horizontal" if pixel.width >= pixel.height else "vertical"
            elif pixel.kind == "swing_double":
                # 当前成对规则只合并左右相邻的两个门弧，因此它们共用竖向门洞。
                preferred_orientation = "vertical"
            orientation, face_pixel1, face_pixel2 = detect_wall_context(
                previous_gray,
                pixel,
                plan_bbox,
                wall_widths,
                overall_width_mm,
                overall_height_mm,
                preferred_orientation,
            )
            raw = raw_cad_door(
                pixel,
                orientation,
                face_pixel1,
                face_pixel2,
                plan_bbox,
                overall_width_mm,
                overall_height_mm,
            )
            normalized, reason = normalize_door_to_wall(raw, msp, wall_widths)
        except RuntimeError as error:
            normalized, reason = None, str(error)
            raw = None
        if normalized is None:
            records.append({
                "id": f"D{index:02d}",
                "status": "REJECTED",
                "reason": reason,
                "pixel": asdict(pixel),
                "mapped_cad": asdict(raw) if raw is not None else None,
            })
            continue
        if overlaps_window(msp, normalized):
            records.append({
                "id": f"D{index:02d}",
                "status": "REJECTED",
                "reason": "门洞与已生成窗户重叠",
                "pixel": asdict(pixel),
                "mapped_cad": asdict(normalized),
            })
            continue
        if any(
            old.orientation == normalized.orientation
            and abs((old.face1 + old.face2) - (normalized.face1 + normalized.face2)) <= 8.0
            and min(old.end, normalized.end) - max(old.start, normalized.start) > 20.0
            for old in accepted
        ):
            records.append({
                "id": f"D{index:02d}",
                "status": "REJECTED",
                "reason": "与已接受门洞重叠",
                "pixel": asdict(pixel),
                "mapped_cad": asdict(normalized),
            })
            continue

        changed, before = split_wall_faces(msp, normalized)
        after = count_wall_overlaps(msp, normalized)
        if after:
            records.append({
                "id": f"D{index:02d}",
                "status": "REJECTED",
                "reason": "门洞范围内仍有墙体边界重叠",
                "pixel": asdict(pixel),
                "mapped_cad": asdict(normalized),
                "wall_overlaps_before": before,
                "wall_overlaps_after": after,
            })
            continue
        geometry = (
            add_sliding_geometry(msp, normalized)
            if normalized.kind == "sliding"
            else add_swing_geometry(msp, normalized, pixel, plan_bbox, overall_width_mm, overall_height_mm)
        )
        accepted.append(normalized)
        records.append({
            "id": f"D{index:02d}",
            "status": "ACCEPTED",
            "reason": None,
            "pixel": asdict(pixel),
            "cad": asdict(normalized),
            "opening_state": "CUT_FROM_CONTINUOUS_WALL" if changed else "EXISTING_OPENING",
            "cut_wall_entities": changed,
            "wall_overlaps_before": before,
            "wall_overlaps_after": after,
            "geometry": geometry,
        })

    if not accepted:
        stage_issues.append("门候选全部被墙体匹配规则拒绝；保留上一阶段图形并继续流程")
    try:
        wall_topology = rebuild_closed_wall_polylines(msp, max(wall_widths) + 1.0)
    except RuntimeError as error:
        stage_issues.append(f"墙体拓扑校核未通过：{error}")
        wall_topology = {"wall_topology": "needs_repair", "topology_error": str(error)}
    final_overlaps = sum(count_wall_overlaps(msp, door) for door in accepted)
    if final_overlaps:
        stage_issues.append(f"闭合墙体重建后门洞内仍有 {final_overlaps} 条墙体边界")
    door_geometry_collisions = validate_door_geometry(msp)
    if door_geometry_collisions:
        stage_issues.append(f"门线与墙体或窗线相交（共 {len(door_geometry_collisions)} 处）")
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output)
    auditor = doc.audit()
    report = {
        "source": str(source.resolve()),
        "previous_stage_image": str(previous_image.resolve()),
        "wall_window_dxf": str(wall_window_dxf.resolve()),
        "output": str(output.resolve()),
        "overall_width_mm": overall_width_mm,
        "overall_height_mm": overall_height_mm,
        "image_wall_bbox": {"x1": plan_bbox[0], "y1": plan_bbox[1], "x2": plan_bbox[2], "y2": plan_bbox[3]},
        "wall_width_candidates_mm": wall_widths,
        "detector": "stage-difference-quarter-arc-and-multiline-sliding-door",
        "preprocessing_evidence": load_preprocessing_evidence(preprocessing_result),
        "pixel_candidates": len(pixel_doors),
        "accepted_doors": len(accepted),
        "rejected_doors": len(records) - len(accepted),
        "rejected_difference_components": rejected_components,
        "converted_wall_polylines": converted,
        "doors": records,
        "topology": {
            "wall_overlaps_after": final_overlaps,
            "wall_topology": "closed-lwpolylines",
            **wall_topology,
            "door_lines": sum(1 for entity in msp if entity.dxftype() == "LINE" and entity.dxf.layer == DOOR_LAYER),
            "door_arcs": sum(1 for entity in msp if entity.dxftype() == "ARC" and entity.dxf.layer == DOOR_LAYER),
            "closed_sliding_panels": sum(1 for entity in msp if entity.dxftype() == "LWPOLYLINE" and entity.dxf.layer == DOOR_LAYER and entity.closed),
            "door_geometry_collisions": door_geometry_collisions,
            "door_geometry_collision_count": len(door_geometry_collisions),
        },
        "audit_errors": len(auditor.errors),
        "audit_fixes": len(auditor.fixes),
        "target_cad_visual_check": False,
        "stage_status": "needs_repair" if stage_issues else "ready",
        "repair_queue": stage_issues,
    }
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_detection_artifacts(source_gray, door_mask, plan_bbox, pixel_doors, output)
    draw_preview(doc, output.with_suffix(".png"), overall_width_mm, overall_height_mm)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="在墙体和窗户 DXF 上识别门、校验门洞并生成平开门或推拉门")
    parser.add_argument("source", type=Path, help="包含门的当前阶段户型图片")
    parser.add_argument("previous_image", type=Path, help="上一阶段仅含墙体和窗户的图片")
    parser.add_argument("wall_window_dxf", type=Path, help="上一阶段已校验的墙体和窗户 DXF")
    parser.add_argument("output", type=Path, help="带门的输出 DXF")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("data.json"), help="包含 wall_width 的 JSON")
    parser.add_argument("--overall-width-mm", type=float, default=DEFAULT_WIDTH_MM)
    parser.add_argument("--overall-height-mm", type=float, default=DEFAULT_HEIGHT_MM)
    parser.add_argument("--image-wall-bbox", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), help="图片中的墙体外包框；省略时自动识别")
    parser.add_argument("--preprocessing-result", type=Path, help="processing.py 生成的 result.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = generate(
        source=args.source,
        previous_image=args.previous_image,
        wall_window_dxf=args.wall_window_dxf,
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
