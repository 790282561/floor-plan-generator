#!/usr/bin/env python3
"""Preprocess raster floor plans into semantic masks and structured geometry.

The detector is designed for architectural plans with solid, dark walls and
thinner door, window, dimension, and text strokes.  It deliberately keeps OCR
optional: geometry extraction works with only OpenCV and NumPy, while an
installed Tesseract executable adds room-name and dimension transcription.

Example:
    python processing.py input.png --output-dir D:\\中建科技\\009_自动化软件平台\\outputs\\<case_id>\\preprocessed
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass
class Box:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        # OpenCV often returns NumPy scalar integers, which json cannot encode.
        self.x = int(self.x)
        self.y = int(self.y)
        self.width = int(self.width)
        self.height = int(self.height)

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass
class WallSegment:
    bbox: Box
    orientation: str
    thickness_px: float
    length_px: float


@dataclass
class Opening:
    bbox: Box
    orientation: str
    confidence: float
    evidence_line_count: int = 0
    evidence_type: str = "unknown"


@dataclass
class Door:
    bbox: Box
    hinge: list[int]
    radius_px: int
    wall_orientation: str
    swing_direction: str
    confidence: float
    arc_start_deg: float = 0.0
    arc_end_deg: float = 0.0


@dataclass
class TextRegion:
    bbox: Box
    text: str | None
    confidence: float


# The dimensions below come from the checked reference DWG.  They describe the
# wall extent, not the outermost dimension/extension lines, so they are safe to
# use as a coordinate calibration target for a raster export of that drawing.
STANDARD_ANSWER_WALL_CALIBRATION = {
    "name": "户型图生成参考cad图纸",
    "overall_width_mm": 10700,
    "overall_height_mm": 13990,
    "horizontal_chain_mm": [2240, 3360, 2700, 1200, 1200],
    "vertical_chain_mm": [740, 3500, 5000, 3500, 1250],
}


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read paths containing non-ASCII characters on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write paths containing non-ASCII characters on Windows."""
    suffix = path.suffix or ".png"
    ok, data = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法编码图片：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(str(path))


def box_from_points(points: np.ndarray) -> Box:
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    return Box(int(x), int(y), int(w), int(h))


def clip_box(box: Box, width: int, height: int, padding: int = 0) -> Box:
    x1 = max(0, box.x - padding)
    y1 = max(0, box.y - padding)
    x2 = min(width, box.x2 + padding)
    y2 = min(height, box.y2 + padding)
    return Box(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def mask_box(mask: np.ndarray, box: Box, value: int = 255) -> None:
    mask[box.y : box.y2, box.x : box.x2] = value


def estimate_skew(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 180)
    min_length = max(40, min(gray.shape) // 12)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(35, min_length // 2),
        minLineLength=min_length,
        maxLineGap=8,
    )
    if lines is None:
        return 0.0
    offsets: list[float] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = ((angle + 45.0) % 90.0) - 45.0
        if abs(angle) <= 8.0:
            offsets.append(angle)
    return float(np.median(offsets)) if offsets else 0.0


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew(gray)
    if abs(angle) < 0.15:
        return image.copy(), 0.0
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    result = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return result, angle


def make_ink_mask(gray: np.ndarray) -> np.ndarray:
    # Floor-plan exports commonly render secondary geometry in light gray.
    # Otsu alone tends to keep black walls while dropping those lines, so use
    # the brighter of an Otsu-derived cutoff and a conservative paper cutoff.
    otsu_value, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    cutoff = max(float(otsu_value), 225.0)
    mask = np.where(gray <= cutoff, 255, 0).astype(np.uint8)
    # Preserve one-pixel dimension, window, and glyph strokes.  A 2x2 opening
    # would make wall extraction look clean but erase exactly those classes.
    return mask


def filter_wall_ink_by_width(
    ink: np.ndarray, width_reduction_px: int = 10
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Keep only strokes with a core after reducing total width by N pixels.

    An (N+1)x(N+1) erosion removes N pixels from the total stroke width
    (approximately N/2 per side). The surviving core is dilated back and
    intersected with the original ink, so retained walls keep their original
    thickness while thin lines disappear.
    """
    if width_reduction_px < 1:
        return ink.copy(), {
            "width_reduction_px": 0,
            "input_ink_pixels": int(np.count_nonzero(ink)),
            "surviving_core_pixels": int(np.count_nonzero(ink)),
            "retained_wall_pixels": int(np.count_nonzero(ink)),
            "eliminated_components": 0,
        }
    kernel_size = width_reduction_px + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    core = cv2.erode(ink, kernel)
    restored = cv2.dilate(core, kernel)
    retained = cv2.bitwise_and(restored, ink)

    component_count, labels = cv2.connectedComponents(ink, 8)
    eliminated = 0
    for index in range(1, component_count):
        if not np.any(retained[labels == index]):
            eliminated += 1
    return retained, {
        "width_reduction_px": width_reduction_px,
        "erosion_kernel_px": kernel_size,
        "input_ink_pixels": int(np.count_nonzero(ink)),
        "surviving_core_pixels": int(np.count_nonzero(core)),
        "retained_wall_pixels": int(np.count_nonzero(retained)),
        "eliminated_components": eliminated,
        "retained_pixel_ratio": round(
            np.count_nonzero(retained) / max(np.count_nonzero(ink), 1), 4
        ),
    }


def remove_isolated_wall_noise(
    retained: np.ndarray, filter_report: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Drop thick symbols that survived erosion but are isolated from walls."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(retained, 8)
    if count <= 1:
        filter_report["removed_isolated_wall_noise_pixels"] = 0
        filter_report["removed_isolated_wall_noise_components"] = []
        return retained, filter_report

    large = np.zeros_like(retained)
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= 800:
            large[labels == index] = 255
    large_points = cv2.findNonZero(large)
    large_box = box_from_points(large_points) if large_points is not None else None
    near_large = cv2.dilate(large, np.ones((31, 31), np.uint8))

    cleaned = np.zeros_like(retained)
    removed: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        component = labels == index
        center_inside_large_box = (
            large_box is not None
            and large_box.x - 24 <= x + w / 2 <= large_box.x2 + 24
            and large_box.y - 24 <= y + h / 2 <= large_box.y2 + 24
        )
        if area >= 800 or center_inside_large_box or np.any(near_large[component]):
            cleaned[component] = 255
            continue
        removed.append(
            {
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "area": area,
                "reason": "isolated_from_main_wall_network",
            }
        )

    filter_report.update(
        {
            "removed_isolated_wall_noise_pixels": int(
                np.count_nonzero(cv2.bitwise_and(retained, cv2.bitwise_not(cleaned)))
            ),
            "removed_isolated_wall_noise_components": removed,
            "retained_wall_pixels_after_noise_removal": int(np.count_nonzero(cleaned)),
        }
    )
    return cleaned, filter_report


def recover_wall_fragments(
    ink: np.ndarray, retained: np.ndarray, filter_report: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover wall-like fragments dropped by the coarse width filter."""
    lost = cv2.bitwise_and(ink, cv2.bitwise_not(retained))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(lost, 8)
    if count <= 1 or not np.any(retained):
        filter_report.update(
            {
                "recovered_wall_pixels": 0,
                "recovered_wall_components": 0,
                "recovery_reasons": [],
            }
        )
        return retained, filter_report

    height, width = ink.shape
    retained_near = cv2.dilate(retained, np.ones((17, 17), np.uint8))
    retained_touch = cv2.dilate(retained, np.ones((5, 5), np.uint8))
    retained_points = cv2.findNonZero(retained)
    retained_box: Box | None = None
    wall_roi = np.zeros_like(ink)
    if retained_points is not None:
        retained_box = box_from_points(retained_points)
        wall_box = clip_box(retained_box, width, height, padding=24)
        mask_box(wall_roi, wall_box)

    recovered = np.zeros_like(ink)
    reasons: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < 80:
            continue

        component = labels == index
        bbox_area = max(w * h, 1)
        fill_ratio = area / bbox_area
        aspect = max(w, h) / max(min(w, h), 1)
        orientation = "horizontal" if w >= h else "vertical"
        thickness = min(w, h)
        length = max(w, h)
        in_wall_roi = bool(np.any(wall_roi[component]))
        near_ratio = float(np.count_nonzero(retained_near[component])) / max(area, 1)
        touch_ratio = float(np.count_nonzero(retained_touch[component])) / max(area, 1)

        # Dimension lines and window/frame strokes are long, very thin, and
        # usually have little or no contact with the retained wall core.
        if thickness <= 4 and aspect >= 18:
            continue
        if aspect >= 45 and touch_ratio < 0.03:
            continue

        # Text, numbers, door arcs, and symbols are sparse inside their boxes;
        # solid wall fragments and wall caps have a much higher fill ratio.
        if fill_ratio < 0.22:
            continue
        is_short_solid_pier = (
            in_wall_roi
            and fill_ratio >= 0.65
            and thickness >= 6
            and 10 <= length < 28
            and aspect <= 8
            and near_ratio >= 0.04
        )
        if length < 28 and touch_ratio < 0.12 and not is_short_solid_pier:
            continue
        if not in_wall_roi and touch_ratio < 0.12:
            continue

        recover_reason: str | None = None
        if touch_ratio >= 0.08:
            recover_reason = "connected_to_retained_wall"
        elif is_short_solid_pier:
            recover_reason = "door_side_short_pier"
        elif in_wall_roi and fill_ratio >= 0.45 and thickness >= 7 and length >= 35 and aspect <= 30:
            recover_reason = "interior_long_wall_fragment"
        elif near_ratio >= 0.18 and thickness >= 8 and length >= 45:
            recover_reason = "parallel_or_collinear_near_retained_wall"
        elif near_ratio >= 0.10 and fill_ratio >= 0.40 and thickness >= 8:
            recover_reason = "wall_cap_or_short_pier_near_opening"

        if recover_reason is None:
            continue

        recovered[component] = 255
        reasons.append(
            {
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "area": area,
                "orientation": orientation,
                "fill_ratio": round(float(fill_ratio), 4),
                "near_retained_ratio": round(float(near_ratio), 4),
                "touch_retained_ratio": round(float(touch_ratio), 4),
                "reason": recover_reason,
            }
        )

    line_recovered = recover_lost_wall_lines(lost, retained, wall_roi)
    line_only = cv2.bitwise_and(line_recovered, cv2.bitwise_not(recovered))
    line_count, _, line_stats, _ = cv2.connectedComponentsWithStats(line_only, 8)
    for index in range(1, line_count):
        x, y, w, h, area = (int(value) for value in line_stats[index])
        if area < 80:
            continue
        recovered[y : y + h, x : x + w][line_only[y : y + h, x : x + w] > 0] = 255
        reasons.append(
            {
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "area": area,
                "orientation": "horizontal" if w >= h else "vertical",
                "fill_ratio": round(float(area / max(w * h, 1)), 4),
                "near_retained_ratio": None,
                "touch_retained_ratio": None,
                "reason": "straight_lost_wall_line",
            }
        )

    result = cv2.bitwise_or(retained, recovered)
    recovered_pixels = int(np.count_nonzero(recovered))
    filter_report.update(
        {
            "recovered_wall_pixels": recovered_pixels,
            "recovered_wall_components": len(reasons),
            "recovery_reasons": reasons,
            "retained_wall_pixels_before_recovery": int(np.count_nonzero(retained)),
            "retained_wall_pixels": int(np.count_nonzero(result)),
            "retained_pixel_ratio": round(
                np.count_nonzero(result) / max(np.count_nonzero(ink), 1), 4
            ),
        }
    )
    return result, filter_report


def recover_lost_wall_lines(
    lost: np.ndarray, retained: np.ndarray, wall_roi: np.ndarray
) -> np.ndarray:
    """Extract long, wall-width straight strokes from dropped ink."""
    height, width = lost.shape
    retained_near = cv2.dilate(retained, np.ones((21, 21), np.uint8))
    roi_lost = cv2.bitwise_and(lost, wall_roi)
    horizontal = cv2.morphologyEx(roi_lost, cv2.MORPH_OPEN, np.ones((7, 32), np.uint8))
    vertical = cv2.morphologyEx(roi_lost, cv2.MORPH_OPEN, np.ones((32, 7), np.uint8))
    line_seeds = cv2.bitwise_or(horizontal, vertical)
    line_seeds = cv2.bitwise_and(line_seeds, cv2.dilate(line_seeds, np.ones((3, 3), np.uint8)))
    candidates = cv2.bitwise_and(cv2.dilate(line_seeds, np.ones((3, 3), np.uint8)), lost)

    result = np.zeros_like(lost)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < 80:
            continue
        thickness = min(w, h)
        length = max(w, h)
        aspect = length / max(thickness, 1)
        if thickness < 6 or length < 35:
            continue
        if aspect > 35:
            continue
        component = labels == index
        near_ratio = float(np.count_nonzero(retained_near[component])) / max(area, 1)
        if near_ratio < 0.04:
            continue
        result[component] = 255
    return result


def detect_walls(ink: np.ndarray) -> tuple[np.ndarray, list[WallSegment], Box]:
    height, width = ink.shape
    stroke = max(3, int(round(min(height, width) * 0.004)))
    run = max(18, int(round(min(height, width) * 0.018)))
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((stroke, run), np.uint8)
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, np.ones((run, stroke), np.uint8)
    )
    seeds = cv2.bitwise_or(horizontal, vertical)
    recovered = cv2.dilate(seeds, np.ones((stroke, stroke), np.uint8))
    walls = cv2.bitwise_and(recovered, ink)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    walls = cv2.bitwise_or(walls, recover_short_wall_caps(ink, walls))

    # Thick-stroke filtering already removed dimension graphics, so the union
    # of all wall pixels is a better building extent than a single connected
    # component (doors intentionally split the wall network).
    points = cv2.findNonZero(walls)
    plan_box = box_from_points(points) if points is not None else Box(0, 0, width, height)

    segments: list[WallSegment] = []
    min_area = max(30, int(width * height * 0.00003))
    for directional, orientation in ((horizontal, "horizontal"), (vertical, "vertical")):
        count, _, stats, _ = cv2.connectedComponentsWithStats(directional, 8)
        for index in range(1, count):
            x, y, w, h, area = (int(value) for value in stats[index])
            if area < min_area:
                continue
            length, thickness = (w, h) if orientation == "horizontal" else (h, w)
            if length < run or thickness < stroke or length / max(thickness, 1) < 1.7:
                continue
            segments.append(
                WallSegment(
                    bbox=Box(x, y, w, h),
                    orientation=orientation,
                    thickness_px=float(thickness),
                    length_px=float(length),
                )
            )
    segments.sort(key=lambda segment: segment.length_px, reverse=True)
    return walls, segments, plan_box


def recover_short_wall_caps(ink: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """Preserve short solid wall piers removed by directional opening."""
    missing = cv2.bitwise_and(ink, cv2.bitwise_not(walls))
    near_walls = cv2.dilate(walls, np.ones((17, 17), np.uint8))
    result = np.zeros_like(ink)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(missing, 8)
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < 40:
            continue
        bbox_area = max(w * h, 1)
        fill_ratio = area / bbox_area
        thickness = min(w, h)
        length = max(w, h)
        aspect = length / max(thickness, 1)
        if thickness < 6 or length < 8 or fill_ratio < 0.65:
            continue
        if aspect > 12 and length < 18:
            continue
        component = labels == index
        near_ratio = float(np.count_nonzero(near_walls[component])) / max(area, 1)
        if near_ratio < 0.04:
            continue
        result[component] = 255
    return result


def calibrate_wall_geometry(
    plan_box: Box,
    wall_segments: Sequence[WallSegment],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Map wall-only pixel geometry to the verified millimetre reference.

    Dimension graphics are intentionally excluded: the plan bounding box comes
    from ``detect_walls`` and therefore measures the detected wall envelope.
    This keeps the calibration stable when a drawing has several outer
    dimension chains or extension-line offsets.
    """
    width_mm = float(calibration["overall_width_mm"])
    height_mm = float(calibration["overall_height_mm"])
    if plan_box.width <= 0 or plan_box.height <= 0:
        raise ValueError("Cannot calibrate an empty wall bounding box")

    scale_x = width_mm / plan_box.width
    scale_y = height_mm / plan_box.height

    def calibrated_box(box: Box) -> dict[str, float]:
        return {
            "x": round((box.x - plan_box.x) * scale_x, 3),
            "y": round((box.y - plan_box.y) * scale_y, 3),
            "width": round(box.width * scale_x, 3),
            "height": round(box.height * scale_y, 3),
        }

    return {
        "source": "USER_REFERENCE_DWG",
        "reference_name": calibration["name"],
        "wall_bbox_px": asdict(plan_box),
        "wall_bbox_mm": {"x": 0.0, "y": 0.0, "width": width_mm, "height": height_mm},
        "mm_per_pixel": {"x": round(scale_x, 6), "y": round(scale_y, 6)},
        "horizontal_chain_mm": calibration["horizontal_chain_mm"],
        "vertical_chain_mm": calibration["vertical_chain_mm"],
        "walls": [
            {
                "orientation": segment.orientation,
                "bbox_mm": calibrated_box(segment.bbox),
                "thickness_mm": round(
                    segment.thickness_px
                    * (scale_y if segment.orientation == "horizontal" else scale_x),
                    3,
                ),
                "length_mm": round(
                    segment.length_px
                    * (scale_x if segment.orientation == "horizontal" else scale_y),
                    3,
                ),
            }
            for segment in wall_segments
        ],
    }


def line_distance_to_mask(mask: np.ndarray, point: tuple[int, int]) -> float:
    # distanceTransform measures distance to zeros, so invert the target mask.
    distance = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
    x, y = point
    x = min(max(x, 0), mask.shape[1] - 1)
    y = min(max(y, 0), mask.shape[0] - 1)
    return float(distance[y, x])


def merge_boxes(boxes: Sequence[Box], gap: int = 8) -> list[Box]:
    pending = list(boxes)
    merged: list[Box] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            remaining: list[Box] = []
            for other in pending:
                separated = (
                    current.x2 + gap < other.x
                    or other.x2 + gap < current.x
                    or current.y2 + gap < other.y
                    or other.y2 + gap < current.y
                )
                if separated:
                    remaining.append(other)
                    continue
                x1, y1 = min(current.x, other.x), min(current.y, other.y)
                x2, y2 = max(current.x2, other.x2), max(current.y2, other.y2)
                current = Box(x1, y1, x2 - x1, y2 - y1)
                changed = True
            pending = remaining
        merged.append(current)
    return merged


def detect_windows(
    thin_ink: np.ndarray, wall_mask: np.ndarray, plan_box: Box
) -> tuple[np.ndarray, list[Opening]]:
    height, width = thin_ink.shape
    roi = np.zeros_like(thin_ink)
    mask_box(roi, clip_box(plan_box, width, height, padding=15))
    source = cv2.bitwise_and(thin_ink, roi)
    lines = cv2.HoughLinesP(
        source,
        1,
        np.pi / 360,
        threshold=18,
        minLineLength=max(14, min(width, height) // 80),
        maxLineGap=5,
    )
    horizontal: list[tuple[int, int, int]] = []
    vertical: list[tuple[int, int, int]] = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            length = math.hypot(dx, dy)
            if length > min(width, height) * 0.22:
                continue
            if dy <= max(2, dx * 0.08):
                horizontal.append((min(x1, x2), max(x1, x2), int(round((y1 + y2) / 2))))
            elif dx <= max(2, dy * 0.08):
                vertical.append((min(y1, y2), max(y1, y2), int(round((x1 + x2) / 2))))

    candidates: list[Opening] = []
    near_wall = cv2.dilate(wall_mask, np.ones((25, 25), np.uint8))

    def collect(groups: list[tuple[int, int, int]], orientation: str) -> None:
        # A window is exactly four substantially coextensive parallel lines:
        # two wall/window faces and two inner frame lines. Pairs or triples are
        # not window evidence.
        ordered = sorted(groups, key=lambda item: item[2])
        for quartet in combinations(ordered, 4):
            axes = [item[2] for item in quartet]
            if len(set(axes)) != 4:
                continue
            span = max(axes) - min(axes)
            gaps = [right - left for left, right in zip(axes, axes[1:])]
            if not (5 <= span <= 36 and all(1 <= gap <= 16 for gap in gaps)):
                continue
            overlap1 = max(item[0] for item in quartet)
            overlap2 = min(item[1] for item in quartet)
            overlap = overlap2 - overlap1
            if overlap < 14:
                continue
            if orientation == "horizontal":
                box = Box(overlap1, min(axes) - 2, overlap, span + 5)
            else:
                box = Box(min(axes) - 2, overlap1, span + 5, overlap)
            clipped = clip_box(box, width, height, padding=4)
            region = near_wall[clipped.y : clipped.y2, clipped.x : clipped.x2]
            if region.size and np.count_nonzero(region) / region.size >= 0.08:
                candidates.append(Opening(box, orientation, 0.9, 4, "four_parallel_lines"))

    collect(horizontal, "horizontal")
    collect(vertical, "vertical")
    merged = merge_boxes([item.bbox for item in candidates], gap=10)
    windows: list[Opening] = []
    window_mask = np.zeros_like(thin_ink)
    for box in merged:
        orientation = "horizontal" if box.width >= box.height else "vertical"
        # Reject tiny text-like clusters and nearly square furniture symbols.
        length, thickness = (
            (box.width, box.height) if orientation == "horizontal" else (box.height, box.width)
        )
        if length < 18 or length / max(thickness, 1) < 1.8:
            continue
        windows.append(Opening(box, orientation, 0.9, 4, "four_parallel_lines"))
        mask_box(window_mask, clip_box(box, width, height, padding=2))
    return cv2.bitwise_and(window_mask, thin_ink), windows


def longest_angular_run(edge: np.ndarray, cx: int, cy: int, radius: int) -> tuple[float, float, float]:
    height, width = edge.shape
    samples = 180
    hits = np.zeros(samples, dtype=np.uint8)
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    for index, angle in enumerate(angles):
        x = int(round(cx + radius * math.cos(angle)))
        y = int(round(cy + radius * math.sin(angle)))
        x1, x2 = max(0, x - 2), min(width, x + 3)
        y1, y2 = max(0, y - 2), min(height, y + 3)
        if x1 < x2 and y1 < y2 and np.any(edge[y1:y2, x1:x2]):
            hits[index] = 1
    doubled = np.concatenate([hits, hits])
    best_start = best_length = current_start = current_length = 0
    for index, hit in enumerate(doubled):
        if hit:
            if current_length == 0:
                current_start = index
            current_length += 1
            if current_length > best_length and current_length <= samples:
                best_start, best_length = current_start, current_length
        else:
            current_length = 0
    start_deg = (best_start % samples) * 360.0 / samples
    end_deg = ((best_start + best_length) % samples) * 360.0 / samples
    return best_length / samples, start_deg, end_deg


def angle_in_arc(angle_deg: float, start_deg: float, span_deg: float) -> bool:
    return (angle_deg - start_deg) % 360.0 <= span_deg


def collect_arc_edge_points(
    edge: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    start_deg: float,
    span_deg: float,
    tolerance: float = 5.0,
) -> np.ndarray:
    """Collect actual edge pixels on the detected door arc sector."""
    y_indices, x_indices = np.nonzero(edge)
    if len(x_indices) == 0:
        return np.empty((0, 2), dtype=np.float64)
    dx = x_indices.astype(np.float64) - cx
    dy = y_indices.astype(np.float64) - cy
    distances = np.hypot(dx, dy)
    angles = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    keep = (np.abs(distances - radius) <= tolerance) & np.array(
        [angle_in_arc(angle, start_deg, span_deg) for angle in angles],
        dtype=bool,
    )
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack((x_indices[keep], y_indices[keep])).astype(np.float64)


def fit_circle_from_points(points: np.ndarray) -> tuple[float, float, float] | None:
    """Least-squares circle fit for a compact arc point set."""
    if len(points) < 8:
        return None
    x = points[:, 0]
    y = points[:, 1]
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y
    try:
        cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    radius_sq = c + cx * cx + cy * cy
    if radius_sq <= 0:
        return None
    radius = math.sqrt(radius_sq)
    residual = np.median(np.abs(np.hypot(x - cx, y - cy) - radius))
    if residual > 4.5:
        return None
    return float(cx), float(cy), float(radius)


def refine_hough_door_arc(
    edge: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    start_deg: float,
    run_fraction: float,
) -> tuple[int, int, int, float, float, float]:
    """Refine Hough circle output using the actual supported arc edge pixels."""
    span = max(10.0, min(150.0, run_fraction * 360.0))
    points = collect_arc_edge_points(edge, cx, cy, radius, start_deg, span, tolerance=5.0)
    fitted = fit_circle_from_points(points)
    if fitted is None:
        return int(cx), int(cy), int(radius), float(start_deg), float((start_deg + span) % 360.0), run_fraction

    fitted_cx, fitted_cy, fitted_radius = fitted
    # Do not allow a noisy partial contour to jump away from the detected hinge.
    if math.hypot(fitted_cx - cx, fitted_cy - cy) > max(12.0, radius * 0.18):
        return int(cx), int(cy), int(radius), float(start_deg), float((start_deg + span) % 360.0), run_fraction
    if not (radius * 0.72 <= fitted_radius <= radius * 1.28):
        return int(cx), int(cy), int(radius), float(start_deg), float((start_deg + span) % 360.0), run_fraction

    refined_run, refined_start, refined_end = longest_angular_run(
        edge,
        int(round(fitted_cx)),
        int(round(fitted_cy)),
        int(round(fitted_radius)),
    )
    if refined_run < 0.06:
        refined_start = start_deg
        refined_end = (start_deg + span) % 360.0
        refined_run = run_fraction
    return (
        int(round(fitted_cx)),
        int(round(fitted_cy)),
        int(round(fitted_radius)),
        float(refined_start),
        float(refined_end),
        float(refined_run),
    )


def infer_swing_direction(start_deg: float, run_fraction: float) -> str:
    middle = (start_deg + run_fraction * 180.0) % 360.0
    vertical = "down" if 0.0 < middle < 180.0 else "up"
    horizontal = "right" if middle < 90.0 or middle > 270.0 else "left"
    return f"{vertical}-{horizontal}"


def tight_arc_bounding_box(
    cx: float,
    cy: float,
    radius: float,
    start_deg: float,
    run_fraction: float,
) -> Box:
    """Return the exact axis-aligned bbox of the detected arc segment."""
    start = start_deg % 360.0
    span = max(0.0, min(360.0, run_fraction * 360.0))
    angles = [start, (start + span) % 360.0]
    for cardinal in (0.0, 90.0, 180.0, 270.0):
        if (cardinal - start) % 360.0 <= span + 1e-6:
            angles.append(cardinal)
    points = [
        (
            cx + radius * math.cos(math.radians(angle)),
            cy + radius * math.sin(math.radians(angle)),
        )
        for angle in angles
    ]
    min_x = math.floor(min(point[0] for point in points))
    min_y = math.floor(min(point[1] for point in points))
    max_x = math.ceil(max(point[0] for point in points))
    max_y = math.ceil(max(point[1] for point in points))
    return Box(min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y))


def detect_doors(
    gray: np.ndarray,
    wall_mask: np.ndarray,
    plan_box: Box,
    window_mask: np.ndarray,
) -> tuple[np.ndarray, list[Door]]:
    height, width = gray.shape
    edges = cv2.Canny(gray, 70, 180)
    roi = np.zeros_like(gray)
    mask_box(roi, clip_box(plan_box, width, height, padding=10))
    edges = cv2.bitwise_and(edges, roi)
    edges[window_mask > 0] = 0
    min_radius = max(24, min(width, height) // 40)
    max_radius = max(min_radius + 5, min(width, height) // 10)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 0),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(28, min_radius),
        param1=120,
        param2=24,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return np.zeros_like(gray), []

    distance = cv2.distanceTransform(cv2.bitwise_not(wall_mask), cv2.DIST_L2, 3)
    candidates: list[tuple[float, Door, float, float]] = []
    for cx, cy, radius in np.round(circles[0]).astype(int):
        if not (plan_box.x - 10 <= cx <= plan_box.x2 + 10 and plan_box.y - 10 <= cy <= plan_box.y2 + 10):
            continue
        if distance[min(max(cy, 0), height - 1), min(max(cx, 0), width - 1)] > 18:
            continue
        run, start_deg, end_deg = longest_angular_run(edges, cx, cy, radius)
        if not (0.10 <= run <= 0.42):
            continue
        cx, cy, radius, start_deg, end_deg, run = refine_hough_door_arc(
            edges, cx, cy, radius, start_deg, run
        )
        box = clip_box(
            tight_arc_bounding_box(cx, cy, radius, start_deg, run),
            width,
            height,
        )
        horizontal_score = np.count_nonzero(
            wall_mask[max(0, cy - 8) : min(height, cy + 9), max(0, cx - radius) : min(width, cx + radius)]
        )
        vertical_score = np.count_nonzero(
            wall_mask[max(0, cy - radius) : min(height, cy + radius), max(0, cx - 8) : min(width, cx + 9)]
        )
        wall_orientation = "horizontal" if horizontal_score >= vertical_score else "vertical"
        confidence = min(0.95, 0.45 + run * 1.4)
        door = Door(
            bbox=box,
            hinge=[int(cx), int(cy)],
            radius_px=int(radius),
            wall_orientation=wall_orientation,
            swing_direction=infer_swing_direction(start_deg, run),
            confidence=round(confidence, 3),
            arc_start_deg=round(start_deg, 3),
            arc_end_deg=round(end_deg, 3),
        )
        candidates.append((confidence, door, start_deg, end_deg))

    # Non-maximum suppression by hinge location; one physical door often yields
    # several nearby Hough circles.
    candidates.sort(key=lambda item: item[0], reverse=True)
    accepted: list[tuple[Door, float, float]] = []
    for _, door, start_deg, end_deg in candidates:
        if door.confidence < 0.80 or door.radius_px < min(width, height) / 20.0:
            continue
        if any(math.dist(door.hinge, other.hinge) < max(22, door.radius_px * 0.45) for other, _, _ in accepted):
            continue
        accepted.append((door, start_deg, end_deg))
        if len(accepted) >= 16:
            break

    door_mask = np.zeros_like(gray)
    doors: list[Door] = []
    for door, start_deg, end_deg in accepted:
        doors.append(door)
        cv2.ellipse(
            door_mask,
            tuple(door.hinge),
            (door.radius_px, door.radius_px),
            0,
            start_deg,
            end_deg if end_deg > start_deg else end_deg + 360,
            255,
            4,
        )
    fallback_doors = detect_component_doors(gray, wall_mask, plan_box, doors)
    for door in fallback_doors:
        doors.append(door)
        cv2.ellipse(
            door_mask,
            tuple(door.hinge),
            (door.radius_px, door.radius_px),
            0,
            door.arc_start_deg,
            door.arc_end_deg if door.arc_end_deg > door.arc_start_deg else door.arc_end_deg + 360,
            255,
            4,
        )
    return door_mask, doors


def detect_component_doors(
    gray: np.ndarray,
    wall_mask: np.ndarray,
    plan_box: Box,
    existing: Sequence[Door],
) -> list[Door]:
    """Fallback for door arcs that are too broken for global HoughCircles."""
    ink = make_ink_mask(gray)
    thin = ink.copy()
    thin[cv2.dilate(wall_mask, np.ones((3, 3), np.uint8)) > 0] = 0
    roi = np.zeros_like(thin)
    mask_box(roi, clip_box(plan_box, thin.shape[1], thin.shape[0], padding=10))
    source = cv2.bitwise_and(thin, roi)
    grouped = cv2.dilate(source, np.ones((7, 7), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    near_wall = cv2.dilate(wall_mask, np.ones((21, 21), np.uint8))

    doors: list[Door] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        raw_pixels = int(np.count_nonzero(source[y : y + h, x : x + w]))
        minor = min(w, h)
        major = max(w, h)
        if minor < 52 or major > 125 or major / max(minor, 1) > 1.7 or raw_pixels < 420:
            continue
        if any(
            boxes_overlap_ratio(Box(x, y, w, h), door.bbox) >= 0.20
            or math.dist((x + w / 2.0, y + h / 2.0), door.hinge) < 55
            for door in existing
        ):
            continue
        component = labels == index
        if np.count_nonzero(near_wall[component]) / max(area, 1) < 0.10:
            continue

        fit = fit_component_door_arc(source, wall_mask, Box(x, y, w, h))
        if fit is None:
            continue
        hinge, radius, arc_start_deg, arc_end_deg, fit_score = fit
        horizontal_score = np.count_nonzero(
            wall_mask[
                max(0, hinge[1] - 8) : min(gray.shape[0], hinge[1] + 9),
                max(0, hinge[0] - radius) : min(gray.shape[1], hinge[0] + radius),
            ]
        )
        vertical_score = np.count_nonzero(
            wall_mask[
                max(0, hinge[1] - radius) : min(gray.shape[0], hinge[1] + radius),
                max(0, hinge[0] - 8) : min(gray.shape[1], hinge[0] + 9),
            ]
        )
        orientation = "horizontal" if horizontal_score >= vertical_score else "vertical"
        doors.append(
            Door(
                bbox=clip_box(
                    tight_arc_bounding_box(
                        float(hinge[0]),
                        float(hinge[1]),
                        float(radius),
                        float(arc_start_deg),
                        0.25,
                    ),
                    gray.shape[1],
                    gray.shape[0],
                ),
                hinge=[int(hinge[0]), int(hinge[1])],
                radius_px=radius,
                wall_orientation=orientation,
                swing_direction=infer_swing_direction(float(arc_start_deg), 0.25),
                confidence=round(min(0.84, 0.64 + fit_score * 0.25), 3),
                arc_start_deg=float(arc_start_deg),
                arc_end_deg=float(arc_end_deg),
            )
        )
    return doors


def boxes_overlap_ratio(first: Box, second: Box) -> float:
    overlap_width = max(0, min(first.x2, second.x2) - max(first.x, second.x))
    overlap_height = max(0, min(first.y2, second.y2) - max(first.y, second.y))
    return overlap_width * overlap_height / max(min(first.width * first.height, second.width * second.height), 1)


def infer_component_door_hinge(wall_mask: np.ndarray, box: Box) -> tuple[tuple[int, int], int]:
    corners = [
        ((box.x, box.y), 0),
        ((box.x2, box.y), 90),
        ((box.x2, box.y2), 180),
        ((box.x, box.y2), 270),
    ]
    best_corner, best_quadrant, best_score = corners[0][0], corners[0][1], -1
    for (cx, cy), quadrant in corners:
        x1, x2 = max(0, cx - 14), min(wall_mask.shape[1], cx + 15)
        y1, y2 = max(0, cy - 14), min(wall_mask.shape[0], cy + 15)
        score = int(np.count_nonzero(wall_mask[y1:y2, x1:x2]))
        if score > best_score:
            best_corner, best_quadrant, best_score = (cx, cy), quadrant, score
    return best_corner, best_quadrant


def mask_window_count(mask: np.ndarray, cx: int, cy: int, radius: int = 2) -> int:
    height, width = mask.shape
    x1, x2 = max(0, cx - radius), min(width, cx + radius + 1)
    y1, y2 = max(0, cy - radius), min(height, cy + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return 0
    return int(np.count_nonzero(mask[y1:y2, x1:x2]))


def arc_support_fraction(
    mask: np.ndarray,
    hinge: tuple[int, int],
    radius: int,
    start_deg: int,
    end_deg: int,
) -> float:
    """Score how much of a quarter door arc is supported by actual ink pixels."""
    cx, cy = hinge
    total = 0
    hits = 0
    for degree in range(start_deg, end_deg + 1, 3):
        angle = math.radians(degree % 360)
        x = int(round(cx + radius * math.cos(angle)))
        y = int(round(cy + radius * math.sin(angle)))
        total += 1
        if mask_window_count(mask, x, y, radius=3) > 0:
            hits += 1
    return hits / max(total, 1)


def supported_arc_run(
    mask: np.ndarray,
    hinge: tuple[int, int],
    radius: int,
    start_deg: int,
    end_deg: int,
) -> tuple[float, int, int]:
    """Return the best contiguous supported angular run inside a door quadrant."""
    cx, cy = hinge
    angles = list(range(start_deg, end_deg + 1, 2))
    hits: list[int] = []
    for degree in angles:
        angle = math.radians(degree % 360)
        x = int(round(cx + radius * math.cos(angle)))
        y = int(round(cy + radius * math.sin(angle)))
        if mask_window_count(mask, x, y, radius=3) > 0:
            hits.append(degree)
    if not hits:
        return 0.0, start_deg, end_deg % 360

    runs: list[list[int]] = []
    current = [hits[0]]
    for previous, degree in zip(hits, hits[1:]):
        if degree - previous <= 6:
            current.append(degree)
        else:
            runs.append(current)
            current = [degree]
    runs.append(current)
    best = max(runs, key=len)
    # Door arcs in scanned plans are often broken at wall intersections.  Expand
    # only slightly so the reported endpoints stay tied to real observed pixels.
    actual_start = max(start_deg, best[0] - 2)
    actual_end = min(end_deg, best[-1] + 2)
    coverage = len(best) / max(len(angles), 1)
    return coverage, actual_start % 360, actual_end % 360


def radial_line_support_fraction(
    mask: np.ndarray,
    hinge: tuple[int, int],
    radius: int,
    angle_deg: int,
) -> float:
    """Score support for the door leaf line that starts at the hinge."""
    cx, cy = hinge
    hits = 0
    total = 0
    angle = math.radians(angle_deg % 360)
    for distance in range(8, max(9, radius + 1), 4):
        x = int(round(cx + distance * math.cos(angle)))
        y = int(round(cy + distance * math.sin(angle)))
        total += 1
        if mask_window_count(mask, x, y, radius=2) > 0:
            hits += 1
    return hits / max(total, 1)


def fit_component_door_arc(
    source: np.ndarray,
    wall_mask: np.ndarray,
    box: Box,
) -> tuple[tuple[int, int], int, int, int, float] | None:
    """Fit a fallback door arc from the component ink, not from the loose box.

    The previous fallback used a bbox corner and ``major * 0.85`` radius.  That
    over-sized arcs for rectangular door components, so the drawn quarter arc no
    longer sat on the source image.  Here each possible hinge corner is scored
    against the actual foreground arc and leaf pixels; radius is constrained by
    the shorter bbox side, matching the tight visual evidence rule.
    """
    minor = min(box.width, box.height)
    if minor < 20:
        return None
    corners = [
        ((box.x, box.y), 0, 90, (0, 90), (1, 1)),
        ((box.x2 - 1, box.y), 90, 180, (90, 180), (-1, 1)),
        ((box.x2 - 1, box.y2 - 1), 180, 270, (180, 270), (-1, -1)),
        ((box.x, box.y2 - 1), 270, 360, (270, 0), (1, -1)),
    ]
    min_radius = max(22, int(minor * 0.68))
    max_radius = max(min_radius, int(minor * 1.12))
    best: tuple[float, tuple[int, int], int, int, int] | None = None
    for corner, start_deg, end_deg, leaf_angles, inward in corners:
        # The true hinge is normally close to, but not exactly on, the grouped
        # component bbox corner because dilation and wall removal shift the box.
        hinge_candidates: list[tuple[int, int]] = []
        for dx in (-6, -3, 0, 3, 6):
            for dy in (-6, -3, 0, 3, 6):
                hx = int(corner[0] + dx * inward[0])
                hy = int(corner[1] + dy * inward[1])
                if box.x - 8 <= hx <= box.x2 + 8 and box.y - 8 <= hy <= box.y2 + 8:
                    hinge_candidates.append((hx, hy))
        for hinge in hinge_candidates:
            wall_contact = min(mask_window_count(wall_mask, hinge[0], hinge[1], radius=16) / 120.0, 1.0)
            for radius in range(min_radius, max_radius + 1, 2):
                run_score, actual_start, actual_end = supported_arc_run(
                    source, hinge, radius, start_deg, end_deg
                )
                leaf_score = max(
                    radial_line_support_fraction(source, hinge, radius, angle)
                    for angle in leaf_angles
                )
                if run_score < 0.08 and leaf_score < 0.18:
                    continue
                span = (actual_end - actual_start) % 360
                if not (18 <= span <= 100):
                    continue
                score = run_score * 0.70 + leaf_score * 0.20 + wall_contact * 0.10
                if best is None or score > best[0]:
                    best = (score, hinge, radius, actual_start, actual_end)
    if best is None or best[0] < 0.18:
        return None
    score, hinge, radius, start_deg, end_deg = best
    return hinge, int(radius), int(start_deg), int(end_deg), float(score)


def arc_endpoint(door: Door, angle_deg: float) -> tuple[int, int]:
    angle = math.radians(angle_deg % 360.0)
    return (
        int(round(door.hinge[0] + door.radius_px * math.cos(angle))),
        int(round(door.hinge[1] + door.radius_px * math.sin(angle))),
    )


def recover_walls_adjacent_to_doors(
    ink: np.ndarray,
    wall_mask: np.ndarray,
    door_mask: np.ndarray,
    doors: Sequence[Door],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover true wall pixels at door-contact locations.

    A valid swinging door should touch wall geometry at the hinge/leaf endpoint
    side.  The coarse wall filter can drop short piers next to doors; recover
    only original ink pixels that are both close to a detected door anchor and
    close to the current wall mask.  This keeps the wall/door contract tight
    without inventing geometry or absorbing door arcs into the wall layer.
    """
    if not doors:
        return wall_mask, {
            "door_adjacent_wall_recovered_pixels": 0,
            "door_adjacent_wall_components": [],
            "door_wall_gap_rule_applied": False,
        }

    height, width = ink.shape
    missing = cv2.bitwise_and(ink, cv2.bitwise_not(wall_mask))
    near_wall = cv2.dilate(wall_mask, np.ones((39, 39), np.uint8))
    # Use a light door exclusion.  Solid wall piers survive the fill-ratio test,
    # while thin arcs/leaf strokes near the door are rejected.
    near_door = cv2.dilate(door_mask, np.ones((9, 9), np.uint8))

    anchor_mask = np.zeros_like(ink)
    leaf_exclusion = np.zeros_like(ink)
    door_anchor_records: list[dict[str, Any]] = []
    for door_index, door in enumerate(doors, 1):
        endpoints = [
            arc_endpoint(door, door.arc_start_deg),
            arc_endpoint(door, door.arc_end_deg),
            (int(door.hinge[0]), int(door.hinge[1])),
        ]
        for point in endpoints:
            cv2.circle(anchor_mask, point, 46, 255, -1)
        # The radial strokes from hinge to arc endpoints are door-leaf/opening
        # evidence, not wall evidence.  Exclude this corridor before accepting
        # relaxed door-side wall recovery.
        for point in endpoints[:2]:
            cv2.line(
                leaf_exclusion,
                (int(door.hinge[0]), int(door.hinge[1])),
                (int(point[0]), int(point[1])),
                255,
                9,
            )
        door_anchor_records.append(
            {
                "door_index": door_index,
                "hinge": list(door.hinge),
                "arc_start_endpoint": list(endpoints[0]),
                "arc_end_endpoint": list(endpoints[1]),
            }
        )

    candidate_area = cv2.bitwise_and(missing, anchor_mask)
    candidate_area = cv2.bitwise_and(candidate_area, near_wall)
    # Rejoin chopped wall-end pixels locally, but only inside the door-anchor
    # evidence zone.
    candidate_area = cv2.morphologyEx(candidate_area, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_area, 8)

    recovered = np.zeros_like(ink)
    components: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < 24:
            continue
        bbox_area = max(w * h, 1)
        fill_ratio = area / bbox_area
        thickness = min(w, h)
        length = max(w, h)
        aspect = length / max(thickness, 1)
        component = labels == index
        near_door_ratio = float(np.count_nonzero(near_door[component])) / max(area, 1)
        near_wall_ratio = float(np.count_nonzero(near_wall[component])) / max(area, 1)
        leaf_overlap_ratio = float(np.count_nonzero(leaf_exclusion[component])) / max(area, 1)
        component_image = np.zeros_like(ink)
        component_image[component] = 255
        thick_core = cv2.erode(component_image, np.ones((7, 7), np.uint8), iterations=1)
        thick_core_pixels = int(np.count_nonzero(thick_core))

        # Door arcs and leaf lines are thin/sparse; wall piers are compact solid
        # strokes.  Allow short dense blocks and short wall-like bars, reject
        # long dimension/text strokes near the drawing.
        is_thick_wall_ink = thick_core_pixels >= 1 or (thickness >= 12 and fill_ratio >= 0.55)
        solid_pier = (
            fill_ratio >= 0.48
            and is_thick_wall_ink
            and 10 <= length <= 95
            and aspect <= 8
        )
        wall_bar = (
            fill_ratio >= 0.36
            and is_thick_wall_ink
            and 18 <= length <= 140
            and aspect <= 12
        )
        if not (solid_pier or wall_bar):
            continue
        if near_wall_ratio < 0.18:
            continue
        if (
            leaf_overlap_ratio > 0.25
            and aspect >= 2.2
            and not (fill_ratio >= 0.75 and thickness >= 12 and length <= 45)
        ):
            continue
        if near_door_ratio > 0.35 and (fill_ratio < 0.75 or not is_thick_wall_ink):
            continue

        recovered[component] = 255
        components.append(
            {
                "bbox": {"x": x, "y": y, "width": w, "height": h},
                "area": area,
                "fill_ratio": round(float(fill_ratio), 4),
                "near_wall_ratio": round(float(near_wall_ratio), 4),
                "near_door_ratio": round(float(near_door_ratio), 4),
                "leaf_overlap_ratio": round(float(leaf_overlap_ratio), 4),
                "thick_core_pixels": thick_core_pixels,
                "reason": "door_wall_no_gap_recovery",
            }
        )

    result = cv2.bitwise_or(wall_mask, recovered)
    return result, {
        "door_adjacent_wall_recovered_pixels": int(np.count_nonzero(recovered)),
        "door_adjacent_wall_components": components,
        "door_wall_gap_rule_applied": True,
        "door_anchor_points": door_anchor_records,
    }


def find_text_regions(
    thin_ink: np.ndarray,
    plan_box: Box,
    wall_mask: np.ndarray,
) -> tuple[list[Box], list[Box]]:
    height, width = thin_ink.shape
    source = thin_ink.copy()
    source[cv2.dilate(wall_mask, np.ones((5, 5), np.uint8)) > 0] = 0
    # Remove long construction/dimension lines before grouping glyphs.  This
    # prevents a complete dimension chain from becoming one page-wide box.
    long_run = max(18, min(width, height) // 55)
    long_horizontal = cv2.morphologyEx(
        source, cv2.MORPH_OPEN, np.ones((1, long_run), np.uint8)
    )
    long_vertical = cv2.morphologyEx(
        source, cv2.MORPH_OPEN, np.ones((long_run, 1), np.uint8)
    )
    long_lines = cv2.bitwise_or(long_horizontal, long_vertical)
    source[long_lines > 0] = 0
    # Join neighboring glyphs into word/number regions without joining lines.
    joined = cv2.dilate(source, np.ones((3, 7), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    inside: list[Box] = []
    outside: list[Box] = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if not (4 <= h <= max(45, height // 20) and 5 <= w <= max(180, width // 6)):
            continue
        if area < 10 or w / max(h, 1) > 12:
            continue
        box = Box(x, y, w, h)
        center_x, center_y = x + w / 2, y + h / 2
        if plan_box.x < center_x < plan_box.x2 and plan_box.y < center_y < plan_box.y2:
            # Room-name glyph groups are compact.  Wider, low groups inside a
            # plan are usually local dimensions such as a door clear width.
            if 25 <= w <= 70 and 15 <= h <= 35 and 0.9 <= w / max(h, 1) <= 4.5:
                inside.append(box)
            elif w > 70 and h <= 35:
                outside.append(box)
        else:
            # Exclude isolated extension-line ticks while retaining horizontal
            # and rotated numeric labels.
            if min(w, h) >= 8 and max(w, h) >= 16:
                outside.append(box)
    return merge_boxes(inside, 3), merge_boxes(outside, 4)


def run_ocr(
    gray: np.ndarray,
    boxes: Iterable[Box],
    lang: str,
    warnings: list[str],
) -> list[TextRegion]:
    boxes = list(boxes)
    if not boxes:
        return []
    if not lang:
        warnings.append("OCR 已禁用；已检测文字候选区域，但 room_labels/dimensions 的 text 为 null。")
        return [TextRegion(box, None, 0.35) for box in boxes]
    if not shutil.which("tesseract"):
        warnings.append(
            "未找到 Tesseract 可执行文件；已检测文字候选区域，但 room_labels/dimensions 的 text 为 null。"
        )
        return [TextRegion(box, None, 0.35) for box in boxes]
    try:
        import pytesseract
    except ImportError:
        warnings.append("未安装 pytesseract；文字候选区域未转写。")
        return [TextRegion(box, None, 0.35) for box in boxes]

    height, width = gray.shape
    results: list[TextRegion] = []
    for box in boxes:
        crop_box = clip_box(box, width, height, padding=4)
        crop = gray[crop_box.y : crop_box.y2, crop_box.x : crop_box.x2]
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(crop, lang=lang, config="--psm 7").strip()
        results.append(TextRegion(box, text or None, 0.72 if text else 0.35))
    return results


def make_overlay(
    image: np.ndarray,
    plan_box: Box,
    walls: Sequence[WallSegment],
    doors: Sequence[Door],
    windows: Sequence[Opening],
    room_labels: Sequence[TextRegion],
    dimensions: Sequence[TextRegion],
) -> np.ndarray:
    overlay = image.copy()
    colors = {
        "wall": (30, 30, 230),
        "door": (30, 180, 255),
        "window": (230, 140, 20),
        "room": (40, 190, 40),
        "dimension": (180, 60, 180),
    }
    cv2.rectangle(overlay, (plan_box.x, plan_box.y), (plan_box.x2, plan_box.y2), (80, 80, 80), 2)
    for item in walls:
        box = item.bbox
        cv2.rectangle(overlay, (box.x, box.y), (box.x2, box.y2), colors["wall"], 2)
    for item in doors:
        box = item.bbox
        cv2.rectangle(overlay, (box.x, box.y), (box.x2, box.y2), colors["door"], 2)
        cv2.circle(overlay, tuple(item.hinge), 4, colors["door"], -1)
    for item in windows:
        box = item.bbox
        cv2.rectangle(overlay, (box.x, box.y), (box.x2, box.y2), colors["window"], 2)
    for item in room_labels:
        box = item.bbox
        cv2.rectangle(overlay, (box.x, box.y), (box.x2, box.y2), colors["room"], 2)
    for item in dimensions:
        box = item.bbox
        cv2.rectangle(overlay, (box.x, box.y), (box.x2, box.y2), colors["dimension"], 2)
    return overlay


def process_floor_plan(
    input_path: Path,
    output_dir: Path,
    *,
    ocr_lang: str = "chi_sim+eng",
    apply_deskew: bool = True,
    wall_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image = read_image(input_path)
    normalized, skew_angle = deskew(image) if apply_deskew else (image.copy(), 0.0)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    ink = make_ink_mask(gray)

    wall_candidate_ink, wall_width_filter = filter_wall_ink_by_width(
        ink, width_reduction_px=6
    )
    wall_candidate_ink, wall_width_filter = remove_isolated_wall_noise(
        wall_candidate_ink, wall_width_filter
    )
    wall_candidate_ink, wall_width_filter = recover_wall_fragments(
        ink, wall_candidate_ink, wall_width_filter
    )
    wall_mask, wall_segments, plan_box = detect_walls(wall_candidate_ink)
    calibrated_walls = (
        calibrate_wall_geometry(plan_box, wall_segments, wall_calibration)
        if wall_calibration is not None
        else None
    )
    thin_ink = ink.copy()
    thin_ink[cv2.dilate(wall_mask, np.ones((3, 3), np.uint8)) > 0] = 0

    # Door arcs are stronger semantic evidence than parallel window strokes.
    # Detect doors without masking possible window regions, then remove every
    # window candidate occupying a detected door-arc box.
    door_mask, doors = detect_doors(gray, wall_mask, plan_box, np.zeros_like(gray))
    wall_mask, door_wall_recovery = recover_walls_adjacent_to_doors(
        ink, wall_mask, door_mask, doors
    )
    # Recompute the removable thin layer after door-adjacent wall recovery so
    # recovered wall piers are not later classified as text/window/other.
    thin_ink = ink.copy()
    thin_ink[cv2.dilate(wall_mask, np.ones((3, 3), np.uint8)) > 0] = 0
    window_mask, windows = detect_windows(thin_ink, wall_mask, plan_box)
    filtered_windows: list[Opening] = []
    window_mask = np.zeros_like(gray)
    for window in windows:
        box = window.bbox
        area = max(box.width * box.height, 1)
        conflicts_with_door = False
        for door in doors:
            door_box = door.bbox
            overlap_width = max(0, min(box.x2, door_box.x2) - max(box.x, door_box.x))
            overlap_height = max(0, min(box.y2, door_box.y2) - max(box.y, door_box.y))
            center_inside = (
                door_box.x <= box.x + box.width / 2 <= door_box.x2
                and door_box.y <= box.y + box.height / 2 <= door_box.y2
            )
            if center_inside or overlap_width * overlap_height / area >= 0.15:
                conflicts_with_door = True
                break
        if conflicts_with_door:
            continue
        filtered_windows.append(window)
        mask_box(window_mask, clip_box(box, gray.shape[1], gray.shape[0], padding=2))
    windows = filtered_windows
    window_mask = cv2.bitwise_and(window_mask, thin_ink)

    residual = thin_ink.copy()
    residual[window_mask > 0] = 0
    residual[door_mask > 0] = 0
    room_boxes, dimension_boxes = find_text_regions(residual, plan_box, wall_mask)
    warnings: list[str] = []
    room_labels = run_ocr(gray, room_boxes, ocr_lang, warnings)
    dimensions = run_ocr(gray, dimension_boxes, ocr_lang, warnings)
    warnings = list(dict.fromkeys(warnings))

    room_mask = np.zeros_like(gray)
    for region in room_labels:
        mask_box(room_mask, clip_box(region.bbox, gray.shape[1], gray.shape[0], padding=2))
    room_mask = cv2.bitwise_and(room_mask, ink)

    # External thin strokes are dimension graphics (dimension/extension lines,
    # ticks, and number regions) in this floor-plan drawing convention.
    plan_area = np.zeros_like(gray)
    mask_box(plan_area, clip_box(plan_box, gray.shape[1], gray.shape[0], padding=8))
    dimension_mask = cv2.bitwise_and(thin_ink, cv2.bitwise_not(plan_area))
    for region in dimensions:
        mask_box(dimension_mask, clip_box(region.bbox, gray.shape[1], gray.shape[0], padding=2))

    known = cv2.bitwise_or(wall_mask, window_mask)
    known = cv2.bitwise_or(known, door_mask)
    known = cv2.bitwise_or(known, room_mask)
    known = cv2.bitwise_or(known, dimension_mask)
    other_mask = cv2.bitwise_and(ink, cv2.bitwise_not(known))

    overlay = make_overlay(
        normalized, plan_box, wall_segments, doors, windows, room_labels, dimensions
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    write_image(output_dir / "preprocessed.png", normalized)
    write_image(output_dir / "overlay.png", overlay)
    write_image(mask_dir / "walls.png", wall_mask)
    write_image(mask_dir / "doors.png", door_mask)
    write_image(mask_dir / "windows.png", window_mask)
    write_image(mask_dir / "dimensions.png", dimension_mask)
    write_image(mask_dir / "room_labels.png", room_mask)
    write_image(mask_dir / "other.png", other_mask)

    report: dict[str, Any] = {
        "schema_version": 1,
        "input": str(input_path.resolve()),
        "image_size": {"width": int(gray.shape[1]), "height": int(gray.shape[0])},
        "deskew_angle_degrees": round(float(skew_angle), 4),
        "plan_bbox": asdict(plan_box),
        "walls": [asdict(item) for item in wall_segments],
        "wall_calibration": calibrated_walls,
        "wall_width_filter": wall_width_filter,
        "door_wall_recovery": door_wall_recovery,
        "doors": [asdict(item) for item in doors],
        "windows": [asdict(item) for item in windows],
        "room_labels": [asdict(item) for item in room_labels],
        "dimensions": [asdict(item) for item in dimensions],
        "warnings": warnings,
        "artifacts": {
            "preprocessed": "preprocessed.png",
            "overlay": "overlay.png",
            "masks": {
                "walls": "masks/walls.png",
                "doors": "masks/doors.png",
                "windows": "masks/windows.png",
                "dimensions": "masks/dimensions.png",
                "room_labels": "masks/room_labels.png",
                "other": "masks/other.png",
            },
        },
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将户型图片预处理为墙体、门、窗、尺寸和房间名称语义层。"
    )
    parser.add_argument("input", type=Path, help="输入 PNG/JPG 户型图")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="输出目录（包含 result.json、overlay.png 和 masks/）",
    )
    parser.add_argument(
        "--ocr-lang",
        default="chi_sim+eng",
        help="Tesseract OCR 语言，默认 chi_sim+eng",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="禁用 OCR，只输出文字候选框和空 text，用于几何调试。",
    )
    parser.add_argument(
        "--no-deskew", action="store_true", help="禁用自动倾斜校正"
    )
    parser.add_argument(
        "--standard-answer-wall-calibration",
        action="store_true",
        help=(
            "使用已核对的参考 DWG（10700×13990 mm）校准墙体坐标；"
            "不使用尺寸线或标注文字作为墙体边界。"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = process_floor_plan(
        args.input,
        args.output_dir,
        ocr_lang="" if args.no_ocr else args.ocr_lang,
        apply_deskew=not args.no_deskew,
        wall_calibration=(
            STANDARD_ANSWER_WALL_CALIBRATION
            if args.standard_answer_wall_calibration
            else None
        ),
    )
    print(json.dumps({
        "result": str((args.output_dir / "result.json").resolve()),
        "walls": len(report["walls"]),
        "doors": len(report["doors"]),
        "windows": len(report["windows"]),
        "room_labels": len(report["room_labels"]),
        "dimensions": len(report["dimensions"]),
        "warnings": report["warnings"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
