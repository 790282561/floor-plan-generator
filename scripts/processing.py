#!/usr/bin/env python3
"""Preprocess raster floor plans into semantic masks and structured geometry.

The detector is designed for architectural plans with solid, dark walls and
thinner door, window, dimension, and text strokes.  It deliberately keeps OCR
optional: geometry extraction works with only OpenCV and NumPy, while an
installed Tesseract executable adds room-name and dimension transcription.

Example:
    python processing.py input.png --output-dir output/preprocessed
"""

from __future__ import annotations

import argparse
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


@dataclass
class Door:
    bbox: Box
    hinge: list[int]
    radius_px: int
    wall_orientation: str
    swing_direction: str
    confidence: float


@dataclass
class TextRegion:
    bbox: Box
    text: str | None
    confidence: float


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
        for i, first in enumerate(groups):
            a1, a2, axis1 = first
            for second in groups[i + 1 :]:
                b1, b2, axis2 = second
                spacing = abs(axis1 - axis2)
                overlap1, overlap2 = max(a1, b1), min(a2, b2)
                overlap = overlap2 - overlap1
                if not (2 <= spacing <= 14 and overlap >= 14):
                    continue
                if orientation == "horizontal":
                    box = Box(overlap1, min(axis1, axis2) - 2, overlap, spacing + 5)
                else:
                    box = Box(min(axis1, axis2) - 2, overlap1, spacing + 5, overlap)
                clipped = clip_box(box, width, height, padding=4)
                region = near_wall[clipped.y : clipped.y2, clipped.x : clipped.x2]
                if region.size and np.count_nonzero(region) / region.size >= 0.08:
                    candidates.append(Opening(box, orientation, 0.72))

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
        windows.append(Opening(box, orientation, 0.72))
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


def infer_swing_direction(start_deg: float, run_fraction: float) -> str:
    middle = (start_deg + run_fraction * 180.0) % 360.0
    vertical = "down" if 0.0 < middle < 180.0 else "up"
    horizontal = "right" if middle < 90.0 or middle > 270.0 else "left"
    return f"{vertical}-{horizontal}"


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
        box = clip_box(Box(cx - radius, cy - radius, radius * 2, radius * 2), width, height)
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
    return door_mask, doors


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
) -> dict[str, Any]:
    image = read_image(input_path)
    normalized, skew_angle = deskew(image) if apply_deskew else (image.copy(), 0.0)
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    ink = make_ink_mask(gray)

    wall_mask, wall_segments, plan_box = detect_walls(ink)
    thin_ink = ink.copy()
    thin_ink[cv2.dilate(wall_mask, np.ones((3, 3), np.uint8)) > 0] = 0

    window_mask, windows = detect_windows(thin_ink, wall_mask, plan_box)
    door_mask, doors = detect_doors(gray, wall_mask, plan_box, window_mask)

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
        "--no-deskew", action="store_true", help="禁用自动倾斜校正"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = process_floor_plan(
        args.input,
        args.output_dir,
        ocr_lang=args.ocr_lang,
        apply_deskew=not args.no_deskew,
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
