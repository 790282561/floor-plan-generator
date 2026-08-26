#!/usr/bin/env python3
"""OpenCV + multimodal-style semantic fusion for floor plan evidence.

This module intentionally uses a lightweight, dependency-free fusion strategy:
- OpenCV supplies geometric evidence (edges, contours, Hough lines, masks)
- available OCR/semantic cues are treated as multimodal hints
- the result is a structured set of fused candidates that can be consumed by
  later CAD generation without bypassing the existing script chain.

It does not replace the project requirement to still run
`processing.py -> generate_by_pic.py -> generate_doors.py -> generate_windows.py`.
It only enriches the evidence layer for ambiguous cases such as the complete
building outline, balcony top corners, room labels and wall discontinuities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    return value


def load_preprocessing_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"预处理结果不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_masks(base_dir: Path) -> dict[str, np.ndarray]:
    masks_dir = base_dir / "masks"
    return {
        "walls": _read_mask(masks_dir / "walls.png"),
        "doors": _read_mask(masks_dir / "doors.png"),
        "windows": _read_mask(masks_dir / "windows.png"),
        "dimensions": _read_mask(masks_dir / "dimensions.png"),
        "room_labels": _read_mask(masks_dir / "room_labels.png"),
        "other": _read_mask(masks_dir / "other.png"),
    }


def _read_mask(path: Path) -> np.ndarray:
    image = read_image(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return mask


def _clip_bbox(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = max(0, min(width - 1, int(x)))
    y1 = max(0, min(height - 1, int(y)))
    x2 = max(x1 + 1, min(width, int(x + w)))
    y2 = max(y1 + 1, min(height, int(y + h)))
    return x1, y1, x2 - x1, y2 - y1


def _report_bbox(item: dict[str, Any]) -> tuple[int, int, int, int] | None:
    value = item.get("bbox") or item.get("bbox_px")
    if isinstance(value, dict):
        try:
            return (
                int(round(float(value.get("x", 0)))),
                int(round(float(value.get("y", 0)))),
                int(round(float(value.get("width", 0)))),
                int(round(float(value.get("height", 0)))),
            )
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            return tuple(int(round(float(part))) for part in value)
        except (TypeError, ValueError):
            return None
    return None


def _plan_bbox(report: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    value = report.get("plan_bbox") or {}
    if isinstance(value, dict):
        box = (
            int(value.get("x", 0)),
            int(value.get("y", 0)),
            int(value.get("width", width)),
            int(value.get("height", height)),
        )
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        box = tuple(int(part) for part in value)
    else:
        box = (0, 0, width, height)
    return _clip_bbox(box, width, height)


def _bridge_opening_on_wall(
    support: np.ndarray,
    wall_mask: np.ndarray,
    box: tuple[int, int, int, int],
    orientation: str,
) -> int:
    """Bridge the wall faces across a known door/window only for outline closure.

    The bridge is written to a temporary outline-support mask.  It never
    modifies ``masks/walls.png`` and therefore cannot become CAD wall geometry.
    """
    height, width = support.shape
    x, y, w, h = _clip_bbox(box, width, height)
    axis_padding = max(5, int(round(max(w, h) * 0.08)))
    cross_padding = max(12, int(round(min(width, height) * 0.018)))
    bridges = 0

    if orientation == "horizontal":
        y1 = max(0, y - cross_padding)
        y2 = min(height, y + h + cross_padding)
        left = wall_mask[y1:y2, max(0, x - axis_padding) : x + 1]
        right = wall_mask[y1:y2, x + w - 1 : min(width, x + w + axis_padding)]
        row_scores = np.count_nonzero(left, axis=1) + np.count_nonzero(right, axis=1)
        if row_scores.size and int(row_scores.max()) > 0:
            threshold = max(2, int(round(float(row_scores.max()) * 0.45)))
            rows = np.where(row_scores >= threshold)[0] + y1
            for row in rows:
                cv2.line(support, (x, int(row)), (x + w - 1, int(row)), 255, 1)
                bridges += 1
    elif orientation == "vertical":
        x1 = max(0, x - cross_padding)
        x2 = min(width, x + w + cross_padding)
        top = wall_mask[max(0, y - axis_padding) : y + 1, x1:x2]
        bottom = wall_mask[y + h - 1 : min(height, y + h + axis_padding), x1:x2]
        column_scores = np.count_nonzero(top, axis=0) + np.count_nonzero(bottom, axis=0)
        if column_scores.size and int(column_scores.max()) > 0:
            threshold = max(2, int(round(float(column_scores.max()) * 0.45)))
            columns = np.where(column_scores >= threshold)[0] + x1
            for column in columns:
                cv2.line(support, (int(column), y), (int(column), y + h - 1), 255, 1)
                bridges += 1
    return bridges


def detect_building_outline(
    gray: np.ndarray,
    masks: dict[str, np.ndarray],
    report: dict[str, Any],
) -> tuple[dict[str, Any] | None, np.ndarray, np.ndarray]:
    """Detect one closed outer footprint before wall/door subtraction.

    Wall, door and window masks are combined only as outline evidence.  Known
    opening bboxes are bridged on their wall faces so exterior doors/windows do
    not make the footprint leak into the page background.  The frozen wall mask
    itself is never changed.
    """
    height, width = gray.shape
    plan_x, plan_y, plan_w, plan_h = _plan_bbox(report, width, height)
    plan_x2, plan_y2 = plan_x + plan_w, plan_y + plan_h
    plan_roi = np.zeros_like(gray)
    plan_roi[plan_y:plan_y2, plan_x:plan_x2] = 255

    # The complete input image is essential here: a still-unclassified window
    # can be absent from windows.png while its thin frame lines already close
    # the true facade (for example a tall balcony-side window).  These source
    # strokes are outline evidence only and are never promoted to wall pixels.
    source_ink = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
    source_ink = cv2.bitwise_and(source_ink, plan_roi)
    support = cv2.bitwise_or(masks["walls"], masks["doors"])
    support = cv2.bitwise_or(support, masks["windows"])
    support = cv2.bitwise_or(support, source_ink)
    support = cv2.bitwise_and(support, plan_roi)

    bridge_count = 0
    for collection, orientation_key in (("doors", "wall_orientation"), ("windows", "orientation")):
        for item in report.get(collection, []):
            box = _report_bbox(item)
            orientation = item.get(orientation_key)
            if box is None or orientation not in {"horizontal", "vertical"}:
                continue
            bridge_count += _bridge_opening_on_wall(support, masks["walls"], box, orientation)

    # Join pixel-scale breaks left by antialiasing while keeping the actual
    # concave footprint; this is deliberately much smaller than door/window
    # spans, which were bridged explicitly above.
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Treat the combined geometry as a temporary boundary barrier.  Flooding
    # from the page background yields the true exterior; everything the flood
    # cannot reach is the building footprint.  This prevents internal walls
    # and door arcs from becoming notches in the outer contour.
    free_space = cv2.bitwise_not(support)
    flooded = free_space.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 128)
    footprint_seed = np.where(flooded == 128, 0, 255).astype(np.uint8)
    footprint_seed = cv2.bitwise_and(footprint_seed, plan_roi)
    contours, _ = cv2.findContours(footprint_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline_mask = np.zeros_like(gray)
    footprint_mask = np.zeros_like(gray)
    if not contours:
        return None, outline_mask, footprint_mask

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    minimum_area = max(100.0, plan_w * plan_h * 0.10)
    if contour_area < minimum_area:
        return None, outline_mask, footprint_mask

    perimeter = float(cv2.arcLength(contour, True))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    concavity_ratio = max(0.0, (hull_area - contour_area) / max(hull_area, 1.0))
    epsilon = max(1.0, perimeter * 0.0015)
    polygon = cv2.approxPolyDP(contour, epsilon, True)
    points = [[int(point[0][0]), int(point[0][1])] for point in polygon]
    x, y, w, h = cv2.boundingRect(contour)
    cv2.drawContours(outline_mask, [contour], -1, 255, 2)
    cv2.drawContours(footprint_mask, [contour], -1, 255, cv2.FILLED)

    bbox_coverage = (w * h) / max(plan_w * plan_h, 1)
    area_ratio = contour_area / max(plan_w * plan_h, 1)
    confidence = min(0.98, 0.72 + min(0.14, bbox_coverage * 0.14) + min(0.12, area_ratio * 0.16))
    candidate = {
        "id": "BUILDING_OUTLINE_01",
        "class": "building_outline",
        "type": "building_outline",
        "bbox": [int(x), int(y), int(w), int(h)],
        "bbox_px": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "polygon_px": points,
        "closed": True,
        "raw_point_count": int(len(contour)),
        "polygon_point_count": int(len(points)),
        "area_px2": round(contour_area, 3),
        "perimeter_px": round(perimeter, 3),
        "plan_bbox_coverage": round(float(bbox_coverage), 4),
        "footprint_area_ratio": round(float(area_ratio), 4),
        "convex_hull_area_px2": round(hull_area, 3),
        "concavity_ratio": round(float(concavity_ratio), 4),
        "is_concave": bool(concavity_ratio >= 0.005),
        "opening_bridge_line_count": int(bridge_count),
        "score": round(float(confidence), 3),
        "confidence": round(float(confidence), 3),
        "status": "DETECTED" if confidence >= 0.85 else "INFERRED",
        "source": "input_image_plus_preprocessed_wall_door_window_masks",
        "evidence": {
            "input_image_ink_pixels": int(np.count_nonzero(source_ink)),
            "wall_mask_pixels": int(np.count_nonzero(masks["walls"])),
            "door_mask_pixels": int(np.count_nonzero(masks["doors"])),
            "window_mask_pixels": int(np.count_nonzero(masks["windows"])),
            "wall_mask_modified": False,
        },
        "reason": "在整张户型图中融合墙体、门、窗边界并闭合已知洞口后提取最大建筑外轮廓",
    }
    return candidate, outline_mask, footprint_mask


def detect_balcony_top_corner_candidates(gray: np.ndarray, wall_mask: np.ndarray) -> list[dict[str, Any]]:
    """Look for balcony-like top corners along upper wall boundaries.

    This is a lightweight semantic anchor rather than a hard geometry rule. The
    geometry remains enforced by `generate_by_pic.py` and the wall topology pass;
    this function only provides evidence for ambiguous features, such as the top
    corner of an enclosed balcony.
    """
    height, width = gray.shape
    top_band = wall_mask[0 : max(60, height // 5), :]
    if top_band.size == 0:
        return []

    hull = cv2.morphologyEx(top_band, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(hull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        if len(contour) < 6:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 200:
            continue
        if h < 18 or w < 18:
            continue
        # Balcony-like candidates usually form a short upper wall segment with a
        # near-vertical continuation and a relatively compact bounding box.
        if not (y < height * 0.45 and x > 0 and x + w < width):
            continue

        corner_score = min(0.95, 0.45 + min(0.35, h / max(80, height)) + min(0.15, w / max(120, width)))
        candidates.append(
            {
                "type": "balcony_top_corner",
                "bbox": [int(x), int(y), int(w), int(h)],
                "score": round(float(corner_score), 3),
                "source": "opencv_wall_top_band",
                "reason": "上部墙体轮廓与紧邻转角特征匹配阳台封顶候选",
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def detect_room_text_candidates(gray: np.ndarray, room_mask: np.ndarray) -> list[dict[str, Any]]:
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    combined = cv2.bitwise_and(mask, room_mask)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 25 or w < 8 or h < 8:
            continue
        # Keep room-label-like elements but mark them as semantic candidates.
        candidates.append(
            {
                "type": "room_text_anchor",
                "bbox": [int(x), int(y), int(w), int(h)],
                "score": round(min(0.9, 0.35 + min(0.55, area / max(1000, gray.size * 0.0005))), 3),
                "source": "opencv_room_mask",
                "reason": "房间标签遮罩中的文字/编号候选",
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def detect_window_candidates(gray: np.ndarray, wall_mask: np.ndarray, window_mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = gray.shape
    roi = np.zeros_like(gray)
    roi[0:height, 0:width] = 255
    # Prefer already-detected window regions as anchors; still add a geometric
    # fallback that looks for compact openings inside wall bands.
    window_contours, _ = cv2.findContours(window_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for contour in window_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 10 or h < 10:
            continue
        candidates.append(
            {
                "type": "window_anchor",
                "bbox": [int(x), int(y), int(w), int(h)],
                "score": 0.82,
                "source": "opencv_window_mask",
                "reason": "墙体切口与已识别窗体掩码一致",
            }
        )

    if candidates:
        return candidates

    edges = cv2.Canny(gray, 50, 150)
    dense = cv2.bitwise_and(edges, wall_mask)
    contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 12 or max(w, h) < 20:
            continue
        candidates.append(
            {
                "type": "window_anchor",
                "bbox": [int(x), int(y), int(w), int(h)],
                "score": 0.58,
                "source": "opencv_edge_fallback",
                "reason": "边缘洞口候选，需由后续门窗脚本进行拓扑确认",
            }
        )
    return candidates[:10]


def write_building_outline_artifacts(
    image: np.ndarray,
    outline_mask: np.ndarray,
    footprint_mask: np.ndarray,
    candidate: dict[str, Any] | None,
    preprocessed_dir: Path,
) -> dict[str, str | None]:
    masks_dir = preprocessed_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    outline_path = masks_dir / "building_outline.png"
    footprint_path = masks_dir / "building_footprint.png"
    overlay_path = preprocessed_dir / "building_outline_overlay.png"

    for path, value in ((outline_path, outline_mask), (footprint_path, footprint_mask)):
        ok, encoded = cv2.imencode(".png", value)
        if not ok:
            raise OSError(f"无法编码轮廓诊断图：{path}")
        encoded.tofile(str(path))

    overlay = image.copy()
    if candidate is not None:
        polygon = np.asarray(candidate.get("polygon_px", []), dtype=np.int32)
        if len(polygon) >= 3:
            cv2.polylines(overlay, [polygon.reshape((-1, 1, 2))], True, (0, 0, 255), 3)
        x, y, w, h = candidate["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 255), 1)
        cv2.putText(
            overlay,
            f"building outline {candidate['confidence']:.3f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise OSError(f"无法编码轮廓叠加图：{overlay_path}")
    encoded.tofile(str(overlay_path))
    return {
        "outline_mask": str(outline_path.resolve()),
        "footprint_mask": str(footprint_path.resolve()),
        "overlay": str(overlay_path.resolve()),
    }


def fuse_semantic_candidates(report: dict[str, Any], image_path: Path, base_dir: Path) -> dict[str, Any]:
    image = read_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = read_masks(base_dir)

    building_outline, outline_mask, footprint_mask = detect_building_outline(gray, masks, report)
    balcony_candidates = detect_balcony_top_corner_candidates(gray, masks["walls"])
    room_text_candidates = detect_room_text_candidates(gray, masks["room_labels"])
    window_candidates = detect_window_candidates(gray, masks["walls"], masks["windows"])

    fused: list[dict[str, Any]] = []
    if building_outline is not None:
        fused.append(building_outline)
    fused.extend({**item, "confidence": item["score"], "status": "candidate"} for item in balcony_candidates)
    fused.extend({**item, "confidence": item["score"], "status": "candidate"} for item in room_text_candidates)
    fused.extend({**item, "confidence": item["score"], "status": "candidate"} for item in window_candidates)

    # Collapse near-duplicate candidates using a simple IoU threshold.
    deduped: list[dict[str, Any]] = []
    for candidate in sorted(fused, key=lambda item: item["confidence"], reverse=True):
        x, y, w, h = candidate["bbox"]
        box_a = (x, y, x + w, y + h)
        duplicate = False
        for existing in deduped:
            ex, ey, ew, eh = existing["bbox"]
            box_b = (ex, ey, ex + ew, ey + eh)
            inter_x1 = max(box_a[0], box_b[0])
            inter_y1 = max(box_a[1], box_b[1])
            inter_x2 = min(box_a[2], box_b[2])
            inter_y2 = min(box_a[3], box_b[3])
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            union_area = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - inter_area)
            iou = inter_area / union_area
            if iou > 0.35 and existing["type"] == candidate["type"]:
                duplicate = True
                break
        if not duplicate:
            deduped.append(candidate)

    result = {
        "schema_version": 3,
        "source_image": str(image_path.resolve()),
        "preprocessing_result": str((base_dir / "result.json").resolve()),
        "vision_backend": "opencv_anchor_fusion",
        "multimodal_enabled": True,
        "processing_order": ["building_outline", "wall", "door", "window"],
        "building_outline": building_outline,
        "building_outline_artifacts": write_building_outline_artifacts(
            image, outline_mask, footprint_mask, building_outline, base_dir
        ),
        "semantic_candidates": deduped,
        "notes": [
            "先识别整张输入图的建筑外轮廓，再按墙体、门、窗顺序处理后续候选。",
            "building_outline 只作为外轮廓与后续缺口分析证据，不修改 masks/walls.png，也不直接生成窗图元。",
            "该融合层不替代 generate_by_pic.py / generate_doors.py / generate_windows.py 的最终 CAD 生成。",
            "阳台顶部转角、外轮廓缺口、窗洞和房间文字仍必须经过拓扑校验和阶段验收。",
        ],
    }
    # Preserve the original project report as a base and append the new evidence.
    result.update({
        "original_report": _safe_json_value(report),
    })
    return result


def generate_multimodal_fusion(image_path: Path, preprocessed_dir: Path, report: dict[str, Any] | None = None) -> dict[str, Any]:
    if report is None:
        report = load_preprocessing_json(preprocessed_dir / "result.json")
    return fuse_semantic_candidates(report, image_path, preprocessed_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 OpenCV 几何证据与语义候选整合为 multimodal 结果。"
    )
    parser.add_argument("input", type=Path, help="原始户型图片")
    parser.add_argument(
        "--preprocessed-dir",
        "-d",
        type=Path,
        required=True,
        help="processing.py 输出目录（应包含 result.json 与 masks/）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出 JSON 文件；默认写到 <preprocessed-dir>/multimodal.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = load_preprocessing_json(args.preprocessed_dir / "result.json")
    fusion = generate_multimodal_fusion(args.input, args.preprocessed_dir, report)
    output_path = args.output or (args.preprocessed_dir / "multimodal.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fusion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path.resolve()),
        "building_outline_detected": fusion.get("building_outline") is not None,
        "building_outline_confidence": (
            fusion["building_outline"].get("confidence")
            if fusion.get("building_outline") is not None
            else None
        ),
        "semantic_candidates": len(fusion.get("semantic_candidates", [])),
        "vision_backend": fusion.get("vision_backend"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
