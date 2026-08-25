#!/usr/bin/env python3
"""OpenCV + multimodal-style semantic fusion for floor plan evidence.

This module intentionally uses a lightweight, dependency-free fusion strategy:
- OpenCV supplies geometric evidence (edges, contours, Hough lines, masks)
- available OCR/semantic cues are treated as multimodal hints
- the result is a structured set of fused candidates that can be consumed by
  later CAD generation without bypassing the existing script chain.

It does not replace the project requirement to still run
`processing.py -> generate_by_pic.py -> generate_windows.py -> generate_doors.py`.
It only enriches the evidence layer for ambiguous cases such as balcony top
corners, room labels and wall discontinuities.
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


def fuse_semantic_candidates(report: dict[str, Any], image_path: Path, base_dir: Path) -> dict[str, Any]:
    image = read_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = read_masks(base_dir)

    balcony_candidates = detect_balcony_top_corner_candidates(gray, masks["walls"])
    room_text_candidates = detect_room_text_candidates(gray, masks["room_labels"])
    window_candidates = detect_window_candidates(gray, masks["walls"], masks["windows"])

    fused: list[dict[str, Any]] = []
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
        "schema_version": 2,
        "source_image": str(image_path.resolve()),
        "preprocessing_result": str((base_dir / "result.json").resolve()),
        "vision_backend": "opencv_anchor_fusion",
        "multimodal_enabled": True,
        "semantic_candidates": deduped,
        "notes": [
            "该融合层仅作为语义候选和几何锚点，不直接替代 generate_by_pic.py / generate_windows.py / generate_doors.py 的最终 CAD 生成。",
            "阳台顶部转角、窗洞和房间文字仍必须经过拓扑校验和阶段验收。",
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
        "semantic_candidates": len(fusion.get("semantic_candidates", [])),
        "vision_backend": fusion.get("vision_backend"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
