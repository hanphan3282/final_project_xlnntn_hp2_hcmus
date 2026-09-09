#!/usr/bin/env python3
"""Đối chứng PaddleOCR giữa ảnh gốc và nhiều biến thể tiền xử lý.

Ảnh gốc luôn được đọc từ file baseline đã OCR trước đó và tuyệt đối không bị
thay đổi. Script chọn mẫu phân tầng (blank/low/high), tạo các bản sao tiền xử
lý, OCR các bản sao, rồi xuất báo cáo TSV/JSON để quyết định có nên tiền xử lý.

Ví dụ:
    python scripts/facebook/experiments/compare_paddle_preprocessing.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

FACEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FACEBOOK_DIR))
from lib import paddle_v6_cpu as paddle_base


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE = ROOT / "data" / "output" / "PaddleV6_full" / "ocr_results.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "experiments" / "paddle_preprocessing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--images-dir", type=Path, default=ROOT / "data" / "mrDuc_data" / "Images")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-group", type=int, default=10,
                        help="Số ảnh mỗi nhóm blank/low/high (mặc định: 10)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=13)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--model-cache", type=Path, default=ROOT / "models" / "paddlex")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metrics(row: dict[str, Any]) -> dict[str, float | int]:
    regions = row.get("paddle", [])
    scores = [float(x.get("score", 0)) for x in regions]
    texts = [str(x.get("text", "")) for x in regions]
    return {
        "regions": len(regions),
        "chars": sum(len(x.replace(" ", "")) for x in texts),
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
    }


def evenly_spaced(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda x: str(x.get("image", "")))
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[int(i)] for i in indices]


def select_sample(rows: list[dict[str, Any]], per_group: int) -> list[tuple[str, dict[str, Any]]]:
    blank, low, high = [], [], []
    for row in rows:
        stat = metrics(row)
        if stat["regions"] == 0:
            blank.append(row)
        elif stat["mean_score"] < 0.50:
            low.append(row)
        # Tránh ngoại lệ trang cực dày (>80 vùng) làm bốn biến thể chiếm CPU
        # hàng chục phút; nhóm này cần một benchmark riêng theo kích thước trang.
        elif stat["mean_score"] >= 0.80 and stat["regions"] <= 80:
            high.append(row)
    selected: list[tuple[str, dict[str, Any]]] = []
    for group, pool in (("blank", blank), ("low", low), ("high", high)):
        selected.extend((group, row) for row in evenly_spaced(pool, per_group))
    return selected


def locate_image(images_dir: Path, image_key: str) -> Path:
    name = Path(image_key).name
    exact = images_dir / name
    if exact.is_file():
        return exact
    stem = Path(name).stem
    candidates = sorted(images_dir.glob(stem + ".*"))
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy ảnh local cho {image_key}")
    return candidates[0]


def crop_uniform_margin(image: np.ndarray) -> np.ndarray:
    """Chỉ cắt khi viền tương đối đồng nhất; ảnh phức tạp được giữ nguyên."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    strip = max(2, min(h, w) // 50)
    border = np.concatenate((gray[:strip, :].ravel(), gray[-strip:, :].ravel(),
                             gray[:, :strip].ravel(), gray[:, -strip:].ravel()))
    if float(np.std(border)) > 28:
        return image.copy()
    background = float(np.median(border))
    difference = cv2.absdiff(gray, np.full_like(gray, round(background)))
    mask = (difference > max(14, float(np.std(border)) * 2.5)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    points = cv2.findNonZero(mask)
    if points is None:
        return image.copy()
    x, y, bw, bh = cv2.boundingRect(points)
    # Không cắt thành một mảnh quá nhỏ hoặc khi lợi ích không đáng kể.
    if bw * bh < 0.20 * w * h or bw * bh > 0.95 * w * h:
        return image.copy()
    pad_x, pad_y = max(8, round(0.02 * w)), max(8, round(0.02 * h))
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
    return image[y0:y1, x0:x1].copy()


def variants(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped = crop_uniform_margin(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    normalized = gray if float(np.median(gray)) >= 128 else cv2.bitwise_not(gray)
    adaptive = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, 15,
    )
    return {
        "crop": cropped,
        "clahe": clahe,
        "adaptive": adaptive,
        "invert": cv2.bitwise_not(image),
    }


def prepare_tasks(
    selected: list[tuple[str, dict[str, Any]]], images_dir: Path, output_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    tasks: list[dict[str, str]] = []
    manifest: dict[str, dict[str, Any]] = {}
    processed_dir = output_dir / "processed"
    for group, baseline in selected:
        source = locate_image(images_dir, str(baseline["image"]))
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV không đọc được {source}")
        stem_dir = processed_dir / source.stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        manifest[str(baseline["image"])] = {"group": group, "baseline": baseline}
        for variant_name, transformed in variants(image).items():
            target = stem_dir / f"{variant_name}.png"
            if not target.is_file():
                if not cv2.imwrite(str(target), transformed):
                    raise OSError(f"Không ghi được {target}")
            key = f"{baseline['image']}::{variant_name}"
            tasks.append({
                "id": str(baseline.get("id", "")),
                "image": key,
                "label": str(baseline.get("label", "")),
                "local_path": str(target),
            })
    return tasks, manifest


def write_reports(output_dir: Path, manifest: dict[str, dict[str, Any]], variants_rows: list[dict[str, Any]]) -> None:
    by_key = {str(row["image"]): row for row in variants_rows}
    detail_rows: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, float]] = {}

    for image_key, item in manifest.items():
        original = metrics(item["baseline"])
        for variant_name in ("original", "crop", "clahe", "adaptive", "invert"):
            row = item["baseline"] if variant_name == "original" else by_key.get(f"{image_key}::{variant_name}", {})
            stat = metrics(row)
            detail_rows.append({
                "image": image_key,
                "group": item["group"],
                "variant": variant_name,
                "regions": stat["regions"],
                "chars": stat["chars"],
                "mean_score": round(float(stat["mean_score"]), 6),
                "delta_regions": int(stat["regions"]) - int(original["regions"]),
                "delta_chars": int(stat["chars"]) - int(original["chars"]),
                "delta_mean_score": round(float(stat["mean_score"]) - float(original["mean_score"]), 6),
                "text": " | ".join(str(x.get("text", "")) for x in row.get("paddle", [])),
            })

            key = f"{item['group']}::{variant_name}"
            bucket = aggregate.setdefault(key, {"images": 0, "regions": 0, "chars": 0,
                                                "score_sum": 0, "blank_recovered": 0})
            bucket["images"] += 1
            bucket["regions"] += int(stat["regions"])
            bucket["chars"] += int(stat["chars"])
            bucket["score_sum"] += float(stat["mean_score"])
            if item["group"] == "blank" and int(stat["regions"]) > 0:
                bucket["blank_recovered"] += 1

    fields = ["image", "group", "variant", "regions", "chars", "mean_score",
              "delta_regions", "delta_chars", "delta_mean_score", "text"]
    with (output_dir / "comparison.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(detail_rows)

    summary = []
    for key, values in sorted(aggregate.items()):
        group, variant_name = key.split("::", 1)
        n = int(values["images"])
        summary.append({
            "group": group,
            "variant": variant_name,
            "images": n,
            "blank_recovered": int(values["blank_recovered"]),
            "mean_regions": round(values["regions"] / n, 3),
            "mean_chars": round(values["chars"] / n, 3),
            "mean_confidence": round(values["score_sum"] / n, 6),
        })
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if not args.baseline.is_file():
        raise SystemExit(f"Không tìm thấy baseline: {args.baseline}")
    baseline_rows = read_jsonl(args.baseline)
    selected = select_sample(baseline_rows, args.per_group)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks, manifest = prepare_tasks(selected, args.images_dir, args.output_dir)
    manifest_path = args.output_dir / "sample_manifest.json"
    manifest_path.write_text(json.dumps([
        {"image": key, "group": value["group"], **metrics(value["baseline"])}
        for key, value in manifest.items()
    ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_jsonl = args.output_dir / "variants_ocr.jsonl"
    done = paddle_base.load_done_keys(output_jsonl)
    pending = [task for task in tasks if task["image"] not in done]
    print(f"Baseline: {len(baseline_rows)} ảnh; mẫu: {len(selected)} ảnh; biến thể: {len(tasks)}")
    print(f"Đã OCR biến thể: {len(done)}; còn lại: {len(pending)}")
    if args.prepare_only:
        return 0

    run_args = SimpleNamespace(
        workers=args.workers,
        cpu_threads=args.cpu_threads,
        model_cache=args.model_cache,
        lang="chinese_cht",
        ocr_version="PP-OCRv6",
        score_threshold=args.score_threshold,
        max_pending_factor=2,
        output_jsonl=output_jsonl,
        error_log=args.output_dir / "errors.jsonl",
    )
    if pending:
        paddle_base.run_parallel(pending, run_args)
    variant_rows = read_jsonl(output_jsonl) if output_jsonl.is_file() else []
    write_reports(args.output_dir, manifest, variant_rows)
    print(f"Báo cáo: {args.output_dir / 'comparison.tsv'}")
    print(f"Tổng hợp: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
