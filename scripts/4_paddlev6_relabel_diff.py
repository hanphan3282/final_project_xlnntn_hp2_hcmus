#!/usr/bin/env python3
"""Dùng PP-OCRv6 tạo nhãn đề xuất cho nhóm Gemini khác caption.

Script không ghi đè caption và không tự tuyên bố ground truth. Kết quả Paddle
được lưu riêng để bước DeepSeek/người đánh giá chọn nguồn phù hợp sau đó.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from lib import paddle_v6_cpu as paddle_base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "output" / "Gemini_diff_Label" / "records.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "output" / "Gemini_diff_Label" / "paddle_v6"


def parse_args() -> argparse.Namespace:
    physical = paddle_base.physical_cpu_count()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=max(1, physical // 4))
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--fallback-confidence", type=float, default=0.65)
    parser.add_argument("--no-preprocess-fallback", action="store_true")
    parser.add_argument("--limit", type=int, help="Chỉ OCR N ảnh diff chưa hoàn thành")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.cpu_threads < 1:
        parser.error("workers/cpu-threads phải lớn hơn 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")
    if not 0 <= args.score_threshold <= 1 or not 0 <= args.fallback_confidence <= 1:
        parser.error("ngưỡng confidence phải nằm trong 0..1")
    return args


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"CẢNH BÁO: JSON lỗi {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            if isinstance(item, dict):
                yield item


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def proposed_text(regions: Any) -> str:
    if not isinstance(regions, list):
        return ""
    return "\n".join(
        str(region.get("text", "")).strip()
        for region in regions
        if isinstance(region, dict) and str(region.get("text", "")).strip()
    )


def build_candidate_labels(
    records: list[dict[str, Any]], raw_rows: list[dict[str, Any]], output_dir: Path,
) -> dict[str, Any]:
    by_image = {str(row.get("image", "")): row for row in raw_rows if row.get("image")}
    labels_path = output_dir / "new_labels.jsonl"
    temp = labels_path.with_suffix(".jsonl.tmp")
    written = blank = 0
    confidence_values: list[float] = []
    with temp.open("w", encoding="utf-8") as handle:
        for record in records:
            image = str(record.get("image", ""))
            result = by_image.get(image)
            if result is None:
                continue
            regions = result.get("paddle", [])
            label = proposed_text(regions)
            scores = [float(x.get("score", 0)) for x in regions if isinstance(x, dict)]
            mean_confidence = sum(scores) / len(scores) if scores else 0.0
            confidence_values.append(mean_confidence)
            blank += not bool(label)
            item = {
                "post_id": record.get("post_id", ""),
                "image": image,
                "local_path": record.get("local_path", ""),
                "original_label": record.get("label", ""),
                "gemini_text": record.get("gemini_text", ""),
                "paddle_text": label,
                "paddle_v6_text": label,
                "proposed_label": label,
                "proposal_source": "PP-OCRv6",
                "paddle_mean_confidence": round(mean_confidence, 6),
                "paddle_regions": regions,
                "selected_preprocess_variant": result.get("ocr_pipeline", {}).get("selected_variant", "original"),
                "needs_review": not bool(label) or mean_confidence < 0.65,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
    os.replace(temp, labels_path)
    try:
        new_labels_display = labels_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        new_labels_display = str(labels_path.resolve())
    summary = {
        "diff_records": len(records),
        "paddle_results": len(raw_rows),
        "candidate_labels_written": written,
        "blank_proposals": blank,
        "missing_paddle_results": len(records) - written,
        "mean_paddle_confidence": round(sum(confidence_values) / len(confidence_values), 6)
        if confidence_values else 0.0,
        "model": "PP-OCRv6_medium_det + PP-OCRv6_medium_rec",
        "new_labels": new_labels_display,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(
            f"LỖI: chưa có nhóm diff: {args.input}. "
            "Hãy chạy scripts/facebook/3_compare_gemini_label.py trước.",
            file=sys.stderr,
        )
        return 2
    records = list(read_jsonl(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = args.output_dir / "ocr_results.jsonl"
    raw_json = args.output_dir / "ocr_results.json"
    error_log = args.output_dir / "ocr_errors.jsonl"
    done = paddle_base.load_done_keys(raw_jsonl)
    tasks: list[dict[str, str]] = []
    missing = 0
    for record in records:
        image = str(record.get("image", ""))
        if not image or image in done:
            continue
        local = Path(str(record.get("local_path", "")))
        if not local.is_file():
            missing += 1
            continue
        tasks.append({
            "id": str(record.get("post_id", "")),
            "image": image,
            "label": str(record.get("label", "")),
            "local_path": str(local),
        })
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print("=== Kế hoạch PP-OCRv6 relabel ===")
    print(f"Diff records: {len(records)}; đã OCR: {len(done)}; thiếu ảnh: {missing}")
    print(f"Sẽ OCR phiên này: {len(tasks)}")
    print(f"CPU: {paddle_base.physical_cpu_count()} lõi vật lý; workers={args.workers}; threads/worker={args.cpu_threads}")
    print(f"Output: {args.output_dir}")
    if args.dry_run:
        return 0

    errors = 0
    if tasks:
        run_args = SimpleNamespace(
            workers=args.workers,
            cpu_threads=args.cpu_threads,
            model_cache=ROOT / "models" / "paddlex",
            lang="chinese_cht",
            ocr_version="PP-OCRv6",
            score_threshold=args.score_threshold,
            fallback_confidence=args.fallback_confidence,
            no_preprocess_fallback=args.no_preprocess_fallback,
            max_pending_factor=2,
            output_jsonl=raw_jsonl,
            error_log=error_log,
        )
        _, _, errors = paddle_base.run_parallel(tasks, run_args)

    paddle_base.compile_json_array(raw_jsonl, raw_json)
    paddle_base.compact_error_log(error_log, paddle_base.load_done_keys(raw_jsonl))
    raw_rows = list(read_jsonl(raw_jsonl)) if raw_jsonl.is_file() else []
    summary = build_candidate_labels(records, raw_rows, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
