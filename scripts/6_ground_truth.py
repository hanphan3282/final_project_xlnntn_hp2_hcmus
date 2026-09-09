#!/usr/bin/env python3
"""Tổng hợp ground truth từ adjudications_valid.jsonl.

Đọc kết quả adjudication đã lọc (confidence >= ngưỡng) và tạo file ground truth
theo định dạng thống nhất:
  - image       : đường dẫn ảnh
  - ground_truth: văn bản ground truth do DeepSeek chọn
  - label       : văn bản OCR từ Paddle V6
  - deepseek    : [{"text": <ground_truth_text>}]  — cùng cấu trúc với gemini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "output" / "DeepSeek_ground_truth" / "adjudications_valid.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "DeepSeek_ground_truth" / "ground_truth.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="File adjudications_valid.jsonl đầu vào")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="File ground_truth.jsonl đầu ra")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Lọc thêm theo confidence (mặc định: 0.0 = không lọc thêm)")
    parser.add_argument(
        "--provider", choices=("deepseek", "qwen"), default="deepseek",
        help="Tên provider dùng làm field kết quả (mặc định: deepseek)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.is_file():
        print(f"LỖI: không tìm thấy {args.input}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = written = skipped = 0

    with args.input.open(encoding="utf-8") as fin, \
         args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)

            confidence = row.get("decision", {}).get("confidence", 0)
            if confidence < args.min_confidence:
                skipped += 1
                continue

            ground_truth_text = row.get("decision", {}).get("ground_truth_text", "")
            paddle_text = row.get("sources", {}).get("paddle_v6", "")
            gemini_text = row.get("sources", {}).get("gemini", "")

            out = {
                "image": row.get("image", ""),
                "ground_truth": ground_truth_text,
                "label": paddle_text,
                "gemini": [{"text": gemini_text}] if gemini_text else [],
                args.provider: [{"text": ground_truth_text}] if ground_truth_text else [],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    print(f"Đọc   : {total} records")
    print(f"Bỏ qua: {skipped} (confidence < {args.min_confidence})")
    print(f"Ghi   : {written} records → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
