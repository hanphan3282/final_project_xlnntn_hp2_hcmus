#!/usr/bin/env python3
"""Static checks that never call APIs or execute the data pipeline."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "pixi.toml",
    ROOT / "pixi.lock",
    ROOT / "requirements.txt",
    ROOT / "main.nf",
    ROOT / "nextflow.config",
    ROOT / "workflow" / "run_stage.py",
    *(ROOT / "scripts" / name for name in (
        "1_fetch_images.py",
        "2_ocr_gemini.py",
        "3_compare_gemini_label.py",
        "4_paddlev6_relabel_diff.py",
        "5_llm_adjudicate_ground_truth.py",
        "split_adjudications.py",
        "6_ground_truth.py",
        "copy_ground_truth_images.py",
        "evaluate_cross_validation.py",
    )),
]


def main() -> int:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    if missing:
        print("Thiếu file bắt buộc: " + ", ".join(missing), file=sys.stderr)
        return 2

    python_files = list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "workflow").rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    print(f"OK: {len(REQUIRED)} file bắt buộc; {len(python_files)} Python file hợp lệ cú pháp.")
    print("Không có API hoặc stage dữ liệu nào được chạy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
