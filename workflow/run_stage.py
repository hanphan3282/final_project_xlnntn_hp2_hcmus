#!/usr/bin/env python3
"""Adopt a valid manual result or execute exactly one pipeline stage."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


STAGE_NAMES = {
    1: "fetch_images",
    2: "gemini_ocr",
    3: "compare_labels",
    4: "paddle_ocr",
    5: "llm_adjudication",
    6: "split_adjudications",
    7: "build_ground_truth",
    8: "copy_ground_truth_images",
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Giá trị boolean không hợp lệ: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("deepseek", "qwen"), default="deepseek")
    parser.add_argument("--reuse-existing", type=parse_bool, default=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("Thiếu command sau --")
    return args


def valid_json(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            json.load(handle)
        return True
    except (OSError, json.JSONDecodeError):
        return False


def valid_jsonl(path: Path, allow_empty: bool = False) -> bool:
    if not path.is_file():
        return False
    rows = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return False
                rows += 1
    except (OSError, json.JSONDecodeError):
        return False
    return allow_empty or rows > 0


def output_dir(data: Path, provider: str) -> Path:
    name = "DeepSeek_ground_truth" if provider == "deepseek" else "Qwen38_ground_truth"
    return data / "output" / name


def stage_checks(stage: int, data: Path, provider: str) -> list[tuple[str, Callable[[], bool]]]:
    ocr = data / "output" / "mrDuc_data_ocr"
    same = data / "output" / "Gemini_same_Label"
    diff = data / "output" / "Gemini_diff_Label"
    paddle = diff / "paddle_v6"
    llm = output_dir(data, provider)

    checks: dict[int, list[tuple[str, Callable[[], bool]]]] = {
        1: [
            (
                str(data / "input" / "Images"),
                lambda: (data / "input" / "Images").is_dir()
                and any((data / "input" / "Images").iterdir()),
            ),
            (str(ocr / "fetch_summary.json"), lambda: valid_json(ocr / "fetch_summary.json")),
        ],
        2: [
            (str(ocr / "facebook_posts_ocr.jsonl"), lambda: valid_jsonl(ocr / "facebook_posts_ocr.jsonl")),
            (str(ocr / "ocr_summary.json"), lambda: valid_json(ocr / "ocr_summary.json")),
        ],
        3: [
            (str(same / "records.jsonl"), lambda: valid_jsonl(same / "records.jsonl", allow_empty=True)),
            (str(diff / "records.jsonl"), lambda: valid_jsonl(diff / "records.jsonl", allow_empty=True)),
            (str(same / "summary.json"), lambda: valid_json(same / "summary.json")),
            (str(diff / "summary.json"), lambda: valid_json(diff / "summary.json")),
        ],
        4: [
            (str(paddle / "new_labels.jsonl"), lambda: valid_jsonl(paddle / "new_labels.jsonl", allow_empty=True)),
            (str(paddle / "summary.json"), lambda: valid_json(paddle / "summary.json")),
        ],
        5: [
            (str(llm / "adjudications.jsonl"), lambda: valid_jsonl(llm / "adjudications.jsonl", allow_empty=True)),
            (str(llm / "summary.json"), lambda: valid_json(llm / "summary.json")),
        ],
        6: [
            (str(llm / "adjudications_valid.jsonl"), lambda: valid_jsonl(llm / "adjudications_valid.jsonl", allow_empty=True)),
            (str(llm / "adjudications_invalid.jsonl"), lambda: valid_jsonl(llm / "adjudications_invalid.jsonl", allow_empty=True)),
        ],
        7: [
            (str(llm / "ground_truth.jsonl"), lambda: valid_jsonl(llm / "ground_truth.jsonl", allow_empty=True)),
        ],
        8: [
            (str(data / "ground_truth_images"), lambda: ground_truth_images_complete(llm / "ground_truth.jsonl", data / "ground_truth_images")),
        ],
    }
    return checks[stage]


def ground_truth_images_complete(ground_truth: Path, images_dir: Path) -> bool:
    if not images_dir.is_dir() or not valid_jsonl(ground_truth, allow_empty=True):
        return False
    try:
        with ground_truth.open(encoding="utf-8") as handle:
            names = {
                Path(str(json.loads(line).get("image", ""))).name
                for line in handle
                if line.strip()
            }
    except (OSError, json.JSONDecodeError):
        return False
    names.discard("")
    return all((images_dir / name).is_file() for name in names)


def check_outputs(stage: int, data: Path, provider: str) -> tuple[bool, list[str]]:
    failed = [label for label, check in stage_checks(stage, data, provider) if not check()]
    return not failed, failed


def write_marker(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    project = args.project_dir.resolve()
    data = args.data_dir.resolve()
    complete, failed = check_outputs(args.stage, data, args.provider)
    started = time.time()

    if args.reuse_existing and complete:
        status = "SKIPPED_EXISTING"
        exit_code = 0
        print(f"[{STAGE_NAMES[args.stage]}] Output hợp lệ đã tồn tại; không chạy lại.")
    else:
        if failed:
            print(f"[{STAGE_NAMES[args.stage]}] Cần tạo/hoàn thiện: {', '.join(failed)}")
        completed = subprocess.run(args.command, cwd=project, check=False)
        exit_code = completed.returncode
        if exit_code != 0:
            print(f"Stage {args.stage} thất bại với exit code {exit_code}.", file=sys.stderr)
            return exit_code
        complete, failed = check_outputs(args.stage, data, args.provider)
        if not complete:
            print(
                f"Stage {args.stage} kết thúc nhưng output chưa hợp lệ: {', '.join(failed)}",
                file=sys.stderr,
            )
            return 3
        status = "COMPLETED"

    write_marker(
        args.marker,
        {
            "stage": args.stage,
            "name": STAGE_NAMES[args.stage],
            "status": status,
            "provider": args.provider,
            "data_dir": str(data),
            "elapsed_seconds": round(time.time() - started, 3),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
