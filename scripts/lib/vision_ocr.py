#!/usr/bin/env python3
"""OCR ảnh mrDuc bằng vision LLM qua API OpenAI-compatible.

Script đọc ``data/mrDuc_data/valid.jsonl``, ghép ảnh local theo
``Images/{post_id}_{index}.jpg`` và ghi JSONL tăng dần để hỗ trợ resume.
Đây là pipeline API/LLM, không phải PP-OCRv6 local.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter


ROOT = Path(__file__).resolve().parents[2]
_OCR_DIR = ROOT / "data" / "output" / "mrDuc_data_ocr"
DEFAULT_INPUT = ROOT / "data" / "input" / "valid.jsonl"
DEFAULT_IMAGES_DIR = ROOT / "data" / "input" / "Images"
DEFAULT_OUTPUT_JSONL = _OCR_DIR / "facebook_posts_ocr.jsonl"
DEFAULT_OUTPUT_JSON = _OCR_DIR / "facebook_posts_ocr.json"
DEFAULT_ERROR_LOG = _OCR_DIR / "ocr_errors.jsonl"
DEFAULT_SUMMARY = _OCR_DIR / "ocr_summary.json"
DEFAULT_ENV = ROOT / ".env"
THREAD_LOCAL = threading.local()
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
ENV_KEYS = {"OCR_API_KEY", "OCR_BASE_URL", "OCR_MODEL"}

PROMPT = """Bạn là hệ thống OCR chuyên chữ Hán phồn thể, giản thể, thư pháp và chữ viết tay.
Hãy chép lại chính xác văn bản thực sự nhìn thấy trong ảnh, không đoán hoặc bổ sung nội dung.
Trích xuất theo từng dòng/cụm chữ cùng bounding box chuẩn hóa 0-1000 theo thứ tự
[ymin, xmin, ymax, xmax]. Giữ đúng thứ tự đọc tự nhiên của ảnh.

Chỉ trả về một JSON object đúng schema:
{"regions": [{"text": "văn bản", "bounding_box": [ymin, xmin, ymax, xmax]}]}
Nếu ảnh không có chữ hoặc không đọc được, trả về {"regions": []}.
Không dùng markdown và không giải thích."""


@dataclass(frozen=True)
class OCRTask:
    post_id: str
    image_key: str
    local_path: str
    label: str


@dataclass
class OCRResult:
    ok: bool
    post_id: str
    image: str
    local_path: str
    attempts: int
    elapsed_seconds: float
    regions: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    upload_width: int = 0
    upload_height: int = 0
    upload_bytes: int = 0
    resized: bool = False
    http_status: int | None = None
    error: str | None = None


class OCRFailure(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--error-log", type=Path, default=DEFAULT_ERROR_LOG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--base-url", help="Ghi đè OCR_BASE_URL, không đặt API key trên CLI")
    parser.add_argument("--model", help="Ghi đè OCR_MODEL")
    parser.add_argument("--start", type=int, default=0, help="Bỏ qua N post đầu")
    parser.add_argument("--limit", type=int, help="Chỉ OCR tối đa N ảnh local chưa hoàn thành")
    parser.add_argument("--workers", type=int, default=4, help="Số request API đồng thời (mặc định: 4)")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-side", type=int, default=2048)
    parser.add_argument("--max-upload-mb", type=float, default=8.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-fail-rate", type=float, default=0.25)
    parser.add_argument("--fail-rate-min-samples", type=int, default=20)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.start < 0:
        parser.error("--start không được âm")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")
    if not 1 <= args.workers <= 32:
        parser.error("--workers phải nằm trong 1..32")
    if args.retries < 1 or args.timeout <= 0:
        parser.error("retries/timeout không hợp lệ")
    if args.max_side < 256 or args.max_upload_mb <= 0:
        parser.error("giới hạn ảnh không hợp lệ")
    if not 1 <= args.jpeg_quality <= 100 or args.max_tokens < 1:
        parser.error("jpeg-quality/max-tokens không hợp lệ")
    if not 0 < args.max_fail_rate <= 1 or args.fail_rate_min_samples < 1:
        parser.error("ngưỡng lỗi không hợp lệ")
    return args


def load_env_file(path: Path) -> None:
    """Chỉ đọc ba biến OCR_*; không thực thi nội dung .env như shell."""
    if not path.is_file():
        return
    pattern = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = pattern.match(line)
        if not match or match.group(1) not in ENV_KEYS:
            continue
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def api_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON lỗi tại {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Dòng {line_number} không phải object")
            yield line_number, value


def load_done(path: Path, model: str | None, overwrite: bool) -> set[str]:
    if overwrite or not path.is_file():
        return set()
    done: set[str] = set()
    models: set[str] = set()
    for _, item in read_jsonl(path):
        key = str(item.get("image", ""))
        if key:
            done.add(key)
        existing_model = str(item.get("ocr_model", ""))
        if existing_model:
            models.add(existing_model)
    if model and models and models != {model}:
        raise ValueError(
            f"Output đang chứa model {sorted(models)}, không thể resume bằng {model}. "
            "Hãy dùng đường dẫn output mới."
        )
    return done


def build_tasks(args: argparse.Namespace, done: set[str]) -> tuple[list[OCRTask], dict[str, int]]:
    rows = list(read_jsonl(args.input))[args.start :]
    tasks: list[OCRTask] = []
    stats = {"selected_posts": len(rows), "image_references": 0, "missing_local": 0, "done": 0}
    for line_number, post in rows:
        post_id = str(post.get("post_id", post.get("id", ""))).strip()
        if not post_id:
            raise ValueError(f"Dòng {line_number} thiếu post_id")
        label = str(post.get("label", ""))
        images = post.get("images", [])
        if not isinstance(images, list):
            raise ValueError(f"Dòng {line_number}: images không phải list")
        for index in range(len(images)):
            stats["image_references"] += 1
            image_key = f"/images/{post_id}_{index}.jpg"
            if image_key in done:
                stats["done"] += 1
                continue
            local = args.images_dir / f"{post_id}_{index}.jpg"
            if not local.is_file() or local.stat().st_size == 0:
                stats["missing_local"] += 1
                continue
            tasks.append(OCRTask(post_id, image_key, str(local), label))
            if args.limit is not None and len(tasks) >= args.limit:
                return tasks, stats
    return tasks, stats


def prepare_image(path: Path, max_side: int, max_bytes: int, quality: int) -> tuple[str, int, int, int, bool]:
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        source_format = str(image.format or "").upper()
        image.load()
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        is_jpeg = source_format in {"JPEG", "JPG"}
        needs_resize = max(width, height) > max_side or len(raw) > max_bytes or not is_jpeg
        if not needs_resize:
            encoded = raw
        else:
            if max(width, height) > max_side:
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, subsampling=0, optimize=True)
            encoded = buffer.getvalue()
            width, height = image.size
    uri = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
    return uri, width, height, len(encoded), needs_resize


def get_session(workers: int) -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers, max_retries=0)
        session.mount("https://", adapter)
        THREAD_LOCAL.session = session
    return session


def strip_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[A-Za-z]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def normalize_regions(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        candidates = value.get("regions")
        if candidates is None:
            candidates = next((v for v in value.values() if isinstance(v, list)), [])
    elif isinstance(value, list):
        candidates = value
    else:
        raise OCRFailure("JSON trả về không phải object/list")
    if not isinstance(candidates, list):
        raise OCRFailure("regions không phải list")
    regions: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        box = item.get("bounding_box")
        if not text or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            coords = [max(0, min(1000, round(float(x)))) for x in box]
        except (TypeError, ValueError):
            continue
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            continue
        regions.append({"text": text, "bounding_box": coords})
    return regions


def extract_content(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OCRFailure(f"Response thiếu choices/message/content: {str(response)[:300]}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    raise OCRFailure("message.content không phải string/list")


def retry_delay(attempt: int, response: requests.Response | None) -> float:
    if response is not None and response.headers.get("Retry-After"):
        try:
            return min(120.0, max(1.0, float(response.headers["Retry-After"])))
        except ValueError:
            pass
    return min(60.0, 2 ** (attempt - 1) + random.uniform(0.2, 1.0))


def ocr_task(task: OCRTask, args: argparse.Namespace, endpoint: str, key: str, model: str) -> OCRResult:
    started = time.monotonic()
    last_error = "unknown error"
    last_status: int | None = None
    try:
        data_uri, width, height, upload_bytes, resized = prepare_image(
            Path(task.local_path), args.max_side, round(args.max_upload_mb * 1024 * 1024), args.jpeg_quality,
        )
    except Exception as exc:
        return OCRResult(False, task.post_id, task.image_key, task.local_path, 0,
                         round(time.monotonic() - started, 3), error=f"ImageError: {exc}")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    if not args.no_response_format:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(1, args.retries + 1):
        response: requests.Response | None = None
        try:
            response = get_session(args.workers).post(
                endpoint, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=args.timeout,
            )
            last_status = response.status_code
            if response.status_code >= 400:
                detail = response.text[:500].replace("\n", " ")
                failure = OCRFailure(f"HTTP {response.status_code}: {detail}", response.status_code)
                if response.status_code not in RETRYABLE_STATUS:
                    raise failure
                last_error = str(failure)
                if attempt < args.retries:
                    time.sleep(retry_delay(attempt, response))
                    continue
                raise failure
            body = response.json()
            parsed = json.loads(strip_fence(extract_content(body)))
            regions = normalize_regions(parsed)
            raw_usage = body.get("usage") or {}
            usage = {k: int(v) for k, v in raw_usage.items() if isinstance(v, (int, float))}
            return OCRResult(True, task.post_id, task.image_key, task.local_path, attempt,
                             round(time.monotonic() - started, 3), regions, usage,
                             width, height, upload_bytes, resized, response.status_code)
        except (requests.RequestException, ValueError, json.JSONDecodeError, OCRFailure) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, OCRFailure) and exc.status is not None:
                last_status = exc.status
            non_retryable = isinstance(exc, OCRFailure) and exc.status is not None and exc.status not in RETRYABLE_STATUS
            if non_retryable or attempt >= args.retries:
                break
            time.sleep(retry_delay(attempt, response))
    return OCRResult(False, task.post_id, task.image_key, task.local_path, attempt,
                     round(time.monotonic() - started, 3), upload_width=width,
                     upload_height=height, upload_bytes=upload_bytes, resized=resized,
                     http_status=last_status, error=last_error)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def compile_json(jsonl_path: Path, json_path: Path) -> int:
    rows = [item for _, item in read_jsonl(jsonl_path)] if jsonl_path.is_file() else []
    write_json_atomic(json_path, rows)
    return len(rows)


def run(tasks: list[OCRTask], args: argparse.Namespace, endpoint: str, key: str, model: str) -> dict[str, Any]:
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.error_log.parent.mkdir(parents=True, exist_ok=True)
    error_temp = args.error_log.with_suffix(args.error_log.suffix + ".tmp")
    completed = succeeded = failed = blank = 0
    usage: dict[str, int] = {}
    aborted = False
    started = time.monotonic()
    max_pending = args.workers * 2
    task_by_key = {task.image_key: task for task in tasks}

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="ocr-api") as executor, \
            args.output_jsonl.open("a", encoding="utf-8") as output, \
            error_temp.open("w", encoding="utf-8") as errors:
        iterator = iter(tasks)
        pending: dict[Future[OCRResult], OCRTask] = {}

        def submit_one() -> bool:
            try:
                task = next(iterator)
            except StopIteration:
                return False
            pending[executor.submit(ocr_task, task, args, endpoint, key, model)] = task
            return True

        while len(pending) < max_pending and submit_one():
            pass
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                completed += 1
                try:
                    result = future.result()
                except Exception as exc:
                    result = OCRResult(False, task.post_id, task.image_key, task.local_path, 0, 0,
                                       error=f"WorkerError: {type(exc).__name__}: {exc}")
                if result.ok:
                    succeeded += 1
                    blank += not bool(result.regions)
                    for name, value in (result.usage or {}).items():
                        usage[name] = usage.get(name, 0) + value
                    source = task_by_key[result.image]
                    item = {
                        "id": result.post_id, "post_id": result.post_id, "image": result.image,
                        "ground_truth": source.label, "label": source.label,
                        "ocr_engine": "openai-compatible-vision", "ocr_model": model,
                        "gemini": result.regions or [], "usage": result.usage or {},
                        "request": {"attempts": result.attempts, "elapsed_seconds": result.elapsed_seconds,
                                    "upload_width": result.upload_width, "upload_height": result.upload_height,
                                    "upload_bytes": result.upload_bytes, "resized": result.resized},
                    }
                    output.write(json.dumps(item, ensure_ascii=False) + "\n")
                    output.flush()
                else:
                    failed += 1
                    errors.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                    errors.flush()

                if completed % args.report_every == 0 or completed == len(tasks):
                    elapsed = max(0.001, time.monotonic() - started)
                    print(f"[{completed}/{len(tasks)}] ok={succeeded}, blank={blank}, lỗi={failed}, "
                          f"{completed / elapsed:.3f} ảnh/s", flush=True)
                if completed >= args.fail_rate_min_samples and failed / completed > args.max_fail_rate:
                    aborted = True
                if not aborted:
                    submit_one()

    os.replace(error_temp, args.error_log)
    elapsed = time.monotonic() - started
    return {"succeeded": succeeded, "blank": blank, "failed": failed,
            "completed_this_run": completed, "not_attempted_due_to_abort": max(0, len(tasks) - completed),
            "elapsed_seconds": round(elapsed, 3), "images_per_second": round(completed / elapsed, 4) if elapsed else 0,
            "usage": usage, "aborted_high_failure_rate": aborted}


def main() -> int:
    args = parse_args()
    if not args.input.is_file() or not args.images_dir.is_dir():
        print(f"LỖI: thiếu input hoặc Images: {args.input} / {args.images_dir}", file=sys.stderr)
        return 2
    load_env_file(args.env_file)
    model = args.model or os.environ.get("OCR_MODEL", "")
    try:
        done = load_done(args.output_jsonl, model or None, args.overwrite)
        tasks, stats = build_tasks(args, done)
    except (OSError, ValueError) as exc:
        print(f"LỖI DỮ LIỆU: {exc}", file=sys.stderr)
        return 2

    print("=== Kế hoạch OCR API ===")
    print(f"Input: {args.input}")
    print(f"Images: {args.images_dir}")
    print(f"Model: {model or '(chưa cấu hình)'}; workers: {args.workers}")
    print(f"Ảnh tham chiếu: {stats['image_references']}; đã xong: {stats['done']}; thiếu local: {stats['missing_local']}")
    print(f"Sẽ OCR phiên này: {len(tasks)}; output: {args.output_jsonl}")
    if args.dry_run or not tasks:
        return 0

    key = os.environ.get("OCR_API_KEY", "")
    base_url = args.base_url or os.environ.get("OCR_BASE_URL", "")
    if not key or not base_url or not model:
        print("LỖI: thiếu OCR_API_KEY/OCR_BASE_URL/OCR_MODEL; dùng --env-file hoặc tạo .env.", file=sys.stderr)
        return 2
    run_stats = run(tasks, args, api_endpoint(base_url), key, model)
    compiled = compile_json(args.output_jsonl, args.output_json)
    summary = {**stats, "scheduled": len(tasks), "model": model, "workers": args.workers,
               **run_stats, "compiled_results": compiled,
               "output_jsonl": str(args.output_jsonl.resolve()), "error_log": str(args.error_log.resolve())}
    write_json_atomic(args.summary, summary)
    print("\n=== Tổng kết ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if run_stats["aborted_high_failure_rate"]:
        print("LỖI: dừng sớm vì tỷ lệ lỗi API quá cao; kiểm tra key/model/quota.", file=sys.stderr)
        return 2
    return 1 if run_stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
