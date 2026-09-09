#!/usr/bin/env python3
"""Tải song song ảnh từ trường ``images`` của file JSONL.

Script dành cho ``data/mrDuc_data/valid.jsonl`` có schema::

    {"post_id": "...", "label": "...", "images": ["https://..."]}

Mỗi ảnh được lưu thành ``{post_id}_{index}.jpg``. JPEG được giữ nguyên byte;
PNG/WebP được chuyển sang JPEG RGB. File chỉ xuất hiện sau khi tải và kiểm tra
thành công, nên có thể chạy lại để resume mà không làm hỏng output.

Ví dụ::

    python scripts/facebook/1_fetch_images.py --dry-run
    python scripts/facebook/1_fetch_images.py --limit 20 --workers 8
    python scripts/facebook/1_fetch_images.py --workers 24
"""

from __future__ import annotations

import argparse
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
from PIL import Image, ImageOps, UnidentifiedImageError
from requests.adapters import HTTPAdapter


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_DIR / "data" / "input" / "valid.jsonl"
DEFAULT_IMAGES_DIR = PROJECT_DIR / "data" / "input" / "Images"
DEFAULT_ERROR_LOG = PROJECT_DIR / "data" / "output" / "mrDuc_data_ocr" / "fetch_errors.jsonl"
DEFAULT_SUMMARY = PROJECT_DIR / "data" / "output" / "mrDuc_data_ocr" / "fetch_summary.json"
DEFAULT_WORKERS = min(24, max(4, (os.cpu_count() or 4) // 2))
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
SAFE_ID = re.compile(r"[^A-Za-z0-9_.=+-]+")
THREAD_LOCAL = threading.local()
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class DownloadTask:
    post_id: str
    image_index: int
    url: str
    target: str


@dataclass
class DownloadResult:
    ok: bool
    post_id: str
    image_index: int
    url: str
    target: str
    attempts: int
    elapsed_seconds: float
    bytes_written: int = 0
    width: int = 0
    height: int = 0
    source_format: str = ""
    http_status: int | None = None
    error: str | None = None


class DownloadFailure(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--error-log", type=Path, default=DEFAULT_ERROR_LOG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--start", type=int, default=0, help="Bỏ qua N post đầu")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý tối đa N post")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Số luồng tải đồng thời (mặc định: {DEFAULT_WORKERS})",
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=60.0)
    parser.add_argument("--max-image-mb", type=float, default=50.0)
    parser.add_argument("--min-side", type=int, default=32)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--max-fail-rate", type=float, default=0.25,
        help="Dừng cấp task mới nếu sau ít nhất 100 lượt, tỷ lệ lỗi vượt ngưỡng",
    )
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument(
        "--verify-existing", action="store_true",
        help="Mở kiểm tra file đã có; mặc định file có size > 0 được resume",
    )
    parser.add_argument("--overwrite", action="store_true", help="Tải lại cả file đã tồn tại")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra dữ liệu và kế hoạch")
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start không được âm")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")
    if args.workers < 1 or args.workers > 128:
        parser.error("--workers phải nằm trong 1..128")
    if args.retries < 1:
        parser.error("--retries phải lớn hơn 0")
    if args.connect_timeout <= 0 or args.read_timeout <= 0:
        parser.error("timeout phải lớn hơn 0")
    if args.max_image_mb <= 0:
        parser.error("--max-image-mb phải lớn hơn 0")
    if args.min_side < 1:
        parser.error("--min-side phải lớn hơn 0")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality phải nằm trong 1..100")
    if not 0 < args.max_fail_rate <= 1:
        parser.error("--max-fail-rate phải nằm trong (0, 1]")
    if args.report_every < 1:
        parser.error("--report-every phải lớn hơn 0")
    return args


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
                raise ValueError(f"Dòng {line_number} không phải JSON object")
            yield line_number, value


def safe_post_id(value: Any, line_number: int) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Dòng {line_number} thiếu post_id")
    safe = SAFE_ID.sub("_", raw)
    if len(safe) > 180:
        raise ValueError(f"Dòng {line_number} có post_id quá dài")
    return safe


def valid_existing(path: Path, verify: bool, min_side: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if not verify:
        return True
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width >= min_side and image.height >= min_side
    except (OSError, UnidentifiedImageError):
        return False


def build_tasks(args: argparse.Namespace) -> tuple[list[DownloadTask], dict[str, int]]:
    rows = list(read_jsonl(args.input))
    selected = rows[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    tasks: list[DownloadTask] = []
    seen_targets: set[str] = set()
    stats = {
        "total_posts": len(rows),
        "selected_posts": len(selected),
        "image_references": 0,
        "already_complete": 0,
        "invalid_urls": 0,
    }
    for line_number, post in selected:
        post_id = safe_post_id(post.get("post_id", post.get("id")), line_number)
        images = post.get("images")
        if not isinstance(images, list):
            raise ValueError(f"Dòng {line_number}: images không phải list")
        for image_index, url_value in enumerate(images):
            stats["image_references"] += 1
            url = str(url_value or "").strip()
            if not url.startswith(("https://", "http://")):
                stats["invalid_urls"] += 1
                continue
            target = args.images_dir / f"{post_id}_{image_index}.jpg"
            target_key = str(target)
            if target_key in seen_targets:
                raise ValueError(f"Output trùng tại dòng {line_number}: {target}")
            seen_targets.add(target_key)
            if not args.overwrite and valid_existing(target, args.verify_existing, args.min_side):
                stats["already_complete"] += 1
                continue
            tasks.append(DownloadTask(post_id, image_index, url, target_key))
    return tasks, stats


def get_session(workers: int) -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
        })
        THREAD_LOCAL.session = session
    return session


def response_bytes(response: requests.Response, max_bytes: int) -> bytes:
    declared = int(response.headers.get("Content-Length") or 0)
    if declared > max_bytes:
        raise DownloadFailure(f"Content-Length {declared} vượt giới hạn {max_bytes}", response.status_code)
    data = bytearray()
    for chunk in response.iter_content(chunk_size=256 * 1024):
        if not chunk:
            continue
        data.extend(chunk)
        if len(data) > max_bytes:
            raise DownloadFailure(f"Nội dung vượt giới hạn {max_bytes} byte", response.status_code)
    if not data:
        raise DownloadFailure("Response rỗng", response.status_code)
    return bytes(data)


def atomic_save_jpeg(
    data: bytes, target: Path, min_side: int, quality: int,
) -> tuple[int, int, int, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    try:
        with Image.open(io.BytesIO(data)) as image:
            source_format = str(image.format or "UNKNOWN").upper()
            image.load()
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            if width < min_side or height < min_side:
                raise DownloadFailure(f"Ảnh quá nhỏ: {width}x{height}")

            if source_format in {"JPEG", "JPG"}:
                temp.write_bytes(data)
            else:
                if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGB", rgba.size, "white")
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    image = background
                elif image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(temp, format="JPEG", quality=quality, subsampling=0, optimize=True)
        os.replace(temp, target)
        return target.stat().st_size, width, height, source_format
    finally:
        temp.unlink(missing_ok=True)


def retry_delay(attempt: int, response: requests.Response | None = None) -> float:
    if response is not None:
        value = response.headers.get("Retry-After")
        if value:
            try:
                return min(60.0, max(1.0, float(value)))
            except ValueError:
                pass
    return min(30.0, (2 ** (attempt - 1)) + random.uniform(0.2, 1.0))


def download_task(task: DownloadTask, args: argparse.Namespace) -> DownloadResult:
    started = time.monotonic()
    last_error = "unknown error"
    last_status: int | None = None
    max_bytes = round(args.max_image_mb * 1024 * 1024)

    for attempt in range(1, args.retries + 1):
        response: requests.Response | None = None
        try:
            session = get_session(args.workers)
            response = session.get(
                task.url,
                timeout=(args.connect_timeout, args.read_timeout),
                allow_redirects=True,
                stream=True,
            )
            last_status = response.status_code
            if response.status_code >= 400:
                message = f"HTTP {response.status_code}"
                if response.status_code not in RETRYABLE_STATUS:
                    raise DownloadFailure(message, response.status_code)
                last_error = message
                if attempt < args.retries:
                    time.sleep(retry_delay(attempt, response))
                    continue
                raise DownloadFailure(message, response.status_code)

            data = response_bytes(response, max_bytes)
            size, width, height, source_format = atomic_save_jpeg(
                data, Path(task.target), args.min_side, args.jpeg_quality,
            )
            return DownloadResult(
                True, task.post_id, task.image_index, task.url, task.target,
                attempt, round(time.monotonic() - started, 3), size, width,
                height, source_format, response.status_code,
            )
        except (requests.RequestException, OSError, UnidentifiedImageError, DownloadFailure) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, DownloadFailure) and exc.status is not None:
                last_status = exc.status
            non_retryable = (
                isinstance(exc, DownloadFailure)
                and exc.status is not None
                and exc.status not in RETRYABLE_STATUS
            )
            if non_retryable or attempt >= args.retries:
                break
            time.sleep(retry_delay(attempt, response))
        finally:
            if response is not None:
                response.close()

    return DownloadResult(
        False, task.post_id, task.image_index, task.url, task.target,
        attempt, round(time.monotonic() - started, 3),
        http_status=last_status, error=last_error,
    )


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def run_downloads(tasks: list[DownloadTask], args: argparse.Namespace) -> dict[str, Any]:
    args.error_log.parent.mkdir(parents=True, exist_ok=True)
    error_temp = args.error_log.with_suffix(args.error_log.suffix + ".tmp")
    downloaded = failed = bytes_written = completed = 0
    aborted = False
    started = time.monotonic()
    max_pending = max(args.workers, args.workers * 3)

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="image-fetch") as executor, \
            error_temp.open("w", encoding="utf-8") as error_handle:
        iterator = iter(tasks)
        pending: dict[Future[DownloadResult], DownloadTask] = {}

        def submit_one() -> bool:
            try:
                task = next(iterator)
            except StopIteration:
                return False
            pending[executor.submit(download_task, task, args)] = task
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
                except Exception as exc:  # giữ batch sống trước lỗi ngoài dự kiến
                    result = DownloadResult(
                        False, task.post_id, task.image_index, task.url, task.target,
                        0, 0.0, error=f"WorkerError: {type(exc).__name__}: {exc}",
                    )
                if result.ok:
                    downloaded += 1
                    bytes_written += result.bytes_written
                else:
                    failed += 1
                    error_handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                    error_handle.flush()

                if completed % args.report_every == 0 or completed == len(tasks):
                    elapsed = max(0.001, time.monotonic() - started)
                    rate = completed / elapsed
                    print(
                        f"[{completed}/{len(tasks)}] tải={downloaded}, lỗi={failed}, "
                        f"{rate:.2f} ảnh/s, {bytes_written / 2**30:.2f} GiB",
                        flush=True,
                    )

                if completed >= 100 and failed / completed > args.max_fail_rate:
                    aborted = True
                if not aborted:
                    submit_one()

    os.replace(error_temp, args.error_log)
    elapsed = time.monotonic() - started
    return {
        "downloaded": downloaded,
        "failed": failed,
        "completed_this_run": completed,
        "not_attempted_due_to_abort": max(0, len(tasks) - completed),
        "bytes_written": bytes_written,
        "gib_written": round(bytes_written / 2**30, 4),
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(completed / elapsed, 3) if elapsed else 0,
        "aborted_high_failure_rate": aborted,
    }


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"LỖI: không tìm thấy input: {args.input}", file=sys.stderr)
        return 2
    args.images_dir.mkdir(parents=True, exist_ok=True)

    try:
        tasks, source_stats = build_tasks(args)
    except (OSError, ValueError) as exc:
        print(f"LỖI DỮ LIỆU: {exc}", file=sys.stderr)
        return 2

    print("=== Kế hoạch tải ảnh ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.images_dir}")
    print(f"Tổng post: {source_stats['total_posts']}; được chọn: {source_stats['selected_posts']}")
    print(f"Tham chiếu ảnh: {source_stats['image_references']}")
    print(f"Đã có hợp lệ: {source_stats['already_complete']}; URL lỗi schema: {source_stats['invalid_urls']}")
    print(f"Cần tải: {len(tasks)}; workers: {args.workers}; retries: {args.retries}")
    print(f"Ngưỡng dừng lỗi: {args.max_fail_rate:.0%} sau ít nhất 100 lượt")

    if args.dry_run:
        return 0

    if tasks:
        run_stats = run_downloads(tasks, args)
    else:
        # Chế độ ảnh local: vẫn tạo đủ artifact để Nextflow xác nhận stage 1
        # hoàn tất, dù không có URL nào cần tải.
        args.error_log.parent.mkdir(parents=True, exist_ok=True)
        args.error_log.write_text("", encoding="utf-8")
        run_stats = {
            "downloaded": 0,
            "failed": 0,
            "completed_this_run": 0,
            "not_attempted_due_to_abort": 0,
            "bytes_written": 0,
            "gib_written": 0.0,
            "elapsed_seconds": 0.0,
            "images_per_second": 0.0,
            "aborted_high_failure_rate": False,
        }
    summary = {
        "input": str(args.input.resolve()),
        "images_dir": str(args.images_dir.resolve()),
        "workers": args.workers,
        "retries": args.retries,
        **source_stats,
        "scheduled": len(tasks),
        **run_stats,
        "error_log": str(args.error_log.resolve()),
    }
    write_json_atomic(args.summary, summary)

    print("\n=== Tổng kết ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if run_stats["aborted_high_failure_rate"]:
        print("LỖI: đã dừng sớm vì URL lỗi quá nhiều; có thể URL đã hết hạn.", file=sys.stderr)
        return 2
    return 1 if run_stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
