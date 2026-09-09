#!/usr/bin/env python3
"""OCR ảnh Facebook bằng PP-OCRv6 trên CPU, có resume và fallback thích nghi.

Script đọc ``facebook_posts_valid.jsonl``, tìm ảnh theo quy ước
``images/{post_id}_{index}.jpg`` và ghi kết quả tăng dần vào
``facebook_posts_ocr_paddle.jsonl``. Kết quả Paddle được lưu riêng, không
ghi đè file OCR Gemini.

Máy nhiều lõi nên dùng vài worker độc lập, mỗi worker dùng nhiều thread
MKL/OpenMP. Mặc định script tự chia toàn bộ lõi vật lý cho tối đa 4 worker.

Pipeline luôn OCR ảnh gốc trước. Chỉ khi kết quả gốc rỗng hoặc có confidence
trung bình thấp, script mới thử đúng một biến thể phù hợp: ``invert`` cho nền
tối/chữ sáng hoặc ``CLAHE`` cho chữ mờ trên nền tương đối đồng đều. Pipeline
không dùng adaptive threshold. Kết quả ảnh gốc luôn được giữ làm baseline.

Ví dụ:
    python scripts/facebook/lib/paddle_v6_cpu.py --dry-run
    python scripts/facebook/lib/paddle_v6_cpu.py --limit 3 --workers 1
    python scripts/facebook/lib/paddle_v6_cpu.py
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "input" / "valid.jsonl"
DEFAULT_IMAGES_DIR = ROOT / "data" / "input" / "Images"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "output" / "PaddleV6_full"
DEFAULT_OUTPUT_JSONL = DEFAULT_OUTPUT_DIR / "ocr_results.jsonl"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "ocr_results.json"
DEFAULT_ERROR_LOG = DEFAULT_OUTPUT_DIR / "ocr_errors.jsonl"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

_OCR: Any = None
_SCORE_THRESHOLD = 0.30
_FALLBACK_CONFIDENCE = 0.65
_ENABLE_FALLBACK = True
_OCR_VERSION = "PP-OCRv6"


def physical_cpu_count() -> int:
    """Đếm lõi vật lý từ /proc/cpuinfo; fallback về một nửa logical CPU."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        pairs: set[tuple[str, str]] = set()
        physical_id = core_id = None
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines() + [""]:
            if line.startswith("physical id"):
                physical_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
            elif not line.strip():
                if physical_id is not None and core_id is not None:
                    pairs.add((physical_id, core_id))
                physical_id = core_id = None
        if pairs:
            return len(pairs)
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def parse_args() -> argparse.Namespace:
    physical = physical_cpu_count()
    default_workers = min(4, max(1, physical // 8))
    default_threads = max(1, physical // default_workers)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--error-log", type=Path, default=DEFAULT_ERROR_LOG)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--limit", type=int, help="Chỉ OCR N ảnh chưa làm đầu tiên")
    selector.add_argument(
        "--last", type=int,
        help="Chỉ OCR N ảnh local cuối cùng; tập ảnh cố định và hỗ trợ resume",
    )
    parser.add_argument(
        "--workers", type=int, default=default_workers,
        help=f"Số process Paddle độc lập (mặc định: {default_workers})",
    )
    parser.add_argument(
        "--cpu-threads", type=int, default=default_threads,
        help=f"Số thread cho mỗi worker (mặc định: {default_threads})",
    )
    parser.add_argument("--lang", default="chinese_cht")
    parser.add_argument(
        "--ocr-version", default="PP-OCRv6",
        choices=("PP-OCRv6",),
        help="Phiên bản model bắt buộc của dự án (mặc định: PP-OCRv6)",
    )
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument(
        "--fallback-confidence", type=float, default=0.65,
        help="Thử fallback khi confidence trung bình original thấp hơn ngưỡng này (mặc định: 0.65)",
    )
    parser.add_argument(
        "--no-preprocess-fallback", action="store_true",
        help="Chỉ OCR original, không thử invert/CLAHE",
    )
    parser.add_argument("--model-cache", type=Path, default=ROOT / "models" / "paddlex")
    parser.add_argument(
        "--max-pending-factor", type=int, default=2,
        help="Số task chờ tối đa = workers × hệ số này (mặc định: 2)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra dữ liệu và cấu hình CPU")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit phải lớn hơn 0")
    if args.last is not None and args.last < 1:
        parser.error("--last phải lớn hơn 0")
    if args.workers < 1:
        parser.error("--workers phải lớn hơn 0")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads phải lớn hơn 0")
    if args.max_pending_factor < 1:
        parser.error("--max-pending-factor phải lớn hơn 0")
    if not 0 <= args.score_threshold <= 1:
        parser.error("--score-threshold phải nằm trong khoảng 0..1")
    if not 0 <= args.fallback_confidence <= 1:
        parser.error("--fallback-confidence phải nằm trong khoảng 0..1")
    return args


def set_cpu_environment(cpu_threads: int, model_cache: str) -> None:
    """Đặt thread trước khi import Paddle/Numpy trong worker."""
    value = str(cpu_threads)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "FLAGS_paddle_num_threads",
    ):
        os.environ[name] = value
    # Không bind OpenMP ở đây. Với nhiều process PaddleOCR, libgomp có thể
    # thu hẹp affinity của *mọi* worker vào cùng một core vật lý (ví dụ hai
    # logical CPU 0,52), khiến workers tranh chấp nhau và chậm nghiêm trọng.
    # Để Linux phân phối các thread trên toàn bộ CPU được phép.
    os.environ.pop("OMP_PROC_BIND", None)
    os.environ.pop("OMP_PLACES", None)
    os.environ["PADDLE_PDX_CACHE_HOME"] = model_cache
    os.environ["MPLCONFIGDIR"] = str(Path(model_cache) / "matplotlib")
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def init_worker(
    cpu_threads: int,
    model_cache: str,
    lang: str,
    ocr_version: str,
    score_threshold: float,
    fallback_confidence: float = 0.65,
    enable_fallback: bool = True,
) -> None:
    """Khởi tạo đúng một PaddleOCR pipeline trong mỗi process."""
    global _OCR, _SCORE_THRESHOLD, _FALLBACK_CONFIDENCE, _ENABLE_FALLBACK, _OCR_VERSION
    set_cpu_environment(cpu_threads, model_cache)
    Path(model_cache).mkdir(parents=True, exist_ok=True)
    _SCORE_THRESHOLD = score_threshold
    _FALLBACK_CONFIDENCE = fallback_confidence
    _ENABLE_FALLBACK = enable_fallback
    _OCR_VERSION = ocr_version
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài PaddleOCR. Hãy kích hoạt môi trường NLP hoặc chạy "
            "pip install -r requirements.txt"
        ) from exc

    common = {
        "lang": lang,
        "ocr_version": ocr_version,
        "device": "cpu",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_rec_score_thresh": score_threshold,
    }
    # Giữ MKLDNN tắt cho PP-OCRv6 trên CPU để tránh lỗi oneDNN đã quan sát với
    # PaddlePaddle 3.3.1; vẫn dùng cpu_threads và nhiều process khi cần.
    enable_mkldnn = ocr_version != "PP-OCRv6"
    # Giữ fallback nếu một bản PaddleOCR khác không nhận hai tùy chọn CPU.
    try:
        _OCR = PaddleOCR(enable_mkldnn=enable_mkldnn, cpu_threads=cpu_threads, **common)
    except (TypeError, ValueError):
        _OCR = PaddleOCR(**common)


def normalized_box(polygon: Any, width: int, height: int) -> list[int]:
    """Đổi polygon pixel thành [ymin, xmin, ymax, xmax] trong 0..1000."""
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return [0, 0, 0, 0]
    if not xs or not ys or width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    values = [
        round(1000 * min(ys) / height),
        round(1000 * min(xs) / width),
        round(1000 * max(ys) / height),
        round(1000 * max(xs) / width),
    ]
    return [max(0, min(1000, int(value))) for value in values]


def extract_regions(source: Any, width: int, height: int) -> list[dict[str, Any]]:
    """Chạy Paddle và chuẩn hóa kết quả của một ảnh/path thành các vùng chữ."""
    predictions = list(_OCR.predict(source))
    if not predictions:
        raise ValueError("PaddleOCR không trả result")
    raw = predictions[0].json
    result = raw.get("res", raw)
    texts = list(result.get("rec_texts", []))
    scores = [float(value) for value in result.get("rec_scores", [])]
    polygons = list(result.get("rec_polys", []))

    regions: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        clean_text = str(text).strip()
        score = scores[index] if index < len(scores) else 0.0
        if not clean_text or score < _SCORE_THRESHOLD:
            continue
        polygon = polygons[index] if index < len(polygons) else []
        regions.append({
            "text": clean_text,
            "bounding_box": normalized_box(polygon, width, height),
            "score": round(score, 6),
        })
    return regions


def region_metrics(regions: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(region.get("score", 0)) for region in regions]
    texts = [str(region.get("text", "")) for region in regions]
    return {
        "region_count": len(regions),
        "character_count": sum(len(text.replace(" ", "")) for text in texts),
        "mean_confidence": round(sum(scores) / len(scores), 6) if scores else 0.0,
    }


def classify_fallback(image: Any) -> tuple[str | None, dict[str, Any]]:
    """Chọn tối đa một fallback dựa trên thống kê ảnh, không dùng threshold nhị phân."""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    strip = max(2, min(height, width) // 50)
    border = np.concatenate((
        gray[:strip, :].ravel(), gray[-strip:, :].ravel(),
        gray[:, :strip].ravel(), gray[:, -strip:].ravel(),
    ))
    p10, p90 = np.percentile(gray, [10, 90])
    median = float(np.median(gray))
    border_std = float(np.std(border))
    bright_ratio = float(np.mean(gray >= 200))
    contrast = float(p90 - p10)

    # Nền tối chiếm ưu thế nhưng vẫn có một lượng nét sáng đáng kể.
    dark_background = median < 105 and bright_ratio >= 0.02
    # Chữ mờ thường có nền sáng/đồng đều và dải tương phản không quá rộng.
    faded_uniform = median >= 130 and border_std <= 35 and contrast <= 125
    profile = {
        "median_gray": round(median, 3),
        "p10_p90_contrast": round(contrast, 3),
        "border_std": round(border_std, 3),
        "bright_pixel_ratio": round(bright_ratio, 6),
        "dark_background": dark_background,
        "faded_uniform_background": faded_uniform,
    }
    if dark_background:
        return "invert", profile
    if faded_uniform:
        return "clahe", profile
    return None, profile


def make_fallback(image: Any, variant: str) -> Any:
    import cv2

    if variant == "invert":
        return cv2.bitwise_not(image)
    if variant == "clahe":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        # PaddleOCR/PaddleX nhận ảnh BGR ổn định hơn ndarray xám một kênh.
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Fallback không hỗ trợ: {variant}")


def should_use_fallback(
    original: list[dict[str, Any]], fallback: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Chọn fallback bảo thủ để confidence cao giả không thay thế baseline."""
    original_metrics = region_metrics(original)
    fallback_metrics = region_metrics(fallback)
    fallback_confidence = float(fallback_metrics["mean_confidence"])
    fallback_chars = int(fallback_metrics["character_count"])

    if not fallback:
        return False, "fallback_blank"
    if not original:
        if (fallback_metrics["region_count"] >= 2 and fallback_chars >= 4
                and fallback_confidence >= max(0.70, _FALLBACK_CONFIDENCE)):
            return True, "recovered_blank_with_strong_fallback"
        return False, "blank_recovery_not_reliable"

    original_confidence = float(original_metrics["mean_confidence"])
    original_chars = max(1, int(original_metrics["character_count"]))
    char_ratio = fallback_chars / original_chars
    original_text = "\n".join(str(x.get("text", "")) for x in original)
    fallback_text = "\n".join(str(x.get("text", "")) for x in fallback)
    similarity = SequenceMatcher(None, original_text, fallback_text).ratio()
    if (fallback_confidence >= original_confidence + 0.05
            and 0.75 <= char_ratio <= 1.50 and similarity >= 0.55):
        return True, "higher_confidence_with_stable_text"
    return False, "fallback_not_safely_better"


def ocr_task(task: dict[str, str]) -> dict[str, Any]:
    """OCR một ảnh. Không ném lỗi ra parent để batch tiếp tục chạy."""
    started = time.monotonic()
    try:
        import cv2

        image = cv2.imread(task["local_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV không đọc được ảnh")
        height, width = image.shape[:2]
        original = extract_regions(task["local_path"], width, height)
        original_metrics = region_metrics(original)
        selected = original
        selected_variant = "original"
        fallback_variant = None
        fallback_regions: list[dict[str, Any]] | None = None
        image_profile: dict[str, Any] = {}
        selection_reason = "original_confidence_is_good"

        original_is_good = bool(original) and (
            float(original_metrics["mean_confidence"]) >= _FALLBACK_CONFIDENCE
        )
        if _ENABLE_FALLBACK and not original_is_good:
            fallback_variant, image_profile = classify_fallback(image)
            if fallback_variant is None:
                selection_reason = "no_safe_preprocessing_matches_image_profile"
            else:
                transformed = make_fallback(image, fallback_variant)
                fallback_height, fallback_width = transformed.shape[:2]
                fallback_regions = extract_regions(transformed, fallback_width, fallback_height)
                use_fallback, selection_reason = should_use_fallback(original, fallback_regions)
                if use_fallback:
                    selected = fallback_regions
                    selected_variant = fallback_variant
        elif not _ENABLE_FALLBACK:
            selection_reason = "preprocess_fallback_disabled"

        return {
            "ok": True,
            "id": task["id"],
            "image": task["image"],
            "ground_truth": task["label"],
            "label": task["label"],
            "ocr_engine": "paddleocr",
            "ocr_version": _OCR_VERSION,
            "paddle": selected,
            "ocr_pipeline": {
                "selected_variant": selected_variant,
                "selection_reason": selection_reason,
                "fallback_confidence_threshold": _FALLBACK_CONFIDENCE,
                "original": {
                    "metrics": original_metrics,
                    "paddle": original,
                },
                "fallback": None if fallback_variant is None else {
                    "variant": fallback_variant,
                    "metrics": region_metrics(fallback_regions or []),
                    "paddle": fallback_regions or [],
                },
                "image_profile": image_profile,
            },
            "image_width": width,
            "image_height": height,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        return {
            "ok": False,
            "id": task["id"],
            "image": task["image"],
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"CẢNH BÁO: bỏ dòng JSON lỗi {path}:{lineno}: {exc}", file=sys.stderr)
                continue
            if isinstance(value, dict):
                yield lineno, value


def load_done_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        str(item.get("image"))
        for _, item in read_jsonl(path)
        if item.get("image")
    }


def image_path(images_dir: Path, post_id: str, index: int) -> Path | None:
    stem = f"{post_id}_{index}"
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def build_tasks(input_path: Path, images_dir: Path, done: set[str]) -> tuple[list[dict[str, str]], int, int]:
    tasks: list[dict[str, str]] = []
    seen: set[str] = set()
    missing = duplicate = 0
    for _, post in read_jsonl(input_path):
        post_id = str(post.get("id", "")).strip()
        label = str(post.get("label", ""))
        images = post.get("images", [])
        if not post_id or not isinstance(images, list):
            continue
        for index in range(len(images)):
            key = f"/images/{post_id}_{index}.jpg"
            if key in seen:
                duplicate += 1
                continue
            seen.add(key)
            if key in done:
                continue
            local = image_path(images_dir, post_id, index)
            if local is None:
                missing += 1
                continue
            tasks.append({"id": post_id, "image": key, "label": label, "local_path": str(local)})
    return tasks, missing, duplicate


def compile_json_array(jsonl_path: Path, json_path: Path) -> int:
    results = [item for _, item in read_jsonl(jsonl_path)] if jsonl_path.is_file() else []
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temp = json_path.with_suffix(json_path.suffix + ".tmp")
    # JSON compact vẫn giữ nguyên toàn bộ dữ liệu nhưng tránh gần 50 MB
    # khoảng trắng/thụt lề ở tập OCR lớn. JSONL tiếp tục được giữ riêng để
    # hỗ trợ resume và xử lý streaming.
    temp.write_text(
        json.dumps(results, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temp.replace(json_path)
    return len(results)


def compact_error_log(error_path: Path, successful_keys: set[str]) -> int:
    """Giữ một lỗi mới nhất/ảnh và bỏ lỗi cũ nếu ảnh đã OCR thành công."""
    latest: dict[str, dict[str, Any]] = {}
    if error_path.is_file():
        for _, item in read_jsonl(error_path):
            key = str(item.get("image", ""))
            if key and key not in successful_keys:
                latest[key] = item
    error_path.parent.mkdir(parents=True, exist_ok=True)
    temp = error_path.with_suffix(error_path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for item in latest.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    temp.replace(error_path)
    return len(latest)


def run_parallel(
    tasks: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    """Giữ hàng đợi nhỏ; chỉ process cha ghi JSONL nên không hỏng file."""
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.error_log.parent.mkdir(parents=True, exist_ok=True)
    success = blank = errors = 0
    total = len(tasks)
    max_pending = max(args.workers, args.workers * args.max_pending_factor)
    context = mp.get_context("spawn")

    executor = ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=init_worker,
        initargs=(
            args.cpu_threads,
            str(args.model_cache.resolve()),
            args.lang,
            args.ocr_version,
            args.score_threshold,
            args.fallback_confidence,
            not args.no_preprocess_fallback,
        ),
    )
    interrupted = False
    try:
        with args.output_jsonl.open("a", encoding="utf-8") as output, args.error_log.open(
            "a", encoding="utf-8"
        ) as error_output:
            iterator = iter(tasks)
            pending: dict[Any, dict[str, str]] = {}

            def submit_one() -> bool:
                try:
                    task = next(iterator)
                except StopIteration:
                    return False
                pending[executor.submit(ocr_task, task)] = task
                return True

            while len(pending) < max_pending and submit_one():
                pass

            completed = 0
            while pending:
                done_futures, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    task = pending.pop(future)
                    completed += 1
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "ok": False,
                            "id": task["id"],
                            "image": task["image"],
                            "error": f"WorkerError: {exc}",
                        }
                    if result.pop("ok", False):
                        region_count = len(result.get("paddle", []))
                        output.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output.flush()
                        if region_count:
                            success += 1
                            selected_variant = result.get("ocr_pipeline", {}).get("selected_variant", "original")
                            status = f"OK {region_count} vùng [{selected_variant}]"
                        else:
                            blank += 1
                            status = "BLANK"
                    else:
                        errors += 1
                        error_output.write(json.dumps(result, ensure_ascii=False) + "\n")
                        error_output.flush()
                        status = f"LỖI {result.get('error', '')}"
                    print(f"[{completed}/{total}] {task['image']} — {status}", flush=True)
                    submit_one()
    except KeyboardInterrupt:
        interrupted = True
        print("\nĐang dừng toàn bộ worker Paddle...", file=sys.stderr, flush=True)
        # ProcessPoolExecutor mặc định chờ task đang chạy khi thoát context.
        # Terminate rõ ràng để một lần Ctrl+C không để lại worker mồ côi.
        for process in list(getattr(executor, "_processes", {}).values()):
            if process.is_alive():
                process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if not interrupted:
            executor.shutdown(wait=True)
    return success, blank, errors


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"LỖI: không tìm thấy input: {args.input}", file=sys.stderr)
        return 2
    if not args.images_dir.is_dir():
        print(f"LỖI: không tìm thấy thư mục ảnh: {args.images_dir}", file=sys.stderr)
        return 2

    done = load_done_keys(args.output_jsonl)
    if args.last is not None:
        # Chọn tail trước khi loại ảnh đã hoàn thành. Vì vậy chạy lại vẫn
        # giữ nguyên tập N ảnh cuối, không kéo thêm ảnh nằm trước tập này.
        all_tasks, missing, duplicates = build_tasks(args.input, args.images_dir, set())
        selected = all_tasks[-args.last :]
        tasks = [task for task in selected if task["image"] not in done]
    else:
        tasks, missing, duplicates = build_tasks(args.input, args.images_dir, done)
        if args.limit is not None:
            tasks = tasks[: args.limit]

    physical = physical_cpu_count()
    logical = os.cpu_count() or physical
    print("=== PaddleOCR CPU plan ===")
    print(f"CPU: {physical} lõi vật lý / {logical} luồng logic")
    print(f"Workers: {args.workers}; threads/worker: {args.cpu_threads}; tổng thread: {args.workers * args.cpu_threads}")
    print(f"Đã OCR trước đó: {len(done)} ảnh")
    if args.last is not None:
        print(f"Phạm vi cố định: {min(args.last, len(all_tasks))} ảnh local cuối cùng")
    print(f"Sẽ OCR phiên này: {len(tasks)} ảnh")
    print(f"Thiếu ảnh local: {missing}; tham chiếu trùng: {duplicates}")
    print(f"Output: {args.output_jsonl}")
    print(f"Model OCR: {args.ocr_version}")
    fallback_status = "tắt" if args.no_preprocess_fallback else "invert/CLAHE thích nghi"
    print(f"Tiền xử lý fallback: {fallback_status}; ngưỡng confidence: {args.fallback_confidence:.2f}")

    if args.workers * args.cpu_threads > logical:
        print("CẢNH BÁO: tổng thread vượt số CPU logic; có thể chậm do oversubscription.")
    if args.dry_run or not tasks:
        return 0

    started = time.monotonic()
    success, blank, errors = run_parallel(tasks, args)
    compiled = compile_json_array(args.output_jsonl, args.output_json)
    unresolved_errors = compact_error_log(args.error_log, load_done_keys(args.output_jsonl))
    elapsed = time.monotonic() - started
    print("\n=== Tổng kết ===")
    print(f"Success: {success}; blank: {blank}; error: {errors}")
    print(f"Tổng kết quả đã gộp: {compiled}")
    print(f"Thời gian: {elapsed / 60:.2f} phút")
    if success + blank:
        print(f"Tốc độ: {elapsed / (success + blank):.2f} giây/ảnh")
    print(f"JSONL resume: {args.output_jsonl}")
    print(f"JSON tổng hợp: {args.output_json}")
    if unresolved_errors:
        print(f"Còn {unresolved_errors} ảnh lỗi (sẽ được thử lại lần sau): {args.error_log}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
