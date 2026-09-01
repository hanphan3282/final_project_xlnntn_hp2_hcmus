# Pipeline tạo ground truth ảnh Facebook

Pipeline này tạo ground truth đề xuất cho ảnh Facebook (chữ Hán thư pháp/viết tay)
từ ba nguồn bằng một flow có kiểm soát:

## Stage 1: OCR & Candidate Generation

```text
data/input/valid.jsonl
  ├─ caption gốc
  └─ URL ảnh
       │
       ├─ Bước 1: Fetch ảnh
       │      ↓
       │   data/input/Images/
       │      │
       │      └─ Bước 2: Gemini Vision OCR
       │             ↓
       │   data/output/mrDuc_data_ocr/
       │   └─ facebook_posts_ocr.jsonl
       │
       └─ Bước 3: So sánh caption ↔ Gemini
              ├─ data/output/Gemini_same_Label/
              │    └─ record + ảnh
              │
              └─ data/output/Gemini_diff_Label/
                   └─ record + ảnh
                        ↓
                   Bước 4: PP-OCRv6 CPU
                        ↓
                   data/output/Gemini_diff_Label/
                   └─ paddle_v6/
                      └─ new_labels.jsonl

Chạy mọi lệnh từ thư mục gốc repository.
```

## Stage 2: Multi-source Adjudication & Ground Truth
```text
Input sources:

1. data/input/valid.jsonl
   ├─ Original image
   └─ Original caption

2. data/output/mrDuc_data_ocr/facebook_posts_ocr.jsonl
   └─ Gemini OCR

3. data/output/Gemini_diff_Label/paddle_v6/new_labels.jsonl
   └─ PP-OCRv6 OCR


Original Image
     +
Original Caption
     +
Gemini OCR
     +
PP-OCRv6 OCR
     ↓
Bước 5: Strict Join by image/post_id
     ↓
Vision LLM Adjudication (DeepSeek)
     ↓
data/output/DeepSeek_ground_truth/adjudications.jsonl
     ↓
Bước 6: Lọc valid / invalid
     ├─ adjudications_valid.jsonl
     └─ adjudications_invalid.jsonl
              ↓
Bước 7: Tổng hợp ground truth
              ↓
data/output/DeepSeek_ground_truth/ground_truth.jsonl
              ↓
Bước 8: Copy ảnh ground truth
              ↓
data/ground_truth_images/
```
## 1. Cấu trúc code

```text
scripts/
├── 1_fetch_images.py               Tải và kiểm tra ảnh từ valid.jsonl; resume và retry
├── 2_ocr_gemini.py                 Entry point OCR bằng vision API (OpenAI-compatible)
├── 3_compare_gemini_label.py       Tính metric, chia same/diff, materialize ảnh
├── 4_paddlev6_relabel_diff.py      OCR lại nhóm diff bằng PP-OCRv6 CPU
├── 5_llm_adjudicate_ground_truth.py  Join ba nguồn, nhờ vision LLM chọn ground truth
├── 6_ground_truth.py               Tổng hợp ground_truth.jsonl từ adjudications_valid
├── split_adjudications.py          Tách adjudications.jsonl → valid / invalid
├── copy_ground_truth_images.py     Copy ảnh có ground truth vào ground_truth_images/
├── lib/
│   ├── vision_ocr.py               Engine API: resize, resume, retry, JSON validation
│   └── paddle_v6_cpu.py            Engine PP-OCRv6 CPU: baseline + fallback invert/CLAHE
└── experiments/
    └── compare_paddle_preprocessing.py  Đối chứng preprocessing trên mẫu phân tầng
```

## 2. Cấu trúc dữ liệu

```text
data/
├── input/
│   ├── valid.jsonl                 29 045 post, mỗi dòng: post_id, label, images[]
│   └── Images/                     ~69 500 ảnh .jpg ({post_id}_{index}.jpg)
├── output/
│   ├── mrDuc_data_ocr/
│   │   ├── facebook_posts_ocr.jsonl    21 482 kết quả Gemini OCR
│   │   ├── fetch_errors.jsonl
│   │   ├── fetch_summary.json
│   │   ├── ocr_errors.jsonl
│   │   └── ocr_summary.json
│   ├── Gemini_same_Label/
│   │   ├── records.jsonl           2 030 record (caption ≈ Gemini OCR)
│   │   ├── comparison.tsv
│   │   ├── summary.json
│   │   └── Images/                 bản sao ảnh vật lý (tạo bởi bước 3)
│   ├── Gemini_diff_Label/
│   │   ├── records.jsonl           19 452 record (caption ≠ Gemini OCR)
│   │   ├── comparison.tsv
│   │   ├── summary.json
│   │   ├── Images/                 bản sao ảnh vật lý (tạo bởi bước 3)
│   │   └── paddle_v6/
│   │       ├── new_labels.jsonl    19 452 nhãn đề xuất từ Paddle
│   │       ├── ocr_results.jsonl
│   │       ├── ocr_results.json
│   │       ├── ocr_errors.jsonl
│   │       └── summary.json
│   ├── DeepSeek_ground_truth/
│   │   ├── adjudications.jsonl     13 097 kết quả adjudication thô
│   │   ├── adjudications.json
│   │   ├── adjudications_valid.jsonl   9 058 (69.2%) — valid ground truth
│   │   ├── adjudications_invalid.jsonl 4 039 (30.8%) — lọc bỏ
│   │   ├── ground_truth.jsonl      9 058 record ground truth cuối
│   │   ├── errors.jsonl
│   │   └── summary.json
│   └── final_mid_project/
│       └── HVH_001/ … HVH_032/    32 văn bản Han-Nom, mỗi thư mục có:
│           ├── HVH_NNN_raw.txt
│           └── HVH_NNN_seg.tsv
├── ground_truth_images/            9 058 ảnh có ground truth (copy từ input/Images)
configs/
├── corpus_catalog.csv              Danh mục corpus
└── ocr_policy.json                 Cấu hình chính sách OCR
```

`Gemini_same_Label/Images` và `Gemini_diff_Label/Images` chứa file ảnh vật lý,
không phải symlink hoặc hardlink. Bước 3 tạo các bản sao này theo mặc định.

## 3. Ràng buộc dữ liệu và an toàn join

1. Khóa ảnh chuẩn là `{post_id}_{image_index}.jpg` sinh từ `valid.jsonl`;
   không join theo thứ tự dòng hoặc fuzzy filename.
2. `image` và `post_id` phải khớp tuyệt đối trong caption, Gemini, tập diff và
   PaddleV6. Thiếu, trùng hoặc khác khóa làm bước 5 dừng trước khi gọi API.
3. Caption và Gemini trong file dẫn xuất phải giống tuyệt đối nguồn gốc.
4. Mỗi request bước 5 lưu `join_fingerprint` SHA-256 của khóa và ba văn bản.
   Resume chỉ hợp lệ khi fingerprint, provider, model và evidence mode không đổi.
5. `--no-image` không tạo ground truth đã xác minh bằng ảnh; mọi record bị đánh
   dấu `needs_review=true`.
6. Không dùng confidence do Paddle/LLM tự báo như accuracy hoặc CER/WER. Muốn
   công bố CER/WER cần bản chép chuẩn do người đọc xác nhận.
7. Không dùng `--overwrite` khi chỉ muốn resume. Tùy chọn này xóa kết quả cũ.

## 4. Cấu hình môi trường

```bash
conda env create -f environment.yml
conda activate NLP
cp .env.example .env   # chỉnh sửa API key trước khi chạy
```

Ví dụ cấu hình `.env`:

```dotenv
OCR_API_KEY="replace_me"
OCR_BASE_URL="https://provider.example/v1"
OCR_MODEL="gemini-vision-model"

DEEPSEEK_API_KEY="replace_me"
DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
DEEPSEEK_MODEL="deepseek-vision-model"
```

Nếu dùng nhiều API key song song, đặt `DEEPSEEK_API_KEYS` (phân cách bằng dấu phẩy).
Không commit `.env`.

## 5. Chạy từng bước

### Bước 1 — Fetch ảnh

```bash
# Kiểm tra kế hoạch
python scripts/1_fetch_images.py --dry-run

# Chạy đầy đủ
python scripts/1_fetch_images.py --workers 24 --retries 4 --report-every 100
```

Tham số quan trọng:
- `--workers`: request đồng thời (I/O-bound, có thể lớn hơn số CPU).
- `--retries`: số lần thử lại lỗi mạng/5xx/429.
- `--verify-existing`: mở và kiểm tra ảnh đã có (mặc định chỉ kiểm tra size).
- `--max-fail-rate`: dừng cấp task mới khi tỷ lệ lỗi quá cao.
- `--overwrite`: tải lại ảnh đã tồn tại; không dùng khi resume bình thường.

Output: `data/input/Images/{post_id}_{image_index}.jpg`

### Bước 2 — Gemini vision OCR

```bash
# Pilot 10 ảnh
python scripts/2_ocr_gemini.py --workers 4 --limit 10 --retries 4 --timeout 240 --report-every 1

# Resume toàn bộ
python scripts/2_ocr_gemini.py --workers 32 --retries 4 --timeout 240 \
  --max-fail-rate 0.10 --fail-rate-min-samples 50 --report-every 50
```

`--workers` bị giới hạn bởi rate limit của provider. Tăng dần `4 → 8 → 16 → 32`;
giảm nếu xuất hiện 429/5xx. Output JSONL flush từng record và hỗ trợ resume.

Output: `data/output/mrDuc_data_ocr/facebook_posts_ocr.jsonl`

### Bước 3 — So sánh và chia nhóm

```bash
python scripts/3_compare_gemini_label.py --dry-run
python scripts/3_compare_gemini_label.py --threshold 0.58 --min-common-han 4
```

Mặc định chép ảnh thật vào hai thư mục `Images/`. Dùng `--no-copy-images` khi
không cần dataset tự chứa.

Quy tắc phân nhóm:
- Chuỗi ≤ 3 ký tự: `same` chỉ khi khớp tuyệt đối.
- Một chuỗi chứa chuỗi còn lại: coverage ngắn ≥ 0,90 và đủ chữ Hán chung.
- Coverage một chiều ≥ 0,72, Character F1 ≥ 0,55 và đủ chữ Hán chung.
- Điểm kết hợp ≥ `--threshold` và đủ `--min-common-han`.
- Gemini rỗng → luôn vào `diff`.

Output: `data/output/Gemini_same_Label/` và `data/output/Gemini_diff_Label/`

### Bước 4 — PP-OCRv6 relabel nhóm diff

```bash
python scripts/4_paddlev6_relabel_diff.py --dry-run
python scripts/4_paddlev6_relabel_diff.py \
  --workers 4 --cpu-threads 13 --score-threshold 0.30 --fallback-confidence 0.65
```

- `workers × cpu-threads` nên gần số lõi vật lý, không phải logical CPU.
- Pipeline luôn OCR `original` trước; chỉ thử một biến thể thích nghi khi
  original rỗng/confidence thấp: `invert` (nền tối/chữ sáng) hoặc `CLAHE`
  (chữ mờ). Fallback chỉ được chọn khi mạnh hơn original.
- Output `new_labels.jsonl` là nhãn đề xuất, chưa phải ground truth.

Output: `data/output/Gemini_diff_Label/paddle_v6/`

### Bước 5 — Vision LLM adjudication

```bash
# Kiểm tra strict join, không gọi API
python scripts/5_llm_adjudicate_ground_truth.py --provider deepseek --dry-run

# Pilot 10 ảnh
python scripts/5_llm_adjudicate_ground_truth.py \
  --provider deepseek --workers 4 --max-tokens 2000 \
  --retries 4 --timeout 240 --limit 10 --report-every 1

# Resume toàn bộ
python scripts/5_llm_adjudicate_ground_truth.py \
  --provider deepseek --workers 4 --max-tokens 8192 \
  --retries 4 --timeout 240 \
  --max-fail-rate 0.10 --fail-rate-min-samples 50 --report-every 50
```

Script gửi ảnh cùng ba ứng viên `caption`, `gemini`, `paddle_v6`. Model trả
`selected_source`, bản chép trực quan, ground truth đề xuất, điểm từng nguồn,
confidence, quan hệ caption, reason codes và `needs_review`.

Output: `data/output/DeepSeek_ground_truth/adjudications.jsonl`

### Bước 6 — Tách valid / invalid

```bash
python scripts/split_adjudications.py
```

Điều kiện invalid: confidence < 0,75; ground truth chỉ là "Reels"; selected_source
là `blank` (không có chữ) hoặc `uncertain` (ảnh mờ); hoặc `image_unclear` trong
reason_codes.

Output: `adjudications_valid.jsonl` (9 058) và `adjudications_invalid.jsonl` (4 039)

### Bước 7 — Tổng hợp ground truth

```bash
python scripts/6_ground_truth.py
```

Đọc `adjudications_valid.jsonl`, ghi `ground_truth.jsonl` theo schema thống nhất:
`image`, `ground_truth`, `label` (Paddle), `deepseek` (cùng cấu trúc với gemini).

Output: `data/output/DeepSeek_ground_truth/ground_truth.jsonl`

### Bước 8 — Copy ảnh ground truth

```bash
python scripts/copy_ground_truth_images.py --dry-run   # xem trước
python scripts/copy_ground_truth_images.py             # thực thi
```

Copy đúng 9 058 ảnh được tham chiếu trong `ground_truth.jsonl` vào
`data/ground_truth_images/`, thay vì commit toàn bộ ~69 500 ảnh thô.

## 6. Metric đánh giá (bước 3)

| Metric | Ý nghĩa | Khoảng |
|---|---|---:|
| Sequence similarity | Mức giống theo thứ tự ký tự (SequenceMatcher) | 0–1 |
| Character F1 | F1 trên số lần xuất hiện ký tự | 0–1 |
| Bigram Dice | Mức trùng các cặp hai ký tự liên tiếp | 0–1 |
| OCR coverage | Phần OCR được bao phủ bởi chuỗi khớp theo thứ tự | 0–1 |
| Label coverage | Phần caption được bao phủ bởi chuỗi khớp theo thứ tự | 0–1 |
| Common Han | Số chữ Hán chung có tính số lần xuất hiện | nguyên |
| Similarity score | `0.30×Seq + 0.25×CharF1 + 0.25×shortCoverage + 0.20×Bigram` | 0–1 |

Bước 5 đặt `needs_review=true` nếu:
- Không gửi ảnh (`--no-image`).
- Chọn `merged` hoặc `uncertain`.
- LLM confidence < 0,85; điểm nguồn cao nhất < 0,75; hoặc margin < 0,08.

## 7. Resume, lỗi và kiểm tra nhanh

- JSONL là nguồn resume; JSON là bản compile để đọc/chia sẻ.
- HTTP 429/5xx có retry; lỗi không thể retry (e.g. `model_not_found`) dừng ngay.
- Luôn đọc `summary.json` sau mỗi bước: `succeeded`, `failed`, `done`,
  `needs_review`, tốc độ và token usage.

Kiểm tra số dòng:

```bash
wc -l \
  data/output/mrDuc_data_ocr/facebook_posts_ocr.jsonl \
  data/output/Gemini_same_Label/records.jsonl \
  data/output/Gemini_diff_Label/records.jsonl \
  data/output/Gemini_diff_Label/paddle_v6/new_labels.jsonl \
  data/output/DeepSeek_ground_truth/adjudications.jsonl \
  data/output/DeepSeek_ground_truth/adjudications_valid.jsonl \
  data/output/DeepSeek_ground_truth/ground_truth.jsonl
```

Kiểm tra ảnh bước 3 (không được là symlink):

```bash
find data/output/Gemini_same_Label/Images -type f | wc -l
find data/output/Gemini_diff_Label/Images -type f | wc -l
find data/output/Gemini_same_Label/Images -type l | wc -l
find data/output/Gemini_diff_Label/Images -type l | wc -l
```

## 8. Git và dữ liệu lớn

Được commit: code trong `scripts/`, `configs/`, README, `environment.yml`,
`.gitignore` và config không chứa bí mật.

Không commit: `.env`, `data/input/Images/`, `data/output/`, `data/ground_truth_images/`,
model cache và file tạm. Dữ liệu chia sẻ qua Google Drive và kiểm tra bằng SHA256SUMS.
