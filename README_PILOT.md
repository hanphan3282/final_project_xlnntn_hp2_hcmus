# Hướng dẫn chạy pilot bằng Nextflow

Tài liệu này hướng dẫn chạy thử pipeline end-to-end với manifest pilot gồm
**8 bài đăng và 10 ảnh**. Có thể chạy bằng URL còn hiệu lực hoặc dùng ảnh đã
chuẩn bị sẵn khi URL Facebook hết hạn.

## 1. Đầu vào cần chuẩn bị

Manifest bắt buộc của pipeline pilot là:

```text
data-pilot/input/valid.jsonl
```

Mỗi dòng là một JSON object, tối thiểu có cấu trúc:

```json
{"post_id":"ma_bai_dang","label":"nhan_goc","images":["https://url-anh-1", "https://url-anh-2"]}
```

File pilot hiện tại đã được rút gọn còn đúng 10 URL. Không cần các file
`valid.full-48.jsonl` hoặc `pilot_selection_mapping.json` để chạy Nextflow.

Không cần tự đặt tên ảnh. Stage 1 tự tải và lưu ảnh theo quy tắc:

```text
{post_id}_{thu_tu_url_trong_images}.jpg
```

Ví dụ URL đầu tiên trong `images` được lưu thành `post_id_0.jpg`.

Nếu URL đã hết hạn, có thể đặt 10 ảnh đã tải sẵn vào:

```text
data-pilot/input/Images/
```

Tên ảnh local phải khớp quy tắc trên. Bộ ảnh pilot trong dự án hiện đã được
chuẩn hóa tên để khớp với `valid.jsonl`.

## 2. Cài môi trường Pixi

Mở terminal ngay tại thư mục chứa file `main.nf`, rồi cài môi trường:

```bash
pixi install --locked
```

Tất cả đường dẫn trong tài liệu đều tương đối tính từ thư mục đang chứa
`main.nf`; không cần tạo thêm hoặc đổi tên thư mục.

Lệnh trên cài đúng các phiên bản đã khóa trong `pixi.lock`, bao gồm Python,
Java, Nextflow, PaddlePaddle và các thư viện Python. Không cài thêm bằng `pip`
vào môi trường này.

Kiểm tra môi trường và cấu hình Nextflow:

```bash
pixi run check
```

## 3. Cấu hình API

Nếu chỉ chạy stage 1 để tải ảnh thì chưa cần API key. Nếu chạy end-to-end,
tạo file `.env`:

```bash
cp .env.example .env
```

Sau đó mở `.env` và điền cấu hình thật:

```dotenv
OCR_API_KEY="..."
OCR_BASE_URL="..."
OCR_MODEL="..."

DEEPSEEK_API_KEY="..."
DEEPSEEK_BASE_URL="..."
DEEPSEEK_MODEL="..."
```

Không đưa `.env` hoặc API key lên GitHub.

## 4. Kiểm tra manifest trước khi tải

Kiểm tra số record và URL bằng thư viện chuẩn của Python:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("data-pilot/input/valid.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
urls = sum(len(row.get("images", [])) for row in rows)
print(f"Records: {len(rows)}")
print(f"Image URLs: {urls}")
PY
```

Kết quả mong đợi:

```text
Records: 8
Image URLs: 10
```

Sau khi cài Pixi, có thể kiểm tra kế hoạch tải mà không gọi mạng:

```bash
pixi run python scripts/1_fetch_images.py \
  --input data-pilot/input/valid.jsonl \
  --images-dir data-pilot/input/Images \
  --limit 10 \
  --workers 4 \
  --dry-run
```

## 5. Chọn cách cung cấp ảnh

### Cách A — Tải từ URL còn hiệu lực

Nếu muốn kiểm tra quá trình tải thật sự cả 10 URL từ đầu nhưng thư mục
`data-pilot/input/Images` đã chứa ảnh, hãy đổi tên thư mục cũ để giữ bản sao:

```bash
mv data-pilot/input/Images data-pilot/input/Images.original-10
mkdir -p data-pilot/input/Images
```

Chỉ thực hiện hai lệnh trên một lần. Nếu `Images.original-10` đã tồn tại thì
không chạy lại lệnh `mv`; hãy chọn một tên backup khác.

Chạy stage 1 bằng Nextflow:

```bash
pixi run nextflow run main.nf -profile pilot \
  --from_stage 1 \
  --to_stage 1
```

Ảnh tải thành công nằm tại:

```text
data-pilot/input/Images/
```

Tóm tắt và URL tải lỗi nằm tại:

```text
data-pilot/output/mrDuc_data_ocr/fetch_summary.json
data-pilot/output/mrDuc_data_ocr/fetch_errors.jsonl
```

Downloader tự bỏ qua ảnh hợp lệ đã tồn tại. Vì vậy nếu không làm trống thư
mục `Images`, số ảnh tải mới có thể nhỏ hơn 10 dù manifest vẫn chứa 10 ảnh.

### Cách B — Dùng ảnh đã chuẩn bị sẵn khi URL hết hạn

Đặt đủ 10 ảnh đúng tên vào:

```text
data-pilot/input/Images/
```

Kiểm tra trước khi chạy:

```bash
pixi run python scripts/1_fetch_images.py \
  --input data-pilot/input/valid.jsonl \
  --images-dir data-pilot/input/Images \
  --limit 10 \
  --workers 4 \
  --dry-run
```

Kết quả mong đợi là:

```text
Tham chiếu ảnh: 10
Đã có hợp lệ: 10
Cần tải: 0
```

Sau đó vẫn có thể chạy từ stage 1 để báo cáo Nextflow thể hiện đủ toàn bộ DAG:

```bash
pixi run nextflow run main.nf -profile pilot \
  --from_stage 1 \
  --to_stage 8
```

Ở chế độ này, stage 1 kiểm tra và ghi nhận 10 ảnh local đã tồn tại, tạo
`fetch_summary.json` với `already_complete=10`, rồi chuyển sang OCR. Stage 1
không gọi các URL đã hết hạn.

Nếu chỉ muốn bắt đầu trực tiếp từ OCR, dùng:

```bash
pixi run nextflow run main.nf -profile pilot \
  --from_stage 2 \
  --to_stage 8
```

Để chạy từ stage 2, thư mục `Images` phải chứa đủ ảnh đúng tên. Cách chạy từ
stage 1 được khuyến nghị khi cần trình bày quy trình end-to-end với giảng viên.

## 6. Chạy pilot end-to-end

Sau khi `.env` đã có cấu hình thật, chạy toàn bộ 8 stage:

```bash
pixi run nextflow run main.nf -profile pilot \
  --from_stage 1 \
  --to_stage 8
```

Các stage gồm:

1. Tải ảnh từ URL trong `valid.jsonl`, hoặc xác nhận ảnh local đã có.
2. OCR ảnh bằng Gemini hoặc endpoint OCR tương thích.
3. So sánh kết quả OCR với label ban đầu.
4. Chạy PP-OCRv6 cho các trường hợp khác label (PaddleOCR 3.7.0).
5. LLM phân xử và đề xuất ground truth.
6. Tách kết quả hợp lệ và không hợp lệ.
7. Tạo `ground_truth.jsonl`.
8. Sao chép các ảnh ground truth sang thư mục kết quả.

Mặc định stage 5 dùng provider `deepseek`. Nếu đã cấu hình Qwen trong `.env`,
có thể thay bằng:

```bash
pixi run nextflow run main.nf -profile pilot \
  --provider qwen \
  --from_stage 1 \
  --to_stage 8
```

Ở lần chạy stage 4 đầu tiên, PaddleOCR tự tải model PP-OCRv6 chính thức vào
`models/paddlex/`. Máy cần kết nối Internet cho lần tải này; các lần chạy sau
sẽ dùng model trong cache. Nếu stage 1–3 đã hoàn tất và pipeline từng dừng ở
stage 4, tiếp tục mà không chạy lại ba stage đầu:

```bash
pixi run nextflow run main.nf -profile pilot \
  --provider qwen \
  --from_stage 4 \
  --to_stage 8
```

## 7. Kết quả cần kiểm tra

Với provider DeepSeek, các kết quả chính nằm tại:

```text
data-pilot/output/mrDuc_data_ocr/facebook_posts_ocr.jsonl
data-pilot/output/Gemini_same_Label/records.jsonl
data-pilot/output/Gemini_diff_Label/records.jsonl
data-pilot/output/Gemini_diff_Label/paddle_v6/new_labels.jsonl
data-pilot/output/DeepSeek_ground_truth/adjudications.jsonl
data-pilot/output/DeepSeek_ground_truth/adjudications_valid.jsonl
data-pilot/output/DeepSeek_ground_truth/adjudications_invalid.jsonl
data-pilot/output/DeepSeek_ground_truth/ground_truth.jsonl
data-pilot/ground_truth_images/
```

Với `--provider qwen`, các file adjudication và ground truth tương ứng nằm
trong `data-pilot/output/Qwen38_ground_truth/`.

Nextflow tạo báo cáo theo dõi tại:

```text
reports/timeline.html
reports/report.html
reports/trace.tsv
reports/dag.html
```

## 8. Chạy lại và xử lý lỗi

### Thiếu package như `requests`

Môi trường Pixi chưa được cài đầy đủ. Chạy:

```bash
pixi install --locked
```

### URL Facebook trả về HTTP 403/404

URL CDN của Facebook có thể hết hạn. Cần crawl lại URL mới rồi cập nhật trường
`images` trong `valid.jsonl` nếu muốn kiểm thử tải qua mạng. Nếu chỉ cần chạy
pilot end-to-end, đặt ảnh đã chuẩn bị sẵn đúng tên vào
`data-pilot/input/Images/`; stage 1 sẽ sử dụng ảnh local và không gọi URL cũ.
Việc đổi tên file ảnh không làm URL hết hạn.

### Thiếu API key hoặc model

Kiểm tra lại `.env`. Stage 2 cần nhóm biến `OCR_*`; stage 5 cần nhóm biến
`DEEPSEEK_*` hoặc `QWEN_*` tương ứng với `--provider`.

### PaddleOCR báo `Invalid OCR version: PP-OCRv6`

Lỗi này cho biết môi trường cũ vẫn đang dùng PaddleOCR 3.4.0. PP-OCRv6 yêu cầu
PaddleOCR 3.7.0 trong dự án này. Đồng bộ lại môi trường đã khóa rồi kiểm tra:

```bash
pixi install --locked
pixi run python -c "import paddleocr; print(paddleocr.__version__)"
```

Kết quả phiên bản phải là `3.7.0`.

### PaddleOCR không tải được model

Kiểm tra kết nối tới Hugging Face, AI Studio hoặc BOS rồi chạy lại stage 4.
Model chỉ cần tải một lần và được cache tại `models/paddlex/`.

### Chạy từ một stage ở giữa

Có thể giới hạn khoảng stage, ví dụ:

```bash
pixi run nextflow run main.nf -profile pilot \
  --from_stage 4 \
  --to_stage 8
```

Khi chạy từ stage ở giữa, các file đầu ra hợp lệ của những stage trước phải có
sẵn trong `data-pilot/`. Nếu chưa có, hãy chạy lại từ stage 1.

### Dừng pipeline

Nhấn `Ctrl+C`. Các script có cơ chế giữ kết quả đã hoàn tất theo từng record;
khi chạy lại, ảnh và record hợp lệ đã có thường được bỏ qua thay vì tải/gọi API
lại.
