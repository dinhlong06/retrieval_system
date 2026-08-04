# Keyframe Pipeline

Package `keyframe_pipeline` đọc keyframe ảnh của layer 2 và tạo:

1. **SigLIP** — embedding 1152 chiều cho mỗi keyframe (chạy trong Docker, GPU);
2. **ReCap** — caption tiếng Anh fine-grained + OCR đã hiệu đính cho mỗi keyframe
   bằng Qwen3-VL qua Ollama HTTP, dùng metadata OCR/object của Layer 3 làm gợi ý.

Hai nhánh độc lập nhau. Chỉ nhánh SigLIP nằm trong image; ReCap gọi Ollama qua
HTTP nên cần một Ollama server chạy sẵn.

## 1. Chạy SigLIP bằng Docker (đường chạy chính)

```bash
./run_siglip.sh                       # embed mọi video trong pipeline_c
./run_siglip.sh --overwrite           # ghi đè .npy đã có
./run_siglip.sh --batch-size 64
PIPELINE=pipeline_a ./run_siglip.sh   # đổi pipeline nguồn của layer 2
FRAMES_DIR=/path/khac ./run_siglip.sh # bỏ qua PIPELINE, chỉ thẳng thư mục frame
GPU_ID=5 ./run_siglip.sh              # ghim GPU thay vì tự chọn
```

- Input mặc định: `layer_2/Keyframe_Extracting/benchmark/<PIPELINE>/`.
- Output: `artifacts/siglip/<VIDEO_ID>.npy` + `<VIDEO_ID>_ids.json`.
- Weight SigLIP (~3,3 GB) nằm ở `.model_cache/huggingface` và được bind-mount vào
  container, không nhét vào image — ổ `/` của host gần đầy. Lần chạy đầu tải model,
  các lần sau dùng lại.
- Script tự chọn GPU còn nhiều VRAM nhất qua `nvidia-smi`. Đây là máy dùng chung,
  đừng mặc định GPU 0.

## 2. Input

Mỗi thư mục video chứa trực tiếp các ảnh keyframe:

```text
<dataset-root>/
└── K01_V001/
    ├── 001.jpg
    ├── 002.jpg
    └── ...
```

- Hỗ trợ `.jpg`, `.jpeg`, `.png`, `.webp`.
- `video_id` là tên thư mục, phải khớp `[A-Z]+<số>_V<số>` (`K01_V001`, `L21_V001`).
- `keyframe_id` là tên file không có extension, ví dụ `001`.
- Ảnh sắp xếp natural order; input không bị sửa hoặc di chuyển.
- `--dataset-root` nhận cả thư mục chứa trực tiếp các thư mục video, lẫn thư mục
  cha có `keyframes/` bên trong.

## 3. Output

```text
artifacts/
├── siglip/
│   ├── K01_V001.npy
│   └── K01_V001_ids.json
└── recap/
    └── K01_V001.jsonl
```

### SigLIP

`K01_V001.npy` là NumPy array:

- shape `(N, 1152)`, dtype `float32`, mỗi hàng đã L2-normalize;
- hàng thứ `i` ứng với ID thứ `i` trong `K01_V001_ids.json`.

File ID là list phẳng, `video_id` suy ra từ tên file:

```json
["001","002","003"]
```

Đây đúng format mà `layer_2/Keyframe_Extracting` ghi embedding BEiT-3
(`src/core/runner.py::_save_embeddings`): cùng tên file, dtype, kiểu JSON, nên một
loader đọc được cả hai. Khác duy nhất là số chiều — SigLIP SO400M ra 1152, BEiT-3
Large ra 1024, nên vector hai bên không so cosine chéo với nhau được.

### ReCap

`K01_V001.jsonl` có đúng một JSON object mỗi dòng, thứ tự khớp thứ tự ảnh và
embedding:

```json
{"video_id":"K01_V001","keyframe_id":"K01_V001_000000_kf0001","caption":"A vibrant sunset over a city skyline with tall buildings illuminated against an orange sky, reflected in a calm river. A small boat sits near the center of the frame.","corrected_ocr":["HTV7 HD","18:29:57"]}
```

Đúng bốn field, không có đường tương thích ngược với schema 3 field cũ:

- `caption` — tiếng Anh, 60–100 từ, 2–4 câu, chỉ mô tả cái nhìn thấy được.
- `corrected_ocr` — list string duy nhất, **giữ nguyên ngôn ngữ gốc** (không dịch),
  là OCR đã được model đối chiếu lại với ảnh. `[]` khi không xác minh được chữ nào.

Metadata Layer 3 join theo `frame_id == keyframe_id`. Keyframe thiếu record vẫn
caption bình thường với hint rỗng; Layer 3 không bao giờ bị ghi đè.

**Frame lỗi không giết cả video.** Frame nào model trả JSON không parse được thì
bị bỏ qua, ghi vào `<VIDEO_ID>.failed.jsonl` cạnh output:

```json
{"video_id":"K01_V001","keyframe_id":"K01_V001_000072_kf0001","error":"Ollama returned invalid structured JSON"}
```

Số frame lỗi có trong summary JSON của CLI (`"failed": N`). Nếu **mọi** frame đều
lỗi thì vẫn raise. Hệ quả: JSONL có thể ngắn hơn số dòng embedding SigLIP của
cùng video — downstream phải join theo `keyframe_id`, không theo vị trí dòng.

## 4. CLI

Entrypoint của image là `python3 -m keyframe_pipeline`, nên mọi subcommand đều
chạy được qua `docker run ... ai26-siglip <subcommand>`:

| Subcommand | Việc |
|---|---|
| `siglip-dataset` | embed cả dataset (`run_siglip.sh` gọi cái này) |
| `siglip-video` | embed một video |
| `recap-video` | caption một video, tái dùng `.npy`/`_ids.json` đã có |
| `recap-dataset` | caption cả dataset |

`recap-video`, `recap-dataset` và `run` nhận thêm `--ocr-json` / `--objects-json`
(tuỳ chọn). Hai file này parse **một lần cho cả dataset**, không phải mỗi video.
`siglip-*` không có hai cờ này.
| `run` | chạy lần lượt SigLIP rồi ReCap cho cả dataset |

Ví dụ một video:

```bash
docker run --rm --gpus device=0 \
  -v /path/frames:/data/frames:ro -v "$PWD/artifacts:/data/output" \
  -v "$PWD/.model_cache/huggingface:/root/.cache/huggingface" \
  -e HF_HOME=/root/.cache/huggingface \
  ai26-siglip siglip-video \
  --keyframe-dir /data/frames/K01_V001 --output-dir /data/output/siglip \
  --device cuda:0 --batch-size 32
```

`scripts/demo_five.py` chạy model thật trên `--limit` ảnh đầu của một video để
smoke-test nhanh cả hai nhánh (không nằm trong image, cần cài package tại chỗ).

## 5. ReCap / Ollama

Model mặc định: `qwen3-vl:8b-instruct` (6.1 GB, ~7.6 GB VRAM khi nạp), host mặc
định `http://127.0.0.1:11501`. Request tắt thinking, `temperature=0`,
`num_predict=768`, và ép structured output bằng JSON schema `{caption,
corrected_ocr}`.

```bash
OLLAMA_HOST=127.0.0.1:11501 ollama pull qwen3-vl:8b-instruct
OLLAMA_HOST=127.0.0.1:11501 ollama show qwen3-vl:8b-instruct   # phải có capability: vision
```

Ba cái bẫy khi đổi model:

- **Phải dùng tag `-instruct`.** `qwen3-vl:8b` trần trỏ cùng digest với
  `qwen3-vl:8b-thinking`; bản thinking đốt sạch `num_predict` vào `<think>` rồi
  trả caption rỗng. Đúng vậy với cả `2b` và `4b`.
- **Model phải có capability `vision`.** Backend preflight bằng `show()` và raise
  `ModelConfigurationError` trước request đầu tiên nếu thiếu. Ví dụ `qwen3:8b`
  (text-only) bị chặn ngay tại đây.
- **Có vision chưa đủ, phải tôn trọng `format` schema.** `qwen3.5:9b` đã được đo
  thử 2026-07-27: có vision nhưng bọc output trong ```json fence và trả
  `corrected_ocr` là string → `json.loads` fail, và chậm hơn (60–63 vs 76–79
  tok/s). Vì output hỏng là `ModelInferenceError` cứng, model đó fail gần như mọi
  frame. Manifest **không** chứng minh được có/không vision (qwen3.5 nhúng vision
  trong một GGUF, `model_families` vẫn chỉ `["qwen35"]`) — chỉ `/api/show` mới
  đáng tin.

Host: `:11434` là service hệ thống và đang rỗng model; `:11501` là server có weight.
Đổi bằng `--ollama-host` nếu chạy chỗ khác.

**Chạy ReCap trong container**: docker ở máy này là rootless + slirp, nên
`--network host` và `172.17.0.1` đều fail; host reachable tại **`10.0.2.3`**.
Khởi động server bằng:

```bash
OLLAMA_HOST=0.0.0.0:11501 OLLAMA_KEEP_ALIVE=-1 CUDA_VISIBLE_DEVICES=<gpu trống> \
  nohup ollama serve > ~/ollama_11501.log 2>&1 &
```

rồi truyền `--ollama-host http://10.0.2.3:11501`. Ảnh được SDK đọc và base64 ở
phía client nên server không cần thấy đường dẫn file.

**Luôn kiểm tra VRAM trước khi chạy batch.** Nếu ollama bị ghim vào GPU đang bận,
nó offload một phần sang CPU mà không báo lỗi: đo được 3.5/7.6 GB trên GPU →
1.6 tok/s (~160 s/frame) thay vì ~76 tok/s (~5 s/frame). `curl -s
http://127.0.0.1:11501/api/ps` phải cho `size_vram` xấp xỉ `size`.

Lệnh production:

```bash
python3 -m keyframe_pipeline recap-dataset \
  --dataset-root ../layer_2/Keyframe_Extracting/benchmark/pipeline_c \
  --ocr-json ../layer_3/OCR/output/output.json \
  --objects-json ../layer_3/ObjectDetection/output/detections.json \
  --output-dir artifacts/recap \
  --ollama-model qwen3-vl:8b-instruct \
  --ollama-host http://127.0.0.1:11501
```

**Giới hạn đã biết của `corrected_ocr`**: với chữ cách điệu / bán trong suốt,
model đoán một giá trị hợp lý thay vì bỏ qua như prompt yêu cầu. Logo thật
"60 giây" bị đọc thành "30 giây"; PaddleOCR ra "ungiây". Chữ overlay thường
(tên kênh, timestamp) thì chính xác. Đừng coi `corrected_ocr` trên chữ đồ hoạ là
đã được xác minh.

VRAM: 8× RTX 2080 Ti 11 GB dùng chung, có lúc chỉ còn ~3 GB rảnh mỗi con. Model
chiếm 7.6 GB khi nạp — cần một GPU đủ trống, nếu không Ollama đẩy một phần sang CPU
và chậm hẳn. Backend đặt `keep_alive="10m"` nên caption xong nó tự nhả.

Đo thực tế (3 keyframe `K01_V001`, 100% GPU): 86 s cho ảnh đầu vì phải nạp model,
sau đó ~4 s/ảnh.

Đổi host bằng `--ollama-host` (ví dụ khi Ollama chạy trên máy khác).

## 6. Public Python API

Module khác import trực tiếp, không cần subprocess/CLI:

```python
from pathlib import Path

from keyframe_pipeline import KeyframePipeline, discover_video

video = discover_video(Path("/path/frames/K01_V001"))
pipeline = KeyframePipeline(siglip_device=None)  # None = tự chọn CUDA nếu có

siglip = pipeline.run_siglip(
    video.keyframes,
    output_dir=Path("artifacts/siglip"),
    batch_size=32,
)
recap = pipeline.run_recap(
    video.keyframes,
    previous_output=siglip,
    output_path=Path("artifacts/recap/K01_V001.jsonl"),
)

print(siglip.embeddings.shape)   # (N, 1152)
print(recap.records[0].caption)
```

Đọc lại artifact đã lưu:

```python
from keyframe_pipeline import load_recap_result, load_siglip_result

siglip = load_siglip_result(
    Path("artifacts/siglip/K01_V001.npy"),
    Path("artifacts/siglip/K01_V001_ids.json"),
)
recap = load_recap_result(Path("artifacts/recap/K01_V001.jsonl"))
```

## 7. Test

Test contract không tải model, không cần Ollama/GPU. Package cần Python >= 3.10
(dùng `dataclass(slots=True)`), trong khi `python3` của host là 3.8 — chạy test
trong container:

```bash
docker run --rm --entrypoint bash -v "$PWD:/w" -w /w keyframe-extractor \
  -c "pip install -q pytest && python3 -m pytest -p no:cacheprovider --basetemp=/tmp/kf"
```

## 8. Lỗi thường gặp

- `Output already exists`: thêm `--overwrite` nếu thực sự muốn thay artifact cũ.
- `CUDA was requested ... but CUDA PyTorch is unavailable`: bỏ `--device cuda:0`;
  Ollama GPU không phụ thuộc CUDA PyTorch.
- `does not expose the 'vision' capability`: model Ollama đang dùng là text-only,
  pull lại model ở mục 5.
- Không kết nối được `127.0.0.1:11501`: chưa chạy `ollama serve`. Từ trong
  container thì dùng `10.0.2.3` (xem mục 5), không phải `127.0.0.1`.
- `Expected .../keyframes or direct <PREFIX>nn_Vnnn video directories`: sai
  `--dataset-root`, hoặc tên thư mục video không khớp pattern ở mục 2.
