# CLAUDE.md — Keyframe Feature & Caption Pipeline

## 1. Mục tiêu và phạm vi

Hãy xây dựng package Python `keyframe_pipeline` xử lý input có cấu trúc giống
`Keyframes_L21` và cung cấp hai khả năng độc lập:

1. **SigLIP**: sinh một embedding 1152 chiều cho mỗi keyframe.
2. **ReCap**: sinh một caption cho mỗi keyframe.

Package phải dùng được theo cả hai cách:

- import và gọi trực tiếp từ module Python khác qua public API ổn định;
- chạy cả dataset hoặc một video qua CLI mỏng.

CLI chỉ parse tham số và gọi public API. Không đặt discovery, validation, model
inference hoặc ghi file riêng trong CLI. Không sửa, đổi tên hoặc ghi thêm file vào
thư mục input.

Thứ tự ưu tiên khi có xung đột: **output contract > public API > validation và
alignment > CLI/chi tiết triển khai**.

---

## 2. Input thực tế cần hỗ trợ

Cấu trúc đã quan sát trong workspace:

```text
Keyframes_L21/
└── keyframes/
    ├── L21_V001/
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    ├── L21_V002/
    ├── L21_V003/
    ├── L21_V005/
    └── ...
```

Dataset hiện có 29 thư mục video và 7.800 file JPEG. `L21_V004` và `L21_V020`
không tồn tại. Đây là input hợp lệ: tuyệt đối không suy diễn rằng ID video hoặc ID
keyframe phải liên tục. Ảnh hiện tại là 1280x720 nhưng không được hard-code kích
thước này; preprocessing phải do processor của model đảm nhiệm.

API cấp dataset phải chấp nhận cả:

- outer root: `Path("Keyframes_L21")` nếu bên trong có đúng thư mục `keyframes`;
- keyframes root: `Path("Keyframes_L21/keyframes")` nếu các thư mục video là con
  trực tiếp.

Quy tắc resolve root phải rõ ràng và deterministic:

1. Nếu `root/keyframes` tồn tại, dùng `root/keyframes`.
2. Nếu không, dùng `root` khi nó có thư mục con khớp `^L\d+_V\d+$`.
3. Nếu không khớp trường hợp nào, raise `DatasetLayoutError`.
4. Chỉ quét thư mục video ở đúng một cấp; không recursive tìm video ở nơi khác.
5. Chỉ quét file ảnh là con trực tiếp của thư mục video.

Với input hiện tại:

```text
video_id    = tên thư mục, ví dụ "L21_V001"
keyframe_id = stem nguyên vẹn của file, ví dụ "001"
```

`keyframe_id` là string. Không convert thành integer và không làm mất zero-padding.
Không suy ra `timestamp_ms` từ tên `001.jpg`; metadata này phải là `None` nếu caller
không cung cấp.

Mặc định nhận các extension `.jpg`, `.jpeg`, `.png`, `.webp` không phân biệt hoa
thường. Ignore file hệ thống/không phải ảnh, nhưng một thư mục video không có ảnh
hợp lệ phải raise lỗi. Sort video ID và keyframe ID bằng natural sort ổn định; với
input `1.jpg`, `02.jpg`, `10.jpg`, thứ tự là `1`, `02`, `10`. Sau discovery, mọi
hàm xử lý chính phải giữ nguyên thứ tự nhận được và không tự sort lại.

---

## 3. Kiểu dữ liệu và discovery public

Định nghĩa các kiểu dùng chung trong `types.py`:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Keyframe:
    video_id: str
    keyframe_id: str
    image_path: Path
    timestamp_ms: int | None = None


@dataclass(frozen=True, slots=True)
class VideoKeyframes:
    video_id: str
    keyframes: tuple[Keyframe, ...]


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    root: Path
    videos: tuple[VideoKeyframes, ...]
```

Public discovery API:

```python
def discover_video(
    video_dir: Path,
    *,
    video_id: str | None = None,
) -> VideoKeyframes:
    """Đọc một thư mục video và trả keyframe theo natural order."""


def discover_dataset(dataset_root: Path) -> DatasetIndex:
    """Resolve outer/keyframes root và đọc toàn bộ video theo natural order."""
```

Validation bắt buộc:

- dataset và mỗi video không rỗng;
- `video_id`/`keyframe_id` không rỗng và không chứa `/`, `\`, `.` hoặc `..` như
  một path segment nguy hiểm;
- mọi keyframe trong `VideoKeyframes` có cùng `video_id`;
- `keyframe_id` duy nhất trong từng video;
- `image_path` tồn tại, là file và ảnh có thể decode;
- lỗi decode không được silently skip;
- caller truyền `Sequence[Keyframe]` thì thứ tự sequence đó là canonical.

---

## 4. Model và backend bắt buộc

Default implementation dùng đúng hai model sau:

```python
DEFAULT_SIGLIP_MODEL_ID = "google/siglip-so400m-patch14-384"
DEFAULT_RECAP_MODEL = "qwen3-vl:8b-instruct"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11501"
DEFAULT_RECAP_PROMPT = """Generate retrieval metadata for the attached image.
... (xem config.py) Write a factual English fine-grained caption of 60 to 100
words in 2 to 4 sentences. Verify OCR against the image, preserve the original
language, never translate, and omit uncertain text.
Return only JSON with keys caption and corrected_ocr."""
```

- SigLIP chạy in-process bằng Hugging Face Transformers, theo tài liệu
  <https://huggingface.co/docs/transformers/model_doc/siglip>.
- ReCap chạy local qua Ollama bằng model
  <https://ollama.com/library/qwen3-vl>, tag `8b-instruct`.
- `:latest` là tag ngầm định khi `DEFAULT_RECAP_MODEL` không ghi tag. Cho phép
  caller truyền tag cụ thể để reproducible.
- (Lịch sử) ReCap từng dùng `qwen3-vl:8b-instruct`; đã thay bằng
  Qwen3-VL. Ghi chú cũ giữ lại vì bài học vẫn đúng: không dùng
  `richardyoung/smolvlm2-2.2b-instruct`, tag đó chỉ công bố
  capability `Text`, thiếu vision projector và Ollama sẽ reject request có ảnh.
  Bản `ahmadwaqar/...` dùng cùng SmolVLM2-2.2B-Instruct nhưng có capability
  `Text, Image` và projector SigLIP.

Vẫn cung cấp protocol public để unit test offline và để application thay backend
mà không thay pipeline contract:

```python
from pathlib import Path
from typing import Protocol, Sequence
import numpy as np


class SiglipBackend(Protocol):
    @property
    def embedding_dim(self) -> int: ...

    def encode_paths(self, paths: Sequence[Path], batch_size: int) -> np.ndarray:
        """Trả array (N, D), cùng thứ tự paths. Backend tự lo decode + batch."""


class CaptionBackend(Protocol):
    def caption_frames(
        self, inputs: Sequence[CaptionInput]
    ) -> Sequence[CaptionOutput]:
        """Trả đúng B CaptionOutput, cùng thứ tự input.

        Nhận CaptionInput (image_path + ocr_hints + object_hints) chứ không phải
        Path, vì nội dung prompt bây giờ khác nhau theo từng frame.
        """
```

Package phải có hai concrete backend:

```python
class TransformersSiglipBackend(SiglipBackend):
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_SIGLIP_MODEL_ID,
        device: str | None = None,
    ) -> None: ...


class OllamaCaptionBackend(CaptionBackend):
    def __init__(
        self,
        *,
        model: str = DEFAULT_RECAP_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        prompt: str = DEFAULT_RECAP_PROMPT,
        timeout_seconds: float = 120.0,
        keep_alive: str = "10m",
        max_retries: int = 2,
        client: object | None = None,  # injection seam cho test
    ) -> None: ...
```

`TransformersSiglipBackend` dùng `AutoImageProcessor.from_pretrained(...)` và
`AutoModel.from_pretrained(...)`. `OllamaCaptionBackend` dùng official Python SDK
`ollama.Client`; không spawn `ollama run` cho từng keyframe và không phụ thuộc
parse stdout CLI. Production code không được dùng fake embedding hoặc caption
placeholder. Unit test không được tải model hoặc gọi Ollama thật; inject fake
backend/client deterministic.

Runtime dependencies tối thiểu gồm `numpy`, `Pillow`, `torch`, `transformers` và
official package `ollama`. Ollama application/server là dependency hệ thống riêng;
`pip install ollama` chỉ cài Python client, không cài server hoặc tự tải model. Pin
version đã test trong lockfile/`pyproject.toml` và ghi command setup trong README.

Model lifecycle:

- SigLIP model/processor load một lần cho mỗi pipeline instance;
- SigLIP dùng `eval()` và `torch.inference_mode()`;
- preprocess RGB bằng processor tương ứng, có xử lý EXIF orientation;
- batch không làm thay đổi thứ tự;
- tensor kết quả phải đưa về CPU, convert `float32` trước khi tạo NumPy output;
- không dùng mutable global để chứa state của request;
- `device=None` của SigLIP tự chọn thiết bị hợp lệ và log device được chọn;
- Ollama server quản lý CPU/GPU; public ReCap API không nhận `device` giả tạo;
- Ollama client được tái sử dụng, gửi `keep_alive` để hạn chế reload model;
- lỗi out-of-memory/inference được bọc bằng `ModelInferenceError` và giữ exception
  gốc qua `raise ... from exc`.

---

## 5. SigLIP contract

### 5.1. Kết quả và public API cấp video

```python
@dataclass(frozen=True, slots=True)
class SiglipResult:
    video_id: str
    embeddings: np.ndarray       # (N, 1152), float32, row L2-normalized
    keyframe_ids: tuple[str, ...] # cùng index với embeddings
    embedding_path: Path | None = None
    ids_path: Path | None = None


def extract_siglip(
    keyframes: Sequence[Keyframe],
    *,
    backend: SiglipBackend | None = None,
    model_id: str = DEFAULT_SIGLIP_MODEL_ID,
    output_dir: Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
    overwrite: bool = False,
) -> SiglipResult:
    """Sinh embedding cho đúng một video, giữ nguyên thứ tự input."""


def extract_siglip_from_dir(
    video_dir: Path,
    *,
    video_id: str | None = None,
    backend: SiglipBackend | None = None,
    model_id: str = DEFAULT_SIGLIP_MODEL_ID,
    output_dir: Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
    overwrite: bool = False,
) -> SiglipResult:
    """discover_video rồi gọi extract_siglip."""
```

Nếu `backend=None`, khởi tạo `TransformersSiglipBackend(model_id=model_id)`; default
là `google/siglip-so400m-patch14-384`. Nếu caller đã inject `backend`, `model_id`
không được âm thầm override backend: hoặc yêu cầu giữ default, hoặc raise
`InvalidArgumentError` khi truyền giá trị khác default. `batch_size` phải lớn hơn
0. Code gọi lặp lại hoặc xử lý dataset nên dùng `KeyframePipeline` để tái sử dụng
model.

Adapter Transformers phải đi theo luồng image feature chính thức:

```python
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device).eval()

inputs = processor(images=rgb_images, return_tensors="pt")
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
with torch.inference_mode():
    image_features = model.get_image_features(**inputs)
```

Checkpoint mặc định có `vision_config.hidden_size == 1152`; validate invariant này
ngay sau khi load và vẫn validate shape runtime. Không dùng
`AutoModelForZeroShotImageClassification`, logits hoặc text prompt để tạo embedding.

Với mỗi batch, dùng image features/pooled image representation của SigLIP, không
dùng logits text-image. Vector đầu ra phải có đúng 1152 phần tử. L2-normalize từng
hàng sau khi convert sang `float32`; vector norm bằng 0 là lỗi, không chia bằng
epsilon để che lỗi.

Validation sau inference:

```python
assert embeddings.ndim == 2
assert embeddings.shape == (len(keyframes), 1152)
assert embeddings.dtype == np.float32
assert np.isfinite(embeddings).all()
assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)
```

### 5.2. API cấp dataset

```python
@dataclass(frozen=True, slots=True)
class SiglipDatasetResult:
    root: Path
    results: tuple[SiglipResult, ...]  # natural video order


def extract_siglip_dataset(
    dataset_root: Path,
    *,
    output_dir: Path,
    backend: SiglipBackend | None = None,
    model_id: str = DEFAULT_SIGLIP_MODEL_ID,
    batch_size: int = 32,
    device: str | None = None,
    overwrite: bool = False,
) -> SiglipDatasetResult:
    """Xử lý mọi video trong Keyframes_L21 bằng cùng một backend instance."""
```

Dataset runner chạy fail-fast. Trước inference, preflight toàn bộ target để phát
hiện collision khi `overwrite=False`. Dataset không được coi là transaction toàn
cục: output video đã hoàn tất có thể còn lại nếu video sau thất bại, nhưng không
bao giờ được trả một result thành công giả hoặc silently bỏ video lỗi.

### 5.3. Output SigLIP trên disk

Khi có `output_dir`, mỗi video tạo đúng hai artifact:

```text
{output_dir}/{video_id}.npy
{output_dir}/{video_id}_ids.json
```

Ví dụ dataset:

```text
artifacts/siglip/
├── L21_V001.npy
├── L21_V001_ids.json
├── L21_V002.npy
└── L21_V002_ids.json
```

`.npy` chứa array `(N, 1152)`, dtype `float32`, row-normalized. JSON IDs là list
phẳng, `video_id` suy ra từ tên file:

```json
["001","002","003"]
```

Schema này khớp với embedding BEiT-3 của `layer_2/Keyframe_Extracting`
(`src/core/runner.py::_save_embeddings`) để hai bên dùng chung một loader.

`ids[i]` ánh xạ tuyệt đối tới `embeddings[i]`. Cung cấp public loader:

```python
def load_siglip_result(
    embedding_path: Path,
    ids_path: Path,
    *,
    mmap_mode: str | None = None,
) -> SiglipResult:
    """Load cả cặp file và validate schema, shape, dtype, finite, norm, alignment."""
```

Không infer `video_id` chỉ từ một file nếu JSON đã cung cấp ID. Nếu tên file và
`video_id` trong JSON mâu thuẫn, raise `ArtifactValidationError`.

---

## 6. ReCap contract

Trong package này, ReCap là bước tạo **một caption độc lập cho từng keyframe**.
Không tự gộp caption thành video summary và không dùng context của frame lân cận.
Backend mặc định bắt buộc là `OllamaCaptionBackend` gọi local Ollama model
`qwen3-vl:8b-instruct`. Output vẫn phải tuân thủ contract dưới đây.

### 6.1. Cài đặt và demo Ollama

Ollama là service ngoài process. README phải hướng dẫn rõ:

```bash
# 1. Sau khi cài Ollama, tải model một lần
ollama pull qwen3-vl:8b-instruct

# 2. Demo tương tác do người dùng yêu cầu
ollama run qwen3-vl:8b-instruct

# 3. Demo caption trực tiếp một ảnh
ollama run qwen3-vl:8b-instruct \
  "Describe this image in exactly one concise factual English sentence of at most 20 words. Output only that sentence." \
  --images Keyframes_L21/keyframes/L21_V001/001.jpg

# Python client cho package
pip install ollama
```

Trước khi chạy pipeline, Ollama phải đang chạy và API local phải truy cập được tại
`http://127.0.0.1:11501` (hoặc `ollama_host` do caller truyền). Pipeline không tự
`pull` model vì đây là network/disk mutation; nếu thiếu model, raise
`OllamaModelNotFoundError` kèm lệnh `ollama pull ...`. Nếu không kết nối được server,
raise `OllamaUnavailableError` kèm hướng dẫn khởi động Ollama.

`OllamaCaptionBackend` thực hiện preflight đúng một lần trước request đầu tiên:
gọi `client.show(model)` để đồng thời kiểm tra kết nối và model local. HTTP 404 được
map thành `OllamaModelNotFoundError`; connection/timeout được map thành
`OllamaUnavailableError`. Response `show()` phải có capability `vision`; model chỉ
text raise `ModelConfigurationError` trước inference. Không gọi `show()` lại cho
từng ảnh.

Concrete backend dùng official Python client theo mẫu:

```python
from ollama import Client

client = Client(host=ollama_host, timeout=timeout_seconds)
response = client.chat(
    model="qwen3-vl:8b-instruct",
    messages=[{
        "role": "user",
        "content": (
            "Describe this image in exactly one concise factual English sentence "
            "of at most 20 words. Output only that sentence."
        ),
        "images": [str(image_path.resolve())],
    }],
    format=CAPTION_RESPONSE_SCHEMA,   # ép JSON {caption, corrected_ocr}
    think=False,                      # bắt buộc: thinking đốt hết token budget
    stream=False,
    keep_alive="10m",
    options={"temperature": 0, "num_predict": NUM_PREDICT},  # 768
)
payload = json.loads(response.message.content)   # hỏng -> ModelInferenceError
```

Gửi đúng một ảnh/request để mapping output không mơ hồ. Không shell-out qua
`ollama run` trong library. SDK được phép nhận path trực tiếp; chỉ base64 encode khi
implementation cố ý dùng REST API thô. `keep_alive`, prompt, timeout, model và host
phải nằm trong config/backend instance, không hard-code rải rác.

Vì Ollama chat không phải tensor batching, tham số ReCap public là
`max_concurrency`, không phải `batch_size`. Default bằng 1 để demo ổn định. Nếu chạy
song song, dùng bounded worker pool, gắn index trước khi submit và ráp caption về
đúng input order; không append theo completion order. Chỉ retry có giới hạn với
lỗi kết nối/5xx tạm thời, không retry lỗi model-not-found hoặc response sai contract.

### 6.2. Kết quả và public API cấp video

```python
@dataclass(frozen=True, slots=True)
class CaptionRecord:
    video_id: str
    keyframe_id: str
    caption: str
    corrected_ocr: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecapResult:
    video_id: str
    records: tuple[CaptionRecord, ...]
    output_path: Path | None = None


def generate_recap(
    keyframes: Sequence[Keyframe],
    *,
    backend: CaptionBackend | None = None,
    model: str = DEFAULT_RECAP_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    previous_output: SiglipResult | None = None,
    output_path: Path | None = None,
    max_concurrency: int = 1,
    overwrite: bool = False,
) -> RecapResult:
    """Sinh đúng một caption cho mỗi keyframe, giữ nguyên thứ tự input."""


def generate_recap_from_dir(
    video_dir: Path,
    *,
    video_id: str | None = None,
    backend: CaptionBackend | None = None,
    model: str = DEFAULT_RECAP_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    previous_embedding_path: Path | None = None,
    previous_ids_path: Path | None = None,
    output_path: Path | None = None,
    max_concurrency: int = 1,
    overwrite: bool = False,
) -> RecapResult:
    """Discover video, load/validate SigLIP nếu có rồi gọi generate_recap."""
```

`previous_output` dùng để kiểm tra alignment, không mặc định đưa embeddings vào
caption model. Khi truyền artifact disk, `previous_embedding_path` và
`previous_ids_path` phải cùng có mặt; chỉ truyền một path là `InvalidArgumentError`.
Validation phải diễn ra trước caption inference:

- cùng `video_id`;
- cùng số keyframe;
- `keyframe_ids` giống nhau tuyệt đối theo thứ tự.

Nếu `backend=None`, khởi tạo `OllamaCaptionBackend(model=model, host=ollama_host)`.
Nếu đã inject backend, không được âm thầm override backend bằng model/host khác
default. Backend trả `CaptionOutput`; caption normalize line break thành khoảng trắng,
trim đầu/cuối và không rỗng, `corrected_ocr` khử trùng lặp giữ thứ tự. Prompt mặc
định yêu cầu caption tiếng Anh fine-grained 60–100 từ cộng OCR đã hiệu đính giữ
nguyên ngôn ngữ gốc. `temperature=0`, `think=False` và `num_predict=768` cho output
ổn định và đủ chỗ cho JSON; prompt và generation options thuộc config của concrete
backend. JSON hỏng, caption rỗng, hoặc `corrected_ocr` sai kiểu là
`ModelInferenceError` chứ không nhận partial data.

### 6.3. API cấp dataset

```python
@dataclass(frozen=True, slots=True)
class RecapDatasetResult:
    root: Path
    results: tuple[RecapResult, ...]  # natural video order


def generate_recap_dataset(
    dataset_root: Path,
    *,
    output_dir: Path,
    backend: CaptionBackend | None = None,
    model: str = DEFAULT_RECAP_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    previous_siglip_dir: Path | None = None,
    max_concurrency: int = 1,
    overwrite: bool = False,
) -> RecapDatasetResult:
    """Sinh một JSONL/video; validate cặp SigLIP tương ứng nếu được truyền."""
```

Nếu `previous_siglip_dir` được truyền, mỗi video phải có đủ `{video_id}.npy` và
`{video_id}_ids.json`; thiếu hoặc thừa alignment đều là lỗi. Preflight output
collision toàn dataset trước inference khi `overwrite=False`.

### 6.4. Output ReCap trên disk

Dataset output:

```text
artifacts/recap/
├── L21_V001.jsonl
├── L21_V002.jsonl
└── ...
```

Mỗi dòng là một JSON object UTF-8, không bọc cả file bằng JSON array:

```json
{"video_id":"L21_V001","keyframe_id":"001","caption":"A person in a dark jacket stands beside a parked silver car on a narrow street. Buildings line both sides and the light suggests late afternoon.","corrected_ocr":["VTV1"]}
{"video_id":"L21_V001","keyframe_id":"002","caption":"The silver car moves through an intersection while two pedestrians wait at the near corner. Traffic lights and shopfronts are visible in the background.","corrected_ocr":[]}
```

Mỗi record có chính xác **bốn** field bắt buộc `video_id`, `keyframe_id`,
`caption`, `corrected_ocr`. Không có đường tương thích ngược với schema 3 field.
File có đúng N dòng theo đúng input order, không caption rỗng, không duplicate ID.
Cung cấp loader public:

```python
def load_recap_result(output_path: Path) -> RecapResult:
    """Load JSONL và validate schema, video ID duy nhất, ID duy nhất, caption."""
```

---

## 7. Facade bắt buộc cho module khác

Đây là interface được khuyến nghị cho application/service vì model được tái sử
dụng qua nhiều video:

```python
class KeyframePipeline:
    def __init__(
        self,
        *,
        siglip_backend: SiglipBackend | None = None,
        caption_backend: CaptionBackend | None = None,
        siglip_model_id: str = DEFAULT_SIGLIP_MODEL_ID,
        siglip_device: str | None = None,
        recap_model: str = DEFAULT_RECAP_MODEL,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
    ) -> None: ...

    def run_siglip(
        self,
        keyframes: Sequence[Keyframe],
        *,
        output_dir: Path | None = None,
        batch_size: int = 32,
        overwrite: bool = False,
    ) -> SiglipResult: ...

    def run_recap(
        self,
        keyframes: Sequence[Keyframe],
        *,
        previous_output: SiglipResult | None = None,
        output_path: Path | None = None,
        max_concurrency: int = 1,
        overwrite: bool = False,
    ) -> RecapResult: ...

    def run_dataset(
        self,
        dataset_root: Path,
        *,
        output_root: Path,
        siglip_batch_size: int = 32,
        recap_max_concurrency: int = 1,
        overwrite: bool = False,
    ) -> tuple[SiglipDatasetResult, RecapDatasetResult]:
        """Discover một lần, chạy SigLIP rồi ReCap với alignment bắt buộc."""
```

`run_dataset` ghi vào `output_root/siglip` và `output_root/recap`. Nó phải discover
dataset đúng một lần và dùng cùng các object `Keyframe` cho cả hai bước.

Ví dụ gọi từ module khác:

```python
from pathlib import Path
from keyframe_pipeline import KeyframePipeline

pipeline = KeyframePipeline(
    siglip_device="cuda:0",
    recap_model="qwen3-vl:8b-instruct",
    ollama_host="http://127.0.0.1:11501",
)
siglip_results, recap_results = pipeline.run_dataset(
    Path("Keyframes_L21"),
    output_root=Path("artifacts"),
)
```

Ví dụ chỉ xử lý một video:

```python
from pathlib import Path
from keyframe_pipeline import discover_video, KeyframePipeline

video = discover_video(Path("Keyframes_L21/keyframes/L21_V001"))
pipeline = KeyframePipeline()

features = pipeline.run_siglip(
    video.keyframes,
    output_dir=Path("artifacts/siglip"),
)
captions = pipeline.run_recap(
    video.keyframes,
    previous_output=features,
    output_path=Path("artifacts/recap/L21_V001.jsonl"),
)
```

---

## 8. Error contract

Tất cả exception public kế thừa một base class:

```python
class KeyframePipelineError(Exception): ...
class InvalidArgumentError(KeyframePipelineError): ...
class DatasetLayoutError(KeyframePipelineError): ...
class InvalidKeyframeInputError(KeyframePipelineError): ...
class AlignmentError(KeyframePipelineError): ...
class ModelConfigurationError(KeyframePipelineError): ...
class ModelInferenceError(KeyframePipelineError): ...
class OllamaUnavailableError(ModelInferenceError): ...
class OllamaModelNotFoundError(ModelConfigurationError): ...
class ArtifactValidationError(KeyframePipelineError): ...
class OutputAlreadyExistsError(KeyframePipelineError): ...
```

Raise lỗi rõ nghĩa khi input rỗng, mixed video IDs, duplicate ID, ảnh hỏng,
`batch_size`/`max_concurrency` không hợp lệ, checkpoint/backend thiếu cấu hình,
dimension khác 1152, Ollama không chạy/model chưa pull, NaN/Inf/zero-norm, caption
rỗng, alignment sai hoặc output đã tồn tại. Map `ollama.ResponseError` theo status
thay vì làm mất nguyên nhân. Không dùng `assert` cho validation runtime vì Python
có thể tắt assert; các `assert` trong tài liệu chỉ mô tả invariant và test.

Không triển khai partial success mặc định. Dataset fail-fast ở lỗi đầu tiên và lỗi
phải chứa context tối thiểu: `video_id`, `keyframe_id`/path nếu có và operation.

---

## 9. I/O, atomicity và overwrite

- Tạo parent output directory khi cần.
- Mọi file được ghi vào temporary file duy nhất trong cùng directory, flush/close,
  rồi `os.replace` sang target.
- Với NumPy, mở file temp bằng binary file handle rồi gọi `np.save(handle, array)`;
  không gọi `np.save("name.npy.tmp", ...)` vì NumPy có thể tự nối thêm `.npy`.
- Xóa temp file của request hiện tại khi ghi thất bại; không xóa artifact cũ.
- `overwrite=False`: nếu bất kỳ target nào đã tồn tại, raise trước inference.
- `overwrite=True`: thay thế từng file atomically, không append.
- Loader chỉ trả result sau khi đã validate đầy đủ cả cặp artifact SigLIP. Hai file
  riêng lẻ không tạo được transaction xuyên file, vì vậy reader không được coi một
  file đơn lẻ là kết quả hoàn chỉnh.
- JSON/JSONL dùng UTF-8, `ensure_ascii=False`, newline `\n`.

Hàm public không đổi current working directory. Mọi path trả về phải là path target
thực tế và nhất quán (ưu tiên absolute resolved path hoặc nhất quán với input;
chọn một policy và test nó).

---

## 10. CLI wrapper

CLI bắt buộc gọi lại facade/function public:

```bash
python -m keyframe_pipeline siglip-dataset \
  --dataset-root Keyframes_L21 \
  --output-dir artifacts/siglip \
  --batch-size 32

python -m keyframe_pipeline recap-dataset \
  --dataset-root Keyframes_L21 \
  --siglip-dir artifacts/siglip \
  --output-dir artifacts/recap \
  --ocr-json ../layer_3/OCR/output/output.json \
  --objects-json ../layer_3/ObjectDetection/output/detections.json \
  --ollama-model qwen3-vl:8b-instruct \
  --ollama-host http://127.0.0.1:11501 \
  --max-concurrency 1

python -m keyframe_pipeline run \
  --dataset-root Keyframes_L21 \
  --output-root artifacts \
  --ocr-json ../layer_3/OCR/output/output.json \
  --objects-json ../layer_3/ObjectDetection/output/detections.json \
  --ollama-model qwen3-vl:8b-instruct \
  --ollama-host http://127.0.0.1:11501
```

`--ocr-json` và `--objects-json` là tuỳ chọn, chỉ có trên `recap-video`,
`recap-dataset` và `run`; `siglip-*` không có. Hai artifact được parse **một lần
cho cả dataset**, không phải mỗi video.

CLI cũng có thể cung cấp lệnh cấp video, nhưng không được thay thế API Python.
Exit code khác 0 khi có lỗi và message không được nuốt traceback/root cause ở chế
độ debug.

---

## 11. Package layout và public exports

```text
keyframe_pipeline/
├── __init__.py
├── __main__.py
├── types.py
├── config.py
├── discovery.py
├── protocols.py
├── exceptions.py
├── siglip.py
├── recap.py
├── facade.py
├── io.py
└── backends/
    ├── __init__.py
    ├── siglip_transformers.py
    └── recap_ollama.py

tests/
├── test_discovery.py
├── test_siglip.py
├── test_recap.py
├── test_ollama_backend.py
├── test_io.py
└── test_integration.py
```

`keyframe_pipeline/__init__.py` phải export ít nhất:

```python
from .types import (
    Keyframe, VideoKeyframes, DatasetIndex,
    SiglipResult, SiglipDatasetResult,
    CaptionRecord, RecapResult, RecapDatasetResult,
)
from .protocols import SiglipBackend, CaptionBackend
from .config import (
    DEFAULT_SIGLIP_MODEL_ID, DEFAULT_RECAP_MODEL, DEFAULT_OLLAMA_HOST,
    DEFAULT_RECAP_PROMPT,
)
from .backends import TransformersSiglipBackend, OllamaCaptionBackend
from .discovery import discover_video, discover_dataset
from .siglip import (
    extract_siglip, extract_siglip_from_dir, extract_siglip_dataset,
    load_siglip_result,
)
from .recap import (
    generate_recap, generate_recap_from_dir, generate_recap_dataset,
    load_recap_result,
)
from .facade import KeyframePipeline
```

Các exception public cũng phải import được từ package root. Không yêu cầu caller
import module private hoặc gọi subprocess để dùng pipeline.

---

## 12. Thứ tự task triển khai

Thực hiện theo thứ tự này để mỗi bước có thể test độc lập:

1. **Scaffold**: package, config/dependencies, dataclass, exception và public exports.
2. **Discovery**: hỗ trợ đúng hai dạng root, natural sort, validation trên fixture
   nhỏ và smoke-test read-only với `Keyframes_L21`.
3. **I/O**: atomic writer và public loader cho `.npy`/JSON/JSONL.
4. **SigLIP**: Transformers adapter cho `google/siglip-so400m-patch14-384`,
   batching, float32, normalize, video API và dataset API.
5. **ReCap**: Ollama client adapter cho
   `qwen3-vl:8b-instruct`, caption cleanup, bounded concurrency,
   alignment, video API và dataset API.
6. **Facade**: model reuse và `run_dataset` discover đúng một lần.
7. **CLI**: wrapper trên public API.
8. **Tests và README**: unit test không cần network/GPU; integration thật là opt-in.

Không để `TODO`, `pass`, placeholder caption hoặc random embedding trong code bàn
giao. Hai default model đã được chỉ định ở mục 4; thiếu Hugging Face checkpoint,
Ollama service hoặc Ollama model local phải báo lỗi cấu hình/kết nối rõ ràng, không
fallback sang model khác và không giả vờ inference thành công.

---

## 13. Acceptance criteria

### Discovery

- `discover_dataset(Path("Keyframes_L21"))` và gọi với
  `Path("Keyframes_L21/keyframes")` tạo cùng index.
- Với dataset workspace hiện tại: 29 video, tổng 7.800 keyframe; video đầu là
  `L21_V001`, `keyframe_id` đầu là `"001"`.
- Không tạo giả `L21_V004`/`L21_V020`; không yêu cầu ID liên tục.
- Fixture `1.jpg`, `02.jpg`, `10.jpg` có natural order đúng và giữ nguyên ID string.

### SigLIP

- Default backend load đúng `google/siglip-so400m-patch14-384`, gọi
  `get_image_features()` và reject checkpoint có vision hidden size khác 1152.
- Với N=3, result/disk array có shape `(3, 1152)`, dtype `float32`, finite và norm
  từng hàng xấp xỉ 1.0.
- IDs có đúng ba string theo input order; reload từ disk giữ nguyên mapping.
- Batch size khác nhau không đổi thứ tự output.
- Fake backend trả dimension sai, NaN hoặc zero vector đều raise đúng exception.

### ReCap

- Default backend dùng đúng `qwen3-vl:8b-instruct` và default host
  `http://127.0.0.1:11501`.
- Unit test bằng mocked `ollama.Client` xác nhận mỗi request có đúng một image path,
  `stream=False`, `think=False`, `temperature=0`, `num_predict=768`, `format` là
  JSON schema `{caption, corrected_ocr}`, prompt chứa `OCR_CANDIDATES=` và
  `OBJECT_CANDIDATES=` (bbox quy về vùng thô kiểu `left-bottom`), và kết quả được
  ráp theo input order dù request hoàn tất khác thứ tự.
- Model trả JSON hỏng hoặc `corrected_ocr` sai kiểu raise `ModelInferenceError`,
  không retry.
- Parse Layer 3: root không phải array, record malformed, `frame_id` trùng đều
  raise `ArtifactValidationError` trước request đầu tiên.
- Join theo `frame_id == keyframe_id`; record của video khác bị bỏ qua; keyframe
  thiếu record nhận hint rỗng; file metadata không khớp keyframe nào thì raise.
- Ollama không chạy và model chưa pull lần lượt raise `OllamaUnavailableError` và
  `OllamaModelNotFoundError` với message có hành động khắc phục.
- Model local không có capability `vision` raise `ModelConfigurationError` trước
  request caption đầu tiên.
- Với N=3, JSONL có đúng ba dòng và mỗi dòng có chính xác bốn field bắt buộc;
  loader reject schema 3 field cũ.
- Record order giống input; caption sau normalize không rỗng.
- Raise `AlignmentError` trước inference khi video ID, count hoặc order của SigLIP
  khác input.
- Loader reject malformed JSON, extra/missing fields, mixed video IDs và duplicate
  keyframe IDs.

### Integration và public API

- Consumer import được mọi symbol liệt kê ở mục 11 từ `keyframe_pipeline`.
- Có test gọi API trực tiếp, không subprocess và không CLI.
- `KeyframePipeline.run_dataset` load SigLIP một lần, tái sử dụng Ollama client với
  `keep_alive` và tạo đúng một cặp SigLIP + một JSONL cho từng video.
- Test unit chạy offline bằng fake backend deterministic; test model thật được đánh
  dấu integration/slow. Có smoke test opt-in caption một ảnh qua local Ollama sau
  khi user đã pull model; test mặc định không tự pull/download.
- Không thay đổi bất kỳ file nào dưới `Keyframes_L21`.

---

## 14. Các invariant không được tự ý thay đổi

- Default SigLIP checkpoint là `google/siglip-so400m-patch14-384` qua Transformers.
- Default ReCap backend là local Ollama model
  `qwen3-vl:8b-instruct` qua official Python client.
- SigLIP embedding dimension là `1152`.
- Dtype trên disk và trong `SiglipResult` là `np.float32`.
- Từng hàng embedding đã L2-normalize.
- Tên file SigLIP là `{video_id}.npy` và `{video_id}_ids.json`.
- Tên file ReCap cấp dataset là `{video_id}.jsonl`.
- ReCap có đúng một record/keyframe và đúng ba field:
  `video_id`, `keyframe_id`, `caption`.
- Mapping theo index giữa ảnh, `keyframe_ids`, embedding và caption phải được bảo
  toàn tuyệt đối.
- Không resize, pad hoặc truncate embedding để che checkpoint sai dimension.
- Library không spawn `ollama run` theo từng ảnh và không tự pull model.
- Public API là yêu cầu bắt buộc; CLI và internal backend không được trở thành cách
  tích hợp duy nhất.
