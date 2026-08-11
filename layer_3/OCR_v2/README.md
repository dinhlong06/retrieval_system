# OCR_v2 -- thử `huypl53/qwen3-vl-4b-vietnamese-ocr-merged`

Đánh giá xem VLM Qwen3-VL-4B (finetune OCR tiếng Việt) có thay được pipeline
production hiện tại (`../OCR`: PaddleOCR detect + VietOCR rec + VLM phục hồi
dấu) không.

**Kết luận: KHÔNG dùng để thay production trên hạ tầng hiện có -- vừa chậm
hơn ~4x vừa có lỗi lặp vô hạn (repetition loop) làm giảm recall xuống 72%.**
Chi tiết bằng chứng ở dưới.

---

## Tóm tắt hành trình

| Môi trường | Cách chạy | Kết quả |
|---|---|---|
| Cụm GPU local (RTX 2080Ti, dùng chung) | `transformers` + bitsandbytes NF4 | 1.2 token/giây -- quá chậm |
| Cụm GPU local | vLLM fp16 | Không khởi động nổi (VRAM không đủ) |
| Colab free (T4) | vLLM | Không cài được (`libcudart.so.13`) |
| Colab free (T4) | `transformers` fp16 + FastAPI + ngrok tunnel | Chạy được nhưng ~0.03 fps |
| Cụm GPU local (GPU rảnh xuất hiện) | **vLLM + NF4 (bitsandbytes), `--enforce-eager`, `--dtype float16`** | **Chạy xong đủ 855/855 frame, ~0.3 fps** |

Bước cuối thành công nhờ `run_vllm.sh` (tự dò GPU rảnh nhất, retry khi bị
tiến trình khác chiếm VRAM giữa lúc khởi động) kết hợp `run_resilient.sh`
(tự phát hiện server chết giữa chừng -- kể cả sau khi đã qua health check,
vì cụm dùng chung không có cách nào khoá VRAM cho riêng mình -- và tự khởi
động lại + resume từ checkpoint). Lần chạy full 855 frame mất ~49 phút thực
tế, trải qua 4 lần crash/restart do tranh chấp GPU trước khi hoàn tất ở lần
thử thứ 5/15.

## Đo tốc độ (đo thật, 855 frame)

- Pipeline production hiện tại (`../OCR`): **~1.3 fps**.
- Qwen3-VL qua vLLM NF4 local: **~0.3 fps** (855 frame / 49.2 phút) --
  chậm hơn production ~4.3 lần. Đã cải thiện ~10x so với `transformers`
  thuần (0.03 fps) nhờ vLLM continuous batching + PagedAttention, nhưng
  vẫn bị giới hạn bởi: (1) chi phí dequant NF4 mỗi forward pass qua kernel
  bitsandbytes (không tối ưu bằng kernel gốc), (2) `--enforce-eager` (phải
  tắt CUDA graph để tránh OOM lúc khởi động), (3) GPU Turing chia sẻ SM
  với tiến trình khác trên cùng card, không chỉ VRAM.
- `dataset_batch1` đầy đủ có **177.321 frame**. Ở ~0.3 fps (giả sử không
  crash) sẽ mất **~6.2 ngày** chạy liên tục; tính cả downtime do crash lặp
  lại như đã quan sát, thực tế nhiều khả năng **vượt 1 tuần**. Production
  xử lý cùng khối lượng đó trong **~1.6 ngày**. Không khả thi để scale lên
  full dataset trên hạ tầng hiện có.

## Chất lượng (đủ 855/855 frame, `output/report.txt`)

Thước đo: recall theo dòng, ascii-normalized (bỏ dấu) so với baseline
`output_hybrid.json` -- vì pipeline search cũng bỏ dấu, dấu đúng/sai không
ảnh hưởng tới tìm kiếm, chỉ nội dung chữ mới quan trọng.

```
recall dòng (ascii, dấu không tính): 2734/3794 = 72.1%
độ dài trung bình text Qwen mỗi frame: 121 ký tự
```

**Nguyên nhân chính khiến recall thấp: lỗi lặp vô hạn (repetition loop).**
Trên các frame có logo/chữ nhỏ lặp lại nhiều lần trong ảnh (banner, watermark
kênh...), Qwen bị kẹt sinh đi sinh lại đúng một dòng cho tới khi chạm giới
hạn `max_new_tokens=256`, nuốt hết ngân sách token và bỏ sót toàn bộ text
thật còn lại trong khung hình. Đếm được **37/855 frame (4.3%)** có một dòng
lặp lại >=10 lần. Ví dụ (`L21_V002_153`):

```
qwen: "Bank of England\nBank of England\n..." (lặp 64 lần, chiếm hết 256 token)
prod: ['Bank of England', ..., '06:45:39', ..., 'TP.HCM khang dinh bao mat thong tin nguoi mua ban vang mieng']
thiếu: 27 dòng, bao gồm cả dòng tin tức chạy chữ ở cuối khung hình
```

Đây là lỗi có tính hệ thống của mô hình khi decode greedy (`temperature=0`)
gặp text lặp trong ảnh, không phải lỗi cấu hình phía mình -- tăng
`max_new_tokens` chỉ khiến nó lặp nhiều hơn chứ không tự thoát vòng lặp.

**Điểm mạnh** trên frame không dính lỗi lặp: phục hồi dấu tốt, đôi khi đọc
được cả câu dài mà baseline chỉ ra ký tự rác (ví dụ `L21_V003_191` -- Qwen
đọc đúng gần trọn đoạn hội thoại hỏi cung mà VietOCR chỉ nhận được vài cụm
rời rạc).

## Kết luận

Qwen3-VL-4B qua vLLM NF4 **chạy được** trên cụm GPU dùng chung hiện có (khác
với kết luận "không khả thi" của lần thử trước, khi mới chỉ có
`transformers` + GPU Colab T4) nhưng **không đạt cả hai tiêu chí thay thế
production**: chậm hơn ~4x và có lỗi lặp vô hạn làm mất text thật trên ~4%
frame. Không khuyến nghị áp dụng cho `dataset_batch1` ở dạng hiện tại.

## Nếu muốn thử lại

- **GPU Ampere+ (A100/L4/RTX 4090...) không tranh chấp**: bỏ NF4 (đủ VRAM
  cho fp16 gốc), bật lại CUDA graph, có FlashAttention-2 -- tốc độ có thể
  lên vùng vài fps, khi đó đáng cân nhắc lại.
- **Sửa lỗi lặp**: thử `repetition_penalty`/`no_repeat_ngram_size` trong
  request tới vLLM (chưa thử ở lần chạy này, `qwen_client.py` mới chỉ
  truyền `temperature=0`), hoặc giảm `max_new_tokens` để giới hạn thiệt hại
  mỗi lần bị kẹt lặp.

Toàn bộ code (`compare.py`, `qwen_client.py`, `analyze.py`, `run_vllm.sh`,
`run_resilient.sh`) tái dùng được ngay cho lần thử sau -- chỉ cần trỏ
`VLLM_URL` vào server mới.

## File

```
compare.py        # pass 1: gọi Qwen qua HTTP (song song, ThreadPoolExecutor, checkpoint)
                   # pass 2: lấy baseline có sẵn từ ../OCR/output_batch1/output_hybrid.json (không chạy lại engine)
qwen_client.py     # client OpenAI-compatible gọi vLLM server
analyze.py         # tính recall (ascii-normalized) so với baseline, xếp hạng frame lệch nhiều nhất
run_compare.sh     # build + chạy so sánh, nhận VLLM_URL (local hoặc remote)
run_vllm.sh        # host vLLM NF4 local: tự dò GPU rảnh nhất, retry khi bị chiếm VRAM lúc khởi động
run_resilient.sh   # vòng lặp tự phục hồi: phát hiện server chết giữa chừng, restart + resume checkpoint
Dockerfile         # image nhẹ cho compare.py: chỉ cần openai client, không cần GPU/paddle/torch
Dockerfile.vllm    # image vLLM cho local (bitsandbytes NF4, PYTORCH_CUDA_ALLOC_CONF=expandable_segments)
output/qwen.json   # kết quả Qwen đủ 855 frame
output/prod.json   # baseline production tương ứng (đọc từ ../OCR/output_batch1/output_hybrid.json)
output/report.txt  # báo cáo recall + 20 frame lệch nhiều nhất
```

Notebook Colab (không nằm trong git, ở scratchpad phiên làm việc):
`qwen_ocr_colab.ipynb` (test nhanh 9 ảnh), `qwen_ocr_colab_full.ipynb` (chạy
full trong Colab, không cần máy chính), `qwen_vllm_colab_host.ipynb` (host +
gọi từ máy chính -- bản dùng `transformers` + FastAPI, không phải vLLM).
