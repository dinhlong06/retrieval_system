# compare.py chỉ còn: gọi HTTP tới vLLM server (pass Qwen) + đọc JSON có sẵn
# ../OCR/output_batch1/output_hybrid.json (pass baseline) -- không tự chạy
# PaddleOCR/VietOCR nữa nên không cần GPU, torch hay paddle ở image này.
FROM python:3.11-slim

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Không COPY code: compare.py, qwen_client.py và ../OCR đều bind-mount lúc
# chạy nên sửa code không phải build lại.
ENTRYPOINT ["python3", "/workspace/v2/compare.py"]
