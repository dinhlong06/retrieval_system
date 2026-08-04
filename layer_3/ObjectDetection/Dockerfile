# Layer 3 (GPU): Object Detection — YOLO (ultralytics)
# Base: PyTorch GPU image, CUDA 11.8 (khớp driver đã dùng ở layer 1)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# libgl1/libglib2.0-0: ultralytics kéo theo opencv-python (không phải headless)
# như một dependency phụ, cần các thư viện đồ họa hệ thống này để import được.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

ENTRYPOINT ["python3", "run_detect.py"]
