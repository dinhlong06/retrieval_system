# Layer 3 (GPU): Object Detection — YOLO (ultralytics)
# Base: PyTorch GPU image, CUDA 11.8 (khớp driver đã dùng ở layer 1)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# libgl1/libglib2.0-0: ultralytics kéo theo opencv-python (không phải headless)
# như một dependency phụ, cần các thư viện đồ họa hệ thống này để import được.
# archive.ubuntu.com's CDN (Fastly) is unreachable from this host (no IPv6
# route, IPv4 times out) -- security.ubuntu.com mirrors main/updates/backports
# too and is reachable, so point sources.list there instead.
RUN sed -i 's|http://archive.ubuntu.com|http://security.ubuntu.com|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

ENTRYPOINT ["python3", "run_detect.py"]
