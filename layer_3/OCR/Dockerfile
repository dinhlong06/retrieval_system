# Stage 1 only (run_paddle.py) -- GPU detection (PaddleOCR) + recognition
# (VietOCR). Stage 2 (run_correct.py) just needs `requests`+`pyyaml` and a
# network path to NVIDIA's API, so it runs on the host, not in this image.
#
# Deliberately NOT based on paddlepaddle's own *-gpu image (10GB+ CUDA/cuDNN/
# TensorRT devel image) -- this host's root disk is chronically near-full
# (shared server, see memory), so this uses the much smaller nvidia
# cudnn-runtime base and pulls only the paddle GPU wheel from PaddlePaddle's
# own index. torch is added the same way (its own GPU wheel index) for
# VietOCR -- see memory paddleocr-vietnamese-charset-gap for why PaddleOCR's
# own recognizer, at any version, can't replace it.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

WORKDIR /workspace

# archive.ubuntu.com's CDN (Fastly) is unreachable from this host (no IPv6
# route, IPv4 times out) -- security.ubuntu.com mirrors main/updates/backports
# too and is reachable, so point sources.list there instead.
RUN sed -i 's|http://archive.ubuntu.com|http://security.ubuntu.com|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace

ENTRYPOINT ["python3", "run_paddle.py"]
