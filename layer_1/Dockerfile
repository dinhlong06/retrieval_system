# Layer 1 (GPU): TransNetV2 (Shot Detection, TensorFlow) + PhoWhisper (ASR, PyTorch)
# Base: TensorFlow GPU image (đã có CUDA/cuDNN khớp để chạy TransNetV2 trên GPU)
FROM tensorflow/tensorflow:2.13.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# TF 2.13 chiếm sạch VRAM của GPU ngay khi tạo TransNetV2, rồi PhoWhisper trong
# cùng process không còn chỗ. Máy dùng chung 8 GPU nên bắt buộc bật growth.
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_CPP_MIN_LOG_LEVEL=2

# ffmpeg: bắt buộc để tách audio (extract_audio_to_wav) trước khi đưa vào PhoWhisper
# git-lfs: repo TransNetV2 lưu file weight (*.pb/*.index/*.data-*/*.h5) qua Git LFS,
# thiếu git-lfs thì clone chỉ ra con trỏ text rỗng thay vì weight thật.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git git-lfs curl xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# ffmpeg 4.2.7 của Ubuntu 20.04 build KHÔNG có libdav1d, nên giải mã AV1 rơi vào
# libaom (decoder tham chiếu): 61 fps so với 290 fps của dav1d. 60% dataset là AV1.
# Bản static có dav1d; giải mã AV1/H.264 là normative nên pixel giống hệt từng byte.
RUN curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
        | tar -xJ -C /tmp \
    && mv /tmp/ffmpeg-*-static/ffmpeg /tmp/ffmpeg-*-static/ffprobe /usr/local/bin/ \
    && rm -rf /tmp/ffmpeg-*-static

# PyTorch (CUDA 11.8, khớp base image) cho PhoWhisper qua transformers pipeline
# torchaudio: Silero VAD (read_audio) dùng để load/resample wav
RUN pip install --no-cache-dir \
        torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# TransNetV2 (chỉ dùng module inference + weights của repo gốc, không có setup.py
# nên clone rồi thêm vào PYTHONPATH thay vì pip install)
RUN git clone --depth 1 https://github.com/soCzech/TransNetV2.git /opt/TransNetV2
ENV PYTHONPATH="/opt/TransNetV2/inference:${PYTHONPATH}"

COPY gpu_shot_and_asr.py /workspace/gpu_shot_and_asr.py

ENTRYPOINT ["python3", "gpu_shot_and_asr.py"]
