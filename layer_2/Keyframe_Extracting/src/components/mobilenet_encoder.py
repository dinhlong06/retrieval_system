"""
mobilenet_encoder.py — MobileNetV3 Large Visual Encoder Wrapper.

Sử dụng MobileNetV3-Large pretrained từ torchvision (ImageNet weights).
Interface giống hệt BEiT3Encoder để có thể swap pipeline chỉ bằng config.

Embedding được lấy từ feature vector trước classifier head (960-dim),
sau đó L2-normalized để dùng với cosine similarity.

Không cần checkpoint thủ công — dùng torchvision.models.mobilenet_v3_large.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image
from torchvision import transforms, models


# Kích thước ảnh MobileNetV3 yêu cầu
_MOBILE_IMG_SIZE = 224

# Normalize theo ImageNet (giống BEiT-3 để embedding dùng chung threshold)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _build_transform() -> transforms.Compose:
    """Tạo pipeline transform: Resize → CenterCrop → ToTensor → Normalize."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(_MOBILE_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class MobileNetEncoder:
    """
    Wrapper MobileNetV3-Large để trích xuất visual embedding.

    Tự động tải ImageNet pretrained weights qua torchvision.
    Không cần file checkpoint thủ công.

    Interface thống nhất với BEiT3Encoder:
      - load()
      - encode_batch(frames) → (indices, embeddings)
      - unload()

    Args:
        device     : "cuda" hoặc "cpu". Mặc định tự detect.
        batch_size : Số frame encode trong mỗi lần forward.
    """

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = 64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model: nn.Module | None = None
        self._transform = _build_transform()

    def load(self) -> None:
        """
        Load MobileNetV3-Large với ImageNet pretrained weights.
        Loại bỏ classifier head để lấy feature vector (960-dim avgpool output).
        """
        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )
        # Giữ lại features + avgpool, bỏ classifier
        # Output shape: (B, 960, 1, 1) → sau AdaptiveAvgPool → (B, 960)
        self._model = nn.Sequential(
            base.features,
            base.avgpool,
            nn.Flatten(),        # (B, 960)
        )
        self._model.eval().to(self.device)
        print(f"[MobileNetEncoder] Loaded MobileNetV3-Large on {self.device}")

    def unload(self) -> None:
        """Xóa model khỏi GPU/memory để giải phóng VRAM."""
        self._model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def encode_batch(
        self,
        frames: List[Tuple[int, np.ndarray]],
    ) -> Tuple[List[int], np.ndarray]:
        """
        Encode một batch frame ảnh thành feature vector đã normalize.

        Args:
            frames : List (frame_idx, BGR image numpy array).

        Returns:
            Tuple:
              - indices    : List frame_idx (giữ nguyên thứ tự).
              - embeddings : numpy array shape (N, 960), L2-normalized, float32.
        """
        if self._model is None:
            raise RuntimeError("Gọi load() trước khi encode.")
        if not frames:
            return [], np.empty((0, 960), dtype=np.float32)

        indices = [idx for idx, _ in frames]
        tensors = [self._preprocess(img) for _, img in frames]

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[i : i + self.batch_size]).to(self.device)
                feat = self._model(batch)                    # (B, 960)
                feat = F.normalize(feat, dim=-1)             # L2 normalize
                all_embeddings.append(feat.cpu().float().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        return indices, embeddings

    def _preprocess(self, bgr_img: np.ndarray) -> torch.Tensor:
        """
        Chuyển BGR numpy array → RGB PIL Image → tensor đã normalize.

        Args:
            bgr_img : Frame ảnh dạng BGR (từ OpenCV).

        Returns:
            Tensor shape (3, 224, 224).
        """
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self._transform(pil)
