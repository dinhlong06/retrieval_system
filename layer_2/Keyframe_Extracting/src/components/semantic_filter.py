"""
semantic_filter.py — Lọc Keyframe dựa trên ngữ nghĩa (Semantic Filtering).

Nhận đầu vào là embedding của Candidate Frames (từ DAKE hoặc toàn bộ frame).
Loại bỏ các frame quá giống nhau dựa trên cosine similarity.
Trả về list frame_idx là Keyframes cuối cùng.

Thuật toán:
  1. Sắp xếp candidate frames theo thứ tự thời gian (frame_idx tăng dần).
  2. Duyệt tuần tự: frame đầu tiên luôn được giữ lại.
  3. Frame tiếp theo được giữ lại nếu cosine similarity với keyframe
     đã chọn gần nhất < threshold (đủ khác biệt).

Tham khảo: KEYFRAME_EXTRACTING.md, Section 8.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


class SemanticFilter:
    """
    Lọc frame dựa trên ngữ nghĩa: chỉ giữ lại frame đủ khác biệt.

    Args:
        similarity_threshold : Cosine similarity tối đa để giữ frame.
            Frame mới được chọn nếu sim < threshold so với KF trước.
            Mặc định 0.90 theo spec KEYFRAME_EXTRACTING.md.
    """

    def __init__(self, similarity_threshold: float = 0.90):
        self.threshold = similarity_threshold

    def filter(
        self,
        frame_indices: List[int],
        embeddings: np.ndarray,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Duyệt tuần tự và loại bỏ frame quá giống với keyframe trước.

        Args:
            frame_indices : Danh sách frame_idx (tuyệt đối), đã sort tăng dần.
            embeddings    : numpy array shape (N, D), L2-normalized.
                            embeddings[i] tương ứng với frame_indices[i].

        Returns:
            Tuple:
              - selected_indices   : List frame_idx của các frame được giữ lại (Keyframes).
              - selected_embeddings: numpy array shape (M, D) — embedding tương ứng
                                      1-1 với selected_indices (đã tính sẵn, không encode lại).
            Luôn có ít nhất 1 frame nếu input không rỗng.
        """
        if len(frame_indices) == 0:
            return [], embeddings

        selected_indices: List[int] = []
        selected_embeddings: List[np.ndarray] = []

        # Frame đầu tiên luôn được giữ
        selected_indices.append(frame_indices[0])
        selected_embeddings.append(embeddings[0])

        for i in range(1, len(frame_indices)):
            emb = embeddings[i]

            # Tính cosine similarity với keyframe đã chọn gần nhất
            last_kf_emb = selected_embeddings[-1]
            sim = float(np.dot(emb, last_kf_emb))    # đã normalize → dot = cosine

            # Chỉ giữ lại nếu đủ khác biệt
            if sim < self.threshold:
                selected_indices.append(frame_indices[i])
                selected_embeddings.append(emb)

        return selected_indices, np.stack(selected_embeddings)

    def filter_with_scores(
        self,
        frame_indices: List[int],
        embeddings: np.ndarray,
    ) -> Tuple[List[int], List[float]]:
        """
        Giống filter() nhưng trả thêm similarity score của từng frame bị loại.
        Dùng cho debug và visualization.

        Returns:
            Tuple:
              - selected_indices : List frame_idx được giữ lại.
              - similarity_scores: List cosine similarity của từng frame với KF trước.
                                   Frame đầu tiên có score = -1.0 (không có KF trước).
        """
        if len(frame_indices) == 0:
            return [], []

        selected_indices: List[int] = [frame_indices[0]]
        selected_embeddings: List[np.ndarray] = [embeddings[0]]
        all_scores: List[float] = [-1.0]

        for i in range(1, len(frame_indices)):
            emb = embeddings[i]
            last_kf_emb = selected_embeddings[-1]
            sim = float(np.dot(emb, last_kf_emb))
            all_scores.append(sim)

            if sim < self.threshold:
                selected_indices.append(frame_indices[i])
                selected_embeddings.append(emb)

        return selected_indices, all_scores
