"""
metrics.py — Tính các chỉ số đánh giá chất lượng Keyframe Extraction.

Gồm hai nhóm metric:
  1. Unsupervised (luôn tính được, không cần Ground Truth):
       - Diversity Score: mức độ đa dạng giữa các keyframe trong cùng shot.
       - Redundancy Score: ngược lại với Diversity.
       - Coverage Score: % thời lượng shot được đại diện bởi keyframe.

  2. Supervised (chỉ dùng khi có Ground Truth ảnh keyframe):
       - Precision, Recall, F1 dựa trên Hungarian Matching.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from src.core.models import ShotKeyframes, ShotRecord


# ---------------------------------------------------------------------------
# Unsupervised metrics
# ---------------------------------------------------------------------------

def compute_diversity_score(embeddings: np.ndarray) -> float:
    """
    Tính Diversity Score = 1 - avg(cosine_similarity) giữa tất cả cặp keyframe.

    Cao (gần 1.0) = các keyframe rất khác nhau về ngữ nghĩa (tốt).
    Thấp (gần 0.0) = các keyframe giống nhau (redundant).

    Args:
        embeddings : numpy array shape (N, D), đã L2-normalized.

    Returns:
        Diversity score trong khoảng [0.0, 1.0].
        Trả về 1.0 nếu chỉ có 1 keyframe (không có cặp nào để so sánh).
    """
    n = len(embeddings)
    if n <= 1:
        return 1.0

    # Ma trận cosine similarity (đã normalize → chỉ cần dot product)
    sim_matrix = embeddings @ embeddings.T  # shape (N, N)

    # Lấy upper triangle (không kể đường chéo chính)
    upper_idx = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_idx]

    avg_sim = float(np.mean(pairwise_sims))
    return max(0.0, 1.0 - avg_sim)


def compute_redundancy_score(embeddings: np.ndarray) -> float:
    """
    Redundancy Score = avg(cosine_similarity) giữa tất cả cặp keyframe.
    Là nghịch đảo của Diversity Score.

    Args:
        embeddings : numpy array shape (N, D), đã L2-normalized.

    Returns:
        Redundancy score trong khoảng [0.0, 1.0].
    """
    return 1.0 - compute_diversity_score(embeddings)


def compute_coverage_score(
    shot: ShotRecord,
    keyframe_indices: List[int],
    window_sec: float = 1.0,
) -> float:
    """
    Coverage Score = % thời gian của shot được "cover" bởi ít nhất một keyframe.

    Mỗi keyframe cover một cửa sổ [frame_idx - w/2, frame_idx + w/2]
    trong đó w = window_sec * fps.

    Args:
        shot             : ShotRecord chứa start_frame, end_frame, fps.
        keyframe_indices : Danh sách frame_idx của các keyframe đã chọn.
        window_sec       : Bán kính coverage mỗi keyframe (giây).

    Returns:
        Coverage ratio trong khoảng [0.0, 1.0].
    """
    total = shot.total_frames
    if total == 0 or not keyframe_indices:
        return 0.0

    window_frames = int(window_sec * shot.fps)
    covered = np.zeros(total, dtype=bool)

    for kf_idx in keyframe_indices:
        # Chuyển frame_idx tuyệt đối → chỉ số tương đối trong shot
        rel = kf_idx - shot.start_frame
        lo = max(0, rel - window_frames // 2)
        hi = min(total - 1, rel + window_frames // 2)
        covered[lo : hi + 1] = True

    return float(covered.sum()) / total


# ---------------------------------------------------------------------------
# Supervised metrics (Hungarian Matching)
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Tính ma trận cosine similarity giữa hai tập embedding.

    Args:
        a : shape (M, D), L2-normalized.
        b : shape (N, D), L2-normalized.

    Returns:
        Ma trận shape (M, N).
    """
    return a @ b.T


def evaluate_with_ground_truth(
    pred_embeddings: np.ndarray,
    gt_embeddings: np.ndarray,
    threshold: float = 0.8,
) -> Tuple[float, float, float]:
    """
    Tính Precision, Recall, F1 dựa trên Nearest Neighbor Matching.

    Mỗi GT keyframe được "match" với predicted keyframe có similarity cao nhất.
    Nếu max similarity >= threshold thì coi là True Positive.

    Args:
        pred_embeddings : shape (P, D) — embedding các keyframe dự đoán, L2-normalized.
        gt_embeddings   : shape (G, D) — embedding Ground Truth, L2-normalized.
        threshold       : Ngưỡng cosine similarity để coi là match (default 0.8).

    Returns:
        Tuple (precision, recall, f1) trong khoảng [0.0, 1.0].
    """
    if len(pred_embeddings) == 0 or len(gt_embeddings) == 0:
        return 0.0, 0.0, 0.0

    sim_matrix = _cosine_similarity_matrix(pred_embeddings, gt_embeddings)
    # shape (P, G)

    # Recall: với mỗi GT, có predicted nào match không?
    best_sim_per_gt = sim_matrix.max(axis=0)        # shape (G,)
    tp_recall = (best_sim_per_gt >= threshold).sum()
    recall = float(tp_recall) / len(gt_embeddings)

    # Precision: với mỗi predicted, có GT nào match không?
    best_sim_per_pred = sim_matrix.max(axis=1)      # shape (P,)
    tp_precision = (best_sim_per_pred >= threshold).sum()
    precision = float(tp_precision) / len(pred_embeddings)

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1
