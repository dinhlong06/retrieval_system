"""
frame_skip.py -- cheap CPU pre-filters, ported from an earlier local Paddle+VietOCR
pipeline's engine.py + preprocessor.py (retired, see memory ocr_v2-package-refactor)

Full-dataset scale means most frames are either blank, blurry, or a near-
duplicate of the frame before it (a news banner sits still for seconds at a
time) -- running Paddle/GPU or calling the rate-limited ising API on every
single one wastes both. These checks run in <1ms combined and let a caller
skip or reuse the previous frame's result instead.
"""

from __future__ import annotations

import cv2
import numpy as np


def read_frame(image_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None
    return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def is_blank_frame(gray: np.ndarray, brightness_threshold: int = 15) -> bool:
    return float(np.mean(gray)) < brightness_threshold


def is_blurry(gray: np.ndarray, blur_threshold: float = 50.0) -> bool:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var()) < blur_threshold


def phash(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct_block = cv2.dct(resized)[:hash_size, :hash_size]
    median = np.median(dct_block[1:, 1:])
    return dct_block > median


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.sum(h1 != h2))


def apply_clahe(bgr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """CLAHE on luminance only (YCrCb) -- helps dark lifelog footage / washed-out broadcasts."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def crop_news_bands(
    bgr: np.ndarray, top_pct: float = 0.12, bottom_pct: float = 0.28, band_ratio: float = 2.0
) -> np.ndarray:
    """
    Crop to the top/bottom strip if it has news-ticker-like edge energy far
    above the mid-frame; otherwise return the frame unchanged (safe default
    for lifelog / signage / scene text where text can be anywhere).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    row_energy = np.mean(np.abs(sobelx), axis=1)

    top_end = max(1, int(h * top_pct))
    bot_start = min(h - 1, int(h * (1.0 - bottom_pct)))
    mid_energy = np.mean(row_energy[top_end:bot_start]) + 1e-6

    bands = []
    if np.mean(row_energy[:top_end]) / mid_energy >= band_ratio:
        bands.append((0, top_end))
    if np.mean(row_energy[bot_start:]) / mid_energy >= band_ratio:
        bands.append((bot_start, h))

    if not bands:
        return bgr
    strips = [bgr[r0:r1, :] for r0, r1 in bands]
    return np.vstack(strips) if len(strips) > 1 else strips[0]


def preprocess(bgr: np.ndarray) -> np.ndarray:
    return crop_news_bands(apply_clahe(bgr))
