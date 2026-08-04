from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Keyframe:
    video_id: str
    keyframe_id: str
    image_path: Path
    timestamp_ms: int | None = None


@dataclass(frozen=True, slots=True)
class VideoKeyframes:
    video_id: str
    keyframes: tuple[Keyframe, ...]


@dataclass(frozen=True, slots=True)
class DatasetIndex:
    root: Path
    videos: tuple[VideoKeyframes, ...]


@dataclass(frozen=True, slots=True)
class SiglipResult:
    video_id: str
    embeddings: np.ndarray
    keyframe_ids: tuple[str, ...]
    embedding_path: Path | None = None
    ids_path: Path | None = None


@dataclass(frozen=True, slots=True)
class SiglipDatasetResult:
    root: Path
    results: tuple[SiglipResult, ...]


@dataclass(frozen=True, slots=True)
class OcrHint:
    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ObjectHint:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class CaptionInput:
    image_path: Path
    ocr_hints: tuple[OcrHint, ...] = ()
    object_hints: tuple[ObjectHint, ...] = ()


@dataclass(frozen=True, slots=True)
class CaptionOutput:
    caption: str
    corrected_ocr: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaptionMetadata:
    ocr_by_frame: dict[str, tuple[OcrHint, ...]] | None = None
    objects_by_frame: dict[str, tuple[ObjectHint, ...]] | None = None


@dataclass(frozen=True, slots=True)
class CaptionRecord:
    video_id: str
    keyframe_id: str
    caption: str
    corrected_ocr: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CaptionFailure:
    keyframe_id: str
    error: str


@dataclass(frozen=True, slots=True)
class RecapResult:
    video_id: str
    records: tuple[CaptionRecord, ...]
    output_path: Path | None = None
    failures: tuple[CaptionFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class RecapDatasetResult:
    root: Path
    results: tuple[RecapResult, ...]

