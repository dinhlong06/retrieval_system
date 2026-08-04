from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .backends import OllamaCaptionBackend, TransformersSiglipBackend
from .config import DEFAULT_OLLAMA_HOST, DEFAULT_RECAP_MODEL, DEFAULT_SIGLIP_MODEL_ID
from .discovery import discover_dataset
from .io import preflight_targets, recap_target_path, siglip_target_paths
from .metadata import load_caption_metadata
from .protocols import CaptionBackend, SiglipBackend
from .recap import generate_recap
from .siglip import extract_siglip
from .types import (
    CaptionMetadata,
    Keyframe,
    RecapDatasetResult,
    RecapResult,
    SiglipDatasetResult,
    SiglipResult,
)


class KeyframePipeline:
    def __init__(
        self,
        *,
        siglip_backend: SiglipBackend | None = None,
        caption_backend: CaptionBackend | None = None,
        siglip_model_id: str = DEFAULT_SIGLIP_MODEL_ID,
        siglip_device: str | None = None,
        recap_model: str = DEFAULT_RECAP_MODEL,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
    ) -> None:
        self.siglip_backend = siglip_backend or TransformersSiglipBackend(
            model_id=siglip_model_id, device=siglip_device
        )
        self.caption_backend = caption_backend or OllamaCaptionBackend(
            model=recap_model, host=ollama_host
        )

    def run_siglip(
        self,
        keyframes: Sequence[Keyframe],
        *,
        output_dir: Path | None = None,
        batch_size: int = 32,
        overwrite: bool = False,
    ) -> SiglipResult:
        return extract_siglip(
            keyframes,
            backend=self.siglip_backend,
            output_dir=output_dir,
            batch_size=batch_size,
            overwrite=overwrite,
        )

    def run_recap(
        self,
        keyframes: Sequence[Keyframe],
        *,
        metadata: CaptionMetadata | None = None,
        previous_output: SiglipResult | None = None,
        output_path: Path | None = None,
        max_concurrency: int = 1,
        overwrite: bool = False,
        resume: bool = False,
    ) -> RecapResult:
        return generate_recap(
            keyframes,
            backend=self.caption_backend,
            metadata=metadata,
            previous_output=previous_output,
            output_path=output_path,
            max_concurrency=max_concurrency,
            overwrite=overwrite,
            resume=resume,
        )

    def run_dataset(
        self,
        dataset_root: Path,
        *,
        output_root: Path,
        ocr_json_path: Path | None = None,
        objects_json_path: Path | None = None,
        siglip_batch_size: int = 32,
        recap_max_concurrency: int = 1,
        overwrite: bool = False,
        resume: bool = False,
    ) -> tuple[SiglipDatasetResult, RecapDatasetResult]:
        index = discover_dataset(dataset_root)
        root = Path(output_root).expanduser().resolve()
        siglip_dir = root / "siglip"
        recap_dir = root / "recap"
        targets = [
            path
            for video in index.videos
            for path in (
                *siglip_target_paths(video.video_id, siglip_dir),
                recap_target_path(video.video_id, recap_dir),
            )
        ]
        preflight_targets(targets, overwrite=overwrite)
        metadata = load_caption_metadata(
            ocr_json_path=ocr_json_path,
            objects_json_path=objects_json_path,
        )

        siglip_results = tuple(
            self.run_siglip(
                video.keyframes,
                output_dir=siglip_dir,
                batch_size=siglip_batch_size,
                overwrite=overwrite,
            )
            for video in index.videos
        )
        recap_results = tuple(
            self.run_recap(
                video.keyframes,
                metadata=metadata,
                previous_output=siglip_results[index_value],
                output_path=recap_target_path(video.video_id, recap_dir),
                max_concurrency=recap_max_concurrency,
                overwrite=overwrite,
                resume=resume,
            )
            for index_value, video in enumerate(index.videos)
        )
        return (
            SiglipDatasetResult(index.root, siglip_results),
            RecapDatasetResult(index.root, recap_results),
        )

