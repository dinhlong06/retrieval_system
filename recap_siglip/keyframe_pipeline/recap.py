from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .backends import OllamaCaptionBackend
from .config import DEFAULT_OLLAMA_HOST, DEFAULT_RECAP_MODEL
from .discovery import discover_dataset, discover_video, validate_keyframes
from .exceptions import AlignmentError, InvalidArgumentError, ModelInferenceError
from .io import (
    load_recap_result,
    load_siglip_result,
    preflight_targets,
    recap_target_path,
    save_recap_result,
)
from .metadata import build_caption_inputs, load_caption_metadata
from .protocols import CaptionBackend
from .types import (
    CaptionFailure,
    CaptionInput,
    CaptionMetadata,
    CaptionOutput,
    CaptionRecord,
    Keyframe,
    RecapDatasetResult,
    RecapResult,
    SiglipResult,
)


def _resolve_backend(
    backend: CaptionBackend | None,
    *,
    model: str,
    ollama_host: str,
) -> CaptionBackend:
    if backend is not None:
        if model != DEFAULT_RECAP_MODEL or ollama_host != DEFAULT_OLLAMA_HOST:
            raise InvalidArgumentError(
                "model/ollama_host cannot override an injected caption backend"
            )
        return backend
    return OllamaCaptionBackend(model=model, host=ollama_host)


def _validate_alignment(
    keyframes: Sequence[Keyframe], previous_output: SiglipResult
) -> None:
    video_id = keyframes[0].video_id
    expected_ids = tuple(item.keyframe_id for item in keyframes)
    if previous_output.video_id != video_id:
        raise AlignmentError(
            f"SigLIP video_id {previous_output.video_id!r} != {video_id!r}"
        )
    if previous_output.keyframe_ids != expected_ids:
        raise AlignmentError(
            f"SigLIP keyframe order/count does not match input for {video_id}"
        )


def _normalize_output(value: object, keyframe_id: str) -> CaptionOutput:
    if not isinstance(value, CaptionOutput):
        raise ModelInferenceError(
            f"Caption output for keyframe {keyframe_id!r} has the wrong type"
        )
    caption = " ".join(value.caption.split()).strip()
    if not caption:
        raise ModelInferenceError(f"Empty caption for keyframe {keyframe_id!r}")
    corrected = []
    for raw in value.corrected_ocr:
        if not isinstance(raw, str) or not raw.strip():
            raise ModelInferenceError(
                f"Invalid corrected OCR for keyframe {keyframe_id!r}"
            )
        text = raw.strip()
        if text not in corrected:
            corrected.append(text)
    return CaptionOutput(caption, tuple(corrected))


def _caption_one(
    backend: CaptionBackend, item: CaptionInput, keyframe_id: str
) -> CaptionOutput:
    values = backend.caption_frames((item,))
    if len(values) != 1:
        raise ModelInferenceError(
            "Caption backend must return one result for one frame request"
        )
    return _normalize_output(values[0], keyframe_id)


def _generate_captions(
    backend: CaptionBackend,
    inputs: Sequence[CaptionInput],
    keyframes: Sequence[Keyframe],
    max_concurrency: int,
) -> tuple[tuple[CaptionOutput | None, ...], dict[int, str]]:
    # Một frame hỏng không được giết cả video: 364 frame là ~30 phút, và model
    # thỉnh thoảng trả JSON cắt dở. Frame lỗi thành None, caller ghi phần còn lại.
    ordered: list[CaptionOutput | None] = [None] * len(inputs)
    errors: dict[int, str] = {}

    if max_concurrency == 1:
        for index, item in enumerate(inputs):
            try:
                ordered[index] = _caption_one(
                    backend, item, keyframes[index].keyframe_id
                )
            except ModelInferenceError as exc:
                errors[index] = str(exc)
        return tuple(ordered), errors

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                _caption_one, backend, item, keyframes[index].keyframe_id
            ): index
            for index, item in enumerate(inputs)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                ordered[index] = future.result()
            except ModelInferenceError as exc:
                errors[index] = str(exc)
    return tuple(ordered), errors


def generate_recap(
    keyframes: Sequence[Keyframe],
    *,
    backend: CaptionBackend | None = None,
    model: str = DEFAULT_RECAP_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    metadata: CaptionMetadata | None = None,
    previous_output: SiglipResult | None = None,
    output_path: Path | None = None,
    max_concurrency: int = 1,
    overwrite: bool = False,
    resume: bool = False,
) -> RecapResult:
    if max_concurrency <= 0:
        raise InvalidArgumentError("max_concurrency must be positive")
    if resume and output_path is None:
        raise InvalidArgumentError("resume needs output_path")
    items = validate_keyframes(keyframes)
    video_id = items[0].video_id
    if previous_output is not None:
        _validate_alignment(items, previous_output)
    done: dict[str, CaptionRecord] = {}
    if resume and Path(output_path).expanduser().resolve().is_file():
        done = {
            record.keyframe_id: record
            for record in load_recap_result(output_path).records
        }
    elif output_path is not None:
        preflight_targets((Path(output_path).expanduser().resolve(),), overwrite=overwrite)
    resolved_backend = _resolve_backend(
        backend, model=model, ollama_host=ollama_host
    )
    # Validate metadata against every keyframe, not just the pending ones: on a
    # resume the leftovers can all be frames without hints, which would trip the
    # "no matching keyframes" guard for the wrong reason.
    all_inputs = build_caption_inputs(items, metadata)
    keep = [index for index, item in enumerate(items) if item.keyframe_id not in done]
    pending = tuple(items[index] for index in keep)
    inputs = tuple(all_inputs[index] for index in keep)
    outputs, errors = _generate_captions(
        resolved_backend, inputs, pending, max_concurrency
    )
    fresh = {
        pending[index].keyframe_id: CaptionRecord(
            video_id, pending[index].keyframe_id, value.caption, value.corrected_ocr
        )
        for index, value in enumerate(outputs)
        if value is not None
    }
    result = RecapResult(
        video_id=video_id,
        records=tuple(
            done.get(item.keyframe_id) or fresh[item.keyframe_id]
            for item in items
            if item.keyframe_id in done or item.keyframe_id in fresh
        ),
        failures=tuple(
            CaptionFailure(pending[index].keyframe_id, message)
            for index, message in sorted(errors.items())
        ),
    )
    if output_path is not None:
        return save_recap_result(result, output_path, overwrite=overwrite or resume)
    return result


def generate_recap_from_dir(
    video_dir: Path,
    *,
    video_id: str | None = None,
    backend: CaptionBackend | None = None,
    model: str = DEFAULT_RECAP_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    ocr_json_path: Path | None = None,
    objects_json_path: Path | None = None,
    previous_embedding_path: Path | None = None,
    previous_ids_path: Path | None = None,
    output_path: Path | None = None,
    max_concurrency: int = 1,
    overwrite: bool = False,
    resume: bool = False,
) -> RecapResult:
    if (previous_embedding_path is None) != (previous_ids_path is None):
        raise InvalidArgumentError(
            "previous_embedding_path and previous_ids_path must be provided together"
        )
    video = discover_video(video_dir, video_id=video_id)
    metadata = load_caption_metadata(
        ocr_json_path=ocr_json_path,
        objects_json_path=objects_json_path,
    )
    previous = (
        load_siglip_result(previous_embedding_path, previous_ids_path)
        if previous_embedding_path is not None and previous_ids_path is not None
        else None
    )
    return generate_recap(
        video.keyframes,
        backend=backend,
        model=model,
        ollama_host=ollama_host,
        metadata=metadata,
        previous_output=previous,
        output_path=output_path,
        max_concurrency=max_concurrency,
        overwrite=overwrite,
        resume=resume,
    )


def generate_recap_dataset(
    dataset_root: Path,
    *,
    output_dir: Path,
    backend: CaptionBackend | None = None,
    model: str = DEFAULT_RECAP_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    ocr_json_path: Path | None = None,
    objects_json_path: Path | None = None,
    previous_siglip_dir: Path | None = None,
    max_concurrency: int = 1,
    overwrite: bool = False,
    resume: bool = False,
) -> RecapDatasetResult:
    if max_concurrency <= 0:
        raise InvalidArgumentError("max_concurrency must be positive")
    index = discover_dataset(dataset_root)
    targets = [recap_target_path(video.video_id, output_dir) for video in index.videos]
    if not resume:
        preflight_targets(targets, overwrite=overwrite)
    metadata = load_caption_metadata(
        ocr_json_path=ocr_json_path,
        objects_json_path=objects_json_path,
    )
    resolved_backend = _resolve_backend(
        backend, model=model, ollama_host=ollama_host
    )
    results: list[RecapResult] = []
    for video in index.videos:
        previous = None
        if previous_siglip_dir is not None:
            embedding_path, ids_path = (
                Path(previous_siglip_dir) / f"{video.video_id}.npy",
                Path(previous_siglip_dir) / f"{video.video_id}_ids.json",
            )
            previous = load_siglip_result(embedding_path, ids_path)
        results.append(
            generate_recap(
                video.keyframes,
                backend=resolved_backend,
                metadata=metadata,
                previous_output=previous,
                output_path=recap_target_path(video.video_id, output_dir),
                max_concurrency=max_concurrency,
                overwrite=overwrite,
                resume=resume,
            )
        )
    return RecapDatasetResult(root=index.root, results=tuple(results))


__all__ = [
    "generate_recap",
    "generate_recap_from_dir",
    "generate_recap_dataset",
    "load_recap_result",
]

