from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from .exceptions import ArtifactValidationError
from .types import (
    CaptionInput,
    CaptionMetadata,
    Keyframe,
    ObjectHint,
    OcrHint,
)

Hint = TypeVar("Hint", OcrHint, ObjectHint)


def _load_array(path: Path) -> list[object]:
    resolved = Path(path).expanduser().resolve()
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ArtifactValidationError(f"Cannot load metadata {resolved}") from exc
    if not isinstance(payload, list):
        raise ArtifactValidationError(f"Metadata root must be an array: {resolved}")
    return payload


def _confidence(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
    ):
        raise ArtifactValidationError(f"Invalid confidence: {value!r}")
    return float(value)


def _bbox(value: object) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(number, bool) or not isinstance(number, (int, float)) for number in value)
    ):
        raise ArtifactValidationError(f"Invalid bbox: {value!r}")
    x1, y1, x2, y2 = (float(number) for number in value)
    if x2 < x1 or y2 < y1:
        raise ArtifactValidationError(f"Invalid bbox order: {value!r}")
    return x1, y1, x2, y2


def _index(
    path: Path,
    field: str,
    parser: Callable[[object], Hint],
) -> dict[str, tuple[Hint, ...]]:
    result: dict[str, tuple[Hint, ...]] = {}
    for record in _load_array(path):
        if (
            not isinstance(record, dict)
            or set(record) != {"frame_id", field}
            or not isinstance(record["frame_id"], str)
            or not record["frame_id"]
            or not isinstance(record[field], list)
        ):
            raise ArtifactValidationError(f"Invalid {field} metadata record")
        frame_id = record["frame_id"]
        if frame_id in result:
            raise ArtifactValidationError(f"Duplicate frame_id {frame_id!r}")
        result[frame_id] = tuple(parser(item) for item in record[field])
    return result


def _ocr_hint(value: object) -> OcrHint:
    if (
        not isinstance(value, dict)
        or set(value) not in (
            {"text", "confidence"},
            {"text", "confidence", "bbox"},
        )
        or not isinstance(value["text"], str)
        or not value["text"].strip()
    ):
        raise ArtifactValidationError("Invalid OCR hint")
    if "bbox" in value:
        _bbox(value["bbox"])
    return OcrHint(value["text"], _confidence(value["confidence"]))


def _object_hint(value: object) -> ObjectHint:
    if (
        not isinstance(value, dict)
        or set(value) != {"label", "confidence", "bbox"}
        or not isinstance(value["label"], str)
        or not value["label"].strip()
    ):
        raise ArtifactValidationError("Invalid object hint")
    return ObjectHint(
        value["label"],
        _confidence(value["confidence"]),
        _bbox(value["bbox"]),
    )


def load_caption_metadata(
    *,
    ocr_json_path: Path | None = None,
    objects_json_path: Path | None = None,
) -> CaptionMetadata:
    return CaptionMetadata(
        ocr_by_frame=(
            _index(ocr_json_path, "texts", _ocr_hint)
            if ocr_json_path is not None
            else None
        ),
        objects_by_frame=(
            _index(objects_json_path, "objects", _object_hint)
            if objects_json_path is not None
            else None
        ),
    )


def build_caption_inputs(
    keyframes: Sequence[Keyframe],
    metadata: CaptionMetadata | None = None,
) -> tuple[CaptionInput, ...]:
    resolved = metadata or CaptionMetadata()
    ids = {item.keyframe_id for item in keyframes}
    for name, index in (
        ("OCR", resolved.ocr_by_frame),
        ("object", resolved.objects_by_frame),
    ):
        if index is not None and not ids.intersection(index):
            raise ArtifactValidationError(
                f"Supplied {name} metadata has no matching keyframes"
            )
    return tuple(
        CaptionInput(
            image_path=item.image_path,
            ocr_hints=(
                resolved.ocr_by_frame.get(item.keyframe_id, ())
                if resolved.ocr_by_frame is not None
                else ()
            ),
            object_hints=(
                resolved.objects_by_frame.get(item.keyframe_id, ())
                if resolved.objects_by_frame is not None
                else ()
            ),
        )
        for item in keyframes
    )
