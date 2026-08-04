from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .config import SIGLIP_EMBEDDING_DIM
from .exceptions import ArtifactValidationError, OutputAlreadyExistsError
from .types import CaptionRecord, RecapResult, SiglipResult


def siglip_target_paths(video_id: str, output_dir: Path) -> tuple[Path, Path]:
    root = Path(output_dir).expanduser().resolve()
    return root / f"{video_id}.npy", root / f"{video_id}_ids.json"


def recap_target_path(video_id: str, output_dir: Path) -> Path:
    return Path(output_dir).expanduser().resolve() / f"{video_id}.jsonl"


def preflight_targets(paths: Iterable[Path], *, overwrite: bool) -> None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise OutputAlreadyExistsError(f"Output already exists: {joined}")


def validate_siglip_result(result: SiglipResult) -> None:
    embeddings = result.embeddings
    if not isinstance(embeddings, np.ndarray):
        raise ArtifactValidationError("embeddings must be a NumPy array")
    expected = (len(result.keyframe_ids), SIGLIP_EMBEDDING_DIM)
    if embeddings.ndim != 2 or embeddings.shape != expected:
        raise ArtifactValidationError(
            f"Invalid embedding shape {embeddings.shape}; expected {expected}"
        )
    if embeddings.dtype != np.float32:
        raise ArtifactValidationError(
            f"Invalid embedding dtype {embeddings.dtype}; expected float32"
        )
    if not np.isfinite(embeddings).all():
        raise ArtifactValidationError("Embeddings contain NaN or Inf")
    if len(set(result.keyframe_ids)) != len(result.keyframe_ids):
        raise ArtifactValidationError("Duplicate keyframe IDs in SigLIP result")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ArtifactValidationError("Embedding rows are not L2-normalized")


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        writer(temp_path)
        # mkstemp tạo file mode 600 và os.replace giữ nguyên mode đó, nên artifact
        # ra 600 -> layer sau chạy bằng user khác không đọc được. Đặt lại 644.
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def save_siglip_result(
    result: SiglipResult,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> SiglipResult:
    validate_siglip_result(result)
    embedding_path, ids_path = siglip_target_paths(result.video_id, output_dir)
    preflight_targets((embedding_path, ids_path), overwrite=overwrite)

    def write_array(temp_path: Path) -> None:
        with temp_path.open("wb") as handle:
            np.save(handle, result.embeddings, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

    def write_ids(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                list(result.keyframe_ids), handle, ensure_ascii=False, separators=(",", ":")
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write(embedding_path, write_array)
    _atomic_write(ids_path, write_ids)
    return SiglipResult(
        video_id=result.video_id,
        embeddings=result.embeddings,
        keyframe_ids=result.keyframe_ids,
        embedding_path=embedding_path,
        ids_path=ids_path,
    )


def load_siglip_result(
    embedding_path: Path,
    ids_path: Path,
    *,
    mmap_mode: str | None = None,
) -> SiglipResult:
    embedding_path = Path(embedding_path).expanduser().resolve()
    ids_path = Path(ids_path).expanduser().resolve()
    try:
        embeddings = np.load(
            embedding_path, mmap_mode=mmap_mode, allow_pickle=False
        )
    except Exception as exc:
        raise ArtifactValidationError(f"Cannot load {embedding_path}") from exc
    try:
        with ids_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ArtifactValidationError(f"Cannot load {ids_path}") from exc

    if not isinstance(payload, list) or not all(
        isinstance(value, str) for value in payload
    ):
        raise ArtifactValidationError("Invalid SigLIP IDs JSON schema")
    video_id = embedding_path.stem
    if ids_path.stem != f"{video_id}_ids":
        raise ArtifactValidationError(
            f"Artifact filenames do not pair up: {embedding_path.name} / {ids_path.name}"
        )

    result = SiglipResult(
        video_id=video_id,
        embeddings=embeddings,
        keyframe_ids=tuple(payload),
        embedding_path=embedding_path,
        ids_path=ids_path,
    )
    validate_siglip_result(result)
    return result


def validate_recap_result(result: RecapResult) -> None:
    if not result.records:
        raise ArtifactValidationError("ReCap result is empty")
    ids: set[str] = set()
    for record in result.records:
        if record.video_id != result.video_id:
            raise ArtifactValidationError("Mixed video IDs in ReCap result")
        if record.keyframe_id in ids:
            raise ArtifactValidationError(
                f"Duplicate keyframe ID {record.keyframe_id!r}"
            )
        ids.add(record.keyframe_id)
        if not isinstance(record.caption, str) or not record.caption.strip():
            raise ArtifactValidationError(
                f"Empty caption for keyframe {record.keyframe_id!r}"
            )
        if (
            not isinstance(record.corrected_ocr, tuple)
            or any(not isinstance(text, str) or not text.strip() for text in record.corrected_ocr)
            or len(set(record.corrected_ocr)) != len(record.corrected_ocr)
        ):
            raise ArtifactValidationError(
                f"Invalid corrected_ocr for keyframe {record.keyframe_id!r}"
            )


def save_recap_result(
    result: RecapResult,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> RecapResult:
    validate_recap_result(result)
    path = Path(output_path).expanduser().resolve()
    preflight_targets((path,), overwrite=overwrite)

    def write_jsonl(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in result.records:
                json.dump(
                    {
                        "video_id": record.video_id,
                        "keyframe_id": record.keyframe_id,
                        "caption": record.caption,
                        "corrected_ocr": list(record.corrected_ocr),
                    },
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_failures(temp_path: Path) -> None:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for failure in result.failures:
                json.dump(
                    {
                        "video_id": result.video_id,
                        "keyframe_id": failure.keyframe_id,
                        "error": failure.error,
                    },
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write(path, write_jsonl)
    failed_path = path.with_suffix(".failed.jsonl")
    if result.failures:
        _atomic_write(failed_path, write_failures)
    else:
        failed_path.unlink(missing_ok=True)
    return RecapResult(result.video_id, result.records, path, result.failures)


def load_recap_result(output_path: Path) -> RecapResult:
    path = Path(output_path).expanduser().resolve()
    records: list[CaptionRecord] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ArtifactValidationError(
                        f"Blank JSONL line at {path}:{line_number}"
                    )
                payload = json.loads(line)
                if not isinstance(payload, dict) or set(payload) != {
                    "video_id",
                    "keyframe_id",
                    "caption",
                    "corrected_ocr",
                }:
                    raise ArtifactValidationError(
                        f"Invalid JSONL schema at {path}:{line_number}"
                    )
                if not all(
                    isinstance(payload[key], str)
                    for key in ("video_id", "keyframe_id", "caption")
                ) or not isinstance(payload["corrected_ocr"], list):
                    raise ArtifactValidationError(
                        f"Invalid JSONL types at {path}:{line_number}"
                    )
                records.append(
                    CaptionRecord(
                        video_id=payload["video_id"],
                        keyframe_id=payload["keyframe_id"],
                        caption=payload["caption"],
                        corrected_ocr=tuple(payload["corrected_ocr"]),
                    )
                )
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError(f"Cannot load {path}") from exc

    if not records:
        raise ArtifactValidationError(f"ReCap artifact is empty: {path}")
    video_id = records[0].video_id
    if path.stem != video_id:
        raise ArtifactValidationError("ReCap filename does not match video_id")
    result = RecapResult(video_id, tuple(records), path)
    validate_recap_result(result)
    return result

