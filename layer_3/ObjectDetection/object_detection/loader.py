"""
loader.py — Frame folder / ZIP archive scanner for object detection
Discovers all supported image files in the input directory or ZIP file,
sorts them by name, and returns (frame_id, absolute_path) pairs ready for processing.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
)


def load_frames(input_path: str) -> list[tuple[str, str]]:
    """
    Scan *input_path* for image files and return them sorted by filename.
    If *input_path* is a `.zip` file, it is automatically extracted to a folder
    `<zip_stem>_extracted` next to the zip file.

    Parameters
    ----------
    input_path : str
        Path to a directory OR a `.zip` file containing video frames.

    Returns
    -------
    list[tuple[str, str]]
        Sorted list of ``(frame_id, absolute_path)`` tuples where
        ``frame_id`` is the file stem (e.g. ``"L21_VID01_0001"``).
    """
    target = Path(input_path).resolve()

    if not target.exists():
        raise FileNotFoundError(f"Input path not found: {target}")

    # Auto-extract if input is a ZIP file
    if target.is_file() and target.suffix.lower() == ".zip":
        extract_dir = target.parent / f"{target.stem}_extracted"
        if not extract_dir.exists():
            logger.info("Extracting '%s' -> '%s' ...", target, extract_dir)
            print(f"[->] Extracting zip file '{target.name}' ...")
            with zipfile.ZipFile(target, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"[✓] Extracted to: {extract_dir}")
        target = extract_dir

    if not target.is_dir():
        raise NotADirectoryError(f"Expected a directory or zip file, got: {target}")

    # Discover images (supports both flat and nested directories inside zip)
    found_files = [
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    # Per-video keyframe folders restart numbering at 1 per video (e.g.
    # dataset_batch1/keyframe/keyframes/<video>/001.jpg), so the bare stem
    # alone collides across videos -- prefix with the parent folder name
    # unless the filename already starts with it.
    frames: list[tuple[str, str]] = sorted(
        (p.stem if p.stem.startswith(p.parent.name) else f"{p.parent.name}_{p.stem}", str(p))
        for p in found_files
    )

    if not frames:
        raise ValueError(
            f"No supported image files found in '{target}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    logger.info("Found %d frames in '%s'.", len(frames), target)
    return frames
