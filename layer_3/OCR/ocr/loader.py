"""loader.py -- frame folder scanner, shared by both pipeline stages."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def load_frames(input_dir: str) -> list[tuple[str, str]]:
    """Return sorted (frame_id, absolute_path) pairs for every supported image in input_dir.

    frame_id is prefixed with the parent folder name unless the filename
    already starts with it -- per-video keyframe folders (e.g.
    dataset_batch1/keyframe/keyframes/<video>/017.jpg) restart numbering at
    001 per video, so the bare stem alone collides across every video.
    """
    folder = Path(input_dir).resolve()
    pairs = []
    for p in folder.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        frame_id = p.stem if p.stem.startswith(p.parent.name) else f"{p.parent.name}_{p.stem}"
        pairs.append((frame_id, str(p)))
    return sorted(pairs)
