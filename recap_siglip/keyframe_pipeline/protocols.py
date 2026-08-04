from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .types import CaptionInput, CaptionOutput


class SiglipBackend(Protocol):
    @property
    def embedding_dim(self) -> int: ...

    def encode_paths(self, paths: Sequence[Path], batch_size: int) -> np.ndarray: ...


class CaptionBackend(Protocol):
    def caption_frames(
        self, inputs: Sequence[CaptionInput]
    ) -> Sequence[CaptionOutput]: ...

