from .backends import OllamaCaptionBackend, TransformersSiglipBackend
from .config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_RECAP_MODEL,
    DEFAULT_RECAP_PROMPT,
    DEFAULT_SIGLIP_MODEL_ID,
    DEFAULT_SIGLIP_NUM_WORKERS,
    SIGLIP_EMBEDDING_DIM,
)
from .discovery import discover_dataset, discover_video
from .exceptions import (
    AlignmentError,
    ArtifactValidationError,
    DatasetLayoutError,
    InvalidArgumentError,
    InvalidKeyframeInputError,
    KeyframePipelineError,
    ModelConfigurationError,
    ModelInferenceError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
    OutputAlreadyExistsError,
)
from .facade import KeyframePipeline
from .metadata import build_caption_inputs, load_caption_metadata
from .protocols import CaptionBackend, SiglipBackend
from .recap import (
    generate_recap,
    generate_recap_dataset,
    generate_recap_from_dir,
    load_recap_result,
)
from .siglip import (
    extract_siglip,
    extract_siglip_dataset,
    extract_siglip_from_dir,
    load_siglip_result,
)
from .types import (
    CaptionFailure,
    CaptionInput,
    CaptionMetadata,
    CaptionOutput,
    CaptionRecord,
    DatasetIndex,
    Keyframe,
    ObjectHint,
    OcrHint,
    RecapDatasetResult,
    RecapResult,
    SiglipDatasetResult,
    SiglipResult,
    VideoKeyframes,
)

__all__ = [name for name in globals() if not name.startswith("_")]

