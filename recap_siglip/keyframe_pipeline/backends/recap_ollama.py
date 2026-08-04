from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from typing import Any, NoReturn

from PIL import Image

from ..config import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_RECAP_MODEL,
    DEFAULT_RECAP_PROMPT,
)
from ..exceptions import (
    ModelConfigurationError,
    ModelInferenceError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from ..types import CaptionInput, CaptionOutput

# Đo trên K01_V001: frame nhiều chữ (13-20 mục OCR, headline bản tin dài) cần
# 244-289 token cho caption 60-100 từ + mảng corrected_ocr. 256 nằm sát mép nên
# 3,6% frame bị cắt giữa JSON -> ModelInferenceError -> hỏng cả video.
NUM_PREDICT = 768

CAPTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "corrected_ocr": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["caption", "corrected_ocr"],
    "additionalProperties": False,
}


class OllamaCaptionBackend:
    """Caption images through a reusable local Ollama client."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_RECAP_MODEL,
        host: str = DEFAULT_OLLAMA_HOST,
        prompt: str = DEFAULT_RECAP_PROMPT,
        timeout_seconds: float = 120.0,
        keep_alive: str = "10m",
        max_retries: int = 2,
        client: object | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ModelConfigurationError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ModelConfigurationError("max_retries cannot be negative")
        self.model = model
        self.host = host
        self.prompt = prompt
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self.max_retries = max_retries
        self._client = client
        self._preflight_done = False
        self._preflight_lock = threading.Lock()

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from ollama import Client
        except Exception as exc:
            raise ModelConfigurationError(
                "Ollama Python SDK is missing; run: pip install ollama"
            ) from exc
        self._client = Client(host=self.host, timeout=self.timeout_seconds)
        return self._client

    def _raise_mapped(self, exc: Exception, operation: str) -> NoReturn:
        status = getattr(exc, "status_code", None)
        message = str(exc)
        if status == 404 or "model" in message.casefold() and "not found" in message.casefold():
            raise OllamaModelNotFoundError(
                f"Ollama model {self.model!r} is not available; run: "
                f"ollama pull {self.model}"
            ) from exc
        class_name = type(exc).__name__.casefold()
        if any(token in class_name for token in ("connect", "timeout", "network")):
            raise OllamaUnavailableError(
                f"Cannot reach Ollama at {self.host} during {operation}; "
                "start the Ollama application/service"
            ) from exc
        raise ModelInferenceError(
            f"Ollama {operation} failed for model {self.model!r}: {message}"
        ) from exc

    def _preflight(self) -> None:
        if self._preflight_done:
            return
        with self._preflight_lock:
            if self._preflight_done:
                return
            try:
                response = self._ensure_client().show(self.model)
            except Exception as exc:
                self._raise_mapped(exc, "preflight")
            capabilities = getattr(response, "capabilities", None)
            if capabilities is None and isinstance(response, dict):
                capabilities = response.get("capabilities")
            if capabilities is not None and "vision" not in capabilities:
                raise ModelConfigurationError(
                    f"Ollama model {self.model!r} does not expose the 'vision' "
                    "capability; choose a Text, Image model with a vision projector"
                )
            self._preflight_done = True

    @staticmethod
    def _response_content(response: Any) -> str:
        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if content is not None:
                return content
        if isinstance(response, dict):
            message = response.get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
        raise ModelInferenceError("Ollama response has no message.content string")

    @staticmethod
    def _region(
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> str:
        x1, y1, x2, y2 = bbox
        x = min(max((x1 + x2) / (2 * width), 0), 0.999999)
        y = min(max((y1 + y2) / (2 * height), 0), 0.999999)
        return f"{('left', 'center', 'right')[int(x * 3)]}-{('top', 'middle', 'bottom')[int(y * 3)]}"

    def _prompt(self, item: CaptionInput) -> str:
        ocr = [
            {"text": hint.text, "confidence": round(hint.confidence, 4)}
            for hint in item.ocr_hints
        ]
        objects = []
        if item.object_hints:
            with Image.open(item.image_path) as image:
                width, height = image.size
            objects = [
                {
                    "label": hint.label,
                    "confidence": round(hint.confidence, 4),
                    "region": self._region(hint.bbox, width, height),
                }
                for hint in item.object_hints
            ]
        return (
            f"{self.prompt}\n"
            f"OCR_CANDIDATES={json.dumps(ocr, ensure_ascii=False, separators=(',', ':'))}\n"
            f"OBJECT_CANDIDATES={json.dumps(objects, ensure_ascii=False, separators=(',', ':'))}"
        )

    @staticmethod
    def _parse_output(content: str) -> CaptionOutput:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ModelInferenceError("Ollama returned invalid structured JSON") from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != {"caption", "corrected_ocr"}
            or not isinstance(payload["caption"], str)
            or not payload["caption"].strip()
            or not isinstance(payload["corrected_ocr"], list)
            or any(not isinstance(text, str) or not text.strip() for text in payload["corrected_ocr"])
        ):
            raise ModelInferenceError("Ollama returned invalid structured caption data")
        corrected = tuple(dict.fromkeys(text.strip() for text in payload["corrected_ocr"]))
        return CaptionOutput(" ".join(payload["caption"].split()), corrected)

    def _caption_one(self, item: CaptionInput) -> CaptionOutput:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._ensure_client().chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": self._prompt(item),
                            "images": [str(item.image_path.resolve())],
                        }
                    ],
                    format=CAPTION_RESPONSE_SCHEMA,
                    think=False,
                    stream=False,
                    keep_alive=self.keep_alive,
                    options={"temperature": 0, "num_predict": NUM_PREDICT},
                )
                return self._parse_output(self._response_content(response))
            except (OllamaModelNotFoundError, OllamaUnavailableError, ModelInferenceError):
                raise
            except Exception as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                transient = status is not None and status >= 500
                class_name = type(exc).__name__.casefold()
                transient = transient or any(
                    token in class_name for token in ("connect", "timeout", "network")
                )
                if not transient or attempt == self.max_retries:
                    self._raise_mapped(exc, f"caption request for {item.image_path}")
                time.sleep(0.25 * (2**attempt))
        raise ModelInferenceError("Unreachable retry state") from last_error

    def caption_frames(
        self, inputs: Sequence[CaptionInput]
    ) -> Sequence[CaptionOutput]:
        self._preflight()
        return [self._caption_one(item) for item in inputs]
