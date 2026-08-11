"""
qwen_client.py -- gọi Qwen3-VL đang chạy sẵn trong vLLM server (run_vllm.sh)
qua OpenAI-compatible API, thay vì tự nạp model trong tiến trình compare.py.

Vì sao đổi từ HF transformers sang vLLM: đo trực tiếp trên GPU của host này
(RTX 2080Ti 11GB, dùng chung, Turing nên không có bf16 tensor core) cho thấy
transformers generate() + bitsandbytes NF4 chỉ đạt ~1.2 token/giây -- 855
frame sẽ mất cả chục tiếng. vLLM giữ model thường trực trên GPU (không nạp
lại mỗi lần gọi) và dùng kernel CUDA tối ưu hơn hẳn generate() từng bước của
transformers.

Không dùng bf16: vLLM chặn cứng bf16 dưới compute capability 8.0 (2080Ti là
7.5), server chạy --dtype float16 (xem run_vllm.sh).
"""

from __future__ import annotations

import base64
from pathlib import Path

DEFAULT_PROMPT = "Trích xuất toàn bộ văn bản trong ảnh. Chỉ trả về văn bản, giữ nguyên dấu tiếng Việt."


class QwenClient:
    def __init__(self, base_url: str, prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 256) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = self.client.models.list().data[0].id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

    def run(self, image_path: str) -> str:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": self.prompt},
                ],
            }],
            max_tokens=self.max_new_tokens,
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()
