"""
corrector.py -- ising-calibration VLM repairs Paddle's raw text against the image

PP-OCRv6's charset is missing 88 of the 134 accented Vietnamese characters (the
whole U+1EA0-U+1EF9 block), so its recognizer cannot emit them at all and CTC
drops them: "Thừa Thiên - Huế kiến nghị" comes out as "Tha Thien - Hu kin ngh".
The VLM therefore repairs two things at once -- missing letters AND missing
marks -- with the image as ground truth and Paddle's text only as a scaffold.

The model answers under a strict "i|text" contract of exactly len(raw_texts)
numbered lines, so results map back by index. An earlier free-form version had
to re-match answers onto originals by string similarity, which silently dropped
any line the model chose not to return (measured: 104 lines left raw on frames
whose ticker was perfectly legible).

Prompt-side hallucination guardrails alone proved unreliable across two rounds
of A/B (measured: the reworded prompt did not reduce fabrication, and one run
fabricated more than the original). `truncated` is the deterministic backstop:
Paddle's own detection box says whether a line's raw text physically touches
the frame edge -- measured on 60 boxed frames, cut lines sit at EXACTLY x=0 or
x=frame_width, not "close to" -- so a line the box says is cut off can't have
its fix accepted if the fix grew past what a mark/spacing repair explains.
"""

from __future__ import annotations

import base64
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from difflib import SequenceMatcher
from pathlib import Path

import requests

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/ising-calibration-1.5-31b"

# Measured (see memory ising-calibration-api-rate-limit): this is a token
# bucket, not a hard cap -- short bursts look clean, sustained load 429s
# heavily past ~6 concurrent workers. Retry with this backoff instead of
# treating a 429 as a hard failure.
MAX_RETRIES = 8
RETRY_BACKOFF = 1.5

# requests' own `timeout=` sometimes fails to fire -- caught live via
# faulthandler on 2026-08-10: 4 worker threads stuck in ssl.py's socket
# read for 140s+ straight past the configured 60s, hanging the whole pool
# forever (no exception, no retry, no progress). This wraps each attempt in
# its own thread with a wall-clock deadline that WILL fire regardless of
# what the underlying socket does. The hung thread is abandoned (Python
# can't kill a thread) but no longer blocks the worker pool.
REQUEST_HARD_TIMEOUT = 90

PROMPT = """Đây là {n} dòng OCR thô đọc từ ảnh này. Chúng bị MẤT KÝ TỰ và MẤT DẤU tiếng Việt:

{raw}

Nhìn ảnh và sửa lại từng dòng cho đúng chữ đang hiển thị trong ảnh.

ĐỊNH DẠNG TRẢ VỀ (bắt buộc):
- Đúng {n} dòng, mỗi dòng dạng "số|nội dung", đánh số 1..{n} theo đúng thứ tự trên.
- Dòng nào không cần sửa vẫn phải chép lại nguyên văn.
- Không gộp dòng, không tách dòng, không bỏ dòng, không thêm dòng, không giải thích.

QUY TẮC SỬA:
- Mọi dòng chữ tiếng Việt BẮT BUỘC phải có dấu. KHÔNG BAO GIỜ trả về chữ Việt không dấu.
- Bổ sung ký tự bị mất và tách chữ dính nhau ("TINCHINH" -> "TIN CHÍNH", "WELM" -> "WELCOME").
- Giữ nguyên kiểu chữ HOA/thường như trong ảnh.
- Chữ chạy ngang (ticker) THƯỜNG BỊ CẮT CỤT ở mép trái/phải khung hình. Chép ĐÚNG phần nhìn
  thấy, kể cả khi câu đứt giữa chừng hoặc đứt giữa một từ. TUYỆT ĐỐI KHÔNG viết nốt câu,
  KHÔNG thêm mệnh đề, KHÔNG thêm dấu chấm cuối.
- Dòng có tiền tố "[CẮT] ": ảnh đo được là chữ đứt đúng tại đó (không phải đoán). Trong PHẦN
  NHÌN THẤY vẫn sửa dấu và bổ sung ký tự/tách chữ dính nhau BÌNH THƯỜNG như mọi dòng khác. Chỉ
  riêng ở ĐIỂM ĐỨT (cuối phần nhìn thấy): DỪNG hẳn tại đó -- không viết thêm chữ, từ, mệnh đề,
  số liệu, hay dấu chấm nào SAU điểm đó, kể cả khi đoán được câu đầy đủ nói gì. Tiền tố
  "[CẮT] " không phải một phần của chữ, đừng chép nó vào câu trả lời.
- Số: chỉ ghi số NHÌN THẤY trong ảnh (kể cả số nằm trong logo). Không suy ra số từ kiến thức
  bên ngoài hay từ ngữ cảnh của câu.
- Chỉ ghi chữ THỰC SỰ nhìn thấy trong ảnh. KHÔNG suy diễn, KHÔNG bịa."""


def _build_prompt(raw_texts: list[str], truncated: list[bool] | None) -> str:
    """
    Inline-tag truncated lines with a "[CẮT] " prefix instead of restructuring
    into labeled sections -- keeps the numbered-list shape the model already
    handles well (proven across v1-v3) instead of asking it to also parse a
    new two-part document structure. An earlier two-section version (NHÓM A /
    NHÓM B with separate rule blocks) measured WORSE overall: bad_format rate
    doubled and undiacritized lines nearly tripled (17->41/90 on a 60-frame
    run), including on untruncated lines that had nothing to do with the
    guard -- the added structural complexity cost more compliance than the
    truncation warning gained. Do not reintroduce section headers without
    re-measuring against that baseline.
    """
    truncated = truncated or [False] * len(raw_texts)
    numbered = "\n".join(
        f"{i}|{'[CẮT] ' if cut else ''}{t}" for i, (t, cut) in enumerate(zip(raw_texts, truncated), 1)
    )
    return PROMPT.format(n=len(raw_texts), raw=numbered)

# "3|Vụ sạt lở" -- also tolerates "3. " / "3) " / "3 |" so a minor format slip
# does not throw away an otherwise good frame. The trailing [|\s]* strips a
# closing delimiter the model sometimes echoes ("3|...nạn nhân|").
NUMBERED = re.compile(r"^\s*(\d+)\s*[|.)]\s*(.*?)[|\s]*$")

# A returned line must match its slot at least this well to be accepted. Loose
# because a heavily-repaired line ("OUIST TO NHA TRANG" -> "WELCOME THE 9
# MILLIONTH TOURIST TO NHA TRANG") is a poor string match to its own original.
MIN_RATIO = 0.45

# How much a truncated line is allowed to grow before a fix is treated as
# fabrication instead of a mark/spacing repair. Same ratio used all day to
# measure "keo_dai" (line-extension) on prompt A/B runs -- reused here as the
# enforcement threshold, not just a metric.
MAX_GROWTH_RATIO = 1.15
MAX_GROWTH_SLACK = 3


def _letter_count(s: str) -> int:
    return len(re.sub(r"\s", "", s))


def _unaccented(s: str) -> str:
    """Marks are what the model is adding, so compare on base letters only."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower()) if not unicodedata.combining(c)
    )


def _place_by_content(lines: list[str], raw_texts: list[str]) -> list[str]:
    """
    Assign the model's lines to slots by what they SAY, not by the number it
    wrote on them.

    The index contract proves coverage (the model addressed all N lines) but not
    identity: it enumerates what it sees in the image, which can be a different
    set of objects than Paddle detected. On one real frame it skipped a cut-off
    fragment, counted a second logo instead, still emitted 1..5, and shifted the
    headline one slot. Matching on content instead makes that impossible.

    Pairs are taken best-first so a strong match claims its slot before a weaker
    one can steal it. A slot nothing matches keeps its raw text -- which also
    drops invented lines that correspond to no detected line at all.
    """
    scored = sorted(
        (
            (SequenceMatcher(None, _unaccented(line), _unaccented(raw)).ratio(), li, ri)
            for li, line in enumerate(lines)
            for ri, raw in enumerate(raw_texts)
        ),
        reverse=True,
    )
    placed = list(raw_texts)
    used_lines: set[int] = set()
    used_slots: set[int] = set()
    for score, li, ri in scored:
        if score < MIN_RATIO:
            break
        if li in used_lines or ri in used_slots:
            continue
        placed[ri] = lines[li]
        used_lines.add(li)
        used_slots.add(ri)
    return placed


def correct(
    api_key: str,
    image_path: str,
    raw_texts: list[str],
    truncated: list[bool] | None = None,
) -> tuple[list[str], str]:
    """
    Repair raw_texts against the image. Always returns exactly len(raw_texts)
    lines so callers can zip confidences 1:1, plus a reason tag ("ok" or why the
    frame fell back to raw) -- 18% of a 714-frame run fell back silently before
    this was reported, which made the failure rate impossible to attribute.

    truncated[i] True means Paddle's detection box for raw_texts[i] touches the
    frame edge -- the caller computes this from `box`, since only it has the
    source image's dimensions. None (no box data available) disables the guard
    rather than assuming every line is untruncated.
    """
    if not raw_texts:
        return [], "empty_input"
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _build_prompt(raw_texts, truncated)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 1024,
        "temperature": 0,  # non-zero made A/B prompt/guard comparisons unreproducible, see memory
        "top_p": 0.95,
        "stream": False,
    }
    resp = None
    for attempt in range(MAX_RETRIES):
        # shutdown(wait=False): if fut.result() times out, the request thread
        # is abandoned rather than joined -- joining would just re-hang here.
        one = ThreadPoolExecutor(max_workers=1)
        fut = one.submit(
            requests.post,
            API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            json=payload,
            timeout=60,
        )
        try:
            resp = fut.result(timeout=REQUEST_HARD_TIMEOUT)
        except FutureTimeoutError:
            one.shutdown(wait=False)
            return raw_texts, "hang_timeout"
        one.shutdown(wait=False)
        if resp.status_code != 429:
            break
        time.sleep(RETRY_BACKOFF * (attempt + 1))
    if resp.status_code != 200:
        return raw_texts, f"http_{resp.status_code}"
    parsed = {}
    for line in resp.json()["choices"][0]["message"]["content"].splitlines():
        m = NUMBERED.match(line)
        if m and m.group(2):
            # The model sometimes echoes our own "[CẮT] " tag back despite the
            # rule telling it not to (measured live on K01_V001_000025_kf0001)
            # -- strip it here rather than trust the prompt alone.
            text = m.group(2)
            parsed[int(m.group(1))] = text[len("[CẮT] "):] if text.startswith("[CẮT] ") else text
    # Demanding the exact index set 1..n is the whole guardrail: a dropped line,
    # an invented extra line, or a repetition loop (~130 near-identical lines on
    # one frame in an earlier experiment) all fail it and fall back to raw.
    if set(parsed) != set(range(1, len(raw_texts) + 1)):
        return raw_texts, "bad_format"
    lines = [parsed[i] for i in range(1, len(raw_texts) + 1)]
    placed = _place_by_content(lines, raw_texts)
    reverted = 0
    if truncated:
        for i, (raw, cut) in enumerate(zip(raw_texts, truncated)):
            if not cut or placed[i] == raw:
                continue
            if _letter_count(placed[i]) > _letter_count(raw) * MAX_GROWTH_RATIO + MAX_GROWTH_SLACK:
                placed[i] = raw
                reverted += 1
    unmatched = sum(1 for p, r in zip(placed, raw_texts) if p == r and p not in lines)
    tag = "ok"
    if unmatched:
        tag += f"_{unmatched}unmatched"
    if reverted:
        tag += f"_{reverted}trunc"
    return placed, tag
