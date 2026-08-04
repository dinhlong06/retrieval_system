"""
paddle_engine.py -- PaddleOCR detection + VietOCR recognition

PaddleOCR's own recognizer (any version: v3/v5/v6, any `lang`) cannot emit 70
of the Vietnamese accented characters (U+1EA0-U+1EF9 block) -- see memory
paddleocr-vietnamese-charset-gap -- so text with those marks comes out with
the letter deleted entirely, not just mis-accented. VietOCR's vgg_seq2seq has
a real Vietnamese vocabulary and doesn't have this gap (measured live on 168
real crops: 54.8% word-drop rate for PaddleOCR's own recognizer vs correctly
restoring nearly all of the same lines with VietOCR, e.g. "Thy din Tuyen
Quang" -> "Thủy điện Tuyên Quang" -- exact). It also doesn't hallucinate
Vietnamese marks onto English text (measured: "WELCOME"/"BREAKING NEWS"/
"SUBSCRIBE" all passed through unchanged).

So detection stays PaddleOCR, recognition moves to VietOCR on the detector's
crops. This mirrors the original (retired) Paddle+VietOCR pipeline's design,
reinstated because the all-Paddle replacement never actually fixed the
charset gap -- it only changed which recognizer version has it.

Detection uses the full `PaddleOCR()` combined pipeline (not the standalone
`TextDetection` class) -- its own `rec_texts`/`rec_scores` are kept as a
fallback, not discarded. Measured live: the standalone detector's raw
polygons are NOT the same boxes the combined pipeline uses (one frame:
standalone gave a 322x166px box spanning two unrelated lines merged
together, and dropped the HTV7HD logo entirely at the same confidence
threshold that keeps it in the combined pipeline) -- the combined pipeline
applies text-line-level postprocessing the bare detector doesn't. Reusing
its boxes keeps that working detection behavior.

The fallback matters because VietOCR is trained on printed/scanned document
text, not stylized broadcast graphics -- measured live on the HTV7HD channel
logo (a 3D/stylized wordmark, not plain ticker text): VietOCR reads it as
'mul' at 0.23 confidence, while PaddleOCR's own recognizer reads it correctly
at 0.992. Since the combined pipeline already computed rec_texts/rec_scores
for free in the same call used for boxes, using Paddle's answer when
VietOCR's confidence is below threshold costs nothing extra and recovers
lines VietOCR alone would silently drop.

Digit/time/percent *words* (clock overlays, page counters, "3%" style
figures, a year inside a longer ticker sentence) get a second, word-level
override even when VietOCR was confident overall: VietOCR was measured to
hallucinate or drop digits on these tokens ("9 TRIEU" -> "99 TRIEU", a frame
-edge-truncated "1" -> "la", verified against the source frames, not just
confidence scores). Paddle's recognizer reads clock/digit fonts more
reliably, so for each word position where Paddle's own word is a pure
digit/punctuation token and differs from VietOCR's, Paddle's word wins.

An earlier version also swapped short (<=2 letter) all-caps words mid
-sentence the same way, on the theory that codes/abbreviations ("MI", "CV")
give VietOCR's decoder too little context. Measured live in production: a
Paddle word being <=2 letters mid-sentence is usually its own vowel-drop bug
("Vu" -> "V", "Bo" -> "B", "My" -> "M") rather than a genuine short code --
that signal only holds when the *whole* line is short, not one word inside a
longer one, so it was removed from the per-word merge and kept only in the
whole-line override below.

The digit override only looks at Paddle's own word at each position (never
VietOCR's) -- the question being asked is "is Paddle's answer here a token
type it's known to read reliably", not anything about what VietOCR said.
Surrounding words are left untouched, and it's skipped entirely when the two
sides don't split into the same number of words -- misaligned counts (a
dropped or merged word on either side) make position-by-position swapping
unsafe, so the line is left as VietOCR produced it.

A second override takes a whole line, not just one word: a <=2-word,
fully-uppercase line (word counts matching between the two recognizers)
takes Paddle's answer outright. Measured against source frames (not
confidence scores) on 28 real disagreements in this bucket: Paddle correct
17/28 (61%), VietOCR correct 6/28 (21%), the rest a tie -- mostly English
brand/logo/signage text ("KIA", "BIA SAIGON", "GROW AGAIN", "60K", "OCTOBER
1ST-3RD") that VietOCR (trained on Vietnamese printed prose) misreads more
often than Paddle does. An earlier "any <=2-word phrase" idea (no all-caps requirement) was rejected
for false-positiving heavily on ordinary Vietnamese headlines ("Chu truong"
-> "Chu truon", "Nguyen" -> "Nguyn" -- Paddle's letter-drop bug from above,
just on shorter lines). All-caps alone doesn't fully protect against this
either: "TIN CHINH"/"TIEP THEO" are themselves all-caps, but they're saved
by the word-count-match requirement above -- Paddle collapses them to one
word ("TINCHINH"/"TIEPTHEO", no space detected) while VietOCR keeps two, so
the mismatched count already skips them without needing a separate check.
"""

from __future__ import annotations

import re
import unicodedata

from PIL import Image


def strip_diacritics(s: str) -> str:
    # Đ/đ are standalone letters, not base+combining, so NFD leaves them alone.
    s = s.replace("Đ", "D").replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


# A run of digits with the punctuation clocks/percentages/counters actually
# use, nothing else -- see module docstring.
_DIGIT_TOKEN_RE = re.compile(r"^[0-9][0-9:.,/%+-]*$")


def _merge_digit_words(text: str, p_text: str) -> str:
    """Word-by-word: a Paddle word that's a pure digit/time/percent token
    replaces VietOCR's word at the same position (see module docstring).
    Skipped entirely when word counts don't match -- position-by-position
    swapping isn't safe if either side dropped or merged a word.

    Deliberately digits-only, not also short all-caps codes: measured live
    that a Paddle word being <=2 letters mid-sentence is usually its own
    vowel-drop bug ("Vu" -> "V", "Bo" -> "B"), not a genuine short code --
    that signal only holds for a whole short line, see
    _prefer_paddle_short_line."""
    words, p_words = text.split(), p_text.split()
    if len(words) != len(p_words):
        return text
    return " ".join(
        p if _DIGIT_TOKEN_RE.match(p) and p != w else w
        for w, p in zip(words, p_words)
    )


def _is_allcaps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _prefer_paddle_short_line(text: str, p_text: str) -> str | None:
    """<=2-word, fully-uppercase line, word counts matching between the two
    recognizers: take Paddle's line outright (see module docstring). Returns
    None when the override doesn't apply, so the caller falls through to
    _merge_digit_words instead."""
    if text == p_text:
        return None
    words, p_words = text.split(), p_text.split()
    if not (1 <= len(words) <= 2) or len(words) != len(p_words):
        return None
    if not _is_allcaps(text):
        return None
    return p_text


# Padding around each detected polygon before cropping for recognition --
# same margin used in the ad-hoc VietOCR re-test (experiments/vietocr_test),
# avoids clipping tall/italic glyphs right at the box edge.
CROP_PAD = 3


class PaddleEngine:
    """PaddleOCR detector + VietOCR recognizer: confidence-filters on VietOCR's own prob."""

    def __init__(
        self,
        lang: str = "vi",  # unused, kept for config.yaml backward compat
        confidence_threshold: float = 0.5,
        ascii_only: bool = False,
        ocr_version: str | None = None,
        unclip_ratio: float | None = None,
    ) -> None:
        from paddleocr import PaddleOCR
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        self.confidence_threshold = confidence_threshold
        self.ascii_only = ascii_only

        det_kwargs = {"ocr_version": ocr_version} if ocr_version else {}
        if unclip_ratio is not None:
            det_kwargs["text_det_unclip_ratio"] = unclip_ratio
        self._det = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,  # CPU oneDNN path hits a PIR-attribute NotImplementedError on this build
            **det_kwargs,
        )

        cfg = Cfg.load_config_from_name("vgg_seq2seq")
        cfg["device"] = "cuda:0"
        cfg["predictor"]["beamsearch"] = False  # 3.3fps vs 0.9fps for beamsearch, no accuracy win -- see memory
        self._rec = Predictor(cfg)

    def run(self, source: "str | object") -> tuple[list[dict], list[dict]]:
        """
        Run detection+recognition on a file path or an in-memory BGR array
        (e.g. from frame_skip.preprocess).

        Returns
        -------
        tuple[list[dict], list[dict]]
            (vietocr_out, paddle_origin_out) -- both
            [{"text": str, "confidence": float, "box": [x1,y1,x2,y2]}, ...].
            vietocr_out is confidence-filtered on VietOCR's own recognition
            probability, falling back to Paddle's recognizer per box below
            threshold (see module docstring). paddle_origin_out is Paddle's
            own recognizer output, filtered on Paddle's own score instead --
            independent of the fallback, kept for comparing the two recognizers.
        """
        if isinstance(source, str):
            img = Image.open(source).convert("RGB")
        else:
            import numpy as np  # BGR (OpenCV convention) -> RGB
            img = Image.fromarray(np.asarray(source)[:, :, ::-1])

        det_result = self._det.predict(source)
        if not det_result:
            return [], []
        boxes = det_result[0].get("rec_boxes", [])
        paddle_texts = det_result[0].get("rec_texts", [])
        paddle_scores = det_result[0].get("rec_scores", [])

        w, h = img.size
        crops, coords, p_texts, p_scores = [], [], [], []
        for box, p_text, p_score in zip(boxes, paddle_texts, paddle_scores):
            bx1, by1, bx2, by2 = [int(v) for v in box]
            x1, y1 = max(0, bx1 - CROP_PAD), max(0, by1 - CROP_PAD)
            x2, y2 = min(w, bx2 + CROP_PAD), min(h, by2 + CROP_PAD)
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            coords.append((x1, y1, x2, y2))
            p_texts.append(p_text.strip())
            p_scores.append(float(p_score))

        if not crops:
            return [], []

        # One batched forward pass for all boxes in the frame instead of one
        # GPU call per box -- predict_batch buckets by resized width so
        # same-width crops share a single torch.cat'd forward pass.
        texts, probs = self._rec.predict_batch(crops, return_prob=True)

        out, out_paddle = [], []
        for (x1, y1, x2, y2), text, prob, p_text, p_score in zip(coords, texts, probs, p_texts, p_scores):
            text, prob = text.strip(), float(prob)
            if prob < self.confidence_threshold:
                # VietOCR unsure (e.g. a stylized logo, not printed text) --
                # fall back to Paddle's own answer for this box instead of
                # dropping the line, see module docstring.
                text, prob = p_text, p_score
            else:
                # VietOCR was confident overall, but a short all-caps line or
                # an embedded digit/time token is more reliably Paddle's,
                # see module docstring.
                text = _prefer_paddle_short_line(text, p_text) or _merge_digit_words(text, p_text)
            if text and prob >= self.confidence_threshold:
                out.append({
                    "text": strip_diacritics(text) if self.ascii_only else text,
                    "confidence": round(prob, 4),
                    "box": [x1, y1, x2, y2],
                })

            if p_text and p_score >= self.confidence_threshold:
                out_paddle.append({
                    "text": strip_diacritics(p_text) if self.ascii_only else p_text,
                    "confidence": round(p_score, 4),
                    "box": [x1, y1, x2, y2],
                })
        return out, out_paddle
