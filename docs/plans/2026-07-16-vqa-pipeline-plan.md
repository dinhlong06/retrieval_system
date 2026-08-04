# VQA/KIS/AVS/KISC Pipeline Plan (seeded from NII-UIT VBS2025/2026 + U-CESE AIC2025 papers)

Sources:
- `docs/paper/_MMM2026__NII_UIT_at_VBS2026...pdf` (Bao Tran et al., MMM2026) — the
  VBS2026 system paper. **System/architecture paper**: names new modules (Answer Span
  Prediction, Candidate Answer Suggestion, In-Video Retrieval) but gives almost no
  implementation detail for them.
- `docs/paper/NII-UIT-at-VBS2025-...pdf` (Gia et al., MMM2025) — the **predecessor**
  system (won VBS2025, is reference [2] in the 2026 paper). Much more implementation
  detail for the shared offline/retrieval backbone the 2026 paper builds on.
- `docs/paper/2605.23274v1.pdf` — **U-CESE** (Le, Nguyen, Lam, Dang, Le; arXiv 2026),
  built for **AI Challenge HCMC (AIC) 2025**. Different competition series from VBS, but
  same problem shape (event retrieval from large-scale Vietnamese TV video). This paper
  is the most **implementation-complete** of the three: gives closed-form formulas,
  pseudocode (Algorithm 1/2), and named model choices for almost everything the two VBS
  papers left as "vision-language models" hand-waving. Use it as the primary
  implementation reference; use the VBS papers for the parts U-CESE doesn't cover
  (query expansion, visual query generation, temporal/KISC-style search).

  **Caveat**: AIC2025's task set is **KIS, VKIS (video-based KIS), VQA, TRAKE (temporal
  alignment: order several key moments of one action)** — not identical to our stated
  HCMAI25 task set (**KIS, AVS, VQA, KISC**). Only KIS and VQA are shared 1:1. TRAKE and
  KISC are both "harder" extensions but in different directions (ordering vs.
  clarification-dialogue) — don't assume U-CESE's Unified Clipping Algorithm or TRAKE
  workflow transfers to KISC without adaptation. AVS (ours, diverse multi-scene
  retrieval) has no direct analogue in U-CESE either.

  **Also worth checking**: AIC2025's organizers provided pre-extracted keyframes,
  Faster R-CNN (OpenImagesV4) object detections, and CLIP ViT-B/32 features as part of
  the dataset (§3.1) — i.e. U-CESE's object detection wasn't something the team
  built, it was given. Confirm whether HCMAI25's organizers (batch2 dataset) provide
  anything similar before assuming we have to build object detection from scratch.
- `docs/paper/2512.13169v1.pdf` — **"Integrated Semantic and Temporal Alignment for
  Interactive Video Retrieval"** (Luu, Nguyen-Dinh, Bui, Tran, Le, Nguyen-Nhu; team
  **AIO_Owlgorithms**, arXiv Dec 2025). A **third, independent** implementation for the
  *same* AIC HCMC 2025 competition as U-CESE (different team). Two named contributions:
  **QUEST** (handles queries about entities the embedding model has no knowledge of) and
  **DANTE** (dynamic-programming algorithm for ordered/temporal multi-event queries,
  built for the TRAKE task). Also runs its own full OCR pipeline in production (not just
  related-work mention like U-CESE). Being a second independent AIC2025 team, agreement
  with U-CESE is a stronger signal than either alone; disagreement (e.g. shot detection
  method) means neither is a consensus default.

Read all four together: 2025 VBS = base retrieval pipeline with real method citations,
2026 VBS = the VQA-specific layer added on top (mostly unspecified), U-CESE and this
4th paper = two independent, more concretely-specified systems for a sibling
competition (AIC2025). This plan merges all four and flags what's still genuinely open
for our own HCMAI25 dataset (see `CLAUDE.md` and `eda_output/`: mixed fps/codec, 605
videos, cross-group near-duplicates).

## NII-UIT (VBS) vs U-CESE (AIC2025): where the two teams agree and disagree

Two independent teams solving the same-shaped problem (large-scale video event
retrieval). Where they converged without coordinating is a stronger signal than either
choice alone; where they diverge is a real design decision we still have to make
ourselves, not something we can just copy.

### Where they agree (converging signal — safer to adopt)

| Aspect | Agreement |
|---|---|
| Overall architecture | Both split into a strict **offline indexing** stage and an **online query/fusion/rerank** stage — neither does real-time end-to-end processing. |
| Vector DB | Both independently chose **Milvus**. |
| Fusion style | Both fuse multi-modal signals with a **simple, rule-based** combiner rather than a learned reranker — NII-UIT: normalize + mean-pool; U-CESE: two-pointer sweep ranked by query-coverage. Same philosophy (cheap, transparent) via different mechanics. |
| Object filtering | Both treat object constraints as a **hard filter applied after** embedding retrieval, not blended into one score. |
| Captioning timing | Both generate captions/descriptions **offline, once, in advance** — never on-the-fly per query. |
| UI philosophy | Both prioritize an interactive verification UI (timeline/candidate panel; suggestion viewer) to **minimize manual browsing** for the user, not just raw ranked lists. |

### Where they disagree (real, unresolved design decisions)

| Aspect | NII-UIT (VBS) | U-CESE (AIC2025) | Why it matters |
|---|---|---|---|
| Keyframe/shot extraction | 2026: unspecified. 2025: model-based (BEiT-3 feature-diff, needs a forward pass per candidate frame) | **DAKE**: zero-model, pure JPEG-file-size heuristic | The single biggest divergence. U-CESE bets on cheap/fast (no inference cost) with an ablation showing recall close to a learned baseline (AutoShot); NII-UIT never tries the no-model route at all. |
| Dense captioning consistency | Not addressed at all — no mechanism to keep captions consistent across shots of the same video | **ReCap**: explicit recurrent-memory architecture, ablated and shown to fix identity/context drift across shots | A real gap in NII-UIT's design that U-CESE actually solves — not just an unspecified detail, but a problem NII-UIT doesn't appear to have noticed. |
| Module structure for query handling | Stays multi-module even in 2026: Answer Span Prediction, Candidate Answer Suggestion, In-Video Retrieval are three separate mechanisms for VQA alone | **Explicitly unifies** everything into one pipeline (Unified Clipping Algorithm), and frames this as a direct improvement over the multi-module style (their own predecessor, CESE, was multi-module and they call it out for "inconsistency and inefficiency... fragmented workflows") | This is a philosophy disagreement, not a missing detail — U-CESE's stated rationale for unifying is a direct critique of the architecture NII-UIT still uses. Worth weighing deliberately before defaulting to NII-UIT's multi-module shape. |
| Query expansion | Explicit module: GPT-4o generates 5 paraphrases | No equivalent module | Unclear if this is because AIC2025 queries are less ambiguous, or U-CESE simply didn't prioritize it — not stated either way. |
| Offline features: built vs. given | Builds everything itself (object detection, embeddings, etc.) | AIC2025 organizers **pre-supplied** keyframes, object detections (Faster R-CNN/OpenImagesV4), and CLIP features — U-CESE only builds on top | A competition-constraint difference, not a technical one, but it explains why U-CESE's own pipeline looks lighter — part of that is because they didn't have to build the whole stack. |

### Practical takeaway

Prioritize adopting the "agree" column outright (Milvus, rule-based fusion, offline
captioning, hard object-filter, verification-first UI). For the "disagree" column: DAKE
and ReCap directly close our two biggest open items (shot detection, caption
consistency) and should be adopted unless a concrete reason shows up not to. The
multi-module-vs-unified question is a real architectural choice for us to make
deliberately, not inherited from either paper by default.

## Pipeline overview

```
[Offline]  video → DAKE keyframe extraction (U-CESE, training-free, JPEG-size steepness)
                     ├→ embed (SigLIP 2026 / MobileCLIP U-CESE / multi-VLM ensemble 2025) → VisualDB (Milvus)
                     ├→ object detection (Co-DETR 2025, or organizer-provided per U-CESE)
                     ├→ ASR transcript (Whisper — U-CESE confirms + 2026 implies)
                     └→ dense caption/timeline: ReCap (U-CESE, Gemini 2.5, recurrent memory) → TextualDB (Milvus + Elasticsearch)
[Online]   query → query expansion (GPT-4o paraphrase x5, 2025) / visual query gen (Stable Diffusion, 2025)
                  → cross-modal search, normalize + mean-pool fusion (2025)
                     OR Unified Clipping Algorithm two-pointer sweep → ranked clip suggestions (U-CESE)
                  → object filter+rerank (2025)
                  → Answer Span Prediction (LVLM over caption timeline, model TBD) → hotspots
                  → Candidate Answer Suggestion (aggregation algorithm still unnamed, 2026)
                  → In-Video Retrieval (text/image/object/OCR/audio, fusion method TBD) → frame-level answer
                  → user verifies via UI (timeline + candidate panel)
```

## Components with a named method/citation

| Component | Method cited | Reference | Paper | Status for us |
|---|---|---|---|---|
| **Keyframe extraction / shot-boundary substitute** | **DAKE**: training-free, no learned model. Uses JPEG-compressed file size as a motion proxy — computes a scale-invariant "steepness" score S(i,j) between frames from file-size deltas (closed-form formula, §4.1), averages steepness over a local window, keeps the top-ρ fraction of frames by aggregated steepness. ρ=0.02 (2% of frames) + a Δ=2×fps matching window reproduces near-perfect recall against AutoShot (a trained shot-boundary detector) | U-CESE §4.1, Algorithm 1 | U-CESE | **Resolves our old "shot detection" open item.** Strong candidate to adopt directly — cheap (no model inference, just JPEG size deltas), tunable via ρ (keyframe-count vs. storage/recall tradeoff), explicitly validated against a learned baseline (AutoShot) in their ablation (§5.1, Fig. 6/7) |
| Keyframe selection (VBS2025 alternative) | Vibro-inspired adaptive sampling: BEiT-3 semantic features every 10th frame, keep only frames with significant feature difference; store as WebP | Hezel et al. 2022 (Vibro) [6] + Wang et al. 2023 (BEiT-3) [16] | 2025 §2.1 | Alternative to DAKE — needs a model forward pass per candidate frame (costlier than DAKE's pure file-size heuristic); worth A/B-ing against DAKE rather than assuming DAKE wins |
| **Dense per-shot captioning** | **ReCap**: recurrent captioning — video divided into shots (via AutoShot in their impl), each shot captioned by an LVLM conditioned on a running memory string M_{t-1} carried from previous shots (RNN-like recurrence: `(C_t, M_t) = LVLM(S_t, M_{t-1})`). Memory preserves entity/identity/setting continuity across shot transitions instead of re-describing generically each time. Ablation (§5.2, Fig. 8) shows this materially improves caption groundedness (e.g. correctly identifies a recurring speaker and links back to previously-mentioned details) vs. no-memory per-shot captioning | U-CESE §4.1, Fig. 3 | U-CESE | **Resolves our old "dense captioning model" open item — as an architecture.** Model used is Gemini 2.5 (closed API); our own model choice is still open (see below), but the recurrent-memory *design* is adoptable regardless of which LVLM backs it |
| Keyframe captioning (non-recurrent variant) | Per-keyframe caption from LVLM given a context window of k preceding/succeeding keyframes + subtitle + target keyframe, single-prompt (background inference + visual analysis + self-generated QA for completeness) | U-CESE §4.1 | U-CESE | Simpler fallback if full ReCap recurrence is too much to build first — still gives context-aware (not isolated) captions |
| **ASR** | Whisper | Radford et al. 2022 [21] | U-CESE §4.1 (Fig. 2) | **Resolves our old ASR open item** — corroborates what the 2026 paper implied but never named. Adopt directly. |
| Image-text embedding (U-CESE) | MobileCLIP — lightweight, fast, mobile-optimized image-text encoder, used for both vision and text encoders | Vasu et al. 2024 [23] | U-CESE | Alternative to SigLIP (2026) — worth comparing if inference throughput at 605-video/19.7M-frame scale becomes a bottleneck |
| Image-text embedding (2026) | SigLIP (sigmoid loss, signal-level regularization) | Zhai et al. 2023 [15] | 2026 | Adopt — replaces CLIP for subtle/compositional queries |
| Image-text embedding (2025, ensemble) | OpenCLIP ViT-L/14, CLIP2Video, ALADIN, BEiT-3, OpenCLIP H-14, InternVL-G — multiple VLMs run in parallel | 2025 refs | 2025 §2.2 | Reference — 2025 uses an ensemble, not a single encoder |
| Vector DB | **Milvus** | — | 2025 §2 **and** U-CESE §4.1 (independently converged on) | Two of three papers use Milvus for embedding storage/search — stronger signal to adopt if we need a vector DB rather than flat search over 605 videos' keyframes |
| Raw-text search | Elasticsearch, alongside Milvus embedding search of captions — text indexed both ways (raw + embedded) | — | U-CESE §4.1 (Fig. 2) | Adopt as the TextualDB pairing: raw keyword search (Elasticsearch) + semantic search (Milvus) side by side, not either/or |
| **Multi-modal fusion into ranked results** | **Unified Clipping Algorithm** (Algorithm 2): retrieve top-k from VisualDB (frame embeddings) + TextualDB (caption embeddings + raw text) per query, flatten and sort by (video, timestamp); for each video, greedily sweep a two-pointer window bounded by max clip length T to group nearby retrieved frames into candidate "suggestions" (clips, not single frames); rank suggestions by (a) number of unique queries covered, tie-broken by (b) max per-frame similarity score. Linear time in number of retrieved items. | U-CESE §4.2, Algorithm 2 | U-CESE | **Directly answers part of our old "In-Video Retrieval fusion method" open item** — gives a concrete, cheap (linear-time, no learned reranker) way to merge multi-modal frame-level retrieval into ranked clip-level suggestions. Doesn't cover OCR/audio fusion specifically (U-CESE has no OCR stage), but the frame→clip grouping + query-coverage ranking logic is reusable as-is |
| Object detection | Co-DETR — optimized for precise detection + counting, identifies all COCO classes per keyframe | Zong, Song, Liu 2023 [17] | 2025 §2.5 | Adopt if we must build our own detector — but check first whether HCMAI25 organizers provide pre-extracted object detections (U-CESE's AIC2025 dataset did, via organizer-provided Faster R-CNN/OpenImagesV4) |
| Object filtering | Hard filter: after score fusion, drop shots missing query-specified objects, then rerank remainder | — | 2025 §2.6 | Adopt as the object-constraint mechanism |
| **OCR** | **Two concrete options, both from AIC2025 systems**: (a) Gemini used directly as the OCR extractor in a production pipeline — each keyframe processed by Gemini to pull on-screen text as `{keyframe_id: ocr_text}`, indexed in Elasticsearch with the **Vietnamese Analysis Plugin** for correct tokenization/search of Vietnamese on-screen text; (b) PARSeq fine-tuned specifically for Vietnamese text recognition, used by a different AIC team (cited in U-CESE's related work, not run by U-CESE itself) | (a) 4th paper §3.2/§4.1, using Gemini [3]; (b) Bautista & Atienza 2022, via U-CESE §2 citing a different team's system | 4th paper (adopted, in production) / U-CESE (related-work mention only) | **Resolves our old OCR open item — now has a real in-production reference**, not just a related-work mention. (a) is the stronger candidate since it's actually deployed and pairs with a Vietnamese-aware search backend, directly relevant since HCMAI25 is also Vietnamese-context. (b) is a dedicated OCR model if Gemini-as-OCR proves too slow/costly at 605-video scale. |
| **Shot/keyframe extraction — three-way disagreement, no consensus** | A third method: **TransNetV2** (trained shot-boundary detector) segments videos into scenes; then a **fixed formula** samples exactly 4 keyframes per scene at proportional positions (start, 1/3, 2/3, end): `k = {K_{a+⌊i×(b-a)/3⌋} : i∈{0,1,2,3}}` — deterministic count per scene, not adaptive to scene length/motion | 4th paper §3.1, citing Souček & Lokoč 2020 (TransNetV2) | 4th paper | Now **3 genuinely different methods** are on the table with no majority: DAKE (U-CESE, training-free, JPEG-size heuristic), Vibro+BEiT-3-diff (VBS2025, trained-model feature-diff), TransNetV2+fixed-4 (this paper, trained shot-boundary model + deterministic sampling). None dominates by consensus — **this is a real open decision requiring our own small-scale benchmark**, not something to inherit by majority vote. |
| Query expansion (paraphrase) | GPT-4o generates 5 alternative paraphrasings; user picks one, or all 5 searched in parallel and fused | Ma, Wu, Ngo 2024 | 2025 §2.3 | Adopt — concrete parameter (n=5) |
| Visual query generation (text→image) | Stable Diffusion generates image(s) from text query as an additional visual query | Rombach et al. 2021/2022 | 2025 §2.4 | Candidate for AVS/KIS when query has no strong visual anchor — **complementary to QUEST below, not redundant**: SD synthesizes from the encoder's own (also frozen) knowledge, so it fails on the exact cases QUEST Branch 2 targets (see note below) |
| **QUEST — Out-of-Knowledge (OOK) query handling** (ADOPTED) | Two-branch framework for queries about entities/concepts the embedding model has no knowledge of (frozen training cutoff). **Branch 1 (Query Rewriting)**: LLM (GPT-4/Gemini) rewrites query q0 into a more descriptive, visually-grounded qr, run through standard retrieval. **Branch 2 (External Image Retrieval)**, triggered if Branch 1 fails/is insufficient: fetch a real representative image of the entity from an external source (e.g. Google Images), encode it, and do image-to-image search instead of text-to-image | 4th paper §4.2, Fig. 3 (Branch 1 inspired by Bui-Tran et al. 2025 [1]) | 4th paper | **New capability, not present in NII-UIT or U-CESE.** Worth adopting if HCMAI25 queries can reference specific/rare/recent named entities our embedding model wouldn't know. See "QUEST vs. Stable Diffusion" note below for why these two don't overlap despite both doing "generate/fetch an image from a text query." |
| Dynamic/temporal search — **three now-documented approaches, meaningfully different** | See dedicated comparison section below ("DANTE vs. real Vitrivr algorithm") — this is not a simple pick, the three approaches (real Vitrivr, NII-UIT's own heuristic, DANTE) behave differently on concrete cases | 2025 §2.6 (citing Vitrivr) / 4th paper §4.3 | 2025, 4th paper | **Directly relevant to KISC** — see comparison section for which to adopt and why |
| 2024-style Q/A handling (predecessor to 2026's Answer Span Prediction) | Convert question into a plain textual description (manually or via LLM), then run it through the same KIS-T temporal-search pipeline — no dedicated QA module | — | 2025 §2.7 | Shows the evolution: 2025 treated QA as "reformulate then retrieve"; 2026 replaced this with dense-caption-timeline + LVLM matching because plain reformulation didn't localize short-lived evidence well enough — useful as a cheap fallback/baseline |
| Answer Span Prediction LVLM (2026) | NVILA | Liu et al. 2024 | 2026 | **Model choice: TBD.** Options: NVILA (as-cited), Qwen3-VL (reuse SSDC serving infra), or Gemini 2.5 (U-CESE's choice for captioning — closed API, but proven for this exact "reason over context + generate" pattern). Not decided yet. |

### QUEST vs. Stable Diffusion visual query generation — why they don't overlap

Both "generate/fetch an image from a text description," so it looks redundant at first
glance. It isn't:

- **Stable Diffusion (2025 baseline)** *synthesizes* a new image from the diffusion
  model's own learned knowledge. SD is itself a frozen model with its own training
  cutoff — if the query names an entity SD never saw during training (the exact
  definition of an OOK query), SD will hallucinate a generic or wrong image. It doesn't
  actually solve the frozen-knowledge problem, it just moves it to a different model.
- **QUEST Branch 2** *retrieves* a real, existing image of the specific entity from an
  external source (e.g. Google Images) at query time. Because it pulls from a live,
  continuously-updated source instead of a static trained model, it works precisely in
  the case SD fails: novel/rare/recent named entities.

So SD is the right tool for queries describing a *generic, plausible* scene ("a red car
in the rain") that a diffusion model can reasonably imagine even without having seen
that exact instance. QUEST Branch 2 is the right tool when the query names a *specific*
entity the encoder/generator genuinely has no knowledge of. They're complementary
fallbacks for different failure modes, not competing options for the same problem.
(This framing is our own inference from how the two mechanisms work, not a comparison
either paper makes explicitly — worth a small validation check against real HCMAI25
queries before leaning on it.)

### DANTE vs. the real Vitrivr temporal-query algorithm — concrete behavioral differences

Earlier framing of NII-UIT's "dynamic temporal search" as "Vitrivr-inspired" needs a
correction: the paper NII-UIT actually cites for this (Gasser et al. 2024, "A new
retrieval engine for vitrivr") is a **software-architecture overview** with no temporal
scoring algorithm at all. The real Vitrivr temporal-query algorithm lives in a
*different* paper — Heller et al., MMM 2022, "Multi-modal Interactive Video Retrieval
with Temporal Queries" (fetched and read directly, not in our `docs/paper/` set since
it's cited transitively, not uploaded) — which NII-UIT's own reference list doesn't
cite for this mechanism. Having read the real algorithm, three concrete differences
from DANTE emerge:

**1. What "order" actually constrains**

- **Real Vitrivr**: each query stage (`qc1, qc2, ...`) is retrieved *independently* into
  its own top-k candidate list. The ordering constraint is on **which query stage** a
  segment came from (`j > i`), not directly on timestamps — a segment from a later
  query stage is allowed to have an *earlier* timestamp than one from an earlier stage;
  it's the query-container index that must increase, not necessarily the clock.
- **DANTE**: the DP recursion (`DP[i,t]` built from `DP[i-1,τ]` for `τ < t` only, via the
  running-max in Algorithm 1) enforces increasing **keyframe index** directly — event
  `i`'s assigned frame must literally come after whatever frame was used for event
  `i-1`. This is a hard constraint on time itself, not just on which sub-query it came
  from.

**Worked example** — query: "(1) person picks up a knife, (2) person chops vegetables."
Suppose in the target video, the *true* knife-pickup moment is at t=100s (a so-so visual
match, score 0.70) and the true chopping moment is at t=120s (a great match, score
0.95). But suppose a *different*, unrelated shot at t=200s (person holding a knife while
talking, nothing to do with cooking) scores slightly higher on stage 1's embedding
(0.75) purely by visual coincidence.

- Real Vitrivr: stage 1's top candidate becomes t=200 (0.75 > 0.70). Stage 2's top
  candidate is t=120 (0.95). Sequence `⟨t=200 (qc1), t=120 (qc2)⟩` is still *allowed*
  by the `j>i` container-order rule (qc2 came after qc1 in the query) even though
  t=120 is *earlier in time* than t=200 — the algorithm doesn't forbid this, it only
  soft-penalizes based on `|actual_gap − expected_gap m|`. If the user's specified `m`
  happens to tolerate a large or negative-looking gap, a temporally-backwards, causally
  nonsensical sequence can still win.
- DANTE: because `t` in `DP[i,t]` is a literal frame index and the recursion only
  builds forward, event 2 (chopping) can never be assigned a frame *before* whatever
  frame event 1 (knife pickup) used. The nonsensical t=200-then-t=120 ordering is
  structurally impossible, not just penalized.

**2. Who has to specify the expected time gap**

- Real Vitrivr: the target gap `m` ("time to next segment") is a value the **end user
  types into the query UI** per query (see Fig. 2 of Heller et al.: a literal numeric
  field for "additional temporal distance between the first and second query
  container"). If the user guesses wrong under time pressure — expects clues ~10s apart
  but the true gap is 90s — the correct sequence gets penalized for not matching the
  user's guess, not for actually being wrong.
- DANTE: `λ` is a **system-level constant**, tuned once by the developers (their own
  reported sweet spots: λ=0.001 for keyframe-index gaps of 3–15, λ=0.01 for gaps of
  1–3) and applied automatically to every query. The end user never has to guess a time
  gap. Under HCMAI25/AIC-style strict per-query time limits, not having to correctly
  estimate a gap before searching is a real practical advantage.

**3. Search space per stage**

- Real Vitrivr: each stage's candidates come from a **pre-truncated top-k list**,
  computed independently per stage across (implicitly) the whole gallery. If the true
  matching frame for stage 1 in the *target* video doesn't make that global top-k cut
  (because a handful of frames from *other, unrelated* videos score marginally higher
  on raw similarity), it never enters the sequence-building step at all — the correct
  video can be missed outright, even if its frames for stages 2 and 3 score very well.
- DANTE: computes the similarity matrix `S[i,t]` over **every keyframe belonging to
  each candidate video** (via each video's `[s_v, e_v]` range in the metadata), not a
  pre-truncated global top-k. A weak-but-correct stage-1 match inside the right video
  is never invisible to the algorithm just because unrelated videos scored higher on
  that one sub-query — it's still available for the DP to combine with strong stage-2/3
  matches in the same video.

**Worked example (this one drawn directly from the DANTE paper's own Fig. 9a case)** —
query: "(1) wide shot of a green rice field, (2) close-up of golden rice grains, (3) a
hand touching rice." Suppose the target video's field shot (stage 1) is slightly hazy
and only ranks ~6th globally on that sub-query — outside a top-5 cutoff — while several
unrelated videos' field shots rank higher purely by lighting/color. Stages 2 and 3,
however, are excellent matches (rank 1 and 2) for the target video.

- Real Vitrivr (top-k-per-stage): the target video's stage-1 frame never enters the
  candidate pool; no full 3-stage sequence can be built for it, and it drops out of
  contention even though 2 of 3 clues match almost perfectly.
- DANTE (full-video DP): stage 1's weaker-but-still-real match inside the target video
  is available regardless of global rank, and the DP can still assemble a strong
  3-event sequence for that video. This matches what the DANTE paper itself reports
  qualitatively (Semantic Search alone "scattered" the frames; DANTE recovered the
  correct video at Top-1, Fig. 9a) — consistent with this being a real, not just
  theoretical, advantage of the full-keyframe search space.

**Bottom line for KISC**: DANTE's hard temporal ordering + full-video search space is
structurally stronger for exactly the failure modes that matter in a multi-clue,
time-pressured task like KISC. Real Vitrivr's exponential-decay *scoring* formula
(`e^{-|l(t-m)|}`) is still a reasonable, published, citable building block — but its
`m`-per-query requirement and top-k-per-stage search space are real weaknesses next to
DANTE. NII-UIT's own bidirectional heuristic remains the least specified of the three
(no formula anywhere), so it's the weakest starting point of the three despite being
the one most directly tied to our KISC-shaped task by NII-UIT's own framing.

### Answer Span Prediction is two distinct sub-stages, not one model call

The 2026 paper's §3.2 wording ("LVLM... aligns its semantic representation with the
caption timeline") conflates two different tasks that should be built and reasoned about
separately — and U-CESE's ReCap confirms this split is real in practice, since ReCap
*only* covers the captioning half:

1. **Dense captioning — offline, once per video, query-independent.** Video → text per
   shot/temporal unit. A generation task; runs once over all 605 videos regardless of
   what anyone later asks. **Architecture now resolved** (ReCap's recurrent-memory
   design, above) — model choice still TBD.
2. **Answer-span matching — online, once per user question.** Question + existing
   caption timeline → ranked/filtered hotspots. A matching/reranking task — doesn't
   touch new video content, just scores captions already on disk against the incoming
   query. Neither U-CESE nor the VBS papers give a concrete matching algorithm beyond
   "LVLM aligns semantic representation" — still open. The Unified Clipping Algorithm's
   query-coverage ranking (above) is a plausible non-LVLM building block for this step.

Model choice for both roles is **TBD** — the two calls should stay implemented as
separate stages regardless of which model(s) end up chosen.

## Still genuinely open (no paper specifies)

1. **"Short temporal unit" duration/boundary for Answer Span Prediction (2026)** — no
   duration or criterion given, and unclear whether this reuses the shot/keyframe
   segmentation (DAKE or ReCap's AutoShot shots) or defines a new one. **Open** —
   reasonable default: reuse whatever shot segmentation we pick for captioning, don't
   invent a third segmentation scheme.
2. **Answer-span matching algorithm** — see above; U-CESE doesn't have an equivalent
   module at all (no VQA-localization task in AIC2025), so this remains VBS2026-only
   and unspecified. **Open.**
3. ~~OCR~~ **Resolved** — see table above (Gemini+Elasticsearch-Vietnamese-plugin, or
   PARSeq). Still worth checking if OCR is needed at all for AVS/KISC specifically
   (on-screen text matters less there than for TRAKE-style ingredient/label tracking),
   but the "how" is no longer open, only the "whether."
4. **Candidate Answer Suggestion aggregation logic (2026)** — how caption+transcript
   cues get merged/ranked into n candidate answers, and what n is. **Open** — no
   analogue in U-CESE either (AIC2025 has no multi-candidate-answer UI pattern).
5. **In-Video Retrieval's OCR/audio-specific fusion (2026)** — Unified Clipping
   (resolved above) covers embedding+text fusion into ranked clips, but the *additional*
   OCR/audio signal fusion described for 2026's In-Video Retrieval specifically is still
   unspecified. **Open**, but now has a concrete non-LLM baseline (Unified Clipping) to
   extend rather than building fusion from nothing.
6. **Caching mechanism for repeated/paraphrased queries (2026)** — mentioned as a
   system optimization, no implementation detail anywhere. **Open**, low priority.
7. **Whether HCMAI25 organizers provide pre-extracted features** (keyframes, object
   detections, CLIP features) the way AIC2025's organizers did for U-CESE — **unchecked
   fact, not a design question**, but blocks knowing how much of the offline pipeline we
   actually need to build ourselves. Check the batch2 dataset directory for anything
   beyond raw video before assuming we start from zero.

## How this maps onto our dataset

- DAKE (resolved keyframe/shot method) is file-size-based and therefore **codec-
  sensitive** — av1 vs h264 (documented in `CLAUDE.md`) compress differently, so the
  steepness formula's scale-invariance (normalized by `s_max` per video) should hold,
  but ρ and the matching window Δ may need separate tuning per codec group rather than
  one global value. Cheap to test empirically since DAKE requires no model inference.
- ReCap's recurrent memory is a per-video sequential process (shot t depends on shot
  t-1's memory) — doesn't parallelize across shots within one video, only across videos.
  At 605 videos this is fine, but worth knowing before assuming naive batch parallelism.
- OCR/ASR must run at 605-video × ~19min-avg scale — cost/throughput is a first-order
  constraint regardless of which tool is chosen.
- Cross-group near-duplicate content (`eda_output/theme_analysis/`) means the Unified
  Clipping Algorithm's per-video grouping (it sweeps within one video's retrieved
  frames) should be safe from cross-video duplicate leakage by construction — but the
  *upstream* embedding retrieval that feeds it could still surface the wrong video's
  near-duplicate scene before clipping ever happens.
- KISC maps closely to temporal/sequential multi-cue search — see the dedicated "DANTE
  vs. real Vitrivr" comparison above for which mechanism to adopt (DANTE's hard-ordered
  DP is the stronger candidate) rather than defaulting to NII-UIT's own unformalized
  bidirectional heuristic. TRAKE (U-CESE's hardest task) is temporal *ordering* of
  multiple sub-events within one action, closer to our AVS (finding multiple matching
  scenes) than to KISC, if we ever need ordering-style features.
- QUEST is worth a cheap validation check: sample a handful of real HCMAI25 queries and
  see if any reference specific/rare/recent named entities before investing in building
  it — if our queries stay generic/descriptive, QUEST's marginal value over the existing
  Stable Diffusion visual-query-generation path may be low.

## Next steps

Priority order, since later items depend on earlier ones:
1. **Check what HCMAI25 organizers already provide** (open item #7) — could eliminate
   entire offline-pipeline steps if features are pre-extracted.
2. Keyframe/shot extraction — adopt **DAKE** (training-free, cheapest to stand up, has a
   validated ablation against a learned baseline); re-tune ρ/Δ per our fps/codec mix.
3. Object detection — adopt Co-DETR *if* organizers don't already provide detections.
4. Embedding + fusion — adopt SigLIP or MobileCLIP (A/B for throughput/quality) +
   mean-pool fusion (2025) or Unified Clipping (U-CESE) as first baselines.
5. Dense captioning — adopt ReCap's recurrent-memory architecture; pick a backing LVLM
   (still TBD — NVILA / Qwen3-VL / Gemini 2.5 / other).
6. ASR — adopt Whisper directly (resolved, no more decision needed).
7. OCR — decide whether it's needed at all for AVS/KISC before picking a tool (method
   itself is resolved: Gemini+Elasticsearch-Vietnamese-plugin, or PARSeq).
8. Temporal/sequential search for KISC — adopt **DANTE's DP formulation** (see dedicated
   comparison section) rather than NII-UIT's unformalized heuristic.
9. QUEST — validate need against real HCMAI25 queries before building (see above); adopt
   only Branch 1 (LLM rewrite) first if needed, Branch 2 (external image search) is more
   engineering for a narrower failure mode.
10. Answer-span matching + Candidate Answer Suggestion + In-Video Retrieval's remaining
    fusion gaps — design once 1-9 exist; these are VBS2026-only concepts with no
    ready-made implementation in any source paper.

For each open item: survey 2-3 candidate methods/tools, small-scale test on a handful of
our videos, decide, document the choice (and why) back into this plan.
