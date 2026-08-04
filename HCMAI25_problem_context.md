# HCMAI25 PROBLEM CONTEXT

## Dataset

```text
/mlcv2025/Datasets/HCMAI25/batch2/video
```

The dataset contains the video collection used for all four tasks:

- KIS
- AVS
- VQA
- KISC

All tasks use the same video source and may share keyframes, timestamps, metadata, embeddings, OCR text, captions, and object information.

---

## 1. KIS — Known-Item Search

### Problem

Given a natural-language description of a specific scene that the user remembers, retrieve the exact video moment that matches the description.

The query may include:

- People or objects
- Actions
- Locations or scenes
- Colors and attributes
- Visible text
- Events occurring before or after the target scene

### Input

```json
{
  "query_id": "kis_001",
  "query": "A man in a blue shirt stands beside a motorcycle before entering a shop."
}
```

### Output

```json
{
  "query_id": "kis_001",
  "results": [
    {
      "video_id": "video_001",
      "frame_id": "frame_001250",
      "timestamp": 52.08,
      "score": 0.94
    }
  ]
}
```

### Objective

Return a small ranked list containing the exact target scene or the closest matching video moments.

---

## 2. AVS — Ad-hoc Video Search

### Problem

Given a general natural-language query, retrieve all video scenes that satisfy the described concept, action, object, or context.

Unlike KIS, an AVS query may have many correct results across different videos.

### Input

```json
{
  "query_id": "avs_001",
  "query": "People using mobile phones on public transportation."
}
```

### Output

```json
{
  "query_id": "avs_001",
  "results": [
    {
      "video_id": "video_005",
      "frame_id": "frame_000820",
      "timestamp": 34.16,
      "score": 0.91
    },
    {
      "video_id": "video_042",
      "frame_id": "frame_003210",
      "timestamp": 133.75,
      "score": 0.87
    }
  ]
}
```

### Objective

Return a diverse ranked list of relevant scenes while limiting duplicate or near-identical results from the same shot.

---

## 3. VQA — Video Question Answering

### Problem

Given a question about a video or a video segment, produce an answer grounded in the visual and temporal content of the video.

The question may concern:

- Objects
- Actions
- Counts
- Colors
- Locations
- Visible text
- Event order
- Events occurring before or after another event

### Input

```json
{
  "question_id": "vqa_001",
  "video_id": "video_012",
  "question": "What does the man do after leaving the shop?"
}
```

### Output

```json
{
  "question_id": "vqa_001",
  "answer": "He gets on a motorcycle and drives away.",
  "evidence": [
    {
      "video_id": "video_012",
      "start_time": 120.5,
      "end_time": 128.2
    }
  ]
}
```

### Objective

Generate a concise answer supported by evidence from the relevant video frames or segment.

---

## 4. KISC — Known-Item Search with Clarification

### Problem

Given an incomplete description of a specific target scene, interact with the user by asking clarification questions and use the answers to narrow the candidate results.

### Initial Input

```json
{
  "session_id": "kisc_001",
  "query": "A person standing beside a vehicle."
}
```

### Clarification Output

```json
{
  "session_id": "kisc_001",
  "action": "ask",
  "question": "Is the vehicle a motorcycle or a car?"
}
```

### User Response

```json
{
  "session_id": "kisc_001",
  "answer": "A motorcycle."
}
```

### Final Output

```json
{
  "session_id": "kisc_001",
  "action": "submit",
  "results": [
    {
      "video_id": "video_031",
      "frame_id": "frame_003420",
      "timestamp": 142.5,
      "score": 0.95
    }
  ]
}
```

### Objective

Find the exact target scene using as few clarification turns as possible.

---

## Shared Result Format

```json
{
  "video_id": "video_001",
  "shot_id": "shot_010",
  "frame_id": "frame_001250",
  "timestamp": 52.08,
  "frame_path": "/path/to/frame.jpg",
  "score": 0.91
}
```

## Task Differences

| Task | Main Input | Main Output | Target |
|---|---|---|---|
| KIS | Specific scene description | Ranked frames or timestamps | One exact scene |
| AVS | General semantic query | Diverse ranked scenes | Multiple relevant scenes |
| VQA | Question and video context | Text answer with evidence | Correct grounded answer |
| KISC | Incomplete scene description | Clarification questions and final result | One exact scene through interaction |
