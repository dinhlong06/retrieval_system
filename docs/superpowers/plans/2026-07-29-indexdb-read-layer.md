# indexdb Read Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the algorithm team (KIS/AVS/VQA) a single Python entry point, `indexdb.read.Reader`, to query the Mongo/Milvus/Elastic stack that `indexdb.ingest` already populates — without them touching store internals, encoding text themselves, or guessing path/URI conventions.

**Architecture:** One new module, `indexdb/read.py`, containing a `Reader` class with 5 methods: `search_vector`, `search_text` (candidate generation — return a normalized `Hit` list), and `get_keyframes`, `get_shots`, `get_transcript` (hydration — return full documents). `Reader` does not encode text and does not rank/fuse across sources; that stays the algorithm team's job. Consumers reach `Reader` by mounting `layer_5/indexdb` read-only into their own container (`-v .../layer_5:/opt/layer5:ro -e PYTHONPATH=/opt/layer5`) and joining the `milvus` Docker network — no `pip install` of this repo, per the "shared server, no ad-hoc pip installs" constraint.

**Tech Stack:** Python 3.14, pymongo 4.9.2, elasticsearch-py 9.0.1, pymilvus 2.6.1 (all already in `layer_5/requirements.txt` — no new dependencies).

## Global Constraints

- `read.py` must not `import numpy` — consumer containers may pin `numpy<2` for unrelated reasons (Paddle/layer_3), and this module must not force a version.
- `read.py` must not encode query text into vectors — callers pass an already-encoded vector (`list[float]`) to `search_vector`.
- No new pip dependency — reuse `pymongo`/`elasticsearch`/`pymilvus` clients already wrapped by `MongoStore`/`ElasticStore`/`MilvusStore`.
- Every `Hit` dict (from both `search_vector` and `search_text`) has exactly these 5 keys: `keyframe_id`, `video_id`, `shot_id`, `timestamp_ms` (int, **milliseconds**, unconverted — matches the unit stored in Mongo/Milvus/ES so callers don't need to guess), `score` (float). `frame_idx` is intentionally excluded — Elasticsearch's `_source` never stores it (see Task 2), and it is redundant with `keyframe_id`, recoverable via `get_keyframes` if a caller truly needs it.
- `get_keyframes` returns results **in the same order and multiplicity as the input id list** (duplicates preserved), and `image_path` in its output is an **absolute path** (`os.path.join(data_root, relative_path)`), never the relative string stored in Mongo.
- Tests follow the existing pattern: `@pytest.mark.integration`, `_test`-suffixed collections/index (`suffix="_test"` for Milvus, `index_name="keyframe_text_test"` for Elastic, `db_name="videoindex_test"` for Mongo via the `clean_mongo` fixture in `tests/conftest.py`), run via `./run.sh test`.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `indexdb/config.py` | modify | add `data_root: str = "/data"` field |
| `indexdb/read.py` | **new** | `Reader` class — all 5 query methods |
| `tests/test_read.py` | **new** | integration tests for `Reader`, `_test`-suffixed stores |
| `tests/test_config.py` | modify | 2 new tests for `data_root` env behavior |
| `run.sh` | modify | `DATA_ROOT` env var in `DB_ENV`; preload `Reader` in `shell` |
| `README.Docker.md` | modify | "Cho team thuật toán" section: `docker run` snippet + example usage |

---

## Task 1: `Config.data_root` + `Reader` skeleton with boundary checks

**Files:**
- Modify: `indexdb/config.py`
- Create: `indexdb/read.py`
- Create: `tests/test_read.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Consumes: `indexdb.config.Config`, `indexdb.mongo.MongoStore`, `indexdb.elastic.ElasticStore`, `indexdb.milvus.MilvusStore` (all existing, unchanged).
- Produces: `Config.data_root: str`; `Reader.__init__(self, cfg: Config | None = None)` — raises `AssertionError` if the Elasticsearch index or either Milvus collection is missing. Later tasks add methods to this same `Reader` class.

- [ ] **Step 1: Write the failing test for `Config.data_root`**

In `tests/test_config.py`, append:

```python
def test_config_data_root_default(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    cfg = Config.from_env()
    assert cfg.data_root == "/data"

def test_config_data_root_reads_env(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/mnt/ds")
    cfg = Config.from_env()
    assert cfg.data_root == "/mnt/ds"
```

- [ ] **Step 2: Run to verify it fails**

Run: `./run.sh test` (or, if editing outside the container isn't possible, note the expected failure — the stack doesn't need to be up for this test since it doesn't touch network)
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'data_root'` or `TypeError` on `Config.from_env()` missing the field.

- [ ] **Step 3: Implement `data_root` in `Config`**

In `indexdb/config.py`, change:

```python
@dataclass(frozen=True)
class Config:
    mongo_uri: str
    elastic_uri: str
    milvus_uri: str
    mongo_db: str = "videoindex"
```

to:

```python
@dataclass(frozen=True)
class Config:
    mongo_uri: str
    elastic_uri: str
    milvus_uri: str
    mongo_db: str = "videoindex"
    data_root: str = "/data"
```

and in `from_env`, change:

```python
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            mongo_uri=os.getenv("MONGO_URI", "mongodb://root:rootpass@localhost:27017"),
            elastic_uri=os.getenv("ELASTICSEARCH_URI", "http://localhost:9200"),
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        )
```

to:

```python
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            mongo_uri=os.getenv("MONGO_URI", "mongodb://root:rootpass@localhost:27017"),
            elastic_uri=os.getenv("ELASTICSEARCH_URI", "http://localhost:9200"),
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            data_root=os.getenv("DATA_ROOT", "/data"),
        )
```

`data_root` must come **after** `mongo_db` (both have defaults) — a dataclass field without a default cannot follow one that has a default, so it cannot go before `mongo_db` without also removing `mongo_db`'s default.

- [ ] **Step 4: Run to verify Config tests pass**

Run: `./run.sh test`
Expected: `tests/test_config.py::test_config_data_root_default PASSED`, `tests/test_config.py::test_config_data_root_reads_env PASSED`

- [ ] **Step 5: Write the failing tests for `Reader._check_ready` boundary checks**

`Reader.__init__` always points at the production index/collection names via `Config`, which makes it awkward for tests to point it at `_test`-suffixed stores. So the boundary check lives in a static method, `_check_ready(es, mv)`, that `__init__` calls with `self.es`/`self.mv` — tests call `_check_ready` directly against `_test`-suffixed stores instead of instantiating a full `Reader`.

Create `tests/test_read.py`:

```python
import pytest
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore
from indexdb.read import Reader


@pytest.mark.integration
def test_check_ready_passes_when_stores_ready(cfg):
    es = ElasticStore(cfg, index_name="keyframe_text_test")
    mv = MilvusStore(cfg, suffix="_test")
    es.drop_index(); es.create_index()
    mv.drop_all(); mv.create_collections()
    try:
        Reader._check_ready(es, mv)  # must not raise
    finally:
        es.drop_index(); mv.drop_all()


@pytest.mark.integration
def test_check_ready_raises_when_elastic_index_missing(cfg):
    es = ElasticStore(cfg, index_name="keyframe_text_test")
    es.drop_index()
    mv = MilvusStore(cfg, suffix="_test")
    mv.drop_all(); mv.create_collections()
    try:
        with pytest.raises(AssertionError, match="keyframe_text_test"):
            Reader._check_ready(es, mv)
    finally:
        mv.drop_all()


@pytest.mark.integration
def test_check_ready_raises_when_milvus_collection_missing(cfg):
    es = ElasticStore(cfg, index_name="keyframe_text_test")
    es.drop_index(); es.create_index()
    mv = MilvusStore(cfg, suffix="_test")
    mv.drop_all()
    try:
        with pytest.raises(AssertionError):
            Reader._check_ready(es, mv)
    finally:
        es.drop_index()
```

- [ ] **Step 6: Run to verify it fails**

Run: `./run.sh test`
Expected: FAIL with `ModuleNotFoundError: No module named 'indexdb.read'`

- [ ] **Step 7: Implement `Reader.__init__` and `_check_ready`**

Create `indexdb/read.py`:

```python
import os
from indexdb.config import Config
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore
from indexdb.mongo import MongoStore


class Reader:
    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config.from_env()
        self.mongo = MongoStore(cfg)
        self.es = ElasticStore(cfg)
        self.mv = MilvusStore(cfg)
        self.data_root = cfg.data_root
        self._check_ready(self.es, self.mv)

    @staticmethod
    def _check_ready(es: ElasticStore, mv: MilvusStore):
        assert es.client.indices.exists(index=es.index_name), (
            f"thiếu index '{es.index_name}' — kiểm tra ELASTICSEARCH_URI, "
            f"hoặc chạy indexdb.init_stores nếu chưa init"
        )
        for name in mv.collection_names().values():
            assert mv.client.has_collection(name), (
                f"thiếu collection '{name}' — kiểm tra MILVUS_URI, "
                f"hoặc chạy indexdb.init_stores nếu chưa init"
            )
```

- [ ] **Step 8: Run to verify tests pass**

Run: `./run.sh test`
Expected: `tests/test_read.py::test_check_ready_passes_when_stores_ready PASSED`, `tests/test_read.py::test_check_ready_raises_when_elastic_index_missing PASSED`, `tests/test_read.py::test_check_ready_raises_when_milvus_collection_missing PASSED`

- [ ] **Step 9: Commit**

```bash
git add indexdb/config.py indexdb/read.py tests/test_config.py tests/test_read.py
git commit -m "feat: add Config.data_root and Reader boundary checks"
```

---

## Task 2: `search_vector` + `search_text`

**Files:**
- Modify: `indexdb/read.py`
- Modify: `tests/test_read.py`

**Interfaces:**
- Consumes: `Reader.mv: MilvusStore`, `Reader.es: ElasticStore` from Task 1. `MilvusStore.collection_names() -> dict[str, str]` ([indexdb/milvus.py:11-12](../../../layer_5/indexdb/milvus.py#L11-L12)). `ElasticStore.index_name: str`, `ElasticStore.client: Elasticsearch`.
- Produces: `Reader.search_vector(self, model: str, vector: list[float], top_k: int = 100, video_ids: list[str] | None = None) -> list[dict]` and `Reader.search_text(self, query: str, top_k: int = 100, video_ids: list[str] | None = None, fields: list[str] | None = None) -> list[dict]`. Both return a list of `Hit` dicts: `{"keyframe_id": str, "video_id": str, "shot_id": str, "timestamp_ms": int, "score": float}` (see Global Constraints for why `frame_idx` is excluded).

- [ ] **Step 1: Write the failing test for `search_vector`**

Append to `tests/test_read.py`:

```python
import numpy as np


@pytest.fixture
def reader_with_data(cfg, clean_mongo, tmp_path):
    from indexdb.writers import MongoWriter
    from indexdb.milvus_indexer import MilvusIndexer
    import json

    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000007", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0}, transcript="lời thoại")
    kfid = "v001_f0001"
    w.upsert_keyframe({"video_id": "v001", "keyframe_id": kfid, "shot_id": "v001_000007",
                       "frame_idx": 1501, "timestamp_ms": 60040,
                       "image_path": "layer_2/x/v001_f0001.jpg"})
    w.enrich_keyframe(kfid, objects={"person": 1}, ocr_text="chợ hoa tết", caption="một khu chợ")

    mv = MilvusStore(cfg, suffix="_test")
    mv.drop_all(); mv.create_collections()
    vec = np.random.rand(1024).astype(np.float32)
    vec /= np.linalg.norm(vec)
    np.save(tmp_path / "v001.npy", vec.reshape(1, -1))
    (tmp_path / "v001_ids.json").write_text(json.dumps([kfid]))
    MilvusIndexer(clean_mongo, mv).index_video("v001", "beit3", str(tmp_path))

    es = ElasticStore(cfg, index_name="keyframe_text_test")
    es.drop_index(); es.create_index()
    from indexdb.elastic_indexer import ElasticIndexer
    ElasticIndexer(clean_mongo, es).index_video("v001")
    es.client.indices.refresh(index="keyframe_text_test")

    yield clean_mongo, mv, es, kfid, vec

    mv.drop_all(); es.drop_index()


@pytest.mark.integration
def test_search_vector_returns_hit(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)  # bypass __init__'s production-name check; point at _test stores directly
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits = r.search_vector("beit3", vec.tolist(), top_k=5)

    assert len(hits) == 1
    hit = hits[0]
    assert hit["keyframe_id"] == kfid
    assert hit["video_id"] == "v001"
    assert hit["shot_id"] == "v001_000007"
    assert hit["timestamp_ms"] == 60040
    assert set(hit.keys()) == {"keyframe_id", "video_id", "shot_id", "timestamp_ms", "score"}


@pytest.mark.integration
def test_search_vector_empty_for_unseen_video(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits = r.search_vector("beit3", vec.tolist(), top_k=5, video_ids=["v999_never_ingested"])

    assert hits == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `./run.sh test`
Expected: FAIL with `AttributeError: 'Reader' object has no attribute 'search_vector'`

- [ ] **Step 3: Implement `search_vector`**

In `indexdb/read.py`, add to `Reader`:

```python
    def search_vector(self, model: str, vector: list[float], top_k: int = 100,
                       video_ids: list[str] | None = None) -> list[dict]:
        name = self.mv.collection_names()[model]
        flt = f'video_id in {video_ids}' if video_ids else None
        results = self.mv.client.search(
            collection_name=name, data=[vector], limit=top_k, filter=flt,
            output_fields=["video_id", "shot_id", "timestamp_ms"],
        )
        return [
            {
                "keyframe_id": hit["id"],
                "video_id": hit["entity"]["video_id"],
                "shot_id": hit["entity"]["shot_id"],
                "timestamp_ms": hit["entity"]["timestamp_ms"],
                "score": hit["distance"],
            }
            for hit in results[0]
        ]
```

`results` is `list[list[dict]]` — one inner list per query vector; since `data=[vector]` sends exactly one vector, only `results[0]` is used. `video_ids` is rendered via Python's `list.__repr__` (e.g. `['v001', 'v002']`) which is valid Milvus filter-expression list syntax for a `VARCHAR in [...]` clause.

- [ ] **Step 4: Run to verify `search_vector` tests pass**

Run: `./run.sh test`
Expected: `tests/test_read.py::test_search_vector_returns_hit PASSED`, `tests/test_read.py::test_search_vector_empty_for_unseen_video PASSED`

- [ ] **Step 5: Write the failing test for `search_text`**

Append to `tests/test_read.py`:

```python
@pytest.mark.integration
def test_search_text_matches_content_all(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits = r.search_text("chợ hoa")

    assert len(hits) == 1
    assert hits[0]["keyframe_id"] == kfid
    assert set(hits[0].keys()) == {"keyframe_id", "video_id", "shot_id", "timestamp_ms", "score"}


@pytest.mark.integration
def test_search_text_restricts_to_given_field(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits_ocr = r.search_text("chợ hoa", fields=["ocr_text"])
    hits_caption_only = r.search_text("chợ hoa", fields=["caption"])

    assert len(hits_ocr) == 1       # "chợ hoa tết" is the ocr_text
    assert len(hits_caption_only) == 0  # caption is "một khu chợ", no "hoa" token
```

- [ ] **Step 6: Run to verify it fails**

Run: `./run.sh test`
Expected: FAIL with `AttributeError: 'Reader' object has no attribute 'search_text'`

- [ ] **Step 7: Implement `search_text`**

In `indexdb/read.py`, add to `Reader`:

```python
    def search_text(self, query: str, top_k: int = 100, video_ids: list[str] | None = None,
                     fields: list[str] | None = None) -> list[dict]:
        field = fields[0] if fields else "content_all"
        es_query = {"match": {field: query}}
        if video_ids:
            es_query = {"bool": {"must": es_query, "filter": {"terms": {"video_id": video_ids}}}}
        resp = self.es.client.search(index=self.es.index_name, query=es_query, size=top_k)
        return [
            {
                "keyframe_id": hit["_source"]["keyframe_id"],
                "video_id": hit["_source"]["video_id"],
                "shot_id": hit["_source"]["shot_id"],
                "timestamp_ms": hit["_source"]["timestamp_ms"],
                "score": hit["_score"],
            }
            for hit in resp["hits"]["hits"]
        ]
```

`fields` currently only uses its first element — `search_text` searches exactly one field per call (`content_all` by default, or the single field named in `fields[0]`), matching the A/B use case from [elastic.py:23-25](../../../layer_5/indexdb/elastic.py#L23-L25) (comparing `ocr_text` vs `ocr_recap`). Multi-field weighted search is not needed for that use case and is not implemented.

- [ ] **Step 8: Run to verify `search_text` tests pass**

Run: `./run.sh test`
Expected: `tests/test_read.py::test_search_text_matches_content_all PASSED`, `tests/test_read.py::test_search_text_restricts_to_given_field PASSED`

- [ ] **Step 9: Commit**

```bash
git add indexdb/read.py tests/test_read.py
git commit -m "feat: add Reader.search_vector and Reader.search_text"
```

---

## Task 3: `get_keyframes` + `get_shots` + `get_transcript`

**Files:**
- Modify: `indexdb/read.py`
- Modify: `tests/test_read.py`

**Interfaces:**
- Consumes: `Reader.mongo: MongoStore`, `Reader.data_root: str` from Task 1. Mongo doc shapes from `indexdb/builders.py`: `build_keyframe_doc` ([indexdb/builders.py:14-28](../../../layer_5/indexdb/builders.py#L14-L28)), `build_shot_doc` ([indexdb/builders.py:1-12](../../../layer_5/indexdb/builders.py#L1-L12)), `build_seg_doc` ([indexdb/builders.py:30-38](../../../layer_5/indexdb/builders.py#L30-L38)).
- Produces: `Reader.get_keyframes(self, ids: list[str]) -> list[dict]`, `Reader.get_shots(self, video_id: str) -> list[dict]`, `Reader.get_transcript(self, video_id: str, start_ms: int, end_ms: int) -> list[dict]`.

- [ ] **Step 1: Write the failing test for `get_keyframes`**

Append to `tests/test_read.py`:

```python
@pytest.mark.integration
def test_get_keyframes_absolutizes_path_and_preserves_order(cfg, clean_mongo):
    from indexdb.writers import MongoWriter
    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000007", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0})
    for i, kfid in enumerate(["v001_f0001", "v001_f0002"], start=1):
        w.upsert_keyframe({"video_id": "v001", "keyframe_id": kfid, "shot_id": "v001_000007",
                           "frame_idx": 1500 + i, "timestamp_ms": 60000 + i,
                           "image_path": f"layer_2/x/{kfid}.jpg"})

    r = Reader.__new__(Reader)
    r.mongo, r.data_root = clean_mongo, "/data"

    result = r.get_keyframes(["v001_f0002", "v001_f0001", "v001_f0002"])

    assert [d["keyframe_id"] for d in result] == ["v001_f0002", "v001_f0001", "v001_f0002"]
    assert result[0]["image_path"] == "/data/layer_2/x/v001_f0002.jpg"


@pytest.mark.integration
def test_get_keyframes_skips_unknown_id(cfg, clean_mongo):
    from indexdb.writers import MongoWriter
    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000007", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0})
    w.upsert_keyframe({"video_id": "v001", "keyframe_id": "v001_f0001", "shot_id": "v001_000007",
                       "frame_idx": 1501, "timestamp_ms": 60001, "image_path": "x.jpg"})

    r = Reader.__new__(Reader)
    r.mongo, r.data_root = clean_mongo, "/data"

    result = r.get_keyframes(["v001_f0001", "does_not_exist"])

    assert [d["keyframe_id"] for d in result] == ["v001_f0001"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `./run.sh test`
Expected: FAIL with `AttributeError: 'Reader' object has no attribute 'get_keyframes'`

- [ ] **Step 3: Implement `get_keyframes`**

In `indexdb/read.py`, add to `Reader`:

```python
    def get_keyframes(self, ids: list[str]) -> list[dict]:
        docs = {d["_id"]: d for d in self.mongo.keyframes.find({"_id": {"$in": ids}})}
        out = []
        for i in ids:
            if i not in docs:
                continue
            doc = dict(docs[i])
            doc["keyframe_id"] = doc.pop("_id")
            doc["image_path"] = os.path.join(self.data_root, doc["image_path"])
            out.append(doc)
        return out
```

Iterating over `ids` (not `docs.values()`) is what preserves both order and duplicates from the caller's input — `docs` is a dict, keyed by `_id`, so MongoDB's `$in` query naturally deduplicates matches; re-expanding against the original `ids` list restores whatever order/repetition the caller asked for.

- [ ] **Step 4: Run to verify `get_keyframes` tests pass**

Run: `./run.sh test`
Expected: `tests/test_read.py::test_get_keyframes_absolutizes_path_and_preserves_order PASSED`, `tests/test_read.py::test_get_keyframes_skips_unknown_id PASSED`

- [ ] **Step 5: Write the failing test for `get_shots`**

Append to `tests/test_read.py`:

```python
@pytest.mark.integration
def test_get_shots_sorted_by_start_ms(cfg, clean_mongo):
    from indexdb.writers import MongoWriter
    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000002", "start_frame": 50,
                   "end_frame": 60, "fps": 25.0}, transcript="thoại 2")
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000001", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0}, transcript="thoại 1")

    r = Reader.__new__(Reader)
    r.mongo = clean_mongo

    shots = r.get_shots("v001")

    assert [s["shot_id"] for s in shots] == ["v001_000001", "v001_000002"]
    assert shots[0]["transcript"] == "thoại 1"
```

- [ ] **Step 6: Run to verify it fails**

Run: `./run.sh test`
Expected: FAIL with `AttributeError: 'Reader' object has no attribute 'get_shots'`

- [ ] **Step 7: Implement `get_shots`**

In `indexdb/read.py`, add to `Reader`:

```python
    def get_shots(self, video_id: str) -> list[dict]:
        return list(self.mongo.shots.find({"video_id": video_id}).sort("start_ms", 1))
```

- [ ] **Step 8: Run to verify `get_shots` test passes**

Run: `./run.sh test`
Expected: `tests/test_read.py::test_get_shots_sorted_by_start_ms PASSED`

- [ ] **Step 9: Write the failing test for `get_transcript`**

Append to `tests/test_read.py`:

```python
@pytest.mark.integration
def test_get_transcript_returns_overlapping_segments(cfg, clean_mongo):
    from indexdb.writers import MongoWriter
    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_seg({"video_id": "v001", "seg_id": "v001_seg000", "start_ms": 0,
                  "end_ms": 5000, "text": "trước"})
    w.upsert_seg({"video_id": "v001", "seg_id": "v001_seg001", "start_ms": 4000,
                  "end_ms": 9000, "text": "giữa"})
    w.upsert_seg({"video_id": "v001", "seg_id": "v001_seg002", "start_ms": 20000,
                  "end_ms": 25000, "text": "xa sau"})

    r = Reader.__new__(Reader)
    r.mongo = clean_mongo

    segs = r.get_transcript("v001", start_ms=3000, end_ms=6000)

    assert [s["seg_id"] for s in segs] == ["v001_seg000", "v001_seg001"]
```

`start_ms=3000, end_ms=6000` overlaps segment 0 (`0-5000`, since `3000 < 5000`) and segment 1 (`4000-9000`, since `4000 < 6000`), but not segment 2 (`20000-25000`, starts after the window ends).

- [ ] **Step 10: Run to verify it fails**

Run: `./run.sh test`
Expected: FAIL with `AttributeError: 'Reader' object has no attribute 'get_transcript'`

- [ ] **Step 11: Implement `get_transcript`**

In `indexdb/read.py`, add to `Reader`:

```python
    def get_transcript(self, video_id: str, start_ms: int, end_ms: int) -> list[dict]:
        query = {"video_id": video_id, "start_ms": {"$lt": end_ms}, "end_ms": {"$gt": start_ms}}
        return list(self.mongo.transcript_segments.find(query).sort("start_ms", 1))
```

A segment overlaps the `[start_ms, end_ms)` window when its own start is before the window's end AND its own end is after the window's start — the standard interval-overlap condition.

- [ ] **Step 12: Run to verify `get_transcript` test passes**

Run: `./run.sh test`
Expected: `tests/test_read.py::test_get_transcript_returns_overlapping_segments PASSED`

- [ ] **Step 13: Commit**

```bash
git add indexdb/read.py tests/test_read.py
git commit -m "feat: add Reader.get_keyframes, get_shots, get_transcript"
```

---

## Task 4: `run.sh` wiring + README for the algorithm team

**Files:**
- Modify: `run.sh`
- Modify: `README.Docker.md`

**Interfaces:**
- Consumes: `DB_ENV` array ([run.sh:24-26](../../../layer_5/run.sh#L24-L26)), `shell` case block ([run.sh:94-101](../../../layer_5/run.sh#L94-L101)), `indexdb.read.Reader` from Tasks 1-3.
- Produces: no new Python interface — this task only wires environment/docs so a human can exercise `Reader` interactively and so other containers can join the `milvus` network correctly.

- [ ] **Step 1: Add `DATA_ROOT` to `DB_ENV`**

In `run.sh`, change:

```bash
DB_ENV=(-e MONGO_URI=mongodb://root:rootpass@mongodb:27017
        -e ELASTICSEARCH_URI=http://elasticsearch:9200
        -e MILVUS_URI=http://standalone:19530)
```

to:

```bash
DB_ENV=(-e MONGO_URI=mongodb://root:rootpass@mongodb:27017
        -e ELASTICSEARCH_URI=http://elasticsearch:9200
        -e MILVUS_URI=http://standalone:19530
        -e DATA_ROOT=/data)
```

This affects all 3 call sites that expand `"${DB_ENV[@]}"`: `py()` ([run.sh:33](../../../layer_5/run.sh#L33)), `test` ([run.sh:84](../../../layer_5/run.sh#L84)), `shell` ([run.sh:96](../../../layer_5/run.sh#L96)).

- [ ] **Step 2: Preload `Reader` in the `shell` case**

In `run.sh`, in the `shell)` case block, change:

```bash
shell)
    docker build -q -t "$IMAGE" . > /dev/null
    docker run --rm -it --network milvus -v "$PWD/..:/data:ro" "${DB_ENV[@]}" "$IMAGE" python -i -c "
from indexdb.config import Config
from indexdb.mongo import MongoStore
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore
cfg = Config.from_env(); s = MongoStore(cfg); es = ElasticStore(cfg); mv = MilvusStore(cfg)
print('Sẵn: s (mongo), es (elastic), mv (milvus). VD: s.keyframes.find_one()')"
    ;;
```

to:

```bash
shell)
    docker build -q -t "$IMAGE" . > /dev/null
    docker run --rm -it --network milvus -v "$PWD/..:/data:ro" "${DB_ENV[@]}" "$IMAGE" python -i -c "
from indexdb.config import Config
from indexdb.mongo import MongoStore
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore
from indexdb.read import Reader
cfg = Config.from_env(); s = MongoStore(cfg); es = ElasticStore(cfg); mv = MilvusStore(cfg)
r = Reader(cfg)
print('Sẵn: r (Reader), s (mongo), es (elastic), mv (milvus). VD: r.search_text(\"chợ hoa\")')"
    ;;
```

- [ ] **Step 3: Verify manually**

This task has no automated test (it wires shell environment, not Python logic already covered by Tasks 1-3's tests). Verify manually once the stack is up:

Run: `./run.sh up` then `./run.sh init` then `./run.sh ingest --videos K01_V001` then `./run.sh shell`
Expected: REPL starts without the `AssertionError` from `Reader.__init__`'s `_check_ready` (confirms `init_stores` created the index/collections and `Reader()` can see them); `r.search_text("chợ hoa")` runs without exception (empty result is fine — proves the wiring, not the data).

- [ ] **Step 4: Add the "Cho team thuật toán" section to `README.Docker.md`**

Read `README.Docker.md` first to find its current end, then append:

```markdown
## Cho team thuật toán (đọc dữ liệu, không cài gì lên host)

Không `pip install` gì lên máy host (server dùng chung). Mount thẳng code
`indexdb` (read-only) vào container của bạn và join network `milvus` —
không cần build lại image `layer5`.

```bash
docker run --rm -it --gpus all \
  --network milvus \
  -v /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026/layer_5:/opt/layer5:ro \
  -v /workingspace_aiclub/WorkingSpace/Personal/vannk/Ai_challange_2026:/data:ro \
  -e PYTHONPATH=/opt/layer5 \
  -e MONGO_URI=mongodb://root:rootpass@mongodb:27017 \
  -e ELASTICSEARCH_URI=http://elasticsearch:9200 \
  -e MILVUS_URI=http://standalone:19530 \
  -e DATA_ROOT=/data \
  <ten-image-cua-ban> python
```

Image của bạn chỉ cần 3 client Python thuần (không cần cài `layer_5`):

```
pymongo==4.9.2
elasticsearch==9.0.1
pymilvus==2.6.1
```

Ví dụ dùng:

```python
from indexdb.read import Reader
r = Reader()

# candidate generation — bạn tự encode text/ảnh thành vector, Reader không encode hộ
hits = r.search_vector("beit3", my_1024d_vector, top_k=100)
hits += r.search_text("chợ hoa tết", top_k=100)
# mỗi hit: {"keyframe_id", "video_id", "shot_id", "timestamp_ms" (mili giây), "score"}

# hydrate — lấy đủ metadata + đường dẫn ảnh tuyệt đối để hiển thị/predict
kfs = r.get_keyframes([h["keyframe_id"] for h in hits[:20]])
kfs[0]["image_path"]   # "/data/layer_2/Keyframe_Extracting/benchmark/pipeline_c/K01_V001/..."

# ngữ cảnh thời gian quanh một mốc (VQA: "trước/sau khi rời cửa hàng")
segs = r.get_transcript("K01_V001", start_ms=118000, end_ms=130000)
```

**Nếu container của bạn không join được `--network milvus`:** dùng
`10.0.2.3` (gateway của Docker rootless) thay cho tên container, với port
đã remap ở host: `MONGO_URI=mongodb://root:rootpass@10.0.2.3:27018`,
`ELASTICSEARCH_URI=http://10.0.2.3:19201`, `MILVUS_URI=http://10.0.2.3:19531`.
`DATA_ROOT` không đổi theo cách này — nó chỉ phụ thuộc vào việc bạn mount
dataset vào đâu trong container của chính bạn, không liên quan tới network.
```

- [ ] **Step 5: Commit**

```bash
git add run.sh README.Docker.md
git commit -m "docs: wire DATA_ROOT into run.sh and document Reader usage for the algorithm team"
```

---

## Self-Review Notes

- **Spec coverage:** all 5 `Reader` methods (Task 2, Task 3), `Config.data_root` (Task 1), boundary check at init (Task 1), `run.sh` wiring (Task 4), README for the algorithm team (Task 4) — all covered.
- **Contract fix found while writing this plan:** the original discussion assumed `Hit` would include `frame_idx` for both `search_vector` and `search_text`. Checking `build_es_doc` ([indexdb/builders.py:40-51](../../../layer_5/indexdb/builders.py#L40-L51)) shows Elasticsearch's `_source` never stores `frame_idx` — only Milvus's schema does. `frame_idx` is dropped from the shared `Hit` contract (Global Constraints) rather than faked or fetched via an extra Mongo round-trip; it's redundant with `keyframe_id` and recoverable through `get_keyframes` when actually needed.
- **Type consistency:** `Hit` shape (`keyframe_id`, `video_id`, `shot_id`, `timestamp_ms`, `score`) used identically in Task 2's `search_vector` and `search_text`. `Reader.__new__(Reader)` + manual attribute assignment (bypassing `__init__`) is used consistently across Task 2/3 tests to point at `_test`-suffixed stores without duplicating `_check_ready`'s production-name assumption.
- **`timestamp_ms` vs the competition's `timestamp` (seconds):** the Shared Result Format in `HCMAI25_problem_context.md` wants `timestamp` in seconds. `Hit.timestamp_ms` deliberately does not do that conversion — `Reader` stays a thin, unit-preserving passthrough over Mongo/Milvus/ES (all three store `timestamp_ms`), and the seconds conversion is left to whichever layer actually assembles the competition submission JSON, not to this shared read layer.
