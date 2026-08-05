import pytest
from fastapi.testclient import TestClient
from api.main import app, get_reader
from indexdb.read import Reader


def test_health_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_ocr_rejects_missing_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    client = TestClient(app)
    resp = client.post("/search/ocr", json={"query": "x"})
    assert resp.status_code == 401


def test_search_ocr_rejects_wrong_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    client = TestClient(app)
    resp = client.post("/search/ocr", json={"query": "x"}, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


@pytest.mark.integration
def test_search_vector_endpoint_returns_hits(cfg, reader_with_data, api_client):
    _, mv, es, kfid, vec = reader_with_data
    test_reader = Reader.__new__(Reader)
    test_reader.mv, test_reader.es, test_reader.data_root = mv, es, cfg.data_root
    app.dependency_overrides[get_reader] = lambda: test_reader

    resp = api_client.post("/search/vector",
                            json={"model": "beit3", "vector": vec.tolist(), "top_k": 5},
                            headers={"X-API-Key": "secret123"})

    assert resp.status_code == 200
    hits = resp.json()
    assert len(hits) == 1
    assert hits[0]["keyframe_id"] == kfid
    assert hits[0]["frame_idx"] == 1501
    assert set(hits[0].keys()) == {"keyframe_id", "video_id", "shot_id", "frame_idx", "timestamp_ms", "score"}


def test_missing_api_key_env_rejects_any_request(monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post("/search/ocr", json={"query": "x"})
    assert resp.status_code == 401
    resp = client.post("/search/ocr", json={"query": "x"}, headers={"X-API-Key": "whatever"})
    assert resp.status_code == 401


@pytest.mark.integration
def test_search_vector_endpoint_filters_by_video_ids(cfg, reader_with_data, api_client):
    _, mv, es, kfid, vec = reader_with_data
    test_reader = Reader.__new__(Reader)
    test_reader.mv, test_reader.es, test_reader.data_root = mv, es, cfg.data_root
    app.dependency_overrides[get_reader] = lambda: test_reader

    resp = api_client.post("/search/vector",
                            json={"model": "beit3", "vector": vec.tolist(), "top_k": 5,
                                  "video_ids": ["v999_never_ingested"]},
                            headers={"X-API-Key": "secret123"})

    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.integration
def test_search_ocr_endpoint_filters_by_video_ids(cfg, reader_with_data, api_client):
    _, mv, es, kfid, vec = reader_with_data
    test_reader = Reader.__new__(Reader)
    test_reader.mv, test_reader.es, test_reader.data_root = mv, es, cfg.data_root
    app.dependency_overrides[get_reader] = lambda: test_reader

    resp = api_client.post("/search/ocr",
                            json={"query": "chợ hoa", "video_ids": ["v999_never_ingested"]},
                            headers={"X-API-Key": "secret123"})

    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.integration
def test_search_ocr_endpoint_returns_hits(cfg, reader_with_data, api_client):
    _, mv, es, kfid, vec = reader_with_data
    test_reader = Reader.__new__(Reader)
    test_reader.mv, test_reader.es, test_reader.data_root = mv, es, cfg.data_root
    app.dependency_overrides[get_reader] = lambda: test_reader

    resp = api_client.post("/search/ocr", json={"query": "chợ hoa"},
                            headers={"X-API-Key": "secret123"})

    assert resp.status_code == 200
    hits = resp.json()
    assert len(hits) == 1
    assert hits[0]["keyframe_id"] == kfid
    assert hits[0]["frame_idx"] == 1501


@pytest.mark.integration
def test_search_asr_endpoint_returns_hits(cfg, reader_with_data, api_client):
    _, mv, es, kfid, vec = reader_with_data
    test_reader = Reader.__new__(Reader)
    test_reader.mv, test_reader.es, test_reader.data_root = mv, es, cfg.data_root
    app.dependency_overrides[get_reader] = lambda: test_reader

    resp = api_client.post("/search/asr", json={"query": "lời thoại"},
                            headers={"X-API-Key": "secret123"})

    assert resp.status_code == 200
    hits = resp.json()
    assert len(hits) == 1
    assert hits[0]["keyframe_id"] == kfid
    assert hits[0]["frame_idx"] == 1501


@pytest.mark.integration
def test_search_all_endpoint_returns_hits(cfg, reader_with_data, api_client):
    _, mv, es, kfid, vec = reader_with_data
    test_reader = Reader.__new__(Reader)
    test_reader.mv, test_reader.es, test_reader.data_root = mv, es, cfg.data_root
    app.dependency_overrides[get_reader] = lambda: test_reader

    resp = api_client.post("/search/all", json={"query": "chợ hoa"},
                            headers={"X-API-Key": "secret123"})

    assert resp.status_code == 200
    hits = resp.json()
    assert len(hits) == 1
    assert hits[0]["keyframe_id"] == kfid
    assert hits[0]["frame_idx"] == 1501


@pytest.mark.integration
def test_search_object_endpoint_returns_hits(cfg, reader_with_data, api_client):
    mongo, mv, es, kfid, vec = reader_with_data
    test_reader = Reader.__new__(Reader)
    test_reader.mv, test_reader.es, test_reader.mongo, test_reader.data_root = mv, es, mongo, cfg.data_root
    app.dependency_overrides[get_reader] = lambda: test_reader

    resp = api_client.post("/search/object", json={"labels": ["person"]},
                            headers={"X-API-Key": "secret123"})

    assert resp.status_code == 200
    hits = resp.json()
    assert len(hits) == 1
    assert hits[0]["keyframe_id"] == kfid
    assert hits[0]["frame_idx"] == 1501


@pytest.mark.integration
def test_frames_endpoint_preserves_order_and_skips_unknown(cfg, clean_mongo, api_client):
    from indexdb.writers import MongoWriter

    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000007", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0})
    for i, kfid in enumerate(["v001_f0001", "v001_f0002"], start=1):
        w.upsert_frame({"video_id": "v001", "keyframe_id": kfid, "shot_id": "v001_000007",
                           "frame_idx": 1500 + i, "timestamp_ms": 60000 + i,
                           "image_path": f"layer_2/x/{kfid}.jpg"})

    test_reader = Reader.__new__(Reader)
    test_reader.mongo, test_reader.data_root = clean_mongo, "/data"
    app.dependency_overrides[get_reader] = lambda: test_reader

    resp = api_client.post("/frames",
                            json={"ids": ["v001_f0002", "v001_f0001", "does_not_exist"]},
                            headers={"X-API-Key": "secret123"})

    assert resp.status_code == 200
    body = resp.json()
    assert [d["keyframe_id"] for d in body] == ["v001_f0002", "v001_f0001"]
    assert body[0]["image_path"] == "/data/layer_2/x/v001_f0002.jpg"


@pytest.mark.integration
def test_shots_endpoint_sorted_by_start_ms(cfg, clean_mongo, api_client):
    from indexdb.writers import MongoWriter

    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000002", "start_frame": 50,
                   "end_frame": 60, "fps": 25.0}, transcript="thoại 2")
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000001", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0}, transcript="thoại 1")

    test_reader = Reader.__new__(Reader)
    test_reader.mongo = clean_mongo
    app.dependency_overrides[get_reader] = lambda: test_reader

    resp = api_client.get("/shots/v001", headers={"X-API-Key": "secret123"})

    assert resp.status_code == 200
    body = resp.json()
    assert [s["shot_id"] for s in body] == ["v001_000001", "v001_000002"]


@pytest.mark.integration
def test_transcript_endpoint_returns_overlapping_segments(cfg, clean_mongo, api_client):
    from indexdb.writers import MongoWriter

    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_seg({"video_id": "v001", "seg_id": "v001_seg000", "start_ms": 0,
                  "end_ms": 5000, "text": "trước"})
    w.upsert_seg({"video_id": "v001", "seg_id": "v001_seg001", "start_ms": 4000,
                  "end_ms": 9000, "text": "giữa"})
    w.upsert_seg({"video_id": "v001", "seg_id": "v001_seg002", "start_ms": 20000,
                  "end_ms": 25000, "text": "xa sau"})

    test_reader = Reader.__new__(Reader)
    test_reader.mongo = clean_mongo
    app.dependency_overrides[get_reader] = lambda: test_reader

    resp = api_client.get("/transcript/v001", params={"start_ms": 3000, "end_ms": 6000},
                           headers={"X-API-Key": "secret123"})

    assert resp.status_code == 200
    body = resp.json()
    assert [s["seg_id"] for s in body] == ["v001_seg000", "v001_seg001"]
