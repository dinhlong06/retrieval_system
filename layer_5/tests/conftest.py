import json
import pytest
import numpy as np
from fastapi.testclient import TestClient
from indexdb.config import Config
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore


@pytest.fixture(scope="session")
def cfg():
    return Config.from_env()


@pytest.fixture
def clean_mongo(cfg):
    from indexdb.mongo import MongoStore
    store_test = MongoStore(cfg, db_name="videoindex_test")
    store_test._client.drop_database("videoindex_test")
    yield store_test
    store_test._client.drop_database("videoindex_test")


@pytest.fixture
def reader_with_data(cfg, clean_mongo, tmp_path):
    from indexdb.writers import MongoWriter
    from indexdb.milvus_indexer import MilvusIndexer

    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000007", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0}, transcript="lời thoại")
    kfid = "v001_f0001"
    w.upsert_frame({"video_id": "v001", "keyframe_id": kfid, "shot_id": "v001_000007",
                    "frame_idx": 1501, "timestamp_ms": 60040,
                    "image_path": "layer_2/x/v001_f0001.jpg"})
    w.enrich_frame(kfid, objects={"person": 0.92}, ocr_text="chợ hoa tết", caption="một khu chợ")

    mv = MilvusStore(cfg, suffix="_test")
    mv.drop_all(); mv.create_collections()
    vec = np.random.rand(1024).astype(np.float32)
    vec /= np.linalg.norm(vec)
    np.save(tmp_path / "v001.npy", vec.reshape(1, -1))
    (tmp_path / "v001_ids.json").write_text(json.dumps([kfid]))
    MilvusIndexer(clean_mongo, mv).index_video("v001", "beit3", str(tmp_path))
    mv.client.flush(mv.collection_names()["beit3"])

    es = ElasticStore(cfg, index_name="frame_text_test")
    es.drop_index(); es.create_index()
    from indexdb.elastic_indexer import ElasticIndexer
    ElasticIndexer(clean_mongo, es).index_video("v001")
    es.client.indices.refresh(index="frame_text_test")

    yield clean_mongo, mv, es, kfid, vec

    mv.drop_all(); es.drop_index()


@pytest.fixture
def api_client(monkeypatch):
    from api.main import app
    monkeypatch.setenv("API_KEY", "secret123")
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
