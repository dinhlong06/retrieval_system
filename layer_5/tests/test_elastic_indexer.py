import pytest
from indexdb.writers import MongoWriter
from indexdb.elastic import ElasticStore
from indexdb.elastic_indexer import ElasticIndexer

@pytest.fixture
def setup(clean_mongo, cfg):
    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000007", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0}, transcript="lời thoại")
    w.upsert_frame({"video_id": "v001", "keyframe_id": "v001_f0001",
                    "shot_id": "v001_000007", "frame_idx": 1523, "timestamp_ms": 60920,
                    "image_path": "/x.jpg"})
    w.enrich_frame("v001_f0001", objects={"person": 2}, ocr_text="BẢN TIN", caption="một người")
    es = ElasticStore(cfg, index_name="frame_text_test")
    es.drop_index(); es.create_index()
    yield clean_mongo, es, w
    es.drop_index()

@pytest.mark.integration
def test_index_video_pushes_docs(setup):
    clean_mongo, es, w = setup
    indexer = ElasticIndexer(clean_mongo, es)
    n = indexer.index_video("v001")
    assert n == 1
    es.client.indices.refresh(index="frame_text_test")
    doc = es.client.get(index="frame_text_test", id="v001_f0001")["_source"]
    assert doc["ocr_text"] == "BẢN TIN"
    assert doc["transcript"] == "lời thoại"      # lấy từ shot cha
    assert doc["object_tags"] == ["person"]
    assert clean_mongo.frames.find_one({"_id": "v001_f0001"})["synced"]["elastic"] is True
