import pytest
from indexdb.writers import MongoWriter

@pytest.fixture
def writer(clean_mongo):
    clean_mongo.ensure_indexes()
    return MongoWriter(clean_mongo)

@pytest.mark.integration
def test_upsert_frame_idempotent(writer, clean_mongo):
    rec = {"video_id": "v001", "keyframe_id": "v001_f0001", "shot_id": "v001_000007",
           "frame_idx": 1523, "timestamp_ms": 60920, "image_path": "/x.jpg"}
    writer.upsert_frame(rec)
    writer.upsert_frame(rec)  # chạy lần 2
    assert clean_mongo.frames.count_documents({}) == 1

@pytest.mark.integration
def test_enrich_frame_merges_layer3(writer, clean_mongo):
    rec = {"video_id": "v001", "keyframe_id": "v001_f0001", "shot_id": "v001_000007",
           "frame_idx": 1523, "timestamp_ms": 60920, "image_path": "/x.jpg"}
    writer.upsert_frame(rec)
    writer.enrich_frame("v001_f0001", objects={"person": 2}, ocr_text="X", caption="Y")
    doc = clean_mongo.frames.find_one({"_id": "v001_f0001"})
    assert doc["objects"] == {"person": 2}
    assert doc["ocr_text"] == "X"
    assert doc["caption"] == "Y"
    assert doc["frame_idx"] == 1523

@pytest.mark.integration
def test_mark_step(writer, clean_mongo):
    writer.mark_step("v001", "ocr", "done", count=315)
    doc = clean_mongo.ingest_status.find_one({"_id": "v001"})
    assert doc["steps"]["ocr"]["status"] == "done"
    assert doc["steps"]["ocr"]["count"] == 315
