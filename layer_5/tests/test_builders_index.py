from indexdb.builders import build_es_doc, build_milvus_entity

KEYFRAME = {
    "_id": "v001_f0001", "video_id": "v001", "shot_id": "v001_000007",
    "frame_idx": 1523, "timestamp_ms": 60920,
    "image_path": "/x.jpg",
    "objects": {"person": 2, "car": 1},
    "ocr_text": "BẢN TIN", "caption": "một người",
}

def test_build_es_doc():
    doc = build_es_doc(KEYFRAME, transcript="lời thoại shot")
    assert doc["frame_id"] == "v001_f0001"
    assert doc["video_id"] == "v001"
    assert doc["shot_id"] == "v001_000007"
    assert doc["frame_idx"] == 1523
    assert doc["timestamp_ms"] == 60920
    assert doc["ocr_text"] == "BẢN TIN"
    assert doc["caption"] == "một người"
    assert doc["transcript"] == "lời thoại shot"
    assert sorted(doc["object_tags"]) == ["car", "person"]
    # content_all KHÔNG set thủ công — do ES copy_to sinh ra
    assert "content_all" not in doc

def test_build_milvus_entity():
    vec = [0.1] * 1024
    ent = build_milvus_entity("v001_f0001", vec, scalars=KEYFRAME)
    assert ent["frame_id"] == "v001_f0001"
    assert ent["video_id"] == "v001"
    assert ent["frame_idx"] == 1523
    assert ent["shot_id"] == "v001_000007"
    assert ent["timestamp_ms"] == 60920
    assert ent["embedding"] == vec
