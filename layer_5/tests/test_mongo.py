import pytest

@pytest.mark.integration
def test_mongo_creates_indexes(clean_mongo):
    clean_mongo.ensure_indexes()
    idx = clean_mongo.frames.index_information()
    keys = [tuple(v["key"]) for v in idx.values()]
    assert (("video_id", 1),) in keys or ("video_id", 1) in [k for ks in keys for k in ks]

@pytest.mark.integration
def test_mongo_collections_accessible(clean_mongo):
    assert clean_mongo.videos.name == "videos"
    assert clean_mongo.shots.name == "shots"
    assert clean_mongo.frames.name == "frames"
    assert clean_mongo.transcript_segments.name == "transcript_segments"
    assert clean_mongo.ingest_status.name == "ingest_status"
