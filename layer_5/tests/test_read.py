import pytest
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore
from indexdb.read import Reader


@pytest.mark.integration
def test_check_ready_passes_when_stores_ready(cfg):
    es = ElasticStore(cfg, index_name="frame_text_test")
    mv = MilvusStore(cfg, suffix="_test")
    es.drop_index(); es.create_index()
    mv.drop_all(); mv.create_collections()
    try:
        Reader._check_ready(es, mv)  # must not raise
    finally:
        es.drop_index(); mv.drop_all()


@pytest.mark.integration
def test_check_ready_raises_when_elastic_index_missing(cfg):
    es = ElasticStore(cfg, index_name="frame_text_test")
    es.drop_index()
    mv = MilvusStore(cfg, suffix="_test")
    mv.drop_all(); mv.create_collections()
    try:
        with pytest.raises(AssertionError, match="frame_text_test"):
            Reader._check_ready(es, mv)
    finally:
        mv.drop_all()


@pytest.mark.integration
def test_check_ready_raises_when_milvus_collection_missing(cfg):
    es = ElasticStore(cfg, index_name="frame_text_test")
    es.drop_index(); es.create_index()
    mv = MilvusStore(cfg, suffix="_test")
    mv.drop_all()
    try:
        with pytest.raises(AssertionError):
            Reader._check_ready(es, mv)
    finally:
        es.drop_index()


@pytest.mark.integration
def test_search_vector_returns_hit(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)  # bypass __init__'s production-name check; point at _test stores directly
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits = r.search_vector("beit3", vec.tolist(), top_k=5)

    assert len(hits) == 1
    hit = hits[0]
    assert hit["frame_id"] == kfid
    assert hit["video_id"] == "v001"
    assert hit["shot_id"] == "v001_000007"
    assert hit["timestamp_ms"] == 60040
    assert set(hit.keys()) == {"frame_id", "video_id", "shot_id", "timestamp_ms", "score"}


@pytest.mark.integration
def test_search_vector_empty_for_unseen_video(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits = r.search_vector("beit3", vec.tolist(), top_k=5, video_ids=["v999_never_ingested"])

    assert hits == []


@pytest.mark.integration
def test_search_ocr_matches_ocr_text(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits = r.search_ocr("chợ hoa")

    assert len(hits) == 1
    assert hits[0]["frame_id"] == kfid
    assert set(hits[0].keys()) == {"frame_id", "video_id", "shot_id", "timestamp_ms", "score"}


@pytest.mark.integration
def test_search_ocr_empty_for_unseen_video(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits = r.search_ocr("chợ hoa", video_ids=["v999_never_ingested"])

    assert hits == []


@pytest.mark.integration
def test_search_asr_matches_shot_transcript(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits = r.search_asr("lời thoại")

    assert len(hits) == 1
    assert hits[0]["frame_id"] == kfid


@pytest.mark.integration
def test_search_all_matches_across_sources(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits_ocr_term = r.search_all("chợ hoa")     # từ ocr_text "chợ hoa tết"
    hits_asr_term = r.search_all("lời thoại")   # từ transcript shot

    assert len(hits_ocr_term) == 1
    assert hits_ocr_term[0]["frame_id"] == kfid
    assert len(hits_asr_term) == 1
    assert hits_asr_term[0]["frame_id"] == kfid


@pytest.mark.integration
def test_search_object_matches_label(cfg, reader_with_data):
    _, mv, es, kfid, vec = reader_with_data
    r = Reader.__new__(Reader)
    r.mv, r.es, r.data_root = mv, es, cfg.data_root

    hits = r.search_object(["person"])
    hits_other = r.search_object(["car"])

    assert len(hits) == 1
    assert hits[0]["frame_id"] == kfid
    assert hits[0]["score"] == 0.92  # confidence thật từ Mongo, không phải hằng số term-match
    assert hits_other == []


@pytest.mark.integration
def test_get_frames_absolutizes_path_and_preserves_order(cfg, clean_mongo):
    from indexdb.writers import MongoWriter
    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000007", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0})
    for i, kfid in enumerate(["v001_f0001", "v001_f0002"], start=1):
        w.upsert_frame({"video_id": "v001", "keyframe_id": kfid, "shot_id": "v001_000007",
                           "frame_idx": 1500 + i, "timestamp_ms": 60000 + i,
                           "image_path": f"layer_2/x/{kfid}.jpg"})

    r = Reader.__new__(Reader)
    r.mongo, r.data_root = clean_mongo, "/data"

    result = r.get_frames(["v001_f0002", "v001_f0001", "v001_f0002"])

    assert [d["frame_id"] for d in result] == ["v001_f0002", "v001_f0001", "v001_f0002"]
    assert result[0]["image_path"] == "/data/layer_2/x/v001_f0002.jpg"


@pytest.mark.integration
def test_get_frames_skips_unknown_id(cfg, clean_mongo):
    from indexdb.writers import MongoWriter
    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000007", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0})
    w.upsert_frame({"video_id": "v001", "keyframe_id": "v001_f0001", "shot_id": "v001_000007",
                       "frame_idx": 1501, "timestamp_ms": 60001, "image_path": "x.jpg"})

    r = Reader.__new__(Reader)
    r.mongo, r.data_root = clean_mongo, "/data"

    result = r.get_frames(["v001_f0001", "does_not_exist"])

    assert [d["frame_id"] for d in result] == ["v001_f0001"]


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
