import json
import numpy as np
import pytest
from indexdb.writers import MongoWriter
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore
from indexdb.elastic_indexer import ElasticIndexer
from indexdb.milvus_indexer import MilvusIndexer

@pytest.mark.integration
def test_full_roundtrip(clean_mongo, cfg, tmp_path):
    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    w.upsert_shot({"video_id": "v001", "shot_id": "v001_000007", "start_frame": 1,
                   "end_frame": 9, "fps": 25.0}, transcript="lời thoại")
    ids = ["v001_f0001", "v001_f0002"]
    for seq, kfid in enumerate(ids, start=1):
        w.upsert_frame({"video_id": "v001", "keyframe_id": kfid, "shot_id": "v001_000007",
                           "frame_idx": 1500 + seq, "timestamp_ms": 60000 + seq,
                           "image_path": f"/{kfid}.jpg"})
        w.enrich_frame(kfid, objects={"person": 1}, ocr_text="tin", caption="cap")

    vecs = np.random.rand(2, 1024).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(tmp_path / "v001.npy", vecs)
    (tmp_path / "v001_ids.json").write_text(json.dumps(ids))

    es = ElasticStore(cfg, index_name="frame_text_test")
    es.drop_index(); es.create_index()
    mv = MilvusStore(cfg, suffix="_test")
    mv.drop_all(); mv.create_collections()

    n_es = ElasticIndexer(clean_mongo, es).index_video("v001")
    n_mv = MilvusIndexer(clean_mongo, mv).index_video("v001", "beit3", str(tmp_path))

    assert n_es == 2
    assert n_mv == 2
    assert clean_mongo.frames.count_documents({"video_id": "v001"}) == 2
    es.client.indices.refresh(index="frame_text_test")
    assert es.client.count(index="frame_text_test")["count"] == 2

    es.drop_index(); mv.drop_all()
