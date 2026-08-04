import json
import numpy as np
import pytest
from indexdb.writers import MongoWriter
from indexdb.milvus import MilvusStore
from indexdb.milvus_indexer import MilvusIndexer

@pytest.fixture
def emb_files(tmp_path):
    ids = ["v001_f0001", "v001_f0002"]
    vecs = np.random.rand(2, 1024).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(tmp_path / "v001.npy", vecs)
    (tmp_path / "v001_ids.json").write_text(json.dumps(ids))
    return tmp_path

@pytest.fixture
def setup(clean_mongo, cfg, emb_files):
    clean_mongo.ensure_indexes()
    w = MongoWriter(clean_mongo)
    for seq, kfid in enumerate(["v001_f0001", "v001_f0002"], start=1):
        w.upsert_frame({"video_id": "v001", "keyframe_id": kfid, "shot_id": "v001_000007",
                           "frame_idx": 1500 + seq, "timestamp_ms": 60000 + seq,
                           "image_path": f"/{kfid}.jpg"})
    mv = MilvusStore(cfg, suffix="_test")
    mv.drop_all(); mv.create_collections()
    yield clean_mongo, mv, emb_files
    mv.drop_all()

@pytest.mark.integration
def test_index_video_inserts(setup):
    clean_mongo, mv, emb_dir = setup
    indexer = MilvusIndexer(clean_mongo, mv)
    n = indexer.index_video("v001", model="beit3", emb_dir=str(emb_dir))
    assert n == 2
    name = mv.collection_names()["beit3"]
    mv.client.flush(name)
    rows = mv.client.query(name, filter='video_id == "v001"',
                           output_fields=["frame_id", "frame_idx"])
    assert len(rows) == 2
    assert clean_mongo.frames.find_one({"_id": "v001_f0001"})["synced"]["milvus"] is True

@pytest.mark.integration
def test_reindex_no_duplicates(setup):
    clean_mongo, mv, emb_dir = setup
    indexer = MilvusIndexer(clean_mongo, mv)
    indexer.index_video("v001", model="beit3", emb_dir=str(emb_dir))
    indexer.index_video("v001", model="beit3", emb_dir=str(emb_dir))  # lần 2
    name = mv.collection_names()["beit3"]
    mv.client.flush(name)
    rows = mv.client.query(name, filter='video_id == "v001"', output_fields=["frame_id"])
    assert len(rows) == 2  # delete-then-insert, không nhân đôi
