import json
import os
import numpy as np
from indexdb.mongo import MongoStore
from indexdb.milvus import MilvusStore
from indexdb.builders import build_milvus_entity

class MilvusIndexer:
    def __init__(self, store: MongoStore, mv: MilvusStore):
        self.store = store
        self.mv = mv

    def index_video(self, video_id: str, model: str, emb_dir: str) -> int:
        vecs = np.load(os.path.join(emb_dir, f"{video_id}.npy"))
        with open(os.path.join(emb_dir, f"{video_id}_ids.json")) as f:
            ids = json.load(f)
        assert len(ids) == vecs.shape[0], "số id không khớp số vector"

        scalars = {
            d["_id"]: d
            for d in self.store.frames.find({"_id": {"$in": ids}})
        }
        entities = [
            build_milvus_entity(fid, vecs[i].tolist(), scalars[fid])
            for i, fid in enumerate(ids)
        ]
        name = self.mv.collection_names()[model]
        # delete-then-insert để idempotent (Milvus insert cùng PK không đè)
        self.mv.client.delete(name, filter=f'video_id == "{video_id}"')
        self.mv.client.insert(collection_name=name, data=entities)

        self.store.frames.update_many(
            {"_id": {"$in": ids}}, {"$set": {"synced.milvus": True,
                                             f"embeddings.{model}": True}}
        )
        return len(entities)
