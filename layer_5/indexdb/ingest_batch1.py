"""Nạp dataset_batch1 (BTC AIC2026) vào Mongo + Milvus + Elastic.

    python -m indexdb.ingest_batch1 --root /path/dataset_batch1 [--videos L21_V001 ...]

Batch1 không có shot detection/OCR/transcript riêng — chỉ có map-keyframes,
ảnh keyframe, object detection (Faster R-CNN/OpenImages) và CLIP-32 embedding
do BTC cung cấp sẵn. shot_id để rỗng, backfill sau khi có shot data.
"""
import argparse
import csv
import json
import os

import numpy as np

from indexdb.builders import build_milvus_entity
from indexdb.config import Config
from indexdb.elastic import ElasticStore
from indexdb.elastic_indexer import ElasticIndexer
from indexdb.milvus import MilvusStore
from indexdb.mongo import MongoStore
from indexdb.writers import MongoWriter

MIN_CONF = 0.5


def _read_map_keyframes(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_objects(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    objects = {}
    for label, score in zip(d["detection_class_entities"], (float(s) for s in d["detection_scores"])):
        if score >= MIN_CONF:
            objects[label] = max(objects.get(label, 0.0), score)
    return objects


def _index_clip32(store: MongoStore, mv: MilvusStore, video_id: str, clip_path: str, frame_ids: list[str]) -> int:
    vecs = np.load(clip_path)
    assert vecs.shape[0] == len(frame_ids), f"{video_id}: số vector CLIP-32 không khớp số frame"
    scalars = {d["_id"]: d for d in store.frames.find({"_id": {"$in": frame_ids}})}
    entities = [build_milvus_entity(fid, vecs[i].tolist(), scalars[fid]) for i, fid in enumerate(frame_ids)]
    name = mv.collection_names()["clip32"]
    mv.client.delete(name, filter=f'video_id == "{video_id}"')
    mv.client.insert(collection_name=name, data=entities)
    store.frames.update_many({"_id": {"$in": frame_ids}},
                             {"$set": {"synced.milvus": True, "embeddings.clip32": True}})
    return len(entities)


def main():
    ap = argparse.ArgumentParser(prog="indexdb.ingest_batch1")
    ap.add_argument("--root", required=True, help="thư mục dataset_batch1")
    ap.add_argument("--videos", nargs="*")
    args = ap.parse_args()

    mk_dir = os.path.join(args.root, "map-keyframes")
    video_ids = args.videos or sorted(f[:-4] for f in os.listdir(mk_dir) if f.endswith(".csv"))

    cfg = Config.from_env()
    store = MongoStore(cfg)
    writer = MongoWriter(store)
    es_idx = ElasticIndexer(store, ElasticStore(cfg))
    mv = MilvusStore(cfg)

    for video_id in video_ids:
        rows = _read_map_keyframes(os.path.join(mk_dir, f"{video_id}.csv"))

        frame_ids = []
        for row in rows:
            n = int(row["n"])
            frame_idx = int(row["frame_idx"])
            frame_id = f"{video_id}_{frame_idx}"
            frame_ids.append(frame_id)
            image_path = os.path.relpath(
                os.path.join(args.root, "keyframe", "keyframes", video_id, f"{n:03d}.jpg"), args.root)
            writer.upsert_frame({
                "keyframe_id": frame_id, "video_id": video_id, "shot_id": "",
                "frame_idx": frame_idx, "timestamp_ms": round(float(row["pts_time"]) * 1000),
                "image_path": image_path, "batch": "batch1",
            })
            objects = _load_objects(os.path.join(args.root, "objects", video_id, f"{n:03d}.json"))
            if objects:
                writer.enrich_frame(frame_id, objects=objects)

        media_info_path = os.path.join(args.root, "media-info", f"{video_id}.json")
        video_doc = {"_id": video_id, "fps": float(rows[0]["fps"]), "batch": "batch1"}
        if os.path.exists(media_info_path):
            with open(media_info_path, encoding="utf-8") as f:
                video_doc["media_info"] = json.load(f)
        writer.upsert_video(video_doc)
        writer.mark_step(video_id, "mongo", "done", len(rows))

        clip_path = os.path.join(args.root, "clip-features-32", f"{video_id}.npy")
        n_vec = _index_clip32(store, mv, video_id, clip_path, frame_ids) if os.path.exists(clip_path) else 0
        writer.mark_step(video_id, "milvus", "done", n_vec)

        n_es = es_idx.index_video(video_id)
        writer.mark_step(video_id, "elastic", "done", n_es)
        print(f"{video_id}: {len(rows)} frame, {n_vec} clip32 vector")


if __name__ == "__main__":
    main()
