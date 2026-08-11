"""Nạp dataset_batch1 (BTC AIC2026) vào Mongo + Milvus + Elastic.

    python -m indexdb.ingest_batch1 --root /path/dataset_batch1 \\
        [--ocr layer_3/OCR/output_batch1/output_vietocr.json] \\
        [--objects layer_3/ObjectDetection/output_batch1/detections.json] \\
        [--siglip-dir recap_siglip/artifacts/siglip_batch1] [--siglip2-dir siglip_output] \\
        [--videos L21_V001 ...]

shot_id được backfill từ layer_1/batch1/shots.jsonl (TransNetV2 chạy trên
dataset_batch1/videos, xem layer_1/run_shards_batch1.sh) bằng cách khớp
frame_idx của từng keyframe vào khoảng [start_frame, end_frame] của shot.
Video/frame chưa có shot data (shots.jsonl chưa chạy hoặc chưa tới) thì
shot_id để rỗng như trước. Transcript theo shot lấy từ
layer_2/shot_transcript/shot_transcripts_batch1.jsonl.

frame_id = {video_id}_{n:03d} theo thứ tự keyframe "n", khớp sẵn với khoá của
OCR/siglip và tên file ảnh. Không dùng frame_idx làm khoá vì 614 keyframe
(trên 192 video) trùng frame_idx với keyframe khác — dùng nó thì Mongo gộp
mất ảnh còn Milvus lại giữ hai entity cùng primary key.
"""
import argparse
import bisect
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

_DATA_ROOT = os.getenv("DATA_ROOT", "/data")
DEFAULT_SHOTS_PATH = os.path.join(_DATA_ROOT, "layer_1", "batch1", "shots.jsonl")
DEFAULT_TRANSCRIPTS_PATH = os.path.join(
    _DATA_ROOT, "layer_2", "shot_transcript", "shot_transcripts_batch1.jsonl")


def _read_map_keyframes(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _group_by_video(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out.setdefault(rec["video_id"], []).append(rec)
    return out


def _build_shot_lookup(shots_by_video):
    """video_id -> (start_frame đã sort, list shot tương ứng) để tra shot_id qua frame_idx."""
    lookup = {}
    for video_id, shots in shots_by_video.items():
        ordered = sorted(shots, key=lambda s: s["start_frame"])
        lookup[video_id] = ([s["start_frame"] for s in ordered], ordered)
    return lookup


def _find_shot_id(lookup, video_id, frame_idx):
    starts, ordered = lookup.get(video_id, ([], []))
    i = bisect.bisect_right(starts, frame_idx) - 1
    if i < 0:
        return ""
    shot = ordered[i]
    return shot["shot_id"] if frame_idx <= shot["end_frame"] else ""


def _load_objects(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    objects_by_frame = {}
    for r in rows:
        labels = {}
        for o in r["objects"]:
            if o["confidence"] >= MIN_CONF:
                labels[o["label"]] = max(labels.get(o["label"], 0.0), o["confidence"])
        if labels:
            objects_by_frame[r["frame_id"]] = labels
    return objects_by_frame


def _load_ocr(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["frame_id"]: " ".join(t["text"] for t in r["texts"] if t["confidence"] >= MIN_CONF)
                for r in json.load(f)}


def _index_vectors(store: MongoStore, mv: MilvusStore, video_id: str, model: str,
                   vecs: np.ndarray, frame_ids: list[str]) -> int:
    assert vecs.shape[0] == len(frame_ids), f"{video_id}: số vector {model} không khớp số frame"
    scalars = {d["_id"]: d for d in store.frames.find({"_id": {"$in": frame_ids}})}
    entities = [build_milvus_entity(fid, vecs[i].tolist(), scalars[fid]) for i, fid in enumerate(frame_ids)]
    name = mv.collection_names()[model]
    mv.client.delete(name, filter=f'video_id == "{video_id}"')
    mv.client.insert(collection_name=name, data=entities)
    store.frames.update_many({"_id": {"$in": frame_ids}},
                             {"$set": {"synced.milvus": True, f"embeddings.{model}": True}})
    return len(entities)


def main():
    ap = argparse.ArgumentParser(prog="indexdb.ingest_batch1")
    ap.add_argument("--root", required=True, help="thư mục dataset_batch1")
    ap.add_argument("--ocr", help="đường dẫn output_vietocr.json (OCR batch1, key theo {video_id}_{n:03d})")
    ap.add_argument("--objects", help="đường dẫn detections.json (object detection batch1, key theo frame_id)")
    ap.add_argument("--siglip-dir", help="thư mục siglip batch1 ({video_id}.npy + {video_id}_ids.json, ids theo n)")
    ap.add_argument("--siglip2-dir", help="thư mục siglip2 batch1, cùng format --siglip-dir (1152 chiều)")
    ap.add_argument("--shots", default=DEFAULT_SHOTS_PATH,
                    help="shots.jsonl batch1 (layer_1/run_shards_batch1.sh); dùng để backfill shot_id qua frame_idx")
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS_PATH,
                    help="shot_transcripts_batch1.jsonl (layer_2/shot_transcript/run_batch1.sh)")
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--resume", action="store_true", help="bỏ qua video đã nạp xong")
    args = ap.parse_args()

    mk_dir = os.path.join(args.root, "map-keyframes")
    video_ids = args.videos or sorted(f[:-4] for f in os.listdir(mk_dir) if f.endswith(".csv"))
    ocr = _load_ocr(args.ocr)
    objects_by_frame = _load_objects(args.objects)
    shots_by_video = _group_by_video(args.shots)
    shot_lookup = _build_shot_lookup(shots_by_video)
    transcripts = {r["shot_id"]: r["text"] for r in
                   (json.loads(l) for l in open(args.transcripts, encoding="utf-8"))} \
        if os.path.exists(args.transcripts) else {}

    cfg = Config.from_env()
    store = MongoStore(cfg)
    writer = MongoWriter(store)
    es_idx = ElasticIndexer(store, ElasticStore(cfg))
    mv = MilvusStore(cfg)

    for video_id in video_ids:
        if args.resume and store.ingest_status.find_one({"_id": video_id, "steps.elastic.status": "done"}):
            print(f"{video_id}: bỏ qua (đã xong)")
            continue

        rows = _read_map_keyframes(os.path.join(mk_dir, f"{video_id}.csv"))

        frame_ids = []
        for row in rows:
            n = f"{int(row['n']):03d}"
            frame_id = f"{video_id}_{n}"
            frame_ids.append(frame_id)
            image_path = os.path.relpath(
                os.path.join(args.root, "keyframe", "keyframes", video_id, f"{n}.jpg"), args.root)
            frame_idx = int(row["frame_idx"])
            shot_id = _find_shot_id(shot_lookup, video_id, frame_idx)
            writer.upsert_frame({
                "keyframe_id": frame_id, "video_id": video_id, "shot_id": shot_id,
                "frame_idx": frame_idx, "timestamp_ms": round(float(row["pts_time"]) * 1000),
                "image_path": image_path, "batch": "batch1",
            })
            objects = objects_by_frame.get(frame_id)
            ocr_text = ocr.get(frame_id)
            if objects or ocr_text or shot_id:
                writer.enrich_frame(frame_id, objects=objects, ocr_text=ocr_text,
                                    shot_id=shot_id or None)

        for shot in shots_by_video.get(video_id, []):
            writer.upsert_shot(shot, transcript=transcripts.get(shot["shot_id"], ""))

        media_info_path = os.path.join(args.root, "media-info", f"{video_id}.json")
        video_doc = {"_id": video_id, "fps": float(rows[0]["fps"]), "batch": "batch1"}
        if os.path.exists(media_info_path):
            with open(media_info_path, encoding="utf-8") as f:
                video_doc["media_info"] = json.load(f)
        writer.upsert_video(video_doc)
        writer.mark_step(video_id, "mongo", "done", len(rows))

        clip_path = os.path.join(args.root, "clip-features-32", f"{video_id}.npy")
        n_vec = _index_vectors(store, mv, video_id, "clip32", np.load(clip_path), frame_ids) \
            if os.path.exists(clip_path) else 0
        models = ["clip32"]

        for model, emb_dir in (("siglip", args.siglip_dir), ("siglip2", args.siglip2_dir)):
            npy_path = os.path.join(emb_dir or "", f"{video_id}.npy")
            if emb_dir and os.path.exists(npy_path):
                with open(os.path.join(emb_dir, f"{video_id}_ids.json"), encoding="utf-8") as f:
                    emb_frame_ids = [f"{video_id}_{i}" for i in json.load(f)]
                n_vec += _index_vectors(store, mv, video_id, model, np.load(npy_path), emb_frame_ids)
                models.append(model)
        writer.mark_step(video_id, "milvus", "done", n_vec)

        n_es = es_idx.index_video(video_id)
        writer.mark_step(video_id, "elastic", "done", n_es)
        print(f"{video_id}: {len(rows)} frame, {n_vec} vector ({'+'.join(models)})")


if __name__ == "__main__":
    main()
