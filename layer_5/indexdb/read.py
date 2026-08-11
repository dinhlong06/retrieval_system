import os
from indexdb.config import Config
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore
from indexdb.mongo import MongoStore


class Reader:
    def __init__(self, cfg: Config | None = None):
        cfg = cfg or Config.from_env()
        self.mongo = MongoStore(cfg)
        self.es = ElasticStore(cfg)
        self.mv = MilvusStore(cfg)
        self.data_root = cfg.data_root
        self._check_ready(self.es, self.mv)

    @staticmethod
    def _check_ready(es: ElasticStore, mv: MilvusStore):
        assert es.client.indices.exists(index=es.index_name), (
            f"thiếu index '{es.index_name}' — kiểm tra ELASTICSEARCH_URI, "
            f"hoặc chạy indexdb.init_stores nếu chưa init"
        )
        for name in mv.collection_names().values():
            assert mv.client.has_collection(name), (
                f"thiếu collection '{name}' — kiểm tra MILVUS_URI, "
                f"hoặc chạy indexdb.init_stores nếu chưa init"
            )

    def search_vector(self, model: str, vector: list[float], top_k: int = 100,
                       video_ids: list[str] | None = None, min_score: float | None = None) -> list[dict]:
        name = self.mv.collection_names()[model]
        flt = f'video_id in {video_ids}' if video_ids else None
        # range search: COSINE similarity nằm trong [radius, range_filter). range_filter=2.0
        # (max lý thuyết của COSINE là 1.0) để không phải lo min_score gần biên top do lỗi
        # làm tròn dấu phẩy động, và vẫn hợp lệ (radius < range_filter) với mọi min_score <= 1.
        search_params = {"params": {"radius": min_score, "range_filter": 2.0}} if min_score is not None else None
        results = self.mv.client.search(
            collection_name=name, data=[vector], limit=top_k, filter=flt, search_params=search_params,
            output_fields=["video_id", "shot_id", "frame_idx", "timestamp_ms"],
        )
        return [
            {
                "keyframe_id": hit["frame_id"],
                "video_id": hit["entity"]["video_id"],
                "shot_id": hit["entity"]["shot_id"],
                "frame_idx": hit["entity"]["frame_idx"],
                "timestamp_ms": hit["entity"]["timestamp_ms"],
                "score": hit["distance"],
            }
            for hit in results[0]
        ]

    def search_ocr(self, query: str, top_k: int = 100, video_ids: list[str] | None = None,
                   frame_ids: list[str] | None = None) -> list[dict]:
        # multi_match cả ocr_text (PaddleOCR gốc) và ocr_api (đã hiệu đính) vì
        # không biết trước nguồn nào khớp; fuzziness="AUTO" chịu lỗi chính tả OCR.
        es_query = {"multi_match": {"query": query, "fields": ["ocr_text", "ocr_api"], "fuzziness": "AUTO"}}
        return self._search(es_query, top_k, video_ids, frame_ids)

    def search_asr(self, query: str, top_k: int = 100, video_ids: list[str] | None = None,
                   frame_ids: list[str] | None = None, min_score: float | None = None) -> list[dict]:
        es_query = {"match": {"transcript": {"query": query, "fuzziness": "AUTO"}}}
        return self._search(es_query, top_k, video_ids, frame_ids, min_score)

    def search_all(self, query: str, top_k: int = 100, video_ids: list[str] | None = None,
                   frame_ids: list[str] | None = None) -> list[dict]:
        # content_all gộp ocr_text/ocr_api/caption/transcript qua copy_to — dùng khi
        # không cần biết khớp từ nguồn nào, chỉ cần tìm nhanh trên toàn bộ text.
        es_query = {"match": {"content_all": {"query": query, "fuzziness": "AUTO"}}}
        return self._search(es_query, top_k, video_ids, frame_ids)

    def search_object(self, labels: list[str], top_k: int = 100, video_ids: list[str] | None = None,
                      frame_ids: list[str] | None = None, min_score: float | None = None) -> list[dict]:
        # object_tags là keyword (nhãn rời rạc từ detector) nên match chính xác qua terms, không fuzzy.
        # ES chỉ biết frame có/không có nhãn, không biết độ tự tin — lấy confidence
        # thật (max giữa các object cùng nhãn) từ Mongo rồi sort lại theo đó.
        # min_score lọc TRÊN confidence thật này, không phải _score của ES (không có ý nghĩa
        # với terms query) — nên lọc sau khi join Mongo, không truyền vào _search.
        hits = self._search({"terms": {"object_tags": labels}}, top_k, video_ids, frame_ids)
        objects_by_frame = {
            d["_id"]: d["objects"]
            for d in self.mongo.frames.find({"_id": {"$in": [h["keyframe_id"] for h in hits]}}, {"objects": 1})
        }
        for hit in hits:
            objects = objects_by_frame[hit["keyframe_id"]]
            hit["score"] = max(objects[label] for label in labels if label in objects)
        if min_score is not None:
            hits = [h for h in hits if h["score"] >= min_score]
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits

    def _search(self, es_query: dict, top_k: int, video_ids: list[str] | None,
                frame_ids: list[str] | None = None, min_score: float | None = None) -> list[dict]:
        # frame_ids=[] phải trả về rỗng (candidate list rỗng khi rerank), khác video_ids
        # nơi [] nghĩa là không lọc — nên check `is not None` thay vì truthiness.
        filters = ([{"terms": {"video_id": video_ids}}] if video_ids else []) \
                  + ([{"terms": {"frame_id": frame_ids}}] if frame_ids is not None else [])
        if filters:
            es_query = {"bool": {"must": es_query, "filter": filters}}
        kwargs = {"min_score": min_score} if min_score is not None else {}
        resp = self.es.client.search(index=self.es.index_name, query=es_query, size=top_k, **kwargs)
        return [
            {
                "keyframe_id": hit["_source"]["frame_id"],
                "video_id": hit["_source"]["video_id"],
                "shot_id": hit["_source"]["shot_id"],
                "frame_idx": hit["_source"]["frame_idx"],
                "timestamp_ms": hit["_source"]["timestamp_ms"],
                "score": hit["_score"],
            }
            for hit in resp["hits"]["hits"]
        ]

    def get_frames(self, ids: list[str]) -> list[dict]:
        docs = {d["_id"]: d for d in self.mongo.frames.find({"_id": {"$in": ids}})}
        out = []
        for i in ids:
            if i not in docs:
                continue
            doc = dict(docs[i])
            doc["keyframe_id"] = doc.pop("_id")
            doc["image_path"] = os.path.join(self.data_root, doc["image_path"])
            out.append(doc)
        return out

    def get_shots(self, video_id: str) -> list[dict]:
        return list(self.mongo.shots.find({"video_id": video_id}).sort("start_ms", 1))

    def get_transcript(self, video_id: str, start_ms: int, end_ms: int) -> list[dict]:
        query = {"video_id": video_id, "start_ms": {"$lt": end_ms}, "end_ms": {"$gt": start_ms}}
        return list(self.mongo.transcript_segments.find(query).sort("start_ms", 1))
