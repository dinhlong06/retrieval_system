from elasticsearch.helpers import bulk
from indexdb.mongo import MongoStore
from indexdb.elastic import ElasticStore
from indexdb.builders import build_es_doc

class ElasticIndexer:
    def __init__(self, store: MongoStore, es: ElasticStore):
        self.store = store
        self.es = es

    def _shot_transcript(self, shot_id: str) -> str:
        shot = self.store.shots.find_one({"_id": shot_id})
        return shot.get("transcript", "") if shot else ""

    def index_video(self, video_id: str) -> int:
        actions = []
        frame_ids = []
        for frame in self.store.frames.find({"video_id": video_id}):
            transcript = self._shot_transcript(frame["shot_id"])
            src = build_es_doc(frame, transcript=transcript)
            actions.append({
                "_index": self.es.index_name,
                "_id": frame["_id"],
                "_source": src,
            })
            frame_ids.append(frame["_id"])
        if not actions:
            return 0
        bulk(self.es.client, actions)
        self.store.frames.update_many(
            {"_id": {"$in": frame_ids}}, {"$set": {"synced.elastic": True}}
        )
        return len(actions)
