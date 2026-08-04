from pymongo import MongoClient, ASCENDING
from indexdb.config import Config

class MongoStore:
    def __init__(self, cfg: Config, db_name: str | None = None):
        self._client = MongoClient(cfg.mongo_uri)
        self.db = self._client[db_name or cfg.mongo_db]

    @property
    def videos(self):              return self.db["videos"]
    @property
    def shots(self):               return self.db["shots"]
    @property
    def frames(self):              return self.db["frames"]
    @property
    def transcript_segments(self): return self.db["transcript_segments"]
    @property
    def ingest_status(self):       return self.db["ingest_status"]

    def ensure_indexes(self):
        self.shots.create_index([("video_id", ASCENDING)])
        self.frames.create_index([("video_id", ASCENDING)])
        self.frames.create_index([("video_id", ASCENDING), ("shot_id", ASCENDING)])
        self.transcript_segments.create_index([("video_id", ASCENDING)])
