import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    mongo_uri: str
    elastic_uri: str
    milvus_uri: str
    mongo_db: str = "videoindex"
    data_root: str = "/data"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            mongo_uri=os.getenv("MONGO_URI", "mongodb://root:rootpass@localhost:27017"),
            elastic_uri=os.getenv("ELASTICSEARCH_URI", "http://localhost:9200"),
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            data_root=os.getenv("DATA_ROOT", "/data"),
        )
