from indexdb.config import Config
from indexdb.mongo import MongoStore
from indexdb.elastic import ElasticStore
from indexdb.milvus import MilvusStore

def init_all(cfg: Config | None = None):
    cfg = cfg or Config.from_env()
    MongoStore(cfg).ensure_indexes()
    ElasticStore(cfg).create_index()
    MilvusStore(cfg).create_collections()
    print("Initialized index/collection for Mongo + Elastic + Milvus.")

if __name__ == "__main__":
    init_all()
