from indexdb.config import Config

def test_config_defaults(monkeypatch):
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.delenv("ELASTICSEARCH_URI", raising=False)
    monkeypatch.delenv("MILVUS_URI", raising=False)
    cfg = Config.from_env()
    assert cfg.mongo_uri == "mongodb://root:rootpass@localhost:27017"
    assert cfg.elastic_uri == "http://localhost:9200"
    assert cfg.milvus_uri == "http://localhost:19530"

def test_config_reads_env(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://x:9999")
    cfg = Config.from_env()
    assert cfg.mongo_uri == "mongodb://x:9999"

def test_config_data_root_default(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    cfg = Config.from_env()
    assert cfg.data_root == "/data"

def test_config_data_root_reads_env(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/mnt/ds")
    cfg = Config.from_env()
    assert cfg.data_root == "/mnt/ds"
