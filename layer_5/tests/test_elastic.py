import pytest
from indexdb.elastic import ElasticStore

@pytest.fixture
def es(cfg):
    store = ElasticStore(cfg, index_name="frame_text_test")
    store.drop_index()
    yield store
    store.drop_index()

@pytest.mark.integration
def test_create_index_mapping(es):
    es.create_index()
    mapping = es.client.indices.get_mapping(index="frame_text_test")
    props = mapping["frame_text_test"]["mappings"]["properties"]
    assert props["ocr_text"]["analyzer"] == "vi_analyzer"
    assert props["ocr_text"]["copy_to"] == ["content_all"]
    assert props["object_tags"]["type"] == "keyword"
    assert props["frame_id"]["type"] == "keyword"
