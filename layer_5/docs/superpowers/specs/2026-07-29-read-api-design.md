# Read API — Design Spec

**Goal:** Expose the existing `indexdb.read.Reader` over HTTP so the algorithm team (KIS/AVS/VQA) can query Mongo/Milvus/Elasticsearch without installing `pymongo`/`pymilvus`/`elasticsearch` client libraries or knowing DB credentials/URIs — plain HTTP from any machine reachable on the VPN.

**Architecture:** A thin FastAPI app (`layer_5/api/main.py`) that constructs one `Reader` instance at startup and maps each of its 5 methods to one REST endpoint. No new business logic, no ranking/fusion, no vector encoding — those stay the algorithm team's responsibility, unchanged from `Reader`'s existing contract. Runs as a new `api` service in the existing `docker-compose.yaml` stack, built from the same image as the `app` service, with its port published to the host (like the DB ports already are) so it's reachable over VPN without SSH tunneling or joining the `milvus` Docker network.

**Tech Stack:** FastAPI + Uvicorn (added to `requirements.txt`), reusing the existing `Dockerfile`/`layer5` image.

## Global Constraints

- `api/main.py` is a consumer of `indexdb.read.Reader`, not part of `indexdb/` — it must not add read/write logic that belongs in `indexdb/`.
- The `Reader` instance is constructed once at process startup (module-level or FastAPI lifespan), not per-request — `Reader.__init__` does an ES/Milvus existence check that should not repeat on every call.
- No vector/text encoding happens in this service. `search_vector` accepts a raw `float[]` in the request body; the caller already encoded it.
- `Hit` shape returned by `search_vector`/`search_text` is unchanged from `Reader`: `{keyframe_id, video_id, shot_id, timestamp_ms, score}` — the API must not reshape it.
- Auth: a single static API key, read from env var `API_KEY`, required via `X-API-Key` header on every endpoint except `/health` and `/docs`. Missing/incorrect key → `401`.
- Docker build context stays `.` (`layer_5/`) — no Dockerfile path needs to change for this feature.
- Host port for the new service: `8020` (ports `8000`, `8010`, `8082` are already used by other services on this shared host — see `docker-compose.override.yaml`).

---

## Endpoints

All endpoints require header `X-API-Key: <value of env API_KEY>` except `/health`.

### `POST /search/vector`

Request body:
```json
{
  "model": "beit3",
  "vector": [0.01, 0.02, "... float[]"],
  "top_k": 100,
  "video_ids": null
}
```
- `model`: required, `str` — passed straight through to `Reader.search_vector(model, ...)`.
- `vector`: required, `list[float]`.
- `top_k`: optional, `int`, default `100`.
- `video_ids`: optional, `list[str] | null`, default `null` (no filter).

Response: `200`, JSON array of `Hit` (`{keyframe_id, video_id, shot_id, timestamp_ms, score}`).

### `POST /search/text`

Request body:
```json
{
  "query": "một khu chợ hoa",
  "top_k": 100,
  "video_ids": null,
  "fields": null
}
```
- `query`: required, `str`.
- `top_k`: optional, `int`, default `100`.
- `video_ids`: optional, `list[str] | null`, default `null`.
- `fields`: optional, `list[str] | null`, default `null` (Reader's own default field set).

Response: `200`, JSON array of `Hit`.

### `POST /keyframes`

Request body:
```json
{ "ids": ["kf_001", "kf_002"] }
```
- `ids`: required, `list[str]`. Order and duplicates in the response mirror `Reader.get_keyframes` (order/dup-preserving, silently skips ids with no matching document).

Response: `200`, JSON array of keyframe documents (as returned by `Reader.get_keyframes`, including absolutized `image_path`).

### `GET /shots/{video_id}`

Path param: `video_id: str`.

Response: `200`, JSON array of shot documents (as returned by `Reader.get_shots`, sorted by `start_ms`).

### `GET /transcript/{video_id}`

Path param: `video_id: str`. Query params: `start_ms: int`, `end_ms: int` (both required).

Response: `200`, JSON array of transcript segment documents (as returned by `Reader.get_transcript`).

### `GET /health`

No auth required. Returns `200 {"status": "ok"}` if the process is up. Does not re-check ES/Milvus (that already happened once at startup via `Reader.__init__`).

## Error Handling

- Missing/incorrect `X-API-Key` → `401 {"detail": "invalid or missing API key"}`.
- Request body/query fails Pydantic validation (e.g. `vector` not a list of floats, missing required field) → FastAPI's default `422` with field-level detail. No custom handling needed.
- Any exception raised inside a `Reader` call (e.g. Mongo/ES/Milvus unreachable mid-request) propagates as FastAPI's default `500` — this service does not add retry/fallback logic; the underlying stores' availability is out of scope here (already covered by `Reader.__init__`'s startup check).

## Deployment

- `requirements.txt`: add `fastapi`, `uvicorn`.
- `docker-compose.yaml`: add a new service `api`, reusing `build: {context: ., dockerfile: Dockerfile}` (same image as `app`), `command: ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]`, same `MONGO_URI`/`ELASTICSEARCH_URI`/`MILVUS_URI` env as `app`, plus `API_KEY` env, `depends_on` the three stores.
- `docker-compose.override.yaml`: publish `"8020:8000"` for the `api` service (following the existing pattern for host-port remaps on this shared host).
- Consumers reach it at `http://<server-ip>:8020` over VPN, no Docker network join needed — same reachability model as the existing Mongo/ES/Milvus published ports.

## Testing

`tests/test_api.py`, following existing conventions (`@pytest.mark.integration`, using `_test`-suffixed store names like `tests/test_read.py`):
- `fastapi.testclient.TestClient(app)` against a `Reader` instance pointed at test stores (same fixture pattern as `test_read.py` — construct via `Reader.__new__(Reader)` or a dependency override, whichever proves simpler when writing the test).
- One test per endpoint verifying it calls through to the correct `Reader` behavior and returns the expected shape.
- One test verifying `401` on missing `X-API-Key` and on a wrong key.
- One test verifying `/health` requires no auth.
