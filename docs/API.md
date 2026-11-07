# Research answer HTTP API

The FastAPI service is an HTTP entry point to the existing `answer_question` pipeline.
It shares MedCPT, Weaviate, reranking, evidence budgets and citation validation with
the CLI. Its OpenAPI schema describes nested answer, citation, evidence and budget fields.
There is no separate HTTP implementation of the retrieval or generation algorithms.

## Run

Use Python 3.11+ and an activated environment:

```powershell
python -m pip install -e ".[api,models,database]"
$env:RAG_LLM_MODEL = "your-installed-model"
$env:RAG_WEAVIATE_COLLECTION = "YourExistingCollection"
python -m uvicorn alzheimers_rag.api.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Set `RAG_LLM_PROVIDER`, `RAG_LLM_BASE_URL`, and `RAG_LLM_API_KEY` as described in the
README when using a different model endpoint. There is no default generation model.
Startup validates configuration, opens the existing collection, loads the query encoder,
and loads the cross encoder only if reranking is enabled. A missing collection or failed
model load aborts startup and closes the database connection. No collections are created
and no corpus/index files are changed by the service.

`create_app()` is a factory: importing or constructing the app does not connect to services
or load models. Offline tests inject a `service_factory(config)` context manager that
supplies the same `QueryService` with mocked retrieval and LLM dependencies.
Resource management uses FastAPI's documented
[lifespan mechanism](https://fastapi.tiangolo.com/advanced/events/).

## Application structure

```text
src/alzheimers_rag/api/
  __init__.py             Lazy compatibility entry point
  main.py                 App factory; includes routers and error handlers
  config.py               ServiceConfig and startup environment parsing
  lifespan.py             Shared-resource startup/shutdown ownership
  dependencies.py         get_service and the typed request dependency
  exception_handlers.py   Centralized HTTP error responses and redaction
  routers/
    health.py             Liveness and readiness endpoints
    query.py              Thin adapter to the answer service
  schemas/
    health.py             Typed health/readiness responses
    query.py              Query, answer, citation and budget models
  services/
    query.py              Model/database lifetime, capacity and pipeline invocation
```

This follows FastAPI's [multiple-file application pattern](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
using `APIRouter`. `main.py` contains application composition only. Settings and service
code use the standard library and existing core modules; schemas and routers import
their optional FastAPI/Pydantic dependencies normally. Importing the package-level
`alzheimers_rag.api` remains safe without the API extra. The earlier
`alzheimers_rag.api:create_app` launch target and `alzheimers_rag.service` Python imports
delegate to the new implementation for compatibility.

Tests and other app factories can override the public service dependency with
`app.dependency_overrides[get_service]`. Overrides and service state belong to each app
instance; no global database/model singleton is created by importing routers.

## Endpoints

| Method and path | Behavior |
| --- | --- |
| `POST /query` | Retrieve, optionally rerank, assemble evidence, generate and validate a structured answer |
| `GET /health` | Liveness, returns `{"status":"ok"}` without querying the database or LLM |
| `GET /ready` | Checks initialized resources and Weaviate readiness; returns `status`, `reranking`, and `parent_context` |
| `GET /docs` | Interactive Swagger UI; try queries from a browser |
| `GET /openapi.json` | Machine-readable request/response schema |

Readiness does not probe the LLM endpoint or generate a test answer. Provider availability
is checked by actual queries. A busy worker can still be ready and respond to health checks.

Request example:

```json
{
  "query": "What is reported about tau and cognitive decline?",
  "top_k": 5,
  "alpha": 0.5,
  "section_type": "results",
  "keyword": "tau",
  "rerank": true,
  "candidate_pool_size": 30,
  "context": "child"
}
```

Only `query` is required. Request strings are trimmed. Unknown fields and wrong JSON
types are rejected; the caller cannot choose model endpoints, credentials, server files
or collection names in a request. Limits:

| Field | Accepted values / default |
| --- | --- |
| `query` | Nonblank string, at most 4,096 characters; must also fit the server's input budget |
| `top_k` | Integer 1–50, default 5 |
| `alpha` | Finite number 0–1, default 0.5 |
| `section_type` | Optional nonblank string up to 100 characters, default null |
| `keyword` | Optional nonblank string up to 200 characters, default null |
| `rerank` | Boolean, default false; requires server capability |
| `candidate_pool_size` | Integer 1–200, default 30; must be at least `top_k` when reranking |
| `context` | `child` or `parent`, default `child`; parent mode requires server capability |

The response is the same object as CLI `query`: `question`, `status`, `answer`,
`statements`, `limitations`, `citations`, `evidence`, `budget`, and `generation`.
`citations` contain only cited source records; `evidence` includes all admitted excerpts.
Each source has stable chunk/parent identifiers, title, known PMID/PMCID/DOI and source
URL, context type, heading path and exact supplied text. See the README for grounding
limitations and identifier/URL resolution rules.

Status and failure semantics:

| HTTP status | Meaning |
| --- | --- |
| 200 | Valid structured answer, including an explicit `insufficient_evidence` response |
| 422 | Invalid request, oversized question, unavailable parent context or disabled reranking |
| 429 | This worker already has an in-flight query; `Retry-After: 1` suggests retrying later |
| 502 | Generated JSON failed the answer/citation contract; no answer is returned |
| 503 | Runtime not initialized, failed readiness, or a database/model/provider operation failed |

Errors use a `detail` field (request validation uses FastAPI's structured error list).
Raw backend exception messages, evidence, URLs and credentials are not echoed in query
errors. Operational warnings log the exception class only. An incomplete/refused provider
response is an upstream failure, not a claim that research evidence is insufficient.

## Server configuration

All settings are read once during startup. Restart the worker to change them.

| Environment variable | Default / meaning |
| --- | --- |
| `RAG_LLM_MODEL` | Required model name |
| `RAG_LLM_PROVIDER` | `ollama` or `openai-compatible`; default `ollama` |
| `RAG_LLM_BASE_URL` | Ollama defaults to `http://localhost:11434`; required for compatible endpoints |
| `RAG_LLM_API_KEY` | Optional bearer key; never part of the HTTP query body |
| `RAG_LLM_API_KEY_ENV` | Optional alternative key variable name, default `RAG_LLM_API_KEY` |
| `RAG_LLM_TIMEOUT` | LLM request timeout in seconds, default 120 |
| `RAG_CONTEXT_WINDOW` | Generation context-window accounting limit, default 8192 |
| `RAG_CONTEXT_TOKEN_BUDGET` | Evidence budget including metadata, default 4096 |
| `RAG_MAX_OUTPUT_TOKENS` | Output reservation, default 1024 |
| `RAG_SAFETY_TOKENS` | Chat-template safety margin, default 128 |
| `RAG_WEAVIATE_HOST` | `localhost` |
| `RAG_WEAVIATE_PORT` | `8080` |
| `RAG_WEAVIATE_GRPC_PORT` | `50051` |
| `RAG_WEAVIATE_COLLECTION` | `AlzheimersChunk`; must already exist |
| `RAG_DATA_DIR` | `data`; reads `processed/pmc_alzheimers_parents.jsonl` if present |
| `RAG_ENABLE_RERANKING` | `false`; accepts `true`, `false`, `1`, `0` |

The parent file must match the selected collection. If absent, child queries still work;
requesting parent mode returns 422. With a parent file, missing individual parents retain
the existing child fallback behavior. Available parent/reranking options appear in `/ready`.
Default budget counts use conservative UTF-8 bytes, as documented in the README; they
are not exact tokenizer or provider usage counts.

## Concurrency and deployment scope

Query handlers use FastAPI's [synchronous handler thread pool](https://fastapi.tiangolo.com/async/),
so blocking retrieval/model work does not run on the HTTP event loop. A nonblocking lock
admits one query per worker, protecting shared model resources. Excess queries return 429
instead of forming an unbounded inference queue. Failures release the slot. Multiple
workers each load their own models, database client and parent map; start with one worker
and budget memory before increasing that count. This is not a distributed rate limiter.

The supplied command binds to loopback. This service has no authentication, persistence
of query history, streaming or browser frontend beyond API docs. For shared deployment,
place it behind your authenticated gateway and TLS termination; CORS is not enabled by
default. A disconnected client does not forcibly cancel an in-flight synchronous model
request. The LLM timeout is not a deadline for the entire pipeline. Citation entailment
and clinical accuracy still require review; the HTTP wrapper adds no scientific validation.

## Test

```powershell
python -m pip install -e ".[api,dev]"
python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
```

HTTP tests use FastAPI's in-process client and mocked evidence/model responses; no model
downloads, sockets or database are required. They cover request bounds, complete cited
responses, insufficient evidence, error redaction, startup/shutdown ownership, model reuse
and health checks during a busy query. API tests skip if optional API/test dependencies
are absent; core generation and service tests continue to run.

For a real HTTP integration test, first run the configured service in another terminal,
then opt in explicitly:

```powershell
$env:RAG_RUN_LIVE_API_TESTS = "1"
$env:RAG_TEST_API_URL = "http://127.0.0.1:8000"
# Optional: RAG_TEST_QUESTION, RAG_TEST_RERANK="1", RAG_TEST_CONTEXT="parent"
python -m unittest discover -s tests -p test_live_generation.py -v
Remove-Item Env:RAG_RUN_LIVE_API_TESTS
```

This checks readiness, a real cited answer, admitted source references and budget bounds.
The separate CLI live test remains disabled unless `RAG_RUN_LIVE_TESTS=1`. Both paths
are read-only against the selected collection and can incur model-provider charges.
Then inspect the returned excerpts and publications to assess whether they support the
answer. Live model/database/HTTP integration was not executed during implementation.
