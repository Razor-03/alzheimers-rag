# Proposed PR: Add research query API and CLI with validated evidence citations

The package currently stops at a retrieval prompt and has no HTTP interface. This change
adds FastAPI `POST /query` and `alzheimers-rag query` to retrieve evidence from Weaviate,
optionally rerank with the existing MedCPT cross encoder, assemble a bounded context,
and generate structured answers through Ollama or an explicitly configured Chat
Completions service. Interactive API docs, health/readiness endpoints and typed
request/response schemas make the pipeline usable by client applications. The existing
`prompt` workflow continues without an LLM request.

Answers contain cited findings, limitations, exact admitted evidence and a locally
resolved source map carrying chunk/parent IDs, literature identifiers, titles and stable
links where known. Missing evidence returns `insufficient_evidence`; invalid schema,
unknown source IDs, incomplete model responses and service failures surface as errors.
Evidence is serialized separately from system policy and treated as untrusted data.

Changes:

- Add validated model/provider/budget configuration and an injectable chat-model contract.
- Add standard-library LLM adapters without agent frameworks.
- Add optional FastAPI/Pydantic/Uvicorn dependencies and HTTP test clients in `dev`.
- Organize the HTTP app into factory, configuration, lifecycle, dependencies, exception
  handlers, routers, schemas and services; retain the original launch/import entry points.
- Share database/models across requests with startup/shutdown ownership, bounded
  inference capacity and sanitized HTTP errors; never create or index collections in the API.
- Extend storage metadata for newly created collections; preserve existing indexes/data.
- Add whole-excerpt context budgeting with output reservation and injectable token counting.
- Preserve prompt helper contracts, MedCPT encoders, hybrid retrieval and reranking.
- Document configuration, limitations, baseline findings and read-only live testing.

Validation: original baseline 17 tests passed; generation increment 39 offline tests
passed. Final suite: 59 offline tests passed and two live integration tests skipped by
default. Ruff lint and formatting pass. Tested with Python 3.13.14; minimum supported
Python remains 3.11. With site-packages disabled, 45 tests passed and 16 optional/live
tests skipped. Live services were not exercised. HTTP tests cover successful answers,
insufficient evidence, validation, errors/redaction, parent/rerank options, readiness,
startup/shutdown, model reuse and health responsiveness during an in-flight query.
Modular-app coverage also checks public dependency overrides, isolation between app
instances and compatibility imports.

Review limits: citation-ID validation does not prove that statements follow from cited
text. No clinical accuracy, universal endpoint compatibility or prompt-injection immunity
is claimed. Default budgeting uses conservative UTF-8 byte counts, not provider billing
tokens. Old collection metadata is not automatically backfilled; use a new collection
for imports requiring the additional metadata. Large parent excerpts can be skipped.
The API is a synchronous research backend with one in-flight query per worker, no
built-in authentication, no persistent history and no frontend beyond Swagger docs.
Server configuration controls the model/index; callers can only set retrieval options.

Branch: `feat/end-to-end-answer-generation`. No commits or PR created by the assistant;
this is a proposed title/body for the repository owner.
