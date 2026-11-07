# Baseline audit and first increment

Date: 2026-11-07 (Asia/Karachi). Repository: `Razor-03/alzheimers-rag`.
Starting branch: `master`; starting HEAD: `aeb7c92` (`add end-to-end workflow`).
Working branch: `feat/end-to-end-answer-generation`. Starting worktree was clean.
No commits or PRs were created.

## Inspection and retained contracts

Inspected all package modules and existing tests, packaging/configuration files,
README files, notebook utility modules, and code-cell structure across every research
notebook. Read-only validation also examined both local raw JSONL corpora. Notebook
saved outputs and pickle artifacts are historical research data, not executable inputs
to this change; they were not rewritten, unpickled or rerun.

| Area | Verified baseline and scope |
| --- | --- |
| Ingestion | PubMed Entrez client, PMID-to-PMC conversion, BioC downloads; network calls lazy; contact email explicit. Retained. |
| Cleaning | BioC section reconstruction/classification; PMID, PMC ID, DOI, title extracted. Retained. |
| Chunking | Parent/child dictionaries, deterministic IDs, sentence/token windows; identifiers already preserved. Retained. |
| Embeddings | MedCPT article/query encoders, CLS default, lazy optional imports; artifact alignment checks. Retained. |
| Weaviate | Self-provided vectors, dot distance, deterministic UUIDs; validates imports and never deletes collections. Extended metadata only. |
| Retrieval | `search(query, RetrievalConfig)` returns property dictionaries with scores; hybrid filters and rank fusion. Retained. |
| Reranking | `rerank(query, hits, RerankConfig)` adds cross-encoder scores; retained without model changes. |
| Generation | `build_prompt`, child/parent helpers; best-child parent ranking and missing-parent fallback. Retained alongside the new answer path. |
| Evaluation | Human judgment pools, unjudged-label handling, recall within candidate pool and MRR. Retained; no generated-answer quality metric claimed. |
| CLI/tests | Stage commands, standard-library unittest suite, Ruff config. Added `query` and tests using the same tools. |
| Notebooks | Historical experiments include duplicate configurations, import-time model/database actions and journal-based rank-fusion identity in `notebooks/utils.py`. Package workflow already avoids these; notebooks preserved. |

## Failures recorded before source changes

1. `python --version` / `python -m unittest ...` could not execute the WindowsApps
   alias inside the sandbox. `.venv/Scripts/python.exe` also required execution
   outside the sandbox. These were environment failures, not failed assertions.
2. The first executable test attempt failed to import both test modules because
   the local package was not installed (`ModuleNotFoundError: alzheimers_rag`).
   Adding `src` to `PYTHONPATH` established a passing 17-test baseline.
3. `ruff` was absent from PATH and the venv. Installed the existing declared `dev`
   extra and local package with `uv pip install --python .venv/Scripts/python.exe -e '.[dev]'`.
   No project dependency declarations were changed. Sandbox access to the default
   uv cache initially failed; installation succeeded with approved execution.

After environment setup, **17 tests passed**, `ruff check src tests` passed, and
`ruff format --check src tests` reported all **24 files** already formatted. No
pre-existing code test/lint failures remained. Runtime: Python 3.13.14; Ruff 0.17.4.

## Verified defects and gaps presented before implementation

- Package generation and PMC generation notebooks stop at context/prompts; the older
  abstract retrieval notebook has an isolated Ollama experiment, not a package contract.
- Storage properties/schema drop PMID, PMC ID and DOI already present in child files.
- Numbered prompt sources lack structured citation records and local identifier/URL resolution.
- Prompt assembly has no generation token budget, answer schema, model injection or
  explicit untrusted-evidence policy. No generation tests existed.
- `data/raw/pmc_corpus.jsonl` fails JSON decoding at line **1220** after 1,219 valid
  rows. `notebooks/data/raw/pmc_corpus.jsonl` validates with **1,319** rows. Neither
  file was modified. The 10 versioned evaluation questions and existing judgments
  were preserved.

## Generation increment decisions

The audit and proposed files were presented before implementation. No separate design
approval was needed within the requested scope. No new frameworks or dependencies.

- Frozen `LLMConfig`, a single injectable chat-model protocol, standard-library HTTP
  adapters for Ollama and explicitly configured Chat Completions endpoints.
- `answer_question` composes the existing retriever/reranker interfaces; `generate_answer`
  can also consume already retrieved hits. Existing prompt functions/CLI remain available.
- Whole-excerpt budgeting includes source metadata and serialized prompt overhead, with
  output reservation and a chat-template margin. A conservative byte counter is the
  default; exact model tokenization is injectable. Oversized excerpts are skipped.
- Evidence and metadata stay in user-role JSON, separated from system policy. No model
  tools are exposed. This is a mitigated boundary, not a prompt-injection guarantee.
- Model output is validated locally. Each finding requires admitted source IDs; unknown
  IDs and model-written citation markers/HTTP(S) URLs fail. Application code supplies
  rendered citations and stable links. No excerpts means no LLM call and explicit
  insufficient evidence. Invalid output/provider failures remain errors.
- Existing chunk/parent IDs and literature metadata are carried into citation records.
  New collections include metadata fields. Existing collections are not migrated,
  regenerated or deleted; full metadata may require a separate named collection.

## Validation and limitations

Commands (inside the activated venv, or invoke `.venv/Scripts/python.exe` and Ruff directly):

```powershell
python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
```

Final result: **40 tests discovered; 39 passed; 1 opt-in live test skipped**. Ruff
lint and formatting pass. New coverage exercises the actual hybrid retrieval code
with a fake Weaviate collection, mocked reranker/LLM, parent context, all evidence
identifiers, budget boundaries, invalid schema/citations, HTTP payload/errors and CLI
prompt-only compatibility. Tests make no model downloads or external service requests.

Also ran the suite with `PYTHONPATH=src` and Python `-S` (site-packages disabled):
**36 passed, 4 skipped** (three optional NumPy artifact checks and the live test).
This verifies that package imports and the new generation path need no optional model,
database, HTTP-client or SDK packages for offline operation. Ruff checked **28 Python
files** after the addition; `git diff --check` passed.

Live generation and database/model integration were not run: no generation endpoint/model
was supplied for verification. The README contains the opt-in, read-only procedure.
Python 3.11 compatibility is targeted by packaging/lint; execution here used 3.13.14.
Provider/model JSON support and chat templates vary. Budget counts are not billing counts.
Structural citation validation cannot establish entailment, clinical accuracy, correct
source metadata, or resistance to every document instruction attack. Human review is
required. No datasets, judgments, pickle artifacts or indexes were overwritten.

## FastAPI increment

The subsequent user request explicitly authorized a FastAPI service. Rechecked the
generation baseline before implementing: **39 offline tests passed, 1 live test skipped**;
Ruff lint/format checks passed. The original generation code and CLI contracts were
reused without changes for HTTP execution.

Added `service.py` for startup environment configuration and process-scoped resources,
and `api.py` for a lazy FastAPI factory with `POST /query`, `/health`, `/ready`, `/docs`
and OpenAPI. Models/client are loaded once per worker and closed on shutdown. Startup
only opens existing collections; missing collections fail without creating/indexing them.
Optional `api` dependencies are FastAPI, Pydantic 2 and Uvicorn. The `dev` extra adds HTTP
test clients; current Starlette prefers HTTPX2, with HTTPX retained for older compatible
FastAPI/Starlette releases. The first run emitted Starlette's HTTPX deprecation warning;
adding its documented preferred test client removed the warning. This is test tooling,
not a new LLM transport dependency.

Requests have strict field/type/size bounds and cannot override provider credentials,
server files or collection/model settings. Query handlers run off the HTTP event loop;
one query per worker is admitted, excess work returns 429. Model-output validation
failures return 502; backend failures return sanitized 503 responses. Insufficient
evidence remains a successful 200 response with explicit status. Readiness checks
initialized resources and Weaviate, not LLM availability or scientific correctness.

Final API increment result: **58 discovered, 56 passed, 2 live tests skipped**. With
site-packages disabled: **44 passed, 14 optional/live tests skipped**. Ruff checks pass
for **32 Python files**. API/lifecycle tests exercise the shared pipeline with synthetic
evidence and mocked LLMs; no live models, databases or HTTP server were needed. The
README and [API guide](API.md) document startup, requests, environment variables,
deployment limits and opt-in live HTTP validation. No commits or PRs created.

## Modular FastAPI application refactor

The user requested a conventional application layout rather than nested schemas/routes
inside a single file. Before refactoring: **56 passed, 2 live tests skipped**; lint and
formatting passed. Replaced `api.py` with `api/` containing a small `main.py` factory,
`config.py`, `lifespan.py`, `dependencies.py`, centralized `exception_handlers.py`, and
`routers/`, `schemas/`, `services/` packages. The runtime and settings were moved from
`service.py`; that module now re-exports the same classes/functions for compatibility.

HTTP paths, query/answer fields, error status codes, startup/shutdown, model reuse,
untrusted-evidence treatment and the core RAG pipeline remain intact. The original
`alzheimers_rag.api:create_app` launch command delegates to the new factory; the documented
canonical target is `alzheimers_rag.api.main:create_app --factory`. Health response models
are now explicit too. No new dependencies, model/database calls or data/index changes.

Web schemas and routers use normal module-level FastAPI/Pydantic imports. The public API
package, configuration, lifecycle and runtime remain importable without those extras.
The package-import regression test imports all modules when the API extra is installed
and excludes only web implementation modules when it is absent.

Result: **61 discovered, 59 passed, 2 live tests skipped**. With site-packages disabled:
**45 passed, 16 optional/live tests skipped**. Ruff passes for **45 Python files**.
New tests verify that request dependencies can be overridden, multiple app instances
keep resource state/overrides separate, and prior service imports resolve to the same
implementation. Live services remain unverified. No commits or PRs created.
