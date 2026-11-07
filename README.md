# Alzheimer's research RAG

A Python pipeline for acquiring PubMed abstracts or open-access PMC papers, cleaning BioC
sections, creating parent/child chunks, embedding with MedCPT, and retrieving evidence
from Weaviate. It builds evidence-grounded prompts, generates structured answers with
Ollama or a configured Chat Completions endpoint, and evaluates retrieval using human
judgments. A FastAPI service exposes the same answer pipeline over HTTP with interactive
API documentation. The `prompt` command remains available without calling an LLM.

## Layout

```text
src/alzheimers_rag/
  config.py              Model names, data paths, retrieval settings
  io.py                  JSONL readers/writers and BioC document loading
  ingestion/             PubMed search/abstracts and PMC downloads
  processing/            BioC cleaning, abstract normalization, token chunking
  embeddings.py          Article/query encoders and embedding artifacts
  storage/               Weaviate schema and validated batch indexing
  retrieval/             Hybrid search, rank fusion, cross-encoder reranking
  generation.py          Evidence budgeting, prompts, answer/citation validation
  llm.py                 Injectable chat-model contract and small HTTP adapters
  provenance.py          Literature identifiers and stable source URLs
  api/                   FastAPI app, settings, lifespan, dependencies and handlers
    routers/             Research-query and health HTTP routes
    schemas/             Query, answer, citation and health contracts
    services/            Shared query resources and capacity management
  service.py             Compatibility imports for earlier Python callers
  evaluation.py          Candidate judgment sheets and retrieval metrics
  analysis.py            Corpus statistics and optional plots
  pipeline.py            File-to-file cleaning and chunking orchestration
  cli.py                 Commands for individual pipeline stages
data/
  raw/                   Downloaded corpora (ignored by Git)
  processed/             Cleaned documents, chunks, vectors, ID maps (ignored)
  charts/                Generated figures (ignored)
  evaluation/            Versioned questions and existing human judgments
notebooks/               Research history and a package-based workflow example
tests/                   Offline regression tests
```

## Setup

Use Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[all,dev]"
alzheimers-rag setup-nltk
```

For a smaller environment install only the extras you need: `ingestion`, `models`,
`database`, `analysis`, `api`, or `dev`. The base package, cleaning, prompt formatting,
and evaluation use the standard library. Models are downloaded when an encoder,
reranker, or tokenizer is constructed. NLTK resources are downloaded only by the
explicit setup command. Install the package into your notebook kernel environment too.

## Run the PMC pipeline

All paths default to `data/` relative to the current working directory. Use the global
`--data-dir PATH` option **before** the command to select another dataset directory.
No machine-specific absolute paths are required.

```powershell
$env:NCBI_EMAIL = "you@example.org"
# Optional: $env:NCBI_API_KEY = "..."
alzheimers-rag fetch-pmc --limit 3000
alzheimers-rag clean
alzheimers-rag chunk
alzheimers-rag embed --batch-size 20
alzheimers-rag index
alzheimers-rag search "How does tau pathology contribute to cognitive decline?" --top-k 5
alzheimers-rag prompt "What causes neurofibrillary tangles?" --context parent --rerank
# Requires a separately configured LLM service; see Answer generation below.
alzheimers-rag query "What causes neurofibrillary tangles?" --model "your-installed-model" --rerank
```

`index`, `search`, `prompt`, `query`, and `pool` need a running Weaviate server with HTTP and
gRPC available (defaults: localhost ports 8080 and 50051). Each accepts `--host`,
`--port`, `--grpc-port`, and `--collection`. The package connects to the server;
it does not provision one. The default collection is `AlzheimersChunk`, with
self-provided vectors and dot-product distance for the PMC CLS embeddings.

Indexing validates row counts, exact ID-map order, unique IDs, and finite vector values
before creating or accessing the collection. It never deletes a collection. Chunk UUIDs
are deterministic, so rerunning an unchanged import targets the same objects. Import
errors fail the command. Rebuild into a new collection when the corpus, schema, or
embedding settings change; importing a changed corpus does not remove stale objects.

Each command can be run separately. `clean`, `chunk`, `embed`, and `analyze` accept
`--input`; inspect `COMMAND --help` for stage-specific options. Fetch and processing
commands overwrite their selected output files, so use separate data directories for
different corpora. The existing raw corpus can be used directly without fetching again.
JSONL outputs are replaced only after the entire write succeeds.

The supplied `data/raw/pmc_corpus.jsonl` is incomplete: line 1220 contains invalid JSON.
The other local copy validates successfully (1,319 documents). Use it explicitly:

```powershell
alzheimers-rag clean --input notebooks/data/raw/pmc_corpus.jsonl
```

## Abstract workflow

Use a separate directory to keep abstract and full-text indexes distinct:

```powershell
alzheimers-rag --data-dir data/abstracts fetch-abstracts --limit 100
alzheimers-rag --data-dir data/abstracts clean-abstracts
alzheimers-rag --data-dir data/abstracts chunk
alzheimers-rag --data-dir data/abstracts embed
alzheimers-rag --data-dir data/abstracts index --collection AlzheimersAbstractChunk
```

Abstracts are deduplicated by PMID and empty abstracts are excluded. Both corpora then
use the shared token chunking and CLS embedding pipeline. The original sentence-window
abstract chunker is also available as `processing.abstracts.chunk_abstract`. Mean pooling
and normalization remain explicit encoder options for experiments; use matching article
and query settings, and regenerate the index when changing them. Historical pickle
embeddings are retained as research artifacts and are not inputs to the new pipeline.

## Retrieval, prompts, and evaluation

Hybrid search supports `--alpha` (0 = BM25, 1 = vector), `--section-type` (canonical
section names such as `results`), and `--keyword`. `--rerank` retrieves a wider pool
(`--candidate-pool-size`, default 30) and scores it with the MedCPT cross encoder.
Parent contexts deduplicate sections using the best child score, prefer reranking
scores when present, and mark a child fallback when its parent is unavailable.
Search results retain chunk and parent identifiers for provenance.

```powershell
alzheimers-rag pool --top-k 10 --output data/evaluation/new_judgments.csv
# Fill the relevant column with 1 or 0; leave uncertain candidates blank.
alzheimers-rag evaluate data/evaluation/new_judgments.csv
alzheimers-rag analyze --output data/charts/metrics.json --chart data/charts/corpus.png
```

The existing evaluation questions and judgment sheet have been copied from the research
notebooks to `data/evaluation/`. Blank labels are reported and excluded without changing
original ranks. Recall@k measures recall within the judged candidate pool, not the full
corpus. Questions without judged relevant candidates are excluded from recall averages;
MRR includes them as zero. These metrics do not measure generated answer quality.

## Python usage

```python
from alzheimers_rag.config import DataPaths, RetrievalConfig
from alzheimers_rag.embeddings import MedCPTEncoder
from alzheimers_rag.generation import build_prompt
from alzheimers_rag.retrieval.search import HybridRetriever
from alzheimers_rag.storage.weaviate import connect

with connect() as client:
    retriever = HybridRetriever(
        client.collections.get("AlzheimersChunk"), MedCPTEncoder.for_queries()
    )
    hits = retriever.search("What causes neurofibrillary tangles?", RetrievalConfig())
    result = build_prompt("What causes neurofibrillary tangles?", hits)
    print(result["prompt"])
```

## Answer generation

`query` runs retrieval, optional existing MedCPT reranking, evidence assembly, and one
LLM request. It prints a JSON answer to stdout and does not write corpus files or indexes.
Generation uses only the standard library; the existing `models` and `database` extras
are needed for live retrieval. Install and serve the generation model separately.

For Ollama, replace the model name below with one already installed in your server:

```powershell
$env:RAG_LLM_MODEL = "your-installed-model"
alzheimers-rag query "What is reported about tau and cognitive decline?"
alzheimers-rag query "What is reported about tau and cognitive decline?" --rerank --context parent
```

For a Chat Completions service, configure its exact model and base URL, including `/v1`
when your server requires it. A remote provider receives the question and admitted
evidence. Set a key only if your endpoint requires authentication:

```powershell
$env:RAG_LLM_PROVIDER = "openai-compatible"
$env:RAG_LLM_MODEL = "your-served-model"
$env:RAG_LLM_BASE_URL = "http://localhost:8000/v1"
# Set RAG_LLM_API_KEY securely in your environment if required by your service.
alzheimers-rag query "What evidence links tau to cognitive decline?"
```

The adapters use the documented [Ollama chat API](https://docs.ollama.com/api/chat)
and [Chat Completions request/response format](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
The compatible endpoint must accept `response_format: {"type": "json_object"}` and
`max_completion_tokens`. This does not cover every server calling itself compatible,
Azure-specific authentication, or every reasoning model. There is no automatic API or
model fallback. Missing, refused, truncated, malformed, or invalidly cited responses fail
the command with a nonzero exit code. Requests have a timeout and are not retried.

| Setting | CLI option | Default / environment |
| --- | --- | --- |
| Provider | `--provider` | `RAG_LLM_PROVIDER`, otherwise `ollama` |
| Generation model | `--model` | `RAG_LLM_MODEL`; required, no model chosen automatically |
| Base URL | `--base-url` | `RAG_LLM_BASE_URL`; Ollama defaults to `http://localhost:11434`; required for compatible endpoints |
| Key variable name | `--api-key-env` | Reads `RAG_LLM_API_KEY`; no key value in CLI arguments |
| Model context window | `--context-window` | `8192`; set to the model/server's supported window |
| Evidence budget | `--context-token-budget` | `4096`, including evidence metadata |
| Output reservation | `--max-output-tokens` | `1024` |
| Chat-template margin | `--safety-tokens` | `128` |
| HTTP timeout | `--timeout` | `120` seconds |

All existing retrieval options also apply: `--top-k`, `--alpha`, `--section-type`,
`--keyword`, `--rerank`, `--candidate-pool-size`, and database connection options.
With reranking, the candidate pool must be at least `--top-k`.

Evidence is admitted in rank order as whole excerpts. Oversized excerpts are skipped;
later smaller excerpts can still fit. Parent mode deduplicates by parent, retains all
retrieved child IDs for that parent, and uses the best child score. A missing parent
falls back to its best child with `parent_missing: true`. An existing oversized parent
is skipped rather than silently replaced or truncated. Increase the budget or use child
mode if parent sections do not fit.

The default counter uses **one UTF-8 byte per accounting unit**, a deliberately
conservative approximation for common byte-based tokenizers. `budget.counter` identifies
it as `byte_token_count`; these counts are not exact provider token usage. Both the
evidence JSON and complete serialized messages are checked. The input limit is the
configured context window minus output reservation and safety margin. Chat templates
vary, so configure the correct model window and margin. Python callers can inject a
model-specific `count_tokens(text)` function; MedCPT's retrieval tokenizer should not be
used as the generation tokenizer.

Output contains:

- `status`: `answered` or `insufficient_evidence`.
- `statements`: model-written finding text and validated `source_ids`.
- `answer`: the statements rendered with application-generated `[S1]` citations.
- `citations`: cited source records, resolved locally from admitted evidence.
- `evidence`: all admitted source records and their exact text, including uncited sources.
- `limitations`, `budget`, and configured provider/model identifiers under `generation`.

Each source record includes `source_id`, `chunk_ids`, `parent_id`, `doc_id`, title,
PMID/PMCID/DOI where known, heading path, and `source_url`. URL preference is PMC,
PubMed, then DOI. Unknown identifiers/links remain null. Legacy `PMC…` and `PMID…`
document IDs can recover the corresponding identifier; arbitrary numeric document IDs
are not guessed. Newly indexed collections store the literature metadata explicitly.
Old indexes are not migrated automatically: use existing child/parent artifacts and a
**new collection name** for a metadata-complete import. No index was rebuilt for this change.

Every answer statement must reference admitted source IDs. Unknown IDs, missing
citations, extra schema fields, and model-written citation markers or HTTP(S) URLs in
statement/limitation text are rejected. No evidence, empty excerpts, or no excerpt that
fits produces an explicit insufficient-evidence result without calling the LLM. The
model can also report that the admitted evidence is insufficient. A provider failure
or invalid answer is an error, not evidence insufficiency.

Evidence text and metadata are serialized into a user-role JSON object, separate from
system instructions. The system policy treats all evidence as untrusted data and tells
the model to ignore instructions found there. No tools or agent actions are exposed to
the model. This separation and the citation checks do **not** guarantee prompt-injection
resistance, semantic grounding, source-link correctness, or clinical accuracy. A valid
citation ID can still accompany an unsupported statement. Human review is needed to
assess entailment, conflicting results, study quality, and answer completeness. Existing
retrieval metrics do not evaluate generated answers, and there is no calibrated
evidence-relevance threshold. This is a research tool, not a validated clinical system.

Generation is injectable for notebooks and offline callers:

```python
from alzheimers_rag.config import LLMConfig
from alzheimers_rag.embeddings import MedCPTEncoder
from alzheimers_rag.generation import answer_question
from alzheimers_rag.llm import HTTPChatModel
from alzheimers_rag.retrieval.search import HybridRetriever
from alzheimers_rag.storage.weaviate import connect

with connect() as client:
    retriever = HybridRetriever(
        client.collections.get("AlzheimersChunk"), MedCPTEncoder.for_queries()
    )
    result = answer_question(
        "What is reported about tau and cognitive decline?",
        retriever,
        HTTPChatModel(),  # Tests inject complete(messages, *, config) -> JSON str.
        config=LLMConfig(model="your-installed-model"),
        # Optional: parents=parent_lookup, reranker=MedCPTReranker(), count_tokens=model_counter
    )
print(result["answer"])
```

## FastAPI service

Install the API extra alongside the retrieval dependencies and configure an existing
Weaviate collection and generation model:

```powershell
python -m pip install -e ".[api,models,database]"
$env:RAG_LLM_MODEL = "your-installed-model"
$env:RAG_WEAVIATE_COLLECTION = "YourExistingCollection"
# Optional: $env:RAG_ENABLE_RERANKING = "true"
python -m uvicorn alzheimers_rag.api.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Visit `http://127.0.0.1:8000/docs` to try the API interactively. `POST /query` accepts
a question and retrieval options, returning the same structured answer, evidence and
citations as the CLI. `/health` reports liveness; `/ready` checks initialization/database
readiness and reports available parent/reranking capabilities. Models and the database
client are loaded once per worker and released on shutdown. The service opens existing
collections and performs no ingestion or indexing.

```powershell
$body = @{ query = "What is reported about tau and cognitive decline?"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -ContentType application/json -Body $body
```

See [the HTTP API guide](docs/API.md) for the request contract, server environment
variables, error codes, concurrency behavior, deployment limits and live HTTP testing.
The application is organized into separate configuration, schemas, routers, dependencies,
exception handlers, lifecycle and service modules. The earlier `alzheimers_rag.api:create_app`
entry point remains available for existing launch commands.

## Validation

```powershell
python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
```

Tests use an injected tokenizer and synthetic documents to exercise cleaning, chunk
limits and overlap, schema consistency, context provenance, rank fusion, metrics, and
the CLI without downloading models or connecting to services. Generation tests use a
mocked LLM and fake collection to exercise the actual hybrid retrieval code, optional
reranking, budget boundaries, untrusted evidence separation, citation resolution, provider
requests/errors, and prompt-only compatibility. NumPy artifact tests skip if NumPy is absent.

Verified on 2026-11-07 with Python 3.13.14 and Ruff 0.17.4: baseline 17/17 tests passed;
after the generation increment 39 offline tests passed and one live test was skipped.
After the modular API refactor, 59 offline tests passed and two live tests were skipped.
Ruff lint and formatting checks pass. With site-packages disabled, 45 tests passed and
16 optional/live tests skipped, verifying operation without optional packages.
Python 3.11 is the supported minimum but was not separately
executed in this environment. See [the audit record](docs/BASELINE_AUDIT.md) and
[the proposed PR summary](docs/PR_SUMMARY.md).

## Live integration procedure

1. Install `.[models,database,dev]` into Python 3.11+ and ensure MedCPT query weights
   are available. If testing reranking, make the cross-encoder weights available too.
2. Start your existing Weaviate server with HTTP and gRPC available. Select an existing
   collection built with the matching MedCPT article encoder, CLS pooling and vector
   settings. The live test never creates, indexes, or deletes a collection.
3. Start the LLM service with an available model that supports JSON output. Set provider,
   model, base URL and credentials as above. These requests may incur provider charges.
4. Set the test collection and opt in explicitly:

```powershell
$env:RAG_TEST_COLLECTION = "YourExistingCollection"
$env:RAG_RUN_LIVE_TESTS = "1"
# Optional: RAG_TEST_HOST, RAG_TEST_PORT, RAG_TEST_GRPC_PORT
# Optional: RAG_TEST_DATA_DIR, RAG_TEST_CONTEXT="parent" (requires matching parents JSONL)
# Optional: RAG_TEST_QUESTION (choose a question supported by the selected corpus)
# Optional: RAG_TEST_RERANK="1"
python -m unittest discover -s tests -p test_live_generation.py -v
Remove-Item Env:RAG_RUN_LIVE_TESTS
```

The test requires an answered response, nonempty citations with chunk IDs and evidence
text, source IDs drawn from admitted evidence, and both budget bounds. Failure to answer
the question is surfaced for investigation; service errors are not silently skipped once
enabled. Run `query` for the same question to inspect its JSON, and manually check that
each statement is supported by the cited text and publication. A passing smoke test does
not establish citation entailment or clinical accuracy. Live services were not exercised
during this implementation; no generation model or endpoint was supplied for validation.
