# Alzheimer's research RAG

A Python pipeline for acquiring PubMed abstracts or open-access PMC papers, cleaning BioC
sections, creating parent/child chunks, embedding with MedCPT, and retrieving evidence
from Weaviate. It builds evidence-grounded prompts and evaluates retrieval using human
judgments. Prompt construction does not call an LLM or generate an answer.

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
  generation.py          Child/parent context assembly and prompt construction
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
`database`, `analysis`, or `dev`. The base package, cleaning, prompt formatting,
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
```

`index`, `search`, `prompt`, and `pool` need a running Weaviate server with HTTP and
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

## Validation

```powershell
python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
```

Tests use an injected tokenizer and synthetic documents to exercise cleaning, chunk
limits and overlap, schema consistency, context provenance, rank fusion, metrics, and
the CLI without downloading models or connecting to services. Full embedding and
Weaviate integration require the optional dependencies, model weights, and a running server.
