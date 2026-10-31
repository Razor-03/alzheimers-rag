# Research notebooks

The original notebooks are retained as research history, including their saved outputs.
They contain old absolute paths, duplicate definitions, and experiments; use the package
and CLI for repeatable runs. `workflow.ipynb` is the maintained example: it imports the
package rather than defining pipeline algorithms in cells.

The main migrations are:

| Research notebooks | Package module |
| --- | --- |
| `pubmed/get_pubmed_abstracts`, `pmc/get_pubmed_central` | `ingestion.pubmed`, `ingestion.pmc` |
| `pmc/corpus_eda` | `processing.cleaning`, `analysis` |
| `create_chunks`, `pmc/create_chunks` | `processing.abstracts`, `processing.chunking` |
| `embeddings`, `pmc/embeddings` | `embeddings` |
| `init_db`, `pmc/init_db` | `storage.weaviate` |
| `retrieval`, `pmc/retrieval`, utility files | `retrieval.search`, `retrieval.reranking` |
| `pmc/prompting`, `pmc/generation` | `generation`, `evaluation` |
| `corpus_analysis`, `pmc/*analysis`, `pmc/question_texts` | `analysis`, JSONL readers, exploratory notebooks |

New experiments should import these modules. Evaluation inputs now also live in
`data/evaluation/`; keep future human labels there. Corpus files under `notebooks/data/`
are left in place; pass their path using `clean --input ...` when needed.
