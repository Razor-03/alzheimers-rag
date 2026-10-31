"""The earlier PubMed abstract workflow, producing the shared document schema."""

from .chunking import split_sentences


def clean_abstracts(records):
    seen = set()
    for record in records:
        pmid = str(record["pmid"])
        text = record.get("abstract")
        if pmid in seen or not isinstance(text, str) or not text.strip():
            continue
        seen.add(pmid)
        yield {**record, "pmid": pmid, "abstract": text.strip()}


def chunk_abstract(text, window_size=8, overlap=2, *, sentence_splitter=split_sentences):
    if window_size < 1 or not 0 <= overlap < window_size:
        raise ValueError("Require 0 <= overlap < window_size")
    sentences = sentence_splitter(text)
    return [
        " ".join(sentences[i : i + window_size])
        for i in range(0, len(sentences), window_size - overlap)
    ]


def abstract_documents(records):
    """Allow abstract corpora to use the same chunk/embed/index commands as PMC."""
    for record in clean_abstracts(records):
        doc_id = f"PMID{record['pmid']}"
        yield {
            "doc_id": doc_id,
            "pmid": record["pmid"],
            "pmc_id": None,
            "doi": None,
            "year": record.get("year"),
            "kwd": [],
            "title": record["title"],
            "n_sections": 1,
            "n_synthetic_sections": 0,
            "sections": [
                {
                    "section_id": f"{doc_id}_s0",
                    "heading_path": ["Abstract"],
                    "section_type": "ABSTRACT",
                    "canonical_section_type": "abstract",
                    "text": record["abstract"],
                    "n_words": len(record["abstract"].split()),
                    "synthetic": False,
                }
            ],
        }
