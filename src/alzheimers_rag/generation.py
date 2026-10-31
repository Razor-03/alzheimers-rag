"""Evidence context and prompts; no LLM request is made by this module."""

SYSTEM_PROMPT = """You are a biomedical assistant synthesizing Alzheimer's disease research.
Answer only using the provided evidence. Explain how findings relate to the question.
Cite empirical claims using [Source N]. Highlight conflicting findings.
If evidence is insufficient, say so explicitly. Do not fabricate claims or citations."""


def _block(record, number, *, missing=False):
    path = record.get("heading_path") or record.get("section_type", "")
    if isinstance(path, list):
        path = " > ".join(path)
    title = record.get("doc_title") or record.get("title") or ""
    marker = " [parent unavailable; child excerpt]" if missing else ""
    return f"[Source {number}] {title} | {path}{marker}\n{record['text']}"


def assemble_child_context(hits):
    return "\n\n---\n\n".join(_block(hit, i) for i, hit in enumerate(hits, 1))


def parent_candidates(hits, parents):
    """Use best child score; rerank scores take precedence when supplied."""
    best = {}
    for hit in hits:
        key = hit.get("parent_id") or hit["chunk_id"]
        score = hit.get("rerank_score", hit.get("score"))
        score = score if score is not None else float("-inf")
        if key not in best or score > best[key][0]:
            best[key] = (score, hit)
    return [
        (parents.get(key, hit), key not in parents, score)
        for key, (score, hit) in sorted(best.items(), key=lambda item: item[1][0], reverse=True)
    ]


def assemble_parent_context(hits, parents):
    candidates = parent_candidates(hits, parents)
    return "\n\n---\n\n".join(
        _block(record, i, missing=missing) for i, (record, missing, _) in enumerate(candidates, 1)
    )


def build_prompt(query, hits, *, parents=None, system_prompt=SYSTEM_PROMPT):
    context = (
        assemble_child_context(hits) if parents is None else assemble_parent_context(hits, parents)
    )
    if not context:
        return {"prompt": None, "context": "", "hits": [], "note": "No chunks retrieved"}
    return {
        "prompt": f"{system_prompt}\n\nContext:\n\n{context}\n\nQuestion: {query}",
        "context": context,
        "hits": hits,
    }
