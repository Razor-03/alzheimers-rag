"""Read-only retrieval experiments; metrics join fresh ranks to pooled qrels by ID."""

import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

from .config import RerankConfig, RetrievalConfig
from .generation import generate_answer
from .io import read_jsonl

METHODS = ("bm25", "dense", "hybrid", "hybrid_rerank")
POLICY = "Unjudged retain rank, contribute zero gain; recall denominator is pooled positives."


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def validate_questions(questions):
    ids = [q["question_id"] for q in questions]
    if (
        not ids
        or any(not isinstance(qid, str) or not qid.strip() for qid in ids)
        or len(set(ids)) != len(ids)
        or any(not isinstance(q["question"], str) or not q["question"].strip() for q in questions)
        or any(
            q.get("answerable") is not None and type(q["answerable"]) is not bool for q in questions
        )
        or any(
            q.get("reference_answer") is not None and not isinstance(q["reference_answer"], str)
            for q in questions
        )
    ):
        raise ValueError("Questions need unique nonempty IDs and nonempty text")


def load_qrels(path, *, granularity="child"):
    """Blank labels are unjudged; identical duplicates coalesce, conflicts fail."""
    qrels = {}
    seen = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["granularity"] != granularity:
                continue
            qid, cid, label = row["question_id"], row["candidate_id"], row["relevant"].strip()
            if not qid or not cid or label not in {"", "0", "1", "2"}:
                raise ValueError("Qrels require IDs and blank/0/1/2 relevance")
            key = (qid, cid)
            if key in seen and seen[key] != label:
                raise ValueError(f"Conflicting judgments for {key}")
            seen[key] = label
            qrels.setdefault(qid, {})
            if label:
                qrels[qid][cid] = int(label)
    return qrels


def ranking_metrics(ranked_ids, qrels, k_values=(3, 5, 10)):
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Rankings must contain unique candidate IDs")
    if any(type(v) is not int or v not in {0, 1, 2} for v in qrels.values()):
        raise ValueError("Qrels grades must be integers 0/1/2")
    if not k_values or any(type(k) is not int or k < 1 for k in k_values):
        raise ValueError("K values must be positive integers")
    positives = sum(v > 0 for v in qrels.values())
    result = {"judged_relevant": positives, "judged_candidates": len(qrels)}
    ideal = sorted(qrels.values(), reverse=True)
    for k in k_values:
        ids = ranked_ids[:k]
        grades = [qrels.get(cid, 0) for cid in ids]
        dcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))
        idcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal[:k]))
        first = next((i for i, grade in enumerate(grades, 1) if grade > 0), None)
        judged = sum(cid in qrels for cid in ids)
        result.update(
            {
                f"recall@{k}": sum(g > 0 for g in grades) / positives if positives else None,
                f"mrr@{k}": 1 / first if first else 0.0,
                f"ndcg@{k}": dcg / idcg if idcg else None,
                f"judgment_coverage@{k}": judged / len(ids) if ids else None,
                f"returned@{k}": len(ids),
                f"unjudged@{k}": len(ids) - judged,
            }
        )
    return result


def summarize(rows, field="metrics"):
    summary = {}
    for method in dict.fromkeys(row["method"] for row in rows):
        subset = [row[field] for row in rows if row["method"] == method]
        metrics = {}
        for name in sorted({key for item in subset for key in item}):
            values = [item[name] for item in subset if item.get(name) is not None]
            metrics[name] = {
                "mean": sum(values) / len(values) if values else None,
                "n": len(values),
            }
        summary[method] = metrics
    return summary


def run_benchmark(
    questions,
    qrels,
    retriever,
    reranker,
    *,
    k_values=(3, 5, 10),
    candidate_pool_size=30,
    alpha=0.5,
    llm=None,
    llm_config=None,
    parents=None,
    clock=perf_counter,
):
    """Use one injected retriever, query set and pool size for all four methods.

    Timings exclude construction/model loading and metric calculation. Generation
    includes evidence assembly and validation. No clock values enter result IDs.
    """
    validate_questions(questions)
    if not k_values or min(k_values) < 1 or candidate_pool_size < max(k_values):
        raise ValueError("Pool size must cover every positive K")
    if not 0 < alpha < 1:
        raise ValueError("Hybrid alpha must be strictly between 0 and 1")
    if reranker is None or (llm is None) != (llm_config is None):
        raise ValueError("All four methods require a reranker; LLM and config must be paired")
    rows = []
    for q in questions:
        for method in METHODS:
            cfg = RetrievalConfig(candidate_pool_size, {"bm25": 0, "dense": 1}.get(method, alpha))
            start = clock()
            candidates = retriever.search(q["question"], cfg)
            retrieved = clock()
            # Validate duplicates before they can be hidden by top-K slicing/reranking.
            ids = [hit["chunk_id"] for hit in candidates]
            if len(ids) != len(set(ids)):
                raise ValueError("Retriever returned duplicate chunk IDs")
            hits = candidates
            rerank_seconds = 0.0
            if method == "hybrid_rerank" and hits:
                rank_start = clock()
                hits = reranker.rerank(
                    q["question"], hits, RerankConfig(candidate_pool_size, max(k_values))
                )
                rerank_seconds = clock() - rank_start
            hits = hits[: max(k_values)]
            response, generation_seconds = None, 0.0
            if llm is not None:
                generation_start = clock()
                response = generate_answer(
                    q["question"], hits, llm, config=llm_config, parents=parents
                )
                generation_seconds = clock() - generation_start
            total = clock() - start
            row = {
                "question_id": q["question_id"],
                "question": q["question"],
                "method": method,
                "retrieval_config": asdict(cfg),
                "candidates": candidates,
                "hits": hits,
                "metrics": ranking_metrics(
                    [h["chunk_id"] for h in hits], qrels.get(q["question_id"], {}), k_values
                ),
                "latency_seconds": {
                    "retrieval": retrieved - start,
                    "reranking": rerank_seconds,
                    "generation": generation_seconds if llm else None,
                    "total": total,
                },
            }
            if response is not None:
                row["response"] = response
            row["result_id"] = digest(
                {key: row[key] for key in ("question_id", "question", "method", "hits")}
                | ({"response": response} if response is not None else {})
            )
            rows.append(row)
    return rows


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (datetime, Path)):
        return str(value)
    return value


def collection_snapshot(collection):
    """Fingerprint actual properties AND vectors, independent of iterator order.

    This reads the whole collection. Freeze ingestion during both snapshots and
    the benchmark; equal endpoints cannot detect a transient concurrent mutation.
    """
    objects = sorted(
        (str(obj.uuid), digest({"properties": obj.properties, "vector": obj.vector}))
        for obj in collection.iterator(include_vector=True)
    )
    return {
        "count": len(objects),
        "sha256": digest(objects),
        "configuration": json_value(collection.config.get()),
    }


def runtime_versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("alzheimers-rag", "weaviate-client", "torch", "transformers", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_artifacts(output, metadata, rows):
    """Exclusive run directories protect earlier runs and human review work."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    artifact = {
        "schema_version": 1,
        "metadata": metadata,
        "unjudged_policy": POLICY,
        "summary": summarize(rows),
        "latency_summary_seconds": summarize(rows, "latency_seconds"),
        "per_query": rows,
    }
    (output / "results.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    flat = []
    for row in rows:
        flat.append(
            {key: row[key] for key in ("question_id", "question", "method", "result_id")}
            | row["metrics"]
            | {f"latency_{k}_seconds": v for k, v in row["latency_seconds"].items()}
            | {"ranked_ids": json.dumps([h["chunk_id"] for h in row["hits"]])}
        )
    with (output / "per_query.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]) if flat else [])
        writer.writeheader()
        writer.writerows(flat)
    # A neutral union pool reduces future single-system pooling bias. Existing
    # qrels are never copied over or modified; reviewers label this new sheet.
    pool = {}
    for row in rows:
        for rank, hit in enumerate(row["candidates"], 1):
            key = (row["question_id"], hit["chunk_id"])
            if key not in pool:
                pool[key] = {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "granularity": "child",
                    "candidate_id": hit["chunk_id"],
                    "rank": rank,
                    "text": hit.get("text", ""),
                    "relevant": "",
                    "methods": [],
                }
            pool[key]["methods"].append(row["method"])
    with (output / "candidate_pool.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(next(iter(pool.values()), {})))
        writer.writeheader()
        writer.writerows({**item, "methods": json.dumps(item["methods"])} for item in pool.values())
    return artifact


def load_run(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_questions(path):
    questions = list(read_jsonl(path))
    validate_questions(questions)
    return questions
