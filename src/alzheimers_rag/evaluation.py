"""Judgment pools and retrieval metrics over labeled candidates."""

import csv
import warnings
from collections import defaultdict
from pathlib import Path

from .config import RetrievalConfig
from .generation import parent_candidates

FIELDS = [
    "question_id",
    "question",
    "granularity",
    "candidate_id",
    "rank",
    "score",
    "doc_title",
    "text_preview",
    "relevant",
]


def build_pool_for_question(question_id, question, retriever, parents, top_k=10):
    hits = retriever.search(question, RetrievalConfig(top_k=top_k))
    rows = []

    def row(record, candidate_id, rank, score, granularity):
        return {
            "question_id": question_id,
            "question": question,
            "granularity": granularity,
            "candidate_id": candidate_id,
            "rank": rank,
            "score": score,
            "doc_title": record.get("doc_title") or record.get("title", ""),
            "text_preview": record["text"][:300].replace("\n", " "),
            "relevant": "",
        }

    for rank, hit in enumerate(hits, 1):
        rows.append(row(hit, hit["chunk_id"], rank, hit.get("score"), "child"))
    rank = 0
    for parent, missing, score in parent_candidates(hits, parents):
        if not missing:
            rank += 1
            rows.append(row(parent, parent["parent_id"], rank, score, "parent"))
    return rows


def write_judgment_sheet(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects even an all-blank sheet someone is annotating.
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_judgments(path):
    grouped = defaultdict(list)
    skipped = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["relevant"].strip()
            if not label:
                skipped += 1
                continue
            if label not in {"0", "1", "2"} or int(row["rank"]) < 1:
                raise ValueError("Judgments require relevance 0/1/2 and a positive rank")
            grouped[(row["question_id"], row["granularity"])].append(
                {
                    "candidate_id": row["candidate_id"],
                    "rank": int(row["rank"]),
                    "relevant": int(label),
                }
            )
    if skipped:
        warnings.warn(
            f"Excluded {skipped} unjudged rows; they are not counted as irrelevant", stacklevel=2
        )
    for rows in grouped.values():
        rows.sort(key=lambda row: row["rank"])
    return grouped


def recall_at_k(candidates, k):
    relevant = sum(c["relevant"] > 0 for c in candidates)
    if not relevant:
        return None
    return sum(c["relevant"] > 0 and c["rank"] <= k for c in candidates) / relevant


def mrr(candidates):
    ranks = [c["rank"] for c in candidates if c["relevant"] > 0]
    return 1 / min(ranks) if ranks else 0.0


def compute_metrics(grouped, k_values=(3, 5, 10)):
    results = defaultdict(lambda: defaultdict(list))
    for (_, granularity), candidates in grouped.items():
        for k in k_values:
            value = recall_at_k(candidates, k)
            if value is not None:
                results[granularity][f"recall@{k}"].append(value)
        results[granularity]["mrr"].append(mrr(candidates))
    return {
        granularity: {
            metric: {"mean": sum(values) / len(values), "n": len(values)}
            for metric, values in metrics.items()
        }
        for granularity, metrics in results.items()
    }
