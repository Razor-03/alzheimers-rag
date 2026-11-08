"""Evaluation commands kept separate from the serving and ingestion paths."""

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

from .benchmark import (
    collection_snapshot,
    digest,
    file_digest,
    load_qrels,
    load_questions,
    load_run,
    ranking_metrics,
    run_benchmark,
    runtime_versions,
    save_artifacts,
    utc_now,
)
from .config import ARTICLE_MODEL, QUERY_MODEL, RERANK_MODEL, LLMConfig
from .io import read_jsonl


def add_commands(commands):
    cmd = commands.add_parser("benchmark", help="Read-only four-method evaluation")
    cmd.add_argument("--questions", type=Path, default=Path("data/evaluation/questions.jsonl"))
    cmd.add_argument("--judgments", type=Path, default=Path("data/evaluation/judgments.csv"))
    cmd.add_argument("--manifest", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True, help="New run directory; must not exist")
    cmd.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    cmd.add_argument("--candidate-pool-size", type=int, default=30)
    cmd.add_argument("--alpha", type=float, default=0.5)
    cmd.add_argument("--host", default="localhost")
    cmd.add_argument("--port", type=int, default=8080)
    cmd.add_argument("--grpc-port", type=int, default=50051)
    cmd.add_argument("--collection", required=True)
    cmd.add_argument("--device", default="cpu", help="Explicit model device; CPU default")
    cmd.add_argument("--query-revision", help="Hugging Face commit for encoder and tokenizer")
    cmd.add_argument("--rerank-revision", help="Hugging Face commit for reranker and tokenizer")
    cmd.add_argument("--answers", action="store_true", help="Also call the configured LLM")
    cmd.add_argument("--context", choices=("child", "parent"), default="child")
    _llm_options(cmd)
    cmd = commands.add_parser("score-retrieval", help="Rescore saved rankings without services")
    cmd.add_argument("run", type=Path)
    cmd.add_argument("--judgments", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd = commands.add_parser("evaluate-answers", help="Score one human OR LLM label file")
    _answer_inputs(cmd)
    cmd.add_argument("--labels", type=Path, required=True)
    cmd.add_argument("--source", choices=("human", "llm"), default="human")
    cmd = commands.add_parser("judge-answers", help="Opt-in LLM labels, separately identified")
    _answer_inputs(cmd)
    _llm_options(cmd)


def _answer_inputs(cmd):
    cmd.add_argument("run", type=Path, help="Benchmark results.json containing responses")
    cmd.add_argument("--questions", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True, help="New output directory")


def _llm_options(cmd):
    cmd.add_argument("--model", default=os.getenv("RAG_LLM_MODEL", ""))
    cmd.add_argument(
        "--provider",
        choices=("ollama", "openai-compatible"),
        default=os.getenv("RAG_LLM_PROVIDER", "ollama"),
    )
    cmd.add_argument("--base-url", default=os.getenv("RAG_LLM_BASE_URL"))
    cmd.add_argument("--api-key-env", default="RAG_LLM_API_KEY")
    cmd.add_argument("--context-window", type=int, default=8192)
    cmd.add_argument("--context-token-budget", type=int, default=4096)
    cmd.add_argument("--max-output-tokens", type=int, default=1024)
    cmd.add_argument("--safety-tokens", type=int, default=128)
    cmd.add_argument("--timeout", type=float, default=120)


def _llm_config(args):
    base_url = args.base_url
    if base_url is None:
        if args.provider != "ollama":
            raise ValueError("Set --base-url for openai-compatible")
        base_url = "http://localhost:11434"
    return LLMConfig(
        **{
            key: getattr(args, key)
            for key in (
                "model",
                "provider",
                "api_key_env",
                "context_window",
                "context_token_budget",
                "max_output_tokens",
                "safety_tokens",
                "timeout",
            )
        },
        base_url=base_url,
    )


def _source_fingerprint():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*.py"))})


def _benchmark(args, paths):
    questions = load_questions(args.questions)
    qrels = load_qrels(args.judgments)
    manifest = load_run(args.manifest)
    for key in ("dataset_version", "corpus_version", "article_encoder", "article_revision"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Manifest requires a nonempty {key}")
    if manifest["corpus_version"].startswith("REPLACE_"):
        raise ValueError("Replace the example corpus_version with your frozen corpus version")
    if manifest["article_encoder"] != ARTICLE_MODEL:
        raise ValueError("Benchmark query encoder requires the matching MedCPT article encoder")
    if (
        manifest.get("article_pooling", "cls") != "cls"
        or manifest.get("article_normalize", False) is not False
    ):
        raise ValueError("Article provenance must match CLS, unnormalized query encoding")
    versions = {q["dataset_version"] for q in questions if "dataset_version" in q}
    if versions and versions != {manifest["dataset_version"]}:
        raise ValueError("Manifest dataset_version differs from the question dataset")
    if not args.k or min(args.k) < 1 or args.candidate_pool_size < max(args.k):
        raise ValueError("Pool size must cover every positive K")
    if not 0 < args.alpha < 1:
        raise ValueError("Hybrid alpha must be strictly between 0 and 1")
    config = _llm_config(args) if args.answers else None
    parents = None
    if args.answers and args.context == "parent":
        parents = {p["parent_id"]: p for p in read_jsonl(paths.parents)}
    inputs = {"questions": args.questions, "judgments": args.judgments, "manifest": args.manifest}
    if parents is not None:
        inputs["parents"] = paths.parents
    hashes = {key: file_digest(path) for key, path in inputs.items()}
    source_hash = _source_fingerprint()
    import torch

    from .embeddings import MedCPTEncoder
    from .llm import HTTPChatModel
    from .retrieval.reranking import MedCPTReranker
    from .retrieval.search import HybridRetriever
    from .storage.weaviate import connect

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    started = utc_now()
    with connect(host=args.host, port=args.port, grpc_port=args.grpc_port) as client:
        if not client.collections.exists(args.collection):
            raise ValueError("Benchmark requires an existing collection; it never creates one")
        collection = client.collections.get(args.collection)
        before = collection_snapshot(collection)
        if not before["count"]:
            raise ValueError("Cannot benchmark an empty collection")
        encoder = MedCPTEncoder.for_queries(device=args.device, revision=args.query_revision)
        reranker = MedCPTReranker(device=args.device, revision=args.rerank_revision)
        rows = run_benchmark(
            questions,
            qrels,
            HybridRetriever(collection, encoder),
            reranker,
            k_values=tuple(sorted(set(args.k))),
            candidate_pool_size=args.candidate_pool_size,
            alpha=args.alpha,
            llm=HTTPChatModel() if args.answers else None,
            llm_config=config,
            parents=parents,
        )
        after = collection_snapshot(collection)
        if before != after:
            raise RuntimeError("Collection changed during benchmark; discard this experiment")
        if hashes != {key: file_digest(path) for key, path in inputs.items()}:
            raise RuntimeError("Evaluation inputs changed during benchmark")
        if source_hash != _source_fingerprint():
            raise RuntimeError("Source code changed during benchmark")
        metadata = {
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "manifest": manifest,
            "input_sha256": hashes,
            "source_sha256": source_hash,
            "runtime": runtime_versions(),
            "collection": args.collection,
            "host": args.host,
            "port": args.port,
            "grpc_port": args.grpc_port,
            "snapshot": before,
            "query_model": QUERY_MODEL,
            "query_revision": getattr(encoder.model.config, "_commit_hash", None),
            "requested_query_revision": args.query_revision,
            "rerank_model": RERANK_MODEL,
            "rerank_revision": getattr(reranker.model.config, "_commit_hash", None),
            "requested_rerank_revision": args.rerank_revision,
            "device": args.device,
            "pooling": encoder.pooling,
            "normalize": encoder.normalize,
            "query_max_length": 64,
            "rerank_max_length": 512,
            "rerank_batch_size": 16,
            "k_values": sorted(set(args.k)),
            "candidate_pool_size": args.candidate_pool_size,
            "hybrid_alpha": args.alpha,
            "seed": 0,
            "deterministic_torch": True,
            "context": args.context,
            "generation": asdict(config) if config else None,
            "timing": "perf_counter; sequential fixed method order; no warmup; model loading excluded",
        }
        artifact = save_artifacts(args.output, metadata, rows)
    if args.answers:
        from .answer_evaluation import write_templates

        write_templates(args.output / "human-label-template.jsonl", artifact, questions)
    return {"output": str(args.output), "summary": artifact["summary"]}


def run_evaluation(args, paths):
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}; choose a new run directory")
    if args.command == "benchmark":
        return _benchmark(args, paths)
    if args.command == "score-retrieval":
        artifact = load_run(args.run)
        qrels = load_qrels(args.judgments)
        k_values = artifact["metadata"]["k_values"]
        rows = artifact["per_query"]
        for row in rows:
            row["metrics"] = ranking_metrics(
                [h["chunk_id"] for h in row["hits"]],
                qrels.get(row["question_id"], {}),
                k_values,
            )
        metadata = artifact["metadata"] | {
            "rescored_at_utc": utc_now(),
            "parent_run_sha256": file_digest(args.run),
            "rescored_judgments_sha256": file_digest(args.judgments),
            "scoring_source_sha256": _source_fingerprint(),
        }
        report = save_artifacts(args.output, metadata, rows)
        return {"output": str(args.output), "summary": report["summary"]}
    from .answer_evaluation import evaluate_answers, judge_answer, write_json_exclusive

    questions = load_questions(args.questions)
    run = load_run(args.run)
    expected_hash = run.get("metadata", {}).get("input_sha256", {}).get("questions")
    if expected_hash and expected_hash != file_digest(args.questions):
        raise ValueError("Use the exact question dataset used to generate this run")
    if args.command == "evaluate-answers":
        labels = list(read_jsonl(args.labels))
        source = args.source
        provenance = {"labels_sha256": file_digest(args.labels)}
    else:
        from .llm import HTTPChatModel

        config = _llm_config(args)
        by_id = {q["question_id"]: q for q in questions}
        labels = [
            judge_answer(row, by_id[row["question_id"]], HTTPChatModel(), config)
            for row in run["per_query"]
            if "response" in row
        ]
        source = "llm"
        provenance = {"judge": asdict(config)}
    report = evaluate_answers(run, questions, labels, source=source)
    report["provenance"] = provenance | {
        "runtime": runtime_versions(),
        "source_sha256": _source_fingerprint(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(args.output / "scores.json", report)
    with (args.output / "labels.jsonl").open("x", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label, ensure_ascii=False, allow_nan=False) + "\n")
    flat = [
        {key: row[key] for key in ("result_id", "question_id", "method")} | row["metrics"]
        for row in report["per_query"]
    ]
    with (args.output / "per_query.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return {"output": str(args.output), "label_source": source, "summary": report["summary"]}
