"""Command-line entry points for independent, repeatable pipeline stages."""

import argparse
import json
import logging
import os
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_QUERY, DataPaths, RerankConfig, RetrievalConfig
from .io import read_jsonl, write_jsonl


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--data-dir", type=Path, default=Path("data"))
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("fetch-pmc", "fetch-abstracts"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--query", default=DEFAULT_QUERY)
        cmd.add_argument("--limit", type=int, default=3000)
        cmd.add_argument("--email", default=os.getenv("NCBI_EMAIL"))
        cmd.add_argument("--api-key", default=os.getenv("NCBI_API_KEY"))
        cmd.add_argument("--output", type=Path)
    for name in ("clean", "clean-abstracts", "chunk", "embed", "analyze"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--input", type=Path)
        if name in ("clean", "clean-abstracts", "analyze"):
            cmd.add_argument("--output", type=Path)
        if name == "chunk":
            cmd.add_argument("--max-tokens", type=int, default=480)
            cmd.add_argument("--overlap-tokens", type=int, default=70)
        if name == "embed":
            cmd.add_argument("--batch-size", type=int, default=20)
        if name == "analyze":
            cmd.add_argument("--chart", type=Path)
    commands.add_parser("setup-nltk", help="Download the sentence tokenizer explicitly")
    for name in ("index", "search", "prompt", "pool"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--host", default="localhost")
        cmd.add_argument("--port", type=int, default=8080)
        cmd.add_argument("--grpc-port", type=int, default=50051)
        cmd.add_argument("--collection", default="AlzheimersChunk")
        if name == "index":
            continue
        cmd.add_argument("--top-k", type=int, default=5)
        if name == "pool":
            cmd.add_argument(
                "--questions", type=Path, default=Path("data/evaluation/questions.jsonl")
            )
            cmd.add_argument("--output", type=Path, default=Path("data/evaluation/judgments.csv"))
            continue
        cmd.add_argument("query")
        cmd.add_argument("--alpha", type=float, default=0.5)
        cmd.add_argument("--section-type")
        cmd.add_argument("--keyword")
        cmd.add_argument("--rerank", action="store_true")
        cmd.add_argument("--candidate-pool-size", type=int, default=30)
        if name == "prompt":
            cmd.add_argument("--context", choices=("child", "parent"), default="child")
    cmd = commands.add_parser("evaluate")
    cmd.add_argument("judgments", type=Path)
    return root


def _database_command(args, paths):
    from .storage.weaviate import connect, index_children

    with connect(host=args.host, port=args.port, grpc_port=args.grpc_port) as client:
        if args.command == "index":
            return index_children(
                client, paths.children, paths.embeddings, paths.id_map, name=args.collection
            )
        from .embeddings import MedCPTEncoder
        from .retrieval.search import HybridRetriever

        retriever = HybridRetriever(
            client.collections.get(args.collection), MedCPTEncoder.for_queries()
        )
        if args.command == "pool":
            from .evaluation import build_pool_for_question, write_judgment_sheet

            parents = {p["parent_id"]: p for p in read_jsonl(paths.parents)}
            rows = []
            for q in read_jsonl(args.questions):
                rows.extend(
                    build_pool_for_question(
                        q["question_id"], q["question"], retriever, parents, args.top_k
                    )
                )
            write_judgment_sheet(rows, args.output)
            return {"candidates": len(rows), "output": str(args.output)}
        cfg = RetrievalConfig(args.top_k, args.alpha, args.section_type, args.keyword)
        search_cfg = replace(cfg, top_k=args.candidate_pool_size) if args.rerank else cfg
        hits = retriever.search(args.query, search_cfg)
        if args.rerank:
            from .retrieval.reranking import MedCPTReranker

            hits = MedCPTReranker().rerank(
                args.query,
                hits,
                RerankConfig(candidate_pool_size=args.candidate_pool_size, rerank_top_k=args.top_k),
            )
        if args.command == "search":
            return hits
        from .generation import build_prompt

        parents = (
            {p["parent_id"]: p for p in read_jsonl(paths.parents)}
            if args.context == "parent"
            else None
        )
        return build_prompt(args.query, hits, parents=parents)


def run(args):
    paths = DataPaths(args.data_dir)
    if args.command in {"fetch-pmc", "fetch-abstracts"}:
        from .ingestion.pubmed import PubMedClient

        client = PubMedClient(args.email, args.api_key)
        pmids = client.search(args.query, args.limit)
        if args.command == "fetch-abstracts":
            output = args.output or paths.root / "raw" / "pubmed_abstracts.jsonl"
            return {"abstracts": write_jsonl(client.abstracts(pmids), output)}
        from .ingestion.pmc import download_papers, pmcids_from_pmids

        return {
            "papers": download_papers(
                pmcids_from_pmids(pmids, args.email), args.output or paths.raw
            )
        }
    if args.command == "clean":
        from .pipeline import clean_corpus

        return {"documents": clean_corpus(args.input or paths.raw, args.output or paths.cleaned)}
    if args.command == "clean-abstracts":
        from .processing.abstracts import abstract_documents

        return {
            "documents": write_jsonl(
                abstract_documents(
                    read_jsonl(args.input or paths.root / "raw" / "pubmed_abstracts.jsonl")
                ),
                args.output or paths.cleaned,
            )
        }
    if args.command == "chunk":
        from .pipeline import chunk_corpus
        from .processing.chunking import ChunkConfig, MedCPTChunker

        chunker = MedCPTChunker(ChunkConfig(args.max_tokens, args.overlap_tokens))
        parents, children = chunk_corpus(
            args.input or paths.cleaned, paths.parents, paths.children, chunker=chunker
        )
        return {"parents": parents, "children": children}
    if args.command == "embed":
        from .embeddings import embed_children

        return {
            "shape": embed_children(
                args.input or paths.children,
                paths.embeddings,
                paths.id_map,
                batch_size=args.batch_size,
            )
        }
    if args.command == "setup-nltk":
        import nltk

        if not nltk.download("punkt_tab"):
            raise RuntimeError("Could not download punkt_tab")
        return {"resource": "punkt_tab"}
    if args.command == "analyze":
        from .analysis import compute_corpus_metrics, plot_corpus_metrics

        metrics = compute_corpus_metrics(read_jsonl(args.input or paths.cleaned))
        if args.chart:
            args.chart.parent.mkdir(parents=True, exist_ok=True)
            plot_corpus_metrics(metrics, args.chart)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics
    if args.command == "evaluate":
        from .evaluation import compute_metrics, load_judgments

        return compute_metrics(load_judgments(args.judgments))
    return _database_command(args, paths)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = run(args)
    except (ValueError, OSError, ImportError, LookupError, RuntimeError) as exc:
        cli.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
