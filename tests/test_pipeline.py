"""Offline regression tests for stage boundaries and notebook bug fixes."""

import importlib
import json
import pkgutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import alzheimers_rag
from alzheimers_rag.analysis import compute_corpus_metrics
from alzheimers_rag.config import RetrievalConfig
from alzheimers_rag.embeddings import article_input
from alzheimers_rag.evaluation import compute_metrics, load_judgments, mrr, recall_at_k
from alzheimers_rag.generation import assemble_parent_context, build_prompt
from alzheimers_rag.io import raw_documents, read_jsonl, write_jsonl
from alzheimers_rag.pipeline import chunk_corpus, clean_corpus
from alzheimers_rag.processing.abstracts import abstract_documents, chunk_abstract
from alzheimers_rag.processing.chunking import ChunkConfig, MedCPTChunker
from alzheimers_rag.processing.cleaning import process_document
from alzheimers_rag.retrieval.search import rrf_fusion
from alzheimers_rag.storage.weaviate import chunk_properties


class WordTokenizer:
    """A predictable tokenizer for exercising window boundaries without downloads."""

    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, ids, **kwargs):
        return " ".join(ids)


def passage(text, kind="paragraph", section="RESULTS", offset=0):
    return {"text": text, "offset": offset, "infons": {"type": kind, "section_type": section}}


def document():
    front = passage("A research title", "front", "TITLE")
    front["infons"].update(year="2024", kwd="amyloid;tau")
    return {
        "id": "PMC123",
        "passages": [
            front,
            passage("Results", "title_1", offset=1),
            passage("one two three four. five six seven eight. nine ten.", offset=2),
            passage("reference junk", section="REF", offset=3),
            passage("raw table", kind="table", section="TABLE", offset=4),
            passage("A useful caption", kind="table_caption", section="TABLE", offset=5),
        ],
    }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.chunker = MedCPTChunker(
            ChunkConfig(5, 2), tokenizer=WordTokenizer(), sentence_splitter=lambda s: s.split(". ")
        )

    def test_clean_chunk_and_storage_schema_agree(self):
        raw, clean = self.root / "raw.jsonl", self.root / "clean.jsonl"
        parents, children = self.root / "parents.jsonl", self.root / "children.jsonl"
        write_jsonl([[{"documents": [document()]}]], raw)
        self.assertEqual(clean_corpus(raw, clean), 1)
        parent_count, child_count = chunk_corpus(clean, parents, children, chunker=self.chunker)
        self.assertGreaterEqual(child_count, parent_count)
        lookup = {p["parent_id"]: p for p in read_jsonl(parents)}
        for child in read_jsonl(children):
            self.assertIn(child["parent_id"], lookup)
            self.assertEqual(child["doc_title"], "A research title")
            self.assertEqual(child["token_count"], child["n_tokens"])
            self.assertEqual(article_input(child)[0], "A research title")
            self.assertLessEqual(child["n_tokens"], 5)
            self.assertEqual(
                chunk_properties(child)["section_type"], child["canonical_section_type"]
            )
        text = " ".join(p["text"] for p in lookup.values())
        self.assertNotIn("reference junk", text)
        self.assertNotIn("raw table", text)
        self.assertIn("A useful caption", text)

    def test_raw_line_shapes(self):
        doc = document()
        path = self.root / "shapes.jsonl"
        write_jsonl([doc, {"documents": [doc]}, [{"documents": [doc]}], [doc]], path)
        self.assertEqual(len(list(raw_documents(path))), 4)
        write_jsonl([{"unexpected": []}], path)
        with self.assertRaises(ValueError):
            list(raw_documents(path))

    def test_oversized_sentences_preserve_all_words(self):
        for kind in ("abstract", "results"):
            text = " ".join(f"word{i}" for i in range(17))
            chunks = self.chunker.chunk_section({"text": text, "canonical_section_type": kind})
            self.assertTrue(all(c["n_tokens"] <= 5 for c in chunks))
            self.assertEqual(" ".join(c["text"] for c in chunks), text)

    def test_overlap_and_filters(self):
        chunker = MedCPTChunker(
            ChunkConfig(6, 2),
            tokenizer=WordTokenizer(),
            sentence_splitter=lambda s: [part.strip() for part in s.split("|")],
        )
        chunks = chunker.chunk_section(
            {"text": "a b | c d | e f | g h", "canonical_section_type": "results"}
        )
        self.assertEqual([c["text"] for c in chunks], ["a b c d e f", "e f g h"])
        self.assertEqual(
            chunker.chunk_section({"text": "\u4e2d" * 10, "canonical_section_type": "results"}), []
        )
        self.assertEqual(
            chunker.chunk_section(
                {"text": "AA: a BB: b CC: c", "canonical_section_type": "results"}
            ),
            [],
        )
        with self.assertRaises(ValueError):
            ChunkConfig(3, 3)

    def test_diagnostics_do_not_leak_between_corpora(self):
        doc = {"id": "PMC9", "passages": [passage("unknown", section="NEW_TYPE")]}
        first = compute_corpus_metrics([process_document(doc)])
        second = compute_corpus_metrics([process_document(doc)])
        self.assertEqual(first["unmapped_section_types"], {"NEW_TYPE": 1})
        self.assertEqual(first, second)

    def test_parent_context_uses_rerank_order_and_fallback(self):
        hits = [
            {"chunk_id": "a", "parent_id": "p", "text": "child", "score": 9, "rerank_score": 1},
            {"chunk_id": "b", "parent_id": "q", "text": "fallback", "score": 1, "rerank_score": 4},
            {"chunk_id": "c", "parent_id": "p", "text": "child2", "score": 8, "rerank_score": 2},
        ]
        context = assemble_parent_context(hits, {"p": {"title": "Parent", "text": "full parent"}})
        self.assertLess(context.index("fallback"), context.index("full parent"))
        self.assertEqual(context.count("full parent"), 1)
        self.assertIn("[Source 2] Parent", context)
        self.assertIn("parent unavailable", context)
        self.assertIsNone(build_prompt("question", [])["prompt"])

    def test_rrf_uses_chunk_identity(self):
        a = {"chunk_id": "a", "journal": "same"}
        b = {"chunk_id": "b", "journal": "same"}
        fused = rrf_fusion([a, b], [b])
        self.assertEqual([c["chunk_id"] for c in fused], ["b", "a"])

    def test_abstracts_use_shared_schema(self):
        records = [
            {"pmid": "1", "title": "Paper", "abstract": "some text"},
            {"pmid": "1", "title": "Duplicate", "abstract": "other text"},
            {"pmid": "2", "title": "Missing", "abstract": None},
        ]
        docs = list(abstract_documents(records))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["sections"][0]["section_id"], "PMID1_s0")
        with self.assertRaises(ValueError):
            chunk_abstract("text", 2, 2)

    def test_metrics_preserve_unjudged_rank_gaps(self):
        path = self.root / "judgments.csv"
        path.write_text(
            "question_id,granularity,candidate_id,rank,relevant\n"
            "q1,child,a,1,\nq1,child,b,2,1\nq1,child,c,4,1\n",
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as caught:
            grouped = load_judgments(path)
        self.assertEqual(len(caught), 1)
        candidates = grouped[("q1", "child")]
        self.assertEqual(recall_at_k(candidates, 3), 0.5)
        self.assertEqual(mrr(candidates), 0.5)
        self.assertIsNone(recall_at_k([], 3))
        self.assertEqual(compute_metrics(grouped)["child"]["mrr"]["n"], 1)

    def test_json_errors_include_line_number(self):
        path = self.root / "bad.jsonl"
        path.write_text("{}\n{broken}\n")
        with self.assertRaisesRegex(ValueError, ":2: invalid JSON"):
            list(read_jsonl(path))

    def test_failed_write_preserves_previous_output(self):
        path = self.root / "output.jsonl"
        write_jsonl([{"old": True}], path)

        def broken_records():
            yield {"new": True}
            raise ValueError("bad input")

        with self.assertRaisesRegex(ValueError, "bad input"):
            write_jsonl(broken_records(), path)
        self.assertEqual(list(read_jsonl(path)), [{"old": True}])
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_configuration_validation(self):
        with self.assertRaises(ValueError):
            RetrievalConfig(alpha=2)

    def test_imports_have_no_model_or_database_side_effects(self):
        # This suite runs without any optional model, HTTP, or database packages installed.
        for module in pkgutil.walk_packages(alzheimers_rag.__path__, alzheimers_rag.__name__ + "."):
            if not module.name.endswith(".__main__"):
                importlib.import_module(module.name)

    def test_cli_can_clean_without_optional_dependencies(self):
        raw = self.root / "raw.jsonl"
        clean = self.root / "clean.jsonl"
        write_jsonl([document()], raw)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alzheimers_rag",
                "clean",
                "--input",
                str(raw),
                "--output",
                str(clean),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"documents": 1})


if __name__ == "__main__":
    unittest.main()
