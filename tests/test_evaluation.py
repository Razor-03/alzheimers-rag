"""Analytic, offline fixtures only: these are not measured retrieval/answer scores."""

import copy
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from alzheimers_rag.answer_evaluation import (
    annotation_template,
    evaluate_answers,
    judge_answer,
    score_answer,
)
from alzheimers_rag.benchmark import (
    METHODS,
    collection_snapshot,
    load_qrels,
    ranking_metrics,
    run_benchmark,
    save_artifacts,
    summarize,
)
from alzheimers_rag.cli import parser, run
from alzheimers_rag.config import LLMConfig
from alzheimers_rag.evaluation import write_judgment_sheet


def hit(cid):
    return {"chunk_id": cid, "text": "Fixture evidence", "score": 0.5}


def answer_row():
    return {
        "result_id": "answer-fingerprint",
        "question_id": "q",
        "question": "Fixture question?",
        "method": "hybrid",
        "response": {
            "status": "answered",
            "statements": [{"text": "Fixture claim", "source_ids": ["S1", "S2"]}],
            "evidence": [
                {"source_id": "S1", "text": "Supporting excerpt"},
                {"source_id": "S2", "text": "Unrelated excerpt"},
            ],
        },
    }


QUESTION = {"question_id": "q", "question": "Fixture question?", "answerable": True}


def human_label(row):
    label = annotation_template(row, QUESTION)
    label.update(annotator="reviewer-1", claims_complete=True, correctness=False)
    label["claims"][0].update(
        faithful=True, correct=False, citation_support={"S1": True, "S2": False}
    )
    return label


class RetrievalMetricTests(unittest.TestCase):
    def test_graded_ndcg_binary_recall_and_cutoff_mrr(self):
        qrels = {"a": 2, "b": 1, "c": 0, "missing": 1}
        scores = ranking_metrics(["unjudged", "b", "a", "c"], qrels, (1, 2, 3, 10))
        self.assertEqual(scores["recall@3"], 2 / 3)
        self.assertEqual(scores["mrr@1"], 0)
        self.assertEqual(scores["mrr@2"], 0.5)
        expected = (1 / math.log2(3) + 3 / math.log2(4)) / (3 + 1 / math.log2(3) + 1 / math.log2(4))
        self.assertAlmostEqual(scores["ndcg@3"], expected)
        self.assertEqual(scores["judgment_coverage@3"], 2 / 3)
        self.assertEqual(scores["returned@10"], 4)
        self.assertEqual(scores["unjudged@3"], 1)

    def test_perfect_ranking_and_no_positive_labels(self):
        scores = ranking_metrics(["a", "b"], {"a": 2, "b": 1}, (2,))
        for key in ("recall@2", "ndcg@2", "mrr@2", "judgment_coverage@2"):
            self.assertEqual(scores[key], 1)
        scores = ranking_metrics(["a"], {"a": 0}, (3,))
        self.assertIsNone(scores["recall@3"])
        self.assertIsNone(scores["ndcg@3"])
        self.assertEqual(scores["mrr@3"], 0)

    def test_empty_retrieval_and_no_judgments(self):
        scores = ranking_metrics([], {"a": 1}, (5,))
        self.assertEqual(scores["recall@5"], 0)
        self.assertEqual(scores["ndcg@5"], 0)
        self.assertIsNone(scores["judgment_coverage@5"])
        self.assertEqual(scores["returned@5"], 0)
        scores = ranking_metrics(["u"], {}, (1,))
        self.assertIsNone(scores["recall@1"])
        self.assertEqual(scores["judgment_coverage@1"], 0)

    def test_invalid_grades_cutoffs_and_duplicates(self):
        for ids, qrels, k in [(["a", "a"], {}, (1,)), ([], {"a": -1}, (1,)), ([], {}, (0,))]:
            with self.assertRaises(ValueError):
                ranking_metrics(ids, qrels, k)

    def test_macro_denominators_are_explicit(self):
        rows = [
            {"method": "bm25", "metrics": ranking_metrics(["a"], {"a": 1}, (1,))},
            {"method": "bm25", "metrics": ranking_metrics([], {}, (1,))},
        ]
        metrics = summarize(rows)["bm25"]
        self.assertEqual(metrics["recall@1"], {"mean": 1, "n": 1})
        self.assertEqual(metrics["mrr@1"], {"mean": 0.5, "n": 2})

    def test_qrels_mixed_labels_conflicts_and_granularity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qrels.csv"
            header = "question_id,granularity,candidate_id,rank,relevant\n"
            path.write_text(
                header + "q,child,a,99,2\nq,child,b,1,0\nq,child,c,2,\nq,parent,a,1,0\n"
            )
            self.assertEqual(load_qrels(path), {"q": {"a": 2, "b": 0}})
            path.write_text(header + "q,child,a,1,1\nq,child,a,2,0\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_qrels(path)
            path.write_text(header + "q,child,a,1,1\nq,child,a,2,\n")
            with self.assertRaises(ValueError):
                load_qrels(path)


class BenchmarkTests(unittest.TestCase):
    def test_cli_uses_existing_collection_and_rejects_corpus_changes(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                questions = root / "questions.jsonl"
                judgments = root / "judgments.csv"
                manifest = root / "manifest.json"
                questions.write_text(json.dumps(QUESTION) + "\n")
                judgments.write_text("question_id,granularity,candidate_id,relevant\nq,child,a,1\n")
                manifest.write_text(
                    json.dumps(
                        {
                            "dataset_version": "fixture-v1",
                            "corpus_version": "fixture-corpus",
                            "article_encoder": "ncbi/MedCPT-Article-Encoder",
                            "article_revision": "fixture-sha",
                        }
                    )
                )
                client = MagicMock()
                client.__enter__.return_value = client
                collection = client.collections.get.return_value
                collection.config.get.return_value = {"distance": "dot"}
                a = SimpleNamespace(uuid="a", properties=hit("a"), vector={"default": [1.0]})
                b = copy.deepcopy(a)
                if changed:
                    b.vector["default"] = [2.0]
                collection.iterator.side_effect = [[a], [b]]
                encoder = SimpleNamespace(
                    pooling="cls",
                    normalize=False,
                    model=SimpleNamespace(config=SimpleNamespace(_commit_hash="query-sha")),
                )
                reranker = Mock()
                reranker.model.config._commit_hash = "rerank-sha"
                reranker.rerank.side_effect = lambda query, hits, cfg: hits
                retriever = Mock(search=Mock(return_value=[hit("a")]))
                args = parser().parse_args(
                    [
                        "benchmark",
                        "--questions",
                        str(questions),
                        "--judgments",
                        str(judgments),
                        "--manifest",
                        str(manifest),
                        "--collection",
                        "Existing",
                        "--k",
                        "1",
                        "--query-revision",
                        "query-sha",
                        "--rerank-revision",
                        "rerank-sha",
                        "--output",
                        str(root / "run"),
                    ]
                )
                with (
                    patch.dict(sys.modules, {"torch": Mock()}),
                    patch("alzheimers_rag.storage.weaviate.connect", return_value=client),
                    patch(
                        "alzheimers_rag.embeddings.MedCPTEncoder.for_queries", return_value=encoder
                    ) as load,
                    patch(
                        "alzheimers_rag.retrieval.reranking.MedCPTReranker", return_value=reranker
                    ),
                    patch(
                        "alzheimers_rag.retrieval.search.HybridRetriever", return_value=retriever
                    ),
                ):
                    if changed:
                        with self.assertRaisesRegex(RuntimeError, "Collection changed"):
                            run(args)
                        self.assertFalse((root / "run").exists())
                    else:
                        run(args)
                        artifact = json.loads((root / "run/results.json").read_text())
                        self.assertEqual(artifact["metadata"]["query_revision"], "query-sha")
                        self.assertEqual(artifact["metadata"]["snapshot"]["count"], 1)
                    load.assert_called_once_with(device="cpu", revision="query-sha")
                client.collections.create.assert_not_called()
                client.collections.delete.assert_not_called()

    def benchmark(self, *, empty=False, answers=False):
        retriever = Mock()
        retriever.search.side_effect = lambda query, cfg: [] if empty else [hit("u"), hit("a")]
        reranker = Mock()
        reranker.rerank.side_effect = lambda query, hits, cfg: list(reversed(hits))
        llm = Mock()
        llm.complete.return_value = json.dumps(
            {
                "status": "answered",
                "statements": [{"text": "Claim", "source_ids": ["S1"]}],
                "limitations": [],
            }
        )
        clock = Mock(side_effect=range(1000))
        rows = run_benchmark(
            [QUESTION],
            {"q": {"a": 1}},
            retriever,
            reranker,
            k_values=(1, 2),
            candidate_pool_size=2,
            clock=clock,
            llm=llm if answers else None,
            llm_config=LLMConfig(model="fixture") if answers else None,
        )
        return rows, retriever, reranker, llm

    def test_four_methods_same_query_pool_and_fresh_ranks(self):
        rows, retriever, reranker, _ = self.benchmark()
        self.assertEqual([row["method"] for row in rows], list(METHODS))
        self.assertEqual(
            [c.args[1].alpha for c in retriever.search.call_args_list], [0, 1, 0.5, 0.5]
        )
        self.assertTrue(all(c.args[1].top_k == 2 for c in retriever.search.call_args_list))
        self.assertEqual(rows[0]["metrics"]["mrr@2"], 0.5)
        self.assertEqual(rows[-1]["metrics"]["mrr@2"], 1)
        self.assertEqual(reranker.rerank.call_count, 1)
        self.assertEqual(
            rows[-1]["latency_seconds"],
            {
                "retrieval": 1,
                "reranking": 1,
                "generation": None,
                "total": 4,
            },
        )

    def test_repeatable_artifacts_and_protection(self):
        first, *_ = self.benchmark(answers=True)
        second, *_ = self.benchmark(answers=True)
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            save_artifacts(a, {"fixture": True}, first)
            save_artifacts(b, {"fixture": True}, second)
            for name in ("results.json", "per_query.csv", "candidate_pool.csv"):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
            with self.assertRaises(FileExistsError):
                save_artifacts(a, {}, first)
            with (a / "candidate_pool.csv").open(encoding="utf-8", newline="") as handle:
                pool = list(csv.DictReader(handle))
            self.assertEqual(len(pool), 2)
            self.assertTrue(all(row["relevant"] == "" for row in pool))

    def test_saved_rankings_can_be_rescored_without_services(self):
        rows, *_ = self.benchmark()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_artifacts(root / "original", {"k_values": [1, 2]}, rows)
            qrels = root / "labels.csv"
            qrels.write_text(
                "question_id,granularity,candidate_id,relevant\nq,child,u,1\nq,child,a,0\n"
            )
            args = parser().parse_args(
                [
                    "score-retrieval",
                    str(root / "original/results.json"),
                    "--judgments",
                    str(qrels),
                    "--output",
                    str(root / "rescored"),
                ]
            )
            result = run(args)
            self.assertEqual(result["summary"]["bm25"]["recall@1"]["mean"], 1)
            rescored = json.loads((root / "rescored/results.json").read_text())
            self.assertEqual(
                rescored["per_query"][0]["latency_seconds"], rows[0]["latency_seconds"]
            )
            self.assertEqual(rescored["per_query"][0]["result_id"], rows[0]["result_id"])

    def test_real_clock_does_not_change_result_identity(self):
        first, *_ = self.benchmark()
        retriever = Mock(search=Mock(return_value=[hit("u"), hit("a")]))
        reranker = Mock(rerank=Mock(side_effect=lambda q, hits, cfg: list(reversed(hits))))
        second = run_benchmark(
            [QUESTION], {"q": {"a": 1}}, retriever, reranker, k_values=(1, 2), candidate_pool_size=2
        )
        self.assertEqual([r["result_id"] for r in first], [r["result_id"] for r in second])

    def test_empty_retrieval_skips_reranker_and_llm_and_times_generation(self):
        rows, _, reranker, llm = self.benchmark(empty=True, answers=True)
        reranker.rerank.assert_not_called()
        llm.complete.assert_not_called()
        for row in rows:
            self.assertEqual(row["response"]["status"], "insufficient_evidence")
            self.assertEqual(row["latency_seconds"]["generation"], 1)
            self.assertEqual(row["metrics"]["recall@2"], 0)

    def test_invalid_experiments_fail(self):
        for kwargs in ({"candidate_pool_size": 1}, {"alpha": 0}, {"alpha": 1}):
            with self.assertRaises(ValueError):
                run_benchmark([QUESTION], {}, Mock(), Mock(), **kwargs)
        with self.assertRaises(ValueError):
            run_benchmark([QUESTION, QUESTION], {}, Mock(), Mock())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_benchmark(
                [QUESTION], {}, Mock(search=Mock(return_value=[hit("a"), hit("a")])), Mock()
            )

    def test_collection_snapshot_includes_vectors_and_ignores_iteration_order(self):
        a = SimpleNamespace(uuid="a", properties=hit("a"), vector={"default": [1.0, 0]})
        b = SimpleNamespace(uuid="b", properties=hit("b"), vector={"default": [0, 1.0]})
        collection = Mock()
        collection.config.get.return_value = {"distance": "dot"}
        collection.iterator.return_value = [a, b]
        before = collection_snapshot(collection)
        collection.iterator.return_value = [b, a]
        self.assertEqual(before, collection_snapshot(collection))
        a.vector["default"][0] = 2.0
        self.assertNotEqual(before["sha256"], collection_snapshot(collection)["sha256"])
        collection.iterator.assert_called_with(include_vector=True)

    def test_pool_never_overwrites_even_blank_human_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "human.csv"
            path.write_text("manually maintained data\n")
            with self.assertRaises(FileExistsError):
                write_judgment_sheet([], path)
            with self.assertRaises(FileExistsError):
                run(parser().parse_args(["pool", "--output", str(path)]))
            self.assertEqual(path.read_text(), "manually maintained data\n")


class AnswerEvaluationTests(unittest.TestCase):
    def test_human_answerability_override_requires_reviewer(self):
        row = answer_row()
        question = QUESTION | {"answerable": None}
        label = annotation_template(row, question)
        label["answerable"] = False
        with self.assertRaisesRegex(ValueError, "annotator"):
            score_answer(row, question, label)
        label["annotator"] = "reviewer"
        self.assertEqual(score_answer(row, question, label)["abstention_accuracy"], 0)

    def test_judge_budget_fails_before_request(self):
        llm = Mock()
        with self.assertRaisesRegex(ValueError, "input budget"):
            judge_answer(
                answer_row(), QUESTION, llm, LLMConfig(model="fixture", context_window=1200)
            )
        llm.complete.assert_not_called()

    def test_faithfulness_correctness_and_citation_entailment_are_distinct(self):
        row = answer_row()
        score = score_answer(row, QUESTION, human_label(row))
        self.assertEqual(score["faithfulness"], 1)
        self.assertEqual(score["claim_correctness"], 0)
        self.assertEqual(score["correctness"], 0)
        self.assertEqual(score["citation_precision"], 0.5)
        self.assertEqual(score["citation_support"], 1)
        self.assertEqual(score["citation_id_validity"], 1)
        self.assertEqual(score["abstention_accuracy"], 1)

    def test_valid_citation_can_be_semantically_unsupported(self):
        row = answer_row()
        label = human_label(row)
        label["claims"][0].update(faithful=False, citation_support={"S1": False, "S2": False})
        score = score_answer(row, QUESTION, label)
        self.assertEqual(score["citation_id_validity"], 1)
        self.assertEqual(score["citation_support"], 0)
        self.assertEqual(score["faithfulness"], 0)

    def test_unknown_citation_cannot_be_marked_supported(self):
        row = answer_row()
        row["response"]["evidence"] = []
        label = human_label(row)
        score = score_answer(row, QUESTION, label)
        self.assertEqual(score["unsupported_citation_ids"], 2)
        self.assertEqual(score["citation_precision"], 0)
        self.assertEqual(score["citation_support"], 0)

    def test_atomic_claim_splits_and_citation_pair_denominator(self):
        row = answer_row()
        label = human_label(row)
        label["claims"].append(copy.deepcopy(label["claims"][0]))
        label["claims"][1].update(text="Second atomic claim", faithful=False, correct=True)
        label["claims"][1]["citation_support"]["S1"] = False
        score = score_answer(row, QUESTION, label)
        self.assertEqual(score["faithfulness"], 0.5)
        self.assertEqual(score["claim_correctness"], 0.5)
        self.assertEqual(score["citation_precision"], 0)
        self.assertEqual(score["citation_support"], 0.5)

    def test_unlabeled_and_partial_labels_never_become_perfect_scores(self):
        row = answer_row()
        score = score_answer(row, QUESTION)
        for metric in ("faithfulness", "correctness", "citation_precision", "citation_support"):
            self.assertIsNone(score[metric])
        label = human_label(row)
        label["claims"][0]["citation_support"]["S2"] = None
        label["claims"][0]["faithful"] = None
        score = score_answer(row, QUESTION, label)
        self.assertEqual(score["citation_label_coverage"], 0.5)
        self.assertEqual(score["faithfulness_label_coverage"], 0)
        self.assertIsNone(score["faithfulness"])

    def test_unanswerable_false_abstentions_and_unknown_answerability(self):
        row = answer_row()
        for answerable, abstain, expected in [
            (False, True, 1),
            (False, False, 0),
            (True, True, 0),
            (True, False, 1),
            (None, True, None),
        ]:
            question = QUESTION | {"answerable": answerable}
            row["response"]["status"] = "insufficient_evidence" if abstain else "answered"
            row["response"]["statements"] = (
                [] if abstain else answer_row()["response"]["statements"]
            )
            score = score_answer(row, question)
            self.assertEqual(score["abstention_accuracy"], expected)
            if abstain:
                self.assertIsNone(score["faithfulness"])
                self.assertIsNone(score["citation_precision"])

    def test_stale_duplicate_and_mixed_source_labels_are_rejected(self):
        row = answer_row()
        label = human_label(row)
        run = {"per_query": [row]}
        for labels, source in [
            ([label, label], "human"),
            ([label | {"result_id": "stale"}], "human"),
            ([label], "llm"),
        ]:
            with self.assertRaises(ValueError):
                evaluate_answers(run, [QUESTION], labels, source=source)

    def test_annotation_integrity_checks(self):
        row = answer_row()
        for change in (
            {"answerable": False},
            {"correctness": 1},
            {"annotator": ""},
            {"claims": []},
        ):
            with self.assertRaises(ValueError):
                score_answer(row, QUESTION, human_label(row) | change)
        label = human_label(row)
        label["claims"][0]["citation_support"] = {"S1": True}
        with self.assertRaises(ValueError):
            score_answer(row, QUESTION, label)

    def test_optional_judge_uses_evidence_and_requires_reference_for_correctness(self):
        row = answer_row()
        label = human_label(row)
        label.update(source="llm", annotator="ollama/fixture", correctness=None)
        label["claims"][0]["correct"] = None
        llm = Mock(complete=Mock(return_value=json.dumps(label)))
        config = LLMConfig(model="fixture")
        result = judge_answer(row, QUESTION, llm, config)
        self.assertEqual(result["source"], "llm")
        messages = llm.complete.call_args.args[0]
        self.assertIn("Supporting excerpt", messages[1]["content"])
        label["correctness"] = True
        llm.complete.return_value = json.dumps(label)
        with self.assertRaisesRegex(ValueError, "reference"):
            judge_answer(row, QUESTION, llm, config)

    def test_answer_cli_exports_human_scores_without_calling_services(self):
        row = answer_row()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.json").write_text(json.dumps({"per_query": [row]}))
            (root / "questions.jsonl").write_text(json.dumps(QUESTION) + "\n")
            (root / "labels.jsonl").write_text(json.dumps(human_label(row)) + "\n")
            args = parser().parse_args(
                [
                    "evaluate-answers",
                    str(root / "run.json"),
                    "--questions",
                    str(root / "questions.jsonl"),
                    "--labels",
                    str(root / "labels.jsonl"),
                    "--output",
                    str(root / "scores"),
                ]
            )
            result = run(args)
            self.assertEqual(result["label_source"], "human")
            self.assertTrue((root / "scores/per_query.csv").exists())
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
