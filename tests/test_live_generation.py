"""Opt-in, read-only test against an existing Weaviate index and an LLM service."""

import json
import os
import unittest
from urllib.request import Request, urlopen

from alzheimers_rag.cli import parser, run


@unittest.skipUnless(os.getenv("RAG_RUN_LIVE_TESTS") == "1", "live services explicitly disabled")
class LiveGenerationTests(unittest.TestCase):
    def test_question_through_existing_index_and_model(self):
        model = os.getenv("RAG_LLM_MODEL")
        collection = os.getenv("RAG_TEST_COLLECTION")
        self.assertTrue(model, "Set RAG_LLM_MODEL to a model installed/available at your endpoint")
        self.assertTrue(collection, "Set RAG_TEST_COLLECTION to an existing compatible collection")
        argv = [
            "--data-dir",
            os.getenv("RAG_TEST_DATA_DIR", "data"),
            "query",
            os.getenv("RAG_TEST_QUESTION", "What is reported about tau and cognitive decline?"),
            "--model",
            model,
            "--collection",
            collection,
            "--host",
            os.getenv("RAG_TEST_HOST", "localhost"),
            "--port",
            os.getenv("RAG_TEST_PORT", "8080"),
            "--grpc-port",
            os.getenv("RAG_TEST_GRPC_PORT", "50051"),
            "--context",
            os.getenv("RAG_TEST_CONTEXT", "child"),
        ]
        if os.getenv("RAG_TEST_RERANK") == "1":
            argv.append("--rerank")
        result = run(parser().parse_args(argv))
        self.assertEqual(result["status"], "answered", result["limitations"])
        self.assertTrue(result["citations"])
        admitted = {s["source_id"] for s in result["evidence"]}
        for statement in result["statements"]:
            self.assertTrue(set(statement["source_ids"]) <= admitted)
        for citation in result["citations"]:
            self.assertTrue(citation["chunk_ids"])
            self.assertTrue(citation["text"].strip())
        self.assertLessEqual(
            result["budget"]["context_tokens"], result["budget"]["context_token_budget"]
        )
        self.assertLessEqual(result["budget"]["input_tokens"], result["budget"]["input_limit"])


@unittest.skipUnless(
    os.getenv("RAG_RUN_LIVE_API_TESTS") == "1", "live HTTP service explicitly disabled"
)
class LiveAPITests(unittest.TestCase):
    def test_running_http_service_returns_a_cited_answer(self):
        base_url = os.getenv("RAG_TEST_API_URL", "http://127.0.0.1:8000").rstrip("/")
        with urlopen(base_url + "/ready", timeout=10) as response:
            self.assertEqual(json.load(response)["status"], "ready")
        payload = {
            "query": os.getenv(
                "RAG_TEST_QUESTION", "What is reported about tau and cognitive decline?"
            ),
            "rerank": os.getenv("RAG_TEST_RERANK") == "1",
            "context": os.getenv("RAG_TEST_CONTEXT", "child"),
        }
        request = Request(
            base_url + "/query",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        self.assertEqual(result["status"], "answered", result["limitations"])
        self.assertTrue(result["citations"])
        admitted = {s["source_id"] for s in result["evidence"]}
        for statement in result["statements"]:
            self.assertTrue(statement["source_ids"])
            self.assertTrue(set(statement["source_ids"]) <= admitted)
        for source in result["citations"]:
            self.assertTrue(source["chunk_ids"])
            self.assertTrue(source["text"].strip())
        self.assertLessEqual(
            result["budget"]["context_tokens"], result["budget"]["context_token_budget"]
        )
        self.assertLessEqual(result["budget"]["input_tokens"], result["budget"]["input_limit"])
