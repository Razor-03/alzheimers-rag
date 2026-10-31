"""Artifact checks run when the optional NumPy dependency is available."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from alzheimers_rag.embeddings import embed_children
from alzheimers_rag.io import read_jsonl, write_jsonl
from alzheimers_rag.storage.weaviate import index_children, validate_alignment


@unittest.skipUnless(importlib.util.find_spec("numpy"), "requires the models extra (NumPy)")
class ArtifactTests(unittest.TestCase):
    def test_alignment_rejects_reordered_or_invalid_vectors(self):
        import numpy as np

        children = [{"chunk_id": "a"}, {"chunk_id": "b"}]
        validate_alignment(children, children, np.zeros((2, 3)))
        for ids, vectors in [
            (children[::-1], np.zeros((2, 3))),
            (children, np.zeros((1, 3))),
            (children, np.full((2, 3), float("nan"))),
            (children, np.zeros((2, 0))),
        ]:
            with self.subTest(ids=ids, shape=vectors.shape):
                with self.assertRaises(ValueError):
                    validate_alignment(children, ids, vectors)
        with self.assertRaises(ValueError):
            validate_alignment([children[0]] * 2, [children[0]] * 2, np.zeros((2, 3)))

    def test_embedding_artifacts_preserve_title_and_row_order(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = [
                {"chunk_id": "b", "title": "Older schema", "text": "Body"},
                {"chunk_id": "a", "doc_title": "New schema", "text": "Other"},
            ]
            write_jsonl(chunks, root / "children.jsonl")
            encoder = Mock()
            encoder.encode.return_value = np.ones((2, 3))
            shape = embed_children(
                root / "children.jsonl", root / "vectors.npy", root / "ids.jsonl", encoder=encoder
            )
            self.assertEqual(shape, (2, 3))
            encoder.encode.assert_called_once_with(
                [["Older schema", "Body"], ["New schema", "Other"]], batch_size=20
            )
            self.assertEqual([r["chunk_id"] for r in read_jsonl(root / "ids.jsonl")], ["b", "a"])
            np.testing.assert_equal(np.load(root / "vectors.npy"), np.ones((2, 3)))

    def test_invalid_artifacts_fail_before_collection_access(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            children = [{"chunk_id": "a", "text": "A"}, {"chunk_id": "b", "text": "B"}]
            write_jsonl(children, root / "children.jsonl")
            write_jsonl(children[::-1], root / "ids.jsonl")
            np.save(root / "vectors.npy", np.zeros((2, 3)))
            with patch("alzheimers_rag.storage.weaviate.ensure_collection") as ensure:
                with self.assertRaises(ValueError):
                    index_children(
                        Mock(), root / "children.jsonl", root / "vectors.npy", root / "ids.jsonl"
                    )
                ensure.assert_not_called()
