"""Offline regression tests for failures during PMC paper downloads."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from alzheimers_rag.ingestion.pmc import BIOC_URL, download_papers
from alzheimers_rag.io import read_jsonl


@unittest.skipUnless(importlib.util.find_spec("requests"), "requires the ingestion extra")
class PMCDownloadTests(unittest.TestCase):
    def test_failed_paper_does_not_stop_remaining_downloads(self):
        import requests

        malformed = requests.Response()
        malformed.status_code = 200
        malformed._content = b"[Error: paper unavailable]"
        server_error = requests.Response()
        server_error.status_code = 500
        forbidden = requests.Response()
        forbidden.status_code = 403
        missing = requests.Response()
        missing.status_code = 404
        failures = [
            ("invalid JSON", malformed),
            ("HTTP error", server_error),
            ("forbidden", forbidden),
            ("missing", missing),
            ("timeout", requests.Timeout("request timed out")),
            ("connection error", requests.ConnectionError("connection failed")),
        ]
        for name, failure in failures:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                first = [{"documents": [{"id": "PMC1", "passages": []}]}]
                last = [{"documents": [{"id": "PMC3", "passages": []}]}]
                session = Mock()
                session.get.side_effect = [
                    Mock(status_code=200, json=Mock(return_value=first)),
                    failure,
                    Mock(status_code=200, json=Mock(return_value=last)),
                ]
                output = Path(directory) / "papers.jsonl"
                with (
                    patch("alzheimers_rag.ingestion.pmc.time.sleep") as sleep,
                    self.assertLogs("alzheimers_rag.ingestion.pmc", level="WARNING") as logs,
                ):
                    count = download_papers(["PMC1", "PMC2", "PMC3"], output, session=session)

                self.assertEqual(count, 2)
                self.assertEqual(list(read_jsonl(output)), [first, last])
                self.assertIn("PMC2", "\n".join(logs.output))
                self.assertEqual(session.get.call_count, 3)
                session.get.assert_called_with(f"{BIOC_URL}/BioC_json/PMC3/unicode", timeout=30)
                self.assertEqual(sleep.call_count, 3)
                self.assertEqual(list(Path(directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
