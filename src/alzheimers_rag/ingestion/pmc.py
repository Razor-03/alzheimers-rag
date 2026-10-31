"""PMID conversion and open-access BioC acquisition."""

import logging
import time

from ..io import write_jsonl

logger = logging.getLogger(__name__)
ID_CONVERSION_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
BIOC_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi"


def pmcids_from_pmids(pmids, email, *, session=None):
    import requests

    session = session or requests
    ids = []
    for start in range(0, len(pmids), 100):
        response = session.get(
            ID_CONVERSION_URL,
            timeout=30,
            params={
                "tool": "alzheimers-rag",
                "email": email,
                "ids": ",".join(pmids[start : start + 100]),
                "format": "json",
            },
        )
        response.raise_for_status()
        ids.extend(r["pmcid"] for r in response.json().get("records", []) if "pmcid" in r)
        time.sleep(0.34)
    return list(dict.fromkeys(ids))


def download_papers(pmcids, output_path, *, session=None):
    import requests

    session = session or requests

    def records():
        for pmcid in dict.fromkeys(pmcids):
            pmcid = "PMC" + str(pmcid).upper().removeprefix("PMC")
            response = session.get(f"{BIOC_URL}/BioC_json/{pmcid}/unicode", timeout=30)
            time.sleep(0.34)
            if response.status_code in (403, 404):
                logger.warning("Skipping unavailable open-access paper %s", pmcid)
                continue
            response.raise_for_status()
            yield response.json()

    return write_jsonl(records(), output_path)
