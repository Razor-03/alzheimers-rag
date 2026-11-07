"""Normalize known literature identifiers; never ask the LLM to invent source URLs."""

import re
from urllib.parse import quote


def source_metadata(record):
    pmid = str(record.get("pmid") or "").strip()
    if not pmid:
        match = re.fullmatch(r"PMID([0-9]+)", str(record.get("doc_id") or ""), re.IGNORECASE)
        pmid = match[1] if match else ""
    pmid = pmid if re.fullmatch(r"[0-9]+", pmid) else None
    pmc = str(record.get("pmc_id") or record.get("pmcid") or "").strip().upper()
    if not pmc and str(record.get("doc_id", "")).upper().startswith("PMC"):
        pmc = str(record["doc_id"]).upper()
    if pmc.isascii() and pmc.isdigit():
        pmc = "PMC" + pmc
    pmc = pmc if re.fullmatch(r"PMC[0-9]+", pmc) else None
    doi = str(record.get("doi") or "").strip()
    doi = doi if re.fullmatch(r"10\.[0-9]{4,9}/\S+", doi) else None
    url = None
    if pmc:
        url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc}/"
    elif pmid:
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    elif doi:
        url = "https://doi.org/" + quote(doi, safe="/")
    return {"pmid": pmid, "pmc_id": pmc, "doi": doi, "source_url": url}
