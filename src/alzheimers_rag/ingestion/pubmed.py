"""NCBI abstract acquisition with caller-supplied contact information."""

import re
import time

from ..config import DEFAULT_QUERY


class PubMedClient:
    def __init__(self, email, api_key=None):
        if not email:
            raise ValueError("Provide an NCBI contact email via --email or NCBI_EMAIL")
        self.email, self.api_key = email, api_key

    def _entrez(self):
        from Bio import Entrez

        Entrez.email = self.email
        Entrez.tool = "alzheimers-rag"
        Entrez.api_key = self.api_key
        return Entrez

    def search(self, query=DEFAULT_QUERY, limit=3000):
        if not 1 <= limit <= 9999:
            raise ValueError("PubMed search limit must be between 1 and 9999")
        entrez = self._entrez()
        with entrez.esearch(db="pubmed", term=query, retmax=limit) as handle:
            return list(entrez.read(handle)["IdList"])

    def abstracts(self, pmids, batch_size=100):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        entrez = self._entrez()
        for start in range(0, len(pmids), batch_size):
            with entrez.efetch(
                db="pubmed",
                id=",".join(pmids[start : start + batch_size]),
                rettype="abstract",
                retmode="xml",
            ) as handle:
                papers = entrez.read(handle)
            for paper in papers.get("PubmedArticle", []):
                citation = paper["MedlineCitation"]
                article = citation["Article"]
                date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                year = re.search(r"\b\d{4}\b", str(date.get("Year") or date.get("MedlineDate", "")))
                yield {
                    "pmid": str(citation["PMID"]),
                    "title": str(article.get("ArticleTitle", "")),
                    "abstract": " ".join(
                        map(str, article.get("Abstract", {}).get("AbstractText", []))
                    ),
                    "journal": str(article.get("Journal", {}).get("Title", "")),
                    "year": int(year[0]) if year else None,
                }
            time.sleep(0.11 if self.api_key else 0.34)
