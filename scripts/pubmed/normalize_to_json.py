import re
import unicodedata
import json
from Bio import Entrez


def extract_paper_data(article):

    citation = article["MedlineCitation"]
    article_data = citation["Article"]
    pmid = str(citation.get("PMID", ""))
    title = str(article_data.get("ArticleTitle", ""))
    abstract = ""

    if "Abstract" in article_data:
        abstract_sections = article_data["Abstract"]["AbstractText"]

        abstract = " ".join(str(x) for x in abstract_sections)

    journal = ""

    if "Journal" in article_data:
        journal = str(
            article_data["Journal"]
            .get("Title", "")
        )

    year = ""

    try:
        year = (
            article_data["Journal"]["JournalIssue"]
            ["PubDate"]["Year"]
        )
    except:
        pass

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year
    }


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


if __name__ == "__main__":
    with open("data/raw/pubmed_batch.xml", "rb") as f:
        papers = Entrez.read(f)

    processed_papers = []

    for article in papers["PubmedArticle"]:
        paper = extract_paper_data(article)
        paper["title"] = clean_text(paper["title"])
        paper["abstract"] = clean_text(
            paper["abstract"]) if paper["abstract"] != "" else None
        paper["text"] = paper["title"] + " " + (paper["abstract"] or "")

        processed_papers.append(paper)

    print(processed_papers[2])

    with open("data/corpus/alzheimers_corpus.jsonl", "w", encoding="utf-8") as f:
        for paper in processed_papers:
            f.write(json.dumps(paper) + "\n")
