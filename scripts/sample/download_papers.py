import requests
import json
import time
from bs4 import BeautifulSoup


def fetch_pmc_xml(pmcid):
    """
    Fetch abstract from PMC XML API
    """
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pmc",
            "id": pmcid,
            "rettype": "xml",
            "retmode": "xml"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml-xml')

        # Extract abstract
        abstract = soup.find('abstract')
        if abstract:
            text = abstract.get_text(strip=True)
            return text
        return None

    except Exception as e:
        print(f"  Error fetching PMC{pmcid}: {e}")
        return None


def download_alzheimers_abstracts(num_papers=100):
    """
    Download Alzheimer's papers from Europe PMC and fetch abstracts via XML API
    """
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    query = "Alzheimer's disease"

    papers = []
    page_size = 50
    pages_needed = (num_papers // page_size) + 1

    print(f"Fetching {num_papers} Alzheimer's papers from Europe PMC...")

    # Step 1: Get paper metadata from Europe PMC
    for page in range(1, pages_needed + 1):
        params = {
            "query": query,
            "pageSize": page_size,
            "pageNumber": page,
            "format": "json",
            # "isOpenAccess": "Y"
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            if "resultList" not in data or not data["resultList"]["result"]:
                break

            for paper in data["resultList"]["result"]:
                # Only process papers that have PMCID
                if not paper.get("pmcid"):
                    continue

                paper_info = {
                    "title": paper.get("title"),
                    "authors": paper.get("authorString", ""),
                    "year": paper.get("pubYear"),
                    "journal": paper.get("journalTitle"),
                    "pmcid": paper.get("pmcid"),
                    "doi": paper.get("doi"),
                    "abstract": None  # Will be filled below
                }
                papers.append(paper_info)

            print(
                f"Page {page}: Found {len(data['resultList']['result'])} papers. Total so far: {len(papers)}")
            time.sleep(0.3)

            if len(papers) >= num_papers:
                break

        except Exception as e:
            print(f"Error on page {page}: {e}")
            continue

    # Step 2: Fetch abstracts for each paper
    print(f"\nFetching abstracts for {len(papers[:num_papers])} papers...")

    papers_with_abstracts = []
    for i, paper in enumerate(papers[:num_papers]):
        print(
            f"[{i+1}/{min(len(papers), num_papers)}] {paper['title'][:60]}...", end=" ")

        pmc_id = paper['pmcid'].replace("PMC", "")

        abstract = fetch_pmc_xml(pmc_id)

        if abstract and len(abstract) > 100:  # Only keep if substantial abstract
            paper['abstract'] = abstract
            papers_with_abstracts.append(paper)
            print("✓")
        else:
            print(f"✗ (no abstract for {pmc_id})")

        time.sleep(0.5)  # Be nice to NCBI servers

    # Step 3: Save to JSON
    with open("alzheimers_papers.json", "w") as f:
        json.dump(papers_with_abstracts, f, indent=2)

    print(f"\n✅ Downloaded {len(papers_with_abstracts)} papers with abstracts")
    print(f"Saved to: alzheimers_papers.json")

    return papers_with_abstracts


if __name__ == "__main__":
    papers = download_alzheimers_abstracts(num_papers=100)

    # Show sample
    print("\n" + "="*70)
    print("Sample paper:")
    print("="*70)
    if papers:
        p = papers[0]
        print(f"Title: {p['title']}")
        print(f"Year: {p['year']}")
        print(f"Journal: {p['journal']}")
        print(f"Authors: {p['authors'][:80]}...")
        print(f"\nAbstract:\n{p['abstract'][:400]}...")
