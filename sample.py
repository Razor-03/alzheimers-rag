import requests
from bs4 import BeautifulSoup

def fetch_pmc_xml(pmcid):
    """
    Fetch paper metadata + abstract from PMC XML API (faster)
    """
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pmc",
            "id": pmcid,
            "rettype": "xml",
            "retmode": "xml"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml-xml')
        print(soup.prettify())  # Debug: print the XML content
        
        # Extract abstract
        abstract = soup.find('abstract')
        if abstract:
            text = abstract.get_text(strip=True)
            return text
        return None
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    
abstract = fetch_pmc_xml("2943100")
print(abstract)