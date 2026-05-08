from Bio import Entrez

Entrez.email = "hr75155@gmail.com"

handle = Entrez.esearch(
    db="pubmed",
    term="Alzheimer Disease AND (amyloid OR tau OR biomarker OR treatment)",
    retmax=1000,
)

record = Entrez.read(handle)

pmids = record["IdList"]

fetch_handle = Entrez.efetch(
    db="pubmed",
    id=",".join(pmids),
    rettype="abstract",
    retmode="xml",
)

raw_xml = fetch_handle.read()

with open("data/raw/pubmed_batch.xml", "wb") as f:
    f.write(raw_xml)