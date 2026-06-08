from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K = 60

doc_tokenizer = AutoTokenizer.from_pretrained(
    "ncbi/MedCPT-Article-Encoder")
doc_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
doc_model.eval()

query_tokenizer = AutoTokenizer.from_pretrained(
    "ncbi/MedCPT-Query-Encoder")
query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
query_model.eval()

rerank_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
rerank_model = AutoModelForSequenceClassification.from_pretrained(
    "ncbi/MedCPT-Cross-Encoder"
).to(device)

rerank_model.eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    return torch.sum(
        token_embeddings * input_mask_expanded,
        1
    ) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9
    )


def encode_documents_batch(
    texts,
    batch_size=32
):
    all_embeddings = []

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding documents",
        unit="batch"
    ):
        batch = texts[i:i + batch_size]

        encoded = doc_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
        }

        with torch.no_grad():
            outputs = doc_model(**encoded)

        embeddings = mean_pooling(
            outputs,
            encoded["attention_mask"]
        )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        all_embeddings.extend(
            embeddings.cpu().tolist()
        )

    return all_embeddings


def encode_query(query):

    encoded_input = query_tokenizer(
        query,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        model_output = query_model(
            **encoded_input
        )

    embeddings = mean_pooling(
        model_output,
        encoded_input["attention_mask"]
    )

    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1
    )

    embeddings = embeddings.cpu().tolist()

    return embeddings[0]


def dense_search(collection, query, limit=10):

    q_vec = encode_query(query)

    results = collection.query.near_vector(
        near_vector=q_vec,
        limit=limit,
        return_properties=["pmid", "title", "text", "year", "journal"]
    )

    return results.objects


def sparse_search(collection, query, limit=10):
    results = collection.query.bm25(
        query=query,
        limit=limit,
        return_properties=["pmid", "title", "text", "year", "journal"]
    )

    return results.objects


def rrf_fusion(dense_results, sparse_results):

    scores = defaultdict(float)
    docs = {}

    for rank, obj in enumerate(dense_results):
        key = str(obj.properties["journal"])
        scores[key] += 1 / (K + rank)
        docs[key] = obj

    for rank, obj in enumerate(sparse_results):
        key = str(obj.properties["journal"])
        scores[key] += 1 / (K + rank)
        docs[key] = obj

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [docs[k] for k, _ in ranked]


def rerank(query, docs, limit=5):
    pairs = [(query, doc.properties['text']) for doc in docs]

    encoded = rerank_tokenizer(
        pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)

    with torch.no_grad():
        scores = rerank_model(**encoded).logits.squeeze()

    scored_docs = list(zip(docs, scores.cpu().tolist()))

    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    return ranked[:limit]


def build_context(docs):
    context = ""

    for i, (doc, score) in enumerate(docs):
        p = doc.properties
        context += f"""
        [DOCUMENT {i+1}]

        TITLE:
        {p["title"]}

        YEAR:
        {p['year']}

        JOURNAL:
        {p['journal']}

        CONTENT:
        {p['text']}

        """

    return context
