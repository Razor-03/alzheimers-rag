import json
import torch
from transformers import AutoTokenizer, AutoModel
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter

# NOT the article encoder — different training objective
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
COLLECTION_NAME = "AlzheimersChunk"
# MedCPT queries are typically short (this is the model's expected query length — verify against model card)
MAX_LENGTH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
model = AutoModel.from_pretrained(QUERY_MODEL_NAME).to(DEVICE)
model.eval()

client = weaviate.connect_to_local()
collection = client.collections.get(COLLECTION_NAME)


def embed_query(query_text: str):
    with torch.no_grad():
        encoded = tokenizer(
            [query_text],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        outputs = model(**encoded)
        # [CLS] pooling — same reasoning as article encoder
        embedding = outputs.last_hidden_state[:, 0, :]
        return embedding.cpu().numpy()[0].tolist()


def hybrid_search(query_text, top_k=10, alpha=0.5, section_type_filter=None, kwd_filter=None):
    """
    alpha: 0 = pure BM25 keyword search, 1 = pure vector search, 0.5 = balanced.
    Sweep this once you have real eval queries — don't trust 0.5 as final.
    """
    query_vector = embed_query(query_text)

    filters = None
    if section_type_filter:
        filters = Filter.by_property("section_type").equal(section_type_filter)
    if kwd_filter:
        kwd_f = Filter.by_property("kwd").contains_any([kwd_filter])
        filters = kwd_f if filters is None else (filters & kwd_f)

    response = collection.query.hybrid(
        query=query_text,          # feeds the BM25 half
        vector=query_vector,       # feeds the vector half — MedCPT query embedding
        alpha=alpha,
        limit=top_k,
        filters=filters,
        return_metadata=wvc.query.MetadataQuery(
            score=True, explain_score=True),
    )

    results = []
    for obj in response.objects:
        results.append({
            "chunk_id": obj.properties.get("chunk_id"),
            "parent_id": obj.properties.get("parent_id"),
            "text": obj.properties.get("text"),
            "doc_title": obj.properties.get("doc_title"),
            "heading_path": obj.properties.get("heading_path"),
            "section_type": obj.properties.get("section_type"),
            "score": obj.metadata.score,
            "explain_score": obj.metadata.explain_score,
        })
    return results
