from transformers import AutoTokenizer, AutoModel
import torch


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


def encode_document(text):

    encoded_input = doc_tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():

        model_output = doc_model(
            **encoded_input
        )

    embeddings = mean_pooling(
        model_output,
        encoded_input["attention_mask"]
    )

    return embeddings[0]


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

    return embeddings[0]


if __name__ == "__main__":
    doc_tokenizer = AutoTokenizer.from_pretrained(
        "ncbi/MedCPT-Article-Encoder")
    doc_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")

    query_tokenizer = AutoTokenizer.from_pretrained(
        "ncbi/MedCPT-Query-Encoder")
    query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")

    query = "Does amyloid beta contribute to cognitive decline?"

    doc = """
    Amyloid-beta accumulation is associated with
    progressive neurodegeneration and memory impairment
    in Alzheimer's disease.
    """

    q_emb = encode_query(query)

    d_emb = encode_document(doc)

    similarity = torch.nn.functional.cosine_similarity(
        q_emb,
        d_emb,
        dim=0
    )

    print(similarity.item())
