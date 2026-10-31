"""MedCPT encoders loaded only when explicitly constructed."""

from pathlib import Path

from .config import ARTICLE_MODEL, QUERY_MODEL
from .io import read_jsonl, write_jsonl


def article_input(chunk):
    return [chunk.get("doc_title") or chunk.get("title") or "", chunk["text"]]


class MedCPTEncoder:
    def __init__(self, model_name=ARTICLE_MODEL, *, device=None, pooling="cls", normalize=False):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be cls or mean")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.pooling, self.normalize = pooling, normalize

    def encode(self, inputs, *, batch_size=20, max_length=512):
        import numpy as np
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        batches = []
        with torch.inference_mode():
            for start in range(0, len(inputs), batch_size):
                encoded = self.tokenizer(
                    inputs[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**encoded).last_hidden_state
                if self.pooling == "cls":
                    vectors = hidden[:, 0, :]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                if self.normalize:
                    vectors = torch.nn.functional.normalize(vectors, dim=1)
                batches.append(vectors.cpu().numpy())
        return (
            np.concatenate(batches)
            if batches
            else np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        )

    def encode_query(self, query):
        return self.encode([query], max_length=64)[0].tolist()

    @classmethod
    def for_queries(cls, **kwargs):
        return cls(QUERY_MODEL, **kwargs)


def embed_children(children_path, embeddings_path, id_map_path, *, encoder=None, batch_size=20):
    import numpy as np

    chunks = list(read_jsonl(children_path))
    if not chunks:
        raise ValueError("No child chunks to embed")
    encoder = encoder or MedCPTEncoder()
    vectors = encoder.encode([article_input(c) for c in chunks], batch_size=batch_size)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match child count")
    Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(embeddings_path).open("wb") as handle:
        np.save(handle, vectors)
    write_jsonl(
        ({"chunk_id": c["chunk_id"], "parent_id": c.get("parent_id")} for c in chunks), id_map_path
    )
    return vectors.shape
