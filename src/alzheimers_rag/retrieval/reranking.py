"""Optional cross-encoder scoring, independent of retrieval and storage."""

from ..config import RERANK_MODEL, RerankConfig


class MedCPTReranker:
    def __init__(self, *, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        self.model.to(self.device).eval()

    def rerank(self, query, hits, config=None):
        import torch

        cfg = config or RerankConfig()
        scored = []
        with torch.inference_mode():
            for start in range(0, len(hits), cfg.batch_size):
                batch = hits[start : start + cfg.batch_size]
                encoded = self.tokenizer(
                    [[query, h["text"]] for h in batch],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                scores = self.model(**encoded).logits.reshape(-1).cpu().tolist()
                scored.extend(
                    {**hit, "rerank_score": score, "original_hybrid_score": hit.get("score")}
                    for hit, score in zip(batch, scores)
                )
        return sorted(scored, key=lambda h: h["rerank_score"], reverse=True)[: cfg.rerank_top_k]
