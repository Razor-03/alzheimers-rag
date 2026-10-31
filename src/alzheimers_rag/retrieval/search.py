"""Hybrid search with explicit collection and query encoder dependencies."""

from ..config import RetrievalConfig


class HybridRetriever:
    def __init__(self, collection, encoder):
        self.collection, self.encoder = collection, encoder

    def search(self, query, config=None):
        from weaviate.classes.query import Filter, MetadataQuery

        cfg = config or RetrievalConfig()
        filters = None
        if cfg.section_type_filter:
            filters = Filter.by_property("section_type").equal(cfg.section_type_filter)
        if cfg.kwd_filter:
            keyword = Filter.by_property("kwd").contains_any([cfg.kwd_filter])
            filters = keyword if filters is None else filters & keyword
        response = self.collection.query.hybrid(
            query=query,
            vector=self.encoder.encode_query(query) if cfg.alpha else None,
            alpha=cfg.alpha,
            limit=cfg.top_k,
            filters=filters,
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )
        return [
            {
                **obj.properties,
                "score": obj.metadata.score,
                "explain_score": obj.metadata.explain_score,
            }
            for obj in response.objects
        ]


def rrf_fusion(*result_lists, k=60):
    """Fuse by chunk identity, preserving different papers from the same journal."""
    if k < 1:
        raise ValueError("k must be positive")
    scores, docs = {}, {}
    for results in result_lists:
        seen = set()
        for rank, hit in enumerate(results, 1):
            key = hit["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            docs[key] = hit
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
    return [
        {**docs[key], "score": scores[key]} for key in sorted(scores, key=scores.get, reverse=True)
    ]
