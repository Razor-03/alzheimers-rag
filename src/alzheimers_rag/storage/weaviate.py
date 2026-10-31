"""Collection schema and validated imports for self-provided vectors."""

from uuid import NAMESPACE_URL, uuid5

from ..io import read_jsonl

COLLECTION_NAME = "AlzheimersChunk"


def connect(*, host="localhost", port=8080, grpc_port=50051):
    import weaviate

    return weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)


def ensure_collection(client, name=COLLECTION_NAME):
    from weaviate.classes.config import Configure, DataType, Property, VectorDistances

    if client.collections.exists(name):
        return client.collections.get(name)
    types = {
        "chunk_id": DataType.TEXT,
        "parent_id": DataType.TEXT,
        "doc_id": DataType.TEXT,
        "doc_title": DataType.TEXT,
        "text": DataType.TEXT,
        "section_type": DataType.TEXT,
        "heading_path": DataType.TEXT_ARRAY,
        "kwd": DataType.TEXT_ARRAY,
        "is_atomic": DataType.BOOL,
        "token_count": DataType.NUMBER,
    }
    return client.collections.create(
        name=name,
        vector_config=Configure.Vectors.self_provided(),
        vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.DOT),
        properties=[Property(name=key, data_type=value) for key, value in types.items()],
    )


def chunk_properties(chunk):
    """Normalize older notebook artifacts at the storage boundary."""
    path = chunk.get("heading_path") or []
    return {
        "chunk_id": chunk["chunk_id"],
        "parent_id": chunk.get("parent_id"),
        "doc_id": chunk.get("doc_id"),
        "doc_title": chunk.get("doc_title") or chunk.get("title") or "",
        "text": chunk["text"],
        "section_type": chunk.get("section_type") or chunk.get("canonical_section_type", ""),
        "heading_path": path.split(" > ") if isinstance(path, str) else path,
        "kwd": chunk.get("kwd") or [],
        "is_atomic": chunk.get("is_atomic", False),
        "token_count": chunk.get("token_count", chunk.get("n_tokens", 0)),
    }


def validate_alignment(children, id_map, vectors):
    import numpy as np

    if (
        vectors.ndim != 2
        or not len(children)
        or len(children) != len(id_map)
        or len(children) != len(vectors)
    ):
        raise ValueError("Children, ID map, and embedding rows must match and be nonempty")
    ids = [c["chunk_id"] for c in children]
    if len(set(ids)) != len(ids) or ids != [c["chunk_id"] for c in id_map]:
        raise ValueError("ID map must match child order exactly, with unique chunk IDs")
    if not np.isfinite(vectors).all() or vectors.shape[1] < 1:
        raise ValueError("Embeddings must have finite values and a nonzero dimension")


def index_children(client, children_path, embeddings_path, id_map_path, name=COLLECTION_NAME):
    import numpy as np

    children = list(read_jsonl(children_path))
    id_map = list(read_jsonl(id_map_path))
    vectors = np.load(embeddings_path, allow_pickle=False)
    validate_alignment(children, id_map, vectors)
    collection = ensure_collection(client, name)
    with collection.batch.dynamic() as batch:
        for chunk, vector in zip(children, vectors):
            batch.add_object(
                properties=chunk_properties(chunk),
                vector=vector.tolist(),
                uuid=str(uuid5(NAMESPACE_URL, f"{name}/{chunk['chunk_id']}")),
            )
    if collection.batch.failed_objects:
        raise RuntimeError(f"{len(collection.batch.failed_objects)} chunks failed to import")
    return len(children)
