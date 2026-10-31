"""File-oriented orchestration; algorithms live in their own modules."""

from .io import raw_documents, read_jsonl, write_jsonl
from .processing.chunking import MedCPTChunker, build_parents_and_children
from .processing.cleaning import process_document


def clean_corpus(input_path, output_path):
    return write_jsonl((process_document(doc) for doc in raw_documents(input_path)), output_path)


def chunk_corpus(input_path, parents_path, children_path, *, chunker=None):
    chunker = chunker or MedCPTChunker()
    parents, children = [], []
    for doc in read_jsonl(input_path):
        p, c = build_parents_and_children(doc, chunker)
        parents.extend(p)
        children.extend(c)
    return write_jsonl(parents, parents_path), write_jsonl(children, children_path)
