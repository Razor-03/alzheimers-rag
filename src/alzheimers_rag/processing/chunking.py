"""Sentence-aware token windows and parent/child provenance."""

import re
from dataclasses import dataclass

from ..config import ARTICLE_MODEL

ATOMIC_TYPES = {"abstract", "figure_footnote", "data_availability", "title", "keywords"}
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
GLOSSARY_RE = re.compile(r"\b[A-Z][A-Za-z0-9\-]{1,8}:\s")
SENTENCE_END_RE = re.compile(r"[.!?]\s+[A-Z]")


@dataclass(frozen=True)
class ChunkConfig:
    max_tokens: int = 480
    overlap_tokens: int = 70

    def __post_init__(self):
        if self.max_tokens < 1 or not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("Require 0 <= overlap_tokens < max_tokens")


def split_sentences(text):
    from nltk.tokenize import sent_tokenize

    return [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]


def looks_like_glossary_or_metadata(text):
    hits = len(GLOSSARY_RE.findall(text))
    return hits >= 3 and hits > len(SENTENCE_END_RE.findall(text))


class MedCPTChunker:
    def __init__(self, config=None, *, tokenizer=None, sentence_splitter=split_sentences):
        self.config = config or ChunkConfig()
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
        self.tokenizer = tokenizer
        self.split_sentences = sentence_splitter

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_section(self, section):
        text = section["text"].strip()
        if not text or len(CJK_RE.findall(text)) / len(text) > 0.15:
            return []
        if looks_like_glossary_or_metadata(text):
            return []
        limit = self.config.max_tokens
        if self.count_tokens(text) <= limit:
            return [{"text": text, "n_tokens": self.count_tokens(text)}]
        units = []
        for sentence in self.split_sentences(text) or [text]:
            ids = self.tokenizer.encode(sentence, add_special_tokens=False)
            if len(ids) <= limit:
                units.append(sentence)
            else:
                # Preserve the tail of sentences longer than one model window.
                units.extend(
                    self.tokenizer.decode(ids[i : i + limit], skip_special_tokens=True)
                    for i in range(0, len(ids), limit)
                )
        overlap = (
            0 if section["canonical_section_type"] in ATOMIC_TYPES else self.config.overlap_tokens
        )
        chunks, current = [], []
        for unit in units:
            if current and self.count_tokens(" ".join([*current, unit])) > limit:
                chunks.append(self._finalize(current))
                tail = []
                for previous in reversed(current):
                    candidate = [previous, *tail]
                    if self.count_tokens(" ".join(candidate)) > overlap:
                        break
                    tail = candidate
                current = tail
                while current and self.count_tokens(" ".join([*current, unit])) > limit:
                    current.pop(0)
            current.append(unit)
        if current:
            chunks.append(self._finalize(current))
        return chunks

    def _finalize(self, sentences):
        text = " ".join(sentences)
        return {"text": text, "n_tokens": self.count_tokens(text)}


def build_parents_and_children(doc, chunker):
    metadata = {key: doc.get(key) for key in ("doc_id", "pmc_id", "pmid", "doi", "year")}
    metadata.update(doc_title=doc.get("title") or "", kwd=doc.get("kwd") or [])
    parents, children = [], []
    for section in doc["sections"]:
        section_type = section["canonical_section_type"]
        common = {
            **metadata,
            "parent_id": section["section_id"],
            "heading_path": section["heading_path"],
            "section_type": section_type,
            "canonical_section_type": section_type,
        }
        parents.append({**common, "text": section["text"], "n_words": section["n_words"]})
        chunks = chunker.chunk_section(section)
        for index, chunk in enumerate(chunks):
            children.append(
                {
                    **common,
                    **chunk,
                    "chunk_id": f"{section['section_id']}_c{index}",
                    "token_count": chunk["n_tokens"],
                    "is_atomic": section_type in ATOMIC_TYPES,
                    "chunk_index": index,
                    "n_chunks_in_parent": len(chunks),
                }
            )
    return parents, children
