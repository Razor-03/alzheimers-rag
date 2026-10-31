"""UTF-8 JSONL interchange shared by pipeline stages."""

import json
import os
import tempfile
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def raw_documents(path):
    """Accept documents, BioC collections, or lists of either per line."""
    for value in read_jsonl(path):
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict) and "passages" in item:
                yield item
            elif isinstance(item, dict) and "documents" in item:
                yield from item["documents"]
            else:
                raise ValueError(f"{path}: expected a BioC document or collection")
