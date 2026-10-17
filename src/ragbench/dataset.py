"""Dataset loading utilities for RAGBench."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union
from ragbench.models import BenchmarkDataset, Document, Query


class DatasetLoadError(Exception):
    """Raised when dataset loading fails."""
    pass


def _ensure_unique_ids(kind: str, ids: List[str]) -> None:
    """Raise DatasetLoadError if any ID occurs more than once."""
    seen = set()
    duplicates = []
    for item_id in ids:
        if item_id in seen and item_id not in duplicates:
            duplicates.append(item_id)
        seen.add(item_id)
    if duplicates:
        raise DatasetLoadError(f"Duplicate {kind} IDs: {', '.join(map(repr, duplicates))}")


def load_dataset_from_dict(data: Dict[str, Any]) -> BenchmarkDataset:
    """Load a BenchmarkDataset from a raw Python dictionary with schema validation."""
    if not isinstance(data, dict):
        raise DatasetLoadError("Dataset root must be a JSON object (dictionary)")

    if "documents" not in data or not isinstance(data["documents"], list):
        raise DatasetLoadError("Dataset must contain a 'documents' list")

    if "queries" not in data or not isinstance(data["queries"], list):
        raise DatasetLoadError("Dataset must contain a 'queries' list")

    try:
        documents = [Document(**d) for d in data["documents"]]
        queries = [Query(**q) for q in data["queries"]]
    except Exception as e:
        raise DatasetLoadError(f"Failed to parse dataset records: {e}") from e

    _ensure_unique_ids("document", [d.id for d in documents])
    _ensure_unique_ids("query", [q.id for q in queries])
    try:
        return BenchmarkDataset(
            name=data.get("name", "unnamed_dataset"),
            description=data.get("description"),
            documents=documents,
            queries=queries,
        )
    except Exception as e:
        raise DatasetLoadError(f"Failed to parse dataset records: {e}") from e


def load_dataset_from_json(file_path: Union[str, Path]) -> BenchmarkDataset:
    """Load a BenchmarkDataset from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise DatasetLoadError(f"Dataset file not found: {path}")
    if not path.is_file():
        raise DatasetLoadError(f"Path is not a file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetLoadError(f"Invalid JSON in {path}: {e}") from e
    except Exception as e:
        raise DatasetLoadError(f"Error reading {path}: {e}") from e

    return load_dataset_from_dict(data)


def load_queries_from_json(file_path: Union[str, Path]) -> List[Query]:
    """Load queries list from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise DatasetLoadError(f"Queries file not found: {path}")
    if not path.is_file():
        raise DatasetLoadError(f"Path is not a file: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetLoadError(f"Invalid JSON in {path}: {e}") from e
    except Exception as e:
        raise DatasetLoadError(f"Error reading {path}: {e}") from e

    if isinstance(data, list):
        raw_queries = data
    elif isinstance(data, dict):
        if "queries" not in data or not isinstance(data["queries"], list):
            raise DatasetLoadError("Queries dictionary must contain a 'queries' list")
        raw_queries = data["queries"]
    else:
        raise DatasetLoadError("Queries JSON must be a list or object with 'queries' key")

    try:
        queries = [Query(**q) for q in raw_queries]
    except Exception as e:
        raise DatasetLoadError(f"Failed to parse queries list: {e}") from e
    _ensure_unique_ids("query", [q.id for q in queries])
    return queries


def load_dataset_from_directory(
    doc_dir: Union[str, Path],
    queries_file: Union[str, Path],
    dataset_name: str = "dir_dataset"
) -> BenchmarkDataset:
    """Load documents from individual text/markdown files and queries from a JSON file."""
    doc_path = Path(doc_dir)
    if not doc_path.exists() or not doc_path.is_dir():
        raise DatasetLoadError(f"Documents directory not found: {doc_path}")

    documents: List[Document] = []
    sources: Dict[str, Path] = {}
    for file_p in sorted(doc_path.glob("**/*")):
        if file_p.is_file() and file_p.suffix.lower() in [".txt", ".md", ".json"]:
            if file_p.suffix.lower() == ".json":
                continue  # Skip JSON config files in doc dir
            doc_id = file_p.stem
            if doc_id in sources:
                raise DatasetLoadError(
                    f"Duplicate document ID {doc_id!r} (derived from the file name): "
                    f"{sources[doc_id]} and {file_p}"
                )
            sources[doc_id] = file_p
            try:
                content = file_p.read_text(encoding="utf-8")
                documents.append(Document(id=doc_id, text=content, metadata={"source_path": str(file_p)}))
            except Exception as e:
                raise DatasetLoadError(f"Failed to read file {file_p}: {e}") from e

    queries = load_queries_from_json(queries_file)
    return BenchmarkDataset(
        name=dataset_name,
        description=f"Loaded from {doc_dir}",
        documents=documents,
        queries=queries,
    )
