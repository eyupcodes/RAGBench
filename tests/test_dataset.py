"""Tests for dataset loading utilities."""

import json
import pytest
from ragbench.dataset import (
    DatasetLoadError,
    load_dataset_from_dict,
    load_dataset_from_directory,
    load_dataset_from_json,
)


def test_load_dataset_from_valid_dict():
    data = {
        "name": "sample_rag",
        "description": "A sample RAG benchmark dataset",
        "documents": [
            {"id": "doc1", "text": "Python is a popular programming language.", "metadata": {"topic": "python"}},
            {"id": "doc2", "text": "Rust focuses on performance and memory safety.", "metadata": {"topic": "rust"}},
        ],
        "queries": [
            {"id": "q1", "query": "Which language emphasizes memory safety?", "relevant_doc_ids": ["doc2"]},
            {"id": "q2", "query": "Tell me about Python", "relevant_doc_ids": ["doc1"]},
        ]
    }
    ds = load_dataset_from_dict(data)
    assert ds.name == "sample_rag"
    assert len(ds.documents) == 2
    assert len(ds.queries) == 2
    assert ds.queries[0].relevant_doc_ids == ["doc2"]


def test_load_dataset_invalid_root_type():
    with pytest.raises(DatasetLoadError, match="must be a JSON object"):
        load_dataset_from_dict(["not", "a", "dict"])


def test_load_dataset_missing_documents():
    with pytest.raises(DatasetLoadError, match="must contain a 'documents' list"):
        load_dataset_from_dict({"queries": []})


def test_load_dataset_missing_queries():
    with pytest.raises(DatasetLoadError, match="must contain a 'queries' list"):
        load_dataset_from_dict({"documents": []})


def test_load_dataset_from_json_file(tmp_path):
    data = {
        "name": "file_dataset",
        "documents": [{"id": "d1", "text": "Hello world"}],
        "queries": [{"id": "q1", "query": "Hello", "relevant_doc_ids": ["d1"]}]
    }
    json_file = tmp_path / "dataset.json"
    json_file.write_text(json.dumps(data), encoding="utf-8")

    ds = load_dataset_from_json(json_file)
    assert ds.name == "file_dataset"
    assert len(ds.documents) == 1


def test_load_dataset_from_missing_file():
    with pytest.raises(DatasetLoadError, match="not found"):
        load_dataset_from_json("non_existent_file_xyz.json")


def test_load_dataset_from_invalid_json_content(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ broken json", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Invalid JSON"):
        load_dataset_from_json(bad_file)


def test_load_dataset_from_directory(tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "intro.md").write_text("# Intro\nThis is introductory text.", encoding="utf-8")
    (doc_dir / "guide.txt").write_text("Detailed guide content here.", encoding="utf-8")

    queries_data = {
        "queries": [
            {"id": "q1", "query": "How to guide?", "relevant_doc_ids": ["guide"]}
        ]
    }
    q_file = tmp_path / "queries.json"
    q_file.write_text(json.dumps(queries_data), encoding="utf-8")

    ds = load_dataset_from_directory(doc_dir=doc_dir, queries_file=q_file, dataset_name="dir_test")
    assert ds.name == "dir_test"
    assert len(ds.documents) == 2
    assert len(ds.queries) == 1
    doc_ids = {d.id for d in ds.documents}
    assert "intro" in doc_ids
    assert "guide" in doc_ids
