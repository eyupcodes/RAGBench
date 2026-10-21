"""Tests for dataset loading utilities."""

import json

import pytest

from ragbench.dataset import (
    DatasetLoadError,
    load_dataset_from_dict,
    load_dataset_from_directory,
    load_dataset_from_json,
    load_queries_from_json,
)


def test_load_dataset_from_valid_dict():
    data = {
        "name": "sample_rag",
        "description": "A sample RAG benchmark dataset",
        "documents": [
            {
                "id": "doc1",
                "text": "Python is a popular programming language.",
                "metadata": {"topic": "python"},
            },
            {
                "id": "doc2",
                "text": "Rust focuses on performance and memory safety.",
                "metadata": {"topic": "rust"},
            },
        ],
        "queries": [
            {
                "id": "q1",
                "query": "Which language emphasizes memory safety?",
                "relevant_doc_ids": ["doc2"],
            },
            {"id": "q2", "query": "Tell me about Python", "relevant_doc_ids": ["doc1"]},
        ],
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
        "queries": [{"id": "q1", "query": "Hello", "relevant_doc_ids": ["d1"]}],
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
    (doc_dir / "intro.md").write_text(
        "# Intro\nThis is introductory text.", encoding="utf-8"
    )
    (doc_dir / "guide.txt").write_text("Detailed guide content here.", encoding="utf-8")

    queries_data = {
        "queries": [
            {"id": "q1", "query": "How to guide?", "relevant_doc_ids": ["guide"]}
        ]
    }
    q_file = tmp_path / "queries.json"
    q_file.write_text(json.dumps(queries_data), encoding="utf-8")

    ds = load_dataset_from_directory(
        doc_dir=doc_dir, queries_file=q_file, dataset_name="dir_test"
    )
    assert ds.name == "dir_test"
    assert len(ds.documents) == 2
    assert len(ds.queries) == 1
    doc_ids = {d.id for d in ds.documents}
    assert "intro" in doc_ids
    assert "guide" in doc_ids


def test_load_dataset_from_directory_rejects_colliding_stems(tmp_path):
    doc_dir = tmp_path / "docs"
    (doc_dir / "a").mkdir(parents=True)
    (doc_dir / "b").mkdir()
    (doc_dir / "a" / "notes.md").write_text("first", encoding="utf-8")
    (doc_dir / "b" / "notes.md").write_text("second", encoding="utf-8")
    q_file = tmp_path / "queries.json"
    q_file.write_text(json.dumps({"queries": []}), encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Duplicate document ID 'notes'") as exc:
        load_dataset_from_directory(doc_dir=doc_dir, queries_file=q_file)
    assert "a" in str(exc.value) and "b" in str(exc.value)


def test_load_dataset_from_directory_rejects_same_stem_different_suffix(tmp_path):
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "x.txt").write_text("one", encoding="utf-8")
    (doc_dir / "x.md").write_text("two", encoding="utf-8")
    q_file = tmp_path / "queries.json"
    q_file.write_text("[]", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Duplicate document ID 'x'"):
        load_dataset_from_directory(doc_dir=doc_dir, queries_file=q_file)


def test_load_dataset_from_dict_rejects_duplicate_document_ids():
    data = {
        "documents": [{"id": "d1", "text": "a"}, {"id": "d1", "text": "b"}],
        "queries": [],
    }
    with pytest.raises(DatasetLoadError, match="Duplicate document IDs: 'd1'"):
        load_dataset_from_dict(data)


def test_load_dataset_from_dict_rejects_duplicate_query_ids():
    data = {
        "documents": [{"id": "d1", "text": "a"}],
        "queries": [
            {"id": "q1", "query": "x", "relevant_doc_ids": ["d1"]},
            {"id": "q1", "query": "y", "relevant_doc_ids": ["d1"]},
        ],
    }
    with pytest.raises(DatasetLoadError, match="Duplicate query IDs: 'q1'"):
        load_dataset_from_dict(data)


def test_load_queries_from_json_rejects_duplicate_ids(tmp_path):
    q_file = tmp_path / "queries.json"
    q_file.write_text(
        json.dumps([{"id": "q1", "query": "x"}, {"id": "q1", "query": "y"}]),
        encoding="utf-8",
    )
    with pytest.raises(DatasetLoadError, match="Duplicate query IDs"):
        load_queries_from_json(q_file)


def test_load_dataset_from_dict_invalid_name_raises_loader_error():
    data = {"name": 123, "documents": [], "queries": []}
    with pytest.raises(DatasetLoadError, match="Failed to parse dataset records"):
        load_dataset_from_dict(data)


def _doc_query_data(relevant):
    return {
        "documents": [{"id": "doc1", "text": "Python."}],
        "queries": [{"id": "q1", "query": "Python?", "relevant_doc_ids": relevant}],
    }


def test_load_dataset_from_dict_rejects_unknown_relevant_doc_id():
    with pytest.raises(DatasetLoadError, match="query 'q1' -> 'doc9'"):
        load_dataset_from_dict(_doc_query_data(["doc1", "doc9"]))


def test_load_dataset_from_dict_accepts_known_relevant_doc_ids():
    dataset = load_dataset_from_dict(_doc_query_data(["doc1"]))
    assert dataset.queries[0].relevant_doc_ids == ["doc1"]


def test_load_dataset_from_directory_rejects_unknown_relevant_doc_id(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc1.txt").write_text("Python.", encoding="utf-8")
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"id": "q1", "query": "Python?", "relevant_doc_ids": ["nope"]}]),
        encoding="utf-8",
    )
    with pytest.raises(DatasetLoadError, match="'nope'"):
        load_dataset_from_directory(docs, queries)


def test_load_dataset_from_directory_rejects_empty_directory(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "config.json").write_text("{}", encoding="utf-8")
    queries = tmp_path / "queries.json"
    queries.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="No .txt or .md documents"):
        load_dataset_from_directory(docs, queries)


@pytest.mark.parametrize(
    "loader",
    [load_dataset_from_json, load_queries_from_json],
    ids=["dataset", "queries"],
)
def test_loaders_reject_a_directory_path(tmp_path, loader):
    with pytest.raises(DatasetLoadError, match="not a file"):
        loader(tmp_path)


@pytest.mark.parametrize(
    "loader",
    [load_dataset_from_json, load_queries_from_json],
    ids=["dataset", "queries"],
)
def test_loaders_report_non_utf8_files_with_the_path(tmp_path, loader):
    path = tmp_path / "data.json"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(DatasetLoadError, match="data.json"):
        loader(path)


def test_load_queries_from_json_missing_file(tmp_path):
    with pytest.raises(DatasetLoadError, match="Queries file not found"):
        load_queries_from_json(tmp_path / "nope.json")


def test_load_queries_from_json_invalid_json(tmp_path):
    path = tmp_path / "q.json"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="Invalid JSON"):
        load_queries_from_json(path)


def test_load_queries_from_json_accepts_a_queries_object(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps({"queries": [{"id": "q1", "query": "x", "relevant_doc_ids": []}]}),
        encoding="utf-8",
    )
    assert [q.id for q in load_queries_from_json(path)] == ["q1"]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"items": []}', "must contain a 'queries' list"),
        ('"just a string"', "must be a list or object"),
        ('[{"id": "q1"}]', "Failed to parse queries list"),
    ],
)
def test_load_queries_from_json_rejects_wrong_shapes(tmp_path, content, message):
    path = tmp_path / "q.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetLoadError, match=message):
        load_queries_from_json(path)


def test_load_dataset_from_dict_rejects_malformed_records():
    data = {"documents": [{"id": "d1"}], "queries": []}
    with pytest.raises(DatasetLoadError, match="Failed to parse dataset records"):
        load_dataset_from_dict(data)


def test_load_dataset_from_directory_missing_directory(tmp_path):
    with pytest.raises(DatasetLoadError, match="Documents directory not found"):
        load_dataset_from_directory(tmp_path / "nope", tmp_path / "q.json")


def test_load_dataset_from_directory_reports_unreadable_document(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "bad.txt").write_bytes(b"\xff\xfe\x00")
    queries = tmp_path / "queries.json"
    queries.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="Failed to read file .*bad.txt"):
        load_dataset_from_directory(docs, queries)
