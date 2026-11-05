"""Tests for the self-contained HTML report."""

from ragbench.html_report import render_html
from ragbench.models import BenchmarkReport, EvaluationMetrics, StrategyResult


def _result(chunking, recall, mrr, latency, **extra):
    return StrategyResult(
        chunking_strategy=chunking,
        retrieval_strategy=extra.pop("retrieval", "bm25"),
        num_chunks=7,
        metrics=EvaluationMetrics(
            recall_at_k={1: recall, 3: 1.0},
            ndcg_at_k={3: 0.5},
            mrr=mrr,
            avg_latency_ms=latency,
        ),
        **extra,
    )


def _report(*results, name="demo"):
    return BenchmarkReport(
        dataset_name=name, num_documents=3, num_queries=2, results=list(results)
    )


def test_page_is_standalone_and_lists_every_result():
    page = render_html(
        _report(_result("fixed", 0.5, 0.6, 9.0), _result("recursive", 1.0, 0.9, 12.0))
    )

    assert page.startswith("<!DOCTYPE html>")
    assert "<script" not in page and "http" not in page
    assert "<th>R@1</th>" in page and "<th>nDCG@3</th>" in page
    assert "fixed" in page and "recursive" in page
    assert "Documents: 3 | Queries: 2" in page


def test_best_value_per_column_is_highlighted_and_latency_is_lowest():
    page = render_html(
        _report(_result("fixed", 0.5, 0.6, 9.0), _result("recursive", 1.0, 0.9, 12.0))
    )

    assert page.count('<td class="best">1.000</td>') >= 1  # R@1 of recursive
    assert '<td class="best">9.00</td>' in page
    assert '<td class="best">12.00</td>' not in page
    assert "<td>0.600</td>" in page  # fixed MRR is not the best
    assert '<td class="best">0.900</td>' in page


def test_user_controlled_text_is_escaped():
    page = render_html(
        _report(
            _result("<img src=x onerror=alert(1)>", 1.0, 1.0, 1.0, retrieval="a&b"),
            name="<script>alert(1)</script>",
        )
    )

    assert "<script>" not in page and "<img" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "a&amp;b" in page


def test_reranker_and_embedding_columns_show_values_or_placeholders():
    page = render_html(
        _report(
            _result("fixed", 1.0, 1.0, 1.0, reranking_strategy="lexical"),
            _result("fixed", 1.0, 1.0, 1.0, embedding_model="mini-lm"),
        )
    )

    assert '<td class="name">lexical</td>' in page
    assert '<td class="name">none</td>' in page
    assert '<td class="name">mini-lm</td>' in page
    assert '<td class="name">-</td>' in page


def test_empty_report_renders_a_notice():
    page = render_html(_report())

    assert "No benchmark results available." in page
    assert "<table" not in page


def test_report_without_ndcg_omits_the_column():
    result = StrategyResult(
        chunking_strategy="fixed",
        retrieval_strategy="bm25",
        num_chunks=1,
        metrics=EvaluationMetrics(recall_at_k={1: 1.0}, mrr=1.0),
    )

    page = render_html(_report(result))

    assert "nDCG" not in page
