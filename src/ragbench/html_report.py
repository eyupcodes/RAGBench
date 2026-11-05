"""Render a benchmark report as one self-contained HTML page.

The page has inline CSS only (no JavaScript, no network access), so it can be
attached to a pull request or ticket as is. Every value that comes from the
dataset or configuration is HTML-escaped. The best value in each metric column
is highlighted: the highest for quality metrics, the lowest for latency.
"""

from html import escape

from ragbench.models import BenchmarkReport, StrategyResult

_STYLE = """
body{font-family:system-ui,sans-serif;margin:2rem;color:#1f2328}
table{border-collapse:collapse}
th,td{border:1px solid #d0d7de;padding:.35rem .7rem;text-align:right}
th{background:#f6f8fa}
td.name{text-align:left}
td.best{background:#dafbe1;font-weight:600}
caption{text-align:left;margin-bottom:.5rem}
"""


def _values(result: StrategyResult, ks: list[int], ndcg_k: int | None) -> list[float]:
    metrics = result.metrics
    values = [metrics.recall_at_k.get(k, 0.0) for k in ks]
    values.append(metrics.mrr)
    if ndcg_k is not None:
        values.append(metrics.ndcg_at_k.get(ndcg_k, 0.0))
    values.append(metrics.avg_latency_ms)
    return values


def render_html(report: BenchmarkReport) -> str:
    """Return the report as a complete HTML document."""
    title = f"RAGBench report: {report.dataset_name}"
    head = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n<style>{_STYLE}</style>\n</head>\n<body>\n"
        f"<h1>{escape(title)}</h1>\n"
        f"<p>Documents: {report.num_documents} | Queries: {report.num_queries}</p>\n"
    )
    if not report.results:
        return head + "<p>No benchmark results available.</p>\n</body>\n</html>\n"

    ks = sorted(report.results[0].metrics.recall_at_k)
    ndcg_k = max(report.results[0].metrics.ndcg_at_k, default=None)
    labels = [f"R@{k}" for k in ks] + ["MRR"]
    if ndcg_k is not None:
        labels.append(f"nDCG@{ndcg_k}")
    labels.append("Latency (ms)")
    higher_is_better = [True] * (len(labels) - 1) + [False]

    rows = [_values(r, ks, ndcg_k) for r in report.results]
    best = [
        (max if higher else min)(row[i] for row in rows)
        for i, higher in enumerate(higher_is_better)
    ]

    header = "".join(
        f"<th>{escape(h)}</th>"
        for h in ["Chunking", "Retrieval", "Reranker", "Embedding", "Chunks", *labels]
    )
    body = []
    for result, row in zip(report.results, rows, strict=True):
        text_cells = [
            result.chunking_strategy,
            result.retrieval_strategy,
            result.reranking_strategy or "none",
            result.embedding_model or "-",
        ]
        cells = [f'<td class="name">{escape(c)}</td>' for c in text_cells]
        cells.append(f"<td>{result.num_chunks}</td>")
        for i, value in enumerate(row):
            fmt = f"{value:.2f}" if not higher_is_better[i] else f"{value:.3f}"
            css = ' class="best"' if value == best[i] else ""
            cells.append(f"<td{css}>{fmt}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    table = (
        "<table>\n<caption>Best value per metric is highlighted.</caption>\n"
        f"<thead><tr>{header}</tr></thead>\n<tbody>\n"
        + "\n".join(body)
        + "\n</tbody>\n</table>\n"
    )
    return head + table + "</body>\n</html>\n"
