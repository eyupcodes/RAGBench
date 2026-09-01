"""Example demonstration script: Running RAGBench programmatically with custom answer evaluation."""

from pathlib import Path
from ragbench.benchmark import BenchmarkRunner
from ragbench.dataset import load_dataset_from_json
from ragbench.metrics import TokenOverlapAnswerEvaluator
from ragbench.models import BenchmarkConfig, RetrievalResult
from ragbench.cli import format_ascii_table


def mock_llm_synthesizer(query: str, retrieved_chunks: list[RetrievalResult]) -> str:
    """A simulated LLM synthesis step taking top retrieved passages and generating an answer."""
    if not retrieved_chunks:
        return "No relevant context found."

    top_chunk_text = retrieved_chunks[0].chunk.text
    return top_chunk_text[:120]


def main():
    dataset_path = Path(__file__).parent / "data" / "sample_tech_docs.json"
    print(f"Loading dataset from: {dataset_path.name}")
    dataset = load_dataset_from_json(dataset_path)

    config = BenchmarkConfig(
        chunking_strategies=["fixed", "recursive"],
        retrieval_strategies=["bm25", "tfidf"],
        k_values=[1, 3, 5],
        chunk_size=200,
        chunk_overlap=30,
    )

    print("\nRunning benchmark across chunking & retrieval combinations...")
    runner = BenchmarkRunner(config=config)
    evaluator = TokenOverlapAnswerEvaluator()

    report = runner.run(
        dataset=dataset,
        answer_evaluator=evaluator,
        answer_generator=mock_llm_synthesizer,
    )

    print("\n" + format_ascii_table(report))


if __name__ == "__main__":
    main()
