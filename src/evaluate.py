"""
Retrieval evaluation for FestivalFinder.

Purpose
-------
Measure how well the system ranks relevant documents for predefined
test queries using manually assigned relevance judgments.

Responsibilities (planned)
--------------------------
- Load at least five evaluation queries (data/evaluation/queries.csv)
- Load relevance judgments (data/evaluation/relevance_judgments.csv)
- Compute Precision@5 (and optionally compare TF-IDF vs BM25)
- Print interpretable evaluation results

Relevance field convention (planned): 0 = not relevant, 1 = relevant
(binary judgments are sufficient for Precision@5).

This module is a stub for the initial repository setup.
"""

from __future__ import annotations

# Planned public API (signatures only; not implemented yet):
#
# def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
#     """Compute Precision@k for one query."""
#
# def run_evaluation(method: str = "tfidf") -> dict:
#     """Evaluate all test queries and return summary metrics."""


def main() -> None:
    """CLI entry point placeholder for future evaluation runs."""
    raise NotImplementedError(
        "Evaluation is not implemented yet. "
        "Queries and judgments will be added after a working retriever exists."
    )


if __name__ == "__main__":
    main()
