"""
Query processing and ranked retrieval for FestivalFinder.

Purpose
-------
Accept natural-language queries, process them like documents, and return
top-ranked results with scores and source links.

Responsibilities (planned)
--------------------------
- Process queries consistently with document preprocessing
- Rank with TF-IDF cosine similarity
- Rank with BM25 (rank-bm25) as an additional method
- Return top-k documents with scores and source URLs

This module is a stub for the initial repository setup.
Ranking algorithms are intentionally not implemented yet.
"""

from __future__ import annotations

# Planned public API (signatures only; not implemented yet):
#
# def search_tfidf(query: str, top_k: int = 5) -> list[dict]:
#     """Rank documents by TF-IDF cosine similarity."""
#
# def search_bm25(query: str, top_k: int = 5) -> list[dict]:
#     """Rank documents with BM25."""


def main() -> None:
    """CLI entry point placeholder for future search demos."""
    raise NotImplementedError(
        "Search/ranking is not implemented yet. "
        "Core retrieval will be built after indexing is in place."
    )


if __name__ == "__main__":
    main()
