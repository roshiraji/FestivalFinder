"""
Document indexing and representation for FestivalFinder.

Purpose
-------
Build searchable document representations. TF-IDF turns each document into a
weighted term vector; rare informative terms get higher weight than common
terms. Cosine similarity can then compare a query vector to document vectors.
An inverted index (term → document IDs) supports efficient term lookup.

Responsibilities (planned)
--------------------------
- Create TF-IDF document-term matrices (scikit-learn)
- Persist or expose an inverted index / term-to-document mapping
- Explain how the representation supports ranked retrieval

This module is a stub for the initial repository setup.
Ranking algorithms are intentionally not implemented yet.
"""

from __future__ import annotations

# Planned public API (signatures only; not implemented yet):
#
# def build_tfidf_index(tokenized_docs: list[list[str]]):
#     """Fit a TF-IDF vectorizer and return the document-term matrix."""
#
# def build_inverted_index(tokenized_docs: list[list[str]]) -> dict[str, list[int]]:
#     """Map each term to the list of document indices containing it."""


def main() -> None:
    """CLI entry point placeholder for future index builds."""
    raise NotImplementedError(
        "Indexing is not implemented yet. "
        "It will be added after collection and preprocessing."
    )


if __name__ == "__main__":
    main()
