"""
Text preprocessing for FestivalFinder.

Purpose
-------
Prepare raw document and query text for indexing and retrieval so that
queries and documents are processed consistently.

Responsibilities (planned)
--------------------------
- Clean HTML artifacts and excess whitespace
- Normalize case and tokenize
- Optionally remove stopwords (NLTK)
- Preserve important festival names and location tokens when useful

This module is a stub for the initial repository setup.
"""

from __future__ import annotations

# Planned public API (signatures only; not implemented yet):
#
# def clean_text(text: str) -> str:
#     """Remove HTML artifacts and normalize whitespace."""
#
# def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
#     """Normalize and tokenize text for indexing/query processing."""
#
# def preprocess_documents(df) -> list[list[str]]:
#     """Apply cleaning and tokenization to a document DataFrame."""


def main() -> None:
    """CLI entry point placeholder for future preprocessing runs."""
    raise NotImplementedError(
        "Preprocessing is not implemented yet. "
        "It will follow the document collection pipeline."
    )


if __name__ == "__main__":
    main()
