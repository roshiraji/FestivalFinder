"""
Document collection for FestivalFinder.

Purpose
-------
Collect authentic text from permitted official festival FAQ and guide pages.
Extract titles, relevant content, source URLs, and festival names, then save
documents to CSV using the schema in data/processed/documents.csv.

Responsibilities (planned)
--------------------------
- Fetch pages with requests; parse HTML with BeautifulSoup
- Handle HTTP errors and missing content gracefully
- Respect robots.txt and site access restrictions
- Write collected rows to CSV (no fabricated festival information)

This module is a stub for the initial repository setup.
Implementation begins in the next development stage (collection pipeline).
"""

from __future__ import annotations

# Planned public API (signatures only; not implemented yet):
#
# def collect_documents(seed_urls: list[str], output_path: str) -> int:
#     """Collect documents from seed URLs and write them to CSV. Return count."""
#
# def fetch_page(url: str) -> str | None:
#     """Fetch a single page body, or return None on failure."""
#
# def parse_document(html: str, url: str, festival_name: str) -> dict | None:
#     """Extract title and content fields from HTML into a document row."""


def main() -> None:
    """CLI entry point placeholder for future collection runs."""
    raise NotImplementedError(
        "Document collection is not implemented yet. "
        "See README.md — Milestone 1: Official festival guide collection."
    )


if __name__ == "__main__":
    main()
