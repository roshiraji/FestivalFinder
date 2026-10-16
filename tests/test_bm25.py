"""Tests for BM25 ranking."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.bm25 import build_bm25_index
from src.search import search_bm25


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "document_id": "title-hydration",
                "festival_name": "Coachella",
                "festival_year": "2026",
                "category": "rules",
                "title": "Hydration",
                "content": "Yes.",
                "source_type": "official_faq",
                "source_url": "https://www.coachella.com/faq/",
                "published_date": "",
            },
            {
                "document_id": "coachella-shuttle",
                "festival_name": "Coachella",
                "festival_year": "2026",
                "category": "transportation",
                "title": "Shuttles",
                "content": "A shuttle leaves from the hotel.",
                "source_type": "official_guide",
                "source_url": "https://www.coachella.com/getting-here/",
                "published_date": "",
            },
            {
                "document_id": "bonnaroo-shuttle",
                "festival_name": "Bonnaroo",
                "festival_year": "2026",
                "category": "transportation",
                "title": "Shuttles",
                "content": "A shuttle leaves from the farm.",
                "source_type": "official_guide",
                "source_url": "https://www.bonnaroo.com/accommodations",
                "published_date": "",
            },
            {
                "document_id": "edc-pack",
                "festival_name": "EDC Las Vegas",
                "festival_year": "2027",
                "category": "weather",
                "title": "Essential Packing List",
                "content": "Pack lightly for camp.",
                "source_type": "official_guide",
                "source_url": "https://lasvegas.electricdaisycarnival.com/guide/faq/",
                "published_date": "",
            },
            {
                "document_id": "noise-hours",
                "festival_name": "Outside Lands",
                "festival_year": "2026",
                "category": "experiences",
                "title": "Hours",
                "content": "Gates open in the afternoon.",
                "source_type": "official_faq",
                "source_url": "https://sfoutsidelands.com/faq-app",
                "published_date": "",
            },
            {
                "document_id": "noise-map",
                "festival_name": "Outside Lands",
                "festival_year": "2026",
                "category": "experiences",
                "title": "Map",
                "content": "A printed map is sold near the entrance.",
                "source_type": "official_faq",
                "source_url": "https://sfoutsidelands.com/faq-app",
                "published_date": "",
            },
        ]
    )


def test_bm25_matches_stemmed_title_and_body_terms():
    index = build_bm25_index(_rows(), title_weight=3)
    hydration = search_bm25("hydration", index, festival_filter=False)
    assert hydration["results"][0]["document_id"] == "title-hydration"
    shuttles = search_bm25("shuttles", index, festival_filter=False)
    assert {hit["document_id"] for hit in shuttles["results"]} == {
        "coachella-shuttle",
        "bonnaroo-shuttle",
    }


def test_bm25_festival_filter_and_empty_queries():
    index = build_bm25_index(_rows())
    filtered = search_bm25("Coachella shuttle", index, festival_filter=True)
    assert [hit["document_id"] for hit in filtered["results"]] == ["coachella-shuttle"]
    assert filtered["festival"] == "Coachella"
    open_search = search_bm25("Coachella shuttle", index, festival_filter=False)
    assert {hit["document_id"] for hit in open_search["results"]} >= {
        "coachella-shuttle",
        "bonnaroo-shuttle",
    }
    empty = search_bm25("   ", index)
    assert empty["results"] == []
    assert "empty" in empty["note"].lower()
    unknown = search_bm25("xyzzyplugh", index)
    assert unknown["results"] == []
    assert "xyzzyplugh" in unknown["missing_terms"]
    only_name = search_bm25("EDC", index, festival_filter=True)
    assert only_name["results"] == []
    assert "only names EDC Las Vegas" in only_name["note"]


def test_bm25_scores_are_reproducible_and_the_csv_is_unchanged():
    source = Path("data/processed/documents.csv")
    manifest = Path("data/processed/indexing_manifest.csv")
    source_before = source.read_bytes()
    manifest_before = manifest.read_bytes()
    frame = _rows()
    first = build_bm25_index(frame)
    second = build_bm25_index(frame)
    one = search_bm25("shuttle", first, festival_filter=False)
    two = search_bm25("shuttle", second, festival_filter=False)
    assert [(hit["document_id"], hit["score"]) for hit in one["results"]] == [
        (hit["document_id"], hit["score"]) for hit in two["results"]
    ]
    real = build_bm25_index()
    assert len(real.document_ids) == 246
    assert source.read_bytes() == source_before
    assert manifest.read_bytes() == manifest_before
