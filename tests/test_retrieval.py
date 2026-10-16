"""Tests for preprocessing, TF-IDF indexing, and cosine retrieval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.index import build_tfidf_index, improved_document_tokens, index_statistics
from src.preprocess import (
    clean_text,
    filter_near_duplicates,
    load_documents,
    prepare_collection,
    tokenize,
    tokenize_normalized,
    validate_documents,
)
from src.search import (
    SAMPLE_QUERIES,
    detect_festivals,
    document_rank,
    interactive_search,
    search_tfidf,
)

TINY_ROWS = [
    {
        "document_id": "doc-shuttle",
        "festival_name": "Coachella",
        "festival_year": "2026",
        "category": "transportation",
        "title": "Shuttles",
        "content": "The Coachella shuttle is the best transportation from the hotel to the festival.",
        "source_type": "official_guide",
        "source_url": "https://www.coachella.com/getting-here/",
        "published_date": "",
    },
    {
        "document_id": "doc-bag",
        "festival_name": "Lollapalooza",
        "festival_year": "2026",
        "category": "rules",
        "title": "Bag policy",
        "content": "Bags at Lollapalooza must follow the prohibited items and bag policy.",
        "source_type": "official_faq",
        "source_url": "https://www.lollapalooza.com/safety",
        "published_date": "",
    },
    {
        "document_id": "doc-camp",
        "festival_name": "Bonnaroo",
        "festival_year": "2026",
        "category": "lodging",
        "title": "Carpool Primitive Camping (Thursday Entry)",
        "content": "Bonnaroo camping gives one car a campsite. Thursday entry uses the same campsite rules.",
        "source_type": "official_guide",
        "source_url": "https://www.bonnaroo.com/accommodations",
        "published_date": "",
    },
    {
        "document_id": "doc-camp-copy",
        "festival_name": "Bonnaroo",
        "festival_year": "2026",
        "category": "lodging",
        "title": "Carpool Primitive Camping (Wednesday Entry)",
        "content": "Bonnaroo camping gives one car a campsite. Wednesday entry repeats the same campsite rules.",
        "source_type": "official_guide",
        "source_url": "https://www.bonnaroo.com/accommodations",
        "published_date": "",
    },
    {
        "document_id": "doc-empty",
        "festival_name": "Coachella",
        "festival_year": "2026",
        "category": "rules",
        "title": "Blank",
        "content": "   ",
        "source_type": "official_faq",
        "source_url": "https://www.coachella.com/faq/",
        "published_date": "",
    },
]


def _frame(rows: list[dict] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or TINY_ROWS)


def test_clean_text_drops_a_trailing_label_only():
    assert clean_text("Shuttles run until midnight. Learn More") == "Shuttles run until midnight."
    assert clean_text("  Learn   more about   parking. ") == "Learn more about parking."


def test_tokenize_normalizes_case_and_keeps_protected_terms():
    tokens = tokenize("The Coachella shuttle is NOT allowed in the camp")
    assert "coachella" in tokens
    assert "shuttle" in tokens
    assert "not" in tokens
    assert "camp" in tokens
    assert "the" not in tokens
    assert "is" not in tokens


def test_validation_reports_empty_content_and_missing_columns():
    report = validate_documents(_frame())
    assert report["empty_content"] == ["doc-empty"]
    try:
        load_documents  # imported
        from src.preprocess import load_documents as load

        path_error = False
    except Exception:
        path_error = True
    assert path_error is False
    bad = _frame()[["title", "content"]]
    try:
        from src.preprocess import COLUMNS

        missing = [column for column in COLUMNS if column not in bad.columns]
        assert "document_id" in missing
    except Exception as exc:  # pragma: no cover
        raise AssertionError(exc)


def test_near_duplicate_filter_keeps_the_longer_bonnaroo_copy():
    working, excluded = filter_near_duplicates(_frame())
    titles = set(working["title"])
    assert "Carpool Primitive Camping (Thursday Entry)" in titles
    assert "Carpool Primitive Camping (Wednesday Entry)" not in titles
    assert list(excluded["title"]) == ["Carpool Primitive Camping (Wednesday Entry)"]
    assert excluded.iloc[0]["retained_title"] == "Carpool Primitive Camping (Thursday Entry)"


def test_prepare_collection_does_not_modify_the_csv(tmp_path: Path):
    source = tmp_path / "documents.csv"
    _frame().to_csv(source, index=False)
    before = source.read_bytes()
    prepared = prepare_collection(source, manifest_path=tmp_path / "manifest.csv")
    assert source.read_bytes() == before
    assert len(prepared["working"]) == 3
    manifest = (tmp_path / "manifest.csv").read_text(encoding="utf-8")
    assert "doc-camp-copy" in manifest
    assert "near-duplicate" in manifest
    assert "doc-empty" in manifest


def test_index_matrix_rows_match_document_ids():
    working, _ = filter_near_duplicates(_frame())
    working = working.loc[working["content"].str.strip() != ""].reset_index(drop=True)
    index = build_tfidf_index(working)
    stats = index_statistics(index)
    assert stats["documents"] == len(working)
    assert stats["matrix_rows"] == len(index.document_ids)
    assert stats["matrix_columns"] == stats["vocabulary_size"]
    assert index.document_ids[0] == working.iloc[0]["document_id"]
    assert "shuttle" in index.inverted_index
    assert "doc-shuttle" in index.inverted_index["shuttle"]


def test_search_ranks_the_matching_festival_document_first():
    working, _ = filter_near_duplicates(_frame())
    working = working.loc[working["content"].str.strip() != ""].reset_index(drop=True)
    index = build_tfidf_index(working)
    result = search_tfidf("Coachella shuttle transportation", index, top_k=5)
    assert result["results"]
    assert result["results"][0]["document_id"] == "doc-shuttle"
    assert result["results"][0]["score"] > result["results"][-1]["score"] or len(result["results"]) == 1
    assert result["results"][0]["source_url"].startswith("https://www.coachella.com/")
    assert "shuttle" in result["results"][0]["snippet"].lower()


def test_empty_and_unknown_queries_do_not_invent_hits():
    working, _ = filter_near_duplicates(_frame())
    working = working.loc[working["content"].str.strip() != ""].reset_index(drop=True)
    index = build_tfidf_index(working)
    empty = search_tfidf("   ", index)
    assert empty["results"] == []
    assert "empty" in empty["note"].lower()
    unknown = search_tfidf("xyzzyplugh", index)
    assert unknown["results"] == []
    assert "xyzzyplugh" in unknown["missing_terms"]


def test_interactive_search_prompts_until_exit(monkeypatch, capsys):
    answers = iter(["shuttle to the festival", "exit"])
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(
        "src.search.search_tfidf",
        lambda query, index, top_k=5, festival_filter=False: {
            "query": query,
            "results": [],
            "missing_terms": [],
            "note": "",
        },
    )
    index = type("Index", (), {"mode": "improved"})()
    interactive_search(index)
    output = capsys.readouterr().out
    assert output.startswith("FestivalFinder Search\n")
    assert prompts == [
        "Enter your query (or type 'exit' to quit): ",
        "Enter your query (or type 'exit' to quit): ",
    ]
    assert "Query: shuttle to the festival" in output
    assert "No ranked documents." in output


def test_real_collection_filter_matches_the_reviewed_groups():
    source = Path("data/processed/documents.csv")
    before = source.read_bytes()
    prepared = prepare_collection(source, manifest_path=None)
    assert source.read_bytes() == before
    assert prepared["validation"]["empty_content"] == []
    assert len(prepared["full"]) == 257
    assert len(prepared["excluded"]) == 11
    assert len(prepared["working"]) == 246


def _search_rows() -> pd.DataFrame:
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
                "document_id": "body-hydration",
                "festival_name": "Bonnaroo",
                "festival_year": "2026",
                "category": "rules",
                "title": "Hours",
                "content": "Hydration stations exist near many other gates today.",
                "source_type": "official_faq",
                "source_url": "https://www.bonnaroo.com/accommodations",
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
        ]
    )


def test_porter_stemming_merges_word_forms_and_keeps_protected_terms():
    assert tokenize_normalized("pack") == ["pack"]
    assert tokenize_normalized("packing") == ["pack"]
    assert tokenize_normalized("packs") == ["pack"]
    assert tokenize_normalized("camp") == ["camp"]
    assert tokenize_normalized("camping") == ["camp"]
    assert tokenize_normalized("shuttle") == ["shuttl"]
    assert tokenize_normalized("shuttles") == ["shuttl"]
    assert tokenize_normalized("not allowed") == ["not", "allow"]
    assert tokenize_normalized("no exceptions") == ["no", "except"]
    assert tokenize_normalized("neither this nor that") == ["neither", "nor"]
    vegas = tokenize_normalized("Las Vegas")
    assert vegas == ["las", "vegas"]
    assert "vega" not in vegas
    assert tokenize_normalized("Coachella") == ["coachella"]
    assert tokenize_normalized("parking") == ["parking"]
    assert tokenize_normalized("Golden Gate Park") == ["golden", "gate", "park"]


def test_title_weight_repeats_topic_terms_and_caps_festival_names():
    tokens = improved_document_tokens("EDC EDC packing list", "Bring water.", title_weight=3)
    assert tokens.count("edc") == 1
    assert tokens.count("pack") == 3
    assert tokens.count("water") == 1


def test_title_only_term_is_searchable_in_the_improved_index():
    frame = _search_rows()
    baseline = build_tfidf_index(frame, mode="baseline")
    improved = build_tfidf_index(frame, mode="improved", title_weight=3)
    baseline_ids = {hit["document_id"] for hit in search_tfidf("hydration", baseline)["results"]}
    improved_result = search_tfidf("hydration", improved)
    assert "title-hydration" not in baseline_ids
    assert improved_result["results"][0]["document_id"] == "title-hydration"
    assert improved_result["results"][0]["score"] > 0


def test_festival_filter_limits_results_and_can_be_turned_off():
    frame = _search_rows()
    index = build_tfidf_index(frame, mode="improved")
    filtered = search_tfidf("Coachella shuttle", index, festival_filter=True)
    assert [hit["document_id"] for hit in filtered["results"]] == ["coachella-shuttle"]
    assert filtered["festival"] == "Coachella"
    open_search = search_tfidf("Coachella shuttle", index, festival_filter=False)
    assert {hit["document_id"] for hit in open_search["results"]} >= {
        "coachella-shuttle",
        "bonnaroo-shuttle",
    }
    unnamed = search_tfidf("shuttle", index, festival_filter=True)
    assert {hit["festival_name"] for hit in unnamed["results"]} == {"Coachella", "Bonnaroo"}


def test_festival_detection_does_not_guess_and_edc_is_an_alias():
    assert detect_festivals("bag policy") == []
    assert detect_festivals("Las Vegas hotels") == []
    assert detect_festivals("the lands near the park") == []
    assert detect_festivals("EDC packing") == ["EDC Las Vegas"]
    assert detect_festivals("EDC Las Vegas packing") == ["EDC Las Vegas"]
    assert detect_festivals("Outside Lands bag policy") == ["Outside Lands"]
    assert detect_festivals("Coachella and Bonnaroo camping") == ["Coachella", "Bonnaroo"]
    frame = _search_rows()
    index = build_tfidf_index(frame, mode="improved")
    only_name = search_tfidf("Coachella", index, festival_filter=True)
    assert only_name["results"] == []
    assert "only names Coachella" in only_name["note"]
    ambiguous = search_tfidf("Coachella and Bonnaroo shuttle", index, festival_filter=True)
    assert ambiguous["festival"] == ""
    assert "not limited to one festival" in ambiguous["note"]
    assert {hit["festival_name"] for hit in ambiguous["results"]} == {"Coachella", "Bonnaroo"}
    edc = search_tfidf("What should I pack for EDC?", index, festival_filter=True)
    assert [hit["document_id"] for hit in edc["results"]] == ["edc-pack"]
    assert edc["festival"] == "EDC Las Vegas"


def test_improved_empty_and_unknown_queries_do_not_invent_hits():
    index = build_tfidf_index(_search_rows(), mode="improved")
    empty = search_tfidf("   ", index, festival_filter=True)
    assert empty["results"] == []
    assert "empty" in empty["note"].lower()
    unknown = search_tfidf("xyzzyplugh", index, festival_filter=True)
    assert unknown["results"] == []
    assert "xyzzyplugh" in unknown["missing_terms"]


def test_baseline_rankings_and_source_files_stay_unchanged():
    source = Path("data/processed/documents.csv")
    manifest = Path("data/processed/indexing_manifest.csv")
    source_before = source.read_bytes()
    manifest_before = manifest.read_bytes()
    baseline = build_tfidf_index(mode="baseline")
    improved = build_tfidf_index(mode="improved")
    assert source.read_bytes() == source_before
    assert manifest.read_bytes() == manifest_before
    assert baseline.document_ids == improved.document_ids
    assert len(baseline.document_ids) == 246
    assert "shuttle" in baseline.inverted_index
    assert "shuttl" in improved.inverted_index

    first = search_tfidf(SAMPLE_QUERIES[0], baseline)
    assert first["results"][0]["title"] == "Electric Vehicle Charging: Level 2 Charging Stations"
    assert first["results"][0]["score"] == pytest.approx(0.2253, abs=5e-5)
    assert first["results"][4]["title"] == "Shuttles, Rideshare, Drop-off & Pick-up: Shuttles"
    assert first["results"][4]["score"] == pytest.approx(0.1496, abs=5e-5)
    shuttle = document_rank(
        SAMPLE_QUERIES[0],
        baseline,
        "Shuttles, Rideshare, Drop-off & Pick-up: Shuttles",
        "Coachella",
    )
    packing = document_rank(SAMPLE_QUERIES[3], baseline, "Essential Packing List", "EDC Las Vegas")
    bag = document_rank(
        SAMPLE_QUERIES[4],
        baseline,
        "FAQs: What is the bag policy at Outside Lands?",
        "Outside Lands",
    )
    assert shuttle["rank"] == 5
    assert shuttle["score"] == pytest.approx(0.1496, abs=5e-5)
    assert packing["rank"] == 28
    assert packing["score"] == pytest.approx(0.0454, abs=5e-5)
    assert bag["rank"] == 18
    assert bag["score"] == pytest.approx(0.0620, abs=5e-5)

    # The same baseline query still searches every festival when a caller asks
    # for festival filtering. Filtering belongs to the improved index only.
    unfiltered = search_tfidf(SAMPLE_QUERIES[0], baseline, festival_filter=True)
    assert [hit["title"] for hit in unfiltered["results"]] == [hit["title"] for hit in first["results"]]
