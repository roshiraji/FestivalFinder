"""Tests for evaluation queries, the judgment pool, and the metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.evaluate import (
    POOL_COLUMNS,
    assemble_judgment_pool,
    average_precision,
    blank_judgment_count,
    evaluate_rankings,
    format_evaluation,
    load_queries,
    mean_average_precision,
    precision_at_k,
    write_judgment_pool,
)
from src.search import SAMPLE_QUERIES

ORIGINAL_QUERIES = set(SAMPLE_QUERIES)
TOPICS = {"transportation", "lodging", "rules", "packing", "experiences"}


def _documents() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "document_id": "doc-a",
                "festival_name": "Coachella",
                "title": "Shuttles",
                "category": "transportation",
                "source_url": "https://www.coachella.com/getting-here/",
            },
            {
                "document_id": "doc-b",
                "festival_name": "Bonnaroo",
                "title": "What is Outeroo?",
                "category": "lodging",
                "source_url": "https://support.bonnaroo.com/outeroo",
            },
            {
                "document_id": "doc-c",
                "festival_name": "Lollapalooza",
                "title": "Bag policy",
                "category": "rules",
                "source_url": "https://www.lollapalooza.com/safety",
            },
        ]
    )


def _queries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"query_id": "q01", "query_text": "Where are the shuttles?"},
            {"query_id": "q02", "query_text": "How does camping work?"},
        ]
    )


def test_evaluation_queries_include_the_original_five_and_cover_each_topic():
    queries = load_queries()
    assert len(queries) >= 10
    assert ORIGINAL_QUERIES <= set(queries["query_text"])
    assert TOPICS <= set(queries["description"])


def test_judgment_pool_is_deduplicated_and_unlabeled():
    hits = {
        "baseline": {
            "q01": [{"document_id": "doc-a"}, {"document_id": "doc-b"}],
            "q02": [{"document_id": "doc-b"}],
        },
        "improved": {
            "q01": [{"document_id": "doc-b"}, {"document_id": "doc-a"}],
            "q02": [{"document_id": "doc-c"}],
        },
        "bm25": {
            "q01": [{"document_id": "doc-a"}],
            "q02": [{"document_id": "doc-b"}],
        },
    }
    pool = assemble_judgment_pool(_queries(), hits, _documents())
    assert list(pool.columns) == POOL_COLUMNS
    assert set(pool["relevance"]) == {""}
    q01 = pool.loc[pool["query_id"] == "q01", "document_id"].tolist()
    assert q01 == ["doc-a", "doc-b"]
    assert pool.loc[pool["query_id"] == "q02", "document_id"].tolist() == ["doc-b", "doc-c"]
    assert pool.loc[pool["document_id"] == "doc-a", "title"].iloc[0] == "Shuttles"


def test_labeled_pool_is_not_overwritten(tmp_path: Path):
    pool = assemble_judgment_pool(
        _queries().iloc[:1],
        {"baseline": {"q01": [{"document_id": "doc-a"}]}, "improved": {}, "bm25": {}},
        _documents(),
    )
    destination = tmp_path / "judgment_pool.csv"
    write_judgment_pool(pool, destination)
    labeled = pool.copy()
    labeled.loc[:, "relevance"] = "1"
    labeled.to_csv(destination, index=False)
    with pytest.raises(FileExistsError):
        write_judgment_pool(pool, destination)


def test_precision_and_map_on_a_completed_synthetic_pool():
    assert precision_at_k(["a", "b", "c", "d", "e"], {"a": 1, "b": 0, "c": 1, "d": 0, "e": 0}, 5) == pytest.approx(0.4)
    assert precision_at_k(["a"], {"a": 1}, 5) == pytest.approx(0.2)
    assert precision_at_k(["u", "a"], {"a": 1}, 2) is None
    # The unlabeled document is removed, so the relevant document is rank 1.
    # Treating it as irrelevant would put the relevant document at rank 2.
    assert average_precision(["u", "a"], {"a": 1, "b": 0}) == pytest.approx(1.0)
    assert average_precision(["b", "a"], {"a": 1, "b": 0}) == pytest.approx(0.5)
    assert average_precision(["b"], {"b": 0}) == 0.0
    assert mean_average_precision([1.0, 0.5]) == pytest.approx(0.75)
    assert mean_average_precision([]) is None


def test_incomplete_judgments_do_not_produce_scores():
    pool = pd.DataFrame(
        [
            {
                "query_id": "q01",
                "query_text": "Where are the shuttles?",
                "document_id": "doc-a",
                "festival_name": "Coachella",
                "title": "Shuttles",
                "category": "transportation",
                "source_url": "https://www.coachella.com/getting-here/",
                "relevance": "",
            }
        ]
    )
    report = evaluate_rankings(pool, {"baseline": {"q01": ["doc-a"]}})
    assert report["complete"] is False
    assert report["methods"] == {}
    assert blank_judgment_count(pool) == 1
    text = format_evaluation(report)
    assert "not reported" in text
    assert "Precision" not in text


def test_completed_judgments_report_per_query_and_average_scores():
    pool = pd.DataFrame(
        [
            {
                "query_id": "q01",
                "query_text": "Where are the shuttles?",
                "document_id": "doc-a",
                "festival_name": "Coachella",
                "title": "Shuttles",
                "category": "transportation",
                "source_url": "https://www.coachella.com/getting-here/",
                "relevance": "1",
            },
            {
                "query_id": "q01",
                "query_text": "Where are the shuttles?",
                "document_id": "doc-b",
                "festival_name": "Bonnaroo",
                "title": "What is Outeroo?",
                "category": "lodging",
                "source_url": "https://support.bonnaroo.com/outeroo",
                "relevance": "0",
            },
        ]
    )
    report = evaluate_rankings(pool, {"baseline": {"q01": ["doc-a", "doc-b"]}})
    assert report["complete"] is True
    per_query = report["methods"]["baseline"]["per_query"][0]
    assert per_query["precision_at_5"] == pytest.approx(0.2)
    assert per_query["precision_at_10"] == pytest.approx(0.1)
    assert per_query["average_precision"] == pytest.approx(1.0)
    text = format_evaluation(report)
    assert "Mean Precision@5" in text
    assert "MAP" in text
