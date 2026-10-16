"""Retrieval evaluation for FestivalFinder.

Metrics
-------
Precision@k is the fraction of the top k retrieved documents that were
judged relevant:

    P@k = (number of relevant documents in the top k) / k

The denominator stays k even when the ranker returns fewer than k documents.
Those missing positions count as not relevant because the system did not
retrieve a document there. That is different from a retrieved document that
has not been judged.

If any document in the top k has a blank judgment, Precision@k is not
computed. A blank is not stored as 0.

Mean Average Precision uses average precision. For one query, with R relevant
documents in the completed judgment pool:

    AP = (1 / R) * sum over relevant ranks r of (Precision at r)

If R is 0 after every pool row for that query has been judged, AP is 0.

Documents that were never placed in the judgment pool have no label. MAP does
not treat those documents as irrelevant. They are removed from the ranked list
before AP is calculated. This is condensed-list MAP: a relevant document's
rank is its position among judged documents only. That can be higher than its
rank in the full list, so this MAP can be higher than a MAP that counted
unjudged documents as not relevant. The choice is stated here so it is not
silent.

Precision@5 and Precision@10 do not need that choice when the pool is the
union of each method's top 10 and every pool row is filled. Every document
in those top 10 lists then has a label.

Scores are reported only after every relevance cell in the judgment pool is
0 or 1. Until then, the evaluation command counts blank rows and stops.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.bm25 import build_bm25_index
from src.index import build_tfidf_index
from src.search import search_bm25, search_tfidf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "queries.csv"
POOL_PATH = PROJECT_ROOT / "data" / "evaluation" / "judgment_pool.csv"

POOL_COLUMNS = [
    "query_id",
    "query_text",
    "document_id",
    "festival_name",
    "title",
    "category",
    "source_url",
    "relevance",
]
METHODS = ("baseline", "improved", "bm25")
POOL_DEPTH = 10


def load_queries(path: str | Path = QUERIES_PATH) -> pd.DataFrame:
    """Read the evaluation queries. The file is not modified."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["query_id", "query_text"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"queries.csv is missing columns: {', '.join(missing)}")
    if frame["query_id"].duplicated().any():
        raise ValueError("queries.csv has duplicate query_id values")
    if (frame["query_text"].str.strip() == "").any():
        raise ValueError("queries.csv has an empty query")
    return frame


def parse_relevance(value: object) -> int | None:
    """Return 0 or 1, or None when the judgment cell is still blank."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return None
    if text in {"0", "1"}:
        return int(text)
    raise ValueError(f"Relevance must be blank, 0, or 1. Found: {value!r}")


def judgments_are_complete(frame: pd.DataFrame) -> bool:
    """True only when every pool row has a 0 or 1 label."""
    if frame.empty:
        return False
    return all(parse_relevance(value) is not None for value in frame["relevance"])


def blank_judgment_count(frame: pd.DataFrame) -> int:
    return sum(parse_relevance(value) is None for value in frame["relevance"])


def assemble_judgment_pool(
    queries: pd.DataFrame,
    hits_by_method: dict[str, dict[str, list[dict[str, object]]]],
    documents: pd.DataFrame,
) -> pd.DataFrame:
    """Deduplicate the top documents from every method into one blank pool.

    Relevance is always an empty string. This function does not decide
    whether a document answers the question.
    """
    metadata = documents.drop_duplicates("document_id").set_index("document_id")
    rows: list[dict[str, str]] = []
    for query in queries.itertuples(index=False):
        seen: set[str] = set()
        query_rows: list[dict[str, str]] = []
        query_id = str(query.query_id)
        for method in METHODS:
            for hit in hits_by_method.get(method, {}).get(query_id, []):
                document_id = str(hit["document_id"])
                if document_id in seen:
                    continue
                seen.add(document_id)
                if document_id not in metadata.index:
                    raise KeyError(f"Ranked document is not in the working collection: {document_id}")
                document = metadata.loc[document_id]
                query_rows.append(
                    {
                        "query_id": query_id,
                        "query_text": str(query.query_text),
                        "document_id": document_id,
                        "festival_name": str(document["festival_name"]),
                        "title": str(document["title"]),
                        "category": str(document["category"]),
                        "source_url": str(document["source_url"]),
                        "relevance": "",
                    }
                )
        query_rows.sort(key=lambda row: row["document_id"])
        rows.extend(query_rows)
    return pd.DataFrame(rows, columns=POOL_COLUMNS)


def collect_method_hits(queries: pd.DataFrame, top_k: int = POOL_DEPTH) -> tuple[dict, pd.DataFrame]:
    """Run baseline TF-IDF, improved TF-IDF, and BM25. Does not write the CSV."""
    baseline = build_tfidf_index(mode="baseline")
    improved = build_tfidf_index(mode="improved")
    bm25 = build_bm25_index()
    indexes = {"baseline": baseline, "improved": improved, "bm25": bm25}
    hits: dict[str, dict[str, list[dict[str, object]]]] = {}
    for method, index in indexes.items():
        festival_filter = method in {"improved", "bm25"}
        search = search_bm25 if method == "bm25" else search_tfidf
        hits[method] = {}
        for query in queries.itertuples(index=False):
            result = search(query.query_text, index, top_k=top_k, festival_filter=festival_filter)
            hits[method][str(query.query_id)] = list(result["results"])
    return hits, improved.documents


def write_judgment_pool(frame: pd.DataFrame, path: str | Path = POOL_PATH) -> None:
    """Write the pool. Refuses to replace a file that already has labels."""
    destination = Path(path)
    if destination.exists():
        existing = pd.read_csv(destination, dtype=str, keep_default_na=False)
        if "relevance" in existing.columns and blank_judgment_count(existing) < len(existing):
            raise FileExistsError(
                f"{destination} already contains relevance labels. "
                "It was not overwritten."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8")


def precision_at_k(
    retrieved_ids: list[str],
    labels: dict[str, int],
    k: int,
) -> float | None:
    """Precision@k, or None when any retrieved document in the top k is unlabeled.

    Missing ranks below the length of ``retrieved_ids`` count in the
    denominator as not relevant. An unlabeled retrieved document does not.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    top = list(retrieved_ids)[:k]
    if any(document_id not in labels for document_id in top):
        return None
    relevant = sum(1 for document_id in top if labels[document_id] == 1)
    return relevant / k


def average_precision(retrieved_ids: list[str], labels: dict[str, int]) -> float | None:
    """Condensed-list average precision for one completed query.

    ``labels`` maps judged document IDs to 0 or 1. Retrieved IDs that are
    absent from ``labels`` are dropped from the ranking. They are not scored
    as irrelevant. Returns None when ``labels`` is empty. Returns 0 when the
    judged set contains no relevant document.
    """
    if not labels:
        return None
    judged_ranking = [document_id for document_id in retrieved_ids if document_id in labels]
    relevant_total = sum(1 for label in labels.values() if label == 1)
    if relevant_total == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, document_id in enumerate(judged_ranking, start=1):
        if labels[document_id] == 1:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / relevant_total


def mean_average_precision(average_precisions: list[float]) -> float | None:
    """Mean of per-query average precision. None when there are no queries."""
    if not average_precisions:
        return None
    return sum(average_precisions) / len(average_precisions)


def labels_for_query(pool: pd.DataFrame, query_id: str) -> dict[str, int] | None:
    """Return judged labels, or None if this query still has a blank cell."""
    rows = pool.loc[pool["query_id"].astype(str) == str(query_id)]
    labels: dict[str, int] = {}
    for row in rows.itertuples(index=False):
        label = parse_relevance(row.relevance)
        if label is None:
            return None
        labels[str(row.document_id)] = label
    return labels


def evaluate_rankings(
    pool: pd.DataFrame,
    rankings: dict[str, dict[str, list[str]]],
) -> dict[str, object]:
    """Score each method. Returns no metric values while any judgment is blank."""
    blank = blank_judgment_count(pool)
    if blank:
        return {
            "complete": False,
            "rows": len(pool),
            "blank_rows": blank,
            "methods": {},
        }
    query_ids = list(dict.fromkeys(pool["query_id"].astype(str)))
    methods: dict[str, dict[str, object]] = {}
    for method, by_query in rankings.items():
        per_query = []
        for query_id in query_ids:
            labels = labels_for_query(pool, query_id)
            if labels is None:
                return {
                    "complete": False,
                    "rows": len(pool),
                    "blank_rows": blank_judgment_count(pool),
                    "methods": {},
                }
            retrieved = by_query.get(query_id, [])
            per_query.append(
                {
                    "query_id": query_id,
                    "precision_at_5": precision_at_k(retrieved, labels, 5),
                    "precision_at_10": precision_at_k(retrieved, labels, 10),
                    "average_precision": average_precision(retrieved, labels),
                }
            )
        methods[method] = {
            "per_query": per_query,
            "precision_at_5": mean_average_precision(
                [row["precision_at_5"] for row in per_query if row["precision_at_5"] is not None]
            ),
            "precision_at_10": mean_average_precision(
                [row["precision_at_10"] for row in per_query if row["precision_at_10"] is not None]
            ),
            "map": mean_average_precision(
                [row["average_precision"] for row in per_query if row["average_precision"] is not None]
            ),
        }
    return {"complete": True, "rows": len(pool), "blank_rows": 0, "methods": methods}


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def format_evaluation(report: dict[str, object]) -> str:
    """Format a completed report. Incomplete judgments produce a stop message."""
    if not report["complete"]:
        return (
            f"Judgment pool rows: {report['rows']}\n"
            f"Blank relevance labels: {report['blank_rows']}\n"
            "Scores are not reported. Fill every relevance cell with 1 or 0, "
            "then run python -m src.evaluate"
        )
    lines = [
        "Evaluation uses completed manual judgments.",
        "Unjudged documents were excluded from MAP, not labeled 0.",
    ]
    methods = report["methods"]
    for method, summary in methods.items():
        lines.append("")
        lines.append(method)
        lines.append(f"  Mean Precision@5: {_format_metric(summary['precision_at_5'])}")
        lines.append(f"  Mean Precision@10: {_format_metric(summary['precision_at_10'])}")
        lines.append(f"  MAP: {_format_metric(summary['map'])}")
        for row in summary["per_query"]:
            lines.append(
                f"  {row['query_id']}: P@5={_format_metric(row['precision_at_5'])} "
                f"P@10={_format_metric(row['precision_at_10'])} "
                f"AP={_format_metric(row['average_precision'])}"
            )
    return "\n".join(lines)


def load_pool(path: str | Path = POOL_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in POOL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Judgment pool is missing columns: {', '.join(missing)}")
    return frame.loc[:, POOL_COLUMNS].copy()


def rankings_from_indexes(queries: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Full positive-score rankings for MAP, plus the order used at search time."""
    from src.search import _query_plan, _score_array

    built = {
        "baseline": build_tfidf_index(mode="baseline"),
        "improved": build_tfidf_index(mode="improved"),
        "bm25": build_bm25_index(),
    }
    rankings: dict[str, dict[str, list[str]]] = {}
    for method, index in built.items():
        rankings[method] = {}
        festival_filter = method in {"improved", "bm25"}
        for query in queries.itertuples(index=False):
            plan = _query_plan(query.query_text, index, festival_filter)
            scores = _score_array(plan, index)
            order = scores.argsort()[::-1]
            ranked = []
            for row in order:
                if float(scores[int(row)]) <= 0:
                    break
                ranked.append(index.document_ids[int(row)])
            rankings[method][str(query.query_id)] = ranked
    return rankings


def main(argv: list[str] | None = None) -> None:
    """Write the blank judgment pool, or score a completed pool."""
    parser = argparse.ArgumentParser(description="Prepare or score FestivalFinder relevance judgments.")
    parser.add_argument(
        "--make-pool",
        action="store_true",
        help="Write data/evaluation/judgment_pool.csv with blank relevance cells.",
    )
    args = parser.parse_args(argv)
    if args.make_pool:
        queries = load_queries()
        hits, documents = collect_method_hits(queries)
        pool = assemble_judgment_pool(queries, hits, documents)
        write_judgment_pool(pool)
        print(f"Wrote {len(pool)} unlabeled rows to {POOL_PATH}")
        print("Fill the relevance column with 1 (relevant) or 0 (not relevant).")
        print("Scores are not computed from this file until those labels exist.")
        return
    if not POOL_PATH.exists():
        print("No judgment pool yet. Create it with: python -m src.evaluate --make-pool")
        return
    pool = load_pool()
    if not judgments_are_complete(pool):
        print(format_evaluation(evaluate_rankings(pool, {})))
        return
    queries = load_queries()
    report = evaluate_rankings(pool, rankings_from_indexes(queries))
    print(format_evaluation(report))


if __name__ == "__main__":
    main()
