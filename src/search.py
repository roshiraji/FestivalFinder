"""Rank FestivalFinder documents with TF-IDF cosine similarity.

A query is cleaned and tokenized with the same function as the documents,
then transformed with the already fitted TF-IDF vectorizer. That applies the
collection's IDF weights to the query. Cosine similarity compares the query
vector with every document vector:

    cosine(q, d) = (q · d) / (||q|| ||d||)

Because the index stores L2-normalized vectors, and the query vector is
normalized the same way, this cosine is the dot product of the two vectors.
The largest scores are the closest documents in that vector space.

BM25 lives in ``src.bm25`` and is selected with ``--mode bm25``. It uses the
same festival filter and the same stemmed tokens as the improved TF-IDF index.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.bm25 import Bm25Index, build_bm25_index
from src.index import TITLE_WEIGHT, TfidfIndex, build_tfidf_index
from src.preprocess import clean_text, tokenize, tokenize_normalized

RetrievalIndex = TfidfIndex | Bm25Index

# Explicit festival mentions. "EDC" alone maps to EDC Las Vegas. A bare
# "Las Vegas", "lands", or "outside" does not, because those words are ambiguous.
_FESTIVAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Coachella", re.compile(r"\bcoachella\b")),
    ("Bonnaroo", re.compile(r"\bbonnaroo\b")),
    ("Lollapalooza", re.compile(r"\blollapalooza\b")),
    ("Outside Lands", re.compile(r"\boutside lands\b")),
    ("EDC Las Vegas", re.compile(r"\bedc las vegas\b|\bedc\b")),
)

SAMPLE_QUERIES = [
    "What are the best transportation options for Coachella?",
    "What can I bring into Lollapalooza?",
    "How does camping work at Bonnaroo?",
    "What should I pack for EDC Las Vegas?",
    "What is the bag policy at Outside Lands?",
]


def _snippet(content: str, tokens: list[str], length: int = 240) -> str:
    """Return a short window of the original text, preferring a query term."""
    text = re.sub(r"\s+", " ", content or "").strip()
    lowered = text.lower()
    for token in tokens:
        position = lowered.find(token)
        if position >= 0:
            start = max(0, position - 80)
            window = text[start : start + length].strip()
            return window
    return text[:length].strip()


def detect_festivals(query: str) -> list[str]:
    """Return festival names explicitly written in the query, in a stable order.

    Zero names means search every festival. More than one name is ambiguous,
    so the caller must not guess between them.
    """
    text = clean_text(query).lower()
    return [name for name, pattern in _FESTIVAL_PATTERNS if pattern.search(text)]


def _festival_tokens(festival: str, query: str) -> set[str]:
    """Tokens removed from the topical query once that festival is selected.

    "EDC" removes only ``edc``. "EDC Las Vegas" also removes ``las`` and
    ``vegas``, which are part of that name. Other festivals remove only
    their own name tokens.
    """
    names = {
        "Coachella": {"coachella"},
        "Bonnaroo": {"bonnaroo"},
        "Lollapalooza": {"lollapalooza"},
        "Outside Lands": {"outside", "lands"},
        "EDC Las Vegas": {"edc"},
    }
    tokens = set(names[festival])
    if festival == "EDC Las Vegas" and re.search(r"\blas vegas\b", clean_text(query).lower()):
        tokens.update({"las", "vegas"})
    return tokens


def _uses_improved_text(index: RetrievalIndex) -> bool:
    """Improved TF-IDF and BM25 share stemming, titles, and festival filtering."""
    return index.mode in {"improved", "bm25"}


def _query_plan(query: str, index: RetrievalIndex, festival_filter: bool) -> dict[str, object]:
    """Decide which tokens are scored and whether a festival filter applies."""
    raw_query = query or ""
    if _uses_improved_text(index):
        tokens = tokenize_normalized(raw_query)
    else:
        tokens = tokenize(raw_query)
    mentioned = detect_festivals(raw_query) if _uses_improved_text(index) else []
    selected = ""
    topical = list(tokens)
    notes: list[str] = []
    if _uses_improved_text(index) and festival_filter:
        if len(mentioned) == 1:
            selected = mentioned[0]
            drop = _festival_tokens(selected, raw_query)
            topical = [token for token in tokens if token not in drop]
            notes.append(
                f"Results are limited to {selected}. "
                "Festival-name terms were removed from the topical query."
            )
        elif len(mentioned) > 1:
            names = " and ".join(mentioned)
            notes.append(
                f"The query names {names}, so it is not limited to one festival."
            )
    return {
        "raw_query": raw_query,
        "tokens": tokens,
        "topical": topical,
        "mentioned": mentioned,
        "selected": selected,
        "notes": notes,
    }


def _scores(plan: dict[str, object], index: TfidfIndex) -> np.ndarray:
    """Cosine similarity of one planned query against every indexed document."""
    raw_query = str(plan["raw_query"])
    if index.mode == "baseline":
        query_vector = index.vectorizer.transform([raw_query])
    else:
        query_vector = index.vectorizer.transform([" ".join(plan["topical"])])
    scores = cosine_similarity(query_vector, index.matrix).ravel()
    selected = str(plan["selected"])
    if selected:
        keep = index.documents["festival_name"].astype(str).to_numpy() == selected
        scores = scores.copy()
        scores[~keep] = 0.0
    return scores


def _score_array(plan: dict[str, object], index: RetrievalIndex) -> np.ndarray:
    """Score every document. BM25 uses rank-bm25; TF-IDF uses cosine similarity."""
    if index.mode != "bm25":
        return _scores(plan, index)
    scores = np.asarray(index.model.get_scores(list(plan["topical"])), dtype=float)
    selected = str(plan["selected"])
    if selected:
        keep = index.documents["festival_name"].astype(str).to_numpy() == selected
        scores = scores.copy()
        scores[~keep] = 0.0
    return scores


def _empty_result(plan: dict[str, object], note: str, missing: list[str]) -> dict[str, object]:
    notes = list(plan["notes"])
    if note:
        notes.append(note)
    return {
        "query": plan["raw_query"],
        "results": [],
        "missing_terms": missing,
        "note": " ".join(notes),
        "mode": "",
        "festival": plan["selected"],
        "festivals_mentioned": plan["mentioned"],
    }


def search_tfidf(
    query: str,
    index: RetrievalIndex | None = None,
    top_k: int = 5,
    festival_filter: bool = False,
) -> dict[str, object]:
    """Return the top TF-IDF cosine hits for one natural-language query.

    An empty query, a festival-only query, or a query whose topical tokens
    are all outside the vocabulary returns no ranked documents and a note.
    Scores come from cosine similarity. They are not hand-assigned.

    ``festival_filter`` applies only to an improved index. The baseline index
    always searches every document, which keeps its original ranking.
    """
    if index is None:
        index = build_tfidf_index()
    plan = _query_plan(query, index, festival_filter)
    topical = list(plan["topical"])
    vocabulary = index.vocabulary if index.mode == "bm25" else index.vectorizer.vocabulary_
    known = [token for token in topical if token in vocabulary]
    missing = [token for token in topical if token not in vocabulary]
    plan["notes"] = list(plan["notes"])

    if not str(plan["raw_query"]).strip() or not plan["tokens"]:
        result = _empty_result(
            plan,
            "The query is empty after preprocessing, so there is nothing to rank.",
            missing,
        )
        result["mode"] = index.mode
        return result
    if plan["tokens"] and not topical:
        result = _empty_result(
            plan,
            f"The query only names {plan['selected']}, so there are no topic terms left to rank.",
            missing,
        )
        result["mode"] = index.mode
        return result
    if not known:
        result = _empty_result(
            plan,
            "None of the query terms are in the indexed vocabulary, so every similarity would be zero.",
            missing,
        )
        result["mode"] = index.mode
        return result

    if missing:
        plan["notes"].append("Ignored query terms that are not in the vocabulary: " + ", ".join(missing))
    scores = _score_array(plan, index)
    # Snippets search the original wording. Festival-name tokens removed from
    # scoring are also removed from the snippet lookup.
    drop = set(plan["tokens"]) - set(topical)
    surface = [token for token in tokenize(str(plan["raw_query"])) if token not in drop]
    order = np.argsort(scores)[::-1]
    results = []
    for row in order:
        score = float(scores[int(row)])
        if score <= 0 or len(results) >= top_k:
            break
        document = index.documents.iloc[int(row)]
        results.append(
            {
                "document_id": str(document["document_id"]),
                "title": str(document["title"]),
                "festival_name": str(document["festival_name"]),
                "festival_year": str(document["festival_year"]),
                "category": str(document["category"]),
                "score": score,
                "snippet": _snippet(str(document["content"]), surface or known),
                "source_url": str(document["source_url"]),
            }
        )
    return {
        "query": plan["raw_query"],
        "results": results,
        "missing_terms": missing,
        "note": " ".join(plan["notes"]),
        "mode": index.mode,
        "festival": plan["selected"],
        "festivals_mentioned": plan["mentioned"],
    }


def search_bm25(
    query: str,
    index: Bm25Index | None = None,
    top_k: int = 5,
    festival_filter: bool = False,
) -> dict[str, object]:
    """Return the top BM25 hits for one natural-language query.

    Scores come from BM25Okapi. They are not cosine similarities and they are
    not limited to 1. Festival filtering matches the improved TF-IDF ranker.
    """
    if index is None:
        index = build_bm25_index()
    if index.mode != "bm25":
        raise TypeError("search_bm25 requires a BM25 index")
    return search_tfidf(query, index, top_k=top_k, festival_filter=festival_filter)


def document_rank(
    query: str,
    index: RetrievalIndex,
    title: str,
    festival_name: str,
    festival_filter: bool = False,
) -> dict[str, object]:
    """Return the 1-based rank and score of one stored document.

    Rank follows the same ordering as search. A document with no positive
    score is unranked. TF-IDF scores are cosine similarities. BM25 scores are
    BM25 weights.
    """
    plan = _query_plan(query, index, festival_filter)
    topical = list(plan["topical"])
    vocabulary = index.vocabulary if index.mode == "bm25" else index.vectorizer.vocabulary_
    if not topical or not any(token in vocabulary for token in topical):
        return {"rank": None, "score": 0.0}
    scores = _score_array(plan, index)
    matches = (index.documents["title"].astype(str) == title) & (
        index.documents["festival_name"].astype(str) == festival_name
    )
    rows = np.flatnonzero(matches.to_numpy())
    if len(rows) == 0:
        return {"rank": None, "score": None}
    target = int(rows[0])
    target_score = float(scores[target])
    if target_score <= 0:
        return {"rank": None, "score": target_score}
    order = np.argsort(scores)[::-1]
    for rank, row in enumerate(order, start=1):
        if int(row) == target:
            return {"rank": rank, "score": target_score}
        if float(scores[int(row)]) <= 0:
            break
    return {"rank": None, "score": target_score}


def format_search(result: dict[str, object]) -> str:
    """Format one search result for the command line."""
    lines = [f"Query: {result['query']}"]
    if result.get("note"):
        lines.append(f"Note: {result['note']}")
    hits = result["results"]
    if not hits:
        lines.append("No ranked documents.")
        return "\n".join(lines)
    for rank, hit in enumerate(hits, start=1):
        lines.append(
            f"{rank}. [{hit['score']:.4f}] {hit['festival_name']} {hit['festival_year']} "
            f"({hit['category']}) {hit['title']}"
        )
        lines.append(f"   {hit['source_url']}")
        lines.append(f"   {hit['snippet']}")
    return "\n".join(lines)


def _search(query: str, index: RetrievalIndex, festival_filter: bool) -> dict[str, object]:
    if index.mode == "bm25":
        return search_bm25(query, index, festival_filter=festival_filter)
    return search_tfidf(query, index, festival_filter=festival_filter)


def run_demo(index: RetrievalIndex, festival_filter: bool = False) -> None:
    """Print the five predefined sample queries and their top results."""
    for query in SAMPLE_QUERIES:
        print(format_search(_search(query, index, festival_filter)))
        print()


def interactive_search(index: RetrievalIndex, festival_filter: bool = False) -> None:
    """Prompt for queries until the user types exit."""
    print("FestivalFinder Search")
    if index.mode == "baseline":
        print("Ranking: original body-only TF-IDF.")
    elif index.mode == "bm25" and festival_filter:
        print("Ranking: BM25. A named festival limits the results.")
    elif index.mode == "bm25":
        print("Ranking: BM25. All festivals are searched.")
    elif festival_filter:
        print("Ranking: improved TF-IDF. A named festival limits the results.")
    else:
        print("Ranking: improved TF-IDF. All festivals are searched.")
    while True:
        try:
            query = input("Enter your query (or type 'exit' to quit): ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.strip().lower() == "exit":
            break
        if not query.strip():
            continue
        print(format_search(_search(query, index, festival_filter)))
        print()


def main(argv: list[str] | None = None) -> None:
    """Start interactive search, or the sample-query demo.

    The default is the improved ranker. ``--mode baseline`` reproduces the
    original body-only TF-IDF rankings.
    """
    parser = argparse.ArgumentParser(
        description="Search FestivalFinder with TF-IDF cosine similarity."
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "improved", "bm25"],
        default="improved",
        help="baseline is body-only TF-IDF. improved adds the title, stemming, and festival filtering. bm25 uses BM25Okapi with the same text as improved. Default: improved.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the five predefined sample queries and exit.",
    )
    parser.add_argument(
        "--no-festival-filter",
        action="store_true",
        help="Search every festival even when the query names one. Applies to improved TF-IDF and BM25.",
    )
    parser.add_argument(
        "--title-weight",
        type=int,
        default=TITLE_WEIGHT,
        help=f"How many times each non-festival title term is counted. Default: {TITLE_WEIGHT}.",
    )
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 term-frequency saturation. Default: 1.5.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 length penalty. Default: 0.75.")
    args = parser.parse_args(argv)
    if args.title_weight < 1:
        parser.error("--title-weight must be at least 1")
    if args.k1 <= 0:
        parser.error("--k1 must be positive")
    if not 0 <= args.b <= 1:
        parser.error("--b must be between 0 and 1")
    if args.mode == "bm25":
        index = build_bm25_index(title_weight=args.title_weight, k1=args.k1, b=args.b)
    else:
        index = build_tfidf_index(mode=args.mode, title_weight=args.title_weight)
    festival_filter = args.mode in {"improved", "bm25"} and not args.no_festival_filter
    if args.demo:
        run_demo(index, festival_filter=festival_filter)
    else:
        interactive_search(index, festival_filter=festival_filter)


if __name__ == "__main__":
    main()
