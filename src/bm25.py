"""BM25 ranking for the FestivalFinder working collection.

The index is ``BM25Okapi`` from the rank-bm25 package. Documents and queries
use the same stemmed tokens as the improved TF-IDF index. Title terms use the
same configurable count: each non-festival title token is repeated
``title_weight`` times, and each festival-name token in the title is kept
once. That raises the term frequency of title words inside the BM25 formula.
It also makes the document a little longer, which the length penalty sees.

For a query term t in document d, BM25Okapi adds

    IDF(t) * (tf(t, d) * (k1 + 1))
             / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

|d| is the token count, including repeated title tokens. avgdl is the average
document length. k1 (default 1.5) saturates term frequency, so a camping pass
that says "camping" many times does not grow its score without limit. b
(default 0.75) penalizes long documents.

rank-bm25's IDF, from the ATIRE variant, is

    IDF(t) = ln(N - df(t) + 0.5) - ln(df(t) + 0.5)

If that value is negative because the term appears in more than half the
documents, it is replaced by epsilon times the average IDF. epsilon is 0.25.
A query term that is not in the collection adds nothing.

These scores are not cosine similarities, and they are not limited to 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rank_bm25 import BM25Okapi

from src.index import TITLE_WEIGHT, improved_document_tokens
from src.preprocess import prepare_collection

BM25_K1 = 1.5
BM25_B = 0.75


@dataclass
class Bm25Index:
    """BM25 artifacts for one working collection.

    Row i of ``model`` is ``document_ids[i]``. ``documents`` keeps the
    original text and metadata in that same order.
    """

    model: BM25Okapi
    document_ids: list[str]
    documents: pd.DataFrame
    vocabulary: set[str]
    mode: str
    title_weight: int
    k1: float
    b: float


def build_bm25_index(
    documents: pd.DataFrame | None = None,
    title_weight: int = TITLE_WEIGHT,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> Bm25Index:
    """Fit BM25 on the same working collection used by TF-IDF.

    The original CSV and the indexing manifest are not written.
    """
    if title_weight < 1:
        raise ValueError("title_weight must be at least 1")
    if k1 <= 0:
        raise ValueError("k1 must be positive")
    if not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1")
    if documents is None:
        documents = prepare_collection(manifest_path=None)["working"]
    if not isinstance(documents, pd.DataFrame):
        raise TypeError("documents must be a DataFrame")
    required = {"document_id", "title", "content"}
    missing = required - set(documents.columns)
    if missing:
        raise ValueError(f"Index input is missing columns: {', '.join(sorted(missing))}")

    frame = documents.reset_index(drop=True)
    token_lists = [
        improved_document_tokens(title, content, title_weight)
        for title, content in zip(frame["title"].astype(str), frame["content"].astype(str), strict=True)
    ]
    model = BM25Okapi(token_lists, k1=k1, b=b)
    vocabulary = {token for tokens in token_lists for token in tokens}
    return Bm25Index(
        model=model,
        document_ids=frame["document_id"].astype(str).tolist(),
        documents=frame,
        vocabulary=vocabulary,
        mode="bm25",
        title_weight=title_weight,
        k1=k1,
        b=b,
    )
