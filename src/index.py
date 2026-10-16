"""Build a TF-IDF index for the FestivalFinder working collection.

What TF-IDF stores
------------------
Each document becomes a vector with one number per vocabulary term.

Term frequency, tf(t, d), is how often term t occurs in document d.
A term that is repeated in a guide gets a larger tf.

Inverse document frequency down-weights terms that appear in many documents.
Scikit-learn's default smoothed IDF, which this index uses, is

    idf(t) = ln((1 + n) / (1 + df(t))) + 1

where n is the number of indexed documents and df(t) is how many of them
contain t. A rare word such as "rideshare" receives a larger idf than a
word that appears in almost every festival guide.

The TF-IDF weight is tf(t, d) * idf(t). Each document vector is then divided
by its Euclidean length (L2 norm). After that normalization, cosine
similarity between a query and a document is their dot product.

Two indexes share this formula
------------------------------
``mode="baseline"`` indexes document bodies only, with the original
tokenizer. That reproduces the Milestone 2 rankings.

``mode="improved"`` indexes the title and the body. A title term's count is
multiplied by ``title_weight`` before IDF is applied:

    tf(t, d) = tf_body(t, d) + title_weight * tf_title(t, d)

Festival-name tokens in the title are counted once, not multiplied. Repeating
"EDC Las Vegas" in a title therefore does not take over the vector. Document
frequency still counts a term once per document, so repeating a title word
raises tf and does not raise idf. L2 normalization then rescales the whole
vector: title terms take a larger share of the unit vector, and cosine
similarity with a query that uses those words goes up. The body is still in
the vector, so a long guide is not replaced by its title.

How this relates to an inverted index
--------------------------------------
The fitted result is a sparse document-term matrix with one row per document
and one column per term. Most entries are zero because a document uses only
a small part of the vocabulary.

A classic inverted index lists, for each term, the documents that contain it
(the postings list). In this matrix, that list is the nonzero entries in the
term's column. ``build_inverted_index`` reads those columns and returns
term -> document IDs. The matrix stores TF-IDF weights in those postings
instead of raw counts, which is what cosine ranking needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import (
    FESTIVAL_NAME_TOKENS,
    prepare_collection,
    tokenize,
    tokenize_normalized,
)

# Each non-festival title term is counted this many times in tf.
# 2 is a modest boost: the title is visible without replacing the body.
TITLE_WEIGHT = 2


@dataclass
class TfidfIndex:
    """Fitted TF-IDF artifacts needed to rank a new query.

    ``matrix`` row i is the vector for ``document_ids[i]``. ``documents``
    holds the original text and metadata for those same rows, in the same
    order, so search results can show titles and URLs.
    """

    vectorizer: TfidfVectorizer
    matrix: sparse.csr_matrix
    document_ids: list[str]
    documents: pd.DataFrame
    inverted_index: dict[str, list[str]]
    mode: str
    title_weight: int


def build_inverted_index(
    document_ids: list[str],
    matrix: sparse.spmatrix,
    vocabulary: dict[str, int],
) -> dict[str, list[str]]:
    """Map each term to the document IDs whose TF-IDF weight is nonzero.

    This is the postings list for that term. Document IDs follow the row
    order of the matrix.
    """
    columns = matrix.tocsc()
    index_to_term = {column: term for term, column in vocabulary.items()}
    postings: dict[str, list[str]] = {}
    for column in range(columns.shape[1]):
        document_rows = columns.getcol(column).nonzero()[0]
        postings[index_to_term[column]] = [document_ids[int(row)] for row in document_rows]
    return postings


def improved_document_tokens(title: str, content: str, title_weight: int = TITLE_WEIGHT) -> list[str]:
    """Build the token list whose counts become tf for one improved document.

    Non-festival title tokens are repeated ``title_weight`` times. Each
    festival-name token is kept once even if the title repeats it. Body
    tokens follow, each counted once per occurrence.
    """
    if title_weight < 1:
        raise ValueError("title_weight must be at least 1")
    weighted: list[str] = []
    seen_festival_tokens: set[str] = set()
    for token in tokenize_normalized(title):
        if token in FESTIVAL_NAME_TOKENS:
            if token in seen_festival_tokens:
                continue
            seen_festival_tokens.add(token)
            weighted.append(token)
            continue
        weighted.extend([token] * title_weight)
    weighted.extend(tokenize_normalized(content))
    return weighted


def build_tfidf_index(
    documents: pd.DataFrame | None = None,
    mode: str = "baseline",
    title_weight: int = TITLE_WEIGHT,
) -> TfidfIndex:
    """Fit TF-IDF on the working collection.

    Pass a dataframe to index a test collection. With no argument, the
    filtered collection from ``documents.csv`` is used. ``mode="baseline"``
    is body-only. ``mode="improved"`` adds stemmed, title-weighted text.
    Neither mode writes ``documents.csv`` or the indexing manifest.
    """
    if mode not in {"baseline", "improved"}:
        raise ValueError("mode must be 'baseline' or 'improved'")
    if documents is None:
        documents = prepare_collection(manifest_path=None)["working"]
    if not isinstance(documents, pd.DataFrame):
        raise TypeError("documents must be a DataFrame")
    required = {"document_id", "content"}
    if mode == "improved":
        required = required | {"title"}
    missing = required - set(documents.columns)
    if missing:
        raise ValueError(f"Index input is missing columns: {', '.join(sorted(missing))}")

    frame = documents.reset_index(drop=True)
    if mode == "baseline":
        vectorizer = TfidfVectorizer(analyzer=tokenize, norm="l2")
        matrix = vectorizer.fit_transform(frame["content"].astype(str))
    else:
        token_lists = [
            improved_document_tokens(title, content, title_weight)
            for title, content in zip(frame["title"].astype(str), frame["content"].astype(str), strict=True)
        ]
        vectorizer = TfidfVectorizer(analyzer=str.split, norm="l2")
        matrix = vectorizer.fit_transform([" ".join(tokens) for tokens in token_lists])
    matrix = sparse.csr_matrix(matrix)
    document_ids = frame["document_id"].astype(str).tolist()
    inverted = build_inverted_index(document_ids, matrix, vectorizer.vocabulary_)
    return TfidfIndex(
        vectorizer=vectorizer,
        matrix=matrix,
        document_ids=document_ids,
        documents=frame,
        inverted_index=inverted,
        mode=mode,
        title_weight=title_weight if mode == "improved" else 0,
    )


def index_statistics(index: TfidfIndex) -> dict[str, int | float]:
    """Summarize the document-term matrix."""
    rows, columns = index.matrix.shape
    nonzero = int(index.matrix.nnz)
    cells = rows * columns
    return {
        "documents": rows,
        "vocabulary_size": columns,
        "matrix_rows": rows,
        "matrix_columns": columns,
        "nonzero_weights": nonzero,
        "sparsity": round(1.0 - (nonzero / cells), 6) if cells else 0.0,
    }


def _print_stats(label: str, index: TfidfIndex) -> None:
    stats = index_statistics(index)
    print(label)
    print(f"  Indexed documents: {stats['documents']}")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")
    print(f"  Matrix dimensions: {stats['matrix_rows']} x {stats['matrix_columns']}")
    print(f"  Nonzero TF-IDF weights: {stats['nonzero_weights']}")
    print(f"  Sparsity: {stats['sparsity']}")


def main() -> None:
    """Print the size of the baseline index and the improved index."""
    _print_stats("Baseline TF-IDF index (body only)", build_tfidf_index(mode="baseline"))
    improved = build_tfidf_index(mode="improved")
    _print_stats(f"Improved TF-IDF index (title weight {improved.title_weight})", improved)


if __name__ == "__main__":
    main()
