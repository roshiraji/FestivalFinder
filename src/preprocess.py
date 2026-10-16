"""Prepare FestivalFinder documents for TF-IDF indexing.

The original CSV is only read. Display text stays in that file. This module
builds a smaller working collection by dropping empty documents and the
Bonnaroo near-duplicates identified during collection. Indexing and search
use that working collection.

Preprocessing steps, in order
-----------------------------
1. Validate the CSV schema and report empty content.
2. Clean a copy of each document for indexing: collapse whitespace and drop
   a trailing "Learn More" label. The stored CSV text is not rewritten.
3. Lowercase and tokenize on letters and numbers.
4. Remove English stopwords, except negation ("not", "no", "nor") and a
   small set of festival, place, and transportation words that must stay
   searchable.

The baseline tokenizer stops there. The improved tokenizer then applies the
Porter stemmer so that "packing" and "pack" become the same term. Festival
names, place names, negation, and "park"/"parking" are not stemmed.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from src.collect import COLUMNS, strip_ui_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "documents.csv"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "indexing_manifest.csv"

_TOKENIZER = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_TAG = re.compile(r"<[^>]+>")

# These tokens stay even when NLTK lists them as stopwords. "not" matters
# because "not allowed" would otherwise look like "allowed".
PRESERVE_TOKENS = frozenset(
    {
        "not",
        "no",
        "nor",
        "coachella",
        "bonnaroo",
        "lollapalooza",
        "edc",
        "outside",
        "lands",
        "vegas",
        "chicago",
        "indio",
        "manchester",
        "shuttle",
        "shuttles",
        "parking",
        "rideshare",
        "camp",
        "camping",
        "rv",
    }
)

# Porter would change these in ways that hurt this collection: "vegas"
# becomes "vega", "lands" becomes "land", and "parking" becomes "park".
# "park" in these guides usually means a venue, such as Golden Gate Park,
# which is a different word from the transportation term "parking".
STEM_EXEMPT = frozenset(
    {
        "not",
        "no",
        "nor",
        "coachella",
        "bonnaroo",
        "lollapalooza",
        "edc",
        "outside",
        "lands",
        "vegas",
        "las",
        "chicago",
        "indio",
        "manchester",
        "park",
        "parking",
    }
)

# Festival-name tokens are recognized as metadata. Title weighting counts
# each of them once so a title that repeats "EDC Las Vegas" does not drown
# out the topical title words.
FESTIVAL_NAME_TOKENS = frozenset(
    {
        "coachella",
        "bonnaroo",
        "lollapalooza",
        "edc",
        "outside",
        "lands",
        "las",
        "vegas",
    }
)

# Near-duplicate Bonnaroo passes from the Milestone 1 review. The kept title
# is the longer copy in each group. Excluded titles are left in documents.csv
# and omitted only from the indexing collection.
NEAR_DUPLICATE_GROUPS: list[dict[str, object]] = [
    {
        "keep": "Carpool Primitive Camping (Thursday Entry)",
        "exclude": ["Carpool Primitive Camping (Wednesday Entry)"],
    },
    {
        "keep": "Wood Frame Safari Tent - Double Twin",
        "exclude": [
            "Luxury Luna Bell Tent - Queen",
            "Luxury Luna Bell Tent - Double Twin",
            "Wood Frame Safari Tent - Queen",
        ],
    },
    {
        "keep": "Non-Shaded Power RV",
        "exclude": ["Shaded Power RV"],
    },
    {
        "keep": "Primitive Car Camping (Thursday Entry)",
        "exclude": ["Primitive Car Camping (Wednesday Entry)"],
    },
    {
        "keep": "Closer Primitive Car Camping",
        "exclude": ["Primitive Car Camping"],
    },
    {
        "keep": "Large Darkroom Tent - Queen",
        "exclude": ["Large Darkroom Tent - Double Twin"],
    },
    {
        "keep": "Closer Power RV",
        "exclude": ["Power RV"],
    },
    {
        "keep": "Luxury Bell Tent - Double Twin",
        "exclude": ["Luxury Bell Tent - Queen", "Four-Person Bell Tent"],
    },
]


def _stopwords() -> set[str]:
    """Load NLTK's English stopword list, downloading it once if needed."""
    import nltk
    from nltk.corpus import stopwords

    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")
    return set(words)


def load_documents(path: str | Path = DEFAULT_DOCUMENTS_PATH) -> pd.DataFrame:
    """Read the document CSV without writing it."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"documents.csv is missing columns: {', '.join(missing)}")
    return frame.loc[:, COLUMNS].copy()


def validate_documents(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Report rows that cannot be indexed because required text is missing."""
    empty_content = frame.loc[frame["content"].str.strip() == "", "document_id"].tolist()
    empty_title = frame.loc[frame["title"].str.strip() == "", "document_id"].tolist()
    duplicate_ids = (
        frame.loc[frame["document_id"].duplicated(keep=False), "document_id"].tolist()
    )
    return {
        "empty_content": empty_content,
        "empty_title": empty_title,
        "duplicate_ids": duplicate_ids,
    }


def clean_text(text: str) -> str:
    """Normalize a copy of document or query text for indexing.

    HTML tags are removed if any remain. Whitespace is collapsed. A standalone
    trailing label such as "Learn More" is removed. Words inside a sentence,
    including "learn more about parking", are kept.
    """
    without_tags = _TAG.sub(" ", text or "")
    collapsed = re.sub(r"\s+", " ", without_tags.replace("\u200d", "")).strip()
    return strip_ui_labels(collapsed)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, tokenize, and optionally drop stopwords.

    The same function is used for documents and queries so their vectors
    live in one vocabulary.
    """
    cleaned = clean_text(text).lower()
    tokens = _TOKENIZER.findall(cleaned)
    if not remove_stopwords:
        return tokens
    blocked = _stopwords() - PRESERVE_TOKENS
    return [token for token in tokens if token not in blocked and len(token) > 1]


def _porter():
    """Return a Porter stemmer. The algorithm is local and needs no extra corpus."""
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def tokenize_normalized(text: str) -> list[str]:
    """Tokenize like the baseline, then stem words that are safe to stem.

    Porter stemming is a rule-based suffix chopper. It is applied to the
    document and to the query, so "shuttle" and "shuttles" land on one term
    ("shuttl"), and "camp"/"camping" and "pack"/"packing" land on "camp" and
    "pack". A WordNet lemmatizer was not used: without a part-of-speech tag it
    leaves the nouns "camping" and "packing" unchanged, so those pairs would
    still miss each other.

    Negation and the names in ``STEM_EXEMPT`` skip the stemmer.
    """
    stemmer = _porter()
    normalized = []
    for token in tokenize(text):
        if token in STEM_EXEMPT:
            normalized.append(token)
            continue
        stemmed = stemmer.stem(token)
        if len(stemmed) > 1:
            normalized.append(stemmed)
    return normalized


def exclusion_map() -> dict[str, str]:
    """Map an excluded Bonnaroo title to the title that was kept instead."""
    mapping: dict[str, str] = {}
    for group in NEAR_DUPLICATE_GROUPS:
        keep = str(group["keep"])
        for title in group["exclude"]:  # type: ignore[union-attr]
            mapping[str(title)] = keep
    return mapping


def filter_near_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the full collection into the indexing set and the held-out copies.

    Matching uses festival name plus the exact title from the Milestone 1
    review. Accessible-camping variants are not in those groups, so they stay.
    """
    excluded_titles = exclusion_map()
    titles = frame["title"].str.strip()
    is_copy = (frame["festival_name"] == "Bonnaroo") & titles.isin(excluded_titles)
    excluded = frame.loc[is_copy].copy()
    excluded["retained_title"] = titles.loc[is_copy].map(excluded_titles)
    working = frame.loc[~is_copy].copy()
    return working, excluded


def prepare_collection(
    path: str | Path = DEFAULT_DOCUMENTS_PATH,
    manifest_path: str | Path | None = DEFAULT_MANIFEST_PATH,
) -> dict[str, object]:
    """Load, validate, and filter documents for indexing.

    Returns the untouched full table, the working table, a validation report,
    and the near-duplicate rows. When ``manifest_path`` is set, writes one
    row per document saying whether it was indexed and why. That manifest
    does not contain document text.
    """
    full = load_documents(path)
    validation = validate_documents(full)
    usable = full.loc[full["content"].str.strip() != ""].copy()
    working, excluded = filter_near_duplicates(usable)
    if manifest_path is not None:
        _write_manifest(full, excluded, Path(manifest_path))
    return {
        "full": full,
        "working": working.reset_index(drop=True),
        "excluded": excluded.reset_index(drop=True),
        "validation": validation,
    }


def _write_manifest(full: pd.DataFrame, excluded: pd.DataFrame, path: Path) -> None:
    """Record which document IDs are indexed without copying their text."""
    excluded_ids = set(excluded["document_id"])
    retained = dict(zip(excluded["document_id"], excluded["retained_title"], strict=True))
    empty_ids = set(full.loc[full["content"].str.strip() == "", "document_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["document_id", "festival_name", "title", "action", "reason", "retained_title"],
        )
        writer.writeheader()
        for row in full.itertuples(index=False):
            if row.document_id in empty_ids:
                action, reason, kept = "excluded", "empty content", ""
            elif row.document_id in excluded_ids:
                action, reason, kept = "excluded", "near-duplicate Bonnaroo pass", retained[row.document_id]
            else:
                action, reason, kept = "indexed", "", ""
            writer.writerow(
                {
                    "document_id": row.document_id,
                    "festival_name": row.festival_name,
                    "title": row.title,
                    "action": action,
                    "reason": reason,
                    "retained_title": kept,
                }
            )


def main() -> None:
    """Validate the CSV and write the indexing manifest."""
    prepared = prepare_collection()
    validation = prepared["validation"]
    working = prepared["working"]
    excluded = prepared["excluded"]
    full = prepared["full"]
    print(f"Documents in {DEFAULT_DOCUMENTS_PATH.name}: {len(full)}")
    print(f"Empty content: {len(validation['empty_content'])}")
    print(f"Empty titles: {len(validation['empty_title'])}")
    print(f"Near-duplicates held out of the index: {len(excluded)}")
    print(f"Working collection for indexing: {len(working)}")
    print(f"Manifest: {DEFAULT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
