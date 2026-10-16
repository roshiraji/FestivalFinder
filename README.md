# FestivalFinder

**FestivalFinder** is a Python information retrieval system for CAP 6776 (Information Retrieval) at Florida Atlantic University. It helps users find relevant information about popular U.S. music festivals—transportation, lodging, rules, weather preparation, and attendee-oriented guidance—using classic IR techniques rather than generative AI or travel booking.

**Course:** CAP 6776 · **Due:** November 8, 2026 · **Stack:** local Python only (no GPU, paid APIs, or cloud IR services required)

---

## Project objective

Search a curated collection of official festival FAQ and guide pages for queries such as:

- Festival transportation, parking, and shuttle services
- Hotels, lodging, and camping accommodations
- Festival rules, policies, and permitted items
- Weather preparation, packing, and clothing
- Experiences and recommendations from previous attendees

**Festivals:** Coachella, Bonnaroo, Lollapalooza, EDC Las Vegas, Outside Lands

Documents come from official festival websites and FAQ guides. The system does not fabricate festival information.

---

## Current progress

Collection, preprocessing, TF-IDF, BM25, and the evaluation worksheet are implemented. Manual relevance labels are still blank, so Precision@5, Precision@10, and MAP have not been scored. The Streamlit interface is not built yet.

| Piece | State |
|-------|--------|
| Official-page collection | 257 documents from five festivals |
| Working collection | 246 documents indexed; 11 Bonnaroo near-duplicate passes are recorded in the manifest and left out of the index |
| Baseline TF-IDF | Body text only. Original rankings stay reproducible |
| Improved TF-IDF | Porter stemming, title weight 2, and festival filtering |
| BM25 | Same tokens, title weight, and festival filter as the improved TF-IDF ranker |
| Search | Interactive command line with `--mode baseline`, `--mode improved`, and `--mode bm25` |
| Evaluation setup | 12 queries and a 166-row judgment pool with a blank `relevance` column |
| Metrics code | Precision@5, Precision@10, and condensed-list MAP. Scores print only after every label is filled |
| Streamlit | Not started |

---

## Architecture

```text
Official festival pages
        │
        ▼
  collect.py  ──►  data/raw + data/processed/documents.csv
        │
        ▼
 preprocess.py  ──►  cleaned / tokenized text (original CSV is not rewritten)
        │
        ├──►  index.py   ──►  baseline TF-IDF  |  improved TF-IDF
        └──►  bm25.py    ──►  BM25Okapi
                    │
                    ▼
              search.py  ──►  top results with title, festival, score, snippet, URL
                    │
                    ▼
             evaluate.py ──►  judgment pool, then Precision@5, Precision@10, and MAP
                    │
                    ▼
               app.py    ──►  Streamlit UI (not built yet)
```

| Component | Role |
|-----------|------|
| `src/collect.py` | Fetch and parse permitted official pages; write CSV documents |
| `src/preprocess.py` | Validate the CSV, clean copies of the text, tokenize, remove stopwords, and stem for the improved rankers |
| `src/index.py` | Build the baseline and improved TF-IDF matrixes and the term-to-document lookup |
| `src/bm25.py` | Build the BM25 index with `rank-bm25` |
| `src/search.py` | Interactive search and the five-query demo for all three rankers |
| `src/evaluate.py` | Build the blank judgment pool; score Precision@5, Precision@10, and MAP after the labels are complete |
| `app.py` | Streamlit search UI, still a stub |

The original `documents.csv` is read-only for indexing and search. Display text in search results comes from that file.

---

## Repository structure

```text
FestivalFinder/
├── data/
│   ├── raw/                         # seed_urls.csv; local HTML dumps stay untracked
│   ├── processed/
│   │   ├── documents.csv            # 257 collected documents
│   │   └── indexing_manifest.csv    # which of those 257 are indexed
│   └── evaluation/
│       ├── queries.csv              # 12 evaluation queries
│       ├── judgment_pool.csv        # 166 rows awaiting 0/1 labels
│       └── relevance_judgments.csv  # unused header-only placeholder
├── src/
│   ├── collect.py
│   ├── preprocess.py
│   ├── index.py
│   ├── bm25.py
│   ├── search.py
│   └── evaluate.py
├── tests/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

### Documents — `data/processed/documents.csv`

| Field | Description |
|-------|-------------|
| `document_id` | Unique document identifier |
| `festival_name` | Coachella, Bonnaroo, Lollapalooza, EDC Las Vegas, or Outside Lands |
| `festival_year` | Year the guide or page applies to |
| `category` | transportation, lodging, rules, weather, or experiences |
| `title` | Document or section title |
| `content` | Main searchable text |
| `source_type` | official_faq or official_guide |
| `source_url` | Canonical URL |
| `published_date` | Publication or last-updated date when available |

The file holds wording copied from official pages. Keep it local unless a festival site permits redistribution.

Indexing uses 246 of these rows. Eleven Bonnaroo lodging passes were near-duplicates of a longer kept copy. They remain in `documents.csv`. `indexing_manifest.csv` records the held-out title and the title that was kept.

### Evaluation — `data/evaluation/`

- **`queries.csv`:** `query_id`, `query_text`, `description` (transportation, lodging, rules, packing, or experiences). Includes the five original demo queries plus seven more.
- **`judgment_pool.csv`:** `query_id`, `query_text`, `document_id`, `festival_name`, `title`, `category`, `source_url`, `relevance`. One row per query and document in the union of each ranker's top 10. `relevance` is blank until labeled `1` (answers the question) or `0` (does not).
- **`relevance_judgments.csv`:** earlier header-only placeholder. The worksheet to fill is `judgment_pool.csv`.

---

## How to run it

From the FestivalFinder project folder in PowerShell:

```powershell
python -m src.search
python -m src.search --mode baseline
python -m src.search --mode improved
python -m src.search --mode bm25
python -m src.search --demo
python -m src.search --mode baseline --demo
python -m src.search --mode bm25 --demo
python -m src.search --mode bm25 --no-festival-filter
python -m src.evaluate --make-pool
python -m src.evaluate
```

`python -m src.search` opens the interactive improved TF-IDF ranker. Type a question, read the top five results, and type `exit` to quit.

| Command | What it runs |
|---------|----------------|
| `--mode baseline` | Original body-only TF-IDF. No stemming and no festival filter |
| `--mode improved` | Stemmed, title-weighted TF-IDF. A named festival limits the results |
| `--mode bm25` | BM25 with the same text handling and festival filter as improved TF-IDF |
| `--demo` | The five original sample queries, then exit |
| `--no-festival-filter` | Search every festival even when the question names one |
| `--title-weight` | How many times each non-festival title term is counted. Default: 2 |
| `--k1`, `--b` | BM25 term-frequency saturation and length penalty. Defaults: 1.5 and 0.75 |

`python -m src.evaluate --make-pool` rewrites `judgment_pool.csv` only while every `relevance` cell is still blank. After any label is entered, that command refuses to overwrite the file. `python -m src.evaluate` reports per-query and average Precision@5, Precision@10, and MAP only after every cell is `0` or `1`.

### What the metrics mean

Precision@5 is the fraction of the first five results marked relevant. Precision@10 uses the first ten. The denominator stays 5 or 10 when the ranker returns fewer documents, because those missing positions are not retrieved documents. A retrieved document with a blank label is different: that metric is skipped rather than treating the blank as irrelevant.

Mean average precision averages, for each query, the precision at each relevant document's rank, then averages those query scores. Documents that never entered the judgment pool are left out of that ranked list. They are not marked irrelevant. This is condensed-list MAP, and it can be higher than a MAP that counted unjudged documents as not relevant.

---

## Technology stack

| Library | Use |
|---------|-----|
| Python 3.11+ | Runtime |
| pandas | CSV / tabular data |
| requests + BeautifulSoup | Permitted page collection |
| NLTK | Stopwords and the Porter stemmer |
| scikit-learn | TF-IDF and cosine similarity |
| rank-bm25 | BM25 ranking |
| Streamlit | Local search UI (not built yet) |
| pytest | Tests for collection, preprocessing, TF-IDF, BM25, and evaluation math |

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
```

---

## Course concepts demonstrated

1. Document collection and preprocessing
2. Tokenization, stopword removal, and Porter stemming
3. TF-IDF document-term matrix and an inverted-index view of its columns
4. Query processing shared with the document text
5. Cosine similarity
6. BM25 document ranking
7. Pooled relevance judgments (labels still to be entered)
8. Precision@5, Precision@10, and MAP (implemented; not yet scored)

Possible later work: WordNet query expansion for wording mismatches such as "bring" versus "allowed" and "prohibited." Document classification and sentiment analysis are not part of the current system.

---

## Development milestones

| Milestone | Status | Goal |
|-----------|--------|------|
| **0. Repository setup** | Complete | Folders, schemas, stubs, README, dependencies |
| **1. Collection pipeline** | Complete | 257 documents from five official festival guides |
| **2. TF-IDF** | Complete | Baseline body-only TF-IDF, then title weighting, stemming, and festival filtering |
| **3. BM25 and evaluation setup** | Complete through the blank judgment pool | BM25 search mode, 12 queries, 166-row pool, metric functions |
| **4. Manual judgments and scored evaluation** | Next | Fill `relevance`, then report Precision@5, Precision@10, and MAP |
| **5. Streamlit UI** | Planned | Local query interface |
| **6. Presentation** | Planned | 15-minute demo of architecture, code, search, and evaluation |

---

## Deliverables checklist (course)

- [x] Functioning retrieval system (command line; three rankers)
- [x] At least five evaluation queries
- [ ] Manual relevance judgments
- [ ] Reported Precision@5 (code is ready; labels are not)
- [ ] 15-minute recorded presentation

---

## Academic integrity note

This project may use AI coding assistance. Implementations should remain modular and explainable so the student can present the system independently and disclose AI assistance in the final presentation.

---

## License / data use

Respect each festival site’s terms of use and robots.txt. Collect only permitted pages. Do not republish copyrighted material beyond fair academic use for this course project.
