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

**Initial festivals:** Coachella, Bonnaroo, Lollapalooza, EDC Las Vegas, Outside Lands

Documents come from official festival websites and FAQ guides. Reddit discussions may be added later. The system must **not** fabricate festival information.

---

## Architecture (planned pipeline)

```text
Official festival pages
        │
        ▼
  collect.py  ──►  data/raw + data/processed/documents.csv
        │
        ▼
 preprocess.py  ──►  cleaned / tokenized text
        │
        ▼
   index.py  ──►  TF-IDF matrix + term→document lookup
        │
        ▼
  search.py  ──►  TF-IDF cosine similarity  |  BM25
        │
        ├──►  app.py (Streamlit UI)
        └──►  evaluate.py  ──►  Precision@5 (queries + judgments)
```

| Component | Role |
|-----------|------|
| `src/collect.py` | Fetch and parse permitted official pages; write CSV documents |
| `src/preprocess.py` | Clean HTML artifacts, normalize, tokenize, optional stopword removal |
| `src/index.py` | Build TF-IDF document representations and inverted-index-style lookup |
| `src/search.py` | Query processing; TF-IDF cosine and BM25 ranking; top-k results with URLs |
| `src/evaluate.py` | ≥5 test queries, relevance judgments, Precision@5 (optional TF-IDF vs BM25) |
| `app.py` | Streamlit search UI (titles, snippets, source links, optional method toggle) |

---

## Repository structure

```text
FestivalFinder/
├── data/
│   ├── raw/                 # seed_urls.csv; local HTML dumps stay untracked
│   ├── processed/
│   │   └── documents.csv    # document collection (schema only for now)
│   └── evaluation/
│       ├── queries.csv
│       └── relevance_judgments.csv
├── src/
│   ├── collect.py
│   ├── preprocess.py
│   ├── index.py
│   ├── search.py
│   └── evaluate.py
├── tests/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset schema

### Documents — `data/processed/documents.csv`

| Field | Description |
|-------|-------------|
| `document_id` | Unique document identifier |
| `festival_name` | e.g. Coachella, Bonnaroo |
| `festival_year` | Year the guide/page applies to |
| `category` | e.g. transportation, lodging, rules, weather, experiences |
| `title` | Document or section title |
| `content` | Main searchable text |
| `source_type` | e.g. official_faq, official_guide |
| `source_url` | Canonical URL |
| `published_date` | Publication or last-updated date when available |

`queries.csv` lists 12 evaluation questions. `judgment_pool.csv` is created by `python -m src.evaluate --make-pool` and starts with a blank relevance column. `documents.csv` is filled locally by `src/collect.py` with wording copied from official pages. Keep that file local unless a festival site permits redistribution. No fictional festival rows are generated.

### Evaluation — `data/evaluation/`

- **`queries.csv`:** `query_id`, `query_text`, `description` (transportation, lodging, rules, packing, or experiences)
- **`judgment_pool.csv`:** `query_id`, `query_text`, `document_id`, `festival_name`, `title`, `category`, `source_url`, `relevance` (blank until labeled `0` or `1`)
- **`relevance_judgments.csv`:** earlier header-only placeholder; the worksheet to fill is `judgment_pool.csv`

Target: at least five evaluation queries with manual judgments and **Precision@5**.

---

## Technology stack

| Library | Use |
|---------|-----|
| Python 3.11+ | Runtime |
| pandas | CSV / tabular data |
| requests + BeautifulSoup | Permitted page collection |
| NLTK | Tokenization / stopwords |
| scikit-learn | TF-IDF and cosine similarity |
| rank-bm25 | BM25 ranking |
| Streamlit | Local search UI |
| pytest | Tests (later) |

Install (when you begin coding against the stubs):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Course concepts demonstrated

1. Document collection and preprocessing  
2. Tokenization and text normalization  
3. Document representation and indexing  
4. Query processing  
5. TF-IDF and cosine similarity  
6. BM25 document ranking  
7. Relevance judgments  
8. Precision@5 evaluation  

Optional later: WordNet query expansion, document classification, sentiment analysis.

---

## Development milestones

| Milestone | Status | Goal |
|-----------|--------|------|
| **0. Repository setup** | Complete | Folders, schemas, stubs, README, dependencies |
| **1. Collection pipeline** | Complete | 257 documents from five official festival guides |
| **2. Preprocessing and TF-IDF** | Complete | Baseline TF-IDF plus a title-aware, festival-filtered ranker |
| **3. Indexing** | Complete for TF-IDF and BM25 | Same 246-document working collection |
| **4. Search / ranking** | **In review** | Interactive baseline TF-IDF, improved TF-IDF, and BM25 |
| **5. Evaluation** | **In review** | 12 queries and a blank judgment pool. Metrics wait for labels |
| **6. Streamlit UI** | Planned | Local query interface |
| **7. Presentation** | Planned | 15-minute demo of architecture, code, search, evaluation |

**Current step:** Milestone 3 BM25 and the evaluation worksheet are ready for review. Relevance labels are still blank, so Precision@5, Precision@10, and MAP are not reported. Streamlit is not implemented.

### Search from PowerShell

Run these commands from the FestivalFinder project folder.

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
```

`python -m src.search` opens the interactive improved TF-IDF ranker. Type a question, read the top five results, and type `exit` to quit. `--mode baseline` is the original body-only TF-IDF index. `--mode bm25` is BM25 with the same stemming, title weight, and festival filter as the improved TF-IDF ranker. `--demo` runs the five original sample queries in the selected mode and exits. `--no-festival-filter` searches every festival even when the question names one.

`python -m src.evaluate --make-pool` writes `data/evaluation/judgment_pool.csv`. Each row is one query and one document from the union of the three rankers' top 10 lists. The `relevance` column is blank. Enter `1` when the document answers the question and `0` when it does not. The command will not replace the file after any label has been entered. `python -m src.evaluate` reports Precision@5, Precision@10, and MAP only after every cell is filled.

### What the metrics mean

Precision@5 is the fraction of the first five results that you marked relevant. Precision@10 uses the first ten. The denominator stays 5 or 10 when the ranker returns fewer documents. Mean average precision walks down the judged list and averages precision at each relevant document, then averages that value across queries. Documents that never entered the judgment pool are left out of MAP. They are not marked irrelevant.

---

## Deliverables checklist (course)

- [ ] Functioning retrieval system  
- [ ] ≥5 evaluation queries with relevance judgments  
- [ ] ≥1 standard metric (Precision@5)  
- [ ] 15-minute recorded presentation  

---

## Academic integrity note

This project may use AI coding assistance. Implementations should remain modular and explainable so the student can present the system independently and disclose AI assistance in the final presentation.

---

## License / data use

Respect each festival site’s terms of use and robots.txt. Collect only permitted pages. Do not republish copyrighted material beyond fair academic use for this course project.
