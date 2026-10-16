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
│   ├── raw/                 # scraped HTML or intermediate dumps
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

Header-only CSV files are checked in; **no fictional festival rows** are generated. Real documents will be added by the collection pipeline.

### Evaluation — `data/evaluation/`

- **`queries.csv`:** `query_id`, `query_text`, `description`
- **`relevance_judgments.csv`:** `query_id`, `document_id`, `relevance` (planned: `0` / `1`)

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
| **0. Repository setup** | **In progress** | Folders, schemas, stubs, README, dependencies |
| **1. Collection pipeline** | Next | ~20–30 authentic documents from official guides |
| **2. Preprocessing** | Planned | Cleaning, tokenization, stopwords |
| **3. Indexing** | Planned | TF-IDF matrix and term→document lookup |
| **4. Search / ranking** | Planned | TF-IDF cosine + BM25; top-k with source links |
| **5. Evaluation** | Planned | ≥5 queries, judgments, Precision@5 |
| **6. Streamlit UI** | Planned | Local query interface |
| **7. Presentation** | Planned | 15-minute demo of architecture, code, search, evaluation |

**Current step:** Milestone 0 only. Collection, ranking, and Streamlit are **not** implemented yet.

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
