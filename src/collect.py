"""Collect official festival guide text for FestivalFinder.

This module is the Milestone 1 collection pipeline. It reads a seed list of
official pages, checks robots.txt, downloads pages that are allowed, and
writes FAQ answers or guide sections to ``data/processed/documents.csv``.

How a run works
---------------
1. Read ``data/raw/seed_urls.csv``. Each row is one official page, with the
   festival name, year, a category hint, a source type, and the URL.
2. Before any guide page is downloaded, fetch that host's robots.txt and
   skip the URL when the file disallows it. A missing robots.txt (HTTP 404)
   means the site published no crawl rules.
3. Download the page with a timeout and a pause between requests. If the
   server returns an anti-bot challenge, stop requesting that host for the
   rest of the run. Those protections are not bypassed.
4. Parse the page into separate documents only where the HTML already marks
   a real unit: an FAQ question and answer, a help-article section, or a
   named lodging/parking description. Navigation, footers, scripts, and
   partner lists are removed. The wording is copied from the page.
5. Keep a document only when it has enough text to search and it matches
   one of the project topics (transportation, lodging, rules, weather,
   experiences). Ticketing, shipping, lockers, vendors, and media requests
   are left out.
6. Assign a stable document id from the festival, year, title, and source
   URL. Drop duplicate text. Leave ``published_date`` blank when the page
   does not show one.
7. Write ``documents.csv``. Existing data rows are kept. A later run appends
   new ids and does not replace rows already stored. Use ``--replace`` only
   when the previous collected rows should be discarded.

The other pipeline modules (preprocessing, indexing, search, evaluation,
and Streamlit) are not used here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_PATH = PROJECT_ROOT / "data" / "raw" / "seed_urls.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "documents.csv"
DEFAULT_HTML_DIR = PROJECT_ROOT / "data" / "raw" / "html"

COLUMNS = [
    "document_id",
    "festival_name",
    "festival_year",
    "category",
    "title",
    "content",
    "source_type",
    "source_url",
    "published_date",
]

# Official festival hosts. Help-center hosts are included so a block can be
# recorded; a challenge stops any further requests to that host.
ALLOWED_HOSTS = {
    "www.coachella.com",
    "coachella.com",
    "www.bonnaroo.com",
    "bonnaroo.com",
    "support.bonnaroo.com",
    "www.lollapalooza.com",
    "lollapalooza.com",
    "support.lollapalooza.com",
    "lasvegas.electricdaisycarnival.com",
    "sfoutsidelands.com",
    "www.sfoutsidelands.com",
}

VALID_CATEGORIES = {"transportation", "lodging", "rules", "weather", "experiences"}

USER_AGENT = (
    "FestivalFinder/0.1 (CAP 6776 academic information retrieval project; "
    "local coursework collection of public festival guides)"
)
REQUEST_TIMEOUT = (10, 20)
REQUEST_DELAY_SECONDS = 2.0
MIN_CONTENT_CHARS = 80

# Whole FAQ groups that are outside the project's search topics.
SKIP_SECTIONS = {
    "passes, order & shipping",
    "ga lockers",
    "charging lockers",
    "wristband & mobile info",
    "download & explore",
    "register via coachella mobile app only",
}

# Questions about ticketing operations, promotion, or jobs, not attendee guides.
OFF_TOPIC_QUESTION = re.compile(
    r"vendor|hiring|photographer|\bmedia\b|flyer|promotional|scalper|"
    r"refund|shipping|\border\b|wristband|payment plan|will call|"
    r"locker|insurance|debit card|presale|live tours|mobile app|"
    r"\bsell\b|vending|transferable",
    re.IGNORECASE,
)

CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "weather",
        re.compile(
            r"weather|how hot|how cold|\brain\b|\bpacking\b|\bpack\b|sunscreen|"
            r"\bheat\b|hoodie|temperature|sun protection",
            re.IGNORECASE,
        ),
    ),
    (
        "transportation",
        re.compile(
            r"parking|shuttle|rideshare|ride-share|airport(?!-)|transit|\btaxi\b|"
            r"\blimo\b|getting here|getting around|drop-?off|pick-?up|\btrain\b|\bbus\b|"
            r"motorcycle|direction|biking|walking|\btravel\b|\barriv|"
            r"bicycle|\bbike\b|gas station|charging station",
            re.IGNORECASE,
        ),
    ),
    (
        "lodging",
        re.compile(
            r"\bcamp|hotel|lodging|glamp|\brv\b|motorhome|accommodation|"
            r"\byurt\b|\bresort\b|campsite",
            re.IGNORECASE,
        ),
    ),
    (
        "rules",
        re.compile(
            r"allowed|prohibited|not allowed|what can i bring|\bpolicy\b|"
            r"\brules?\b|safety|permitted|banned|\bage\b|evacuation|\bpolicies\b",
            re.IGNORECASE,
        ),
    ),
    (
        "experiences",
        re.compile(
            r"\btips?\b|restroom|\bshade\b|lost & found|medical|\bfood\b|"
            r"\bwater\b|\bmap\b|amenit|comfortable|"
            r"at the festival|festival end|festival info|festival hours",
            re.IGNORECASE,
        ),
    ),
]

CHALLENGE_MARKERS = (
    "just a moment",
    "cf-browser-verification",
    "attention required",
    "enable javascript and cookies",
    "checking your browser",
)

logger = logging.getLogger("festivalfinder.collect")


# Button labels left behind when a link or accordion control is removed.
_UI_LABEL = r"learn more|read more|show more|show less|see more|view more"
_UI_LABELS = re.compile(
    rf"(?:(?<=^)|(?<=[.!?]))\s*(?:{_UI_LABEL})\b(?!\s+[A-Za-z])\.?"
    rf"|\s+(?:{_UI_LABEL})\s*$"
    rf"|\bexpand(?:expand)+\b",
    re.IGNORECASE,
)


def strip_ui_labels(text: str) -> str:
    """Remove leftover control labels without deleting those words inside a sentence.

    "Learn more about parking" stays intact. A trailing "Learn More" after a
    sentence, or a repeated "expandexpand" control, is removed.
    """
    cleaned = _UI_LABELS.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_text(text: str) -> str:
    """Collapse whitespace and drop leftover interface labels."""
    text = text.replace("\u200d", "").replace("\ufeff", "")
    return strip_ui_labels(re.sub(r"\s+", " ", text).strip())


def normalize_heading(text: str) -> str:
    return normalize_text(text).casefold()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "document"


def make_document_id(festival_name: str, festival_year: str, title: str, source_url: str) -> str:
    """Build an id that stays the same when the same section is collected again.

    The hash includes the source URL so two sections with the same title do
    not collide, and so re-running the collector does not invent a new id.
    """
    slug = slugify(f"{festival_name}-{festival_year}-{title}")[:70].strip("-")
    digest = hashlib.sha1(
        f"{festival_name}|{festival_year}|{source_url}|{title}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{slug}-{digest}"


def infer_category(*parts: str) -> str | None:
    """Map a heading or question onto one project topic.

    The question is checked before the section heading, so a hotel question
    inside a travel section is stored as lodging. When a title mentions both
    parking and camping, lodging wins because the question is about where
    to stay.
    """
    for part in parts:
        text = normalize_text(part)
        if not text:
            continue
        matches = [category for category, pattern in CATEGORY_PATTERNS if pattern.search(text)]
        if not matches:
            continue
        if "lodging" in matches and re.search(
            r"\bcamp|hotel|\brv\b|motorhome", text, re.IGNORECASE
        ):
            return "lodging"
        return matches[0]
    return None


def is_access_challenge(status_code: int, body: str) -> bool:
    """True when the response is an anti-bot interstitial rather than a guide."""
    lowered = body.casefold()
    if any(marker in lowered for marker in CHALLENGE_MARKERS):
        return True
    return status_code == 403 and "cloudflare" in lowered


def evaluate_robots(status_code: int, body: str, url: str, user_agent: str = USER_AGENT) -> tuple[bool, str]:
    """Decide whether robots.txt allows ``url``.

    HTTP 404 means the host did not publish a robots.txt file, so there is
    no disallow rule to apply. A challenge page is treated as "do not fetch,"
    because the rules could not be read.
    """
    if is_access_challenge(status_code, body):
        return False, "robots.txt request was blocked by an anti-bot challenge"
    if status_code == 404:
        return True, "no robots.txt was published (HTTP 404)"
    if status_code != 200:
        return False, f"robots.txt could not be read (HTTP {status_code})"

    parser = RobotFileParser()
    parser.parse(body.splitlines())
    if parser.can_fetch(user_agent, url):
        return True, "robots.txt allows this URL"
    return False, "robots.txt disallows this URL"


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _page_key(url: str) -> str:
    """Compare page URLs without a fragment or a trailing slash."""
    return url.split("#", 1)[0].rstrip("/").casefold()


def make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _remove_noise(root: Tag) -> None:
    for tag in root.select("script, style, noscript, svg, iframe, form, nav, footer, button"):
        tag.decompose()


def _extract_year(soup: BeautifulSoup, fallback: str) -> str:
    """Use a year printed in the page title or top heading.

    The seed year is only a fallback when the page itself does not name one.
    """
    sources: list[str] = []
    if soup.title:
        sources.append(soup.title.get_text(" ", strip=True))
    heading = soup.find("h1")
    if heading:
        sources.append(heading.get_text(" ", strip=True))
    for text in sources:
        years = re.findall(r"\b(20\d{2})\b", text)
        if years:
            return years[-1]
    return fallback if re.fullmatch(r"20\d{2}", fallback or "") else ""


def _extract_published_date(soup: BeautifulSoup) -> str:
    timestamp = soup.select_one("time[datetime]")
    if timestamp and timestamp.get("datetime"):
        match = re.search(r"\d{4}-\d{2}-\d{2}", timestamp["datetime"])
        if match:
            return match.group(0)
    return ""


def _row(
    *,
    festival_name: str,
    festival_year: str,
    category: str,
    title: str,
    content: str,
    source_type: str,
    source_url: str,
    published_date: str,
) -> dict[str, str]:
    title = normalize_text(title)
    content = normalize_text(content)
    return {
        "document_id": make_document_id(festival_name, festival_year, title, source_url),
        "festival_name": festival_name,
        "festival_year": festival_year,
        "category": category,
        "title": title,
        "content": content,
        "source_type": source_type,
        "source_url": source_url,
        "published_date": published_date,
    }


def compose_title(section: str, question: str) -> str:
    """Keep the page's own heading, and add the section name when it is distinct."""
    section = normalize_text(section)
    question = normalize_text(question)
    if not section or section.casefold() in question.casefold():
        return question
    return f"{section}: {question}"


def _accept_section(
    section: str,
    title: str,
    content: str,
    category_hint: str,
) -> str | None:
    """Return a category when this section is on-topic and long enough."""
    if normalize_heading(section) in SKIP_SECTIONS:
        return None
    if OFF_TOPIC_QUESTION.search(title):
        return None
    if len(normalize_text(content)) < MIN_CONTENT_CHARS:
        return None
    # This whole Coachella block is the permitted-items policy, including
    # medical items and bag sizes. Question keywords such as "medical" or
    # "backpack" would otherwise file those answers under another topic.
    if normalize_heading(section) == "what can i bring?":
        return "rules"
    category = infer_category(title, content[:500], section)
    if category is None and category_hint in VALID_CATEGORIES:
        category = category_hint
    return category


def _parse_accordions(
    soup: BeautifulSoup,
    page_url: str,
    festival_name: str,
    festival_year: str,
    category_hint: str,
    source_type: str,
    published_date: str,
) -> list[dict[str, str]]:
    """Turn each Coachella-style FAQ block into its own document."""
    documents: list[dict[str, str]] = []
    blocks = soup.select("details.c-accordion__block, details.c-accordion-wysiwyg__block")
    for block in blocks:
        section_root = block.find_parent(class_="js-accordion-section")
        heading_tag = section_root.find("h2") if isinstance(section_root, Tag) else None
        section = heading_tag.get_text(" ", strip=True) if heading_tag else ""
        question_tag = block.select_one(
            ".c-accordion__question, .c-accordion-wysiwyg__question"
        )
        answer_tag = block.select_one(
            ".c-accordion__answer, .c-accordion-wysiwyg__answer"
        )
        if question_tag is None or answer_tag is None:
            continue
        title = compose_title(section, question_tag.get_text(" ", strip=True))
        content = answer_tag.get_text(" ", strip=True)
        category = _accept_section(section, title, content, category_hint)
        if category is None:
            continue
        fragment = block.get("id")
        source_url = f"{page_url}#{fragment}" if fragment else page_url
        documents.append(
            _row(
                festival_name=festival_name,
                festival_year=festival_year,
                category=category,
                title=title,
                content=content,
                source_type=source_type,
                source_url=source_url,
                published_date=published_date,
            )
        )
    return documents


def _section_text_after(heading: Tag) -> str:
    chunks: list[str] = []
    for sibling in heading.find_next_siblings():
        if sibling.name in {"h2", "h3"}:
            break
        chunks.append(sibling.get_text(" ", strip=True))
    return normalize_text(" ".join(chunks))


def _parse_help_article(
    soup: BeautifulSoup,
    page_url: str,
    festival_name: str,
    festival_year: str,
    category_hint: str,
    source_type: str,
    published_date: str,
) -> list[dict[str, str]]:
    """Extract a Zendesk help article, splitting on real headings when present."""
    body = soup.select_one(".article-body")
    title_tag = soup.select_one("h1.article-title")
    if body is None or title_tag is None:
        return []
    _remove_noise(body)
    page_title = title_tag.get_text(" ", strip=True)
    headings = [tag for tag in body.find_all(["h2", "h3"]) if tag.get_text(strip=True)]
    pieces: list[tuple[str, str]] = []
    if len(headings) >= 2:
        intro_chunks: list[str] = []
        for child in body.children:
            if isinstance(child, Tag) and child.name in {"h2", "h3"}:
                break
            intro_chunks.append(child.get_text(" ", strip=True) if isinstance(child, Tag) else str(child))
        intro = normalize_text(" ".join(intro_chunks))
        if intro:
            pieces.append((page_title, intro))
        pieces.extend((heading.get_text(" ", strip=True), _section_text_after(heading)) for heading in headings)
    else:
        pieces.append((page_title, body.get_text(" ", strip=True)))

    documents: list[dict[str, str]] = []
    for title, content in pieces:
        category = _accept_section(page_title, title, content, category_hint)
        if category is None:
            continue
        documents.append(
            _row(
                festival_name=festival_name,
                festival_year=festival_year,
                category=category,
                title=title,
                content=content,
                source_type=source_type,
                source_url=page_url,
                published_date=published_date,
            )
        )
    return documents


def _parse_lodging_cards(
    soup: BeautifulSoup,
    page_url: str,
    festival_name: str,
    festival_year: str,
    category_hint: str,
    source_type: str,
    published_date: str,
) -> list[dict[str, str]]:
    """Extract named camping and parking descriptions from Bonnaroo's guide."""
    documents: list[dict[str, str]] = []
    for card in soup.select("span.ticket-info"):
        title = normalize_text(card.get("name") or "")
        description = card.find_parent(class_="ticket")
        if not title or description is None:
            continue
        content_tag = description.select_one(".description")
        if content_tag is None:
            continue
        content = content_tag.get_text(" ", strip=True)
        section = normalize_text(card.get("type") or "")
        category = _accept_section(section, title, content, category_hint)
        if category is None:
            continue
        documents.append(
            _row(
                festival_name=festival_name,
                festival_year=festival_year,
                category=category,
                title=title,
                content=content,
                source_type=source_type,
                source_url=page_url,
                published_date=published_date,
            )
        )
    return documents


SKIP_HEADINGS = {
    "related",
    "sponsors",
    "partners",
    "gallery",
    "presale",
    "tickets & wristbands",
    "get involved",
    "press",
    "food vendors",
    "ticketing faq",
    "press credentials",
    "web accessibility",
}


def _heading_skipped(label: str) -> bool:
    heading = normalize_heading(label)
    if heading in SKIP_HEADINGS or heading in SKIP_SECTIONS:
        return True
    if "partner" in heading or heading.startswith("sponsor"):
        return True
    return "short-cut" in heading or "scroll to each topic" in heading


def _parse_definition_accordions(
    soup: BeautifulSoup,
    page_url: str,
    festival_name: str,
    festival_year: str,
    category_hint: str,
    source_type: str,
    published_date: str,
) -> list[dict[str, str]]:
    """Extract question/answer pairs from definition-list accordions."""
    documents: list[dict[str, str]] = []
    for block in soup.select("dl.accordion"):
        question = block.find("dt")
        answer = block.find("dd")
        if question is None or answer is None:
            continue
        section_tag = block.find_previous(["h2", "h3"])
        section = section_tag.get_text(" ", strip=True) if section_tag else ""
        if _heading_skipped(section):
            continue
        title = compose_title(section, question.get_text(" ", strip=True))
        content = answer.get_text(" ", strip=True)
        category = _accept_section(section, title, content, category_hint)
        if category is None:
            continue
        documents.append(
            _row(
                festival_name=festival_name,
                festival_year=festival_year,
                category=category,
                title=title,
                content=content,
                source_type=source_type,
                source_url=page_url,
                published_date=published_date,
            )
        )
    return documents


def _parse_faq_items(
    soup: BeautifulSoup,
    page_url: str,
    festival_name: str,
    festival_year: str,
    category_hint: str,
    source_type: str,
    published_date: str,
) -> list[dict[str, str]]:
    """Extract FAQ items whose question is a heading inside each item."""
    documents: list[dict[str, str]] = []
    for item in soup.select("div.faqItem"):
        question_tag = item.find(["h4", "h3", "h2"])
        if question_tag is None:
            continue
        question = normalize_text(question_tag.get_text(" ", strip=True))
        section_tag = item.find_previous("h3")
        section = section_tag.get_text(" ", strip=True) if section_tag else ""
        if _heading_skipped(section) or _heading_skipped(question):
            continue
        content = normalize_text(item.get_text(" ", strip=True))
        if content.casefold().startswith(question.casefold()):
            content = content[len(question):].strip()
        title = compose_title(section, question)
        category = _accept_section(section, title, content, category_hint)
        if category is None:
            continue
        documents.append(
            _row(
                festival_name=festival_name,
                festival_year=festival_year,
                category=category,
                title=title,
                content=content,
                source_type=source_type,
                source_url=page_url,
                published_date=published_date,
            )
        )
    return documents


def _parse_heading_sections(
    soup: BeautifulSoup,
    page_url: str,
    festival_name: str,
    festival_year: str,
    category_hint: str,
    source_type: str,
    published_date: str,
) -> list[dict[str, str]]:
    """Extract guide sections that the page already splits with headings."""
    root = soup.select_one("main") or soup.body
    if root is None:
        return []
    documents: list[dict[str, str]] = []
    for heading in root.find_all(["h2", "h3"]):
        section = normalize_text(heading.get_text(" ", strip=True))
        if not section or _heading_skipped(section):
            continue
        chunks: list[str] = []
        for sibling in heading.find_next_siblings():
            if not isinstance(sibling, Tag):
                continue
            if sibling.name in {"h2", "h3", "h4"} or sibling.find(["h2", "h3", "h4"]):
                break
            chunks.append(sibling.get_text(" ", strip=True))
        content = normalize_text(" ".join(chunks))
        category = _accept_section("", section, content, category_hint)
        if category is None:
            continue
        documents.append(
            _row(
                festival_name=festival_name,
                festival_year=festival_year,
                category=category,
                title=section,
                content=content,
                source_type=source_type,
                source_url=page_url,
                published_date=published_date,
            )
        )
    return documents


def _dedupe_documents(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop repeated sections from a single page before they reach the CSV."""
    unique: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for document in documents:
        content_key = _content_key(document["content"])
        if document["document_id"] in seen_ids or content_key in seen_content:
            continue
        seen_ids.add(document["document_id"])
        seen_content.add(content_key)
        unique.append(document)
    return unique


def parse_documents(
    html: str,
    url: str,
    festival_name: str,
    festival_year: str,
    category_hint: str,
    source_type: str,
) -> list[dict[str, str]]:
    """Extract on-topic guide documents from one downloaded page.

    Returns an empty list when the page has no recognizable FAQ answers or
    guide sections. It does not invent text to fill the dataset.
    """
    soup = make_soup(html)
    if is_access_challenge(200, html) and soup.select_one(".article-body, details.c-accordion__block") is None:
        return []

    year = _extract_year(soup, festival_year)
    published_date = _extract_published_date(soup)
    for tag in soup.select("script, style, noscript, svg, iframe, nav, footer"):
        tag.decompose()

    if soup.select_one(".article-body") and soup.select_one("h1.article-title"):
        parsed = _parse_help_article(
            soup, url, festival_name, year, category_hint, source_type, published_date
        )
    elif soup.select_one("span.ticket-info[name]"):
        parsed = _parse_lodging_cards(
            soup, url, festival_name, year, category_hint, source_type, published_date
        )
    elif soup.select("details.c-accordion__block, details.c-accordion-wysiwyg__block"):
        parsed = _parse_accordions(
            soup, url, festival_name, year, category_hint, source_type, published_date
        )
    elif soup.select("div.faqItem"):
        parsed = _parse_faq_items(
            soup, url, festival_name, year, category_hint, source_type, published_date
        )
    elif len(soup.select("dl.accordion")) >= 3:
        parsed = _parse_definition_accordions(
            soup, url, festival_name, year, category_hint, source_type, published_date
        )
    elif soup.select("main h2, main h3, article h2, article h3"):
        parsed = _parse_heading_sections(
            soup, url, festival_name, year, category_hint, source_type, published_date
        )
    else:
        logger.warning("No FAQ or guide sections recognized at %s", url)
        return []
    return _dedupe_documents(parsed)


def load_seed_rows(seed_path: Path) -> list[dict[str, str]]:
    with seed_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in ("festival_name", "festival_year", "category", "source_type", "url") if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Seed file is missing columns: {', '.join(missing)}")
        rows = []
        for row in reader:
            if not (row.get("url") or "").strip():
                continue
            rows.append({key: (row.get(key) or "").strip() for key in row})
        return rows


def load_existing_documents(output_path: Path) -> list[dict[str, str]]:
    if not output_path.exists():
        return []
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        documents = []
        for row in reader:
            if not (row.get("document_id") or "").strip():
                continue
            documents.append({column: (row.get(column) or "").strip() for column in COLUMNS})
        return documents


def write_documents(output_path: Path, documents: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(documents)


def _content_key(content: str) -> str:
    return normalize_text(content).casefold()


def _decode_response(response: requests.Response) -> str:
    """Decode a page as UTF-8 when those bytes are valid.

    Coachella sends UTF-8 HTML without a charset header. Requests would
    otherwise treat the body as Latin-1 and turn apostrophes into mojibake.
    """
    raw = response.content
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        encoding = response.apparent_encoding or "utf-8"
        return raw.decode(encoding, errors="replace")


class Collector:
    """Fetch allowed pages and merge them into the document CSV.

    ``fetcher`` and ``robots_reader`` can be replaced in tests so the unit
    tests do not touch the network.
    """

    def __init__(
        self,
        fetcher: Callable[[str], tuple[int, str]] | None = None,
        robots_reader: Callable[[str], tuple[int, str]] | None = None,
        delay_seconds: float = REQUEST_DELAY_SECONDS,
        html_dir: Path = DEFAULT_HTML_DIR,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.html_dir = html_dir
        self._last_request = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._fetcher = fetcher or self._fetch
        self._robots_reader = robots_reader or self._fetch
        self._robots_cache: dict[str, tuple[int, str]] = {}
        self.blocked_hosts: set[str] = set()
        self.events: list[str] = []

    def _pause(self) -> None:
        if self._last_request:
            remaining = self.delay_seconds - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request = time.monotonic()

    def _fetch(self, url: str) -> tuple[int, str]:
        self._pause()
        try:
            response = self._session.get(url, timeout=REQUEST_TIMEOUT)
            return response.status_code, _decode_response(response)
        except requests.RequestException as exc:
            self._log(f"Request failed for {url}: {exc}")
            return 0, ""

    def _log(self, message: str) -> None:
        self.events.append(message)
        logger.info(message)

    def _robots_allows(self, url: str) -> bool:
        origin = _origin(url)
        if origin not in self._robots_cache:
            status, body = self._robots_reader(urljoin_robots(origin))
            self._robots_cache[origin] = (status, body)
            if is_access_challenge(status, body):
                host = urlparse(url).netloc.lower()
                self.blocked_hosts.add(host)
        status, body = self._robots_cache[origin]
        allowed, reason = evaluate_robots(status, body, url)
        self._log(f"{url}: {reason}")
        return allowed

    def collect(
        self,
        seed_rows: list[dict[str, str]],
        output_path: Path,
        festivals: set[str] | None = None,
        replace: bool = False,
    ) -> dict[str, int]:
        """Collect seed pages and return counts for the run summary."""
        existing = [] if replace else load_existing_documents(output_path)
        if replace and output_path.exists() and load_existing_documents(output_path):
            self._log(
                "Replace mode is on. Previously collected rows in "
                f"{output_path} will be discarded and rewritten from this run."
            )
        elif existing:
            self._log(
                f"Preserving {len(existing)} existing document rows. "
                "Matching ids will not be overwritten. New ids will be appended."
            )
        else:
            self._log(
                "The output CSV has no document rows yet. This run will write "
                "the first collected rows under the existing header. "
                "Later runs keep those rows unless --replace is used."
            )

        kept_ids = {row["document_id"] for row in existing}
        seen_content = {_content_key(row["content"]) for row in existing if row["content"]}
        existing_pages = {_page_key(row["source_url"]) for row in existing}
        collected: list[dict[str, str]] = []
        stats = {
            "pages_fetched": 0,
            "pages_failed": 0,
            "pages_blocked": 0,
            "pages_not_requested": 0,
            "pages_already_collected": 0,
            "documents_extracted": 0,
            "documents_added": 0,
            "duplicates_skipped": 0,
            "existing_preserved": len(existing),
        }
        selected = [
            row for row in seed_rows
            if festivals is None or row["festival_name"].casefold() in festivals
        ]

        for seed in selected:
            url = seed["url"]
            host = urlparse(url).netloc.lower()
            festival_name = seed["festival_name"]
            if host not in ALLOWED_HOSTS:
                stats["pages_failed"] += 1
                self._log(f"Skipped {url}: host is not an approved festival site")
                continue
            if _page_key(url) in existing_pages and not replace:
                stats["pages_already_collected"] += 1
                self._log(
                    f"Preserved existing documents from {url}. "
                    "The page was not downloaded again."
                )
                continue
            if host in self.blocked_hosts:
                stats["pages_not_requested"] += 1
                self._log(
                    f"Skipped {url}: {host} already returned an anti-bot challenge, "
                    "so no further automated requests were sent to that host."
                )
                continue
            if not self._robots_allows(url):
                stats["pages_failed"] += 1
                continue

            status, html = self._fetcher(url)
            if status == 0:
                stats["pages_failed"] += 1
                continue
            if is_access_challenge(status, html):
                stats["pages_blocked"] += 1
                self.blocked_hosts.add(host)
                self._log(
                    f"Skipped {url}: HTTP {status} anti-bot challenge. "
                    "The page was not parsed and the host will not be requested again this run."
                )
                continue
            if status != 200 or not html.strip():
                stats["pages_failed"] += 1
                self._log(f"Skipped {url}: HTTP {status} and no document was created")
                continue

            stats["pages_fetched"] += 1
            self._save_html(url, html)
            documents = parse_documents(
                html,
                url,
                festival_name,
                seed.get("festival_year", ""),
                seed.get("category", ""),
                seed.get("source_type", "official_guide"),
            )
            self._log(f"Parsed {len(documents)} on-topic sections from {url}")
            for document in documents:
                stats["documents_extracted"] += 1
                content_key = _content_key(document["content"])
                if document["document_id"] in kept_ids or content_key in seen_content:
                    stats["duplicates_skipped"] += 1
                    continue
                kept_ids.add(document["document_id"])
                seen_content.add(content_key)
                collected.append(document)
                stats["documents_added"] += 1

        write_documents(output_path, existing + collected)
        stats["documents_in_file"] = len(existing) + len(collected)
        return stats

    def _save_html(self, url: str, html: str) -> None:
        self.html_dir.mkdir(parents=True, exist_ok=True)
        path = self.html_dir / f"{slugify(url)[:90]}.html"
        path.write_text(html, encoding="utf-8")


def urljoin_robots(origin: str) -> str:
    return origin.rstrip("/") + "/robots.txt"


def collect_documents(
    seed_path: str | Path = DEFAULT_SEED_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    festivals: list[str] | None = None,
    replace: bool = False,
) -> dict[str, int]:
    """Collect the seed pages and return run counts, including documents stored."""
    collector = Collector()
    seed_rows = load_seed_rows(Path(seed_path))
    selected = {name.casefold() for name in festivals} if festivals else None
    stats = collector.collect(seed_rows, Path(output_path), festivals=selected, replace=replace)
    logger.info("Collection summary: %s", stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    """Download the seed pages and update the document CSV."""
    parser = argparse.ArgumentParser(description="Collect official festival guide documents.")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--festivals",
        default="",
        help="Comma-separated festival names to collect. Default: every festival in the seed file.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Discard previously collected rows and rewrite the CSV. Omit this to keep existing rows.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    festivals = [name.strip() for name in args.festivals.split(",") if name.strip()]
    stats = collect_documents(
        args.seed,
        args.output,
        festivals=festivals or None,
        replace=args.replace,
    )
    print("Collection summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
