"""Offline tests for the official-guide collector."""

from __future__ import annotations

import csv
from pathlib import Path

from src.collect import (
    COLUMNS,
    Collector,
    evaluate_robots,
    infer_category,
    is_access_challenge,
    make_document_id,
    normalize_text,
    parse_documents,
)

FAQ_HTML = """
<html><body>
<h1>2026 Support & FAQ</h1>
<div class="c-accordion c-section js-accordion-section">
  <h2>Passes, Order &amp; Shipping</h2>
  <details class="c-accordion__block" id="order-1">
    <div class="c-accordion__question">When is my order being shipped?</div>
    <div class="c-accordion__answer"><p>Orders ship about two weeks before the festival weekend.</p></div>
  </details>
</div>
<div class="c-accordion c-section js-accordion-section">
  <h2>Travel</h2>
  <details class="c-accordion__block" id="park-1">
    <div class="c-accordion__question">What's the deal with parking?</div>
    <div class="c-accordion__answer"><p>Day parking opens at 11am Friday through Sunday. Cars left after 2am are towed. There is no overnight parking without a camping pass.</p></div>
  </details>
  <details class="c-accordion__block" id="short-1">
    <div class="c-accordion__question">Where is the venue?</div>
    <div class="c-accordion__answer"><p>See the map.</p></div>
  </details>
</div>
<nav><a>Buy passes</a></nav>
</body></html>
"""

ARTICLE_HTML = """
<html><body>
<header><nav>Skip to main content</nav></header>
<article>
  <h1 class="article-title">What should I pack?</h1>
  <time datetime="2026-04-15T15:41:00Z"></time>
  <div class="article-body">
    <p>Bring layers because the desert cools down after sunset and daytime sun is strong.</p>
    <h2>Rain and cold</h2>
    <p>Pack a light jacket and a warm layer for the evening. Temperatures can drop after dark, so a hoodie is worth the space in your bag.</p>
    <h2>Sun</h2>
    <p>Bring non-aerosol sunscreen, a hat, and sunglasses. Reapply sunscreen through the afternoon because shade is limited in open fields.</p>
  </div>
</article>
<footer>Related articles and newsletter signup</footer>
</body></html>
"""

CARD_HTML = """
<html><body>
<h1>2026 Camping &amp; Parking</h1>
<div class="ticket">
  <span class="ticket-info" name="Primitive Car Camping" type="Car Camping"></span>
  <div class="description">
    <p>Primitive car camping gives one vehicle a campsite of about 20 by 20 feet. Every person still needs a festival ticket, and the camping pass is for the vehicle rather than each person.</p>
  </div>
</div>
<div class="ticket">
  <span class="ticket-info" name="Primitive Car Camping" type="Car Camping"></span>
  <div class="description">
    <p>Primitive car camping gives one vehicle a campsite of about 20 by 20 feet. Every person still needs a festival ticket, and the camping pass is for the vehicle rather than each person.</p>
  </div>
</div>
<footer><h3>Partners</h3><p>Sponsor names listed in the footer should never be collected.</p></footer>
</body></html>
"""

CHALLENGE_HTML = "<html><title>Just a moment...</title><p>Enable JavaScript and cookies to continue.</p></html>"


def test_robots_missing_file_allows_public_guide():
    allowed, reason = evaluate_robots(404, "not found", "https://www.coachella.com/faq/")
    assert allowed
    assert "404" in reason


def test_robots_disallow_is_respected():
    body = "User-agent: *\nDisallow: /private\n"
    allowed, reason = evaluate_robots(200, body, "https://example.com/private")
    assert not allowed
    assert "disallows" in reason
    allowed, _ = evaluate_robots(200, body, "https://example.com/faq/")
    assert allowed


def test_challenge_page_is_not_treated_as_a_guide():
    assert is_access_challenge(403, CHALLENGE_HTML)
    documents = parse_documents(
        CHALLENGE_HTML,
        "https://support.bonnaroo.com/hc/en-us/articles/1",
        "Bonnaroo",
        "2026",
        "weather",
        "official_guide",
    )
    assert documents == []


def test_faq_parser_keeps_parking_answer_and_drops_shipping_section():
    documents = parse_documents(
        FAQ_HTML,
        "https://www.coachella.com/faq/",
        "Coachella",
        "",
        "general",
        "official_faq",
    )
    assert len(documents) == 1
    document = documents[0]
    assert document["title"] == "Travel: What's the deal with parking?"
    assert document["category"] == "transportation"
    assert document["festival_year"] == "2026"
    assert document["source_url"].endswith("#park-1")
    assert "towed" in document["content"]
    assert "Buy passes" not in document["content"]
    assert document["published_date"] == ""


def test_help_article_splits_on_headings_and_keeps_update_date():
    documents = parse_documents(
        ARTICLE_HTML,
        "https://support.bonnaroo.com/hc/en-us/articles/4402795571220",
        "Bonnaroo",
        "2026",
        "weather",
        "official_guide",
    )
    titles = [document["title"] for document in documents]
    assert "Rain and cold" in titles
    assert "Sun" in titles
    assert all(document["published_date"] == "2026-04-15" for document in documents)
    assert all("newsletter" not in document["content"] for document in documents)


def test_duplicate_lodging_cards_collapse_to_one_row():
    documents = parse_documents(
        CARD_HTML,
        "https://www.bonnaroo.com/accommodations",
        "Bonnaroo",
        "2026",
        "lodging",
        "official_guide",
    )
    assert len(documents) == 1
    assert documents[0]["category"] == "lodging"
    assert documents[0]["title"] == "Primitive Car Camping"
    assert "Sponsor" not in documents[0]["content"]


def test_document_ids_are_stable_and_distinct_for_different_urls():
    first = make_document_id("Coachella", "2026", "Parking", "https://www.coachella.com/faq/#1")
    again = make_document_id("Coachella", "2026", "Parking", "https://www.coachella.com/faq/#1")
    other = make_document_id("Coachella", "2026", "Parking", "https://www.coachella.com/getting-here/")
    assert first == again
    assert first != other


def test_airport_style_screening_is_a_rule_and_policies_stay_rules():
    assert infer_category("Entry protocols", "airport-style screening for safety") == "rules"
    assert infer_category("Festival Policies") == "rules"
    assert infer_category("Getting Around") == "transportation"
    assert infer_category("How big is a medium backpack?") is None
    assert infer_category("What should I pack?") == "weather"


def test_rv_camping_question_is_lodging_not_parking():
    assert infer_category("Where can I park my RV or motorhome?") == "lodging"
    assert infer_category("What's the deal with parking?") == "transportation"


def test_existing_rows_are_preserved(tmp_path: Path):
    output = tmp_path / "documents.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow({
            "document_id": "already-there",
            "festival_name": "Coachella",
            "festival_year": "2026",
            "category": "rules",
            "title": "Existing rule",
            "content": "This row was collected earlier and must stay in the file.",
            "source_type": "official_faq",
            "source_url": "https://www.coachella.com/festival-info/#existing",
            "published_date": "",
        })

    def fetcher(url: str) -> tuple[int, str]:
        return 200, FAQ_HTML

    def robots(url: str) -> tuple[int, str]:
        return 404, ""

    stats = Collector(fetcher=fetcher, robots_reader=robots, delay_seconds=0).collect(
        [{
            "festival_name": "Coachella",
            "festival_year": "2026",
            "category": "general",
            "source_type": "official_faq",
            "url": "https://www.coachella.com/faq/",
        }],
        output,
    )
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["document_id"] == "already-there"
    assert rows[0]["content"].startswith("This row was collected earlier")
    assert any(row["title"] == "Travel: What's the deal with parking?" for row in rows)
    assert stats["existing_preserved"] == 1
    assert stats["documents_added"] == 1


def test_anti_bot_challenge_stops_later_requests_to_that_host(tmp_path: Path):
    requested: list[str] = []

    def fetcher(url: str) -> tuple[int, str]:
        requested.append(url)
        return 403, CHALLENGE_HTML

    def robots(url: str) -> tuple[int, str]:
        return 200, "User-agent: *\nDisallow:\n"

    Collector(fetcher=fetcher, robots_reader=robots, delay_seconds=0).collect(
        [
            {
                "festival_name": "Bonnaroo",
                "festival_year": "2026",
                "category": "weather",
                "source_type": "official_guide",
                "url": "https://support.bonnaroo.com/hc/en-us/articles/1",
            },
            {
                "festival_name": "Bonnaroo",
                "festival_year": "2026",
                "category": "rules",
                "source_type": "official_faq",
                "url": "https://support.bonnaroo.com/hc/en-us/articles/2",
            },
        ],
        tmp_path / "documents.csv",
    )
    assert requested == ["https://support.bonnaroo.com/hc/en-us/articles/1"]


def test_trailing_learn_more_is_removed_without_cutting_a_sentence():
    assert normalize_text("Shuttles run until midnight. Learn More") == "Shuttles run until midnight."
    kept = normalize_text("Learn more about parking before you leave for the festival grounds.")
    assert kept.lower().startswith("learn more about parking")


HEADING_HTML = """
<html><head><title>Hours &amp; Info – EDC Las Vegas 2027</title></head>
<body><main>
  <h2>Weather</h2>
  <p>The festival will take place rain or shine. Pack a light layer for the desert night and sunscreen for the afternoon heat.</p>
  <h2>Sponsors</h2>
  <p>Partner logos and promotional copy that should never become a searchable festival document for attendees.</p>
</main></body></html>
"""

DEFINITION_HTML = """
<html><body>
<h2>Festival FAQ</h2>
<dl class="accordion"><dt>What time does the event start and end each day?</dt>
<dd>Gates open in the afternoon and music continues into the morning. Arrive early if you want a shorter entry line.</dd></dl>
<h2>Ticketing FAQ</h2>
<dl class="accordion"><dt>Where is my order?</dt>
<dd>Orders can be checked in the ticket account. This whole ticketing section should be skipped.</dd></dl>
<dl class="accordion"><dt>Can I enter anytime?</dt>
<dd>You can enter whenever the gates are open as long as the festival is still admitting guests for that day.</dd></dl>
</body></html>
"""

FAQ_ITEM_HTML = """
<html><body>
<h3>Travel &amp; Transport</h3>
<div class="faqItem"><h4>Where can I park?</h4><p>There is no general event parking. Use the official shuttle, public transit, or a rideshare drop-off.</p></div>
<h3>Tickets &amp; Wristbands</h3>
<div class="faqItem"><h4>Can I get a refund?</h4><p>Refund questions belong to ticketing and should not be stored as attendee guide documents.</p></div>
</body></html>
"""


def test_heading_sections_keep_weather_and_drop_sponsors():
    documents = parse_documents(
        HEADING_HTML,
        "https://lasvegas.electricdaisycarnival.com/guide/hours-and-info/",
        "EDC Las Vegas",
        "2026",
        "rules",
        "official_guide",
    )
    assert [document["title"] for document in documents] == ["Weather"]
    assert documents[0]["category"] == "weather"
    assert documents[0]["festival_year"] == "2027"
    assert "Partner logos" not in documents[0]["content"]


def test_definition_accordions_skip_ticketing_section():
    documents = parse_documents(
        DEFINITION_HTML,
        "https://lasvegas.electricdaisycarnival.com/guide/faq/",
        "EDC Las Vegas",
        "2027",
        "experiences",
        "official_faq",
    )
    titles = [document["title"] for document in documents]
    assert any("event start" in title for title in titles)
    assert all("order" not in title.lower() for title in titles)


def test_faq_items_use_the_section_and_skip_ticketing():
    documents = parse_documents(
        FAQ_ITEM_HTML,
        "https://sfoutsidelands.com/faq-app",
        "Outside Lands",
        "2026",
        "general",
        "official_faq",
    )
    assert len(documents) == 1
    assert documents[0]["category"] == "transportation"
    assert documents[0]["title"] == "Travel & Transport: Where can I park?"
    assert "refund" not in documents[0]["content"].lower()


def test_pages_already_in_the_csv_are_not_downloaded_again(tmp_path: Path):
    output = tmp_path / "documents.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow({
            "document_id": "kept-row",
            "festival_name": "Coachella",
            "festival_year": "2026",
            "category": "rules",
            "title": "Existing rule",
            "content": "This collected answer must remain exactly as it was stored.",
            "source_type": "official_faq",
            "source_url": "https://www.coachella.com/faq/#park-1",
            "published_date": "",
        })
    requested: list[str] = []

    def fetcher(url: str) -> tuple[int, str]:
        requested.append(url)
        return 200, FAQ_HTML

    stats = Collector(fetcher=fetcher, robots_reader=lambda url: (404, ""), delay_seconds=0).collect(
        [{
            "festival_name": "Coachella",
            "festival_year": "2026",
            "category": "general",
            "source_type": "official_faq",
            "url": "https://www.coachella.com/faq/",
        }],
        output,
    )
    assert requested == []
    assert stats["pages_already_collected"] == 1
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{
        "document_id": "kept-row",
        "festival_name": "Coachella",
        "festival_year": "2026",
        "category": "rules",
        "title": "Existing rule",
        "content": "This collected answer must remain exactly as it was stored.",
        "source_type": "official_faq",
        "source_url": "https://www.coachella.com/faq/#park-1",
        "published_date": "",
    }]
