"""
Fetches the ECB's own monetary-policy press-conference statements
("Introductory statement [with Q&A]", "Monetary policy statement") for the
EA-MPD event dates, from the ECB website, and caches each as plain text under
data/ecb_pressconf/<YYYY-MM-DD>.txt.

WHY THIS EXISTS: the "All ECB speeches" export lacks almost all of these
(only 33/315 dates matched), and matching them inside the 390MB BIS archive
found 228 but with 3 wrong links (see data.py). The ECB publishes them as
individual pages; this pulls the official text directly.

HOW: for each year, the ECB publishes an index page
  https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/<YEAR>/html/index_include.en.html
listing <dt isoDate="YYYY-MM-DD"> entries with an href to the statement page.
We map date -> statement URL from those, then fetch only the dates we need.

BEHAVIOUR: polite (1 request/second, identifying User-Agent), resumable
(skips dates already cached), official ecb.europa.eu URLs only. Dates with no
press-conference statement (unscheduled meetings that only produced a press
release) are reported as not-found and NOT filled with anything.

Usage:  python fetch_ecb_pressconf.py
"""
from __future__ import annotations

import datetime as dt
import html
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from data import load_market_vectors

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = DATA_DIR / "ecb_pressconf"
BASE = "https://www.ecb.europa.eu"
INDEX_URL = BASE + "/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html"
UA = "Mozilla/5.0 (research; Precedent-RAG data collection; polite 1 req/s)"


def http_get(url: str, retries: int = 3) -> str:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001 - report and retry
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


_HREF = re.compile(r'href="([^"]+\.en\.html)"')


def _is_statement_url(u: str) -> bool:
    """Only genuine press-conference statement pages: NOT press releases
    (/press/pr/), podcasts, blog posts, etc."""
    return ("/press_conference/monetary-policy-statement/" in u) or ("/press/pressconf/" in u)


def index_for_year(year: int) -> dict[dt.date, str]:
    """date -> statement URL, parsing each <dt isoDate=...> entry SEPARATELY
    (an earlier version used one greedy regex across the page and, when an
    entry had no matching link, silently borrowed the NEXT entry's link)."""
    try:
        page = http_get(INDEX_URL.format(year=year))
    except RuntimeError:
        return {}
    out: dict[dt.date, str] = {}
    chunks = re.split(r'(?=<dt isoDate=")', page)
    for ch in chunks:
        m = re.match(r'<dt isoDate="(\d{4}-\d{2}-\d{2})"', ch)
        if not m:
            continue
        d = dt.date.fromisoformat(m.group(1))
        # stop at the next <dt (chunk already ends there); take statement links only
        urls = [h if h.startswith("http") else BASE + h for h in _HREF.findall(ch)]
        urls = [u for u in urls if _is_statement_url(u)]
        if urls:
            out.setdefault(d, urls[0])
    return out


_DATE_IN_TEXT = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})"
)


def printed_date(text: str):
    """First full date printed near the top of the statement page."""
    m = _DATE_IN_TEXT.search(text[:900])
    if not m:
        return None
    return dt.datetime.strptime(" ".join(m.groups()), "%d %B %Y").date()


class _TextExtractor(HTMLParser):
    """Collects visible text from the statement's main content, skipping
    scripts/styles/nav/footers."""

    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "form"}

    def __init__(self):
        super().__init__()
        self.depth_skip = 0
        self.in_main = False
        self.main_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("main",) or (tag == "div" and "ecb-pressContent" in (a.get("class") or "")):
            self.in_main = True
        if tag in self.SKIP:
            self.depth_skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth_skip:
            self.depth_skip -= 1
        if tag in ("p", "li", "h1", "h2", "h3", "br", "div"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self.depth_skip == 0 and self.in_main:
            self.parts.append(data)


def extract_text(page: str) -> str:
    p = _TextExtractor()
    p.feed(page)
    text = html.unescape("".join(p.parts))
    text = re.sub(r"[ \t\r\f\v\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if len(text) < 500:  # main-container heuristic missed; fall back to whole body
        q = _TextExtractor()
        q.in_main = True
        q.feed(page)
        text = re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t\r\f\v\xa0]+", " ", html.unescape("".join(q.parts)))).strip()
    # Cut the site footer / feedback widget / cookie banner that follows the statement.
    for marker in ("Reproduction is permitted provided", "Media contacts", "Are you happy with this page?",
                   "Our website uses cookies"):
        i = text.find(marker)
        if i > 500:
            text = text[:i].rstrip()
    # Older ECB pages contain U+FFFD where an apostrophe was mangled at the source ("today�s").
    # Repair only the unambiguous contraction/possessive cases; leave every other U+FFFD alone.
    text = re.sub(r"(?<=[A-Za-z])�(?=(?:s|t|d|m|ll|re|ve)\b)", "'", text)
    return text


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = sorted(load_market_vectors(DATA_DIR / "Dataset_EA-MPD.xlsx"))
    print(f"{len(targets)} EA-MPD event dates to look up.")

    years = sorted({d.year for d in targets})
    url_by_date: dict[dt.date, str] = {}
    for y in years:
        idx = index_for_year(y)
        url_by_date.update(idx)
        print(f"  index {y}: {len(idx)} statements listed")
        time.sleep(1)

    found, cached, missing, failed = 0, 0, [], []
    for d in targets:
        out = OUT_DIR / f"{d.isoformat()}.txt"
        if out.exists() and out.stat().st_size > 500:
            cached += 1
            continue
        url = url_by_date.get(d)
        if url is None:
            missing.append(d)
            continue
        try:
            text = extract_text(http_get(url))
        except RuntimeError as e:
            failed.append((d, str(e)))
            continue
        if len(text) < 500:
            failed.append((d, f"extracted only {len(text)} chars from {url}"))
            continue
        pd_ = printed_date(text)
        # The URL itself carries the meeting date ("...is060608..." = 2006-06-08). Accept if the printed
        # date OR the URL date equals the event date: a mis-linked entry fails both, while a typo in the
        # ECB's own page (2006-06-08's page prints "6 June 2006") only fails the printed date.
        m = re.search(r"is(\d{6})", url)
        url_date = None
        if m:
            try:
                url_date = dt.datetime.strptime(m.group(1), "%y%m%d").date()
            except ValueError:
                pass
        if pd_ != d and url_date != d:
            failed.append((d, f"REJECTED: page prints date {pd_}, URL date {url_date}, event date is {d} ({url})"))
            continue
        if pd_ != d:
            print(f"  NOTE {d}: page prints {pd_} but URL date matches -- accepted (ECB page typo)")
        out.write_text(f"SOURCE: {url}\n\n{text}\n", encoding="utf-8")
        found += 1
        time.sleep(1)

    print(f"\nfetched now: {found}   already cached: {cached}   "
          f"no statement listed on ECB site: {len(missing)}   failed: {len(failed)}")
    if missing:
        print("dates with NO press-conference statement on the ECB site (left unlinked, not faked):")
        print("  " + ", ".join(d.isoformat() for d in missing))
    for d, why in failed:
        print("  FAILED", d, why)


if __name__ == "__main__":
    sys.exit(main())
