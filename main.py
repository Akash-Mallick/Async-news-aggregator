"""
Async News Aggregator CLI
Fetches RSS feeds concurrently, filters by keyword, saves a summary report.
"""

import asyncio
import argparse
import functools
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator, Iterator
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import aiohttp

# ── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Custom Exception ─────────────────────────────────────────────────────────

class FetchError(Exception):
    """Raised when an RSS feed cannot be fetched or parsed."""

    def __init__(self, url: str, message: str):
        self.url = url
        super().__init__(f"[{url}] {message}")


# ── Decorator: timing ─────────────────────────────────────────────────────────

def timed(func):
    """Log how long an async coroutine takes to complete."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log.info("%-45s  completed in %.2fs", func.__name__, elapsed)
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            log.warning("%-45s  failed after %.2fs", func.__name__, elapsed)
            raise
    return wrapper


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class FeedItem:
    title: str
    link: str
    published: str
    summary: str
    source: str
    fetched_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        short = self.title[:60] + "…" if len(self.title) > 60 else self.title
        return f"FeedItem({self.source!r}, {short!r}, {self.published!r})"

    def __lt__(self, other: "FeedItem") -> bool:
        # Sort lexicographically by published date string (ISO-ish formats sort correctly)
        return self.published < other.published

    def to_text(self) -> str:
        lines = [
            f"  Title   : {self.title}",
            f"  Source  : {self.source}",
            f"  Date    : {self.published}",
            f"  Link    : {self.link}",
        ]
        if self.summary:
            clean = " ".join(self.summary.split())[:300]
            lines.append(f"  Summary : {clean}")
        return "\n".join(lines)


# ── Feed Fetcher ──────────────────────────────────────────────────────────────

class FeedFetcher:
    """Validates a feed URL and fetches its XML content asynchronously."""

    def __init__(self, url: str):
        self._url = url

    @property
    def url(self) -> str:
        return self._url

    @url.setter
    def url(self, value: str):
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {value!r}")
        self._url = value

    @property
    def is_valid(self) -> bool:
        try:
            p = urlparse(self._url)
            return p.scheme in ("http", "https") and bool(p.netloc)
        except Exception:
            return False

    @timed
    async def fetch(self, session: aiohttp.ClientSession, timeout: int = 10) -> str:
        """Fetch the raw RSS XML, raising FetchError on any failure."""
        if not self.is_valid:
            raise FetchError(self._url, "Invalid URL")
        try:
            async with session.get(
                self._url,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={"User-Agent": "AsyncNewsAggregator/1.0"},
            ) as resp:
                if resp.status != 200:
                    raise FetchError(self._url, f"HTTP {resp.status}")
                return await resp.text()
        except aiohttp.ClientError as exc:
            raise FetchError(self._url, "Network error") from exc
        except asyncio.TimeoutError as exc:
            raise FetchError(self._url, f"Timed out after {timeout}s") from exc


# ── Async fetch coroutine ─────────────────────────────────────────────────────

async def fetch_feed(url: str, session: aiohttp.ClientSession) -> tuple[str, str]:
    """Coroutine: returns (url, raw_xml). Propagates FetchError."""
    fetcher = FeedFetcher(url)
    xml = await fetcher.fetch(session)
    return url, xml


# ── Parsers / Generators ──────────────────────────────────────────────────────

def _text(element, tag: str) -> str:
    """Safely extract text from an XML child element."""
    child = element.find(tag)
    return (child.text or "").strip() if child is not None else ""


def parse_items(url: str, xml_text: str) -> Generator[FeedItem, None, None]:
    """Generator that lazily yields FeedItem objects from raw RSS XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise FetchError(url, "XML parse error") from exc

    # Support both RSS <item> and Atom <entry>
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    channel_title = ""

    # RSS 2.0
    channel = root.find("channel")
    if channel is not None:
        channel_title = _text(channel, "title") or url
        for item in channel.findall("item"):
            yield FeedItem(
                title=_text(item, "title") or "(no title)",
                link=_text(item, "link"),
                published=_text(item, "pubDate"),
                summary=_text(item, "description"),
                source=channel_title,
            )
        return

    # Atom 1.0
    feed_title_el = root.find("atom:title", ns) or root.find("title")
    channel_title = (feed_title_el.text or "").strip() if feed_title_el is not None else url
    for entry in root.findall("atom:entry", ns) or root.findall("entry"):
        link_el = entry.find("atom:link", ns) or entry.find("link")
        link = ""
        if link_el is not None:
            link = link_el.get("href", "") or (link_el.text or "")
        summary_el = (
            entry.find("atom:summary", ns)
            or entry.find("atom:content", ns)
            or entry.find("summary")
            or entry.find("content")
        )
        published_el = (
            entry.find("atom:published", ns)
            or entry.find("atom:updated", ns)
            or entry.find("published")
            or entry.find("updated")
        )
        title_el = entry.find("atom:title", ns) or entry.find("title")
        yield FeedItem(
            title=(title_el.text or "").strip() if title_el is not None else "(no title)",
            link=link,
            published=(published_el.text or "").strip() if published_el is not None else "",
            summary=(summary_el.text or "").strip() if summary_el is not None else "",
            source=channel_title,
        )


def filter_by_keyword(items: Iterator[FeedItem], kw: str) -> Generator[FeedItem, None, None]:
    """Generator expression: yield only items whose title/summary contain kw (case-insensitive)."""
    kw_lower = kw.lower()
    return (
        item for item in items
        if kw_lower in item.title.lower() or kw_lower in item.summary.lower()
    )


# ── Context manager: report writer ───────────────────────────────────────────

@contextmanager
def report_writer(path: Path):
    """Context manager that opens a report file safely and yields a write callable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            yield fh
        tmp.replace(path)   # atomic rename — only committed if no exception
        log.info("Report saved → %s", path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ── Main orchestration ────────────────────────────────────────────────────────

DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://feeds.reuters.com/reuters/topNews",
    "https://hnrss.org/frontpage",           # Hacker News
    "https://www.theguardian.com/world/rss", # The Guardian
]


async def run(feeds: list[str], keyword: str | None, output: Path, limit: int):
    """Core async pipeline."""
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_feed(url, session) for url in feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items: list[FeedItem] = []
    errors: list[str] = []

    for url, result in zip(feeds, results):
        if isinstance(result, Exception):
            errors.append(f"  FAILED  {url}\n          {result}")
            log.error("Feed failed: %s — %s", url, result)
            continue
        _, xml_text = result
        try:
            items = list(parse_items(url, xml_text))
            log.info("Parsed %d items from %s", len(items), url)
            all_items.extend(items)
        except FetchError as exc:
            errors.append(f"  PARSE ERROR  {url}\n               {exc}")

    # Apply keyword filter (generator expression)
    if keyword:
        all_items = list(filter_by_keyword(iter(all_items), keyword))
        log.info("After keyword filter %r: %d items", keyword, len(all_items))

    # Sort newest-first (reverse lexicographic on published)
    all_items.sort(reverse=True)

    # Truncate
    displayed = all_items[:limit]

    # Write report
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with report_writer(output) as fh:
        fh.write("=" * 70 + "\n")
        fh.write("  ASYNC NEWS AGGREGATOR REPORT\n")
        fh.write(f"  Generated : {now}\n")
        fh.write(f"  Feeds     : {len(feeds)}\n")
        fh.write(f"  Keyword   : {keyword or '(none)'}\n")
        fh.write(f"  Items     : {len(displayed)} / {len(all_items)} total\n")
        fh.write("=" * 70 + "\n\n")

        if errors:
            fh.write("── Fetch Errors ──────────────────────────────────────────────────────\n")
            fh.write("\n".join(errors) + "\n\n")

        for i, item in enumerate(displayed, 1):
            fh.write(f"[{i:03d}] ────────────────────────────────────────────────────────\n")
            fh.write(item.to_text() + "\n\n")

        if not displayed:
            fh.write("  No items matched your query.\n")

    print(f"\n✓  {len(displayed)} items written to {output}")
    if errors:
        print(f"⚠  {len(errors)} feed(s) failed — see report for details.")


def main():
    parser = argparse.ArgumentParser(
        prog="news-aggregator",
        description="Fetch RSS feeds concurrently and save a filtered report.",
    )
    parser.add_argument(
        "-f", "--feeds",
        nargs="+",
        metavar="URL",
        help="RSS feed URLs to fetch (default: 5 built-in feeds)",
    )
    parser.add_argument(
        "-k", "--keyword",
        metavar="WORD",
        help="Filter items to those containing this keyword (case-insensitive)",
    )
    parser.add_argument(
        "-o", "--output",
        default="report.txt",
        metavar="FILE",
        help="Output report file path (default: report.txt)",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=50,
        metavar="N",
        help="Maximum number of items to include in the report (default: 50)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    feeds = args.feeds or DEFAULT_FEEDS
    output = Path(args.output)

    print(f"Fetching {len(feeds)} feed(s)…")
    asyncio.run(run(feeds, args.keyword, output, args.limit))


if __name__ == "__main__":
    main()