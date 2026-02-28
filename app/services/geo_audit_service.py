"""GEO Audit Service — fetch & parse site pages for GEO readiness signals.

Pure-function architecture (no class wrapper) matching bi_service.py pattern.
Fetch layer uses httpx.AsyncClient; parse layer is pure (no I/O).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import defusedxml.ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from app.schemas.geo_audit import (
    AnswerFirstPageRow,
    BotAccessRow,
    ContentAuditSection,
    CrawlAccessSection,
    EeatPageRow,
    MetaRobotsRow,
    PageFreshnessRow,
    SchemaAuditSection,
    SchemaTypeFound,
    SiteAuditResponse,
)

logger = logging.getLogger(__name__)

# Bots we check in robots.txt
_AI_BOTS = [
    "Googlebot",
    "Bingbot",
    "GPTBot",
    "ChatGPT-User",
    "ClaudeBot",
    "PerplexityBot",
    "YandexBot",
    "Applebot-Extended",
    "GoogleOther",
]

# Schema.org types recommended for GEO
_RECOMMENDED_SCHEMA_TYPES = {
    "Organization",
    "Product",
    "FAQPage",
    "HowTo",
    "Article",
    "Review",
    "BreadcrumbList",
    "WebSite",
    "LocalBusiness",
}

# Answer-First summary patterns (case-insensitive)
_SUMMARY_PATTERNS = re.compile(
    r"(tl;?\s*dr|key\s+takeaway|in\s+short|in\s+summary|quick\s+answer|short\s+answer|"
    r"коротко|кратко|итого|ключевой\s+вывод|резюме)",
    re.IGNORECASE,
)

_FETCH_TIMEOUT = 10.0  # seconds per request
_SEMAPHORE_LIMIT = 5   # max concurrent fetches


# ── Fetch layer ──────────────────────────────────────────────────


async def fetch_robots_txt(client: httpx.AsyncClient, domain: str) -> str | None:
    """Fetch robots.txt for a domain. Returns text or None on failure."""
    url = f"https://{domain}/robots.txt"
    try:
        resp = await client.get(url, timeout=_FETCH_TIMEOUT, follow_redirects=True)
        if resp.status_code == 200 and "text" in resp.headers.get("content-type", ""):
            return resp.text
    except (httpx.HTTPError, Exception) as exc:
        logger.warning("Failed to fetch robots.txt for %s: %s", domain, exc)
    return None


async def fetch_page(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, dict[str, str]]:
    """Fetch a page HTML + response headers. Returns (html, headers) or (None, {}) on failure."""
    async with semaphore:
        try:
            resp = await client.get(url, timeout=_FETCH_TIMEOUT, follow_redirects=True)
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "")
                if "html" in ct or "text" in ct:
                    return resp.text, dict(resp.headers)
        except (httpx.HTTPError, Exception) as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
    return None, {}


def _url_belongs_to_domain(url: str, domain: str) -> bool:
    """Check that a URL belongs to the expected domain (SSRF protection)."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        expected = domain.lower().rstrip(".")
        return host == expected or host.endswith("." + expected)
    except Exception:
        return False


async def discover_urls_from_sitemap(
    client: httpx.AsyncClient,
    domain: str,
    limit: int = 10,
) -> tuple[list[str], str | None]:
    """Try to parse sitemap.xml and return up to `limit` page URLs.

    Returns (urls, sitemap_url). Falls back to just the homepage.
    All discovered URLs are validated against the expected domain to prevent SSRF.
    """
    sitemap_url = f"https://{domain}/sitemap.xml"
    try:
        resp = await client.get(sitemap_url, timeout=_FETCH_TIMEOUT, follow_redirects=True)
        if resp.status_code == 200:
            root = ET.fromstring(resp.text)
            # Handle namespace
            ns = ""
            if root.tag.startswith("{"):
                ns = root.tag.split("}")[0] + "}"

            urls: list[str] = []
            # Check for sitemap index
            for sm in root.findall(f"{ns}sitemap"):
                loc = sm.find(f"{ns}loc")
                if loc is not None and loc.text:
                    child_url = loc.text.strip()
                    # SSRF protection: only fetch child sitemaps from the same domain
                    if not _url_belongs_to_domain(child_url, domain):
                        logger.warning(
                            "Skipping child sitemap from foreign domain: %s (expected %s)",
                            child_url, domain,
                        )
                        continue
                    # Fetch child sitemap
                    try:
                        child_resp = await client.get(
                            child_url, timeout=_FETCH_TIMEOUT, follow_redirects=True,
                        )
                        if child_resp.status_code == 200:
                            child_root = ET.fromstring(child_resp.text)
                            child_ns = ""
                            if child_root.tag.startswith("{"):
                                child_ns = child_root.tag.split("}")[0] + "}"
                            for u in child_root.findall(f"{child_ns}url"):
                                loc_el = u.find(f"{child_ns}loc")
                                if loc_el is not None and loc_el.text:
                                    page_url = loc_el.text.strip()
                                    if _url_belongs_to_domain(page_url, domain):
                                        urls.append(page_url)
                                if len(urls) >= limit:
                                    break
                    except Exception:
                        pass
                    if len(urls) >= limit:
                        break

            # Direct urlset
            for u in root.findall(f"{ns}url"):
                loc = u.find(f"{ns}loc")
                if loc is not None and loc.text:
                    page_url = loc.text.strip()
                    if _url_belongs_to_domain(page_url, domain):
                        urls.append(page_url)
                if len(urls) >= limit:
                    break

            if urls:
                return urls[:limit], sitemap_url
    except Exception as exc:
        logger.debug("Sitemap parse failed for %s: %s", domain, exc)

    # Fallback: just homepage
    return [f"https://{domain}/"], None


# ── Parse layer (pure functions, no I/O) ─────────────────────────


def parse_robots_for_bots(robots_txt: str | None) -> list[BotAccessRow]:
    """Parse robots.txt and return access info for AI-relevant bots.

    Strategy: split into user-agent blocks first, then evaluate each bot
    against matching blocks. A bot-specific block takes priority over wildcard.
    """
    if not robots_txt:
        return [BotAccessRow(bot=b, allowed=None) for b in _AI_BOTS]

    # Step 1: Parse into blocks — each block = (agents[], directives[])
    blocks: list[tuple[list[str], list[tuple[str, str]]]] = []
    current_agents: list[str] = []
    current_directives: list[tuple[str, str]] = []  # (type, value)

    for raw_line in robots_txt.splitlines():
        line = raw_line.split("#")[0].strip()
        if not line:
            # Empty line: finalize current block if we have both agents and directives
            if current_agents and current_directives:
                blocks.append((current_agents[:], current_directives[:]))
            elif current_agents and not current_directives:
                pass  # agents without directives, wait for directives
            else:
                current_agents = []
                current_directives = []
            continue

        lower = line.lower()
        if lower.startswith("user-agent:"):
            agent = line.split(":", 1)[1].strip()
            if current_directives:
                # We had directives already → finalize previous block, start new
                if current_agents:
                    blocks.append((current_agents[:], current_directives[:]))
                current_agents = []
                current_directives = []
            current_agents.append(agent)
        elif lower.startswith("disallow:"):
            path = line.split(":", 1)[1].strip()
            current_directives.append(("disallow", path))
        elif lower.startswith("allow:"):
            path = line.split(":", 1)[1].strip()
            current_directives.append(("allow", path))
        elif lower.startswith("crawl-delay:"):
            val = line.split(":", 1)[1].strip()
            current_directives.append(("crawl-delay", val))
        # sitemap/host directives don't affect blocks

    # Finalize last block
    if current_agents and current_directives:
        blocks.append((current_agents, current_directives))

    # Step 2: For each bot, find the best matching block
    results: dict[str, BotAccessRow] = {}
    for bot in _AI_BOTS:
        bot_lower = bot.lower()
        # Prefer bot-specific block, fallback to wildcard
        matching_block = None
        wildcard_block = None
        for agents, directives in blocks:
            for agent in agents:
                if agent.lower() == bot_lower:
                    matching_block = directives
                    break
                if agent == "*":
                    wildcard_block = directives
            if matching_block:
                break

        block = matching_block or wildcard_block
        if not block:
            results[bot] = BotAccessRow(bot=bot, allowed=None)
            continue

        # Evaluate directives
        allowed = True  # default: allowed unless Disallow: /
        crawl_delay = None
        for dtype, val in block:
            if dtype == "disallow" and val == "/":
                allowed = False
            elif dtype == "allow" and val == "/":
                allowed = True
            elif dtype == "crawl-delay":
                try:
                    crawl_delay = int(val)
                except ValueError:
                    pass

        results[bot] = BotAccessRow(bot=bot, allowed=allowed, crawl_delay=crawl_delay)

    return [results[b] for b in _AI_BOTS]


def extract_meta_robots(html: str, headers: dict[str, str], url: str) -> MetaRobotsRow:
    """Extract meta robots tag and X-Robots-Tag header."""
    soup = BeautifulSoup(html, "lxml")
    meta_tag = soup.find("meta", attrs={"name": re.compile(r"^robots$", re.I)})
    meta_content = meta_tag.get("content", "") if meta_tag else None

    x_robots = headers.get("x-robots-tag") or headers.get("X-Robots-Tag")

    return MetaRobotsRow(url=url, meta_robots=meta_content or None, x_robots_tag=x_robots)


def extract_jsonld_types(html: str, url: str) -> list[SchemaTypeFound]:
    """Find all JSON-LD blocks and extract @type + property counts."""
    soup = BeautifulSoup(html, "lxml")
    results: list[SchemaTypeFound] = []

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue

            # Handle @graph
            if "@graph" in item:
                graph = item["@graph"]
                if isinstance(graph, list):
                    items.extend(graph)
                continue

            type_name = item.get("@type")
            if not type_name:
                continue
            if isinstance(type_name, list):
                type_name = type_name[0] if type_name else "Unknown"

            # Count top-level properties (excluding @type, @context, @id)
            props = [k for k in item.keys() if not k.startswith("@")]
            results.append(SchemaTypeFound(
                type_name=str(type_name),
                page_url=url,
                properties_count=len(props),
            ))

    return results


def parse_answer_first(html: str, url: str) -> AnswerFirstPageRow:
    """Score a page for Answer-First content pattern."""
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Find main content area, fallback to body
    main = soup.find("main") or soup.find("article") or soup.find("body")
    first_p = main.find("p") if main else None
    first_p_text = first_p.get_text(strip=True) if first_p else ""
    first_p_length = len(first_p_text)

    # Check for summary/TL;DR patterns in full page text
    page_text = soup.get_text(" ", strip=True)
    has_summary = bool(_SUMMARY_PATTERNS.search(page_text[:3000]))

    # Composite score: short first paragraph (direct answer) + summary pattern
    score = 0.0
    if first_p_length > 0:
        # Ideal first paragraph: 50-300 chars (concise answer)
        if 50 <= first_p_length <= 300:
            score += 60.0
        elif first_p_length < 50:
            score += 30.0
        elif first_p_length <= 500:
            score += 40.0
        else:
            score += 20.0
    if has_summary:
        score += 40.0

    return AnswerFirstPageRow(
        url=url,
        title=title[:200],
        first_p_length=first_p_length,
        has_summary_pattern=has_summary,
        score=min(score, 100.0),
    )


def parse_eeat(html: str, url: str) -> EeatPageRow:
    """Check for E-E-A-T signals on a page."""
    soup = BeautifulSoup(html, "lxml")

    # Author detection
    has_author = False
    schema_author = False

    # rel="author"
    if soup.find("a", attrs={"rel": "author"}):
        has_author = True
    # Common author class/itemprop patterns
    if soup.find(attrs={"itemprop": "author"}) or soup.find(class_=re.compile(r"author", re.I)):
        has_author = True
    # JSON-LD author
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict) and "author" in item:
                    schema_author = True
                    has_author = True
        except (json.JSONDecodeError, TypeError):
            pass

    # Date published
    has_date = bool(
        soup.find("time", attrs={"datetime": True})
        or soup.find(attrs={"itemprop": re.compile(r"datePublished|dateModified", re.I)})
        or soup.find("meta", attrs={"property": re.compile(r"article:published_time|article:modified_time")})
    )

    # References / citations (outbound links to external domains)
    links = soup.find_all("a", href=True)
    external_links = 0
    for a in links:
        href = a.get("href", "")
        if href.startswith("http") and not any(
            d in href for d in [url.split("/")[2]] if "/" in url and len(url.split("/")) > 2
        ):
            external_links += 1
    has_references = external_links >= 3

    # About page link
    has_about = bool(soup.find("a", href=re.compile(r"/about|/о-компании|/о-нас", re.I)))

    return EeatPageRow(
        url=url,
        has_author=has_author,
        has_date_published=has_date,
        has_references=has_references,
        has_about_page_link=has_about,
        schema_author=schema_author,
    )


def parse_freshness(html: str, headers: dict[str, str], url: str) -> PageFreshnessRow:
    """Extract date signals and compute freshness."""
    # Last-Modified header
    last_modified = headers.get("last-modified") or headers.get("Last-Modified")

    # JSON-LD dateModified
    json_ld_modified = None
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    dm = item.get("dateModified") or item.get("datePublished")
                    if dm:
                        json_ld_modified = str(dm)
                        break
        except (json.JSONDecodeError, TypeError):
            pass
        if json_ld_modified:
            break

    # meta article:modified_time
    meta_modified = None
    meta_tag = soup.find("meta", attrs={"property": "article:modified_time"})
    if meta_tag:
        meta_modified = meta_tag.get("content")
    if not meta_modified:
        meta_tag = soup.find("meta", attrs={"property": "article:published_time"})
        if meta_tag:
            meta_modified = meta_tag.get("content")

    # Compute freshness_days from the most recent date
    freshness_days = _compute_freshness_days(last_modified, json_ld_modified, meta_modified)

    return PageFreshnessRow(
        url=url,
        last_modified_header=last_modified,
        json_ld_date_modified=json_ld_modified,
        meta_article_modified=meta_modified,
        freshness_days=freshness_days,
    )


def _compute_freshness_days(*date_strings: str | None) -> int | None:
    """Parse multiple date strings and return days since the most recent one."""
    now = datetime.now(timezone.utc)
    most_recent = None

    for ds in date_strings:
        if not ds:
            continue
        parsed = _try_parse_date(ds)
        if parsed and (most_recent is None or parsed > most_recent):
            most_recent = parsed

    if most_recent is None:
        return None

    delta = now - most_recent
    return max(0, delta.days)


def _try_parse_date(s: str) -> datetime | None:
    """Try common date formats."""
    formats = [
        "%a, %d %b %Y %H:%M:%S %Z",     # HTTP Last-Modified
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",           # ISO 8601
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ── Orchestration ────────────────────────────────────────────────


async def run_site_audit(
    client: httpx.AsyncClient,
    domain: str,
    page_urls: list[str],
    project_id: int,
) -> SiteAuditResponse:
    """Run full GEO site audit: fetch robots + pages, parse everything, assemble response."""

    sem = asyncio.Semaphore(_SEMAPHORE_LIMIT)

    # 1. Fetch robots.txt + discover sitemap info
    robots_text = await fetch_robots_txt(client, domain)
    robots_bots = parse_robots_for_bots(robots_text)

    # Check sitemap
    sitemap_url = f"https://{domain}/sitemap.xml"
    sitemap_found = False
    try:
        resp = await client.head(sitemap_url, timeout=_FETCH_TIMEOUT, follow_redirects=True)
        sitemap_found = resp.status_code == 200
    except Exception:
        pass

    # Also check robots.txt for Sitemap directive
    if robots_text and not sitemap_found:
        for line in robots_text.splitlines():
            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[1].strip()
                sitemap_found = True
                break

    # 2. Fetch all pages concurrently
    tasks = [fetch_page(client, url, sem) for url in page_urls]
    page_results = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. Parse each page
    schema_types: list[SchemaTypeFound] = []
    answer_first_rows: list[AnswerFirstPageRow] = []
    eeat_rows: list[EeatPageRow] = []
    freshness_rows: list[PageFreshnessRow] = []
    meta_robots_rows: list[MetaRobotsRow] = []
    pages_scanned = 0

    for i, result in enumerate(page_results):
        url = page_urls[i]
        if isinstance(result, Exception):
            logger.warning("Page fetch exception for %s: %s", url, result)
            continue

        html, headers = result
        if not html:
            continue

        pages_scanned += 1

        # Meta robots
        meta_robots_rows.append(extract_meta_robots(html, headers, url))

        # Schema.org
        schema_types.extend(extract_jsonld_types(html, url))

        # Answer-First
        answer_first_rows.append(parse_answer_first(html, url))

        # E-E-A-T
        eeat_rows.append(parse_eeat(html, url))

        # Freshness
        freshness_rows.append(parse_freshness(html, headers, url))

    # 4. Compute recommended missing schema types
    found_types = {st.type_name for st in schema_types}
    recommended_missing = sorted(_RECOMMENDED_SCHEMA_TYPES - found_types)

    # 5. Assemble response
    return SiteAuditResponse(
        project_id=project_id,
        domain=domain,
        pages_scanned=pages_scanned,
        crawl_access=CrawlAccessSection(
            robots_bots=robots_bots,
            meta_robots=meta_robots_rows,
            sitemap_found=sitemap_found,
            sitemap_url=sitemap_url if sitemap_found else None,
        ),
        schema_audit=SchemaAuditSection(
            types_found=schema_types,
            recommended_missing=recommended_missing,
        ),
        content_audit=ContentAuditSection(
            answer_first=answer_first_rows,
            eeat=eeat_rows,
            freshness=freshness_rows,
        ),
    )
