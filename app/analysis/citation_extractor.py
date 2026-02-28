"""Citation & Grounding Extractor — Pipeline Step 5.

Extracts citations and source references from LLM responses:
  - Footnote references: [1], [^2], [source]
  - Inline hyperlinks: [text](url)
  - Bare URLs: https://example.com
  - Native citations from vendor API (e.g. Perplexity citations)
  - Domain extraction from URLs
"""

from __future__ import annotations

import re
import logging
from urllib.parse import urlparse

from app.analysis.types import CitationInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URL / link extraction patterns
# ---------------------------------------------------------------------------

# Markdown-style links: [anchor text](url)
_MD_LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\((https?://[^\s\)]+)\)",
)

# Bare URLs
_BARE_URL_PATTERN = re.compile(
    r"(?<!\()(https?://[^\s\)\]\"'>]+)",
)

# Footnote markers: [1], [^2], [source 1]
_FOOTNOTE_PATTERN = re.compile(
    r"\[\^?(\d+)\]",
)

# Footnote definitions at end of text: [1]: https://...
_FOOTNOTE_DEF_PATTERN = re.compile(
    r"^\s*\[\^?(\d+)\]:?\s+(https?://\S+)",
    re.MULTILINE,
)


def _extract_domain(url: str) -> str:
    """Extract domain from URL, stripping www. prefix."""
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        if domain.startswith("www."):
            domain = domain[4:]
        return domain.lower()
    except Exception:
        return ""


def extract_citations(
    text: str,
    native_urls: list[str] | None = None,
) -> list[CitationInfo]:
    """Extract all citations from the response text.

    Args:
        text: Cleaned response text.
        native_urls: URLs provided by the vendor API (e.g. Perplexity citations).

    Returns:
        List of CitationInfo objects, deduplicated by URL.
    """
    citations: list[CitationInfo] = []
    seen_urls: set[str] = set()

    # 1. Native citations from vendor API
    for url in native_urls or []:
        url = url.strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            citations.append(
                CitationInfo(
                    url=url,
                    domain=_extract_domain(url),
                    is_native=True,
                )
            )

    # 2. Footnote definitions: [1]: https://...
    footnote_defs: dict[int, str] = {}
    for match in _FOOTNOTE_DEF_PATTERN.finditer(text):
        idx = int(match.group(1))
        url = match.group(2).strip()
        footnote_defs[idx] = url

    # 3. Markdown links: [text](url)
    for match in _MD_LINK_PATTERN.finditer(text):
        anchor = match.group(1).strip()
        url = match.group(2).strip()
        if url not in seen_urls:
            seen_urls.add(url)
            citations.append(
                CitationInfo(
                    url=url,
                    domain=_extract_domain(url),
                    anchor_text=anchor,
                )
            )

    # 4. Footnote markers with definitions: [1] → look up URL
    for match in _FOOTNOTE_PATTERN.finditer(text):
        idx = int(match.group(1))
        if idx in footnote_defs:
            url = footnote_defs[idx]
            if url not in seen_urls:
                seen_urls.add(url)
                citations.append(
                    CitationInfo(
                        url=url,
                        domain=_extract_domain(url),
                        footnote_index=idx,
                    )
                )

    # 5. Bare URLs (not already captured by markdown links)
    for match in _BARE_URL_PATTERN.finditer(text):
        url = match.group(1).strip().rstrip(".,;:")
        if url not in seen_urls:
            seen_urls.add(url)
            citations.append(
                CitationInfo(
                    url=url,
                    domain=_extract_domain(url),
                )
            )

    return citations


def get_unique_domains(citations: list[CitationInfo]) -> list[str]:
    """Get a deduplicated list of domains from citations."""
    seen: set[str] = set()
    domains: list[str] = []
    for c in citations:
        if c.domain and c.domain not in seen:
            seen.add(c.domain)
            domains.append(c.domain)
    return domains
