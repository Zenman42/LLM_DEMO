"""Tests for Citation & Grounding Extractor."""

from app.analysis.citation_extractor import (
    extract_citations,
    get_unique_domains,
)


class TestMarkdownLinks:
    """Test Markdown hyperlink extraction."""

    def test_single_link(self):
        text = "See [Tinkoff](https://www.tinkoff.ru/business/) for details."
        cits = extract_citations(text)
        assert len(cits) == 1
        assert cits[0].url == "https://www.tinkoff.ru/business/"
        assert cits[0].domain == "tinkoff.ru"
        assert cits[0].anchor_text == "Tinkoff"

    def test_multiple_links(self):
        text = "Compare [Tinkoff](https://tinkoff.ru) and [Sber](https://sberbank.ru/products)."
        cits = extract_citations(text)
        assert len(cits) == 2
        domains = {c.domain for c in cits}
        assert "tinkoff.ru" in domains
        assert "sberbank.ru" in domains

    def test_www_stripped_from_domain(self):
        text = "[Link](https://www.example.com/page)"
        cits = extract_citations(text)
        assert cits[0].domain == "example.com"


class TestBareURLs:
    """Test bare URL extraction."""

    def test_bare_url(self):
        text = "Visit https://tinkoff.ru/business/ for more info."
        cits = extract_citations(text)
        assert len(cits) == 1
        assert "tinkoff.ru" in cits[0].url

    def test_url_with_trailing_punctuation(self):
        text = "Check https://example.com/page. It's great!"
        cits = extract_citations(text)
        assert len(cits) == 1
        # Trailing period should be stripped
        assert not cits[0].url.endswith(".")

    def test_no_duplicate_from_markdown_link(self):
        text = "See [link](https://example.com) or visit https://example.com."
        cits = extract_citations(text)
        # Should be deduplicated
        assert len(cits) == 1


class TestFootnotes:
    """Test footnote-style citations."""

    def test_footnote_with_definition(self):
        text = "Тинькофф [1] — лучший банк.\n\n[1]: https://tinkoff.ru/reviews"
        cits = extract_citations(text)
        assert any(c.url == "https://tinkoff.ru/reviews" for c in cits)
        footnote_cit = [c for c in cits if c.footnote_index == 1]
        assert len(footnote_cit) == 1

    def test_multiple_footnotes(self):
        text = "Bank A [1] and Bank B [2] are popular.\n\n[1]: https://banka.com\n[2]: https://bankb.com"
        cits = extract_citations(text)
        assert len(cits) >= 2
        urls = {c.url for c in cits}
        assert "https://banka.com" in urls
        assert "https://bankb.com" in urls

    def test_caret_footnote(self):
        text = "Source [^1] confirms this.\n\n[^1]: https://source.com/article"
        cits = extract_citations(text)
        assert any(c.url == "https://source.com/article" for c in cits)


class TestNativeCitations:
    """Test vendor-provided native citations (e.g. Perplexity)."""

    def test_native_urls(self):
        text = "Some response text."
        native = ["https://native1.com/page", "https://native2.com/article"]
        cits = extract_citations(text, native_urls=native)
        assert len(cits) == 2
        assert all(c.is_native for c in cits)

    def test_native_and_inline_combined(self):
        text = "See [link](https://inline.com/page) for details."
        native = ["https://native.com/source"]
        cits = extract_citations(text, native_urls=native)
        assert len(cits) == 2
        native_cits = [c for c in cits if c.is_native]
        inline_cits = [c for c in cits if not c.is_native]
        assert len(native_cits) == 1
        assert len(inline_cits) == 1

    def test_native_deduplication(self):
        text = "Visit https://example.com for info."
        native = ["https://example.com"]
        cits = extract_citations(text, native_urls=native)
        # Native takes precedence, bare URL should not duplicate
        assert len(cits) == 1
        assert cits[0].is_native is True


class TestGetUniqueDomains:
    """Test domain deduplication helper."""

    def test_unique_domains(self):
        from app.analysis.types import CitationInfo

        cits = [
            CitationInfo(url="https://a.com/1", domain="a.com"),
            CitationInfo(url="https://a.com/2", domain="a.com"),
            CitationInfo(url="https://b.com/1", domain="b.com"),
        ]
        domains = get_unique_domains(cits)
        assert domains == ["a.com", "b.com"]

    def test_empty_citations(self):
        assert get_unique_domains([]) == []
