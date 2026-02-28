"""Tests for LLM-based Entity Extractor."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.analysis.llm_entity_extractor import (
    _brands_in_prompt,
    _names_match,
    extract_entities_with_llm,
    match_entities,
    transliterate_to_cyrillic,
    transliterate_to_latin,
)
from app.analysis.types import MentionType


# ---------------------------------------------------------------------------
# Transliteration tests
# ---------------------------------------------------------------------------


class TestTransliteration:
    def test_cyrillic_to_latin_fonbet(self):
        assert transliterate_to_latin("Фонбет") == "fonbet"

    def test_cyrillic_to_latin_melbet(self):
        assert transliterate_to_latin("Мелбет") == "melbet"

    def test_cyrillic_to_latin_yandex(self):
        result = transliterate_to_latin("Яндекс")
        assert result == "yandeks"

    def test_cyrillic_to_latin_mixed_case(self):
        assert transliterate_to_latin("ФОНБЕТ") == "fonbet"

    def test_latin_to_cyrillic_fonbet(self):
        # Note: 'e' → 'э' in our simple table, but matching still works
        # because we compare both transliteration directions
        result = transliterate_to_cyrillic("fonbet")
        assert "фонб" in result  # Core part matches

    def test_latin_to_cyrillic_melbet(self):
        result = transliterate_to_cyrillic("melbet")
        assert "м" in result and "л" in result  # Core consonants match

    def test_latin_passthrough(self):
        # Latin text stays as-is when transliterating to Latin
        assert transliterate_to_latin("Fonbet") == "fonbet"

    def test_empty_string(self):
        assert transliterate_to_latin("") == ""
        assert transliterate_to_cyrillic("") == ""

    def test_multi_char_sequences(self):
        # "sh" → "ш", "ch" → "ч", etc.
        result = transliterate_to_cyrillic("shchuka")
        assert "щ" in result  # shch → щ


# ---------------------------------------------------------------------------
# Name matching tests
# ---------------------------------------------------------------------------


class TestNamesMatch:
    def test_exact_match(self):
        assert _names_match("Fonbet", "Fonbet") is True

    def test_case_insensitive(self):
        assert _names_match("Fonbet", "fonbet") is True
        assert _names_match("MELBET", "melbet") is True

    def test_cyrillic_vs_latin_fonbet(self):
        assert _names_match("Фонбет", "Fonbet") is True

    def test_cyrillic_vs_latin_melbet(self):
        assert _names_match("Мелбет", "Melbet") is True

    def test_no_match(self):
        assert _names_match("Фонбет", "Parimatch") is False

    def test_containment(self):
        assert _names_match("1xBet", "1xBet.com") is True

    def test_short_names_no_containment(self):
        # Short names (< 3 chars) should not use containment
        assert _names_match("BK", "BookMaker") is False

    def test_transliteration_both_directions(self):
        # Latin → Cyrillic matching
        assert _names_match("Fonbet", "Фонбет") is True

    def test_different_brands(self):
        assert _names_match("Winline", "BetBoom") is False

    def test_pari_vs_parimatch(self):
        # "PARI" is contained in "Parimatch" and both >= 3 chars,
        # so containment check matches. This is acceptable since these
        # are related brands and LLM extraction produces exact entity names.
        assert _names_match("PARI", "Parimatch") is True

    def test_google_maps_vs_google_karty(self):
        """'Google Maps' should match 'Google Карты' via synonym normalization."""
        assert _names_match("Google Maps", "Google Карты") is True

    def test_yandex_karty_with_dot(self):
        """'Яндекс Карты' (no dot) matches 'Яндекс.Карты' (with dot)."""
        # Dot in brand name is normalized to space → exact match
        assert _names_match("Яндекс Карты", "Яндекс.Карты") is True

    def test_google_maps_vs_maps_me_no_false_positive(self):
        """'Google Maps' should NOT match 'MAPS.ME' — different products."""
        assert _names_match("Google Maps", "MAPS.ME") is False

    def test_containment_still_works(self):
        """Containment matching still works for substrings (≥70% ratio)."""
        # "карты" (5 chars) in "google карты" (12 chars) → ratio 0.42 < 0.70 → no match
        assert _names_match("Карты", "Google Карты") is False
        # Corporate prefix stripping: "Группа ПИК" → "ПИК" matches "ПИК"
        assert _names_match("Группа ПИК", "ПИК") is True

    def test_corporate_chain_prefix(self):
        """Multi-word corporate prefixes like 'Группа компаний' are stripped."""
        assert _names_match("Группа компаний ПИК", "ПИК") is True
        assert _names_match("Группа Компаний ЛСР", "ЛСР") is True
        assert _names_match("Управляющая компания Самолёт", "Самолёт") is True
        assert _names_match("ПАО Группа компаний ПИК", "ПИК") is True
        # Reverse order should also work
        assert _names_match("ПИК", "Группа компаний ПИК") is True

    def test_vk_video_vs_vk_muzyka_no_match(self):
        """Different VK products should not match."""
        assert _names_match("ВК Видео", "ВК Музыка") is False

    def test_yandex_karty_synonym_normalize(self):
        """'Яндекс.Карты' matches 'Яндекс Maps' via synonym normalization."""
        # "Яндекс.Карты" → normalize → "яндекс карты"
        # "Яндекс Maps" → normalize → "яндекс maps" → synonym → "яндекс карты"
        assert _names_match("Яндекс.Карты", "Яндекс Maps") is True


# ---------------------------------------------------------------------------
# match_entities tests
# ---------------------------------------------------------------------------


class TestMatchEntities:
    def test_brand_detected_latin(self):
        """Target brand in Cyrillic matches Latin entity."""
        extracted = ["Fonbet", "Parimatch", "1xBet"]
        text = "Fonbet is one of the top betting platforms. Parimatch and 1xBet are also popular."

        target, comps, discovered = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Parimatch": []},
            full_text=text,
        )

        assert target.is_mentioned is True
        assert target.name == "Фонбет"
        assert "Fonbet" in target.aliases_matched

    def test_competitor_detected_latin(self):
        """Competitor in Cyrillic matches Latin entity."""
        extracted = ["Fonbet", "Melbet", "BetBoom"]
        text = "Fonbet, Melbet and BetBoom are popular platforms."

        target, comps, discovered = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": [], "BetBoom": []},
            full_text=text,
        )

        melbet = next(c for c in comps if c.name == "Мелбет")
        assert melbet.is_mentioned is True
        assert "Melbet" in melbet.aliases_matched

        betboom = next(c for c in comps if c.name == "BetBoom")
        assert betboom.is_mentioned is True

    def test_discovered_entities(self):
        """Entities not in brand/competitors go to discovered list."""
        extracted = ["Fonbet", "Parimatch", "Bet365"]
        text = "Fonbet, Parimatch, and Bet365 are popular."

        target, comps, discovered = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={},
            full_text=text,
        )

        assert "Parimatch" in discovered
        assert "Bet365" in discovered

    def test_no_entities(self):
        """Empty entity list returns no matches."""
        target, comps, discovered = match_entities(
            extracted=[],
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": []},
            full_text="Some text without brands.",
        )

        assert target.is_mentioned is False
        assert all(not c.is_mentioned for c in comps)
        assert discovered == []

    def test_context_extraction(self):
        """Matched entities have context extracted from text."""
        text = "In Russia, Fonbet is the most popular betting platform with millions of users."
        extracted = ["Fonbet"]

        target, _, _ = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={},
            full_text=text,
        )

        assert target.is_mentioned is True
        assert target.mention_context != ""
        assert "Fonbet" in target.mention_context
        assert target.char_offset >= 0

    def test_brand_alias_match(self):
        """Target brand matches via alias."""
        extracted = ["fon.bet"]
        text = "Check out fon.bet for the latest odds."

        target, _, _ = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=["fon.bet"],
            competitors={},
            full_text=text,
        )

        assert target.is_mentioned is True

    def test_mention_type_is_direct(self):
        """All LLM-extracted entities get DIRECT mention type initially."""
        extracted = ["Fonbet"]
        text = "Fonbet offers great odds."

        target, _, _ = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={},
            full_text=text,
        )

        assert target.mention_type == MentionType.DIRECT

    def test_unmentioned_competitors_not_mentioned(self):
        """Competitors not in extracted list should have is_mentioned=False."""
        extracted = ["Fonbet"]
        text = "Fonbet is great."

        target, comps, _ = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": [], "BetBoom": []},
            full_text=text,
        )

        assert all(not c.is_mentioned for c in comps)


# ---------------------------------------------------------------------------
# Prompt filtering tests — brands in the prompt should NOT count as mentions
# ---------------------------------------------------------------------------


class TestBrandsInPrompt:
    def test_target_brand_in_prompt_cyrillic(self):
        """Target brand in Cyrillic prompt is detected."""
        result = _brands_in_prompt(
            "что лучше Фонбет или Мелбет?",
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": []},
        )
        assert "Фонбет" in result
        assert "Мелбет" in result

    def test_target_brand_not_in_prompt(self):
        """Brand NOT in prompt is not in the set."""
        result = _brands_in_prompt(
            "лучшие букмекеры 2024",
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": []},
        )
        assert result == set()

    def test_alias_in_prompt(self):
        """Brand alias in prompt is detected."""
        result = _brands_in_prompt(
            "Is fon.bet reliable?",
            target_brand="Фонбет",
            target_aliases=["fon.bet"],
            competitors={},
        )
        assert "Фонбет" in result

    def test_transliteration_in_prompt(self):
        """Cyrillic brand detected via transliteration in Latin prompt."""
        result = _brands_in_prompt(
            "Is Fonbet the best bookmaker?",
            target_brand="Фонбет",
            target_aliases=[],
            competitors={},
        )
        assert "Фонбет" in result

    def test_empty_prompt(self):
        """Empty prompt returns empty set."""
        result = _brands_in_prompt(
            "",
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": []},
        )
        assert result == set()


class TestMatchEntitiesPromptFiltering:
    def test_brand_in_prompt_not_counted(self):
        """Brand mentioned in prompt should NOT count as mention in response."""
        extracted = ["Fonbet", "Melbet"]
        text = "Fonbet is a popular bookmaker. Melbet is also well-known."

        target, comps, discovered = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": []},
            full_text=text,
            prompt_text="что лучше Фонбет или Мелбет?",
        )

        # Both brands are in the prompt, so neither should be counted
        assert target.is_mentioned is False
        melbet = next(c for c in comps if c.name == "Мелбет")
        assert melbet.is_mentioned is False

    def test_brand_not_in_prompt_still_counted(self):
        """Brand NOT in prompt should still be counted as mention."""
        extracted = ["Fonbet", "Melbet", "BetBoom"]
        text = "Fonbet, Melbet, and BetBoom are popular bookmakers."

        target, comps, discovered = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": [], "BetBoom": []},
            full_text=text,
            prompt_text="что лучше Фонбет или Мелбет?",
        )

        # Фонбет and Мелбет are in prompt → not counted
        assert target.is_mentioned is False
        melbet = next(c for c in comps if c.name == "Мелбет")
        assert melbet.is_mentioned is False

        # BetBoom is NOT in prompt → should be counted
        betboom = next(c for c in comps if c.name == "BetBoom")
        assert betboom.is_mentioned is True

    def test_no_prompt_all_counted(self):
        """Without prompt_text, all entities are counted normally."""
        extracted = ["Fonbet", "Melbet"]
        text = "Fonbet and Melbet are popular bookmakers."

        target, comps, _ = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": []},
            full_text=text,
            prompt_text="",
        )

        assert target.is_mentioned is True
        melbet = next(c for c in comps if c.name == "Мелбет")
        assert melbet.is_mentioned is True

    def test_generic_prompt_all_counted(self):
        """Generic prompt without brands → all entities counted."""
        extracted = ["Fonbet", "Melbet"]
        text = "Fonbet and Melbet are the top bookmakers in Russia."

        target, comps, _ = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": []},
            full_text=text,
            prompt_text="лучшие букмекеры России",
        )

        assert target.is_mentioned is True
        melbet = next(c for c in comps if c.name == "Мелбет")
        assert melbet.is_mentioned is True

    def test_only_target_in_prompt(self):
        """Only target brand in prompt — competitors still counted."""
        extracted = ["Fonbet", "Melbet"]
        text = "Fonbet is good. Melbet is also good."

        target, comps, _ = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={"Мелбет": []},
            full_text=text,
            prompt_text="расскажи про Фонбет",
        )

        assert target.is_mentioned is False  # in prompt
        melbet = next(c for c in comps if c.name == "Мелбет")
        assert melbet.is_mentioned is True  # NOT in prompt

    def test_discovered_entities_still_work_with_prompt(self):
        """Entities not matching any known brand still go to discovered."""
        extracted = ["Fonbet", "Bet365"]
        text = "Fonbet and Bet365 are popular."

        target, comps, discovered = match_entities(
            extracted=extracted,
            target_brand="Фонбет",
            target_aliases=[],
            competitors={},
            full_text=text,
            prompt_text="расскажи про Фонбет",
        )

        assert target.is_mentioned is False  # in prompt
        assert "Bet365" in discovered  # unknown brand still discovered


# ---------------------------------------------------------------------------
# extract_entities_with_llm tests (mocked HTTP)
# ---------------------------------------------------------------------------


def _make_mock_httpx(response_data):
    """Helper to create properly mocked httpx.AsyncClient for tests."""
    mock_client_cls = MagicMock()
    mock_client = AsyncMock()
    mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_resp = MagicMock()  # MagicMock since json() and raise_for_status() are sync
    mock_resp.json.return_value = response_data
    mock_client.post.return_value = mock_resp

    return mock_client_cls, mock_client, mock_resp


class TestExtractEntitiesWithLlm:
    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        """LLM returns valid JSON with entities (enriched format)."""
        api_response = {
            "choices": [{"message": {"content": json.dumps({"entities": ["Fonbet", "Parimatch", "1xBet"]})}}]
        }

        mock_cls, _, _ = _make_mock_httpx(api_response)
        with patch("app.analysis.llm_entity_extractor.httpx.AsyncClient", mock_cls):
            result = await extract_entities_with_llm("Some text about betting", "test-key")

        # Now returns list[dict] with enriched format; bare strings get wrapped
        assert len(result) == 3
        assert [e["name"] for e in result] == ["Fonbet", "Parimatch", "1xBet"]
        assert all(isinstance(e, dict) and "name" in e for e in result)

    @pytest.mark.asyncio
    async def test_extraction_with_markdown_wrapper(self):
        """LLM wraps JSON in markdown code block."""
        raw_content = '```json\n{"entities": ["Fonbet", "Melbet"]}\n```'
        api_response = {"choices": [{"message": {"content": raw_content}}]}

        mock_cls, _, _ = _make_mock_httpx(api_response)
        with patch("app.analysis.llm_entity_extractor.httpx.AsyncClient", mock_cls):
            result = await extract_entities_with_llm("text", "test-key")

        assert len(result) == 2
        assert [e["name"] for e in result] == ["Fonbet", "Melbet"]

    @pytest.mark.asyncio
    async def test_extraction_failure_returns_empty(self):
        """On HTTP error, return empty list."""
        import httpx

        mock_cls, mock_client, _ = _make_mock_httpx({})
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=MagicMock(status_code=429)
        )
        mock_client.post.return_value = mock_resp

        with patch("app.analysis.llm_entity_extractor.httpx.AsyncClient", mock_cls):
            result = await extract_entities_with_llm("text", "test-key")

        assert result == []

    @pytest.mark.asyncio
    async def test_extraction_invalid_json_returns_empty(self):
        """On invalid JSON from LLM, return empty list."""
        api_response = {"choices": [{"message": {"content": "not json at all"}}]}

        mock_cls, _, _ = _make_mock_httpx(api_response)
        with patch("app.analysis.llm_entity_extractor.httpx.AsyncClient", mock_cls):
            result = await extract_entities_with_llm("text", "test-key")

        assert result == []

    @pytest.mark.asyncio
    async def test_extraction_empty_entities(self):
        """LLM returns empty entities list."""
        api_response = {"choices": [{"message": {"content": json.dumps({"entities": []})}}]}

        mock_cls, _, _ = _make_mock_httpx(api_response)
        with patch("app.analysis.llm_entity_extractor.httpx.AsyncClient", mock_cls):
            result = await extract_entities_with_llm("text", "test-key")

        assert result == []
