"""Tests for the Analysis Pipeline orchestrator."""

import pytest

from app.analysis.pipeline import analyze_batch, analyze_response
from app.analysis.types import (
    SanitizationFlag,
    StructureType,
)
from app.gateway.types import GatewayResponse, GatewayVendor, RequestStatus


def _make_response(
    text: str = "",
    vendor: GatewayVendor = GatewayVendor.CHATGPT,
    cited_urls: list[str] | None = None,
    status: RequestStatus = RequestStatus.SUCCESS,
) -> GatewayResponse:
    """Helper to create a test GatewayResponse."""
    return GatewayResponse(
        request_id="test-001",
        vendor=vendor,
        model_version="gpt-4o-mini",
        status=status,
        raw_response_text=text,
        cited_urls=cited_urls or [],
        session_id="session-001",
        tenant_id="tenant-001",
        project_id=1,
        llm_query_id=10,
        prompt_id="prompt-001",
        latency_ms=1200,
        cost_usd=0.003,
        total_tokens=500,
    )


class TestFullPipeline:
    """Test the complete analysis pipeline end-to-end.

    Pipeline uses heuristic fallback when no OPENAI_API_KEY is set (test env).
    """

    @pytest.mark.asyncio
    async def test_numbered_list_with_brand_at_top(self):
        text = (
            "Вот топ-5 банков для малого бизнеса:\n\n"
            "1. Тинькофф — быстрое открытие счёта, удобное приложение\n"
            "2. Сбербанк — широкая сеть отделений\n"
            "3. Альфа-Банк — хорошее мобильное приложение\n"
            "4. Модульбанк — лучший для ИП\n"
            "5. Точка — отличная техподдержка"
        )
        resp = _make_response(text=text)

        result = await analyze_response(
            response=resp,
            target_brand="Тинькофф",
            target_aliases=["Т-Банк", "T-Bank"],
            competitors={
                "Сбербанк": ["Сбер"],
                "Альфа-Банк": ["Альфа"],
            },
            intent="comparative",
            persona="beginner",
        )

        # Identity passthrough
        assert result.request_id == "test-001"
        assert result.vendor == "chatgpt"
        assert result.prompt_id == "prompt-001"
        assert result.intent == "comparative"
        assert result.latency_ms == 1200

        # Sanitization
        assert result.sanitization.flag == SanitizationFlag.CLEAN

        # Entity resolution
        assert result.target_brand.is_mentioned is True
        assert result.target_brand.name == "Тинькофф"

        # Structure
        assert result.structure.structure_type == StructureType.NUMBERED_LIST
        assert result.structure.total_items == 5

        # Position
        assert result.target_brand.position_rank == 1
        assert result.target_brand.position_weight == 1.0

        # Scoring
        assert result.response_visibility_score > 0
        assert result.share_of_model_local > 0

        # Competitors
        sber = [c for c in result.competitors if c.name == "Сбербанк"]
        assert len(sber) == 1
        assert sber[0].is_mentioned is True
        assert sber[0].position_rank == 2

    @pytest.mark.asyncio
    async def test_brand_not_mentioned(self):
        text = "Сбербанк и Альфа-Банк — крупнейшие банки России."
        resp = _make_response(text=text)

        result = await analyze_response(
            response=resp,
            target_brand="Тинькофф",
            competitors={"Сбербанк": [], "Альфа-Банк": []},
        )

        assert result.target_brand.is_mentioned is False
        assert result.response_visibility_score == 0.0
        assert result.share_of_model_local == 0.0

    @pytest.mark.asyncio
    async def test_deepseek_think_stripped(self):
        text = "<think>Let me analyze the banking options...</think>Тинькофф предлагает лучшие условия для бизнеса."
        resp = _make_response(text=text, vendor=GatewayVendor.DEEPSEEK)

        result = await analyze_response(
            response=resp,
            target_brand="Тинькофф",
        )

        assert result.sanitization.flag == SanitizationFlag.DEEPSEEK_THINK_STRIPPED
        assert result.target_brand.is_mentioned is True
        assert result.sanitization.think_content != ""

    @pytest.mark.asyncio
    async def test_censored_response(self):
        text = "[CENSORED_BY_VENDOR]"
        resp = _make_response(text=text, vendor=GatewayVendor.GEMINI)

        result = await analyze_response(
            response=resp,
            target_brand="Тинькофф",
        )

        assert result.sanitization.flag == SanitizationFlag.CENSORED
        assert result.target_brand.is_mentioned is False
        assert result.response_visibility_score == 0.0

    @pytest.mark.asyncio
    async def test_empty_response(self):
        resp = _make_response(text="")
        result = await analyze_response(response=resp, target_brand="Тинькофф")
        assert result.sanitization.flag == SanitizationFlag.EMPTY_RESPONSE

    @pytest.mark.asyncio
    async def test_vendor_refusal(self):
        text = "Я ИИ и не могу давать финансовых рекомендаций."
        resp = _make_response(text=text, vendor=GatewayVendor.YANDEXGPT)

        result = await analyze_response(
            response=resp,
            target_brand="Тинькофф",
        )

        assert result.sanitization.flag == SanitizationFlag.VENDOR_REFUSAL

    @pytest.mark.asyncio
    async def test_citations_extracted(self):
        text = (
            "Тинькофф — лучший банк. "
            "Подробнее: [официальный сайт](https://tinkoff.ru/business/)\n\n"
            "Источник [1]\n\n"
            "[1]: https://example.com/review"
        )
        resp = _make_response(text=text)

        result = await analyze_response(
            response=resp,
            target_brand="Тинькофф",
        )

        assert len(result.citations) >= 2
        domains = {c.domain for c in result.citations}
        assert "tinkoff.ru" in domains

    @pytest.mark.asyncio
    async def test_native_citations_from_perplexity(self):
        text = "Тинькофф — хороший банк для бизнеса."
        native = ["https://native1.com/article", "https://native2.com/review"]
        resp = _make_response(
            text=text,
            vendor=GatewayVendor.PERPLEXITY,
            cited_urls=native,
        )

        result = await analyze_response(
            response=resp,
            target_brand="Тинькофф",
        )

        native_cits = [c for c in result.citations if c.is_native]
        assert len(native_cits) == 2

    @pytest.mark.asyncio
    async def test_narrative_response(self):
        text = (
            "Тинькофф — один из ведущих цифровых банков в России. "
            "Он предлагает широкий спектр услуг для бизнеса, включая "
            "бесплатное открытие счёта и удобное мобильное приложение. "
            "По сравнению с конкурентами, Сбербанк предлагает более "
            "широкую сеть отделений."
        )
        resp = _make_response(text=text)

        result = await analyze_response(
            response=resp,
            target_brand="Тинькофф",
            competitors={"Сбербанк": []},
        )

        assert result.structure.structure_type == StructureType.NARRATIVE
        assert result.target_brand.is_mentioned is True
        assert result.target_brand.position_weight > 0

    @pytest.mark.asyncio
    async def test_bulleted_list(self):
        text = "Лучшие банки для бизнеса:\n- Тинькофф\n- Сбербанк\n- Альфа-Банк"
        resp = _make_response(text=text)

        result = await analyze_response(
            response=resp,
            target_brand="Тинькофф",
            competitors={"Сбербанк": [], "Альфа-Банк": []},
        )

        assert result.structure.structure_type == StructureType.BULLETED_LIST
        assert result.target_brand.position_weight == 0.8

    @pytest.mark.asyncio
    async def test_to_dict_complete(self):
        text = "1. Тинькофф — лучший\n2. Сбербанк — второй"
        resp = _make_response(text=text)

        result = await analyze_response(
            response=resp,
            target_brand="Тинькофф",
            competitors={"Сбербанк": []},
            intent="comparative",
            persona="cto",
        )

        d = result.to_dict()
        assert "request_id" in d
        assert "extracted_metrics" in d
        assert "calculated_scores" in d
        assert d["prompt_metadata"]["intent"] == "comparative"
        assert d["prompt_metadata"]["persona"] == "cto"


class TestBatchAnalysis:
    """Test batch analysis of multiple responses."""

    @pytest.mark.asyncio
    async def test_batch_analysis(self):
        responses = [
            _make_response(text="1. Тинькофф\n2. Сбербанк"),
            _make_response(text="Сбербанк — крупнейший банк"),
            _make_response(text=""),
        ]

        results = await analyze_batch(
            responses=responses,
            target_brand="Тинькофф",
            competitors={"Сбербанк": []},
        )

        assert len(results) == 3
        assert results[0].target_brand.is_mentioned is True
        assert results[1].target_brand.is_mentioned is False
        assert results[2].sanitization.flag == SanitizationFlag.EMPTY_RESPONSE

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        results = await analyze_batch(responses=[], target_brand="Тинькофф")
        assert results == []


class TestEnglishContent:
    """Test pipeline with English content."""

    @pytest.mark.asyncio
    async def test_english_numbered_list(self):
        text = (
            "Top 5 payment providers:\n"
            "1. Stripe - best for developers\n"
            "2. PayPal - most popular worldwide\n"
            "3. Square - best for physical stores\n"
            "4. Adyen - enterprise solution\n"
            "5. Braintree - good PayPal integration"
        )
        resp = _make_response(text=text)

        result = await analyze_response(
            response=resp,
            target_brand="Stripe",
            competitors={"PayPal": ["Paypal"], "Square": []},
        )

        assert result.target_brand.is_mentioned is True
        assert result.target_brand.position_rank == 1
        assert result.structure.structure_type == StructureType.NUMBERED_LIST
        assert result.response_visibility_score > 0
