"""Tests for the Prompt Engineering Module (Module 1).

Tests cover all 5 layers of the pipeline plus the orchestrator
and API endpoints.
"""

import pytest
from httpx import AsyncClient

from app.prompt_engine.adapters import (
    adapt_for_chatgpt,
    adapt_for_deepseek,
    adapt_for_gemini,
    adapt_for_perplexity,
    adapt_for_yandexgpt,
    adapt_prompts,
)
from app.prompt_engine.dispatcher import parameterize_prompts
from app.prompt_engine.expansion import (
    apply_conversational_variations,
    apply_semantic_pivots,
    expand_skeletons,
)
from app.prompt_engine.matrix import build_skeletons
from app.prompt_engine.normalization import (
    deduplicate_list,
    generate_brand_aliases,
    normalize_brand,
    normalize_profile,
    normalize_text,
)
from app.prompt_engine.pipeline import PromptPipeline
from app.prompt_engine.types import (
    AdaptedPrompt,
    AuditProfile,
    ExpandedPrompt,
    FinalPrompt,
    Intent,
    MeasurementType,
    Persona,
    PromptSkeleton,
    ScenarioConfig,
    TargetLLM,
)
from app.schemas.prompt_engine import AuditProfileCreate


# ==========================================================================
# Layer 1: Normalization tests
# ==========================================================================


class TestNormalization:
    def test_normalize_text_strips_whitespace(self):
        assert normalize_text("  hello  world  ") == "hello world"

    def test_normalize_text_collapses_spaces(self):
        assert normalize_text("hello   world") == "hello world"

    def test_normalize_brand_preserves_case(self):
        assert normalize_brand("Ahrefs") == "Ahrefs"
        assert normalize_brand("  SEMrush  ") == "SEMrush"

    def test_deduplicate_list_preserves_order(self):
        result = deduplicate_list(["Apple", "Banana", "apple", "Cherry", "banana"])
        assert result == ["Apple", "Banana", "Cherry"]

    def test_deduplicate_list_empty(self):
        assert deduplicate_list([]) == []
        assert deduplicate_list(["  ", ""]) == []

    def test_generate_brand_aliases_basic(self):
        aliases = generate_brand_aliases("Ahrefs")
        assert "ahrefs" in aliases

    def test_generate_brand_aliases_camel_case(self):
        aliases = generate_brand_aliases("SEMrush")
        assert any("sem" in a.lower() for a in aliases)

    def test_generate_brand_aliases_with_dot(self):
        aliases = generate_brand_aliases("Яндекс.Метрика")
        assert any("Яндекс Метрика" in a or "яндекс метрика" in a for a in aliases)

    def test_normalize_profile_basic(self):
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["SEMrush", "Moz"],
            categories=["SEO"],
            market="ru",
        )
        profile = normalize_profile(input_data)
        assert profile.brand == "Ahrefs"
        assert len(profile.competitors) == 2
        assert "SEMrush" in profile.competitors
        assert "Moz" in profile.competitors
        assert profile.market == "ru"

    def test_normalize_profile_removes_brand_from_competitors(self):
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["Ahrefs", "SEMrush", "Moz"],
            categories=["SEO"],
        )
        profile = normalize_profile(input_data)
        assert "Ahrefs" not in profile.competitors

    def test_normalize_profile_deduplicates_categories(self):
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["SEMrush"],
            categories=["SEO", "seo", "  SEO  ", "Link Analysis"],
        )
        profile = normalize_profile(input_data)
        assert len(profile.categories) == 2

    def test_normalize_profile_generates_aliases(self):
        input_data = AuditProfileCreate(
            brand="SEMrush",
            competitors=["Ahrefs"],
            categories=["SEO"],
            brand_aliases=["semrush.com"],
        )
        profile = normalize_profile(input_data)
        assert len(profile.brand_aliases) > 0
        assert "semrush.com" in profile.brand_aliases

    def test_normalize_profile_empty_categories_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AuditProfileCreate(
                brand="Test",
                competitors=["Comp"],
                categories=[],
            )


# ==========================================================================
# Layer 2: Combinatorial Matrix tests
# ==========================================================================


class TestMatrix:
    @pytest.fixture
    def profile(self):
        return AuditProfile(
            brand="Ahrefs",
            competitors=["SEMrush", "Moz"],
            scenarios=[ScenarioConfig(name="SEO-аналитика")],
            market="ru",
        )

    def test_build_skeletons_basic(self, profile):
        skeletons = build_skeletons(profile)
        assert len(skeletons) > 0

    def test_build_skeletons_all_personas_covered(self, profile):
        skeletons = build_skeletons(profile)
        personas_found = {s.persona for s in skeletons}
        assert personas_found == set(Persona)

    def test_build_skeletons_all_intents_covered(self, profile):
        skeletons = build_skeletons(profile)
        intents_found = {s.intent for s in skeletons}
        assert intents_found == set(Intent)

    def test_build_skeletons_brand_in_text(self, profile):
        skeletons = build_skeletons(profile)
        brand_skeletons = [s for s in skeletons if "Ahrefs" in s.template]
        assert len(brand_skeletons) > 0

    def test_build_skeletons_competitor_expansion(self, profile):
        """Templates with {competitor} should be expanded per competitor."""
        skeletons = build_skeletons(profile)
        # Find templates that mention competitors
        comp_skeletons = [s for s in skeletons if "SEMrush" in s.template or "Moz" in s.template]
        assert len(comp_skeletons) > 0

    def test_build_skeletons_filtered_personas(self, profile):
        skeletons = build_skeletons(profile, personas=[Persona.CTO])
        personas_found = {s.persona for s in skeletons}
        assert personas_found == {Persona.CTO}

    def test_build_skeletons_filtered_intents(self, profile):
        skeletons = build_skeletons(profile, intents=[Intent.COMPARATIVE])
        intents_found = {s.intent for s in skeletons}
        assert intents_found == {Intent.COMPARATIVE}

    def test_build_skeletons_en_market(self):
        profile = AuditProfile(
            brand="Ahrefs",
            competitors=["SEMrush"],
            scenarios=[ScenarioConfig(name="SEO analytics")],
            market="en",
        )
        skeletons = build_skeletons(profile)
        assert len(skeletons) > 0
        # Check that at least some are in English
        assert any("explain" in s.template.lower() or "what" in s.template.lower() for s in skeletons)

    def test_build_skeletons_multiple_categories(self):
        profile = AuditProfile(
            brand="Ahrefs",
            competitors=["SEMrush"],
            scenarios=[
                ScenarioConfig(name="SEO"),
                ScenarioConfig(name="Backlink Analysis"),
                ScenarioConfig(name="Site Audit"),
            ],
            market="en",
        )
        skeletons = build_skeletons(profile)
        categories_found = {s.category for s in skeletons}
        assert categories_found == {"SEO", "Backlink Analysis", "Site Audit"}


# ==========================================================================
# Layer 3: Expansion tests
# ==========================================================================


class TestExpansion:
    @pytest.fixture
    def skeletons(self):
        return [
            PromptSkeleton(
                template="Посоветуй лучший сервис для SEO",
                persona=Persona.BEGINNER,
                intent=Intent.COMPARATIVE,
                category="SEO",
                brand="Ahrefs",
                competitors=["SEMrush"],
                market="ru",
            ),
            PromptSkeleton(
                template="Где найти документацию Ahrefs для SEO",
                persona=Persona.CTO,
                intent=Intent.NAVIGATIONAL,
                category="SEO",
                brand="Ahrefs",
                competitors=["SEMrush"],
                market="ru",
            ),
        ]

    def test_semantic_pivots_generates_more(self, skeletons):
        expanded = apply_semantic_pivots(skeletons, market="ru")
        # First skeleton has "лучший" which triggers pivots
        assert len(expanded) > len(skeletons)

    def test_semantic_pivots_keeps_originals(self, skeletons):
        expanded = apply_semantic_pivots(skeletons, market="ru")
        originals = [e for e in expanded if e.pivot_axis == "original"]
        assert len(originals) == len(skeletons)

    def test_semantic_pivots_en(self):
        skeletons = [
            PromptSkeleton(
                template="What is the best SEO tool",
                persona=Persona.BEGINNER,
                intent=Intent.COMPARATIVE,
                category="SEO",
                brand="Ahrefs",
                competitors=["SEMrush"],
                market="en",
            ),
        ]
        expanded = apply_semantic_pivots(skeletons, market="en")
        assert len(expanded) > 1
        pivots = [e for e in expanded if e.pivot_axis != "original"]
        assert any("reliable" in e.text.lower() or "cheapest" in e.text.lower() for e in pivots)

    def test_conversational_variations(self, skeletons):
        expanded = apply_semantic_pivots(skeletons, market="ru")
        varied = apply_conversational_variations(expanded, market="ru")
        assert len(varied) > len(expanded)

    def test_expand_skeletons_full(self, skeletons):
        profile = AuditProfile(
            brand="Ahrefs",
            competitors=["SEMrush"],
            scenarios=[ScenarioConfig(name="SEO")],
            market="ru",
        )
        expanded = expand_skeletons(skeletons, profile)
        assert len(expanded) > len(skeletons)
        # Check all have required fields
        for e in expanded:
            assert e.text
            assert e.persona
            assert e.intent
            assert e.category

    def test_expand_without_pivots(self, skeletons):
        profile = AuditProfile(
            brand="Ahrefs",
            competitors=["SEMrush"],
            scenarios=[ScenarioConfig(name="SEO")],
            market="ru",
        )
        expanded = expand_skeletons(skeletons, profile, enable_pivots=False)
        # No pivot variants, only originals + conversational
        pivot_variants = [
            e for e in expanded if "original" not in e.pivot_axis and "conversational" not in e.pivot_axis
        ]
        assert len(pivot_variants) == 0


# ==========================================================================
# Layer 4: Adapter tests
# ==========================================================================


class TestAdapters:
    @pytest.fixture
    def prompt_ru(self):
        return ExpandedPrompt(
            text="Посоветуй лучший сервис для SEO-аналитики",
            persona=Persona.BEGINNER,
            intent=Intent.COMPARATIVE,
            category="SEO-аналитика",
            brand="Ahrefs",
            competitors=["SEMrush", "Moz"],
            market="ru",
            pivot_axis="original",
        )

    @pytest.fixture
    def prompt_en(self):
        return ExpandedPrompt(
            text="What is the best SEO analytics tool",
            persona=Persona.CTO,
            intent=Intent.COMPARATIVE,
            category="SEO analytics",
            brand="Ahrefs",
            competitors=["SEMrush"],
            market="en",
            pivot_axis="original",
        )

    def test_chatgpt_adapter_has_system_prompt(self, prompt_ru):
        adapted = adapt_for_chatgpt(prompt_ru)
        assert adapted.target_llm == TargetLLM.CHATGPT
        assert adapted.system_prompt
        assert "консультант" in adapted.system_prompt.lower() or "ИИ" in adapted.system_prompt

    def test_chatgpt_adapter_persona_specific(self, prompt_en):
        adapted = adapt_for_chatgpt(prompt_en)
        assert "analyst" in adapted.system_prompt.lower() or "technical" in adapted.system_prompt.lower()

    def test_gemini_adapter_preserves_user_prompt(self, prompt_ru):
        adapted = adapt_for_gemini(prompt_ru)
        assert adapted.target_llm == TargetLLM.GEMINI
        # Gemini adapter should preserve original user prompt text
        assert adapted.user_prompt == prompt_ru.text

    def test_gemini_adapter_factual_system(self, prompt_ru):
        adapted = adapt_for_gemini(prompt_ru)
        assert "факт" in adapted.system_prompt.lower() or "аналитическ" in adapted.system_prompt.lower()
        # Reframing instructions should be in system prompt, not user prompt
        assert "субъективн" in adapted.system_prompt.lower() or "обзор" in adapted.system_prompt.lower()

    def test_deepseek_adapter_preserves_user_prompt(self, prompt_ru):
        adapted = adapt_for_deepseek(prompt_ru)
        assert adapted.target_llm == TargetLLM.DEEPSEEK
        # DeepSeek adapter should preserve original user prompt text
        assert adapted.user_prompt == prompt_ru.text
        assert "эксперт" in adapted.system_prompt.lower() or "аналитик" in adapted.system_prompt.lower()

    def test_deepseek_adapter_en(self, prompt_en):
        adapted = adapt_for_deepseek(prompt_en)
        # Should preserve original text, not wrap it
        assert adapted.user_prompt == prompt_en.text
        assert "expert" in adapted.system_prompt.lower() or "analyst" in adapted.system_prompt.lower()

    def test_yandexgpt_adapter_preserves_user_prompt(self, prompt_ru):
        adapted = adapt_for_yandexgpt(prompt_ru)
        assert adapted.target_llm == TargetLLM.YANDEXGPT
        # YandexGPT adapter should preserve original user prompt text
        assert adapted.user_prompt == prompt_ru.text

    def test_yandexgpt_adapter_no_roleplay(self, prompt_ru):
        adapted = adapt_for_yandexgpt(prompt_ru)
        assert "Действуй как" not in adapted.system_prompt
        assert "ты —" not in adapted.system_prompt.lower()

    def test_yandexgpt_adapter_geo_in_system(self, prompt_ru):
        adapted = adapt_for_yandexgpt(prompt_ru)
        # Geo context should be in system prompt, not appended to user prompt
        assert "росси" in adapted.system_prompt.lower()

    def test_perplexity_adapter_preserves_user_prompt(self, prompt_ru):
        adapted = adapt_for_perplexity(prompt_ru)
        # Perplexity adapter should preserve original user prompt text
        assert adapted.user_prompt == prompt_ru.text
        # Temporal/freshness context should be in system prompt
        assert "актуальн" in adapted.system_prompt.lower() or "свеж" in adapted.system_prompt.lower()

    def test_adapt_prompts_multiple_providers(self, prompt_ru):
        adapted = adapt_prompts([prompt_ru], target_llms=["chatgpt", "deepseek", "yandexgpt"])
        assert len(adapted) == 3

    def test_adapt_prompts_with_perplexity(self, prompt_ru):
        adapted = adapt_prompts([prompt_ru], target_llms=["perplexity"])
        assert len(adapted) == 1
        assert "perplexity:" in adapted[0].skeleton_id


# ==========================================================================
# Layer 5: Dispatcher tests
# ==========================================================================


class TestDispatcher:
    def test_parameterize_sets_temperature(self):
        adapted = [
            _make_adapted(Intent.INFORMATIONAL),
            _make_adapted(Intent.COMPARATIVE),
        ]
        final = parameterize_prompts(adapted)
        assert final[0].temperature == 0.0  # factual
        assert final[1].temperature == 0.7  # creative

    def test_parameterize_sets_resilience(self):
        adapted = [
            _make_adapted(Intent.COMPARATIVE),
            _make_adapted(Intent.NAVIGATIONAL),
        ]
        final = parameterize_prompts(adapted)
        assert final[0].resilience_runs == 3  # important, multi-run
        assert final[1].resilience_runs == 1  # factual, single run

    def test_parameterize_unique_ids(self):
        adapted = [_make_adapted(Intent.COMPARATIVE) for _ in range(10)]
        final = parameterize_prompts(adapted)
        ids = [f.prompt_id for f in final]
        # IDs should be unique
        assert len(set(ids)) == len(ids)

    def test_parameterize_to_dict(self):
        final = parameterize_prompts([_make_adapted(Intent.COMPARATIVE)])[0]
        d = final.to_dict()
        assert "prompt_id" in d
        assert "target_llm" in d
        assert "system_prompt" in d
        assert "user_prompt" in d
        assert "temperature" in d
        assert "metadata" in d
        assert d["metadata"]["intent"] == "comparative"
        assert "measurement_type" in d["metadata"]

    def test_measurement_type_propagation(self):
        """Verify measurement_type flows through adapters → dispatcher → FinalPrompt."""
        adapted = AdaptedPrompt(
            target_llm=TargetLLM.CHATGPT,
            system_prompt="Test system prompt",
            user_prompt="Test user prompt",
            persona=Persona.BEGINNER,
            intent=Intent.COMPARATIVE,
            category="Test",
            brand="TestBrand",
            competitors=["CompA"],
            market="en",
            measurement_type="top_of_mind",
        )
        final = parameterize_prompts([adapted])[0]
        assert final.measurement_type == "top_of_mind"
        d = final.to_dict()
        assert d["metadata"]["measurement_type"] == "top_of_mind"


# ==========================================================================
# Pipeline orchestrator tests
# ==========================================================================


class TestPipeline:
    def test_pipeline_basic_ru(self):
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["SEMrush", "Moz"],
            categories=["SEO-аналитика"],
            market="ru",
            target_llms=["chatgpt"],
        )
        pipeline = PromptPipeline()
        result = pipeline.run(input_data)

        assert result.total_prompts > 0
        assert result.skeletons_count > 0
        assert result.expanded_count >= result.skeletons_count
        assert "chatgpt" in result.by_provider

    def test_pipeline_basic_en(self):
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["SEMrush"],
            categories=["SEO analytics"],
            market="en",
            target_llms=["chatgpt"],
        )
        pipeline = PromptPipeline()
        result = pipeline.run(input_data)

        assert result.total_prompts > 0

    def test_pipeline_multiple_providers(self):
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["SEMrush"],
            categories=["SEO"],
            market="ru",
            target_llms=["chatgpt", "deepseek", "yandexgpt"],
        )
        pipeline = PromptPipeline()
        result = pipeline.run(input_data)

        assert len(result.by_provider) == 3
        assert "chatgpt" in result.by_provider
        assert "deepseek" in result.by_provider
        assert "yandexgpt" in result.by_provider

    def test_pipeline_multiple_categories(self):
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["SEMrush"],
            categories=["SEO", "Backlinks", "Site Audit"],
            market="en",
            target_llms=["chatgpt"],
        )
        pipeline = PromptPipeline()
        result = pipeline.run(input_data)

        # Should have prompts for all categories
        categories = set()
        for p in result.prompts:
            categories.add(p.category)
        assert categories == {"SEO", "Backlinks", "Site Audit"}

    def test_pipeline_filtered_personas(self):
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["SEMrush"],
            categories=["SEO"],
            market="en",
            target_llms=["chatgpt"],
            personas=["cto", "skeptic"],
        )
        pipeline = PromptPipeline()
        result = pipeline.run(input_data)

        personas_found = set(result.by_persona.keys())
        assert "cto" in personas_found
        assert "skeptic" in personas_found
        assert "beginner" not in personas_found

    def test_pipeline_no_expansion(self):
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["SEMrush"],
            categories=["SEO"],
            market="en",
            target_llms=["chatgpt"],
            enable_expansion=False,
            enable_pivots=False,
        )
        pipeline = PromptPipeline()
        result = pipeline.run(input_data)

        # Without pivots, expanded == skeletons + conversational variants
        assert result.total_prompts > 0

    def test_pipeline_to_response_dict(self):
        input_data = AuditProfileCreate(
            brand="Test",
            competitors=["Comp"],
            categories=["Cat"],
            market="en",
            target_llms=["chatgpt"],
        )
        pipeline = PromptPipeline()
        result = pipeline.run(input_data)
        resp = result.to_response_dict()

        assert "total_prompts" in resp
        assert "prompts" in resp
        assert isinstance(resp["prompts"], list)
        assert len(resp["prompts"]) == resp["total_prompts"]

    def test_pipeline_to_summary_dict(self):
        input_data = AuditProfileCreate(
            brand="Test",
            competitors=["Comp"],
            categories=["Cat"],
            market="en",
            target_llms=["chatgpt"],
        )
        pipeline = PromptPipeline()
        result = pipeline.run(input_data)
        summary = result.to_summary_dict()

        assert "total_prompts" in summary
        assert "prompts" not in summary  # Summary doesn't include full prompts

    def test_pipeline_scale_5_categories_3_competitors(self):
        """Test realistic scale: 5 categories × 3 competitors."""
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["SEMrush", "Moz", "Serpstat"],
            categories=[
                "SEO-аналитика",
                "Анализ обратных ссылок",
                "Аудит сайтов",
                "Анализ конкурентов",
                "Мониторинг позиций",
            ],
            market="ru",
            target_llms=["chatgpt", "deepseek", "yandexgpt"],
            enable_pivots=True,
        )
        pipeline = PromptPipeline()
        result = pipeline.run(input_data)

        # Should generate a substantial number of prompts
        assert result.total_prompts > 100
        assert result.skeletons_count > 50

        # Personas, intents and providers should be covered
        assert len(result.by_intent) >= 2
        assert len(result.by_persona) >= 2
        assert len(result.by_provider) == 3


# ==========================================================================
# API endpoint tests
# ==========================================================================


@pytest.fixture
async def project_for_prompts(client: AsyncClient, auth_headers):
    """Create a project for prompt engine tests."""
    resp = await client.post(
        "/api/v1/projects/",
        json={
            "name": "Prompt Engine Test",
            "domain": "example.com",
            "track_llm": True,
            "brand_name": "TestBrand",
            "llm_providers": ["chatgpt"],
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    return resp.json()


@pytest.mark.asyncio
async def test_api_generate(client: AsyncClient, auth_headers):
    resp = await client.post(
        "/api/v1/prompt-engine/generate",
        json={
            "brand": "Ahrefs",
            "competitors": ["SEMrush"],
            "categories": ["SEO"],
            "market": "en",
            "target_llms": ["chatgpt"],
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_prompts"] > 0
    assert "prompts" in data
    assert len(data["prompts"]) == data["total_prompts"]


@pytest.mark.asyncio
async def test_api_preview(client: AsyncClient, auth_headers):
    resp = await client.post(
        "/api/v1/prompt-engine/preview",
        json={
            "brand": "Ahrefs",
            "competitors": ["SEMrush", "Moz"],
            "categories": ["SEO", "Backlinks"],
            "market": "ru",
            "target_llms": ["chatgpt", "deepseek"],
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_prompts"] > 0
    assert "prompts" not in data  # Preview doesn't include prompts


@pytest.mark.asyncio
async def test_api_generate_and_save(client: AsyncClient, auth_headers, project_for_prompts):
    pid = project_for_prompts["id"]
    resp = await client.post(
        f"/api/v1/prompt-engine/generate-and-save/{pid}?max_prompts=50",
        json={
            "brand": "TestBrand",
            "competitors": ["CompA"],
            "categories": ["Testing"],
            "market": "en",
            "target_llms": ["chatgpt"],
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["saved"]["created"] > 0
    assert data["project_id"] == pid

    # Verify queries were actually saved
    queries_resp = await client.get(
        f"/api/v1/projects/{pid}/llm-queries/",
        headers=auth_headers,
    )
    assert queries_resp.status_code == 200
    assert queries_resp.json()["total"] > 0


@pytest.mark.asyncio
async def test_api_generate_and_save_deduplication(client: AsyncClient, auth_headers, project_for_prompts):
    """Running generate-and-save twice should skip duplicates."""
    pid = project_for_prompts["id"]
    body = {
        "brand": "TestBrand",
        "competitors": ["CompA"],
        "categories": ["Testing"],
        "market": "en",
        "target_llms": ["chatgpt"],
        "enable_pivots": False,
    }

    resp1 = await client.post(
        f"/api/v1/prompt-engine/generate-and-save/{pid}?max_prompts=20",
        json=body,
        headers=auth_headers,
    )
    created_first = resp1.json()["saved"]["created"]

    resp2 = await client.post(
        f"/api/v1/prompt-engine/generate-and-save/{pid}?max_prompts=20",
        json=body,
        headers=auth_headers,
    )
    data2 = resp2.json()
    assert data2["saved"]["skipped"] >= created_first


@pytest.mark.asyncio
async def test_api_generate_validation_errors(client: AsyncClient, auth_headers):
    # Missing required field
    resp = await client.post(
        "/api/v1/prompt-engine/generate",
        json={
            "brand": "Test",
            # Missing competitors and categories
        },
        headers=auth_headers,
    )
    assert resp.status_code == 422


# ==========================================================================
# Generator tests (LLM-based prompt generation)
# ==========================================================================


class TestGenerator:
    """Tests for the LLM prompt generator module."""

    def test_parse_generation_output_json(self):
        from app.prompt_engine.generator import _parse_generation_output

        text = '{"queries": ["Подскажи что лучше для SEO — Ahrefs или SEMrush", "Какой инструмент выбрать новичку"]}'
        result = _parse_generation_output(text)
        assert len(result) == 2
        assert "Ahrefs" in result[0]

    def test_parse_generation_output_markdown_block(self):
        from app.prompt_engine.generator import _parse_generation_output

        text = '```json\n{"queries": ["Стоит ли платить за Ahrefs если я начинающий SEO-специалист"]}\n```'
        result = _parse_generation_output(text)
        assert len(result) == 1

    def test_parse_generation_output_array(self):
        from app.prompt_engine.generator import _parse_generation_output

        text = '["Что лучше для анализа ссылок", "Ahrefs или Moz для начинающих"]'
        result = _parse_generation_output(text)
        assert len(result) == 2

    def test_parse_generation_output_filters_short(self):
        from app.prompt_engine.generator import _parse_generation_output

        text = '{"queries": ["Ahrefs", "Подскажи что лучше для SEO — Ahrefs или SEMrush"]}'
        result = _parse_generation_output(text)
        assert len(result) == 1  # "Ahrefs" alone is too short (<10 chars)

    def test_parse_generation_output_line_fallback(self):
        from app.prompt_engine.generator import _parse_generation_output

        text = (
            "1. Подскажи что лучше для SEO — Ahrefs или SEMrush\n2. Какой инструмент выбрать новичку для анализа ссылок"
        )
        result = _parse_generation_output(text)
        assert len(result) == 2

    def test_fallback_generates_prompts(self):
        from app.prompt_engine.generator import generate_prompts_fallback

        profile = AuditProfile(
            brand="Ahrefs",
            competitors=["SEMrush"],
            scenarios=[ScenarioConfig(name="SEO")],
            market="en",
        )
        result = generate_prompts_fallback(profile)
        assert len(result) > 0
        for p in result:
            assert p.text
            assert p.brand == "Ahrefs"


# ==========================================================================
# Measurement Type tests
# ==========================================================================


class TestMeasurementTypes:
    """Tests for the measurement type axis and compatibility matrix."""

    def test_measurement_type_enum_values(self):
        assert len(MeasurementType) == 7
        assert MeasurementType.TOP_OF_MIND.value == "top_of_mind"
        assert MeasurementType.COMPARISON.value == "comparison"
        assert MeasurementType.BY_EVENT.value == "by_event"
        assert MeasurementType.BY_ATTRIBUTE.value == "by_attribute"
        assert MeasurementType.PERSONA_SCENARIO.value == "persona_scenario"
        assert MeasurementType.NEGATIVE.value == "negative"
        assert MeasurementType.ASSOCIATIVE.value == "associative"

    def test_golden_facts_on_audit_profile(self):
        profile = AuditProfile(
            brand="Fonbet",
            competitors=["1xBet"],
            scenarios=[ScenarioConfig(name="Букмекеры")],
            market="ru",
            golden_facts=["Быстрый вывод", "Лицензия ФНС"],
        )
        assert len(profile.golden_facts) == 2
        assert "Быстрый вывод" in profile.golden_facts

    def test_golden_facts_normalization(self):
        input_data = AuditProfileCreate(
            brand="Fonbet",
            competitors=["1xBet"],
            categories=["Букмекеры"],
            market="ru",
            golden_facts=["  Быстрый вывод  ", "Лицензия ФНС", "быстрый вывод"],
        )
        profile = normalize_profile(input_data)
        # Should be deduplicated (case-insensitive)
        assert len(profile.golden_facts) == 2

    def test_adapter_propagates_measurement_type(self):
        """Verify that adapters propagate measurement_type field."""
        prompt = ExpandedPrompt(
            text="Test query for SEO",
            persona=Persona.BEGINNER,
            intent=Intent.COMPARATIVE,
            category="SEO",
            brand="Ahrefs",
            competitors=["SEMrush"],
            market="en",
            measurement_type="top_of_mind",
        )
        adapted = adapt_for_chatgpt(prompt)
        assert adapted.measurement_type == "top_of_mind"

        adapted_gemini = adapt_for_gemini(prompt)
        assert adapted_gemini.measurement_type == "top_of_mind"

        adapted_deepseek = adapt_for_deepseek(prompt)
        assert adapted_deepseek.measurement_type == "top_of_mind"

        adapted_yandex = adapt_for_yandexgpt(prompt)
        assert adapted_yandex.measurement_type == "top_of_mind"

    def test_pipeline_by_measurement_type_property(self):
        """Verify PipelineResult.by_measurement_type works."""
        from app.prompt_engine.pipeline import PipelineResult

        result = PipelineResult(
            profile=AuditProfile(
                brand="Test",
                competitors=["Comp"],
                scenarios=[ScenarioConfig(name="Cat")],
                market="en",
            ),
            prompts=[
                FinalPrompt(
                    prompt_id="1",
                    target_llm=TargetLLM.CHATGPT,
                    system_prompt="sys",
                    user_prompt="usr",
                    temperature=0.0,
                    resilience_runs=1,
                    max_tokens=2048,
                    persona=Persona.BEGINNER,
                    intent=Intent.INFORMATIONAL,
                    category="Cat",
                    brand="Test",
                    competitors=["Comp"],
                    market="en",
                    measurement_type="top_of_mind",
                ),
                FinalPrompt(
                    prompt_id="2",
                    target_llm=TargetLLM.CHATGPT,
                    system_prompt="sys",
                    user_prompt="usr2",
                    temperature=0.7,
                    resilience_runs=3,
                    max_tokens=2048,
                    persona=Persona.BEGINNER,
                    intent=Intent.COMPARATIVE,
                    category="Cat",
                    brand="Test",
                    competitors=["Comp"],
                    market="en",
                    measurement_type="comparison",
                ),
                FinalPrompt(
                    prompt_id="3",
                    target_llm=TargetLLM.CHATGPT,
                    system_prompt="sys",
                    user_prompt="usr3",
                    temperature=0.0,
                    resilience_runs=1,
                    max_tokens=2048,
                    persona=Persona.BEGINNER,
                    intent=Intent.INFORMATIONAL,
                    category="Cat",
                    brand="Test",
                    competitors=["Comp"],
                    market="en",
                    measurement_type="top_of_mind",
                ),
            ],
        )
        by_mt = result.by_measurement_type
        assert by_mt["top_of_mind"] == 2
        assert by_mt["comparison"] == 1

    def test_summary_dict_includes_measurement_type(self):
        from app.prompt_engine.pipeline import PipelineResult

        result = PipelineResult(
            profile=AuditProfile(
                brand="Test",
                competitors=["Comp"],
                scenarios=[ScenarioConfig(name="Cat")],
                market="en",
            ),
            prompts=[
                FinalPrompt(
                    prompt_id="1",
                    target_llm=TargetLLM.CHATGPT,
                    system_prompt="sys",
                    user_prompt="usr",
                    temperature=0.0,
                    resilience_runs=1,
                    max_tokens=2048,
                    persona=Persona.BEGINNER,
                    intent=Intent.INFORMATIONAL,
                    category="Cat",
                    brand="Test",
                    competitors=["Comp"],
                    market="en",
                    measurement_type="top_of_mind",
                ),
            ],
        )
        summary = result.to_summary_dict()
        assert "by_measurement_type" in summary
        assert summary["by_measurement_type"]["top_of_mind"] == 1

        resp = result.to_response_dict()
        assert "by_measurement_type" in resp


# ==========================================================================
# Brand Description tests
# ==========================================================================


class TestBrandDescription:
    """Tests for the brand_description field across the pipeline."""

    def test_brand_description_on_audit_profile(self):
        """brand_description field exists on AuditProfile dataclass."""
        profile = AuditProfile(
            brand="Металлпрофиль",
            competitors=["Grand Line"],
            scenarios=[ScenarioConfig(name="Кровельные материалы")],
            market="ru",
            brand_description="Производитель кровельных и фасадных систем из тонколистовой стали.",
        )
        assert profile.brand_description == "Производитель кровельных и фасадных систем из тонколистовой стали."

    def test_brand_description_default_empty(self):
        """brand_description defaults to empty string."""
        profile = AuditProfile(
            brand="Test",
            competitors=["Comp"],
            scenarios=[ScenarioConfig(name="Cat")],
            market="en",
        )
        assert profile.brand_description == ""

    def test_normalize_profile_with_brand_description(self):
        """Normalization passes brand_description through."""
        input_data = AuditProfileCreate(
            brand="Металлпрофиль",
            competitors=["Grand Line"],
            categories=["Кровельные материалы"],
            market="ru",
            brand_description="  Производитель кровельных  и  фасадных систем.  ",
        )
        profile = normalize_profile(input_data)
        assert profile.brand_description == "Производитель кровельных и фасадных систем."

    def test_normalize_profile_without_brand_description(self):
        """Backward compat: normalization works without brand_description."""
        input_data = AuditProfileCreate(
            brand="Ahrefs",
            competitors=["SEMrush"],
            categories=["SEO"],
            market="en",
        )
        profile = normalize_profile(input_data)
        assert profile.brand_description == ""

    def test_pipeline_with_brand_description(self):
        """Pipeline works with brand_description in input."""
        input_data = AuditProfileCreate(
            brand="Металлпрофиль",
            competitors=["Grand Line"],
            categories=["Кровельные материалы"],
            market="ru",
            target_llms=["chatgpt"],
            brand_description="Производитель кровельных и фасадных систем.",
        )
        pipeline = PromptPipeline()
        result = pipeline.run(input_data)
        assert result.total_prompts > 0

    def test_audit_profile_create_brand_description_field(self):
        """AuditProfileCreate schema has brand_description field."""
        data = AuditProfileCreate(
            brand="Test",
            competitors=["Comp"],
            categories=["Cat"],
            market="en",
            brand_description="Test company description.",
        )
        assert data.brand_description == "Test company description."

    def test_audit_profile_create_brand_description_default(self):
        """AuditProfileCreate.brand_description defaults to empty string."""
        data = AuditProfileCreate(
            brand="Test",
            competitors=["Comp"],
            categories=["Cat"],
        )
        assert data.brand_description == ""


# ==========================================================================
# Helpers
# ==========================================================================


def _make_adapted(intent: Intent) -> AdaptedPrompt:
    """Create a minimal AdaptedPrompt for testing."""
    return AdaptedPrompt(
        target_llm=TargetLLM.CHATGPT,
        system_prompt="Test system prompt",
        user_prompt=f"Test user prompt for {intent.value}",
        persona=Persona.BEGINNER,
        intent=intent,
        category="Test",
        brand="TestBrand",
        competitors=["CompA"],
        market="en",
    )
