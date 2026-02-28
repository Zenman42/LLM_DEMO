"""PromptPipeline — orchestrator for the 5-layer prompt generation pipeline.

Usage:
    from app.prompt_engine.pipeline import PromptPipeline
    from app.schemas.prompt_engine import AuditProfileCreate

    profile_input = AuditProfileCreate(
        brand="Ahrefs",
        competitors=["SEMrush", "Moz", "Serpstat"],
        categories=["SEO-аналитика", "Анализ обратных ссылок", "Аудит сайтов"],
        market="ru",
    )

    pipeline = PromptPipeline()
    result = pipeline.run(profile_input)
    # result.prompts → list[FinalPrompt]
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from app.prompt_engine.adapters import adapt_prompts
from app.prompt_engine.dispatcher import parameterize_prompts
from app.prompt_engine.expansion import expand_skeletons
from app.prompt_engine.generator import generate_prompts_fallback, generate_prompts_v2
from app.prompt_engine.matrix import build_skeletons
from app.prompt_engine.normalization import normalize_profile
from app.prompt_engine.types import (
    AuditProfile,
    FinalPrompt,
    Intent,
    Persona,
)
from app.schemas.prompt_engine import AuditProfileCreate

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result of the prompt generation pipeline."""

    profile: AuditProfile
    prompts: list[FinalPrompt]
    skeletons_count: int = 0
    expanded_count: int = 0
    adapted_count: int = 0
    generation_method: str = ""  # "llm_v2" | "template" | "template_fallback"

    @property
    def total_prompts(self) -> int:
        return len(self.prompts)

    @property
    def by_provider(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for p in self.prompts:
            counter[p.target_llm.value] += 1
        return dict(counter)

    @property
    def by_intent(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for p in self.prompts:
            counter[p.intent.value] += 1
        return dict(counter)

    @property
    def by_persona(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for p in self.prompts:
            counter[p.persona.value] += 1
        return dict(counter)

    @property
    def by_measurement_type(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for p in self.prompts:
            if p.measurement_type:
                counter[p.measurement_type] += 1
        return dict(counter)

    @property
    def by_query_class(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for p in self.prompts:
            if p.query_class:
                counter[p.query_class] += 1
        return dict(counter)

    @property
    def by_query_subtype(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for p in self.prompts:
            if p.query_subtype:
                counter[p.query_subtype] += 1
        return dict(counter)

    def to_response_dict(self) -> dict:
        """Convert to API response format."""
        return {
            "total_prompts": self.total_prompts,
            "skeletons_generated": self.skeletons_count,
            "expanded_prompts": self.expanded_count,
            "adapted_prompts": self.adapted_count,
            "generation_method": self.generation_method,
            "by_provider": self.by_provider,
            "by_intent": self.by_intent,
            "by_persona": self.by_persona,
            "by_measurement_type": self.by_measurement_type,
            "by_query_class": self.by_query_class,
            "by_query_subtype": self.by_query_subtype,
            "prompts": [p.to_dict() for p in self.prompts],
        }

    def to_summary_dict(self) -> dict:
        """Convert to summary response (without full prompt list)."""
        return {
            "total_prompts": self.total_prompts,
            "skeletons_generated": self.skeletons_count,
            "expanded_prompts": self.expanded_count,
            "adapted_prompts": self.adapted_count,
            "by_provider": self.by_provider,
            "by_intent": self.by_intent,
            "by_persona": self.by_persona,
            "by_measurement_type": self.by_measurement_type,
            "by_query_class": self.by_query_class,
            "by_query_subtype": self.by_query_subtype,
        }


class PromptPipeline:
    """Orchestrator for the 5-layer prompt generation pipeline.

    Pipeline stages:
        1. Input & Normalization Gateway
        2. Combinatorial Matrix Engine
        3. Semantic Expansion & Pivot Engine
        4. Target-LLM Adapter
        5. Metadata Registry & Dispatcher
    """

    def run(
        self,
        input_data: AuditProfileCreate,
    ) -> PipelineResult:
        """Execute the full pipeline synchronously (rule-based only).

        For LLM-based expansion, use run_async() instead.
        """
        # Layer 1: Normalize input
        profile = normalize_profile(input_data)

        # Parse persona/intent filters
        personas = self._parse_personas(input_data.personas)
        intents = self._parse_intents(input_data.intents)
        target_llms = input_data.target_llms

        # Layer 2: Build combinatorial matrix
        skeletons = build_skeletons(profile, personas=personas, intents=intents)
        skeletons_count = len(skeletons)

        # Layer 3: Semantic expansion (rule-based)
        expanded = expand_skeletons(
            skeletons,
            profile,
            enable_pivots=input_data.enable_pivots,
            enable_conversational=True,
        )
        expanded_count = len(expanded)

        # Layer 4: Adapt for target LLMs
        adapted = adapt_prompts(expanded, target_llms=target_llms)
        adapted_count = len(adapted)

        # Layer 5: Parameterize
        final_prompts = parameterize_prompts(adapted)

        logger.info(
            "Pipeline complete: %d skeletons → %d expanded → %d adapted → %d final prompts",
            skeletons_count,
            expanded_count,
            adapted_count,
            len(final_prompts),
        )

        return PipelineResult(
            profile=profile,
            prompts=final_prompts,
            skeletons_count=skeletons_count,
            expanded_count=expanded_count,
            adapted_count=adapted_count,
        )

    async def run_async(
        self,
        input_data: AuditProfileCreate,
        expansion_api_key: str | None = None,
        expansion_model: str = "gemini-3-flash-preview",
        prompts_per_scenario: int = 10,
    ) -> PipelineResult:
        """Execute the pipeline with LLM-based V2 prompt generation.

        Primary path: Gemini generates natural prompts per scenario.

        Fallback: If no API key or LLM call fails, falls back to
        rule-based templates (matrix + expansion).
        """
        # Layer 1: Normalize input
        profile = normalize_profile(input_data)

        # Parse persona filters (intents are now derived from subtypes)
        personas = self._parse_personas(input_data.personas)
        intents = self._parse_intents(input_data.intents)
        target_llms = input_data.target_llms

        # Layer 2+3: Generate prompts via V2 LLM or fallback to templates
        use_llm = bool(expansion_api_key)

        if use_llm:
            try:
                expanded = await generate_prompts_v2(
                    profile=profile,
                    api_key=expansion_api_key,
                    personas=personas,
                    prompts_per_scenario=prompts_per_scenario,
                    model=expansion_model,
                )
                generation_method = "llm_v2"
                logger.info(
                    "V2 LLM generation produced %d prompts",
                    len(expanded),
                )
            except Exception as e:
                logger.warning(
                    "V2 LLM generation failed, falling back to templates: %s",
                    e,
                )
                expanded = generate_prompts_fallback(
                    profile=profile,
                    personas=personas,
                    intents=intents,
                )
                generation_method = "template_fallback"

            # Safety net: if LLM returned 0 prompts (e.g. invalid key, API down),
            # fall back to templates to ensure the user always gets something
            if not expanded and generation_method == "llm_v2":
                logger.warning("V2 LLM generation returned 0 prompts, falling back to templates")
                expanded = generate_prompts_fallback(
                    profile=profile,
                    personas=personas,
                    intents=intents,
                )
                generation_method = "template_fallback"
        else:
            expanded = generate_prompts_fallback(
                profile=profile,
                personas=personas,
                intents=intents,
            )
            generation_method = "template"

        expanded_count = len(expanded)

        # Layer 4: Adapt for target LLMs
        adapted = adapt_prompts(expanded, target_llms=target_llms)
        adapted_count = len(adapted)

        # Layer 5: Parameterize
        final_prompts = parameterize_prompts(adapted)

        logger.info(
            "Async pipeline complete (%s): %d expanded → %d adapted → %d final prompts",
            generation_method,
            expanded_count,
            adapted_count,
            len(final_prompts),
        )

        return PipelineResult(
            profile=profile,
            prompts=final_prompts,
            skeletons_count=0 if generation_method.startswith("llm") else expanded_count,
            expanded_count=expanded_count,
            adapted_count=adapted_count,
            generation_method=generation_method,
        )

    @staticmethod
    def _parse_personas(names: list[str]) -> list[Persona]:
        """Convert string persona names to Persona enums."""
        result = []
        for name in names:
            try:
                result.append(Persona(name.lower()))
            except ValueError:
                logger.warning("Unknown persona: %s, skipping", name)
        return result or list(Persona)

    @staticmethod
    def _parse_intents(names: list[str]) -> list[Intent]:
        """Convert string intent names to Intent enums."""
        result = []
        for name in names:
            try:
                result.append(Intent(name.lower()))
            except ValueError:
                logger.warning("Unknown intent: %s, skipping", name)
        return result or list(Intent)
