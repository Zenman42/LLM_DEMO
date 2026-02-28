"""Core types and enums for the Prompt Engineering Module.

V2 architecture: two query classes (thematic / branded) with subtypes.
Legacy enums (Intent, MeasurementType) preserved for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════
# V2: Query Class / Subtype — primary generation axis
# ═══════════════════════════════════════════════════════════════════════════


class QueryClass(str, Enum):
    """Top-level query classification for brand visibility tracking."""

    THEMATIC = "thematic"  # No brand in query → measure which brands LLM recalls
    BRANDED = "branded"  # Brand in query → measure LLM's opinion about brand


class ThematicSubtype(str, Enum):
    """Subtypes for thematic (brand-free) queries."""

    CATEGORY = "category"  # Generic "who is best" in category
    ATTRIBUTE = "attribute"  # Focused on specific property (reliability, price, etc.)
    SCENARIO = "scenario"  # User situation + recommendation request
    EVENT = "event"  # Tied to event/trend/season + who to choose


class BrandedSubtype(str, Enum):
    """Subtypes for branded (brand-in-query) queries."""

    REPUTATION = "reputation"  # What does LLM know/think about brand
    COMPARISON = "comparison"  # Direct X vs Y comparison
    NEGATIVE = "negative"  # Drawbacks, problems, complaints
    FACT_CHECK = "fact_check"  # Verify specific facts (golden facts)


# Unified subtype type for convenience
QuerySubtype = ThematicSubtype | BrandedSubtype

# All subtype values as flat list (for validation)
ALL_SUBTYPES = [e.value for e in ThematicSubtype] + [e.value for e in BrandedSubtype]


THEMATIC_SUBTYPE_DESCRIPTIONS = {
    ThematicSubtype.CATEGORY: {
        "ru": (
            "Категорийный — общий запрос «кто лучший в категории». "
            "Запрос НЕ содержит брендов. Ответ LLM ОБЯЗАН содержать конкретные бренды/компании. "
            "Примеры: «какой застройщик лучше в москве», «рейтинг лучших CRM для малого бизнеса»."
        ),
        "en": (
            "Category — generic 'who is the best in category' query. "
            "Query contains NO brands. LLM response MUST name specific brands/companies. "
            "Examples: 'best real estate developer in Moscow', 'top CRM for small business'."
        ),
    },
    ThematicSubtype.ATTRIBUTE: {
        "ru": (
            "По атрибуту — фокус на конкретном свойстве: надёжность, цена, сроки, качество, поддержка. "
            "Запрос НЕ содержит брендов. Ответ LLM ОБЯЗАН содержать конкретные бренды/компании. "
            "Примеры: «у какого застройщика самые надёжные сроки сдачи», "
            "«какая CRM самая дешёвая для стартапа»."
        ),
        "en": (
            "By attribute — focused on a specific property: reliability, price, speed, quality, support. "
            "Query contains NO brands. LLM response MUST name specific brands/companies. "
            "Examples: 'which developer has the most reliable delivery dates', "
            "'cheapest CRM for startups'."
        ),
    },
    ThematicSubtype.SCENARIO: {
        "ru": (
            "Сценарный — описание конкретной ситуации пользователя + просьба порекомендовать. "
            "Запрос НЕ содержит брендов. Ответ LLM ОБЯЗАН содержать конкретные бренды/компании. "
            "Примеры: «выбираю квартиру для семьи с детьми, каких застройщиков смотреть», "
            "«посоветуй CRM если у меня команда из 5 человек и бюджет 10к в месяц»."
        ),
        "en": (
            "Scenario — describes user's specific situation + asks for recommendation. "
            "Query contains NO brands. LLM response MUST name specific brands/companies. "
            "Examples: 'choosing an apartment for a family with kids, which developers to consider', "
            "'recommend a CRM for a 5-person team with $200/mo budget'."
        ),
    },
    ThematicSubtype.EVENT: {
        "ru": (
            "По событию/тренду — привязка к конкретному событию, сезону, рыночной ситуации. "
            "Запрос НЕ содержит брендов. Ответ LLM ОБЯЗАН содержать конкретные бренды/компании. "
            "Примеры: «какие застройщики дают лучшие условия при текущих ставках ипотеки», "
            "«у кого из CRM лучшие интеграции с новыми AI-инструментами в 2025»."
        ),
        "en": (
            "By event/trend — tied to specific event, season, or market situation. "
            "Query contains NO brands. LLM response MUST name specific brands/companies. "
            "Examples: 'which developers offer best terms with current mortgage rates', "
            "'which CRM has the best AI integrations in 2025'."
        ),
    },
}

BRANDED_SUBTYPE_DESCRIPTIONS = {
    BrandedSubtype.REPUTATION: {
        "ru": (
            "Репутационный — что LLM знает и думает о бренде. "
            "Запрос СОДЕРЖИТ название бренда. "
            "Примеры: «что за застройщик Кортрос, стоит ли брать», "
            "«расскажи про CRM Битрикс24, для чего она подходит»."
        ),
        "en": (
            "Reputation — what the LLM knows and thinks about the brand. "
            "Query CONTAINS the brand name. "
            "Examples: 'what is Kortros as a developer, is it worth buying from them', "
            "'tell me about Bitrix24 CRM, what is it good for'."
        ),
    },
    BrandedSubtype.COMPARISON: {
        "ru": (
            "Сравнение — прямое сравнение бренда с 1-2 конкурентами. "
            "Запрос СОДЕРЖИТ 2-3 названия брендов. "
            "Примеры: «Кортрос или ПИК что лучше», «Битрикс24 vs AmoCRM сравнение»."
        ),
        "en": (
            "Comparison — direct comparison of brand with 1-2 competitors. "
            "Query CONTAINS 2-3 brand names. "
            "Examples: 'Kortros vs PIK which is better', 'Bitrix24 vs AmoCRM comparison'."
        ),
    },
    BrandedSubtype.NEGATIVE: {
        "ru": (
            "Негативный — запрос о недостатках, проблемах, жалобах на бренд. "
            "Запрос СОДЕРЖИТ название бренда. "
            "Примеры: «минусы Кортрос отзывы», «проблемы с Битрикс24», "
            "«почему не стоит выбирать Кортрос»."
        ),
        "en": (
            "Negative — query about drawbacks, problems, complaints about the brand. "
            "Query CONTAINS the brand name. "
            "Examples: 'Kortros downsides reviews', 'problems with Bitrix24', "
            "'why not to choose Kortros'."
        ),
    },
    BrandedSubtype.FACT_CHECK: {
        "ru": (
            "Фактчек — проверка конкретного факта о бренде (golden facts). "
            "Запрос СОДЕРЖИТ название бренда + конкретное утверждение. "
            "Цель: знает ли LLM реальные факты или галлюцинирует. "
            "Примеры: «правда что у Кортрос есть рассрочка без процентов», "
            "«у Битрикс24 есть интеграция с 1С?»."
        ),
        "en": (
            "Fact-check — verify specific facts about the brand (golden facts). "
            "Query CONTAINS the brand name + specific claim. "
            "Goal: does the LLM know real facts or hallucinate. "
            "Examples: 'is it true that Kortros offers interest-free installments', "
            "'does Bitrix24 integrate with QuickBooks?'."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# V1 (legacy): Personas — still used for generation variety
# ═══════════════════════════════════════════════════════════════════════════


class Persona(str, Enum):
    """User archetype that defines the communication style and focus."""

    BEGINNER = "beginner"  # Novice / student — simple language
    CTO = "cto"  # Executive / expert — specs, reliability, guarantees
    SKEPTIC = "skeptic"  # Skeptic — hidden costs, drawbacks, pitfalls
    MARKETER = "marketer"  # Marketer / decision-maker — value, price/quality


PERSONA_DESCRIPTIONS = {
    Persona.BEGINNER: {
        "ru": "Новичок, который ищет базовую информацию простым языком",
        "en": "Beginner looking for basic information in simple language",
    },
    Persona.CTO: {
        "ru": "Руководитель / эксперт, ищущий детальные характеристики, надёжность, гарантии",
        "en": "Executive / expert looking for detailed specs, reliability, guarantees",
    },
    Persona.SKEPTIC: {
        "ru": "Скептик, целенаправленно ищущий скрытые расходы, недостатки, подводные камни",
        "en": "Skeptic looking for hidden costs, drawbacks, and pitfalls",
    },
    Persona.MARKETER: {
        "ru": "Маркетолог / ЛПР, ищущий лучшее соотношение цена/качество и выгоду",
        "en": "Marketer / decision-maker looking for best value and results",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# V1 (legacy): Intents — preserved for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════


class Intent(str, Enum):
    """Search intent type (legacy — used by old pipeline and adapters)."""

    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    COMPARATIVE = "comparative"
    TRANSACTIONAL = "transactional"


INTENT_DESCRIPTIONS = {
    Intent.INFORMATIONAL: {
        "ru": "Информационный — пользователь хочет разобраться в теме",
        "en": "Informational — user wants to understand the topic",
    },
    Intent.NAVIGATIONAL: {
        "ru": "Навигационный — пользователь ищет конкретную информацию о бренде",
        "en": "Navigational — user is looking for specific brand information",
    },
    Intent.COMPARATIVE: {
        "ru": "Сравнительный — пользователь выбирает между вариантами",
        "en": "Comparative — user is choosing between options",
    },
    Intent.TRANSACTIONAL: {
        "ru": "Транзакционный — пользователь готов к действию",
        "en": "Transactional — user is ready to act",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# V1 (legacy): Measurement Types — preserved for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════


class MeasurementType(str, Enum):
    """Measurement category (legacy — mapped to V2 subtypes in new generator)."""

    TOP_OF_MIND = "top_of_mind"
    COMPARISON = "comparison"
    BY_EVENT = "by_event"
    BY_ATTRIBUTE = "by_attribute"
    PERSONA_SCENARIO = "persona_scenario"
    NEGATIVE = "negative"
    ASSOCIATIVE = "associative"


# Legacy descriptions kept minimal — full descriptions in V2 subtypes above
MEASUREMENT_TYPE_DESCRIPTIONS = {
    MeasurementType.TOP_OF_MIND: {
        "ru": "Top-of-mind — общие категорийные запросы без брендов",
        "en": "Top-of-mind — generic category queries without brands",
    },
    MeasurementType.COMPARISON: {
        "ru": "Сравнение брендов",
        "en": "Brand comparison",
    },
    MeasurementType.BY_EVENT: {
        "ru": "По событиям/контексту",
        "en": "By event/context",
    },
    MeasurementType.BY_ATTRIBUTE: {
        "ru": "По атрибуту",
        "en": "By attribute",
    },
    MeasurementType.PERSONA_SCENARIO: {
        "ru": "Персональный сценарий",
        "en": "Persona scenario",
    },
    MeasurementType.NEGATIVE: {
        "ru": "Негативные запросы",
        "en": "Negative queries",
    },
    MeasurementType.ASSOCIATIVE: {
        "ru": "Ассоциативные/фактчек",
        "en": "Associative/fact-check",
    },
}


# ---------------------------------------------------------------------------
# Target LLM providers
# ---------------------------------------------------------------------------


class TargetLLM(str, Enum):
    """Target LLM provider for prompt adaptation."""

    CHATGPT = "chatgpt"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    YANDEXGPT = "yandexgpt"
    GIGACHAT = "gigachat"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """Per-scenario configuration for prompt generation.

    Each scenario (customer use-case) can have its own set of
    enabled query subtypes, description, and context hints.
    """

    name: str
    description: str = ""
    context_hints: str = ""
    thematic_subtypes: list[str] = field(
        default_factory=lambda: [s.value for s in ThematicSubtype],
    )
    branded_subtypes: list[str] = field(
        default_factory=list,  # Empty by default — branded only when explicitly requested
    )

    @property
    def rich_label(self) -> str:
        """Combine name + description + hints into a rich category label for prompts."""
        parts = [self.name]
        if self.description:
            parts.append(self.description)
        if self.context_hints:
            parts.append(self.context_hints)
        return ". ".join(parts)

    @staticmethod
    def from_raw(raw) -> "ScenarioConfig":
        """Create from string (legacy) or dict (new format)."""
        if isinstance(raw, str):
            return ScenarioConfig(name=raw)
        if isinstance(raw, dict):
            return ScenarioConfig(
                name=raw.get("name", ""),
                description=raw.get("description", ""),
                context_hints=raw.get("context_hints", ""),
                thematic_subtypes=raw.get("thematic_subtypes", [s.value for s in ThematicSubtype]),
                branded_subtypes=raw.get("branded_subtypes", []),
            )
        if isinstance(raw, ScenarioConfig):
            return raw
        raise ValueError(f"Cannot convert {type(raw)} to ScenarioConfig")


@dataclass
class AuditProfile:
    """Input profile for prompt generation — what to audit."""

    brand: str
    competitors: list[str]
    scenarios: list[ScenarioConfig] = field(default_factory=list)
    market: str = "ru"  # ru / en / global
    brand_aliases: list[str] = field(default_factory=list)  # synonyms, abbreviations
    geo: str = ""  # geographic context, e.g. "Russia", "Moscow"
    golden_facts: list[str] = field(default_factory=list)  # anti-hallucination facts
    brand_description: str = ""  # company business description

    @property
    def categories(self) -> list[str]:
        """Backward-compat: return scenario names as flat string list."""
        return [s.name for s in self.scenarios]

    def all_brand_names(self) -> list[str]:
        """Return brand + all aliases as a flat list."""
        return [self.brand] + self.brand_aliases


@dataclass
class PromptSkeleton:
    """Raw skeleton prompt generated by the Combinatorial Matrix Engine."""

    template: str  # Template text with placeholders
    persona: Persona
    intent: Intent
    category: str
    brand: str
    competitors: list[str]
    market: str

    def render(self) -> str:
        """Render the template with actual values."""
        return self.template


@dataclass
class ExpandedPrompt:
    """A natural-language prompt generated by the LLM or template engine."""

    text: str  # Natural language prompt text
    persona: Persona
    intent: Intent
    category: str
    brand: str
    competitors: list[str]
    market: str
    skeleton_id: str = ""  # Reference to the source skeleton
    pivot_axis: str = ""  # Semantic pivot applied (e.g. "cheapest", "most reliable")
    measurement_type: str = ""  # Legacy MeasurementType value

    # V2 fields
    query_class: str = ""  # "thematic" | "branded"
    query_subtype: str = ""  # "category" | "attribute" | ... | "reputation" | ...


@dataclass
class AdaptedPrompt:
    """A prompt adapted for a specific target LLM."""

    target_llm: TargetLLM
    system_prompt: str  # System/role prompt for the target LLM
    user_prompt: str  # User message text
    persona: Persona
    intent: Intent
    category: str
    brand: str
    competitors: list[str]
    market: str
    skeleton_id: str = ""
    pivot_axis: str = ""
    measurement_type: str = ""

    # V2 fields
    query_class: str = ""
    query_subtype: str = ""


@dataclass
class FinalPrompt:
    """Fully parameterized prompt ready for API dispatch (output of Module 1)."""

    prompt_id: str  # Unique identifier
    target_llm: TargetLLM
    system_prompt: str
    user_prompt: str
    temperature: float  # 0.0 for factual, 0.7 for creative
    resilience_runs: int  # How many times to repeat (1, 3, or 5)
    max_tokens: int

    # Metadata for dashboard filtering
    persona: Persona
    intent: Intent
    category: str
    brand: str
    competitors: list[str]
    market: str
    skeleton_id: str = ""
    pivot_axis: str = ""
    measurement_type: str = ""

    # V2 fields
    query_class: str = ""  # "thematic" | "branded"
    query_subtype: str = ""  # "category" | "attribute" | ... | "reputation" | ...

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for API payload."""
        return {
            "prompt_id": self.prompt_id,
            "target_llm": self.target_llm.value,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "temperature": self.temperature,
            "resilience_runs": self.resilience_runs,
            "max_tokens": self.max_tokens,
            "metadata": {
                "persona": self.persona.value,
                "intent": self.intent.value,
                "category": self.category,
                "brand": self.brand,
                "competitors": self.competitors,
                "market": self.market,
                "skeleton_id": self.skeleton_id,
                "pivot_axis": self.pivot_axis,
                "measurement_type": self.measurement_type,
                "query_class": self.query_class,
                "query_subtype": self.query_subtype,
            },
        }
