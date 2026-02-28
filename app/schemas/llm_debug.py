"""Pydantic response models for LLM Debug Console endpoints."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Per-snapshot row in the debug table
# ---------------------------------------------------------------------------


class DebugSnapshotItem(BaseModel):
    """Single snapshot with all debug-relevant fields."""

    id: int
    query_id: int
    query_text: str
    query_type: str
    date: date
    provider: str
    model: str
    brand_mentioned: bool
    mention_type: str
    mention_context: str | None = None
    competitor_mentions: dict | None = None
    cited_urls: list[str] | None = None
    tokens: int | None = None
    cost: float | None = None

    # Computed contributions (for AIVS debug)
    rvs: float = 0.0
    rvs_source: str = ""
    intent_weight: float = 0.0
    intent_source: str = ""
    contribution: float = 0.0

    # Whether the brand name appears in the query text itself
    brand_in_prompt: bool = False


# ---------------------------------------------------------------------------
# AIVS Debug Trace
# ---------------------------------------------------------------------------


class AivsDebug(BaseModel):
    """Full trace of AIVS computation."""

    final_score: float
    weighted_sum: float
    weight_sum: float
    raw_ratio: float
    capped: bool
    formula: str
    per_snapshot: list[DebugSnapshotItem]


# ---------------------------------------------------------------------------
# SoM Debug Trace
# ---------------------------------------------------------------------------


class SomDebug(BaseModel):
    """Full trace of Share of Model computation."""

    final_score: float
    brand_count: int
    total_mentions: int
    per_competitor: dict[str, int]
    formula: str


# ---------------------------------------------------------------------------
# Resilience Debug Trace
# ---------------------------------------------------------------------------


class ResilienceGroupDebug(BaseModel):
    """Single (query_id, provider) group in resilience trace."""

    query_id: int
    query_text: str
    provider: str
    total: int
    mentioned: int
    resilience: float


class ResilienceDebug(BaseModel):
    """Full trace of Resilience computation."""

    final_score: float
    groups: list[ResilienceGroupDebug]
    formula: str


# ---------------------------------------------------------------------------
# Mention Rate Debug Trace
# ---------------------------------------------------------------------------


class MentionRateDebug(BaseModel):
    """Full trace of Mention Rate computation — total + organic (unprompted)."""

    final_rate: float
    brand_mentions: int
    total: int
    formula: str

    # Organic rate: only queries where brand is NOT in the prompt text
    organic_rate: float = 0.0
    organic_mentions: int = 0
    organic_total: int = 0
    organic_formula: str = ""

    # Prompted rate: queries where brand IS in the prompt text
    prompted_rate: float = 0.0
    prompted_mentions: int = 0
    prompted_total: int = 0
    prompted_formula: str = ""


# ---------------------------------------------------------------------------
# Full Debug Response
# ---------------------------------------------------------------------------


class LlmDebugResponse(BaseModel):
    """Complete debug trace for all metrics — response of /api/v1/llm/debug/{project_id}."""

    project_id: int
    days: int
    snapshot_count: int

    aivs_debug: AivsDebug
    som_debug: SomDebug
    resilience_debug: ResilienceDebug
    mention_rate_debug: MentionRateDebug

    snapshots: list[DebugSnapshotItem]


# ---------------------------------------------------------------------------
# Pipeline Trace (single snapshot re-analysis)
# ---------------------------------------------------------------------------


class TraceTargetBrand(BaseModel):
    """Entity resolution result for target brand."""

    name: str = ""
    is_mentioned: bool = False
    position_rank: int = 0
    position_weight: float = 0.0
    mention_type: str = "none"
    mention_context: str = ""
    char_offset: int = -1
    sentence_index: int = -1


class TraceCompetitor(BaseModel):
    """Entity resolution result for a competitor."""

    name: str = ""
    is_mentioned: bool = False
    position_rank: int = 0
    position_weight: float = 0.0
    mention_type: str = "none"


class TraceSanitization(BaseModel):
    """Step 1: Sanitization result."""

    flag: str = "clean"
    stripped_chars: int = 0
    think_content: str = ""


class TraceEntityResolution(BaseModel):
    """Step 2: Entity resolution results."""

    target_brand: TraceTargetBrand
    competitors: list[TraceCompetitor]


class TraceStructure(BaseModel):
    """Step 3: Structure & ranking analysis."""

    type: str = "narrative"
    total_items: int = 0
    brands_in_list: list[str] = Field(default_factory=list)


class TraceContext(BaseModel):
    """Step 4: Context evaluation (LLM-as-a-Judge)."""

    mention_type: str = "none"
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    sentiment_multiplier: float = 1.0
    context_tags: list[str] = Field(default_factory=list)
    is_recommended: bool = False
    is_hallucination: bool = False

    # Judge debug transparency
    judge_method: str = ""  # "llm_judge" or "heuristic"
    judge_prompt_system: str = ""  # System prompt sent to judge LLM
    judge_prompt_user: str = ""  # User prompt sent to judge LLM
    judge_raw_response: str = ""  # Raw JSON response from judge LLM


class TraceCitation(BaseModel):
    """Single citation from step 5."""

    url: str = ""
    domain: str = ""
    is_native: bool = False


class TraceScoring(BaseModel):
    """Step 6: Micro-scoring results with full RVS breakdown."""

    rvs: float = 0.0
    share_of_model_local: float = 0.0

    # --- RVS formula breakdown: RVS = presence × position_weight × sentiment_multiplier ---
    presence: float = 0.0  # 1.0 if brand mentioned, 0.0 otherwise
    position_rank: int = 0  # rank in list (1-based), 0 for narrative
    position_weight: float = 0.0  # 0.1–1.0 from ranking parser
    position_source: str = ""  # e.g. "numbered_list: rank #5 → 1.0 - (5-1)×0.1 = 0.6"
    sentiment_multiplier: float = 1.0  # 0.5–1.2 from context evaluator
    sentiment_label: str = ""  # "positive" / "neutral" / "mixed" / "negative"
    sentiment_source: str = ""  # e.g. "positive → 1.2"
    structure_type: str = ""  # "numbered_list" / "bulleted_list" / "narrative" / ...
    rvs_formula: str = ""  # human-readable: "RVS = 1.0 × 0.6 × 1.2 = 0.72"

    # --- Share of Model breakdown ---
    total_mentioned_brands: int = 0  # how many brands mentioned in this response
    mentioned_competitors: list[str] = Field(default_factory=list)  # which competitors were mentioned
    som_formula: str = ""  # "SoM_local = 1 / 2 = 0.50"


class TracePromptOrigin(BaseModel):
    """How the query_text was generated — genealogy from template to API call."""

    matched_template: str = ""  # e.g. "Где найти честные отзывы о {brand} для {category}"
    persona: str = ""  # e.g. "skeptic"
    intent: str = ""  # e.g. "navigational"
    match_confidence: str = "none"  # "exact" | "fuzzy" | "none"


class TraceLlmQuery(BaseModel):
    """Step 0: LLM Query — what was actually sent to the LLM API."""

    system_prompt: str = ""
    user_prompt: str = ""
    provider: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    query_type: str = ""
    target_brand: str = ""
    competitors: list[str] = Field(default_factory=list)
    # Full API payload as it was sent to the vendor (reconstructed)
    api_payload: dict = Field(default_factory=dict)
    # Prompt origin — how the query was generated
    prompt_origin: TracePromptOrigin = Field(default_factory=TracePromptOrigin)


class PipelineTraceResponse(BaseModel):
    """Full pipeline trace for a single snapshot — response of /api/v1/llm/debug/trace/{id}."""

    snapshot_id: int
    query_text: str
    provider: str
    raw_response: str = ""

    step_0_llm_query: TraceLlmQuery
    step_1_sanitization: TraceSanitization
    step_2_entity_resolution: TraceEntityResolution
    step_3_structure: TraceStructure
    step_4_context: TraceContext
    step_5_citations: list[TraceCitation]
    step_6_scoring: TraceScoring


# ---------------------------------------------------------------------------
# Single snapshot response (for inspector fix)
# ---------------------------------------------------------------------------


class SnapshotWithQueryText(BaseModel):
    """Single snapshot with query_text included — for the fixed inspector modal."""

    id: int
    llm_query_id: int
    query_text: str
    date: date
    llm_provider: str
    llm_model: str
    brand_mentioned: bool
    mention_type: str
    mention_context: str | None = None
    competitor_mentions: dict | None = None
    cited_urls: list[str] | None = None
    response_tokens: int | None = None
    cost_usd: float | None = None
    raw_response: str | None = None
    collected_at: datetime | None = None
