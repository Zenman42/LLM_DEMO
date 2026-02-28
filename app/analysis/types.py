"""Core types and DTOs for the Analysis & Evaluation Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SanitizationFlag(str, Enum):
    """Flags assigned during text preprocessing."""

    CLEAN = "clean"  # Normal text, passed through
    DEEPSEEK_THINK_STRIPPED = "deepseek_think_stripped"
    VENDOR_REFUSAL = "vendor_refusal"  # Guardrail / safety refusal detected
    CENSORED = "censored"  # Already marked CENSORED by Gateway
    EMPTY_RESPONSE = "empty_response"


class MentionType(str, Enum):
    """How a brand is mentioned in the response."""

    DIRECT = "direct"  # Named explicitly
    RECOMMENDED = "recommended"  # Recommended / best choice / top pick
    COMPARED = "compared"  # Mentioned alongside competitors
    NEGATIVE = "negative"  # Mentioned in negative context
    NONE = "none"  # Not mentioned


class StructureType(str, Enum):
    """Detected structural format of the response."""

    NUMBERED_LIST = "numbered_list"  # 1. Brand A  2. Brand B ...
    BULLETED_LIST = "bulleted_list"  # - Brand A  - Brand B ...
    NARRATIVE = "narrative"  # Continuous prose
    TABLE = "table"  # Markdown table
    MIXED = "mixed"  # Combination


class SentimentLabel(str, Enum):
    """Coarse sentiment bucket."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


# ---------------------------------------------------------------------------
# Data containers — atomic analysis results
# ---------------------------------------------------------------------------


@dataclass
class BrandMention:
    """Entity resolution result for a single brand within a response."""

    name: str = ""  # Canonical brand name
    aliases_matched: list[str] = field(default_factory=list)  # Which aliases triggered
    is_mentioned: bool = False
    position_rank: int = 0  # 0 = not in a ranked list
    position_weight: float = 0.0  # 0.0–1.0 decay weight
    mention_type: MentionType = MentionType.NONE
    mention_context: str = ""  # Text fragment around mention
    char_offset: int = -1  # Character offset of first mention
    sentence_index: int = -1  # Sentence number (0-based)

    # Context Evaluator output (LLM-as-a-Judge)
    sentiment_score: float = 0.0  # -1.0 (negative) .. +1.0 (positive)
    sentiment_label: SentimentLabel = SentimentLabel.NEUTRAL
    sentiment_multiplier: float = 1.0  # Applied to RVS
    context_tags: list[str] = field(default_factory=list)  # e.g. ["price", "quality"]
    is_recommended: bool = False
    is_hallucination: bool = False

    # Judge debug trace (for transparency)
    judge_method: str = ""  # "llm_judge" or "heuristic"
    judge_prompt_system: str = ""  # System prompt sent to judge
    judge_prompt_user: str = ""  # User prompt sent to judge
    judge_raw_response: str = ""  # Raw JSON response from judge LLM


@dataclass
class CitationInfo:
    """A single citation / source reference extracted from the response."""

    url: str = ""
    domain: str = ""
    anchor_text: str = ""  # Text label if [text](url) format
    footnote_index: int = 0  # [1], [2] etc.
    is_native: bool = False  # From vendor API (Perplexity citations)


@dataclass
class StructureAnalysis:
    """Result of structural / ranking analysis."""

    structure_type: StructureType = StructureType.NARRATIVE
    total_items: int = 0  # Number of items in list
    brands_in_list: list[str] = field(default_factory=list)  # Ordered list of brand names found


@dataclass
class SanitizedText:
    """Output of the text preprocessor."""

    text: str = ""  # Cleaned text for analysis
    original_text: str = ""  # Raw text before cleaning
    flag: SanitizationFlag = SanitizationFlag.CLEAN
    think_content: str = ""  # Extracted <think> block (DeepSeek)
    stripped_chars: int = 0  # How many chars were removed


# ---------------------------------------------------------------------------
# Main output DTO — AnalyzedResponse
# ---------------------------------------------------------------------------


@dataclass
class AnalyzedResponse:
    """Complete analysis result for a single GatewayResponse.

    This is the output of the full analysis pipeline (Module 3).
    """

    # Identity / passthrough from GatewayResponse
    request_id: str = ""
    session_id: str = ""
    vendor: str = ""
    model_version: str = ""
    prompt_id: str = ""
    tenant_id: str = ""
    project_id: int = 0
    llm_query_id: int = 0

    # Prompt metadata (from Module 1 via Gateway context)
    intent: str = ""
    persona: str = ""

    # Pipeline step 1: Sanitization
    sanitization: SanitizedText = field(default_factory=SanitizedText)

    # Pipeline step 2: Entity Resolution
    target_brand: BrandMention = field(default_factory=BrandMention)
    competitors: list[BrandMention] = field(default_factory=list)
    discovered_entities: list[str] = field(default_factory=list)  # brands found but not matched
    raw_entities: list = field(default_factory=list)  # enriched: [{"name", "context", "type"}] or legacy [str]

    # Pipeline step 3: Structure & Ranking
    structure: StructureAnalysis = field(default_factory=StructureAnalysis)

    # Pipeline step 5: Citations
    citations: list[CitationInfo] = field(default_factory=list)

    # Pipeline step 6: Micro-Scoring
    response_visibility_score: float = 0.0  # RVS
    share_of_model_local: float = 0.0  # % of brands mentioned in this response

    # Gateway metrics passthrough
    latency_ms: int = 0
    cost_usd: float = 0.0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for storage/API."""
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "vendor": self.vendor,
            "model_version": self.model_version,
            "prompt_id": self.prompt_id,
            "prompt_metadata": {
                "intent": self.intent,
                "persona": self.persona,
            },
            "extracted_metrics": {
                "target_brand": {
                    "name": self.target_brand.name,
                    "is_mentioned": self.target_brand.is_mentioned,
                    "position_rank": self.target_brand.position_rank,
                    "position_weight": self.target_brand.position_weight,
                    "sentiment_score": self.target_brand.sentiment_score,
                    "sentiment_multiplier": self.target_brand.sentiment_multiplier,
                    "context_tags": self.target_brand.context_tags,
                    "is_hallucination": self.target_brand.is_hallucination,
                    "is_recommended": self.target_brand.is_recommended,
                    "mention_type": self.target_brand.mention_type.value,
                },
                "competitors_mentioned": [
                    {
                        "name": c.name,
                        "is_mentioned": c.is_mentioned,
                        "position_rank": c.position_rank,
                        "position_weight": c.position_weight,
                        "sentiment_score": c.sentiment_score,
                        "mention_type": c.mention_type.value,
                    }
                    for c in self.competitors
                    if c.is_mentioned
                ],
                "citations_found": [{"domain": c.domain, "url": c.url} for c in self.citations],
            },
            "calculated_scores": {
                "response_visibility_score": round(self.response_visibility_score, 4),
                "share_of_model_local": round(self.share_of_model_local, 2),
            },
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }
