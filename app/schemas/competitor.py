from datetime import date, datetime

from pydantic import BaseModel, Field


class AliasInfo(BaseModel):
    """Entity-based alias: an entity that is an alternative name for another entity."""

    id: int
    entity_name: str
    mention_count: int = 0


class BrandInfo(BaseModel):
    name: str
    mention_count: int = 0
    aliases: list[str] = Field(default_factory=list)  # project.brand_aliases (manual strings)
    entity_aliases: list[AliasInfo] = Field(default_factory=list)  # entity-based aliases (via alias_of_id)
    entity_mention_count: int = 0  # entity's own DiscoveredEntity.mention_count
    entity_mention_count_with_aliases: int = 0  # entity + all aliases
    entity_mention_count_total: int = 0  # entity + aliases + all sub-brands with their aliases


class CompetitorInfo(BaseModel):
    name: str
    mention_count: int = 0  # aggregated: own + all aliases
    aliases: list[AliasInfo] = Field(default_factory=list)
    entity_mention_count: int = 0  # entity's own DiscoveredEntity.mention_count
    entity_mention_count_with_aliases: int = 0  # entity + all aliases
    entity_mention_count_total: int = 0  # entity + aliases + all sub-brands with their aliases


class CompetitorListResponse(BaseModel):
    brand: BrandInfo | None = None
    competitors: list[CompetitorInfo]
    discovered_count: int = 0


class AddCompetitorRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class DiscoveredEntityResponse(BaseModel):
    id: int
    entity_name: str
    mention_count: int
    first_seen: date
    last_seen: date
    status: str
    verified_at: datetime | None = None
    verified_by: str | None = None
    suggested_parent: str | None = None
    llm_explanation: str | None = None
    alias_of_id: int | None = None

    model_config = {"from_attributes": True}


class VerifyRequest(BaseModel):
    entities: list[str] = Field(min_length=1, max_length=50)


class VerifyResult(BaseModel):
    entity: str
    is_competitor: bool
    is_sub_brand: bool = False
    parent_brand: str | None = None
    is_alias: bool = False
    alias_of: str | None = None
    explanation: str = ""


class VerifyResponse(BaseModel):
    results: list[VerifyResult]


class PromoteRequest(BaseModel):
    entities: list[str] = Field(min_length=1, max_length=50)


class RejectRequest(BaseModel):
    entities: list[str] = Field(min_length=1, max_length=50)


class VerifyApplyAction(BaseModel):
    entity: str
    action: str  # "accept" | "accept_and_add" | "disagree"
    is_competitor: bool = False
    is_sub_brand: bool = False
    parent_brand: str | None = None
    is_alias: bool = False
    alias_of: str | None = None
    explanation: str = ""


class VerifyApplyRequest(BaseModel):
    decisions: list[VerifyApplyAction] = Field(min_length=1, max_length=200)


class SubBrandCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    aliases: list[str] = Field(default_factory=list)


class SubBrandResponse(BaseModel):
    name: str
    mention_count: int = 0  # aggregated: own + all aliases
    aliases: list[AliasInfo] = Field(default_factory=list)
    entity_mention_count: int = 0  # entity's own DiscoveredEntity.mention_count
    entity_mention_count_with_aliases: int = 0  # entity + all aliases


class MergeBrandRequest(BaseModel):
    source: str = Field(min_length=1, max_length=255, description="Brand/competitor to merge FROM (will be removed)")
    target: str = Field(
        min_length=1, max_length=255, description="Brand/competitor to merge INTO (will receive aliases)"
    )


class SetAliasRequest(BaseModel):
    entity_name: str = Field(min_length=1, max_length=255, description="Entity to mark as alias")
    target_entity_name: str = Field(min_length=1, max_length=255, description="Entity it is an alias of")


class RemoveAliasRequest(BaseModel):
    entity_name: str = Field(min_length=1, max_length=255, description="Entity to un-alias")


class ResetEntityAliasesRequest(BaseModel):
    entity_name: str = Field(min_length=1, max_length=255, description="Entity whose aliases to bulk-release")
