"""Base collector interface."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CollectionResult:
    """Result of a single collector run."""

    collected: int = 0
    errors: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


class BaseCollector(ABC):
    """Abstract base for all collectors."""

    @abstractmethod
    async def collect(self, **kwargs) -> CollectionResult:
        """Run collection. Returns CollectionResult."""
        ...
