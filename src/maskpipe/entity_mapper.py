from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Union, Required, cast

class EntityResult(TypedDict):
    start: Required[int]
    end: Required[int]
    label: Required[str]
    score: Required[float]

@dataclass(frozen=True)
class BaseEntityMapper(ABC):
    """Base class for mapping external model results to canonical entity format."""
    start: str = "start"
    end: str = "end"
    label: str = "label"
    label_fallback: Optional[str] = None
    score: str = "score"    
    
    def _get_label(self, entity: Dict[str, Any]) -> str:
        """Extract label from entity, raising KeyError if not found."""
        label = entity.get(self.label)
        if label:
            return label
        if self.label_fallback:
            return entity[self.label_fallback]
        raise KeyError(self.label)

    @abstractmethod
    def map(self, result: Any, default_score: float) -> List[EntityResult]:
        """Map external format to canonical EntityResult list."""
        raise NotImplementedError

@dataclass(frozen=True)
class EntityMapper(BaseEntityMapper):
    """Maps external keys to canonical format {start, end, label, score}."""

    def map(
        self,
        result: Union[
            List[Dict[str, Any]],
            Dict[str, List[Dict[str, Any]]]
        ],
        default_score: float = 0.5
    ) -> List[EntityResult]:
        """Remap keys for each batch item."""
        
        if isinstance(result, dict) and "entities" in result:
            result: List[Dict[str, Any]] = result["entities"]
        
        entities = [
            EntityResult(
                start=entity[self.start],  
                end=entity[self.end],
                label=self._get_label(entity),
                score=entity.get(self.score, default_score)
            )
            for entity in result if isinstance(entity, dict)
        ]
        return entities

@dataclass(frozen=True)
class Gliner2Mapper(BaseEntityMapper):
    """Maps GLiNER2's nested {label: {start, end, confidence}} format."""

    def map(
        self,
        result: Union[
            Dict[str, Dict[str, Any]],
            Dict[str, Dict[str, Dict[str, Any]]]
        ],
        default_score: float = 0.5
    ) -> List[EntityResult]:
        """Flatten nested structure per batch item."""
        
        if "entities" in result:
            result: Dict[str, Dict[str, Any]] = result["entities"]
            
        entities = [
            EntityResult(
                start=cast(int, entity[self.start]),
                end=cast(int, entity[self.end]),
                label=label_key,
                score=cast(float, entity.get(self.score, default_score))
            )
            for label_key, entity in result.items()
        ]
        return entities

# Pre-configured constants
GLINER_MAPPER = EntityMapper(label="label", score="score")
HF_NER_MAPPER = EntityMapper(label="entity_group", label_fallback="entity", score="score")
GLINER2_MAPPER = Gliner2Mapper(score="confidence")
OPENMED_MAPPER = EntityMapper(label="label", score="confidence")