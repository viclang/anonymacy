from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TypedDict,
    Required,
    NotRequired,
)
from collections.abc import Sequence
from dataclasses import dataclass, replace, MISSING
from spacy.tokens import Span, Doc

RedactorFunc = Union[str, Callable[[], str], Callable[[str], str]]
CustomMatcherFunc = Callable[[Doc], List[Tuple[int, int, float]]]

class Pattern(TypedDict):
    pattern: Required[Union[str, List[Dict[str, Any]]]]
    score: NotRequired[float]
    id: NotRequired[str]

class ContextPattern(TypedDict):
    pattern: Required[Union[str, List[Dict[str, Any]]]]
    context_label: NotRequired[str]

@dataclass(frozen=True)
class Entity:
    """Configuration for a sensitive entity type."""
    
    label: str
    patterns: Optional[Sequence[Pattern]] = None
    custom_matcher: Optional[CustomMatcherFunc] = None
    validator: Optional[Callable[[Span], bool]] = None
    context_patterns: Optional[Sequence[ContextPattern]] = None
    redactor: Optional[RedactorFunc] = None

    def __post_init__(self):
        """Convert sequences to tuples for immutability."""
        if self.patterns and not isinstance(self.patterns, tuple):
            object.__setattr__(self, 'patterns', tuple(self.patterns))
        if self.context_patterns and not isinstance(self.context_patterns, tuple):
            object.__setattr__(self, 'context_patterns', tuple(self.context_patterns))

    def replace(self, **changes: Any) -> "Entity":
        """Create a copy with modified attributes."""
        return replace(self, **changes)