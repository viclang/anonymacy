from typing import List, Dict, Any, Union, Callable, Optional, Tuple
from dataclasses import dataclass, replace
from copy import deepcopy
from spacy.tokens import Span, Doc

ReplacerFunc = Union[str, Callable[[], str], Callable[[str], str]]
CustomMatcherFunc = Callable[[Doc], List[Tuple[int, int, int]]]

@dataclass(frozen=True)
class Entity:
    """Configuration for a sensitive entity type."""
    
    label: str
    patterns: Optional[List[Union[str, List[Dict[str, Any]]]]] = None
    custom_matcher: Optional[CustomMatcherFunc] = None
    validator: Optional[Callable[[Span], bool]] = None
    context_patterns: Optional[List[Dict[str, Any]]] = None
    replacer: Optional[ReplacerFunc] = None

    def __post_init__(self):
        # Deepcopy lists to isolate references and prevent shared mutations
        if self.patterns and isinstance(self.patterns, list):
            object.__setattr__(self, 'patterns', deepcopy(self.patterns))
        if self.context_patterns and isinstance(self.context_patterns, list):
            object.__setattr__(self, 'context_patterns', deepcopy(self.context_patterns))

        if self.patterns:
            self._normalize_patterns(self.patterns)
        if self.context_patterns:
            self._normalize_patterns(self.context_patterns)

    def _normalize_patterns(self, patterns: List[Dict]):
        for pattern in patterns:
            pattern["label"] = self.label

    def replace(self, **changes: Any) -> "Entity":
        """Create a copy with modified attributes."""
        return replace(self, **changes)