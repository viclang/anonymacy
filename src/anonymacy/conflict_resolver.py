from typing import Any, Dict, List, Union, Tuple, Optional
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher
from spacy.pipeline import Pipe
import itertools
from typing import (
    Any,
    Dict,
    Set,
    List,
    Optional,
    Callable,
    Iterable,
    Tuple,
    Union,
    TypedDict,
    Required,
    NotRequired,
)

def highest_confidence_filter(*spans: Iterable["Span"]) -> List[Span]:
    """Filter overlapping spans by selecting the one with the highest confidence score.
    
    When spans overlap, the span with the highest score is preferred. If scores are
    equal, the longer span is preferred. If both score and length are equal, the
    span that starts first is preferred.
    
    Args:
        spans: Iterable of potentially overlapping spans with confidence scores
        
    Returns:
        List of non-overlapping spans
    """
    spans = itertools.chain(*spans)
    sorted_spans = sorted(
        spans,
        key=lambda span: (
            getattr(span._, "score", 0.0),
            span.end - span.start,
            -span.start
        ),
        reverse=True
    )
    
    result: List[Span] = []
    seen_tokens: Set[int] = set()
    
    for span in sorted_spans:
        # Check if any token in this span has already been claimed
        if all(token.i not in seen_tokens for token in span):
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
            
    return sorted(result, key=lambda s: s.start)

SpansFilterFunc = Callable[[Iterable[Span], Iterable[Span]], Iterable[Span]]

@Language.factory("conflict_resolver")
class ConflictResolver(Pipe):
    """Conflict resolver for overlapping spans that can be used standalone or as a pipeline component."""
    
    def __init__(
        self,
        nlp: Optional[Language] = None,
        name: str = "conflict_resolver",
        spans_key: str = "sc",
        style: str = "ent",
        spans_filter: SpansFilterFunc = highest_confidence_filter,
        threshold: float = 0.5
    ):
        """Initialize the ConflictResolver.
        
        Args:
            nlp: The spaCy Language object (optional for standalone use)
            name: Name of the pipeline component
            strategy: Conflict resolution strategy. Options:
                - "keep_first": Keep first/existing spans
                - "keep_last": Keep last/newest spans  
                - "highest_confidence": Keep span with highest score. If equal, keep longest.
            spans_key: Key to read spans from doc.spans
            style: Where to write results. Options:
                - "span": Write to doc.spans[spans_key]
                - "ent": Write to doc.ents
            threshold: Minimum score threshold. Spans below this score are filtered out.
        """
        self.nlp = nlp
        self.name = name
        self.spans_key = spans_key
        self.style = style
        self.spans_filter = spans_filter
        self.threshold = threshold
    
    def __call__(self, doc: Doc) -> Doc:
        """Process document and resolve span conflicts (pipeline method)."""
        spans = list(doc.spans.get(self.spans_key, []))
        if not spans:
            return doc
        
        # Resolve conflicts
        resolved_spans = self.spans_filter(spans)
        
        # Apply threshold filtering
        if self.threshold > 0.0:
            resolved_spans = [
                span for span in resolved_spans 
                if getattr(span._, "score", 0.0) >= self.threshold
            ]
        
        # Output to specified target
        if self.style == "ent":
            doc.ents = resolved_spans
        else:
            doc.spans[self.spans_key] = resolved_spans
        
        return doc