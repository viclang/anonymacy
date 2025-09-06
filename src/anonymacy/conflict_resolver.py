from typing import Optional
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.pipeline import Pipe
from anonymacy import span_filter
from typing import (
    Optional,
    Callable,
    Iterable
)

SpansFilterFunc = Callable[[Iterable[Span], Iterable[Span]], Iterable[Span]]

DEFAULT_RESOLVER_CONFIG = {
    "spans_key": "sc",
    "style": "ent",
    "spans_filter": {"@misc": "anonymacy.highest_confidence_filter.v1"},
    "threshold": 0.5
}

@Language.factory("conflict_resolver", assigns=["doc.ents"], default_config=DEFAULT_RESOLVER_CONFIG)
class ConflictResolver(Pipe):
    """Conflict resolver for overlapping spans that can be used standalone or as a pipeline component."""
    
    def __init__(
        self,
        nlp: Optional[Language] = None,
        name: str = "conflict_resolver",
        spans_key: str = "sc",
        style: str = "ent",
        spans_filter: SpansFilterFunc = span_filter.highest_confidence_filter,
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