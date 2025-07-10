from typing import Dict, List, Union, Any
from spacy.tokens import Doc, Span
from spacy.pipeline import Pipe
from anonymacy import ConflictResolver
from abc import ABC, abstractmethod


ContextPattern = Dict[str, Union[str, bool, List[Dict[str, Any]]]]

class BaseRecognizer(Pipe, ABC):
    """Base class for custom spaCy recognizers with automatic deduplication.
    
    Example:
        class MyRecognizer(Recognizer):
            def _recognize(self, doc: Doc) -> List[Span]:
                spans = []
                # Add recognition logic
                for match in doc.matcher(doc):
                    span = doc[match[1]:match[2]]
                    span._.score = 0.9
                    spans.append(span)
                return spans
    """
    def __init__(
        self,
        nlp,
        name: str = "recognizer",
        context_patterns: List[ContextPattern] = None,
        default_score: float = 0.8,
        style: str = "span",
        spans_key: str = "sc",
        allow_overlap: bool = True,
        conflict_strategy: str = "highest_confidence"
    ):
        self.nlp = nlp
        self.name = name
        self.context_patterns = context_patterns if context_patterns else []
        self.default_score = default_score
        self.style = style
        self.spans_key = spans_key
        self.allow_overlap = allow_overlap
        
        if self.style == "ent":
            # Overlap not allowed for entities
            self.allow_overlap = False
        
        if not self.allow_overlap:
            self.conflict_resolver = ConflictResolver(strategy=conflict_strategy)
        else:
            self.conflict_resolver = None
        
        if not Span.has_extension("score"):
            Span.set_extension("score", default=0.0)
    
    def __call__(self, doc: Doc) -> Doc:
        """Process document with automatic deduplication."""

        new_spans = self._recognize(doc)
        if not new_spans:
            return doc

        existing_spans = (
            list(doc.ents) if self.style == "ent" 
            else list(doc.spans.get(self.spans_key, []))
        )
        
        # Build position index from existing spans
        spans_by_position = {}
        for span in existing_spans:
            if not hasattr(span._, "score") or span._.score is None:
                span._.score = self.default_score
            spans_by_position[(span.start, span.end)] = span

        for span in new_spans:
            if self._invalidate_result(span.text, span.label_):
                continue
            
            key = (span.start, span.end)
            print(f"Processing span: {span.text} with label: {span.label_} and score: {span._.score}")
            if key not in spans_by_position or spans_by_position[key]._.score < span._.score:
                spans_by_position[key] = span

        all_spans = list(spans_by_position.values())
        
        if not self.allow_overlap:
            all_spans = self.conflict_resolver.resolve(all_spans)

        if self.style == "ent":
            doc.ents = all_spans
        else:
            doc.spans[self.spans_key] = all_spans
        
        return doc

    @abstractmethod
    def _recognize(self, doc: Doc) -> List[Span]:
        """Recognize spans in the document. Must be implemented by subclasses."""
        pass
    
    def _invalidate_result(self, text: str, label: str = None) -> bool:
        """Check if a result should be invalidated based on custom logic."""
        return False