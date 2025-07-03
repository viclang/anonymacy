from typing import Any, Dict, List, Union, Tuple, Optional
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher
from spacy.pipeline import Pipe

@Language.factory("conflict_resolver")
class ConflictResolver(Pipe):
    """Conflict resolver for overlapping spans that can be used standalone or as a pipeline component."""
    
    def __init__(
        self,
        nlp: Optional[Language] = None,
        name: str = "conflict_resolver",
        strategy: str = "highest_confidence",
        spans_key: str = "sc",
        style: str = "ent",
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
        self.threshold = threshold
        
        valid_strategies = {"keep_first", "keep_last", "highest_confidence"}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")
        self.strategy = strategy
        
        valid_outputs = {"span", "ent"}
        if style not in valid_outputs:
            raise ValueError(f"Invalid style: {style}. Must be one of {valid_outputs}")
        self.style = style
    
    def resolve(self, spans: List[Span]) -> List[Span]:
        """Resolve conflicts between overlapping spans (standalone method).
        
        Args:
            spans: List of potentially overlapping spans
            
        Returns:
            List of non-overlapping spans based on the resolution strategy
        """
        if not spans or len(spans) <= 1:
            return spans
        
        # Sort spans by start position
        sorted_spans = sorted(spans, key=lambda s: (s.start, s.end))
        
        # Group overlapping spans
        groups = self._group_overlapping_spans(sorted_spans)
        
        # Resolve each group
        resolved = []
        for group in groups:
            if len(group) == 1:
                resolved.append(group[0])
            else:
                winner = self._select_winner(group)
                resolved.append(winner)
        
        return sorted(resolved, key=lambda s: (s.start, s.end))
    
    def __call__(self, doc: Doc) -> Doc:
        """Process document and resolve span conflicts (pipeline method)."""
        spans = list(doc.spans.get(self.spans_key, []))
        if not spans:
            return doc
        
        # Resolve conflicts
        resolved_spans = self.resolve(spans)
        
        # Apply threshold filtering
        if self.threshold > 0.0:
            resolved_spans = [
                span for span in resolved_spans 
                if getattr(span._, "score", 1.0) >= self.threshold
            ]
        
        # Output to specified target
        if self.style == "ent":
            doc.ents = resolved_spans
        else:
            doc.spans[self.spans_key] = resolved_spans
        
        return doc
    
    def _group_overlapping_spans(self, spans: List[Span]) -> List[List[Span]]:
        """Group spans that overlap with each other."""
        groups = []
        current_group = [spans[0]]
        
        for span in spans[1:]:
            # Check if span overlaps with any span in current group
            overlaps = False
            for group_span in current_group:
                if span.start < group_span.end and group_span.start < span.end:
                    overlaps = True
                    break
            
            if overlaps:
                current_group.append(span)
            else:
                groups.append(current_group)
                current_group = [span]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _select_winner(self, group: List[Span]) -> Span:
        """Select the winning span from a group of overlapping spans."""
        if self.strategy == "keep_first":
            # Return the first span in the list (by processing order)
            return group[0]
        
        elif self.strategy == "keep_last":
            # Return the last added span (assumes order is preserved)
            return group[-1]
        
        elif self.strategy == "highest_confidence":
            # Sort by score (descending), then by length (descending)
            return max(group, key=lambda s: (
                getattr(s._, "score", 1.0),
                s.end - s.start
            ))