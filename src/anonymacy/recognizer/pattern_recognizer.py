from typing import Any, Dict, List, Union
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher
from .base_recognizer import BaseRecognizer
from anonymacy.context_enhancer import ContextPattern

MAX_SCORE = 1.0
Pattern = Dict[str, Union[str, float, List[Dict[str, Any]]]]

@Language.factory("pattern_recognizer")
class PatternRecognizer(BaseRecognizer):
    def __init__(self,
        nlp: Language,
        name: str = "pattern_recognizer",
        patterns: List[Pattern] = None,
        context_patterns: List[ContextPattern] = None,
        default_score: float = 0.8,
        style: str = "span",
        spans_key: str = "sc",
        allow_overlap: bool = True,
        conflict_strategy: str = "highest_confidence"
    ):
        """Custom spaCy pipeline component for rule-based pattern recognition.

        Args:
            nlp (Language): The spaCy nlp object.
            name (str): The name of the pipeline component.
            patterns (List[Pattern]):
                List of patterns to match. Each pattern should be a dictionary with:
                - "label": The label for the matched span
                - "pattern": The pattern to match (list of dictionaries)
                - "score": Optional confidence score (default 1.0, capped at 1.0)
            style (str): Output style - "ent" for doc.ents or "span" for doc.spans[spans_key].
            spans_key (str): The spans key to save the spans under. Defaults to "sc".
            allow_overlap (bool): Whether to allow overlapping spans. If False, uses conflict_strategy to resolve.
            conflict_strategy (str): Strategy for handling conflicts when allow_overlap=False:
                - "keep_first": Keep existing spans, ignore new pattern matches
                - "keep_last": Replace overlapping spans with new pattern match  
                - "highest_confidence": Keep span with highest score. If scores are equal, keep longest span.
        """
        # Call parent's __init__
        super().__init__(
            nlp=nlp,
            name=name,            
            context_patterns=context_patterns,
            default_score=default_score,
            style=style,
            spans_key=spans_key,
            allow_overlap=allow_overlap,
            conflict_strategy=conflict_strategy
        )
        
        # Initialize matcher and patterns specific to this recognizer
        self.matcher = Matcher(nlp.vocab)
        
        if not patterns:
            patterns = []

        self._patterns = []
        for pattern in patterns:
            self._add_pattern(pattern)
    
    def _add_pattern(self, pattern: Pattern) -> None:
        """Add pattern for matching."""
        try:
            label = pattern["label"]
            score = min(pattern.get("score", self.default_score), MAX_SCORE)
            pattern_list = pattern["pattern"]
            
            pattern_id = f"{label}_{len(self._patterns)}"
            self.matcher.add(pattern_id, [pattern_list])
            self._patterns.append({"label": label, "pattern": pattern_list, "score": score})
            
        except KeyError as e:
            raise ValueError(
                f"Missing key {str(e)} in pattern {pattern}. Required: 'label', 'pattern'. Optional: 'score'."
            )

    def _recognize(self, doc: Doc) -> List[Span]:
        """Recognize spans using patterns. Must be implemented by subclasses."""
        matches = self.matcher(doc)
        if not matches:
            return []
        
        # Create spans from pattern matches
        new_spans = []
        for match_id, start, end in matches:
            match_str = self.nlp.vocab.strings[match_id]
            label, idx_str = match_str.rsplit("_", 1)
            idx = int(idx_str)
            
            span = Span(doc, start, end, label=label)
            span._.score = self._patterns[idx]["score"]
            new_spans.append(span)
        return new_spans
    
    def _invalidate_result(self, text: str, label: str = None) -> bool:
        """Override to add custom invalidation logic if needed."""
        return False