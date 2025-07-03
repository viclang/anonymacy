from typing import Any, Dict, List, Union, Tuple, Optional
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher
from spacy.pipeline import Pipe
from conflict_resolver import ConflictResolver

@Language.factory("pattern_recognizer")
class PatternRecognizer(Pipe):
    def __init__(self,
        nlp: Language,
        name: str = "pattern_recognizer",
        style: str = "span",
        spans_key: str = "sc",
        allow_overlap: bool = True,
        conflict_strategy: str = "highest_confidence"
    ):
        """Custom spaCy pipeline component for rule-based pattern recognition.

        Args:
            nlp (Language): The spaCy nlp object.
            name (str): The name of the pipeline component.
            style (str): Output style - "ent" for doc.ents or "span" for doc.spans[spans_key].
            spans_key (str): The spans key to save the spans under. Defaults to "sc".
            allow_overlap (bool): Whether to allow overlapping spans. If False, uses conflict_strategy to resolve.
            conflict_strategy (str): Strategy for handling conflicts when allow_overlap=False:
                - "keep_first": Keep existing spans, ignore new pattern matches
                - "keep_last": Replace overlapping spans with new pattern match  
                - "highest_confidence": Keep span with highest score. If scores are equal, keep longest span.
        """
        self.nlp = nlp
        self.name = name
        self.style = style
        self.spans_key = spans_key
        self.allow_overlap = allow_overlap
        self.conflict_strategy = conflict_strategy
        
        if self.style == "ent":
            # Overlap not allowed for entities
            self.allow_overlap = False  
        
        self.matcher = Matcher(nlp.vocab)
        self._patterns = []
        
        # Initialize conflict resolver if overlaps are not allowed
        if not self.allow_overlap:
            self.conflict_resolver = ConflictResolver(strategy=conflict_strategy)
        else:
            self.conflict_resolver = None

        if not Span.has_extension("score"):
            Span.set_extension("score", default=1.0)

    def add_patterns(self, patterns: List[Dict[str, Union[str, float, List[Dict[str, Any]]]]]):
        """Add patterns for matching."""
        
        for pattern in patterns:
            try:
                label = pattern["label"]
                score = min(pattern.get("score", 1.0), 1.0)  # Cap at 1.0
                pattern_list = pattern["pattern"]
                
                pattern_id = f"{label}_{len(self._patterns)}"
                self.matcher.add(pattern_id, [pattern_list])
                self._patterns.append({"label": label, "pattern": pattern_list, "score": score})
                
            except KeyError as e:
                raise ValueError(
                    f"Missing key {str(e)} in pattern {pattern}. Required: 'label', 'pattern'. Optional: 'score'."
                )

    def __call__(self, doc: Doc) -> Doc:
        """Process document and handle entity span conflicts."""
        matches = self.matcher(doc)
        if not matches:
            return doc

        # Get existing spans/ents
        existing_spans = self._get_existing_spans(doc)
        
        # Create spans from pattern matches
        new_spans = []
        for match_id, start, end in matches:
            match_str = self.nlp.vocab.strings[match_id]
            label, idx_str = match_str.rsplit("_", 1)
            idx = int(idx_str)
            
            span = Span(doc, start, end, label=label)
            span._.score = self._patterns[idx]["score"]
            new_spans.append(span)
        
        # Combine existing and new spans
        all_spans = existing_spans + new_spans
        
        # Handle based on overlap setting
        if not self.allow_overlap:
            all_spans = self.conflict_resolver.resolve(all_spans)
        
        # Assign to appropriate location
        if self.style == "ent":
            doc.ents = all_spans
        else:
            doc.spans[self.spans_key] = all_spans
        
        return doc

    def _get_existing_spans(self, doc: Doc) -> List[Span]:
        """Get existing spans from the appropriate source."""
        if self.style == "ent":
            spans = list(doc.ents)
        else:
            spans = list(doc.spans.get(self.spans_key, []))
        
        # Ensure all spans have scores
        for span in spans:
            if not hasattr(span._, "score") or span._.score is None:
                span._.score = 1.0
        
        return spans