"""
Integration of the simple anonymizer with the Anonymacy pipeline builder.
"""

from typing import Callable, Dict, List, Optional, Union
from spacy import Language
import spacy


class Pipeline:
    """Simple builder for creating anonymization pipelines."""
    
    def __init__(self, model: str = "nl_core_news_md"):
        """Initialize with a spaCy model."""
        self.nlp = spacy.load(model, disable=["ner"])
        self._patterns = []
        self._context_patterns = []
        self._anonymizers = {}
        
    def add_pattern(self, label: str, pattern: List[Dict], score: float = 1.0) -> 'Pipeline':
        """Add a pattern for entity recognition."""
        self._patterns.append({
            "label": label,
            "pattern": pattern,
            "score": score
        })
        return self
    
    def add_context(self, label: str, pattern: List[Dict]) -> 'Pipeline':
        """Add context pattern to boost confidence."""
        self._context_patterns.append({
            "label": label,
            "pattern": pattern
        })
        return self
    
    def add_anonymizer(self, label: str, value: Union[str, Callable[[str], str]]) -> 'Pipeline':
        """Add anonymization for a specific label.
        
        Args:
            label: Entity label
            value: Either a string for simple replacement or a function that takes text
        """
        self._anonymizers[label] = value
        return self
    
    def build(self) -> Language:
        """Build the complete pipeline."""
        # Add pattern recognizer
        if self._patterns:
            recognizer = self.nlp.add_pipe("pattern_recognizer", config={
                "spans_key": "sc",
                "allow_overlap": True
            })
            recognizer.add_patterns(self._patterns)
        
        # Add context enhancer
        if self._context_patterns:
            enhancer = self.nlp.add_pipe("context_enhancer", config={
                "spans_key": "sc"
            })
            enhancer.add_context_patterns(self._context_patterns)
        
        # Add conflict resolver
        self.nlp.add_pipe("conflict_resolver", config={
            "spans_key": "sc",
            "style": "span",
            "strategy": "highest_confidence"
        })
        
        # Add anonymizer
        if self._anonymizers:
            self.nlp.add_pipe("anonymizer", config={
                "spans_key": "sc",
                "operators": self._anonymizers
            })
        
        return self.nlp


# Preset pipelines
def create_dutch_pii_pipeline() -> Language:
    """Create a pre-configured Dutch PII detection and anonymization pipeline."""
    from anonymacy.helpers import DutchFaker, mask, redact
    
    pipeline = Pipeline("nl_core_news_md")
    
    # Common patterns
    pipeline.add_pattern("bsn", [{"LENGTH": 9, "IS_DIGIT": True}], 0.7)
    pipeline.add_pattern("email", [{"LIKE_EMAIL": True}], 0.9)
    pipeline.add_pattern("telefoon", [
        {"TEXT": {"REGEX": r"^(\+31|0)\d{9,10}$"}}
    ], 0.8)
    
    # Context boosters
    pipeline.add_context("bsn", [{"LOWER": "bsn"}])
    pipeline.add_context("bsn", [{"LOWER": "burgerservicenummer"}])
    pipeline.add_context("email", [{"LOWER": "email"}])
    pipeline.add_context("email", [{"LOWER": "e-mail"}])
    
    # Anonymizers - mix of strings and functions
    pipeline.add_anonymizer("persoon", DutchFaker.name)
    pipeline.add_anonymizer("bsn", mask(2, 2, "X"))
    pipeline.add_anonymizer("email", DutchFaker.email)
    pipeline.add_anonymizer("telefoon", "06-XXXXXXXX")  # Simple string
    pipeline.add_anonymizer("adres", "[ADRES VERWIJDERD]")  # Simple string
    
    return pipeline.build()