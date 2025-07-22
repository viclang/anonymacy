from typing import List, Set, Optional
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.pipeline import Pipe
from .base_recognizer import BaseRecognizer
import phonenumbers
from anonymacy.context_enhancer import ContextPattern

@Language.factory("phone_recognizer")
class PhoneRecognizer(BaseRecognizer):
    """spaCy pipeline component for recognizing phone numbers using phonenumbers library."""
    
    CONTEXT_PATTERNS = [
            {
                "label": "PHONE_NUMBER",
                "pattern": [ {"LEMMA": { "IN": ["telefoon", "mobiel", "telefoonnummer", "number"] } } ],
            }
        ]

    DEFAULT_SUPPORTED_REGIONS = { "NL", "US", "UK", "DE", "BE", "FR", "IT", "ES" }

    def __init__(
        self,
        nlp: Language,
        name: str = "phone_recognizer",
        context_patterns: Optional[List[ContextPattern]] = None,
        supported_regions: Set[str] = DEFAULT_SUPPORTED_REGIONS,
        leniency: int = 1,
        score: float = 0.4,
        style: str = "span",
        spans_key: str = "sc",
        allow_overlap: bool = True,
        conflict_strategy: str = "highest_confidence"
    ):
        """Initialize phone recognizer.
        
        Args:
            supported_regions: List of region codes (e.g., ["NL", "US", "UK"])
            leniency: Strictness level (0-3, where 0 is most lenient)
            score: Default confidence score for recognized numbers
            style: "span" or "ent" for output style
            spans_key: Key for doc.spans if style="span"
        """
        self.supported_regions = supported_regions or ["NL", "US", "UK", "DE", "BE", "FR", "IT", "ES"]
        self.leniency = leniency
        context_patterns = context_patterns if context_patterns else self.CONTEXT_PATTERNS

        super().__init__(
            nlp=nlp,
            name=name,            
            context_patterns=context_patterns,
            default_score=score,
            style=style,
            spans_key=spans_key,
            allow_overlap=allow_overlap,
            conflict_strategy=conflict_strategy
        )
            
    def _recognize(self, doc: Doc) -> List[Span]:
        """Find phone numbers in the document."""
        text = doc.text
        phone_spans = []
        
        # Try each region
        for region in self.supported_regions:
            for match in phonenumbers.PhoneNumberMatcher(text, region, leniency=self.leniency):
                # Convert character positions to token positions
                start_token = None
                end_token = None
                
                for token in doc:
                    if start_token is None and token.idx <= match.start < token.idx + len(token.text):
                        start_token = token.i
                    if token.idx < match.end <= token.idx + len(token.text):
                        end_token = token.i + 1
                        break
                
                if start_token is not None and end_token is not None:
                    span = Span(doc, start_token, end_token, label="PHONE_NUMBER")
                    span._.score = self.default_score
                    span._.source = self.name
                    
                    # Avoid duplicates
                    is_duplicate = any(
                        s.start == span.start and s.end == span.end 
                        for s in phone_spans
                    )
                    if not is_duplicate:
                        phone_spans.append(span)
        
        return phone_spans