from typing import Any, Callable, Dict, List, Optional, Union
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.pipeline import Pipe


@Language.factory("anonymizer")
class Anonymizer(Pipe):
    """Simple spaCy pipeline component for anonymizing entities."""
    
    def __init__(
        self,
        nlp: Language,
        name: str = "anonymizer",
        style: str = "ent",
        spans_key: str = "sc",
        operators: Optional[Dict[str, Union[str, Callable[[str], str]]]] = None
    ):
        """Initialize the Anonymizer component.
        
        Args:
            nlp: The spaCy Language object
            name: Name of the component
            spans_key: Key to read spans from doc.spans
            style: Where to read entities from ("span" or "ent")
            operators: Dict mapping entity types to:
                - string: for simple replacement
                - function: that takes text and returns replacement
                Example: {"PERSON": "Jan Doe",
                         "EMAIL": lambda text: text[0] + "***@***.***"}
        """
        self.nlp = nlp
        self.name = name
        self.spans_key = spans_key
        self.style = style
        self.operators = operators or {}
        
        # Register Doc extensions
        if not Doc.has_extension("anonymized_text"):
            Doc.set_extension("anonymized_text", default=None)
        if not Doc.has_extension("anonymized_spans"):
            Doc.set_extension("anonymized_spans", default=[])
    
    def __call__(self, doc: Doc) -> Doc:
        """Process the document and anonymize entities."""
        # Get spans based on style
        spans = self._get_spans(doc)
        if not spans:
            doc._.anonymized_text = doc.text
            return doc
        
        # Sort spans by start position
        sorted_spans = sorted(spans, key=lambda s: s.start_char)
        
        # Build anonymized text
        anonymized_parts = []
        last_end = 0
        anonymized_spans = []
        
        for span in sorted_spans:
            # Add text before this span
            anonymized_parts.append(doc.text[last_end:span.start_char])
            
            # Get operator for this entity type
            operator = self.operators.get(span.label_)
            
            if operator:
                # Handle both string and function operators
                if isinstance(operator, str):
                    anonymized_value = operator
                else:
                    # Pass just the text, not the whole span
                    anonymized_value = operator(span.text)
            else:
                # Default: replace with entity type
                anonymized_value = f"<{span.label_}>"
            
            # Add anonymized value
            anonymized_parts.append(anonymized_value)
            
            # Track anonymized spans
            anonymized_span_info = {
                "start": span.start_char,
                "end": span.end_char,
                "text": span.text,
                "label": span.label_,
                "anonymized": anonymized_value
            }
            anonymized_spans.append(anonymized_span_info)
            
            last_end = span.end_char
        
        # Add remaining text
        anonymized_parts.append(doc.text[last_end:])
        
        # Store results
        doc._.anonymized_text = "".join(anonymized_parts)
        doc._.anonymized_spans = anonymized_spans
        
        return doc
    
    def _get_spans(self, doc: Doc) -> List[Span]:
        """Get spans from document based on style setting."""
        if self.style == "ent":
            return list(doc.ents) if doc.ents else []
        return list(doc.spans.get(self.spans_key, []))
    
    def set_operators(self, operators: Dict[str, Union[str, Callable[[str], str]]]) -> None:
        """Update operators."""
        self.operators = operators