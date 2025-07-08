from typing import Any, Callable, Dict, List, Optional, Union
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.pipeline import Pipe
import logging

logger = logging.getLogger("anonymizer")

NoArgOperator = Callable[[], str]
TextOperator = Callable[[str], str]

@Language.factory("anonymizer")
class Anonymizer(Pipe):
    """Simple spaCy pipeline component for anonymizing entities."""
    
    def __init__(
        self,
        nlp: Language,
        name: str = "anonymizer",
        style: str = "ent",
        spans_key: str = "sc"
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
        self.operators: Dict[str, Union[str, NoArgOperator, TextOperator]] = {}
        
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
            
            default_anonymized_value = f"[{span.label_.upper()}]"
            operator = self.operators.get(span.label_)
            
            if operator:
                # Handle both string and function operators
                if isinstance(operator, str):
                    anonymized_value = operator
                else:
                    try:
                        argcount = operator.__code__.co_argcount
                        # Handle bound methods (like fake.name)
                        if hasattr(operator, '__self__'):
                            argcount -= 1  # Subtract 'self' parameter
                        
                        if argcount == 0:
                            anonymized_value = operator()
                        else:
                            anonymized_value = operator(span.text)
                    except AttributeError:
                        try:
                            anonymized_value = str(operator(span.text))
                        except Exception as e:
                            # Log warning for truly unsupported cases
                            op_name = getattr(operator, '__name__', repr(operator))
                            logger.warning(
                                f"Operator for '{span.label_}' ({op_name}) failed: {e}. "
                                f"This operator cannot be called with text. "
                                f"Please wrap it in a lambda if needed."
                            )
                            # Use entity type as fallback
                            anonymized_value = default_anonymized_value
            else:
                # Default: replace with entity type
                anonymized_value = default_anonymized_value
            
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
    
    def add_operators(self, operators: Dict[str, Union[str, NoArgOperator, TextOperator]]) -> None:
        """Update operators."""
        self.operators = operators