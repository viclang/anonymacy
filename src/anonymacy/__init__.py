from .context_enhancer import ContextEnhancer
from .conflict_resolver import ConflictResolver
from .anonymizer import Anonymizer
from .recognizer import Recognizer
from . import validator
from . import span_filter
from spacy.tokens import Doc, Span

if not Span.has_extension("score"):
    Span.set_extension("score", default=0.0)

if not Span.has_extension("context"):
    Span.set_extension("context", default=[])

if not Doc.has_extension("anonymized"):
    Doc.set_extension("anonymized", default=None)

__all__ = [
    'ContextEnhancer',
    'ConflictResolver',
    'Anonymizer',
    'Recognizer',
    'registry',
    'validator',
    'span_filter'
]