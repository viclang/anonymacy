from spacy.tokens import Doc, Span

from . import span_filter
from .anonymizer import Anonymizer
from .conflict_resolver import ConflictResolver
from .context_enhancer import ContextEnhancer
from .doc_builder import DocBuilder
from .pipeline_builder import PipelineBuilder
from .recognizer import Recognizer

if not Doc.has_extension("anonymized"):
    Doc.set_extension("anonymized", default=None)

if not Doc.has_extension("context_words"):
    Doc.set_extension("context_words", default=[])

if not Span.has_extension("score"):
    Span.set_extension("score", default=0.0)

if not Span.has_extension("context"):
    Span.set_extension("context", default=[])

__all__ = [
    "span_filter",
    "Anonymizer",
    "ConflictResolver",
    "ContextEnhancer",
    "DocBuilder",
    "PipelineBuilder",
    "Recognizer",
]