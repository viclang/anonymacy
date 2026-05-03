from . import span_filter
from .context_enhancer import ContextEnhancer
from .conflict_resolver import ConflictResolver
from .anonymizer import Anonymizer
from .recognizer import Recognizer
from .doc_builder import DocBuilder
from .pipeline_builder import PipelineBuilder
from spacy.tokens import Doc, Span

if not Span.has_extension("score"):
    Span.set_extension("score", default=0.0)

if not Span.has_extension("context"):
    Span.set_extension("context", default=[])

if not Doc.has_extension("anonymized"):
    Doc.set_extension("anonymized", default=None)

if not Doc.has_extension("context_words"):
    Doc.set_extension("context_words", default=[])

__all__ = [
    "span_filter",
    "Anonymizer",
    "ConflictResolver",
    "ContextEnhancer",
    "DocBuilder",
    "PipelineBuilder",
    "Recognizer",
]