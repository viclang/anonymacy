from spacy.tokens import Doc, Span

from . import span_filter
from .anonymizer import Anonymizer
from .conflict_resolver import ConflictResolver
from .context_enhancer import ContextEnhancer
from .doc_builder import DocBuilder
from .pipeline_builder import PipelineBuilder
from .recognizer import Recognizer
from .structured_analyzer import StructuredAnalyzer
from .entity_mapper import (
    BaseEntityMapper,
    EntityMapper,
    Gliner2Mapper,
    EntityResult,
    GLINER2_MAPPER,
    GLINER_MAPPER,
    HF_NER_MAPPER,
    OPENMED_MAPPER,
)
if not Doc.has_extension("masked"):
    Doc.set_extension("masked", default=None)

if not Doc.has_extension("context_words"):
    Doc.set_extension("context_words", default=[])

if not Span.has_extension("score"):
    Span.set_extension("score", default=0.0)

if not Span.has_extension("context"):
    Span.set_extension("context", default=[])

if not Span.has_extension("replacement"):
    Span.set_extension("replacement", default=None)

__all__ = [
    "span_filter",
    "Anonymizer",
    "ConflictResolver",
    "ContextEnhancer",
    "DocBuilder",
    "PipelineBuilder",
    "Recognizer",
    "StructuredAnalyzer",
    "BaseEntityMapper",
    "EntityMapper",
    "Gliner2Mapper",
    "EntityResult",
    "GLINER2_MAPPER",
    "GLINER_MAPPER",
    "HF_NER_MAPPER",
    "OPENMED_MAPPER",
]