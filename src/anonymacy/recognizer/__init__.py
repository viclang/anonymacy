from .pattern_recognizer import PatternRecognizer, Pattern
from .bsn_recognizer import BsnRecognizer
from .phone_recognizer import PhoneRecognizer
from .base_recognizer import BaseRecognizer
from spacy.tokens import Span

if not Span.has_extension("score"):
    Span.set_extension("score", default=0.0)

__all__ = ['PatternRecognizer', 'Pattern', 'BsnRecognizer', 'PhoneRecognizer', 'BaseRecognizer']
