from spacy.tokens import Span

from .entity import Entity


def _luhn_checksum(span: Span) -> bool:
    only_digits = ''.join(filter(str.isdigit, span.text))
    digits = [int(d) for d in only_digits]
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(int(dig) for dig in str(d * 2))
    return checksum % 10 == 0

CREDIT_CARD = Entity(
    label="CREDIT_CARD",
    patterns=[{
        "score": 0.6,
        "pattern": [{"TEXT": {"REGEX": r"\b3[47]\d{2}[- ]?\d{6}[- ]?\d{5}\b|\b(?:4\d{3}|5[1-5]\d{2}|6(?:011|22[1-9]\d|2[3-9]\d{2}|[4-9]\d{2}|5\d{2}))[- ]?\d{4}[- ]?\d{4}[- ]?(?:\d{4})?\b"}}],
    }],    
    validator=_luhn_checksum,
    context_patterns=[
        {"pattern": [{"LEMMA": {"IN": ["creditcard", "bankpas", "debetkaart", "betaalpas"]}}]},
        {"pattern": [{"LOWER": {"IN": ["creditcard", "bankpas", "debetkaart", "betaalpas"]}}]},
        {"pattern": [{"LEMMA": "kaart"}, {"LEMMA": {"IN": ["credit", "debet", "bank"]}}]},
        {"pattern": [{"LOWER": "kaart"}, {"LOWER": {"IN": ["credit", "debet", "bank"]}}]},
    ]
)