from typing import List, Callable
from phonenumbers import PhoneNumberMatcher
from spacy.tokens import Doc, Span
from anonymacy.entities import Entity

def _phone_matcher(
    doc: Doc,
    supported_regions: List[str] = ["NL", "BE", "DE", "FR", "IT", "ES", "US", "UK"],
    leniency: int = 1,
    score: float = 0.4,
) -> set[tuple[int, int, float]]:
    matches = set()
    for region in supported_regions:
        for match in PhoneNumberMatcher(doc.text, region, leniency=leniency):
            try:
                span = doc.char_span(match.start, match.end)
                if span:
                    matches.add((span.start, span.end, score))                            
            except Exception:
                continue

    return matches

PHONE_NUMBER = Entity(
    label="PHONE_NUMBER",
    custom_matcher=_phone_matcher,
    context_patterns=[
        {"pattern": [{"LEMMA": {"IN": [
            "telefoon", "telefoonnummer", "mobiel", "mobieltje", "gsm",
            "nummer", "bellen", "app", "appen", "contact",
            "phone", "phonenumber", "mobile", "number",
            "call", "whatsapp", "signal", "telegram"
        ]}}]},
        {"pattern": [{"LOWER": {"FUZZY": "telefoonnummer"}}]},
        {"pattern": [{"LOWER": {"FUZZY": "phonenumber"}}]},
        {"pattern": [{"LOWER": {"FUZZY1": "whatsapp"}}]},
        {"pattern": [{"LEMMA": "telefoon"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "nummer"}]},
        {"pattern": [{"LEMMA": "mobiel"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "nummer"}]},
        {"pattern": [{"LEMMA": "phone"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "number"}]},
        {"pattern": [{"LEMMA": "cell"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "phone"}]},
    ],
)