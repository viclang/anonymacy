from typing import List, Tuple

from phonenumbers import PhoneNumberMatcher as Matcher
from spacy.tokens import Doc

from .entity import Entity

class PhoneNumberMatcher():
    """Custom matcher for phone numbers using the phonenumbers library."""
    
    def __init__(self,
        regions: List[str]= ["NL", "BE", "DE", "FR", "IT", "ES", "US", "UK"],
        leniency: int = 1,
        score: float = 0.4
    ):
        self.regions = regions
        self.leniency = leniency
        self.score = score
    
    def __call__(self, doc: Doc) -> List[Tuple[int, int, float]]:
        matches = []
        for region in self.regions:
            for match in Matcher(text=doc.text, region=region, leniency=self.leniency):
                try:
                    span = doc.char_span(match.start, match.end)
                    if span:
                        matches.append((span.start, span.end, self.score))                            
                except Exception:
                    continue
        return matches 

PHONE_NUMBER = Entity(
    label="PHONE_NUMBER",
    custom_matcher=PhoneNumberMatcher(),
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