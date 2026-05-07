from .entity import Entity

CREDIT_CARD = Entity(
    label="CREDIT_CARD",
    patterns=[{
        "score": 0.6,
        "pattern": [{"TEXT": {"REGEX": r"\b3[47]\d{2}[- ]?\d{6}[- ]?\d{5}\b|\b(?:4\d{3}|5[1-5]\d{2}|6(?:011|22[1-9]\d|2[3-9]\d{2}|[4-9]\d{2}|5\d{2}))[- ]?\d{4}[- ]?\d{4}[- ]?(?:\d{4})?\b"}}],
    }],
    context_patterns=[
        {"pattern": [{"LEMMA": {"IN": ["creditcard", "bankpas", "debetkaart", "betaalpas"]}}]},
        {"pattern": [{"LOWER": {"IN": ["creditcard", "bankpas", "debetkaart", "betaalpas"]}}]},
        {"pattern": [{"LEMMA": "kaart"}, {"LEMMA": {"IN": ["credit", "debet", "bank"]}}]},
        {"pattern": [{"LOWER": "kaart"}, {"LOWER": {"IN": ["credit", "debet", "bank"]}}]},
    ]
)