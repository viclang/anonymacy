from spacy.tokens import Span
from anonymacy.entities import Entity

def _elf_proef(span: Span) -> bool:
    """Validate BSN using the '11-proef' (elf proef) algorithm."""
    only_digits = ''.join(filter(str.isdigit, span.text))
    if len(set(only_digits)) == 1:
        return False

    if len(only_digits) == 8:
        only_digits = "0" + only_digits
        
    if len(only_digits) != 9:
        return False

    total = 0
    for char, factor in zip(only_digits, [9, 8, 7, 6, 5, 4, 3, 2, -1]):
        total += int(char) * factor

    return total % 11 == 0

BSN = Entity(
    label="BSN",
    patterns=[
        {"score": 0.5, "pattern": [{"LENGTH":  8, "IS_DIGIT": True}]},
        {"score": 0.5, "pattern": [{"LENGTH":  9, "IS_DIGIT": True}]},
        {"score": 0.4, "pattern": [{"SHAPE": {"IN": ["ddd.ddd.ddd", "ddd-ddd-ddd"]}}]},
        {"score": 0.4, "pattern": [{"SHAPE": "ddd"}, {"SHAPE": "ddd"}, {"SHAPE": "ddd"}]},
    ],
    validator=_elf_proef,
    context_patterns=[
        # --- Dutch: Official single tokens --- 
        {
            "pattern": [
                {"LEMMA": {"IN": ["bsn", "sofi", "sofinummer", "bsnnummer", "burgerservicenummer", "persoonsnummer"]}}
            ]
        },

        # --- Dutch: Fuzzy single tokens ---
        {"pattern": [{"LOWER": {"FUZZY": "burgerservicenummer"}}]},
        {"pattern": [{"LOWER": {"FUZZY1": "burgerservice"}}]},
        {"pattern": [{"LOWER": {"FUZZY1": "bsnnummer"}}]},
        {"pattern": [{"LOWER": {"FUZZY": "persoonsnummer"}}]},

        # --- Dutch: Multi-token phrases ---
        {"pattern": [{"LEMMA": "burger"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "service"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "nummer"}]},
        {"pattern": [{"LEMMA": "sociaal"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "fiscaal"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "nummer"}]},
        {"pattern": [{"LEMMA": "bsn"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "nummer"}]},
        {"pattern": [{"LEMMA": "sofi"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "nummer"}]},
        {"pattern": [{"LEMMA": "persoonlijk"}, {"TEXT": "-", "OP": "?"}, {"LEMMA": "nummer"}]},

        # --- English: Direct translations ---
        {"pattern": [{"LEMMA": "social"}, {"LEMMA": "security"}, {"LEMMA": "number"}]},
        {"pattern": [{"LEMMA": "citizen"}, {"LEMMA": "service"}, {"LEMMA": "number"}]},
        {"pattern": [{"LEMMA": "social"}, {"LEMMA": "fiscal"}, {"LEMMA": "number"}]},
        {"pattern": [{"LEMMA": "personal"}, {"LEMMA": "number"}]},
    ]
)