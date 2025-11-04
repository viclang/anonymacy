from schwifty import IBAN
from schwifty.exceptions import InvalidChecksumDigits
from anonymacy.entities import Entity

def _iban_validator(iban) -> bool:
                # Returns True if the IBAN is valid, False otherwise
                try:
                    iban = IBAN(iban)
                    return True
                except InvalidChecksumDigits:
                    return False

IBAN = Entity(
    label="IBAN",
    patterns=[{
        "score": 0.5,
        "pattern": [
            {
                "REGEX": r"""                              # https://github.com/microsoft/presidio/blob/main/presidio-analyzer/presidio_analyzer/predefined_recognizers/generic/iban_recognizer.py
                    ([A-Z]{2}[ \-]?[0-9]{2})               # country + check digits
                    (?=(?:[ \-]?[A-Z0-9]){9,30})           # look-ahead: ≥9 alphanumerics ahead
                    ((?:[ \-]?[A-Z0-9]{3,5}){2})           # 2 mandatory blocks of 3-5 chars
                    ([ \-]?[A-Z0-9]{3,5})?                 # optional blocks (up to 5×)
                    ([ \-]?[A-Z0-9]{3,5})?
                    ([ \-]?[A-Z0-9]{3,5})?
                    ([ \-]?[A-Z0-9]{3,5})?
                    ([ \-]?[A-Z0-9]{3,5})?
                    ([ \-]?[A-Z0-9]{1,3})?                   # final 1-3 chars tail
                """
            }
        ],
    }],
    validator=_iban_validator,
    context_patterns=[
        # --- Dutch: Official single tokens --- 
        {"pattern": [{"LEMMA": {"IN": [
            "iban", "bankrekeningnummer", "internationaal", "bankrekening"]}}]},

        # --- Dutch: Fuzzy single tokens ---
        {"pattern": [{"LOWER": {"FUZZY": "bankrekeningnummer"}}]},
        {"pattern": [{"LOWER": {"FUZZY1": "iban"}}]},
        {"pattern": [{"LOWER": {"FUZZY1": "bankrekening"}}]},

        # --- Dutch: Multi-token phrases ---
        {"pattern": [{"LEMMA": "internationaal"}, {"LEMMA": "bankrekeningnummer"}]},
        {"pattern": [{"LEMMA": "iban"}, {"LEMMA": "nummer"}]},
        {"pattern": [{"LEMMA": "bank"}, {"LEMMA": "rekeningnummer"}]},

        # --- English: Direct translations ---
        {"pattern": [{"LEMMA": "international"}, {"LEMMA": "bank"}, {"LEMMA": "account"}, {"LEMMA": "number"}]},
        {"pattern": [{"LEMMA": "iban"}, {"LEMMA": "number"}]},
        {"pattern": [{"LEMMA": "bank"}, {"LEMMA": "account"}, {"LEMMA": "number"}]},
    ]
)