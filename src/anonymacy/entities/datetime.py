from anonymacy.entities import Entity

# Day - Match only 1-2 digit days (not 3+ digits)
DD = r"(?:0?[1-9]|1[0-9]|2[0-9]|3[01])"
DD_ORDINAL = r"(?:0?[1-9](?:st|nd|rd|th|de|ste|e)?|[12][0-9](?:st|nd|rd|th|de|ste|e)?|3[01](?:st|nd|rd|th|de|ste|e)?)"

#  Month - Match only 1-2 digit months
MM = r"(?:0?[1-9]|1[0-2])"
MMM = [
    # NL
    "jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec",
    # EN
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
]
MMMM = [
    # NL
    "januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december",
    # EN
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"
]
MMMM_OR_MMM = MMM + MMMM

# Year
YY = r"(?:\d{2})"
YYYY = r"(?:19|20)\d{2}"

DATETIME = Entity(
    label="DATETIME",
    patterns=[
        # ISO 8601 formats
        {"score": 0.9, "pattern": [{"TEXT": {"REGEX": r"^(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z))$"}}]},

        # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
        # or DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        {"score": 0.85, "pattern": [{"TEXT": {"REGEX": {"IN": [
            rf"^{YYYY}\-{MM}\-{DD}$",
            rf"^{YYYY}\/{MM}\/{DD}$",
            rf"^{YYYY}\.{MM}\.{DD}$",
            rf"^{DD}\-{MM}\-{YYYY}$",
            rf"^{DD}\/{MM}\/{YYYY}$",
            rf"^{DD}\.{MM}\.{YYYY}$"
        ]}}}]},

        # DD MM YYYY
        {"score": 0.65, "pattern": [{"TEXT": {"REGEX": rf"^{DD}$"}}, {"TEXT": {"REGEX": rf"^{MM}$"}}, {"TEXT": {"REGEX": rf"^{YYYY}$"}}]},
        
        # DD MM YY
        {"score": 0.5, "pattern": [{"TEXT": {"REGEX": rf"^{DD}$"}}, {"TEXT": {"REGEX": rf"^{MM}$"}}, {"TEXT": {"REGEX": rf"^{YY}$"}}]},


        # DD-MM-YY or DD/MM/YY or DD.MM.YY
        {"score": 0.6, "pattern": [{"TEXT": {"REGEX": {"IN": [
            rf"^{DD}\-{MM}\-{YY}$",
            rf"^{DD}\/{MM}\/{YY}$",
            rf"^{DD}\.{MM}\.{YY}$"
        ]}}}]},

        # DD MMMM YYYY or DD MMM YYYY
        {"score": 0.75, "pattern": [
            {"TEXT": {"REGEX": rf"^{DD}$"}},
            {"LOWER": {"IN": MMMM_OR_MMM}},
            {"TEXT": {"REGEX": rf"^{YYYY}$"}}
        ]},

        # LIKE_NUM DD MMMM YYYY or LIKE_NUM DD MMM YYYY (ordinal support via spaCy)
        {"score": 0.75, "pattern": [
            {"LIKE_NUM": True},
            {"LOWER": {"IN": ["of", "van"]}, "OP": "?"},
            {"LOWER": {"IN": MMMM_OR_MMM}},
            {"TEXT": {"REGEX": rf"^{YYYY}$"}}
        ]},
        
        # MMMM DD, YYYY or MMM DD, YYYY
        {"score": 0.75, "pattern": [
            {"LOWER": {"IN": MMMM_OR_MMM}},
            {"TEXT": {"REGEX": rf"^{DD}$"}},
            {"ORTH": ",", "OP": "?"},
            {"TEXT": {"REGEX": rf"^{YYYY}$"}}
        ]},

        # MMMM LIKE_NUM, YYYY or MMM LIKE_NUM, YYYY (ordinal support via spaCy)
        {"score": 0.75, "pattern": [
            {"LOWER": {"IN": MMMM_OR_MMM}},
            {"LIKE_NUM": True},
            {"ORTH": ",", "OP": "?"},
            {"TEXT": {"REGEX": rf"^{YYYY}$"}}
        ]},
        
        # YYYY-MM-DD HH:MM:SS
        {"score": 0.85, "pattern": [
            {"SHAPE": "dddd-dd-dd"},
            {"SHAPE": "dd:dd:dd"}
        ]},
        
        # HH:MM:SS or HH:MM
        {"score": 0.6, "pattern": [{"SHAPE": "dd:dd:dd"}]},
        {"score": 0.5, "pattern": [{"SHAPE": "dd:dd"}]},
        
        # DD_ORDINAL (of/van) MMMM YYYY or DD_ORDINAL (of/van) MMM YYYY
        {"score": 0.75, "pattern": [
            {"TEXT": {"REGEX": rf"^{DD_ORDINAL}$"}},
            {"LOWER": {"IN": ["of", "van"]}, "OP": "?"},
            {"LOWER": {"IN": MMMM_OR_MMM}},
            {"TEXT": {"REGEX": rf"^{YYYY}$"}}
        ]},

        # MMMM DD_ORDINAL, YYYY or MMM DD_ORDINAL, YYYY
        {"score": 0.75, "pattern": [
            {"LOWER": {"IN": MMMM_OR_MMM}},
            {"TEXT": {"REGEX": rf"^{DD_ORDINAL}$"}},
            {"ORTH": ",", "OP": "?"},
            {"TEXT": {"REGEX": rf"^{YYYY}$"}}
        ]},

        # DD MMMM (without year)
        {"score": 0.6, "pattern": [
            {"TEXT": {"REGEX": rf"^{DD}$"}},
            {"LOWER": {"IN": MMMM_OR_MMM}}
        ]},

        # LIKE_NUM MMMM (without year)
        {"score": 0.6, "pattern": [
            {"LIKE_NUM": True},
            {"LOWER": {"IN": ["of", "van"]}, "OP": "?"},
            {"LOWER": {"IN": MMMM_OR_MMM}}
        ]},

        # DD_ORDINAL (of/van) MMMM (without year)
        {"score": 0.6, "pattern": [
            {"TEXT": {"REGEX": rf"^{DD_ORDINAL}$"}},
            {"LOWER": {"IN": ["of", "van"]}, "OP": "?"},
            {"LOWER": {"IN": MMMM_OR_MMM}}
        ]},

        # LIKE_NUM (of/van) MMMM (without year)
        {"score": 0.6, "pattern": [
            {"LIKE_NUM": True},
            {"LOWER": {"IN": ["of", "van"]}, "OP": "?"},
            {"LOWER": {"IN": MMMM_OR_MMM}}
        ]},

        # MMMM DD_ORDINAL (without year or comma)
        {"score": 0.6, "pattern": [
            {"LOWER": {"IN": MMMM_OR_MMM}},
            {"TEXT": {"REGEX": rf"^{DD_ORDINAL}$"}}
        ]},

        # MMMM LIKE_NUM (without year)
        {"score": 0.6, "pattern": [
            {"LOWER": {"IN": MMMM_OR_MMM}},
            {"LIKE_NUM": True}
        ]},
    ],
    context_patterns=[
        # Date/time context (required for loose patterns like DD MM YYYY)
        {"pattern": [{"LEMMA": {"IN": ["datum", "date", "op", "on"]}}]},
        {"pattern": [{"LEMMA": {"IN": ["tijd", "tijdstip", "time", "timestamp"]}}]},
        {"pattern": [{"LEMMA": {"IN": ["wanneer", "when"]}}]},
        
        # DATE_OF_BIRTH context patterns
        {"context_label": "DATE_OF_BIRTH", "pattern": [{"LEMMA": {"IN": ["geboortedatum", "geboorte"]}}]},
        {"context_label": "DATE_OF_BIRTH", "pattern": [{"LOWER": {"FUZZY": "geboortedatum"}}]},
        {"context_label": "DATE_OF_BIRTH", "pattern": [{"LEMMA": "geboren"}]},
        {"context_label": "DATE_OF_BIRTH", "pattern": [{"LEMMA": "datum"}, {"LEMMA": "van"}, {"LEMMA": "geboorte"}]},
        {"context_label": "DATE_OF_BIRTH", "pattern": [{"TEXT": {"IN": ["geb.", "Geb."]}}]},
        {"context_label": "DATE_OF_BIRTH", "pattern": [{"LEMMA": {"IN": ["birthday", "birthdate", "dob"]}}]},
        {"context_label": "DATE_OF_BIRTH", "pattern": [{"LEMMA": "date"}, {"LEMMA": "of"}, {"LEMMA": "birth"}]},
        {"context_label": "DATE_OF_BIRTH", "pattern": [{"LEMMA": "born"}]},
        {"context_label": "DATE_OF_BIRTH", "pattern": [{"TEXT": {"IN": ["DOB", "D.O.B."]}}]},
    ]
)
