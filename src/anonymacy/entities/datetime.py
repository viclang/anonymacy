from anonymacy.entities import Entity

DATETIME = Entity(
    label="DATETIME",
    patterns=[
        {"score": 0.8, "pattern": [{"REGEX": r"(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d\.\d+([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z))|(\d{4}-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d([+-][0-2]\d:[0-5]\d|Z)"}]},
        
        # ISO format: YYYY-MM-DD (International)
        {"score": 0.7, "pattern": [{"SHAPE": "dddd-dd-dd"}]},
        
        # --- Dutch date formats ---
        # DD-MM-YYYY or DD/MM/YYYY or DD.MM.YYYY
        {"score": 0.6, "pattern": [{"SHAPE": "dd-dd-dddd"}]},
        {"score": 0.6, "pattern": [{"SHAPE": "dd/dd/dddd"}]},
        {"score": 0.6, "pattern": [{"SHAPE": "dd.dd.dddd"}]},

        # Short format: DD-MM-YY or DD/MM/YY or DD.MM.YY
        {"score": 0.5, "pattern": [{"SHAPE": "dd-dd-dd"}]},
        {"score": 0.5, "pattern": [{"SHAPE": "dd/dd/dd"}]},
        {"score": 0.5, "pattern": [{"SHAPE": "dd.dd.dd"}]},

        # Dutch month names: DD maand YYYY
        {"score": 0.7, "pattern": [
            {"IS_DIGIT": True, "LENGTH": {"IN": [1, 2]}},
            {"LOWER": {"IN": ["januari", "februari", "maart", "april", "mei", "juni",
                              "juli", "augustus", "september", "oktober", "november", "december",
                              "jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]}},
            {"IS_DIGIT": True, "LENGTH": {"IN": [2, 4]}}
        ]},
        
        # --- English date formats ---
        # month DD, YYYY
        {"score": 0.7, "pattern": [
            {"LOWER": {"IN": ["january", "february", "march", "april", "may", "june",
                              "july", "august", "september", "october", "november", "december",
                              "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]}},
            {"IS_DIGIT": True, "LENGTH": {"IN": [1, 2]}},
            {"ORTH": ",", "OP": "?"},
            {"IS_DIGIT": True, "LENGTH": {"IN": [2, 4]}}
        ]},
        
        # DD month YYYY
        {"score": 0.7, "pattern": [
            {"IS_DIGIT": True, "LENGTH": {"IN": [1, 2]}},
            {"LOWER": {"IN": ["january", "february", "march", "april", "may", "june",
                              "july", "august", "september", "october", "november", "december",
                              "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]}},
            {"IS_DIGIT": True, "LENGTH": {"IN": [2, 4]}}
        ]},
        
        # YYYY-MM-DD HH:MM:SS
        {"score": 0.8, "pattern": [
            {"SHAPE": "dddd-dd-dd"},
            {"SHAPE": "dd:dd:dd"}
        ]},
        
        # --- Time only patterns ---
        {"score": 0.4, "pattern": [{"SHAPE": "dd:dd"}]},
        {"score": 0.5, "pattern": [{"SHAPE": "dd:dd:dd"}]},
    ],
    context_patterns=[
        {"pattern": [{"LEMMA": "datum"}]},
        {"pattern": [{"LEMMA": "date"}]},
        {"pattern": [{"LEMMA": {"IN": ["tijd", "tijdstip", "time", "timestamp"]}}]},
        
        # Invalidation patterns to reduce false positives
        {"pattern": [{"LEMMA": {"IN": ["geboortedatum", "geboorte"]}}], "invalidate": True},
        {"pattern": [{"LOWER": {"FUZZY": "geboortedatum"}}], "invalidate": True},
        {"pattern": [{"LEMMA": "geboren"}], "invalidate": True},
        {"pattern": [{"LEMMA": "datum"}, {"LEMMA": "van"}, {"LEMMA": "geboorte"}], "invalidate": True},
        {"pattern": [{"TEXT": {"IN": ["geb.", "Geb."]}}], "invalidate": True},
        {"pattern": [{"LEMMA": {"IN": ["birthday", "birthdate", "dob"]}}], "invalidate": True},
        {"pattern": [{"LEMMA": "date"}, {"LEMMA": "of"}, {"LEMMA": "birth"}], "invalidate": True},
        {"pattern": [{"LEMMA": "born"}], "invalidate": True},
        {"pattern": [{"TEXT": {"IN": ["DOB", "D.O.B."]}}], "invalidate": True},
    ],
)
