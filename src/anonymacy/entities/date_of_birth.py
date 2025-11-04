from anonymacy_builder.entities import DATETIME

DATE_OF_BIRTH = DATETIME.replace(
    label="DATE_OF_BIRTH",
    context_patterns=[
        # --- Dutch context patterns ---
        {"pattern": [{"LEMMA": {"IN": ["geboortedatum", "geboorte"]}}]},
        {"pattern": [{"LOWER": {"FUZZY": "geboortedatum"}}]},
        {"pattern": [{"LEMMA": "geboren"}]},
        {"pattern": [{"LEMMA": "datum"}, {"LEMMA": "van"}, {"LEMMA": "geboorte"}]},
        {"pattern": [{"TEXT": {"IN": ["geb.", "Geb."]}}]},
        
        # --- English context patterns ---
        {"pattern": [{"LEMMA": {"IN": ["birthday", "birthdate", "dob"]}}]},
        {"pattern": [{"LEMMA": "date"}, {"LEMMA": "of"}, {"LEMMA": "birth"}]},
        {"pattern": [{"LEMMA": "born"}]},
        {"pattern": [{"TEXT": {"IN": ["DOB", "D.O.B."]}}]},
    ]
)