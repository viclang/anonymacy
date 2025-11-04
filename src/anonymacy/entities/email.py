from anonymacy.entities import Entity

EMAIL = Entity(
    label="EMAIL",
    patterns=[
        # Match tokens that look like email addresses
        {"score": 0.9, "pattern": [{"LIKE_EMAIL": True}]},
    ],
    context_patterns=[
        {"pattern": [{"LEMMA": "email"}]},
        {"pattern": [{"LEMMA": "mail"}]},
        {"pattern": [{"LEMMA": "address"}]},
        {"pattern": [{"LOWER": "e-mail"}]},
    ],
)
