from anonymacy.entities import Entity

NUMBER = Entity(
    label="NUMBER",
    patterns=[
        # Match tokens that look like numbers
        {"score": 0.9, "pattern": [{"LIKE_NUM": True}]},
    ],
    context_patterns=[
        {"pattern": [{"LEMMA": "number"}]},
        {"pattern": [{"LEMMA": "aantal"}]},
        {"pattern": [{"LEMMA": "count"}]},
        {"pattern": [{"LEMMA": "quantity"}]},
    ],
)
