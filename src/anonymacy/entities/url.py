from anonymacy.entities import Entity

URL = Entity(
    label="URL",
    patterns=[
        # Match tokens that look like URLs
        {"score": 0.9, "pattern": [{"LIKE_URL": True}]},
    ],
    context_patterns=[
        {"pattern": [{"LEMMA": "url"}]},
        {"pattern": [{"LEMMA": "link"}]},
        {"pattern": [{"LEMMA": "website"}]},
        {"pattern": [{"LEMMA": "web"}]},
    ],
)
