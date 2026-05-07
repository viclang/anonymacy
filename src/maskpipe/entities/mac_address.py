from .entity import Entity

MAC_ADDRESS = Entity(
    label="MAC_ADDRESS",
    patterns=[
        {"score": 0.75, "pattern": [{"TEXT": {"REGEX": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"}}]},
    ],
    context_patterns=[
        {"pattern": [{"LEMMA": {"IN": ["mac", "mac-adres", "macaddress"]}}]},
    ]  
)