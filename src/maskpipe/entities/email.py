from .entity import Entity

# Characters allowed in an unquoted local part atom (before/after dots)
EMAIL_LOCAL_ATOM = r"[a-zA-Z0-9!#$%&'*+/=?^_`{|}~-]+"

# Full local part: atoms separated by dots (prevents leading/trailing/consecutive dots)
EMAIL_LOCAL_PART = rf"{EMAIL_LOCAL_ATOM}(?:\.{EMAIL_LOCAL_ATOM})*"

# Single domain label: starts/ends with alphanumeric, hyphens only in middle
EMAIL_DOMAIN_LABEL = r"[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?"

# Full domain: labels separated by dots (subdomains + final label before TLD)
EMAIL_DOMAIN = rf"{EMAIL_DOMAIN_LABEL}(?:\.{EMAIL_DOMAIN_LABEL})*"

# Top-level domain: dot + at least 2 letters (filters noise like .c)
EMAIL_TLD = r"[a-zA-Z]{2,}"

# Complete email regex with word boundaries to avoid partial matches
EMAIL_REGEX = rf"\b{EMAIL_LOCAL_PART}@{EMAIL_DOMAIN}\.{EMAIL_TLD}\b"

EMAIL = Entity(
    label="EMAIL",
    patterns=[{"score": 0.7, "pattern": [{ "TEXT": { "REGEX": EMAIL_REGEX }}]}],
    context_patterns=[
        # ── Strong context (+0.3 → 1.0) ──
    
        # Unambiguous email nouns (pre-entity, within 5 tokens)
        {"score": 0.3, "pattern": [{"LOWER": {"IN": ["mail", "email", "e-mail", "mailadres", "e-mailadres"]}}]},
        
        # Verb + email noun (pre-entity, within 5 tokens)
        # LEMMA catches inflections; explicit email noun prevents bare-verb ambiguity
        {"score": 0.3, "pattern": [
            {"LEMMA": {"IN": ["send", "sturen", "write", "schrijven", "mailen"]}},
            {"LOWER": {"IN": ["me", "mij", "je", "jou", "u", "uw", "ons", "hem", "haar", "us", "you", "him", "her"]}, "OP": "?"},
            {"LOWER": {"IN": ["an", "een", "a"]}, "OP": "?"},
            {"LOWER": {"IN": ["email", "e-mail", "mailadres", "e-mailadres"]}}
        ]},

        {"score": 0.2, "pattern": [{"LEMMA": {"IN": ["schrijven", "sturen", "mailen", "bereiken"]}}]},
    ],
)
