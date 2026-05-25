import re

from spacy.tokens import Span

from .entity import Entity


def _valid_mac(span: Span) -> bool:
    """
    Validate MAC address format (48-bit address with hex groups).
    
    Supports three formats:
    - Colon-separated: 00:1A:2B:3C:4D:5E
    - Hyphen-separated: 00-1A-2B-3C-4D-5E
    - Cisco format (dot-separated groups of 4): 0012.3456.789A
    
    Rejects broadcast (FF:FF:FF:FF:FF:FF) and all-zeros (00:00:00:00:00:00) addresses.
    """
    mac_text = span.text.strip()
    
    # Remove separators and validate hex characters and length
    cleaned = re.sub(r'[:\-.]', '', mac_text)
    
    # Must be exactly 12 hex characters
    if not re.fullmatch(r"[0-9A-Fa-f]{12}", cleaned):
        return False
    
    # Reject broadcast (FF:FF:FF:FF:FF:FF) and all-zeros (00:00:00:00:00:00)
    if cleaned.upper() in ('FFFFFFFFFFFF', '000000000000'):
        return False
    
    return True


MAC_ADDRESS = Entity(
    label="MAC_ADDRESS",
    patterns=[
        # Colon or hyphen-separated: Uses backreference \1 to ensure consistent separator
        {"score": 0.75, "pattern": [{"TEXT": {"REGEX": r"\b[0-9A-Fa-f]{2}([:-])(?:[0-9A-Fa-f]{2}\1){4}[0-9A-Fa-f]{2}\b"}}]},
        # Cisco format: dot-separated groups of 4 hex digits
        {"score": 0.70, "pattern": [{"TEXT": {"REGEX": r"\b[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\.[0-9A-Fa-f]{4}\b"}}]},
    ],
    validator=_valid_mac,
    context_patterns=[
        {"pattern": [{"LEMMA": {"IN": ["mac", "mac-adres", "macaddress", "hardware", "physical"]}}]},
    ]
)