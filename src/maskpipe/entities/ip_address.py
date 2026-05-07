import ipaddress

from spacy.tokens import Span

from .entity import Entity

def _valid_ip(span: Span) -> bool:
    try:
        ipaddress.ip_interface(span.text)
        return True
    except ValueError:
        return False

IPV4 = Entity(
    label="IPV4",
    patterns=[
        {"score": 0.6, "pattern": [{"TEXT": {"REGEX": r"\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"}}]},
    ],
    validator=_valid_ip,
    context_patterns=[
        {"pattern": [{"LEMMA": {"IN": ["ip", "ipv4", "ipv6"]}}]},
    ]
)

IPV6 = Entity(
    label="IPV6",
    patterns=[
        {"score": 0.6, "pattern": [{"TEXT": {"REGEX": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"}}]},
    ], 
    validator=_valid_ip,
    context_patterns=[
        {"pattern": [{"LEMMA": {"IN": ["ip", "ipv4", "ipv6"]}}]},
    ]
)
    