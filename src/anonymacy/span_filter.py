from spacy import registry
import itertools
from spacy.tokens import Span
from typing import (
    Set,
    List,
    Iterable
)

@registry.misc("anonymacy.highest_confidence_filter.v1")
def make_highest_confidence_filter():
    return highest_confidence_filter

def highest_confidence_filter(*spans: Iterable["Span"]) -> List[Span]:
    """Filter overlapping spans by selecting the one with the highest confidence score.
    
    When spans overlap, the span with the highest score is preferred. If scores are
    equal, the longer span is preferred. If both score and length are equal, the
    span that starts first is preferred.
    
    Args:
        spans: Iterable of potentially overlapping spans with confidence scores
        
    Returns:
        List of non-overlapping spans
    """
    spans = itertools.chain(*spans)
    sorted_spans = sorted(
        spans,
        key=lambda span: (
            getattr(span._, "score", 0.0),
            span.end - span.start,
            -span.start
        ),
        reverse=True
    )
    
    result: List[Span] = []
    seen_tokens: Set[int] = set()
    
    for span in sorted_spans:
        # Check if any token in this span has already been claimed
        if all(token.i not in seen_tokens for token in span):
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
            
    return sorted(result, key=lambda s: s.start)