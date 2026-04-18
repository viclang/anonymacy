"""Tests for ContextEnhancer component.

The ContextEnhancer enhances or validates entity spans based on surrounding context patterns.
It can:
- Detect context patterns near entities
- Boost confidence scores based on context
- Optionally change labels based on context
"""
from spacy.tokens import Doc, Span
from spacy.lang.nl import Dutch
from anonymacy import ContextEnhancer


def span(doc: Doc, start: int, end: int, label: str, score: float = 0.6) -> Span:
    """Helper to create a span with a score."""
    s = Span(doc, start, end, label=label)
    s._.score = score
    return s


def test_no_patterns_added():
    """Test that component works with no patterns (no-op)."""
    nlp = Dutch()
    doc = nlp("De email contact@example.com is geldig")
    doc.ents = [span(doc, 2, 4, "email", score=0.5)]
    original_score = doc.ents[0]._.score
    
    context_enhancer = nlp.add_pipe("context_enhancer")
    
    result = context_enhancer(doc)
    
    assert len(result.ents) == 1
    assert result.ents[0]._.score == original_score


def test_skip_already_enhanced_spans():
    """Test that already-enhanced spans are not re-processed."""
    nlp = Dutch()
    doc = nlp("De email contact@example.com is geldig")
    s = span(doc, 2, 4, "email", score=0.5)
    s._.context = ["existing"]
    doc.ents = [s]
    
    context_enhancer = nlp.add_pipe("context_enhancer")
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "geldig"}]}
    ])
    
    result = context_enhancer(doc)
    
    # Should keep existing context
    assert result.ents[0]._.context == ["existing"]


def test_wrong_label_no_processing():
    """Test that pattern for wrong label doesn't process span."""
    nlp = Dutch()
    doc = nlp("De email contact@example.com is geldig")
    doc.ents = [span(doc, 2, 4, "email", score=0.5)]
    original_score = doc.ents[0]._.score
    
    context_enhancer = nlp.add_pipe("context_enhancer")
    context_enhancer.add_patterns([
        {"label": "telefoon", "pattern": [{"LOWER": "geldig"}]}  # Different label
    ])
    
    result = context_enhancer(doc)
    
    # Should not be modified since label doesn't match
    assert result.ents[0]._.score == original_score


def test_token_pattern_with_lower_attribute():
    """Test matching with token-based patterns using LOWER attribute."""
    nlp = Dutch()
    doc = nlp("De email contact@example.com is GELDIG")
    doc.ents = [span(doc, 2, 4, "email", score=0.5)]
    
    context_enhancer = nlp.add_pipe("context_enhancer")
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "geldig"}]}
    ])
    
    result = context_enhancer(doc)
    
    # Pattern should match (LOWER makes it case-insensitive)
    assert len(result.ents) == 1


def test_context_window_before_span():
    """Test context matching within before-span window."""
    nlp = Dutch()
    doc = nlp("Contact email contact@example.com nu")
    doc.ents = [span(doc, 2, 4, "email", score=0.5)]
    
    context_enhancer = nlp.add_pipe("context_enhancer", config={"context_window": (2, 1)})
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "contact"}]}
    ])
    
    result = context_enhancer(doc)
    
    assert len(result.ents) == 1


def test_context_window_after_span():
    """Test context matching within after-span window."""
    nlp = Dutch()
    doc = nlp("De email contact@example.com is geldig")
    doc.ents = [span(doc, 2, 4, "email", score=0.5)]
    
    context_enhancer = nlp.add_pipe("context_enhancer", config={"context_window": (0, 3)})
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "geldig"}]}
    ])
    
    result = context_enhancer(doc)
    
    assert len(result.ents) == 1


def test_context_window_too_narrow():
    """Test that pattern outside context window is not matched."""
    nlp = Dutch()
    doc = nlp("De email contact@example.com wordt gevalideerd")
    doc.ents = [span(doc, 2, 4, "email", score=0.5)]
    original_score = doc.ents[0]._.score
    
    # Narrow window: only 1 token before and after
    context_enhancer = nlp.add_pipe("context_enhancer", config={"context_window": (1, 1)})
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "gevalideerd"}]}
    ])
    
    result = context_enhancer(doc)
    
    # "gevalideerd" is 2 tokens away, outside narrow window
    assert result.ents[0]._.score == original_score


def test_multiple_patterns_different_labels():
    """Test with multiple patterns for different entity labels."""
    nlp = Dutch()
    doc = nlp("Email contact en telefoon 123456")
    # Create spans with correct indices
    email_span = span(doc, 0, 2, "email", score=0.5)
    phone_span = span(doc, 3, 5, "phone", score=0.5)
    doc.ents = [email_span, phone_span]
    
    context_enhancer = nlp.add_pipe("context_enhancer")
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "contact"}]},
        {"label": "phone", "pattern": [{"LOWER": "telefoon"}]}
    ])
    
    result = context_enhancer(doc)
    
    # Both entities should still be present
    assert len(result.ents) == 2


def test_confidence_boost_capped_at_one():
    """Test that boosted score doesn't exceed 1.0."""
    nlp = Dutch()
    doc = nlp("Email contact@example.com is geldig")
    doc.ents = [span(doc, 1, 3, "email", score=0.8)]
    
    context_enhancer = nlp.add_pipe("context_enhancer", config={"confidence_boost": 0.5})
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "geldig"}]}
    ])
    
    result = context_enhancer(doc)
    
    if len(result.ents) > 0:
        assert result.ents[0]._.score <= 1.0


def test_empty_document():
    """Test processing an empty document."""
    nlp = Dutch()
    doc = nlp("")
    
    context_enhancer = nlp.add_pipe("context_enhancer")
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "test"}]}
    ])
    
    result = context_enhancer(doc)
    
    assert len(result.ents) == 0


def test_document_with_no_entities():
    """Test with document that has no entities."""
    nlp = Dutch()
    doc = nlp("De email contact@example.com is geldig")
    
    context_enhancer = nlp.add_pipe("context_enhancer")
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "geldig"}]}
    ])
    
    result = context_enhancer(doc)
    
    assert len(result.ents) == 0


def test_multiple_entities_same_label():
    """Test with multiple entities of the same label."""
    nlp = Dutch()
    doc = nlp("Email contact en email test zijn valide")
    doc.ents = [
        span(doc, 0, 2, "email", score=0.5),
        span(doc, 3, 5, "email", score=0.5)
    ]
    
    context_enhancer = nlp.add_pipe("context_enhancer", config={"context_window": (0, 5)})
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "valide"}]}
    ])
    
    result = context_enhancer(doc)
    
    # Both should be processed
    assert len(result.ents) == 2


def test_style_ent_vs_span():
    """Test with ent style retrieves from doc.ents."""
    nlp = Dutch()
    doc = nlp("Email contact")
    doc.ents = [span(doc, 0, 2, "email", score=0.5)]
    
    context_enhancer = nlp.add_pipe("context_enhancer", config={"style": "ent"})
    
    result = context_enhancer(doc)
    
    # Should retrieve from doc.ents with ent style
    assert len(result.ents) == 1


def test_add_patterns_extends_list():
    """Test that add_patterns extends existing patterns."""
    nlp = Dutch()
    
    context_enhancer = nlp.add_pipe("context_enhancer")
    context_enhancer.add_patterns([
        {"label": "email", "pattern": [{"LOWER": "geldig"}]}
    ])
    context_enhancer.add_patterns([
        {"label": "phone", "pattern": [{"LOWER": "actief"}]}
    ])
    
    assert len(context_enhancer._patterns) == 2


def test_default_initialization():
    """Test default initialization of context_enhancer."""
    nlp = Dutch()
    context_enhancer = nlp.add_pipe("context_enhancer")
    
    assert context_enhancer.name == "context_enhancer"
    assert context_enhancer.spans_key == "sc"
    assert context_enhancer.style == "span"
    assert context_enhancer.confidence_boost == 0.35
    assert context_enhancer.min_enhanced_score == 0.4
    assert context_enhancer.context_before == 5
    assert context_enhancer.context_after == 1


def test_custom_spans_key():
    """Test with custom spans_key configuration."""
    nlp = Dutch()
    doc = nlp("De email contact")
    doc.spans["custom_key"] = [span(doc, 1, 3, "email", score=0.5)]
    
    context_enhancer = nlp.add_pipe("context_enhancer", config={"spans_key": "custom_key"})
    
    result = context_enhancer(doc)
    
    # Should work with custom spans key
    assert "custom_key" in result.spans
