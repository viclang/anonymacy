from typing import Dict, List, Optional, Tuple, Union, Any
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher
from spacy.pipeline import Pipe
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TypedDict,
    Required,
    NotRequired,
)

logger = logging.getLogger("context-enhancer")

class ContextPatternType(TypedDict):
    label: Required[str]
    context: Required[Union[str, List[Dict[str, Any]]]]
    invalidate: NotRequired[bool]

@Language.factory("context_enhancer")
class ContextEnhancer(Pipe):
    """spaCy pipeline component that enhances or invalidates entities based on context."""
    
    def __init__(
        self,
        nlp: Language,
        name: str = "context_enhancer",
        spans_key: str = "sc",
        style: str = "span",
        patterns: List[ContextPatternType] = None,
        added_context_words: Optional[List[str]] = None,
        confidence_boost: float = 0.35,
        min_enhanced_score: float = 0.4,
        context_window: Tuple[int, int] = (5, 2),
        allow_dependency_link: bool = True,
        ignore_recognizer_context: bool = False,
        ignore_sources: List[str] = ["unknown"]
    ):
        """Initialize ContextEnhancer.
        
        Args:
            nlp: The spaCy Language object
            name: Name of the component
            patterns: List of context patterns for enhancement/invalidation
            added_context_words: Additional context words to append to doc
            confidence_boost: Score increase for enhanced spans
            min_enhanced_score: Minimum score after enhancement
            context_window: (before, after) token window for context matching
            allow_dependency_link: If True, allows dependency-based context matching
            style: "ent" or "span" for where to store results
            spans_key: Key for doc.spans if style="span"
            ignore_recognizer_context: If True, ignores context patterns from recognizers
            ignore_sources: List of sources to ignore when enhancing spans
        """
        self.nlp = nlp
        self.name = name
        self.added_context_words = added_context_words if added_context_words else []
        self.confidence_boost = confidence_boost
        self.min_enhanced_score = min_enhanced_score
        self.context_before, self.context_after = context_window
        self.allow_dependency_link = allow_dependency_link
        self.style = style
        self.spans_key = spans_key
        self.ignore_recognizer_context = ignore_recognizer_context
        self.ignore_sources = ignore_sources

        # Store our own patterns
        self._patterns = patterns if patterns else []

        # Ensure extensions exist
        if not Span.has_extension("score"):
            Span.set_extension("score", default=0.0)
        if not Span.has_extension("supportive_context"):
            Span.set_extension("supportive_context", default=[])
    
    def __call__(self, doc: Doc) -> Doc:
        """Process document."""
        # Get spans
        spans = self._get_spans(doc)
        if not spans:
            return doc
        
        # Collect all patterns
        all_patterns = []
        all_patterns.extend(self._patterns)
        
        # Collect patterns from recognizers unless ignoring them
        if not self.ignore_recognizer_context:
            for name, pipe in self.nlp.pipeline:
                if hasattr(pipe, 'context_patterns') and pipe.context_patterns:
                    all_patterns.extend(pipe.context_patterns)
        
        if not all_patterns:
            return doc
        
        # Build matcher with all patterns
        matcher = Matcher(self.nlp.vocab)
        pattern_map = {}  # Maps pattern_id to pattern config
        
        for idx, pattern_config in enumerate(all_patterns):
            try:
                label = pattern_config["label"]
                pattern_list = pattern_config["pattern"]
                invalidate = pattern_config.get("invalidate", False)
                
                pattern_id = f"{label}_{idx}"
                matcher.add(pattern_id, [pattern_list])
                pattern_map[pattern_id] = {
                    "label": label,
                    "pattern": pattern_list,
                    "invalidate": invalidate
                }
            except KeyError as e:
                logger.warning(f"Skipping invalid pattern {pattern_config}: missing {e}")
                continue
        
        # Create extended doc if needed
        if self.added_context_words:
            extended_text = doc.text + " " + " ".join(self.added_context_words)
            extended_doc = self.nlp.make_doc(extended_text)
        else:
            extended_doc = doc
        
        # Find all matches
        matches = matcher(extended_doc)
        
        # Process each span
        processed_spans = []
        for span in spans:

            # Skip if source is in ignore list
            source = getattr(span._, "source", "unknown")
            if source in self.ignore_sources:
                processed_spans.append(span)
                continue

            # Skip if already enhanced
            if hasattr(span._, "supportive_context") and span._.supportive_context:
                processed_spans.append(span)
                continue
         
            should_invalidate = False
            supportive_context = []
            
            # Check all matches for this span
            for match_id, start, end in matches:
                match = doc[start:end]
                if not self._in_context(span, match, len(doc)):
                    continue
                
                pattern_id = self.nlp.vocab.strings[match_id]
                pattern_config = pattern_map.get(pattern_id)
                
                if pattern_config and pattern_config["label"] == span.label_:
                    if pattern_config["invalidate"]:
                        should_invalidate = True
                        break  # No need to continue if invalidating
                    else:
                        context_text = extended_doc[start:end].text
                        supportive_context.append(context_text)
            
            # Apply results
            if should_invalidate:
                continue  # Skip invalidated spans
            
            if supportive_context:
                enhanced_span = self._create_enhanced_span(span, supportive_context)
                processed_spans.append(enhanced_span)
            else:
                processed_spans.append(span)
        
        # Update spans
        self._set_spans(doc, processed_spans)
        return doc

    def add_patterns(self, patterns: List[ContextPatternType]) -> None:
        """Add patterns to the context enhancer."""
        for pattern in patterns:
            self._patterns.append(pattern)

    def _get_spans(self, doc: Doc) -> List[Span]:
        """Get current spans."""
        return (
            list(doc.ents) if self.style == "ent" 
            else list(doc.spans.get(self.spans_key, []))
        )
    
    def _set_spans(self, doc: Doc, spans: List[Span]) -> None:
        """Set spans on document."""
        if self.style == "ent":
            doc.ents = spans
        else:
            doc.spans[self.spans_key] = spans

    def _has_dependency_link(self, span: Span, match_tokens: Span) -> bool:
        """Check if span and match_tokens are directly related via dependency (head or child)."""
        if not span.doc.has_annotation("DEP"):
            return False

        for token in span:
            for match_token in match_tokens:
                if token.head == match_token or match_token.head == token:
                    return True
        return False
    
    def _in_context(self, span: Span, match: Span, doc_length: int) -> bool:
        """Check if match is in span's context window."""
        # Check if match is one of the added_context_words

        if match.start >= doc_length:
            return True
        
        context_window_start = max(0, span.start - self.context_before)
        context_window_end = min(doc_length, span.end + self.context_after)

        # Before context - check if match ends in the before-window
        if (context_window_start < match.end <= span.start):
            return True
        
        # After context - check if match starts in the after-window
        if (span.end <= match.start < context_window_end):
            return True

        # Outside of window: try dependency-based relation as fallback
        if self.allow_dependency_link:
            return self._has_dependency_link(span, match)
        
        return False
    
    def _create_enhanced_span(self, span: Span, context: List[str]) -> Span:
        """Create enhanced version of span."""
        logger.debug(f"Enhancing span: {span.text} [{span.label_}] with contexts: {context}")
        enhanced = Span(span.doc, span.start, span.end, label=span.label_)
        
        # Calculate new score
        original_score = getattr(span._, "score", 1.0)
        new_score = min(
            max(original_score + self.confidence_boost, self.min_enhanced_score),
            1.0
        )
        
        enhanced._.score = new_score
        enhanced._.supportive_context = context
        
        return enhanced