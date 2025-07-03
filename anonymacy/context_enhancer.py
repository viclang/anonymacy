from typing import Dict, List, Optional, Tuple
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.matcher import Matcher
from spacy.pipeline import Pipe
import logging

logger = logging.getLogger("context-enhancer")


@Language.factory("context_enhancer")
class ContextEnhancer(Pipe):
    """spaCy pipeline component that enhances or invalidates entities based on context."""
    
    def __init__(
        self,
        nlp: Language,
        name: str = "context_enhancer",
        context_words: Optional[List[str]] = None,
        confidence_boost: float = 0.35,
        min_enhanced_score: float = 0.4,
        context_window: Tuple[int, int] = (5, 1),
        style: str = "span",
        spans_key: str = "sc"
    ):
        self.nlp = nlp
        self.name = name
        self.context_words = context_words if context_words else []
        self.confidence_boost = confidence_boost
        self.min_enhanced_score = min_enhanced_score
        self.context_before, self.context_after = context_window
        self.style = style
        self.spans_key = spans_key
        
        # Single matcher and pattern storage like pattern_recognizer
        self.matcher = Matcher(nlp.vocab)
        self._patterns = []
        
        # Ensure extensions exist
        if not Span.has_extension("score"):
            Span.set_extension("score", default=1.0)
        if not Span.has_extension("context_enhanced"):
            Span.set_extension("context_enhanced", default=False)
        if not Span.has_extension("supportive_context"):
            Span.set_extension("supportive_context", default=[])
    
    def add_context_patterns(self, patterns: List[Dict]) -> None:
        """Add context patterns."""
        for pattern_config in patterns:
            try:
                label = pattern_config["label"]
                pattern_list = pattern_config["pattern"]
                invalidate = pattern_config.get("invalidate", False)
                
                pattern_id = f"{label}_{len(self._patterns)}"
                self.matcher.add(pattern_id, [pattern_list])
                self._patterns.append({
                    "label": label, 
                    "pattern": pattern_list, 
                    "invalidate": invalidate
                })
                
            except KeyError as e:
                raise ValueError(
                    f"Missing key {str(e)} in pattern {pattern_config}. Required: 'label', 'pattern'. Optional: 'invalidate'."
                )
    
    def __call__(self, doc: Doc) -> Doc:
        """Process document."""
        # Get spans
        spans = self._get_spans(doc)
        if not spans:
            return doc
        
        # Find all pattern matches once
        if self.context_words:
            # Combine original text with context
            extended_text = doc.text + " " + " ".join(self.context_words)
            extended_doc = self.nlp.make_doc(extended_text)
        else:
            extended_doc = doc

        matches = self.matcher(extended_doc)
        
        # Process each span
        processed_spans = []
        for span in spans:
            if span._.context_enhanced:
                processed_spans.append(span)
                continue
            
            should_invalidate = False
            supportive_contexts = []
            
            # Loop through matches once and do everything
            for match_id, start, end in matches:
                if not self._in_context(span, start, end, len(doc)):
                    continue
                
                pattern_id = self.nlp.vocab.strings[match_id]
                try:
                    pattern_index = int(pattern_id.split('_')[-1])
                    pattern_config = self._patterns[pattern_index]
                    
                    if pattern_config["label"] == span.label_:
                        if pattern_config["invalidate"]:
                            should_invalidate = True
                            break  # No need to continue if invalidating
                        else:
                            context_text = extended_doc[start:end].text
                            supportive_contexts.append(context_text)
                except (ValueError, IndexError):
                    continue
            
            # Apply results
            if should_invalidate:
                continue  # Skip invalidated spans
            
            if supportive_contexts:
                enhanced_span = self._create_enhanced_span(span, supportive_contexts)
                processed_spans.append(enhanced_span)
            else:
                processed_spans.append(span)
        
        # Update spans
        self._set_spans(doc, processed_spans)
        return doc
    
    def _get_spans(self, doc: Doc) -> List[Span]:
        """Get current spans."""
        if self.style == "ent":
            return list(doc.ents) if doc.ents else []
        return list(doc.spans.get(self.spans_key, []))
    
    def _set_spans(self, doc: Doc, spans: List[Span]) -> None:
        """Set spans on document."""
        if self.style == "ent":
            doc.ents = spans
        else:
            doc.spans[self.spans_key] = spans
    
    def _in_context(self, span: Span, match_start: int, match_end: int, doc_length: int) -> bool:
        """Check if match is in span's context window."""
        
        # Check if match is one of the context_words
        if match_start >= doc_length:
            return True

        context_window_start = max(0, span.start - self.context_before)
        context_window_end = min(doc_length, span.end + self.context_after)

        # Before context - check if match ends in the before-window
        if (context_window_start < match_end <= span.start):
            return True
        
        # After context - check if match starts in the after-window
        if (span.end <= match_start < context_window_end):
            return True

        return False
    
    def _create_enhanced_span(self, span: Span, contexts: List[str]) -> Span:
        """Create enhanced version of span."""
        enhanced = Span(span.doc, span.start, span.end, label=span.label_)
        
        # Calculate new score
        original_score = getattr(span._, "score", 1.0)
        new_score = min(
            max(original_score + self.confidence_boost, self.min_enhanced_score),
            1.0
        )
        
        enhanced._.score = new_score
        enhanced._.context_enhanced = True
        enhanced._.supportive_context = contexts
        
        return enhanced
    
    def clear_patterns(self) -> None:
        """Clear all patterns."""
        self.matcher = Matcher(self.nlp.vocab)
        self._patterns = []