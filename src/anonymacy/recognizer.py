from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.pipeline import Pipe
from spacy.matcher import Matcher, PhraseMatcher
from spacy.matcher.levenshtein import levenshtein_compare
from .conflict_resolver import highest_confidence_filter, SpansFilterFunc
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    TypedDict,
    Required,
    NotRequired,
)

class PatternType(TypedDict):
    label: Required[str]
    pattern: Required[Union[str, List[Dict[str, Any]]]]
    score: NotRequired[float]
    id: NotRequired[str]

# Simplified custom matcher - just one function
CustomMatcherFunc = Callable[[Doc], List[Span]]

@Language.factory("recognizer")
class Recognizer(Pipe):
    """Base class for custom spaCy recognizers with automatic deduplication."""
    
    def __init__(
        self,
        nlp: Language,
        name: str = "recognizer",
        spans_key: Optional[str] = "sc",
        custom_matcher: Optional[CustomMatcherFunc] = None,
        spans_filter: SpansFilterFunc = highest_confidence_filter,
        annotate_ents: bool = False,
        ents_filter: SpansFilterFunc = highest_confidence_filter,
        phrase_matcher_attr: Optional[Union[int, str]] = None,
        matcher_fuzzy_compare: Callable = levenshtein_compare,
        default_score: float = 0.6,
        validate_patterns: bool = False,
        overwrite: bool = False,
    ):
        self.nlp = nlp
        self.name = name
        self.spans_key = spans_key
        self.custom_matcher = custom_matcher
        self.spans_filter = spans_filter
        self.ents_filter = ents_filter
        self.default_score = default_score
        self.annotate_ents = annotate_ents
        self.phrase_matcher_attr = phrase_matcher_attr
        self.matcher_fuzzy_compare = matcher_fuzzy_compare
        self._match_label_id_map: Dict[str, Dict[str, Any]] = {}
        self.validate_patterns = validate_patterns
        self.overwrite = overwrite
        self.clear()
                        
        # Set up span extensions
        if not Span.has_extension("score"):
            Span.set_extension("score", default=0.0)

        if not Span.has_extension("source"):
            Span.set_extension("source", default="unknown")

    def __call__(self, doc: Doc) -> Doc:
        """Process document with automatic deduplication."""
        matches = self.match(doc)
        self.set_annotations(doc, matches)
        return doc

    def match(self, doc: Doc) -> List[Span]:
        """Find all matches in the document."""
        spans_by_position = {}
        
        # Get matches from the main pattern matcher
        matches = cast(
            List[Tuple[int, int, int]],
            list(self.matcher(doc)) + list(self.phrase_matcher(doc)),
        )

        for match_id, start, end in matches:
            if match_id not in self._match_label_id_map or start == end:
                continue
            
            pattern_info = self._match_label_id_map[match_id]
            label = pattern_info["label"]
            score = pattern_info.get("score", self.default_score)
            
            span = doc[start:end]
            span.label_ = label
            span._.score = score
            span._.source = self.name

            if self._validators and not self._is_valid(span):
                continue

            key = (start, end)
            if key not in spans_by_position or spans_by_position[key]._.score < score:
                spans_by_position[key] = span
        
        if self.custom_matcher:
            custom_matches = self.custom_matcher(doc)
            
            for span in custom_matches:
                if not hasattr(span._, "score") or span._.score is None:
                    span._.score = self.default_score
                if not hasattr(span._, "source") or not span._.source:
                    span._.source = self.name
                
                if self._validators and not self._is_valid(span):
                    continue

                key = (span.start, span.end)
                if key not in spans_by_position or spans_by_position[key]._.score < span._.score:
                    spans_by_position[key] = span
        
        return sorted(spans_by_position.values(), key=lambda s: (s.start, s.end))
        
    def set_annotations(self, doc, matches):
        """Modify the document in place"""
        if self.spans_key:
            spans = []
            if self.spans_key in doc.spans and not self.overwrite:
                spans = doc.spans[self.spans_key]
            spans.extend(
                self.spans_filter(spans, matches) if self.spans_filter else matches
            )

            doc.spans[self.spans_key] = spans
        
        # set doc.ents if annotate_ents is set
        if self.annotate_ents:
            spans = []
            if not self.overwrite:
                spans = list(doc.ents)
            spans = self.ents_filter(spans, matches)

            doc.ents = sorted(spans)

    def add_patterns(self, patterns: List[Dict[str, Any]]) -> None:
        """Add patterns to the matcher."""

        phrase_pattern_ids = []
        phrase_pattern_texts = []

        for entry in patterns:
            p_label = entry["label"]
            p_id = entry.get("id", "")
            p_score = entry.get("score", self.default_score)
            
            pattern_id = f"{p_label}_{len(self._patterns)}"
            
            pattern_string_id = self.nlp.vocab.strings.add(pattern_id)
            self._match_label_id_map[pattern_string_id] = {
                "label": p_label,
                "id": p_id,
                "score": p_score
            }
            
            if isinstance(entry["pattern"], str):
                # Phrase pattern - store for later processing
                phrase_pattern_ids.append(pattern_id)
                phrase_pattern_texts.append(entry["pattern"])
            elif isinstance(entry["pattern"], list):
                # Token pattern - add directly
                self.matcher.add(pattern_id, [entry["pattern"]])
            else:
                raise ValueError(f"Pattern must be string or list of dicts, got {type(entry['pattern'])}")
            
            self._patterns.append(entry)

        # Temporarily disable the nlp components after this one in case they haven't been
        # initialized / deserialized yet
        try:
            current_index = -1
            for i, (name, pipe) in enumerate(self.nlp.pipeline):
                if self == pipe:
                    current_index = i
                    break
            subsequent_pipes = [pipe for pipe in self.nlp.pipe_names[current_index:]]
        except ValueError:
            subsequent_pipes = []

        with self.nlp.select_pipes(disable=subsequent_pipes):
            # Process phrase patterns through the pipeline to tokenize them
            for pattern_id, pattern_doc in zip(
                phrase_pattern_ids,
                self.nlp.pipe(phrase_pattern_texts),
            ):
                # Add the tokenized pattern to phrase_matcher
                label_id = self.nlp.vocab.strings.add(pattern_id)
                self.phrase_matcher.add(label_id, [pattern_doc])
                # Update the label mapping with the actual ID used by PhraseMatcher
                self._match_label_id_map[label_id] = self._match_label_id_map.pop(
                    self.nlp.vocab.strings.add(pattern_id)
                )

    def add_validators(self, validators: Dict[str, Callable[[Doc], bool]]) -> None:
        """Add validation functions to the recognizer.

        Args:
            validators: A Dict of functions that take a Doc and return a bool.
        """
        self._validators.update(validators)

    def _is_valid(self, span: Span) -> bool:
        return self._validators.get(span.label_, lambda x: True)(span)

    def clear(self) -> None:
        """Reset all patterns.

        RETURNS: None
        """
        self._patterns: List[PatternType] = []
        self._validators = {}
        self.custom_matcher = None
        self.matcher: Matcher = Matcher(
            self.nlp.vocab,
            validate=self.validate_patterns,
            fuzzy_compare=self.matcher_fuzzy_compare,
        )
        self.phrase_matcher: PhraseMatcher = PhraseMatcher(
            self.nlp.vocab,
            attr=self.phrase_matcher_attr,
            validate=self.validate_patterns,
        )


    @property
    def labels(self) -> Tuple[str, ...]:
        """All labels present in the match patterns."""
        return tuple(sorted(set([p["label"] for p in self._patterns])))

    @property
    def patterns(self) -> List[Dict[str, Any]]:
        """Get all patterns that were added."""
        return self._patterns