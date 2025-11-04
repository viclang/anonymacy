from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.pipeline import Pipe
from spacy.matcher import Matcher, PhraseMatcher
from spacy.matcher.levenshtein import levenshtein_compare
from anonymacy import span_filter
from anonymacy.util import read_pickle, write_pickle
from spacy import util
from spacy.errors import Errors
import srsly
from spacy import util
from spacy.util import SimpleFrozenList, ensure_path
from pathlib import Path
from spacy import registry

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    TypedDict,
    Required,
    NotRequired,
)
import logging

logger = logging.getLogger("anonymacy.recognizer")

class PatternType(TypedDict):
    label: Required[str]
    pattern: Required[Union[str, List[Dict[str, Any]]]]
    score: NotRequired[float]
    id: NotRequired[str]

CustomMatcherFunc = Callable[[Doc], List[Span]]

SpansFilterFunc = Callable[[Iterable[Span], Iterable[Span]], Iterable[Span]]

RECOGNIZER_DEFAULT_SPANS_KEY = "sc"

# Compatibility function for registry
def anonymacy_levenshtein_compare(s1: str, s2: str, max_dist: int) -> bool:
    return levenshtein_compare(s1, s2, max_dist)

@registry.misc("spacy.levenshtein_compare.v1")
def make_levenshtein_compare():
    return anonymacy_levenshtein_compare

DEFAULT_RECOGNIZER_CONFIG = {
    "spans_key": RECOGNIZER_DEFAULT_SPANS_KEY,
    "spans_filter": None,
    "annotate_ents": False,
    "ents_filter": {"@misc": "anonymacy.highest_confidence_filter.v1"},
    "phrase_matcher_attr": None,
    "matcher_fuzzy_compare": {"@misc": "spacy.levenshtein_compare.v1"},
    "default_score": 0.6,
    "validate_patterns": False,
    "overwrite": False
}

@Language.factory("recognizer", assigns=["doc.spans"], default_config=DEFAULT_RECOGNIZER_CONFIG,)
class Recognizer(Pipe):
    
    def __init__(
        self,
        nlp: Language,
        name: str = "recognizer",
        spans_key: Optional[str] = RECOGNIZER_DEFAULT_SPANS_KEY,
        spans_filter: Optional[SpansFilterFunc] = None,
        annotate_ents: bool = False,
        ents_filter: SpansFilterFunc = span_filter.highest_confidence_filter,
        phrase_matcher_attr: Optional[Union[int, str]] = None,
        matcher_fuzzy_compare: Callable = anonymacy_levenshtein_compare,
        default_score: float = 0.6,
        validate_patterns: bool = False,
        overwrite: bool = False,
    ):
        self.nlp = nlp
        self.name = name
        self.spans_key = spans_key
        self.spans_filter = spans_filter
        self.ents_filter = ents_filter
        self.default_score = min(default_score, 1.0)
        self.annotate_ents = annotate_ents
        self.phrase_matcher_attr = phrase_matcher_attr
        self.matcher_fuzzy_compare = matcher_fuzzy_compare
        self._match_label_id_map: Dict[str, Dict[str, Any]] = {}
        self.validate_patterns = validate_patterns
        self.overwrite = overwrite
        self.clear()

    def __call__(self, doc: Doc) -> Doc:
        """Process document with automatic deduplication."""
        matches = self.match(doc)
        self.set_annotations(doc, matches)
        return doc
    
    @property
    def labels(self) -> Tuple[str, ...]:
        """All labels present in the match patterns."""
        labels = [p["label"] for p in self._patterns]
        if self._custom_matchers:
            labels.extend(self._custom_matchers.keys())

        return tuple(sorted(set(labels)))

    @property
    def patterns(self) -> List[Dict[str, Any]]:
        """Get all patterns that were added."""
        return self._patterns

    def __len__(self) -> int:
        """The number of all labels added to the span ruler."""
        return len(self.labels)

    def __contains__(self, label: str) -> bool:
        """Whether a label is present in the patterns."""
        for label_id in self._match_label_id_map.values():
            if label_id["label"] == label:
                return True
        return False

    def match(self, doc: Doc) -> List[Span]:
        """Find all matches in the document."""
        spans: dict[tuple[int, int], Span] = {}
        
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
            score = min(pattern_info.get("score", self.default_score), 1.0)
            
            key = (start, end)
            if key in spans and spans[key]._.score >= score:
                continue
            
            span = doc[start:end]
            span.label_ = label
            span_id = pattern_info.get("id", "")
            if isinstance(span_id, str):
                span_id = doc.vocab.strings.add(span_id)
            span.id = span_id
            span._.score = score
            if self._validators and not self._is_valid(span):
                continue

            spans[key] = span

        if self._custom_matchers:
            for label, custom_matcher in self._custom_matchers.items():
                custom_matches = custom_matcher(doc)
                
                for start, end, score in custom_matches:
                    key = (start, end)
                    
                    score = min(score, 1.0)
                    if score <= 0.0:
                        score = self.default_score
                    
                    if key in spans and spans[key]._.score >= score:
                        continue

                    span = doc[start:end]
                    span.label_ = label
                    span._.score = score

                    if self._validators and not self._is_valid(span):
                        logger.debug("dropped span %r label=%s validator=False", span.text, span.label_)
                        continue
                
                    spans[key] = span

        return sorted(spans.values(), key=lambda s: (s.start, s.end))

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

    def add_patterns(self, patterns: List[PatternType]) -> None:
        """Add patterns to the matcher."""

        phrase_pattern_labels = []
        phrase_pattern_texts = []

        for entry in patterns:
            p_label = entry["label"]
            p_id = entry.get("id", "")
            p_score = entry.get("score", self.default_score)
            
            label = f"{p_label}_{len(self._patterns)}"

            self._match_label_id_map[self.nlp.vocab.strings.as_int(label)] = {
                "label": p_label,
                "id": p_id,
                "score": p_score
            }
            
            if isinstance(entry["pattern"], str):
                # Phrase pattern - store for later processing
                phrase_pattern_labels.append(label)
                phrase_pattern_texts.append(entry["pattern"])
            elif isinstance(entry["pattern"], list):
                # Token pattern - add directly
                self.matcher.add(label, [entry["pattern"]])
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
            for label, pattern in zip(
                phrase_pattern_labels,
                self.nlp.pipe(phrase_pattern_texts)
            ):
                self.phrase_matcher.add(label, [pattern])

    def add_custom_matchers(self, matchers: Dict[str, Callable[[Doc], List[Tuple[int, int, float]]]]) -> None:
        """Add custom matchers to the recognizer.

        Args:
            matchers (Dict[str, Callable[[Doc], List[Tuple[int, int, float]]]]): A dictionary of custom matchers.
        """
        self._custom_matchers.update(matchers)

    def add_validators(self, validators: Dict[str, Callable[[Span], bool]]) -> None:
        """Add validation functions to the recognizer.

        Args:
            validators: A Dict of functions that take a Span and return a bool.
        """
        self._validators.update(validators)

    def _is_valid(self, span: Span) -> bool:
        return self._validators.get(span.label_, lambda x: True)(span)

    def clear(self) -> None:
        """Reset all patterns.
        """
        self._patterns: List[PatternType] = []
        self._validators: Dict[str, Callable[[Span], bool]] = {}
        self._custom_matchers: Dict[str, Callable[[Doc], List[Span]]] = {}
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

    def remove(self, label: str) -> None:
        """Remove patterns by their label.

        label (str): Label of the patterns to be removed.
        """
        if not any(p["label"] == label for p in self._patterns):
            raise ValueError(
                Errors.E1024.format(attr_type="label", label=label, component=self.name)
            )
        self._patterns = [p for p in self._patterns if p["label"] != label]
        for m_label in self._match_label_id_map:
            if self._match_label_id_map[m_label]["label"] == label:
                m_label_str = self.nlp.vocab.strings.as_string(m_label)
                if m_label_str in self.phrase_matcher:
                    self.phrase_matcher.remove(m_label_str)
                if m_label_str in self.matcher:
                    self.matcher.remove(m_label_str)
                del self._match_label_id_map[m_label]

    def remove_by_id(self, pattern_id: str) -> None:
        """Remove a pattern by its pattern ID.

        pattern_id (str): ID of the pattern to be removed.
        """
        # Check if any pattern has the given ID
        if not any(p.get("id") == pattern_id for p in self._patterns):
            raise ValueError(
                Errors.E1024.format(attr_type="ID", label=pattern_id, component=self.name)
            )
        
        # Remove patterns from internal list
        self._patterns = [p for p in self._patterns if p.get("id") != pattern_id]
        
        for m_label in self._match_label_id_map:
            if self._match_label_id_map[m_label]["id"] == pattern_id:
                m_label_str = self.nlp.vocab.strings.as_string(m_label)
                if m_label_str in self.phrase_matcher:
                    self.phrase_matcher.remove(m_label_str)
                if m_label_str in self.matcher:
                    self.matcher.remove(m_label_str)
                del self._match_label_id_map[m_label]

    def from_bytes(self, bytes_data: bytes, *, exclude: Iterable[str] = SimpleFrozenList()) -> "Recognizer":
        """Load the span ruler from a bytestring.

        bytes_data (bytes): The bytestring to load.
        RETURNS (Recognizer): The loaded recognizer.
        """
        self.clear()
        deserializers = {
            "patterns": lambda b: self.add_patterns(srsly.json_loads(b)),
            "custom_matchers": lambda b: self.add_custom_matchers(srsly.pickle_loads(b)),
            "validators": lambda b: self.add_validators(srsly.pickle_loads(b)),
        }
        util.from_bytes(bytes_data, deserializers, exclude)
        return self

    def to_bytes(self, *, exclude: Iterable[str] = SimpleFrozenList()) -> bytes:
        """Serialize the span ruler to a bytestring.

        RETURNS (bytes): The serialized patterns.
        """
        serializers = {
            "patterns": lambda: srsly.json_dumps(self.patterns),
            "custom_matchers": lambda: srsly.pickle_dumps(self._custom_matchers),
            "validators": lambda: srsly.pickle_dumps(self._validators),
        }
        return util.to_bytes(serializers, exclude)

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "Recognizer":
        """Load the span ruler from a directory.

        path (Union[str, Path]): A path to a directory.
        RETURNS (Recognizer): The loaded recognizer.
        """
        self.clear()
        path = ensure_path(path)

        deserializers = {
            "patterns": lambda p: self.add_patterns(srsly.read_jsonl(p)),
            "custom_matchers": lambda p: self.add_custom_matchers(read_pickle(p)),
            "validators": lambda p: self.add_validators(read_pickle(p)),
        }
        util.from_disk(path, deserializers, exclude)
        return self

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """Save the recognizer patterns to a directory.

        path (Union[str, Path]): A path to a directory.
        """
        path = ensure_path(path)

        serializers = {
            "patterns": lambda p: srsly.write_jsonl(p, self.patterns),
            "custom_matchers": lambda p: write_pickle(p, self._custom_matchers),
            "validators": lambda p: write_pickle(p, self._validators),
        }
        util.to_disk(path, serializers, exclude)