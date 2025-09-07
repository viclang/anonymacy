from typing import Callable, Dict, List, Union, Iterable
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.pipeline import Pipe
from spacy import util
from spacy.util import ensure_path, SimpleFrozenList
from anonymacy.span_filter import highest_confidence_filter
from anonymacy.util import read_pickle, write_pickle
from pathlib import Path
import srsly

NoArgOperator = Callable[[], str]
TextOperator = Callable[[str], str]

@Language.factory("anonymizer")
class Anonymizer(Pipe):
    """
    Replace sensitive entities with user-defined surrogates.

    The pipe stores an anonymized copy of the document under
    `doc._.anonymized` and leaves the original document untouched.
    Operators can be fixed strings or callables (0- or 1-argument).
    """

    def __init__(
        self,
        nlp: Language,
        name: str = "anonymizer",
        spans_key: str = "sc",
        style: str = "ent",
    ):
        self.nlp = nlp
        self.name = name
        self.spans_key = spans_key
        self.style = style
        self.clear()

    def __call__(self, doc: Doc) -> Doc:
        """
        Process a document and attach an anonymised copy under `doc._.anonymized`.
        The original document is returned unchanged.
        """
        doc._.anonymized = self._make_anonymized_doc(doc)
        return doc

    def add_operators(
        self, operators: Dict[str, Union[str, NoArgOperator, TextOperator]]
    ) -> None:
        """
        Register or update replacement rules for entity types.

        Parameters
        ----------
        operators : dict
            Maps entity labels to either:
            - a fixed string, or
            - a zero-argument callable returning a string, or
            - a one-argument callable receiving the original text and returning
              its replacement.

            If no operator is registered for a label, the default replacement
            is ``[LABEL]``.
        """
        self._operators.update(operators)

    def clear(self) -> None:
        """Remove all registered operators."""
        self._operators: Dict[str, Union[str, NoArgOperator, TextOperator]] = {}

    def _get_spans(self, doc: Doc) -> List[Span]:
        if self.style == "ent":
            return list(doc.ents)
        spans = list(doc.spans.get(self.spans_key, []))
        return highest_confidence_filter(spans)

    def _apply_operator(self, label: str, text: str) -> str:
        operator = self._operators.get(label)
        if not operator:
            return f"[{label.upper()}]"

        if isinstance(operator, str):
            return operator

        try:
            argcount = operator.__code__.co_argcount
            if hasattr(operator, "__self__"):
                argcount -= 1
        except AttributeError:
            argcount = 1

        if argcount == 0:
            return operator()
        return operator(text)

    def _make_anonymized_doc(self, original_doc: Doc) -> Doc:
        sensitive_spans = sorted(self._get_spans(original_doc), key=lambda span: span.start)

        # fast path: no sensitive data
        if not sensitive_spans:
            return Doc(
                original_doc.vocab,
                words=[token.text for token in original_doc],
                spaces=[bool(token.whitespace_) for token in original_doc],
            )

        new_tokens: list[str] = []
        new_spaces: list[bool] = []
        span_index: int = 0
        output_position: int = 0

        span_info: list[tuple[int, int, str, str]] = []

        for original_index, original_token in enumerate(original_doc):
            # outside any sensitive span → copy token
            if span_index >= len(sensitive_spans) or original_index < sensitive_spans[span_index].start:
                new_tokens.append(original_token.text)
                new_spaces.append(bool(original_token.whitespace_))
                output_position += 1
                continue

            # hit a sensitive span → replace it
            current_span = sensitive_spans[span_index]
            replacement_tokens = self._apply_operator(current_span.label_, current_span.text)
            if isinstance(replacement_tokens, str):
                replacement_tokens = [replacement_tokens]

            start_position = output_position
            new_tokens.extend(replacement_tokens)

            # only the last replacement token inherits the original space flag
            last_has_space: bool = bool(current_span[-1].whitespace_)
            new_spaces.extend([False] * (len(replacement_tokens) - 1) + [last_has_space])
            output_position += len(replacement_tokens)

            span_info.append(
                (start_position, output_position, current_span.label_, current_span.text)
            )
            span_index += 1

        # build the new document and create spans
        anonymized_document = Doc(original_doc.vocab, words=new_tokens, spaces=new_spaces)

        new_spans: list[Span] = []
        for start, end, label, original_text in span_info:
            new_span = anonymized_document[start:end]
            new_span.label_ = label
            new_span._.original_text = original_text
            new_spans.append(new_span)

        if self.style == "ent":
            anonymized_document.ents = list(anonymized_document.ents) + new_spans
        else:
            key = self.spans_key
            if key not in anonymized_document.spans:
                anonymized_document.spans[key] = []
            anonymized_document.spans[key].extend(new_spans)

        return anonymized_document

    def from_bytes(
        self,
        bytes_data: bytes,
        *,
        exclude: Iterable[str] = SimpleFrozenList(),
    ) -> "Anonymizer":
        """
        Load operators from a byte blob.

        Parameters
        ----------
        bytes_data : bytes
            Serialised operators.
        exclude : iterable, optional
            Ignored; kept for compatibility with spaCy's serialisation protocol.

        Returns
        -------
        Anonymizer
            self, with operators restored.
        """
        self.clear()
        deserializers = {"operators": lambda b: self._operators.update(srsly.pickle_loads(b))}
        util.from_bytes(bytes_data, deserializers)
        return self

    def to_bytes(
        self, *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> bytes:
        """
        Serialise operators to a byte blob.

        Parameters
        ----------
        exclude : iterable, optional
            Ignored; kept for compatibility with spaCy's serialisation protocol.

        Returns
        -------
        bytes
            Pickled operators dictionary.
        """
        serializers = {"operators": lambda: srsly.pickle_dumps(self._operators)}
        return util.to_bytes(serializers)

    def from_disk(
        self,
        path: Union[str, Path],
        *,
        exclude: Iterable[str] = SimpleFrozenList(),
    ) -> "Anonymizer":
        """
        Load operators from disk.

        Parameters
        ----------
        path : str or Path
            Directory containing the file "operators".
        exclude : iterable, optional
            Ignored; kept for compatibility with spaCy's serialisation protocol.

        Returns
        -------
        Anonymizer
            self, with operators restored.
        """
        self.clear()
        path = ensure_path(path)
        deserializers = {
            "operators": lambda p: self._operators.update(read_pickle(p))
        }
        util.from_disk(path, deserializers, {})
        return self

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """
        Save operators to disk.

        Parameters
        ----------
        path : str or Path
            Target directory; the file "operators" will be created inside.
        exclude : iterable, optional
            Ignored; kept for compatibility with spaCy's serialisation protocol.
        """
        path = ensure_path(path)
        serializers = {"operators": lambda p: write_pickle(p, self._operators)}
        util.to_disk(path, serializers, {})