import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Union,
    cast,
)

import srsly
from spacy import util
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Doc, Span
from spacy.util import SimpleFrozenList, ensure_path

from .span_filter import DEFAULT_HIERARCHY, hierarchical_merge_filter
from .util import read_pickle, write_pickle

NoArgReplacement = Callable[[], str]
TextReplacement = Callable[[str], str]
SpansFilterFunc = Callable[[Iterable[Span]], Iterable[Span]]

DEFAULT_ANONYMIZER_CONFIG = {
    "spans_key": "sc",
    "spans_filter": {
        "@misc": "maskpipe.hierarchical_merge_filter.v1",
        "hierarchy": DEFAULT_HIERARCHY
    },
    "style": "ent",
}

@Language.factory("anonymizer", assigns=["doc.ents"], default_config=DEFAULT_ANONYMIZER_CONFIG)
class Anonymizer(Pipe):
    """
    Replace sensitive entities with user-defined surrogates.

    The pipe stores an anonymized copy of the document under
    `doc._.anonymized` and leaves the original document untouched.
    Replacements can be fixed strings or callables (0- or 1-argument).
    """

    def __init__(
        self,
        nlp: Language,
        name: str = "anonymizer",
        spans_key: str = "sc",
        spans_filter: SpansFilterFunc = hierarchical_merge_filter,
        style: str = "ent",
    ):
        self.nlp = nlp
        self.name = name
        self.spans_key = spans_key
        self.spans_filter = spans_filter
        self.style = style
        self.clear()

    def __call__(self, doc: Doc) -> Doc:
        """
        Process a document and attach an anonymised copy under `doc._.anonymized`.
        The original document is returned unchanged.
        """
        doc._.anonymized = self._make_anonymized_doc(doc)
        return doc

    def add_redactors(
        self, redactors: Dict[str, Union[str, NoArgReplacement, TextReplacement]]
    ) -> None:
        """
        Register or update replacement rules for entity types.

        Parameters
        ----------
        redactors : dict
            Maps entity labels to either:
            - a fixed string, or
            - a zero-argument callable returning a string, or
            - a one-argument callable receiving the original text and returning
              its replacement.

            If no replacement is registered for a label, the default replacement
            is ``[LABEL]``.
        """
        self._redactors.update(redactors)

    def remove(self, label: str) -> None:
        """
        Remove redactor rules for the given entity labels.

        Parameters
        ----------
        label : str
            Entity label to remove redactors for.
        """
        self._redactors.pop(label, None)

    def clear(self) -> None:
        """Remove all registered redactors."""
        self._redactors: Dict[str, Union[str, NoArgReplacement, TextReplacement]] = {}

    def _get_spans(self, doc: Doc) -> Iterable[Span]:
        if self.style == "ent":
            return list(doc.ents)
        spans = list(doc.spans.get(self.spans_key, []))
        return self.spans_filter(spans)

    def _apply_redactor(self, label: str, text: str) -> str:
        redactor = self._redactors.get(label)
        if redactor is None:
            return f"[{label.upper()}]"

        if isinstance(redactor, str):
            return redactor

        sig = inspect.signature(redactor)
        params = tuple(sig.parameters.values())

        if params and params[0].name == "self":
            params = params[1:]

        if not params:
            no_arg_redactor = cast(NoArgReplacement, redactor)
            return no_arg_redactor()

        if len(params) == 1:
            text_redactor = cast(TextReplacement, redactor)
            return text_redactor(text)

        raise TypeError(
            f"Replacement for label {label!r} must be a str or a callable "
            "taking zero or one positional argument"
        )

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
        token_index: int = 0
        doc_length: int = len(original_doc)

        span_info: list[tuple[int, int, str]] = []
        while token_index < doc_length:
            # outside any sensitive span → copy token
            if span_index >= len(sensitive_spans) or token_index < sensitive_spans[span_index].start:
                token = original_doc[token_index]
                new_tokens.append(token.text)
                new_spaces.append(bool(token.whitespace_))
                token_index += 1
                continue

            # hit a sensitive span → replace it
            current_span = sensitive_spans[span_index]
            replacement_tokens = self._apply_redactor(current_span.label_, current_span.text)
            
            start_position = len(new_tokens)
            replacement_doc = self.nlp.tokenizer(replacement_tokens)
            for new_token in replacement_doc:
                new_tokens.append(new_token.text)
                new_spaces.append(bool(new_token.whitespace_))

            if new_spaces:
                new_spaces[-1] = bool(current_span[-1].whitespace_)

            span_info.append(
                (
                    start_position,
                    start_position + len(replacement_doc),
                    current_span.label_
                )
            )
            token_index = current_span.end
            span_index += 1

        # build the new document and create spans
        anonymized_document = Doc(original_doc.vocab, words=new_tokens, spaces=new_spaces)
        new_spans: list[Span] = []
        for start, end, label in span_info:
            new_span = anonymized_document[start:end]
            new_span.label_ = label
            new_spans.append(new_span)

        if self.style == "ent":
            anonymized_document.ents = new_spans
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
        Load redactors from a byte blob.

        Parameters
        ----------
        bytes_data : bytes
            Serialised redactors.
        exclude : iterable, optional
            Ignored; kept for compatibility with spaCy's serialisation protocol.

        Returns
        -------
        Anonymizer
            self, with redactors restored.
        """
        self.clear()
        deserializers = {"redactors": lambda b: self._redactors.update(srsly.pickle_loads(b))} # ty:ignore[no-matching-overload]
        util.from_bytes(bytes_data, deserializers, exclude) # ty:ignore[invalid-argument-type]
        return self

    def to_bytes(
        self, *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> bytes:
        """
        Serialize redactors to a byte blob.

        Parameters
        ----------
        exclude : iterable, optional
            Ignored; kept for compatibility with spaCy's serialisation protocol.

        Returns
        -------
        bytes
            Pickled redactors dictionary.
        """
        serializers = {"redactors": lambda: srsly.pickle_dumps(self._redactors)}
        return util.to_bytes(serializers, exclude)  # ty:ignore[invalid-argument-type]

    def from_disk(
        self,
        path: Union[str, Path],
        *,
        exclude: Iterable[str] = SimpleFrozenList(),
    ) -> "Anonymizer":
        """
        Load redactors from disk.

        Parameters
        ----------
        path : str or Path
            Directory containing the file "redactors".
        exclude : iterable, optional
            Ignored; kept for compatibility with spaCy's serialisation protocol.

        Returns
        -------
        Anonymizer
            self, with redactors restored.
        """
        self.clear()
        path = ensure_path(path)
        deserializers = {
            "redactors": lambda p: self._redactors.update(read_pickle(p))
        }
        util.from_disk(path, deserializers, exclude)  # ty:ignore[invalid-argument-type]
        return self

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """
        Save redactors to disk.

        Parameters
        ----------
        path : str or Path
            Target directory; the file "redactors" will be created inside.
        exclude : iterable, optional
            Ignored; kept for compatibility with spaCy's serialisation protocol.
        """
        path = ensure_path(path)
        serializers = {"redactors": lambda p: write_pickle(p, self._redactors)}
        util.to_disk(path, serializers, exclude)  # ty:ignore[invalid-argument-type]