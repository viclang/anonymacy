from typing import Callable, Dict, List, Union, Iterable
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.pipeline import Pipe
from spacy import util
from spacy.util import ensure_path, SimpleFrozenList
from pathlib import Path
import pickle

NoArgOperator = Callable[[], str]
TextOperator = Callable[[str], str]


@Language.factory("anonymizer")
class Anonymizer(Pipe):

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
        self._operators: Dict[str, Union[str, NoArgOperator, TextOperator]] = {}

        if not Doc.has_extension("anonymized"):
            Doc.set_extension("anonymized", default=None)

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
            - a one-argument callable receiving the original text and returning its replacement.
        """
        self._operators.update(operators)

    def clear(self) -> None:
        """Remove all registered operators."""
        self._operators.clear()

    def _get_spans(self, doc: Doc) -> List[Span]:
        if self.style == "ent":
            return list(doc.ents)
        return list(doc.spans.get(self.spans_key, []))

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

    def _make_anonymized_doc(self, doc: Doc) -> Doc:
        spans = sorted(self._get_spans(doc), key=lambda span: span.start_char)
        if not spans:
            return self.nlp.make_doc(doc.text)

        parts, offset, entity_starts, entity_ends, entity_labels = [], 0, [], [], []

        for span in spans:
            parts.append(doc.text[offset : span.start_char])
            replacement = self._apply_operator(span.label_, span.text)
            parts.append(replacement)

            entity_starts.append(len("".join(parts[:-1])))
            entity_ends.append(entity_starts[-1] + len(replacement))
            entity_labels.append(span.label_)

            offset = span.end_char
        parts.append(doc.text[offset:])

        new_text = "".join(parts)
        new_doc = self.nlp.make_doc(new_text)

        entities = [
            new_doc.char_span(start, end, label=label)
            for start, end, label in zip(entity_starts, entity_ends, entity_labels)
            if start is not None and end is not None
        ]
        entities = [entity for entity in entities if entity is not None]

        if self.style == "ent":
            new_doc.ents = entities
        else:
            new_doc.spans[self.spans_key] = entities

        return new_doc

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
        deserializers = {"operators": lambda b: self._operators.update(pickle.loads(b))}
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
        serializers = {"operators": lambda: pickle.dumps(self._operators)}
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
            "operators": lambda p: self._operators.update(pickle.load(open(p, "rb")))
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
        serializers = {"operators": lambda p: pickle.dump(self._operators, open(p, "wb"))}
        util.to_disk(path, serializers, {})