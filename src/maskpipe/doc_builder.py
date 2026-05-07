from collections.abc import Iterator
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from spacy.language import Language
from spacy.tokens import Doc, Span

DOC_BUILDER_DEFAULT_SPANS_KEY = "sc"

class DocBuilder:
    """Utility for building spacy docs with entity context for anonymacy pipelines."""
    def __init__(
        self,
        nlp: Language,
        text: str,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        spans_key: Optional[str] = DOC_BUILDER_DEFAULT_SPANS_KEY,
        annotate_ents: bool = False,
        default_score: float = 0.6,
        alignment_mode: str = "strict",
    ):
        self.nlp = nlp
        self.doc = self.nlp._ensure_doc(text)
        self.spans_key = spans_key
        self.annotate_ents = annotate_ents
        self.label_mapping = label_mapping if label_mapping else {}
        self.default_score = default_score
        self.alignment_mode = alignment_mode

    @classmethod
    def build_batch_with_hf_ner(
        cls,
        nlp: Language,
        texts: List[str],
        entities_list: List[List[Dict[str, Any]]],
        **builder_kwargs
    ) -> Iterator[Doc]:
        """Build batch of docs with HuggingFace NER entities."""
        for text, entities in zip(texts, entities_list):
            yield cls(nlp, text, **builder_kwargs).with_hf_ner(entities).build()

    @classmethod
    def build_batch_with_gliner(
        cls,
        nlp: Language,
        texts: List[str],
        entities_list: List[List[Dict[str, Any]]],
        **builder_kwargs
    ) -> Iterator[Doc]:
        """Build batch of docs with GLiNER entities."""
        for text, entities in zip(texts, entities_list):
            yield cls(nlp, text, **builder_kwargs).with_gliner(entities).build()

    @classmethod
    def build_batch_with_gliner2(
        cls,
        nlp: Language,
        texts: List[str],
        entities_list: List[Dict[str, Dict[str, Any]]],
        **builder_kwargs
    ) -> Iterator[Doc]:
        """Build batch of docs with GLiNER2 entities."""
        for text, entities in zip(texts, entities_list):
            yield cls(nlp, text, **builder_kwargs).with_gliner2(entities).build()

    @classmethod
    def build_batch_with_openmed(
        cls,
        nlp: Language,
        texts: List[str],
        entities_list: List[List[Dict[str, Any]]],
        **builder_kwargs
    ) -> Iterator[Doc]:
        """Build batch of docs with OpenMed entities."""
        for text, entities in zip(texts, entities_list):
            yield cls(nlp, text, **builder_kwargs).with_openmed(entities).build()

    def with_context_words(self, context_words: List[str]) -> "DocBuilder":
        """Add _context to the doc with the provided context words."""
        self.doc._.context_words = context_words
        return self

    def with_hf_ner(self, result: List[Dict[str, Any]]) -> "DocBuilder":
        spans = self._create_spans_from_entities(result, label_key="entity", score_key="score")
        return self._apply_spans(spans)

    def with_gliner(self, result: List[Dict[str, Any]]) -> "DocBuilder":
        spans = self._create_spans_from_entities(result, label_key="label", score_key="score")
        return self._apply_spans(spans)

    def with_gliner2(self, result: Union[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Any]]]) -> "DocBuilder":
        if "entities" in result:
            result = result["entities"]

        entities = [
            {"start": entity["start"], "end": entity["end"], "label": label, "confidence": entity.get("confidence")}
            for label, entity in result.items()
        ]
        spans = self._create_spans_from_entities(entities, label_key="label", score_key="confidence")
        return self._apply_spans(spans)

    def with_openmed(self, result: List[Dict[str, Any]]) -> "DocBuilder":
        spans = self._create_spans_from_entities(result, label_key="label", score_key="confidence")
        return self._apply_spans(spans)

    def _create_spans_from_entities(
        self,
        entities: List[Dict[str, Any]],
        label_key: str,
        score_key: str,
    ) -> List[Span]:
        """Private helper: create spans from normalized entities."""
        spans = []
        for entity in entities:
            label = self.label_mapping.get(entity[label_key], entity[label_key])
            
            span = self.doc.char_span(
                entity["start"],
                entity["end"],
                label=label,
                alignment_mode=self.alignment_mode
            )
            if span is None:
                continue

            span._.score = min(float(entity.get(score_key, self.default_score)), 1.0)
            spans.append(span)
        return spans

    def _apply_spans(self, spans: List[Span]) -> "DocBuilder":
        """Private helper: apply spans to doc."""
        if spans:
            if self.spans_key:
                all_spans = list(self.doc.spans.get(self.spans_key, []))
                all_spans.extend(spans)
                self.doc.spans[self.spans_key] = all_spans
            if self.annotate_ents:
                all_ents = list(self.doc.ents)
                all_ents.extend(spans)
                self.doc.ents = tuple(all_ents)
        return self

    def build(self) -> Doc:
        """Return the built Doc with spans."""
        return self.doc