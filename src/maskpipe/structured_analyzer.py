import random
from collections import Counter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypedDict,
)
from spacy.language import Language
from .doc_builder import DocBuilder
from .entity_mapper import BaseEntityMapper, EntityResult

DOC_ANALYZER_DEFAULT_SPANS_KEY = "sc"

class ColumnAnalysis(TypedDict):
    label: str
    coverage: float
    score: float
    entity_distribution: Dict[str, int]
    entities: List[EntityResult]

class StructuredAnalyzer:

    def __init__(
        self,
        nlp: Language,
        *,
        label_mapping: Optional[Dict[str, str]] = None,
        spans_key: Optional[str] = DOC_ANALYZER_DEFAULT_SPANS_KEY,
        annotate_ents: bool = False,
        default_score: float = 0.6,
    ):
        self.nlp = nlp
        self.label_mapping = label_mapping
        self.spans_key = spans_key
        self.annotate_ents = annotate_ents
        self.default_score = default_score

    def analyze(
        self,
        data: Dict[str, List[Any]],
        n: Optional[int] = None,
        batch_extractor: Optional[Callable[[List[str]], List[Any]]] = None,
        entity_mapper: Optional[BaseEntityMapper] = None,
        alignment_mode: str = "strict",
    ) -> Dict[str, ColumnAnalysis]:
        if not data:
            return {}

        if n is None:
            first_values = next(iter(data.values()))
            if not first_values:
                return {}
            n = len(first_values)

        results: Dict[str, ColumnAnalysis] = {}

        for column, values in data.items():
            sample_size = min(n, len(values))
            sampled_values = random.sample(values, sample_size)

            extracted = (
                batch_extractor(sampled_values) if batch_extractor else None
            )
            docs = DocBuilder.build_batch(
                nlp=self.nlp,
                texts=sampled_values,
                context_words=[column],
                entities_list=extracted,
                entity_mapper=entity_mapper,
                alignment_mode=alignment_mode,
                label_mapping=self.label_mapping,
                spans_key=self.spans_key,
                annotate_ents=self.annotate_ents,
                default_score=self.default_score,
            )

            all_entities: List[EntityResult] = []
            cell_labels: List[str] = []

            for doc in self.nlp.pipe(docs):
                cell_label, doc_entities = self._process_doc(doc)
                cell_labels.append(cell_label)
                all_entities.extend(doc_entities)

            column_analysis = self._classify_column(cell_labels, all_entities)
            results[column] = column_analysis

        return results

    def _process_doc(self, doc) -> tuple[str, List[EntityResult]]:
        """Single-pass extraction of cell label and entities."""
        
        if not doc.ents:
            return "NON_PII", []

        unique_labels: set[str] = set()
        entities: List[EntityResult] = []

        for ent in doc.ents:
            unique_labels.add(ent.label_)
            entities.append(
                EntityResult(
                    start=ent.start_char,
                    end=ent.end_char,
                    label=ent.label_,
                    score=(
                        ent._.score
                        if ent.has_extension("score")
                        else self.default_score
                    ),
                )
            )

        cell_label = (
            "PII" if len(unique_labels) > 1 else next(iter(unique_labels))
        )

        return cell_label, entities

    def _classify_column(self, cell_labels: List[str], all_entities: List[EntityResult]) -> ColumnAnalysis:
        """Most common non-NON_PII label; confidence over all cells."""
        if not cell_labels:
            return ColumnAnalysis(
                label="NON_PII",
                coverage=0.0,
                score=0.0,
                entity_distribution=dict(),
                entities=[],
            )

        label_counts = Counter(cell_labels)
        total_cells = len(cell_labels)

        if label_counts.get("NON_PII", 0) == total_cells:
            return ColumnAnalysis(
                label="NON_PII",
                coverage=1.0,
                score=1.0,
                entity_distribution=dict(),
                entities=[],
            )

        candidate_counts = label_counts.copy()
        candidate_counts.pop("NON_PII", None)

        (column_label, winner_count), = candidate_counts.most_common(1)
        coverage = winner_count / total_cells

        winning_label_scores = [
            e['score'] for e in all_entities 
            if e['label'] == column_label
        ]
        score = (
            sum(winning_label_scores) / len(winning_label_scores) 
            if winning_label_scores else 0.0
        )

        return ColumnAnalysis(
                label=column_label,
                coverage=coverage,
                score=score,
                entity_distribution=dict(
                    Counter(e['label'] for e in all_entities)
                ),
                entities=all_entities,
            )