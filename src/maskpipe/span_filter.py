import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List

from spacy import registry
from spacy.tokens import Span

DEFAULT_HIERARCHY = {
    'date': ['date_of_birth', 'date_time'],
    'name': ['first_name', 'last_name', 'full_name'],
    'phone': ['phone_number', 'fax_number', 'mobile_number'],
    'address': ['street_address', 'home_address', 'billing_address'],
    'ip': ['ipv4', 'ipv6'],
    'id': ['ssn', 'bsn', 'medical_record_number', 'account_number', 'employee_id'],
    'national_id': ['nir', 'insee', 'steuer_id', 'steuernummer', 'codice_fiscale'],
}

@dataclass(frozen=True)
class HierarchicalMergeFilter:
    """A spans filter that owns its hierarchy.

    Callable signature: (ents, spans) -> spans  — fully compatible.
    But hierarchy is carried as object state, not as a call parameter.
    """

    hierarchy: Dict[str, List[str]]

    def __call__(
        self, *spans: Iterable[Span]
    ) -> Iterable[Span]:
        label_to_group: Dict[str, str] = {}
        for parent, children in self.hierarchy.items():
            label_to_group[parent.lower()] = parent.lower()
            label_to_group.update({child.lower(): parent.lower() for child in children})
        
        
        return self._merge_hierarchical(*spans, label_to_group=label_to_group)

    def _merge_hierarchical(
        self,
        *inputs: Iterable[Span],
        label_to_group: Dict[str, str],
    ) -> List[Span]:
        """Filter and merge overlapping spans respecting label hierarchy.

        Args:
            label_to_group: Precomputed mapping of label -> parent group
            inputs: One or more iterables of potentially overlapping spans

        Returns:
            List of merged, non-conflicting spans sorted by start position
        """
        spans = list(itertools.chain(*inputs))
        if not spans:
            return []

        doc = spans[0].doc

        spans.sort(
            key=lambda s: (getattr(s._, "score", 0.0), s.end - s.start, -s.start),
            reverse=True,
        )

        token_group: Dict[int, str] = {}
        accepted: List[Span] = []

        for span in spans:
            group = label_to_group.get(span.label_.lower(), span.label_.lower())
            has_conflict = any(
                t in token_group and token_group[t] != group
                for t in range(span.start, span.end)
            )

            if not has_conflict:
                for t in range(span.start, span.end):
                    token_group[t] = group
                accepted.append(span)

        if len(accepted) <= 1:
            return accepted

        accepted.sort(key=lambda s: s.start)

        merged: List[Span] = [accepted[0]]

        for current in accepted[1:]:
            previous = merged[-1]
            prev_label = previous.label_.lower()
            curr_label = current.label_.lower()
            prev_group = label_to_group.get(prev_label, prev_label)
            curr_group = label_to_group.get(curr_label, curr_label)

            is_same_group = prev_group == curr_group
            is_adjacent_or_overlapping = current.start <= previous.end

            if is_adjacent_or_overlapping and is_same_group:
                curr_is_more_specific = curr_label != curr_group
                prev_is_more_specific = prev_label != prev_group

                curr_priority = (curr_is_more_specific, getattr(current._, "score", 0.0))
                prev_priority = (prev_is_more_specific, getattr(previous._, "score", 0.0))

                winner = current if curr_priority > prev_priority else previous

                merged_span = Span(
                    doc,
                    previous.start,
                    max(previous.end, current.end),
                    label=winner.label_,
                )
                merged_span._.score = getattr(winner._, "score", 0.0)
                merged[-1] = merged_span
            else:
                merged.append(current)

        return merged

@registry.misc("maskpipe.hierarchical_merge_filter.v1")
def make_hierarchical_merge_filter(
    hierarchy: Dict[str, List[str]] = DEFAULT_HIERARCHY,
) -> HierarchicalMergeFilter:
    return HierarchicalMergeFilter(hierarchy=hierarchy)

hierarchical_merge_filter = HierarchicalMergeFilter(hierarchy=DEFAULT_HIERARCHY)