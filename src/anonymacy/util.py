from spacy.util import registry as spacy_registry
from spacy.tokens import Doc, Span
from spacy.util import catalogue
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Union,
)
spacy_registry.ensure_populated()

class AnonymacyRegistry(spacy_registry):
    
    @classmethod
    def matchers(cls, matchers: Dict[str, Callable[[Doc], List[Span]]]) -> Dict[str, Union[str, List[str]]]:
        """
        Register each span‐matcher in spaCy's misc registry and
        return a config‐friendly mapping of label → {"@misc": key}.
        """
        return AnonymacyRegistry._register_misc("anonymacy.matchers", matchers)

    @classmethod
    def validators(cls, validators: Dict[str, Callable[[Span], bool]]) -> Dict[str, Union[str, List[str]]]:
        """
        Register each span‐validator in spaCy's misc registry and
        return a config‐friendly mapping of label → {"@misc": key}.
        """
        return AnonymacyRegistry._register_misc("anonymacy.validators", validators)

    @classmethod
    def operators(cls, operators: Dict[str, Union[str, Callable[[], str], Callable[[str], str]]]) -> Dict[str, Union[str, List[str]]]:
        """
        Register each span‐operator in spaCy's misc registry and
        return a config‐friendly mapping of label → {"@misc": key}.
        """
        return AnonymacyRegistry._register_misc("anonymacy.operators", operators)

    @staticmethod
    def _register_misc(key, mapping):
        store=dict(mapping)

        @spacy_registry.misc(key)
        def _factory(labels: List[str]|None=None):
            if labels is None:
                return store
            missing = [label for label in labels if label not in store]
            if missing:
                raise KeyError(
                    f"The following validator label(s) are not registered in "
                    f"the misc registry under '{key}': {missing!r}"
                )
            return { label: store[label] for label in labels }

        return {"@misc": key, "labels": list(store.keys())}

registry = AnonymacyRegistry
__all__ = ["registry"]