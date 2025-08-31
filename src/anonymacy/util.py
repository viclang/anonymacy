from spacy import registry
from spacy.tokens import Doc, Span
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Union,
)

def register_validators(validators: Dict[str, Callable[[Span], bool]]) -> Dict[str, Union[str, List[str]]]:
    """
    Register each span‐validator in spaCy's misc registry and
    return a config‐friendly mapping of label → {"@misc": key}.
    """
    return _register_misc("anonymacy.validators", validators)

def register_operators(operators: Dict[str, Union[str, Callable[[], str], Callable[[str], str]]]) -> Dict[str, Union[str, List[str]]]:
    """
    Register each span‐operator in spaCy's misc registry and
    return a config‐friendly mapping of label → {"@misc": key}.
    """
    return _register_misc("anonymacy.operators", operators)

def register_matchers(matchers: Dict[str, Callable[[Doc], List[Span]]]) -> Dict[str, Union[str, List[str]]]:
    """
    Register each span‐matcher in spaCy's misc registry and
    return a config‐friendly mapping of label → {"@misc": key}.
    """
    return _register_misc("anonymacy.matchers", matchers)

def _register_misc(key: str, mapping: Dict[str, Any]) -> Dict[str, Union[str, List[str]]]:
    """
    Register a misc entry by label.
    """
    store: Dict[str, Dict[str, str]] = dict(mapping)
    
    @registry.misc(key)
    def _factory(labels: List[str]):
        missing = [label for label in labels if label not in store]
        if missing:
            raise KeyError(
                f"The following validator label(s) are not registered in "
                f"the misc registry under '{key}': {missing!r}"
            )
        return {label: store[label] for label in labels}

    return {"@misc": key, "labels": list(store.keys()) }