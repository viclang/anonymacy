from typing import List
from spacy.language import Language
from anonymacy.recognizer import PatternRecognizer, Pattern
from anonymacy.context_enhancer import ContextPattern

@Language.factory("bsn_recognizer")
class BsnRecognizer(PatternRecognizer):
    """Custom spaCy pipeline component for recognizing BSN numbers."""
    
    PATTERNS: List[Pattern] = [
        { "label" : "BSN", "score" : 0.4, "pattern": [{"LENGTH" : 9, "IS_DIGIT" : True}] },
        { "label": "BSN", "score": 0.3, "pattern": [{"LENGTH": 8, "IS_DIGIT": True}] },
        { "label": "BSN", "score": 0.1, "pattern": [
                {"SHAPE": "dd"}, {"TEXT": "."}, {"SHAPE": "ddd"}, {"TEXT": "."}, {"SHAPE": "ddd"}] },
        { "label": "BSN", "score": 0.1, "pattern": [
                {"SHAPE": "dd"}, {"TEXT": "-"}, {"SHAPE": "ddd"}, {"TEXT": "-"}, {"SHAPE": "ddd"}] },
        { "label": "BSN", "score": 0.1, "pattern": [
                {"SHAPE": "dd"}, {"IS_SPACE": True}, {"SHAPE": "ddd"}, {"IS_SPACE": True}, {"SHAPE": "ddd"}] },
    ]
    
    CONTEXT_PATTERNS = [
        {
            "label": "BSN",
            "pattern": [ {"LEMMA": { "IN": [
                "bsn",
                "bsnnummer",
                "bsn-nummer",
                "burgerservice",
                "burgerservicenummer",
                "sofinummer",
                "sofi-nummer",
            ] } } ],
        }
    ]

    def __init__(
        self,
        nlp: Language,
        name: str = "bsn_recognizer",
        patterns: List[Pattern] = None,
        context_patterns: List[ContextPattern] = None,
        style: str = "span",
        spans_key: str = "sc",
        allow_overlap: bool = True,
        conflict_strategy: str = "highest_confidence"
    ):
        patterns = patterns if patterns else self.PATTERNS
        context_patterns = context_patterns if context_patterns else self.CONTEXT_PATTERNS
        super().__init__(
            nlp=nlp,
            name=name,
            patterns=patterns,
            context_patterns=context_patterns,
            style=style,
            spans_key=spans_key,
            allow_overlap=allow_overlap,
            conflict_strategy=conflict_strategy
        )

    def _invalidate_result(self, pattern_text: str, label: str = None) -> bool:
        """
        BSN is invalid if it doesn't pass the 11-proef
        """
        only_digits = "".join(c for c in pattern_text if c.isdigit())
        if all(only_digits[0] == c for c in only_digits):
            # cannot be all same digit
            return True

        if len(only_digits) == 8:
            only_digits = "0" + only_digits

        if len(only_digits) != 9:
            return True

        # 11-proef
        total = 0
        for char, factor in zip(only_digits, [9, 8, 7, 6, 5, 4, 3, 2, -1]):
            total += int(char) * factor

        return total % 11 != 0