from typing import List, Optional, Dict
from spacy.language import Language
from anonymacy.entities import Entity

class PipelineBuilder:
    """Minimal API every language builder must honour."""
    
    COMPONENTS = ["recognizer", "context_enhancer", "conflict_resolver", "anonymizer"]

    def __init__(self, nlp: Language, disable: List[str] = []):
        self.nlp = nlp
        for component in self.COMPONENTS:
            if component in disable:
                continue
            
            if not nlp.has_pipe(component):
                nlp.add_pipe(component)

    def add_entities(self, entities: List[Entity]):
        recognizer = self._get_pipe("recognizer")
        context_enhancer = self._get_pipe("context_enhancer")
        anonymizer = self._get_pipe("anonymizer")
        
        for entity in entities:
            if recognizer is not None:
                if entity.patterns:
                    recognizer.add_patterns(entity.patterns)
                if entity.custom_matcher:
                    recognizer.add_custom_matchers({entity.label: entity.custom_matcher})
                if entity.validator:
                    recognizer.add_validators({entity.label: entity.validator})

            if context_enhancer is not None:
                context_enhancer.add_patterns(entity.context_patterns)

            if anonymizer is not None:
                anonymizer.add_replacements({entity.label: entity.replacer})

        return self

    def _get_pipe(self, pipe_name: str):
        return self.nlp.get_pipe(pipe_name) if self.nlp.has_pipe(pipe_name) else None

    def build(self) -> Language:
        return self.nlp