from typing import List, Dict, Any, cast, Optional
from spacy.language import Language
from anonymacy.entities import Entity
from anonymacy import Recognizer, ContextEnhancer, Anonymizer, ConflictResolver

class PipelineBuilder:
    """Minimal API every language builder must honour."""
    
    def __init__(self, nlp: Language, disable: Optional[List[str]] = None):
        self.components = ["recognizer", "context_enhancer", "conflict_resolver", "anonymizer"]
        self.nlp = nlp
        disabled = set(disable) if disable else set()
        self.components = [c for c in self.components if c not in disabled]

        for component in disabled:
            if nlp.has_pipe(component):
                nlp.remove_pipe(component)

        for component in self.components:
            if not nlp.has_pipe(component):
                nlp.add_pipe(component)
        
    def add_entities(self, entities: List[Entity]):
        """Partition entities and add their patterns, matchers, and redactors to relevant components."""
        
        batch = self._partition_entities_for_components(entities)
        if "recognizer" in self.components:
            recognizer = cast(Recognizer, self.nlp.get_pipe("recognizer"))
            recognizer.add_patterns(batch["recognizer"]["patterns"])
            recognizer.add_custom_matchers(batch["recognizer"]["custom_matchers"])
            recognizer.add_validators(batch["recognizer"]["validators"])

        if "context_enhancer" in self.components:
            context_enhancer = cast(ContextEnhancer, self.nlp.get_pipe("context_enhancer"))
            context_enhancer.add_patterns(batch["context_enhancer"]["patterns"])
        
        if "anonymizer" in self.components:
            anonymizer = cast(Anonymizer, self.nlp.get_pipe("anonymizer"))
            anonymizer.add_redactors(batch["anonymizer"]["redactors"])

    def _partition_entities_for_components(self, entities: List[Entity]) -> Dict[str, Dict[str, Any]]:
        batch = {
            "recognizer": {
                "patterns": [],
                "custom_matchers": {},
                "validators": {}
            },
            "context_enhancer": {
                "patterns": []
            },
            "anonymizer": {
                "redactors": {}
            }
        }
        for entity in entities:
            if "recognizer" in self.components:
                if entity.patterns:
                    patterns_with_label = [
                        {**pattern, "label": entity.label}
                        for pattern in entity.patterns
                    ]
                    batch["recognizer"]["patterns"].extend(patterns_with_label)
                
                if entity.custom_matcher:
                    batch["recognizer"]["custom_matchers"][entity.label] = entity.custom_matcher
            
                if entity.validator:
                    batch["recognizer"]["validators"][entity.label] = entity.validator
        
            if "context_enhancer" in self.components:                
                if entity.context_patterns:
                    context_patterns_with_label = [
                        {**pattern, "label": entity.label}
                        for pattern in entity.context_patterns
                    ]
                    batch["context_enhancer"]["patterns"].extend(context_patterns_with_label)
            
            if "anonymizer" in self.components:
                if entity.redactor:
                    batch["anonymizer"]["redactors"][entity.label] = entity.redactor    
            
        return batch

    def build(self) -> Language:
        return self.nlp