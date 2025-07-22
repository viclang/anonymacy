import sys
import os
from pathlib import Path
from spacy import Language
from spacy.lang.nl import Dutch
from spacy.pipeline import SpanRuler
from spacy.tokens import Span
from spacy import displacy
import spacy

from anonymacy import ContextEnhancer
from anonymacy.recognizer import PatternRecognizer, BsnRecognizer

custom_spacy_config = {
    "gliner_model": "E3-JSI/gliner-multi-pii-domains-v1",
    "chunk_size": 250,
    "threshold": 0.5,
    "labels": [
        "persoon",
        "organisatie",
        "email",
        "adres",
        "klacht",
        "medicijn",
        "ziekenhuis",
        "zorginstelling",
        "beroep",
        "verzekering"
    ],
    "style": "span",
    "map_location": "cpu"
}



nlp = spacy.load("nl_core_news_sm", disable=["ner"])
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)
nlp.add_pipe("bsn_recognizer")
nlp.add_pipe("phone_recognizer")

enhancer_config = {
    "patterns": [
        {"label": "bsn", "pattern": [{"LOWER": "bsn"}] }
    ]
}
enhancer = nlp.add_pipe("context_enhancer", config=enhancer_config)

nlp.add_pipe("conflict_resolver", last=True)

anonymizer_config = {
    "operators": {
        "bsn": "FakeBSN"
    }
}
    
anonymizer = nlp.add_pipe("anonymizer", config=anonymizer_config)

text = "1982g bij4za#a april 23e De heer De Vries met bsn 376174316 en telefoonnummer 0612345678 had vandaag een beetje last van duizeligheid na het innemen van zijn medicatie (Metoprolol 50mg) om 10:00."
doc = nlp(text)

if "sc" in doc.spans:
    spans = doc.spans["sc"]
else:
    spans = doc.ents

# Print span text, label, and score
for span in spans:
    print(f"Text: {span.text}, Label: {span.label_}, Score: {span._.score:.2f}")
    if span._.supportive_context:
        print(f"Context found: {span._.supportive_context}")