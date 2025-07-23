# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anonymacy==0.0.1",
#     "gliner-spacy==0.0.11",
#     "nl-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.8.0/nl_core_news_sm-3.8.0-py3-none-any.whl",
#     "spacy==3.8.7",
# ]
#
# [tool.uv.sources]
# anonymacy = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os
    from pathlib import Path
    import marimo as mo
    from spacy import Language
    from spacy.lang.nl import Dutch
    from spacy.pipeline import SpanRuler
    from spacy.tokens import Span
    from spacy import displacy
    import spacy

    from anonymacy import ContextEnhancer
    from anonymacy.recognizer import PatternRecognizer, BsnRecognizer
    return displacy, mo, spacy


@app.cell(hide_code=True)
def _(mo):
    text_area = mo.ui.text_area(value="1982g bij4za#a april 23e De heer De Vries met 376174316 bsn met telefoon 0612345678 had vandaag een beetje last van duizeligheid na het innemen van zijn medicatie (Metoprolol 50mg) om 10:00. Bloeddruk gemeten om 11:00: 88/47. Arts geïnformeerd. Het advies is om medicatie in de middag te herhalen en te controleren op mogelijk onderliggende oorzaken. Cliënt gaf aan dat hij dit eerder heeft ervaren na het aanpassen van zijn dieet. Mevrouw Janssen klaagde vanmiddag opnieuw over hevige buikpijn. Dit is de derde keer in deze maand dat ze deze klachten meldt. Er is een afspraak gemaakt voor een echografie op 21-09-2023 in het St. Antonius Ziekenhuis. Haar partner, Peter Bakker, werd op de hoogte gebracht van de situatie en gaf aan bij de afspraak aanwezig te willen zijn. De heer Van der Zee heeft vanochtend tijdens de groepsessie aangegeven dat hij zich emotioneel onstabiel voelt sinds zijn ontslag uit de verslavingskliniek. Hij gaf aan last te hebben van terugvalverlangens richting alcohol. Een crisisinterventie werd telefonisch gepland met zijn verslavingscoach. Aangepaste medicatie voorgesteld door de psychiater, overleg volgt.", rows= 10)
    text_area
    return (text_area,)


@app.cell
def _(displacy, mo, spacy, text_area):
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

    nlp.add_pipe("conflict_resolver")


    anonymizer_config = {
        "operators": {
            "bsn": "FakeBSN"
        }
    }

    anonymizer = nlp.add_pipe("anonymizer", config=anonymizer_config)

    text = text_area.value
    doc = nlp(text)

    html = displacy.render(
        doc,
        style="ent",
        options={
            "colors": {
                "persoon": "#ffcccc",
                "organisatie": "#ccffcc",
                "email": "#ccccff",
                "adres": "#ffffcc",
                "klacht": "#ffccff",
                "medicijn": "#ccffff",
                "ziekenhuis": "#ffcce5",
                "zorginstelling": "#e5ccff",
                "beroep": "#ffe5cc",
                "verzekering": "#e5ffe5",
                "bsn": "#d9d9d9"
            }
        }
    )

    mo.iframe(html, height=400)
    return (doc,)


@app.cell
def _(doc):
    print(doc._.anonymized_text)
    return


@app.cell
def _(doc):
    print(doc._.anonymized_spans)
    return


@app.cell
def _(doc):
    if "sc" in doc.spans:
        spans = doc.spans["sc"]
    else:
        spans = doc.ents

    # Print span text, label, and score
    for span in spans:
        print(f"Text: {span.text}, Label: {span.label_}, Score: {span._.score:.2f}")
        if span._.supportive_context:
            print(f"BSN Found: {span._.supportive_context}")

    return


if __name__ == "__main__":
    app.run()
