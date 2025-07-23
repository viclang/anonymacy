# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "anonymacy==0.0.2",
#     "gliner-spacy==0.0.11",
#     "nl-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.8.0/nl_core_news_sm-3.8.0-py3-none-any.whl",
#     "spacy==3.8.7",
# ]
#
# [tool.uv.sources]
# anonymacy = { path = "../", editable = true }
# ///

import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from spacy import displacy
    import spacy

    return displacy, mo, spacy


@app.cell(hide_code=True)
def _(mo):
    text_area = mo.ui.text_area(
        value="Mijn naam is Anna de Vries en ik wil graag een melding doen over een probleem dat ik heb ervaren bij het Medisch Centrum Amsterdam. Tijdens mijn opname in het ziekenhuis vorig jaar ontving ik een onjuiste behandeling, wat heeft geleid tot ernstige bijwerkingen van het medicijn metoprolol. Ik heb dit meerdere keren besproken met mijn behandelend arts, maar zonder resultaat. Mijn bsn is 376174316 en mijn telefoonnummer is 0612345678. Ik ben woonachtig aan de Dorpsstraat 42 in Haarlem en ben verzekerd bij Zorgzaam Verzekeringen. Ik werk zelf als verpleegkundige bij het Woonzorgcentrum De Lentehof, waar ik dagelijks mensen help met dementiezorg. Mijn e-mailadres is anna.devries1980@gmail.com en ik wil erop aandringen dat deze klacht serieus wordt genomen. De communicatie met de organisatie houdt te wensen over. Ik voel mij onvoldoende gehoord en wil dat hier iets aan wordt gedaan. Uiteindelijk wil ik benadrukken dat deze situatie mijn vertrouwen in de zorginstelling aanzienlijk heeft geschaad.",
        rows= 10)
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

    nlp.add_pipe("anonymizer", config=anonymizer_config)

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
            print(f"Context Found: {span._.supportive_context}")

    return


if __name__ == "__main__":
    app.run()
