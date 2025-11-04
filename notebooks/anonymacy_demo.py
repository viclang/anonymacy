# /// script
# requires-python = ">=3.9,<3.14"
# dependencies = [
#     "anonymacy",
#     "marimo",
#     "gliner-spacy",
#     "gliner[tokenizers]",
#     "hf_xet",
#     "nl-core-news-sm @ https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.8.0/nl_core_news_sm-3.8.0-py3-none-any.whl",
#     "spacy",
#     "faker==37.6.0",
#     "phonenumbers==9.0.15",
# ]
#
# [tool.uv.sources]
# anonymacy = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from spacy import displacy, Language
    from spacy.tokens import Doc, Span
    import spacy
    from anonymacy import PipelineBuilder
    from anonymacy.entities import nl
    from faker import Faker
    return Faker, PipelineBuilder, displacy, mo, nl, spacy


@app.cell(hide_code=True)
def _(mo):
    text_area = mo.ui.text_area(
        value="Mijn naam is Anna de Vries en ik wil graag een melding doen over een probleem dat ik heb ervaren bij het Medisch Centrum Amsterdam. Tijdens mijn opname in het ziekenhuis vorig jaar ontving ik een onjuiste behandeling, wat heeft geleid tot ernstige bijwerkingen van het medicijn metoprolol. Ik heb dit meerdere keren besproken met mijn behandelend arts, maar zonder resultaat. Mijn bsnnummer is 692 015 644 en mijn telefoonnummer is 0612345678. Ik ben woonachtig aan de Dorpsstraat 42 in Haarlem en ben verzekerd bij Zorgzaam Verzekeringen. Ik werk zelf als verpleegkundige bij het Woonzorgcentrum De Lentehof, waar ik dagelijks mensen help met dementiezorg. Mijn e-mailadres is anna.devries1980@gmail.com en ik wil erop aandringen dat deze klacht serieus wordt genomen. De communicatie met de organisatie houdt te wensen over. Ik voel mij onvoldoende gehoord en wil dat hier iets aan wordt gedaan. Uiteindelijk wil ik benadrukken dat deze situatie mijn vertrouwen in de zorginstelling aanzienlijk heeft geschaad.",
        rows= 10)
    text_area
    return (text_area,)


@app.cell
def _(Faker, PipelineBuilder, displacy, mo, nl, spacy, text_area):

    nlp = spacy.load("nl_core_news_sm", disable=["ner"])

    nlp.add_pipe("gliner_spacy", config={
        "gliner_model": "knowledgator/gliner-x-large",
        # "gliner_model": "E3-JSI/gliner-multi-pii-domains-v1",
        # "gliner_model": "urchade/gliner_multi-v2.1",
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
            "verzekering",
        ],
        "style": "span",
        "map_location": "cpu",
    })

    builder = PipelineBuilder(nlp)
    fake = Faker("nl_NL")
    builder.add_entities([
        nl.BSN.replace(replacer=fake.ssn),
        nl.PHONE_NUMBER.replace(replacer=fake.phone_number)
    ])

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
    print(doc._.anonymized)
    return


@app.cell
def _(doc):
    doc._.anonymized.ents
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
        if span._.context:
            print(f"Context Found: {span._.context}")
    return


if __name__ == "__main__":
    app.run()
