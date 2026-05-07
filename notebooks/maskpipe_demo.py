# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "maskpipe",
#     "marimo>=0.23.3",
#     "ipython",
#     "gliner==0.2.26",
#     "gliner[tokenizers]",
#     "hf_xet",
#     "faker==38.2.0",
#     "nl_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.8.0/nl_core_news_sm-3.8.0-py3-none-any.whl",
#     "pyzmq>=27.1.0",
# ]
#
# [tool.uv.sources]
# maskpipe = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from spacy import displacy
    import spacy
    from gliner import GLiNER
    from maskpipe import PipelineBuilder, DocBuilder
    from maskpipe import entities
    from maskpipe.entities import nl
    from faker import Faker

    return (
        DocBuilder,
        Faker,
        GLiNER,
        PipelineBuilder,
        displacy,
        entities,
        mo,
        nl,
        spacy,
    )


@app.cell(hide_code=True)
def _(mo):
    text_area = mo.ui.text_area(
              value="Mijn naam is Anna de Vries en ik wil graag een melding doen over een probleem dat ik heb ervaren bij het Medisch Centrum Amsterdam. Tijdens mijn opname in het ziekenhuis vorig jaar ontving ik een onjuiste behandeling, wat heeft geleid tot ernstige bijwerkingen van het medicijn metoprolol. Ik heb dit meerdere keren besproken met mijn behandelend arts, maar zonder resultaat. Mijn bsnnummer is 692 015 644 en mijn telefoonnummer is 0612345678. Ik ben woonachtig aan de Dorpsstraat 42 in Haarlem en ben verzekerd bij Zorgzaam Verzekeringen. Ik werk zelf als verpleegkundige bij het Woonzorgcentrum De Lentehof, waar ik dagelijks mensen help met dementiezorg. Mijn e-mailadres is anna.devries1980@gmail.com en ik wil erop aandringen dat deze klacht serieus wordt genomen. De communicatie met de organisatie houdt te wensen over. Ik voel mij onvoldoende gehoord en wil dat hier iets aan wordt gedaan. Uiteindelijk wil ik benadrukken dat deze situatie mijn vertrouwen in de zorginstelling aanzienlijk heeft geschaad. geboren 2024-01-15T14:30:45.123Z of de veertiende van januari 2025",
     rows= 10)
    text_area
    return (text_area,)


@app.cell
def _(Faker, GLiNER, PipelineBuilder, entities, nl, spacy):
    model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
    nlp = spacy.load("nl_core_news_sm", disable=["ner"])

    builder = PipelineBuilder(nlp)
    fake = Faker("nl_NL")

    builder.add_entities([
        nl.BSN.replace(redactor=fake.ssn),
        entities.PHONE_NUMBER.replace(redactor=fake.phone_number),
        entities.DATE,
    ])
    return model, nlp


@app.cell
def _(nlp):
    print(nlp.pipe_names)
    return


@app.cell
def _(model, text_area):
    text = text_area.value
    predicted_entities = model.predict_entities(
        text,
        threshold=0.5,
        labels=[
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
    )
    return predicted_entities, text


@app.cell
def _(DocBuilder, nlp, predicted_entities, text):
    doc = (DocBuilder(nlp, text)
        .with_gliner(predicted_entities)
        .build())

    doc = nlp(doc)
    return (doc,)


@app.cell
def _(displacy, doc, mo):
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

    mo.iframe(html, width="100%")
    return


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
