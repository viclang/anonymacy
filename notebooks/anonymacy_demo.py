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
# ]
#
# [tool.uv.sources]
# anonymacy = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from spacy import displacy, Language
    from spacy.tokens import Doc, Span
    import spacy
    from anonymacy import ContextEnhancer, Recognizer, conflict_resolver
    from spacy.util import registry
    from anonymacy import validator, util
    import anonymacy
    from faker import Faker
    return Faker, displacy, mo, spacy, util, validator


@app.cell
def _(Faker, util, validator):
    SPACY_MODEL = "nl_core_news_sm"

    SPACY_CONFIG = {
        "gliner_model": "knowledgator/gliner-x-large",
        # "gliner_model": "E3-JSI/gliner-multi-pii-domains-v1"
        # "gliner_model": "urchade/gliner_multi-v2.1"
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
    }

    RECOGNIZER_PATTERNS = [
        {
            "label": "BSN",
            "score": 0.4,
            "pattern": [{"LENGTH": 9, "IS_DIGIT": True}],
        },
        {
            "label": "BSN",
            "score": 0.3,
            "pattern": [{"LENGTH": 8, "IS_DIGIT": True}],
        },
        {
            "label": "BSN",
            "score": 0.1,
            "pattern": [
                {"SHAPE": "dd"},
                {"TEXT": "."},
                {"SHAPE": "ddd"},
                {"TEXT": "."},
                {"SHAPE": "ddd"},
            ],
        },
        {
            "label": "BSN",
            "score": 0.1,
            "pattern": [
                {"SHAPE": "dd"},
                {"TEXT": "-"},
                {"SHAPE": "ddd"},
                {"TEXT": "-"},
                {"SHAPE": "ddd"},
            ],
        },
        {
            "label": "BSN",
            "score": 0.1,
            "pattern": [
                {"SHAPE": "dd"},
                {"IS_SPACE": True},
                {"SHAPE": "ddd"},
                {"IS_SPACE": True},
                {"SHAPE": "ddd"},
            ],
        },
    ]

    CONTEXT_PATTERNS = [
        {
            "label": "BSN",
            "pattern": [
                {
                    "LEMMA": {
                        "IN": [
                            "bsn",
                            "bsnnummer",
                            "bsn-nummer",
                            "burgerservice",
                            "burgerservicenummer",
                            "sofinummer",
                            "sofi-nummer",
                        ]
                    }
                }
            ],
        },
        {
            "label": "PHONE_NUMBER",
            "pattern": [
                {
                    "LEMMA": {
                        "IN": [
                            "telefoon",
                            "mobiel",
                            "telefoonnummer",
                            "bellen",
                            "mobiele telefoon",
                        ]
                    }
                }
            ],
        },
    ]

    validators = util.register_validators({
        "BSN" : validator.elf_proef,
    })

    fake = Faker("nl_NL")
    operators = util.register_operators({
        "persoon" : fake.name
    })
    return (
        CONTEXT_PATTERNS,
        RECOGNIZER_PATTERNS,
        SPACY_CONFIG,
        SPACY_MODEL,
        operators,
        validators,
    )


@app.cell(hide_code=True)
def _(mo):
    text_area = mo.ui.text_area(
        value="Mijn naam is Anna de Vries en ik wil graag een melding doen over een probleem dat ik heb ervaren bij het Medisch Centrum Amsterdam. Tijdens mijn opname in het ziekenhuis vorig jaar ontving ik een onjuiste behandeling, wat heeft geleid tot ernstige bijwerkingen van het medicijn metoprolol. Ik heb dit meerdere keren besproken met mijn behandelend arts, maar zonder resultaat. Mijn bsnnummer is 123456789 en mijn telefoonnummer is 0612345678. Ik ben woonachtig aan de Dorpsstraat 42 in Haarlem en ben verzekerd bij Zorgzaam Verzekeringen. Ik werk zelf als verpleegkundige bij het Woonzorgcentrum De Lentehof, waar ik dagelijks mensen help met dementiezorg. Mijn e-mailadres is anna.devries1980@gmail.com en ik wil erop aandringen dat deze klacht serieus wordt genomen. De communicatie met de organisatie houdt te wensen over. Ik voel mij onvoldoende gehoord en wil dat hier iets aan wordt gedaan. Uiteindelijk wil ik benadrukken dat deze situatie mijn vertrouwen in de zorginstelling aanzienlijk heeft geschaad.",
        rows= 10)
    text_area
    return (text_area,)


@app.cell
def _(
    CONTEXT_PATTERNS,
    RECOGNIZER_PATTERNS,
    SPACY_CONFIG,
    SPACY_MODEL,
    displacy,
    mo,
    operators,
    spacy,
    text_area,
    validators,
):
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
    nlp.add_pipe("gliner_spacy", config=SPACY_CONFIG)
    recognizer = nlp.add_pipe("recognizer", config={
        "validators": validators,
    })
    recognizer.add_patterns(RECOGNIZER_PATTERNS)

    context_enhancer = nlp.add_pipe("context_enhancer")
    context_enhancer.add_patterns(CONTEXT_PATTERNS)
    #nlp.to_disk("./recognizers")

    nlp2 = spacy.load("./recognizers")
    print(nlp2.pipe_names)
    recognizer2 = nlp2.get_pipe("recognizer")
    print(recognizer2.validators)
    nlp.add_pipe("conflict_resolver")

    anonymizer = nlp.add_pipe("anonymizer")
    anonymizer.add_operators(operators)

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
