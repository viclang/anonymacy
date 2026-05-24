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
#     "pandas>=3.0.3",
# ]
#
# [tool.uv.sources]
# maskpipe = { path = "../", editable = true }
#
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    from maskpipe import PipelineBuilder, StructuredAnalyzer, GLINER_MAPPER
    from maskpipe import entities
    from maskpipe.entities import nl
    import spacy
    from gliner import GLiNER

    return (
        GLINER_MAPPER,
        GLiNER,
        PipelineBuilder,
        StructuredAnalyzer,
        pd,
        spacy,
    )


@app.cell
def _(GLiNER, PipelineBuilder, spacy):
    model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
    nlp = spacy.load("nl_core_news_sm", disable=["ner"])
    PipelineBuilder(nlp, disable=["anonymizer"]).build()
    return model, nlp


@app.cell
def _(nlp):
    print(nlp.pipe_names)
    return


@app.cell
def _(GLINER_MAPPER, StructuredAnalyzer, model, nlp, pd):
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": ["25", "30", "35"],
            "city": ["New York", "Los Angeles", "Chicago"],
        }
    )
    analyzer = StructuredAnalyzer(nlp)

    results = analyzer.analyze(
        df.to_dict(orient='list'),
        batch_extractor=lambda texts: model.inference(
            texts,
            batch_size=3,
            threshold=0.5,
            labels=[
                "naam",
                "leeftijd",
                "stad",
            ],
        ),
        entity_mapper=GLINER_MAPPER,
    )
    results
    return


if __name__ == "__main__":
    app.run()
