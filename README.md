# MaskPipe

MaskPipe is a spaCy-native toolkit for detecting, refining, resolving, and redacting PII.

Use it when you want one of these workflows:
- detect PII with built-in and custom rules, then redact it
- take entities from another NER system, run overlap resolution, then redact them
- combine both approaches in one spaCy pipeline

## What MaskPipe Does

MaskPipe gives you four composable pipeline components:
- `recognizer`: finds spans from token patterns, phrase patterns, and custom matchers
- `context_enhancer`: boosts scores or relabels spans from nearby context
- `conflict_resolver`: resolves overlap and filters low-confidence spans
- `anonymizer`: writes masked output to `doc._.masked`

The original `doc.text` is never modified.

## Installation

```bash
pip install maskpipe
python -m spacy download nl_core_news_sm
```

Requirements:
- Python 3.11-3.14
- spaCy 3.8+

Optional dependencies for examples and integrations:

```bash
pip install faker gliner transformers
```

## Quick Start: Built-in Detection + Masking

This is the default workflow if you want MaskPipe to detect PII itself.

```python
import spacy
from maskpipe import PipelineBuilder
from maskpipe import entities
from maskpipe.entities import nl

nlp = spacy.load("nl_core_news_sm", disable=["ner"])

builder = PipelineBuilder(nlp)
builder.add_entities([
    nl.BSN.replace(redactor="[BSN]"),
    entities.PHONE_NUMBER.replace(redactor="[PHONE_NUMBER]"),
    entities.EMAIL.replace(redactor="[EMAIL]"),
])

nlp = builder.build()

doc = nlp("Mijn BSN is 692015644, bel me op 0612345678 of mail naar info@example.com")

print(doc.text)
print(doc._.masked)
for ent in doc.ents:
    print(ent.text, ent.label_, ent._.score, ent._.replacement)
# Mijn BSN is 692015644, bel me op 0612345678 of mail naar info@example.com
# Mijn BSN is [BSN], bel me op [PHONE_NUMBER] of mail naar [EMAIL]
# 692015644 BSN 0.85 [BSN]
# 0612345678 PHONE_NUMBER 0.75 [PHONE_NUMBER]
# info@example.com EMAIL 1.0 [EMAIL]
```

Output model:
- `doc.text`: original text
- `doc._.masked`: masked text
- `doc.ents`: resolved spans after conflict resolution
- `span._.replacement`: replacement chosen by the anonymizer

If no redactor is registered for a label, MaskPipe uses `[LABEL]`.

## Quick Start: External NER + Masking

This is the right setup if another model already produced entity offsets.

```python
import spacy
from transformers import pipeline
from maskpipe import PipelineBuilder, DocBuilder

# Load your NER model
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# Set up MaskPipe to only resolve overlaps and mask (no local detection).
nlp = spacy.load("nl_core_news_sm", disable=["ner"])
builder = PipelineBuilder(nlp, disable=["recognizer", "context_enhancer"])
nlp = builder.build()

text = "Alice works at Google. Contact her at alice@example.com or 555-1234."
results = ner(text)

# results is a list of dicts like:
# [{"word": "Alice", "start": 0, "end": 5, "entity_group": "B-PER", "score": 0.98}, ...]

doc = DocBuilder(nlp, text).with_hf_ner(results).build()
doc = nlp(doc)
print(doc._.masked)
# [PERSON] works at [ORG]. Contact her at [EMAIL] or [PHONE_NUMBER].
```

Why this works:
- `DocBuilder.with_hf_ner()` reads the NER output and converts it to spaCy spans.
- `conflict_resolver` deduplicates overlapping spans and writes clean results to `doc.ents`.
- `anonymizer` reads `doc.ents` and generates `doc._.masked`.

## Built-in Entities

Available in `maskpipe.entities`:
- `CREDIT_CARD`
- `DATE`
- `EMAIL`
- `IBAN`
- `IPV4`
- `IPV6`
- `MAC_ADDRESS`
- `NUMBER`
- `PHONE_NUMBER`
- `URL`

Available in `maskpipe.entities.nl`:
- `BSN`

Entity objects are immutable configs. Use `.replace(...)` to override one field without rebuilding the whole entity:

```python
from maskpipe import entities

masked_email = entities.EMAIL.replace(redactor="[EMAIL]")
```

## Creating Custom Entities

```python
from maskpipe.entities import Entity

EMPLOYEE_ID = Entity(
    label="EMPLOYEE_ID",
    patterns=[
        {"pattern": [{"TEXT": {"REGEX": r"EMP-\\d{5}"}}], "score": 0.9, "id": "employee-id"},
    ],
    context_patterns=[
        {"pattern": [{"LOWER": "employee"}]},
        {"context_label": "STAFF_ID", "pattern": [{"LOWER": "staff"}, {"LOWER": "id"}]},
    ],
    validator=lambda span: span.text.startswith("EMP-"),
    redactor=lambda text: "EMP-XXXXX",
)
```

Supported redactors:
- fixed string: `"[MASK]"`
- zero-argument callable: `lambda: "generated-value"`
- one-argument callable: `lambda text: text[:1] + "*" * (len(text) - 1)`

## DocBuilder

`DocBuilder` converts character offsets from external NER systems into spaCy spans with scores.

### Basic Usage

```python
from maskpipe import DocBuilder

# Create a doc and add entities
doc = DocBuilder(nlp, text).with_entities(
    entities=[
        {"start": 0, "end": 5, "label": "PERSON", "score": 0.95},
        {"start": 30, "end": 45, "label": "EMAIL", "score": 0.99},
    ]
).build()

doc = nlp(doc)
print(doc._.masked)
```

### Entity Format

`with_entities()` expects a list of dicts with at least:
- `start`: character offset (int)
- `end`: character offset (int)
- `label`: entity type (str)
- `score`: confidence [0.0, 1.0] (float, optional)

```python
entities = [
    {"start": 0, "end": 5, "label": "PERSON", "score": 0.95},
    {"start": 30, "end": 45, "label": "EMAIL", "score": 0.99},
]
doc = DocBuilder(nlp, text).with_entities(entities).build()
```

### Entity Mappers

Use `entity_mapper` to normalize different NER output formats. MaskPipe provides pre-configured mappers:

| Mapper | Use For | Key Fields |
|--------|---------|-----------|
| `GLINER_MAPPER` | GLiNER (x-large) | `start`, `end`, `label`, `score` |
| `GLINER2_MAPPER` | GLiNER2 (nested format) | nested `{label: {start, end, confidence}}` |
| `HF_NER_MAPPER` | HuggingFace NER | `entity_group` (or `entity`), `start`, `end`, `score` |
| `OPENMED_MAPPER` | OpenMed (biomedical) | `start`, `end`, `label`, `confidence` |

Example with GLiNER:

```python
from maskpipe import GLINER_MAPPER, DocBuilder
from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = "Patient John Doe, email: john@example.com"
predictions = model.predict_entities(text, labels=["person", "email"], threshold=0.5)

doc = DocBuilder(nlp, text).with_entities(predictions, entity_mapper=GLINER_MAPPER).build()
doc = nlp(doc)
print(doc._.masked)
# [PERSON], email: [EMAIL]
```

Example with HuggingFace NER:

```python
from maskpipe import HF_NER_MAPPER, DocBuilder
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER")
text = "Contact: alice@example.com"
results = ner(text)

doc = DocBuilder(nlp, text).with_entities(results, entity_mapper=HF_NER_MAPPER).build()
doc = nlp(doc)
print(doc._.masked)
# Contact: [EMAIL]
```

### Custom Mappers

Create custom mappers for other NER systems:

```python
from maskpipe import EntityMapper

# For any system with {start, end, label, score}
custom_mapper = EntityMapper(label="type", score="confidence")

# For systems with conditional label fields
fallback_mapper = EntityMapper(
    label="entity_type",
    label_fallback="category",  # use if entity_type not found
    score="conf"
)
```

### Batch Processing

Use `build_batch()` to process multiple texts at once:

```python
docs = list(DocBuilder.build_batch(
    nlp=nlp,
    texts=["text1", "text2", "text3"],
    entities_list=[
        [{"start": 0, "end": 5, "label": "PERSON", "score": 0.9}],
        [{"start": 10, "end": 20, "label": "EMAIL", "score": 0.95}],
        [],  # no entities
    ],
))

for doc in nlp.pipe(docs):
    print(doc._.masked)
```

### Context Words

Add context to help the `context_enhancer` component make relabeling decisions:

```python
doc = DocBuilder(nlp, text).with_context_words(["email", "contact"]).build()
```

## Customizing Components

### PipelineBuilder

`PipelineBuilder` adds the default component chain in this order:
1. `recognizer`
2. `context_enhancer`
3. `conflict_resolver`
4. `anonymizer`

You can disable components you do not need:

```python
from maskpipe import PipelineBuilder

builder = PipelineBuilder(
    nlp,
    label_mapping={"persoon": "PERSON"},
    disable=["context_enhancer"],
)
```

### Context Enhancement

Add context patterns directly to the component:

```python
context_enhancer = nlp.get_pipe("context_enhancer")
context_enhancer.add_patterns([
    {
        "label": "EMAIL",
        "pattern": [{"LOWER": {"IN": ["email", "mail", "e-mail"]}}],
    }
])
```

Important:
- context patterns match by label
- score changes come from component config such as `confidence_boost`
- `context_label` can relabel a matched span
- `doc._.context_words` lets you add extra context terms not present in the text

### Anonymizer

```python
anonymizer = nlp.get_pipe("anonymizer")
anonymizer.add_redactors({
    "EMAIL": "[REDACTED]",
    "ID": lambda: "ID-000001",
    "PERSON": lambda text: text[0] + "." * (len(text) - 1),
})
```

The anonymizer:
- leaves `doc.text` unchanged
- stores masked output in `doc._.masked`
- stores the chosen replacement in `span._.replacement`

## spaCy Extensions Added by MaskPipe

Document extensions:
- `doc._.masked`
- `doc._.context_words`

Span extensions:
- `span._.score`
- `span._.context`
- `span._.replacement`

## Minimal API Reference

```python
PipelineBuilder(nlp, label_mapping=None, disable=None)
DocBuilder(
    nlp,
    text,
    label_mapping=None,
    spans_key="sc",
    annotate_ents=False,
    default_score=0.6,
    alignment_mode="strict",
)
Entity(
    label,
    patterns=None,
    custom_matcher=None,
    validator=None,
    context_patterns=None,
    redactor=None,
)
```

## Development

```bash
uv sync --dev
uv run pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
