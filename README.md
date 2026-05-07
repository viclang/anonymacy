# MaskPipe
MaskPipe is a modular, spaCy-native PII de-identification pipeline. Refine and orchestrate entity results from GLiNER, HuggingFace, or any NER source with context-aware boosting, rule-based validation, and flexible redaction.

Here's an initial README for MaskPipe:

# MaskPipe

A spaCy-powered Python library for detecting and anonymizing sensitive entities in text. Built for Dutch and English text processing with extensible support for custom entity types.

## Features

- **Entity Recognition** — Pattern-based and custom matching for PII and sensitive data
- **Context Enhancement** — Boost confidence scores using surrounding context patterns
- **Conflict Resolution** — Hierarchical merging of overlapping spans
- **Flexible Anonymization** — Replace entities with static strings or custom redaction functions
- **Multi-source Integration** — Built-in support for spaCy, GLiNER, HuggingFace NER, and custom entity formats

## Built-in Entities

| Entity | Description | Validation |
|--------|-------------|------------|
| `BSN` | Dutch citizen service number | 11-proef checksum |
| `CREDIT_CARD` | Major card formats (Visa, Mastercard, Amex, etc.) | — |
| `DATE` | ISO 8601, Dutch/English formats, ordinals | — |
| `DATE_OF_BIRTH` | Dates with birth context | Context label |
| `EMAIL` | Standard email addresses | — |
| `IBAN` | International bank account numbers | `schwifty` checksum |
| `IPV4` / `IPV6` | IP addresses | `ipaddress` validation |
| `MAC_ADDRESS` | Hardware addresses | — |
| `PHONE_NUMBER` | International phone numbers | `phonenumbers` library |
| `URL` | Web addresses | — |

## Quick Start

```python
import spacy
from maskpipe import PipelineBuilder, DocBuilder
from maskpipe import entities
from maskpipe.entities import nl
from faker import Faker

# Load spaCy model
nlp = spacy.load("nl_core_news_sm", disable=["ner"])

# Build pipeline with entities
builder = PipelineBuilder(nlp)
fake = Faker("nl_NL")

builder.add_entities([
    nl.BSN.replace(redactor=fake.ssn),
    entities.PHONE_NUMBER.replace(redactor=fake.phone_number),
    entities.DATE,
])

# Process text
doc = nlp("Mijn bsn is 692015644 en mijn telefoon is 0612345678")
print(doc._.anonymized.text)
# Output: "Mijn bsn is [REDACTED] en mijn telefoon is [REDACTED]"
```

## Pipeline Components

| Component | Purpose |
|-----------|---------|
| `recognizer` | Pattern matching and custom entity detection |
| `context_enhancer` | Context-aware score boosting and label refinement |
| `conflict_resolver` | Overlap resolution with hierarchical label merging |
| `anonymizer` | Entity replacement with configurable redaction |

## Custom Entity Definition

```python
from maskpipe import Entity

CUSTOM_ENTITY = Entity(
    label="EMPLOYEE_ID",
    patterns=[
        {"score": 0.8, "pattern": [{"TEXT": {"REGEX": r"EMP-\d{5}"}}]},
    ],
    context_patterns=[
        {"pattern": [{"LEMMA": "employee"}]},
    ],
    validator=lambda span: span.text.startswith("EMP-"),
    redactor=lambda text: "EMP-XXXXX",
)
```

## Integration with GLiNER

```python
from gliner import GLiNER
from maskpipe import DocBuilder

model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
predictions = model.predict_entities(text, labels=["person", "organization"])

doc = DocBuilder(nlp, text).with_gliner(predictions).build()
```

## Installation

```bash
pip install maskpipe
```

Requires Python 3.11–3.14 and spaCy 3.8+.

## License

MIT License — see [LICENSE](LICENSE)
