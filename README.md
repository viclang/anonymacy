# MaskPipe

MaskPipe is a modular, spaCy-native PII de-identification pipeline. Refine and orchestrate entity results from GLiNER, HuggingFace, or any NER source with context-aware boosting, rule-based validation, and flexible redaction.

## Features

- **Entity Recognition** — Pattern-based matching (token patterns, phrase patterns, regex) and custom matchers for PII and sensitive data
- **Context Enhancement** — Boost confidence scores or refine labels using surrounding context patterns
- **Conflict Resolution** — Hierarchical merging of overlapping spans with configurable label hierarchies
- **Flexible Anonymization** — Replace entities with static strings, zero-argument callables, or text-transforming callables
- **Multi-source Integration** — Built-in support for spaCy, GLiNER, HuggingFace NER, OpenMed, and custom entity formats
- **Validation** — Per-entity validator functions to filter false positives (e.g., IBAN checksum, BSN 11-proef)
- **spaCy-native** — Full integration with spaCy pipelines, serialization, and custom attributes

## Built-in Entities

| Entity | Description | Validation | Redactor |
|--------|-------------|------------|----------|
| `BSN` | Dutch citizen service number | 11-proef checksum | — |
| `CREDIT_CARD` | Major card formats (Visa, Mastercard, Amex, etc.) | — | — |
| `DATE` | ISO 8601, Dutch/English formats, ordinals | — | — |
| `DATE_OF_BIRTH` | Dates with birth context | Context label | — |
| `EMAIL` | Standard email addresses | — | — |
| `IBAN` | International bank account numbers | `schwifty` checksum | — |
| `IPV4` / `IPV6` | IP addresses | `ipaddress` validation | — |
| `MAC_ADDRESS` | Hardware addresses | — | — |
| `NUMBER` | Numeric tokens | — | — |
| `PHONE_NUMBER` | International phone numbers | `phonenumbers` library | — |
| `URL` | Web addresses | — | — |

## Installation

```bash
pip install maskpipe
```

Requires Python 3.11–3.14 and spaCy 3.8+.

Install a spaCy model for your language:

```bash
# Dutch example
python -m spacy download nl_core_news_sm
```

## Quick Start

### Basic Pipeline Setup

```python
import spacy
from maskpipe import PipelineBuilder, DocBuilder
from maskpipe import entities
from maskpipe.entities import nl
from faker import Faker

# Load spaCy model (disable default NER if using custom recognizers)
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
print(doc._.masked)
# Output: "Mijn bsn is 692015644 en mijn telefoon is 0612345678"
# (Note: entities are detected but not yet masked in the original doc)
```

### Accessing Masked Text

The `Anonymizer` component stores the masked version under `doc._.masked`:

```python
# After pipeline processing
print(doc._.masked)
# With redactors configured: "Mijn bsn is [BSN] en mijn telefoon is [PHONE_NUMBER]"
```

## Pipeline Components

| Component | Purpose | Configurable Options |
|-----------|---------|-------------------|
| `recognizer` | Pattern matching and custom entity detection | `spans_key`, `default_score`, `annotate_ents`, `overwrite` |
| `context_enhancer` | Context-aware score boosting and label refinement | `confidence_boost`, `min_enhanced_score`, `context_window`, `style` |
| `conflict_resolver` | Overlap resolution with hierarchical label merging | `threshold`, `spans_filter`, `style` |
| `anonymizer` | Entity replacement with configurable redaction | `spans_key`, `spans_filter`, `style` |

### Component Configuration

```python
# Custom pipeline configuration
nlp = spacy.load("nl_core_news_sm", disable=["ner"])

# Add components with custom settings
nlp.add_pipe("recognizer", config={
    "default_score": 0.7,
    "annotate_ents": True,  # Also populate doc.ents
    "overwrite": False       # Preserve existing spans
})

nlp.add_pipe("context_enhancer", config={
    "confidence_boost": 0.35,
    "min_enhanced_score": 0.4,
    "context_window": (5, 3),  # (before, after) tokens
    "style": "span"            # or "ent"
})

nlp.add_pipe("conflict_resolver", config={
    "threshold": 0.5,
    "style": "ent"             # Write resolved spans to doc.ents
})

nlp.add_pipe("anonymizer", config={
    "style": "ent",            # Use doc.ents as source
    "spans_key": "sc"
})
```

## Using PipelineBuilder

The `PipelineBuilder` is the recommended way to set up the full pipeline:

```python
from maskpipe import PipelineBuilder

# Initialize with optional label mapping and disabled components
builder = PipelineBuilder(
    nlp,
    label_mapping={"persoon": "PERSON"},  # Map external labels to internal
    disable=["conflict_resolver"]          # Skip this component
)

# Add entities with patterns, validators, context, and redactors
builder.add_entities([
    nl.BSN,
    entities.EMAIL,
    entities.DATE,
])

nlp = builder.build()
```

## DocBuilder: Integrating External NER Sources

`DocBuilder` converts entity predictions from various sources into spaCy spans.

### With GLiNER

```python
from gliner import GLiNER
from maskpipe import DocBuilder

model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
predictions = model.predict_entities(
    text,
    threshold=0.5,
    labels=["person", "organization", "email"]
)

doc = DocBuilder(nlp, text).with_gliner(predictions).build()
doc = nlp(doc)  # Run through MaskPipe pipeline
```

### With HuggingFace NER

```python
from transformers import pipeline

hf_ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
results = hf_ner(text)

doc = DocBuilder(nlp, text).with_hf_ner(results).build()
```

### With Custom Entities

```python
custom_entities = [
    {"start": 10, "end": 20, "label": "CUSTOM", "score": 0.9},
]

doc = DocBuilder(nlp, text).with_custom(
    custom_entities,
    label_key="label",
    score_key="score"
).build()
```

### Batch Processing

```python
texts = ["text 1", "text 2", "text 3"]
entities_list = [[...], [...], [...]]

docs = list(DocBuilder.build_batch_with_gliner(nlp, texts, entities_list))
```

## Custom Entity Definition

Create your own entity types with patterns, validators, context, and redactors:

```python
from maskpipe import Entity

CUSTOM_ENTITY = Entity(
    label="EMPLOYEE_ID",
    patterns=[
        {
            "score": 0.8,
            "pattern": [{"TEXT": {"REGEX": r"EMP-\d{5}"}}]
        },
    ],
    context_patterns=[
        {"pattern": [{"LEMMA": "employee"}]},
        {"pattern": [{"LEMMA": "id"}, {"TEXT": {"REGEX": r":"}}]},
    ],
    validator=lambda span: span.text.startswith("EMP-"),
    redactor=lambda text: "EMP-XXXXX",
)
```

### Entity Configuration Options

| Attribute | Type | Description |
|-----------|------|-------------|
| `label` | `str` | Entity type label |
| `patterns` | `List[Pattern]` | spaCy Matcher token/phrase patterns with optional `score` |
| `custom_matcher` | `Callable[[Doc], List[Tuple[int, int, float]]]` | Custom matching function returning (start, end, score) |
| `validator` | `Callable[[Span], bool]` | Validation function to filter false positives |
| `context_patterns` | `List[ContextPattern]` | Patterns that boost confidence when found near entity |
| `redactor` | `str \| Callable[[], str] \| Callable[[str], str]` | Replacement strategy |

### Pattern Types

```python
# Token pattern (regex on token text)
{"score": 0.9, "pattern": [{"TEXT": {"REGEX": r"\d{4}-\d{2}-\d{2}"}}]}

# Token pattern (shape-based)
{"score": 0.6, "pattern": [{"SHAPE": "dd:dd:dd"}]}

# Token pattern (multi-token)
{"score": 0.75, "pattern": [
    {"TEXT": {"REGEX": r"\b\d{1,2}\b"}},
    {"LOWER": {"IN": ["januari", "februari"]}},
    {"TEXT": {"REGEX": r"\b\d{4}\b"}}
]}

# Phrase pattern (string)
{"score": 0.8, "pattern": "Anna de Vries"}

# Pattern with ID for later removal
{"score": 0.9, "pattern": "...", "id": "specific-pattern"}
```

### Context Patterns

Context patterns can optionally specify a `context_label` to refine the entity label:

```python
context_patterns=[
    # Generic boost
    {"pattern": [{"LEMMA": {"IN": ["datum", "date"]}}]},
    
    # Label refinement: DATE becomes DATE_OF_BIRTH
    {
        "context_label": "DATE_OF_BIRTH",
        "pattern": [{"LEMMA": {"IN": ["geboortedatum", "birthdate"]}}]
    },
]
```

## Context Enhancement

The `ContextEnhancer` boosts entity confidence scores when context patterns match nearby tokens:

```python
# Access the component directly
context_enhancer = nlp.get_pipe("context_enhancer")

# Add patterns programmatically
context_enhancer.add_patterns([
    {
        "label": "EMAIL",
        "pattern": [{"LOWER": {"IN": ["mail", "email", "e-mail"]}}],
        "score": 0.3  # Optional: override default boost
    },
])
```

Context matching supports:
- Token window (configurable before/after span)
- Dependency-based relations (fallback when outside window)
- Added context words via `doc._.context_words`

## Conflict Resolution

The `ConflictResolver` and `HierarchicalMergeFilter` handle overlapping spans:

```python
from maskpipe.span_filter import HierarchicalMergeFilter

# Default hierarchy
DEFAULT_HIERARCHY = {
    'date': ['date_of_birth', 'date_time'],
    'name': ['first_name', 'last_name', 'full_name'],
    'phone': ['phone_number', 'fax_number', 'mobile_number'],
    'address': ['street_address', 'home_address', 'billing_address'],
    'ip': ['ipv4', 'ipv6'],
    'id': ['ssn', 'bsn', 'medical_record_number', 'account_number', 'employee_id'],
    'national_id': ['nir', 'insee', 'steuer_id', 'steuernummer', 'codice_fiscale'],
}

# Custom hierarchy
custom_filter = HierarchicalMergeFilter(hierarchy={
    'person': ['patient', 'doctor', 'employee'],
    'location': ['hospital', 'clinic', 'address'],
})
```

Resolution rules:
1. Higher-scoring spans win when overlapping
2. Longer spans preferred on ties
3. Child labels merge into parent labels when adjacent
4. More specific labels win during merging

## Anonymization

Configure redactors per entity type:

```python
anonymizer = nlp.get_pipe("anonymizer")

# Fixed string
anonymizer.add_redactors({"EMAIL": "[REDACTED]"})

# Zero-argument callable (called per entity)
import uuid
anonymizer.add_redactors({"ID": lambda: str(uuid.uuid4())[:8]})

# One-argument callable (receives original text)
anonymizer.add_redactors({"NAME": lambda text: text[0] + "." * (len(text) - 1)})

# Default fallback: entities without redactors become [LABEL]
```

### Anonymizer Output

The anonymizer stores results in custom attributes:

```python
doc = nlp(text)

# Masked text string
print(doc._.masked)

# Per-span replacements
for ent in doc.ents:
    print(f"{ent.text} -> {ent._.replacement}")
```

## Advanced: Custom Matchers

For complex matching logic, provide a custom matcher function:

```python
from spacy.tokens import Doc
from typing import List, Tuple

def my_matcher(doc: Doc) -> List[Tuple[int, int, float]]:
    """Return (start_token, end_token, score) tuples."""
    matches = []
    for token in doc:
        if token.text.startswith("SECRET-"):
            matches.append((token.i, token.i + 1, 0.95))
    return matches

entity = Entity(
    label="SECRET",
    custom_matcher=my_matcher,
    redactor="[SECRET]",
)
```

## Advanced: Context Words

Provide additional context words that aren't in the original text:

```python
doc = DocBuilder(nlp, text)\
    .with_context_words(["patient_record", "confidential"])\
    .with_gliner(predictions)\
    .build()
```

## Serialization

Components support spaCy's serialization protocol:

```python
# Save redactors
anonymizer.to_disk("./models/redactors")

# Save patterns
recognizer.to_disk("./models/patterns")

# Bytes serialization
blob = recognizer.to_bytes()
recognizer.from_bytes(blob)
```

## Complete Example: Healthcare Text

```python
import spacy
from maskpipe import PipelineBuilder, DocBuilder
from maskpipe import entities
from maskpipe.entities import nl
from faker import Faker
from gliner import GLiNER

# Setup
nlp = spacy.load("nl_core_news_sm", disable=["ner"])
builder = PipelineBuilder(nlp)
fake = Faker("nl_NL")

# Configure entities with redactors
builder.add_entities([
    nl.BSN.replace(redactor=fake.ssn),
    entities.PHONE_NUMBER.replace(redactor=fake.phone_number),
    entities.EMAIL.replace(redactor=fake.email),
    entities.DATE,
    entities.IBAN,
])

# Add GLiNER for names/organizations
model = GLiNER.from_pretrained("knowledgator/gliner-x-large")
text = "Patient John Doe, BSN 123456782, tel: 0612345678"
predictions = model.predict_entities(text, labels=["person", "organization"])

# Build doc and process
doc = DocBuilder(nlp, text).with_gliner(predictions).build()
doc = nlp(doc)

print(doc._.masked)
```

## spaCy Extension Attributes

MaskPipe adds these custom attributes:

| Attribute | Type | On | Description |
|-----------|------|-----|-------------|
| `doc._.masked` | `str` | Doc | Masked text after anonymization |
| `doc._.context_words` | `List[str]` | Doc | Additional context words |
| `span._.score` | `float` | Span | Confidence score (0.0–1.0) |
| `span._.context` | `List[str]` | Span | Matched context patterns |
| `span._.replacement` | `str` | Span | Applied replacement text |

## API Reference

### PipelineBuilder

```python
PipelineBuilder(
    nlp: Language,
    label_mapping: Optional[Dict[str, str]] = None,
    disable: Optional[List[str]] = None
)
```

### DocBuilder

```python
DocBuilder(
    nlp: Language,
    text: str,
    label_mapping: Optional[Dict[str, str]] = None,
    spans_key: Optional[str] = "sc",
    annotate_ents: bool = False,
    default_score: float = 0.6,
    alignment_mode: str = "strict"
)
```

### Entity

```python
Entity(
    label: str,
    patterns: Optional[Sequence[Pattern]] = None,
    custom_matcher: Optional[CustomMatcherFunc] = None,
    validator: Optional[Callable[[Span], bool]] = None,
    context_patterns: Optional[Sequence[ContextPattern]] = None,
    redactor: Optional[RedactorFunc] = None
)
```

## License

MIT License — see [LICENSE](LICENSE)