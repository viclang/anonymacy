from __future__ import annotations
import uuid
import pytest  # type:ignore[import-untyped]
from spacy.tokens import Doc, Span
from spacy.lang.nl import Dutch
from maskpipe import Anonymizer

def ent(doc: Doc, start: int, end: int, label: str, score: float = 0.9) -> Span:
    span = Span(doc, start, end, label=label)
    span._.score = score
    return span

def test_no_redactors_registered():
    nlp = Dutch()
    doc = nlp("Mijn naam is Anna de Vries")
    doc.ents = [ent(doc, 3, 6, "persoon")]
    anonymizer = Anonymizer(nlp)
    
    doc = anonymizer(doc)
    out: str = doc._.masked

    assert out == "Mijn naam is [PERSOON]"
    assert doc.ents[0].label_ == "persoon"
    assert doc.ents[0]._.replacement == "[PERSOON]"

def test_fixed_string_redactor():
    nlp = Dutch()
    doc = nlp("Contact foo@bar.com")
    doc.ents = [ent(doc, 1, 2, "email")]    
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"email": "[EMAIL]"})
    
    doc = anonymizer(doc)
    out: str = doc._.masked
    
    assert out == "Contact [EMAIL]"
    assert doc.ents[0]._.replacement == "[EMAIL]"


def test_zero_arg_callable():
    nlp = Dutch()
    fake_bsn = "123456789"
    doc = nlp(f"BSN {fake_bsn}")
    doc.ents = [ent(doc, 1, 2, "BSN")]
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"BSN": lambda: str(uuid.uuid4())[:8]})
    
    doc = anonymizer(doc)
    replacement = doc.ents[0]._.replacement
    assert replacement != fake_bsn
    assert len(replacement) == 8

def test_one_arg_callable():
    nlp = Dutch()
    doc = nlp("Naam Clara")
    doc.ents = [ent(doc, 1, 2, "persoon")]
    reverse = lambda txt: txt[::-1]
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"persoon": reverse})
    
    doc = anonymizer(doc)
    out: str = doc._.masked
    
    assert out == "Naam aralC"


def test_operator_exception_bubbles():
    nlp = Dutch()
    doc = nlp("XYZ")
    doc.ents = [ent(doc, 0, 1, "fail")]

    def _boom(_):
        raise RuntimeError("operator error")

    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"fail": _boom})
    with pytest.raises(RuntimeError, match="operator error"):
        anonymizer(doc)

def test_empty_redactor():
    nlp = Dutch()
    
    doc = nlp("Delete [PIN] 1234 ok")
    doc.ents = [ent(doc, 1, 4, "PIN")]
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"PIN": ""})
    
    doc = anonymizer(doc)
    out: str = doc._.masked
    
    assert out == "Delete 1234 ok"
    assert doc.ents[0]._.replacement == ""


def test_overlapping_spans_only_longest_kept():
    nlp = Dutch()
    doc = nlp("123456789 is BSN")
    span1 = ent(doc, 0, 1, "BSN", score=0.9)
    span2 = ent(doc, 0, 1, "BSN", score=0.8)
    doc.spans["sc"] = [span1, span2]
    anonymizer = Anonymizer(nlp, style="span", spans_key="sc")
    anonymizer.add_redactors({"BSN": "[B]"})
    
    doc = anonymizer(doc)
    out: str = doc._.masked
    
    assert out.count("[B]") == 1
    assert out == "[B] is BSN"


def test_whole_doc_replaced():
    nlp = Dutch()
    words = "Dit is een test".split()
    doc = nlp("Dit is een test")
    doc.ents = [ent(doc, 0, len(words), "sentence")]
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"sentence": "XXX"})
    
    doc = anonymizer(doc)
    out: str = doc._.masked
    
    assert out == "XXX"
    assert doc.ents[0]._.replacement == "XXX"


def test_preserve_surrounding_spaces():
    nlp = Dutch()
    txt = "Start  123456789  End"
    doc = nlp(txt)
    doc.ents = [ent(doc, 2, 3, "BSN")]
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"BSN": "X" * 9})
    
    doc = anonymizer(doc)
    out: str = doc._.masked
    
    assert out == "Start  XXXXXXXXX  End"    
    original_doc = nlp(txt)
    spaces = [bool(t.whitespace_) for t in original_doc]    
    assert spaces == [bool(t.whitespace_) for t in doc]


def test_bytes_roundtrip():
    nlp = Dutch()
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"foo": "bar", "num": lambda: "42"})
    blob = anonymizer.to_bytes()
    fresh = Anonymizer(nlp)
    fresh.from_bytes(blob)
    assert fresh._redactors["foo"] == "bar"
    doc = nlp("hello")
    doc.ents = [ent(doc, 0, 1, "foo")]
    doc = fresh(doc)
    out: str = doc._.masked
    assert out == "bar"


def test_clear_redactors():
    nlp = Dutch()
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"label": "SECRET"})
    d1 = nlp("token")
    d1.ents = [ent(d1, 0, 1, "label")]
    
    d1 = anonymizer(d1)
    assert d1._.masked == "SECRET"
    anonymizer.clear()
    d2 = nlp("token")
    d2.ents = [ent(d2, 0, 1, "label")]
    d2 = anonymizer(d2)
    assert d2._.masked == "[LABEL]"