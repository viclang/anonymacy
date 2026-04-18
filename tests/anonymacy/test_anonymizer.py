from __future__ import annotations
import uuid
import pytest  # type:ignore[import-untyped]
from spacy.tokens import Doc, Span
from spacy.lang.nl import Dutch
from anonymacy import Anonymizer

def ent(doc: Doc, start: int, end: int, label: str, score: float = 0.9) -> Span:
    span = Span(doc, start, end, label=label)
    span._.score = score
    return span

def test_no_redactors_registered():
    nlp = Dutch()
    doc = nlp("Mijn naam is Anna de Vries")
    doc.ents = [ent(doc, 3, 6, "persoon")]
    anonymizer = Anonymizer(nlp)
    
    out: Doc = anonymizer(doc)._.anonymized

    assert out.text == "Mijn naam is [PERSOON]"
    assert out.ents[0].label_ == "persoon"
    assert out.ents[0].text == "[PERSOON]"

def test_fixed_string_redactor():
    nlp = Dutch()
    doc = nlp("Contact foo@bar.com")
    doc.ents = [ent(doc, 1, 2, "email")]    
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"email": "[EMAIL]"})
    
    out: Doc = anonymizer(doc)._.anonymized
    
    assert out.text == "Contact [EMAIL]"
    assert out.ents[0].text == "[EMAIL]"


def test_zero_arg_callable():
    nlp = Dutch()
    fake_bsn = "123456789"
    doc = nlp(f"BSN {fake_bsn}")
    doc.ents = [ent(doc, 1, 2, "BSN")]
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"BSN": lambda: str(uuid.uuid4())[:8]})
    
    out: Doc = anonymizer(doc)._.anonymized
    
    assert out.ents[0].text != fake_bsn
    assert len(out.ents[0].text) == 8

def test_one_arg_callable():
    nlp = Dutch()
    doc = nlp("Naam Clara")
    doc.ents = [ent(doc, 1, 2, "persoon")]
    reverse = lambda txt: txt[::-1]
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"persoon": reverse})
    
    out: Doc = anonymizer(doc)._.anonymized
    
    assert out.text == "Naam aralC"


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
    
    out: Doc = anonymizer(doc)._.anonymized
    
    assert out.text == "Delete 1234 ok"
    assert len(out.ents) == 0


def test_overlapping_spans_only_longest_kept():
    nlp = Dutch()
    doc = nlp("123456789 is BSN")
    span1 = ent(doc, 0, 1, "BSN", score=0.9)
    span2 = ent(doc, 0, 1, "BSN", score=0.8)
    doc.spans["sc"] = [span1, span2]
    anonymizer = Anonymizer(nlp, style="span", spans_key="sc")
    anonymizer.add_redactors({"BSN": "[B]"})
    
    out: Doc = anonymizer(doc)._.anonymized
    
    assert out.text.count("[B]") == 1
    assert out.text == "[B] is BSN"


def test_whole_doc_replaced():
    nlp = Dutch()
    words = "Dit is een test".split()
    doc = nlp("Dit is een test")
    doc.ents = [ent(doc, 0, len(words), "sentence")]
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"sentence": "XXX"})
    
    out: Doc = anonymizer(doc)._.anonymized
    
    assert out.text == "XXX"
    assert len(out) == 1


def test_preserve_surrounding_spaces():
    nlp = Dutch()
    txt = "Start  123456789  End"
    doc = nlp(txt)
    doc.ents = [ent(doc, 2, 3, "BSN")]
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"BSN": "X" * 9})
    
    out: Doc = anonymizer(doc)._.anonymized
    
    assert out.text == "Start  XXXXXXXXX  End"    
    spaces = [bool(t.whitespace_) for t in out]    
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
    out = fresh(doc)._.anonymized
    assert out.text == "bar"


def test_clear_redactors():
    nlp = Dutch()
    anonymizer = Anonymizer(nlp)
    anonymizer.add_redactors({"label": "SECRET"})
    d1 = nlp("token")
    d1.ents = [ent(d1, 0, 1, "label")]
    
    assert anonymizer(d1)._.anonymized.text == "SECRET"
    anonymizer.clear()
    d2 = nlp("token")
    d2.ents = [ent(d2, 0, 1, "label")]
    assert anonymizer(d2)._.anonymized.text == "[LABEL]"