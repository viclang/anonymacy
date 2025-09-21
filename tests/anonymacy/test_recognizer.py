from spacy.lang.nl import Dutch

def test_pattern_without_score_gets_default_score():
    nlp = Dutch()
    default_score = 0.8
    pattern_score = 0.9
    recognizer = nlp.add_pipe("recognizer", config={ "default_score": default_score })
    recognizer.add_patterns([
        { "label": "persoon", "pattern": "Anna de Vries" },
        { "label": "organisatie", "pattern": "Acme Corp.", "score": pattern_score },
    ])

    doc = nlp("Mijn naam is Anna de Vries en ik werk bij Acme Corp.")

    assert doc.spans['sc'][0]._.score == default_score
    assert doc.spans['sc'][1]._.score == pattern_score

def test_add_pattern_without_label_raises_error():
    nlp = Dutch()
    recognizer = nlp.add_pipe("recognizer")

    try:
        recognizer.add_patterns([
            { "pattern": "Anna de Vries" },
        ])
        assert False, "Expected KeyError for pattern without label"
    except KeyError as e:
        assert str(e) == "'label'"

def test_add_pattern_without_pattern_raises_error():
    nlp = Dutch()
    recognizer = nlp.add_pipe("recognizer")

    try:
        recognizer.add_patterns([
            { "label": "persoon" },
        ])
        assert False, "Expected KeyError for pattern without pattern"
    except KeyError as e:
        assert str(e) == "'pattern'"

def test_add_pattern_with_id():
    nlp = Dutch()
    recognizer = nlp.add_pipe("recognizer")

    recognizer.add_patterns([
        { "label": "persoon", "pattern": "Anna de Vries", "id": "anna" },
    ])

    doc = nlp("Mijn naam is Anna de Vries.")

    assert doc.spans['sc'][0].id_ == "anna"

def test_annotate_ents():
    nlp = Dutch()
    pattern = "Anna de Vries"
    label = "persoon"
    recognizer = nlp.add_pipe("recognizer", config={ "annotate_ents": True })
    recognizer.add_patterns([
        { "label": "persoon", "pattern": pattern },
    ])

    doc = nlp("Mijn naam is Anna de Vries.")

    assert len(doc.ents) == 1
    assert doc.ents[0].label_ == label
    assert doc.ents[0].text == pattern

def test_limit_to_max_score():
    nlp = Dutch()
    recognizer = nlp.add_pipe("recognizer")
    recognizer.add_patterns([
        { "label": "persoon", "pattern": "Anna de Vries", "score": 2.0 },
    ])

    doc = nlp("Mijn naam is Anna de Vries en ik werk bij Acme Corp.")

    assert len(doc.spans['sc']) == 1

def test_no_duplicate_spans():
    nlp = Dutch()
    recognizer = nlp.add_pipe("recognizer")
    recognizer.add_patterns([
        { "label": "persoon", "pattern": "Anna de Vries", "score": 0.9 },
        { "label": "persoon", "pattern": "Anna de Vries", "score": 0.8 },
    ])

    doc = nlp("Mijn naam is Anna de Vries en ik werk bij Acme Corp.")

    assert len(doc.spans['sc']) == 1
    assert doc.spans['sc'][0]._.score == 0.9