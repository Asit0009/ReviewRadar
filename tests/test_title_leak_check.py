"""The leak experiment rebuilds the old, leaky text. That reconstruction must change only the right reviews."""

import pandas as pd

from reviewpulse.modeling.title_leak_check import with_leaky_text
from reviewpulse.pipeline.cleaner import clean_and_normalize_record


def record(title, text, rating=5.0):
    return {"rating": rating, "title": title, "text": text, "asin": "A1", "parent_asin": "P1", "user_id": "U1", "timestamp": 1609459200000, "helpful_vote": 0, "verified_purchase": True}


def test_only_automatic_titles_are_put_back():
    rows = [clean_and_normalize_record(record(t, b), "Electronics") for t, b in [("Five Stars", "Works well."), ("One Star", "Broke."), ("Broke fast", "The strap snapped."), ("Five stars for this blender", "Crushes ice.")]]
    df = pd.DataFrame(rows)
    leaky = with_leaky_text(df)
    assert leaky.tolist() == ["Five Stars. Works well.", "One Star. Broke.", "Broke fast. The strap snapped.", "Five stars for this blender. Crushes ice."]
    assert df["combined_text"].tolist()[:2] == ["Works well.", "Broke."], "the clean text must not contain the automatic title"


def test_a_missing_title_is_left_alone():
    df = pd.DataFrame([clean_and_normalize_record(record("", "Only a body here."), "Electronics")])
    assert with_leaky_text(df).tolist() == ["Only a body here."]
