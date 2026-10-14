"""Data pipeline: raw JSONL in, validated and cleaned Parquet out, with exact accounting of every rejected record."""

import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

import reviewpulse.pipeline.run_pipeline as rp
from reviewpulse.pipeline.cleaner import clean_and_normalize_record, clean_text, is_default_rating_title
from reviewpulse.pipeline.run_pipeline import REVIEW_PYARROW_SCHEMA


def raw(**over):
    base = {
        "rating": 5.0, "title": "Nice", "text": "Works well for me.", "asin": "A1", "parent_asin": "P1",
        "user_id": "U1", "timestamp": 1609459200000, "helpful_vote": 1, "verified_purchase": True,
    }
    base.update(over)
    return base


# ---------- Amazon's automatic "Five Stars" titles must not reach the models ----------
@pytest.mark.parametrize("title", ["Five Stars", "One Star", "Two Stars", "Three Stars", "Four Stars", "five stars", "FIVE STARS", "5 stars", "5 Stars", "1 star", "3 Stars.", "Four Stars!", "5.0 stars", "Five-Star", "  Five Stars  "])
def test_default_rating_titles_are_recognised(title):
    assert is_default_rating_title(title)


@pytest.mark.parametrize(
    "title",
    ["Five stars for this blender", "Great", "Stars of the show", "Would give 5 stars but", "Five Stars, would buy again", "Starter kit", "Two stars because it scratched", "", None, "Star Wars fan"],
)
def test_real_titles_are_not_mistaken_for_default_ones(title):
    assert not is_default_rating_title(title)


@pytest.mark.parametrize("rating,title", [(5.0, "Five Stars"), (1.0, "One Star"), (3.0, "Three Stars"), (4.0, "4 stars"), (2.0, "Two Stars.")])
def test_default_rating_title_is_left_out_of_the_model_text_but_kept_for_display(rating, title):
    rec = clean_and_normalize_record(raw(rating=rating, title=title, text="The strap broke after a week."), "Electronics")
    assert rec["combined_text"] == "The strap broke after a week."
    assert "star" not in rec["combined_text"].lower()
    assert rec["title"] == title  # the review card still shows what Amazon displayed


def test_a_real_title_is_still_joined_to_the_body():
    rec = clean_and_normalize_record(raw(title="Five stars for this blender", text="Crushes ice easily."), "Electronics")
    assert rec["combined_text"] == "Five stars for this blender. Crushes ice easily."
    rec = clean_and_normalize_record(raw(title="Broke fast", text="The strap broke after a week."), "Electronics")
    assert rec["combined_text"] == "Broke fast. The strap broke after a week."


def test_review_ids_do_not_depend_on_the_text():
    """Fixing the model text must not change review ids, or saved topic tags and search vectors would point at nothing."""
    a = clean_and_normalize_record(raw(title="Five Stars"), "Electronics")["review_id"]
    b = clean_and_normalize_record(raw(title="Something else entirely", text="other body"), "Electronics")["review_id"]
    assert a == b


# ---------- text cleaning ----------
@pytest.mark.parametrize(
    "dirty,clean",
    [
        ("Great<br /><br />sound", "Great sound"),
        ("line<BR>break<br/>here", "line break here"),
        ("<b>bold</b> and <i>italic</i>", "bold and italic"),
        ("5 < 6 and 7 > 3", "5 < 6 and 7 > 3"),
        ("Dead port. [[VIDEOID:ee3a6dd8b09a5a625dd3c5553784761c]] After one day it died", "Dead port. After one day it died"),
        ("See [[ASIN:B00ABC1234 Fancy Case]] for the one I bought", "See for the one I bought"),
        ("two[[videoid:abc]]markers[[ASIN:X]]here", "two markers here"),
        ("Toner-[[ASIN:B00PBJSOWQ [Skin Food] Premium Toner 180ml]] -$12.98 is great", "Toner- -$12.98 is great"),
        ("a [[not a marker]] stays and so does [[VIDEOIDS]]", "a [[not a marker]] stays and so does [[VIDEOIDS]]"),
        ("Tom &amp; Jerry &quot;ok&quot;", 'Tom & Jerry "ok"'),
        ("  lots   of \n\t space ", "lots of space"),
        ("it&#39;s fine", "it's fine"),
    ],
)
def test_clean_text(dirty, clean):
    assert clean_text(dirty) == clean


def test_cleaned_records_never_contain_br_tags():
    rec = clean_and_normalize_record(raw(text="One<br />Two<br /><br />Three", title="A<br>B"), "Electronics")
    assert "<" not in rec["text"] and "<" not in rec["title"] and rec["combined_text"] == "A B. One Two Three"


def test_review_id_is_deterministic_and_unique_per_interaction():
    a = clean_and_normalize_record(raw(), "Electronics")["review_id"]
    assert a == clean_and_normalize_record(raw(), "Electronics")["review_id"]
    assert a != clean_and_normalize_record(raw(timestamp=1609459200001), "Electronics")["review_id"]
    assert a != clean_and_normalize_record(raw(asin="A2"), "Electronics")["review_id"]


# ---------- process_category end to end ----------
@pytest.fixture
def category_files(tmp_path, monkeypatch):
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir(), out_dir.mkdir()
    monkeypatch.setattr(rp, "DATA_RAW_DIR", raw_dir)
    monkeypatch.setattr(rp, "DATA_PROCESSED_DIR", out_dir)

    lines = [
        json.dumps(raw(user_id="U1", asin="A1", rating=5.0)),  # valid
        json.dumps(raw(user_id="U2", asin="A2", rating=3.0, text="Okay<br />product")),  # valid, has a <br />
        json.dumps(raw(user_id="U3", asin="A3", rating=1.0, parent_asin="PMETA")),  # valid, matches the metadata file
        "{this is not json",  # malformed
        "",  # blank lines are ignored, not counted
        json.dumps({k: v for k, v in raw(user_id="U4").items() if k != "asin"}),  # missing required field
        json.dumps(raw(user_id="U5", asin="A5", rating=0.0)),  # rating too low
        json.dumps(raw(user_id="U6", asin="A6", rating=6.0)),  # rating too high
        json.dumps(raw(user_id="U7", asin="A7", rating="abc")),  # not a number
        json.dumps(raw(user_id="U8", asin="A8", text="  ")),  # empty text
        json.dumps(raw(user_id="U9", asin="A9", timestamp=-5)),  # bad timestamp
        json.dumps(raw(user_id="U1", asin="A1", rating=5.0)),  # exact duplicate of the first record
    ]
    (raw_dir / "Demo_reviews.jsonl").write_text("\n".join(lines), encoding="utf-8")
    (raw_dir / "Demo_meta.jsonl").write_text(json.dumps({"parent_asin": "PMETA", "title": "Demo Gadget", "price": 9.99}) + "\n", encoding="utf-8")
    return out_dir


def test_process_category_counts_every_rejection_reason(category_files):
    df, report = rp.process_category("Demo")
    r = report.to_dict()
    assert r["total_raw_read"] == 10  # 11 non-blank lines minus the one that is not JSON
    assert r["malformed_json"] == 1
    assert r["missing_required_fields"] == 1
    assert r["invalid_rating"] == 3
    assert r["empty_or_null_text"] == 1
    assert r["invalid_timestamp"] == 1
    assert r["duplicate_records"] == 1
    assert r["valid_records"] == 4  # the 3 good records plus the duplicate, which passes validation and is dropped afterwards
    assert r["retained_after_cleaning"] == len(df) == 3
    assert r["rejection_rate_percent"] == pytest.approx(70.0)


def test_process_category_output_is_clean_and_enriched(category_files):
    df, _ = rp.process_category("Demo")
    assert sorted(df["sentiment"]) == ["negative", "neutral", "positive"]
    assert df["review_id"].is_unique and df["rating"].between(1, 5).all()
    assert not df["combined_text"].str.contains("<").any()
    assert df.loc[df["user_id"] == "U3", "product_title"].iloc[0] == "Demo Gadget"
    assert (df.loc[df["user_id"] != "U3", "product_title"] == "").all()  # no metadata match = empty, not invented


def test_process_category_writes_parquet_with_the_declared_schema(category_files):
    rp.process_category("Demo")
    table = pq.read_table(category_files / "Demo.parquet")
    assert table.schema.names == REVIEW_PYARROW_SCHEMA.names and table.num_rows == 3
    assert table.schema.field("rating").type == REVIEW_PYARROW_SCHEMA.field("rating").type


def test_missing_raw_file_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "DATA_RAW_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Run acquisition first"):
        rp.process_category("Nothing")


# ---------- real data (skipped if not built) ----------
@pytest.mark.realdata
def test_real_processed_data_invariants(real_settings):
    df = pd.read_parquet(real_settings.reviews_parquet)
    assert df["review_id"].is_unique
    assert df["rating"].between(1, 5).all()
    expected = df["rating"].map(lambda r: "negative" if r <= 2 else "neutral" if r == 3 else "positive")
    assert (df["sentiment"] == expected).all(), "sentiment must be exactly the rating-derived label"
    assert not df["combined_text"].str.contains(r"<\s*/?\s*br\s*/?\s*>", case=False, regex=True).any()
    # The validator requires a body of at least 2 characters. (Search needs 3, so a handful of reviews such as "OK"
    # stay in the statistics but are not eligible for semantic search; the index reports eligible vs indexed.)
    assert df["combined_text"].str.strip().str.len().ge(2).all()
    assert not df["combined_text"].str.contains(r"\[\[(?:VIDEOID|ASIN):", case=False, regex=True).any(), "Amazon markers must be removed"
    # Amazon's automatic "Five Stars" titles are the label itself and must not be part of the model text.
    default_title = df["title"].fillna("").map(is_default_rating_title)
    assert default_title.sum() > 1000, "this data set is expected to contain many default titles"
    leaked = [t for t, title in zip(df.loc[default_title, "combined_text"], df.loc[default_title, "title"]) if t.lower().startswith(title.strip().lower() + ".")]
    assert not leaked, f"{len(leaked)} reviews still start with their default rating title"
    assert df["review_date"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()
    assert not df.duplicated(subset=["user_id", "asin", "timestamp"]).any()


# ---------- the manifest says what the processed data covers ----------
BACKSLASH = chr(92)  # a Windows path separator, written this way so no editor or shell can change it
@pytest.fixture
def full_pipeline_dirs(tmp_path, monkeypatch):
    import reviewpulse.acquisition.downloader as dl

    raw_dir, out_dir = tmp_path / "data" / "raw", tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True), out_dir.mkdir()
    monkeypatch.setattr(rp, "DATA_RAW_DIR", raw_dir)
    monkeypatch.setattr(rp, "DATA_PROCESSED_DIR", out_dir)
    manifest_path = tmp_path / "data" / "manifest.json"
    monkeypatch.setattr(dl, "MANIFEST_PATH", manifest_path)
    manifest_path.write_text(
        json.dumps({"categories": {"Demo": {"target_limit_requested": 7, "distinct_reviewers": 5, "review_file_path": BACKSLASH.join(["data", "raw", "Demo_reviews.jsonl"]), "sampling": {"strategy": "spread", "chunks": 10, "seed": 42, "lines_per_chunk": 1}}}}),
        encoding="utf-8",
    )
    lines = [
        json.dumps(raw(user_id="U1", asin="A1", rating=5.0, title="Five Stars", text="Works well")),
        json.dumps(raw(user_id="U2", asin="A2", rating=1.0, title="One Star", text="Broke at once")),
        json.dumps(raw(user_id="U3", asin="A3", rating=4.0, title="Solid", text="Good. [[VIDEOID:abc123]] Would buy again")),
        json.dumps(raw(user_id="U4", asin="A4", rating=3.0, title="Fine", text="It is fine")),
        json.dumps(raw(user_id="U5", asin="A5", rating=9.0)),  # invalid rating: rejected
    ]
    (raw_dir / "Demo_reviews.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return manifest_path


def test_manifest_records_the_scope_and_what_the_cleaning_did(full_pipeline_dirs):
    df = rp.run_full_pipeline(["Demo"])
    processing = json.loads(full_pipeline_dirs.read_text(encoding="utf-8"))["pipeline_processing"]
    scope = processing["scope"]
    assert len(df) == 4 and processing["total_retained_reviews"] == 4
    assert scope["categories"] == ["Demo"] and scope["category_count"] == 1
    assert scope["target_reviews_per_category"] == [7] and scope["downloaded_reviews"] == 5
    assert scope["retained_reviews"] == 4 and scope["rejected_reviews"] == 1
    assert scope["reviews_per_category"] == {"Demo": 4} and scope["distinct_reviewers_in_download"] == 5
    assert scope["review_dates"] == {"first": "2021-01-01", "last": "2021-01-01"}
    assert json.loads(scope["sampling"][0]) == {"chunks": 10, "seed": 42, "strategy": "spread"}
    assert scope["automatic_titles_left_out"] == 2 and scope["reviews_with_amazon_markers"] == 1
    report = processing["category_reports"]["Demo"]
    assert report["automatic_titles_left_out"] == 2 and report["reviews_with_amazon_markers"] == 1
    assert any("Five Stars" in rule for rule in scope["cleaning_rules"]) and any("VIDEOID" in rule for rule in scope["cleaning_rules"])


def test_manifest_paths_use_forward_slashes_on_every_machine(full_pipeline_dirs):
    rp.run_full_pipeline(["Demo"])
    manifest = json.loads(full_pipeline_dirs.read_text(encoding="utf-8"))
    assert manifest["categories"]["Demo"]["review_file_path"] == "data/raw/Demo_reviews.jsonl"
    assert "\\" not in manifest["pipeline_processing"]["processed_file"]
