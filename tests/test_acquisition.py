"""Spread sampling of the huge remote JSONL files, tested against an in-memory fake file (no network)."""

import json

import pytest

import reviewpulse.acquisition.downloader as dl


def make_remote_file(n_users=400, reviews_per_user=10):
    """A file laid out like the real ones: runs of reviews by the same user, one JSON object per line."""
    lines = []
    for u in range(n_users):
        for i in range(reviews_per_user):
            lines.append(json.dumps({"user_id": f"U{u:04d}", "asin": f"A{i}", "timestamp": 1_600_000_000_000 + u * 1000 + i, "rating": 5.0, "text": f"review {u}-{i} " + "x" * 50}))
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.fixture
def fake_remote(monkeypatch):
    data = make_remote_file()
    requests = []

    def fake_range_request(url, start, end, timeout=45):
        requests.append((start, end))
        return data[start : end + 1], len(data)

    monkeypatch.setattr(dl, "_range_request", fake_range_request)
    return data, requests


# ---------- chunk parsing ----------
def test_a_chunk_from_mid_file_drops_its_partial_first_and_last_lines():
    chunk = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n{"d": 4'  # starts mid-record, ends mid-record
    assert dl._lines_from_chunk(chunk, from_file_start=False, max_lines=10) == ['{"b": 2}', '{"c": 3}']


def test_a_chunk_from_the_start_of_the_file_keeps_its_first_line():
    chunk = b'{"a": 1}\n{"b": 2}\n{"c"'
    assert dl._lines_from_chunk(chunk, from_file_start=True, max_lines=10) == ['{"a": 1}', '{"b": 2}']


def test_invalid_json_lines_and_blanks_are_skipped_and_the_limit_is_respected():
    chunk = b'junk\n{"a": 1}\n\nnot json\n{"b": 2}\n{"c": 3}\n{"d"'
    assert dl._lines_from_chunk(chunk, from_file_start=True, max_lines=2) == ['{"a": 1}', '{"b": 2}']


# ---------- sampling ----------
def test_spread_sampling_returns_the_requested_number_of_valid_records(fake_remote, tmp_path):
    out = tmp_path / "r.jsonl"
    result = dl.sample_download_jsonl("http://x/file.jsonl", out, max_records=600, n_chunks=100, workers=4)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == result["records"] == 600
    assert all({"user_id", "asin", "timestamp", "rating", "text"} <= set(r) for r in rows)
    assert result["sampling"]["strategy"] == "spread" and result["sampling"]["chunks"] == 100


def test_requests_are_spread_across_the_whole_file(fake_remote, tmp_path):
    data, requests = fake_remote
    dl.sample_download_jsonl("http://x/file.jsonl", tmp_path / "r.jsonl", max_records=300, n_chunks=60, workers=4)
    starts = sorted(s for s, _ in requests if s > 0 or True)
    assert starts[0] == 0  # the first chunk starts at the beginning
    assert starts[-1] > 0.9 * len(data) - 70 * 1024  # and the last reaches the end of the file
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert max(gaps) < 3 * (len(data) / 60) + 1  # no huge unsampled region


def test_spread_sampling_finds_many_more_reviewers_than_reading_the_head(fake_remote, tmp_path):
    data, _ = fake_remote
    out = tmp_path / "r.jsonl"
    result = dl.sample_download_jsonl("http://x/file.jsonl", out, max_records=300, n_chunks=100, workers=4)
    head_users = len({json.loads(line)["user_id"] for line in data.decode().splitlines()[:300]})
    assert head_users == 30  # 10 reviews each: the head covers only 30 reviewers
    assert result["distinct_users"] > 3 * head_users


def test_sampling_is_reproducible_for_a_given_seed_and_differs_for_another(fake_remote, tmp_path):
    a, b, c = (tmp_path / n for n in ("a.jsonl", "b.jsonl", "c.jsonl"))
    dl.sample_download_jsonl("http://x/f", a, 200, n_chunks=50, seed=1, workers=2)
    dl.sample_download_jsonl("http://x/f", b, 200, n_chunks=50, seed=1, workers=2)
    dl.sample_download_jsonl("http://x/f", c, 200, n_chunks=50, seed=2, workers=2)
    assert a.read_text() == b.read_text() != c.read_text()


def test_duplicate_records_are_removed(monkeypatch, tmp_path):
    line = json.dumps({"user_id": "U1", "asin": "A1", "timestamp": 1, "rating": 5, "text": "same"}) + "\n"
    data = (line * 400).encode()
    monkeypatch.setattr(dl, "_range_request", lambda url, s, e, timeout=45: (data[s : e + 1], len(data)))
    out = tmp_path / "r.jsonl"
    result = dl.sample_download_jsonl("http://x/f", out, max_records=50, n_chunks=10, workers=2)
    assert result["records"] == 1  # all copies are the same (user, asin, timestamp) review


def test_a_failed_chunk_does_not_abort_the_download(monkeypatch, fake_remote, tmp_path):
    data, _ = fake_remote
    calls = {"n": 0}

    def flaky(url, start, end, timeout=45):
        calls["n"] += 1
        if start > 0 and calls["n"] % 7 == 0:
            raise OSError("connection reset")
        return data[start : end + 1], len(data)

    monkeypatch.setattr(dl, "_range_request", flaky)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    result = dl.sample_download_jsonl("http://x/f", tmp_path / "r.jsonl", 200, n_chunks=60, workers=1, max_retries=2)
    assert result["records"] >= 150  # a few chunks were lost, the rest were kept


def test_missing_file_size_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "_range_request", lambda url, s, e, timeout=45: (b"", 0))
    with pytest.raises(RuntimeError, match="file size"):
        dl.sample_download_jsonl("http://x/f", tmp_path / "r.jsonl", 10, n_chunks=2)


def test_unknown_sampling_mode_is_rejected():
    with pytest.raises(ValueError, match="spread"):
        dl.acquire_category_data("Electronics", sampling="random")
