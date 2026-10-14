"""Serving the React build from FastAPI: static files, client-side routes and the not-built message."""

import dataclasses

import pytest
from fastapi.testclient import TestClient

from data_factory import fake_encoder
from reviewpulse.api.app import create_app

INDEX_HTML = "<!doctype html><html><head><title>ReviewPulse test build</title></head><body><div id='root'></div><script src='/assets/app-abc123.js'></script></body></html>"


@pytest.fixture(scope="module")
def built_client(fixture_settings, tmp_path_factory):
    dist = tmp_path_factory.mktemp("dist")
    (dist / "assets").mkdir()
    (dist / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (dist / "assets" / "app-abc123.js").write_text("console.log('hi')", encoding="utf-8")
    (dist / "assets" / "style-def456.css").write_text("body{color:red}", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    (dist.parent / "secret.txt").write_text("TOP SECRET", encoding="utf-8")  # next to dist, must never be served
    settings = dataclasses.replace(fixture_settings, frontend_dist_dir=dist)
    with TestClient(create_app(settings, encoder=fake_encoder)) as tc:
        yield tc


def test_root_serves_the_react_app(built_client):
    resp = built_client.get("/")
    assert resp.status_code == 200 and "ReviewPulse test build" in resp.text
    assert resp.headers["content-type"].startswith("text/html") and resp.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize("route", ["/reviews", "/sentiment", "/topics", "/predictions", "/deep-learning", "/assistant", "/monitoring", "/reviews/deep/link", "/nonexistent-page"])
def test_client_side_routes_fall_back_to_index_html_so_refresh_works(built_client, route):
    resp = built_client.get(route)
    assert resp.status_code == 200 and "ReviewPulse test build" in resp.text


def test_hashed_assets_are_served_with_long_cache(built_client):
    js = built_client.get("/assets/app-abc123.js")
    assert js.status_code == 200 and js.text == "console.log('hi')"
    assert "javascript" in js.headers["content-type"]
    assert "immutable" in js.headers["cache-control"] and "max-age=31536000" in js.headers["cache-control"]
    css = built_client.get("/assets/style-def456.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")


def test_missing_asset_is_a_404_not_the_html_app(built_client):
    assert built_client.get("/assets/does-not-exist.js").status_code == 404


def test_root_level_static_files_are_served(built_client):
    resp = built_client.get("/favicon.svg")
    assert resp.status_code == 200 and "svg" in resp.headers["content-type"]


def test_api_still_works_next_to_the_frontend(built_client):
    assert built_client.get("/api/v1/health").json()["status"] == "healthy"
    assert built_client.get("/docs").status_code == 200
    assert built_client.get("/openapi.json").json()["info"]["title"] == "ReviewPulse API"


@pytest.mark.parametrize("path", ["/api/v1/nope", "/api/nope", "/api"])
def test_unknown_api_paths_stay_json_404s(built_client, path):
    resp = built_client.get(path)
    assert resp.status_code == 404 and resp.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize("path", ["/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/assets/../../secret.txt", "/assets/%2e%2e/%2e%2e/secret.txt"])
def test_path_traversal_cannot_read_files_outside_the_build(built_client, path):
    resp = built_client.get(path)
    assert "TOP SECRET" not in resp.text


def test_missing_build_shows_instructions_instead_of_an_error(client):
    resp = client.get("/")  # the shared fixture app has no dist folder
    assert resp.status_code == 200 and "npm run build" in resp.text
    assert client.get("/api/v1/health").status_code == 200


def test_default_settings_point_at_the_react_build_folder():
    from reviewpulse.config import FRONTEND_DIST_DIR

    assert FRONTEND_DIST_DIR.parts[-2:] == ("frontend", "dist")


@pytest.mark.realdata
def test_real_react_build_is_served_by_the_real_app(real_settings):
    """If frontend/dist exists, the production app must serve its real index.html and every asset it references."""
    import re

    index = real_settings.frontend_dist_dir / "index.html"
    if not index.exists():
        pytest.skip("frontend/dist has not been built")
    with TestClient(create_app(real_settings)) as tc:
        html = tc.get("/").text
        assert html == index.read_text(encoding="utf-8")
        assets = re.findall(r'(?:src|href)="(/[^"]+)"', html)
        assert assets, "index.html references no assets"
        for asset in assets:
            assert tc.get(asset).status_code == 200, asset
        assert tc.get("/reviews").text == html
