"""FastAPI application: JSON API under /api/v1 and the React dashboard at /.

One entry point serves everything in production:

    uvicorn reviewpulse.api.app:app --host 127.0.0.1 --port 8000

 * /api/v1/...      the JSON API (OpenAPI docs at /docs)
 * /assets/...      the built React files (hashed file names, cached for a year)
 * everything else  index.html, so client-side routes such as /reviews work on refresh

Build the dashboard first with `npm run build` in frontend/. If it has not been built, / explains how.
"""

import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from reviewpulse.api.routes import API_VERSION, router
from reviewpulse.api.store import ReviewStore
from reviewpulse.assistant.gemini_assistant import ReviewAssistant
from reviewpulse.assistant.gemini_client import GeminiClient
from reviewpulse.config import HOST, PORT, Settings, default_settings
from reviewpulse.modeling.inference import SentimentPredictors
from reviewpulse.nlp.semantic_search import Encoder, ReviewSemanticRetriever, ids_fingerprint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class _JsonFile:
    """Reads a JSON file and re-reads it only when it changes on disk (the training scripts rewrite these)."""

    def __init__(self, path: Path):
        self.path = path
        self._mtime: Optional[float] = None
        self._data: Optional[Dict[str, Any]] = None

    def get(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            self._mtime = mtime
        return self._data


def _warm_models(retriever: ReviewSemanticRetriever, predictors: SentimentPredictors) -> None:
    """Load the big models once, one after the other, in the background so the first requests are not slow."""
    for name, warm in (("embedding model", retriever.warm_up), ("DistilBERT", predictors.warm_up)):
        try:
            warm()
            logger.info("%s is loaded and warm.", name)
        except Exception as exc:  # a failed warm-up only means the first request loads that model itself
            logger.warning("%s warm-up failed: %s", name, exc)


class _ImmutableAssets(StaticFiles):
    """Vite gives built files content-hashed names, so browsers may cache them for a year."""

    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response


_NOT_BUILT_HTML = """<!doctype html><meta charset="utf-8"><title>ReviewPulse</title>
<body style="font-family:system-ui;max-width:40rem;margin:4rem auto;line-height:1.5">
<h1>ReviewPulse API is running</h1>
<p>The dashboard has not been built yet. From the repository root run:</p>
<pre>cd frontend
npm install
npm run build</pre>
<p>Then reload this page. The API documentation is at <a href="/docs">/docs</a>.</p></body>"""


def mount_frontend(app: FastAPI, dist: Path) -> bool:
    """Serve the React build from `dist` with a fallback to index.html for client-side routes."""
    index = dist / "index.html"
    if not index.exists():

        @app.get("/", include_in_schema=False)
        def not_built() -> HTMLResponse:
            return HTMLResponse(_NOT_BUILT_HTML, status_code=200)

        return False

    if (dist / "assets").is_dir():
        app.mount("/assets", _ImmutableAssets(directory=str(dist / "assets")), name="assets")
    root = dist.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        # Unknown API paths must stay JSON 404s, not the HTML app.
        if full_path.startswith("api/") or full_path == "api":
            raise HTTPException(status_code=404, detail="Not found")
        if full_path:
            candidate = (root / full_path).resolve()
            # resolve() + the parents check stops "../" tricks from reading files outside the build folder.
            if candidate.is_file() and root in candidate.parents:
                return FileResponse(candidate)
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return True


def create_app(
    settings: Optional[Settings] = None,
    encoder: Optional[Encoder] = None,
    gemini_client: Optional[GeminiClient] = None,
) -> FastAPI:
    """Application factory. Tests pass their own Settings (temporary data), a fake encoder and a mock Gemini client."""
    settings = settings or default_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initializing ReviewPulse (data in %s)...", settings.processed_dir)
        store = ReviewStore(settings.reviews_parquet, settings.review_topics_parquet)
        predictors = SentimentPredictors(settings.models_dir)
        retriever = ReviewSemanticRetriever(settings.vector_store_dir, encoder=encoder, min_score=settings.retrieval_min_score)
        app.state.settings = settings
        app.state.store = store
        app.state.predictors = predictors
        app.state.retriever = retriever
        # Load the big models in the background so the first question or prediction is not slow. Startup does not wait.
        threading.Thread(target=_warm_models, args=(retriever, predictors), name="model-warmup", daemon=True).start()

        topic_labels, metrics = _JsonFile(settings.topic_labels_json), _JsonFile(settings.metrics_json)
        app.state.topic_labels_doc = topic_labels.get
        app.state.metrics_doc = metrics.get
        app.state.topic_label_map = lambda: {t["topic_id"]: t["label"] for t in (topic_labels.get() or {}).get("topics", [])}
        app.state.allowed_categories = lambda: sorted(set(settings.categories) | set(store.known_categories()))

        # Fingerprint of the current data, to notice an index that was built from an older dataset.
        ids_cache: Dict[str, Any] = {}

        def eligible_ids_md5() -> Optional[str]:
            if not store.has_reviews:
                return None
            key = settings.reviews_parquet.stat().st_mtime
            if ids_cache.get("key") != key:
                ids_cache.update(key=key, value=ids_fingerprint(store.eligible_review_ids()))
            return ids_cache["value"]

        app.state.eligible_ids_md5 = eligible_ids_md5
        app.state.assistant = (
            ReviewAssistant(retriever=retriever, store=store, settings=settings, client=gemini_client) if store.has_reviews else None
        )
        if settings.gemini_configured:
            logger.info("Gemini is configured (model %s). Answers will be written by the model.", settings.gemini_model)
        else:
            logger.info("No GEMINI_API_KEY set: the assistant will return retrieval-only results.")
        yield
        logger.info("Shutting down ReviewPulse API...")

    app = FastAPI(
        title="ReviewPulse API",
        description=(
            "Amazon Reviews 2023 analytics: validated aggregates, sentiment prediction, topic discovery, "
            "semantic review search and a retrieval-grounded assistant."
        ),
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS is only needed when the page is served from a different origin than the API (the Vite dev server).
    # The production build is served by this same app, so it never triggers CORS. No cookies are used.
    origins = list(settings.cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if "*" in origins else origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Callable):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    app.include_router(router)
    mount_frontend(app, settings.frontend_dist_dir)  # must come after the API routes so /api/* wins
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("reviewpulse.api.app:app", host=HOST, port=PORT, reload=False)
