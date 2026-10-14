"""FastAPI router for ReviewPulse.

All database access goes through ReviewStore (bound SQL parameters) and all request parameters are validated in
reviewpulse/api/filters.py, so nothing a client sends is ever pasted into SQL.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from reviewpulse.api.filters import OrderName, SortName, resolve_category, review_filters, scope_filters, topic_filters
from reviewpulse.api.schemas import (
    AssistantChatRequest,
    AssistantChatResponse,
    CategoriesResponse,
    CategoryItem,
    CategoryMetricsTable,
    CategoryModelMetrics,
    HealthResponse,
    ModelMetricsResponse,
    MonitoringStatusResponse,
    OverviewResponse,
    PaginatedReviewsResponse,
    PredictRequest,
    PredictResponse,
    RatingBucket,
    RatingDistributionResponse,
    ReviewItem,
    SentimentTrendItem,
    SentimentTrendResponse,
    TopicItem,
    TopicsResponse,
)
from reviewpulse.api.store import ReviewFilters
from reviewpulse.modeling.evaluation import MIN_CLASS_SUPPORT_FOR_RELIABLE_METRICS
from reviewpulse.modeling.inference import ModelUnavailable
from reviewpulse.setup_steps import SETUP_COMMANDS, missing_steps
from reviewpulse.monitoring.drift_checker import run_drift_analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["ReviewPulse Core API"])

API_VERSION = "0.2.0"


def _require_data(request: Request) -> None:
    if not request.app.state.store.has_reviews:
        raise HTTPException(status_code=503, detail=f"The processed dataset is missing. Build it with: {SETUP_COMMANDS['parquet_data']} (see the README, 'Build the data and models from scratch').")


def _iso(d) -> Optional[str]:
    return d.isoformat() if d else None


def _pct(part: int, whole: int) -> float:
    return round(part * 100.0 / whole, 2) if whole else 0.0


# ================= 1. Health =================
@router.get("/health", response_model=HealthResponse)
def get_health(request: Request):
    """Liveness plus what the backend has loaded, including how much of the dataset search covers."""
    state = request.app.state
    store, predictors, retriever, settings = state.store, state.predictors, state.retriever, state.settings

    coverage = retriever.coverage(store.eligible_for_retrieval(), state.eligible_ids_md5())
    artifacts_ready = {
        "parquet_data": store.has_reviews,
        "sentiment_models": predictors.classical_ready,
        "vector_search_faiss": bool(coverage["ready"]),
        "nmf_topics": settings.topic_labels_json.exists() and store.has_topics,
        "distilbert_model": predictors.distilbert_artifacts_present(),
    }
    required = ("parquet_data", "sentiment_models", "vector_search_faiss", "nmf_topics")  # DistilBERT is optional
    return HealthResponse(
        status="healthy" if all(artifacts_ready[k] for k in required) else "degraded",
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        version=API_VERSION,
        artifacts_ready=artifacts_ready,
        total_reviews=store.total_reviews(),
        retrieval=coverage,
        assistant={"llm_configured": settings.gemini_configured, "llm_model": settings.gemini_model},
        setup_steps=missing_steps(artifacts_ready),
    )


# ================= 2. Categories =================
@router.get("/categories", response_model=CategoriesResponse)
def get_categories(request: Request):
    """Every category in the dataset with summary numbers."""
    _require_data(request)
    items = [CategoryItem(**row) for row in request.app.state.store.categories_summary()]
    return CategoriesResponse(total_categories=len(items), categories=items)


# ================= 3. Overview =================
@router.get("/overview", response_model=OverviewResponse)
def get_overview(request: Request, f: ReviewFilters = Depends(scope_filters)):
    """Headline numbers. Filter with ?category= and ?date_from= / ?date_to= (the filters change the SQL)."""
    _require_data(request)
    row = request.app.state.store.overview(f)
    total = int(row["total_reviews"])
    counts = {"positive": int(row["positive"]), "neutral": int(row["neutral"]), "negative": int(row["negative"])}
    return OverviewResponse(
        total_reviews=total,
        total_products=int(row["total_products"]),
        total_users=int(row["total_users"]),
        average_rating=row["average_rating"],
        date_min=row["date_min"],
        date_max=row["date_max"],
        sentiment_distribution=counts,
        sentiment_percentages={k: _pct(v, total) for k, v in counts.items()},
        category_filter=f.category,
        date_from=_iso(f.date_from),
        date_to=_iso(f.date_to),
    )


# ================= 4. Sentiment trend =================
@router.get("/sentiment/trend", response_model=SentimentTrendResponse)
def get_sentiment_trend(request: Request, f: ReviewFilters = Depends(scope_filters)):
    """Monthly review volume, average rating and positive / neutral / negative counts."""
    _require_data(request)
    rows = request.app.state.store.trend(f)
    items = [
        SentimentTrendItem(
            year_month=r["year_month"],
            review_count=int(r["review_count"]),
            avg_rating=r["avg_rating"],
            positive_count=int(r["positive_count"]),
            neutral_count=int(r["neutral_count"]),
            negative_count=int(r["negative_count"]),
            positive_ratio=round(r["positive_count"] / r["review_count"], 4),
            negative_ratio=round(r["negative_count"] / r["review_count"], 4),
        )
        for r in rows
    ]
    return SentimentTrendResponse(
        category_filter=f.category, date_from=_iso(f.date_from), date_to=_iso(f.date_to), total_data_points=len(items), trend=items
    )


# ================= 5. Rating distribution =================
@router.get("/ratings/distribution", response_model=RatingDistributionResponse)
def get_rating_distribution(request: Request, f: ReviewFilters = Depends(scope_filters)):
    """How many reviews gave 1, 2, 3, 4 and 5 stars. Always returns all five buckets."""
    _require_data(request)
    counts = request.app.state.store.rating_counts(f)
    total = sum(counts.values())
    buckets = [RatingBucket(stars=s, count=counts.get(s, 0), pct=_pct(counts.get(s, 0), total)) for s in (5, 4, 3, 2, 1)]
    return RatingDistributionResponse(
        category_filter=f.category, date_from=_iso(f.date_from), date_to=_iso(f.date_to), total_reviews=total, buckets=buckets
    )


# ================= 6. Topics =================
@router.get("/topics", response_model=TopicsResponse)
def get_topics(request: Request, f: ReviewFilters = Depends(topic_filters)):
    """NMF topics. Counts follow the filters, so ?sentiment=negative gives the topics of negative reviews."""
    state = request.app.state
    labels = state.topic_labels_doc()
    if labels is None:
        raise HTTPException(status_code=503, detail="Topic model is not trained yet. Run: python -m reviewpulse.nlp.topic_model")

    store = state.store
    tags_available = store.has_topics
    counts = {c["topic_id"]: c for c in store.topic_counts(f)} if tags_available else {}
    considered = sum(c["review_count"] for c in counts.values())
    items = []
    for t in labels["topics"]:
        c = counts.get(t["topic_id"])
        if tags_available:
            n = int(c["review_count"]) if c else 0
            items.append(
                TopicItem(
                    topic_id=t["topic_id"],
                    label=t["label"],
                    named_by=t.get("named_by", "top words"),
                    top_words=t["top_words"],
                    sample_reviews=t.get("sample_reviews", []),
                    prevalence_count=n,
                    prevalence_pct=_pct(n, considered),
                    avg_rating=c["avg_rating"] if c else None,
                    sentiment_counts={
                        "positive": int(c["positive_count"]) if c else 0,
                        "neutral": int(c["neutral_count"]) if c else 0,
                        "negative": int(c["negative_count"]) if c else 0,
                    },
                )
            )
        else:  # no per-review tags: fall back to the corpus-wide numbers saved with the model
            items.append(TopicItem(named_by=t.get("named_by", "top words"), **{k: t[k] for k in ("topic_id", "label", "top_words", "sample_reviews", "prevalence_count", "prevalence_pct")}))
    items.sort(key=lambda i: (-i.prevalence_count, i.topic_id))
    return TopicsResponse(
        total_topics=len(items),
        topics=items,
        category_filter=f.category,
        sentiment_filter=f.sentiment,
        date_from=_iso(f.date_from),
        date_to=_iso(f.date_to),
        reviews_considered=considered,
        review_tags_available=tags_available,
    )


# ================= 7. Reviews =================
@router.get("/reviews", response_model=PaginatedReviewsResponse)
def get_reviews(
    request: Request,
    f: ReviewFilters = Depends(review_filters),
    page: int = Query(1, ge=1, le=1_000_000, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Records per page"),
    sort: SortName = Query("date", description="date, rating or helpful"),
    order: OrderName = Query("desc", description="asc or desc"),
):
    """Paginated reviews. Every filter is applied in SQL: category, sentiment, rating range, date range, text, topic."""
    _require_data(request)
    store = request.app.state.store
    total = store.count_reviews(f)
    labels = request.app.state.topic_label_map()
    rows = store.reviews_page(f, sort, order, page, limit)
    items = []
    for r in rows:
        topic_id = r.pop("topic_id", None)
        score = r.pop("topic_score", None)
        items.append(
            ReviewItem(
                **r,
                topic_id=int(topic_id) if topic_id is not None else None,
                topic_label=labels.get(int(topic_id)) if topic_id is not None else None,
                topic_score=round(float(score), 3) if score is not None else None,
            )
        )
    return PaginatedReviewsResponse(
        total_records=total, page=page, limit=limit, total_pages=max(1, math.ceil(total / limit)), reviews=items, sort=sort, order=order
    )


# ================= 8. Predict =================
@router.post("/predict", response_model=PredictResponse)
def predict_sentiment(req: PredictRequest, request: Request):
    """Predict review sentiment with the model named in `model_type`. Unknown fields and unknown model names are rejected (422)."""
    try:
        result = request.app.state.predictors.predict(req.text, req.model_type)
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return PredictResponse(text=req.text, is_rating_derived_label=True, **result)


# ================= 9. Model metrics =================
@router.get("/models/metrics", response_model=ModelMetricsResponse)
def get_model_metrics(request: Request, category: Optional[str] = Query(None, max_length=80)):
    """Saved evaluation results. These are read from metrics.json, which only the training scripts write."""
    state = request.app.state
    data = state.metrics_doc()
    if data is None:
        raise HTTPException(status_code=503, detail="No evaluation results saved yet. Run: python -m reviewpulse.modeling.train_classical")
    canonical = resolve_category(request, category)

    metrics: Dict[str, Any] = {k: v for k, v in data.items() if k != "per_category"}
    category_metrics = None
    if canonical:
        per_model = {m: cats[canonical] for m, cats in data.get("per_category", {}).items() if canonical in cats}
        if per_model:
            category_metrics = CategoryModelMetrics(category=canonical, models=per_model)
    return ModelMetricsResponse(
        status="available",
        available_models=list(data.get("models", {}).keys()),
        metrics=metrics,
        category_filter=canonical,
        category_metrics=category_metrics,
    )


@router.get("/models/metrics/by-category", response_model=CategoryMetricsTable)
def get_model_metrics_by_category(request: Request):
    """Per-category test results for every model, with a flag saying which scores rest on enough reviews."""
    data = request.app.state.metrics_doc()
    if data is None:
        raise HTTPException(status_code=503, detail="No evaluation results saved yet. Run: python -m reviewpulse.modeling.train_classical")
    per_category = data.get("per_category", {})
    categories = sorted({c for cats in per_category.values() for c in cats})
    return CategoryMetricsTable(categories=categories, min_class_support_for_reliable=MIN_CLASS_SUPPORT_FOR_RELIABLE_METRICS, models=per_category)


# ================= 10. Assistant chat =================
@router.post("/assistant/chat", response_model=AssistantChatResponse)
def assistant_chat(req: AssistantChatRequest, request: Request):
    """Retrieval-grounded Q&A. `llm_used` says whether Gemini wrote the answer."""
    assistant = request.app.state.assistant
    if assistant is None:
        raise HTTPException(status_code=503, detail=f"The assistant needs the processed dataset, which is missing. Build it with: {SETUP_COMMANDS['parquet_data']}")
    category = resolve_category(request, req.category)
    return AssistantChatResponse(**assistant.answer_query(question=req.question.strip(), category=category, top_k=req.top_k))


# ================= 11. Monitoring status =================
@router.get("/monitoring/status", response_model=MonitoringStatusResponse)
def get_monitoring_status(request: Request):
    """Statistical data drift metrics (Kolmogorov-Smirnov test and PSI)."""
    _require_data(request)
    report = run_drift_analysis(reference_df=None, current_df=None, parquet_path=request.app.state.settings.reviews_parquet)
    if report.get("status") == "unavailable":
        raise HTTPException(status_code=503, detail=report.get("message"))
    return MonitoringStatusResponse(**report)
