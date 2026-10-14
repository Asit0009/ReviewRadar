"""Pydantic schemas and API contracts for ReviewPulse.

The React frontend's types (frontend/src/api/types.ts) mirror these models. After changing anything here,
run `npm run check:contract` in frontend/ to confirm the two still agree.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SentimentName = Literal["positive", "neutral", "negative"]
PredictModelName = Literal["logistic_regression", "linear_svc", "distilbert"]


# ================= Health =================
class RetrievalInfo(BaseModel):
    """How much of the dataset the semantic search index actually covers."""

    ready: bool
    indexed_reviews: int = 0
    eligible_reviews: int = 0
    coverage_pct: float = 0.0
    embedding_model: Optional[str] = None
    built_at: Optional[str] = None
    # Why the index is smaller than the dataset, if it is (e.g. "max_samples", "memory"). None means full coverage.
    limited_by: Optional[str] = None


class AssistantInfo(BaseModel):
    llm_configured: bool
    llm_model: str


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    timestamp_utc: str
    version: str = Field(..., example="0.2.0")
    artifacts_ready: Dict[str, bool]
    total_reviews: int = 0
    retrieval: RetrievalInfo
    assistant: AssistantInfo
    # For every artifact that is missing, the command that builds it. Empty when everything is in place.
    setup_steps: Dict[str, str] = Field(default_factory=dict)


# ================= Categories =================
class CategoryItem(BaseModel):
    category: str
    review_count: int
    unique_products: int
    mean_rating: float
    verified_pct: float
    mean_text_length: float


class CategoriesResponse(BaseModel):
    total_categories: int
    categories: List[CategoryItem]


# ================= Overview =================
class OverviewResponse(BaseModel):
    total_reviews: int
    total_products: int
    total_users: int
    # None when the filters match no reviews (there is no average of nothing).
    average_rating: Optional[float]
    date_min: Optional[str]
    date_max: Optional[str]
    sentiment_distribution: Dict[str, int]
    sentiment_percentages: Dict[str, float]
    category_filter: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


# ================= Sentiment trend =================
class SentimentTrendItem(BaseModel):
    year_month: str
    review_count: int
    avg_rating: float
    positive_count: int
    neutral_count: int
    negative_count: int
    positive_ratio: float
    negative_ratio: float


class SentimentTrendResponse(BaseModel):
    category_filter: Optional[str]
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    total_data_points: int
    trend: List[SentimentTrendItem]


# ================= Rating distribution =================
class RatingBucket(BaseModel):
    stars: int
    count: int
    pct: float


class RatingDistributionResponse(BaseModel):
    category_filter: Optional[str]
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    total_reviews: int
    buckets: List[RatingBucket]


# ================= Topics =================
class TopicSentimentCounts(BaseModel):
    positive: int = 0
    neutral: int = 0
    negative: int = 0


class TopicItem(BaseModel):
    topic_id: int
    label: str
    # "reviewed": the name was chosen by reading the topic's reviews. "top words": made automatically from its top words.
    named_by: str = "top words"
    top_words: List[str]
    sample_reviews: List[str]
    # Counts below follow the request filters (category, sentiment, dates), not the whole corpus.
    prevalence_count: int
    prevalence_pct: float
    avg_rating: Optional[float] = None
    sentiment_counts: TopicSentimentCounts = TopicSentimentCounts()


class TopicsResponse(BaseModel):
    total_topics: int
    topics: List[TopicItem]
    category_filter: Optional[str] = None
    sentiment_filter: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    reviews_considered: int = 0
    # True when every review has a topic tag (review-level assignments exist).
    review_tags_available: bool = False


# ================= Reviews =================
class ReviewItem(BaseModel):
    review_id: str
    category: str
    rating: float
    sentiment: str
    title: str
    text: str
    product_title: Optional[str]
    review_date: str
    verified_purchase: bool
    helpful_vote: int
    topic_id: Optional[int] = None
    topic_label: Optional[str] = None
    topic_score: Optional[float] = None


class PaginatedReviewsResponse(BaseModel):
    total_records: int
    page: int
    limit: int
    total_pages: int
    reviews: List[ReviewItem]
    sort: str = "date"
    order: str = "desc"


# ================= Predict =================
class StrictRequest(BaseModel):
    """Request bodies refuse fields they do not know. A typo such as `model` instead of `model_type` is a 422 error;
    it must never be ignored while a default is quietly used instead."""

    model_config = ConfigDict(extra="forbid")


class PredictRequest(StrictRequest):
    text: str = Field(..., min_length=2, max_length=5000, example="This blender broke after three uses. Terrible customer support.")
    # Required on purpose: the caller always names the model, and an unknown name is rejected, never replaced.
    model_type: PredictModelName = Field(..., description="Which saved model to use.", example="logistic_regression")

    @field_validator("text")
    @classmethod
    def text_needs_real_content(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("text must contain at least 2 characters that are not spaces")
        return value


class PredictResponse(BaseModel):
    text: str
    model_type: PredictModelName = Field(..., description="The model that produced this prediction (the one that was requested).", example="logistic_regression")
    predicted_sentiment: str = Field(..., example="negative")
    sentiment_encoded: int = Field(..., example=0)
    probabilities: Optional[Dict[str, float]]
    model_used: str
    is_rating_derived_label: bool = True


# ================= Model Metrics =================
class CategoryModelMetrics(BaseModel):
    category: str
    # Test-set results for this category only, per model.
    models: Dict[str, Dict[str, Any]]


class ModelMetricsResponse(BaseModel):
    status: str
    available_models: List[str]
    metrics: Dict[str, Any]
    category_filter: Optional[str] = None
    # Present when ?category= was given and per-category results were saved with the metrics.
    category_metrics: Optional[CategoryModelMetrics] = None


class CategoryMetricsTable(BaseModel):
    """Test-set results for every model in every category (the per-category comparison table)."""

    categories: List[str]
    min_class_support_for_reliable: int
    # model name -> category -> {test_samples, class_support, reliable, accuracy, macro_f1, ...}
    models: Dict[str, Dict[str, Dict[str, Any]]]


# ================= Assistant =================
class AssistantChatRequest(StrictRequest):
    question: str = Field(..., min_length=2, max_length=500, example="What are customers saying about build quality and durability?")
    category: Optional[str] = Field(None, max_length=80, example="Electronics")
    top_k: int = Field(4, ge=1, le=10)


AssistantStatus = Literal[
    "success",  # Gemini wrote the answer
    "setup_required",  # no API key configured on the server
    "invalid_key",  # Gemini rejected the key
    "rate_limited",  # Gemini returned 429
    "timeout",  # Gemini did not answer in time
    "network_error",  # could not reach Gemini
    "service_error",  # Gemini returned a 5xx error
    "blocked",  # Gemini refused to answer (safety filter)
    "api_error",  # anything else
]


class AssistantRetrievalInfo(BaseModel):
    indexed_reviews: int
    eligible_reviews: int
    coverage_pct: float
    min_score: float
    scope: str


class AssistantChatResponse(BaseModel):
    response: str
    llm_used: bool
    mode: Literal["llm", "retrieval_only"]
    status: AssistantStatus
    # Name of the model that wrote `response`. None unless llm_used is true.
    llm_model: Optional[str] = None
    error_detail: Optional[str] = None
    numerical_facts: Dict[str, Any]
    evidence_reviews: List[Dict[str, Any]]
    evidence_count: int
    retrieval: AssistantRetrievalInfo


# ================= Monitoring =================
class MonitoringStatusResponse(BaseModel):
    status: str
    reference_sample_size: int
    current_sample_size: int
    drifted_features_count: int
    total_features_evaluated: int
    feature_metrics: Dict[str, Any]
    # The oldest 70% of reviews (reference) and newest 30% (current) that were compared.
    reference_period: Optional[Dict[str, str]] = None
    current_period: Optional[Dict[str, str]] = None
