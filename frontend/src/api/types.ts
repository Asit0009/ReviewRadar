/**
 * Types for the ReviewPulse backend. They mirror reviewpulse/api/schemas.py and the live OpenAPI schema
 * (GET /openapi.json). `npm run check:contract` compares them with the running backend.
 */

export type Sentiment = 'positive' | 'neutral' | 'negative'

// ---------- /health ----------
export interface RetrievalInfo {
  ready: boolean
  indexed_reviews: number
  eligible_reviews: number
  coverage_pct: number
  embedding_model: string | null
  built_at: string | null
  /** Why the index is smaller than the dataset ("max_samples", "memory", "stale_index"), or null for full coverage. */
  limited_by: string | null
}
export interface AssistantInfo {
  llm_configured: boolean
  llm_model: string
}
export interface HealthResponse {
  status: string // "healthy" | "degraded"
  timestamp_utc: string
  version: string
  artifacts_ready: Record<string, boolean>
  total_reviews: number
  retrieval: RetrievalInfo
  assistant: AssistantInfo
  /** For each missing file, the command that builds it. Empty when everything is in place. */
  setup_steps: Record<string, string>
}

// ---------- shared filters ----------
/** Filters accepted by /overview, /sentiment/trend, /ratings/distribution and (with more fields) /topics and /reviews. */
export interface ScopeParams {
  category?: string | null
  date_from?: string | null // YYYY-MM-DD
  date_to?: string | null
}

// ---------- /categories ----------
export interface CategoryItem {
  category: string
  review_count: number
  unique_products: number
  mean_rating: number
  verified_pct: number
  mean_text_length: number
}
export interface CategoriesResponse {
  total_categories: number
  categories: CategoryItem[]
}

// ---------- /overview ----------
export interface OverviewResponse {
  total_reviews: number
  total_products: number
  total_users: number
  /** null when the filters match no reviews. */
  average_rating: number | null
  date_min: string | null
  date_max: string | null
  sentiment_distribution: Record<Sentiment, number>
  sentiment_percentages: Record<Sentiment, number>
  category_filter: string | null
  date_from: string | null
  date_to: string | null
}

// ---------- /sentiment/trend ----------
export interface TrendPoint {
  year_month: string // "2023-03"
  review_count: number
  avg_rating: number
  positive_count: number
  neutral_count: number
  negative_count: number
  positive_ratio: number
  negative_ratio: number
}
export interface TrendResponse {
  category_filter: string | null
  date_from: string | null
  date_to: string | null
  total_data_points: number
  trend: TrendPoint[]
}

// ---------- /ratings/distribution ----------
export interface RatingBucket {
  stars: 1 | 2 | 3 | 4 | 5
  count: number
  pct: number
}
export interface RatingDistributionResponse {
  category_filter: string | null
  date_from: string | null
  date_to: string | null
  total_reviews: number
  buckets: RatingBucket[]
}

// ---------- /topics ----------
export interface TopicItem {
  topic_id: number
  label: string
  /** 'reviewed': the name was chosen by reading the topic's reviews. 'top words': made automatically from its top words. */
  named_by: 'reviewed' | 'top words'
  top_words: string[]
  sample_reviews: string[]
  /** Counts follow the request filters (category, sentiment, dates). */
  prevalence_count: number
  prevalence_pct: number
  avg_rating: number | null
  sentiment_counts: Record<Sentiment, number>
}
export interface TopicsParams extends ScopeParams {
  sentiment?: Sentiment | null
}
export interface TopicsResponse {
  total_topics: number
  topics: TopicItem[]
  category_filter: string | null
  sentiment_filter: string | null
  date_from: string | null
  date_to: string | null
  reviews_considered: number
  review_tags_available: boolean
}

// ---------- /reviews ----------
export type ReviewSort = 'date' | 'rating' | 'helpful'
export type SortOrder = 'asc' | 'desc'
export interface ReviewItem {
  review_id: string
  category: string
  rating: number
  sentiment: Sentiment | string
  title: string
  text: string
  product_title: string | null
  review_date: string // YYYY-MM-DD
  verified_purchase: boolean
  helpful_vote: number
  topic_id: number | null
  topic_label: string | null
  topic_score: number | null
}
export interface ReviewsResponse {
  total_records: number
  page: number
  limit: number
  total_pages: number
  reviews: ReviewItem[]
  sort: string
  order: string
}
export interface ReviewsParams extends ScopeParams {
  page?: number
  limit?: number
  sentiment?: Sentiment | null
  min_rating?: number | null
  max_rating?: number | null
  search?: string | null
  verified_only?: boolean | null
  topic_id?: number | null
  sort?: ReviewSort
  order?: SortOrder
}

// ---------- /predict ----------
export type PredictModel = 'logistic_regression' | 'linear_svc' | 'distilbert'
export interface PredictRequest {
  text: string
  model_type: PredictModel
}
export interface PredictResponse {
  text: string
  /** The model that produced this prediction. It always equals the model that was requested. */
  model_type: PredictModel
  predicted_sentiment: Sentiment | string
  sentiment_encoded: number
  /** null for LinearSVC, which has no predict_proba. */
  probabilities: Record<Sentiment, number> | null
  model_used: string
  is_rating_derived_label: boolean
}

// ---------- /models/metrics ----------
export interface SplitMetrics {
  accuracy?: number
  macro_f1?: number
  weighted_f1?: number
  negative_f1?: number
  neutral_f1?: number
  positive_f1?: number
  negative_precision?: number
  neutral_precision?: number
  positive_precision?: number
  negative_recall?: number
  neutral_recall?: number
  positive_recall?: number
  macro_precision?: number
  macro_recall?: number
  confusion_matrix?: number[][]
}
export interface ClassicalModelMetrics {
  validation?: SplitMetrics
  test?: SplitMetrics
  hyperparameters?: { C?: number; class_weight?: string }
}
export interface DistilbertTraining {
  base_model?: string
  device?: string
  train_samples?: number
  train_split_size?: number
  epochs?: number
  batch_size?: number
  gradient_accumulation_steps?: number
  max_length?: number
  learning_rate?: number
  class_weighted_loss?: boolean
  frozen_word_embeddings?: boolean
  loss_per_epoch?: number[]
  training_seconds?: number
}
export interface DistilbertMetrics extends ClassicalModelMetrics {
  model_name?: string
  evaluated_on?: { validation_samples?: number; test_samples?: number }
  training?: DistilbertTraining
  trained_at?: string
  artifacts_saved_to?: string
}
export interface SplitInfo {
  source_rows?: number
  total_deduplicated_samples?: number
  train_samples?: number
  val_samples?: number
  test_samples?: number
  class_distribution_train?: Record<Sentiment, number>
  class_distribution_val?: Record<Sentiment, number>
  class_distribution_test?: Record<Sentiment, number>
  /** e.g. "StratifiedGroupKFold by user_id (7 folds: 1 test, 1 validation, rest train)" */
  split_strategy?: string
  train_reviewers?: number
  val_reviewers?: number
  test_reviewers?: number
  reviewers_shared_between_splits?: number
}
export interface ModelMetricsPayload {
  split_info?: SplitInfo
  trained_at?: string
  updated_at?: string
  models?: {
    logistic_regression?: ClassicalModelMetrics
    linear_svc?: ClassicalModelMetrics
    distilbert?: DistilbertMetrics
  }
}
export interface CategoryModelResult {
  test_samples: number
  class_support: Record<Sentiment, number>
  reliable: boolean
  accuracy: number
  macro_f1: number
  negative_f1: number
  neutral_f1: number
  positive_f1: number
  note?: string
}
export interface ModelMetricsResponse {
  status: string
  available_models: string[]
  metrics: ModelMetricsPayload
  category_filter: string | null
  category_metrics: { category: string; models: Record<string, CategoryModelResult> } | null
}
export interface CategoryMetricsTable {
  categories: string[]
  min_class_support_for_reliable: number
  models: Record<string, Record<string, CategoryModelResult>>
}

// ---------- /assistant/chat ----------
export interface AssistantChatRequest {
  question: string
  category?: string | null
  top_k?: number
}
export interface EvidenceReview {
  review_id: string
  vector_id: number
  score: number // cosine similarity from FAISS
  rating: number
  sentiment: Sentiment | string
  category: string
  title: string
  text: string // truncated excerpt
  full_text: string
  product_title: string | null
  timestamp_utc: string
  verified_purchase: boolean
  helpful_vote: number
}
export interface NumericalFacts {
  scope?: string
  total_reviews?: number
  unique_products?: number
  average_rating?: number | null
  verified_purchase_pct?: number
  positive_pct?: number
  neutral_pct?: number
  negative_pct?: number
}
export type AssistantStatus =
  | 'success'
  | 'setup_required'
  | 'invalid_key'
  | 'rate_limited'
  | 'timeout'
  | 'network_error'
  | 'service_error'
  | 'blocked'
  | 'api_error'
export interface AssistantRetrieval {
  indexed_reviews: number
  eligible_reviews: number
  coverage_pct: number
  min_score: number
  scope: string
}
export interface AssistantChatResponse {
  response: string
  /** True only when Gemini wrote `response`. */
  llm_used: boolean
  mode: 'llm' | 'retrieval_only'
  status: AssistantStatus
  /** Name of the model that wrote the answer; null unless llm_used. */
  llm_model: string | null
  error_detail: string | null
  numerical_facts: NumericalFacts
  evidence_reviews: EvidenceReview[]
  evidence_count: number
  retrieval: AssistantRetrieval
}

// ---------- /monitoring/status ----------
export interface FeatureDrift {
  metric: string
  ks_statistic?: number
  ks_pvalue?: number
  psi: number
  drift_detected: boolean
}
export interface MonitoringResponse {
  status: string // "healthy" | "drift_detected"
  reference_sample_size: number
  current_sample_size: number
  drifted_features_count: number
  total_features_evaluated: number
  feature_metrics: Record<string, FeatureDrift>
  /** Oldest 70% of reviews (reference) and newest 30% (current), by review date. */
  reference_period: { from: string; to: string } | null
  current_period: { from: string; to: string } | null
}
