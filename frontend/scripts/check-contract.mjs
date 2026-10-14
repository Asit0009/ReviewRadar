/**
 * Checks that the running backend still matches what src/api/types.ts expects.
 *
 *   npm run check:contract
 *
 * It reads GET /openapi.json from the API and verifies every endpoint, query parameter
 * and response field the frontend uses.
 */

// The app itself uses a relative base ("/api/v1"). This script runs in Node, so it needs a full address:
// set API_ORIGIN (default http://127.0.0.1:8000) or give VITE_API_BASE_URL as an absolute URL.
const configured = process.env.VITE_API_BASE_URL || ''
const apiBase = (configured.startsWith('http') ? configured : `${process.env.API_ORIGIN || 'http://127.0.0.1:8000'}/api/v1`).replace(/\/+$/, '')
const origin = new URL(apiBase).origin
const prefix = new URL(apiBase).pathname.replace(/\/+$/, '')

// path -> what the frontend relies on. `schemas` lists component schemas and the fields read from them.
const SCOPE = ['category', 'date_from', 'date_to']

const EXPECTED = {
  'GET /health': {
    schemas: {
      HealthResponse: ['status', 'timestamp_utc', 'version', 'artifacts_ready', 'total_reviews', 'retrieval', 'assistant', 'setup_steps'],
      RetrievalInfo: ['ready', 'indexed_reviews', 'eligible_reviews', 'coverage_pct', 'embedding_model', 'built_at', 'limited_by'],
      AssistantInfo: ['llm_configured', 'llm_model'],
    },
  },
  'GET /categories': {
    schemas: {
      CategoriesResponse: ['total_categories', 'categories'],
      CategoryItem: ['category', 'review_count', 'unique_products', 'mean_rating', 'verified_pct', 'mean_text_length'],
    },
  },
  'GET /overview': {
    params: SCOPE,
    schemas: {
      OverviewResponse: [
        'total_reviews', 'total_products', 'total_users', 'average_rating', 'date_min', 'date_max',
        'sentiment_distribution', 'sentiment_percentages', 'category_filter', 'date_from', 'date_to',
      ],
    },
  },
  'GET /sentiment/trend': {
    params: SCOPE,
    schemas: {
      SentimentTrendResponse: ['category_filter', 'date_from', 'date_to', 'total_data_points', 'trend'],
      SentimentTrendItem: ['year_month', 'review_count', 'avg_rating', 'positive_count', 'neutral_count', 'negative_count', 'positive_ratio', 'negative_ratio'],
    },
  },
  'GET /ratings/distribution': {
    params: SCOPE,
    schemas: {
      RatingDistributionResponse: ['category_filter', 'date_from', 'date_to', 'total_reviews', 'buckets'],
      RatingBucket: ['stars', 'count', 'pct'],
    },
  },
  'GET /topics': {
    params: [...SCOPE, 'sentiment'],
    schemas: {
      TopicsResponse: ['total_topics', 'topics', 'category_filter', 'sentiment_filter', 'date_from', 'date_to', 'reviews_considered', 'review_tags_available'],
      TopicItem: ['topic_id', 'label', 'named_by', 'top_words', 'sample_reviews', 'prevalence_count', 'prevalence_pct', 'avg_rating', 'sentiment_counts'],
    },
  },
  'GET /reviews': {
    params: [...SCOPE, 'page', 'limit', 'sentiment', 'min_rating', 'max_rating', 'search', 'verified_only', 'topic_id', 'sort', 'order'],
    schemas: {
      PaginatedReviewsResponse: ['total_records', 'page', 'limit', 'total_pages', 'reviews', 'sort', 'order'],
      ReviewItem: [
        'review_id', 'category', 'rating', 'sentiment', 'title', 'text', 'product_title', 'review_date',
        'verified_purchase', 'helpful_vote', 'topic_id', 'topic_label', 'topic_score',
      ],
    },
  },
  'POST /predict': {
    schemas: {
      PredictRequest: ['text', 'model_type'],
      PredictResponse: ['text', 'model_type', 'predicted_sentiment', 'sentiment_encoded', 'probabilities', 'model_used', 'is_rating_derived_label'],
    },
  },
  'GET /models/metrics': {
    params: ['category'],
    schemas: { ModelMetricsResponse: ['status', 'available_models', 'metrics', 'category_filter', 'category_metrics'] },
  },
  'GET /models/metrics/by-category': {
    schemas: { CategoryMetricsTable: ['categories', 'min_class_support_for_reliable', 'models'] },
  },
  'POST /assistant/chat': {
    schemas: {
      AssistantChatRequest: ['question', 'category', 'top_k'],
      AssistantChatResponse: [
        'response', 'llm_used', 'mode', 'status', 'llm_model', 'error_detail', 'numerical_facts',
        'evidence_reviews', 'evidence_count', 'retrieval',
      ],
      AssistantRetrievalInfo: ['indexed_reviews', 'eligible_reviews', 'coverage_pct', 'min_score', 'scope'],
    },
  },
  'GET /monitoring/status': {
    schemas: { MonitoringStatusResponse: ['status', 'reference_sample_size', 'current_sample_size', 'drifted_features_count', 'total_features_evaluated', 'feature_metrics', 'reference_period', 'current_period'] },
  },
}

const problems = []
let spec
try {
  const res = await fetch(`${origin}/openapi.json`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  spec = await res.json()
} catch (err) {
  console.error(`Could not read ${origin}/openapi.json (${err.message}). Is the backend running?`)
  process.exit(2)
}

for (const [key, expectation] of Object.entries(EXPECTED)) {
  const [method, path] = key.split(' ')
  const operation = spec.paths?.[`${prefix}${path}`]?.[method.toLowerCase()]
  if (!operation) {
    problems.push(`${key}: endpoint is missing from the API`)
    continue
  }
  const declared = new Set((operation.parameters ?? []).map((p) => p.name))
  for (const name of expectation.params ?? []) {
    if (!declared.has(name)) problems.push(`${key}: query parameter "${name}" is not accepted (API accepts: ${[...declared].join(', ') || 'none'})`)
  }
  for (const [schemaName, fields] of Object.entries(expectation.schemas)) {
    const props = Object.keys(spec.components?.schemas?.[schemaName]?.properties ?? {})
    if (props.length === 0) {
      problems.push(`${key}: schema ${schemaName} is missing`)
      continue
    }
    for (const field of fields) {
      if (!props.includes(field)) problems.push(`${key}: ${schemaName} has no field "${field}" (API has: ${props.join(', ')})`)
    }
  }
}

if (problems.length) {
  console.error(`Contract check FAILED against ${origin}:\n - ${problems.join('\n - ')}`)
  process.exit(1)
}
console.log(`Contract check passed: ${Object.keys(EXPECTED).length} endpoints match src/api/types.ts (${origin}${prefix}).`)
