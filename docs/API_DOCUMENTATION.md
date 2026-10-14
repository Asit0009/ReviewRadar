# API Documentation

ReviewRadar exposes a RESTful API built with **FastAPI**. All endpoints are mounted under `/api/v1`.

Interactive Swagger documentation is available at `/docs` and ReDoc at `/redoc` when the server is running.

---

## Base URL & General Rules

- **Base URL:** `http://127.0.0.1:8000/api/v1`
- **Format:** JSON (`Content-Type: application/json`)
- **Validation:** Powered by Pydantic v2 schemas. Missing required parameters or invalid parameter formats return HTTP `422 Unprocessable Entity`.
- **System Status:** If a requested model or dataset artifact is missing from disk, endpoints return HTTP `503 Service Unavailable` with details on the required generation command.

### Standard Filter Parameters
Several endpoints (`/overview`, `/sentiment/trend`, `/ratings/distribution`, `/topics`, `/reviews`) support standard scoping parameters:

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `category` | string | No | One of the 10 supported retail categories (case-insensitive) |
| `date_from` | string | No | Start date filter in `YYYY-MM-DD` format (inclusive) |
| `date_to` | string | No | End date filter in `YYYY-MM-DD` format (inclusive) |

---

## Endpoints

### 1. System Health
- **Route:** `GET /health`
- **Description:** Returns the runtime health of the service, list of loaded artifacts, vector index size, and assistant configuration.
- **Example Response:**
  ```json
  {
    "status": "healthy",
    "timestamp_utc": "2026-10-14T20:00:00Z",
    "version": "0.2.0",
    "artifacts_ready": {
      "parquet_data": true,
      "sentiment_models": true,
      "nmf_topics": true,
      "vector_search_faiss": true,
      "distilbert_model": true
    },
    "total_reviews": 29946,
    "retrieval": {
      "ready": true,
      "indexed_reviews": 29946,
      "eligible_reviews": 29946,
      "coverage_pct": 100.0,
      "embedding_model": "all-MiniLM-L6-v2"
    },
    "assistant": {
      "llm_configured": true,
      "llm_model": "gemini-3.5-flash-lite"
    }
  }
  ```

---

### 2. Categories
- **Route:** `GET /categories`
- **Description:** Lists all 10 product categories along with total review count, unique product count, mean rating, and verified purchase ratio.

---

### 3. Analytics Overview
- **Route:** `GET /overview`
- **Parameters:** `category`, `date_from`, `date_to`
- **Description:** Returns aggregate KPI numbers, average star ratings, and positive/neutral/negative counts for the selected scope.

---

### 4. Sentiment Trend
- **Route:** `GET /sentiment/trend`
- **Parameters:** `category`, `date_from`, `date_to`
- **Description:** Aggregates monthly review counts segmented by sentiment (positive, neutral, negative) and tracks average star ratings over time.

---

### 5. Star Rating Distribution
- **Route:** `GET /ratings/distribution`
- **Parameters:** `category`, `date_from`, `date_to`
- **Description:** Computes the count and percentage for each star level (1 to 5 stars).

---

### 6. Topic Discovery
- **Route:** `GET /topics`
- **Parameters:** `category`, `sentiment`, `date_from`, `date_to`
- **Description:** Returns the 20 extracted NMF topics, top associated terms, average star rating per topic, and review volume. Setting `sentiment=negative` retrieves the most common complaint topics.

---

### 7. Review Explorer
- **Route:** `GET /reviews`
- **Parameters:**
  - Standard filters (`category`, `date_from`, `date_to`)
  - `sentiment`: `"positive"`, `"neutral"`, or `"negative"`
  - `min_rating` / `max_rating`: `1.0` to `5.0`
  - `search`: Keyword search matching title or body
  - `verified_only`: Boolean (`true` / `false`)
  - `topic_id`: Integer topic identifier (`0` to `19`)
  - `sort`: `"date"`, `"rating"`, or `"helpful"`
  - `order`: `"asc"` or `"desc"`
  - `page`: Page index (default: `1`)
  - `limit`: Records per page (default: `20`, max: `100`)
- **Description:** Paginated review lookup with multi-attribute filtering.

---

### 8. Sentiment Inference
- **Route:** `POST /predict`
- **Request Body:**
  ```json
  {
    "text": "The battery life on this camera is incredible, lasted all weekend without recharging.",
    "model_type": "distilbert"
  }
  ```
  *(Supported `model_type` values: `"distilbert"`, `"logistic_regression"`, `"linear_svc"`)*
- **Response:**
  ```json
  {
    "sentiment": "positive",
    "confidence": 0.962,
    "model_type": "distilbert"
  }
  ```

---

### 9. Evidence-Grounded AI Assistant (RAG)
- **Route:** `POST /assistant/chat`
- **Request Body:**
  ```json
  {
    "query": "What are the common complaints about battery life in electronics?",
    "category": "Electronics"
  }
  ```
- **Description:** Performs hybrid retrieval combining DuckDB statistical aggregations with FAISS dense vector search (`all-MiniLM-L6-v2`). Synthesizes an evidence-based answer via Gemini, directly citing retrieved review IDs. If no Gemini API key is configured, the endpoint returns in `retrieval_only` mode with verified facts and review cards.

---

### 10. Monitoring & Drift Detection
- **Route:** `GET /monitoring/status`
- **Description:** Compares the oldest 70% of reviews (reference distribution) against the newest 30% (current distribution). Evaluates Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics across review text length, rating, and helpful vote distributions.
