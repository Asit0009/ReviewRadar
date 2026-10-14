# ReviewRadar

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end customer sentiment and review intelligence platform analyzing 30,000+ customer reviews from the **Amazon Reviews 2023** benchmark across 10 product categories.

ReviewRadar bridges the gap between raw natural language data and actionable product insights. It integrates data cleaning, classical machine learning baselines, transformer fine-tuning (DistilBERT), unsupervised topic extraction (NMF), sub-millisecond semantic search (FAISS), and an evidence-grounded retrieval-augmented generation (RAG) assistant powered by Google Gemini—served through a single FastAPI backend with an interactive React dashboard.

---

## Key Features

- **Interactive Analytics Dashboard**: Filter review volume, 1-to-5 star rating distributions, and sentiment trajectories across 10 retail categories and customizable date ranges.
- **Sentiment Classification Engine**:
  - **Classical Baselines**: TF-IDF paired with class-weighted Logistic Regression and Linear SVC.
  - **Deep Learning**: Fine-tuned `distilbert-base-uncased` with frozen lower layers and class-weighted loss to handle real-world review imbalance.
- **Unsupervised Topic Discovery**: 20 latent topics extracted using Non-Negative Matrix Factorization (NMF) with automated lexical naming and complaint-filtering.
- **Semantic Vector Search**: Dense vector index using `sentence-transformers/all-MiniLM-L6-v2` and FAISS exact inner-product (cosine) search for concept-based review lookup.
- **Evidence-Grounded RAG Assistant**: Hybrid question-answering combining exact DuckDB SQL aggregations with semantic vector retrieval. Synthesizes answers via Google Gemini, citing verified review IDs to prevent hallucinations.
- **Production-Ready Architecture**: Single-port deployment serving both the FastAPI REST endpoints and the compiled React frontend, DuckDB in-memory OLAP over Parquet, and strict Pydantic v2 schema validation.

---

## Architecture Overview

```text
[ McAuley Lab: Amazon Reviews 2023 (10 Categories) ]
                        │
                        ▼ (HTTP Range Sample Ingestion)
[ Data Validation & Cleaning (Pydantic, Regex, Deduplication) ]
                        │
                        ▼
         [ Processed Parquet Store (reviews_all.parquet) ]
                        │
         ┌──────────────┼──────────────┬──────────────┐
         ▼              ▼              ▼              ▼
   [ TF-IDF + SVC ] [ DistilBERT ] [ NMF (20 Topics) ] [ FAISS Vector Index ]
         │              │              │              │
         └──────────────┼──────────────┴──────────────┘
                        ▼
         [ FastAPI REST API (/api/v1) + DuckDB Engine ]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
[ React + TypeScript Dashboard ] [ Grounded Gemini RAG Copilot ]
```

---

## Dataset & Leak-Free Validation

Data is sourced from the **Amazon Reviews 2023** benchmark (McAuley Lab) evenly sampled across ten product categories:
*Automotive, Beauty & Personal Care, Cell Phones & Accessories, Electronics, Health & Household, Home & Kitchen, Office Products, Pet Supplies, Sports & Outdoors, Toys & Games.*

### Methodological Rigor:
- **Leak-Free Splitting**: To avoid optimistic evaluation bias, splitting is performed with `StratifiedGroupKFold` grouped strictly by `user_id`. No reviewer appears in more than one split (train, validation, or test).
- **Title Leak Prevention**: Amazon automatically generates star titles ("Five Stars", "One Star") when reviewers omit a title. In ~99.7% of these cases, the title gives away the ground truth rating. These automated placeholders are stripped from the model training text so models must learn from actual review semantics.
- **Handling Class Imbalance**: E-commerce reviews are heavily skewed positive (~76.5% 4–5★). All models utilize class weighting, and model selection prioritizes **Macro-F1** rather than raw accuracy.

---

## Quickstart

### Prerequisites
- **Python 3.10+**
- **Node.js 20+**

### 1. Clone the Repository
```bash
git clone https://github.com/Asit0009/ReviewRadar.git
cd ReviewRadar
```

### 2. Set Up Python Environment
```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional)
```bash
# Windows:
copy .env.example .env
# Linux/macOS:
cp .env.example .env
```

To enable natural-language answers in the AI Assistant, add your Google Gemini API key to `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Without an API key, the assistant runs in grounded-only mode, returning verified database statistics and matching review cards).*

### 4. Build & Run (Single Port Production)
```bash
# Build the React frontend
cd frontend
npm install
npm run build
cd ..

# Start the unified FastAPI server
python -m uvicorn reviewpulse.api.app:app --host 127.0.0.1 --port 8000
```

- **Dashboard:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

---

## Development Workflow

For active development with hot-module reloading:

1. **Terminal 1 (Backend API):**
   ```bash
   python -m uvicorn reviewpulse.api.app:app --host 127.0.0.1 --port 8000 --reload
   ```

2. **Terminal 2 (Frontend Dev Server):**
   ```bash
   cd frontend
   npm run dev
   ```
   Open [http://localhost:5173/](http://localhost:5173/). Vite is preconfigured to proxy `/api` calls directly to port `8000`.

---

## Reproducing the Pipeline from Scratch

All pipeline stages can be executed independently:

| Stage | Command | Description |
| :--- | :--- | :--- |
| **1. Ingestion** | `python -m reviewpulse.acquisition.downloader` | Samples ~3,000 reviews per category via HTTP range requests |
| **2. Cleaning** | `python -m reviewpulse.pipeline.run_pipeline` | Validates, normalizes, and produces `reviews_all.parquet` |
| **3. EDA** | `python -m reviewpulse.eda.run_eda` | Generates summary statistics and figures |
| **4. Classical Models** | `python -m reviewpulse.modeling.train_classical` | Trains TF-IDF + Logistic Regression and Linear SVC |
| **5. Topic Model** | `python -m reviewpulse.nlp.topic_model` | Fits 20-topic NMF and tags corpus |
| **6. Semantic Search** | `python -m reviewpulse.nlp.semantic_search --verify` | Generates dense embeddings and builds FAISS index |
| **7. DistilBERT** | `python -m reviewpulse.modeling.train_distilbert` | Fine-tunes DistilBERT transformer with PyTorch |
| **8. Reporting** | `python -m reviewpulse.modeling.report` | Recompiles benchmark metrics and markdown reports |

---

## Benchmark Results

Evaluated on the held-out test split (4,220 unseen reviews from unseen customer accounts):

| Model Architecture | Accuracy | Macro-F1 | Weighted-F1 | Negative-F1 | Neutral-F1 | Positive-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 83.93% | 0.6440 | 0.8453 | 0.7118 | 0.2984 | 0.9219 |
| **TF-IDF + Linear SVC** | 86.21% | 0.6319 | 0.8528 | 0.7343 | 0.2288 | 0.9326 |
| **DistilBERT (Fine-Tuned)** | **88.53%** | **0.7158** | **0.8909** | **0.8039** | **0.3906** | **0.9530** |

*Note: Macro-F1 is the primary comparison metric to reflect accuracy on hard minority classes (neutral and negative complaints).*

---

## API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | System health, loaded models, vector index coverage |
| `GET` | `/api/v1/categories` | Product category breakdown and review counts |
| `GET` | `/api/v1/overview` | High-level metrics, rating averages, sentiment split |
| `GET` | `/api/v1/sentiment/trend` | Monthly sentiment time-series counts |
| `GET` | `/api/v1/ratings/distribution` | Star rating distribution (1–5 stars) |
| `GET` | `/api/v1/topics` | Extracted topics, keyword weights, complaint filters |
| `GET` | `/api/v1/reviews` | Paginated review explorer with multi-filtering |
| `POST`| `/api/v1/predict` | Real-time sentiment prediction using any trained model |
| `GET` | `/api/v1/models/metrics` | Model benchmark metrics overall and by category |
| `POST`| `/api/v1/assistant/chat` | Grounded RAG query answering with review citations |
| `GET` | `/api/v1/monitoring/status` | Dataset drift detection (PSI & KS tests) |

Full schema documentation is detailed in [`docs/API_DOCUMENTATION.md`](docs/API_DOCUMENTATION.md) and interactive Swagger UI at `/docs`.

---

## Testing

```bash
# Run backend test suite
python -m pytest -q

# Run security & SQL injection prevention tests
python -m pytest tests/test_security.py -q

# Run frontend tests & linting
cd frontend
npm run lint
npm run typecheck
npm test
```

---

## Repository Structure

```text
ReviewRadar/
├── reviewpulse/               # Core Python package
│   ├── acquisition/          # Data ingestion from Hugging Face
│   ├── pipeline/             # Data cleaning, validation, and Parquet export
│   ├── eda/                  # Exploratory data analysis scripts
│   ├── modeling/             # Classical ML & DistilBERT training and inference
│   ├── nlp/                  # NMF topic modeling and FAISS semantic search
│   ├── assistant/            # Gemini client and hybrid RAG engine
│   ├── monitoring/           # Temporal data drift detection
│   └── api/                  # FastAPI app, schemas, routing, DuckDB store
├── frontend/                 # React 19 + TypeScript + Vite web application
├── data/                     # Raw and processed Parquet datasets
├── artifacts/                # Trained models, vector store, and topic weights
├── reports/                  # Detailed EDA and model benchmark summaries
├── tests/                    # Pytest test suite (unit, integration, security)
├── docs/                     # Architecture, schema, and API documentation
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Project packaging configuration
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
