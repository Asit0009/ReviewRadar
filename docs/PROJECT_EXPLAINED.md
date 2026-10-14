# ReviewRadar Explained

ReviewRadar takes real Amazon customer reviews (the Amazon Reviews 2023 dataset from McAuley Lab), cleans them, analyses them, trains sentiment models, finds topics, lets you search reviews by meaning, and shows everything in a web dashboard. This page walks through each stage in the order the data flows. Detailed counts and figures are documented in `reports/data_summary.md` and `reports/model_evaluation/model_evaluation_report.md`.

---

## 1. The big picture

```
Hugging Face (Amazon Reviews 2023, 10 categories)
        |  spread-out HTTP range requests
        v
 data/raw/*.jsonl  ---- validate + clean ---->  data/processed/reviews_all.parquet
                                                        |
        +-----------------------+-----------------------+----------------------+
        v                       v                       v                      v
   EDA report          sentiment models          topic model (NMF)      search index (FAISS)
 (reports/eda)     (TF-IDF models, DistilBERT)   tags every review      embeds every review
        |                       |                       |                      |
        +-----------------------+-----------+-----------+----------------------+
                                            v
                                FastAPI backend  (reviewpulse/api)
                     JSON API at /api/v1  +  React dashboard at /
                                            ^
                                            |
                          Gemini (optional) writes assistant answers
```

Everything the dashboard shows comes from a backend response. Anything the backend cannot calculate is shown as "Not available yet" instead of a made-up value.

---

## 2. Stage by stage

### Stage 1. Getting the data (`reviewpulse/acquisition/downloader.py`)
The source files are huge (5 to 30 GB each), so only a sample is downloaded: about 3,000 reviews per category. The downloader asks the server for roughly 500 small pieces spread evenly over each file, which gives many different reviewers and years. (The first version read only the first lines of each file, which came from just a few hundred people per category; see TECH_CHOICES.) It records source URLs, file checksums, sampling settings and the computer's RAM and disk in `data/manifest.json`.

### Stage 2. Validation and cleaning (`reviewpulse/pipeline`)
Every raw line is checked: valid JSON, required fields present, rating between 1 and 5, non-empty text, a sensible timestamp. Bad records are dropped and counted by reason. Duplicates are removed. Text is cleaned: HTML entities and tags such as `<br />` are removed, Amazon's own `[[VIDEOID:...]]` and `[[ASIN:...]]` markers are removed, and whitespace is tidied. The sentiment label is derived from the star rating. The result is one Parquet file with a fixed schema.

**Preventing label leakage from automatic titles.** When a reviewer writes no title, Amazon fills in the star rating as words ("Five Stars", "One Star"). About one review in ten had such a title, and in 99.7% of them the number equals the rating. Because the model text is "title. body", the models could read the answer directly off the title. The cleaner now strips these automatic titles out of the model text (the `title` column still preserves them on the review card).

### Stage 3. Exploratory analysis (`reviewpulse/eda/run_eda.py`)
DuckDB computes statistics (ratings, categories, review length, verified purchases, reviews over time) and Matplotlib/Seaborn draw the charts. `reports/eda/eda_report.md` is written from the computed numbers. The main finding to keep in mind: most reviews are positive, so accuracy alone is a poor way to judge a sentiment model.

### Stage 4. Sentiment models (`reviewpulse/modeling`)
- **Split:** identical texts are removed, then reviews are split into train, validation and test **by reviewer**, so no person appears in two splits.
- **TF-IDF + Logistic Regression** and **TF-IDF + Linear SVC:** the regularisation strength is chosen on the validation split; the test split is scored once.
- **DistilBERT:** a pretrained transformer fine-tuned for this task. On a CPU it can only use part of the training data; the saved result records how much. If it has not been trained, no score is shown.
- Results are written to `artifacts/models/metrics.json`, including scores per category and a flag for scores that rest on too few reviews. `reviewpulse/modeling/report.py` turns that file into the evaluation report.

### Stage 5. Topics (`reviewpulse/nlp/topic_model.py`)
TF-IDF + NMF finds groups of words that appear together: 20 topics, using the 8,000 most frequent words and two-word phrases. Each review is tagged with its strongest topic (`review_topics.parquet`). That lets the API filter by topic and show topic counts for any category, date range or sentiment. "Top complaints" are the topics of negative reviews.

The settings were chosen by comparing five setups with three random seeds each (`python -m reviewpulse.nlp.topic_experiments`). The first version (14 topics) put about a third of all reviews into one catch-all topic. Using 20 topics and dropping praise-only words ("excellent", "thank") shrank the biggest topic to roughly a quarter while keeping the same word coherence. A catch-all topic is still there, and it is named honestly ("General use and results (mixed products)") because its reviews really have no single theme. Topic names were chosen by reading the reviews of each topic; a name is only used if its key words are among the topic's top words. Everything, with example reviews, is in `reports/topic_report.md`.

### Stage 6. Semantic search (`reviewpulse/nlp/semantic_search.py`)
A sentence-transformer model turns every review into a vector; FAISS finds the vectors closest to a question. By default all reviews are indexed. The index description (`index_info.json`) says how many reviews it covers, and the API reports it. `--verify` checks that vectors really belong to the review IDs stored next to them.

### Stage 7. The assistant (`reviewpulse/assistant`)
For a question the backend (1) computes exact numbers with SQL for the chosen category, (2) retrieves the most similar reviews, and (3) if a Gemini key is configured, asks Gemini to answer using only those. The response always says whether Gemini wrote it (`llm_used`, `mode`) and why not if it did not. Reviews must be at least 0.40 similar to the question to count as evidence, so an unrelated question gets no evidence and an honest "not enough evidence" answer. In the dashboard, the review IDs that the answer cites are links to the evidence cards, and an ID that is not among the shown evidence is marked, so an invented citation is visible. See TECH_CHOICES section 7.

### Stage 8. The API (`reviewpulse/api`)
FastAPI exposes health, categories, overview, monthly sentiment trend, rating distribution, topics, reviews, prediction, model metrics (overall and per category), assistant chat and drift monitoring. Request values are validated and then passed to SQL as bound parameters. Request bodies are strict: an unknown field (for example `model` instead of `model_type` in `/predict`) is rejected with a 422 error and never ignored. If a saved file is missing, the endpoint answers 503 with the exact command that builds it, and `/health` lists every missing step. The same app serves the built React dashboard.

### Stage 9. The dashboard (`frontend/`)
React + TypeScript. Pages: Overview, Reviews Analysis, Sentiment Analysis, Topic Modeling, ML Predictions, Deep Learning (NLP), LLM Assistant, Deploy & Monitor. Category and date filters are sent to the API with every request, so every number, chart and review list is recalculated by the backend.

### Stage 10. Monitoring and tests
- **Drift:** oldest 70% versus newest 30% of reviews, using PSI and a KS test with an effect-size rule. Exposed at `/api/v1/monitoring/status` and on Deploy & Monitor.
- **Tests:** backend `pytest` (unit, API, security, retrieval, models, real-data reconciliation) and frontend `vitest`, plus a contract check between the two. See the README for commands and test suite details.

---

## 3. Things this project does not do

- It does not label sentiment from the text: labels come from star ratings.
- The reviews are a sample of the source files, not a random sample of all Amazon reviews.
- Product titles are mostly unavailable (product metadata is skipped by default).
- Dates are filtered by review date only; there is no "previous period" comparison.
- There is no user login and no Docker setup.
- The drift check compares older and newer reviews in one file; it is not a live stream of new data.
