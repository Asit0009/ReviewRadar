# Technology Choices

This page explains why each tool was picked, what else could have been used, and what the architectural trade-offs were. Specific model metrics and dataset statistics are documented in the `reports/` directory.

---

## 1. Data: Parquet files, DuckDB and pandas

**What we use.** The cleaned reviews are stored in one Parquet file. DuckDB runs SQL on it. pandas is used in the data pipeline and for the exploratory analysis (EDA).

**Why.**
- A Parquet file stores data by column and compresses it, so it is small and fast to scan. Every column has a fixed type, which catches mistakes early.
- DuckDB is a database that runs inside our Python process. There is no server to install, and it reads the Parquet file directly. This is exactly what an analytics dashboard needs: counts, averages and group-by queries over a few tens of thousands of rows.
- pandas is convenient for cleaning and for scikit-learn, which expects DataFrames or arrays.

**Alternatives.** SQLite is simpler but stores rows, so analytical queries are slower. PostgreSQL would work but needs a running server, which is more than this project needs. Keeping everything in pandas in memory would work for small data but cannot run SQL filters safely or share the data between requests.

**Trade-off.** The Parquet file is read-only here. If the data changed all the time we would want a real database.

**Safety.** All SQL that uses values from a request goes through bound parameters (`?` placeholders) in `reviewpulse/api/store.py`. Sort columns come from a fixed list, never from the request. An earlier version built SQL text with f-strings, which allowed SQL injection; `tests/test_security.py` now guards against that.

---

## 2. Getting the data: HTTP range requests, not "the first lines"

The source files on Hugging Face are 5 to 30 GB each, so we cannot download them. The first version of this project read the first few thousand lines of each file. That turned out to be a bad sample: the files are stored in runs of reviews by the same reviewer, so the first 3,000 lines came from only about 160 to 830 people per category.

Now the downloader asks the server for about 500 small pieces spread evenly over each file (HTTP `Range` requests), keeps complete lines only, removes duplicates, and fixes the random choice with a seed so the same command fetches the same reviews. The result has several times more distinct reviewers and covers many years.

**Limits.** It is still a sample, not a random draw from all of Amazon. Scores and charts describe this sample.

---

## 3. Splitting the data: by reviewer

A normal random split puts some of a person's reviews in the training set and some in the test set. People write consistently, so the model is then partly tested on habits it has already seen, and the score comes out too high. We measured this: with the first sample, 98% of test reviews came from a reviewer who also appeared in training.

We therefore use scikit-learn's `StratifiedGroupKFold`, grouped by `user_id`. Every reviewer's reviews end up in exactly one of train, validation or test, and the share of negative, neutral and positive reviews stays similar in each. Reviews with identical text are also removed before splitting.

---

## 4. Sentiment models: TF-IDF + linear models, and DistilBERT

**What we use.**
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVC
3. DistilBERT (`distilbert-base-uncased`), fine-tuned

**How the labels work.** Sentiment is taken from the star rating (1-2 negative, 3 neutral, 4-5 positive). That is a proxy, not a human judgement, and the dashboard says so wherever it matters.

**Why TF-IDF models.** TF-IDF turns each review into numbers by weighting words that are frequent in that review but rare overall. A linear model on top is fast to train, runs in milliseconds on a CPU, and its weights can be inspected. They make a strong baseline for text, and a deep model should be compared against one.

**Why DistilBERT.** DistilBERT is a smaller, faster version of BERT. It reads whole sentences and can use word order and context, which TF-IDF ignores. It needs far more computing power. On this project's CPU-only machine it can only be trained on a subset of the training data, and the saved result states exactly how many reviews and epochs were used. If it has not been trained, the dashboard says so and shows no scores.

**How the evaluation is kept honest.** The vectorizer learns its vocabulary from the training split only. The regularisation strength `C` is chosen on the validation split. The test split is scored once, and only those numbers are shown. Because the classes are very unbalanced (most reviews are positive), accuracy alone is misleading, so macro F1 (the plain average of the three per-class F1 scores) is the main number to compare.

**Text that gives the answer away.** Two leaks were found and fixed. First, splitting by review instead of by reviewer (section 3). Second, Amazon's automatic titles such as "Five Stars": about one review in ten has one, and it equals the rating almost every time, so the model text used to contain the label. `reviewpulse/pipeline/cleaner.py` now leaves those titles out of the model text. A test with the same split and the same regularisation showed what the leak was worth: about 2 to 4 points of accuracy on the reviews that had such a title, and under half a point overall.

**Alternatives.** Lexicon tools such as VADER need no training but cannot learn category-specific meaning ("cheap" can be praise or criticism). Larger transformers such as RoBERTa-large need a GPU.

---

## 5. Topics: NMF on TF-IDF

**What it does.** Non-negative matrix factorization (NMF) splits the review-by-word TF-IDF table into two smaller tables: which words define each topic, and how much of each topic a review contains. Because nothing is negative, a topic reads like a list of words that tend to appear together.

**Why NMF.** It is fast, gives the same result every time with a fixed seed, and works well on short texts. LDA is the usual alternative but is more sensitive to settings and is noisier on short reviews. BERTopic (embeddings plus clustering) gives nicer topics but needs more memory and time. Clustering the search embeddings with k-means is another option; its clusters are harder to explain in words.

**What we do with it.** Each review is tagged with its strongest topic, so the dashboard can filter reviews by topic and show topic counts for any category, date range or sentiment. "Top complaints" are simply the topics of negative (1-2 star) reviews.

**How the settings were chosen.** The first version used 14 topics and one topic held about a third of all reviews. The words of that topic were generic ("using, small, light, long, worked") and its reviews were about unrelated products, so it was a catch-all. Five setups were compared (`python -m reviewpulse.nlp.topic_experiments`), each with three random seeds, using four measures: the size of the biggest topic, word coherence (NPMI, how often a topic's top words appear in the same review), how many different words the topics use, and how many reviews are weakly assigned. Findings:
- More topics always shrink the biggest topic, but also make assignments weaker. 30 topics gave an 11% biggest topic and half of all reviews weakly assigned.
- Adding more everyday filler words to the stop list made things worse: the biggest topic grew to over 40%.
- **20 topics, praise-only words removed ("excellent", "thank", "recommend"), and only words in at least 20 reviews** kept coherence at the level of the first version while cutting the biggest topic from about a third to about a quarter. This is what is used.

The numbers are in `reports/topic_report.md` (generated by `python -m reviewpulse.nlp.topic_report`). One caveat: the saved model uses seed 42, and with seeds 0 to 2 the same settings gave a biggest topic of 23% to 26% (seed 42 gave about 20%), so a single value is approximate.

**Names.** A topic is a list of words, so a name is an interpretation. Each name was chosen by reading real reviews of that topic, and it is only used if its key words are among the topic's top words (`TOPIC_NAMES` in `reviewpulse/nlp/topic_model.py`). If someone retrains with other settings, topics that no longer match keep an automatic label made of their top words, and the API reports which kind each name is (`named_by`).

**Limits.** The largest topic is still a catch-all and is named as one ("General use and results (mixed products)"). A topic can mix two ideas: "year" in "my 5 year old" and in "broke after one year" land together. A review gets one topic although it may mention several. Reviews with none of the model's words get no tag.

---

## 6. Semantic search: sentence embeddings and FAISS

**What it does.** A sentence-transformer model (`all-MiniLM-L6-v2`) turns every review into a vector of 384 numbers so that reviews with similar meaning have vectors that point in similar directions. FAISS finds the closest vectors to a question. We use an exact index (`IndexFlatIP` on normalized vectors, which equals cosine similarity), so nothing is approximated.

**Why.** Keyword search misses synonyms ("durable" versus "sturdy"). Embeddings find them. For tens of thousands of vectors an exact index is fast enough, so a vector database would only add things to run.

**Coverage is explicit.** By default every review with text is indexed. If the machine lacked memory, or someone chose a cap, the reason and the numbers are saved in `index_info.json` and shown in the API and dashboard. Each vector is stored together with its review ID, and `python -m reviewpulse.nlp.semantic_search --verify` checks that a sample of vectors really belongs to the reviews they claim to. Results below a minimum similarity are not shown as evidence. The minimum is 0.40, chosen by looking at real scores instead of guessing: for questions the reviews can answer, the best match scored 0.53 to 0.78, while unrelated questions (the capital of Australia, a football result, a poem) never scored above 0.38. At the earlier value of 0.25 those unrelated questions still produced four look-alike reviews such as "Was a Christmas gift", which were then displayed as "evidence".

---

## 7. The assistant: facts from SQL, evidence from search, text from Gemini

The assistant answers from two sources only. Numbers (counts, averages, percentages) come from SQL, never from the language model. Review text comes from the most similar reviews found by the search above. If a Gemini API key is configured on the server, both are sent to Gemini and it writes a short answer that cites review IDs. The prompt tells the model to treat review text as data and not to follow instructions inside it.

If there is no key, or the call fails (bad key, rate limit, timeout, network problem, blocked answer), the API says exactly why. `llm_used` is `false`, `mode` is `retrieval_only`, and the facts and evidence are still returned. A retrieval-only response is never shown as if the model wrote it.

Gemini is called with plain HTTPS (`httpx`) instead of Google's Python SDK. The older SDK is deprecated, and a direct call keeps the timeout, retry and error handling visible and testable without a network. The key is read from the server environment (a git-ignored `.env` file) and is sent only in a request header. It never appears in a URL, a log line, an API response or a test fixture.

**What a real run taught us.** Calling the real service found problems that mock tests could not:
- *The default model had been retired.* `gemini-2.5-flash` answered "no longer available to new users". Model names change, so `GEMINI_MODEL` is a setting, and `.env.example` explains how to list the models a key can use. The default is `gemini-3.5-flash-lite`.
- *The free tier has a daily request limit per model.* About 25 calls in one day of testing used up the limit of the larger model, and the API answered with a rate-limit error. The lite model has its own limit and is enough for this job: it only has to summarise four reviews against exact numbers. The app reported the rate limit honestly and still returned the numbers and reviews, which is the behaviour the error handling was built for.
- *Answers were cut off.* Newer models "think" before answering and that thinking counts against the output limit (630 to 910 tokens in one probe), so a limit of 800 sometimes ended an answer mid-sentence. The limit is now 4096, and if a reply still stops at the limit the response says so.

**Checking that an answer is grounded.** A language model can still invent a review ID or a statistic. The dashboard marks any cited ID that is not among the shown evidence, and the opt-in live test checks that every cited ID was retrieved and every number is either a database number or appears in a retrieved review. This cannot prove the wording is faithful; it catches invented citations and invented numbers.

---

## 8. Backend: FastAPI

FastAPI gives typed request and response models (Pydantic), automatic validation (a bad date or an unknown category becomes a clear 422 error, not a crash), and an OpenAPI description at `/docs`. The same app serves the built React dashboard, so a deployment is one process.

Request bodies refuse fields they do not know (`extra="forbid"`). This closed a real bug: `POST /predict` used to ignore an unknown field, so `{"text": "...", "model": "linear_svc"}` quietly ran the default model while the caller believed another had run. Now the model name (`model_type`) is required, an unknown name is a 422 error that lists the allowed names, and the response repeats the model that ran (`model_type`) so a client can check it.

Alternatives such as Flask need extra packages for validation and documentation; Django REST Framework brings an ORM and sessions this project does not use.

---

## 9. Frontend: React, TypeScript, Vite

React with TypeScript gives reusable components and compile-time checking of the API types, which catches a renamed field before it reaches a user. Vite builds quickly. Tailwind CSS handles styling, Recharts draws the charts, TanStack Query handles loading, caching and errors for API calls, and React Router provides the pages and links.

The production build is plain static files in `frontend/dist`, which FastAPI serves. In development the Vite server proxies `/api` to the backend, so the browser always talks to one address and no CORS setup is needed.

---

## 10. Monitoring: PSI and the KS test

The drift check asks whether newer reviews look different from older ones. Reviews are sorted by date: the oldest 70% are the reference and the newest 30% the current data. For text length, rating and helpful votes we compute the Population Stability Index (PSI) and, for the first two, the Kolmogorov-Smirnov (KS) test. A feature is flagged if PSI is at least 0.2, or if the KS test is significant and its statistic is at least 0.1. The size condition matters: with tens of thousands of reviews the p-value alone flags differences too small to matter.

---

## 11. Testing

- **Backend (`pytest`)**: unit tests, API tests against a small synthetic data set with known answers, security tests with SQL-injection inputs, strict request validation, retrieval and model tests, and "real data" tests that reconcile counts across the manifest, Parquet file, topic tags, index and API. For topics, every saved tag is recomputed from the review's own text, category-specific counts are compared with pandas, and topic names are checked against the categories their reviews come from. Gemini is tested with a mock transport covering every failure mode, plus opt-in tests that call the real service. A fresh-clone test starts the API on empty folders and checks that every endpoint names the command that fixes what is missing.
- **Frontend (`vitest`)**: component and API-client tests with a mocked `fetch`.
- **End to end (`npm run e2e`)**: a real Chrome drives the built app served by FastAPI at four screen widths and checks pages, filters, prediction, the assistant, failures, console errors and horizontal overflow.
- **Contract check (`npm run check:contract`)**: compares the frontend's types with the running API's OpenAPI schema.

Passing frontend tests says nothing about backend security or model quality, so the suites are run and reported separately.

---

## 12. Deployment

One command starts everything: `uvicorn reviewpulse.api.app:app`. It serves the API under `/api/v1`, the documentation at `/docs`, and the React build at `/`, with a fallback to `index.html` so links such as `/reviews` work after a refresh. Where data, trained models and reports live can be changed with `REVIEWPULSE_DATA_DIR`, `REVIEWPULSE_ARTIFACTS_DIR` and `REVIEWPULSE_REPORTS_DIR`; by default they are inside the repository, so nothing depends on one machine's paths. No Docker setup is included; the app needs Python, the saved artifacts and the built frontend, and Docker could be added later around that single command.
