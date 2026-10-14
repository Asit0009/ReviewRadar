# ReviewRadar Data Dictionary

What every data file contains. Dataset counts and summary figures are documented in `reports/data_summary.md`.

---

## 1. Source data

**Dataset:** Amazon Reviews 2023, McAuley Lab, on Hugging Face (`McAuley-Lab/Amazon-Reviews-2023`).

**Categories used** (names are the dataset's own file names):
Electronics, Home_and_Kitchen, Beauty_and_Personal_Care, Sports_and_Outdoors, Toys_and_Games, Pet_Supplies, Automotive, Cell_Phones_and_Accessories, Health_and_Household, Office_Products.

**How a sample is taken.** Each source file is 5 to 30 GB, so the downloader fetches about 500 small chunks spread evenly over the file (HTTP range requests), keeps whole lines, and removes duplicates. The chunk positions are fixed by a seed, so the same command downloads the same reviews. How the sample was taken is stored per category in `data/manifest.json` (`sampling`).

### Raw review record (`data/raw/<Category>_reviews.jsonl`)
| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `rating` | float | Star rating, 1.0 to 5.0 | `5.0` |
| `title` | string | Review headline | `"Works great"` |
| `text` | string | Review body. May contain HTML such as `<br />` | `"Easy to set up.<br />Very quiet."` |
| `asin` | string | Product variant ID | `"B08XYZ1234"` |
| `parent_asin` | string | Product family ID | `"B08XYZ1234"` |
| `user_id` | string | Anonymized reviewer ID | `"AG6SWXYZ..."` |
| `timestamp` | int | Review time in milliseconds since 1970 | `1685000000000` |
| `helpful_vote` | int | Helpful votes from other customers | `3` |
| `verified_purchase` | bool | Bought through Amazon | `true` |

Product metadata (`<Category>_meta.jsonl`) is optional and off by default. With spread sampling almost no review would match the first metadata lines, so `product_title` is empty for most reviews. The dashboard shows a product line only when it exists.

---

## 2. Processed dataset: `data/processed/reviews_all.parquet`

Written by `python -m reviewpulse.pipeline.run_pipeline`. Snappy-compressed Parquet with a fixed schema (`REVIEW_PYARROW_SCHEMA`).

| Column | Type | Description |
| :--- | :--- | :--- |
| `review_id` | string | Deterministic SHA-1 hash of `(user_id, asin, timestamp)`, first 16 hex characters. Unique per review. |
| `category` | string | One of the ten category names above |
| `rating` | float32 | Star rating, 1.0 to 5.0 |
| `sentiment` | string | `negative` (1-2 stars), `neutral` (3), `positive` (4-5). **Derived from the rating**, not from the text |
| `sentiment_encoded` | int8 | 0, 1 or 2 for negative, neutral, positive |
| `title` | string | Cleaned title, exactly as Amazon showed it (including automatic titles such as "Five Stars") |
| `text` | string | Cleaned body |
| `combined_text` | string | `"<title>. <text>"`, except that an automatic rating title ("Five Stars", "One Star", "4 stars") is left out, because it would give the rating away. This is the only text the models, the topic model and the search index see |
| `text_length` | int32 | Characters in `text` |
| `word_count` | int32 | Words in `text` |
| `user_id` | string | Anonymized reviewer ID |
| `asin`, `parent_asin` | string | Product IDs |
| `timestamp` | int64 | Original time in milliseconds |
| `timestamp_utc` | string | ISO-8601 UTC time |
| `review_date` | string | `YYYY-MM-DD` (the field used for date filters) |
| `helpful_vote` | int32 | Helpful votes |
| `verified_purchase` | bool | Verified purchase |
| `product_title` | string | From metadata when available, otherwise empty |
| `product_price` | string | From metadata when available, otherwise empty |

### Cleaning and validation rules
1. **Validation** (a failing record is dropped and counted by reason in the manifest): required fields present; rating a number from 1.0 to 5.0; text at least 2 non-blank characters; timestamp a positive integer; valid JSON.
2. **Duplicates:** records with the same `(user_id, asin, timestamp)` are kept once.
3. **Text cleaning:** HTML entities are unescaped (`&amp;` becomes `&`); HTML tags such as `<br />` are removed (a line break becomes a space); Amazon's `[[VIDEOID:...]]` and `[[ASIN:...]]` markers are removed; runs of whitespace become one space. Text like `5 < 6 and 7 > 3` is kept because only things shaped like tags (`<` followed by a letter or `/`) are removed.
4. **Sentiment label:** `rating <= 2` negative, `rating == 3` neutral, otherwise positive.
5. **Automatic rating titles:** when the title is only Amazon's star label (`one`..`five` or `1`..`5`, followed by `star` or `stars`), it is not put into `combined_text`. About one review in ten has such a title and it matches the rating 99.7% of the time. A handful of reviews whose body is only two characters ("OK") therefore have a two-character `combined_text`; they stay in the statistics but are too short for semantic search.

### Modeling dataset (derived when training)
Identical review texts are removed, then the reviews are split **by reviewer** into train, validation and test (`StratifiedGroupKFold` on `user_id`), so no reviewer appears in two splits. Exact split sizes are in `artifacts/models/metrics.json` under `split_info`.

---

## 3. Model artifacts (`artifacts/models/`)

| File | Description |
| :--- | :--- |
| `tfidf_vectorizer.joblib` | TF-IDF vectorizer fitted on the training split only (1-2 word n-grams, up to 25,000 features) |
| `logistic_regression.joblib`, `linear_svc.joblib` | The two classical classifiers |
| `distilbert_sentiment/` | Fine-tuned DistilBERT and its tokenizer (only if training was run) |
| `metrics.json` | **The only source of model scores shown anywhere.** Written by the training scripts after evaluation |

### `metrics.json` structure
```
split_info          sizes, class counts, reviewers per split, split strategy, test_ids_md5
models.<name>.validation / .test
                    accuracy, macro_f1, weighted_f1, per-class precision/recall/f1,
                    macro_precision, macro_recall, confusion_matrix (rows = true class)
models.<name>.hyperparameters   chosen C and the search that chose it (classical models)
models.distilbert.training      samples used, epochs, batch size, device, loss per epoch, seconds
models.distilbert.evaluated_on  how many validation and test reviews were scored
per_category.<name>.<category>  test_samples, class_support, reliable, accuracy, macro_f1, per-class f1
trained_at, updated_at          timestamps
```
`reliable` is false when a sentiment class has fewer than 20 test reviews in that category, because the score is then too noisy to compare.

---

## 4. Topic artifacts (`artifacts/nmf_topics/`)

| File | Description |
| :--- | :--- |
| `topic_labels.json` | Topics: `topic_id`, `label` (the reviewed name when one matches, otherwise the top-word label), `auto_label` (always the top-word label), `named_by` (`reviewed` or `top words`), `top_words`, `sample_reviews` (two different real review excerpts, preferring longer reviews), `prevalence_count` and `prevalence_pct` (share of the **tagged** reviews, the same as the API); also `tagged_reviews` and the training parameters |
| `review_topics.parquet` | One row per tagged review: `review_id`, `topic_id` (its strongest topic), `topic_score` (that topic's share of the review's topic weight, 0 to 1). Reviews containing none of the model's words have no row |
| `nmf_vectorizer.joblib`, `nmf_model.joblib` | Let new text be assigned to a topic (`assign_topics`). Large binary files, so they are git-ignored |

Two generated reports belong to the topics: `reports/topic_experiments.json` (the comparison of five NMF setups, three seeds each) and `reports/topic_report.md` (topic sizes, names, example reviews and findings).

---

## 5. Search index (`artifacts/vector_store/`)

| File | Description |
| :--- | :--- |
| `faiss_index.bin` | FAISS `IndexFlatIP` of L2-normalized review embeddings. Vector number `i` is row `i` of the metadata file |
| `vector_metadata.parquet` | `vector_id`, `review_id`, `category`, `rating`, `sentiment`, `title`, `text`, `combined_text`, `product_title`, `timestamp_utc`, `helpful_vote`, `verified_purchase` |
| `index_info.json` | `eligible_reviews`, `indexed_reviews`, `coverage_pct`, `limited_by` (null when everything is indexed), `embedding_model`, `built_at`, `eligible_ids_md5` (detects an index built from older data), counts per category |

---

## 6. Manifest (`data/manifest.json`)

Per category: source URL, raw file path and SHA-256, number of downloaded reviews, distinct reviewers, sampling settings (`strategy`, `chunks`, `seed`, source file size), acquisition time. `pipeline_processing.category_reports` holds, per category, how many records were read and how many were rejected for each reason. `hardware_snapshot` records the machine's RAM, disk and CPU at download time.
