"""Configuration management for ReviewPulse.

Paths and limits live here. Everything that can change between machines (API keys, CORS origins,
data limits) is read from environment variables, so no secret ever needs to be written into code.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

def _env_path(name: str, default: Path) -> Path:
    """A folder from an environment variable (relative paths are resolved against the current directory), else the default."""
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else default


# Directories. Everything lives under the repository by default; set these to keep the data, the trained models
# and the reports somewhere else (another disk, a shared folder).
DATA_DIR = _env_path("REVIEWPULSE_DATA_DIR", BASE_DIR / "data")
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "manifest.json"

ARTIFACTS_DIR = _env_path("REVIEWPULSE_ARTIFACTS_DIR", BASE_DIR / "artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
NLP_DIR = ARTIFACTS_DIR / "nmf_topics"
VECTOR_STORE_DIR = ARTIFACTS_DIR / "vector_store"

# The production dashboard is the React build. `npm run build` in frontend/ writes it here.
FRONTEND_DIST_DIR = Path(os.getenv("FRONTEND_DIST_DIR", str(BASE_DIR / "frontend" / "dist")))

REPORTS_DIR = _env_path("REVIEWPULSE_REPORTS_DIR", BASE_DIR / "reports")
EDA_REPORTS_DIR = REPORTS_DIR / "eda"
EDA_FIGURES_DIR = EDA_REPORTS_DIR / "figures"
MODEL_REPORTS_DIR = REPORTS_DIR / "model_evaluation"
MODEL_FIGURES_DIR = MODEL_REPORTS_DIR / "figures"

# The ten product categories of the project, using the dataset's own file names
# (McAuley-Lab/Amazon-Reviews-2023). Override with REVIEWPULSE_CATEGORIES="A,B,C" for experiments.
_DEFAULT_CATEGORIES = [
    "Electronics",
    "Home_and_Kitchen",
    "Beauty_and_Personal_Care",
    "Sports_and_Outdoors",
    "Toys_and_Games",
    "Pet_Supplies",
    "Automotive",
    "Cell_Phones_and_Accessories",
    "Health_and_Household",
    "Office_Products",
]


def _categories_from_env() -> list:
    raw = os.getenv("REVIEWPULSE_CATEGORIES", "").strip()
    if not raw:
        return list(_DEFAULT_CATEGORIES)
    return [c.strip() for c in raw.split(",") if c.strip()]


OFFICIAL_CATEGORIES = _categories_from_env()

# Hugging Face Official Endpoints
HF_REVIEWS_BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"
HF_META_BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"

# Working limits. The source files are many GB each, so only DEFAULT_LIMIT_PER_CATEGORY reviews per category are kept.
DEFAULT_LIMIT_PER_CATEGORY = int(os.getenv("DATA_LIMIT_PER_CATEGORY", "3000"))
DEFAULT_META_LIMIT_PER_CATEGORY = int(os.getenv("META_LIMIT_PER_CATEGORY", "0"))  # 0 = skip product metadata

# Models & Embeddings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Server configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Everything the API needs at runtime. Tests build their own Settings that point at temporary data."""

    processed_dir: Path = DATA_PROCESSED_DIR
    models_dir: Path = MODELS_DIR
    nlp_dir: Path = NLP_DIR
    vector_store_dir: Path = VECTOR_STORE_DIR
    frontend_dist_dir: Path = FRONTEND_DIST_DIR
    categories: Tuple[str, ...] = field(default_factory=lambda: tuple(OFFICIAL_CATEGORIES))
    # Browsers only need CORS when the page and API are on different origins (e.g. Vite dev server on :5173).
    cors_origins: Tuple[str, ...] = field(
        default_factory=lambda: _split_csv(os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"))
    )
    # repr=False keeps the key out of logs, tracebacks and test output if a Settings object is ever printed.
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""), repr=False)
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    gemini_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("GEMINI_TIMEOUT_SECONDS", "25")))
    # Reviews less similar to the question than this are not shown as evidence.
    retrieval_min_score: float = field(default_factory=lambda: float(os.getenv("RETRIEVAL_MIN_SCORE", "0.40")))

    @property
    def reviews_parquet(self) -> Path:
        return self.processed_dir / "reviews_all.parquet"

    @property
    def review_topics_parquet(self) -> Path:
        return self.nlp_dir / "review_topics.parquet"

    @property
    def topic_labels_json(self) -> Path:
        return self.nlp_dir / "topic_labels.json"

    @property
    def metrics_json(self) -> Path:
        return self.models_dir / "metrics.json"

    @property
    def gemini_configured(self) -> bool:
        key = self.gemini_api_key.strip()
        return bool(key) and key != "your_gemini_api_key_here"


def ensure_directories():
    """Ensure all required project directories exist."""
    for directory in [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        NLP_DIR,
        VECTOR_STORE_DIR,
        EDA_FIGURES_DIR,
        MODEL_FIGURES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def default_settings(frontend_dist_dir: Optional[Path] = None) -> Settings:
    return Settings(frontend_dist_dir=frontend_dist_dir or FRONTEND_DIST_DIR)
