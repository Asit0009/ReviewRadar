"""The command that creates each saved artifact.

Error messages and GET /health use these, so a missing file always says how to build it. The same steps, with
explanations and run times, are in the README ("Build the data and models from scratch").
"""

from typing import Dict, Mapping

# In build order: each step needs the ones above it.
SETUP_COMMANDS: Dict[str, str] = {
    "parquet_data": "python -m reviewpulse.acquisition.downloader && python -m reviewpulse.pipeline.run_pipeline",
    "sentiment_models": "python -m reviewpulse.modeling.train_classical",
    "nmf_topics": "python -m reviewpulse.nlp.topic_model",
    "vector_search_faiss": "python -m reviewpulse.nlp.semantic_search --verify",
    "distilbert_model": "python -m reviewpulse.modeling.train_distilbert",
}


def missing_steps(artifacts_ready: Mapping[str, bool]) -> Dict[str, str]:
    """For every artifact that is not ready, the command that builds it (empty when everything exists)."""
    return {name: command for name, command in SETUP_COMMANDS.items() if name in artifacts_ready and not artifacts_ready[name]}
