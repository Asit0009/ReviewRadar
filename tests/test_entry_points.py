"""The commands in the README must be safe to explore: `--help` prints help and does not run the job.

Before this was checked, `python -m reviewpulse.modeling.train_classical --help` silently retrained the models,
because the script ignored its arguments. Each command is run here against empty, redirected folders, so even a
regression cannot touch real data, and the test fails if anything is written.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MODULES = [
    "reviewpulse.acquisition.downloader",
    "reviewpulse.pipeline.run_pipeline",
    "reviewpulse.eda.run_eda",
    "reviewpulse.eda.data_summary",
    "reviewpulse.modeling.train_classical",
    "reviewpulse.modeling.train_distilbert",
    "reviewpulse.modeling.report",
    "reviewpulse.nlp.topic_model",
    "reviewpulse.nlp.topic_experiments",
    "reviewpulse.nlp.topic_report",
    "reviewpulse.nlp.semantic_search",
]


@pytest.mark.parametrize("module", MODULES)
def test_help_prints_usage_and_runs_nothing(module, tmp_path):
    env = {
        **os.environ,
        "REVIEWPULSE_DATA_DIR": str(tmp_path / "data"),
        "REVIEWPULSE_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
        "REVIEWPULSE_REPORTS_DIR": str(tmp_path / "reports"),
    }
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, env=env, cwd=str(REPO), timeout=180)
    assert result.returncode == 0, result.stderr[-500:]
    assert "usage:" in result.stdout.lower(), f"{module} did not print help"
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert not written, f"{module} --help wrote files: {written[:3]}"
