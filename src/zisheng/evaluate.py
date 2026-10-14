"""Evaluation and figure-generation interfaces."""

from pathlib import Path
from typing import Any


def evaluate(samples: Path, protocol: Path, evaluator: Any) -> Any:
    """Compute and save per-sample and aggregate metrics under a fixed protocol."""
    raise NotImplementedError("Evaluation is not implemented yet.")


def make_figures(results: Path, output_directory: Path) -> None:
    """Create deterministic figures from saved result tables and metadata."""
    raise NotImplementedError("Figure generation is not implemented yet.")
