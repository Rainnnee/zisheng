"""Reproducible conditional sampling interface."""

from pathlib import Path
from typing import Any


def sample(
    checkpoint: Path,
    character_ids: Any,
    *,
    seed: int,
    settings: Path,
) -> Any:
    """Generate images and, optionally, a saved reverse-sampling trajectory."""
    raise NotImplementedError("Sampling is not implemented yet.")
