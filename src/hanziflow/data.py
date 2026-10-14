"""Data manifests, character labels, and image preparation interfaces."""

from pathlib import Path
from typing import Any


def prepare_data(
    source: Path,
    character_manifest: Path,
    split_specification: Path,
) -> Any:
    """Prepare verified images and a manifest from documented source files."""
    raise NotImplementedError("Data parsing and validation are not implemented yet.")
