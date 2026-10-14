"""Training-loop and checkpoint interfaces."""

from typing import Any


def training_step(batch: Any, optimizer: Any, model: Any, diffusion: Any) -> Any:
    """Run one documented optimization step and return its measurements."""
    raise NotImplementedError("Training is not implemented yet.")
