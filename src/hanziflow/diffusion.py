"""Forward and reverse diffusion interfaces."""

from typing import Any


def q_sample(clean_images: Any, timesteps: Any, noise: Any) -> Any:
    """Return images corrupted at the supplied timesteps using supplied noise."""
    raise NotImplementedError("The noise schedule and forward process are not implemented yet.")


def reverse_step(
    noisy_images: Any,
    timesteps: Any,
    character_ids: Any,
    *,
    generator: Any = None,
) -> Any:
    """Take one reverse-process step for a batch of requested characters."""
    raise NotImplementedError("The reverse sampler is not implemented yet.")
