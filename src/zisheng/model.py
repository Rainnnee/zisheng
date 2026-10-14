"""Conditional denoiser interface."""

from typing import Any


class ConditionalDenoiser:
    """Placeholder for a denoiser conditioned on time and character ID."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("The model architecture has not been selected.")
