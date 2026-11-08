"""The finite Gaussian forward process and its learned reverse sampler."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _at_timestep(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Gather one schedule value per image and add broadcast dimensions."""

    selected = values.gather(0, timesteps)
    return selected.reshape(selected.shape[0], *([1] * (reference.ndim - 1)))


class GaussianDiffusion(nn.Module):
    """A DDPM with a linear beta schedule and the epsilon-prediction objective.

    At each training example, the closed-form marginal creates x_t directly from
    x_0. The denoiser learns the added noise; the reverse sampler uses that
    estimate to move one step at a time from Gaussian noise back toward a glyph.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("A diffusion schedule needs at least two timesteps.")
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_previous", alpha_bars_previous)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def q_sample(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Draw x_t from q(x_t|x_0) without simulating all earlier steps."""

        alpha_bar = _at_timestep(self.alpha_bars, timesteps, clean_images)
        return alpha_bar.sqrt() * clean_images + (1.0 - alpha_bar).sqrt() * noise

    def training_loss(
        self,
        denoiser: nn.Module,
        clean_images: torch.Tensor,
        character_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Train the network to recover the Gaussian noise added to x_0."""

        batch_size = clean_images.shape[0]
        timesteps = torch.randint(
            0, self.timesteps, (batch_size,), device=clean_images.device
        )
        actual_noise = torch.randn_like(clean_images)
        noisy_images = self.q_sample(clean_images, timesteps, actual_noise)
        predicted_noise = denoiser(noisy_images, timesteps, character_ids)
        return F.mse_loss(predicted_noise, actual_noise)

    @torch.inference_mode()
    def sample(
        self,
        denoiser: nn.Module,
        character_ids: torch.Tensor,
        *,
        image_size: int = 32,
        seed: int = 7,
        snapshot_steps: Sequence[int] | None = None,
        step_callback: Callable[[int, torch.Tensor], None] | None = None,
        callback_interval: int = 10,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Generate requested classes and optionally keep or stream reverse steps.

        The returned images are still scaled to [-1, 1]. A snapshot keyed by N
        represents the state after N forward-noise steps, so key 0 is the final
        generated glyph and key ``timesteps`` is the starting Gaussian noise.
        When ``step_callback`` is set, it receives the initial noise and then
        actual model states at the requested interval; no frames are interpolated.
        """

        if character_ids.ndim != 1:
            raise ValueError("character_ids must be a one-dimensional tensor")
        if callback_interval < 1:
            raise ValueError("callback_interval must be at least 1")
        device = character_ids.device
        batch_size = character_ids.shape[0]
        torch.manual_seed(seed)
        if device.type == "mps" and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)

        current = torch.randn(batch_size, 1, image_size, image_size, device=device)
        requested_steps = set(snapshot_steps or ())
        requested_steps.update({0, self.timesteps})
        snapshots: dict[int, torch.Tensor] = {}
        if self.timesteps in requested_steps:
            snapshots[self.timesteps] = current.detach().cpu()
        if step_callback is not None:
            # Emit x_T before the model starts so the browser can show the true
            # starting noise while the reverse chain is still running.
            step_callback(self.timesteps, current.detach().cpu())

        for index in reversed(range(self.timesteps)):
            step_batch = torch.full((batch_size,), index, device=device, dtype=torch.long)
            predicted_noise = denoiser(current, step_batch, character_ids)
            beta = self.betas[index]
            alpha = self.alphas[index]
            alpha_bar = self.alpha_bars[index]

            # This is the DDPM posterior mean after substituting the predicted
            # noise for epsilon in the closed-form expression.
            mean = (current - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
            if index > 0:
                noise = torch.randn_like(current)
                current = mean + self.posterior_variance[index].sqrt() * noise
            else:
                current = mean

            remaining_forward_steps = index
            if remaining_forward_steps in requested_steps:
                snapshots[remaining_forward_steps] = current.detach().cpu()
            if step_callback is not None and (
                remaining_forward_steps % callback_interval == 0
                or remaining_forward_steps == 0
            ):
                # The last visible state should match the clipped final output.
                visible_state = current.clamp(-1.0, 1.0) if remaining_forward_steps == 0 else current
                step_callback(remaining_forward_steps, visible_state.detach().cpu())

        return current.clamp(-1.0, 1.0), snapshots
