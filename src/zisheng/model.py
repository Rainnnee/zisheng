"""A compact U-Net that predicts the noise in a character image."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Turn an integer diffusion step into the smooth embedding used by a U-Net."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.dimension % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    """Convolve image features, then inject the timestep/class information."""

    def __init__(self, input_channels: int, output_channels: int, condition_size: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.condition = nn.Linear(condition_size, output_channels)
        self.norm2 = nn.GroupNorm(8, output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(features)))
        # Each image location receives the same learned summary of time and class.
        hidden = hidden + self.condition(F.silu(condition))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return (hidden + self.skip(features)) / math.sqrt(2.0)


class ConditionalDenoiser(nn.Module):
    """Predict Gaussian noise from (noisy image, diffusion step, character ID).

    The down/up path gives the model both stroke-level detail and a wider view of
    the glyph. Skip connections preserve fine line placement as spatial size falls.
    """

    def __init__(
        self,
        num_classes: int = 12,
        *,
        time_size: int = 64,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.time_projection = nn.Sequential(
            SinusoidalTimeEmbedding(time_size),
            nn.Linear(time_size, time_size * 2),
            nn.SiLU(),
            nn.Linear(time_size * 2, time_size),
        )
        self.class_embedding = nn.Embedding(num_classes, time_size)

        self.stem = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        self.full_resolution = ResidualBlock(base_channels, base_channels, time_size)
        self.downsample1 = nn.Conv2d(
            base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.half_resolution = ResidualBlock(
            base_channels * 2, base_channels * 2, time_size
        )
        self.downsample2 = nn.Conv2d(
            base_channels * 2, base_channels * 3, kernel_size=4, stride=2, padding=1
        )
        self.middle1 = ResidualBlock(base_channels * 3, base_channels * 3, time_size)
        self.middle2 = ResidualBlock(base_channels * 3, base_channels * 3, time_size)

        self.upsample1 = nn.ConvTranspose2d(
            base_channels * 3, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.merge_half = ResidualBlock(
            base_channels * 4, base_channels * 2, time_size
        )
        self.upsample2 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1
        )
        self.merge_full = ResidualBlock(base_channels * 2, base_channels, time_size)
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        character_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate the specific noise tensor used to create each noisy image."""

        condition = self.time_projection(timesteps) + self.class_embedding(character_ids)
        full = self.full_resolution(self.stem(noisy_images), condition)
        half = self.half_resolution(self.downsample1(full), condition)
        low = self.downsample2(half)
        low = self.middle2(self.middle1(low, condition), condition)
        half_out = self.merge_half(torch.cat((self.upsample1(low), half), dim=1), condition)
        full_out = self.merge_full(
            torch.cat((self.upsample2(half_out), full), dim=1), condition
        )
        return self.output(F.silu(self.output_norm(full_out)))
