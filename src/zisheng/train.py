"""Training loop for the small font-rendered conditional diffusion pilot."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch

from .data import IMAGE_SIZE, PILOT_CHARACTERS, build_pilot_dataset, write_dataset_manifest
from .diffusion import GaussianDiffusion
from .model import ConditionalDenoiser


def choose_device(requested: str = "auto") -> torch.device:
    """Prefer Apple MPS or CUDA when requested and available; otherwise use CPU."""

    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("The requested MPS device is not available in this PyTorch install.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The requested CUDA device is not available.")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Smooth model weights; EMA samples tend to be steadier than the last step."""

    with torch.no_grad():
        for ema_parameter, parameter in zip(
            ema_model.parameters(), model.parameters(), strict=True
        ):
            ema_parameter.lerp_(parameter, 1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
            ema_buffer.copy_(buffer)


def training_step(
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
) -> float:
    """Do one optimizer update and return its scalar noise-prediction loss."""

    clean_images, character_ids = batch
    optimizer.zero_grad(set_to_none=True)
    loss = diffusion.training_loss(model, clean_images, character_ids)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def train_model(
    output_directory: Path,
    font_directory: Path,
    *,
    steps: int = 4000,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    timesteps: int = 1000,
    variants_per_font: int = 128,
    seed: int = 42,
    requested_device: str = "auto",
) -> Path:
    """Train the pilot model, then save a self-describing EMA checkpoint."""

    if steps < 1:
        raise ValueError("steps must be positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = choose_device(requested_device)
    _seed_everything(seed)
    print(f"Device: {device}", flush=True)
    print(
        f"Rendering the balanced {len(PILOT_CHARACTERS)}-character font pilot…",
        flush=True,
    )

    dataset = build_pilot_dataset(
        font_directory,
        variants_per_font=variants_per_font,
        seed=seed,
        image_size=IMAGE_SIZE,
    )
    write_dataset_manifest(
        dataset,
        output_directory / "dataset_manifest.json",
        seed=seed,
        variants_per_font=variants_per_font,
        image_size=IMAGE_SIZE,
    )

    # Keeping this small synthetic pilot in memory avoids disk and loader overhead.
    images = torch.from_numpy(dataset.images).to(device)
    labels = torch.from_numpy(dataset.labels).to(device)
    model = ConditionalDenoiser(num_classes=len(PILOT_CHARACTERS)).to(device)
    ema_model = deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    diffusion = GaussianDiffusion(timesteps=timesteps).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )

    log_interval = max(1, steps // 20)
    recent_losses: list[float] = []
    loss_history: list[dict[str, float | int]] = []
    print(
        f"Training on {len(labels):,} rendered images, "
        f"{len(PILOT_CHARACTERS)} classes, {steps:,} updates…",
        flush=True,
    )

    for step in range(1, steps + 1):
        # Sampling balanced dataset indices keeps each character equally likely.
        indices = torch.randint(len(labels), (batch_size,), device=device)
        loss = training_step(
            (images[indices], labels[indices]), optimizer, model, diffusion
        )
        _update_ema(ema_model, model, decay=0.995)
        recent_losses.append(loss)

        if step % log_interval == 0 or step == steps:
            mean_loss = float(np.mean(recent_losses))
            loss_history.append({"step": step, "mean_noise_mse": mean_loss})
            print(
                f"step {step:>5}/{steps}: noise-prediction MSE {mean_loss:.4f}",
                flush=True,
            )
            recent_losses.clear()

    checkpoint_path = output_directory / "zisheng-pilot.pt"
    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in ema_model.state_dict().items()
        },
        "config": {
            "num_classes": len(PILOT_CHARACTERS),
            "image_size": IMAGE_SIZE,
            "timesteps": timesteps,
            "beta_schedule": "linear",
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "prediction_target": "added Gaussian noise epsilon",
            "training_steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "variants_per_font": variants_per_font,
            "ema_decay": 0.995,
            "seed": seed,
            "device": str(device),
            "characters": [
                {
                    "glyph": item.glyph,
                    "class_id": item.class_id,
                    "stroke_count": item.stroke_count,
                    "gloss": item.gloss,
                }
                for item in PILOT_CHARACTERS
            ],
            "fonts": dataset.font_manifest,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    (output_directory / "training_loss.json").write_text(
        json.dumps(loss_history, indent=2) + "\n"
    )
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    return checkpoint_path
