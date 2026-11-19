"""Train the four paired runs for the glyph-entropy experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import time
import tomllib

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch

from zisheng.diffusion import GaussianDiffusion
from zisheng.model import ConditionalDenoiser
from zisheng.train import _update_ema, choose_device, training_step


def load_training_glyphs(data_dir: Path):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    archive = data_dir / "glyphs.npz"
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    fonts = manifest["fonts"]
    size = manifest["image_size"]
    classes = [character["class_id"] for character in manifest["characters"]]
    with np.load(archive, allow_pickle=False) as bundle:
        glyphs = bundle["glyphs"]
    expected_shape = (len(fonts), len(classes), size, size)
    if glyphs.shape != expected_shape or glyphs.dtype != np.uint8:
        raise ValueError(f"Expected uint8 glyphs with shape {expected_shape}.")
    if classes != list(range(24)) or size != 48:
        raise ValueError("This study requires 24 ordered class IDs and 48-pixel images.")
    train_ids = [index for index, font in enumerate(fonts) if font["split"] == "train"]
    split_counts = [
        sum(font["split"] == split for font in fonts) for split in ("train", "val", "test")
    ]
    if len(fonts) != 6 or split_counts != [3, 1, 2]:
        raise ValueError("Expected exactly three train, one validation, and two test fonts.")
    return glyphs[train_ids], [fonts[index]["font_id"] for index in train_ids], digest


def augment(glyphs: np.ndarray, *, variants: int, seed: int):
    """Create the same ordered bank of train-only examples for every run."""

    rng = np.random.default_rng(seed)
    rendered = []
    labels = []
    for font_glyphs in glyphs:
        for class_id, pixels in enumerate(font_glyphs):
            rendered.append(pixels)
            labels.append(class_id)
            image = Image.fromarray(pixels)
            for _ in range(variants):
                shift = tuple(map(int, rng.integers(-2, 3, size=2)))
                rotated = image.rotate(
                    float(rng.uniform(-5.0, 5.0)),
                    resample=Image.Resampling.BILINEAR,
                    translate=shift,
                    fillcolor=255,
                )
                rendered.append(np.asarray(rotated))
                labels.append(class_id)
    stacked = np.stack(rendered)
    images = torch.from_numpy(stacked).to(torch.float32).unsqueeze(1) / 127.5 - 1.0
    return images, torch.tensor(labels, dtype=torch.long), hashlib.sha256(stacked).hexdigest()


def train_one(images, labels, directory: Path, settings: dict):
    """Write a full EMA checkpoint after the configured number of updates."""

    width = settings["base_channels"]
    seed = settings["seed"]
    steps = settings["training_steps"]
    batch_size = settings["batch_size"]
    checkpoint = directory / f"w{width}-s{seed}.pt"
    if checkpoint.exists():
        existing = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if existing["config"] != settings:
            raise ValueError(f"{checkpoint} has different settings. Use a new output directory.")
        print(f"Reusing completed {checkpoint.name}", flush=True)
        return existing["training_log"]

    torch.manual_seed(seed)
    device = images.device
    model = ConditionalDenoiser(
        num_classes=settings["num_classes"], base_channels=width
    ).to(device)
    ema = deepcopy(model).eval()
    ema.requires_grad_(False)
    diffusion = GaussianDiffusion(
        timesteps=settings["timesteps"],
        beta_start=settings["beta_start"],
        beta_end=settings["beta_end"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=1e-4)
    sampler = torch.Generator().manual_seed(settings["index_seed"])
    clock = torch.Generator().manual_seed(settings["timestep_seed"])
    gaussian = torch.Generator().manual_seed(settings["noise_seed"])

    log = []
    running_loss = 0.0
    count = 0
    start = time.perf_counter()
    for update in range(1, steps + 1):
        indices = torch.randint(len(labels), (batch_size,), generator=sampler).to(device)
        times = torch.randint(diffusion.timesteps, (batch_size,), generator=clock).to(device)
        noise = torch.randn((batch_size, *images.shape[1:]), generator=gaussian).to(device)
        loss = training_step(
            (images[indices], labels[indices]),
            optimizer,
            model,
            diffusion,
            timesteps=times,
            noise=noise,
        )
        if not np.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss in {checkpoint.name} at update {update}.")
        _update_ema(ema, model, decay=settings["ema_decay"])
        running_loss += loss
        count += 1
        if update % max(1, steps // 8) == 0 or update == steps:
            elapsed = time.perf_counter() - start
            mean_loss = running_loss / count
            log.append({"update": update, "noise_mse": mean_loss, "elapsed_s": elapsed})
            print(
                f"{checkpoint.stem} {update:4d}/{steps}  noise MSE {mean_loss:.5f}  {elapsed:.1f}s",
                flush=True,
            )
            running_loss = 0.0
            count = 0

    temporary = checkpoint.with_suffix(".pt.part")
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in ema.state_dict().items()},
            "config": settings,
            "training_log": log,
        },
        temporary,
    )
    temporary.replace(checkpoint)
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/study"))
    parser.add_argument("--config", type=Path, default=Path("configs/glyph_entropy.toml"))
    parser.add_argument("--output", type=Path, default=Path("results/runs/glyph-entropy-models"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, help="Override for a separate validation run.")
    parser.add_argument("--widths", nargs="+", type=int, choices=(16, 32))
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    steps = config["steps"] if args.steps is None else args.steps
    if steps < 1 or config["batch_size"] < 1 or config["variants_per_glyph"] < 0:
        parser.error("steps and batch size must be positive, and variants must be nonnegative")

    torch.set_num_threads(4)
    glyphs, train_fonts, dataset_sha256 = load_training_glyphs(args.data)
    images, labels, augmented_sha256 = augment(
        glyphs, variants=config["variants_per_glyph"], seed=config["augmentation_seed"]
    )
    device = choose_device(args.device)
    images = images.to(device)
    labels = labels.to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{device}: {len(labels)} train-only images from {', '.join(train_fonts)}", flush=True)

    metadata = {
        "num_classes": 24,
        "image_size": 48,
        "device": str(device),
        "dataset_sha256": dataset_sha256,
        "augmented_sha256": augmented_sha256,
        "train_fonts": train_fonts,
        "training_examples": len(labels),
        "training_steps": steps,
        "prediction_target": "epsilon",
        "beta_schedule": "linear",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "pillow_version": pillow_version,
        **{key: value for key, value in config.items() if key not in {"steps", "widths", "seeds"}},
    }
    runs = []
    for seed in args.seeds or config["seeds"]:
        for width in args.widths or config["widths"]:
            settings = {
                **metadata,
                "base_channels": width,
                "seed": seed,
                "index_seed": seed + 10000,
                "timestep_seed": seed + 20000,
                "noise_seed": seed + 30000,
            }
            log = train_one(images, labels, args.output, settings)
            path = args.output / f"w{width}-s{seed}.pt"
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            runs.append({"file": path.name, "sha256": digest, "config": settings, "log": log})

    temporary = args.output / "runs.json.part"
    temporary.write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output / "runs.json")


if __name__ == "__main__":
    main()
