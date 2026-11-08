"""Load a saved model and create samples, a reverse-process animation, and a page."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .data import (
    PILOT_CHARACTERS,
    build_pilot_dataset,
    ensure_pilot_fonts,
    write_dataset_manifest,
)
from .diffusion import GaussianDiffusion
from .evaluate import (
    make_contact_sheet,
    save_dataset_preview,
    save_reverse_gif,
    save_training_curve,
    write_demo_page,
)
from .model import ConditionalDenoiser
from .train import choose_device


def sample(
    checkpoint: Path,
    character_ids: torch.Tensor | None = None,
    *,
    seed: int = 7,
    output_directory: Path | None = None,
    font_directory: Path | None = None,
    samples_per_character: int = 4,
    requested_device: str = "auto",
) -> Path:
    """Run the reverse chain for each class and save a local visual report."""

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_directory = (output_directory or checkpoint.parent).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    device = choose_device(requested_device)

    # The checkpoint contains only tensors and basic Python values, so it can be
    # loaded with PyTorch's safer weights-only mode.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = ConditionalDenoiser(num_classes=int(config["num_classes"])).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    diffusion = GaussianDiffusion(
        timesteps=int(config["timesteps"]),
        beta_start=float(config["beta_start"]),
        beta_end=float(config["beta_end"]),
    ).to(device)

    if character_ids is None:
        class_ids = list(range(len(PILOT_CHARACTERS)))
    else:
        class_ids = [int(item) for item in character_ids.detach().cpu().tolist()]
    labels = torch.tensor(
        [class_id for class_id in class_ids for _ in range(samples_per_character)],
        dtype=torch.long,
        device=device,
    )

    # Snapshot a few points through the reverse chain. These make the process
    # inspectable without storing every intermediate tensor from every step.
    total = int(config["timesteps"])
    snapshot_steps = sorted({0, total, round(total * 0.25), round(total * 0.5), round(total * 0.75)})
    generated, snapshots = diffusion.sample(
        model,
        labels,
        image_size=int(config["image_size"]),
        seed=seed,
        snapshot_steps=snapshot_steps,
    )
    labels_cpu = labels.detach().cpu()
    generated_cpu = generated.detach().cpu()

    font_directory = font_directory or Path("data/fonts")
    fonts = ensure_pilot_fonts(font_directory)
    font_path = fonts[0]
    grid = make_contact_sheet(
        generated_cpu,
        labels_cpu,
        font_path=font_path,
        title="Class-conditional generated samples",
        cell_size=108,
    )
    grid.save(output_directory / "samples.png")
    save_reverse_gif(
        snapshots,
        labels_cpu,
        output_directory / "reverse_diffusion.gif",
        font_path=font_path,
    )

    loss_path = checkpoint.parent / "training_loss.json"
    save_training_curve(loss_path, output_directory / "training_curve.png")

    variants_per_font = int(config.get("variants_per_font", 128))
    dataset = build_pilot_dataset(
        font_directory,
        variants_per_font=variants_per_font,
        seed=int(config["seed"]),
        image_size=int(config["image_size"]),
    )
    write_dataset_manifest(
        dataset,
        output_directory / "dataset_manifest.json",
        seed=int(config["seed"]),
        variants_per_font=variants_per_font,
        image_size=int(config["image_size"]),
    )
    save_dataset_preview(
        dataset,
        output_directory / "training_examples.png",
        font_path=font_path,
    )

    report_config = dict(config)
    report_config["device"] = str(device)
    report_config["sample_seed"] = seed
    report_config["samples_per_character"] = samples_per_character
    report_config["characters"] = [
        {
            "glyph": item.glyph,
            "class_id": item.class_id,
            "stroke_count": item.stroke_count,
            "gloss": item.gloss,
        }
        for item in PILOT_CHARACTERS
    ]
    (output_directory / "sample_config.json").write_text(
        json.dumps(report_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    page = write_demo_page(output_directory, config=report_config)
    print(f"Saved visual demo: {page}", flush=True)
    return page
