"""Create readable, traceable figures for a pilot run."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .data import PILOT_CHARACTERS, RenderedDataset


INK = (35, 48, 60)
MUTED = (97, 111, 123)
PAPER = (250, 249, 246)
CARD = (255, 255, 255)
ACCENT = (20, 131, 127)


def _load_font(font_path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is not None and font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default(size=size)


def _tensor_image(tensor: torch.Tensor | np.ndarray, side: int) -> Image.Image:
    """Convert a [-1, 1] single-channel model image into a visible grayscale tile."""

    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = tensor
    array = np.asarray(array, dtype=np.float32).squeeze()
    # In this project paper is +1 and ink is -1, so this maps back to ordinary L pixels.
    pixels = np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L").resize(
        (side, side), Image.Resampling.NEAREST
    )
    return image.convert("RGB")


def make_contact_sheet(
    images: torch.Tensor | np.ndarray,
    labels: np.ndarray | torch.Tensor,
    *,
    font_path: Path | None,
    title: str,
    cell_size: int = 104,
) -> Image.Image:
    """Lay generated samples in labeled rows, one row for each character."""

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    labels = np.asarray(labels)
    images = np.asarray(images)
    count_per_row = max(int(np.count_nonzero(labels == item.class_id)) for item in PILOT_CHARACTERS)
    margin = 28
    title_height = 58
    label_width = 92
    gap = 12
    width = margin * 2 + label_width + count_per_row * cell_size + max(count_per_row - 1, 0) * gap
    height = margin * 2 + title_height + len(PILOT_CHARACTERS) * (cell_size + 42)
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(font_path, 22)
    label_font = _load_font(font_path, 20)
    small_font = _load_font(font_path, 13)
    draw.text((margin, margin), title, font=title_font, fill=INK)

    for row, character in enumerate(PILOT_CHARACTERS):
        row_y = margin + title_height + row * (cell_size + 42)
        draw.text(
            (margin, row_y + 3),
            character.glyph,
            font=label_font,
            fill=ACCENT,
        )
        draw.text(
            (margin, row_y + 31),
            f"{character.stroke_count} strokes",
            font=small_font,
            fill=MUTED,
        )
        positions = np.flatnonzero(labels == character.class_id)
        for column, image_index in enumerate(positions):
            x = margin + label_width + column * (cell_size + gap)
            draw.rounded_rectangle(
                (x, row_y, x + cell_size, row_y + cell_size),
                radius=9,
                fill=CARD,
                outline=(223, 224, 220),
            )
            tile = _tensor_image(images[image_index], cell_size - 16)
            canvas.paste(tile, (x + 8, row_y + 8))
            draw.text(
                (x + 4, row_y + cell_size + 7),
                f"sample {column + 1}",
                font=small_font,
                fill=MUTED,
            )
    return canvas


def save_dataset_preview(
    dataset: RenderedDataset,
    output_path: Path,
    *,
    font_path: Path,
) -> None:
    """Show representative rendered training glyphs by class and font family."""

    selection: list[int] = []
    selected_labels: list[int] = []
    for character in PILOT_CHARACTERS:
        for font in dataset.font_manifest:
            matches = [
                index
                for index, record in enumerate(dataset.records)
                if record["class_id"] == character.class_id
                and record["font_family"] == font["family"]
            ][:3]
            selection.extend(matches)
            selected_labels.extend([character.class_id] * len(matches))

    # The standard sample sheet expects one row per class. Its selected examples
    # still show both font families, and the caption states how they were made.
    chosen = dataset.images[selection]
    labels = np.asarray(selected_labels)
    image = make_contact_sheet(
        chosen,
        labels,
        font_path=font_path,
        title="Font variation: samples 1–3 Sans, 4–6 Serif",
        cell_size=76,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_training_curve(loss_path: Path, output_path: Path) -> None:
    """Plot logged training noise MSE (not a validation or recognition score)."""

    history = json.loads(loss_path.read_text()) if loss_path.exists() else []
    width, height = 820, 420
    margin_left, margin_top, margin_right, margin_bottom = 76, 62, 28, 58
    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=24)
    label_font = ImageFont.load_default(size=16)
    small_font = ImageFont.load_default(size=13)
    draw.text((margin_left, 18), "Training noise-prediction loss", font=title_font, fill=INK)
    if not history:
        draw.text((margin_left, margin_top + 30), "No training history was recorded.", font=label_font, fill=MUTED)
        canvas.save(output_path)
        return

    steps = [float(item["step"]) for item in history]
    losses = [float(item["mean_noise_mse"]) for item in history]
    chart_left = margin_left
    chart_top = margin_top
    chart_right = width - margin_right
    chart_bottom = height - margin_bottom
    low = min(losses)
    high = max(losses)
    span = max(high - low, 1e-6)
    points = []
    for step, loss in zip(steps, losses, strict=True):
        x = chart_left + (step - min(steps)) / max(max(steps) - min(steps), 1) * (chart_right - chart_left)
        y = chart_bottom - (loss - low) / span * (chart_bottom - chart_top)
        points.append((int(x), int(y)))
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill=(180, 186, 188), width=2)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill=(180, 186, 188), width=2)
    if len(points) > 1:
        draw.line(points, fill=ACCENT, width=4)
    for point in points:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=ACCENT)
    draw.text((chart_left, chart_bottom + 14), "optimizer updates", font=small_font, fill=MUTED)
    draw.text((chart_left, chart_top - 22), f"MSE {high:.3f}", font=small_font, fill=MUTED)
    draw.text((chart_left, chart_bottom - 18), f"MSE {low:.3f}", font=small_font, fill=MUTED)
    draw.text(
        (margin_left, height - 25),
        "Lower training noise error is not, by itself, evidence of character recognition or generalization.",
        font=small_font,
        fill=MUTED,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def save_reverse_gif(
    snapshots: dict[int, torch.Tensor],
    labels: torch.Tensor,
    output_path: Path,
    *,
    font_path: Path,
) -> None:
    """Animate saved states from initial noise to the final character images."""

    frames: list[Image.Image] = []
    for remaining_steps in sorted(snapshots, reverse=True):
        state_name = "starting Gaussian noise" if remaining_steps == max(snapshots) else (
            "final generated glyph" if remaining_steps == 0 else f"{remaining_steps} forward-noise steps remain"
        )
        sheet = make_contact_sheet(
            snapshots[remaining_steps],
            labels,
            font_path=font_path,
            title=f"Reverse diffusion · {state_name}",
            cell_size=76,
        )
        frames.append(sheet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=750,
        loop=0,
        optimize=True,
    )


def write_demo_page(
    output_directory: Path,
    *,
    config: dict[str, Any],
) -> Path:
    """Write a self-contained local page for the outputs of one reproducible run."""

    characters = ", ".join(
        f"{item['glyph']} ({item['stroke_count']} strokes)" for item in config["characters"]
    )
    font_names = ", ".join(font["family"] for font in config["fonts"])
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>字生 — pilot run</title>
  <style>
    :root {{ color-scheme: light; --ink: #23303c; --muted: #61717b; --accent: #14837f; --paper: #faf9f6; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 42px 24px 72px; }}
    h1 {{ margin-bottom: 0.2em; font-size: clamp(2rem, 6vw, 3.4rem); }}
    h2 {{ margin-top: 2.2em; }}
    .lead {{ max-width: 760px; color: var(--muted); }}
    .tag {{ display: inline-block; border-radius: 999px; background: #e3f2ef; color: #146d68; padding: 4px 11px; font-size: 0.86rem; }}
    section {{ margin-top: 28px; }}
    figure {{ margin: 14px 0; padding: 16px; background: white; border: 1px solid #e2e2dc; border-radius: 14px; }}
    img {{ display: block; width: 100%; height: auto; border-radius: 8px; }}
    figcaption, .note {{ margin-top: 10px; color: var(--muted); font-size: 0.92rem; }}
    code {{ padding: 2px 5px; background: #eee; border-radius: 4px; }}
    footer {{ margin-top: 40px; border-top: 1px solid #deded8; padding-top: 16px; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <span class="tag">Runnable research pilot · {html.escape(str(config['device']))}</span>
    <h1>字生: from noise to characters</h1>
    <p class="lead">This run trains a small class-conditional diffusion model on rendered examples of {html.escape(characters)}. The samples below are model outputs from the saved checkpoint.</p>

    <section>
      <h2>Generated samples</h2>
      <figure><img src="samples.png" alt="Generated character samples grouped by requested class"><figcaption>Each row is conditioned on one requested character. The grid is a qualitative pilot view; no recognition metric has been computed.</figcaption></figure>
    </section>

    <section>
      <h2>Reverse process</h2>
      <figure><img src="reverse_diffusion.gif" alt="Animation from Gaussian noise to generated characters"><figcaption>The animation shows selected states while the denoiser repeatedly predicts and removes noise.</figcaption></figure>
    </section>

    <section>
      <h2>Training and data</h2>
      <figure><img src="training_curve.png" alt="Training noise-prediction mean squared error"><figcaption>Training noise-prediction MSE is an optimization trace, not a validation score.</figcaption></figure>
      <figure><img src="training_examples.png" alt="Examples of rendered training glyphs"><figcaption>Training images were rendered locally from {html.escape(font_names)}. The fonts are used under the SIL Open Font License 1.1; see the dataset manifest for source URLs and hashes.</figcaption></figure>
      <p class="note">Configuration: {int(config['image_size'])}×{int(config['image_size'])} grayscale images; {int(config['timesteps'])}-step linear beta schedule; {int(config['training_steps']):,} updates; seed {int(config['seed'])}.</p>
      <p class="note"><a href="dataset_manifest.json">Dataset manifest</a> · <a href="sample_config.json">Sample configuration</a></p>
    </section>

    <footer>This is a synthetic printed-font pilot, not handwriting, a human-written dataset, or evidence that the model generalizes to unseen fonts. Review the dataset manifest and model card before interpreting the samples.</footer>
  </main>
</body>
</html>
"""
    output_path = output_directory / "index.html"
    output_path.write_text(html_text, encoding="utf-8")
    return output_path
