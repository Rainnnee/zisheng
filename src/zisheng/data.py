"""Create a documented, font-rendered character pilot dataset.

The expanded pilot uses synthetic printed glyphs so that the data source and
labels are known. These samples are useful for checking a diffusion pipeline;
they do not represent handwriting or the full variation of Chinese writing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_SIZE = 48
SUPERSAMPLE = 4


@dataclass(frozen=True)
class Character:
    """A pilot label and its conventional stroke-count annotation."""

    glyph: str
    class_id: int
    stroke_count: int
    gloss: str


PILOT_CHARACTERS = (
    Character("山", 0, 3, "mountain"),
    Character("水", 1, 4, "water"),
    Character("火", 2, 4, "fire"),
    Character("心", 3, 4, "heart / mind"),
    Character("永", 4, 5, "everlasting"),
    Character("安", 5, 6, "peace / safety"),
    Character("志", 6, 7, "aspiration"),
    Character("花", 7, 8, "flower"),
    Character("明", 8, 8, "bright / clear"),
    Character("和", 9, 8, "harmony"),
    Character("信", 10, 9, "trust"),
    Character("笑", 11, 10, "smile / laughter"),
)


# The release tags make the source reproducible. Noto CJK is released under
# the SIL Open Font License 1.1; the license texts are downloaded beside fonts.
FONT_SOURCES = (
    {
        "name": "Noto Sans CJK SC Regular",
        "release": "Sans2.004",
        "filename": "NotoSansCJKsc-Regular.otf",
        "url": (
            "https://raw.githubusercontent.com/notofonts/noto-cjk/"
            "Sans2.004/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
        ),
        "license_filename": "Noto-Sans-CJK-LICENSE.txt",
        "license_url": (
            "https://raw.githubusercontent.com/notofonts/noto-cjk/"
            "main/Sans/LICENSE"
        ),
    },
    {
        "name": "Noto Serif CJK SC Regular",
        "release": "Serif2.003",
        "filename": "NotoSerifCJKsc-Regular.otf",
        "url": (
            "https://raw.githubusercontent.com/notofonts/noto-cjk/"
            "Serif2.003/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf"
        ),
        "license_filename": "Noto-Serif-CJK-LICENSE.txt",
        "license_url": (
            "https://raw.githubusercontent.com/notofonts/noto-cjk/"
            "main/Serif/LICENSE"
        ),
    },
)


@dataclass(frozen=True)
class RenderedDataset:
    """An in-memory balanced dataset plus source records for traceability."""

    images: np.ndarray  # (examples, 1, height, width), scaled to [-1, 1]
    labels: np.ndarray  # (examples,), class IDs from PILOT_CHARACTERS
    records: tuple[dict[str, object], ...]
    font_manifest: tuple[dict[str, object], ...]


def _download(url: str, destination: Path) -> None:
    """Fetch a file atomically, keeping incomplete downloads out of the data."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1024:
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "zisheng-research-demo/0.1"},
    )
    print(f"Downloading {destination.name} from the Noto CJK release…", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    if len(content) < 1024:
        raise RuntimeError(f"The download from {url} was unexpectedly small.")
    temporary.write_bytes(content)
    temporary.replace(destination)


def ensure_pilot_fonts(font_directory: Path) -> tuple[Path, ...]:
    """Download the two openly licensed Simplified Chinese pilot fonts."""

    paths: list[Path] = []
    for source in FONT_SOURCES:
        font_path = font_directory / str(source["filename"])
        license_path = font_directory / str(source["license_filename"])
        _download(str(source["url"]), font_path)
        _download(str(source["license_url"]), license_path)

        license_text = license_path.read_text(encoding="utf-8", errors="replace")
        if "SIL OPEN FONT LICENSE" not in license_text or "Version 1.1" not in license_text:
            raise RuntimeError(f"The expected OFL 1.1 license was not found for {source['name']}.")

        # This also catches HTML error pages saved under an .otf filename.
        ImageFont.truetype(str(font_path), size=64)
        paths.append(font_path)
    return tuple(paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_variant(
    glyph: str,
    font_path: Path,
    rng: np.random.Generator,
    *,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Render one small, ink-on-paper glyph with mild label-preserving changes."""

    scale = SUPERSAMPLE
    canvas_size = image_size * scale
    # A little size and weight variation makes examples less like exact copies.
    # Keep the font-size range proportional so older 32px configs can still be
    # regenerated while the current 48px pilot has more room for dense glyphs.
    min_font_size = round(image_size * 0.75)
    max_font_size = round(image_size * 0.875)
    font_size = int(rng.integers(min_font_size, max_font_size + 1)) * scale
    stroke_width = int(rng.choice([0, 0, 2, 4]))
    font = ImageFont.truetype(str(font_path), size=font_size)
    canvas = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(canvas)

    # anchor="mm" centers the glyph's typographic box. Small offsets and
    # rotations simulate harmless layout differences without changing identity.
    offset_x = int(rng.integers(-2, 3)) * scale
    offset_y = int(rng.integers(-2, 3)) * scale
    draw.text(
        (canvas_size // 2 + offset_x, canvas_size // 2 + offset_y),
        glyph,
        font=font,
        fill=0,
        stroke_width=stroke_width,
        stroke_fill=0,
        anchor="mm",
    )
    angle = float(rng.uniform(-5.0, 5.0))
    canvas = canvas.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor=255,
    )
    small = canvas.resize((image_size, image_size), Image.Resampling.LANCZOS)

    # The diffusion equations use a symmetric range: dark ink is -1, paper is +1.
    pixels = np.asarray(small, dtype=np.float32)
    return (pixels / 127.5 - 1.0)[None, ...]


def build_pilot_dataset(
    font_directory: Path,
    *,
    variants_per_font: int = 128,
    seed: int = 2026,
    image_size: int = IMAGE_SIZE,
) -> RenderedDataset:
    """Render a balanced three-character dataset and its source manifest."""

    fonts = ensure_pilot_fonts(font_directory)
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict[str, object]] = []
    font_manifest: list[dict[str, object]] = []

    for source, font_path in zip(FONT_SOURCES, fonts, strict=True):
        # Fail early if a requested character is missing from a font.
        font = ImageFont.truetype(str(font_path), size=96)
        for character in PILOT_CHARACTERS:
            if font.getmask(character.glyph).getbbox() is None:
                raise ValueError(
                    f"{character.glyph} is not covered by {source['name']}."
                )

        font_manifest.append(
            {
                "family": source["name"],
                "release": source["release"],
                "filename": font_path.name,
                "sha256": _sha256(font_path),
                "license": "SIL Open Font License 1.1",
                "license_file": source["license_filename"],
                "license_sha256": _sha256(font_directory / str(source["license_filename"])),
                "license_url": source["license_url"],
                "source_url": source["url"],
            }
        )

    for character in PILOT_CHARACTERS:
        for source, font_path in zip(FONT_SOURCES, fonts, strict=True):
            for variant in range(variants_per_font):
                images.append(
                    _render_variant(
                        character.glyph,
                        font_path,
                        rng,
                        image_size=image_size,
                    )
                )
                labels.append(character.class_id)
                records.append(
                    {
                        "glyph": character.glyph,
                        "class_id": character.class_id,
                        "font_family": source["name"],
                        "font_release": source["release"],
                        "variant": variant,
                    }
                )

    return RenderedDataset(
        images=np.stack(images).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        records=tuple(records),
        font_manifest=tuple(font_manifest),
    )


def write_dataset_manifest(
    dataset: RenderedDataset,
    output_path: Path,
    *,
    seed: int,
    variants_per_font: int,
    image_size: int,
) -> None:
    """Save enough metadata to rebuild the exact synthetic pilot dataset."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_type": "synthetic rendered-font pilot",
        "seed": seed,
        "image_size": image_size,
        "variants_per_character_per_font": variants_per_font,
        "characters": [asdict(character) for character in PILOT_CHARACTERS],
        "sample_count": int(len(dataset.labels)),
        "samples_per_class": {
            character.glyph: int(np.count_nonzero(dataset.labels == character.class_id))
            for character in PILOT_CHARACTERS
        },
        "fonts": dataset.font_manifest,
        "rendering": {
            "background": "white",
            "ink": "black",
            "font_size_range_px": [
                round(image_size * 0.75),
                round(image_size * 0.875),
            ],
            "stroke_width_supersample_px": [0, 2, 4],
            "stroke_width_native_px": [0.0, 0.5, 1.0],
            "rotation_degrees": [-5, 5],
            "horizontal_vertical_shift_px": [-2, 2],
            "pixel_range": [-1.0, 1.0],
        },
        "limitations": [
            "Synthetic printed glyphs are not handwriting.",
            "The pilot uses two font families and label-preserving image variation.",
            "Font-rendered variants are not independent human-written examples.",
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
