#!/usr/bin/env python3
"""Download pinned sources, render the glyph study, and check split integrity."""

import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO, StringIO
from itertools import combinations
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np
from PIL import Image, ImageFont, __version__ as pillow_version, features
from fontTools.ttLib import TTFont


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data" / "study"
IMAGE_SIZE = 48
INK_BOX = 40
RASTER_SIZE = 192
FOREGROUND_MAX = 127
ABSENT_CHAR = "\U0010ffff"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def pinned_file(cache, filename, url, expected):
    path = cache / filename
    if path.exists():
        checksum = sha256(path.read_bytes())
    else:
        with urlopen(url, timeout=90) as response:
            content = response.read()
        checksum = sha256(content)
        if checksum == expected:
            path.write_bytes(content)
    if checksum != expected:
        raise ValueError(f"{path}: expected SHA-256 {expected}, got {checksum}")
    return path


def verified_characters(cache, sources):
    unicode = sources["unicode"]
    archive = pinned_file(cache, "Unihan.zip", unicode["source"], unicode["sha256"])
    with ZipFile(archive) as unihan:
        rows = unihan.read(unicode["member"]).decode().splitlines()
    strokes = {
        parts[0]: parts[2] for row in rows if not row.startswith("#")
        if len(parts := row.split("\t")) == 3 and parts[1] == unicode["property"]
    }
    characters = []
    for expected, group in sources["characters_by_stroke_count"].items():
        for glyph in group:
            codepoint = f"U+{ord(glyph):04X}"
            count = int(strokes[codepoint])
            if count != int(expected):
                raise ValueError(f"{glyph} {codepoint}: expected {expected} strokes, got {count}")
            characters.append({
                "class_id": len(characters),
                "glyph": glyph,
                "codepoint": codepoint,
                "stroke_count": count,
                "stroke_source": f"Unicode {unicode['version']} {unicode['property']} ({unicode['source']} / {unicode['member']})",
            })
    if len({row["glyph"] for row in characters}) != 24 or Counter(
        row["stroke_count"] for row in characters
    ) != {4: 6, 6: 6, 8: 6, 10: 6}:
        raise ValueError("Expected six distinct characters per stroke group")
    return characters


def ink_mask(font, glyph):
    mask = font.getmask(glyph, mode="L")
    return Image.frombytes("L", mask.size, bytes(mask))


def render(mask):
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Font raster is empty")
    cropped = mask.crop(bbox)
    width, height = cropped.size
    long_side = max(width, height)
    fitted = cropped.resize(
        (max(1, round(width * INK_BOX / long_side)), max(1, round(height * INK_BOX / long_side))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
    canvas.paste(
        0,
        ((IMAGE_SIZE - fitted.width) // 2, (IMAGE_SIZE - fitted.height) // 2),
        fitted,
    )
    return np.asarray(canvas)


def font_glyphs(cache, source, characters):
    filename = source["source"].rsplit("/", 1)[-1]
    path = pinned_file(cache, filename, source["source"], source["sha256"])
    license_path = pinned_file(
        cache, f"{source['font_id']}.license", source["license_url"], source["license_sha256"],
    )
    font = ImageFont.truetype(str(path), RASTER_SIZE, layout_engine=ImageFont.Layout.BASIC)
    fallback = ink_mask(font, ABSENT_CHAR)
    glyphs = []
    coverage = []
    with TTFont(path, lazy=True) as ft_font:
        font_record = {
            **source,
            "filename": filename,
            "font_version": ft_font["name"].getDebugName(5),
            "license_notice": license_path.read_text().splitlines()[0],
            "glyph_coverage": len(characters),
        }
        cmap = ft_font.getBestCmap()
        if ord(ABSENT_CHAR) in cmap:
            raise ValueError(f"{filename}: missing-glyph sentinel is present in cmap")
        seen = set()
        for character in characters:
            glyph = character["glyph"]
            glyph_name = cmap.get(ord(glyph))
            if glyph_name is None:
                raise ValueError(f"{filename} {glyph}: missing from Unicode cmap")
            glyph_id = ft_font.getGlyphID(glyph_name)
            if glyph_id == 0 or glyph_id in seen:
                raise ValueError(f"{filename} {glyph}: .notdef or reused glyph ID {glyph_id}")
            seen.add(glyph_id)
            try:
                mask = ink_mask(font, glyph)
                if mask == fallback:
                    raise ValueError("FreeType raster matches the missing-glyph sentinel")
                raster = render(mask)
                if not np.any(raster <= FOREGROUND_MAX):
                    raise ValueError("Rendered raster has no foreground")
                if np.any(raster[[0, -1], :] != 255) or np.any(raster[:, [0, -1]] != 255):
                    raise ValueError("Rendered ink touches the image border")
            except ValueError as error:
                raise ValueError(f"{filename} {glyph}: {error}") from error
            coverage.append({
                "font_id": source["font_id"],
                "split": source["split"],
                "class_id": character["class_id"],
                "glyph": glyph,
                "codepoint": character["codepoint"],
                "glyph_name": glyph_name,
                "glyph_id": glyph_id,
                "glyph_sha256": sha256(raster.tobytes()),
                "foreground_sha256": sha256((raster <= FOREGROUND_MAX).tobytes()),
            })
            glyphs.append(raster)
    return font_record, glyphs, coverage


def validate_splits(fonts, characters, glyphs):
    if Counter(font["split"] for font in fonts) != {"train": 3, "val": 1, "test": 2}:
        raise ValueError("Expected split sizes train=3, val=1 and test=2")
    for field in ("font_id", "family", "lineage", "sha256"):
        if len({font[field] for font in fonts}) != len(fonts):
            raise ValueError(f"Each font must have a unique {field}")
    masks = glyphs <= FOREGROUND_MAX
    comparisons = 0
    grayscale = []
    foreground = []
    for left, right in combinations(range(len(fonts)), 2):
        if fonts[left]["split"] == fonts[right]["split"]:
            continue
        for column, character in enumerate(characters):
            comparisons += 1
            match = {
                "font_ids": [fonts[left]["font_id"], fonts[right]["font_id"]],
                "class_id": character["class_id"],
                "glyph": character["glyph"],
            }
            if np.array_equal(glyphs[left, column], glyphs[right, column]):
                grayscale.append(match)
            if np.array_equal(masks[left, column], masks[right, column]):
                foreground.append(match)
    report = {
        "cross_split_same_character_pairs": comparisons,
        "exact_grayscale_duplicates": grayscale,
        "exact_foreground_duplicates": foreground,
        "cmap_covered_pairs": len(fonts) * len(characters),
        "fallback_rasters": 0,
        "empty_foreground_rasters": 0,
        "ink_on_border_rasters": 0,
    }
    if grayscale or foreground:
        raise ValueError(f"Duplicate glyphs across splits: {report}")
    return report


def glyph_npz(glyphs):
    buffer = BytesIO()
    array = BytesIO()
    np.lib.format.write_array(array, glyphs, version=(1, 0), allow_pickle=False)
    with ZipFile(buffer, "w") as archive:
        archive.writestr(ZipInfo("glyphs.npy"), array.getvalue(), compress_type=ZIP_STORED)
    return buffer.getvalue()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def csv_bytes(rows):
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def build(cache):
    sources = json.loads((STUDY / "sources.json").read_text())
    characters = verified_characters(cache, sources)
    fonts, rasters, coverage = [], [], []
    for source in sources["fonts"]:
        font_record, glyphs, records = font_glyphs(cache, source, characters)
        fonts.append(font_record)
        rasters.append(glyphs)
        coverage.extend(records)
    glyphs = np.asarray(rasters, dtype=np.uint8)
    qc = validate_splits(fonts, characters, glyphs)
    archive = glyph_npz(glyphs)
    manifest = {
        "schema_version": 1,
        "image_size": IMAGE_SIZE,
        "fonts": fonts,
        "characters": characters,
        "stroke_source": sources["unicode"],
        "rendering": {
            "pillow_version": pillow_version,
            "freetype_version": features.version_module("freetype2"),
            "layout": "Pillow BASIC / one Unicode scalar",
            "initial_font_size_px": RASTER_SIZE,
            "antialias": "FreeType 8-bit grayscale",
            "crop": "Tight bounding box of positive rendered ink before resizing",
            "fit_box_px": [INK_BOX, INK_BOX],
            "fit": "Preserve aspect ratio, round scaled dimensions to nearest integer with ties to even",
            "placement": "Center the resized box, floor left and top offsets",
            "resampling": "Pillow LANCZOS",
            "ink": 0,
            "paper": 255,
            "foreground_max": FOREGROUND_MAX,
        },
        "glyph_file": "glyphs.npz",
        "glyph_key": "glyphs",
        "glyph_file_sha256": sha256(archive),
        "glyph_shape": list(glyphs.shape),
        "glyph_dtype": str(glyphs.dtype),
        "quality_checks": qc,
    }
    return {
        "glyphs.npz": archive,
        "characters.csv": csv_bytes(characters),
        "fonts.csv": csv_bytes(fonts),
        "coverage.csv": csv_bytes(coverage),
        "quality_checks.json": json_bytes(qc),
        "manifest.json": json_bytes(manifest),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "raw" / "glyph-entropy")
    parser.add_argument("--output-dir", type=Path, default=STUDY)
    parser.add_argument("--verify", action="store_true", help="Compare all generated files byte for byte")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    results = build(args.cache_dir)
    if args.verify:
        different = [
            name for name, content in results.items()
            if not (path := args.output_dir / name).is_file() or path.read_bytes() != content
        ]
        if different:
            raise SystemExit(f"Generated files differ: {', '.join(different)}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, contents in results.items():
            (args.output_dir / name).write_bytes(contents)
    print(f"{'Verified' if args.verify else 'Wrote'} 144 glyphs in {args.output_dir}")
    print(f"glyphs.npz SHA-256 {sha256(results['glyphs.npz'])}")


if __name__ == "__main__":
    main()
