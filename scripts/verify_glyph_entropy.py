"""Audit a frozen glyph-entropy evaluation from masks, glyphs and EMA weights."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path
import sys
import tomllib

import numpy as np
import torch

from zisheng.model import ConditionalDenoiser


FONT_SPLIT = {
    "train": ("noto-sans-cjk-sc", "zcool-xiaowei", "ma-shan-zheng"),
    "val": ("zcool-qingke-huangyou",),
    "test": ("zcool-kuaile", "long-cang"),
}
PANEL_SHA256 = "a0f84add514c1f9088c9648d369436b675d1badee4a263379255258d300693dc"
MANIFEST_SHA256 = "d4fdd79acd9877f4c29f1c229d92169cc63a9b9848c14af22cd5372d3d753158"
STROKES = (4, 6, 8, 10)
FEATURES = ("ink_fraction", "perimeter_complexity", "entropy")
CLASSES = 24
DRAWS = 16
SIDE = 48
PIXELS = SIDE**2
NOISE_SEEDS = {"test": 48117, "val": 48118}
TARGETS = tuple(range(-5, 1))
SCHEDULE = {"timesteps": 1000, "beta_start": 1e-4, "beta_end": 0.02}
NEURAL = tuple(("denoiser", w, s) for w, s in product((16, 32), (101, 202)))
LABEL = ("label-only", None, None)
NEAREST = ("gaussian-nearest", None, None)
METHODS = (*NEURAL, LABEL, NEAREST)
STATS_ATOL = 3e-12
CPU_X0_ATOL = 2e-4
ROOT = Path(__file__).resolve().parents[1]


class AuditError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256(path, decompress=False):
    open_bytes = gzip.open if decompress else open
    with open_bytes(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(actual, expected, location, atol=STATS_ATOL):
    result = np.asarray(actual)
    reference = np.asarray(expected)
    require(result.shape == reference.shape, f"{location}: shape {result.shape}, expected {reference.shape}")
    diff = np.max(np.abs(result - reference))
    require(diff <= atol, f"{location}: maximum absolute difference {diff:.6g} exceeds {atol:.6g}")
    return float(diff)


def verify_fields(actual, expected, location):
    """Compare an independently built portion of the summary, with indexed error locations."""
    if isinstance(expected, dict):
        for name, value in expected.items():
            verify_fields(actual[name], value, f"{location}.{name}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{location}: expected {len(expected)} entries")
        for i, (observed, wanted) in enumerate(zip(actual, expected)):
            verify_fields(observed, wanted, f"{location}[{i}]")
    elif isinstance(expected, (float, np.floating)):
        require(isinstance(actual, (float, int)), f"{location}: expected a number")
        tolerance = STATS_ATOL
        if location.endswith(".actual_log2_snr"):
            tolerance = 3e-6
        elif location.endswith(".alpha_bar"):
            tolerance = 3e-7
        assert_close(actual, expected, location, atol=tolerance)
    else:
        require(actual == expected, f"{location}: expected {expected!r}, got {actual!r}")


def read_panel(directory):
    manifest_file = directory / "manifest.json"
    manifest = json_read(manifest_file)
    manifest_hash = sha256(manifest_file)
    require(manifest_hash == MANIFEST_SHA256, "A1 manifest does not match the frozen fingerprint")
    archive = directory / "glyphs.npz"
    glyph_hash = sha256(archive)
    require(glyph_hash == PANEL_SHA256 == manifest["glyph_file_sha256"], "A1 glyph archive fingerprint differs")
    with np.load(archive, allow_pickle=False) as data:
        glyphs = data["glyphs"]
    fonts = manifest["fonts"]
    chars = manifest["characters"]
    require(glyphs.shape == (6, CLASSES, SIDE, SIDE) and glyphs.dtype == np.uint8, "Invalid panel format")
    require([f["font_id"] for f in fonts] == list(sum(FONT_SPLIT.values(), ())), "A1 font order differs")
    actual_split = {s: tuple(f["font_id"] for f in fonts if f["split"] == s) for s in FONT_SPLIT}
    require(actual_split == FONT_SPLIT, "A1 font split differs")
    require(len({f["lineage"] for f in fonts}) == 6, "A1 font lineages overlap")
    require([c["class_id"] for c in chars] == list(range(CLASSES)), "A1 class IDs differ")
    require(len({c["glyph"] for c in chars}) == CLASSES, "A1 labels are duplicated")
    require(all(c["codepoint"] == f"U+{ord(c['glyph']):04X}" for c in chars), "A1 codepoints differ")
    groups = np.array([c["stroke_count"] for c in chars])
    require(Counter(groups) == dict.fromkeys(STROKES, 6), "A1 stroke counts differ")
    masks = glyphs <= 127
    require(masks.any(axis=(-2, -1)).all(), "A1 foreground mask is empty")
    cross_split_pairs = 0
    for i, j in combinations(range(len(fonts)), 2):
        if fonts[i]["split"] == fonts[j]["split"]:
            continue
        cross_split_pairs += CLASSES
        require(not np.any(np.all(masks[i] == masks[j], axis=(-2, -1))),
                f"A1 foreground duplicate between {fonts[i]['font_id']} and {fonts[j]['font_id']}")
    identities = {"dataset_sha256": glyph_hash, "manifest_sha256": manifest_hash}
    return glyphs, chars, groups, identities, cross_split_pairs


def checkpoint_models(directory, dataset_hash, split):
    runs = json_read(directory / "runs.json")
    run_index = {run["file"]: run for run in runs}
    expected_files = {f"w{width}-s{seed}.pt" for _, width, seed in NEURAL}
    require(len(runs) == len(NEURAL) and set(run_index) == expected_files, "Unexpected run index")
    with (ROOT / "configs/glyph_entropy.toml").open("rb") as stream:
        frozen = tomllib.load(stream)
    final_steps = frozen.pop("steps")
    frozen.pop("widths")
    frozen.pop("seeds")
    expected = {
        **frozen, "num_classes": CLASSES, "image_size": SIDE, "dataset_sha256": dataset_hash,
        "prediction_target": "epsilon", "beta_schedule": "linear",
        "train_fonts": list(FONT_SPLIT["train"]),
        "training_examples": CLASSES * len(FONT_SPLIT["train"]) * (1 + frozen["variants_per_glyph"]),
    }
    models, entries = [], []
    for _, width, seed in NEURAL:
        name = f"w{width}-s{seed}.pt"
        path = directory / name
        digest = sha256(path)
        run = run_index[name]
        require(digest == run["sha256"], f"{name}: SHA-256 differs from runs.json")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        config = saved["config"]
        require(run["config"] == config, f"{name}: indexed checkpoint configuration differs")
        require(run["log"] == saved["training_log"], f"{name}: indexed training log differs")
        valid_steps = (final_steps,) if split == "test" else (128, final_steps)
        require(config["training_steps"] in valid_steps, f"{name}: incorrect update budget")
        require(saved["training_log"][-1]["update"] == config["training_steps"],
                f"{name}: last saved update differs from budget")
        checks = {
            **expected, "base_channels": width, "seed": seed,
            "index_seed": seed + 10000, "timestep_seed": seed + 20000, "noise_seed": seed + 30000,
        }
        for key, value in checks.items():
            require(config.get(key) == value, f"{name}: checkpoint {key} differs from frozen setting")
        model = ConditionalDenoiser(num_classes=CLASSES, base_channels=width).eval()
        model.load_state_dict(saved["model_state_dict"])
        models.append(model)
        entries.append({"file": name, "checkpoint_id": digest, "config": config})
    shared = set(entries[0]["config"]) - {"base_channels", "seed", "index_seed", "timestep_seed", "noise_seed"}
    require(all(all(entry["config"][k] == entries[0]["config"][k] for k in shared) for entry in entries),
            "Training settings or augmented training data differ across checkpoints")
    return models, entries


def schedule_steps():
    betas = np.linspace(SCHEDULE["beta_start"], SCHEDULE["beta_end"], SCHEDULE["timesteps"])
    alpha_bar = np.cumprod(1 - betas)
    snr = np.log2(alpha_bar / (1 - alpha_bar))
    indices = [int(np.argmin(np.abs(snr - target))) for target in TARGETS]
    require(len(set(indices)) == len(TARGETS), "SNR indices are duplicated")
    return [
        {"target_log2_snr": target, "timestep": step,
         "actual_log2_snr": float(snr[step]), "alpha_bar": float(alpha_bar[step])}
        for target, step in zip(TARGETS, indices)
    ]


def describe(method):
    return dict(zip(("method", "width", "seed"), method))


def get_scores(directory, font_ids, chars, steps, checkpoints):
    """Index the table by semantic keys and keep an explicit mapping to replay mask rows."""
    plain = directory / "scores.csv"
    scores_file = plain if plain.exists() else directory / "scores.csv.gz"
    open_text = gzip.open if scores_file.suffix == ".gz" else open
    shape = (len(METHODS), len(TARGETS), len(font_ids), CLASSES, DRAWS)
    row_numbers = np.full(shape, -1)
    row_seen = np.zeros(row_numbers.size, dtype=bool)
    stored_scores = np.empty(shape)
    nearest_fonts = np.empty(shape[1:], dtype=np.int8)
    actual_snrs = np.full(len(TARGETS), np.nan)
    template_ids = sorted(FONT_SPLIT["train"])
    columns = {
        "mask_row", "method", "font_id", "class_id", "glyph", "target_log2_snr",
        "timestep", "actual_log2_snr", "width", "seed", "noise_id", "checkpoint_id",
        "template_font_id", "dice",
    }
    with open_text(scores_file, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(set(reader.fieldnames or []) == columns and len(reader.fieldnames) == len(columns),
                "Invalid score CSV columns")
        for line, row in enumerate(reader, start=2):
            require(None not in row and None not in row.values(), f"CSV line {line}: invalid field count")
            try:
                width = int(row["width"]) if row["width"] else None
                seed = int(row["seed"]) if row["seed"] else None
                method = METHODS.index((row["method"], width, seed))
                snr = TARGETS.index(int(row["target_log2_snr"]))
                font = font_ids.index(row["font_id"])
                c, draw, mask_row = map(int, (row["class_id"], row["noise_id"], row["mask_row"]))
                score = float(row["dice"])
                reported_snr = float(row["actual_log2_snr"])
            except (ValueError, TypeError) as exc:
                raise AuditError(f"CSV line {line}: malformed row key or value") from exc
            require(0 <= c < CLASSES and 0 <= draw < DRAWS and 0 <= mask_row < len(row_seen),
                    f"CSV line {line}: key out of range")
            require(0 <= score <= 1 and row["glyph"] == chars[c]["glyph"],
                    f"CSV line {line}: invalid Dice or glyph")
            index = (method, snr, font, c, draw)
            require(not row_seen[mask_row] and row_numbers[index] == -1,
                    f"CSV line {line}: duplicate semantic key or replay row")
            expected_checkpoint = checkpoints[method]["checkpoint_id"] if method < len(NEURAL) else ""
            require(row["checkpoint_id"] == expected_checkpoint, f"CSV line {line}: wrong checkpoint")
            require(row["timestep"] == str(steps[snr]["timestep"]), f"CSV line {line}: wrong timestep")
            if np.isnan(actual_snrs[snr]):
                assert_close(reported_snr, steps[snr]["actual_log2_snr"], "CSV log2 SNR", atol=3e-6)
                actual_snrs[snr] = reported_snr
            require(reported_snr == actual_snrs[snr], f"CSV line {line}: inconsistent actual SNR")
            if METHODS[method] == NEAREST:
                require(row["template_font_id"] in template_ids, f"CSV line {line}: invalid selected template")
                nearest_fonts[index[1:]] = template_ids.index(row["template_font_id"])
            else:
                require(row["template_font_id"] == "", f"CSV line {line}: unexpected template ID")
            row_seen[mask_row] = True
            row_numbers[index] = mask_row
            stored_scores[index] = score
    require(row_seen.all() and np.all(row_numbers >= 0), f"Expected {row_numbers.size} complete score rows")
    return scores_file, row_numbers, stored_scores, nearest_fonts, actual_snrs


def unpack(packed, shape):
    require(packed.dtype == np.uint8 and packed.shape == (*shape, PIXELS // 8),
            f"Packed masks must be uint8 with shape {(*shape, PIXELS // 8)}")
    return np.unpackbits(packed, axis=-1, bitorder="little").view(bool)


def dice(a, b):
    foreground_total = a.sum(axis=-1, dtype=np.int32) + b.sum(axis=-1, dtype=np.int32)
    twice_overlap = 2 * (a & b).sum(axis=-1, dtype=np.int32)
    return np.divide(twice_overlap, foreground_total, out=np.ones(foreground_total.shape), where=foreground_total != 0)


def source_contrast(pred_a, pred_b, clean_a, clean_b):
    return (dice(pred_a, clean_a) + dice(pred_b, clean_b)
            - (dice(pred_a, clean_b) + dice(pred_b, clean_a))) / 2


def read_replay(directory, panel):
    with np.load(directory / "replay.npz", allow_pickle=False) as archive:
        replay = {name: archive[name] for name in archive.files}
    font_ids = tuple(replay["font_ids"].tolist())
    split = next((name for name in ("val", "test") if font_ids == FONT_SPLIT[name]), None)
    require(split is not None, "Replay evaluation fonts differ from either frozen holdout split")
    clean = panel[-2:] if split == "test" else panel[3:4]
    require(replay["clean_uint8"].dtype == np.uint8 and np.array_equal(replay["clean_uint8"], clean),
            "Replay clean rasters differ from A1 panel")
    require(np.array_equal(replay["class_ids"], np.arange(CLASSES)), "Replay class IDs differ")
    require(np.array_equal(replay["noise_ids"], np.arange(DRAWS)), "Replay noise IDs differ")
    require(np.array_equal(replay["target_log2_snr"], TARGETS), "Replay target SNRs differ")
    reference = (clean <= 127).reshape(len(font_ids), CLASSES, PIXELS)
    require(np.array_equal(unpack(replay["clean_masks_packed"], reference.shape[:-1]), reference),
            "Packed clean masks disagree with the frozen grayscale rasters")
    return replay, split, font_ids, clean, reference


def train_features(panel, chars):
    features = []
    for train in panel[:len(FONT_SPLIT["train"])]:
        rows = []
        for glyph in train:
            ink = (glyph <= 127).astype(np.int32)
            area = int(ink.sum())
            windows = np.lib.stride_tricks.sliding_window_view(ink, (2, 2)).reshape(-1, 4)
            window_ink = windows.sum(axis=1)
            mixed = windows[(window_ink > 0) & (window_ink < 4)]
            _, counts = np.unique(mixed, axis=0, return_counts=True)
            total_mixed = counts.sum()
            entropy = math.log2(total_mixed) - sum(int(n) * math.log2(n) for n in counts) / total_mixed
            adjacent = sum((ink.take(range(1, SIDE), axis=axis) *
                            ink.take(range(SIDE - 1), axis=axis)).sum() for axis in (0, 1))
            perimeter = 4 * area - 2 * int(adjacent)
            rows.append([area / PIXELS, perimeter**2 / (math.tau * 2 * area), entropy])
        features.append(rows)
    all_features = np.asarray(features)
    median = np.median(all_features, axis=0)
    summary = [
        {**char, **dict(zip(FEATURES, row)), "training_fonts": {
            name: dict(zip(FEATURES, all_features[f, c])) for f, name in enumerate(FONT_SPLIT["train"])
        }}
        for c, (char, row) in enumerate(zip(chars, median))
    ]
    return median, summary


def fit_ridge(x, groups, y, x_eval, groups_eval):
    """Eliminate the unpenalized intercepts by centering within each stroke group."""
    scale = np.std(x, axis=0)
    vary = np.ptp(x, axis=0) != 0
    center = x.mean(axis=0)
    z = np.divide(x - center, scale, out=np.zeros_like(x), where=vary)
    z_eval = np.divide(x_eval - center, scale, out=np.zeros_like(x_eval), where=vary)
    z_residual = z.copy()
    y_residual = y.copy()
    z_means, y_means = [], []
    for group in STROKES:
        members = groups == group
        z_means.append(z[members].mean(axis=0))
        y_means.append(y[members].mean())
        z_residual[members] -= z_means[-1]
        y_residual[members] -= y_means[-1]
    slopes = np.linalg.solve(z_residual.T @ z_residual + np.eye(x.shape[1]), z_residual.T @ y_residual)
    intercepts = y_means - np.asarray(z_means) @ slopes
    get_group = np.searchsorted(STROKES, groups_eval)
    return intercepts[get_group] + z_eval @ slopes, intercepts, slopes


def held_character(x, groups, target):
    preds = np.empty((len(target), 2))
    for held in range(len(target)):
        train = np.arange(len(target)) != held
        for model, cols in enumerate((2, 3)):
            fit, _, _ = fit_ridge(x[train, :cols], groups[train], target[train],
                                  x[held:held+1, :cols], groups[held:held+1])
            preds[held, model] = fit[0]
    return preds, np.abs(preds - target[:, None])


def compute_primary(dices, x, groups, font_ids):
    cases = []
    sensitivity = np.zeros(CLASSES)
    for w, width in enumerate((16, 32)):
        for f, font_id in enumerate(font_ids):
            target = dices[2*w:2*w+2, :, f].mean(axis=(0, 1, 3))
            preds, errors = held_character(x, groups, target)
            baseline, entropy = errors.mean(axis=0)
            cases.append({
                "font_id": font_id, "width": width, "mae_base": baseline,
                "mae_entropy": entropy, "delta_mae": baseline - entropy,
                "characters": [
                    {"class_id": c, "mean_dice": target[c], "prediction_base": p[0],
                     "prediction_entropy": p[1], "error_base": e[0],
                     "error_entropy": e[1], "delta_absolute_error": e[0] - e[1]}
                    for c, (p, e) in enumerate(zip(preds, errors))
                ],
            })
            for removed in range(CLASSES):
                keep = np.arange(CLASSES) != removed
                _, err = held_character(x[keep], groups[keep], target[keep])
                sensitivity[removed] += (err[:, 0] - err[:, 1]).mean()
    sensitivity /= len(cases)
    return {
        "delta_mae": float(np.mean([case["delta_mae"] for case in cases])),
        "delete_one_min": float(sensitivity.min()), "delete_one_max": float(sensitivity.max()),
        "delete_one": [{"removed_class_id": c, "delta_mae": delta} for c, delta in enumerate(sensitivity)],
        "conditions": cases,
    }


def compute_width(dices, x, groups, font_ids, steps):
    delta = dices[2:4] - dices[:2]
    target = delta.mean(axis=(0, 1, 2, 4))
    fitted, intercepts, slopes = fit_ridge(x, groups, target, x, groups)
    return {
        "mean_delta_dice": float(target.mean()), "entropy_coefficient": float(slopes[-1]),
        "group_intercepts": dict(zip(map(str, STROKES), intercepts)),
        "standardized_coefficients": dict(zip(FEATURES, slopes)),
        "characters": [
            {"class_id": c, "mean_delta_dice": target[c], "fitted_delta_dice": fitted[c]} for c in range(CLASSES)
        ],
        "conditions": [
            {"font_id": name, "seed": seed, **step, "mean_delta_dice": delta[s, t, f].mean()}
            for (s, seed), (t, step), (f, name) in product(
                enumerate((101, 202)), enumerate(steps), enumerate(font_ids)
            )
        ],
    }


def compute_crossed(predicted, reference, font_ids, steps):
    if len(font_ids) != 2:
        return None, []
    a = np.array([True, True, False])
    b = np.array([True, False, True])
    assert_close(source_contrast(a, a, a, b), 0.0, "Toy equal-prediction contrast")
    assert_close(source_contrast(a, b, a, b), 1.0 - dice(a, b), "Toy exact-recovery contrast")
    contrast = source_contrast(predicted[:, :, 0], predicted[:, :, 1],
                               reference[0, :, None], reference[1, :, None])
    require(np.count_nonzero(contrast[METHODS.index(LABEL)]) == 0, "Label-only crossed contrast is nonzero")
    char_mean = contrast.mean(axis=-1)
    by_method = char_mean.mean(axis=-1)
    summary = {
        "font_a": font_ids[0], "font_b": font_ids[1],
        "conditions": [
            {**describe(method), **step, "mean_contrast": by_method[m, t],
             "character_mean_contrasts": char_mean[m, t].tolist()}
            for (m, method), (t, step) in product(enumerate(METHODS), enumerate(steps))
        ],
        "endpoints": [
            {**describe(method), "contrast_at_minus_5": row[0], "contrast_at_0": row[-1],
             "contrast_delta_0_minus_minus_5": row[-1] - row[0]}
            for method, row in zip(METHODS, by_method)
        ],
    }
    groups = (
        (("denoiser", 16, None), by_method[:2].mean(axis=0)),
        (("denoiser", 32, None), by_method[2:4].mean(axis=0)),
        (LABEL, by_method[4]),
        (NEAREST, by_method[5]),
    )
    audit = [
        {**describe(method), "mean_contrasts": row.tolist(),
         "contrast_delta_0_minus_minus_5": float(row[-1] - row[0])}
        for method, row in groups
    ]
    return summary, audit


def check_baselines_and_inputs(replay, panel, clean, nearest_fonts, steps, predicted, noise_seed):
    noise = replay["noise"]
    expected_shape = (len(clean), CLASSES, DRAWS, SIDE, SIDE)
    require(noise.dtype == np.float32 and noise.shape == expected_shape, "Invalid Gaussian noise bank shape or type")
    expected_noise = torch.randn(
        (*expected_shape[:3], 1, SIDE, SIDE),
        generator=torch.Generator(device="cpu").manual_seed(noise_seed),
    ).squeeze(3).numpy()
    noise_diff = assert_close(noise, expected_noise, "Seeded Gaussian noise bank", atol=2e-6)
    example_shape = (len(TARGETS), *clean.shape)
    examples = replay["example_predictions"]
    inputs = replay["example_noisy"]
    require(inputs.dtype == np.float32 and inputs.shape == example_shape, "Invalid noisy examples")
    require(examples.dtype == np.float32 and examples.shape == (len(METHODS), *example_shape)
            and np.isfinite(examples).all(), "Invalid clean prediction examples")
    example_signs = examples.reshape(*predicted.shape[:-2], PIXELS) < 0
    require(np.array_equal(example_signs, predicted[..., 0, :]), "Saved example signs disagree with packed masks")
    template_order = [FONT_SPLIT["train"].index(name) for name in sorted(FONT_SPLIT["train"])]
    gray_templates = panel[template_order].reshape(3, CLASSES, PIXELS).transpose(1, 0, 2)
    templates = gray_templates / 127.5 - 1.0
    clean_scaled = clean.reshape(len(clean), CLASSES, 1, PIXELS) / 127.5 - 1.0
    noise_pixels = noise.reshape(len(clean), CLASSES, DRAWS, PIXELS)
    label_mask = gray_templates.sum(axis=1) <= 382
    label_masks = predicted[METHODS.index(LABEL)]
    require(np.all(label_masks == label_mask[None, None, :, None, :]), "Label-only predictions differ from mean template")
    label_x0 = templates.mean(axis=1).reshape(CLASSES, SIDE, SIDE)
    label_examples = examples[METHODS.index(LABEL)]
    assert_close(label_examples, np.broadcast_to(label_x0, label_examples.shape), "Saved label-only examples", atol=2e-6)

    max_input_diff = 0.0
    for t, step in enumerate(steps):
        strength = math.sqrt(step["alpha_bar"])
        noise_scale = math.sqrt(1 - step["alpha_bar"])
        noisy = strength * clean_scaled + noise_scale * noise_pixels
        example_noisy = noisy[..., 0, :].reshape(clean.shape)
        max_input_diff = max(max_input_diff, assert_close(
            inputs[t], example_noisy, f"Noisy examples at SNR {TARGETS[t]}", atol=2e-6
        ))
        # The noisy-image squared norm cancels between templates.
        objective = strength**2 * np.square(templates).sum(axis=-1)[None, :, None, :]
        distances = objective - 2 * strength * np.einsum("fcdi,cki->fcdk", noisy, templates)
        selected = distances.argmin(axis=-1)
        require(np.array_equal(selected, nearest_fonts[t]), f"Nearest template IDs differ at SNR {TARGETS[t]}")
        selected_templates = templates[np.arange(CLASSES)[None, :, None], selected]
        require(np.array_equal(selected_templates < 0, predicted[METHODS.index(NEAREST), t]),
                f"Nearest-template masks differ at SNR {TARGETS[t]}")
        selected_examples = selected_templates[..., 0, :].reshape(clean.shape)
        assert_close(examples[METHODS.index(NEAREST), t], selected_examples,
                     f"Saved nearest-template examples at SNR {TARGETS[t]}", atol=2e-6)
    return {"noise_bank_max_abs_diff": noise_diff, "noisy_examples_max_abs_diff": max_input_diff}


@torch.inference_mode()
def replay_sample(models, clean, replay, predicted, steps):
    # Four glyphs and two draw indices per timestep, across three SNRs and each held-out font.
    sample = [
        (t, f, c, d) for t, f, c, d in product((0, 3, 5), range(len(clean)), (0, 7, 14, 23), (0, 11))
    ]
    inputs, signals, noise_scales = [], [], []
    for t, f, c, d in sample:
        alpha = steps[t]["alpha_bar"]
        gray = clean[f, c] / 127.5 - 1.0
        noise = replay["noise"][f, c, d]
        signals.append(math.sqrt(alpha))
        noise_scales.append(math.sqrt(1.0 - alpha))
        inputs.append(signals[-1] * gray + noise_scales[-1] * noise.astype(np.float64))
    noisy = np.asarray(inputs)
    tensor_in = torch.tensor(noisy[:, None], dtype=torch.float32)
    times = torch.tensor([steps[t]["timestep"] for t, _, _, _ in sample])
    classes = torch.tensor([c for _, _, c, _ in sample])
    signals = np.asarray(signals)[:, None, None]
    noise_scales = np.asarray(noise_scales)[:, None, None]
    max_example_diff = max_flipped_magnitude = 0.0
    mask_flips = 0
    flip_locations = []
    for m, model in enumerate(models):
        epsilon = model(tensor_in, times, classes).numpy()[:, 0]
        x0 = (noisy - noise_scales * epsilon) / signals
        require(np.isfinite(x0).all(), f"{NEURAL[m]}: nonfinite CPU replay")
        for i, (t, f, c, d) in enumerate(sample):
            description = f"{NEURAL[m]}, SNR {TARGETS[t]}, font index {f}, class {c}, draw {d}"
            if d == 0:
                stored = replay["example_predictions"][m, t, f, c]
                diff = assert_close(x0[i], stored, description, atol=CPU_X0_ATOL)
                max_example_diff = max(max_example_diff, diff)
            flipped = (x0[i].ravel() < 0) != predicted[m, t, f, c, d]
            if flipped.any():
                max_value = float(np.abs(x0[i].ravel()[flipped]).max())
                require(max_value <= CPU_X0_ATOL, f"{description}: CPU mask flip outside threshold tolerance")
                max_flipped_magnitude = max(max_flipped_magnitude, max_value)
                mask_flips += int(flipped.sum())
                if len(flip_locations) < 8:
                    flip_locations.append({**describe(NEURAL[m]), "target_log2_snr": TARGETS[t],
                                           "font_id": str(replay["font_ids"][f]), "class_id": c,
                                           "noise_id": d, "flipped_pixels": int(flipped.sum())})
    return {
        "checkpoint_samples": len(models) * len(sample),
        "pixels_checked": len(models) * len(sample) * PIXELS,
        "saved_example_max_abs_x0_diff": max_example_diff,
        "threshold_tolerance": CPU_X0_ATOL,
        "near_threshold_mask_flips": mask_flips,
        "max_abs_x0_at_flip": max_flipped_magnitude,
        "first_replays_with_mask_flips": flip_locations,
    }


def check_artifacts(directory, scores_file, summary_hashes):
    replay_name = "replay.npz"
    original = "scores.csv"
    compressed = original + ".gz"
    replay_hash = sha256(directory / replay_name)
    raw_score_hash = sha256(scores_file, decompress=scores_file.suffix == ".gz")
    require(replay_hash == summary_hashes[replay_name], "replay.npz SHA-256 differs from summary")
    if original in summary_hashes:
        require(raw_score_hash == summary_hashes[original], "Uncompressed score SHA-256 differs from summary")
    compressed_file = directory / compressed
    hashes = {replay_name: replay_hash, original: raw_score_hash}
    if compressed_file.exists():
        require(sha256(compressed_file, decompress=True) == raw_score_hash, "Plain and compressed scores differ")
        hashes[compressed] = sha256(compressed_file)
    if compressed in summary_hashes:
        require(hashes.get(compressed) == summary_hashes[compressed], "Compressed score SHA-256 differs from summary")
    require(original in summary_hashes or compressed in summary_hashes, "Missing score SHA-256 in summary")
    return hashes


def audit(data_dir, checkpoint_dir, evaluation):
    torch.set_num_threads(4)
    panel, chars, groups, identities, cross_split_pairs = read_panel(data_dir)
    replay, split, font_ids, clean, reference = read_replay(evaluation, panel)
    steps = schedule_steps()
    models, entries = checkpoint_models(checkpoint_dir, identities["dataset_sha256"], split)
    score_file, mask_rows, stored, nearest_fonts, actual_snrs = get_scores(evaluation, font_ids, chars, steps, entries)
    flat_masks = unpack(replay["predicted_masks_packed"], (mask_rows.size,))
    predicted = flat_masks[mask_rows]
    dices = dice(predicted, reference[:, :, None, :])
    max_dice_diff = assert_close(stored, dices, "Per-mask foreground Dice", atol=0.0)
    noise_seed = NOISE_SEEDS[split]
    cpu = check_baselines_and_inputs(replay, panel, clean, nearest_fonts, steps, predicted, noise_seed)
    cpu.update(replay_sample(models, clean, replay, predicted, steps))
    noise_hash = hashlib.sha256(replay["noise"].astype("<f4", copy=False).tobytes()).hexdigest()

    x, feature_rows = train_features(panel, chars)
    primary = compute_primary(dices, x, groups, font_ids)
    width = compute_width(dices, x, groups, font_ids, steps)
    crossed, crossed_audit = compute_crossed(predicted, reference, font_ids, steps)
    denoising_curves = [
        {**describe(method), "font_id": name, **step, "mean_dice": dices[m, t, f].mean()}
        for (m, method), (t, step), (f, name) in product(
            enumerate(METHODS), enumerate(steps), enumerate(font_ids)
        )
    ]

    summary_path = evaluation / "summary.json"
    summary = json_read(summary_path)
    evaluator_hash = sha256(ROOT / "scripts/evaluate_glyph_entropy.py")
    n_neural = mask_rows[:len(NEURAL)].size
    n_baseline = mask_rows[len(NEURAL):].size
    expected_replay = {
        "mask_bitorder": "little", "mask_pixel_order": "row-major", "example_noise_id": 0,
        "example_condition_order": [describe(method) for method in METHODS],
    }
    comparable = {
        **identities, "evaluator_sha256": evaluator_hash,
        "schema_version": 1, "split": split, "image_size": SIDE, "classes": CLASSES,
        "eval_font_ids": list(font_ids), "training_font_ids": list(FONT_SPLIT["train"]),
        "template_font_ids": sorted(FONT_SPLIT["train"]), "noise_draws": DRAWS,
        "noise_seed": noise_seed, "noise_device": "cpu", "noise_sha256_float32_le": noise_hash,
        "schedule": SCHEDULE,
        "training_steps": entries[0]["config"]["training_steps"], "checkpoints": entries,
        "ridge_lambda": 1.0, "n_neural_rows": n_neural, "n_baseline_rows": n_baseline,
        "timesteps": steps, "replay": expected_replay, "features": feature_rows, "primary": primary,
        "paired_width": width, "crossed_source": crossed, "denoising_curves": denoising_curves,
    }
    verify_fields(summary, comparable, "summary")
    assert_close([s["actual_log2_snr"] for s in summary["timesteps"]], actual_snrs, "CSV/summary SNR", atol=0)
    hashes = check_artifacts(evaluation, score_file, summary["artifacts_sha256"])

    return {
        "status": "PASS",
        "split": split,
        "font_ids": list(font_ids),
        "cross_split_panel_pairs": cross_split_pairs,
        "score_rows": mask_rows.size,
        "neural_rows": n_neural,
        "baseline_rows": n_baseline,
        "max_abs_dice_diff": max_dice_diff,
        "training_renders_checked": CLASSES * len(FONT_SPLIT["train"]),
        "statistics_absolute_tolerance": STATS_ATOL,
        "primary": {**{k: v for k, v in primary.items() if k not in {"conditions", "delete_one"}},
                    "conditions": [{k: v for k, v in c.items() if k != "characters"} for c in primary["conditions"]]},
        "crossed_source": crossed_audit,
        "paired_width": {
            "mean_delta_dice": width["mean_delta_dice"], "entropy_coefficient": width["entropy_coefficient"],
            "mean_delta_by_font_and_snr": [
                {"font_id": name, "target_log2_snr": t, "mean_delta_dice": float(np.mean([
                    row["mean_delta_dice"] for row in width["conditions"]
                    if row["font_id"] == name and row["target_log2_snr"] == t
                ]))}
                for name, t in product(font_ids, TARGETS)
            ],
        },
        "cpu_replay": cpu,
        "artifact_sha256": {
            **hashes, **identities, "evaluator": evaluator_hash, "summary.json": sha256(summary_path),
            "runs.json": sha256(checkpoint_dir / "runs.json"),
            **{c["file"]: c["checkpoint_id"] for c in entries},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/study"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path for the JSON verdict")
    args = parser.parse_args()
    checkpoint_files = [f"w{width}-s{seed}.pt" for _, width, seed in NEURAL]
    protected = (
        *(args.data / name for name in ("manifest.json", "glyphs.npz")),
        *(args.checkpoint_dir / name for name in ("runs.json", *checkpoint_files)),
        *(args.evaluation / name for name in ("scores.csv", "scores.csv.gz", "replay.npz", "summary.json")),
    )
    if args.output.resolve() in {item.resolve() for item in protected}:
        parser.error("--output would overwrite an existing input file")
    try:
        result = audit(args.data, args.checkpoint_dir, args.evaluation)
    except Exception as exc:
        result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"[:400]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    console = {k: result[k] for k in ("status", "split", "score_rows", "error") if k in result}
    print(json.dumps(console, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
