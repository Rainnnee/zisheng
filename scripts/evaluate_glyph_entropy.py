"""Score paired single-step glyph recovery and run the fixed character analysis."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from zisheng.diffusion import GaussianDiffusion
from zisheng.model import ConditionalDenoiser
from zisheng.train import choose_device

WIDTHS = (16, 32)
SEEDS = (101, 202)
TARGET_SNRS = tuple(range(-5, 1))
STROKE_GROUPS = (4, 6, 8, 10)
NOISE_DRAWS = 16
NOISE_SEEDS = {"val": 48118, "test": 48117}
IMAGE_SIZE = 48
N_CLASSES = 24
LAMBDA = 1.0
SCHEDULE = {"timesteps": 1000, "beta_start": 0.0001, "beta_end": 0.02}
NEURAL = "denoiser"
LABEL = "label-only"
NEAREST = "gaussian-nearest"
BASELINES = ((LABEL, None, None), (NEAREST, None, None))
NEURAL_RUNS = tuple((NEURAL, w, s) for w, s in itertools.product(WIDTHS, SEEDS))
CONDITIONS = (*NEURAL_RUNS, *BASELINES)
FEATURE_NAMES = ("ink_fraction", "perimeter_complexity", "entropy")


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_glyphs(data_dir):
    manifest_file = data_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    archive = data_dir / "glyphs.npz"
    archive_hash = sha256_file(archive)
    if manifest["glyph_file_sha256"] != archive_hash:
        raise ValueError("The glyph archive SHA-256 differs from the manifest.")
    with np.load(archive, allow_pickle=False) as bundle:
        glyphs = bundle["glyphs"]
    fonts = manifest["fonts"]
    chars = manifest["characters"]
    split_counts = Counter(font["split"] for font in fonts)
    strokes = Counter(char["stroke_count"] for char in chars)
    if (
        manifest["image_size"] != IMAGE_SIZE
        or glyphs.shape != (6, N_CLASSES, IMAGE_SIZE, IMAGE_SIZE)
        or glyphs.dtype != np.uint8
        or [c["class_id"] for c in chars] != list(range(N_CLASSES))
        or len({c["glyph"] for c in chars}) != N_CLASSES
        or strokes != {group: 6 for group in STROKE_GROUPS}
        or split_counts != {"train": 3, "val": 1, "test": 2}
        or len({f["font_id"] for f in fonts}) != len(fonts)
    ):
        raise ValueError("The glyph bundle does not match the study's image, label, or split design.")
    return manifest, glyphs, archive_hash, sha256_file(manifest_file)


def load_checkpoints(directory, dataset_hash, train_font_ids):
    runs_file = directory / "runs.json"
    expected_hashes = None
    if runs_file.exists():
        index = json.loads(runs_file.read_text(encoding="utf-8"))
        expected_hashes = {run["file"]: run["sha256"] for run in index}
    models, checkpoints, configs = {}, [], []
    for key in NEURAL_RUNS:
        _, width, seed = key
        name = f"w{width}-s{seed}.pt"
        path = directory / name
        digest = sha256_file(path)
        if expected_hashes is not None and expected_hashes.get(name) != digest:
            raise ValueError(f"{name} is missing from or differs from runs.json.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        expected = {
            "base_channels": width,
            "seed": seed,
            "num_classes": N_CLASSES,
            "image_size": IMAGE_SIZE,
            "dataset_sha256": dataset_hash,
            **SCHEDULE,
        }
        for field, value in expected.items():
            if config.get(field) != value:
                raise ValueError(f"{name}: expected {field}={value!r}, got {config.get(field)!r}.")
        if "train_fonts" in config and config["train_fonts"] != train_font_ids:
            raise ValueError(f"{name}: training font IDs do not match the manifest.")
        if not isinstance(config["training_steps"], int) or config["training_steps"] < 1:
            raise ValueError(f"{name}: invalid training_steps.")
        if "training_log" in checkpoint:
            log = checkpoint["training_log"]
            if not log or log[-1]["update"] != config["training_steps"]:
                raise ValueError(f"{name}: training log does not reach the configured final update.")
        model = ConditionalDenoiser(num_classes=N_CLASSES, base_channels=width).eval()
        state = checkpoint["model_state_dict"]
        if any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise ValueError(f"{name}: checkpoint contains a nonfinite weight.")
        model.load_state_dict(state, strict=True)
        models[key] = model
        configs.append(config)
        checkpoints.append({
            "checkpoint_id": digest, "file": name, "config": config,
            "parameter_count": sum(value.numel() for value in model.parameters()),
        })

    random_fields = {"seed", "index_seed", "timestep_seed", "noise_seed"}
    shared = [{k: v for k, v in config.items() if k not in random_fields | {"base_channels"}}
              for config in configs]
    if any(config != shared[0] for config in shared[1:]):
        raise ValueError("The four checkpoints differ in their shared training settings.")
    for seed in SEEDS:
        streams = [{k: config[k] for k in random_fields if k in config}
                   for config in configs if config["seed"] == seed]
        if streams[0] != streams[1]:
            raise ValueError(f"Training random streams do not match for seed {seed}.")
    return models, checkpoints


def glyph_features(mask):
    """Return ink fraction, 4-neighbor perimeter complexity, and mixed-pattern entropy."""
    area = np.count_nonzero(mask)
    bits = mask.astype(np.uint8)
    patterns = (
        bits[:-1, :-1] + 2 * bits[:-1, 1:] + 4 * bits[1:, :-1] + 8 * bits[1:, 1:]
    )
    counts = np.bincount(patterns.ravel(), minlength=16)[1:15]
    counts = counts[counts > 0]
    if not area or not len(counts):
        raise ValueError("A training glyph has no ink or no mixed 2×2 windows.")
    p = counts / counts.sum()
    padded = np.pad(mask, 1)
    perimeter = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    perimeter += np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    return np.array([area / mask.size, perimeter**2 / (4 * np.pi * area), -p @ np.log2(p)])


def dice(predicted, reference):
    """Compute Dice over the last two axes with broadcasting over examples."""
    overlap = np.count_nonzero(predicted & reference, axis=(-2, -1))
    denominator = np.count_nonzero(predicted, axis=(-2, -1))
    denominator = denominator + np.count_nonzero(reference, axis=(-2, -1))
    return np.divide(2 * overlap, denominator, out=np.ones(overlap.shape), where=denominator != 0)


def crossed_source(predicted, clean):
    """Return per-class, paired-noise crossed Dice for two source fonts."""
    own_a = dice(predicted[0], clean[0, :, None])
    own_b = dice(predicted[1], clean[1, :, None])
    cross_a = dice(predicted[0], clean[1, :, None])
    cross_b = dice(predicted[1], clean[0, :, None])
    return 0.5 * ((own_a - cross_a) + (own_b - cross_b))


def gaussian_nearest(noisy, templates, alpha_bar):
    """Choose the first minimum per image. The templates are sorted by font ID."""
    squared_distance = (noisy[:, None] - alpha_bar.sqrt() * templates).square()
    return squared_distance.sum(dim=(-3, -2, -1)).argmin(dim=1)


def ridge(train_x, train_groups, y, predict_x, predict_groups):
    """Fit unpenalized group intercepts and standardized ridge slopes."""
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0, ddof=0)
    # np.std can be nonzero for a repeated non-integer constant because of summation.
    variable = np.ptp(train_x, axis=0) > 0

    def design(x, groups):
        z = np.divide(x - center, scale, out=np.zeros_like(x), where=variable)
        return np.column_stack((groups[:, None] == STROKE_GROUPS, z))

    lhs = design(train_x, train_groups)
    penalty = np.diag([0.0] * len(STROKE_GROUPS) + [np.sqrt(LAMBDA)] * train_x.shape[1])
    slopes = np.linalg.lstsq(
        np.vstack((lhs, penalty)), np.concatenate((y, np.zeros(len(penalty)))), rcond=None
    )[0]
    return design(predict_x, predict_groups) @ slopes, slopes


def loco(x, groups, y):
    predictions = np.empty((len(y), 2))
    for held in range(len(y)):
        train = np.arange(len(y)) != held
        for model, cols in enumerate((2, 3)):
            predictions[held, model] = ridge(
                x[train, :cols], groups[train], y[train],
                x[held:held + 1, :cols], groups[held:held + 1],
            )[0][0]
    errors = np.abs(predictions - y[:, None])
    return predictions, errors


def average_ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def describe(method, width, seed):
    return {"method": method, "width": width, "seed": seed}


def analyze(scores, crossed, training_features, chars, font_ids, train_font_ids, steps):
    """Summarize predeclared outcomes from matched arrays of raw Dice scores."""
    x = np.median(training_features, axis=0)
    groups = np.array([c["stroke_count"] for c in chars])
    features = [
        {**char, **dict(zip(FEATURE_NAMES, row)),
         "training_fonts": {
             font_id: dict(zip(FEATURE_NAMES, training_features[font_idx, class_id]))
             for font_idx, font_id in enumerate(train_font_ids)
         }}
        for class_id, (char, row) in enumerate(zip(chars, x))
    ]
    entropy_ranks = np.stack([average_ranks(values) for values in training_features[:, :, 2]])
    entropy_stability = [
        {"font_a": train_font_ids[i], "font_b": train_font_ids[j],
         "spearman_rho": float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) and np.ptp(b) else None}
        for (i, a), (j, b) in itertools.combinations(enumerate(entropy_ranks), 2)
    ]
    primary_conditions, holdout_targets, curves = [], [], []
    for width in WIDTHS:
        by_character = np.mean([scores[NEURAL, width, s] for s in SEEDS], axis=0).mean(axis=(0, 3))
        for font_id, y in zip(font_ids, by_character):
            predictions, errors = loco(x, groups, y)
            baseline_mae, full_mae = errors.mean(axis=0)
            primary_conditions.append({
                "font_id": font_id, "width": width, "mae_base": baseline_mae,
                "mae_entropy": full_mae, "delta_mae": baseline_mae - full_mae,
                "characters": [
                    {"class_id": c, "mean_dice": y[c], "prediction_base": predictions[c, 0],
                     "prediction_entropy": predictions[c, 1], "error_base": errors[c, 0],
                     "error_entropy": errors[c, 1], "delta_absolute_error": errors[c, 0] - errors[c, 1]}
                    for c in range(N_CLASSES)
                ],
            })
            holdout_targets.append(y)
    delete_one = []
    for c in range(N_CLASSES):
        keep = np.arange(N_CLASSES) != c
        delta = [loco(x[keep], groups[keep], y[keep])[1] @ np.array([1.0, -1.0])
                 for y in holdout_targets]
        delete_one.append({"removed_class_id": c, "delta_mae": float(np.mean(delta))})
    sensitivity = [row["delta_mae"] for row in delete_one]
    primary = {
        "delta_mae": float(np.mean([row["delta_mae"] for row in primary_conditions])),
        "conditions": primary_conditions,
        "delete_one": delete_one,
        "delete_one_min": min(sensitivity), "delete_one_max": max(sensitivity),
    }

    width_delta = np.stack([scores[NEURAL, 32, seed] - scores[NEURAL, 16, seed] for seed in SEEDS])
    # Axes are seed, SNR, font, class, and noise draw.
    y = width_delta.mean(axis=(0, 1, 2, 4))
    fitted, coefficients = ridge(x, groups, y, x, groups)
    paired_width = {
        "mean_delta_dice": float(y.mean()),
        "entropy_coefficient": coefficients[-1],
        "group_intercepts": dict(zip(map(str, STROKE_GROUPS), coefficients[:len(STROKE_GROUPS)])),
        "standardized_coefficients": dict(zip(FEATURE_NAMES, coefficients[len(STROKE_GROUPS):])),
        "characters": [
            {"class_id": c, "mean_delta_dice": y[c], "fitted_delta_dice": fitted[c]}
            for c in range(N_CLASSES)
        ],
        "conditions": [
            {"font_id": font_id, "seed": seed, **step,
             "mean_delta_dice": width_delta[seed_idx, step_idx, font_idx].mean()}
            for (seed_idx, seed), (step_idx, step), (font_idx, font_id) in itertools.product(
                enumerate(SEEDS), enumerate(steps), enumerate(font_ids)
            )
        ],
    }
    for key in CONDITIONS:
        for (step_idx, step), (font_idx, font_id) in itertools.product(
            enumerate(steps), enumerate(font_ids)
        ):
            curves.append({
                **describe(*key), "font_id": font_id, **step,
                "mean_dice": scores[key][step_idx, font_idx].mean(),
            })

    cross_summary = None
    if crossed:
        cross_summary = {
            "font_a": font_ids[0],
            "font_b": font_ids[1],
            "conditions": [
                {**describe(*key), **step, "mean_contrast": crossed[key][i].mean(),
                 "character_mean_contrasts": crossed[key][i].mean(axis=1).tolist()}
                for key in CONDITIONS for i, step in enumerate(steps)
            ],
            "endpoints": [
                {**describe(*key), "contrast_at_minus_5": crossed[key][0].mean(),
                 "contrast_at_0": crossed[key][-1].mean(),
                 "contrast_delta_0_minus_minus_5": (crossed[key][-1] - crossed[key][0]).mean()}
                for key in CONDITIONS
            ],
        }
    return {
        "features": features, "training_entropy_spearman": entropy_stability, "primary": primary,
        "paired_width": paired_width, "denoising_curves": curves, "crossed_source": cross_summary,
    }


def normalized(glyphs):
    return torch.from_numpy(glyphs).to(torch.float32).unsqueeze(-3) / 127.5 - 1.0


def pack_masks(masks):
    flat = masks.reshape(*masks.shape[:-2], IMAGE_SIZE**2)
    return np.packbits(flat, axis=-1, bitorder="little")


@torch.inference_mode()
def score(
    diffusion, models, train_glyphs, eval_glyphs, device, batch_size, noise_seed,
):
    # Draw on CPU once; each noise ID is reused at every timestep and for both widths and seeds.
    n_fonts = len(eval_glyphs)
    input_shape = (n_fonts, N_CLASSES, NOISE_DRAWS, 1, IMAGE_SIZE, IMAGE_SIZE)
    noise = torch.randn(
        input_shape, device="cpu", dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(noise_seed),
    )
    clean = normalized(eval_glyphs).unsqueeze(2).expand(input_shape).reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    templates = normalized(train_glyphs).transpose(0, 1)
    label_templates = templates.mean(dim=1)
    clean_masks = eval_glyphs <= 127
    ids = torch.arange(N_CLASSES)[None, :, None].expand(input_shape[:3]).reshape(-1)
    actual_snrs = torch.log2(diffusion.alpha_bars / (1.0 - diffusion.alpha_bars))
    indices = [int((actual_snrs - snr).abs().argmin()) for snr in TARGET_SNRS]
    if len(set(indices)) != len(TARGET_SNRS):
        raise ValueError("SNR selection did not produce six distinct indices.")
    steps = [
        {"target_log2_snr": snr, "timestep": index, "actual_log2_snr": float(actual_snrs[index]),
         "alpha_bar": float(diffusion.alpha_bars[index])}
        for snr, index in zip(TARGET_SNRS, indices)
    ]
    outcome_shape = (len(steps), *input_shape[:3])
    example_shape = (len(steps), *eval_glyphs.shape)
    scores = {key: np.empty(outcome_shape) for key in CONDITIONS}
    masks = {key: np.empty((*outcome_shape, IMAGE_SIZE**2 // 8), dtype=np.uint8)
             for key in CONDITIONS}
    examples = {key: np.empty(example_shape, dtype=np.float32) for key in CONDITIONS}
    crossed = ({key: np.empty((len(steps), N_CLASSES, NOISE_DRAWS)) for key in CONDITIONS}
               if n_fonts == 2 else {})
    noisy_examples = np.empty(example_shape, dtype=np.float32)
    nearest_ids = np.empty(outcome_shape, dtype=np.uint8)
    image_dims = (*input_shape[:3], IMAGE_SIZE, IMAGE_SIZE)
    device_ids = ids.to(device)
    for model in models.values():
        model.to(device)
    for i, index in enumerate(indices):
        print(f"Scoring target log2 SNR {TARGET_SNRS[i]} (t={index}) on {device}", flush=True)
        times = torch.full((len(ids),), index, dtype=torch.long)
        noisy = diffusion.q_sample(clean, times, noise.reshape_as(clean))
        alpha_bar = diffusion.alpha_bars[index]
        noisy_examples[i] = noisy.numpy().reshape(image_dims)[:, :, 0]
        selected = gaussian_nearest(noisy, templates[ids], alpha_bar)
        nearest_ids[i] = selected.numpy().reshape(input_shape[:3])
        device_times = times.to(device)
        device_noisy = noisy.to(device)
        signal, noise_scale = alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt()
        for key in CONDITIONS:
            method, _, _ = key
            if method == LABEL:
                prediction = label_templates[ids]
            elif method == NEAREST:
                prediction = templates[ids, selected]
            else:
                batches = []
                for start in range(0, len(ids), batch_size):
                    batch = slice(start, start + batch_size)
                    predicted_noise = models[key](device_noisy[batch], device_times[batch], device_ids[batch])
                    # Reuse the CPU forward input and compute x0 with the original CPU schedule.
                    batches.append((noisy[batch] - noise_scale * predicted_noise.cpu()) / signal)
                prediction = torch.cat(batches)
            if not torch.isfinite(prediction).all():
                raise ValueError(f"Nonfinite prediction for {key} at t={index}.")
            prediction = prediction.numpy().reshape(image_dims)
            examples[key][i] = prediction[:, :, 0]
            predicted_masks = prediction < 0
            scores[key][i] = dice(predicted_masks, clean_masks[:, :, None])
            masks[key][i] = pack_masks(predicted_masks)
            if crossed:
                crossed[key][i] = crossed_source(predicted_masks, clean_masks)
    if crossed and np.any(crossed[LABEL, None, None] != 0):
        raise RuntimeError("The deterministic crossed label-only contrast must be zero.")
    return steps, scores, masks, crossed, noise.numpy()[..., 0, :, :], noisy_examples, examples, nearest_ids


def save_outputs(output, summary, scores, masks, noise, noisy_examples, examples,
                 nearest_ids, eval_glyphs):
    output.mkdir(parents=True, exist_ok=True)
    font_ids = summary["eval_font_ids"]
    train_ids = summary["template_font_ids"]
    steps = summary["timesteps"]
    chars = summary["features"]
    checkpoints = {
        (c["config"]["base_channels"], c["config"]["seed"]): c["checkpoint_id"]
        for c in summary["checkpoints"]
    }
    fields = [
        "mask_row", "method", "font_id", "class_id", "glyph", "target_log2_snr",
        "timestep", "actual_log2_snr", "width", "seed", "noise_id", "checkpoint_id",
        "template_font_id", "dice",
    ]
    temporary = output / "scores.csv.part"
    row = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in CONDITIONS:
            method, width, seed = key
            for i, f, c, draw in np.ndindex(scores[key].shape):
                writer.writerow({
                    "mask_row": row, **describe(*key), "font_id": font_ids[f], "class_id": c,
                    "glyph": chars[c]["glyph"], "target_log2_snr": steps[i]["target_log2_snr"],
                    "timestep": steps[i]["timestep"], "actual_log2_snr": steps[i]["actual_log2_snr"],
                    "noise_id": draw, "checkpoint_id": checkpoints.get((width, seed)),
                    "template_font_id": train_ids[nearest_ids[i, f, c, draw]] if method == NEAREST else None,
                    "dice": scores[key][i, f, c, draw],
                })
                row += 1
    if row != summary["n_neural_rows"] + summary["n_baseline_rows"]:
        raise RuntimeError("The score count does not match the fixed design.")
    temporary.replace(output / "scores.csv")
    temporary = output / "replay.npz.part"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            noise=noise,
            predicted_masks_packed=np.concatenate([masks[k].reshape(-1, IMAGE_SIZE**2 // 8)
                                                   for k in CONDITIONS]),
            clean_masks_packed=pack_masks(eval_glyphs <= 127),
            clean_uint8=eval_glyphs,
            example_noisy=noisy_examples,
            example_predictions=np.stack([examples[key] for key in CONDITIONS]),
            font_ids=np.asarray(font_ids),
            class_ids=np.arange(N_CLASSES),
            noise_ids=np.arange(NOISE_DRAWS),
            target_log2_snr=np.asarray(TARGET_SNRS),
        )
    temporary.replace(output / "replay.npz")
    summary["artifacts_sha256"] = {
        name: sha256_file(output / name) for name in ("scores.csv", "replay.npz")
    }
    temporary = output / "summary.json.part"
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/study"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/runs/glyph-entropy-models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    torch.set_num_threads(4)
    manifest, glyphs, dataset_hash, manifest_hash = load_glyphs(args.data)
    fonts = manifest["fonts"]
    train_indices = [i for i, f in enumerate(fonts) if f["split"] == "train"]
    train_ids = [fonts[i]["font_id"] for i in train_indices]
    template_indices = sorted(train_indices, key=lambda i: fonts[i]["font_id"])
    template_ids = [fonts[i]["font_id"] for i in template_indices]
    eval_indices = [i for i, f in enumerate(fonts) if f["split"] == args.split]
    font_ids = [fonts[i]["font_id"] for i in eval_indices]
    models, checkpoints = load_checkpoints(args.checkpoint_dir, dataset_hash, train_ids)
    training_features = np.array([[glyph_features(image <= 127) for image in glyphs[idx]]
                                  for idx in train_indices])

    device = choose_device(args.device)
    diffusion = GaussianDiffusion(**SCHEDULE)
    eval_glyphs = glyphs[eval_indices]
    steps, scores, masks, crossed, noise, noisy_examples, examples, nearest_ids = score(
        diffusion, models, glyphs[template_indices], eval_glyphs, device, args.batch_size,
        NOISE_SEEDS[args.split],
    )
    n_scored = len(steps) * len(font_ids) * N_CLASSES * NOISE_DRAWS
    summary = {
        "schema_version": 1, "dataset_sha256": dataset_hash, "manifest_sha256": manifest_hash,
        "evaluator_sha256": sha256_file(Path(__file__)), "image_size": IMAGE_SIZE,
        "split": args.split, "eval_font_ids": font_ids, "training_font_ids": train_ids,
        "template_font_ids": template_ids, "classes": N_CLASSES, "noise_draws": NOISE_DRAWS,
        "noise_seed": NOISE_SEEDS[args.split], "noise_device": "cpu",
        "noise_sha256_float32_le": hashlib.sha256(noise.astype("<f4", copy=False).tobytes()).hexdigest(),
        "inference_device": str(device),
        "batch_size": args.batch_size, "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "schedule": SCHEDULE, "timesteps": steps, "checkpoints": checkpoints,
        "training_steps": checkpoints[0]["config"]["training_steps"],
        "ridge_lambda": LAMBDA,
        "n_neural_rows": n_scored * len(NEURAL_RUNS), "n_baseline_rows": n_scored * len(BASELINES),
        "replay": {
            "predicted_masks_packed": ["CSV mask_row", "packed pixels"],
            "clean_masks_packed": ["font", "class", "packed pixels"],
            "mask_bitorder": "little",
            "mask_pixel_order": "row-major",
            "noise": ["font", "class", "noise_id", "y", "x"],
            "clean_uint8": ["font", "class", "y", "x"],
            "example_noisy": ["SNR", "font", "class", "y", "x"],
            "example_predictions": ["condition", "SNR", "font", "class", "y", "x"],
            "example_condition_order": [describe(*key) for key in CONDITIONS],
            "example_noise_id": 0,
            "example_scale": "float32; unbounded x_t/x0, grayscale x=-1 black and x=1 white",
        },
        **analyze(scores, crossed, training_features, manifest["characters"], font_ids, train_ids, steps),
    }
    save_outputs(args.output, summary, scores, masks, noise, noisy_examples, examples,
                 nearest_ids, eval_glyphs)
    print(f"Wrote {summary['n_neural_rows']} model and {summary['n_baseline_rows']} baseline scores to {args.output}")


if __name__ == "__main__":
    main()
