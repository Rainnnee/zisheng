"""Build paper figures and LaTeX values from the saved test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("denoiser", 16, "U-Net 16", "#23648F", "-", "o"),
    ("denoiser", 32, "U-Net 32", "#AF4E2D", "-", "s"),
    ("gaussian-nearest", None, "Gaussian nearest", "#177568", "--", "^"),
    ("label-only", None, "Label only", "#626974", ":", "D"),
)


def curve(rows, measure, method, width):
    return [mean(row[measure] for row in rows
                 if row["method"] == method and row["width"] == width
                 and row["target_log2_snr"] == target) for target in range(-5, 1)]


def draw_curves(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.85))
    fig.subplots_adjust(left=.08, right=.98, bottom=.19, top=.79, wspace=.32)
    dice_rows = summary["denoising_curves"]
    source_rows = summary["crossed_source"]["conditions"]
    for method, width, label, color, style, marker in METHODS:
        for axis, rows, measure in zip(axes, (dice_rows, source_rows), ("mean_dice", "mean_contrast")):
            axis.plot(range(-5, 1), curve(rows, measure, method, width), label=label,
                      color=color, linestyle=style, marker=marker, markersize=3.7, linewidth=1.65)
    for axis, title, label in zip(
        axes, ("A  Mask recovery", "B  Source preference"), ("Foreground Dice", "Crossed-font contrast S")
    ):
        axis.set_title(title, loc="left", fontweight="bold", fontsize=10)
        axis.set_ylabel(label, fontsize=9)
        axis.set_xlabel("Target log$_2$ SNR", fontsize=9)
        axis.set_xticks(range(-5, 1))
        axis.tick_params(labelsize=8)
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
    axes[1].set_ylim(bottom=min(-.002, axes[1].get_ylim()[0]))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(.53, .99),
               frameon=False, fontsize=8)
    fig.savefig(output / "curves.pdf")
    fig.savefig(output / "curves.png", dpi=200)
    plt.close(fig)


def draw_example(summary, manifest, replay, output):
    class_id = next(row["class_id"] for row in manifest["characters"] if row["glyph"] == "明")
    families = {row["font_id"]: row["family"] for row in manifest["fonts"]}
    font_ids = summary["eval_font_ids"]
    order = summary["replay"]["example_condition_order"]
    wide = order.index({"method": "denoiser", "width": 32, "seed": 101})
    label = order.index({"method": "label-only", "width": None, "seed": None})
    clean = replay["clean_uint8"][:, class_id].astype(np.float32) / 127.5 - 1.0
    noisy = replay["example_noisy"][:, :, class_id]
    predictions = replay["example_predictions"][:, :, :, class_id]
    images = (clean, noisy[0], predictions[wide, 0], noisy[-1], predictions[wide, -1], predictions[label, 0])
    fig, axes = plt.subplots(2, 6, figsize=(7.25, 2.35))
    fig.subplots_adjust(left=.15, right=.99, top=.83, bottom=.04, wspace=.10, hspace=.13)
    titles = ("Clean", "Input (−5)", "Wide (−5)", "Input (0)", "Wide (0)", "Label only")
    for c, (column, title) in enumerate(zip(images, titles)):
        for f, glyph in enumerate(column):
            axis = axes[f, c]
            axis.imshow(glyph, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#B6BCC4")
            if f == 0:
                axis.set_title(title, fontsize=8)
            if c == 0:
                axis.set_ylabel(families[font_ids[f]], rotation=0, va="center", ha="right",
                                fontsize=8, labelpad=5)
    fig.savefig(output / "illustration.png", dpi=240)
    plt.close(fig)


def write_tex(summary, manifest, output):
    values = {}
    primary = summary["primary"]
    width = summary["paired_width"]
    values["TrainSteps"] = f'{summary["training_steps"]:,}'.replace(",", "{,}")
    values["PrimaryDelta"] = f'{100 * primary["delta_mae"]:+.3f}'
    values["MeanControl"] = f'{100 * mean(row["mae_base"] for row in primary["conditions"]):.3f}'
    values["MeanExpanded"] = f'{100 * mean(row["mae_entropy"] for row in primary["conditions"]):.3f}'
    values["DeleteLow"] = f'{100 * primary["delete_one_min"]:+.3f}'
    values["DeleteHigh"] = f'{100 * primary["delete_one_max"]:+.3f}'
    values["MeanWidthGain"] = f'{100 * width["mean_delta_dice"]:+.3f}'
    values["WidthEntropySlope"] = f'{100 * width["entropy_coefficient"]:+.3f}'
    width_by_snr = {
        target: mean(row["mean_delta_dice"] for row in width["conditions"]
                     if row["target_log2_snr"] == target)
        for target in (-5, 0)
    }
    values["WidthLow"] = f"{100 * width_by_snr[-5]:+.3f}"
    values["WidthHigh"] = f"{100 * width_by_snr[0]:+.3f}"
    entropy_correlations = [row["spearman_rho"] for row in summary["training_entropy_spearman"]]
    values["EntropyRankLow"] = f"{min(entropy_correlations):.2f}"
    values["EntropyRankHigh"] = f"{max(entropy_correlations):.2f}"
    for alias, (method, model_width, *_unused) in zip(("Narrow", "Wide", "Nearest", "Label"), METHODS):
        for prefix, rows, measure in (
            ("Dice", summary["denoising_curves"], "mean_dice"),
            ("Source", summary["crossed_source"]["conditions"], "mean_contrast"),
        ):
            series = curve(rows, measure, method, model_width)
            values[f"{prefix}{alias}Low"] = f"{100 * series[0]:.2f}"
            values[f"{prefix}{alias}High"] = f"{100 * series[-1]:.2f}"
    lines = [r"\newcommand{\%s}{%s}" % (key, value) for key, value in values.items()]
    families = {row["font_id"]: row["family"] for row in manifest["fonts"]}
    for row in primary["conditions"]:
        name = families[row["font_id"]].replace("&", r"\&")
        lines.append(
            f'{name} & {row["width"]} & {100 * row["mae_base"]:.3f} '
            f'& {100 * row["mae_entropy"]:.3f} & {100 * row["delta_mae"]:+.3f} ' + r"\\"
        )
    split = len(values)
    (output / "numbers.tex").write_text("\n".join(lines[:split]) + "\n", encoding="utf-8")
    table = "\n".join(lines[split:])
    (output / "primary-rows.tex").write_text(
        "\\newcommand{\\PrimaryRows}{%\n" + table + "\n}\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=Path("results/glyph-entropy-v1"))
    parser.add_argument("--manifest", type=Path, default=Path("data/study/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("report/glyph-entropy/generated"))
    args = parser.parse_args()
    summary = json.loads((args.evaluation / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if summary["split"] != "test":
        parser.error("paper figures require the frozen two-font test evaluation")
    args.output.mkdir(parents=True, exist_ok=True)
    draw_curves(summary, args.output)
    with np.load(args.evaluation / "replay.npz", allow_pickle=False) as replay:
        draw_example(summary, manifest, replay, args.output)
    write_tex(summary, manifest, args.output)


if __name__ == "__main__":
    main()
