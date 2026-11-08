# Experiment protocol

## First implementation pilot

The first run checks whether the proposed pipeline can learn three class labels and produce visible samples. It is a code and scope demonstration, not the final experiment.

- **Input domain:** locally rendered Simplified Chinese print glyphs from the two font releases documented in `docs/data-card.md`.
- **Classes:** 山, 水, 永; 256 generated variants per class.
- **Representation:** 32×32 grayscale images scaled to $[-1,1]$.
- **Model:** compact U-Net with timestep and character-ID embeddings; output is predicted Gaussian noise.
- **Objective:** mean squared error between sampled and predicted noise.
- **Schedule:** 1,000 linear beta steps from 0.0001 to 0.02.
- **Optimization:** AdamW, learning rate 0.0002, batch size 64, 4,000 updates, EMA decay 0.995.
- **Seeds:** 42 for rendering and training; 43 for the recorded sample grid.
- **Hardware:** Apple MPS for the recorded run. The CLI also selects CUDA or CPU when available.

## Expanded visual pilot

The follow-up demo broadens the label set and image size so the interface can show more varied character shapes. It is still a qualitative pipeline check, not a test of the relationship between stroke count and recognizability.

- **Classes:** 山, 水, 火, 心, 永, 安, 志, 花, 明, 和, 信, 笑; Unicode 18.0.0 `kTotalStrokes` counts range from 3 to 10.
- **Representation:** 48×48 grayscale pixels, scaled to $[-1,1]$.
- **Data:** 256 rendered examples per class from two Noto CJK Simplified Chinese fonts; all fonts appear in training for all classes.
- **Variation:** font size, line weight, small translation, and rotation are randomized. Scale and line weight are not separate model controls.
- **Optimization:** AdamW at 0.0002, batch size 64, 8,000 updates, EMA decay 0.995; training seed 42.
- **Display:** the web interface streams one actual reverse state per 10 DDPM steps. Samples are separate images; selecting one shows only that sample's reverse path.

## What is recorded

The run stores the configuration, font names and hashes, dataset-generation settings, checkpoint, training noise-prediction MSE, sample grid, selected reverse-process states, and training examples. Outputs are written under `results/runs/` and are ignored by Git unless explicitly copied into `demo/assets/`.

The current visual check is qualitative. The loss chart shows training noise-prediction MSE only; it is not a held-out metric. No OCR classifier, human rating protocol, font holdout, or uncertainty estimate has been run.

## Requirements for the next experiment

Before making a claim about recognizability, define a fixed identity measure and evaluate generated samples separately from image quality. A held-out-font split is needed to test whether the model handles font styles absent from training. The split must be chosen before model fitting, with all variants from a font kept together. Use multiple training seeds and report per-character results.

The expanded set spans stroke counts 3 through 10, but the counts are unevenly represented and each character has its own shape. It is suitable for a broader visual demo, not a reliable estimate of stroke-complexity effects. Any future analysis should use more characters per stroke-count group, account for image density and structure, and avoid causal language unless the design supports it.
