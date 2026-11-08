# Results and generated figures

The first trained model is a qualitative three-character pilot. Its sample grid, reverse-process animation, training trace, and checkpoint are kept in [`demo/assets/`](../demo/assets/). Full run outputs are written to `results/runs/` and ignored by Git.

The training curve reports noise-prediction MSE on training batches. It does not measure class recognition, image quality, or generalization. No held-out-font split, OCR evaluation, or human rating has been completed.

## Forward-corruption illustration

[`figures/forward_corruption_schematic.png`](figures/forward_corruption_schematic.png) remains a separate teaching figure. It renders 山 in the locally installed macOS Songti SC Regular font and adds one seeded Gaussian-noise draw. It is not a sample from the Noto training fonts or from the trained model.

Each run saves its own configuration, random seeds, font hashes, dataset manifest, checkpoint, training log, and visual outputs. Keep those records together when comparing later runs.
