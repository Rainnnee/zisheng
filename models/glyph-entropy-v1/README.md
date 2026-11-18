# Glyph entropy v1 checkpoints

Four compact EMA state dictionaries from the frozen 24-class study. Each loads with PyTorch `weights_only=True`; the saved config names the architecture, label count, training allocation and hashes. `runs.json` binds each filename to its SHA-256 and optimizer log.

| Width | Parameters | Training seeds | Updates per run |
|---|---:|---|---:|
| 16 | 240,993 | 101, 202 | 4,000 |
| 32 | 881,153 | 101, 202 | 4,000 |

These weights use the class IDs in the [study manifest](../../data/study/manifest.json). The [CPU evaluator](../../scripts/evaluate_glyph_entropy.py) loads the four weights together. The interactive pilot uses a separate 12-label checkpoint and label map.

Future architecture or data variants get a new study directory under `models/` and a matching directory under `results/`. Keep configs, data fingerprints and small reproducibility files beside the weights. As a future artifact grows, place it in a versioned release or artifact store and keep the verification manifest in Git.
