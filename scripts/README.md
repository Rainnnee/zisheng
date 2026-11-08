# Running the project

The executable interface is installed as the `zisheng` command. The demo uses Python, PyTorch, NumPy, and Pillow; see the root README for environment setup.

```bash
python -m zisheng demo --steps 4000 --device auto --open
```

This command downloads the documented fonts if needed, renders the pilot images, trains a model, and writes a local report with a sample grid, training trace, and reverse-diffusion animation. Outputs go under `results/runs/pilot/` by default.

To generate new samples from an existing checkpoint:

```bash
python -m zisheng sample --checkpoint demo/assets/zisheng-pilot.pt --output results/runs/new-samples --open
```

Training and sample seeds, font hashes, settings, and images are stored with each run. The `--device` option accepts `auto`, `mps`, `cuda`, or `cpu`.
