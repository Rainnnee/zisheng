# Data

The first demo uses synthetic font renderings rather than a downloaded handwriting dataset. On first run, the package downloads two Noto CJK Simplified Chinese fonts and their license texts into `data/fonts/`. The font binaries and license files are ignored by Git.

The renderer creates a balanced set for 山, 水, and 永. It saves the source versions, checksums, seed, image dimensions, and rendering ranges in the run's `dataset_manifest.json`. See [`docs/data-card.md`](../docs/data-card.md) for the source and limitations.

The current split is for pipeline development only: all generated images are used for training, and no held-out-font evaluation is run. Font-rendered variants must not be described as independent handwriting examples. If handwriting is considered later, preserve writer IDs and split by writer wherever the source supports it.
