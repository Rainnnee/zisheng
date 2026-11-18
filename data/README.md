# Data

The prepared 24-character study panel, source files and regeneration instructions are in [data/study](study/).

The first demo uses synthetic font renderings rather than a downloaded handwriting dataset. On first run, the package downloads two Noto CJK Simplified Chinese fonts and their license texts into `data/fonts/`. The font binaries and license files are ignored by Git.

The renderer creates a balanced set for 山, 水, and 永. It saves the source versions, checksums, seed, image dimensions, and rendering ranges in the run's `dataset_manifest.json`. See [`docs/data-card.md`](../docs/data-card.md) for the source and limitations.

The first demo uses its entire generated batch for pipeline development and training. The study panel described above uses an explicit held-out-font split. If a later project uses human handwriting, preserve writer IDs and split by writer wherever the source supports it.
