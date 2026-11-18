# Glyph entropy v1 results

The evaluation uses 24 known character labels in ZCOOL KuaiLe and Long Cang test renderings. It contains 18,432 model scores and 9,216 template scores. These are repeated measurements on the selected panel.

| File | Contents |
|---|---|
| `summary.json` | Metrics, per-character predictions, sensitivity analysis, configuration fingerprint and checkpoint hashes |
| `scores.csv.gz` | Score rows indexed by `mask_row`; the decompressed CSV hash is in `summary.json` |
| `replay.npz` | Clean rasters, CPU noise bank, packed foreground predictions and fixed visual examples |
| `audit.json` | Independent verification from the archive, scores and checkpoint weights |

In `replay.npz`, `predicted_masks_packed[mask_row]` is the score row’s image. Unpack with `np.unpackbits(row, bitorder="little").reshape(48, 48)`. The summary includes the full replay axis order and visualization metadata.

See the [evaluation protocol](../../docs/glyph-entropy-protocol.md) for estimands and the [paper](../../report/glyph-entropy/paper.pdf) for interpretation. Reproduce figures with `python scripts/render_glyph_entropy_report.py` from the repository root after installing the artifact dependencies.
