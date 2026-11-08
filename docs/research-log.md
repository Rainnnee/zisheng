# Research log

## 2026-10-13 — Planning and repository scaffold

- Drafted the project roadmap, learning plan, and data/experiment protocol in the planning chat.
- Created the repository scaffold and interface placeholders.
- No data were acquired; no model was implemented, trained, or evaluated.
- Next decision: confirm course requirements and select the initial data domain after source and license checks.

## 2026-10-26 — First faculty brief and corruption illustration

- Prepared a project-scope brief with the research question, proposed pilot, forward-process derivation, and questions for Professor Wu.
- Generated a reproducible schematic of Gaussian corruption applied to 山 rendered in macOS Songti SC Regular. The figure was an illustration, not handwriting, training data, model output, or an experimental result.
- No dataset or model was used for training at this stage.

## 2026-11-07 — Three-character conditional diffusion pilot

- Downloaded Noto Sans CJK SC Regular (`Sans2.004`) and Noto Serif CJK SC Regular (`Serif2.003`) from the Noto CJK project. Both font families are under the SIL Open Font License 1.1. Binary font files remain local; the run manifest records their source URLs and SHA-256 hashes.
- Rendered a balanced pilot set of 768 grayscale 32×32 images: 256 each for 山, 水, and 永. Stroke counts (3, 4, and 5) were checked against `kTotalStrokes` in Unicode 18.0.0 Unihan data.
- Implemented a compact conditional U-Net with sinusoidal timestep embeddings and learned character embeddings. The model predicts added Gaussian noise and is sampled through a 1,000-step linear DDPM schedule.
- Trained for 4,000 AdamW updates on Apple MPS using seed 42; used EMA weights for sampling. Training noise-prediction MSE went from 0.1505 at update 200 to 0.0167 at update 4,000. The sample run used seed 43.
- A 100-step linear schedule with the same beta endpoints left the final cumulative signal coefficient near 0.36. Starting that reverse chain from pure Gaussian noise produced poor samples, so the demo uses 1,000 steps, where the terminal signal is close to zero.
- The saved gallery was visually inspected: outputs included recognizable examples of all three labels, alongside malformed and overly thick glyphs. This is qualitative inspection only; no held-out-font, OCR, or human recognition evaluation has been completed.
- Next: confirm the main research question with Professor Wu, then design a held-out-font evaluation before expanding the character set.

## 2026-11-07 — Expanded 12-character visual pilot

- Added nine new classes to the original set, for 12 total: 山, 水, 火, 心, 永, 安, 志, 花, 明, 和, 信, 笑. Their Unicode 18.0.0 `kTotalStrokes` values range from 3 to 10 and are recorded in `metadata/characters.csv`.
- Increased image resolution from 32×32 to 48×48 to preserve more detail in denser glyphs. The balanced synthetic dataset contains 3,072 images: 12 characters × 2 fonts × 128 variants per font.
- Trained on Apple MPS for 8,000 AdamW updates using seed 42. Training noise-prediction MSE reached 0.0132. This is not a recognition score.
- The sample gallery contains readable examples across the classes, along with malformed and unusually heavy outputs. Inspection is qualitative; no held-out-font, OCR, or human recognition evaluation has been completed.
- Reworked the local page to stream 101 actual states per 1,000-step reverse chain and show each sample independently. Selecting a sample focuses the timeline on that sample instead of a preassembled image strip.
- The class label still has no scale or line-weight condition. The training renderer varies both, so different seeds can produce different apparent sizes and stroke weights. A controlled-size experiment would need normalized glyph bounds or explicit appearance conditions.
- Next: define a fixed recognition measure and held-out-font split before making claims about quality or stroke complexity.
