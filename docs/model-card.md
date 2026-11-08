# Model card: 字生 expanded visual pilot

## Purpose and status

This checkpoint is an educational prototype for inspecting a class-conditional DDPM on small printed glyphs. It is not intended for handwriting recognition, OCR, or general Chinese character generation.

## Architecture and conditioning

- Compact U-Net with 48×48, one-channel inputs and 12 learned character classes.
- Learned character-ID embedding plus sinusoidal timestep embedding.
- Predicts the Gaussian noise added to $x_0$; sampled with a 1,000-step DDPM reverse chain.
- Sampling uses an exponential moving average of model weights.

## Training data and settings

- 3,072 synthetic font renderings of 12 characters from 山 through 笑; 256 per class. The labels and stroke counts are listed in `metadata/characters.csv`.
- Two Noto CJK Simplified Chinese font families, both under SIL Open Font License 1.1; see `docs/data-card.md` for versions and hashes.
- Training seed 42; sample seed 43; batch size 64; 8,000 optimizer updates; AdamW at 0.0002; EMA decay 0.995.
- Recorded device: Apple MPS. The run configuration is stored with the checkpoint.

## Evaluation

Training noise-prediction MSE reached 0.0132 at update 8,000. This is an optimization trace, not an accuracy result. Samples are for visual inspection only; no held-out test, human study, OCR evaluation, or font-generalization measurement has been completed.

## Limitations

The model sees two font families and transformed copies of their glyphs. It may learn those specific shapes and does not establish generalization to handwriting or unseen fonts. The character class is the only appearance condition; scale and line weight vary in the training data and are not controllable at sampling time. Stroke-count groups are uneven and confounded with character identity and shape, so comparisons are descriptive only.

## Checkpoint

`demo/assets/expanded-12char-48px/zisheng-pilot.pt` is the expanded visual-pilot checkpoint. To generate new examples, run the `sample` command in the project README. Its configuration records the labels, training settings, and font-source hashes.
