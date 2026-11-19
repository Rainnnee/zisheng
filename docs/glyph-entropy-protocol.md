# Glyph entropy and source recovery protocol

Frozen design · 2026-11-18 · STATS 100A Honors

## Questions

The primary question is whether a character's training-font local pattern entropy reduces out-of-fold error when predicting its foreground Dice on two unseen typeface families. The secondary questions measure the paired change from a wider denoiser and whether reconstructions preserve the source rendering once the correct character label is available. The original generation-from-noise demo remains a separate qualitative experiment.

## Data and training

Use the verified, pinned `data/study/manifest.json` and `glyphs.npz`, with 24 simplified characters and six per 4-, 6-, 8-, and 10-stroke group. Each character has one centered 48×48 grayscale rendering per family. There are three training families, one validation family, and two test families. These are selected digital typefaces, including handwriting-inspired outlines. All 24 class IDs appear during training. Split decisions and glyph-coverage/duplicate checks use dataset provenance and clean renders before outcomes exist.

The training script augments only the three training fonts with fixed-seed translations and rotations and uses the full 1,000-step Gaussian schedule. The two widths share the example, timestep, and noise streams within each training seed. Use width 16 and 32, 101 and 202 as the two paired seeds, batch 24, the same update count, and the final EMA weights. A 128-update validation-only run verifies both widths and informs the feasible update budget. Holdout scores, overlays, and analysis are produced only after configuration, dataset, and scoring code are versioned.

## Measurements

- Foreground is gray value ≤127 in the clean image and `x̂₀<0` on the model's [-1, 1] scale. Foreground Dice is `2 |A∩B| / (|A|+|B|)`. Two empty masks give 1.
- Form each predictor from only the three canonical training-font renders. For each render, take the entropy of all complete 2×2 binary windows after removing both uniform patterns and renormalizing the 14 remaining bins. Ink fraction is the foreground fraction. Perimetric complexity is `L²/(4πA)`, where `L` is the four-connected digital perimeter and `A` is the foreground area. Use the median of each of these quantities over the three training renders.
- Select the nearest schedule indices for log₂ SNR of −5, −4, −3, −2, −1, and 0. Use the same predetermined bank of 16 Gaussian draws per font and character across the two model widths, training seeds, and SNR levels. A model predicts ε from the noisy image, step and known character ID. Compute a single clean estimate by rearranging the Gaussian forward equation.
- The label-only baseline uses the grayscale mean of the three training templates for that label. Gaussian nearest chooses the template minimizing `‖xₜ−sqrt(ᾱₜ) T‖²`. Resolve a tie by sorted font ID. Use exactly the same threshold and Dice for both baselines.

## Confirmatory descriptive comparisons

For each test font and width, average Dice within character over the six SNRs, two seeds and 16 noise draws. Predict this 24-character vector with held-one-character-out ridge regression. The control uses separate, unpenalized intercepts for each stroke group and standardized ink fraction and perimetric complexity. The expanded estimator adds entropy. Within each fold standardize continuous predictors using only the fit rows, with population standard deviation and a zero column if constant. The loss is summed squared error plus one times the squared norm of the slopes. The primary result is the equally weighted average of `MAE(control)−MAE(+entropy)` across the four font-by-width conditions. Report each condition and a descriptive range from deleting a character and refitting the entire procedure.

For width, match character, font, timestep, seed and noise draw before subtracting narrow from wide Dice. Average within character and fit the same expanded ridge specification once across the 24 values to describe the adjusted entropy slope for width gain. Report the means by font and SNR. This contrast matches sample exposure and updates, with different compute and parameter counts.

## Crossed-font source comparison

Fix this supplement before any test scoring. With test renderings `A` and `B` of the same character and their predicted foregrounds `P_A` and `P_B`, compute

```text
S = (Dice(P_A, A) − Dice(P_A, B) + Dice(P_B, B) − Dice(P_B, A)) / 2
```

Compute S at each SNR for both widths and both template baselines. Pair the two fonts by character, SNR, model seed, and noise draw index. Average noise draws and seeds within character, then weight characters equally. A fixed label-only prediction is exactly zero. The pre-evaluation prediction is that S at log₂ SNR 0 exceeds S at −5 for both widths. S measures source-mask agreement for these two renditions, including stroke position and weight. Gaussian nearest provides a training-template strategy that can also respond to the input. Use toy masks to check that identical predictions cancel and exact distinct reconstructions produce `1 − Dice(A,B)`.

The report illustration uses the preselected character 明, width 32, training seed 101 and noise draw 0 at the two SNR endpoints. The figure is a visual example. The statistics use the complete panel.

The analysis is conditional on this panel, split, architecture, and training budget. Treat the character as the unit for descriptive uncertainty. Reserve choices motivated by held-out scores for a clearly labeled exploratory analysis. Store schedule/seed, masks, row keys, code revision and checkpoint fingerprints so the result can be checked from original predictions.
