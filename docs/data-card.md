# Data card: rendered-font pilot

## Status

This is a synthetic development dataset for the expanded visual pilot. It is not handwriting and is not used to make a claim about recognition accuracy.

## Source and terms

The glyphs are rendered locally from two Simplified Chinese fonts in the [Noto CJK project](https://github.com/notofonts/noto-cjk):

- Noto Sans CJK SC Regular, release `Sans2.004` ([font file](https://github.com/notofonts/noto-cjk/blob/Sans2.004/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf), [license](https://github.com/notofonts/noto-cjk/blob/main/Sans/LICENSE)).
- Noto Serif CJK SC Regular, release `Serif2.003` ([font file](https://github.com/notofonts/noto-cjk/blob/Serif2.003/Serif/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf), [license](https://github.com/notofonts/noto-cjk/blob/main/Serif/LICENSE)).

Both font families are distributed under the SIL Open Font License 1.1. The setup script downloads the font files and license texts into `data/fonts/`; the `.otf` and `.txt` files are ignored by Git. The recorded font hashes were `2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b` for Noto Sans and `2a2eae2628df83556c54018c41e20fa532c1b862c5256ae8b3f23feb918d12ca` for Noto Serif. Each run's `dataset_manifest.json` also records the source URLs and hashes.

The initial three-character pilot contained 768 32×32 images. The expanded set contains 3,072 images: 12 characters × 2 fonts × 128 variants per font. Each character has 256 rendered examples. Images are grayscale, 48×48 pixels, with paper at +1 and ink at −1.

## Character selection

The expanded classes are 山 (3 strokes); 水, 火, and 心 (4); 永 (5); 安 (6); 志 (7); 花, 明, and 和 (8); 信 (9); and 笑 (10). Counts follow `kTotalStrokes` from Unicode 18.0.0 Unihan data, described in [Unicode Standard Annex #38](https://www.unicode.org/reports/tr38/). The set includes compact single-part glyphs and top-bottom and left-right structures, with meanings such as peace, aspiration, harmony, trust, and laughter. It broadens the visual examples but does not balance the number of characters at each stroke count, so it is still not a controlled stroke-complexity study.

## Rendering and variation

The renderer centers each glyph on a 48×48 canvas. It samples font size from 36–42 pixels, adds a translation of up to two pixels, rotates by up to five degrees, and varies stroke weight from 0 to 1 native pixel before downsampling from a four-times-larger canvas. The random seed and these settings are saved in the run manifest. These variations are intentional: a class label names the character, but the model receives no separate target for scale or line weight. Different random seeds therefore sample different appearances from this training distribution. These are transformations of font glyphs, not independent examples written by people.

## Split and limitations

The pilot uses all rendered images for optimization. It has no held-out font split, validation set, or test set. Both font families contribute examples for every class, and the evaluation does not measure whether the model can generate a style it has not seen. Font rendering gives known labels and reproducible samples, but only two families provide limited visual variation. The images also omit handwriting, paper texture, pen pressure, and writer-specific variation. Generated outputs are grayscale pixel arrays; the model does not represent characters as pen-stroke sequences or vector paths.

The font binaries and generated training arrays stay local. The committed sample checkpoint and figures are outputs of this synthetic pilot; they are not raw font files.
