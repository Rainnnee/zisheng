# Glyph entropy study data

This fixed convenience panel contains 24 common characters used in simplified Chinese, with six at each of 4, 6, 8, and 10 strokes. It contains one centered grayscale rendering per character in each of six independently designed digital font families. The split holds out families while keeping character labels fixed.

| Split | Font family | Design source |
| --- | --- | --- |
| Train | Noto Sans CJK SC | Adobe and Google’s Source Han Sans project |
| Train | ZCOOL XiaoWei | Li Dawei and the Zuozi team |
| Train | Ma Shan Zheng | Ma Shan Zheng’s brush-style design |
| Validation | ZCOOL QingKe HuangYou | Zheng Qingke’s rounded display design |
| Test | ZCOOL KuaiLe | Display design developed by Liu Bingke’s workshop and typography apprentices |
| Test | Long Cang | Chen Xiaomin’s calligraphic design |

Adobe confirms that Source Han Sans and Noto Sans CJK are releases of the same design. The panel includes one representative of that lineage. XiaoWei, QingKe HuangYou, and KuaiLe share a ZCOOL release initiative. Each has a separate design team, project, and documented design origin. The WenKai candidate was removed during provenance checks because its upstream acknowledgments cite contributions from both Source Han Sans and Source Han Serif. The Liu Jian Mao Cao candidate was also checked for coverage and set aside to include both a display design and a calligraphic design in the test split. These decisions were based on provenance, coverage, and typeface style before model evaluation. [Adobe’s Source Han Sans account](https://blog.adobe.com/en/publish/2021/04/08/source-han-sans-goes-variable) · [WenKai’s pinned acknowledgments](https://github.com/lxgw/LxgwWenKai/blob/4becfb4d5cf0c92cfdb60a6e9b504bda43315325/README.md#致谢) · [Pinned font and license sources](sources.json)

The character shortlist began with the prior study inventory and added familiar characters until the target stroke groups were full. Unicode 17.0.0 assigns 花 seven strokes, so 国 replaces it in the eight-stroke group. 山, 永, 志, and 信 from the prior inventory fall outside the target counts. Inclusion required the intended single Unicode scalar, the pinned stroke value, a usable glyph in every font, and a clean raster. For a coverage or duplication failure, the study would resolve the issue and rebuild the full panel before training. [Unicode 17.0.0 Unihan](https://www.unicode.org/Public/17.0.0/ucd/Unihan.zip) · [Unicode’s stroke count definition](https://www.unicode.org/reports/tr38/tr38-39.html#kTotalStrokes)

## Rebuild and verify

From the repository root, using Python 3.12 in an environment for this project, run

```sh
python -m pip install 'Pillow==13.0.0' 'fonttools==4.68.0' 'numpy==2.5.4'
python scripts/build_glyph_entropy_data.py --verify
```

Omit `--verify` to regenerate the outputs. The builder downloads the pinned font binaries, license texts, and Unihan archive into the ignored `data/raw/glyph-entropy/` cache and checks every input against its locked SHA-256. Use `--cache-dir` to check fresh downloads and `--output-dir` to generate into a separate directory. The frozen bundle was built and verified with Python 3.12.14, Pillow 13.0.0, and FreeType 2.14.3. FreeType differences can change rasters, so use the recorded renderer versions for byte-for-byte verification.

`glyphs.npz` contains a single `glyphs` array of shape `(6, 24, 48, 48)` and dtype `uint8`, with axes ordered by the font and character arrays in [manifest.json](manifest.json). Ink is 0 and paper is 255. The builder renders each glyph separately with FreeType at 192 px, crops to the nonzero ink bounding box, uses Lanczos to scale the longest side to 40 px with aspect ratio preserved, and centers the result on the 48 px canvas. It uses the frozen foreground threshold of grayscale 127 or lower. The archive SHA-256 is `a0f84add514c1f9088c9648d369436b675d1badee4a263379255258d300693dc`.

## Verification

- [characters.csv](characters.csv) records the class ID, codepoint, and stroke source for all 24 labels.
- [fonts.csv](fonts.csv) records fonts, lineage, split, versions, download locations, checksums, and OFL 1.1 licensing.
- [coverage.csv](coverage.csv) records the Unicode cmap glyph name and ID and the grayscale and foreground-mask hashes for all 144 combinations.
- [quality_checks.json](quality_checks.json) reports 264 cross-split same-label comparisons, zero exact grayscale or foreground-mask repeats, and zero fallback, empty, or border-touching rasters. All 144 raster tiles were also inspected together for consistent positioning and visible ink.
- A separate outline check compared the 264 same-label cross-split pairs after decomposing components and normalizing translation and font units. It found zero identical pen-command sequences. This check detects exact path reuse in the same command representation.
- A from-empty-cache rebuild matched all generated files byte for byte.
