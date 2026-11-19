# Chinese Character Generation and Denoising

A project report for UCLA STATS 100A Honors. It uses [the original generation of 花](generated/pilot-hua.png) to explain diffusion and [matched recoveries of 笑](generated/width-example.png) to show what happens at the two tested model widths. [Read the report](paper.pdf), [open the LaTeX source](paper.tex), or use the [Chinese learning notes](learning-notes.zh-CN.md) for worked examples and code. The analysis uses the verified [test artifacts](../../results/glyph-entropy-v1/) and the [frozen protocol](../../docs/glyph-entropy-protocol.md).

To regenerate the figures and numbers from the repository root, install `.[artifacts]` and run `python scripts/render_glyph_entropy_report.py`. Build the report with a standard LaTeX distribution and BibTeX.

```bash
mkdir -p results/runs/glyph-entropy-paper
latexmk -cd -pdf -outdir=../../results/runs/glyph-entropy-paper report/glyph-entropy/paper.tex
```

The generated charts can also be used in a website or research presentation.
