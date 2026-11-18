# Local Edge Entropy and Source Recovery in Chinese Glyph Denoising

Short research report for UCLA STATS 100A Honors. [Read the PDF](paper.pdf) or open [the LaTeX source](paper.tex). The report uses the verified [test artifacts](../../results/glyph-entropy-v1/) and the [frozen protocol](../../docs/glyph-entropy-protocol.md).

To regenerate the figures and numbers from the repository root, install `.[artifacts]` and run `python scripts/render_glyph_entropy_report.py`. Build the paper with a standard LaTeX distribution and BibTeX:

```bash
mkdir -p results/runs/glyph-entropy-paper
latexmk -cd -pdf -outdir=../../results/runs/glyph-entropy-paper report/glyph-entropy/paper.tex
```

The generated charts can also be used in a website or research presentation.
