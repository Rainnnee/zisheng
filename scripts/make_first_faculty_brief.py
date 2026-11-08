"""Create the first faculty scope brief and its font-rendered corruption figure.

This is a document-generation helper, not part of the diffusion implementation.
Install the optional ``artifacts`` dependencies, then run this file from any
directory. It writes a Markdown source, a one-page PDF, and a PNG figure.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
DATE_LABEL = "October 26, 2026"
T = 1_000
BETA_START = 1e-4
BETA_END = 0.02
NOISE_SEED = 2026
CHARACTER = "山"
CJK_FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
CJK_FONT_INDEX = 6
CJK_FONT_NAME = "Songti SC Regular"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _make_font_glyph(size: int = 64) -> list[list[float]]:
    """Render 山 from the installed Songti SC typeface and normalize to [-1, 1]."""
    if not CJK_FONT_PATH.exists():
        raise FileNotFoundError(
            f"Expected macOS system font not found: {CJK_FONT_PATH}. "
            "Install Songti SC or update CJK_FONT_PATH and CJK_FONT_INDEX."
        )
    font = ImageFont.truetype(str(CJK_FONT_PATH), 256, index=CJK_FONT_INDEX)
    bounds = font.getbbox(CHARACTER)
    glyph = Image.new("L", (bounds[2] - bounds[0], bounds[3] - bounds[1]), 0)
    ImageDraw.Draw(glyph).text((-bounds[0], -bounds[1]), CHARACTER, font=font, fill=255)

    max_side = size - 14
    scale = min(max_side / glyph.width, max_side / glyph.height)
    rendered_size = (max(1, round(glyph.width * scale)), max(1, round(glyph.height * scale)))
    glyph = glyph.resize(rendered_size, Image.Resampling.LANCZOS)
    patch = Image.new("L", (size, size), 255)
    glyph_ink = Image.eval(glyph, lambda value: 255 - value)
    patch.paste(glyph_ink, ((size - glyph.width) // 2, (size - glyph.height) // 2))
    return [[(patch.getpixel((x, y)) / 127.5) - 1.0 for x in range(size)] for y in range(size)]


def _schedule_cumulative_alphas() -> list[float]:
    betas = [BETA_START + i * (BETA_END - BETA_START) / (T - 1) for i in range(T)]
    cumulative = 1.0
    values: list[float] = []
    for beta in betas:
        cumulative *= 1.0 - beta
        values.append(cumulative)
    return values


def _to_display_byte(value: float) -> int:
    clipped = max(-1.0, min(1.0, value))
    return int(round((clipped + 1.0) * 127.5))


def _paint_figure(path: Path) -> dict[int, tuple[float, float]]:
    image_width, image_height = 1440, 540
    canvas = Image.new("RGB", (image_width, image_height), "#F6F4EF")
    draw = ImageDraw.Draw(canvas)
    dark, muted, accent = "#1D2B36", "#65727C", "#187D78"
    draw.text((46, 24), "FORWARD PROCESS  /  FONT RENDER  /  ILLUSTRATION ONLY", fill=accent, font=_font(16, bold=True))
    draw.text((46, 50), "How a font glyph gradually loses signal", fill=dark, font=_font(34, bold=True))
    draw.text(
        (46, 99),
        "x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps,     eps ~ N(0, I)",
        fill=dark,
        font=_font(19),
    )

    clean = _make_font_glyph()
    rng = random.Random(NOISE_SEED)
    epsilon = [[rng.gauss(0.0, 1.0) for _ in range(64)] for _ in range(64)]
    alpha_bars = _schedule_cumulative_alphas()
    images: list[tuple[str, str, list[list[float]]]] = [("Clean font glyph x_0", f"{CHARACTER} | {CJK_FONT_NAME}", clean)]
    alphas_used: dict[int, tuple[float, float]] = {}
    for timestep in (100, 400, 900):
        alpha_bar = alpha_bars[timestep - 1]
        snr = alpha_bar / (1.0 - alpha_bar)
        alphas_used[timestep] = (alpha_bar, snr)
        noisy = [
            [math.sqrt(alpha_bar) * clean[y][x] + math.sqrt(1.0 - alpha_bar) * epsilon[y][x] for x in range(64)]
            for y in range(64)
        ]
        images.append(
            (
                f"t = {timestep}",
                f"alpha_bar = {alpha_bar:.4f}  |  SNR = {snr:.3g}",
                noisy,
            )
        )

    card_width, card_height, gap, start_x, card_top = 324, 334, 22, 46, 148
    patch_size = 208
    for index, (heading, detail, values) in enumerate(images):
        x = start_x + index * (card_width + gap)
        draw.rounded_rectangle(
            (x, card_top, x + card_width, card_top + card_height), radius=18, fill="#FFFFFF", outline="#E1E0DA", width=2
        )
        draw.text((x + 20, card_top + 18), heading, fill=dark, font=_font(19, bold=True))
        detail_font = ImageFont.truetype(str(CJK_FONT_PATH), 14, index=CJK_FONT_INDEX) if index == 0 else _font(14)
        draw.text((x + 20, card_top + 48), detail, fill=muted, font=detail_font)
        patch = Image.new("L", (64, 64))
        patch.putdata([_to_display_byte(values[y][x0]) for y in range(64) for x0 in range(64)])
        patch = patch.resize((patch_size, patch_size), Image.Resampling.NEAREST).convert("RGB")
        canvas.paste(patch, (x + (card_width - patch_size) // 2, card_top + 82))
    draw.text(
        (46, 496),
        "One shared Gaussian draw; T=1,000 linear beta schedule. Panel schedule SNR = alpha_bar/(1-alpha_bar).",
        fill=dark,
        font=_font(15),
    )
    draw.text(
        (46, 520),
        f"Font: macOS {CJK_FONT_NAME}; synthetic print glyph, not handwriting or model output.",
        fill=muted,
        font=_font(14),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)
    return alphas_used


def _markdown_text(figure_relative_path: str) -> str:
    return f"""# First faculty brief: project scope and forward corruption

**STATS 100A Honors | Professor Wu | {DATE_LABEL}**

## Research question

Can a small class-conditional diffusion model generate recognizable images of selected Chinese characters? How do Gaussian corruption and character stroke count relate to recognizability?

## Proposed study

- **Data:** begin with a controlled rendered-font pilot. Consider isolated handwriting for the main study only if the source, labels, writer-level splits, coverage, and use conditions can be verified. Keep the main analysis within one data domain.
- **Model:** condition a small denoiser on a character ID and timestep. Start with a three-character pipeline pilot, then scale to at least nine characters if the data and training gates pass; 24 is a target, not a commitment.
- **Evaluation:** assess requested-character identity separately from image fidelity. Treat stroke-count patterns as associations within the selected set, not evidence that stroke count causes performance differences.

## The first mathematical object

For a fixed clean image $x_0$, the forward marginal is

$$x_t = \\sqrt{{\\bar{{\\alpha}}_t}}x_0 + \\sqrt{{1-\\bar{{\\alpha}}_t}}\\epsilon,\\qquad \\epsilon\\sim\\mathcal{{N}}(0,I).$$

Therefore, $x_t\\mid x_0\\sim\\mathcal{{N}}(\\sqrt{{\\bar{{\\alpha}}_t}}x_0,(1-\\bar{{\\alpha}}_t)I)$. The first formula constructs the corrupted image; the second states its conditional mean and covariance. This is the forward process, not the learned reverse process. See [Ho, Jain, and Abbeel (2020), Section 2, Equation 4](https://arxiv.org/abs/2006.11239).

![Gaussian corruption of 山 rendered in macOS Songti SC Regular]({figure_relative_path})

**Figure status:** a deterministic illustration using 山 rendered in the locally installed macOS Songti SC Regular font and one seeded Gaussian-noise realization. Panel SNR labels show the schedule ratio $\\bar{{\\alpha}}_t/(1-\\bar{{\\alpha}}_t)$, not measured image SNR. It is a synthetic print glyph, not handwriting, an acquired training sample, model output, or research result. Values are clipped to $[-1,1]$ for display.

## Current status and next step

The repository contains planning notes and interface placeholders only. No dataset has been acquired, and no model has been implemented, trained, or evaluated. For the actual dataset, next choose an openly licensed font, verify character coverage and terms, then produce a labeled contact sheet.

## Questions for Professor Wu

1. Is rendered-font generation an acceptable controlled main domain if handwriting access or use terms delay the project?
2. Which should lead the evaluation: generated-character recognition or corruption/restoration of held-out images?
3. What report format, final deadline, and level of mathematical derivation should I plan for?
"""


def _make_pdf(path: Path, figure_path: Path, alphas_used: dict[int, tuple[float, float]]) -> None:
    page_width, _ = letter
    margin = 39
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="KickerCustom", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=colors.HexColor("#187D78"), spaceAfter=4))
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23, leading=25, alignment=TA_LEFT, textColor=colors.HexColor("#1D2B36"), spaceAfter=3))
    styles.add(ParagraphStyle(name="MetaCustom", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=10, textColor=colors.HexColor("#65727C"), spaceAfter=7))
    styles.add(ParagraphStyle(name="SectionCustom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=colors.HexColor("#187D78"), spaceBefore=5, spaceAfter=2, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.0, leading=11.2, textColor=colors.HexColor("#26343F"), spaceAfter=4))
    styles.add(ParagraphStyle(name="SmallCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.0, leading=9.4, textColor=colors.HexColor("#65727C"), spaceAfter=2))
    styles.add(ParagraphStyle(name="FigureCaptionCustom", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.7, leading=9.0, textColor=colors.HexColor("#65727C"), alignment=TA_CENTER, spaceBefore=1, spaceAfter=4))

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=29,
        bottomMargin=29,
        title="First Faculty Brief: Project Scope and Forward Corruption",
        author="ZiSheng project",
    )
    story = [
        Paragraph("STATS 100A HONORS  /  PROFESSOR WU  /  {DATE}".format(DATE=DATE_LABEL.upper()), styles["KickerCustom"]),
        Paragraph("From noise to Chinese characters", styles["TitleCustom"]),
        Paragraph("First scope brief and mathematical illustration | ZiSheng project", styles["MetaCustom"]),
        Paragraph("Research question", styles["SectionCustom"]),
        Paragraph(
            "Can a small class-conditional diffusion model generate recognizable images of selected Chinese characters? How do Gaussian corruption and character stroke count relate to recognizability?",
            styles["BodyCustom"],
        ),
        Paragraph("Proposed study", styles["SectionCustom"]),
        Paragraph(
            "<b>Data.</b> Begin with a controlled rendered-font pilot. Consider isolated handwriting for the main study only after verifying source access, labels, writer-level splits, character coverage, and use conditions. Keep the main analysis within one data domain.",
            styles["BodyCustom"],
        ),
        Paragraph(
            "<b>Model.</b> Condition a small denoiser on a character ID and timestep. Start with a three-character pipeline pilot; scale to at least nine characters if the data and training gates pass. Twenty-four is a target, not a commitment.",
            styles["BodyCustom"],
        ),
        Paragraph(
            "<b>Evaluation.</b> Assess requested-character identity separately from image fidelity. Treat stroke-count patterns as associations within the selected set, not evidence that stroke count causes performance differences.",
            styles["BodyCustom"],
        ),
        Paragraph("Forward corruption: the first mathematical object", styles["SectionCustom"]),
        Paragraph(
            "x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps, where eps ~ N(0, I). Thus, x_t given x_0 is Normal with mean sqrt(alpha_bar_t) x_0 and covariance (1 - alpha_bar_t) I. The first expression constructs the noisy image; the second gives its conditional mean and covariance. This is the fixed forward process, not the learned reverse process.",
            styles["BodyCustom"],
        ),
        PDFImage(str(figure_path), width=page_width - 2 * margin, height=(page_width - 2 * margin) * 540 / 1440),
        Paragraph(
            "Figure 1. T=1,000 with a linear beta schedule from 0.0001 to 0.02; one shared seeded noise draw. Panel schedule SNR is alpha_bar/(1-alpha_bar). The glyph is rendered in macOS Songti SC Regular. Values are clipped to [-1, 1]. This synthetic font illustration is not handwriting, training data, or model output.",
            styles["FigureCaptionCustom"],
        ),
        Paragraph("Status and next step", styles["SectionCustom"]),
        Paragraph(
            "The repository has planning notes and interface placeholders only. No data have been acquired, and no model has been implemented, trained, or evaluated. Next: choose an openly licensed font for the actual dataset, verify character coverage and terms, then make a labeled contact sheet.",
            styles["BodyCustom"],
        ),
        Paragraph("Questions for our meeting", styles["SectionCustom"]),
        Paragraph(
            "1. Is rendered-font generation an acceptable controlled main domain if handwriting access or use terms delay the project?",
            styles["SmallCustom"],
        ),
        Paragraph("2. Which outcome should lead: generated-character recognition or corruption/restoration of held-out images?", styles["SmallCustom"]),
        Paragraph("3. What report format, final deadline, and mathematical depth should I plan for?", styles["SmallCustom"]),
        Spacer(1, 2),
        Paragraph(
            'Reference: Ho, Jain, and Abbeel (2020), <i>Denoising Diffusion Probabilistic Models</i>, Section 2, Equation 4. <link href="https://arxiv.org/abs/2006.11239" color="#187D78">Open paper</link>.',
            styles["SmallCustom"],
        ),
    ]
    # The exact coefficients are returned by the figure builder and are useful
    # in the script's output metadata; labels in the PNG are computed there.
    assert set(alphas_used) == {100, 400, 900}
    doc.build(story)


def main() -> None:
    figure_path = ROOT / "results" / "figures" / "forward_corruption_schematic.png"
    pdf_path = ROOT / "output" / "pdf" / "2026-10-26-first-faculty-brief.pdf"
    markdown_path = ROOT / "docs" / "meetings" / "2026-10-26-first-faculty-brief.md"
    alphas_used = _paint_figure(figure_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown_text("../../results/figures/forward_corruption_schematic.png"), encoding="utf-8")
    _make_pdf(pdf_path, figure_path, alphas_used)
    print(f"Wrote {markdown_path.relative_to(ROOT)}")
    print(f"Wrote {figure_path.relative_to(ROOT)}")
    print(f"Wrote {pdf_path.relative_to(ROOT)}")
    print(f"Rendered character {CHARACTER} using {CJK_FONT_NAME} from {CJK_FONT_PATH}")
    for timestep, (alpha_bar, snr) in alphas_used.items():
        print(f"t={timestep}: alpha_bar={alpha_bar:.8f}, schedule_snr={snr:.8f}")


if __name__ == "__main__":
    main()
