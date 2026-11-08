# Project proposal

## Working title

字生 (Zìshēng): From Noise to Chinese Characters

## Research question

Can a small class-conditional diffusion model generate recognizable images of selected Chinese characters, and how do corruption level and character stroke count relate to recognizability?

## Proposed scope

The project uses Python and PyTorch to study a carefully documented character set. The first runnable pilot uses a controlled printed-font domain, a Gaussian forward process, a timestep- and class-conditioned denoiser, reverse sampling, and a fixed record of settings and outputs. Handwriting remains a possible later domain if data access, parsing, writer-level splits, coverage, and use terms can be verified in time.

The proposed questions are related but require different evaluation designs. The current demo implements conditional generation from noise. Corruption and restoration can be added only after the main project question and its evaluation are agreed upon.

## Current status

The repository first established a three-class end-to-end demo. The expanded visual pilot now uses 12 characters with Unicode stroke counts from 3 to 10, rendered at 48×48 from two openly licensed Simplified Chinese fonts. The new run uses 3,072 images and 8,000 optimizer updates; it remains a qualitative demonstration.

This pilot demonstrates the implementation path and a wider range of shapes. It does not establish accuracy, generalization to new fonts, or a relationship between stroke count and recognizability. The number of classes per stroke count is uneven, and character identity is tied to shape; a held-out evaluation protocol and a more controlled design are still needed before making those claims.

## Next decisions

1. Confirm with Professor Wu whether conditional generation should be the main study or whether restoration should replace it.
2. Decide whether font-rendered images are a suitable main domain for the course project.
3. Define a recognition measure and held-out font split before drawing conclusions from the expanded dataset.
4. Check the course contract for the final report requirements and mathematical depth.
