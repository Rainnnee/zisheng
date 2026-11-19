# AI assistance and authorship notes

Use this log to record substantive AI assistance, including the date, task, suggestions used, changes made by the student, and how the result was checked. Preserve the student's own derivations, implementation decisions, and explanations. Follow the course's disclosure and authorship requirements.

## 2026-11-07 — Pilot implementation

- **Assistance:** Codex helped implement the font-rendering data path, forward and reverse DDPM equations, conditional U-Net, training loop, sampling CLI, report figures, and documentation.
- **Direction:** The project owner requested a runnable three-character pilot and asked for explanatory comments and organized documentation.
- **Changes made:** The initial 100-step variant was replaced after inspecting its terminal schedule and generated samples. The recorded demo uses 1,000 steps.
- **Checks:** The code was run on Apple MPS. The generated grid was visually inspected. This was not an independent code review, and no held-out recognition evaluation was performed.
- **Student review:** The derivations, code, and course-policy disclosure still need the student's own review before submission.

## 2026-11-18 — Held-out glyph experiment

- **Assistance.** Codex coordinated the research question, frozen data panel, paired DDPM runs, scorer, separate results audit, figures, and LaTeX draft. Subagents supplied literature comparison, data provenance and implementation, scorer implementation, a crossed-font diagnostic, and the independent recomputation.
- **Evidence.** The [agent workflow](agent-matrix.md) records what each role received, the changes it prompted, and how its output was checked. The [test archive](../results/glyph-entropy-v1/) preserves predictions, configurations and numerical checks.
- **Student review.** Rain should read and revise the paper and document the derivations, implementation and interpretation they can substantiate before course submission.
