# How the 字生 agent matrix worked

This log records what each role received, what evidence it returned, and why the lead acted on it. Times and dates use America/Los_Angeles. The experiment contract is [Glyph entropy and source recovery protocol](glyph-entropy-protocol.md).

## Round 1 · November 17, 2026

| Role | Delegation | Decision |
|---|---|---|
| Literature | GPT-6.2 Astra / medium, two targeted lookups at most | Keep the 2×2 local-pattern question narrow. Matsuo uses local-feature entropy for reference selection; FontDiffuser groups Chinese generation metrics by strokes. The proposed test specifically measures incremental prediction of restoration. [Matsuo, table 1](https://arxiv.org/html/2606.16502v1), [FontDiffuser, table 1](https://arxiv.org/html/2312.12142v1) |
| Feasibility | GPT-6.2 Sol / low, read local repository at `aff9ffd` | Reuse the small DDPM and q-sampling path. The two widths instantiate with 240,993 and 881,153 parameters for 24 classes. |
| Methods audit | Separate GPT-6.2 Sol / high, received the frozen candidate protocol | Initial verdict was revise. Fixed the primary loss and averaging, the holdout unit, deterministic scoring, evaluation isolation, and the precise range of inference. A short second pass agreed to protocol drafting. |

## Round 2 · November 18, 2026

| Work package | Owner and inputs | Handoff and decision |
|---|---|---|
| INSIGHT-02 | GPT-6.2 Astra / extra high, protocol brief only, one bounded research pass | Use the two held-out renditions to compare a prediction to its own and the other font. Adoption criterion was a falsifiable comparison with a label-only zero check using saved masks. The entropy endpoint and training budget stay as specified in the main protocol. |
| A1 · Data | GPT-6.2 Sol / medium, exclusive write scope `data/study/` and the builder | Supply pinned sources, a six-family split and a clean archive for both the trainer and scorer. Track lineage and licenses. Report the final archive fingerprint before training. |
| A2 · Scoring | GPT-6.2 Sol / medium, exclusive write scope for the evaluator and its tests | Consume the shared class IDs, uint8 rasters and four named EMA weights. Produce the row keys, masks, summary and metrics. Test with fixtures and validation data. |
| Training and integration | Project lead, existing DDPM and A1 asset contract | Benchmark MPS locally, lock a common update count after the validation-only dry run, and record optimizer/seed streams for the four runs. Use the same code for the public reproduction entry point. |
| VER-02 · Results audit | Separate GPT-6.2 Sol / high, received the frozen protocol, file schemas, and direct evidence | Wrote an independent verifier using per-stratum residualization for ridge, re-derived both template methods, recalculated Dice on the full score table, and replayed a subset of the checkpoint inference. |

### Decisions and evidence

- **Select families from provenance before scores.** A1 found possible overlap between the first candidate, LXGW WenKai, and Source Han source material. It kept all train and validation rasters fixed, checked original font descriptions, and chose ZCOOL KuaiLe and Long Cang for the test split. KuaiLe has a separate typeface design within the same distribution initiative as two other ZCOOL families. The completed check covers 144 clean rasters and 264 cross-split same-label pairs, plus a byte-identical rebuild.
- **Lock data and scoring separately.** Commit `a13a846` locked the corpus, training code, and configuration. In the technical short run the two 128-update MPS checkpoints finished in about 1.8 and 2.4 seconds. That supported 4,000 updates at each width and seed on the same machine. Commit `c71dce2` locked the scorer, the crossed measure, and the selection rule for the example figure before looking at holdout outputs.
- **Run the same path on validation first.** The seven deterministic scorer fixtures cover the observation model, crossed contrast, paired means, fold isolation, and mask replay. The agent also verified a complete synthetic CPU run. The local 128-update runs then produced 13,824 real validation rows through the evaluation CLI. Test evaluation used separate CPU noise seeds with the frozen models.
- **Spend by failure impact.** The lead reserved one extra-high Astra pass for the question whose answer could change what the study measures. Building data and scoring used medium Sol with exclusive file ownership. Training, repeated scoring, and file hashing ran in scripts. The four formal runs took about 247 seconds of measured update time on Apple MPS. The trial used the existing local machine.
- **Release after independent checks.** VER-02 recovered 27,648 test scores from bit-packed masks, reproduced all primary and crossed-font results, and verified the archive and checkpoint hashes. A separate CPU inference replay covered 192 scored rows. One threshold-near pixel out of 442,368 changed sign in replay. Its magnitude was within the stated tolerance and the saved-mask statistics matched exactly. The [audit output](../results/glyph-entropy-v1/audit.json) records this.
- **Close routine work locally.** The independent verdict arrived before the agent hit its usage limit. The lead reran its verifier against the release directory, including the compressed score file, without another model assignment. The packaged result passed the same checks.

## How to reproduce the delegation

The lead first fixes what flows between roles. Here that was the manifest class and font order, the 48×48 uint8 archive, named width/seed checkpoints, and per-row prediction keys. The data builder and evaluator then work simultaneously in disjoint files. Give the specialist a use decision, a narrow output scope and a stop condition. Reserve the high-effort slot for an insight that changes the protocol or for an audit that can overturn a result. Leave batches, file checks and training to scripts. Provide the independent results reviewer the frozen rules and independently inspectable evidence before sharing the report narrative.

For another wave, record the task ID, model and effort, input version, allowed files, expected output and smallest validation. On return, record the evidence, the resulting change, the follow-up requirement, and whether the change preceded evaluation. A failed audit reopens the affected phase. Preserve the original verdict and the resolving check in the same log.

This is evidence of AI assistance. Rain should use the [project's AI assistance record](ai-assistance.md) and course guidance when reviewing the draft and deciding how to describe their own implementation and analysis.
