# Compressed runbook — five sessions to submission

Replaces the eight-session plan. Session 1 is complete. `docs/review_response_plan.md` and `CLAUDE.md` remain authoritative for *what* things are; this file is authoritative for *what still gets done*.

**Standing rules.** Start each session in Plan mode. Read the plan before approving. Commit between sessions, then `/clear`. Never let Claude Code commit. One session per conversation.

**Cut from the original plan — do not do these:** the `_run_flower_core` task-adapter refactor and its bitwise gate; Dirichlet partitions; FedYogi; the R10 architecture comparison; ρ=0.67; multiple random position assignments; the client-masked evaluation protocol; full split-seed replication. If a session proposes any of them, decline.

**Write the paper in parallel.** Sections I–V depend on no number that is still moving. Start them during S3, not after S6.

---

## S1 close-out (30 minutes)

> Two things to finish Session 1.
>
> **1. Commit the client-ordering fix.** Sort client results by `client_id` before delegating to FedAvg. Reproducibility is a claim the paper makes — "mean ± std over 3 seeds" is not meaningful if one seed does not reproduce itself — and the ~1e-7 change is four orders of magnitude below anything reported. Commit it on its own with a message naming the cause: Ray returns actors in completion order and float addition is not associative.
>
> Record in `CHANGES.md` that the archived August artifacts are hereby retired as a bitwise reference, since they were produced under non-deterministic ordering and cannot be reproduced. The reference run generated in S2 replaces them.
>
> **2. Finish `CHANGES.md`** — fill in the end-to-end section from the measurements you already have, including the detector reproducing R1 (1 distinct residual, standardized MSE 1.0000407, skill −0.0028, both gates failed, override stamped) and the ordering finding.
>
> Then stop.

**Commit:** two commits — the ordering fix, then `CHANGES.md`.

---

## S2 — Direction check, partitioner, reference run

**Branch:** `git checkout main && git merge <session-1 branch> && git checkout -b s2-partitioner`

> Read `docs/review_response_plan.md` §2 R11, §1.1, §1.2, §3 item 3, and `docs/sensor_layout.md`.
>
> **First, answer R11 read-only, before writing any partitioner code.** The protocol had all subjects walking a single direction, yet `direction` varies across five of the eight `(source, subject)` strata. Report the distinct values cross-tabbed by source and subject with run counts, where the field is derived in the discovery code, and whether mean speed differs by direction within a subject. If it is a labeling artifact, stratify on `(source, subject)` only. If 20251124 genuinely has two directions, keep the three-part stratum and note it in the partitioner docstring — it is also a protocol difference between the two buildings that Section III must state.
>
> **Then rewrite the partitioner.**
>
> - Replace the brute-force search (`_remainder_assignments`, `_solve_subject_blocks`, `_assign_profiles`) with a round-robin over speed-ordered blocks. K must be a real parameter, working at K=2 and K=3.
> - Add the **natural source split** (K=2, one client per building) as a partition scheme.
> - Replace the `EXPECTED_*` constants and the config validator's hard-coded 3 clients / 60 rounds / 1 local epoch with the property assertions in §3 item 3. Move exact numbers into tests as independent literals.
> - Add the §1.1 split-integrity assertions plus a test that a deliberately leaky split fires them.
> - Assert the IID property: every client holds ≥1 run per subject, no client holds all runs of any subject.
> - Collapse the C2 duplication into one constants module (subject list ×7, channel list ×4, sample rate ×5). Update the stale docstring at `combined_iid_fl_partitioning.py:8-9`.
>
> **Finally, generate the reference run** — one full config under current settings, archived. This replaces the retired August artifacts as the comparison baseline for everything downstream.
>
> Propose a plan first.

**Check before approving:** exact numbers move *to tests*, not deleted. The natural K=2 split is included.
**Commit:** `git commit -m "S2: K-parameterized partitioner, natural source split, property assertions"`

---

## S3 — Fix the regression collapse

**Branch:** `git checkout main && git merge s2-partitioner && git checkout -b s3-r1-fix`

This is the session the paper depends on. Nothing downstream is worth running until it passes.

> Read `docs/review_response_plan.md` §2 R1. The diagnosis is corrected: clients are not learning locally — a fresh Adam per client per round over ~10 steps, versus a single persistent Adam over 1680 steps centrally.
>
> **Do not refactor `_run_flower_core`.** That work is cut; edit in place.
>
> - Parameterize local epochs, LR schedule (constant and per-round-restart options), and batch size.
> - Add **client optimizer state persistence across rounds** as a first-class option. Given the diagnosis this is likely the highest-value knob.
> - Add **FedAdam** as a selectable server strategy. Skip FedYogi.
> - Log each client's loss at the **start and end** of local training each round, not only the online average.
> - Report matched-budget step counts (FL total gradient steps vs centralized) in the run summary.
>
> Sweep in two stages: `local_epochs ∈ {1, 5, 10} × client_optimizer_state ∈ {fresh, persistent}` under FedAvg — six configs isolating the local axis — then FedAdam at the best local setting. Give me the table, **including the failing configurations**; the failures are a result, not noise.
>
> The R1 acceptance criteria must be asserted and fail loudly: final `global_train_standardized_mse` < 0.95, more than 10 distinct predicted residuals, positive skill score.
>
> `test_combined_iid_flower.py` pins the exact 60-round schedule and will break on parameterization — update it.
>
> Propose a plan first.

**Verify yourself:** open `test_window_predictions.csv` and count distinct values.
**If nothing in the sweep clears the criteria, stop and escalate before proceeding.**
**Commit:** `git commit -m "S3: fix federated regression collapse, client optimizer persistence, FedAdam"`

---

## S4 — Metrics, local-only, seeds, channel masks

**Branch:** `git checkout main && git merge s3-r1-fix && git checkout -b s4-metrics`

> Read `docs/review_response_plan.md` §2 R2, R5, R6, R11, and §1.3.
>
> **R2.** Consolidate `regression_metrics`, `_extended_metrics`, `_extended_regression_metrics` into one function returning baseline RMSE, model RMSE, ΔRMSE, R² (total), R² (within-subject), and skill score. Skill score is the primary metric; the baseline appears in every regression output. Add bootstrap 95% CIs over **runs**, not windows.
>
> **R5.** Add a `local_only` run mode — each client trains on its own partition only, evaluated on the shared global test set. Per-client and mean ± std. Both tasks, any partition scheme. This is the comparison the paper's central claim rests on.
>
> **R6.** Separate the split seed from the initialization seed; a single `config["seed"] = 4601` currently drives both.
>
> **R11 follow-through**, only if S2 found the direction variation real: add a `(source, subject, direction)`-keyed baseline alongside `(source, subject)`, report skill against both, and report mean speed by direction per subject.
>
> **§1.3 channel masking.** Partition the 8 hallway **positions**, expand to channel indices; channels 8 and 10 are co-located and travel together. Implement by **masking a 9-channel input** — zero masked channels after normalization, model stays plain `SimpleCNN1D` at 9 input channels, assert identical parameter shapes across clients. Contiguous spatial blocks only; skip the random-assignment variant. Support ρ ∈ {1.0, 0.33, 0.0} — do not special-case ρ=0.0, its collapse is a reported result.
>
> Do not use `MaskAwareSimpleCNN1D`, `training.py`, or `sensor_non_iid.py` — all quarantined.
>
> Propose a plan first.

**Check before approving:** "mask" and "identical parameter shapes" appear. Reject any proposal to slice the input or use per-client first layers.
**Commit:** `git commit -m "S4: unified metrics with skill score, local-only mode, seed separation, position masking"`

---

## S5 — Experiment matrix and derived metrics

**Branch:** `git checkout main && git merge s4-metrics && git checkout -b s5-experiments`

Mostly compute. Start it, then write Sections VI–VII against placeholders.

> Run the matrix:
>
> - **K=2, 3 init seeds each:** IID, natural building split, local-only. Both tasks.
> - **K=3, 1 seed each:** IID (ρ=1.0), channel ρ=0.33, channel ρ=0.0, local-only. Both tasks.
> - Global 9-channel evaluation protocol only.
> - Every run must pass the degenerate detector before its numbers are recorded. Flag failures rather than recording them silently.
>
> Then three derived results, none of which need new training runs:
>
> - **R3 end-to-end:** classifier predicts subject → look up that subject's train prior → add predicted residual → compare to true speed. Report beside the oracle-identity number.
> - **R4 run-level classification metrics** as primary: majority vote and mean probability, with Wilson 95% intervals. The effective test set is 28 runs, not 307 windows.
> - **R4 source probe:** same architecture, label = source instead of subject. One training run. Quantifies how separable the two buildings are on their own.
>
> Write results to a structured directory the figure module can read without retraining.
>
> Propose a plan first.

**Skip:** the subject-003 cross-session control unless S2–S4 finished early. It is the strongest defense against the site confound, but it needs its own split path.
**Commit:** `git commit -m "S5: experiment matrix, end-to-end regression, run-level classification, source probe"`

---

## S6 — Figures, tables, wrap

**Branch:** `git checkout main && git merge s5-experiments && git checkout -b s6-figures`

> Read `docs/review_response_plan.md` §4. Build `paper_figures.py` generating everything from saved artifacts with no retraining, via one `make_all_figures(results_dir, out_dir)` entry point.
>
> IEEE style exactly: 3.5in / 7.16in widths, 8pt base font, vector PDF plus 300dpi PNG, no titles inside figures, colorblind-safe with redundant encoding.
>
> Figures, in priority order — build them in this order so a time cut drops the last ones:
>
> 1. **Convergence** — metric vs round, with horizontal reference lines for centralized and the trivial baseline, band over seeds. This is the figure that would have caught R1.
> 2. **Skill comparison** — grouped bars across {local-only, FedAvg, FedAdam, centralized} × {IID, natural, channel ρ}, error bars over seeds, zero line marking the trivial baseline.
> 3. **Run-level scatter** — predicted vs actual speed, identity line, marked by subject, panels for centralized / FL / local-only.
> 4. **Partitions** — runs per client × subject, one panel per scheme.
> 5. **Dataset overview** — speed distribution per subject, annotated with within-subject std, split by source.
> 6. **Heterogeneity** — performance vs ρ, with the ρ=0 collapse plotted, not omitted.
> 7. **Confusion matrix** — run-level, hardest setting only.
> 8. **Residual diagnostics** — Bland–Altman against APDM, residual vs speed.
>
> Tables T1–T5 as LaTeX booktabs and CSV.
>
> Then two loose ends: write `PRIVACY_NOTES.md` (§2 R7 — threat model, FedAvg as data minimization not a formal guarantee), and record APDM reference precision (§2 R8) from runs with repeated or overlapping estimates, saying plainly if the data does not support it.
>
> Propose a plan first.

**Commit:** `git commit -m "S6: paper figure and table suite"`

---

## Stop and escalate if

- S2's direction check shows the field means something unexpected.
- No S3 configuration clears the R1 acceptance criteria.
- Any S5 run produces a flat convergence curve.
- Skill against a direction-keyed baseline collapses toward zero.
