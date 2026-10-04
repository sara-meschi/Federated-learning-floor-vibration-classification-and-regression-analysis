# Compressed runbook — five sessions to submission

Replaces the eight-session plan. Session 1 is complete. `docs/review_response_plan.md` and `CLAUDE.md` remain authoritative for *what* things are; this file is authoritative for *what still gets done*.

**Standing rules.** Start each session in Plan mode. Read the plan before approving. Commit between sessions, then `/clear`. Never let Claude Code commit. One session per conversation.

**Schedule — deadline 31 Oct 2026, manuscript complete.** Re-cut on Oct 4; the buffer is gone, so slippage now costs scope rather than time.

| Dates | Work |
|---|---|
| Oct 4 – 6 | **S2** partitioner rewrite |
| Oct 7 – 12 | **S3** the R1 fix — blocking; nothing downstream is worth running until it passes |
| Oct 13 – 16 | **S4** metrics, local-only, seed separation, position masking, Dirichlet |
| Oct 17 – 21 | **S5** experiment matrix (~26 runs, one overnight) and derived results |
| Oct 22 – 24 | **S6** figures and tables |
| Oct 25 – 31 | Manuscript assembly, advisor review, polish |

**Hard checkpoint: if S3 has not passed its acceptance criteria by Oct 14, cut to K=2 only** — natural cross-session vs IID vs local-only, both tasks, and drop the sensor and Dirichlet axes entirely. That is still a complete, defensible paper.

**Cut — do not do these.** Anything not named below is out of scope:

`_run_flower_core` refactor and its bitwise gate · FedYogi · C2 constants consolidation · R3 end-to-end path · R10 architecture comparison · R11 direction-keyed baseline · R8 APDM precision measurement (state it was not characterized) · bootstrap CIs (mean ± std over seeds is enough) · random position assignments (contiguous blocks only) · client-masked evaluation protocol (global 9-channel only) · extra split seeds · ρ = 0.67 · Dirichlet α = 0.5 · the dataset-overview, residual-diagnostics and communication-cost figures. If a session proposes any of them, decline.

**Not cuttable:** local-only baselines, matched-K pairing, run-level classification metrics with Wilson intervals, the source probe, and the subject-003 cross-session control. Those four are what defend the results.

**Write in parallel.** Sections III, IV and V depend on no number that is still moving — draft them from Oct 1, alongside S3; Sections I and II from Oct 6. That is six of ten pages written before any result lands, which is what makes the date achievable without daily work.

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
> **R11 is resolved — do not re-investigate.** The `direction` field is real, not an artifact: `20251124_Testing` alternated N→S / S→N by run-index parity per its documented protocol, verified against `testNotes.xlsx` (96/96) and independently from the vibration itself via energy-weighted centroid timing (95/96). `Test_2` is unidirectional and its single direction is recoverable as S→N. **Record that finding in the docs only — do not relabel the artifacts.** The relabel would require rebuilding both canonical artifacts for a cosmetic change with no effect on strata counts or partitioning; the paper states the direction in prose instead. Speed is direction-invariant (max |Δ| 0.028 m/s against a within-stratum std of 0.032, pooled permutation p = 0.15).
>
> **Keep the three-part `(source, subject, direction)` stratum key** — not for speed balance, which direction does not affect, but for signal coverage: direction reverses the order in which the corridor sensors activate, so a client that only ever sees N→S may fail on S→N. Note that reasoning in the partitioner docstring.
>
> **Then rewrite the partitioner.**
>
> - Replace the brute-force search (`_remainder_assignments`, `_solve_subject_blocks`, `_assign_profiles`) with a round-robin over speed-ordered blocks. K must be a real parameter, working at K=2 and K=3.
> - Add the **natural source split** (K=2, one client per collection campaign) as a partition scheme.
> - Replace the `EXPECTED_*` constants and the config validator's hard-coded 3 clients / 60 rounds / 1 local epoch with the property assertions in §3 item 3. Move exact numbers into tests as independent literals.
> - Add the §1.1 split-integrity assertions plus a test that a deliberately leaky split fires them.
> - Assert the IID property: every client holds ≥1 run per subject, no client holds all runs of any subject.
> - Update the stale docstring at `combined_iid_fl_partitioning.py:8-9`. (C2 constants consolidation is cut — pure hygiene, no reviewer sees it.)
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

> Read `docs/review_response_plan.md` §2 R2, R5, R6, and §1.3.
>
> **R2.** Consolidate `regression_metrics`, `_extended_metrics`, `_extended_regression_metrics` into one function returning baseline RMSE, model RMSE, ΔRMSE, R² (total), R² (within-subject), and skill score. Skill score is the primary metric; the baseline appears in every regression output. Report mean ± std over seeds. Skip bootstrap CIs.
>
> **R5.** Add a `local_only` run mode — each client trains on its own partition only, evaluated on the shared global test set. Per-client and mean ± std. Both tasks, any partition scheme. This is the comparison the paper's central claim rests on.
>
> **R6.** Separate the split seed from the initialization seed; a single `config["seed"] = 4601` currently drives both.
>
> **§1.3 channel masking.** Partition the 8 hallway **positions**, expand to channel indices; channels 8 and 10 are co-located and travel together. Implement by **masking a 9-channel input** — zero masked channels after normalization, model stays plain `SimpleCNN1D` at 9 input channels, assert identical parameter shapes across clients. Contiguous spatial blocks only — skip the random-assignment variant. Support ρ ∈ {1.0, 0.33, 0.0}. Do not special-case ρ=0.0 — its collapse is a reported result. ρ=1.0 is identical to IID at K=3, so run it once and use it for both.
>
> **Also add a Dirichlet label-skew partitioner** over subjects at the **run** level, K=3, α = 0.1 only. This is the label-space non-IID axis, paired with the sensor axis above. Keep run indivisibility. If a draw leaves a client with fewer than 2 subjects, resample, and record that a floor was applied since the paper must state it.
>
> Do not use `MaskAwareSimpleCNN1D`, `training.py`, or `sensor_non_iid.py` — all quarantined.
>
> Propose a plan first.

**Check before approving:** "mask" and "identical parameter shapes" appear. Reject any proposal to slice the input or use per-client first layers.
**Commit:** `git commit -m "S4: unified metrics with skill score, local-only mode, seed separation, position masking"`

---

## S5 — Experiment matrix and derived metrics

**Branch:** `git checkout main && git merge s4-metrics && git checkout -b s5-experiments`

Mostly compute — roughly 26 runs, one overnight. Start it, then write Sections VI–VII against placeholders.

> Run the matrix. Both tasks throughout, global 9-channel evaluation only.
>
> - **K=2:** IID and natural cross-session, 3 init seeds each; local-only, 1 seed.
> - **K=3, 1 seed each — sensor axis:** IID (= ρ=1.0), ρ=0.33, ρ=0.0, local-only.
> - **K=3, 1 seed — subject axis:** Dirichlet α=0.1. Local-only at K=3 is already covered above.
>
> Every run must pass the degenerate detector before its numbers are recorded — flag failures rather than recording them silently. Expect α=0.1 classification to degrade sharply for the same structural reason as ρ=0: a client that never sees a subject contributes nothing about that class. Report it as the characterized endpoint, not a failed run.
>
> Then three things that need little or no new training:
>
> - **Run-level classification metrics** as primary (§2 R4): majority vote and mean probability, with Wilson 95% intervals. The effective test set is 28 runs, not 307 windows. No new runs needed.
> - **Source probe** (§2 R4): same architecture, label = source instead of subject. One training run. Quantifies how separable the two campaigns are on their own.
> - **Subject-003 cross-session control** (§2 R4 item 3). First the free breakdown: 003 already has test runs from both sources, so split the existing predictions by source. Then the real control — a small leave-one-source-out run, train on `Test_2` only (001, 002, 003), test on 003's runs from `20251124_Testing`. Three classes, ~56 training runs, fast. It needs its own evaluation split, permitted under the `CLAUDE.md` carve-out: additive code path only, no changes to `assign_run_splits`, the seed-4601 split, or the main artifacts.
>
> Write results to a structured directory the figure module can read without retraining.
>
> Propose a plan first.

**Commit:** `git commit -m "S5: experiment matrix, run-level metrics, source probe, 003 control"`

---

## S6 — Figures, tables, wrap

**Branch:** `git checkout main && git merge s5-experiments && git checkout -b s6-figures`

> Read `docs/review_response_plan.md` §4. Build `paper_figures.py` generating everything from saved artifacts with no retraining, via one `make_all_figures(results_dir, out_dir)` entry point.
>
> IEEE style exactly: 3.5in / 7.16in widths, 8pt base font, vector PDF plus 300dpi PNG, no titles inside figures, colorblind-safe with redundant encoding.
>
> **Five figures**, in this order so a time cut drops the last ones:
>
> 1. **Convergence** — metric vs round, with horizontal reference lines for centralized and the trivial baseline, band over seeds. This is the figure that would have caught R1.
> 2. **Skill comparison** — grouped bars across {local-only, FedAvg, FedAdam, centralized} × {IID, natural, channel ρ, Dirichlet}, error bars over seeds, zero line marking the trivial baseline.
> 3. **Heterogeneity** — performance vs ρ, with the ρ=0 collapse plotted rather than omitted.
> 4. **Partitions** — runs per client × subject, one panel per scheme.
> 5. **Run-level scatter** — predicted vs actual speed, identity line, marked by subject, panels for centralized / FL / local-only.
>
> **Four tables** as LaTeX booktabs and CSV: T1 dataset, T2 centralized reference, T3 FL under IID at K=2 and K=3, T4 FL under non-IID. Every regression table carries the trivial baseline as a row.
>
> A run-level confusion matrix only if the hardest non-IID setting is genuinely off ceiling; skip it otherwise rather than printing four near-identical saturated matrices.
>
> Then one loose end: write `PRIVACY_NOTES.md` (§2 R7 — threat model, raw vibration never leaves its silo, FedAvg as data minimization not a formal guarantee, gradient inversion out of scope).
>
> Propose a plan first.

**Commit:** `git commit -m "S6: paper figure and table suite"`

---

## Stop and escalate if

- No S3 configuration clears the R1 acceptance criteria — hard stop, by Oct 7.
- Any S5 run produces a flat convergence curve.
- A session proposes work not named in this file.
