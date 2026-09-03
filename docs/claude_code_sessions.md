# Claude Code runbook — session by session

Work through these in order. One session per work package. Between every session: commit, then `/clear`.

**Standing rules**

- Start every session in **Plan mode** (the mode indicator is at the bottom of the prompt box; `Shift+Tab` cycles, or set `claudeCode.initialPermissionMode` to `plan` in VS Code settings).
- Read the plan before approving. `Ctrl+G` opens it in your editor so you can edit it directly.
- When approving, choose **"Yes, manually approve edits"** for sessions 0–4. You can loosen to automatic edits for sessions 5–8 once you trust it on this codebase.
- **Never let Claude Code run `git commit`.** You commit, after you've read the diff.
- If it starts fixing things you didn't ask for: *"That's in §5b, out of scope. Stay on this session's work."*
- If it reports success without showing you evidence: ask for the numbers. A collapsed model is how this project got into trouble in the first place.

---

## Session 0 — Orientation and verification (no code changes)

**Mode:** Plan. Stay in Plan the whole session.
**Branch:** none needed.

> Read `CLAUDE.md` and `docs/review_response_plan.md` in full, then read the codebase and the saved artifacts.
>
> Do not change any code in this session. I want three things:
>
> 1. Your own independent verification of the failure described in §2 (R1). Check `regression/test_window_predictions.csv` for the number of distinct predicted residuals, and the round-by-round `global_train_standardized_mse` history. Tell me what you actually find — do not assume the report is correct.
> 2. A list of anything in the codebase that contradicts the ground truth in `CLAUDE.md` under "Data" or "Signal processing" — especially the sampling rates, the subject list, the subject-006 exclusion, and the channel indexing.
> 3. A short inventory of which files the eight work packages will touch, so I can see where the risk is concentrated.
>
> Report all three. No edits.

**Before moving on:** if its verification of R1 disagrees with the plan, stop and tell me. Everything downstream assumes that diagnosis is right.

---

## Session 1 — WP0 guardrails

**Branch:** `wp0-guardrails` already exists — if you are already on it, stay. Otherwise `git checkout wp0-guardrails`.

> Read `docs/review_response_plan.md` §3 (WP0) and implement items 1, 2, 4, and 5 only — skip item 3, the partitioner, which is the next session.
>
> Specifically: unify `_set_random_seeds` into one shared implementation including the cudnn determinism block; make `--verify-only` write nothing; add the degenerate-model detector that runs at the end of every regression run; add `conftest.py` and `pyproject.toml` so pytest runs from the repo root, add pytest to requirements, and mark data-dependent tests with `@pytest.mark.requires_data`.
>
> Also from §2: fix R9 — `test_evaluation_counter` must actually increment on every touch of the test set. And R8 — the sampling-rate override is already surfaced in the centralized artifact summaries, `setup_audit.conditions`, and `run_manifest.csv`; the only gap is that the FL per-task `training_summary.json` carries no sample-rate field. Extend that, don't rebuild what exists.
>
> Do not touch preprocessing, the artifact schema, or the run-splitting logic. Propose a plan first.
>
> Then four items from your Session 0 findings, in this order:
>
> **Quarantine the legacy pipeline — in this order, because the order matters.** `federated.py:15` imports `training` at module level, so guarding `training.py` first would propagate an exception through `federated.py` into `combined_iid_flower.py:33` and take down the live pipeline.
>
> 1. Create `parameters.py` containing `get_parameters` and `set_parameters` (currently `federated.py:304` and `:308`), with no dependency on `training.py`. Session 4's local-only mode will need these too.
> 2. Update the two consumers: `combined_iid_flower.py:33` and `test_combined_iid_flower.py:15`.
> 3. Verify the combined pipeline still runs. Only then proceed.
> 4. Move `preprocessing.py`, `sensor_non_iid.py`, `iid_partitioning.py`, `training.py`, `config.py`, `federated.py`, the 17 legacy configs, the 14 legacy scripts, and the 2 legacy test files into `legacy/` (using `legacy/scripts/` and `legacy/tests/`), with an import-time guard that raises. The test files must move — dead imports there would break collection for the whole suite, which defeats the pytest work in this same session.
> 5. `MaskAwareSimpleCNN1D` shares `models.py` with `SimpleCNN1D`: move the **class**, leave the file. `SimpleCNN1D` stays put.
>
> Verify with the full test suite and one combined config end to end.
>
> **Delete the stale artifacts.** `artifacts/raw_windows*` contain subject 006 (A2, materialized on disk). Remove them so nothing can read them by accident.
>
> **Fix the stale docs** — README.md lines 14, 39, 54-55, 95, 100 and PLAN.md lines 22, 34, 35 (B1–B6). They currently claim 20 channels, a 60/20/20 train/val/test split, a validation-selected best model, and a single-source dataset. All false, and later sessions will read them.
>
> **Add the C1 assertion**: the 1-indexed → 0-indexed channel conversion at `combined_centralized_classification.py:433` and `combined_centralized_regression.py:171` is correct but unasserted. Assert it.
>
> Also write your full Session 0 contradiction list to `docs/session0_findings.md` so it survives `/clear`.

**Check before approving:** nothing in the plan touches `preprocessing.py` or changes any number listed in `CLAUDE.md`.
**Verify after:** `pytest` runs from root. Run `--verify-only` and confirm `git status` shows no new files.
**Commit:** `git add -A && git commit -m "S1: guardrails, determinism, verify-only, degenerate detector"`

---

## Session 2 — Partitioner rewrite

**Branch:** `git checkout main && git merge wp0-guardrails && git checkout -b wp0-partitioner`

> Read `docs/review_response_plan.md` §3 item 3, plus §1.1 and §1.2, plus `docs/sensor_layout.md`.
>
> Rewrite the partitioner:
>
> - Replace `build_combined_iid_partitions`'s brute-force search (`_remainder_assignments`, `_solve_subject_blocks`, `_assign_profiles`) with a round-robin over speed-ordered blocks. It must accept K as a real parameter and work for K=2 and K=3.
> - Replace the `EXPECTED_*` outcome constants and the config validator's hard-coded 3 clients / 60 rounds / 1 local epoch with the property assertions listed in §3 item 3.
> - Move the exact expected numbers into the test files as independent literals, not imported from the library modules.
> - Add the split-integrity assertions from §1.1, including a test that constructs a deliberately leaky split and confirms the assertion fires.
>
> Also collapse the C2 duplication from your Session 0 findings into one constants module: the subject list appears 7 times, the channel list 4 times, the sample rate 5 times. §5b names this the one duplication worth fixing before submission, and you are already editing several of the sites. Update the stale docstring at `combined_iid_fl_partitioning.py:8-9` (B7), which still claims the design is intentionally fixed to three clients and seed 4601.
>
> Propose a plan first.

**Check before approving:** the IID property "no client holds all runs of any subject" is in there, and the exact numbers are moving *to tests*, not being deleted.
**Verify after:** run the partitioner at K=2 and K=3 and eyeball the per-client counts.

**Then regenerate the reference run.** The round-robin produces a different run-to-client assignment than the old brute-force search, so the archived August artifacts stop being a valid comparison point here. Re-run one full config under the current (still-broken) settings and archive the outputs — this is the baseline Session 3's bitwise gate compares against, not the August artifacts. ~16 min on CPU for both tasks; one task is enough.

**Commit:** `git commit -m "S2: K-parameterized partitioner, property assertions, split-integrity tests"`

---

## Session 3 — Fix the regression collapse (the critical one)

**Branch:** `git checkout main && git merge wp0-partitioner && git checkout -b wp1-regression-fix`

> Read `docs/review_response_plan.md` §2, R1. Note the corrected diagnosis: clients are not learning locally — the cause is a fresh Adam per client per round over ~10 steps, not aggregation.
>
> **Step 1, before any behavioral change.** Refactor `_run_flower_core` with a task adapter (`build_model` / `build_loader` / `evaluate` / `history_row`) collapsing the seven task branch points. Then re-run one config and confirm the outputs are **bitwise identical to the reference run archived at the end of Session 2** — not to the archived August artifacts, which are stale because Session 1 changed RNG consumption and Session 2 changed the partition assignment. If they differ, revert the refactor and continue without it. Do not proceed to step 2 until this gate passes. See the exception note in §5b.
>
> **Step 2.** Parameterize local epochs, LR schedule (constant and per-round-restart options), batch size, and server optimizer. Add **client optimizer state persistence across rounds** as a first-class option. Add FedAdam and FedYogi as selectable Flower strategies. Add the matched-budget reporting from R1.
>
> Also add the local-training instrumentation: log each client's loss at the **start and end** of its local training each round, not only the online average.
>
> **Step 3.** Sweep in two stages. First `local_epochs ∈ {1, 5, 10} × client_optimizer_state ∈ {fresh, persistent}` under FedAvg — six configs isolating the local axis. Then FedAdam at the best local setting. Give me the comparison table.
>
> The R1 acceptance criteria must be asserted in code and fail the run loudly: final `global_train_standardized_mse` < 0.95, more than 10 distinct predicted residuals, positive skill score vs the subject-mean baseline.
>
> Note that `test_combined_iid_flower.py` pins the exact 60-round schedule and will break on parameterization — update it.
>
> Propose a plan first.

**Check before approving:** it is not changing the model architecture, the residual target, or the run-balanced loss. Only the optimization setup.
**Verify after:** open `test_window_predictions.csv` yourself and count distinct values. This is the number the whole paper turns on.
**Commit:** `git commit -m "S3: fix federated regression collapse, add FedAdam/FedYogi, sweep"`

**If nothing in the sweep clears the acceptance criteria, stop and tell me before continuing.**

---

## Session 4 — Metrics re-cut and local-only baseline

**Branch:** `git checkout main && git merge wp1-regression-fix && git checkout -b wp2-metrics`

> Read `docs/review_response_plan.md` §2, R2, R5, and R6.
>
> R2: consolidate `regression_metrics`, `_extended_metrics`, and `_extended_regression_metrics` into one function returning baseline RMSE, model RMSE, ΔRMSE, R² (total), R² (within-subject), and skill score. Make skill score the primary reported metric, with the baseline present in every regression output. Add bootstrap 95% CIs computed over **runs**, not windows.
>
> R5: add a `local_only` run mode — each client trains on its own partition only and is evaluated on the shared global test set. Report per-client and mean ± std. Both tasks, any partition scheme.
>
> R6 prerequisite: separate the split seed from the initialization seed. A single `config["seed"] = 4601` currently drives both, which makes "3 initialization seeds on the fixed split" impossible. This is a source change and belongs here, not in the execution session.
>
> Propose a plan first.

**Check before approving:** the consolidated metrics function is used everywhere, not added as a fourth implementation.
**Commit:** `git commit -m "S4: unified regression metrics with skill score, local-only baseline"`

---

## Session 5 — Position masking and the ρ sweep

**Branch:** `git checkout main && git merge wp2-metrics && git checkout -b wp5-channel-noniid`

> Read `docs/review_response_plan.md` §1.3 and `docs/sensor_layout.md` in full.
>
> Two notes before you start. **Do not use `MaskAwareSimpleCNN1D` or `training.py`** — both are legacy and quarantined as of Session 1. The mask belongs in the dataset's `__getitem__`, applied after normalization; the model stays plain `SimpleCNN1D` with 9 input channels and needs no mask awareness. That is what keeps parameter shapes identical across clients, which is the entire point. The legacy 18-channel doubling concatenated a mask-indicator channel, which is unnecessary here because each client's mask is static.
>
> Likewise `sensor_non_iid.py` must **not** be reused — it slices rather than masks, assumes K=4, pulls in out-of-hallway channels 16–19, and splits train/test inside a run.
>
> Implement the channel-availability non-IID partitioning:
>
> - Partition the 8 hallway **positions**, then expand to channel indices. Channels 8 and 10 are co-located and always travel together. ρ is defined over positions.
> - Implement by **masking a 9-channel input**, never by slicing. Zero masked channels after normalization. Assert that all clients have identical model parameter shapes.
> - Contiguous spatial blocks as the primary scheme; support ≥3 random position assignments as a robustness variant.
> - Sweep ρ ∈ {1.0, 0.67, 0.33, 0.0}. Do not skip or special-case ρ=0.0 — the collapse there is a result we are reporting.
> - Both evaluation protocols: global test with all 9 channels, and client-masked test.
> - Log positions per client and channels per client separately.
>
> Propose a plan first.

**Check before approving:** the words "mask" and "identical parameter shapes" appear. If it proposes slicing the input tensor or per-client first layers, reject — FedAvg cannot average those.
**Commit:** `git commit -m "S5: position-level channel-availability non-IID via masking, rho sweep"`

---

## Session 6 — Run the experiment matrix

**Branch:** `git checkout main && git merge wp5-channel-noniid && git checkout -b experiments`

This session is mostly execution, but not purely — confirm the split/init seed separation from Session 4 landed before starting. Kick it off, then write Sections I–V of the paper while it runs.

> Read `docs/review_response_plan.md` §5 (the schedule table, days 9–13) and run the full experiment matrix:
>
> - K=2: IID, natural 2-building split, local-only
> - K=3: IID, channel-availability at ρ ∈ {0.67, 0.33, 0.0}, local-only
> - Both tasks, both evaluation protocols where applicable, 3 initialization seeds, ≥3 position assignments for the channel runs.
>
> Every run must pass the degenerate-model detector before its numbers are recorded. Write results to a structured results directory so the figure module can read them without retraining. Report progress as runs complete, and flag any run that fails the acceptance criteria rather than silently recording it.

**Check while running:** spot-check one convergence history. Flat lines mean something is still wrong.
**Commit:** `git commit -m "Experiment matrix: K=2 and K=3, IID / natural / channel non-IID, 3 seeds"`

---

## Session 7 — End-to-end path and classification honesty

**Branch:** `git checkout main && git merge experiments && git checkout -b wp3-classification`

> Read `docs/review_response_plan.md` §2, R3 and R4.
>
> R3: implement the end-to-end regression path — classifier predicts subject, look up that subject's train-set prior, add the predicted residual, compare to true speed. Report beside the oracle-identity number.
>
> R4: implement run-level classification metrics as primary (majority vote and mean probability, with Wilson 95% intervals); the source-classification probe; and the subject-003 cross-session control described in R4 item 3.
>
> The 003 control needs its own evaluation split, which is permitted — see the carve-out in `CLAUDE.md`. It must be a separate, additive code path that does not modify `assign_run_splits`, the canonical seed-4601 split, or the main artifacts.
>
> Do not ablate window length — see R4 item 4.
>
> Propose a plan first.

**Commit:** `git commit -m "S7: end-to-end regression, run-level metrics, source probe, subject-003 control"`

---

## Session 8 — Figures and tables

**Branch:** `git checkout main && git merge wp3-classification && git checkout -b wp7-figures`

> Read `docs/review_response_plan.md` §4 (WP7) and build `paper_figures.py`.
>
> All 10 figures and 5 tables, generated from saved artifacts with no retraining, via a single `make_all_figures(results_dir, out_dir)` entry point plus individually callable functions. Follow the IEEE style requirements in §4 exactly: 3.5in / 7.16in widths, 8pt base font, vector PDF plus 300dpi PNG, no titles inside the figures, colorblind-safe palette with redundant encoding.
>
> While you are here, consolidate whatever duplicated plotting code survives Session 1's quarantine. Session 0 found that quarantine removes most of it for free — `_as_index_list` drops from 3 copies to 0 and `_confusion_matrix` from 5 to 1, since those copies live in files that moved. Re-count before acting rather than working from §4's list, which predates the quarantine, and consolidate what actually remains: the scatter implementations, `_resolve_path`, `_write_rows`, and the prediction-row builders.
>
> Tables emit as both LaTeX booktabs source and CSV.
>
> Finally, two loose ends with no other home: write `PRIVACY_NOTES.md` per §2 R7, and record the APDM reference precision per §2 R8 — compute the spread from runs with repeated or overlapping APDM estimates if the data supports it, and say so plainly if it does not.
>
> Propose a plan first.

**Check after:** Figure 4 (convergence with baseline and centralized reference lines) is the one that would have caught the original collapse. Look at it closely.
**Commit:** `git commit -m "S8: paper figure and table suite"`

---

## Optional Session 9 — architecture check

Only if days 18–20 are genuinely free. See §2, R10. Centralized-only comparison of SimpleCNN1D against a small 1D-ResNet. Cut this first if anything slipped.

---

## Where to stop and ask

- Session 0's verification disagrees with the plan's diagnosis of R1.
- No configuration in Session 3's sweep clears the acceptance criteria.
- Session 6 produces a flat convergence curve for any run.
- Any session proposes changing a number listed in `CLAUDE.md`.
