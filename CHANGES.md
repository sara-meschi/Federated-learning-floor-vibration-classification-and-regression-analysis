# CHANGES

What changed, and every number that moved as a result. Newest first.

---

## Session 3 — R1 sweep results, escalation, and regression-skill analysis (2026-10-19)

Branch `s3-r1-fix`. The sweep ran overnight on 2026-10-18/19 from commit `0819075`, under
the rule committed in `407f03f`. It **ended in escalation**, and the PI resolved it
(below). Per-run outputs are in `artifacts/s3_r1_sweep/` (gitignored except
`sweep_table.md` and `sweep_summary.json`).

### Stage 1 and 2: IID K=3, regression, 60 rounds, seed 4601

| Config | Steps | Final train std. MSE | Distinct test residuals | Test skill | Test RMSE (m/s) | Test R² | Detector | Eligible / rank | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg E=1 fresh | 1,800 | 1.0001 | **1** | +0.004 | — | — | failed (degeneracy, MSE) | — | 12.8 min |
| FedAvg E=1 persistent | 1,800 | 1.0000 | 238 (std 5e-7) | +0.000 | — | — | failed (MSE) | — | 14.3 min |
| FedAvg E=5 fresh | 9,000 | 1.0000 | **1** | +0.001 | — | — | failed (degeneracy, MSE) | — | 27.6 min |
| **FedAvg E=5 persistent** | 9,000 | 0.0549 | 238 | **+0.101** | 0.0384 | 0.932 | **passed** | **yes, rank 1** | 28.1 min |
| FedAvg E=10 fresh | 18,000 | 0.0176 | 238 | −0.011 | — | — | failed (skill) | — | 42.0 min |
| FedAvg E=10 persistent | 18,000 | 0.0220 | 238 | +0.302 | 0.0339 | 0.947 | passed | no (E cap, §2) | 43.1 min |
| FedAdam E=5 persistent | 9,000 | 0.3808 | 238 | passed | — | — | passed | yes, rank 2 | 28.2 min |

- **E=1 fresh reproduces R1 on the M5:** 1 distinct residual across 238 test windows,
  standardized MSE 1.00007 (the original evidence: 1.00004).
- **Persistent client optimizer state is the fix.** At E=5, `fresh` never leaves the
  trivial baseline, while `persistent` reaches train MSE 0.055. The start/end
  instrumentation shows the mechanism: under `fresh`, each client's loss at the end of
  local training equals its loss at the start, to 4 decimals, in every round.
- **E=10 fresh is a different failure:** 100 cold steps per round leave the plateau,
  but the model then overfits (lowest train MSE in the table, negative test skill). The
  detector's skill gate caught it.
- Verified by hand: E=5 persistent `test_window_predictions.csv` has 238 rows,
  238 distinct predicted residuals, std 0.040 m/s.
- The rule doc's E=10 cost estimate (~2 h per regression run) was too high: measured
  42–43 min.

### Natural K=2 confirmation (rule §4): both tasks

| Config | Classification final train acc. (reference E=1 fresh: 0.971) | Regression final train std. MSE | Distinct residuals | Regression test skill | Outcome |
|---|---|---|---|---|---|
| FedAvg E=5 persistent | 0.998 ✓ | 0.119 | 238 | **−0.207** | failed (skill) → fallback |
| FedAdam E=5 persistent | 1.000 ✓ | 0.606 | 238 | **−0.002** | failed (skill) → escalate |

`STOP AND ESCALATE: No eligible config passed the natural K=2 confirmation (rule §4).`

### Escalation resolution (PI, 2026-10-19)

The confirmation gate fired on **skill under the natural split, not on collapse.** Both
ranked configs trained, gave 238 distinct residuals, and passed classification. The PI
chose to **proceed to S5 with the pre-registered IID winner, FedAvg E=5 persistent**,
for both tasks and for local-only. The natural-split regression result is to be
reported as measured. This decision was made after seeing the confirmation results.
It is recorded as an appended section in `docs/s3_winner_rule.md`, and the original rule
text is unchanged.

### Matched-budget centralized reference (new, declared)

`configs/centralized_combined_residual_runbalanced_9ch_no006_300e.yaml`: identical to
the 60-epoch baseline except 300 epochs (cosine `T_max` follows), i.e. **8,400
optimizer steps**, against 9,000 for the FL winner. It does not replace the 60-epoch
baseline. To allow it, `load_residual_run_config` now accepts a non-60-epoch run only
when `training.epochs_override_reason` is set, and records the reason in the summary.

| Centralized residual | Steps | Final train std. MSE | Test skill | Test RMSE (m/s) | Test R² |
|---|---|---|---|---|---|
| 60 epochs (canonical) | 1,680 | 0.716 | +0.121 | 0.0380 | 0.934 |
| 300 epochs (matched budget) | 8,400 | **4e-6** | +0.272 | 0.0346 | 0.945 |

The 300-epoch run memorizes its training runs: train loss is 0.0317 at epoch 100 and
reaches 0 by epoch 250. **At a matched budget, FL (+0.101) is below centralized
(+0.272).** The 60-epoch centralized number understates what centralized training can
reach with the same number of steps.

### Run-level bootstrap and subject-007 sensitivity

`scripts/analyze_s3_regression_skill.py`. Skill = 1 − SSE_model / SSE_baseline against
the source/subject train mean, over the 28 held-out test runs. 95% percentile interval
from 10,000 resamples of test runs with replacement, seed 4601, with the same index
draws for every run. Per-source intervals resample within the source.

| Run | Overall (n=28) | Test_2 (n=13) | 20251124 (n=15) | Overall without 007 (n=25) |
|---|---|---|---|---|
| Centralized 60e | +0.121 [−0.288, +0.235] | −0.042 [−0.687, +0.245] | +0.136 [−0.348, +0.295] | −0.118 [−0.625, +0.210] |
| Centralized 300e (matched budget) | +0.272 [−2.299, +0.682] | −1.879 [−3.430, −0.431] | +0.472 [−2.400, +0.773] | −1.175 [−4.287, +0.184] |
| IID K=3 FedAvg E=5 persistent (winner) | +0.101 [−2.263, +0.480] | −0.443 [−1.417, +0.276] | +0.152 [−3.103, +0.543] | −1.159 [−4.331, +0.102] |
| IID K=3 FedAvg E=10 persistent | +0.302 [−2.082, +0.667] | −0.810 [−2.164, +0.307] | +0.405 [−2.714, +0.723] | −0.934 [−4.218, +0.287] |
| Natural K=2 FedAvg E=5 persistent | −0.207 [−4.128, +0.334] | −1.293 [−3.123, −0.183] | −0.106 [−5.441, +0.426] | −1.820 [−8.050, +0.179] |
| Natural K=2 FedAdam E=5 persistent | −0.002 [−1.876, +0.319] | −0.150 [−1.068, +0.427] | +0.012 [−2.716, +0.389] | −0.775 [−3.755, +0.300] |

**What this means for the paper. These are findings, not defects:**

1. **No regression model's skill is distinguishable from zero.** Every overall 95%
   interval, centralized included, contains 0.
2. **All of the positive skill comes from subject 007.** 007's three test runs carry
   ~73% of the baseline's squared error (baseline RMSE 0.106 m/s for 007, against 0.040
   overall). With 007 removed, **every** model, including both centralized runs, is worse
   than the subject-mean lookup.
3. **More training helps on 007 and hurts on Test_2.** Matched-budget centralized and
   FL E=10 both gain overall and lose on Test_2 (−1.88 and −0.81). On this test set,
   the CNN's advantage over the lookup is not a general within-subject speed signal.
4. Natural K=2 is the only setting with an interval entirely below zero (Test_2
   FedAvg: [−3.123, −0.183]).

### Run-level speed by subject and source (post-processing, no training)

From the canonical regression `run_manifest.csv` (run-level APDM speed, 112 train and
28 test runs). The train mean is the lookup baseline; it matches
`train_source_subject_means.csv` exactly. SD is the sample SD across runs. "Share" is
the subject's share of the baseline's total squared error on the 28 test runs (overall
baseline RMSE 0.0405 m/s).

| Source | Subject | Train runs | Test runs | Train mean (m/s) | Test mean (m/s) | Train SD (m/s) | Test SD (m/s) | Baseline RMSE on test (m/s) | Share of baseline SSE |
|---|---|---|---|---|---|---|---|---|---|
| Test_2 | 001 | 17 | 4 | 1.1376 | 1.1450 | 0.0128 | 0.0227 | 0.0210 | 3.8% |
| Test_2 | 002 | 18 | 5 | 1.2417 | 1.2440 | 0.0138 | 0.0147 | 0.0134 | 2.0% |
| Test_2 | 003 | 17 | 4 | 1.3650 | 1.3800 | 0.0384 | 0.0108 | 0.0177 | 2.7% |
| 20251124 | 003 | 12 | 3 | 1.4700 | 1.4533 | 0.0427 | 0.0189 | 0.0227 | 3.4% |
| 20251124 | 004 | 12 | 3 | 1.4879 | 1.4783 | 0.0268 | 0.0208 | 0.0195 | 2.5% |
| 20251124 | 005 | 13 | 3 | 1.6473 | 1.6467 | 0.0255 | 0.0104 | 0.0085 | 0.5% |
| 20251124 | **007** | 11 | 3 | 1.3259 | **1.3950** | 0.0592 | **0.0984** | **0.1059** | **73.2%** |
| 20251124 | 008 | 12 | 3 | 1.4121 | 1.4400 | 0.0327 | 0.0397 | 0.0428 | 11.9% |

**Subject 007 is the only subject whose test runs lie off its training distribution.**
Its three test runs (1.305, 1.38, 1.50 m/s) average 0.069 m/s above its train mean, and
one is faster than every 007 training run (train max 1.455). The lookup's error there
is 5–12× any other subject's, so "skill over the lookup" on this test set is largely
whether a model tracks 007's speed shift.

### Numbers that moved

- New reference: **centralized residual, 300 epochs: skill +0.272, RMSE 0.0346 m/s,
  R² 0.945.**
- First non-degenerate federated regression: **IID K=3 FedAvg E=5 persistent: skill
  +0.101, RMSE 0.0384 m/s, R² 0.932.**
- The 60-epoch centralized reference is unchanged from the M5 migration entry (+0.121).

### Reproducibility check: 60-epoch centralized residual

`configs/centralized_combined_residual_runbalanced_9ch_no006_60e.yaml` re-run with
`--output-root` into scratch on the M5 (534 s), using the S3 code with the relaxed
epoch check. Against the committed artifacts (`e62ab4b`), `test_run_predictions.csv`,
`training_history.csv`, `test_window_predictions.csv` and
`train_source_subject_means.csv` are **byte-identical**. The canonical baseline
reproduces exactly, and the `epochs_override_reason` change leaves it untouched. The
committed artifacts were not written.

### S5 scope (decided 2026-10-19)

**Every condition runs both tasks**, with the S3 winner setting (**FedAvg, E=5,
persistent client optimizer state**) for every federated and local-only run:

| Family | Conditions |
|---|---|
| Centralized | reference |
| K=2 | IID · natural cross-session · local-only |
| K=3 | channel ρ = 1.0 (= IID) · ρ = 0.33 · ρ = 0 · Dirichlet α = 0.1 · local-only |

**Every regression output reports**, as in `scripts/analyze_s3_regression_skill.py`:
overall skill and per-source skill (Test_2, 20251124), each with a run-level bootstrap
95% CI (10,000 resamples of test runs, fixed seed), plus overall skill without subject
007's test runs. S4 builds this into the consolidated regression metrics function so
it is emitted automatically.

**Fallback, decided before S5 runs:** if S5 is not complete by the morning of 31 Oct
2026, drop **regression** from the Dirichlet α=0.1 run first, then from ρ=0.33.
Classification for those runs still goes ahead.

---

## Session 3 — R1 fix: implementation (2026-10-18; sweep results to follow)

Branch `s3-r1-fix`. Covers `docs/review_response_plan.md` §2 R1. The winner rule,
`docs/s3_winner_rule.md`, was committed (`407f03f`) before any of this code was written.
`_run_flower_core` was edited in place, not refactored (cut in the runbook).

### Defect found: the regression client crashed at `local_epochs > 1`

`FlowerResidualClient.fit` accumulated the run UIDs it saw across *all* local epochs,
then asserted there were no duplicates. At E ≥ 2 every run appears once per epoch, so
every E > 1 config raised `"saw a run more than once in a local epoch"` in round 1. That
is four of the six sweep configs. The check now runs once per epoch: every epoch must see
each owned run exactly once. Classification had no such check and was unaffected. Test:
`test_regression_client_trains_more_than_one_local_epoch`.

### What was added

- **Config keys** (validated; defaults reproduce pre-S3 behaviour):
  - `federated.aggregation ∈ {FedAvg, FedAdam}`
  - `federated.client_optimizer_state ∈ {fresh, persistent}`
  - `training.lr_schedule ∈ {cosine_over_rounds, constant, cosine_restart_per_round}`
  - an optional `federated.server_optimizer` block for FedAdam
- **FedAdam** server hyperparameters are fixed at the pre-registered η 1e-2, β₁ 0.9,
  β₂ 0.99, τ 1e-3. The tracking strategy subclasses whichever base is configured, so the
  client-ordering fix and all aggregation checks apply to both.
- **Client optimizer persistence.** Each client saves its Adam `state_dict` to
  `<run>/<task>/client_optimizer_state/<client_id>.pt` and reloads it next round. Model
  parameters are still replaced by the global model, and the LR is reset from the
  schedule. The directory is wiped at run start. A missing file at round ≥ 2, or a file
  present before round 1, raises. At run end, the Adam step counters are checked against
  the step-budget formula.
- **Local-training instrumentation** in `local_client_fit_history.csv`: loss at the
  start and end of each client's local training (measured on an unshuffled loader, so it
  draws nothing from the training RNG), measured local steps, and Adam steps before and
  after each round.
- **Matched budget** in every run summary: FL total gradient steps = Σ_k ⌈n_k/B⌉ × E × R,
  checked against the steps the clients actually took, next to the centralized
  reference. Regression: IID K=3 is 1800·E and natural K=2 is 1680·E, against 1680
  centralized. Classification: centralized is 2340.
- **Failure record.** A regression run that fails the detector (override off) writes
  `degenerate_model_check_FAILED.json` and no `training_summary.json`.
  `DegenerateModelError` now carries the detector report.
- **Runner** `--set SECTION.KEY=VALUE`, which requires `--output-root`.
- **Sweep:** `src/redo_by_sara/r1_sweep.py` (the winner rule as code) and
  `scripts/run_r1_sweep.py` (driver). Preflight refuses to start unless the rule doc is
  committed and unmodified, `src/` `scripts/` `configs/` are clean, and the §6
  determinism test passes. Child runs execute under `caffeinate -i` on macOS.
- Fixed in passing: the regression summary's `client_aggregation_weight` was hard-coded
  to "(37/38/37)", which was wrong for every K=2 run. It is now derived from the
  partition.

### Verification

- **Defaults are bitwise unchanged.** New code at default settings, 2 rounds,
  `client_num_cpus: 4`, compared against the pre-S3 smoke runs on this machine. For IID
  K=3 and natural K=2, both tasks, every compared file (`global_train_history.csv`,
  `test_predictions.csv`, `test_window_predictions.csv`, `test_run_predictions.csv`) is
  byte-identical.
- **Persistent-mode determinism (rule §6):** two runs of natural K=2, E=2, persistent,
  2 rounds, give bitwise-identical final parameters, Adam state files, and client
  history.
- **FedAdam + persistent + E=2, 2 rounds, override off:** classification completed.
  Regression failed the detector as expected (standardized MSE 0.9999, skill −0.0011)
  and wrote only the failure record. In that run, client_0's local loss went
  1.5715 → 1.5715 over 20 local steps: the R1 mechanism, measured directly.
- `pytest`: **100 passed, 1 skipped** (cudnn, macOS). `-m "not requires_data"`: 85 passed.

**No reported number moved.** Sweep results go in the next entry.

---

## Machine migration — M5 environment and centralized baselines re-run (2026-10-18)

On `main`, between S2 and S3. The project moved to a fresh clone on an M5 MacBook Air.
All paper runs now happen there: CPU only, conda env `fl` (Python 3.11.15, torch 2.14.1,
`flwr 1.29.0`, `ray 2.51.1`, pinned to match every earlier run). CPU float results are
not bitwise identical across machines, so every number in the paper must come from this
one. Old-machine results are compared within tolerance, never bitwise. Recorded as a
working-agreement line in `CLAUDE.md`.

### Changes

- `client_num_cpus: 4 → 3` in `configs/combined_iid_flower_3clients_60r.yaml` and
  `configs/combined_natural_flower_2clients_60r.yaml`. At K=3, 3 × 3 = 9 of the M5's 10
  cores, leaving one for the driver. The value also sets each client's torch thread
  count, so it is fixed at 3 for every paper run, K=2 and K=3 alike.
- The canonical `.pt` window artifacts (gitignored) were rebuilt from `TestData/` with
  `--preprocess-only`.
- The three centralized baselines were re-run on the M5, overwriting their tracked
  outputs in `artifacts/centralized_combined_*`.
- `run_manifest.csv` (both artifacts): only the absolute `hdf5_path` / `speed_label_path`
  columns changed (`/home/coder/workspace/...` → the M5 path). Every data column is
  identical.

### Rebuild verification

All counts exact: 161 discovered → 15 excluded (subject 006) + 6 skipped → 140 usable →
112 train / 28 test; classification 1549 (1242/307); regression 1204 (966/238);
train/test run overlap 0. `pytest`: 68 passed, 1 skipped (`test_guardrails.py:67`, cudnn
flags — no cudnn on macOS; it ran on the old machine, hence S2's 69).

A 2-round federated smoke run (`--allow-degenerate`, into scratch) exited 0 for IID K=3
and natural K=2. It matched S2's smoke run: final standardized MSE 1.0000512 vs 1.000051
(K=3) and 1.0001157 vs 1.000114 (K=2). The K=2 difference in the 7th digit is
cross-machine float drift. That run still shows the unfixed R1 behaviour; it is not a
result.

### Numbers that moved

Tolerances were fixed before the new numbers were seen. Old values come from
`git show HEAD:artifacts/...` at `f05828f`.

| Metric | Old | New (M5) | Diff | Tolerance | Result |
|---|---|---|---|---|---|
| Run / window counts | 112/28, 966/238 | identical | 0 | exact | PASS |
| Subject-mean baseline R², run level | 0.9246404 | 0.9246404 | 0 (bit-equal) | exact | PASS |
| Subject-mean baseline RMSE (m/s) | 0.0405244 | 0.0405244 | 0 (bit-equal) | exact | PASS |
| Classification errors / 307 | 1 | 1 | 0 | 0–3 | PASS |
| Absolute regression R², run level | 0.812368 | 0.812753 | +0.000385 | ±0.02 | PASS |
| Residual regression R², run level | 0.933207 | 0.933771 | +0.000564 | ±0.01 | PASS |
| Residual skill vs subject mean | 0.113674 | 0.121158 | +0.007484 | ±0.05 | PASS |
| Residual RMSE (m/s), info | 0.038152 | 0.037990 | −0.000161 | — | — |
| Absolute RMSE (m/s), info | 0.063944 | 0.063878 | −0.000066 | — | — |
| Classification accuracy / macro-F1, info | 0.996743 / 0.997423 | identical | 0 | — | — |

The bit-identical subject-mean baseline (a lookup, no training) confirms that the data
pipeline reproduces exactly. Every trained number moved within its tolerance. **The
centralized reference numbers for the paper are now the M5 values:** residual skill
+0.121, R² 0.9338, RMSE 0.0380 m/s; absolute regression R² 0.8128, RMSE 0.0639 m/s;
classification 1 error in 307. Any document still quoting +0.11 / 0.933 / 0.0382 refers
to the old machine.

The new absolute-regression `training_summary.json` also carries
`distinct_predictions` / `num_predictions` keys that the old file predates (S1 detector).
No existing key changed meaning.

---

## Session 2 — K-parameterized partitioner, natural split, property assertions (2026-10-04/05)

Branch `s2-partitioner`. Covers `docs/review_response_plan.md` §1.1, §1.2, §3 item 3 and B7,
and closes R11. Four rulings amended the plan:

- the root `CLAUDE.md` was already updated;
- the Test_2 S→N relabel is cut (recorded in the docs only, artifacts not rebuilt);
- the R11 direction-keyed baseline is cut;
- the reference run is an end-to-end smoke test, not a bitwise reference.

### Numbers that moved

**No reference number in `CLAUDE.md` changed.** 161 discovered → 15 excluded + 6 skipped →
140 usable → 112 train / 28 test; classification 1549 (1242/307); regression 1204
(966/238); held-out class support `[69, 32, 43, 56, 25, 25, 27, 30]`. The canonical artifacts
were not rebuilt, and the split code is untouched.

**The K=3 IID client partition moved.** The new partitioner is a different algorithm, and
the old triple was an artifact of a brute-force search pinned to its own output (review
plan §3 item 3: "the exact triple no longer needs to be reproduced").

| K=3 IID, per client | before (retired) | after |
|---|---|---|
| runs | (37, 38, 37) | (37, 38, 37) |
| classification windows | (412, 418, 412) | (413, 421, 408) |
| regression (walking) windows | (321, 324, 321) | (321, 328, 317) |
| no-walking windows | (91, 94, 91) | (92, 93, 91) |
| 20251124 runs per client, N→S / S→N | 10 / 10 each | (9/10, 11/10, 10/10) |

The run counts land on the same triple by coincidence of the round-robin; the run
*assignment* differs. The old partitioner forced exactly 10 runs per direction per
client. The new one guarantees that every client holds every
`(source, subject, direction)` stratum and that totals are within one run, which is the
property §3 item 3 asks for.

**New partitions, measured and now pinned in tests:**

| scheme | K | runs | classification windows | regression windows | no-walking |
|---|---|---|---|---|---|
| IID | 2 | (56, 56) | (618, 624) | (480, 486) | (138, 138) |
| natural | 2 | (52, 60) | (587, 655) | (431, 535) | (156, 120) |

Natural: `client_0` = `test_2`, which holds subjects 001, 002, 003. `client_1` =
`testing_20251124`, which holds 003, 004, 005, 007, 008.

Test count: **39 → 69 passed** (`-m "not requires_data"`: 32 → 56).

### 1. Partitioner rewrite (`combined_iid_fl_partitioning.py`)

**Removed:**

- the nine `EXPECTED_*` outcome tables and `CLIENT_IDS`;
- `_base_counts_from_blocks`, `_remainder_assignments` (the `itertools.product`
  brute force), `_solve_subject_blocks` (the rank-minimizing DP) and `_assign_profiles`;
- the `num_clients != 3`, `seed != 4601` and `112/28` guards.

**New entry point:** `build_combined_partitions(..., scheme, num_clients, seed)`, with
`scheme ∈ {iid, natural}` and K ∈ {2, 3} (K > 3 is refused, per §1.2).
`build_combined_iid_partitions` is kept as a wrapper, so existing callers keep working.

- **IID** is a round-robin over speed-ordered blocks of K within each
  `(source, subject, direction)` stratum. Each complete block gives one run to every
  client, through a seeded permutation. Remainder runs go to distinct least-loaded
  clients, which keeps totals within one. A stratum with fewer than K training runs is
  refused, because no IID partition can cover it.
- **The docstring records why direction is a stratum key: signal coverage, not speed
  balance.** Speed is direction-invariant (R11), but direction reverses the order in which
  the corridor sensors are excited.
- **Natural** is one client per collection campaign, in sorted source order. The two
  campaigns are the same corridor 19.6 months apart, a cross-session split.

**Assertions are layered by scheme.**

- *All schemes:*
  - client run and window sets are disjoint and cover the training set exactly once;
  - no client owns a test run;
  - both tasks give every run the same owner;
  - every window a client owns maps back to one of its runs (measured from the window
    metadata);
  - subject 006 is absent.
- *IID only:*
  - run spread ≤ 1;
  - every client holds every subject and every stratum;
  - no client holds all runs of any subject;
  - every complete block is split one per client.
- *Natural only:* each client holds exactly one source.
- **Balance and coverage are deliberately not asserted for the natural split.** Its
  3-vs-5 subject coverage and 52/60 sizes are the heterogeneity under study.

The summary is now `schema_version: 2`. It gains `scheme`, `client_sources` (natural) and
`direction_stratification_reason` (IID), and its audit keys name the checks that actually ran.

### 2. Split integrity (§1.1)

New `guardrails.assert_split_integrity`. It measures, from the indices and the per-window
metadata:

- duplicate-free, disjoint index lists;
- every window in exactly one split;
- the metadata `split` agrees with the list each window is in;
- no run has windows on both sides.

It returns the measured counts.

- **Build time:** both artifact builders now call it, in place of their inline run-overlap
  checks. `train_test_run_overlap` in the summary is the measured value instead of a
  literal `0`. The value is unchanged, so the artifacts were not rebuilt; the code path
  runs on the next rebuild.
- **Load time:** called by the partitioner, the federated setup audit (which previously
  trusted the summary flag), the residual regression validator, and both centralized
  trainers.
- **Tests:** a deliberately leaky split (two windows of one run on opposite sides) fires
  it, both directly and through the partitioner. So do a window in both lists, a window in
  neither, and a metadata/index disagreement.

### 3. Federated runner (`combined_iid_flower.py`)

**Config validator.** The hard 3 clients / 60 rounds / 1 local epoch rejection is gone.
It now requires:

- `num_clients ∈ {2, 3}`;
- `num_rounds ≥ 1` and `local_epochs ≥ 1`;
- a new `federated.partition_scheme` (default `iid`), with `natural` requiring K=2.

FedAvg-only is kept; that is S3's job.

**Outcome pins replaced by properties:**

- `total_runs != 112` → equals the artifact's unique train runs;
- `(1242, 307)` / `(966, 238)` → the summary must agree with the measured split;
- `expected_support = [69, …]` → equals the support counted from the artifact's test
  targets;
- `!= 28` → equals the artifact's unique test runs.

All of those literals now live in `test_combined_iid_fl_partitioning.py`.

**Names.** Experiment names and the partition plot title are derived from K and the
scheme, and the "three-client" wording is gone.

**Configs.** `configs/combined_iid_flower_3clients_60r.yaml` gains
`partition_scheme: iid`. New `configs/combined_natural_flower_2clients_60r.yaml` (natural,
K=2, writing to `artifacts/combined_natural_flower_2clients_60r/`) has not been run at
60 rounds; S5 runs it.

The centralized-builder pins (140 / 112 / 28 / 1204) are untouched.

### 4. Documentation

- `docs/session0_findings.md` gains a "Session 2 findings" section:
  - **R11:** the cross-tab, the derivation, testNotes 96/96, the HDF5 field blank on
    91/96, and the physical centroid-timing check (95/96, all 65 Test_2 runs one sign =
    S→N). Speed-by-direction: max |Δ| 0.028 m/s, pooled p = 0.15.
  - **The corridor:** one corridor, two campaigns 19.6 months apart.
  - **Subject 003:** identity confirmed; the Test_2 HDF5 demographics are erroneous.
  - **Sensor 3:** HDF5 905 in vs testNotes 805 in, unresolved.
  - **Disposition table:** C2 → cut, B7 → done.
- `docs/sensor_layout.md` §3 gains the corridor coordinate table and records sensor 3 as
  unresolved (805 is consistent with monotone ordering). It drops "Confirmed against the
  layout" and records the Test_2 S→N finding. `docs/review_response_plan.md` §1.3 drops the
  same "confirmed against the drawing" claim.
- **No cross-building framing remains.** Fixed in `README.md:10` and `:49`, two
  comments in `test_guardrails.py`, and `legacy/README.md:22`.
- **Test_2 S→N, recorded but not applied.** The pipeline keeps `single_direction_unknown`. A
  new test proves a relabel would not move the seed-4601 split: the single-direction branch
  of `_direction_balanced_test_records` seeds from the stratum key alone.

### End-to-end smoke test (replaces the planned bitwise reference run)

Both configs ran with `--task both --rounds 2 --allow-degenerate` into scratch.
Production `artifacts/` was not written.

- Both runs, both tasks: exit 0.
- Partition files and the balance plot were written; the summaries report `iid` K=3
  (37/38/37) and `natural` K=2 (52/60).
- `test_evaluations_during_training: 0` in every run.
- **Regression at 2 rounds is still the unfixed R1 behaviour.** It is not a result:

| run | final standardized MSE | distinct residuals | skill vs subject mean | detector |
|---|---|---|---|---|
| IID K=3 | 1.000051 | 238 | +0.0014 | degeneracy ✓, skill ✗ → failed, override stamped |
| natural K=2 | 1.000114 | 238 | +0.0014 | degeneracy ✓, skill ✗ → failed, override stamped |

  The degeneracy gate passes after 2 rounds only because the zero-initialized head has
  barely moved off zero. The skill gate fails. Its skill-score condition (+0.0014 > 0)
  passes, but its standardized-MSE condition (needs < 0.95, got 1.0001) does not. That
  condition is what flags these runs invalid at this point, which is the case the
  detector's two-part skill gate exists for.
- Classification macro-F1 after 2 rounds is 0.046 in both runs, which is chance: the model
  has not trained yet.

**These numbers only show that the plumbing works for both schemes at both K.**

- `--verify-only` on both configs exits 0 and creates no output directory.
- `pytest` from the repo root: **69 passed**. `-m "not requires_data"`: **56 passed**,
  13 deselected.

## Session 1 — WP0 guardrails and legacy quarantine (2026-09-03/04)

Branch `wp0-guardrails`. `docs/review_response_plan.md` §3 items 1, 2, 4, 5, plus R8, R9,
C1 and B1–B6. §3 item 3, the partitioner, is Session 2.

### Numbers that moved

**No reference number in `CLAUDE.md` changed.** 161 discovered → 15 excluded + 6 skipped →
140 usable → 112 train / 28 test; classification 1549 (1242/307); regression 1204
(966/238); channels `[0,1,2,3,4,5,6,7,9]`; split seed 4601. All verified identical in the
end-to-end run, including the three partition files bit for bit.

**Federated metrics move by roughly 1e-7 relative**, from the client-ordering fix in §9 —
four orders of magnitude below anything reported, and the price of a pipeline that
reproduces itself. Nothing else moved: unifying the seeding, which was the change most
likely to disturb a number, provably did not. Round 0 and round 1 of the 60-round run are
bit-identical to the archived artifacts, and a seeding difference would surface at round 0,
before any client trains.

### 1. Legacy pipeline quarantined

39 files moved to `legacy/` with `git mv`: 6 modules (`preprocessing`, `sensor_non_iid`,
`iid_partitioning`, `training`, `config`, `federated`), 17 configs, 14 scripts, 2 test
files. Each raises `ImportError` on import *and* on direct execution; the three package
`__init__.py` files raise too. `legacy/README.md` records the A1–A6 evidence.

`MaskAwareSimpleCNN1D` moved out of `models.py` to `legacy/mask_aware_model.py` — the
class, not the file. `SimpleCNN1D` stays; four `combined_*` modules import it.

Ordering was forced by `federated.py:15`, a module-level `from .training import ...`.
Guarding `training.py` first would have propagated an exception into
`combined_iid_flower.py` and taken down the live pipeline. So `get_parameters` and
`set_parameters` were extracted to `src/redo_by_sara/parameters.py` first, verbatim, and
the two consumers repointed, before anything moved.

**Correction to the session brief.** It stated that `_set_random_seeds` had four copies
"some of which live in files that are about to move". Re-measured: all four are in
`combined_*` modules — `combined_centralized_classification.py`,
`combined_centralized_regression.py`, `combined_residual_run_regression.py`,
`combined_iid_flower.py` — and **none of them moved**. The seeding work had no ordering
dependency on the quarantine. There is a *fifth*, unnamed seeding site at
`training.py:154`, a bare `torch.manual_seed(seed)` with no `random`, `numpy` or cudnn
handling, and that one was resolved by quarantine rather than unification.

### 2. Stale artifacts deleted

`artifacts/raw_windows*` — 8 files, 221 MB. The four tracked `.summary.json` went via
`git rm`, so the A2 evidence stays recoverable in history. What they contained:

| File | Subjects | Sample shape | Split |
|---|---|---|---|
| `raw_windows` | 003–**006**–008 | `[20, 2000]` | 451/149/149 train/val/test |
| `..._sensor_non_iid_4c` | 003–**006**–008 | `[20, 2000]` | 240/0/85 |
| `..._4c_3ch_no006` | 003–008 less 006 | `[12, 2000]` | 208/0/71 |
| `..._4c_3ch_no006_3swin` | 003–008 less 006 | `[12, **1200**]` | 492/0/71 |

All four are non-compliant, not just the two containing subject 006: 20 or 12 channels
rather than 9, a validation split that does not exist in the design, and in the last case
1200 samples — a 3 s window, against the settled 5 s.

### 3. Guardrails — `src/redo_by_sara/guardrails.py`

**Seeding parity.** Four `_set_random_seeds` copies collapsed to one. The federated copy
silently omitted the cudnn determinism block, so federated and centralized runs were
seeded differently at the same seed. Now one implementation, six call sites,
`assert_determinism_flags()` at each of the four run entry points, and a `seeding` block
in every run summary.

Measured on this host: `torch.cuda.is_available()` is `False` and `device_count()` is 0,
but `torch.backends.cudnn.is_available()` is **`True`**, so the block does execute and
flips `cudnn.deterministic` `False → True` (`benchmark` was already `False`). No cudnn
kernel runs, so it should be numerically inert.

**`--verify-only` is read-only.** It previously wrote `setup_audit.json`, three partition
files and a PNG. All four writes now route through one `GuardedWriter`. Verified: run
against a non-existent output root, and the directory is not created, `git status` is
unchanged, and the run reports the seven paths it *would* have written.

**Degenerate-model detector**, at the end of all three regression paths. Two named gates,
because the four diagnostics do not mean the same thing:

| Gate | Diagnostics | Enforced on |
|---|---|---|
| `degeneracy` | distinct model outputs, prediction std | every regression path |
| `skill` | skill score, final standardized MSE | residual paths only |

`enforced_gates` is passed explicitly per run type and never inferred from the numbers.
Both gates are always *recorded*. Fail-closed; `--allow-degenerate` stamps
`allow_degenerate_override_used: true` and a warning string into the artifact, so a run
permitted through identifies itself as invalid.

The absolute-speed path is exempt from `skill` for a stated reason, not a threshold
judgement. Measured against its archived predictions (28 test runs):

| Baseline | Skill | Baseline RMSE |
|---|---|---|
| overall train mean | **+0.8129** | 0.1478 |
| subject mean | −1.1388 | 0.0437 |
| source/subject mean | **−1.4898** | 0.0405 |

Model RMSE 0.0639, 28 distinct predictions, std 0.136. It passes `degeneracy` and records
skill −1.49 without raising. Had the gates stayed merged, this **live** config would have
started failing on a model that trains perfectly well. Standardized MSE is recorded as
`null` there with a stated reason: it is defined against the train residual scale, which
does not exist on the absolute path.

**Two corrections to `docs/review_response_plan.md` §2 R2 fall out of this.** The table's
skill of −0.75 for condition A did not follow from its own RMSEs; the correct figure is
−1.49, confirmed two ways — `1 − (0.0639/0.0405)² = −1.49` and
`1 − (1−0.812)/(1−0.9246) = −1.494`. Separately, its **R² of 0.812 is exactly the skill
score against the overall train mean**, which is the sharpest statement of why R² against
total variance is the misleading framing: it credits the model for variance the subject
prior already explains. Both are now fixed in that document.

**The degeneracy gate reads the model's own output, not the reported prediction.** On the
residual paths the reported speed is `source_subject_train_mean + residual`, so a
completely constant model still yields one distinct value per source/subject stratum.
Measured on the archived collapse:

| Quantity | Distinct values |
|---|---|
| `predicted_residual_mps`, 238 windows | **1** |
| `predicted_speed_mps`, 238 windows | 8 |
| `predicted_speed_mps`, 28 runs | 8 |
| `(source, subject)` strata | 8 |
| `(source, subject, direction)` strata | 13 |

Eight is below the threshold of 10, so gating on the speed *would* have caught this
particular collapse — by a margin of two, on a quantity that counts strata in the baseline
lookup rather than anything the network did. A third building or a few more subjects puts
the count above 10 and a constant model passes silently. The residual is 1 regardless of
how many strata exist. `test_guardrails.py` pins both behaviours, including the 12-stratum
case where a speed-gated detector goes quiet.

**Test-set access accounting (R9).** `test_evaluation_counter` was a local dict that was
never incremented, so the guard reading it could not fail and
`"test_evaluations_during_training": 0` was asserted rather than measured. The finalizers
kept a separate counter the training-phase check never saw. `TestSetAccessGuard` is now
the only route to `artifact["test_indices"]`; one instance spans training and
finalization, and it is *sealed* during training, where an access both records itself and
raises.

### 4. R8 — sampling rate in the federated summaries

Already surfaced in the centralized `artifact_summary.json`, in
`setup_audit["conditions"]` and per run in `run_manifest.csv`. The federated per-task
`training_summary.json` was the one gap. Added a `sample_rates` block sourced from the
already-validated config and artifact summary, so this is not a sixth copy of the rate.

### 5. C1 — channel index conversion asserted

`assert_channel_index_conversion` at both 1-indexed → 0-indexed sites. Asserts the input
is exactly `[1,2,3,4,5,6,7,8,10]`, that the conversion is a strict shift to
`[0,1,2,3,4,5,6,7,9]`, that it is strictly increasing and duplicate-free, and that index 8
— the dropped y axis of the co-located position-8 unit — is absent. Purely additive.

### 6. pytest from the repo root

`pyproject.toml` (`pythonpath = ["src"]`, `norecursedirs` excluding the raising
`legacy/__init__.py`, `--strict-markers`) and `conftest.py`, which auto-skips
`requires_data` tests by path presence rather than behind an opt-in flag. **39 tests**, of
which **32 need no data**; the 7 data-dependent ones skip with a message naming the
missing path.

### 7. `--output-root` on the centralized runners

Not in the original plan; found while refusing to re-run a config to test a guard.
`scripts/run_combined_centralized_regression.py` had no way to write anywhere but
`artifacts/centralized_combined_regression_9ch_no006_walking_80_20_60e/` — live artifacts
that the federated run reads as `centralized_absolute_regression_summary`. Same class of
problem as `--verify-only`, and Session 2's reference run and Session 3's bitwise gate
both need to write into scratch.

All three centralized runners now take `--output-root`. On the two artifact builders the
*input* artifact is still read from its canonical location, because redirecting it would
silently trigger a rebuild from the 7.6 GB `TestData/` tree; combining it with
`--rebuild-artifact` is refused.

### 8. Documentation (B1–B6)

Six false claims in `README.md` and `PLAN.md`: 20 channels, a 60/20/20 train/val/test
split, validation-selected best checkpoints, a single-source dataset, and a stale repo
root. The quarantine also moved every file the README's structure, workflow and command
sections pointed at, so those were rewritten around the live `combined_*` stack. `PLAN.md`
keeps its original phases under a header marking it historical.

### 9. FedAvg aggregation made reproducible

Found while trying to compare the end-to-end run against the archived artifacts. Two runs
of **identical code at the same seed** did not reproduce each other, diverging at round 1
in the 9th significant digit and compounding until `test_predictions.csv` differed:

```
run A round 1: 1.95027232899566      run B round 1: 1.9502723228528305
run A round 2: 1.8180251060092698    run B round 2: 1.818069477012192
```

Cause: Ray returns the three `ClientAppActor`s in whatever order they finish,
`FedAvg.aggregate_fit` sums their weighted parameters in that order, and float addition is
not associative. Measured client arrival order, rounds 1–6, across two runs:

```
201 | 201 | 120 | 102 | 021 | 210
021 | 210 | 012 | 012 | 012 | 201
```

Fixed by sorting the client results by `client_id` before delegating to `FedAvg`. Verified:
two independent runs now produce bitwise-identical `global_train_history.csv`,
`local_client_fit_history.csv`, `test_predictions.csv` and `test_confusion_matrix.csv`.

Pre-existing, and unrelated to everything else in this session. Federated numbers move by
roughly **1e-7 relative** — four orders of magnitude below anything reported, and worth
paying, because the paper reports mean ± std over 3 initialization seeds and that is
meaningless if one seed does not reproduce itself.

### End-to-end verification

Full 60-round run, both tasks, into scratch. Production
`artifacts/combined_iid_flower_3clients_60r/` was never written — it holds the R1 evidence.

**The detector reproduced R1 as a measurement**, which is the point of the whole exercise:

```
distinct_predictions            : 1        (of 238 window residuals)
degeneracy_evaluated_on         : window predicted residual
prediction_std                  : 0.0
final_standardized_mse          : 1.0000407    (1.0 = predict zero residual)
skill_score_vs_subject_mean     : -0.0027581   (worse than the lookup table)
enforced_gates                  : ['degeneracy', 'skill']   -> both FAILED
allow_degenerate_override_used  : True
window_level                    : residual 1 distinct, speed 8 distinct
```

The artifact carries the warning string and identifies itself as invalid. Without
`--allow-degenerate` the run raises. R9 verified end to end alongside it:
`test_evaluations_during_training: 0`, with a non-empty access log showing exactly one
post-training touch, from the same guard object that was sealed for all 60 rounds.

**Comparison against the archived August artifacts:**

| | Result |
|---|---|
| `partitions/*` (3 files) | **identical** |
| `regression/train_source_subject_means.csv` | **identical** |
| `classification/test_predictions.csv`, `test_confusion_matrix.csv` | **identical** |
| Round **0**, both tasks | **identical** |
| Round **1**, both tasks | **identical** |
| Round 2 onward | diverges at the 9th significant digit, compounding |
| Final model tensors | 16/16 differ; max abs 0.358 (cls), 0.0187 (reg) |
| `training_summary.json` | 0 unexpected new keys, 0 removed, in both tasks |

Round 0 and round 1 being identical is what rules out the seeding change as a cause: a
seeding difference would appear at round 0, before any client trains. Everything past
round 1 is the client-ordering non-determinism described in §9 above.

### The archived August artifacts are retired as a bitwise reference

`artifacts/combined_iid_flower_3clients_60r/` was produced under non-deterministic client
ordering. It cannot be reproduced — not by this code, and not by the code that generated
it. It remains valid as **evidence of the R1 collapse**, which is what it is cited for
throughout `docs/session0_findings.md` and §2 R1, and those facts are order-independent:
one distinct residual, a flat standardized MSE, a negative skill score.

It must not be used as a **bitwise reference** for any future comparison. Session 2's
reference run, produced after the ordering fix, replaces it in that role and is the first
federated output in this project that can be reproduced exactly.

(The refactor-and-bitwise-gate that would have depended on such a reference has since been
cut from the session plan, so nothing now turns on this. The reproducibility fix stands on
its own: it is a precondition for the multi-seed results the paper reports.)

### Verification performed

- `pytest` from the repo root, no flags: **39 passed**, 0 collection errors, 0 warnings,
  `legacy/` not descended into despite its raising `__init__.py`.
- `pytest -m "not requires_data"`: **32 passed**, 7 deselected.
- With the artifacts hidden, the 7 `requires_data` tests skip with a message naming the
  missing path.
- `--verify-only` against a non-existent output root: directory not created, `git status`
  unchanged, and the run reports the 7 paths it would have written.
- Full 60-round federated run, both tasks, as above.
- Two independent 5-round runs, bitwise identical after the §9 fix.
- `--rebuild-artifact` combined with `--output-root` is refused rather than silently
  resolved.

### Open for Session 2

- §3 item 3: the partitioner rewrite, the `EXPECTED_*` outcome pins, and the config
  validator's hard-coded 3 clients / 60 rounds / 1 local epoch.
- C2: the subject list (7 sites), channel list (4) and sample rate (5) still have multiple
  sources of truth. The `sample_rates` block added here reads from config and the artifact
  summary rather than adding a sixth.
- B7: `combined_iid_fl_partitioning.py:8-9` still says the design is "intentionally fixed
  to the three-client, seed-4601 design". True today; false once K is a parameter.
