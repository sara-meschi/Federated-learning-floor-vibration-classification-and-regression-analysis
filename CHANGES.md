# CHANGES

What changed, and every number that moved as a result. Newest first.

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
