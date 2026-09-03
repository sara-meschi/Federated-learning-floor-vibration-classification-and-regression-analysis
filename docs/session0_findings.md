# Session 0 findings — contradictions with CLAUDE.md

Produced in Session 0 (orientation, no code changes) per `docs/claude_code_sessions.md`.
Referenced from `CLAUDE.md` § "The legacy pipeline is quarantined".

**Scope.** Everything here contradicts the authoritative "Data" or "Signal processing" sections of
`CLAUDE.md`, or the invariants that follow from them. Line numbers are as of Session 0 and will
drift once Session 1 lands.

**What is *not* wrong.** The `combined_*` stack behind every number in the paper is compliant.
Verified against `run_manifest.csv` and `artifact_summary.json`: 161 discovered records → 15 excluded
(subject 006) + 6 skipped → 140 usable → 112 train / 28 test runs; 1549 classification windows
(1242/307); 1204 regression windows (966/238); `selected_channels_zero_based: [0,1,2,3,4,5,6,7,9]`.
Every item below is in the **legacy** stack, its 17 configs, or the docs.

---

## Disposition summary

| Item | Resolved by | How |
|---|---|---|
| A1 no Test_2 fs override | Session 1 | quarantine |
| A2 subject 006 not excluded | Session 1 | quarantine + delete stale `artifacts/raw_windows*` |
| A3 `run_uid` without `source_id` | Session 1 | quarantine |
| A4 within-run train/test split | Session 1 | quarantine |
| A5 window replicated across clients | Session 1 | quarantine |
| A6 out-of-hallway channels, slicing, K=4 | Session 1 | quarantine; superseded by Session 5 |
| B1–B6 stale README / PLAN claims | Session 1 | doc fixes |
| B7 stale partitioner docstring | Session 2 | rewrite with the K-parameterized partitioner |
| C1 unasserted channel index conversion | Session 1 | add the assertion |
| C2 multiple sources of truth | Session 2 | one constants module |
| C3 masking design | Session 1 | quarantine the wrong approach; decision recorded in `CLAUDE.md` |
| C4 missing repo infrastructure | Sessions 1 and 8 | see item |

---

## Severity A — would produce wrong numbers if run today

### A1 — No Test_2 sampling-rate override in the legacy path

- `src/redo_by_sara/preprocessing.py:269` — `original_rate = _get_general_parameter(general_parameters, "fs")`
  reads the HDF5 metadata value (1652 Hz) directly
- `src/redo_by_sara/preprocessing.py:273` → `_resample_trials` at `:126-130`

No override exists anywhere in this path. Currently masked only because all 17 legacy configs set
`dataset_root: TestData/20251124_Testing` (line 6 of each). Point one at `Test_2` and every window is
time-warped by +3.3% — and walking speed is a time-derived quantity, so every label would be wrong
relative to the signal.

### A2 — Subject 006 is not excluded

Assigned to a client in five configs:

- `configs/fl_classification.yaml:55` and `configs/fl_regression.yaml:55` — `"1": ["005", "006"]`
- `configs/fl_semi_non_iid_classification.yaml:56`,
  `configs/fl_semi_non_iid_classification_15r_3e.yaml:56`,
  `configs/fl_semi_non_iid_regression.yaml:56` — `"1": ["005", "006", "007", "008"]`

These same configs also omit subjects 001 and 002 entirely.

Root cause: `preprocessing.py` contains **zero** exclusion logic (0 matches for `exclud`).

**This is materialized on disk, not hypothetical.** `artifacts/raw_windows.summary.json` reports
`subjects: ['003','004','005','006','007','008']`, `sample_shape: [20, 2000]`, and a 451/149/149
train/val/test window split. `artifacts/raw_windows_sensor_non_iid_4c.summary.json` likewise contains
006 and all 20 channels. Session 1 deletes these.

The four `fl_sensor_non_iid_*` configs *do* exclude 006 correctly (`exclude_subjects: ["006"]`,
line 40 of each) via `src/redo_by_sara/sensor_non_iid.py:269`.

### A3 — `run_uid` keyed without `source_id`

- `src/redo_by_sara/preprocessing.py:165-237` `_assign_subject_run_splits`, especially `:176-177`
  and `:227` — groups on `(subject_id, run_index)`
- `src/redo_by_sara/iid_partitioning.py:39-41` and `:116-119`
- `src/redo_by_sara/federated.py:115`, `:155`

Subject 003 exists in both sources, so `003:run_000` from `Test_2` and from `20251124_Testing`
collide. Contradicts the canonical atom `{source}:{subject:03d}:{run:03d}` (invariant 1). Latent only
because these paths load one source at a time.

### A4 — Train/test split *inside* a run

- `src/redo_by_sara/sensor_non_iid.py:162-194` `_split_raw_run_indices`, especially `:186-193` —
  splits a single run at a time index into train `(raw_start, train_end)` and test
  `(test_start, raw_end)`
- Purge gap defaults to **zero**: `src/redo_by_sara/sensor_non_iid.py:266`, and all four sensor
  configs set `purge_gap_seconds: 0.0` explicitly at line 39

Windows do not literally overlap — there is an interval check at `:235-261` — but train and test
windows come from the same walking pass with the same speed label. This is the exact leakage
invariant 1 exists to prevent.

### A5 — The same window is owned by two clients

- `src/redo_by_sara/federated.py:136` — `shared_subjects` = subjects with more than one owner
- `src/redo_by_sara/federated.py:157-158` —
  `for owner_client_id in owners: partitions[owner_client_id].append(index)`

Replicates every index of a shared subject into each owning client. Violates invariant 3. Exercised
by the `fl_semi_non_iid_*` configs, where subjects 007 and 008 are shared across both clients.

### A6 — Sensor non-IID uses out-of-hallway channels and slices rather than masks

- `selected_sensors` at line 28 of `configs/fl_sensor_non_iid_classification_4c_3ch_no006.yaml`,
  `configs/fl_sensor_non_iid_classification_4c_3ch_no006_3swin.yaml`, and
  `configs/fl_sensor_non_iid_regression_4c_3ch_no006.yaml`:
  `["1","2","4","12","13","3","5","6","14","7","8z","15"]`.
  Sensors **12, 13, 14, 15** are channels 16–19, marked *not in the hallway* by
  `docs/sensor_layout.md` §1. Only `8z` (channel 10) is used; channel 8 (`8x`, the only
  horizontal-axis channel) is absent.
- `configs/fl_sensor_non_iid_classification_4c.yaml:28` uses all 20 sensors with
  `sensors_per_client: 5`
- `num_clients: 4` at line 57 or 58 of each — §1.2 forbids exceeding K=3
- Slicing mechanism: `src/redo_by_sara/sensor_non_iid.py:53-128`, which *validates* that clients hold
  disjoint sensor sets, and `:131-138` `_sensor_channel_positions`

Violates invariant 5 (masking, not slicing) and the a-priori channel selection. Superseded by
Session 5.

---

## Severity B — documentation stating the wrong facts

| # | Location | Claims | Truth |
|---|---|---|---|
| B1 | `README.md:14` | "the default baseline uses all 20 physical channels" | 9 channels `[1..8, 10]`. Live at `configs/classification.yaml:28` and `configs/regression.yaml:28`; materialized as `sample_shape: [20, 2000]` in `raw_windows.summary.json` |
| B2 | `README.md:54-55` | "assign subjects `003`-`008` across 3 clients" | includes 006, omits 001 and 002 |
| B3 | `README.md:95` | "Default ratios: 60% train, 20% val, 20% test" | 80/20 by whole run, no validation split. Live at `configs/classification.yaml:32-34` and `configs/regression.yaml:32-34` |
| B4 | `README.md:100` | "the current thesis dataset layout in `../TestData/20251124_Testing`" | two-source combined dataset |
| B5 | `README.md:39` | rooted at `/home/coder/workspace/ENGR859-final-project` | stale path |
| B6 | `PLAN.md:22`, `:34`, `:35` | "explicit train, validation, and held-out test splits"; "save the best model by validation metric" | no validation split; a single post-training test evaluation |
| B7 | `src/redo_by_sara/combined_iid_fl_partitioning.py:8-9` | "intentionally fixed to the three-client, seed-4601 design" | Session 2 makes K a real parameter |

---

## Severity C — compliant, but missing a guarantee CLAUDE.md asks for

### C1 — 1-indexed → 0-indexed channel conversion is never asserted

- `src/redo_by_sara/combined_centralized_classification.py:433`
- `src/redo_by_sara/combined_centralized_regression.py:171`

Both are correct, and the result *is* recorded in the artifact summary
(`selected_channels_zero_based: [0,1,2,3,4,5,6,7,9]`), but never asserted. `CLAUDE.md` line 27
requires the explicit assertion, because a silent off-by-one would shift every channel and is
undetectable downstream.

The same unasserted conversion appears in the legacy path at `preprocessing.py:249` and
`sensor_non_iid.py:147`; those go to quarantine.

### C2 — Multiple sources of truth

`docs/review_response_plan.md` §5b names this the one duplication worth fixing before submission.

- **Subject list, 7 sites:** `combined_iid_fl_partitioning.py:32`,
  `combined_centralized_regression.py:259`, `combined_centralized_classification.py:423`,
  `combined_iid_flower.py:634`, `:1132`, `:1346`, `:1471`
- **Channel list, 4 sites:** `combined_centralized_classification.py:84` (message at `:86`),
  `combined_iid_flower.py:67-78`, `:1470`, and `in_channels=9` hardcoded at
  `combined_residual_run_regression.py:280`
- **Sample rate, 5 sites:** `combined_iid_flower.py:81`, `:1466`,
  `configs/combined_iid_flower_3clients_60r.yaml:36`,
  `configs/centralized_combined_9ch_no006_nowalking_80_20_60e.yaml:11`,
  `configs/centralized_combined_regression_9ch_no006_walking_80_20_60e.yaml:12`

### C3 — Masking belongs in the dataset, not the model

Recorded so the wrong approach is not re-derived in Session 5.

- `src/redo_by_sara/models.py:36-38` — `MaskAwareSimpleCNN1D` calls
  `super().__init__(in_channels=signal_channels * 2, …)`, giving an 18-channel input by
  concatenating a mask-indicator channel
- `src/redo_by_sara/training.py:61` — raises
  `"Mask-aware model is currently implemented for classification only."`

**Both are quarantined and neither is a blocker, because Session 5 must not use either.** Each
client's mask is *static*, so the model gains nothing from mask awareness, and a 9-channel input is
precisely what keeps parameter shapes identical across clients — which is the whole reason FedAvg
can average them. The mask goes in the dataset's `__getitem__`, applied **after** normalization, so a
masked channel reads as the training mean ("no information") rather than an extreme value.

`src/redo_by_sara/combined_residual_run_regression.py:280` hardcodes `in_channels=9`. That value is
**correct and must stay 9**; it is not a shape to change.

### C4 — Missing repo infrastructure

Absent: `CHANGES.md`, `PRIVACY_NOTES.md`, `conftest.py`, `pyproject.toml`, conda environment file.
`pytest` is missing from `requirements.txt` **and from both conda envs** (`pytorch`, `torch311`).
Five of seven test files have no `sys.path` bootstrap and fail to import from the repo root.

Also absent from the codebase entirely, all still to be built: local-only run mode, FedAdam/FedYogi,
skill score, degenerate-model detector, Wilson interval, bootstrap CI, Dirichlet partitioner,
multi-seed driver.

---

## Import graph — read before doing Session 1

### The question that matters, answered plainly

**No `combined_*` module imports `training.py`. No `combined_*` module imports
`MaskAwareSimpleCNN1D`. The Session 1 quarantine list does not need revising.**

Verified exhaustively, including lazy and function-local imports: the only occurrences of the token
`training` anywhere in the `combined_*` stack are local variable names, YAML config keys, and string
literals in summaries. `MaskAwareSimpleCNN1D` appears in exactly one file outside `models.py`, and
that file is `training.py` itself.

The entire live pipeline reaches outside itself for exactly **two** things:

| Live import | Target | Status |
|---|---|---|
| `combined_iid_flower.py:34`, `combined_centralized_classification.py:22`, `combined_centralized_regression.py:28`, `combined_residual_run_regression.py:22` | `models.SimpleCNN1D` | stays put |
| `combined_iid_flower.py:33` | `federated.get_parameters`, `federated.set_parameters` | **the one coupling to a quarantine target** |

`combined_iid_fl_partitioning.py` and `combined_iid_flower_plots.py` have no intra-package imports
at all.

### Consumers of each quarantine target

| Target | Imported by | Live consumer? |
|---|---|---|
| `preprocessing.py` | `sensor_non_iid.py:144`, `:282` (function-local); `scripts/analyze_sensor_snr.py:25`; `scripts/preprocess_raw.py:14` | none |
| `sensor_non_iid.py` | `scripts/run_flower_sensor_non_iid.py:30`, `plot_sensor_regression_predictions.py:24`, `preprocess_sensor_non_iid.py:14`, `plot_classification_confusion.py:22`; `test_sensor_non_iid_partitioning.py:19` | none |
| `iid_partitioning.py` | `scripts/run_flower_iid.py:32` | none |
| `training.py` | `federated.py:15` (**module level**); `scripts/train_baseline.py:22` | none directly — see hazard below |
| `MaskAwareSimpleCNN1D` (`models.py:36-39`) | `training.py:11` | none |
| `federated.py` | `iid_partitioning.py:9`; **`combined_iid_flower.py:33`**; `test_combined_iid_flower.py:15`; 9 legacy scripts | **yes**, for two symbols |
| `config.py` | `preprocessing.py:16`, `sensor_non_iid.py:15`; 13 legacy scripts; `test_semi_non_iid_partitioning.py:18` | none |

### Four consequences for Session 1

**1. Ordering hazard — `federated.py` imports `training.py` at module level.**
`src/redo_by_sara/federated.py:15` is `from .training import EvalResult, create_model`. An
import-time guard on `training.py` therefore propagates through `federated.py` and breaks
`combined_iid_flower.py:33`, taking down the live pipeline. `get_parameters` (`federated.py:304`) and
`set_parameters` (`:308`) are the only symbols the combined stack needs.
**Extract those two into a module that does not import `training.py`** — a new
`src/redo_by_sara/parameters.py` — and update `combined_iid_flower.py:33` and
`test_combined_iid_flower.py:15` *before* adding any guard. This constrains the order of Session 1's
work, not its scope.

**2. `MaskAwareSimpleCNN1D` shares a file with `SimpleCNN1D`.**
Both live in `src/redo_by_sara/models.py`; `SimpleCNN1D` stays and is imported by four `combined_*`
modules. Move or delete the class at `models.py:36-39` — not the file. `training.py:11` imports it,
so the two travel together.

**3. `config.py` is legacy-only and is not on the runbook's list.**
No `combined_*` module imports it; all 17 consumers are legacy modules and legacy scripts. It is a
quarantine candidate.

**4. Fourteen scripts and two test files consume the quarantined modules** and become dead imports
unless they move too: `scripts/preprocess_raw.py`, `preprocess_sensor_non_iid.py`,
`train_baseline.py`, `run_flower.py`, `run_flower_iid.py`, `run_flower_sensor_non_iid.py`,
`analyze_regression_splits.py`, `analyze_sensor_snr.py`, `plot_sensor_regression_predictions.py`,
`plot_classification_confusion.py`, `plot_regression_subject_predictions.py`,
`compare_round_confusion.py`, `log_iid_flower_wandb_plots.py`, `log_fl_comparison_wandb.py`; and
`test_semi_non_iid_partitioning.py`, `test_sensor_non_iid_partitioning.py`.

### Knock-on effect: quarantine shrinks Session 8's dedupe list

| Helper | Copies today | Live after quarantine |
|---|---|---|
| `_as_index_list` | 3 (`sensor_non_iid.py`, `iid_partitioning.py`, `federated.py`) | **0** |
| `_confusion_matrix` | 5 | **1** (`combined_centralized_classification.py:688`) |
| `_resolve_path` | 3 | **2** (the `config.py` copy goes) |
| `_write_rows` | 3 | 3 — all in `combined_*` |
| actual-vs-predicted scatter | 3 | 3 — all in `combined_*` |

`docs/review_response_plan.md` §4 line 257 still says 4 copies of the confusion renderer and 4 of
`_as_index_list`; the measured counts are 5 and 3. The runbook corrects both.

---

## Appendix — R1 verification, for the record

Independently measured from the artifacts, confirming the corrected diagnosis now in
`docs/review_response_plan.md` §2 R1.

| Quantity | Federated | Centralized |
|---|---|---|
| Distinct predicted residuals, 238 test rows | **1** (`-0.0002014615893131122`) | 238 |
| `global_train_standardized_mse` | r0 0.999999992 → r60 1.000040599 (range 1.74e-4) | e0 1.000000 → e60 0.693081 |
| Run-level R² (baseline 0.92464) | 0.92443 | 0.93321 |
| Skill score, run level, n=28 | **−0.0028** | **+0.1137** |
| Optimizer | fresh Adam per client per round, ~10 steps (`combined_iid_flower.py:404`) | one persistent Adam, ~1680 steps (`combined_residual_run_regression.py:685`) |

Per-client `local_online_standardized_mse` is flat across all 60 rounds — client_0 0.7383→0.7383,
client_1 0.8365→0.8349, client_2 1.4350→1.4314 — so the clients are not learning locally and the
failure is upstream of aggregation.

Federated `predicted_speed_mps` takes exactly 8 distinct values, one per source/subject train mean,
each shifted by −0.0002014: the model is the lookup table minus 0.2 mm/s.
