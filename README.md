# Federated learning for gait analysis from structural floor vibration

Two tasks on the same 5-second windows of raw floor-vibration signal:

1. **Classification**: who is walking. 8 (source, subject) identities plus a `no_walking`
   class, 9 classes.
2. **Regression**: walking speed.

Both are trained centralized, federated (Flower, FedAvg) and local-only, under IID and
non-IID partitions.

The claim under test is that federated learning across data silos is feasible and useful
without centralizing raw vibration data. The two silos are two collection campaigns in the
**same instrumented corridor, 19.6 months apart**. That makes this a cross-session split,
not two buildings. Cross-building generalization is untested and is future work.

**`CLAUDE.md` is the authoritative description of the data and the invariants.** Where
this README and `CLAUDE.md` disagree, `CLAUDE.md` is right and this file is a bug. Other
documents:
- `docs/review_response_plan.md`: the work plan;
- `docs/sensor_layout.md`: physical sensor placement;
- `CHANGES.md`: every change and every number that moved, newest first.

## Data

| Source | Subjects | True sampling rate | Resample to 400 Hz |
|---|---|---|---|
| `TestData/Test_2` (2024-04-06) | 001, 002, 003 | **1706.667 Hz** | `resample_poly(15, 64)` |
| `TestData/20251124_Testing` (2025-11-24) | 003, 004, 005, 007, 008 | ≈ 1651.61 Hz | `resample_poly(31, 128)` |

- **Every subject is a (source, subject) identity.** Subject IDs are numbered per campaign
  and never merged across campaigns: `test_2:003` and `testing_20251124:003` are two
  different people. Identity strings are `"{source}:{subject:03d}"`.
- **8 identities:** test_2:001, :002, :003; testing_20251124:003, :004, :005, :007, :008.
  They are 7 distinct people. The IDs are not contiguous.
- **One person was recorded in both campaigns:** `test_2:002` and `testing_20251124:003`.
  By lab decision they stay two identities, i.e. two classes. The link is recorded only in
  `configs/identity_links.yaml`, and only evaluation controls read it, never label logic.
- **Subject 006 is excluded from every experiment** (data collection problems). Deliberate.
- **The Test_2 sampling rate is overridden on purpose.** The file metadata says 1652 Hz,
  which was recorded incorrectly. The true rate is 1706.667 Hz. Using the metadata rate
  would time-warp every Test_2 window by +3.3%, and walking speed is derived from time.
- **Identity and source are fully collinear.** Every identity belongs to exactly one
  campaign, and a source-only classifier separates the two campaigns perfectly (see
  Results). This is a known confound.

161 discovered records → 15 excluded (subject 006) + 6 skipped → **140 usable runs**.

## Signal processing

- **Windows: 5 s, 1 s stride, 2000 samples at 400 Hz.** A settled design decision, not a
  tunable.
- **Channels `[1, 2, 3, 4, 5, 6, 7, 8, 10]` → 9 channels** covering 8 distinct hallway
  positions.
  - Sensors 1–7 are one unit each.
  - Channels 8 (x) and 10 (z) are two uniaxial units co-located at an eighth point;
    channel 9 (y) is dropped.
  - The other 11 of the 20 channels were not in the hallway where walking occurred.
  - **This selection was made a priori from physical sensor placement, not from model
    performance.** See `docs/sensor_layout.md`.
- Config channel indices are **1-indexed**; numpy needs 0-indexed. The conversion is
  asserted explicitly (`guardrails.assert_channel_index_conversion`), because a silent
  off-by-one would shift every channel and could not be detected downstream.
- Normalization statistics come from **train windows only**, applied lazily in
  `__getitem__`.

## Splits

- **80/20 train/test by complete run.** There is **no validation split** and no
  best-checkpoint selection. The final-epoch model is the reported model, and the held-out
  set is evaluated exactly once, after training.
- Split seed 4601 → **112 train / 28 test runs**.
- `run_uid = "{source}:{subject:03d}:{run:03d}"` is the indivisible unit. A run is never
  split across train/test or across federated clients: adjacent 5 s windows share 4 s of
  signal, so splitting a run would be direct leakage.
- The window-level split is therefore **not** exactly 80/20. Both counts are reported:

| Task | Windows | Train | Test |
|---|---|---|---|
| Classification (9 classes) | 1549 | 1242 | 307 |
| Regression (walking only) | 1204 | 966 | 238 |

## Experimental setup

- **Model:** `SimpleCNN1D`, 9 input channels, for every family and condition.
- **Federated setting:**
  - FedAvg, E = 5 local epochs, persistent client optimizer, 60 rounds;
  - chosen by the S3 sweep under a pre-registered rule (`docs/s3_winner_rule.md`).
- **Families:**
  - centralized;
  - FL;
  - local-only, where each client trains alone on its own partition.
- **Evaluation:** every model is scored on the same 28 test runs with all 9 channels.
- **Seeds:** split seed 4601 everywhere. Initialization seeds 4601 / 4602 / 4603 where a
  condition has three, otherwise 4601 only.
- **Regression target:** the centralized reference is the residual, run-balanced model.
  It predicts speed minus the train-run mean of the run's identity.

| Condition | Clients | What differs between clients |
|---|---|---|
| K=2 IID | 2 | Runs split at random; every client holds every identity |
| K=2 natural | 2 | One campaign per client (Test_2 vs 20251124) |
| K=3 ρ = 1.0 / 0.375 / 0 | 3 | IID runs, but each client sees only some sensor positions. ρ is the overlap of positions. Missing channels are masked, never sliced |
| K=3 Dirichlet α = 0.3 / 0.1 | 3 | Identity proportions drawn from Dirichlet(α); training runs per client 18 / 21 / 73 (α = 0.3) and 20 / 17 / 75 (α = 0.1) |

## Results

These are the current results: the relabel queue (32 of 32 jobs, 9 classes) plus the S5
regression results it kept. The source tables are in `CHANGES.md` ("Relabel — Results",
2026-11-10). Classification is run accuracy: a majority vote over each test run's windows,
counted over the 28 test runs, with a Wilson 95% interval. Regression is skill against the
lookup baseline, with a run-level bootstrap 95% interval.

### Classification (9 classes)

| Condition | Centralized / FL | Local-only (mean ± std over clients) |
|---|---|---|
| Centralized (3 seeds) | 28/28 [0.879, 1.000] on every seed | — |
| K=2 IID | 28/28 on every seed | 1.000 ± 0.000 |
| K=2 natural | 28/28 on every seed | 0.500 ± 0.051 (13/28 and 15/28) |
| K=3 ρ = 1.0 / 0.375 / 0 | 28/28 at every ρ | 0.988 ± 0.021 / 0.976 ± 0.041 / 0.976 ± 0.041 |
| K=3 Dirichlet α = 0.3 | 25/28 [0.728, 0.963] on every seed | 0.607 ± 0.270 (seed 4601) |
| K=3 Dirichlet α = 0.1 | 22/28 [0.605, 0.898] on every seed | 0.452 ± 0.297 |

- **Ceiling:** centralized and FL are 28/28 in every condition except Dirichlet.
- **Dirichlet FL errors:**
  - every error is a 20251124 run, and Test_2 is 13/13 on every model;
  - the misclassified identities are the ones missing from the largest client;
  - some of those runs are voted `no_walking`.
- **Local-only K=2 natural:** each client has seen only one campaign. It gets that
  campaign's runs right and the other campaign's runs wrong.
- **FL vs local-only:** FL matches or beats local-only in every condition.

### Regression (skill = 1 − MSE_model / MSE_lookup)

The lookup predicts each test run as the mean training speed of its identity, with RMSE
0.0405 m/s.

| Condition | Centralized / FL | Local-only (mean ± std over trained clients) |
|---|---|---|
| Centralized residual (3 seeds) | +0.098 ± 0.021 | — |
| K=2 IID | +0.172 ± 0.096 | +0.010 ± 0.358 |
| K=2 natural | per seed: −0.207, −0.123, collapsed | −0.726 ± 0.869 |
| K=3 ρ = 1.0 | +0.101 [−2.263, +0.480] | −0.125 ± 0.343 |
| K=3 ρ = 0.375 | +0.046, did not train | −0.329 ± 0.389 |
| K=3 ρ = 0 | collapsed | −0.073 ± 0.122 (2 of 3 clients) |
| K=3 Dirichlet α = 0.3 | −0.240 [−4.913, +0.476] | −0.227 ± 0.150 |
| K=3 Dirichlet α = 0.1 | +0.079 [−0.106, +0.226] | −0.140 ± 0.457 (2 of 3 clients) |

- **No model shows positive skill distinguishable from zero.** No interval has its lower
  bound above 0. Model RMSEs range from 0.034 to 0.062 m/s, against the lookup's 0.0405.
- **Outcome labels:** every regression run passes a trained gate before its numbers are
  used. Three models collapsed to a constant output:
  - FL K=2 natural seed 4603;
  - FL K=3 ρ = 0;
  - local-only Dirichlet α = 0.1 client_2.

  Two did not train: FL K=3 ρ = 0.375 and local-only K=3 ρ = 0 client_0. They are reported
  as measured, never dropped.

### Controls

- **Source probe:** the subject classifier's recipe, with the campaign as the label, scores
  1.000 on test windows. That includes the `no_walking` windows, so the campaigns are
  perfectly separable even with nobody walking. Identity accuracy can therefore not be
  read as person recognition independent of session.
- **Cross-session control:** a separate evaluation split trains on one campaign and tests
  on the same person in the other. It is described in `CHANGES.md`; its results are not
  part of the figure workbooks.

## Outputs

| Output | Built by | Contents |
|---|---|---|
| `artifacts/` (gitignored) | the runners and queues | Per-job results: `artifacts/s5/` (S5 queue), `artifacts/relabel/` (relabel queue, checks, re-score, `derived/`) |
| `excel/` | `scripts/make_excel_figures.py` | **Current.** One workbook per figure with native, editable Excel charts, plus `confusion_matrices.xlsx` and `tables.xlsx` |
| `figures/`, `tables/` | `scripts/make_paper_figures.py` | matplotlib figures and LaTeX tables. **Stale:** built 2026-10-27 from S5 (8 classes), not yet rebuilt from the relabel |

Each Excel workbook has three kinds of sheet:
- a `data` sheet with the numbers and 95% intervals;
- a `chart` sheet;
- a `notes` sheet with metric definitions, seeds, the source file of every number and the
  commit it was built from.

The confusion matrices cover all 49 classification models, per run (8 × 9) and per window
(9 × 9).

## The live pipeline

Every number comes from the `combined_*` stack and the modules around it in
`src/redo_by_sara/`:

| Module | Purpose |
|---|---|
| `combined_centralized_classification.py` | Artifact build + centralized subject ID |
| `combined_centralized_regression.py` | Centralized absolute-speed regression |
| `combined_residual_run_regression.py` | Centralized residual, run-balanced regression |
| `combined_iid_fl_partitioning.py` | Leakage-safe client partitioning: IID, natural, Dirichlet |
| `combined_iid_flower.py` | Flower federated and local-only runs, both tasks |
| `channel_availability.py` | Sensor-position masks for the K=3 ρ conditions |
| `identities.py` | (source, subject) identity labels |
| `regression_metrics.py` | Skill, lookup baseline, run-level bootstrap |
| `run_level_classification.py` | Run accuracy (majority vote), Wilson intervals |
| `guardrails.py` | Seeding, guarded writes, degenerate-model detector, test-set accounting |
| `models.py` | `SimpleCNN1D`, the only architecture used |
| `parameters.py` | `get_parameters` / `set_parameters` for Flower |
| `r1_sweep.py` | S3 federated-setting sweep |
| `s5_queue.py`, `relabel_queue.py` | Resumable experiment queues |
| `relabel_checks.py`, `relabel_rescore.py` | What the relabel leaves unchanged; re-score of saved S5 predictions |
| `source_probe.py`, `subject003_control.py` | Controls |
| `paper_figures.py`, `excel_figures.py` | Figures and tables from saved results (nothing trains) |

An older pipeline lives under `legacy/` and **must not be run, imported, or copied from**.
It raises on import. See `legacy/README.md` for what is wrong with it.

## Setup

The paper runs are CPU only, on an Apple M5: conda environment `fl` (Python 3.11), flwr
1.29.0, ray 2.51.1, `client_num_cpus: 3`. All paper numbers come from this machine, so
never compare bitwise against results from another machine.

```bash
conda create -n fl python=3.11
conda activate fl
pip install -r requirements.txt     # numpy scipy PyYAML torch matplotlib openpyxl h5py wandb flwr[simulation] pytest
```

The raw data (`TestData/`, about 7.6 GB) and `artifacts/` are not in git.

## Running it

All commands are run from the repository root with `conda activate fl`.

```bash
# Centralized runs
python scripts/run_combined_centralized_classification.py \
    --config configs/centralized_combined_9ch_no006_nowalking_80_20_60e.yaml
python scripts/run_combined_residual_run_regression.py \
    --config configs/centralized_combined_residual_runbalanced_9ch_no006_60e.yaml

# One federated condition, one task (configs/relabel_*.yaml hold every condition)
python scripts/run_combined_iid_flower.py --config configs/relabel_k2_natural.yaml --task classification

# Audit a federated setup without training and without writing anything
python scripts/run_combined_iid_flower.py --config configs/relabel_k2_natural.yaml --verify-only

# The full relabel queue (resumable; about 19 h of training)
python scripts/run_relabel_queue.py --dry-run         # plan, status and ETA
python scripts/run_relabel_queue.py                   # run every job not yet done
python scripts/run_relabel_queue.py --derived-only    # rebuild results_index.csv and derived/

# Figures and tables from saved results (nothing trains, nothing is written under artifacts/)
python scripts/make_excel_figures.py                  # → excel/
python scripts/make_paper_figures.py                  # → figures/, tables/
```

Useful flags:

- `--output-root DIR`: send results somewhere other than the live artifact directory.
  Use it for any exploratory run. On the two artifact-building runners, the input artifact
  is still read from its canonical location.
- `--verify-only`: audit only. Writes nothing at all, and reports what a real run would
  have produced.
- `--allow-degenerate`: let a regression run that fails an enforced health gate finish
  instead of raising.
  - The failure is still measured and recorded in the run's `degenerate_model_check`
    block, so artifacts produced this way identify themselves as invalid.
  - It is needed only for the known-collapsed pre-fix federated regression baseline.
- `--stop-by HH:MM` (queues): start only the jobs that will finish by then. Queue children
  run under `caffeinate -i`, so keep the machine open and on power.

## Tests

```bash
python -m pytest                          # 230 tests: 229 pass, 1 skips on macOS (cudnn)
python -m pytest -m "not requires_data"   # leaves out the 26 tests marked requires_data
```

- Tests marked `requires_data` need the gitignored `.pt` artifacts or `TestData/`.
- The figure and workbook tests skip themselves when the saved results under `artifacts/`
  are absent.
- Reference counts (140 runs, 112/28, window counts) are written into the tests as
  literals, not imported from the library.

## Guardrails

`src/redo_by_sara/guardrails.py`, all of it exercised by `test_guardrails.py`:

- **Seeding parity:** one `set_random_seeds`, including the cudnn determinism block, is
  used by federated and centralized runs alike. The resulting flags are asserted at run
  start and recorded in every run summary.
- **Guarded writing:** `--verify-only` is genuinely read-only.
- **Degenerate-model detector:** runs at the end of every regression run, with two gates.
  - `degeneracy` (distinct model outputs, relative output spread) is enforced everywhere.
  - `skill` (skill against the lookup, final standardized MSE) is enforced on the
    residual paths.
  - Both are always *recorded*. A regression number is used only if its run passes.
- **Test-set access accounting:** the held-out set is reachable only through
  `TestSetAccessGuard`, which is sealed during training. "The test set was untouched
  during training" is a measurement, not an assertion.

## Metrics

- **Skill score = 1 − MSE_model / MSE_baseline** is the primary regression metric. It is
  always reported alongside the baseline it is measured against.
- R² against total variance is misleading here. Roughly 92% of speed variance is explained
  by identity alone, so R² mostly credits that prior. In fact, R² against total variance
  is exactly the skill score against the *overall* train mean.
- **Classification:** run accuracy (majority vote, ties broken by the higher run-mean
  probability) with a Wilson 95% interval over the 28 test runs. Window accuracy is
  reported as secondary.

## Data processing details

### APDM labelling
Each window carries the average walking speed of its trial,
`speed = (left + right) / 2`, from the APDM gait exports in m/s. A usable run without a
speed label is an error, not a default.

### Resampling
Polyphase (`scipy.signal.resample_poly`) to 400 Hz, with the per-source ratios above.

### Windowing
5 s windows, 1 s stride, taken only within the valid walking range from
`runPeramiters.csv`. Each window carries source, subject, run index, direction and start
time.

### Normalization
Channel-wise mean and standard deviation from **training windows only**, applied as
`(x - channel_mean) / channel_std`.

## Quarantined legacy pipeline

`legacy/` holds the earlier pipeline: 6 modules, 17 configs, 14 scripts and 2 test files.
Every one raises `ImportError` on import or direct execution. It:
- reads the wrong sampling rate for Test_2;
- does not exclude subject 006;
- keys `run_uid` without the source;
- splits train/test *inside* a run;
- lets two clients own the same window;
- slices channels rather than masking them.

`legacy/README.md` has the details and the evidence.
