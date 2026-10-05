# Federated learning for gait analysis from structural floor vibration

Two tasks on the same 5-second windows of raw floor-vibration signal:

1. **Classification** — subject identification (7 subjects plus a `no_walking` class).
2. **Regression** — walking-speed estimation.
3. **Federated learning** — Flower simulation of the above across clients, under IID and
   non-IID conditions.

The claim under test is that federated learning across data silos is feasible and useful
without centralizing raw vibration data. The two silos are two collection campaigns in the
**same instrumented corridor, 19.6 months apart** — a cross-session split, not two
buildings. Cross-building generalization is untested and is future work.

**`CLAUDE.md` is the authoritative description of the data and the invariants.** Where
this README and `CLAUDE.md` disagree, `CLAUDE.md` is right and this file is a bug.
`docs/review_response_plan.md` holds the work plan, `docs/sensor_layout.md` the physical
sensor placement, and `CHANGES.md` the record of what has moved.

## The live pipeline

Every number in the paper comes from the `combined_*` modules in `src/redo_by_sara/`:

| Module | Purpose |
|---|---|
| `combined_centralized_classification.py` | artifact build + centralized subject ID |
| `combined_centralized_regression.py` | centralized absolute-speed regression |
| `combined_residual_run_regression.py` | centralized residual, run-balanced regression |
| `combined_iid_fl_partitioning.py` | leakage-safe client partitioning |
| `combined_iid_flower.py` | Flower federated runs, both tasks |
| `combined_iid_flower_plots.py` | figures for the federated runs |
| `guardrails.py` | seeding, guarded writes, degenerate-model detector, test-set accounting |
| `models.py` | `SimpleCNN1D`, the only architecture used |
| `parameters.py` | `get_parameters` / `set_parameters` for Flower |

An older pipeline lives under `legacy/` and **must not be run, imported, or copied from**.
It raises on import. See `legacy/README.md` for what is wrong with it.

## Data

| Source | Subjects | True sampling rate | Resample to 400 Hz |
|---|---|---|---|
| `TestData/Test_2` | 001, 002, 003 | **1706.667 Hz** | `resample_poly(15, 64)` |
| `TestData/20251124_Testing` | 003, 004, 005, 007, 008 | ≈ 1651.61 Hz | `resample_poly(31, 128)` |

- The 1652 Hz in the Test_2 file metadata was **recorded incorrectly**. The override to
  1706.667 Hz is deliberate. It time-warps every Test_2 window by +3.3% relative to the
  metadata rate, and walking speed is a time-derived quantity, so this matters.
- **Subject 006 is excluded from every experiment** (data collection problems). Deliberate.
- **Subject 003 appears in both sources, 19.6 months apart.** That is a genuine
  cross-session recording of one person and a scientific asset. It is never
  deduplicated.
- Effective subjects: **001, 002, 003, 004, 005, 007, 008 → 7**. The IDs are not
  contiguous.

161 discovered records → 15 excluded (subject 006) + 6 skipped → **140 usable runs**.

## Signal processing

- **Windows: 5 s, 1 s stride, 2000 samples at 400 Hz.** A settled design decision, not a
  tunable.
- **Channels `[1, 2, 3, 4, 5, 6, 7, 8, 10]` → 9 channels** covering 8 distinct hallway
  positions. Sensors 1–7 are one unit each; channels 8 (x) and 10 (z) are two uniaxial
  units co-located at an eighth point, and channel 9 (y) is dropped. The other 11 of the
  20 channels were not in the hallway where walking occurred. **This selection was made a
  priori from physical sensor placement, not from model performance.** See
  `docs/sensor_layout.md`.
- Config channel indices are **1-indexed**; numpy needs 0-indexed. The conversion is
  asserted explicitly (`guardrails.assert_channel_index_conversion`), because a silent
  off-by-one shifts every channel and is undetectable downstream.
- Normalization statistics come from **train windows only**, applied lazily in
  `__getitem__`.

## Splits

- **80/20 train/test by complete run.** There is **no validation split** and no
  best-checkpoint selection; the final-epoch model is the reported model, and the held-out
  set is evaluated exactly once, after training.
- Split seed 4601 → **112 train / 28 test runs**.
- `run_uid = "{source}:{subject:03d}:{run:03d}"` is the indivisible unit. A run is never
  split across train/test or across federated clients: adjacent 5 s windows share 4 s of
  signal, so splitting a run would be direct leakage.
- The window-level split is therefore **not** exactly 80/20. Both counts are reported:

| Task | Windows | Train | Test |
|---|---|---|---|
| Classification | 1549 | 1242 | 307 |
| Regression (walking only) | 1204 | 966 | 238 |

## Running it

All commands from the repository root, using the `torch311` environment — it is the only
one with `flwr`, `h5py`, `scipy` and `matplotlib`.

```bash
PY=/home/coder/conda/envs/torch311/bin/python

# Centralized subject identification (builds the artifact on first run)
$PY scripts/run_combined_centralized_classification.py \
    --config configs/centralized_combined_9ch_no006_nowalking_80_20_60e.yaml

# Centralized absolute-speed regression
$PY scripts/run_combined_centralized_regression.py \
    --config configs/centralized_combined_regression_9ch_no006_walking_80_20_60e.yaml

# Centralized residual, run-balanced regression
$PY scripts/run_combined_residual_run_regression.py \
    --config configs/centralized_combined_residual_runbalanced_9ch_no006_60e.yaml

# Federated, both tasks
$PY scripts/run_combined_iid_flower.py \
    --config configs/combined_iid_flower_3clients_60r.yaml --task both

# Audit the federated setup without training and without writing anything
$PY scripts/run_combined_iid_flower.py \
    --config configs/combined_iid_flower_3clients_60r.yaml --verify-only
```

Useful flags:

- `--output-root DIR` — send results somewhere other than the live artifact directory.
  Use it for any exploratory run. On the two artifact-building runners the input artifact
  is still read from its canonical location.
- `--verify-only` — audit only. Writes nothing at all, and reports what a real run would
  have produced.
- `--allow-degenerate` — let a regression run that fails an enforced health gate finish
  instead of raising. The failure is still measured and is stamped into the run's
  `degenerate_model_check` block, so artifacts produced this way identify themselves as
  invalid. Needed only for the known-collapsed pre-fix federated regression baseline.

## Tests

```bash
/home/coder/conda/envs/torch311/bin/python -m pytest              # 39 tests
/home/coder/conda/envs/torch311/bin/python -m pytest -m "not requires_data"   # 32, no data needed
```

Tests marked `requires_data` need the gitignored `.pt` artifacts or the 7.6 GB `TestData/`
tree, and skip automatically when those are absent.

## Guardrails

`src/redo_by_sara/guardrails.py`, all of it exercised by `test_guardrails.py`:

- **Seeding parity** — one `set_random_seeds` including the cudnn determinism block, used
  by federated and centralized runs alike, with the resulting flags asserted at run start
  and recorded in every run summary.
- **Guarded writing** — `--verify-only` is genuinely read-only.
- **Degenerate-model detector** — runs at the end of every regression run. Two gates:
  `degeneracy` (distinct model outputs, prediction std) is enforced everywhere; `skill`
  (skill score vs the source/subject-mean baseline, final standardized MSE) is enforced on
  the residual paths, where it applies. Both are always *recorded*.
- **Test-set access accounting** — the held-out set is reachable only through
  `TestSetAccessGuard`, which is sealed during training. "The test set was untouched
  during training" is a measurement, not an assertion.

## Metrics

Report **skill score = 1 − MSE_model / MSE_baseline** as the primary regression metric,
always alongside the baseline it is measured against. R² against total variance is
misleading here: roughly 92% of speed variance is explained by subject identity alone, so
R² mostly credits the subject prior. R² against total variance is exactly the skill score
against the *overall* train mean, which makes the point precisely.

## Data processing details

### APDM labelling
Each window carries the average walking speed of its trial,
`speed = (left + right) / 2`, from the APDM gait exports in m/s. A usable run without a
speed label is an error, not a default.

### Resampling
Polyphase (`scipy.signal.resample_poly`) to 400 Hz, per-source ratios above.

### Windowing
5 s windows, 1 s stride, taken only within the valid walking range from
`runPeramiters.csv`. Each window carries source, subject, run index, direction and start
time.

### Normalization
Channel-wise mean and standard deviation from **training windows only**, applied as
`(x - channel_mean) / channel_std`.

## Quarantined legacy pipeline

`legacy/` holds the earlier pipeline: 6 modules, 17 configs, 14 scripts and 2 test files.
Every one raises `ImportError` on import or direct execution. It reads the wrong sampling
rate for Test_2, does not exclude subject 006, keys `run_uid` without the source, splits
train/test *inside* a run, lets two clients own the same window, and slices channels
rather than masking them. `legacy/README.md` has the details and the evidence.
