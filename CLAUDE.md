# CLAUDE.md — durable project context

Read this at the start of every session. These facts are authoritative. Any code, comment, config, or docstring that contradicts them is wrong and should be corrected.

## Project

Federated learning for gait analysis from structural floor vibration, targeting an IEEE conference paper (deadline: ~3 weeks from 2026-08-31). Two tasks on the same windows: **subject identification** (classification) and **walking speed estimation** (regression). The claim is that federated learning across buildings is feasible and useful under IID and non-IID conditions without centralizing raw vibration data.

Full work plan: `docs/review_response_plan.md`. Sensor detail: `docs/sensor_layout.md`.

## Data

| Source | Subjects | Sampling rate | Resample to 400 Hz |
|---|---|---|---|
| `TestData/Test_2` | 001, 002, 003 | **True fs = 1706.667 Hz** — the 1652 Hz in file metadata was recorded incorrectly; the override is deliberate and correct | `resample_poly(15, 64)` |
| `TestData/20251124_Testing` | 003, 004, 005, 006, 007, 008 | ≈ 1651.61 Hz | `resample_poly(31, 128)` |

- **Subject 006 is excluded from every experiment** (data collection problems). Deliberate, not a bug.
- **Subject 003 appears in BOTH sources, on different days.** This is a genuine cross-session, cross-building recording of the same person — a scientific asset, not duplication. Never deduplicate it.
- Effective subjects: **001, 002, 003, 004, 005, 007, 008 → 7 subjects.** IDs are not contiguous; never assume `range(1, 8)`.
- Subject identity and source are nearly collinear (003 is the only overlap). This is a known confound; subject 003 is the instrument for addressing it.

## Signal processing — fixed, do not ablate

- **Windows: 5 s, 1 s stride, 2000 samples at 400 Hz.** Settled design decision.
- **Channels: `[1, 2, 3, 4, 5, 6, 7, 8, 10]` → 9 channels covering 8 distinct hallway positions.** Sensors 1–7 are one unit each at seven points; channels 8, 9, 10 are three separate uniaxial units **co-located at an eighth point**, oriented x, y, z. Channel 9 (y) is dropped; channels 8 (x) and 10 (z) are kept and belong to the same position. Channel 8 is the only horizontal-axis channel. See `docs/sensor_layout.md`. The other 11 of 20 channels were not located in the hallway where walking occurred. **Selection was made a priori from physical sensor placement, not from model performance.**
- Config channel indices are **1-indexed** → numpy indices `[0..7, 9]`. Assert this conversion explicitly; a silent off-by-one is undetectable downstream.
- Normalization statistics come from **train windows only**, applied lazily in `__getitem__`.

## Invariants that must never break

1. **`run_uid = "{source}:{subject:03d}:{run:03d}"` is the indivisible atom.** A run is never split across train/test or across federated clients. With 5 s windows at 1 s stride, adjacent windows share 4 s of signal — splitting a run is direct leakage that invalidates every number in the paper.
2. `set(train_run_uids) ∩ set(test_run_uids) == ∅`, asserted at artifact build time and again at load time.
3. No `run_uid` appears in two clients; both tasks assign a given run to the same client.
4. IID partitions: every client holds ≥1 run from every subject, **and no client holds all runs of any subject**.
5. Channel-availability partitions are built by **masking a 9-channel input**, never by slicing — all clients must have identical model parameter shapes or FedAvg cannot average them. The partition unit is the **position** (8 of them), so channels 8 and 10 always travel together; the overlap ratio ρ is defined over positions.
6. Test data is never touched during training, and the counter that proves it must actually increment.

## Reference numbers (current pipeline)

161 discovered records → 15 excluded (subject 006) + 6 skipped → **140 usable runs**. Split seed 4601, 80/20 by whole run → **112 train / 28 test runs**. Classification: 1549 windows (1242/307). Regression (walking interval only): 1204 windows (966/238).

Note the window-level split is not exactly 80/20; report both run-level and window-level counts.

These belong in **test files as independent literals**, not as constants imported from the library modules.

## Known defect being fixed

Federated regression previously collapsed to a constant predictor: one distinct prediction across 238 test rows, `global_train_standardized_mse` flat at 1.0 for 60 rounds, test R² (0.9244) fractionally *below* the subject-mean baseline (0.9246), while the summary reported it as a near-match to centralized. Any regression run must pass the degenerate-model detector before its numbers are used.

## Working agreement

- Do **not** rewrite the preprocessing pipeline, the artifact schema, or the run-splitting logic. They are correct.
- Propose a plan before any change larger than a single function, and wait for approval.
- One work package per session; commit and `/clear` between them.
- Append a summary of what changed — and every number that moved — to `CHANGES.md`.
- Report skill score (1 − MSE_model / MSE_baseline) as the primary regression metric, always alongside the baseline.
