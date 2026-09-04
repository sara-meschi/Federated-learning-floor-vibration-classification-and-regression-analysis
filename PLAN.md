# Plan

> **Historical.** This describes the original ENGR 859 rewrite of the raw-vibration code,
> written before the project became an IEEE conference submission. It is kept for
> provenance. **`CLAUDE.md` and `docs/review_response_plan.md` are authoritative**; where
> this file disagrees with them, they are right.
>
> Corrected 2026-09-04: Phase 2 and Phase 4 described a train/validation/test split and
> best-checkpoint selection by validation metric. Neither exists. The pipeline splits
> 80/20 train/test by whole run and evaluates the held-out set exactly once, after the
> final epoch.

## Goal

Rewrite the raw-vibration portion of the thesis code in a simpler ENGR 859 style, then
build a clean baseline model path for both regression and classification.

## Phase 1: Data Understanding

- Read the HDF5 vibration files directly.
- Read the run parameter CSV to know which time range is valid walking data.
- Read APDM walking-speed labels from either export format used in this repo.
- Visualize a few raw sensor windows to confirm scale, noise, and timing.

## Phase 2: Preprocessing

- Use the sensor table as the source of truth and keep sensor names separate from channel ids.
- Resample all runs to a common sample rate of 400 Hz.
- Window each run using 5-second windows with 1-second stride.
- Keep metadata for dataset, subject, run, and window start time.
- Build one shared artifact for both tasks.
- Normalize using train-split statistics only.
- Split 80/20 into train and held-out test by whole run, keyed on
  `run_uid = "{source}:{subject:03d}:{run:03d}"`. There is no validation split: adjacent
  windows overlap by 4 s, so a run must never be divided across splits.

## Phase 3: Baseline Modeling

- Start with a small 1D CNN on raw windows.
- Use the same backbone for both tasks.
- Use MSE + RMSE reporting for regression.
- Use cross-entropy + accuracy reporting for classification.

## Phase 4: Evaluation

- Split by whole run, never by window.
- Report train metrics each epoch. There are no validation metrics to report.
- The final-epoch model is the reported model; there is no best-checkpoint selection,
  because there is no validation set to select against.
- Evaluate the held-out test split exactly once, after the final epoch.

## Phase 5: Next Steps

- Compare raw-time baseline against deeper models later.
- Add better plots and error analysis.
- Only after the raw baseline is stable, revisit CWT.
