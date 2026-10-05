# Quarantined legacy pipeline

**Do not run these files, do not import them, and do not use them as reference
implementations.** Every module and script here raises `ImportError` at import time, and
`legacy/`, `legacy/scripts/` and `legacy/tests/` raise on package import as well.

They are kept rather than deleted because they are the provenance of the earlier results
and because knowing *why* an approach was rejected stops it being re-derived. They are
not kept because any part of them is reusable.

## Why

Every number in the paper comes from the `combined_*` stack in `src/redo_by_sara/`, which
is verified compliant with `CLAUDE.md`. The files here contradict it in six ways that
would produce wrong numbers **silently**, which is what makes them dangerous rather than
merely obsolete. Full evidence is in `docs/session0_findings.md`; the summary:

| | Defect | Where |
|---|---|---|
| **A1** | No Test_2 sampling-rate override — reads 1652 Hz from HDF5 metadata instead of the true 1706.667 Hz, time-warping every Test_2 window by **+3.3%**. Walking speed is a time-derived quantity, so every label-signal relation would be wrong. | `preprocessing.py:269`, `:126-130` |
| **A2** | No subject-006 exclusion. Five configs assign 006 to a client and omit 001 and 002 entirely. The module contains zero exclusion logic. | `preprocessing.py`; `configs/fl_classification.yaml:55` and four others |
| **A3** | `run_uid` keyed without `source_id`, so subject 003's runs **collide across sources**. 003 is recorded in both sources; that overlap is a scientific asset, and this silently merges it. | `preprocessing.py:165-237`, `iid_partitioning.py:39-41`, `federated.py:115` |
| **A4** | Train/test split **inside a run**, with a purge gap defaulting to zero. Train and test windows come from the same walking pass with the same speed label — the exact leakage `CLAUDE.md` invariant 1 exists to prevent. | `sensor_non_iid.py:162-194`, `:266` |
| **A5** | The same window is owned by **two clients** — every index of a shared subject is replicated into each owning client. Violates invariant 3. | `federated.py:136`, `:157-158` |
| **A6** | Sensor non-IID uses out-of-hallway channels 16-19, **slices** the input rather than masking it, and assumes K=4. Slicing gives clients different first-layer shapes, which FedAvg cannot average. Violates invariant 5 and the a-priori channel selection. | `sensor_non_iid.py:53-128`; the four `fl_sensor_non_iid_*` configs |

Two of these are **materialized on disk**, not hypothetical: the `artifacts/raw_windows*`
files recorded subject 006, a `[20, 2000]` sample shape and a 451/149/149 train/val/test
split. They were deleted in the same session that created this directory.

## What is here

| Path | Contents |
|---|---|
| `legacy/*.py` | 6 library modules — `preprocessing`, `sensor_non_iid`, `iid_partitioning`, `training`, `config`, `federated` — plus `mask_aware_model.py` |
| `legacy/configs/` | 17 YAML configs |
| `legacy/scripts/` | 14 scripts |
| `legacy/tests/` | 2 test files, moved so their dead imports cannot break collection for the live suite |

Their relative imports still resolve *structurally* within this package, and their
`from redo_by_sara.<legacy module>` imports no longer resolve at all. Neither matters:
the guard raises first. They are frozen, not maintained.

## What was carried forward instead of quarantined

Three things left this directory rather than entering it:

- **`SimpleCNN1D`** stays in `src/redo_by_sara/models.py`. It is the paper's architecture
  for both tasks and four `combined_*` modules import it.
- **`get_parameters` / `set_parameters`** moved to `src/redo_by_sara/parameters.py`. They
  were the only symbols the live stack needed from `federated.py`, and extracting them
  first is what made this quarantine possible without taking down the pipeline —
  `federated.py:15` imports `training` at module level, so guarding `training.py` first
  would have propagated an exception straight into `combined_iid_flower.py`.
- **`MaskAwareSimpleCNN1D`** was moved *here*, to `mask_aware_model.py`, out of
  `models.py`. The class, not the file. See that module's docstring for why the 18-channel
  mask-indicator approach is wrong for this project: masking belongs in the dataset's
  `__getitem__` after normalization, and a 9-channel input on every client is the only
  thing keeping parameter shapes identical across clients.

## If you are looking for something that used to be here

Use the `combined_*` stack:

| Task | Entry point |
|---|---|
| Centralized classification | `scripts/run_combined_centralized_classification.py` |
| Centralized absolute-speed regression | `scripts/run_combined_centralized_regression.py` |
| Centralized residual run-balanced regression | `scripts/run_combined_residual_run_regression.py` |
| Federated (Flower) | `scripts/run_combined_iid_flower.py` |
