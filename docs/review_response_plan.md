# Claude Code task brief — fix reviewer-blocking defects and build the paper figure suite

## 0. Read this before touching anything

This repository supports a submission to an IEEE conference paper on federated learning for gait analysis from structural floor vibration. An external review of the codebase and its artifacts identified defects that would get the paper rejected. Your job is to fix those defects and to build the plotting layer for the results section. **Do not rewrite the preprocessing pipeline, the artifact schema, or the run-splitting logic** — they are correct and the numbers they produce are the ones in the draft. Work surgically.

Before you change code, do this:

1. Read the existing modules and the committed artifacts. Build your own picture.
2. **Independently verify the central claim below** (§2, R1) against the artifacts on disk. Do not take my word for it. Report what you find before you start fixing.
3. Produce a short plan with file-by-file diffs you intend to make, and wait for my approval before executing anything larger than a single function.

**The authoritative work sequence is `docs/claude_code_sessions.md`.** Follow the session order there. Only two work packages are named in this document — the guardrails in §3 and the figure suite in §4 — and everything else is referenced by its R-number in §2 or by section (notably §1.3, the channel-availability non-IID scheme, which has no R-number but is a required deliverable). There is no WP1–WP6; do not reconstruct one.

**§2 R1 must be fixed before any new federated experiment is run**, or every new run will reproduce the same collapse.

After each work package: run the test suite, write a summary of what changed and what the new numbers are, and stop for review. Append each summary to `CHANGES.md` at the repo root.

---

## 1. Ground truth about the data — treat this as authoritative

Any code, comment, config, or docstring that contradicts this is wrong and should be corrected.

**Sources (two collection campaigns in the SAME instrumented corridor, 19.6 months apart — not two buildings):**

| Source | Subjects | Sampling rate | Notes |
|---|---|---|---|
| `TestData/Test_2` | 001, 002, 003 | **True fs = 1706.667 Hz** | The 1652 Hz in the file metadata was **recorded incorrectly**. The 1706.667 override is deliberate and correct. Resample to 400 Hz with `resample_poly(15, 64)`. |
| `TestData/20251124_Testing` | 003, 004, 005, 006, 007, 008 | fs ≈ 1651.61 Hz | Resample to 400 Hz with `resample_poly(31, 128)`. |

**Critical facts:**

- **Subject 006 is excluded from every experiment** due to data collection problems. This is a deliberate exclusion, not a bug. Keep it, and make the exclusion reason a named constant with a comment rather than a magic filter.
- **Subject 003 appears in BOTH sources, 19.6 months apart.** Not duplication — a genuine cross-session recording of the same person, and scientifically valuable (see §2, R4). Any code or comment implying subject 003 should be deduplicated is wrong. **Identity is confirmed.** The subject-003 demographics in the `Test_2` HDF5 are erroneous; `testNotes.xlsx` is authoritative for this subject.
- Effective subject set: **001, 002, 003, 004, 005, 007, 008 → 7 subjects.** Subject IDs are not contiguous; never assume `range(1, 8)`.
- Subject identity and source are **nearly, but not fully, collinear** (003 is the only overlap). This is a confound the paper must address, and subject 003 is the instrument for addressing it.
- `run_uid = "{source}:{subject:03d}:{run:03d}"` is the canonical atom. **A run is never split** across train/test or across federated clients — windows overlap 80%, so splitting a run leaks. This invariant is already enforced; do not weaken it.
- **Windowing is fixed: 5 s windows, 1 s stride, 2000 samples at 400 Hz.** This is a settled design decision — do not ablate it or treat it as a tunable.
- **Channels: `[1, 2, 3, 4, 5, 6, 7, 8, 10]` → 9 channels.** The other 11 of the 20 sensor channels were **not located in the hallway where the walking tests were conducted**, so they carry no gait signal. This selection was made *a priori from physical sensor placement, not from model performance* — record that justification in the config, in the run summary, and in any docstring that touches channel selection, because "why 9 of 20 channels?" is a question a reviewer will ask and the a priori answer is the strong one. Assert the indexing convention explicitly (config is 1-indexed → numpy indices `[0..7, 9]`); a silent off-by-one would shift every channel and is undetectable downstream.
- Normalization statistics come from **train windows only**, applied lazily in `__getitem__`.
- Current split: seed 4601, 80/20 by whole run → 112 train / 28 test runs from 140 usable.

### 1.1 Split integrity — the invariant that must never break

The 80/20 split is **by whole run, never by window**. With 5 s windows at 1 s stride, adjacent windows share 4 s of signal, so a single run's windows landing on both sides of the split is direct leakage that would invalidate every number in the paper. This is already enforced via `run_uid`; your job is to make it *provable*:

- Assert `set(train_run_uids) ∩ set(test_run_uids) == ∅`, and that every window maps to exactly one split — at artifact build time and again at load time.
- Assert the same for client partitions: no `run_uid` appears in two clients, and both tasks assign a given run to the same client.
- Emit into the run summary the run-level split (112/28) **and** the resulting window-level split, which is *not* exactly 80/20 (1242/307 classification, 966/238 regression). Report both in the paper.
- Add a test that constructs a deliberately leaky split — two windows of one run on opposite sides — and confirms the assertion fires.

### 1.2 Number of clients — decided, use these

Client count is not a free parameter to guess at. The two experiment families answer different questions and take different K:

| Family | K | Client = | Purpose |
|---|---|---|---|
| **Natural / cross-silo (headline)** | **2** | one collection campaign | `Test_2` (2024-04-06) = client 0, `20251124_Testing` (2025-11-24) = client 1. Same corridor, 19.6 months apart. Genuine feature skew (different true fs, different DAQ trigger config, 19 months of environmental and structural drift), label skew (cohorts overlap only at 003), protocol skew (unidirectional vs bidirectional), quantity skew. |
| **Channel-availability non-IID (§1.3)** | **3** | a site with a partial sensor deployment | Each client holds an overlapping subset of the 9 channels. Overlap ratio is swept to give a controlled severity curve. |
| **IID control** | **2 and 3** | simulated site | Must be run at **both** K values, to match the two above. |
| **Dirichlet label skew** *(optional, cut first)* | 3 | simulated site | α ∈ {0.1, 0.5} over subjects at run level. Cheap to add (partitioner only, no architecture change) and it is the controlled non-IID baseline reviewers expect. Add only if the schedule holds. |

**IID partition requirement (assert as a property):** every client holds ≥1 run from every subject, **and no client holds all runs of any subject**. This is currently true by accident of the blocking scheme; after the round-robin rewrite it must be checked explicitly.

**The rule that matters: IID and non-IID must be compared at matched K.** Comparing an IID K=3 run against a natural K=2 run confounds partition heterogeneity with client count, and a reviewer will say so. Every non-IID number needs an IID number *and* a local-only number at the same K.

**Do not exceed K=3.** With 112 training runs, K=6 leaves ~18 runs per client and roughly 4 optimizer steps per local epoch at batch size 4 — that structurally reproduces the R1 collapse. It also weakens the claim, since "one client = one campaign" gets less credible as K exceeds the number of campaigns actually collected.

**K=2 needs no apology.** Few clients holding substantial data each is the standard *cross-silo* federated setting, as distinct from cross-device FL with thousands of participants. Use the term "cross-silo" explicitly.

Note that the current partitioner is structurally K=3 — it blocks runs in threes within each `(source, subject, direction)` stratum, one per client. The round-robin rewrite in WP0-3 is therefore a **prerequisite** for all of the above, not optional cleanup.

### 1.3 Channel-availability non-IID — build it by masking, not slicing

This is the paper's controlled non-IID axis. It is physically motivated: different sites instrument different corridor segments with different numbers of sensors, so partial and overlapping sensor coverage is a real deployment constraint for structural vibration sensing. **This is now the only axis in the paper standing in for spatial heterogeneity across sites**, since the natural split turned out to be cross-session rather than cross-building — so it carries more weight than originally planned.

**Implement by masking. Do not slice the input tensor.** `SimpleCNN1D`'s first conv weight has shape `(out_ch, in_ch=9, kernel)`. If clients hold different numbers of channels, their first-layer tensors have different shapes and **FedAvg cannot average them**. Instead:

- Keep the input tensor **9 channels wide on every client**, always.
- Pass a per-client boolean channel mask into the dataset; zero the unavailable channels in `__getitem__` **after normalization**, so a masked channel equals the training mean ("no information") rather than an extreme value.
- Architecture is then identical across clients and FedAvg works mechanically.
- Assert at client construction that every client's model has identical parameter shapes.

**Framing — this matters for acceptance.** Splitting *features* across clients is vertical FL, and FedAvg is a horizontal algorithm; describing raw channel-splitting as "non-IID FL" will be flagged. With masking, every client shares the same feature space and some clients simply observe zeros in part of it. That is **covariate shift / sensor-availability heterogeneity**, which is horizontal and legitimately non-IID. State explicitly in the paper that this is not vertical federated learning, and why.

**Parameterize the overlap over POSITIONS, not channels.** The 9 channels cover **8 distinct hallway positions**: Sensors 1–7 are one unit each, and channels 8 (x) and 10 (z) are two separate uniaxial units **co-located at an eighth point** (see `docs/sensor_layout.md`). Partition the 8 positions, then expand to channel indices; channels 8 and 10 always travel together. Splitting them is possible in hardware terms but gains a client no independent spatial coverage and breaks the "each site instruments one stretch of hallway" argument.

Define a shared core of positions held by all clients plus per-client private positions, controlled by an overlap ratio ρ. Sweep **ρ ∈ {1.0, 0.67, 0.33, 0.0}**, where ρ=1.0 recovers IID and ρ=0.0 is fully disjoint (positions {1,2,3}/{4,5,6}/{7,8} at K=3, giving 3 channels each). Prior experiments showed the disjoint case fails to converge — **report that, do not hide it.** "Performance degrades smoothly with sensor overlap and collapses at zero overlap" is a finding, and it is the most interesting point on the curve.

Log **positions per client and channels per client separately** — the client holding position 8 gets 3 channels from only 2 positions.

**Two cautions:**

- Positions are **not exchangeable**, but they are spatially ordered: positions 1 → 8 run down the corridor (one coordinate, sensor 3, is unresolved between two records — see `docs/sensor_layout.md` §3; the contiguous blocks hold either way), so contiguous position blocks are contiguous corridor segments. Use **contiguous spatial blocks** as the primary scheme. Orientation also varies — channel 8 is the only horizontal-axis channel, so whichever client holds position 8 sees an extra, qualitatively different view of the same footstep. Report over **≥3 random position assignments** so no single layout drives the number, and log which client holds position 8.
- Decide and report the **evaluation protocol** explicitly, both variants: (a) global test set with all 9 channels — does federation recover full-sensor performance? (b) client-local test with the client's own mask — deployment reality. A reviewer will ask which number they are reading.

Normalization statistics stay global (train windows, all channels) for comparability across ρ; note this choice in the paper.

---

## 2. The defects to fix, in priority order

### R1 (blocking) — The federated regression model does not train, and the summary reports it as a near-match to centralized

Evidence to verify first:

- `regression/test_window_predictions.csv` contains **one distinct** `predicted_residual_mps` across all 238 test rows (≈ `-0.0002014615893131122`). The centralized run has 238 distinct values.
- `global_train_standardized_mse` is flat across all 60 rounds (1.000000 → 0.999902 → 1.000041). Standardized MSE of exactly 1.0 means "predict zero residual", i.e. the trivial baseline. The centralized run drives the same quantity 1.0000 → 0.6931.
- Reported test R² 0.9244 vs the source/subject-mean baseline R² 0.9246 — the federated model is fractionally **worse than doing nothing**.
- `comparison_vs_centralized_residual_cnn` reports `rmse_difference_mps: 0.00243`, which reads as success. It is a collapse.
- Client fit history shows clients **do not learn locally**: per-client `local_online_standardized_mse` is flat across all 60 rounds (client_0 0.7383→0.7383, client_1 0.8365→0.8349, client_2 1.4350→1.4314). The failure is therefore **upstream of aggregation** — FedAvg is faithfully averaging three near-zero updates.

*Corrected diagnosis (verified against artifacts, supersedes an earlier claim that the failure was at aggregation):* each client constructs a **fresh Adam optimizer every round** and takes only ~10 steps with it (37 runs at batch 4, 1 local epoch). Adam's second-moment estimate never becomes useful, so the client replays Adam's warmup 60 times and never leaves it. Centralized training used a **single persistent Adam over ~1680 steps** and needed ~560 of them before the loss moved at all — with a zero-init head the early gradients are tiny, and 10 steps from a cold optimizer cannot move them. Cosine LR decay to 1e-5 compounds this.

**What to build:**

- Parameterize local epochs, LR schedule (including a per-round restart option and a constant option), batch size, and server-side optimizer. Remove the config validator's hard rejection of anything but 3 clients / 60 rounds / 1 local epoch (see WP0-3).
- **Add client optimizer state persistence across rounds** as a first-class option. Carrying Adam moments between rounds is legitimate and standard in cross-silo FL, where clients are persistent institutions rather than transient devices. Given the corrected diagnosis this may be the highest-value single knob.
- **Instrument local training directly:** log each client's loss at the *start* and *end* of its local training each round, not only the online average. This measures whether local training does anything, rather than inferring it.
- Add **FedAdam** and **FedYogi** server strategies alongside FedAvg, selectable from config. Note that server-side adaptivity cannot rescue clients that produce no update — expect these to matter only once the local axis is fixed.
- Sweep in two stages: first `local_epochs ∈ {1, 5, 10} × client_optimizer_state ∈ {fresh, persistent}` under FedAvg (6 configs, isolating the local axis), then FedAdam at the best local setting. Report the table with `local_epochs` understood as the causal variable.
- Add a **matched-budget** mode: total gradient steps for FL (clients × local_epochs × steps_per_epoch × rounds) reported alongside the centralized step count, so the comparison is defensible either way. Log both numbers into the run summary.

**Acceptance criteria (assert these in code, and fail the run loudly if violated):**

- Final `global_train_standardized_mse < 0.95`.
- Number of distinct predicted residuals on the test set `> 10`.
- Federated test skill score vs the subject-mean baseline `> 0` (see R2 for the definition).
- A regression test that loads a known-degenerate prediction file and confirms the detector fires.

### R2 (blocking) — Regression is reported against the wrong reference, hiding that skill over the trivial baseline is small

The residual formulation hands the model the subject identity for free, so R² against total variance is dominated by "who is walking". Recomputed from the existing artifacts:

| Condition | R² (total) | RMSE (m/s) | Skill vs subject-mean |
|---|---|---|---|
| Source/subject mean lookup | 0.9246 | ~0.0405 | — (reference) |
| A: absolute regression | 0.812 | 0.0639 | **−1.49** (far worse than the lookup) |
| B: residual + run-balanced (centralized) | 0.933 | 0.0382 | +0.11 |
| C: residual, federated | 0.9244 | ~0.0407 | −0.003 |

Skill = 1 − MSE_model / MSE_baseline. Condition A verified two ways: 1 − (0.0639/0.0405)² = −1.49, and 1 − (1−0.812)/(1−0.9246) = −1.494. An earlier draft of this table gave −0.75 for condition A; that was an arithmetic error, and the corrected figure strengthens the point rather than weakening it. Note also that **R² of 0.812 is exactly skill against the overall train mean** — the clearest statement of why R² against total variance is the misleading framing here, since it credits the model for variance the subject prior already explains.

**What to build:**

- A single metrics function that returns, for every regression evaluation: baseline RMSE, model RMSE, ΔRMSE, R² (total), **R² (residual / within-subject)**, and **skill score = 1 − MSE_model / MSE_baseline**.
- Make **skill score the primary reported metric**. Every regression table and every regression figure must carry the baseline as an explicit row or reference line.
- Consolidate `regression_metrics` / `_extended_metrics` / `_extended_regression_metrics` (3 names, 2 implementations) into this one function. This is the one deduplication I want done now, because the paper's numbers depend on it being singular.

### R3 (major) — The residual target assumes an oracle subject identity at inference

`speed − source/subject train mean` requires knowing who is walking. In deployment that comes from the classifier.

**What to build:** an end-to-end evaluation path — classifier predicts subject → look up that subject's train-set prior → add the CNN's predicted residual → compare against true speed. Report it beside the oracle-identity number in the same table. Run it for centralized and federated, IID and non-IID.

### R4 (major) — Classification is saturated and confounded with recording site

FL and centralized agree on all 307 test windows including the same single error (index 1255, subject_002_walking → subject_003_walking), and the two confusion CSVs share an md5. The experiment cannot discriminate FL from centralized at this operating point. Separately, subject and source are nearly collinear, so the classifier may be reading **session** signature rather than gait. This is worse than a building confound would have been: the corridor is identical, so whatever separates the sources is pure nuisance — DAQ trigger configuration, true sampling rate, and 19 months of drift.

**What to build:**

1. **Run-level metrics** as the primary classification result: aggregate window predictions per run by majority vote and by mean probability, report run-level accuracy and macro-F1 with a **Wilson 95% interval**. The effective test set is 28 runs, not 307 overlapping windows — say so.
2. A **source-classification probe**: same architecture, label = source instead of subject. Quantifies how separable the two campaigns are on their own.
3. A **subject-003 cross-session control** — required, not optional. Two stages:
   - *Free first look:* subject 003 already has test runs from both sources (4 from Test_2, 3 from 20251124). Break down the existing model's predictions on 003 by source. Informative, but not conclusive, since the model saw 003 from both campaigns in training and may have learned two representations of one person.
   - *The actual control:* leave-one-source-out. Train on `Test_2` only (subjects 001, 002, 003), test on subject 003's runs from `20251124_Testing`. If identity transfers across a 19-month gap and a DAQ configuration change, the model is reading gait; if not, it is reading session. Small run — three classes, ~56 training runs.

   This is the single cleanest defense against the confound, and it matters more now that the two sources share a corridor: whatever separates them is pure nuisance rather than anything meaningful.
4. **Do not ablate window length.** 5 s / 1 s stride is fixed (§1). If classification stays saturated, the honest response is to report the run-level Wilson interval and state plainly that the task is at ceiling and cannot discriminate FL from centralized — not to manufacture headroom under deadline.

### R5 (major) — No local-only baseline, so the paper's central claim is untested

The claim is that FL helps when data comes from different silos. The comparison that establishes this is **each client training alone on its own shard**, not FL vs centralized. Centralized is the upper bound, local-only is the lower bound; FL must sit meaningfully above local-only.

**What to build:** a `local_only` run mode — for each client, train on that client's partition only, evaluate on the shared global test set. Report per-client and mean ± std. Both tasks. Every partition scheme. This must appear in every results table alongside FL and centralized.

### R6 (major) — Single seed, single split

**What to build (deadline-trimmed):** a multi-seed driver running **3 initialization seeds on the fixed split (seed 4601)** for every reported configuration. All tables report mean ± std. Add bootstrap 95% CIs over **runs** (not windows). State the effective sample size given 80% window overlap once, in the run summary.

Full re-splitting (3 different split seeds) is more convincing but costs a full pipeline rebuild per seed. Run it for the **single headline configuration only**, and only if the schedule in §5 is holding.

### R7 (moderate) — Privacy claim is unsupported by FedAvg alone

No code change required unless we add DP. **Add a `PRIVACY_NOTES.md`** stating the threat model precisely: raw vibration data never leaves its silo; model updates are shared; FedAvg provides data minimization, not a formal guarantee; gradient inversion is out of scope. If time permits later, leave a clean hook for DP-SGD noise injection on client updates, but do not implement it now.

### R8 (moderate) — Preprocessing facts that need to be visible

- Emit the Test_2 sampling-rate override (1652 metadata → 1706.667 true, +3.3%) into the run summary and the setup audit as an explicit, annotated field, with the justification in the comment. It time-warps every Test_2 sample and speed is a time-derived quantity, so a reviewer will ask.
- Add a field for the **APDM ground-truth precision**, and compute what you can from the data: for runs with repeated or overlapping APDM estimates, report the spread. If the reference system's own error is ~0.03–0.05 m/s, the best RMSE of 0.0382 sits at or below it, and that must be discussed. Surface the number; do not guess it.

### R9 (integrity) — Fabricated audit value

`combined_iid_flower.py:783 / 1000 / 1020` — `test_evaluation_counter` is never incremented, so the guard cannot fail and `"test_evaluations_during_training": 0` is asserted rather than measured. **Make the counter actually increment on every touch of the test set**, keep the guard, and re-run the audit. If the paper claims the test set was untouched during training, this must be a measurement.

### R11 — What does the `direction` field actually encode? **[RESOLVED in S2; the direction-keyed-baseline follow-up is CUT]**

The collection protocol had all subjects walking in a **single direction**, so `direction` should be constant. It is not: `(source, subject, direction)` gives 13 strata against 8 for `(source, subject)`, so the field varies for five of the eight. The arithmetic points at Test_2 being single-valued and all five 20251124 strata being two-valued (3 + 5×2 = 13) — meaning either the second campaign genuinely collected both directions, or the field is derived differently between sources.

**This is load-bearing.** Runs are speed-ordered and blocked within `(source, subject, direction)` strata, so the partitioner — and therefore every federated result — depends on the field meaning what its name says.

**What to check, read-only, before the partitioner rewrite:**

- Distinct `direction` values in `run_manifest.csv`, cross-tabbed by source and subject with run counts.
- Where in the discovery code the field is derived, and from what.
- Whether mean walking speed differs between direction values within a subject.

**Then, depending on the outcome:**

- *Labeling artifact* → stratify on `(source, subject)` only, and this item closes.
- *Real directional variation in 20251124* → a documented protocol difference between the two campaigns, which strengthens the natural non-IID story. It also means direction sits in the residual for half the data, and the CNN can identify direction trivially from the order the hallway sensors activate — so recompute skill against a `(source, subject, direction)`-keyed baseline to check how much of the +0.11 is direction detection rather than gait-speed estimation. Report both baselines in every regression table, plus mean speed by direction per subject.

### R10 (optional, last, cuttable) — Is the backbone too weak?

A reviewer may ask whether `SimpleCNN1D` limits the results. Answer it cheaply and do **not** swap the paper's architecture:

- Classification is already saturated (one error on 307 windows); extra capacity cannot help and re-architecting invalidates every existing number.
- Regression skill is ~11% over baseline, but best RMSE (0.0382 m/s) is at or below plausible APDM reference precision — the limit may be label noise, not capacity (see R8).
- More parameters averaged over ~9 optimizer steps per local epoch is exactly the regime that produced the R1 collapse.
- A small model strengthens the communication-cost / feasibility argument.

**What to build, only if days 18–20 are free:** a **centralized-only** comparison of `SimpleCNN1D` against a small 1D-ResNet (≈4 residual blocks, width 32–64 — not a ported ImageNet ResNet), both tasks, one table (T6), no federated runs. If the ResNet does not meaningfully win, that is a positive result justifying the design choice. If it does win on regression, report it as future work rather than rebuilding the paper.

---

## 3. WP0 — Guardrails to do first

1. **Seeding parity.** `_set_random_seeds` exists in 4 copies; the FL copy silently drops the cudnn determinism block, so FL and centralized are seeded differently. Collapse to one implementation in a shared module, import it everywhere, and assert at run start that determinism flags are set identically across run types.
2. **`--verify-only` must be read-only.** It currently writes `setup_audit.json`, `partitions/`, and a PNG. A verification that mutates outputs is not a verification. Route all writes through a single guarded writer that is a no-op in verify mode.
3. **Replace outcome assertions with property assertions.** The nine `EXPECTED_*` tables at `combined_iid_fl_partitioning.py:31-64`, the `expected` dict with `num_examples: 1204` inside `build_regression_artifact`, `if total_runs != 112`, `expected_support = [69, 32, ...]`, and the config validator that rejects anything but 3 clients / 60 rounds / 1 local epoch all pin *outcomes*. They must become *properties*, because the R1 fix and the non-IID work require these parameters to vary:
   - partitions are disjoint and covering; no run split across clients; `train ∩ test = ∅`
   - per-client run counts within ±1 of balanced (for IID only)
   - both tasks assign the same run to the same client
   - every client holds every subject and both directions (IID only — explicitly *not* required for non-IID)
   - Move the exact numbers (140 / 112 / 28 / 1549 / 1242 / 307 / 1204 / 966 / 238 / (37,38,37) / (412,418,412)) into the **test files as independent literals**, not imported from the module. Accidental drift still fails loudly; a deliberate change becomes a one-line test edit.
   - While you are here: `build_combined_iid_partitions` currently spends ~200 lines (`_remainder_assignments` brute-forcing `itertools.product`, `_solve_subject_blocks` rank-minimizing DP, `_assign_profiles`) reproducing (37, 38, 37). Replace with a round-robin over speed-ordered blocks, which produces a valid balanced partition directly and generalizes to K clients. Verify the properties hold; the exact triple no longer needs to be reproduced.
4. **Degenerate-model detector**, run automatically at the end of every regression run: distinct-prediction count, prediction std, final standardized MSE, skill score. Fail the run if it looks like a constant predictor. This is the check that would have caught R1.
5. **`conftest.py` and `pyproject.toml`** at the repo root so `pytest` works from root; add `pytest` to `requirements.txt` and the conda environment file. Mark the 9 tests that require gitignored `.pt` files or the 7.6 GB `TestData/` tree with `@pytest.mark.requires_data` so the fast subset runs anywhere.

---

## 4. WP7 — Paper figure and table suite

Build a single module (e.g. `paper_figures.py`) that produces every figure and table in the results section from saved artifacts, **without re-running any training**. Preserve the existing good pattern: every plotting function accepts either a path or an in-memory object.

**Global style requirements:**

- IEEE two-column: single-column width **3.5 in**, double-column width **7.16 in**. Provide both variants where relevant.
- Base font 8 pt, tick labels 7 pt, matching IEEE body text.
- Save **vector PDF and 300+ dpi PNG** of every figure, same basename, into `figures/`.
- **No titles inside the figure** — captions live in LaTeX. Axis labels and legends only.
- Colorblind-safe palette; never encode meaning by color alone (use marker/linestyle too).
- Deterministic filenames and deterministic ordering, so a re-run produces byte-comparable output.
- One `make_all_figures(results_dir, out_dir)` entry point plus individually callable functions.

**Figures to implement:**

1. **`plot_dataset_overview`** — speed distribution per subject (violin or strip), annotated with within-subject std, split by source. Shows subject 003 twice, once per source. This figure carries the "92% of variance is subject identity" point.
2. **`plot_system_diagram_data`** — export the counts needed for the system/pipeline figure (runs, windows, splits per source and subject) as a table; the diagram itself will be drawn in a vector editor.
3. **`plot_partitions`** — runs per client × subject heatmap or stacked bar, one panel per partition scheme (IID, natural cross-session, channel ρ). This is the figure that makes the non-IID contribution legible.
4. **`plot_convergence`** — metric vs communication round, with **horizontal reference lines for the centralized result and the trivial baseline**, and a band over seeds. Both tasks, IID and non-IID overlaid. *This single figure would have made R1 obvious at a glance; treat it as the highest-value plot in the module.*
5. **`plot_run_level_scatter`** — predicted vs actual walking speed at run level, identity line, colored/marked by subject, side-by-side panels for centralized / FL / local-only. Annotate RMSE and skill score in-panel.
6. **`plot_residual_diagnostics`** — Bland–Altman (mean vs difference) against APDM ground truth, plus residual-vs-speed to expose regression-to-the-mean.
7. **`plot_skill_comparison`** — grouped bar of skill score across {local-only, FedAvg, FedAdam/FedProx, centralized} × {IID, non-IID variants}, with error bars over seeds and a zero line marking the trivial baseline.
8. **`plot_confusion`** — run-level normalized confusion matrix. Only render for the hardest / non-saturated settings; do not produce four near-identical saturated matrices.
9. **`plot_communication_cost`** — parameters, MB per round, and rounds-to-reach-95%-of-centralized, as a small two-panel figure. Supports the "feasible" half of the claim.
10. **`plot_heterogeneity`** — performance vs channel overlap ratio ρ (1.0 → 0.0), with the ρ=0 collapse plotted rather than omitted, a band over the ≥3 random channel assignments, and horizontal reference lines for centralized and local-only. The natural cross-session split appears as a marked point, not a curve. Both evaluation protocols on the same axes (different linestyles).

**Tables to emit** (as LaTeX `booktabs` source *and* CSV, into `tables/`):

- **T1** Dataset: sources, subjects, runs (161 → 140 usable, with the subject-006 exclusion stated), windows per task, split counts, speed mean/std/range **and within-subject std**.
- **T2** Centralized reference, both tasks, with baseline row and skill score.
- **T3** FL under IID at **K=2 and K=3**: {centralized, FedAvg, FedAdam, local-only} × both tasks.
- **T4** FL under non-IID: {natural cross-session (K=2), channel-availability at ρ ∈ {0.67, 0.33, 0.0} (K=3)} × {local-only, FedAvg, robust variant, centralized}. Every row pairs with a matched-K IID row from T3. Channel rows report both evaluation protocols (global 9-channel test and client-masked test).
- **T6** *(optional, cut first)* Centralized architecture comparison: SimpleCNN1D vs small 1D-ResNet, both tasks, centralized only. See §2 R10.
- **T5** Communication and compute cost.
- All tables: mean ± std over seeds, and the run-level metric as primary.

Consolidate the existing duplicated plot code while you do this — 3 copies of the actual-vs-predicted scatter (~80 lines each), 5 copies of the confusion-matrix renderer, 2 identical prediction-row builders, 3 copies of `_resolve_path`, 3 of `_write_rows` (with two different argument orders), 3 of `_as_index_list`. One implementation each, in the new module or a shared utils module.

---

## 5. Schedule and scope cuts — deadline is under three weeks

Indicative pacing. If a stage slips, cut from the bottom of the list, never from the guardrails (§3) or the R1 fix.

| Days | Session (`docs/claude_code_sessions.md`) | Work |
|---|---|---|
| 0 | S0 | Orientation and independent verification of R1. No edits. |
| 1–3 | S1, S2 | Guardrails (§3 items 1, 2, 4, 5 + R9), then the partitioner rewrite (§3 item 3). Nothing else runs until this lands. |
| 4–6 | S3 | §2 R1: fix the regression collapse. Sweep local epochs × server optimizer. Acceptance criteria must pass. |
| 7–8 | S4, S5 | §2 R2 metrics re-cut (skill score) and R5 local-only baseline; then §1.3 position-masking and the ρ sweep. |
| 9–13 | S6 | Experiment matrix: K=2 {IID, natural, local-only} and K=3 {IID, channel ρ=0.67/0.33/0.0, local-only}, both tasks, both evaluation protocols, 3 init seeds, ≥3 position assignments. |
| 14–15 | S7 | §2 R3 end-to-end path; R4 run-level metrics + source probe + subject-003 control. |
| 16–17 | S8 | §4 figures and tables. Writing runs in parallel from day 9 using placeholder numbers. |
| 18–20 | S9 | Buffer, then R10 architecture comparison only if genuinely free. |

**Cut under deadline pressure, in this order:** R10 architecture comparison → Dirichlet label skew → client-masked evaluation protocol (keep the global one) → communication-cost figure (Fig 9) → third channel assignment → full split-seed replication → the FedYogi arm of the R1 sweep. **Never cut:** the local-only baseline, the matched-K pairing, the ρ=0.0 collapse point, or the degenerate-model detector.

## 5b. Out of scope for now — do not do these

Already cut by the deadline: window-length ablation (§2 R4-4), leave-one-subject-out, DP-SGD implementation, speed-band skew unless the schedule holds.

Real debt, invisible to a reviewer, not worth the risk before submission:

- **`_run_flower_core` (340 lines) — one exception, gated.** Four sessions edit this function, so a single contained refactor is cheaper and safer than four stacked edits. Do the task-adapter refactor (`build_model` / `build_loader` / `evaluate` / `history_row`, collapsing the seven task branch points) **at the start of Session 3, as a standalone step**, then re-run one existing config and require **bitwise-identical outputs** before making any behavioral change. If outputs differ, revert and proceed without the refactor. Do not refactor after the R1 changes land.
- Merging the two Flower client classes into a base class.
- Cleaning the 13 unread YAML keys, the unread `config["experiment"]` block, or the `experiment` parameter naming collision — **except** to fix the three-way duplication of the sampling rates and channel/subject lists, which must have exactly one source of truth given §1 above.
- The double-execution inefficiencies (`--task both` running the audit twice, `_group_run_records` running six times, O(rounds²) history rewrites). Fix only if they cost more than a few minutes per run.
- Repo hygiene (the 3.3 MB of staged PNGs, `full_run.log`, the interrupted-round-1 directory). I will handle these.

---

## 6. Definition of done

- [ ] Federated regression trains: final standardized MSE < 0.95, > 10 distinct predictions, positive skill score. Sweep table produced.
- [ ] Skill score is the primary regression metric everywhere; baseline appears in every table and figure.
- [ ] End-to-end (predicted-subject) regression path implemented and reported.
- [ ] Local-only baseline implemented and reported for both tasks and all partitions.
- [ ] Run-level classification metrics with Wilson intervals; source probe and subject-003 cross-session control implemented.
- [ ] 3 initialization seeds for every reported config; mean ± std in all tables; bootstrap CIs over runs.
- [ ] Split seed and initialization seed are **separate config keys** — a single `seed: 4601` currently drives both.
- [ ] R8 surfaced: the Test_2 sampling-rate override annotated in the run summary and setup audit; APDM reference precision computed where possible and recorded.
- [ ] R7: `PRIVACY_NOTES.md` written, stating the threat model and that FedAvg provides data minimization rather than a formal guarantee.
- [ ] Every non-IID result has a matched-K IID result and a matched-K local-only result. K ∈ {2, 3} only.
- [ ] Channel-availability partitions implemented by **masking** (9-channel input everywhere); identical parameter shapes across clients asserted; ρ sweep including ρ=0.0 reported; ≥3 channel assignments; both evaluation protocols reported.
- [ ] IID partitions assert every client holds ≥1 run per subject and no client holds all runs of any subject.
- [ ] Split-integrity assertions in place at build and load time, plus a test proving they fire on a leaky split.
- [ ] Channel selection rationale (hallway placement, a priori) recorded in config, run summary, and docstrings; 1-indexed → 0-indexed conversion asserted.
- [ ] `test_evaluation_counter` measures rather than asserts; `--verify-only` writes nothing; seeding is identical across run types.
- [ ] `EXPECTED_*` outcome pins replaced by property assertions; exact numbers live in tests as independent literals; partitioner generalizes to K clients.
- [ ] Degenerate-model detector runs automatically and has its own test.
- [ ] `pytest` runs from the repo root; fast subset passes without the data tree.
- [ ] `paper_figures.py` regenerates all 10 figures and 5 tables from artifacts with one command, in IEEE format, without retraining.
- [ ] `CHANGES.md` records what changed and every number that moved as a result.
