# CHANGES

What changed, and every number that moved as a result. Newest first.

---

## Session 5 (amendment 3) — Dirichlet classification at two more init seeds (2026-10-22)

PI request: 4 new queue jobs, group 7. The only code change is the job list in
`s5_queue.build_jobs`.
- **The jobs:** Dirichlet α = 0.1 classification at init seeds 4602 and 4603, FL first,
  then local-only.
- **Settings:** split seed 4601, `configs/s5_k3_dirichlet0.1.yaml`, FedAvg E=5
  persistent, `client_num_cpus: 3`, global 9-channel evaluation, same gates.
- **Ordering:** the jobs are appended, so jobs 1–43 keep their order, IDs and outcomes.
  A restart skips all 43 and runs only these 4.
- **Estimated time:** 2.9 h, from the measured seed-4601 Dirichlet runs (FL 42 min,
  local-only 45 min).
- **Effect on the tables:** with 3 seeds, Dirichlet classification moves into
  `derived/seed_aggregates.csv` for FL and for local-only. The seed-4601 values in the
  results entry below are unchanged.
- **Tests:** the job-matrix test literals become 47 jobs, 44 training jobs and group 7 = 4.
  New assertions pin group 7's composition and that jobs 1–43 are unchanged.
  `test_s5.py`: 35 passed.

---

## Session 5 — Results (2026-10-22)

The S5 queue finished: **43 of 43 jobs terminal, none crashed**. It ran over three
nights:

| Night | Commit | Jobs |
|---|---|---|
| 2026-10-19/20 | `c05883c` | 1–13 |
| 2026-10-20/21 | `c71e8d5` | 14–33 |
| 2026-10-21/22 | `34babba` | 34–43 |

The gates and labels in this entry are the current ones (amendments 1 and 2 below). Jobs
1–33 were re-derived under them with `--derived-only`, without retraining.

**Measured wall time:** 20.7 h of training over 40 training jobs. FL classification took
33–48 min, FL regression 25–35 min, local-only classification 45–48 min, local-only
regression 32–38 min, centralized 7–12 min, the source probe 6.8 min and the 003 control
2.4 min.

**Setting:**
- FL and local-only: FedAvg, E=5, persistent client optimizer, 60 rounds,
  `client_num_cpus: 3`.
- Split seed 4601 everywhere. Every model is evaluated on the same 28 held-out test runs,
  with all 9 channels.

**How to read the numbers:**
- Classification is run-level accuracy by majority vote, with a Wilson 95% interval.
  Mean probability gives the same counts wherever it isn't listed.
- Regression is skill = 1 − SSE_model / SSE_lookup against the source/subject train-mean
  lookup, with a run-level bootstrap 95% CI (10,000 resamples, seed 4601). The lookup
  RMSE is 0.0405 m/s in every run.
- **Labels:** `trained`, `did not train` (has output spread, fails trained) or
  `collapsed` (fails the output-spread check).
- **The trained gate:** FL and centralized models are gated on global-train standardized
  MSE < 0.95; a local-only client on its own-data standardized MSE < 0.95.

### Summary

| Condition | Classification: FL (centralized for row 1) | Classification: local-only (mean ± std over clients) | Regression: FL (centralized for row 1) | Regression: local-only (mean ± std over trained clients) |
|---|---|---|---|---|
| **Centralized** (3 init seeds) | 28/28 [0.879, 1.000] on every seed | — | **Residual:** +0.098 ± 0.021, 3 of 3 trained. Per seed: +0.121 [−0.288, +0.235], +0.082 [−0.097, +0.186], +0.091 [−0.034, +0.165]. **Absolute:** −1.739 ± 0.446. Per seed: −1.485 [−5.574, −0.848], −1.478 [−5.760, −0.772], −2.254 [−7.358, −1.382]. | — |
| **K=2 IID** (FL: 3 seeds) | 28/28 [0.879, 1.000] on every seed | 1.000 ± 0.000. Each client 28/28 [0.879, 1.000]. | +0.172 ± 0.096, 3 of 3 trained. Per seed: +0.089 [−3.123, +0.576], +0.150 [−2.074, +0.543], +0.277 [−2.307, +0.658]. | +0.010 ± 0.358, 0 of 2 clients failed |
| **K=2 natural** (FL: 3 seeds) | 28/28 [0.879, 1.000] on every seed | 0.554 ± 0.126. client_0 13/28 [0.295, 0.642]; client_1 18/28 [0.458, 0.793]. | **Per seed; 1 of 3 seeds collapsed.** 4601: −0.207 [−4.128, +0.334] trained. 4602: −0.123 [−4.367, +0.501] trained. 4603: −1.1e-6 [−3.5e-6, +4.4e-7] **collapsed**. | −0.726 ± 0.869, 0 of 2 clients failed |
| **K=3 ρ = 1.0** (= IID) | 28/28 [0.879, 1.000] | 1.000 ± 0.000. Each client 28/28. | +0.101 [−2.263, +0.480], trained | −0.125 ± 0.343, 0 of 3 clients failed |
| **K=3 ρ = 0.375** | 28/28 [0.879, 1.000] | 0.976 ± 0.041. 28/28, 28/28, 26/28 [0.774, 0.980]. | +0.046 [−0.025, +0.077], **did not train** | −0.329 ± 0.389, 0 of 3 clients failed |
| **K=3 ρ = 0** | 28/28 [0.879, 1.000] | 0.976 ± 0.021. 27/28 [0.823, 0.994], 28/28, 27/28. | −1.9e-4 [−4.8e-4, +7.9e-5], **collapsed** | −0.073 ± 0.122 over 2 clients; **1 of 3 clients failed** (client_0 did not train) |
| **K=3 Dirichlet α = 0.1** | **22/28 [0.605, 0.898]** | **0.452 ± 0.297.** 10/28 [0.207, 0.542]; 6/28 [0.102, 0.395]; 22/28 [0.605, 0.898]. | −0.272 [−3.937, +0.258], trained | −2.237 ± 3.767, 0 of 3 clients failed |

What the table shows (description, not interpretation):
- **Classification is at ceiling** (28/28) in every FL and centralized condition except
  Dirichlet. Local-only falls below FL under natural K=2 and Dirichlet.
- **No regression skill interval excludes zero in the positive direction.** Every FL
  regression run that trained has a CI containing 0. The local-only means are at or
  below 0.
- **The channel axis grades monotonically for FL regression:** ρ = 1.0 trained,
  ρ = 0.375 did not train, ρ = 0 collapsed. FL classification is unaffected (28/28 at
  every ρ).

### Regression detail

Per-source skill uses intervals resampled within the source (Test_2: 13 runs; 20251124:
15 runs). "Gated MSE" is the value the trained gate read.

| Run | Label | Skill, overall | Skill, Test_2 | Skill, 20251124 | RMSE (m/s) | R² within subject | Gated MSE | Spread ratio |
|---|---|---|---|---|---|---|---|---|
| Centralized residual 4601 | trained | +0.121 [−0.288, +0.235] | −0.042 [−0.687, +0.245] | +0.136 [−0.348, +0.295] | 0.0380 | 0.050 | 0.716 | 0.386 |
| Centralized residual 4602 | trained | +0.082 [−0.097, +0.186] | +0.070 [−0.384, +0.331] | +0.083 [−0.121, +0.208] | 0.0388 | 0.007 | 0.855 | 0.209 |
| Centralized residual 4603 | trained | +0.091 [−0.034, +0.165] | +0.045 [−0.195, +0.219] | +0.095 [−0.064, +0.194] | 0.0386 | 0.017 | 0.922 | 0.116 |
| Centralized absolute 4601 | trained (no trained gate) | −1.485 [−5.574, −0.848] | −3.855 [−10.762, −1.115] | −1.264 [−5.893, −0.638] | 0.0639 | −1.687 | n/a | 0.921 |
| Centralized absolute 4602 | trained (no trained gate) | −1.478 [−5.760, −0.772] | −4.213 [−11.966, −1.081] | −1.224 [−5.952, −0.565] | 0.0638 | −1.680 | n/a | 0.911 |
| Centralized absolute 4603 | trained (no trained gate) | −2.254 [−7.358, −1.382] | −4.985 [−12.054, −1.395] | −2.000 [−7.919, −1.159] | 0.0731 | −2.519 | n/a | 0.923 |
| FL K=2 IID 4601 | trained | +0.089 [−3.123, +0.576] | −1.095 [−1.950, −0.196] | +0.199 [−4.074, +0.644] | 0.0387 | 0.015 | 0.0091 | 1.48 |
| FL K=2 IID 4602 | trained | +0.150 [−2.074, +0.543] | −2.369 [−4.652, −0.485] | +0.384 [−1.699, +0.630] | 0.0374 | 0.081 | 0.0013 | 1.51 |
| FL K=2 IID 4603 | trained | +0.277 [−2.307, +0.658] | −0.517 [−1.538, +0.384] | +0.351 [−3.236, +0.709] | 0.0344 | 0.218 | 0.0084 | 1.30 |
| FL K=2 natural 4601 | trained | −0.207 [−4.128, +0.334] | −1.293 [−3.123, −0.183] | −0.106 [−5.441, +0.426] | 0.0445 | −0.305 | 0.119 | 1.18 |
| FL K=2 natural 4602 | trained | −0.123 [−4.367, +0.501] | −1.225 [−2.261, −0.209] | −0.020 [−5.952, +0.572] | 0.0429 | −0.214 | 0.0398 | 1.51 |
| FL K=2 natural 4603 | **collapsed** | −1.1e-6 [−3.5e-6, +4.4e-7] | ≈ 0 | ≈ 0 | 0.0405 | −0.081 | 1.000 | 1.3e-10 (205 distinct values) |
| FL K=3 ρ=1.0 | trained | +0.101 [−2.263, +0.480] | −0.443 [−1.417, +0.276] | +0.152 [−3.103, +0.543] | 0.0384 | 0.028 | 0.0549 | 1.01 |
| FL K=3 ρ=0.375 | **did not train** | +0.046 [−0.025, +0.077] | +0.039 [−0.032, +0.189] | +0.047 [−0.049, +0.082] | 0.0396 | −0.032 | 0.991 | 0.064 |
| FL K=3 ρ=0 | **collapsed** | −1.9e-4 [−4.8e-4, +7.9e-5] | −0.001 | ≈ 0 | 0.0405 | −0.082 | 1.000 | 8.3e-11 (19 distinct values) |
| FL K=3 Dirichlet | trained | −0.272 [−3.937, +0.258] | −0.986 [−2.747, −0.322] | −0.206 [−5.320, +0.397] | 0.0457 | −0.376 | 0.314 | 1.14 |

- **Reproductions:**
  - FL K=2 natural seed 4601 reproduces the S3 confirmation run exactly: −0.207
    [−4.128, +0.334].
  - FL K=3 ρ=1.0 reproduces the S3 IID K=3 winner: +0.101 [−2.263, +0.480].
  - Centralized residual 4601 reproduces S3/S4: +0.121.
- **Test_2 skill has an interval entirely below 0** in:
  - FL K=2 IID seeds 4601 and 4602;
  - FL K=2 natural seeds 4601 and 4602;
  - FL Dirichlet;
  - all three centralized absolute runs.

  No centralized residual run is in that list.

**Local-only regression, per client.** "Own" is the own-data standardized MSE that the
trained gate reads; "global" is reported, not gated.

| Condition | Client | Label | Skill | RMSE (m/s) | Own | Global |
|---|---|---|---|---|---|---|
| K=2 IID | client_0 | trained | +0.263 [−0.445, +0.386] | 0.0348 | 3.1e-6 | 0.472 |
| K=2 IID | client_1 | trained | −0.243 [−0.936, −0.012] | 0.0452 | 0.525 | 0.764 |
| K=2 natural | client_0 (Test_2) | trained | −0.112 [−0.556, +0.015] | 0.0427 | 1.1e-6 | 0.827 |
| K=2 natural | client_1 (20251124) | trained | −1.341 [−10.652, +0.031] | 0.0620 | 2.0e-6 | 3.235 |
| K=3 ρ=1.0 | client_0 | trained | +0.199 [−2.157, +0.557] | 0.0363 | 0.0017 | 0.799 |
| K=3 ρ=1.0 | client_1 | trained | −0.089 [−1.402, +0.146] | 0.0423 | 0.516 | 0.832 |
| K=3 ρ=1.0 | client_2 | trained | −0.484 [−4.415, +0.090] | 0.0494 | 0.097 | 0.976 |
| K=3 ρ=0.375 | client_0 | trained | −0.740 [−3.094, −0.126] | 0.0535 | 0.269 | 1.380 |
| K=3 ρ=0.375 | client_1 | trained | +0.033 [−1.093, +0.236] | 0.0399 | 5.1e-6 | 1.003 |
| K=3 ρ=0.375 | client_2 | trained | −0.280 [−3.265, +0.201] | 0.0458 | 0.023 | 0.881 |
| K=3 ρ=0 | client_0 | **did not train** | −0.014 [−0.074, −0.003] | 0.0408 | 0.980 | 0.992 |
| K=3 ρ=0 | client_1 | trained | −0.159 [−2.025, +0.172] | 0.0436 | 0.012 | 1.106 |
| K=3 ρ=0 | client_2 | trained | +0.013 [−1.149, +0.221] | 0.0403 | 0.282 | 0.949 |
| K=3 Dirichlet | client_0 | trained | −0.243 [−1.618, +0.037] | 0.0452 | 0.102 | 1.046 |
| K=3 Dirichlet | client_1 | trained | −6.582 [−36.318, −1.799] | 0.1116 | 4.5e-5 | 12.616 |
| K=3 Dirichlet | client_2 | trained | +0.114 [−2.059, +0.461] | 0.0382 | 1.1e-5 | 0.957 |

### Classification detail

- **Window accuracy:**
  - centralized 0.9967 / 1.0000 / 0.9967 (mean 0.9978 ± 0.0019);
  - K=2 IID FL 0.9967 / 1.0000 / 0.9967;
  - K=2 natural FL 0.9902 / 0.9772 / 0.9935 (mean 0.9870 ± 0.0086);
  - K=3 ρ = 1.0 / 0.375 / 0: 0.9967 / 0.9674 / 0.9772;
  - Dirichlet 0.8274.
- **Run level:** majority vote and mean probability give the same run counts everywhere
  except Dirichlet FL, where both are 22/28 and macro-F1 is 0.641. Every other FL or
  centralized model has run macro-F1 1.000.
- **Dirichlet FL:**
  - by source: Test_2 13/13 [0.772, 1.000]; 20251124 9/15 [0.357, 0.802];
  - all 6 errors are whole subjects: 005 → 003 (3/3 runs) and 007 → 004 (3/3 runs);
  - at split seed 4601, 005's runs sit only on client_0 (13 of its 21 runs) and 007's
    only on client_1 (11 of 15). client_2 holds 76 of the 112 training runs.
- **Local-only by subject:**
  - **K=2 natural, client_0 (Test_2 only):** every Test_2 run correct (13/13), and every
    20251124 run, 003 included, labelled 002.
  - **K=2 natural, client_1 (20251124 only):** every 20251124 run correct (15/15). On
    Test_2: 001 → 008 (4/4), 002 → 007 (4) / 004 (1), 003 → 003 (3/4).
  - **Dirichlet:** client_2 is 22/28, missing 005 and 007. client_0 is 10/28. client_1 is
    6/28 and labels every Test_2 run 004.

### Source probe (§2 R4 item 2): `source_probe/seed4601`

This is the subject classifier's recipe with label = source (2 classes), on the same
windows and split.

| | Result |
|---|---|
| Test window accuracy | **1.000**, on no_walking (ambient) windows (69) and walking windows (238) alike |
| Run level | 28/28 [0.879, 1.000]; Test_2 13/13 [0.772, 1.000]; 20251124 15/15 [0.796, 1.000] |
| Majority class of test windows | 0.537 |
| Final train accuracy | 1.000 |

The two collection campaigns are perfectly separable, including on windows with nobody
walking. In the 1-epoch smoke run, test accuracy was already 1.000.

### Subject-003 cross-session control (§2 R4 item 3): `subject003_control/seed4601`

- **Training:** walking windows from the 52 Test_2 canonical-train runs (431 windows),
  3 classes (001, 002, 003).
- **Normalization:** from those windows only.
- **Final train accuracy:** 1.000.

| Set | Runs | Windows | Majority-vote result | Predicted labels by window |
|---|---|---|---|---|
| In-session (Test_2 canonical test) | 13 | 103 | **13/13** [0.772, 1.000]; mean probability 13/13 | each subject 1.00 to its own label |
| **Cross-session: 003 in 20251124** | 15 | 142 | **0/15 [0.000, 0.204]**; mean probability 0/15 | 002 0.93, 001 0.07, **003 0.00**. Runs: 14 → 002, 1 → 001. |
| Negative control 004 | 15 | 127 | not in the classes | 002 1.00 (15 runs → 002) |
| Negative control 005 | 16 | 134 | not in the classes | 002 0.88, 001 0.12 (15 → 002, 1 → 001) |
| Negative control 007 | 14 | 125 | not in the classes | 002 1.00 (14 → 002) |
| Negative control 008 | 15 | 142 | not in the classes | 002 0.94, 001 0.06 (15 → 002) |

Trained on Test_2 alone, the model labels 003's 20251124 recordings exactly as it labels
four people it has never seen. Not one 20251124 window is predicted 003.

**The free first look**, from the existing predictions (`derived/cross_session_free_breakdown.csv`):
- Every centralized and FL model, all seeds and all conditions including Dirichlet,
  classifies 003's test runs correctly in both sources: Test_2 4/4 and 20251124 3/3.
- Those models saw 003 from both sessions in training, which is why the runbook calls
  this look not conclusive.

**A reverse-direction look, unplanned, from the existing local-only K=2 natural run.**
Recorded here because it bears on the control:
- **client_1** trained only on 20251124 (subjects 003, 004, 005, 007, 008). It labels
  003's Test_2 runs **003 in 3 of 4 by majority vote** (Wilson [0.301, 0.954]), but
  **2 of 4 by mean probability** ([0.150, 0.850]).
- **client_0** trained only on Test_2. It labels every 20251124 run 002, 003 included,
  as the control does.
- These are n = 4 runs, from a model with 5 subject classes rather than 3. This is a
  single observation and not a planned control.

**Negative control for the reverse direction.** client_1's predictions on the Test_2
canonical-test runs, walking windows only. 001 and 002 are not among its training
subjects. Counted from its saved `test_predictions.csv` and `run_level_predictions.csv`:

| Test_2 subject | Runs | Windows | Window-level predicted labels | Runs, majority vote | Runs, mean probability |
|---|---|---|---|---|---|
| 001 (unseen) | 4 | 32 | 008 1.000 (32) | 4 → 008 | 4 → 008 |
| 002 (unseen) | 5 | 43 | 007 0.698 (30), 004 0.279 (12), **003 0.023 (1)** | 4 → 007, 1 → 004 | 4 → 007, 1 → 004 |
| **003** (seen in 20251124) | 4 | 28 | **003 0.571 (16)**, 004 0.429 (12) | **3 → 003**, 1 → 004 | **2 → 003**, 2 → 004 |

**Per run for 003** (windows predicted 003 / total; majority; mean probability):
- `test_2:003:006`: 5/7; 003; 003 (0.79)
- `test_2:003:011`: 4/7; 003; **004** (0.51)
- `test_2:003:013`: 6/8; 003; 003 (0.77)
- `test_2:003:018`: 1/6; 004; 004 (0.71)

No majority vote was tied.

**What the table shows:**
- **"003" is predicted almost only on 003's own windows:** 16 of 28 of them. Among the
  75 windows of the two unseen subjects there is 1 window (002) and no run.
- **The unseen subjects aren't pooled onto one label.** 001 maps entirely to 008 and 002
  mostly to 007, not to 003 and not to a single session label. That differs from the
  forward control, where all four unseen 20251124 subjects and 003 went to 002.
- **The 003 signal is weak.** Only 57% of 003's windows, 3/4 runs by vote and 2/4 by mean
  probability.

So in this direction the "003" label isn't simply what the model gives an unfamiliar
Test_2 recording. This is still one local-only model, 4 runs, unplanned, with wide
intervals, and it isn't a replacement for the planned control. The two directions also
differ in classes (5 vs 3), training size, and whether no_walking windows were in
training (yes for the local-only model, no for the control).

### Gate outcomes

| | Count | Models |
|---|---|---|
| **Collapsed** | 2 | FL K=2 natural seed 4603 (spread 1.3e-10, 205 distinct values); FL K=3 ρ=0 (spread 8.3e-11, 19 distinct values) |
| **Did not train** | 2 | FL K=3 ρ=0.375 (global-train MSE 0.991, spread 0.064); local-only K=3 ρ=0 client_0 (own-data 0.980, spread 0.035) |
| **Trained** | all other regression models | 6 centralized; FL: K=2 IID ×3, K=2 natural ×2, K=3 ρ=1.0, Dirichlet; local-only: 15 of 16 clients |

- Per-seed reporting applies to one group: FL K=2 natural regression, "per seed; 1 of 3
  seeds collapsed".
- Every other multi-seed group is "mean ± std; 3 of 3 seeds trained" (regression) or
  plain mean ± std (classification).
- **FLAT** flags: the 3 FL regression runs above that did not train or collapsed, and
  local-only K=3 ρ=0 regression. **No classification run was flagged FLAT.**
- **Job statuses:** 36 `completed`, 3 `failed_detector` (the three FL regression runs
  above), 1 `completed_with_failed_clients` (local-only K=3 ρ=0 regression),
  3 `reused_canonical`.
- **31 Oct fallback:** not triggered; nothing was dropped.

**Numbers that moved:** none of the earlier reported numbers. Everything above is new S5
output; the reproductions are noted under the regression detail.

---

## Session 5 (amendment 2) — Local-only trained gate, three outcome labels (2026-10-21)

Night 2 (2026-10-20 23:03 → 2026-10-21 05:03) brought the queue to 33 of 43 jobs. Two
corrections followed, both PI-approved. **They change labels and gates only; no number
moved.** Of the 613 numeric metric values in the re-derived job records, every one is
identical to its value before re-derivation.

### 1. Local-only clients are judged "trained" on their own training data

**The defect:** a local-only client was gated on the *global-train* standardized MSE. That
value covers every client's training runs, most of which a local-only client never sees,
so it measures generalization to other silos, not whether the client trained.

Job 14 (`local_only/k2_natural/seed4601/regression`) shows it:

| Client | Global-train standardized MSE | Old gate | Own data |
|---|---|---|---|
| client_1 (20251124) | 3.235 | "failed" | own-data loss 1.3765 → 0.0000 |
| client_0 (Test_2) | 0.827 | passed | — |

The runner therefore averaged over client_0 alone.

**Rule** (`guardrails.own_data_standardized_mse`):

    own-data standardized MSE = own-data MSE at the end of the last round
                                / own-data MSE at the start of round 1      < 0.95

- **Why it is built this way:** the residual model's last layer is zero-initialized, so
  the round-1 start is exactly the client's own "predict zero residual" MSE. 1.0 keeps
  that meaning, and the threshold stays at `MAXIMUM_STANDARDIZED_MSE`. Both losses are
  measured by the client on its own runs, with its channel mask.
- **Why not a bare 0.95 on the raw own-data value:** client_0's untrained start (0.566)
  is already below 0.95, and client_1's (1.376) is above it.
- **What is still enforced and reported:** the output-spread degeneracy check is still
  enforced. The global-train MSE is reported per client and not gated.
- **Federated and centralized gates are unchanged.**

**In the runner** (`_finalize_regression`, `run_local_only_experiment`):
- Local-only clients pass the own-data value to the trained gate and record
  `trained_gate_basis`.
- `local_only_summary.json` gains `clients_failed` ("k of n clients failed"), a
  `training_outcome` per client, and the gate basis.
- mean ± std covers every client that passes the gates.

**In the queue:**
- Local-only metrics are mean ± std over the trained clients, plus `num_clients_in_mean`,
  `clients_failed` and `client_outcomes`.
- `--derived-only` rewrites a local-only regression `local_only_summary.json` in the
  runner's layout, so the two agree. The run-time file is kept once as
  `local_only_summary.at_run.json`.
- A local-only job's outcome now follows its re-derived client gates. The run-time value
  is kept as `outcome_at_run`.

### 2. Three outcome labels replace "collapsed"

`guardrails.training_outcome` assigns one label per regression model, from the current
detector re-applied to the saved predictions:

| Label | Meaning |
|---|---|
| `collapsed` | fails degeneracy: the output-spread check |
| `did not train` | has spread, but fails `trained` |
| `trained` | passes both |

- The absolute path has no trained gate, so it is labelled on degeneracy alone.
- **Index and tables:** `results_index.csv` gets a `training_outcome` column, replacing
  `collapsed`. `derived/seed_aggregates.csv` gets `training_outcome_seed<N>`.
- **Per-seed reporting:** any seed that is not `trained` puts its group on per-seed
  reporting ("k of n seeds collapsed" / "… did not train").
- **FLAT** now means "not trained".

### Re-derivation (`--derived-only`, no retraining): labels that changed

| Model | Before | Now |
|---|---|---|
| `local_only/k2_natural/seed4601/regression` client_1 | collapsed (global-train gate) | **trained** (own-data 2.0e-6) |
| `local_only/k2_natural/seed4601/regression`, the job | `completed_with_failed_clients` | **`completed`**: 0 of 2 clients failed, both clients in the mean, skill −0.726 ± 0.869 |
| `fl/k3_rho0.375/seed4601/regression` | collapsed | **did not train**: spread 0.064, global-train standardized MSE 0.991 |

- **Unchanged:** `fl/k3_rho0/seed4601/regression` and `fl/k2_natural/seed4603/regression`
  stay **collapsed** (spreads 8.3e-11 and 1.3e-10). The other 15 regression models stay
  `trained`.
- **Job 14's skill mean doesn't change:** the queue's figure already included both
  clients. Only the runner summary's mean changes, from client_0 alone to both clients.
- **The spread rule still holds:** the S5 centralized residual seeds sit at spread
  ratios 0.209 (4602) and 0.116 (4603). These are lower than the 0.386 cited when the
  1e-3 threshold was fixed, but still about 116× above it.
- **Tests:**
  - **Fixtures:** `test_fixtures/s5_local_only_k2_natural_seed4601_regression.json`
    (job 14, both clients) and `test_fixtures/s5_k3_rho0.375_seed4601_regression.json`,
    both copied verbatim from the gitignored run folders.
  - **Covered:**
    - the three labels on real runs (4603 collapsed, ρ=0.375 did not train);
    - the own-data rule and why a bare 0.95 fails;
    - job 14 is trained under the new rule, and client_1 did not train under the old one;
    - local-only aggregation and "k of n clients failed".
- `pytest` (full suite, data present): **175 passed, 1 skipped** (cudnn, macOS), in
  5 min 49 s.

---

## Session 5 (amendment) — Seed-4603 collapse, per-seed reporting, degeneracy spread rule (2026-10-20)

The first queue night (2026-10-19 20:56 → 2026-10-20 05:14, stopped by `--stop-by 06:00`)
finished 13 of 43 jobs: all 12 K=2 FL jobs and local-only K=2 natural classification.
One job collapsed. The full results get their own entry when the queue completes.

### Natural K=2, init seed 4603, regression collapsed: reported as measured

`fl/k2_natural/seed4603/regression` → `failed_detector`, FLAT.
- **The numbers:**
  - global-train standardized MSE is 1.0000000 in every one of the 60 rounds;
  - skill −1.1e-6;
  - 205 distinct window residuals, but their std is 5.2e-12 m/s.
- **This is the R1 mechanism on one seed.** In every round, each client's local loss ends
  where it started (client_0 0.5656 → 0.5656, client_1 1.3765 → 1.3765).
- **Seed 4601 starts on the same plateau and leaves it.** By round 10 client_1 moves
  (1.3637 → 1.2929), and by round 60 the standardized MSE is 0.119.
- **IID K=2 seed 4603 trained normally** (skill +0.277), so the collapse is that seed
  combined with the natural partition.
- With 1 collapse in 3 seeds against 0 in 3, no causal claim about non-IID is supported.

**PI decision:**
- Report it as measured. The folder and its failure record are kept unchanged, and it is
  not re-run.
- Natural K=2 regression is reported **per seed, not as mean ± std**, with the collapse
  counted as **"1 of 3 seeds collapsed"**.
- **Why:** with n = 3 and a collapsed run whose skill is ≈ 0, a mean would sit above both
  trained seeds. (−0.207, −0.123 and 0.000 average to −0.110.)

**Implemented:**
- **The rule is general:** any multi-seed group containing a collapsed seed is reported
  this way. Today that is only natural K=2 regression.
- **Definition:** "collapsed" means the current detector, re-applied to the saved
  predictions, fails `degeneracy` or `trained`.
- **`results_index.csv`** gains `collapsed` and `seed_reporting` columns.
- **`derived/seed_aggregates.csv`** gains a `reporting` column:
  - a collapsed group gets per-seed columns (`skill_seed4601`, …, `collapsed_seed4603`)
    and no mean or std;
  - every other group keeps mean ± std.
- The FLAT flag for regression now means "collapsed".

### Degeneracy gate: relative spread rule (`guardrails.check_regression_health`)

**Why the gate missed it:** a distinct-value count is fooled by floating-point noise.
4603's 205 distinct values passed the "> 10 distinct" check; only the `trained` gate
caught it.

**New condition:** the degeneracy gate also fails when

    std(model output) / std(target) < 1e-3   (MINIMUM_RELATIVE_OUTPUT_SPREAD)

- **Model output and target:** as before, the model output is the window predicted
  residual on residual paths and the predicted speed on the absolute path. The target is
  what the network is trained to emit, over the evaluated runs: `actual − lookup` for
  residuals, `actual` for absolute speed.
- **The ratio is scale-free.** The report gains `target_std`, `relative_output_spread`
  and the threshold.

**Why 1e-3 and not 1e-6** (proposed by the PI): the ratio was measured on every
regression run on disk *before* the threshold was fixed.

| Runs | Ratio |
|---|---|
| Every trained run: centralized absolute, residual 60e and 300e; all S3 sweep runs that trained; the 5 trained S5 runs | 0.39 – 1.78 (weakest: centralized residual 60e, 0.386) |
| R1 collapse, S3 E=1 fresh, S3 E=5 fresh | 0 |
| S3 E=1 persistent (standardized MSE 1.0) | 1.2e-5 |
| S5 natural K=2 seed 4603 | 1.3e-10 |

1e-6 would have passed the S3 E=1 persistent collapse. 1e-3 sits about 80× above the
highest collapse and about 390× below the weakest trained run.

**Re-applied without retraining** (`--derived-only`, which now re-collects every finished
job's metrics with the current code and stamps `metrics_recomputed_at`):

| | Result |
|---|---|
| Regression jobs re-scored | all 6 finished S5 regression jobs |
| Passing (degeneracy, trained) | 5, ratios 1.18 – 1.51 |
| Failing | **only `fl/k2_natural/seed4603/regression`**: degeneracy ✗ (it passed at run time), trained ✗ |

- Classification job metrics were recomputed and are unchanged.
- The seed-4601 centralized runs that S5 reuses (group 3, not yet reached) sit at 0.93
  (absolute) and 0.386 (residual), so they pass.
- **Tests:**
  - `test_fixtures/s5_k2_natural_seed4603_regression.json` holds 4603's real
    predictions, copied verbatim from the gitignored run folder;
  - the tests show 4603 passes the distinct-count check but fails the spread rule, and
    raises when degeneracy alone is enforced;
  - they check the threshold on both sides of 1e-3 and at the two measured boundary
    ratios (1.2e-5 fails, 0.386 passes);
  - they check the per-seed reporting rule and the per-seed aggregate table.

`pytest` (full suite, data present): **171 passed, 1 skipped** (cudnn, macOS).

**No reported number moved.** No completed job changes outcome. The gate only changes for
4603, which had already failed on `trained`.

---

## Session 5 — Experiment queue, run-level metrics, source probe, 003 control (2026-10-19)

Branch `s5-experiments`. This entry covers the code and configs; no paper experiment
has been run yet, and **no reported number moved.** The queue's results get their own
entry. The PI amended the runbook's S5 section:

- **Setting for every FL and local-only run:** FedAvg, E=5, persistent client
  optimizer, `client_num_cpus: 3`, both tasks, global 9-channel evaluation only.
- **K=2 IID and K=2 natural** run at init seeds 4601/4602/4603 (split seed 4601 always).
- **K=3** ρ = 1.0 (this is IID K=3, run once), 0.375 and 0, plus **Dirichlet α=0.1**, run
  at init seed 4601.
- **Local-only** runs once for each of the 6 partition schemes.
- **Centralized** classification, absolute and residual are added at init seeds 4602 and
  4603, so centralized also has 3 seeds.

Gates are as in S4: degeneracy and trained are enforced, skill is reported and never
enforced.

### The queue: `scripts/run_s5_queue.py` (logic in `src/redo_by_sara/s5_queue.py`)

**43 jobs: 40 training jobs and 3 reuses.** Each job is one task, so a restart and the
fallback work per task. The order, most important first:

| Group | Jobs | First estimate |
|---|---|---|
| 1 | FL K=2 natural and IID × 3 seeds × 2 tasks | 12 jobs, 7.7 h |
| 2 | Local-only K=2 natural and IID × 2 tasks | 4 jobs, 4.3 h |
| 3 | Centralized seed 4601, reused (3); centralized seeds 4602/4603 (6); source probe (1); 003 control (1) | 8 training jobs, 1.6 h |
| 4 | FL K=3 ρ = 1.0, 0.375, 0 × 2 tasks | 6 jobs, 3.1 h |
| 5 | FL Dirichlet α=0.1 × 2 tasks | 2 jobs, 1.8 h |
| 6 | Local-only K=3 ρ = 1.0, 0.375, 0 and Dirichlet × 2 tasks | 8 jobs, 10.8 h |
| | **Total** | **≈ 29.4 h** |

**Where the estimates come from:**
- **Natural K=2 FL:** measured in S3 at this exact setting: classification 42.5 min,
  regression 33.7 min.
- **IID K=3 regression:** measured at 28.1 min.
- **Everything else is scaled.** Local-only is the least certain: its clients train one
  after another, and it has never run past 2 rounds.
- **Recalibration:** once a job of the same timing class finishes, the queue uses its
  measured wall time instead.

**Output:**
- Everything goes under `artifacts/s5/<family>/<condition>/seed<N>/<task>/`, one folder
  per job, each with `run.log` and `job.json`. FL and local-only jobs pass `--set
  init_seed=N` to the runner.
- **Logs:** `queue.log` holds commands, exit codes and timestamps. `status.log` holds one
  line per finished job, which is also printed.
- **Tables:** `results_index.csv` has one row per job, with the FLAT flag and the headline
  metrics. `derived/seed_aggregates.csv` has mean ± std over seeds.
  `derived/run_level_classification.csv` and `derived/cross_session_free_breakdown.csv`
  hold the run-level results.
- Per-job folders are gitignored; the index, the status log and `derived/` are tracked.

**Restart:**
- A terminal `job.json` is skipped. The terminal outcomes are `completed`,
  `completed_with_failed_clients`, `failed_detector`, `dropped_fallback` and
  `reused_canonical`.
- A killed job's folder has no `job.json`. It is renamed to `*.incomplete-<timestamp>`,
  never deleted, and re-run. A crashed job is renamed to `*.crashed-<timestamp>` and
  re-run.

**Interruption:**
- SIGINT or SIGTERM stops the child cleanly: SIGINT to its process group, then SIGKILL
  after 60 s.
- Each child runs under `caffeinate -i`. That stops idle sleep but not lid-close sleep.

**Preflight:**
- refuses to start if `src/`, `scripts/` or `configs/` have uncommitted changes;
- checks flwr 1.29.0, ray 2.51.1 and the conda env `fl`;
- checks that the canonical inputs exist;
- refuses a results root that is, or is inside, a canonical artifacts folder;
- checks that every S5 FL config carries the winner setting, and builds and audits each
  partition with all writes disabled.

**`--stop-by HH:MM`:**
- It means the next occurrence of that time: started at 21:00, `--stop-by 05:00` means
  05:00 tomorrow.
- Before starting a job, the queue checks whether its estimate × 1.10 would end after
  that time. If so, it stops cleanly and prints the job it did not start, what is left,
  and the next job in line.
- `--dry-run --stop-by` shows which jobs fit tonight.

**Fallback (S3 "S5 scope"), automatic by the clock:** from 31 Oct 2026 08:00, any
not-yet-started Dirichlet regression jobs, then ρ = 0.375 regression jobs, are recorded as
`dropped_fallback`. This covers FL and local-only. Classification still runs.

**FLAT** is a flag; the queue keeps going. It shows in the status line and in
`results_index.csv` when:
- a classification run's global-train accuracy rises by less than 0.05 from round 0;
- a classification run predicts a single test class;
- a centralized classifier's final train accuracy is below chance + 0.05;
- a regression run failed the `trained` gate.

Degradation at ρ=0 and α=0.1 is a result, not a flag.

**Seed-4601 centralized is reused, not re-run.** The committed run is scored read-only;
S4 showed a re-run reproduces it byte for byte.
- The committed regression summaries predate S4's `regression_report`, so it is computed
  from the committed `test_run_predictions.csv`.
- The residual run's `trained` gate is derived from its recorded standardized MSE
  (0.716 < 0.95).
- **Scored this way, the residual reproduces S3/S4 exactly:** skill +0.121
  [−0.288, +0.235].

### Configs

- **FL:** `configs/s5_{k2_natural, k2_iid, k3_iid, k3_rho0.375, k3_rho0, k3_dirichlet0.1}.yaml`.
  Each carries the winner setting and explicit `split_seed: 4601`, `init_seed: 4601`.
  `k3_iid` has no `channel_overlap` key, so the ρ = 1.0 condition is the plain IID run.
- **Centralized:** `configs/s5_centralized_{classification, absolute, residual}_init{4602, 4603}.yaml`.
  Each residual config points `baseline_summary_path` at the absolute run with the same
  seed (a comparison field only). The queue runs absolute before residual.
- **Bug caught by a test:** every config loader resolves relative paths from
  `config_path.parents[1]`. The configs were first written to `configs/s5/` and would have
  read `configs/artifacts/...`. They now live in `configs/` directly. A test asserts the
  resolved artifact paths.

### Run-level classification metrics (§2 R4 item 1): `run_level_classification.py`

- **Unit:** the held-out run, 28 of them, using the windows whose true class is a subject.
  A run whose vote is won by `no_walking` counts as wrong.
- **Majority vote:** a tie goes to the higher mean probability over the run, then to the
  lower class index.
- **Mean probability:** the argmax of the run's mean softmax.
- **Reported:** accuracy with a Wilson 95% interval, macro-F1, per-source results, and a
  subject × source breakdown. The free subject-003 look is that breakdown filtered to
  subjects recorded in more than one source; no subject ID is hard-coded.
- **No training code was touched.** Probabilities are recomputed from each run's saved
  checkpoint with the normalization stored in it. Each window's argmax is **asserted to
  equal the saved `predicted_class`**.
- **Measured on the committed seed-4601 centralized classifier** (a new number, not a
  moved one): 28/28 runs, Wilson [0.879, 1.000]; Test_2 13/13 [0.772, 1.000]; 20251124
  15/15 [0.796, 1.000]. Majority vote and mean probability agree. Window accuracy is 0.9967.

### Source probe (§2 R4 item 2): `source_probe.py`, `scripts/run_source_probe.py`

- An in-memory copy of the classification artifact with label = source. The windows, the
  seed-4601 split, the normalization and the centralized recipe are unchanged.
- Reports window accuracy, split into no_walking and walking windows, and run-level
  accuracy over all of a run's windows.

### Subject-003 cross-session control (§2 R4 item 3): `subject003_control.py`, `scripts/run_subject003_control.py`

This is a separate evaluation split under the CLAUDE.md carve-out: an in-memory derived
artifact, with `assign_run_splits`, the seed-4601 split and the saved artifacts untouched.

| Set | Runs | Windows | Contents |
|---|---|---|---|
| train | 52 | 431 | Test_2 canonical-train runs, walking windows, 3 classes (001, 002, 003) |
| in-session test | 13 | 103 | Test_2 canonical-test runs |
| cross-session test | 15 | 142 | **all** of 003's 20251124 runs, canonical train and test. The model never sees 20251124. |
| negative control | 60 | 528 | 20251124 subjects 004, 005, 007, 008, evaluation only |

- **Normalization is recomputed from the control's train windows only.** The canonical
  statistics include 20251124 windows. The recipe is tested to reproduce the canonical
  statistics bitwise when given the canonical train indices.
- **The negative control (approved by the PI)** reports the predicted-label distribution
  for each subject, at window level and run level, next to 003's. This shows whether a
  cross-session "hit" is the person or the session.

### Verification

- **`test_s5.py`, 26 tests:**
  - Wilson intervals against closed-form values;
  - the vote, tie and background rules, and the macro-F1;
  - the job matrix: 43 jobs, 40 training jobs, the group sizes, the order, absolute before
    residual, the fallback levels, and every `--output-root` under the root;
  - every S5 config's setting and resolved paths;
  - the `--stop-by` next-occurrence rule and margin, the fallback clock, estimate
    recalibration, and outcome detection.
  - With data, they also check the control's split literals and its separate
    normalization, the probe relabel, and run-level scoring of the canonical checkpoint.
- **Smoke run with kill and restart** (`--smoke`, 1 round or 1 epoch, scratch root, the
  fallback clock forced into the past):
  - The queue was killed with SIGTERM during `fl/k2_natural/seed4601/regression`. It
    exited 130, the child stopped in 9 s, and no Ray processes were left.
  - The restart skipped the finished classification job, set the killed folder aside as
    `regression.incomplete-…`, re-ran it, and completed the rest.
  - **Ran:** local-only (both tasks), the three reused seed-4601 jobs, the probe, the
    control, and FL ρ=0.375 and Dirichlet classification.
  - **Dropped by the fallback:** ρ=0.375 and Dirichlet regression.
  - **Not run in smoke:** the centralized runners, which pin 60 epochs and are covered by
    the S4 end-to-end tests.
  - A third start skipped every job.
  - With `--stop-by` 30 minutes ahead, the queue ran a job estimated at 0.9 min, then
    refused the 90-min local-only K=3 job and printed the next job in line.
- `pytest` (full suite, data present): **166 passed, 1 skipped** (cudnn, macOS), in
  6 min 34 s. That is the S4 140 plus the 26 new tests.

## Session 4 — Metrics, seeds, local-only, channel masks, Dirichlet (2026-10-19)

Branch `s4-metrics`. Covers `docs/review_response_plan.md` §2 R2, R5, R6 and §1.3, plus
the Dirichlet label-skew axis. Three rulings amended the plan:

- ρ = 0.375, not 0.33;
- skill is enforced in no S5 condition;
- one local-only run per partition scheme.

### 1. One regression metrics implementation (R2)

`src/redo_by_sara/regression_metrics.py` replaces four copies: `regression_metrics`,
`_extended_metrics`, `_extended_regression_metrics`, and the plot module's
`_regression_metrics`.

- **`point_metrics`** keeps every existing key, plus `median_absolute_error_mps`.
- **`regression_report`** is run-level:
  - overall and per-source: baseline RMSE, model RMSE, ΔRMSE, R² (total), R² (within
    subject), and **skill with a run-level bootstrap 95% CI** (10,000 resamples,
    seed 4601, identical draws across models; per-source intervals resample within the
    source);
  - the **per-subject × source breakdown**: number of test runs, model RMSE, lookup
    RMSE, and their difference.

  Groups come from the IDs passed in; a test asserts that no subject or source ID
  appears in the module.
- **Within-subject R²** is 1 − SSE_model / Σ(r − r̄)², with r = actual − lookup. Without
  the centering it would equal skill.
- **Every regression output now carries it:** `training_summary.json` →
  `regression_report`, plus `test_per_subject_breakdown.csv`. This covers centralized
  absolute, centralized residual, federated, and local-only runs, and federated failure
  records.
- It reproduces the S3 analysis exactly. Centralized 60e: +0.121 [−0.288, +0.235];
  Test_2 −0.042; 20251124 CI [−0.348, +0.295].

### 2. Detector policy for S5

The S3 "skill" gate is split in two: **`trained`** (final standardized MSE < 0.95) and
**`skill`** (skill > 0). **S4/S5 policy: `degeneracy` and `trained` are enforced on
every residual path; `skill` is reported with CIs for every run and enforced in no S5
condition**, IID FL and centralized included.

**Why:** S5 runs multiple seeds. Enforcing skill would silently drop the seeds whose
skill is negative and bias the reported mean upward. The skill gate was the S3
acceptance criterion for the R1 fix, and it stays recorded there.

`r1_sweep` now computes the three S3 conditions from the report's numbers, so S3
results are read as before regardless of which gates a run enforced.

### 3. Split seed and initialization seed (R6)

- New config keys **`split_seed`** (run split and every partition) and **`init_seed`**
  (initialization, shuffling, client seeds), via `guardrails.resolve_seeds`.
- A legacy single `seed:` sets both, so every existing config reproduces bitwise.
  Mixing the legacy key with a new one is rejected.
- `--set init_seed=4602` on a legacy config keeps `split_seed` at 4601.
- The federated audit asserts that the config's split seed equals the one baked into
  each artifact.
- S5 init seeds: {4601, 4602, 4603}.

### 4. Local-only mode (R5)

`run_local_only_experiment`, or `--mode local_only` on the runner.

- **Same per-client budget by construction:** it reuses the Flower client classes
  unchanged, with the same rounds × E, per-round LR, client seeds, persistent optimizer
  and channel mask. Each round the client gets its own parameters back instead of the
  aggregate. The measured step count is asserted against the budget formula.
- Every client is evaluated on the shared global 9-channel test set, with the full
  regression report per client. A `local_only_summary.json` reports mean ± std over
  clients.
- A client that fails an enforced gate is recorded and the other clients still run;
  the run as a whole still fails loudly.
- **Disclosed assumption:** local-only regression uses the same global train
  source/subject lookup as FL and centralized, so all three are scored against one
  baseline. This gives a local-only client the means of subjects it never trained on,
  which favours the local-only lower bound.

### 5. Channel availability by masking (§1.3)

`src/redo_by_sara/channel_availability.py`.

- **8 positions → 9 channels**, with channels 8 and 10 kept together. The 1-indexed →
  raw `[0..7, 9]` and → artifact `[0..8]` conversions are asserted at import.
- **The mask is applied in both datasets' `__getitem__` after normalization**, so a
  masked channel equals the train mean.
- **The model stays plain 9-channel `SimpleCNN1D`.** Every client asserts that the
  incoming parameters match its model's shapes.
- Normalization stays global (train windows, all 9 channels). Evaluation is the global
  9-channel test set only.

Layouts at K=3, with ρ = |positions held by every client| / 8:

| ρ | client_0 | client_1 | client_2 | channels per client |
|---|---|---|---|---|
| 1.0 (= IID K=3) | 1–8 | 1–8 | 1–8 | 9 / 9 / 9 |
| **0.375** | {1,2,3,4,5} | {1,2,3,6,7} | {1,2,3,8} | 5 / 5 / 5 |
| 0.0 | {1,2,3} | {4,5,6} | {7,8} | 3 / 3 / 3 |

Position 8 (the only horizontal channel) is held by `client_2` in both partial layouts,
and the audit logs it. **ρ = 0.375 replaces "0.33" everywhere:** configs, the docs (7
occurrences in `CHANGES.md`, `docs/claude_code_sessions_v2.md` and
`docs/review_response_plan.md`), and filenames. "3 of 8" cannot be a third. ρ = 0 is
not special-cased.

### 6. Dirichlet label skew (K=3, α = 0.1)

- `partition_scheme: dirichlet` with `dirichlet_alpha`. Per subject, draw
  p ~ Dir(α·1_K), then deal that subject's runs (seeded order, largest-remainder
  rounding). Run-atomic, and the same assignment for both tasks.
- **Floor:** a draw leaving any client with fewer than 2 subjects is redrawn; the number
  of draws and `floor_applied` are recorded.
- **On the real data at split seed 4601 the first draw is accepted (no floor needed):**

  | | client_0 | client_1 | client_2 |
  |---|---|---|---|
  | runs | 21 | 15 | 76 |
  | subjects | 3 | 2 | 5 |
  | classification windows | 222 | 161 | 859 |
  | regression windows | 179 | 131 | 656 |

- IID balance and coverage are explicitly not required for this scheme.

### Verification

- **Bitwise gate.** Before any S4 edit, I generated a 2-round reference with the winner
  setting (E=5, persistent), both tasks, natural K=2 and IID K=3. After S4, the same
  commands give **byte-identical** files across both tasks: global and client
  histories, test predictions, confusion matrix, run and window predictions, train
  means, and partition files (11 per condition, 22 in total).
- **Centralized bitwise gate.** The three 60-epoch centralized baselines were re-run on
  the S4 code into scratch and compared against the committed artifacts:
  - classification: `training_history.csv`, `test_predictions.csv` and
    `test_confusion_matrix.csv` are identical;
  - absolute regression: `training_history.csv`, `test_run_predictions.csv` and
    `test_window_predictions.csv` are identical;
  - residual: `training_history.csv`, `test_run_predictions.csv`,
    `test_window_predictions.csv` and `train_source_subject_means.csv` are identical.

  The residual run's new report gives skill +0.121 [−0.288, +0.235], the S3 value. Its
  gates read degeneracy (enforced, passed), trained (enforced, passed), skill (not
  enforced, passed).
- **Bug found by that gate and fixed:** the residual pipeline wrote
  `test_per_subject_breakdown.csv` before `output_dir` was defined and crashed after
  training. No test exercises that pipeline end to end, so only the full re-run caught
  it. The write now follows the directory setup.
- 2-round smoke runs, exit 0: local-only on natural K=2, both tasks; ρ = 0 IID K=3;
  Dirichlet α=0.1 K=3.
- **End-to-end runner tests** (`test_end_to_end_runners.py`, `requires_data`): each
  runner runs to completion in 1–2 epochs or rounds into a temp folder.
  - Covered: centralized classification, absolute regression and residual regression;
    the federated CLI; and `--mode local_only`.
  - Each writes `training_summary.json`. Every regression path also writes
    `test_per_subject_breakdown.csv` (8 source × subject rows) and a 28-run
    `regression_report`.
  - The classification and absolute configs pin 60 epochs in their loader, so those two
    tests shorten the loaded config in memory and call the functions the runner scripts
    call.
- `pytest` (full suite, data present): **140 passed, 1 skipped** (cudnn, macOS), in
  6 min 18 s. New test files: `test_regression_metrics.py`,
  `test_channel_availability.py`, `test_s4_partitions_and_modes.py`,
  `test_end_to_end_runners.py`. Two S2 tests that used `"dirichlet"` as their
  example of an unknown scheme now use `"by_window"`.

**No reported number moved.** Default behaviour is bitwise unchanged; summaries only
gain keys.

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
| K=3 | channel ρ = 1.0 (= IID) · ρ = 0.375 · ρ = 0 · Dirichlet α = 0.1 · local-only |

**Every regression output reports**, as in `scripts/analyze_s3_regression_skill.py`:
overall skill and per-source skill (Test_2, 20251124), each with a run-level bootstrap
95% CI (10,000 resamples of test runs, fixed seed), plus overall skill without subject
007's test runs. S4 builds this into the consolidated regression metrics function so
it is emitted automatically.

**Fallback, decided before S5 runs:** if S5 is not complete by the morning of 31 Oct
2026, drop **regression** from the Dirichlet α=0.1 run first, then from ρ=0.375.
Classification for those runs still goes ahead.

**Amended 2026-10-19, before S4: the without-007 number is dropped from the standard
regression outputs.** It is replaced by a **per-subject breakdown** for every regression
run: for each subject × source, the number of test runs, model RMSE, lookup RMSE, and
their difference. Overall and per-source skill with run-level bootstrap CIs are kept.
**Why:** 007 is the only subject whose test speeds differ from its training speeds (see
the table above). A metric that removes one named subject singles it out after the
fact. The per-subject view shows where skill comes from, for every subject at once,
without choosing which subject to exclude. No subject ID is hard-coded in the metrics
code. The S3 without-007 numbers above remain as recorded.

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
