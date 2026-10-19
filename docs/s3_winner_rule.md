# S3 winner rule — pre-registered before the sweep

This file fixes how the S3 sweep (review plan §2 R1) picks the federated local-training
setting that S5 uses. It was committed **before any sweep code was written and before any
sweep result existed**. `scripts/run_r1_sweep.py` refuses to start unless this file is
tracked and unmodified, and it records this file's commit hash in the sweep summary.
Changing the rule after results exist requires a new commit, and `CHANGES.md` must
report it as a post-hoc change.

## 1. What is swept

All sweep runs use the **regression task**, the **IID K=3 partition** (the original R1
setting, with the smallest clients at ~10 optimizer steps per round), and the seed-4601
split and initialization. Held fixed for every sweep run:

| Setting | Value |
|---|---|
| Rounds | 60 |
| Regression batch size | 4 runs |
| LR schedule | `cosine_over_rounds` (1e-3 → 1e-5), the pre-S3 schedule |
| Client optimizer | Adam, weight decay 1e-4 |
| `client_num_cpus` | 3 |
| Machine | M5, CPU only, conda env `fl` (see `CLAUDE.md`) |

**Stage 1 — FedAvg, six configs**, isolating the local axis:

| | `fresh` | `persistent` |
|---|---|---|
| E = 1 | S1-a | S1-b |
| E = 5 | S1-c | S1-d |
| E = 10 | S1-e | S1-f |

`fresh` builds a new Adam on every client every round (the pre-S3 behaviour).
`persistent` carries each client's Adam state across rounds; each round the learning
rate is reset from the schedule and the model parameters are replaced by the global
model.

**Stage 2 — FedAdam, one config**, at the local setting (E, optimizer state) that the
rule below selects from stage 1. Server hyperparameters are fixed here and **not tuned**:

| η (server LR) | β₁ | β₂ | τ |
|---|---|---|---|
| 1e-2 | 0.9 | 0.99 | 1e-3 |

Every config is run and reported, **including the ones that fail**. The failures are a
result.

## 2. Eligibility

A config is **eligible** only if it passes all three gates of the degenerate-model
detector (`guardrails.py`), with `--allow-degenerate` off:

1. final `global_train_standardized_mse` < 0.95;
2. more than 10 distinct predicted residuals on the test set;
3. skill score vs the source/subject-mean baseline > 0.

**and** it satisfies the compute cap: **E ≤ 5**. A config with E = 10 becomes eligible
only if **no** E ≤ 5 config passes the three gates.

**Reason for the cap:** the S5 compute budget on the M5. A both-task federated run at
E=10 is estimated at ~3 h on this machine, and S5 has ~10 federated runs, so about three
nights. That does not fit a 31 Oct deadline alongside S6 and manuscript assembly. The
cap was fixed before the sweep, not after seeing results. E=10 configs are still run and
reported.

## 3. Ranking

Among eligible configs, **stage 1 and stage 2 are ranked together under one rule**:

1. **Primary:** lowest final `global_train_standardized_mse`, computed on the training
   data after round 60.
2. **Tie:** every config within **0.01** of the best value counts as tied. Among tied
   configs, the one with **fewer total gradient steps** wins. Total steps are
   Σ_k ⌈runs_k / batch⌉ × E × rounds, summed over clients.
3. **Remaining tie** (same steps): the simpler config wins. FedAvg beats FedAdam, and
   `fresh` beats `persistent`.

**If FedAdam wins, S5 uses FedAdam.** That choice is made here, before any FedAdam
number exists.

**The test set is never used for ranking.** It enters only through detector gates 2 and
3 as pass/fail. No test RMSE, R² or skill value breaks a tie or orders configs.

If no config is eligible, there is **no winner**: stop and escalate (runbook, "Stop and
escalate if").

## 4. Confirmation on the natural K=2 split

The winner is confirmed on the **natural K=2** split (the paper's headline). This step is
**pass/fail only, both tasks, `--allow-degenerate` off**.

- **Regression passes** if the run passes all three detector gates.
- **Classification passes** unless its final global-train accuracy is **clearly below**
  that of a natural K=2 classification run at **E=1 / `fresh` / FedAvg**. That reference
  is run as part of this step, with the same seed and split. "Clearly below" means
  **more than 2 percentage points lower**. If it is clearly below: stop and escalate.

**Fallback:** if the winner fails on natural K=2, try the **next-ranked eligible config
from the IID K=3 ranking in §3**, and keep going down that list. If no eligible config
passes, stop and escalate.

**Natural-split results never re-rank configs.** They only accept or reject, in the
order §3 fixed.

## 5. What the winner sets in S5

The winning setting applies to **both tasks** in every S5 federated run: E, optimizer
state, and server strategy (FedAvg, or FedAdam with the §1 hyperparameters). Local-only
baselines use the same E and optimizer state, so local-only gets the same per-client
training budget as federated.

## 6. Determinism precondition

Persistent mode adds a new place for non-reproducibility to come in: optimizer-state
files written to and read from disk between rounds. Before the sweep runs, a test must
show that the **same config run twice** (E=2, `client_optimizer_state: persistent`, a
few rounds, small enough to be quick) gives **bitwise-identical final model
parameters**. If it does not, the sweep does not start.

---

## Escalation resolution (2026-10-19)

*Appended after the sweep ran. Sections 1–6 above are the rule as committed in
`407f03f`, before any result existed, and are unchanged.*

The sweep ended with `STOP AND ESCALATE: No eligible config passed the natural K=2
confirmation (rule §4)`.

- Stage 1 selected **FedAvg, E=5, persistent** (rank 1). FedAdam at that setting
  passed the detector and ranked 2. E=10 persistent passed but was not eligible under
  the §2 cap.
- On natural K=2, **both ranked configs failed the confirmation on the skill gate, not
  on collapse.** Both trained (final global-train standardized MSE 0.119 and 0.606) and
  produced 238 distinct test residuals. Their test skill against the subject-mean
  baseline was **−0.207** (FedAvg) and **−0.002** (FedAdam). Classification passed for
  both (final global-train accuracy 0.998 and 1.000, against the E=1/`fresh` reference
  of 0.971).

**Resolution, by the PI:** proceed to S5 with the pre-registered IID winner (**FedAvg,
E=5, persistent client optimizer state**) for both tasks and for the local-only
baselines. The natural-split regression result is reported **as measured**, as a
heterogeneity outcome, not as a failed run. This is a decision made after seeing the
confirmation results, and it is disclosed as such here and in `CHANGES.md`.
