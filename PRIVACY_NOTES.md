# Privacy notes

Review plan §2 R7. This file states what the federated setup protects, what it does not,
and where the simulation takes shortcuts that a deployment would not. It makes no formal
privacy claim. Line references are to `src/redo_by_sara/combined_iid_flower.py` unless
another file is named.

## Silos

- **The natural K=2 split** has one client per collection campaign: `Test_2` (2024-04-06)
  and `20251124_Testing` (2025-11-24). Both were recorded in the same instrumented
  corridor on floor 4 of the Science and Engineering Innovation Center, 19.6 months
  apart. These are two sessions in one building, **not two buildings**.
- **The IID, ρ and Dirichlet partitions** are simulated silos: run-level partitions of
  the same pooled recordings, not separate organizations.
- **Unit of data:** a silo holds whole runs, never part of one (invariant 1 in
  `CLAUDE.md`).

## Threat model

- **The server is honest but curious.** It runs the protocol as specified and may
  inspect everything it receives.
- **Clients are honest but curious.** They may inspect the global model they receive
  each round.
- **There are no malicious participants:** no poisoned updates and no protocol
  deviation.
- **There is no secure aggregation and no differential privacy.** The server sees each
  client's update in the clear.

## What leaves a silo

**Raw vibration never leaves its silo.** Neither do window samples, per-run speed labels
or the client optimizer state. The persistent Adam state is kept per client and never
sent; `_save_client_optimizer` (line 400) writes it to the client's own folder.

What a client does send:

| When | What | Where in the code |
|---|---|---|
| Every round | Full model parameters (`SimpleCNN1D`, 9 input channels, identical shapes on every client) | `fit` returns `arrays` (lines 699, 781) |
| Every round | Its example count, used as the FedAvg weight. This reveals how many runs (regression) or windows (classification) the client holds. | the same returns |
| Every round | Diagnostic metrics on its own training data: online loss, RMSE and R², the loss at the start and end of local training, step counts and the learning rate | the metrics dict in the same returns |
| Once, at setup (regression) | Per-(source, subject) sufficient statistics of its training-run speeds: n, Σ and Σ² | `_federated_residual_baseline` (lines 500–565) |
| Once, at setup | Per-channel sufficient statistics of its training windows (count, Σ, Σ²) for input normalization | `_normalization_sufficient_statistics_audit` (lines 571–600) |

Two of these carry information about people, not just about models:
- **The setup-time speed statistics** disclose, for each enrolled subject on a client,
  how many runs the client holds and that subject's mean speed and its spread.
- **The classification label space is global.** Every client's output layer holds a
  class for every enrolled subject, so each client learns the full list of enrolled
  subjects, including people it never recorded.

## FedAvg is data minimization, not a privacy guarantee

FedAvg keeps raw data local and shares only model updates and aggregates. That limits
what is collected, but it is not a formal guarantee:
- Model updates can leak training data. Gradient inversion and membership inference
  against such updates are documented attacks.
- **Gradient inversion, membership inference and attribute inference are out of scope**
  for this work. We neither measure nor defend against them.
- **There is no differential-privacy accounting.** No noise is added to updates, so no
  (ε, δ) can be stated.
- **The released models identify people.** The subject classifier is, by construction,
  an identifier for the enrolled people. Sharing the final global model shares that
  identifier.

## Where the simulation takes shortcuts

The simulation runs in one process with the pooled training data available. It uses
that pooled data in four places a deployment could not:

1. **The per-subject lookup (residual baseline).** The FL runner computes the lookup
   without pooling:
   - each client contributes its (n, Σ, Σ²) per (source, subject);
   - the server sums them into the means and the residual scale (lines 500–530).

   The pooled computation (`compute_train_source_subject_means`,
   `combined_residual_run_regression.py:155`) is used only as a cross-check. The run
   aborts unless the federated means agree with it to 1e-12 and the residual scale
   agrees to 1e-12 (lines 534–540). The pooling is a convenience for verification, not
   a dependency.
2. **Input normalization.** Every run uses the per-channel mean and std saved in the
   artifact, which were computed from the pooled training windows. At setup, the runner
   checks that summing the clients' per-channel (count, Σ, Σ²) reproduces those
   statistics to within 2e-6 (line 590). A deployment would exchange those sufficient
   statistics instead.
3. **Global-train evaluation.** The server evaluates the global model on the pooled
   training set each round (`evaluate_global_train`, line 1185). Flower's federated
   evaluation is switched off (`fraction_evaluate=0.0`, line 1381). This metric drives
   the convergence figure and the `trained` gate.
   - In a deployment it would be computed by each client and summed as sufficient
     metrics, which `aggregate_regression_sufficient_metrics` (line 604) already
     implements.
   - As run, it requires pooled data and is a **simulation-only diagnostic**.
4. **Held-out test evaluation.** The 28 test runs are evaluated once, centrally, after
   the last round. `guardrails.TestSetAccessGuard` counts every access to them. In a
   deployment those runs would stay in their silos.

**The local-only baseline.** Local-only clients also use the all-client lookup (line
2253), so they are scored against the same baseline as FL and centralized. This was
disclosed in S4. It gives a local-only client the speed means of subjects it never
recorded: information a truly isolated silo would not have.

## Deployment caveats beyond privacy

**The residual regression model needs to know who is walking.** It predicts speed as
the subject's lookup mean plus a residual, so at inference it must know the walker's
identity (and campaign). This is review item R3.

## Future work

- A hook for DP-SGD noise on client updates, with privacy accounting.
- Secure aggregation, so the server sees only the sum of the updates.
- An empirical gradient-inversion or membership-inference evaluation against the
  released models.

None of these is implemented. The paper's claim is limited to data minimization: raw
vibration and labels stay in their silo.
