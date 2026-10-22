"""S5: run-level classification metrics, source probe, subject-003 control, and the queue.

Exact counts are independent literals (CLAUDE.md), not imported from the modules.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from redo_by_sara.run_level_classification import aggregate_runs, summarize_runs, wilson_interval
from redo_by_sara.s5_queue import (
    STATIC_MINUTES,
    Job,
    build_jobs,
    command,
    detect_outcome,
    estimate_minutes,
    fallback_drops,
    fits_before,
    parse_stop_by,
    select_jobs,
)

ROOT = Path(__file__).resolve().parent
CLS_DIR = ROOT / "artifacts/centralized_combined_9ch_no006_nowalking_80_20_60e"


# ------------------------------------------------------------------ Wilson and aggregation


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (28, 28, 0.879357, 1.0),  # n / (n + z^2)
        (0, 15, 0.0, 0.203883),  # z^2 / (n + z^2)
        (10, 20, 0.299298, 0.700702),
        (13, 13, 0.771905, 1.0),
    ],
)
def test_wilson_interval_known_values(k: int, n: int, low: float, high: float) -> None:
    assert wilson_interval(k, n) == pytest.approx((low, high), abs=1e-6)


def test_wilson_interval_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(3, 2)


CLASSES = ["no_walking", "subject_a_walking", "subject_b_walking"]


def _rows(run_uid: str, true: list[str], source: str = "s1", subject: str = "a") -> list[dict[str, str]]:
    return [{"run_uid": run_uid, "source_id": source, "subject_id": subject, "true_class": t,
             "artifact_index": "0"} for t in true]


def test_majority_tie_goes_to_higher_mean_probability() -> None:
    rows = _rows("r1", ["subject_a_walking"] * 2)
    # one vote each for a and b; b has the higher mean probability
    probabilities = np.array([[0.0, 0.55, 0.45], [0.0, 0.05, 0.95]])
    [record] = aggregate_runs(rows, probabilities, CLASSES)
    assert record["majority_vote_tied"] is True
    assert record["majority_vote_label"] == "subject_b_walking"
    assert record["mean_probability_label"] == "subject_b_walking"
    assert record["majority_vote_correct"] is False


def test_background_windows_are_excluded_but_background_votes_count() -> None:
    rows = _rows("r1", ["no_walking", "subject_a_walking", "subject_a_walking", "subject_a_walking"])
    probabilities = np.array([
        [0.9, 0.05, 0.05],  # background window: excluded from the run
        [0.8, 0.1, 0.1],  # walking windows predicted no_walking twice
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
    ])
    [record] = aggregate_runs(rows, probabilities, CLASSES)
    assert record["num_windows"] == 3
    assert record["majority_vote_label"] == "no_walking"
    assert record["majority_vote_correct"] is False


def test_unknown_true_label_is_a_negative_control_not_an_error() -> None:
    rows = _rows("r9", ["subject_z_walking"] * 2, subject="z")
    probabilities = np.array([[0.0, 0.9, 0.1], [0.0, 0.8, 0.2]])
    [record] = aggregate_runs(rows, probabilities, CLASSES, exclude_true=())
    assert record["majority_vote_label"] == "subject_a_walking"
    assert record["majority_vote_correct"] is None
    assert record["mean_probability_of_true"] is None


def test_summarize_runs_accuracy_macro_f1_and_groups() -> None:
    rows = (_rows("r1", ["subject_a_walking"]) + _rows("r2", ["subject_a_walking"])
            + _rows("r3", ["subject_b_walking"], source="s2", subject="b"))
    probabilities = np.array([[0, 1.0, 0], [0, 0, 1.0], [0, 0, 1.0]])
    summary = summarize_runs(aggregate_runs(rows, probabilities, CLASSES))
    vote = summary["majority_vote"]
    assert (vote["num_correct"], vote["num_runs"]) == (2, 3)
    # a: tp 1, fn 1 -> F1 2/3; b: tp 1, fp 1 -> F1 2/3
    assert vote["macro_f1"] == pytest.approx(2 / 3)
    assert vote["per_source"]["s2"]["accuracy"] == 1.0
    assert [(r["source_id"], r["subject_id"], r["num_runs"]) for r in summary["per_subject_source"]] == [
        ("s1", "a", 2), ("s2", "b", 1)
    ]


# ------------------------------------------------------------------ queue: the matrix


def test_job_matrix_counts_and_order() -> None:
    jobs = build_jobs()
    assert len(jobs) == 47
    assert sum(not j.reuse_canonical for j in jobs) == 44
    assert len({j.job_id for j in jobs}) == 47
    assert [j.group for j in jobs] == sorted(j.group for j in jobs)
    by_group = {g: sum(j.group == g and not j.reuse_canonical for j in jobs) for g in range(1, 8)}
    assert by_group == {1: 12, 2: 4, 3: 8, 4: 6, 5: 2, 6: 8, 7: 4}
    assert jobs[0].job_id == "fl/k2_natural/seed4601/classification"
    assert jobs[1].job_id == "fl/k2_natural/seed4601/regression"
    fl = [j for j in jobs if j.family == "fl"]
    assert {(j.condition, j.seed) for j in fl if j.condition.startswith("k2")} == {
        (c, s) for c in ("k2_natural", "k2_iid") for s in (4601, 4602, 4603)
    }
    assert all(j.seed == 4601 for j in jobs if j.group < 7 and (j.family == "local_only" or j.condition.startswith("k3")))
    # group 7 (added 2026-10-22): Dirichlet classification at the two extra init seeds
    assert [(j.family, j.seed, j.task) for j in jobs if j.group == 7] == [
        ("fl", 4602, "classification"), ("fl", 4603, "classification"),
        ("local_only", 4602, "classification"), ("local_only", 4603, "classification"),
    ]
    assert all(j.condition == "k3_dirichlet0.1" and j.fallback_level == 0 for j in jobs if j.group == 7)
    assert [j.job_id for j in jobs[:43]] == [j.job_id for j in jobs if j.group < 7]
    assert {j.condition for j in jobs if j.family == "local_only"} == {
        "k2_natural", "k2_iid", "k3_iid", "k3_rho0.375", "k3_rho0", "k3_dirichlet0.1"
    }
    # rho = 1.0 is the IID K=3 run, not a separate job
    assert not any("rho1" in j.job_id for j in jobs)


def test_centralized_absolute_runs_before_same_seed_residual() -> None:
    ids = [j.job_id for j in build_jobs()]
    for seed in (4602, 4603):
        assert ids.index(f"centralized/absolute/seed{seed}") < ids.index(f"centralized/residual/seed{seed}")


def test_fallback_levels() -> None:
    levels = {j.job_id: j.fallback_level for j in build_jobs() if j.fallback_level}
    assert levels == {
        "fl/k3_rho0.375/seed4601/regression": 2,
        "fl/k3_dirichlet0.1/seed4601/regression": 1,
        "local_only/k3_rho0.375/seed4601/regression": 2,
        "local_only/k3_dirichlet0.1/seed4601/regression": 1,
    }


def test_commands_write_only_under_the_root(tmp_path: Path) -> None:
    for job in build_jobs():
        cmd = command(job, tmp_path, "python", smoke=False)
        if job.reuse_canonical:
            assert cmd is None
            continue
        out = Path(cmd[cmd.index("--output-root") + 1])
        assert out == tmp_path / job.job_id
        if job.family in ("fl", "local_only"):
            assert cmd[cmd.index("--set") + 1] == f"init_seed={job.seed}"
            assert ("--mode" in cmd) == (job.family == "local_only")
            assert "--rounds" not in cmd and "--allow-degenerate" not in cmd
        if "--config" in cmd:
            config = Path(cmd[cmd.index("--config") + 1])
            assert config.exists() and config.parent == ROOT / "configs" and config.name.startswith("s5_")


def test_smoke_commands_shorten_and_skip_centralized(tmp_path: Path) -> None:
    for job in build_jobs():
        cmd = command(job, tmp_path, "python", smoke=True)
        if job.family == "centralized":
            assert cmd is None
        elif job.family in ("fl", "local_only"):
            assert cmd[cmd.index("--rounds") + 1] == "1"
        else:
            assert cmd[cmd.index("--epochs") + 1] == "1"


def test_select_jobs_by_glob() -> None:
    selected = select_jobs(build_jobs(), ["fl/k2_natural/seed4601/*", "source_probe/*"])
    assert [j.job_id for j in selected] == [
        "fl/k2_natural/seed4601/classification", "fl/k2_natural/seed4601/regression", "source_probe/seed4601"
    ]


def test_s5_configs_carry_the_winner_setting() -> None:
    from redo_by_sara.combined_iid_flower import load_combined_iid_flower_config

    expected = {
        "k2_natural": (2, "natural", None, None), "k2_iid": (2, "iid", None, None),
        "k3_iid": (3, "iid", None, None), "k3_rho0.375": (3, "iid", 0.375, None),
        "k3_rho0": (3, "iid", 0.0, None), "k3_dirichlet0.1": (3, "dirichlet", None, 0.1),
    }
    for name, (k, scheme, rho, alpha) in expected.items():
        config = load_combined_iid_flower_config(ROOT / f"configs/s5_{name}.yaml", {"init_seed": 4603})
        assert (config["num_clients"], config["partition_scheme"], config["channel_overlap"], config["dirichlet_alpha"]) == (k, scheme, rho, alpha)
        assert (config["aggregation"], config["local_epochs"], config["client_optimizer_state"]) == ("FedAvg", 5, "persistent")
        assert (config["num_rounds"], config["client_num_cpus"], config["lr_schedule"]) == (60, 3.0, "cosine_over_rounds")
        assert (config["split_seed"], config["init_seed"]) == (4601, 4603)
        # Loaders resolve relative paths from the config's grandparent, so a config in a
        # subfolder of configs/ would silently point at configs/artifacts/...
        assert config["classification_artifact"] == CLS_DIR / "raw_windows_combined.pt"
        assert config["centralized_classification_dir"] == CLS_DIR


def test_s5_centralized_configs_keep_the_split_and_vary_init() -> None:
    from redo_by_sara.combined_centralized_classification import load_combined_config
    from redo_by_sara.combined_centralized_regression import load_regression_config
    from redo_by_sara.combined_residual_run_regression import load_residual_run_config

    for seed in (4602, 4603):
        for loader, name in ((load_combined_config, "classification"), (load_regression_config, "absolute"),
                             (load_residual_run_config, "residual")):
            config = loader(ROOT / f"configs/s5_centralized_{name}_init{seed}.yaml")
            assert (config["split_seed"], config["init_seed"], config["epochs"]) == (4601, seed, 60)
            if name != "residual":
                assert config["output_dir"].parent == ROOT / "artifacts"  # locates the input artifact
        residual = load_residual_run_config(ROOT / f"configs/s5_centralized_residual_init{seed}.yaml")
        assert residual["baseline_summary_path"] == ROOT / f"artifacts/s5/centralized/absolute/seed{seed}/training_summary.json"


# ------------------------------------------------------------------ queue: scheduling


def test_stop_by_is_the_next_occurrence() -> None:
    evening = datetime(2026, 10, 19, 21, 0)
    assert parse_stop_by("05:00", evening) == datetime(2026, 10, 20, 5, 0)
    assert parse_stop_by("23:30", evening) == datetime(2026, 10, 19, 23, 30)
    assert parse_stop_by("21:00", evening) == datetime(2026, 10, 20, 21, 0)
    with pytest.raises(ValueError):
        parse_stop_by("5am", evening)


def test_fits_before_uses_the_margin() -> None:
    now, stop = datetime(2026, 10, 20, 4, 0), datetime(2026, 10, 20, 5, 0)
    assert fits_before(now, 54, stop)  # 54 x 1.1 = 59.4 min
    assert not fits_before(now, 55, stop)  # 60.5 min
    assert fits_before(now, 10_000, None)


def test_fallback_drops_only_marked_jobs_after_the_cutoff() -> None:
    jobs = {j.job_id: j for j in build_jobs()}
    dirichlet = jobs["fl/k3_dirichlet0.1/seed4601/regression"]
    dirichlet_cls = jobs["fl/k3_dirichlet0.1/seed4601/classification"]
    before, after = datetime(2026, 10, 31, 7, 59), datetime(2026, 10, 31, 8, 0)
    assert not fallback_drops(dirichlet, before)
    assert fallback_drops(dirichlet, after)
    assert not fallback_drops(dirichlet_cls, after)


def _record(directory: Path, **fields: object) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "job.json").write_text(json.dumps(fields))


def test_estimate_uses_measured_time_of_the_same_class(tmp_path: Path) -> None:
    jobs = build_jobs()
    first, later = jobs[0], jobs[4]  # natural K=2 classification, seeds 4601 and 4602
    assert first.timing_key == later.timing_key
    assert estimate_minutes(later, tmp_path, jobs, smoke=False) == STATIC_MINUTES[later.timing_key]
    _record(tmp_path / first.job_id, outcome="completed", wall_seconds=3000.0, smoke=False)
    assert estimate_minutes(later, tmp_path, jobs, smoke=False) == pytest.approx(50.0)
    assert estimate_minutes(later, tmp_path, jobs, smoke=True) == STATIC_MINUTES[later.timing_key]


def test_detect_outcome(tmp_path: Path) -> None:
    fl_reg = Job("fl/x/seed4601/regression", 1, "fl", "k2_natural", 4601, "regression")
    assert detect_outcome(fl_reg, tmp_path, 1, "") == "crashed"
    result = fl_reg.result_dir(tmp_path)
    result.mkdir(parents=True)
    (result / "degenerate_model_check_FAILED.json").write_text("{}")
    assert detect_outcome(fl_reg, tmp_path, 1, "") == "failed_detector"
    (result / "training_summary.json").write_text("{}")
    assert detect_outcome(fl_reg, tmp_path, 0, "") == "completed"

    local = Job("local_only/x/seed4601/regression", 2, "local_only", "k2_natural", 4601, "regression")
    summary = local.result_dir(tmp_path) / "local_only_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"failed_clients": ["client_1"]}))
    assert detect_outcome(local, tmp_path, 1, "") == "completed_with_failed_clients"
    summary.write_text(json.dumps({"failed_clients": []}))
    assert detect_outcome(local, tmp_path, 0, "") == "completed"

    residual = Job("centralized/residual/seed4602", 3, "centralized", "residual", 4602, "regression")
    assert detect_outcome(residual, tmp_path, 1, "DegenerateModelError: ...") == "failed_detector"
    assert detect_outcome(residual, tmp_path, 1, "Traceback: KeyError") == "crashed"


# ------------------------------------------------------------------ data-backed


@pytest.fixture(scope="module")
def classification_artifact() -> dict:
    return torch.load(CLS_DIR / "raw_windows_combined.pt", map_location="cpu", weights_only=False)


@pytest.mark.requires_data
def test_control_statistics_reproduce_the_canonical_recipe(classification_artifact: dict) -> None:
    from redo_by_sara.subject003_control import train_channel_statistics

    mean, std = train_channel_statistics(
        classification_artifact["samples"].numpy(), classification_artifact["train_indices"].tolist()
    )
    assert torch.equal(torch.from_numpy(mean), classification_artifact["channel_mean"])
    assert torch.equal(torch.from_numpy(std), classification_artifact["channel_std"])


@pytest.mark.requires_data
def test_cross_session_control_split(classification_artifact: dict) -> None:
    from redo_by_sara.guardrails import assert_split_integrity
    from redo_by_sara.subject003_control import build_control_artifact, cross_session_control_split

    split = cross_session_control_split(classification_artifact, "test_2")
    metadata = classification_artifact["metadata"]

    def runs(key: str) -> int:
        return len({metadata[i]["run_uid"] for i in split[key]})

    assert split["classes"] == ["subject_001_walking", "subject_002_walking", "subject_003_walking"]
    assert split["cross_session_subjects"] == ["003"]
    assert [(runs(k), len(split[k])) for k in ("train", "in_session", "cross_session", "negative_control")] == [
        (52, 431), (13, 103), (15, 142), (60, 528)
    ]
    assert {metadata[i]["source_id"] for i in split["train"] + split["in_session"]} == {"test_2"}
    assert {metadata[i]["source_id"] for i in split["cross_session"] + split["negative_control"]} == {"testing_20251124"}
    # the cross-session set uses all of 003's 20251124 runs, canonical train and test
    assert {metadata[i]["split"] for i in split["cross_session"]} == {"train", "test"}

    derived = build_control_artifact(classification_artifact, split)
    audit = assert_split_integrity(derived, "control")
    assert audit["train_test_run_overlap"] == 0
    assert len(derived["train_indices"]) == 431 and len(derived["test_indices"]) == 245
    assert not torch.equal(derived["channel_mean"], classification_artifact["channel_mean"])
    train_rows = [derived["metadata"][i] for i in derived["train_indices"].tolist()]
    assert {(r["source_id"], r["canonical_split"]) for r in train_rows} == {("test_2", "train")}


@pytest.mark.requires_data
def test_source_relabeled_artifact(classification_artifact: dict) -> None:
    from redo_by_sara.source_probe import source_relabeled_artifact

    probe = source_relabeled_artifact(classification_artifact)
    assert probe["index_to_class"] == ["test_2", "testing_20251124"]
    targets = probe["classification_targets"].tolist()
    assert all(probe["index_to_class"][t] == m["source_id"] for t, m in zip(targets, classification_artifact["metadata"]))
    assert torch.equal(probe["train_indices"], classification_artifact["train_indices"])
    assert torch.equal(probe["channel_mean"], classification_artifact["channel_mean"])
    assert classification_artifact["index_to_class"][0] == "no_walking"  # the original is untouched


@pytest.mark.requires_data
def test_run_level_on_the_canonical_centralized_checkpoint(classification_artifact: dict, tmp_path: Path) -> None:
    from redo_by_sara.run_level_classification import run_level_for_result

    summary = run_level_for_result(CLS_DIR, "combined_centralized_cnn_final.pt", classification_artifact["samples"], tmp_path)
    assert summary["checkpoint_reproduces_saved_window_predictions"] is True
    assert summary["num_windows_scored"] == 307
    assert summary["majority_vote"]["num_runs"] == 28
    assert {s: b["num_runs"] for s, b in summary["majority_vote"]["per_source"].items()} == {
        "test_2": 13, "testing_20251124": 15
    }
    assert len(summary["per_subject_source"]) == 8
    assert (tmp_path / "run_level_predictions.csv").read_text().count("\n") == 1 + 28


# ------------------------------------------------------------------ degeneracy spread rule (2026-10-20)

SEED4603 = json.loads((ROOT / "test_fixtures/s5_k2_natural_seed4603_regression.json").read_text())


def _health_4603(**overrides: object) -> dict:
    from redo_by_sara.guardrails import check_regression_health

    arguments = dict(
        actual=SEED4603["run_actual_speed_mps"],
        predicted=SEED4603["run_predicted_speed_mps"],
        baseline=SEED4603["run_baseline_mps"],
        final_standardized_mse=SEED4603["final_standardized_mse"],
        enforced_gates=(),
        run_label="s5-natural-k2-seed4603",
        model_output=SEED4603["window_predicted_residual_mps"],
        model_output_name="window predicted residual",
    )
    arguments.update(overrides)
    return check_regression_health(**arguments)


def test_seed4603_passes_a_distinct_count_but_fails_the_spread_rule() -> None:
    report = _health_4603()
    # What fooled the old gate: 205 distinct values (> 10) and a std above zero.
    assert report["distinct_predictions"] == 205
    assert 0.0 < report["prediction_std"] < 1e-10
    assert report["relative_output_spread"] < 1e-9
    gate = report["gates"]["degeneracy"]
    assert gate["passed"] is False
    assert len(gate["failures"]) == 1 and "numerical noise" in gate["failures"][0]
    assert report["gates"]["trained"]["passed"] is False


def test_seed4603_raises_when_degeneracy_is_enforced_alone() -> None:
    from redo_by_sara.guardrails import DegenerateModelError

    with pytest.raises(DegenerateModelError, match="numerical noise"):
        _health_4603(enforced_gates=("degeneracy",))


def test_spread_rule_threshold_is_relative_to_the_target() -> None:
    from redo_by_sara.guardrails import MINIMUM_RELATIVE_OUTPUT_SPREAD

    assert MINIMUM_RELATIVE_OUTPUT_SPREAD == 1e-3
    rng = np.random.default_rng(11)
    baseline = rng.uniform(1.0, 1.5, size=28)
    actual = baseline + rng.normal(0.0, 0.04, size=28)
    target_std = float(np.std(actual - baseline))
    noise = rng.normal(0.0, 1.0, size=238)
    noise /= noise.std()

    def degeneracy(relative: float) -> bool:
        return _health_4603(actual=actual, predicted=actual, baseline=baseline,
                            model_output=noise * relative * target_std)["gates"]["degeneracy"]["passed"]

    assert degeneracy(2e-3) is True
    assert degeneracy(5e-4) is False
    # S3 E=1 persistent collapse (std 4.8e-7 m/s against a 0.039 target, ratio 1.2e-5)
    assert degeneracy(1.2e-5) is False
    # the weakest trained run on disk (centralized 60e residual, ratio 0.386)
    assert degeneracy(0.386) is True


def test_seed_reporting_rule() -> None:
    from redo_by_sara.s5_queue import seed_reporting

    natural = [{"skill": -0.207, "training_outcome": "trained"}, {"skill": -0.123, "training_outcome": "trained"},
               {"skill": -0.000001, "training_outcome": "collapsed"}]
    assert seed_reporting(natural) == "per seed; 1 of 3 seeds collapsed"
    assert seed_reporting(natural[:2]) == "mean ± std over 2 seeds; 2 of 2 seeds trained"
    mixed = natural[:2] + [{"skill": 0.04, "training_outcome": "did not train"}]
    assert seed_reporting(mixed) == "per seed; 1 of 3 seeds did not train"
    assert seed_reporting([{"window_accuracy": 1.0}, {"window_accuracy": 0.99}]) == "mean ± std over 2 seeds"
    # why: the mean of the three would sit above both trained seeds
    assert np.mean([m["skill"] for m in natural]) > max(m["skill"] for m in natural[:2])


def test_seed_aggregates_report_an_untrained_group_per_seed(tmp_path: Path) -> None:
    from redo_by_sara.s5_queue import rebuild_index

    jobs = build_jobs()
    skills = {("k2_natural", 4601): (-0.207, "trained"), ("k2_natural", 4602): (-0.123, "trained"),
              ("k2_natural", 4603): (-0.000001, "collapsed"), ("k2_iid", 4601): (0.089, "trained"),
              ("k2_iid", 4602): (0.150, "trained"), ("k2_iid", 4603): (0.277, "trained")}
    for job in jobs:
        key = (job.condition, job.seed)
        if job.family == "fl" and job.task == "regression" and key in skills:
            skill, label = skills[key]
            _record(tmp_path / job.job_id, outcome="completed" if label == "trained" else "failed_detector",
                    wall_seconds=60.0, smoke=False, flat=label != "trained",
                    metrics={"skill": skill, "model_rmse_mps": 0.04, "training_outcome": label})
    index = {row["job_id"]: row for row in rebuild_index(jobs, tmp_path)}
    assert index["fl/k2_natural/seed4603/regression"]["training_outcome"] == "collapsed"
    assert index["fl/k2_natural/seed4601/regression"]["seed_reporting"] == "per seed; 1 of 3 seeds collapsed"
    assert index["fl/k2_iid/seed4601/regression"]["seed_reporting"] == "mean ± std over 3 seeds; 3 of 3 seeds trained"

    import csv

    rows = {r["condition"]: r for r in csv.DictReader((tmp_path / "derived/seed_aggregates.csv").open())}
    natural, iid = rows["k2_natural"], rows["k2_iid"]
    assert natural["reporting"] == "per seed; 1 of 3 seeds collapsed"
    assert not natural.get("skill_mean") and float(natural["skill_seed4601"]) == -0.207
    assert natural["training_outcome_seed4603"] == "collapsed"
    assert float(iid["skill_mean"]) == pytest.approx((0.089 + 0.150 + 0.277) / 3)


# ------------------------------------------------------------------ outcome labels and local-only gates (2026-10-21)

JOB14 = json.loads((ROOT / "test_fixtures/s5_local_only_k2_natural_seed4601_regression.json").read_text())
RHO0375 = json.loads((ROOT / "test_fixtures/s5_k3_rho0.375_seed4601_regression.json").read_text())


def _health(model: dict, final_standardized_mse: float | None) -> dict:
    from redo_by_sara.guardrails import check_regression_health

    return check_regression_health(
        actual=model["run_actual_speed_mps"], predicted=model["run_predicted_speed_mps"],
        baseline=model["run_baseline_mps"], final_standardized_mse=final_standardized_mse,
        enforced_gates=(), run_label="fixture", model_output=model["window_predicted_residual_mps"],
        model_output_name="window predicted residual",
    )


def test_three_outcome_labels_on_real_runs() -> None:
    from redo_by_sara.guardrails import training_outcome

    assert training_outcome(_health_4603()) == "collapsed"
    rho = _health(RHO0375, RHO0375["final_standardized_mse"])
    assert RHO0375["final_standardized_mse"] == pytest.approx(0.99124, abs=1e-5)
    assert rho["gates"]["degeneracy"]["passed"] is True and rho["relative_output_spread"] > 0.05
    assert training_outcome(rho) == "did not train"


def test_own_data_mse_is_relative_to_the_clients_zero_residual_start() -> None:
    from redo_by_sara.guardrails import own_data_standardized_mse

    rows = [{"round": "2", "local_start_standardized_mse": 0.4, "local_end_standardized_mse": 0.3},
            {"round": "1", "local_start_standardized_mse": 0.5, "local_end_standardized_mse": 0.45}]
    assert own_data_standardized_mse(rows) == pytest.approx(0.3 / 0.5)  # last end / round-1 start
    with pytest.raises(ValueError):
        own_data_standardized_mse(rows[:1] and [rows[0]])  # no round 1


def test_job14_both_local_only_clients_trained_on_their_own_data() -> None:
    from redo_by_sara.guardrails import MAXIMUM_STANDARDIZED_MSE, own_data_standardized_mse, training_outcome

    own = {c: own_data_standardized_mse(m["local_client_fit_history"]) for c, m in JOB14["clients"].items()}
    assert own["client_0"] < 1e-5 and own["client_1"] < 1e-5
    # The old basis: client_1's global-train MSE fails, although it fit its own data.
    assert JOB14["clients"]["client_1"]["global_train_standardized_mse"] > 3.0
    assert JOB14["clients"]["client_0"]["global_train_standardized_mse"] < MAXIMUM_STANDARDIZED_MSE
    # A bare 0.95 on the raw own-data value would be wrong: client_0's untrained start
    # (0.566) is already below it, client_1's (1.376) above.
    starts = {c: m["local_client_fit_history"][0]["local_start_standardized_mse"] for c, m in JOB14["clients"].items()}
    assert starts["client_0"] < MAXIMUM_STANDARDIZED_MSE < starts["client_1"]
    for client, model in JOB14["clients"].items():
        old = _health(model, model["global_train_standardized_mse"])
        new = _health(model, own[client])
        assert training_outcome(new) == "trained", client
        if client == "client_1":
            assert training_outcome(old) == "did not train"


def test_local_only_aggregation_counts_failed_clients(tmp_path: Path) -> None:
    from redo_by_sara.s5_queue import collect_metrics

    job = next(j for j in build_jobs() if j.job_id == "local_only/k2_natural/seed4601/regression")
    result = job.result_dir(tmp_path)

    def write_client(name: str, model: dict, history: list[dict]) -> None:
        import csv

        d = result / name
        d.mkdir(parents=True)
        runs = [{"actual_speed_mps": a, "predicted_speed_mps": p, "source_subject_train_mean_mps": b,
                 "run_uid": f"r{i}", "source_id": "s1" if i % 2 else "s2", "subject_id": f"{i % 3}"}
                for i, (a, p, b) in enumerate(zip(model["run_actual_speed_mps"], model["run_predicted_speed_mps"], model["run_baseline_mps"]))]
        for filename, rows in (("test_run_predictions.csv", runs),
                               ("test_window_predictions.csv", [{"predicted_residual_mps": v} for v in model["window_predicted_residual_mps"]]),
                               ("local_client_fit_history.csv", history),
                               ("global_train_history.csv", [{"round": 60, "global_train_standardized_mse": model["global_train_standardized_mse"]}])):
            with (d / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        from redo_by_sara.regression_metrics import report_from_run_rows

        (d / "training_summary.json").write_text(json.dumps({
            "regression_report": report_from_run_rows(runs, "source_subject_train_mean_mps"),
            "degenerate_model_check": {"gates": {"degeneracy": {"passed": True}}},
        }))

    clients = JOB14["clients"]
    write_client("client_0", clients["client_0"], clients["client_0"]["local_client_fit_history"])
    write_client("client_1", clients["client_1"], clients["client_1"]["local_client_fit_history"])
    collected = collect_metrics(job, tmp_path, "completed", lambda: None)
    m = collected["metrics"]
    assert m["training_outcome"] == "trained" and m["clients_failed"] == "0 of 2 clients failed"
    assert m["num_clients_in_mean"] == 2 and collected["flat"] is False

    # A client that does not train on its own data is excluded from the mean and counted.
    zero = clients["client_0"]["local_client_fit_history"][0]["local_start_standardized_mse"]
    flat_history = [dict(row, local_start_standardized_mse=zero, local_end_standardized_mse=zero)
                    for row in clients["client_0"]["local_client_fit_history"]]
    import shutil

    shutil.rmtree(result / "client_0")
    write_client("client_0", clients["client_0"], flat_history)
    m = collect_metrics(job, tmp_path, "completed", lambda: None)["metrics"]
    assert m["clients_failed"] == "1 of 2 clients failed" and m["num_clients_in_mean"] == 1
    assert json.loads(m["client_outcomes"]) == {"client_0": "did not train", "client_1": "trained"}
    assert m["skill_mean"] == pytest.approx(collect_metrics(job, tmp_path, "completed", lambda: None)["per_client"]["client_1"]["skill"])
