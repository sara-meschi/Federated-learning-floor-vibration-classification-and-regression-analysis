"""S6: the paper figure and table suite.

Exact counts and reproduced numbers are independent literals (CLAUDE.md), not imported
from the modules. The data tests read the saved S5 / S3 / centralized results, which are
gitignored per-job folders; they skip where those folders are absent.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from redo_by_sara.paper_figures import (
    PDF_METADATA,
    Results,
    _family_marker,
    _save,
    _tex,
    _within_std,
    convergence_data,
    load_results,
    make_all_figures,
    partition_matrix,
    partition_schemes,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

#: The saved per-job results the suite reads (gitignored, so absent on a fresh clone).
S6_INPUTS = (
    ARTIFACTS / "s5/fl/k2_natural/seed4603/regression",
    ARTIFACTS / "s3_r1_sweep/fedavg_e05_persistent/regression",
    ARTIFACTS / "centralized_combined_residual_runbalanced_9ch_no006_300e/test_run_predictions.csv",
)
requires_results = pytest.mark.skipif(
    not all(p.exists() for p in S6_INPUTS), reason="saved S5/S3 per-job results not present"
)


# ------------------------------------------------------------------ fast: helpers


def test_tex_escapes_and_math() -> None:
    assert _tex("client_0") == r"client\_0"
    assert _tex("−0.207 ± 0.1") == r"$-$0.207 $\pm$ 0.1"
    assert _tex("K=3 rho=0.375") == r"K=3 $\rho$=0.375"
    assert _tex("Dirichlet alpha=0.1") == r"Dirichlet $\alpha$=0.1"
    assert _tex("95%") == r"95\%"


def test_within_std_is_pooled_over_groups() -> None:
    # groups centred on different means; within-group SS = 2 + 8 = 10, dof = 6 - 2
    assert math.isclose(_within_std([[1.0, 2.0, 3.0], [10.0, 12.0, 14.0]]), math.sqrt(10 / 4))


def test_outcome_markers_override_the_family_marker() -> None:
    assert _family_marker("fl", "trained")["marker"] == "o"
    assert _family_marker("local_only", None)["marker"] == "s"
    assert _family_marker("fl", "did not train")["marker"] == "^"
    assert _family_marker("local_only", "collapsed")["marker"] == "X"
    assert _family_marker("fl", "collapsed")["markerfacecolor"] == "white"


def test_partition_matrix_counts_runs_per_client_and_stratum() -> None:
    clients, groups, matrix = partition_matrix(
        {"a:001:000": "client_1", "a:001:001": "client_0", "b:003:000": "client_0", "a:003:002": "client_0"}
    )
    assert clients == ["client_0", "client_1"]
    assert groups == [("a", "001"), ("a", "003"), ("b", "003")]
    assert matrix.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_rebase_reroots_saved_absolute_paths(tmp_path: Path) -> None:
    results = Results(root=tmp_path, index=[], run_level=[], seed_aggregates=[])
    assert results.rebase("/elsewhere/repo/artifacts/s5/fl/x/regression") == tmp_path / "s5/fl/x/regression"
    with pytest.raises(ValueError):
        results.rebase("/no/marker/here")


def test_save_is_byte_identical_and_untimestamped(tmp_path: Path) -> None:
    assert PDF_METADATA["CreationDate"] is None

    def draw(folder: Path) -> list[Path]:
        fig, ax = plt.subplots(figsize=(3.5, 2.0))
        ax.plot([0, 1, 2], [0, 1, 4], marker="o")
        return _save(fig, folder, "probe")

    first, second = draw(tmp_path / "a"), draw(tmp_path / "b")
    for a, b in zip(first, second):
        assert a.read_bytes() == b.read_bytes()
    assert b"CreationDate" not in first[0].read_bytes()


def test_load_results_rejects_a_non_terminal_job(tmp_path: Path) -> None:
    (tmp_path / "s5/derived").mkdir(parents=True)
    (tmp_path / "s5/results_index.csv").write_text("job_id,family,condition,seed,task,status\nx,fl,k2_iid,4601,regression,running\n")
    for name in ("run_level_classification.csv", "seed_aggregates.csv"):
        (tmp_path / "s5/derived" / name).write_text("job_id\n")
    with pytest.raises(AssertionError, match="terminal"):
        load_results(tmp_path)


# ------------------------------------------------------------------ data: the real build


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("s6")
    make_all_figures(ARTIFACTS, out)
    return out


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@requires_results
def test_every_output_is_written(build: Path) -> None:
    figures = sorted(p.name for p in (build / "figures").iterdir())
    assert figures == sorted(
        [f"fig{i}_{n}.{ext}" for i, n in enumerate(
            ["convergence", "comparison", "sensor_axis", "partitions", "run_scatter"], start=1)
         for ext in ("pdf", "png")] + ["captions.md"]
    )
    tables = sorted(p.name for p in (build / "tables").iterdir())
    assert tables == sorted(
        [f"{stem}.{ext}" for stem in ("t1_dataset", "t2_centralized", "t3_s3_sweep", "t4_main_matrix", "t5_controls")
         for ext in ("tex", "csv")] + ["t4_main_matrix_summary.csv"]
    )


@requires_results
def test_rebuild_is_byte_identical(build: Path, tmp_path: Path) -> None:
    make_all_figures(ARTIFACTS, tmp_path)
    for path in sorted(build.rglob("*")):
        if path.is_file():
            assert (tmp_path / path.relative_to(build)).read_bytes() == path.read_bytes(), path.name


@requires_results
def test_figures_embed_truetype_not_type3(build: Path) -> None:
    for pdf in (build / "figures").glob("*.pdf"):
        data = pdf.read_bytes()
        assert b"/Type3" not in data and b"/FontFile2" in data, pdf.name


@requires_results
def test_t1_dataset_counts(build: Path) -> None:
    rows = _csv(build / "tables/t1_dataset.csv")
    total = next(r for r in rows if r["source"] == "All")
    assert (int(total["runs_train"]), int(total["runs_test"])) == (112, 28)
    assert (int(total["cls_train"]), int(total["cls_test"])) == (1242, 307)
    assert (int(total["reg_train"]), int(total["reg_test"])) == (966, 238)
    funnel = next(r for r in rows if r["source"] == "run funnel")
    assert (int(funnel["recorded_runs"]), int(funnel["excluded_runs"]), int(funnel["skipped_runs"]),
            int(funnel["usable_runs"])) == (161, 15, 6, 140)
    assert funnel["excluded_subjects"] == "006"
    # subject 003 appears once per source
    assert sorted(r["source"] for r in rows if r["subject"] == "003") == ["20251124", "Test_2"]


@requires_results
def test_t2_matched_steps_reference_reproduces_s3(build: Path) -> None:
    rows = _csv(build / "tables/t2_centralized.csv")
    matched = next(r for r in rows if r["model"] == "regression residual (matched steps)")
    assert math.isclose(float(matched["skill"]), 0.27166117850066207, abs_tol=1e-12)
    assert math.isclose(float(matched["ci_low"]), -2.2994175090340283, abs_tol=1e-12)
    assert math.isclose(float(matched["ci_high"]), 0.682232219675111, abs_tol=1e-12)
    assert (int(matched["epochs"]), int(matched["steps"]), matched["outcome"]) == (300, 8400, "trained")
    residual = [r for r in rows if r["model"] == "regression residual"]
    assert [round(float(r["skill"]), 3) for r in residual] == [0.121, 0.082, 0.091]
    assert {int(r["steps"]) for r in residual} == {1680}


@requires_results
def test_t3_sweep_keeps_failures_and_relabels(build: Path) -> None:
    rows = _csv(build / "tables/t3_s3_sweep.csv")
    assert len(rows) == 9
    assert [r["outcome_s5"] for r in rows] == ["collapsed"] * 3 + ["trained"] * 6
    assert sum(r["s3_acceptance"] == "passed" for r in rows) == 3
    e10_fresh = next(r for r in rows if r["config"] == "fedavg_e10_fresh")
    assert (e10_fresh["s3_acceptance"], e10_fresh["outcome_s5"]) == ("failed: skill", "trained")


@requires_results
def test_t4_matches_the_s5_results_entry(build: Path) -> None:
    rows = _csv(build / "tables/t4_main_matrix.csv")

    def pick(family: str, condition: str, task: str) -> list[dict[str, str]]:
        return [r for r in rows if (r["family"], r["condition"], r["task"]) == (family, condition, task)]

    natural = pick("fl", "k2_natural", "regression")
    assert [(int(r["seed"]), r["outcome"]) for r in natural] == [(4601, "trained"), (4602, "trained"), (4603, "collapsed")]
    assert [round(float(r["value"]), 3) for r in natural[:2]] == [-0.207, -0.123]
    dirichlet_local = pick("local_only", "k3_dirichlet0.1", "classification")
    assert sorted({int(r["seed"]) for r in dirichlet_local}) == [4601, 4602, 4603]
    assert [int(r["correct"]) for r in dirichlet_local] == [10, 6, 22] * 3
    assert [int(r["correct"]) for r in pick("fl", "k3_dirichlet0.1", "classification")] == [22, 22, 22]
    rho0_local = pick("local_only", "k3_rho0", "regression")
    assert [r["outcome"] for r in rho0_local] == ["did not train", "trained", "trained"]
    summary = {r["condition"]: r for r in _csv(build / "tables/t4_main_matrix_summary.csv")}
    assert summary["k2_natural"]["reg_local"].startswith("−0.726 ± 0.869 over 2")
    assert summary["k2_iid"]["reg_fl"].startswith("+0.172 ± 0.096, 3 of 3 trained")
    assert summary["centralized"]["reg_fl"].startswith("+0.098 ± 0.021")
    assert summary["k2_natural"]["reg_fl"].startswith("per seed; 1 of 3 seeds collapsed")
    assert summary["k3_rho0.375"]["reg_fl"].endswith("did not train")
    assert summary["k2_natural"]["cls_local"].startswith("0.554 ± 0.126")


@requires_results
def test_t5_controls(build: Path) -> None:
    rows = _csv(build / "tables/t5_controls.csv")
    cross = next(r for r in rows if r["set"] == "cross_session")
    assert (int(cross["run_vote_correct"]), int(cross["runs"]), int(cross["windows"])) == (0, 15, 142)
    reverse = [r for r in rows if r["control"] == "reverse direction (unplanned)"]
    s003 = next(r for r in reverse if r["subject"] == "003")
    assert (int(s003["run_vote_correct"]), int(s003["run_meanprob_correct"]), int(s003["runs"])) == (3, 2, 4)
    assert s003["window_labels"].startswith("003 0.571 (16)")
    assert [r["seen_in_training"] for r in reverse] == ["False", "False", "True"]
    negatives = [r for r in rows if r["control"] == "negative control"]
    assert sorted(r["subject"] for r in negatives) == ["004", "005", "007", "008"]


@requires_results
def test_convergence_draws_the_collapsed_seed_separately() -> None:
    data = convergence_data(load_results(ARTIFACTS))
    natural = data["curves"][("regression", "k2_natural")]
    assert [(s["seed"], s["label"]) for s in natural] == [(4601, None), (4602, None), (4603, "collapsed")]
    assert all(s["label"] is None for s in data["curves"][("regression", "k2_iid")])
    assert data["references"]["regression"]["centralized_300e_final_standardized_mse"] < 1e-5


@requires_results
def test_rho_layouts_share_one_run_assignment() -> None:
    schemes = partition_schemes(load_results(ARTIFACTS))
    assert [s["conditions"] for s in schemes] == [
        ["k2_iid"], ["k2_natural"], ["k3_iid", "k3_rho0.375", "k3_rho0"], ["k3_dirichlet0.1"]
    ]
    assert [len(set(s["assignment"].values())) for s in schemes] == [2, 2, 3, 3]
    assert all(len(s["assignment"]) == 112 for s in schemes)
