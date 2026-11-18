"""Alpha sweep (``alpha_sweep``, ``scripts/run_alpha_sweep.py``): job matrix, partition seed,
descriptors, the pre-registered statistics, the workbook.

Expected numbers are independent literals: the step-1 diagnosis of the relabel alpha = 0.1
and alpha = 0.3 regression runs (seed 4601) and the relabel results entry in CHANGES.md.
The data tests read gitignored results and skip where they are absent.
"""

from __future__ import annotations

import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
from openpyxl import load_workbook
from scipy.stats import spearmanr

from redo_by_sara import alpha_sweep as sweep
from redo_by_sara.combined_iid_fl_partitioning import build_combined_partitions
from redo_by_sara.combined_iid_flower import load_combined_iid_flower_config
from test_combined_iid_fl_partitioning import _synthetic_artifacts
from test_excel_figures import NOTES_XLSX, _hits, _name_hits

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RELABEL_FL = ARTIFACTS / "relabel/fl"
RELABEL_ARTIFACT = ARTIFACTS / "relabel/data/classification/raw_windows_combined.pt"
REGRESSION_ARTIFACT = ARTIFACTS / "centralized_combined_regression_9ch_no006_walking_80_20_60e/raw_walking_windows_regression.pt"
RELABEL_RUNS = [RELABEL_FL / f"k3_dirichlet{a}/seed4601/regression" for a in ("0.1", "0.3")]
needs_relabel_runs = pytest.mark.skipif(
    not all((d / "regression/test_run_predictions.csv").exists() for d in RELABEL_RUNS),
    reason="relabel Dirichlet regression results not present")
needs_artifacts = pytest.mark.skipif(not (RELABEL_ARTIFACT.exists() and REGRESSION_ARTIFACT.exists()),
                                     reason="identity-labelled / regression artifacts not present")


# ------------------------------------------------------------------ jobs and configs


def test_job_matrix() -> None:
    jobs = sweep.build_jobs()
    assert len(jobs) == 15 and len({j.job_id for j in jobs}) == 15
    confirmatory = {(j.alpha, j.partition_seed, j.seed) for j in jobs if j.role == "confirmatory"}
    controls = {(j.alpha, j.partition_seed, j.seed) for j in jobs if j.role == "init_control"}
    assert len(confirmatory) == 13
    assert controls == {(0.1, 4601, 4602), (0.3, 4601, 4602)}
    expected = {(a, p, 4601) for a in (0.05, 0.1, 0.3, 1.0, 10.0) for p in (4601, 5601, 6601)}
    assert confirmatory == expected - {(0.1, 4601, 4601), (0.3, 4601, 4601)}
    assert [j.group for j in jobs] == [1] * 3 + [2] * 5 + [3] * 5 + [4] * 2
    assert all(j.family == "fl" and j.task == "regression" for j in jobs)
    assert all("relabel" not in j.job_id and "s5" not in j.job_id for j in jobs)

    runs = sweep.sweep_runs(Path("/r/alpha_sweep"), Path("/r/relabel"))
    assert len(runs) == 17 and sum(r.role == "confirmatory" for r in runs) == 15
    reused = [r for r in runs if r.reused]
    assert [(r.alpha, r.draw, r.init_seed) for r in reused] == [(0.1, 1, 4601), (0.3, 1, 4601)]
    assert all(str(r.job_dir).startswith("/r/relabel/fl/k3_dirichlet0.") for r in reused)
    assert all(str(r.job_dir).startswith("/r/alpha_sweep/") for r in runs if not r.reused)


def test_commands_pass_both_seeds_and_write_under_the_root(tmp_path: Path) -> None:
    job = next(j for j in sweep.build_jobs() if j.alpha == 10.0 and j.partition_seed == 6601)
    cmd = sweep.command(job, tmp_path, "python")
    assert cmd[cmd.index("--config") + 1].endswith("configs/alpha_sweep_k3_dirichlet10.yaml")
    assert cmd[cmd.index("--output-root") + 1] == str(tmp_path / "fl/k3_dirichlet10/partition6601/init4601/regression")
    assert ["--set", "init_seed=4601", "--set", "partition_seed=6601"] == cmd[-4:]
    assert sweep.estimate_minutes(job, tmp_path, sweep.build_jobs()) == 32.0


def test_sweep_configs_are_the_relabel_dirichlet_config_with_only_alpha_changed() -> None:
    reference = load_combined_iid_flower_config(ROOT / "configs/relabel_k3_dirichlet0.1.yaml")
    for alpha in (0.05, 0.1, 0.3, 1.0, 10.0):
        config = load_combined_iid_flower_config(sweep.config_path(alpha))
        assert config["dirichlet_alpha"] == alpha
        assert config["partition_seed"] is None
        assert "artifacts/alpha_sweep/" in str(config["output_dir"])
        for key in set(reference) - {"config_path", "dirichlet_alpha", "output_dir"}:
            assert config[key] == reference[key], (alpha, key)
    assert (reference["aggregation"], reference["local_epochs"], reference["client_optimizer_state"],
            reference["num_rounds"], reference["client_num_cpus"], reference["split_seed"], reference["init_seed"]) == (
        "FedAvg", 5, "persistent", 60, 3.0, 4601, 4601)


# ------------------------------------------------------------------ partition seed


def test_partition_seed_is_dirichlet_only() -> None:
    with pytest.raises(ValueError, match="partition_seed"):
        load_combined_iid_flower_config(ROOT / "configs/relabel_k2_iid.yaml", {"partition_seed": 5601})
    config = load_combined_iid_flower_config(sweep.config_path(0.3), {"partition_seed": 5601})
    assert config["partition_seed"] == 5601
    classification, regression = _synthetic_artifacts()
    with pytest.raises(ValueError, match="dirichlet_seed"):
        build_combined_partitions(classification, regression, "iid", 3, dirichlet_seed=5601)


def test_absent_partition_seed_leaves_every_partition_and_audit_unchanged() -> None:
    classification, regression = _synthetic_artifacts()
    _, before = build_combined_partitions(classification, regression, "dirichlet", 3, seed=4601, dirichlet_alpha=0.3)
    _, same = build_combined_partitions(classification, regression, "dirichlet", 3, seed=4601, dirichlet_alpha=0.3,
                                        dirichlet_seed=4601)
    _, other = build_combined_partitions(classification, regression, "dirichlet", 3, seed=4601, dirichlet_alpha=0.3,
                                         dirichlet_seed=6601)
    assert "partition_seed" not in before["dirichlet"]
    assert same["run_to_client"] == before["run_to_client"] and same["dirichlet"]["partition_seed"] == 4601
    assert other["run_to_client"] != before["run_to_client"]
    assert before["seed"] == other["seed"] == 4601  # the split seed is untouched


@needs_artifacts
@needs_relabel_runs
def test_partition_seed_4601_rebuilds_the_relabel_partitions_byte_for_byte() -> None:
    from redo_by_sara.relabel_checks import load_artifact, run_to_client_text

    classification, regression = load_artifact(RELABEL_ARTIFACT), load_artifact(REGRESSION_ARTIFACT)
    for alpha, job_dir in zip((0.1, 0.3), RELABEL_RUNS):
        _, summary = build_combined_partitions(classification, regression, "dirichlet", 3, seed=4601,
                                               dirichlet_alpha=alpha, dirichlet_seed=4601)
        saved = (job_dir / "partitions/combined_iid_run_to_client.csv").read_bytes()
        assert run_to_client_text(summary).encode("utf-8") == saved, alpha
    # every new draw meets the 2-identity floor
    for job in sweep.build_jobs():
        _, summary = build_combined_partitions(classification, regression, "dirichlet", 3, seed=4601,
                                               dirichlet_alpha=job.alpha, dirichlet_seed=job.partition_seed)
        assert all(len(c["identities"]) >= 2 for c in summary["clients"]), job.job_id


# ------------------------------------------------------------------ descriptors and statistics


def _assignment(holdings: dict[str, dict[str, int]]) -> dict[str, str]:
    out, n = {}, 0
    for identity, clients in holdings.items():
        for client, count in clients.items():
            for _ in range(count):
                out[f"{identity}:{n:03d}"] = client
                n += 1
    return out


def test_partition_descriptors() -> None:
    holdings = {  # the relabel alpha = 0.3 partition (step 1), as counts
        "test_2:001": {"client_0": 4, "client_1": 2, "client_2": 11}, "test_2:002": {"client_2": 18},
        "test_2:003": {"client_2": 17}, "testing_20251124:003": {"client_0": 5, "client_1": 4, "client_2": 3},
        "testing_20251124:004": {"client_0": 9, "client_2": 3}, "testing_20251124:005": {"client_1": 13},
        "testing_20251124:007": {"client_1": 1, "client_2": 10}, "testing_20251124:008": {"client_1": 1, "client_2": 11},
    }
    d = sweep.partition_descriptors(_assignment(holdings))
    assert d["client_runs"] == "18/21/73" and d["largest_client"] == "client_2"
    assert math.isclose(d["largest_client_share"], 73 / 112)
    sizes = np.array([18, 21, 73])
    assert math.isclose(d["client_run_cv"], sizes.std() / sizes.mean())
    assert math.isclose(d["focus_share_on_largest"], 10 / 11) and d["focus_num_clients"] == 2
    assert d["posthoc_isolated_list"] == "testing_20251124:005" and d["posthoc_isolated_identities"] == 1
    assert math.isclose(d["share_on_largest:test_2:002"], 1.0)
    with pytest.raises(AssertionError):
        sweep.partition_descriptors({"test_2:001:000": "client_0"})


def test_spearman_and_permutation_p() -> None:
    rng = np.random.default_rng(3)
    x = np.repeat([0.05, 0.1, 0.3, 1.0, 10.0], 3)  # ties, as alpha has
    y = rng.normal(size=15)
    assert math.isclose(sweep.spearman(x, y), spearmanr(x, y).statistic, abs_tol=1e-12)
    p = sweep.permutation_p(x, y, permutations=2000)
    assert 0 < p <= 1 and p == sweep.permutation_p(x, y, permutations=2000)
    monotone = sweep.permutation_p(x, x + rng.normal(scale=1e-3, size=15), permutations=2000)
    assert monotone < 0.01
    assert math.isnan(sweep.spearman([1, 1, 1], [1, 2, 3]))


def _rows(rmse: dict[float, list[float]], spread: dict[float, list[float]]) -> list[dict]:
    rows = []
    for draw in range(3):
        for alpha in (0.05, 0.1, 0.3, 1.0, 10.0):
            rows.append({"role": "confirmatory", "alpha": alpha, "draw": draw + 1, "status": "completed",
                         "rmse": rmse[alpha][draw], "skill": 1 - (rmse[alpha][draw] / 0.0405) ** 2,
                         "spread_runs": spread[alpha][draw]})
    return rows


def test_pre_registered_rules() -> None:
    alphas = (0.05, 0.1, 0.3, 1.0, 10.0)
    # RMSE set by spread, alpha irrelevant; big draw-to-draw scatter within every alpha
    spread = {a: [0.2, 1.0, 0.5] if i % 2 else [1.0, 0.2, 0.6] for i, a in enumerate(alphas)}
    rmse = {a: [0.035 + 0.01 * s for s in spread[a]] for a in alphas}
    analysis = sweep.analyse(_rows(rmse, spread))
    assert analysis["complete"] is True and analysis["confirmatory_runs_measured"] == 15
    assert analysis["spread_vs_rmse"]["rho"] == pytest.approx(1.0)
    assert analysis["spread_rule_supported"] is True
    assert analysis["variance_rule"]["draw_dominates"] is True
    # RMSE set by alpha, identical draws: the variance rule goes the other way
    rmse = {a: [0.03 + 0.002 * i] * 3 for i, a in enumerate(alphas)}
    rmse = {a: [v + 1e-5 * d for d, v in enumerate(vs)] for a, vs in rmse.items()}
    analysis = sweep.analyse(_rows(rmse, spread))
    assert analysis["variance_rule"]["draw_dominates"] is False
    assert analysis["primary_alpha_vs_rmse"]["rho"] > 0.9 and analysis["primary_alpha_vs_rmse"]["p"] < 0.05
    assert analysis["spread_rule_supported"] is False
    # partial: nothing is concluded
    partial = sweep.analyse(_rows(rmse, spread)[:2])
    assert partial["complete"] is False and partial["spread_rule_supported"] is None
    assert partial["variance_rule"]["draw_dominates"] is None


@needs_relabel_runs
def test_descriptors_of_the_relabel_runs_match_the_step1_diagnosis() -> None:
    runs = {r.alpha: sweep.run_row(r) for r in sweep.sweep_runs() if r.reused}
    a1, a3 = runs[0.1], runs[0.3]
    for row, rmse, skill, spread, saved, focus, other in (
            (a1, 0.0389, 0.079, 0.1752, 0.1973, -0.003825, 0.000177),
            (a3, 0.0451, -0.240, 1.0044, 1.7718, -0.022454, 0.033471)):
        assert row["outcome"] == "trained"
        assert abs(row["rmse"] - rmse) < 5e-5 and abs(row["skill"] - skill) < 5e-4
        assert abs(row["spread_runs"] - spread) < 5e-5 and abs(row["spread_saved_window"] - saved) < 5e-5
        assert abs(row["dsse_focus"] - focus) < 5e-7 and abs(row["dsse_other"] - other) < 5e-7
        assert abs(row["lookup_rmse"] - 0.0405) < 5e-5
    assert (a1["client_runs"], a3["client_runs"]) == ("20/17/75", "18/21/73")
    assert a1["posthoc_isolated_list"] == "testing_20251124:004 testing_20251124:005"
    assert a3["posthoc_isolated_list"] == "testing_20251124:005"
    assert (a1["focus_share_on_largest"], a1["focus_num_clients"]) == (1.0, 1)


# ------------------------------------------------------------------ workbook

IDENTITIES = ("test_2:001", "test_2:002", "test_2:003", "testing_20251124:003", "testing_20251124:004",
              "testing_20251124:005", "testing_20251124:007", "testing_20251124:008")


def synthetic_rows(seed: int = 11) -> list[dict]:
    """17 measured rows in ``run_row``'s shape (15 confirmatory, 2 controls)."""
    rng = np.random.default_rng(seed)
    rows = []
    for run in sweep.sweep_runs(Path("/r/alpha_sweep"), Path("/r/relabel")):
        rmse = float(rng.uniform(0.034, 0.05))
        skill = 1 - (rmse / 0.0405) ** 2
        rows.append({
            "role": run.role, "alpha": run.alpha, "draw": run.draw, "partition_seed": run.partition_seed,
            "init_seed": run.init_seed, "reused_from_relabel": run.reused, "status": "completed", "outcome": "trained",
            "rmse": rmse, "lookup_rmse": 0.0405, "rmse_ci_low": rmse - 0.01, "rmse_ci_high": rmse + 0.01,
            "skill": skill, "skill_ci_low": skill - 0.5, "skill_ci_high": skill + 0.3,
            "spread_runs": float(rng.uniform(0.1, 1.2)), "spread_saved_window": float(rng.uniform(0.1, 2.0)),
            "dsse_focus": float(rng.uniform(-0.03, 0)), "dsse_other": float(rng.uniform(0, 0.04)),
            "client_runs": "37/38/37", "largest_client": "client_1",
            "largest_client_share": float(rng.uniform(0.34, 0.7)), "client_run_cv": float(rng.uniform(0, 0.8)),
            "mean_identity_share_on_largest": float(rng.uniform(0.3, 0.7)),
            **{f"share_on_largest:{i}": float(rng.uniform(0, 1)) for i in IDENTITIES},
            "focus_share_on_largest": float(rng.uniform(0, 1)), "focus_num_clients": int(rng.integers(1, 4)),
            "focus_clients": "{}", "posthoc_isolated_identities": int(rng.integers(0, 4)),
            "posthoc_isolated_list": "", "identity_client_runs": "{}",
            "predictions_file": "artifacts/x/test_run_predictions.csv", "partition_file": "artifacts/x/run_to_client.csv",
        })
    return rows


def test_workbook_formulas_point_at_their_columns() -> None:
    rows = synthetic_rows()
    wb = sweep.build_workbook(rows, sweep.analyse(rows), "test")
    ws = wb["data"]
    header = [c.value for c in ws[1]]
    letter = {h: ws.cell(row=1, column=i + 1).column_letter for i, h in enumerate(header) if h}
    for rank, source in (("rank_alpha", "alpha"), ("rank_rmse", "value"), ("rank_spread", "spread_runs")):
        formula = ws[f"{letter[rank]}2"].value
        assert f"${letter[source]}$2:${letter[source]}$16" in formula and f"<{letter[source]}2" in formula, rank
        assert f'{letter["role"]}17="confirmatory"' in ws[f"{letter[rank]}17"].value, rank  # controls rank blank
    assert ws[f"{letter['err_plus']}2"].value == f"={letter['ci_upper']}2-{letter['value']}2"
    stats = [str(c.value) for row in wb["stats"].iter_rows() for c in row if c.value]
    assert any(s.startswith("=CORREL(") for s in stats) and any(s.startswith("=STDEV(") for s in stats)
    assert len([v for v in (ws.cell(row=r, column=header.index("summary_alpha") + 1).value for r in range(2, 8)) if v]) == 5



@needs_relabel_runs
def test_workbook_follows_the_excel_conventions(tmp_path: Path) -> None:
    path = sweep.make_workbook(ARTIFACTS, tmp_path / "alpha_sweep.xlsx", ROOT)
    wb = load_workbook(path)
    assert wb.sheetnames[:2] == ["data", "chart"] and {"stats", "notes"} <= set(wb.sheetnames)
    assert wb.calculation.fullCalcOnLoad
    assert len(wb["chart"]._charts) >= 2  # noqa: SLF001
    notes = " ".join(str(c.value) for row in wb["notes"].iter_rows() for c in row if c.value)
    assert "PRE-REGISTERED" in notes and "Format Data Point" in notes and "commit" in notes
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None:
                    assert cell.font.name == "Arial", (ws.title, cell.coordinate)
    header = [c.value for c in wb["data"][1]]
    col = {h: i for i, h in enumerate(header) if h}
    rows = [r for r in wb["data"].iter_rows(min_row=2, values_only=True) if r[col["role"]]]
    assert {(r[col["alpha"]], r[col["draw"]]) for r in rows} >= {(0.1, 1), (0.3, 1)}
    for r in rows:
        assert r[col["err_plus"]].startswith("=") and r[col["skill_err_minus"]].startswith("=")
    with zipfile.ZipFile(path) as z:
        for part in (p for p in z.namelist() if p.startswith("xl/charts/")):
            for ref in re.findall(r"<f>(.*?)</f>", z.read(part).decode()):
                sheet, cells = ref.rsplit("!", 1)
                assert sheet.strip("'") in wb.sheetnames and re.fullmatch(r"\$[A-Z]+\$\d+(:\$[A-Z]+\$\d+)?", cells)
    assert _hits([path], re.compile("linked", re.IGNORECASE)) == []
    if NOTES_XLSX.exists():
        assert _name_hits([path]) == []


def test_sweep_module_reads_no_merged_or_cross_session_field() -> None:
    for path in (ROOT / "src/redo_by_sara/alpha_sweep.py", ROOT / "scripts/run_alpha_sweep.py",
                 ROOT / "scripts/make_alpha_sweep_excel.py"):
        source = path.read_text()
        for forbidden in ("metrics_by_subject", "speed_distribution_by_subject", "linked_identity", "identity_links",
                          "cross_session", "subject003_control"):
            assert forbidden not in source, (path.name, forbidden)
        assert not re.search(r"(?<!source_)subject_train_run_mean", source), path.name
