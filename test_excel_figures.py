"""Excel figure workbooks (``excel_figures``).

The expected numbers are independent literals taken from the CHANGES.md relabel results
entry (its T4 summary and regression detail), not imported from the modules. The data
tests read the saved relabel / S5 per-job results, which are gitignored; they skip where
those folders are absent.
"""

from __future__ import annotations

import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from openpyxl import load_workbook

from redo_by_sara.excel_figures import matched_bootstrap
from redo_by_sara.regression_metrics import bootstrap_skill_ci

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
NOTES_XLSX = ROOT / "TestData" / "20251124_Testing" / "testNotes.xlsx"

INPUTS = (
    ARTIFACTS / "relabel/local_only/k3_dirichlet0.1/seed4601/regression/local_only/regression/client_2",
    ARTIFACTS / "relabel/fl/k2_iid/seed4601/classification/classification",
    ARTIFACTS / "s5/fl/k2_natural/seed4603/regression",
    ARTIFACTS / "s3_r1_sweep/fedavg_e05_persistent/regression",
    ARTIFACTS / "centralized_combined_residual_runbalanced_9ch_no006_300e/test_run_predictions.csv",
)
requires_results = pytest.mark.skipif(
    not all(p.exists() for p in INPUTS), reason="saved relabel/S5/S3 per-job results not present"
)

WORKBOOKS = (
    "fig1_convergence.xlsx",
    "fig2_classification_comparison.xlsx",
    "fig2_regression_skill.xlsx",
    "fig2_regression_rmse.xlsx",
    "fig3_sensor_axis.xlsx",
    "fig4_partitions.xlsx",
    "fig5_run_scatter.xlsx",
    "fig6_subject_skew.xlsx",
    "confusion_matrices.xlsx",
    "tables.xlsx",
)
TOL = 5e-4  # CHANGES.md prints 3 decimals (4 for RMSE)

SWEEP_DERIVED = ARTIFACTS / "alpha_sweep" / "derived" / "alpha_sweep_runs.csv"
requires_sweep = pytest.mark.skipif(
    not (all(p.exists() for p in INPUTS) and SWEEP_DERIVED.exists()),
    reason="saved relabel results or the alpha-sweep derived tables not present")


# ------------------------------------------------------------------ fast


def test_matched_bootstrap_reproduces_the_skill_interval() -> None:
    rng = np.random.default_rng(7)
    actual = rng.normal(1.2, 0.1, 28)
    baseline = actual + rng.normal(0, 0.04, 28)
    predicted = actual + rng.normal(0, 0.035, 28)
    boot = matched_bootstrap(actual, predicted, baseline)
    assert (boot["skill_ci_low"], boot["skill_ci_high"]) == bootstrap_skill_ci(actual, predicted, baseline)
    rmse = math.sqrt(float(np.mean((predicted - actual) ** 2)))
    assert boot["rmse_ci_low"] < rmse < boot["rmse_ci_high"]


def test_excel_suite_reads_neither_merged_fields_nor_cross_session_results() -> None:
    """No field that keys on subject alone (merging the two 003s), and no reader of the
    cross-session control, its re-scores or the per-identity breakdown across campaigns."""
    for path in (ROOT / "src/redo_by_sara/excel_figures.py", ROOT / "scripts/make_excel_figures.py"):
        source = path.read_text()
        for forbidden in ("metrics_by_subject", "speed_distribution_by_subject", "linked_identity_breakdown",
                          "identity_links", "reverse_direction", "cross_session_control",
                          "subject003_control_summary"):
            assert forbidden not in source, (path.name, forbidden)
        assert not re.search(r"(?<!source_)subject_train_run_mean", source), path.name


# ------------------------------------------------------------------ data


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from redo_by_sara.excel_figures import make_all_workbooks

    out = tmp_path_factory.mktemp("excel")
    paths = make_all_workbooks(ARTIFACTS, out, repo=ROOT)
    assert [p.name for p in paths] == list(WORKBOOKS)
    return out


def _rows(path: Path, sheet: str = "data") -> list[dict[str, object]]:
    ws = load_workbook(path)[sheet]
    header = [c.value for c in ws[1]]
    header = header[: header.index(None)] if None in header else header  # the main table only
    out = []
    for row in ws.iter_rows(min_row=2, max_col=len(header), values_only=True):
        if any(v is not None for v in row):
            out.append(dict(zip(header, row)))
    return out


def _keyed(rows: list[dict[str, object]]) -> dict[tuple, dict[str, object]]:
    return {(r["condition"], r["family"], r.get("client") or "", r["seed"]): r for r in rows}


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


@requires_results
def test_every_workbook_opens_with_charts_notes_arial_and_calc_on_load(built: Path) -> None:
    for name in WORKBOOKS:
        wb = load_workbook(built / name)
        assert wb.calculation.fullCalcOnLoad, name
        assert "notes" in wb.sheetnames, name
        charts = sum(len(ws._charts) for ws in wb.worksheets)  # noqa: SLF001
        if name == "tables.xlsx":
            assert {"T1_dataset", "T2_centralized", "T3_r1_sweep", "T4_main_matrix", "T5_source_probe",
                    "RMSE"} | ({"T6_alpha_sweep"} if SWEEP_DERIVED.exists() else set()) <= set(wb.sheetnames)
        else:
            assert wb.sheetnames[:2] == ["data", "chart"], name
            assert charts >= 1, name
        notes = " ".join(str(c.value) for row in wb["notes"].iter_rows() for c in row if c.value)
        assert "Format Data Point" in notes and "commit" in notes, name
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    if cell.value is not None:
                        assert cell.font.name == "Arial", (name, ws.title, cell.coordinate)


@requires_results
def test_numbers_are_numbers_and_error_bars_are_formulas(built: Path) -> None:
    for name in WORKBOOKS:
        wb = load_workbook(built / name)
        for ws in wb.worksheets:
            header = [c.value for c in ws[1]]
            if not {"value", "ci_lower", "ci_upper", "err_plus", "err_minus"} <= set(header):
                continue
            col = {h: i for i, h in enumerate(header) if h}
            letter = {h: ws.cell(row=1, column=i + 1).column_letter for h, i in col.items()}
            for r, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
                if row[col["value"]] is None:
                    continue
                for key in ("value", "ci_lower", "ci_upper"):
                    if row[col[key]] is not None:
                        assert isinstance(row[col[key]], (int, float)), (name, ws.title, key, r)
                if row[col["ci_lower"]] is None:
                    continue
                assert row[col["err_plus"]] == f"={letter['ci_upper']}{r}-{letter['value']}{r}", (name, ws.title, r)
                assert row[col["err_minus"]] == f"={letter['value']}{r}-{letter['ci_lower']}{r}", (name, ws.title, r)
                assert row[col["ci_lower"]] <= row[col["value"]] + 1e-12 <= row[col["ci_upper"]] + 2e-12, (name, r)


def _assert_chart_refs(path: Path) -> None:
    """Every chart reference names a sheet of the workbook and an absolute cell range."""
    wb = load_workbook(path)
    with zipfile.ZipFile(path) as z:
        for part in (p for p in z.namelist() if p.startswith("xl/charts/")):
            refs = re.findall(r"<f>(.*?)</f>", z.read(part).decode())
            assert refs, (path.name, part)
            for ref in refs:
                sheet, cells = ref.rsplit("!", 1)
                assert sheet.strip("'") in wb.sheetnames, (path.name, part, ref)
                assert re.fullmatch(r"\$[A-Z]+\$\d+(:\$[A-Z]+\$\d+)?", cells), (path.name, part, ref)


@requires_results
def test_charts_reference_cells_of_the_workbook(built: Path) -> None:
    for name in WORKBOOKS:
        _assert_chart_refs(built / name)


@requires_results
def test_fig2_classification_matches_t4(built: Path) -> None:
    rows = _rows(built / "fig2_classification_comparison.xlsx")
    assert len(rows) == 49
    k = _keyed(rows)
    for seed in (4601, 4602, 4603):
        for condition, family in (("centralized", "centralized"), ("k2_iid", "fl"), ("k2_natural", "fl")):
            r = k[(condition, family, "", seed)]
            assert (r["correct"], r["runs"]) == (28, 28)
            assert _close(r["ci_lower"], 0.879) and _close(r["ci_upper"], 1.000)
        r = k[("k3_dirichlet0.1", "fl", "", seed)]
        assert (r["correct"], _close(r["ci_lower"], 0.605), _close(r["ci_upper"], 0.898)) == (22, True, True)
        r = k[("k3_dirichlet0.3", "fl", "", seed)]
        assert (r["correct"], _close(r["ci_lower"], 0.728), _close(r["ci_upper"], 0.963)) == (25, True, True)
    for condition in ("k3_iid", "k3_rho0.375", "k3_rho0"):
        assert k[(condition, "fl", "", 4601)]["correct"] == 28

    def local(condition: str, seed: int = 4601) -> list[int]:
        return [k[(condition, "local_only", f"client_{c}", seed)]["correct"]
                for c in range(3) if (condition, "local_only", f"client_{c}", seed) in k]

    assert local("k2_iid") == [28, 28]
    assert local("k2_natural") == [13, 15]
    assert _close(k[("k2_natural", "local_only", "client_0", 4601)]["ci_lower"], 0.295)
    assert _close(k[("k2_natural", "local_only", "client_1", 4601)]["ci_upper"], 0.705)
    assert local("k3_iid") == [27, 28, 28]
    assert _close(k[("k3_iid", "local_only", "client_0", 4601)]["ci_lower"], 0.823)
    assert local("k3_rho0.375") == [28, 28, 26]
    assert local("k3_rho0") == [28, 28, 26]
    assert _close(k[("k3_rho0", "local_only", "client_2", 4601)]["ci_upper"], 0.980)
    for seed in (4601, 4602, 4603):
        assert local("k3_dirichlet0.1", seed) == [10, 6, 22]
    assert local("k3_dirichlet0.3", 4601) == [10, 16, 25]
    assert local("k3_dirichlet0.3", 4602) == [10, 16, 24]
    assert local("k3_dirichlet0.3", 4603) == [10, 16, 25]
    for condition, mean, std in (("k2_natural", 0.500, 0.051), ("k3_iid", 0.988, 0.021),
                                 ("k3_rho0.375", 0.976, 0.041), ("k3_rho0", 0.976, 0.041),
                                 ("k3_dirichlet0.1", 0.452, 0.297), ("k3_dirichlet0.3", 0.607, 0.270)):
        m, s = _mean_std([c / 28 for c in local(condition)])
        assert _close(m, mean) and _close(s, std), condition
    m, s = _mean_std([c / 28 for c in local("k3_dirichlet0.3", 4602)])
    assert _close(m, 0.595) and _close(s, 0.251)


SKILL_FL = {  # (condition, seed): (skill, ci_low, ci_high, outcome); None = not printed in CHANGES
    ("k2_natural", 4601): (-0.207, None, None, "trained"),
    ("k2_natural", 4602): (-0.123, None, None, "trained"),
    ("k2_natural", 4603): (None, None, None, "collapsed"),
    ("k3_iid", 4601): (0.101, -2.263, 0.480, "trained"),
    ("k3_rho0.375", 4601): (0.046, None, None, "did not train"),
    ("k3_rho0", 4601): (None, None, None, "collapsed"),
    ("k3_dirichlet0.1", 4601): (0.079, -0.106, 0.226, "trained"),
    ("k3_dirichlet0.3", 4601): (-0.240, -4.913, 0.476, "trained"),
}
SKILL_LOCAL_DIRICHLET = {  # (condition, client): (skill, ci_low, ci_high, outcome, rmse)
    ("k3_dirichlet0.1", "client_0"): (-0.463, -2.985, 0.110, "trained", 0.0490),
    ("k3_dirichlet0.1", "client_1"): (0.183, -0.391, 0.378, "trained", 0.0366),
    ("k3_dirichlet0.1", "client_2"): (0.003, -0.001, 0.008, "collapsed", 0.0405),
    ("k3_dirichlet0.3", "client_0"): (-0.362, -1.855, -0.064, "trained", 0.0473),
    ("k3_dirichlet0.3", "client_1"): (-0.065, -1.569, 0.234, "trained", 0.0418),
    ("k3_dirichlet0.3", "client_2"): (-0.254, -5.301, 0.526, "trained", 0.0454),
}
LOCAL_MEAN_TRAINED = {  # condition: (mean, std, trained clients)
    "k2_iid": (0.010, 0.358, 2), "k2_natural": (-0.726, 0.869, 2), "k3_iid": (-0.125, 0.343, 3),
    "k3_rho0.375": (-0.329, 0.389, 3), "k3_rho0": (-0.073, 0.122, 2),
    "k3_dirichlet0.1": (-0.140, 0.457, 2), "k3_dirichlet0.3": (-0.227, 0.150, 3),
}


@requires_results
def test_fig2_regression_skill_matches_t4(built: Path) -> None:
    rows = _rows(built / "fig2_regression_skill.xlsx")
    assert len(rows) == 33
    k = _keyed(rows)
    central = [k[("centralized", "centralized", "", s)] for s in (4601, 4602, 4603)]
    assert _close(_mean_std([r["value"] for r in central])[0], 0.098)
    assert _close(_mean_std([r["value"] for r in central])[1], 0.021)
    assert all(r["outcome"] == "trained" for r in central)
    iid = [k[("k2_iid", "fl", "", s)]["value"] for s in (4601, 4602, 4603)]
    assert _close(_mean_std(iid)[0], 0.172) and _close(_mean_std(iid)[1], 0.096)
    for (condition, seed), (skill, low, high, outcome) in SKILL_FL.items():
        r = k[(condition, "fl", "", seed)]
        assert r["outcome"] == outcome, (condition, seed)
        for got, want in ((r["value"], skill), (r["ci_lower"], low), (r["ci_upper"], high)):
            assert want is None or _close(got, want), (condition, seed)
    # natural K=2 is per seed: three FL rows, never one averaged row
    assert sorted(s for (c, f, _, s) in k if c == "k2_natural" and f == "fl") == [4601, 4602, 4603]
    for (condition, client), (skill, low, high, outcome, _) in SKILL_LOCAL_DIRICHLET.items():
        r = k[(condition, "local_only", client, 4601)]
        assert (r["outcome"], _close(r["value"], skill), _close(r["ci_lower"], low),
                _close(r["ci_upper"], high)) == (outcome, True, True, True), (condition, client)
    for condition, (mean, std, n) in LOCAL_MEAN_TRAINED.items():
        trained = [r["value"] for (c, f, _, _), r in k.items()
                   if c == condition and f == "local_only" and r["outcome"] == "trained"]
        assert len(trained) == n, condition
        assert _close(_mean_std(trained)[0], mean) and _close(_mean_std(trained)[1], std), condition
    # the outcome label is in the data labels too
    with zipfile.ZipFile(built / "fig2_regression_skill.xlsx") as z:
        chart = z.read("xl/charts/chart1.xml").decode()
    def labels(text: str) -> int:
        return chart.count('formatCode="' + ";".join([f"&quot;{text}&quot;"] * 3) + '"')

    assert (labels("collapsed"), labels("did not train"), labels("trained")) == (3, 2, 28)


@requires_results
def test_fig2_regression_skill_has_full_and_zoomed_charts(built: Path) -> None:
    with zipfile.ZipFile(built / "fig2_regression_skill.xlsx") as z:
        first, second = (z.read(f"xl/charts/chart{i}.xml").decode() for i in (1, 2))
    assert "(full range)" in first
    assert "Regression skill (zoomed: some intervals extend beyond the axis)" in second
    assert '<max val="1" /><min val="-1.5" />' in second
    assert re.search(r'<min val="-(1[01](\.5)?)" />', first)


@requires_results
def test_fig2_regression_rmse_is_exact(built: Path) -> None:
    rows = _rows(built / "fig2_regression_rmse.xlsx")
    assert len(rows) == 33
    k = _keyed(rows)
    assert all(_close(r["lookup_rmse"], 0.0405, 5e-5) for r in rows)
    assert _close(k[("k3_dirichlet0.1", "fl", "", 4601)]["value"], 0.0389, 5e-5)
    assert _close(k[("k3_dirichlet0.3", "fl", "", 4601)]["value"], 0.0451, 5e-5)
    for (condition, client), (*_, rmse) in SKILL_LOCAL_DIRICHLET.items():
        assert _close(k[(condition, "local_only", client, 4601)]["value"], rmse, 5e-5), (condition, client)
    for r in rows:  # RMSE and skill agree exactly: skill = 1 - (rmse / lookup)^2
        assert math.isclose(r["skill"], 1 - (r["value"] / r["lookup_rmse"]) ** 2, abs_tol=1e-9)


@requires_results
def test_fig3_fig4_fig5_contents(built: Path) -> None:
    fig3 = _rows(built / "fig3_sensor_axis.xlsx")
    fl = {(r["task"], r["rho"]): r for r in fig3 if r["family"] == "fl"}
    assert {k: v["outcome"] for k, v in fl.items() if k[0] == "regression"} == {
        ("regression", 1.0): "trained", ("regression", 0.375): "did not train", ("regression", 0.0): "collapsed"}
    assert _close(fl[("regression", 1.0)]["value"], 0.101)
    assert all(fl[("classification", rho)]["value"] == 1.0 for rho in (1.0, 0.375, 0.0))

    fig4 = _rows(built / "fig4_partitions.xlsx")
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for r in fig4:
        totals[(r["scheme"], r["client"])] += r["runs"]
    assert totals[("K=2 IID", "client_0")] == totals[("K=2 IID", "client_1")] == 56
    assert [totals[("K=3 IID (ρ=1.0, 0.375, 0)", f"client_{c}")] for c in range(3)] == [37, 38, 37]
    assert [totals[("K=3 Dirichlet α=0.1", f"client_{c}")] for c in range(3)] == [20, 17, 75]
    assert [totals[("K=3 Dirichlet α=0.3", f"client_{c}")] for c in range(3)] == [18, 21, 73]
    assert len({r["identity"] for r in fig4}) == 8
    assert {r["scheme"] for r in fig4} == {"K=2 IID", "K=2 natural", "K=3 IID (ρ=1.0, 0.375, 0)",
                                           "K=3 Dirichlet α=0.3", "K=3 Dirichlet α=0.1"}

    fig5 = _rows(built / "fig5_run_scatter.xlsx")
    assert len(fig5) == 4 * 28
    assert {r["identity"] for r in fig5} == {
        "test_2:001", "test_2:002", "test_2:003", "testing_20251124:003", "testing_20251124:004",
        "testing_20251124:005", "testing_20251124:007", "testing_20251124:008"}


@requires_results
def test_confusion_matrices_match_t4(built: Path) -> None:
    rows = _rows(built / "confusion_matrices.xlsx")
    totals: dict[tuple[str, str], int] = defaultdict(int)
    diagonal: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        totals[(r["unit"], r["model_key"])] += r["count"]
        if r["true_class"] == r["predicted_class"]:
            diagonal[(r["unit"], r["model_key"])] += r["count"]
    run = {k: v for (u, k), v in totals.items() if u == "run"}
    window = {k: v for (u, k), v in totals.items() if u == "window"}
    assert len(run) == 49 and set(run.values()) == {28}
    assert len(window) == 47 and set(window.values()) == {307}
    assert set(run) - set(window) == {"local_only/k2_natural/seed4601/client_0",
                                      "local_only/k2_natural/seed4601/client_1"}
    classes = {r["predicted_class"] for r in rows}
    assert classes == {"no_walking", "test_2:001", "test_2:002", "test_2:003", "testing_20251124:003",
                       "testing_20251124:004", "testing_20251124:005", "testing_20251124:007",
                       "testing_20251124:008"}
    assert {r["true_class"] for r in rows if r["unit"] == "run"} == classes - {"no_walking"}

    def correct(key: str) -> int:
        return diagonal[("run", key)]

    for seed in (4601, 4602, 4603):
        for key in (f"centralized/seed{seed}", f"fl/k2_iid/seed{seed}", f"fl/k2_natural/seed{seed}"):
            assert correct(key) == 28, key
        assert correct(f"fl/k3_dirichlet0.1/seed{seed}") == 22
        assert correct(f"fl/k3_dirichlet0.3/seed{seed}") == 25
        assert [correct(f"local_only/k3_dirichlet0.1/seed{seed}/client_{c}") for c in range(3)] == [10, 6, 22]
    assert [correct(f"local_only/k2_natural/seed4601/client_{c}") for c in range(2)] == [13, 15]
    assert [correct(f"local_only/k3_iid/seed4601/client_{c}") for c in range(3)] == [27, 28, 28]
    assert [correct(f"local_only/k3_rho0/seed4601/client_{c}") for c in range(3)] == [28, 28, 26]
    assert [correct(f"local_only/k3_dirichlet0.3/seed4602/client_{c}") for c in range(3)] == [10, 16, 24]

    recall = _rows(built / "confusion_matrices.xlsx", "recall")
    assert len(recall) == 49 * 8
    for key in run:
        assert sum(r["correct"] for r in recall if r["model_key"] == key) == correct(key), key
        assert sum(r["runs"] for r in recall if r["model_key"] == key) == 28, key


def _text_content(xml: str) -> str:
    """Text nodes and attribute values; tag and attribute names are left out (the chart
    schema's own ``sourceLinked`` attribute name is markup, not content)."""
    return " ".join(re.findall(r">([^<]+)<", xml) + re.findall(r'="([^"]*)"', xml))


def _hits(paths: list[Path], pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    found = []
    for path in paths:
        with zipfile.ZipFile(path) as z:
            for part in z.namelist():
                if part.endswith((".xml", ".rels")) and pattern.search(_text_content(z.read(part).decode())):
                    found.append((path.name, part))
    return found


def _name_hits(paths: list[Path]) -> list[tuple[str, str]]:
    """(workbook, XML part) pairs containing a personal name. The names are read here from
    the local, untracked demographics sheet and never leave this function."""
    wb = load_workbook(NOTES_XLSX, read_only=True)
    rows = list(wb["TestOverview"].iter_rows(values_only=True))
    header = next(i for i, row in enumerate(rows) if "Name" in row)
    column = rows[header].index("Name")
    number = rows[header].index("Number used")
    tokens = set()
    for row in rows[header + 1:]:
        # the subject table only: free-text notes further down share the column
        if not re.fullmatch(r"\d{3}", str(row[number] or "")):
            break
        value = row[column]
        if isinstance(value, str) and value.strip():
            tokens.add(value.strip())
            tokens.update(t for t in re.split(r"[\s,.]+", value) if len(t) >= 3)
    if not tokens:
        raise AssertionError("no names found in the demographics sheet")
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(tokens)) + r")\b", re.IGNORECASE)
    return _hits(paths, pattern)


@requires_results
def test_no_workbook_mentions_linked(built: Path) -> None:
    assert _hits([built / n for n in WORKBOOKS], re.compile("linked", re.IGNORECASE)) == []


@requires_results
@pytest.mark.skipif(not NOTES_XLSX.exists(), reason="demographics sheet not present")
def test_no_workbook_contains_a_personal_name(built: Path) -> None:
    assert _name_hits([built / n for n in WORKBOOKS]) == []


# ------------------------------------------------------------------ alpha sweep, Fig 6, T6

#: The 17-row alpha-sweep table (CHANGES.md, "Alpha sweep — Results"), as independent
#: literals: (confirmatory, alpha, draw, partition seed, init seed, client runs, outcome,
#: RMSE, RMSE CI low, high, skill, skill CI low, high, spread over 28 runs, saved spread).
SWEEP_TABLE = (
    (True, 0.05, 1, 4601, 4601, "37/36/39", "trained", 0.0377, 0.0199, 0.0551, 0.134, -1.655, 0.402, 0.613, 1.005),
    (True, 0.1, 1, 4601, 4601, "20/17/75", "trained", 0.0389, 0.0175, 0.0603, 0.079, -0.106, 0.226, 0.175, 0.197),
    (True, 0.3, 1, 4601, 4601, "18/21/73", "trained", 0.0451, 0.0294, 0.0599, -0.240, -4.913, 0.476, 1.004, 1.772),
    (True, 1.0, 1, 4601, 4601, "38/39/35", "trained", 0.0390, 0.0211, 0.0548, 0.073, -2.278, 0.445, 0.678, 0.899),
    (True, 10.0, 1, 4601, 4601, "38/40/34", "trained", 0.0392, 0.0206, 0.0551, 0.063, -2.439, 0.452, 0.700, 1.307),
    (True, 0.05, 2, 5601, 4601, "34/55/23", "trained", 0.0402, 0.0248, 0.0549, 0.018, -1.580, 0.362, 0.606, 0.911),
    (True, 0.1, 2, 5601, 4601, "22/9/81", "collapsed", 0.0407, 0.0189, 0.0631, -0.008, -0.022, 0.003, 0.000, 0.000),
    (True, 0.3, 2, 5601, 4601, "37/37/38", "trained", 0.0370, 0.0218, 0.0512, 0.167, -1.777, 0.469, 0.636, 1.076),
    (True, 1.0, 2, 5601, 4601, "18/49/45", "trained", 0.0383, 0.0214, 0.0547, 0.108, -1.598, 0.390, 0.564, 1.073),
    (True, 10.0, 2, 5601, 4601, "41/36/35", "trained", 0.0420, 0.0251, 0.0590, -0.075, -2.135, 0.256, 0.683, 0.857),
    (True, 0.05, 3, 6601, 4601, "52/40/20", "trained", 0.0473, 0.0289, 0.0638, -0.364, -3.880, 0.231, 0.819, 0.990),
    (True, 0.1, 3, 6601, 4601, "13/48/51", "trained", 0.0371, 0.0222, 0.0506, 0.161, -2.556, 0.567, 0.764, 1.377),
    (True, 0.3, 3, 6601, 4601, "42/33/37", "trained", 0.0344, 0.0233, 0.0468, 0.279, -1.018, 0.499, 0.578, 0.773),
    (True, 1.0, 3, 6601, 4601, "46/33/33", "trained", 0.0386, 0.0201, 0.0580, 0.092, -0.406, 0.203, 0.216, 0.244),
    (True, 10.0, 3, 6601, 4601, "36/42/34", "trained", 0.0354, 0.0244, 0.0466, 0.239, -1.201, 0.519, 0.578, 0.927),
    (False, 0.1, 1, 4601, 4602, "20/17/75", "trained", 0.0383, 0.0230, 0.0544, 0.106, -1.546, 0.377, 0.738, 0.892),
    (False, 0.3, 1, 4601, 4602, "18/21/73", "trained", 0.0379, 0.0243, 0.0515, 0.125, -2.098, 0.457, 0.765, 1.091),
)
RMSE_TOL, THREE_DP_TOL = 6e-5, 6e-4  # literals printed to 4 and 3 decimals


def _assert_sweep_table(rows: list[dict[str, object]], confirmatory_of) -> None:
    assert len(rows) == len(SWEEP_TABLE) == 17
    keyed = {(confirmatory_of(r), r["alpha"], r["draw"], r["init_seed"]): r for r in rows}
    assert len(keyed) == 17
    for (conf, alpha, draw, partition, init, clients, outcome, rmse, low, high, skill, s_low, s_high, spread,
         saved) in SWEEP_TABLE:
        r = keyed[(conf, alpha, draw, init)]
        assert (r["partition_seed"], r["client_runs"], r["outcome"]) == (partition, clients, outcome), (alpha, draw, init)
        for key, want in (("value", rmse), ("ci_lower", low), ("ci_upper", high)):
            assert _close(r[key], want, RMSE_TOL), (alpha, draw, init, key)
        for key, want in (("skill", skill), ("skill_ci_lower", s_low), ("skill_ci_upper", s_high),
                          ("spread_runs", spread), ("spread_saved_window", saved)):
            assert _close(r[key], want, THREE_DP_TOL), (alpha, draw, init, key)


@pytest.fixture(scope="module")
def sweep_built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from redo_by_sara.alpha_sweep import make_workbook

    return make_workbook(ARTIFACTS, tmp_path_factory.mktemp("sweep") / "alpha_sweep.xlsx", ROOT)


def _stats(path: Path) -> list[tuple]:
    return [row for row in load_workbook(path)["stats"].iter_rows(values_only=True)]


@requires_sweep
def test_t6_matches_the_17_row_table(built: Path) -> None:
    rows = _rows(built / "tables.xlsx", "T6_alpha_sweep")
    _assert_sweep_table(rows, lambda r: r["confirmatory"] == "yes")
    assert {r["confirmatory"] for r in rows} == {"yes", "no (init-seed control)"}
    ws = load_workbook(built / "tables.xlsx")["T6_alpha_sweep"]
    header = [c.value for c in ws[1]]
    first = header.index("summary_alpha")
    summary = {row[first]: dict(zip(header[first:], row[first:]))
               for row in ws.iter_rows(min_row=2, max_row=6, values_only=True)}
    assert list(summary) == [0.05, 0.1, 0.3, 1.0, 10.0]
    outcome = ws.cell(row=1, column=header.index("outcome") + 1).column_letter
    for alpha, cells in summary.items():
        # means and stds over every confirmatory row of that α: the outcome column is not a criterion
        for key in ("summary_n", "summary_rmse_mean", "summary_rmse_std", "summary_skill_mean", "summary_skill_std"):
            assert f"${outcome}$" not in cells[key], (alpha, key)
        assert f"${outcome}$" in cells["summary_collapsed"] and '"collapsed"' in cells["summary_collapsed"]
    alpha01 = [r for r in rows if r["confirmatory"] == "yes" and r["alpha"] == 0.1]
    assert _close(_mean_std([r["value"] for r in alpha01])[0], 0.0389, RMSE_TOL)
    assert _close(_mean_std([r["value"] for r in alpha01])[1], 0.0018, RMSE_TOL)
    assert [r["outcome"] for r in alpha01].count("collapsed") == 1


@requires_sweep
def test_alpha_sweep_workbook_matches_the_17_row_table(sweep_built: Path) -> None:
    rows = _rows(sweep_built)
    _assert_sweep_table(rows, lambda r: r["role"] == "confirmatory")
    _assert_chart_refs(sweep_built)
    wb = load_workbook(sweep_built)
    assert wb.calculation.fullCalcOnLoad
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None:
                    assert cell.font.name == "Arial", (ws.title, cell.coordinate)


@requires_sweep
def test_alpha_sweep_collapsed_run_stays_in_every_statistic(sweep_built: Path) -> None:
    rows = _rows(sweep_built)
    # display: the collapsed run is the last confirmatory row, still inside every confirmatory range
    roles = [r["role"] for r in rows]
    assert roles == ["confirmatory"] * 15 + ["init_control"] * 2
    assert rows[14]["outcome"] == "collapsed" and (rows[14]["alpha"], rows[14]["draw"]) == (0.1, 2)
    ws = load_workbook(sweep_built)["data"]
    header = [c.value for c in ws[1]]
    letter = {h: ws.cell(row=1, column=i + 1).column_letter for i, h in enumerate(header) if h}
    for rank in ("rank_alpha", "rank_rmse", "rank_spread"):
        assert "$2:$" in ws[f"{letter[rank]}16"].value and "$16<" in ws[f"{letter[rank]}16"].value, rank
    # the per-α formulas filter on role and α only, over every data row
    a = header.index("summary_alpha") + 1
    for r in range(2, 7):
        for c in range(a + 1, a + 6):
            formula = ws.cell(row=r, column=c).value
            assert f"${letter['outcome']}$" not in formula and "collapsed" not in formula
            assert f"${letter['role']}$2:${letter['role']}$18" in formula
    alpha01 = [r for r in rows if r["role"] == "confirmatory" and r["alpha"] == 0.1]
    assert len(alpha01) == 3
    mean, std = _mean_std([r["value"] for r in alpha01])
    assert _close(mean, 0.0389, RMSE_TOL) and _close(std, 0.0018, RMSE_TOL)
    mean, std = _mean_std([r["skill"] for r in alpha01])
    assert _close(mean, 0.077, THREE_DP_TOL) and _close(std, 0.085, THREE_DP_TOL)
    # the stats sheet: pre-registered α = 0.1 includes the collapsed run ...
    stats = _stats(sweep_built)
    labels = [str(row[0] or "") for row in stats]
    rmse = stats[labels.index("per α, every confirmatory run: α=0.1 RMSE mean / std (n = 3)")]
    skill = stats[labels.index("per α, every confirmatory run: α=0.1 skill mean / std (n = 3)")]
    assert _close(rmse[1], 0.0389, RMSE_TOL) and _close(rmse[2], 0.0018, RMSE_TOL)
    assert _close(skill[1], 0.077, THREE_DP_TOL) and _close(skill[2], 0.085, THREE_DP_TOL)
    # ... and the without-collapse value appears only inside the labelled sensitivity block
    start = next(i for i, label in enumerate(labels) if label.startswith("SENSITIVITY, NOT PRE-REGISTERED"))
    end = next(i for i in range(start + 1, len(labels)) if not labels[i])
    hits = [i for i, row in enumerate(stats) if isinstance(row[1], float) and _close(row[1], 0.0380, RMSE_TOL)
            and isinstance(row[2], float) and _close(row[2], 0.0012, RMSE_TOL)]
    assert hits and all(start < i < end for i in hits), (hits, start, end)


@requires_sweep
def test_alpha_sweep_charts(sweep_built: Path) -> None:
    with zipfile.ZipFile(sweep_built) as z:
        rmse, skill, spread = (_text_content(z.read(f"xl/charts/chart{i}.xml").decode()) for i in (1, 2, 3))
        raw = [z.read(f"xl/charts/chart{i}.xml").decode() for i in (1, 2)]
    for chart in (rmse, skill):
        for title in ("draw 1 (partition seed 4601)", "draw 2 (partition seed 5601)", "draw 3 (partition seed 6601)",
                      "collapsed (α=0.1 draw 2), in every statistic",
                      "init-seed 4602 controls (same partitions as draw 1)",
                      "mean ± std over draws (all 3, collapsed included)", "centralized residual (mean of 3 seeds)",
                      "centralized min–max over 3 seeds"):
            assert title in chart, title
    assert "lookup (0.0405 m/s)" in rmse and "zero" in skill
    assert all('<c:logBase val="10"' in x or '<logBase val="10"' in x for x in raw)
    assert "RMSE vs prediction spread" in spread and "init-seed 4602 controls" in spread
    # the centralized reference: 3 seeds, mean RMSE 0.0385 and skill +0.098 (CHANGES.md)
    ws = load_workbook(sweep_built)["data"]
    block = [row for row in ws.iter_rows(values_only=True) if "centralized_seed" in row]
    assert len(block) == 1
    header_row = next(i for i, row in enumerate(ws.iter_rows(values_only=True), start=1) if "centralized_seed" in row)
    col = list(block[0]).index("centralized_seed")
    seeds = [ws.cell(row=header_row + k, column=col + 1).value for k in (1, 2, 3)]
    rmses = [ws.cell(row=header_row + k, column=col + 2).value for k in (1, 2, 3)]
    skills = [ws.cell(row=header_row + k, column=col + 3).value for k in (1, 2, 3)]
    assert seeds == [4601, 4602, 4603]
    assert _close(np.mean(rmses), 0.0385, RMSE_TOL) and _close(np.mean(skills), 0.098, THREE_DP_TOL)


@requires_sweep
def test_alpha_sweep_workbook_mentions_neither_linked_nor_names(sweep_built: Path) -> None:
    assert _hits([sweep_built], re.compile("linked", re.IGNORECASE)) == []
    if NOTES_XLSX.exists():
        assert _name_hits([sweep_built]) == []


@requires_results
def test_fig6_subject_skew(built: Path) -> None:
    rows = _rows(built / "fig6_subject_skew.xlsx")
    plotted = [r for r in rows if r["plotted"] == "yes"]
    assert all(r["seed"] == 4601 for r in plotted)
    fl = {r["condition"]: r for r in plotted if r["family"] == "fl"}
    for condition, correct, low, high in (("k3_iid", 28, 0.879, 1.000), ("k3_dirichlet0.3", 25, 0.728, 0.963),
                                          ("k3_dirichlet0.1", 22, 0.605, 0.898), ("k2_natural", 28, 0.879, 1.000)):
        r = fl[condition]
        assert (r["correct"], r["runs"]) == (correct, 28), condition
        assert _close(r["ci_lower"], low) and _close(r["ci_upper"], high), condition
    for condition, mean, std, n in (("k3_iid", 0.988, 0.021, 3), ("k3_dirichlet0.3", 0.607, 0.270, 3),
                                    ("k3_dirichlet0.1", 0.452, 0.297, 3), ("k2_natural", 0.500, 0.051, 2)):
        values = [r["value"] for r in plotted if r["condition"] == condition and r["family"] == "local_only"]
        assert len(values) == n, condition
        m, s = _mean_std(values)  # sample std (n - 1)
        assert _close(m, mean) and _close(s, std), condition
    assert [r["correct"] for r in rows if r["family"] == "centralized"] == [28, 28, 28]
    ws = load_workbook(built / "fig6_subject_skew.xlsx")["data"]
    header = [c.value for c in ws[1]]
    std_formula = ws.cell(row=2, column=header.index("local_std") + 1).value
    assert std_formula.startswith("=SQRT(SUMPRODUCT(") and std_formula.endswith("-1))")
    assert [ws.cell(row=r, column=header.index("summary_level") + 1).value for r in range(2, 6)] == [
        "IID (K=3)", "Dirichlet α=0.3 (K=3)", "Dirichlet α=0.1 (K=3)", "Natural split (K=2, disjoint subjects)"]
    notes = " ".join(str(c.value) for row in load_workbook(built / "fig6_subject_skew.xlsx")["notes"].iter_rows()
                     for c in row if c.value)
    for text in ("Plotted at init seed 4601", "FL counts are identical across seeds",
                 "Dirichlet α=0.3 (K=3): local-only client_2 at seed 4602 has 24/28 correct, against 25/28",
                 "sample standard deviation (n − 1)", "not part of the α ladder", "recording campaign"):
        assert text in notes, text


@requires_results
def test_fig2_regression_notes_point_to_the_alpha_sweep(built: Path) -> None:
    for name in ("fig2_regression_skill.xlsx", "fig2_regression_rmse.xlsx"):
        notes = " ".join(str(c.value) for row in load_workbook(built / name)["notes"].iter_rows() for c in row if c.value)
        assert "Dirichlet points are partition draw 1; see alpha_sweep.xlsx for 3 draws per α and 5 α values." in notes
        assert "α=0.1 RMSE 0.0389 m/s" in notes and "α=0.3 RMSE 0.0451 m/s" in notes, name
