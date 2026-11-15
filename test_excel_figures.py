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
    "confusion_matrices.xlsx",
    "tables.xlsx",
)
TOL = 5e-4  # CHANGES.md prints 3 decimals (4 for RMSE)


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
                    "RMSE"} <= set(wb.sheetnames)
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


@requires_results
def test_charts_reference_cells_of_the_workbook(built: Path) -> None:
    for name in WORKBOOKS:
        wb = load_workbook(built / name)
        with zipfile.ZipFile(built / name) as z:
            for part in (p for p in z.namelist() if p.startswith("xl/charts/")):
                refs = re.findall(r"<f>(.*?)</f>", z.read(part).decode())
                assert refs, (name, part)
                for ref in refs:
                    sheet, cells = ref.rsplit("!", 1)
                    assert sheet.strip("'") in wb.sheetnames, (name, part, ref)
                    assert re.fullmatch(r"\$[A-Z]+\$\d+(:\$[A-Z]+\$\d+)?", cells), (name, part, ref)


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
