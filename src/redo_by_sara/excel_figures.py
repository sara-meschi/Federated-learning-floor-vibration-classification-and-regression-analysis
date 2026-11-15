"""Excel workbooks of the paper figures, with native, editable Excel charts.

Everything here is built from saved results; nothing trains, and nothing is written
under ``artifacts/``. The data is the relabel record (9 classes: 8 (source, subject)
identities + ``no_walking``) plus the S5 results the relabel kept, exactly as
``relabel/derived/s5_disposition.csv`` records them:

- ``kept`` S5 jobs (regression, the source probe) are read from ``s5/``;
- ``rerun`` jobs and the new Dirichlet alpha = 0.3 jobs are read from ``relabel/``;
- the re-scored local-only natural K=2 classification job reads its run-level numbers
  from ``relabel/rescore/local_only_k2_natural/`` (its partitions from ``s5/``).

The S5 cross-session control, its re-score and every per-identity breakdown across the
two campaigns are not read. Records come from the ``paper_figures`` loaders, through a
``Results`` subclass that routes each job to its folder, so the numbers are the ones the
paper tables are built from.

Every workbook has a ``data`` sheet (one row per plotted point; CIs in ``ci_lower`` /
``ci_upper`` with formula columns ``err_plus`` / ``err_minus`` that feed the error bars),
a ``chart`` sheet (openpyxl charts that reference the data cells) and a ``notes`` sheet.
Arial throughout; ``fullCalcOnLoad`` makes Excel compute the formulas on open.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.chart.axis import ChartLines
from openpyxl.chart.data_source import (
    AxDataSource,
    MultiLevelStrRef,
    NumDataSource,
    NumRef,
    StrRef,
)
from openpyxl.chart.error_bar import ErrorBars
from openpyxl.chart.label import DataLabel, DataLabelList
from openpyxl.chart.legend import Legend, LegendEntry
from openpyxl.chart.marker import DataPoint, Marker
from openpyxl.chart.series import SeriesLabel
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.text import RichText, Text
from openpyxl.chart.title import Title
from openpyxl.drawing.line import LineProperties
from openpyxl.drawing.text import (
    CharacterProperties,
    Font as DrawingFont,
    Paragraph,
    ParagraphProperties,
    RegularTextRun,
)
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter, quote_sheetname
from openpyxl.workbook.properties import CalcProperties
from openpyxl.worksheet.worksheet import Worksheet

from . import paper_figures as pf
from .regression_metrics import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, report_from_run_rows

PathLike = str | Path

# --------------------------------------------------------------------------------------
# what is shown
# --------------------------------------------------------------------------------------

#: Display order and plain-text labels of the partition conditions (Fig 2 order).
CONDITION_LABELS = {
    "k2_iid": "K=2 IID",
    "k2_natural": "K=2 natural",
    "k3_iid": "K=3 ρ=1.0 (IID)",
    "k3_rho0.375": "K=3 ρ=0.375",
    "k3_rho0": "K=3 ρ=0",
    "k3_dirichlet0.3": "K=3 Dirichlet α=0.3",
    "k3_dirichlet0.1": "K=3 Dirichlet α=0.1",
}
FAMILY_LABELS = {"centralized": "Centralized", "fl": "FL (FedAvg)", "local_only": "Local-only (per client)"}
SOURCE_LABELS = {"test_2": "Test_2", "testing_20251124": "20251124"}
OUTCOMES = ("trained", "did not train", "collapsed")

#: The one S5 job the relabel re-scored that is not read here (see the module docstring).
EXCLUDED_S5_JOBS = {"subject003_control/seed4601"}
RESCORED_LOCAL_NATURAL = "local_only/k2_natural/seed4601/classification"

# --------------------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------------------

FONT = "Arial"
INK = "222222"
MUTED = "6B6B6B"
GRID = "D9D9D9"
FL_COLOR = "0072B2"
LOCAL_COLOR = "D55E00"
CENTRAL_COLOR = INK
COLLAPSED_COLOR = "CC79A7"
#: Family encoding (as the paper): colour + marker + fill.
FAMILY_MARKER = {
    "centralized": ("diamond", CENTRAL_COLOR, True),
    "fl": ("circle", FL_COLOR, True),
    "local_only": ("square", LOCAL_COLOR, False),
}
#: Regression outcomes other than trained override the marker (as the paper).
OUTCOME_SYMBOL = {"did not train": "triangle", "collapsed": "x"}
#: Identity colours: the paper's seven, plus olive (#999933, from Tol's palette) for the
#: eighth identity. Test_2 identities are filled, 20251124 identities hollow.
IDENTITY_COLORS = ("0072B2", "D55E00", "009E73", "882255", "E69F00", "56B4E9", "CC79A7", "999933")
IDENTITY_SYMBOLS = ("circle", "square", "triangle", "diamond", "circle", "square", "triangle", "diamond")
SEED_COLORS = ("0072B2", "009E73", "E69F00")

CHART_WIDTH_CM = 17.0
CHART_HEIGHT_CM = 9.0
FIG2_SLOT_CM = 0.55

STYLING_NOTE = (
    "Per-point marker styles and outcome data labels are set when the workbook is built "
    "from the family and outcome columns; values and error bars follow the data cells live. "
    "To restyle one point in Excel: click the series, click the point again to select only "
    "it, then right-click → Format Data Point (marker, fill, line) or Format Data Label."
)

# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


@dataclass
class RelabelResults(pf.Results):
    """``paper_figures.Results`` over the relabel record plus the kept S5 jobs.

    ``origin`` maps every job ID to the results folder it is read from (``relabel`` or
    ``s5``); ``job_dir`` routes by it. ``conditions`` adds Dirichlet alpha = 0.3."""

    origin: dict[str, str] = field(default_factory=dict)

    def job_dir(self, row: Mapping[str, str]) -> Path:
        return self.root / self.origin[row["job_id"]] / row["job_id"]

    def conditions(self, family: str) -> list[str]:
        present = {row["condition"] for row in self.index if row["family"] == family}
        return [c for c in CONDITION_LABELS if c in present]

    def relative(self, path: PathLike) -> str:
        """A path as written in the notes: relative to the repository root."""
        return Path(path).resolve().relative_to(self.root.resolve().parent).as_posix()


def _rescored_run_level(root: Path) -> list[dict[str, str]]:
    """Run-level rows (``derived/run_level_classification.csv`` schema) for the
    re-scored local-only natural K=2 job, one per client, majority vote."""
    folder = root / "relabel" / "rescore" / "local_only_k2_natural"
    rows = []
    for client_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        vote = pf._read_json(client_dir / "run_level_metrics.json")["majority_vote"]
        per_source = vote["per_source"]
        rows.append({
            "job_id": RESCORED_LOCAL_NATURAL, "family": "local_only", "condition": "k2_natural",
            "seed": "4601", "client": client_dir.name, "method": "majority_vote",
            "accuracy": repr(vote["accuracy"]),
            "accuracy_test_2": repr(per_source["test_2"]["accuracy"]),
            "accuracy_testing_20251124": repr(per_source["testing_20251124"]["accuracy"]),
            "macro_f1": repr(vote["macro_f1"]), "num_correct": str(vote["num_correct"]),
            "num_runs": str(vote["num_runs"]),
            "wilson95_low": repr(vote["wilson95"][0]), "wilson95_high": repr(vote["wilson95"][1]),
        })
    return rows


def load_relabel_results(artifacts_root: PathLike) -> RelabelResults:
    root = Path(artifacts_root)
    relabel_index = pf._read_csv(root / "relabel" / "results_index.csv")
    s5_index = {row["job_id"]: row for row in pf._read_csv(root / "s5" / "results_index.csv")}
    disposition = pf._read_csv(root / "relabel" / "derived" / "s5_disposition.csv")
    relabel_ids = {row["job_id"] for row in relabel_index}

    index = list(relabel_index)
    origin = {job_id: "relabel" for job_id in relabel_ids}
    for row in disposition:
        job_id, what = row["s5_job_id"], row["disposition"]
        if what == "kept":
            if job_id in relabel_ids:
                raise AssertionError(f"{job_id}: kept from S5 but also in the relabel index")
            index.append(s5_index[job_id])
            origin[job_id] = "s5"
        elif what == "rerun":
            if job_id not in relabel_ids:
                raise AssertionError(f"{job_id}: marked rerun but missing from the relabel index")
        elif what == "re-scored":
            if job_id == RESCORED_LOCAL_NATURAL:
                index.append(s5_index[job_id])
                origin[job_id] = "s5"
            elif job_id not in EXCLUDED_S5_JOBS:
                raise AssertionError(f"{job_id}: re-scored job with no reader here")
        else:
            raise AssertionError(f"{job_id}: unknown disposition {what!r}")
    bad = [(r["job_id"], r["status"]) for r in index if r["status"] not in pf.TERMINAL_STATUSES]
    if bad:
        raise AssertionError(f"Jobs not in a terminal state: {bad}")
    unknown = {r["condition"] for r in index if r["family"] in ("fl", "local_only")} - set(CONDITION_LABELS)
    if unknown:
        raise AssertionError(f"Conditions without a display label: {sorted(unknown)}")
    return RelabelResults(
        root=root,
        index=index,
        run_level=pf._read_csv(root / "relabel" / "derived" / "run_level_classification.csv")
        + _rescored_run_level(root),
        seed_aggregates=pf._read_csv(root / "relabel" / "derived" / "seed_aggregates.csv"),
        origin=origin,
    )


# --------------------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------------------


def _model_label(record: Mapping[str, Any]) -> str:
    if record["client"]:
        return f"{record['client'].replace('_', ' ')}, seed {record['seed']}"
    return f"{'FL' if record['family'] == 'fl' else 'centralized'}, seed {record['seed']}"


def _classification_source(results: RelabelResults, record: Mapping[str, Any]) -> str:
    if record["family"] == "local_only" and record["condition"] == "k2_natural":
        return f"artifacts/relabel/rescore/local_only_k2_natural/{record['client']}/run_level_metrics.json"
    return "artifacts/relabel/derived/run_level_classification.csv"


def classification_points(results: RelabelResults) -> list[dict[str, Any]]:
    """Every classification model in Fig 2 order: centralized, then per condition FL and
    local-only, each sorted by seed and client."""
    groups = [("centralized", pf.classification_records(results, "centralized", "classification"))]
    for condition in results.conditions("fl"):
        groups.append((condition, pf.classification_records(results, "fl", condition)
                       + pf.classification_records(results, "local_only", condition)))
    out = []
    for condition, records in groups:
        for family in ("centralized", "fl", "local_only"):
            for r in sorted((r for r in records if r["family"] == family), key=pf._seed_sort_key):
                out.append({**r, "condition": condition, "model": _model_label(r),
                            "source_file": _classification_source(results, r)})
    return out


def _run_rows(record: Mapping[str, Any]) -> list[dict[str, str]]:
    return pf._read_csv(Path(record["result_dir"]) / "test_run_predictions.csv")


def matched_bootstrap(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    """Percentile 95% intervals of skill and RMSE from one set of run resamples: the same
    generator, seed and draws as ``regression_metrics.bootstrap_skill_ci``."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = actual.size
    indices = rng.integers(0, n, size=(int(BOOTSTRAP_RESAMPLES), n))
    sse_model = np.sum((predicted[indices] - actual[indices]) ** 2, axis=1)
    sse_base = np.sum((baseline[indices] - actual[indices]) ** 2, axis=1)
    valid = sse_base > 0.0
    skill = 1.0 - sse_model[valid] / sse_base[valid]
    rmse = np.sqrt(sse_model / n)
    skill_low, skill_high = np.percentile(skill, [2.5, 97.5])
    rmse_low, rmse_high = np.percentile(rmse, [2.5, 97.5])
    return {"skill_ci_low": float(skill_low), "skill_ci_high": float(skill_high),
            "rmse_ci_low": float(rmse_low), "rmse_ci_high": float(rmse_high)}


def exact_regression(results: RelabelResults, record: Mapping[str, Any]) -> dict[str, Any]:
    """Exact RMSE, lookup RMSE and the matched RMSE interval from the saved run
    predictions, each checked against the record it extends."""
    rows = _run_rows(record)
    overall = report_from_run_rows(rows, "source_subject_train_mean_mps")["overall"]
    actual = np.asarray([float(r["actual_speed_mps"]) for r in rows])
    predicted = np.asarray([float(r["predicted_speed_mps"]) for r in rows])
    baseline = np.asarray([float(r["source_subject_train_mean_mps"]) for r in rows])
    boot = matched_bootstrap(actual, predicted, baseline)
    checks = (
        ("RMSE", overall["model_rmse_mps"], record["rmse"]),
        ("lookup RMSE", overall["baseline_rmse_mps"], record["baseline_rmse"]),
        ("skill", overall["skill"], record["skill"]),
        ("skill CI low", overall["skill_ci95"][0], record["ci_low"]),
        ("skill CI high", overall["skill_ci95"][1], record["ci_high"]),
        ("skill CI low, matched draws", boot["skill_ci_low"], record["ci_low"]),
        ("skill CI high, matched draws", boot["skill_ci_high"], record["ci_high"]),
    )
    for what, recomputed, saved in checks:
        if not math.isclose(recomputed, saved, rel_tol=0, abs_tol=1e-9):
            raise AssertionError(f"{record['result_dir']}: {what} {recomputed} != saved {saved}")
    return {"rmse": overall["model_rmse_mps"], "lookup_rmse": overall["baseline_rmse_mps"],
            "rmse_ci_low": boot["rmse_ci_low"], "rmse_ci_high": boot["rmse_ci_high"],
            "predictions_file": results.relative(Path(record["result_dir"]) / "test_run_predictions.csv")}


def regression_points(results: RelabelResults) -> list[dict[str, Any]]:
    """Every regression model in Fig 2 order (centralized = the residual model), with the
    exact RMSE from its predictions."""
    groups = [("centralized", pf.regression_records(results, "centralized", "residual"))]
    for condition in results.conditions("fl"):
        groups.append((condition, pf.regression_records(results, "fl", condition)
                       + pf.regression_records(results, "local_only", condition)))
    out = []
    for condition, records in groups:
        for family in ("centralized", "fl", "local_only"):
            for r in sorted((r for r in records if r["family"] == family), key=pf._seed_sort_key):
                out.append({**r, **exact_regression(results, r), "condition": condition,
                            "model": _model_label(r)})
    return out


# --------------------------------------------------------------------------------------
# sheet writing
# --------------------------------------------------------------------------------------


@dataclass
class Table:
    """Where a table landed on its sheet; builds absolute range strings for charts."""

    ws: Worksheet
    columns: list[str]
    header_row: int
    n: int
    first_col: int = 1

    def letter(self, key: str) -> str:
        return get_column_letter(self.first_col + self.columns.index(key))

    def ref(self, key: str, start: int = 0, stop: int | None = None) -> str:
        """``'sheet'!$C$2:$C$9`` over data rows ``start .. stop - 1`` (0-based)."""
        stop = self.n if stop is None else stop
        col = self.letter(key)
        top, bottom = self.header_row + 1 + start, self.header_row + stop
        return f"{quote_sheetname(self.ws.title)}!${col}${top}:${col}${bottom}"

    def span(self, first: str, last: str, start: int = 0, stop: int | None = None) -> str:
        stop = self.n if stop is None else stop
        top, bottom = self.header_row + 1 + start, self.header_row + stop
        return (f"{quote_sheetname(self.ws.title)}!${self.letter(first)}${top}:"
                f"${self.letter(last)}${bottom}")

    def header(self, key: str) -> str:
        return f"{quote_sheetname(self.ws.title)}!${self.letter(key)}${self.header_row}"

    def cell(self, key: str, i: int, absolute: bool = True) -> str:
        col, row = self.letter(key), self.header_row + 1 + i
        return f"${col}${row}" if absolute else f"{col}{row}"

    def rows_where(self, rows: Sequence[Mapping[str, Any]], **match: Any) -> tuple[int, int]:
        """The contiguous block of data rows matching ``match`` (asserted contiguous)."""
        hits = [i for i, r in enumerate(rows) if all(r.get(k) == v for k, v in match.items())]
        if not hits or hits != list(range(hits[0], hits[-1] + 1)):
            raise AssertionError(f"rows matching {match} are not one contiguous block")
        return hits[0], hits[-1] + 1


#: Error-bar formula columns, written wherever a table has ``value`` and a CI.
ERROR_FORMULAS = {"err_plus": "={ci_upper}-{value}", "err_minus": "={value}-{ci_lower}"}


def _font(bold: bool = False, size: float = 10, italic: bool = False) -> Font:
    return Font(name=FONT, bold=bold, size=size, italic=italic)


def write_table(ws: Worksheet, rows: Sequence[Mapping[str, Any]], columns: Sequence[str], *,
                header_row: int = 1, first_col: int = 1,
                formulas: Mapping[str, str] | None = None) -> Table:
    """Write ``rows`` under a bold header. ``formulas`` maps a column to a template whose
    ``{key}`` placeholders become that row's cell of column ``key``."""
    columns = list(columns)
    formulas = dict(formulas or {})
    if {"value", "ci_lower", "ci_upper"} <= set(columns):
        for key, template in ERROR_FORMULAS.items():
            if key not in columns:
                columns.insert(columns.index("ci_upper") + 1 + list(ERROR_FORMULAS).index(key), key)
            formulas.setdefault(key, template)
    table = Table(ws, columns, header_row, len(rows), first_col)
    for j, key in enumerate(columns):
        cell = ws.cell(row=header_row, column=first_col + j, value=key)
        cell.font = _font(bold=True)
        ws.column_dimensions[get_column_letter(first_col + j)].width = max(10, min(44, len(key) + 3))
    for i, row in enumerate(rows):
        cells = {key: table.cell(key, i, absolute=False) for key in columns}
        for j, key in enumerate(columns):
            if key in formulas:
                value: Any = formulas[key].format_map(cells)
            else:
                value = row.get(key)
                if isinstance(value, np.generic):
                    value = value.item()
            if value is None or value == "":
                continue
            cell = ws.cell(row=header_row + 1 + i, column=first_col + j, value=value)
            cell.font = _font()
            if isinstance(value, float):
                cell.number_format = "0.0000"
    if header_row == 1 and first_col == 1:
        ws.freeze_panes = "A2"
    return table


def _new_workbook() -> Workbook:
    wb = Workbook()
    wb.calculation = CalcProperties(fullCalcOnLoad=True)
    # The default cell font (style 0) is Arial too, so cells typed in later match.
    wb._fonts[0] = Font(name=FONT, size=10)  # noqa: SLF001 — openpyxl has no public setter
    wb.active.title = "data"
    return wb


def _sheet(wb: Workbook, title: str) -> Worksheet:
    return wb.create_sheet(title)


# --------------------------------------------------------------------------------------
# chart helpers
# --------------------------------------------------------------------------------------


def _char(size: float, bold: bool = False, color: str = INK) -> CharacterProperties:
    return CharacterProperties(latin=DrawingFont(typeface=FONT), sz=int(size * 100), b=bold, solidFill=color)


def _txpr(size: float = 8, color: str = INK) -> RichText:
    cp = _char(size, color=color)
    return RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=cp), endParaRPr=cp)])


def _title(text: str, size: float = 10, bold: bool = True) -> Title:
    cp = _char(size, bold)
    paragraph = Paragraph(pPr=ParagraphProperties(defRPr=cp), r=[RegularTextRun(rPr=cp, t=text)])
    return Title(tx=Text(rich=RichText(p=[paragraph])), overlay=False)


def _line(color: str | None, width_pt: float = 1.0, dash: str | None = None) -> GraphicalProperties:
    if color is None:
        return GraphicalProperties(ln=LineProperties(noFill=True))
    return GraphicalProperties(ln=LineProperties(solidFill=color, w=int(width_pt * 12700), prstDash=dash))


def _marker(symbol: str, color: str, filled: bool, size: int = 7) -> Marker:
    fill = color if filled else "FFFFFF"
    if symbol in ("x", "plus", "star", "dash", "none"):
        spPr = GraphicalProperties(ln=LineProperties(solidFill=color, w=15875))
    else:
        spPr = GraphicalProperties(solidFill=fill, ln=LineProperties(solidFill=color, w=12700))
    return Marker(symbol=symbol, size=size, spPr=spPr)


def _point_marker(family: str, outcome: str | None) -> Marker:
    symbol, color, filled = FAMILY_MARKER[family]
    if outcome in OUTCOME_SYMBOL:
        return _marker(OUTCOME_SYMBOL[outcome], color, False, 9 if outcome == "collapsed" else 8)
    return _marker(symbol, color, filled)


def _custom_error_bars(table: Table, start: int = 0, stop: int | None = None, *, direction: str | None = None,
                       plus: str = "err_plus", minus: str = "err_minus", color: str = MUTED) -> ErrorBars:
    return ErrorBars(
        errDir=direction, errBarType="both", errValType="cust", noEndCap=False,
        plus=NumDataSource(numRef=NumRef(f=table.ref(plus, start, stop))),
        minus=NumDataSource(numRef=NumRef(f=table.ref(minus, start, stop))),
        spPr=_line(color, 0.75),
    )


def _outcome_labels(indices_outcomes: Sequence[tuple[int, str]], size: float = 7,
                    position: str = "r") -> DataLabelList:
    """Per-point data labels whose text is the outcome: a number format made only of the
    quoted label (all three sections, so negative values don't gain a minus sign)."""
    labels = []
    for idx, outcome in indices_outcomes:
        text = '"' + outcome + '"'
        labels.append(DataLabel(
            idx=idx, numFmt=";".join([text] * 3), txPr=_txpr(size), dLblPos=position,
            showLegendKey=False, showVal=True, showCatName=False, showSerName=False,
            showPercent=False, showBubbleSize=False))
    return DataLabelList(dLbl=labels, showLegendKey=False, showVal=False, showCatName=False,
                         showSerName=False, showPercent=False, showBubbleSize=False)


def _style_axis(axis: Any, title: str | None, *, number_format: str | None = None,
                minimum: float | None = None, maximum: float | None = None, log: bool = False,
                gridlines: bool = True, label_position: str | None = None,
                reverse: bool = False, major_unit: float | None = None) -> None:
    axis.delete = False
    axis.title = _title(title, 9, bold=False) if title else None
    axis.txPr = _txpr(8)
    if number_format:
        axis.number_format = number_format
    axis.scaling.min = minimum
    axis.scaling.max = maximum
    if log:
        axis.scaling.logBase = 10
    if reverse:
        axis.scaling.orientation = "maxMin"
    if major_unit is not None:
        axis.majorUnit = major_unit
    axis.majorGridlines = ChartLines(spPr=_line(GRID, 0.5)) if gridlines else None
    if label_position:
        axis.tickLblPos = label_position


def _place_axes(chart: Any) -> None:
    chart.x_axis.axPos = "b"
    chart.y_axis.axPos = "l"


def _legend(chart: Any, deleted: Sequence[int] = (), position: str = "b") -> None:
    chart.legend = Legend(legendPos=position, overlay=False, txPr=_txpr(8),
                          legendEntry=[LegendEntry(idx=i, delete=True) for i in deleted])


def _size(chart: Any, width_cm: float = CHART_WIDTH_CM, height_cm: float = CHART_HEIGHT_CM) -> None:
    chart.width, chart.height = width_cm, height_cm


def _series(values: str, xvalues: str | None = None, title: str | None = None,
            title_ref: str | None = None) -> Series:
    s = Series(Reference(range_string=values), Reference(range_string=xvalues) if xvalues else None)
    if title_ref:
        s.tx = SeriesLabel(strRef=StrRef(f=title_ref))
    elif title is not None:
        s.tx = SeriesLabel(v=title)
    return s


def _scatter(title: str, x_title: str, y_title: str) -> ScatterChart:
    chart = ScatterChart(scatterStyle="lineMarker")
    chart.title = _title(title)
    chart.style = 2
    _style_axis(chart.x_axis, x_title, gridlines=False)
    _style_axis(chart.y_axis, y_title)
    _place_axes(chart)
    _size(chart)
    return chart


def _line_only(s: Series, color: str, width_pt: float = 1.25, dash: str | None = None) -> Series:
    s.marker = Marker(symbol="none")
    s.graphicalProperties = _line(color, width_pt, dash)
    s.smooth = False
    return s


def _markers_only(s: Series, marker: Marker) -> Series:
    s.marker = marker
    s.graphicalProperties = _line(None)
    s.smooth = False
    return s


# --------------------------------------------------------------------------------------
# notes
# --------------------------------------------------------------------------------------

METRIC_DEFINITIONS = (
    ("run accuracy", "Majority vote of a test run's window predictions (ties broken by the "
     "higher run-mean probability), as the fraction of the 28 held-out test runs whose vote is "
     "the run's identity class. 9 classes: 8 (source, subject) identities + no_walking."),
    ("Wilson 95% interval", "Wilson score interval for k correct out of n = 28 runs, z = 1.96."),
    ("skill", "1 − SSE_model / SSE_lookup over the 28 test runs (run-mean speed). 0 = the "
     "lookup; negative = worse than the lookup. Primary regression metric, always shown with "
     "the lookup."),
    ("lookup RMSE", "RMSE of the lookup baseline: each test run predicted by the mean speed of "
     "the training runs of its own (source, subject). 0.0405 m/s in every run."),
    ("bootstrap 95% interval", f"Percentile interval over {BOOTSTRAP_RESAMPLES:,} resamples of "
     f"the 28 test runs with replacement, generator seed {BOOTSTRAP_SEED}. Every model is "
     "resampled with the same draws."),
    ("outcome", "Regression trained gate (S5): collapsed = output spread std(out)/std(target) < "
     "1e-3; did not train = has spread but final standardized MSE ≥ 0.95 (global-train for FL, "
     "own data for a local-only client); trained = both pass."),
)

SEEDS_TEXT = ("Split seed 4601 (80/20 by whole run: 112 train / 28 test runs). Initialization "
              "seeds 4601 / 4602 / 4603 where a condition has three; otherwise 4601 only. Every "
              "model is evaluated on the same 28 test runs with all 9 channels.")


@dataclass
class Build:
    """What every notes sheet records about the build."""

    commit: str
    results: RelabelResults

    @cached_property
    def classification(self) -> list[dict[str, Any]]:
        return classification_points(self.results)

    @cached_property
    def regression(self) -> list[dict[str, Any]]:
        return regression_points(self.results)


def git_commit(repo: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"
    return head + (" (with uncommitted changes)" if dirty else "")


def write_notes(wb: Workbook, build: Build, title: str, description: str,
                columns: Mapping[str, str], sources: Sequence[str], extra: Sequence[str] = (),
                metrics: Sequence[str] | None = None, styled_points: bool = True) -> Worksheet:
    ws = _sheet(wb, "notes")
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 120
    line = 1

    def put(a: str, b: str = "", bold: bool = False) -> None:
        nonlocal line
        ws.cell(row=line, column=1, value=a).font = _font(bold=bold)
        if b:
            cell = ws.cell(row=line, column=2, value=b)
            cell.font = _font()
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        line += 1

    put(title, bold=True)
    put("what", description)
    put("built from commit", build.commit)
    put("formulas", "err_plus = ci_upper − value and err_minus = value − ci_lower (and any other "
        "formula cells) are computed by Excel when the file opens (fullCalcOnLoad). Edit a value "
        "or a CI and the error bars follow.")
    if styled_points:
        put("point styles", STYLING_NOTE)
    for text in extra:
        put("note", text)
    put("seeds", SEEDS_TEXT)
    line += 1
    put("Metric definitions", bold=True)
    for name, text in METRIC_DEFINITIONS:
        if metrics is None or name in metrics:
            put(name, text)
    line += 1
    put("Columns", bold=True)
    for name, text in columns.items():
        put(name, text)
    line += 1
    put("Source files (repository-relative)", bold=True)
    for source in sorted(set(sources)):
        put("", source)
    return ws


COMMON_COLUMNS = {
    "task": "classification or regression.",
    "condition": "Partition condition key (centralized, k2_iid, k2_natural, k3_iid = ρ 1.0, "
                 "k3_rho0.375, k3_rho0, k3_dirichlet0.3, k3_dirichlet0.1).",
    "condition_label": "Display label of the condition.",
    "axis_group": "Chart helper: the condition label on the first row of each condition, blank "
                  "otherwise (the outer level of the two-level category axis).",
    "model": "Inner category label: family member and initialization seed.",
    "family": "centralized, fl (FedAvg) or local_only (one client trained alone).",
    "client": "Local-only client (client_0, client_1, client_2); blank for centralized and FL.",
    "seed": "Initialization seed.",
    "outcome": "Regression trained-gate outcome: trained / did not train / collapsed.",
    "value": "The plotted value (see the chart's axis title).",
    "ci_lower": "Lower end of the 95% interval.",
    "ci_upper": "Upper end of the 95% interval.",
    "err_plus": "Formula: ci_upper − value (upper error bar length).",
    "err_minus": "Formula: value − ci_lower (lower error bar length).",
    "source_file": "File the row's numbers are read from.",
}


def _columns_doc(columns: Sequence[str], extra: Mapping[str, str]) -> dict[str, str]:
    merged = {**COMMON_COLUMNS, **extra}
    return {c: merged[c] for c in columns if c in merged} | {
        c: extra[c] for c in extra if c not in columns}


# --------------------------------------------------------------------------------------
# Fig 1: convergence
# --------------------------------------------------------------------------------------

CONVERGENCE_METRIC = {"classification": "global_train_accuracy", "regression": "global_train_standardized_mse"}
CONVERGENCE_PANELS = (("classification", "k2_iid"), ("classification", "k2_natural"),
                      ("regression", "k2_iid"), ("regression", "k2_natural"))


def convergence_curves(results: RelabelResults) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Per panel, one curve per seed (as ``paper_figures.convergence_data``)."""
    curves = {}
    for task, condition in CONVERGENCE_PANELS:
        seeds = []
        for row in results.rows("fl", condition, task):
            history = pf._history(results, row)
            seeds.append({"seed": int(row["seed"]), "label": pf._not_trained(row, task),
                          "round": [int(h["round"]) for h in history],
                          "value": [float(h[CONVERGENCE_METRIC[task]]) for h in history],
                          "source_file": results.relative(results.result_dir(row) / "global_train_history.csv")})
        curves[(task, condition)] = seeds
    return curves


def convergence_references(results: RelabelResults) -> list[dict[str, Any]]:
    """Reference values, one row each. The classification majority-class rate is read
    from the identity-labelled artifact (9 classes)."""
    refs: list[dict[str, Any]] = []
    for row in results.rows("centralized", "classification", "classification"):
        path = results.result_dir(row) / "training_summary.json"
        refs.append({"reference": f"classification: centralized final train accuracy, seed {row['seed']}",
                     "value": float(pf._read_json(path)["final_train_accuracy"]),
                     "source_file": results.relative(path)})
    artifact = results.root / "relabel" / "data" / "classification" / "artifact_summary.json"
    counts = {name: c["train"] for name, c in pf._read_json(artifact)["class_window_counts"].items()}
    refs.append({"reference": "classification: majority-class train rate (no_walking)",
                 "value": max(counts.values()) / sum(counts.values()),
                 "source_file": results.relative(artifact)})
    if max(counts, key=counts.get) != "no_walking":
        raise AssertionError("The majority training class is not no_walking.")
    refs.append({"reference": "regression: lookup (standardized MSE = 1 by definition)", "value": 1.0,
                 "source_file": "definition"})
    for r in pf.regression_records(results, "centralized", "residual"):
        path = Path(r["result_dir"]) / "training_summary.json"
        refs.append({"reference": f"regression: centralized 60 epochs final standardized MSE, seed {r['seed']}",
                     "value": r["gated_mse"], "source_file": results.relative(path)})
    path = results.root / pf.CENTRALIZED_RESIDUAL_300E_DIR / "training_summary.json"
    refs.append({"reference": "regression: centralized matched steps (300 epochs) final standardized MSE",
                 "value": float(pf._read_json(path)["degenerate_model_check"]["final_standardized_mse"]),
                 "source_file": results.relative(path)})
    return refs


def _seed_column(task: str, condition: str, curve: Mapping[str, Any]) -> str:
    prefix = f"{task} {CONDITION_LABELS[condition]}"
    if curve["label"]:
        return f"{prefix} {curve['label']} (seed {curve['seed']})"
    return f"{prefix} seed {curve['seed']}"


def build_fig1(build: Build) -> Workbook:
    results = build.results
    curves = convergence_curves(results)
    rounds = curves[CONVERGENCE_PANELS[0]][0]["round"]
    for seeds in curves.values():
        if any(s["round"] != rounds for s in seeds):
            raise AssertionError("Convergence curves have different round lists.")
    refs = convergence_references(results)

    wb = _new_workbook()
    ws = wb["data"]
    seed_columns: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for panel, seeds in curves.items():
        seed_columns[panel] = [(_seed_column(*panel, s), s) for s in seeds]
    names = [name for cols in seed_columns.values() for name, _ in cols]

    # reference block to the right of the curves; the line columns point at its cells
    ref_first_col = len(names) + 8
    ref_table = write_table(ws, refs, ["reference", "value", "source_file"], first_col=ref_first_col)
    n_cls_central = sum(r["reference"].startswith("classification: centralized") for r in refs)
    n_reg_central = sum(r["reference"].startswith("regression: centralized 60") for r in refs)
    mean_row = ref_table.header_row + ref_table.n + 2
    value_col = ref_table.letter("value")
    ref_name_col = ref_table.first_col
    cls_first = ref_table.header_row + 1
    reg_first = cls_first + n_cls_central + 2
    means = [
        ("classification: centralized final train accuracy, mean of seeds",
         f"=AVERAGE({value_col}{cls_first}:{value_col}{cls_first + n_cls_central - 1})"),
        ("regression: centralized 60 epochs final standardized MSE, mean of seeds",
         f"=AVERAGE({value_col}{reg_first}:{value_col}{reg_first + n_reg_central - 1})"),
    ]
    for k, (label, formula) in enumerate(means):
        ws.cell(row=mean_row + k, column=ref_name_col, value=label).font = _font()
        ws.cell(row=mean_row + k, column=ref_name_col + 1, value=formula).font = _font()
    cell_of = {
        "cls_central": f"${value_col}${mean_row}",
        "cls_majority": f"${value_col}${cls_first + n_cls_central}",
        "reg_lookup": f"${value_col}${cls_first + n_cls_central + 1}",
        "reg_central": f"${value_col}${mean_row + 1}",
        "reg_300": f"${value_col}${reg_first + n_reg_central}",
    }
    line_columns = [
        ("classification", "ref: centralized final train accuracy (mean of seeds)", "cls_central", CENTRAL_COLOR, "dash"),
        ("classification", "ref: majority-class train rate", "cls_majority", MUTED, "sysDot"),
        ("regression", "ref: lookup (= 1)", "reg_lookup", MUTED, None),
        ("regression", "ref: centralized 60 epochs (mean of seeds)", "reg_central", CENTRAL_COLOR, "dash"),
        ("regression", "ref: centralized matched steps (300 epochs)", "reg_300", CENTRAL_COLOR, "lgDashDot"),
    ]

    rows = []
    for i, rnd in enumerate(rounds):
        row: dict[str, Any] = {"round": rnd}
        for cols in seed_columns.values():
            for name, s in cols:
                row[name] = s["value"][i]
        rows.append(row)
    columns = ["round", *names, *[name for _, name, _, _, _ in line_columns]]
    formulas = {name: "=" + cell_of[key] for _, name, key, _, _ in line_columns}
    table = write_table(ws, rows, columns, formulas=formulas)
    for name in names:
        ws.column_dimensions[table.letter(name)].width = 16

    charts = _sheet(wb, "chart")
    anchors = ("A1", "L1", "A20", "L20")
    for (panel, cols), anchor in zip(seed_columns.items(), anchors):
        task, condition = panel
        y_title = "Global-train accuracy" if task == "classification" else "Global-train standardized MSE (log)"
        chart = _scatter(f"{task.capitalize()}, {CONDITION_LABELS[condition]}", "Communication round", y_title)
        _size(chart, 15.5, 8.5)
        for k, (name, s) in enumerate(cols):
            color = COLLAPSED_COLOR if s["label"] else SEED_COLORS[k % len(SEED_COLORS)]
            chart.series.append(_line_only(_series(table.ref(name), table.ref("round"), title_ref=table.header(name)),
                                           color, 1.5 if s["label"] else 1.25, "dash" if s["label"] else None))
        for line_task, name, _, color, dash in line_columns:
            if line_task == task:
                chart.series.append(_line_only(_series(table.ref(name), table.ref("round"),
                                                       title_ref=table.header(name)), color, 1.0, dash))
        _style_axis(chart.x_axis, "Communication round", minimum=0, maximum=max(rounds), gridlines=False,
                    major_unit=10)
        if task == "classification":
            _style_axis(chart.y_axis, y_title, minimum=0, maximum=1.04, number_format="0.0")
        else:
            _style_axis(chart.y_axis, y_title, minimum=1e-6, maximum=10, log=True, number_format="0.0E+0")
        chart.y_axis.crosses = "min"
        chart.x_axis.crosses = "min"
        _legend(chart)
        charts.add_chart(chart, anchor)

    sources = [s["source_file"] for seeds in curves.values() for s in seeds] + [
        r["source_file"] for r in refs if r["source_file"] != "definition"]
    write_notes(
        wb, build, "Fig 1: convergence (FL, K=2)",
        "Global-train metric after every communication round for FL K=2 IID and K=2 natural, "
        "both tasks, one series per initialization seed. Classification: global-train accuracy. "
        "Regression: global-train standardized MSE (residual target; 1 = the lookup), log axis. "
        "The natural K=2 regression seed that collapsed is its own series, labelled collapsed. "
        "Reference lines are columns of formulas pointing at the reference block on the data "
        "sheet; edit a reference value there and its line moves.",
        {"round": "Communication round (1–60; FedAvg, E = 5 local epochs).",
         "<task> <condition> seed <s>": "Global-train metric of that FL run after the round.",
         "<task> <condition> collapsed (seed <s>)": "The same, for the seed whose trained gate "
         "reads collapsed (one distinct prediction on the test set).",
         "ref: …": "Formula: the reference value it points at, repeated on every round.",
         "reference / value / source_file": "Reference block: per-seed centralized values, the "
         "majority-class train rate (no_walking, 276 of 1242 training windows) and the lookup; "
         "the two 'mean of seeds' rows below the block are AVERAGE formulas."},
        sources,
        extra=["This sheet is wide by design (one column per seed): the one exception to the "
               "one-row-per-point layout.",
               "Classification has no lookup baseline; its reference is the majority-class train "
               "rate (always predicting no_walking)."],
        metrics=("outcome",))
    return wb


# --------------------------------------------------------------------------------------
# Fig 2: centralized vs FL vs local-only, every condition
# --------------------------------------------------------------------------------------


def _with_axis_groups(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out, previous = [], None
    for p in points:
        label = "Centralized" if p["condition"] == "centralized" else CONDITION_LABELS[p["condition"]]
        out.append({**p, "condition_label": label, "axis_group": label if p["condition"] != previous else None})
        previous = p["condition"]
    return out


def _slot_chart(table: Table, rows: Sequence[Mapping[str, Any]], *, title: str, y_title: str,
                y_min: float, y_max: float, number_format: str, key_column: str,
                lines: Sequence[tuple[str, str, str, str | None]] = (), outcome_labels: bool = False,
                major_unit: float | None = None) -> LineChart:
    """One category slot per point: markers only, styled per point by family/outcome,
    custom y error bars, a two-level category axis (condition, model). Empty legend-key
    series (one per family, plus the non-trained outcomes) make the legend."""
    chart = LineChart()
    chart.title = _title(title)
    _size(chart, max(CHART_WIDTH_CM, FIG2_SLOT_CM * len(rows) + 5), 11.0)
    categories = AxDataSource(multiLvlStrRef=MultiLevelStrRef(f=table.span("axis_group", "model")))

    values = _series(table.ref("value"), title="value")
    values.cat = categories
    _markers_only(values, _marker("circle", INK, True))
    values.dPt = [DataPoint(idx=i, marker=_point_marker(r["family"], r.get("outcome")), spPr=_line(None))
                  for i, r in enumerate(rows)]
    values.errBars = _custom_error_bars(table)
    if outcome_labels:
        values.dLbls = _outcome_labels([(i, r["outcome"]) for i, r in enumerate(rows)], size=6.5)
    chart.series.append(values)

    for name, column, color, dash in lines:
        s = _series(table.ref(column), title_ref=table.header(column) if name is None else None, title=name)
        s.cat = categories
        chart.series.append(_line_only(s, color, 1.0, dash))

    families = [f for f in ("centralized", "fl", "local_only") if any(r["family"] == f for r in rows)]
    outcomes = [o for o in ("did not train", "collapsed") if any(r.get("outcome") == o for r in rows)]
    for family in families:
        s = _series(table.ref(key_column), title=FAMILY_LABELS[family])
        s.cat = categories
        chart.series.append(_markers_only(s, _point_marker(family, None)))
    for outcome in outcomes:
        s = _series(table.ref(key_column), title=f"regression: {outcome}")
        s.cat = categories
        chart.series.append(_markers_only(s, _marker(OUTCOME_SYMBOL[outcome], INK, False, 8)))

    _style_axis(chart.x_axis, None, gridlines=False, label_position="low")
    chart.x_axis.noMultiLvlLbl = False
    _style_axis(chart.y_axis, y_title, minimum=y_min, maximum=y_max, number_format=number_format,
                major_unit=major_unit)
    chart.x_axis.crosses = "min"
    _place_axes(chart)
    _legend(chart, deleted=[0])
    return chart


def _fig2_rows(points: Sequence[dict[str, Any]], value: str, low: str, high: str,
               extra: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for p in _with_axis_groups(points):
        rows.append({**p, "value": p[value], "ci_lower": p[low], "ci_upper": p[high],
                     **{k: p.get(k) for k in extra}})
    return rows


def build_fig2_classification(build: Build) -> Workbook:
    points = build.classification
    rows = _fig2_rows(points, "accuracy", "ci_low", "ci_high", ())
    wb = _new_workbook()
    columns = ["condition", "axis_group", "model", "condition_label", "family", "client", "seed",
               "correct", "runs", "value", "ci_lower", "ci_upper", "source_file", "legend_key"]
    table = write_table(wb["data"], rows, columns)
    chart = _slot_chart(table, rows, title="Run accuracy: centralized, FL and local-only",
                        y_title="Run accuracy (majority vote), Wilson 95% CI", y_min=0, y_max=1.05,
                        number_format="0.0", key_column="legend_key", major_unit=0.2)
    _sheet(wb, "chart").add_chart(chart, "A1")
    write_notes(
        wb, build, "Fig 2 (classification): run accuracy in every condition",
        "Run accuracy of every evaluated classification model on the 28 held-out test runs: "
        "centralized (3 seeds), then per condition FL (one point per seed) and local-only (one "
        "point per client and seed). Error bars are Wilson 95% intervals.",
        _columns_doc(table.columns, {
            "correct": "Test runs whose majority vote is correct.",
            "runs": "Test runs scored (28).",
            "value": "Run accuracy = correct / runs.",
            "legend_key": "Chart helper, intentionally empty: the legend-key series (one per "
                          "family) reference it, so they draw no points."}),
        [p["source_file"] for p in points],
        metrics=("run accuracy", "Wilson 95% interval"))
    return wb


def build_fig2_regression_skill(build: Build) -> Workbook:
    points = build.regression
    rows = _fig2_rows(points, "skill", "ci_low", "ci_high", ())
    for r in rows:
        r["zero_line"] = 0.0
        r["source_file"] = r["predictions_file"]
    wb = _new_workbook()
    columns = ["condition", "axis_group", "model", "condition_label", "family", "client", "seed",
               "outcome", "value", "ci_lower", "ci_upper", "zero_line", "source_file", "legend_key"]
    table = write_table(wb["data"], rows, columns)
    low = min(r["ci_lower"] for r in rows)
    y_min = math.floor(low) - 0.5 if low < -1 else -1.0
    common = dict(y_title="Skill vs lookup, run-level bootstrap 95% CI", number_format="0.0",
                  key_column="legend_key", lines=[("zero", "zero_line", MUTED, None)], outcome_labels=True)
    full = _slot_chart(table, rows, title="Regression skill: centralized, FL and local-only (full range)",
                       y_min=y_min, y_max=1.0, major_unit=1.0, **common)
    zoom = _slot_chart(table, rows, title="Regression skill (zoomed: some intervals extend beyond the axis)",
                       y_min=-1.5, y_max=1.0, major_unit=0.5, **common)
    charts = _sheet(wb, "chart")
    charts.add_chart(full, "A1")
    charts.add_chart(zoom, "A25")
    write_notes(
        wb, build, "Fig 2 (regression): skill in every condition",
        "Skill (1 − SSE_model / SSE_lookup) of every evaluated regression model on the 28 test "
        "runs: centralized residual model (3 seeds), then per condition FL and local-only. "
        "Natural K=2 FL is shown per seed and never averaged; its collapsed seed is marked. "
        "Error bars are run-level bootstrap 95% intervals. The first chart spans every "
        "interval; the second has the y-axis fixed to −1.5…1, so intervals reaching beyond it "
        "are cut at the plot edge (their values are in ci_lower / ci_upper).",
        _columns_doc(table.columns, {
            "value": "Skill vs the lookup.",
            "zero_line": "0 on every row: the zero line (skill of the lookup itself).",
            "source_file": "The model's saved run-level predictions; skill and CI are recomputed "
                           "from them and checked equal to the recorded values.",
            "legend_key": "Chart helper, intentionally empty: the legend-key series reference it."}),
        [r["source_file"] for r in rows] + ["artifacts/relabel/results_index.csv", "artifacts/s5/results_index.csv",
                                            "artifacts/relabel/derived/s5_disposition.csv"],
        metrics=("skill", "lookup RMSE", "bootstrap 95% interval", "outcome"))
    return wb


def build_fig2_regression_rmse(build: Build) -> Workbook:
    points = build.regression
    lookup = pf._one_value([p["lookup_rmse"] for p in points], "lookup RMSE across models")
    rows = _fig2_rows(points, "rmse", "rmse_ci_low", "rmse_ci_high", ())
    for r in rows:
        r["source_file"] = r["predictions_file"]
    wb = _new_workbook()
    ws = wb["data"]
    columns = ["condition", "axis_group", "model", "condition_label", "family", "client", "seed",
               "outcome", "value", "ci_lower", "ci_upper", "lookup_rmse", "lookup_line", "skill",
               "source_file", "legend_key"]
    # the lookup line points at one reference cell, right of the table
    ref_col = len(columns) + 3
    ws.cell(row=1, column=ref_col, value="lookup_rmse_reference").font = _font(bold=True)
    ws.cell(row=2, column=ref_col, value=lookup).font = _font()
    ws.cell(row=2, column=ref_col).number_format = "0.0000"
    ws.column_dimensions[get_column_letter(ref_col)].width = 24
    ref_cell = f"${get_column_letter(ref_col)}$2"
    table = write_table(ws, rows, columns, formulas={"lookup_line": "=" + ref_cell})
    high = max(r["ci_upper"] for r in rows)
    chart = _slot_chart(table, rows, title="Regression RMSE: centralized, FL and local-only",
                        y_title="RMSE (m/s), matched run-level bootstrap 95% CI", y_min=0,
                        y_max=math.ceil(high * 100) / 100, number_format="0.00", key_column="legend_key",
                        lines=[(f"lookup ({lookup:.4f} m/s)", "lookup_line", MUTED, "dash")],
                        outcome_labels=True, major_unit=0.01)
    _sheet(wb, "chart").add_chart(chart, "A1")
    write_notes(
        wb, build, "Fig 2 (regression): RMSE in every condition",
        "RMSE (m/s) of every evaluated regression model on the 28 test runs, recomputed exactly "
        "from each model's saved run predictions (local-only: each client's own predictions). "
        "The dashed line is the lookup RMSE.",
        _columns_doc(table.columns, {
            "value": "RMSE of the run-mean speed prediction, m/s (exact, from the predictions).",
            "ci_lower": "Lower end of the matched bootstrap 95% interval of RMSE (derived here).",
            "ci_upper": "Upper end of the matched bootstrap 95% interval of RMSE (derived here).",
            "lookup_rmse": "Lookup RMSE recomputed from the same predictions file (m/s).",
            "lookup_line": "Formula: the lookup_rmse_reference cell; the reference line.",
            "lookup_rmse_reference": "The single lookup RMSE (equal for every model). Edit it to "
                                     "move the line.",
            "skill": "Skill vs the lookup, for reference.",
            "source_file": "The model's saved run-level predictions.",
            "legend_key": "Chart helper, intentionally empty: the legend-key series reference it."}),
        [r["source_file"] for r in rows],
        extra=["RMSE intervals are new, derived numbers: the percentile interval of RMSE over the "
               "same 10,000 run resamples (same generator, seed and draws) as the saved skill "
               "intervals. The build re-derives every saved skill interval from those draws and "
               "checks it is equal."],
        metrics=("lookup RMSE", "bootstrap 95% interval", "outcome"))
    return wb


# --------------------------------------------------------------------------------------
# Fig 3: sensor axis
# --------------------------------------------------------------------------------------

RHO_DX = 0.03


def build_fig3(build: Build) -> Workbook:
    rows: list[dict[str, Any]] = []
    for task, points in (("classification", build.classification), ("regression", build.regression)):
        for family in ("fl", "local_only"):
            for condition, rho in sorted(pf.RHO_CONDITIONS.items(), key=lambda kv: -kv[1]):
                rows += [{**p, "task": task, "rho": rho, "x_plot": rho + (RHO_DX if family == "fl" else -RHO_DX)}
                         for p in points if p["family"] == family and p["condition"] == condition]
        rows += [{**p, "task": task} for p in points if p["family"] == "centralized"]
    for r in rows:
        if r["task"] == "classification":
            r.update(value=r["accuracy"], ci_lower=r["ci_low"], ci_upper=r["ci_high"], outcome=None)
        else:
            r.update(value=r["skill"], ci_lower=r["ci_low"], ci_upper=r["ci_high"],
                     source_file=r["predictions_file"])

    wb = _new_workbook()
    ws = wb["data"]
    columns = ["task", "family", "condition", "rho", "x_plot", "client", "seed", "outcome", "value",
               "ci_lower", "ci_upper", "source_file"]
    table = write_table(ws, rows, columns)
    rng = {k: table.ref(k).split("!")[1] for k in ("task", "family", "rho", "outcome", "value")}

    # Summary block right of the table: local-only mean ± std per ρ, centralized mean and
    # zero lines. Every y / std / count is a formula over the table, so edits flow through.
    s_columns = ["summary_task", "summary_line", "summary_rho", "summary_x", "summary_y", "summary_std",
                 "summary_n"]
    plan = []
    for task in ("classification", "regression"):
        plan += [(task, "local-only mean ± std", rho - RHO_DX, rho)
                 for rho in sorted(set(pf.RHO_CONDITIONS.values()), reverse=True)]
        plan += [(task, "centralized (mean of seeds)", x, None) for x in (1.1, -0.1)]
        if task == "regression":
            plan += [(task, "zero", x, None) for x in (1.1, -0.1)]
    s_table = Table(ws, s_columns, 1, len(plan), len(columns) + 3)
    summary = []
    for i, (task, line, x, rho) in enumerate(plan):
        cell = {k: s_table.cell(k, i, absolute=False) for k in s_columns}
        row: dict[str, Any] = {"summary_task": task, "summary_line": line, "summary_rho": rho, "summary_x": x}
        if line == "local-only mean ± std":
            where = (f'{rng["task"]},"{task}",{rng["family"]},"local_only",{rng["rho"]},{cell["summary_rho"]}'
                     + (f',{rng["outcome"]},"trained"' if task == "regression" else ""))
            mask = (f'({rng["task"]}="{task}")*({rng["family"]}="local_only")*({rng["rho"]}={cell["summary_rho"]})'
                    + (f'*({rng["outcome"]}="trained")' if task == "regression" else ""))
            row["summary_y"] = f'=AVERAGEIFS({rng["value"]},{where})'
            row["summary_n"] = f"=COUNTIFS({where})"
            row["summary_std"] = (f'=SQRT(SUMPRODUCT({mask}*({rng["value"]}-{cell["summary_y"]})^2)'
                                  f'/({cell["summary_n"]}-1))')
        elif line.startswith("centralized"):
            row["summary_y"] = f'=AVERAGEIFS({rng["value"]},{rng["task"]},"{task}",{rng["family"]},"centralized")'
        else:
            row["summary_y"] = 0.0
        summary.append(row)
    write_table(ws, summary, s_columns, first_col=s_table.first_col)
    ws.column_dimensions[s_table.letter("summary_line")].width = 28

    charts = _sheet(wb, "chart")
    for task, anchor in (("classification", "A1"), ("regression", "A24")):
        y_title = "Run accuracy (Wilson 95% CI)" if task == "classification" else "Skill vs lookup (bootstrap 95% CI)"
        chart = _scatter(f"{task.capitalize()} vs position overlap ρ (K=3)", "Position overlap ρ (K=3)", y_title)
        _size(chart, 16, 10)
        a, b = table.rows_where(rows, task=task, family="fl")
        fl = _series(table.ref("value", a, b), table.ref("x_plot", a, b), title="FL (FedAvg), 95% CI")
        fl.marker = _marker("circle", FL_COLOR, True)
        fl.graphicalProperties = _line(FL_COLOR, 1.0)
        fl.smooth = False
        fl.errBars = _custom_error_bars(table, a, b, direction="y", color=FL_COLOR)
        fl.dPt = [DataPoint(idx=i, marker=_point_marker("fl", rows[a + i].get("outcome"))) for i in range(b - a)]
        if task == "regression":
            fl.dLbls = _outcome_labels([(i, rows[a + i]["outcome"]) for i in range(b - a)], position="t")
        chart.series.append(fl)

        a, b = table.rows_where(rows, task=task, family="local_only")
        lo = _markers_only(_series(table.ref("value", a, b), table.ref("x_plot", a, b), title="Local-only client"),
                           _marker("square", LOCAL_COLOR, False, 6))
        lo.dPt = [DataPoint(idx=i, marker=_point_marker("local_only", rows[a + i].get("outcome")))
                  for i in range(b - a)]
        if task == "regression":
            lo.dLbls = _outcome_labels([(i, rows[a + i]["outcome"]) for i in range(b - a)], size=6, position="l")
        chart.series.append(lo)

        a, b = s_table.rows_where(summary, summary_task=task, summary_line="local-only mean ± std")
        mean = _series(s_table.ref("summary_y", a, b), s_table.ref("summary_x", a, b),
                       title="Local-only mean ± std" + (" (trained clients)" if task == "regression" else ""))
        mean.marker = _marker("square", LOCAL_COLOR, True, 8)
        mean.graphicalProperties = _line(LOCAL_COLOR, 1.0, "dash")
        mean.smooth = False
        mean.errBars = _custom_error_bars(s_table, a, b, direction="y", plus="summary_std", minus="summary_std",
                                          color=LOCAL_COLOR)
        chart.series.append(mean)

        a, b = s_table.rows_where(summary, summary_task=task, summary_line="centralized (mean of seeds)")
        chart.series.append(_line_only(_series(s_table.ref("summary_y", a, b), s_table.ref("summary_x", a, b),
                                               title="Centralized (mean of seeds)"), CENTRAL_COLOR, 1.0, "dash"))
        if task == "regression":
            a, b = s_table.rows_where(summary, summary_task=task, summary_line="zero")
            chart.series.append(_line_only(_series(s_table.ref("summary_y", a, b), s_table.ref("summary_x", a, b),
                                                   title="zero"), MUTED, 0.75))
        _style_axis(chart.x_axis, "Position overlap ρ (K=3)", minimum=-0.1, maximum=1.1, reverse=True,
                    gridlines=False, number_format="0.0##", major_unit=0.125)
        if task == "classification":
            _style_axis(chart.y_axis, y_title, minimum=0.7, maximum=1.02, number_format="0.00")
        else:
            low = min(r["ci_lower"] for r in rows if r["task"] == "regression" and r.get("rho") is not None)
            _style_axis(chart.y_axis, y_title, minimum=math.floor(low) - 0.5, maximum=1.0, number_format="0.0")
        chart.y_axis.crosses = "max"
        chart.x_axis.crosses = "min"
        _legend(chart)
        charts.add_chart(chart, anchor)

    write_notes(
        wb, build, "Fig 3: the channel-availability (sensor) axis",
        "Both tasks against the position overlap ρ of the K=3 channel-availability partitions "
        "(ρ = 1.0 is K=3 IID with all positions; 0.375 and 0 mask positions per client on the "
        "same run assignment). FL with its 95% interval; local-only clients; local-only mean ± "
        "std (regression: over trained clients only, as in the paper); centralized mean as a "
        "reference. Every regression point carries its outcome label. The x axis is reversed "
        "(ρ = 1.0 on the left); x_plot offsets FL and local-only by ±0.03 so they don't overlap.",
        _columns_doc(table.columns, {
            "rho": "Position overlap ρ of the condition (blank for centralized rows).",
            "x_plot": "Plotted x: ρ + 0.03 for FL, ρ − 0.03 for local-only (blank for centralized).",
            "value": "Run accuracy (classification) or skill (regression).",
            "summary_* (block right of the table)":
                "Summary lines. Local-only rows: summary_y = AVERAGEIFS over that ρ's local-only "
                "clients (regression: outcome = trained), summary_std = sample std via SUMPRODUCT, "
                "summary_n = COUNTIFS; the error bars use summary_std. Centralized rows: "
                "summary_y = AVERAGEIFS over the task's centralized rows, at summary_x = 1.1 and "
                "−0.1. Zero rows: the zero line."}),
        [r["source_file"] for r in rows],
        metrics=("run accuracy", "Wilson 95% interval", "skill", "bootstrap 95% interval", "outcome"))
    return wb


# --------------------------------------------------------------------------------------
# Fig 4: partitions
# --------------------------------------------------------------------------------------


def _identity(source: str, subject: str) -> str:
    return f"{source}:{subject}"


def scheme_label(conditions: Sequence[str]) -> str:
    if len(conditions) == 1:
        return CONDITION_LABELS[conditions[0]]
    rhos = [pf.RHO_CONDITIONS[c] for c in conditions if c in pf.RHO_CONDITIONS]
    if len(rhos) == len(conditions):
        return "K=3 IID (ρ=" + ", ".join("1.0" if r == 1.0 else f"{r:g}" for r in sorted(rhos, reverse=True)) + ")"
    return " / ".join(CONDITION_LABELS[c] for c in conditions)


def build_fig4(build: Build) -> Workbook:
    results = build.results
    schemes = pf.partition_schemes(results)
    wb = _new_workbook()
    rows, blocks = [], []
    for scheme in schemes:
        clients, groups, matrix = pf.partition_matrix(scheme["assignment"])
        identities = [_identity(*g) for g in groups]
        label = scheme_label(scheme["conditions"])
        first_row = results.rows("fl", scheme["conditions"][0])[0]
        source = results.relative(results.job_dir(first_row) / "partitions" / "combined_iid_run_to_client.csv")
        for i, client in enumerate(clients):
            for j, identity in enumerate(identities):
                rows.append({"scheme": label, "client": client, "identity": identity,
                             "source": groups[j][0], "runs": int(matrix[i, j]), "source_file": source})
        blocks.append((label, clients, identities, scheme["conditions"]))
    table = write_table(wb["data"], rows, ["scheme", "client", "identity", "source", "runs", "source_file"])
    rng = {k: table.ref(k) for k in ("scheme", "client", "identity", "runs")}

    charts = _sheet(wb, "chart")
    heat = _sheet(wb, "heatmap")
    heat.column_dimensions["A"].width = 30
    vmax = max(r["runs"] for r in rows)
    top = 1
    for k, (label, clients, identities, conditions) in enumerate(blocks):
        heat.cell(row=top, column=1, value=label).font = _font(bold=True)
        heat.cell(row=top, column=2 + len(identities) + 1,
                  value="conditions: " + ", ".join(conditions)).font = _font(italic=True, size=9)
        header = top + 1
        heat.cell(row=header, column=1, value="client").font = _font(bold=True)
        for j, identity in enumerate(identities):
            heat.cell(row=header, column=2 + j, value=identity).font = _font(bold=True, size=9)
            heat.column_dimensions[get_column_letter(2 + j)].width = 19
        heat.cell(row=header, column=2 + len(identities), value="total").font = _font(bold=True)
        for i, client in enumerate(clients):
            r = header + 1 + i
            heat.cell(row=r, column=1, value=client).font = _font()
            for j in range(len(identities)):
                col = get_column_letter(2 + j)
                formula = (f'=SUMIFS({rng["runs"]},{rng["scheme"]},$A${top},{rng["client"]},$A{r},'
                           f'{rng["identity"]},{col}${header})')
                heat.cell(row=r, column=2 + j, value=formula).font = _font()
            last = get_column_letter(1 + len(identities))
            heat.cell(row=r, column=2 + len(identities), value=f"=SUM(B{r}:{last}{r})").font = _font(bold=True)
        body = f"B{header + 1}:{get_column_letter(1 + len(identities))}{header + len(clients)}"
        heat.conditional_formatting.add(body, ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF",
                                                             end_type="num", end_value=vmax, end_color="2166AC"))

        chart = BarChart()
        chart.type = "col"
        chart.grouping = "stacked"
        chart.overlap = 100
        chart.gapWidth = 60
        chart.title = _title(f"Training runs per client: {label}")
        for j, identity in enumerate(identities):
            col = get_column_letter(2 + j)
            sheet = quote_sheetname(heat.title)
            s = _series(f"{sheet}!${col}${header + 1}:${col}${header + len(clients)}",
                        title_ref=f"{sheet}!${col}${header}")
            s.cat = AxDataSource(strRef=StrRef(f=f"{sheet}!$A${header + 1}:$A${header + len(clients)}"))
            color = IDENTITY_COLORS[j % len(IDENTITY_COLORS)]
            s.graphicalProperties = GraphicalProperties(solidFill=color, ln=LineProperties(solidFill="FFFFFF", w=6350))
            chart.series.append(s)
        _style_axis(chart.x_axis, "Client", gridlines=False)
        _style_axis(chart.y_axis, "Training runs", minimum=0, number_format="0")
        _place_axes(chart)
        _legend(chart, position="r")
        _size(chart, 16, 8)
        charts.add_chart(chart, f"A{1 + 17 * k}")
        top = header + len(clients) + 3

    write_notes(
        wb, build, "Fig 4: client × identity training-run counts per partition scheme",
        "Training runs per client and (source, subject) identity for every partition scheme. "
        "Excel has no heatmap chart: the heatmap sheet holds one table per scheme (SUMIFS "
        "formulas over the data sheet) with a white→blue colour scale; the chart sheet shows the "
        "same counts as stacked columns. Conditions sharing one run assignment (the K=3 ρ "
        "layouts mask channels on the K=3 IID runs) are one scheme; the assignment is checked "
        "identical across tasks and seeds of each condition.",
        _columns_doc(table.columns, {
            "scheme": "Partition scheme (one distinct run assignment).",
            "client": "Client.",
            "identity": "(source, subject) identity, as source:subject.",
            "source": "Recording campaign of the identity.",
            "runs": "Training runs of that identity held by that client.",
            "source_file": "Run → client assignment file the counts come from."}),
        [r["source_file"] for r in rows],
        metrics=())
    return wb


# --------------------------------------------------------------------------------------
# Fig 5: run-level scatter
# --------------------------------------------------------------------------------------


def build_fig5(build: Build) -> Workbook:
    results = build.results
    panels = pf.scatter_panels(results)
    identities = sorted({_identity(r["source_id"], r["subject_id"]) for p in panels for r in p["runs"]})
    if len(identities) > len(IDENTITY_COLORS):
        raise AssertionError("More identities than palette slots.")
    style = {identity: i for i, identity in enumerate(identities)}
    rows, line_rows, panel_info = [], [], []
    values = [float(r[k]) for p in panels for r in p["runs"] for k in ("actual_speed_mps", "predicted_speed_mps")]
    lo, hi = math.floor(min(values) * 20) / 20, math.ceil(max(values) * 20) / 20
    for panel in panels:
        record = panel["record"]
        exact = exact_regression(results, record)
        for run in sorted(panel["runs"], key=lambda r: (_identity(r["source_id"], r["subject_id"]), r["run_uid"])):
            rows.append({"panel": panel["label"], "identity": _identity(run["source_id"], run["subject_id"]),
                         "source": run["source_id"], "run_uid": run["run_uid"],
                         "actual_speed_mps": float(run["actual_speed_mps"]),
                         "predicted_speed_mps": float(run["predicted_speed_mps"]),
                         "source_file": exact["predictions_file"]})
        for x in (lo, hi):
            line_rows.append({"panel": panel["label"], "x": x})
        panel_info.append((panel["label"], record, exact))

    wb = _new_workbook()
    ws = wb["data"]
    columns = ["panel", "identity", "source", "run_uid", "actual_speed_mps", "predicted_speed_mps", "source_file"]
    table = write_table(ws, rows, columns)
    first = len(columns) + 3
    l_table = write_table(ws, line_rows, ["panel", "x", "y"], first_col=first, formulas={"y": "={x}"})

    charts = _sheet(wb, "chart")
    for k, (label, record, exact) in enumerate(panel_info):
        title = (f"{label}: RMSE {exact['rmse']:.4f} m/s, lookup {exact['lookup_rmse']:.4f} m/s, "
                 f"skill {record['skill']:+.3f} [{record['ci_low']:+.2f}, {record['ci_high']:+.2f}]")
        chart = _scatter(title, "Actual speed (m/s)", "Predicted speed (m/s)")
        chart.title = _title(title, 9)
        _size(chart, 13, 11)
        for identity in identities:
            hits = [i for i, r in enumerate(rows) if r["panel"] == label and r["identity"] == identity]
            if not hits:
                continue
            a, b = table.rows_where(rows, panel=label, identity=identity)
            i = style[identity]
            filled = identity.startswith("test_2:")
            chart.series.append(_markers_only(
                _series(table.ref("predicted_speed_mps", a, b), table.ref("actual_speed_mps", a, b), title=identity),
                _marker(IDENTITY_SYMBOLS[i], IDENTITY_COLORS[i], filled, 7)))
        a, b = l_table.rows_where(line_rows, panel=label)
        chart.series.append(_line_only(_series(l_table.ref("y", a, b), l_table.ref("x", a, b), title="identity line (y = x)"),
                                       MUTED, 0.75))
        for axis, name in ((chart.x_axis, "Actual speed (m/s)"), (chart.y_axis, "Predicted speed (m/s)")):
            _style_axis(axis, name, minimum=lo, maximum=hi, number_format="0.00", major_unit=0.05)
        chart.x_axis.crosses = "min"
        chart.y_axis.crosses = "min"
        _legend(chart, position="r")
        charts.add_chart(chart, ("A1", "J1", "A24", "J24")[k])

    write_notes(
        wb, build, "Fig 5: predicted vs actual run speed",
        "Run-mean predicted vs actual walking speed on the 28 test runs, seed 4601: centralized "
        "residual model, FL K=2 IID, and each local-only K=2 IID client. Points are grouped by "
        "(source, subject) identity, one series each; Test_2 identities are filled markers, "
        "20251124 identities hollow. The grey line is y = x. Chart titles give the exact RMSE, "
        "the lookup RMSE and skill [bootstrap 95% CI]; titles are text set at build time.",
        _columns_doc(table.columns, {
            "panel": "Chart the row is plotted in.",
            "identity": "(source, subject) identity, as source:subject; one series per identity.",
            "source": "Recording campaign.",
            "run_uid": "Test run.",
            "actual_speed_mps": "Measured run-mean speed (APDM), m/s.",
            "predicted_speed_mps": "Model's run-mean speed prediction, m/s.",
            "source_file": "The model's saved run-level predictions.",
            "x / y (block right of the table)": "Identity line end points; y is a formula = x."}),
        [r["source_file"] for r in rows],
        metrics=("skill", "lookup RMSE", "bootstrap 95% interval"))
    return wb


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def _dataset_rows(results: RelabelResults) -> list[dict[str, Any]]:
    """T1 numbers, as ``paper_figures.table_dataset`` (strata are (source, subject))."""
    data = pf.dataset_table_data(results)
    out = []

    def emit(source: str, identity: str, members: list[dict[str, Any]]) -> None:
        speeds = [x for m in members for x in m["speeds"]]
        within = pf._within_std([m["speeds"] for m in members]) if len(members) > 1 else float(np.std(speeds, ddof=1))
        values = {k: sum(m[k] for m in members) for k in
                  ("runs_train", "runs_test", "cls_train", "cls_test", "reg_train", "reg_test")}
        out.append({"source": source, "identity": identity,
                    "usable_runs": values["runs_train"] + values["runs_test"], **values,
                    "speed_mean_mps": float(np.mean(speeds)), "speed_std_mps": float(np.std(speeds, ddof=1)),
                    "speed_min_mps": float(np.min(speeds)), "speed_max_mps": float(np.max(speeds)),
                    "within_identity_std_mps": within})

    for source in sorted({s for s, _ in data["strata"]}):
        keys = [k for k in data["strata"] if k[0] == source]
        for key in keys:
            emit(source, _identity(*key), [data["strata"][key]])
        emit(source, "all", [data["strata"][k] for k in keys])
    emit("all", "all", list(data["strata"].values()))
    funnel = data["status"]
    out.append({"source": "run funnel", "identity": "", "recorded_runs": data["num_recorded_runs"],
                "excluded_runs": funnel["excluded"], "skipped_runs": funnel["skipped"],
                "usable_runs": funnel["train"] + funnel["test"]})
    for source, counts in data["by_source"].items():
        out.append({"source": f"run funnel {source}", "identity": "", "recorded_runs": sum(counts.values()),
                    "excluded_runs": counts.get("excluded", 0), "skipped_runs": counts.get("skipped", 0),
                    "usable_runs": counts.get("train", 0) + counts.get("test", 0)})
    return out


def _centralized_rows(results: RelabelResults) -> list[dict[str, Any]]:
    """T2 numbers, as ``paper_figures.table_centralized``."""
    steps = pf.gradient_steps(results)
    out = []
    for r in pf.classification_records(results, "centralized", "classification"):
        out.append({"model": "classification", "epochs": 60, "steps": steps["centralized_60e_classification"],
                    "seed": r["seed"], "metric": "run accuracy", "value": r["accuracy"], "ci_lower": r["ci_low"],
                    "ci_upper": r["ci_high"], "correct": r["correct"], "runs": r["runs"],
                    "window_accuracy": r["window_accuracy"]})
    lookup = pf._one_value([r["baseline_rmse"] for r in pf.regression_records(results, "centralized", "residual")],
                           "lookup RMSE across centralized runs")
    out.append({"model": "lookup baseline", "metric": "skill", "value": 0.0, "rmse_mps": lookup})
    for condition in ("absolute", "residual"):
        for r in pf.regression_records(results, "centralized", condition):
            out.append({"model": f"regression {condition}", "epochs": 60, "steps": steps["centralized_60e_regression"],
                        "seed": r["seed"], "metric": "skill", "value": r["skill"], "ci_lower": r["ci_low"],
                        "ci_upper": r["ci_high"], "rmse_mps": r["rmse"],
                        "final_train_standardized_mse": r["gated_mse"], "outcome": r["outcome"]})
    r300 = pf.residual_300e_record(results)
    out.append({"model": "regression residual (matched steps)", "epochs": r300["epochs"],
                "steps": steps["centralized_300e_regression"], "seed": r300["seed"], "metric": "skill",
                "value": r300["skill"], "ci_lower": r300["ci_low"], "ci_upper": r300["ci_high"],
                "rmse_mps": r300["rmse"], "final_train_standardized_mse": r300["gated_mse"],
                "outcome": r300["outcome"]})
    return out


def _source_probe_rows(results: RelabelResults) -> list[dict[str, Any]]:
    """T5, source probe only (the first block of ``paper_figures.controls_records``)."""
    row = results.rows("source_probe")[0]
    probe = pf._read_json(results.job_dir(row) / "source_probe_summary.json")
    vote = probe["run_level"]["majority_vote"]
    out = [{"set": "canonical test", "source": "both", "runs": vote["num_runs"], "correct": vote["num_correct"],
            "value": vote["num_correct"] / vote["num_runs"], "ci_lower": vote["wilson95"][0],
            "ci_upper": vote["wilson95"][1], "window_accuracy": probe["test_window_accuracy"],
            "window_accuracy_no_walking": probe["test_window_accuracy_by_activity"]["no_walking"]["accuracy"],
            "window_accuracy_walking": probe["test_window_accuracy_by_activity"]["walking"]["accuracy"],
            "majority_class_rate": probe["chance_accuracy_majority_class_test_windows"]}]
    for source, block in sorted(vote["per_source"].items()):
        out.append({"set": "canonical test", "source": source, "runs": block["num_runs"],
                    "correct": block["num_correct"], "value": block["num_correct"] / block["num_runs"],
                    "ci_lower": block["wilson95"][0], "ci_upper": block["wilson95"][1]})
    return out


def build_tables(build: Build) -> Workbook:
    results = build.results
    wb = _new_workbook()
    del wb["data"]
    sheets = []

    t1 = _dataset_rows(results)
    sheets.append(("T1_dataset", t1, ["source", "identity", "usable_runs", "runs_train", "runs_test", "cls_train",
                                      "cls_test", "reg_train", "reg_test", "speed_mean_mps", "speed_std_mps",
                                      "speed_min_mps", "speed_max_mps", "within_identity_std_mps", "recorded_runs",
                                      "excluded_runs", "skipped_runs"]))
    t2 = _centralized_rows(results)
    sheets.append(("T2_centralized", t2, ["model", "epochs", "steps", "seed", "metric", "value", "ci_lower",
                                          "ci_upper", "correct", "runs", "window_accuracy", "rmse_mps",
                                          "final_train_standardized_mse", "outcome"]))
    t3 = [{**r, "value": r["skill"], "ci_lower": r["ci_low"], "ci_upper": r["ci_high"]}
          for r in pf.sweep_records(results)]
    sheets.append(("T3_r1_sweep", t3, ["partition", "config", "stage", "aggregation", "local_epochs",
                                       "optimizer_state", "steps", "final_standardized_mse", "distinct_residuals",
                                       "relative_output_spread", "value", "ci_lower", "ci_upper", "rmse",
                                       "s3_acceptance", "s3_status", "outcome_s5"]))
    t4 = [{**r, "ci_lower": r["ci_low"], "ci_upper": r["ci_high"]} for r in pf.main_matrix_long(results)]
    sheets.append(("T4_main_matrix", t4, ["family", "condition", "task", "seed", "client", "metric", "value",
                                          "ci_lower", "ci_upper", "correct", "runs", "rmse_mps", "outcome"]))
    t5 = _source_probe_rows(results)
    sheets.append(("T5_source_probe", t5, ["set", "source", "runs", "correct", "value", "ci_lower", "ci_upper",
                                           "window_accuracy", "window_accuracy_no_walking",
                                           "window_accuracy_walking", "majority_class_rate"]))
    rmse = [{"family": p["family"], "condition": p["condition"], "seed": p["seed"], "client": p["client"],
             "outcome": p["outcome"], "value": p["rmse"], "ci_lower": p["rmse_ci_low"], "ci_upper": p["rmse_ci_high"],
             "lookup_rmse": p["lookup_rmse"], "delta_rmse": p["rmse"] - p["lookup_rmse"], "skill": p["skill"],
             "skill_ci_lower": p["ci_low"], "skill_ci_upper": p["ci_high"], "source_file": p["predictions_file"]}
            for p in build.regression]
    sheets.append(("RMSE", rmse, ["family", "condition", "seed", "client", "outcome", "value", "ci_lower", "ci_upper",
                                  "lookup_rmse", "delta_rmse", "skill", "skill_ci_lower", "skill_ci_upper",
                                  "source_file"]))
    for title, rows, columns in sheets:
        ws = wb.create_sheet(title)
        write_table(ws, rows, columns)

    sources = [
        "artifacts/relabel/results_index.csv", "artifacts/relabel/derived/run_level_classification.csv",
        "artifacts/relabel/derived/s5_disposition.csv", "artifacts/s5/results_index.csv",
        "artifacts/relabel/rescore/local_only_k2_natural/client_0/run_level_metrics.json",
        "artifacts/relabel/rescore/local_only_k2_natural/client_1/run_level_metrics.json",
        f"artifacts/{pf.CENTRALIZED_CLASSIFICATION_DIR}/run_manifest.csv (T1 runs and windows)",
        f"artifacts/{pf.CENTRALIZED_REGRESSION_DIR}/run_manifest.csv (T1 speeds and walking windows)",
        f"artifacts/{pf.CENTRALIZED_RESIDUAL_300E_DIR}/ (T2 matched-steps row)",
        "artifacts/s3_r1_sweep/sweep_summary.json and the sweep runs' predictions (T3)",
        "artifacts/s5/source_probe/seed4601/source_probe_summary.json (T5)",
    ] + [r["source_file"] for r in rmse]
    write_notes(
        wb, build, "Tables T1–T5 and the regression RMSE table",
        "One sheet per table. T1 dataset per (source, subject) identity and the run funnel; T2 "
        "centralized references; T3 the R1 sweep; T4 the main matrix, one row per evaluated "
        "model; T5 the source probe only; RMSE per regression model, exact from the saved "
        "predictions.",
        {"value": "T2/T4: run accuracy (classification) or skill (regression), per the metric column. "
                  "T3: skill. T5: run accuracy of the source probe. RMSE: RMSE in m/s.",
         "ci_lower / ci_upper": "95% interval: Wilson (run accuracy), run-level bootstrap (skill), "
                                "matched run-level bootstrap (RMSE sheet; derived here).",
         "err_plus / err_minus": "Formulas: ci_upper − value and value − ci_lower.",
         "T1 columns": "runs_* = runs by split; cls_* = classification windows (incl. no_walking); "
                       "reg_* = regression (walking) windows; speed_* over runs (one APDM mean speed "
                       "per run); within_identity_std_mps = pooled within-identity std.",
         "T2 columns": "steps = gradient steps; window_accuracy = test-window accuracy; "
                       "final_train_standardized_mse = the gated training MSE.",
         "T3 columns": "As the paper's T3: S3 configuration, steps, final standardized MSE, distinct "
                       "residual predictions, relative output spread, S3 acceptance and the "
                       "outcome re-applied with S5 definitions.",
         "T4 columns": "family, condition, task, seed, client; correct / runs for classification; "
                       "rmse_mps and outcome for regression.",
         "T5 columns": "Run vote of the source probe (label = recording campaign), per source; "
                       "window accuracies and the majority-class test-window rate.",
         "RMSE columns": "value = exact RMSE; lookup_rmse; delta_rmse = value − lookup_rmse; skill "
                         "with its saved bootstrap interval."},
        sources,
        extra=["This workbook holds tables only, no charts."])
    return wb


# --------------------------------------------------------------------------------------
# everything
# --------------------------------------------------------------------------------------

WORKBOOKS: dict[str, Callable[[Build], Workbook]] = {
    "fig1_convergence.xlsx": build_fig1,
    "fig2_classification_comparison.xlsx": build_fig2_classification,
    "fig2_regression_skill.xlsx": build_fig2_regression_skill,
    "fig2_regression_rmse.xlsx": build_fig2_regression_rmse,
    "fig3_sensor_axis.xlsx": build_fig3,
    "fig4_partitions.xlsx": build_fig4,
    "fig5_run_scatter.xlsx": build_fig5,
    "tables.xlsx": build_tables,
}


def make_all_workbooks(artifacts_root: PathLike, out_dir: PathLike, repo: PathLike | None = None) -> list[Path]:
    artifacts_root = Path(artifacts_root).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir == artifacts_root or artifacts_root in out_dir.parents:
        raise ValueError(f"Refusing to write under the results folder: {out_dir}")
    build = Build(commit=git_commit(Path(repo) if repo else artifacts_root.parent),
                  results=load_relabel_results(artifacts_root))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, builder in WORKBOOKS.items():
        path = out_dir / name
        builder(build).save(path)
        paths.append(path)
    return paths
