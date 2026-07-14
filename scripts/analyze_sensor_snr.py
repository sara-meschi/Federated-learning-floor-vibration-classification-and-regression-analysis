from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.config import load_config
from redo_by_sara.preprocessing import _find_apdm_csv, _parse_speed_labels


SNR_DB_NOISY = 3.0
SNR_DB_GOOD = 10.0
SNR_DB_EXCELLENT = 20.0


def _clean(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _get_general_parameter(perams: np.ndarray, name: str) -> float | None:
    mask = perams["parameter"] == name.encode()
    matches = perams[mask]
    if len(matches) == 0:
        return None
    return float(_clean(matches["value"][0]))


def _subject_id_from_hdf5(path: Path) -> str:
    return path.stem.split("_")[-1]


def _load_run_segments(path: Path) -> dict[tuple[str, int], dict[str, float | int]]:
    segments: dict[tuple[str, int], dict[str, float | int]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = f"{int(row['Subject']):03d}"
            run_index = int(row["Run"])
            segments[(subject_id, run_index)] = {
                "no_step_start": float(row["No Step Start (s)"]),
                "no_step_end": float(row["No Step End (s)"]),
                "data_start": float(row["Data Start (s)"]),
                "data_end": float(row["Data End (s)"] or 0.0),
                "skip_run": int(row["Skip Run"]),
            }
    return segments


def _sensor_rows(handle: h5py.File) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if "experiment/sensors" not in handle:
        return rows

    sensors = handle["experiment/sensors"][:]
    for channel_number, row in enumerate(sensors, start=1):
        rows[channel_number] = {
            "daq_channel": _clean(row["channel"]),
            "model": _clean(row["model"]),
            "serial": _clean(row["serial"]),
            "units": _clean(row["units"]),
            "location_x": float(row["location_x"]),
            "location_y": float(row["location_y"]),
            "location_z": float(row["location_z"]),
            "direction_x": float(row["direction_x"]),
            "direction_y": float(row["direction_y"]),
            "direction_z": float(row["direction_z"]),
        }
    return rows


def _sensor_channel_pairs(
    sensor_channel_map: dict[str, list[int]],
    sensor_order: Iterable[str],
) -> list[tuple[str, int, str]]:
    pairs: list[tuple[str, int, str]] = []
    for sensor_name in sensor_order:
        channels = sensor_channel_map[sensor_name]
        for channel in channels:
            label = sensor_name if len(channels) == 1 else f"{sensor_name}:ch{channel}"
            pairs.append((sensor_name, int(channel), label))
    return pairs


def _segment_slice(start_sec: float, end_sec: float, fs: float, sample_count: int) -> slice | None:
    start_idx = max(0, min(sample_count, int(round(start_sec * fs))))
    end_idx = max(0, min(sample_count, int(round(end_sec * fs))))
    if end_idx <= start_idx:
        return None
    return slice(start_idx, end_idx)


def _demeaned_power(values: np.ndarray) -> tuple[float, float, float]:
    centered = values.astype(np.float64, copy=False) - float(np.mean(values))
    power = float(np.mean(np.square(centered)))
    rms = float(math.sqrt(power))
    peak_to_peak = float(np.ptp(values))
    return power, rms, peak_to_peak


def _db_ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0.0 or denominator <= 0.0:
        return float("nan")
    return 10.0 * math.log10(numerator / denominator)


def _status(snr_db: float) -> str:
    if not math.isfinite(snr_db):
        return "invalid"
    if snr_db >= SNR_DB_EXCELLENT:
        return "excellent"
    if snr_db >= SNR_DB_GOOD:
        return "good"
    if snr_db >= SNR_DB_NOISY:
        return "weak"
    return "noisy"


def _format_float(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "-inf" if value < 0 else "inf"
        return f"{value:.8g}"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_float(row.get(key, "")) for key in fieldnames})


def _finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _stats(values: Iterable[float]) -> dict[str, float]:
    finite = _finite(values)
    if not finite:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    array = np.array(finite, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def _summarize_sensors(rows: list[dict[str, Any]], sensor_labels: list[str]) -> list[dict[str, Any]]:
    by_sensor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["include_in_summary"] and math.isfinite(row["snr_db"]):
            by_sensor[row["sensor_label"]].append(row)

    summaries: list[dict[str, Any]] = []
    for sensor_label in sensor_labels:
        sensor_rows = by_sensor.get(sensor_label, [])
        snr_values = [row["snr_db"] for row in sensor_rows]
        snr_stats = _stats(snr_values)
        count = len(sensor_rows)
        summaries.append(
            {
                "sensor_label": sensor_label,
                "sensor_name": sensor_rows[0]["sensor_name"] if sensor_rows else sensor_label,
                "channel": sensor_rows[0]["channel"] if sensor_rows else "",
                "trial_count": count,
                "subject_count": len({row["subject_id"] for row in sensor_rows}),
                "mean_snr_db": snr_stats["mean"],
                "median_snr_db": snr_stats["median"],
                "p10_snr_db": snr_stats["p10"],
                "p90_snr_db": snr_stats["p90"],
                "min_snr_db": snr_stats["min"],
                "max_snr_db": snr_stats["max"],
                "mean_signal_rms": float(np.mean([row["signal_rms"] for row in sensor_rows])) if sensor_rows else float("nan"),
                "mean_noise_rms": float(np.mean([row["noise_rms"] for row in sensor_rows])) if sensor_rows else float("nan"),
                "noisy_trial_count": sum(row["snr_db"] < SNR_DB_NOISY for row in sensor_rows),
                "good_trial_count": sum(row["snr_db"] >= SNR_DB_GOOD for row in sensor_rows),
                "excellent_trial_count": sum(row["snr_db"] >= SNR_DB_EXCELLENT for row in sensor_rows),
                "noisy_fraction": (
                    sum(row["snr_db"] < SNR_DB_NOISY for row in sensor_rows) / count
                    if count
                    else float("nan")
                ),
                "median_status": _status(snr_stats["median"]),
            }
        )
    return summaries


def _summarize_subject_trials(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if math.isfinite(row["snr_db"]):
            grouped[(row["subject_id"], row["run_index"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (subject_id, run_index), trial_rows in sorted(grouped.items()):
        valid_rows = [row for row in trial_rows if row["include_in_summary"]]
        source_rows = valid_rows or trial_rows
        sorted_by_snr = sorted(source_rows, key=lambda row: row["snr_db"])
        snr_stats = _stats(row["snr_db"] for row in source_rows)
        noisy = [row["sensor_label"] for row in sorted_by_snr if row["snr_db"] < SNR_DB_NOISY]
        good = [row["sensor_label"] for row in sorted_by_snr if row["snr_db"] >= SNR_DB_GOOD]
        summaries.append(
            {
                "subject_id": subject_id,
                "run_index": run_index,
                "trial_number": run_index + 1,
                "skip_run": source_rows[0]["skip_run"] if source_rows else "",
                "included_in_summary": int(any(row["include_in_summary"] for row in trial_rows)),
                "sensor_count": len(source_rows),
                "median_snr_db": snr_stats["median"],
                "min_snr_db": snr_stats["min"],
                "max_snr_db": snr_stats["max"],
                "noisy_sensor_count": len(noisy),
                "good_sensor_count": len(good),
                "best_sensor": sorted_by_snr[-1]["sensor_label"] if sorted_by_snr else "",
                "best_sensor_snr_db": sorted_by_snr[-1]["snr_db"] if sorted_by_snr else float("nan"),
                "worst_sensor": sorted_by_snr[0]["sensor_label"] if sorted_by_snr else "",
                "worst_sensor_snr_db": sorted_by_snr[0]["snr_db"] if sorted_by_snr else float("nan"),
                "noisy_sensors": ";".join(noisy),
                "good_sensors": ";".join(good),
            }
        )
    return summaries


def _summarize_subject_sensors(rows: list[dict[str, Any]], sensor_labels: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["include_in_summary"] and math.isfinite(row["snr_db"]):
            grouped[(row["subject_id"], row["sensor_label"])].append(row)

    summaries: list[dict[str, Any]] = []
    subjects = sorted({row["subject_id"] for row in rows})
    for subject_id in subjects:
        for sensor_label in sensor_labels:
            sensor_rows = grouped.get((subject_id, sensor_label), [])
            snr_stats = _stats(row["snr_db"] for row in sensor_rows)
            summaries.append(
                {
                    "subject_id": subject_id,
                    "sensor_label": sensor_label,
                    "trial_count": len(sensor_rows),
                    "mean_snr_db": snr_stats["mean"],
                    "median_snr_db": snr_stats["median"],
                    "min_snr_db": snr_stats["min"],
                    "max_snr_db": snr_stats["max"],
                    "median_status": _status(snr_stats["median"]),
                }
            )
    return summaries


def _plot_sensor_summary(path: Path, sensor_summary: list[dict[str, Any]]) -> None:
    labels = [row["sensor_label"] for row in sensor_summary]
    values = [row["median_snr_db"] for row in sensor_summary]
    display_values = [value if math.isfinite(value) else 0.0 for value in values]
    colors = [
        "#2f6fbb" if value >= SNR_DB_GOOD else "#d08a22" if value >= SNR_DB_NOISY else "#b83a3a"
        for value in display_values
    ]

    fig_width = max(10.0, len(labels) * 0.48)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.bar(labels, display_values, color=colors)
    ax.axhline(SNR_DB_NOISY, color="#9a6500", linestyle="--", linewidth=1, label="weak/noisy threshold")
    ax.axhline(SNR_DB_GOOD, color="#245c2d", linestyle="--", linewidth=1, label="good threshold")
    ax.set_ylabel("Median SNR (dB)")
    ax.set_xlabel("Sensor")
    ax.set_title("Median walking-to-no-step SNR by sensor")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subject_heatmaps(
    output_dir: Path,
    rows: list[dict[str, Any]],
    sensor_labels: list[str],
) -> None:
    finite_values = _finite(row["snr_db"] for row in rows)
    if not finite_values:
        return
    vmin = float(np.percentile(finite_values, 5))
    vmax = float(np.percentile(finite_values, 95))
    if math.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    rows_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_subject[row["subject_id"]].append(row)

    for subject_id, subject_rows in sorted(rows_by_subject.items()):
        run_indices = sorted({row["run_index"] for row in subject_rows})
        values = np.full((len(run_indices), len(sensor_labels)), np.nan, dtype=np.float64)
        run_lookup = {run_index: idx for idx, run_index in enumerate(run_indices)}
        sensor_lookup = {sensor_label: idx for idx, sensor_label in enumerate(sensor_labels)}
        for row in subject_rows:
            if not math.isfinite(row["snr_db"]):
                continue
            values[run_lookup[row["run_index"]], sensor_lookup[row["sensor_label"]]] = row["snr_db"]

        fig_width = max(10.0, len(sensor_labels) * 0.46)
        fig_height = max(4.5, len(run_indices) * 0.32)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Subject {subject_id}: trial-by-sensor SNR")
        ax.set_xlabel("Sensor")
        ax.set_ylabel("Trial number")
        ax.set_xticks(np.arange(len(sensor_labels)), labels=sensor_labels, rotation=90)
        ax.set_yticks(np.arange(len(run_indices)), labels=[str(run_index + 1) for run_index in run_indices])
        fig.colorbar(image, ax=ax, label="SNR (dB)")
        fig.tight_layout()
        fig.savefig(output_dir / f"subject_{subject_id}_trial_sensor_snr_heatmap.png", dpi=180)
        plt.close(fig)


def _write_markdown_report(
    path: Path,
    dataset_root: Path,
    rows: list[dict[str, Any]],
    sensor_summary: list[dict[str, Any]],
) -> None:
    valid_rows = [row for row in rows if row["include_in_summary"] and math.isfinite(row["snr_db"])]
    subjects = sorted({row["subject_id"] for row in rows})
    trials = sorted({(row["subject_id"], row["run_index"]) for row in rows})
    ranked = sorted(
        [row for row in sensor_summary if math.isfinite(row["median_snr_db"])],
        key=lambda row: row["median_snr_db"],
        reverse=True,
    )
    best = ranked[:5]
    worst = list(reversed(ranked[-5:]))

    lines = [
        "# Sensor SNR Report",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Subjects scanned: {', '.join(subjects)}",
        f"- Subject/trial pairs scanned: {len(trials)}",
        f"- Sensor/trial rows included in summaries: {len(valid_rows)}",
        "",
        "SNR definition: each channel is demeaned inside the interval being measured. "
        "`snr_db = 10 * log10(walking_power / no_step_power)`, where no-step timing "
        "comes from `runPeramiters.csv` and walking timing comes from `Data Start (s)` "
        "to `Data End (s)`; `Data End (s)=0` means the end of the HDF5 record.",
        "",
        "Interpretation bands used in the CSVs: noisy `< 3 dB`, weak `3-10 dB`, "
        "good `10-20 dB`, excellent `>= 20 dB`.",
        "",
        "## Strongest Sensors By Median SNR",
        "",
    ]
    lines.extend(
        f"- `{row['sensor_label']}`: {row['median_snr_db']:.2f} dB median, "
        f"{row['noisy_fraction']:.0%} noisy trials"
        for row in best
    )
    lines.extend(["", "## Noisiest Sensors By Median SNR", ""])
    lines.extend(
        f"- `{row['sensor_label']}`: {row['median_snr_db']:.2f} dB median, "
        f"{row['noisy_fraction']:.0%} noisy trials"
        for row in worst
    )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `sensor_trial_snr.csv`: one row per subject, trial, and sensor/channel.",
            "- `sensor_summary.csv`: aggregate sensor ranking across non-skipped trials.",
            "- `subject_trial_summary.csv`: best/worst/noisy sensor lists for each trial.",
            "- `subject_sensor_summary.csv`: per-subject sensor medians across trials.",
            "- `sensor_summary_median_snr.png`: median SNR bar chart.",
            "- `subject_*_trial_sensor_snr_heatmap.png`: one heatmap per subject.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else config.data.dataset_root
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sensor_order = list(config.data.sensor_channel_map.keys())
    if args.selected_only:
        sensor_order = list(config.data.selected_sensors)
    sensor_pairs = _sensor_channel_pairs(config.data.sensor_channel_map, sensor_order)
    sensor_labels = [label for _, _, label in sensor_pairs]

    run_segments = _load_run_segments(dataset_root / "runPeramiters.csv")
    hdf5_files = sorted((dataset_root / "vib" / "Data").glob("*.hdf5"))
    if args.subject:
        requested = {f"{int(subject):03d}" for subject in args.subject}
        hdf5_files = [path for path in hdf5_files if _subject_id_from_hdf5(path) in requested]

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for hdf5_path in hdf5_files:
        subject_id = _subject_id_from_hdf5(hdf5_path)
        apdm_csv = _find_apdm_csv(dataset_root, subject_id)
        speeds: list[float] = []
        if apdm_csv is not None:
            try:
                speeds = _parse_speed_labels(apdm_csv)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Could not parse APDM speeds for subject {subject_id}: {exc}")

        with h5py.File(hdf5_path, "r") as handle:
            dataset = handle["experiment/data"]
            general_parameters = handle["experiment/general_parameters"][:]
            fs = _get_general_parameter(general_parameters, "fs")
            if fs is None:
                raise ValueError(f"Missing sample rate in {hdf5_path}")
            record_seconds = dataset.shape[-1] / fs
            hdf5_sensor_metadata = _sensor_rows(handle)

            run_indices = range(dataset.shape[0])
            if args.run is not None:
                requested_runs = {run - 1 if args.one_based_run else run for run in args.run}
                run_indices = [idx for idx in run_indices if idx in requested_runs]

            for run_index in run_indices:
                segment = run_segments.get((subject_id, int(run_index)))
                if segment is None:
                    warnings.append(f"No runPeramiters.csv row for subject {subject_id}, run {run_index}")
                    continue

                no_step_start = float(segment["no_step_start"])
                no_step_end = float(segment["no_step_end"])
                data_start = float(segment["data_start"])
                data_end = float(segment["data_end"]) or record_seconds
                no_step_slice = _segment_slice(no_step_start, no_step_end, fs, dataset.shape[-1])
                signal_slice = _segment_slice(data_start, data_end, fs, dataset.shape[-1])
                if no_step_slice is None or signal_slice is None:
                    warnings.append(f"Invalid segment timing for subject {subject_id}, run {run_index}")
                    continue

                trial = dataset[run_index, :, :]
                speed = speeds[run_index] if run_index < len(speeds) else float("nan")
                skip_run = int(segment["skip_run"])
                include_in_summary = args.include_skipped or skip_run == 0

                for sensor_name, channel, sensor_label in sensor_pairs:
                    channel_index = channel - 1
                    signal = trial[channel_index, signal_slice]
                    noise = trial[channel_index, no_step_slice]
                    signal_power, signal_rms, signal_peak_to_peak = _demeaned_power(signal)
                    noise_power, noise_rms, noise_peak_to_peak = _demeaned_power(noise)
                    snr_db = _db_ratio(signal_power, noise_power)
                    excess_snr_db = _db_ratio(max(signal_power - noise_power, 0.0), noise_power)
                    sensor_meta = hdf5_sensor_metadata.get(channel, {})
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "run_index": int(run_index),
                            "trial_number": int(run_index) + 1,
                            "sensor_name": sensor_name,
                            "sensor_label": sensor_label,
                            "channel": channel,
                            "daq_channel": sensor_meta.get("daq_channel", ""),
                            "model": sensor_meta.get("model", ""),
                            "serial": sensor_meta.get("serial", ""),
                            "units": sensor_meta.get("units", ""),
                            "location_x": sensor_meta.get("location_x", ""),
                            "location_y": sensor_meta.get("location_y", ""),
                            "location_z": sensor_meta.get("location_z", ""),
                            "direction_x": sensor_meta.get("direction_x", ""),
                            "direction_y": sensor_meta.get("direction_y", ""),
                            "direction_z": sensor_meta.get("direction_z", ""),
                            "fs_hz": float(fs),
                            "record_seconds": float(record_seconds),
                            "no_step_start_s": no_step_start,
                            "no_step_end_s": no_step_end,
                            "data_start_s": data_start,
                            "data_end_s": data_end,
                            "no_step_samples": int(no_step_slice.stop - no_step_slice.start),
                            "signal_samples": int(signal_slice.stop - signal_slice.start),
                            "speed_mps": speed,
                            "skip_run": skip_run,
                            "include_in_summary": int(include_in_summary),
                            "noise_rms": noise_rms,
                            "signal_rms": signal_rms,
                            "rms_ratio": signal_rms / noise_rms if noise_rms > 0.0 else float("nan"),
                            "noise_power": noise_power,
                            "signal_power": signal_power,
                            "snr_db": snr_db,
                            "excess_snr_db": excess_snr_db,
                            "noise_peak_to_peak": noise_peak_to_peak,
                            "signal_peak_to_peak": signal_peak_to_peak,
                            "status": _status(snr_db),
                            "hdf5_file": str(hdf5_path.relative_to(ROOT) if hdf5_path.is_relative_to(ROOT) else hdf5_path),
                        }
                    )

    sensor_summary = _summarize_sensors(rows, sensor_labels)
    subject_trial_summary = _summarize_subject_trials(rows)
    subject_sensor_summary = _summarize_subject_sensors(rows, sensor_labels)

    trial_fields = [
        "subject_id",
        "run_index",
        "trial_number",
        "sensor_name",
        "sensor_label",
        "channel",
        "daq_channel",
        "model",
        "serial",
        "units",
        "location_x",
        "location_y",
        "location_z",
        "direction_x",
        "direction_y",
        "direction_z",
        "fs_hz",
        "record_seconds",
        "no_step_start_s",
        "no_step_end_s",
        "data_start_s",
        "data_end_s",
        "no_step_samples",
        "signal_samples",
        "speed_mps",
        "skip_run",
        "include_in_summary",
        "noise_rms",
        "signal_rms",
        "rms_ratio",
        "noise_power",
        "signal_power",
        "snr_db",
        "excess_snr_db",
        "noise_peak_to_peak",
        "signal_peak_to_peak",
        "status",
        "hdf5_file",
    ]
    sensor_summary_fields = [
        "sensor_label",
        "sensor_name",
        "channel",
        "trial_count",
        "subject_count",
        "mean_snr_db",
        "median_snr_db",
        "p10_snr_db",
        "p90_snr_db",
        "min_snr_db",
        "max_snr_db",
        "mean_signal_rms",
        "mean_noise_rms",
        "noisy_trial_count",
        "good_trial_count",
        "excellent_trial_count",
        "noisy_fraction",
        "median_status",
    ]
    subject_trial_fields = [
        "subject_id",
        "run_index",
        "trial_number",
        "skip_run",
        "included_in_summary",
        "sensor_count",
        "median_snr_db",
        "min_snr_db",
        "max_snr_db",
        "noisy_sensor_count",
        "good_sensor_count",
        "best_sensor",
        "best_sensor_snr_db",
        "worst_sensor",
        "worst_sensor_snr_db",
        "noisy_sensors",
        "good_sensors",
    ]
    subject_sensor_fields = [
        "subject_id",
        "sensor_label",
        "trial_count",
        "mean_snr_db",
        "median_snr_db",
        "min_snr_db",
        "max_snr_db",
        "median_status",
    ]

    _write_csv(output_dir / "sensor_trial_snr.csv", rows, trial_fields)
    _write_csv(output_dir / "sensor_summary.csv", sensor_summary, sensor_summary_fields)
    _write_csv(output_dir / "subject_trial_summary.csv", subject_trial_summary, subject_trial_fields)
    _write_csv(output_dir / "subject_sensor_summary.csv", subject_sensor_summary, subject_sensor_fields)

    if not args.no_plots:
        _plot_sensor_summary(output_dir / "sensor_summary_median_snr.png", sensor_summary)
        _plot_subject_heatmaps(output_dir, rows, sensor_labels)

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "hdf5_files": [str(path) for path in hdf5_files],
        "sensor_count": len(sensor_pairs),
        "row_count": len(rows),
        "subject_count": len({row["subject_id"] for row in rows}),
        "trial_count": len({(row["subject_id"], row["run_index"]) for row in rows}),
        "included_row_count": sum(row["include_in_summary"] for row in rows),
        "warnings": warnings,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    _write_markdown_report(output_dir / "README.md", dataset_root, rows, sensor_summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate no-step-baseline SNR for every vibration sensor and trial."
    )
    parser.add_argument("--config", default="configs/regression.yaml", help="Config with dataset root and sensor map.")
    parser.add_argument("--dataset-root", default=None, help="Override the dataset root from the config.")
    parser.add_argument("--output-dir", default="artifacts/sensor_snr", help="Directory for CSVs and plots.")
    parser.add_argument("--subject", action="append", help="Optional subject filter, e.g. --subject 003.")
    parser.add_argument("--run", action="append", type=int, help="Optional run/trial filter. Zero-based unless --one-based-run is set.")
    parser.add_argument("--one-based-run", action="store_true", help="Treat --run values as one-based trial numbers.")
    parser.add_argument("--selected-only", action="store_true", help="Analyze only config.data.selected_sensors instead of every mapped sensor.")
    parser.add_argument("--include-skipped", action="store_true", help="Include runPeramiters.csv Skip Run=1 trials in summaries.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSVs only.")
    args = parser.parse_args()

    summary = analyze(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
