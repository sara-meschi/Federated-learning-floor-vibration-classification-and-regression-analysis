from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from scipy.signal import spectrogram, welch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def _clean(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _get_general_parameter(parameters: np.ndarray, name: str) -> float | None:
    matches = parameters[parameters["parameter"] == name.encode()]
    if len(matches) == 0:
        return None
    return float(_clean(matches["value"][0]))


def _sensor_metadata(handle: h5py.File) -> list[dict[str, Any]]:
    if "experiment/sensors" not in handle:
        return [{"sensor_label": str(idx + 1), "channel": idx + 1} for idx in range(20)]

    rows = []
    for idx, row in enumerate(handle["experiment/sensors"][:], start=1):
        rows.append(
            {
                "sensor_label": str(idx),
                "channel": idx,
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
        )
    return rows


def _format_float(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.8g}"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_float(row.get(key, "")) for key in fieldnames})


def _find_common_stomp(data: np.ndarray, fs: float) -> dict[str, float | int]:
    centered = data - data.mean(axis=1, keepdims=True)
    energy_trace = np.sqrt(np.mean(centered**2, axis=0))
    stomp_index = int(np.argmax(energy_trace))
    return {"stomp_sample": stomp_index, "stomp_time_s": stomp_index / fs}


def _auto_walking_window(
    signal: np.ndarray,
    fs: float,
    stomp_time_s: float,
    win_s: float,
    step_s: float,
) -> dict[str, float | int]:
    """Find the strongest non-stomp walking window for one sensor."""
    centered = signal - np.mean(signal)
    win = int(round(win_s * fs))
    step = int(round(step_s * fs))
    starts = np.arange(0, centered.shape[0] - win + 1, step)
    rms = np.array([np.sqrt(np.mean(centered[start : start + win] ** 2)) for start in starts])
    centers = (starts + win / 2) / fs

    post_stomp = (starts / fs) >= stomp_time_s + 1.0
    if not np.any(post_stomp):
        best_idx = int(np.argmax(rms))
    else:
        masked = np.where(post_stomp, rms, -np.inf)
        best_idx = int(np.argmax(masked))

    start = int(starts[best_idx])
    end = int(start + win)
    return {
        "walking_start_sample": start,
        "walking_end_sample": end,
        "walking_start_s": start / fs,
        "walking_end_s": end / fs,
        "walking_center_s": (start + end) / (2 * fs),
        "walking_window_rms": float(rms[best_idx]),
    }


def _fft_amplitude(segment: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    segment = segment.astype(np.float64, copy=False)
    segment = segment - np.mean(segment)
    window = np.hanning(len(segment))
    spectrum = np.fft.rfft(segment * window)
    freq = np.fft.rfftfreq(len(segment), d=1.0 / fs)
    amplitude = 2.0 * np.abs(spectrum) / np.sum(window)
    return freq, amplitude


def _top_frequency(freq: np.ndarray, values: np.ndarray, min_freq: float, max_freq: float) -> tuple[float, float]:
    mask = (freq >= min_freq) & (freq <= max_freq)
    if not np.any(mask):
        return float("nan"), float("nan")
    local_freq = freq[mask]
    local_values = values[mask]
    idx = int(np.argmax(local_values))
    return float(local_freq[idx]), float(local_values[idx])


def _plot_overlay(
    path: Path,
    rows_by_sensor: dict[str, tuple[np.ndarray, np.ndarray]],
    ylabel: str,
    title: str,
    max_freq: float,
    semilogy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for sensor_label, (freq, values) in rows_by_sensor.items():
        mask = freq <= max_freq
        if semilogy:
            ax.semilogy(freq[mask], values[mask], linewidth=1, label=sensor_label)
        else:
            ax.plot(freq[mask], values[mask], linewidth=1, label=sensor_label)
    ax.set_xlim(0, max_freq)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(title="Sensor", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_sensor_frequency(
    path: Path,
    sensor_label: str,
    fft_freq: np.ndarray,
    fft_amp: np.ndarray,
    psd_freq: np.ndarray,
    psd: np.ndarray,
    max_freq: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    fft_mask = fft_freq <= max_freq
    psd_mask = psd_freq <= max_freq
    axes[0].plot(fft_freq[fft_mask], fft_amp[fft_mask], color="#1f5aa6")
    axes[0].set_ylabel("FFT amplitude")
    axes[0].set_title(f"Sensor {sensor_label}: frequency-domain walking window")
    axes[0].grid(True, alpha=0.35)
    axes[1].semilogy(psd_freq[psd_mask], psd[psd_mask], color="#a64d1f")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_spectrogram(
    path: Path,
    x: np.ndarray,
    fs: float,
    sensor_label: str,
    max_freq: float,
    nperseg: int,
) -> None:
    noverlap = int(round(nperseg * 0.75))
    freq, time, power = spectrogram(
        x - np.mean(x),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd",
    )
    mask = freq <= max_freq
    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.pcolormesh(time, freq[mask], 10 * np.log10(power[mask] + 1e-20), shading="gouraud")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Sensor {sensor_label}: full-trial spectrogram")
    fig.colorbar(image, ax=ax, label="PSD (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    hdf5_path = Path(args.hdf5_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_index = args.trial - 1 if args.one_based_trial else args.trial
    if run_index < 0:
        raise ValueError("Trial/run index must be non-negative.")

    with h5py.File(hdf5_path, "r") as handle:
        dataset = handle["experiment/data"]
        if run_index >= dataset.shape[0]:
            raise IndexError(f"Trial {args.trial} is outside available range 0..{dataset.shape[0] - 1}.")
        general_parameters = handle["experiment/general_parameters"][:]
        fs = _get_general_parameter(general_parameters, "fs")
        if fs is None:
            raise ValueError(f"Missing sampling frequency in {hdf5_path}.")
        data = dataset[run_index, :, :].astype(np.float64)
        sensors = _sensor_metadata(handle)

    detected_stomp = _find_common_stomp(data, fs)

    fft_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    fft_by_sensor: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    psd_by_sensor: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    window_samples = int(round(args.walking_window_seconds * fs))
    nperseg = min(args.nperseg, window_samples)
    if nperseg < 8:
        raise ValueError("Walking window is too short for frequency analysis.")
    noverlap = nperseg // 2

    for sensor in sensors:
        channel_index = int(sensor["channel"]) - 1
        sensor_label = str(sensor["sensor_label"])
        full_signal = data[channel_index]
        detected_window = _auto_walking_window(
            full_signal,
            fs,
            float(detected_stomp["stomp_time_s"]),
            args.walking_window_seconds,
            args.walking_step_seconds,
        )
        start = int(detected_window["walking_start_sample"])
        end = int(detected_window["walking_end_sample"])
        walking_signal = full_signal[start:end]

        fft_freq, fft_amp = _fft_amplitude(walking_signal, fs)
        psd_freq, psd = welch(
            walking_signal - np.mean(walking_signal),
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
        )

        fft_mask = fft_freq <= args.max_freq
        psd_mask = psd_freq <= args.max_freq
        for freq, amplitude in zip(fft_freq[fft_mask], fft_amp[fft_mask]):
            fft_rows.append({"sensor_label": sensor_label, "channel": sensor["channel"], "frequency_hz": float(freq), "amplitude": float(amplitude)})
        for freq, value in zip(psd_freq[psd_mask], psd[psd_mask]):
            psd_rows.append({"sensor_label": sensor_label, "channel": sensor["channel"], "frequency_hz": float(freq), "psd": float(value)})

        fft_by_sensor[sensor_label] = (fft_freq, fft_amp)
        psd_by_sensor[sensor_label] = (psd_freq, psd)
        fft_peak_freq, fft_peak_amp = _top_frequency(fft_freq, fft_amp, args.min_peak_freq, args.max_freq)
        psd_peak_freq, psd_peak_value = _top_frequency(psd_freq, psd, args.min_peak_freq, args.max_freq)
        summary_rows.append(
            {
                **sensor,
                "trial_number": args.trial if args.one_based_trial else args.trial + 1,
                "run_index": run_index,
                "fs_hz": float(fs),
                "stomp_time_s": detected_stomp["stomp_time_s"],
                "walking_start_s": detected_window["walking_start_s"],
                "walking_end_s": detected_window["walking_end_s"],
                "walking_center_s": detected_window["walking_center_s"],
                "fft_peak_frequency_hz": fft_peak_freq,
                "fft_peak_amplitude": fft_peak_amp,
                "psd_peak_frequency_hz": psd_peak_freq,
                "psd_peak": psd_peak_value,
                "walking_rms": float(np.sqrt(np.mean((walking_signal - np.mean(walking_signal)) ** 2))),
            }
        )

        _plot_sensor_frequency(
            output_dir / f"sensor_{sensor_label}_fft_psd.png",
            sensor_label,
            fft_freq,
            fft_amp,
            psd_freq,
            psd,
            args.max_freq,
        )
        if args.spectrograms:
            _plot_spectrogram(
                output_dir / f"sensor_{sensor_label}_spectrogram.png",
                full_signal,
                fs,
                sensor_label,
                args.max_freq,
                args.spectrogram_nperseg,
            )

    _write_csv(output_dir / "fft_amplitude.csv", fft_rows, ["sensor_label", "channel", "frequency_hz", "amplitude"])
    _write_csv(output_dir / "welch_psd.csv", psd_rows, ["sensor_label", "channel", "frequency_hz", "psd"])
    _write_csv(
        output_dir / "sensor_frequency_summary.csv",
        summary_rows,
        [
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
            "trial_number",
            "run_index",
            "fs_hz",
            "stomp_time_s",
            "walking_start_s",
            "walking_end_s",
            "walking_center_s",
            "walking_rms",
            "fft_peak_frequency_hz",
            "fft_peak_amplitude",
            "psd_peak_frequency_hz",
            "psd_peak",
        ],
    )

    _plot_overlay(
        output_dir / "all_sensors_fft_amplitude.png",
        fft_by_sensor,
        "FFT amplitude",
        "All sensors: walking-window FFT amplitude",
        args.max_freq,
    )
    _plot_overlay(
        output_dir / "all_sensors_welch_psd.png",
        psd_by_sensor,
        "PSD",
        "All sensors: walking-window Welch PSD",
        args.max_freq,
        semilogy=True,
    )

    summary = {
        "hdf5_path": str(hdf5_path),
        "output_dir": str(output_dir),
        "trial_number": args.trial if args.one_based_trial else args.trial + 1,
        "run_index": run_index,
        "sensor_count": len(sensors),
        "sample_count": int(data.shape[1]),
        "fs_hz": float(fs),
        "duration_s": float(data.shape[1] / fs),
        "max_plotted_frequency_hz": float(args.max_freq),
        "walking_window_mode": "per_sensor_strongest_post_stomp",
        "sensor_walking_start_s_min": float(min(row["walking_start_s"] for row in summary_rows)),
        "sensor_walking_start_s_max": float(max(row["walking_start_s"] for row in summary_rows)),
        "sensor_walking_end_s_min": float(min(row["walking_end_s"] for row in summary_rows)),
        "sensor_walking_end_s_max": float(max(row["walking_end_s"] for row in summary_rows)),
        **detected_stomp,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one trial from all floor sensors into frequency-domain outputs.")
    parser.add_argument("--hdf5-path", default="TestData/20251124_Testing/vib/Data/TestSetup_003.hdf5")
    parser.add_argument("--trial", type=int, default=1, help="Trial/run number. One-based by default.")
    parser.add_argument("--one-based-trial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="artifacts/frequency_domain_subject003_trial01")
    parser.add_argument("--max-freq", type=float, default=200.0)
    parser.add_argument("--min-peak-freq", type=float, default=0.5)
    parser.add_argument("--nperseg", type=int, default=2048)
    parser.add_argument("--walking-window-seconds", type=float, default=2.0)
    parser.add_argument("--walking-step-seconds", type=float, default=0.05)
    parser.add_argument("--spectrograms", action="store_true", help="Also save one full-trial spectrogram per sensor.")
    parser.add_argument("--spectrogram-nperseg", type=int, default=1024)
    args = parser.parse_args()

    print(json.dumps(analyze(args), indent=2))


if __name__ == "__main__":
    main()
