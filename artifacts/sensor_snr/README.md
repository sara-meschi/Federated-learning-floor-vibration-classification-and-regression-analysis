# Sensor SNR Report

- Dataset root: `/home/coder/workspace/Federated-learning-floor-vibration-classification-and-regression-analysis/TestData/20251124_Testing`
- Subjects scanned: 003, 004, 005, 006, 007, 008
- Subject/trial pairs scanned: 96
- Sensor/trial rows included in summaries: 1780

SNR definition: each channel is demeaned inside the interval being measured. `snr_db = 10 * log10(walking_power / no_step_power)`, where no-step timing comes from `runPeramiters.csv` and walking timing comes from `Data Start (s)` to `Data End (s)`; `Data End (s)=0` means the end of the HDF5 record.

Interpretation bands used in the CSVs: noisy `< 3 dB`, weak `3-10 dB`, good `10-20 dB`, excellent `>= 20 dB`.

## Strongest Sensors By Median SNR

- `5`: 10.09 dB median, 2% noisy trials
- `6`: 8.61 dB median, 9% noisy trials
- `4`: 7.64 dB median, 16% noisy trials
- `2`: 4.53 dB median, 22% noisy trials
- `3`: 4.40 dB median, 26% noisy trials

## Noisiest Sensors By Median SNR

- `10`: -0.01 dB median, 100% noisy trials
- `9z`: 0.03 dB median, 100% noisy trials
- `9y`: 0.03 dB median, 100% noisy trials
- `9x`: 0.11 dB median, 100% noisy trials
- `16`: 0.17 dB median, 100% noisy trials

## Output Files

- `sensor_trial_snr.csv`: one row per subject, trial, and sensor/channel.
- `sensor_summary.csv`: aggregate sensor ranking across non-skipped trials.
- `subject_trial_summary.csv`: best/worst/noisy sensor lists for each trial.
- `subject_sensor_summary.csv`: per-subject sensor medians across trials.
- `sensor_summary_median_snr.png`: median SNR bar chart.
- `subject_*_trial_sensor_snr_heatmap.png`: one heatmap per subject.
