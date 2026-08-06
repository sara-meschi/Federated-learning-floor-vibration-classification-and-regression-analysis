from __future__ import annotations

from pathlib import Path

from redo_by_sara.combined_centralized_classification import (
    RunRecord,
    assign_run_splits,
    resample_ratio,
    window_start_indices,
)


def _record(source: str, subject: str, run_index: int, direction: str) -> RunRecord:
    return RunRecord(
        source_id=source,
        subject_id=subject,
        run_index=run_index,
        run_uid=f"{source}:{subject}:{run_index:03d}",
        hdf5_path=Path(f"{source}_{subject}.hdf5"),
        metadata_sample_rate=1652.0,
        effective_sample_rate=1706.667 if source == "test_2" else 1651.6129032258063,
        raw_num_samples=49560,
        no_step_start=1.0,
        no_step_end=8.0 if source == "test_2" else 7.0,
        data_start=10.0,
        data_end=20.0,
        skip_run=False,
        excluded=False,
        direction=direction,
    )


def test_correct_resampling_ratios() -> None:
    assert resample_ratio(1706.667, 400.0) == (15, 64)
    assert resample_ratio(1651.6129032258063, 400.0) == (31, 128)


def test_no_walking_window_counts() -> None:
    test_2_starts = window_start_indices(1.0, 8.0, 11616, 400.0, 5.0, 1.0)
    newer_starts = window_start_indices(1.0, 7.0, 12000, 400.0, 5.0, 1.0)
    assert len(test_2_starts) == 3
    assert len(newer_starts) == 2


def test_source_aware_80_20_run_split() -> None:
    counts = {
        ("test_2", "001"): 21,
        ("test_2", "002"): 23,
        ("test_2", "003"): 21,
        ("testing_20251124", "003"): 15,
        ("testing_20251124", "004"): 15,
        ("testing_20251124", "005"): 16,
        ("testing_20251124", "007"): 14,
        ("testing_20251124", "008"): 15,
    }
    records = []
    for (source, subject), count in counts.items():
        for run_index in range(count):
            direction = (
                "single_direction_unknown"
                if source == "test_2"
                else ("N_to_S" if run_index % 2 == 0 else "S_to_N")
            )
            records.append(_record(source, subject, run_index, direction))

    split_by_uid = assign_run_splits(records, seed=4601, test_ratio=0.2)
    assert sum(split == "train" for split in split_by_uid.values()) == 112
    assert sum(split == "test" for split in split_by_uid.values()) == 28
    assert set(split_by_uid) == {record.run_uid for record in records}

    subject_003_sources = {
        record.source_id
        for record in records
        if record.subject_id == "003" and split_by_uid[record.run_uid] == "test"
    }
    assert subject_003_sources == {"test_2", "testing_20251124"}
