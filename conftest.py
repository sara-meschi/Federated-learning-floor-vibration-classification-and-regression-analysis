"""Repo-root pytest configuration.

Two jobs:

1. Make ``redo_by_sara`` importable. ``pyproject.toml`` sets ``pythonpath = ["src"]``,
   which is the real mechanism; the bootstrap below is a fallback for anyone invoking
   pytest in a way that bypasses the ini file.

2. Auto-skip ``@pytest.mark.requires_data`` tests when the data they need is absent.
   Those tests read the gitignored ``.pt`` artifacts or the 7.6 GB ``TestData/`` tree,
   neither of which is in the repository, so the fast subset has to run without them.
   Skipping on presence rather than behind an opt-in flag means the tests simply run
   wherever the data exists and skip cleanly where it does not — nobody has to remember
   a flag, and nobody gets a spurious failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


#: What ``requires_data`` actually requires. Kept as a mapping so a skip message can name
#: the specific missing path rather than saying "data missing".
REQUIRED_DATA_PATHS = {
    "TestData tree": ROOT / "TestData",
    "classification artifact": (
        ROOT
        / "artifacts/centralized_combined_9ch_no006_nowalking_80_20_60e"
        / "raw_windows_combined.pt"
    ),
    "regression artifact": (
        ROOT
        / "artifacts/centralized_combined_regression_9ch_no006_walking_80_20_60e"
        / "raw_walking_windows_regression.pt"
    ),
}


def _missing_data() -> list[str]:
    return [
        f"{label} ({path})"
        for label, path in REQUIRED_DATA_PATHS.items()
        if not path.exists()
    ]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    missing = _missing_data()
    if not missing:
        return
    skip = pytest.mark.skip(
        reason="requires_data: not present -> " + "; ".join(missing)
    )
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)
