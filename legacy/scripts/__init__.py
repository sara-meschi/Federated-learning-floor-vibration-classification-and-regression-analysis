"""Quarantined legacy scripts. Importing this package is an error, by design.

See ``legacy/README.md`` and ``docs/session0_findings.md`` A1-A6.
"""

from __future__ import annotations

raise ImportError(
    "legacy/scripts is quarantined and must not be imported or collected. Every number "
    "in the paper comes from the combined_* stack in src/redo_by_sara/. "
    "See legacy/README.md."
)
