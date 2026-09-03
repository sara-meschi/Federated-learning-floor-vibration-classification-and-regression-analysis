"""Quarantined legacy pipeline. Importing this package is an error, by design.

Every number in the paper comes from the ``combined_*`` stack in ``src/redo_by_sara/``.
The modules under ``legacy/`` predate it and violate ``CLAUDE.md`` in ways that produce
wrong numbers silently rather than loudly. See ``legacy/README.md`` for the specifics and
``docs/session0_findings.md`` A1-A6 for the evidence.

Each module also carries its own guard, so running one directly as a script
(``python legacy/scripts/run_flower.py``) fails with the same message even though this
package ``__init__`` is never executed in that case.
"""

from __future__ import annotations

raise ImportError(
    "The legacy pipeline is quarantined and must not be imported. It contradicts "
    "CLAUDE.md (no Test_2 sampling-rate override, no subject-006 exclusion, run_uid "
    "without source_id, train/test split inside a run, windows owned by two clients, "
    "out-of-hallway channels sliced rather than masked). Every number in the paper "
    "comes from the combined_* stack in src/redo_by_sara/. See legacy/README.md."
)
