"""Subject identities: one per (source, subject) pair.

PI ground truth (2026-11-08): no subject is merged across collection campaigns. The
classification label is the (source, subject) identity, so the same subject ID in two
sources is two classes, and the one person recorded in both campaigns is also two
classes. Identities are derived from the data; no subject ID is listed here.

Which identities are the same person is a separate fact, recorded as data in
``configs/identity_links.yaml``. Only evaluation controls read it, never the label logic.
No personal name is recorded anywhere; a link is identity strings only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LINKS_PATH = ROOT / "configs/identity_links.yaml"
WALKING_SUFFIX = "_walking"


def identity_id(source_id: Any, subject_id: Any) -> str:
    """``"{source}:{subject:03d}"``: the run_uid prefix, and the classification identity."""
    return f"{source_id}:{int(subject_id):03d}"


def walking_class_name(source_id: Any, subject_id: Any) -> str:
    """The classification class of an identity's walking windows."""
    return identity_id(source_id, subject_id) + WALKING_SUFFIX


def split_identity(identity: str) -> tuple[str, str]:
    source, separator, subject = identity.rpartition(":")
    if not separator or not source or len(subject) != 3 or not subject.isdigit():
        raise ValueError(f"Not an identity of the form source:NNN: {identity!r}.")
    return source, subject


def load_identity_links(path: str | Path = DEFAULT_LINKS_PATH) -> list[tuple[str, ...]]:
    """Groups of identities that are the same person, recorded in different sources."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    groups = [tuple(str(member) for member in group) for group in raw.get("same_person") or []]
    seen: set[str] = set()
    for group in groups:
        if len(group) < 2:
            raise ValueError(f"A same-person link needs at least two identities: {group}.")
        sources = [split_identity(member)[0] for member in group]
        if len(set(sources)) != len(sources):
            raise ValueError(f"A same-person link must join identities from different sources: {group}.")
        if seen & set(group):
            raise ValueError(f"An identity appears in two same-person links: {sorted(seen & set(group))}.")
        seen.update(group)
    return groups


def linked_identities(identity: str, links: Sequence[Sequence[str]]) -> list[str]:
    """The other identities recorded as the same person as ``identity`` (often none)."""
    for group in links:
        if identity in group:
            return sorted(member for member in group if member != identity)
    return []
