"""Relabel step 2: prove what the identity relabel leaves unchanged, before rerunning.

Three checks, each written as JSON under ``artifacts/relabel/checks/``:

1. **Artifact equality.** The identity-labelled classification artifact against the
   canonical (S5) one: samples, split indices, normalization, run manifest and every
   metadata field except ``activity`` must be bitwise equal. Labels may differ only by
   naming each walking window's (source, subject) identity.
2. **Partitions.** Every K=2 and K=3 IID partition (and natural K=2) rebuilt by the
   current code from the identity-labelled artifact must reproduce, byte for byte, every
   S5 copy of ``combined_iid_run_to_client.csv`` for that condition. IID properties are
   reported per client over the 8 identities. Dirichlet partitions are rebuilt over
   identities, and their per-client run and identity counts and redraw counts reported.
3. **Result comparison.** A rerun result folder against the committed S5 one: every CSV
   and every tensor in every ``.pt`` file must be identical. JSON files are compared value
   by value. Listed but allowed: absolute paths, the documented label-naming fields, and
   keys the S5 record lacks because the code that writes them came after that job ran
   (``added_in_rerun``; a value present on both sides must still match). Any other
   difference fails.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .combined_iid_fl_partitioning import build_combined_partitions
from .identities import identity_id, walking_class_name

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CLASSIFICATION_ARTIFACT = (
    ROOT / "artifacts/centralized_combined_9ch_no006_nowalking_80_20_60e/raw_windows_combined.pt"
)
RELABEL_CLASSIFICATION_ARTIFACT = ROOT / "artifacts/relabel/data/classification/raw_windows_combined.pt"
REGRESSION_ARTIFACT = (
    ROOT / "artifacts/centralized_combined_regression_9ch_no006_walking_80_20_60e/raw_walking_windows_regression.pt"
)
S5_ROOT = ROOT / "artifacts/s5"
CHECKS_DIR = ROOT / "artifacts/relabel/checks"
RUN_TO_CLIENT = "combined_iid_run_to_client.csv"
#: Conditions whose run assignment must be unchanged (PI, 2026-11-08). The rho layouts
#: share the IID K=3 assignment; only their masks differ.
KEPT_ASSIGNMENT_CONDITIONS = ("k2_iid", "k2_natural", "k3_iid", "k3_rho0.375", "k3_rho0")
DIRICHLET_CONDITIONS = ("k3_dirichlet0.1", "k3_dirichlet0.3")
ABSENT = "<absent>"
#: Queue bookkeeping in an S5 job folder; not runner output.
QUEUE_FILES = ("job.json", "run.log")
#: JSON key fragments whose values name classes, subjects or identities: they change by
#: construction with the relabel, and are listed rather than failed.
LABEL_FIELD_MARKERS = (
    "classification_class_window_counts", "subjects", "identities", "subject_run_counts",
    "identity_run_counts", "every_client_holds_every", "no_client_holds_all_runs_of",
    "classification_artifact", "centralized_classification_dir", "config_path", "architecture",
)


# --------------------------------------------------------------------------------------
# 1. artifact equality
# --------------------------------------------------------------------------------------


def artifact_equality(relabelled: Mapping[str, Any], canonical: Mapping[str, Any]) -> dict[str, Any]:
    """Raises on any difference other than the identity labels; returns the audit."""
    for key in ("samples", "train_indices", "test_indices", "channel_mean", "channel_std"):
        if not torch.equal(relabelled[key], canonical[key]):
            raise AssertionError(f"Relabelled artifact differs from the canonical one in {key!r}.")
    if relabelled["run_manifest"] != canonical["run_manifest"]:
        raise AssertionError("Run manifests differ.")
    if len(relabelled["metadata"]) != len(canonical["metadata"]):
        raise AssertionError("Metadata lengths differ.")
    new_names, old_names = relabelled["index_to_class"], canonical["index_to_class"]
    new_targets = relabelled["classification_targets"].tolist()
    old_targets = canonical["classification_targets"].tolist()
    pairs: Counter[tuple[str, str]] = Counter()
    for index, (new, old) in enumerate(zip(relabelled["metadata"], canonical["metadata"])):
        if {k: v for k, v in new.items() if k != "activity"} != {k: v for k, v in old.items() if k != "activity"}:
            raise AssertionError(f"Metadata differs beyond 'activity' at window {index}.")
        new_class, old_class = new_names[new_targets[index]], old_names[old_targets[index]]
        if new["activity"] != new_class or old["activity"] != old_class:
            raise AssertionError(f"Window {index}: activity and target disagree.")
        expected = "no_walking" if old_class == "no_walking" else walking_class_name(new["source_id"], new["subject_id"])
        if new_class != expected:
            raise AssertionError(f"Window {index}: class {new_class!r}, expected {expected!r}.")
        pairs[(old_class, new_class)] += 1
    old_to_new: dict[str, set[str]] = {}
    for old_class, new_class in pairs:
        old_to_new.setdefault(old_class, set()).add(new_class)
    summary_differences = sorted(
        key for key in set(relabelled["summary"]) | set(canonical["summary"])
        if relabelled["summary"].get(key) != canonical["summary"].get(key)
    )
    return {
        "bitwise_equal": ["samples", "train_indices", "test_indices", "channel_mean", "channel_std",
                          "run_manifest", "metadata (every field except activity)"],
        "num_windows": len(new_targets),
        "num_train_windows": int(relabelled["train_indices"].numel()),
        "num_test_windows": int(relabelled["test_indices"].numel()),
        "canonical_classes": list(old_names),
        "relabelled_classes": list(new_names),
        "window_count_by_old_and_new_class": [
            {"canonical_class": o, "relabelled_class": n, "windows": c} for (o, n), c in sorted(pairs.items())
        ],
        "canonical_classes_split_by_relabel": {o: sorted(n) for o, n in sorted(old_to_new.items()) if len(n) > 1},
        "summary_keys_that_differ": summary_differences,
    }


# --------------------------------------------------------------------------------------
# 2. partitions
# --------------------------------------------------------------------------------------


def run_to_client_text(summary: Mapping[str, Any]) -> str:
    """The manifest exactly as ``save_combined_iid_partitions`` writes it."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["run_uid", "client_id"])
    writer.writeheader()
    for run_uid, client_id in sorted(summary["run_to_client"].items()):
        writer.writerow({"run_uid": run_uid, "client_id": client_id})
    return buffer.getvalue()


def s5_run_to_client_copies(condition: str, s5_root: Path = S5_ROOT) -> list[Path]:
    """Every saved S5 manifest for a condition: FL and local-only, every seed and task."""
    return sorted(s5_root.glob(f"*/{condition}/seed*/*/partitions/{RUN_TO_CLIENT}"))


def iid_identity_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Per client: runs and runs per identity; the two IID properties over identities."""
    totals: Counter[str] = Counter()
    for client in summary["clients"]:
        totals.update(client["identity_run_counts"])
    clients = {}
    for client in summary["clients"]:
        counts = client["identity_run_counts"]
        clients[client["client_id"]] = {
            "num_runs": client["num_runs"],
            "identity_run_counts": counts,
            "holds_every_identity": set(counts) == set(totals),
            "holds_all_runs_of": sorted(i for i, n in counts.items() if n == totals[i]),
        }
    return {
        "identity_training_run_totals": dict(sorted(totals.items())),
        "clients": clients,
        "every_client_holds_every_identity": all(c["holds_every_identity"] for c in clients.values()),
        "no_client_holds_all_runs_of_an_identity": not any(c["holds_all_runs_of"] for c in clients.values()),
    }


def partition_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    out = {
        "clients": [
            {
                "client_id": c["client_id"],
                "num_runs": c["num_runs"],
                "num_identities": len(c["identities"]),
                "identities": c["identities"],
                "identity_run_counts": c["identity_run_counts"],
                "num_classification_windows": c["num_classification_windows"],
                "num_regression_windows": c["num_regression_windows"],
            }
            for c in summary["clients"]
        ],
    }
    if "dirichlet" in summary:
        d = summary["dirichlet"]
        out["dirichlet"] = {k: d[k] for k in ("alpha", "min_identities_per_client", "draws", "floor_applied", "floor_note")}
        out["dirichlet"]["redraws"] = d["draws"] - 1
        out["dirichlet"]["identity_client_run_counts"] = d["identity_client_run_counts"]
    if "channel_availability" in summary:
        out["channel_positions"] = {c["client_id"]: c["positions"] for c in summary["channel_availability"]["clients"]}
    return out


def partition_checks(
    classification: Mapping[str, Any],
    regression: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    s5_root: Path = S5_ROOT,
) -> dict[str, Any]:
    """``configs``: condition -> loaded relabel config. Raises nothing; reports ``passed``."""
    result: dict[str, Any] = {"conditions": {}}
    passed = True
    for condition, config in configs.items():
        _, summary = build_combined_partitions(
            classification, regression, scheme=config["partition_scheme"], num_clients=config["num_clients"],
            seed=config["split_seed"], dirichlet_alpha=config["dirichlet_alpha"], channel_overlap=config["channel_overlap"],
        )
        text = run_to_client_text(summary)
        entry: dict[str, Any] = {"scheme": config["partition_scheme"], "num_clients": config["num_clients"],
                                 **partition_report(summary)}
        copies = s5_run_to_client_copies(condition, s5_root)
        entry["s5_copies"] = [str(p.relative_to(s5_root)) for p in copies]
        expected = text.encode("utf-8")
        entry["s5_copies_byte_identical_to_rebuild"] = {str(p.relative_to(s5_root)): p.read_bytes() == expected for p in copies}
        if condition in KEPT_ASSIGNMENT_CONDITIONS:
            ok = bool(copies) and all(entry["s5_copies_byte_identical_to_rebuild"].values())
            entry["assignment_must_be_unchanged"] = True
            entry["assignment_unchanged"] = ok
            passed &= ok
        if config["partition_scheme"] == "iid":
            entry["iid_identity_properties"] = iid_identity_report(summary)
            passed &= entry["iid_identity_properties"]["every_client_holds_every_identity"]
            passed &= entry["iid_identity_properties"]["no_client_holds_all_runs_of_an_identity"]
        result["conditions"][condition] = entry
    result["passed"] = bool(passed)
    return result


# --------------------------------------------------------------------------------------
# 3. result comparison
# --------------------------------------------------------------------------------------


def _same_object(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same_object(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_same_object(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def json_differences(a: Any, b: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                out.append((f"{path}/{key}", a.get(key, ABSENT), b.get(key, ABSENT)))
            else:
                out.extend(json_differences(a[key], b[key], f"{path}/{key}"))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in json_differences(x, y, f"{path}[{i}]")]
    return [] if a == b else [(path, a, b)]


def _classify(path: str, old: Any, new: Any) -> str:
    if isinstance(old, str) and isinstance(new, str) and old.startswith("/") and new.startswith("/"):
        return "path"
    if any(marker in path for marker in LABEL_FIELD_MARKERS):
        return "label_field"
    if old == ABSENT:
        return "added_in_rerun"
    return "other"


def compare_result_dirs(s5_dir: Path, rerun_dir: Path) -> dict[str, Any]:
    """File-by-file comparison of a rerun against a committed S5 job folder."""
    s5_dir, rerun_dir = Path(s5_dir), Path(rerun_dir)
    s5_files = {p.relative_to(s5_dir) for p in s5_dir.rglob("*") if p.is_file() and p.name not in QUEUE_FILES}
    rerun_files = {p.relative_to(rerun_dir) for p in rerun_dir.rglob("*") if p.is_file() and p.name not in QUEUE_FILES}
    report: dict[str, Any] = {
        "s5_dir": str(s5_dir), "rerun_dir": str(rerun_dir),
        "only_in_s5": sorted(map(str, s5_files - rerun_files)),
        "only_in_rerun": sorted(map(str, rerun_files - s5_files)),
        "identical": [], "differing": [], "png_not_compared": [], "json_differences": {},
    }
    for relative in sorted(s5_files & rerun_files):
        a, b = s5_dir / relative, rerun_dir / relative
        name = str(relative)
        if relative.suffix == ".png":
            report["png_not_compared"].append(name)
        elif relative.suffix == ".json":
            diffs = json_differences(json.loads(a.read_text()), json.loads(b.read_text()))
            if not diffs:
                report["identical"].append(name)
                continue
            classified = [{"key": k, "kind": _classify(k, x, y), "s5": x, "rerun": y} for k, x, y in diffs]
            report["json_differences"][name] = classified
            if any(d["kind"] == "other" for d in classified):
                report["differing"].append(name)
        elif relative.suffix == ".pt":
            same = _same_object(torch.load(a, map_location="cpu", weights_only=False),
                                torch.load(b, map_location="cpu", weights_only=False))
            (report["identical"] if same else report["differing"]).append(name)
        else:
            (report["identical"] if a.read_bytes() == b.read_bytes() else report["differing"]).append(name)
    kinds = Counter(d["kind"] for diffs in report["json_differences"].values() for d in diffs)
    report["json_difference_kinds"] = dict(kinds)
    report["passed"] = not report["differing"] and not report["only_in_s5"] and not report["only_in_rerun"]
    return report


def write_check(name: str, payload: Mapping[str, Any], checks_dir: Path = CHECKS_DIR) -> Path:
    checks_dir.mkdir(parents=True, exist_ok=True)
    path = checks_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_artifact(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def identity_window_counts(artifact: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Walking windows per identity and split, from metadata (a reporting helper)."""
    counts: dict[str, Counter[str]] = {}
    names = artifact["index_to_class"]
    targets = artifact["classification_targets"].tolist()
    for index, item in enumerate(artifact["metadata"]):
        if names[targets[index]] == "no_walking":
            continue
        counts.setdefault(identity_id(item["source_id"], item["subject_id"]), Counter())[item["split"]] += 1
    return {k: dict(v) for k, v in sorted(counts.items())}


def conditions_configs(conditions: Sequence[str]) -> dict[str, dict[str, Any]]:
    from .combined_iid_flower import load_combined_iid_flower_config

    return {c: load_combined_iid_flower_config(ROOT / f"configs/relabel_{c}.yaml") for c in conditions}
