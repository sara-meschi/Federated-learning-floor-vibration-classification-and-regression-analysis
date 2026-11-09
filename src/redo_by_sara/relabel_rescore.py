"""Re-score saved S5 classification predictions under (source, subject) identities.

PI ground truth (2026-11-08): every (source, subject) pair is its own identity, and the
one person recorded in both campaigns is linked only as data
(``configs/identity_links.yaml``). The S5 models had 8 outputs: ``no_walking`` plus one per
*subject ID*, so a subject ID recorded in both sources shared one output.

A model trained on **one source only** never saw the other source's holder of a shared
ID, so each of its outputs names exactly one identity. Its saved predictions can be
re-scored with no retraining. The rule, derived from the data with no subject ID
hard-coded: an output ``subject_X`` maps to the identity (s, X) with s among the model's
training sources, or else to the only identity with subject X. Anything ambiguous is an
error. Models trained on both sources can't be re-scored this way and are rerun.

Re-scoring relabels the same output units, so votes, tie-breaks (by run-mean probability,
then output index) and mean-probability argmaxes are unchanged. Only the names, the true
labels and so the correctness change. Run-level predictions are read from the S5
``run_level_predictions.csv`` files, which were themselves recomputed from each
checkpoint and asserted against the saved window predictions.

What this module produces, under ``--out`` (default ``artifacts/relabel/rescore``):

- ``local_only_k2_natural/``: both local-only natural K=2 clients, run-level and window
  accuracy on the 28 canonical test runs;
- ``cross_session_control/``: the S5 control model (trained on one source only),
  redefined: a cross-session run is a run of an identity *linked* to one of the model's
  classes, and its target is the linked identity. Every other-source identity with no
  link is a negative control. In-session results are per identity;
- ``reverse_direction/``: each local-only natural client on the other source's canonical
  test runs: a linked identity counts as a hit when predicted as its link; an unlinked
  identity counts as a false match when predicted as any linked label;
- ``reverse_direction_all_runs_supplementary/``: the same analysis over **every** run of
  the other source (canonical train and test; the client never saw that source), by
  inference from the saved client checkpoint. Labelled supplementary.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .identities import (
    DEFAULT_LINKS_PATH,
    WALKING_SUFFIX,
    identity_id,
    linked_identities,
    load_identity_links,
    split_identity,
    walking_class_name,
)
from .run_level_classification import (
    BACKGROUND_CLASSES,
    PREDICTIONS_FILENAME,
    RUN_PREDICTIONS_FILENAME,
    aggregate_runs,
    load_classifier,
    read_prediction_rows,
    summarize_runs,
    wilson_interval,
    window_probabilities,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_S5_ROOT = ROOT / "artifacts/s5"
DEFAULT_OUT = ROOT / "artifacts/relabel/rescore"
CANONICAL_CLASSIFICATION_ARTIFACT = (
    ROOT / "artifacts/centralized_combined_9ch_no006_nowalking_80_20_60e/raw_windows_combined.pt"
)
LOCAL_ONLY_NATURAL_JOB = "local_only/k2_natural/seed4601/classification"
CONTROL_JOB = "subject003_control/seed4601"
LOCAL_ONLY_CHECKPOINT = "flower_iid_global_classification_model.pt"
SUMMARY_FILENAME = "rescore_summary.json"
#: S5 class names: one output per subject ID, regardless of source.
OLD_CLASS = re.compile(r"^subject_(\d{3})_walking$")


# --------------------------------------------------------------------------------------
# the mapping and the relabel
# --------------------------------------------------------------------------------------


def old_class_mapping(
    class_names: Sequence[str], training_sources: Sequence[str], identities: Sequence[str]
) -> dict[str, str]:
    """Old S5 class name -> identity class name, for a model trained on ``training_sources``."""
    training_sources = set(training_sources)
    mapping: dict[str, str] = {}
    for name in class_names:
        if name in BACKGROUND_CLASSES:
            mapping[name] = name
            continue
        match = OLD_CLASS.match(name)
        if not match:
            raise ValueError(f"Not an S5 subject class name: {name!r}.")
        candidates = [i for i in identities if split_identity(i)[1] == match.group(1)]
        own = [i for i in candidates if split_identity(i)[0] in training_sources]
        chosen = own or candidates
        if len(chosen) != 1:
            raise AssertionError(
                f"Output {name!r} of a model trained on {sorted(training_sources)} names "
                f"{len(chosen)} identities {chosen}; it cannot be re-scored without retraining."
            )
        mapping[name] = chosen[0] + WALKING_SUFFIX
    if len(set(mapping.values())) != len(mapping):
        raise AssertionError(f"Two outputs map to one identity: {mapping}.")
    return mapping


def _bool(value: Any) -> bool | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, bool):
        return value
    return str(value) == "True"


def relabel_record(record: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    """One saved run-level record with identity labels and recomputed correctness.

    The true label is the run's own identity. A run whose identity has no output in the
    model can never be predicted correctly, so it counts as wrong, as it would in a
    9-output model that never trained that class. ``true_label_in_model_outputs`` says so.
    """
    true = walking_class_name(record["source_id"], record["subject_id"])
    known = true in set(mapping.values())
    counts = {mapping[k]: int(v) for k, v in json.loads(record["window_label_counts"]).items()}
    majority = mapping[record["majority_vote_label"]]
    mean_label = mapping[record["mean_probability_label"]]
    same_column = mapping.get(str(record["true_label"])) == true
    old_of_true = record.get("mean_probability_of_true")
    return {
        "run_uid": record["run_uid"],
        "source_id": record["source_id"],
        "subject_id": record["subject_id"],
        "identity": identity_id(record["source_id"], record["subject_id"]),
        "true_label": true,
        "num_windows": int(record["num_windows"]),
        "window_correct": counts.get(true, 0),
        "majority_vote_label": majority,
        "majority_vote_count": int(record["majority_vote_count"]),
        "majority_vote_tied": _bool(record["majority_vote_tied"]),
        "mean_probability_label": mean_label,
        "mean_probability_max": float(record["mean_probability_max"]),
        "mean_probability_of_true": float(old_of_true) if same_column and old_of_true not in (None, "", "None") else None,
        "window_label_counts": json.dumps(dict(sorted(counts.items()))),
        "majority_vote_correct": majority == true,
        "mean_probability_correct": mean_label == true,
        "true_label_in_model_outputs": known,
        "s5_true_label": record["true_label"],
        "s5_majority_vote_label": record["majority_vote_label"],
        "s5_majority_vote_correct": _bool(record.get("majority_vote_correct")),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _proportion(k: int, n: int) -> dict[str, Any]:
    low, high = wilson_interval(k, n)
    return {"k": k, "n": n, "proportion": k / n if n else float("nan"), "wilson95": [low, high]}


def identities_from_manifest(manifest_path: Path) -> list[str]:
    """Every (source, subject) identity with a usable (train or test) run."""
    return sorted(
        {identity_id(r["source_id"], r["subject_id"]) for r in _read_csv(manifest_path) if r["split"] in ("train", "test")}
    )


def training_sources_by_client(partition_summary_path: Path) -> dict[str, list[str]]:
    summary = json.loads(partition_summary_path.read_text())
    return {client["client_id"]: sorted(client["source_run_counts"]) for client in summary["clients"]}


# --------------------------------------------------------------------------------------
# (1) local-only natural K=2
# --------------------------------------------------------------------------------------


def window_accuracy(prediction_rows: Sequence[Mapping[str, str]], mapping: Mapping[str, str]) -> float:
    """All test windows, identity labels: background windows keep their class."""
    correct = 0
    for row in prediction_rows:
        true = row["true_class"] if row["true_class"] in BACKGROUND_CLASSES else walking_class_name(row["source_id"], row["subject_id"])
        correct += mapping[row["predicted_class"]] == true
    return correct / len(prediction_rows)


def rescore_local_only(job_dir: Path, identities: Sequence[str], out_dir: Path) -> dict[str, Any]:
    sources = training_sources_by_client(job_dir / "partitions" / "combined_iid_partition_summary.json")
    result = job_dir / "local_only" / "classification"
    clients: dict[str, Any] = {}
    for client_dir in sorted(p for p in result.iterdir() if p.is_dir() and p.name.startswith("client_")):
        client = client_dir.name
        if len(sources[client]) != 1:
            raise AssertionError(f"{client} trained on {sources[client]}; only single-source clients can be re-scored.")
        old_metrics = json.loads((client_dir / "run_level" / "run_level_metrics.json").read_text())
        records_in = _read_csv(client_dir / "run_level" / RUN_PREDICTIONS_FILENAME)
        # Every output, in index order: the confusion-matrix header. The prediction rows
        # alone name only the outputs that were predicted.
        with (client_dir / "test_confusion_matrix.csv").open(newline="") as handle:
            outputs = next(csv.reader(handle))[1:]
        mapping = old_class_mapping(outputs, sources[client], identities)
        records = [relabel_record(r, mapping) for r in records_in]
        summary = summarize_runs(records)
        windows = window_accuracy(read_prediction_rows(client_dir / PREDICTIONS_FILENAME), mapping)
        summary.update({
            "training_sources": sources[client],
            "output_mapping": mapping,
            "window_accuracy_all_test_windows": windows,
            "s5": {
                "majority_vote": {k: old_metrics["majority_vote"][k] for k in ("num_correct", "num_runs", "accuracy", "wilson95")},
                "mean_probability": {k: old_metrics["mean_probability"][k] for k in ("num_correct", "num_runs", "accuracy", "wilson95")},
                "window_accuracy_all_test_windows": old_metrics["window_accuracy_all_test_windows"],
            },
        })
        _write_csv(out_dir / client / RUN_PREDICTIONS_FILENAME, records)
        _write_csv(out_dir / client / "run_level_subject_source.csv", summary["per_subject_source"])
        _write_json(out_dir / client / "run_level_metrics.json", summary)
        clients[client] = summary

    def mean_std(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {"mean": float(array.mean()), "std": float(array.std(ddof=1)) if array.size > 1 else 0.0, "n": int(array.size)}

    overview = {
        "job": str(job_dir),
        "clients": {
            c: {
                "training_sources": s["training_sources"],
                "run_vote": f"{s['majority_vote']['num_correct']}/{s['majority_vote']['num_runs']}",
                "run_vote_wilson95": s["majority_vote"]["wilson95"],
                "run_meanprob": f"{s['mean_probability']['num_correct']}/{s['mean_probability']['num_runs']}",
                "run_meanprob_wilson95": s["mean_probability"]["wilson95"],
                "window_accuracy": s["window_accuracy_all_test_windows"],
                "s5_run_vote": f"{s['s5']['majority_vote']['num_correct']}/{s['s5']['majority_vote']['num_runs']}",
                "s5_window_accuracy": s["s5"]["window_accuracy_all_test_windows"],
            }
            for c, s in clients.items()
        },
        "run_vote_accuracy_over_clients": mean_std([s["majority_vote"]["accuracy"] for s in clients.values()]),
        "run_meanprob_accuracy_over_clients": mean_std([s["mean_probability"]["accuracy"] for s in clients.values()]),
        "window_accuracy_over_clients": mean_std([s["window_accuracy_all_test_windows"] for s in clients.values()]),
        "s5_run_vote_accuracy_over_clients": mean_std([s["s5"]["majority_vote"]["accuracy"] for s in clients.values()]),
    }
    _write_json(out_dir / "local_only_rescore_summary.json", overview)
    return overview


# --------------------------------------------------------------------------------------
# (2) the cross-session control, redefined by identity links
# --------------------------------------------------------------------------------------


def _label_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    windows: Counter[str] = Counter()
    for record in records:
        windows.update(json.loads(record["window_label_counts"]))
    total = sum(windows.values())
    return {
        "num_runs": len(records),
        "num_windows": total,
        "window_label_fractions": {k: v / total for k, v in sorted(windows.items())},
        "window_label_counts": dict(sorted(windows.items())),
        "runs_majority_vote": dict(sorted(Counter(r["majority_vote_label"] for r in records).items())),
        "runs_mean_probability": dict(sorted(Counter(r["mean_probability_label"] for r in records).items())),
    }


def rescore_control(
    control_dir: Path, identities: Sequence[str], links: Sequence[Sequence[str]], out_dir: Path
) -> dict[str, Any]:
    summary_s5 = json.loads((control_dir / "subject003_control_summary.json").read_text())
    train_source = summary_s5["train_source"]
    mapping = old_class_mapping(summary_s5["classes"], [train_source], identities)
    classes = set(mapping.values())
    records_in = _read_csv(control_dir / RUN_PREDICTIONS_FILENAME)

    rows: list[dict[str, Any]] = []
    for record in records_in:
        relabelled = relabel_record(record, mapping)
        identity = relabelled["identity"]
        if record["source_id"] == train_source:
            role, targets = "in_session", [relabelled["true_label"]]
        else:
            targets = [i + WALKING_SUFFIX for i in linked_identities(identity, links) if i + WALKING_SUFFIX in classes]
            role = "cross_session" if targets else "negative_control"
        relabelled.update({
            "control_set": role,
            "s5_control_set": record["control_set"],
            "target_labels": json.dumps(targets),
            "majority_vote_hit": relabelled["majority_vote_label"] in targets if targets else None,
            "mean_probability_hit": relabelled["mean_probability_label"] in targets if targets else None,
        })
        rows.append(relabelled)

    def of(role: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["control_set"] == role]

    cross = of("cross_session")
    if not cross:
        raise AssertionError("No other-source identity is linked to one of the control's classes.")
    cross_targets = sorted({t for r in cross for t in json.loads(r["target_labels"])})
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_identity[row["identity"]].append(row)

    negative = {}
    for identity, members in sorted(by_identity.items()):
        if members[0]["control_set"] != "negative_control":
            continue
        negative[identity] = {
            **_label_distribution(members),
            "runs_majority_vote_to_cross_session_target": {
                t: _proportion(sum(r["majority_vote_label"] == t for r in members), len(members)) for t in cross_targets
            },
            "runs_mean_probability_to_cross_session_target": {
                t: _proportion(sum(r["mean_probability_label"] == t for r in members), len(members)) for t in cross_targets
            },
        }
    negative_runs = of("negative_control")
    in_session = of("in_session")
    result = {
        "definition": (
            "Cross-session run: a run of an other-source identity linked (configs/identity_links.yaml) "
            "to one of the model's classes; hit = predicted as the linked identity. Negative control: "
            "every other-source identity with no such link. In-session: the training source's "
            "canonical test runs, scored against their own identity."
        ),
        "train_source": train_source,
        "output_mapping": mapping,
        "links": [list(group) for group in links],
        "cross_session": {
            "identities": sorted({r["identity"] for r in cross}),
            "targets": cross_targets,
            "majority_vote_hits": _proportion(sum(bool(r["majority_vote_hit"]) for r in cross), len(cross)),
            "mean_probability_hits": _proportion(sum(bool(r["mean_probability_hit"]) for r in cross), len(cross)),
            **_label_distribution(cross),
        },
        "negative_control": {
            "per_identity": negative,
            "pooled_runs_majority_vote_to_cross_session_target": {
                t: _proportion(sum(r["majority_vote_label"] == t for r in negative_runs), len(negative_runs)) for t in cross_targets
            },
            "pooled_runs_mean_probability_to_cross_session_target": {
                t: _proportion(sum(r["mean_probability_label"] == t for r in negative_runs), len(negative_runs)) for t in cross_targets
            },
        },
        "in_session": {
            **{k: v for k, v in summarize_runs(in_session).items() if k in ("majority_vote", "mean_probability", "per_subject_source")},
            "per_identity": {i: _label_distribution(m) for i, m in sorted(by_identity.items()) if m[0]["control_set"] == "in_session"},
        },
        "set_membership_matches_s5": all(r["control_set"] == r["s5_control_set"] for r in rows),
        "s5": {
            "cross_session_majority_vote": summary_s5["cross_session"]["majority_vote"]["num_correct"],
            "cross_session_runs": summary_s5["cross_session"]["majority_vote"]["num_runs"],
            "cross_session_s5_target": "the same subject ID in the training source (now known to be a different person)",
        },
    }
    _write_csv(out_dir / "rescored_run_predictions.csv", rows)
    _write_json(out_dir / "control_rescore_summary.json", result)
    return result


# --------------------------------------------------------------------------------------
# (3) reverse direction: single-source local-only clients on the other source
# --------------------------------------------------------------------------------------


def cross_source_report(
    records: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
    training_sources: Sequence[str],
    links: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Per other-source identity: what a single-source model predicts for it.

    A linked identity (same person as one of the model's classes) scores a **hit** when
    its run is predicted as that class. An unlinked identity (a different person) scores
    a **false match** when predicted as any class that is linked to an other-source
    identity: the model's "same person" label going to someone else.
    """
    trained = {v for v in mapping.values() if v not in BACKGROUND_CLASSES
               and split_identity(v.removesuffix(WALKING_SUFFIX))[0] in set(training_sources)}
    link_labels = sorted(
        label for label in trained
        if any(split_identity(other)[0] not in set(training_sources)
               for other in linked_identities(label.removesuffix(WALKING_SUFFIX), links))
    )
    by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["source_id"] not in set(training_sources):
            by_identity[record["identity"]].append(record)
    out: dict[str, Any] = {"training_sources": list(training_sources), "linked_labels_in_model": link_labels, "identities": {}}
    for identity, members in sorted(by_identity.items()):
        targets = sorted(i + WALKING_SUFFIX for i in linked_identities(identity, links) if i + WALKING_SUFFIX in trained)
        entry: dict[str, Any] = {**_label_distribution(members)}
        if targets:
            entry["relation"] = "same person as a model class"
            entry["targets"] = targets
            entry["majority_vote_hits"] = _proportion(sum(r["majority_vote_label"] in targets for r in members), len(members))
            entry["mean_probability_hits"] = _proportion(sum(r["mean_probability_label"] in targets for r in members), len(members))
            entry["window_hits"] = _proportion(sum(json.loads(r["window_label_counts"]).get(t, 0) for r in members for t in targets), entry["num_windows"])
        else:
            entry["relation"] = "different person from every model class"
            entry["majority_vote_false_matches"] = _proportion(sum(r["majority_vote_label"] in link_labels for r in members), len(members))
            entry["mean_probability_false_matches"] = _proportion(sum(r["mean_probability_label"] in link_labels for r in members), len(members))
            entry["window_false_matches"] = _proportion(sum(json.loads(r["window_label_counts"]).get(t, 0) for r in members for t in link_labels), entry["num_windows"])
        out["identities"][identity] = entry
    return out


def rescore_reverse_direction(
    job_dir: Path, identities: Sequence[str], links: Sequence[Sequence[str]], out_dir: Path, local_only_out: Path
) -> dict[str, Any]:
    """From the re-scored local-only natural predictions (28 canonical test runs)."""
    sources = training_sources_by_client(job_dir / "partitions" / "combined_iid_partition_summary.json")
    report = {"scope": "canonical test runs of the other source (saved predictions)", "clients": {}}
    for client, training in sorted(sources.items()):
        records = _read_csv(local_only_out / client / RUN_PREDICTIONS_FILENAME)
        mapping = json.loads((local_only_out / client / "run_level_metrics.json").read_text())["output_mapping"]
        report["clients"][client] = cross_source_report(records, mapping, training, links)
    _write_json(out_dir / "reverse_direction_summary.json", report)
    return report


def rescore_reverse_direction_all_runs(
    job_dir: Path,
    identities: Sequence[str],
    links: Sequence[Sequence[str]],
    out_dir: Path,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Supplementary: every run of the other source, by inference from each client's
    saved checkpoint (no training). The client never saw that source, so its canonical
    train runs are valid evaluation runs here. Windows in the canonical test split are
    asserted to reproduce the saved window predictions."""
    sources = training_sources_by_client(job_dir / "partitions" / "combined_iid_partition_summary.json")
    metadata = artifact["metadata"]
    names = artifact["index_to_class"]
    targets = artifact["classification_targets"].tolist()
    report = {
        "label": "supplementary (PI request 2026-11-08); inference only from the saved local-only checkpoints",
        "scope": "every walking window of every run of the other source, canonical train and test",
        "clients": {},
    }
    for client, training in sorted(sources.items()):
        client_dir = job_dir / "local_only" / "classification" / client
        classifier = load_classifier(client_dir / LOCAL_ONLY_CHECKPOINT)
        indices = [i for i, item in enumerate(metadata)
                   if item["source_id"] not in set(training) and names[targets[i]] not in BACKGROUND_CLASSES]
        probabilities = window_probabilities(classifier, artifact["samples"], indices)
        saved = {int(r["artifact_index"]): r["predicted_class"] for r in read_prediction_rows(client_dir / PREDICTIONS_FILENAME)}
        argmax = [classifier["class_names"][int(k)] for k in probabilities.argmax(axis=1)]
        checked = [(i, saved[i], a) for i, a in zip(indices, argmax) if i in saved]
        mismatched = [c for c in checked if c[1] != c[2]]
        if not checked or mismatched:
            raise AssertionError(f"{client}: checkpoint does not reproduce saved predictions ({len(mismatched)} of {len(checked)}).")
        rows = [{"artifact_index": str(i), "run_uid": metadata[i]["run_uid"], "source_id": metadata[i]["source_id"],
                 "subject_id": metadata[i]["subject_id"], "true_class": names[targets[i]]} for i in indices]
        records_old = aggregate_runs(rows, probabilities, classifier["class_names"], exclude_true=())
        mapping = old_class_mapping(classifier["class_names"], training, identities)
        split_of_run = {metadata[i]["run_uid"]: metadata[i]["split"] for i in indices}
        records = []
        for record in records_old:
            relabelled = relabel_record({k: ("" if v is None else v) for k, v in record.items()}, mapping)
            relabelled["canonical_split"] = split_of_run[record["run_uid"]]
            records.append(relabelled)
        entry = cross_source_report(records, mapping, training, links)
        entry["num_runs"] = len(records)
        entry["num_windows"] = len(indices)
        entry["canonical_test_windows_checked_against_saved_predictions"] = len(checked)
        report["clients"][client] = entry
        _write_csv(out_dir / client / RUN_PREDICTIONS_FILENAME, records)
    _write_json(out_dir / "reverse_direction_all_runs_summary.json", report)
    return report


# --------------------------------------------------------------------------------------
# everything
# --------------------------------------------------------------------------------------


def run_rescore(
    s5_root: Path = DEFAULT_S5_ROOT,
    out_root: Path = DEFAULT_OUT,
    links_path: Path = DEFAULT_LINKS_PATH,
    artifact_path: Path = CANONICAL_CLASSIFICATION_ARTIFACT,
) -> dict[str, Any]:
    import torch

    s5_root, out_root = Path(s5_root).resolve(), Path(out_root).resolve()
    if out_root == s5_root or s5_root in out_root.parents:
        raise ValueError(f"--out {out_root} is inside the S5 root; S5 results are never overwritten.")
    links = load_identity_links(links_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    identities = sorted({identity_id(m["source_id"], m["subject_id"]) for m in artifact["metadata"]})
    job_dir = s5_root / LOCAL_ONLY_NATURAL_JOB
    local_only = rescore_local_only(job_dir, identities, out_root / "local_only_k2_natural")
    control = rescore_control(s5_root / CONTROL_JOB, identities, links, out_root / "cross_session_control")
    reverse = rescore_reverse_direction(job_dir, identities, links, out_root / "reverse_direction", out_root / "local_only_k2_natural")
    supplementary = rescore_reverse_direction_all_runs(
        job_dir, identities, links, out_root / "reverse_direction_all_runs_supplementary", artifact
    )
    summary = {
        "inputs": {"s5_root": str(s5_root), "links": str(links_path), "artifact_for_inference": str(artifact_path)},
        "identities": identities,
        "links": [list(g) for g in links],
        "local_only_k2_natural": local_only,
        "cross_session_control": {k: control[k] for k in ("train_source", "cross_session", "set_membership_matches_s5", "s5")},
        "reverse_direction": reverse,
        "reverse_direction_all_runs_supplementary": supplementary,
    }
    _write_json(out_root / SUMMARY_FILENAME, summary)
    return summary
