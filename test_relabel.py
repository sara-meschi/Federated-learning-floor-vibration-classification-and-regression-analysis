"""Relabel (PI ground truth 2026-11-08): classification label = (source, subject) identity.

Exact counts and expected results are independent literals (CLAUDE.md), not imported
from the modules. The data tests read the identity-labelled artifact
(``artifacts/relabel/data``) and the saved S5 results, both gitignored; they skip where
those are absent.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import pytest

from redo_by_sara import relabel_queue, s5_queue
from redo_by_sara.combined_iid_fl_partitioning import (
    ClientPartition,
    _assert_iid_properties,
    _build_run_profiles,
    build_combined_partitions,
)
from redo_by_sara.identities import (
    identity_id,
    linked_identities,
    load_identity_links,
    split_identity,
    walking_class_name,
)
from redo_by_sara.relabel_rescore import old_class_mapping, relabel_record
from test_combined_iid_fl_partitioning import _synthetic_artifacts

ROOT = Path(__file__).resolve().parent
RELABEL_ARTIFACT = ROOT / "artifacts/relabel/data/classification/raw_windows_combined.pt"
CANONICAL_ARTIFACT = ROOT / "artifacts/centralized_combined_9ch_no006_nowalking_80_20_60e/raw_windows_combined.pt"
REGRESSION_ARTIFACT = ROOT / "artifacts/centralized_combined_regression_9ch_no006_walking_80_20_60e/raw_walking_windows_regression.pt"
S5_ROOT = ROOT / "artifacts/s5"

needs_relabel_data = pytest.mark.skipif(
    not (RELABEL_ARTIFACT.exists() and CANONICAL_ARTIFACT.exists() and REGRESSION_ARTIFACT.exists()),
    reason="identity-labelled or canonical artifacts absent (gitignored)",
)
needs_s5 = pytest.mark.skipif(
    not (S5_ROOT / "local_only/k2_natural/seed4601/classification/job.json").exists(),
    reason="saved S5 per-job results absent (gitignored)",
)

IDENTITIES = [
    "test_2:001", "test_2:002", "test_2:003",
    "testing_20251124:003", "testing_20251124:004", "testing_20251124:005",
    "testing_20251124:007", "testing_20251124:008",
]
S5_CLASSES = ["no_walking"] + [f"subject_{s}_walking" for s in ("001", "002", "003", "004", "005", "007", "008")]


# --- identities -----------------------------------------------------------------------


def test_identity_strings() -> None:
    assert identity_id("test_2", "2") == "test_2:002"
    assert identity_id("testing_20251124", 3) == "testing_20251124:003"
    assert walking_class_name("test_2", "003") == "test_2:003_walking"
    assert split_identity("testing_20251124:007") == ("testing_20251124", "007")
    with pytest.raises(ValueError):
        split_identity("test_2:7")


def test_the_repo_link_is_identity_strings_only() -> None:
    links = load_identity_links()
    assert links == [("test_2:002", "testing_20251124:003")]
    assert linked_identities("testing_20251124:003", links) == ["test_2:002"]
    assert linked_identities("test_2:003", links) == []
    text = (ROOT / "configs/identity_links.yaml").read_text()
    values = re.findall(r"\[(.*?)\]", text.split("same_person:")[1])
    for member in ",".join(values).split(","):
        assert re.fullmatch(r"[a-z0-9_]+:\d{3}", member.strip())


@pytest.mark.parametrize(
    "body, message",
    [
        ("same_person:\n  - [test_2:002]\n", "at least two"),
        ("same_person:\n  - [test_2:001, test_2:002]\n", "different sources"),
        ("same_person:\n  - [test_2:001, b:001]\n  - [test_2:001, c:001]\n", "two same-person links"),
    ],
)
def test_bad_links_are_rejected(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "links.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_identity_links(path)


def test_no_subject_id_is_hard_coded_in_the_relabel_modules() -> None:
    for module in ("identities", "relabel_rescore", "relabel_queue", "relabel_checks"):
        source = (ROOT / f"src/redo_by_sara/{module}.py").read_text()
        assert not re.search(r"""["']\d{3}["']""", source), module
        assert "test_2:0" not in source and "testing_20251124:0" not in source, module


def test_builder_label_logic_names_identities() -> None:
    source = (ROOT / "src/redo_by_sara/combined_centralized_classification.py").read_text()
    assert 'f"subject_{' not in source
    assert source.count("walking_class_name(") >= 3


def test_nothing_in_s6_reads_the_fields_that_merge_subjects() -> None:
    """PI 2026-11-08: these secondary regression fields key on subject alone, merging the
    two 003s. They stay (bitwise), but no paper table or figure may read them."""
    for path in (ROOT / "src/redo_by_sara/paper_figures.py", ROOT / "scripts/make_paper_figures.py"):
        source = path.read_text()
        assert "metrics_by_subject" not in source, path
        assert "speed_distribution_by_subject" not in source, path
        assert not re.search(r"(?<!source_)subject_train_run_mean", source), path


# --- partitions -----------------------------------------------------------------------


def test_dirichlet_groups_by_identity() -> None:
    classification, regression = _synthetic_artifacts()
    _, summary = build_combined_partitions(classification, regression, "dirichlet", 3, seed=4601, dirichlet_alpha=0.1)
    assert sorted(summary["dirichlet"]["identity_client_shares"]) == [
        "test_2:001", "test_2:002", "test_2:003", "testing_20251124:003", "testing_20251124:004",
    ]
    assert summary["dirichlet"]["min_identities_per_client"] == 2


def test_iid_check_is_per_identity_not_per_subject_id() -> None:
    """client_0 takes every testing_20251124:003 run. client_1 still holds a subject "003"
    (Test_2's), so the old subject-ID check passed this partition; the identity check
    rejects it: client_0 holds all of one identity's runs."""
    classification, regression = _synthetic_artifacts()
    profiles, *_ = _build_run_profiles(classification, regression)
    owners = {uid: 0 for uid, p in profiles.items() if p.identity == "testing_20251124:003"}
    by_stratum: dict[tuple, list[str]] = {}
    for uid, profile in sorted(profiles.items()):
        if uid not in owners:
            by_stratum.setdefault(profile.stratum, []).append(uid)
    # Interleave the other strata so both clients hold each of them, then balance.
    interleaved = [uid for group in zip(*by_stratum.values()) for uid in group]
    interleaved += [uid for uids in by_stratum.values() for uid in uids if uid not in interleaved]
    share = (len(profiles) // 2) - len(owners)
    owners.update({uid: 0 if position < share else 1 for position, uid in enumerate(interleaved)})
    sizes = Counter(owners.values())
    assert abs(sizes[0] - sizes[1]) <= 1
    partitions = [
        ClientPartition(f"client_{c}", tuple(u for u, o in owners.items() if o == c), (), ()) for c in (0, 1)
    ]
    assert {profiles[u].subject_id for u in partitions[1].run_uids} >= {"003"}
    with pytest.raises(AssertionError, match=r"client_0 holds every training run of identities \['testing_20251124:003'\]"):
        _assert_iid_properties(partitions, profiles, [])


def test_client_class_counts_use_the_artifact_class_names() -> None:
    classification, regression = _synthetic_artifacts()
    _, summary = build_combined_partitions(classification, regression, "natural", 2)
    counts = summary["clients"][0]["classification_class_window_counts"]
    assert set(counts) == {"no_walking", "subject_001_walking", "subject_002_walking", "subject_003_walking"}


# --- re-score mapping -------------------------------------------------------------------


def test_single_source_outputs_map_to_one_identity_each() -> None:
    test_2 = old_class_mapping(S5_CLASSES, ["test_2"], IDENTITIES)
    assert test_2["subject_003_walking"] == "test_2:003_walking"
    assert test_2["subject_004_walking"] == "testing_20251124:004_walking"
    assert test_2["no_walking"] == "no_walking"
    newer = old_class_mapping(S5_CLASSES, ["testing_20251124"], IDENTITIES)
    assert newer["subject_003_walking"] == "testing_20251124:003_walking"
    assert newer["subject_001_walking"] == "test_2:001_walking"


def test_a_model_trained_on_both_sources_cannot_be_rescored() -> None:
    with pytest.raises(AssertionError, match="cannot be re-scored"):
        old_class_mapping(S5_CLASSES, ["test_2", "testing_20251124"], IDENTITIES)


def _record(source: str, subject: str, majority: str, mean: str | None = None) -> dict[str, str]:
    return {
        "run_uid": f"{source}:{subject}:000", "source_id": source, "subject_id": subject,
        "true_label": f"subject_{subject}_walking", "num_windows": "4", "window_correct": "0",
        "majority_vote_label": majority, "majority_vote_count": "3", "majority_vote_tied": "False",
        "mean_probability_label": mean or majority, "mean_probability_max": "0.9",
        "mean_probability_of_true": "0.5", "window_label_counts": f'{{"{majority}": 3, "no_walking": 1}}',
        "majority_vote_correct": "True", "mean_probability_correct": "True",
    }


def test_relabel_record_scores_against_the_own_identity() -> None:
    mapping = old_class_mapping(S5_CLASSES, ["testing_20251124"], IDENTITIES)
    # A Test_2 subject-003 run labelled "003" by a 20251124-only model: right in S5, wrong now.
    other = relabel_record(_record("test_2", "003", "subject_003_walking"), mapping)
    assert other["true_label"] == "test_2:003_walking"
    assert other["majority_vote_label"] == "testing_20251124:003_walking"
    assert other["majority_vote_correct"] is False and other["s5_majority_vote_correct"] is True
    assert other["true_label_in_model_outputs"] is False
    assert other["mean_probability_of_true"] is None
    own = relabel_record(_record("testing_20251124", "003", "subject_003_walking"), mapping)
    assert own["majority_vote_correct"] is True and own["window_correct"] == 3
    assert own["mean_probability_of_true"] == 0.5


# --- queue ------------------------------------------------------------------------------


def test_relabel_job_matrix() -> None:
    jobs = relabel_queue.build_jobs()
    assert len(jobs) == 32
    assert len({j.job_id for j in jobs}) == 32
    assert dict(Counter(j.group for j in jobs)) == {1: 9, 2: 11, 3: 1, 4: 11}
    assert jobs[0].job_id == "centralized/classification/seed4601"
    group1 = [j.job_id for j in jobs if j.group == 1]
    assert group1 == [
        "centralized/classification/seed4601",
        "fl/k2_natural/seed4601/classification", "fl/k2_iid/seed4601/classification",
        "fl/k2_natural/seed4602/classification", "fl/k2_iid/seed4602/classification",
        "fl/k2_natural/seed4603/classification", "fl/k2_iid/seed4603/classification",
        "centralized/classification/seed4602", "centralized/classification/seed4603",
    ]
    regression = sorted(j.job_id for j in jobs if j.task == "regression")
    assert regression == [
        "fl/k3_dirichlet0.1/seed4601/regression", "fl/k3_dirichlet0.3/seed4601/regression",
        "local_only/k3_dirichlet0.1/seed4601/regression", "local_only/k3_dirichlet0.3/seed4601/regression",
    ]
    for family in ("fl", "local_only"):
        for alpha in ("0.1", "0.3"):
            seeds = sorted(j.seed for j in jobs if j.family == family and j.condition == f"k3_dirichlet{alpha}" and j.task == "classification")
            assert seeds == [4601, 4602, 4603]
        for condition in ("k3_iid", "k3_rho0.375", "k3_rho0"):
            assert [j.task for j in jobs if j.family == family and j.condition == condition] == ["classification"]
    assert [j.job_id for j in jobs if j.group == 3] == ["local_only/k2_iid/seed4601/classification"]
    # FL before local-only within K=3; alpha 0.1 before 0.3.
    group2 = [j.condition for j in jobs if j.group == 2]
    assert group2.index("k3_dirichlet0.1") < group2.index("k3_dirichlet0.3")
    assert all(j.family == "fl" for j in jobs if j.group == 2)
    assert all(j.family == "local_only" for j in jobs if j.group == 4)


def test_relabel_commands(tmp_path: Path) -> None:
    jobs = {j.job_id: j for j in relabel_queue.build_jobs()}
    reference = str(tmp_path / "centralized/classification/seed4601")
    fl_cls = relabel_queue.command(jobs["fl/k2_natural/seed4602/classification"], tmp_path, "py")
    assert "configs/relabel_k2_natural.yaml" in " ".join(fl_cls)
    assert f"centralized_classification_dir={reference}" in fl_cls and "init_seed=4602" in fl_cls
    assert fl_cls[fl_cls.index("--output-root") + 1] == str(tmp_path / "fl/k2_natural/seed4602/classification")
    lo_reg = relabel_queue.command(jobs["local_only/k3_dirichlet0.3/seed4601/regression"], tmp_path, "py")
    assert "--mode" in lo_reg and not any("centralized_classification_dir" in a for a in lo_reg)
    central = relabel_queue.command(jobs["centralized/classification/seed4603"], tmp_path, "py")
    assert "configs/relabel_centralized_classification_init4603.yaml" in " ".join(central)
    for job in jobs.values():
        config = Path(relabel_queue.command(job, tmp_path, "py")[3])
        assert config.exists(), config


def test_relabel_configs_carry_the_s5_setting_and_the_identity_artifact() -> None:
    from redo_by_sara.combined_iid_flower import load_combined_iid_flower_config

    expected = {
        "k2_natural": (2, "natural", None, None), "k2_iid": (2, "iid", None, None),
        "k3_iid": (3, "iid", None, None), "k3_rho0.375": (3, "iid", 0.375, None),
        "k3_rho0": (3, "iid", 0.0, None), "k3_dirichlet0.1": (3, "dirichlet", None, 0.1),
        "k3_dirichlet0.3": (3, "dirichlet", None, 0.3),
    }
    for condition, (k, scheme, rho, alpha) in expected.items():
        config = load_combined_iid_flower_config(ROOT / f"configs/relabel_{condition}.yaml")
        assert (config["num_clients"], config["partition_scheme"], config["channel_overlap"], config["dirichlet_alpha"]) == (k, scheme, rho, alpha)
        assert (config["aggregation"], config["local_epochs"], config["client_optimizer_state"], config["num_rounds"],
                config["client_num_cpus"], config["split_seed"], config["init_seed"]) == ("FedAvg", 5, "persistent", 60, 3.0, 4601, 4601)
        assert config["classification_artifact"] == RELABEL_ARTIFACT
        assert config["regression_artifact"] == REGRESSION_ARTIFACT
        assert config["centralized_classification_dir"] == ROOT / "artifacts/relabel/centralized/classification/seed4601"
        assert config["centralized_regression_dir"] == ROOT / "artifacts/centralized_combined_residual_runbalanced_9ch_no006_60e"
    from redo_by_sara.combined_centralized_classification import load_combined_config

    for seed in (4601, 4602, 4603):
        config = load_combined_config(ROOT / f"configs/relabel_centralized_classification_init{seed}.yaml")
        assert (config["split_seed"], config["init_seed"], config["epochs"]) == (4601, seed, 60)
        assert config["output_dir"] / config["artifact_name"] == RELABEL_ARTIFACT


def test_s5_disposition() -> None:
    rows = relabel_queue.s5_disposition_rows()
    assert len(rows) == 47
    assert dict(Counter(r["disposition"] for r in rows)) == {"kept": 21, "re-scored": 2, "rerun": 24}
    rescored = sorted(r["s5_job_id"] for r in rows if r["disposition"] == "re-scored")
    assert rescored == ["local_only/k2_natural/seed4601/classification", "subject003_control/seed4601"]
    kept = [r for r in rows if r["disposition"] == "kept"]
    assert sum(r["task"] == "regression" for r in kept) == 20 and sum(r["family"] == "source_probe" for r in kept) == 1
    # Every S5 job marked "rerun" is a relabel job; the relabel adds only alpha = 0.3.
    relabel_ids = {j.job_id for j in relabel_queue.build_jobs()}
    rerun = {r["s5_job_id"] for r in rows if r["disposition"] == "rerun"}
    assert rerun <= relabel_ids
    assert sorted(relabel_ids - rerun) == sorted(j for j in relabel_ids if "dirichlet0.3" in j)
    assert len(relabel_ids - rerun) == 8


def test_relabel_estimates_start_from_measured_s5_times(tmp_path: Path) -> None:
    jobs = relabel_queue.build_jobs()
    total = sum(relabel_queue.estimate_minutes(j, tmp_path, jobs) for j in jobs)
    assert total == pytest.approx(1371.5)


def test_result_comparison_classifies_json_differences() -> None:
    from redo_by_sara.relabel_checks import _classify, json_differences

    old = {"model_path": "/a/m.pt", "skill": 0.1, "audits": {"subjects": ["003"]}, "gate": {"passed": True}}
    new = {"model_path": "/b/m.pt", "skill": 0.1, "audits": {"identities": ["test_2:003"]},
           "gate": {"passed": True, "relative_output_spread": 1.18}}
    kinds = {k: _classify(k, a, b) for k, a, b in json_differences(old, new)}
    assert kinds == {"/model_path": "path", "/audits/identities": "label_field", "/audits/subjects": "label_field",
                     "/gate/relative_output_spread": "added_in_rerun"}
    changed = {k: _classify(k, a, b) for k, a, b in json_differences(old, {**new, "skill": 0.2})}
    assert changed["/skill"] == "other"
    removed = {k: _classify(k, a, b) for k, a, b in json_differences(old, {k: v for k, v in new.items() if k != "gate"})}
    assert removed["/gate"] == "other"


# --- data -------------------------------------------------------------------------------


@needs_relabel_data
def test_relabelled_artifact_labels_and_counts() -> None:
    import torch

    from redo_by_sara.relabel_checks import artifact_equality

    relabelled = torch.load(RELABEL_ARTIFACT, map_location="cpu", weights_only=False)
    canonical = torch.load(CANONICAL_ARTIFACT, map_location="cpu", weights_only=False)
    assert relabelled["index_to_class"] == ["no_walking"] + [f"{i}_walking" for i in IDENTITIES]
    assert relabelled["summary"]["identities"] == IDENTITIES
    assert (relabelled["summary"]["num_examples"], relabelled["summary"]["num_train"], relabelled["summary"]["num_test"]) == (1549, 1242, 307)
    counts = {k: (v["all"], v["train"], v["test"]) for k, v in relabelled["summary"]["class_window_counts"].items()}
    assert counts == {
        "no_walking": (345, 276, 69),
        "test_2:001_walking": (188, 156, 32),
        "test_2:002_walking": (194, 151, 43),
        "test_2:003_walking": (152, 124, 28),
        "testing_20251124:003_walking": (142, 114, 28),
        "testing_20251124:004_walking": (127, 102, 25),
        "testing_20251124:005_walking": (134, 109, 25),
        "testing_20251124:007_walking": (125, 98, 27),
        "testing_20251124:008_walking": (142, 112, 30),
    }
    # The S5 control matches subjects by ID; on identity labels it must refuse, not mis-score.
    from redo_by_sara.subject003_control import cross_session_control_split

    with pytest.raises(AssertionError, match="not among the control's classes"):
        cross_session_control_split(relabelled, "test_2")
    audit = artifact_equality(relabelled, canonical)
    assert audit["canonical_classes_split_by_relabel"] == {
        "subject_003_walking": ["test_2:003_walking", "testing_20251124:003_walking"]
    }
    assert audit["summary_keys_that_differ"] == ["class_names", "class_window_counts", "identities", "label_scheme"]


@needs_relabel_data
@needs_s5
def test_kept_assignments_are_byte_identical_to_s5_and_dirichlet_is_rebuilt() -> None:
    from redo_by_sara.relabel_checks import conditions_configs, load_artifact, partition_checks

    conditions = ("k2_iid", "k2_natural", "k3_iid", "k3_rho0.375", "k3_rho0", "k3_dirichlet0.1", "k3_dirichlet0.3")
    result = partition_checks(load_artifact(RELABEL_ARTIFACT), load_artifact(REGRESSION_ARTIFACT), conditions_configs(conditions), S5_ROOT)
    assert result["passed"] is True
    entries = result["conditions"]
    for condition, copies in (("k2_iid", 8), ("k2_natural", 8), ("k3_iid", 4), ("k3_rho0.375", 4), ("k3_rho0", 4)):
        assert entries[condition]["assignment_unchanged"] is True
        assert len(entries[condition]["s5_copies"]) == copies
    k2 = entries["k2_iid"]["iid_identity_properties"]["clients"]
    assert k2["client_0"]["identity_run_counts"] == {
        "test_2:001": 9, "test_2:002": 9, "test_2:003": 8, "testing_20251124:003": 6,
        "testing_20251124:004": 6, "testing_20251124:005": 6, "testing_20251124:007": 6, "testing_20251124:008": 6,
    }
    assert k2["client_1"]["identity_run_counts"] == {
        "test_2:001": 8, "test_2:002": 9, "test_2:003": 9, "testing_20251124:003": 6,
        "testing_20251124:004": 6, "testing_20251124:005": 7, "testing_20251124:007": 5, "testing_20251124:008": 6,
    }

    def shape(condition: str) -> tuple:
        clients = entries[condition]["clients"]
        return (tuple(c["num_runs"] for c in clients), tuple(c["num_identities"] for c in clients),
                tuple(c["num_classification_windows"] for c in clients), tuple(c["num_regression_windows"] for c in clients),
                entries[condition]["dirichlet"]["draws"])

    assert shape("k3_dirichlet0.1") == ((20, 17, 75), (3, 2, 6), (222, 179, 841), (181, 145, 640), 1)
    assert shape("k3_dirichlet0.3") == ((18, 21, 73), (3, 5, 7), (203, 228, 811), (163, 184, 619), 1)
    # The old 7-subject Dirichlet draw (21 / 15 / 76 runs) is gone.
    assert not any(entries["k3_dirichlet0.1"]["s5_copies_byte_identical_to_rebuild"].values())


@needs_relabel_data
@needs_s5
def test_rescore_of_saved_s5_predictions(tmp_path: Path) -> None:
    from redo_by_sara.relabel_rescore import run_rescore

    summary = run_rescore(S5_ROOT, tmp_path / "rescore")
    local = summary["local_only_k2_natural"]["clients"]
    assert (local["client_0"]["run_vote"], local["client_1"]["run_vote"]) == ("13/28", "15/28")
    assert (local["client_0"]["s5_run_vote"], local["client_1"]["s5_run_vote"]) == ("13/28", "18/28")
    assert summary["local_only_k2_natural"]["run_vote_accuracy_over_clients"]["mean"] == pytest.approx(0.5)

    control = summary["cross_session_control"]
    assert control["set_membership_matches_s5"] is True
    assert control["cross_session"]["identities"] == ["testing_20251124:003"]
    assert control["cross_session"]["targets"] == ["test_2:002_walking"]
    hits = control["cross_session"]["majority_vote_hits"]
    assert (hits["k"], hits["n"]) == (14, 15)

    def kn(entry: dict) -> tuple[int, int]:
        return (entry["k"], entry["n"])

    reverse = summary["reverse_direction"]["clients"]["client_1"]["identities"]
    assert kn(reverse["test_2:002"]["majority_vote_hits"]) == (0, 5)
    assert kn(reverse["test_2:003"]["majority_vote_false_matches"]) == (3, 4)
    assert kn(reverse["test_2:003"]["mean_probability_false_matches"]) == (2, 4)
    assert kn(reverse["test_2:001"]["majority_vote_false_matches"]) == (0, 4)

    allruns = summary["reverse_direction_all_runs_supplementary"]["clients"]["client_1"]
    assert allruns["num_runs"] == 65
    every = allruns["identities"]
    assert kn(every["test_2:002"]["majority_vote_hits"]) == (0, 23)
    assert kn(every["test_2:003"]["majority_vote_false_matches"]) == (19, 21)
    assert kn(every["test_2:003"]["mean_probability_false_matches"]) == (18, 21)
    assert kn(every["test_2:001"]["majority_vote_false_matches"]) == (0, 21)

    with (tmp_path / "rescore/cross_session_control/rescored_run_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    negative = [r for r in rows if r["control_set"] == "negative_control"]
    assert len(negative) == 60
    assert sum(r["majority_vote_label"] == "test_2:002_walking" for r in negative) == 59


def test_s5_queue_is_untouched_by_the_relabel() -> None:
    assert len(s5_queue.build_jobs()) == 47
