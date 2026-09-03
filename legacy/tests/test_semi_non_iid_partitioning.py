#!/usr/bin/env python
"""
Test script to verify semi non-IID partitioning with shared subjects works correctly.
This tests the partitioning logic without running the full federated simulation.
"""

_QUARANTINED_LEGACY_MODULE = "legacy/tests/test_semi_non_iid_partitioning.py"
raise ImportError(
    "Quarantined legacy module: legacy/tests/test_semi_non_iid_partitioning.py. It contradicts CLAUDE.md (see "
    "docs/session0_findings.md A1-A6) and would silently produce wrong numbers. "
    "Every number in the paper comes from the combined_* stack. Do not run it, "
    "import it, or copy from it."
)


import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from redo_by_sara.config import load_config
from redo_by_sara.federated import build_client_partitions, create_partition_summary


def test_semi_non_iid_partitioning():
    """Test semi non-IID partitioning with shared subjects."""
    
    # Load config
    config = load_config("configs/fl_semi_non_iid_classification.yaml")
    print(f"✓ Loaded config: {config.config_path}")
    
    # Load artifact
    artifact = torch.load(config.artifact_path, map_location="cpu", weights_only=False)
    print(f"✓ Loaded artifact: {config.artifact_path}")
    print(f"  - Global train: {len(artifact['train_indices'])} windows")
    print(f"  - Global val: {len(artifact['val_indices'])} windows")
    print(f"  - Global test: {len(artifact['test_indices'])} windows")
    
    # Build partitions
    client_partitions, resolved_subjects = build_client_partitions(
        artifact=artifact,
        num_clients=config.federated.num_clients,
        client_subjects=config.federated.client_subjects,
    )
    print(f"\n✓ Built {len(client_partitions)} client partitions")
    
    # Display partition summary
    print("\n=== Client Partitions ===")
    index_clients = {}
    
    for partition in client_partitions:
        print(f"\nClient {partition.client_id}:")
        print(f"  Subjects: {partition.subject_ids}")
        print(f"  Training windows: {len(partition.train_indices)}")
        
        for idx in partition.train_indices:
            index_clients.setdefault(int(idx), []).append(partition.client_id)
    
    # Check coverage
    global_train_set = set(int(idx) if isinstance(idx, torch.Tensor) else idx for idx in artifact["train_indices"])
    all_indices_int = set(index_clients)
    if all_indices_int == global_train_set:
        print(f"\n✓ Full coverage: {len(all_indices_int)} unique training windows")
    else:
        missing = global_train_set - all_indices_int
        extra = all_indices_int - global_train_set
        if missing:
            print(f"\n✗ Missing {len(missing)} windows")
        if extra:
            print(f"\n✗ Extra {len(extra)} windows")
    
    shared_indices = {
        idx: clients
        for idx, clients in index_clients.items()
        if len(clients) > 1
    }
    expected_shared_pattern = tuple(str(client_id) for client_id in range(config.federated.num_clients))
    
    # Show shared data info
    if shared_indices:
        print(f"\n✓ Shared data detected: {len(shared_indices)} windows shared across clients")
        print("  Shared indices appear in:")
        # Group by which clients share them
        sharing_patterns = {}
        for idx, clients in shared_indices.items():
            pattern = tuple(sorted(clients))
            if pattern not in sharing_patterns:
                sharing_patterns[pattern] = 0
            sharing_patterns[pattern] += 1
        
        for pattern, count in sharing_patterns.items():
            print(f"    - Clients {pattern}: {count} windows")
        unexpected_patterns = set(sharing_patterns) - {expected_shared_pattern}
        if unexpected_patterns:
            raise AssertionError(f"Unexpected shared-client patterns: {sorted(unexpected_patterns)}")
    else:
        print("\n⚠ No shared data detected (all subjects exclusive)")
    
    # Verify no inter-client overlap for exclusive subjects
    metadata = artifact["metadata"]
    subject_to_clients = {}
    for partition in client_partitions:
        for subject_id in partition.subject_ids:
            if subject_id not in subject_to_clients:
                subject_to_clients[subject_id] = []
            subject_to_clients[subject_id].append(partition.client_id)
    
    shared_subjects = {s: c for s, c in subject_to_clients.items() if len(c) > 1}
    print(f"\n✓ Shared subjects: {shared_subjects if shared_subjects else 'None'}")
    
    # Create and display partition summary
    summary = create_partition_summary(artifact, client_partitions, resolved_subjects)
    print("\n=== Partition Summary ===")
    print(json.dumps(summary, indent=2))
    
    # Verify data integrity
    print("\n=== Data Integrity Checks ===")
    
    # Check train/val/test no overlap
    train_indices = set()
    for partition in client_partitions:
        train_indices.update(int(idx) if isinstance(idx, torch.Tensor) else idx for idx in partition.train_indices)
    
    val_indices = set(int(idx) if isinstance(idx, torch.Tensor) else idx for idx in artifact["val_indices"])
    test_indices = set(int(idx) if isinstance(idx, torch.Tensor) else idx for idx in artifact["test_indices"])
    
    if not (train_indices & val_indices):
        print("✓ No overlap between client training and global validation")
    else:
        print(f"✗ Overlap detected: {train_indices & val_indices}")
    
    if not (train_indices & test_indices):
        print("✓ No overlap between client training and global test")
    else:
        print(f"✗ Overlap detected: {train_indices & test_indices}")
    
    if not (val_indices & test_indices):
        print("✓ No overlap between validation and test")
    else:
        print(f"✗ Overlap detected: {val_indices & test_indices}")
    
    print("\n✓ Semi non-IID partitioning test PASSED!")
    return True


if __name__ == "__main__":
    try:
        test_semi_non_iid_partitioning()
    except Exception as e:
        print(f"\n✗ Test FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)
