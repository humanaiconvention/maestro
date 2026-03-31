"""
libs.lattice — Participant knowledge-packet lattice: core data contract for the HAIC interview pipeline.

This package defines:
  - LatticeNode: content-addressed Merkle DAG node (9 kinds)
  - SessionLattice: complete Merkle-committed participant session
  - ConsentRecord: 5-layer consent record, committed as a node
  - build_session_lattice(), verify_lattice(), save/load_lattice()
  - GFS derived layer: 6-dimension grounding quality scores, versioned, recomputable

These are protocol-level data contracts, not pipeline utilities.
They live here rather than in tools/ because:
  - The gateway (apps/gateway/participation.py) imports them directly
  - They define the canonical participant data shape
  - They are stable, versioned, and test-covered

Pipeline tools (export, improvement loop) live in tools/ and import from here.
"""

from libs.lattice.lattice import (
    LatticeNode,
    SessionLattice,
    SessionInput,
    ConsentRecord,
    NodeKind,
    TrainingUse,
    make_node,
    build_session_lattice,
    verify_node,
    verify_lattice,
    merkle_root,
    to_receipt_dict,
    save_lattice,
    load_lattice,
    compute_content_hash,
    compute_node_hash,
)

from libs.lattice.gfs_layer import (
    GFSScore,
    GFS_DIMENSIONS,
    GFS_VERSION,
    annotate_lattice,
    recompute_gfs_layer,
    aggregate_gfs_scores,
    make_revision_node,
    score_turn,
)

__all__ = [
    # lattice
    "LatticeNode", "SessionLattice", "SessionInput", "ConsentRecord",
    "NodeKind", "TrainingUse",
    "make_node", "build_session_lattice",
    "verify_node", "verify_lattice",
    "merkle_root", "to_receipt_dict",
    "save_lattice", "load_lattice",
    "compute_content_hash", "compute_node_hash",
    # gfs
    "GFSScore", "GFS_DIMENSIONS", "GFS_VERSION",
    "annotate_lattice", "recompute_gfs_layer",
    "aggregate_gfs_scores", "make_revision_node", "score_turn",
]
