# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""CoAnaPure 1.3 -- a pure-NumPy/SciPy Stage A engine for oligomer contact analysis.

CoAnaPure removes the MDAnalysis dependency from the expensive trajectory-facing
step (Stage A).  It re-implements the pieces of MDAnalysis the original
:mod:`oligomer_analysis.precompute` used:

* ``MDAnalysis.lib.distances.minimize_vectors``  -> :mod:`coanapure.pbc`
* ``MDAnalysis.lib.distances.capped_distance``   -> :mod:`coanapure.neighbors`
* ``MDAnalysis.Universe`` trajectory reading     -> :mod:`coanapure.pdb_io`,
  :mod:`coanapure.gro_io`, :mod:`coanapure.trr_io`, :mod:`coanapure.xtc_io`
  (PDB, GRO, TRR, XTC)

and exposes :func:`precompute_cache_pure`, a signature-compatible drop-in for
``precompute_cache`` that writes the identical ``.npz`` cache format consumed by
every downstream Stage B / lineage / plotting module.  The engine streams
trajectories at constant memory, runs the per-frame contact loop on threads or
whole frame chunks on processes (``parallel="process"``, PDB + byte-offset
index), supports atom selections (``cfg.selection``), unequal monomer sizes
(``segment_mode="atom_counts"``), progress/ETA reporting, and crash-safe
checkpoint/resume for walltime-limited batch jobs.

Because the cache format is unchanged, existing analysis code keeps working:

    from oligomer_analysis import load_cache, build_lineages_from_cache
    from oligomer_analysis.coanapure import precompute_cache_pure, PrecomputeConfig

    cfg = PrecomputeConfig(
        topology_file=None,
        traj_file="traj.pdb",
        frames=list(range(250)),
        cutoff_contact=6.5,
        intra_exclude=2,
        segment_mode="atom_blocks",
        atoms_per_monomer=83,
        n_monomers=128,
    )
    precompute_cache_pure(cfg, out_npz="cache_pure.npz")
    cache = load_cache("cache_pure.npz")

The same run from a shell (see ``coana --help``)::

    coana precompute --traj traj.pdb --frames 0:249 --cutoff 6.5 \\
        --segment-mode atom_blocks --atoms-per-monomer 83 --n-monomers 128 \\
        -o cache_pure.npz
"""

from __future__ import annotations

from ..models import PrecomputeConfig
from . import gro_io, neighbors, pbc, pdb_io, selection, trr_io, xtc_io
from .gro_io import GROFrameStream, read_gro_topology
from .neighbors import capped_distance, self_capped_pairs
from .pbc import box6_from_vectors, minimize_vectors, triclinic_vectors, wrap_positions
from .pdb_io import (
    PDBFrameStream,
    PDBModelIndex,
    PDBStructure,
    ResidueTopology,
    build_model_index,
    read_pdb,
)
from .precompute_pure import build_topology_cache_pure, precompute_cache_pure
from .selection import apply_selection, atom_mask_from_selection, parse_selection
from .trr_io import TRRFrameStream
from .xtc_io import XTCFrameStream, count_xtc_frames

__version__ = "1.3"

__all__ = [
    "__version__",
    "PrecomputeConfig",
    "precompute_cache_pure",
    "build_topology_cache_pure",
    "read_pdb",
    "read_gro_topology",
    "build_model_index",
    "apply_selection",
    "atom_mask_from_selection",
    "parse_selection",
    "PDBStructure",
    "ResidueTopology",
    "PDBModelIndex",
    "PDBFrameStream",
    "GROFrameStream",
    "TRRFrameStream",
    "XTCFrameStream",
    "count_xtc_frames",
    "minimize_vectors",
    "triclinic_vectors",
    "box6_from_vectors",
    "wrap_positions",
    "capped_distance",
    "self_capped_pairs",
    "pbc",
    "neighbors",
    "pdb_io",
    "gro_io",
    "trr_io",
    "xtc_io",
    "selection",
]
