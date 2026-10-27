# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Stage A precomputation example for the V2 protein aggregation package.

Run this only when you need to create or rebuild the `.npz` cache.

Stage A requires MDAnalysis and access to the trajectory/topology files.

Outputs:
- cache_cut6.5_compactness.npz
- cache_cut6.5_compactness_contact_stats.csv
"""

from oligomer_analysis import PrecomputeConfig, precompute_cache


cfg = PrecomputeConfig(
    topology_file="initial.pdb",                 # or None for PDB-only trajectory
    traj_file="dynamic_700frames_330K.pdb",
    frames=list(range(0, 700)),
    cutoff_contact=6.5,
    intra_exclude=2,
    n_jobs=-1,
    chunk_size=50,
    in_memory=False,
    fallback_box=None,                           # e.g. (a,b,c,90,90,90) if trajectory lacks box
)

precompute_cache(
    cfg,
    out_npz="cache_cut6.5_compactness.npz",
)
