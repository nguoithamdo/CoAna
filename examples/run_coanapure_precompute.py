# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Build a Stage A cache with CoAnaPure 1.0 -- no MDAnalysis required.

The resulting ``.npz`` is identical in format to the one produced by the original
``precompute_cache``, so every downstream analysis keeps working unchanged.
"""

from oligomer_analysis import load_cache, build_lineages_from_cache, TargetStateConfig
from oligomer_analysis.coanapure import precompute_cache_pure, PrecomputeConfig

TOPOLOGY = "initial1.pdb"
TRAJECTORY = "md1_centered_filtered_1956frames.pdb"
OUT = "cache_coanapure_1956frames.npz"

cfg = PrecomputeConfig(
    topology_file=TOPOLOGY,
    traj_file=TRAJECTORY,
    frames=list(range(0, 1956)),
    cutoff_contact=6.5,
    intra_exclude=2,
    segment_mode="atom_blocks",
    atoms_per_monomer=83,
    n_monomers=128,
)

# broadcast_box=False (default) reproduces the MDAnalysis pipeline exactly.
# Set broadcast_box=True to apply the CRYST1 box to every frame (correct PBC for
# a constant-volume run that only writes CRYST1 once).
precompute_cache_pure(cfg, out_npz=OUT, broadcast_box=False)

cache = load_cache(OUT)
print("frames:", len(cache.frames), "monomers:", cache.meta["nseg"])

lineages = build_lineages_from_cache(
    cache,
    inter_thresh=1,
    target_config=TargetStateConfig(min_size=10, min_duration=5, min_contribution=0.70),
)
print("lineages found:", len(lineages.lineages))
