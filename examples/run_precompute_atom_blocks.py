# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Stage A precomputation for large Martini PDB systems with reused chain IDs.

Use this when a PDB file contains many physical monomers but the one-character
PDB chain ID column cannot uniquely label all chains. Example: 128 monomers × 83
atoms/beads = 10624 atoms, but only ~63 unique PDB chain IDs.

The key setting is:

    segment_mode="atom_blocks"
    atoms_per_monomer=83

This defines physical monomer index as:

    monomer_index = atom_index // atoms_per_monomer

instead of using PDB chain IDs.
"""

from oligomer_analysis import PrecomputeConfig, precompute_cache, pdb_chain_id_summary

TOPOLOGY = r"E:\LinuxData\project\People\Thu_huongdan_2025\NganKHSK\initial1.pdb"
TRAJECTORY = r"E:\LinuxData\project\People\Thu_huongdan_2025\NganKHSK\md1_centered_filtered_1956frames.pdb"

# Optional diagnostic: confirms whether PDB chain IDs are reused.
pdb_chain_id_summary(TOPOLOGY, atoms_per_monomer=83)

cfg = PrecomputeConfig(
    topology_file=TOPOLOGY,
    traj_file=TRAJECTORY,
    frames=list(range(0, 1956)),
    cutoff_contact=6.5,
    intra_exclude=2,
    n_jobs=-1,
    chunk_size=50,
    in_memory=False,
    fallback_box=None,

    # Critical fix for >63-chain PDB systems.
    segment_mode="atom_blocks",
    atoms_per_monomer=83,
    n_monomers=128,
)

precompute_cache(cfg, out_npz="cache_md1_1956frames_cut6.5_atomblocks.npz")
