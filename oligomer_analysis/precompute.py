# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

import json
from typing import List, Optional, Tuple, Any

import MDAnalysis as mda
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from MDAnalysis.lib.distances import capped_distance, minimize_vectors

from .cache_io import save_cache
from .models import FrameContacts, PrecomputeConfig


def valid_box(box: Optional[np.ndarray]) -> bool:
    """Return True when ``box`` looks like a valid MDAnalysis 6-vector box."""
    if box is None:
        return False
    box = np.asarray(box, dtype=float)
    return box.shape == (6,) and np.all(box[:3] > 0.0)


def get_box(ts, fallback_box: Optional[Tuple[float, float, float, float, float, float]]) -> Optional[np.ndarray]:
    """Use trajectory box if valid, else the configured fallback box."""
    box = np.asarray(ts.dimensions, dtype=float) if hasattr(ts, "dimensions") else None
    if valid_box(box):
        return box
    if fallback_box is not None:
        fb = np.asarray(fallback_box, dtype=float)
        if valid_box(fb):
            return fb
    return None


def residue_cog_pbc(res_positions: np.ndarray, box: Optional[np.ndarray]) -> np.ndarray:
    """PBC-robust residue geometry center without requiring bonded topology."""
    pos = np.asarray(res_positions, dtype=np.float32)
    if pos.size == 0:
        return np.zeros(3, dtype=np.float32)
    if box is None:
        return pos.mean(axis=0)

    anchor = pos[0]
    vec = pos - anchor
    vec_mic = minimize_vectors(vec, box)
    unwrapped = anchor + vec_mic
    return unwrapped.mean(axis=0)


def segment_cog_pbc(points_xyz: np.ndarray, box: Optional[np.ndarray]) -> np.ndarray:
    """PBC-robust segment geometry center from a set of 3D points."""
    pos = np.asarray(points_xyz, dtype=np.float32)
    if pos.size == 0:
        return np.zeros(3, dtype=np.float32)
    if box is None:
        return pos.mean(axis=0)

    anchor = pos[0]
    vec = pos - anchor
    vec_mic = minimize_vectors(vec, box)
    unwrapped = anchor + vec_mic
    return unwrapped.mean(axis=0)



def _validate_block_size(total: int, block: Optional[int], n_monomers: Optional[int], name: str) -> Tuple[int, int]:
    """Validate fixed-size monomer block settings.

    Returns
    -------
    nseg, block_size
    """
    if block is None:
        if n_monomers is None:
            raise ValueError(
                f"{name}_per_monomer or n_monomers must be provided when using block-based segmentation."
            )
        if int(n_monomers) <= 0:
            raise ValueError("n_monomers must be positive.")
        if total % int(n_monomers) != 0:
            raise ValueError(
                f"Cannot infer equal {name} blocks: total {name}s={total} is not divisible by n_monomers={n_monomers}."
            )
        block = total // int(n_monomers)
    block = int(block)
    if block <= 0:
        raise ValueError(f"{name}_per_monomer must be positive.")
    if total % block != 0:
        raise ValueError(
            f"total {name}s={total} is not divisible by {name}_per_monomer={block}. "
            "This means the trajectory is not arranged as equal monomer blocks, or the block size is wrong."
        )
    nseg = total // block
    if n_monomers is not None and int(n_monomers) != nseg:
        raise ValueError(
            f"n_monomers={n_monomers} conflicts with total {name}s / {name}_per_monomer = {nseg}."
        )
    return int(nseg), int(block)


def _make_block_segids(nseg: int) -> List[str]:
    """Create stable synthetic monomer labels for block-based segmentation."""
    return [f"M{i:04d}" for i in range(int(nseg))]


def build_topology_cache(u: mda.Universe, cfg: Optional[PrecomputeConfig] = None) -> dict:
    """Residue-to-monomer lookup tables used in every frame.

    This function supports three monomer definitions:

    1. ``segment_mode='mdanalysis'``
       Original behavior. One MDAnalysis segment is one monomer. This works only
       when your topology truly contains one unique segment per physical chain.

    2. ``segment_mode='atom_blocks'``
       One monomer is one consecutive block of atoms. This fixes the common PDB
       chain-ID limitation for large Martini systems. For example, 128 monomers
       with 83 beads/atoms each should use ``atoms_per_monomer=83``. The code
       then assigns ``seg_of_res = first_atom_index // 83`` instead of using the
       one-character PDB chain ID.

    3. ``segment_mode='residue_blocks'``
       One monomer is one consecutive block of residues. Use this when residues
       are ordered monomer-by-monomer and each monomer has the same number of
       residues/beads.
    """
    mode = "mdanalysis" if cfg is None else str(getattr(cfg, "segment_mode", "mdanalysis")).lower()
    residues = list(u.residues)

    if mode in ("mdanalysis", "segments", "segids"):
        segs = u.segments
        nseg = len(segs)
        if nseg == 0:
            raise ValueError("No segments found. Check segid/chain/segmentation in the topology.")

        seg_of_res = np.empty(len(residues), dtype=np.int32)
        res_index_in_seg = np.empty(len(residues), dtype=np.int32)
        seg_to_index = {seg: i for i, seg in enumerate(segs)}
        per_seg_counter = {seg: 0 for seg in segs}

        for k, res in enumerate(residues):
            seg = res.segment
            si = seg_to_index[seg]
            seg_of_res[k] = si
            res_index_in_seg[k] = per_seg_counter[seg]
            per_seg_counter[seg] += 1

        segids = [str(seg.segid) for seg in segs]
        mode_save = "mdanalysis"
        block_info: dict[str, Any] = {}

    elif mode in ("atom_blocks", "atom_block", "atoms"):
        if cfg is None:
            raise ValueError("cfg is required for segment_mode='atom_blocks'.")
        n_atoms = int(u.atoms.n_atoms)
        nseg, atoms_per_monomer = _validate_block_size(
            n_atoms,
            getattr(cfg, "atoms_per_monomer", None),
            getattr(cfg, "n_monomers", None),
            "atom",
        )
        seg_of_res = np.empty(len(residues), dtype=np.int32)
        res_index_in_seg = np.empty(len(residues), dtype=np.int32)
        per_seg_counter = np.zeros(nseg, dtype=np.int32)

        for k, res in enumerate(residues):
            atom_indices = np.asarray(res.atoms.indices, dtype=np.int64)
            if atom_indices.size == 0:
                raise ValueError(f"Residue index {k} has no atoms; cannot assign to atom block.")
            block_ids = atom_indices // atoms_per_monomer
            unique_blocks = np.unique(block_ids)
            if unique_blocks.size != 1:
                raise ValueError(
                    f"Residue {k} spans multiple atom blocks {unique_blocks.tolist()}. "
                    "Use residue_blocks mode or fix the topology ordering."
                )
            si = int(unique_blocks[0])
            if si < 0 or si >= nseg:
                raise ValueError(f"Computed monomer index {si} out of range 0..{nseg-1} for residue {k}.")
            seg_of_res[k] = si
            res_index_in_seg[k] = per_seg_counter[si]
            per_seg_counter[si] += 1

        segids = _make_block_segids(nseg)
        mode_save = "atom_blocks"
        block_info = {"atoms_per_monomer": int(atoms_per_monomer), "topology_n_atoms": n_atoms}

    elif mode in ("residue_blocks", "residue_block", "residues"):
        if cfg is None:
            raise ValueError("cfg is required for segment_mode='residue_blocks'.")
        n_residues = int(len(residues))
        nseg, residues_per_monomer = _validate_block_size(
            n_residues,
            getattr(cfg, "residues_per_monomer", None),
            getattr(cfg, "n_monomers", None),
            "residue",
        )
        seg_of_res = np.empty(n_residues, dtype=np.int32)
        res_index_in_seg = np.empty(n_residues, dtype=np.int32)
        for k in range(n_residues):
            si = k // residues_per_monomer
            seg_of_res[k] = si
            res_index_in_seg[k] = k % residues_per_monomer

        segids = _make_block_segids(nseg)
        mode_save = "residue_blocks"
        block_info = {"residues_per_monomer": int(residues_per_monomer), "topology_n_residues": n_residues}

    else:
        raise ValueError(
            "segment_mode must be one of: 'mdanalysis', 'atom_blocks', or 'residue_blocks'. "
            f"Got {mode!r}."
        )

    res_by_seg = [[] for _ in range(int(nseg))]
    for res_idx, seg_idx in enumerate(seg_of_res):
        res_by_seg[int(seg_idx)].append(res_idx)
    res_by_seg = [np.asarray(v, dtype=np.int32) for v in res_by_seg]

    return {
        "nseg": int(nseg),
        "residues": residues,
        "seg_of_res": seg_of_res,
        "res_index_in_seg": res_index_in_seg,
        "res_by_seg": res_by_seg,
        "segids": segids,
        "segment_mode": mode_save,
        **block_info,
    }

def _try_enable_fast_vectorized_centers(u: mda.Universe, fallback_box) -> bool:
    """Try the unwrap/wrap transform path for fast COG computation."""
    try:
        if u.atoms.n_bonds <= 0:
            return False
        u.trajectory[0]
        box0 = get_box(u.trajectory.ts, fallback_box)
        if box0 is None:
            return False
        from MDAnalysis.transformations import unwrap, wrap

        u.trajectory.add_transformations(
            unwrap(u.atoms),
            wrap(u.atoms, compound="segments"),
        )
        return True
    except Exception:
        return False


def compute_contacts_for_frame(
    u: mda.Universe,
    frame_index: int,
    topo_cache: dict,
    cutoff_contact: float,
    intra_exclude: int,
    fallback_box: Optional[Tuple[float, float, float, float, float, float]],
    *,
    fast_vectorized_cog: bool,
) -> FrameContacts:
    """Compute the segment contact map and segment centers for one frame."""
    u.trajectory[frame_index]
    box = get_box(u.trajectory.ts, fallback_box)

    box_save = np.zeros(6, dtype=np.float32)
    if box is not None:
        box_save[:] = np.asarray(box, dtype=np.float32)

    nseg = topo_cache["nseg"]
    residues = topo_cache["residues"]
    seg_of_res = topo_cache["seg_of_res"]
    res_index_in_seg = topo_cache["res_index_in_seg"]
    res_by_seg = topo_cache["res_by_seg"]

    if fast_vectorized_cog:
        pos = u.atoms.center_of_geometry(compound="residues").astype(np.float32, copy=False)
        seg_centers = u.atoms.center_of_geometry(compound="segments").astype(np.float32, copy=False)
    else:
        pos = np.empty((len(residues), 3), dtype=np.float32)
        for k, res in enumerate(residues):
            pos[k] = residue_cog_pbc(res.atoms.positions, box)

        seg_centers = np.zeros((nseg, 3), dtype=np.float32)
        for si in range(nseg):
            seg_centers[si] = segment_cog_pbc(pos[res_by_seg[si]], box)

    pairs = capped_distance(pos, pos, max_cutoff=cutoff_contact, box=box, return_distances=False)
    cmap = np.zeros((nseg, nseg), dtype=np.int32)

    if len(pairs) == 0:
        return FrameContacts(
            frame=frame_index,
            nintra=0,
            ninter=0,
            ntotal=0,
            n_free=nseg,
            n_bound=0,
            cmap=cmap,
            seg_centers=seg_centers,
            box=box_save,
        )

    pairs = np.asarray(pairs, dtype=np.int32)
    i = pairs[:, 0]
    j = pairs[:, 1]

    keep = i < j
    i = i[keep]
    j = j[keep]

    si = seg_of_res[i]
    sj = seg_of_res[j]
    same_seg = si == sj

    if intra_exclude <= 1:
        nintra = int(same_seg.sum())
    else:
        di = res_index_in_seg[i]
        dj = res_index_in_seg[j]
        nintra = int((same_seg & (np.abs(dj - di) >= intra_exclude)).sum())

    inter = ~same_seg
    if inter.any():
        a = si[inter]
        b = sj[inter]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        flat = lo * nseg + hi
        counts = np.bincount(flat, minlength=nseg * nseg).astype(np.int32)
        upper = counts.reshape((nseg, nseg))
        cmap = upper + upper.T
        np.fill_diagonal(cmap, 0)
        ninter = int(np.triu(cmap, 1).sum())
    else:
        ninter = 0

    ntotal = nintra + ninter
    row_sum = cmap.sum(axis=1)
    n_free = int((row_sum == 0).sum())
    n_bound = int(nseg - n_free)

    return FrameContacts(
        frame=frame_index,
        nintra=nintra,
        ninter=ninter,
        ntotal=ntotal,
        n_free=n_free,
        n_bound=n_bound,
        cmap=cmap,
        seg_centers=seg_centers,
        box=box_save,
    )


def _chunkify(seq: List[int], size: int) -> List[List[int]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _precompute_chunk(cfg: PrecomputeConfig, frames_chunk: List[int]) -> List[FrameContacts]:
    """Worker entry point; creates its own Universe instance."""
    if cfg.topology_file is None:
        u = mda.Universe(cfg.traj_file, in_memory=cfg.in_memory)
    else:
        u = mda.Universe(cfg.topology_file, cfg.traj_file, in_memory=cfg.in_memory)

    topo_cache = build_topology_cache(u, cfg)

    # Fast segment COG via MDAnalysis segments is only valid when one MDAnalysis
    # segment is truly one physical monomer. For atom/residue block segmentation,
    # segment centers must be computed from the block-assigned residue COGs.
    if topo_cache.get("segment_mode") == "mdanalysis":
        fast_vectorized_cog = _try_enable_fast_vectorized_centers(u, cfg.fallback_box)
    else:
        fast_vectorized_cog = False

    return [
        compute_contacts_for_frame(
            u=u,
            frame_index=fr,
            topo_cache=topo_cache,
            cutoff_contact=cfg.cutoff_contact,
            intra_exclude=cfg.intra_exclude,
            fallback_box=cfg.fallback_box,
            fast_vectorized_cog=fast_vectorized_cog,
        )
        for fr in frames_chunk
    ]


def precompute_cache(cfg: PrecomputeConfig, out_npz: str) -> None:
    """Run Stage A and save a compressed cache plus per-frame CSV statistics."""
    chunks = _chunkify(cfg.frames, cfg.chunk_size)
    print(f"• Precompute: {len(cfg.frames)} frames, chunk_size={cfg.chunk_size}, n_jobs={cfg.n_jobs}")

    results_nested = Parallel(n_jobs=cfg.n_jobs)(
        delayed(_precompute_chunk)(cfg, chunk) for chunk in chunks
    )
    results = [item for chunk in results_nested for item in chunk]
    results.sort(key=lambda x: x.frame)

    if cfg.topology_file is None:
        u0 = mda.Universe(cfg.traj_file, in_memory=False)
    else:
        u0 = mda.Universe(cfg.topology_file, cfg.traj_file, in_memory=False)
    topo_cache0 = build_topology_cache(u0, cfg)

    frames = np.array([r.frame for r in results], dtype=np.int32)
    meta = {
        "topology_file": cfg.topology_file,
        "traj_file": cfg.traj_file,
        "cutoff_contact": cfg.cutoff_contact,
        "intra_exclude": cfg.intra_exclude,
        "nseg": int(topo_cache0["nseg"]),
        "segids": topo_cache0["segids"],
        "segment_mode": topo_cache0.get("segment_mode", getattr(cfg, "segment_mode", "mdanalysis")),
        "atoms_per_monomer": getattr(cfg, "atoms_per_monomer", None),
        "residues_per_monomer": getattr(cfg, "residues_per_monomer", None),
        "n_monomers": getattr(cfg, "n_monomers", None),
        "topology_n_atoms": int(getattr(u0.atoms, "n_atoms", len(u0.atoms))),
        "topology_n_residues": int(len(u0.residues)),
        "fallback_box": cfg.fallback_box,
        "frames_min": int(frames.min()) if len(frames) else None,
        "frames_max": int(frames.max()) if len(frames) else None,
        "n_frames": int(len(frames)),
    }

    save_cache(out_npz, results=results, meta=meta, save_csv_stats=True)
    print(f"✓ Saved cache: {out_npz}")
    print(f"✓ Saved stats: {out_npz.replace('.npz', '_contact_stats.csv')}")
