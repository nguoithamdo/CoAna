"""CoAnaPure Stage A -- MDAnalysis-free precomputation.

This is a drop-in replacement for :func:`oligomer_analysis.precompute.precompute_cache`
that produces a byte-compatible ``.npz`` cache using only NumPy and the pure PDB
reader / PBC / neighbour modules in this subpackage.

It reproduces the *slow, correct* Stage A path of the original code (per-residue
PBC-safe centre of geometry, then a ``capped_distance`` contact search), which is
the path the original toolkit already uses for ``atom_blocks`` and
``residue_blocks`` segmentation.  The MDAnalysis-only "fast vectorised COG" path
(unwrap/wrap transforms) is intentionally not reproduced because it is only ever
enabled for ``segment_mode='mdanalysis'`` and gives a different, transform
dependent result; CoAnaPure always uses the deterministic per-residue COG.

Version 1.2 additions: native ``.xtc`` reading, atom selections
(``cfg.selection``), unequal monomer sizes (``segment_mode='atom_counts'``),
progress reporting with ETA, crash-safe checkpoint/resume, and a
process-parallel mode that distributes the *parsing* itself across CPUs using
the PDB model byte-offset index (threads only parallelise the contact math; on
parse-bound trajectories processes are the HPC-scale option).

Version 1.3 additions, all driven by mixtures: ``cfg.species_spec`` declares the
composition once (``"aSyn:295x50,Ab40:92x10"``) and derives the monomer block
sizes from it, so every downstream species analysis reads the composition from
the cache instead of being told again; ``cfg.contact_classes`` decomposes
inter-chain residue contacts by residue chemistry; and ``cfg.hotspot_blocks``
accumulates residue-resolved cross-species contact maps.  The last two need the
residue-pair list that is otherwise discarded after binning to segments, which
is why they are opt-in and cannot be added to a finished cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..cache_io import save_cache
from ..models import FrameContacts, PrecomputeConfig
from .neighbors import self_capped_pairs
from .pbc import minimize_vectors
from .pdb_io import (
    _DROPPED_BOX_MSG,
    PDBFrameStream,
    PDBModelIndex,
    PDBStructure,
    ResidueTopology,
    default_index_path,
    read_pdb,
)
from .selection import apply_selection

__all__ = ["build_topology_cache_pure", "precompute_cache_pure"]

ENGINE_VERSION = "coanapure-1.3"


def _validate_block_size(total: int, block: Optional[int], n_monomers: Optional[int], name: str) -> Tuple[int, int]:
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
            f"total {name}s={total} is not divisible by {name}_per_monomer={block}."
        )
    nseg = total // block
    if n_monomers is not None and int(n_monomers) != nseg:
        raise ValueError(
            f"n_monomers={n_monomers} conflicts with total {name}s / {name}_per_monomer = {nseg}."
        )
    return int(nseg), int(block)


def _make_block_segids(nseg: int) -> List[str]:
    return [f"M{i:04d}" for i in range(int(nseg))]


def _residues_within_blocks(
    res_first_atom: np.ndarray,
    res_atom_counts: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Segment index per residue given atom-block ``boundaries``; every atom of a
    residue must fall inside one block."""
    first_block = np.searchsorted(boundaries, res_first_atom, side="right") - 1
    last_atom = res_first_atom + res_atom_counts.astype(np.int64) - 1
    last_block = np.searchsorted(boundaries, last_atom, side="right") - 1
    bad = np.flatnonzero(first_block != last_block)
    if bad.size:
        raise ValueError(
            f"Residue {int(bad[0])} spans multiple atom blocks. "
            "Use residue_blocks mode or fix the topology ordering."
        )
    return first_block.astype(np.int32)


def build_topology_cache_pure(struct: PDBStructure, cfg: Optional[PrecomputeConfig]) -> dict:
    """Build the residue->monomer lookup tables, mirroring precompute.build_topology_cache."""
    mode = "mdanalysis" if cfg is None else str(getattr(cfg, "segment_mode", "mdanalysis")).lower()
    n_res = struct.n_residues
    res_first_atom = struct.res_first_atom

    if mode in ("mdanalysis", "segments", "segids"):
        # One MDAnalysis segment == one unique chain id value (in first-seen order).
        # This reproduces the original behaviour, including chain-id reuse merging.
        chain_to_idx: Dict[str, int] = {}
        seg_of_res = np.empty(n_res, dtype=np.int32)
        for k in range(n_res):
            chain = struct.res_chain[k]
            if chain not in chain_to_idx:
                chain_to_idx[chain] = len(chain_to_idx)
            seg_of_res[k] = chain_to_idx[chain]
        nseg = len(chain_to_idx)
        if nseg == 0:
            raise ValueError("No segments found. Check chain ids in the topology.")
        segids = list(chain_to_idx.keys())
        mode_save = "mdanalysis"
        block_info: dict = {}

    elif mode in ("atom_blocks", "atom_block", "atoms"):
        if cfg is None:
            raise ValueError("cfg is required for segment_mode='atom_blocks'.")
        n_atoms = struct.n_atoms
        nseg, atoms_per_monomer = _validate_block_size(
            n_atoms, getattr(cfg, "atoms_per_monomer", None), getattr(cfg, "n_monomers", None), "atom"
        )
        boundaries = np.arange(nseg + 1, dtype=np.int64) * atoms_per_monomer
        seg_of_res = _residues_within_blocks(res_first_atom, struct.res_atom_counts, boundaries)
        if seg_of_res.min() < 0 or seg_of_res.max() >= nseg:
            raise ValueError("Computed monomer index out of range for some residue.")
        segids = _make_block_segids(nseg)
        mode_save = "atom_blocks"
        block_info = {"atoms_per_monomer": int(atoms_per_monomer), "topology_n_atoms": int(n_atoms)}

    elif mode in ("atom_counts", "monomer_atom_counts"):
        if cfg is None:
            raise ValueError("cfg is required for segment_mode='atom_counts'.")
        counts_raw = getattr(cfg, "monomer_atom_counts", None)
        if not counts_raw:
            raise ValueError(
                "segment_mode='atom_counts' needs cfg.monomer_atom_counts "
                "(one atom count per monomer, in file order)."
            )
        counts = [int(c) for c in counts_raw]
        if any(c <= 0 for c in counts):
            raise ValueError("monomer_atom_counts must all be positive.")
        n_atoms = struct.n_atoms
        if sum(counts) != n_atoms:
            raise ValueError(
                f"monomer_atom_counts sum to {sum(counts)} but the topology has "
                f"{n_atoms} atoms."
            )
        n_mono = getattr(cfg, "n_monomers", None)
        if n_mono is not None and int(n_mono) != len(counts):
            raise ValueError(
                f"n_monomers={n_mono} conflicts with len(monomer_atom_counts)={len(counts)}."
            )
        nseg = len(counts)
        boundaries = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        seg_of_res = _residues_within_blocks(res_first_atom, struct.res_atom_counts, boundaries)
        segids = _make_block_segids(nseg)
        mode_save = "atom_counts"
        block_info = {"monomer_atom_counts": counts, "topology_n_atoms": int(n_atoms)}

    elif mode in ("residue_blocks", "residue_block", "residues"):
        if cfg is None:
            raise ValueError("cfg is required for segment_mode='residue_blocks'.")
        nseg, residues_per_monomer = _validate_block_size(
            n_res, getattr(cfg, "residues_per_monomer", None), getattr(cfg, "n_monomers", None), "residue"
        )
        seg_of_res = (np.arange(n_res) // residues_per_monomer).astype(np.int32)
        segids = _make_block_segids(nseg)
        mode_save = "residue_blocks"
        block_info = {"residues_per_monomer": int(residues_per_monomer), "topology_n_residues": int(n_res)}

    else:
        raise ValueError(
            "segment_mode must be one of: 'mdanalysis', 'atom_blocks', "
            "'atom_counts', or 'residue_blocks'. "
            f"Got {mode!r}."
        )

    # res_index_in_seg: running count of residues within each segment (file order).
    res_index_in_seg = np.empty(n_res, dtype=np.int32)
    counter = np.zeros(nseg, dtype=np.int32)
    for k in range(n_res):
        s = int(seg_of_res[k])
        res_index_in_seg[k] = counter[s]
        counter[s] += 1

    # First residue (lowest index) of each segment, used as PBC anchor.
    first_res_of_seg = np.full(nseg, -1, dtype=np.int64)
    for k in range(n_res):
        s = int(seg_of_res[k])
        if first_res_of_seg[s] < 0:
            first_res_of_seg[s] = k

    return {
        "nseg": int(nseg),
        "seg_of_res": seg_of_res,
        "res_index_in_seg": res_index_in_seg,
        "first_res_of_seg": first_res_of_seg,
        "segids": segids,
        "segment_mode": mode_save,
        **block_info,
    }


def _CONTACT_CLASS_NAMES():
    """Contact-class names, imported lazily to keep the engine's imports light."""
    from ..chemistry import CONTACT_CLASSES

    return CONTACT_CLASSES


def _resolve_species_cfg(cfg: PrecomputeConfig, verbose: bool) -> PrecomputeConfig:
    """Expand ``cfg.species_spec`` into concrete monomer block sizes.

    Declaring the composition once is the whole point: the spec fixes the
    segmentation *and* travels into the cache metadata, so no later analysis has
    to be told again which chains are which.  When the caller also passed
    ``monomer_atom_counts`` the two are cross-checked rather than one silently
    winning.
    """
    spec = getattr(cfg, "species_spec", None)
    if not spec:
        return cfg

    from dataclasses import replace

    from ..species import monomer_atom_counts_from_spec, parse_species_spec

    parsed = parse_species_spec(spec)
    derived = monomer_atom_counts_from_spec(spec)

    existing = getattr(cfg, "monomer_atom_counts", None)
    if existing:
        if [int(c) for c in existing] != derived:
            raise ValueError(
                f"species_spec {spec!r} implies {len(derived)} monomers "
                f"{derived[:4]}... but monomer_atom_counts has "
                f"{len(existing)} entries {list(existing)[:4]}...; "
                "pass only one of the two."
            )

    n_mono = getattr(cfg, "n_monomers", None)
    if n_mono is not None and int(n_mono) != len(derived):
        raise ValueError(
            f"species_spec {spec!r} describes {len(derived)} chains but "
            f"n_monomers={n_mono}."
        )

    mode = str(getattr(cfg, "segment_mode", "mdanalysis")).lower()
    changes = {"monomer_atom_counts": derived, "n_monomers": len(derived)}
    if mode in ("mdanalysis", "segments", "segids"):
        # The default mode keys off chain ids, which Martini PDBs reuse after ~62
        # chains. A spec is an explicit statement of the block layout, so honour
        # it rather than segmenting by an identifier we know to be unreliable.
        changes["segment_mode"] = "atom_counts"
        if verbose:
            print(
                "  species_spec given: segment_mode -> atom_counts "
                f"({len(derived)} chains)",
                flush=True,
            )
    elif mode not in ("atom_counts", "monomer_atom_counts"):
        raise ValueError(
            f"species_spec is only meaningful with segment_mode='atom_counts' "
            f"(got {mode!r}); drop one of the two."
        )

    if verbose:
        desc = ", ".join(f"{n}:{b}x{c}" for n, b, c in parsed)
        print(f"  species: {desc}", flush=True)
    return replace(cfg, **changes)


def _species_tables(topo: dict, cfg: PrecomputeConfig) -> Optional[dict]:
    """Per-segment species index and per-species residue counts.

    Returns None when the run declares no species, which is the one-component
    case where a cross-species decomposition has nothing to say.
    """
    spec = getattr(cfg, "species_spec", None)
    if not spec:
        return None

    from ..species import parse_species_spec

    parsed = parse_species_spec(spec)
    names = [n for n, _b, _c in parsed]
    seg_species = np.concatenate(
        [np.full(count, k, dtype=np.int16) for k, (_n, _b, count) in enumerate(parsed)]
    )
    nseg = int(topo["nseg"])
    if seg_species.size != nseg:
        raise ValueError(
            f"species_spec describes {seg_species.size} chains but the topology "
            f"segments into {nseg}."
        )

    res_per_seg = np.bincount(np.asarray(topo["seg_of_res"]), minlength=nseg)
    res_per_species = []
    for k in range(len(names)):
        sizes = np.unique(res_per_seg[seg_species == k])
        if sizes.size != 1:
            raise ValueError(
                f"species {names[k]!r} has chains of {sizes.size} different "
                f"residue counts {sizes.tolist()}; the topology ordering does "
                "not match the species spec."
            )
        res_per_species.append(int(sizes[0]))

    return {
        "names": names,
        "seg_species": seg_species.astype(np.int16),
        "res_per_species": res_per_species,
        "n_res_max": int(max(res_per_species)),
    }


def _residue_cogs(struct: PDBStructure, frame_positions: np.ndarray, box: Optional[np.ndarray]) -> np.ndarray:
    """PBC-safe residue centre of geometry for all residues (vectorised)."""
    residue_of_atom = struct.residue_of_atom
    anchor_atom = struct.res_first_atom[residue_of_atom]  # (n_atoms,)
    anchors = frame_positions[anchor_atom]                # (n_atoms, 3)

    if box is None:
        unwrapped = frame_positions.astype(np.float32, copy=False)
    else:
        vec = frame_positions.astype(np.float64) - anchors.astype(np.float64)
        vec_mic = minimize_vectors(vec, box).astype(np.float64)
        unwrapped = anchors.astype(np.float64) + vec_mic

    n_res = struct.n_residues
    sums = np.zeros((n_res, 3), dtype=np.float64)
    np.add.at(sums, residue_of_atom, unwrapped)
    counts = struct.res_atom_counts.astype(np.float64)[:, None]
    return (sums / counts).astype(np.float32)


def _segment_cogs(
    res_cog: np.ndarray, seg_of_res: np.ndarray, first_res_of_seg: np.ndarray, nseg: int, box: Optional[np.ndarray]
) -> np.ndarray:
    """PBC-safe segment centre of geometry from residue COGs (vectorised)."""
    anchors = res_cog[first_res_of_seg[seg_of_res]]  # (n_res, 3)
    if box is None:
        unwrapped = res_cog.astype(np.float64)
    else:
        vec = res_cog.astype(np.float64) - anchors.astype(np.float64)
        vec_mic = minimize_vectors(vec, box).astype(np.float64)
        unwrapped = anchors.astype(np.float64) + vec_mic

    sums = np.zeros((nseg, 3), dtype=np.float64)
    np.add.at(sums, seg_of_res, unwrapped)
    counts = np.bincount(seg_of_res, minlength=nseg).astype(np.float64)[:, None]
    counts[counts == 0] = 1.0
    return (sums / counts).astype(np.float32)


def _class_matrix_from_pairs(
    class_a: np.ndarray, class_b: np.ndarray, n_classes: int
) -> np.ndarray:
    """Symmetric class-pair count matrix from one frame's inter-chain pairs.

    Each unordered residue pair is supplied once, so the raw bincount is
    triangular in effect; adding its transpose symmetrises it, and the diagonal
    is restored to the raw (undoubled) same-class count.  ``np.triu(M).sum()``
    is then exactly the number of inter-chain residue contacts.
    """
    flat = class_a.astype(np.int64) * n_classes + class_b.astype(np.int64)
    raw = np.bincount(flat, minlength=n_classes * n_classes).reshape(
        (n_classes, n_classes)
    )
    matrix = raw + raw.T
    np.fill_diagonal(matrix, np.diag(raw))
    return matrix.astype(np.int64, copy=False)


def _hotspot_from_pairs(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    res_a: np.ndarray,
    res_b: np.ndarray,
    seg_species: np.ndarray,
    n_species: int,
    n_res: int,
) -> np.ndarray:
    """Residue-resolved contact tensor ``(S, S, R, R)`` for one frame.

    Entry ``[s, t, p, q]`` counts contacts between residue ``p`` of a species-``s``
    chain and residue ``q`` of a species-``t`` chain, summed over all chain
    pairs.  Both orientations of every pair are recorded, so the tensor satisfies
    ``H[s, t] == H[t, s].T`` and the cross-species block can be read directly as
    "which part of A touches which part of B".
    """
    sp_a = seg_species[seg_a].astype(np.int64)
    sp_b = seg_species[seg_b].astype(np.int64)
    ra = res_a.astype(np.int64)
    rb = res_b.astype(np.int64)

    size = n_species * n_species * n_res * n_res
    fwd = ((sp_a * n_species + sp_b) * n_res + ra) * n_res + rb
    rev = ((sp_b * n_species + sp_a) * n_res + rb) * n_res + ra
    counts = np.bincount(np.concatenate((fwd, rev)), minlength=size)
    return counts.reshape((n_species, n_species, n_res, n_res)).astype(
        np.int32, copy=False
    )


def _compute_frame_arrays(
    res_topo,
    topo: dict,
    frame_index: int,
    frame_positions: np.ndarray,
    box: Optional[np.ndarray],
    cutoff_contact: float,
    intra_exclude: int,
    fallback_box: Optional[Tuple[float, float, float, float, float, float]],
    class_of_res: Optional[np.ndarray] = None,
    n_classes: int = 0,
    species_tables: Optional[dict] = None,
    contact_mode: str = "residue_cog",
) -> FrameContacts:
    """Contact statistics for one frame given its coordinates and box.

    ``res_topo`` is anything exposing the residue tables
    (:class:`~oligomer_analysis.coanapure.pdb_io.ResidueTopology` or
    :class:`~oligomer_analysis.coanapure.pdb_io.PDBStructure`).  Taking the
    per-frame arrays explicitly (instead of a whole in-memory trajectory) is what
    lets Stage A stream a multi-GB file at constant memory and lets the frame
    loop run on a thread pool.
    """
    nseg = topo["nseg"]
    seg_of_res = topo["seg_of_res"]
    res_index_in_seg = topo["res_index_in_seg"]

    if box is None and fallback_box is not None:
        fb = np.asarray(fallback_box, dtype=np.float64)
        if fb.shape == (6,) and np.all(fb[:3] > 0):
            box = fb

    box_save = np.zeros(6, dtype=np.float32)
    if box is not None:
        box_save[:] = np.asarray(box, dtype=np.float32)

    # Residue centroids are needed either way: the segment centres stored in
    # the cache are built from them.
    pos = _residue_cogs(res_topo, frame_positions, box)
    seg_centers = _segment_cogs(pos, seg_of_res, topo["first_res_of_seg"], nseg, box)

    cmap = np.zeros((nseg, nseg), dtype=np.int32)
    if contact_mode == "min_atom":
        # Any-atom criterion: search over atoms, then collapse to the unique
        # residue pairs they imply. Two residues are in contact when their
        # closest atoms are within the cutoff, which is what OLIGAMIX and
        # ATRANET mean by a contact.
        residue_of_atom = np.asarray(res_topo.residue_of_atom)
        atom_pairs = self_capped_pairs(frame_positions, cutoff_contact, box)
        if len(atom_pairs):
            ri = residue_of_atom[atom_pairs[:, 0]]
            rj = residue_of_atom[atom_pairs[:, 1]]
            lo = np.minimum(ri, rj)
            hi = np.maximum(ri, rj)
            keep = lo != hi                      # atoms of the same residue
            if keep.any():
                n_res_total = int(res_topo.n_residues)
                flat = np.unique(lo[keep].astype(np.int64) * n_res_total + hi[keep])
                pairs = np.stack((flat // n_res_total, flat % n_res_total), axis=1)
            else:
                pairs = np.empty((0, 2), dtype=np.int64)
        else:
            pairs = np.empty((0, 2), dtype=np.int64)
    elif contact_mode == "residue_cog":
        pairs = self_capped_pairs(pos, cutoff_contact, box)
    else:
        raise ValueError(
            f"unknown contact_mode {contact_mode!r}; "
            "expected 'residue_cog' or 'min_atom'"
        )

    want_classes = class_of_res is not None and n_classes > 0
    want_hotspot = species_tables is not None

    if len(pairs) == 0:
        return FrameContacts(
            frame=frame_index, nintra=0, ninter=0, ntotal=0,
            n_free=nseg, n_bound=0, cmap=cmap, seg_centers=seg_centers, box=box_save,
            class_cmap=(
                np.zeros((n_classes, n_classes), dtype=np.int64) if want_classes else None
            ),
            hotspot=(
                np.zeros(
                    (
                        len(species_tables["names"]),
                        len(species_tables["names"]),
                        species_tables["n_res_max"],
                        species_tables["n_res_max"],
                    ),
                    dtype=np.int32,
                )
                if want_hotspot
                else None
            ),
        )

    i = pairs[:, 0]
    j = pairs[:, 1]
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
    class_cmap = (
        np.zeros((n_classes, n_classes), dtype=np.int64) if want_classes else None
    )
    hotspot = None
    if want_hotspot:
        n_sp = len(species_tables["names"])
        n_res = species_tables["n_res_max"]
        hotspot = np.zeros((n_sp, n_sp, n_res, n_res), dtype=np.int32)

    if inter.any():
        a = si[inter]
        b = sj[inter]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        flat = lo.astype(np.int64) * nseg + hi.astype(np.int64)
        counts = np.bincount(flat, minlength=nseg * nseg).astype(np.int32)
        upper = counts.reshape((nseg, nseg))
        cmap = upper + upper.T
        np.fill_diagonal(cmap, 0)
        ninter = int(np.triu(cmap, 1).sum())

        # The residue-pair list is still in scope here; once this function
        # returns, only the segment-level cmap survives. Anything that needs
        # residue identity has to be accumulated now.
        if want_classes:
            inter_i = i[inter]
            inter_j = j[inter]
            class_cmap = _class_matrix_from_pairs(
                class_of_res[inter_i], class_of_res[inter_j], n_classes
            )
        if want_hotspot:
            inter_i = i[inter]
            inter_j = j[inter]
            hotspot = _hotspot_from_pairs(
                a,
                b,
                res_index_in_seg[inter_i],
                res_index_in_seg[inter_j],
                species_tables["seg_species"],
                len(species_tables["names"]),
                species_tables["n_res_max"],
            )
    else:
        ninter = 0

    ntotal = nintra + ninter
    row_sum = cmap.sum(axis=1)
    n_free = int((row_sum == 0).sum())
    n_bound = int(nseg - n_free)

    return FrameContacts(
        frame=frame_index, nintra=nintra, ninter=ninter, ntotal=ntotal,
        n_free=n_free, n_bound=n_bound, cmap=cmap, seg_centers=seg_centers, box=box_save,
        class_cmap=class_cmap, hotspot=hotspot,
    )


def _compute_frame(
    struct: PDBStructure,
    frame_index: int,
    topo: dict,
    cutoff_contact: float,
    intra_exclude: int,
    fallback_box: Optional[Tuple[float, float, float, float, float, float]],
    class_of_res: Optional[np.ndarray] = None,
    n_classes: int = 0,
    species_tables: Optional[dict] = None,
    contact_mode: str = "residue_cog",
) -> FrameContacts:
    """Back-compat wrapper over :func:`_compute_frame_arrays` for an in-memory
    :class:`PDBStructure` (as returned by :func:`read_pdb`)."""
    return _compute_frame_arrays(
        struct,
        topo,
        frame_index,
        struct.positions[frame_index],
        struct.frame_box(frame_index),
        cutoff_contact,
        intra_exclude,
        fallback_box,
        class_of_res,
        n_classes,
        species_tables,
        contact_mode,
    )


NO_BOX_MSG = (
    "No periodic box in this trajectory, so contacts were computed WITHOUT "
    "periodic boundaries. With whole molecules this misses every contact across "
    "a box face (10% of chain-chain contacts on a 128-chain Martini run); with "
    "wrapped coordinates it also creates fake contacts. Use a trajectory that "
    "carries the box (XTC/TRR/GRO, or CRYST1 in every PDB model), use "
    "--broadcast-box (broadcast_box=True) for a PDB whose box is only on the "
    "first model, or --fallback-box a,b,c (fallback_box=...) to supply it."
)


class NoBoxWarning(UserWarning):
    """The cache was built without periodic boundaries."""


def _resolve_workers(n_jobs: Optional[int]) -> int:
    """Map a joblib-style ``n_jobs`` to a thread count (-1 -> all CPUs)."""
    n = 1 if n_jobs is None else int(n_jobs)
    if n == 0:
        return 1
    if n < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n)
    return n


def _run_frame_loop(
    frame_iter: Iterable[Tuple[int, np.ndarray, Optional[np.ndarray]]],
    compute_one,
    workers: int,
) -> Iterator[FrameContacts]:
    """Drive ``compute_one`` over the streamed frames, optionally on threads.

    The producer (the trajectory parser) runs on the calling thread; frame
    computations run on up to ``workers`` threads with a bounded in-flight
    window, so memory stays at O(workers) frames regardless of trajectory size.
    NumPy ufuncs and the scipy cKDTree release the GIL for the bulk of the frame
    work, which is what makes threads (rather than processes, which would have
    to re-read the file) effective here.
    """
    if workers <= 1:
        for item in frame_iter:
            yield compute_one(*item)
        return
    window = workers * 3
    with ThreadPoolExecutor(max_workers=workers) as ex:
        pending = set()
        for item in frame_iter:
            pending.add(ex.submit(compute_one, *item))
            if len(pending) >= window:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    yield fut.result()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                yield fut.result()


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #

def _fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "?"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class _ProgressReporter:
    """Rate/ETA progress lines suitable for interactive shells and batch logs.

    Prints at most every ``min_interval`` seconds (plus a final line), always as
    whole flushed lines so SLURM/PBS output files stay readable.
    """

    def __init__(self, total: int, enabled: bool, *, min_interval: float = 2.0) -> None:
        self.total = int(total)
        self.enabled = bool(enabled) and self.total > 0
        self.min_interval = float(min_interval)
        self.done = 0
        self._t0 = time.monotonic()
        self._last_print = 0.0

    def update(self, n_new: int = 1) -> None:
        self.done += int(n_new)
        if not self.enabled:
            return
        now = time.monotonic()
        if self.done < self.total and (now - self._last_print) < self.min_interval:
            return
        self._last_print = now
        elapsed = now - self._t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("inf")
        pct = 100.0 * self.done / self.total
        print(
            f"  [{self.done}/{self.total}] {pct:5.1f}% | {rate:7.2f} frames/s | "
            f"elapsed {_fmt_duration(elapsed)} | ETA {_fmt_duration(eta)}",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# Checkpoint / resume
# --------------------------------------------------------------------------- #

def _checkpoint_path(out_npz: str) -> str:
    return str(out_npz) + ".checkpoint.npz"


def _config_fingerprint(cfg: PrecomputeConfig, broadcast_box: bool) -> str:
    """Hash of everything that affects the numerical result, so a stale or
    mismatched checkpoint can never silently leak into a cache."""
    try:
        traj_size = os.path.getsize(cfg.traj_file)
    except OSError:
        traj_size = -1
    payload = {
        "topology_file": cfg.topology_file,
        "traj_file": cfg.traj_file,
        "traj_size": traj_size,
        "frames": sorted({int(f) for f in cfg.frames}),
        "cutoff_contact": float(cfg.cutoff_contact),
        "contact_mode": getattr(cfg, "contact_mode", "residue_cog"),
        "intra_exclude": int(cfg.intra_exclude),
        "fallback_box": cfg.fallback_box,
        "segment_mode": getattr(cfg, "segment_mode", "mdanalysis"),
        "atoms_per_monomer": getattr(cfg, "atoms_per_monomer", None),
        "residues_per_monomer": getattr(cfg, "residues_per_monomer", None),
        "n_monomers": getattr(cfg, "n_monomers", None),
        "monomer_atom_counts": getattr(cfg, "monomer_atom_counts", None),
        "selection": getattr(cfg, "selection", None),
        "species_spec": getattr(cfg, "species_spec", None),
        "contact_classes": bool(getattr(cfg, "contact_classes", False)),
        "hotspot_blocks": getattr(cfg, "hotspot_blocks", None),
        "broadcast_box": bool(broadcast_box),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def _write_checkpoint(
    ckpt_path: str,
    results: Sequence[FrameContacts],
    fingerprint: str,
    topo_meta: dict,
    hotspot_acc: Optional[np.ndarray] = None,
) -> None:
    results = sorted(results, key=lambda r: r.frame)
    tmp = ckpt_path + ".tmp"
    extra = {}
    if results and all(r.class_cmap is not None for r in results):
        extra["class_cmaps"] = np.stack([r.class_cmap for r in results]).astype(
            np.int64, copy=False
        )
    if hotspot_acc is not None:
        # The hotspot tensor is a running sum over the frames already in
        # ``results``; checkpointing the two together is what keeps a resumed run
        # from losing (or double-counting) the frames it had folded in.
        extra["hotspot_acc"] = np.asarray(hotspot_acc, dtype=np.int64)
    np.savez_compressed(
        tmp,
        fingerprint=np.array(fingerprint, dtype=object),
        topo_meta_json=np.array(json.dumps(topo_meta), dtype=object),
        frames=np.array([r.frame for r in results], dtype=np.int32),
        nintra=np.array([r.nintra for r in results], dtype=np.int32),
        ninter=np.array([r.ninter for r in results], dtype=np.int32),
        ntotal=np.array([r.ntotal for r in results], dtype=np.int32),
        n_free=np.array([r.n_free for r in results], dtype=np.int32),
        n_bound=np.array([r.n_bound for r in results], dtype=np.int32),
        cmaps=np.stack([r.cmap for r in results]).astype(np.int32, copy=False),
        seg_centers=np.stack([r.seg_centers for r in results]).astype(np.float32, copy=False),
        boxes=np.stack([r.box for r in results]).astype(np.float32, copy=False),
        **extra,
    )
    os.replace(tmp + ".npz", ckpt_path)


def _load_checkpoint(
    ckpt_path: str, fingerprint: str, verbose: bool
) -> Tuple[List[FrameContacts], Optional[dict], Optional[np.ndarray]]:
    if not os.path.exists(ckpt_path):
        return [], None, None
    try:
        with np.load(ckpt_path, allow_pickle=True) as d:
            if str(d["fingerprint"].item()) != fingerprint:
                if verbose:
                    print(
                        f"  ignoring checkpoint {ckpt_path}: configuration changed "
                        "(different trajectory, frames, cutoff, selection, ...).",
                        flush=True,
                    )
                return [], None, None
            topo_meta = json.loads(str(d["topo_meta_json"].item()))
            keys = set(d.files)
            has_classes = "class_cmaps" in keys
            results = [
                FrameContacts(
                    frame=int(d["frames"][i]),
                    nintra=int(d["nintra"][i]),
                    ninter=int(d["ninter"][i]),
                    ntotal=int(d["ntotal"][i]),
                    n_free=int(d["n_free"][i]),
                    n_bound=int(d["n_bound"][i]),
                    cmap=d["cmaps"][i].copy(),
                    seg_centers=d["seg_centers"][i].copy(),
                    box=d["boxes"][i].copy(),
                    class_cmap=(d["class_cmaps"][i].copy() if has_classes else None),
                )
                for i in range(len(d["frames"]))
            ]
            hotspot_acc = d["hotspot_acc"].copy() if "hotspot_acc" in keys else None
        return results, topo_meta, hotspot_acc
    except (OSError, KeyError, ValueError) as exc:
        if verbose:
            print(f"  ignoring unreadable checkpoint {ckpt_path}: {exc}", flush=True)
        return [], None, None


# --------------------------------------------------------------------------- #
# Topology / stream helpers
# --------------------------------------------------------------------------- #

def _load_topology_file(topology_file: str) -> ResidueTopology:
    ext = os.path.splitext(topology_file)[1].lower()
    if ext in (".pdb", ".ent"):
        return ResidueTopology.from_pdb_file(topology_file)
    if ext == ".gro":
        from .gro_io import read_gro_topology

        return read_gro_topology(topology_file)
    raise ValueError(
        f"Unsupported topology format {ext!r} (use a .pdb or .gro file)."
    )


def _resolve_pdb_index(
    traj_file: str, index_file: Optional[str], verbose: bool
) -> Optional[PDBModelIndex]:
    """Load (or build) a model byte-offset index for random frame access.

    * ``index_file`` given and exists -> load it (stale index raises).
    * ``index_file`` given but missing -> build the index and save it there.
    * ``index_file`` None -> use the default sidecar only if present and fresh.
    """
    if index_file is not None:
        if os.path.exists(index_file):
            return PDBModelIndex.load(index_file, traj_file)
        if verbose:
            print(f"  building model index -> {index_file}")
        idx = PDBModelIndex.build(traj_file)
        idx.save(index_file)
        return idx
    sidecar = default_index_path(traj_file)
    if os.path.exists(sidecar):
        try:
            return PDBModelIndex.load(sidecar, traj_file)
        except ValueError:
            if verbose:
                print(f"  ignoring stale model index {sidecar}")
    return None


def _topology_from_index(traj_file: str, idx: PDBModelIndex) -> ResidueTopology:
    """Residue topology from model 0 of an indexed PDB (no full-file parse)."""
    with open(traj_file, "rb") as fh:
        fh.seek(int(idx.atoms_start[0]))
        blob = fh.read(int(idx.atoms_end[0]) - int(idx.atoms_start[0]))
    lines = [
        raw.decode("ascii", "replace")
        for raw in blob.splitlines()
        if raw[:4] == b"ATOM" or raw[:6] == b"HETATM"
    ]
    return ResidueTopology.from_pdb_lines(lines)


def _topo_meta_dict(
    topo: dict,
    res_topo_full: ResidueTopology,
    res_topo_selected: ResidueTopology,
    selection: Optional[str],
    traj_format: str,
    *,
    species_spec: Optional[str] = None,
    species_tables: Optional[dict] = None,
    class_names: Optional[Sequence[str]] = None,
    class_census: Optional[Dict[str, int]] = None,
) -> dict:
    """The result-independent part of the cache metadata (also checkpointed)."""
    meta = {
        "nseg": int(topo["nseg"]),
        "segids": list(topo["segids"]),
        "segment_mode": topo.get("segment_mode", "mdanalysis"),
        "topology_n_atoms": int(res_topo_full.n_atoms),
        "topology_n_residues": int(res_topo_full.n_residues),
        "traj_format": traj_format,
    }
    for key in ("atoms_per_monomer", "residues_per_monomer", "monomer_atom_counts"):
        if key in topo:
            meta[key] = topo[key]
    if selection:
        meta["selection"] = selection
        meta["selection_n_atoms"] = int(res_topo_selected.n_atoms)
        meta["selection_n_residues"] = int(res_topo_selected.n_residues)
    if species_tables is not None:
        from ..species import parse_species_spec

        parsed = parse_species_spec(species_spec) if species_spec else []
        meta["species_spec"] = species_spec
        meta["species"] = [
            {
                "name": name,
                "beads": int(beads),
                "count": int(count),
                "n_residues": int(species_tables["res_per_species"][k]),
            }
            for k, (name, beads, count) in enumerate(parsed)
        ]
    if class_names is not None:
        meta["contact_classes"] = list(class_names)
        if class_census:
            meta["contact_class_census"] = class_census
    return meta


# --------------------------------------------------------------------------- #
# Process-parallel worker (PDB + index)
# --------------------------------------------------------------------------- #

def _process_pdb_chunk(args: tuple) -> List[FrameContacts]:
    """Compute one chunk of frames inside a worker process.

    The worker opens the trajectory itself and *seeks* to its frames through
    the byte-offset index, so file parsing -- the bottleneck the GIL prevents
    threads from sharing -- is genuinely distributed across CPUs.
    """
    (
        traj_file, idx, chunk, res_topo, atom_mask, topo,
        cutoff_contact, intra_exclude, fallback_box, broadcast_box, n_atoms_full,
        class_of_res, n_classes, species_tables, contact_mode,
    ) = args
    stream = PDBFrameStream(
        traj_file,
        wanted=set(int(k) for k in chunk),
        index=idx,
        capture_topology=False,
        n_atoms_expected=n_atoms_full,
        broadcast_box=broadcast_box,
        warn_dropped_box=False,
    )
    out: List[FrameContacts] = []
    for k, pos, box in stream:
        if atom_mask is not None:
            pos = pos[atom_mask]
        out.append(
            _compute_frame_arrays(
                res_topo, topo, k, pos, box, cutoff_contact, intra_exclude,
                fallback_box, class_of_res, n_classes, species_tables,
                contact_mode,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def precompute_cache_pure(
    cfg: PrecomputeConfig,
    out_npz: str,
    *,
    verbose: bool = True,
    broadcast_box: bool = False,
    index_file: Optional[str] = None,
    parallel: str = "thread",
    checkpoint_every: Optional[int] = None,
    resume: bool = False,
    progress: Optional[bool] = None,
) -> None:
    """Run CoAnaPure Stage A and save the compressed cache + per-frame CSV stats.

    Signature-compatible with :func:`oligomer_analysis.precompute.precompute_cache`.
    The trajectory is **streamed**: frames are parsed and consumed one at a time,
    so peak memory is O(``cfg.n_jobs`` frames) regardless of trajectory length.
    ``cfg.n_jobs`` (joblib convention, ``-1`` = all CPUs) sets the worker count.

    Supported trajectory formats: ``.pdb`` (single- or multi-``MODEL``), ``.gro``
    (single- or multi-frame), ``.trr``, and ``.xtc`` (``.trr``/``.xtc`` require
    ``cfg.topology_file`` as a ``.pdb`` or ``.gro``).

    Parameters
    ----------
    broadcast_box : bool
        When True, apply the first valid ``CRYST1`` box to every frame (physically
        correct for a constant-volume run that writes ``CRYST1`` only once).  The
        default (False) reproduces MDAnalysis' per-model box attachment so the
        cache is bit-comparable with the original pipeline; when that default
        silently drops a leading ``CRYST1``, a :class:`UserWarning` is emitted.
    index_file : str, optional
        PDB only.  Path to a model byte-offset index built by
        :func:`~oligomer_analysis.coanapure.pdb_io.build_model_index`.  If the
        file does not exist yet it is built and saved there.  With an index,
        sparse frame subsets seek directly to the wanted models instead of
        streaming the file prefix.  When omitted, a fresh default sidecar index
        (``<traj>.coanapure-index.npz``) is picked up automatically.
    parallel : {"thread", "process"}
        ``"thread"`` (default) parses on the calling thread and runs the contact
        math on ``cfg.n_jobs`` threads -- best for small jobs and always
        available.  ``"process"`` distributes whole frame chunks (parsing
        included) across ``cfg.n_jobs`` worker processes seeking through the
        model byte-offset index; PDB trajectories only (the index is built
        automatically on first use).  On parse-bound multi-GB PDB files this is
        the mode that scales with the cores of an HPC node.  Results are
        identical in every mode.

        .. note:: On Windows/macOS (spawn start method) a *script* using
           ``parallel="process"`` must make the call under
           ``if __name__ == "__main__":`` -- spawn re-imports the calling
           script in every worker, and an unguarded script would re-execute
           itself once per worker.  The ``coana`` CLI is already guarded, and
           Linux (fork -- the HPC case) needs no guard.
    checkpoint_every : int, optional
        Write a crash-safe checkpoint (``<out_npz>.checkpoint.npz``) after every
        N newly computed frames.  With ``resume=True`` an interrupted run picks
        up where it stopped instead of recomputing -- essential for walltime-
        limited batch jobs.  The checkpoint is fingerprinted against the full
        configuration and deleted after a successful save.
    resume : bool
        Load a matching checkpoint before computing, skipping frames it already
        holds.  A checkpoint whose configuration fingerprint (trajectory, frames,
        cutoff, selection, ...) differs is ignored with a message.
    progress : bool, optional
        Print per-interval progress lines with rate and ETA.  Defaults to the
        value of ``verbose``.

    Notes
    -----
    Three ``cfg`` fields shape a mixture run.  ``cfg.species_spec``
    (``"aSyn:295x50,Ab40:92x10"``) declares the composition once: it fills the
    monomer block sizes and is written into the cache metadata, so no later
    analysis has to be told the composition again.  ``cfg.contact_classes``
    additionally stores a per-frame residue-chemistry decomposition of the
    inter-chain contacts, and ``cfg.hotspot_blocks`` stores residue-resolved
    cross-species contact maps in that many equal time blocks.  Both extras read
    the residue-pair list that is discarded once contacts have been binned to
    segments, so neither can be added to a cache afterwards -- they cost one
    Stage A re-run, and nothing if left off.
    """
    cfg = _resolve_species_cfg(cfg, verbose)
    frames = sorted({int(f) for f in cfg.frames})
    wanted = set(frames)
    show_progress = bool(verbose if progress is None else progress)
    selection = getattr(cfg, "selection", None)

    want_classes = bool(getattr(cfg, "contact_classes", False))
    n_hotspot_blocks = int(getattr(cfg, "hotspot_blocks", None) or 0)
    if n_hotspot_blocks and not getattr(cfg, "species_spec", None):
        raise ValueError(
            "hotspot_blocks needs cfg.species_spec: a cross-species residue map "
            "is only defined once the chains have been named."
        )
    if n_hotspot_blocks < 0:
        raise ValueError("hotspot_blocks must be >= 0")

    # Blocks are contiguous, equal-length runs of the *requested* frame list, so
    # the mapping is identical whether a run completes in one go or resumes.
    block_of_frame: Dict[int, int] = {}
    block_ranges: List[List[int]] = []
    if n_hotspot_blocks:
        if n_hotspot_blocks > len(frames):
            raise ValueError(
                f"hotspot_blocks={n_hotspot_blocks} exceeds the {len(frames)} "
                "requested frames."
            )
        for b, chunk in enumerate(np.array_split(np.array(frames), n_hotspot_blocks)):
            for fr in chunk:
                block_of_frame[int(fr)] = b
            block_ranges.append([int(chunk[0]), int(chunk[-1])])

    # Populated once the topology (and therefore the residue tables) is known.
    extras: Dict[str, object] = {
        "class_of_res": None,
        "n_classes": 0,
        "species_tables": None,
        "class_census": None,
    }

    def build_extras(topo_local: dict, res_topo_local) -> None:
        """Derive the residue-class array and species tables from the topology."""
        if want_classes:
            from ..chemistry import (
                CONTACT_CLASSES,
                classify_residues,
                contact_class_census,
                describe_contact_classes,
            )

            class_of_res = classify_residues(res_topo_local.res_resname)
            extras["class_of_res"] = class_of_res
            extras["n_classes"] = len(CONTACT_CLASSES)
            extras["class_census"] = contact_class_census(class_of_res)
            if verbose:
                print(f"  contact classes: {describe_contact_classes(extras['class_census'])}", flush=True)
        if getattr(cfg, "species_spec", None):
            extras["species_tables"] = _species_tables(topo_local, cfg)

    hotspot_acc: Optional[np.ndarray] = None

    def fold_hotspots(new_results: Sequence[FrameContacts]) -> None:
        """Add each frame's residue tensor to its block sum, then release it.

        Holding every frame's ``(S, S, R, R)`` tensor until the end would cost
        hundreds of megabytes on a long trajectory for data that is only ever
        summed, so the reduction happens as results arrive.
        """
        nonlocal hotspot_acc
        if not n_hotspot_blocks:
            return
        for fc in new_results:
            if fc.hotspot is None:
                continue
            if hotspot_acc is None:
                hotspot_acc = np.zeros(
                    (n_hotspot_blocks,) + fc.hotspot.shape, dtype=np.int64
                )
            hotspot_acc[block_of_frame[int(fc.frame)]] += fc.hotspot
            fc.hotspot = None

    # ---- checkpoint / resume ------------------------------------------------
    ckpt_path = _checkpoint_path(out_npz)
    fingerprint = _config_fingerprint(cfg, broadcast_box)
    seed_results: List[FrameContacts] = []
    ckpt_topo_meta: Optional[dict] = None
    if resume:
        seed_results, ckpt_topo_meta, ckpt_hotspot = _load_checkpoint(
            ckpt_path, fingerprint, verbose
        )
        kept = [r for r in seed_results if r.frame in wanted]
        if ckpt_hotspot is not None and len(kept) == len(seed_results):
            # The hotspot sum covers exactly the checkpointed frames, so it can
            # only be reused when every one of them is still wanted.
            hotspot_acc = ckpt_hotspot
        seed_results = kept
        done = {r.frame for r in seed_results}
        wanted = wanted - done
        if verbose and seed_results:
            print(
                f"  resume: {len(seed_results)} frames restored from checkpoint, "
                f"{len(wanted)} remaining",
                flush=True,
            )

    # ---- topology (and selection) from a separate file, if given ------------
    res_topo_full: Optional[ResidueTopology] = None
    if cfg.topology_file is not None:
        res_topo_full = _load_topology_file(cfg.topology_file)
    atom_mask: Optional[np.ndarray] = None
    res_topo: Optional[ResidueTopology] = None
    if res_topo_full is not None:
        atom_mask, res_topo = apply_selection(res_topo_full, selection)

    ext = os.path.splitext(cfg.traj_file)[1].lower()
    workers = _resolve_workers(getattr(cfg, "n_jobs", 1))

    if parallel not in ("thread", "process"):
        raise ValueError(f"parallel must be 'thread' or 'process', got {parallel!r}.")
    use_processes = parallel == "process" and workers > 1
    if parallel == "process" and ext not in (".pdb", ".ent"):
        warnings.warn(
            "parallel='process' needs a PDB trajectory (frame seeking uses the "
            "model byte-offset index); falling back to threads for "
            f"{ext!r}.",
            UserWarning,
            stacklevel=2,
        )
        use_processes = False

    topo: Optional[dict] = None
    topo_meta: Optional[dict] = None
    results: List[FrameContacts] = list(seed_results)
    new_since_ckpt = 0

    def maybe_checkpoint() -> None:
        nonlocal new_since_ckpt
        if checkpoint_every and new_since_ckpt >= int(checkpoint_every) and topo_meta:
            _write_checkpoint(ckpt_path, results, fingerprint, topo_meta, hotspot_acc)
            new_since_ckpt = 0
            if verbose:
                print(f"  checkpoint: {len(results)} frames -> {ckpt_path}", flush=True)

    # ---- everything already done (full resume) ------------------------------
    if not wanted and seed_results:
        topo_meta = ckpt_topo_meta or {}
        traj_format = topo_meta.get("traj_format", ext.lstrip("."))
    elif use_processes:
        # ---- process-parallel PDB path --------------------------------------
        idx = _resolve_pdb_index(cfg.traj_file, index_file, verbose)
        if idx is None:
            sidecar = default_index_path(cfg.traj_file)
            if verbose:
                print(f"  building model index -> {sidecar}", flush=True)
            idx = PDBModelIndex.build(cfg.traj_file)
            idx.save(sidecar)
        if res_topo_full is None:
            res_topo_full = _topology_from_index(cfg.traj_file, idx)
            atom_mask, res_topo = apply_selection(res_topo_full, selection)
        if idx.n_atoms != res_topo_full.n_atoms:
            raise ValueError(
                f"Coordinate atom count {idx.n_atoms} does not match topology "
                f"atom count {res_topo_full.n_atoms}."
            )
        out_of_range = [f for f in sorted(wanted) if f >= idx.n_models]
        if out_of_range:
            raise IndexError(
                f"Requested frame {out_of_range[0]} out of range 0..{idx.n_models - 1}"
            )
        if (
            not broadcast_box
            and idx.n_models > 1
            and idx.leading_box is not None
            and not idx.has_box.any()
        ):
            # Workers run with warn_dropped_box=False; warn once from the parent.
            warnings.warn(_DROPPED_BOX_MSG, UserWarning, stacklevel=2)
        topo = build_topology_cache_pure(res_topo, cfg)
        build_extras(topo, res_topo)
        traj_format = "pdb+index"
        topo_meta = _topo_meta_dict(
            topo, res_topo_full, res_topo, selection, traj_format,
            species_spec=getattr(cfg, "species_spec", None),
            species_tables=extras["species_tables"],
            class_names=(list(_CONTACT_CLASS_NAMES()) if want_classes else None),
            class_census=extras["class_census"],
        )
        if verbose:
            print(
                f"CoAnaPure precompute: {len(wanted)} frames from {cfg.traj_file} "
                f"[{traj_format}], {workers} processes",
                flush=True,
            )
            print(f"  segment_mode={topo['segment_mode']} nseg={topo['nseg']}", flush=True)

        remaining = np.array(sorted(wanted), dtype=np.int64)
        n_chunks = max(1, min(len(remaining), workers * 4))
        chunks = [c for c in np.array_split(remaining, n_chunks) if len(c)]
        reporter = _ProgressReporter(len(remaining), show_progress)
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(
                        _process_pdb_chunk,
                        (
                            cfg.traj_file, idx, chunk.tolist(), res_topo, atom_mask, topo,
                            cfg.cutoff_contact, cfg.intra_exclude, cfg.fallback_box,
                            broadcast_box, res_topo_full.n_atoms,
                            extras["class_of_res"], extras["n_classes"],
                            extras["species_tables"],
                            getattr(cfg, "contact_mode", "residue_cog"),
                        ),
                    )
                    for chunk in chunks
                ]
                pending = set(futures)
                while pending:
                    done_futs, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done_futs:
                        chunk_results = fut.result()
                        fold_hotspots(chunk_results)
                        results.extend(chunk_results)
                        new_since_ckpt += len(chunk_results)
                        reporter.update(len(chunk_results))
                        maybe_checkpoint()
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "parallel='process' workers died during startup. On Windows and "
                "macOS, Python starts workers by re-importing the calling script "
                "(spawn), so the script must be import-safe: put the "
                "precompute_cache_pure(...) call under "
                "`if __name__ == \"__main__\":`, or use the `coana` CLI, which "
                "is already guarded. On Linux (fork; the HPC case) no guard is "
                "needed."
            ) from exc
    else:
        # ---- streaming path (serial or threaded) ----------------------------
        if ext in (".pdb", ".ent"):
            index = _resolve_pdb_index(cfg.traj_file, index_file, verbose)
            stream = PDBFrameStream(
                cfg.traj_file,
                wanted=wanted if wanted else {0},
                broadcast_box=broadcast_box,
                index=index,
                capture_topology=(res_topo_full is None),
                n_atoms_expected=(None if res_topo_full is None else res_topo_full.n_atoms),
            )
            traj_format = "pdb" + ("+index" if index is not None else "")
        elif ext == ".gro":
            from .gro_io import GROFrameStream

            stream = GROFrameStream(
                cfg.traj_file,
                wanted=wanted if wanted else {0},
                capture_topology=(res_topo_full is None),
                n_atoms_expected=(None if res_topo_full is None else res_topo_full.n_atoms),
            )
            traj_format = "gro"
        elif ext == ".trr":
            if res_topo_full is None:
                raise ValueError(
                    "A .trr trajectory carries no residue information; provide "
                    "cfg.topology_file as a .pdb or .gro file."
                )
            from .trr_io import TRRFrameStream

            stream = TRRFrameStream(
                cfg.traj_file,
                wanted=wanted if wanted else {0},
                n_atoms_expected=res_topo_full.n_atoms,
            )
            traj_format = "trr"
        elif ext == ".xtc":
            if res_topo_full is None:
                raise ValueError(
                    "An .xtc trajectory carries no residue information; provide "
                    "cfg.topology_file as a .pdb or .gro file."
                )
            from .xtc_io import XTCFrameStream

            stream = XTCFrameStream(
                cfg.traj_file,
                wanted=wanted if wanted else {0},
                n_atoms_expected=res_topo_full.n_atoms,
            )
            traj_format = "xtc"
        else:
            raise ValueError(
                f"Unsupported trajectory format {ext!r}. CoAnaPure reads .pdb, "
                ".gro, .trr and .xtc; for other formats convert with "
                "`gmx trjconv` or use the MDAnalysis engine "
                "(oligomer_analysis.precompute_cache)."
            )

        if verbose:
            print(
                f"CoAnaPure precompute: {len(wanted)} frames from {cfg.traj_file} "
                f"[{traj_format}], n_jobs={workers}",
                flush=True,
            )

        def frame_source() -> Iterator[Tuple[int, np.ndarray, Optional[np.ndarray]]]:
            nonlocal res_topo_full, res_topo, atom_mask, topo, topo_meta
            for k, pos, box in stream:
                if res_topo_full is None:
                    res_topo_full = stream.topology
                    atom_mask, res_topo = apply_selection(res_topo_full, selection)
                if topo is None:
                    topo = build_topology_cache_pure(res_topo, cfg)
                    build_extras(topo, res_topo)
                    topo_meta = _topo_meta_dict(
                        topo, res_topo_full, res_topo, selection, traj_format,
                        species_spec=getattr(cfg, "species_spec", None),
                        species_tables=extras["species_tables"],
                        class_names=(
                            list(_CONTACT_CLASS_NAMES()) if want_classes else None
                        ),
                        class_census=extras["class_census"],
                    )
                    if verbose:
                        print(f"  segment_mode={topo['segment_mode']} nseg={topo['nseg']}")
                if k in wanted:
                    yield k, pos, box

        def compute_one(k: int, pos: np.ndarray, box: Optional[np.ndarray]) -> FrameContacts:
            if atom_mask is not None:
                pos = pos[atom_mask]
            return _compute_frame_arrays(
                res_topo, topo, k, pos, box,
                cfg.cutoff_contact, cfg.intra_exclude, cfg.fallback_box,
                extras["class_of_res"], extras["n_classes"], extras["species_tables"],
                getattr(cfg, "contact_mode", "residue_cog"),
            )

        reporter = _ProgressReporter(len(wanted), show_progress)
        for fc in _run_frame_loop(frame_source(), compute_one, workers):
            fold_hotspots([fc])
            results.append(fc)
            new_since_ckpt += 1
            reporter.update()
            maybe_checkpoint()
        if not wanted:
            results = list(seed_results)  # frame 0 only pulled for the topology

    if topo_meta is None:  # pragma: no cover - stream always yields one frame
        raise ValueError("Trajectory produced no frames.")
    results.sort(key=lambda x: x.frame)

    frames_arr = np.array([r.frame for r in results], dtype=np.int32)
    meta = {
        "topology_file": cfg.topology_file,
        "traj_file": cfg.traj_file,
        "cutoff_contact": cfg.cutoff_contact,
        "contact_mode": getattr(cfg, "contact_mode", "residue_cog"),
        "intra_exclude": cfg.intra_exclude,
        "atoms_per_monomer": getattr(cfg, "atoms_per_monomer", None),
        "residues_per_monomer": getattr(cfg, "residues_per_monomer", None),
        "n_monomers": getattr(cfg, "n_monomers", None),
        "fallback_box": cfg.fallback_box,
        "frames_min": int(frames_arr.min()) if len(frames_arr) else None,
        "frames_max": int(frames_arr.max()) if len(frames_arr) else None,
        "n_frames": int(len(frames_arr)),
        "engine": ENGINE_VERSION,
        "parallel": ("process" if use_processes else "thread"),
        **topo_meta,
    }
    has_box = any(r.box is not None and np.any(np.asarray(r.box)[:3] > 0) for r in results)
    meta["has_box"] = bool(has_box)
    if not has_box:
        warnings.warn(NO_BOX_MSG, NoBoxWarning, stacklevel=2)
    if hotspot_acc is not None:
        meta["hotspot_blocks"] = block_ranges
        tables = extras["species_tables"] or {}
        meta["hotspot_residues_per_species"] = list(tables.get("res_per_species", []))

    save_cache(
        out_npz,
        results=results,
        meta=meta,
        save_csv_stats=True,
        hotspot_maps=hotspot_acc,
    )
    if os.path.exists(ckpt_path):
        try:
            os.remove(ckpt_path)
        except OSError:
            pass
    if verbose:
        print(f"Saved cache: {out_npz}")
        print(f"Saved stats: {out_npz.replace('.npz', '_contact_stats.csv')}")
