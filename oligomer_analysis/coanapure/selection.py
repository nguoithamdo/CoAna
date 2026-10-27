# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Atom selection for CoAnaPure Stage A.

Restricting the contact analysis to a subset of atoms (backbone beads only,
protein without solvent, one chain, ...) changes what "contact" means, so the
selection is applied *before* segmentation: block-based segment modes then count
selected atoms, e.g. a monomer of 83 beads with 20 backbone beads uses
``selection="resname BB"`` together with ``atoms_per_monomer=20``.

The grammar is deliberately small and unambiguous -- one clause, optionally
negated::

    all
    resname NAME [NAME ...]          e.g. "resname BB SC1"
    not resname NAME [NAME ...]      e.g. "not resname W ION NA CL"
    resid RANGE [RANGE ...]          e.g. "resid 1-128 200"
    not resid RANGE [RANGE ...]
    chain ID [ID ...]                e.g. "chain A B"
    not chain ID [ID ...]
    index RANGE [RANGE ...]          e.g. "index 0-9959" (0-based atom index)
    not index RANGE [RANGE ...]

``RANGE`` is ``N`` or ``A-B``/``A:B`` (inclusive on both ends).  ``resname``,
``resid`` and ``chain`` select whole residues from the topology; ``index``
selects individual atoms.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .pdb_io import ResidueTopology

__all__ = ["parse_selection", "atom_mask_from_selection", "apply_selection"]

_KEYWORDS = ("resname", "resid", "chain", "index")

_GRAMMAR_HINT = (
    "Supported selections: 'all', '[not] resname NAME...', '[not] resid RANGE...', "
    "'[not] chain ID...', '[not] index RANGE...' (RANGE = N or A-B, inclusive)."
)


def _parse_ranges(tokens: List[str], keyword: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for tok in tokens:
        t = tok.replace(":", "-")
        lo: Optional[int] = None
        hi: Optional[int] = None
        try:
            lo = hi = int(t)
        except ValueError:
            # Split at a '-' that is not a leading sign, so "-5" and "-5--2" work.
            sep = t.find("-", 1)
            if sep > 0:
                try:
                    lo, hi = int(t[:sep]), int(t[sep + 1 :])
                except ValueError:
                    lo = hi = None
        if lo is None or hi is None:
            raise ValueError(f"Bad {keyword} range {tok!r}. {_GRAMMAR_HINT}")
        if hi < lo:
            raise ValueError(f"Empty {keyword} range {tok!r} (upper bound below lower).")
        ranges.append((lo, hi))
    return ranges


def parse_selection(selection: str) -> Tuple[bool, str, List[str]]:
    """Parse a selection string into ``(negate, keyword, tokens)``.

    ``"all"`` parses to ``(False, "all", [])``.
    """
    tokens = selection.split()
    if not tokens:
        raise ValueError(f"Empty selection string. {_GRAMMAR_HINT}")
    negate = False
    if tokens[0].lower() == "not":
        negate = True
        tokens = tokens[1:]
        if not tokens:
            raise ValueError(f"Selection 'not' needs a clause. {_GRAMMAR_HINT}")
    keyword = tokens[0].lower()
    args = tokens[1:]
    if keyword == "all":
        if negate or args:
            raise ValueError(f"'all' takes no arguments. {_GRAMMAR_HINT}")
        return False, "all", []
    if keyword not in _KEYWORDS:
        raise ValueError(f"Unknown selection keyword {keyword!r}. {_GRAMMAR_HINT}")
    if not args:
        raise ValueError(f"Selection {keyword!r} needs at least one argument. {_GRAMMAR_HINT}")
    return negate, keyword, args


def atom_mask_from_selection(topo: ResidueTopology, selection: str) -> np.ndarray:
    """Boolean per-atom keep-mask for ``selection`` over ``topo``."""
    negate, keyword, args = parse_selection(selection)
    n_atoms = int(topo.n_atoms)

    if keyword == "all":
        return np.ones(n_atoms, dtype=bool)

    if keyword == "index":
        mask = np.zeros(n_atoms, dtype=bool)
        for lo, hi in _parse_ranges(args, "index"):
            if lo < 0 or hi >= n_atoms:
                raise ValueError(
                    f"index range {lo}-{hi} outside 0..{n_atoms - 1}."
                )
            mask[lo : hi + 1] = True
    else:
        # Residue-level clauses: build a per-residue mask, then expand to atoms.
        if keyword == "resname":
            wanted = {a.upper() for a in args}
            res_mask = np.array(
                [name.strip().upper() in wanted for name in topo.res_resname], dtype=bool
            )
        elif keyword == "chain":
            wanted = {a for a in args}
            res_mask = np.array([ch.strip() in wanted for ch in topo.res_chain], dtype=bool)
        else:  # resid
            res_mask = np.zeros(int(topo.n_residues), dtype=bool)
            resids = topo.res_resid
            for lo, hi in _parse_ranges(args, "resid"):
                res_mask |= (resids >= lo) & (resids <= hi)
        mask = res_mask[topo.residue_of_atom]

    if negate:
        mask = ~mask
    if not mask.any():
        raise ValueError(f"Selection {selection!r} matches no atoms.")
    return mask


def apply_selection(
    topo: ResidueTopology, selection: Optional[str]
) -> Tuple[Optional[np.ndarray], ResidueTopology]:
    """Return ``(atom_mask, selected_topology)`` for ``selection``.

    ``atom_mask`` is ``None`` when the selection keeps every atom (so callers
    can skip per-frame filtering).  The returned topology re-derives the residue
    tables over the kept atoms only, preserving file order; a residue that loses
    every atom disappears, one that loses some atoms keeps the remainder.
    """
    if selection is None or selection.strip().lower() == "all":
        return None, topo
    mask = atom_mask_from_selection(topo, selection)
    if mask.all():
        return None, topo

    old_res_of_atom = topo.residue_of_atom[mask]
    # Kept residues in file order, remapped to a dense range.
    kept_res, new_res_of_atom = np.unique(old_res_of_atom, return_inverse=True)
    new_res_of_atom = new_res_of_atom.astype(np.int32)
    res_atom_counts = np.bincount(new_res_of_atom, minlength=len(kept_res)).astype(np.int32)
    res_first_atom = np.concatenate(
        ([0], np.cumsum(res_atom_counts[:-1]))
    ).astype(np.int64)

    sub = ResidueTopology(
        n_atoms=int(mask.sum()),
        n_residues=int(len(kept_res)),
        residue_of_atom=new_res_of_atom,
        res_first_atom=res_first_atom,
        res_atom_counts=res_atom_counts,
        res_resid=topo.res_resid[kept_res],
        res_resname=[topo.res_resname[int(r)] for r in kept_res],
        res_chain=[topo.res_chain[int(r)] for r in kept_res],
    )
    return mask, sub
