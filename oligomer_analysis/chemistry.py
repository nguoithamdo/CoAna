# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Interaction-type decomposition of inter-chain contacts.

A contact map answers *that* two chains stick together.  It cannot answer *why*,
and "why" is the part a referee asks about: an aggregate held together by a
hydrophobic core behaves differently under salt than one held together by salt
bridges, yet both look identical in ``ninter``.

This module assigns every residue a chemistry class and lets Stage A accumulate
the inter-chain residue contacts into a small ``(n_classes, n_classes)`` matrix
per frame.  The classes are deliberately coarse and sequence-derived, so they
apply to any force field that keeps standard residue names -- Martini included,
where one coarse-grained residue is one amino acid:

======== ==================================================
class    residues
======== ==================================================
apolar   ALA CYS GLY ILE LEU MET PRO VAL
aromatic PHE TRP TYR
polar    ASN GLN SER THR, neutral HIS tautomers, and the neutral
         protonation variants (ASPP/ASPH/ASH, GLUP/GLUH/GLH,
         LYSN/LSN/LYN, ARGN)
cation   ARG LYS LYSH and the protonated HIS variants (HSP/HISH/HIP)
anion    ASP GLU
other    everything else -- ions, water, PEG, lipids, caps
======== ==================================================

Charge, not the residue name, decides the class.  A name such as ``ASPP`` or
``GLUP`` means *protonated* Asp/Glu, which is electrically neutral, and
``ARGN`` means neutral Arg; filing those as anion/cation would report salt
bridges between side chains that carry no charge.  Pass ``overrides`` to
:func:`classify_residues` for a force field that names them differently.

The pairing of interest is then read straight off the matrix:
``cation-anion`` is the salt-bridge channel, ``apolar-apolar`` plus
``aromatic-*`` is the hydrophobic channel.  :func:`contact_class_summary` turns
those into fractions, and :func:`salt_bridge_timeseries` pulls out the single
channel a salt titration is supposed to move.

Stage A must be asked for this (``coana precompute --contact-classes``); it is
not recoverable from an existing cache, because the residue-pair list is
discarded once it has been binned to segments.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "CONTACT_CLASSES",
    "RESIDUE_CLASS",
    "classify_residues",
    "contact_class_dataframe",
    "contact_class_matrix",
    "contact_class_summary",
    "salt_bridge_timeseries",
    "hydrophobic_fraction_timeseries",
    "contact_class_census",
    "describe_contact_classes",
    "oligomer_class_matrix",
    "segment_chemistry_timeseries",
    "oligomer_chemistry_tables",
]


#: Class names, in the order they index ``cache.class_cmaps``.
CONTACT_CLASSES: Tuple[str, ...] = (
    "apolar",
    "aromatic",
    "polar",
    "cation",
    "anion",
    "other",
)

_APOLAR = (
    "ALA", "CYS", "GLY", "ILE", "LEU", "MET", "PRO", "VAL",
    # disulfide-bonded (AMBER CYX) and explicitly-neutral (ffAMBER CYN)
    # cysteine: both uncharged, so they belong with CYS rather than with the
    # thiolate CYM in _ANION
    "CYX", "CYN",
)
_AROMATIC = ("PHE", "TRP", "TYR")
# Neutral forms live here, including the protonation variants whose names say
# "protonated" but whose *charge* is zero.  Getting this wrong is not cosmetic:
# cation-anion is the salt-bridge channel, so a neutral residue filed as charged
# invents salt bridges that the force field never modelled.  ASPP/ASPH/ASH are
# protonated (neutral) Asp and GLUP/GLUH/GLH protonated (neutral) Glu -- the
# proton neutralises the carboxylate.  LYSN/LSN/LYN are deprotonated (neutral)
# Lys.  ARGN is neutral Arg.  HID/HIE/HSD/HSE/HISD/HISE are the neutral His
# tautomers, as against the protonated HSP/HISH/HIP, which are cationic.
_POLAR = (
    "ASN", "GLN", "SER", "THR",
    # neutral histidine tautomers (CHARMM HSD/HSE, GROMACS HISD/HISE, AMBER
    # HID/HIE) as against the protonated, cationic forms below
    "HIS", "HSD", "HSE", "HISD", "HISE", "HID", "HIE",
    # protonated -- hence NEUTRAL -- Asp and Glu
    "ASPP", "ASPH", "ASH", "ASP0",
    "GLUP", "GLUH", "GLH", "GLU0",
    # deprotonated -- hence neutral -- Lys (GROMACS LYSN, CHARMM/ffAMBER LSN,
    # AMBER LYN, Martini-2 LSN), and GROMACS neutral Arg
    "LYSN", "LSN", "LYN",
    "ARGN",
)
_CATION = (
    "ARG", "LYS",
    "LYSH", "LYP",                      # protonated, charged lysine
    "HSP", "HISH", "HIP", "HIH",        # protonated, cationic histidine
)
_ANION = (
    "ASP", "GLU",
    "CYM",                              # deprotonated cysteine (thiolate)
)

#: Residue name -> class name.  Case-insensitive lookups go through
#: :func:`classify_residues`, which also strips whitespace.
#:
#: Every entry above is a name verified against a published convention
#: (GROMACS pdb2gmx, AMBER/ffAMBER, CHARMM, or the Martini-2 amino-acid
#: topology).  The table is deliberately not padded with plausible-looking
#: variants: an unrecognised name falls to ``other``, and
#: :func:`classify_residues` warns and names it, which is safer than guessing a
#: charge.  Add what your force field actually uses via ``overrides``.
RESIDUE_CLASS: Dict[str, str] = {}
for _names, _cls in (
    (_APOLAR, "apolar"),
    (_AROMATIC, "aromatic"),
    (_POLAR, "polar"),
    (_CATION, "cation"),
    (_ANION, "anion"),
):
    for _n in _names:
        RESIDUE_CLASS[_n] = _cls
del _names, _cls, _n

_CLASS_INDEX = {name: i for i, name in enumerate(CONTACT_CLASSES)}
_OTHER_INDEX = _CLASS_INDEX["other"]

#: Names that belong in ``other`` and are not worth warning about: solvent,
#: ions and the common Martini non-protein beads. Warning on these would fire on
#: every real system and train the reader to ignore the warning.
_NON_PROTEIN = frozenset({
    "W", "WF", "PW", "SOL", "HOH", "TIP3", "SPC",
    "ION", "NA", "NA+", "CL", "CL-", "K", "K+", "CA", "CA2+", "MG", "MG2+",
    "ZN", "BR", "I", "F",
    "PEG", "PEO", "DUM", "DUMMY",
    "ACE", "NAC", "NH2", "NME", "NHE", "CT3",       # terminal caps
})


def classify_residues(
    resnames: Sequence[str],
    *,
    overrides: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Map residue names to contact-class indices.

    Unknown names fall into ``other`` rather than raising: a Martini box is full
    of legitimate non-protein residues (W, ION, PEG), and a hard failure there
    would make the feature unusable on exactly the systems it is for.

    Parameters
    ----------
    resnames:
        One residue name per residue, in topology order.
    overrides:
        Extra or replacement ``{resname: class}`` entries, e.g.
        ``{"PEG": "apolar"}``.  Class names must be in :data:`CONTACT_CLASSES`.
    """
    table = dict(RESIDUE_CLASS)
    for name, cls in (overrides or {}).items():
        if cls not in _CLASS_INDEX:
            raise ValueError(
                f"override for {name!r} names class {cls!r}; "
                f"expected one of {list(CONTACT_CLASSES)}"
            )
        table[str(name).strip().upper()] = cls

    out = np.full(len(resnames), _OTHER_INDEX, dtype=np.int8)
    unknown: Dict[str, int] = {}
    for i, raw in enumerate(resnames):
        key = str(raw).strip().upper()
        cls = table.get(key)
        if cls is not None:
            out[i] = _CLASS_INDEX[cls]
        else:
            unknown[key] = unknown.get(key, 0) + 1

    # Falling back to ``other`` is right -- a box is full of legitimate
    # non-protein residues -- but doing it silently is not. A residue the table
    # does not know contributes to no channel at all, so an unrecognised
    # protonation variant (say a force field spelling neutral Asp some way not
    # listed) quietly removes those contacts from the analysis instead of
    # misfiling them, and the census then shows only a large ``other`` count
    # with no clue what is in it. Name them.
    protein_like = {k: v for k, v in unknown.items() if k not in _NON_PROTEIN}
    if protein_like:
        shown = ", ".join(
            f"{k} x{v}" for k, v in sorted(
                protein_like.items(), key=lambda kv: -kv[1])[:12]
        )
        more = "" if len(protein_like) <= 12 else f" (+{len(protein_like) - 12} more)"
        warnings.warn(
            f"contact-class table does not recognise {len(protein_like)} residue "
            f"name(s): {shown}{more}. They are counted as 'other' and so "
            "contribute to no interaction channel -- including the salt-bridge "
            "channel, which will be an undercount if any of them is charged. "
            "If these are protonation variants or non-standard residues, pass "
            "classify_residues(overrides={'NAME': 'cation'|'anion'|'polar'|"
            "'apolar'|'aromatic'}) to place them.",
            UserWarning,
            stacklevel=2,
        )
    return out


def contact_class_census(class_of_res: np.ndarray) -> Dict[str, int]:
    """How many residues landed in each class, as ``{class: count}``.

    Only classes that are actually populated appear, so a box with no aromatic
    residues has no ``aromatic`` key rather than a zero.
    """
    counts = np.bincount(np.asarray(class_of_res, dtype=np.int64), minlength=len(CONTACT_CLASSES))
    return {name: int(c) for name, c in zip(CONTACT_CLASSES, counts) if c}


def describe_contact_classes(census) -> str:
    """One-line rendering of a census (or of a class-index array)."""
    if isinstance(census, str):
        return census  # already rendered (caches written before the dict form)
    if not isinstance(census, dict):
        census = contact_class_census(census)
    return ", ".join(f"{name}={int(c)}" for name, c in census.items())


# --------------------------------------------------------------------------
# post-hoc analysis of a cache that carries class_cmaps
# --------------------------------------------------------------------------


def _require_class_cmaps(cache) -> Tuple[np.ndarray, List[str]]:
    if getattr(cache, "class_cmaps", None) is None:
        raise ValueError(
            "this cache carries no contact-class decomposition. Rebuild Stage A "
            "with `coana precompute --contact-classes` (or "
            "PrecomputeConfig(contact_classes=True)); the residue-pair list it "
            "needs is not recoverable from a finished cache."
        )
    names = list(cache.meta.get("contact_classes") or CONTACT_CLASSES)
    return np.asarray(cache.class_cmaps), names


def contact_class_matrix(cache, *, normalize: bool = False) -> np.ndarray:
    """The ``(n_frames, n_classes, n_classes)`` inter-chain contact tensor.

    Entry ``[f, i, j]`` counts residue-residue contacts between *different*
    chains where one residue is of class ``i`` and the other of class ``j``.
    The matrix is symmetric and its diagonal is already the same-class count
    (not double-counted).  With ``normalize=True`` each frame is divided by its
    own total, giving interaction-type fractions that can be compared between
    systems of different size.

    Note what ``normalize=True`` does and does not sum to.  The denominator is
    the number of *distinct* contacts, so the UPPER TRIANGLE sums to 1 -- which
    is the set :func:`contact_class_dataframe` emits and
    :func:`contact_class_summary` reports.  The full symmetric matrix sums to
    more than 1, because every heterotypic fraction appears in both
    orientations.  Sum the upper triangle, not the whole array.
    """
    matrix, _names = _require_class_cmaps(cache)
    out = matrix.astype(np.float64)
    if normalize:
        # The matrix is symmetric, so summing all of it counts every
        # heterotypic pair twice while counting each homotypic pair once. The
        # denominator must be the number of DISTINCT contacts, which is the
        # upper triangle including the diagonal -- the same total
        # contact_class_summary uses, and the same set contact_class_dataframe
        # emits. Dividing by the full sum understated every fraction by
        # U/(U+H): exactly 2x when all contacts are heterotypic, and the
        # emitted fractions summed to 0.5 instead of 1.
        tri = np.triu(np.ones(out.shape[-2:], dtype=bool))
        totals = (out * tri).sum(axis=(1, 2), keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(totals > 0, out / totals, 0.0)
    return out


def contact_class_dataframe(cache, *, normalize: bool = False) -> pd.DataFrame:
    """Long-form ``(frame, class_a, class_b, contacts)`` table."""
    matrix = contact_class_matrix(cache, normalize=normalize)
    _m, names = _require_class_cmaps(cache)
    frames = np.asarray(cache.frames)
    rows = []
    for a in range(len(names)):
        for b in range(a, len(names)):
            for i in range(matrix.shape[0]):
                rows.append(
                    {
                        "frame": int(frames[i]) if i < len(frames) else i,
                        "class_a": names[a],
                        "class_b": names[b],
                        "contacts": float(matrix[i, a, b]),
                    }
                )
    return pd.DataFrame(rows)


def contact_class_summary(cache, *, last_fraction: float = 0.25) -> pd.DataFrame:
    """Mean contacts per class pair over the whole run and over its tail.

    The tail (the last ``last_fraction`` of frames, 25% by default) is what to
    quote for an equilibrated assembly; the whole-run mean includes the approach
    to it.  Columns: ``class_a``, ``class_b``, ``mean_all``, ``mean_late``,
    ``fraction_late`` and ``channel``, where ``channel`` names the physical
    interaction the pair represents.
    """
    matrix, names = _require_class_cmaps(cache)
    matrix = matrix.astype(np.float64)
    n_frames = matrix.shape[0]
    if not 0.0 < last_fraction <= 1.0:
        raise ValueError("last_fraction must be in (0, 1]")
    lo = max(0, n_frames - max(1, int(round(n_frames * last_fraction))))
    late = matrix[lo:]
    # The matrix is symmetric and the rows below enumerate each unordered pair
    # once, so the total must be the upper triangle, not the full sum.
    late_total = np.triu(late).sum(axis=(1, 2)).mean()

    rows = []
    for a in range(len(names)):
        for b in range(a, len(names)):
            mean_all = float(matrix[:, a, b].mean()) if n_frames else 0.0
            mean_late = float(late[:, a, b].mean()) if len(late) else 0.0
            rows.append(
                {
                    "class_a": names[a],
                    "class_b": names[b],
                    "channel": _channel_name(names[a], names[b]),
                    "mean_all": mean_all,
                    "mean_late": mean_late,
                    "fraction_late": (
                        float(mean_late / late_total) if late_total > 0 else float("nan")
                    ),
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values("mean_late", ascending=False).reset_index(drop=True)


def _channel_name(a: str, b: str) -> str:
    """Physical label for a class pair, so the table reads without a legend."""
    pair = {a, b}
    if pair == {"cation", "anion"}:
        return "salt bridge"
    if pair <= {"cation"} or pair <= {"anion"}:
        return "like-charge repulsive contact"
    if pair <= {"apolar", "aromatic"}:
        return "hydrophobic"
    if "aromatic" in pair and pair & {"cation"}:
        return "cation-pi"
    # polar-polar must be tested before the mixed case: {"polar"} is a subset of
    # {"polar", "apolar", "aromatic"}, so testing the mixed case first filed every
    # polar-polar contact under "mixed polar/apolar".
    if pair == {"polar"}:
        return "polar"
    if pair <= {"polar", "apolar", "aromatic"}:
        return "mixed polar/apolar"
    if "other" in pair:
        return "involves unclassified residues"
    return "polar/charged"


def salt_bridge_timeseries(cache) -> np.ndarray:
    """Per-frame cation-anion inter-chain contacts.

    This is the channel a salt titration is meant to screen.  Reporting it
    directly turns "aggregation weakens with added NaCl, so it must be
    electrostatic" into a measurement.
    """
    matrix, names = _require_class_cmaps(cache)
    try:
        i, j = names.index("cation"), names.index("anion")
    except ValueError as exc:  # pragma: no cover - only with custom class sets
        raise ValueError(
            f"cache classes {names} do not include 'cation' and 'anion'"
        ) from exc
    return matrix[:, i, j].astype(np.float64)


def hydrophobic_fraction_timeseries(cache) -> np.ndarray:
    """Per-frame fraction of inter-chain contacts that are apolar/aromatic."""
    matrix, names = _require_class_cmaps(cache)
    idx = [names.index(n) for n in ("apolar", "aromatic") if n in names]
    if not idx:  # pragma: no cover - only with custom class sets
        raise ValueError(f"cache classes {names} include no hydrophobic class")
    sub = matrix[:, idx][:, :, idx]  # (n_frames, k, k); np.triu acts on the last two axes
    hydro = np.triu(sub).sum(axis=(1, 2)).astype(np.float64)
    total = np.triu(matrix).sum(axis=(1, 2)).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, hydro / total, np.nan)


# ---------------------------------------------------------------------------
# per-oligomer chemistry
# ---------------------------------------------------------------------------
#
# The system-wide class matrix answers "how do chains stick in this box". The
# per-chain-pair rows answer it for one oligomer: summing the rows whose two
# chains both belong to the oligomer gives the class matrix of the contacts that
# hold that oligomer together, and following the oligomer through its segment
# and lineage gives that chemistry over its lifetime.


def _require_pair_classes(cache) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if getattr(cache, "pair_class_rows", None) is None:
        raise ValueError(
            "this cache carries no per-chain-pair contact classes. Rebuild Stage A "
            "with `coana precompute --contact-classes` (coana 3.8 or later); "
            "earlier caches store the class decomposition for the whole system only."
        )
    names = list(cache.meta.get("contact_classes") or CONTACT_CLASSES)
    return cache.pair_class_rows, cache.pair_class_offsets, names


def _channel_key(a: str, b: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in _channel_name(a, b)).strip("_")


def _channel_index(names: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    """Channel names and, for every (i, j) class pair, the channel it belongs to."""
    keys: List[str] = []
    lookup = np.zeros((len(names), len(names)), dtype=np.int64)
    for i in range(len(names)):
        for j in range(i, len(names)):
            key = _channel_key(names[i], names[j])
            if key not in keys:
                keys.append(key)
            lookup[i, j] = lookup[j, i] = keys.index(key)
    return keys, lookup


def oligomer_class_matrix(cache, frame_index: int, members) -> np.ndarray:
    """Class-pair matrix of the inter-chain contacts inside one oligomer.

    ``frame_index`` is a cache index (as in ``segment.start_idx``), ``members``
    the chain indices of the oligomer in that frame. Only contacts between two
    member chains are counted; contacts with chains outside the oligomer are
    not. The matrix follows the convention of ``cache.class_cmaps``: symmetric,
    with ``np.triu(M).sum()`` equal to the number of residue contacts.
    """
    rows, offsets, names = _require_pair_classes(cache)
    n = len(names)
    r = rows[offsets[frame_index]:offsets[frame_index + 1]]
    member_idx = np.fromiter((int(m) for m in members), dtype=np.int64)
    mask = np.zeros(cache.cmaps.shape[1], dtype=bool)
    mask[member_idx] = True
    r = r[mask[r[:, 0]] & mask[r[:, 1]]]
    raw = np.zeros((n, n), dtype=np.int64)
    np.add.at(raw, (r[:, 2], r[:, 3]), r[:, 4])
    matrix = raw + raw.T
    np.fill_diagonal(matrix, np.diag(raw))
    return matrix


def _channel_counts(matrix: np.ndarray, keys: List[str], lookup: np.ndarray) -> np.ndarray:
    upper = np.triu(matrix)
    return np.bincount(lookup.ravel(), weights=upper.ravel(), minlength=len(keys))


def segment_chemistry_timeseries(cache, segment) -> pd.DataFrame:
    """Contact chemistry of one oligomer segment, frame by frame.

    One row per frame of the segment: its size, the residue contacts between
    its chains, the contacts in each interaction channel (salt bridge,
    hydrophobic, ...) and the salt-bridge and hydrophobic fractions.
    """
    _rows, _offsets, names = _require_pair_classes(cache)
    keys, lookup = _channel_index(names)
    out = []
    for k, members in enumerate(segment.sets):
        idx = int(segment.start_idx) + k
        counts = _channel_counts(oligomer_class_matrix(cache, idx, members), keys, lookup)
        total = float(counts.sum())
        row = {
            "segment_id": segment.segment_id,
            "lineage_id": segment.lineage_id,
            "frame_idx": idx,
            "frame": int(cache.frames[idx]),
            "size": len(members),
            "contacts": int(total),
        }
        row.update({key: int(c) for key, c in zip(keys, counts)})
        for key in ("salt_bridge", "hydrophobic"):
            if key in keys:
                row[f"{key}_fraction"] = (
                    float(counts[keys.index(key)] / total) if total > 0 else float("nan")
                )
        out.append(row)
    return pd.DataFrame(out)


def _pooled_rows(label: dict, sums: np.ndarray, n_frames: int, keys: List[str]) -> dict:
    total = float(sums.sum())
    row = dict(label)
    row["frames"] = int(n_frames)
    row["contacts"] = int(total)
    row["mean_contacts"] = total / n_frames if n_frames else float("nan")
    for key, s in zip(keys, sums):
        row[f"{key}_fraction"] = float(s / total) if total > 0 else float("nan")
    return row


def oligomer_chemistry_tables(cache, result) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Contact chemistry of every oligomer segment and of every lineage.

    Returns ``(segments, lineages)``. Each row pools the contacts of all frames
    of the segment (or of all segments of the lineage) and gives the fraction of
    them in each interaction channel, so a channel fraction is weighted by
    contacts, not averaged over frames. ``mean_contacts`` is the mean number of
    residue contacts holding the oligomer together per frame.
    """
    _rows, _offsets, names = _require_pair_classes(cache)
    keys, lookup = _channel_index(names)
    seg_rows, lin_sums, lin_frames = [], {}, {}
    for seg in result.segments:
        sums = np.zeros(len(keys), dtype=np.float64)
        for k, members in enumerate(seg.sets):
            sums += _channel_counts(
                oligomer_class_matrix(cache, int(seg.start_idx) + k, members), keys, lookup
            )
        n = len(seg.sets)
        seg_rows.append(_pooled_rows(
            {"segment_id": seg.segment_id, "lineage_id": seg.lineage_id,
             "start_frame": seg.start_frame, "max_size": seg.max_size},
            sums, n, keys,
        ))
        lid = seg.lineage_id
        lin_sums[lid] = lin_sums.get(lid, 0.0) + sums
        lin_frames[lid] = lin_frames.get(lid, 0) + n
    lin_rows = [
        _pooled_rows({"lineage_id": lid}, lin_sums[lid], lin_frames[lid], keys)
        for lid in lin_sums
    ]
    return pd.DataFrame(seg_rows), pd.DataFrame(lin_rows)
