# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
from __future__ import annotations

"""Contact hotspots, at chain resolution and at residue resolution.

Two layers live here, distinguished by what Stage A stored.

**Chain level** (always available).  Which chains or chain pairs are enriched in
on-pathway versus off-pathway lineages, computed from the segment-segment
contact maps every cache carries.

**Residue level** (needs ``coana precompute --hotspot-blocks``).  Which part of
one sequence touches which part of another -- the question behind every claim
that a particular region drives an interaction.  For alpha-synuclein and a
partner, "does the partner bind the aggregation-prone NAC region or the acidic
C-terminus?" is exactly this measurement, and it cannot be answered from
chain-level contacts at all.

The residue tensor is accumulated in time blocks, so an interface that forms
late can be told from one present throughout.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import ContactCache, LineageTrackingResult


def _lineage_class_map(result: LineageTrackingResult) -> dict[str, str]:
    return {lin.lineage_id: lin.classification for lin in result.lineages}


def contact_frequency_by_fate(cache: ContactCache, result: LineageTrackingResult, *, inter_thresh: int) -> pd.DataFrame:
    """Compute internal segment-pair contact frequency in on/off/uncertain lineages."""
    class_map = _lineage_class_map(result)
    counts: dict[tuple[str, int, int], int] = {}
    totals: dict[str, int] = {}

    for seg in result.segments:
        fate = class_map.get(seg.lineage_id or "", "uncertain")
        for local_t, members in enumerate(seg.sets):
            idx = seg.start_idx + local_t
            mem = sorted(list(members))
            if len(mem) < 2:
                continue
            totals[fate] = totals.get(fate, 0) + 1
            cmap = cache.cmaps[idx]
            for i in range(len(mem) - 1):
                for j in range(i + 1, len(mem)):
                    a, b = mem[i], mem[j]
                    if cmap[a, b] >= inter_thresh:
                        key = (fate, int(a), int(b))
                        counts[key] = counts.get(key, 0) + 1

    rows = []
    for (fate, a, b), n in counts.items():
        rows.append({
            "fate": fate,
            "seg_i": a,
            "seg_j": b,
            "contact_count": n,
            "n_snapshots_fate": totals.get(fate, 0),
            "frequency": n / max(1, totals.get(fate, 0)),
        })
    return pd.DataFrame(rows)


def contact_enrichment_on_vs_off(freq_df: pd.DataFrame, *, pseudocount: float = 1e-3) -> pd.DataFrame:
    """Compute on/off enrichment for segment-pair contacts."""
    if freq_df.empty:
        return pd.DataFrame()
    pivot = freq_df.pivot_table(index=["seg_i", "seg_j"], columns="fate", values="frequency", fill_value=0.0).reset_index()
    if "on-pathway" not in pivot:
        pivot["on-pathway"] = 0.0
    if "off-pathway" not in pivot:
        pivot["off-pathway"] = 0.0
    pivot["on_frequency"] = pivot["on-pathway"]
    pivot["off_frequency"] = pivot["off-pathway"]
    pivot["enrichment_on_vs_off"] = (pivot["on_frequency"] + pseudocount) / (pivot["off_frequency"] + pseudocount)
    pivot["log2_enrichment_on_vs_off"] = np.log2(pivot["enrichment_on_vs_off"])
    return pivot.sort_values("log2_enrichment_on_vs_off", ascending=False)


def rank_contact_hotspots(cache: ContactCache, result: LineageTrackingResult, *, inter_thresh: int, top_n: int = 20) -> pd.DataFrame:
    """Return top segment-pair contacts enriched in on-pathway lineages."""
    freq = contact_frequency_by_fate(cache, result, inter_thresh=inter_thresh)
    enrich = contact_enrichment_on_vs_off(freq)
    return enrich.head(int(top_n))


# --------------------------------------------------------------------------
# residue-level cross-species hotspots
# --------------------------------------------------------------------------


def _require_hotspots(cache: ContactCache) -> Tuple[np.ndarray, List[str], List[int]]:
    if getattr(cache, "hotspot_maps", None) is None:
        raise ValueError(
            "this cache carries no residue-resolved hotspot maps. Rebuild Stage A "
            "with `coana precompute --species ... --hotspot-blocks N` (or "
            "PrecomputeConfig(hotspot_blocks=N)); residue identity is discarded "
            "once contacts have been binned to chains."
        )
    names = [str(d["name"]) for d in (cache.meta.get("species") or [])]
    if not names:
        names = [f"sp{i}" for i in range(np.asarray(cache.hotspot_maps).shape[1])]
    res_per_species = [
        int(x) for x in (cache.meta.get("hotspot_residues_per_species") or [])
    ]
    if not res_per_species:
        res_per_species = [np.asarray(cache.hotspot_maps).shape[-1]] * len(names)
    return np.asarray(cache.hotspot_maps), names, res_per_species


def residue_hotspot_matrix(
    cache: ContactCache,
    species_a: str | int,
    species_b: str | int,
    *,
    blocks: Optional[Sequence[int]] = None,
    normalize: bool = False,
) -> np.ndarray:
    """Residue-residue contact counts between two species.

    Returns an ``(n_res_a, n_res_b)`` matrix summed over the requested time
    blocks (all of them by default) and over every chain pair, trimmed to each
    species' real residue count.  With ``normalize=True`` the matrix is divided
    by its own total, giving the fraction of the interface carried by each
    residue pair.

    For ``species_a == species_b`` the matrix is the *inter-chain* self-contact
    map -- how one chain of that species packs against another -- not an
    intra-chain map.
    """
    maps, names, res_per_species = _require_hotspots(cache)
    ia = names.index(species_a) if isinstance(species_a, str) else int(species_a)
    ib = names.index(species_b) if isinstance(species_b, str) else int(species_b)

    sel = maps if blocks is None else maps[list(blocks)]
    block = sel.sum(axis=0)[ia, ib]
    out = block[: res_per_species[ia], : res_per_species[ib]].astype(np.float64)
    if normalize:
        total = out.sum()
        if total > 0:
            out = out / total
    return out


def residue_hotspot_profile(
    cache: ContactCache,
    species_a: str | int,
    species_b: str | int,
    *,
    blocks: Optional[Sequence[int]] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Per-residue contact profile of ``species_a`` against ``species_b``.

    One row per residue of ``species_a`` with its total contacts to ``species_b``
    -- the one-dimensional projection that maps directly onto a sequence plot,
    and the form in which a "region X drives binding" claim is normally made.
    ``resid`` is 1-based to match the sequence numbering.
    """
    matrix = residue_hotspot_matrix(
        cache, species_a, species_b, blocks=blocks, normalize=False
    )
    profile = matrix.sum(axis=1)
    total = profile.sum()
    _maps, names, _res = _require_hotspots(cache)
    name_a = names[names.index(species_a)] if isinstance(species_a, str) else names[int(species_a)]
    name_b = names[names.index(species_b)] if isinstance(species_b, str) else names[int(species_b)]
    return pd.DataFrame(
        {
            "resid": np.arange(1, profile.size + 1),
            "species": name_a,
            "partner": name_b,
            "contacts": profile,
            "fraction": (profile / total if (normalize and total > 0) else profile),
        }
    )


def top_residue_pairs(
    cache: ContactCache,
    species_a: str | int,
    species_b: str | int,
    *,
    top_n: int = 20,
    blocks: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """The residue pairs carrying most of the interface, ranked."""
    matrix = residue_hotspot_matrix(cache, species_a, species_b, blocks=blocks)
    total = matrix.sum()
    flat = np.argsort(matrix, axis=None)[::-1][: int(top_n)]
    ra, rb = np.unravel_index(flat, matrix.shape)
    _maps, names, _res = _require_hotspots(cache)
    name_a = names[names.index(species_a)] if isinstance(species_a, str) else names[int(species_a)]
    name_b = names[names.index(species_b)] if isinstance(species_b, str) else names[int(species_b)]
    return pd.DataFrame(
        {
            "species_a": name_a,
            "resid_a": ra + 1,
            "species_b": name_b,
            "resid_b": rb + 1,
            "contacts": matrix[ra, rb],
            "fraction": (matrix[ra, rb] / total if total > 0 else np.nan),
        }
    )


def region_contact_table(
    cache: ContactCache,
    species_a: str | int,
    species_b: str | int,
    regions: dict,
    *,
    blocks: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Contacts to ``species_b`` grouped into named regions of ``species_a``.

    ``regions`` maps a name to an inclusive 1-based residue range, e.g.
    ``{"N-term": (1, 60), "NAC": (61, 95), "C-term": (96, 140)}``.  The table
    reports each region's share of the interface next to its share of the
    sequence, so enrichment is separated from "that region is simply bigger".
    """
    profile = residue_hotspot_profile(
        cache, species_a, species_b, blocks=blocks, normalize=False
    )
    contacts = profile["contacts"].to_numpy(dtype=float)
    n_res = contacts.size
    total = contacts.sum()

    rows = []
    for name, (lo, hi) in regions.items():
        lo_i, hi_i = int(lo) - 1, int(hi)
        if lo_i < 0 or hi_i > n_res or lo_i >= hi_i:
            raise ValueError(
                f"region {name!r} = ({lo}, {hi}) is outside residues 1..{n_res}"
            )
        n_in = hi_i - lo_i
        c_in = float(contacts[lo_i:hi_i].sum())
        share = c_in / total if total > 0 else float("nan")
        length_share = n_in / n_res
        rows.append(
            {
                "region": name,
                "resid_lo": int(lo),
                "resid_hi": int(hi),
                "n_residues": n_in,
                "contacts": c_in,
                "contact_share": share,
                "length_share": length_share,
                # >1 means the region carries more of the interface than its
                # length alone would give.
                "enrichment": (share / length_share if length_share > 0 else float("nan")),
            }
        )
    return pd.DataFrame(rows).sort_values("enrichment", ascending=False).reset_index(drop=True)


def hotspot_block_dataframe(
    cache: ContactCache,
    species_a: str | int,
    species_b: str | int,
    regions: Optional[dict] = None,
) -> pd.DataFrame:
    """Per-time-block region shares, to see an interface form (or shift).

    Without regions the table gives each block's total cross-species contacts;
    with them, each region's share within each block.  A region whose share
    climbs across blocks is being recruited into the interface as the assembly
    matures -- which a whole-trajectory average hides completely.
    """
    maps, _names, _res = _require_hotspots(cache)
    block_ranges = cache.meta.get("hotspot_blocks") or [
        [i, i] for i in range(maps.shape[0])
    ]
    rows = []
    for b, (lo, hi) in enumerate(block_ranges):
        if regions:
            table = region_contact_table(cache, species_a, species_b, regions, blocks=[b])
            for _, r in table.iterrows():
                rows.append(
                    {
                        "block": b,
                        "frame_lo": int(lo),
                        "frame_hi": int(hi),
                        "region": r["region"],
                        "contacts": r["contacts"],
                        "contact_share": r["contact_share"],
                        "enrichment": r["enrichment"],
                    }
                )
        else:
            matrix = residue_hotspot_matrix(cache, species_a, species_b, blocks=[b])
            rows.append(
                {
                    "block": b,
                    "frame_lo": int(lo),
                    "frame_hi": int(hi),
                    "contacts": float(matrix.sum()),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "contact_frequency_by_fate",
    "contact_enrichment_on_vs_off",
    "rank_contact_hotspots",
    "residue_hotspot_matrix",
    "residue_hotspot_profile",
    "top_residue_pairs",
    "region_contact_table",
    "hotspot_block_dataframe",
]
