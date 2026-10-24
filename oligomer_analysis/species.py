"""Species-aware analysis for multi-component systems.

Stage A treats every monomer as an interchangeable segment, which is correct for
a one-component box but hides the interesting physics in a mixture: with 50
alpha-synuclein chains and 10 Abeta40 chains, a single ``n_bound`` number is
dominated by the majority species and says nothing about whether Abeta is
binding alpha-synuclein or itself.

This module adds a species layer *on top of an existing cache*.  Nothing here
needs Stage A to be re-run: the per-frame segment contact maps (``cache.cmaps``)
already carry every cross-species pair, and ``monomer_atom_counts`` recorded in
the cache metadata is usually enough to recover which segment is which species.

Typical use::

    from oligomer_analysis import load_cache
    from oligomer_analysis.species import SpeciesMap, composition_timeseries

    cache = load_cache("cache_4.4x10_pp_rep1.npz")
    smap = SpeciesMap.from_cache(cache, spec="aSyn:295x50,Ab40:92x10")
    df = composition_timeseries(cache, smap)
    df.to_csv("species_stats.csv", index=False)

The ``spec`` grammar is ``name:beads x count`` groups separated by commas, in
the same order the monomers appear in the topology::

    "aSyn:295x50,Ab40:92x10"

When ``spec`` is omitted the map is inferred by grouping consecutive monomers of
equal size, which is right whenever each species is a contiguous block (the
normal GROMACS layout).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .graph_tools import clusters_from_cmap

__all__ = [
    "SpeciesMap",
    "parse_species_spec",
    "species_contact_matrix",
    "species_bound_free",
    "cluster_compositions",
    "composition_timeseries",
    "species_contact_dataframe",
    "partition_coefficients",
    "partition_summary",
    "free_monomer_concentration",
    "validate_segmentation",
    "SegmentationReport",
]


def parse_species_spec(spec: str) -> List[Tuple[str, int, int]]:
    """Parse ``"aSyn:295x50,Ab40:92x10"`` into ``[(name, beads, count), ...]``.

    One grammar, used by :meth:`SpeciesMap.from_spec`, by Stage A (which derives
    ``monomer_atom_counts`` from it), and by the CLI, so a composition is only
    ever written once and in one form.
    """
    out: List[Tuple[str, int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad species spec {part!r}; expected 'name:BEADSxCOUNT'")
        name, rhs = part.split(":", 1)
        rhs = rhs.lower().replace(" ", "")
        if "x" not in rhs:
            raise ValueError(f"bad species spec {part!r}; expected 'name:BEADSxCOUNT'")
        beads_s, count_s = rhs.split("x", 1)
        try:
            beads, count = int(beads_s), int(count_s)
        except ValueError as exc:
            raise ValueError(
                f"bad species spec {part!r}; BEADS and COUNT must be integers"
            ) from exc
        if beads <= 0 or count <= 0:
            raise ValueError(f"species {name.strip()!r} needs positive beads and count.")
        out.append((name.strip(), beads, count))

    if not out:
        raise ValueError("empty species spec.")
    names = [n for n, _b, _c in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate species names in spec: {names}")
    return out


def monomer_atom_counts_from_spec(spec: str) -> List[int]:
    """Per-monomer bead counts implied by a species spec, in topology order."""
    counts: List[int] = []
    for _name, beads, count in parse_species_spec(spec):
        counts.extend([beads] * count)
    return counts


# --------------------------------------------------------------------------
# species map
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SpeciesMap:
    """Which species each segment belongs to.

    Attributes
    ----------
    names:
        Species names, in topology order (e.g. ``["aSyn", "Ab40"]``).
    seg_species:
        ``(nseg,)`` array of indices into :attr:`names`.
    """

    names: List[str]
    seg_species: np.ndarray

    def __post_init__(self) -> None:
        self.seg_species = np.asarray(self.seg_species, dtype=np.int16)
        if self.seg_species.ndim != 1:
            raise ValueError("seg_species must be one-dimensional.")
        if len(self.names) == 0:
            raise ValueError("at least one species name is required.")
        hi = int(self.seg_species.max(initial=-1))
        if hi >= len(self.names):
            raise ValueError(
                f"seg_species references species {hi} but only "
                f"{len(self.names)} names were given."
            )

    # -- constructors ------------------------------------------------------

    @property
    def n_species(self) -> int:
        return len(self.names)

    @property
    def nseg(self) -> int:
        return int(self.seg_species.size)

    def counts(self) -> np.ndarray:
        """Number of chains of each species."""
        return np.bincount(self.seg_species, minlength=self.n_species).astype(np.int64)

    def mask(self, species: str | int) -> np.ndarray:
        """Boolean mask selecting the segments of one species."""
        k = self.index(species)
        return self.seg_species == k

    def index(self, species: str | int) -> int:
        if isinstance(species, str):
            if species not in self.names:
                raise KeyError(f"unknown species {species!r}; have {self.names}")
            return self.names.index(species)
        k = int(species)
        if not 0 <= k < self.n_species:
            raise IndexError(f"species index {k} out of range")
        return k

    @classmethod
    def from_spec(
        cls,
        spec: str,
        monomer_atom_counts: Optional[Sequence[int]] = None,
    ) -> "SpeciesMap":
        """Build a map from a ``"aSyn:295x50,Ab40:92x10"`` style string.

        When ``monomer_atom_counts`` is given the spec is checked against it, so
        a typo in the bead counts raises instead of silently mislabelling
        chains.
        """
        parsed = parse_species_spec(spec)
        names: List[str] = [name for name, _b, _c in parsed]
        blocks: List[Tuple[int, int]] = [(beads, count) for _n, beads, count in parsed]

        seg_species = np.concatenate(
            [np.full(count, k, dtype=np.int16) for k, (_, count) in enumerate(blocks)]
        )

        if monomer_atom_counts is not None:
            expected = np.concatenate(
                [np.full(count, beads, dtype=np.int64) for beads, count in blocks]
            )
            actual = np.asarray(list(monomer_atom_counts), dtype=np.int64)
            if expected.shape != actual.shape or not np.array_equal(expected, actual):
                raise ValueError(
                    "species spec does not match the cache's monomer_atom_counts: "
                    f"spec describes {len(expected)} monomers "
                    f"{_summarise_counts(expected)}, cache has {len(actual)} "
                    f"{_summarise_counts(actual)}."
                )
        return cls(names=names, seg_species=seg_species)

    @classmethod
    def auto_from_counts(
        cls,
        monomer_atom_counts: Sequence[int],
        *,
        size_names: Optional[Dict[int, str]] = None,
    ) -> "SpeciesMap":
        """Infer species by grouping *consecutive* monomers of equal size.

        Contiguity matters: two separated blocks of the same size become two
        species, because in a topology they are usually distinct molecules that
        merely happen to have the same bead count.
        """
        counts = [int(c) for c in monomer_atom_counts]
        if not counts:
            raise ValueError("monomer_atom_counts is empty.")
        runs: List[List[int]] = []
        for c in counts:
            if runs and runs[-1][0] == c:
                runs[-1][1] += 1
            else:
                runs.append([c, 1])

        size_names = dict(size_names or {})
        names: List[str] = []
        used: Dict[str, int] = {}
        for size, _n in runs:
            base = size_names.get(size, f"sp{size}")
            if base in used:
                used[base] += 1
                base = f"{base}_{used[base]}"
            else:
                used[base] = 1
            names.append(base)

        seg_species = np.concatenate(
            [np.full(n, k, dtype=np.int16) for k, (_size, n) in enumerate(runs)]
        )
        return cls(names=names, seg_species=seg_species)

    @classmethod
    def from_cache(
        cls,
        cache,
        spec: Optional[str] = None,
        *,
        size_names: Optional[Dict[int, str]] = None,
    ) -> "SpeciesMap":
        """Recover a species map from a Stage A cache.

        Uses ``meta['monomer_atom_counts']`` when present.  A uniform cache
        (``segment_mode='atom_blocks'``) has a single species, which is the
        right answer for a one-component box.
        """
        meta = getattr(cache, "meta", {}) or {}
        counts = meta.get("monomer_atom_counts")
        nseg = int(np.asarray(cache.cmaps).shape[1])

        # Stage A records the species when it was given one, which is both more
        # informative than bead counts (it carries the real names) and immune to
        # two species that happen to have the same size. An explicit ``spec``
        # still wins, so a caller can always override what the cache believes.
        stored = meta.get("species")
        if spec is None and stored:
            names = [str(d["name"]) for d in stored]
            seg_species = np.concatenate(
                [np.full(int(d["count"]), k, dtype=np.int16) for k, d in enumerate(stored)]
            )
            if seg_species.size != nseg:
                raise ValueError(
                    f"cache metadata describes {seg_species.size} chains but "
                    f"cmaps have {nseg} segments."
                )
            return cls(names=names, seg_species=seg_species)

        if counts is None:
            if spec is not None:
                smap = cls.from_spec(spec)
                if smap.nseg != nseg:
                    raise ValueError(
                        f"spec describes {smap.nseg} monomers but the cache has "
                        f"{nseg} segments."
                    )
                return smap
            per = meta.get("atoms_per_monomer")
            name = f"sp{int(per)}" if per else "all"
            return cls(names=[name], seg_species=np.zeros(nseg, dtype=np.int16))

        counts = [int(c) for c in counts]
        if len(counts) != nseg:
            raise ValueError(
                f"cache metadata lists {len(counts)} monomers but cmaps have "
                f"{nseg} segments."
            )
        if spec is not None:
            return cls.from_spec(spec, monomer_atom_counts=counts)
        return cls.auto_from_counts(counts, size_names=size_names)

    def describe(self) -> str:
        counts = self.counts()
        return ", ".join(f"{n}x{c}" for n, c in zip(self.names, counts))


def _summarise_counts(counts: np.ndarray) -> str:
    vals, cnts = np.unique(counts, return_counts=True)
    return "{" + ", ".join(f"{int(c)}x{int(v)}" for v, c in zip(vals, cnts)) + "}"


# --------------------------------------------------------------------------
# species-resolved observables
# --------------------------------------------------------------------------


def species_contact_matrix(
    cache,
    smap: SpeciesMap,
    *,
    unique: bool = True,
) -> np.ndarray:
    """Per-frame contact counts decomposed by species pair.

    Returns a ``(n_frames, K, K)`` array.  The result is symmetric; with
    ``unique=True`` (the default) entry ``[f, i, j]`` is the number of distinct
    contacting monomer pairs between species ``i`` and ``j`` in frame ``f``,
    including the diagonal, which is halved so that a same-species pair is not
    counted from both ends.

    This is the observable that separates "Abeta sticks to alpha-synuclein" from
    "Abeta sticks to itself"; both look identical in the scalar ``ninter``.
    """
    cmaps = np.asarray(cache.cmaps)
    if cmaps.ndim != 3:
        raise ValueError("cache.cmaps must be (n_frames, nseg, nseg).")
    nseg = cmaps.shape[1]
    if smap.nseg != nseg:
        raise ValueError(
            f"species map covers {smap.nseg} segments, cache has {nseg}."
        )

    onehot = np.zeros((nseg, smap.n_species), dtype=np.float64)
    onehot[np.arange(nseg), smap.seg_species] = 1.0

    work = cmaps.astype(np.float64, copy=True)
    diag = np.arange(nseg)
    work[:, diag, diag] = 0.0  # intra-monomer contacts are not pair contacts

    matrix = np.einsum("ai,fab,bj->fij", onehot, work, onehot, optimize=True)
    if unique:
        d = np.arange(smap.n_species)
        matrix[:, d, d] *= 0.5
    return matrix


def species_bound_free(
    cache,
    smap: SpeciesMap,
    *,
    inter_thresh: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bound and free chain counts, resolved per species.

    Returns ``(n_bound, n_free, n_total)`` where the first two are
    ``(n_frames, K)`` and the last is ``(K,)``.  A chain counts as bound when it
    has at least ``inter_thresh`` contacts with any *other* chain.
    """
    cmaps = np.asarray(cache.cmaps)
    nseg = cmaps.shape[1]
    adj = cmaps >= int(inter_thresh)
    diag = np.arange(nseg)
    adj[:, diag, diag] = False
    bound = adj.any(axis=2)  # (F, nseg)

    n_frames = cmaps.shape[0]
    n_bound = np.zeros((n_frames, smap.n_species), dtype=np.int32)
    for k in range(smap.n_species):
        n_bound[:, k] = bound[:, smap.seg_species == k].sum(axis=1)
    n_total = smap.counts().astype(np.int32)
    n_free = n_total[None, :] - n_bound
    return n_bound, n_free, n_total


def cluster_compositions(
    cache,
    smap: SpeciesMap,
    *,
    inter_thresh: int = 1,
    min_size: int = 2,
) -> List[List[Tuple[FrozenSet[int], np.ndarray]]]:
    """Per frame, the species composition of every cluster.

    Returns one list per frame of ``(members, composition)`` pairs, where
    ``composition[k]`` counts the chains of species ``k`` in that cluster.
    """
    cmaps = np.asarray(cache.cmaps)
    out: List[List[Tuple[FrozenSet[int], np.ndarray]]] = []
    for f in range(cmaps.shape[0]):
        rows: List[Tuple[FrozenSet[int], np.ndarray]] = []
        for clu in clusters_from_cmap(cmaps[f], inter_thresh):
            if len(clu) < int(min_size):
                continue
            members = np.fromiter(clu, dtype=np.int64, count=len(clu))
            comp = np.bincount(
                smap.seg_species[members], minlength=smap.n_species
            ).astype(np.int32)
            rows.append((clu, comp))
        out.append(rows)
    return out


def composition_timeseries(
    cache,
    smap: SpeciesMap,
    *,
    inter_thresh: int = 1,
    min_size: int = 2,
    size_species: Optional[Sequence[str | int]] = None,
) -> pd.DataFrame:
    """The headline per-frame table for a mixed system.

    One row per frame with, for every species: bound/free counts, the
    composition of the largest cluster, and how many clusters are pure versus
    mixed.  ``size_species`` restricts what "largest" means -- passing
    ``["aSyn"]`` ranks clusters by their alpha-synuclein content only, which is
    what you want when a handful of Abeta chains should not promote a small
    cluster over a genuinely larger one.
    """
    n_bound, n_free, _n_total = species_bound_free(
        cache, smap, inter_thresh=inter_thresh
    )
    contacts = species_contact_matrix(cache, smap, unique=True)
    per_frame = cluster_compositions(
        cache, smap, inter_thresh=inter_thresh, min_size=min_size
    )

    if size_species is None:
        weight = np.ones(smap.n_species, dtype=np.int64)
    else:
        weight = np.zeros(smap.n_species, dtype=np.int64)
        for s in size_species:
            weight[smap.index(s)] = 1

    frames = np.asarray(cache.frames)
    rows = []
    for i, clusters in enumerate(per_frame):
        n_pure = np.zeros(smap.n_species, dtype=np.int32)
        n_hetero = 0
        best_comp = np.zeros(smap.n_species, dtype=np.int32)
        best_score = -1
        in_hetero = np.zeros(smap.n_species, dtype=np.int32)

        for _members, comp in clusters:
            present = comp > 0
            if present.sum() == 1:
                n_pure[int(np.argmax(present))] += 1
            else:
                n_hetero += 1
                in_hetero += comp
            score = int((comp * weight).sum())
            if score > best_score:
                best_score = score
                best_comp = comp

        row: Dict[str, object] = {
            "frame": int(frames[i]) if i < len(frames) else i,
            "n_clusters": len(clusters),
            "n_hetero_clusters": n_hetero,
            "largest_size": int(best_comp.sum()),
            "largest_score": int(max(best_score, 0)),
        }
        for k, name in enumerate(smap.names):
            row[f"n_bound_{name}"] = int(n_bound[i, k])
            row[f"n_free_{name}"] = int(n_free[i, k])
            row[f"largest_{name}"] = int(best_comp[k])
            row[f"n_pure_{name}_clusters"] = int(n_pure[k])
            row[f"n_{name}_in_hetero"] = int(in_hetero[k])
        for a in range(smap.n_species):
            for b in range(a, smap.n_species):
                row[f"contacts_{smap.names[a]}_{smap.names[b]}"] = float(
                    contacts[i, a, b]
                )
        rows.append(row)

    return pd.DataFrame(rows)


def species_contact_dataframe(
    cache,
    smap: SpeciesMap,
    *,
    unique: bool = True,
) -> pd.DataFrame:
    """Long-form species-pair contact table (frame, species_a, species_b, contacts)."""
    matrix = species_contact_matrix(cache, smap, unique=unique)
    frames = np.asarray(cache.frames)
    rows = []
    for a in range(smap.n_species):
        for b in range(a, smap.n_species):
            for i in range(matrix.shape[0]):
                rows.append(
                    {
                        "frame": int(frames[i]) if i < len(frames) else i,
                        "species_a": smap.names[a],
                        "species_b": smap.names[b],
                        "contacts": float(matrix[i, a, b]),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# partitioning between the dense assembly and the surrounding solution
# --------------------------------------------------------------------------


def partition_coefficients(
    cache,
    smap: SpeciesMap,
    *,
    inter_thresh: int = 1,
    min_size: int = 2,
    size_species: Optional[Sequence[str | int]] = None,
) -> pd.DataFrame:
    """Per-frame enrichment of each species inside the largest assembly.

    For species ``k`` the partition coefficient is

    .. math::

        K_k = \frac{n_k^{\text{cluster}} / N^{\text{cluster}}}
                    {n_k^{\text{total}} / N^{\text{total}}}

    -- the mole fraction of ``k`` inside the largest cluster over its mole
    fraction in the whole box.  ``K > 1`` means the species is recruited into the
    assembly beyond what random mixing would give; ``K < 1`` means it is
    excluded.  This is the quantity microscopy reports as a partition or
    recruitment ratio, so it is what a simulation should quote when comparing
    with an experiment rather than a raw bound count.

    ``size_species`` picks which species define "largest" (see
    :func:`composition_timeseries`); ranking a mixed box by total chain count
    lets a handful of the minority species promote a small cluster.

    Returns one row per frame with ``K_<species>`` and ``x_<species>`` (the mole
    fraction inside the cluster) columns, plus the cluster size.
    """
    per_frame = cluster_compositions(
        cache, smap, inter_thresh=inter_thresh, min_size=min_size
    )
    totals = smap.counts().astype(np.float64)
    bulk_fraction = totals / totals.sum()

    if size_species is None:
        weight = np.ones(smap.n_species, dtype=np.int64)
    else:
        weight = np.zeros(smap.n_species, dtype=np.int64)
        for sp in size_species:
            weight[smap.index(sp)] = 1

    frames = np.asarray(cache.frames)
    rows = []
    for i, clusters in enumerate(per_frame):
        best_comp = np.zeros(smap.n_species, dtype=np.float64)
        best_score = -1
        for _members, comp in clusters:
            score = int((comp * weight).sum())
            if score > best_score:
                best_score = score
                best_comp = comp.astype(np.float64)

        size = best_comp.sum()
        row: Dict[str, object] = {
            "frame": int(frames[i]) if i < len(frames) else i,
            "cluster_size": int(size),
        }
        for k, name in enumerate(smap.names):
            if size > 0:
                x_in = best_comp[k] / size
                # A species absent from the box has no defined partitioning.
                kappa = x_in / bulk_fraction[k] if bulk_fraction[k] > 0 else np.nan
            else:
                x_in = np.nan
                kappa = np.nan
            row[f"n_{name}"] = int(best_comp[k])
            row[f"x_{name}"] = float(x_in)
            row[f"K_{name}"] = float(kappa)
        rows.append(row)
    return pd.DataFrame(rows)


def partition_summary(
    cache,
    smap: SpeciesMap,
    *,
    inter_thresh: int = 1,
    min_size: int = 2,
    size_species: Optional[Sequence[str | int]] = None,
    last_fraction: float = 0.25,
) -> pd.DataFrame:
    """Mean partition coefficient per species with a spread, over the run tail.

    One row per species: ``K_mean``, ``K_sd``, ``K_sem`` and the number of frames
    that contributed.  The spread matters -- a mean partition coefficient whose
    standard deviation exceeds it is not a measurement, and reporting the two
    together is what stops that being read as a result.

    ``K_sem`` is corrected for autocorrelation: consecutive frames are not
    independent samples, so the effective sample size is
    ``n / (2 * tau)`` rather than ``n``.  ``K_tau`` is the integrated
    autocorrelation time of the K series, ``K_n_eff`` the resulting
    independent-sample count, and ``K_sem_naive`` the uncorrected
    ``sd / sqrt(n)`` -- kept so the size of the correction is visible, because a
    large ratio between the two is itself the finding that the tail is too
    short.  The correction assumes the tail is stationary; it cannot rescue an
    error bar on a still-drifting observable, which is what ``coana converged``
    is for.

    K is undefined in a frame holding no cluster, and those frames are dropped
    before the autocorrelation is measured.  ``K_n_dropped`` counts them.  When
    it is non-zero the retained samples are not adjacent in time, so ``K_tau``
    is in units of retained samples rather than trajectory frames and should
    not be quoted as a correlation time.  ``K_n_eff`` and ``K_sem`` are robust
    to this, because thinning scales ``tau`` and the sample count together and
    their ratio is what the error bar depends on.

    The ``degenerate`` column marks the case where K carries no information at
    all: in a one-component box the mole fraction inside the assembly and in the
    box are the same quantity, so K is identically 1 however the chains behave.
    The value is still correct, but it is an identity rather than a measurement
    and must not be quoted as "no enrichment was observed".

    Two further columns say whether the *selection* of the assembly produced the
    answer.  Clusters are ranked by total chain count, so a cluster can win
    because the minority species joined it -- and K then measures the minority
    species inside a cluster chosen for containing it.  ``K_host_ranked`` repeats
    the measurement with clusters ranked by the majority species alone, and
    ``rank_agreement`` is the fraction of frames in which both rules pick the
    same assembly.  When agreement is high the two K values coincide and the
    number is a property of the system; when it is not, the number is a property
    of the definition.
    """
    df = partition_coefficients(
        cache, smap, inter_thresh=inter_thresh, min_size=min_size,
        size_species=size_species,
    )
    if not 0.0 < last_fraction <= 1.0:
        raise ValueError("last_fraction must be in (0, 1]")
    n = len(df)
    lo = max(0, n - max(1, int(round(n * last_fraction))))
    tail = df.iloc[lo:]

    # K compares a species' mole fraction inside the assembly with its mole
    # fraction in the box; with one species those are the same number.
    degenerate = smap.n_species < 2

    # Ranking clusters by total size lets the minority species promote the very
    # cluster it is then measured in.  Repeat the measurement with the ranking
    # done on the majority species alone: agreement means the number describes
    # the system rather than the choice of "largest".
    host_tail = None
    agreement = float("nan")
    if not degenerate and size_species is None:
        host = smap.names[int(np.argmax(smap.counts()))]
        host_df = partition_coefficients(
            cache, smap, inter_thresh=inter_thresh, min_size=min_size,
            size_species=[host],
        )
        host_tail = host_df.iloc[lo:]
        comp_cols = [f"n_{name}" for name in smap.names]
        same = (tail[comp_cols].to_numpy() == host_tail[comp_cols].to_numpy()).all(axis=1)
        agreement = float(same.mean()) if same.size else float("nan")

    # Consecutive trajectory frames are not independent samples, and this
    # package exists partly to say so: see convergence.py, where "a standard
    # error from correlated frames can be ten times too small" is the stated
    # reason that module was written.  Quoting sd/sqrt(n) here would make that
    # mistake in the one number most likely to be published, so the error bar
    # is corrected by the integrated autocorrelation time of the K series
    # itself.  The uncorrected value is kept as ``K_sem_naive`` so the size of
    # the correction stays visible rather than being silently absorbed.
    from .convergence import integrated_autocorrelation_time

    rows = []
    for name in smap.names:
        vals = tail[f"K_{name}"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n_ok = int(vals.size)
        mean = float(vals.mean()) if n_ok else float("nan")
        sd = float(vals.std(ddof=1)) if n_ok > 1 else float("nan")
        n_dropped = int(len(tail) - n_ok)
        if n_ok > 1:
            # tau is measured in units of RETAINED samples. When frames were
            # dropped (K is undefined in a frame with no cluster) the retained
            # samples are not adjacent in time, so tau is not a count of
            # trajectory frames. n_eff survives this: thinning scales tau and
            # the sample count together, leaving n/(2*tau) nearly invariant. The
            # reported tau does not, hence K_n_dropped alongside it.
            tau = float(integrated_autocorrelation_time(vals))
            n_eff = max(1.0, n_ok / (2.0 * max(tau, 0.5)))
            sem = sd / np.sqrt(n_eff)
            sem_naive = sd / np.sqrt(n_ok)
        else:
            tau = float("nan")
            n_eff = float(n_ok)
            sem = float("nan")
            sem_naive = float("nan")
        rows.append(
            {
                "species": name,
                "n_chains": int(smap.counts()[smap.index(name)]),
                "K_mean": mean,
                "K_sd": sd,
                # autocorrelation-corrected; this is the one to quote
                "K_sem": sem,
                "K_sem_naive": sem_naive,
                "K_tau": tau,
                "K_n_eff": n_eff,
                "K_n_dropped": n_dropped,
                "n_frames": n_ok,
                "frames_used": f"last {100.0 * last_fraction:.0f}%",
                "degenerate": degenerate,
                "K_host_ranked": _tail_mean(host_tail, f"K_{name}"),
                "rank_agreement": agreement,
            }
        )
    return pd.DataFrame(rows)


def _tail_mean(tail, column: str) -> float:
    """Mean of a K column over the frames where it is defined."""
    if tail is None:
        return float("nan")
    vals = tail[column].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def free_monomer_concentration(
    cache,
    smap: SpeciesMap,
    *,
    box_volume_nm3: Optional[float] = None,
    inter_thresh: int = 1,
    last_fraction: float = 0.25,
) -> pd.DataFrame:
    """Free-chain counts, and their concentration when a box volume is known.

    The free-monomer concentration of an equilibrated two-phase system is its
    saturation concentration, so this is the natural c_sat estimator -- but only
    where a dilute phase actually exists.  The returned table therefore always
    carries ``n_free_mean`` and its spread next to the concentration, and sets
    ``dilute_phase`` False when almost every chain is condensed, which is the
    regime where the number is meaningless however small its formal error bar.

    ``box_volume_nm3`` defaults to the mean box volume stored in the cache when
    the trajectory carried one.
    """
    _n_bound, n_free, n_total = species_bound_free(
        cache, smap, inter_thresh=inter_thresh
    )
    if not 0.0 < last_fraction <= 1.0:
        raise ValueError("last_fraction must be in (0, 1]")
    n_frames = n_free.shape[0]
    lo = max(0, n_frames - max(1, int(round(n_frames * last_fraction))))
    tail = n_free[lo:].astype(np.float64)

    volume = box_volume_nm3
    if volume is None:
        boxes = np.asarray(cache.boxes, dtype=np.float64)
        if boxes.size and np.all(boxes[:, :3] > 0):
            # Cache boxes are in angstrom; 1 nm^3 = 1000 A^3.
            volume = float(np.prod(boxes[:, :3], axis=1).mean() / 1000.0)

    # n / (V * N_A) with V in nm^3 gives molar; 1 nm^3 = 1e-24 L.
    to_micromolar = (1.0 / (6.02214076e23 * 1e-24)) * 1e6 if volume else None

    rows = []
    for k, name in enumerate(smap.names):
        vals = tail[:, k]
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
        total = int(n_total[k])
        row: Dict[str, object] = {
            "species": name,
            "n_chains": total,
            "n_free_mean": mean,
            "n_free_sd": sd,
            "free_fraction": mean / total if total else float("nan"),
            # Below ~5% free, the "dilute phase" is a handful of chains and the
            # concentration derived from it is noise, not a c_sat.
            "dilute_phase": bool(total and mean / total >= 0.05),
        }
        if volume:
            row["box_volume_nm3"] = volume
            row["c_free_uM"] = mean * to_micromolar / volume
            row["c_free_sd_uM"] = (
                sd * to_micromolar / volume if np.isfinite(sd) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# segmentation validation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SegmentationReport:
    """Outcome of :func:`validate_segmentation`."""

    ok: bool
    n_atoms_expected: int
    n_atoms_topology: Optional[int]
    n_monomers: int
    straddling: List[int]
    chain_lengths: Optional[List[Tuple[int, int]]]
    messages: List[str]

    def __str__(self) -> str:
        head = "OK" if self.ok else "FAILED"
        return f"segmentation {head}\n  " + "\n  ".join(self.messages)


def validate_segmentation(
    monomer_atom_counts: Sequence[int],
    gro_path: Optional[str] = None,
) -> SegmentationReport:
    """Check that a monomer block layout really matches the structure file.

    ``segment_mode='atom_counts'`` slices the atom array into consecutive
    blocks.  A wrong list does not raise -- it silently produces chimeric
    "monomers" spanning two real chains, and every downstream number is quietly
    wrong.  This checks the block boundaries against the residue numbering in a
    ``.gro``: a boundary must fall where the residue index restarts.
    """
    counts = [int(c) for c in monomer_atom_counts]
    expected_atoms = int(sum(counts))
    messages: List[str] = []
    straddling: List[int] = []
    chain_lengths: Optional[List[Tuple[int, int]]] = None
    n_atoms_topology: Optional[int] = None
    ok = True

    if any(c <= 0 for c in counts):
        ok = False
        messages.append("monomer_atom_counts contains a non-positive entry.")

    if gro_path is None:
        messages.append(
            f"{len(counts)} monomers, {expected_atoms} atoms "
            f"{_summarise_counts(np.asarray(counts))}; no .gro given, "
            "boundaries not checked."
        )
        return SegmentationReport(
            ok, expected_atoms, None, len(counts), straddling, None, messages
        )

    with open(gro_path, "r") as fh:
        fh.readline()
        n_atoms_topology = int(fh.readline().strip())
        resids = np.empty(n_atoms_topology, dtype=np.int64)
        for i in range(n_atoms_topology):
            resids[i] = int(fh.readline()[0:5])

    if n_atoms_topology != expected_atoms:
        ok = False
        messages.append(
            f"atom count mismatch: counts sum to {expected_atoms}, "
            f"{gro_path} has {n_atoms_topology}."
        )
        return SegmentationReport(
            ok, expected_atoms, n_atoms_topology, len(counts), straddling, None, messages
        )

    # real chain starts: first atom, and wherever the residue index restarts
    starts = [0]
    for i in range(1, n_atoms_topology):
        if resids[i] < resids[i - 1]:
            starts.append(i)
    lengths = [
        (starts[i + 1] - starts[i]) if i + 1 < len(starts) else n_atoms_topology - starts[i]
        for i in range(len(starts))
    ]
    vals, cnts = np.unique(np.asarray(lengths), return_counts=True)
    chain_lengths = [(int(v), int(c)) for v, c in zip(vals, cnts)]

    real_boundaries = set(starts) | {n_atoms_topology}
    block_boundaries = set(np.cumsum([0] + counts).tolist())
    bad = sorted(block_boundaries - real_boundaries)
    if bad:
        ok = False
        straddling = bad
        messages.append(
            f"{len(bad)} monomer boundaries fall inside a real chain "
            f"(first at atom {bad[0]}); the block list does not match the "
            "structure."
        )
    else:
        messages.append(
            f"{len(counts)} monomers, {expected_atoms} atoms; all boundaries "
            "align with chain starts."
        )
    messages.append(
        "structure chains: "
        + ", ".join(f"{c}x{v}" for v, c in chain_lengths)
    )
    return SegmentationReport(
        ok, expected_atoms, n_atoms_topology, len(counts), straddling, chain_lengths, messages
    )
