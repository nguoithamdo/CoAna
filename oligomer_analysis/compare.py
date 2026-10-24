"""Summarise many runs into one table, and group them without pooling mistakes.

A campaign produces dozens of caches that differ in concentration, force-field
variant, salt, and replicate.  Turning them into a comparison table is the step
where results are actually made -- and it is also where the two easiest errors
live:

1. **Pooling across a condition that matters.**  Averaging two force-field
   variants at the same concentration produces a mean that describes neither.
   The mistake is invisible in the output: the row looks exactly like a
   legitimate group mean.
2. **Sweeping in runs a filter was never meant to match.**  A regex over run
   names that accidentally admits a different series will happily invert a
   trend.

:func:`group_summary` guards against the first by reporting, for every group,
which descriptive columns were *not* constant within it.  A non-empty
``mixed_columns`` entry means the group is pooling something; it is then a
decision to make rather than an accident to discover later.  Against the second,
:func:`compare_caches` keeps the source path on every row, so a group can always
be expanded back into the exact runs behind it.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .cache_io import load_cache
from .convergence import assess_convergence
from .graph_tools import clusters_from_cmap
from .lineage import build_lineages_from_cache, filter_lineages
from .models import TargetStateConfig

__all__ = [
    "summarize_cache",
    "compare_caches",
    "group_summary",
]

#: Columns produced by :func:`summarize_cache` that are measurements rather than
#: labels.  Everything else is treated as a descriptor when grouping.
DEFAULT_METRICS = (
    "largest_cluster_final",
    "largest_cluster_late",
    "bound_fraction_late",
    "ninter_late",
    "n_lineages",
    "on_fraction",
    "converged",
)

#: Column names :func:`summarize_cache` may produce itself, which a descriptor
#: therefore must not be called. The unconditional keys are checked against the
#: row directly; these are the ones added by the optional convergence, species
#: and lineage blocks *after* the descriptors are merged, so a collision there
#: would be silently overwritten rather than overwriting. Every name in
#: DEFAULT_METRICS is reserved too: group_summary excludes metric names from its
#: pooling check, so a descriptor wearing one would never be reported as mixed.
_RESERVED_COLUMNS = frozenset(DEFAULT_METRICS) | {
    # convergence block
    "converged", "geweke_z", "relative_drift", "autocorr_time",
    "largest_cluster_sem",
    # species block (the K_<name>/bound_fraction_<name> columns are per species,
    # so those are checked against the row once they exist)
    "species", "rank_agreement",
    # lineage block
    "n_lineages", "n_on_pathway", "n_off_pathway", "on_fraction",
    "lineage_min_duration", "lineage_warnings", "hetero_lineage_fraction",
    # identity columns group_summary ignores by default
    "label", "path", "error",
}


def _late_mean(x: np.ndarray, last_fraction: float) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    lo = max(0, n - max(1, int(round(n * last_fraction))))
    return float(np.mean(x[lo:]))


def summarize_cache(
    cache_or_path,
    *,
    label: Optional[str] = None,
    inter_thresh: int = 1,
    cluster_min_size: int = 2,
    last_fraction: float = 0.25,
    target_config: Optional[TargetStateConfig] = None,
    lineage_min_duration: int = 1,
    species_spec: Optional[str] = None,
    include_lineage: bool = True,
    include_convergence: bool = True,
    descriptors: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Reduce one cache to a single row of numbers.

    The row carries the run's identity (``label``, ``path``, plus any extra
    ``descriptors``), its late-time observables, its convergence verdict, and --
    when lineages are built -- the on-pathway fraction and, for a mixture, the
    composition of its assemblies.

    ``last_fraction`` sets what "late" means; the default quotes the final
    quarter of the trajectory, which is the part worth comparing between runs of
    different length.
    """
    # Validated here rather than in _late_mean, which several call sites share:
    # an out-of-range value there produces a silently wrong "late" window --
    # last_fraction=0 averages the final frame alone, last_fraction=5 averages
    # the whole run -- and both look like legitimate numbers. species.py and
    # convergence.py validate the same argument, so this keeps the package
    # consistent.
    if not 0.0 < last_fraction <= 1.0:
        raise ValueError(
            f"last_fraction must be in (0, 1]; got {last_fraction}"
        )

    path = None
    if isinstance(cache_or_path, (str, os.PathLike)):
        path = str(cache_or_path)
        cache = load_cache(path)
    else:
        cache = cache_or_path

    if label is None:
        label = (
            os.path.splitext(os.path.basename(path))[0] if path else "cache"
        )

    cmaps = np.asarray(cache.cmaps)
    n_frames, nseg = cmaps.shape[0], cmaps.shape[1]
    largest = np.array(
        [
            max(
                (len(c) for c in clusters_from_cmap(cmaps[f], inter_thresh) if len(c) >= cluster_min_size),
                default=0,
            )
            for f in range(n_frames)
        ],
        dtype=float,
    )
    n_bound = np.asarray(cache.n_bound, dtype=float)

    row: Dict[str, object] = {
        "label": label,
        "path": path,
        "n_frames": int(n_frames),
        "n_chains": int(nseg),
        "frame_first": int(cache.frames[0]) if n_frames else -1,
        "frame_last": int(cache.frames[-1]) if n_frames else -1,
        "largest_cluster_final": float(largest[-1]) if n_frames else float("nan"),
        "largest_cluster_late": _late_mean(largest, last_fraction),
        "largest_cluster_max": float(largest.max()) if n_frames else float("nan"),
        "bound_fraction_late": _late_mean(n_bound, last_fraction) / nseg if nseg else float("nan"),
        "ninter_late": _late_mean(np.asarray(cache.ninter, dtype=float), last_fraction),
    }
    # A descriptor sharing a name with a computed column used to overwrite that
    # measurement here -- or, for the convergence columns added just below, be
    # overwritten by it. Either way one of the two silently disappeared, and a
    # descriptor named e.g. "largest_cluster_late" would replace the number it
    # was meant to label. group_summary also excludes metric names from its
    # pooling check, so such a descriptor would never be reported as mixed.
    # None of that is recoverable downstream, so it is refused here.
    collisions = sorted(
        k for k in (descriptors or {}) if k in row or k in _RESERVED_COLUMNS
    )
    if collisions:
        raise ValueError(
            f"descriptor name(s) {collisions} collide with computed columns of "
            "the run summary. A descriptor must not be named after a "
            "measurement: it would replace that column's value and would also "
            "be skipped by group_summary's pooling check. Rename the "
            "descriptor(s), e.g. with a 'cond_' prefix."
        )
    for key, value in (descriptors or {}).items():
        row[key] = value

    if include_convergence:
        report = assess_convergence(
            cache, "largest_cluster", inter_thresh=inter_thresh,
            min_size=cluster_min_size, series=largest,
        )
        row["converged"] = bool(report.converged)
        row["geweke_z"] = report.geweke_z
        row["relative_drift"] = report.relative_drift
        row["autocorr_time"] = report.autocorr_time
        row["largest_cluster_sem"] = report.sem_corrected

    smap = None
    if species_spec is not None or (cache.meta.get("species")):
        from .species import SpeciesMap, partition_summary, species_bound_free

        smap = SpeciesMap.from_cache(cache, spec=species_spec)
        row["species"] = smap.describe()
        n_bound_sp, n_free_sp, n_total_sp = species_bound_free(
            cache, smap, inter_thresh=inter_thresh
        )
        for k, name in enumerate(smap.names):
            row[f"bound_fraction_{name}"] = (
                _late_mean(n_bound_sp[:, k].astype(float), last_fraction) / n_total_sp[k]
                if n_total_sp[k]
                else float("nan")
            )
        if smap.n_species > 1:
            part = partition_summary(
                cache, smap, inter_thresh=inter_thresh,
                min_size=cluster_min_size, last_fraction=last_fraction,
            ).set_index("species")
            for name in smap.names:
                row[f"K_{name}"] = float(part.loc[name, "K_mean"])
                row[f"K_{name}_sd"] = float(part.loc[name, "K_sd"])
                # Carry the selection-rule check into the campaign table too.
                # Without it a comparison table shows only the total-ranked K,
                # which is the value that can be an artifact of calling the
                # biggest cluster "the assembly" -- and a table is exactly where
                # that number gets quoted from.
                row[f"K_{name}_host_ranked"] = float(part.loc[name, "K_host_ranked"])
            row["rank_agreement"] = float(part["rank_agreement"].iloc[0])

    if include_lineage:
        result = build_lineages_from_cache(
            cache,
            inter_thresh=inter_thresh,
            cluster_min_size=cluster_min_size,
            target_config=target_config or TargetStateConfig(),
            species_map=smap,
        )
        if lineage_min_duration > 1:
            result = filter_lineages(result, min_duration=lineage_min_duration)
        labels = [lin.class_label for lin in result.lineages]
        n_lin = len(labels)
        n_on = sum(1 for x in labels if x == "on-pathway")
        n_off = sum(1 for x in labels if x == "off-pathway")
        row["n_lineages"] = n_lin
        row["n_on_pathway"] = n_on
        row["n_off_pathway"] = n_off
        row["on_fraction"] = n_on / n_lin if n_lin else float("nan")
        row["lineage_min_duration"] = int(lineage_min_duration)
        row["lineage_warnings"] = "; ".join(result.warnings)
        if smap is not None and smap.n_species > 1 and n_lin:
            hetero = [lin for lin in result.lineages if (lin.composition > 0).sum() > 1]
            row["hetero_lineage_fraction"] = len(hetero) / n_lin
            on_hetero = sum(
                1 for lin in hetero if lin.class_label == "on-pathway"
            )
            # The join that a separate species table and lineage table cannot make:
            # are the productive lineages the mixed ones?
            row["on_fraction_hetero"] = (
                on_hetero / len(hetero) if hetero else float("nan")
            )
            pure = [lin for lin in result.lineages if (lin.composition > 0).sum() <= 1]
            row["on_fraction_pure"] = (
                sum(1 for lin in pure if lin.class_label == "on-pathway") / len(pure)
                if pure
                else float("nan")
            )
    return row


def compare_caches(
    caches: Iterable,
    *,
    labels: Optional[Sequence[str]] = None,
    descriptor_fn: Optional[Callable[[str], Dict[str, object]]] = None,
    on_error: str = "raise",
    **kwargs,
) -> pd.DataFrame:
    """Summarise many caches into one tidy table, one row per run.

    ``descriptor_fn`` receives each run's label and returns the condition
    columns it encodes -- concentration, force-field variant, replicate.  Those
    columns are what :func:`group_summary` then groups by and, just as
    importantly, what it checks for accidental pooling.

    ``on_error='skip'`` keeps going past an unreadable cache and records the
    failure in the returned table, which is what a campaign-wide sweep wants;
    the default raises.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError("on_error must be 'raise' or 'skip'")

    caches = list(caches)
    labels = list(labels) if labels is not None else [None] * len(caches)
    if len(labels) != len(caches):
        raise ValueError("labels must be the same length as caches")

    rows: List[Dict[str, object]] = []
    # Which columns came from descriptor_fn, recorded rather than guessed later.
    # group_summary used to tell descriptors from results by dtype, which is
    # right for the path regex (its groupdict values are always strings) but
    # wrong for any descriptor_fn returning a number: a numeric condition such
    # as salt_mM then looked like a measurement and was never reported as
    # pooled. Provenance is known here, so it is carried instead of re-derived.
    descriptor_cols: List[str] = []
    for cache, label in zip(caches, labels):
        name = label
        if name is None and isinstance(cache, (str, os.PathLike)):
            name = os.path.splitext(os.path.basename(str(cache)))[0]
        try:
            descriptors = descriptor_fn(name) if descriptor_fn and name else None
            for key in (descriptors or {}):
                if key not in descriptor_cols:
                    descriptor_cols.append(str(key))
            rows.append(
                summarize_cache(cache, label=label, descriptors=descriptors, **kwargs)
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            if on_error == "raise":
                raise
            rows.append({"label": name, "path": str(cache), "error": repr(exc)})
    out = pd.DataFrame(rows)
    # ``attrs`` can be dropped by some pandas operations, so group_summary also
    # accepts the list directly; this is the convenient default, not the only way.
    out.attrs["descriptor_columns"] = list(descriptor_cols)
    return out


def group_summary(
    df: pd.DataFrame,
    by: Sequence[str],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    ignore: Sequence[str] = ("label", "path", "lineage_warnings", "error"),
    descriptor_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Mean, spread and count per group, with a warning for hidden pooling.

    For every metric the table gives ``<metric>_mean``, ``<metric>_sd`` and
    ``<metric>_sem``; ``n_runs`` counts the rows behind each group.

    The column that makes this worth using is ``mixed_columns``: the descriptors
    that vary *within* a group but are not part of ``by``.  If a group averages
    two force-field variants, that column names the variant, and the row can be
    rejected before it becomes a result.  An empty string means the group really
    is one condition.

    Which columns count as descriptors is taken from ``descriptor_columns``, or
    from ``df.attrs['descriptor_columns']`` as set by :func:`compare_caches`.
    Only when neither is available does it fall back to "every non-numeric
    column", which silently misses a numeric condition such as ``salt_mM``: that
    descriptor looks like a measurement, so a group pooling two salt conditions
    reports ``mixed_columns=''`` and the row is indistinguishable from a
    legitimate group mean.  Pass ``descriptor_columns`` explicitly whenever the
    table did not come straight from :func:`compare_caches`.
    """
    by = list(by)
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise KeyError(f"grouping columns not in the table: {missing}")

    metrics = [m for m in metrics if m in df.columns]
    ignore_set = set(ignore) | set(by) | set(metrics)

    # Every non-numeric column is a candidate descriptor. Numeric ones are
    # assumed to be results -- including integer metrics such as
    # ``n_on_pathway``, which would otherwise be flagged as "mixed" in every
    # group and bury the warning this column exists for.
    heuristic = [
        c
        for c in df.columns
        if c not in ignore_set and not pd.api.types.is_numeric_dtype(df[c])
    ]

    declared = descriptor_columns
    if declared is None:
        declared = df.attrs.get("descriptor_columns")

    # Declared provenance ADDS the numeric descriptors the heuristic cannot
    # see; it does not replace it. Replacing it was a regression: an empty
    # declared list (which compare_caches sets whenever descriptor_fn is None)
    # switched the heuristic off entirely, so a built-in string descriptor such
    # as ``species`` stopped being checked and two differently-composed systems
    # could be pooled by n_chains with mixed_columns=''.
    if declared is None:
        descriptor_cols = heuristic
    else:
        extra = [c for c in declared if c in df.columns and c not in ignore_set]
        descriptor_cols = list(dict.fromkeys(heuristic + extra))

    rows: List[Dict[str, object]] = []
    for keys, group in df.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row: Dict[str, object] = dict(zip(by, keys))
        row["n_runs"] = int(len(group))
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = vals.size
            mean = float(vals.mean()) if n else float("nan")
            sd = float(vals.std(ddof=1)) if n > 1 else float("nan")
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_sem"] = sd / np.sqrt(n) if n > 1 else float("nan")
            row[f"{metric}_n"] = n

        mixed = [
            c
            for c in descriptor_cols
            if group[c].nunique(dropna=False) > 1
        ]
        row["mixed_columns"] = ", ".join(mixed)
        row["runs"] = ", ".join(str(x) for x in group.get("label", pd.Series(dtype=str)))
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(by).reset_index(drop=True)
