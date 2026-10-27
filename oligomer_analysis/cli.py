# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""``coana`` -- the command-line interface of the CoAna package.

Runs the full CoAnaPure Stage A workflow (and the HTML report) without writing
Python, which is what batch schedulers want::

    # one-time random-access index for a huge PDB trajectory
    coana index md1_centered_filtered_1956frames.pdb

    # Stage A on all CPU cores of the node, crash-safe for walltime limits
    coana precompute --traj md1.pdb --frames all --cutoff 6.5 --intra-exclude 2 \\
        --segment-mode atom_blocks --atoms-per-monomer 83 --n-monomers 128 \\
        --n-jobs -1 --parallel process --checkpoint-every 200 --resume \\
        -o cache_cut6.5.npz

    # inspect a cache, then build the self-contained HTML report
    coana info cache_cut6.5.npz
    coana report cache_cut6.5.npz -o report.html --title "128-mer aggregation"

    # the interactive split/merge timeline, laid out on a real frame axis
    coana timeline cache_cut6.5.npz -o TIMELINE.html --frame-lo 640 --frame-hi 820

    # cross-validate CoAnaPure against MDAnalysis (needs the mdanalysis extra)
    coana benchmark --traj md1.pdb --frames 0:250

Frame specifications use Python slice semantics: ``start:stop[:step]`` with an
*exclusive* stop (``0:1956`` = frames 0..1955), a comma list (``0,10,20``), a
single frame number, or ``all``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

__all__ = ["main"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _parse_frames_spec(spec: str) -> Optional[List[int]]:
    """``all`` -> None; else Python-slice / comma-list / single frame."""
    spec = spec.strip().lower()
    if spec in ("all", ""):
        return None
    if ":" in spec:
        parts = spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        frames = list(range(start, stop, step))
        if not frames:
            raise SystemExit(f"error: frame spec {spec!r} selects no frames")
        return frames
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    return [int(spec)]


def _count_frames(traj: str, topology: Optional[str], verbose: bool) -> int:
    """Number of frames in a trajectory, as cheaply as the format allows."""
    ext = os.path.splitext(traj)[1].lower()
    if ext in (".pdb", ".ent"):
        from .coanapure.pdb_io import PDBModelIndex, default_index_path

        sidecar = default_index_path(traj)
        if os.path.exists(sidecar):
            try:
                return PDBModelIndex.load(sidecar, traj).n_models
            except ValueError:
                pass
        if verbose:
            print(f"counting frames (building index sidecar {sidecar}) ...", flush=True)
        idx = PDBModelIndex.build(traj)
        idx.save(sidecar)
        return idx.n_models
    if ext == ".gro":
        with open(traj, "r", errors="replace") as fh:
            fh.readline()
            n_atoms = int(fh.readline().split()[0])
            n_lines = 2 + sum(1 for _ in fh)
        per_frame = n_atoms + 3
        if n_lines % per_frame:
            raise SystemExit(
                f"error: cannot count frames of {traj} (irregular .gro); "
                "pass --frames explicitly"
            )
        return n_lines // per_frame
    if ext == ".trr":
        from .coanapure.trr_io import TRRFrameStream

        stream = TRRFrameStream(traj, wanted=set())
        for _ in stream:
            pass
        return stream.n_frames_seen
    if ext == ".xtc":
        from .coanapure.xtc_io import count_xtc_frames

        return count_xtc_frames(traj)
    raise SystemExit(f"error: cannot count frames for format {ext!r}; pass --frames")


def _parse_box(spec: Optional[str]):
    if not spec:
        return None
    vals = [float(x) for x in spec.replace(",", " ").split()]
    if len(vals) == 3:
        vals += [90.0, 90.0, 90.0]
    if len(vals) != 6:
        raise SystemExit("error: --fallback-box needs 'a,b,c' or 'a,b,c,alpha,beta,gamma'")
    return tuple(vals)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def _cmd_precompute(args: argparse.Namespace) -> int:
    from .coanapure import precompute_cache_pure
    from .models import PrecomputeConfig

    verbose = not args.quiet
    frames = _parse_frames_spec(args.frames)
    if frames is None:
        n = _count_frames(args.traj, args.topology, verbose)
        frames = list(range(n))
        if verbose:
            print(f"trajectory has {n} frames", flush=True)

    counts = None
    if args.monomer_atom_counts:
        counts = [int(x) for x in args.monomer_atom_counts.replace(",", " ").split()]

    cfg = PrecomputeConfig(
        topology_file=args.topology,
        traj_file=args.traj,
        frames=frames,
        cutoff_contact=args.cutoff,
        intra_exclude=args.intra_exclude,
        n_jobs=args.n_jobs,
        fallback_box=_parse_box(args.fallback_box),
        segment_mode=args.segment_mode,
        atoms_per_monomer=args.atoms_per_monomer,
        residues_per_monomer=args.residues_per_monomer,
        n_monomers=args.n_monomers,
        monomer_atom_counts=counts,
        selection=args.selection,
        species_spec=args.species,
        contact_classes=args.contact_classes,
        hotspot_blocks=(args.hotspot_blocks or None),
        contact_mode=args.contact_mode,
    )
    if not args.allow_no_box and not _first_frame_has_box(cfg, args):
        from .coanapure.precompute_pure import NO_BOX_MSG
        print(f"error: {NO_BOX_MSG}\n"
              "Pass --allow-no-box only for a system that "
              "really has no periodic boundaries.", file=sys.stderr)
        return 2
    precompute_cache_pure(
        cfg,
        out_npz=args.out,
        verbose=verbose,
        broadcast_box=args.broadcast_box,
        index_file=args.index_file,
        parallel=args.parallel,
        checkpoint_every=(args.checkpoint_every or None),
        resume=args.resume,
    )
    return 0


def _first_frame_has_box(cfg, args) -> bool:
    """Run Stage A on the first requested frame only and look at its box."""
    import dataclasses
    import tempfile
    import warnings

    import numpy as np

    from .coanapure import precompute_cache_pure
    from .coanapure.precompute_pure import NoBoxWarning

    probe = dataclasses.replace(cfg, frames=list(cfg.frames)[:1], n_jobs=1)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "probe.npz")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoBoxWarning)
            warnings.simplefilter("ignore", UserWarning)
            precompute_cache_pure(probe, out_npz=out, verbose=False,
                                  broadcast_box=args.broadcast_box,
                                  index_file=args.index_file)
        with np.load(out, allow_pickle=False) as z:
            return bool(np.any(z["boxes"][:, :3] > 0))


def _cmd_index(args: argparse.Namespace) -> int:
    from .coanapure.pdb_io import build_model_index, default_index_path

    out = args.out or default_index_path(args.traj)
    idx = build_model_index(args.traj, out)
    print(f"indexed {idx.n_models} models x {idx.n_atoms} atoms -> {out}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    import numpy as np

    from .cache_io import load_cache

    cache = load_cache(args.cache)
    meta = cache.meta
    print(f"cache: {args.cache}")
    print(f"  frames : {len(cache.frames)} (frame {int(cache.frames.min())}..{int(cache.frames.max())})")
    print(f"  nseg   : {meta.get('nseg')} monomers, segment_mode={meta.get('segment_mode')}")
    print(f"  cmaps  : {cache.cmaps.shape} int32")
    print(f"  engine : {meta.get('engine')}  format={meta.get('traj_format')}  parallel={meta.get('parallel', '-')}")
    has_box = bool(np.any(cache.boxes[:, :3] > 0))
    print(f"  pbc    : {'yes' if has_box else 'no box stored'}")
    if meta.get("selection"):
        print(f"  select : {meta['selection']} ({meta.get('selection_n_atoms')} atoms)")
    if meta.get("species"):
        desc = ", ".join(
            f"{d['name']}:{d['beads']}x{d['count']}" for d in meta["species"]
        )
        print(f"  species: {desc}")
    extras = []
    if cache.class_cmaps is not None:
        extras.append(f"contact classes {cache.class_cmaps.shape[1]}x{cache.class_cmaps.shape[2]}")
    if cache.hotspot_maps is not None:
        extras.append(f"residue hotspots in {cache.hotspot_maps.shape[0]} block(s)")
    print(f"  extras : {', '.join(extras) if extras else 'none'}")
    print(f"  cutoff : {meta.get('cutoff_contact')} A, intra_exclude={meta.get('intra_exclude')}")
    print(f"  bound  : final {int(cache.n_bound[-1])}/{int(cache.n_bound[-1] + cache.n_free[-1])} monomers bound")
    if args.json:
        print(json.dumps(meta, indent=2, default=str))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .models import TargetStateConfig
    from .report import generate_html_report

    calibration = None
    if args.max_size:
        from .plotting_common import PlotCalibration

        calibration = PlotCalibration(
            max_oligomer_size=args.max_size,
            matrix_max_size=args.max_size,
            node_size_transform="sqrt",
        )
    generate_html_report(
        args.cache,
        args.out,
        title=args.title,
        subtitle=args.subtitle,
        author=args.author,
        inter_thresh=args.inter_thresh,
        include_lineage=not args.no_lineage,
        target_config=TargetStateConfig(
            min_size=args.min_size,
            min_duration=args.min_duration,
            min_contribution=args.min_contribution,
        ),
        calibration=calibration,
        max_table_rows=args.max_table_rows,
        verbose=not args.quiet,
    )
    return 0


def _cmd_timeline(args: argparse.Namespace) -> int:
    from .models import TargetStateConfig
    from .timeline import generate_lineage_timeline

    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(args.cache))[0]
        out = f"{base}_timeline.html"

    generate_lineage_timeline(
        args.cache,
        out,
        frame_lo=args.frame_lo,
        frame_hi=args.frame_hi,
        target_config=TargetStateConfig(
            min_size=args.min_size,
            min_duration=args.min_duration,
            min_contribution=args.min_contribution,
        ),
        inter_thresh=args.inter_thresh,
        cluster_min_size=args.cluster_min_size,
        overlap_min=args.overlap_min,
        parent_frac_min=args.parent_frac_min,
        child_frac_min=args.child_frac_min,
        union_cover_min=args.union_cover_min,
        payload_json=args.payload_json,
        check_js=args.check_js,
        verbose=not args.quiet,
    )
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .coanapure import benchmark

    old_argv = sys.argv
    try:
        sys.argv = ["coana benchmark"] + args.rest
        benchmark.main()
    finally:
        sys.argv = old_argv
    return 0


def _cmd_version(args: argparse.Namespace) -> int:
    from . import __version__ as pkg_version
    from .coanapure import __version__ as engine_version

    print(f"coana {pkg_version} (CoAnaPure engine {engine_version})")
    return 0


def _cmd_citations(args: argparse.Namespace) -> int:
    from .citations import citation_bibtex, citation_text

    print(citation_bibtex() if args.bibtex else citation_text(), end="" if args.bibtex else "\n")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #



def _cmd_species(args) -> int:
    """Species-resolved statistics for a mixed-component cache."""
    from .cache_io import load_cache
    from .species import SpeciesMap, composition_timeseries, species_contact_matrix

    cache = load_cache(args.cache)
    smap = SpeciesMap.from_cache(cache, spec=args.species)
    print(f"cache   : {args.cache}")
    print(f"species : {smap.describe()}  ({smap.n_species} species, {smap.nseg} segments)")

    size_species = args.size_species.split(",") if args.size_species else None
    df = composition_timeseries(
        cache, smap, inter_thresh=args.inter_thresh,
        min_size=args.min_size, size_species=size_species,
    )
    matrix = species_contact_matrix(cache, smap)
    last = matrix[-1]
    print("\nfinal-frame contacts (unique pairs):")
    for a in range(smap.n_species):
        for b in range(a, smap.n_species):
            print(f"  {smap.names[a]:>8s} - {smap.names[b]:<8s} {last[a, b]:10.0f}")
    print("\nfinal-frame bound fraction:")
    for k, name in enumerate(smap.names):
        nb = int(df.iloc[-1][f"n_bound_{name}"])
        nf = int(df.iloc[-1][f"n_free_{name}"])
        print(f"  {name:>8s} {nb:4d}/{nb + nf:<4d} ({100.0 * nb / max(nb + nf, 1):5.1f}%)")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}  ({len(df)} frames, {len(df.columns)} columns)")
    return 0


def _cmd_validate(args) -> int:
    """Check that a monomer block layout matches a structure file."""
    from .species import validate_segmentation

    counts: list[int] = []
    for part in args.monomer_atom_counts.replace(" ", ",").split(","):
        if part:
            counts.append(int(part))
    report = validate_segmentation(counts, args.topology)
    print(report)
    return 0 if report.ok else 1

def _species_map(cache, spec):
    from .species import SpeciesMap

    return SpeciesMap.from_cache(cache, spec=spec)


def _target_config(args):
    from .models import TargetStateConfig

    return TargetStateConfig(
        min_size=args.min_size,
        min_duration=args.min_duration,
        min_contribution=args.min_contribution,
    )


def _write_csv(df, path, what):
    if path:
        df.to_csv(path, index=False)
        print(f"\nwrote {path}  ({len(df)} rows)")
    return df


def _cmd_partition(args) -> int:
    """Enrichment of each species inside the largest assembly."""
    import numpy as np

    from .cache_io import load_cache
    from .species import free_monomer_concentration, partition_summary

    cache = load_cache(args.cache)
    smap = _species_map(cache, args.species)
    print(f"cache   : {args.cache}")
    print(f"species : {smap.describe()}")

    summary = partition_summary(
        cache, smap, inter_thresh=args.inter_thresh, min_size=args.min_size,
        last_fraction=args.last_fraction,
    )
    print("\npartition coefficient (>1 = enriched in the assembly):")
    print(summary.to_string(index=False))
    # K_sem is autocorrelation-corrected; K_sem_naive is what sd/sqrt(n) would
    # have claimed. A large ratio means the tail holds few independent samples,
    # which is a statement about the run rather than about the species.
    with np.errstate(invalid="ignore", divide="ignore"):
        inflation = (summary["K_sem"] / summary["K_sem_naive"]).max()
    if np.isfinite(inflation) and inflation > 2.0:
        n_eff = float(summary["K_n_eff"].min())
        tau = float(summary["K_tau"].max())
        print(
            f"\nnote: K_sem is corrected for autocorrelation and is {inflation:.1f}x "
            f"the naive sd/sqrt(n) (K_sem_naive). The tail's integrated "
            f"autocorrelation time is {tau:.1f} frames, so it holds only about "
            f"{n_eff:.0f} independent samples. Quote K_sem, not K_sem_naive, and "
            "treat a tail this correlated as a reason to run longer rather than "
            "a tighter measurement."
        )
    if bool(summary["degenerate"].any()):
        print(
            "\nnote: only one species is present, so K compares this species "
            "with itself and is identically 1 whatever the chains do. That is an "
            "identity, not a measurement -- do not read it as 'no enrichment'. "
            "A partition coefficient needs at least two species."
        )
    else:
        agreement = float(summary["rank_agreement"].iloc[0])
        gap = (summary["K_mean"] - summary["K_host_ranked"]).abs().max()
        if np.isfinite(agreement) and (agreement < 0.99 or gap > 0.05):
            print(
                f"\nnote: 'largest assembly' was decided by total chain count, and "
                f"in {100.0 * (1.0 - agreement):.0f}% of frames that is not the "
                "assembly with the most chains of the majority species -- a "
                "minority chain can make a cluster the largest and is then "
                "measured inside it. Ranking by the majority species instead "
                "gives the K_host_ranked column "
                f"(largest change {gap:.3f}). Quote the enrichment only if the "
                "two agree; otherwise it is a property of the definition."
            )

    free = free_monomer_concentration(
        cache, smap, inter_thresh=args.inter_thresh, last_fraction=args.last_fraction,
    )
    print("\nfree chains in the surrounding solution:")
    print(free.to_string(index=False))
    if not free["dilute_phase"].any():
        print(
            "\nnote: essentially every chain is condensed, so there is no dilute "
            "phase to measure. A saturation concentration from these numbers "
            "would be noise, not a c_sat."
        )
    _write_csv(summary, args.out, "partition summary")
    return 0


def _cmd_chemistry(args) -> int:
    """Interaction-type decomposition of inter-chain contacts."""
    from .cache_io import load_cache
    from .chemistry import (
        contact_class_dataframe,
        describe_contact_classes,
        contact_class_summary,
        hydrophobic_fraction_timeseries,
        salt_bridge_timeseries,
    )

    cache = load_cache(args.cache)
    summary = contact_class_summary(cache, last_fraction=args.last_fraction)
    print(f"cache : {args.cache}")
    print(f"classes: {', '.join(cache.class_names)}")
    if cache.meta.get("contact_class_census"):
        print(f"residues: {describe_contact_classes(cache.meta['contact_class_census'])}")
    print("\ninter-chain contacts by interaction type:")
    print(summary.head(args.top).to_string(index=False))

    salt = salt_bridge_timeseries(cache)
    hydro = hydrophobic_fraction_timeseries(cache)
    n = len(salt)
    lo = max(0, n - max(1, int(round(n * args.last_fraction))))
    import numpy as np

    print(
        f"\nsalt bridges (late mean)      : {float(np.mean(salt[lo:])):.1f}"
    )
    print(
        f"hydrophobic share (late mean) : "
        f"{100.0 * float(np.nanmean(hydro[lo:])):.1f}%"
    )
    _write_csv(
        contact_class_dataframe(cache) if args.long else summary, args.out, "chemistry"
    )
    return 0


def _cmd_hotspots(args) -> int:
    """Residue-resolved cross-species contact maps."""
    from .cache_io import load_cache
    from .hotspots import (
        hotspot_block_dataframe,
        region_contact_table,
        residue_hotspot_profile,
        top_residue_pairs,
    )

    cache = load_cache(args.cache)
    names = cache.species_names
    a = args.species_a or (names[0] if names else 0)
    b = args.species_b or (names[1] if len(names) > 1 else a)
    print(f"cache : {args.cache}")
    print(f"pair  : {a} - {b}")

    regions = None
    if args.regions:
        regions = {}
        for part in args.regions.split(","):
            name, _, span = part.partition(":")
            lo, _, hi = span.partition("-")
            regions[name.strip()] = (int(lo), int(hi))

    if regions:
        table = region_contact_table(cache, a, b, regions)
        print("\ncontacts by region of " + str(a) + " (enrichment > 1 = more than "
              "its share of the sequence):")
        print(table.to_string(index=False))
        blocks = hotspot_block_dataframe(cache, a, b, regions)
        print("\nper time block:")
        print(blocks.to_string(index=False))
        out_df = table
    else:
        pairs = top_residue_pairs(cache, a, b, top_n=args.top)
        print("\ntop residue pairs:")
        print(pairs.to_string(index=False))
        out_df = residue_hotspot_profile(cache, a, b)

    _write_csv(out_df, args.out, "hotspots")
    return 0


def _cmd_converged(args) -> int:
    """Has the run reached a steady state?"""
    from .cache_io import load_cache
    from .convergence import assess_convergence, convergence_dataframe

    cache = load_cache(args.cache)
    print(f"cache : {args.cache}  ({len(cache.frames)} frames)")
    if args.all:
        df = convergence_dataframe(
            cache, inter_thresh=args.inter_thresh, min_size=args.min_size
        )
        print(
            df[["observable", "mean", "sem_corrected", "autocorr_time",
                "geweke_z", "relative_drift", "converged"]].to_string(index=False)
        )
        _write_csv(df, args.out, "convergence")
        return 0 if bool(df["converged"].all()) else 1

    report = assess_convergence(
        cache, args.observable, inter_thresh=args.inter_thresh, min_size=args.min_size
    )
    print()
    print(report)
    if args.out:
        import pandas as pd

        _write_csv(pd.DataFrame([report.to_dict()]), args.out, "convergence")
    return 0 if report.converged else 1


def _cmd_lineage(args) -> int:
    """Lineage fate table, species-aware and persistence-filtered."""
    from .cache_io import load_cache
    from .lineage import (
        build_lineages_from_cache,
        directed_fate_summary,
        directed_fates,
        filter_lineages,
        lineage_summary_dataframe,
        sweep_verdict,
        target_size_sweep,
    )

    cache = load_cache(args.cache)
    smap = None
    if args.species or cache.meta.get("species"):
        smap = _species_map(cache, args.species)
        print(f"species : {smap.describe()}")

    result = build_lineages_from_cache(
        cache,
        inter_thresh=args.inter_thresh,
        cluster_min_size=args.cluster_min_size,
        target_config=_target_config(args),
        species_map=smap,
    )
    print(f"cache   : {args.cache}")
    print(f"built   : {len(result.lineages)} lineages, {len(result.segments)} segments, "
          f"{len(result.events)} events")
    for msg in result.warnings:
        print(f"  warning: {msg}")

    directed = directed_fates(result)
    if not directed.empty:
        print("\ndirected fates of nucleation events (forward descendants only):")
        print(directed_fate_summary(directed, min_span=args.lineage_min_duration).to_string(index=False))
    if args.directed_out:
        _write_csv(directed, args.directed_out, "directed fates")

    if args.lineage_min_duration > 1:
        result = filter_lineages(result, min_duration=args.lineage_min_duration)
        print(f"filtered: {len(result.lineages)} lineages with duration >= "
              f"{args.lineage_min_duration}")

    df = lineage_summary_dataframe(result)
    if df.empty:
        print("no lineages survived the filter.")
        return 0

    counts = df["class_label"].value_counts()
    print("\nfate:")
    for label in ("on-pathway", "uncertain", "off-pathway"):
        n = int(counts.get(label, 0))
        print(f"  {label:<12s} {n:5d}  ({100.0 * n / len(df):5.1f}%)")

    if "is_hetero" in df.columns:
        # The join a separate species table cannot make: are the productive
        # lineages the mixed ones?
        hetero = df[df["is_hetero"]]
        pure = df[~df["is_hetero"]]
        print("\ncomposition vs fate:")
        for name, sub in (("mixed", hetero), ("single-species", pure)):
            if len(sub):
                on = int((sub["class_label"] == "on-pathway").sum())
                print(f"  {name:<15s} {len(sub):5d} lineages, "
                      f"{100.0 * on / len(sub):5.1f}% on-pathway")

    if args.sweep:
        sizes = [int(x) for x in args.sweep.replace(",", " ").split()]
        sweep = target_size_sweep(
            cache, sizes, inter_thresh=args.inter_thresh,
            cluster_min_size=args.cluster_min_size,
            min_duration=args.lineage_min_duration,
            target_config=_target_config(args), species_map=smap,
        )
        print("\ntarget-size robustness:")
        print(sweep.to_string(index=False))
        print(f"\n{sweep_verdict(sweep)}")

    _write_csv(df, args.out, "lineage summary")
    return 0


def _cmd_mechanism(args) -> int:
    """Incorporation, catalysis, or neither."""
    from .cache_io import load_cache
    from .lineage import build_lineages_from_cache
    from .mechanism import (
        exchange_statistics,
        mechanism_summary,
        residence_dataframe,
        residence_events,
    )

    cache = load_cache(args.cache)
    smap = _species_map(cache, args.species)
    if smap.n_species < 2:
        raise SystemExit(
            "error: mechanism analysis needs at least two species; this cache "
            "has one. Pass --species or rebuild Stage A with --species."
        )
    result = build_lineages_from_cache(
        cache,
        inter_thresh=args.inter_thresh,
        cluster_min_size=args.cluster_min_size,
        target_config=_target_config(args),
        species_map=smap,
    )
    if args.lineage_min_duration > 1:
        from .lineage import filter_lineages
        result = filter_lineages(result, min_duration=args.lineage_min_duration)
    guest = args.guest or smap.names[int(smap.counts().argmin())]
    events = residence_events(
        result,
        guest_species=guest,
        host_species=args.host,
        commit_size=args.commit_size,
        after_window=args.after_window,
        frames=cache.frames,
        commit_duration=args.commit_duration,
        mode=args.mode,
    )
    n_guest = int(smap.counts()[smap.index(guest)])
    if args.host:
        host = args.host
    else:
        # Same rule as residence_events: the most abundant other species.
        counts = smap.counts()
        host = max(
            (name for name in smap.names if name != guest),
            key=lambda name: int(counts[smap.index(name)]),
        )
    n_host = int(smap.counts()[smap.index(host)])
    print(f"cache : {args.cache}")
    print(f"guest : {guest} ({n_guest} chains)")
    print(f"host  : {host} ({n_host} chains{'' if args.host else ', most abundant other species'})")
    print(f"rule  : {args.mode} (commitment judged on "
          f"{'the guest' + chr(39) + 's own assembly' if args.mode == 'branch' else 'the whole lineage, 3.4.0 rule'})")
    print(f"filter: lineages lasting >= {args.lineage_min_duration} frame(s)")
    print(f"visits: {len(events)}")

    summary = mechanism_summary(events, n_guest_chains=n_guest)
    if summary.empty:
        print("\nno visits recorded; the guest never joined a tracked assembly.")
        return 0
    print("\nmechanism:")
    print(summary.to_string(index=False))

    stats = exchange_statistics(
        events, n_frames=len(cache.frames), n_guest_chains=n_guest,
        frame_interval_ns=args.frame_interval_ns,
    )
    print("\nexchange:")
    for key, value in stats.items():
        print(f"  {key:<38s} {value:.4g}")

    _write_csv(residence_dataframe(events), args.out, "residence events")
    return 0


def _cmd_compare(args) -> int:
    """One row per run, grouped without pooling mistakes."""
    import re

    from .compare import compare_caches, group_summary

    # Campaign caches are usually all called the same thing inside a directory
    # named for the condition, so label by directory when stems collide and
    # match the descriptor regex against the whole path, not the label.
    paths = [str(c).replace("\\", "/") for c in args.caches]
    stems = [os.path.splitext(os.path.basename(p_)) [0] for p_ in paths]
    if len(set(stems)) == len(stems):
        labels = list(stems)
    else:
        labels = [f"{os.path.basename(os.path.dirname(p_))}/{st}" for p_, st in zip(paths, stems)]
        if len(set(labels)) != len(labels):
            labels = list(paths)
    path_of = dict(zip(labels, paths))

    descriptor_fn = None
    if args.descriptor_regex:
        pattern = re.compile(args.descriptor_regex)

        def descriptor_fn(label):  # noqa: F811
            match = pattern.search(path_of.get(label, label) or "")
            return dict(match.groupdict()) if match else {}

        n_hit = sum(1 for p_ in paths if pattern.search(p_))
        if n_hit == 0:
            print(f"warning: --descriptor-regex matched none of the {len(paths)} paths")
        elif n_hit < len(paths):
            print(f"warning: --descriptor-regex matched {n_hit}/{len(paths)} paths")

    df = compare_caches(
        args.caches,
        labels=labels,
        descriptor_fn=descriptor_fn,
        inter_thresh=args.inter_thresh,
        cluster_min_size=args.cluster_min_size,
        last_fraction=args.last_fraction,
        target_config=_target_config(args),
        lineage_min_duration=args.lineage_min_duration,
        include_lineage=not args.no_lineage,
        on_error=("skip" if args.skip_errors else "raise"),
    )
    failed = df[df.get("error").notna()] if "error" in df.columns else df.iloc[0:0]
    if len(failed):
        print(f"{len(failed)} cache(s) failed:")
        for _, row in failed.iterrows():
            print(f"  {row['label']}: {row['error']}")
        df = df[df["error"].isna()]

    cols = [c for c in ("label", "n_chains", "largest_cluster_late",
                        "bound_fraction_late", "on_fraction", "converged")
            if c in df.columns]
    print(df[cols].to_string(index=False))

    if args.group_by:
        by = [c.strip() for c in args.group_by.split(",") if c.strip()]
        grouped = group_summary(df, by)
        # The full table (every metric with mean/sd/sem/n) goes to the CSV;
        # the terminal gets the few columns a person compares by eye.
        shown = list(by) + ["n_runs"]
        for metric in ("largest_cluster_late", "bound_fraction_late", "on_fraction",
                       "on_fraction_hetero", "converged"):
            for suffix in ("_mean", "_sem"):
                if metric + suffix in grouped.columns:
                    shown.append(metric + suffix)
        shown.append("mixed_columns")
        print("\ngrouped (mean and SEM over runs; full table in the CSV):")
        print(grouped[shown].to_string(index=False))
        mixed = grouped[grouped["mixed_columns"] != ""]
        if len(mixed):
            print(
                "\nwarning: these groups pool runs that differ in a column not "
                "in --group-by. Averaging across it produces a mean that "
                "describes neither condition:"
            )
            for _, row in mixed.iterrows():
                keys = ", ".join(f"{c}={row[c]}" for c in by)
                print(f"  {keys}: varies in {row['mixed_columns']}")
        if args.out:
            grouped.to_csv(args.out, index=False)
            print(f"\nwrote {args.out}  (grouped)")
            # The per-run rows are what a grouped mean is accountable to, so
            # they are kept beside it rather than discarded.
            runs_path = os.path.splitext(str(args.out))[0] + "_runs.csv"
            df.to_csv(runs_path, index=False)
            print(f"wrote {runs_path}  ({len(df)} runs)")
            return 0
    _write_csv(df, args.out, "comparison")
    return 0


def _cmd_export(args) -> int:
    """GEXF and transition-matrix export."""
    from .cache_io import load_cache
    from .lineage import build_lineages_from_cache
    from .network import (
        transition_counts,
        transition_dataframe,
        write_lineage_gexf,
        write_transition_gexf,
    )

    cache = load_cache(args.cache)
    smap = None
    if args.species or cache.meta.get("species"):
        smap = _species_map(cache, args.species)

    base = os.path.splitext(os.path.basename(args.cache))[0]
    out_dir = args.out_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    written = []

    if args.what in ("lineage", "both"):
        result = build_lineages_from_cache(
            cache,
            inter_thresh=args.inter_thresh,
            cluster_min_size=args.cluster_min_size,
            target_config=_target_config(args),
            species_map=smap,
        )
        path = os.path.join(out_dir, f"{base}_lineage.gexf")
        write_lineage_gexf(result, path, min_duration=args.lineage_min_duration)
        written.append(path)

    if args.what in ("transitions", "both"):
        names, counts = transition_counts(
            cache,
            inter_thresh=args.inter_thresh,
            min_size=args.cluster_min_size,
            species_map=(smap if args.composition_states else None),
            max_size=args.max_state_size,
        )
        gexf = os.path.join(out_dir, f"{base}_transitions.gexf")
        write_transition_gexf(names, counts, gexf, min_count=args.min_count)
        written.append(gexf)
        csv = os.path.join(out_dir, f"{base}_transitions.csv")
        transition_dataframe(names, counts, min_count=args.min_count).to_csv(
            csv, index=False
        )
        written.append(csv)
        print(f"{len(names)} states, {int(counts.sum())} observed transitions")

    for path in written:
        print(f"wrote {path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coana",
        description="CoAnaPure oligomer contact analysis: Stage A precomputation, "
        "cache inspection, HTML reporting, and engine cross-validation.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "precompute",
        help="run Stage A and write the contact cache (.npz)",
        description="Run CoAnaPure Stage A over a trajectory and write the "
        "compressed contact cache plus per-frame CSV statistics.",
    )
    p.add_argument("--traj", required=True, help="trajectory (.pdb/.gro/.trr/.xtc)")
    p.add_argument("--topology", default=None,
                   help=".pdb/.gro topology (required for .trr/.xtc; otherwise "
                        "defaults to the trajectory's first frame)")
    p.add_argument("--frames", default="all",
                   help="'all' (default), start:stop[:step] (stop exclusive), "
                        "comma list, or a single frame")
    p.add_argument("--cutoff", type=float, default=6.5, help="contact cutoff in angstrom (default 6.5)")
    p.add_argument("--intra-exclude", type=int, default=2,
                   help="ignore intra-monomer contacts closer than this in sequence (default 2)")
    p.add_argument("--segment-mode", default="mdanalysis",
                   choices=["mdanalysis", "atom_blocks", "residue_blocks", "atom_counts"],
                   help="how atoms map to monomers (use atom_blocks for large "
                        "Martini PDBs with reused chain ids)")
    p.add_argument("--atoms-per-monomer", type=int, default=None)
    p.add_argument("--residues-per-monomer", type=int, default=None)
    p.add_argument("--n-monomers", type=int, default=None)
    p.add_argument("--monomer-atom-counts", default=None,
                   help="comma/space list of per-monomer atom counts "
                        "(segment_mode=atom_counts, unequal monomers)")
    p.add_argument("--selection", default=None,
                   help="atom selection, e.g. \"not resname W ION\" or \"resname BB\"")
    p.add_argument("--species", default=None, metavar="SPEC",
                   help="composition in topology order, e.g. "
                        "'aSyn:295x50,Ab40:92x10'. Sets the monomer block sizes "
                        "and records the species in the cache, so no later "
                        "analysis has to be told the composition again")
    p.add_argument("--contact-classes", action="store_true",
                   help="also store the residue-chemistry decomposition of "
                        "inter-chain contacts (hydrophobic / polar / salt "
                        "bridge); cannot be added to a finished cache")
    p.add_argument("--hotspot-blocks", type=int, default=0, metavar="N",
                   help="also store residue-resolved cross-species contact maps "
                        "in N time blocks (needs --species; 1 = whole run)")
    p.add_argument(
        "--contact-mode", choices=("residue_cog", "min_atom"), default="residue_cog",
        help="how a contact is decided: 'residue_cog' (default) uses the "
             "distance between residue centres of geometry; 'min_atom' uses the "
             "minimum distance over all atom pairs, the convention used by "
             "OLIGAMIX, ATRANET and most of the aggregation literature. At a "
             "fixed cutoff 'min_atom' finds at least as many contacts.")
    p.add_argument("--broadcast-box", action="store_true",
                   help="apply the first CRYST1 box to every frame (physically "
                        "correct PBC for constant-volume runs; default keeps "
                        "bit-compatibility with the original pipeline)")
    p.add_argument("--allow-no-box", action="store_true",
                   help="run even if the trajectory has no periodic box "
                        "(only for systems without periodic boundaries)")
    p.add_argument("--fallback-box", default=None,
                   help="box to use when the trajectory has none: 'a,b,c[,al,be,ga]'")
    p.add_argument("--n-jobs", type=int, default=1, help="workers; -1 = all CPUs")
    p.add_argument("--parallel", default="thread", choices=["thread", "process"],
                   help="'process' distributes parsing across CPUs via the model "
                        "index (PDB; recommended on HPC nodes)")
    p.add_argument("--checkpoint-every", type=int, default=0, metavar="N",
                   help="write a crash-safe checkpoint every N frames (0 = off)")
    p.add_argument("--resume", action="store_true",
                   help="continue from a matching checkpoint if one exists")
    p.add_argument("--index-file", default=None,
                   help="PDB model index path (built if missing); default sidecar "
                        "is picked up automatically")
    p.add_argument("-o", "--out", required=True, help="output cache .npz path")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=_cmd_precompute)

    p = sub.add_parser("index", help="build the byte-offset model index for a PDB trajectory")
    p.add_argument("traj")
    p.add_argument("-o", "--out", default=None, help="index path (default: <traj>.coanapure-index.npz)")
    p.set_defaults(func=_cmd_index)

    p = sub.add_parser("info", help="summarize a Stage A cache")
    p.add_argument("cache")
    p.add_argument("--json", action="store_true", help="also dump the full metadata as JSON")
    p.set_defaults(func=_cmd_info)

    p = sub.add_parser(
        "report",
        help="generate the self-contained HTML analysis report from a cache",
        description="Build a publication-grade, single-file HTML report "
        "(figures embedded) covering system evolution, oligomer populations, "
        "contact maps, and lineage/fate analysis.",
    )
    p.add_argument("cache")
    p.add_argument("-o", "--out", default=None,
                   help="output .html (default: <cache>_report.html)")
    p.add_argument("--title", default=None)
    p.add_argument("--subtitle", default=None)
    p.add_argument("--author", default=None)
    p.add_argument("--inter-thresh", type=int, default=1,
                   help="contacts required for a monomer-monomer edge (default 1)")
    p.add_argument("--no-lineage", action="store_true",
                   help="skip the lineage/fate section (faster)")
    p.add_argument("--min-size", type=int, default=10,
                   help="target state: minimum aggregate size (default 10)")
    p.add_argument("--min-duration", type=int, default=5,
                   help="target state: minimum lifetime in frames (default 5)")
    p.add_argument("--min-contribution", type=float, default=0.70,
                   help="on-pathway contribution threshold (default 0.70)")
    p.add_argument("--max-size", type=int, default=None,
                   help="plot calibration: largest oligomer size axis")
    p.add_argument("--max-table-rows", type=int, default=30)
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser(
        "timeline",
        help="build the standalone interactive lineage timeline page from a cache",
        description="Render the split/merge lineage graph on a real frame axis "
        "and inline it into one self-contained HTML page. Unlike the "
        "force-directed graph in the report, every node sits at its true time "
        "coordinate, so events read left to right in the order they happened.",
    )
    p.add_argument("cache")
    p.add_argument("-o", "--out", default=None,
                   help="output .html (default: <cache>_timeline.html)")
    p.add_argument("--frame-lo", type=int, default=None,
                   help="first frame of the window (default: start of cache)")
    p.add_argument("--frame-hi", type=int, default=None,
                   help="last frame of the window, inclusive (default: end of cache)")
    # Lineage construction. Same meanings as build_lineages_from_cache; the
    # defaults match it, so a bare invocation reproduces the library behaviour.
    p.add_argument("--inter-thresh", type=int, default=1,
                   help="contacts required for a monomer-monomer edge (default 1)")
    p.add_argument("--cluster-min-size", type=int, default=2,
                   help="smallest cluster tracked as an oligomer (default 2)")
    p.add_argument("--overlap-min", type=int, default=1,
                   help="shared monomers required to link segments (default 1)")
    p.add_argument("--parent-frac-min", type=float, default=0.30,
                   help="fraction of a parent that must carry into a child (default 0.30)")
    p.add_argument("--child-frac-min", type=float, default=0.30,
                   help="fraction of a child that must come from a parent (default 0.30)")
    p.add_argument("--union-cover-min", type=float, default=0.70,
                   help="coverage needed to call a clean split/merge (default 0.70)")
    # Target state: decides the on/off-pathway colouring of the page.
    p.add_argument("--min-size", type=int, default=10,
                   help="target state: minimum aggregate size (default 10)")
    p.add_argument("--min-duration", type=int, default=5,
                   help="target state: minimum lifetime in frames (default 5)")
    p.add_argument("--min-contribution", type=float, default=0.70,
                   help="on-pathway contribution threshold (default 0.70)")
    p.add_argument("--payload-json", default=None, metavar="PATH",
                   help="also write the intermediate JSON payload")
    p.add_argument("--check-js", action="store_true",
                   help="syntax-check the page's inline script with node, if installed")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=_cmd_timeline)

    p = sub.add_parser(
        "benchmark",
        help="cross-validate CoAnaPure vs MDAnalysis (forwards options to "
        "oligomer_analysis.coanapure.benchmark)",
        add_help=False,
    )
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=_cmd_benchmark)


    p = sub.add_parser(
        "species",
        help="species-resolved contacts and composition for a mixed system",
        description="Decompose an existing cache by species: cross-species "
        "contacts, per-species bound/free counts, and cluster composition. "
        "Reads the cache only -- Stage A does not need to be re-run.",
    )
    p.add_argument("cache")
    p.add_argument("--species", default=None,
                   help="species spec, e.g. 'aSyn:295x50,Ab40:92x10'; "
                        "inferred from the cache when omitted")
    p.add_argument("--inter-thresh", type=int, default=1,
                   help="contacts needed to call two chains bound (default 1)")
    p.add_argument("--min-size", type=int, default=2,
                   help="smallest cluster counted as an oligomer (default 2)")
    p.add_argument("--size-species", default=None,
                   help="rank cluster size by these species only, e.g. 'aSyn'")
    p.add_argument("-o", "--out", default=None, help="write the per-frame table to CSV")
    p.set_defaults(func=_cmd_species)

    p = sub.add_parser(
        "validate",
        help="check a monomer_atom_counts layout against a structure file",
        description="segment_mode='atom_counts' slices atoms into consecutive "
        "blocks; a wrong list silently creates chimeric monomers. This checks "
        "every block boundary against the chain starts in a .gro.",
    )
    p.add_argument("topology", help=".gro structure file")
    p.add_argument("--monomer-atom-counts", required=True,
                   help="comma/space separated per-monomer atom counts")
    p.set_defaults(func=_cmd_validate)

    def add_lineage_args(parser, *, with_target=True):
        """Options shared by every command that builds lineages."""
        parser.add_argument("--inter-thresh", type=int, default=1,
                            help="contacts required for a chain-chain edge (default 1)")
        parser.add_argument("--cluster-min-size", type=int, default=2,
                            help="smallest cluster tracked as an oligomer (default 2). "
                                 "The default tracks every transient dimer; pair it "
                                 "with --lineage-min-duration")
        parser.add_argument("--lineage-min-duration", type=int, default=1, metavar="N",
                            help="keep only lineages lasting at least N frames. The "
                                 "raw table is dominated by two-frame dimers, so a "
                                 "floor here is usually the difference between output "
                                 "and a result")
        if with_target:
            parser.add_argument("--min-size", type=int, default=10,
                                help="target state: minimum aggregate size (default 10)")
            parser.add_argument("--min-duration", type=int, default=5,
                                help="target state: minimum lifetime in frames (default 5)")
            parser.add_argument("--min-contribution", type=float, default=0.70,
                                help="on-pathway contribution threshold (default 0.70)")

    p = sub.add_parser(
        "partition",
        help="how strongly each species is enriched inside the assembly",
        description="Partition coefficient per species (mole fraction inside the "
        "largest assembly over its mole fraction in the box) with a spread, plus "
        "free-chain counts. This is the quantity a microscopy recruitment "
        "measurement reports, so it is what a simulation should quote beside one.",
    )
    p.add_argument("cache")
    p.add_argument("--species", default=None,
                   help="species spec; read from the cache when omitted")
    p.add_argument("--inter-thresh", type=int, default=1)
    p.add_argument("--min-size", type=int, default=2,
                   help="smallest cluster counted as an assembly (default 2)")
    p.add_argument("--last-fraction", type=float, default=0.25,
                   help="fraction of the trajectory treated as 'late' (default 0.25)")
    p.add_argument("-o", "--out", default=None, help="write the summary to CSV")
    p.set_defaults(func=_cmd_partition)

    p = sub.add_parser(
        "chemistry",
        help="decompose inter-chain contacts by interaction type",
        description="Split inter-chain contacts into hydrophobic, polar and "
        "salt-bridge channels using residue chemistry. Needs a cache built with "
        "`precompute --contact-classes`: the residue-pair list is discarded once "
        "contacts have been binned to chains, so it cannot be added afterwards.",
    )
    p.add_argument("cache")
    p.add_argument("--last-fraction", type=float, default=0.25)
    p.add_argument("--top", type=int, default=10, help="class pairs to print")
    p.add_argument("--long", action="store_true",
                   help="write the per-frame long-form table instead of the summary")
    p.add_argument("-o", "--out", default=None)
    p.set_defaults(func=_cmd_chemistry)

    p = sub.add_parser(
        "hotspots",
        help="residue-resolved cross-species contact maps",
        description="Which part of one sequence touches which part of another. "
        "Needs a cache built with `precompute --species ... --hotspot-blocks N`.",
    )
    p.add_argument("cache")
    p.add_argument("--species-a", default=None, help="first species (default: the first)")
    p.add_argument("--species-b", default=None, help="second species (default: the second)")
    p.add_argument("--regions", default=None, metavar="SPEC",
                   help="named 1-based residue ranges of species A, e.g. "
                        "'N-term:1-60,NAC:61-95,C-term:96-140'")
    p.add_argument("--top", type=int, default=20, help="residue pairs to print")
    p.add_argument("-o", "--out", default=None)
    p.set_defaults(func=_cmd_hotspots)

    p = sub.add_parser(
        "converged",
        help="test whether an observable has reached a steady state",
        description="Geweke two-window comparison plus a late-window drift "
        "slope, with an autocorrelation-corrected error bar. Exits non-zero when "
        "the run has not converged, so a batch sweep can flag its own runs.",
    )
    p.add_argument("cache")
    p.add_argument("--observable", default="largest_cluster",
                   help="largest_cluster (default), mean_cluster_size, "
                        "n_clusters, n_bound, or ninter")
    p.add_argument("--all", action="store_true", help="test every observable")
    p.add_argument("--inter-thresh", type=int, default=1)
    p.add_argument("--min-size", type=int, default=2)
    p.add_argument("-o", "--out", default=None)
    p.set_defaults(func=_cmd_converged)

    p = sub.add_parser(
        "lineage",
        help="lineage fate table, species-aware and persistence-filtered",
        description="Build the split/merge lineage graph and summarise the fate "
        "of every assembly. With a species map the table also carries each "
        "lineage's composition, which is what answers whether the productive "
        "lineages are the mixed ones.",
    )
    p.add_argument("cache")
    p.add_argument("--species", default=None,
                   help="species spec; read from the cache when omitted")
    p.add_argument("--sweep", default=None, metavar="SIZES",
                   help="also re-classify against these target sizes, e.g. "
                        "'6,10,15', and report whether the conclusion survives")
    add_lineage_args(p)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--directed-out", default=None, metavar="CSV",
                   help="write the directed fate of every nucleation event "
                        "(forward descendants only) to CSV")
    p.set_defaults(func=_cmd_lineage)

    p = sub.add_parser(
        "mechanism",
        help="is the minority species incorporated, catalytic, or a bystander?",
        description="Classify every visit a guest chain makes to an assembly by "
        "its relationship to the assembly's commitment: incorporated (present at "
        "commitment and at the end), catalytic (present at commitment, then "
        "leaves), pre-commitment (left before the commitment size was reached), passenger (joins an already-committed aggregate), promoting, or "
        "bystander. This is the molecular form of a heterogeneous-nucleation "
        "claim.",
    )
    p.add_argument("cache")
    p.add_argument("--species", default=None,
                   help="species spec; read from the cache when omitted")
    p.add_argument("--guest", default=None,
                   help="species whose mechanism is in question (default: the "
                        "least abundant). Passing the majority species is not a "
                        "fair control; designate random host chains as "
                        "pseudo-guests instead")
    p.add_argument("--host", default=None,
                   help="species whose assembly is counted (default: the most "
                        "abundant other species)")
    p.add_argument("--commit-size", type=int, default=None,
                   help="host chains an assembly must reach to count as "
                        "committed (default: --min-size)")
    p.add_argument("--after-window", type=int, default=10, metavar="N",
                   help="frames past a departure to check whether the assembly "
                        "kept growing (default 10)")
    p.add_argument("--commit-duration", type=int, default=None, metavar="N",
                   help="consecutive frames an assembly must hold --commit-size "
                        "host chains to count as committed (default: --min-duration)")
    p.add_argument("--mode", choices=("branch", "lineage"), default="branch",
                   help="judge commitment on the guest's own assembly (branch, "
                        "default since 3.5.0) or on the whole split/merge lineage "
                        "(lineage, the 3.4.0 rule, kept for comparison)")
    p.add_argument("--frame-interval-ns", type=float, default=None,
                   help="ns between cache frames, to report exchange rates in time")
    add_lineage_args(p)
    p.add_argument("-o", "--out", default=None,
                   help="write the per-visit residence table to CSV")
    p.set_defaults(func=_cmd_mechanism)

    p = sub.add_parser(
        "compare",
        help="summarise many caches into one table and group them safely",
        description="One row per run, then optional grouping. The grouped table "
        "reports which descriptor columns vary *within* each group, so pooling "
        "two conditions into one mean is visible rather than silent.",
    )
    p.add_argument("caches", nargs="+")
    p.add_argument("--group-by", default=None, metavar="COLS",
                   help="comma-separated descriptor columns to group by")
    p.add_argument("--descriptor-regex", default=None, metavar="RE",
                   help="regex with named groups run against each cache's name; "
                        "the groups become descriptor columns, e.g. "
                        r"'(?P<conc>\d\.\d)(?P<variant>_\w+)_rep(?P<rep>\d)'")
    p.add_argument("--last-fraction", type=float, default=0.25)
    p.add_argument("--no-lineage", action="store_true",
                   help="skip lineage construction (much faster)")
    p.add_argument("--skip-errors", action="store_true",
                   help="keep going past unreadable caches and list them")
    add_lineage_args(p)
    p.add_argument("-o", "--out", default=None)
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser(
        "export",
        help="write GEXF graphs and the state-transition matrix",
        description="Export the lineage provenance graph and/or the coarse "
        "state-transition network as GEXF (Gephi, Cytoscape) plus a CSV edge "
        "table. The row-normalised transition counts are the Markov matrix the "
        "transition-network literature starts from.",
    )
    p.add_argument("cache")
    p.add_argument("--what", default="both",
                   choices=["lineage", "transitions", "both"])
    p.add_argument("--species", default=None)
    p.add_argument("--composition-states", action="store_true",
                   help="label transition states by size *and* composition, so a "
                        "decamer that recruited two guest chains is a different "
                        "state from a pure one")
    p.add_argument("--max-state-size", type=int, default=None, metavar="N",
                   help="collapse every cluster of N or more chains into one "
                        "capped state '>=N', keeping the network readable. It is "
                        "not absorbing: a cluster dropping below N leaves it again")
    p.add_argument("--min-count", type=int, default=1,
                   help="drop edges observed fewer than this many times")
    p.add_argument("--out-dir", default=None)
    add_lineage_args(p)
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("version", help="print package and engine versions")
    p.set_defaults(func=_cmd_version)

    p = sub.add_parser("citations", help="what to cite when CoAna contributes to published work")
    p.add_argument("--bibtex", action="store_true", help="print BibTeX entries instead of text")
    p.set_defaults(func=_cmd_citations)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["benchmark"]:
        # argparse.REMAINDER cannot capture a leading "--help"; forward verbatim.
        return _cmd_benchmark(argparse.Namespace(rest=argv[1:]))
    args = _build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
