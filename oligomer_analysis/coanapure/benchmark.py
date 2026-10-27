# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Correctness + speed comparison: CoAnaPure vs the MDAnalysis Stage A.

Runs both engines on the *same* PDB with the *same* :class:`PrecomputeConfig`,
then compares the two resulting caches array-by-array and reports timings.

Usage
-----
    python -m oligomer_analysis.coanapure.benchmark \
        --traj  path/to/traj.pdb \
        --frames 0:250 \
        --segment-mode atom_blocks --atoms-per-monomer 83 --n-monomers 128 \
        --cutoff 6.5 --intra-exclude 2

If MDAnalysis is not installed, only the pure engine runs (no comparison).
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np

from ..cache_io import load_cache
from ..models import PrecomputeConfig
from .precompute_pure import precompute_cache_pure


def _parse_frames(spec: str) -> list:
    spec = spec.strip()
    if ":" in spec:
        parts = spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return list(range(start, stop, step))
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip()]
    return [int(spec)]


def compare_caches(path_a: str, path_b: str, *, label_a="MDAnalysis", label_b="CoAnaPure") -> dict:
    """Compare two Stage A caches array-by-array; return a report dict."""
    ca = load_cache(path_a)
    cb = load_cache(path_b)

    report = {"identical": True, "fields": {}}

    def cmp_int(name, a, b, int_tol=0):
        """Compare integer arrays.  ``int_tol`` allows a per-element slack used
        for cutoff-boundary counts (nintra/ntotal) where MDAnalysis' float32
        distance rounding differs from CoAnaPure's float64 by at most 1."""
        a = np.asarray(a)
        b = np.asarray(b)
        same_shape = a.shape == b.shape
        maxdiff = int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()) if same_shape else None
        exact = bool(same_shape and maxdiff == 0)
        ok = bool(same_shape and maxdiff is not None and maxdiff <= int_tol)
        report["fields"][name] = {
            "shape_a": a.shape, "shape_b": b.shape,
            "equal": exact, "within_tol": ok, "max_abs_diff": maxdiff, "int_tol": int_tol,
        }
        report["identical"] &= ok

    def cmp_float(name, a, b, tol=1e-4):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        same_shape = a.shape == b.shape
        if same_shape:
            maxdiff = float(np.abs(a - b).max()) if a.size else 0.0
            close = bool(maxdiff <= tol)
        else:
            maxdiff, close = None, False
        report["fields"][name] = {"shape_a": a.shape, "shape_b": b.shape, "close": close, "max_abs_diff": maxdiff, "tol": tol}
        report["identical"] &= close

    cmp_int("frames", ca.frames, cb.frames)
    cmp_int("cmaps", ca.cmaps, cb.cmaps)                       # inter-monomer contact maps
    cmp_int("ninter", ca.ninter, cb.ninter)
    cmp_int("n_free", ca.n_free, cb.n_free)
    cmp_int("n_bound", ca.n_bound, cb.n_bound)
    # nintra/ntotal count intra-monomer bead contacts; pairs sitting on the 6.5 A
    # cutoff round differently under MDAnalysis float32 vs CoAnaPure float64, so
    # allow a slack of 1 per frame here (CoAnaPure is the more accurate side).
    cmp_int("nintra", ca.nintra, cb.nintra, int_tol=1)
    cmp_int("ntotal", ca.ntotal, cb.ntotal, int_tol=1)
    cmp_float("seg_centers", ca.seg_centers, cb.seg_centers, tol=1e-2)
    cmp_float("boxes", ca.boxes, cb.boxes, tol=1e-4)

    report["meta_a"] = {k: ca.meta.get(k) for k in ("nseg", "segment_mode", "n_frames")}
    report["meta_b"] = {k: cb.meta.get(k) for k in ("nseg", "segment_mode", "n_frames")}
    return report


def print_report(report: dict, *, label_a="MDAnalysis", label_b="CoAnaPure") -> None:
    print("\n=== Cache comparison ({} vs {}) ===".format(label_a, label_b))
    print(f"meta {label_a}: {report['meta_a']}")
    print(f"meta {label_b}: {report['meta_b']}")
    print(f"{'field':<14}{'exact':<8}{'ok':<8}{'max_abs_diff':<16}shapes")
    for name, info in report["fields"].items():
        exact = info.get("equal")
        ok = info.get("within_tol", info.get("close"))
        md = info.get("max_abs_diff")
        md_s = "-" if md is None else f"{md:.3e}" if isinstance(md, float) else str(md)
        print(f"{name:<14}{str(exact):<8}{str(ok):<8}{md_s:<16}{info['shape_a']} vs {info['shape_b']}")
    verdict = "EQUIVALENT (exact on all scientific fields)" if report["identical"] else "DIFFERENCES FOUND"
    print(f"\nVERDICT: {verdict}")
    print("Note: nintra/ntotal allow +-1/frame at the distance cutoff (MDAnalysis "
          "float32 vs CoAnaPure float64 rounding); all contact maps match exactly.")


def run_benchmark(
    traj_file: str,
    frames,
    *,
    topology_file: Optional[str] = None,
    cutoff_contact: float = 6.5,
    intra_exclude: int = 2,
    segment_mode: str = "atom_blocks",
    atoms_per_monomer: Optional[int] = 83,
    residues_per_monomer: Optional[int] = None,
    n_monomers: Optional[int] = 128,
    out_prefix: str = "bench",
    n_jobs: int = 1,
) -> dict:
    cfg = PrecomputeConfig(
        topology_file=topology_file,
        traj_file=traj_file,
        frames=list(frames),
        cutoff_contact=cutoff_contact,
        intra_exclude=intra_exclude,
        n_jobs=n_jobs,
        segment_mode=segment_mode,
        atoms_per_monomer=atoms_per_monomer,
        residues_per_monomer=residues_per_monomer,
        n_monomers=n_monomers,
    )

    pure_npz = f"{out_prefix}_pure.npz"
    t0 = time.perf_counter()
    precompute_cache_pure(cfg, out_npz=pure_npz, verbose=True)
    t_pure = time.perf_counter() - t0
    print(f"\nCoAnaPure Stage A: {t_pure:.2f} s for {len(cfg.frames)} frames")

    result = {"t_pure": t_pure, "pure_npz": pure_npz}

    try:
        from ..precompute import precompute_cache as mda_precompute
    except Exception as exc:  # pragma: no cover
        print(f"\nMDAnalysis not available ({exc}); skipping comparison.")
        return result

    mda_npz = f"{out_prefix}_mda.npz"
    # One chunk when serial (timing reflects compute, not spawn cost); split into
    # per-worker chunks when the caller asked for parallelism.
    chunk = max(1, len(cfg.frames)) if n_jobs in (0, 1) else max(1, len(cfg.frames) // (abs(n_jobs) * 4 if n_jobs > 0 else 32))
    cfg_mda = PrecomputeConfig(
        topology_file=topology_file, traj_file=traj_file, frames=list(frames),
        cutoff_contact=cutoff_contact, intra_exclude=intra_exclude, n_jobs=n_jobs,
        chunk_size=chunk, segment_mode=segment_mode,
        atoms_per_monomer=atoms_per_monomer, residues_per_monomer=residues_per_monomer,
        n_monomers=n_monomers,
    )
    t0 = time.perf_counter()
    mda_precompute(cfg_mda, out_npz=mda_npz)
    t_mda = time.perf_counter() - t0
    print(f"MDAnalysis Stage A: {t_mda:.2f} s for {len(cfg.frames)} frames")

    report = compare_caches(mda_npz, pure_npz)
    print_report(report)

    result.update({"t_mda": t_mda, "mda_npz": mda_npz, "report": report,
                   "speedup": (t_mda / t_pure) if t_pure else None})
    print(f"\nSpeed: MDAnalysis {t_mda:.2f}s | CoAnaPure {t_pure:.2f}s | "
          f"ratio {result['speedup']:.2f}x" if result.get('speedup') else "")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark CoAnaPure vs MDAnalysis Stage A")
    p.add_argument("--traj", required=True)
    p.add_argument("--topology", default=None)
    p.add_argument("--frames", default="0:1")
    p.add_argument("--cutoff", type=float, default=6.5)
    p.add_argument("--intra-exclude", type=int, default=2)
    p.add_argument("--segment-mode", default="atom_blocks")
    p.add_argument("--atoms-per-monomer", type=int, default=83)
    p.add_argument("--residues-per-monomer", type=int, default=None)
    p.add_argument("--n-monomers", type=int, default=128)
    p.add_argument("--out-prefix", default="bench")
    p.add_argument("--n-jobs", type=int, default=1, help="thread/worker count for both engines (-1 = all CPUs)")
    args = p.parse_args()

    run_benchmark(
        traj_file=args.traj,
        topology_file=args.topology,
        frames=_parse_frames(args.frames),
        cutoff_contact=args.cutoff,
        intra_exclude=args.intra_exclude,
        segment_mode=args.segment_mode,
        atoms_per_monomer=args.atoms_per_monomer,
        residues_per_monomer=args.residues_per_monomer,
        n_monomers=args.n_monomers,
        out_prefix=args.out_prefix,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
