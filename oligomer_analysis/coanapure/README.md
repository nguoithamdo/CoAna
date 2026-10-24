# CoAnaPure 1.1

A pure-NumPy/SciPy re-implementation of **Stage A** (the expensive,
trajectory-facing precomputation) for the oligomer-analysis toolkit. CoAnaPure
removes the hard dependency on **MDAnalysis** by re-implementing exactly the
pieces the original `oligomer_analysis.precompute` used.

## What's new in 1.1

| # | Feature | Where |
| --- | --- | --- |
| 1 | **Non-periodic cell list**: the box-less neighbour search no longer falls back to O(N^2) without scipy | `neighbors.py` |
| 2 | **Constant-memory streaming**: frames are parsed and consumed one at a time; peak RSS no longer grows with trajectory length | `PDBFrameStream`, `precompute_cache_pure` |
| 3 | **Random access**: a model byte-offset index (`build_model_index`) lets sparse frame subsets seek directly instead of parsing the file prefix | `PDBModelIndex` |
| 4 | **GRO and TRR readers** (pure NumPy, nm -> angstrom): `.gro` single/multi-frame, `.trr` with a `.pdb`/`.gro` topology; XTC via `gmx trjconv -f traj.xtc -o traj.trr` | `gro_io.py`, `trr_io.py` |
| 5 | **Threaded frame loop**: `cfg.n_jobs` (joblib convention, `-1` = all CPUs) runs contact computation on a thread pool fed by the streaming parser | `precompute_cache_pure` |
| 6 | **Dropped-box warning**: when MDAnalysis semantics silently discard a leading `CRYST1`, a `UserWarning` points to `broadcast_box=True` | `pdb_io.py` |

All 1.1 paths (streamed, threaded, indexed, no-scipy) produce **byte-identical
caches** to CoAnaPure 1.0, which was validated field-by-field against
MDAnalysis.

| MDAnalysis feature | CoAnaPure replacement |
| --- | --- |
| `MDAnalysis.lib.distances.minimize_vectors` | `coanapure.pbc.minimize_vectors` |
| `MDAnalysis.lib.distances.capped_distance` | `coanapure.neighbors.capped_distance` |
| `MDAnalysis.Universe(...)` PDB reading | `coanapure.pdb_io.read_pdb` |

The output cache format (`.npz`) is **unchanged**, so every downstream module
(Stage B, lineage/fate tracking, LLPS, hotspots, plotting) works without edits.

## Why

- **No MDAnalysis install** needed to build caches -- only `numpy`.
- **Transparent geometry**: PBC minimum image and the neighbour search are plain,
  readable NumPy you can audit and modify.
- **Validated**: `coanapure.benchmark` runs both engines on the same trajectory
  and compares the caches array-by-array.

### Dependencies

- Required: `numpy` and `scipy` (the neighbour search uses
  `scipy.spatial.cKDTree`; C-speed, periodic-box aware).
- Without scipy the search falls back to pure-NumPy cell lists (periodic when a
  box is present, bounding-box gridded when not) -- byte-identical results, a
  `RuntimeWarning` at import, and several times slower. No MDAnalysis in any
  case.

## Usage

```python
from oligomer_analysis import load_cache
from oligomer_analysis.coanapure import precompute_cache_pure, PrecomputeConfig

cfg = PrecomputeConfig(
    topology_file=None,           # topology read from the trajectory PDB itself
    traj_file="traj.pdb",
    frames=list(range(250)),
    cutoff_contact=6.5,
    intra_exclude=2,
    segment_mode="atom_blocks",   # monomers = fixed atom blocks (reused chain IDs)
    atoms_per_monomer=83,
    n_monomers=128,
)
precompute_cache_pure(cfg, out_npz="cache_pure.npz")
cache = load_cache("cache_pure.npz")
```

New in 1.1 -- parallel frames, random access, and other formats:

```python
# All CPU threads for the per-frame contact computation:
cfg = PrecomputeConfig(..., n_jobs=-1)

# Build a byte-offset index once, then sparse subsets seek instead of streaming:
from oligomer_analysis.coanapure import build_model_index
build_model_index("traj.pdb")                     # writes traj.pdb.coanapure-index.npz
cfg = PrecomputeConfig(..., frames=[1900, 1950])  # now seeks directly
precompute_cache_pure(cfg, out_npz="tail.npz")    # fresh sidecar picked up automatically

# GRO trajectory (multi-frame, e.g. from gmx trjconv -o traj.gro):
cfg = PrecomputeConfig(topology_file=None, traj_file="traj.gro", ...)

# TRR trajectory with a PDB or GRO topology:
cfg = PrecomputeConfig(topology_file="conf.gro", traj_file="traj.trr", ...)
```

## Benchmark / validation

```bash
python -m oligomer_analysis.coanapure.benchmark \
    --traj traj.pdb --frames 0:250 \
    --segment-mode atom_blocks --atoms-per-monomer 83 --n-monomers 128 \
    --cutoff 6.5 --intra-exclude 2
```

This runs MDAnalysis and CoAnaPure, then prints a per-field comparison and a
`VERDICT`. On the reference 128-monomer / 250-frame Martini trajectory:

- `cmaps` (inter-monomer contact maps), `ninter`, `n_free`, `n_bound`, `frames`,
  and `boxes` match **exactly** on every frame;
- `seg_centers` match to ~1e-4 A (float32 rounding only);
- `nintra`/`ntotal` (intra-monomer bead-contact counts) differ by at most 1 on a
  handful of frames. This is a distance-cutoff boundary effect: MDAnalysis
  evaluates distances in float32 and includes/excludes pairs whose true distance
  is within ~1e-5 A of the 6.5 A cutoff differently from CoAnaPure's float64.
  CoAnaPure is the more accurate side (verified: MDAnalysis counted a pair at
  6.500011 A as "in" and one at 6.499980 A as "out").

### Measured performance (128 monomers, cutoff 6.5 A)

| Trajectory | Frames | MDAnalysis | CoAnaPure | Notes |
| --- | --- | --- | --- | --- |
| `250frame.pdb` (212 MB) | 250 | 47.8 s | **5.8 s** (8.3x) | full file, both engines |
| `md1_...1956frames.pdb` (1.66 GB) | 30 | 48.0 s | 57.3 s | subset: MDAnalysis seeks 30 frames lazily; CoAnaPure still streams the whole file |
| `md1_...1956frames.pdb` (1.66 GB) | 1956 | ~50 min (est.\*) | **90 s** (~35x) | CoAnaPure reads the file once, ~0.15 GB RAM |

\* MDAnalysis processes this multi-model PDB by random access at ~1.6 s/frame,
so the full trajectory extrapolates to roughly 50 minutes single-threaded.
CoAnaPure's advantage grows with the fraction of the trajectory analysed, because
it streams the file once instead of seeking per frame.

## Important: periodic box behaviour

The MDAnalysis PDB reader attaches a `CRYST1` box to a model **only** when the
`CRYST1` record sits inside that model's `MODEL`/`ENDMDL` block. A *leading*
`CRYST1` (one line at the top of the file, before the first `MODEL`) is honoured
only for single-model files; in a **multi-model trajectory it is dropped from
every frame**.

Consequently, a typical Martini trajectory PDB -- many models, one top-of-file
`CRYST1` -- is processed by the original pipeline **without periodic boundaries**.

CoAnaPure reproduces this exactly by default (`broadcast_box=False`) so its cache
is bit-comparable with the original. For new, physically-correct analyses pass
`broadcast_box=True` to apply the single `CRYST1` box to every frame:

```python
precompute_cache_pure(cfg, out_npz="cache_pbc.npz", broadcast_box=True)
```

## Supported inputs

- PDB topology + trajectory (single- or multi-`MODEL`).
- GRO topology and/or trajectory (single- or multi-frame, as written by
  `gmx trjconv -o out.gro`); coordinates converted nm -> angstrom.
- TRR trajectory (float or double precision) with a PDB or GRO topology.
- XTC is intentionally not parsed (lossy 3dfcoord decompression); convert first:
  `gmx trjconv -f traj.xtc -s topol.tpr -o traj.trr`.
- Segmentation modes: `atom_blocks`, `residue_blocks`, and `mdanalysis`
  (unique-chain-id grouping, including the reused-chain-id merging behaviour).
- Orthorhombic and triclinic boxes (orthorhombic matches MDAnalysis bit-for-bit;
  triclinic uses a 27-image minimum-image search).

## Not reproduced on purpose

The MDAnalysis-only "fast vectorised COG" path (`unwrap`/`wrap` transformations,
enabled only for `segment_mode="mdanalysis"` with bonded topology) is not
reproduced. CoAnaPure always uses the deterministic per-residue PBC-safe centre
of geometry, which is the path the original code already uses for
`atom_blocks`/`residue_blocks`.
