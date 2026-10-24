# CoAnaPure User Guide

**Package:** `oligomer-analysis` 3.5.1 · **Engine:** CoAnaPure 1.3 · **Guide date:** 2026-10-09

A complete, example-driven guide to the toolkit: from a raw MD trajectory to
oligomer contact maps, aggregation-pathway tracking, split/merge-aware lineage
and fate analysis, publication figures, and the one-command HTML report.
Numbers quoted in this guide come from a reference dataset (a Martini system
of 128 monomers × 83 beads, 1956 frames), produced with exactly the code shown
next to them.

> This guide is about *using* the toolkit. Section 14 covers running on a
> SLURM cluster.

---

## Contents

1. [Concepts: what the toolkit computes](#1-concepts-what-the-toolkit-computes)
2. [Installation](#2-installation)
3. [Quick start: trajectory → report in three commands](#3-quick-start-trajectory--report-in-three-commands)
4. [Stage A: building the contact cache](#4-stage-a-building-the-contact-cache)
5. [The cache: loading, inspecting, subsetting](#5-the-cache-loading-inspecting-subsetting)
6. [Whole-system analysis](#6-whole-system-analysis)
7. [Classic pathway tracking and the 9-panel dashboard](#7-classic-pathway-tracking-and-the-9-panel-dashboard)
8. [Lineage and fate analysis](#8-lineage-and-fate-analysis)
9. [Publication layer: commitment, predictors, LLPS, hotspots](#9-publication-layer-commitment-predictors-llps-hotspots)
10. [Tables and exports](#10-tables-and-exports)
11. [Interactive temporal networks](#11-interactive-temporal-networks)
12. [Plot calibration and display options](#12-plot-calibration-and-display-options)
13. [The one-command HTML report](#13-the-one-command-html-report)
14. [Running on an HPC cluster](#14-running-on-an-hpc-cluster)
15. [Validation and reliability](#15-validation-and-reliability)
16. [Troubleshooting and FAQ](#16-troubleshooting-and-faq)
17. [Mixed systems and mechanism (v3.4)](#17-mixed-systems-and-mechanism-v34)
18. [Appendix: API index](#18-appendix-api-index)

---

## 1. Concepts: what the toolkit computes

The toolkit answers a specific scientific question about self-assembling
(aggregating) proteins in MD trajectories:

> Which oligomeric lineages progress toward productive aggregation, and which
> are off-pathway traps, dead ends, or transient nonproductive species?

Four definitions carry the whole analysis:

| Term | Definition |
|---|---|
| **Contact** | Two residues (bead groups) whose PBC-safe centres of geometry lie within `cutoff_contact` (typically 6.5 Å). Residue pairs closer than `intra_exclude` positions in sequence are excluded from the intra-monomer count. |
| **Contact map** `cmap[i,j]` | The number of residue–residue contacts between monomers *i* and *j* in one frame. |
| **Oligomer / cluster** | A connected component of the monomer graph in which an edge exists when `cmap[i,j] ≥ inter_thresh`. Size 1 = free monomer; size ≥ 2 = oligomer. |
| **On/off-pathway** | Decided by a lineage's **future fate** — whether its monomers persistently commit to a target aggregate — never by size alone. A temporary split (10-mer → 6+4 → 10-mer) stays one lineage with explicit split and merge events. |

The architecture is two-stage, and the stages talk through one file:

```
trajectory (.pdb/.gro/.trr/.xtc)
        │  Stage A — CoAnaPure engine (expensive, run once)
        ▼
  cache .npz  (cmaps · ninter · n_free · n_bound · seg_centers · boxes · meta)
        │  Stage B and beyond — cache-only (cheap, repeat freely)
        ▼
  system analysis · pathway tracking · lineages & fate · publication figures
  tables & Excel · interactive networks · HTML report
```

Because everything downstream reads only the cache, you can re-run any
analysis, change thresholds, or make new figures without touching the
trajectory again.

## 2. Installation

```bash
pip install coana        # from PyPI; installs the `coana` command
pip install .            # or from the package directory
coana version            # -> oligomer-analysis 3.5.1 (CoAnaPure engine 1.3)
python -m pytest tests -q   # optional self-check: 62 passed
```

Requirements: Python ≥ 3.10 with NumPy, SciPy, pandas, matplotlib (plus
joblib, openpyxl, ipywidgets, plotly, networkx for specific features). No
compiled extensions of its own, so it installs anywhere — including cluster
login nodes.

**MDAnalysis is not required.** Install it only for cross-validation or exotic
formats: `pip install ".[mdanalysis]"`.

In a notebook, either install the package or put the package directory on the
path:

```python
from pathlib import Path
import sys
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

## 3. Quick start: trajectory → report in three commands

```bash
# 1. Stage A: build the contact cache (all CPU cores, crash-safe)
coana precompute --traj md1_centered_filtered_1956frames.pdb \
    --topology initial1.pdb --frames all \
    --cutoff 6.5 --intra-exclude 2 \
    --segment-mode atom_blocks --atoms-per-monomer 83 --n-monomers 128 \
    --n-jobs -1 --parallel process --checkpoint-every 200 --resume \
    -o cache_cut6.5.npz

# 2. Sanity-check what was computed
coana info cache_cut6.5.npz

# 3. One self-contained HTML analysis report (figures embedded, prints to PDF)
coana report cache_cut6.5.npz -o report.html --title "128-mer aggregation" --max-size 100
```

The same from Python:

```python
from oligomer_analysis import (
    PrecomputeConfig, precompute_cache_pure, load_cache, generate_html_report,
)

cfg = PrecomputeConfig(
    topology_file="initial1.pdb",
    traj_file="md1_centered_filtered_1956frames.pdb",
    frames=list(range(1956)),
    cutoff_contact=6.5, intra_exclude=2, n_jobs=-1,
    segment_mode="atom_blocks", atoms_per_monomer=83, n_monomers=128,
)
precompute_cache_pure(cfg, out_npz="cache_cut6.5.npz", parallel="process")
generate_html_report("cache_cut6.5.npz", "report.html", title="128-mer aggregation")
```

## 4. Stage A: building the contact cache

### 4.1 Input formats

| Format | Support | Notes |
|---|---|---|
| `.pdb` | native, streaming | single- or multi-`MODEL`; byte-offset index for random access; the fastest path with `parallel="process"` |
| `.gro` | native, streaming | single- or multi-frame |
| `.trr` | native | float32/float64 XDR; needs `topology_file` (`.pdb`/`.gro`) |
| `.xtc` | native | pure-Python 3dfcoord decompressor; needs `topology_file`; the slowest reader — for repeated heavy use convert once: `gmx trjconv -f traj.xtc -o traj.trr` |
| anything else (DCD, …) | via MDAnalysis extra | `precompute_cache` (original engine), or convert with `gmx trjconv` |

Coordinates from GRO/TRR/XTC are converted nm → Å automatically.

### 4.2 Defining monomers (segment modes)

The single most common pitfall in large Martini systems: **PDB chain IDs are
one character**, so a 128-monomer system may be read as ~63 "chains" — merging
distinct monomers and silently corrupting every count. Diagnose with:

```python
from oligomer_analysis import pdb_chain_id_summary
pdb_chain_id_summary("initial1.pdb", atoms_per_monomer=83)
```

Then pick the segment mode that matches your topology:

| `segment_mode` | Monomer definition | When to use |
|---|---|---|
| `"mdanalysis"` (default) | unique chain-ID values | small systems with truly unique chain IDs |
| `"atom_blocks"` | `monomer = atom_index // atoms_per_monomer` | **the safe choice for large Martini PDBs**; equal-size monomers |
| `"residue_blocks"` | consecutive residue blocks of `residues_per_monomer` | monomers defined by residue count |
| `"atom_counts"` | explicit per-monomer atom counts, `monomer_atom_counts=[83, 83, 120, ...]` | mixed systems with unequal monomers |

### 4.3 Atom selection

Restrict the analysis to a subset of atoms *before* segmentation — for
example, drop solvent or keep only backbone beads:

```python
cfg = PrecomputeConfig(..., selection="not resname W ION")   # drop solvent
cfg = PrecomputeConfig(..., selection="resname BB")          # backbone beads only
```

Grammar (one clause, optionally negated; ranges are inclusive):

```
all
[not] resname NAME [NAME ...]     e.g. "not resname W ION NA CL"
[not] resid RANGE [RANGE ...]     e.g. "resid 1-128"
[not] chain ID [ID ...]           e.g. "chain A B"
[not] index RANGE [RANGE ...]     e.g. "index 0-9959"   (0-based atom index)
```

**Important:** with block segmentation, `atoms_per_monomer` then counts the
*selected* atoms per monomer (e.g. 83-bead monomers with 20 backbone beads →
`selection="resname BB", atoms_per_monomer=20`). The selection is recorded in
the cache metadata.

### 4.4 Periodic boundaries: an explicit policy

A quirk inherited from the original pipeline (full story in the engine
report, §8): in a multi-model PDB, a single *leading* `CRYST1` line is
**dropped from every frame** by MDAnalysis semantics, so such trajectories
have always been analysed **without PBC**.

- **Default (`broadcast_box=False`)** reproduces that behaviour bit-for-bit —
  old results stay reproducible — and a `UserWarning` tells you when it
  happens.
- **`broadcast_box=True`** (CLI: `--broadcast-box`) applies the box to every
  frame — physically correct for constant-volume runs and **recommended for
  new analyses**. State your choice in your methods section.
- `fallback_box=(a, b, c, alpha, beta, gamma)` supplies a box when the file
  has none at all.

### 4.5 Performance: workers, threads, processes, the index

```python
precompute_cache_pure(cfg, out_npz="cache.npz")                       # serial
precompute_cache_pure(cfg, out_npz="cache.npz")                       # cfg.n_jobs=-1: threads
precompute_cache_pure(cfg, out_npz="cache.npz", parallel="process")   # processes
```

- **Threads** (`cfg.n_jobs`, default mode) parallelise only the contact math.
  On parse-bound trajectories they add ~10 % at best.
- **`parallel="process"`** distributes whole frame chunks — *parsing
  included* — across worker processes that seek through the byte-offset
  index (built automatically). This is the mode that scales: on the
  reference 1.66 GB trajectory, 44.8 s serial → **13.0 s** on 8 cores, with
  byte-identical output. PDB input only; other formats fall back to threads
  with a warning.
- **The index** (`coana index traj.pdb`, or automatic) is a one-off byte-offset
  sidecar (`<traj>.coanapure-index.npz`, staleness-checked) that also makes
  sparse frame subsets seek directly: a 56-frame tail of the reference file
  drops from 26.6 s to 1.3 s.

Full measured tables for every version and mode: engine report §6.6–§6.7.

### 4.6 Long jobs: progress, checkpoint, resume

```python
precompute_cache_pure(cfg, out_npz="cache.npz", parallel="process",
                      checkpoint_every=200, resume=True)
```

- Progress lines (rate + ETA) print at most every 2 s — readable in a
  terminal *and* in a SLURM log.
- `checkpoint_every=N` writes an atomic checkpoint
  (`cache.npz.checkpoint.npz`) every N frames. `resume=True` continues an
  interrupted run. The checkpoint is fingerprinted against the complete
  configuration (trajectory, frames, cutoff, selection, …), so a changed
  parameter can never contaminate a resumed cache. It is deleted after a
  successful save, and a resumed cache is byte-identical to an uninterrupted
  run.
- **Windows note:** a *script* using `parallel="process"` must call it under
  `if __name__ == "__main__":` (spawn re-imports the script in each worker;
  CoAnaPure raises a clear error if you forget). The `coana` CLI is already
  guarded; Linux needs no guard.

### 4.7 CLI reference

```
coana precompute   run Stage A            (--traj --topology --frames --cutoff
                                           --intra-exclude --segment-mode
                                           --atoms-per-monomer --n-monomers
                                           --monomer-atom-counts --selection
                                           --broadcast-box --fallback-box
                                           --n-jobs --parallel --checkpoint-every
                                           --resume --index-file -o --quiet)
coana index        build the byte-offset model index for a PDB
coana info         summarize a cache      (--json for full metadata)
coana report       build the HTML report  (see §13 for options)
coana benchmark    cross-validate CoAnaPure vs MDAnalysis
coana version      print versions
```

Frame specifications use Python slice semantics: `0:1956` = frames 0..1955
(stop exclusive), `0:1956:5` = every 5th, `0,10,20` = a list, `all` = the
whole trajectory (frame count resolved via the index or format headers).
`python -m oligomer_analysis …` is equivalent to `coana …`.

## 5. The cache: loading, inspecting, subsetting

```python
from oligomer_analysis import load_cache
cache = load_cache("cache_cut6.5.npz")

cache.frames        # (n_frames,)  trajectory frame numbers
cache.cmaps         # (n_frames, nseg, nseg)  monomer contact maps
cache.nintra, cache.ninter, cache.ntotal   # per-frame contact counts
cache.n_free, cache.n_bound                # per-frame monomer states
cache.seg_centers   # (n_frames, nseg, 3)  monomer centres of geometry
cache.boxes         # (n_frames, 6)        per-frame unit cell
cache.meta          # dict: cutoff, segment mode, engine tag, selection, ...
```

Utilities:

```python
from oligomer_analysis import (
    subset_cache_by_frames, subset_cache_by_indices, save_contact_cache,
)
cache_250 = subset_cache_by_frames(cache, range(1, 251))   # by frame number
save_contact_cache(cache_250, "cache_250.npz")             # save the subset
```

From the shell: `coana info cache.npz` prints frames, monomers, engine,
PBC status, selection, and the final bound fraction.

Alongside every cache, Stage A writes `<name>_contact_stats.csv` with the
per-frame `nintra/ninter/ntotal/n_free/n_bound` series — convenient for quick
plotting in any tool.

## 6. Whole-system analysis

The first look at any run: how much aggregation happened, and when.

```python
from oligomer_analysis import (
    load_cache, TargetStateConfig, PlotCalibration,
    build_lineages_from_cache, plot_whole_system_background,
)

cache = load_cache("cache_cut6.5.npz")
cal = PlotCalibration(max_oligomer_size=100, size_tick_step=10,
                      matrix_max_size=100, node_size_transform="sqrt")

lineage_result = build_lineages_from_cache(
    cache, inter_thresh=1, cluster_min_size=2, overlap_min=1,
    parent_frac_min=0.30, child_frac_min=0.30, union_cover_min=0.70,
    target_config=TargetStateConfig(min_size=10, min_duration=5, min_contribution=0.70),
)

fig, axes, df_system = plot_whole_system_background(cache, lineage_result, calibration=cal)
```

*Six panels from the reference dataset: oligomer count, largest and mean
oligomer size, contact counts, free vs bound monomers, and bound fraction —
the run aggregates from free monomers to a majority-bound system.*

The returned `df_system` DataFrame holds the same series numerically
(`build_system_summary_dataframe` builds it directly if you don't need the
figure).

## 7. Classic pathway tracking and the 9-panel dashboard

The original tracker follows individual oligomers as linear pathways — still
the best tool for a detailed look at one aggregate.

```python
from oligomer_analysis import analyze_from_cache

res = analyze_from_cache(
    cache, inter_thresh=1, size_thresh=3, continue_min_size=1,
    overlap_min=1, shrink_tolerance=3, dur_store_min=5,
)
finished = res["finished"]          # list of Pathway objects
best = max(finished, key=lambda p: (p.duration, max(p.sizes)))
print(best.label, best.duration, max(best.sizes))
# reference dataset: 1261 pathways; the longest lives 113 frames and reaches size 80
```

> **Recommended for new analyses: `follow="main_fragment"`.** The default
> (`follow="legacy"`) reproduces the historical tracker bit-for-bit — including
> a known artifact where, after a split like 10 = 8 + 2, the pathway follows
> the *small* fragment (sizes crash to 1–2 and recover; on the reference
> dataset 14.7 % of continuation steps and 83 % of pathways are affected). With
> `analyze_from_cache(..., follow="main_fragment")` each pathway follows the
> fragment sharing the most members (ties to the larger cluster): zero
> artifact events, 752 coherent pathways instead of 1,261, and the longest
> pathway lives 705 frames instead of 113.

### Size evolution of all pathways

```python
from oligomer_analysis.monitor import plot_pathway_sizes
plot_pathway_sizes(cache.frames, finished, min_duration=30, calibration=cal)
```

*Each trace is one tracked pathway (labels in short `ordinal` mode, §12).*

### Presence heatmap: which pathways exist when

```python
from oligomer_analysis import pathway_timeline_dataframe, plot_pathway_presence_heatmap_safe

df_pathways = pathway_timeline_dataframe(cache, finished, min_duration=1, label_mode="ordinal")
plot_pathway_presence_heatmap_safe(df_pathways, finished=finished,
                                   min_duration=30, max_pathways=40)
```

*The "safe" variant caps rows/columns so even hundreds of pathways cannot
produce an unreadable or memory-hungry figure.*

### The 9-panel dashboard for one oligomer

```python
from oligomer_analysis import monitor_oligomer_label

monitor_oligomer_label(cache, finished, label=best.label, inter_thresh=1,
                       calibration=cal, panel9_node_size=180)
```

*One pathway in depth: size trajectory, inter-monomer contacts, a
free-energy-style size landscape, the size transition matrix, connection
graphs at start/peak/end, compactness, MSD, and small-world network metrics.*

Single panels are available via
`plot_oligomer_panel(cache, finished, label, panel=1..9, inter_thresh=1)`,
with node-size controls `panel6_node_size=` / `panel9_node_size=`.

### Network snapshots

```python
from oligomer_analysis import plot_panel9_style_snapshot

plot_panel9_style_snapshot(cache, best.sets[-1],
                           best.start_idx + best.duration - 1,
                           inter_thresh=1, frame_is_index=True, node_size=400)
```

*The contact network of one oligomer in one frame; `plot_small_world_range`
renders a series of these across a frame window.*

### Cluster-size transition graph

```python
import numpy as np
from oligomer_analysis import build_cluster_transition_matrix, draw_transition_graph_from_prob

M = build_cluster_transition_matrix(lineage_result.frames_clusters, min_size=2)
P = M / np.where(M.sum(axis=1, keepdims=True) == 0, 1, M.sum(axis=1, keepdims=True))
draw_transition_graph_from_prob(P, thr=0.05, min_size=2, max_size=40, calibration=cal)
```

*How often clusters of size *n* become size *m* in the next frame — growth
runs up the diagonal, fragmentation below it.*

## 8. Lineage and fate analysis

The V2 layer replaces linear pathways with a **split/merge-aware graph**:

- an `OligomerSegment` is one continuous episode of a connected oligomer —
  a segment deliberately *ends* at every split or merge;
- a `LineageEvent` records each split, merge, birth, death, or rearrangement
  with full provenance (overlap matrices included);
- a `Lineage` is the connected family of segments, and **classification
  happens here**, from future fate.

```python
result = build_lineages_from_cache(
    cache,
    inter_thresh=1,          # edge rule: cmap >= 1 contact
    cluster_min_size=2,      # size 1 = monomer, not an oligomer
    overlap_min=1,           # min shared monomers to link frames
    parent_frac_min=0.30,    # min fraction of parent carried to child
    child_frac_min=0.30,     # min fraction of child coming from parent
    union_cover_min=0.70,    # split/merge grouping coverage
    target_config=TargetStateConfig(
        min_size=10,         # a "productive aggregate" is >= 10 monomers ...
        min_duration=5,      # ... persisting >= 5 frames
        min_contribution=0.70,
    ),
)

from oligomer_analysis import lineage_summary_dataframe
summary = lineage_summary_dataframe(result)   # one row per lineage
```

On the reference dataset this yields **2,165 lineages** (1,712 on-pathway,
159 off-pathway, 294 uncertain) built from 9,790 segments linked by 9,102
split/merge/birth/death events.

The classifier is a transparent rule set (auditable, not fitted): reached the
target → on-pathway; high future contribution and low trap score →
on-pathway; low contribution *and* (short-lived, or trapped, or non-growing)
→ off-pathway; otherwise uncertain.

### Lineage graph

On a system this size the full-run graph (9,790 segments) is unreadable —
zoom into the window you care about with `subset_cache_by_frames` and rebuild
the lineages there (tracking a subset is fast):

```python
from oligomer_analysis import plot_lineage_graph, subset_cache_by_frames

window = subset_cache_by_frames(cache, range(650, 801))   # the active phase
result_win = build_lineages_from_cache(
    window, inter_thresh=1, cluster_min_size=2, overlap_min=1,
    parent_frac_min=0.30, child_frac_min=0.30, union_cover_min=0.70,
    target_config=TargetStateConfig(min_size=10, min_duration=5, min_contribution=0.70),
)
plot_lineage_graph(result_win, calibration=cal, node_size_by="max_size",
                   show_labels=False)
```

*Frames 650–800 (179 lineages, 513 segments): nodes are oligomer episodes
(sized by max size, colored by fate class); edges are split/merge events,
colored by event type. The large on-pathway families sit at the centre;
short-lived off-pathway and uncertain episodes ring the outside. A lineage
that fragments and re-forms stays one connected family.*

### Fate plot

```python
from oligomer_analysis import plot_lineage_fate
plot_lineage_fate(result, calibration=cal)
```

*Size vs time for every lineage, colored by classification — the one-figure
summary of which aggregates mattered.*

For deep dives, `plot_on_off_pathway_dashboards(cache, result, inter_thresh=1)`
renders the full 9-panel dashboard for a representative on-pathway and
off-pathway lineage side by side.

## 9. Publication layer: commitment, predictors, LLPS, hotspots

This layer turns monitoring output into hypothesis tests: *can structure
predict fate?*

```python
from oligomer_analysis import (
    build_feature_table, target_state_table,
    figure_commitment_predictors, figure_llps_coupling, figure_contact_hotspots,
)

features = build_feature_table(cache, result, inter_thresh=1)  # per-segment descriptors
targets  = target_state_table(result)                          # the productive aggregates
```

`features` contains, per segment: `mean_contact_density`,
`mean_weighted_contacts`, `contact_persistence`, `exchange_fraction`,
`size_slope`, `mean_rg`, small-worldness, and (joined by default) the future
**commitment score** — the fraction of the segment's monomers that end up in
a target aggregate.

### Do structural descriptors predict commitment?

```python
figure_commitment_predictors(cache, result, inter_thresh=1)
```

*Each panel scatters one early structural descriptor against the future
commitment score — the correct inference direction for a publication claim
(fate is assigned independently, structure is tested as a predictor).*

### LLPS-style dense/dilute observables

```python
figure_llps_coupling(cache, result)
```

*Left: dense-phase fraction over time (grid-density definition from cached
monomer centres). Right: dense-phase exposure by fate class. These are
operational labels — validate against visualization and replicates.*

### Contact hotspots

```python
figure_contact_hotspots(cache, result, inter_thresh=1)
```

*Monomer-pair contacts enriched in on-pathway vs off-pathway lineages. The
default cache stores monomer-level contacts; extend Stage A if residue-level
hotspot maps are needed.*

## 10. Tables and exports

```python
from oligomer_analysis import (
    oligomer_sizes_by_frame_dataframe, oligomer_summary_by_frame_dataframe,
    save_oligomer_frame_excel, save_all_monitoring_csvs, collect_cluster_sizes,
)

df_olig    = oligomer_sizes_by_frame_dataframe(cache, result.frames_clusters, min_size=2)
df_summary = oligomer_summary_by_frame_dataframe(df_olig)
save_oligomer_frame_excel(df_olig, df_summary, "oligomer_frame_analysis.xlsx")
```

- `df_olig`: one row per oligomer per frame (frame, size, members).
- `df_summary`: per-frame aggregate statistics.
- The Excel export writes both as two sheets of one file — the format
  collaborators actually open.
- `save_all_monitoring_csvs(...)` dumps the classic monitoring series;
  `lineage_summary_dataframe`, `build_feature_table`, `target_state_table`
  are ordinary DataFrames — use `.to_csv()` / `.to_excel()` freely.

## 11. Interactive temporal networks

For exploring lineages interactively (hover, zoom, filter) in a notebook:

```python
from oligomer_analysis import (
    build_temporal_network_dataframe, plot_lineage_temporal_network,
    export_lineage_temporal_html, lineage_network_filter_panel,
)

df_net = build_temporal_network_dataframe(cache, result)
plot_lineage_temporal_network(df_net)                    # Plotly figure
export_lineage_temporal_html(df_net, "lineage_network.html")   # shareable file
```

`lineage_network_filter_panel(...)` adds ipywidgets controls (fate class,
size, frame window) and `make_lineage_temporal_animation(...)` renders an
animated version. These are Plotly-based and therefore live in the notebook /
exported HTML rather than in this static guide.

## 12. Plot calibration and display options

Large aggregates (up to ~100 monomers) break naive plots. One
`PlotCalibration` object fixes axes, ticks, matrix sizes, and network node
sizes across *every* figure:

```python
from oligomer_analysis import PlotCalibration

cal = PlotCalibration(
    max_oligomer_size=100,   # size axes end here
    size_tick_step=10,
    matrix_max_size=100,     # transition-matrix display cap
    node_size_min=10, node_size_max=120,
    node_size_transform="sqrt",   # node area ~ sqrt(size): big nodes can't dominate
    pathway_label_mode="ordinal", # "oli8" instead of "oli8_16_20_25_26"
)
```

Pass `calibration=cal` to any plotting function. Related controls:

- **Label modes**: pathway labels store their founding members
  (`oli8_16_20_25_26`); display them short with `mode="ordinal"` globally
  (via calibration) or per plot (`label_mode="ordinal"`). Full labels remain
  the internal identifiers either way. `format_oligomer_label` /
  `display_oligomer_label` convert explicitly.
- **Node sizes**: `panel6_node_size=` and `panel9_node_size=` on the
  dashboard and panel functions.
- **Legends**: `show_legend=False` (or via calibration) for crowded figures.

## 13. The one-command HTML report

Everything in sections 6–10, generated automatically as **one self-contained
HTML file** (all figures embedded as images; no external resources; prints to
PDF from any browser):

```bash
coana report cache_cut6.5.npz -o report.html \
    --title "128-mer oligomer aggregation" --max-size 100 \
    --inter-thresh 1 --min-size 10 --min-duration 5 --min-contribution 0.70
```

```python
from oligomer_analysis import generate_html_report
generate_html_report("cache_cut6.5.npz", "report.html",
                     title="128-mer oligomer aggregation")
```

The report contains: headline stat tiles, the full run configuration and
methods with your actual parameters, system evolution figures, oligomer
population statistics and tables, contact maps, the lineage/fate analysis,
reproducibility metadata (including the exact regeneration command), and an
engine-capabilities appendix. `--no-lineage` skips the slow lineage build for
a quick population-only report. Figures render headless (Agg), so the report
can be generated inside a batch job. See `CoAnaPure_Analysis_Report.html` in
the project root for a real example (1956 frames, 1.7 MB, one file).

### The interactive lineage timeline

The report's lineage graph uses a force-directed layout: good for topology,
useless for chronology. `coana timeline` renders the same segments and events
on a **real frame axis**, so splits and merges read left to right in the order
they happened, and inlines the result into one self-contained page:

```bash
coana timeline cache_cut6.5.npz -o TIMELINE.html \
    --frame-lo 640 --frame-hi 820 \
    --inter-thresh 1 --min-size 10 --min-duration 5
```

```python
from oligomer_analysis import generate_lineage_timeline
generate_lineage_timeline("cache_cut6.5.npz", "TIMELINE.html", frame_lo=640, frame_hi=820)
```

The lineage flags match `build_lineages_from_cache` and the `--min-*` flags
define the target state that drives the on/off-pathway colouring, so a system
unlike the 128-monomer reference run is a matter of flags, not code. Cost grows
with the number of *segments*, not frames, so window a long trajectory to keep
the page responsive. `LINEAGE_TIMELINE.html` in the project root is a real
example (1956 frames, 9790 segments, 4.1 MB); `docs/lineage_timeline/README.md`
documents the page's marks and controls.

## 14. Running on an HPC cluster

A minimal batch script:

```bash
#SBATCH --cpus-per-task=16 --mem=8G --time=01:00:00
coana precompute --traj traj.pdb --topology topo.pdb --frames all \
    --cutoff 6.5 --intra-exclude 2 \
    --segment-mode atom_blocks --atoms-per-monomer 83 --n-monomers 128 \
    --n-jobs "$SLURM_CPUS_PER_TASK" --parallel process \
    --checkpoint-every 200 --resume -o cache.npz
coana report cache.npz -o report.html
```

Why this works well on a cluster: constant-memory streaming (the 1.66 GB
reference runs in <0.5 GB serial), process-parallel parsing that scales with
the node's cores, checkpoint/resume so walltime kills cost nothing (resubmit
the same script), progress lines in the SLURM log, and a single-file report
you `rsync` home.

## 15. Validation and reliability

- **Cross-validated against MDAnalysis 2.9**: contact maps, inter-monomer
  contacts, free/bound counts, frames and boxes match *exactly*; segment
  centres to ~1e-4 Å; `nintra`/`ntotal` within ±1 on a few frames (documented
  float32-vs-float64 rounding at the cutoff — CoAnaPure computes in float64
  and is the more accurate side). Re-run the comparison on your own data any
  time: `coana benchmark --traj traj.pdb --frames 0:250` (needs the
  `mdanalysis` extra).
- **Byte-identity across execution modes**: serial, threaded,
  process-parallel, indexed, checkpoint-resumed — all produce byte-identical
  caches, verified on the real production trajectory.
- **62 automated tests** (`python -m pytest tests -q`), including brute-force
  neighbour-search references, XTC decoding vs MDAnalysis-written files, and
  a synthetic split/merge lineage scenario with known ground truth.
- Full measured performance history: engine report §6.

## 16. Troubleshooting and FAQ

**"My 128-monomer system shows only ~63 segments."**
Chain-ID reuse — use `segment_mode="atom_blocks"` (§4.2).

**"A warning says the analysis runs WITHOUT periodic boundaries."**
Your multi-model PDB has one leading `CRYST1`, which the legacy-compatible
default drops (§4.4). Pass `broadcast_box=True` for correct PBC on new work.

**"`parallel='process'` crashes on my Windows machine."**
Put the call under `if __name__ == "__main__":` — the error message says
exactly this. Notebooks and the CLI are unaffected; Linux is unaffected.

**"XTC reading is slow."**
Expected — the decompressor is pure Python by design (zero dependencies).
Convert once with `gmx trjconv -f traj.xtc -o traj.trr` for repeated use.

**"My batch job hit the walltime."**
Run with `--checkpoint-every 200 --resume` and simply resubmit; the job
continues where it stopped (§4.6).

**"Frames don't match what I asked for."**
Frame specs are Python slices — `0:250` means frames 0..249. `coana info`
shows what a cache actually contains.

**"The old MDAnalysis engine prints garbage / crashes on Windows."**
That engine prints Unicode glyphs; run it with `PYTHONIOENCODING=utf-8`.
CoAnaPure and `coana` print plain ASCII.

**"Can I trust a resumed / parallel cache?"**
Yes — checkpoints are configuration-fingerprinted and every execution mode is
byte-identical (§15).

## 17. Mixed systems and mechanism (v3.4)

Version 3.4 turns the toolkit from a single-component aggregation tracker into
one that can answer the questions a **co-aggregation** study actually asks:
which species is inside the assembly, which residues touch which, whether the
minority component is *part of* the aggregate or merely *helps it form*, and
whether any of those numbers are converged and comparable across runs. Every
feature is reachable both from Python and from a `coana` subcommand, and every
one was built against a synthetic system whose answer is known by construction
(`tests/test_v34_features.py`).

### 17.1 Tell Stage A the composition once: `--species`

```bash
coana precompute --traj prod.xtc --topology start.gro --frames all --cutoff 6.5 \
    --species "aSyn:295x50,Ab40:92x10" --contact-classes --hotspot-blocks 5 \
    --n-jobs -1 --parallel process -o cache.npz
```

`--species` is `name:beads_per_chain x n_chains`, comma-separated, **in
topology order**. It sets the monomer block sizes (so `--segment-mode` and
`--monomer-atom-counts` are no longer needed for mixtures), switches to
`atom_counts` segmentation automatically (Martini PDBs reuse chain ids after
~62 chains, so chain-id segmentation is unsafe for large boxes) and records the
species in the cache. From then on nothing downstream has to be told the
composition again: `SpeciesMap.from_cache(cache)` recovers names and counts,
and two species of equal size can no longer be confused.

Two optional extras are computed while the residue-pair list is still in
scope, and **cannot be added to a finished cache** (residue identity is gone
once contacts are binned to chains):

| Option | Stores | Cost |
|---|---|---|
| `--contact-classes` | per frame, a 6×6 matrix of inter-chain contacts by residue chemistry: `apolar`, `aromatic`, `polar`, `cation`, `anion`, `other` | negligible |
| `--hotspot-blocks N` | a `(N, S, S, R, R)` residue×residue contact tensor between species, summed in `N` time blocks | ~`N·S²·R²·4` bytes, e.g. 5 blocks × 295² × 4 species-pairs ≈ 7 MB; folded as frames arrive, never held per frame |

`coana info` shows the species line and an `extras :` line naming what the
cache carries.

### 17.2 Where does each species sit? `coana partition`

```bash
coana partition cache.npz --last-fraction 0.25 -o partition.csv
```

Reports, per species, the **partition coefficient** K = (mole fraction inside
the largest assembly) / (mole fraction in the box), with SD and SEM over the
late window, and the free-chain count and concentration. K > 1 means enriched
in the assembly. This is the quantity a recruitment/co-localisation experiment
reports, so it is the number a simulation should quote beside one.

The free-chain table also carries `dilute_phase`: `False` when fewer than ~5 %
of a species' chains are ever free. In that regime a saturation concentration
is noise (its SD exceeds its mean) and the tool says so rather than returning
a c_sat that looks like a measurement.

Python: `partition_coefficients`, `partition_summary`,
`free_monomer_concentration`.

### 17.3 What holds the assembly together? `coana chemistry`

```bash
coana chemistry cache.npz --last-fraction 0.25 -o chemistry.csv
```

Needs `--contact-classes` at Stage A. Prints the inter-chain contact budget by
interaction type, each pair labelled by its physical channel (`salt bridge`,
`hydrophobic`, `cation-pi`, `polar/charged`, `like-charge repulsive contact`),
with the late-window mean and its share of all contacts; `fraction_late` sums
to 1. Two derived series are printed directly: the salt-bridge count and the
hydrophobic share.

This is how an electrostatics conclusion is *shown* rather than inferred: if
salt suppresses aggregation, the salt-bridge channel should be the one that
shrinks. Python: `contact_class_summary`, `contact_class_dataframe`,
`salt_bridge_timeseries`, `hydrophobic_fraction_timeseries`.

### 17.4 Which residues bind which? `coana hotspots`

```bash
coana hotspots cache.npz --species-a aSyn --species-b Ab40 \
    --regions "N-term:1-60,NAC:61-95,C-term:96-140" --top 20 -o hotspots.csv
```

Needs `--hotspot-blocks` at Stage A. Gives the residue-level cross-species
contact matrix, the per-residue profile (1-based `resid`, ready for a sequence
plot), the top residue pairs, and — with `--regions` — a region table whose
`enrichment` is the region's share of the interface divided by its share of
the sequence, so "the NAC region drives binding" is separated from "the NAC
region is simply long". The per-block table shows an interface forming or
shifting over time.

Python: `residue_hotspot_matrix`, `residue_hotspot_profile`,
`top_residue_pairs`, `region_contact_table`, `hotspot_block_dataframe`.

### 17.5 Is the run converged? `coana converged`

```bash
coana converged cache.npz --all -o convergence.csv     # exit status 1 if not
```

Two independent tests must both pass: a Geweke comparison of the first 10 %
and last 50 % of frames using autocorrelation-corrected standard errors, and
a late-window relative drift below 10 %. The table also reports the integrated
autocorrelation time, the effective sample count and both the naive and the
corrected SEM — over 1000 correlated frames the naive error can be ten times
too small. A non-converged run is not hidden: the mean is printed with the
instruction to quote it as a lower bound or extend the run.

Python: `assess_convergence`, `convergence_dataframe`,
`integrated_autocorrelation_time`.

### 17.6 Lineages that know their composition: `coana lineage`

```bash
coana lineage cache.npz --min-size 10 --min-duration 5 \
    --lineage-min-duration 50 --sweep 6,8,10,12,15 -o lineages.csv
```

The lineage table now carries `n_<species>`, `peak_<species>`,
`n_species_present`, `is_hetero` and `hetero_fraction` whenever a species map
is available, so the central question of a co-aggregation study — *are the
on-pathway lineages the mixed ones?* — is one line of the output
(`composition vs fate`). One-component tables keep their original shape.

Three guardrails, each tied to a misreading that has actually happened:

* **Warnings** are printed automatically when the tracker is counting
  transient dimers (more lineages than chains), when the target size cannot be
  reached in this box, when no lineage reaches it, or when the median lineage
  lasts ≤ 2 frames. The same list is on `result.warnings`.
* **`--lineage-min-duration N`** keeps only lineages lasting ≥ N frames and
  returns a self-consistent result (segments and events filtered with them).
  With the default `--cluster-min-size 2` a 50-chain box produces thousands of
  two-frame dimers; a persistence floor is usually the difference between raw
  output and a result.
* **`--sweep`** re-classifies against several target sizes and prints a
  verdict: *stable*, *varies monotonically* (quote the trend), or
  *non-monotonic and wide* (the target size is driving the result).

Python: `build_lineages_from_cache(..., species_map=smap)`,
`attach_species_compositions`, `lineage_warnings`, `filter_lineages`,
`target_size_sweep`, `sweep_verdict`.

### 17.7 Incorporation or catalysis? `coana mechanism`

```bash
coana mechanism cache.npz --guest Ab40 --commit-size 10 --after-window 20 \
    --frame-interval-ns 10 --min-size 10 --min-duration 5 -o residence.csv
```

Follows every chain of the *guest* species through the host assemblies it
visits and classifies each visit by its relationship to the host's
**commitment**. Since 3.5.0, commitment is judged on the guest's *own* assembly,
followed by chain identity, and requires `--commit-size` host chains (by default
the lineage target size) held for `--commit-duration` consecutive frames (by
default `--min-duration`). The 3.4.0 rule, which used the whole split/merge
lineage and could hand a guest another branch's commitment, is still available
as `--mode lineage` for comparison; see CHANGELOG.md.

| Class | Meaning |
|---|---|
| `incorporated` | present at commitment and still there at the end — part of the product |
| `catalytic` | present at commitment, then left while the assembly held or kept growing — promotes formation without joining the product |
| `passenger` | joined an assembly that had already committed — recruited by the aggregate, not contributing to its formation |
| `promoting` | the assembly grew during the visit but never committed |
| `transient` | present at commitment but the assembly shrank after departure |
| `bystander` | no commitment and no growth — an encounter, not a mechanism |

The summary gives both `visit_share` (each visit weighted equally) and
`frame_share` (weighted by residence time); reporting only the former makes
bystanders look like the dominant mechanism. Exchange statistics (visits per
chain, mean residence, bound fraction, exchange rate per chain per µs) follow.

Running it with the majority species as guest is **not** a fair control: those
chains are then judged against an assembly of the minority species, a different
question. The fair control designates a random subset of host chains (as many as
there are guest chains) as pseudo-guests and keeps the rest as host, so guest and
pseudo-guest are judged against the same assembly. Use several random draws.

Pass `--lineage-min-duration` (e.g. 50) here as everywhere else: since 3.5.1 the
filter removes short-lived assemblies from the classification itself, not just
from the output.

Python: `residence_events`, `residence_dataframe`, `mechanism_summary`,
`exchange_statistics`.

### 17.8 Many runs, one table: `coana compare`

```bash
coana compare runs/*/cache.npz \
    --descriptor-regex "runs/(?P<cond>[0-9.]+[a-z]+)_(?P<ff>[a-z]+)/" \
    --group-by cond --min-size 10 --min-duration 5 --lineage-min-duration 50 \
    -o campaign.csv
```

One row per cache: identity, late observables, convergence verdict, per-species
bound fractions and K, lineage counts, `on_fraction`, and the species×lineage
join (`on_fraction_hetero` vs `on_fraction_pure`). Descriptors come from a
named-group regex applied to the **path**; when all caches share a file name
the label becomes `directory/stem`.

With `--group-by`, `-o` receives the grouped table and the per-run rows are
written beside it as `<stem>_runs.csv` — a grouped mean is only defensible next
to the rows behind it.

`--group-by` aggregates (mean, SD, SEM, n) and — critically — reports
`mixed_columns`: descriptors that vary *within* a group but are not in
`--group-by`. This is the direct guard against pooling runs of two force-field
variants under one condition label: the terminal prints an explicit warning
naming the offending column. `--no-lineage` makes a campaign-wide sweep fast;
`--skip-errors` records unreadable caches instead of stopping.

Python: `summarize_cache`, `compare_caches`, `group_summary`.

### 17.9 Networks for Gephi: `coana export`

```bash
coana export cache.npz --what both --composition-states --max-state-size 15 \
    --min-size 10 --out-dir gexf/
```

Writes two GEXF 1.2 files (no library dependency): the lineage graph (one node
per lineage, edges for split/merge events) and the coarse state-transition
network whose states are cluster sizes, or size + composition with
`--composition-states` (`"12(10aSyn+2Ab40)"`). `--max-state-size N` collapses
everything at or above N into one absorbing state, which keeps the network
readable when a run ends in one large aggregate. A `solution` pseudo-state
absorbs dissolving clusters and emits appearing ones so the matrix
row-normalises properly; unvisited states are left as zero rows, never uniform
— an unvisited state has no measured dynamics. The transition matrix is also
written as CSV.

Python: `state_labels_by_frame`, `transition_counts`, `transition_matrix`,
`transition_dataframe`, `write_lineage_gexf`, `write_transition_gexf`.

### 17.10 Speed

Per-frame clustering now uses `scipy.sparse.csgraph.connected_components`
with the original component ordering preserved (a pure-Python fallback is
kept and tested against it). Building lineages for a 2000-frame, 60-chain
cache dropped from ~80 s to a few seconds, which is what makes the target-size
sweep and campaign-wide comparison practical.

---

## 18. Appendix: API index

Everything importable from `oligomer_analysis` (`from oligomer_analysis import *`):

| Area | Names |
|---|---|
| Stage A | `PrecomputeConfig`, `precompute_cache_pure`, `precompute_cache` (MDAnalysis extra), `build_model_index`, `pdb_chain_id_summary` |
| Cache | `load_cache`, `validate_cache`, `save_contact_cache`, `subset_cache_by_frames`, `subset_cache_by_indices` |
| Whole system | `build_system_summary_dataframe`, `collect_cluster_sizes`, `filter_clusters_by_size`, `build_cluster_transition_matrix`, `clusters_from_cmap` (graph_tools) |
| Classic tracker | `analyze_from_cache`, `monitor_oligomer_label`, `plot_oligomer_panel`, `plot_panel9_style_snapshot`, `plot_small_world_range`, `pathway_timeline_dataframe`, `plot_pathway_presence_heatmap_safe`, `draw_transition_graph_from_prob` |
| Lineages | `build_lineages_from_cache`, `TargetStateConfig`, `lineage_summary_dataframe`, `representative_pathways_for_lineages`, `plot_whole_system_background`, `plot_lineage_graph`, `plot_lineage_fate`, `plot_on_off_pathway_dashboards` |
| Publication | `build_feature_table`, `compute_segment_features`, `compute_lineage_features`, `find_target_segments`, `target_state_table`, `compute_segment_commitment_scores`, `compute_lineage_commitment_scores`, `figure_commitment_predictors`, `figure_llps_coupling`, `figure_contact_hotspots`, `figure_system_overview`, `figure_lineage_overview`, `dense_phase_fraction_timeseries`, `radial_density_profile`, `lineage_dense_phase_table`, `rank_contact_hotspots`, `contact_frequency_by_fate`, `contact_enrichment_on_vs_off` |
| Exports | `oligomer_sizes_by_frame_dataframe`, `oligomer_summary_by_frame_dataframe`, `save_oligomer_frame_excel`, `save_all_monitoring_csvs` |
| Interactive | `build_temporal_network_dataframe`, `plot_lineage_temporal_network`, `make_lineage_temporal_animation`, `export_lineage_temporal_html`, `lineage_network_filter_panel` |
| Display | `PlotCalibration`, `format_oligomer_label`, `display_oligomer_label`, `legend_enabled` |
| Report | `generate_html_report` |
| Debug | `debug_show_big_clusters`, `debug_cmap_pairs`, `debug_row_sums`, `debug_edge_changes`, `debug_edge_changes_focus`, `debug_pathway`, `debug_cluster_overlap`, `debug_frame_transition`, `debug_lineage_summary`, `debug_lineage_events` |
| Species (v3.4) | `parse_species_spec`, `monomer_atom_counts_from_spec`, `SpeciesMap`, `partition_coefficients`, `partition_summary`, `free_monomer_concentration` |
| Chemistry (v3.4) | `CONTACT_CLASSES`, `classify_residues`, `contact_class_census`, `describe_contact_classes`, `contact_class_summary`, `contact_class_dataframe`, `salt_bridge_timeseries`, `hydrophobic_fraction_timeseries` |
| Residue hotspots (v3.4) | `residue_hotspot_matrix`, `residue_hotspot_profile`, `top_residue_pairs`, `region_contact_table`, `hotspot_block_dataframe` |
| Convergence (v3.4) | `assess_convergence`, `convergence_dataframe`, `integrated_autocorrelation_time` |
| Lineages (v3.4) | `attach_species_compositions`, `lineage_warnings`, `filter_lineages`, `target_size_sweep`, `sweep_verdict` |
| Mechanism (v3.4) | `residence_events`, `residence_dataframe`, `mechanism_summary`, `exchange_statistics` |
| Comparison (v3.4) | `summarize_cache`, `compare_caches`, `group_summary` |
| Networks (v3.4) | `state_labels_by_frame`, `transition_counts`, `transition_matrix`, `transition_dataframe`, `write_lineage_gexf`, `write_transition_gexf` |
| Models | `ContactCache`, `FrameContacts`, `Pathway`, `OligomerSegment`, `LineageEvent`, `Lineage`, `LineageTrackingResult` |

CLI: `coana precompute | index | info | report | timeline | benchmark | species | validate | partition | chemistry | hotspots | converged | lineage | mechanism | compare | export | version`
(also `python -m oligomer_analysis`).

---

*Written for oligomer-analysis 3.5.1 / CoAnaPure 1.3; the command line and API
are unchanged in coana 3.6.0.*
