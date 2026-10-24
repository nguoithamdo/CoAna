# Changelog

## 3.6.0 — 2026-10-23

Merges a correctness audit of the 3.4.0 analysis layer (sixteen defects, each
with regression tests proven to fail without its fix) onto 3.5.1, and licenses
the package under MIT. Branch-local mechanism and directed fates from 3.5.x are
unchanged.

### Renamed
- The distribution is now **`coana`** (`pip install coana`), and `import coana`
  is the canonical import. `import oligomer_analysis` keeps working: both names
  refer to the same module objects, so objects made through one pass
  `isinstance` checks written against the other. The CLI banner and report
  footer read "coana" / "CoAna".

### Recompute these
- **Any cache built from a trajectory without a periodic box.** Stage A ran
  without periodic boundaries and said nothing. With whole molecules this
  misses every contact across a box face: 10.3 % of chain-chain contacts and a
  largest cluster of 54.7 instead of 66.4 chains on average, on a 128-chain
  Martini run. With wrapped coordinates it also creates fake
  contacts (largest cluster 26 instead of 13). `meta["has_box"]` now records it.
- **Frame 0 of a GROMACS-written PDB** (CRYST1 before every MODEL): its box was
  dropped, so frame 0 alone ran without periodic boundaries.
- **Dense-phase / LLPS numbers.** `dense_phase_fraction_timeseries`,
  `lineage_dense_phase_table` and both panels of `figure_llps_coupling` returned
  exactly 1.0 for every frame of every trajectory. Any earlier value carried no
  information.
- **Contact persistence** (`compute_segment_features`, `build_feature_table`,
  `figure_commitment_predictors`). It was divided by segment duration instead of
  co-residency, which made it anti-correlated with growth by construction.
- **Rg and MSD across periodic boundaries.** Without MDAnalysis installed no
  minimum image was applied at all; with it, the oligomer centre and the Rg
  centroid were taken from wrapped coordinates (Rg 49.0 instead of 1.0 for a
  dimer astride a boundary). Caches without a box are unaffected by this fix —
  but they had no periodic boundaries to begin with.
- **Partition-coefficient error bars.** `K_sem` was a naive sd/sqrt(n) over
  correlated frames, about 10x too small at an autocorrelation time of 100 frames.
- **Salt-bridge counts**, only if your topology uses protonation-variant names
  (ASPP, GLUP, ARGN, ...): neutral residues were counted as charged.

### Added
- `calibrate_dense_phase` / `DensePhaseCalibration`: one dense-phase rule shared
  by every consumer, with the smoothing bandwidth in Angstroms and a
  `relative_to_reference` column marking quantile-calibrated (relative) results.
- `gather_cluster_images`: puts a cluster into one periodic image along its own
  contact graph, independent of member order.
- Free-energy profiles carry `F_stderr` (and the size trace's autocorrelation
  time); `min_count` drops sizes seen too rarely to estimate; panel 3 of the
  monitor dashboard draws error bars.
- Partition summary columns `K_sem_naive`, `K_tau`, `K_n_eff`, `K_n_dropped`.
- Residence tables: `later_commit_idx`, `later_commit_frame`,
  `departure_to_commit_lag` — whether the guest's own assembly committed only
  after the guest left. Data for an ensemble test against pseudo-guests, not a
  label.
- Mechanism label `pre-commitment` in `mode="lineage"`: a guest that left before
  its lineage committed, with growth during the visit. Branch mode keeps its
  3.5.0 rule (OPEN-01).
- `classify_residues` warns, naming them, when residues fall into "other".

### Fixed
- Chemistry: neutral protonation variants classified as charged; `normalize=True`
  divided by the full symmetric sum (fractions summed to 0.5).
- Compare: numeric descriptors were never checked for pooling; descriptor names
  could silently overwrite measurements; `last_fraction` was unvalidated.
- CLI: `coana benchmark --help` exited with "unrecognized arguments".
- Network: empty transition tables raised `KeyError`; quotes in names broke GEXF.
- Convergence: overlapping Geweke windows were accepted, biasing z toward 0.
- Mechanism: two reason strings asserted facts a visit-scoped rule cannot see
  ("never reached N", "no commitment").
- Lineage: notes claimed persistence was a fate criterion; it never was.
  `_classify_lineage` no longer accepts the unused argument; classifications are
  provably unchanged.

### Changed — may break scripts
- `coana precompute` refuses a trajectory without a periodic box (exit 2).
  Supply the box (`--broadcast-box`, `--fallback-box`) or, for a system that
  really has no periodic boundaries, pass `--allow-no-box`. The Python API
  warns with `NoBoxWarning` instead of refusing.
- GEXF node attribute `occupancy` renamed `n_departures` (it counts departures).
- `assess_convergence` raises on overlapping windows; `summarize_cache` raises on
  an out-of-range `last_fraction` or a descriptor named after a measurement.
- The dense-phase threshold is calibrated once per series; pass
  `dense_threshold` for an absolute rule that is comparable between runs.

### License
MIT. Before this release the package carried no license.

## 3.5.1 — 2026-10-09

Fixes four defects in the 3.5.0 branch-local mechanism and directed fates, found
by an independent review and
reproduced on synthetic caches. Every 3.5.0 guest-role and directed-fate number
should be recomputed.

### Fixed
- **Commitment leaked between the halves of a split.** Sustained-commitment runs
  were shared through each assembly's predecessor, so after a split both halves
  carried one run: a guest on a half that shrank was labelled `passenger` because
  the *other* half stayed big. Commitment is now per assembly, along its own
  main-predecessor chain.
- **A dip below `commit_size` erased commitment.** An assembly that committed,
  dipped below the threshold and recovered started a new run, so a guest joining
  during the dip was credited with the recommitment (`catalytic`). Commitment is
  now irreversible along the assembly's own history; that guest is `passenger`.
- **A hop carried by a second guest was one visit.** Visit continuity used any
  shared chain, so two guests moving together into an unrelated assembly kept one
  visit. Continuity now needs a shared *host* chain whenever both assemblies hold
  hosts.
- **Directed fates used invented parent/child edges.** Every parent of a
  split/merge event was linked to every child (the Cartesian product), so a
  parent that fed only a dead-end child was scored `on-pathway` through a sibling
  it never touched. `parent_ids` / `child_ids` now follow the actual overlaps.
  Lineage membership (the undirected label) is unchanged.
- **`coana mechanism` ignored `--lineage-min-duration`**, and the mechanism
  classifier read every cluster in the trajectory even when given a filtered
  result. The filter is now applied, and assemblies of filtered-out lineages are
  removed from the classification itself.
- **`commit_frame` was an index.** Residence tables now carry `commit_idx` and
  the real frame number `commit_frame`.
- Docstring, CLI help and GUIDE no longer call the majority-species-as-guest run a
  control (it judges a different assembly); they describe the pseudo-guest control.

### Known limitation
Equal host overlaps between two successor candidates are still broken by total
cluster size.

### Tests
`tests/test_v351_regressions.py`: 8 cases with hand-worked answers, each paired
with a control; all 8 fail on 3.5.0.

### Correction to the 3.5.0 entry
The campaign numbers under "Effect on the alpha-synuclein campaign" below are
smoke-test values written before the full re-run and are superseded; the 3.5.0
re-run itself is superseded by the 3.5.1 re-run.

## 3.5.0 — 2026-10-06

### Fixed: guest roles were judged on the wrong assembly
`residence_events` / `coana mechanism` judged commitment, survival and
post-departure growth on the whole lineage. A lineage is every segment ever
linked by any split or merge, in both directions, including clusters that merge
in only later. A guest on a small assembly that never grew was therefore
labelled `passenger` or `catalytic` when a *different* branch had committed, or
when its own cluster merged into that branch after the guest had left.

The default rule is now `mode="branch"`:
- **Own-assembly tracks.** Every rule is judged on the guest's own assembly,
  followed through time by chain identity (successor and predecessor = the
  cluster holding most of its host chains).
- **Sustained commitment.** An assembly commits only when it holds
  `commit_size` host chains for `commit_duration` consecutive frames (default:
  the target's `min_duration`), so a one-frame touch of the threshold is not a
  commitment. The commit frame is the first frame of that run.
- **Hops start new visits.** A visit is cut when the guest hops directly into an
  unrelated assembly.

The 3.4.0 rule remains available as `mode="lineage"` / `--mode lineage`. On
real data it reproduces the 3.4.0 residence tables exactly.

Effect on the alpha-synuclein campaign (production PEG, share of bound frames):
- Ab40 passenger 28.6% → ~12.6%, incorporated 10.3% → ~21.7%.
- The catalytic class is essentially unchanged.
- The host-species control is unchanged.

### Added: directed fates
`directed_fates(result)` and `directed_fate_summary(df, min_span=...)`, plus
`coana lineage --directed-out CSV`, score every nucleation event (a segment with
no parent) by whether a target state is reachable *forward* through its own
descendants. The undirected lineage label is unchanged and still reported. It
groups shed fragments and late mergers with the product, so its on-pathway
fraction answers a looser question.

### Tests
`tests/test_v35_branch_local.py` adds 10 cases with hand-worked answers. Three of
them fail on the 3.4.0 rule, as intended.

## 3.4.0 — 2026-09-03
Species-aware Stage A; `partition`, `chemistry`, `hotspots`, `converged`,
`lineage`, `mechanism`, `compare` and `export` subcommands. See GUIDE.md §17.
