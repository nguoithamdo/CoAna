# The history of CoAna

**CoAna**, the *Coupled Analysis Framework for Quantifying Biomolecular
Assembly and Condensation Dynamics*, has been developed since **January 2024**
by **Dr. Nguyen Truong Co** (Truong Co Group). It began as a single analysis notebook for one question:
in a molecular dynamics simulation of aggregating proteins, which of the many
small oligomers actually lead to the final aggregate? It grew into a tested,
documented Python package that reads trajectories directly and answers that
question for mixed, crowded systems as well.

This page records how it got there. Dates before September 2026 come from the
surviving source files and notebooks and from theses that used CoAna; from
3.4.0 on they come from the changelog and version control.

---

## 2024–2025 · The notebook years

CoAna started as research code inside Dr. Co's own studies of protein
aggregation with coarse-grained (Martini) molecular dynamics. Every analysis
lived in a Jupyter notebook: load a trajectory, build monomer–monomer contact
maps frame by frame, find the connected clusters, and follow them through time.

Two ideas from this period still shape CoAna today:

- **Follow oligomers, not just count them.** Size histograms say how many
  dimers or decamers exist; they do not say which of them went on to build
  the aggregate. Tracking each cluster from frame to frame was the core of the
  notebook from the start.
- **Analysis must be repeatable from a saved state.** Re-reading a
  multi-gigabyte trajectory for every new plot was too slow, which later
  became the two-stage design.

By December 2025 CoAna was a working research tool, built in Python on top of
MDAnalysis, and two theses had applied it:

- **A bachelor's thesis (2025)** in materials science on the
  pre-condensation of α-synuclein, the protein behind Parkinson's disease, ran
  8 µs of Martini 3 molecular dynamics on 30 α-synuclein chains in a 25 nm box
  at 300 K and analysed it with CoAna. CoAna followed every oligomer through
  the run (more than 25 of at least three chains lived longer than 50 ns; the
  longest lived 3.1 µs), counted the residue contacts inside each oligomer
  over time, and gave free-energy surfaces and size-transition networks not
  only for the whole system but for **one chosen oligomer**, here the two
  longest-lived ones. Looking inside one oligomer, not only at the system
  average, has been part of CoAna since.
- **A master's thesis (December 2025)** in
  theoretical physics on liquid–liquid phase separation of α-synuclein used
  Dr. Co's CoAna for its whole analysis: how box size and temperature (290–330 K) change
  oligomerisation. CoAna supplied contact counts, oligomer numbers, sizes and
  structure, the lifetime of every oligomer, free-energy surfaces, contact
  maps, size-transition matrices and size-transition networks for single
  oligomers.
- The analysis notebook (`oligomer_monitor_manuscript_330K`) was used for
  manuscript figures of a 330 K aggregation run.

## February 2026 · From notebook to package

The notebook was refactored into a package, **Oligomer Tracker**: separate
modules for contacts, clustering, tracking and plotting, a command-line entry
point, configuration files and the first unit tests. The development notebooks
of this month (`tracking_oligomer_dev`, `trackingoligomer_new`) worked on a
700-frame, 330 K trajectory that later became a standard test case.

## March–April 2026 · The two-stage design (CoAna v1, v2)

The decisive architectural step was to split the work in two:

- **Stage A**, run once: read the trajectory, compute every contact map with
  periodic boundaries, and save them to a compact **contact cache** (`.npz`);
- **Stage B**, run as often as needed: clustering, tracking, statistics and
  figures, all from the cache in seconds.

This appears in the `v1_oligomer_cached_full_notebook` series (March 2026) and
in the packaged workflow notebooks **CoAna v1** and **CoAna v2**
("Oligomer Analysis Professional Workflow").

## May–June 2026 · oligomer-analysis 3.x, and a second field

**Third thesis to use CoAna (June 2026).** A bachelor's thesis in pharmacy on the
pre-aggregation of islet amyloid polypeptide (IAPP), the peptide whose
aggregation kills β-cells in type 2 diabetes, analysed a 128-monomer Martini 3
system (5 µs at 300 K) with Dr. Co's CoAna. Contacts were defined between residue
centres within 6.5 Å, an oligomer had to persist for at least 5 frames, and
CoAna traced oligomer sizes, free and bound monomers, lifetimes, merging and
splitting, free-energy surfaces and size-transition networks. The thesis
describes CoAna as applicable beyond proteins and peptides: to DNA, RNA,
protein–RNA and protein–DNA complexes and other biomolecular condensates.

In parallel, the code became an installable package, `oligomer-analysis`:

- **3.0**: one package with precomputation, atom-block segmentation of
  monomers, the classic pathway tracker, whole-system plots, and a complete
  feature-demo notebook.
- **3.1 / 3.1.2**: helpers for large analyses (frame subsets, Excel export,
  pathway-presence heatmaps), node-size and legend controls for the network
  panels, and a fix for large PDB files that reuse chain IDs.
- **The split/merge-aware lineage layer.** Clusters split and merge, so a
  single "pathway" is the wrong model. Lineages follow every split and merge
  and are classified by **fate**, not size: a small oligomer is on-pathway if
  it feeds a later productive aggregate; a large one is off-pathway if it is a
  trap that never grows.

## July–August 2026 · CoAnaPure: no MDAnalysis needed

- **CoAnaPure**, a new Stage A engine written with NumPy and SciPy only. It
  reads PDB, GRO, TRR and XTC files itself, streams the trajectory at constant
  memory, uses all CPU cores, and checkpoints so it survives cluster walltime
  limits. Version 1.0 was validated field by field against the original
  MDAnalysis engine; 1.1 added parallel frames and random access with
  byte-identical output; 1.2 (August 2026) became the default engine.
- **3.3.0** (24 August 2026): a diagnosis of the classic tracker showed it
  could follow the *small* fragment after a split (10 → 8 + 2, recording the
  2). The new `follow="main_fragment"` mode follows the fragment that keeps the
  most members; the old behaviour stays available bit for bit.
- **The one-command HTML report** and the interactive **lineage timeline**.
- A movie tool for aggregation trajectories (CoAna2-movie 1.0) and the start
  of **coanala**, a sibling tool that applies the same analysis to lattice
  models of protein aggregation.

## September 2026 · 3.4.0: mixed and crowded systems

Real aggregation rarely involves one protein. **3.4.0** (3 September 2026)
made every step species-aware: declare the composition once
(`--species "aSyn:295x50,Ab40:92x10"`) and CoAna knows which chain is which.
New commands: `partition` (is a species enriched in the assembly?),
`chemistry` (which interactions hold it together?), `hotspots` (which residues
bind which?), `converged` (has the observable reached steady state?),
`lineage`, `compare`, `export` (for Gephi), and **`mechanism`**: what did a
guest peptide actually do while it was inside an assembly? Incorporated,
catalytic, passenger, promoting, or bystander.

## October 2026 · Getting it right: 3.5 and 3.6

- **3.5.0** (6 October): guest roles are judged on the guest's *own* branch of
  a lineage. Judging them on the whole lineage had credited guests with
  commitments that happened on a different branch. Directed fates added.
- **3.5.1** (9 October): four further defects in the mechanism and fates,
  found by an independent review, fixed and reproduced on synthetic data.
- **3.6.0** (23–24 October): a full correctness audit of the analysis layer.
  Sixteen defects were fixed, each with a regression test proven to fail on
  the old code. The package was renamed **`coana`** (`pip install coana`,
  `import coana`; `import oligomer_analysis` still works), licensed under
  **MIT** (relicensed with every other version under the LGPL-2.1-or-later in
  3.7.0), and hardened:
  - a trajectory without a periodic box is refused rather than analysed
    silently, since contacts across the box faces would be lost;
  - PDB files written by GROMACS keep the box of their first frame;
  - a re-run reproduces a production contact cache **bit for bit**.

  CoAna was published on GitHub at
  [nguoithamdo/CoAna](https://github.com/nguoithamdo/CoAna), tested
  continuously on Linux and Windows (255 tests).

---

## Version summary

| Period | Version | Milestone |
|---|---|---|
| Jan 2024 | — | Development begins (analysis notebooks) |
| 2025 | — | First two theses to use CoAna: α-synuclein pre-condensation (bachelor's) and phase separation (master's, Dec 2025) |
| Feb 2026 | Oligomer Tracker | Notebook refactored into a package |
| Mar–Apr 2026 | CoAna v1, v2 | Two-stage design: contact cache + fast analysis |
| May–Jun 2026 | 3.0 – 3.1.2 | Installable package; lineage and fate layer; third thesis to use CoAna (IAPP, type 2 diabetes) |
| Aug 2026 | 3.3.0, CoAnaPure 1.2 | Own trajectory engine; main-fragment tracking |
| 3 Sep 2026 | 3.4.0 | Mixed systems; guest mechanism |
| 6 / 9 Oct 2026 | 3.5.0 / 3.5.1 | Branch-local mechanism; review fixes |
| 23–24 Oct 2026 | 3.6.0 | Audit, MIT license, renamed `coana`, public on GitHub |
| 27 Oct 2026 | 3.7.0 | All versions relicensed under the LGPL-2.1-or-later; `coana citations` |

---

*CoAna is developed by Dr. Nguyen Truong Co, Truong Co Group ·
[ORCID 0000-0001-5642-3641](https://orcid.org/0000-0001-5642-3641) ·
[Google Scholar](https://scholar.google.com/citations?user=0ttB9NsAAAAJ)*
