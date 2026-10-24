from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .species import SpeciesMap


@dataclass(slots=True)
class PrecomputeConfig:
    """Configuration for Stage A cache construction.

    Stage A is the expensive, trajectory-facing step.  It builds PBC-aware
    residue/segment contact maps and stores them in a reusable cache.  Stage B
    and the lineage/fate analysis below can then run repeatedly without touching
    the trajectory again.
    """

    topology_file: Optional[str]
    traj_file: str
    frames: List[int]
    cutoff_contact: float
    intra_exclude: int

    # How a contact is decided:
    #   "residue_cog" : distance between residue centres of geometry (default;
    #                   every cache and published number so far uses this)
    #   "min_atom"    : minimum distance over all atom pairs of the two
    #                   residues, which is the convention used by OLIGAMIX,
    #                   ATRANET and most of the aggregation literature.
    # A centroid is never closer than the closest atoms, so at a fixed cutoff
    # "min_atom" finds at least as many contacts as "residue_cog".
    contact_mode: str = "residue_cog"
    n_jobs: int = 1
    chunk_size: int = 10
    in_memory: bool = False
    fallback_box: Optional[Tuple[float, float, float, float, float, float]] = None

    # How to define one physical monomer/chain for contact-map binning.
    #
    # "mdanalysis"    : use MDAnalysis segments. This is the original behavior and
    #                    works when the topology contains one unique segment per
    #                    physical monomer.
    # "atom_blocks"    : ignore PDB chain/segment IDs and define monomers by
    #                    consecutive atom blocks. This is the safest option for
    #                    large Martini PDB systems where PDB chain IDs are reused
    #                    after about 62-63 chains.
    # "residue_blocks" : define monomers by consecutive residue blocks. Use this
    #                    when each monomer has a fixed number of residues/beads and
    #                    the trajectory is ordered monomer-by-monomer.
    # "atom_counts"    : like "atom_blocks" but for mixed systems where monomers
    #                    have different sizes; provide the per-monomer atom counts
    #                    in file order via ``monomer_atom_counts``.
    segment_mode: str = "mdanalysis"
    atoms_per_monomer: Optional[int] = None
    residues_per_monomer: Optional[int] = None
    n_monomers: Optional[int] = None
    monomer_atom_counts: Optional[List[int]] = None

    # CoAnaPure-only: restrict the analysis to a subset of atoms before
    # segmentation. One clause with optional negation, e.g. "resname BB",
    # "not resname W ION", "resid 1-128", "chain A B", "index 0-829".
    # Ranges are inclusive; resid/chain follow the topology file, index is the
    # 0-based atom index. The MDAnalysis engine ignores this field.
    selection: Optional[str] = None

    # CoAnaPure-only, mixtures. One string naming every molecular species in
    # topology order, e.g. "aSyn:295x50,Ab40:92x10". It fills
    # ``monomer_atom_counts`` (so ``segment_mode='atom_counts'`` needs no second
    # list), and records the species in the cache metadata so every downstream
    # species-resolved analysis works without re-declaring the composition.
    species_spec: Optional[str] = None

    # Optional Stage A extras. Both need one pass over the residue-pair list that
    # is otherwise discarded after binning to segments, so they are opt-in.
    #
    # contact_classes: decompose inter-chain residue contacts by residue
    #   chemistry (hydrophobic / polar / charged / ...), giving a (n_frames,
    #   n_classes, n_classes) matrix. This is what turns "these chains stick"
    #   into "these chains stick through salt bridges".
    # hotspot_blocks: accumulate residue-resolved cross-species contact maps in
    #   this many equal time blocks (1 = one map for the whole trajectory,
    #   0/None = off). Answers which sequence region of species A touches
    #   species B.
    contact_classes: bool = False
    hotspot_blocks: Optional[int] = None


@dataclass(slots=True)
class FrameContacts:
    """Per-frame contact summary saved before cache aggregation."""

    frame: int
    nintra: int
    ninter: int
    ntotal: int
    n_free: int
    n_bound: int
    cmap: np.ndarray
    seg_centers: np.ndarray
    box: np.ndarray

    # Optional Stage A extras (see PrecomputeConfig.contact_classes /
    # .hotspot_blocks). ``hotspot`` is folded into the running block sums as soon
    # as the frame is consumed and then dropped, so peak memory stays at a few
    # frames rather than the whole trajectory.
    class_cmap: Optional[np.ndarray] = None
    hotspot: Optional[np.ndarray] = None


@dataclass(slots=True)
class ContactCache:
    """Stage A cache loaded from ``.npz``.

    The cache is the stable interface between expensive MDAnalysis trajectory
    processing and all downstream pathway/lineage analyses.
    """

    frames: np.ndarray
    cmaps: np.ndarray
    nintra: np.ndarray
    ninter: np.ndarray
    ntotal: np.ndarray
    n_free: np.ndarray
    n_bound: np.ndarray
    seg_centers: np.ndarray
    boxes: np.ndarray
    meta: Dict[str, Any]

    # Optional Stage A extras, present only when they were requested. Caches
    # written before these existed load with both set to None, so every reader
    # must treat them as optional.
    #
    # class_cmaps: (n_frames, n_classes, n_classes) inter-chain residue-contact
    #   counts decomposed by residue chemistry class. Class names live in
    #   ``meta['contact_classes']``.
    # hotspot_maps: (n_blocks, n_species, n_species, n_res, n_res) residue-
    #   resolved contact counts between chains of each species pair, summed over
    #   all chain pairs and over the frames of each time block. Block frame
    #   ranges live in ``meta['hotspot_blocks']``.
    class_cmaps: Optional[np.ndarray] = None
    hotspot_maps: Optional[np.ndarray] = None

    @property
    def class_names(self) -> List[str]:
        """Contact-class names, or an empty list when Stage A did not store them."""
        return list(self.meta.get("contact_classes") or [])

    @property
    def species_names(self) -> List[str]:
        """Species names recorded at Stage A, or an empty list."""
        return [str(d["name"]) for d in (self.meta.get("species") or [])]


@dataclass(slots=True)
class Pathway:
    """Original linear oligomer pathway object.

    This object is intentionally preserved for backward compatibility with the
    old Stage B tracker and the mature 9-panel dashboard.  The new lineage layer
    uses :class:`OligomerSegment`, :class:`LineageEvent`, and :class:`Lineage`.
    """

    label: str
    start_idx: int
    start_frame: int
    duration: int
    sizes: List[int]
    sets: List[FrozenSet[int]]


@dataclass(slots=True)
class TargetStateConfig:
    """Operational definition of productive aggregation.

    On/off-pathway classification is only meaningful after defining a target
    aggregated state.  A transparent first definition is a persistent aggregate:

    - cluster size >= ``min_size``
    - continuous lifetime >= ``min_duration`` frames

    ``min_contribution`` controls how much monomer continuity into such target
    states is required to call a lineage on-pathway.  The explicit
    ``on_path_contribution_threshold`` alias is retained for backward
    compatibility with earlier package versions.
    """

    min_size: int = 10
    min_duration: int = 5
    min_contribution: float = 0.70
    on_path_contribution_threshold: Optional[float] = None
    off_path_contribution_threshold: float = 0.30

    def __post_init__(self) -> None:
        """Keep old and new threshold names synchronized.

        Earlier package versions used ``on_path_contribution_threshold`` while
        the cleaner public API uses ``min_contribution``. The alias is kept so
        old notebooks do not break, but if the user only sets
        ``min_contribution`` the classifier must obey that value.
        """
        if self.on_path_contribution_threshold is None:
            self.on_path_contribution_threshold = float(self.min_contribution)


@dataclass(slots=True)
class OligomerSegment:
    """One continuous connected oligomer episode.

    A segment is deliberately *not* allowed to pass through a split or merge.
    When an oligomer splits or merges, the old segment ends and one or more new
    segments start.  The higher-level lineage graph then records the biological
    relationship between these segments.

    This distinction is critical for aggregation: temporary fragmentation does
    not mean off-pathway.  Fate must be interpreted at the lineage level.
    """

    segment_id: str
    start_idx: int
    end_idx: int
    frames: List[int]
    sizes: List[int]
    sets: List[FrozenSet[int]]
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    event_start: str = "birth"
    event_end: str = "terminal"
    lineage_id: Optional[str] = None
    notes: str = ""

    # ``(duration, n_species)`` chain counts per frame, attached only when the
    # lineage was built with a species map.  Without it every entry is None and
    # the segment behaves exactly as in a one-component analysis.
    compositions: Optional[np.ndarray] = None

    @property
    def duration(self) -> int:
        return len(self.frames)

    @property
    def start_frame(self) -> int:
        return int(self.frames[0]) if self.frames else -1

    @property
    def end_frame(self) -> int:
        return int(self.frames[-1]) if self.frames else -1

    @property
    def max_size(self) -> int:
        return int(max(self.sizes)) if self.sizes else 0

    @property
    def mean_size(self) -> float:
        return float(np.mean(self.sizes)) if self.sizes else 0.0

    @property
    def initial_set(self) -> FrozenSet[int]:
        return self.sets[0] if self.sets else frozenset()

    @property
    def final_set(self) -> FrozenSet[int]:
        return self.sets[-1] if self.sets else frozenset()

    @property
    def last_set(self) -> FrozenSet[int]:
        return self.final_set

    @property
    def union_members(self) -> FrozenSet[int]:
        members: set[int] = set()
        for state in self.sets:
            members.update(state)
        return frozenset(members)

    @property
    def parent_segment_ids(self) -> List[str]:
        return self.parent_ids

    @property
    def child_segment_ids(self) -> List[str]:
        return self.child_ids

    @property
    def peak_composition(self) -> Optional[np.ndarray]:
        """Species composition at the frame where the segment was largest."""
        if self.compositions is None or not self.sizes:
            return None
        return self.compositions[int(np.argmax(self.sizes))]

    @property
    def final_composition(self) -> Optional[np.ndarray]:
        if self.compositions is None or len(self.compositions) == 0:
            return None
        return self.compositions[-1]


@dataclass(slots=True)
class LineageEvent:
    """Event connecting parent and child segments across a frame boundary.

    The event object stores enough provenance to audit why a split, merge, birth,
    death, or rearrangement was called.
    """

    event_id: str
    event_type: str
    frame_idx: int
    prev_frame: int
    next_frame: int
    parent_segment_ids: List[str]
    child_segment_ids: List[str]
    parent_sets: List[FrozenSet[int]]
    child_sets: List[FrozenSet[int]]
    overlap_matrix: Optional[np.ndarray] = None
    parent_fraction_matrix: Optional[np.ndarray] = None
    child_fraction_matrix: Optional[np.ndarray] = None
    description: str = ""
    notes: str = ""

    @property
    def frame_number(self) -> int:
        """Frame number at which the event is observed, using the child frame."""
        return int(self.next_frame)


@dataclass(slots=True)
class Lineage:
    """Aggregation lineage linking segments through split/merge-aware events.

    Classification is done here because a lineage may fragment and re-form while
    remaining biologically on-pathway.  Size alone is not a valid on/off-pathway
    criterion; future fate is central.
    """

    lineage_id: str
    segment_ids: List[str]
    event_ids: List[str]
    representative_segment_id: Optional[str]
    class_label: str
    target_reached: bool
    contribution_score: float
    growth_score: float
    persistence_score: float
    reversibility_score: float
    structural_score: float
    trap_score: float = 0.0
    max_size: int = 0
    duration: int = 0
    start_frame: int = -1
    end_frame: int = -1
    notes: str = ""

    # Species composition, filled only when the lineage was built with a species
    # map.  ``composition`` counts the distinct chains of each species that were
    # ever part of the lineage; ``peak_composition`` is the composition of the
    # largest state it reached; ``hetero_fraction`` is the fraction of the
    # lineage's frames in which more than one species was present.  These are the
    # columns that answer "are the productive lineages the mixed ones?".
    species_names: Optional[List[str]] = None
    composition: Optional[np.ndarray] = None
    peak_composition: Optional[np.ndarray] = None
    hetero_fraction: float = float("nan")

    @property
    def classification(self) -> str:
        return self.class_label

    @property
    def scores(self) -> Dict[str, float]:
        return {
            "target_reached": float(self.target_reached),
            "target_contribution_score": float(self.contribution_score),
            "growth_score": float(self.growth_score),
            "persistence_score": float(self.persistence_score),
            "trap_score": float(self.trap_score),
            "reversibility_score": float(self.reversibility_score),
            "structural_score": float(self.structural_score),
            "max_size": float(self.max_size),
            "duration": float(self.duration),
        }


@dataclass(slots=True)
class LineageTrackingResult:
    """Full result of split/merge-aware aggregation tracking."""

    frames_clusters: List[List[FrozenSet[int]]]
    segments: List[OligomerSegment]
    events: List[LineageEvent]
    lineages: List[Lineage]
    target_config: TargetStateConfig

    # The species map used to build the compositions, when one was given.
    species_map: Optional["SpeciesMap"] = None
    # Non-fatal problems detected while building (see lineage.lineage_warnings).
    warnings: List[str] = field(default_factory=list)

    @property
    def segment_dict(self) -> Dict[str, OligomerSegment]:
        return {seg.segment_id: seg for seg in self.segments}

    @property
    def event_dict(self) -> Dict[str, LineageEvent]:
        return {ev.event_id: ev for ev in self.events}

    @property
    def lineage_dict(self) -> Dict[str, Lineage]:
        return {lin.lineage_id: lin for lin in self.lineages}
