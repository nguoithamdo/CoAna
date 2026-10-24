from __future__ import annotations

import numpy as np

from oligomer_analysis import ContactCache, TargetStateConfig, build_lineages_from_cache


def _cache_from_cluster_series(cluster_series, nseg=12):
    cmaps = []
    n_free = []
    n_bound = []
    for clusters in cluster_series:
        cmap = np.zeros((nseg, nseg), dtype=np.int32)
        bound = set()
        for clu in clusters:
            clu = sorted(clu)
            if len(clu) >= 2:
                bound.update(clu)
            for a_i in range(len(clu) - 1):
                for b_i in range(a_i + 1, len(clu)):
                    a, b = clu[a_i], clu[b_i]
                    cmap[a, b] = 1
                    cmap[b, a] = 1
        cmaps.append(cmap)
        n_bound.append(len(bound))
        n_free.append(nseg - len(bound))
    cmaps = np.stack(cmaps)
    frames = np.arange(len(cluster_series), dtype=np.int32)
    return ContactCache(
        frames=frames,
        cmaps=cmaps,
        nintra=np.zeros(len(frames), dtype=np.int32),
        ninter=np.array([int(np.triu(c, 1).sum()) for c in cmaps], dtype=np.int32),
        ntotal=np.array([int(np.triu(c, 1).sum()) for c in cmaps], dtype=np.int32),
        n_free=np.array(n_free, dtype=np.int32),
        n_bound=np.array(n_bound, dtype=np.int32),
        seg_centers=np.zeros((len(frames), nseg, 3), dtype=np.float32),
        boxes=np.zeros((len(frames), 6), dtype=np.float32),
        meta={"nseg": nseg, "segids": [str(i) for i in range(nseg)]},
    )


def test_ten_splits_to_six_and_four_and_merges_back_to_one_lineage():
    cluster_series = [
        [frozenset(range(10))],
        [frozenset(range(6)), frozenset(range(6, 10))],
        [frozenset(range(6)), frozenset(range(6, 10))],
        [frozenset(range(10))],
    ]
    cache = _cache_from_cluster_series(cluster_series, nseg=10)
    result = build_lineages_from_cache(
        cache,
        inter_thresh=1,
        cluster_min_size=2,
        target_config=TargetStateConfig(min_size=10, min_duration=1),
    )
    event_types = [ev.event_type for ev in result.events]
    assert "split" in event_types
    assert "merge" in event_types
    assert len(result.lineages) == 1
    assert result.lineages[0].class_label == "on-pathway"
    assert len(result.lineages[0].segment_ids) == 4


def test_transient_small_oligomer_is_off_pathway_or_uncertain_not_on_pathway():
    cluster_series = [[], [frozenset({0, 1})], []]
    cache = _cache_from_cluster_series(cluster_series, nseg=4)
    result = build_lineages_from_cache(
        cache,
        inter_thresh=1,
        cluster_min_size=2,
        target_config=TargetStateConfig(min_size=5, min_duration=2),
    )
    assert len(result.lineages) == 1
    assert result.lineages[0].class_label in {"off-pathway", "uncertain"}
    assert result.lineages[0].class_label != "on-pathway"


def test_stable_medium_non_growing_trap_is_not_on_pathway():
    cluster_series = [[frozenset({0, 1, 2, 3})] for _ in range(8)]
    cache = _cache_from_cluster_series(cluster_series, nseg=8)
    result = build_lineages_from_cache(
        cache,
        inter_thresh=1,
        cluster_min_size=2,
        target_config=TargetStateConfig(min_size=8, min_duration=3),
    )
    assert len(result.lineages) == 1
    assert result.lineages[0].class_label in {"off-pathway", "uncertain"}
    assert result.lineages[0].target_reached is False


# ---------------------------------------------------------------------------
# FIX-12 (GAP-02): persistence is reported, never used as a fate criterion
# ---------------------------------------------------------------------------

def test_classifier_no_longer_accepts_an_unused_persistence_score():
    """_classify_lineage took persistence_score and never read it, while every
    lineage's notes claimed it was weighed. The parameter is gone, so a caller
    can no longer believe it is influencing the result."""
    import pytest
    from oligomer_analysis.lineage import _classify_lineage

    common = dict(target_reached=False, contribution_score=0.0,
                  growth_score=0.0, trap_score=0.9, lineage_duration=1,
                  config=TargetStateConfig(min_size=10, min_duration=5))
    assert _classify_lineage(**common) in ("on-pathway", "off-pathway", "uncertain")
    with pytest.raises(TypeError):
        _classify_lineage(persistence_score=0.5, **common)


def test_lineage_notes_state_the_criteria_actually_used():
    """The stated method must match the applied one, and persistence must
    still be reported for ranking."""
    from oligomer_analysis import lineage_summary_dataframe

    cluster_series = [
        [frozenset(range(10))],
        [frozenset(range(6)), frozenset(range(6, 10))],
        [frozenset(range(10))],
    ]
    cache = _cache_from_cluster_series(cluster_series, nseg=10)
    result = build_lineages_from_cache(
        cache, inter_thresh=1, cluster_min_size=2,
        target_config=TargetStateConfig(min_size=10, min_duration=1),
    )
    assert result.lineages
    for lin in result.lineages:
        assert "growth, persistence, and trap" not in lin.notes
        assert "not a classification criterion" in lin.notes
    assert "persistence_score" in lineage_summary_dataframe(result).columns
