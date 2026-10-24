import numpy as np

from oligomer_analysis.commitment import compute_segment_commitment_scores, target_state_table
from oligomer_analysis.predictors import compute_lineage_features
from oligomer_analysis.hotspots import contact_enrichment_on_vs_off
from oligomer_analysis.models import TargetStateConfig


def test_target_config_alias_sync():
    cfg = TargetStateConfig(min_contribution=0.85)
    assert cfg.on_path_contribution_threshold == 0.85


def test_commitment_empty_result_shape():
    # This is a lightweight import/shape test. Synthetic biological behavior is
    # covered by test_lineage_synthetic.py from the lineage layer.
    assert callable(compute_segment_commitment_scores)
    assert callable(target_state_table)
    assert callable(compute_lineage_features)


def test_hotspot_enrichment_empty():
    import pandas as pd
    out = contact_enrichment_on_vs_off(pd.DataFrame())
    assert out.empty


# ---------------------------------------------------------------------------
# contact persistence: the denominator must be co-residency, not duration
# ---------------------------------------------------------------------------

def _segment_cache(memberships):
    """A cache in which every co-resident pair is in contact in every frame.

    Built so the correct persistence is exactly 1.0 by construction, whatever
    the membership schedule does.
    """
    from oligomer_analysis.models import ContactCache, OligomerSegment

    n_fr = len(memberships)
    n_mon = max(max(m) for m in memberships) + 1
    cmaps = np.zeros((n_fr, n_mon, n_mon), dtype=np.int32)
    for t, mem in enumerate(memberships):
        for a in mem:
            for b in mem:
                if a != b:
                    cmaps[t, a, b] = 5
    cache = ContactCache(
        frames=np.arange(n_fr), cmaps=cmaps,
        nintra=np.zeros(n_fr), ninter=np.zeros(n_fr), ntotal=np.zeros(n_fr),
        n_free=np.zeros(n_fr), n_bound=np.zeros(n_fr),
        seg_centers=np.zeros((n_fr, n_mon, 3)), boxes=np.zeros((n_fr, 3)), meta={},
    )
    seg = OligomerSegment(
        segment_id="seg0", start_idx=0, end_idx=n_fr - 1,
        frames=list(range(n_fr)),
        sizes=[len(m) for m in memberships],
        sets=[frozenset(m) for m in memberships],
    )
    return cache, seg


def test_contact_persistence_is_one_when_every_possible_contact_is_present():
    """A pair must not be charged for frames in which it could not exist.

    Three schedules, all of them perfectly stable: every pair of co-resident
    chains is in contact in every frame they share.  Persistence is therefore
    1.0 by construction.  Dividing by the full segment duration instead gives
    0.667 for the growing trimer and 0.400 for the 2 -> 11 growth, which would
    make persistence anti-correlated with growth -- the opposite of what this
    predictor is tested against.
    """
    from oligomer_analysis.predictors import segment_contact_persistence

    schedules = {
        "static dimer": [{0, 1}] * 10,
        "growing trimer": [{0, 1}] * 5 + [{0, 1, 2}] * 5,
        "2 -> 11 growth": [set(range(2 + k // 2)) for k in range(20)],
    }
    for name, memberships in schedules.items():
        cache, seg = _segment_cache(memberships)
        got = segment_contact_persistence(cache, seg, inter_thresh=1)
        assert got == 1.0, f"{name}: expected 1.0, got {got}"


def test_contact_persistence_counts_a_broken_contact_against_the_pair():
    """The metric must still fall when a co-resident pair actually separates.

    Members {0,1} for 10 frames, in contact in 6 of them -> 0.6 exactly.
    """
    from oligomer_analysis.models import ContactCache, OligomerSegment
    from oligomer_analysis.predictors import segment_contact_persistence

    n_fr = 10
    cmaps = np.zeros((n_fr, 2, 2), dtype=np.int32)
    for t in range(6):
        cmaps[t, 0, 1] = cmaps[t, 1, 0] = 5
    cache = ContactCache(
        frames=np.arange(n_fr), cmaps=cmaps,
        nintra=np.zeros(n_fr), ninter=np.zeros(n_fr), ntotal=np.zeros(n_fr),
        n_free=np.zeros(n_fr), n_bound=np.zeros(n_fr),
        seg_centers=np.zeros((n_fr, 2, 3)), boxes=np.zeros((n_fr, 3)), meta={},
    )
    seg = OligomerSegment(
        segment_id="seg0", start_idx=0, end_idx=n_fr - 1,
        frames=list(range(n_fr)), sizes=[2] * n_fr,
        sets=[frozenset({0, 1})] * n_fr,
    )
    assert segment_contact_persistence(cache, seg, inter_thresh=1) == 0.6
