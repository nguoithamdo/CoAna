from __future__ import annotations

import numpy as np
import pandas as pd

from oligomer_analysis import (
    ContactCache,
    subset_cache_by_frames,
    subset_cache_by_indices,
    oligomer_sizes_by_frame_dataframe,
    oligomer_summary_by_frame_dataframe,
    format_oligomer_label,
    draw_transition_graph_from_prob,
)


def _cache(n_frames=5, nseg=4):
    cmaps = np.zeros((n_frames, nseg, nseg), dtype=np.int32)
    return ContactCache(
        frames=np.arange(1, n_frames + 1, dtype=np.int32),
        cmaps=cmaps,
        nintra=np.zeros(n_frames, dtype=np.int32),
        ninter=np.zeros(n_frames, dtype=np.int32),
        ntotal=np.zeros(n_frames, dtype=np.int32),
        n_free=np.full(n_frames, nseg, dtype=np.int32),
        n_bound=np.zeros(n_frames, dtype=np.int32),
        seg_centers=np.zeros((n_frames, nseg, 3), dtype=np.float32),
        boxes=np.zeros((n_frames, 6), dtype=np.float32),
        meta={"nseg": nseg},
    )


def test_subset_cache_helpers():
    c = _cache()
    sub = subset_cache_by_indices(c, [0, 2, 4])
    assert sub.frames.tolist() == [1, 3, 5]
    sub2 = subset_cache_by_frames(c, [2, 4])
    assert sub2.frames.tolist() == [2, 4]


def test_oligomer_frame_tables():
    c = _cache(n_frames=2, nseg=5)
    frames_clusters = [[frozenset({0, 1}), frozenset({2, 3, 4})], [frozenset({0}), frozenset({1, 2})]]
    df = oligomer_sizes_by_frame_dataframe(c, frames_clusters, min_size=2)
    assert list(df["oligomer_size"]) == [3, 2, 2]
    summary = oligomer_summary_by_frame_dataframe(df)
    assert summary.loc[summary["frame"] == 1, "n_oligomers"].iloc[0] == 2
    assert summary.loc[summary["frame"] == 2, "largest_oligomer"].iloc[0] == 2


def test_label_format():
    assert format_oligomer_label("oli8_16_20_25", mode="ordinal") == "oli8"
    assert format_oligomer_label("oli8_16_20_25", mode="full") == "oli8_16_20_25"
