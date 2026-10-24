from __future__ import annotations

"""Lineage metric helpers.

The main lineage metrics are computed inside :mod:`oligomer_analysis.lineage`
because they depend on target-state discovery.  This module provides small public
helpers for downstream notebooks and future extensions.
"""

from typing import Iterable

import numpy as np

from .models import OligomerSegment


def segment_size_slope(segment: OligomerSegment) -> float:
    """Least-squares slope of segment size vs local time index."""
    if segment.duration < 2:
        return 0.0
    x = np.arange(segment.duration, dtype=float)
    y = np.asarray(segment.sizes, dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def segment_union_overlap_fraction(segment: OligomerSegment, target_members: Iterable[int]) -> float:
    """Fraction of a segment's union membership contained in a target set."""
    members = segment.union_members
    if not members:
        return 0.0
    target = set(target_members)
    return float(len(set(members) & target) / len(members))


__all__ = ["segment_size_slope", "segment_union_overlap_fraction"]
