"""Has the aggregation actually settled?

Every number quoted from a trajectory -- a mean cluster size, a bound fraction,
a partition coefficient -- assumes the run reached a steady state.  For
aggregation that assumption fails often, because the process is slow and the
trajectory is finite: a system still coarsening at the end of the run will hand
back a confident-looking mean that is really just a snapshot of a moving target.

This module makes that check a first-class, reportable step rather than
something done by eye on a plot:

* :func:`observable_series` pulls a scalar time series out of a cache;
* :func:`integrated_autocorrelation_time` measures how many frames it takes for
  that series to forget itself, which is what turns *n* frames into a much
  smaller number of independent samples;
* :func:`assess_convergence` combines a Geweke-style two-window comparison with
  a drift slope and returns a verdict plus an honest error bar.

The error bar is the part that matters most.  A naive standard error over 1000
correlated frames can be ten times too small, and quoting it is how a drifting
run comes to look converged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .graph_tools import clusters_from_cmap

__all__ = [
    "OBSERVABLES",
    "ConvergenceReport",
    "assess_convergence",
    "convergence_dataframe",
    "integrated_autocorrelation_time",
    "observable_series",
]


#: Scalar observables that can be tested, and what each one is for.
OBSERVABLES: Dict[str, str] = {
    "largest_cluster": "size of the largest connected assembly",
    "mean_cluster_size": "mean size of clusters at or above the size floor",
    "n_clusters": "number of clusters at or above the size floor",
    "n_bound": "chains with at least one inter-chain contact",
    "ninter": "total inter-chain residue contacts",
}


def observable_series(
    cache,
    observable: str = "largest_cluster",
    *,
    inter_thresh: int = 1,
    min_size: int = 2,
) -> np.ndarray:
    """Extract one scalar per frame from a cache.

    ``n_bound`` and ``ninter`` are read straight off the cache; the cluster
    observables are recomputed from the contact maps because they depend on
    ``inter_thresh`` and the size floor.
    """
    if observable not in OBSERVABLES:
        raise ValueError(
            f"unknown observable {observable!r}; choose from {sorted(OBSERVABLES)}"
        )
    if observable == "n_bound":
        return np.asarray(cache.n_bound, dtype=float)
    if observable == "ninter":
        return np.asarray(cache.ninter, dtype=float)

    cmaps = np.asarray(cache.cmaps)
    out = np.zeros(cmaps.shape[0], dtype=float)
    for f in range(cmaps.shape[0]):
        sizes = [
            len(c) for c in clusters_from_cmap(cmaps[f], inter_thresh) if len(c) >= min_size
        ]
        if observable == "largest_cluster":
            out[f] = max(sizes) if sizes else 0.0
        elif observable == "n_clusters":
            out[f] = len(sizes)
        else:  # mean_cluster_size
            out[f] = float(np.mean(sizes)) if sizes else 0.0
    return out


def integrated_autocorrelation_time(x: Sequence[float], *, c: float = 6.0) -> float:
    """Integrated autocorrelation time in frames, by the automatic-window method.

    Returns ``tau`` such that ``n_effective = n / (2 * tau)`` independent
    samples.  The window is truncated at the first ``M`` where ``M >= c * tau``
    (Sokal's rule), which keeps the estimate from being swamped by the noisy
    long-lag tail.  A constant series has no autocorrelation to measure and
    returns ``0.5``, the value that makes ``n_effective == n``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return 0.5
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0:
        return 0.5

    # FFT autocorrelation: pad to avoid the circular wrap-around.
    size = 1
    while size < 2 * n:
        size *= 2
    f = np.fft.rfft(x, size)
    acf = np.fft.irfft(f * np.conjugate(f), size)[:n].real
    acf /= acf[0]

    taus = 2.0 * np.cumsum(acf) - 1.0
    for m in range(1, n):
        if m >= c * taus[m]:
            return float(max(0.5, taus[m] / 2.0))
    return float(max(0.5, taus[-1] / 2.0))


def _blocked_sem(x: np.ndarray, tau: float) -> float:
    """Standard error corrected for autocorrelation."""
    n = x.size
    if n < 2:
        return float("nan")
    n_eff = max(1.0, n / (2.0 * max(tau, 0.5)))
    return float(np.std(x, ddof=1) / np.sqrt(n_eff))


@dataclass(slots=True)
class ConvergenceReport:
    """Outcome of :func:`assess_convergence`."""

    observable: str
    n_frames: int
    mean: float
    sd: float
    sem_naive: float
    sem_corrected: float
    autocorr_time: float
    n_effective: float
    geweke_z: float
    drift_per_1000_frames: float
    relative_drift: float
    converged: bool
    messages: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "converged" if self.converged else "NOT converged"

    def __str__(self) -> str:
        return (
            f"{self.observable}: {self.mean:.2f} +/- {self.sem_corrected:.2f} "
            f"({self.verdict})\n  " + "\n  ".join(self.messages)
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "observable": self.observable,
            "n_frames": self.n_frames,
            "mean": self.mean,
            "sd": self.sd,
            "sem_naive": self.sem_naive,
            "sem_corrected": self.sem_corrected,
            "autocorr_time": self.autocorr_time,
            "n_effective": self.n_effective,
            "geweke_z": self.geweke_z,
            "drift_per_1000_frames": self.drift_per_1000_frames,
            "relative_drift": self.relative_drift,
            "converged": self.converged,
            "notes": "; ".join(self.messages),
        }


def assess_convergence(
    cache,
    observable: str = "largest_cluster",
    *,
    inter_thresh: int = 1,
    min_size: int = 2,
    first_fraction: float = 0.10,
    last_fraction: float = 0.50,
    z_threshold: float = 2.0,
    drift_threshold: float = 0.10,
    series: Optional[Sequence[float]] = None,
) -> ConvergenceReport:
    """Test whether an observable has stopped changing, and by how much it might be.

    Two independent checks have to agree before a run is called converged:

    **Geweke z.**  The mean of an early window (the first ``first_fraction`` of
    frames, skipping nothing else) is compared with the mean of a late window
    (the last ``last_fraction``), each with an autocorrelation-corrected
    standard error.  ``|z| < z_threshold`` says the two windows are
    statistically indistinguishable.

    **Relative drift.**  A straight line through the late window, expressed as
    the change over that window divided by the overall mean.  This catches the
    slow, steady coarsening that a two-window comparison can miss when the drift
    is small relative to the frame-to-frame noise.

    The returned report carries both, the autocorrelation time, and the
    corrected standard error, so a non-converged run can still be quoted
    honestly rather than silently.
    """
    x = (
        np.asarray(series, dtype=float)
        if series is not None
        else observable_series(
            cache, observable, inter_thresh=inter_thresh, min_size=min_size
        )
    )
    # Overlapping windows share frames, which drags the two means together and
    # biases z toward 0 -- i.e. toward calling a drifting run converged. The
    # defaults (0.10, 0.50) are disjoint; a caller's are not guaranteed to be.
    if not 0.0 < first_fraction <= 1.0 or not 0.0 < last_fraction <= 1.0:
        raise ValueError("first_fraction and last_fraction must be in (0, 1]")
    if first_fraction + last_fraction > 1.0:
        raise ValueError(
            f"first_fraction ({first_fraction}) + last_fraction ({last_fraction}) "
            f"= {first_fraction + last_fraction:.2f} > 1, so the early and late "
            "windows would overlap and share frames. That biases the Geweke z "
            "toward 0, which reads as agreement between the windows and would "
            "make a drifting run look converged."
        )
    n = x.size
    messages: List[str] = []
    if n < 8:
        return ConvergenceReport(
            observable=observable, n_frames=n,
            mean=float(x.mean()) if n else float("nan"),
            sd=float(x.std(ddof=1)) if n > 1 else float("nan"),
            sem_naive=float("nan"), sem_corrected=float("nan"),
            autocorr_time=float("nan"), n_effective=float(n),
            geweke_z=float("nan"), drift_per_1000_frames=float("nan"),
            relative_drift=float("nan"), converged=False,
            messages=[f"only {n} frames; too short to assess convergence."],
        )

    tau = integrated_autocorrelation_time(x)
    n_eff = n / (2.0 * max(tau, 0.5))
    mean = float(x.mean())
    sd = float(x.std(ddof=1))

    n_first = max(2, int(round(n * first_fraction)))
    n_last = max(2, int(round(n * last_fraction)))
    # The fractional check above is necessary but not sufficient: both sizes are
    # rounded and clamped up to 2, so they can still overlap even when the
    # fractions sum to <= 1. With n=10, first=0.1, last=0.9 the fractions are
    # allowed but the realized sizes are 2 and 9, and frame 1 lands in both
    # windows -- which is the very sharing the fractional guard exists to stop.
    if n_first + n_last > n:
        raise ValueError(
            f"the early and late windows would overlap: {n} frames split into "
            f"first {n_first} and last {n_last} (from first_fraction="
            f"{first_fraction}, last_fraction={last_fraction}; each window is "
            "rounded and is at least 2 frames). Overlapping windows share "
            "frames, which biases the Geweke z toward 0 and would make a "
            "drifting run look converged. Use a shorter window or more frames."
        )
    a, b = x[:n_first], x[-n_last:]
    sem_a = _blocked_sem(a, integrated_autocorrelation_time(a))
    sem_b = _blocked_sem(b, integrated_autocorrelation_time(b))
    denom = float(np.sqrt(sem_a**2 + sem_b**2))
    diff = float(a.mean() - b.mean())
    if denom > 0:
        z = diff / denom
    elif diff == 0.0:
        z = 0.0
    else:
        # Both windows are constant but at different levels: a step change
        # with no noise around it, which is certainty that they differ, not
        # agreement.  Report it as infinitely many standard errors apart.
        z = float(np.copysign(np.inf, diff))

    t = np.arange(b.size, dtype=float)
    slope = float(np.polyfit(t, b, 1)[0]) if b.size > 2 else 0.0
    drift_1000 = slope * 1000.0
    window_change = slope * b.size
    if window_change == 0.0:
        relative_drift = 0.0
    elif mean:
        relative_drift = float(abs(window_change) / abs(mean))
    else:
        relative_drift = float("inf")

    ok_z = abs(z) < z_threshold
    ok_drift = relative_drift < drift_threshold
    converged = bool(ok_z and ok_drift)

    messages.append(
        f"autocorrelation time {tau:.1f} frames -> {n_eff:.0f} independent "
        f"samples out of {n} frames"
    )
    messages.append(
        f"Geweke z = {z:+.2f} "
        f"({'early and late windows agree' if ok_z else 'early and late windows differ'})"
    )
    messages.append(
        f"late-window drift {window_change:+.2f} = {100.0 * relative_drift:.1f}% of "
        f"the mean ({'flat' if ok_drift else 'still moving'})"
    )
    if not converged:
        messages.append(
            "quote this run as a lower bound or extend it; the mean below is "
            "not a plateau value."
        )
    return ConvergenceReport(
        observable=observable,
        n_frames=int(n),
        mean=mean,
        sd=sd,
        sem_naive=float(sd / np.sqrt(n)),
        sem_corrected=_blocked_sem(x, tau),
        autocorr_time=float(tau),
        n_effective=float(n_eff),
        geweke_z=z,
        drift_per_1000_frames=drift_1000,
        relative_drift=relative_drift,
        converged=converged,
        messages=messages,
    )


def convergence_dataframe(
    cache,
    observables: Optional[Sequence[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """One row per observable, so a run can be judged on more than one signal.

    A run whose largest cluster has plateaued while its contact count still
    climbs is compacting, not growing -- a distinction worth seeing before
    quoting either number.
    """
    names = list(observables or OBSERVABLES.keys())
    return pd.DataFrame(
        [assess_convergence(cache, name, **kwargs).to_dict() for name in names]
    )
