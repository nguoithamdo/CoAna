from __future__ import annotations

from itertools import count
from typing import Dict, FrozenSet, List, Optional

import numpy as np
import pandas as pd

from .graph_tools import build_cluster_transition_matrix, clusters_from_cmap, intra_cluster_contacts
from .metrics import compute_free_energy_from_sizes, free_energy_to_dataframe
from .models import ContactCache, Pathway
from .plotting_common import format_oligomer_label


def analyze_from_cache(
    cache: ContactCache,
    *,
    inter_thresh: int,
    size_thresh: int,
    continue_min_size: int,
    overlap_min: int,
    shrink_tolerance: int,
    dur_store_min: int,
    follow: str = "legacy",
    plot_dur_min: int = 15,
    do_plots: bool = False,
    out_prefix: Optional[str] = None,
    do_network: bool = False,
    network_thr: float = 1e-3,
    do_free_energy: bool = False,
    free_energy_oligomer_thresh: int = 1,
    temperature_K: float = 330.0,
    free_energy_per_long_pathways: bool = False,
    free_energy_life_threshold: Optional[int] = None,
    pathway_label_mode: str = "full",
    show_legend: Optional[bool] = None,
) -> dict:
    """Stage B analysis from cached contact maps.

    Returns structured ``Pathway`` objects instead of raw dictionaries.

    Parameters
    ----------
    follow : {"legacy", "main_fragment"}
        How a pathway is continued across a frame boundary.

        ``"legacy"`` (default) keeps the original cluster-centric greedy
        matching bit-for-bit, so existing results stay reproducible.  It is
        known to follow the wrong fragment after splits: when a cluster breaks
        10 -> 8+2 it can record the 2 (and even a stray monomer) instead of the
        main fragment -- on the reference dataset 14.7 % of all continuation
        steps do this and 83 % of pathways are affected.

        ``"main_fragment"`` continues each pathway with the overlapping cluster
        that shares the **most members** with it (ties broken toward the larger
        cluster, then the older pathway), assigning contended clusters globally
        by descending overlap.  After a 10 -> 8+2 split the pathway follows the
        8.  This removes the size-crash artifact entirely (0 wrong-branch
        events on the reference dataset) and is the recommended mode for new
        analyses.
    """
    if follow not in ("legacy", "main_fragment"):
        raise ValueError(
            f"follow must be 'legacy' or 'main_fragment', got {follow!r}."
        )
    frames = cache.frames
    frames_clusters = [clusters_from_cmap(cmap, inter_thresh) for cmap in cache.cmaps]

    label_counter = count(0)

    def new_label(cluster: FrozenSet[int]) -> str:
        return f"oli{next(label_counter)}_" + "_".join(map(str, sorted(cluster)))

    def close_pathway(finished: List[Pathway], info: dict, end_idx: int) -> None:
        duration = end_idx - info["start_idx"]
        if duration >= dur_store_min:
            finished.append(
                Pathway(
                    label=info["label"],
                    start_idx=info["start_idx"],
                    start_frame=int(frames[info["start_idx"]]),
                    duration=duration,
                    sizes=list(info["sizes"]),
                    sets=list(info["sets"]),
                )
            )

    current: Dict[str, dict] = {}
    finished: List[Pathway] = []

    for ti, clusters in enumerate(frames_clusters):
        candidates = [c for c in clusters if len(c) >= continue_min_size]
        starters = [c for c in clusters if len(c) >= size_thresh]
        updated = set()

        def continue_with(lbl: str, clu: FrozenSet[int]) -> None:
            """Record ``clu`` as pathway ``lbl``'s next frame (shared by both modes)."""
            cur = current[lbl]
            cur["sizes"].append(len(clu))
            cur["sets"].append(clu)
            cur["below"] = cur["below"] + 1 if len(clu) < size_thresh else 0
            updated.add(lbl)
            if cur["below"] > shrink_tolerance:
                cur["sizes"].pop()
                cur["sets"].pop()
                updated.discard(lbl)
                close_pathway(finished, current.pop(lbl), ti)

        # 1) continue existing pathways
        if follow == "main_fragment":
            # Pathway-centric matching: every eligible (pathway, cluster) pair
            # is ranked by (overlap, cluster size, pathway age) and assigned
            # greedily, so each pathway follows the fragment that carries the
            # most of its members -- after a 10 -> 8+2 split, the 8.  The
            # legacy subset bonus is gone: a degenerate 2-member pathway can no
            # longer outbid a 60-member overlap for the re-merged cluster.
            pairs = []
            for age, (lbl, info) in enumerate(current.items()):
                prev = info["sets"][-1]
                for clu in candidates:
                    overlap = len(prev & clu)
                    subset = overlap == len(prev) and overlap > 0
                    if subset or overlap >= overlap_min:
                        pairs.append((overlap, len(clu), -age, lbl, clu))
            pairs.sort(key=lambda x: x[:3], reverse=True)
            taken = set()
            for overlap, _, _, lbl, clu in pairs:
                if lbl in updated or lbl not in current or clu in taken:
                    continue
                taken.add(clu)
                continue_with(lbl, clu)
        else:
            for clu in candidates:
                best_lbl = None
                best_score = -1

                for lbl, info in current.items():
                    prev = info["sets"][-1]
                    overlap = len(prev & clu)
                    subset = overlap == len(prev) and overlap > 0
                    score = (100000 if subset else 0) + overlap
                    if score > best_score and (subset or overlap >= overlap_min):
                        best_score = score
                        best_lbl = lbl

                if best_lbl is None or best_lbl in updated:
                    continue

                continue_with(best_lbl, clu)

        # 2) start new pathways
        for clu in starters:
            explained = False
            for lbl in updated:
                if len(current[lbl]["sets"][-1] & clu) >= overlap_min:
                    explained = True
                    break
            if explained:
                continue

            lbl = new_label(clu)
            current[lbl] = {
                "label": lbl,
                "start_idx": ti,
                "sizes": [len(clu)],
                "sets": [clu],
                "below": 0,
            }
            updated.add(lbl)

        # 3) close pathways not updated
        for lbl in list(current.keys()):
            if lbl not in updated:
                info = current.pop(lbl)
                close_pathway(finished, info, ti)

    for lbl, info in list(current.items()):
        close_pathway(finished, info, len(frames))

    if out_prefix:
        pd.DataFrame(
            {
                "label": [p.label for p in finished],
                "display_label": [format_oligomer_label(p.label, pathway_label_mode) for p in finished],
                "start_frame": [p.start_frame for p in finished],
                "duration": [p.duration for p in finished],
                "final_size": [len(p.sets[-1]) if p.sets else 0 for p in finished],
                "max_size": [int(np.max(p.sizes)) if p.sizes else 0 for p in finished],
                "sizes": [",".join(map(str, p.sizes)) for p in finished],
            }
        ).to_csv(f"{out_prefix}_pathways.csv", index=False)

    if do_plots:
        from .monitor import (
            network_plot_global,
            plot_cluster_contacts,
            plot_size_transition_matrix,
            plot_total_contacts_per_frame,
            plot_total_intercontacts_per_frame,
            plot_pathway_sizes,
        )

        long_paths = [p for p in finished if p.duration >= plot_dur_min]
        plot_pathway_sizes(frames, long_paths, min_duration=plot_dur_min, label_mode=pathway_label_mode, show_legend=show_legend)

        T = build_cluster_transition_matrix(frames_clusters)
        M = plot_size_transition_matrix(T, normalise="all", cmap="Reds")
        plot_cluster_contacts(finished, cache, min_duration=plot_dur_min, label_mode=pathway_label_mode, show_legend=show_legend)
        plot_total_contacts_per_frame(cache)
        plot_total_intercontacts_per_frame(cache)

        if do_network:
            network_plot_global(M, thr=network_thr)

        if do_free_energy:
            cluster_sizes = [
                len(cluster)
                for frame in frames_clusters
                for cluster in frame
                if len(cluster) >= int(free_energy_oligomer_thresh)
            ]
            df = free_energy_to_dataframe(
                compute_free_energy_from_sizes(cluster_sizes, temperature_K=temperature_K)
            )
            if out_prefix:
                df.to_csv(f"{out_prefix}_free_energy_global.csv", index=False)

            if free_energy_per_long_pathways:
                threshold = plot_dur_min if free_energy_life_threshold is None else int(free_energy_life_threshold)
                for pathway in finished:
                    if pathway.duration >= threshold:
                        dfp = free_energy_to_dataframe(
                            compute_free_energy_from_sizes(pathway.sizes, temperature_K=temperature_K)
                        )
                        if out_prefix:
                            safe = pathway.label.replace("/", "_")
                            dfp.to_csv(f"{out_prefix}_free_energy_{safe}.csv", index=False)

    return {"frames_clusters": frames_clusters, "finished": finished}
