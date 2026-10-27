# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Static preview of the interactive timeline.

Mirrors the page's layout rules (largest lineages first, greedy lane packing,
class-coloured bars, event connectors) with matplotlib so the same picture can
go in a report or a slide.  Useful as a check that the lane packing reads the
way it should before opening the interactive page.

    python docs/lineage_timeline/preview_static.py --lineages 6 --out preview.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

EV_COLOR = {"split": "#eb6834", "merge": "#2a78d6",
            "split_merge": "#1baf7a", "rearrangement": "#7c8494"}
EV_GLYPH = {"split": "v", "merge": "^", "split_merge": "D", "rearrangement": "o"}
CLS_COLOR = {"on-pathway": "#0ca30c", "uncertain": "#fab219", "off-pathway": "#683090"}
LANE = 1.0


def pack_lanes(segs):
    """Greedy lane packing, identical to the page's packLanes()."""
    lane_end, lane_of = [], {}
    for s in sorted(segs, key=lambda s: (s["f0"], s["id"])):
        idx = next((i for i, end in enumerate(lane_end) if end < s["f0"]), None)
        if idx is None:
            idx = len(lane_end)
            lane_end.append(-10**9)
        lane_end[idx] = s["f1"]
        lane_of[s["id"]] = idx
    return lane_of, max(1, len(lane_end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default=os.path.join(HERE, "out", "lineage_timeline.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "lineage_timeline",
                                                  "out", "timeline_preview.png"))
    ap.add_argument("--lineages", type=int, default=6)
    ap.add_argument("--frame-lo", type=int, default=None)
    ap.add_argument("--frame-hi", type=int, default=None)
    ap.add_argument("--min-size", type=int, default=2)
    args = ap.parse_args()

    with open(args.payload, encoding="utf-8") as fh:
        data = json.load(fh)
    seg_by_id = {s["id"]: s for s in data["segments"]}
    lin_by_id = {l["id"]: l for l in data["lineages"]}

    lo = args.frame_lo if args.frame_lo is not None else data["meta"]["frame_first"]
    hi = args.frame_hi if args.frame_hi is not None else data["meta"]["frame_last"]

    order = sorted(data["lineages"], key=lambda l: (-l["max"], -l["dur"], l["id"]))
    picked = [l["id"] for l in order[: args.lineages]]

    segs_of = {}
    for s in data["segments"]:
        if s["lin"] in picked and s["max"] >= args.min_size and s["f1"] >= lo and s["f0"] <= hi:
            segs_of.setdefault(s["lin"], []).append(s)
    picked = [p for p in picked if segs_of.get(p)]

    y_of, rows, y = {}, [], 0.0
    for lid in picked:
        lane_of, n = pack_lanes(segs_of[lid])
        for sid, li in lane_of.items():
            y_of[sid] = y + li * LANE
        rows.append({"lin": lin_by_id[lid], "y0": y - 0.6, "h": n * LANE + 0.4, "n": n})
        y += n * LANE + 1.4
    total_h = y

    fig, ax = plt.subplots(figsize=(15, max(4.2, 0.30 * total_h + 2.4)),
                           constrained_layout=True)
    ax.set_facecolor("#ffffff")

    for i, r in enumerate(rows):
        ax.add_patch(Rectangle((lo, r["y0"]), hi - lo, r["h"],
                               facecolor="#f2f5f8" if i % 2 == 0 else "#ffffff",
                               edgecolor="none", zorder=0))
        ax.text(lo - (hi - lo) * 0.012, r["y0"] + r["h"] / 2,
                f'{r["lin"]["id"]}\n{r["lin"]["max"]}-mer', ha="right", va="center",
                fontsize=8, family="monospace",
                color=CLS_COLOR.get(r["lin"]["cls"], "#555"))

    drawn = set(y_of)
    n_ev = 0
    for e in data["events"]:
        if e["type"] not in EV_COLOR or not (lo <= e["f"] <= hi):
            continue
        pys = [y_of[p] for p in e["par"] if p in drawn]
        cys = [y_of[c] for c in e["chi"] if c in drawn]
        if not pys or not cys:
            continue
        n_ev += 1
        col = EV_COLOR[e["type"]]
        hub = sum(pys + cys) / len(pys + cys)
        for yy in pys:
            ax.add_patch(FancyArrowPatch((e["fprev"], yy), (e["f"], hub),
                                         connectionstyle="arc3,rad=0.15",
                                         arrowstyle="-", color=col, lw=0.7,
                                         alpha=0.65, zorder=2))
        for yy in cys:
            ax.add_patch(FancyArrowPatch((e["f"], hub), (e["f"], yy),
                                         connectionstyle="arc3,rad=0.15",
                                         arrowstyle="-", color=col, lw=0.7,
                                         alpha=0.65, zorder=2))
        ax.plot([e["f"]], [hub], EV_GLYPH[e["type"]], color=col, ms=3.2,
                mec="white", mew=0.5, zorder=4)

    n_seg = 0
    for lid in picked:
        cls = lin_by_id[lid]["cls"]
        for s in segs_of[lid]:
            n_seg += 1
            h = 0.16 + 0.72 * (s["max"] ** 0.5) / (100 ** 0.5)
            ax.add_patch(Rectangle((s["f0"], y_of[s["id"]] - h), max(0.6, s["f1"] - s["f0"]), h,
                                   facecolor=CLS_COLOR.get(cls, "#888"), edgecolor="none",
                                   alpha=0.85, zorder=3))

    ax.set_xlim(lo - (hi - lo) * 0.055, hi + (hi - lo) * 0.01)
    ax.set_ylim(total_h, -1.4)
    ax.set_xlabel("Frame")
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", alpha=0.25, zorder=1)
    ax.set_title(f"Oligomer lineage timeline — {len(picked)} largest lineages, "
                 f"frames {lo}–{hi} ({n_seg} segments, {n_ev} events)", fontsize=11)

    handles = [Line2D([], [], marker=EV_GLYPH[t], color=EV_COLOR[t], ls="none", ms=6,
                      label=t.replace("_", " + ")) for t in EV_COLOR]
    handles += [Line2D([], [], marker="s", color=CLS_COLOR[c], ls="none", ms=7, label=c)
                for c in CLS_COLOR]
    ax.legend(handles=handles, ncol=7, frameon=False, fontsize=8.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.09))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=115, bbox_inches="tight", facecolor="white")
    print(f"written {args.out}  ({n_seg} segments, {n_ev} events, "
          f"{len(picked)} lineages, frames {lo}-{hi})")


if __name__ == "__main__":
    main()
