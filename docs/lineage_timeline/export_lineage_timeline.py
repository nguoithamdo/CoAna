# Copyright (c) 2026 Nguyen Truong Co
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Export the split/merge lineage graph as a time-ordered JSON payload.

Feeds `build_timeline_html.py`, which inlines the payload into a standalone
interactive page.

**The implementation now lives in the package** as
`oligomer_analysis.timeline.build_timeline_payload`, so the same code backs
this script, the `coana timeline` subcommand, and any direct library use. This
file remains as the two-step entry point: use it when you want to keep the
intermediate JSON, diff two runs, or feed the payload to a different renderer.

For a single command, prefer::

    coana timeline <cache> -o TIMELINE.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from oligomer_analysis.models import TargetStateConfig  # noqa: E402
from oligomer_analysis.timeline import (  # noqa: E402
    DEFAULT_BUILD_PARAMS,
    build_timeline_payload,
)

# The cache shipped with the repository, so a bare invocation works on a fresh
# clone. Any other cache can be passed with --cache.
DEFAULT_CACHE = os.path.join(ROOT, "notebooks", "cache_md1_1956frames_cut6.5_atomblocks.npz")


def export(cache_path, out_path, frame_lo=None, frame_hi=None, *,
           target_config=None, **build_params):
    """Build the payload for ``cache_path`` and write it to ``out_path``."""
    payload = build_timeline_payload(
        cache_path, frame_lo=frame_lo, frame_hi=frame_hi,
        target_config=target_config, **build_params)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"written {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")
    print(json.dumps(payload["meta"], indent=2))
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "lineage_timeline.json"))
    ap.add_argument("--frame-lo", type=int, default=None)
    ap.add_argument("--frame-hi", type=int, default=None)
    for name, value in DEFAULT_BUILD_PARAMS.items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    ap.add_argument("--min-size", type=int, default=10,
                    help="target state: minimum aggregate size")
    ap.add_argument("--min-duration", type=int, default=5,
                    help="target state: minimum lifetime in frames")
    ap.add_argument("--min-contribution", type=float, default=0.70,
                    help="on-pathway contribution threshold")
    args = ap.parse_args()

    export(
        args.cache, args.out, args.frame_lo, args.frame_hi,
        target_config=TargetStateConfig(
            min_size=args.min_size,
            min_duration=args.min_duration,
            min_contribution=args.min_contribution,
        ),
        **{name: getattr(args, name) for name in DEFAULT_BUILD_PARAMS},
    )
