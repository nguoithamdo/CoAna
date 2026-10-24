"""Inline an exported lineage payload into a standalone interactive page.

    python docs/lineage_timeline/export_lineage_timeline.py
    python docs/lineage_timeline/build_timeline_html.py

Writes LINEAGE_TIMELINE.html at the project root. The result is one file with
no external dependencies except the Google Fonts stylesheet, so it opens from
disk and can be shared as-is.

**The implementation now lives in the package** as
`oligomer_analysis.timeline.render_timeline_html`, which also owns the page
template (`oligomer_analysis/templates/timeline_template.html`) so it travels
with an installed wheel. This file is the second half of the two-step
workflow; for a single command, prefer::

    coana timeline <cache> -o TIMELINE.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from oligomer_analysis.timeline import (  # noqa: E402
    default_template_path,
    render_timeline_html,
)


def build(template: Path, payload: Path, out: Path) -> None:
    """Inline ``payload`` into ``template`` and write ``out``."""
    data = json.loads(payload.read_text(encoding="utf-8"))
    # node is available in this repo's toolchain, so keep the syntax check on:
    # a broken page is much cheaper to catch here than in a browser.
    render_timeline_html(data, out, template=template, check_js=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--template", default=str(default_template_path()))
    ap.add_argument("--payload", default=str(HERE / "out" / "lineage_timeline.json"))
    ap.add_argument("--out", default=str(ROOT / "LINEAGE_TIMELINE.html"))
    args = ap.parse_args()
    build(Path(args.template), Path(args.payload), Path(args.out))
