# Interactive lineage timeline

A time-ordered, interactive view of the split/merge lineage graph — the same
nodes and edges a force-directed lineage plot shows, but positioned on a real
frame axis so events read left to right in the order they happened.

**Output:** `LINEAGE_TIMELINE.html` at the project root. One self-contained
file; open it straight from disk.

## Build one for your own data

One command, any Stage A cache:

```bash
coana timeline my_cache.npz -o MY_TIMELINE.html
```

Everything that shapes the page is a flag, so a system unlike the 128-monomer
reference run does not need code edits:

```bash
coana timeline my_cache.npz -o MY_TIMELINE.html \
    --frame-lo 640 --frame-hi 820 \        # window a long trajectory
    --inter-thresh 1 --cluster-min-size 2 \# lineage construction
    --min-size 8 --min-duration 4          # what counts as a productive target
```

`--min-size/--min-duration/--min-contribution` define the target state, which
decides the on/off-pathway colouring — revisit them for a system much smaller
or larger than 128 monomers, where "size >= 10 held for >= 5 frames" may make
no sense. `coana timeline --help` lists them all; the defaults reproduce
`build_lineages_from_cache`.

From Python:

```python
from oligomer_analysis import generate_lineage_timeline

generate_lineage_timeline("my_cache.npz", "MY_TIMELINE.html",
                          frame_lo=640, frame_hi=820, inter_thresh=1)
```

## Regenerate the shipped page

The two-step form keeps the intermediate JSON, which is useful for diffing two
runs or feeding a different renderer:

```bash
python docs/lineage_timeline/export_lineage_timeline.py   # cache  -> out/lineage_timeline.json
python docs/lineage_timeline/build_timeline_html.py       # json   -> LINEAGE_TIMELINE.html
node   docs/lineage_timeline/smoke_test.js                # 29 checks against the built page
```

Both scripts accept `--cache`/`--out`, the same lineage and target flags as
`coana timeline`, and `--frame-lo/--frame-hi`:

```bash
python docs/lineage_timeline/export_lineage_timeline.py --frame-lo 640 --frame-hi 820
```

Both are thin wrappers: the implementation lives in
[`oligomer_analysis/timeline.py`](../../oligomer_analysis/timeline.py), which
also owns the page template so it travels with an installed wheel.

A static version of the same layout, for reports and slides:

```bash
python docs/lineage_timeline/preview_static.py --lineages 6 --frame-lo 640 --frame-hi 820
```

## Reading the page

| Mark | Meaning |
|---|---|
| Horizontal bar | one **segment** — a continuous oligomer episode. Length = lifetime, height ∝ log(peak size) |
| Bar fill | the **lineage class**: green on-pathway, amber uncertain, violet off-pathway |
| Connector + dot | an **event** linking parent segments to child segments at one frame |
| ▼ orange | split · ▲ blue merge · ◆ aqua split + merge · ● grey re-arrangement |
| Lane stack | concurrent branches of the same lineage; the top lane is the surviving main fragment |
| Teal vertical line | the playhead — the frame you are currently standing on |

Each lineage gets its own horizontal band, largest aggregate first. Within a
band, segments are packed into lanes greedily by start frame, so two segments
never share a lane at the same time (the smoke test asserts this).

## Following events step by step

The right rail is the chronological control surface:

- **Event log** — every event in frame order, each row reading
  `frame · type · 80 → 78 + 2`, so a split states its own arithmetic.
- **◀ / ▶ or the arrow keys** — walk the events one at a time; the playhead,
  the plot window and the inspector all follow.
- **space** — play through them; the speed selector goes from slow to turbo.
- **Frame box** — type a frame number to jump to the first event at or after it.

### Why the sizes often don't add up

A row like `f1640 ▼ split 4 → 2 + 3` is not a mass balance. The numbers are the
**cluster sizes in each frame**: a 4-mer at frame 1639 became a 2-mer and a
3-mer at frame 1640. The extra member arrived because free monomers are not
tracked — `cluster_min_size=2` keeps size-1 clusters out of the lineage graph —
so a fragment can bind a monomer out of solution (or release one back into it)
in the same step that it splits. In this real case monomer 117 was free at
frame 1639 and bound to the `{35, 51}` fragment at 1640.

Rows therefore carry a `+n` / `−n` chip counting members gained from, or lost
to, the untracked pool, and the inspector spells it out in words. Only **21 %**
of the 1,151 splits in the production run conserve their members exactly; 38 %
recruit, 14 % shed, and 28 % do both.

### Why some events are called "rearrangement"

`rearrangement` is the label the lineage layer uses when it can see that the
fragments are connected across a step but declines to name the step a clean
split or merge. The test is a provenance fraction, checked against
`union_cover_min` (0.70):

- **many → 1** is a merge only if ≥ 70 % of the child came from the tracked parents;
- **1 → many** is a split only if ≥ 70 % of the parent survived into the children.

Below the bar the result is mostly material the tracker was not following, so
calling it a merge would overstate what is actually known. `f1 ● rearrangement
2 + 2 → 7 (+4 −1)` is the extreme case: two dimers `{69,73}` and `{72,88}`
became the 7-mer `{32,69,72,73,84,95,109}` — member 88 left and four free
monomers joined at once, leaving coverage at 3/7 = **0.43**. Frame 1 is the
nucleation burst (eleven births fire in the same step), so most of that 7-mer
condensed out of solution rather than arriving from either dimer.

The label is rare — **91 of 9,102 events**, 46 would-be splits and 45 would-be
merges — and mostly marginal: the many→1 group runs min 0.29, median 0.67, max
0.69, i.e. nearly all of them sit just under the threshold. The inspector and
the hover tooltip show the exact percentage next to the bar it was tested
against, so the label always carries its own justification.

### Member sets — one click deeper

Every event and segment in the inspector carries a collapsed **member sets**
panel. Closed, the page stays a summary; open it and the actual monomer ids
appear in set notation, e.g. for the frame-1 rearrangement:

```
f0 before   {69, 73} + {72, 88}
f1 after    {32, 69, 72, 73, 84, 95, 109}
```

**Bold** members are newly bound from solution, ~~struck~~ members do not
survive the step — the same counts the `+n` / `−n` chip states, now with
names. For a segment the panel shows its membership at both endpoints
(recovered from its bounding events), so you can see exactly which monomers a
fragment traded during its lifetime.

Click a bar to focus its lineage (everything else dims) and fill the inspector;
click a gutter label to toggle that focus and fit the window to the lineage.
Drag the plot to pan, ctrl+wheel to zoom, and drag the overview strip at the
bottom to move the visible window across the whole trajectory.

## Notes on scale

The production cache builds 9,790 segments and 9,102 events across 2,165
lineages. Zoomed all the way out there are more segments than pixels, so the
page draws a bounded number of marks — biggest first, with a per-band quota so
no lineage renders as an empty stripe — and the footer always states how many
were held back. Zoom in and everything in the window is drawn.

The lineage layer is independent of the classic tracker's `follow` mode; it was
never affected by the wrong-fragment artifact of the legacy tracker (it
could follow the small fragment after a split), which is why it is the right basis for
this view.
