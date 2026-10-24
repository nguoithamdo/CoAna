/* Headless smoke test for LINEAGE_TIMELINE.html.
 *
 *     node docs/lineage_timeline/smoke_test.js
 *
 * Runs the page's own inline script against a minimal DOM shim and exercises
 * the real code paths: boot, layout, lane packing, event ordering, stepping,
 * zoom clamping, focus, and every filter.  The page exposes `__timeline` for
 * exactly this purpose.
 */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const PAGE = process.argv[2] ||
  path.join(__dirname, "..", "..", "LINEAGE_TIMELINE.html");
const HTML = fs.readFileSync(PAGE, "utf8");
const start = HTML.indexOf("<script>") + "<script>".length;
const script = HTML.slice(start, HTML.indexOf("</script>", start));

const ARROW = String.fromCharCode(0x2192);   // avoids any encoding round-trip

/* ---------- DOM shim ---------- */
const nodes = new Map();
function mkNode(id) {
  const n = {
    id, innerHTML: "", textContent: "", value: "", hidden: false, disabled: false,
    style: {}, dataset: {},
    classList: {add() {}, remove() {}, contains() { return false; }},
    clientWidth: 1200, clientHeight: 420, scrollTop: 0, _attrs: {},
    setAttribute(k, v) { this._attrs[k] = String(v); },
    getAttribute(k) { return k in this._attrs ? this._attrs[k] : null; },
    removeAttribute(k) { delete this._attrs[k]; },
    addEventListener() {}, removeEventListener() {}, appendChild() {},
    closest() { return null; }, querySelectorAll() { return []; },
    getBoundingClientRect() {
      return {left: 0, top: 0, width: 1200, height: 420, right: 1200, bottom: 420};
    },
  };
  nodes.set(id, n);
  return n;
}
[
  "axis", "plot", "mini", "scroller", "empty", "logscroll", "logpad", "inspector",
  "meta", "foot", "stepnow", "stepcount", "logcount", "framebox", "play", "prev",
  "next", "first", "last", "zin", "zout", "fitall", "fitsel", "pickmode", "picktop",
  "pickclass", "pickone", "linlist", "minsize", "theme", "tip",
].forEach(mkNode);

const chips = [];
for (const cls of ["on-pathway", "uncertain", "off-pathway"]) {
  const n = mkNode("chip-cls-" + cls);
  n.dataset.cls = cls; n.setAttribute("aria-pressed", "true"); chips.push(n);
}
for (const ev of ["split", "merge", "split_merge", "rearrangement"]) {
  const n = mkNode("chip-ev-" + ev);
  n.dataset.ev = ev; n.setAttribute("aria-pressed", "true"); chips.push(n);
  mkNode("c-" + ev);
}

const document = {
  documentElement: {_t: null, setAttribute(k, v) { this._t = v; }, getAttribute() { return this._t; }},
  getElementById: id => nodes.get(id) || mkNode(id),
  querySelectorAll: sel => {
    if (sel === "[data-cls]") return chips.filter(c => c.dataset.cls);
    if (sel === "[data-ev]") return chips.filter(c => c.dataset.ev);
    return [];
  },
  addEventListener() {},
};

const sandbox = {
  document, console,
  window: {addEventListener() {}, innerWidth: 1400, innerHeight: 900},
  innerWidth: 1400, innerHeight: 900,
  getComputedStyle: () => ({getPropertyValue: () => "#000000"}),
  matchMedia: () => ({matches: false}),
  requestAnimationFrame: () => 1, cancelAnimationFrame() {},
  setInterval: () => 1, clearInterval() {}, setTimeout: () => 1,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

/* ---------- harness ---------- */
let fails = 0;
function check(name, fn) {
  try {
    const msg = fn();
    console.log("  PASS  " + name + (msg ? "  " + msg : ""));
  } catch (e) {
    fails++;
    console.log("  FAIL  " + name + "  " + (e && e.message));
  }
}

console.log("running the page script...");
try {
  vm.runInContext(script, sandbox, {filename: "timeline.js"});
  console.log("  PASS  boot (the page initialised without throwing)");
} catch (e) {
  console.log("  FAIL  boot  " + e.stack);
  process.exit(1);
}
const S = sandbox.__timeline;
if (!S) { console.log("  FAIL  page did not expose __timeline"); process.exit(1); }

check("layout built", () => {
  const L = S.layout;
  if (!L.rows.length) throw new Error("no lineage rows");
  if (!L.segs.length) throw new Error("no segments");
  if (!L.events.length) throw new Error("no events");
  return `${L.rows.length} rows, ${L.segs.length} segments, ${L.events.length} events, ${L.height}px tall`;
});

check("bands are ordered largest aggregate first", () => {
  const maxes = S.layout.rows.map(r => r.lin.max);
  for (let i = 1; i < maxes.length; i++) {
    if (maxes[i] > maxes[i - 1]) throw new Error("out of order: " + maxes.join(","));
  }
  return "peak sizes " + maxes.join(", ");
});

check("no two segments share a lane at the same time", () => {
  const byLane = new Map();
  for (const s of S.layout.segs) {
    const y = S.layout.laneOf.get(s.id);
    if (!byLane.has(y)) byLane.set(y, []);
    byLane.get(y).push(s);
  }
  let n = 0;
  for (const [y, list] of byLane) {
    list.sort((a, b) => a.f0 - b.f0);
    for (let i = 1; i < list.length; i++) {
      n++;
      if (list[i].f0 <= list[i - 1].f1) {
        throw new Error(`lane ${y}: ${list[i - 1].id} (${list[i - 1].f0}-${list[i - 1].f1}) ` +
          `overlaps ${list[i].id} (${list[i].f0}-${list[i].f1})`);
      }
    }
  }
  return `${byLane.size} lanes, ${n} adjacent pairs checked`;
});

check("the event log is in chronological order", () => {
  const list = S.layout.events;
  for (let i = 1; i < list.length; i++) {
    if (list[i].f < list[i - 1].f) throw new Error(`row ${i}: frame ${list[i].f} follows ${list[i - 1].f}`);
  }
  return `frame ${list[0].f} through ${list[list.length - 1].f}`;
});

check("every event in the log touches a drawn segment", () => {
  const ids = new Set(S.layout.segs.map(s => s.id));
  for (const e of S.layout.events) {
    if (!e.par.some(p => ids.has(p)) && !e.chi.some(c => ids.has(c))) {
      throw new Error("orphan event " + e.id);
    }
  }
  return "no orphans";
});

check("the plot emits bars and event markers", () => {
  const html = nodes.get("plot").innerHTML;
  const segs = (html.match(/data-seg=/g) || []).length;
  const evs = (html.match(/data-ev=/g) || []).length;
  if (!segs) throw new Error("no segment bars");
  if (!evs) throw new Error("no event markers");
  return `${segs} bars, ${evs} markers, ${(html.length / 1000).toFixed(0)}k chars of SVG`;
});

check("the mark budget is reported, never silent", () => {
  const foot = nodes.get("foot").innerHTML;
  const html = nodes.get("plot").innerHTML;
  const drawn = (html.match(/data-seg=/g) || []).length;
  const total = S.layout.segs.filter(s => s.f1 >= S.state.f0 && s.f0 <= S.state.f1).length;
  if (total > drawn && !/hidden at this zoom/.test(foot)) {
    throw new Error(`${total - drawn} bars culled with no notice in the footer`);
  }
  return total > drawn ? `${total - drawn} held back and announced` : "nothing culled";
});

check("axis, minimap and log all render", () => {
  if (!nodes.get("axis").innerHTML.includes("FRAME")) throw new Error("axis empty");
  if (!nodes.get("mini").innerHTML.includes("brush")) throw new Error("minimap has no window");
  if (!nodes.get("logpad").innerHTML.includes("data-evrow=")) throw new Error("log empty");
  return "all three";
});

check("stepping advances through time", () => {
  S.gotoEvent(0);
  const f0 = S.state.frame, id0 = S.state.selEv;
  for (let i = 0; i < 25; i++) S.gotoEvent(S.state.evIdx + 1);
  if (S.state.evIdx !== 25) throw new Error("index is " + S.state.evIdx);
  if (S.state.frame < f0) throw new Error("the frame went backwards");
  if (S.state.selEv === id0) throw new Error("the selection did not move");
  return `frame ${f0} to ${S.state.frame} over 25 steps`;
});

check("stepping stops at both ends", () => {
  S.gotoEvent(-5);
  if (S.state.evIdx !== 0) throw new Error("start clamp: " + S.state.evIdx);
  S.gotoEvent(1e9);
  if (S.state.evIdx !== S.layout.events.length - 1) throw new Error("end clamp: " + S.state.evIdx);
  return "clamped";
});

check("event markers stay inside the plot area", () => {
  S.gotoEvent(0);
  const html = nodes.get("plot").innerHTML;
  const cxs = [...html.matchAll(/class="ev"[^>]*cx="([\d.]+)"/g)].map(m => parseFloat(m[1]));
  if (!cxs.length) throw new Error("no markers");
  const bad = cxs.filter(x => x < 137.99 || x > S.plotW + 1);
  if (bad.length) throw new Error(bad.length + " markers outside the plot");
  return `${cxs.length} markers inside [138, ${S.plotW}]`;
});

check("zoom stays inside the trajectory", () => {
  S.zoom(0.25, 900);
  const {f0, f1} = S.state;
  if (f0 < 0 || f1 > 1955 || f1 <= f0) throw new Error(`window ${f0}-${f1}`);
  S.zoom(100);
  if (S.state.f0 !== 0 || S.state.f1 !== 1955) throw new Error(`zoom out gave ${S.state.f0}-${S.state.f1}`);
  return "clamped to 0-1955";
});

check("focusing a lineage dims the rest", () => {
  S.setFocus(S.layout.rows[Math.min(1, S.layout.rows.length - 1)].lin.id);
  if (!nodes.get("plot").innerHTML.includes('class="seg dim"')) throw new Error("nothing dimmed");
  S.setFocus(null);
  if (nodes.get("plot").innerHTML.includes('class="seg dim"')) throw new Error("dimming stuck on");
  return "on and off";
});

check("single-lineage mode isolates one lineage", () => {
  const want = S.layout.rows[0].lin.id;
  S.state.pickmode = "one"; S.state.oneLin = want;
  S.rebuild({keepWindow: false});
  if (S.layout.rows.length !== 1) throw new Error("rows=" + S.layout.rows.length);
  if (S.layout.rows[0].lin.id !== want) throw new Error("wrong lineage");
  const n = S.layout.segs.length, e = S.layout.events.length;
  S.state.pickmode = "top"; S.rebuild({keepWindow: false});
  return `${want}: ${n} segments, ${e} events`;
});

check("the event-type filter works both ways", () => {
  S.state.types.delete("split");
  S.rebuild();
  if (S.layout.events.some(e => e.type === "split")) throw new Error("splits survived");
  S.state.types.add("split");
  S.rebuild();
  if (!S.layout.events.some(e => e.type === "split")) throw new Error("splits did not return");
  return "off and on";
});

check("the class filter works both ways", () => {
  S.state.classes.delete("on-pathway");
  S.rebuild({keepWindow: false});
  if (S.layout.rows.some(r => r.lin.cls === "on-pathway")) throw new Error("class still shown");
  S.state.classes.add("on-pathway");
  S.rebuild({keepWindow: false});
  return "off and on";
});

check("the minimum-size filter drops small segments", () => {
  S.state.minSize = 20; S.rebuild();
  if (S.layout.segs.some(s => s.max < 20)) throw new Error("a small segment survived");
  const n = S.layout.segs.length;
  S.state.minSize = 2; S.rebuild();
  if (S.layout.segs.length <= n) throw new Error("the filter did not reset");
  return `${n} segments at 20, ${S.layout.segs.length} at 2`;
});

check("the inspector fills for a segment and for an event", () => {
  S.showSegment(S.layout.segs[0].id);
  if (!nodes.get("inspector").innerHTML.includes("frames")) throw new Error("segment view empty");
  S.showEvent(S.layout.events[0].id);
  if (!nodes.get("inspector").innerHTML.includes("parent")) throw new Error("event view empty");
  return "both";
});

check("a split reads as parent " + ARROW + " children", () => {
  const sp = S.layout.events.find(e => e.type === "split" && e.psz.length === 1 && e.csz.length >= 2);
  if (!sp) throw new Error("no simple split in view");
  const story = S.sizeStory(sp);
  if (!story.includes(ARROW)) throw new Error("no arrow in: " + story);
  const [lhs, rhs] = story.split(ARROW);
  if (parseInt(lhs, 10) !== sp.psz[0]) throw new Error("parent size wrong: " + story);
  if (rhs.split("+").length !== sp.csz.length) throw new Error("child count wrong: " + story);
  return story.trim();
});

check("a non-conserving split is labelled, not left as broken arithmetic", () => {
  const e = S.layout.events.find(x => x.type === "split" && x.g && x.psz.length === 1);
  if (!e) throw new Error("no recruiting split in view");
  const parent = e.psz[0], kids = e.csz.reduce((a, b) => a + b, 0);
  if (kids - parent !== (e.g || 0) - (e.l || 0)) {
    throw new Error(`balance mismatch: ${parent} vs ${kids}, tag ${e.g || 0}/${e.l || 0}`);
  }
  const tag = S.balanceTag(e), words = S.balanceWords(e);
  if (!tag.includes("+")) throw new Error("no gain chip: " + tag);
  if (!/joined from solution/.test(words)) throw new Error("no explanation: " + words);
  S.showEvent(e.id);
  if (!nodes.get("inspector").innerHTML.includes("joined from solution")) {
    throw new Error("the inspector does not explain the imbalance");
  }
  return `${S.sizeStory(e)}  ${tag}  (${words})`;
});

check("every event's chip matches its own size arithmetic", () => {
  let annotated = 0;
  for (const e of S.layout.events) {
    const p = e.psz.reduce((a, b) => a + b, 0);
    const c = e.csz.reduce((a, b) => a + b, 0);
    // Overlapping parents cannot happen: clusters in one frame are disjoint,
    // so the sums differ by exactly gained - lost.
    if (c - p !== (e.g || 0) - (e.l || 0)) {
      throw new Error(`${e.id}: ${p} -> ${c} but tag says +${e.g || 0}/-${e.l || 0}`);
    }
    if (e.g || e.l) annotated++;
  }
  return `${annotated} of ${S.layout.events.length} events carry a chip`;
});

check("the rearrangement label is justified by a provenance number below the bar", () => {
  const bar = S.DATA.meta.params.union_cover_min;
  let rearr = 0, clean = 0;
  for (const e of S.layout.events) {
    const shaped = (e.psz.length === 1) !== (e.csz.length === 1);  // 1->many or many->1
    if (!shaped) { if (e.cov != null) throw new Error(`${e.id}: cov on a ${e.psz.length}->${e.csz.length} event`); continue; }
    if (e.cov == null) throw new Error(`${e.id}: ${e.psz.length}->${e.csz.length} carries no coverage`);
    // The classifier's own rule, re-checked from the payload the page shows.
    if (e.type === "rearrangement") { if (e.cov >= bar) throw new Error(`${e.id}: cov ${e.cov} >= bar`); rearr++; }
    else { if (e.cov < bar) throw new Error(`${e.id}: ${e.type} with cov ${e.cov} < bar`); clean++; }
  }
  if (!rearr) throw new Error("no rearrangements to check");
  return `${rearr} rearrangements all below ${bar}, ${clean} clean splits/merges all at or above it`;
});

check("a diluted merge explains itself in words", () => {
  const e = S.layout.events.find(x => x.type === "rearrangement" && x.psz.length > 1 && x.cov < 0.5);
  if (!e) throw new Error("no strongly diluted rearrangement found");
  const words = S.provenanceWords(e);
  for (const want of ["rearrangement", "merge", "%"]) {
    if (!words.includes(want)) throw new Error(`missing ${JSON.stringify(want)}: ${words}`);
  }
  return `f${e.f} ${S.sizeStory(e)} ${ARROW} "${words}"`;
});

check("bar height grows with oligomer size", () => {
  const h2 = S.barH(2), h20 = S.barH(20), h100 = S.barH(100);
  if (!(h2 < h20 && h20 < h100)) throw new Error(`${h2} / ${h20} / ${h100}`);
  return `size 2 ${ARROW} ${h2.toFixed(1)}px, 20 ${ARROW} ${h20.toFixed(1)}px, 100 ${ARROW} ${h100.toFixed(1)}px`;
});

check("off-pathway is violet in both themes, and never the old red", () => {
  if (HTML.includes("#d03b3b")) throw new Error("old red still present");
  const light = /--cls-off:\s*#683090/.test(HTML);
  const dark = (HTML.match(/--cls-off:\s*#9048a0/g) || []).length;
  if (!light) throw new Error("light violet #683090 missing");
  if (dark < 2) throw new Error(`dark violet #9048a0 must appear in the media block and the [data-theme] block, found ${dark}`);
  return "light #683090, dark #9048a0 in both dark scopes";
});

check("every event carries member sets that match its declared sizes", () => {
  for (const e of S.layout.events) {
    if (!e.pm || !e.cm) throw new Error(`${e.id}: no member sets`);
    e.pm.forEach((s, i) => { if (s.length !== e.psz[i]) throw new Error(`${e.id}: parent ${i} has ${s.length} members but psz says ${e.psz[i]}`); });
    e.cm.forEach((s, i) => { if (s.length !== e.csz[i]) throw new Error(`${e.id}: child ${i} has ${s.length} members but csz says ${e.csz[i]}`); });
    // The +n/−n chip must be re-derivable from the sets themselves.
    const pu = new Set(e.pm.flat()), cu = new Set(e.cm.flat());
    const g = [...cu].filter(m => !pu.has(m)).length, l = [...pu].filter(m => !cu.has(m)).length;
    if (g !== (e.g || 0) || l !== (e.l || 0)) throw new Error(`${e.id}: sets say +${g}/−${l}, chip says +${e.g || 0}/−${e.l || 0}`);
  }
  return `${S.layout.events.length} events, sizes and chips all re-derived from the sets`;
});

check("member sets stay hidden behind a click and mark who joined and left", () => {
  const e = S.layout.events.find(x => x.f === 1 && x.type === "rearrangement");
  if (!e) throw new Error("frame-1 rearrangement not found");
  const html = S.membersPanel(e);
  if (!html.startsWith("<details")) throw new Error("not a <details> panel");
  if (html.includes("<details open") || html.includes("open>")) throw new Error("panel is open by default");
  for (const want of ["{", "}", 'class="new"', 'class="gone"', "solution"]) {
    if (!html.includes(want)) throw new Error(`missing ${JSON.stringify(want)}`);
  }
  const news = (html.match(/class="new"/g) || []).length;
  const gones = (html.match(/class="gone"/g) || []).length;
  if (news !== (e.g || 0) || gones !== (e.l || 0)) throw new Error(`marked ${news} new / ${gones} gone, chip says +${e.g || 0}/−${e.l || 0}`);
  return `f1 panel: ${news} joined highlighted, ${gones} departure struck, closed until clicked`;
});

check("a segment recovers its member sets from its bounding events", () => {
  const s = S.layout.segs.find(x => S.segM0.get(x.id) && S.segM1.get(x.id));
  if (!s) throw new Error("no segment with both endpoints recorded");
  const html = S.segMembersPanel(s);
  if (!html.startsWith("<details")) throw new Error("not a <details> panel");
  if (!html.includes(`f${s.f0}`) || !html.includes(`f${s.f1}`)) throw new Error("endpoint frames missing");
  const m0 = S.segM0.get(s.id), m1 = S.segM1.get(s.id);
  return `${s.id}: {${m0.join(", ")}} at f${s.f0} ${ARROW} {${m1.join(", ")}} at f${s.f1}`;
});

console.log(fails ? "\n" + fails + " CHECK(S) FAILED" : "\nall checks passed");
process.exit(fails ? 1 : 0);
