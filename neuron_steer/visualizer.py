"""neuron_steer.visualizer — 3Blue1Brown-style neural circuit visualizer.

Self-contained HTML, no dependencies. Open in any browser.

Usage
-----
    from neuron_steer.visualizer import visualize_network

    circuit = steerer.discover_circuit("What is the capital of France?", " Paris")
    visualize_network(circuit)

    graph = steerer.discover_edges("...", circuit)
    visualize_network(graph, output="out.html", open_browser=False)

Parameters
----------
    max_layers       : layers to show (None = all layers 0..max)
    n_per_col        : total neurons per column; circuit neurons snap to their
                       proportional slot, background fills the rest
    intermediate_size: MLP intermediate dimension (inferred if None)
"""

import json
import os
import random
import tempfile
import webbrowser
from typing import Optional, Union

from neuron_steer.core import Circuit, CircuitGraph


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def visualize_network(
    obj: Union[Circuit, CircuitGraph],
    output: Optional[str] = None,
    open_browser: bool = True,
    title: Optional[str] = None,
    max_layers: Optional[int] = None,
    n_per_col: int = 50,
    intermediate_size: Optional[int] = None,
) -> str:
    """3B1B-style neural network visualization.

    Each column is a transformer layer with exactly n_per_col neuron slots.
    Circuit neurons snap to the slot matching their index position within the
    MLP (neuron 8079 / 14336 → slot 57 of 100, etc.). Background neurons fill
    remaining slots. Circuit edges span directly to the *next circuit layer*,
    threading visually through any empty intermediate layers.
    """
    if isinstance(obj, CircuitGraph):
        circuit  = obj.circuit
        bn_set   = {(l, n) for (l, n), _, _ in obj.bottleneck()}
        sw_list  = obj.detect_super_weights()
        sw_set   = {(l, n) for (l, n), _, _, _ in sw_list}
        ew: dict = {}
        for e in obj.edges:
            ew[(e.source.layer, e.source.neuron,
                e.target.layer,  e.target.neuron)] = e.weight
    else:
        circuit = obj
        bn_set  = set()
        sw_set  = set()
        ew      = {}

    max_abs = max(abs(a) for a in circuit.neurons.values()) or 1.0

    # ── collapse by (layer, neuron) — same neuron at different token positions
    layer_attr: dict = {}
    collapsed:  dict = {}   # (layer, neuron_idx) -> (attr, representative_nidx)
    for nidx, attr in circuit.neurons.items():
        key = (nidx.layer, nidx.neuron)
        layer_attr[nidx.layer] = layer_attr.get(nidx.layer, 0) + abs(attr)
        if key not in collapsed or abs(attr) > abs(collapsed[key][0]):
            collapsed[key] = (attr, nidx)

    # ── which layers to show ─────────────────────────────────────────────────
    all_circuit_layers = sorted(layer_attr)
    n_layers = max(all_circuit_layers) + 1

    if max_layers is None:
        shown = list(range(n_layers))
    elif len(all_circuit_layers) <= max_layers:
        shown = all_circuit_layers
    else:
        shown = sorted(
            sorted(all_circuit_layers,
                   key=lambda l: layer_attr[l], reverse=True)[:max_layers]
        )
    col_idx = {l: i for i, l in enumerate(shown)}
    n_cols  = len(shown)

    # ── infer intermediate_size ──────────────────────────────────────────────
    if intermediate_size is None:
        max_neuron = max(nidx.neuron for nidx in circuit.neurons)
        intermediate_size = max(max_neuron + 1, int(max_neuron * 1.08) + 64)

    # ── build per-column slot grids ──────────────────────────────────────────
    # Each column has exactly n_per_col evenly-spaced slots (0..n_per_col-1).
    # Circuit neurons snap to round(y_frac * (n_per_col-1)); background fills rest.
    rng = random.Random(42)
    neurons_out: list = []

    def _find_free_slot(occupied: set, ideal: int) -> int:
        if ideal not in occupied:
            return ideal
        for delta in range(1, n_per_col):
            for candidate in (ideal - delta, ideal + delta):
                if 0 <= candidate < n_per_col and candidate not in occupied:
                    return candidate
        return -1  # column full (shouldn't happen with reasonable n_per_col)

    # Group circuit neurons by layer
    by_circuit_layer: dict = {}   # layer -> [(neuron_idx, attr, nidx)]
    for (layer, neuron_idx), (attr, nidx) in collapsed.items():
        by_circuit_layer.setdefault(layer, []).append((neuron_idx, attr, nidx))

    for layer in shown:
        col      = col_idx[layer]
        occupied: set = set()
        cn_list  = sorted(by_circuit_layer.get(layer, []),
                          key=lambda x: abs(x[1]), reverse=True)

        # Place circuit neurons
        circuit_slots: dict = {}   # neuron_idx -> slot
        for neuron_idx, attr, nidx in cn_list:
            y_frac   = min(neuron_idx / intermediate_size, 0.999)
            ideal    = round(y_frac * (n_per_col - 1))
            slot     = _find_free_slot(occupied, ideal)
            if slot < 0:
                continue
            occupied.add(slot)
            circuit_slots[neuron_idx] = slot
            y_snapped = slot / (n_per_col - 1)
            neurons_out.append({
                "id":          f"c_{layer}_{neuron_idx}",
                "col":         col,
                "layer":       layer,
                "neuron":      neuron_idx,
                "slot":        slot,
                "y_frac":      round(y_snapped, 6),
                "attribution": round(attr, 6),
                "is_circuit":  True,
                "is_bn":       (layer, neuron_idx) in bn_set,
                "is_sw":       (layer, neuron_idx) in sw_set,
                "label":       f"L{layer} / N{neuron_idx}",
            })

        # Fill remaining slots with background neurons
        for slot in range(n_per_col):
            if slot in occupied:
                continue
            y_frac = slot / (n_per_col - 1)
            neurons_out.append({
                "id":          f"bg_{layer}_{slot}",
                "col":         col,
                "layer":       layer,
                "neuron":      -1,
                "slot":        slot,
                "y_frac":      round(y_frac, 6),
                "attribution": 0.0,
                "is_circuit":  False,
                "is_bn":       False,
                "is_sw":       False,
                "label":       f"L{layer} background",
            })

    # ── build edges ──────────────────────────────────────────────────────────
    # Background: all-to-all between adjacent columns (dense gray web).
    # Circuit:    each circuit neuron → every neuron in the NEXT circuit layer,
    #             threading visually through any empty intermediate columns.

    col_neurons: dict = {}
    for n in neurons_out:
        col_neurons.setdefault(n["col"], []).append(n)

    edges_out: list = []

    # Background edges (adjacent columns only)
    for ci in range(n_cols - 1):
        for s in col_neurons.get(ci, []):
            for t in col_neurons.get(ci + 1, []):
                if s["is_circuit"] and t["is_circuit"]:
                    continue   # circuit-circuit pairs handled below
                edges_out.append({
                    "src":         s["id"],
                    "tgt":         t["id"],
                    "weight":      0.0,
                    "is_circuit":  False,
                    "spans_cols":  1,
                    "pulse":       False,
                })

    # Circuit edges: src → next circuit layer (may skip empty columns)
    circuit_layers_sorted = sorted(by_circuit_layer.keys())
    next_ckt = {circuit_layers_sorted[i]: circuit_layers_sorted[i + 1]
                for i in range(len(circuit_layers_sorted) - 1)}

    for src_layer in circuit_layers_sorted:
        if src_layer not in next_ckt or src_layer not in col_idx:
            continue
        tgt_layer = next_ckt[src_layer]
        if tgt_layer not in col_idx:
            continue
        spans = col_idx[tgt_layer] - col_idx[src_layer]
        for src_n, src_attr, _ in by_circuit_layer[src_layer]:
            for tgt_n, tgt_attr, _ in by_circuit_layer[tgt_layer]:
                w = ew.get(
                    (src_layer, src_n, tgt_layer, tgt_n),
                    ew.get((tgt_layer, tgt_n, src_layer, src_n),
                           src_attr * tgt_attr * 0.1),
                )
                edges_out.append({
                    "src":        f"c_{src_layer}_{src_n}",
                    "tgt":        f"c_{tgt_layer}_{tgt_n}",
                    "weight":     round(w, 6),
                    "is_circuit": True,
                    "spans_cols": spans,
                    "pulse":      False,
                })

    # Mark top circuit edges for animated pulses
    circuit_edges = sorted(
        [e for e in edges_out if e["is_circuit"]],
        key=lambda e: abs(e["weight"]), reverse=True,
    )
    n_pulses  = min(80, max(20, len(circuit_edges)))
    pulse_set = {id(e) for e in circuit_edges[:n_pulses]}
    for e in edges_out:
        e["pulse"] = id(e) in pulse_set

    payload = {
        "n_cols":            n_cols,
        "n_per_col":         n_per_col,
        "shown_layers":      shown,
        "max_abs_attr":      round(max_abs, 6),
        "intermediate_size": intermediate_size,
        "neurons":           neurons_out,
        "edges":             edges_out,
        "prompt":            circuit.prompt,
        "target_token":      circuit.target_token,
        "logit_diff":        round(circuit.total_logit_diff, 4),
        "n_circuit":         len(collapsed),
        "n_edges":           len([e for e in edges_out if e["is_circuit"]]),
    }

    _title = title or f"Circuit \u2014 {circuit.target_token.strip()!r}"
    _sub   = (
        f"{payload['n_circuit']} circuit neurons"
        + f" \u00b7 {payload['n_edges']} circuit edges"
        + f" \u00b7 logit_diff {circuit.total_logit_diff:+.3f}"
        + f" \u00b7 {n_per_col} slots/col"
        + f" \u00b7 intermediate_size {intermediate_size}"
        + (f" \u00b7 all {n_cols} layers" if max_layers is None
           else f" \u00b7 layers: {shown}")
        + f" \u00b7 \u201c{circuit.prompt[:80]}\u201d"
    )

    html = _TEMPLATE.replace("%%TITLE%%", _title)
    html = html.replace("%%SUBTITLE%%", _sub)
    html = html.replace("%%DATA%%", json.dumps(payload))

    if output is None:
        fd, output = tempfile.mkstemp(prefix="neuron_network_", suffix=".html")
        os.close(fd)

    with open(output, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"Saved \u2192 {output}")
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(output)}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>%%TITLE%%</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:#080c14;color:#c9d1d9;
  font-family:'JetBrains Mono','Fira Code','Courier New',monospace;
  overflow:hidden;user-select:none;
}
#shell{display:flex;flex-direction:column;width:100vw;height:100vh}
#hdr{padding:13px 22px 10px;border-bottom:1px solid #0f1d2e;flex-shrink:0}
#hdr-t{font-size:16px;font-weight:700;color:#f0f6fc;letter-spacing:.04em}
#hdr-s{font-size:10px;color:#2d4255;margin-top:3px;line-height:1.5}
#wrap{flex:1;position:relative;overflow:hidden}
svg{width:100%;height:100%;display:block}
#tip{
  position:absolute;background:#0a1524;border:1px solid #1e3a5f;
  border-radius:7px;padding:9px 13px;font-size:11px;color:#e2e8f0;
  pointer-events:none;opacity:0;transition:opacity .12s;
  max-width:270px;line-height:1.8;z-index:99;white-space:nowrap;
}
#tip.on{opacity:1}
#footer{
  display:flex;align-items:center;gap:18px;padding:7px 22px;
  border-top:1px solid #0f1d2e;font-size:10px;color:#243344;flex-shrink:0;
}
.leg{display:flex;align-items:center;gap:5px}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
#fhint{margin-left:auto;color:#1c2c3c}
@keyframes pulse{0%,100%{opacity:.92}50%{opacity:.38}}
.cn{animation:pulse 2.2s ease-in-out infinite}
</style>
</head>
<body>
<div id="shell">
  <div id="hdr">
    <div id="hdr-t">%%TITLE%%</div>
    <div id="hdr-s">%%SUBTITLE%%</div>
  </div>
  <div id="wrap">
    <svg id="viz" xmlns="http://www.w3.org/2000/svg"
         xmlns:xlink="http://www.w3.org/1999/xlink">
      <defs>
        <!-- neuron filters -->
        <filter id="fd" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="2.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="fm" x="-130%" y="-130%" width="360%" height="360%">
          <feGaussianBlur stdDeviation="5"  result="b1"/>
          <feGaussianBlur stdDeviation="12" result="b2"/>
          <feMerge><feMergeNode in="b2"/><feMergeNode in="b1"/>
          <feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="fs" x="-200%" y="-200%" width="500%" height="500%">
          <feGaussianBlur stdDeviation="8"  result="b1"/>
          <feGaussianBlur stdDeviation="20" result="b2"/>
          <feMerge><feMergeNode in="b2"/><feMergeNode in="b2"/>
          <feMergeNode in="b1"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <!-- edge glow: narrow halo for normal circuit edges -->
        <filter id="fe" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="2.5" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <!-- edge glow: wide halo for long-range spanning connections -->
        <filter id="fes" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="5" result="b1"/>
          <feGaussianBlur stdDeviation="12" result="b2"/>
          <feMerge><feMergeNode in="b2"/><feMergeNode in="b1"/>
          <feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <g id="g-bg-edges"></g>
      <g id="g-edge-glow"></g>
      <g id="g-edges"></g>
      <g id="g-pulses"></g>
      <g id="g-neurons"></g>
      <g id="g-labels"></g>
    </svg>
    <div id="tip"></div>
  </div>
  <div id="footer">
    <div class="leg"><div class="dot" style="background:#f97316;opacity:.9"></div>Positive attribution</div>
    <div class="leg"><div class="dot" style="background:#3b82f6;opacity:.9"></div>Negative attribution</div>
    <div class="leg"><div class="dot" style="background:#fbbf24"></div>Super-weight</div>
    <div class="leg"><div class="dot" style="background:none;border:1px solid #f97316"></div>Bottleneck</div>
    <div class="leg">
      <svg width="14" height="14" style="flex-shrink:0">
        <circle cx="7" cy="7" r="5" fill="none" stroke="#334155" stroke-width="1.2"/>
      </svg>Background
    </div>
    <div id="fhint">Y-slot = neuron index in MLP &nbsp;&middot;&nbsp; Hover to inspect &nbsp;&middot;&nbsp; Click to lock</div>
  </div>
</div>
<script>
const D   = %%DATA%%;
const NS  = 'http://www.w3.org/2000/svg';
const XL  = 'http://www.w3.org/1999/xlink';
const SVG = document.getElementById('viz');

function mk(tag, a, p) {
  const e = document.createElementNS(NS, tag);
  for (const [k,v] of Object.entries(a||{})) {
    if (k==='xl') e.setAttributeNS(XL,'href',v); else e.setAttribute(k,v);
  }
  p && p.appendChild(e);
  return e;
}

/* attribution → color */
function ac(a, mx) {
  const t = Math.max(-1, Math.min(1, a/(mx||1)));
  if (t>=0) {
    return `rgb(${Math.round(60+179*t)},${Math.round(20+103*Math.pow(t,.7))},${Math.round(5+17*(1-t))})`;
  }
  const s=-t;
  return `rgb(${Math.round(20+39*s)},${Math.round(20+110*s)},${Math.round(30+215*s)})`;
}
function ec(w){ return w>=0 ? '#f97316' : '#3b82f6'; }

/* layout */
const wr = document.getElementById('wrap');
const W  = wr.clientWidth  || window.innerWidth;
const H  = wr.clientHeight || (window.innerHeight - 80);

const NC      = D.n_cols;
const NR      = D.n_per_col;
const PAD_X   = 70, PAD_Y = 48;
const COL_W   = (W - 2*PAD_X) / (NC - 1 || 1);
const ROW_H   = (H - 2*PAD_Y) / (NR - 1 || 1);
const R_BG    = Math.max(2.5, Math.min(5, ROW_H * 0.28));
const R_MAX   = Math.max(5,   Math.min(11, ROW_H * 0.52));
const R_MIN   = Math.max(3,   Math.min(7,  ROW_H * 0.34));

function cx(col)  { return PAD_X + col * COL_W; }
function cy(frac) { return PAD_Y + frac * (H - 2*PAD_Y); }

const pos = {};
D.neurons.forEach(n => { pos[n.id] = { x: cx(n.col), y: cy(n.y_frac) }; });
const maxA  = D.max_abs_attr;
const maxCW = Math.max(...D.edges.filter(e=>e.is_circuit).map(e=>Math.abs(e.weight)), 1e-9);

/* ── background edges (dim, adjacent columns only) ─────────────── */
const gBg = document.getElementById('g-bg-edges');
D.edges.forEach(edge => {
  if (edge.is_circuit) return;
  const s = pos[edge.src], t = pos[edge.tgt];
  if (!s || !t) return;
  mk('line',{x1:s.x,y1:s.y,x2:t.x,y2:t.y,
    stroke:'#18304a','stroke-width':.45,opacity:.22,class:'bg-edge'},gBg);
});

/* ── circuit edges (glow + sharp layers, spanning any distance) ─── */
const gEg = document.getElementById('g-edge-glow');
const gEd = document.getElementById('g-edges');
const gPu = document.getElementById('g-pulses');

D.edges.forEach((edge, i) => {
  if (!edge.is_circuit) return;
  const s = pos[edge.src], t = pos[edge.tgt];
  if (!s || !t) return;

  const aw      = Math.abs(edge.weight) / maxCW;
  const col     = ec(edge.weight);
  const spanning = edge.spans_cols > 1;

  /* Stroke weight & opacity — spanning edges slightly bolder */
  const sw  = (spanning ? 1.0 : 0.7) + 2.6 * aw;
  const op  = (spanning ? 0.55 : 0.42) + 0.45 * aw;

  /* Bezier path — S-curve using horizontal midpoint as pivot.
     For spanning edges, add a slight vertical bow so they arc visually
     over the intermediate layers rather than cutting straight through. */
  const mx  = (s.x + t.x) / 2;
  const bow = spanning ? (t.y - s.y) * 0.08 : 0;   // gentle arc on long jumps
  const d   = `M${s.x} ${s.y} C${mx} ${s.y+bow} ${mx} ${t.y-bow} ${t.x} ${t.y}`;

  const pid     = `ep${i}`;
  const glowFlt = spanning ? 'url(#fes)' : 'url(#fe)';
  const glowSw  = sw + (spanning ? 9 : 5);
  const glowOp  = op * (spanning ? 0.45 : 0.32);

  /* Glow layer */
  mk('path',{d,fill:'none',stroke:col,filter:glowFlt,
    'stroke-width':glowSw,opacity:glowOp,'stroke-linecap':'round'},gEg);

  /* Sharp layer */
  mk('path',{id:pid,d,fill:'none',stroke:col,
    'stroke-width':sw,opacity:op,'stroke-linecap':'round',
    class:'ced','data-src':edge.src,'data-tgt':edge.tgt},gEd);

  /* Animated pulse dot */
  if (edge.pulse) {
    const pr  = spanning ? 3.0 : 2.2;
    const dur = `${(0.7 + (1-aw) * 1.5 + (spanning ? 0.4 : 0)).toFixed(2)}s`;
    const dot = mk('circle',{r:pr,fill:col,opacity:.98},gPu);
    const am  = mk('animateMotion',{
      dur, repeatCount:'indefinite',
      begin:`${((i*.17)%3).toFixed(2)}s`,
      calcMode:'spline',keySplines:'0.42 0 0.58 1',keyTimes:'0;1',
    },dot);
    mk('mpath',{xl:'#'+pid},am);
  }
});

/* ── neurons ─────────────────────────────────────────────────────── */
const gN  = document.getElementById('g-neurons');
const nEl = {};

D.neurons.forEach(n => {
  const {x,y} = pos[n.id];

  if (!n.is_circuit) {
    mk('circle',{cx:x,cy:y,r:R_BG,
      fill:'#0a1220',stroke:'#1e3352','stroke-width':.9,opacity:.55},gN);
    return;
  }

  const aa = Math.abs(n.attribution)/maxA;
  let fill,filt,r;
  if      (n.is_sw) { fill='#fbbf24';           filt='url(#fs)'; r=R_MAX; }
  else if (n.is_bn) { fill=ac(n.attribution,maxA); filt='url(#fm)'; r=R_MIN+5*aa; }
  else {
    fill=ac(n.attribution,maxA);
    filt = aa>.55 ? 'url(#fm)' : 'url(#fd)';
    r    = R_MIN + (R_MAX-R_MIN)*aa;
  }

  const delay=`${(Math.random()*2.2).toFixed(2)}s`;
  const dur  =`${(1.5+Math.random()*1.2).toFixed(2)}s`;

  mk('circle',{cx:x,cy:y,r:r+5,fill,opacity:.10+.17*aa,
    filter:filt,class:'cn',
    style:`animation-delay:${delay};animation-duration:${dur}`},gN);

  if(n.is_bn||n.is_sw)
    mk('circle',{cx:x,cy:y,r:r+2.5,fill:'none',stroke:fill,
      'stroke-width':1.2,opacity:.5},gN);

  const c=mk('circle',{cx:x,cy:y,r,fill,
    stroke:'rgba(255,255,255,0.25)','stroke-width':.8,filter:filt,
    class:'cn cn-body',
    style:`cursor:pointer;animation-delay:${delay};animation-duration:${dur}`,
    'data-id':n.id},gN);
  nEl[n.id]=c;
});

/* ── axis ruler (neuron index ticks) ─────────────────────────────── */
const gLb = document.getElementById('g-labels');
const IS  = D.intermediate_size;
for(let i=0;i<=5;i++){
  const frac=i/5, idx=Math.round(frac*IS), yy=cy(frac);
  mk('line',{x1:PAD_X-18,y1:yy,x2:PAD_X-8,y2:yy,
    stroke:'#1c3050','stroke-width':.8},gLb);
  const tl=mk('text',{x:PAD_X-22,y:yy+3.5,'text-anchor':'end',
    fill:'#1c3050','font-size':8,'font-family':'monospace'},gLb);
  tl.textContent=idx.toLocaleString();
}
const vtx=mk('text',{x:PAD_X-42,y:H/2,'text-anchor':'middle',
  fill:'#1c3050','font-size':9,'font-family':'monospace',
  transform:`rotate(-90,${PAD_X-42},${H/2})`},gLb);
vtx.textContent='neuron idx';

/* ── layer labels ────────────────────────────────────────────────── */
D.shown_layers.forEach((layer,ci) => {
  const x=cx(ci);
  const t=mk('text',{x,y:PAD_Y-18,'text-anchor':'middle',
    fill:'#4d7fc4','font-size':11,'font-family':'monospace','font-weight':'bold'},gLb);
  t.textContent=`L${layer}`;
  const cnt=D.neurons.filter(n=>n.is_circuit&&n.layer===layer).length;
  if(cnt>0){
    const s=mk('text',{x,y:PAD_Y-7,'text-anchor':'middle',
      fill:'#2d4a6b','font-size':8,'font-family':'monospace'},gLb);
    s.textContent=`${cnt}\u00d7`;
  }
});

/* ── tooltip + interaction ───────────────────────────────────────── */
const tip=document.getElementById('tip');
let locked=null;

function showTip(evt,n){
  const col=n.is_sw?'#fbbf24':n.is_bn?'#f97316':ac(n.attribution,maxA);
  const sgn=n.attribution>=0?'+':'';
  const pct=(n.y_frac*100).toFixed(1);
  tip.innerHTML=`<b style="color:#f0f6fc">${n.label}</b><br>`
    +`attr:&nbsp;<span style="color:${col}">${sgn}${n.attribution.toFixed(5)}</span><br>`
    +`slot&nbsp;${n.slot}&nbsp;/&nbsp;${D.n_per_col-1}&nbsp;`
    +`<span style="color:#7aa2c8">(${pct}% · N=${n.neuron.toLocaleString()})</span>`
    +(n.is_bn?`<br><span style="color:#f97316">&#9679; BOTTLENECK</span>`:'')
    +(n.is_sw?`<br><span style="color:#fbbf24">&#9733; SUPER-WEIGHT</span>`:'');
  tip.classList.add('on'); moveTip(evt);
}
function moveTip(evt){
  const r=wr.getBoundingClientRect();
  let tx=evt.clientX-r.left+14,ty=evt.clientY-r.top-12;
  if(tx+280>W)tx-=294; if(ty<0)ty=4;
  tip.style.left=tx+'px'; tip.style.top=ty+'px';
}
function dimEdges(id){
  document.querySelectorAll('.ced').forEach(p=>{
    p.style.opacity=id?(p.dataset.src===id||p.dataset.tgt===id?'1':'0.02'):'';
  });
  document.querySelectorAll('.bg-edge').forEach(p=>{
    p.style.opacity=id?'0.03':'';
  });
}
document.querySelectorAll('.cn-body').forEach(c=>{
  c.addEventListener('mouseenter',evt=>{
    if(locked)return;
    const n=D.neurons.find(x=>x.id===c.dataset.id);
    if(n){showTip(evt,n);dimEdges(n.id);}
  });
  c.addEventListener('mousemove',evt=>{if(!locked)moveTip(evt);});
  c.addEventListener('mouseleave',()=>{if(!locked){tip.classList.remove('on');dimEdges(null);}});
  c.addEventListener('click',evt=>{
    evt.stopPropagation();
    const id=c.dataset.id;
    if(locked===id){locked=null;tip.classList.remove('on');dimEdges(null);}
    else{locked=id;const n=D.neurons.find(x=>x.id===id);if(n){showTip(evt,n);dimEdges(id);}}
  });
});
SVG.addEventListener('click',()=>{
  if(locked){locked=null;tip.classList.remove('on');dimEdges(null);}
});
</script>
</body>
</html>"""
