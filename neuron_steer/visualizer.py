"""neuron_steer.visualizer — 3Blue1Brown-style neural circuit visualizer.

Self-contained HTML, no dependencies. Open in any browser.

Usage
-----
    from neuron_steer.visualizer import visualize_network

    circuit = steerer.discover_circuit("What is the capital of France?", " Paris")
    visualize_network(circuit)

    graph = steerer.discover_edges("...", circuit)
    visualize_network(graph, title="capitals hourglass")

    path = visualize_network(circuit, output="out.html", open_browser=False)
"""

import json
import os
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
    max_layers: int = 7,
    n_per_col: int = 14,
) -> str:
    """3B1B-style neural network visualization.

    Shows selected layers as full columns of circles (hollow = background,
    bright filled = circuit neuron), with dense edges between adjacent
    columns — circuit edges colored orange/blue, background edges dim gray.

    max_layers : how many layers to show (picks highest-attribution ones)
    n_per_col  : circles per column (circuit neurons + background filler)
    """
    if isinstance(obj, CircuitGraph):
        circuit   = obj.circuit
        bn_set    = {(l, n) for (l, n), _, _ in obj.bottleneck()}
        sw_list   = obj.detect_super_weights()
        sw_set    = {(l, n) for (l, n), _, _, _ in sw_list}
        # Build edge-weight lookup: (src_layer, src_neuron, tgt_layer, tgt_neuron) -> weight
        ew: dict = {}
        for e in obj.edges:
            ew[(e.source.layer, e.source.neuron,
                e.target.layer,  e.target.neuron)] = e.weight
    else:
        circuit = obj
        bn_set  = set()
        sw_set  = set()
        ew      = {}

    n_layers = max(n.layer for n in circuit.neurons) + 1
    max_abs  = max(abs(a) for a in circuit.neurons.values()) or 1.0

    # ── pick which layers to show ──────────────────────────────────────
    layer_attr: dict = {}
    by_layer:   dict = {}
    for nidx, attr in circuit.neurons.items():
        by_layer.setdefault(nidx.layer, []).append((nidx, attr))
        layer_attr[nidx.layer] = layer_attr.get(nidx.layer, 0) + abs(attr)

    for l in by_layer:
        by_layer[l].sort(key=lambda x: abs(x[1]), reverse=True)

    all_circuit_layers = sorted(layer_attr)
    if len(all_circuit_layers) <= max_layers:
        shown = all_circuit_layers
    else:
        shown = sorted(
            sorted(all_circuit_layers,
                   key=lambda l: layer_attr[l], reverse=True)[:max_layers]
        )

    col_idx = {l: i for i, l in enumerate(shown)}  # layer -> column index
    n_cols  = len(shown)

    # ── build neuron list ──────────────────────────────────────────────
    neurons_out: list = []
    id_map: dict = {}  # (layer, position, neuron) -> id

    for layer in shown:
        cn   = by_layer.get(layer, [])
        n_cn = min(len(cn), n_per_col)
        start     = max(0, (n_per_col - n_cn) // 2)
        occupied  = set()

        for i, (nidx, attr) in enumerate(cn[:n_cn]):
            row = start + i
            occupied.add(row)
            nid = f"c_{nidx.layer}_{nidx.position}_{nidx.neuron}"
            id_map[(nidx.layer, nidx.position, nidx.neuron)] = nid
            neurons_out.append({
                "id":          nid,
                "col":         col_idx[layer],
                "layer":       layer,
                "neuron":      nidx.neuron,
                "row":         row,
                "attribution": round(attr, 6),
                "is_circuit":  True,
                "is_bn":       (nidx.layer, nidx.neuron) in bn_set,
                "is_sw":       (nidx.layer, nidx.neuron) in sw_set,
                "label":       f"L{layer} / N{nidx.neuron}",
            })

        # Background neurons fill remaining rows
        bg_rows = [r for r in range(n_per_col) if r not in occupied]
        for row in bg_rows:
            neurons_out.append({
                "id":          f"bg_{layer}_{row}",
                "col":         col_idx[layer],
                "layer":       layer,
                "neuron":      -1,
                "row":         row,
                "attribution": 0.0,
                "is_circuit":  False,
                "is_bn":       False,
                "is_sw":       False,
                "label":       f"L{layer} background",
            })

    # ── build dense edges between adjacent shown layers ────────────────
    # Build per-col neuron lookup
    col_neurons: dict = {}
    for n in neurons_out:
        col_neurons.setdefault(n["col"], []).append(n)

    edges_out: list = []
    for ci in range(n_cols - 1):
        src_col = col_neurons.get(ci, [])
        tgt_col = col_neurons.get(ci + 1, [])
        src_layer = shown[ci]
        tgt_layer = shown[ci + 1]

        for s in src_col:
            for t in tgt_col:
                is_circuit_edge = s["is_circuit"] and t["is_circuit"]
                if is_circuit_edge:
                    # Look up edge weight; fall back to attribution product
                    w = ew.get(
                        (s["layer"], s["neuron"], t["layer"], t["neuron"]),
                        ew.get((t["layer"], t["neuron"], s["layer"], s["neuron"]),
                               s["attribution"] * t["attribution"] * 0.1),
                    )
                else:
                    w = 0.0
                edges_out.append({
                    "src":        s["id"],
                    "tgt":        t["id"],
                    "weight":     round(w, 6),
                    "is_circuit": is_circuit_edge,
                })

    # Top circuit edges for animated pulses
    circuit_edges = sorted(
        [e for e in edges_out if e["is_circuit"]],
        key=lambda e: abs(e["weight"]), reverse=True
    )
    pulse_ids = {id(e) for e in circuit_edges[:25]}
    for e in edges_out:
        e["pulse"] = id(e) in pulse_ids

    payload = {
        "n_cols":        n_cols,
        "n_per_col":     n_per_col,
        "shown_layers":  shown,
        "max_abs_attr":  round(max_abs, 6),
        "neurons":       neurons_out,
        "edges":         edges_out,
        "prompt":        circuit.prompt,
        "target_token":  circuit.target_token,
        "logit_diff":    round(circuit.total_logit_diff, 4),
        "n_circuit":     len(circuit.neurons),
        "n_edges":       len([e for e in edges_out if e["is_circuit"]]),
    }

    _title = title or f"Circuit \u2014 {circuit.target_token.strip()!r}"
    _sub   = (
        f"{payload['n_circuit']} circuit neurons"
        + f" \u00b7 {payload['n_edges']} circuit edges"
        + f" \u00b7 logit_diff {circuit.total_logit_diff:+.3f}"
        + f" \u00b7 layers shown: {shown}"
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
  pointer-events:none;opacity:0;transition:opacity .1s;
  max-width:260px;line-height:1.75;z-index:99;white-space:nowrap;
}
#tip.on{opacity:1}
#footer{
  display:flex;align-items:center;gap:18px;padding:7px 22px;
  border-top:1px solid #0f1d2e;font-size:10px;color:#243344;flex-shrink:0;
}
.leg{display:flex;align-items:center;gap:5px}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
#fhint{margin-left:auto;color:#1c2c3c}
@keyframes pulse{0%,100%{opacity:.92}50%{opacity:.45}}
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
      </defs>
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
    <div class="leg"><div class="dot" style="background:#f97316;border:1px solid #f97316;background:none"></div>Bottleneck</div>
    <div class="leg">
      <svg width="14" height="14" style="flex-shrink:0">
        <circle cx="7" cy="7" r="5" fill="none" stroke="#334155" stroke-width="1.2"/>
      </svg>Background neuron
    </div>
    <div id="fhint">Hover to inspect &nbsp;&middot;&nbsp; Click to lock &nbsp;&middot;&nbsp; Click canvas to unlock</div>
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
    // dark → orange → bright orange-red
    const r = Math.round(60  + 179*t);
    const g = Math.round(20  + 103*Math.pow(t,.7));
    const b = Math.round(5   +  17*(1-t));
    return `rgb(${r},${g},${b})`;
  }
  const s=-t;
  return `rgb(${Math.round(20+39*s)},${Math.round(20+110*s)},${Math.round(30+215*s)})`;
}
function ec(w){ return w>0 ? '#f97316' : '#3b82f6'; }

/* layout */
const wr = document.getElementById('wrap');
let W = wr.clientWidth  || window.innerWidth;
let H = wr.clientHeight || (window.innerHeight - 80);

const NC = D.n_cols, NR = D.n_per_col;
const PAD_X = 70, PAD_Y = 50;
const COL_W = (W - 2*PAD_X) / (NC - 1 || 1);
const ROW_H = (H - 2*PAD_Y) / (NR - 1 || 1);
const R_BG  = Math.max(4, Math.min(9,  ROW_H * 0.32));   // background neuron radius
const R_MAX = Math.max(6, Math.min(14, ROW_H * 0.48));   // max circuit neuron radius
const R_MIN = Math.max(4, Math.min(9,  ROW_H * 0.28));

function cx(col) { return PAD_X + col * COL_W; }
function cy(row) { return PAD_Y + row * ROW_H; }

/* build pos map */
const pos = {};
D.neurons.forEach(n => { pos[n.id] = { x: cx(n.col), y: cy(n.row) }; });
const maxA = D.max_abs_attr;

/* ── edges ─────────────────────────────────────────────────── */
const gEd = document.getElementById('g-edges');
const gPu = document.getElementById('g-pulses');
const maxCW = Math.max(...D.edges.filter(e=>e.is_circuit).map(e=>Math.abs(e.weight)), 1);

D.edges.forEach((edge, i) => {
  const s = pos[edge.src], t = pos[edge.tgt];
  if (!s || !t) return;

  if (!edge.is_circuit) {
    // background edges: very dim straight lines
    mk('line', {
      x1:s.x, y1:s.y, x2:t.x, y2:t.y,
      stroke:'#1a2840', 'stroke-width':.55, opacity:.35,
      class:'bg-edge',
    }, gEd);
    return;
  }

  // circuit edge
  const aw  = Math.abs(edge.weight) / maxCW;
  const col = ec(edge.weight);
  const sw  = 0.6 + 2.0 * aw;
  const op  = 0.25 + 0.65 * aw;

  const pid = `ep${i}`;
  // Slight S-curve so edges don't all overlap exactly
  const dx = t.x - s.x, mx = s.x + dx/2;
  const d  = `M${s.x} ${s.y} C${mx} ${s.y} ${mx} ${t.y} ${t.x} ${t.y}`;

  mk('path', {
    id:pid, d, fill:'none', stroke:col,
    'stroke-width':sw, opacity:op, 'stroke-linecap':'round',
    class:'ced', 'data-src':edge.src, 'data-tgt':edge.tgt,
  }, gEd);

  // Animated pulse on top circuit edges
  if (edge.pulse) {
    const dot = mk('circle', {r:2.2, fill:col, opacity:.95}, gPu);
    const am  = mk('animateMotion', {
      dur:`${.9+(1-aw)*1.8}s`, repeatCount:'indefinite',
      begin:`${((i*.23)%3).toFixed(2)}s`,
      calcMode:'spline', keySplines:'0.42 0 0.58 1', keyTimes:'0;1',
    }, dot);
    mk('mpath', {xl:'#'+pid}, am);
  }
});

/* ── neurons ───────────────────────────────────────────────── */
const gN  = document.getElementById('g-neurons');
const nEl = {};

D.neurons.forEach(n => {
  const {x, y} = pos[n.id];

  if (!n.is_circuit) {
    // Hollow background circle — like the 3B1B style
    mk('circle', {
      cx:x, cy:y, r:R_BG,
      fill:'#0a1220', stroke:'#253548', 'stroke-width':1.1, opacity:.7,
    }, gN);
    return;
  }

  const aa   = Math.abs(n.attribution) / maxA;
  let fill, filt, r;

  if (n.is_sw) {
    fill='#fbbf24'; filt='url(#fs)'; r=R_MAX;
  } else if (n.is_bn) {
    fill=ac(n.attribution,maxA); filt='url(#fm)'; r=R_MIN+4*aa;
  } else {
    fill=ac(n.attribution,maxA);
    filt = aa>.55 ? 'url(#fm)' : 'url(#fd)';
    r    = R_MIN + (R_MAX-R_MIN)*aa;
  }

  const delay = `${(Math.random()*2.2).toFixed(2)}s`;
  const dur   = `${(1.5+Math.random()*1.2).toFixed(2)}s`;

  // Outer halo
  mk('circle',{cx:x,cy:y,r:r+5,fill,
    opacity:.12+.18*aa,filter:filt,class:'cn',
    style:`animation-delay:${delay};animation-duration:${dur}`},gN);

  // Bottleneck ring
  if(n.is_bn||n.is_sw)
    mk('circle',{cx:x,cy:y,r:r+2.5,fill:'none',stroke:fill,
      'stroke-width':1.2,opacity:.5},gN);

  // Body
  const c = mk('circle',{cx:x,cy:y,r,fill,
    stroke:'rgba(255,255,255,0.28)','stroke-width':.8,filter:filt,
    class:'cn cn-body',
    style:`cursor:pointer;animation-delay:${delay};animation-duration:${dur}`,
    'data-id':n.id},gN);
  nEl[n.id]=c;
});

/* ── layer labels ──────────────────────────────────────────── */
const gLb = document.getElementById('g-labels');
D.shown_layers.forEach((layer, ci) => {
  const x = cx(ci);
  // Top label
  const t = mk('text',{x, y:PAD_Y-16,'text-anchor':'middle',
    fill:'#4d7fc4','font-size':10,'font-family':'monospace','font-weight':'bold'},gLb);
  t.textContent = `L${layer}`;
  // Neuron count below label
  const cn_count = D.neurons.filter(n=>n.is_circuit && n.layer===layer).length;
  if(cn_count>0){
    const s=mk('text',{x,y:PAD_Y-5,'text-anchor':'middle',
      fill:'#2d4a6b','font-size':8,'font-family':'monospace'},gLb);
    s.textContent=`${cn_count} active`;
  }
});

/* ── tooltip + interaction ─────────────────────────────────── */
const tip=document.getElementById('tip');
let locked=null;

function showTip(evt,n){
  const col=n.is_sw?'#fbbf24':n.is_bn?'#f97316':ac(n.attribution,maxA);
  const sgn=n.attribution>=0?'+':'';
  tip.innerHTML=`<b style="color:#f0f6fc">${n.label}</b><br>`
    +`attr:&nbsp;<span style="color:${col}">${sgn}${n.attribution.toFixed(5)}</span>`
    +(n.is_bn?`<br><span style="color:#f97316">&#9679; BOTTLENECK</span>`:'')
    +(n.is_sw?`<br><span style="color:#fbbf24">&#9733; SUPER-WEIGHT</span>`:'');
  tip.classList.add('on'); moveTip(evt);
}
function moveTip(evt){
  const r=wr.getBoundingClientRect();
  let tx=evt.clientX-r.left+14, ty=evt.clientY-r.top-12;
  if(tx+270>W)tx-=284; if(ty<0)ty=4;
  tip.style.left=tx+'px'; tip.style.top=ty+'px';
}
function dimEdges(id){
  document.querySelectorAll('.ced').forEach(p=>{
    p.style.opacity=id?(p.dataset.src===id||p.dataset.tgt===id?'0.92':'0.02'):'';
  });
  document.querySelectorAll('.bg-edge').forEach(p=>{
    p.style.opacity=id?'0.04':'';
  });
}
document.querySelectorAll('.cn-body').forEach(c=>{
  c.addEventListener('mouseenter',evt=>{
    if(locked)return;
    const n=D.neurons.find(n=>n.id===c.dataset.id);
    if(n){showTip(evt,n);dimEdges(n.id);}
  });
  c.addEventListener('mousemove',evt=>{if(!locked)moveTip(evt);});
  c.addEventListener('mouseleave',()=>{if(!locked){tip.classList.remove('on');dimEdges(null);}});
  c.addEventListener('click',evt=>{
    evt.stopPropagation();
    const id=c.dataset.id;
    if(locked===id){locked=null;tip.classList.remove('on');dimEdges(null);}
    else{locked=id;const n=D.neurons.find(n=>n.id===id);if(n){showTip(evt,n);dimEdges(id);}}
  });
});
SVG.addEventListener('click',()=>{
  if(locked){locked=null;tip.classList.remove('on');dimEdges(null);}
});
</script>
</body>
</html>"""
