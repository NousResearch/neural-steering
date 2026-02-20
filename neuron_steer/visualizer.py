"""neuron_steer.visualizer — 3Blue1Brown-style neural circuit visualizer.

Generates a fully self-contained HTML file. No server, no pip installs beyond
the base package. Just open the file in a browser.

Visual design
-------------
  - Every model layer is a column; neurons are circles
  - Circuit neurons glow and are colored by attribution value
      blue = suppressing the target token
      red/orange = promoting the target token
  - Bottleneck neurons pulse with an outer ring
  - Super-weight neurons are rendered in gold
  - When a CircuitGraph is provided, bezier edges connect neurons with
    animated dot pulses traveling along them (faster = stronger edge)
  - Hover a neuron to inspect it; click to lock its connections in view

Usage
-----
    from neuron_steer.visualizer import visualize_network

    circuit = steerer.discover_circuit("What is the capital of France?", " Paris")
    visualize_network(circuit)                        # neurons only

    graph = steerer.discover_edges("...", circuit)
    visualize_network(graph, title="capitals hourglass")  # + animated edges

    # save without opening
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
    max_edges: int = 60,
    n_rows: int = 18,
) -> str:
    """Generate a 3B1B-style neural network circuit visualization.

    Accepts either a Circuit (neurons only) or CircuitGraph (neurons + edges).
    Returns the path to the generated HTML file.
    """
    if isinstance(obj, CircuitGraph):
        circuit   = obj.circuit
        bn_set    = {(l, n) for (l, n), _, _ in obj.bottleneck()}
        sw_list   = obj.detect_super_weights()
        sw_set    = {(l, n) for (l, n), _, _, _ in sw_list}
        raw_edges = obj.top_edges(max_edges)
        max_ew    = max((abs(e.weight) for e in raw_edges), default=1.0)
    else:
        circuit   = obj
        bn_set    = set()
        sw_set    = set()
        raw_edges = []
        max_ew    = 1.0

    n_layers = max(n.layer for n in circuit.neurons) + 1
    max_abs  = max(abs(a) for a in circuit.neurons.values()) or 1.0

    # Group + sort circuit neurons per layer
    by_layer: dict = {}
    for nidx, attr in circuit.neurons.items():
        by_layer.setdefault(nidx.layer, []).append((nidx, attr))
    for layer in by_layer:
        by_layer[layer].sort(key=lambda x: abs(x[1]), reverse=True)

    # Assign rows and build neuron list + id map
    neurons_out: list = []
    id_map: dict = {}   # (layer, position, neuron) -> id

    for layer_idx in range(n_layers):
        cn   = by_layer.get(layer_idx, [])
        n_cn = min(len(cn), n_rows)
        start     = max(0, (n_rows - n_cn) // 2)
        occupied  = set()

        for i, (nidx, attr) in enumerate(cn[:n_cn]):
            row = start + i
            nid = f"c_{nidx.layer}_{nidx.position}_{nidx.neuron}"
            occupied.add(row)
            id_map[(nidx.layer, nidx.position, nidx.neuron)] = nid
            neurons_out.append({
                "id":          nid,
                "layer":       nidx.layer,
                "neuron":      nidx.neuron,
                "attribution": round(attr, 6),
                "is_circuit":  True,
                "is_bn":       (nidx.layer, nidx.neuron) in bn_set,
                "is_sw":       (nidx.layer, nidx.neuron) in sw_set,
                "label":       f"L{nidx.layer} / N{nidx.neuron}",
                "row":         row,
            })

        # Background neurons fill remaining rows
        bg_rows  = [r for r in range(n_rows) if r not in occupied]
        step     = max(1, len(bg_rows) // 8)
        for row in bg_rows[::step][:8]:
            neurons_out.append({
                "id":          f"bg_{layer_idx}_{row}",
                "layer":       layer_idx,
                "neuron":      -1,
                "attribution": 0.0,
                "is_circuit":  False,
                "is_bn":       False,
                "is_sw":       False,
                "label":       f"L{layer_idx} background",
                "row":         row,
            })

    # Build edge list
    edges_out: list = []
    for e in raw_edges:
        src_id = id_map.get((e.source.layer, e.source.position, e.source.neuron))
        tgt_id = id_map.get((e.target.layer, e.target.position, e.target.neuron))
        if src_id and tgt_id:
            edges_out.append({
                "src":    src_id,
                "tgt":    tgt_id,
                "weight": round(e.weight, 6),
            })

    payload = {
        "n_layers":        n_layers,
        "n_rows":          n_rows,
        "max_abs_attr":    round(max_abs, 6),
        "max_edge_weight": round(max_ew, 6),
        "neurons":         neurons_out,
        "edges":           edges_out,
        "prompt":          circuit.prompt,
        "target_token":    circuit.target_token,
        "logit_diff":      round(circuit.total_logit_diff, 4),
        "n_circuit":       len(circuit.neurons),
        "n_edges":         len(edges_out),
    }

    _title = title or f"Circuit \u2014 {circuit.target_token.strip()!r}"
    _sub   = (
        f"{payload['n_circuit']} circuit neurons"
        + (f" \u00b7 {payload['n_edges']} edges" if edges_out else "")
        + f" \u00b7 logit_diff {circuit.total_logit_diff:+.3f}"
        + f" \u00b7 \u201c{circuit.prompt[:90]}\u201d"
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
# HTML / JS template
# ─────────────────────────────────────────────────────────────────────────────

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%%TITLE%%</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  background:#060710;color:#c9d1d9;
  font-family:'JetBrains Mono','Fira Code','Courier New',monospace;
  overflow:hidden;user-select:none;
}
#shell{display:flex;flex-direction:column;width:100vw;height:100vh}
#hdr{padding:13px 20px 9px;border-bottom:1px solid #111d2e;flex-shrink:0}
#hdr-t{font-size:16px;font-weight:700;color:#f0f6fc;letter-spacing:.04em}
#hdr-s{font-size:10px;color:#3d5166;margin-top:3px;line-height:1.5}
#wrap{flex:1;position:relative;overflow:hidden}
svg{width:100%;height:100%;display:block}
#tip{
  position:absolute;background:#0b1726;border:1px solid #1e3a5f;
  border-radius:7px;padding:9px 13px;font-size:11px;color:#e2e8f0;
  pointer-events:none;opacity:0;transition:opacity .1s;
  max-width:260px;line-height:1.75;z-index:99;
}
#tip.on{opacity:1}
#footer{
  display:flex;align-items:center;gap:18px;
  padding:7px 20px;border-top:1px solid #111d2e;
  font-size:10px;color:#2d4155;flex-shrink:0;
}
.leg{display:flex;align-items:center;gap:5px}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
#fhint{margin-left:auto;color:#243344}
@keyframes pulse{0%,100%{opacity:.88}50%{opacity:.4}}
.cn{animation:pulse 2.4s ease-in-out infinite}
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
          <feGaussianBlur stdDeviation="11" result="b2"/>
          <feMerge><feMergeNode in="b2"/><feMergeNode in="b1"/>
          <feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="fs" x="-200%" y="-200%" width="500%" height="500%">
          <feGaussianBlur stdDeviation="7"  result="b1"/>
          <feGaussianBlur stdDeviation="18" result="b2"/>
          <feMerge><feMergeNode in="b2"/><feMergeNode in="b2"/>
          <feMergeNode in="b1"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="fe" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="1.6" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <g id="g-spines"></g>
      <g id="g-bg"></g>
      <g id="g-edges"></g>
      <g id="g-pulses"></g>
      <g id="g-neurons"></g>
      <g id="g-labels"></g>
    </svg>
    <div id="tip"></div>
  </div>
  <div id="footer">
    <div class="leg"><div class="dot" style="background:#ef4444"></div>Positive attr</div>
    <div class="leg"><div class="dot" style="background:#3b82f6"></div>Negative attr</div>
    <div class="leg"><div class="dot" style="background:#f97316"></div>Bottleneck</div>
    <div class="leg"><div class="dot" style="background:#fbbf24"></div>Super-weight</div>
    <div class="leg"><div class="dot" style="background:#111d2e"></div>Background</div>
    <div id="fhint">Hover to inspect &nbsp;&middot;&nbsp; Click to lock connections &nbsp;&middot;&nbsp; Click canvas to unlock</div>
  </div>
</div>
<script>
const D  = %%DATA%%;
const NS = 'http://www.w3.org/2000/svg';
const XL = 'http://www.w3.org/1999/xlink';
const SVG = document.getElementById('viz');

function mk(tag, a, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k,v] of Object.entries(a||{})) {
    if (k==='xl') e.setAttributeNS(XL,'href',v); else e.setAttribute(k,v);
  }
  parent && parent.appendChild(e);
  return e;
}

/* attribution → color: blue (neg) → dark → red (pos) */
function ac(a, mx) {
  const t = Math.max(-1, Math.min(1, a/(mx||1)));
  if (t>=0) {
    return `rgb(${Math.round(20+219*t)},${Math.round(8+60*t)},${Math.round(12*( 1-t))})`;
  }
  const s=-t;
  return `rgb(${Math.round(20+39*s)},${Math.round(8+122*s)},${Math.round(12+234*s)})`;
}
function ec(w){ return w>0?'#ef4444':'#3b82f6'; }

/* layout */
const wr = document.getElementById('wrap');
let W = wr.clientWidth||window.innerWidth;
let H = wr.clientHeight||(window.innerHeight-90);
const PX=55, PY=38, NL=D.n_layers, NR=D.n_rows;
const lx = l => PX + (l/(NL-1))*(W-2*PX);
const ry = r => PY + (r/(NR-1))*(H-2*PY);

const pos={};
D.neurons.forEach(n=>{ pos[n.id]={x:lx(n.layer),y:ry(n.row)}; });

const maxA=D.max_abs_attr;

/* spines */
const gSp=document.getElementById('g-spines');
for(let l=0;l<NL;l++){
  mk('line',{x1:lx(l),y1:PY,x2:lx(l),y2:H-PY,
    stroke:'#0c1520','stroke-width':1,opacity:.6},gSp);
}

/* labels */
const gLb=document.getElementById('g-labels');
const cLayers=new Set(D.neurons.filter(n=>n.is_circuit).map(n=>n.layer));
for(let l=0;l<NL;l++){
  const hot=cLayers.has(l);
  if(!hot && l%4!==0 && l!==NL-1) continue;
  const t=mk('text',{x:lx(l),y:13,'text-anchor':'middle',
    fill:hot?'#4d7fc4':'#1b2a38',
    'font-size':hot?9.5:7.5,'font-family':'monospace',
    'font-weight':hot?'bold':'normal'},gLb);
  t.textContent='L'+l;
}

/* background neurons */
const gBg=document.getElementById('g-bg');
D.neurons.filter(n=>!n.is_circuit).forEach(n=>{
  const {x,y}=pos[n.id];
  mk('circle',{cx:x,cy:y,r:2.2,fill:'#101c2c',stroke:'#1a2a3a',
    'stroke-width':.5,opacity:.5},gBg);
});

/* edges */
const gEd=document.getElementById('g-edges');
const gPu=document.getElementById('g-pulses');
D.edges.forEach((edge,i)=>{
  const s=pos[edge.src], t=pos[edge.tgt];
  if(!s||!t) return;
  const aw=Math.abs(edge.weight)/D.max_edge_weight;
  const col=ec(edge.weight);
  const dx=t.x-s.x;
  const d=`M${s.x} ${s.y} C${s.x+dx*.42} ${s.y} ${s.x+dx*.58} ${t.y} ${t.x} ${t.y}`;
  const pid=`ep${i}`;
  mk('path',{id:pid,d,fill:'none',stroke:col,
    'stroke-width':.4+2.2*aw, opacity:.07+.55*aw,
    'stroke-linecap':'round',
    filter:aw>.4?'url(#fe)':'',
    class:'ced','data-src':edge.src,'data-tgt':edge.tgt},gEd);
  if(aw>0.18){
    const dot=mk('circle',{r:2,fill:col,opacity:.9},gPu);
    const am=mk('animateMotion',{
      dur:`${1.1+(1-aw)*2.2}s`,repeatCount:'indefinite',
      begin:`${((i*.29)%3.2).toFixed(2)}s`,
      calcMode:'spline',keySplines:'0.42 0 0.58 1',keyTimes:'0;1'},dot);
    mk('mpath',{xl:'#'+pid},am);
  }
});

/* circuit neurons */
const gN=document.getElementById('g-neurons');
const nEls={};
D.neurons.filter(n=>n.is_circuit).forEach(n=>{
  const {x,y}=pos[n.id];
  const aa=Math.abs(n.attribution)/maxA;
  let fill,filt,r;
  if(n.is_sw){    fill='#fbbf24'; filt='url(#fs)'; r=9; }
  else if(n.is_bn){fill='#f97316'; filt='url(#fs)'; r=6+4*aa; }
  else{           fill=ac(n.attribution,maxA); filt=aa>.5?'url(#fm)':'url(#fd)'; r=3+7*aa; }
  const delay=`${(Math.random()*2.4).toFixed(2)}s`;
  const dur=`${(1.5+Math.random()*1.3).toFixed(2)}s`;
  /* halo */
  mk('circle',{cx:x,cy:y,r:r+5,fill,
    opacity:.1+.2*aa, filter:filt,
    class:'cn',style:`animation-delay:${delay};animation-duration:${dur}`},gN);
  /* ring for bottleneck/sw */
  if(n.is_bn||n.is_sw)
    mk('circle',{cx:x,cy:y,r:r+2,fill:'none',stroke:fill,
      'stroke-width':1.1,opacity:.4},gN);
  /* body */
  const c=mk('circle',{cx:x,cy:y,r,fill,
    stroke:'rgba(255,255,255,0.22)','stroke-width':.6,filter:filt,
    class:'cn cn-body',style:`cursor:pointer;animation-delay:${delay};animation-duration:${dur}`,
    'data-id':n.id},gN);
  nEls[n.id]=c;
});

/* tooltip + interaction */
const tip=document.getElementById('tip');
let locked=null;
function showTip(evt,n){
  const col=n.is_sw?'#fbbf24':n.is_bn?'#f97316':ac(n.attribution,maxA);
  const sgn=n.attribution>=0?'+':'';
  tip.innerHTML=`<b style="color:#f0f6fc">${n.label}</b><br>`
    +`attr: <span style="color:${col}">${sgn}${n.attribution.toFixed(5)}</span>`
    +(n.is_bn?`<br><span style="color:#f97316">&#9679; BOTTLENECK</span>`:'')
    +(n.is_sw?`<br><span style="color:#fbbf24">&#9733; SUPER-WEIGHT</span>`:'');
  tip.classList.add('on');
  moveTip(evt);
}
function moveTip(evt){
  const r=document.getElementById('wrap').getBoundingClientRect();
  let tx=evt.clientX-r.left+14, ty=evt.clientY-r.top-12;
  if(tx+270>W) tx-=284;
  if(ty<0) ty=4;
  tip.style.left=tx+'px'; tip.style.top=ty+'px';
}
function hideTip(){ if(!locked) tip.classList.remove('on'); }
function dimEdges(id){
  document.querySelectorAll('.ced').forEach(p=>{
    p.style.opacity = id ? (p.dataset.src===id||p.dataset.tgt===id?'0.88':'0.02') : '';
  });
}
document.querySelectorAll('.cn-body').forEach(c=>{
  c.addEventListener('mouseenter',evt=>{
    if(locked) return;
    const n=D.neurons.find(n=>n.id===c.dataset.id);
    if(n){showTip(evt,n);dimEdges(n.id);}
  });
  c.addEventListener('mousemove', evt=>{ if(!locked) moveTip(evt); });
  c.addEventListener('mouseleave',()=>{ if(!locked){hideTip();dimEdges(null);} });
  c.addEventListener('click',evt=>{
    evt.stopPropagation();
    const id=c.dataset.id;
    if(locked===id){ locked=null; tip.classList.remove('on'); dimEdges(null); }
    else {
      locked=id;
      const n=D.neurons.find(n=>n.id===id);
      if(n){showTip(evt,n);dimEdges(id);}
    }
  });
});
SVG.addEventListener('click',()=>{
  if(locked){ locked=null; tip.classList.remove('on'); dimEdges(null); }
});
</script>
</body>
</html>"""
