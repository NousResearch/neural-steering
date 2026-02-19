"""neuron_steer.visualizer — Interactive HTML dashboards for circuit analysis.

Generates self-contained Plotly HTML files (no server needed, just open in
a browser). Requires plotly: pip install plotly

Usage
-----
    from neuron_steer.visualizer import visualize

    # Works on Circuit or CircuitGraph — auto-dispatches
    path = visualize(circuit)
    path = visualize(graph, title="Capitals hourglass")

    # Or explicitly
    from neuron_steer.visualizer import visualize_circuit, visualize_graph
    path = visualize_circuit(circuit, output="my_circuit.html")
    path = visualize_graph(graph, output="my_graph.html")
"""

import os
import webbrowser
import tempfile
from typing import Optional, Union

from neuron_steer.core import Circuit, CircuitGraph


# ============================================================
# Public API
# ============================================================

def visualize(
    obj: Union[Circuit, CircuitGraph],
    output: Optional[str] = None,
    open_browser: bool = True,
    title: Optional[str] = None,
) -> str:
    """Visualize a Circuit or CircuitGraph as an interactive HTML dashboard.

    Auto-dispatches to visualize_circuit or visualize_graph based on type.
    Returns the path to the generated HTML file.
    """
    if isinstance(obj, CircuitGraph):
        return visualize_graph(obj, output=output, open_browser=open_browser, title=title)
    elif isinstance(obj, Circuit):
        return visualize_circuit(obj, output=output, open_browser=open_browser, title=title)
    else:
        raise TypeError(f"Expected Circuit or CircuitGraph, got {type(obj)}")


def visualize_circuit(
    circuit: Circuit,
    output: Optional[str] = None,
    open_browser: bool = True,
    title: Optional[str] = None,
) -> str:
    """Three-panel interactive dashboard for a Circuit.

    Panels:
      1. Layer distribution — neuron count + attribution mass per layer
      2. Neuron scatter    — every neuron plotted at (layer, attribution),
                             size ∝ |attribution|, color = layer
      3. Attribution waterfall — top 25 neurons ranked by |attribution|

    Returns the path to the generated HTML file.
    """
    _require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not circuit.neurons:
        raise ValueError("Circuit is empty — nothing to visualize.")

    # ── data extraction ───────────────────────────────────────────
    n_layers = max(n.layer for n in circuit.neurons) + 1
    by_layer = circuit.by_layer()
    layers = list(range(n_layers))

    layer_counts  = [len(by_layer.get(l, [])) for l in layers]
    layer_attr_mass = [sum(abs(a) for _, a in by_layer.get(l, [])) for l in layers]
    total_attr = sum(layer_attr_mass) or 1.0
    layer_attr_frac = [m / total_attr for m in layer_attr_mass]

    sc_layers, sc_attrs, sc_hover, sc_sizes = [], [], [], []
    max_abs = max(abs(a) for a in circuit.neurons.values()) or 1.0
    for nidx, attr in circuit.neurons.items():
        sc_layers.append(nidx.layer)
        sc_attrs.append(attr)
        sc_hover.append(
            f"<b>L{nidx.layer} / N{nidx.neuron}</b><br>"
            f"position: {nidx.position}<br>"
            f"attribution: {attr:+.5f}"
        )
        sc_sizes.append(4 + 18 * abs(attr) / max_abs)

    top25 = circuit.top(25)
    wf_labels = [f"L{n.layer}/N{n.neuron}" for n, _ in top25]
    wf_vals   = [a for _, a in top25]
    wf_colors = ["#e74c3c" if v > 0 else "#3b82f6" for v in wf_vals]

    # ── build figure ─────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Layer distribution",
            "All neurons  (size = |attr|, color = layer)",
            "Top 25 by |attribution|",
        ),
        column_widths=[0.25, 0.42, 0.33],
        horizontal_spacing=0.07,
    )

    # Panel 1a: neuron count bars
    fig.add_trace(go.Bar(
        x=layers, y=layer_counts,
        name="Neurons",
        marker_color="#4ecdc4", opacity=0.85,
        hovertemplate="Layer %{x}<br>%{y} neurons<extra></extra>",
    ), row=1, col=1)

    # Panel 1b: attribution mass line (normalised to same scale)
    max_count = max(layer_counts) or 1
    scaled_attr = [f * max_count for f in layer_attr_frac]
    fig.add_trace(go.Scatter(
        x=layers, y=scaled_attr,
        name="Attribution mass",
        mode="lines+markers",
        marker=dict(size=4, color="#f97316"),
        line=dict(color="#f97316", width=2),
        hovertemplate="Layer %{x}<br>attr fraction: %{customdata:.3f}<extra></extra>",
        customdata=layer_attr_frac,
    ), row=1, col=1)

    # Panel 2: scatter
    fig.add_trace(go.Scatter(
        x=sc_layers, y=sc_attrs,
        mode="markers",
        text=sc_hover, hoverinfo="text",
        marker=dict(
            size=sc_sizes,
            color=sc_layers,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Layer", thickness=12, x=0.69, len=0.75),
            line=dict(width=0.4, color="#1a1a1a"),
            opacity=0.8,
        ),
        name="Neurons",
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#555566", row=1, col=2)

    # Panel 3: waterfall (flipped so top neuron is first)
    fig.add_trace(go.Bar(
        y=wf_labels[::-1], x=wf_vals[::-1],
        orientation="h",
        marker_color=wf_colors[::-1],
        text=[f"{v:+.4f}" for v in wf_vals[::-1]],
        textposition="outside",
        hovertemplate="%{y}<br>%{x:+.4f}<extra></extra>",
        name="Attribution",
    ), row=1, col=3)

    # ── layout ───────────────────────────────────────────────────
    _title = title or f"Circuit — target: {circuit.target_token.strip()!r}"
    subtitle = (
        f"{len(circuit.neurons)} neurons | "
        f"logit_diff = {circuit.total_logit_diff:+.3f} | "
        f"\"{circuit.prompt[:70]}\""
    )

    fig.update_layout(
        title=dict(text=f"<b>{_title}</b><br><sup>{subtitle}</sup>", x=0.5),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(family="'JetBrains Mono', 'Fira Code', monospace", size=11, color="#c9d1d9"),
        height=540,
        showlegend=False,
        margin=dict(t=90, b=50, l=50, r=60),
    )
    fig.update_xaxes(title_text="Layer", gridcolor="#21262d", row=1, col=1)
    fig.update_yaxes(title_text="Count", gridcolor="#21262d", row=1, col=1)
    fig.update_xaxes(title_text="Layer", gridcolor="#21262d", row=1, col=2)
    fig.update_yaxes(title_text="Attribution", gridcolor="#21262d", row=1, col=2)
    fig.update_xaxes(title_text="Attribution", gridcolor="#21262d", row=1, col=3)
    fig.update_yaxes(tickfont=dict(size=9), row=1, col=3)

    return _save_and_open(fig, output, open_browser, prefix="circuit")


def visualize_graph(
    graph: CircuitGraph,
    output: Optional[str] = None,
    open_browser: bool = True,
    title: Optional[str] = None,
) -> str:
    """Four-panel interactive dashboard for a CircuitGraph.

    Panels:
      1. Sankey diagram   — layer-to-layer attribution flow (reveals hourglass)
      2. Neuron scatter   — same as visualize_circuit but with bottlenecks/
                            super-weights highlighted in a different color
      3. Hub analysis     — top source hubs (fan-out) vs target hubs (fan-in)
      4. Top edges        — highest-weight individual neuron→neuron connections

    Returns the path to the generated HTML file.
    """
    _require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    circuit = graph.circuit
    if not graph.edges:
        raise ValueError("CircuitGraph has no edges — run discover_edges() first.")

    # ── shared circuit data (same as visualize_circuit panel 2) ──
    n_layers = max(n.layer for n in circuit.neurons) + 1
    max_abs = max(abs(a) for a in circuit.neurons.values()) or 1.0

    bn_set = {(l, n) for (l, n), _, _ in graph.bottleneck()}
    sw_list = graph.detect_super_weights()
    sw_set  = {(l, n) for (l, n), _, _, _ in sw_list}

    sc_layers, sc_attrs, sc_hover, sc_sizes, sc_colors = [], [], [], [], []
    for nidx, attr in circuit.neurons.items():
        key = (nidx.layer, nidx.neuron)
        sc_layers.append(nidx.layer)
        sc_attrs.append(attr)
        sc_sizes.append(4 + 18 * abs(attr) / max_abs)

        if key in sw_set:
            color = "#fbbf24"   # gold — super weight
            role = "SUPER WEIGHT"
        elif key in bn_set:
            color = "#f97316"   # orange — bottleneck
            role = "BOTTLENECK"
        else:
            color = "#4ecdc4"
            role = ""

        sc_colors.append(color)
        sc_hover.append(
            f"<b>L{nidx.layer} / N{nidx.neuron}</b><br>"
            f"attribution: {attr:+.5f}<br>"
            + (f"<b>{role}</b>" if role else "")
        )

    # ── Sankey data ───────────────────────────────────────────────
    flow = graph.layer_flow()
    unique_layers = sorted({l for pair in flow for l in pair})
    layer_to_idx  = {l: i for i, l in enumerate(unique_layers)}

    sankey_src, sankey_tgt, sankey_val, sankey_color = [], [], [], []
    for (src, tgt), w in flow.items():
        if w > 0:
            sankey_src.append(layer_to_idx[src])
            sankey_tgt.append(layer_to_idx[tgt])
            sankey_val.append(w)
            # Edges crossing more layers get a warmer color
            gap = tgt - src
            sankey_color.append(f"rgba(231,76,60,{min(0.85, 0.2 + gap * 0.15):.2f})")

    sankey_labels = [f"L{l}" for l in unique_layers]

    # ── Hub analysis ─────────────────────────────────────────────
    hubs = graph.hub_analysis()
    src_hubs = hubs["source_hubs"][:12]
    tgt_hubs = hubs["target_hubs"][:12]

    sh_labels = [f"L{l}/N{n}" for (l, n), _, _ in src_hubs]
    sh_vals   = [deg for _, deg, _ in src_hubs]
    sh_colors = ["#fbbf24" if (l, n) in sw_set else "#e74c3c" for (l, n), _, _ in src_hubs]

    th_labels = [f"L{l}/N{n}" for (l, n), _, _ in tgt_hubs]
    th_vals   = [deg for _, deg, _ in tgt_hubs]
    th_colors = ["#f97316" if (l, n) in bn_set else "#3b82f6" for (l, n), _, _ in tgt_hubs]

    # ── Top edges ────────────────────────────────────────────────
    top_edges = graph.top_edges(20)
    te_labels = [f"L{e.source.layer}/N{e.source.neuron} → L{e.target.layer}/N{e.target.neuron}"
                 for e in top_edges]
    te_vals   = [e.weight for e in top_edges]
    te_colors = ["#e74c3c" if w > 0 else "#3b82f6" for w in te_vals]

    # ── Build figure ──────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Layer-to-layer attribution flow  (Sankey)",
            "Circuit neurons  (orange=bottleneck  gold=super-weight)",
            "Hub analysis  (fan-out vs fan-in)",
            "Top 20 edges  by |weight|",
        ),
        specs=[
            [{"type": "sankey"}, {"type": "scatter"}],
            [{"type": "bar"},    {"type": "bar"}],
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    # Panel 1: Sankey
    fig.add_trace(go.Sankey(
        arrangement="snap",
        node=dict(
            label=sankey_labels,
            color="#4ecdc4",
            pad=12, thickness=18,
            line=dict(color="#0d1117", width=0.5),
        ),
        link=dict(
            source=sankey_src,
            target=sankey_tgt,
            value=sankey_val,
            color=sankey_color,
            hovertemplate="%{source.label} → %{target.label}<br>flow: %{value:.2f}<extra></extra>",
        ),
    ), row=1, col=1)

    # Panel 2: Scatter with role-colored neurons
    fig.add_trace(go.Scatter(
        x=sc_layers, y=sc_attrs,
        mode="markers",
        text=sc_hover, hoverinfo="text",
        marker=dict(
            size=sc_sizes, color=sc_colors,
            line=dict(width=0.4, color="#1a1a1a"),
            opacity=0.85,
        ),
        name="Neurons",
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dot", line_color="#555566", row=1, col=2)

    # Panel 3a: source hubs
    fig.add_trace(go.Bar(
        y=sh_labels[::-1], x=sh_vals[::-1],
        orientation="h",
        marker_color=sh_colors[::-1],
        name="Out-degree",
        hovertemplate="%{y}<br>out-degree: %{x}<extra></extra>",
    ), row=2, col=1)

    # Panel 3b: target hubs (superimposed as a second color)
    fig.add_trace(go.Bar(
        y=th_labels[::-1], x=th_vals[::-1],
        orientation="h",
        marker_color=th_colors[::-1],
        name="In-degree",
        hovertemplate="%{y}<br>in-degree: %{x}<extra></extra>",
        visible=True,
    ), row=2, col=1)

    # Panel 4: top edges
    fig.add_trace(go.Bar(
        y=te_labels[::-1], x=te_vals[::-1],
        orientation="h",
        marker_color=te_colors[::-1],
        text=[f"{v:+.3f}" for v in te_vals[::-1]],
        textposition="outside",
        name="Edge weight",
        hovertemplate="%{y}<br>weight: %{x:+.4f}<extra></extra>",
    ), row=2, col=2)

    # ── layout ────────────────────────────────────────────────────
    _title = title or f"Circuit graph — target: {circuit.target_token.strip()!r}"
    subtitle = (
        f"{len(circuit.neurons)} neurons | "
        f"{len(graph.edges)} edges | "
        f"{len(bn_set)} bottleneck | "
        f"{len(sw_set)} super-weight"
    )

    fig.update_layout(
        title=dict(text=f"<b>{_title}</b><br><sup>{subtitle}</sup>", x=0.5),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(family="'JetBrains Mono', 'Fira Code', monospace", size=11, color="#c9d1d9"),
        height=900,
        showlegend=False,
        margin=dict(t=90, b=50, l=50, r=60),
        barmode="overlay",
    )
    fig.update_xaxes(title_text="Layer", gridcolor="#21262d", row=1, col=2)
    fig.update_yaxes(title_text="Attribution", gridcolor="#21262d", row=1, col=2)
    fig.update_xaxes(title_text="Degree", gridcolor="#21262d", row=2, col=1)
    fig.update_yaxes(tickfont=dict(size=9), row=2, col=1)
    fig.update_xaxes(title_text="Edge weight", gridcolor="#21262d", row=2, col=2)
    fig.update_yaxes(tickfont=dict(size=8), row=2, col=2)

    return _save_and_open(fig, output, open_browser, prefix="circuit_graph")


# ============================================================
# Helpers
# ============================================================

def _require_plotly():
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError(
            "plotly is required for visualization.\n"
            "Install with: pip install plotly"
        )


def _save_and_open(fig, output: Optional[str], open_browser: bool, prefix: str) -> str:
    import plotly.io as pio

    if output is None:
        fd, output = tempfile.mkstemp(prefix=f"neuron_{prefix}_", suffix=".html")
        os.close(fd)

    pio.write_html(
        fig, file=output,
        include_plotlyjs="cdn",   # ~1KB stub, fetches plotly from CDN
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": True},
    )

    print(f"Saved to: {output}")
    if open_browser:
        webbrowser.open(f"file://{os.path.abspath(output)}")

    return output
