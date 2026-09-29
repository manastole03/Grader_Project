"""Static 3-D aspect-space plot for the CLI (`yelp-agents viz`).

Polarity colours are the diverging red / grey / blue set validated for colour-vision
deficiency; each group also gets its own marker shape so colour is never the only cue.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from yelp_core.schemas import ASPECTS, SENTIMENT_LABELS

COLORS = {"negative": "#e34948", "neutral": "#8a8984", "positive": "#2a78d6"}
SYMBOL = {"negative": "square", "neutral": "diamond", "positive": "circle"}
LABEL = {"negative": "Negative (1–2★)", "neutral": "Neutral (3★)", "positive": "Positive (4–5★)"}


def _fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2f}"


def aspect_space_3d(df: pd.DataFrame, color_by: str = "label", height: int = 640) -> go.Figure:
    """One point per review at (food, service, ambience); not-mentioned aspects sit at 0."""
    fig = go.Figure()
    rng = np.random.default_rng(0)  # tiny fixed jitter: many reviews share exact coordinates
    for pol in SENTIMENT_LABELS:
        d = df[df[color_by] == pol]
        if d.empty:
            continue
        xyz = d[list(ASPECTS)].fillna(0.0).to_numpy(dtype=float) + rng.uniform(-0.025, 0.025, (len(d), 3))
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            name=LABEL[pol] if color_by == "label" else f"Predicted {pol}",
            text=[f"<b>{b}</b><br>{s:.0f}★ actual · lead agent: {o}<br>"
                  f"food {_fmt(f)} · service {_fmt(sv)} · ambience {_fmt(am)}"
                  for b, s, o, f, sv, am in zip(d["business"], d["stars"], d["overall"],
                                                d["food"], d["service"], d["ambience"])],
            hovertemplate="%{text}<extra></extra>",
            marker={"size": 5, "color": COLORS[pol], "symbol": SYMBOL[pol], "opacity": 0.9,
                    "line": {"color": "#fcfcfb", "width": 1}},
        ))
    axis = lambda t: {"title": {"text": t}, "range": [-1.1, 1.1], "tickvals": [-1, -0.5, 0, 0.5, 1]}
    fig.update_layout(
        height=height, margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title="Yelp reviews in 3-D aspect space (scores computed in Python)",
        scene={"xaxis": axis("Food"), "yaxis": axis("Service"), "zaxis": axis("Ambience"), "aspectmode": "cube"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.98, "x": 0},
    )
    return fig
