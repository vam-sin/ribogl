import argparse
import re
import os
from collections import defaultdict
from itertools import chain

import dash_ag_grid as dag
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy as sp
from Bio import SeqIO
from dash import Dash, Input, Output, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, wasserstein_distance
from waitress import serve
from tqdm.auto import trange

from configuration import PLOT_COLORS


ATT_H5_FNAME = "RiboGL_Attributions.h5"
DATA_DIRPATH = "data"
ENSEMBL_DIRPATH = "data"


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def extract_gene_symbol(input_string):
    pattern = r"gene_symbol:(\S+)"  # (\S+) captures everything until a space
    match = re.search(pattern, input_string)
    return match.group(1)


data = []
with open(
    os.path.join(ENSEMBL_DIRPATH, "ensembl.cds.fa"),
    mode="r",
) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        data.append(
            [record.id, str(record.seq), extract_gene_symbol(record.description)]
        )

# Create transcripts to sequences mapping
df_trans_to_seq = pd.DataFrame(data, columns=["transcript", "sequence", "gene_symbol"])


def _get_y_true(idx):
    with h5py.File(
        os.path.join(DATA_DIRPATH, ATT_H5_FNAME),
        "r",
    ) as f:
        return np.array(f["y_true"][idx])


def _get_trancript(idx):
    with h5py.File(
        os.path.join(DATA_DIRPATH, ATT_H5_FNAME),
        "r",
    ) as f:
        return np.array(f["transcript"][idx])


def _get_y_pred(idx):
    with h5py.File(
        os.path.join(DATA_DIRPATH, ATT_H5_FNAME),
        "r",
    ) as f:
        return np.array(f["y_pred"][idx])


def _get_node_attr(idx):
    with h5py.File(
        os.path.join(DATA_DIRPATH, ATT_H5_FNAME),
        "r",
    ) as f:
        return np.array(f["node_attr"][idx])


def _get_edge_attr(idx):
    with h5py.File(
        os.path.join(DATA_DIRPATH, ATT_H5_FNAME),
        "r",
    ) as f:
        return np.array(f["edge_attr"][idx])


def _get_idxs_attr(idx):
    with h5py.File(
        os.path.join(DATA_DIRPATH, ATT_H5_FNAME),
        "r",
    ) as f:
        return f["edge_index"][idx]


with h5py.File(
    os.path.join(DATA_DIRPATH, ATT_H5_FNAME),
    "r",
) as f:
    n_sequences = f["y_true"].shape[0]

metrics = defaultdict(list)
for idx in trange(n_sequences, desc="Preprocessing data"):
    y_true, y_pred = _get_y_true(idx), _get_y_pred(idx)
    mask = ~np.isnan(y_true)
    metrics["transcript"].append(str(_get_trancript(idx), encoding="utf-8"))
    metrics["PCC"].append(pearsonr(y_true[mask], y_pred[mask])[0])

rowData = pd.DataFrame(metrics).merge(df_trans_to_seq).assign(id=lambda df: df.index)

app = Dash()


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (rho, theta)


def pol2cart(rho, theta):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return (x, y)


def serve_layout():
    layout = html.Div(
        [
            html.H1("RiboGL Post-Hoc Analysis"),
            html.H2("Docs"),
            html.Ol(
                [
                    html.Li("Select a gene in the Sample Selector table below."),
                    html.Li(
                        "Select a ribosome count (codon position) to explain on the Circular Plot, by clicking on a green dot."
                    ),
                ]
            ),
            html.H2("Sample Selector"),
            html.Div(
                dag.AgGrid(
                    id="datatable",
                    rowData=rowData.to_dict("records"),
                    columnDefs=[
                        {
                            "field": col,
                            "sortable": col in rowData.select_dtypes("number"),
                            "hide": col in ["id", "sequence"],
                            "resizable": True,
                            "columnSize": "responsiveSizeToFit",
                        }
                        for col in rowData.columns
                    ],
                    dashGridOptions=dict(rowSelection="single", ensureDomOrder=True),
                    style={"height": 500, "width": 800},
                )
            ),
            html.H2("Circular Plot with Secondary Structure"),
            dcc.Graph(
                id="mrna",
                config={
                    "toImageButtonOptions": {
                        "format": "svg",
                    }
                },
            ),
            html.H2("mRNA Sequence Plot"),
            dcc.Graph(
                id="sequence",
                config={
                    "toImageButtonOptions": {
                        "format": "svg",
                    }
                },
            ),
        ]
    )
    return layout


app.layout = serve_layout


@callback(
    Output("mrna", "figure"),
    [Input("datatable", "selectedRows"), Input("mrna", "clickData")],
)
def plot_graph(selected_row, clickData):
    if ctx.triggered_id == "datatable":
        clickData = None
    sample_idx = selected_row[0]["id"] if selected_row else 0
    y_true = _get_y_true(sample_idx)  # [:-1]
    y_pred = _get_y_pred(sample_idx)
    n_codons = len(y_true)
    idxs_attr = _get_idxs_attr(sample_idx).reshape(2, -1).astype(int)
    adj_matrix = np.zeros((n_codons, n_codons))
    for x, y in idxs_attr.T:
        adj_matrix[x, y] = 1
    nx_graph = nx.from_numpy_array(adj_matrix)
    node_attr = _get_node_attr(sample_idx).reshape(n_codons, n_codons)
    edge_attr = _get_edge_attr(sample_idx).reshape(n_codons, 3, -1)
    edge_attr = np.swapaxes(edge_attr, 1, 2)
    sequence = rowData.iloc[sample_idx].sequence
    sequence = re.findall("...", sequence)

    base_r = 0.5
    max_y = 2 * np.nanmax(y_true)

    assert n_codons == len(nx_graph.nodes())

    def to_custom_polar(idx, start=2 * np.pi / 16):
        return base_r, np.pi - start / 2 - (2 * np.pi - start) / n_codons * idx

    data = []
    edge_r = []
    edge_theta = []
    for edge in nx_graph.edges():
        r0, theta0 = to_custom_polar(edge[0])  # cart2pol(x0, y0)
        r1, theta1 = to_custom_polar(edge[1])  # cart2pol(x1, y1)
        edge_r.append(r0)
        edge_r.append(r1)
        edge_r.append(np.nan)
        edge_theta.append(theta0)
        edge_theta.append(theta1)
        edge_theta.append(np.nan)

    edge_trace = go.Scatterpolar(
        r=edge_r,
        theta=edge_theta,
        # color=,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
        thetaunit="radians",
        showlegend=True,
        name="secondary structure",
    )
    data.append(edge_trace)
    if clickData:
        attr_idx = int(clickData["points"][0]["text"])
        edge_weight = np.full((n_codons + 1, n_codons + 1), np.nan)
        for a, b, w in edge_attr[attr_idx]:
            if np.any(np.array([a, b]) == n_codons):
                continue
            a, b = int(a), int(b)
            edge_weight[a][b] = np.abs(w)
        threshold = np.quantile(edge_weight[~np.isnan(edge_weight)], 0.95)
        first = True
        for a in range(n_codons):
            for b in range(a + 1, n_codons):
                w1, w2 = edge_weight[a][b], edge_weight[b][a]
                if np.all(np.isnan([w1, w2])):
                    continue
                w = np.nanmean([w1, w2])
                if w < threshold:
                    continue
                w = w / np.nanmax(edge_weight) * 100
                r0, theta0 = to_custom_polar(a)
                r1, theta1 = to_custom_polar(b)
                red_rgb = px.colors.hex_to_rgb(PLOT_COLORS["edge_attrib"])
                line_trace = go.Scatterpolar(
                    r=[r0, r1],
                    theta=[theta0, theta1],
                    line=dict(
                        width=1.5,
                        color=f"rgba({red_rgb[0]}, {red_rgb[1]}, {red_rgb[2]}, {w})",
                    ),
                    hoverinfo="none",
                    mode="lines",
                    thetaunit="radians",
                    legendgroup="1",
                    name="important edges",
                    showlegend=first,
                )
                first = False
                data.append(line_trace)

    node_r = []
    node_theta = []
    for node in nx_graph.nodes():
        r, theta = to_custom_polar(node)  # cart2pol(x, y)
        node_r.append(r)
        node_theta.append(theta)

    mask_nan = np.isnan(y_true)
    nan_trace = dict(
        type="scatterpolar",
        r=np.array(node_r)[mask_nan],
        theta=np.array(node_theta)[mask_nan],
        mode="markers",
        name="NaNs",
        marker_color=PLOT_COLORS["nans"],
        marker=dict(symbol="x"),
    )
    data.append(nan_trace)

    node_trace = go.Scatterpolar(
        r=node_r,
        theta=node_theta,
        mode="markers",
        hoverinfo="text",
        thetaunit="radians",
        marker=dict(
            colorscale="YlGnBu",
            color=[],
            size=2,
            line_width=2,
        ),
        showlegend=False,
    )
    data.append(node_trace)

    ann_r = list(
        chain.from_iterable((base_r, base_r + a / max_y, np.nan) for a in y_true)
    )
    y_theta = list(chain.from_iterable((theta, theta, np.nan) for theta in node_theta))
    green_rgb = px.colors.hex_to_rgb(PLOT_COLORS["y_true"])
    y_true_lines_trace = go.Scatterpolar(
        r=ann_r,
        theta=y_theta,
        marker=dict(
            line_width=1,
            color=f"rgba({green_rgb[0]}, {green_rgb[1]},{green_rgb[2]}, 0.5)",
        ),
        thetaunit="radians",
        mode="lines",
        name="ribosome counts",
    )
    data.append(y_true_lines_trace)

    y_true_points_trace = go.Scatterpolar(
        r=base_r + y_true / max_y,
        theta=node_theta,
        text=np.arange(n_codons),
        marker=dict(line_width=2, color=PLOT_COLORS["y_true"]),
        thetaunit="radians",
        name="ribosome counts",
        mode="markers",
        showlegend=False,
        customdata=sequence,
        hovertemplate="x=%{text}, %{customdata}",
    )
    data.append(y_true_points_trace)

    if clickData:
        codon_idx = int(clickData["points"][0]["text"])
        attr_idx = codon_idx
        temp_node_attr = node_attr[attr_idx]
        temp_node_attr = np.abs(temp_node_attr)
        max_attr = 4 * np.max(temp_node_attr)
        ann_r = list(
            chain.from_iterable(
                (base_r, base_r + a / max_attr, np.nan) for a in temp_node_attr
            )
        )
        y_theta = list(
            chain.from_iterable((theta, theta, np.nan) for theta in node_theta)
        )
        node_attr_trace = go.Scatterpolar(
            r=ann_r,
            theta=y_theta,
            marker=dict(line_width=2, color=px.colors.qualitative.Plotly[4]),
            thetaunit="radians",
            mode="lines",
            name="codon importance",
        )
        data.append(node_attr_trace)
        r0, theta0 = to_custom_polar(codon_idx)
        selected_node_trace = go.Scatterpolar(
            r=[base_r + y_true[codon_idx] / max_y],
            theta=[node_theta[codon_idx]],
            thetaunit="radians",
            marker=dict(
                symbol="circle-open",
                size=8,
                line=dict(width=4),
                color=px.colors.qualitative.Plotly[1],
            ),
            name="selected annotation",
            mode="markers",
        )
        data.append(selected_node_trace)

    r0, theta0 = to_custom_polar(0)
    arrow_trace = go.Scatterpolar(
        r=[base_r + 0.04, base_r + 0.01],
        theta=[np.pi - 1 / 40 * np.pi, theta0 + 5e-3],
        thetaunit="radians",
        marker=dict(
            size=10,
            symbol="arrow-bar-up",
            angleref="previous",
            color=px.colors.qualitative.Plotly[0],
        ),
        name="sequence start",
    )
    data.append(arrow_trace)

    return go.Figure(
        data=data,
        layout=dict(
            title=rowData.iloc[sample_idx].gene_symbol,
            autosize=False,
            font_size=15,
            width=1000,
            height=1000,
            legend=dict(
                yanchor="top", y=0.5, xanchor="left", x=-0.3, bgcolor="rgba(0,0,0,0)"
            ),
            polar=dict(
                bgcolor="rgba(0, 0, 0, 0)",
                radialaxis=dict(
                    angle=180,
                    tickmode="array",
                    tickvals=[],
                    showline=False,
                    tickangle=180,
                    autorange="max",
                ),
                angularaxis=dict(tickvals=[]),
            ),
        ),
    )


@callback(
    Output("sequence", "figure"),
    [Input("datatable", "selectedRows"), Input("mrna", "clickData")],
)
def make_sequence_graph(selected_row, clickData):
    if ctx.triggered_id == "datatable":
        clickData = None
    sample_idx = selected_row[0]["id"] if selected_row else 0
    y_true = _get_y_true(sample_idx)
    y_pred = _get_y_pred(sample_idx)
    max_density = np.max(np.concatenate((y_true, y_pred)))
    n_codons = len(y_true)
    sequence = rowData.iloc[sample_idx].sequence
    sequence = re.findall("...", sequence)
    idxs_attr = _get_idxs_attr(sample_idx)
    node_attr = _get_node_attr(sample_idx).reshape(n_codons, n_codons)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    annot_trace = dict(
        type="scatter",
        x=np.arange(n_codons),
        y=y_pred,
        name="predicted values",
        mode="lines+markers",
        customdata=sequence,
        hovertemplate="x=%{x}, y=%{y:.2f}, %{customdata}",
        marker_color=px.colors.qualitative.Plotly[0],
    )
    fig.add_trace(annot_trace, secondary_y=False)

    pred_trace = dict(
        type="scatter",
        x=np.arange(n_codons),
        y=y_true,
        customdata=sequence,
        hovertemplate="x=%{x}, y=%{y:.2f}, %{customdata}",
        mode="lines+markers",
        name="true values",
        marker_color=px.colors.qualitative.Plotly[2],
    )
    fig.add_trace(pred_trace, secondary_y=False)

    x_none = np.where(np.isnan(y_true))[0]
    annot_trace = dict(
        type="scatter",
        x=x_none,
        y=[0] * len(x_none),
        mode="markers",
        name="NaNs",
        marker_color=px.colors.qualitative.Plotly[4],
        marker=dict(symbol="x"),
    )
    fig.add_trace(annot_trace, secondary_y=False)

    attr_idx = None
    shapes = []
    if clickData:
        codon_idx = int(clickData["points"][0]["text"])
        attr_idx = codon_idx

    if attr_idx:
        temp_node_attr = node_attr[attr_idx]
        temp_node_attr = np.abs(temp_node_attr)
        max_attr = np.max(temp_node_attr)

        attributions = temp_node_attr / max_attr

        codon_importance = dict(
            type="bar",
            x=np.arange(n_codons),
            y=attributions,
            marker_color=px.colors.qualitative.Plotly[4],
            name="codon importance",
            width=2 / 3,
        )
        fig.add_trace(codon_importance, secondary_y=True)

        highlight_curr_codon = dict(
            type="scatter",
            x=[codon_idx, codon_idx],
            y=[y_true[codon_idx], y_pred[codon_idx]],
            marker_color=px.colors.qualitative.Plotly[1],
            mode="markers",
            name="selected annotation",
            marker=dict(symbol="circle-open", size=8, line=dict(width=4)),
        )
        fig.add_trace(highlight_curr_codon, secondary_y=False)

    fig.update_layout(
        template="none",
        title=dict(
            y=0.82,
            x=0.06,
            xanchor="left",
            yanchor="top",
            text=rowData.iloc[sample_idx].gene_symbol,
        ),
        font_size=15,
        barmode="overlay",
        shapes=shapes,
        xaxis=dict(
            rangeslider=dict(visible=False),
            title="codon position",
            range=[-5, n_codons + 5],
        ),
        yaxis={
            "title": "ribosome counts",
            "range": [0, max_density],
            "tickformat": ".3f",
        },
        yaxis2={
            "tickmode": "sync",
            "domain": [0.0, 1.0],
            "title": "importance",
        },
        legend=dict(yanchor="bottom", y=1, xanchor="right", x=0.93, orientation="h"),
    )

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run server.")
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the Waitress server on (default: 8050)",
    )

    args = parser.parse_args()
    port = args.port

    print(f"Starting server on port {port}...")
    serve(app.server, host="0.0.0.0", port=port)
    # app.run(debug=True)
