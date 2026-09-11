############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) :                                                         #
#   - Giovanni Chierchia                                                   #
#   - Benjamin Perret                                                      #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg
import numpy as np
from scipy.__config__ import show
from .utils import COLORS, MARKERS, lighten_color

try:
    import plotly.graph_objects as go
    import networkx as nx
    __plotly_available = True
except:
    __plotly_available = False


def plot_graph(graph, *, vertex_positions=None, vertex_labels=None, edge_weights=None, show_fig=True):
    """
    Plot the given graph.

    Requires the ``plotly`` library.

    :param graph: Input graph
    :param vertex_positions: 2d array containing the coordinates of each vertex of the graph
    :param labels: Optional: vertex labels
    :return: None
    """
    assert __plotly_available, "The plot graph function requires plotly"

    sources, targets = graph.edge_list()

    marker_kwargs = dict(
        size=8,
        line=dict(width=1, color='black'),
    )

    if vertex_positions is None:
        pos = nx.spring_layout(nx.Graph(zip(sources)), seed=42)
        vertex_positions = np.array([pos[i] for i in range(graph.num_vertices())])

    if vertex_labels is not None:
        marker_kwargs['color'] = vertex_labels
        marker_kwargs['colorscale'] = 'Viridis'
        hover_text = [f"Vertex {v} (Label: {l})" for v, l in enumerate(vertex_labels)]
    else:
        marker_kwargs['color'] = 'white'
        hover_text = [f"Vertex {v}" for v in range(len(vertex_positions))]

    trace_nodes = go.Scatter(
        x=vertex_positions[:, 0],
        y=vertex_positions[:, 1],
        mode='markers',
        marker=marker_kwargs,
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    )

    fig = go.Figure(data=[trace_nodes])

    if edge_weights is not None:
        edge_widths = np.interp(edge_weights, (edge_weights.min(), edge_weights.max()), (1, 8))

        lines = []
        for src, tgt, width in zip(sources, targets, edge_widths):
            lines.append(dict(
                type='line',
                x0=vertex_positions[src, 0],
                y0=vertex_positions[src, 1],
                x1=vertex_positions[tgt, 0],
                y1=vertex_positions[tgt, 1],
                line=dict(width=width, color='black'),
            ))

        mid_x = (vertex_positions[sources, 0] + vertex_positions[targets, 0]) / 2
        mid_y = (vertex_positions[sources, 1] + vertex_positions[targets, 1]) / 2

        trace_labels = go.Scatter(
            x=mid_x,
            y=mid_y,
            mode='text',
            text=[str(label) for label in edge_weights],
            textposition='middle center',
            hoverinfo='none',
            showlegend=False
        )

        fig.add_scatter(trace_labels)

    else:
        lines = []
        for src, tgt in zip(sources, targets):
            lines.append(dict(
                type='line',
                x0=vertex_positions[src, 0],
                y0=vertex_positions[src, 1],
                x1=vertex_positions[tgt, 0],
                y1=vertex_positions[tgt, 1],
                line=dict(width=2, color='black'),
            ))

    fig.update_layout(dict(
        shapes=lines,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10)
    ))

    if show_fig:
        fig.show()
    return fig

def plot_graph_cut(graph_cut, *, vertex_positions=None, edge_weights=None, show_fig=True):
    """
    Plot the given graph cut.

    Requires the ``plotly`` library.

    :param graph_cut: Input graph cut
    :param vertex_positions: 2d array containing the coordinates of each vertex of the graph
    :param edge_weights: Optional: edge weights
    :param show_fig: Optional: whether to display the figure
    :return: None
    """
    assert __plotly_available, "The plot graph function requires plotly"

    connected_components = hg.connected_components_labeling(graph_cut)

    fig = plot_graph(graph_cut, vertex_positions=vertex_positions, vertex_labels=connected_components, edge_weights=edge_weights, show_fig=False)

    unique_components = np.unique(connected_components)

    colors = [
        f"hsl({i * 360 / len(unique_components)}, 70%, 50%)"
        for i in range(len(unique_components))
    ]

    component_to_color = dict(zip(unique_components, colors))

    fig.data[0].marker.color = [
        component_to_color[component]
        for component in connected_components
    ]

    sources, _ = graph_cut.edge_list()
    for shape, source in zip(fig.layout.shapes, sources):
        shape.line.color = component_to_color[connected_components[source]]

    if show_fig:
        fig.show()
    return fig