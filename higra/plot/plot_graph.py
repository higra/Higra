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
from .utils import higra_graph_to_networkx

try:
    import plotly.graph_objects as go
    __plotly_available = True
except:
    __plotly_available = False

try:
    import networkx as nx
    __networkx_available = True
except:
    __networkx_available = False


def plot_graph(graph, *, 
               vertex_positions=None, 
               vertex_labels=None, 
               display_vertex_labels=True, 
               show_connected_components=False, 
               edge_weights=None, 
               display_edge_weights=False,
               show_fig=True, 
               seed=None):
    """
    Plot a graph interactively using Plotly.

    Let ``G = (V, E)`` be the input graph, where ``V`` is the set of vertices and ``E`` is the set of edges:

    .. math::

        |V| = n, \\qquad |E| = m

    Each vertex :math:`v \\in V` is mapped to a 2D Cartesian coordinate:

    .. math::

        p(v) = (p_0(v), p_1(v)) \\in \\mathbb{R}^2

    If ``vertex_positions`` is not provided, node coordinates are computed automatically using
    the Fruchterman-Reingold force-directed algorithm:

    .. math::

        p = \\operatorname{spring\\_layout}(G)

    Edges :math:`e = (u, v) \\in E` are drawn as line shapes. When ``edge_weights`` are given,
    the visual thickness :math:`w(e)` of each edge is linearly interpolated:

    .. math::

        w(e) = 1 + 2 \\cdot \\frac{W(e) - \\min(W)}{\\max(W) - \\min(W)}

    where :math:`W(e)` is the weight of edge :math:`e`.

    When ``show_connected_components`` is enabled, a labeling function assigns a component ID to each vertex:

    .. math::

        c: V \\to \\mathbb{N}

    Each component is mapped to a distinct color in the HSL color space to color the corresponding vertices and edges.

    Complexity

    - **Layout computation** (if ``vertex_positions`` is None): :math:`O(|V|^2)` per iteration using NetworkX spring layout.
    - **Connected components** (if ``show_connected_components`` is True): :math:`O(|V| + |E|)` time complexity.
    - **Trace rendering**: :math:`O(|V| + |E|)` time and memory to create Plotly traces and line shapes.

    Example
    ==========

    1. Basic graph plot:

    .. code-block:: python

        graph = hg.random_undirected_graph_erdos_renyi(10, 0.9, seed=11)
        fig1 = plot_graph(graph, seed=11)

    .. display_graph:: 

        GRAPH_RANDOM
        num_vertices = 10
        mean_degree = 0.9
        seed = 11

    2. Graph plot with custom vertex labels:

    .. code-block:: python

        graph = hg.random_undirected_graph_erdos_renyi(10, 0.9, seed=11)    
        vertex_labels=[1, 2, 4, 8, 16, 32, 64, 18, 25, 52]
        fig2 = plot_graph(graph, vertex_labels=vertex_labels, seed=11)

    .. display_graph:: 

        GRAPH_RANDOM
        num_vertices = 10
        mean_degree = 0.9
        seed = 11
        vertex_labels = [1, 2, 4, 8, 16, 32, 64, 18, 25, 52]

    3. Graph plot with edge weights and weight labels:

    .. code-block:: python

        graph = hg.random_undirected_graph_erdos_renyi(10, 0.9, seed=11)
        edge_weights=np.asarray([1, 100, 0, 27, 2, 0, 28, 16])
        fig3 = plot_graph(graph, edge_weights=weights, display_edge_weights=True, seed=11)

    .. display_graph:: 

        GRAPH_RANDOM
        num_vertices = 10
        mean_degree = 0.9
        seed = 11
        edge_weights = [1, 100, 0, 27, 2, 0, 28, 16]
        display_edge_weights = True

    4. Graph plot showing connected components:

    .. code-block:: python

        graph = hg.random_undirected_graph_erdos_renyi(10, 0.9, seed=11)
        fig4 = plot_graph(graph, show_connected_components=True, seed=11)

    .. display_graph:: 

        GRAPH_RANDOM
        num_vertices = 10
        mean_degree = 0.9
        seed = 11
        show_connected_components = True

    5. Combined graph plot with vertex labels, edge weights, and connected components:

    .. code-block:: python

        graph = hg.random_undirected_graph_erdos_renyi(10, 0.9, seed=11)
        v_labels = [1, 2, 4, 8, 16, 32, 64, 18, 25, 52]
        weights = np.asarray([1, 100, 0, 27, 2, 0, 28, 16])
        fig5 = plot_graph(
            graph,
            vertex_labels=v_labels,
            edge_weights=weights,
            display_edge_weights=True,
            show_connected_components=True,
            seed=11
        )

    .. display_graph:: 

        GRAPH_RANDOM
        num_vertices = 10
        mean_degree = 0.9
        seed = 11
        vertex_labels = [1, 2, 4, 8, 16, 32, 64, 18, 25, 52]
        edge_weights = [1, 100, 0, 27, 2, 0, 28, 16]
        display_edge_weights = True
        show_connected_components = True

    :param graph: input graph (Concept :class:`~higra.CptGraph`).
    :param vertex_positions: optional 2D array of shape ``(|V|, 2)`` containing the Cartesian coordinates of each vertex. Will default to ``None``, in which case the layout is computed automatically using the Fruchterman-Reingold force-directed algorithm.
    :param vertex_labels: optional array-like of shape ``(|V|,)`` containing labels or values for each vertex.
    :param display_vertex_labels: whether to display vertex labels on top of vertex markers (default: True).
    :param show_connected_components: whether to color vertices and edges by connected components (default: False).
    :param edge_weights: optional 1D array of shape ``(|E|,)`` containing numerical weights for each edge. Will default to ``None``, in which case all edges are assigned a weight of 1.
    :param display_edge_weights: whether to display edge weights as text labels on edge midpoints (default: False).
    :param show_fig: whether to display the figure interactively (default: True).
    :param seed: optional random seed for ``networkx.spring_layout`` reproducibility.
    :return: a :class:`plotly.graph_objects.Figure` object representing the graph.
    """
    assert __plotly_available, "The plot graph function requires plotly"
    assert __networkx_available, "The plot graph function requires networkx"

    sources, targets = graph.edge_list()

    if vertex_positions is None:
        vertex_positions = _get_vertex_positions(graph, seed=seed)

    marker_color, edge_colors = _get_colors(graph, show_connected_components, sources)

    marker_kwargs = dict(
        size=20,
        line=dict(width=1, color='black'),
        color=marker_color,
    )

    hover_text, display_text = _get_vertex_text(vertex_labels, vertex_positions)

    trace_nodes = go.Scatter(
        x=vertex_positions[:, 0],
        y=vertex_positions[:, 1],
        mode='markers'+('+text' if display_vertex_labels else ''),
        marker=marker_kwargs,
        text= display_text,
        hoverinfo='text',
        hovertext=hover_text,
        showlegend=False
    )

    fig = go.Figure(data=[trace_nodes])

    if display_edge_weights:
        mid_x = (vertex_positions[sources, 0] + vertex_positions[targets, 0]) / 2
        mid_y = (vertex_positions[sources, 1] + vertex_positions[targets, 1]) / 2
        text_positions = _get_label_position(vertex_positions, sources, targets)

        trace_labels = go.Scatter(
            x=mid_x,
            y=mid_y,
            mode='text',
            text=[str(label) + " " for label in edge_weights],
            textposition=text_positions,
            hoverinfo='text',
            hovertext=[f"Edge ({src}, {tgt}) (Weight: {weight})" for src, tgt, weight in zip(sources, targets, edge_weights)],
            showlegend=False,
        )
        fig.add_trace(trace_labels)

    edge_widths = _get_edge_widths(edge_weights, sources)

    lines = [
        dict(
            type='line',
            x0=vertex_positions[src, 0],
            y0=vertex_positions[src, 1],
            x1=vertex_positions[tgt, 0],
            y1=vertex_positions[tgt, 1],
            line=dict(width=width, color=color),
            layer='below'
        )
        for src, tgt, width, color in zip(sources, targets, edge_widths, edge_colors)
    ]

    fig.update_layout(
        shapes=lines,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10)
    )

    if show_fig:
        fig.show()
    return fig

def _get_vertex_positions(graph, seed=None):
    """
    Get the 2D Cartesian coordinates of each vertex in the graph.

    :param graph: input graph (Concept :class:`~higra.CptGraph`).
    :param seed: optional random seed for ``networkx.spring_layout`` reproducibility.
    :return: a 2D array of shape ``(|V|, 2)`` containing the Cartesian coordinates of each vertex.
    """
    nxGraph = higra_graph_to_networkx(graph)
    pos = nx.spring_layout(nxGraph, seed=seed)
    vertex_positions = np.array([pos[n] for n in nxGraph.nodes()])
    return vertex_positions

def _get_colors(graph, show_connected_components, sources):
    """
    Get a list of colors for the edges of the graph based on connected components.

    :param graph: input graph (Concept :class:`~higra.CptGraph`).
    :param show_connected_components: whether to color vertices and edges by connected components.
    :param sources: optional array of source vertices for edges. If not provided, all edges are considered.
    :return: a tuple of two lists of colors corresponding to each edge in the graph.
    """
    if show_connected_components:
        connected_components = hg.connected_components_labeling(graph)
        unique_components = np.unique(connected_components)
        
        colors = [f"hsl({i * 360 / len(unique_components)}, 70%, 50%)" for i in range(len(unique_components))]
        component_to_color = dict(zip(unique_components, colors))
    
        marker_color = [component_to_color[component] for component in connected_components]
        edge_colors = [component_to_color[connected_components[src]] for src in sources]
    else:
        marker_color, edge_colors = 'white', ['black'] * len(sources)

    return marker_color, edge_colors

def _get_vertex_text(vertex_labels, vertex_positions):
    """
    Get two lists of labels for the vertices of the graph.

    :param graph: input graph (Concept :class:`~higra.CptGraph`).
    :param vertex_labels: optional array-like of shape ``(|V|,)`` containing labels or values for each vertex.
    :return: a tuple of two lists of labels corresponding to each vertex in the graph.
    """
    if vertex_labels is not None:
        hover_text = [f"Vertex {v} (Label: {l})" for v, l in enumerate(vertex_labels)]
        display_text = [l for l in vertex_labels]
    else:
        hover_text = [f"Vertex {v}" for v in range(len(vertex_positions))]
        display_text = [v for v in range(len(vertex_positions))]

    return hover_text, display_text

def _get_edge_widths(edge_weights, sources):
    """
    Get a list of widths for the edges of the graph based on edge weights.

    :param edge_weights: optional 1D array of shape ``(|E|,)`` containing numerical weights for each edge.
    :param sources: array of source vertex indices for edges.
    :return: a list of widths corresponding to each edge in the graph.
    """
    if edge_weights is not None:
        if isinstance(edge_weights, list):
            edge_weights = np.array(edge_weights)
            min_w, max_w = edge_weights.min(), edge_weights.max()
            if min_w == max_w:
                edge_widths = np.full_like(edge_weights, 1, dtype=float)
            else:
                edge_widths = np.interp(edge_weights, (min_w, max_w), (1, 3))
    else:
        edge_widths = [1] * len(sources)

    return edge_widths

def _get_label_position(vertex_positions, sources, targets):
    """
    Get the text position for edge weight labels based on the angle of the edge.

    :param vertex_positions: array of vertex positions.
    :param sources: array of source vertex indices.
    :param targets: array of target vertex indices.
    :return: a list of strings representing the text positions for the labels.
    """
    dist_x = vertex_positions[targets, 0] - vertex_positions[sources, 0]
    dist_y = vertex_positions[targets, 1] - vertex_positions[sources, 1]
    angles = np.degrees(np.arctan2(dist_y, dist_x)) % 180

    text_positions = []
    for angle in angles:
        if 45 <= angle <= 135:
            text_positions.append("middle left")
        else:
            text_positions.append("top center")

    return text_positions