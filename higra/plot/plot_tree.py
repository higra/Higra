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

import numpy as np
import higra as hg

try:
    from scipy.cluster.hierarchy import dendrogram
    __scipy_available = True
except:
    __scipy_available = False

try:
    import plotly.graph_objects as go
    __plotly_available = True
except:
    __plotly_available = False

def plot_partition_tree(tree, *, altitudes=None, n_clusters=0, lastp=30, show_fig=True):
    """Plot the given tree as a dendrogram using Plotly.

    Requires the ``plotly`` and ``scipy`` libraries.

    :param tree: Input tree
    :param altitudes: Tree node altitudes (will default to
        :func:`~higra.attribute_regular_altitudes(tree)` if ``None``)
    :param n_clusters: Colorize the :attr:`n_clusters` largest clusters of the
        dendrogram with different colors
    :param lastp: Collapse subtrees containing less than :attr:`lastp` leaves.
    :param show_fig: Whether to display the plot interactively
    :return: plotly.graph_objects.Figure
    """
    assert __scipy_available, "The plot tree function requires scipy"
    assert __plotly_available, "The plot tree function requires plotly"

    num_leaves = tree.num_leaves()

    if altitudes is None:
        altitudes = hg.attribute_regular_altitudes(tree)    

    if np.max(tree.num_children()) > 2:
        tree, nmap = hg.tree_2_binary_tree(tree)
        if altitudes is not None:
            altitudes = altitudes[nmap]

    linkage_matrix = hg.binary_hierarchy_to_scipy_linkage_matrix(tree, altitudes)

    extra = {} if lastp is None else dict(truncate_mode='lastp', p=lastp)

    if n_clusters > 1:
        dsort = np.sort(linkage_matrix[:, 2])
        idx = min(n_clusters - 1, len(dsort))
        color_threshold = dsort[-idx]
    else:
        color_threshold = 0
    
    ddata = dendrogram(linkage_matrix, no_labels=False, above_threshold_color="k", no_plot=True, color_threshold=color_threshold, **extra)

    unique_clusters = sorted(list({c for c in ddata["color_list"] if c != "k"}), key=lambda x: int(x[1:]))

    n_colors = len(unique_clusters)
    if n_colors > 0:
        colors = [f"hsl({int(i * 360 / n_colors)}, 70%, 50%)" for i in range(n_colors)]
        color_map = dict(zip(unique_clusters, colors))
    else:
        color_map = {}
    color_map["k"] = "black"

    fig = go.Figure()

    foot_color = {}
    for x, y, color_key in zip(ddata["icoord"], ddata["dcoord"], ddata["color_list"]):
        line_color = color_map.get(color_key, "black")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=line_color, width=2),
                hoverinfo="none",
                showlegend=False
            )
        )
        foot_color[(round(x[0], 3), round(y[0], 3))] = line_color
        foot_color[(round(x[3], 3), round(y[3], 3))] = line_color

    leaf_x = {leaf: 5 + 10 * rank for rank, leaf in enumerate(ddata["leaves"])}

    if len(ddata["dcoord"]) == len(linkage_matrix):
        for i, row in enumerate(linkage_matrix):
            child1, child2, alt = int(row[0]), int(row[1]), row[2]
            node_id = num_leaves + i
            x_pos = (leaf_x[child1] + leaf_x[child2]) / 2
            y_pos = alt
            leaf_x[node_id] = x_pos
 
            line_color = "black"
            for x, y, color_key in zip(ddata["icoord"], ddata["dcoord"], ddata["color_list"]):
                branch_x = (x[1] + x[2]) / 2
                if np.isclose(branch_x, x_pos) and np.isclose(y[1], y_pos):
                    line_color = color_map.get(color_key, "black")
                    break

            fig.add_trace(
                go.Scatter(
                    x=[x_pos],
                    y=[y_pos],
                    mode="markers+text",
                    marker=dict(size=20, color="white", line=dict(width=1.5, color=line_color)),
                    text=[str(node_id)],
                    textposition="middle center",
                    textfont=dict(size=10, color="black"),
                    hoverinfo="text",
                    hovertext=f"Node {node_id} (Altitude: {y_pos})",
                    showlegend=False,
                )
            )

    internal_alt = altitudes[num_leaves:]
    internal_min_alt = float(np.min(internal_alt))
    internal_max_alt = float(np.max(internal_alt))
    alt_span = internal_max_alt - internal_min_alt
    stem_gap = alt_span * 0.08 if alt_span > 0 else 1.0

    lowest_drawn_y = 0.0
    for leaf in range(num_leaves):
        x = leaf_x[leaf]
        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[-stem_gap, 0],
                mode="lines",
                line=dict(color="black", width=2),
                hoverinfo="none",
                showlegend=False,
            )
        )
    lowest_drawn_y = -stem_gap

    for leaf in range(num_leaves):
        x = leaf_x[leaf]
        y = -stem_gap
        line_color = foot_color.get((round(x, 3), 0.0), "black")
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(size=20, color="white", line=dict(width=1.5, color=line_color)),
                text=[str(leaf)],
                textposition="middle center",
                textfont=dict(size=10, color="black"),
                hoverinfo="text",
                hovertext=f"Leaf {leaf}",
                showlegend=False,
            )
        )

    pad = alt_span * 0.1 if alt_span > 0 else 1.0

    fig.update_layout(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(title="Altitude", showgrid=True, range=[lowest_drawn_y - pad, internal_max_alt + pad]),
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40)
        )

    if show_fig:
        fig.show()
    return fig