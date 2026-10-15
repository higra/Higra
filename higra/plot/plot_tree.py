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


def plot_partition_tree(tree, *, altitudes=None, n_clusters=0, lastp=30, show_fig=True,
                       child_ordering="default"):
    """Plot a hierarchy as an interactive dendrogram using Plotly.

    Let :math:`T=(V,E)` be a rooted tree with :math:`n` leaves. Internal node
    altitudes :math:`h(v)` are given by ``altitudes`` or computed as the
    topological height:

    .. math::

        h(v) = \\begin{cases}
            0, & v \\in L \\\\
            1 + \\max_{u \\in C(v)} h(u), & v \\in I
        \\end{cases}

    Here :math:`C(v)` denotes the children of :math:`v`. The resulting
    altitudes build a SciPy linkage matrix with one row per internal node:

    .. math::

        (u_1, u_2, h(v), |L(v)|)

    where :math:`L(v)` is the set of descendant leaves. ``child_ordering``
    determines the row's left and right children. For each child :math:`u`,
    let :math:`a(u)=\min L(u)`, :math:`b(u)=\max L(u)`, and
    :math:`s(u)=|L(u)|`. The criteria compare:

    - ``"default"``: input order.
    - ``"smallest_label"`` / ``"largest_label"``: :math:`a(u)` ascending /
      :math:`b(u)` descending.
    - ``"smallest_subtree"`` / ``"largest_subtree"``: :math:`s(u)` ascending /
      descending.
    - ``"smallest_altitude"`` / ``"largest_altitude"``: :math:`h(u)` ascending /
      descending.

    Ties use the smallest descendant leaf label. Ordering changes only the
    layout. SciPy assigns horizontal coordinates to leaves in dendrogram order;
    each internal node is placed midway between its children:

    .. math::

        x(v)=\frac{x(u_1)+x(u_2)}{2}, \qquad y(v)=h(v)

    SciPy computes the branch segments. ``lastp`` limits the displayed merges;
    ``n_clusters`` selects their color threshold. The Plotly figure is returned
    and optionally displayed.

    Complexity

    The runtime complexity is :math:`\mathcal{O}(n^2)` for a complete
    dendrogram, where :math:`n` is the number of nodes in ``tree``. This is
    dominated by matching each internal node to its dendrogram segment.
    Altitude and linkage conversion and child ordering take :math:`\mathcal{O}(n)`;
    selecting the color threshold takes :math:`\mathcal{O}(n\log(n))` when
    ``n_clusters > 1``.

    Example
    =======

    1. Plot a binary partition tree:

    .. code-block:: python

        graph = hg.get_4_adjacency_graph((3, 3))
        edge_weights = np.asarray((0, 6, 2, 6, 0, 0, 5, 4, 5, 3, 0, 1))
        tree, altitudes = hg.bpt_canonical(graph, edge_weights)
        fig_default = plot_partition_tree(tree)

    .. display_graph::


    2. Use custom altitudes:

    .. code-block:: python

        custom_altitudes = 2.0 * altitudes
        fig_custom_altitudes = plot_partition_tree(
            tree, altitudes=custom_altitudes, show_fig=False)

    .. display_graph::


    3. Color three clusters:

    .. code-block:: python

        fig_three_clusters = plot_partition_tree(
            tree, altitudes=altitudes, n_clusters=3, show_fig=False)

    .. display_graph::


    4. Truncate the dendrogram:

    .. code-block:: python

        fig_truncated = plot_partition_tree(
            tree, altitudes=altitudes, lastp=4, show_fig=False)

    .. display_graph::


    5. Display the complete dendrogram:

    .. code-block:: python

        fig_complete_tree = plot_partition_tree(
            tree, altitudes=altitudes, lastp=None, show_fig=False)

    .. display_graph::


    The following tree makes the ordering criteria distinct:

    .. code-block:: python

        linkage = np.asarray((
            (2, 1, 0.5, 2),
            (4, 0, 0.7, 2),
            (7, 3, 1.5, 3),
            (8, 9, 2.5, 5),
            (5, 6, 3.0, 2),
            (10, 11, 3.2, 7),
        ))
        ordering_tree, ordering_altitudes, _ = \\
            hg.scipy_linkage_matrix_to_binary_hierarchy(linkage)

    6. Keep the input order:

    .. code-block:: python

        fig_default_order = plot_partition_tree(
            ordering_tree, altitudes=ordering_altitudes, lastp=None,
            child_ordering="default", show_fig=False)

    .. display_graph::


    7. Smallest leaf label first:

    .. code-block:: python

        fig_smallest_label = plot_partition_tree(
            ordering_tree, altitudes=ordering_altitudes, lastp=None,
            child_ordering="smallest_label", show_fig=False)

    .. display_graph::


    8. Largest leaf label first:

    .. code-block:: python

        fig_largest_label = plot_partition_tree(
            ordering_tree, altitudes=ordering_altitudes, lastp=None,
            child_ordering="largest_label", show_fig=False)

    .. display_graph::


    9. Fewest leaves first:

    .. code-block:: python

        fig_smallest_subtree = plot_partition_tree(
            ordering_tree, altitudes=ordering_altitudes, lastp=None,
            child_ordering="smallest_subtree", show_fig=False)

    .. display_graph::


    10. Most leaves first:

    .. code-block:: python

        fig_largest_subtree = plot_partition_tree(
            ordering_tree, altitudes=ordering_altitudes, lastp=None,
            child_ordering="largest_subtree", show_fig=False)

    .. display_graph::


    11. Lowest altitude first:

    .. code-block:: python

        fig_smallest_altitude = plot_partition_tree(
            ordering_tree, altitudes=ordering_altitudes, lastp=None,
            child_ordering="smallest_altitude", show_fig=False)

    .. display_graph::


    12. Highest altitude first:

    .. code-block:: python

        fig_largest_altitude = plot_partition_tree(
            ordering_tree, altitudes=ordering_altitudes, lastp=None,
            child_ordering="largest_altitude", show_fig=False)

    .. display_graph::

    :param tree: Input hierarchy to plot.
    :param altitudes: Optional array of altitudes, with one value per tree node.
        If ``None``, integer topological heights are computed with
        :func:`~higra.attribute_topological_height`; leaves have altitude zero.
    :param n_clusters: Number of clusters to color in the dendrogram. Values
        greater than one select a cut for that many clusters; the default,
        ``0``, disables cluster coloring.
    :param lastp: Number of last merges to display using SciPy's ``lastp``
        truncation mode. Set to ``None`` to display the complete dendrogram.
        The default is ``30``.
    :param show_fig: If ``True``, display the figure interactively. If
        ``False``, return it without displaying it. The default is ``True``.
    :param child_ordering: Criterion for ordering the two children at every
        merge. One of ``"default"``, ``"smallest_label"``, ``"largest_label"``,
        ``"smallest_subtree"``, ``"largest_subtree"``,
        ``"smallest_altitude"``, or ``"largest_altitude"``. See the child
        ordering definitions above. The default is ``"default"``.
    :return: A :class:`plotly.graph_objects.Figure` representing the dendrogram.
    """
    assert __scipy_available, "The plot tree function requires scipy"
    assert __plotly_available, "The plot tree function requires plotly"

    num_leaves = tree.num_leaves()

    if altitudes is None:
        altitudes = hg.attribute_topological_height(tree)

    if np.max(tree.num_children()) > 2:
        tree, node_map = hg.tree_2_binary_tree(tree)
        altitudes = altitudes[node_map]

    linkage_matrix = hg.binary_hierarchy_to_scipy_linkage_matrix(tree, altitudes)
    linkage_matrix = _order_linkage_matrix(linkage_matrix, child_ordering)

    extra = {} if lastp is None else dict(truncate_mode='lastp', p=lastp)

    color_threshold = _get_color_threshold(linkage_matrix, n_clusters)

    ddata = dendrogram(linkage_matrix, no_labels=False, above_threshold_color="k", no_plot=True,
                       color_threshold=color_threshold, **extra)

    color_map = _get_cluster_color_map(ddata)

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
            node_id, x_pos, y_pos, line_color = _get_internal_node_position(
                i, row, num_leaves, leaf_x, ddata, color_map)
            leaf_x[node_id] = x_pos

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

    internal_min_alt, internal_max_alt, pad = _get_altitude_layout(altitudes, num_leaves)

    if len(ddata["dcoord"]) != len(linkage_matrix):
        visible_leaves = ddata["leaves"]
    else:
        visible_leaves = range(num_leaves)

    for leaf in visible_leaves:
        x = leaf_x[leaf]
        y = 0
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

    yaxis_options = dict(title="Altitude", showgrid=False, zeroline=False,
                         range=[min(0, internal_min_alt), internal_max_alt + pad])
    if np.all(np.equal(altitudes, np.rint(altitudes))):
        yaxis_options.update(tickmode="linear", tick0=0, dtick=1)

    fig.update_layout(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=yaxis_options,
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=40)
        )

    if show_fig:
        fig.show()
    return fig

def _get_color_threshold(linkage_matrix, n_clusters):
    """Get the dendrogram altitude threshold for the requested cluster count."""
    if n_clusters > 1:
        sorted_altitudes = np.sort(linkage_matrix[:, 2])
        idx = min(n_clusters - 1, len(sorted_altitudes))
        return sorted_altitudes[-idx]
    return 0

def _get_cluster_color_map(dendrogram_data):
    """Map SciPy cluster color codes to Plotly colors."""
    unique_clusters = sorted(
        {color for color in dendrogram_data["color_list"] if color != "k"},
        key=lambda color: int(color[1:]))
    n_colors = len(unique_clusters)
    colors = [f"hsl({int(i * 360 / n_colors)}, 70%, 50%)" for i in range(n_colors)]
    color_map = dict(zip(unique_clusters, colors))
    color_map["k"] = "black"
    return color_map

def _get_internal_node_position(index, row, num_leaves, leaf_x, dendrogram_data, color_map):
    """Get an internal node's position and the matching dendrogram branch color."""
    child1, child2, altitude = int(row[0]), int(row[1]), row[2]
    node_id = num_leaves + index
    x_pos = (leaf_x[child1] + leaf_x[child2]) / 2

    line_color = "black"
    for x, y, color_key in zip(dendrogram_data["icoord"], dendrogram_data["dcoord"],
                               dendrogram_data["color_list"]):
        branch_x = (x[1] + x[2]) / 2
        if np.isclose(branch_x, x_pos) and np.isclose(y[1], altitude):
            line_color = color_map.get(color_key, "black")
            break
    return node_id, x_pos, altitude, line_color

def _get_altitude_layout(altitudes, num_leaves):
    """Get the internal altitude bounds and y-axis padding."""
    internal_altitudes = altitudes[num_leaves:]
    internal_min_alt = float(np.min(internal_altitudes))
    internal_max_alt = float(np.max(internal_altitudes))
    altitude_span = internal_max_alt - internal_min_alt
    pad = altitude_span * 0.1 if altitude_span > 0 else 1.0
    return internal_min_alt, internal_max_alt, pad


def _order_linkage_matrix(linkage_matrix, child_ordering):
    """Order linkage children according to the selected left-to-right criterion."""
    valid_criteria = ("default", "smallest_label", "largest_label", "smallest_subtree",
                      "largest_subtree", "smallest_altitude", "largest_altitude")
    if child_ordering not in valid_criteria:
        raise ValueError(f"child_ordering must be one of {valid_criteria}, got {child_ordering!r}")
    if child_ordering == "default":
        return linkage_matrix

    num_leaves = linkage_matrix.shape[0] + 1
    min_leaf = np.zeros(2 * num_leaves - 1, dtype=np.int64)
    max_leaf = np.zeros(2 * num_leaves - 1, dtype=np.int64)
    min_leaf[:num_leaves] = np.arange(num_leaves)
    max_leaf[:num_leaves] = np.arange(num_leaves)
    subtree_size = np.ones(2 * num_leaves - 1, dtype=np.int64)
    subtree_altitude = np.zeros(2 * num_leaves - 1, dtype=linkage_matrix.dtype)
    ordered_linkage = linkage_matrix.copy()

    for i, row in enumerate(ordered_linkage):
        left, right = int(row[0]), int(row[1])
        if child_ordering == "smallest_label":
            left_key, right_key = min_leaf[left], min_leaf[right]
        elif child_ordering == "largest_label":
            left_key, right_key = -max_leaf[left], -max_leaf[right]
        elif child_ordering in ("smallest_subtree", "largest_subtree"):
            left_key = (subtree_size[left], min_leaf[left])
            right_key = (subtree_size[right], min_leaf[right])
            if child_ordering == "largest_subtree":
                left_key, right_key = (-left_key[0], left_key[1]), (-right_key[0], right_key[1])
        else:
            left_key = (subtree_altitude[left], min_leaf[left])
            right_key = (subtree_altitude[right], min_leaf[right])
            if child_ordering == "largest_altitude":
                left_key = (-left_key[0], left_key[1])
                right_key = (-right_key[0], right_key[1])
        if right_key < left_key:
            row[0], row[1] = right, left
            left, right = right, left

        node_id = num_leaves + i
        min_leaf[node_id] = min(min_leaf[left], min_leaf[right])
        max_leaf[node_id] = max(max_leaf[left], max_leaf[right])
        subtree_size[node_id] = subtree_size[left] + subtree_size[right]
        subtree_altitude[node_id] = row[2]

    return ordered_linkage
