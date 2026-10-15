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

import json
import numbers
import warnings

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


def plot_partition_tree(tree, *, altitudes=None, title=None, show_fig=True,
                        show_node_labels="all", extra_info=None, show_extra_info="all",
                        line_width=2.0, node_size=20.0, label_size=10.0, n_clusters=0,
                        lastp=30, child_ordering="default"):
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
    let :math:`a(u)=\\min L(u)`, :math:`b(u)=\\max L(u)`, and
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

        x(v)=\\frac{x(u_1)+x(u_2)}{2}, \\qquad y(v)=h(v)

    SciPy computes the branch segments. ``lastp`` limits the displayed merges;
    ``n_clusters`` selects their color threshold. Hovering a node shows its
    identifier, its altitude and the values requested with ``extra_info``. The
    Plotly figure is returned and optionally displayed.

    Complexity

    Let :math:`n` be the number of nodes in ``tree``. Converting the altitudes,
    building and ordering the linkage matrix, and placing the nodes take
    :math:`\\mathcal{O}(n)`; selecting the color threshold takes
    :math:`\\mathcal{O}(n\\log(n))` when ``n_clusters > 1``.

    Example
    =======

    The examples use three binary partition trees: one built on a 3x3
    grayscale image, one on a 6x6 grayscale image, and one on an arbitrary
    graph whose vertices are points of the plane:

    .. code-block:: python

        image = np.asarray(((10, 14, 30), (12, 45, 36), (22, 52, 70)))
        graph = hg.get_4_adjacency_graph(image.shape)
        tree, altitudes = hg.bpt_canonical(
            graph, hg.weight_graph(graph, image, hg.WeightFunction.L1))

        yy, xx = np.mgrid[0:6, 0:6]
        big_image = 20 * xx + 10 * yy + \\
            np.random.default_rng(2).normal(0, 8, (6, 6))
        big_graph = hg.get_4_adjacency_graph(big_image.shape)
        big_tree, big_altitudes = hg.bpt_canonical(
            big_graph,
            hg.weight_graph(big_graph, big_image, hg.WeightFunction.L1))

        points = np.asarray(((0, 0), (2.1, 1.4), (0.9, 3.2), (3.9, 2.4),
                             (3.2, 0.8), (6.0, 4.5), (2.2, 6.3)))
        point_graph = hg.UndirectedGraph(7)
        point_graph.add_edges((0, 1, 1, 2, 3, 3, 4, 5),
                              (1, 2, 4, 6, 4, 5, 5, 6))
        sources, targets = point_graph.edge_list()
        point_tree, point_altitudes = hg.bpt_canonical(
            point_graph,
            np.linalg.norm(points[sources] - points[targets], axis=1))

    1. Use all the default parameters (topological heights, input order of
       the children, the last 30 merges, all identifiers) and display the
       figure:

    .. code-block:: python

        fig_default = plot_partition_tree(tree)

    .. display_graph::


    2. Order the children by smallest leaf label, color three clusters, and
       display the altitude next to the internal nodes only:

    .. code-block:: python

        fig_clusters = plot_partition_tree(
            tree, altitudes=altitudes, child_ordering="smallest_label",
            n_clusters=3, show_node_labels="all", extra_info="altitude",
            show_extra_info="internal", show_fig=False)

    .. display_graph::


    3. Order the children by largest leaf label, keep the last 4 merges (the
       collapsed subtrees are drawn as leaves) and display all the named
       criteria next to all the nodes:

    .. code-block:: python

        fig_truncated = plot_partition_tree(
            tree, altitudes=altitudes, child_ordering="largest_label",
            lastp=4, extra_info=["altitude", "num_descendants", "area",
                                 "depth", "height", "dynamics"],
            show_extra_info="all", show_fig=False)

    .. display_graph::


    4. Display the complete dendrogram of a larger tree with four clusters,
       ordered by subtree size, with identifiers and information on the
       internal nodes only:

    .. code-block:: python

        fig_complete = plot_partition_tree(
            big_tree, altitudes=big_altitudes,
            child_ordering="smallest_subtree", lastp=None, n_clusters=4,
            show_node_labels="internal", extra_info=["altitude", "area"],
            show_extra_info=True, show_fig=False)

    .. display_graph::


    5. Display custom values instead of the identifiers, with a larger
       drawing:

    .. code-block:: python

        fig_custom_values = plot_partition_tree(
            tree, altitudes=altitudes, child_ordering="largest_subtree",
            show_node_labels=False,
            extra_info={"test": [0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 1, 1, 2, 3, 5, 8, 13, 21]},
            line_width=6, node_size=28, label_size=14, show_fig=False)

    .. display_graph::


    6. Order the children by lowest altitude and display a custom score next
       to the leaves only:

    .. code-block:: python

        fig_leaf_info = plot_partition_tree(
            tree, altitudes=altitudes, child_ordering="smallest_altitude",
            show_node_labels="leaves", n_clusters=2,
            extra_info={"score": np.linspace(0, 1, tree.num_vertices())},
            show_extra_info="leaves", show_fig=False)

    .. display_graph::


    7. Order the children by highest altitude, show the extra information
       only when hovering the nodes, use a thinner drawing and add a title:

    .. code-block:: python

        fig_hover_info = plot_partition_tree(
            point_tree, altitudes=point_altitudes,
            child_ordering="largest_altitude", show_node_labels=True,
            extra_info=["altitude", "area"], show_extra_info=False,
            line_width=1, node_size=4, label_size=8,
            title="Partition tree of a point graph", show_fig=False)

    .. display_graph::

    :param tree: Input hierarchy to plot.
    :param altitudes: Optional array of altitudes, with one value per tree node.
        If ``None``, integer topological heights are computed with
        :func:`~higra.attribute_topological_height`; leaves have altitude zero.
    :param title: Figure title. The default, ``None``, displays no title.
    :param show_fig: If ``True``, display the figure interactively. If
        ``False``, return it without displaying it. The default is ``True``.
    :param show_node_labels: Nodes whose identifier is displayed inside their
        marker: ``"internal"`` (or ``True``) for the internal nodes,
        ``"leaves"`` for the leaves, ``"all"`` for both, and ``False`` for
        none. The default is ``"all"``.
    :param extra_info: Information displayed when hovering a node and in a
        label next to the nodes selected by ``show_extra_info``. Either a name
        or a list of names among ``"altitude"``, ``"num_descendants"`` (number
        of nodes strictly below the node), ``"area"`` (number of leaves below
        the node), ``"depth"``, ``"height"`` and ``"dynamics"``, or a
        dictionary mapping a label to an array with one value per node of
        ``tree``. The default, ``None``, displays no information.
    :param show_extra_info: Nodes with an ``extra_info`` label next to them:
        ``"internal"`` (or ``True``) for the internal nodes, ``"leaves"`` for
        the leaves, ``"all"`` for both, and ``False`` for none. Hovering a node
        always shows its information. The default is ``"all"``.
    :param line_width: Positive width of the branches, in pixels. The default
        is ``2``.
    :param node_size: Positive diameter of the node markers, in pixels. A
        marker displaying an identifier is enlarged when needed to contain it.
        The default is ``20``.
    :param label_size: Positive font size of the node identifiers and of the
        ``extra_info`` labels. The default is ``10``.
    :param n_clusters: Number of clusters to color in the dendrogram. Values
        greater than one select a cut for that many clusters; the default,
        ``0``, disables cluster coloring.
    :param lastp: Number of last merges to display using SciPy's ``lastp``
        truncation mode. The subtrees below these merges are collapsed and
        drawn as leaves labeled with the identifier of their root node. Set to
        ``None`` to display the complete dendrogram. The default is ``30``.
    :param child_ordering: Criterion for ordering the two children at every
        merge. One of ``"default"``, ``"smallest_label"``, ``"largest_label"``,
        ``"smallest_subtree"``, ``"largest_subtree"``,
        ``"smallest_altitude"``, or ``"largest_altitude"``. See the child
        ordering definitions above. The default is ``"default"``.
    :return: A :class:`plotly.graph_objects.Figure` representing the dendrogram.
    """
    assert __scipy_available, "The plot tree function requires scipy"
    assert __plotly_available, "The plot tree function requires plotly"

    # Validation and per-node values, computed on the input tree
    _check_positive_numbers(line_width=line_width, node_size=node_size, label_size=label_size)
    show_internal_labels, show_leaf_labels = _get_label_targets(show_node_labels)
    show_internal_info, show_leaf_info = _get_label_targets(show_extra_info, "show_extra_info")
    altitudes = _get_altitudes(tree, altitudes)
    extra_infos = _get_node_extra_info(tree, altitudes, extra_info)

    if np.max(tree.num_children()) > 2:
        tree, node_map = hg.tree_2_binary_tree(tree)
        altitudes = altitudes[node_map]
        extra_infos = {name: values[node_map] for name, values in extra_infos.items()}
    num_leaves = tree.num_leaves()

    # Dendrogram layout
    linkage_matrix = hg.binary_hierarchy_to_scipy_linkage_matrix(tree, altitudes)
    linkage_matrix = _order_linkage_matrix(linkage_matrix, child_ordering)
    truncation = {} if lastp is None else dict(truncate_mode="lastp", p=lastp)
    dendrogram_data = dendrogram(
        linkage_matrix, no_plot=True, above_threshold_color="k",
        color_threshold=_get_color_threshold(linkage_matrix, n_clusters), **truncation)
    color_map = _get_cluster_color_map(dendrogram_data)
    nodes = _get_dendrogram_nodes(linkage_matrix, dendrogram_data, color_map)
    hover_texts = _get_hover_texts(nodes["id"], altitudes, num_leaves, extra_infos,
                                   nodes["num_collapsed_leaves"])
    show_labels = np.where(nodes["is_leaf"], show_leaf_labels, show_internal_labels)
    marker_sizes = _get_marker_sizes(nodes["id"], show_labels, node_size, label_size)

    # Drawing
    fig = go.Figure()
    _add_dendrogram_branches(fig, dendrogram_data, color_map, line_width)
    for name, mask, show in (("Leaves", nodes["is_leaf"], show_leaf_labels),
                             ("Internal nodes", ~nodes["is_leaf"], show_internal_labels)):
        _add_nodes(fig, np.stack([nodes["x"][mask], nodes["y"][mask]], axis=1),
                   nodes["id"][mask], hover_texts[mask], marker_sizes[mask],
                   show_labels=show, fill_color="white", line_color=nodes["color"][mask],
                   label_size=label_size, name=name)
    show_info = np.where(nodes["is_leaf"], show_leaf_info, show_internal_info)
    if extra_infos and np.any(show_info):
        _add_extra_info_labels(fig, nodes, show_info, marker_sizes, extra_infos, label_size)

    min_altitude, max_altitude, pad = _get_altitude_layout(altitudes, num_leaves)
    yaxis = dict(title="Altitude", showgrid=False, zeroline=False,
                 range=[min(0, min_altitude) - pad, max_altitude + pad])
    if np.all(np.equal(altitudes, np.rint(altitudes))):
        yaxis.update(tickmode="linear", tick0=0, dtick=1)

    top_margin = 60 if title else 40
    if extra_infos and np.any(show_info):
        top_margin += _get_extra_info_label_height(extra_infos, label_size)
    fig.update_layout(
        title=title,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=yaxis,
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=top_margin, b=40),
    )

    if show_fig:
        fig.show()
    return fig


def plot_partition_tree_3d(tree, *, altitudes=None, title=None, show_fig=True,
                           show_node_labels=False, extra_info=None, line_width=4.0,
                           node_size=7.0, label_size=10.0, image=None, z_scale=1.0,
                           node_placement="centroid", highlight="all"):
    """Plot a partition tree above its corresponding image in 3D.

    Let :math:`T=(V,E)` be a rooted tree with :math:`N` leaves :math:`L` and
    internal nodes :math:`I`. Each node :math:`v` receives a position
    :math:`(x(v), y(v), z(v))`.

    **Leaves.** If ``image`` of shape :math:`H \\times W` is given, the leaves
    are its pixels (:math:`N=HW`, row-major order): leaf :math:`i` at row
    :math:`r_i` and column :math:`c_i` is placed at

    .. math::

        x(i)=c_i, \\qquad y(i)=H-1-r_i

    Without image, the leaves are placed automatically on a near-square grid
    with :math:`W=\\lceil\\sqrt{N}\\rceil` columns and
    :math:`H=\\lceil N/W\\rceil` rows, using the same formulas. This layout
    carries no information about the underlying graph: it only spreads the
    leaves so that the tree is readable.

    **Internal nodes.** The area of a node is the number of its descendant
    leaves, :math:`a(v)=|L(v)|`. The position of an internal node in the
    :math:`(x, y)` plane depends on ``node_placement``:

    - ``"centroid"`` (default): the node is placed at the area-weighted
      centroid of its children :math:`C(v)`, which is also the centroid of its
      descendant leaves:

      .. math::

          (x(v), y(v)) = \\frac{\\sum_{u \\in C(v)} a(u)\\,(x(u), y(u))}
                             {\\sum_{u \\in C(v)} a(u)}
                       = \\frac{1}{a(v)} \\sum_{i \\in L(v)} (x(i), y(i))

      A node is therefore drawn closer to its largest children.

    - ``"midpoint"``: the node is placed at the unweighted mean of the
      positions of its children, whatever their sizes:

      .. math::

          (x(v), y(v)) = \\frac{1}{|C(v)|} \\sum_{u \\in C(v)} (x(u), y(u))

      For a binary node, this is the middle of the segment joining its two
      children, as in a 2D dendrogram.

    **Altitudes.** The altitudes :math:`h(v)` are given by ``altitudes`` or
    computed as the topological height (:math:`h(i)=0` on leaves and
    :math:`h(v)=1+\\max_{u\\in C(v)}h(u)` otherwise). With :math:`s` =
    ``z_scale`` and :math:`h_{min}` the minimal altitude over the internal
    nodes:

    .. math::

        z(v) = s\\,h(v) \\;(v \\in I), \\qquad
        z(i) = s\\,h_{min} - 1 \\;(i \\in L)

    The leaves are therefore always one unit below the internal nodes of
    lowest altitude: altitude 0 is not drawn at the pixel level. Each node is
    linked to its parent by a vertical segment followed by a horizontal
    segment at the altitude of the parent. The vertical axis is graduated with
    the altitudes :math:`h`, with a horizontal line at each graduation.

    **Hovering.** Hovering a node shows its identifier, its altitude and the
    values requested with ``extra_info``. Depending on ``highlight``, it also
    highlights the contour of the region of the node in the image (the pixels
    of its descendant leaves), and the branches and markers of its subtree.
    The highlight is computed by JavaScript code run in the page displaying
    the figure: it is active in the figure displayed with ``show_fig=True``.
    To keep it when displaying or exporting the returned figure later, pass
    this code to Plotly, which stores it in ``fig.layout.meta``:

    .. code-block:: python

        script = fig.layout.meta["highlight_script"]
        fig.show(post_script=script)
        fig.write_html("tree.html", post_script=script)

    **Image.** If ``image`` is given, it is drawn as a textured rectangle
    (one colored square per pixel) at the leaf level, so that the
    :math:`(x, y)` axes show the image. This requires the leaves to be the
    pixels of the image, i.e. a tree computed on the pixel graph of this
    image.

    Complexity

    Let :math:`n=|V|` be the number of nodes of ``tree``. Computing the
    positions and the branches takes :math:`\\mathcal{O}(n)`, and building the
    image rectangle takes :math:`\\mathcal{O}(N)`. The overall runtime and
    memory complexity is :math:`\\mathcal{O}(n)` since :math:`N \\le n`.

    Example
    =======

    The examples use binary partition trees built on a 3x3 grayscale image,
    a 6x6 grayscale image, a 3x4 color image made of four noisy regions, and
    an arbitrary graph whose vertices are points of the plane:

    .. code-block:: python

        image = np.asarray(((10, 14, 30), (12, 45, 36), (22, 52, 70)))
        graph = hg.get_4_adjacency_graph(image.shape)
        tree, altitudes = hg.bpt_canonical(
            graph, hg.weight_graph(graph, image, hg.WeightFunction.L1))

        yy, xx = np.mgrid[0:6, 0:6]
        big_image = 20 * xx + 10 * yy + \\
            np.random.default_rng(2).normal(0, 8, (6, 6))
        big_graph = hg.get_4_adjacency_graph(big_image.shape)
        big_tree, big_altitudes = hg.bpt_canonical(
            big_graph,
            hg.weight_graph(big_graph, big_image, hg.WeightFunction.L1))

        rng = np.random.default_rng(1)
        regions = np.asarray(((0, 0, 1, 1), (0, 2, 1, 3), (2, 2, 3, 3)))
        colors = np.asarray(((220, 40, 40), (40, 120, 220),
                             (40, 160, 60), (240, 200, 40)), dtype=float)
        rgb = np.clip(colors[regions] + rng.normal(0, 25, (3, 4, 3)),
                      0, 255).astype(np.uint8)
        rgb_graph = hg.get_4_adjacency_graph(rgb.shape[:2])
        rgb_tree, rgb_altitudes = hg.bpt_canonical(
            rgb_graph, hg.weight_graph(
                rgb_graph, rgb.reshape(-1, 3).astype(float),
                hg.WeightFunction.L2))

        points = np.asarray(((0, 0), (2.1, 1.4), (0.9, 3.2), (3.9, 2.4),
                             (3.2, 0.8), (6.0, 4.5), (2.2, 6.3)))
        point_graph = hg.UndirectedGraph(7)
        point_graph.add_edges((0, 1, 1, 2, 3, 3, 4, 5),
                              (1, 2, 4, 6, 4, 5, 5, 6))
        sources, targets = point_graph.edge_list()
        point_tree, point_altitudes = hg.bpt_canonical(
            point_graph,
            np.linalg.norm(points[sources] - points[targets], axis=1))

    1. Plot a tree above its image with all the default parameters
       (centroid placement, topological heights, no identifiers, hovering
       highlights the region and the subtree) and display the figure:

    .. code-block:: python

        fig_default = plot_partition_tree_3d(tree, image=image)

    .. display_graph::


    2. Place the nodes at the middle of their children, display all the
       identifiers, and show extra information when hovering the nodes:

    .. code-block:: python

        fig_midpoint = plot_partition_tree_3d(
            tree, image=image, altitudes=altitudes, z_scale=0.2,
            node_placement="midpoint", show_node_labels="all",
            extra_info=["altitude", "area", "num_descendants"],
            highlight=True, show_fig=False)

    .. display_graph::


    3. Plot a larger tree, with the identifiers of the internal nodes, and
       only highlight the image region of the hovered node:

    .. code-block:: python

        fig_region = plot_partition_tree_3d(
            big_tree, image=big_image, altitudes=big_altitudes, z_scale=0.1,
            node_placement="midpoint", show_node_labels=True,
            highlight="region", show_fig=False)

    .. display_graph::


    4. Plot a tree above a color image with explicit centroid placement,
       the identifiers of the leaves, the area of the hovered node, a larger
       drawing, and only highlight the subtree of the hovered node:

    .. code-block:: python

        fig_color = plot_partition_tree_3d(
            rgb_tree, image=rgb, altitudes=rgb_altitudes, z_scale=0.03,
            node_placement="centroid", show_node_labels="leaves",
            extra_info="area", highlight="subtree", line_width=8,
            node_size=12, label_size=12, show_fig=False)

    .. display_graph::


    5. Plot a tree built on an arbitrary graph, without image, with a
       thinner drawing, custom hover values, no highlight and a title:

    .. code-block:: python

        fig_graph = plot_partition_tree_3d(
            point_tree, altitudes=point_altitudes,
            show_node_labels="internal",
            extra_info={"index": np.arange(point_tree.num_vertices())},
            highlight=False, line_width=1, node_size=4, label_size=8,
            title="Partition tree of a point graph", show_fig=False)

    .. display_graph::

    :param tree: Input partition tree. If ``image`` is given, its leaves must
        be the pixels of the image in row-major order.
    :param altitudes: Optional array of altitudes, with one value per tree
        node. If ``None``, integer topological heights are computed with
        :func:`~higra.attribute_topological_height`; leaves have altitude zero.
    :param title: Figure title. The default, ``None``, displays no title.
    :param show_fig: If ``True``, display the figure interactively. If
        ``False``, return it without displaying it. The default is ``True``.
    :param show_node_labels: Nodes whose identifier is displayed inside their
        marker: ``"internal"`` (or ``True``) for the internal nodes,
        ``"leaves"`` for the leaves, ``"all"`` for both, and ``False`` for
        none. Displaying the leaf labels of a large image makes the figure
        slow and unreadable. The default is ``False``.
    :param extra_info: Information displayed when hovering a node. Either a
        name or a list of names among ``"altitude"``, ``"num_descendants"``
        (number of nodes strictly below the node), ``"area"`` (number of
        leaves below the node), ``"depth"``, ``"height"`` and ``"dynamics"``,
        or a dictionary mapping a label to an array with one value per node of
        ``tree``. The default, ``None``, displays no information.
    :param line_width: Positive width of the branches, in pixels. The default
        is ``4``.
    :param node_size: Positive diameter of the internal nodes, in pixels. The
        leaves are drawn at half this size. A marker displaying an identifier
        is enlarged when needed to contain it. The default is ``7``.
    :param label_size: Positive font size of the node identifiers. The
        default is ``10``.
    :param image: Optional grayscale image with shape ``(height, width)``, or
        color image with shape ``(height, width, 3)`` or
        ``(height, width, 4)``, from which the tree was computed. It defines
        the position of the leaves and is displayed under the tree. If
        ``None``, the leaves are placed on an automatic near-square grid and
        no image is displayed.
    :param z_scale: Positive finite factor applied to the node altitudes. Use
        a small value when altitudes are large compared to the image size.
        The default is ``1.0``.
    :param node_placement: Position of the internal nodes in the
        :math:`(x, y)` plane: ``"centroid"`` for the centroid of their
        descendant leaves, or ``"midpoint"`` for the mean of their children
        (see above). The default is ``"centroid"``.
    :param highlight: What hovering a node highlights: ``"region"`` for the
        contour of its region in the image (only when ``image`` is given),
        ``"subtree"`` for the branches and markers of its subtree, ``"all"``
        (or ``True``) for both, and ``False`` for nothing. The default is
        ``"all"``.
    :return: A :class:`plotly.graph_objects.Figure` with the interactive 3D
        plot.
    """
    assert __plotly_available, "The 3D plot requires plotly"

    # Validation and per-node values
    _check_positive_numbers(line_width=line_width, node_size=node_size,
                            label_size=label_size, z_scale=z_scale)
    show_internal_labels, show_leaf_labels = _get_label_targets(show_node_labels)
    altitudes = _get_altitudes(tree, altitudes)
    extra_infos = _get_node_extra_info(tree, altitudes, extra_info)
    num_leaves = tree.num_leaves()
    if show_leaf_labels and num_leaves > 50:
        warnings.warn(
            f"Displaying the labels of {num_leaves} leaves makes the figure slow "
            "and unreadable.", stacklevel=2)

    highlight_region, highlight_subtree = _get_highlight_targets(highlight)

    # 3D layout
    leaf_xy, pixel_colors, grid_width = _get_leaf_layout_3d(image, num_leaves)
    positions = _get_node_positions_3d(tree, leaf_xy, altitudes, z_scale, node_placement)
    node_ids = np.arange(tree.num_vertices())
    leaves = node_ids < num_leaves
    hover_texts = _get_hover_texts(node_ids, altitudes, num_leaves, extra_infos)
    marker_sizes = np.where(
        leaves,
        _get_marker_sizes(node_ids, show_leaf_labels, node_size / 2, label_size),
        _get_marker_sizes(node_ids, show_internal_labels, node_size, label_size))
    branch_path = _get_branch_paths_3d(tree, positions, marker_sizes)
    image_z = positions[0, 2] - 0.05  # just below the leaves, so as not to hide them
    tick_altitudes = _get_altitude_ticks(altitudes[num_leaves:])

    # Drawing
    fig = go.Figure()
    if pixel_colors is not None:
        _add_image_3d(fig, positions[leaves], pixel_colors, image_z)
    _add_branches_3d(fig, branch_path, line_width)
    _add_nodes(fig, positions[leaves], node_ids[leaves], hover_texts[leaves],
               marker_sizes[leaves], show_labels=show_leaf_labels,
               fill_color="white" if show_leaf_labels else _LEAF_COLOR_3D,
               line_color=_LEAF_COLOR_3D, label_size=label_size, name="Leaves")
    _add_nodes(fig, positions[~leaves], node_ids[~leaves], hover_texts[~leaves],
               marker_sizes[~leaves], show_labels=show_internal_labels, fill_color="white",
               line_color="black", label_size=label_size, name="Internal nodes")
    highlight_script = None
    if (highlight_region and pixel_colors is not None) or highlight_subtree:
        highlight_script = _add_highlight_3d(
            fig, tree, grid_width, image_z, line_width,
            highlight_region=highlight_region and pixel_colors is not None,
            highlight_subtree=highlight_subtree, show_leaf_labels=show_leaf_labels)

    axis = dict(showgrid=False, zeroline=False)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="Image X", **axis),
            yaxis=dict(title="Image Y", **axis),
            zaxis=dict(title="Altitude", zeroline=False, showgrid=True,
                       tickvals=tick_altitudes * z_scale,
                       ticktext=[f"{value:.4g}" for value in tick_altitudes]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=50 if title else 0, b=0),
    )

    if show_fig:
        fig.show(post_script=highlight_script)
    return fig


# ---------------------------------------------------------------------------
# Helpers shared by the 2D and 3D plots
# ---------------------------------------------------------------------------

def _check_positive_numbers(**values):
    """Raise a ValueError if one of the given values is not a positive finite number."""
    for name, value in values.items():
        if (not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be a positive finite number, got {value!r}")


def _get_label_targets(value, name="show_node_labels"):
    """Get whether a label is displayed on the internal nodes and on the leaves, from a
    parameter that is a boolean (``True`` for the internal nodes) or one of ``"internal"``,
    ``"leaves"`` and ``"all"``."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value), False

    valid_targets = ("internal", "leaves", "all")
    if not isinstance(value, str) or value not in valid_targets:
        raise ValueError(f"{name} must be a boolean or one of {valid_targets}, got {value!r}")

    return value in ("internal", "all"), value in ("leaves", "all")


def _get_highlight_targets(highlight):
    """Get whether hovering a node highlights its image region and its subtree, from a
    parameter that is a boolean (``True`` for both) or one of ``"region"``, ``"subtree"``
    and ``"all"``."""
    if isinstance(highlight, (bool, np.bool_)):
        return bool(highlight), bool(highlight)

    valid_targets = ("region", "subtree", "all")
    if not isinstance(highlight, str) or highlight not in valid_targets:
        raise ValueError(
            f"highlight must be a boolean or one of {valid_targets}, got {highlight!r}")

    return highlight in ("region", "all"), highlight in ("subtree", "all")


def _get_altitudes(tree, altitudes):
    """Get the node altitudes as a float array, the topological height by default."""
    if altitudes is None:
        altitudes = hg.attribute_topological_height(tree)

    altitudes = np.asarray(altitudes, dtype=float).reshape(-1)
    if len(altitudes) != tree.num_vertices():
        raise ValueError(f"Expected {tree.num_vertices()} altitudes, got {len(altitudes)}")
    if not np.all(np.isfinite(altitudes)):
        raise ValueError("altitudes must contain only finite values")
    return altitudes


def _get_node_extra_info(tree, altitudes, extra_info):
    """Get the requested per-node values, as a dictionary mapping a display name to an array."""
    num_nodes = tree.num_vertices()
    criteria = {
        "altitude": lambda: altitudes,
        "num_descendants": lambda: hg.accumulate_and_add_sequential(
            tree,
            np.ones(num_nodes, dtype=np.int64),
            np.ones(tree.num_leaves(), dtype=np.int64),
            hg.Accumulators.sum,
        ) - 1,
        "area": lambda: hg.attribute_area(tree),
        "depth": lambda: hg.attribute_depth(tree),
        "height": lambda: hg.attribute_height(tree, altitudes),
        "dynamics": lambda: hg.attribute_dynamics(tree, altitudes),
    }
    node_info = {}

    if extra_info is None:
        return node_info

    if isinstance(extra_info, (str, list, tuple, np.ndarray)):
        if isinstance(extra_info, str):
            extra_info = [extra_info]
        for info in extra_info:
            if info not in criteria:
                raise ValueError(
                    f"extra_info criteria must be one of {tuple(criteria)}, got {info!r}")
            node_info[info.capitalize()] = np.asarray(criteria[info]())

    elif isinstance(extra_info, dict):
        for info, values in extra_info.items():
            values = np.asarray(values)
            if values.ndim == 0 or values.shape[0] != num_nodes:
                raise ValueError(
                    f"extra_info dictionary values must be arrays of length {num_nodes}, "
                    f"got shape {values.shape} for key {info!r}")
            node_info[str(info)] = values

    else:
        raise ValueError(
            "extra_info must be a name, a list of names or a dictionary of arrays, "
            f"got {type(extra_info).__name__}")

    return node_info


def _format_extra_info(node_id, extra_infos):
    """Format the extra information of a node as a multi-line Plotly text."""
    lines = []
    for name, values in extra_infos.items():
        value = values[node_id]
        if np.issubdtype(np.asarray(value).dtype, np.number):
            lines.append(f"{name}: {value:.4g}")
        else:
            lines.append(f"{name}: {value}")
    return "<br>".join(lines)


def _get_hover_texts(node_ids, altitudes, num_leaves, extra_infos, num_collapsed_leaves=None):
    """Get the hover text of the given nodes: identifier, altitude and extra information.

    The altitude of the internal nodes is written once, in the extra information if it
    contains it. ``num_collapsed_leaves`` gives, for each node, the number of leaves of the
    subtree it represents when it is drawn collapsed in a truncated dendrogram, and 0
    otherwise.
    """
    show_altitude = "Altitude" not in extra_infos
    texts = []
    for k, node_id in enumerate(node_ids):
        if node_id < num_leaves:
            lines = [f"Leaf {node_id}"]
        else:
            lines = [f"Node {node_id}"]
            if show_altitude:
                lines.append(f"Altitude: {altitudes[node_id]:.4g}")
            if num_collapsed_leaves is not None and num_collapsed_leaves[k] > 0:
                lines.append(f"Collapsed subtree of {num_collapsed_leaves[k]} leaves")
        if extra_infos:
            lines.append(_format_extra_info(node_id, extra_infos))
        texts.append("<br>".join(lines))
    return np.asarray(texts, dtype=object)


def _get_marker_sizes(node_ids, show_labels, node_size, label_size):
    """Get the diameter of the node markers: ``node_size``, enlarged to contain the node
    identifier where it is displayed (``show_labels`` is a boolean or a boolean array)."""
    node_ids = np.asarray(node_ids)
    num_digits = np.floor(np.log10(np.maximum(node_ids, 1))).astype(int) + 1
    label_fit = label_size * (0.6 * num_digits + 0.8)
    return np.where(show_labels, np.maximum(node_size, label_fit), float(node_size))


def _add_nodes(fig, positions, node_ids, hover_texts, marker_sizes, *, show_labels,
               fill_color, line_color, label_size, name):
    """Add the given nodes as a single trace, in 2D or 3D depending on ``positions``.

    The identifiers are written inside the markers when ``show_labels`` is ``True``.
    """
    trace_type = go.Scatter if positions.shape[1] == 2 else go.Scatter3d
    fig.add_trace(
        trace_type(
            **dict(zip("xyz", positions.T)),
            mode="markers+text" if show_labels else "markers",
            marker=dict(size=marker_sizes, color=fill_color, opacity=1,
                        line=dict(width=1.5, color=np.asarray(line_color).tolist())),
            text=[str(node_id) for node_id in node_ids] if show_labels else None,
            textposition="middle center",
            textfont=dict(size=label_size, color="black"),
            hovertext=list(hover_texts),
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
            name=name,
        )
    )


# ---------------------------------------------------------------------------
# 2D dendrogram helpers
# ---------------------------------------------------------------------------

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


def _get_dendrogram_nodes(linkage_matrix, dendrogram_data, color_map):
    """Get the nodes drawn in a dendrogram, as a dictionary of arrays with one entry per node.

    The drawn nodes are the leaves of the dendrogram (the leaves of the tree or, when it is
    truncated, the collapsed subtrees), then the internal nodes above the truncation, i.e.
    whose two children are drawn. Each node gets its position, the color of its branch, its
    side relative to its sibling and its number of collapsed leaves (0 if not collapsed).
    """
    num_leaves = len(linkage_matrix) + 1

    def point_key(x, y):
        return round(float(x), 6), round(float(y), 6)

    # A segment of SciPy is drawn as [left foot, left top, right top, right foot]: its top
    # center is the node and its feet are the children.
    node_color = {}
    foot_color = {}
    for x, y, color_key in zip(dendrogram_data["icoord"], dendrogram_data["dcoord"],
                               dendrogram_data["color_list"]):
        color = color_map.get(color_key, "black")
        node_color[point_key((x[1] + x[2]) / 2, y[1])] = color
        foot_color[point_key(x[0], y[0])] = color
        foot_color[point_key(x[3], y[3])] = color

    node_x = {leaf: 5 + 10 * rank for rank, leaf in enumerate(dendrogram_data["leaves"])}
    node_ids = list(dendrogram_data["leaves"])
    ys = [0.0] * len(node_ids)
    colors = [foot_color.get(point_key(node_x[leaf], 0), "black") for leaf in node_ids]
    num_collapsed_leaves = [0 if leaf < num_leaves else int(linkage_matrix[leaf - num_leaves, 3])
                            for leaf in node_ids]

    for i, (left, right, altitude, _) in enumerate(linkage_matrix):
        left, right = int(left), int(right)
        if left not in node_x or right not in node_x:
            continue
        node_id = num_leaves + i
        node_x[node_id] = (node_x[left] + node_x[right]) / 2
        node_ids.append(node_id)
        ys.append(altitude)
        colors.append(node_color.get(point_key(node_x[node_id], altitude), "black"))
        num_collapsed_leaves.append(0)

    on_left = np.zeros(2 * num_leaves - 1, dtype=bool)
    on_left[linkage_matrix[:, 0].astype(int)] = True
    node_ids = np.asarray(node_ids, dtype=np.int64)
    is_leaf = np.zeros(len(node_ids), dtype=bool)
    is_leaf[:len(dendrogram_data["leaves"])] = True
    return dict(
        id=node_ids,
        x=np.asarray([node_x[node_id] for node_id in node_ids], dtype=float),
        y=np.asarray(ys, dtype=float),
        color=np.asarray(colors, dtype=object),
        on_left=on_left[node_ids],
        is_leaf=is_leaf,
        num_collapsed_leaves=np.asarray(num_collapsed_leaves, dtype=np.int64),
    )


def _get_altitude_layout(altitudes, num_leaves):
    """Get the internal altitude bounds and y-axis padding."""
    internal_altitudes = altitudes[num_leaves:]
    internal_min_alt = float(np.min(internal_altitudes))
    internal_max_alt = float(np.max(internal_altitudes))
    altitude_span = internal_max_alt - internal_min_alt
    pad = altitude_span * 0.1 if altitude_span > 0 else 1.0
    return internal_min_alt, internal_max_alt, pad


def _add_dendrogram_branches(fig, dendrogram_data, color_map, line_width):
    """Add the dendrogram segments computed by SciPy, as one trace per color."""
    paths = {}
    for x, y, color_key in zip(dendrogram_data["icoord"], dendrogram_data["dcoord"],
                               dendrogram_data["color_list"]):
        xs, ys = paths.setdefault(color_map.get(color_key, "black"), ([], []))
        xs.extend([*x, None])
        ys.extend([*y, None])

    for color, (xs, ys) in paths.items():
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=line_width),
                hoverinfo="none",
                showlegend=False,
            )
        )


_EXTRA_INFO_LABEL_OFFSET = 25  # vertical offset of the labels above their node, in pixels


def _get_extra_info_label_height(extra_infos, label_size):
    """Get the height in pixels that an extra information label rises above its node."""
    if not extra_infos:
        return 0
    return _EXTRA_INFO_LABEL_OFFSET + 1.3 * label_size * len(extra_infos) + 10


def _add_extra_info_labels(fig, nodes, show_info, marker_sizes, extra_infos, label_size):
    """Add a label with the extra information of the drawn nodes where ``show_info`` is
    ``True``, above the node and on its side of its parent's subtree."""
    annotations = []
    for node_id, x, y, color, on_left, size, show in zip(
            nodes["id"], nodes["x"], nodes["y"], nodes["color"], nodes["on_left"],
            marker_sizes, show_info):
        if not show:
            continue
        annotations.append(dict(
            x=x,
            y=y,
            xref="x",
            yref="y",
            text=_format_extra_info(node_id, extra_infos),
            showarrow=True,
            arrowhead=0,
            arrowcolor=color,
            # Annotations are drawn above traces: stop the arrow at the edge of the
            # node marker so that it seems to go behind it.
            standoff=size / 2 + 1,
            ax=-25 if on_left else 25,
            ay=-_EXTRA_INFO_LABEL_OFFSET,
            xanchor="right" if on_left else "left",
            yanchor="bottom",
            align="left",
            font=dict(size=label_size, color="black"),
            bgcolor="white",
            bordercolor=color,
            borderwidth=1,
            borderpad=4,
        ))
    fig.update_layout(annotations=annotations)


# ---------------------------------------------------------------------------
# 3D tree helpers
# ---------------------------------------------------------------------------

_LEAF_COLOR_3D = "#444444"
_HIGHLIGHT_COLOR_3D = "#e6007e"
# Approximate size, in pixels, of the largest dimension of the scene in the default view.
# Used to convert the radius of the node markers (in pixels) to scene units.
_SCENE_SIZE_PX = 450


def _get_leaf_layout_3d(image, num_leaves):
    """Get the (x, y) position of the leaves, their RGB color if ``image`` is given (``None``
    otherwise), and the width of the grid of leaves.

    The leaves are the pixels of ``image`` or, without image, are placed on a near-square
    grid.
    """
    if image is None:
        width = int(np.ceil(np.sqrt(num_leaves)))
        height = int(np.ceil(num_leaves / width))
        pixel_colors = None
    else:
        image = np.asarray(image)
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
            raise ValueError(
                "image must have shape (height, width), (height, width, 3), "
                "or (height, width, 4)")
        height, width = image.shape[:2]
        if height * width != num_leaves:
            raise ValueError(
                f"The image contains {height * width} pixels, but the tree has {num_leaves} "
                "leaves. The leaves must be the pixels of the image.")
        pixel_colors = _get_pixel_colors(image)

    leaf_ids = np.arange(num_leaves)
    leaf_xy = np.stack([leaf_ids % width, height - 1 - leaf_ids // width], axis=1)
    return leaf_xy.astype(float), pixel_colors, width


def _get_pixel_colors(image):
    """Get the colors of the pixels of an image as an (N, 3) uint8 RGB array. A grayscale
    image is stretched to the full [0, 255] range."""
    if image.ndim == 2:
        gray = image.astype(float).ravel()
        span = gray.max() - gray.min()
        gray = (gray - gray.min()) / span if span > 0 else np.zeros_like(gray)
        return np.repeat(np.round(gray * 255).astype(np.uint8)[:, None], 3, axis=1)

    rgb = image[..., :3].astype(float)
    if np.issubdtype(image.dtype, np.integer):
        rgb = rgb / np.iinfo(image.dtype).max
    return np.round(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8).reshape(-1, 3)


def _get_node_positions_3d(tree, leaf_xy, altitudes, z_scale, node_placement):
    """Get the (x, y, z) position of every node of a 3D partition tree.

    The leaves keep their position ``leaf_xy``, one unit below the lowest internal node. The
    internal nodes are placed at the centroid of their descendant leaves (``"centroid"``) or
    at the mean of the positions of their children (``"midpoint"``), at the height given by
    their altitude.
    """
    valid_placements = ("centroid", "midpoint")
    if node_placement not in valid_placements:
        raise ValueError(
            f"node_placement must be one of {valid_placements}, got {node_placement!r}")

    num_leaves = tree.num_leaves()
    positions = np.zeros((tree.num_vertices(), 3), dtype=float)

    if node_placement == "centroid":
        areas = hg.accumulate_sequential(tree, np.ones(num_leaves), hg.Accumulators.sum)
        positions[:, :2] = hg.accumulate_sequential(
            tree, leaf_xy, hg.Accumulators.sum) / areas[:, None]
    else:
        # Mean of the children computed as (1 / number of children) * (sum of the children).
        # Accumulators.mean is not used: its result is wrong once the children of the
        # tree have been computed (its internal counter is not reset between nodes).
        num_children = tree.num_children(np.arange(tree.num_vertices())).astype(float)
        inverse_num_children = 1.0 / np.maximum(num_children, 1.0)
        positions[:, :2] = hg.accumulate_and_multiply_sequential(
            tree,
            np.repeat(inverse_num_children[:, None], 2, axis=1),
            leaf_xy,
            hg.Accumulators.sum,
        )

    positions[:, 2] = altitudes * z_scale
    positions[:num_leaves, 2] = positions[num_leaves:, 2].min() - 1.0
    return positions


def _get_altitude_ticks(altitudes, max_ticks=8):
    """Get round altitude values spanning ``altitudes``, at most ``max_ticks`` of them."""
    low, high = float(np.min(altitudes)), float(np.max(altitudes))
    if high <= low:
        return np.asarray([low])
    raw_step = (high - low) / max_ticks
    magnitude = 10 ** np.floor(np.log10(raw_step))
    step = next(factor * magnitude for factor in (1, 2, 2.5, 5, 10)
                if factor * magnitude >= raw_step)
    ticks = np.arange(np.ceil(low / step) * step, high + step * 1e-9, step)
    return np.round(ticks, 10)


def _get_branch_paths_3d(tree, positions, marker_sizes):
    """Get the branch of every node except the root, as an array of shape (n - 1, 3, 3):
    start, elbow and end points of a vertical segment up to the altitude of the parent
    followed by a horizontal segment to the parent.

    The branches start and end at the edge of the node markers rather than at their center:
    the markers are flat disks facing the camera, and a branch reaching their center would
    be drawn in front of them when it comes towards the camera. Branches shorter than the
    space between two markers are replaced by NaN.
    """
    children = np.arange(tree.num_vertices() - 1)  # every node except the root
    parents = tree.parents()[children]
    child_xyz = positions[children]
    parent_xyz = positions[parents]

    # Radius of the markers in scene units, in the default view
    extent = np.max(np.ptp(positions, axis=0) + [1, 1, 0])
    gaps = marker_sizes / 2 * extent / _SCENE_SIZE_PX
    start_length = gaps[children]
    vertical_length = parent_xyz[:, 2] - child_xyz[:, 2]
    horizontal = parent_xyz[:, :2] - child_xyz[:, :2]
    horizontal_length = np.linalg.norm(horizontal, axis=1)
    end_length = vertical_length + horizontal_length - gaps[parents]
    direction = horizontal / np.maximum(horizontal_length, 1e-12)[:, None]

    def point_at(length):
        """Point at a given length along the branch, measured from the child."""
        vertical_part = np.minimum(length, vertical_length)
        horizontal_part = np.maximum(length - vertical_length, 0)
        xyz = child_xyz.copy()
        xyz[:, :2] += direction * horizontal_part[:, None]
        xyz[:, 2] += vertical_part
        return xyz

    elbow_length = np.clip(vertical_length, start_length, end_length)
    paths = np.stack([point_at(start_length), point_at(elbow_length), point_at(end_length)],
                     axis=1)
    paths[end_length <= start_length] = np.nan
    return paths


def _add_image_3d(fig, leaf_positions, pixel_colors, image_z):
    """Add the image at height ``image_z``, as one colored square per pixel centered on the
    leaves."""
    corners = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    vertices = (leaf_positions[:, None, :2] + corners[None]).reshape(-1, 2)
    first = 4 * np.arange(len(leaf_positions))[:, None]
    triangles = np.concatenate([first + [0, 1, 2], first + [0, 2, 3]], axis=0)

    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=np.full(len(vertices), image_z),
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            facecolor=np.concatenate([pixel_colors, pixel_colors], axis=0),
            flatshading=True,
            hoverinfo="none",
            showlegend=False,
            name="Image",
        )
    )


def _add_branches_3d(fig, branch_paths, line_width):
    """Add the branches computed by :func:`_get_branch_paths_3d` as a single trace."""
    gap = np.full((len(branch_paths), 1, 3), np.nan)  # interrupts the line between branches
    path = np.concatenate([branch_paths, gap], axis=1).reshape(-1, 3)

    fig.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines",
            line=dict(color="rgba(60, 60, 60, 0.8)", width=line_width),
            hoverinfo="none",
            showlegend=False,
            name="Branches",
        )
    )


def _add_highlight_3d(fig, tree, grid_width, image_z, line_width, *, highlight_region,
                      highlight_subtree, show_leaf_labels):
    """Add the (initially empty) traces that highlight the hovered node, and return the
    JavaScript code that fills them when a node is hovered.

    The branches and the nodes must already be in ``fig``, in the traces named "Branches",
    "Leaves" and "Internal nodes".
    """
    trace_index = {trace.name: index for index, trace in enumerate(fig.data)}
    num_leaves = tree.num_leaves()
    data = dict(
        parents=tree.parents().tolist(),
        numLeaves=num_leaves,
        width=int(grid_width),
        imageZ=float(image_z) + 0.02,  # between the image and the leaves
        color=_HIGHLIGHT_COLOR_3D,
        leafColor="white" if show_leaf_labels else _LEAF_COLOR_3D,
        leafTrace=trace_index["Leaves"],
        nodeTrace=trace_index["Internal nodes"],
        regionTrace=-1,
        subtreeTrace=-1,
        branches=None,
    )

    for name, enabled in (("Highlighted region", highlight_region),
                          ("Highlighted subtree", highlight_subtree)):
        if not enabled:
            continue
        data["regionTrace" if name == "Highlighted region" else "subtreeTrace"] = len(fig.data)
        fig.add_trace(
            go.Scatter3d(
                x=[np.nan], y=[np.nan], z=[np.nan],  # created empty, it would be drawn in white
                mode="lines",
                line=dict(color=_HIGHLIGHT_COLOR_3D, width=line_width + 2),
                hoverinfo="none",
                showlegend=False,
                name=name,
            )
        )

    if highlight_subtree:
        branches = fig.data[trace_index["Branches"]]
        path = np.stack([branches.x, branches.y, branches.z], axis=1).reshape(-1, 4, 3)
        path = np.round(path[:, :3], 6).reshape(-1)
        data["branches"] = [None if np.isnan(value) else float(value) for value in path]

    script = _HIGHLIGHT_SCRIPT_3D.replace("__DATA__", json.dumps(data))
    fig.update_layout(meta=dict(highlight_script=script))
    return script


# JavaScript run in the page displaying a 3D tree: on hover, computes the descendants of the
# hovered node (parents are after their children), then draws the contour of their pixels
# and copies their branches into the highlight traces, and colors their markers.
_HIGHLIGHT_SCRIPT_3D = """
(function () {
    var gd = document.getElementById("{plot_id}");
    var D = __DATA__;
    var n = D.parents.length, numLeaves = D.numLeaves, current = -1;

    function descendants(node) {
        var inside = new Uint8Array(n);
        if (node < 0) return inside;
        inside[node] = 1;
        for (var i = node - 1; i >= 0; i--) inside[i] = inside[D.parents[i]];
        return inside;
    }

    function regionContour(inside) {
        var w = D.width, h = Math.ceil(numLeaves / w), z = D.imageZ;
        var xs = [], ys = [], zs = [];
        function edge(x0, y0, x1, y1) {
            xs.push(x0, x1, NaN); ys.push(y0, y1, NaN); zs.push(z, z, NaN);
        }
        for (var i = 0; i < numLeaves; i++) {
            if (!inside[i]) continue;
            var c = i % w, r = Math.floor(i / w), x = c, y = h - 1 - r;
            if (r === 0 || !inside[i - w]) edge(x - .5, y + .5, x + .5, y + .5);
            if (i + w >= numLeaves || !inside[i + w]) edge(x - .5, y - .5, x + .5, y - .5);
            if (c === 0 || !inside[i - 1]) edge(x - .5, y - .5, x - .5, y + .5);
            if (c === w - 1 || !inside[i + 1]) edge(x + .5, y - .5, x + .5, y + .5);
        }
        return [xs, ys, zs];
    }

    function subtreeBranches(inside, node) {
        var p = D.branches, xs = [], ys = [], zs = [];
        for (var c = 0; c < node; c++) {
            if (!inside[c]) continue;
            for (var k = 0; k < 3; k++) {
                var j = 3 * (3 * c + k);
                xs.push(p[j]); ys.push(p[j + 1]); zs.push(p[j + 2]);
            }
            xs.push(NaN); ys.push(NaN); zs.push(NaN);
        }
        return [xs, ys, zs];
    }

    function markerColors(inside, from, to, base) {
        var colors = [];
        for (var i = from; i < to; i++) colors.push(inside[i] ? D.color : base);
        return colors;
    }

    function update(node) {
        var inside = descendants(node), traces = [], lines = [];
        var xs = [], ys = [], zs = [];
        if (D.regionTrace >= 0) {
            traces.push(D.regionTrace); lines.push(regionContour(inside));
        }
        if (D.subtreeTrace >= 0) {
            traces.push(D.subtreeTrace); lines.push(subtreeBranches(inside, node));
        }
        lines.forEach(function (l) {
            // A 3D line trace must never be empty: emptying it breaks the WebGL rendering
            if (l[0].length === 0) l = [[NaN], [NaN], [NaN]];
            xs.push(l[0]); ys.push(l[1]); zs.push(l[2]);
        });
        Plotly.restyle(gd, {x: xs, y: ys, z: zs}, traces);
        if (D.subtreeTrace >= 0) {
            Plotly.restyle(gd, {"marker.color": [
                markerColors(inside, 0, numLeaves, D.leafColor),
                markerColors(inside, numLeaves, n, "white")]}, [D.leafTrace, D.nodeTrace]);
        }
    }

    gd.on("plotly_hover", function (event) {
        var point = event.points[0], node;
        if (point.curveNumber === D.leafTrace) node = point.pointNumber;
        else if (point.curveNumber === D.nodeTrace) node = numLeaves + point.pointNumber;
        else return;
        if (node !== current) { current = node; update(node); }
    });
    gd.on("plotly_unhover", function () {
        if (current >= 0) { current = -1; update(-1); }
    });
})();
"""
