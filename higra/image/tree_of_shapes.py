############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import higra as hg
import numpy as np


def component_tree_tree_of_shapes_image2d(image, padding='mean', original_size=True, immersion=True, exterior_vertex=0):
    """
    Tree of shapes of a 2d image.

    The Tree of Shapes was described in [1]_.
    The algorithm used in this implementation was first described in [2]_.

    The tree is computed in the interpolated multivalued Khalimsky space to provide a continuous and autodual representation of
    input image.

    Possible values of `padding` are `'none'`, `'mean'`, and `'zero'`.
    If `padding` is different from 'none', an extra border of pixels is added to the input image before
    anything else. This will ensure the existence of a shape encompassing all the shapes inside the input image
    (if exterior_vertex is inside the extra border): this shape will be the root of the tree.
    The padding value can be:

      - 0 if :attr:`padding` is equal to ``"zero"``;
      - the mean value of the boundary pixels of the input image if :attr:`padding` is equal to ``"mean"``.

    If :attr:`original_size` is ``True``, all the nodes corresponding to pixels not belonging to the input image
    are removed (except for the root node).
    If :attr:`original_size` is ``False``, the returned tree is the tree constructed in the interpolated/padded space.
    In practice if the size of the input image is :math:`(h, w)`, the leaves of the returned tree will correspond to an
    image of size:

      - :math:`(h, w)` if :attr:`original_size` is ``True``;
      - :math:`(h * 2 - 1, w * 2 - 1)` is :attr:`original_size` is ``False`` and :attr:`padding` is ``"none"``; and
      - :math:`((h + 2) * 2 - 1, (w + 2) * 2 - 1)` otherwise.

    :Advanced options:

    :attr:`immersion` defines if the initial image should be first converted as an equivalent continuous representation
    called a *plain map*. If :attr:`immersion` is set to ``False`` the level lines of the shapes of the image may intersect
    (if the image is not well composed) and the result of the algorithm is undefined (the algorithm will arbitrarily
    break level lines to get a set of shapes forming a tree). If :attr:`immersion` is ``False``, the result size will be:

      - :math:`(h, w)` if :attr:`original_size` is ``True`` or if :attr:`padding` is ``"none"``;
      - :math:`(h + 2, w + 2)` otherwise.

    :attr:`Exterior_vertex` defines the linear coordinates of the pixel corresponding to the exterior
    (interior and exterior of a shape is defined with respect to this point). The coordinate of this point must be
    given in the padded/interpolated space.

    .. [1] Pa. Monasse, and F. Guichard, "Fast computation of a contrast-invariant image representation," \
    Image Processing, IEEE Transactions on, vol.9, no.5, pp.860-872, May 2000

    .. [2] Th. Géraud, E. Carlinet, S. Crozet, and L. Najman, "A Quasi-linear Algorithm to Compute the Tree \
    of Shapes of nD Images", ISMM 2013.

    :param image: must be a 2d array
    :param padding: possible values are `'none'`, `'zero'`, and `'mean'` (default = `'mean'`)
    :param original_size: remove all nodes corresponding to interpolated/padded pixels (default = `True`)
    :param immersion: performs a plain map continuous immersion fo the original image (default = `True`)
    :param exterior_vertex: linear coordinate of the exterior point
    :return: a tree (Concept :class:`~higra.CptHierarchy`) and its node altitudes
    """

    assert len(image.shape) == 2, "This tree of shapes implementation only supports 2d images."
    immersion = bool(immersion)

    tree, altitudes = hg.cpp._component_tree_tree_of_shapes_image2d(image, padding, original_size, immersion, exterior_vertex)

    if original_size or ((not immersion) and padding == "none"):
        size = image.shape
    else:
        if padding == "none":
            size = (image.shape[0] * 2 - 1, image.shape[1] * 2 - 1)
        else:
            if immersion:
                size = ((image.shape[0] + 2) * 2 - 1, (image.shape[1] + 2) * 2 - 1)
            else:
                size = (image.shape[0] + 2, image.shape[1] + 2)

    g = hg.get_4_adjacency_graph(size)
    hg.CptHierarchy.link(tree, g)

    return tree, altitudes


def component_tree_multivariate_tree_of_shapes_image2d(image, padding='mean', original_size=True, immersion=True):
    """
    Multivariate tree of shapes for a 2d multi-band image. This tree is defined as a fusion of the marginal
    trees of shapes. The method is described in:

        E. Carlinet.
        `A Tree of shapes for multivariate images <https://pastel.archives-ouvertes.fr/tel-01280131/file/TH2015PESC1118.pdf>`_.
        PhD Thesis, Université Paris-Est, 2015.

    The input :attr:`image` must be a 3d array of shape :math:`(height, width, channel)`.

    Note that the constructed hierarchy doesn't have natural altitudes associated to its node: as a node is generally
    a fusion of several marginal nodes, we can't associate a single canonical value from the original image to this node.

    The parameters :attr:`padding`, :attr:`original_size`, and :attr:`immersion` are forwarded to the function
    :func:`~higra.component_tree_tree_of_shapes_image2d`: please look at this function documentation for more details.

    :Complexity:

    The worst case runtime complexity of this method is :math:`\mathcal{O}(N^2D^2)` with :math:`N` the number of pixels
    and :math:`D` the number of bands. If the image pixel values are quantized on :math:`K<<N` different values
    (eg. with a color image in :math:`[0..255]^D`), then the worst case runtime complexity can be tightened to
    :math:`\mathcal{O}(NKD^2)`.

    :See:

    This function relies on :func:`~higra.tree_fusion_depth_map` to compute the fusion of the marinal trees.

    :param image: input *color* 2d image
    :param padding: possible values are `'none'`, `'zero'`, and `'mean'` (default = `'mean'`)
    :param original_size: remove all nodes corresponding to interpolated/padded pixels (default = `True`)
    :param immersion: performs a plain map continuous immersion fo the original image (default = `True`)
    :return: a tree (Concept :class:`~higra.CptHierarchy`)
    """
    assert len(
        image.shape) == 3, "This multivariate tree of shapes implementation only supports multichannel 2d images."

    ndim = image.shape[2]

    trees = tuple(
        hg.component_tree_tree_of_shapes_image2d(
            image[:, :, k],
            padding,
            original_size=False,
            immersion=immersion)[0]
        for k in range(ndim))
    g = hg.CptHierarchy.get_leaf_graph(trees[0])
    shape = hg.CptGridGraph.get_shape(g)

    depth_map = hg.tree_fusion_depth_map(trees)

    tree, altitudes = hg.component_tree_tree_of_shapes_image2d(np.reshape(depth_map, shape), padding="none",
                                                               original_size=False, immersion=False)

    if original_size and (immersion or padding != "none"):
        deleted_vertices = np.ones((tree.num_leaves(),), dtype=np.bool)
        deleted = np.reshape(deleted_vertices, shape)

        if immersion:
            if padding != "none":
                deleted[2:-2:2, 2:-2:2] = False
            else:
                deleted[0::2, 0::2] = False
        else:
            if padding != "none":
                deleted[1:-1, 1::-1] = False

        all_deleted = hg.accumulate_sequential(tree, deleted_vertices, hg.Accumulators.min)
        shape = (image.shape[0], image.shape[1])
    else:
        all_deleted = np.zeros((tree.num_vertices(),), dtype=np.bool)

    holes = altitudes < altitudes[tree.parents()]

    all_deleted = np.logical_or(all_deleted, holes)

    tree, _ = hg.simplify_tree(tree, all_deleted, process_leaves=True)

    g = hg.get_4_adjacency_graph(shape)
    hg.CptHierarchy.link(tree, g)

    return tree
