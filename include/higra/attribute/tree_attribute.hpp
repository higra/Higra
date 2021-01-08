/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include "../graph.hpp"
#include "../accumulator/tree_accumulator.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xnoalias.hpp"

namespace hg {

    /**
     * The area  of a node n of the tree t is equal to the sum of the area of the leaves in the subtree rooted in n.
     *
     * area(n) = sum_{l in leaves(t), l is a  descendant of n} area(n)
     *
     * @tparam tree_t tree type
     * @tparam T xexpression derived type of xleaf_area
     * @param tree input tree
     * @param xleaf_area area of the leaves of the input tree
     * @return an array with the area of each node of the tree
     */
    template<typename tree_t, typename T>
    auto attribute_area(const tree_t &tree, const xt::xexpression<T> &xleaf_area) {
        auto &leaf_area = xleaf_area.derived_cast();
        hg_assert_leaf_weights(tree, leaf_area);

        return accumulate_sequential(tree, leaf_area, accumulator_sum());
    }

    /**
     * The area  of a node n of the tree t is equal to the number of leaves in the subtree rooted in n.
     *
     * area(n) = |{l in leaves(t), l is a  descendant of n}|
     *
     * @tparam tree_t tree type
     * @param tree input tree
     * @return an array with the area of each node of the tree
     */
    template<typename tree_t>
    auto attribute_area(const tree_t &tree) {
        return attribute_area(tree, xt::ones<index_t>({num_leaves(tree)}));
    }

    /**
     * The volume of a node n of the tree t is defined recursively as:
     *    volume(n) = abs(altitude(n) - altitude(parent(n)) * area(n) + sum_{c in children(n, t)} volume(c)
     *
     * @tparam tree_t tree type
     * @tparam T1 xexpression derived type of xnode_altitude
     * @tparam T2 xexpression derived type of xnode_area
     * @param tree input tree
     * @param xnode_altitude altitude of the nodes of the input tree
     * @param xnode_area area of the nodes of the input tree
     * @return an array with the volume of each node of the tree
     */
    template<typename tree_t, typename T1, typename T2>
    auto attribute_volume(const tree_t &tree, const xt::xexpression<T1> &xnode_altitude,
                          const xt::xexpression<T2> &xnode_area) {
        auto &node_area = xnode_area.derived_cast();
        auto &node_altitude = xnode_altitude.derived_cast();
        hg_assert_node_weights(tree, node_area);
        hg_assert_1d_array(node_area);
        hg_assert_node_weights(tree, node_altitude);
        hg_assert_1d_array(node_altitude);
        tree.compute_children();
        auto &parent = tree.parents();
        array_1d<double> volume = xt::empty<double>({tree.num_vertices()});
        xt::view(volume, xt::range(0, num_leaves(tree))) = 0;
        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            volume(i) = std::fabs(node_altitude(i) - node_altitude(parent(i))) * node_area(i);
            for (auto c: children_iterator(i, tree)) {
                volume(i) += volume(c);
            }
        }
        return volume;
    }

    /**
     * The depth of a node n of the tree t is equal to the number of ancestors of n.
     *
     * @tparam tree_t tree type
     * @param tree input tree
     * @return an array with the depth of each node of the tree
     */
    template<typename tree_t>
    auto attribute_depth(const tree_t &tree) {
        array_1d<index_t> depth = xt::empty<index_t>({tree.num_vertices()});
        depth(tree.root()) = 0;
        for (auto i: root_to_leaves_iterator(tree, leaves_it::include, root_it::exclude)) {
            depth(i) = depth(parent(i, tree)) + 1;
        }
        return depth;
    };

    /**
     * In a tree :math:`t`, given that the altitudes of the nodes vary monotically from the leaves to the root,
     * the height of a node :math:`n` of :math:`t` is equal to the difference between the altitude of the parent
     * of :math:`n` and the altitude of the deepest non-leaf node in the subtree of :math:`t` rooted in :math:`n`.
     *
     * If :attr:`increasing_altitude` is true, this means that altitudes are increasing from the leaves to the root
     * (ie. for any node :math:`n`, :math:`altitudes(n) \leq altitudes(parent(n))`.
     * Else, if :attr:`increasing_altitude` is false, this means that altitudes are decreasing from the leaves to the root
     * (ie. for any node :math:`n`, :math:`altitude(n) \geq altitude(parent(n))`.
     *
     * @tparam tree_t tree type
     * @tparam T xexpression derived type of xnode_altitude
     * @param tree input tree
     * @param xnode_altitude altitude of the nodes of the input tree
     * @param increasing_altitudes must be true if altitude is increasing, false if it is decreasing
     * @return an array with the height of each node of the tree
     */
    template<typename tree_t, typename T>
    auto
    attribute_height(const tree_t &tree,
                     const xt::xexpression<T> &xaltitudes,
                     bool increasing_altitudes) {
        auto &altitudes = xaltitudes.derived_cast();
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);
        using value_type = typename T::value_type;

        tree.compute_children();
        if (increasing_altitudes) {
            auto min_depth = xt::empty_like(altitudes);
            xt::noalias(xt::view(min_depth, xt::range(0, num_leaves(tree)))) =
                    xt::view(xt::index_view(altitudes, tree.parents()), xt::range(0, num_leaves(tree)));
            for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude)) {
                min_depth(n) = (std::numeric_limits<value_type>::max)();
                bool flag = true;
                for (auto c: children_iterator(n, tree)) {
                    if (!is_leaf(c, tree)) {
                        flag = false;
                        if (min_depth(c) < min_depth(n)) {
                            min_depth(n) = min_depth(c);
                        }
                    }
                }
                if (flag) {
                    min_depth(n) = altitudes(n);
                }
            }
            return xt::eval(xt::index_view(altitudes, tree.parents()) - min_depth);
        } else {
            auto max_depth = xt::empty_like(altitudes);
            xt::noalias(xt::view(max_depth, xt::range(0, num_leaves(tree)))) =
                    xt::view(xt::index_view(altitudes, tree.parents()), xt::range(0, num_leaves(tree)));
            for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude)) {
                max_depth(n) = std::numeric_limits<value_type>::lowest();
                bool flag = true;
                for (auto c: children_iterator(n, tree)) {
                    if (!is_leaf(c, tree)) {
                        flag = false;
                        if (max_depth(c) > max_depth(n)) {
                            max_depth(n) = max_depth(c);
                        }
                    }
                }
                if (flag) {
                    max_depth(n) = altitudes(n);
                }
            }
            return xt::eval(max_depth - xt::index_view(altitudes, tree.parents()));
        }
    };

    /**
     * Identify nodes in a hierarchy that represent extrema.
     *
     * An extremum of the hierarchy :math:`T` with altitudes :math:`alt` is a node :math:`n` of :math:`T` such that
     * the altitude of any non leaf node included in :math:`n` is equal to the altitude of :math:`n` and the altitude
     * of the parent of :math:`n` is different from the altitude of :math:`n`.
     *
     * The result is a boolean array such that :math:`result(n)=true` if the node :math:`n` is an extremum.
     *
     * @tparam tree_t
     * @tparam T
     * @param tree Input tree
     * @param xaltitudes Tree node altitudes
     * @return
     */
    template<typename tree_t, typename T>
    auto
    attribute_extrema(const tree_t &tree,
                      const xt::xexpression<T> &xaltitudes) {
        auto &altitudes = xaltitudes.derived_cast();
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);

        tree.compute_children();
        array_1d<bool> extrema = xt::zeros<bool>({num_vertices(tree)});
        for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            bool flag = true;
            for (auto c: children_iterator(n, tree)) {
                bool c_non_canonical = altitudes(c) == altitudes(n);
                if (!(is_leaf(c, tree) || (c_non_canonical && extrema(c)))) {
                    flag = false;
                }
                extrema(c) = extrema(c) && !c_non_canonical;
            }
            extrema(n) = flag;
        }
        return extrema;
    }

    /**
     * The extinction value of a node :math:`n` of the input tree :math:`t` with increasing altitudes :math:`alt`
     * for the increasing attribute :math:`att` is the equal to the threshold :math:`k` such that the node :math:`n`
     * is still in an minima of :math:`t` when all nodes having an attribute value smaller than :math:`k` are removed.
     *
     * Formally, let :math:`\{M_i\}` be the set of minima of :math:`(t, altitudes)`. Let :math:`prec` be a total
     * ordering of :math:`\{M_i\}` such that :math:`M_i \prec M_j \Rightarrow alt(M_i) \leq alt(M_j)`.
     * Let :math:`r(M_i)` be the smallest node of :math:`t` containing :math:`M_i` and another minima :math`M_j` such
     * that :math:`M_j \prec M_i`. The extinction value of :math`M_i` is then defined as :math:`alt(r(M_i)) - alt(M_i)`.
     *
     * Extinction values of minima are then extended to other nodes in the tree with the following rules:
     *
     *   - the extinction value of a non-leaf node :math:`n` which is not a minimum is defined as the largest
     *     extinction values among all the minima contained in :math:`n`
     *     (and 0 if :math:`n` does not contain any minima); and
     *   - the extinction value of a leaf node :math:`n` belonging to a minima :math:`M_i` is equal to the extinction
     *     value of :math:`M_i`. I :math:`n` does not belong to any minima its extinction value is 0.
     *
     * If :attr:`increasing_altitude` is true, this means that altitudes are increasing from the leaves to the root
     * (ie. for any node :math:`n`, :math:`altitudes(n) \leq altitudes(parent(n))`.
     * Else, if :attr:`increasing_altitude` is false, this means that altitudes are decreasing from the leaves to the root
     * (ie. for any node :math:`n`, :math:`altitude(n) \geq altitude(parent(n))`: you should then replace minima by maxima
     * in the description above.
     *
     * @tparam tree_t tree type
     * @tparam T xexpression derived type of xaltitude
     * @param tree input tree
     * @param xaltitudes altitude of the nodes of the input tree
     * @param xattribute attribute used for filtering
     * @param increasing_altitudes must be true if altitude is increasing, false if it is decreasing
     * @return an array with the dynamics of each node of the tree
     */
    template<typename tree_t,
            typename T1,
            typename T2>
    auto attribute_extinction_value(const tree_t &tree,
                                    const xt::xexpression<T1> &xaltitudes,
                                    const xt::xexpression<T2> &xattribute,
                                    bool increasing_altitudes) {
        auto &altitudes = xaltitudes.derived_cast();
        auto &attribute = xattribute.derived_cast();
        using value_type = typename T1::value_type;
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);
        hg_assert_node_weights(tree, attribute);
        hg_assert_1d_array(attribute);

        tree.compute_children();

        // identify path to the deepest extrema
        array_1d<index_t> ref_son({num_vertices(tree)}, invalid_index);
        if (increasing_altitudes) {
            auto min_depth = xt::empty_like(altitudes);
            for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude)) {
                min_depth(n) = (std::numeric_limits<value_type>::max)();
                bool flag = true;
                for (auto c: children_iterator(n, tree)) {
                    if (!is_leaf(c, tree)) {
                        flag = false;
                        if (min_depth(c) < min_depth(n)) {
                            min_depth(n) = min_depth(c);
                            ref_son(n) = c;
                        }
                    }
                }
                if (flag) {
                    min_depth(n) = altitudes(n);
                }
            }
        } else {
            auto max_depth = xt::empty_like(altitudes);
            for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude)) {
                max_depth(n) = std::numeric_limits<value_type>::lowest();
                bool flag = true;
                for (auto c: children_iterator(n, tree)) {
                    if (!is_leaf(c, tree)) {
                        flag = false;
                        if (max_depth(c) > max_depth(n)) {
                            max_depth(n) = max_depth(c);
                            ref_son(n) = c;
                        }
                    }
                }
                if (flag) {
                    max_depth(n) = altitudes(n);
                }
            }
        }

        // extinction of non leaf nodes
        array_1d<typename T2::value_type> extinction = array_1d<typename T2::value_type>::from_shape(
                {num_vertices(tree)});
        extinction(root(tree)) = attribute(root(tree));
        for (auto n: root_to_leaves_iterator(tree, leaves_it::exclude, root_it::exclude)) {
            auto pn = parent(n, tree);
            if (n == ref_son(pn)) {
                extinction(n) = extinction(pn);
            } else {
                extinction(n) = attribute(n);
            }
        }

        auto extrema = attribute_extrema(tree, altitudes);
        auto indices = xt::eval(xt::arange<index_t>(num_vertices(tree)));
        if (!extrema(root(tree))) {
            indices(root(tree)) = -1;
        }
        auto extrema_leaves = propagate_sequential(tree, indices, !extrema);
        for (auto n: leaves_iterator(tree)) {
            if (extrema_leaves(n) != -1) {
                extinction(n) = extinction(extrema_leaves(n));
            } else {
                extinction(n) = 0;
            }
        }

        return extinction;
    };

    /**
     * Given a node :math:`n` of the tree :math:`t`, the dynamics of :math:`n` is the difference between
     * the altitude of the deepest minima of the subtree rooted in :math:`n` and the altitude
     * of the closest ancestor of :math:`n` that has a deeper minima in its subtree. If no such
     * ancestor exists then, the dynamics of :math:`n` is equal to the difference between the
     * altitude of the highest node of the tree (the root) and the depth of the deepest minima.
     *
     * The dynamics is the *extinction values* for the attribute *height*.
     *
     * If :attr:`increasing_altitude` is true, this means that altitudes are increasing from the leaves to the root
     * (ie. for any node :math:`n`, :math:`altitudes(n) \leq altitudes(parent(n))`.
     * Else, if :attr:`increasing_altitude` is false, this means that altitudes are decreasing from the leaves to the root
     * (ie. for any node :math:`n`, :math:`altitude(n) \geq altitude(parent(n))`.
     *
     * @tparam tree_t tree type
     * @tparam T xexpression derived type of xaltitudes
     * @param tree input tree
     * @param xaltitudes altitude of the nodes of the input tree
     * @param increasing_altitudes must be true if altitude is increasing, false if it is decreasing
     * @return an array with the dynamics of each node of the tree
     */
    template<typename tree_t, typename T>
    auto attribute_dynamics(const tree_t &tree,
                            const xt::xexpression<T> &xaltitudes,
                            bool increasing_altitudes) {
        auto &altitudes = xaltitudes.derived_cast();
        using value_type = typename T::value_type;
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);

        auto height = attribute_height(tree, altitudes, increasing_altitudes);

        return attribute_extinction_value(tree, altitudes, height, increasing_altitudes);
    };

    /**
     * For each node `n` which is the `k`-th child of its parent node `p` among `N` children,
     * the attribute sibling of `n` is the index of the `(k + skip) % N`-th child of `p`.
     *
     * The sibling of the root node is itself.
     *
     * The sibling attribute enables to easily emulates a (doubly) linked list among brothers.
     *
     * In a binary tree, the sibling attribute of a node is effectively its only brother (with `skip` equals to 1).
     *
     * @tparam tree_t
     * @param tree Input tree
     * @param skip Number of skipped element in the children list (including yourself)
     * @return an array with the sibling index of each node of the tree
     */
    template<typename tree_t>
    auto attribute_sibling(const tree_t &tree, index_t skip = 1) {

        tree.compute_children();
        array_1d<index_t> attribute = xt::empty<index_t>({num_vertices(tree)});
        for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude, root_it::include)) {
            index_t nchs = num_children(n, tree);
            for (index_t i = 0; i < nchs; i++) {

                index_t j = (i + skip) % nchs;
                if (j < 0) { // stupid c modulo
                    j += nchs;
                }

                attribute(child(i, n, tree)) = child(j, n, tree);
            }
        }
        attribute(root(tree)) = root(tree);
        return attribute;
    }

    /**
     * Computes the contour length (perimeter) of each node of the input component tree.
     *
     * Warning: does not work for tree of shapes left in original space (the problem is that
     * two children of a node may become adjacent when the interpolated pixels are removed).
     *
     * @tparam tree_t
     * @tparam graph_t
     * @tparam T1
     * @tparam T2
     * @param tree input tree
     * @param base_graph graph on the leaves of tree
     * @param xvertex_perimeter perimeter of each vertex of the base graph
     * @param xedge_length length of each edge of the base graph (length of the frontier between the two adjacent vertices)
     * @return
     */
    template<typename tree_t, typename graph_t, typename T1, typename T2>
    auto attribute_contour_length_component_tree(
            const tree_t &tree,
            const graph_t &base_graph,
            const xt::xexpression<T1> &xvertex_perimeter,
            const xt::xexpression<T2> &xedge_length) {
        hg_assert_component_tree(tree);
        auto &vertex_perimeter = xvertex_perimeter.derived_cast();
        hg_assert_1d_array(vertex_perimeter);
        hg_assert_leaf_weights(tree, vertex_perimeter);
        auto &edge_length = xedge_length.derived_cast();
        hg_assert_1d_array(edge_length);
        hg_assert_edge_weights(base_graph, edge_length);

        array_1d<double> res = array_1d<double>::from_shape({num_vertices(tree)});
        xt::noalias(xt::view(res, xt::range(0, num_leaves(tree)))) = vertex_perimeter;
        array_1d<bool> visited({num_leaves(tree)}, false);
        tree.compute_children();

        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            res(i) = 0;
            for (auto c: children_iterator(i, tree)) {
                res(i) += res(c);
                if (is_leaf(c, tree)) {
                    for (auto e: out_edge_iterator(c, base_graph)) {
                        if (visited(target(e, base_graph))) {
                            res(i) -= 2.0 * edge_length(e);
                        }
                    }
                    visited(c) = true;
                }
            }
        }
        return res;
    }


    /**
     * Given a node :math:`n` whose parent is :math:`p`, the attribute value of :math:`n` is the rank of :math:`n`
     * in the list of children of :math:`p`. In other :math:`attribute(n)=i` means that :math:`n` is the :math:`i`-th
     * child of :math:`p`.
     *
     * The root of the tree, who has no parent, take the value -1.
     *
     * @tparam tree_t
     * @param tree
     * @return
     */
    template<typename tree_t>
    auto attribute_child_number(const tree_t &tree) {
        tree.compute_children();
        array_1d<index_t> res = xt::empty<index_t>({num_vertices(tree)});
        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            for (index_t c = 0; c < (index_t) num_children(i, tree); c++) {
                auto cc = child(c, i, tree);
                res(cc) = c;
            }
        }
        res(root(tree)) = invalid_index;
        return res;
    }


    /**
     * Given two trees :math:`t_1` and :math:`t_2` defined over the same domain, ie sharing the same set of leaves.
     * For each node :math:`n` of :math:`t1`, computes the index of the smallest node of :math:`t2` containing :math:`n`.
     *
     * @tparam tree_t
     * @param t1
     * @param t2
     * @return
     */
    template<typename tree_t>
    auto attribute_smallest_enclosing_shape(const tree_t &t1, const tree_t &t2) {
        array_1d<index_t> attr({num_vertices(t1)}, invalid_index);
        xt::noalias(xt::view(attr, xt::range(0, num_leaves(t1)))) = xt::arange(num_leaves(t1));

        for (auto i: leaves_to_root_iterator(t1)) {
            auto p = parent(i, t1);
            if (attr(p) == -1) {
                attr(p) = attr(i);
            } else {
                attr(p) = lowest_common_ancestor(attr(p), attr(i), t2);
            }
        }

        return attr;
    }

    /**
     * Given a tree :math:`T` with node weights :math:`w`: the children pair sum product for a node :math:`n` sums for
     * every pairs :math:`(c_i, c_j)` of children of :math:`n`, the product of the node weights of :math:`c_i` and
     * :math:`c_j`. Formally:
     *
     * .. math::
     *
     *      res(n) = \sum_{i=0}^{i<numc(n)} \sum_{j=0}^{j<i} w(child(i, n)) * w(child(j, n))
     *
     * where :math:`numc(n)` is the number of children of :math:`n` and :math:`child(i, n)` is the :math:`i`-th child
     * of the node :math:`n`.
     *
     * The result is thus an array of the same shape of :attr:`node_weights`
     *
     * @tparam tree_t
     * @tparam T
     * @tparam value_type
     * @param tree
     * @param xnode_weights
     * @return
     */
    template<typename tree_t, typename T, typename value_type=typename T::value_type>
    auto attribute_children_pair_sum_product(const tree_t &tree, const xt::xexpression<T> &xnode_weights) {
        auto &node_weights = xnode_weights.derived_cast();
        hg_assert_node_weights(tree, node_weights);

        tree.compute_children();
        array_nd<value_type> res = xt::zeros<value_type>(node_weights.shape());
        const auto num_v = num_vertices(tree);
        const auto num_l = num_leaves(tree);

        if (res.dimension() == 1) {
            for (index_t i = num_l; i < (index_t) num_v; i++) {
                for (index_t c1 = 0; c1 < (index_t) num_children(i, tree); c1++) {
                    for (index_t c2 = c1 + 1; c2 < (index_t) num_children(i, tree); c2++) {
                        res(i) += node_weights(child(c1, i, tree)) * node_weights(child(c2, i, tree));
                    }
                }
            }
        } else {
            const size_t num_d = node_weights.size() / num_v;
            auto vnode_weights = xt::reshape_view(node_weights, {num_v, num_d});
            auto vres = xt::reshape_view(res, {num_v, num_d});

            for (index_t i = num_l; i < (index_t) num_v; i++) {
                for (index_t c1 = 0; c1 < (index_t) num_children(i, tree); c1++) {
                    for (index_t c2 = c1 + 1; c2 < (index_t) num_children(i, tree); c2++) {
                        for (index_t k = 0; k < (index_t) num_d; k++) {
                            vres(i, k) += vnode_weights(child(c1, i, tree), k) * vnode_weights(child(c2, i, tree), k);
                        }
                    }
                }
            }
        }
        return res;
    }

}
