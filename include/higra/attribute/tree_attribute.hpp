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
#include "../structure/array.hpp"
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

        auto &parent = tree.parents();
        array_1d<double> volume = xt::empty<double>({tree.num_vertices()});
        xt::view(volume, xt::range(0, num_leaves(tree))) = 0;
        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            volume(i) = std::fabs(node_altitude(i) - node_altitude(parent(i))) * node_area(i);
            for (auto c: tree.children(i)) {
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
     * In a tree t, given that the altitudes of the nodes vary monotically from the leaves to the root,
     * the height of a node n of t is equal to the difference between the altitude of n and
     * the altitude of the deepest leave in the subtree of t rooted in n.
     *
     *
     * If increasing_altitude is true, this means that altitudes are increasing from the leaves to the root
     * (ie. for any node n, altitude(n) <= altitude(parent(n)).
     * Else, if increasing_altitude is false, this means that altitudes are decreasing from the leaves to the root
     * (ie. for any node n, altitude(n) >= altitude(parent(n)).
     *
     * PRE-CONDITION: altitudes of the nodes vary monotically from the leaves to the root
     *
     * @tparam tree_t tree type
     * @tparam T xexpression derived type of xnode_altitude
     * @param tree input tree
     * @param xnode_altitude altitude of the nodes of the input tree
     * @param increasing_altitude true if altitude is increasing, false if it is decreasing
     * @return an array with the height of each node of the tree
     */
    template<typename tree_t, typename T>
    auto
    attribute_height(const tree_t &tree, const xt::xexpression<T> &xnode_altitude, bool increasing_altitude = true) {
        auto &node_altitude = xnode_altitude.derived_cast();
        hg_assert_node_weights(tree, node_altitude);
        hg_assert_1d_array(node_altitude);

        if (increasing_altitude) {
            auto extrema = accumulate_sequential(tree, xt::view(node_altitude, xt::range(0, num_leaves(tree))),
                                                 accumulator_min());
            return xt::eval(node_altitude - extrema);
        } else {
            auto extrema = accumulate_sequential(tree, xt::view(node_altitude, xt::range(0, num_leaves(tree))),
                                                 accumulator_max());
            return xt::eval(extrema - node_altitude);
        }
    };

    /**
     * Given a node n of the tree t, the dynamics of n is the difference between
     * the altitude of the deepest minima of the subtree rooted in n and the altitude
     * of the closest ancestor of n that has a deeper minima in its subtree. If no such
     * ancestor exists then, the dynamics of n is equal to the difference between the
     * altitude of the highest node of the tree (the root) and the depth of the deepest minima.
     *
     * @tparam tree_t tree type
     * @tparam T xexpression derived type of xaltitude
     * @param tree input tree
     * @param xaltitude altitude of the nodes of the input tree
     * @return an array with the dynamics of each node of the tree
     */
    template<typename tree_t, typename T>
    auto attribute_dynamics(const tree_t &tree, const xt::xexpression<T> &xaltitude) {
        auto &altitude = xaltitude.derived_cast();
        using value_type = typename T::value_type;
        hg_assert_node_weights(tree, altitude);
        hg_assert_1d_array(altitude);

        array_1d<index_t> min_son({num_vertices(tree)}, invalid_index);
        auto min_depth = xt::empty_like(altitude);
        for (auto n: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            min_depth(n) = std::numeric_limits<value_type>::max();
            bool flag = true;
            for (auto c: children_iterator(n, tree)) {
                if (!is_leaf(c, tree)) {
                    flag = false;
                    if (min_depth(c) < min_depth(n)) {
                        min_depth(n) = min_depth(c);
                        min_son(n) = c;
                    }
                }
            }
            if (flag) {
                min_depth(n) = altitude(n);
            }
        }

        auto dynamics = xt::empty_like(altitude);
        dynamics(root(tree)) = altitude(root(tree)) - min_depth(root(tree));
        xt::view(dynamics, xt::range(0, num_leaves(tree))) = 0;
        for (auto n: root_to_leaves_iterator(tree, leaves_it::exclude, root_it::exclude)) {
            auto pn = parent(n, tree);
            if (n == min_son(pn)) {
                dynamics(n) = dynamics(pn);
            } else {
                dynamics(n) = altitude(pn) - min_depth(n);
            }
        }

        return dynamics;
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
     * Computes the perimeter length of each node of the input component tree.
     *
     * @tparam tree_t
     * @tparam graph_t
     * @tparam T1
     * @tparam T2
     * @param tree input tree
     * @param base_graph graph on the leaves of tree
     * @param xvertex_perimeter perimeter length of each vertex of the base graph
     * @param xedge_length length of each edge of the base graph (length of the frontier between the two adjacent vertices)
     * @return
     */
    template<typename tree_t, typename graph_t, typename T1, typename T2>
    auto attribute_perimeter_length_component_tree(
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
}
