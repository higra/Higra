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

#include "xtensor/xgenerator.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xview.hpp"
#include "../structure/tree_graph.hpp"
#include "../accumulator/tree_accumulator.hpp"
#include "../hierarchy/common.hpp"
#include "higra/sorting.hpp"
#include <queue>

namespace hg {

    /**
     * Each leaf of the tree takes the weight of its closest non deleted ancestor.
     *
     * @tparam tree_t
     * @tparam T1
     * @tparam T2
     * @param tree
     * @param altitudes
     * @param deleted_nodes
     * @return
     */
    template<typename tree_t,
            typename T1,
            typename T2>
    auto reconstruct_leaf_data(const tree_t &tree,
                               const xt::xexpression<T1> &altitudes,
                               const xt::xexpression<T2> &deleted_nodes) {
        HG_TRACE();
        auto reconstruction = propagate_sequential(tree,
                                                   altitudes,
                                                   deleted_nodes);
        return xt::eval(xt::strided_view(reconstruction, {xt::range(0, num_leaves(tree)), xt::ellipsis()}));
    };

    /**
     * Labelize tree leaves according to an horizontal cut in the tree.
     *
     * Two leaves are in the same region (ie. have the same label) if
     * the altitude of their lowest common ancestor is smaller or equal
     * than the specified threshold.
     *
     * The label of a leave l is equal to the index of the smallest node containing l
     * and whose altitude is strictly greater than the specified threshold.
     *
     * @tparam tree_t
     * @tparam T
     * @tparam value_t
     * @param tree
     * @param xaltitudes
     * @param threshold
     * @return
     */
    template<typename tree_t,
            typename T,
            typename value_t>
    auto labelisation_horizontal_cut_from_threshold(const tree_t &tree,
                                                    const xt::xexpression<T> &xaltitudes,
                                                    const value_t threshold) {
        HG_TRACE();
        auto &altitudes = xaltitudes.derived_cast();
        return reconstruct_leaf_data(tree,
                                     xt::arange<index_t>(num_vertices(tree)),
                                     xt::index_view(altitudes, tree.parents())
                                     <= static_cast<typename T::value_type>(threshold));
    };

    /**
     * Labelize the tree leaves into supervertices.
     *
     * Two leaves are in the same supervertex if they have a common ancestor of altitude 0.
     *
     * This functions guaranties that the labels are in the range [0, num_supervertices-1].
     *
     * @tparam tree_t
     * @tparam T
     * @param tree
     * @param xaltitudes
     * @return
     */
    template<typename tree_t,
            typename T>
    auto labelisation_hierarchy_supervertices(const tree_t &tree,
                                              const xt::xexpression<T> &xaltitudes) {
        HG_TRACE();
        auto &altitudes = xaltitudes.derived_cast();
        hg_assert_node_weights(tree, altitudes);

        auto labels = labelisation_horizontal_cut_from_threshold(tree, altitudes, 0);
        // remap labels to 0...num_labels - 1
        array_1d<index_t> map({num_vertices(tree)}, invalid_index);
        index_t current_label = 0;
        for (auto n: leaves_iterator(tree)) {
            auto lbl = labels(n);
            if (map(lbl) == invalid_index) {
                map(lbl) = current_label;
                current_label++;
            }
            labels(n) = map(lbl);
        }
        return labels;
    };

    /**
     * A simple structure to hold the result of supervertices_hierarchy algorithm.
     *
     * @tparam node_map_t
     * @tparam tree_t
     * @tparam node_map_t
     *
     */
    template<typename supervertex_labels_t, typename tree_t, typename node_map_t>
    struct supervertex_hierarchy {
        supervertex_labels_t supervertex_labelisation;
        tree_t tree;
        node_map_t node_map;
    };

    /**
     * Extract the supervertices associated to the given tree and create the equivalent tree on this supervertices.
     *
     * Two leaves are in the same supervertex if they have a common ancestor of altitude 0.
     *
     * The equivalent tree is obtained by removing all nodes of the given tree which does not contain
     * any of the supervertices. Its leaves are thus the supervertices.
     *
     * Also returns an array that maps any node index i of the new tree, to the index of this node in the original tree.
     *
     * @tparam tree_t
     * @tparam T
     * @param tree
     * @param xaltitudes
     * @return
     */
    template<typename tree_t,
            typename T>
    auto supervertices_hierarchy(const tree_t &tree,
                                 const xt::xexpression<T> &xaltitudes) {
        HG_TRACE();
        auto &altitudes = xaltitudes.derived_cast();
        hg_assert_node_weights(tree, altitudes);

        // new index of each node (deleted node contains the index of the node they are merged into)
        array_1d<index_t> new_order({num_vertices(tree)}, invalid_index);

        // *******************************
        // Identification of supervertices
        // (with bottom-up propagation)
        // *******************************

        // number of identified supervertices
        index_t current_label = 0;
        // indexes of the supervertex nodes (largest node in a supervertex)
        std::vector<index_t> super_vertex_nodes;
        // number of nodes smaller than a supervertex
        index_t removed = 0;

        tree.compute_children();
        stackv<index_t> stack;
        for (auto n: leaves_iterator(tree)) {
            auto e = n;
            index_t lbl = invalid_index;
            // move up until we find the top node of the supervertex of an already labeled node is found
            do {
                stack.push(e);
                e = parent(e, tree);
                if (new_order(e) != invalid_index) {
                    lbl = new_order(e);
                }
            } while (altitudes(e) == 0 && lbl == invalid_index);

            removed += stack.size();

            // we have found a new supervertex node
            if (lbl == invalid_index) {
                super_vertex_nodes.push_back(stack.top());
                lbl = current_label++;
                removed--;
            }

            // labelize visited nodes
            while (!stack.empty()) {
                new_order(stack.top()) = lbl;
                stack.pop();
            }
        }

        // *******************************
        // Topological sort of remaining vertices
        // (with top-down traversal)
        // *******************************
        auto num_nodes_new_tree = num_vertices(tree) - removed;
        array_1d<index_t> parents = xt::empty<index_t>({num_nodes_new_tree});
        array_1d<index_t> node_map = xt::empty<index_t>({num_nodes_new_tree});
        index_t node_number = num_nodes_new_tree - 1;
        std::queue<index_t> queue;
        queue.push(root(tree));
        while (!queue.empty()) {
            auto e = queue.front();
            queue.pop();
            new_order(e) = node_number;
            parents(node_number) = new_order(parent(e, tree));
            node_map(node_number) = e;
            node_number--;
            for (auto c: children_iterator(e, tree)) {
                if (new_order(c) == invalid_index) {
                    queue.push(c);
                }
            }
        }

        index_t i = 0;
        for (auto n: super_vertex_nodes) {
            parents(i) = new_order(parent(n, tree));
            node_map(i) = n;
            i++;
        }

        auto supervertex_labels = xt::eval(xt::view(new_order, xt::range(0, num_leaves(tree))));

        return supervertex_hierarchy<array_1d<index_t>, hg::tree, array_1d<index_t>>{
                std::move(supervertex_labels), hg::tree(parents, tree.category()), std::move(node_map)
        };
    };

    /**
     * Test if 2 trees are isomorph assuming that they share the same leaves.
     *
     * By this definition t1 is isomorph to t2 if there exist a bijection f from vertices(t1) to vertices(t2) such that:
     *   1) for any leaf node n of t1, f(n) = n and n
     *   2) for any node n of t1, f(t1.parent(n)) = t2.parent(f(n))
     *
     * Note that the root r node of a tree t is defined by t.parent(r) = r, thus 2) becomes
     *   for the root node r1 of t1, f(r1) = t2.parent(f(r1)), i.e. f(r1) is the root node of t2
     *
     * @tparam tree1_t
     * @tparam tree2_t
     * @param t1
     * @param t2
     * @return
     */
    template<typename tree1_t, typename tree2_t>
    bool test_tree_isomorphism(const tree1_t &t1, const tree2_t &t2) {
        HG_TRACE();
        if (num_vertices(t1) != num_vertices(t2) || num_leaves(t1) != num_leaves(t2))
            return false;

        size_t num_v = num_vertices(t1);
        index_t num_l = (index_t) num_leaves(t1);

        array_1d<index_t> f({num_v}, invalid_index);

        for (auto i: leaves_to_root_iterator(t1)) {
            if (i < num_l)
                f[i] = i; // 1) for any leaf node n of t1, f(n) = n and n

            auto t1pi = parent(i, t1); // t1.parent(n)
            index_t t2pfi = parent(f(i), t2); //t2.parent(f(n))

            if (f(t1pi) == invalid_index) { // f(t1.parent(n)) not defined
                f(t1pi) = t2pfi; // f(t1.parent(n)) = t2.parent(f(n))
            } else if (f(t1pi) != t2pfi) { // f(t1.parent(n)) was already set by a brother of n and is not compatible
                return false;
            }// else ok
        }
        return true;
    }

    /**
     * Given two binary markers o (object) and b (background) (given by their indicator functions)
     * on the leaves of a tree t, the corresponding binary labelization of the leaves of t is defined as 
     * the union of all the nodes intersecting o but not b.
     * 
     * final_object = union {R in T | R cap o neq emptyset and R cap b = emptyset}
     * 
     * @tparam tree_t tree type
     * @tparam T1 xtensor type, value_type must be castable to bool
     * @tparam T2 xtensor type, value_type must be castable to bool
     * @param tree input tree
     * @param xobject_marker indicator function of the object marker
     * @param xbackground_marker indicator function of the background marker
     * @return indicator function of the final_object
     */
    template<typename tree_t, typename T1, typename T2>
    auto binary_labelisation_from_markers(
            const tree_t &tree,
            const xt::xexpression<T1> &xobject_marker,
            const xt::xexpression<T2> &xbackground_marker) {
        HG_TRACE();
        auto &object_marker = xobject_marker.derived_cast();
        auto &background_marker = xbackground_marker.derived_cast();
        hg_assert_leaf_weights(tree, object_marker);
        hg_assert_leaf_weights(tree, background_marker);

        array_1d<char> attr({num_vertices(tree)}, 0);

        tree.compute_children();
        for (auto i: leaves_iterator(tree)) {
            if (static_cast<bool>(background_marker(i))) {
                attr(i) = 1;
            } else if (static_cast<bool>(object_marker(i))) {
                attr(i) = 2;
            }
        }

        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            for (auto c: children_iterator(i, tree)) {
                attr(i) |= attr(c);
            }
        }

        if (attr(root(tree)) == 0) { // both markers are empty
            attr(root(tree)) = 1;
        }

        for (auto i: root_to_leaves_iterator(tree, leaves_it::include, root_it::exclude)) {
            if (attr(i) == 0) {
                if (attr(parent(i, tree)) == 2) {
                    attr(i) = 2;
                } else {
                    attr(i) = 1;
                }
            }
        }

        return xt::eval(xt::view(attr, xt::range(0, num_leaves(tree))) - 1);
    }

    /**
     * Sort the nodes of a tree according to their altitudes.
     * The altitudes must be increasing, i.e. for any nodes i, j such that j is an ancestor of j, then
     * altitudes[i] <= altitudes[j].
     *
     * The result is a new tree and a node map, isomorph to the input tree such that for any nodes i and j,
     * i<j => altitudes[node_map[i]] <= altitudes[node_map[j]]
     *
     * The latter condition is stronger than the original condition on the altitudes as j is an ancestor of i implies
     * i<j while the converse is not true.
     *
     * Note that the altitudes of the new tree can be obtained with:
     *
     *  auto res = sort_hierarchy_with_altitudes(tree, altitudes);
     *  auto & new_tree = res.tree;
     *  auto new_altitudes = xt::index_view(altitudes, res.node_map);
     *
     * @tparam tree_t
     * @tparam T
     * @param tree
     * @param xaltitudes
     * @return
     */
    template<typename tree_t,
            typename T>
    auto sort_hierarchy_with_altitudes(const tree_t &tree,
                                       const xt::xexpression<T> &xaltitudes) {
        HG_TRACE();
        auto &altitudes = xaltitudes.derived_cast();
        hg_assert_node_weights(tree, altitudes);
        hg_assert_1d_array(altitudes);

        array_1d<index_t> sorted = stable_arg_sort(altitudes);

        array_1d<index_t> reverse_sorted = xt::empty_like(sorted);
        for (index_t i = 0; i < (index_t) reverse_sorted.size(); i++) {
            reverse_sorted(sorted(i)) = i;
        }

        auto &par = parents(tree);
        array_1d<index_t> new_par = xt::empty_like(sorted);
        for (index_t i = 0; i < (index_t) reverse_sorted.size(); i++) {
            new_par(i) = reverse_sorted(par(sorted(i)));
        }

        return make_remapped_tree(hg::tree(new_par, tree.category()), std::move(sorted));
    };


}
