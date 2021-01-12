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

#include "common.hpp"
#include "higra/structure/unionfind.hpp"
#include "higra/graph.hpp"
#include "higra/sorting.hpp"
#include "higra/accumulator/tree_accumulator.hpp"
#include "higra/structure/lca_fast.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xnoalias.hpp"
#include <utility>
#include <tuple>
#include <queue>

namespace hg {

    /**
     * A simple structure to hold the result of canonical bpt function.
     *
     * See make_node_weighted_tree_and_mst for construction
     *
     * @tparam tree_t
     * @tparam altitude_t
     * @tparam mst_t
     */
    template<typename tree_t, typename altitude_t>
    struct node_weighted_tree_and_mst {
        tree_t tree;
        altitude_t altitudes;
        array_1d<index_t> mst_edge_map;
    };

    template<typename tree_t, typename altitude_t>
    decltype(auto) make_node_weighted_tree_and_mst(
            tree_t &&tree,
            altitude_t &&node_altitude,
            array_1d<index_t> &&mst_edge_map) {
        return node_weighted_tree_and_mst<tree_t, altitude_t>{std::forward<tree_t>(tree),
                                                              std::forward<altitude_t>(node_altitude),
                                                              std::forward<array_1d<index_t> >(mst_edge_map)};
    }

    namespace hierarchy_core_internal {
        template<typename E1, typename E2, typename T>
        auto bpt_canonical_from_sorted_edges(const xt::xexpression<E1> &xsources,
                                             const xt::xexpression<E2> &xtargets,
                                             const xt::xexpression<T> &xsorted_edge_indices,
                                             const index_t num_vertices) {
            HG_TRACE();
            auto &sorted_edge_indices = xsorted_edge_indices.derived_cast();
            auto &sources = xsources.derived_cast();
            auto &targets = xtargets.derived_cast();
            hg_assert_1d_array(sources);
            hg_assert_same_shape(sources, targets);
            hg_assert_same_shape(sources, sorted_edge_indices);
            hg_assert_integral_value_type(sources);
            hg_assert_integral_value_type(targets);
            hg_assert_integral_value_type(sorted_edge_indices);

            auto num_edge_mst = num_vertices - 1;

            array_1d<index_t> mst_edge_map = xt::empty<index_t>({num_edge_mst});

            union_find uf(num_vertices);

            array_1d<index_t> roots = xt::arange<index_t>(num_vertices);
            array_1d<index_t> parents = xt::arange<index_t>(num_vertices * 2 - 1);

            index_t num_nodes = num_vertices;
            index_t num_edge_found = 0;
            index_t i = 0;

            while (num_edge_found < num_edge_mst && i < (index_t) sorted_edge_indices.size()) {
                auto ei = sorted_edge_indices[i];
                auto c1 = uf.find(sources(ei));
                auto c2 = uf.find(targets(ei));
                if (c1 != c2) {
                    parents[roots[c1]] = num_nodes;
                    parents[roots[c2]] = num_nodes;
                    auto newRoot = uf.link(c1, c2);
                    roots[newRoot] = num_nodes;
                    mst_edge_map(num_edge_found) = ei;
                    num_nodes++;
                    num_edge_found++;
                }
                i++;
            }
            hg_assert(num_edge_found == num_edge_mst, "Input graph must be connected.");

            return std::make_pair(
                    parents,
                    std::move(mst_edge_map));
        };
    }

    /**
     * Compute the canonical binary partition tree (or binary partition tree by altitude ordering) of the given
     * edge weighted graph.
     *
     * The algorithm returns a tuple composed of:
     *  - the binary partition tree,
     *  - the levels of the vertices of the tree,
     *  - the minimum spanning tree of the given graph that corresponds to this tree.
     *
     * L. Najman, J. Cousty, B. Perret. Playing with Kruskal: algorithms for morphological trees in edge-weighted graphs.
     * In, 11th International Symposium on Mathematical Morphology, ISMM 2013, Uppsala, Sweden, Mai 2013.
     *
     * @tparam graph_t
     * @tparam T
     * @param graph
     * @param xedge_weights
     * @return
     */
    template<typename graph_t, typename T>
    auto bpt_canonical(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);

        array_1d<index_t> sorted_edges_indices = stable_arg_sort(edge_weights);

        auto res = hierarchy_core_internal::bpt_canonical_from_sorted_edges(sources(graph),
                                                                            targets(graph),
                                                                            sorted_edges_indices,
                                                                            num_vertices(graph));
        auto &parents = res.first;
        auto &mst_edge_map = res.second;

        auto num_points = num_vertices(graph);

        array_1d<typename T::value_type> levels = xt::zeros<typename T::value_type>({parents.size()});
        xt::noalias(xt::view(levels, xt::range(num_points, levels.size()))) = xt::index_view(edge_weights,
                                                                                             mst_edge_map);

        return make_node_weighted_tree_and_mst(
                tree(std::move(parents)),
                std::move(levels),
                std::move(mst_edge_map));
    };


    /**
     * Creates a copy of the current Tree and deletes the nodes such that the criterion function is true.
     * Also returns an array that maps any node index i of the new tree, to the index of this node in the original tree.
     *
     * The criterion function is a predicate that associates true (this node must be deleted) or
     * false (do not delete this node) to a node index (with operator ()).
     *
     * @tparam criterion_t
     * @param t input tree
     * @param criterion For any vertex n of the tree, n has to be removed if criterion(n) == true
     * @param process_leaves If false, a leaf vertex will never be removed disregarding the value of criterion.
     * @return a remapped_tree
     */
    template<typename criterion_t>
    auto simplify_tree(const tree &t, const criterion_t &criterion, bool process_leaves = false) {
        HG_TRACE();


        // this case is significantly harder because a reordering of the nodes
        // may append if an internal node becomes a leaf
        if (process_leaves) {

            t.compute_children();

            // ********************************
            // identification of deleted sub-trees
            // true if all nodes below a given  node are deleted
            // a non deleted non leaf node i such that removed_branch(i) && !removed_branch(parent(i)) is thus a new leaf
            array_1d<bool> removed_branch = xt::zeros<bool>({num_vertices(t)});
            for (index_t i = 0; i < (index_t) num_leaves(t); i++) {
                removed_branch(i) = criterion(i);
            }

            for (index_t i = num_leaves(t); i < (index_t) num_vertices(t); i++) {
                bool flag = true;
                for (index_t c = 0; flag && c < (index_t) num_children(i, t); c++) {
                    auto cc = child(c, i, t);
                    flag = flag && removed_branch(cc) && criterion(cc);
                }
                removed_branch(i) = flag;
            }

            // ********************************
            // Identification and labeling of leaves

            std::vector<index_t> new_leaves;
            index_t removed = 0;

            for (index_t i : leaves_iterator(t)) {
                if (!removed_branch(i)) {
                    new_leaves.push_back(i);
                } else {
                    removed++;
                }
            }

            for (index_t i : leaves_to_root_iterator(t, leaves_it::exclude, root_it::exclude)) {
                if (!criterion(i)) {
                    if (removed_branch(i) && !removed_branch(parent(i, t))) {
                        new_leaves.push_back(i);
                    }
                } else {
                    removed++;
                }
            }

            if (removed_branch(root(t))) {
                new_leaves.push_back(root(t));
            }

            // *******************************
            // Topological sort of remaining vertices
            // (with top-down traversal)
            // *******************************
            auto num_nodes_new_tree = num_vertices(t) - removed;
            array_1d<index_t> new_parent = xt::empty<index_t>({num_nodes_new_tree});
            array_1d<index_t> node_map = xt::empty<index_t>({num_nodes_new_tree});
            index_t node_number = num_nodes_new_tree - 1;

            std::queue<index_t> queue;

            // new index of each node
            array_1d<index_t> new_order({num_vertices(t)}, invalid_index);
            for (index_t i = 0; i < (index_t) new_leaves.size(); i++) {
                new_order(new_leaves[i]) = i;
            }


            queue.push(root(t));
            while (!queue.empty()) {
                auto e = queue.front();
                queue.pop();
                if (!criterion(e) || e == root(t)) {
                    new_order(e) = node_number;
                    new_parent(node_number) = new_order(parent(e, t));
                    node_map(node_number) = e;
                    node_number--;
                } else {
                    new_order(e) = new_order(parent(e, t));
                }

                for (auto c: children_iterator(e, t)) {
                    if (new_order(c) == invalid_index) { // new leaves
                        queue.push(c);
                    }
                }
            }

            index_t i = 0;
            for (auto n: new_leaves) {
                new_parent(i) = new_order(parent(n, t));
                node_map(i) = n;
                i++;
            }
            return make_remapped_tree(tree(new_parent, t.category()), std::move(node_map));
        } else {
            auto n_nodes = num_vertices(t);
            auto n_leaves = num_leaves(t);

            array_1d<index_t> new_ranks = xt::empty<index_t>({n_nodes});

            xt::view(new_ranks, xt::range(0, n_leaves)) = xt::arange<index_t>(n_leaves);
            index_t count = n_leaves;

            for (auto i: leaves_to_root_iterator(t, leaves_it::exclude, root_it::exclude)) {
                if (!criterion(i)) {
                    new_ranks(i) = count;
                    ++count;
                }
            }

            new_ranks(root(t)) = count;
            ++count;

            array_1d<index_t> new_parent = xt::empty<index_t>({count});
            array_1d<index_t> node_map = xt::empty<index_t>({count});

            count = new_parent.size() - 1;

            new_parent(count) = count;
            node_map(count) = root(t);
            --count;

            for (auto i: root_to_leaves_iterator(t, leaves_it::include, root_it::exclude)) {
                if (!criterion(i) || is_leaf(i, t)) {
                    new_parent(count) = new_ranks(parent(i, t));
                    node_map(count) = i;
                    --count;
                } else {
                    new_ranks(i) = new_ranks(parent(i, t));
                }
            }

            return make_remapped_tree(tree(std::move(new_parent), t.category()), std::move(node_map));
        }

    };

    /**
     * Compute the quasi-flat zone hierarchy of an edge weighted graph.
     * For a given positive real value lamba:
     *  - a set of vertices X is lambda-connected if, for any two vertices x, y in X there exists an xy-path in X
     *    composed of edges of weights smaller of equal than lambda;
     *  - a lambda-connected component is a lambda-connected set of maximal extent;
     *  - the set of lambda-connected components forms a partition, called lambda-partition, of the graph vertices.
     *
     * The quasi-flat zone hierarchy is composed of the sequence of lambda-partitions obtained
     * for all lambda in edge_weights.
     *
     * @tparam graph_t Input graph type
     * @tparam T xepression derived type of input edge weights
     * @param graph Input graph
     * @param xedge_weights Input graph weights
     * @return A node weighted tree
     */
    template<typename graph_t, typename T>
    auto quasi_flat_zone_hierarchy(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);

        auto bpt = bpt_canonical(graph, edge_weights);
        auto &tree = bpt.tree;
        auto &altitudes = bpt.altitudes;

        auto altitude_parents = propagate_parallel(tree, altitudes);

        auto qfz = simplify_tree(tree, xt::equal(altitudes, altitude_parents));
        auto &qfz_tree = qfz.tree;
        auto &node_map = qfz.node_map;

        auto qfz_altitude = xt::eval(xt::index_view(altitudes, node_map));

        return make_node_weighted_tree(std::move(qfz_tree), std::move(qfz_altitude));
    }

    /**
     * Compute the saliency map of the given hierarchy for the given graph.
     * The saliency map is a weighting of the graph edges.
     * The weight of an edge {x, y} is the altitude of the lowest common ancestor of x and y in the
     * hierarchy.
     *
     * @tparam graph_t Input graph type
     * @tparam tree_t Input tree type
     * @tparam T xepression derived type of input altitudes
     * @param graph Input graph
     * @param tree Input tree
     * @param xaltitudes Input node altitudes of the given tree
     * @return An array of shape (num_edges(graph)) and with the same value type as T.
     */
    template<typename graph_t, typename tree_t, typename T>
    auto saliency_map(const graph_t &graph,
                      const tree_t &tree,
                      const xt::xexpression<T> &xaltitudes) {
        auto &altitudes = xaltitudes.derived_cast();
        lca_fast lca(tree);
        auto lca_edges = lca.lca(edge_iterator(graph));
        return xt::eval(xt::index_view(altitudes, lca_edges));
    }

    /**
     * Transforms a tree into a binary tree.
     *
     * Each non-leaf node of the input tree must have at least 2 children!
     *
     * Whenever a non-leaf node :math:`n` with :math:`k > 2` children is found:
     *
     *  - an extra node :math:`m` is created;
     *  - the first 2 children of :math:`n` become children of the new node :math:`m`; and
     *  - the new node :math:`m` becomes the first child of :math:`n`.
     *
     * The number of children of :math:`n` is thus reduced by 1.
     * This operation is repeated :math:`k-2` times, i.e. until :math:`n` has only 2 children.
     *
     * @tparam tree_t Input tree type
     * @param tree Input tree
     * @return a remapped_tree
     */
    template<typename tree_t>
    auto tree_2_binary_tree(const tree_t &tree) {

        auto num_v = num_vertices(tree);
        auto num_l = num_leaves(tree);
        auto num_v_res = num_l * 2 - 1;

        tree.compute_children();
        array_1d<index_t> node_map = array_1d<index_t>::from_shape({num_v});
        array_1d<index_t> reverse_node_map = array_1d<index_t>::from_shape({num_v_res});
        for (index_t i = 0; i < (index_t) num_l; i++) {
            node_map(i) = i;
            reverse_node_map(i) = i;
        }

        array_1d<index_t> new_parents = array_1d<index_t>::from_shape({num_v_res});
        index_t cur_par_index = num_l;

        for (auto i: leaves_to_root_iterator(tree, leaves_it::exclude)) {
            auto num_c = (index_t) num_children(i, tree);
            new_parents(node_map(child(0, i, tree))) = cur_par_index;
            new_parents(node_map(child(1, i, tree))) = cur_par_index;

            for (index_t c = 2; c < num_c; c++) {
                new_parents(cur_par_index) = cur_par_index + 1;
                reverse_node_map(cur_par_index) = i;
                cur_par_index++;
                new_parents(node_map(child(c, i, tree))) = cur_par_index;
            }
            node_map(i) = cur_par_index;
            reverse_node_map(cur_par_index) = i;
            cur_par_index++;
        }

        new_parents(num_v_res - 1) = num_v_res - 1;

        return make_remapped_tree(hg::tree(new_parents, tree.category()), std::move(reverse_node_map));
    }
}
