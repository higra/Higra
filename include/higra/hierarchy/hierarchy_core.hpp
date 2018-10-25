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
#include "higra/accumulator/tree_accumulator.hpp"
#include "xtensor/xindex_view.hpp"
#include <algorithm>
#include <utility>
#include <tuple>

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
    template<typename tree_t, typename altitude_t, typename mst_t>
    struct node_weighted_tree_and_mst {
        tree_t tree;
        altitude_t node_altitude;
        mst_t mst;
    };

    template<typename tree_t, typename altitude_t, typename mst_t>
    decltype(auto) make_node_weighted_tree_and_mst(tree_t &&tree, altitude_t &&node_altitude, mst_t &&mst) {
        return node_weighted_tree_and_mst<tree_t, altitude_t, mst_t>{std::forward<tree_t>(tree),
                                                                     std::forward<altitude_t>(node_altitude),
                                                                     std::forward<mst_t>(mst)};
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

        array_1d<index_t> sorted_edges_indices = xt::arange(num_edges(graph));
        std::stable_sort(sorted_edges_indices.begin(), sorted_edges_indices.end(),
                         [&edge_weights](index_t i, index_t j) { return edge_weights[i] < edge_weights[j]; });

        auto num_points = num_vertices(graph);

        auto num_edge_mst = num_points - 1;
        ugraph mst(num_points);

        union_find uf(num_points);

        array_1d<index_t> roots = xt::arange(num_points);
        array_1d<index_t> parents = xt::arange(num_points * 2 - 1);

        array_1d<typename T::value_type> levels = xt::zeros<typename T::value_type>({num_points * 2 - 1});

        size_t num_nodes = num_points;
        size_t num_edge_found = 0;
        index_t i = 0;

        while (num_edge_found < num_edge_mst && i < (index_t) sorted_edges_indices.size()) {
            auto ei = sorted_edges_indices[i];
            auto e = edge_from_index(ei, graph);
            auto c1 = uf.find(source(e, graph));
            auto c2 = uf.find(target(e, graph));
            if (c1 != c2) {
                num_edge_found++;
                levels[num_nodes] = edge_weights[ei];
                parents[roots[c1]] = num_nodes;
                parents[roots[c2]] = num_nodes;
                auto newRoot = uf.link(c1, c2);
                roots[newRoot] = num_nodes;
                mst.add_edge(e);
                num_nodes++;
            }
            i++;
        }
        hg_assert(num_edge_found == num_edge_mst, "Input graph must be connected.");

        return make_node_weighted_tree_and_mst(tree(parents), std::move(levels), std::move(mst));

    };

    /**
     * A simple structure to hold the result of simplify_tree algorithm.
     *
     * See make_node_weighted_tree for construction
     *
     * @tparam tree_t
     * @tparam node_map_t
     */
    template<typename tree_t, typename node_map_t>
    struct simplified_tree {
        tree_t tree;
        node_map_t node_map;
    };

    /// Creates a copy of the current Tree and deletes the nodes such that the criterion function is true.
    /// Also returns an array that maps any node index i of the new tree, to the index of this node in the original tree
    ///
    /// The criterion function is a predicate that associates true (this node must be deleted) or
    /// false (do not delete this node) to a node index (with operator ()).
    ///
    /// \tparam criterion_t
    /// \param tree
    /// \param criterion
    /// \return std::pair<tree, node_map>
    template<typename criterion_t>
    auto simplify_tree(const tree &t, const criterion_t &criterion) {
        HG_TRACE();
        auto n_nodes = num_vertices(t);
        auto copy_parent = parents(t);

        std::size_t count = 0;
        array_1d<tree::vertex_descriptor> deleted_map = xt::zeros<tree::vertex_descriptor>(copy_parent.shape());
        array_1d<bool> deleted = xt::zeros<bool>(copy_parent.shape());

        // from root to leaves, compute the new parent relation,
        // don't care of the  holes in the parent tab
        for (auto i: root_to_leaves_iterator(t, leaves_it::exclude, root_it::exclude)) {
            auto parent = copy_parent(i);
            if (criterion(i)) {
                for (auto c: children_iterator(i, t)) {
                    copy_parent(c) = parent;
                }
                count++;
            }
            // number of deleted nodes after node i
            deleted_map(i) = count;
        }

        //correct the mapping
        deleted_map = count - deleted_map;

        array_1d<tree::vertex_descriptor> new_parent = xt::arange<tree::vertex_descriptor>(0, n_nodes - count);
        array_1d<tree::vertex_descriptor> node_map = xt::zeros<tree::vertex_descriptor>({n_nodes - count});

        count = 0;

        for (auto i: leaves_to_root_iterator(t, leaves_it::include, root_it::exclude)) {
            if (i < num_leaves(t) || !criterion(i)) {
                auto par = copy_parent(i);
                auto new_par = par - deleted_map(par);
                node_map(count) = i;
                new_parent(count) = new_par;
                count++;
            }
        }

        node_map(node_map.size() - 1) = root(t);
        return simplified_tree<tree, decltype(node_map)>{tree(new_parent), std::move(node_map)};
    };

    template<typename graph_t, typename T>
    auto quasi_flat_zones_hierarchy(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);
    
        auto bpt = bpt_canonical(graph, edge_weights);
        auto &tree = bpt.tree;
        auto &altitudes = bpt.node_altitude;

        auto altitude_parents = propagate_parallel(tree, altitudes);

        auto qfz = simplify_tree(tree, xt::equal(altitudes, altitude_parents));
        auto &qfz_tree = qfz.tree;
        auto &node_map = qfz.node_map;

        auto qfz_altitude = xt::eval(xt::index_view(altitudes, node_map));

        return make_node_weighted_tree(std::move(qfz_tree), std::move(qfz_altitude));
    }


}
