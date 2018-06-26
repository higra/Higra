//
// Created by user on 4/5/18.
//

#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xgenerator.hpp"
#include "higra/structure/unionfind.hpp"
#include "higra/graph.hpp"
#include <algorithm>
#include <utility>
#include <tuple>

namespace hg {

    template<typename graph_t, typename T>
    auto bptCanonical(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert(edge_weights.dimension() == 1, "Edge weights must be scalar.");
        hg_assert(num_edges(graph) == edge_weights.size(),
                  "Edge weights size does not match the number of edge in the graph.");

        array_1d<std::size_t> sorted_edges_indices = xt::arange(num_edges(graph));
        std::stable_sort(sorted_edges_indices.begin(), sorted_edges_indices.end(),
                         [&edge_weights](std::size_t i, std::size_t j) { return edge_weights[i] < edge_weights[j]; });

        auto num_points = num_vertices(graph);

        auto num_edge_mst = num_points - 1;
        ugraph mst(num_points);

        union_find uf(num_points);

        array_1d<std::size_t> roots = xt::arange(num_points);
        array_1d<std::size_t> parents = xt::arange(num_points * 2 - 1);

        array_1d<typename T::value_type> levels = xt::zeros<typename T::value_type>({num_points * 2 - 1});

        std::size_t num_nodes = num_points;
        std::size_t num_edge_found = 0;
        std::size_t i = 0;

        while (num_edge_found < num_edge_mst && i < sorted_edges_indices.size()) {
            auto ei = sorted_edges_indices[i];
            auto e = edge(ei, graph);
            auto c1 = uf.find(source(e, graph));
            auto c2 = uf.find(target(e, graph));
            if (c1 != c2) {
                num_edge_found++;
                levels[num_nodes] = edge_weights[ei];
                parents[roots[c1]] = num_nodes;
                parents[roots[c2]] = num_nodes;
                parents[num_nodes] = num_nodes;
                auto newRoot = uf.link(c1, c2);
                roots[newRoot] = num_nodes;
                mst.add_edge(e);
                num_nodes++;
            }
            i++;
        }
        hg_assert(num_edge_found == num_edge_mst, "Input graph must be connected.");

        return std::make_tuple(tree(parents), std::move(levels), std::move(mst));

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
    auto simplify_tree(const tree &t, criterion_t &criterion) {
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
            if (!criterion(i)) {
                auto par = copy_parent(i);
                auto new_par = par - deleted_map(par);
                node_map(count) = i;
                new_parent(count) = new_par;
                count++;
            }
        }
        node_map(node_map.size() - 1) = root(t);
        return std::make_pair(tree(new_parent), std::move(node_map));
    };

}
