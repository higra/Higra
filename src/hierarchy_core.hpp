//
// Created by user on 4/5/18.
//

#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xgenerator.hpp"
#include "unionfind.hpp"
#include "graph.hpp"
#include <algorithm>
#include <utility>
#include <tuple>

namespace hg {


    namespace hierarchy_core_internal {


    }


    template<typename graph_t, typename T>
    auto bptCanonical(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
        auto &edge_weights = xedge_weights.derived_cast();

        xt::xarray<std::size_t> sorted_edges_indices = xt::arange(num_edges(graph));
        std::stable_sort(sorted_edges_indices.begin(), sorted_edges_indices.end(),
                         [&edge_weights](std::size_t i, std::size_t j) { return edge_weights(i) < edge_weights(j); });

        auto num_points = num_vertices(graph);

        auto num_edge_mst = num_points - 1;
        ugraph mst(num_points);

        union_find uf(num_points);

        xt::xarray<std::size_t> roots = xt::arange<std::size_t>(num_points);

        xt::xarray<std::size_t> parents = xt::arange<std::size_t>(num_points * 2 - 1);
        xt::xarray<typename T::value_type> levels = xt::zeros<std::size_t>({num_points * 2 - 1});

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

}
