//
// Created by perretb on 17/07/18.
//

#pragma once

#include "../graph.hpp"
#include "../structure/fibonacci_heap.hpp"

namespace hg {

    namespace binary_partition_tree_internal {

        template<typename T>
        struct heap_element {
            using self_t = heap_element<T>;
            T value;
            index_t index;

            bool operator==(const self_t &rhs) const { return value == rhs.value; }

            bool operator!=(const self_t &rhs) const { return value != rhs.value; }

            bool operator<(const self_t &rhs) const { return value < rhs.value; }

            bool operator>(const self_t &rhs) const { return value > rhs.value; }

            bool operator<=(const self_t &rhs) const { return value <= rhs.value; }

            bool operator>=(const self_t &rhs) const { return value >= rhs.value; }

        };

        template<typename graph_t,
                typename T,
                typename heap_t>
        void init_heap(const graph_t &graph,
                       const xt::xexpression<T> &xedge_weights,
                       array_1d<bool> &active,
                       heap_t &heap) {
            auto &edge_weights = xedge_weights.derived_cast();

            for (auto v: vertex_iterator(graph)) {
                auto min_value = std::numeric_limits<typename T::value_type>::max();
                auto index = invalid_index;
                for (auto ei: out_edge_index_iterator(v, graph)) {
                    if (edge_weights[ei] < min_value) {
                        min_value = edge_weights[ei];
                        index = ei;
                    }
                }
                if (!active[index]) {
                    heap.push({min_value, index});
                    active[index] = true;
                }
            }
        };

        template<typename graph_t, typename T>
        auto binary_partition_tree(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
            auto g = copy_graph<undirected_graph<hash_setS> >(graph); // optimized for removal
            auto &edge_weights = xedge_weights.derived_cast();

            auto num_points = num_vertices(g);

            auto num_nodes_tree = num_points * 2 - 1;

            array_1d<index_t> parents = xt::arange(num_nodes_tree);
            array_1d<typename T::value_type> levels = xt::zeros<typename T::value_type>({num_nodes_tree});
            array_1d<bool> active = xt::zeros<bool>({num_edges(g)});

            fibonacci_heap<heap_element<typename T::value_type>> heap;

            init_heap(graph, edge_weights, active, heap);

            size_t current_num_nodes_tree = num_points;
            while (!heap.empty() && current_num_nodes_tree < num_nodes_tree) {

                auto min_element = heap.top()->get_value();

                auto fusion_edge_index = min_element.index;
                auto fusion_edge_weight = min_element.value;

                heap.pop();
                if (active[fusion_edge_index]) {
                    auto new_parent = g.add_vertex();
                    auto fusion_edge = edge(fusion_edge_index, g);
                    auto region1 = source(fusion_edge, g);
                    auto region2 = target(fusion_edge, g);
                    parents[region1] = new_parent;
                    parents[region2] = new_parent;
                    levels[new_parent] = fusion_edge_weight;
                    current_num_nodes_tree++;

                    remove_edge(fusion_edge_index, g);
                    active[fusion_edge_index] = false;
                }
            }
        }


    }
}
