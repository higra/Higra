//
// Created by perretb on 17/05/18.
//

#pragma once

#include "../graph.hpp"
#include <stack>
#include <map>
#include "../structure/details/light_axis_view.hpp"
#include "../accumulator/accumulator.hpp"

namespace hg {

    struct region_adjacency_graph {
        ugraph rag;
        array_1d<index_t> vertex_map;
        array_1d<index_t> edge_map;
    };

    /**
     * Construct a region adjacency graph from a vertex labeled graph.
     * @tparam graph_t
     * @tparam T
     * @param graph
     * @param xvertex_labels
     * @return
     */
    template<typename graph_t, typename T>
    auto
    make_region_adjacency_graph(const graph_t &graph, const xt::xexpression<T> &xvertex_labels) {
        HG_TRACE();
        static_assert(std::is_integral<typename T::value_type>::value, "Labels must have an integral type.");
        auto &vertex_labels = xvertex_labels.derived_cast();
        hg_assert(vertex_labels.dimension() == 1, "Vertex labels must be scalar numbers.");
        hg_assert(vertex_labels.size() == num_vertices(graph),
                  "Vertex labels size does not match graph number of vertices.");

        ugraph rag;

        array_1d<index_t> vertex_map({num_vertices(graph)}, invalid_index);
        array_1d<index_t> edge_map({num_edges(graph)}, invalid_index);

        index_t num_regions = 0;
        index_t num_edges = 0;

        std::vector<long> canonical_edge_indexes;

        std::stack<index_t> s;

        auto explore_component =
                [&s, &graph, &vertex_labels, &rag, &vertex_map, &edge_map, &num_regions, &num_edges, &canonical_edge_indexes]
                        (index_t start_vertex) {

                    auto label_region = vertex_labels[start_vertex];
                    s.push(start_vertex);
                    vertex_map[start_vertex] = num_regions;
                    add_vertex(rag);
                    auto lowest_edge = num_edges;
                    while (!s.empty()) {
                        auto v = s.top();
                        s.pop();

                        for (auto ei: out_edge_index_iterator(v, graph)) {
                            auto e = edge(ei, graph);
                            auto adjv = other_vertex(e, v, graph);
                            if (vertex_labels[adjv] == label_region) {
                                if (vertex_map[adjv] == invalid_index) {
                                    vertex_map[adjv] = num_regions;
                                    s.push(adjv);
                                    canonical_edge_indexes.push_back(-1);
                                }
                            } else {
                                if (vertex_map[adjv] != invalid_index) {
                                    auto num_region_adjacent = vertex_map[adjv];
                                    if (canonical_edge_indexes[num_region_adjacent] < lowest_edge) {
                                        add_edge(num_region_adjacent, num_regions, rag);
                                        edge_map[ei] = num_edges;
                                        canonical_edge_indexes[num_region_adjacent] = num_edges;
                                        num_edges++;
                                    } else {
                                        edge_map[ei] = canonical_edge_indexes[num_region_adjacent];
                                    }
                                }
                            }
                        }
                    }
                    num_regions++;
                };


        for (auto v: vertex_iterator(graph)) {
            if (vertex_map[v] != invalid_index)
                continue;
            explore_component(v);

        }

        return region_adjacency_graph{std::move(rag), std::move(vertex_map), std::move(edge_map)};
    }

    namespace rag_internal {
        template<bool vectorial, typename T>
        auto
        rag_back_project_weights(const array_1d<index_t> &rag_map, const xt::xexpression<T> &xrag_weights) {
            HG_TRACE();
            auto &rag_weights = xrag_weights.derived_cast();

            index_t numv = rag_map.size();
            std::vector<std::size_t> shape;
            shape.push_back(numv);
            shape.insert(shape.end(), rag_weights.shape().begin() + 1, rag_weights.shape().end());
            array_nd<typename T::value_type> weights = xt::zeros<typename T::value_type>(shape);


            auto input_view = make_light_axis_view<vectorial>(rag_weights);
            auto output_view = make_light_axis_view<vectorial>(weights);

            for (index_t i = 0; i < numv; ++i) {
                if (rag_map.data()[i] != invalid_index) {
                    output_view.set_position(i);
                    input_view.set_position(rag_map.data()[i]);
                    output_view = input_view;
                }
            }

            return weights;
        }

        template<bool vectorial,
                typename T,
                typename accumulator_t,
                typename output_t = typename T::value_type>
        auto
        rag_accumulate(const array_1d<index_t> &rag_map,
                       const xt::xexpression<T> &xweights,
                       const accumulator_t &accumulator) {
            HG_TRACE();
            auto &weights = xweights.derived_cast();
            hg_assert(weights.shape()[0] == rag_map.size(), "Weights dimension does not match rag map dimension.");

            index_t size = xt::amax(rag_map)() + 1;
            auto data_shape = std::vector<std::size_t>(weights.shape().begin() + 1, weights.shape().end());
            auto output_shape = accumulator_t::get_output_shape(data_shape);
            output_shape.insert(output_shape.begin(), size);
            array_nd<typename T::value_type> rag_weights = array_nd<typename T::value_type>::from_shape(output_shape);

            auto input_view = make_light_axis_view<vectorial>(weights);
            auto output_view = make_light_axis_view<vectorial>(rag_weights);

            std::vector<decltype(accumulator.template make_accumulator<vectorial>(output_view))> accs;
            accs.reserve(size);


            for (index_t i = 0; i < size; ++i) {
                output_view.set_position(i);
                accs.push_back(accumulator.template make_accumulator<vectorial>(output_view));
                accs[i].initialize();
            }

            index_t map_size = rag_map.size();
            for (index_t i = 0; i < map_size; ++i) {
                if (rag_map.data()[i] != invalid_index) {
                    input_view.set_position(i);
                    accs[rag_map.data()[i]].accumulate(input_view.begin());
                }
            }

            for (auto &acc: accs) {
                acc.finalize();
            }

            return rag_weights;
        }
    }

    template<typename T>
    auto
    rag_back_project_weights(const array_1d<index_t> &rag_map, const xt::xexpression<T> &xrag_weights) {
        if (xrag_weights.derived_cast().dimension() == 1) {
            return rag_internal::rag_back_project_weights<false>(rag_map, xrag_weights);
        } else {
            return rag_internal::rag_back_project_weights<true>(rag_map, xrag_weights);
        }

    }

    template<typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto rag_accumulate(const array_1d<index_t> &rag_map,
                        const xt::xexpression<T> &xweights,
                        const accumulator_t &accumulator) {
        if (xweights.derived_cast().dimension() == 1) {
            return rag_internal::rag_accumulate<false, T, accumulator_t, output_t>(rag_map, xweights, accumulator);
        } else {
            return rag_internal::rag_accumulate<true, T, accumulator_t, output_t>(rag_map, xweights, accumulator);
        }
    };
}
