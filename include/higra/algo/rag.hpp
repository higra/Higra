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

#include "xtensor/xsort.hpp"

#include "../graph.hpp"
#include "../accumulator/at_accumulator.hpp"


namespace hg {

    /**
     * Result of the region adjacency graph (rag) construction algorithm
     */
    struct region_adjacency_graph {
        /**
         * The region adjacency graph
         */
        ugraph rag;

        /**
         * An array indicating for each vertex of the original graph, the corresponding vertex of the rag
         */
        array_1d<index_t> vertex_map;

        /**
         * An array indicating for each edge of the original graph, the corresponding edge of the rag.
         * An edge with no corresponding edge in the rag (edge within a region) is indicated with the value invalid_index.
         */
        array_1d<index_t> edge_map;
    };

    /**
     * Construct a region adjacency graph from a vertex labeled graph in linear time.
     * @tparam graph_t
     * @tparam T
     * @param graph
     * @param xvertex_labels
     * @return see struct region_adjacency_graph
     */
    template<typename graph_t, typename T>
    auto
    make_region_adjacency_graph_from_labelisation(const graph_t &graph, const xt::xexpression<T> &xvertex_labels) {
        HG_TRACE();
        auto &vertex_labels = xvertex_labels.derived_cast();
        hg_assert_vertex_weights(graph, vertex_labels);
        hg_assert_1d_array(vertex_labels);
        hg_assert_integral_value_type(vertex_labels);

        ugraph rag;

        array_1d<index_t> vertex_map({num_vertices(graph)}, invalid_index);
        array_1d<index_t> edge_map({num_edges(graph)}, invalid_index);

        index_t num_regions = 0;
        index_t num_edges = 0;

        std::vector<index_t> canonical_edge_indexes;

        stackv<index_t> s;

        auto explore_component =
                [&s, &graph, &vertex_labels, &rag, &vertex_map, &edge_map, &num_regions, &num_edges, &canonical_edge_indexes]
                        (index_t start_vertex) {

                    auto label_region = vertex_labels[start_vertex];
                    s.push(start_vertex);
                    vertex_map[start_vertex] = num_regions;
                    canonical_edge_indexes.push_back(-1);
                    add_vertex(rag);
                    auto lowest_edge = num_edges;
                    while (!s.empty()) {
                        auto v = s.top();
                        s.pop();

                        for (auto e: out_edge_iterator(v, graph)) {
                            auto adjv = target(e, graph);
                            if (vertex_labels[adjv] == label_region) {
                                if (vertex_map[adjv] == invalid_index) {
                                    vertex_map[adjv] = num_regions;
                                    s.push(adjv);
                                }
                            } else {
                                if (vertex_map[adjv] != invalid_index) {
                                    auto num_region_adjacent = vertex_map[adjv];
                                    if (canonical_edge_indexes[num_region_adjacent] < lowest_edge) {
                                        add_edge(num_region_adjacent, num_regions, rag);
                                        edge_map(e) = num_edges;
                                        canonical_edge_indexes[num_region_adjacent] = num_edges;
                                        num_edges++;
                                    } else {
                                        edge_map(e) = canonical_edge_indexes[num_region_adjacent];
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

    /**
     * Construct a region adjacency graph from a graph cut in linear time.
     * Any edge with weight different from 0 belongs to the cut.
     *
     * TODO factorize method with  make_region_adjacency_graph_from_labelisation
     *
     * @tparam graph_t
     * @tparam T
     * @param graph
     * @param xedge_weights
     * @return see struct region_adjacency_graph
     */
    template<typename graph_t, typename T>
    auto
    make_region_adjacency_graph_from_graph_cut(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);

        ugraph rag;

        array_1d<index_t> vertex_map({num_vertices(graph)}, invalid_index);
        array_1d<index_t> edge_map({num_edges(graph)}, invalid_index);

        index_t num_regions = 0;
        index_t num_edges = 0;

        std::vector<index_t> canonical_edge_indexes;

        stackv<index_t> s;

        auto explore_component =
                [&s, &graph, &edge_weights, &rag, &vertex_map, &edge_map, &num_regions, &num_edges, &canonical_edge_indexes]
                        (index_t start_vertex) {
                    s.push(start_vertex);
                    vertex_map[start_vertex] = num_regions;
                    add_vertex(rag);
                    canonical_edge_indexes.push_back(-1);
                    auto lowest_edge = num_edges;
                    while (!s.empty()) {
                        auto v = s.top();
                        s.pop();

                        for (auto e: out_edge_iterator(v, graph)) {
                            auto adjv = target(e, graph);
                            if (edge_weights(index(e, graph)) == 0) {
                                if (vertex_map[adjv] == invalid_index) {
                                    vertex_map[adjv] = num_regions;
                                    s.push(adjv);
                                }
                            } else {
                                if (vertex_map[adjv] != invalid_index) {
                                    auto num_region_adjacent = vertex_map[adjv];
                                    if (canonical_edge_indexes[num_region_adjacent] < lowest_edge) {
                                        add_edge(num_region_adjacent, num_regions, rag);
                                        edge_map(e) = num_edges;
                                        canonical_edge_indexes[num_region_adjacent] = num_edges;
                                        num_edges++;
                                    } else {
                                        edge_map(e) = canonical_edge_indexes[num_region_adjacent];
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
            std::vector<size_t> shape;
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

    }

    /**
     * Projects weights on the rag (vertices or edges) onto the original graph.
     * @tparam T
     * @param rag_map rag vertex_map or rag edge_map (see struct region_adjacency_graph)
     * @param xrag_weights node or edge weights of the rag (depending of the provided rag_map)
     * @return original graph (vertices or edges) weights
     */
    template<typename T>
    auto
    rag_back_project_weights(const array_1d<index_t> &rag_map, const xt::xexpression<T> &xrag_weights) {
        if (xrag_weights.derived_cast().dimension() == 1) {
            return rag_internal::rag_back_project_weights<false>(rag_map, xrag_weights);
        } else {
            return rag_internal::rag_back_project_weights<true>(rag_map, xrag_weights);
        }

    }

    /**
     * Accumulate original graph (vertices or edges) weights onto the rag (vertices or edges)
     * @tparam T
     * @tparam accumulator_t
     * @tparam output_t
     * @param rag_map rag vertex_map or rag edge_map (see struct region_adjacency_graph)
     * @param xweights node or edge weights of the original graph (depending of the provided rag_map)
     * @param accumulator
     * @return rag (vertices or edges) weights
     */
    template<typename T, typename accumulator_t, typename output_t = typename T::value_type>
    auto rag_accumulate(const array_1d<index_t> &rag_map,
                        const xt::xexpression<T> &xweights,
                        const accumulator_t &accumulator) {
        return accumulate_at(rag_map, xweights, accumulator);
    }
}
