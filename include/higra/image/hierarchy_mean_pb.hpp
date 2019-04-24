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
#include "graph_image.hpp"
#include "contour_2d.hpp"
#include "../algo/watershed.hpp"
#include "../algo/rag.hpp"
#include "../algo/graph_weights.hpp"
#include "../hierarchy/binary_partition_tree.hpp"


namespace hg {

    /**
     * Compute the *oriented watershed* as described in [ArbelaezPAMI2011]_ .
     * Given a 4 adjacency graph with edge boundary probabilities and estimated boundary orientations, the algorithms computes:
     *
     *  - a region adjacency graph of the watershed regions of the edge boundary probabilities
     *  - the boundaries between watershed regions are vectorized and simplified (see contour_2d class)
     *  - the orientation of each boundary element is estimated
     *  - the edge boundary probabilities are reweighted according to the concordance between user provided boundary orientations and estimated orientation of boundary elements
     *  - the weight of the region adjacency graph edges as the mean value of reweighted edge boundary probabilities on the frontier between the 2 regions
     *
     *  The algorithm returns the region adjacency graph of watershed pixels and its edge weights.
     *
     * .. [ArbelaezPAMI2011] Arbelaez, P., Maire, M., Fowlkes, C., & Malik, J..
     *    Contour detection and hierarchical image segmentation.
     *    IEEE transactions on pattern analysis and machine intelligence, 33(5), 898-916.
     *
     * @tparam graph_t
     * @tparam T1
     * @tparam T2
     * @param graph
     * @param embedding
     * @param xedge_weights
     * @param xedge_orientations
     * @return
     */
    template<typename graph_t, typename T1, typename T2>
    auto oriented_watershed(const graph_t &graph,
                            const embedding_grid_2d &embedding,
                            const xt::xexpression<T1> &xedge_weights,
                            const xt::xexpression<T2> &xedge_orientations = array_nd<int>()) {
        HG_TRACE();
        using value_t = typename T1::value_type;
        const auto &edge_weights = xedge_weights.derived_cast();
        const auto &edge_orientations = xedge_orientations.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);
        hg_assert(num_vertices(graph) == embedding.size(),
                  "Graph number of vertices does not match the size of the embedding.");
        auto watershed_labels = labelisation_watershed(graph, edge_weights);
        auto rag = make_region_adjacency_graph_from_labelisation(graph, watershed_labels);

        array_1d<value_t> final_weights = xt::zeros<value_t>({num_edges(graph)});

        if (edge_orientations.dimension() != 0) {
            // reweighting contours according to contour orientations
            hg_assert_edge_weights(graph, edge_orientations);
            hg_assert_1d_array(edge_orientations);

            auto watershed_cut = weight_graph(graph, watershed_labels, weight_functions::L0);
            auto contour2d = fit_contour_2d(graph, embedding, watershed_cut);
            contour2d.subdivide();

            for (auto &polyline: contour2d) {
                for (auto &segment: polyline) {
                    auto segment_orientation = std::fmod(segment.angle(), xt::numeric_constants<double>::PI);

                    for (auto element: segment) {
                        auto edge_index = element.first;
                        auto edge_weight = edge_weights(edge_index);
                        auto edge_orientation = edge_orientations(edge_index);
                        auto new_weight = edge_weight
                                          * std::abs(
                                std::cos(edge_orientation - xt::numeric_constants<double>::PI_2 - segment_orientation));
                        if (new_weight > final_weights(edge_index)) {
                            final_weights(edge_index) = new_weight;
                        }
                    }
                }
            }
        } else {
            final_weights = edge_weights;
        }


        // compute rag edge weights
        auto rag_edge_weights = rag_accumulate(rag.edge_map, final_weights, accumulator_mean());

        return std::make_pair(std::move(rag), std::move(rag_edge_weights));
    }

    /**
     * Compute the *mean probability boundary hierarchy* as described in [ArbelaezPAMI2011]_ .
     * Given a 4 adjacency graph with edge boundary probabilities and estimated boundary orientations, the algorithms computes:
     *
     *  - the oriented watershed of the given graph
     *  - the average linkage clustering ot the oriented watershed
     *
     *  The algorithm returns the region adjacency graph of watershed pixels and teh valued tree computed on this graph.
     *
     * .. [ArbelaezPAMI2011] Arbelaez, P., Maire, M., Fowlkes, C., & Malik, J..
     *    Contour detection and hierarchical image segmentation.
     *    IEEE transactions on pattern analysis and machine intelligence, 33(5), 898-916.
     *
     * @tparam graph_t
     * @tparam T1
     * @tparam T2
     * @param graph
     * @param embedding
     * @param xedge_weights
     * @param xedge_orientations
     * @return
     */
    template<typename graph_t, typename T1, typename T2>
    auto mean_pb_hierarchy(const graph_t &graph,
                           const embedding_grid_2d &embedding,
                           const xt::xexpression<T1> &xedge_weights,
                           const xt::xexpression<T2> &xedge_orientations = array_nd<int>()) {
        HG_TRACE();

        using value_t = typename T1::value_type;
        const auto &edge_weights = xedge_weights.derived_cast();
        const auto &edge_orientations = xedge_orientations.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);
        hg_assert(num_vertices(graph) == embedding.size(),
                  "Graph number of vertices does not match the size of the embedding.");

        auto ows = oriented_watershed(graph, embedding, edge_weights, edge_orientations);
        auto &rag = ows.first;
        auto &rag_edge_weights = ows.second;

        auto rag_edge_length = rag_accumulate(rag.edge_map, edge_weights, accumulator_counter());

        auto tree = binary_partition_tree_average_linkage(rag.rag,
                                                          rag_edge_weights,
                                                          rag_edge_length);
        return std::make_pair(std::move(rag), std::move(tree));
    }
}
