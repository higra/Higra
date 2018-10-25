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
    template <typename graph_t, typename T1, typename T2>
    auto mean_pb_hierarchy(const graph_t & graph,
                 const embedding_grid_2d &embedding,
                 const xt::xexpression<T1> & xedge_weights,
                 const xt::xexpression<T2> & xedge_orientations = array_nd<int>()){
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

        if(edge_orientations.dimension() != 0){
            // reweighting contours according to contour orientations
            hg_assert_edge_weights(graph, edge_orientations);
            hg_assert_1d_array(edge_orientations);

            auto watershed_cut =  weight_graph(graph, watershed_labels, weight_functions::L0);
            auto contour2d = fit_contour_2d(graph, embedding, watershed_cut);
            contour2d.subdivide();



            for(auto & polyline: contour2d){
                for(auto & segment: polyline){
                    auto segment_orientation = std::fmod(segment.angle(), M_PI);

                    for(auto element: segment){
                        auto edge_index = element.first;
                        auto edge_weight = edge_weights(edge_index);
                        auto edge_orientation = edge_orientations(edge_index);
                        auto new_weight = edge_weight
                                          * std::abs(std::cos(edge_orientation - M_PI/2.0 - segment_orientation));
                        if(new_weight > final_weights(edge_index)){
                            final_weights(edge_index) = new_weight;
                        }
                    }
                }
            }
        }else{
            final_weights = edge_weights;
        }


        // compute rag edge weights
        auto rag_edge_weights = rag_accumulate(rag.edge_map, final_weights, accumulator_mean());
        auto rag_edge_length = rag_accumulate(rag.edge_map, final_weights, accumulator_counter());

        auto tree = binary_partition_tree(rag.rag,
                                         rag_edge_weights,
                                         make_binary_partition_tree_average_linkage(rag_edge_weights, rag_edge_length));
        return std::make_pair(std::move(rag), std::move(tree));
    }
}
