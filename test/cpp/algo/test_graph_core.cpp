/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "../test_utils.hpp"
#include "higra/algo/graph_core.hpp"
#include "higra/utils.hpp"
#include "higra/image/graph_image.hpp"

using namespace hg;

namespace test_graph_core {
    TEST_CASE("graph cut 2 labelisation", "[graph_algorithm]") {

        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<char> edge_weights = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0};

        auto labels = graph_cut_2_labelisation(graph, edge_weights);

        array_1d<index_t> ref_labels = {1, 2, 2, 1, 1, 3, 1, 3, 3};
        REQUIRE(is_in_bijection(labels, ref_labels));
    }

    TEST_CASE("labelisation 2 graph cut", "[graph_algorithm]") {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<index_t> labels = {1, 2, 2, 1, 1, 3, 1, 3, 3};

        auto edge_weights = labelisation_2_graph_cut(graph, labels);

        array_1d<char> ref_edge_weights = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0};

        REQUIRE(is_in_bijection(edge_weights, ref_edge_weights));
    }

    TEST_CASE("minimum spanning tree", "[graph_algorithm]") {
        auto graph = get_4_adjacency_graph({2, 3});

        xt::xarray<double> edge_weights{1, 0, 2, 1, 1, 1, 2};

        auto res = minimum_spanning_tree(graph, edge_weights);
        auto &mst = res.mst;
        auto &mst_edge_map = res.mst_edge_map;

        REQUIRE(num_vertices(mst) == 6);
        REQUIRE(num_edges(mst) == 5);
        std::vector<ugraph::edge_descriptor> ref = {{0, 3, 0},
                                                    {0, 1, 1},
                                                    {1, 4, 2},
                                                    {2, 5, 3},
                                                    {1, 2, 4}};
        for (index_t i = 0; i < (index_t) ref.size(); i++) {
            auto e = edge_from_index(i, mst);
            REQUIRE(e == ref[i]);
        }
        REQUIRE(xt::equal(mst_edge_map, array_1d<int>({1, 0, 3, 4, 2}))());
    }

    TEST_CASE("minimum spanning forest", "[graph_algorithm]") {
        ugraph graph(6);
        add_edge(0, 1, graph);
        add_edge(0, 2, graph);
        add_edge(1, 2, graph);

        add_edge(3, 4, graph);
        add_edge(3, 5, graph);
        add_edge(4, 5, graph);

        xt::xarray<double> edge_weights{0, 1, 2, 3, 4, 5};

        auto res = minimum_spanning_tree(graph, edge_weights);
        auto &mst = res.mst;
        auto &mst_edge_map = res.mst_edge_map;

        REQUIRE(num_vertices(mst) == 6);
        REQUIRE(num_edges(mst) == 4);
        std::vector<ugraph::edge_descriptor> ref = {{0, 1, 0},
                                                    {0, 2, 1},
                                                    {3, 4, 2},
                                                    {3, 5, 3}};
        for (index_t i = 0; i < (index_t) ref.size(); i++) {
            auto e = edge_from_index(i, mst);
            REQUIRE(e == ref[i]);
        }

        REQUIRE((mst_edge_map == array_1d<int>({0, 1, 3, 4})));
    }

}
