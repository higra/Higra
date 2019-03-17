/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <boost/test/unit_test.hpp>
#include "../test_utils.hpp"
#include "higra/algo/graph_core.hpp"
#include "higra/utils.hpp"
#include "higra/image/graph_image.hpp"

using namespace hg;

BOOST_AUTO_TEST_SUITE(test_cut);


    BOOST_AUTO_TEST_CASE(test_graph_cut_2_labelisation) {
        auto graph = get_4_adjacency_graph({3, 3});
        array_1d<char> edge_weights={1,0,0,1,1,0,0,1,1,0,1,0};

        auto labels = graph_cut_2_labelisation(graph, edge_weights);

        array_1d<index_t> ref_labels = {1, 2, 2, 1, 1, 3, 1, 3, 3};
        BOOST_CHECK(is_in_bijection(labels, ref_labels));
    }

    BOOST_AUTO_TEST_CASE(test_labelisation_2_graph_cut) {
            auto graph = get_4_adjacency_graph({3, 3});
            array_1d<index_t> labels = {1, 2, 2, 1, 1, 3, 1, 3, 3};

            auto edge_weights = labelisation_2_graph_cut(graph, labels);

            array_1d<char> ref_edge_weights={1,0,0,1,1,0,0,1,1,0,1,0};

            BOOST_CHECK(is_in_bijection(edge_weights, ref_edge_weights));
    }

    BOOST_AUTO_TEST_CASE(test_minimum_spanning_tree) {
            auto graph = get_4_adjacency_graph({2, 3});

            xt::xarray<double> edge_weights{1, 0, 2, 1, 1, 1, 2};

            auto res = minimum_spanning_tree(graph, edge_weights);
            auto &mst = res.mst;
            auto &mst_edge_map = res.mst_edge_map;

            BOOST_CHECK(num_vertices(mst) == 6);
            BOOST_CHECK(num_edges(mst) == 5);
            std::vector<ugraph::edge_descriptor> ref = {{0, 3, 0},
                                                        {0, 1, 1},
                                                        {1, 4, 2},
                                                        {2, 5, 3},
                                                        {1, 2, 4}};
            for (index_t i = 0; i < (index_t) ref.size(); i++) {
                    auto e = edge_from_index(i, mst);
                    BOOST_CHECK(e == ref[i]);
            }
            BOOST_CHECK(mst_edge_map == array_1d<int>({1, 0, 3, 4, 2}));
    }

    BOOST_AUTO_TEST_CASE(test_minimum_spanning_forest) {
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

            BOOST_CHECK(num_vertices(mst) == 6);
            BOOST_CHECK(num_edges(mst) == 4);
            std::vector<ugraph::edge_descriptor> ref = {{0, 1, 0},
                                                        {0, 2, 1},
                                                        {3, 4, 2},
                                                        {3, 5, 3}};
            for (index_t i = 0; i < (index_t) ref.size(); i++) {
                    auto e = edge_from_index(i, mst);
                    BOOST_CHECK(e == ref[i]);
            }

            BOOST_CHECK(mst_edge_map == array_1d<int>({0, 1, 3, 4}));
    }

BOOST_AUTO_TEST_SUITE_END();


