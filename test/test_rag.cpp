//
// Created by perretb on 17/05/18.
//

#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/algo/rag.hpp"
#include "higra/algo/graph_image.hpp"


using namespace hg;

BOOST_AUTO_TEST_SUITE(region_adjacency_graph);

    BOOST_AUTO_TEST_CASE(rag_simple) {

        auto g = hg::get_4_adjacency_graph({4, 4});
        array_1d<int> vertex_labels{1, 1, 5, 5,
                                    1, 1, 5, 5,
                                    1, 1, 3, 3,
                                    1, 1, 10, 10};

        auto rag_and_maps = make_region_adjacency_graph(g, vertex_labels);

        auto &rag = rag_and_maps.rag;
        auto &vertex_map = rag_and_maps.vertex_map;
        auto &edge_map = rag_and_maps.edge_map;


        BOOST_CHECK(num_vertices(rag) == 4);
        BOOST_CHECK(num_edges(rag) == 5);

        std::vector<std::pair<std::size_t, std::size_t>> expected_edges = {{0, 1},
                                                                           {1, 2},
                                                                           {0, 2},
                                                                           {2, 3},
                                                                           {0, 3}};
        std::size_t i = 0;
        for (auto e: edge_iterator(rag)) {
            BOOST_CHECK(e == expected_edges[i++]);
        }

        std::vector<std::vector<std::size_t>> expected_vertex_map{{0,  1, 4, 5, 8, 9, 12, 13},
                                                                  {2,  3, 6, 7},
                                                                  {10, 11},
                                                                  {14, 15}};
        BOOST_CHECK(vertex_map == expected_vertex_map);

        std::vector<std::vector<std::size_t>> expected_edge_map{{2,  9},
                                                                {12, 13},
                                                                {16},
                                                                {19, 20},
                                                                {22}};

        BOOST_CHECK(edge_map == expected_edge_map);

    }

BOOST_AUTO_TEST_SUITE_END();