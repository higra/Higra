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
            index_t i = 0;
        for (auto e: edge_iterator(rag)) {
            BOOST_CHECK(e == expected_edges[i++]);
        }

            array_1d<index_t> expected_vertex_map{0, 0, 1, 1,
                                                  0, 0, 1, 1,
                                                  0, 0, 2, 2,
                                                  0, 0, 3, 3};
        BOOST_CHECK(vertex_map == expected_vertex_map);

            auto iv = invalid_index;
            array_1d<index_t> expected_edge_map{iv, iv, 0, iv, iv, iv, iv,
                                                iv, iv, 0, iv, iv, 1, 1,
                                                iv, iv, 2, iv, iv, 3, 3,
                                                iv, 4, iv};

        BOOST_CHECK(edge_map == expected_edge_map);

    }

BOOST_AUTO_TEST_SUITE_END();